//! One ATR series per (instrument, period), shared by every strategy.
//!
//! The Python runner computes the series per strategy with
//! `raptorbt.atr(high, low, close, period)` (see `atr_series` in
//! `strategy/_commands.py`): that is [`volatility::atr`] on the bar columns,
//! and an unusable result degrades to "no series". The cache calls the same
//! function on the same columns and applies the same degradation, so a
//! strategy reading from it sees bit-identical numbers.
//!
//! Resolving the period from a strategy's stop and target configuration stays
//! with the caller (`resolve_atr_period`); the cache takes the resolved value.

use std::collections::HashMap;
use std::sync::Arc;

use crate::indicators::volatility;
use crate::runtime::ids::InstrumentId;
use crate::runtime::memory::{arc_slice_bytes, HeapSize};
use crate::runtime::store::BarColumns;

/// Lookups answered from the cache and lookups that computed a series.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AtrStats {
    pub hits: u64,
    pub computed: u64,
}

/// ATR series keyed by `(instrument, period)`. Not thread-safe; fill it while
/// strategies are being set up.
#[derive(Debug, Default)]
pub struct AtrCache {
    series: HashMap<(InstrumentId, usize), Option<Arc<[f64]>>>,
    stats: AtrStats,
}

impl AtrCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The ATR series of `bars`, which must be `instrument`'s own columns.
    ///
    /// `None` when `period` is 0 (no ATR-based level is configured) or when
    /// the batch function rejects the columns; the outcome is cached either
    /// way. A period at least as long as the data gives an all-NaN series,
    /// exactly as the batch function does.
    pub fn get(
        &mut self,
        instrument: InstrumentId,
        period: usize,
        bars: &BarColumns,
    ) -> Option<Arc<[f64]>> {
        if period == 0 {
            return None;
        }
        if let Some(entry) = self.series.get(&(instrument, period)) {
            self.stats.hits += 1;
            return entry.clone();
        }
        self.stats.computed += 1;
        let series = volatility::atr(&bars.high, &bars.low, &bars.close, period)
            .ok()
            .map(Arc::<[f64]>::from);
        self.series.insert((instrument, period), series.clone());
        series
    }

    /// Distinct `(instrument, period)` entries, including cached failures.
    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    pub fn stats(&self) -> AtrStats {
        self.stats
    }
}

impl HeapSize for AtrCache {
    /// Each series once, however many strategies hold it; the map's table
    /// slack is not included.
    fn heap_bytes(&self) -> usize {
        self.series.capacity() * std::mem::size_of::<((InstrumentId, usize), Option<Arc<[f64]>>)>()
            + self.series.values().flatten().map(arc_slice_bytes).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    fn bars(rows: &[(f64, f64, f64)]) -> BarColumns {
        let n = rows.len();
        let col = |f: fn(&(f64, f64, f64)) -> f64| rows.iter().map(f).collect::<Vec<f64>>();
        BarColumns::from_vecs(
            "t",
            (0..n as i64).collect(),
            col(|r| r.2),
            col(|r| r.0),
            col(|r| r.1),
            col(|r| r.2),
            vec![1.0; n],
        )
        .unwrap()
    }

    #[test]
    fn matches_the_batch_function_bit_for_bit_and_is_shared() {
        let mut runner = TestRunner::new(Config {
            cases: 300,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(0xA7C0_0001),
            ..Config::default()
        });
        // (low, range, close position in the range) per bar.
        let row = (1.0f64..500.0, 0.0f64..20.0, 0.0f64..1.0);
        let strategy = (prop::collection::vec(row, 0..80), 0usize..100, 0u32..4);
        runner
            .run(&strategy, |(rows, period, id)| {
                let rows: Vec<(f64, f64, f64)> =
                    rows.iter().map(|&(lo, r, f)| (lo + r, lo, lo + r * f)).collect();
                let data = bars(&rows);
                let inst = InstrumentId::new(id);
                let mut cache = AtrCache::new();
                let got = cache.get(inst, period, &data);
                if period == 0 {
                    prop_assert!(got.is_none());
                    prop_assert!(cache.is_empty());
                    return Ok(());
                }
                let want = volatility::atr(&data.high, &data.low, &data.close, period).unwrap();
                let got = got.unwrap();
                prop_assert_eq!(got.len(), want.len());
                for (g, w) in got.iter().zip(&want) {
                    prop_assert_eq!(g.to_bits(), w.to_bits());
                }
                let again = cache.get(inst, period, &data).unwrap();
                prop_assert!(Arc::ptr_eq(&got, &again));
                prop_assert_eq!(cache.stats(), AtrStats { hits: 1, computed: 1 });
                prop_assert_eq!(cache.len(), 1);
                // Another instrument is another series.
                let other = cache.get(InstrumentId::new(id + 1), period, &data).unwrap();
                prop_assert!(!Arc::ptr_eq(&got, &other));
                prop_assert_eq!(cache.len(), 2);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn heap_bytes_counts_each_series_once() {
        let rows: Vec<(f64, f64, f64)> =
            (0..50).map(|i| (101.0 + i as f64, 99.0 + i as f64, 100.0 + i as f64)).collect();
        let data = bars(&rows);
        let mut cache = AtrCache::new();
        let before = cache.heap_bytes();
        let a = cache.get(InstrumentId::new(0), 14, &data).unwrap();
        let _again = cache.get(InstrumentId::new(0), 14, &data).unwrap();
        let after = cache.heap_bytes();
        let entry = std::mem::size_of::<((InstrumentId, usize), Option<Arc<[f64]>>)>();
        // One series payload plus its header, and at least one map entry.
        assert_eq!(arc_slice_bytes(&a), 16 + 50 * 8);
        assert!(after - before >= arc_slice_bytes(&a) + entry);
    }
}
