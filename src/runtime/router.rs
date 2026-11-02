//! Subscription routing: which strategies listen to which instruments and
//! aggregations.
//!
//! Both indexes use a compressed-sparse-row layout: one flat `members` vector
//! and an `offsets` vector, so a lookup is two loads and a slice with no
//! per-key allocation.

use crate::core::error::RaptorError;
use crate::core::Result;
use crate::runtime::ids::{AggId, InstrumentId, StrategyId};
use crate::runtime::memory::HeapSize;

/// Build `offsets` (len `keys + 1`) from the key of each row.
fn offsets_for(keys: usize, row_keys: impl Iterator<Item = usize>) -> Result<Vec<u32>> {
    let mut offsets = vec![0u32; keys + 1];
    for k in row_keys {
        offsets[k + 1] += 1;
    }
    for i in 0..keys {
        offsets[i + 1] = offsets[i]
            .checked_add(offsets[i + 1])
            .ok_or_else(|| RaptorError::invalid_parameter("subscription index overflows u32"))?;
    }
    Ok(offsets)
}

/// Instrument -> subscribed strategies.
///
/// Each instrument's slice is sorted ascending by [`StrategyId`] (registration
/// order) and holds a strategy at most once.
#[derive(Debug, Clone, Default)]
pub struct SubscriptionIndex {
    offsets: Vec<u32>,
    members: Vec<StrategyId>,
}

impl SubscriptionIndex {
    /// Strategies subscribed to `instrument`; empty for an unknown id.
    #[inline]
    pub fn subscribers(&self, instrument: InstrumentId) -> &[StrategyId] {
        let i = instrument.index();
        match (self.offsets.get(i), self.offsets.get(i + 1)) {
            (Some(&lo), Some(&hi)) => &self.members[lo as usize..hi as usize],
            _ => &[],
        }
    }

    /// Instruments the index was built for.
    pub fn instruments(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total (instrument, strategy) pairs after dedup.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

impl HeapSize for SubscriptionIndex {
    fn heap_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.members.capacity() * std::mem::size_of::<StrategyId>()
    }
}

/// Collects registrations, then builds a [`SubscriptionIndex`].
#[derive(Debug)]
pub struct SubscriptionIndexBuilder {
    instruments: usize,
    /// `(instrument, strategy)` pairs in registration order.
    pairs: Vec<(InstrumentId, StrategyId)>,
}

impl SubscriptionIndexBuilder {
    pub fn new(instruments: usize) -> Self {
        Self { instruments, pairs: Vec::new() }
    }

    /// Record that `strategy` listens to `instruments`. Repeats are fine.
    pub fn register(&mut self, strategy: StrategyId, instruments: &[InstrumentId]) -> Result<()> {
        if let Some(bad) = instruments.iter().find(|i| i.index() >= self.instruments) {
            return Err(RaptorError::invalid_parameter(format!(
                "instrument id {} is outside the {} registered instruments",
                bad.raw(),
                self.instruments
            )));
        }
        self.pairs.extend(instruments.iter().map(|&i| (i, strategy)));
        Ok(())
    }

    pub fn build(mut self) -> Result<SubscriptionIndex> {
        self.pairs.sort_unstable();
        self.pairs.dedup();
        let offsets = offsets_for(self.instruments, self.pairs.iter().map(|(i, _)| i.index()))?;
        let members = self.pairs.iter().map(|&(_, s)| s).collect();
        Ok(SubscriptionIndex { offsets, members })
    }
}

/// Aggregation -> `(strategy, local stream)` deliveries.
///
/// A strategy's *local stream* is the position of the subscription in that
/// strategy's own list. Each slice is ordered by strategy, then by that
/// strategy's own subscription order, so a strategy subscribed twice to one
/// aggregation appears twice, in order.
#[derive(Debug, Clone, Default)]
pub struct AggSubscriptions {
    offsets: Vec<u32>,
    members: Vec<(StrategyId, u16)>,
}

impl AggSubscriptions {
    #[inline]
    pub fn subscribers(&self, agg: AggId) -> &[(StrategyId, u16)] {
        let i = agg.index();
        match (self.offsets.get(i), self.offsets.get(i + 1)) {
            (Some(&lo), Some(&hi)) => &self.members[lo as usize..hi as usize],
            _ => &[],
        }
    }

    /// Total subscriptions.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

impl HeapSize for AggSubscriptions {
    fn heap_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.members.capacity() * std::mem::size_of::<(StrategyId, u16)>()
    }
}

/// Collects `(agg, strategy, local stream)` rows, then builds
/// [`AggSubscriptions`].
#[derive(Debug)]
pub struct AggSubscriptionsBuilder {
    aggs: usize,
    rows: Vec<(AggId, StrategyId, u16)>,
}

impl AggSubscriptionsBuilder {
    pub fn new(aggs: usize) -> Self {
        Self { aggs, rows: Vec::new() }
    }

    pub fn push(&mut self, agg: AggId, strategy: StrategyId, local_stream: u16) -> Result<()> {
        if agg.index() >= self.aggs {
            return Err(RaptorError::invalid_parameter(format!(
                "aggregation id {} is outside the {} registered aggregations",
                agg.raw(),
                self.aggs
            )));
        }
        self.rows.push((agg, strategy, local_stream));
        Ok(())
    }

    pub fn build(mut self) -> Result<AggSubscriptions> {
        self.rows.sort_unstable();
        let offsets = offsets_for(self.aggs, self.rows.iter().map(|(a, _, _)| a.index()))?;
        let members = self.rows.iter().map(|&(_, s, l)| (s, l)).collect();
        Ok(AggSubscriptions { offsets, members })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    fn inst(i: u32) -> InstrumentId {
        InstrumentId::new(i)
    }

    fn strat(i: u32) -> StrategyId {
        StrategyId::new(i)
    }

    #[test]
    fn subscribers_are_sorted_and_deduplicated() {
        let mut b = SubscriptionIndexBuilder::new(4);
        b.register(strat(2), &[inst(1), inst(1), inst(3)]).unwrap();
        b.register(strat(0), &[inst(3), inst(1)]).unwrap();
        b.register(strat(1), &[inst(3)]).unwrap();
        b.register(strat(2), &[inst(1)]).unwrap();
        let idx = b.build().unwrap();
        assert_eq!(idx.subscribers(inst(1)), &[strat(0), strat(2)]);
        assert_eq!(idx.subscribers(inst(3)), &[strat(0), strat(1), strat(2)]);
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn empty_and_unknown_instruments_have_no_subscribers() {
        let mut b = SubscriptionIndexBuilder::new(3);
        b.register(strat(0), &[inst(1)]).unwrap();
        let idx = b.build().unwrap();
        assert!(idx.subscribers(inst(0)).is_empty());
        assert!(idx.subscribers(inst(2)).is_empty());
        assert!(idx.subscribers(inst(99)).is_empty());
        assert_eq!(idx.instruments(), 3);

        let none = SubscriptionIndexBuilder::new(0).build().unwrap();
        assert!(none.is_empty());
        assert!(none.subscribers(inst(0)).is_empty());
    }

    #[test]
    fn out_of_range_ids_are_refused() {
        let mut b = SubscriptionIndexBuilder::new(2);
        assert!(b.register(strat(0), &[inst(2)]).is_err());
        assert!(AggSubscriptionsBuilder::new(1).push(AggId::new(1), strat(0), 0).is_err());
    }

    #[test]
    fn agg_subscriptions_keep_each_strategys_own_order() {
        let mut b = AggSubscriptionsBuilder::new(2);
        // Strategy 1 holds agg 0 as local streams 2 and 0; strategy 0 as 1.
        b.push(AggId::new(0), strat(1), 2).unwrap();
        b.push(AggId::new(0), strat(0), 1).unwrap();
        b.push(AggId::new(0), strat(1), 0).unwrap();
        b.push(AggId::new(1), strat(0), 0).unwrap();
        let subs = b.build().unwrap();
        assert_eq!(subs.subscribers(AggId::new(0)), &[(strat(0), 1), (strat(1), 0), (strat(1), 2)]);
        assert_eq!(subs.subscribers(AggId::new(1)), &[(strat(0), 0)]);
        assert!(subs.subscribers(AggId::new(5)).is_empty());
    }

    #[test]
    fn heap_bytes_counts_offsets_and_members() {
        let mut b = SubscriptionIndexBuilder::new(2);
        b.register(strat(0), &[inst(0), inst(1)]).unwrap();
        let idx = b.build().unwrap();
        assert!(idx.heap_bytes() >= 3 * 4 + 2 * 4);
    }

    #[test]
    fn index_matches_a_naive_model_on_large_random_cases() {
        let mut runner = TestRunner::new(Config {
            cases: 64,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(0x5EED_0A11),
            ..Config::default()
        });
        let instruments = 300usize;
        let strategy =
            prop::collection::vec(prop::collection::vec(0..instruments as u32, 0..40), 0..400);
        runner
            .run(&strategy, |regs| {
                let mut model: Vec<Vec<StrategyId>> = vec![Vec::new(); instruments];
                let mut b = SubscriptionIndexBuilder::new(instruments);
                let mut agg_rows = Vec::new();
                let mut agg_b = AggSubscriptionsBuilder::new(instruments);
                for (s, list) in regs.iter().enumerate() {
                    let ids: Vec<InstrumentId> = list.iter().map(|&i| inst(i)).collect();
                    b.register(strat(s as u32), &ids).unwrap();
                    for &i in list {
                        if !model[i as usize].contains(&strat(s as u32)) {
                            model[i as usize].push(strat(s as u32));
                        }
                    }
                    for (local, &i) in list.iter().enumerate() {
                        agg_b.push(AggId::new(i), strat(s as u32), local as u16).unwrap();
                        agg_rows.push((i, s as u32, local as u16));
                    }
                }
                let idx = b.build().unwrap();
                for (i, want) in model.iter().enumerate() {
                    prop_assert_eq!(idx.subscribers(inst(i as u32)), want.as_slice());
                }
                let subs = agg_b.build().unwrap();
                prop_assert_eq!(subs.len(), agg_rows.len());
                for i in 0..instruments as u32 {
                    let want: Vec<(StrategyId, u16)> =
                        agg_rows.iter().filter(|r| r.0 == i).map(|r| (strat(r.1), r.2)).collect();
                    prop_assert_eq!(subs.subscribers(AggId::new(i)), want.as_slice());
                }
                Ok(())
            })
            .unwrap();
    }
}
