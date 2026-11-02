//! Immutable market data, held once per process.
//!
//! Each instrument owns its columns as `Arc<[T]>`. A reader keeps ids plus an
//! `Arc` clone of what it needs and indexes in place; nothing is copied per
//! strategy and no per-event schedule is materialised.

use std::sync::Arc;

use crate::core::error::RaptorError;
use crate::core::types::TickData;
use crate::core::Result;
use crate::data::book::DepthTick;
use crate::data::events::{quote_from_row, trade_from_row, QuoteTick, TradeTick};
use crate::data::feed::ensure_time_sorted;
use crate::portfolio::kernel::KernelBar;
use crate::runtime::ids::{InstrumentId, InstrumentTable};
use crate::runtime::memory::{arc_slice_bytes, HeapSize};

/// Rows are addressed by `u32` throughout the runtime.
fn check_rows(label: &str, len: usize) -> Result<()> {
    if u32::try_from(len).is_err() {
        return Err(RaptorError::invalid_parameter(format!(
            "{label}: {len} rows exceed the u32 row index"
        )));
    }
    Ok(())
}

fn check_len(label: &str, column: &str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(RaptorError::invalid_parameter(format!(
            "{label}: column `{column}` has {actual} rows, expected {expected}"
        )));
    }
    Ok(())
}

/// One instrument's bars, a column per field: 48 bytes per bar.
#[derive(Debug, Clone)]
pub struct BarColumns {
    pub ts: Arc<[i64]>,
    pub open: Arc<[f64]>,
    pub high: Arc<[f64]>,
    pub low: Arc<[f64]>,
    pub close: Arc<[f64]>,
    pub volume: Arc<[f64]>,
}

impl BarColumns {
    /// Take ownership of columns, copying each once into its `Arc`.
    ///
    /// Errors if the lengths differ or timestamps ever decrease.
    pub fn from_vecs(
        label: &str,
        ts: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    ) -> Result<Self> {
        Self::from_slices(label, &ts, &open, &high, &low, &close, &volume)
    }

    /// Copy columns from slices, once.
    pub fn from_slices(
        label: &str,
        ts: &[i64],
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<Self> {
        let n = ts.len();
        for (name, len) in [
            ("open", open.len()),
            ("high", high.len()),
            ("low", low.len()),
            ("close", close.len()),
            ("volume", volume.len()),
        ] {
            check_len(label, name, n, len)?;
        }
        check_rows(label, n)?;
        ensure_time_sorted(label, ts.iter().copied())?;
        Ok(Self {
            ts: ts.into(),
            open: open.into(),
            high: high.into(),
            low: low.into(),
            close: close.into(),
            volume: volume.into(),
        })
    }

    /// Transpose row-form bars into columns, once.
    pub fn from_kernel_bars(label: &str, bars: &[KernelBar]) -> Result<Self> {
        check_rows(label, bars.len())?;
        ensure_time_sorted(label, bars.iter().map(|b| b.timestamp))?;
        Ok(Self {
            ts: bars.iter().map(|b| b.timestamp).collect(),
            open: bars.iter().map(|b| b.open).collect(),
            high: bars.iter().map(|b| b.high).collect(),
            low: bars.iter().map(|b| b.low).collect(),
            close: bars.iter().map(|b| b.close).collect(),
            volume: bars.iter().map(|b| b.volume).collect(),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }

    /// The bar at `row`, assembled from the columns.
    #[inline]
    pub fn bar(&self, row: u32) -> Option<KernelBar> {
        let i = row as usize;
        Some(KernelBar {
            timestamp: *self.ts.get(i)?,
            open: self.open[i],
            high: self.high[i],
            low: self.low[i],
            close: self.close[i],
            volume: self.volume[i],
        })
    }
}

impl HeapSize for BarColumns {
    fn heap_bytes(&self) -> usize {
        arc_slice_bytes(&self.ts)
            + arc_slice_bytes(&self.open)
            + arc_slice_bytes(&self.high)
            + arc_slice_bytes(&self.low)
            + arc_slice_bytes(&self.close)
            + arc_slice_bytes(&self.volume)
    }
}

/// One instrument's raw ticks, holding the same fields as [`TickData`].
///
/// `ts`, `ltp`, `bid` and `ask` are always present. The remaining columns are
/// `None` when the feed never carried them, and read as `0.0` (the value
/// `TickData` uses for "unavailable"), without a zero-filled allocation.
#[derive(Debug, Clone)]
pub struct TickColumns {
    pub ts: Arc<[i64]>,
    pub ltp: Arc<[f64]>,
    pub bid: Arc<[f64]>,
    pub ask: Arc<[f64]>,
    pub buy_qty_delta: Option<Arc<[f64]>>,
    pub sell_qty_delta: Option<Arc<[f64]>>,
    pub oi: Option<Arc<[f64]>>,
    pub ltq: Option<Arc<[f64]>>,
    pub bid_qty: Option<Arc<[f64]>>,
    pub ask_qty: Option<Arc<[f64]>>,
}

/// Borrowed tick columns, for building [`TickColumns`] straight from the
/// caller's arrays. `None` means the feed never carried the column.
#[derive(Debug, Clone, Copy, Default)]
pub struct TickSlices<'a> {
    pub ts: &'a [i64],
    pub ltp: &'a [f64],
    pub bid: Option<&'a [f64]>,
    pub ask: Option<&'a [f64]>,
    pub buy_qty_delta: Option<&'a [f64]>,
    pub sell_qty_delta: Option<&'a [f64]>,
    pub oi: Option<&'a [f64]>,
    pub ltq: Option<&'a [f64]>,
    pub bid_qty: Option<&'a [f64]>,
    pub ask_qty: Option<&'a [f64]>,
}

impl TickColumns {
    /// Copy borrowed columns, once. A missing `bid` or `ask` becomes zeros
    /// (no quote); the other missing columns stay absent and read as `0.0`.
    /// Errors name `label` and the column when a length differs.
    pub fn from_slices(label: &str, s: &TickSlices<'_>) -> Result<Self> {
        let n = s.ts.len();
        check_len(label, "ltp", n, s.ltp.len())?;
        let optional = |name: &str, column: Option<&[f64]>| -> Result<Option<Arc<[f64]>>> {
            column
                .map(|c| {
                    check_len(label, name, n, c.len())?;
                    Ok(c.into())
                })
                .transpose()
        };
        let sided = |name: &str, column: Option<&[f64]>| -> Result<Arc<[f64]>> {
            Ok(optional(name, column)?.unwrap_or_else(|| vec![0.0; n].into()))
        };
        check_rows(label, n)?;
        ensure_time_sorted(label, s.ts.iter().copied())?;
        Ok(Self {
            ts: s.ts.into(),
            ltp: s.ltp.into(),
            bid: sided("bid", s.bid)?,
            ask: sided("ask", s.ask)?,
            buy_qty_delta: optional("buy_qty_delta", s.buy_qty_delta)?,
            sell_qty_delta: optional("sell_qty_delta", s.sell_qty_delta)?,
            oi: optional("oi", s.oi)?,
            ltq: optional("ltq", s.ltq)?,
            bid_qty: optional("bid_qty", s.bid_qty)?,
            ask_qty: optional("ask_qty", s.ask_qty)?,
        })
    }

    /// Copy a [`TickData`] into columns, once. An empty optional column
    /// becomes absent; any other length mismatch is an error.
    pub fn from_tick_data(label: &str, ticks: &TickData) -> Result<Self> {
        let n = ticks.timestamps.len();
        for (name, len) in
            [("ltp", ticks.ltp.len()), ("bid", ticks.bid.len()), ("ask", ticks.ask.len())]
        {
            check_len(label, name, n, len)?;
        }
        let optional = |name: &str, column: &[f64]| -> Result<Option<Arc<[f64]>>> {
            if column.is_empty() && n > 0 {
                return Ok(None);
            }
            check_len(label, name, n, column.len())?;
            Ok(Some(column.into()))
        };
        check_rows(label, n)?;
        ensure_time_sorted(label, ticks.timestamps.iter().copied())?;
        Ok(Self {
            ts: ticks.timestamps.as_slice().into(),
            ltp: ticks.ltp.as_slice().into(),
            bid: ticks.bid.as_slice().into(),
            ask: ticks.ask.as_slice().into(),
            buy_qty_delta: optional("buy_qty_delta", &ticks.buy_qty_delta)?,
            sell_qty_delta: optional("sell_qty_delta", &ticks.sell_qty_delta)?,
            oi: optional("oi", &ticks.oi)?,
            ltq: optional("ltq", &ticks.ltq)?,
            bid_qty: optional("bid_qty", &ticks.bid_qty)?,
            ask_qty: optional("ask_qty", &ticks.ask_qty)?,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }

    /// Whether `row` carries a trade print (a positive last price).
    #[inline]
    pub fn has_trade(&self, row: u32) -> bool {
        self.ltp[row as usize] > 0.0
    }

    /// Whether `row` carries a two-sided quote.
    #[inline]
    pub fn has_quote(&self, row: u32) -> bool {
        self.bid[row as usize] > 0.0 && self.ask[row as usize] > 0.0
    }

    /// The trade print of `row`, by the same rules as `tick_data_to_events`.
    #[inline]
    pub fn trade(&self, row: u32) -> Option<TradeTick> {
        let i = row as usize;
        trade_from_row(
            *self.ts.get(i)?,
            self.ltp[i],
            opt(&self.buy_qty_delta, i),
            opt(&self.sell_qty_delta, i),
            opt(&self.ltq, i),
            opt(&self.oi, i),
        )
    }

    /// The quote of `row`, by the same rules as `tick_data_to_events`.
    #[inline]
    pub fn quote(&self, row: u32) -> Option<QuoteTick> {
        let i = row as usize;
        quote_from_row(
            *self.ts.get(i)?,
            self.bid[i],
            self.ask[i],
            opt(&self.bid_qty, i),
            opt(&self.ask_qty, i),
        )
    }
}

/// An optional column's value at `row`, `0.0` when the column is absent.
#[inline]
fn opt(column: &Option<Arc<[f64]>>, row: usize) -> f64 {
    column.as_ref().map_or(0.0, |c| c[row])
}

impl HeapSize for TickColumns {
    fn heap_bytes(&self) -> usize {
        let optional = |c: &Option<Arc<[f64]>>| c.as_ref().map_or(0, arc_slice_bytes);
        arc_slice_bytes(&self.ts)
            + arc_slice_bytes(&self.ltp)
            + arc_slice_bytes(&self.bid)
            + arc_slice_bytes(&self.ask)
            + optional(&self.buy_qty_delta)
            + optional(&self.sell_qty_delta)
            + optional(&self.oi)
            + optional(&self.ltq)
            + optional(&self.bid_qty)
            + optional(&self.ask_qty)
    }
}

/// One instrument's depth snapshots.
#[derive(Debug, Clone)]
pub struct DepthColumns {
    pub ticks: Arc<[DepthTick]>,
}

impl DepthColumns {
    /// Copy snapshots, once. Errors if timestamps ever decrease.
    pub fn from_slice(label: &str, depth: &[DepthTick]) -> Result<Self> {
        check_rows(label, depth.len())?;
        ensure_time_sorted(label, depth.iter().map(|d| d.timestamp))?;
        Ok(Self { ticks: depth.into() })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ticks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ticks.is_empty()
    }
}

impl HeapSize for DepthColumns {
    fn heap_bytes(&self) -> usize {
        arc_slice_bytes(&self.ticks)
    }
}

/// Every instrument's columns, indexed by [`InstrumentId`].
///
/// Built single-threaded at registration time, then shared (`Send + Sync`)
/// by reference or `Arc`; readers never mutate it.
#[derive(Debug, Default, Clone)]
pub struct MarketStore {
    symbols: InstrumentTable,
    bars: Vec<Option<BarColumns>>,
    ticks: Vec<Option<TickColumns>>,
    depth: Vec<Option<DepthColumns>>,
}

impl MarketStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an instrument (idempotent per symbol).
    pub fn add_instrument(&mut self, symbol: &str) -> Result<InstrumentId> {
        let id = self.symbols.intern(symbol)?;
        if id.index() == self.bars.len() {
            self.bars.push(None);
            self.ticks.push(None);
            self.depth.push(None);
        }
        Ok(id)
    }

    fn check(&self, id: InstrumentId) -> Result<()> {
        if id.index() >= self.bars.len() {
            return Err(RaptorError::IndexOutOfBounds {
                index: id.index(),
                length: self.bars.len(),
            });
        }
        Ok(())
    }

    /// Attach (replace) an instrument's bars.
    pub fn set_bars(&mut self, id: InstrumentId, bars: BarColumns) -> Result<()> {
        self.check(id)?;
        self.bars[id.index()] = Some(bars);
        Ok(())
    }

    /// Attach (replace) an instrument's ticks. An attached but empty tick
    /// series still counts as present.
    pub fn set_ticks(&mut self, id: InstrumentId, ticks: TickColumns) -> Result<()> {
        self.check(id)?;
        self.ticks[id.index()] = Some(ticks);
        Ok(())
    }

    /// Attach (replace) an instrument's depth snapshots.
    pub fn set_depth(&mut self, id: InstrumentId, depth: DepthColumns) -> Result<()> {
        self.check(id)?;
        self.depth[id.index()] = Some(depth);
        Ok(())
    }

    pub fn symbols(&self) -> &InstrumentTable {
        &self.symbols
    }

    /// Number of registered instruments.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    pub fn bars(&self, id: InstrumentId) -> Option<&BarColumns> {
        self.bars.get(id.index())?.as_ref()
    }

    pub fn ticks(&self, id: InstrumentId) -> Option<&TickColumns> {
        self.ticks.get(id.index())?.as_ref()
    }

    pub fn depth(&self, id: InstrumentId) -> Option<&DepthColumns> {
        self.depth.get(id.index())?.as_ref()
    }

    /// The bar at `row` of an instrument.
    #[inline]
    pub fn bar(&self, id: InstrumentId, row: u32) -> Option<KernelBar> {
        self.bars(id)?.bar(row)
    }

    /// The trade print at tick `row` of an instrument.
    #[inline]
    pub fn trade(&self, id: InstrumentId, row: u32) -> Option<TradeTick> {
        self.ticks(id)?.trade(row)
    }

    /// The quote at tick `row` of an instrument.
    #[inline]
    pub fn quote(&self, id: InstrumentId, row: u32) -> Option<QuoteTick> {
        self.ticks(id)?.quote(row)
    }

    /// The depth snapshot at `row` of an instrument.
    #[inline]
    pub fn depth_at(&self, id: InstrumentId, row: u32) -> Option<DepthTick> {
        self.depth(id)?.ticks.get(row as usize).copied()
    }
}

impl HeapSize for MarketStore {
    fn heap_bytes(&self) -> usize {
        fn table<T: HeapSize>(v: &Vec<Option<T>>) -> usize {
            v.capacity() * std::mem::size_of::<Option<T>>()
                + v.iter().flatten().map(HeapSize::heap_bytes).sum::<usize>()
        }
        self.symbols.heap_bytes() + table(&self.bars) + table(&self.ticks) + table(&self.depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(n: usize) -> Vec<KernelBar> {
        (0..n)
            .map(|i| KernelBar {
                timestamp: i as i64,
                open: 1.0,
                high: 2.0,
                low: 0.5,
                close: 1.5,
                volume: 10.0,
            })
            .collect()
    }

    #[test]
    fn store_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MarketStore>();
        assert_send_sync::<BarColumns>();
    }

    #[test]
    fn bar_columns_round_trip_and_cost_48_bytes_a_bar() {
        let src = bars(100);
        let cols = BarColumns::from_kernel_bars("AAA", &src).unwrap();
        assert_eq!(cols.len(), 100);
        let b = cols.bar(7).unwrap();
        assert_eq!((b.timestamp, b.close), (7, 1.5));
        assert!(cols.bar(100).is_none());
        // 6 columns * 100 rows * 8 B, plus a 16 B header per column.
        assert_eq!(cols.heap_bytes(), 48 * 100 + 6 * 16);
    }

    #[test]
    fn unsorted_or_ragged_input_is_rejected() {
        let mut src = bars(3);
        src[2].timestamp = -1;
        assert!(BarColumns::from_kernel_bars("AAA", &src).is_err());
        let ragged = BarColumns::from_vecs(
            "AAA",
            vec![1, 2],
            vec![1.0],
            vec![1.0; 2],
            vec![1.0; 2],
            vec![1.0; 2],
            vec![1.0; 2],
        );
        assert!(ragged.is_err());
    }

    #[test]
    fn absent_tick_columns_cost_nothing() {
        let ticks = TickData {
            timestamps: vec![1, 2],
            ltp: vec![1.0, 2.0],
            bid: vec![0.0; 2],
            ask: vec![0.0; 2],
            buy_qty_delta: vec![],
            sell_qty_delta: vec![],
            oi: vec![],
            ltq: vec![],
            bid_qty: vec![],
            ask_qty: vec![],
        };
        let cols = TickColumns::from_tick_data("AAA", &ticks).unwrap();
        assert!(cols.ltq.is_none() && cols.oi.is_none());
        assert_eq!(cols.heap_bytes(), 4 * (2 * 8 + 16));
        let mut bad = ticks;
        bad.oi = vec![0.0];
        assert!(TickColumns::from_tick_data("AAA", &bad).is_err());
    }

    #[test]
    fn tick_slices_copy_once_and_name_the_bad_column() {
        let ts = [1i64, 2, 3];
        let ltp = [1.0, 2.0, 3.0];
        let cols = TickColumns::from_slices(
            "ticks['AAA']",
            &TickSlices { ts: &ts, ltp: &ltp, ltq: Some(&[5.0; 3]), ..Default::default() },
        )
        .unwrap();
        assert_eq!(cols.trade(1).unwrap().size, 5.0);
        assert!(!cols.has_quote(0) && cols.oi.is_none());
        let err = TickColumns::from_slices(
            "ticks['AAA']",
            &TickSlices { ts: &ts, ltp: &ltp, oi: Some(&[1.0]), ..Default::default() },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("ticks['AAA']") && err.contains("oi"), "{err}");
        let unsorted = TickSlices { ts: &[2, 1], ltp: &[1.0, 1.0], ..Default::default() };
        let err = TickColumns::from_slices("ticks['B']", &unsorted).unwrap_err().to_string();
        assert!(err.contains("ticks['B']") && err.contains("must not decrease"), "{err}");
    }

    #[test]
    fn store_registers_symbols_and_accounts_heap() {
        let mut store = MarketStore::new();
        let a = store.add_instrument("AAA").unwrap();
        assert_eq!(store.add_instrument("AAA").unwrap(), a);
        let b = store.add_instrument("BBB").unwrap();
        let cols = BarColumns::from_kernel_bars("AAA", &bars(10)).unwrap();
        let col_bytes = cols.heap_bytes();
        store.set_bars(a, cols).unwrap();
        assert!(store.bars(a).is_some() && store.bars(b).is_none());
        assert!(store.heap_bytes() >= col_bytes);
        assert!(store
            .set_bars(InstrumentId::new(9), BarColumns::from_kernel_bars("x", &[]).unwrap())
            .is_err());
    }
}
