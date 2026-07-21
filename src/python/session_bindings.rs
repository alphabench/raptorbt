//! Python bindings for the multi-instrument event session.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::types::{BacktestConfig, Direction, InstrumentConfig, TickData};
use crate::portfolio::kernel::{KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::data::{BookLevel, BookSide, DepthTick};
use crate::portfolio::session::{EventSession, ScheduleData};

use super::bindings::{
    convert_result, PyBacktestConfig, PyInstrumentConfig, PyInstrumentSummary, PyPortfolioResult,
};
use super::instrument_bindings::PyInstrumentSpec;
use super::numpy_bridge::{numpy_to_vec_f64, numpy_to_vec_i64};
use super::strategy_bindings::{
    parse_account_mode, parse_qty_spec, submit_order_on, PyEngineEvent, PyPositionSnapshot,
};

/// Visible levels of one side of a depth snapshot.
fn book_levels(book: &DepthTick, side: BookSide) -> &[BookLevel] {
    match side {
        BookSide::Bid => &book.bids[..book.bid_len as usize],
        BookSide::Ask => &book.asks[..book.ask_len as usize],
    }
}

/// Multi-instrument session over deterministically merged bar streams.
///
/// One shared account funds every instrument. With `account_type="margin"`
/// they also share one pool of locked initial margin, so leverage applies
/// portfolio-wide and a margin call halts all instruments at once.
///
/// Protocol: `add_instrument` + `set_bars` per symbol, `seal()`, then loop
/// `current()` / `apply_current(...)` to completion and `finish()`.
#[pyclass]
pub struct PyPortfolioSession {
    session: Option<EventSession>,
}

impl PyPortfolioSession {
    fn session_ref(&self) -> PyResult<&EventSession> {
        self.session
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }

    fn session_mut(&mut self) -> PyResult<&mut EventSession> {
        self.session
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("session is finished; create a new one"))
    }
}

#[pymethods]
impl PyPortfolioSession {
    #[new]
    #[pyo3(signature = (config=None, account_type="cash", leverage=1.0))]
    fn new(
        config: Option<&PyBacktestConfig>,
        account_type: &str,
        leverage: f64,
    ) -> PyResult<Self> {
        let rust_config: BacktestConfig = config.map(BacktestConfig::from).unwrap_or_default();
        let account = parse_account_mode(account_type, leverage)?;
        Ok(Self { session: Some(EventSession::with_account(rust_config, account)) })
    }

    /// Register an instrument; returns its index for routing.
    #[pyo3(signature = (symbol, direction=1, instrument_config=None, instrument=None, oms_type="netting"))]
    fn add_instrument(
        &mut self,
        symbol: &str,
        direction: i32,
        instrument_config: Option<&PyInstrumentConfig>,
        instrument: Option<&PyInstrumentSpec>,
        oms_type: &str,
    ) -> PyResult<usize> {
        let direction = Direction::from_int(direction)
            .ok_or_else(|| PyValueError::new_err("direction must be 1 or -1"))?;
        let policy = match oms_type {
            "netting" => PositionPolicy::Net,
            "hedging" => PositionPolicy::Independent,
            other => {
                return Err(PyValueError::new_err(format!(
                    "oms_type must be 'netting' or 'hedging', got {other:?}"
                )))
            }
        };
        let spec = match instrument {
            Some(py_spec) => {
                let spec = crate::instruments::InstrumentSpec::from(py_spec);
                if !spec.kind.tradable() {
                    return Err(PyValueError::new_err(format!(
                        "instrument {:?} is not tradable",
                        spec.symbol
                    )));
                }
                Some(spec)
            }
            None => None,
        };
        let inst: Option<InstrumentConfig> = instrument_config.map(InstrumentConfig::from);
        Ok(self.session_mut()?.add_instrument(
            symbol.to_string(),
            direction,
            spec,
            inst.as_ref(),
            policy,
        ))
    }

    /// Attach an instrument's bar arrays (ascending timestamps).
    #[allow(clippy::too_many_arguments)]
    fn set_bars(
        &mut self,
        instrument: usize,
        timestamps: PyReadonlyArray1<i64>,
        open: PyReadonlyArray1<f64>,
        high: PyReadonlyArray1<f64>,
        low: PyReadonlyArray1<f64>,
        close: PyReadonlyArray1<f64>,
        volume: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let ts = numpy_to_vec_i64(timestamps);
        let o = numpy_to_vec_f64(open);
        let h = numpy_to_vec_f64(high);
        let l = numpy_to_vec_f64(low);
        let c = numpy_to_vec_f64(close);
        let v = numpy_to_vec_f64(volume);
        let n = ts.len();
        if [o.len(), h.len(), l.len(), c.len(), v.len()].iter().any(|&len| len != n) {
            return Err(PyValueError::new_err("all bar arrays must share one length"));
        }
        let bars: Vec<KernelBar> = (0..n)
            .map(|i| KernelBar {
                timestamp: ts[i],
                open: o[i],
                high: h[i],
                low: l[i],
                close: c[i],
                volume: v[i],
            })
            .collect();
        self.session_mut()?.set_bars(instrument, bars);
        Ok(())
    }

    /// Attach an instrument's tick arrays (ascending timestamps).
    ///
    /// A row with `ltp > 0` yields a trade print; a row with both `bid > 0`
    /// and `ask > 0` yields a quote. The trade precedes the quote of the
    /// same row, since the print is what that book state followed.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (instrument, timestamps, ltp, bid=None, ask=None, buy_qty_delta=None, sell_qty_delta=None))]
    fn set_ticks(
        &mut self,
        instrument: usize,
        timestamps: PyReadonlyArray1<i64>,
        ltp: PyReadonlyArray1<f64>,
        bid: Option<PyReadonlyArray1<f64>>,
        ask: Option<PyReadonlyArray1<f64>>,
        buy_qty_delta: Option<PyReadonlyArray1<f64>>,
        sell_qty_delta: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let ts = numpy_to_vec_i64(timestamps);
        let ltp = numpy_to_vec_f64(ltp);
        let n = ts.len();
        let optional = |arr: Option<PyReadonlyArray1<f64>>| -> PyResult<Vec<f64>> {
            match arr {
                Some(a) => {
                    let v = numpy_to_vec_f64(a);
                    if v.len() != n {
                        return Err(PyValueError::new_err(
                            "all tick arrays must share one length",
                        ));
                    }
                    Ok(v)
                }
                None => Ok(vec![0.0; n]),
            }
        };
        if ltp.len() != n {
            return Err(PyValueError::new_err("all tick arrays must share one length"));
        }
        let ticks = TickData {
            timestamps: ts,
            ltp,
            bid: optional(bid)?,
            ask: optional(ask)?,
            buy_qty_delta: optional(buy_qty_delta)?,
            sell_qty_delta: optional(sell_qty_delta)?,
            oi: vec![0.0; n],
        };
        self.session_mut()?.set_ticks(instrument, ticks);
        Ok(())
    }

    /// Attach an instrument's depth snapshots.
    ///
    /// Price/size arrays are `(n_snapshots, levels)`, best level first:
    /// bids descending, asks ascending. Levels beyond the book's capacity
    /// are truncated.
    #[allow(clippy::too_many_arguments)]
    fn set_depth(
        &mut self,
        instrument: usize,
        timestamps: PyReadonlyArray1<i64>,
        bid_prices: PyReadonlyArray2<f64>,
        bid_sizes: PyReadonlyArray2<f64>,
        ask_prices: PyReadonlyArray2<f64>,
        ask_sizes: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let ts = numpy_to_vec_i64(timestamps);
        let n = ts.len();
        let bp = bid_prices.as_array();
        let bs = bid_sizes.as_array();
        let ap = ask_prices.as_array();
        let asz = ask_sizes.as_array();
        for arr in [&bp, &bs, &ap, &asz] {
            if arr.shape()[0] != n {
                return Err(PyValueError::new_err(
                    "depth arrays must have one row per timestamp",
                ));
            }
        }
        if bp.shape()[1] != bs.shape()[1] || ap.shape()[1] != asz.shape()[1] {
            return Err(PyValueError::new_err(
                "price and size arrays must have the same level count",
            ));
        }
        let snapshots: Vec<DepthTick> = (0..n)
            .map(|row| {
                let bids: Vec<BookLevel> = (0..bp.shape()[1])
                    .map(|l| BookLevel { price: bp[[row, l]], size: bs[[row, l]] })
                    .filter(|level| level.price > 0.0)
                    .collect();
                let asks: Vec<BookLevel> = (0..ap.shape()[1])
                    .map(|l| BookLevel { price: ap[[row, l]], size: asz[[row, l]] })
                    .filter(|level| level.price > 0.0)
                    .collect();
                DepthTick::from_levels(ts[row], &bids, &asks)
            })
            .collect();
        self.session_mut()?.set_depth(instrument, snapshots);
        Ok(())
    }

    /// Full levels of the pending depth event, or `None` for other kinds.
    ///
    /// Returns `(bids, asks)` as `(price, size)` lists, best first.
    #[allow(clippy::type_complexity)]
    fn current_depth(&self) -> PyResult<Option<(Vec<(f64, f64)>, Vec<(f64, f64)>)>> {
        let session = self.session_ref()?;
        let Some(entry) = session.current() else { return Ok(None) };
        let ScheduleData::Depth(handle) = entry.data else { return Ok(None) };
        Ok(session.depth_at(handle.slot).map(|book| {
            let levels = |side| {
                book_levels(&book, side)
                    .iter()
                    .map(|l| (l.price, l.size))
                    .collect::<Vec<_>>()
            };
            (levels(BookSide::Bid), levels(BookSide::Ask))
        }))
    }

    /// Merge all streams into the deterministic schedule.
    fn seal(&mut self) -> PyResult<()> {
        self.session_mut()?.seal();
        Ok(())
    }

    /// Number of scheduled events.
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.session_ref()?.len())
    }

    /// The pending event: `(instrument, local_idx, ts, o, h, l, c, v)`.
    ///
    /// Bar sessions only. A session carrying ticks must use
    /// [`PyPortfolioSession::current_event`], which names the payload kind;
    /// this returns `None` on a non-bar event rather than mispresenting a
    /// print as a bar.
    #[allow(clippy::type_complexity)]
    fn current(&self) -> PyResult<Option<(usize, usize, i64, f64, f64, f64, f64, f64)>> {
        Ok(self.session_ref()?.current().and_then(|e| match e.data {
            ScheduleData::Bar(bar) => Some((
                e.instrument,
                e.local_idx,
                bar.timestamp,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            )),
            _ => None,
        }))
    }

    /// The pending event, tagged by kind:
    /// `(kind, instrument, local_idx, ts, a, b, c, d, e)` where `kind` is
    /// `"bar"` (o/h/l/c/v), `"trade"` (price, size, 0, 0, 0) or `"quote"`
    /// (bid, ask, 0, 0, 0).
    #[allow(clippy::type_complexity)]
    fn current_event(
        &self,
    ) -> PyResult<Option<(String, usize, usize, i64, f64, f64, f64, f64, f64)>> {
        Ok(self.session_ref()?.current().map(|e| match e.data {
            ScheduleData::Bar(bar) => (
                "bar".to_string(),
                e.instrument,
                e.local_idx,
                bar.timestamp,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ),
            ScheduleData::Trade(t) => (
                "trade".to_string(),
                e.instrument,
                e.local_idx,
                t.timestamp,
                t.price,
                t.size,
                0.0,
                0.0,
                0.0,
            ),
            ScheduleData::Depth(d) => {
                // Full levels come from `current_depth`; the scalar slots
                // carry the touch so a strategy can act without them.
                let book = self.session_ref().ok().and_then(|s| s.depth_at(d.slot));
                let (bid, bid_size) = book
                    .and_then(|b| b.bids.first().copied().filter(|_| b.bid_len > 0))
                    .map(|l| (l.price, l.size))
                    .unwrap_or((0.0, 0.0));
                let (ask, ask_size) = book
                    .and_then(|b| b.asks.first().copied().filter(|_| b.ask_len > 0))
                    .map(|l| (l.price, l.size))
                    .unwrap_or((0.0, 0.0));
                (
                    "book".to_string(),
                    e.instrument,
                    e.local_idx,
                    d.timestamp,
                    bid,
                    ask,
                    bid_size,
                    ask_size,
                    0.0,
                )
            }
            ScheduleData::Quote(q) => (
                "quote".to_string(),
                e.instrument,
                e.local_idx,
                q.timestamp,
                q.bid,
                q.ask,
                0.0,
                0.0,
                0.0,
            ),
        }))
    }

    /// Step the pending event through its instrument's kernel and advance.
    #[pyo3(signature = (entry=false, exit=false, atr=0.0, size_mult=None, stop_price=None, target_price=None))]
    fn apply_current(
        &mut self,
        entry: bool,
        exit: bool,
        atr: f64,
        size_mult: Option<f64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
    ) -> PyResult<Vec<PyEngineEvent>> {
        let input = StepInput {
            entry,
            exit,
            atr,
            size_mult,
            stop_price_override: stop_price,
            target_price_override: target_price,
        };
        Ok(self
            .session_mut()?
            .apply_current(input)
            .into_iter()
            .map(PyEngineEvent::from)
            .collect())
    }

    /// Submit a typed order routed to one instrument's kernel.
    #[pyo3(signature = (
        instrument, side, kind, submitted_idx, submitted_ts, client_id,
        units=None, size_frac=None, limit_price=None, trigger_price=None,
        tif="gtc", expire_ns=None, stop_price=None, target_price=None,
        offset=None, offset_kind="price", limit_offset=0.0,
        post_only=false, reduce_only=false, parent_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit_order(
        &mut self,
        instrument: usize,
        side: &str,
        kind: &str,
        submitted_idx: usize,
        submitted_ts: i64,
        client_id: &str,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
        tif: &str,
        expire_ns: Option<i64>,
        stop_price: Option<f64>,
        target_price: Option<f64>,
        offset: Option<f64>,
        offset_kind: &str,
        limit_offset: f64,
        post_only: bool,
        reduce_only: bool,
        parent_id: Option<u64>,
    ) -> PyResult<u64> {
        submit_order_on(
            self.session_mut()?.kernel_mut(instrument),
            side,
            kind,
            submitted_idx,
            submitted_ts,
            client_id,
            units,
            size_frac,
            limit_price,
            trigger_price,
            tif,
            expire_ns,
            stop_price,
            target_price,
            offset,
            offset_kind,
            limit_offset,
            post_only,
            reduce_only,
            parent_id,
        )
    }

    fn cancel_order(&mut self, instrument: usize, idx: usize, order_id: u64) -> PyResult<bool> {
        Ok(self.session_mut()?.kernel_mut(instrument).cancel_order(idx, order_id))
    }

    /// Replace a working order's prices and/or quantity on one instrument.
    #[pyo3(signature = (instrument, order_id, units=None, size_frac=None, limit_price=None, trigger_price=None))]
    fn modify_order(
        &mut self,
        instrument: usize,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> PyResult<bool> {
        let qty = parse_qty_spec(units, size_frac)?;
        Ok(self
            .session_mut()?
            .kernel_mut(instrument)
            .modify_order(order_id, qty, limit_price, trigger_price))
    }

    fn cancel_all_orders(&mut self, instrument: usize, idx: usize) -> PyResult<Vec<u64>> {
        Ok(self.session_mut()?.kernel_mut(instrument).cancel_all_orders(idx))
    }

    fn link_oco(&mut self, instrument: usize, order_ids: Vec<u64>) -> PyResult<()> {
        self.session_mut()?.kernel_mut(instrument).link_oco(&order_ids);
        Ok(())
    }

    fn request_close(&mut self, instrument: usize, position_id: u64) -> PyResult<()> {
        self.session_mut()?.kernel_mut(instrument).request_close(position_id);
        Ok(())
    }

    /// Open positions of one instrument, in opening order.
    fn positions(&self, instrument: usize) -> PyResult<Vec<PyPositionSnapshot>> {
        Ok(self
            .session_ref()?
            .kernel(instrument)
            .position_snapshots()
            .into_iter()
            .map(super::strategy_bindings::convert_snapshot)
            .collect())
    }

    /// Earliest open position of one instrument, or None.
    fn position(&self, instrument: usize) -> PyResult<Option<PyPositionSnapshot>> {
        Ok(self
            .session_ref()?
            .kernel(instrument)
            .position_snapshot()
            .map(super::strategy_bindings::convert_snapshot))
    }

    /// Portfolio equity: pool plus every instrument's last-known mark.
    fn equity(&self) -> PyResult<f64> {
        Ok(self.session_ref()?.equity())
    }

    /// Shared cash balance. In margin mode this includes locked initial
    /// margin; see `free_capital()` for what can fund a new position.
    fn cash(&self) -> PyResult<f64> {
        Ok(self.session_ref()?.cash())
    }

    /// Capital available to open new positions across all instruments.
    ///
    /// Equals `cash()` in cash mode; net of locked initial margin otherwise.
    fn free_capital(&self) -> PyResult<f64> {
        Ok(self.session_ref()?.free_capital())
    }

    /// Whether a margin call or drawdown kill-switch has latched.
    fn is_halted(&self) -> PyResult<bool> {
        Ok(self.session_ref()?.is_halted())
    }

    /// Force-close all instruments and compute portfolio metrics.
    fn finish(&mut self) -> PyResult<PyPortfolioResult> {
        let session = self
            .session
            .take()
            .ok_or_else(|| PyValueError::new_err("session is already finished"))?;
        let outcome = session.finish();
        Ok(PyPortfolioResult {
            result: convert_result(outcome.result),
            per_instrument: outcome
                .instruments
                .into_iter()
                .map(|o| PyInstrumentSummary {
                    symbol: o.symbol,
                    trades: o.trades,
                    pnl: o.pnl,
                    rejected_entries: o.rejected_entries,
                })
                .collect(),
            rejected_entries: outcome.rejected_entries,
            halted: outcome.halted,
            halted_at: outcome.halted_at,
        })
    }
}
