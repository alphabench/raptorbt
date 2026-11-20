//! Multi-leg options spread backtesting implementation.
//!
//! Provides high-performance spread backtesting for:
//! - Straddles and Strangles
//! - Vertical spreads (bull/bear call/put)
//! - Iron Condors and Iron Butterflies
//! - Calendar and Diagonal spreads
//!
//! Each leg is an option instrument of one [`PortfolioBook`] with its own
//! premium series, so the legs trade in one account and follow the kernel's
//! fill, fee, margin, expiry and risk rules. This module decides what is
//! particular to a structure: the legs open and close together, a structure
//! that cannot be funded whole does not open, and the legs of one entry are
//! reported as one trade.

use std::collections::HashMap;

use crate::core::types::{
    BacktestMetrics, BacktestResult, Direction, ExitReason, InstrumentConfig, Trade,
};
use crate::execution::indian_costs::FeeBreakdown;
use crate::instruments::lifecycle::{SettlementPrice, SettlementStyle, SettlementTerms};
use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::engine::compute_backtest_metrics_with_config;
use crate::portfolio::kernel::{KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::strategies::legs::{account_mode, bar_entry};

pub use super::spreads_config::{
    create_iron_condor_config, create_straddle_config, create_strangle_config,
    create_vertical_spread_config, LegConfig, OptionType, SpreadConfig, SpreadType,
};

/// Spread backtest runner.
pub struct SpreadBacktest {
    config: SpreadConfig,
}

impl SpreadBacktest {
    /// Create a new spread backtest.
    ///
    /// The fee model comes from `BacktestConfig::fee_model`, so setting
    /// `fee_segment` charges the itemized regulatory schedule -- per-order
    /// brokerage excluded -- and leaving it unset keeps the flat `fees` rate.
    pub fn new(config: SpreadConfig) -> Self {
        Self { config }
    }

    /// Run the spread backtest.
    ///
    /// # Arguments
    /// * `timestamps` - Timestamp array
    /// * `underlying_close` - Underlying close prices
    /// * `legs_premiums` - Premium series for each leg (Vec of Vec)
    /// * `entries` - Entry signals
    /// * `exits` - Exit signals
    ///
    /// # Returns
    /// Backtest result with metrics, trades, and equity curve
    pub fn run(
        &self,
        timestamps: &[i64],
        underlying_close: &[f64],
        legs_premiums: &[Vec<f64>],
        entries: &[bool],
        exits: &[bool],
    ) -> BacktestResult {
        self.run_with_opens(timestamps, underlying_close, legs_premiums, None, entries, exits)
    }

    /// [`Self::run`], with each leg's opening premiums supplied.
    ///
    /// Under [`FillTiming::NextBarOpen`] a signal fill lands on the bar
    /// after the decision; with `legs_open_premiums` present it prices the
    /// legs at that bar's OPEN premiums -- real quotes the caller observed --
    /// instead of the bar's (later) settled values. Nothing is synthesized:
    /// without the series, the next bar's value remains the honest causal
    /// fill. The opens are ignored outside `NextBarOpen`.
    ///
    /// A structure is opened whole or not at all: when its legs together cost
    /// more than the account has free, no leg opens. Each leg fills, pays
    /// costs and slips as any order does, and a leg left to expire settles at
    /// its series' value on that bar and owes no exit costs. The structure
    /// closes on an exit signal, the session square-off, `max_loss`,
    /// `target_profit`, or once its last leg has expired. A close that is not
    /// a signal -- `max_loss`, `target_profit`, the square-off -- fills when a
    /// signal exit would: on the bar's premiums, or the next bar's under
    /// `NextBarOpen`.
    ///
    /// [`FillTiming::NextBarOpen`]: crate::core::types::FillTiming::NextBarOpen
    pub fn run_with_opens(
        &self,
        timestamps: &[i64],
        _underlying_close: &[f64],
        legs_premiums: &[Vec<f64>],
        legs_open_premiums: Option<&[Vec<f64>]>,
        entries: &[bool],
        exits: &[bool],
    ) -> BacktestResult {
        let n = timestamps.len();

        // Open premiums, when given, must mirror the premium series' shape
        // exactly -- same legs, same bars. A mismatch would price some legs
        // off a different series than their marks, silently.
        if let Some(opens) = legs_open_premiums {
            if opens.len() != legs_premiums.len() || opens.iter().any(|o| o.len() != n) {
                return self.empty_result(n);
            }
        }

        // Validate inputs
        if legs_premiums.len() != self.config.leg_configs.len() {
            return self.empty_result(n);
        }

        for premiums in legs_premiums {
            if premiums.len() != n {
                return self.empty_result(n);
            }
        }

        // Expiries are matched to legs by position, so a list of the wrong
        // length would settle the wrong leg or leave the trailing legs
        // immortal -- both silently. Refuse instead.
        if let Some(expiries) = self.config.leg_expiry_timestamps.as_ref() {
            if expiries.len() != self.config.leg_configs.len() {
                return self.empty_result(n);
            }
        }

        let base = &self.config.base;
        let next_open = base.resolved_fill_timing() == crate::core::types::FillTiming::NextBarOpen;
        let expiries = self.config.leg_expiry_timestamps.as_deref();

        // Session squareoff. Computed once up front rather than per bar: the
        // day boundary depends only on the timestamps, not on position state.
        let squareoff: Vec<bool> = match base.squareoff_time_minutes {
            Some(minutes) => crate::core::session::squareoff_flags(
                timestamps,
                minutes,
                base.session_tz_offset_ns,
            ),
            None => vec![false; n],
        };

        let legs = &self.config.leg_configs;
        // This runner flattens on its own flags above; the book's kernels must not
        // square off a second time from the same setting.
        let mut book_config = base.clone();
        book_config.squareoff_time_minutes = None;
        book_config.squareoff_time_by_symbol.clear(); // a spread's legs square off together
        let mut book = PortfolioBook::new(book_config, account_mode(base));
        book.reserve_curves(n);
        for (k, leg) in legs.iter().enumerate() {
            book.add_instrument(
                format!("SPREAD_LEG{k}"),
                if leg.is_long() { Direction::Long } else { Direction::Short },
                Some(leg_spec(k, leg, expiries.map(|e| e[k]))),
                Some(&InstrumentConfig::default()),
                PositionPolicy::Net,
            );
        }
        // Contracts per leg: quantity in lots times the lot size. A leg of no
        // quantity places no order.
        let units: Vec<f64> =
            legs.iter().map(|l| l.quantity.unsigned_abs() as f64 * l.lot_size as f64).collect();
        let traded: Vec<usize> = (0..legs.len()).filter(|&k| units[k] > 0.0).collect();

        // Why a structure closed when the kernel cannot say: the exit it was
        // asked for is a signal exit to the kernel. Keyed by entry bar.
        let mut forced: HashMap<usize, ExitReason> = HashMap::new();
        let mut unwind = false;

        let specs: Vec<InstrumentSpec> = legs
            .iter()
            .enumerate()
            .map(|(k, leg)| leg_spec(k, leg, expiries.map(|e| e[k])))
            .collect();
        let mut order: Vec<usize> = (0..legs.len()).collect();
        for i in 0..n {
            let ts = timestamps[i];

            // The structure's decisions, taken once per bar on the state
            // before any leg steps.
            let held: Vec<usize> =
                traded.iter().copied().filter(|&j| book.kernel(j).is_in_position()).collect();
            let waiting = traded.iter().any(|&j| book.kernel(j).has_pending_entry());
            let mut exiting = false;
            if !held.is_empty() {
                // A leg not yet expired and not held, beside one that is: an
                // entry the kernel refused part of.
                let partial = !waiting
                    && traded
                        .iter()
                        .any(|&j| !held.contains(&j) && expiries.is_none_or(|e| ts < e[j]));
                let live_pnl: f64 = held
                    .iter()
                    .filter(|&&j| expiries.is_none_or(|e| ts < e[j]))
                    .map(|&j| {
                        book.kernel(j).position_snapshot().map_or(0.0, |s| {
                            s.direction.multiplier()
                                * (legs_premiums[j][i] - s.entry_price)
                                * s.size
                        })
                    })
                    .sum();
                let reason = if exits[i] || unwind || partial {
                    Some(ExitReason::Signal)
                } else if squareoff[i] {
                    Some(ExitReason::Squareoff)
                } else if self.config.max_loss.is_some_and(|m| live_pnl < -m) {
                    Some(ExitReason::StopLoss)
                } else if self.config.target_profit.is_some_and(|t| live_pnl > t) {
                    Some(ExitReason::TakeProfit)
                } else {
                    None
                };
                if let Some(reason) = reason {
                    exiting = true;
                    if reason != ExitReason::Signal {
                        if let Some(s) = book.kernel(held[0]).position_snapshot() {
                            forced.entry(s.entry_idx).or_insert(reason);
                        }
                    }
                }
            }
            unwind = false;

            // One dead leg is enough to block re-entry: a structure
            // containing an expired contract is not one anybody can open.
            let any_expired = expiries.is_some_and(|e| e.iter().any(|&x| ts >= x));
            let flat_after = held.is_empty() || exiting;
            let mut entering = flat_after
                && !waiting
                && entries[i]
                && !any_expired
                && !squareoff[i]
                && !traded.is_empty();
            if entering {
                let rows =
                    LegRows { premiums: legs_premiums, opens: legs_open_premiums, next_open };
                entering = self.fits(&book, &traded, &units, &specs, &rows, i, ts);
            }

            // A leg that is expiring on this bar steps first, on its own bar:
            // the book settles a contract that expired before it has been
            // shown that bar at the last price it saw instead. Entering,
            // bought legs go before sold ones, so a sold leg is funded with
            // its hedge already open.
            order.sort_by_key(|&k| {
                let expiring =
                    expiries.is_some_and(|e| ts >= e[k]) && book.kernel(k).is_in_position();
                (!expiring, entering && !legs[k].is_long(), k)
            });
            for &k in &order {
                let premium = legs_premiums[k][i];
                let open = match (next_open, legs_open_premiums) {
                    (true, Some(opens)) => opens[k][i],
                    _ => premium,
                };
                let bar = KernelBar {
                    timestamp: ts,
                    open,
                    high: premium.max(open),
                    low: premium.min(open),
                    close: premium,
                    volume: 0.0,
                };
                let trades_leg = entering && units[k] > 0.0;
                let input = StepInput {
                    entry: trades_leg,
                    exit: exiting,
                    units: trades_leg.then_some(units[k]),
                    ..StepInput::default()
                };
                let events = book.apply(&bar_entry(k, i, bar), None, input);
                // A leg refused after its siblings opened leaves a structure
                // nobody asked for; it is closed on the next decision.
                if entering
                    && events.iter().any(|e| {
                        matches!(e, crate::portfolio::kernel::EngineEvent::EntryRejected { .. })
                    })
                {
                    unwind = true;
                }
            }
        }

        let outcome = book.finish();
        let mut result = outcome.result;
        let trades = aggregate(&result.trades, &forced);
        result.metrics = compute_backtest_metrics_with_config(
            &result.equity_curve,
            &result.drawdown_curve,
            &result.returns,
            &trades,
            timestamps,
            &self.config.base,
        );
        result.trades = trades;
        result
    }

    /// Whether the legs together fit in the capital free now.
    ///
    /// Each leg's funding and costs as the kernel would take them, summed. When
    /// the account margins option lines together the structure's requirement
    /// is the model's for the legs as a group, not the sum of each leg alone.
    #[allow(clippy::too_many_arguments)]
    fn fits(
        &self,
        book: &PortfolioBook,
        traded: &[usize],
        units: &[f64],
        specs: &[InstrumentSpec],
        rows: &LegRows<'_>,
        i: usize,
        ts: i64,
    ) -> bool {
        use crate::accounts::LineView;
        let mut needed = 0.0;
        let mut lines: Vec<LineView<'_>> = Vec::new();
        for &k in traded {
            let premium = rows.premiums[k][i];
            let open = match (rows.next_open, rows.opens) {
                (true, Some(opens)) => opens[k][i],
                _ => premium,
            };
            let bar = KernelBar {
                timestamp: ts,
                open,
                high: premium.max(open),
                low: premium.min(open),
                close: premium,
                volume: 0.0,
            };
            match book.kernel(k).entry_outlay(&bar, units[k]) {
                Some((funded, fees)) => needed += funded + fees,
                // A refusal of the funding itself is the kernel's to report.
                None => return true,
            }
            let sign = if self.config.leg_configs[k].is_long() { 1.0 } else { -1.0 };
            lines.push(LineView::new(&specs[k], sign * units[k], open));
        }
        let grouped = self
            .config
            .base
            .margin_model
            .as_deref()
            .filter(|model| model.groups_options())
            .and_then(|model| model.group_required(&lines, ts).ok());
        if let Some(required) = grouped {
            // The group's requirement replaces the deposits; the costs stay.
            let fees: f64 = traded
                .iter()
                .filter_map(|&k| book.kernel(k).entry_outlay(&leg_bar(rows, k, i, ts), units[k]))
                .map(|(_, fees)| fees)
                .sum();
            return required + fees <= book.free_capital();
        }
        needed <= book.free_capital()
    }

    /// Create an empty result (used for validation failures).
    fn empty_result(&self, n: usize) -> BacktestResult {
        BacktestResult {
            unverified_fee_rows: Vec::new(),
            metrics: BacktestMetrics::default(),
            equity_curve: vec![self.config.base.initial_capital; n],
            drawdown_curve: vec![0.0; n],
            trades: Vec::new(),
            returns: vec![0.0; n],
            // This path synthesises a result from leg P&L; it runs no
            // order book, so there are no orders to report.
            orders: Vec::new(),
            halted: false,
            halted_at: None,
            rejected_entries: 0,
            unfunded_entries: 0,
        }
    }
}

/// The premium rows a bar's entry is priced from.
struct LegRows<'a> {
    premiums: &'a [Vec<f64>],
    opens: Option<&'a [Vec<f64>]>,
    next_open: bool,
}

/// Leg `k`'s bar `i` as the kernel takes it.
fn leg_bar(rows: &LegRows<'_>, k: usize, i: usize, ts: i64) -> KernelBar {
    let premium = rows.premiums[k][i];
    let open = match (rows.next_open, rows.opens) {
        (true, Some(opens)) => opens[k][i],
        _ => premium,
    };
    KernelBar {
        timestamp: ts,
        open,
        high: premium.max(open),
        low: premium.min(open),
        close: premium,
        volume: 0.0,
    }
}

/// The instrument one leg trades: an option on its premium series, one
/// contract to the unit, that settles at the series' value on its expiry.
fn leg_spec(k: usize, leg: &LegConfig, expiry: Option<i64>) -> InstrumentSpec {
    let right = match leg.option_type {
        OptionType::Call => OptionRight::Call,
        OptionType::Put => OptionRight::Put,
    };
    let mut spec = InstrumentSpec::new(
        format!("SPREAD_LEG{k}"),
        InstrumentKind::Option {
            strike: leg.strike,
            right,
            // One underlying, so the legs hedge each other under a margin
            // model that groups option lines.
            underlying: Some("SPREAD".to_string()),
            binary: false,
        },
    );
    spec.expiration_ns = expiry;
    // The engine does not compute intrinsic value here: the series carries
    // the leg's settlement value at and after its expiry.
    spec.settlement =
        SettlementTerms { style: SettlementStyle::Cash, price: SettlementPrice::LastPrice };
    spec
}

/// One trade per structure from the trades of its legs.
///
/// Legs that opened together share an entry bar. The structure's prices are
/// its net premium, a debit positive and a credit negative; its profit and
/// costs are the legs' summed, so the record is the money the account saw.
fn aggregate(leg_trades: &[Trade], forced: &HashMap<usize, ExitReason>) -> Vec<Trade> {
    let mut by_entry: Vec<(usize, Vec<&Trade>)> = Vec::new();
    for trade in leg_trades {
        match by_entry.iter_mut().find(|(idx, _)| *idx == trade.entry_idx) {
            Some((_, group)) => group.push(trade),
            None => by_entry.push((trade.entry_idx, vec![trade])),
        }
    }
    by_entry.sort_by_key(|(idx, _)| *idx);

    by_entry
        .into_iter()
        .enumerate()
        .map(|(n, (entry_idx, group))| {
            let signed = |trade: &Trade, price: f64| {
                let sign = if trade.direction == Direction::Long { 1.0 } else { -1.0 };
                sign * price * trade.size
            };
            let entry_premium: f64 = group.iter().map(|t| signed(t, t.entry_price)).sum();
            let exit_premium: f64 = group.iter().map(|t| signed(t, t.exit_price)).sum();
            let pnl: f64 = group.iter().map(|t| t.pnl).sum();
            let entry_fees: f64 = group.iter().map(|t| t.entry_fees).sum();
            let exit_fees: f64 = group.iter().map(|t| t.exit_fees).sum();
            let last = group.iter().max_by_key(|t| t.exit_idx).expect("a group has a trade");
            let mut breakdown: Option<FeeBreakdown> = None;
            for trade in &group {
                if let Some(b) = trade.fee_breakdown {
                    breakdown.get_or_insert_with(FeeBreakdown::default).add(&b);
                }
            }
            // Whole-structure money, when every leg booked its own.
            let money =
                group.iter().map(|t| t.money).collect::<Option<Vec<_>>>().and_then(|moneys| {
                    let first = *moneys.first()?;
                    Some(moneys.iter().skip(1).fold(first, |mut acc, m| {
                        acc.entry_notional += m.entry_notional;
                        acc.exit_notional += m.exit_notional;
                        acc.entry_fees += m.entry_fees;
                        acc.exit_fees += m.exit_fees;
                        acc.pnl += m.pnl;
                        acc
                    }))
                });
            Trade {
                id: n as u64 + 1,
                symbol: "SPREAD".to_string(),
                entry_idx,
                exit_idx: last.exit_idx,
                entry_price: entry_premium,
                exit_price: exit_premium,
                size: 1.0,
                direction: Direction::Long, // Spreads are treated as "long spread"
                pnl,
                return_pct: return_pct(pnl, entry_premium),
                entry_time: group.iter().map(|t| t.entry_time).min().unwrap_or(0),
                exit_time: last.exit_time,
                fees: entry_fees + exit_fees,
                entry_fees,
                exit_fees,
                fee_breakdown: breakdown,
                exit_reason: forced.get(&entry_idx).copied().unwrap_or(last.exit_reason),
                // The structure is synthesised from its legs, which are not
                // measured as one position. None means "not measured".
                mae_price: None,
                mfe_price: None,
                mae_pnl: None,
                mfe_pnl: None,
                money,
            }
        })
        .collect()
}

/// Return as a percentage of the capital the structure opened with.
///
/// Zero when the structure cost nothing to open. A spread whose legs
/// finance each other exactly -- a calendar entered at equal premiums is
/// the everyday case -- has no denominator to divide by, and the answer
/// is not infinity, it is undefined. Reporting 0.0 keeps the metric
/// finite; the P&L in rupees is the honest figure there.
fn return_pct(net_pnl: f64, entry_premium: f64) -> f64 {
    if entry_premium.abs() > 0.0 {
        net_pnl / entry_premium.abs() * 100.0
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "spreads_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "spreads_settlement_tests.rs"]
mod settlement_tests;
