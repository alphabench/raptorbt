//! Multi-leg options spread backtesting implementation.
//!
//! Provides high-performance spread backtesting for:
//! - Straddles and Strangles
//! - Vertical spreads (bull/bear call/put)
//! - Iron Condors and Iron Butterflies
//! - Calendar and Diagonal spreads
//!
//! Key features:
//! - Single-pass O(n) algorithm
//! - Coordinated entry/exit across all legs
//! - Net premium P&L calculation
//! - Combined Greeks tracking

use crate::core::types::{BacktestMetrics, BacktestResult, Direction, ExitReason, Trade};
use crate::execution::{indian_costs::FeeBreakdown, FeeModel};
use crate::metrics::streaming::StreamingMetrics;

pub use super::spreads_config::{
    create_iron_condor_config, create_straddle_config, create_strangle_config,
    create_vertical_spread_config, LegConfig, OptionType, SpreadConfig, SpreadType,
};

/// State for a single leg position.
#[derive(Debug, Clone)]
struct LegPosition {
    /// Entry premium price.
    pub entry_premium: f64,
    /// Entry index.
    #[allow(dead_code)]
    pub entry_idx: usize,
    /// Current premium price.
    pub current_premium: f64,
    /// Leg configuration.
    pub config: LegConfig,
    /// Bar index at which this leg settled at its own expiry, once it has.
    /// `None` while the leg is still live.
    pub settled_idx: Option<usize>,
}

impl LegPosition {
    fn new(config: LegConfig, entry_premium: f64, entry_idx: usize) -> Self {
        Self { entry_premium, entry_idx, current_premium: entry_premium, config, settled_idx: None }
    }

    /// Whether this leg has reached its own expiry and been settled.
    fn is_settled(&self) -> bool {
        self.settled_idx.is_some()
    }

    /// P&L this leg still contributes to a mark-to-market reading.
    ///
    /// Zero once the leg has settled. The settled amount was credited to
    /// cash on its expiry bar, and cash is the other half of the equity
    /// line -- counting it here too would bank the same rupees twice.
    fn open_pnl(&self) -> f64 {
        if self.is_settled() {
            0.0
        } else {
            self.unrealized_pnl()
        }
    }

    /// Calculate unrealized P&L for this leg.
    fn unrealized_pnl(&self) -> f64 {
        // No negation here, deliberately. `LegConfig.quantity` is ALREADY
        // signed (+1 long, -1 short -- see its doc comment, and `is_short()`,
        // which is literally `quantity < 0`), so the direction convention is
        // applied exactly once, by that sign:
        //
        //   short (-1) + premium falls (-30) -> (-1) * (-30) * 75 = +2250 gain
        //   long  (+1) + premium falls (-30) -> (+1) * (-30) * 75 = -2250 loss
        //
        // A leading minus applies the same convention a second time and
        // negates every result. It shipped through 0.6.3 and inverted `pnl`,
        // the equity curve, and -- worse -- the `max_loss` / `target_profit`
        // triggers, which closed winning structures on a stop.
        let premium_change = self.current_premium - self.entry_premium;
        let quantity = self.config.quantity as f64;
        let lot_size = self.config.lot_size as f64;
        quantity * premium_change * lot_size
    }
}

/// Spread position state.
#[derive(Debug, Clone)]
struct SpreadPosition {
    /// Individual leg positions.
    pub legs: Vec<LegPosition>,
    /// Entry bar index.
    pub entry_idx: usize,
    /// Entry net premium (positive = credit, negative = debit).
    pub entry_net_premium: f64,
    /// Entry timestamp.
    pub entry_time: i64,
    /// Whether position is open.
    pub is_open: bool,
    /// Costs charged when this position was opened.
    ///
    /// Retained rather than recomputed at exit: the entry is charged against
    /// entry premiums, and re-deriving it from exit premiums would both bill a
    /// different amount and leave the trade record disagreeing with the cash
    /// that actually left the account.
    pub entry_fees: f64,
    /// Itemized entry costs, when an itemized fee model is configured.
    pub entry_breakdown: Option<FeeBreakdown>,
    /// P&L from legs that reached their own expiry before the structure
    /// closed, and which has already been credited to cash.
    ///
    /// Held apart from the mark-to-market so a settled leg is counted once,
    /// and added back when the trade is reported so the record covers the
    /// whole structure rather than only the legs that outlived it.
    pub realized_pnl: f64,
}

impl SpreadPosition {
    fn new(legs: Vec<LegPosition>, entry_idx: usize, entry_time: i64) -> Self {
        let entry_net_premium: f64 = legs
            .iter()
            .map(|leg| leg.entry_premium * leg.config.quantity as f64 * leg.config.lot_size as f64)
            .sum();

        Self {
            legs,
            entry_idx,
            entry_net_premium,
            entry_time,
            is_open: true,
            entry_fees: 0.0,
            entry_breakdown: None,
            realized_pnl: 0.0,
        }
    }

    /// Mark-to-market P&L across the legs that are still live.
    fn total_unrealized_pnl(&self) -> f64 {
        self.legs.iter().map(|leg| leg.open_pnl()).sum()
    }

    /// P&L of the whole structure, settled legs included. What a trade
    /// reports, as against what is still owed to cash.
    fn lifetime_pnl(&self) -> f64 {
        self.realized_pnl + self.total_unrealized_pnl()
    }

    /// Update leg premiums.
    ///
    /// A settled leg keeps the premium it settled at. Its contract no longer
    /// exists, so whatever the series carries past that bar is not a price
    /// -- a stale quote there must not reach the exit price or the P&L.
    fn update_premiums(&mut self, leg_premiums: &[f64]) {
        for (leg, &premium) in self.legs.iter_mut().zip(leg_premiums.iter()) {
            if leg.is_settled() {
                continue;
            }
            leg.current_premium = premium;
        }
    }

    /// Settle every leg whose own expiry has now passed, returning the cash
    /// to credit for them.
    ///
    /// A leg settles once, at whatever premium its series carries on the
    /// settlement bar -- the engine does not compute intrinsic value, and
    /// the contract on `SpreadConfig::leg_expiry_timestamps` says so. Its
    /// P&L moves out of the mark-to-market and into `realized_pnl`, so the
    /// caller can credit it to cash without the equity line double-counting.
    fn settle_expired_legs(&mut self, idx: usize, timestamp: i64, expiries: &[i64]) -> f64 {
        let mut credited = 0.0;
        for (leg, &expiry) in self.legs.iter_mut().zip(expiries.iter()) {
            if leg.is_settled() || timestamp < expiry {
                continue;
            }
            credited += leg.unrealized_pnl();
            leg.settled_idx = Some(idx);
        }
        self.realized_pnl += credited;
        credited
    }

    /// Whether every leg has reached its own expiry.
    fn all_legs_settled(&self) -> bool {
        self.legs.iter().all(|leg| leg.is_settled())
    }

    /// Close the position and return the P&L still owed to cash.
    ///
    /// Live legs only: anything already settled was credited on its own
    /// expiry bar.
    fn close(&mut self) -> f64 {
        self.is_open = false;
        self.total_unrealized_pnl()
    }
}

/// Spread backtest runner.
pub struct SpreadBacktest {
    config: SpreadConfig,
    fee_model: FeeModel,
}

impl SpreadBacktest {
    /// Create a new spread backtest.
    ///
    /// The fee model comes from `BacktestConfig::fee_model`, so setting
    /// `fee_segment` charges the itemized regulatory schedule -- per-order
    /// brokerage included -- and leaving it unset keeps the flat `fees` rate.
    pub fn new(config: SpreadConfig) -> Self {
        let fee_model = config.base.fee_model();
        Self { config, fee_model }
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
        _underlying_close: &[f64],
        legs_premiums: &[Vec<f64>],
        entries: &[bool],
        exits: &[bool],
    ) -> BacktestResult {
        let n = timestamps.len();

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

        let mut metrics = StreamingMetrics::with_initial_capital(self.config.base.initial_capital);
        let mut equity_curve = Vec::with_capacity(n);
        let mut drawdown_curve = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        let mut trades: Vec<Trade> = Vec::new();
        let mut trade_id: u64 = 0;

        let mut cash = self.config.base.initial_capital;
        let mut position: Option<SpreadPosition> = None;
        let mut prev_equity = cash;

        // Session squareoff. Computed once up front rather than per bar: the
        // day boundary depends only on the timestamps, not on position state.
        // Empty (all-false) when squareoff is disabled, so the hot loop reads
        // one bool either way.
        let squareoff: Vec<bool> = match self.config.base.squareoff_time_minutes {
            Some(minutes) => crate::core::session::squareoff_flags(
                timestamps,
                minutes,
                self.config.base.session_tz_offset_ns,
            ),
            None => vec![false; n],
        };

        // Single-pass O(n) algorithm
        for i in 0..n {
            // Get current leg premiums
            let current_premiums: Vec<f64> = legs_premiums.iter().map(|p| p[i]).collect();

            // Update position premiums if open
            if let Some(ref mut pos) = position {
                pos.update_premiums(&current_premiums);
            }

            // Settle any leg that reached its own expiry at this bar, before
            // anything reads the position's P&L.
            //
            // Each leg dies on its own date. Closing the whole structure when
            // the FIRST one expires is what 0.7.4 did, and it made a calendar
            // spread -- sell the near expiry, buy the far one -- unmeasurable:
            // the engine closed before the far leg had done anything, so the
            // result was not merely optimistic, it was unrelated to the trade.
            if let (Some(pos), Some(expiries)) =
                (position.as_mut(), self.config.leg_expiry_timestamps.as_ref())
            {
                // No exit fee: an option left to expire is never sold, so no
                // order is placed and no brokerage or transaction tax is owed.
                cash += pos.settle_expired_legs(i, timestamps[i], expiries);
            }

            // Calculate unrealized P&L for exit checks
            let unrealized_pnl = position.as_ref().map(|p| p.total_unrealized_pnl()).unwrap_or(0.0);

            // The structure is finished only once its LAST leg has expired.
            let is_expiry = position.as_ref().is_some_and(|p| p.all_legs_settled());

            // Check for exit signals or conditions
            let should_exit = position.is_some()
                && (exits[i]
                    || is_expiry
                    || squareoff[i]
                    || self.check_max_loss(&position, unrealized_pnl)
                    || self.check_target_profit(&position, unrealized_pnl));

            if should_exit {
                if let Some(mut pos) = position.take() {
                    let pnl = pos.close();

                    // Resolved before costs are charged: settlement pays no
                    // exit-side fee, so the reason decides the bill.
                    let exit_reason = if is_expiry {
                        ExitReason::Settlement
                    } else if exits[i] {
                        ExitReason::Signal
                    } else if squareoff[i] {
                        ExitReason::Squareoff
                    } else if self.check_max_loss(&Some(pos.clone()), pnl) {
                        ExitReason::StopLoss
                    } else {
                        ExitReason::TakeProfit
                    };

                    trade_id += 1;
                    let trade = self.emit_trade(&pos, trade_id, i, timestamps[i], exit_reason);
                    cash += pnl - trade.exit_fees;
                    Self::record_trade(&mut metrics, &trade, &pos);
                    trades.push(trade);
                }
            }

            // Check for entry signals.
            //
            // One dead leg is enough to block re-entry, not all of them: a
            // structure containing an expired contract is not one anybody can
            // open, and entering it would price that leg off a series that is
            // no longer quoting anything. Identical to the old all-expired
            // test for a same-expiry structure, where the two coincide.
            let any_expired = self
                .config
                .leg_expiry_timestamps
                .as_ref()
                .is_some_and(|expiries| expiries.iter().any(|&exp_ts| timestamps[i] >= exp_ts));
            if position.is_none() && entries[i] && !any_expired && !squareoff[i] {
                let legs: Vec<LegPosition> = self
                    .config
                    .leg_configs
                    .iter()
                    .zip(current_premiums.iter())
                    .map(|(cfg, &premium)| LegPosition::new(cfg.clone(), premium, i))
                    .collect();

                let mut new_position = SpreadPosition::new(legs, i, timestamps[i]);

                let (entry_fees, entry_breakdown) = self.calculate_side(&new_position, true);
                new_position.entry_fees = entry_fees;
                new_position.entry_breakdown = entry_breakdown;
                cash -= entry_fees;

                position = Some(new_position);
            }

            // Update equity tracking
            let equity = cash + position.as_ref().map(|p| p.total_unrealized_pnl()).unwrap_or(0.0);
            equity_curve.push(equity);

            let daily_return =
                if prev_equity > 0.0 { (equity - prev_equity) / prev_equity } else { 0.0 };
            returns.push(daily_return);
            prev_equity = equity;

            // Update drawdown
            metrics.update_equity(equity);
            drawdown_curve.push(metrics.current_drawdown_pct());
        }

        // Close any remaining open position at end.
        //
        // This MUST record a Trade, not merely settle into `cash`. Until
        // 0.7.2 it did the latter: the position's P&L reached `end_value`
        // and the equity curve, while `trades()` stayed empty and
        // `total_open_trades` read 0. A caller auditing trade-by-trade saw
        // a clean, empty book for a run whose return was driven entirely by
        // a position that never closed -- the worst shape for a defect,
        // because every trade-level check passes.
        if let Some(mut pos) = position.take() {
            let last = n - 1;
            let pnl = pos.close();

            trade_id += 1;
            let trade =
                self.emit_trade(&pos, trade_id, last, timestamps[last], ExitReason::EndOfData);
            cash += pnl - trade.exit_fees;
            Self::record_trade(&mut metrics, &trade, &pos);
            trades.push(trade);
        }

        // Finalize metrics
        let final_metrics = metrics.finalize(self.config.base.initial_capital, cash, &returns);

        BacktestResult { metrics: final_metrics, equity_curve, drawdown_curve, trades, returns }
    }

    /// Check if max loss threshold is hit.
    fn check_max_loss(&self, _position: &Option<SpreadPosition>, unrealized_pnl: f64) -> bool {
        if let Some(max_loss) = self.config.max_loss {
            if unrealized_pnl < -max_loss {
                return true;
            }
        }
        false
    }

    /// Check if target profit threshold is hit.
    fn check_target_profit(&self, _position: &Option<SpreadPosition>, unrealized_pnl: f64) -> bool {
        if let Some(target) = self.config.target_profit {
            if unrealized_pnl > target {
                return true;
            }
        }
        false
    }

    /// Costs for one side of a spread, charged leg by leg.
    ///
    /// Each leg is a separate exchange order, so a flat per-order charge is
    /// levied once per leg -- an N-leg structure pays it N times per side, and
    /// summing the legs' premiums first would collect it only once.
    ///
    /// A leg's direction is the sign of its quantity, and `is_entry` decides
    /// which way that points: a short leg is sold to open and bought to close,
    /// so side-specific charges (transaction tax on the sell, stamp duty on the
    /// buy) land on the leg and side that actually owes them.
    ///
    /// Returns the total and, for itemized models, the component breakdown.
    fn calculate_side(
        &self,
        position: &SpreadPosition,
        is_entry: bool,
    ) -> (f64, Option<FeeBreakdown>) {
        let mut total = 0.0;
        let mut breakdown: Option<FeeBreakdown> = None;

        for leg in &position.legs {
            // A leg holding nothing places no order, so it owes no per-order
            // charge. Without this a zero-quantity leg would be billed full
            // brokerage for a trade that never happened.
            if leg.config.quantity == 0 {
                continue;
            }

            // A leg left to expire is never traded out, so it owes nothing on
            // the way out. Only on the exit side: the order that opened it was
            // real regardless of how it ended, and entry costs must not change
            // retroactively.
            if !is_entry && leg.is_settled() {
                continue;
            }

            let premium = if is_entry { leg.entry_premium } else { leg.current_premium };
            // Contract count, not lots: a two-lot leg trades twice the
            // contracts of a one-lot leg and owes proportionally more.
            let size = (leg.config.quantity.unsigned_abs() as f64) * leg.config.lot_size as f64;
            let direction = if leg.config.is_long() { Direction::Long } else { Direction::Short };

            match self.fee_model.breakdown(premium.abs(), size, direction, is_entry) {
                Some(leg_breakdown) => {
                    total += leg_breakdown.total();
                    breakdown.get_or_insert_with(FeeBreakdown::default).add(&leg_breakdown);
                }
                None => total += self.fee_model.calculate(premium.abs(), size, direction),
            }
        }

        (total, breakdown)
    }

    /// Costs for closing a position, and the round-trip breakdown to report.
    ///
    /// An option left to expire is never traded out: no order is placed, so no
    /// brokerage and no transaction tax are owed. Charging a full exit there
    /// would overstate the cost of every structure held to expiry, so
    /// settlement pays nothing on the way out. Entry costs still stand.
    fn calculate_exit(
        &self,
        position: &SpreadPosition,
        exit_reason: ExitReason,
    ) -> (f64, Option<FeeBreakdown>) {
        let (exit_fees, exit_breakdown) = if exit_reason == ExitReason::Settlement {
            (0.0, None)
        } else {
            self.calculate_side(position, false)
        };

        // Entry components plus exit components, so the itemized total equals
        // the fees actually deducted from the equity curve.
        let combined = match (position.entry_breakdown, exit_breakdown) {
            (Some(entry), Some(exit)) => {
                let mut total = entry;
                total.add(&exit);
                Some(total)
            }
            (entry, exit) => entry.or(exit),
        };

        (exit_fees, combined)
    }

    /// Build the trade record for a position that has just closed.
    ///
    /// Both exit paths -- the in-loop exit and the end-of-data sweep -- route
    /// through here. They were duplicated blocks until 0.8.0, differing only
    /// in whether they guarded a division; keeping one body is what stops the
    /// two from drifting apart on how a closed position is reported.
    ///
    /// The reported `pnl` covers the whole structure. What the caller credits
    /// to cash is narrower -- the live legs' P&L less the exit fee -- because
    /// a settled leg was credited on its own expiry bar and the entry side
    /// left cash when the position opened.
    fn emit_trade(
        &self,
        position: &SpreadPosition,
        trade_id: u64,
        exit_idx: usize,
        exit_time: i64,
        exit_reason: ExitReason,
    ) -> Trade {
        let (exit_fees, fee_breakdown) = self.calculate_exit(position, exit_reason);
        let entry_fees = position.entry_fees;
        let fees = entry_fees + exit_fees;
        // The whole structure, legs that settled early included -- not just
        // the ones that were still alive at the close.
        let net_pnl = position.lifetime_pnl() - fees;

        let entry_premium = position.entry_net_premium;
        // From the position's own legs, not the bar row: a settled leg is
        // frozen at what it settled for, and the series past its expiry is
        // a quote on a contract that no longer exists.
        let exit_premium: f64 = position
            .legs
            .iter()
            .map(|leg| {
                leg.current_premium * leg.config.quantity as f64 * leg.config.lot_size as f64
            })
            .sum();

        Trade {
            id: trade_id,
            symbol: "SPREAD".to_string(),
            entry_idx: position.entry_idx,
            exit_idx,
            entry_price: entry_premium,
            exit_price: exit_premium,
            size: 1.0,
            direction: Direction::Long, // Spreads are treated as "long spread"
            pnl: net_pnl,
            return_pct: Self::return_pct(net_pnl, entry_premium),
            entry_time: position.entry_time,
            exit_time,
            fees,
            entry_fees,
            exit_fees,
            fee_breakdown,
            exit_reason,
        }
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

    /// Fold a closed trade into the running metrics.
    ///
    /// A structure opened for nothing is skipped rather than recorded with a
    /// zero return, so it cannot drag the average return toward zero on a
    /// figure that was never measurable.
    fn record_trade(metrics: &mut StreamingMetrics, trade: &Trade, position: &SpreadPosition) {
        if trade.entry_price.abs() > 0.0 {
            metrics.record_fees(trade.fees);
            metrics.record_trade(
                trade.pnl,
                Self::return_pct(trade.pnl, trade.entry_price),
                trade.exit_idx - position.entry_idx,
            );
        }
    }

    /// Create an empty result (used for validation failures).
    fn empty_result(&self, n: usize) -> BacktestResult {
        BacktestResult {
            metrics: BacktestMetrics::default(),
            equity_curve: vec![self.config.base.initial_capital; n],
            drawdown_curve: vec![0.0; n],
            trades: Vec::new(),
            returns: vec![0.0; n],
        }
    }
}

#[cfg(test)]
#[path = "spreads_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "spreads_settlement_tests.rs"]
mod settlement_tests;
