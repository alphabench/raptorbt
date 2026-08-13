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

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, Direction, ExitReason, Trade,
};
use crate::execution::{indian_costs::FeeBreakdown, FeeModel};
use crate::metrics::streaming::StreamingMetrics;
use serde::{Deserialize, Serialize};

/// Spread type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpreadType {
    Straddle,
    Strangle,
    VerticalCall,
    VerticalPut,
    IronCondor,
    IronButterfly,
    ButterflyCall,
    ButterflyPut,
    Calendar,
    Diagonal,
    LongCall,
    LongPut,
    NakedCall,
    NakedPut,
    Custom,
}

/// Option type for a leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    /// Parse a broker option-type code (`CE`/`CALL`/`C`, `PE`/`PUT`/`P`).
    ///
    /// Case-insensitive. Returns `None` for anything else rather than
    /// guessing, because defaulting an unrecognised code to `Call` would
    /// price a put as a call.
    pub fn from_code(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "CE" | "CALL" | "C" => Some(OptionType::Call),
            "PE" | "PUT" | "P" => Some(OptionType::Put),
            _ => None,
        }
    }
}

impl std::str::FromStr for OptionType {
    type Err = ();

    /// Enables `"CE".parse::<OptionType>()`.
    ///
    /// Previously an inherent `from_str` shadowed this trait method, so
    /// `.parse()` did not work while `OptionType::from_str` did -- the kind of
    /// asymmetry that reads as a bug at the call site.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_code(s).ok_or(())
    }
}

/// Configuration for a single leg of a spread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegConfig {
    /// Option type (Call or Put).
    pub option_type: OptionType,
    /// Strike price.
    pub strike: f64,
    /// Position quantity (+1 long, -1 short).
    pub quantity: i32,
    /// Lot size for the option.
    pub lot_size: usize,
}

impl LegConfig {
    pub fn new(option_type: OptionType, strike: f64, quantity: i32, lot_size: usize) -> Self {
        Self { option_type, strike, quantity, lot_size }
    }

    /// Check if this is a long position.
    pub fn is_long(&self) -> bool {
        self.quantity > 0
    }

    /// Check if this is a short position.
    pub fn is_short(&self) -> bool {
        self.quantity < 0
    }
}

/// Configuration for spread backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadConfig {
    /// Base backtest configuration.
    pub base: BacktestConfig,
    /// Spread type.
    pub spread_type: SpreadType,
    /// Leg configurations.
    pub leg_configs: Vec<LegConfig>,
    /// Maximum loss threshold (optional, for early exit).
    pub max_loss: Option<f64>,
    /// Target profit threshold (optional, for early exit).
    pub target_profit: Option<f64>,
    /// Per-leg expiry timestamps in nanoseconds (optional, for settlement logic).
    /// When provided, positions are force-closed at or after the earliest leg expiry.
    pub leg_expiry_timestamps: Option<Vec<i64>>,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            spread_type: SpreadType::Custom,
            leg_configs: Vec::new(),
            max_loss: None,
            target_profit: None,
            leg_expiry_timestamps: None,
        }
    }
}

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
}

impl LegPosition {
    fn new(config: LegConfig, entry_premium: f64, entry_idx: usize) -> Self {
        Self { entry_premium, entry_idx, current_premium: entry_premium, config }
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
        }
    }

    /// Calculate total unrealized P&L across all legs.
    fn total_unrealized_pnl(&self) -> f64 {
        self.legs.iter().map(|leg| leg.unrealized_pnl()).sum()
    }

    /// Update leg premiums.
    fn update_premiums(&mut self, leg_premiums: &[f64]) {
        for (leg, &premium) in self.legs.iter_mut().zip(leg_premiums.iter()) {
            leg.current_premium = premium;
        }
    }

    /// Close the position and return P&L.
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

            // Calculate unrealized P&L for exit checks
            let unrealized_pnl = position.as_ref().map(|p| p.total_unrealized_pnl()).unwrap_or(0.0);

            // Check if any leg has expired at this bar
            let is_expiry =
                position.is_some()
                    && self.config.leg_expiry_timestamps.as_ref().is_some_and(|expiries| {
                        expiries.iter().any(|&exp_ts| timestamps[i] >= exp_ts)
                    });

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

                    let (exit_fees, fee_breakdown) = self.calculate_exit(&pos, exit_reason);
                    let entry_fees = pos.entry_fees;
                    let fees = entry_fees + exit_fees;
                    // Entry costs already left `cash` when the position
                    // opened, so only the exit side is charged here.
                    let net_pnl = pnl - fees;

                    cash += pnl - exit_fees;

                    // Record trade
                    trade_id += 1;

                    let entry_premium = pos.entry_net_premium;
                    let exit_premium: f64 = current_premiums
                        .iter()
                        .zip(self.config.leg_configs.iter())
                        .map(|(&p, cfg)| p * cfg.quantity as f64 * cfg.lot_size as f64)
                        .sum();

                    trades.push(Trade {
                        id: trade_id,
                        symbol: "SPREAD".to_string(),
                        entry_idx: pos.entry_idx,
                        exit_idx: i,
                        entry_price: entry_premium,
                        exit_price: exit_premium,
                        size: 1.0,
                        direction: Direction::Long, // Spreads are treated as "long spread"
                        pnl: net_pnl,
                        return_pct: if entry_premium.abs() > 0.0 {
                            net_pnl / entry_premium.abs() * 100.0
                        } else {
                            0.0
                        },
                        entry_time: pos.entry_time,
                        exit_time: timestamps[i],
                        fees,
                        entry_fees,
                        exit_fees,
                        fee_breakdown,
                        exit_reason,
                    });

                    metrics.record_fees(fees);
                    metrics.record_trade(
                        net_pnl,
                        net_pnl / entry_premium.abs() * 100.0,
                        i - pos.entry_idx,
                    );
                }
            }

            // Check for entry signals (don't re-enter after all legs expired)
            let all_expired = self
                .config
                .leg_expiry_timestamps
                .as_ref()
                .is_some_and(|expiries| expiries.iter().all(|&exp_ts| timestamps[i] >= exp_ts));
            if position.is_none() && entries[i] && !all_expired && !squareoff[i] {
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
            let current_premiums: Vec<f64> = legs_premiums.iter().map(|p| p[last]).collect();
            let entry_premium = pos.entry_net_premium;
            let exit_premium: f64 = current_premiums
                .iter()
                .zip(self.config.leg_configs.iter())
                .map(|(&p, cfg)| p * cfg.quantity as f64 * cfg.lot_size as f64)
                .sum();

            let pnl = pos.close();
            let (exit_fees, fee_breakdown) = self.calculate_exit(&pos, ExitReason::EndOfData);
            let entry_fees = pos.entry_fees;
            let fees = entry_fees + exit_fees;
            let net_pnl = pnl - fees;
            cash += pnl - exit_fees;

            trade_id += 1;
            trades.push(Trade {
                id: trade_id,
                symbol: "SPREAD".to_string(),
                entry_idx: pos.entry_idx,
                exit_idx: last,
                entry_price: entry_premium,
                exit_price: exit_premium,
                size: 1.0,
                direction: Direction::Long,
                pnl: net_pnl,
                return_pct: if entry_premium.abs() > 0.0 {
                    net_pnl / entry_premium.abs() * 100.0
                } else {
                    0.0
                },
                entry_time: pos.entry_time,
                exit_time: timestamps[last],
                fees,
                entry_fees,
                exit_fees,
                fee_breakdown,
                exit_reason: ExitReason::EndOfData,
            });

            if entry_premium.abs() > 0.0 {
                metrics.record_fees(fees);
                metrics.record_trade(
                    net_pnl,
                    net_pnl / entry_premium.abs() * 100.0,
                    last - pos.entry_idx,
                );
            }
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

/// Convenience function to create a straddle spread config.
pub fn create_straddle_config(
    base: BacktestConfig,
    strike: f64,
    lot_size: usize,
    short: bool,
) -> SpreadConfig {
    let quantity = if short { -1 } else { 1 };
    SpreadConfig {
        base,
        spread_type: SpreadType::Straddle,
        leg_configs: vec![
            LegConfig::new(OptionType::Call, strike, quantity, lot_size),
            LegConfig::new(OptionType::Put, strike, quantity, lot_size),
        ],
        ..Default::default()
    }
}

/// Convenience function to create a strangle spread config.
pub fn create_strangle_config(
    base: BacktestConfig,
    call_strike: f64,
    put_strike: f64,
    lot_size: usize,
    short: bool,
) -> SpreadConfig {
    let quantity = if short { -1 } else { 1 };
    SpreadConfig {
        base,
        spread_type: SpreadType::Strangle,
        leg_configs: vec![
            LegConfig::new(OptionType::Call, call_strike, quantity, lot_size),
            LegConfig::new(OptionType::Put, put_strike, quantity, lot_size),
        ],
        ..Default::default()
    }
}

/// Convenience function to create an iron condor spread config.
pub fn create_iron_condor_config(
    base: BacktestConfig,
    short_put_strike: f64,
    long_put_strike: f64,
    short_call_strike: f64,
    long_call_strike: f64,
    lot_size: usize,
) -> SpreadConfig {
    SpreadConfig {
        base,
        spread_type: SpreadType::IronCondor,
        leg_configs: vec![
            LegConfig::new(OptionType::Put, short_put_strike, -1, lot_size),
            LegConfig::new(OptionType::Put, long_put_strike, 1, lot_size),
            LegConfig::new(OptionType::Call, short_call_strike, -1, lot_size),
            LegConfig::new(OptionType::Call, long_call_strike, 1, lot_size),
        ],
        ..Default::default()
    }
}

/// Convenience function to create a vertical spread config.
pub fn create_vertical_spread_config(
    base: BacktestConfig,
    option_type: OptionType,
    long_strike: f64,
    short_strike: f64,
    lot_size: usize,
) -> SpreadConfig {
    let spread_type = match option_type {
        OptionType::Call => SpreadType::VerticalCall,
        OptionType::Put => SpreadType::VerticalPut,
    };

    SpreadConfig {
        base,
        spread_type,
        leg_configs: vec![
            LegConfig::new(option_type, long_strike, 1, lot_size),
            LegConfig::new(option_type, short_strike, -1, lot_size),
        ],
        ..Default::default()
    }
}
#[cfg(test)]
#[path = "spreads_tests.rs"]
mod tests;
