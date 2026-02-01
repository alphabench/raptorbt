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
    Custom,
}

/// Option type for a leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "CE" | "CALL" | "C" => Some(OptionType::Call),
            "PE" | "PUT" | "P" => Some(OptionType::Put),
            _ => None,
        }
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
    /// Whether to close at end of day.
    pub close_at_eod: bool,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            spread_type: SpreadType::Custom,
            leg_configs: Vec::new(),
            max_loss: None,
            target_profit: None,
            close_at_eod: false,
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
        // For short positions: profit when premium decreases
        // For long positions: profit when premium increases
        let premium_change = self.current_premium - self.entry_premium;
        let quantity = self.config.quantity as f64;
        let lot_size = self.config.lot_size as f64;
        -quantity * premium_change * lot_size
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
}

impl SpreadPosition {
    fn new(legs: Vec<LegPosition>, entry_idx: usize, entry_time: i64) -> Self {
        let entry_net_premium: f64 = legs
            .iter()
            .map(|leg| leg.entry_premium * leg.config.quantity as f64 * leg.config.lot_size as f64)
            .sum();

        Self { legs, entry_idx, entry_net_premium, entry_time, is_open: true }
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
}

impl SpreadBacktest {
    /// Create a new spread backtest.
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

            // Check for exit signals or conditions
            let should_exit = position.is_some()
                && (exits[i]
                    || self.check_max_loss(&position, unrealized_pnl)
                    || self.check_target_profit(&position, unrealized_pnl));

            if should_exit {
                if let Some(mut pos) = position.take() {
                    let pnl = pos.close();
                    let fees = self.calculate_fees(&pos);
                    let net_pnl = pnl - fees;

                    cash += net_pnl;

                    // Record trade
                    trade_id += 1;
                    let exit_reason = if exits[i] {
                        ExitReason::Signal
                    } else if self.check_max_loss(&Some(pos.clone()), pnl) {
                        ExitReason::StopLoss
                    } else {
                        ExitReason::TakeProfit
                    };

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
                        exit_reason,
                    });

                    metrics.record_trade(
                        net_pnl,
                        net_pnl / entry_premium.abs() * 100.0,
                        i - pos.entry_idx,
                    );
                }
            }

            // Check for entry signals
            if position.is_none() && entries[i] {
                let legs: Vec<LegPosition> = self
                    .config
                    .leg_configs
                    .iter()
                    .zip(current_premiums.iter())
                    .map(|(cfg, &premium)| LegPosition::new(cfg.clone(), premium, i))
                    .collect();

                let new_position = SpreadPosition::new(legs, i, timestamps[i]);

                // Calculate entry fees
                let entry_fees = self.calculate_entry_fees(&new_position);
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

        // Close any remaining open position at end
        if let Some(mut pos) = position.take() {
            let pnl = pos.close();
            let fees = self.calculate_fees(&pos);
            cash += pnl - fees;
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

    /// Calculate entry fees for a position.
    fn calculate_entry_fees(&self, position: &SpreadPosition) -> f64 {
        let total_premium: f64 = position
            .legs
            .iter()
            .map(|leg| leg.entry_premium.abs() * leg.config.lot_size as f64)
            .sum();
        total_premium * self.config.base.fees
    }

    /// Calculate exit fees for a position.
    fn calculate_fees(&self, position: &SpreadPosition) -> f64 {
        let total_premium: f64 = position
            .legs
            .iter()
            .map(|leg| leg.current_premium.abs() * leg.config.lot_size as f64)
            .sum();
        total_premium * self.config.base.fees * 2.0 // Entry + Exit
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
mod tests {
    use super::*;
    use crate::core::types::StopConfig;
    use crate::core::types::TargetConfig;

    fn sample_data() -> (Vec<i64>, Vec<f64>, Vec<Vec<f64>>, Vec<bool>, Vec<bool>) {
        let n = 20;
        let timestamps: Vec<i64> = (0..n as i64).collect();
        let underlying: Vec<f64> = (100..120).map(|x| x as f64).collect();

        // Call and Put premiums
        let call_premiums: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64 * 0.2)).collect();
        let put_premiums: Vec<f64> = (0..n).map(|i| 5.0 - (i as f64 * 0.1)).collect();

        let legs_premiums = vec![call_premiums, put_premiums];

        let entries = vec![
            false, true, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false,
        ];
        let exits = vec![
            false, false, false, false, false, false, false, false, false, true, false, false,
            false, false, false, false, false, false, false, false,
        ];

        (timestamps, underlying, legs_premiums, entries, exits)
    }

    #[test]
    fn test_straddle_backtest() {
        let base_config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.001,
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
        };

        let config = create_straddle_config(base_config, 100.0, 50, true);
        let backtest = SpreadBacktest::new(config);

        let (timestamps, underlying, legs_premiums, entries, exits) = sample_data();

        let result = backtest.run(&timestamps, &underlying, &legs_premiums, &entries, &exits);

        assert_eq!(result.trades.len(), 1);
        assert!(result.equity_curve.len() == timestamps.len());
    }

    #[test]
    fn test_iron_condor_backtest() {
        let base_config = BacktestConfig::default();

        let config = create_iron_condor_config(
            base_config,
            95.0,  // short put
            90.0,  // long put
            105.0, // short call
            110.0, // long call
            50,
        );

        let backtest = SpreadBacktest::new(config);

        let n = 20;
        let timestamps: Vec<i64> = (0..n as i64).collect();
        let underlying: Vec<f64> = vec![100.0; n];

        // Four legs: short put, long put, short call, long call
        let legs_premiums = vec![
            vec![3.0; n], // short put
            vec![1.5; n], // long put
            vec![3.0; n], // short call
            vec![1.5; n], // long call
        ];

        let mut entries = vec![false; n];
        entries[1] = true;

        let mut exits = vec![false; n];
        exits[15] = true;

        let result = backtest.run(&timestamps, &underlying, &legs_premiums, &entries, &exits);

        assert_eq!(result.trades.len(), 1);
    }
}
