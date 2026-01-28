//! Options strategy backtest implementation.
//!
//! Supports dynamic strike selection and options-specific position sizing.

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, ExitReason, OhlcvData, Trade,
};
use crate::execution::FeeModel;
use crate::metrics::streaming::StreamingMetrics;

/// Options position type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

/// Strike selection mode.
#[derive(Debug, Clone, Copy)]
pub enum StrikeSelection {
    /// At-the-money (closest to spot).
    Atm,
    /// In-the-money by N strikes.
    Itm(usize),
    /// Out-of-the-money by N strikes.
    Otm(usize),
    /// Fixed strike offset from ATM in percentage.
    PercentOffset(f64),
    /// Delta-based selection.
    Delta(f64),
}

impl Default for StrikeSelection {
    fn default() -> Self {
        StrikeSelection::Atm
    }
}

/// Position size type for options.
#[derive(Debug, Clone, Copy)]
pub enum SizeType {
    /// Fixed number of contracts.
    Contracts(usize),
    /// Percentage of capital.
    Percent(f64),
    /// Fixed notional value.
    Notional(f64),
    /// Risk-based (percentage of capital at risk).
    RiskPercent(f64),
}

impl Default for SizeType {
    fn default() -> Self {
        SizeType::Percent(1.0)
    }
}

/// Options backtest configuration.
#[derive(Debug, Clone)]
pub struct OptionsConfig {
    /// Base backtest config.
    pub base: BacktestConfig,
    /// Option type (call/put).
    pub option_type: OptionType,
    /// Strike selection mode.
    pub strike_selection: StrikeSelection,
    /// Position size type.
    pub size_type: SizeType,
    /// Lot size (contracts per lot).
    pub lot_size: usize,
    /// Strike interval.
    pub strike_interval: f64,
    /// Days to expiry preference.
    pub target_dte: Option<usize>,
}

impl Default for OptionsConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            option_type: OptionType::Call,
            strike_selection: StrikeSelection::Atm,
            size_type: SizeType::Percent(1.0),
            lot_size: 1,
            strike_interval: 50.0,
            target_dte: None,
        }
    }
}

/// Options backtest runner.
#[derive(Debug)]
pub struct OptionsBacktest {
    /// Configuration.
    config: OptionsConfig,
    /// Fee model.
    fee_model: FeeModel,
}

impl OptionsBacktest {
    /// Create a new options backtest.
    pub fn new(config: OptionsConfig) -> Self {
        Self {
            fee_model: FeeModel::percentage(config.base.fees),
            config,
        }
    }

    /// Run options backtest.
    ///
    /// # Arguments
    /// * `spot_ohlcv` - Spot/underlying OHLCV data
    /// * `option_prices` - Option premium prices (parallel array)
    /// * `signals` - Trading signals
    ///
    /// # Returns
    /// Backtest result
    pub fn run(
        &self,
        spot_ohlcv: &OhlcvData,
        option_prices: &[f64],
        signals: &CompiledSignals,
    ) -> BacktestResult {
        let n = spot_ohlcv.len();
        assert_eq!(n, option_prices.len());
        assert_eq!(n, signals.len());

        // Clean signals
        let processor = crate::signals::processor::SignalProcessor::new();
        let (entries, exits) = processor.clean_signals(&signals.entries, &signals.exits);

        // Initialize state
        let mut cash = self.config.base.initial_capital;
        let mut position: Option<OptionsPosition> = None;
        let mut equity_curve = vec![cash; n];
        let mut drawdown_curve = vec![0.0; n];
        let mut returns = vec![0.0; n];
        let mut trades: Vec<Trade> = Vec::new();
        let mut streaming = StreamingMetrics::new();
        let mut peak_equity = cash;
        let mut trade_counter = 0u64;

        // Main simulation loop
        for i in 0..n {
            let spot_price = spot_ohlcv.close[i];
            let option_price = option_prices[i];

            // Check for exit
            if exits[i] {
                if let Some(pos) = position.take() {
                    let exit_price = option_price;
                    let fees = self.fee_model.calculate(
                        exit_price,
                        pos.contracts as f64,
                        signals.direction,
                    );

                    let pnl = self.calculate_pnl(&pos, exit_price) - fees;
                    let cost_basis =
                        pos.entry_price * pos.contracts as f64 * self.config.lot_size as f64;
                    let return_pct = if cost_basis > 0.0 {
                        pnl / cost_basis * 100.0
                    } else {
                        0.0
                    };

                    cash += exit_price * pos.contracts as f64 * self.config.lot_size as f64 - fees;

                    trades.push(Trade {
                        id: trade_counter,
                        symbol: signals.symbol.clone(),
                        entry_idx: pos.entry_idx,
                        exit_idx: i,
                        entry_price: pos.entry_price,
                        exit_price,
                        size: pos.contracts as f64,
                        direction: signals.direction,
                        pnl,
                        return_pct,
                        entry_time: spot_ohlcv.timestamps[pos.entry_idx],
                        exit_time: spot_ohlcv.timestamps[i],
                        fees,
                        exit_reason: ExitReason::Signal,
                    });

                    trade_counter += 1;
                    streaming.update(return_pct / 100.0);
                }
            }

            // Check for entry
            if entries[i] && position.is_none() {
                let strike = self.select_strike(spot_price);
                let contracts = self.calculate_contracts(option_price, cash);

                if contracts > 0 {
                    let entry_cost = option_price * contracts as f64 * self.config.lot_size as f64;
                    let fees =
                        self.fee_model
                            .calculate(option_price, contracts as f64, signals.direction);

                    cash -= entry_cost + fees;

                    position = Some(OptionsPosition {
                        entry_idx: i,
                        entry_price: option_price,
                        strike,
                        contracts,
                        option_type: self.config.option_type,
                    });
                }
            }

            // Update equity
            let position_value = if let Some(ref pos) = position {
                option_price * pos.contracts as f64 * self.config.lot_size as f64
            } else {
                0.0
            };
            let equity = cash + position_value;
            equity_curve[i] = equity;

            // Update drawdown
            if equity > peak_equity {
                peak_equity = equity;
            }
            drawdown_curve[i] = (peak_equity - equity) / peak_equity * 100.0;

            // Calculate return
            if i > 0 {
                returns[i] = (equity - equity_curve[i - 1]) / equity_curve[i - 1];
            }
        }

        // Close any remaining position
        if let Some(pos) = position.take() {
            let last_idx = n - 1;
            let exit_price = option_prices[last_idx];
            let fees =
                self.fee_model
                    .calculate(exit_price, pos.contracts as f64, signals.direction);

            let pnl = self.calculate_pnl(&pos, exit_price) - fees;
            let cost_basis = pos.entry_price * pos.contracts as f64 * self.config.lot_size as f64;
            let return_pct = if cost_basis > 0.0 {
                pnl / cost_basis * 100.0
            } else {
                0.0
            };

            trades.push(Trade {
                id: trade_counter,
                symbol: signals.symbol.clone(),
                entry_idx: pos.entry_idx,
                exit_idx: last_idx,
                entry_price: pos.entry_price,
                exit_price,
                size: pos.contracts as f64,
                direction: signals.direction,
                pnl,
                return_pct,
                entry_time: spot_ohlcv.timestamps[pos.entry_idx],
                exit_time: spot_ohlcv.timestamps[last_idx],
                fees,
                exit_reason: ExitReason::EndOfData,
            });

            streaming.update(return_pct / 100.0);
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&equity_curve, &drawdown_curve, &trades, &streaming);

        BacktestResult::new(metrics, equity_curve, drawdown_curve, trades, returns)
    }

    /// Select strike price based on configuration.
    fn select_strike(&self, spot_price: f64) -> f64 {
        let interval = self.config.strike_interval;
        let atm_strike = (spot_price / interval).round() * interval;

        match self.config.strike_selection {
            StrikeSelection::Atm => atm_strike,
            StrikeSelection::Itm(n) => match self.config.option_type {
                OptionType::Call => atm_strike - (n as f64 * interval),
                OptionType::Put => atm_strike + (n as f64 * interval),
            },
            StrikeSelection::Otm(n) => match self.config.option_type {
                OptionType::Call => atm_strike + (n as f64 * interval),
                OptionType::Put => atm_strike - (n as f64 * interval),
            },
            StrikeSelection::PercentOffset(pct) => {
                let offset = spot_price * pct;
                match self.config.option_type {
                    OptionType::Call => atm_strike + offset,
                    OptionType::Put => atm_strike - offset,
                }
            }
            StrikeSelection::Delta(_) => atm_strike, // Simplified - would need options chain
        }
    }

    /// Calculate number of contracts based on size type.
    fn calculate_contracts(&self, option_price: f64, available_capital: f64) -> usize {
        if option_price <= 0.0 {
            return 0;
        }

        let contract_cost = option_price * self.config.lot_size as f64;

        match self.config.size_type {
            SizeType::Contracts(n) => n,
            SizeType::Percent(pct) => {
                let allocation = available_capital * pct;
                (allocation / contract_cost) as usize
            }
            SizeType::Notional(value) => (value / contract_cost) as usize,
            SizeType::RiskPercent(pct) => {
                // Max loss is the premium paid
                let risk_amount = available_capital * pct;
                (risk_amount / contract_cost) as usize
            }
        }
    }

    /// Calculate P&L for a position.
    fn calculate_pnl(&self, position: &OptionsPosition, current_price: f64) -> f64 {
        let multiplier = self.config.lot_size as f64;
        (current_price - position.entry_price) * position.contracts as f64 * multiplier
    }

    /// Calculate metrics.
    fn calculate_metrics(
        &self,
        equity_curve: &[f64],
        drawdown_curve: &[f64],
        trades: &[Trade],
        streaming: &StreamingMetrics,
    ) -> BacktestMetrics {
        let start_value = self.config.base.initial_capital;
        let end_value = *equity_curve.last().unwrap_or(&start_value);

        let total_return_pct = (end_value - start_value) / start_value * 100.0;
        let max_drawdown_pct = drawdown_curve.iter().fold(0.0f64, |a, &b| a.max(b));

        let total_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = trades.iter().filter(|t| t.pnl < 0.0).count();

        let win_rate_pct = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };

        let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestMetrics {
            total_return_pct,
            sharpe_ratio: streaming.sharpe_ratio(252.0),
            sortino_ratio: streaming.sortino_ratio(252.0),
            calmar_ratio: if max_drawdown_pct > 0.0 {
                total_return_pct / max_drawdown_pct
            } else {
                0.0
            },
            max_drawdown_pct,
            win_rate_pct,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            start_value,
            end_value,
            ..Default::default()
        }
    }
}

/// Internal options position state.
#[derive(Debug, Clone)]
struct OptionsPosition {
    entry_idx: usize,
    entry_price: f64,
    #[allow(dead_code)]
    strike: f64,
    contracts: usize,
    #[allow(dead_code)]
    option_type: OptionType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strike_selection_atm() {
        let config = OptionsConfig {
            strike_interval: 50.0,
            strike_selection: StrikeSelection::Atm,
            ..Default::default()
        };
        let backtest = OptionsBacktest::new(config);

        // Spot at 17834, ATM should be 17850
        let strike = backtest.select_strike(17834.0);
        assert!((strike - 17850.0).abs() < 1e-10);
    }

    #[test]
    fn test_strike_selection_otm() {
        let config = OptionsConfig {
            strike_interval: 50.0,
            strike_selection: StrikeSelection::Otm(2),
            option_type: OptionType::Call,
            ..Default::default()
        };
        let backtest = OptionsBacktest::new(config);

        // Spot at 17834, ATM=17850, OTM 2 strikes = 17950
        let strike = backtest.select_strike(17834.0);
        assert!((strike - 17950.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_sizing_percent() {
        let config = OptionsConfig {
            size_type: SizeType::Percent(0.5),
            lot_size: 50,
            ..Default::default()
        };
        let backtest = OptionsBacktest::new(config);

        // 50% of 100000 = 50000, option at 100 * lot 50 = 5000 per contract
        let contracts = backtest.calculate_contracts(100.0, 100_000.0);
        assert_eq!(contracts, 10);
    }
}
