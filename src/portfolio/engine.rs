//! Event-driven portfolio simulation engine.

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, Direction, ExitReason,
    InstrumentConfig, OhlcvData, Price, StopConfig, TargetConfig, Trade,
};
use crate::execution::{FeeModel, FillPrice, SlippageModel};
use crate::indicators::volatility::atr;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::position::PositionManager;
use crate::signals::processor::SignalProcessor;

/// Portfolio simulation engine.
///
/// Single-pass O(n) algorithm for simulating portfolio performance.
#[derive(Debug)]
pub struct PortfolioEngine {
    /// Configuration.
    pub config: BacktestConfig,
    /// Fee model.
    pub fee_model: FeeModel,
    /// Slippage model.
    pub slippage_model: SlippageModel,
    /// Fill price model.
    pub fill_price: FillPrice,
    /// Signal processor.
    pub signal_processor: SignalProcessor,
}

impl Default for PortfolioEngine {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

impl PortfolioEngine {
    /// Create a new portfolio engine with the given configuration.
    pub fn new(config: BacktestConfig) -> Self {
        let fee_model = FeeModel::percentage(config.fees);
        let fill_price = if config.upon_bar_close { FillPrice::Close } else { FillPrice::Open };

        Self {
            config,
            fee_model,
            slippage_model: SlippageModel::None,
            fill_price,
            signal_processor: SignalProcessor::new(),
        }
    }

    /// Set fee model.
    pub fn with_fee_model(mut self, fee_model: FeeModel) -> Self {
        self.fee_model = fee_model;
        self
    }

    /// Set slippage model.
    pub fn with_slippage_model(mut self, slippage_model: SlippageModel) -> Self {
        self.slippage_model = slippage_model;
        self
    }

    /// Run backtest on single instrument.
    ///
    /// # Arguments
    /// * `ohlcv` - OHLCV data
    /// * `signals` - Compiled trading signals
    ///
    /// # Returns
    /// Backtest result
    pub fn run_single(&self, ohlcv: &OhlcvData, signals: &CompiledSignals) -> BacktestResult {
        self.run_single_with_instrument_config(ohlcv, signals, None)
    }

    /// Run backtest on single instrument with optional per-instrument configuration.
    ///
    /// # Arguments
    /// * `ohlcv` - OHLCV data
    /// * `signals` - Compiled trading signals
    /// * `inst_config` - Optional per-instrument config (lot_size, capital cap, stop/target overrides)
    ///
    /// # Returns
    /// Backtest result
    pub fn run_single_with_instrument_config(
        &self,
        ohlcv: &OhlcvData,
        signals: &CompiledSignals,
        inst_config: Option<&InstrumentConfig>,
    ) -> BacktestResult {
        let n = ohlcv.len();
        assert_eq!(n, signals.len(), "OHLCV and signals must have same length");

        // Clean signals
        let (entries, exits) =
            self.signal_processor.clean_signals(&signals.entries, &signals.exits);

        // Initialize state
        let mut position = PositionManager::new(signals.symbol.clone());
        let mut cash = self.config.initial_capital;
        let mut equity_curve = vec![cash; n];
        let mut drawdown_curve = vec![0.0; n];
        let mut returns = vec![0.0; n];
        let mut trades: Vec<Trade> = Vec::new();
        let mut streaming = StreamingMetrics::new();
        let mut peak_equity = cash;

        // Determine effective stop/target configs (per-instrument overrides take precedence)
        let effective_stop =
            inst_config.and_then(|ic| ic.stop.as_ref()).unwrap_or(&self.config.stop);
        let effective_target =
            inst_config.and_then(|ic| ic.target.as_ref()).unwrap_or(&self.config.target);

        // Pre-calculate ATR for ATR-based stops
        let atr_values = if matches!(effective_stop, StopConfig::Atr { .. })
            || matches!(effective_target, TargetConfig::Atr { .. })
        {
            let period = match effective_stop {
                StopConfig::Atr { period, .. } => *period,
                _ => match effective_target {
                    TargetConfig::Atr { period, .. } => *period,
                    _ => 14,
                },
            };
            atr(&ohlcv.high, &ohlcv.low, &ohlcv.close, period).unwrap_or_else(|_| vec![0.0; n])
        } else {
            vec![0.0; n]
        };

        // Main simulation loop
        for i in 0..n {
            let close = ohlcv.close[i];
            let high = ohlcv.high[i];
            let low = ohlcv.low[i];
            let timestamp = ohlcv.timestamps[i];

            // Update position price tracking
            position.update_price(high, low);

            // Check for exits first (stops and signals)
            if position.is_in_position() {
                let mut exit_reason: Option<ExitReason> = None;
                let mut exit_price = close;

                // Check stop-loss
                if position.is_stop_hit(low, high) {
                    exit_reason = Some(ExitReason::StopLoss);
                    exit_price = position.position.stop_price.unwrap();

                    // Adjust for gap through stop
                    match position.position.direction {
                        Direction::Long => {
                            if ohlcv.open[i] < exit_price {
                                exit_price = ohlcv.open[i];
                            }
                        }
                        Direction::Short => {
                            if ohlcv.open[i] > exit_price {
                                exit_price = ohlcv.open[i];
                            }
                        }
                    }
                }

                // Check take-profit
                if exit_reason.is_none() && position.is_target_hit(low, high) {
                    exit_reason = Some(ExitReason::TakeProfit);
                    exit_price = position.position.target_price.unwrap();
                }

                // Check exit signal
                if exit_reason.is_none() && exits[i] {
                    exit_reason = Some(ExitReason::Signal);
                    exit_price = self.get_fill_price(ohlcv, i, signals.direction, false);
                }

                // Execute exit
                if let Some(reason) = exit_reason {
                    // Apply slippage
                    exit_price = self.slippage_model.apply(
                        exit_price,
                        position.position.direction,
                        false,
                        Some(ohlcv.volume[i]),
                    );

                    // Calculate fees
                    let fees = self.fee_model.calculate(
                        exit_price,
                        position.position.size,
                        position.position.direction,
                    );

                    // Close position
                    if let Some(trade) = position.close_position(
                        i,
                        timestamp,
                        exit_price,
                        ohlcv.timestamps[position.position.entry_idx],
                        reason,
                        fees,
                    ) {
                        // Update cash
                        let exit_value = exit_price * trade.size;
                        cash += exit_value - fees;

                        // Track return for this trade
                        streaming.update(trade.return_pct / 100.0);

                        trades.push(trade);
                    }
                }

                // Update trailing stop if position still open
                if position.is_in_position() {
                    if let StopConfig::Trailing { percent } = effective_stop {
                        position.update_trailing_stop(*percent);
                    }
                }
            }

            // Check for entries
            if !position.is_in_position() && entries[i] {
                let entry_price = self.get_fill_price(ohlcv, i, signals.direction, true);

                // Apply slippage
                let adjusted_price = self.slippage_model.apply(
                    entry_price,
                    signals.direction,
                    true,
                    Some(ohlcv.volume[i]),
                );

                // Calculate position size
                // Use per-instrument capital if set, capped at available cash
                let available = inst_config
                    .and_then(|ic| ic.alloted_capital)
                    .map(|cap| cap.min(cash))
                    .unwrap_or(cash);

                // VectorBT formula: size = cash / (price * (1 + fees))
                // This ensures the position value plus entry fee equals available cash
                let fee_rate = self.config.fees;
                let raw_size = if let Some(ref sizes) = signals.position_sizes {
                    sizes[i] * available / (adjusted_price * (1.0 + fee_rate))
                } else {
                    available / (adjusted_price * (1.0 + fee_rate))
                };

                // Round to lot_size
                let size = inst_config.map(|ic| ic.round_to_lot(raw_size)).unwrap_or(raw_size);

                if size > 0.0 {
                    // Calculate entry fees
                    let entry_fees =
                        self.fee_model.calculate(adjusted_price, size, signals.direction);

                    // Calculate stop and target prices
                    let (stop_price, target_price) = self.calculate_stop_target_with_config(
                        adjusted_price,
                        signals.direction,
                        &atr_values,
                        i,
                        effective_stop,
                        effective_target,
                    );

                    // Open position (passing entry_fees for trade PnL tracking)
                    position.open_position(
                        i,
                        timestamp,
                        adjusted_price,
                        size,
                        signals.direction,
                        stop_price,
                        target_price,
                        entry_fees,
                    );

                    // Deduct cost
                    cash -= adjusted_price * size + entry_fees;
                }
            }

            // Calculate equity
            let position_value =
                if position.is_in_position() { close * position.position.size } else { 0.0 };
            let equity = cash + position_value;
            equity_curve[i] = equity;

            // Calculate drawdown
            if equity > peak_equity {
                peak_equity = equity;
            }
            drawdown_curve[i] = (peak_equity - equity) / peak_equity * 100.0;

            // Calculate return
            if i > 0 {
                returns[i] = (equity - equity_curve[i - 1]) / equity_curve[i - 1];
            }
        }

        // Mark any open position at end of data (no exit fees, matching VectorBT behavior)
        if position.is_in_position() {
            let last_idx = n - 1;
            let exit_price = ohlcv.close[last_idx];
            // No exit fees for EndOfData - position is marked-to-market but not actually closed
            // This matches VectorBT's behavior for "Open" trades
            let exit_fees = 0.0;

            if let Some(trade) = position.close_position(
                last_idx,
                ohlcv.timestamps[last_idx],
                exit_price,
                ohlcv.timestamps[position.position.entry_idx],
                ExitReason::EndOfData,
                exit_fees,
            ) {
                streaming.update(trade.return_pct / 100.0);
                trades.push(trade);
            }
        }

        // Calculate final metrics
        let metrics =
            self.calculate_metrics(&equity_curve, &drawdown_curve, &returns, &trades, &streaming);

        BacktestResult::new(metrics, equity_curve, drawdown_curve, trades, returns)
    }

    /// Get fill price based on model.
    fn get_fill_price(
        &self,
        ohlcv: &OhlcvData,
        idx: usize,
        direction: Direction,
        is_entry: bool,
    ) -> Price {
        self.fill_price.get_price_from_arrays(
            ohlcv.open[idx],
            ohlcv.high[idx],
            ohlcv.low[idx],
            ohlcv.close[idx],
            direction,
            is_entry,
        )
    }

    /// Calculate stop and target prices using the global config.
    #[allow(dead_code)]
    fn calculate_stop_target(
        &self,
        entry_price: Price,
        direction: Direction,
        atr_values: &[f64],
        idx: usize,
    ) -> (Option<Price>, Option<Price>) {
        self.calculate_stop_target_with_config(
            entry_price,
            direction,
            atr_values,
            idx,
            &self.config.stop,
            &self.config.target,
        )
    }

    /// Calculate stop and target prices with explicit stop/target configs.
    fn calculate_stop_target_with_config(
        &self,
        entry_price: Price,
        direction: Direction,
        atr_values: &[f64],
        idx: usize,
        stop_config: &StopConfig,
        target_config: &TargetConfig,
    ) -> (Option<Price>, Option<Price>) {
        let multiplier = direction.multiplier();

        // Calculate stop price
        let stop_price = match stop_config {
            StopConfig::None => None,
            StopConfig::Fixed { percent } => Some(entry_price * (1.0 - multiplier * percent)),
            StopConfig::Atr { multiplier: m, .. } => {
                let atr = atr_values.get(idx).copied().unwrap_or(0.0);
                if atr > 0.0 {
                    Some(entry_price - multiplier * m * atr)
                } else {
                    None
                }
            }
            StopConfig::Trailing { percent } => Some(entry_price * (1.0 - multiplier * percent)),
        };

        // Calculate target price
        let target_price = match target_config {
            TargetConfig::None => None,
            TargetConfig::Fixed { percent } => Some(entry_price * (1.0 + multiplier * percent)),
            TargetConfig::Atr { multiplier: m, .. } => {
                let atr = atr_values.get(idx).copied().unwrap_or(0.0);
                if atr > 0.0 {
                    Some(entry_price + multiplier * m * atr)
                } else {
                    None
                }
            }
            TargetConfig::RiskReward { ratio } => {
                if let Some(stop) = stop_price {
                    let risk = (entry_price - stop).abs();
                    Some(entry_price + multiplier * risk * ratio)
                } else {
                    None
                }
            }
        };

        (stop_price, target_price)
    }

    /// Calculate backtest metrics.
    fn calculate_metrics(
        &self,
        equity_curve: &[f64],
        drawdown_curve: &[f64],
        returns: &[f64],
        trades: &[Trade],
        _streaming: &StreamingMetrics,
    ) -> BacktestMetrics {
        let start_value = self.config.initial_capital;
        let end_value = *equity_curve.last().unwrap_or(&start_value);

        let total_return_pct = (end_value - start_value) / start_value * 100.0;
        let max_drawdown_pct = drawdown_curve.iter().fold(0.0f64, |a, &b| a.max(b));

        // Calculate max drawdown duration
        let max_drawdown_duration = self.calculate_max_drawdown_duration(drawdown_curve);

        // Trade statistics
        let total_trades = trades.len();

        // Separate closed vs open trades (EndOfData means still open)
        let total_open_trades =
            trades.iter().filter(|t| matches!(t.exit_reason, ExitReason::EndOfData)).count();
        let total_closed_trades = total_trades.saturating_sub(total_open_trades);

        // Open trade PnL
        let open_trade_pnl: f64 = trades
            .iter()
            .filter(|t| matches!(t.exit_reason, ExitReason::EndOfData))
            .map(|t| t.pnl)
            .sum();

        // Only count closed trades for win/loss statistics
        let closed_trades: Vec<_> =
            trades.iter().filter(|t| !matches!(t.exit_reason, ExitReason::EndOfData)).collect();

        let winning_trades = closed_trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = closed_trades.iter().filter(|t| t.pnl < 0.0).count();

        let win_rate_pct = if total_closed_trades > 0 {
            winning_trades as f64 / total_closed_trades as f64 * 100.0
        } else {
            0.0
        };

        // Total fees paid
        let total_fees_paid: f64 = trades.iter().map(|t| t.fees).sum();

        // Best and worst trade
        let best_trade_pct =
            trades.iter().map(|t| t.return_pct).fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let best_trade_pct = if best_trade_pct.is_infinite() { 0.0 } else { best_trade_pct };

        let worst_trade_pct =
            trades.iter().map(|t| t.return_pct).fold(f64::INFINITY, |a, b| a.min(b));
        let worst_trade_pct = if worst_trade_pct.is_infinite() { 0.0 } else { worst_trade_pct };

        // Profit factor (based on closed trades)
        let gross_profit: f64 = closed_trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 =
            closed_trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Expectancy = average trade PnL
        let expectancy = if total_closed_trades > 0 {
            closed_trades.iter().map(|t| t.pnl).sum::<f64>() / total_closed_trades as f64
        } else {
            0.0
        };

        // SQN = (Expectancy / StdDev of trade PnL) * sqrt(total trades)
        let sqn = if total_closed_trades > 1 {
            let trade_pnls: Vec<f64> = closed_trades.iter().map(|t| t.pnl).collect();
            let mean = expectancy;
            let variance = trade_pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                / (total_closed_trades - 1) as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 {
                (mean / std_dev) * (total_closed_trades as f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Average returns
        let avg_trade_return_pct = if total_trades > 0 {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        let avg_win_pct = if winning_trades > 0 {
            closed_trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.return_pct).sum::<f64>()
                / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss_pct = if losing_trades > 0 {
            closed_trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.return_pct).sum::<f64>()
                / losing_trades as f64
        } else {
            0.0
        };

        // Average winning/losing trade duration
        let avg_winning_duration = if winning_trades > 0 {
            closed_trades
                .iter()
                .filter(|t| t.pnl > 0.0)
                .map(|t| t.holding_period() as f64)
                .sum::<f64>()
                / winning_trades as f64
        } else {
            0.0
        };

        let avg_losing_duration = if losing_trades > 0 {
            closed_trades
                .iter()
                .filter(|t| t.pnl < 0.0)
                .map(|t| t.holding_period() as f64)
                .sum::<f64>()
                / losing_trades as f64
        } else {
            0.0
        };

        // Consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = self.calculate_consecutive(trades);

        // Holding period
        let avg_holding_period = if total_trades > 0 {
            trades.iter().map(|t| t.holding_period() as f64).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        // Exposure (time in market)
        let bars_in_position: usize = trades.iter().map(|t| t.holding_period()).sum();
        let exposure_pct = if !equity_curve.is_empty() {
            bars_in_position as f64 / equity_curve.len() as f64 * 100.0
        } else {
            0.0
        };

        // Risk-adjusted metrics (calculated from daily portfolio returns, not trade returns)
        // This matches VectorBT's calculation methodology
        let (sharpe_ratio, sortino_ratio, omega_ratio) = self.calculate_risk_metrics(returns);

        // Calmar ratio: CAGR / max drawdown
        // VectorBT uses Compound Annual Growth Rate (CAGR)
        let num_periods = equity_curve.len().max(1) as f64;
        let years = num_periods / 365.25; // Convert to years using 365.25 days
        let total_return_frac = total_return_pct / 100.0;
        // CAGR = (end/start)^(1/years) - 1 = (1 + total_return)^(1/years) - 1
        let cagr =
            if years > 0.0 { (1.0 + total_return_frac).powf(1.0 / years) - 1.0 } else { 0.0 };
        let calmar_ratio = if max_drawdown_pct > 0.0 {
            cagr / (max_drawdown_pct / 100.0) // Both as fractions
        } else if total_return_pct > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestMetrics {
            total_return_pct,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            omega_ratio,
            max_drawdown_pct,
            max_drawdown_duration,
            win_rate_pct,
            profit_factor,
            expectancy,
            sqn,
            total_trades,
            total_closed_trades,
            total_open_trades,
            open_trade_pnl,
            winning_trades,
            losing_trades,
            start_value,
            end_value,
            total_fees_paid,
            best_trade_pct,
            worst_trade_pct,
            avg_trade_return_pct,
            avg_win_pct,
            avg_loss_pct,
            avg_winning_duration,
            avg_losing_duration,
            max_consecutive_wins,
            max_consecutive_losses,
            avg_holding_period,
            exposure_pct,
        }
    }

    /// Calculate max drawdown duration from drawdown curve.
    fn calculate_max_drawdown_duration(&self, drawdown_curve: &[f64]) -> usize {
        let mut max_duration = 0;
        let mut current_duration = 0;

        for &dd in drawdown_curve {
            if dd > 0.0 {
                current_duration += 1;
                max_duration = max_duration.max(current_duration);
            } else {
                current_duration = 0;
            }
        }

        max_duration
    }

    /// Calculate max consecutive wins and losses.
    fn calculate_consecutive(&self, trades: &[Trade]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else if trade.pnl < 0.0 {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        (max_wins, max_losses)
    }

    /// Calculate risk-adjusted metrics from daily portfolio returns.
    /// Returns (sharpe_ratio, sortino_ratio, omega_ratio).
    /// Uses 365 days for annualization to match VectorBT.
    fn calculate_risk_metrics(&self, returns: &[f64]) -> (f64, f64, f64) {
        if returns.len() < 2 {
            return (0.0, 0.0, 1.0);
        }

        // VectorBT uses 365 days (calendar days) for annualization
        let periods_per_year: f64 = 365.0;
        let _n = returns.len() as f64;

        // Filter out NaN values
        let valid_returns: Vec<f64> = returns.iter().filter(|r| !r.is_nan()).copied().collect();

        if valid_returns.len() < 2 {
            return (0.0, 0.0, 1.0);
        }

        let n_valid = valid_returns.len() as f64;

        // Calculate mean return
        let mean = valid_returns.iter().sum::<f64>() / n_valid;

        // Calculate standard deviation
        let variance =
            valid_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n_valid - 1.0);
        let std_dev = variance.sqrt();

        // Sharpe Ratio = (mean * periods_per_year) / (std_dev * sqrt(periods_per_year))
        // Simplified: Sharpe = mean / std_dev * sqrt(periods_per_year)
        let sharpe_ratio =
            if std_dev > 0.0 { (mean / std_dev) * periods_per_year.sqrt() } else { 0.0 };

        // Sortino Ratio - uses downside deviation (only negative returns)
        let downside_returns: Vec<f64> =
            valid_returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / n_valid // Divide by total count, not downside count
        } else {
            0.0
        };
        let downside_std = downside_variance.sqrt();

        let sortino_ratio = if downside_std > 0.0 {
            (mean / downside_std) * periods_per_year.sqrt()
        } else if mean > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Omega Ratio = sum of returns above threshold / |sum of returns below threshold|
        // With threshold = 0
        let sum_positive: f64 = valid_returns.iter().filter(|&&r| r > 0.0).sum();
        let sum_negative: f64 = valid_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

        let omega_ratio = if sum_negative > 0.0 {
            sum_positive / sum_negative
        } else if sum_positive > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };

        (sharpe_ratio, sortino_ratio, omega_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlcv() -> OhlcvData {
        OhlcvData {
            timestamps: (0..20).map(|i| i as i64).collect(),
            open: vec![
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 101.0,
                102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            ],
            high: vec![
                101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 102.0,
                103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
            ],
            low: vec![
                99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0,
                101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
            ],
            close: vec![
                100.5, 101.5, 102.5, 103.5, 104.5, 105.0, 104.0, 103.0, 102.0, 101.0, 100.5, 101.5,
                102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5,
            ],
            volume: vec![1000.0; 20],
        }
    }

    fn sample_signals() -> CompiledSignals {
        CompiledSignals {
            symbol: "TEST".to_string(),
            entries: vec![
                false, true, false, false, false, false, false, false, false, false, false, true,
                false, false, false, false, false, false, false, false,
            ],
            exits: vec![
                false, false, false, false, false, true, false, false, false, false, false, false,
                false, false, false, true, false, false, false, false,
            ],
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        }
    }

    #[test]
    fn test_basic_backtest() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.0,
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
        };

        let engine = PortfolioEngine::new(config);
        let ohlcv = sample_ohlcv();
        let signals = sample_signals();

        let result = engine.run_single(&ohlcv, &signals);

        // Should have 2 trades
        assert_eq!(result.trades.len(), 2);

        // First trade: entry at 101.5, exit at 105.0
        let trade1 = &result.trades[0];
        assert!((trade1.entry_price - 101.5).abs() < 1e-10);
        assert!((trade1.exit_price - 105.0).abs() < 1e-10);
        assert!(trade1.pnl > 0.0); // Profitable

        // Equity curve should have correct length
        assert_eq!(result.equity_curve.len(), 20);
    }

    #[test]
    fn test_with_fees() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.001, // 0.1%
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
        };

        let engine = PortfolioEngine::new(config);
        let ohlcv = sample_ohlcv();
        let signals = sample_signals();

        let result = engine.run_single(&ohlcv, &signals);

        // Trades should have fees deducted
        for trade in &result.trades {
            assert!(trade.fees > 0.0);
        }
    }

    #[test]
    fn test_with_stop_loss() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.0,
            slippage: 0.0,
            stop: StopConfig::Fixed { percent: 0.02 }, // 2% stop
            target: TargetConfig::None,
            upon_bar_close: true,
        };

        let engine = PortfolioEngine::new(config);

        // Create data where stop would be hit
        let mut ohlcv = sample_ohlcv();
        // Add a big drop after entry
        ohlcv.low[3] = 95.0; // Big drop
        ohlcv.close[3] = 96.0;

        let signals = sample_signals();
        let result = engine.run_single(&ohlcv, &signals);

        // First trade should exit on stop loss
        assert_eq!(result.trades[0].exit_reason, ExitReason::StopLoss);
    }
}
