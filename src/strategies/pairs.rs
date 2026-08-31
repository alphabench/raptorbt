//! Pairs trading strategy backtest implementation.
//!
//! Supports long/short legs with hedge ratios.

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, Direction, ExitReason,
    OhlcvData, Trade,
};
use crate::execution::FeeModel;
use crate::metrics::streaming::StreamingMetrics;

/// Pairs trading configuration.
#[derive(Debug, Clone)]
pub struct PairsConfig {
    /// Base backtest config.
    pub base: BacktestConfig,
    /// Hedge ratio (units of leg2 per unit of leg1).
    pub hedge_ratio: f64,
    /// Whether to dynamically update hedge ratio.
    pub dynamic_hedge: bool,
    /// Lookback period for dynamic hedge calculation.
    pub hedge_lookback: usize,
    /// Maximum spread for entry.
    pub max_spread: Option<f64>,
    /// Entry z-score threshold.
    pub entry_zscore: f64,
    /// Exit z-score threshold.
    pub exit_zscore: f64,
}

impl Default for PairsConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            hedge_ratio: 1.0,
            dynamic_hedge: false,
            hedge_lookback: 20,
            max_spread: None,
            entry_zscore: 2.0,
            exit_zscore: 0.5,
        }
    }
}

/// Pairs trading backtest runner.
#[derive(Debug)]
pub struct PairsBacktest {
    /// Configuration.
    config: PairsConfig,
    /// Fee model.
    fee_model: FeeModel,
}

impl PairsBacktest {
    /// Create a new pairs backtest.
    pub fn new(config: PairsConfig) -> Self {
        Self { fee_model: config.base.fee_model(), config }
    }

    /// Run pairs trading backtest.
    ///
    /// # Arguments
    /// * `leg1_ohlcv` - OHLCV data for leg 1 (long leg when spread widens)
    /// * `leg2_ohlcv` - OHLCV data for leg 2 (short leg when spread widens)
    /// * `signals` - Entry/exit signals based on spread
    ///
    /// # Returns
    /// Backtest result
    pub fn run(
        &self,
        leg1_ohlcv: &OhlcvData,
        leg2_ohlcv: &OhlcvData,
        signals: &CompiledSignals,
    ) -> BacktestResult {
        let n = leg1_ohlcv.len();
        assert_eq!(n, leg2_ohlcv.len());
        assert_eq!(n, signals.len());

        // Clean signals
        let processor = crate::signals::processor::SignalProcessor::new();
        let (entries, exits) = processor.clean_signals(&signals.entries, &signals.exits);

        // Execution timing. Under NextBarOpen a bar-i decision executes on
        // bar i+1 at each leg's own open: the cleaned stream shifts one bar
        // forward (a last-bar decision never trades) and fills read the
        // open. The hedge ratio is decision-time information and is computed
        // from the decision bar's window, not the fill bar's — the fill
        // bar's close does not exist when the ratio is chosen. SameBarClose
        // is the historical behavior, byte-identical; SameBarOpenLookahead
        // has no distinct history in this runner and also fills at the
        // close.
        let next_open =
            self.config.base.resolved_fill_timing() == crate::core::types::FillTiming::NextBarOpen;
        let (entries, exits) = if next_open {
            let shift = crate::signals::processor::shift_signals;
            (shift(&entries, 1), shift(&exits, 1))
        } else {
            (entries, exits)
        };

        // Initialize state
        let mut cash = self.config.base.initial_capital;
        let mut position: Option<PairsPosition> = None;
        let mut equity_curve = vec![cash; n];
        let mut drawdown_curve = vec![0.0; n];
        let mut returns = vec![0.0; n];
        let mut trades: Vec<Trade> = Vec::new();
        let mut peak_equity = cash;
        let mut trade_counter = 0u64;

        // Main simulation loop
        for i in 0..n {
            let leg1_price = leg1_ohlcv.close[i];
            let leg2_price = leg2_ohlcv.close[i];
            // Fills pay this bar's tradeable price; equity marks at the close.
            let leg1_fill = if next_open { leg1_ohlcv.open[i] } else { leg1_price };
            let leg2_fill = if next_open { leg2_ohlcv.open[i] } else { leg2_price };

            // Check for exit
            if exits[i] {
                if let Some(pos) = position.take() {
                    let (pnl, fees) = self.close_position(&pos, leg1_fill, leg2_fill);
                    let cost_basis = pos.leg1_cost + pos.leg2_cost;
                    let return_pct = if cost_basis > 0.0 { pnl / cost_basis * 100.0 } else { 0.0 };

                    // Return capital
                    cash += pos.leg1_size * leg1_fill + pos.leg2_size * leg2_fill - fees;

                    // Record trades for both legs
                    trades.push(Trade {
                        id: trade_counter,
                        symbol: format!("{}_LEG1", signals.symbol),
                        entry_idx: pos.entry_idx,
                        exit_idx: i,
                        entry_price: pos.leg1_entry_price,
                        exit_price: leg1_fill,
                        size: pos.leg1_size,
                        direction: pos.leg1_direction,
                        pnl: pnl / 2.0, // Split P&L attribution
                        return_pct: return_pct / 2.0,
                        entry_time: leg1_ohlcv.timestamps[pos.entry_idx],
                        exit_time: leg1_ohlcv.timestamps[i],
                        fees: fees / 2.0,
                        entry_fees: 0.0,
                        exit_fees: fees / 2.0,
                        fee_breakdown: None,
                        exit_reason: ExitReason::Signal,
                    });

                    trade_counter += 1;

                    trades.push(Trade {
                        id: trade_counter,
                        symbol: format!("{}_LEG2", signals.symbol),
                        entry_idx: pos.entry_idx,
                        exit_idx: i,
                        entry_price: pos.leg2_entry_price,
                        exit_price: leg2_fill,
                        size: pos.leg2_size,
                        direction: pos.leg2_direction,
                        pnl: pnl / 2.0,
                        return_pct: return_pct / 2.0,
                        entry_time: leg2_ohlcv.timestamps[pos.entry_idx],
                        exit_time: leg2_ohlcv.timestamps[i],
                        fees: fees / 2.0,
                        entry_fees: 0.0,
                        exit_fees: fees / 2.0,
                        fee_breakdown: None,
                        exit_reason: ExitReason::Signal,
                    });

                    trade_counter += 1;
                }
            }

            // Check for entry
            if entries[i] && position.is_none() {
                // Determine direction from signal direction
                let (leg1_dir, leg2_dir) = match signals.direction {
                    Direction::Long => (Direction::Long, Direction::Short),
                    Direction::Short => (Direction::Short, Direction::Long),
                };

                // Hedge ratio from the decision bar's window: under
                // NextBarOpen the decision was made on bar i-1, whose close
                // is the newest price the ratio may see.
                let decision_idx = if next_open { i - 1 } else { i };
                let hedge_ratio = if self.config.dynamic_hedge
                    && decision_idx >= self.config.hedge_lookback
                {
                    self.calculate_hedge_ratio(
                        &leg1_ohlcv.close[decision_idx - self.config.hedge_lookback..=decision_idx],
                        &leg2_ohlcv.close[decision_idx - self.config.hedge_lookback..=decision_idx],
                    )
                } else {
                    self.config.hedge_ratio
                };

                // Calculate position sizes, at the prices the fills pay
                let allocation = cash * 0.5; // Use 50% per leg
                let leg1_size = allocation / leg1_fill;
                let leg2_size = (allocation * hedge_ratio) / leg2_fill;

                let leg1_cost = leg1_size * leg1_fill;
                let leg2_cost = leg2_size * leg2_fill;
                let entry_fees = self.fee_model.calculate(leg1_fill, leg1_size, leg1_dir)
                    + self.fee_model.calculate(leg2_fill, leg2_size, leg2_dir);

                cash -= leg1_cost + leg2_cost + entry_fees;

                position = Some(PairsPosition {
                    entry_idx: i,
                    leg1_entry_price: leg1_fill,
                    leg2_entry_price: leg2_fill,
                    leg1_size,
                    leg2_size,
                    leg1_direction: leg1_dir,
                    leg2_direction: leg2_dir,
                    leg1_cost,
                    leg2_cost,
                    hedge_ratio,
                });
            }

            // Update equity
            let position_value = if let Some(ref pos) = position {
                let _leg1_value = pos.leg1_size * leg1_price;
                let _leg2_value = pos.leg2_size * leg2_price;

                // For pairs, value is long leg - short leg + cash equivalent
                let leg1_pnl = (leg1_price - pos.leg1_entry_price)
                    * pos.leg1_size
                    * pos.leg1_direction.multiplier();
                let leg2_pnl = (leg2_price - pos.leg2_entry_price)
                    * pos.leg2_size
                    * pos.leg2_direction.multiplier();

                pos.leg1_cost + pos.leg2_cost + leg1_pnl + leg2_pnl
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
            let leg1_price = leg1_ohlcv.close[last_idx];
            let leg2_price = leg2_ohlcv.close[last_idx];

            let (pnl, fees) = self.close_position(&pos, leg1_price, leg2_price);
            let cost_basis = pos.leg1_cost + pos.leg2_cost;
            let return_pct = if cost_basis > 0.0 { pnl / cost_basis * 100.0 } else { 0.0 };

            trades.push(Trade {
                id: trade_counter,
                symbol: signals.symbol.clone(),
                entry_idx: pos.entry_idx,
                exit_idx: last_idx,
                entry_price: pos.leg1_entry_price,
                exit_price: leg1_price,
                size: pos.leg1_size + pos.leg2_size,
                direction: pos.leg1_direction,
                pnl,
                return_pct,
                entry_time: leg1_ohlcv.timestamps[pos.entry_idx],
                exit_time: leg1_ohlcv.timestamps[last_idx],
                fees,
                entry_fees: 0.0,
                exit_fees: fees,
                fee_breakdown: None,
                exit_reason: ExitReason::EndOfData,
            });
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(
            &equity_curve,
            &drawdown_curve,
            &returns,
            leg1_ohlcv.timestamps.as_slice(),
            &trades,
        );

        BacktestResult::new(metrics, equity_curve, drawdown_curve, trades, returns)
    }

    /// Calculate hedge ratio using OLS regression.
    fn calculate_hedge_ratio(&self, leg1_prices: &[f64], leg2_prices: &[f64]) -> f64 {
        let n = leg1_prices.len() as f64;
        if n < 2.0 {
            return self.config.hedge_ratio;
        }

        let sum_x: f64 = leg2_prices.iter().sum();
        let sum_y: f64 = leg1_prices.iter().sum();
        let sum_xy: f64 = leg1_prices.iter().zip(leg2_prices.iter()).map(|(y, x)| x * y).sum();
        let sum_x2: f64 = leg2_prices.iter().map(|x| x * x).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return self.config.hedge_ratio;
        }

        let beta = (n * sum_xy - sum_x * sum_y) / denominator;
        beta.clamp(0.1, 10.0) // Constrain to reasonable range
    }

    /// Close position and calculate P&L.
    fn close_position(
        &self,
        position: &PairsPosition,
        leg1_price: f64,
        leg2_price: f64,
    ) -> (f64, f64) {
        let leg1_pnl = (leg1_price - position.leg1_entry_price)
            * position.leg1_size
            * position.leg1_direction.multiplier();

        let leg2_pnl = (leg2_price - position.leg2_entry_price)
            * position.leg2_size
            * position.leg2_direction.multiplier();

        let exit_fees =
            self.fee_model.calculate(leg1_price, position.leg1_size, position.leg1_direction)
                + self.fee_model.calculate(leg2_price, position.leg2_size, position.leg2_direction);

        let total_pnl = leg1_pnl + leg2_pnl - exit_fees;

        (total_pnl, exit_fees)
    }

    /// Calculate metrics.
    fn calculate_metrics(
        &self,
        equity_curve: &[f64],
        drawdown_curve: &[f64],
        returns: &[f64],
        timestamps: &[i64],
        trades: &[Trade],
    ) -> BacktestMetrics {
        let start_value = self.config.base.initial_capital;
        let end_value = *equity_curve.last().unwrap_or(&start_value);

        let total_return_pct = (end_value - start_value) / start_value * 100.0;
        let max_drawdown_pct = drawdown_curve.iter().fold(0.0f64, |a, &b| a.max(b));

        // For pairs, count trade pairs (every 2 trades = 1 round trip)
        let total_trades = trades.len() / 2;
        let winning_trades =
            trades.chunks(2).filter(|chunk| chunk.iter().map(|t| t.pnl).sum::<f64>() > 0.0).count();
        let losing_trades = total_trades.saturating_sub(winning_trades);

        let win_rate_pct = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };

        let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Sharpe/Sortino from per-bar returns, matching run_single_backtest.
        //
        // Through 0.4.1 this path annualized per-*trade* returns at a hardcoded
        // 252, which assumes one trade per trading day and inflates the ratio by
        // roughly sqrt(n_bars / n_trades). legacy_annualization restores that
        // basis as well as the constant, so old results stay reproducible.
        let (sharpe_ratio, sortino_ratio) = if self.config.base.legacy_annualization {
            let mut streaming = StreamingMetrics::new();
            for trade in trades {
                streaming.update(trade.return_pct / 100.0);
            }
            (
                streaming.sharpe_ratio(crate::metrics::annualization::LEGACY_PERIODS_STRATEGIES),
                streaming.sortino_ratio(crate::metrics::annualization::LEGACY_PERIODS_STRATEGIES),
            )
        } else {
            let periods_per_year =
                crate::metrics::annualization::resolve_periods_per_year_with_session(
                    self.config.base.periods_per_year,
                    timestamps,
                    self.config.base.session_spec(),
                    crate::metrics::annualization::LEGACY_PERIODS_STRATEGIES,
                );
            let (sharpe, sortino, _omega) = crate::portfolio::engine::risk_metrics(
                returns,
                periods_per_year,
                self.config.base.risk_free_rate,
            );
            (sharpe, sortino)
        };

        BacktestMetrics {
            total_return_pct,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio: crate::metrics::drawdown::calmar_ratio(
                total_return_pct,
                max_drawdown_pct,
            ),
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

/// Internal pairs position state.
#[derive(Debug, Clone)]
struct PairsPosition {
    entry_idx: usize,
    leg1_entry_price: f64,
    leg2_entry_price: f64,
    leg1_size: f64,
    leg2_size: f64,
    leg1_direction: Direction,
    leg2_direction: Direction,
    leg1_cost: f64,
    leg2_cost: f64,
    #[allow(dead_code)]
    hedge_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pairs_data() -> (OhlcvData, OhlcvData, CompiledSignals) {
        let n = 20;

        // Leg 1: Trending up
        let leg1 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (100..100 + n).map(|x| x as f64).collect(),
            high: (101..101 + n).map(|x| x as f64).collect(),
            low: (99..99 + n).map(|x| x as f64).collect(),
            close: (100..100 + n).map(|x| x as f64 + 0.5).collect(),
            volume: vec![1000.0; n],
        };

        // Leg 2: Correlated but with different magnitude
        let leg2 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (50..50 + n).map(|x| x as f64).collect(),
            high: (51..51 + n).map(|x| x as f64).collect(),
            low: (49..49 + n).map(|x| x as f64).collect(),
            close: (50..50 + n).map(|x| x as f64 + 0.2).collect(),
            volume: vec![2000.0; n],
        };

        let mut entries = vec![false; n];
        let mut exits = vec![false; n];
        entries[2] = true;
        exits[10] = true;

        let signals = CompiledSignals {
            symbol: "PAIR".to_string(),
            entries,
            exits,
            position_sizes: None,
            direction: Direction::Long, // Long leg1, short leg2
            weight: 1.0,
        };

        (leg1, leg2, signals)
    }

    #[test]
    fn test_pairs_backtest() {
        let config = PairsConfig::default();
        let backtest = PairsBacktest::new(config);
        let (leg1, leg2, signals) = sample_pairs_data();

        let result = backtest.run(&leg1, &leg2, &signals);

        // Should have trades for both legs
        assert!(result.trades.len() >= 2);
        assert_eq!(result.equity_curve.len(), 20);
    }

    #[test]
    fn test_hedge_ratio_calculation() {
        let config = PairsConfig { dynamic_hedge: true, hedge_lookback: 5, ..Default::default() };
        let backtest = PairsBacktest::new(config);

        let leg1 = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let leg2 = vec![50.0, 51.0, 52.0, 53.0, 54.0];

        let ratio = backtest.calculate_hedge_ratio(&leg1, &leg2);

        // Ratio should be approximately 2 (leg1 moves 2x leg2)
        assert!(ratio > 1.5 && ratio < 2.5);
    }
}
