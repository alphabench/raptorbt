//! Multi-strategy backtest implementation.
//!
//! Supports running multiple strategies on the same instrument.

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, OhlcvData, Trade,
};
use crate::execution::FeeModel;
use crate::metrics::streaming::StreamingMetrics;

/// Strategy combination mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineMode {
    /// Enter when any strategy signals.
    Any,
    /// Enter when all strategies signal.
    All,
    /// Enter when majority of strategies signal.
    Majority,
    /// Run strategies independently with separate capital.
    Independent,
    /// Vote-weighted combination.
    Weighted,
}

impl Default for CombineMode {
    fn default() -> Self {
        CombineMode::Any
    }
}

/// Multi-strategy configuration.
#[derive(Debug, Clone)]
pub struct MultiStrategyConfig {
    /// Base backtest config.
    pub base: BacktestConfig,
    /// Strategy combination mode.
    pub combine_mode: CombineMode,
    /// Capital allocation per strategy (for independent mode).
    pub capital_per_strategy: Option<f64>,
    /// Strategy weights (for weighted mode).
    pub strategy_weights: Vec<f64>,
}

impl Default for MultiStrategyConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            combine_mode: CombineMode::Any,
            capital_per_strategy: None,
            strategy_weights: vec![],
        }
    }
}

/// Multi-strategy backtest runner.
#[derive(Debug)]
pub struct MultiStrategyBacktest {
    /// Configuration.
    config: MultiStrategyConfig,
    /// Fee model.
    #[allow(dead_code)]
    fee_model: FeeModel,
}

impl MultiStrategyBacktest {
    /// Create a new multi-strategy backtest.
    pub fn new(config: MultiStrategyConfig) -> Self {
        Self {
            fee_model: FeeModel::percentage(config.base.fees),
            config,
        }
    }

    /// Run multi-strategy backtest.
    ///
    /// # Arguments
    /// * `ohlcv` - OHLCV data for the instrument
    /// * `strategies` - Vector of compiled signals from each strategy
    ///
    /// # Returns
    /// Combined backtest result
    pub fn run(&self, ohlcv: &OhlcvData, strategies: &[CompiledSignals]) -> BacktestResult {
        if strategies.is_empty() {
            return self.empty_result();
        }

        let n = ohlcv.len();
        for signals in strategies {
            assert_eq!(
                signals.len(),
                n,
                "All strategies must have same length as OHLCV"
            );
        }

        match self.config.combine_mode {
            CombineMode::Independent => self.run_independent(ohlcv, strategies),
            _ => self.run_combined(ohlcv, strategies),
        }
    }

    /// Run strategies independently with separate capital.
    fn run_independent(&self, ohlcv: &OhlcvData, strategies: &[CompiledSignals]) -> BacktestResult {
        let n_strategies = strategies.len();
        let capital_per = self
            .config
            .capital_per_strategy
            .unwrap_or(self.config.base.initial_capital / n_strategies as f64);

        // Run each strategy independently
        let mut all_trades: Vec<Trade> = Vec::new();
        let mut strategy_equities: Vec<Vec<f64>> = Vec::new();

        for (strat_idx, signals) in strategies.iter().enumerate() {
            let single_config = BacktestConfig {
                initial_capital: capital_per,
                ..self.config.base.clone()
            };
            let single = crate::strategies::single::SingleBacktest::new(single_config);
            let result = single.run(ohlcv, signals);

            // Tag trades with strategy index
            for mut trade in result.trades {
                trade.symbol = format!("{}_{}", trade.symbol, strat_idx);
                all_trades.push(trade);
            }

            strategy_equities.push(result.equity_curve);
        }

        // Combine equity curves
        let n = ohlcv.len();
        let mut combined_equity = vec![0.0; n];
        for i in 0..n {
            for equity in &strategy_equities {
                combined_equity[i] += equity[i];
            }
        }

        // Calculate drawdown
        let mut peak = combined_equity[0];
        let mut drawdown_curve = vec![0.0; n];
        for i in 0..n {
            if combined_equity[i] > peak {
                peak = combined_equity[i];
            }
            drawdown_curve[i] = (peak - combined_equity[i]) / peak * 100.0;
        }

        // Calculate returns
        let mut returns = vec![0.0; n];
        for i in 1..n {
            returns[i] = (combined_equity[i] - combined_equity[i - 1]) / combined_equity[i - 1];
        }

        // Calculate metrics
        let mut streaming = StreamingMetrics::new();
        for trade in &all_trades {
            streaming.update(trade.return_pct / 100.0);
        }

        let metrics = self.calculate_metrics(
            &combined_equity,
            &drawdown_curve,
            &all_trades,
            &streaming,
            self.config.base.initial_capital,
        );

        BacktestResult::new(
            metrics,
            combined_equity,
            drawdown_curve,
            all_trades,
            returns,
        )
    }

    /// Run strategies with combined signals.
    fn run_combined(&self, ohlcv: &OhlcvData, strategies: &[CompiledSignals]) -> BacktestResult {
        let n = ohlcv.len();
        let n_strategies = strategies.len();

        // Combine entry signals
        let mut combined_entries = vec![false; n];
        let mut combined_exits = vec![false; n];

        for i in 0..n {
            let entry_count = strategies.iter().filter(|s| s.entries[i]).count();
            let exit_count = strategies.iter().filter(|s| s.exits[i]).count();

            combined_entries[i] = match self.config.combine_mode {
                CombineMode::Any => entry_count > 0,
                CombineMode::All => entry_count == n_strategies,
                CombineMode::Majority => entry_count > n_strategies / 2,
                CombineMode::Weighted => {
                    let weighted_sum: f64 = strategies
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| s.entries[i])
                        .map(|(idx, _)| {
                            self.config
                                .strategy_weights
                                .get(idx)
                                .copied()
                                .unwrap_or(1.0)
                        })
                        .sum();
                    let total_weight: f64 = self
                        .config
                        .strategy_weights
                        .iter()
                        .sum::<f64>()
                        .max(n_strategies as f64);
                    weighted_sum / total_weight > 0.5
                }
                CombineMode::Independent => unreachable!(),
            };

            // Exit when any strategy wants to exit (conservative)
            combined_exits[i] = exit_count > 0;
        }

        // Use first strategy's direction and symbol
        let direction = strategies[0].direction;
        let symbol = strategies[0].symbol.clone();

        let combined_signals = CompiledSignals {
            symbol,
            entries: combined_entries,
            exits: combined_exits,
            position_sizes: None,
            direction,
            weight: 1.0,
        };

        // Run single backtest with combined signals
        let single = crate::strategies::single::SingleBacktest::new(self.config.base.clone());
        single.run(ohlcv, &combined_signals)
    }

    /// Calculate metrics.
    fn calculate_metrics(
        &self,
        equity_curve: &[f64],
        drawdown_curve: &[f64],
        trades: &[Trade],
        streaming: &StreamingMetrics,
        initial_capital: f64,
    ) -> BacktestMetrics {
        let start_value = initial_capital;
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

    /// Create empty result.
    fn empty_result(&self) -> BacktestResult {
        BacktestResult::new(
            BacktestMetrics {
                start_value: self.config.base.initial_capital,
                end_value: self.config.base.initial_capital,
                ..Default::default()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_strategies() -> (OhlcvData, Vec<CompiledSignals>) {
        let n = 20;

        let ohlcv = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (100..100 + n).map(|x| x as f64).collect(),
            high: (101..101 + n).map(|x| x as f64).collect(),
            low: (99..99 + n).map(|x| x as f64).collect(),
            close: (100..100 + n).map(|x| x as f64 + 0.5).collect(),
            volume: vec![1000.0; n],
        };

        // Strategy 1: Early entry
        let mut entries1 = vec![false; n];
        let mut exits1 = vec![false; n];
        entries1[2] = true;
        exits1[8] = true;

        // Strategy 2: Later entry
        let mut entries2 = vec![false; n];
        let mut exits2 = vec![false; n];
        entries2[4] = true;
        exits2[10] = true;

        let signals1 = CompiledSignals {
            symbol: "TEST".to_string(),
            entries: entries1,
            exits: exits1,
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        };

        let signals2 = CompiledSignals {
            symbol: "TEST".to_string(),
            entries: entries2,
            exits: exits2,
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        };

        (ohlcv, vec![signals1, signals2])
    }

    #[test]
    fn test_multi_any_mode() {
        let config = MultiStrategyConfig {
            combine_mode: CombineMode::Any,
            ..Default::default()
        };
        let backtest = MultiStrategyBacktest::new(config);
        let (ohlcv, strategies) = sample_strategies();

        let result = backtest.run(&ohlcv, &strategies);

        // With Any mode, should enter at index 2 (first strategy)
        assert!(!result.trades.is_empty());
    }

    #[test]
    fn test_multi_all_mode() {
        let config = MultiStrategyConfig {
            combine_mode: CombineMode::All,
            ..Default::default()
        };
        let backtest = MultiStrategyBacktest::new(config);
        let (ohlcv, strategies) = sample_strategies();

        let result = backtest.run(&ohlcv, &strategies);

        // With All mode, should not enter (strategies don't signal at same time)
        assert!(result.trades.is_empty() || result.trades.len() < 2);
    }

    #[test]
    fn test_multi_independent_mode() {
        let config = MultiStrategyConfig {
            combine_mode: CombineMode::Independent,
            ..Default::default()
        };
        let backtest = MultiStrategyBacktest::new(config);
        let (ohlcv, strategies) = sample_strategies();

        let result = backtest.run(&ohlcv, &strategies);

        // With Independent mode, should have trades from both strategies
        assert!(result.trades.len() >= 2);
    }
}
