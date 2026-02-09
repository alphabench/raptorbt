//! Basket/collective strategy backtest implementation.
//!
//! Supports multiple instruments with synchronized signals.

use std::collections::HashMap;

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, ExitReason, InstrumentConfig,
    OhlcvData, Trade,
};
use crate::execution::FeeModel;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::allocation::{AllocationStrategy, CapitalAllocator};
use crate::signals::processor::SignalProcessor;
use crate::signals::synchronizer::{SignalSynchronizer, SyncMode};

/// Basket backtest configuration.
#[derive(Debug, Clone)]
pub struct BasketConfig {
    /// Base backtest config.
    pub base: BacktestConfig,
    /// Signal synchronization mode.
    pub sync_mode: SyncMode,
    /// Capital allocation strategy.
    pub allocation: AllocationStrategy,
    /// Whether to rebalance on each signal.
    pub rebalance_on_signal: bool,
}

impl Default for BasketConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            sync_mode: SyncMode::All,
            allocation: AllocationStrategy::EqualWeight,
            rebalance_on_signal: false,
        }
    }
}

/// Basket/collective strategy backtest runner.
#[derive(Debug)]
pub struct BasketBacktest {
    /// Configuration.
    config: BasketConfig,
    /// Signal synchronizer.
    synchronizer: SignalSynchronizer,
    /// Capital allocator.
    #[allow(dead_code)]
    allocator: CapitalAllocator,
    /// Signal processor.
    signal_processor: SignalProcessor,
    /// Fee model.
    fee_model: FeeModel,
}

impl BasketBacktest {
    /// Create a new basket backtest.
    pub fn new(config: BasketConfig) -> Self {
        let allocator = CapitalAllocator::new(config.base.initial_capital)
            .with_strategy(config.allocation.clone());

        Self {
            synchronizer: SignalSynchronizer::new(config.sync_mode),
            allocator,
            signal_processor: SignalProcessor::new(),
            fee_model: FeeModel::percentage(config.base.fees),
            config,
        }
    }

    /// Run basket backtest with multiple instruments.
    ///
    /// # Arguments
    /// * `instruments` - Vector of (OhlcvData, CompiledSignals) pairs for each instrument
    ///
    /// # Returns
    /// Combined backtest result
    pub fn run(&self, instruments: &[(OhlcvData, CompiledSignals)]) -> BacktestResult {
        self.run_with_instrument_configs(instruments, None)
    }

    /// Run basket backtest with optional per-instrument configurations.
    ///
    /// # Arguments
    /// * `instruments` - Vector of (OhlcvData, CompiledSignals) pairs for each instrument
    /// * `instrument_configs` - Optional map of symbol -> InstrumentConfig
    ///
    /// # Returns
    /// Combined backtest result
    pub fn run_with_instrument_configs(
        &self,
        instruments: &[(OhlcvData, CompiledSignals)],
        instrument_configs: Option<&HashMap<String, InstrumentConfig>>,
    ) -> BacktestResult {
        if instruments.is_empty() {
            return self.empty_result();
        }

        let n_instruments = instruments.len();
        let n_bars = instruments[0].0.len();

        // Verify all instruments have same length
        for (ohlcv, signals) in instruments {
            assert_eq!(ohlcv.len(), n_bars, "All instruments must have same number of bars");
            assert_eq!(signals.len(), n_bars, "Signals must match OHLCV length");
        }

        // Synchronize signals
        let entry_signals: Vec<&[bool]> =
            instruments.iter().map(|(_, s)| s.entries.as_slice()).collect();
        let exit_signals: Vec<&[bool]> =
            instruments.iter().map(|(_, s)| s.exits.as_slice()).collect();

        let synced_entries = self.synchronizer.sync_entries(&entry_signals);
        let synced_exits = self.synchronizer.sync_exits(&exit_signals);

        // Clean signals
        let (clean_entries, clean_exits) =
            self.signal_processor.clean_signals(&synced_entries, &synced_exits);

        // Initialize state
        let mut cash = self.config.base.initial_capital;
        let mut positions: Vec<Option<PositionState>> = vec![None; n_instruments];
        let mut equity_curve = vec![cash; n_bars];
        let mut drawdown_curve = vec![0.0; n_bars];
        let mut returns = vec![0.0; n_bars];
        let mut trades: Vec<Trade> = Vec::new();
        let mut streaming = StreamingMetrics::new();
        let mut peak_equity = cash;
        let mut trade_counter = 0u64;

        // Main simulation loop
        for i in 0..n_bars {
            // Calculate current position values
            let mut _total_position_value = 0.0;
            for (inst_idx, (ohlcv, _)) in instruments.iter().enumerate() {
                if let Some(ref pos) = positions[inst_idx] {
                    _total_position_value += pos.size * ohlcv.close[i];
                }
            }

            // Check for exit
            if clean_exits[i] {
                for (inst_idx, (ohlcv, signals)) in instruments.iter().enumerate() {
                    if let Some(pos) = positions[inst_idx].take() {
                        let exit_price = ohlcv.close[i];
                        let fees =
                            self.fee_model.calculate(exit_price, pos.size, signals.direction);

                        let pnl = (exit_price - pos.entry_price)
                            * pos.size
                            * signals.direction.multiplier()
                            - fees;

                        let cost_basis = pos.entry_price * pos.size;
                        let return_pct =
                            if cost_basis > 0.0 { pnl / cost_basis * 100.0 } else { 0.0 };

                        cash += exit_price * pos.size - fees;

                        trades.push(Trade {
                            id: trade_counter,
                            symbol: signals.symbol.clone(),
                            entry_idx: pos.entry_idx,
                            exit_idx: i,
                            entry_price: pos.entry_price,
                            exit_price,
                            size: pos.size,
                            direction: signals.direction,
                            pnl,
                            return_pct,
                            entry_time: ohlcv.timestamps[pos.entry_idx],
                            exit_time: ohlcv.timestamps[i],
                            fees,
                            exit_reason: ExitReason::Signal,
                        });

                        trade_counter += 1;
                        streaming.update(return_pct / 100.0);
                    }
                }
            }

            // Check for entry
            if clean_entries[i] && positions.iter().all(|p| p.is_none()) {
                // Calculate position sizes
                let prices: Vec<f64> = instruments.iter().map(|(o, _)| o.close[i]).collect();
                let weights: Vec<f64> = instruments.iter().map(|(_, s)| s.weight).collect();
                let symbols: Vec<&str> =
                    instruments.iter().map(|(_, s)| s.symbol.as_str()).collect();
                let sizes = self.calculate_sizes_with_configs(
                    &prices,
                    &weights,
                    cash,
                    &symbols,
                    instrument_configs,
                );

                // Enter positions
                for (inst_idx, (ohlcv, signals)) in instruments.iter().enumerate() {
                    let size = sizes[inst_idx];
                    if size > 0.0 {
                        let entry_price = ohlcv.close[i];
                        let fees = self.fee_model.calculate(entry_price, size, signals.direction);
                        cash -= entry_price * size + fees;

                        positions[inst_idx] =
                            Some(PositionState { entry_idx: i, entry_price, size });
                    }
                }
            }

            // Update equity
            let mut position_value = 0.0;
            for (inst_idx, (ohlcv, _)) in instruments.iter().enumerate() {
                if let Some(ref pos) = positions[inst_idx] {
                    position_value += pos.size * ohlcv.close[i];
                }
            }
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

        // Close any remaining positions
        let last_idx = n_bars - 1;
        for (inst_idx, (ohlcv, signals)) in instruments.iter().enumerate() {
            if let Some(pos) = positions[inst_idx].take() {
                let exit_price = ohlcv.close[last_idx];
                let fees = self.fee_model.calculate(exit_price, pos.size, signals.direction);

                let pnl =
                    (exit_price - pos.entry_price) * pos.size * signals.direction.multiplier()
                        - fees;

                let cost_basis = pos.entry_price * pos.size;
                let return_pct = if cost_basis > 0.0 { pnl / cost_basis * 100.0 } else { 0.0 };

                trades.push(Trade {
                    id: trade_counter,
                    symbol: signals.symbol.clone(),
                    entry_idx: pos.entry_idx,
                    exit_idx: last_idx,
                    entry_price: pos.entry_price,
                    exit_price,
                    size: pos.size,
                    direction: signals.direction,
                    pnl,
                    return_pct,
                    entry_time: ohlcv.timestamps[pos.entry_idx],
                    exit_time: ohlcv.timestamps[last_idx],
                    fees,
                    exit_reason: ExitReason::EndOfData,
                });

                trade_counter += 1;
                streaming.update(return_pct / 100.0);
            }
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&equity_curve, &drawdown_curve, &trades, &streaming);

        BacktestResult::new(metrics, equity_curve, drawdown_curve, trades, returns)
    }

    /// Calculate position sizes for each instrument.
    #[allow(dead_code)]
    fn calculate_sizes(&self, prices: &[f64], weights: &[f64], available_capital: f64) -> Vec<f64> {
        let symbols: Vec<&str> = vec![""; prices.len()];
        self.calculate_sizes_with_configs(prices, weights, available_capital, &symbols, None)
    }

    /// Calculate position sizes with optional per-instrument config (lot_size rounding, capital caps).
    fn calculate_sizes_with_configs(
        &self,
        prices: &[f64],
        weights: &[f64],
        available_capital: f64,
        symbols: &[&str],
        instrument_configs: Option<&HashMap<String, InstrumentConfig>>,
    ) -> Vec<f64> {
        let n = prices.len();
        let total_weight: f64 = weights.iter().sum();

        if total_weight == 0.0 {
            return vec![0.0; n];
        }

        prices
            .iter()
            .zip(weights.iter())
            .enumerate()
            .map(|(idx, (&price, &weight))| {
                if price <= 0.0 {
                    return 0.0;
                }
                let default_allocation = available_capital * (weight / total_weight);

                // Use per-instrument alloted_capital if set, capped at default allocation
                let inst_config = instrument_configs
                    .and_then(|configs| symbols.get(idx).and_then(|sym| configs.get(*sym)));

                let allocation = inst_config
                    .and_then(|ic| ic.alloted_capital)
                    .map(|cap| cap.min(default_allocation))
                    .unwrap_or(default_allocation);

                let raw_size = allocation / price;

                // Round to lot_size
                inst_config.map(|ic| ic.round_to_lot(raw_size)).unwrap_or(raw_size)
            })
            .collect()
    }

    /// Calculate metrics for the backtest.
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
        let gross_loss: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let sharpe_ratio = streaming.sharpe_ratio(252.0);
        let sortino_ratio = streaming.sortino_ratio(252.0);
        let calmar_ratio = if max_drawdown_pct > 0.0 {
            total_return_pct / max_drawdown_pct
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

/// Internal position state.
#[derive(Debug, Clone)]
struct PositionState {
    entry_idx: usize,
    entry_price: f64,
    size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Direction;

    fn sample_instruments() -> Vec<(OhlcvData, CompiledSignals)> {
        let n = 20;

        let ohlcv1 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (100..100 + n).map(|x| x as f64).collect(),
            high: (101..101 + n).map(|x| x as f64).collect(),
            low: (99..99 + n).map(|x| x as f64).collect(),
            close: (100..100 + n).map(|x| x as f64 + 0.5).collect(),
            volume: vec![1000.0; n],
        };

        let ohlcv2 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (50..50 + n).map(|x| x as f64).collect(),
            high: (51..51 + n).map(|x| x as f64).collect(),
            low: (49..49 + n).map(|x| x as f64).collect(),
            close: (50..50 + n).map(|x| x as f64 + 0.25).collect(),
            volume: vec![2000.0; n],
        };

        let mut entries1 = vec![false; n];
        let mut exits1 = vec![false; n];
        entries1[2] = true;
        exits1[8] = true;

        let mut entries2 = vec![false; n];
        let mut exits2 = vec![false; n];
        entries2[2] = true;
        exits2[8] = true;

        let signals1 = CompiledSignals {
            symbol: "INST1".to_string(),
            entries: entries1,
            exits: exits1,
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        };

        let signals2 = CompiledSignals {
            symbol: "INST2".to_string(),
            entries: entries2,
            exits: exits2,
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        };

        vec![(ohlcv1, signals1), (ohlcv2, signals2)]
    }

    #[test]
    fn test_basket_backtest() {
        let config = BasketConfig::default();
        let backtest = BasketBacktest::new(config);
        let instruments = sample_instruments();

        let result = backtest.run(&instruments);

        // Should have trades for both instruments
        assert!(result.trades.len() >= 2);
        assert_eq!(result.equity_curve.len(), 20);
    }

    #[test]
    fn test_sync_mode_all() {
        let config = BasketConfig { sync_mode: SyncMode::All, ..Default::default() };
        let backtest = BasketBacktest::new(config);
        let instruments = sample_instruments();

        let result = backtest.run(&instruments);

        // With All mode, both instruments should enter at same time
        assert!(result.trades.len() >= 2);
    }

    #[test]
    fn test_empty_instruments() {
        let config = BasketConfig::default();
        let backtest = BasketBacktest::new(config);

        let result = backtest.run(&[]);

        assert_eq!(result.trades.len(), 0);
        assert!(result.equity_curve.is_empty());
    }
}
