//! Basket/collective strategy backtest implementation.
//!
//! Supports multiple instruments with synchronized signals. The legs trade in
//! one [`PortfolioBook`], so they draw on one account and follow the same fill,
//! fee, margin and risk rules as every other runner.

use std::collections::HashMap;

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, InstrumentConfig, OhlcvData,
};
use crate::portfolio::allocation::AllocationStrategy;
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::StepInput;
use crate::portfolio::ledger::PositionPolicy;
use crate::signals::processor::SignalProcessor;
use crate::signals::synchronizer::{SignalSynchronizer, SyncMode};
use crate::strategies::legs::{account_mode, all_flat, bar_entry, half_entered, kernel_bar};

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
    /// Signal processor.
    signal_processor: SignalProcessor,
}

impl BasketBacktest {
    /// Create a new basket backtest.
    pub fn new(config: BasketConfig) -> Self {
        Self {
            synchronizer: SignalSynchronizer::new(config.sync_mode),
            signal_processor: SignalProcessor::new(),
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
    /// The synchronized signal opens every leg together, only when all of them
    /// are flat, and closes them together. Each leg is sized by its weight's
    /// share of the capital free at the entry, capped by its `alloted_capital`
    /// and rounded to its lot size; its fill, fees and slippage are the
    /// kernel's, at the price the configured fill timing names.
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

        let n_legs = instruments.len();
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

        // Clean signals. When the entry fills is the kernel's call: under
        // `NextBarOpen` it holds the decision and fills it at the next bar's
        // open, at the size the decision bar chose.
        let (clean_entries, clean_exits) =
            self.signal_processor.clean_signals(&synced_entries, &synced_exits);

        let mut book =
            PortfolioBook::new(self.config.base.clone(), account_mode(&self.config.base));
        book.reserve_curves(n_bars);
        for (_, signals) in instruments {
            // The cap is applied to the leg's share below, where the weights
            // are known; the kernel would apply it to a fraction of it.
            let leg_config = instrument_configs
                .and_then(|configs| configs.get(&signals.symbol))
                .map(|config| InstrumentConfig { alloted_capital: None, ..config.clone() });
            book.add_instrument(
                signals.symbol.clone(),
                signals.direction,
                None,
                leg_config.as_ref(),
                PositionPolicy::Net,
            );
        }
        let weights: Vec<f64> = instruments.iter().map(|(_, s)| s.weight).collect();
        let caps: Vec<Option<f64>> = instruments
            .iter()
            .map(|(_, s)| {
                instrument_configs
                    .and_then(|configs| configs.get(&s.symbol))
                    .and_then(|config| config.alloted_capital)
            })
            .collect();

        // The size fraction each leg asks for, per entry. Legs fill in order,
        // so leg k sizes against what the earlier legs left of the capital.
        let mut fractions: Vec<Option<f64>> = vec![None; n_legs];
        // A leg whose entry a limit refused leaves the others held without it;
        // they are closed on the next bar rather than carried as a partial basket.
        let mut unwind = false;
        for i in 0..n_bars {
            let mut refused = Vec::new();
            let mut entering = false;
            for (k, (ohlcv, _)) in instruments.iter().enumerate() {
                if k == 0 {
                    entering = clean_entries[i] && all_flat(&book, n_legs);
                    if entering {
                        fractions = entry_fractions(&weights, &caps, book.free_capital());
                    }
                }
                let input = StepInput {
                    entry: entering && fractions[k].is_some(),
                    exit: clean_exits[i] || unwind,
                    size_mult: if entering { fractions[k] } else { None },
                    ..StepInput::default()
                };
                refused.extend(book.apply(&bar_entry(k, i, kernel_bar(ohlcv, i)), None, input));
            }
            unwind = half_entered(&book, n_legs, &refused);
        }

        book.finish().result
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

/// The fraction of the capital free when leg k is sized that gives it its
/// weight's share of the capital free at the entry.
///
/// Leg k sees what legs before it left, `free * (sum of weights from k) / (sum
/// of all weights)`, so its own share is `weight_k / sum of weights from k` of
/// that. A cap lowers a leg's share to the cap. `None` for a leg that gets
/// nothing: no weight, or no capital.
fn entry_fractions(weights: &[f64], caps: &[Option<f64>], free: f64) -> Vec<Option<f64>> {
    let total: f64 = weights.iter().sum();
    let mut from_here = total;
    weights
        .iter()
        .zip(caps)
        .map(|(&weight, cap)| {
            let remaining = from_here;
            from_here -= weight;
            if total <= 0.0 || weight <= 0.0 || remaining <= 0.0 || free <= 0.0 {
                return None;
            }
            let share = free * weight / total;
            let budget = cap.map_or(share, |cap| cap.min(share));
            Some(budget / (free * remaining / total))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Direction;

    fn sample_instruments() -> Vec<(OhlcvData<'static>, CompiledSignals)> {
        let n = 20;

        let ohlcv1 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (100..100 + n).map(|x| x as f64).collect(),
            high: (101..101 + n).map(|x| x as f64).collect(),
            low: (99..99 + n).map(|x| x as f64).collect(),
            close: (100..100 + n).map(|x| x as f64 + 0.5).collect(),
            volume: vec![1000.0; n].into(),
        };

        let ohlcv2 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (50..50 + n).map(|x| x as f64).collect(),
            high: (51..51 + n).map(|x| x as f64).collect(),
            low: (49..49 + n).map(|x| x as f64).collect(),
            close: (50..50 + n).map(|x| x as f64 + 0.25).collect(),
            volume: vec![2000.0; n].into(),
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
