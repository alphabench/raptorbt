//! Shared-capital multi-instrument backtest.
//!
//! Simulates N instruments against **one** cash pool, with portfolio-level
//! constraints gating every entry. This is what "run a portfolio with
//! max_positions=2" actually requires.
//!
//! It differs from the two existing multi-instrument paths in ways that matter:
//!
//! - [`BasketBacktest`](crate::strategies::basket::BasketBacktest) is strictly
//!   all-in/all-out: it enters every instrument together only when all are flat.
//!   A per-instrument position limit has no meaning there.
//! - Running one single-instrument backtest per symbol and summing the equity
//!   curves -- what callers typically do today -- gives each symbol its own
//!   private capital. Five symbols each "spend" the full initial capital, so
//!   the combined curve describes a portfolio that was never tradeable, and no
//!   cross-symbol constraint can be enforced.
//!
//! Here each instrument gets its own [`EngineKernel`], but they draw from and
//! return to a shared pool, and a shared [`RiskGate`] decides which entries are
//! allowed to open.

use std::collections::HashMap;

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, InstrumentConfig, OhlcvData,
    Trade,
};
use crate::execution::{FillPrice, SlippageModel};
use crate::indicators::volatility::atr;
use crate::portfolio::engine::compute_backtest_metrics_with_config;
use crate::portfolio::kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
use crate::portfolio::risk::RiskGate;
use crate::signals::processor::SignalProcessor;

/// How capital is divided among instruments when an entry opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CapitalAllocation {
    /// Each entry may use `pool / max_concurrent` of the pool.
    ///
    /// Reserves room for the other positions the constraints permit, so the
    /// first signal of the day does not consume the whole account.
    #[default]
    EqualWeight,
    /// Each entry may use the entire remaining pool.
    ///
    /// Reproduces the single-instrument sizing rule; with several instruments
    /// this is first-come-first-served.
    Full,
}

/// Configuration for a shared-capital portfolio run.
#[derive(Debug, Clone, Default)]
pub struct PortfolioBacktestConfig {
    /// Shared execution config, including the risk constraints.
    pub base: BacktestConfig,
    /// How much of the pool a single entry may consume.
    pub allocation: CapitalAllocation,
}

/// Per-instrument outcome, for attribution.
#[derive(Debug, Clone)]
pub struct InstrumentSummary {
    pub symbol: String,
    pub trades: usize,
    pub pnl: f64,
    /// Entries refused because the portfolio was already at its limit.
    pub rejected_entries: usize,
}

/// Result of a shared-capital portfolio run.
#[derive(Debug, Clone)]
pub struct PortfolioBacktestResult {
    pub result: BacktestResult,
    pub per_instrument: Vec<InstrumentSummary>,
    /// Total entries refused by the risk gate across all instruments.
    pub rejected_entries: usize,
    /// Whether the drawdown kill-switch tripped.
    pub halted: bool,
    /// Bar index at which the kill-switch tripped, if it did.
    pub halted_at: Option<usize>,
}

/// Shared-capital portfolio backtest runner.
#[derive(Debug)]
pub struct PortfolioBacktest {
    config: PortfolioBacktestConfig,
}

impl PortfolioBacktest {
    pub fn new(config: PortfolioBacktestConfig) -> Self {
        Self { config }
    }

    /// Run the portfolio backtest.
    ///
    /// All instruments must share a bar count and timeline; index `i` is the
    /// same instant for every instrument.
    pub fn run(
        &self,
        instruments: &[(OhlcvData, CompiledSignals)],
        instrument_configs: Option<&HashMap<String, InstrumentConfig>>,
    ) -> PortfolioBacktestResult {
        assert!(!instruments.is_empty(), "portfolio backtest needs at least one instrument");

        let n_bars = instruments[0].0.len();
        for (ohlcv, signals) in instruments {
            assert_eq!(ohlcv.len(), n_bars, "all instruments must have the same number of bars");
            assert_eq!(signals.len(), n_bars, "signals must match OHLCV length");
        }

        let n_instruments = instruments.len();
        let fee_model = self.config.base.fee_model();
        let slippage_model = if self.config.base.apply_slippage && self.config.base.slippage > 0.0 {
            SlippageModel::percentage(self.config.base.slippage)
        } else {
            SlippageModel::None
        };
        let fill_price = FillPrice::for_timing(self.config.base.resolved_fill_timing());

        // Clean each instrument's signals independently.
        let processor = SignalProcessor::new();
        let cleaned: Vec<(Vec<bool>, Vec<bool>)> = instruments
            .iter()
            .map(|(_, s)| processor.clean_signals(&s.entries, &s.exits))
            .collect();

        // Precompute ATR only when a stop or target needs it.
        let atr_series = self.precompute_atr(instruments, instrument_configs, n_bars);

        // One kernel per instrument. Each starts with zero cash: the pool is
        // handed to it bar by bar, so no kernel holds capital of its own.
        let mut kernels: Vec<EngineKernel> = instruments
            .iter()
            .map(|(_, signals)| {
                let inst_config = instrument_configs.and_then(|m| m.get(&signals.symbol));
                let mut kernel = EngineKernel::new(
                    self.config.base.clone(),
                    fee_model.clone(),
                    slippage_model.clone(),
                    fill_price,
                    signals.symbol.clone(),
                    signals.direction,
                    inst_config,
                );
                kernel.set_cash(0.0);
                kernel
            })
            .collect();

        let mut risk =
            RiskGate::new(self.config.base.max_positions, self.config.base.max_drawdown_pct);

        let mut cash = self.config.base.initial_capital;
        let mut equity_curve = vec![cash; n_bars];
        let mut drawdown_curve = vec![0.0; n_bars];
        let mut returns = vec![0.0; n_bars];
        let mut trades: Vec<Trade> = Vec::new();
        let mut peak_equity = cash;
        let mut per_instrument_trades = vec![0usize; n_instruments];
        let mut per_instrument_pnl = vec![0.0f64; n_instruments];
        let mut rejected = vec![0usize; n_instruments];
        let mut halted_at: Option<usize> = None;

        for i in 0..n_bars {
            // Exits first, across all instruments, so capital freed this bar is
            // available to entries on the same bar.
            for (idx, kernel) in kernels.iter_mut().enumerate() {
                if !kernel.is_in_position() {
                    continue;
                }
                let bar = bar_at(&instruments[idx].0, i);
                let input = StepInput {
                    entry: false,
                    exit: cleaned[idx].1[i],
                    atr: atr_series[idx].get(i).copied().unwrap_or(0.0),
                    size_mult: instruments[idx].1.position_sizes.as_ref().map(|s| s[i]),
                    ..StepInput::default()
                };

                kernel.set_cash(0.0);
                for event in kernel.step(i, &bar, input) {
                    if let EngineEvent::Exited { trade, .. } = event {
                        per_instrument_trades[idx] += 1;
                        per_instrument_pnl[idx] += trade.pnl;
                        trades.push(trade);
                    }
                }
                // Whatever the exit produced returns to the pool.
                cash += kernel.cash();
                kernel.set_cash(0.0);
            }

            // Entries, gated on the shared pool and the shared constraints.
            let open_now = kernels.iter().filter(|k| k.is_in_position()).count();
            let mut open_count = open_now;

            for (idx, kernel) in kernels.iter_mut().enumerate() {
                if kernel.is_in_position() || !cleaned[idx].0[i] {
                    continue;
                }

                if risk.check_entry(open_count).is_err() {
                    risk.record_rejection();
                    rejected[idx] += 1;
                    continue;
                }

                let budget = self.entry_budget(cash, open_count, n_instruments);
                if budget <= 0.0 {
                    continue;
                }

                let bar = bar_at(&instruments[idx].0, i);
                let input = StepInput {
                    entry: true,
                    exit: false,
                    atr: atr_series[idx].get(i).copied().unwrap_or(0.0),
                    size_mult: instruments[idx].1.position_sizes.as_ref().map(|s| s[i]),
                    ..StepInput::default()
                };

                // Lend the kernel its slice of the pool; it spends what it needs
                // and hands back the remainder.
                kernel.set_cash(budget);
                let events = kernel.step(i, &bar, input);
                let spent = budget - kernel.cash();
                kernel.set_cash(0.0);
                cash -= spent;

                if events.iter().any(|e| matches!(e, EngineEvent::Entered { .. })) {
                    open_count += 1;
                }
            }

            // Mark to market.
            let position_value: f64 = kernels
                .iter()
                .enumerate()
                .map(|(idx, k)| k.position_value(instruments[idx].0.close[i]))
                .sum();
            let equity = cash + position_value;
            equity_curve[i] = equity;

            if equity > peak_equity {
                peak_equity = equity;
            }
            drawdown_curve[i] =
                if peak_equity > 0.0 { (peak_equity - equity) / peak_equity * 100.0 } else { 0.0 };

            if i > 0 && equity_curve[i - 1] != 0.0 {
                returns[i] = (equity - equity_curve[i - 1]) / equity_curve[i - 1];
            }

            let was_halted = risk.is_halted();
            risk.on_equity(equity, peak_equity);
            if !was_halted && risk.is_halted() {
                halted_at = Some(i);
            }
        }

        // Mark any still-open positions at the final close, matching the
        // single-instrument engine's zero-fee EndOfData convention.
        let last_idx = n_bars - 1;
        for (idx, kernel) in kernels.iter_mut().enumerate() {
            if !kernel.is_in_position() {
                continue;
            }
            let bar = bar_at(&instruments[idx].0, last_idx);
            kernel.set_cash(0.0);
            if let Some(trade) = kernel.finalize(last_idx, &bar) {
                per_instrument_trades[idx] += 1;
                per_instrument_pnl[idx] += trade.pnl;
                trades.push(trade);
            }
            cash += kernel.cash();
            kernel.set_cash(0.0);
        }

        if n_bars > 0 {
            equity_curve[last_idx] = cash;
        }

        let metrics = self.finalize_metrics(
            &equity_curve,
            &drawdown_curve,
            &returns,
            &trades,
            &instruments[0].0.timestamps,
        );

        let per_instrument = instruments
            .iter()
            .enumerate()
            .map(|(idx, (_, s))| InstrumentSummary {
                symbol: s.symbol.clone(),
                trades: per_instrument_trades[idx],
                pnl: per_instrument_pnl[idx],
                rejected_entries: rejected[idx],
            })
            .collect();

        PortfolioBacktestResult {
            result: BacktestResult::new(metrics, equity_curve, drawdown_curve, trades, returns),
            per_instrument,
            rejected_entries: risk.rejected_entries(),
            halted: risk.is_halted(),
            halted_at,
        }
    }

    /// Capital a single entry may consume.
    ///
    /// `slots` is how many positions could still open in the worst case: the
    /// position limit when set, otherwise the instrument count. Dividing by it
    /// reserves room for the others, so the first instrument to signal does not
    /// consume the whole pool and starve the rest.
    fn entry_budget(&self, cash: f64, open_count: usize, n_instruments: usize) -> f64 {
        match self.config.allocation {
            CapitalAllocation::Full => cash,
            CapitalAllocation::EqualWeight => {
                let ceiling = self.config.base.max_positions.unwrap_or(n_instruments);
                if ceiling > open_count {
                    cash / (ceiling - open_count) as f64
                } else {
                    0.0
                }
            }
        }
    }

    fn precompute_atr(
        &self,
        instruments: &[(OhlcvData, CompiledSignals)],
        instrument_configs: Option<&HashMap<String, InstrumentConfig>>,
        n_bars: usize,
    ) -> Vec<Vec<f64>> {
        use crate::core::types::{StopConfig, TargetConfig};

        instruments
            .iter()
            .map(|(ohlcv, signals)| {
                let inst_config = instrument_configs.and_then(|m| m.get(&signals.symbol));
                let stop = inst_config
                    .and_then(|ic| ic.stop.as_ref())
                    .copied()
                    .unwrap_or(self.config.base.stop);
                let target = inst_config
                    .and_then(|ic| ic.target.as_ref())
                    .copied()
                    .unwrap_or(self.config.base.target);

                let needs_atr = matches!(stop, StopConfig::Atr { .. })
                    || matches!(target, TargetConfig::Atr { .. });
                if !needs_atr {
                    return vec![0.0; n_bars];
                }

                let period = match stop {
                    StopConfig::Atr { period, .. } => period,
                    _ => match target {
                        TargetConfig::Atr { period, .. } => period,
                        _ => 14,
                    },
                };
                atr(&ohlcv.high, &ohlcv.low, &ohlcv.close, period)
                    .unwrap_or_else(|_| vec![0.0; n_bars])
            })
            .collect()
    }

    fn finalize_metrics(
        &self,
        equity_curve: &[f64],
        drawdown_curve: &[f64],
        returns: &[f64],
        trades: &[Trade],
        timestamps: &[i64],
    ) -> BacktestMetrics {
        compute_backtest_metrics_with_config(
            equity_curve,
            drawdown_curve,
            returns,
            trades,
            timestamps,
            &self.config.base,
        )
    }
}

/// Extract a single bar from an OHLCV series.
fn bar_at(ohlcv: &OhlcvData, i: usize) -> KernelBar {
    KernelBar {
        timestamp: ohlcv.timestamps[i],
        open: ohlcv.open[i],
        high: ohlcv.high[i],
        low: ohlcv.low[i],
        close: ohlcv.close[i],
        volume: ohlcv.volume[i],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Direction;

    const DAY: i64 = 86_400_000_000_000;

    /// Flat-priced instrument, so P&L never obscures capital accounting.
    fn flat_instrument(_symbol: &str, price: f64, n: usize) -> OhlcvData {
        OhlcvData {
            timestamps: (0..n as i64).map(|i| i * DAY).collect(),
            open: vec![price; n],
            high: vec![price; n],
            low: vec![price; n],
            close: vec![price; n],
            volume: vec![1_000_000.0; n],
        }
    }

    fn signals(symbol: &str, entries: Vec<bool>, exits: Vec<bool>) -> CompiledSignals {
        CompiledSignals {
            symbol: symbol.to_string(),
            entries,
            exits,
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        }
    }

    /// Three instruments all signalling entry on bar 1.
    fn three_way_entry(n: usize) -> Vec<(OhlcvData, CompiledSignals)> {
        ["A", "B", "C"]
            .iter()
            .map(|sym| {
                let mut entries = vec![false; n];
                entries[1] = true;
                (flat_instrument(sym, 100.0, n), signals(sym, entries, vec![false; n]))
            })
            .collect()
    }

    #[test]
    fn max_positions_caps_concurrent_entries() {
        let instruments = three_way_entry(10);
        let config = PortfolioBacktestConfig {
            base: BacktestConfig {
                initial_capital: 300_000.0,
                fees: 0.0,
                max_positions: Some(2),
                ..Default::default()
            },
            allocation: CapitalAllocation::EqualWeight,
        };

        let out = PortfolioBacktest::new(config).run(&instruments, None);

        // Three instruments signalled; only two may open.
        let opened: usize = out.per_instrument.iter().filter(|s| s.trades > 0).count();
        assert_eq!(opened, 2, "expected 2 positions, got {opened}");
        assert_eq!(out.rejected_entries, 1, "third entry must be refused");
    }

    #[test]
    fn unconstrained_allows_all_entries() {
        let instruments = three_way_entry(10);
        let config = PortfolioBacktestConfig {
            base: BacktestConfig {
                initial_capital: 300_000.0,
                fees: 0.0,
                max_positions: None,
                ..Default::default()
            },
            allocation: CapitalAllocation::EqualWeight,
        };

        let out = PortfolioBacktest::new(config).run(&instruments, None);
        let opened: usize = out.per_instrument.iter().filter(|s| s.trades > 0).count();
        assert_eq!(opened, 3);
        assert_eq!(out.rejected_entries, 0);
    }

    /// The defect this runner exists to fix: capital is shared, not duplicated.
    #[test]
    fn capital_is_shared_across_instruments() {
        let n = 10;
        let instruments = three_way_entry(n);
        let capital = 300_000.0;
        let config = PortfolioBacktestConfig {
            base: BacktestConfig {
                initial_capital: capital,
                fees: 0.0,
                max_positions: Some(3),
                ..Default::default()
            },
            allocation: CapitalAllocation::EqualWeight,
        };

        let out = PortfolioBacktest::new(config).run(&instruments, None);

        // Prices are flat and fees zero, so equity must never exceed the pool.
        // Summing three independent single-instrument runs would report 3x here.
        for (i, &equity) in out.result.equity_curve.iter().enumerate() {
            assert!(
                equity <= capital * 1.000_001,
                "bar {i}: equity {equity} exceeds the {capital} pool"
            );
        }
    }

    #[test]
    fn drawdown_kill_switch_blocks_later_entries() {
        // Instrument A loses hard early; B signals afterwards.
        let n = 12;
        let mut a_close = vec![100.0; n];
        for (i, c) in a_close.iter_mut().enumerate().skip(2) {
            *c = 100.0 - (i as f64) * 8.0; // deep, sustained drawdown
        }
        let a = OhlcvData {
            timestamps: (0..n as i64).map(|i| i * DAY).collect(),
            open: a_close.clone(),
            high: a_close.clone(),
            low: a_close.clone(),
            close: a_close,
            volume: vec![1_000_000.0; n],
        };
        let mut a_entries = vec![false; n];
        a_entries[1] = true;

        let mut b_entries = vec![false; n];
        b_entries[9] = true; // after the drawdown has developed

        let instruments = vec![
            (a, signals("A", a_entries, vec![false; n])),
            (flat_instrument("B", 100.0, n), signals("B", b_entries, vec![false; n])),
        ];

        let config = PortfolioBacktestConfig {
            base: BacktestConfig {
                initial_capital: 100_000.0,
                fees: 0.0,
                max_drawdown_pct: Some(15.0),
                ..Default::default()
            },
            allocation: CapitalAllocation::Full,
        };

        let out = PortfolioBacktest::new(config).run(&instruments, None);

        assert!(out.halted, "kill-switch should have tripped");
        let b = out.per_instrument.iter().find(|s| s.symbol == "B").unwrap();
        assert_eq!(b.trades, 0, "B must not open after the halt");
        assert!(b.rejected_entries > 0, "B's entry should be recorded as refused");
    }

    #[test]
    fn equal_weight_reserves_room_for_remaining_slots() {
        let instruments = three_way_entry(10);
        let config = PortfolioBacktestConfig {
            base: BacktestConfig {
                initial_capital: 300_000.0,
                fees: 0.0,
                max_positions: Some(3),
                ..Default::default()
            },
            allocation: CapitalAllocation::EqualWeight,
        };

        let out = PortfolioBacktest::new(config).run(&instruments, None);

        // Each of three entries gets a third of the pool: 1000 shares at 100.
        for summary in &out.per_instrument {
            assert!(summary.trades > 0, "{} should have opened", summary.symbol);
        }
        let sizes: Vec<f64> = out.result.trades.iter().map(|t| t.size).collect();
        for size in &sizes {
            assert!((size - 1000.0).abs() < 1.0, "expected ~1000 shares per leg, got {size}");
        }
    }
}
