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

use crate::accounts::AccountMode;
use crate::core::types::{
    BacktestConfig, BacktestResult, CompiledSignals, InstrumentConfig, OhlcvData,
};
use crate::indicators::volatility::atr;
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::StepInput;
use crate::portfolio::ledger::PositionPolicy;
use crate::signals::processor::SignalProcessor;
use crate::strategies::legs::{bar_entry, kernel_bar};

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
    /// Entries the account could not fund (sized to nothing, or refused for
    /// capital or margin); not in `rejected_entries`.
    pub unfunded_entries: usize,
}

/// Result of a shared-capital portfolio run.
#[derive(Debug, Clone)]
pub struct PortfolioBacktestResult {
    pub result: BacktestResult,
    pub per_instrument: Vec<InstrumentSummary>,
    /// Total entries refused by the risk gate across all instruments.
    pub rejected_entries: usize,
    /// Total entries that could not be funded, across all instruments.
    pub unfunded_entries: usize,
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

        // Clean each instrument's signals independently.
        let processor = SignalProcessor::new();
        let cleaned: Vec<(Vec<bool>, Vec<bool>)> = instruments
            .iter()
            .map(|(_, s)| processor.clean_signals(&s.entries, &s.exits))
            .collect();

        // Precompute ATR only when a stop or target needs it.
        let atr_series = self.precompute_atr(instruments, instrument_configs, n_bars);

        // One book: the shared account, one kernel per instrument, the curves
        // and the portfolio-wide limits. What is particular to this runner is
        // the order of the bar (every exit before any entry, so capital freed
        // this bar funds this bar's entries) and the share of the pool an
        // entry may draw (`CapitalAllocation`, handed to the kernel as the
        // step's budget).
        let mut book = PortfolioBook::new(self.config.base.clone(), AccountMode::Cash);
        for (_, signals) in instruments {
            let inst_config = instrument_configs.and_then(|m| m.get(&signals.symbol));
            book.add_instrument(
                signals.symbol.clone(),
                signals.direction,
                None,
                inst_config,
                PositionPolicy::Net,
            );
        }

        // The limits judge the equity of the whole bar, every held instrument
        // marked, not the equity between two instruments' steps.
        book.judge_per_instant(true);
        let timestamps = &instruments[0].0.timestamps;
        let mut halted_at: Option<usize> = None;

        for i in 0..n_bars {
            let ts = timestamps[i];
            let step_input = |idx: usize, entry: bool, exit: bool, budget: Option<f64>| StepInput {
                entry,
                exit,
                atr: atr_series[idx].as_ref().map_or(0.0, |a| a.get(i).copied().unwrap_or(0.0)),
                size_mult: instruments[idx].1.position_sizes.as_ref().map(|s| s[i]),
                budget,
                ..StepInput::default()
            };

            // Exits first, across all instruments.
            for idx in 0..n_instruments {
                if !book.kernel(idx).is_in_position() {
                    continue;
                }
                let bar = kernel_bar(&instruments[idx].0, i);
                book.apply(
                    &bar_entry(idx, i, bar),
                    None,
                    step_input(idx, false, cleaned[idx].1[i], None),
                );
            }

            // Entries, gated by the book's limits and sized from the pool.
            for idx in 0..n_instruments {
                // Under NextBarOpen the previous bar's signal fills in this
                // bar's step, so a flat kernel with a waiting entry must be
                // stepped even though this bar has no signal of its own.
                let pending = book.kernel(idx).has_pending_entry();
                if book.kernel(idx).is_in_position() || !(cleaned[idx].0[i] || pending) {
                    continue;
                }
                let open_count = book.open_total();
                let budget = self.entry_budget(book.free_capital(), open_count, n_instruments);
                // A limit that will refuse this entry: the position limit, a
                // latched halt (drawdown, kill) or the day's loss block.
                let at_limit = self.config.base.max_positions.is_some_and(|m| open_count >= m)
                    || book.is_halted()
                    || book.any_day_blocked();
                if budget <= 0.0 && !at_limit {
                    // Nothing to give this entry: it is dropped, waiting one included.
                    if pending {
                        book.kernel_mut(idx).clear_pending_entry();
                    }
                    continue;
                }
                // Under a limit the step is taken so the book's gate refuses
                // the entry and counts it.
                let budget = budget.max(0.0);
                let bar = kernel_bar(&instruments[idx].0, i);
                let input = step_input(idx, cleaned[idx].0[i], false, Some(budget));
                book.apply(&bar_entry(idx, i, bar), None, input);
            }

            book.judge(ts, i);
            if halted_at.is_none() && book.is_halted() {
                halted_at = Some(i);
            }
        }

        let outcome = book.finish();
        let per_instrument = instruments
            .iter()
            .zip(&outcome.instruments)
            .map(|((_, s), o)| InstrumentSummary {
                symbol: s.symbol.clone(),
                trades: o.trades,
                pnl: o.pnl,
                rejected_entries: o.rejected_entries,
                unfunded_entries: o.unfunded_entries,
            })
            .collect();

        PortfolioBacktestResult {
            result: outcome.result,
            per_instrument,
            rejected_entries: outcome.rejected_entries,
            unfunded_entries: outcome.unfunded_entries,
            halted: outcome.halted,
            halted_at: halted_at.filter(|_| outcome.halted),
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
    ) -> Vec<Option<Vec<f64>>> {
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

                // `None` rather than a zero-filled array when this instrument
                // has no ATR-based stop or target -- see the matching comment
                // in `PortfolioEngine::run_single_with_instrument_config`. Here
                // the saving is per instrument, so a book of 50 symbols was
                // allocating 50 such arrays.
                let needs_atr = matches!(stop, StopConfig::Atr { .. })
                    || matches!(target, TargetConfig::Atr { .. });
                if !needs_atr {
                    return None;
                }

                let period = match stop {
                    StopConfig::Atr { period, .. } => period,
                    _ => match target {
                        TargetConfig::Atr { period, .. } => period,
                        _ => 14,
                    },
                };
                Some(
                    atr(&ohlcv.high, &ohlcv.low, &ohlcv.close, period)
                        .unwrap_or_else(|_| vec![0.0; n_bars]),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Direction;

    const DAY: i64 = 86_400_000_000_000;

    /// Flat-priced instrument, so P&L never obscures capital accounting.
    fn flat_instrument(_symbol: &str, price: f64, n: usize) -> OhlcvData<'static> {
        OhlcvData {
            timestamps: (0..n as i64).map(|i| i * DAY).collect(),
            open: vec![price; n].into(),
            high: vec![price; n].into(),
            low: vec![price; n].into(),
            close: vec![price; n].into(),
            volume: vec![1_000_000.0; n].into(),
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
    fn three_way_entry(n: usize) -> Vec<(OhlcvData<'static>, CompiledSignals)> {
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
            open: a_close.clone().into(),
            high: a_close.clone().into(),
            low: a_close.clone().into(),
            close: a_close.into(),
            volume: vec![1_000_000.0; n].into(),
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
