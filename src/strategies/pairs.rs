//! Pairs trading strategy backtest implementation.
//!
//! Supports long/short legs with hedge ratios. The two legs are instruments of
//! one [`PortfolioBook`], so the short leg is funded, charged and marked like
//! any other short and the pair draws on one account.

use crate::core::types::{BacktestConfig, BacktestResult, CompiledSignals, Direction, OhlcvData};
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::StepInput;
use crate::portfolio::ledger::PositionPolicy;
use crate::signals::processor::SignalProcessor;
use crate::strategies::legs::{account_mode, all_flat, bar_entry, half_entered, kernel_bar};

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
}

impl PairsBacktest {
    /// Create a new pairs backtest.
    pub fn new(config: PairsConfig) -> Self {
        Self { config }
    }

    /// Run pairs trading backtest.
    ///
    /// Leg 1 is bought and leg 2 sold when `signals.direction` is long, the
    /// reverse when short. The pair opens when both legs are flat and closes
    /// together. Each leg holds half of the capital free at the entry, leg 2
    /// scaled by the hedge ratio; a ratio above one shrinks both legs so the
    /// pair still fits the account. The ratio is the one in force on the
    /// decision bar. The record is two trades per round trip, one per leg
    /// (`<symbol>_LEG1`, `<symbol>_LEG2`), each with the profit and costs of
    /// its own leg.
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

        let (entries, exits) =
            SignalProcessor::new().clean_signals(&signals.entries, &signals.exits);

        let (leg1_dir, leg2_dir) = match signals.direction {
            Direction::Long => (Direction::Long, Direction::Short),
            Direction::Short => (Direction::Short, Direction::Long),
        };
        let legs = [(leg1_ohlcv, leg1_dir, "LEG1"), (leg2_ohlcv, leg2_dir, "LEG2")];

        let mut book =
            PortfolioBook::new(self.config.base.clone(), account_mode(&self.config.base));
        for (_, direction, name) in &legs {
            book.add_instrument(
                format!("{}_{name}", signals.symbol),
                *direction,
                None,
                None,
                PositionPolicy::Net,
            );
        }

        let mut fractions = [0.0f64; 2];
        // A leg whose entry a limit refused leaves the other held alone; it is
        // closed on the next bar rather than carried as an unhedged position.
        let mut unwind = false;
        for i in 0..n {
            let mut refused = Vec::new();
            let mut entering = false;
            for (k, (ohlcv, _, _)) in legs.iter().enumerate() {
                if k == 0 {
                    entering = entries[i] && all_flat(&book, 2);
                    if entering {
                        // The ratio sees up to the decision bar's close; under
                        // `NextBarOpen` the kernel fills that decision a bar
                        // later and the fill bar's close does not exist yet.
                        fractions =
                            self.leg_fractions(&leg1_ohlcv.close[..=i], &leg2_ohlcv.close[..=i]);
                    }
                }
                let input = StepInput {
                    entry: entering && fractions[k] > 0.0,
                    exit: exits[i] || unwind,
                    size_mult: entering.then_some(fractions[k]),
                    ..StepInput::default()
                };
                refused.extend(book.apply(&bar_entry(k, i, kernel_bar(ohlcv, i)), None, input));
            }
            unwind = half_entered(&book, 2, &refused);
        }

        book.finish().result
    }

    /// The fraction of the free capital each leg asks for when it is sized.
    ///
    /// Each leg's notional is half the capital, leg 2's times the hedge ratio.
    /// Above a ratio of one that is more than the account holds, so both are
    /// scaled to fit. Leg 2 is sized after leg 1 has taken its share.
    /// `closes` end at the decision bar.
    fn leg_fractions(&self, leg1_closes: &[f64], leg2_closes: &[f64]) -> [f64; 2] {
        let decision = leg1_closes.len() - 1;
        let hedge_ratio = if self.config.dynamic_hedge && decision >= self.config.hedge_lookback {
            self.calculate_hedge_ratio(
                &leg1_closes[decision - self.config.hedge_lookback..=decision],
                &leg2_closes[decision - self.config.hedge_lookback..=decision],
            )
        } else {
            self.config.hedge_ratio
        };
        let fit = (2.0 / (1.0 + hedge_ratio)).min(1.0);
        let leg1 = 0.5 * fit;
        let leg2 = 0.5 * hedge_ratio * fit;
        [leg1, leg2 / (1.0 - leg1)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pairs_data() -> (OhlcvData<'static>, OhlcvData<'static>, CompiledSignals) {
        let n = 20;

        // Leg 1: Trending up
        let leg1 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (100..100 + n).map(|x| x as f64).collect(),
            high: (101..101 + n).map(|x| x as f64).collect(),
            low: (99..99 + n).map(|x| x as f64).collect(),
            close: (100..100 + n).map(|x| x as f64 + 0.5).collect(),
            volume: vec![1000.0; n].into(),
        };

        // Leg 2: Correlated but with different magnitude
        let leg2 = OhlcvData {
            timestamps: (0..n as i64).collect(),
            open: (50..50 + n).map(|x| x as f64).collect(),
            high: (51..51 + n).map(|x| x as f64).collect(),
            low: (49..49 + n).map(|x| x as f64).collect(),
            close: (50..50 + n).map(|x| x as f64 + 0.2).collect(),
            volume: vec![2000.0; n].into(),
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
