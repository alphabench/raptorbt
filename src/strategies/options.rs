//! Options strategy backtest implementation.
//!
//! One option, traded from a premium series with the sizing options use. The
//! premium series is a one-instrument book, so the position follows the
//! kernel's fill, fee, margin and risk rules; lots are the instrument's lot
//! size.

use crate::core::types::{
    BacktestConfig, BacktestResult, CompiledSignals, InstrumentConfig, OhlcvData,
};
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::{KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::signals::processor::SignalProcessor;
use crate::strategies::legs::{account_mode, bar_entry};

/// Options position type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

/// Strike selection mode.
#[derive(Debug, Clone, Copy, Default)]
pub enum StrikeSelection {
    /// At-the-money (closest to spot).
    #[default]
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
}

impl OptionsBacktest {
    /// Create a new options backtest.
    pub fn new(config: OptionsConfig) -> Self {
        Self { config }
    }

    /// Run options backtest.
    ///
    /// # Arguments
    /// * `spot_ohlcv` - Spot/underlying OHLCV data (its timestamps date the bars)
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
        self.run_with_opens(spot_ohlcv, option_prices, None, signals)
    }

    /// [`Self::run`], with the premium series' opening prices supplied.
    ///
    /// Under [`FillTiming::NextBarOpen`] a fill lands on the bar after the
    /// decision; with `option_open_prices` present it prices at that bar's
    /// OPEN premium — a real quote the caller observed — instead of the
    /// bar's (later) settled value. Nothing is synthesized: without the
    /// series, the next bar's value remains the honest causal fill. The
    /// opens are ignored outside `NextBarOpen`, where fills coincide with
    /// the decision bar's value by design.
    ///
    /// The position is long or short as `signals.direction` says, and a short
    /// one gains when the premium falls. The strike is the premium series'
    /// own: nothing here derives a price from the underlying.
    ///
    /// [`FillTiming::NextBarOpen`]: crate::core::types::FillTiming::NextBarOpen
    pub fn run_with_opens(
        &self,
        spot_ohlcv: &OhlcvData,
        option_prices: &[f64],
        option_open_prices: Option<&[f64]>,
        signals: &CompiledSignals,
    ) -> BacktestResult {
        let n = spot_ohlcv.len();
        assert_eq!(n, option_prices.len());
        assert_eq!(n, signals.len());
        if let Some(opens) = option_open_prices {
            assert_eq!(n, opens.len(), "option_open_prices must match the premium series");
        }

        let (entries, exits) =
            SignalProcessor::new().clean_signals(&signals.entries, &signals.exits);
        let next_open =
            self.config.base.resolved_fill_timing() == crate::core::types::FillTiming::NextBarOpen;

        let lot = self.config.lot_size as f64;
        let mut book =
            PortfolioBook::new(self.config.base.clone(), account_mode(&self.config.base));
        let leg_config = InstrumentConfig { lot_size: Some(lot), ..Default::default() };
        book.add_instrument(
            signals.symbol.clone(),
            signals.direction,
            None,
            Some(&leg_config),
            PositionPolicy::Net,
        );

        for i in 0..n {
            let premium = option_prices[i];
            // The premium series has one value per bar: a fill at the bar's
            // open prices at the caller's open premium when there is one.
            let open = match (next_open, option_open_prices) {
                (true, Some(opens)) => opens[i],
                _ => premium,
            };
            let bar = KernelBar {
                timestamp: spot_ohlcv.timestamps[i],
                open,
                high: premium.max(open),
                low: premium.min(open),
                close: premium,
                volume: spot_ohlcv.volume[i],
            };
            let mut input = StepInput { entry: entries[i], exit: exits[i], ..StepInput::default() };
            if entries[i] {
                match self.config.size_type {
                    SizeType::Contracts(lots) => input.units = Some(lots as f64 * lot),
                    SizeType::Percent(fraction) | SizeType::RiskPercent(fraction) => {
                        input.size_mult = Some(fraction)
                    }
                    SizeType::Notional(value) => {
                        let free = book.free_capital();
                        if free > 0.0 {
                            input.size_mult = Some(value / free);
                        } else {
                            input.entry = false;
                        }
                    }
                }
            }
            book.apply(&bar_entry(0, i, bar), None, input);
        }

        book.finish().result
    }
}

#[cfg(test)]
#[path = "options_tests.rs"]
mod tests;
