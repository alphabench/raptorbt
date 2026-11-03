//! Shared plumbing for the runs that decide on whole signal arrays and trade
//! one or more legs: basket, pairs, options, spreads and ticks.
//!
//! None of them simulates execution. Each registers its legs in a
//! [`PortfolioBook`] (the account, the per-instrument kernels and the
//! sampled curves every other runner uses) and feeds it one schedule entry
//! per leg and bar, so fills, fees, slippage, margin, risk limits, expiry and
//! fixed-point cash follow one set of rules. What stays here is what is
//! particular to the structure: which legs enter together, how a leg is
//! sized, when the group exits.

use crate::accounts::AccountMode;
use crate::core::types::{BacktestConfig, OhlcvData};
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::KernelBar;
use crate::portfolio::session::{ScheduleData, ScheduleEntry};

/// The account these runs trade in: fully funded, unless the config chose a
/// margin model, which only a margin account reads.
pub(crate) fn account_mode(config: &BacktestConfig) -> AccountMode {
    if config.margin_model.is_some() {
        AccountMode::Margin { leverage: 1.0 }
    } else {
        AccountMode::Cash
    }
}

/// One bar of an OHLCV series as the kernel takes it.
#[inline]
pub(crate) fn kernel_bar(ohlcv: &OhlcvData, i: usize) -> KernelBar {
    KernelBar {
        timestamp: ohlcv.timestamps[i],
        open: ohlcv.open[i],
        high: ohlcv.high[i],
        low: ohlcv.low[i],
        close: ohlcv.close[i],
        volume: ohlcv.volume[i],
    }
}

/// The schedule entry for leg `instrument`'s `local_idx`-th bar.
#[inline]
pub(crate) fn bar_entry(instrument: usize, local_idx: usize, bar: KernelBar) -> ScheduleEntry {
    ScheduleEntry { instrument, local_idx, data: ScheduleData::Bar(bar) }
}

/// Whether leg `k` holds nothing and has nothing waiting to fill.
#[inline]
pub(crate) fn is_flat(book: &PortfolioBook, k: usize) -> bool {
    let kernel = book.kernel(k);
    !kernel.is_in_position() && !kernel.has_pending_entry()
}

/// Whether every leg is flat.
pub(crate) fn all_flat(book: &PortfolioBook, legs: usize) -> bool {
    (0..legs).all(|k| is_flat(book, k))
}

/// Whether the bar just applied leaves a group half entered because a risk
/// limit or the account refused one leg's entry: some leg holds a position,
/// another does not, and `events` carry a refused entry. The caller closes
/// the legs that did enter on the next bar, so a pair or basket is held
/// whole or not at all.
pub(crate) fn half_entered(
    book: &PortfolioBook,
    legs: usize,
    events: &[crate::portfolio::kernel::EngineEvent],
) -> bool {
    let refused = events
        .iter()
        .any(|event| matches!(event, crate::portfolio::kernel::EngineEvent::EntryRejected { .. }));
    let held = (0..legs).filter(|&k| book.kernel(k).is_in_position()).count();
    refused && held > 0 && held < legs
}
