//! Costs as data: the rates that price a fill, the rows that say which rates
//! apply where, and the rate card document that carries them.
//!
//! One computation serves every caller. [`FeeRates::charge_fill`] produces the
//! itemised [`FeeBreakdown`]; the kernel charges its `total()` to equity, so
//! the breakdown a trade reports and the money taken are the same numbers.

pub mod card;
pub mod rates;
pub mod rounding;
pub mod schedule;

pub use card::{percent_text, RateCard};
pub use rates::{
    BrokeragePer, DayFee, DaySide, FeeBreakdown, FeeRates, GstBase, Rounding, TaxKind,
};
pub use rounding::Round;
pub use schedule::{
    exchange_rates, FeeRow, FeeSchedule, FeeSegment, DAY_FEE_DELIVERY_SELL, EXCHANGE_PRESETS,
    EXCHANGE_SOURCE,
};
