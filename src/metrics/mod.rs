//! Performance metrics for RaptorBT.

pub mod annualization;
pub mod drawdown;
pub mod streaming;
pub mod trade_stats;

pub use annualization::{elapsed_years, infer_periods_per_year, resolve_periods_per_year};
pub use drawdown::DrawdownTracker;
pub use streaming::StreamingMetrics;
pub use trade_stats::TradeStatistics;
