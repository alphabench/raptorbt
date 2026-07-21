//! Bar aggregation specifications.
//!
//! The full unit enum is declared up front so the *type* surface is stable
//! across 0.5.x even while implementations phase in; constructing a builder
//! for an unimplemented unit returns [`SpecError::Unimplemented`] rather
//! than silently degrading.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// What one aggregation step counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregationUnit {
    // Time-based (implemented).
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    // Calendar months/years are irregular; timestamps are ns epoch, so
    // these use civil-calendar arithmetic on UTC.
    Month,
    Year,
    // Count-based (implemented).
    /// N source records (ticks or bars).
    Tick,
    /// N units of traded volume.
    Volume,
    /// N units of traded value (price * volume).
    Value,
    // Declared for spec stability; builders arrive in later 0.5.x releases.
    Renko,
    TickImbalance,
    TickRuns,
    VolumeImbalance,
    VolumeRuns,
    ValueImbalance,
    ValueRuns,
}

impl AggregationUnit {
    /// Parse the Python-facing unit string.
    pub fn parse(s: &str) -> Result<Self, SpecError> {
        Ok(match s {
            "ms" | "millisecond" => AggregationUnit::Millisecond,
            "s" | "sec" | "second" => AggregationUnit::Second,
            "m" | "min" | "minute" => AggregationUnit::Minute,
            "h" | "hour" => AggregationUnit::Hour,
            "d" | "day" => AggregationUnit::Day,
            "w" | "week" => AggregationUnit::Week,
            "month" => AggregationUnit::Month,
            "y" | "year" => AggregationUnit::Year,
            "tick" => AggregationUnit::Tick,
            "volume" => AggregationUnit::Volume,
            "value" => AggregationUnit::Value,
            "renko" => AggregationUnit::Renko,
            "tick_imbalance" => AggregationUnit::TickImbalance,
            "tick_runs" => AggregationUnit::TickRuns,
            "volume_imbalance" => AggregationUnit::VolumeImbalance,
            "volume_runs" => AggregationUnit::VolumeRuns,
            "value_imbalance" => AggregationUnit::ValueImbalance,
            "value_runs" => AggregationUnit::ValueRuns,
            other => return Err(SpecError::UnknownUnit(other.to_string())),
        })
    }

    /// Nanoseconds per unit for fixed-duration time units.
    pub fn fixed_ns(&self) -> Option<i64> {
        match self {
            AggregationUnit::Millisecond => Some(1_000_000),
            AggregationUnit::Second => Some(1_000_000_000),
            AggregationUnit::Minute => Some(60_000_000_000),
            AggregationUnit::Hour => Some(3_600_000_000_000),
            AggregationUnit::Day => Some(86_400_000_000_000),
            AggregationUnit::Week => Some(7 * 86_400_000_000_000),
            _ => None,
        }
    }
}

/// One aggregation: `step` × `unit` (e.g. 5 × Minute).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BarSpec {
    pub step: u32,
    pub unit: AggregationUnit,
}

impl BarSpec {
    pub fn new(step: u32, unit: AggregationUnit) -> Result<Self, SpecError> {
        if step == 0 {
            return Err(SpecError::ZeroStep);
        }
        Ok(Self { step, unit })
    }
}

/// Errors constructing or using a bar spec.
#[derive(Debug, Error, PartialEq)]
pub enum SpecError {
    #[error("unknown aggregation unit {0:?}")]
    UnknownUnit(String),
    #[error("aggregation step must be >= 1")]
    ZeroStep,
    #[error("aggregation unit {0:?} is not implemented yet")]
    Unimplemented(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases() {
        assert_eq!(AggregationUnit::parse("m").unwrap(), AggregationUnit::Minute);
        assert_eq!(AggregationUnit::parse("minute").unwrap(), AggregationUnit::Minute);
        assert_eq!(AggregationUnit::parse("volume").unwrap(), AggregationUnit::Volume);
        assert!(AggregationUnit::parse("fortnight").is_err());
    }

    #[test]
    fn zero_step_refused() {
        assert_eq!(BarSpec::new(0, AggregationUnit::Minute), Err(SpecError::ZeroStep));
    }
}
