//! Configuration types for multi-leg options spreads.
//!
//! The shapes a caller supplies -- what kind of spread, which legs, and the
//! thresholds that close it early -- separated from the backtest loop that
//! consumes them in `spreads.rs`. Re-exported from there, so every existing
//! `strategies::spreads::{...}` import keeps working.

use crate::core::types::BacktestConfig;
use serde::{Deserialize, Serialize};

/// Spread type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpreadType {
    Straddle,
    Strangle,
    VerticalCall,
    VerticalPut,
    IronCondor,
    IronButterfly,
    ButterflyCall,
    ButterflyPut,
    Calendar,
    Diagonal,
    LongCall,
    LongPut,
    NakedCall,
    NakedPut,
    Custom,
}

/// Option type for a leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    /// Parse a broker option-type code (`CE`/`CALL`/`C`, `PE`/`PUT`/`P`).
    ///
    /// Case-insensitive. Returns `None` for anything else rather than
    /// guessing, because defaulting an unrecognised code to `Call` would
    /// price a put as a call.
    pub fn from_code(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "CE" | "CALL" | "C" => Some(OptionType::Call),
            "PE" | "PUT" | "P" => Some(OptionType::Put),
            _ => None,
        }
    }
}

impl std::str::FromStr for OptionType {
    type Err = ();

    /// Enables `"CE".parse::<OptionType>()`.
    ///
    /// Previously an inherent `from_str` shadowed this trait method, so
    /// `.parse()` did not work while `OptionType::from_str` did -- the kind of
    /// asymmetry that reads as a bug at the call site.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_code(s).ok_or(())
    }
}

/// Configuration for a single leg of a spread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegConfig {
    /// Option type (Call or Put).
    pub option_type: OptionType,
    /// Strike price.
    pub strike: f64,
    /// Position quantity (+1 long, -1 short).
    pub quantity: i32,
    /// Lot size for the option.
    pub lot_size: usize,
}

impl LegConfig {
    pub fn new(option_type: OptionType, strike: f64, quantity: i32, lot_size: usize) -> Self {
        Self { option_type, strike, quantity, lot_size }
    }

    /// Check if this is a long position.
    pub fn is_long(&self) -> bool {
        self.quantity > 0
    }

    /// Check if this is a short position.
    pub fn is_short(&self) -> bool {
        self.quantity < 0
    }
}

/// Configuration for spread backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadConfig {
    /// Base backtest configuration.
    pub base: BacktestConfig,
    /// Spread type.
    pub spread_type: SpreadType,
    /// Leg configurations.
    pub leg_configs: Vec<LegConfig>,
    /// Maximum loss threshold (optional, for early exit).
    pub max_loss: Option<f64>,
    /// Target profit threshold (optional, for early exit).
    pub target_profit: Option<f64>,
    /// Expiry timestamp for each leg, in nanoseconds. Optional.
    ///
    /// Matched to `leg_configs` by position, and a list of a different length
    /// is refused outright -- it would otherwise settle the wrong leg or
    /// leave the trailing legs immortal, silently.
    ///
    /// Each leg settles on its own date and the survivors keep marking, so a
    /// calendar or diagonal spread runs to its far expiry. The structure
    /// closes when the last leg goes.
    ///
    /// **The premium series must carry the leg's settlement value at and
    /// after its expiry.** The engine settles a leg at whatever its series
    /// reads on that bar and then freezes it; it does not compute intrinsic
    /// value, and it never invents a price. A caller settling options against
    /// the underlying computes intrinsic itself and writes it into the series
    /// before calling in.
    pub leg_expiry_timestamps: Option<Vec<i64>>,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            spread_type: SpreadType::Custom,
            leg_configs: Vec::new(),
            max_loss: None,
            target_profit: None,
            leg_expiry_timestamps: None,
        }
    }
}

/// Convenience function to create a straddle spread config.
pub fn create_straddle_config(
    base: BacktestConfig,
    strike: f64,
    lot_size: usize,
    short: bool,
) -> SpreadConfig {
    let quantity = if short { -1 } else { 1 };
    SpreadConfig {
        base,
        spread_type: SpreadType::Straddle,
        leg_configs: vec![
            LegConfig::new(OptionType::Call, strike, quantity, lot_size),
            LegConfig::new(OptionType::Put, strike, quantity, lot_size),
        ],
        ..Default::default()
    }
}

/// Convenience function to create a strangle spread config.
pub fn create_strangle_config(
    base: BacktestConfig,
    call_strike: f64,
    put_strike: f64,
    lot_size: usize,
    short: bool,
) -> SpreadConfig {
    let quantity = if short { -1 } else { 1 };
    SpreadConfig {
        base,
        spread_type: SpreadType::Strangle,
        leg_configs: vec![
            LegConfig::new(OptionType::Call, call_strike, quantity, lot_size),
            LegConfig::new(OptionType::Put, put_strike, quantity, lot_size),
        ],
        ..Default::default()
    }
}

/// Convenience function to create an iron condor spread config.
pub fn create_iron_condor_config(
    base: BacktestConfig,
    short_put_strike: f64,
    long_put_strike: f64,
    short_call_strike: f64,
    long_call_strike: f64,
    lot_size: usize,
) -> SpreadConfig {
    SpreadConfig {
        base,
        spread_type: SpreadType::IronCondor,
        leg_configs: vec![
            LegConfig::new(OptionType::Put, short_put_strike, -1, lot_size),
            LegConfig::new(OptionType::Put, long_put_strike, 1, lot_size),
            LegConfig::new(OptionType::Call, short_call_strike, -1, lot_size),
            LegConfig::new(OptionType::Call, long_call_strike, 1, lot_size),
        ],
        ..Default::default()
    }
}

/// Convenience function to create a vertical spread config.
pub fn create_vertical_spread_config(
    base: BacktestConfig,
    option_type: OptionType,
    long_strike: f64,
    short_strike: f64,
    lot_size: usize,
) -> SpreadConfig {
    let spread_type = match option_type {
        OptionType::Call => SpreadType::VerticalCall,
        OptionType::Put => SpreadType::VerticalPut,
    };

    SpreadConfig {
        base,
        spread_type,
        leg_configs: vec![
            LegConfig::new(option_type, long_strike, 1, lot_size),
            LegConfig::new(option_type, short_strike, -1, lot_size),
        ],
        ..Default::default()
    }
}
