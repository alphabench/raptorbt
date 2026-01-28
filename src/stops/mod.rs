//! Stop-loss and take-profit mechanisms for RaptorBT.

pub mod atr;
pub mod fixed;
pub mod trailing;

pub use atr::AtrStop;
pub use fixed::FixedStop;
pub use trailing::TrailingStop;

use crate::core::types::{Direction, Price};

/// Stop-loss calculator trait.
pub trait StopCalculator {
    /// Calculate stop price for a new position.
    fn calculate_stop(&self, entry_price: Price, direction: Direction) -> Option<Price>;

    /// Update stop price for trailing stops.
    fn update_stop(
        &self,
        current_stop: Option<Price>,
        current_price: Price,
        high: Price,
        low: Price,
        direction: Direction,
    ) -> Option<Price>;
}

/// Take-profit calculator trait.
pub trait TargetCalculator {
    /// Calculate target price for a new position.
    fn calculate_target(
        &self,
        entry_price: Price,
        stop_price: Option<Price>,
        direction: Direction,
    ) -> Option<Price>;
}
