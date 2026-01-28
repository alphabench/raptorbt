//! Strategy implementations for different backtest types.

pub mod basket;
pub mod multi;
pub mod options;
pub mod pairs;
pub mod single;

pub use basket::BasketBacktest;
pub use multi::MultiStrategyBacktest;
pub use options::OptionsBacktest;
pub use pairs::PairsBacktest;
pub use single::SingleBacktest;
