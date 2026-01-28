//! Technical indicators for RaptorBT.
//!
//! All indicators are implemented as pure functions that take slice inputs
//! and return Vec outputs. NaN values are used for the warmup period.

pub mod momentum;
pub mod strength;
pub mod trend;
pub mod volatility;
pub mod volume;

pub use momentum::{macd, rsi, stochastic, MacdResult, StochasticResult};
pub use strength::adx;
pub use trend::{ema, sma, supertrend, SupertrendResult};
pub use volatility::{atr, bollinger_bands, BollingerBandsResult};
pub use volume::{obv, vwap};
