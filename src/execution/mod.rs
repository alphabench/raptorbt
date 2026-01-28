//! Order execution simulation for RaptorBT.

pub mod fees;
pub mod fill;
pub mod slippage;

pub use fees::FeeModel;
pub use fill::{FillModel, FillPrice};
pub use slippage::SlippageModel;
