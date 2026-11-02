//! Order execution simulation for RaptorBT.

pub mod algos;
pub mod costs;
pub mod fees;
pub mod fill;
pub mod indian_costs;
pub mod orders;
pub mod queue;
pub mod slippage;

pub use algos::{AlgoEngine, AlgoError, AlgoSchedule, ExecAlgorithm, PendingSlice};
pub use costs::{FeeBreakdown, FeeRates, FeeRow, FeeSchedule, FeeSegment, RateCard};
pub use fees::FeeModel;
pub use fill::{FillModel, FillPrice};
pub use indian_costs::{calculate_side, Segment};
pub use queue::{QueueTracker, QueueVerdict};
pub use slippage::SlippageModel;
