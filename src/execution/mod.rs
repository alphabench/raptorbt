//! Order execution simulation for RaptorBT.

pub mod fees;
pub mod fill;
pub mod indian_costs;
pub mod algos;
pub mod orders;
pub mod queue;
pub mod slippage;

pub use fees::FeeModel;
pub use fill::{FillModel, FillPrice};
pub use indian_costs::{calculate_side, CostSchedule, FeeBreakdown, Segment};
pub use slippage::SlippageModel;
pub use queue::{QueueTracker, QueueVerdict};
pub use algos::{AlgoEngine, AlgoError, AlgoSchedule, ExecAlgorithm, PendingSlice};
