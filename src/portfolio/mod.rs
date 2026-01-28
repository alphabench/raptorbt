//! Portfolio simulation engine for RaptorBT.

pub mod allocation;
pub mod engine;
pub mod position;

pub use allocation::{AllocationStrategy, CapitalAllocator};
pub use engine::PortfolioEngine;
pub use position::PositionManager;
