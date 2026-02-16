//! Portfolio simulation engine for RaptorBT.

pub mod allocation;
pub mod engine;
pub mod monte_carlo;
pub mod position;

pub use allocation::{AllocationStrategy, CapitalAllocator};
pub use engine::PortfolioEngine;
pub use monte_carlo::{simulate_portfolio_forward, MonteCarloConfig, MonteCarloResult};
pub use position::PositionManager;
