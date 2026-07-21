//! Portfolio simulation engine for RaptorBT.

pub mod allocation;
pub mod engine;
pub mod kernel;
pub mod monte_carlo;
pub mod position;
pub mod risk;

pub use allocation::{AllocationStrategy, CapitalAllocator};
pub use engine::PortfolioEngine;
pub use kernel::{EngineEvent, EngineKernel, KernelBar, StepInput};
pub use monte_carlo::{simulate_portfolio_forward, MonteCarloConfig, MonteCarloResult};
pub use position::PositionManager;
pub use risk::{RejectReason, RiskGate};
