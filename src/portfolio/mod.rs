//! Portfolio simulation engine for RaptorBT.

pub mod allocation;
pub mod covariance;
pub mod engine;
pub mod errors;
pub mod factor_panel;
pub mod kernel;
mod kernel_orders;
pub mod ledger;
pub mod monte_carlo;
pub mod optimize;
pub mod position;
pub mod rebalance;
pub mod risk;
pub mod risk_contrib;
pub mod runner;
pub mod session;

pub use allocation::{AllocationStrategy, CapitalAllocator};
pub use covariance::{chol_strict, ledoit_wolf, RiskModel};
pub use errors::PortfolioMathError;
pub use factor_panel::{
    composite_scores, momentum_panel, rank_ic, rank_panel, winsorize_panel, zscore_panel, RankIc,
};
pub use optimize::{optimize_long_only, OptimizationResult, OptimizerConfig};
pub use rebalance::{
    simulate_rebalance_policy, RebalanceConfig, RebalancePolicy, RebalanceSimResult,
};
pub use risk_contrib::{risk_contributions, RiskContributions};
pub use engine::PortfolioEngine;
pub use kernel::{EngineEvent, EngineKernel, KernelBar, PositionSnapshot, StepInput};
pub use ledger::{ManagedPosition, PositionLedger, PositionPolicy};
pub use monte_carlo::{simulate_portfolio_forward, MonteCarloConfig, MonteCarloResult};
pub use position::PositionManager;
pub use risk::{RejectReason, RiskGate};
pub use runner::SingleRunner;
pub use session::{EventSession, InstrumentOutcome, ScheduleEntry};
