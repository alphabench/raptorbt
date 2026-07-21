//! Order objects, state machine, and resting-order matching.
//!
//! New in 0.5.0 for the class-based strategy contract: strategies submit
//! [`Order`]s (market, limit, stop-market, stop-limit) with a time-in-force,
//! and the kernel matches resting ones against each incoming bar. The
//! signal-array path does not use this module and is unaffected by it.

pub mod matching;
pub mod order;

pub use matching::{MatchOutcome, OrderEngine};
pub use order::{Order, OrderKind, OrderSide, OrderStatus, QtySpec, TimeInForce, TrailOffset};
