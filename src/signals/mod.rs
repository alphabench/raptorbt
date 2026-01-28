//! Signal processing for RaptorBT.
//!
//! This module handles signal cleaning, synchronization, and expression evaluation.

pub mod expression;
pub mod processor;
pub mod synchronizer;

pub use processor::SignalProcessor;
pub use synchronizer::{SignalSynchronizer, SyncMode};
