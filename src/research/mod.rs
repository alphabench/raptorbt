//! Research numerics: the deflated Sharpe ratio, the stationary bootstrap, the
//! "without the best days" check, walk-forward windows and parameter designs.
//!
//! Every draw comes from [`SplitMix64`] seeded from a declared seed, and every sum is
//! taken with the crate's own correctly rounded sum (as Python's `math.fsum`), so the
//! numbers match a Python reference on the same inputs to 1e-12 (most of them exactly). The sweep that runs a strategy
//! over a design and a set of windows is Python orchestration in
//! `python/raptorbt/research.py`; it calls these.

pub mod design;
pub mod splitmix64;
pub mod stats;
pub mod walk_forward;

pub use design::{Design, ParamDraw, ParamSpace};
pub use splitmix64::SplitMix64;
pub use stats::{
    deflated_sharpe, stationary_bootstrap, without_best, Bootstrap, DeflatedSharpe, Luck,
};
pub use walk_forward::{WalkForward, WindowRange};
