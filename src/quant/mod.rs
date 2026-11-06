//! The expression engine: kernels that run bit-identically over a whole column or one bar at
//! a time, the causal window primitives they share, and the registry that names them.

pub mod eval;
pub mod fns;
pub mod kernel;
pub mod lang;
pub mod registry;
pub mod window;
