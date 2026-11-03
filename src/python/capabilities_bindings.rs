//! The engine-side half of `raptorbt.capabilities()`.

use pyo3::prelude::*;

/// The engine's capability manifest as JSON text; `raptorbt.capabilities()`
/// parses it and adds what the Python layer can see.
#[pyfunction]
pub fn capabilities_json() -> String {
    crate::capabilities::manifest().to_string()
}
