//! Zero-copy numpy array interface.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Convert numpy array to Vec<f64>.
pub fn numpy_to_vec_f64(arr: PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_slice().unwrap().to_vec()
}

/// Convert numpy array to Vec<i64>.
pub fn numpy_to_vec_i64(arr: PyReadonlyArray1<i64>) -> Vec<i64> {
    arr.as_slice().unwrap().to_vec()
}

/// Convert numpy bool array to Vec<bool>.
pub fn numpy_to_vec_bool(arr: PyReadonlyArray1<bool>) -> Vec<bool> {
    arr.as_slice().unwrap().to_vec()
}

/// Convert Vec<f64> to numpy array.
pub fn vec_to_numpy_f64<'py>(py: Python<'py>, vec: Vec<f64>) -> &'py PyArray1<f64> {
    PyArray1::from_vec(py, vec)
}

/// Convert Vec<i64> to numpy array.
pub fn vec_to_numpy_i64<'py>(py: Python<'py>, vec: Vec<i64>) -> &'py PyArray1<i64> {
    PyArray1::from_vec(py, vec)
}

/// Convert Vec<bool> to numpy array.
pub fn vec_to_numpy_bool<'py>(py: Python<'py>, vec: Vec<bool>) -> &'py PyArray1<bool> {
    PyArray1::from_vec(py, vec)
}
