//! Zero-copy numpy array interface.

use std::borrow::Cow;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Convert a 2-D numpy array to a row-major flat Vec plus its shape.
///
/// Iterates the ndarray view in logical order, so a non-C-contiguous input
/// (e.g. a transposed slice) is copied correctly rather than rejected.
pub fn numpy_to_vec2_f64(arr: PyReadonlyArray2<f64>) -> (Vec<f64>, usize, usize) {
    let view = arr.as_array();
    let (rows, cols) = (view.nrows(), view.ncols());
    (view.iter().cloned().collect(), rows, cols)
}

/// Convert numpy array to Vec<f64>.
///
/// Iterates the view in logical order, so a strided input (a column of a
/// transposed matrix, `a[::2]`) is copied correctly rather than refused.
pub fn numpy_to_vec_f64(arr: PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

/// Convert numpy array to Vec<i64>.
pub fn numpy_to_vec_i64(arr: PyReadonlyArray1<i64>) -> Vec<i64> {
    arr.as_array().iter().copied().collect()
}

/// Borrow a numpy `f64` array when it is contiguous, copy it when it is not
/// (a strided view such as `a[::2]` or a column of a transposed matrix).
pub fn numpy_cow_f64<'a>(arr: &'a PyReadonlyArray1<'_, f64>) -> Cow<'a, [f64]> {
    match arr.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

/// Borrow a numpy `i64` array when it is contiguous, copy it when it is not.
pub fn numpy_cow_i64<'a>(arr: &'a PyReadonlyArray1<'_, i64>) -> Cow<'a, [i64]> {
    match arr.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

/// Convert numpy bool array to Vec<bool>.
pub fn numpy_to_vec_bool(arr: PyReadonlyArray1<bool>) -> Vec<bool> {
    arr.as_array().iter().copied().collect()
}

/// An owned-by-Python 1-D `f64` array, as returned to Python.
pub type F64Array<'py> = Bound<'py, PyArray1<f64>>;

/// Convert Vec<f64> to numpy array.
pub fn vec_to_numpy_f64<'py>(py: Python<'py>, vec: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, vec)
}

/// Convert Vec<i64> to numpy array.
pub fn vec_to_numpy_i64<'py>(py: Python<'py>, vec: Vec<i64>) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, vec)
}

/// Convert Vec<bool> to numpy array.
pub fn vec_to_numpy_bool<'py>(py: Python<'py>, vec: Vec<bool>) -> Bound<'py, PyArray1<bool>> {
    PyArray1::from_vec(py, vec)
}
