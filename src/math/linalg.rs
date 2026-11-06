//! Dense linear algebra for the statistics: a small row-major matrix and the factorisations the
//! tests need (Householder QR with column pivoting, Cholesky, cyclic Jacobi symmetric
//! eigendecomposition, one-sided Jacobi SVD) with the solve and least squares built on them.
//!
//! The problems are small (Johansen at most 12 columns, PCA at most 40, ADF about 32 lags at
//! 5,000 bars), so these favour accuracy and a fixed order of operations over speed: the same
//! input gives the same bits on every machine, and nothing here calls a platform BLAS.
//!
//! Sources: Golub and Van Loan, *Matrix Computations* (Householder QR with column pivoting,
//! 5.4.1; symmetric Jacobi, 8.5; one-sided Jacobi SVD after Hestenes, 8.6) and Higham,
//! *Accuracy and Stability of Numerical Algorithms* for the error bounds the tests state.
//!
//! Numerically degenerate input is a [`LinAlgError`], never a panic and never a NaN result:
//! zero-size, non-square, non-finite, not symmetric, not positive definite, singular.
//!
//! Bounds, with `eps = f64::EPSILON`, that the tests check:
//! - Householder QR: `|Q'Q - I|_F` and `|A P - Q R|_F / |A|_F` are `O(m n eps)` whatever the
//!   condition of `A`.
//! - Jacobi eigendecomposition: `|V'V - I|_F = O(n eps)`; each eigenvalue is within
//!   `O(n eps |A|_2)` of the true one.
//! - one-sided Jacobi SVD: `|A - U S V'|_F = O(n eps |A|_F)` and `|V'V - I|_F = O(n eps)`;
//!   `|U'U - I|_F = O(n eps cond(A))` (the columns of `U` for singular values at or below
//!   `max(m, n) eps s_max` are replaced by an orthonormal completion, so `U` always has
//!   orthonormal columns).
//! - solving `A x = b` by pivoted QR: the relative error of `x` is `O(n eps cond(A))`.

use std::ops::{Index, IndexMut};

use thiserror::Error;

const EPS: f64 = f64::EPSILON;
/// Largest asymmetry `max |a_ij - a_ji|` accepted, relative to `max |a_ij|`. Inputs built as
/// `X'X` are symmetric to a few eps; this is wide enough for a covariance assembled two ways and
/// narrow enough to refuse a matrix that is not symmetric at all.
const SYMMETRY_RTOL: f64 = 1e-10;
const MAX_SWEEPS: usize = 60;

/// Why a matrix operation refused its input or failed.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LinAlgError {
    #[error("{op}: the matrix is empty ({rows} x {cols})")]
    ZeroSize { op: &'static str, rows: usize, cols: usize },
    #[error("{op}: the matrix is not square ({rows} x {cols})")]
    NotSquare { op: &'static str, rows: usize, cols: usize },
    #[error("{op}: a value is NaN or infinite")]
    NonFinite { op: &'static str },
    #[error("{op}: the matrix is not symmetric (asymmetry {asymmetry:e} against scale {scale:e})")]
    NotSymmetric { op: &'static str, asymmetry: f64, scale: f64 },
    #[error("cholesky: not positive definite at pivot {pivot} (value {value:e}, floor {floor:e})")]
    NotPositiveDefinite { pivot: usize, value: f64, floor: f64 },
    #[error("{op}: singular to working precision (rank {rank} of {size})")]
    Singular { op: &'static str, rank: usize, size: usize },
    #[error("{op}: {what} has length {found}, expected {expected}")]
    DimensionMismatch { op: &'static str, what: &'static str, expected: usize, found: usize },
    #[error("{op}: no convergence in {sweeps} sweeps")]
    NoConvergence { op: &'static str, sweeps: usize },
}

/// A dense matrix of finite numbers, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

fn require_size(op: &'static str, rows: usize, cols: usize) -> Result<(), LinAlgError> {
    if rows == 0 || cols == 0 {
        return Err(LinAlgError::ZeroSize { op, rows, cols });
    }
    Ok(())
}

fn require_finite(op: &'static str, values: &[f64]) -> Result<(), LinAlgError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(LinAlgError::NonFinite { op })
    }
}

impl Mat {
    /// A matrix of zeros; a zero dimension is refused.
    pub fn zeros(rows: usize, cols: usize) -> Result<Mat, LinAlgError> {
        require_size("zeros", rows, cols)?;
        Ok(Mat { rows, cols, data: vec![0.0; rows * cols] })
    }

    /// The `n x n` identity.
    pub fn identity(n: usize) -> Result<Mat, LinAlgError> {
        let mut m = Mat::zeros(n, n)?;
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        Ok(m)
    }

    /// Row-major `data` of length `rows * cols`, all finite.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Mat, LinAlgError> {
        require_size("from_vec", rows, cols)?;
        if data.len() != rows * cols {
            return Err(LinAlgError::DimensionMismatch {
                op: "from_vec",
                what: "data",
                expected: rows * cols,
                found: data.len(),
            });
        }
        require_finite("from_vec", &data)?;
        Ok(Mat { rows, cols, data })
    }

    /// From rows of equal length, all finite.
    pub fn from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Mat, LinAlgError> {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, |r| r.as_ref().len());
        require_size("from_rows", nrows, ncols)?;
        let mut data = Vec::with_capacity(nrows * ncols);
        for r in rows {
            let r = r.as_ref();
            if r.len() != ncols {
                return Err(LinAlgError::DimensionMismatch {
                    op: "from_rows",
                    what: "row",
                    expected: ncols,
                    found: r.len(),
                });
            }
            data.extend_from_slice(r);
        }
        Mat::from_vec(nrows, ncols, data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The elements, row-major.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[(i, j)]).collect()
    }

    pub fn transpose(&self) -> Mat {
        let mut t = Mat { rows: self.cols, cols: self.rows, data: vec![0.0; self.data.len()] };
        for i in 0..self.rows {
            for j in 0..self.cols {
                t[(j, i)] = self[(i, j)];
            }
        }
        t
    }

    /// `self * other`, accumulated in a fixed order.
    pub fn matmul(&self, other: &Mat) -> Result<Mat, LinAlgError> {
        if self.cols != other.rows {
            return Err(LinAlgError::DimensionMismatch {
                op: "matmul",
                what: "right operand rows",
                expected: self.cols,
                found: other.rows,
            });
        }
        let mut out =
            Mat { rows: self.rows, cols: other.cols, data: vec![0.0; self.rows * other.cols] };
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                for j in 0..other.cols {
                    out[(i, j)] += a * other[(k, j)];
                }
            }
        }
        require_finite("matmul", &out.data)?;
        Ok(out)
    }

    /// `self * v`.
    pub fn matvec(&self, v: &[f64]) -> Result<Vec<f64>, LinAlgError> {
        if v.len() != self.cols {
            return Err(LinAlgError::DimensionMismatch {
                op: "matvec",
                what: "vector",
                expected: self.cols,
                found: v.len(),
            });
        }
        let out: Vec<f64> =
            (0..self.rows).map(|i| self.row(i).iter().zip(v).map(|(a, b)| a * b).sum()).collect();
        require_finite("matvec", &out)?;
        Ok(out)
    }

    /// Frobenius norm.
    pub fn frobenius(&self) -> f64 {
        norm2(&self.data)
    }

    /// Largest absolute element.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    fn require_square(&self, op: &'static str) -> Result<(), LinAlgError> {
        if self.rows != self.cols {
            return Err(LinAlgError::NotSquare { op, rows: self.rows, cols: self.cols });
        }
        Ok(())
    }

    /// Refuses a matrix whose halves differ by more than [`SYMMETRY_RTOL`] of its scale.
    fn require_symmetric(&self, op: &'static str) -> Result<(), LinAlgError> {
        self.require_square(op)?;
        let scale = self.max_abs();
        let mut asymmetry = 0.0f64;
        for i in 0..self.rows {
            for j in 0..i {
                asymmetry = asymmetry.max((self[(i, j)] - self[(j, i)]).abs());
            }
        }
        if asymmetry > SYMMETRY_RTOL * scale {
            return Err(LinAlgError::NotSymmetric { op, asymmetry, scale });
        }
        Ok(())
    }
}

/// Euclidean norm without overflow or underflow of the squares.
pub fn norm2(v: &[f64]) -> f64 {
    let scale = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let ss: f64 = v.iter().map(|x| (x / scale) * (x / scale)).sum();
    scale * ss.sqrt()
}

/// `sqrt(a^2 + b^2)` without overflow, from IEEE operations only (the C library `hypot` is not
/// specified to the last bit).
fn hypot(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    if big == 0.0 {
        return 0.0;
    }
    let r = small / big;
    big * (1.0 + r * r).sqrt()
}

// ------------------------------------------------------------------------------------- QR

/// Householder QR with column pivoting, `A P = Q R`, as LAPACK stores it: `R` on and above the
/// diagonal of `qr`, the Householder vectors (unit first element implied) below it.
#[derive(Debug, Clone)]
pub struct PivotedQr {
    qr: Mat,
    tau: Vec<f64>,
    perm: Vec<usize>,
    rank: usize,
}

/// The least-squares solution of `A x = b`.
#[derive(Debug, Clone, PartialEq)]
pub struct Lstsq {
    /// The basic solution: the columns beyond the numerical rank get coefficient zero.
    pub x: Vec<f64>,
    /// Numerical rank of `A`.
    pub rank: usize,
    /// Sum of squared residuals `|b - A x|^2`.
    pub rss: f64,
}

/// Numerical rank threshold of QR: a diagonal of `R` at or below `max(m, n) eps |R_00|` counts as
/// zero.
fn default_rank_tol(rows: usize, cols: usize) -> f64 {
    rows.max(cols) as f64 * EPS
}

/// Pivoted QR of `a`; the rank is counted with tolerance `max(m, n) eps`.
pub fn qr_pivoted(a: &Mat) -> Result<PivotedQr, LinAlgError> {
    qr_pivoted_tol(a, default_rank_tol(a.rows, a.cols))
}

/// Pivoted QR of `a` with the rank counted as the diagonal entries of `R` above
/// `rank_tol * |R_00|`.
pub fn qr_pivoted_tol(a: &Mat, rank_tol: f64) -> Result<PivotedQr, LinAlgError> {
    let (m, n) = (a.rows, a.cols);
    let k = m.min(n);
    let mut qr = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut tau = vec![0.0; k];
    let mut col = vec![0.0; m];
    for j in 0..k {
        // the remaining column with the largest norm below row j goes to position j; recomputed
        // each step rather than downdated, which cannot drift
        let mut best = j;
        let mut best_norm = -1.0f64;
        for l in j..n {
            for i in j..m {
                col[i - j] = qr[(i, l)];
            }
            let nl = norm2(&col[..m - j]);
            if nl > best_norm {
                best_norm = nl;
                best = l;
            }
        }
        if best != j {
            for i in 0..m {
                let t = qr[(i, j)];
                qr[(i, j)] = qr[(i, best)];
                qr[(i, best)] = t;
            }
            perm.swap(j, best);
        }
        let alpha = qr[(j, j)];
        for i in j + 1..m {
            col[i - j - 1] = qr[(i, j)];
        }
        let xnorm = norm2(&col[..m - j - 1]);
        if xnorm == 0.0 {
            tau[j] = 0.0; // H = I, R_jj = alpha
        } else {
            let beta = -hypot(alpha, xnorm).copysign(alpha);
            tau[j] = (beta - alpha) / beta;
            let scale = 1.0 / (alpha - beta);
            for i in j + 1..m {
                qr[(i, j)] *= scale;
            }
            qr[(j, j)] = beta;
        }
        let t = tau[j];
        if t != 0.0 {
            for c in j + 1..n {
                let mut w = qr[(j, c)];
                for i in j + 1..m {
                    w += qr[(i, j)] * qr[(i, c)];
                }
                let tw = t * w;
                qr[(j, c)] -= tw;
                for i in j + 1..m {
                    let v = qr[(i, j)];
                    qr[(i, c)] -= tw * v;
                }
            }
        }
    }
    require_finite("qr_pivoted", &qr.data)?;
    let r00 = qr[(0, 0)].abs();
    let floor = rank_tol * r00;
    let rank =
        if r00 == 0.0 { 0 } else { (0..k).take_while(|&j| qr[(j, j)].abs() > floor).count() };
    Ok(PivotedQr { qr, tau, perm, rank })
}

impl PivotedQr {
    /// Numerical rank of the factored matrix.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Column permutation: column `j` of `A P` is column `perm()[j]` of `A`.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// The diagonal of `R`, nonincreasing in magnitude.
    pub fn r_diag(&self) -> Vec<f64> {
        (0..self.tau.len()).map(|j| self.qr[(j, j)]).collect()
    }

    /// `R`, `min(m, n) x n`, upper trapezoidal.
    pub fn r(&self) -> Mat {
        let k = self.tau.len();
        let mut r = Mat { rows: k, cols: self.qr.cols, data: vec![0.0; k * self.qr.cols] };
        for i in 0..k {
            for j in i..self.qr.cols {
                r[(i, j)] = self.qr[(i, j)];
            }
        }
        r
    }

    /// The thin `Q`, `m x min(m, n)`, orthonormal columns.
    pub fn q(&self) -> Mat {
        let (m, k) = (self.qr.rows, self.tau.len());
        let mut e = Mat { rows: m, cols: k, data: vec![0.0; m * k] };
        for j in 0..k {
            e[(j, j)] = 1.0;
        }
        for j in (0..k).rev() {
            let t = self.tau[j];
            if t == 0.0 {
                continue;
            }
            for c in 0..k {
                let mut w = e[(j, c)];
                for i in j + 1..m {
                    w += self.qr[(i, j)] * e[(i, c)];
                }
                let tw = t * w;
                e[(j, c)] -= tw;
                for i in j + 1..m {
                    e[(i, c)] -= tw * self.qr[(i, j)];
                }
            }
        }
        e
    }

    /// `b <- Q' b` in place (`b` has `m` entries).
    pub fn apply_qt(&self, b: &mut [f64]) -> Result<(), LinAlgError> {
        let m = self.qr.rows;
        if b.len() != m {
            return Err(LinAlgError::DimensionMismatch {
                op: "apply_qt",
                what: "vector",
                expected: m,
                found: b.len(),
            });
        }
        for j in 0..self.tau.len() {
            let t = self.tau[j];
            if t == 0.0 {
                continue;
            }
            let mut w = b[j];
            for (i, bi) in b.iter().enumerate().skip(j + 1) {
                w += self.qr[(i, j)] * bi;
            }
            let tw = t * w;
            b[j] -= tw;
            for (i, bi) in b.iter_mut().enumerate().skip(j + 1) {
                *bi -= tw * self.qr[(i, j)];
            }
        }
        Ok(())
    }

    /// The basic least-squares solution of `A x = b`, in the original column order.
    pub fn solve_basic(&self, b: &[f64]) -> Result<Lstsq, LinAlgError> {
        require_finite("lstsq", b)?;
        let mut c = b.to_vec();
        self.apply_qt(&mut c)?;
        let (r, n) = (self.rank, self.qr.cols);
        let mut z = vec![0.0; r];
        for i in (0..r).rev() {
            let mut s = c[i];
            for (j, zj) in z.iter().enumerate().skip(i + 1) {
                s -= self.qr[(i, j)] * zj;
            }
            z[i] = s / self.qr[(i, i)];
        }
        let mut x = vec![0.0; n];
        for (i, zi) in z.iter().enumerate() {
            x[self.perm[i]] = *zi;
        }
        require_finite("lstsq", &x)?;
        let rss = c[r..].iter().map(|v| v * v).sum();
        Ok(Lstsq { x, rank: r, rss })
    }
}

/// Least squares `min |A x - b|` by pivoted QR. A rank-deficient `A` gives a basic solution and
/// reports the rank; the caller decides whether that is acceptable.
pub fn lstsq(a: &Mat, b: &[f64]) -> Result<Lstsq, LinAlgError> {
    if b.len() != a.rows {
        return Err(LinAlgError::DimensionMismatch {
            op: "lstsq",
            what: "right-hand side",
            expected: a.rows,
            found: b.len(),
        });
    }
    qr_pivoted(a)?.solve_basic(b)
}

/// Solves the square system `A x = b`; singular to working precision is an error.
pub fn solve(a: &Mat, b: &[f64]) -> Result<Vec<f64>, LinAlgError> {
    a.require_square("solve")?;
    if b.len() != a.rows {
        return Err(LinAlgError::DimensionMismatch {
            op: "solve",
            what: "right-hand side",
            expected: a.rows,
            found: b.len(),
        });
    }
    let qr = qr_pivoted(a)?;
    if qr.rank < a.rows {
        return Err(LinAlgError::Singular { op: "solve", rank: qr.rank, size: a.rows });
    }
    Ok(qr.solve_basic(b)?.x)
}

/// The inverse of a square matrix by pivoted QR; singular to working precision is an error.
pub fn inverse(a: &Mat) -> Result<Mat, LinAlgError> {
    a.require_square("inverse")?;
    let n = a.rows;
    let qr = qr_pivoted(a)?;
    if qr.rank < n {
        return Err(LinAlgError::Singular { op: "inverse", rank: qr.rank, size: n });
    }
    let mut inv = Mat { rows: n, cols: n, data: vec![0.0; n * n] };
    let mut e = vec![0.0; n];
    for j in 0..n {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[j] = 1.0;
        let x = qr.solve_basic(&e)?.x;
        for i in 0..n {
            inv[(i, j)] = x[i];
        }
    }
    Ok(inv)
}

// ------------------------------------------------------------------------------- Cholesky

/// `A = L L'` with `L` lower triangular.
#[derive(Debug, Clone)]
pub struct Cholesky {
    l: Mat,
}

/// Cholesky factorisation of a symmetric positive definite matrix. The lower triangle is used.
/// A pivot at or below `n eps max_i a_ii` (the floor LAPACK's pivoted Cholesky uses) is refused,
/// so a matrix that is positive definite only in exact arithmetic is not factored.
pub fn cholesky(a: &Mat) -> Result<Cholesky, LinAlgError> {
    a.require_symmetric("cholesky")?;
    let n = a.rows;
    let max_diag = (0..n).fold(0.0f64, |m, i| m.max(a[(i, i)]));
    let floor = n as f64 * EPS * max_diag;
    let mut l = Mat { rows: n, cols: n, data: vec![0.0; n * n] };
    for j in 0..n {
        let mut d = a[(j, j)];
        for k in 0..j {
            d -= l[(j, k)] * l[(j, k)];
        }
        if d <= floor || !d.is_finite() {
            return Err(LinAlgError::NotPositiveDefinite { pivot: j, value: d, floor });
        }
        let ljj = d.sqrt();
        l[(j, j)] = ljj;
        for i in j + 1..n {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = s / ljj;
        }
    }
    require_finite("cholesky", &l.data)?;
    Ok(Cholesky { l })
}

impl Cholesky {
    /// The lower triangular factor.
    pub fn l(&self) -> &Mat {
        &self.l
    }

    /// Solves `A x = b`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, LinAlgError> {
        let n = self.l.rows;
        if b.len() != n {
            return Err(LinAlgError::DimensionMismatch {
                op: "cholesky solve",
                what: "right-hand side",
                expected: n,
                found: b.len(),
            });
        }
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for (k, yk) in y.iter().enumerate().take(i) {
                s -= self.l[(i, k)] * yk;
            }
            y[i] = s / self.l[(i, i)];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for (k, xk) in x.iter().enumerate().skip(i + 1) {
                s -= self.l[(k, i)] * xk;
            }
            x[i] = s / self.l[(i, i)];
        }
        require_finite("cholesky solve", &x)?;
        Ok(x)
    }

    /// `ln det A`.
    pub fn log_det(&self) -> f64 {
        2.0 * (0..self.l.rows).map(|i| self.l[(i, i)].ln()).sum::<f64>()
    }
}

// ------------------------------------------------------------------------------- eigenvalues

/// Eigenvalues (descending) and orthonormal eigenvectors (columns) of a symmetric matrix. The
/// sign of an eigenvector is arbitrary but deterministic.
#[derive(Debug, Clone)]
pub struct SymEigen {
    pub values: Vec<f64>,
    pub vectors: Mat,
}

/// Cyclic Jacobi eigendecomposition. The matrix must be symmetric to [`SYMMETRY_RTOL`]; it is
/// averaged with its transpose first. Jacobi gives small eigenvalues of a positive definite
/// matrix with high relative accuracy.
pub fn sym_eigen(a: &Mat) -> Result<SymEigen, LinAlgError> {
    a.require_symmetric("sym_eigen")?;
    let n = a.rows;
    let mut w = Mat { rows: n, cols: n, data: vec![0.0; n * n] };
    for i in 0..n {
        for j in 0..n {
            w[(i, j)] = 0.5 * (a[(i, j)] + a[(j, i)]);
        }
    }
    let mut v = Mat::identity(n)?;
    let frob = w.frobenius();
    let mut sweeps = 0;
    loop {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += w[(p, q)] * w[(p, q)];
            }
        }
        if (2.0 * off).sqrt() <= EPS * frob {
            break;
        }
        if sweeps == MAX_SWEEPS {
            return Err(LinAlgError::NoConvergence { op: "sym_eigen", sweeps });
        }
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let apq = w[(p, q)];
                if apq == 0.0 {
                    continue;
                }
                let theta = (w[(q, q)] - w[(p, p)]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    1.0 / (2.0 * theta)
                } else {
                    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                    sign / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                w[(p, p)] -= t * apq;
                w[(q, q)] += t * apq;
                w[(p, q)] = 0.0;
                w[(q, p)] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let (arp, arq) = (w[(r, p)], w[(r, q)]);
                    let (np, nq) = (c * arp - s * arq, s * arp + c * arq);
                    w[(r, p)] = np;
                    w[(p, r)] = np;
                    w[(r, q)] = nq;
                    w[(q, r)] = nq;
                }
                for r in 0..n {
                    let (vp, vq) = (v[(r, p)], v[(r, q)]);
                    v[(r, p)] = c * vp - s * vq;
                    v[(r, q)] = s * vp + c * vq;
                }
            }
        }
        sweeps += 1;
    }
    let diag: Vec<f64> = (0..n).map(|i| w[(i, i)]).collect();
    require_finite("sym_eigen", &diag)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| diag[y].total_cmp(&diag[x]));
    let values = order.iter().map(|&i| diag[i]).collect();
    let mut vectors = Mat { rows: n, cols: n, data: vec![0.0; n * n] };
    for (new, &old) in order.iter().enumerate() {
        for r in 0..n {
            vectors[(r, new)] = v[(r, old)];
        }
    }
    Ok(SymEigen { values, vectors })
}

// ----------------------------------------------------------------------------------- SVD

/// Thin SVD `A = U diag(s) V'`, `s` descending: `U` is `m x k`, `V` is `n x k`, `k = min(m, n)`.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Mat,
    pub s: Vec<f64>,
    pub v: Mat,
}

impl Svd {
    /// Singular values above `rcond * s_max`.
    pub fn rank(&self, rcond: f64) -> usize {
        let floor = rcond * self.s[0];
        self.s.iter().filter(|&&x| x > floor).count()
    }

    /// `s_max / s_min`; infinite when the smallest singular value is zero.
    pub fn condition(&self) -> f64 {
        let smin = self.s[self.s.len() - 1];
        if smin == 0.0 {
            f64::INFINITY
        } else {
            self.s[0] / smin
        }
    }
}

/// One-sided (Hestenes) Jacobi SVD.
pub fn svd(a: &Mat) -> Result<Svd, LinAlgError> {
    let (m, n) = (a.rows, a.cols);
    if m < n {
        let t = svd(&a.transpose())?;
        return Ok(Svd { u: t.v, s: t.s, v: t.u });
    }
    let scale = a.max_abs();
    if scale == 0.0 {
        let mut u = Mat { rows: m, cols: n, data: vec![0.0; m * n] };
        complete_columns(&mut u, 0)?;
        return Ok(Svd { u, s: vec![0.0; n], v: Mat::identity(n)? });
    }
    let mut u = Mat { rows: m, cols: n, data: a.data.iter().map(|x| x / scale).collect() };
    let mut v = Mat::identity(n)?;
    // a column whose norm is at rounding level against the largest starting column is a null
    // direction (a duplicate or a combination of the others, cancelled): rotating it against
    // another column only re-normalises noise and never settles, so such a pair counts as
    // orthogonal. The threshold is the rank threshold below, in squared norms.
    let widest = (0..n).map(|j| norm2(&u.col(j))).fold(0.0f64, f64::max);
    let negligible = (m.max(n) as f64 * EPS * widest).powi(2);
    let mut converged = false;
    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (x, y) = (u[(i, p)], u[(i, q)]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma == 0.0
                    || alpha.min(beta) <= negligible
                    || gamma.abs() <= EPS * (alpha * beta).sqrt()
                {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (x, y) = (u[(i, p)], u[(i, q)]);
                    u[(i, p)] = c * x - s * y;
                    u[(i, q)] = s * x + c * y;
                }
                for i in 0..n {
                    let (x, y) = (v[(i, p)], v[(i, q)]);
                    v[(i, p)] = c * x - s * y;
                    v[(i, q)] = s * x + c * y;
                }
            }
        }
        if !rotated {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinAlgError::NoConvergence { op: "svd", sweeps: MAX_SWEEPS });
    }
    let sigma: Vec<f64> = (0..n).map(|j| norm2(&u.col(j))).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| sigma[y].total_cmp(&sigma[x]));
    let mut us = Mat { rows: m, cols: n, data: vec![0.0; m * n] };
    let mut vs = Mat { rows: n, cols: n, data: vec![0.0; n * n] };
    let mut s = Vec::with_capacity(n);
    for (new, &old) in order.iter().enumerate() {
        s.push(sigma[old] * scale);
        for i in 0..m {
            us[(i, new)] = u[(i, old)];
        }
        for i in 0..n {
            vs[(i, new)] = v[(i, old)];
        }
    }
    require_finite("svd", &s)?;
    // the columns of U for a singular value that is zero to working precision cannot be
    // normalised; they are replaced below by an orthonormal completion
    let tol = m.max(n) as f64 * EPS * sigma[order[0]];
    let mut kept = 0;
    for j in 0..n {
        let sj = sigma[order[j]];
        if sj > tol {
            for i in 0..m {
                us[(i, j)] /= sj;
            }
            kept = j + 1;
        } else {
            for i in 0..m {
                us[(i, j)] = 0.0;
            }
        }
    }
    complete_columns(&mut us, kept)?;
    Ok(Svd { u: us, s, v: vs })
}

/// Fills the all-zero columns `from..` of `u` with unit vectors orthogonal to all earlier
/// columns: the first standard basis vector, in index order, that survives two rounds of
/// Gram-Schmidt with norm above 1/2.
fn complete_columns(u: &mut Mat, from: usize) -> Result<(), LinAlgError> {
    let (m, n) = (u.rows, u.cols);
    for j in from..n {
        let mut filled = false;
        for e in 0..m {
            let mut x = vec![0.0; m];
            x[e] = 1.0;
            for _ in 0..2 {
                for k in 0..j {
                    let dot: f64 = (0..m).map(|i| u[(i, k)] * x[i]).sum();
                    for i in 0..m {
                        x[i] -= dot * u[(i, k)];
                    }
                }
            }
            let nx = norm2(&x);
            if nx > 0.5 {
                for i in 0..m {
                    u[(i, j)] = x[i] / nx;
                }
                filled = true;
                break;
            }
        }
        if !filled {
            return Err(LinAlgError::NoConvergence { op: "svd completion", sweeps: 0 });
        }
    }
    Ok(())
}

/// Minimum-norm least squares by SVD, as `numpy.linalg.lstsq`: singular values at or below
/// `rcond * s_max` are treated as zero. `rcond = None` uses `max(m, n) eps`.
pub fn lstsq_min_norm(a: &Mat, b: &[f64], rcond: Option<f64>) -> Result<Lstsq, LinAlgError> {
    if b.len() != a.rows {
        return Err(LinAlgError::DimensionMismatch {
            op: "lstsq_min_norm",
            what: "right-hand side",
            expected: a.rows,
            found: b.len(),
        });
    }
    require_finite("lstsq_min_norm", b)?;
    let f = svd(a)?;
    let rcond = rcond.unwrap_or_else(|| a.rows.max(a.cols) as f64 * EPS);
    let rank = f.rank(rcond);
    let mut x = vec![0.0; a.cols];
    for j in 0..rank {
        let coef: f64 = (0..a.rows).map(|i| f.u[(i, j)] * b[i]).sum::<f64>() / f.s[j];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += coef * f.v[(i, j)];
        }
    }
    require_finite("lstsq_min_norm", &x)?;
    let fitted = a.matvec(&x)?;
    let rss = b.iter().zip(&fitted).map(|(y, f)| (y - f) * (y - f)).sum();
    Ok(Lstsq { x, rank, rss })
}
