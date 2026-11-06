//! Linear algebra: reconstruction and orthogonality bounds stated in `eps` (f64::EPSILON), the
//! dimensions and, where the bound depends on it, the condition number; matrices with known
//! answers; and the typed refusals.
//!
//! The bounds (Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 8, 10, 19, 20):
//! - Householder QR: `|A P - Q R|_F <= c m n eps |A|_F` and `|Q'Q - I|_F <= c m n eps`, whatever
//!   the condition of `A`;
//! - Cholesky: `|A - L L'|_F <= c n eps |A|_F`;
//! - Jacobi eigen and SVD: `|V'V - I|_F <= c n eps` and the reconstruction `<= c n eps |A|_F`,
//!   with `|U'U - I|_F <= c n eps cond(A)` for the singular vectors of the non-null part;
//! - a solve: `|x_hat - x| / |x| <= c n eps cond(A)`.
//!
//! `c` is 8 for QR and the solves and 32 for the Jacobi methods, which apply O(n) rotations to
//! each column in each of several sweeps.

use raptorbt::math::linalg::*;

const EPS: f64 = f64::EPSILON;

struct Rng(u64);

impl Rng {
    /// SplitMix64, uniform in [-1, 1).
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random(m: usize, n: usize, seed: u64) -> Mat {
    let mut rng = Rng(seed);
    Mat::from_vec(m, n, (0..m * n).map(|_| rng.next()).collect()).unwrap()
}

fn eye(n: usize) -> Mat {
    Mat::identity(n).unwrap()
}

fn sub(a: &Mat, b: &Mat) -> Mat {
    let data = a.as_slice().iter().zip(b.as_slice()).map(|(x, y)| x - y).collect();
    Mat::from_vec(a.rows(), a.cols(), data).unwrap()
}

fn hilbert(n: usize) -> Mat {
    let data = (0..n * n).map(|k| 1.0 / ((k / n + k % n + 1) as f64)).collect();
    Mat::from_vec(n, n, data).unwrap()
}

fn binomial(n: u128, k: u128) -> u128 {
    let k = k.min(n - k);
    (0..k).fold(1u128, |acc, i| acc * (n - i) / (i + 1))
}

/// The exact inverse of the Hilbert matrix, an integer matrix:
/// `(-1)^(i+j) (i+j-1) C(n+i-1, n-j) C(n+j-1, n-i) C(i+j-2, i-1)^2` (1-based). Exact in f64 for
/// n up to 8 (entries stay under 2^53).
fn hilbert_inverse(n: usize) -> Mat {
    let mut data = Vec::with_capacity(n * n);
    for i in 1..=n as u128 {
        for j in 1..=n as u128 {
            let nn = n as u128;
            let mag = (i + j - 1)
                * binomial(nn + i - 1, nn - j)
                * binomial(nn + j - 1, nn - i)
                * binomial(i + j - 2, i - 1).pow(2);
            let v = mag as f64;
            data.push(if (i + j) % 2 == 0 { v } else { -v });
        }
    }
    Mat::from_vec(n, n, data).unwrap()
}

fn vec_diff_norm(a: &[f64], b: &[f64]) -> f64 {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    norm2(&d)
}

// ------------------------------------------------------------------------------------- Mat

#[test]
fn matrices_refuse_degenerate_construction() {
    assert!(matches!(Mat::zeros(0, 3), Err(LinAlgError::ZeroSize { .. })));
    assert!(matches!(Mat::identity(0), Err(LinAlgError::ZeroSize { .. })));
    assert!(matches!(Mat::from_vec(0, 0, vec![]), Err(LinAlgError::ZeroSize { .. })));
    assert!(matches!(Mat::from_rows::<Vec<f64>>(&[]), Err(LinAlgError::ZeroSize { .. })));
    assert!(matches!(
        Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0]),
        Err(LinAlgError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        Mat::from_rows(&[vec![1.0, 2.0], vec![3.0]]),
        Err(LinAlgError::DimensionMismatch { .. })
    ));
    assert!(matches!(Mat::from_vec(1, 2, vec![1.0, f64::NAN]), Err(LinAlgError::NonFinite { .. })));
    assert!(matches!(
        Mat::from_vec(1, 2, vec![f64::INFINITY, 1.0]),
        Err(LinAlgError::NonFinite { .. })
    ));
}

#[test]
fn products_check_shapes_and_overflow() {
    let a = Mat::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    assert!(matches!(a.matmul(&a), Err(LinAlgError::DimensionMismatch { .. })));
    assert!(matches!(a.matvec(&[1.0, 2.0]), Err(LinAlgError::DimensionMismatch { .. })));
    // by hand: [1 2 3; 4 5 6] * [1 0 -1]' = [-2, -2]
    assert_eq!(a.matvec(&[1.0, 0.0, -1.0]).unwrap(), vec![-2.0, -2.0]);
    let at = a.transpose();
    let aat = a.matmul(&at).unwrap();
    assert_eq!(aat.as_slice(), &[14.0, 32.0, 32.0, 77.0]);
    let big = Mat::from_vec(1, 1, vec![1e308]).unwrap();
    assert!(matches!(big.matmul(&big), Err(LinAlgError::NonFinite { .. })));
}

#[test]
fn norm2_does_not_overflow_or_underflow() {
    assert_eq!(norm2(&[3.0, 4.0]), 5.0);
    assert_eq!(norm2(&[]), 0.0);
    assert_eq!(norm2(&[0.0, 0.0]), 0.0);
    let big = norm2(&[1e200, 1e200]);
    assert!((big / 1e200 - std::f64::consts::SQRT_2).abs() < 4.0 * EPS);
    let small = norm2(&[1e-200, 1e-200]);
    assert!((small / 1e-200 - std::f64::consts::SQRT_2).abs() < 4.0 * EPS);
}

// -------------------------------------------------------------------------------------- QR

fn qr_bound(m: usize, n: usize) -> f64 {
    8.0 * (m * n) as f64 * EPS
}

fn check_qr(a: &Mat) -> PivotedQr {
    let (m, n) = (a.rows(), a.cols());
    let k = m.min(n);
    let f = qr_pivoted(a).unwrap();
    let q = f.q();
    let r = f.r();
    assert_eq!((q.rows(), q.cols()), (m, k));
    assert_eq!((r.rows(), r.cols()), (k, n));
    let mut ap = Mat::zeros(m, n).unwrap();
    for j in 0..n {
        for i in 0..m {
            ap[(i, j)] = a[(i, f.perm()[j])];
        }
    }
    let err = sub(&q.matmul(&r).unwrap(), &ap).frobenius();
    assert!(
        err <= qr_bound(m, n) * a.frobenius(),
        "{m}x{n}: |AP - QR| = {err:e}, bound {:e}",
        qr_bound(m, n) * a.frobenius()
    );
    let orth = sub(&q.transpose().matmul(&q).unwrap(), &eye(k)).frobenius();
    assert!(orth <= qr_bound(m, n), "{m}x{n}: |Q'Q - I| = {orth:e}, bound {:e}", qr_bound(m, n));
    let d = f.r_diag();
    for w in d.windows(2) {
        assert!(w[1].abs() <= w[0].abs() * (1.0 + 1e-12), "R diagonal not nonincreasing: {d:?}");
    }
    f
}

#[test]
fn qr_is_orthogonal_and_reconstructs() {
    for (m, n, seed) in
        [(6, 4, 1), (4, 6, 2), (8, 8, 3), (50, 5, 4), (1, 1, 5), (3, 1, 6), (1, 3, 7)]
    {
        check_qr(&random(m, n, seed));
    }
}

#[test]
fn qr_stays_orthogonal_on_hilbert_matrices() {
    // Householder orthogonality does not depend on the condition number; the Hilbert matrices
    // reach cond ~ 1e16 at n = 12
    for n in [4, 8, 10, 12] {
        check_qr(&hilbert(n));
    }
}

#[test]
fn qr_reports_the_rank() {
    let outer = |u: &[f64], v: &[f64]| {
        let data = u.iter().flat_map(|a| v.iter().map(move |b| a * b)).collect();
        Mat::from_vec(u.len(), v.len(), data).unwrap()
    };
    assert_eq!(check_qr(&outer(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, -1.0, 2.0])).rank(), 1);
    assert_eq!(check_qr(&Mat::from_rows(&[[1.0, 2.0], [2.0, 4.0]]).unwrap()).rank(), 1);
    // exactly rank 2: row 3 = 2 row 2 - row 1
    let rank2 = Mat::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    assert_eq!(check_qr(&rank2).rank(), 2);
    assert_eq!(check_qr(&eye(4)).rank(), 4);
    assert_eq!(check_qr(&random(7, 3, 11)).rank(), 3);
    assert_eq!(check_qr(&random(3, 7, 12)).rank(), 3);
    // the zero matrix has rank 0, a tiny column against a unit one is below max(m, n) eps
    assert_eq!(qr_pivoted(&Mat::zeros(3, 3).unwrap()).unwrap().rank(), 0);
    assert_eq!(check_qr(&Mat::from_rows(&[[1.0, 0.0], [0.0, 1e-20]]).unwrap()).rank(), 1);
    assert_eq!(check_qr(&Mat::from_rows(&[[1.0, 0.0], [0.0, 1e-10]]).unwrap()).rank(), 2);
    // Hilbert(12) has cond ~ 1.7e16 > 1 / eps: at least one direction is lost
    let h12 = check_qr(&hilbert(12));
    assert!((10..=11).contains(&h12.rank()), "rank {}", h12.rank());
}

#[test]
fn qr_pivoting_puts_the_largest_column_first() {
    let a = Mat::from_rows(&[[1.0, 10.0, 3.0], [1.0, 10.0, -3.0]]).unwrap();
    let f = qr_pivoted(&a).unwrap();
    assert_eq!(f.perm()[0], 1);
}

#[test]
fn solve_matches_the_exact_inverse_of_hilbert_matrices() {
    for n in 3..=8usize {
        let h = hilbert(n);
        let hinv = hilbert_inverse(n);
        let kappa = h.frobenius() * hinv.frobenius();
        let bound = 8.0 * n as f64 * kappa * EPS;
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let x = solve(&h, &e).unwrap();
            let want = hinv.col(j);
            let err = vec_diff_norm(&x, &want) / norm2(&want);
            assert!(
                err <= bound,
                "H{n} column {j}: error {err:e}, bound {bound:e} (cond {kappa:e})"
            );
        }
        let inv = inverse(&h).unwrap();
        let err = sub(&inv, &hinv).frobenius() / hinv.frobenius();
        assert!(err <= bound, "inverse(H{n}): error {err:e}, bound {bound:e}");
    }
}

#[test]
fn solve_near_singular_system_within_the_condition_bound() {
    // [[1, 1], [1, 1 + d]] x = [2, 2 + d] has the solution [1, 1]; cond is about 4 / d
    let d = 1e-9;
    let a = Mat::from_rows(&[[1.0, 1.0], [1.0, 1.0 + d]]).unwrap();
    let kappa = svd(&a).unwrap().condition();
    assert!(kappa > 1e9 && kappa < 1e10, "cond {kappa:e}");
    let x = solve(&a, &[2.0, 2.0 + d]).unwrap();
    let err = vec_diff_norm(&x, &[1.0, 1.0]) / 2f64.sqrt();
    assert!(err <= 8.0 * 2.0 * kappa * EPS, "error {err:e}");
}

#[test]
fn solve_refuses_singular_and_malformed_systems() {
    let singular = Mat::from_rows(&[[1.0, 2.0], [2.0, 4.0]]).unwrap();
    assert!(matches!(
        solve(&singular, &[1.0, 2.0]),
        Err(LinAlgError::Singular { rank: 1, size: 2, .. })
    ));
    assert!(matches!(inverse(&singular), Err(LinAlgError::Singular { .. })));
    let rect = Mat::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    assert!(matches!(solve(&rect, &[1.0, 2.0]), Err(LinAlgError::NotSquare { .. })));
    assert!(matches!(inverse(&rect), Err(LinAlgError::NotSquare { .. })));
    assert!(matches!(solve(&eye(2), &[1.0, 2.0, 3.0]), Err(LinAlgError::DimensionMismatch { .. })));
    assert!(matches!(solve(&eye(2), &[1.0, f64::NAN]), Err(LinAlgError::NonFinite { .. })));
    assert!(matches!(lstsq(&rect, &[1.0, 2.0, 3.0]), Err(LinAlgError::DimensionMismatch { .. })));
}

#[test]
fn lstsq_known_line_fit() {
    // y on t = 0..3 with y = [1, 3, 2, 5]. By hand: sum t = 6, sum t^2 = 14, sum y = 11,
    // sum t y = 0 + 3 + 4 + 15 = 22, so slope = (4*22 - 6*11) / (4*14 - 36) = 22/20 = 1.1 and
    // intercept = (11 - 1.1*6) / 4 = 1.1. Fitted 1.1, 2.2, 3.3, 4.4; residuals -0.1, 0.8, -1.3,
    // 0.6; sum of squares 0.01 + 0.64 + 1.69 + 0.36 = 2.7.
    let a = Mat::from_rows(&[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]).unwrap();
    let r = lstsq(&a, &[1.0, 3.0, 2.0, 5.0]).unwrap();
    assert_eq!(r.rank, 2);
    assert!((r.x[0] - 1.1).abs() < 1e-13 && (r.x[1] - 1.1).abs() < 1e-13, "x = {:?}", r.x);
    assert!((r.rss - 2.7).abs() < 1e-13, "rss {}", r.rss);
    // the exact line y = 2 + 3 t is recovered with a residual of rounding size
    let exact = lstsq(&a, &[2.0, 5.0, 8.0, 11.0]).unwrap();
    assert!((exact.x[0] - 2.0).abs() < 1e-13 && (exact.x[1] - 3.0).abs() < 1e-13);
    assert!(exact.rss < 1e-24);
}

#[test]
fn lstsq_on_rank_deficient_input_reports_the_rank() {
    // two proportional columns: the QR solution is basic (the second column gets 0)
    let a = Mat::from_rows(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]).unwrap();
    let basic = lstsq(&a, &[3.0, 3.0, 3.0]).unwrap();
    assert_eq!(basic.rank, 1);
    assert!(basic.rss < 1e-24);
    let fit = a.matvec(&basic.x).unwrap();
    assert!(vec_diff_norm(&fit, &[3.0, 3.0, 3.0]) < 1e-13);
    // the SVD solution is the minimum-norm one: x = a (1, 2) with 5a = 3, so (0.6, 1.2)
    let min_norm = lstsq_min_norm(&a, &[3.0, 3.0, 3.0], None).unwrap();
    assert_eq!(min_norm.rank, 1);
    assert!((min_norm.x[0] - 0.6).abs() < 1e-13 && (min_norm.x[1] - 1.2).abs() < 1e-13);
}

#[test]
fn lstsq_min_norm_agrees_with_qr_on_full_rank_input() {
    let a = random(30, 4, 21);
    let b: Vec<f64> = (0..30).map(|i| (i as f64 * 0.37).sin()).collect();
    let q = lstsq(&a, &b).unwrap();
    let s = lstsq_min_norm(&a, &b, None).unwrap();
    let kappa = svd(&a).unwrap().condition();
    assert_eq!((q.rank, s.rank), (4, 4));
    let err = vec_diff_norm(&q.x, &s.x) / norm2(&s.x);
    assert!(err <= 8.0 * 30.0 * kappa * EPS, "{err:e}");
    assert!((q.rss - s.rss).abs() <= 1e-12 * q.rss.max(1.0));
}

// ----------------------------------------------------------------------------- Cholesky

#[test]
fn cholesky_reconstructs_and_solves() {
    let b = random(6, 4, 31);
    let btb = b.transpose().matmul(&b).unwrap();
    let a = Mat::from_vec(
        4,
        4,
        btb.as_slice()
            .iter()
            .enumerate()
            .map(|(k, v)| v + if k % 5 == 0 { 1.0 } else { 0.0 })
            .collect(),
    )
    .unwrap();
    let c = cholesky(&a).unwrap();
    let l = c.l();
    for i in 0..4 {
        for j in i + 1..4 {
            assert_eq!(l[(i, j)], 0.0, "upper triangle");
        }
        assert!(l[(i, i)] > 0.0);
    }
    let err = sub(&l.matmul(&l.transpose()).unwrap(), &a).frobenius();
    assert!(err <= 8.0 * 4.0 * EPS * a.frobenius(), "|A - LL'| = {err:e}");
    // A * ones = b, so the solve returns ones within 8 n eps cond
    let ones = vec![1.0; 4];
    let rhs = a.matvec(&ones).unwrap();
    let x = c.solve(&rhs).unwrap();
    let kappa = svd(&a).unwrap().condition();
    assert!(vec_diff_norm(&x, &ones) / 2.0 <= 8.0 * 4.0 * kappa * EPS);
    // ln det A = sum of ln of the eigenvalues
    let eig = sym_eigen(&a).unwrap();
    let want: f64 = eig.values.iter().map(|v| v.ln()).sum();
    assert!((c.log_det() - want).abs() <= 1e-11 * (1.0 + want.abs()));
    assert!(matches!(c.solve(&[1.0]), Err(LinAlgError::DimensionMismatch { .. })));
}

#[test]
fn cholesky_of_known_matrix() {
    // [[4, 2], [2, 5]] = L L' with L = [[2, 0], [1, 2]]
    let c = cholesky(&Mat::from_rows(&[[4.0, 2.0], [2.0, 5.0]]).unwrap()).unwrap();
    assert_eq!(c.l().as_slice(), &[2.0, 0.0, 1.0, 2.0]);
    assert!((c.log_det() - 16f64.ln()).abs() < 1e-15);
}

#[test]
fn cholesky_refuses_what_is_not_positive_definite() {
    let m = |rows: &[[f64; 2]]| Mat::from_rows(rows).unwrap();
    // eigenvalues 3 and -1
    assert!(matches!(
        cholesky(&m(&[[1.0, 2.0], [2.0, 1.0]])),
        Err(LinAlgError::NotPositiveDefinite { pivot: 1, .. })
    ));
    // singular
    assert!(matches!(
        cholesky(&m(&[[1.0, 1.0], [1.0, 1.0]])),
        Err(LinAlgError::NotPositiveDefinite { pivot: 1, .. })
    ));
    assert!(matches!(
        cholesky(&m(&[[1.0, 0.0], [0.0, -1.0]])),
        Err(LinAlgError::NotPositiveDefinite { pivot: 1, .. })
    ));
    assert!(matches!(
        cholesky(&m(&[[0.0, 0.0], [0.0, 0.0]])),
        Err(LinAlgError::NotPositiveDefinite { pivot: 0, .. })
    ));
    // positive definite in exact arithmetic (determinant eps) but below the n eps floor
    assert!(matches!(
        cholesky(&m(&[[1.0, 1.0], [1.0, 1.0 + EPS]])),
        Err(LinAlgError::NotPositiveDefinite { pivot: 1, .. })
    ));
    // and the same shape one digit of headroom above the floor factors
    assert!(cholesky(&m(&[[1.0, 1.0], [1.0, 1.0 + 1e-10]])).is_ok());
}

#[test]
fn cholesky_refuses_malformed_input() {
    assert!(matches!(
        cholesky(&Mat::from_rows(&[[1.0, 2.0], [3.0, 1.0]]).unwrap()),
        Err(LinAlgError::NotSymmetric { .. })
    ));
    assert!(matches!(
        cholesky(&Mat::from_rows(&[[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]).unwrap()),
        Err(LinAlgError::NotSquare { .. })
    ));
    // asymmetry below 1e-10 of the scale is rounding, not a different matrix
    assert!(cholesky(&Mat::from_rows(&[[2.0, 1.0 + 1e-13], [1.0, 2.0]]).unwrap()).is_ok());
}

// --------------------------------------------------------------------------- eigenvalues

fn check_eigen(a: &Mat) -> SymEigen {
    let n = a.rows();
    let e = sym_eigen(a).unwrap();
    let bound = 64.0 * n as f64 * EPS;
    let orth = sub(&e.vectors.transpose().matmul(&e.vectors).unwrap(), &eye(n)).frobenius();
    assert!(orth <= bound, "|V'V - I| = {orth:e}, bound {bound:e}");
    let mut lam = Mat::zeros(n, n).unwrap();
    for i in 0..n {
        lam[(i, i)] = e.values[i];
    }
    let recon = e.vectors.matmul(&lam).unwrap().matmul(&e.vectors.transpose()).unwrap();
    let err = sub(&recon, a).frobenius();
    assert!(err <= bound * a.frobenius().max(1.0), "|V L V' - A| = {err:e}");
    for w in e.values.windows(2) {
        assert!(w[0] >= w[1], "eigenvalues not descending: {:?}", e.values);
    }
    e
}

#[test]
fn eigenvalues_of_known_matrices() {
    // [[2,1],[1,2]]: 3 and 1
    let e = check_eigen(&Mat::from_rows(&[[2.0, 1.0], [1.0, 2.0]]).unwrap());
    assert!((e.values[0] - 3.0).abs() <= 8.0 * 2.0 * EPS * 3.0);
    assert!((e.values[1] - 1.0).abs() <= 8.0 * 2.0 * EPS * 3.0);
    // [[0,1],[1,0]]: 1 and -1
    let e = check_eigen(&Mat::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap());
    assert!((e.values[0] - 1.0).abs() < 1e-15 && (e.values[1] + 1.0).abs() < 1e-15);
    // diag(2) with the block [[3,4],[4,9]]: trace 12 and determinant 11 give 11 and 1
    let a = Mat::from_rows(&[[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 4.0, 9.0]]).unwrap();
    let e = check_eigen(&a);
    for (got, want) in e.values.iter().zip([11.0, 2.0, 1.0]) {
        assert!((got - want).abs() <= 8.0 * 3.0 * EPS * 11.0, "{:?}", e.values);
    }
}

#[test]
fn eigenvalues_of_the_second_difference_matrix() {
    // tridiagonal(-1, 2, -1) of order n has eigenvalues 2 - 2 cos(k pi / (n + 1)), k = 1..n
    let n = 10;
    let mut a = Mat::zeros(n, n).unwrap();
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i + 1 < n {
            a[(i, i + 1)] = -1.0;
            a[(i + 1, i)] = -1.0;
        }
    }
    let e = check_eigen(&a);
    for (i, got) in e.values.iter().enumerate() {
        let k = (n - i) as f64;
        let want = 2.0 - 2.0 * (k * std::f64::consts::PI / (n as f64 + 1.0)).cos();
        // |A|_2 < 4
        assert!((got - want).abs() <= 8.0 * n as f64 * EPS * 4.0, "k = {k}: {got} vs {want}");
    }
}

#[test]
fn eigen_handles_repeated_zero_and_diagonal_input() {
    let e = check_eigen(&eye(3));
    assert_eq!(e.values, vec![1.0, 1.0, 1.0]);
    let e = check_eigen(&Mat::from_rows(&[[0.0, 0.0], [0.0, 0.0]]).unwrap());
    assert_eq!(e.values, vec![0.0, 0.0]);
    let d = Mat::from_rows(&[[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0]]).unwrap();
    let e = check_eigen(&d);
    assert_eq!(e.values, vec![5.0, 3.0, 1.0]);
    check_eigen(&Mat::from_rows(&[[7.0]]).unwrap());
}

#[test]
fn eigen_of_random_symmetric_and_gram_matrices() {
    for (n, seed) in [(5, 41), (12, 42), (30, 43)] {
        let b = random(n, n, seed);
        let sym = Mat::from_vec(
            n,
            n,
            (0..n * n).map(|k| 0.5 * (b[(k / n, k % n)] + b[(k % n, k / n)])).collect(),
        )
        .unwrap();
        check_eigen(&sym);
        check_eigen(&b.transpose().matmul(&b).unwrap());
    }
}

#[test]
fn eigenvalues_of_hilbert_have_the_known_determinant() {
    // det H4 = 1 / 6048000 exactly; the product of the eigenvalues carries a relative error
    // of at most 8 n eps cond
    let h = hilbert(4);
    let e = check_eigen(&h);
    let det: f64 = e.values.iter().product();
    let kappa = h.frobenius() * hilbert_inverse(4).frobenius();
    let bound = 8.0 * 4.0 * kappa * EPS;
    assert!(((det * 6_048_000.0) - 1.0).abs() <= bound, "det {det:e}, bound {bound:e}");
    assert!(e.values.iter().all(|&v| v > 0.0));
}

#[test]
fn eigen_refuses_malformed_input() {
    assert!(matches!(
        sym_eigen(&Mat::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap()),
        Err(LinAlgError::NotSymmetric { .. })
    ));
    assert!(matches!(
        sym_eigen(&Mat::from_rows(&[[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]).unwrap()),
        Err(LinAlgError::NotSquare { .. })
    ));
}

// ------------------------------------------------------------------------------------ SVD

fn check_svd(a: &Mat) -> Svd {
    let (m, n) = (a.rows(), a.cols());
    let k = m.min(n);
    let f = svd(a).unwrap();
    assert_eq!((f.u.rows(), f.u.cols()), (m, k));
    assert_eq!((f.v.rows(), f.v.cols()), (n, k));
    assert_eq!(f.s.len(), k);
    for w in f.s.windows(2) {
        assert!(w[0] >= w[1] && w[1] >= 0.0, "singular values {:?}", f.s);
    }
    let mut sig = Mat::zeros(k, k).unwrap();
    for i in 0..k {
        sig[(i, i)] = f.s[i];
    }
    let recon = f.u.matmul(&sig).unwrap().matmul(&f.v.transpose()).unwrap();
    let c = 32.0 * m.max(n) as f64 * EPS;
    let err = sub(&recon, a).frobenius();
    assert!(
        err <= c * a.frobenius(),
        "{m}x{n}: |A - U S V'| = {err:e}, bound {:e}",
        c * a.frobenius()
    );
    let v_orth = sub(&f.v.transpose().matmul(&f.v).unwrap(), &eye(k)).frobenius();
    assert!(v_orth <= c, "{m}x{n}: |V'V - I| = {v_orth:e}, bound {c:e}");
    // U's columns for singular values above max(m, n) eps s_max carry cond-sized error; the
    // others are an orthonormal completion
    let null_tol = m.max(n) as f64 * EPS * f.s[0];
    let kept = f.s.iter().filter(|&&s| s > null_tol).count();
    let cond_kept = if kept == 0 { 1.0 } else { f.s[0] / f.s[kept - 1] };
    let u_orth = sub(&f.u.transpose().matmul(&f.u).unwrap(), &eye(k)).frobenius();
    assert!(
        u_orth <= c * cond_kept,
        "{m}x{n}: |U'U - I| = {u_orth:e}, bound {:e} (cond of the non-null part {cond_kept:e})",
        c * cond_kept
    );
    f
}

#[test]
fn svd_reconstructs_random_matrices_of_every_shape() {
    for (m, n, seed) in [(7, 4, 51), (4, 7, 52), (6, 6, 53), (40, 3, 54), (1, 5, 55), (5, 1, 56)] {
        check_svd(&random(m, n, seed));
    }
}

#[test]
fn svd_of_known_matrices() {
    let f = check_svd(&Mat::from_rows(&[[3.0, 0.0], [0.0, 4.0]]).unwrap());
    assert!((f.s[0] - 4.0).abs() < 1e-15 && (f.s[1] - 3.0).abs() < 1e-15);
    // [[3, 4]]: one singular value, 5, with right vector (0.6, 0.8)
    let f = check_svd(&Mat::from_rows(&[[3.0, 4.0]]).unwrap());
    assert!((f.s[0] - 5.0).abs() < 1e-14);
    let sign = f.u[(0, 0)].signum();
    assert!((sign * f.v[(0, 0)] - 0.6).abs() < 1e-14 && (sign * f.v[(1, 0)] - 0.8).abs() < 1e-14);
    // [[1,1],[1,1]] has singular values 2 and 0
    let f = check_svd(&Mat::from_rows(&[[1.0, 1.0], [1.0, 1.0]]).unwrap());
    assert!((f.s[0] - 2.0).abs() < 1e-15 && f.s[1] < 1e-15);
    assert_eq!(f.rank(8.0 * 4.0 * EPS), 1);
    // the singular values of a symmetric matrix are the absolute eigenvalues: [[0,1],[1,0]]
    let f = check_svd(&Mat::from_rows(&[[0.0, 1.0], [1.0, 0.0]]).unwrap());
    assert!((f.s[0] - 1.0).abs() < 1e-15 && (f.s[1] - 1.0).abs() < 1e-15);
}

#[test]
fn svd_on_rank_deficient_input() {
    let outer = |u: &[f64], v: &[f64]| {
        let data = u.iter().flat_map(|a| v.iter().map(move |b| a * b)).collect();
        Mat::from_vec(u.len(), v.len(), data).unwrap()
    };
    // exactly rank 1: the rest of the singular values is rounding noise of size m n eps |A|
    let a = outer(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, -1.0, 2.0]);
    let f = check_svd(&a);
    let noise = 8.0 * 15.0 * EPS;
    assert!(f.s[1] <= noise * a.frobenius() && f.s[2] <= noise * a.frobenius(), "{:?}", f.s);
    assert_eq!(f.rank(noise), 1);
    // |a| = sqrt(55) * sqrt(6): the one real singular value
    assert!((f.s[0] - (55.0f64 * 6.0).sqrt()).abs() <= 1e-14 * f.s[0]);
    // exactly rank 2 (row 3 = 2 row 2 - row 1)
    let b = Mat::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).unwrap();
    let f = check_svd(&b);
    assert_eq!(f.rank(8.0 * 9.0 * EPS), 2);
    assert!(f.s[2] <= 8.0 * 9.0 * EPS * b.frobenius());
    assert!(f.condition() > 1e15);
    // the zero matrix: no singular values, an orthonormal U all the same
    let z = check_svd(&Mat::zeros(3, 2).unwrap());
    assert_eq!(z.s, vec![0.0, 0.0]);
    assert_eq!(z.condition(), f64::INFINITY);
    assert_eq!(z.rank(1e-12), 0);
}

/// Near-duplicate instruments give panels of wide rank deficiency: columns that are repeated,
/// shifted copies of one pattern, or zero. The Jacobi sweeps must settle (a column at rounding
/// level against the largest is a null direction), and `check_svd` states the bounds: the
/// reconstruction within 32 max(m, n) eps |A|, V orthonormal, U orthonormal on the non-null part.
#[test]
fn svd_settles_on_wide_rank_deficiency() {
    // a period-29 pattern shifted 40 ways, standardised: rank 28 of 40 (the period sums to zero)
    let cols: Vec<Vec<f64>> = (0..40)
        .map(|j| (0..80).map(|t| (((t * 7 + j * 13) % 29) as f64 - 14.0) / 100.0).collect())
        .collect();
    let a = Mat::from_vec(80, 40, (0..80).flat_map(|t| cols.iter().map(move |c| c[t])).collect())
        .unwrap();
    let f = check_svd(&a);
    assert_eq!(f.rank(a.rows().max(a.cols()) as f64 * EPS), 28);

    // 40 columns, 5 distinct: rank 5
    let base = random(60, 5, 71);
    let dup = Mat::from_vec(
        60,
        40,
        (0..60).flat_map(|i| (0..40).map(move |j| (i, j % 5))).map(|(i, j)| base[(i, j)]).collect(),
    )
    .unwrap();
    let f = check_svd(&dup);
    assert_eq!(f.rank(60.0 * EPS), 5);
    // each distinct column repeated 8 times: the singular values are sqrt(8) times those of base
    let g = svd(&base).unwrap();
    for i in 0..5 {
        assert!((f.s[i] - 8f64.sqrt() * g.s[i]).abs() <= 32.0 * 60.0 * EPS * f.s[0], "{i}");
    }

    // zero columns among live ones
    let live = random(60, 40, 72);
    let mut data = live.as_slice().to_vec();
    for i in 0..60 {
        for j in (0..40).filter(|j| j % 3 == 0) {
            data[i * 40 + j] = 0.0;
        }
    }
    let z = Mat::from_vec(60, 40, data).unwrap();
    let f = check_svd(&z);
    assert_eq!(f.rank(60.0 * EPS), 26, "40 columns, 14 of them zero");

    // all columns zero but one
    let mut one = vec![0.0; 60 * 40];
    for i in 0..60 {
        one[i * 40 + 7] = (i as f64 + 1.0) / 60.0;
    }
    let f = check_svd(&Mat::from_vec(60, 40, one).unwrap());
    assert_eq!(f.rank(60.0 * EPS), 1);
}

#[test]
fn svd_of_hilbert_matrices_within_the_condition_bound() {
    for n in [4, 8, 12] {
        check_svd(&hilbert(n));
    }
    // the product of the singular values is |det| = 1 / 6048000 for n = 4
    let h = hilbert(4);
    let f = svd(&h).unwrap();
    let det: f64 = f.s.iter().product();
    let kappa = h.frobenius() * hilbert_inverse(4).frobenius();
    assert!((det * 6_048_000.0 - 1.0).abs() <= 8.0 * 4.0 * kappa * EPS);
}

#[test]
fn svd_scales_extreme_entries() {
    let big = Mat::from_rows(&[[3e200, 0.0], [0.0, 4e200]]).unwrap();
    let f = svd(&big).unwrap();
    assert!((f.s[0] / 4e200 - 1.0).abs() < 4.0 * EPS && (f.s[1] / 3e200 - 1.0).abs() < 4.0 * EPS);
    let small = Mat::from_rows(&[[3e-200, 0.0], [0.0, 4e-200]]).unwrap();
    let f = svd(&small).unwrap();
    assert!((f.s[0] / 4e-200 - 1.0).abs() < 4.0 * EPS);
}

// ------------------------------------------------------------------------------ determinism

#[test]
fn factorisations_are_bit_identical_across_runs() {
    let a = random(9, 5, 61);
    let s1 = svd(&a).unwrap();
    let s2 = svd(&a).unwrap();
    assert_eq!(bits(s1.u.as_slice()), bits(s2.u.as_slice()));
    assert_eq!(bits(&s1.s), bits(&s2.s));
    assert_eq!(bits(s1.v.as_slice()), bits(s2.v.as_slice()));
    let g = a.transpose().matmul(&a).unwrap();
    let e1 = sym_eigen(&g).unwrap();
    let e2 = sym_eigen(&g).unwrap();
    assert_eq!(bits(&e1.values), bits(&e2.values));
    assert_eq!(bits(e1.vectors.as_slice()), bits(e2.vectors.as_slice()));
    let q1 = qr_pivoted(&a).unwrap();
    let q2 = qr_pivoted(&a).unwrap();
    assert_eq!(bits(q1.q().as_slice()), bits(q2.q().as_slice()));
    assert_eq!(q1.perm(), q2.perm());
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
