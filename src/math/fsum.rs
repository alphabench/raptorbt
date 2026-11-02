//! Correctly rounded summation: Shewchuk's partials with the half-even fix-up,
//! exactly the algorithm behind CPython's `math.fsum` (Modules/mathmodule.c).
//!
//! A Kahan or Neumaier sum agrees with `math.fsum` to an ulp or two, this one
//! agrees bit for bit, which is what lets the report and the research numbers
//! match a `math.fsum` reference to 1e-12 and usually exactly.

/// Accumulates a correctly rounded sum.
#[derive(Debug, Default, Clone)]
pub(crate) struct FSum {
    partials: Vec<f64>,
    /// Plain sum of non-finite inputs; once set it decides the result.
    special: Option<f64>,
}

impl FSum {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn add(&mut self, value: f64) {
        if !value.is_finite() {
            self.special = Some(self.special.unwrap_or(0.0) + value);
            return;
        }
        let mut x = value;
        let mut i = 0;
        for j in 0..self.partials.len() {
            let mut y = self.partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                self.partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        self.partials.truncate(i);
        self.partials.push(x);
    }

    pub(crate) fn value(&self) -> f64 {
        if let Some(s) = self.special {
            return s;
        }
        let p = &self.partials;
        let mut n = p.len();
        if n == 0 {
            return 0.0;
        }
        n -= 1;
        let mut hi = p[n];
        let mut lo = 0.0;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = p[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        // Round half to even across the remaining partials, as CPython does.
        if n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
        hi
    }
}

/// `math.fsum` over an iterator.
pub(crate) fn fsum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut acc = FSum::new();
    for v in values {
        acc.add(v);
    }
    acc.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_where_a_naive_sum_is_not() {
        // Values pinned from CPython 3.12 `math.fsum`.
        assert_eq!(fsum([0.1; 10]), 1.0);
        assert_ne!([0.1f64; 10].iter().sum::<f64>(), 1.0);
        assert_eq!(fsum([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]), 1e-100);
        assert_eq!(fsum([1.0, 1e-16, 1e-16]), 1.0000000000000002);
        assert_eq!(fsum(std::iter::empty()), 0.0);
    }

    #[test]
    fn non_finite_inputs_propagate() {
        assert!(fsum([1.0, f64::NAN]).is_nan());
        assert_eq!(fsum([1.0, f64::INFINITY]), f64::INFINITY);
        assert!(fsum([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    }
}
