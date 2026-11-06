//! Causal rolling primitives shared by kernels. Each is fed one value per bar together with the
//! history `Hist` (the bars before it), so batch and streaming drive it identically. A window of
//! `p` holds the current value and the `p - 1` before it; the value leaving is `ago(p)`, so a
//! kernel using these reports `lookback() >= p`.
//!
//! Non-finite values count as NaN: a window holding one outputs NaN until it leaves, then
//! recovers.

use super::kernel::Hist;

/// Neumaier compensated accumulator.
#[derive(Debug, Clone, Copy, Default)]
struct Neumaier {
    s: f64,
    c: f64,
}

impl Neumaier {
    fn add(&mut self, v: f64) {
        let t = self.s + v;
        if self.s.abs() >= v.abs() {
            self.c += (self.s - t) + v;
        } else {
            self.c += (v - t) + self.s;
        }
        self.s = t;
    }

    fn value(&self) -> f64 {
        self.s + self.c
    }
}

/// Steps between exact re-sums. A step counter, so batch and streaming re-sum on the same bar.
fn resum_interval(p: usize) -> u64 {
    p.max(1024) as u64
}

/// Rolling sum, Neumaier add and remove, with an exact re-sum from history every
/// `max(p, 1024)` steps.
#[derive(Debug, Clone)]
pub struct RollSum {
    p: usize,
    acc: Neumaier,
    bad: usize,
    n: u64,
}

impl RollSum {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "window of at least 1");
        Self { p, acc: Neumaier::default(), bad: 0, n: 0 }
    }

    pub fn push<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) -> f64 {
        self.n += 1;
        if x.is_finite() {
            self.acc.add(x);
        } else {
            self.bad += 1;
        }
        if self.n > self.p as u64 {
            let old = h.ago(input, self.p);
            if old.is_finite() {
                self.acc.add(-old);
            } else {
                self.bad -= 1;
            }
        }
        if self.n >= self.p as u64 && self.n.is_multiple_of(resum_interval(self.p)) {
            self.resum(x, h, input);
        }
        self.value()
    }

    fn resum<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) {
        self.acc = Neumaier::default();
        self.bad = 0;
        for k in (1..self.p).rev() {
            let v = h.ago(input, k);
            if v.is_finite() {
                self.acc.add(v);
            } else {
                self.bad += 1;
            }
        }
        if x.is_finite() {
            self.acc.add(x);
        } else {
            self.bad += 1;
        }
    }

    /// The window sum, NaN before the window is full or while it holds a non-finite value.
    pub fn value(&self) -> f64 {
        if self.n < self.p as u64 || self.bad > 0 {
            f64::NAN
        } else {
            self.acc.value()
        }
    }
}

/// Rolling mean over a [`RollSum`].
#[derive(Debug, Clone)]
pub struct RollMean {
    sum: RollSum,
}

impl RollMean {
    pub fn new(p: usize) -> Self {
        Self { sum: RollSum::new(p) }
    }

    pub fn push<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) -> f64 {
        self.sum.push(x, h, input) / self.sum.p as f64
    }
}

/// Rolling variance by the sliding Welford update. The state is rebuilt exactly from history
/// on the periodic step counter, when the window recovers from a non-finite value, and when the
/// variance falls by three orders of magnitude from its peak since the last rebuild (a large
/// value just left, so the update's cancellation error would dominate). A window of one repeated
/// value has variance exactly 0.
#[derive(Debug, Clone)]
pub struct RollVar {
    p: usize,
    ddof: usize,
    n: u64,
    mean: f64,
    m2: f64,
    peak: f64,
    bad: usize,
    dirty: bool,
    run: usize,
}

impl RollVar {
    pub fn new(p: usize, ddof: usize) -> Self {
        assert!(p >= 1, "window of at least 1");
        Self { p, ddof, n: 0, mean: 0.0, m2: 0.0, peak: 0.0, bad: 0, dirty: false, run: 0 }
    }

    pub fn push<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) -> f64 {
        self.n += 1;
        let prev = h.ago(input, 1);
        self.run = if x.is_finite() {
            if x == prev {
                self.run + 1
            } else {
                1
            }
        } else {
            0
        };
        if !x.is_finite() {
            self.bad += 1;
        }
        if self.n > self.p as u64 && !h.ago(input, self.p).is_finite() {
            self.bad -= 1;
        }
        if self.n < self.p as u64 {
            return f64::NAN;
        }
        if self.bad > 0 {
            self.dirty = true;
            return f64::NAN;
        }
        if self.run >= self.p {
            // Flat window: state is exact and the answer is exactly 0.
            self.mean = x;
            self.m2 = 0.0;
            self.peak = 0.0;
            self.dirty = false;
        } else if self.dirty
            || self.n == self.p as u64
            || self.n.is_multiple_of(resum_interval(self.p))
        {
            self.resum(x, h, input);
        } else {
            let old = h.ago(input, self.p);
            let p = self.p as f64;
            let new_mean = self.mean + (x - old) / p;
            self.m2 += (x - old) * ((x - new_mean) + (old - self.mean));
            self.mean = new_mean;
            self.peak = self.peak.max(self.m2);
            if self.m2 < 0.0 || self.m2 < self.peak * 1e-3 {
                self.resum(x, h, input);
            }
        }
        self.value()
    }

    fn resum<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) {
        let mut s = Neumaier::default();
        for k in (1..self.p).rev() {
            s.add(h.ago(input, k));
        }
        s.add(x);
        let mean = s.value() / self.p as f64;
        let mut m = Neumaier::default();
        for k in (1..self.p).rev() {
            let d = h.ago(input, k) - mean;
            m.add(d * d);
        }
        let d = x - mean;
        m.add(d * d);
        self.mean = mean;
        self.m2 = m.value();
        self.peak = self.m2;
        self.dirty = false;
    }

    fn value(&self) -> f64 {
        if self.p <= self.ddof {
            return f64::NAN;
        }
        if self.run >= self.p {
            return 0.0;
        }
        self.m2.max(0.0) / (self.p - self.ddof) as f64
    }
}

/// Rolling min or max by a monotonic deque of (index, value) in a ring of `p` slots, amortised
/// O(1). Allocated once.
#[derive(Debug, Clone)]
pub struct RollExt {
    p: usize,
    is_max: bool,
    idx: Box<[u64]>,
    val: Box<[f64]>,
    head: usize,
    len: usize,
    n: u64,
    last_bad: Option<u64>,
}

impl RollExt {
    pub fn new(p: usize, is_max: bool) -> Self {
        assert!(p >= 1, "window of at least 1");
        Self {
            p,
            is_max,
            idx: vec![0; p].into_boxed_slice(),
            val: vec![f64::NAN; p].into_boxed_slice(),
            head: 0,
            len: 0,
            n: 0,
            last_bad: None,
        }
    }

    fn slot(&self, j: usize) -> usize {
        (self.head + j) % self.p
    }

    /// Heap owned: the two rings.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.idx) + std::mem::size_of_val(&*self.val)
    }

    pub fn push(&mut self, x: f64) -> f64 {
        let cur = self.n;
        self.n += 1;
        while self.len > 0 && cur - self.idx[self.head] >= self.p as u64 {
            self.head = self.slot(1);
            self.len -= 1;
        }
        if x.is_finite() {
            while self.len > 0 {
                let back = self.val[self.slot(self.len - 1)];
                if (self.is_max && back <= x) || (!self.is_max && back >= x) {
                    self.len -= 1;
                } else {
                    break;
                }
            }
            let at = self.slot(self.len);
            self.idx[at] = cur;
            self.val[at] = x;
            self.len += 1;
        } else {
            self.last_bad = Some(cur);
        }
        if self.n < self.p as u64 || self.last_bad.is_some_and(|b| cur - b < self.p as u64) {
            return f64::NAN;
        }
        self.val[self.head]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fsum::fsum;
    use crate::quant::kernel::SliceHist;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    fn runner(seed: u32) -> TestRunner {
        TestRunner::new(Config {
            cases: 200,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(seed as u64),
            ..Config::default()
        })
    }

    fn exact_var(w: &[f64], ddof: usize) -> f64 {
        let n = w.len() as f64;
        let mean = fsum(w.iter().copied()) / n;
        fsum(w.iter().map(|v| (v - mean) * (v - mean))) / (w.len() - ddof) as f64
    }

    #[test]
    fn variance_survives_a_huge_spike_leaving_the_window() {
        let mut r = runner(1);
        let strategy = (2usize..30, 40usize..120, prop::collection::vec(-1.0f64..1.0, 200));
        r.run(&strategy, |(p, spike_at, noise)| {
            let mut data = noise;
            data[spike_at] = 1e9;
            let mut rv = RollVar::new(p, 0);
            for i in 0..data.len() {
                let h = SliceHist::<1>::new([&data], i);
                let got = rv.push(data[i], &h, 0);
                if i >= spike_at + p {
                    let want = exact_var(&data[i + 1 - p..=i], 0);
                    let tol = 1e-9 * want.abs().max(f64::MIN_POSITIVE);
                    prop_assert!((got - want).abs() <= tol, "i={i} got={got} want={want}");
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn variance_of_a_flat_window_is_exactly_zero() {
        for c in [0.1, 3.0, -7.3e5, 1e-300] {
            let mut data = vec![1.0, 2.5, 9.0];
            data.extend(std::iter::repeat_n(c, 30));
            let p = 5;
            let mut rv = RollVar::new(p, 1);
            for i in 0..data.len() {
                let h = SliceHist::<1>::new([&data], i);
                let got = rv.push(data[i], &h, 0);
                if i >= 3 + p - 1 {
                    assert_eq!(got.to_bits(), 0.0f64.to_bits(), "c={c} i={i}");
                }
            }
        }
    }

    #[test]
    fn variance_recovers_after_a_nan_leaves() {
        let mut data: Vec<f64> = (0..40).map(|v| ((v * 7) % 11) as f64).collect();
        data[10] = f64::NAN;
        let p = 4;
        let mut rv = RollVar::new(p, 0);
        for i in 0..data.len() {
            let h = SliceHist::<1>::new([&data], i);
            let got = rv.push(data[i], &h, 0);
            if i < p - 1 || (10..10 + p).contains(&i) {
                assert!(got.is_nan(), "i={i}");
            } else {
                let want = exact_var(&data[i + 1 - p..=i], 0);
                assert!((got - want).abs() <= 1e-12 * want.max(1.0), "i={i} {got} {want}");
            }
        }
    }

    #[test]
    fn rolling_sum_is_compensated_against_exact() {
        let data: Vec<f64> = (0..3000).map(|i| if i % 97 == 0 { 1e12 } else { 0.1 }).collect();
        let p = 50;
        let mut rs = RollSum::new(p);
        for i in 0..data.len() {
            let h = SliceHist::<1>::new([&data], i);
            let got = rs.push(data[i], &h, 0);
            if i >= p - 1 {
                let want = fsum(data[i + 1 - p..=i].iter().copied());
                assert!((got - want).abs() <= 1e-9 * want.abs(), "i={i} {got} {want}");
            }
        }
    }

    fn naive_ext(w: &[f64], is_max: bool) -> f64 {
        if w.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        w.iter().copied().fold(if is_max { f64::MIN } else { f64::MAX }, |a, b| {
            if is_max {
                a.max(b)
            } else {
                a.min(b)
            }
        })
    }

    #[test]
    fn extrema_equal_the_naive_computation() {
        let mut r = runner(2);
        let value = prop_oneof![9 => -100.0f64..100.0, 1 => Just(f64::NAN)];
        let strategy = (1usize..20, prop::collection::vec(value, 1..120));
        r.run(&strategy, |(p, data)| {
            for is_max in [true, false] {
                let mut e = RollExt::new(p, is_max);
                for i in 0..data.len() {
                    let got = e.push(data[i]);
                    let want =
                        if i + 1 >= p { naive_ext(&data[i + 1 - p..=i], is_max) } else { f64::NAN };
                    prop_assert!(
                        got == want || (got.is_nan() && want.is_nan()),
                        "p={p} max={is_max} i={i} {got} {want}"
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    }
}
