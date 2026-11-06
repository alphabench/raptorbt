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
    /// Knuth's branch-free two-sum: `c` gets the exact rounding error, the same value the
    /// larger-operand-first form produces, without a data-dependent branch.
    #[inline(always)]
    fn add(&mut self, v: f64) {
        let t = self.s + v;
        let bb = t - self.s;
        self.c += (self.s - (t - bb)) + (v - bb);
        self.s = t;
    }

    #[inline(always)]
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
    /// The step count of the next exact re-sum: a multiple of the interval, so batch and
    /// streaming re-sum on the same bar, without a division per bar.
    next_resum: u64,
}

impl RollSum {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "window of at least 1");
        Self { p, acc: Neumaier::default(), bad: 0, n: 0, next_resum: resum_interval(p) }
    }

    #[inline(always)]
    pub fn push<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) -> f64 {
        self.n += 1;
        // A non-finite value is counted and adds 0, so no select on an input sits on the
        // accumulator's dependency chain.
        let fx = x.is_finite();
        self.acc.add(if fx { x } else { 0.0 });
        self.bad += usize::from(!fx);
        if self.n > self.p as u64 {
            let old = h.ago(input, self.p);
            let fo = old.is_finite();
            self.acc.add(if fo { -old } else { 0.0 });
            self.bad -= usize::from(!fo);
        }
        if self.n >= self.next_resum {
            self.next_resum += resum_interval(self.p);
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

    #[inline(always)]
    pub fn push<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) -> f64 {
        self.sum.push(x, h, input) / self.sum.p as f64
    }
}

/// Sums over a window of the deviations `d = v - base` from a reference near the window's mean:
/// `sd = sum d`, `sdd = sum d^2` and `swd = sum i * d_i` with weight `i = 1` on the oldest
/// value. Variance, the least-squares line and the linear-weighted mean all come out of them in
/// O(1) a bar, and because `d` is the size of the window's spread and not of its level, none of
/// them loses digits to a high price (a window at 130000 moving by 1 is as exact as one at 0).
///
/// The sums are rebuilt exactly from the window (and `base` moved to its mean) on four
/// occasions, all decided from the state alone, so batch and streaming rebuild on the same bar:
/// the first full window and each `max(p, 1024)` steps after it; when the window recovers from
/// a non-finite value; when `base` has drifted from the mean by more than a few spreads
/// (`sdd` over 8 times the centred sum, where the centred form `sdd - sd^2 / p` would cancel);
/// and when the centred sum falls below 1/64 of its peak since the last rebuild (a large value
/// left the window, and the rounding it left in `sdd` would dominate). Between rebuilds the sums
/// are plain additions of terms of the window's own size, so the error is a few ulps of the
/// spread. A window of one repeated value reports itself flat, with spread exactly 0.
#[derive(Debug, Clone)]
pub struct ShiftedWin {
    p: usize,
    n: u64,
    bad: usize,
    dirty: bool,
    base: f64,
    sd: f64,
    sdd: f64,
    swd: f64,
    peak: f64,
    last: f64,
    run: usize,
    next_resum: u64,
    /// `1 / p`, for the rebuild test (which needs the centred sum only roughly).
    inv_p: f64,
}

impl ShiftedWin {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "window of at least 1");
        Self {
            p,
            n: 0,
            bad: 0,
            dirty: false,
            base: 0.0,
            sd: 0.0,
            sdd: 0.0,
            swd: 0.0,
            peak: 0.0,
            last: f64::NAN,
            run: 0,
            next_resum: resum_interval(p),
            inv_p: 1.0 / p as f64,
        }
    }

    /// Take the bar `x`. `old` is the value `p` bars back (read only once the window is
    /// full); `at(j)` is the window's `j`-th value, oldest first and `x` last, read only when
    /// the sums are rebuilt. Returns whether the window is full and holds only finite values;
    /// the readers below are meaningful only then.
    #[inline(always)]
    pub fn push(&mut self, x: f64, old: f64, at: impl Fn(usize) -> f64) -> bool {
        self.n += 1;
        let fx = x.is_finite();
        self.bad += usize::from(!fx);
        self.run = if fx {
            if x == self.last {
                self.run + 1
            } else {
                1
            }
        } else {
            0
        };
        self.last = x;
        let p = self.p as u64;
        if self.n > p && !old.is_finite() {
            self.bad -= 1;
        }
        if self.n < p {
            return false;
        }
        if self.bad > 0 {
            self.dirty = true;
            return false;
        }
        if self.dirty || self.n == p || self.n >= self.next_resum {
            if self.n >= self.next_resum {
                self.next_resum += resum_interval(self.p);
            }
            self.resum(at);
            return true;
        }
        let (dn, dold) = (x - self.base, old - self.base);
        self.swd += self.p as f64 * dn - self.sd;
        self.sd += dn - dold;
        self.sdd += (dn - dold) * (dn + dold);
        if self.run < self.p {
            let m2 = self.sdd - self.sd * self.sd * self.inv_p;
            if m2 > self.peak {
                self.peak = m2;
            }
            if m2 < 0.0 || self.sdd > 8.0 * m2 || m2 < self.peak * (1.0 / 64.0) {
                self.resum(at);
            }
        }
        true
    }

    fn resum(&mut self, at: impl Fn(usize) -> f64) {
        let mut s = Neumaier::default();
        for j in 0..self.p {
            s.add(at(j));
        }
        self.base = s.value() / self.p as f64;
        let mut sd = Neumaier::default();
        let (mut sdd, mut swd) = (0.0, 0.0);
        for j in 0..self.p {
            let d = at(j) - self.base;
            sd.add(d);
            sdd += d * d;
            swd += (j + 1) as f64 * d;
        }
        self.sd = sd.value();
        self.sdd = sdd;
        self.swd = swd;
        self.peak = (sdd - self.sd * self.sd / self.p as f64).max(0.0);
        self.dirty = false;
    }

    /// The window holds one repeated value.
    pub fn flat(&self) -> bool {
        self.run >= self.p
    }

    pub fn mean(&self) -> f64 {
        if self.flat() {
            self.last
        } else {
            self.base + self.sd / self.p as f64
        }
    }

    /// Sum of squared deviations from the mean; exactly 0 for a flat window.
    pub fn m2(&self) -> f64 {
        if self.flat() {
            0.0
        } else {
            (self.sdd - self.sd * self.sd / self.p as f64).max(0.0)
        }
    }

    /// Mean with weights `1..=p`, the newest heaviest.
    pub fn weighted_mean(&self) -> f64 {
        if self.flat() {
            self.last
        } else {
            self.base + self.swd / ((self.p * (self.p + 1)) as f64 / 2.0)
        }
    }

    /// Slope per bar of the least-squares line through the window; exactly 0 when flat.
    pub fn slope(&self) -> f64 {
        if self.flat() {
            return 0.0;
        }
        let n = self.p as f64;
        let xbar = (n - 1.0) / 2.0;
        let sxx = n * (n * n - 1.0) / 12.0;
        ((self.swd - self.sd) - xbar * self.sd) / sxx
    }
}

/// [`ShiftedWin`]'s counterpart for a window of pairs: the deviation sums of two series from a
/// reference each (`sx`, `sy`, `sxx`, `syy`) and of their products (`sxy`), for correlation,
/// covariance and beta in O(1) a bar. The rebuild rules are the same, applied to each series; a
/// pair counts as non-finite when either value is.
#[derive(Debug, Clone)]
pub struct PairWin {
    p: usize,
    n: u64,
    bad: usize,
    dirty: bool,
    base: [f64; 2],
    s: [f64; 2],
    ss: [f64; 2],
    sxy: f64,
    peak: [f64; 2],
    last: [f64; 2],
    run: [usize; 2],
    next_resum: u64,
}

impl PairWin {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "window of at least 1");
        Self {
            p,
            n: 0,
            bad: 0,
            dirty: false,
            base: [0.0; 2],
            s: [0.0; 2],
            ss: [0.0; 2],
            sxy: 0.0,
            peak: [0.0; 2],
            last: [f64::NAN; 2],
            run: [0; 2],
            next_resum: resum_interval(p),
        }
    }

    /// Take the pair `x`; `old` is the pair `p` bars back (read only once the window is full)
    /// and `at(j)` the window's `j`-th pair, oldest first and `x` last (read on rebuilds). A
    /// pair with a non-finite value is a gap. Returns whether the window is full and clean.
    #[inline(always)]
    pub fn push(&mut self, x: [f64; 2], old: [f64; 2], at: impl Fn(usize) -> [f64; 2]) -> bool {
        self.n += 1;
        let fin = x[0].is_finite() && x[1].is_finite();
        self.bad += usize::from(!fin);
        for ((run, last), v) in self.run.iter_mut().zip(&mut self.last).zip(x) {
            *run = if !fin {
                0
            } else if v == *last {
                *run + 1
            } else {
                1
            };
            *last = v;
        }
        let p = self.p as u64;
        if self.n > p && !(old[0].is_finite() && old[1].is_finite()) {
            self.bad -= 1;
        }
        if self.n < p {
            return false;
        }
        if self.bad > 0 {
            self.dirty = true;
            return false;
        }
        if self.dirty || self.n == p || self.n >= self.next_resum {
            if self.n >= self.next_resum {
                self.next_resum += resum_interval(self.p);
            }
            self.resum(at);
            return true;
        }
        let dn = [x[0] - self.base[0], x[1] - self.base[1]];
        let dold = [old[0] - self.base[0], old[1] - self.base[1]];
        self.sxy += dn[0] * dn[1] - dold[0] * dold[1];
        let inv_p = 1.0 / self.p as f64;
        let mut rebuild = false;
        for k in 0..2 {
            self.s[k] += dn[k] - dold[k];
            self.ss[k] += (dn[k] - dold[k]) * (dn[k] + dold[k]);
            if self.run[k] < self.p {
                let m2 = self.ss[k] - self.s[k] * self.s[k] * inv_p;
                if m2 > self.peak[k] {
                    self.peak[k] = m2;
                }
                rebuild |= m2 < 0.0 || self.ss[k] > 8.0 * m2 || m2 < self.peak[k] * (1.0 / 64.0);
            }
        }
        if rebuild {
            self.resum(at);
        }
        true
    }

    fn resum(&mut self, at: impl Fn(usize) -> [f64; 2]) {
        let mut m = [Neumaier::default(); 2];
        for j in 0..self.p {
            let v = at(j);
            m[0].add(v[0]);
            m[1].add(v[1]);
        }
        let pf = self.p as f64;
        self.base = [m[0].value() / pf, m[1].value() / pf];
        let mut s = [Neumaier::default(); 2];
        let (mut ss, mut sxy) = ([0.0; 2], 0.0);
        for j in 0..self.p {
            let v = at(j);
            let d = [v[0] - self.base[0], v[1] - self.base[1]];
            s[0].add(d[0]);
            s[1].add(d[1]);
            ss[0] += d[0] * d[0];
            ss[1] += d[1] * d[1];
            sxy += d[0] * d[1];
        }
        self.s = [s[0].value(), s[1].value()];
        self.ss = ss;
        self.sxy = sxy;
        self.peak = std::array::from_fn(|k| (ss[k] - self.s[k] * self.s[k] / pf).max(0.0));
        self.dirty = false;
    }

    /// Means of the two series.
    pub fn means(&self) -> [f64; 2] {
        let pf = self.p as f64;
        std::array::from_fn(|k| {
            if self.run[k] >= self.p {
                self.last[k]
            } else {
                self.base[k] + self.s[k] / pf
            }
        })
    }

    /// Centred sums `[sxx, syy, sxy]`. A flat series has zero spread and zero covariance with
    /// anything.
    pub fn centred(&self) -> [f64; 3] {
        let pf = self.p as f64;
        let flat = [self.run[0] >= self.p, self.run[1] >= self.p];
        let c = |k: usize| {
            if flat[k] {
                0.0
            } else {
                (self.ss[k] - self.s[k] * self.s[k] / pf).max(0.0)
            }
        };
        let sxy = if flat[0] || flat[1] { 0.0 } else { self.sxy - self.s[0] * self.s[1] / pf };
        [c(0), c(1), sxy]
    }
}

/// Rolling variance over a [`ShiftedWin`]. A window of one repeated value has variance exactly
/// 0.
#[derive(Debug, Clone)]
pub struct RollVar {
    w: ShiftedWin,
    ddof: usize,
}

impl RollVar {
    pub fn new(p: usize, ddof: usize) -> Self {
        Self { w: ShiftedWin::new(p), ddof }
    }

    #[inline(always)]
    pub fn push<const I: usize, H: Hist<I>>(&mut self, x: f64, h: &H, input: usize) -> f64 {
        let p = self.w.p;
        if !self.w.push(
            x,
            h.ago(input, p),
            |j| if j + 1 == p { x } else { h.ago(input, p - 1 - j) },
        ) || p <= self.ddof
        {
            return f64::NAN;
        }
        self.w.m2() / (p - self.ddof) as f64
    }
}

/// Windows up to this long are scanned rather than kept in a deque: a rescan of a few values
/// beats the deque's data-dependent pops, and it is rare.
const SCAN_MAX: usize = 32;

/// Rolling min or max. Windows of up to [`SCAN_MAX`] values keep the extreme and where it is,
/// and rescan the window (oldest first, so the newest of equal extremes wins) only when it
/// leaves; longer windows use a monotonic deque of (index, value) in a ring of the window's size
/// rounded up to a power of two, amortised O(1). Both give the same value on every input. Memory
/// is allocated once.
#[derive(Debug, Clone)]
pub struct RollExt {
    p: u64,
    is_max: bool,
    n: u64,
    /// Bars to go until the last non-finite value has left the window; 0 when none is in it.
    bad_left: u64,
    kind: ExtKind,
}

#[derive(Debug, Clone)]
enum ExtKind {
    /// The last `p` values twice over (`buf[i]` and `buf[i + p]`), so the window is one slice;
    /// a non-finite value (and a bar not yet seen) is stored as the sentinel, -inf for a max and
    /// +inf for a min, so it never wins and needs no test.
    Scan {
        buf: Box<[f64]>,
        slot: usize,
        best: f64,
        at: u64,
    },
    Deque {
        slots: Box<[(u64, f64)]>,
        mask: usize,
        head: usize,
        len: usize,
    },
}

impl RollExt {
    pub fn new(p: usize, is_max: bool) -> Self {
        assert!(p >= 1, "window of at least 1");
        let sentinel = if is_max { f64::NEG_INFINITY } else { f64::INFINITY };
        let kind = if p <= SCAN_MAX {
            ExtKind::Scan {
                buf: vec![sentinel; 2 * p].into_boxed_slice(),
                slot: p - 1,
                best: sentinel,
                at: 0,
            }
        } else {
            let cap = p.next_power_of_two();
            ExtKind::Deque {
                slots: vec![(0, f64::NAN); cap].into_boxed_slice(),
                mask: cap - 1,
                head: 0,
                len: 0,
            }
        };
        Self { p: p as u64, is_max, n: 0, bad_left: 0, kind }
    }

    /// Bars since the extreme set by the last [`push`](Self::push) (0 when it is the newest
    /// bar); ties keep the latest bar. Meaningful when that push returned a value.
    #[inline(always)]
    pub fn since(&self) -> usize {
        let newest = self.n - 1;
        let at = match &self.kind {
            ExtKind::Scan { at, .. } => *at,
            ExtKind::Deque { slots, head, .. } => slots[*head].0,
        };
        (newest - at) as usize
    }

    /// Heap owned: the ring.
    pub fn heap_bytes(&self) -> usize {
        match &self.kind {
            ExtKind::Scan { buf, .. } => std::mem::size_of_val(&**buf),
            ExtKind::Deque { slots, .. } => std::mem::size_of_val(&**slots),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, x: f64) -> f64 {
        let cur = self.n;
        self.n += 1;
        self.bad_left = self.bad_left.saturating_sub(1);
        let fx = x.is_finite();
        if !fx {
            self.bad_left = self.p;
        }
        let (p, is_max) = (self.p, self.is_max);
        let value = match &mut self.kind {
            ExtKind::Scan { buf, slot, best, at } => {
                let pu = p as usize;
                let sentinel = if is_max { f64::NEG_INFINITY } else { f64::INFINITY };
                *slot = if *slot + 1 == pu { 0 } else { *slot + 1 };
                let v = if fx { x } else { sentinel };
                buf[*slot] = v;
                buf[*slot + pu] = v;
                if *at + p <= cur {
                    // The extreme left: its value by a branch-free pass, then the newest bar
                    // holding it.
                    let win = &buf[*slot + 1..*slot + 1 + pu];
                    let mut m = sentinel;
                    if is_max {
                        for &w in win {
                            m = if w > m { w } else { m };
                        }
                    } else {
                        for &w in win {
                            m = if w < m { w } else { m };
                        }
                    }
                    *best = m;
                    *at = cur + 1 + win.iter().rposition(|&w| w == m).unwrap_or(pu - 1) as u64 - p;
                } else if (is_max && v >= *best) || (!is_max && v <= *best) {
                    *best = v;
                    *at = cur;
                }
                *best
            }
            ExtKind::Deque { slots, mask, head, len } => {
                // Indices rise by one a bar, so at most one entry (index `cur - p`) leaves.
                if *len > 0 && cur - slots[*head].0 >= p {
                    *head = (*head + 1) & *mask;
                    *len -= 1;
                }
                if fx {
                    while *len > 0 {
                        let back = slots[(*head + *len - 1) & *mask].1;
                        if (is_max && back <= x) || (!is_max && back >= x) {
                            *len -= 1;
                        } else {
                            break;
                        }
                    }
                    slots[(*head + *len) & *mask] = (cur, x);
                    *len += 1;
                }
                slots[*head].1
            }
        };
        if self.n < p || self.bad_left > 0 {
            f64::NAN
        } else {
            value
        }
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
        let strategy = (1usize..70, prop::collection::vec(value, 1..200));
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

    /// A price near 130000 moving by under 1: the case a running sum of squares loses.
    #[test]
    fn window_sums_are_exact_at_a_high_price_level() {
        let mut r = runner(3);
        let strategy = (2usize..60, prop::collection::vec(0.0f64..1.0, 40..400));
        r.run(&strategy, |(p, noise)| {
            let data: Vec<f64> = noise.iter().map(|v| 130000.0 + v).collect();
            let mut rv = RollVar::new(p, 0);
            let mut sw = ShiftedWin::new(p);
            for i in 0..data.len() {
                let h = SliceHist::<1>::new([&data], i);
                let got = rv.push(data[i], &h, 0);
                let ok = sw.push(data[i], h.ago(0, p), |j| {
                    if j + 1 == p {
                        data[i]
                    } else {
                        h.ago(0, p - 1 - j)
                    }
                });
                if i + 1 >= p {
                    prop_assert!(ok);
                    let w = &data[i + 1 - p..=i];
                    let want = exact_var(w, 0);
                    prop_assert!((got - want).abs() <= 1e-11 * want, "i={i} got={got} want={want}");
                    let mean = fsum(w.iter().copied()) / p as f64;
                    prop_assert!((sw.mean() - mean).abs() <= 1e-11, "mean i={i}");
                    let n = p as f64;
                    let xbar = (n - 1.0) / 2.0;
                    let num =
                        fsum(w.iter().enumerate().map(|(k, v)| (k as f64 - xbar) * (v - mean)));
                    let want_slope = num / (n * (n * n - 1.0) / 12.0);
                    prop_assert!(
                        (sw.slope() - want_slope).abs() <= 1e-11 * want_slope.abs().max(1e-3),
                        "slope i={i} {} {want_slope}",
                        sw.slope()
                    );
                    let wsum: f64 = (p * (p + 1)) as f64 / 2.0;
                    let wmean =
                        fsum(w.iter().enumerate().map(|(k, v)| (k + 1) as f64 * (v - mean))) / wsum
                            + mean;
                    prop_assert!((sw.weighted_mean() - wmean).abs() <= 1e-11, "wmean i={i}");
                } else {
                    prop_assert!(got.is_nan() && !ok);
                }
            }
            Ok(())
        })
        .unwrap();
    }

    /// After a spike of 1e12 leaves, the sums are rebuilt and the variance is back to exact.
    #[test]
    fn window_sums_recover_after_a_spike() {
        let mut r = runner(4);
        let strategy = (3usize..40, 50usize..150, prop::collection::vec(100.0f64..101.0, 400));
        r.run(&strategy, |(p, spike_at, base)| {
            let mut data = base;
            data[spike_at] = 1e12;
            let mut rv = RollVar::new(p, 0);
            for i in 0..data.len() {
                let h = SliceHist::<1>::new([&data], i);
                let got = rv.push(data[i], &h, 0);
                if i >= spike_at + p {
                    let want = exact_var(&data[i + 1 - p..=i], 0);
                    prop_assert!((got - want).abs() <= 1e-11 * want, "i={i} got={got} want={want}");
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn pair_sums_are_exact_at_a_high_price_level() {
        let mut r = runner(5);
        let strategy = (2usize..40, prop::collection::vec((0.0f64..1.0, 0.0f64..1.0), 40..300));
        r.run(&strategy, |(p, noise)| {
            let a: Vec<f64> = noise.iter().map(|v| 130000.0 + v.0).collect();
            let b: Vec<f64> = noise.iter().map(|v| 52000.0 + v.1 + v.0).collect();
            let mut pw = PairWin::new(p);
            for i in 0..a.len() {
                let h = SliceHist::<2>::new([&a, &b], i);
                let x = [a[i], b[i]];
                let ok = pw.push(x, [h.ago(0, p), h.ago(1, p)], |j| {
                    if j + 1 == p {
                        x
                    } else {
                        [h.ago(0, p - 1 - j), h.ago(1, p - 1 - j)]
                    }
                });
                if i + 1 >= p {
                    prop_assert!(ok);
                    let (wa, wb) = (&a[i + 1 - p..=i], &b[i + 1 - p..=i]);
                    let n = p as f64;
                    let (ma, mb) = (fsum(wa.iter().copied()) / n, fsum(wb.iter().copied()) / n);
                    let sxx = fsum(wa.iter().map(|v| (v - ma) * (v - ma)));
                    let syy = fsum(wb.iter().map(|v| (v - mb) * (v - mb)));
                    let sxy = fsum(wa.iter().zip(wb).map(|(u, v)| (u - ma) * (v - mb)));
                    let [gxx, gyy, gxy] = pw.centred();
                    prop_assert!((gxx - sxx).abs() <= 1e-11 * sxx, "sxx i={i}");
                    prop_assert!((gyy - syy).abs() <= 1e-11 * syy, "syy i={i}");
                    prop_assert!((gxy - sxy).abs() <= 1e-11 * (sxx * syy).sqrt(), "sxy i={i}");
                }
            }
            Ok(())
        })
        .unwrap();
    }
}
