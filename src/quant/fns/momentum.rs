//! Momentum indicators: TA-Lib's "Momentum Indicators" group (30 functions) and raptorbt's own
//! fast `stochastic`.
//!
//! Names that raptorbt already had carry a trailing `compat` parameter, `"native"` (default) or
//! `"talib"`:
//!
//! * `rsi`, `macd` replicate the streaming cores (`StreamingRsi`, `StreamingMacd`) bit for bit on
//!   finite data; `mfi`, `adx`, `plus_di`, `minus_di` replicate the array functions in
//!   `indicators::volume` and `indicators::strength` bit for bit. `compat="talib"` matches
//!   TA-Lib. The native forms differ from TA-Lib in the seed (the directional system sums
//!   `period` raw movements where TA-Lib sums `period - 1` and then smooths; MACD seeds its fast
//!   average on the first bar the slow one is ready in TA-Lib) and in a flat market (RSI 100 and
//!   MFI 100 against TA-Lib's 0, the DI lines NaN against 0).
//! * `roc` is pinned to `StreamingRoc` and agrees with TA-Lib; it has no `compat`.
//! * `stochastic` is raptorbt's fast stochastic, pinned to `indicators::momentum::stochastic`.
//!   TA-Lib's `STOCH` (slow) and `STOCHF` (fast) are the new names `stoch` and `stochf`.
//!
//! Every other function follows TA-Lib's convention on clean data: the same first valid bar, the
//! same seeds, the same value on a flat range. The NaN policy of `design-core.md` is applied on
//! top: windowed kernels are NaN while a NaN is in the window, recursive ones skip the bar, and a
//! composite that smooths one series with another starts the second on the first finite value of
//! the first (a NaN that has not been preceded by a value never seeds anything).
//!
//! Moving averages selected by `matype` are [`MaCore`]; MAMA (7) is refused until the Hilbert
//! slice lands.

use crate::quant::fns::overlap::{MaCore, MaKind};
use crate::quant::fns::vol_price::true_range;
use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Legacy, Memory, NanPolicy,
    NanRule, OutKind, ParamDef, ParamKind, ParamVal, Reference, Relation,
};
use crate::quant::window::RollExt;

const MAX_PERIOD: i64 = 1 << 24;
/// `cci` and `mfi` walk their window on every bar (their native arithmetic is a fresh sum), so a
/// period past this costs more than it is worth.
const MAX_WALKED: i64 = 100_000;

fn finite_or_nan(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NAN
    }
}

fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// What `Iterator::sum` starts from. The legacy array functions sum with it, so a native kernel
/// that must equal them bit for bit (the sign of a zero included) starts from the same value.
fn sum_identity() -> f64 {
    std::iter::empty::<f64>().sum()
}

// ---------------------------------------------------------------------------------- averages

/// SMA-seeded EMA with an explicit smoothing constant: `Ema`'s arithmetic (and the streaming
/// core's), `alpha * x + (1 - alpha) * prev`, or with `classic` TA-Lib's, `(x - prev) * alpha +
/// prev`, which rounds differently (the difference of two close averages, as in MACD on a quiet
/// market, shows it). `alpha` is 2 / (p + 1), or TA-Lib's fixed 0.15 and 0.075 for `macdfix`. A
/// non-finite value neither seeds nor advances the state.
struct EmaA {
    p: usize,
    alpha: f64,
    classic: bool,
    value: Option<f64>,
    seed_sum: f64,
    seen: usize,
}

impl EmaA {
    fn new(p: usize, alpha: f64, classic: bool) -> Self {
        Self { p: p.max(1), alpha, classic, value: None, seed_sum: 0.0, seen: 0 }
    }

    fn step(&mut self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        match self.value {
            Some(prev) => {
                let v = if self.classic {
                    (x - prev).mul_add(self.alpha, prev)
                } else {
                    self.alpha * x + (1.0 - self.alpha) * prev
                };
                self.value = Some(v);
                v
            }
            None => {
                self.seed_sum += x;
                self.seen += 1;
                if self.seen == self.p {
                    let v = self.seed_sum / self.p as f64;
                    self.value = Some(v);
                    v
                } else {
                    f64::NAN
                }
            }
        }
    }
}

/// TA-Lib's simple moving average, in TA-Lib's order: add the new value, read the total, then
/// subtract the value leaving. `Sma` and the overlap cores subtract first (the streaming core's
/// order), which rounds differently; where a window of equal values follows large ones the two
/// leave different residues around 0, and the oscillators built on this one are compared with
/// TA-Lib's. A window holding a non-finite value is NaN until it leaves.
struct TaSma {
    p: usize,
    ring: Box<[f64]>,
    n: u64,
    total: f64,
    bad: usize,
}

impl TaSma {
    fn new(p: usize) -> Self {
        Self { p, ring: vec![f64::NAN; p].into_boxed_slice(), n: 0, total: 0.0, bad: 0 }
    }

    fn step(&mut self, x: f64) -> f64 {
        let p = self.p as u64;
        let slot = (self.n % p) as usize;
        self.ring[slot] = x;
        if x.is_finite() {
            self.total += x;
        } else {
            self.bad += 1;
        }
        self.n += 1;
        if self.n < p {
            return f64::NAN;
        }
        let out = if self.bad > 0 { f64::NAN } else { self.total / self.p as f64 };
        let leaving = self.ring[(self.n % p) as usize];
        if leaving.is_finite() {
            self.total -= leaving;
        } else {
            self.bad -= 1;
        }
        out
    }
}

enum Avg {
    Ema(EmaA),
    Sma(TaSma),
    Core(Box<MaCore>),
}

/// An average over a series that starts at bar `offset` of the finite values it sees. TA-Lib
/// starts each average of a composite so that its first output lands on the composite's first
/// bar: a fast EMA inside MACD is seeded by the mean of the `fast` values ending at the slow
/// EMA's first bar, not by the first `fast` values of the series. A non-finite value is fed to an
/// average that has started (so windowed kinds see the NaN) and ignored by one that has not.
struct Fed {
    avg: Avg,
    offset: usize,
    seen: usize,
}

impl Fed {
    fn ema(p: usize, alpha: f64, classic: bool, offset: usize) -> Self {
        Self { avg: Avg::Ema(EmaA::new(p, alpha, classic)), offset, seen: 0 }
    }

    fn core(kind: MaKind, p: usize, offset: usize) -> Result<Self, String> {
        let avg = if kind == MaKind::Sma && p > 1 {
            Avg::Sma(TaSma::new(p))
        } else {
            Avg::Core(Box::new(MaCore::new(kind, p)?))
        };
        Ok(Self { avg, offset, seen: 0 })
    }

    fn push(&mut self, x: f64) -> f64 {
        if x.is_finite() {
            let i = self.seen;
            self.seen += 1;
            if i < self.offset {
                return f64::NAN;
            }
        } else if self.seen <= self.offset {
            return f64::NAN;
        }
        match &mut self.avg {
            Avg::Ema(e) => e.step(x),
            Avg::Sma(a) => a.step(x),
            Avg::Core(c) => c.step(x),
        }
    }
}

// ----------------------------------------------------------------------------- rate of change

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChangeKind {
    Mom,
    Roc,
    Rocp,
    Rocr,
    Rocr100,
}

/// `mom`, `roc`, `rocp`, `rocr`, `rocr100` against the value `period` bars back.
///
/// `roc` is `StreamingRoc`'s arithmetic, `(x - base) / base * 100`, and on a zero base it keeps
/// the last value like the streaming core does. The other three follow TA-Lib: a zero base gives
/// 0.
pub struct Change {
    kind: ChangeKind,
    p: usize,
    last: f64,
}

impl Kernel<1, 1> for Change {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let x = x[0];
        let base = h.ago(0, self.p);
        if !(x.is_finite() && base.is_finite()) {
            return [f64::NAN];
        }
        let v = match self.kind {
            ChangeKind::Mom => x - base,
            ChangeKind::Roc => {
                if base != 0.0 {
                    self.last = (x - base) / base * 100.0;
                }
                self.last
            }
            ChangeKind::Rocp => {
                if base != 0.0 {
                    (x - base) / base
                } else {
                    0.0
                }
            }
            ChangeKind::Rocr => {
                if base != 0.0 {
                    x / base
                } else {
                    0.0
                }
            }
            ChangeKind::Rocr100 => {
                if base != 0.0 {
                    x / base * 100.0
                } else {
                    0.0
                }
            }
        };
        [finite_or_nan(v)]
    }
}

/// (close - open) / (high - low); 0 when the bar has no range.
pub struct Bop;

impl Kernel<4, 1> for Bop {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<4>>(&mut self, x: [f64; 4], _h: &H) -> [f64; 1] {
        if !all_finite(&x) {
            return [f64::NAN];
        }
        let range = x[1] - x[2];
        if range > 0.0 {
            [finite_or_nan((x[3] - x[0]) / range)]
        } else {
            [0.0]
        }
    }
}

// ------------------------------------------------------------------------------ gain and loss

/// Wilder-smoothed average gain and loss of the changes between consecutive accepted values.
/// Seeded with the mean of the first `period` changes. `talib` selects the smoothing form
/// `(avg * (p - 1) + x) / p`; the native form is the streaming core's `alpha * x + (1 - alpha) *
/// avg`. Both are the same number up to rounding.
struct GainLoss {
    p: usize,
    talib: bool,
    alpha: f64,
    prev: Option<f64>,
    seed_gain: f64,
    seed_loss: f64,
    seen: usize,
    avg: Option<(f64, f64)>,
}

impl GainLoss {
    fn new(p: usize, talib: bool) -> Self {
        let p = p.max(1);
        Self {
            p,
            talib,
            alpha: 1.0 / p as f64,
            prev: None,
            seed_gain: sum_identity(),
            seed_loss: sum_identity(),
            seen: 0,
            avg: None,
        }
    }

    /// Feed one finite value; the averages once seeded.
    fn push(&mut self, x: f64) -> Option<(f64, f64)> {
        let prev = self.prev.replace(x)?;
        let change = x - prev;
        let (gain, loss) = if change > 0.0 { (change, 0.0) } else { (0.0, -change) };
        match self.avg {
            Some((ag, al)) => {
                let next = if self.talib {
                    let m = (self.p - 1) as f64;
                    ((ag * m + gain) / self.p as f64, (al * m + loss) / self.p as f64)
                } else {
                    (
                        self.alpha * gain + (1.0 - self.alpha) * ag,
                        self.alpha * loss + (1.0 - self.alpha) * al,
                    )
                };
                self.avg = Some(next);
            }
            None => {
                self.seed_gain += gain;
                self.seed_loss += loss;
                self.seen += 1;
                if self.seen == self.p {
                    self.avg =
                        Some((self.seed_gain / self.p as f64, self.seed_loss / self.p as f64));
                }
            }
        }
        self.avg
    }
}

/// TA-Lib's RSI: 100 * gain / (gain + loss), 0 when both are 0.
fn rsi_talib(ag: f64, al: f64) -> f64 {
    let s = ag + al;
    if s != 0.0 {
        100.0 * (ag / s)
    } else {
        0.0
    }
}

/// The streaming core's RSI: 100 when there is no loss, flat or not.
fn rsi_native(ag: f64, al: f64) -> f64 {
    if al == 0.0 {
        100.0
    } else {
        100.0 - 100.0 / (1.0 + ag / al)
    }
}

/// Relative strength index, Wilder smoothing.
pub struct Rsi {
    core: GainLoss,
}

impl Kernel<1, 1> for Rsi {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        if !x[0].is_finite() {
            return [f64::NAN];
        }
        match self.core.push(x[0]) {
            Some((ag, al)) => {
                let v = if self.core.talib { rsi_talib(ag, al) } else { rsi_native(ag, al) };
                [finite_or_nan(v)]
            }
            None => [f64::NAN],
        }
    }
}

/// Chande momentum oscillator: 100 * (gain - loss) / (gain + loss), 0 when both are 0.
pub struct Cmo {
    core: GainLoss,
}

impl Kernel<1, 1> for Cmo {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        if !x[0].is_finite() {
            return [f64::NAN];
        }
        match self.core.push(x[0]) {
            Some((ag, al)) => {
                let s = ag + al;
                [finite_or_nan(if s != 0.0 { 100.0 * ((ag - al) / s) } else { 0.0 })]
            }
            None => [f64::NAN],
        }
    }
}

// --------------------------------------------------------------------------------------- MACD

/// MACD line, signal and histogram. All three are NaN until the signal average is ready, like the
/// streaming core and TA-Lib.
pub struct Macd {
    fast: Fed,
    slow: Fed,
    signal: Fed,
}

impl Macd {
    fn alpha(p: usize) -> f64 {
        2.0 / (p as f64 + 1.0)
    }

    /// The streaming core: each EMA seeded on its own first `period` values.
    fn native(f: usize, s: usize, g: usize) -> Self {
        Self {
            fast: Fed::ema(f, Self::alpha(f), false, 0),
            slow: Fed::ema(s, Self::alpha(s), false, 0),
            signal: Fed::ema(g, Self::alpha(g), false, 0),
        }
    }

    /// TA-Lib: the shorter period is the fast one, and the fast EMA is seeded on the first bar of
    /// the slow EMA.
    fn talib(f: usize, s: usize, g: usize) -> Self {
        let (f, s) = if s < f { (s, f) } else { (f, s) };
        Self {
            fast: Fed::ema(f, Self::alpha(f), true, s - f),
            slow: Fed::ema(s, Self::alpha(s), true, 0),
            signal: Fed::ema(g, Self::alpha(g), true, 0),
        }
    }

    /// TA-Lib's MACDFIX: 12 and 26 with the constants 0.15 and 0.075, not 2 / 13 and 2 / 27.
    fn fixed(g: usize) -> Self {
        Self {
            fast: Fed::ema(12, 0.15, true, 14),
            slow: Fed::ema(26, 0.075, true, 0),
            signal: Fed::ema(g, Self::alpha(g), true, 0),
        }
    }

    /// TA-Lib's MACDEXT: any `matype` for each leg; the slower leg is the one with the longer
    /// period, and each average starts so that both first land on the later lookback.
    fn ext(
        (fk, fp): (MaKind, usize),
        (sk, sp): (MaKind, usize),
        (gk, gp): (MaKind, usize),
    ) -> Result<Self, String> {
        let ((fk, fp), (sk, sp)) =
            if sp < fp { ((sk, sp), (fk, fp)) } else { ((fk, fp), (sk, sp)) };
        let (lf, ls) = (fk.lookback(fp), sk.lookback(sp));
        let l = lf.max(ls);
        Ok(Self {
            fast: Fed::core(fk, fp, l - lf)?,
            slow: Fed::core(sk, sp, l - ls)?,
            signal: Fed::core(gk, gp, 0)?,
        })
    }
}

impl Kernel<1, 3> for Macd {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 3] {
        let f = self.fast.push(x[0]);
        let s = self.slow.push(x[0]);
        let m = f - s;
        let g = self.signal.push(m);
        if g.is_nan() {
            return [f64::NAN; 3];
        }
        [m, g, m - g].map(finite_or_nan)
    }
}

/// APO and PPO: the difference of two averages, absolute or as a percentage of the slow one.
pub struct Po {
    ppo: bool,
    fast: Fed,
    slow: Fed,
}

impl Po {
    /// TA-Lib computes both averages over the whole series (the fast one is not delayed to start
    /// beside the slow one as in MACD), so the oscillator is NaN until the slower one is ready.
    fn new(ppo: bool, kind: MaKind, f: usize, s: usize) -> Result<Self, String> {
        let (f, s) = if s < f { (s, f) } else { (f, s) };
        Ok(Self { ppo, fast: Fed::core(kind, f, 0)?, slow: Fed::core(kind, s, 0)? })
    }
}

impl Kernel<1, 1> for Po {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        let f = self.fast.push(x[0]);
        let s = self.slow.push(x[0]);
        if f.is_nan() || s.is_nan() {
            return [f64::NAN];
        }
        let v = if !self.ppo {
            f - s
        } else if s != 0.0 {
            (f - s) / s * 100.0
        } else {
            0.0
        };
        [finite_or_nan(v)]
    }
}

/// 1-bar rate of change, in percent, of a triple-smoothed EMA.
pub struct Trix {
    e: [Fed; 3],
    prev: Option<f64>,
}

impl Kernel<1, 1> for Trix {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        let v1 = self.e[0].push(x[0]);
        let v2 = self.e[1].push(v1);
        let v3 = self.e[2].push(v2);
        if v3.is_nan() {
            return [f64::NAN];
        }
        let out = match self.prev.replace(v3) {
            Some(p) if p != 0.0 => ((v3 / p) - 1.0) * 100.0,
            Some(_) => 0.0,
            None => f64::NAN,
        };
        [finite_or_nan(out)]
    }
}

// ------------------------------------------------------------------------------ stochastics

/// Raw %K over a window of highs and lows: 100 * (close - lowest low) / (highest high - lowest
/// low), 0 when the range is 0 (TA-Lib's value; the legacy `stochastic` says 50).
struct StochK {
    hh: RollExt,
    ll: RollExt,
}

impl StochK {
    fn new(p: usize) -> Self {
        Self { hh: RollExt::new(p, true), ll: RollExt::new(p, false) }
    }

    fn push(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let hh = self.hh.push(high);
        let ll = self.ll.push(low);
        if !close.is_finite() || hh.is_nan() || ll.is_nan() {
            return f64::NAN;
        }
        // TA-Lib divides by a hundredth of the range rather than multiplying by 100; the two
        // differ in the last bit.
        let diff = (hh - ll) / 100.0;
        finite_or_nan(if diff != 0.0 { (close - ll) / diff } else { 0.0 })
    }
}

/// TA-Lib's STOCHF: fast %K and its `fastd_matype` average. Both are NaN until %D is ready.
pub struct StochF {
    k: StochK,
    d: Fed,
}

impl Kernel<3, 2> for StochF {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 2] {
        let fk = self.k.push(x[0], x[1], x[2]);
        let fd = self.d.push(fk);
        if fd.is_nan() {
            [f64::NAN; 2]
        } else {
            [fk, fd]
        }
    }
}

/// TA-Lib's STOCH: fast %K smoothed by `slowk_matype` is slow %K, smoothed again by
/// `slowd_matype` is slow %D. Both are NaN until %D is ready.
pub struct Stoch {
    k: StochK,
    sk: Fed,
    sd: Fed,
}

impl Kernel<3, 2> for Stoch {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 2] {
        let fk = self.k.push(x[0], x[1], x[2]);
        let slow_k = self.sk.push(fk);
        let slow_d = self.sd.push(slow_k);
        if slow_d.is_nan() {
            [f64::NAN; 2]
        } else {
            [slow_k, slow_d]
        }
    }
}

/// TA-Lib's STOCHRSI: the fast stochastic of the RSI (TA-Lib's, flat gives 0) and its
/// `fastd_matype` average. Both are NaN until %D is ready.
pub struct StochRsi {
    rsi: GainLoss,
    k: StochK,
    d: Fed,
}

impl Kernel<1, 2> for StochRsi {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 2] {
        let r = if x[0].is_finite() {
            self.rsi.push(x[0]).map_or(f64::NAN, |(ag, al)| rsi_talib(ag, al))
        } else {
            f64::NAN
        };
        let fk = self.k.push(r, r, r);
        let fd = self.d.push(fk);
        if fd.is_nan() {
            [f64::NAN; 2]
        } else {
            [fk, fd]
        }
    }
}

/// raptorbt's own fast stochastic (`indicators::momentum::stochastic`): %K over `k_period` bars
/// (50 when the range is 0) and %D, its simple mean over `d_period`. Both are NaN until %D is
/// ready, so the expression has one warm-up; the array function reports %K earlier.
pub struct Stochastic {
    hh: RollExt,
    ll: RollExt,
    ring: Box<[f64]>,
    n: u64,
}

impl Kernel<3, 2> for Stochastic {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 2] {
        let hh = self.hh.push(x[0]);
        let ll = self.ll.push(x[1]);
        let k = if x[2].is_finite() && hh.is_finite() && ll.is_finite() {
            let range = hh - ll;
            if range > 0.0 {
                ((x[2] - ll) / range) * 100.0
            } else {
                50.0
            }
        } else {
            f64::NAN
        };
        let d_period = self.ring.len() as u64;
        self.ring[(self.n % d_period) as usize] = k;
        self.n += 1;
        if self.n < d_period {
            return [f64::NAN; 2];
        }
        let mut sum = 0.0;
        for j in (self.n - d_period)..self.n {
            let v = self.ring[(j % d_period) as usize];
            if !v.is_finite() {
                return [f64::NAN; 2];
            }
            sum += v;
        }
        [finite_or_nan(k), finite_or_nan(sum / d_period as f64)]
    }
}

/// Williams %R: -100 * (highest high - close) / (highest high - lowest low), 0 when the range is 0.
pub struct Willr {
    hh: RollExt,
    ll: RollExt,
}

impl Kernel<3, 1> for Willr {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 1] {
        let hh = self.hh.push(x[0]);
        let ll = self.ll.push(x[1]);
        if !x[2].is_finite() || hh.is_nan() || ll.is_nan() {
            return [f64::NAN];
        }
        let diff = (hh - ll) / -100.0;
        [finite_or_nan(if diff != 0.0 { (hh - x[2]) / diff } else { 0.0 })]
    }
}

// ------------------------------------------------------------------------------------- Aroon

/// Rolling max or min over `p` bars that also reports how many bars ago it was set. Ties keep
/// the latest bar. Same deque as `RollExt`, which does not expose the index.
struct ExtIdx {
    p: usize,
    is_max: bool,
    idx: Box<[u64]>,
    val: Box<[f64]>,
    head: usize,
    len: usize,
    n: u64,
    last_bad: Option<u64>,
}

impl ExtIdx {
    fn new(p: usize, is_max: bool) -> Self {
        let p = p.max(1);
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

    /// Bars since the extreme of the last `p` bars; None before the window is full or while it
    /// holds a non-finite value.
    fn push(&mut self, x: f64) -> Option<usize> {
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
            return None;
        }
        Some((cur - self.idx[self.head]) as usize)
    }
}

/// Bars since the highest high and the lowest low of the last `period + 1` bars.
struct AroonCore {
    p: usize,
    hi: ExtIdx,
    lo: ExtIdx,
}

impl AroonCore {
    fn new(p: usize) -> Self {
        Self { p, hi: ExtIdx::new(p + 1, true), lo: ExtIdx::new(p + 1, false) }
    }

    /// (bars since the highest high, bars since the lowest low).
    fn push(&mut self, high: f64, low: f64) -> Option<(usize, usize)> {
        let h = self.hi.push(high);
        let l = self.lo.push(low);
        Some((h?, l?))
    }

    fn factor(&self) -> f64 {
        100.0 / self.p as f64
    }
}

/// Aroon down and up: 100 * (period - bars since the lowest low / highest high) / period.
pub struct Aroon(AroonCore);

impl Kernel<2, 2> for Aroon {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 2] {
        match self.0.push(x[0], x[1]) {
            Some((bh, bl)) => {
                let (f, p) = (self.0.factor(), self.0.p);
                [f * (p - bl) as f64, f * (p - bh) as f64]
            }
            None => [f64::NAN; 2],
        }
    }
}

/// Aroon up minus Aroon down.
pub struct AroonOsc(AroonCore);

impl Kernel<2, 1> for AroonOsc {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        match self.0.push(x[0], x[1]) {
            Some((bh, bl)) => [self.0.factor() * (bl as f64 - bh as f64)],
            None => [f64::NAN],
        }
    }
}

// ----------------------------------------------------------------------- directional movement

/// Raw directional movements and true range of an accepted bar against the previous accepted bar.
fn raw_moves(prev: [f64; 3], x: [f64; 3]) -> (f64, f64, f64) {
    let up = x[0] - prev[0];
    let down = prev[1] - x[1];
    let plus = if up > down && up > 0.0 { up } else { 0.0 };
    let minus = if down > up && down > 0.0 { down } else { 0.0 };
    (plus, minus, true_range(x[0], x[1], prev[2]))
}

/// Wilder-smoothed +DM, -DM and true range.
///
/// `talib`: the first `period - 1` movements are summed and every later bar (from bar `period`)
/// smooths, `s - s / p + x`; the first value is on bar `period`. Native
/// (`indicators::strength`): the first `period` movements are summed, so the first value, on bar
/// `period` too, is that plain sum.
struct DirCore {
    p: usize,
    talib: bool,
    prev: Option<[f64; 3]>,
    i: usize,
    s: [f64; 3],
}

impl DirCore {
    fn new(p: usize, talib: bool) -> Self {
        Self { p: p.max(1), talib, prev: None, i: 0, s: [sum_identity(); 3] }
    }

    fn push(&mut self, x: [f64; 3]) -> Option<[f64; 3]> {
        let prev = self.prev.replace(x)?;
        let (plus, minus, tr) = raw_moves(prev, x);
        self.i += 1;
        let p = self.p as f64;
        let accumulate = if self.talib { self.i < self.p } else { self.i <= self.p };
        for (s, v) in self.s.iter_mut().zip([plus, minus, tr]) {
            *s = if accumulate { *s + v } else { *s - *s / p + v };
        }
        (self.i >= self.p).then_some(self.s)
    }
}

/// +DM or -DM. TA-Lib's value: the sum of the first `period - 1` movements on bar `period - 1`,
/// then `s - s / p + x`; at period 1 the raw movement from bar 1.
pub struct Dm {
    plus: bool,
    p: usize,
    prev: Option<[f64; 2]>,
    i: usize,
    s: f64,
}

impl Kernel<2, 1> for Dm {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        if !all_finite(&x) {
            return [f64::NAN];
        }
        let Some(prev) = self.prev.replace(x) else {
            return [f64::NAN];
        };
        let up = x[0] - prev[0];
        let down = prev[1] - x[1];
        let dm = if self.plus {
            if up > down && up > 0.0 {
                up
            } else {
                0.0
            }
        } else if down > up && down > 0.0 {
            down
        } else {
            0.0
        };
        self.i += 1;
        if self.i < self.p {
            self.s += dm;
            return if self.i + 1 >= self.p { [self.s] } else { [f64::NAN] };
        }
        self.s = self.s - self.s / self.p as f64 + dm;
        [self.s]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirKind {
    PlusDi,
    MinusDi,
    Dx,
    Adx,
    Adxr,
}

/// The directional-movement system over (high, low, close): +DI, -DI, DX, ADX and ADXR share one
/// smoothing. A bar with a non-finite input is skipped.
///
/// Native DI is NaN while the smoothed true range is 0; TA-Lib's is 0 there. At `period` 1
/// TA-Lib's DI is the raw movement over the true range without the factor 100, and `talib` keeps
/// that.
pub struct Dir {
    kind: DirKind,
    talib: bool,
    core: DirCore,
    dx_seen: usize,
    dx_sum: f64,
    adx: Option<f64>,
    ring: Box<[f64]>,
    adx_seen: usize,
}

impl Dir {
    fn new(kind: DirKind, p: usize, talib: bool) -> Self {
        let ring = if kind == DirKind::Adxr { vec![f64::NAN; p.max(1)] } else { Vec::new() };
        Self {
            kind,
            talib,
            core: DirCore::new(p, talib),
            dx_seen: 0,
            dx_sum: sum_identity(),
            adx: None,
            ring: ring.into_boxed_slice(),
            adx_seen: 0,
        }
    }

    /// (+DI, -DI, DX) from the smoothed sums.
    fn indicators(&self, s: [f64; 3]) -> (f64, f64, f64) {
        let [sp, sm, st] = s;
        if self.talib {
            if st == 0.0 {
                return (0.0, 0.0, 0.0);
            }
            let scale = if self.core.p == 1 { 1.0 } else { 100.0 };
            let (pdi, mdi) = (scale * (sp / st), scale * (sm / st));
            let sum = pdi + mdi;
            let dx = if sum != 0.0 { 100.0 * ((pdi - mdi).abs() / sum) } else { 0.0 };
            (pdi, mdi, dx)
        } else if st > 0.0 {
            let pdi = 100.0 * sp / st;
            let mdi = 100.0 * sm / st;
            let sum = pdi + mdi;
            let dx = if sum > 0.0 { 100.0 * (pdi - mdi).abs() / sum } else { 0.0 };
            (pdi, mdi, dx)
        } else {
            (f64::NAN, f64::NAN, 0.0)
        }
    }
}

impl Kernel<3, 1> for Dir {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 1] {
        if !all_finite(&x) {
            return [f64::NAN];
        }
        let Some(s) = self.core.push(x) else {
            return [f64::NAN];
        };
        let (pdi, mdi, dx) = self.indicators(s);
        let p = self.core.p;
        let v = match self.kind {
            DirKind::PlusDi => pdi,
            DirKind::MinusDi => mdi,
            DirKind::Dx => dx,
            DirKind::Adx | DirKind::Adxr => {
                self.dx_seen += 1;
                let adx = if self.dx_seen < p {
                    self.dx_sum += dx;
                    return [f64::NAN];
                } else if self.dx_seen == p {
                    self.dx_sum += dx;
                    self.dx_sum / p as f64
                } else {
                    (self.adx.unwrap_or(f64::NAN) * (p - 1) as f64 + dx) / p as f64
                };
                self.adx = Some(adx);
                if self.kind == DirKind::Adx {
                    adx
                } else {
                    // ADX `p - 1` readings back lives in slot (m + 1) % p, which the new reading
                    // (slot m % p) does not touch for p >= 2.
                    let m = self.adx_seen;
                    self.adx_seen += 1;
                    let back = if m + 1 >= p { self.ring[(m + 1) % p] } else { f64::NAN };
                    self.ring[m % p] = adx;
                    (adx + back) / 2.0
                }
            }
        };
        [finite_or_nan(v)]
    }
}

// ----------------------------------------------------------------- walked windows: CCI and MFI

/// (high + low + close) / 3 of the bar `k` back, `k = 0` being the current one.
fn typical<const I: usize, H: Hist<I>>(x: &[f64; I], h: &H, k: usize) -> f64 {
    if k == 0 {
        (x[0] + x[1] + x[2]) / 3.0
    } else {
        (h.ago(0, k) + h.ago(1, k) + h.ago(2, k)) / 3.0
    }
}

/// Commodity channel index: (typical price - its mean) / (0.015 * mean absolute deviation) over
/// the last `period` bars, 0 when there is no deviation.
///
/// TA-Lib keeps the typical prices in a ring and sums it in slot order, not oldest first; the
/// rounding of a window of equal values depends on that order, so the walk below follows the
/// slots (bar `n` lives in slot `n % period`).
pub struct Cci {
    p: usize,
    n: u64,
}

impl Kernel<3, 1> for Cci {
    fn lookback(&self) -> usize {
        self.p - 1
    }

    fn step<H: Hist<3>>(&mut self, x: [f64; 3], h: &H) -> [f64; 1] {
        let cur = (self.n % self.p as u64) as usize;
        self.n += 1;
        // The typical price held by slot j: the bar (cur - j) mod p back.
        let slot = |j: usize| typical(&x, h, (cur + self.p - j) % self.p);
        let mut sum = 0.0;
        for j in 0..self.p {
            let t = slot(j);
            if !t.is_finite() {
                return [f64::NAN];
            }
            sum += t;
        }
        let mean = sum / self.p as f64;
        let mut dev = 0.0;
        for j in 0..self.p {
            dev += (slot(j) - mean).abs();
        }
        let num = typical(&x, h, 0) - mean;
        [finite_or_nan(if num != 0.0 && dev != 0.0 {
            num / (0.015 * (dev / self.p as f64))
        } else {
            0.0
        })]
    }
}

/// Money flow index, native: 100 when there is no negative flow (flat included), else
/// `100 - 100 / (1 + pos / neg)`. The flows of the last `period` bars are summed afresh in order
/// on every bar, as `indicators::volume::mfi` does.
pub struct Mfi {
    p: usize,
}

impl Kernel<4, 1> for Mfi {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<4>>(&mut self, x: [f64; 4], h: &H) -> [f64; 1] {
        let at = |i: usize, k: usize| if k == 0 { x[i] } else { h.ago(i, k) };
        let mut prev = typical(&x, h, self.p);
        if !prev.is_finite() {
            return [f64::NAN];
        }
        let (mut pos, mut neg) = (0.0, 0.0);
        for k in (0..self.p).rev() {
            let t = typical(&x, h, k);
            let v = at(3, k);
            if !(t.is_finite() && v.is_finite()) {
                return [f64::NAN];
            }
            let flow = t * v;
            if t > prev {
                pos += flow;
            } else if t < prev {
                neg += flow;
            }
            prev = t;
        }
        let out = if neg == 0.0 { 100.0 } else { 100.0 - (100.0 / (1.0 + pos / neg)) };
        [finite_or_nan(out)]
    }
}

/// Money flow index, TA-Lib's: `100 * pos / (pos + neg)`, and 0 when the two sum to less than 1.
/// The two flow sums roll (subtract the flow leaving, then add the new one) in TA-Lib's order, so
/// the rounding residue that a window with no positive (or no negative) flow leaves behind is
/// TA-Lib's too, not an exact 0. A non-finite bar spoils the windows it is in.
pub struct MfiRoll {
    p: usize,
    prev_tp: f64,
    pos: Box<[f64]>,
    neg: Box<[f64]>,
    n: u64,
    sums: (f64, f64),
    bad: usize,
}

impl MfiRoll {
    fn new(p: usize) -> Self {
        Self {
            p,
            prev_tp: f64::NAN,
            pos: vec![f64::NAN; p].into_boxed_slice(),
            neg: vec![f64::NAN; p].into_boxed_slice(),
            n: 0,
            sums: (0.0, 0.0),
            bad: 0,
        }
    }
}

impl Kernel<4, 1> for MfiRoll {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<4>>(&mut self, x: [f64; 4], _h: &H) -> [f64; 1] {
        let slot = (self.n % self.p as u64) as usize;
        if self.n >= self.p as u64 {
            if self.pos[slot].is_nan() {
                self.bad -= 1;
            } else {
                self.sums.0 -= self.pos[slot];
                self.sums.1 -= self.neg[slot];
            }
        }
        let tp = (x[0] + x[1] + x[2]) / 3.0;
        let (pos, neg) = if tp.is_finite() && x[3].is_finite() && self.prev_tp.is_finite() {
            let flow = tp * x[3];
            if tp > self.prev_tp {
                (flow, 0.0)
            } else if tp < self.prev_tp {
                (0.0, flow)
            } else {
                (0.0, 0.0)
            }
        } else {
            (f64::NAN, f64::NAN)
        };
        self.prev_tp = tp;
        self.pos[slot] = pos;
        self.neg[slot] = neg;
        if pos.is_nan() || neg.is_nan() {
            self.bad += 1;
        } else {
            self.sums.0 += pos;
            self.sums.1 += neg;
        }
        self.n += 1;
        if self.n <= self.p as u64 || self.bad > 0 {
            return [f64::NAN];
        }
        let total = self.sums.0 + self.sums.1;
        [finite_or_nan(if total < 1.0 { 0.0 } else { 100.0 * (self.sums.0 / total) })]
    }
}

// --------------------------------------------------------------------------------- ULTOSC

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

#[derive(Debug, Clone, Copy, Default)]
struct WinAcc {
    acc: Neumaier,
    bad: usize,
    /// Non-zero finite values in the window: an all-zero window sums to exactly 0, not to a
    /// compensation residue.
    nonzero: usize,
}

impl WinAcc {
    fn add(&mut self, v: f64) {
        if v.is_finite() {
            self.acc.add(v);
            self.nonzero += usize::from(v != 0.0);
        } else {
            self.bad += 1;
        }
    }

    fn remove(&mut self, v: f64) {
        if v.is_finite() {
            self.acc.add(-v);
            self.nonzero -= usize::from(v != 0.0);
        } else {
            self.bad -= 1;
        }
    }
}

/// Rolling sums of one derived series over three windows, on a ring it owns (the series is
/// computed inside the kernel, so it is not in `Hist`). Neumaier add and remove, with an exact
/// re-sum from the ring every `max(p, 1024)` pushes.
struct Sums3 {
    ps: [usize; 3],
    ring: Box<[f64]>,
    n: u64,
    w: [WinAcc; 3],
}

impl Sums3 {
    fn new(ps: [usize; 3]) -> Self {
        let cap = ps[0].max(ps[1]).max(ps[2]);
        Self { ps, ring: vec![f64::NAN; cap].into_boxed_slice(), n: 0, w: [WinAcc::default(); 3] }
    }

    fn push(&mut self, v: f64) {
        let cap = self.ring.len() as u64;
        for (w, &p) in self.w.iter_mut().zip(&self.ps) {
            w.add(v);
            if self.n >= p as u64 {
                w.remove(self.ring[((self.n - p as u64) % cap) as usize]);
            }
        }
        self.ring[(self.n % cap) as usize] = v;
        self.n += 1;
        for (w, &p) in self.w.iter_mut().zip(&self.ps) {
            if self.n >= p as u64 && self.n.is_multiple_of(p.max(1024) as u64) {
                *w = WinAcc::default();
                for j in (self.n - p as u64)..self.n {
                    w.add(self.ring[(j % cap) as usize]);
                }
            }
        }
    }

    fn sum(&self, i: usize) -> f64 {
        let w = &self.w[i];
        if self.n < self.ps[i] as u64 || w.bad > 0 {
            f64::NAN
        } else if w.nonzero == 0 {
            0.0
        } else {
            w.acc.value()
        }
    }
}

/// Ultimate oscillator: 100 * (4 * A1 + 2 * A2 + A3) / 7 with `Ai` the sum of buying pressure
/// (close - min(low, previous close)) over the sum of true range, for the three periods in
/// ascending order. A window with no range contributes 0.
pub struct Ultosc {
    bp: Sums3,
    tr: Sums3,
}

impl Kernel<3, 1> for Ultosc {
    fn lookback(&self) -> usize {
        1
    }

    fn step<H: Hist<3>>(&mut self, x: [f64; 3], h: &H) -> [f64; 1] {
        let pc = h.ago(2, 1);
        let (bp, tr) = if all_finite(&x) && pc.is_finite() {
            let lo = x[1].min(pc);
            (x[2] - lo, x[0].max(pc) - lo)
        } else {
            (f64::NAN, f64::NAN)
        };
        self.bp.push(bp);
        self.tr.push(tr);
        let mut out = 0.0;
        for (i, weight) in [4.0, 2.0, 1.0].into_iter().enumerate() {
            let (a, b) = (self.bp.sum(i), self.tr.sum(i));
            if a.is_nan() || b.is_nan() {
                return [f64::NAN];
            }
            if b != 0.0 {
                out += weight * (a / b);
            }
        }
        [finite_or_nan(100.0 * (out / 7.0))]
    }
}

// ------------------------------------------------------------------------------------ params

const X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const HL: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
];
const HLC: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
];
const OHLC: &[InputDef] = &[
    InputDef { name: "open", default_field: Some("open") },
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
];
const HLCV: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
    InputDef { name: "volume", default_field: Some("volume") },
];

const fn int(name: &'static str, min: i64, max: i64, default: Option<i64>) -> ParamDef {
    ParamDef {
        name,
        kind: ParamKind::Int { min, max },
        default: match default {
            Some(v) => Some(ParamVal::Int(v)),
            None => None,
        },
    }
}

/// TA-Lib's `timeperiod`, registered as `period` like `sma` and `atr`; no default.
const fn period(min: i64) -> ParamDef {
    int("period", min, MAX_PERIOD, None)
}

const fn matype(name: &'static str) -> ParamDef {
    int(name, 0, 8, Some(0))
}

const COMPAT: ParamDef = ParamDef {
    name: "compat",
    kind: ParamKind::Enum(&["native", "talib"]),
    default: Some(ParamVal::Enum("native")),
};

const NO_PARAMS: &[ParamDef] = &[];
const PERIOD1: &[ParamDef] = &[period(1)];
const PERIOD2: &[ParamDef] = &[period(2)];
const PERIOD1_COMPAT: &[ParamDef] = &[period(1), COMPAT];
const PERIOD2_COMPAT: &[ParamDef] = &[period(2), COMPAT];
const WALKED2: &[ParamDef] = &[int("period", 2, MAX_WALKED, None)];
const WALKED2_COMPAT: &[ParamDef] = &[int("period", 2, MAX_WALKED, None), COMPAT];
const MACD_PARAMS: &[ParamDef] = &[
    int("fastperiod", 2, MAX_PERIOD, Some(12)),
    int("slowperiod", 2, MAX_PERIOD, Some(26)),
    int("signalperiod", 1, MAX_PERIOD, Some(9)),
    COMPAT,
];
const MACDEXT_PARAMS: &[ParamDef] = &[
    int("fastperiod", 2, MAX_PERIOD, Some(12)),
    matype("fastmatype"),
    int("slowperiod", 2, MAX_PERIOD, Some(26)),
    matype("slowmatype"),
    int("signalperiod", 1, MAX_PERIOD, Some(9)),
    matype("signalmatype"),
];
const MACDFIX_PARAMS: &[ParamDef] = &[int("signalperiod", 1, MAX_PERIOD, Some(9))];
const PO_PARAMS: &[ParamDef] = &[
    int("fastperiod", 2, MAX_PERIOD, Some(12)),
    int("slowperiod", 2, MAX_PERIOD, Some(26)),
    matype("matype"),
];
const STOCH_PARAMS: &[ParamDef] = &[
    int("fastk_period", 1, MAX_PERIOD, Some(5)),
    int("slowk_period", 1, MAX_PERIOD, Some(3)),
    matype("slowk_matype"),
    int("slowd_period", 1, MAX_PERIOD, Some(3)),
    matype("slowd_matype"),
];
const STOCHF_PARAMS: &[ParamDef] = &[
    int("fastk_period", 1, MAX_PERIOD, Some(5)),
    int("fastd_period", 1, MAX_PERIOD, Some(3)),
    matype("fastd_matype"),
];
const STOCHRSI_PARAMS: &[ParamDef] = &[
    period(2),
    int("fastk_period", 1, MAX_PERIOD, Some(5)),
    int("fastd_period", 1, MAX_PERIOD, Some(3)),
    matype("fastd_matype"),
];
const STOCHASTIC_PARAMS: &[ParamDef] =
    &[int("k_period", 1, MAX_PERIOD, Some(14)), int("d_period", 1, MAX_PERIOD, Some(3))];
const ULTOSC_PARAMS: &[ParamDef] = &[
    int("timeperiod1", 1, MAX_PERIOD, Some(7)),
    int("timeperiod2", 1, MAX_PERIOD, Some(14)),
    int("timeperiod3", 1, MAX_PERIOD, Some(28)),
];

/// A period parameter within the range its definition declares.
fn per(p: &[ParamVal], i: usize, min: usize, max: i64) -> Result<usize, String> {
    let v = int_param(p, i)?;
    if v < min || v as i64 > max {
        return Err(format!("parameter {i} must be in {min}..={max}, got {v}"));
    }
    Ok(v)
}

fn ma_kind(p: &[ParamVal], i: usize) -> Result<MaKind, String> {
    MaKind::from_code(int_param(p, i)? as i64)
}

fn is_talib(p: &[ParamVal], i: usize) -> bool {
    matches!(p.get(i), Some(ParamVal::Enum("talib")))
}

fn build<K, const I: usize, const O: usize>(
    k: K,
    ins: &[usize],
    outs: &[usize],
    what: &str,
) -> Result<Box<dyn Node>, String>
where
    K: Kernel<I, O> + 'static,
{
    Ok(Box::new(Adapter::new(
        k,
        regs::<I>(ins, &format!("{what} inputs"))?,
        regs::<O>(outs, &format!("{what} outputs"))?,
    )))
}

fn make_change(
    kind: ChangeKind,
    name: &str,
    p: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = Change { kind, p: per(p, 0, 1, MAX_PERIOD)?, last: f64::NAN };
    build(k, ins, outs, name)
}

fn make_mom(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_change(ChangeKind::Mom, "mom", p, i, o)
}
fn make_roc(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_change(ChangeKind::Roc, "roc", p, i, o)
}
fn make_rocp(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_change(ChangeKind::Rocp, "rocp", p, i, o)
}
fn make_rocr(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_change(ChangeKind::Rocr, "rocr", p, i, o)
}
fn make_rocr100(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_change(ChangeKind::Rocr100, "rocr100", p, i, o)
}

fn make_bop(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    build(Bop, ins, outs, "bop")
}

fn make_rsi(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let core = GainLoss::new(per(p, 0, 1, MAX_PERIOD)?, is_talib(p, 1));
    build(Rsi { core }, ins, outs, "rsi")
}

fn make_cmo(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let core = GainLoss::new(per(p, 0, 2, MAX_PERIOD)?, true);
    build(Cmo { core }, ins, outs, "cmo")
}

fn make_macd(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let (f, s, g) =
        (per(p, 0, 1, MAX_PERIOD)?, per(p, 1, 1, MAX_PERIOD)?, per(p, 2, 1, MAX_PERIOD)?);
    let k = if is_talib(p, 3) { Macd::talib(f, s, g) } else { Macd::native(f, s, g) };
    build(k, ins, outs, "macd")
}

fn make_macdext(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Macd::ext(
        (ma_kind(p, 1)?, per(p, 0, 2, MAX_PERIOD)?),
        (ma_kind(p, 3)?, per(p, 2, 2, MAX_PERIOD)?),
        (ma_kind(p, 5)?, per(p, 4, 1, MAX_PERIOD)?),
    )?;
    build(k, ins, outs, "macdext")
}

fn make_macdfix(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    build(Macd::fixed(per(p, 0, 1, MAX_PERIOD)?), ins, outs, "macdfix")
}

fn make_apo(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Po::new(false, ma_kind(p, 2)?, per(p, 0, 2, MAX_PERIOD)?, per(p, 1, 2, MAX_PERIOD)?)?;
    build(k, ins, outs, "apo")
}

fn make_ppo(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Po::new(true, ma_kind(p, 2)?, per(p, 0, 2, MAX_PERIOD)?, per(p, 1, 2, MAX_PERIOD)?)?;
    build(k, ins, outs, "ppo")
}

fn make_trix(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let n = per(p, 0, 2, MAX_PERIOD)?;
    let a = 2.0 / (n as f64 + 1.0);
    let e = || Fed::ema(n, a, true, 0);
    let k = Trix { e: [e(), e(), e()], prev: None };
    build(k, ins, outs, "trix")
}

fn make_stochf(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = StochF {
        k: StochK::new(per(p, 0, 1, MAX_PERIOD)?),
        d: Fed::core(ma_kind(p, 2)?, per(p, 1, 1, MAX_PERIOD)?, 0)?,
    };
    build(k, ins, outs, "stochf")
}

fn make_stoch(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Stoch {
        k: StochK::new(per(p, 0, 1, MAX_PERIOD)?),
        sk: Fed::core(ma_kind(p, 2)?, per(p, 1, 1, MAX_PERIOD)?, 0)?,
        sd: Fed::core(ma_kind(p, 4)?, per(p, 3, 1, MAX_PERIOD)?, 0)?,
    };
    build(k, ins, outs, "stoch")
}

fn make_stochrsi(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = StochRsi {
        rsi: GainLoss::new(per(p, 0, 2, MAX_PERIOD)?, true),
        k: StochK::new(per(p, 1, 1, MAX_PERIOD)?),
        d: Fed::core(ma_kind(p, 3)?, per(p, 2, 1, MAX_PERIOD)?, 0)?,
    };
    build(k, ins, outs, "stochrsi")
}

fn make_stochastic(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let kp = per(p, 0, 1, MAX_PERIOD)?;
    let dp = per(p, 1, 1, MAX_PERIOD)?;
    let k = Stochastic {
        hh: RollExt::new(kp, true),
        ll: RollExt::new(kp, false),
        ring: vec![f64::NAN; dp].into_boxed_slice(),
        n: 0,
    };
    build(k, ins, outs, "stochastic")
}

fn make_willr(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let n = per(p, 0, 2, MAX_PERIOD)?;
    build(Willr { hh: RollExt::new(n, true), ll: RollExt::new(n, false) }, ins, outs, "willr")
}

fn make_aroon(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    build(Aroon(AroonCore::new(per(p, 0, 2, MAX_PERIOD)?)), ins, outs, "aroon")
}

fn make_aroonosc(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    build(AroonOsc(AroonCore::new(per(p, 0, 2, MAX_PERIOD)?)), ins, outs, "aroonosc")
}

fn make_dm(
    plus: bool,
    p: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = Dm { plus, p: per(p, 0, 1, MAX_PERIOD)?, prev: None, i: 0, s: 0.0 };
    build(k, ins, outs, if plus { "plus_dm" } else { "minus_dm" })
}
fn make_plus_dm(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_dm(true, p, i, o)
}
fn make_minus_dm(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_dm(false, p, i, o)
}

fn make_dir(
    kind: DirKind,
    name: &str,
    min: usize,
    p: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    build(Dir::new(kind, per(p, 0, min, MAX_PERIOD)?, is_talib(p, 1)), ins, outs, name)
}
fn make_plus_di(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_dir(DirKind::PlusDi, "plus_di", 1, p, i, o)
}
fn make_minus_di(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_dir(DirKind::MinusDi, "minus_di", 1, p, i, o)
}
fn make_adx(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    make_dir(DirKind::Adx, "adx", 2, p, i, o)
}
/// `dx` and `adxr` are new names: TA-Lib's convention, no compat.
fn make_dx(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    build(Dir::new(DirKind::Dx, per(p, 0, 2, MAX_PERIOD)?, true), i, o, "dx")
}
fn make_adxr(p: &[ParamVal], i: &[usize], o: &[usize]) -> Result<Box<dyn Node>, String> {
    build(Dir::new(DirKind::Adxr, per(p, 0, 2, MAX_PERIOD)?, true), i, o, "adxr")
}

fn make_cci(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    build(Cci { p: per(p, 0, 2, MAX_WALKED)?, n: 0 }, ins, outs, "cci")
}

fn make_mfi(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let n = per(p, 0, 2, MAX_WALKED)?;
    if is_talib(p, 1) {
        build(MfiRoll::new(n), ins, outs, "mfi")
    } else {
        build(Mfi { p: n }, ins, outs, "mfi")
    }
}

fn make_ultosc(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let mut ps = [per(p, 0, 1, MAX_PERIOD)?, per(p, 1, 1, MAX_PERIOD)?, per(p, 2, 1, MAX_PERIOD)?];
    // TA-Lib sorts the three periods; the shortest gets the weight 4.
    ps.sort_unstable();
    build(Ultosc { bp: Sums3::new(ps), tr: Sums3::new(ps) }, ins, outs, "ultosc")
}

// ----------------------------------------------------------------------------- warm-up, memory

fn p0(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(1)
}

fn warmup_p(p: &[ParamVal]) -> usize {
    p0(p)
}
fn warmup_p_minus_1(p: &[ParamVal]) -> usize {
    p0(p).saturating_sub(1)
}
fn warmup_zero(_: &[ParamVal]) -> usize {
    0
}
fn warmup_adx(p: &[ParamVal]) -> usize {
    (2 * p0(p)).saturating_sub(1)
}
fn warmup_adxr(p: &[ParamVal]) -> usize {
    (3 * p0(p)).saturating_sub(2)
}
fn warmup_dm(p: &[ParamVal]) -> usize {
    p0(p).saturating_sub(1).max(1)
}
fn warmup_trix(p: &[ParamVal]) -> usize {
    3 * p0(p).saturating_sub(1) + 1
}
fn warmup_ultosc(p: &[ParamVal]) -> usize {
    (0..3).map(|i| int_param(p, i).unwrap_or(1)).max().unwrap_or(1)
}
fn warmup_macd(p: &[ParamVal]) -> usize {
    let (f, s, g) = (p0(p), int_param(p, 1).unwrap_or(1), int_param(p, 2).unwrap_or(1));
    f.max(s).saturating_sub(1) + g.saturating_sub(1)
}
fn warmup_macdfix(p: &[ParamVal]) -> usize {
    25 + p0(p).saturating_sub(1)
}
/// Lookback of the average of kind code at `ki` and period at `pi`.
fn ma_lookback(p: &[ParamVal], pi: usize, ki: usize) -> usize {
    match (int_param(p, pi), ma_kind(p, ki)) {
        (Ok(period), Ok(k)) => k.lookback(period),
        _ => 0,
    }
}
fn warmup_macdext(p: &[ParamVal]) -> usize {
    ma_lookback(p, 0, 1).max(ma_lookback(p, 2, 3)) + ma_lookback(p, 4, 5)
}
fn warmup_po(p: &[ParamVal]) -> usize {
    ma_lookback(p, 0, 2).max(ma_lookback(p, 1, 2))
}
fn warmup_stochf(p: &[ParamVal]) -> usize {
    p0(p).saturating_sub(1) + ma_lookback(p, 1, 2)
}
fn warmup_stoch(p: &[ParamVal]) -> usize {
    p0(p).saturating_sub(1) + ma_lookback(p, 1, 2) + ma_lookback(p, 3, 4)
}
fn warmup_stochrsi(p: &[ParamVal]) -> usize {
    p0(p) + int_param(p, 1).unwrap_or(1).saturating_sub(1) + ma_lookback(p, 2, 3)
}
fn warmup_stochastic(p: &[ParamVal]) -> usize {
    p0(p).saturating_sub(1) + int_param(p, 1).unwrap_or(1).saturating_sub(1)
}

/// Bars `k` until `(k + 1)^(stages - 1) * rho^k <= eps`: the bound on a seed's influence after
/// `stages` cascaded Wilder smoothings of `period`.
fn wilder_settle_bars(period: usize, stages: u32, eps: f64) -> usize {
    if period <= 1 {
        return 0;
    }
    let (lr, le) = ((1.0 - 1.0 / period as f64).ln(), eps.max(1e-300).ln());
    if stages <= 1 {
        return (le / lr).ceil().max(0.0) as usize;
    }
    let mut k = 0usize;
    while k < (1 << 26) && f64::from(stages - 1) * ((k + 1) as f64).ln() + k as f64 * lr > le {
        k += 1;
    }
    k
}

fn settle_wilder1(p: &[ParamVal], eps: f64) -> usize {
    wilder_settle_bars(p0(p), 1, eps)
}
/// Smoothed movements, then the average of DX: two cascaded Wilder stages (ADXR reads the same
/// ADX twice).
fn settle_wilder2(p: &[ParamVal], eps: f64) -> usize {
    wilder_settle_bars(p0(p), 2, eps)
}
fn settle_macd(p: &[ParamVal], eps: f64) -> usize {
    let (f, s, g) = (p0(p), int_param(p, 1).unwrap_or(1), int_param(p, 2).unwrap_or(1));
    MaKind::Ema.settle(f.max(s), eps) + MaKind::Ema.settle(g, eps)
}
fn settle_macdfix(p: &[ParamVal], eps: f64) -> usize {
    // The slow leg's constant 0.075 decays as 0.925^k, slower than the fast leg's 0.85^k.
    let slow = (eps.max(1e-300).ln() / 0.925f64.ln()).ceil().max(0.0) as usize;
    slow + MaKind::Ema.settle(p0(p), eps)
}
fn ma_settle(p: &[ParamVal], pi: usize, ki: usize, eps: f64) -> usize {
    match (int_param(p, pi), ma_kind(p, ki)) {
        (Ok(period), Ok(k)) => k.settle(period, eps),
        _ => 0,
    }
}
fn settle_macdext(p: &[ParamVal], eps: f64) -> usize {
    ma_settle(p, 0, 1, eps).max(ma_settle(p, 2, 3, eps)) + ma_settle(p, 4, 5, eps)
}
fn settle_po(p: &[ParamVal], eps: f64) -> usize {
    ma_settle(p, 0, 2, eps).max(ma_settle(p, 1, 2, eps))
}
fn settle_trix(p: &[ParamVal], eps: f64) -> usize {
    MaKind::Tema.settle(p0(p), eps)
}
fn settle_stochf(p: &[ParamVal], eps: f64) -> usize {
    ma_settle(p, 1, 2, eps)
}
fn settle_stoch(p: &[ParamVal], eps: f64) -> usize {
    ma_settle(p, 1, 2, eps) + ma_settle(p, 3, 4, eps)
}
fn settle_stochrsi(p: &[ParamVal], eps: f64) -> usize {
    wilder_settle_bars(p0(p), 1, eps) + ma_settle(p, 2, 3, eps)
}

// ------------------------------------------------------------------------------------ table

pub const DEFS: &[FnDef] = &[
    FnDef {
        name: "mom",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD1,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("MOM"),
        doc: "Momentum: x - x[period bars ago]. NaN if either value is NaN.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_mom,
    },
    FnDef {
        name: "roc",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD1,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::streaming::StreamingRoc",
            tolerance: 0.0,
        }),
        talib_name: Some("ROC"),
        doc: "Rate of change in percent, (x - base) / base * 100 with base the value `period` bars back. Same arithmetic as the streaming core, which agrees with TA-Lib's ((x / base) - 1) * 100 to rounding. On a base of exactly 0 it keeps the last value (the streaming core's behaviour); TA-Lib, rocp, rocr and rocr100 give 0 there.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Measured (legacy_vs_talib.txt): same first valid bar, difference at rounding level.
        relation: Relation::Exact,
        make: make_roc,
    },
    FnDef {
        name: "rocp",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD1,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("ROCP"),
        doc: "Rate of change as a fraction, (x - base) / base. 0 when base is 0 (TA-Lib's value).",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_rocp,
    },
    FnDef {
        name: "rocr",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD1,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("ROCR"),
        doc: "Rate of change ratio, x / base. 0 when base is 0 (TA-Lib's value).",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_rocr,
    },
    FnDef {
        name: "rocr100",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD1,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("ROCR100"),
        doc: "Rate of change ratio on a 100 scale, x / base * 100. 0 when base is 0 (TA-Lib's value).",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_rocr100,
    },
    FnDef {
        name: "bop",
        family: Family::Momentum,
        inputs: OHLC,
        params: NO_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_zero,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("BOP"),
        doc: "Balance of power, (close - open) / (high - low). A bar with no range (flat, or high not above low) gives 0, TA-Lib's value; any positive range is divided (TA-Lib 0.6.4 does so down to 1e-9, measured).",
        refs: &["Livshin, Balance of Power, Stocks & Commodities, 2001"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Elementwise,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_bop,
    },
    FnDef {
        name: "rsi",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD1_COMPAT,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Infinite { settle: settle_wilder1 },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::streaming::StreamingRsi",
            tolerance: 0.0,
        }),
        talib_name: Some("RSI"),
        doc: "Relative strength index, Wilder smoothing, seeded with the mean of the first `period` gains and losses; first value on bar `period`. The two compat modes agree to rounding (about 1e-14) wherever the average loss is above 0; they differ only where it is exactly 0. compat=\"native\" (default) gives 100 whenever the average loss is 0, so on a flat series (a name that has not traded) the RSI is 100 and a rule like `rsi > 70` is TRUE. compat=\"talib\" gives 100 * gain / (gain + loss), and 0 when both are 0, so a flat series reads 0 and `rsi < 30` is TRUE instead. This is a decided product behaviour. A series that only rises is 100 in both; one that only falls is 0 in both.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Native replicates the streaming core (flat: 100, TA-Lib 0). The talib path replays Exact.
        relation: Relation::DefinedDifference,
        make: make_rsi,
    },
    FnDef {
        name: "cmo",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Infinite { settle: settle_wilder1 },
        legacy: None,
        talib_name: Some("CMO"),
        doc: "Chande momentum oscillator: 100 * (gain - loss) / (gain + loss) on the Wilder-smoothed average gain and loss (TA-Lib's smoothing, the same as rsi). A flat series gives 0.",
        refs: &["Chande, Kroll, The New Technical Trader, 1994"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_cmo,
    },
    FnDef {
        name: "macd",
        family: Family::Momentum,
        inputs: X,
        params: MACD_PARAMS,
        outputs: &["macd", "macdsignal", "macdhist"],
        out_kind: OutKind::Series,
        warmup: warmup_macd,
        memory: Memory::Infinite { settle: settle_macd },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::streaming::StreamingMacd",
            tolerance: 0.0,
        }),
        talib_name: Some("MACD"),
        doc: "MACD line (fast EMA - slow EMA), signal (EMA of the line) and histogram. All three are NaN until the signal is ready, bar slow + signal - 2 (33 at 12, 26, 9). compat=\"native\" (default) seeds each EMA on its own first `period` values, like the streaming core. compat=\"talib\" starts the fast EMA on the first bar of the slow one (its seed is the mean of the `fast` values ending there) and, like TA-Lib, takes the shorter of fast and slow as the fast one. The two differ by the fast seed, which decays as (1 - 2 / (fast + 1)) per bar. At signalperiod 1 TA-Lib prints a spurious 0 on the bar before the first real value; this is NaN until the signal is ready.",
        refs: &["Appel, Technical Analysis: Power Tools for Active Investors, 1999"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Native against TA-Lib 0.6.4 (legacy_vs_talib.txt): up to 0.14 by seeding. The talib
        // path replays Exact.
        relation: Relation::DefinedDifference,
        make: make_macd,
    },
    FnDef {
        name: "macdext",
        family: Family::Momentum,
        inputs: X,
        params: MACDEXT_PARAMS,
        outputs: &["macd", "macdsignal", "macdhist"],
        out_kind: OutKind::Series,
        warmup: warmup_macdext,
        memory: Memory::Infinite { settle: settle_macdext },
        legacy: None,
        talib_name: Some("MACDEXT"),
        doc: "MACD with a moving-average type (`matype`: 0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA, 6 KAMA, 8 T3; 7, MAMA, is refused until the Hilbert slice lands) for each of the fast, slow and signal averages. The slower period is the slow one. Each average starts so that the fast and slow lines first land on the later lookback, as TA-Lib does. Kinds other than 0, 2 and 5 hold their state on a NaN.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_macdext,
    },
    FnDef {
        name: "macdfix",
        family: Family::Momentum,
        inputs: X,
        params: MACDFIX_PARAMS,
        outputs: &["macd", "macdsignal", "macdhist"],
        out_kind: OutKind::Series,
        warmup: warmup_macdfix,
        memory: Memory::Infinite { settle: settle_macdfix },
        legacy: None,
        talib_name: Some("MACDFIX"),
        doc: "MACD fixed at 12 and 26 with TA-Lib's smoothing constants 0.15 and 0.075 (not 2/13 and 2/27), signal EMA of `signalperiod`. The fast EMA starts on the first bar of the slow one.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_macdfix,
    },
    FnDef {
        name: "apo",
        family: Family::Momentum,
        inputs: X,
        params: PO_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_po,
        memory: Memory::Infinite { settle: settle_po },
        legacy: None,
        talib_name: Some("APO"),
        doc: "Absolute price oscillator: fast average - slow average, both of type `matype` (0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA, 6 KAMA, 8 T3; 7 is refused). The slower period is the slow one. Kinds other than 0, 2 and 5 hold their state on a NaN.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_apo,
    },
    FnDef {
        name: "ppo",
        family: Family::Momentum,
        inputs: X,
        params: PO_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_po,
        memory: Memory::Infinite { settle: settle_po },
        legacy: None,
        talib_name: Some("PPO"),
        doc: "Percentage price oscillator: (fast average - slow average) / slow average * 100, `matype` as in apo. 0 when the slow average is 0 (TA-Lib's value).",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_ppo,
    },
    FnDef {
        name: "trix",
        family: Family::Momentum,
        inputs: X,
        params: PERIOD2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_trix,
        memory: Memory::Infinite { settle: settle_trix },
        legacy: None,
        talib_name: Some("TRIX"),
        doc: "TRIX: the one-bar rate of change in percent of an EMA of an EMA of an EMA (each seeded with the mean of the previous stage's first `period` values). 0 when the previous triple EMA is 0. Period at least 2: TA-Lib's period 1 returns values unrelated to the definition.",
        refs: &["Hutson, Good Trix, Technical Analysis of Stocks & Commodities, 1983"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_trix,
    },
    FnDef {
        name: "stochastic",
        family: Family::Momentum,
        inputs: HLC,
        params: STOCHASTIC_PARAMS,
        outputs: &["k", "d"],
        out_kind: OutKind::Series,
        warmup: warmup_stochastic,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::momentum::stochastic",
            tolerance: 0.0,
        }),
        talib_name: None,
        doc: "raptorbt's FAST stochastic: k = 100 * (close - lowest low) / (highest high - lowest low) over `k_period` bars, d = the simple mean of the last `d_period` values of k. It is not TA-Lib's STOCH, which is the SLOW stochastic (its %K is already smoothed): for TA-Lib's use `stoch` (slow) or `stochf` (fast, whose fastd at matype 0 is this d). A range of 0 gives k = 50 (stoch and stochf give 0). Both outputs are NaN until d is ready (bar k_period + d_period - 2); the array function reports k from bar k_period - 1.",
        refs: &["Lane, Lane's Stochastics, 1984"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Hand("fns::momentum::tests::stochastic_hand_values"),
        relation: Relation::NoLibrary,
        make: make_stochastic,
    },
    FnDef {
        name: "stochf",
        family: Family::Momentum,
        inputs: HLC,
        params: STOCHF_PARAMS,
        outputs: &["fastk", "fastd"],
        out_kind: OutKind::Series,
        warmup: warmup_stochf,
        memory: Memory::Infinite { settle: settle_stochf },
        legacy: None,
        talib_name: Some("STOCHF"),
        doc: "TA-Lib's fast stochastic: fastk = 100 * (close - lowest low) / (highest high - lowest low) over `fastk_period` bars, fastd = its `fastd_matype` average over `fastd_period`. A range of 0 gives 0 (TA-Lib's value; the older `stochastic` says 50). Both outputs are NaN until fastd is ready, as in TA-Lib. `stochastic` is raptorbt's own fast stochastic, with a simple-mean d.",
        refs: &["Lane, Lane's Stochastics, 1984"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_stochf,
    },
    FnDef {
        name: "stoch",
        family: Family::Momentum,
        inputs: HLC,
        params: STOCH_PARAMS,
        outputs: &["slowk", "slowd"],
        out_kind: OutKind::Series,
        warmup: warmup_stoch,
        memory: Memory::Infinite { settle: settle_stoch },
        legacy: None,
        talib_name: Some("STOCH"),
        doc: "TA-Lib's SLOW stochastic: the fast %K over `fastk_period` bars, smoothed by `slowk_matype` over `slowk_period` is slowk, smoothed again by `slowd_matype` over `slowd_period` is slowd. A range of 0 gives 0 (TA-Lib's value). Both outputs are NaN until slowd is ready. Note `stochastic` is a different thing: raptorbt's fast stochastic (use `stochf` for TA-Lib's fast one).",
        refs: &["Lane, Lane's Stochastics, 1984"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_stoch,
    },
    FnDef {
        name: "stochrsi",
        family: Family::Momentum,
        inputs: X,
        params: STOCHRSI_PARAMS,
        outputs: &["fastk", "fastd"],
        out_kind: OutKind::Series,
        warmup: warmup_stochrsi,
        memory: Memory::Infinite { settle: settle_stochrsi },
        legacy: None,
        talib_name: Some("STOCHRSI"),
        doc: "Stochastic RSI: the fast stochastic over `fastk_period` bars of the RSI of `period` (TA-Lib's RSI, so a flat series reads RSI 0), and its `fastd_matype` average. A flat RSI window gives 0. Both outputs are NaN until fastd is ready.",
        refs: &["Chande, Kroll, The New Technical Trader, 1994"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_stochrsi,
    },
    FnDef {
        name: "willr",
        family: Family::Momentum,
        inputs: HLC,
        params: PERIOD2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p_minus_1,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("WILLR"),
        doc: "Williams %R: -100 * (highest high - close) / (highest high - lowest low) over `period` bars, in [-100, 0]. A range of 0 gives 0 (TA-Lib's value).",
        refs: &["Williams, How I Made One Million Dollars Last Year Trading Commodities, 1973"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_willr,
    },
    FnDef {
        name: "aroon",
        family: Family::Momentum,
        inputs: HL,
        params: PERIOD2,
        outputs: &["aroondown", "aroonup"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("AROON"),
        doc: "Aroon over the last `period + 1` bars: up = 100 * (period - bars since the highest high) / period, down likewise for the lowest low. A tie keeps the latest bar, so a flat series gives up = down = 100.",
        refs: &["Chande, A New Concept in Technical Trend Analysis, 1995"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_aroon,
    },
    FnDef {
        name: "aroonosc",
        family: Family::Momentum,
        inputs: HL,
        params: PERIOD2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("AROONOSC"),
        doc: "Aroon oscillator, aroon up - aroon down, in [-100, 100]. A flat series gives 0.",
        refs: &["Chande, A New Concept in Technical Trend Analysis, 1995"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_aroonosc,
    },
    FnDef {
        name: "plus_dm",
        family: Family::Momentum,
        inputs: HL,
        params: PERIOD1,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_dm,
        memory: Memory::Infinite { settle: settle_wilder1 },
        legacy: None,
        talib_name: Some("PLUS_DM"),
        doc: "Smoothed plus directional movement. The raw movement of a bar is high - previous high when that exceeds previous low - low and is positive, else 0. TA-Lib's smoothing: the sum of the first period - 1 movements on bar period - 1, then s - s / period + movement. At period 1 it is the raw movement from bar 1. A flat series gives 0.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_plus_dm,
    },
    FnDef {
        name: "minus_dm",
        family: Family::Momentum,
        inputs: HL,
        params: PERIOD1,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_dm,
        memory: Memory::Infinite { settle: settle_wilder1 },
        legacy: None,
        talib_name: Some("MINUS_DM"),
        doc: "Smoothed minus directional movement; as plus_dm for the move of the low. At period 1 it is the raw movement from bar 1. A flat series gives 0.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_minus_dm,
    },
    FnDef {
        name: "plus_di",
        family: Family::Momentum,
        inputs: HLC,
        params: PERIOD1_COMPAT,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Infinite { settle: settle_wilder1 },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::strength::directional_movement",
            tolerance: 0.0,
        }),
        talib_name: Some("PLUS_DI"),
        doc: "Plus directional indicator, 100 * smoothed plus DM / smoothed true range; first value on bar `period`. compat=\"native\" (default) seeds the smoothing with the plain sum of the first `period` movements, as the array function in indicators::strength does (that function is all NaN for a series shorter than 2 * period; the expression is not), and is NaN while the smoothed true range is 0 (a flat series). compat=\"talib\" sums the first period - 1 movements and then smooths from bar `period`, and gives 0 on a flat series; at period 1 TA-Lib's value is the raw movement over the true range without the factor 100, which talib keeps. The seeds differ and the difference decays as (1 - 1 / period) per bar.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::DefinedDifference,
        make: make_plus_di,
    },
    FnDef {
        name: "minus_di",
        family: Family::Momentum,
        inputs: HLC,
        params: PERIOD1_COMPAT,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Infinite { settle: settle_wilder1 },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::strength::directional_movement",
            tolerance: 0.0,
        }),
        talib_name: Some("MINUS_DI"),
        doc: "Minus directional indicator; as plus_di for the minus movement, with the same two compat modes and the same flat-series and period-1 behaviour.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::DefinedDifference,
        make: make_minus_di,
    },
    FnDef {
        name: "dx",
        family: Family::Momentum,
        inputs: HLC,
        params: PERIOD2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Infinite { settle: settle_wilder1 },
        legacy: None,
        talib_name: Some("DX"),
        doc: "Directional movement index, 100 * |+DI - -DI| / (+DI + -DI), TA-Lib's seeding (see plus_di compat=\"talib\"). A flat series, or equal DI lines of 0, gives 0.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_dx,
    },
    FnDef {
        name: "adx",
        family: Family::Momentum,
        inputs: HLC,
        params: PERIOD2_COMPAT,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_adx,
        memory: Memory::Infinite { settle: settle_wilder2 },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::strength::adx",
            tolerance: 0.0,
        }),
        talib_name: Some("ADX"),
        doc: "Average directional index: the mean of the first `period` DX values on bar 2 * period - 1, then (adx * (period - 1) + dx) / period. compat=\"native\" (default) is the array function in indicators::strength (its DM seed is the plain sum of the first `period` movements); compat=\"talib\" is TA-Lib's (sum of period - 1 movements, then smoothing). The seeds differ and the difference decays as (1 - 1 / period) per bar (measured against TA-Lib 0.6.4: up to 0.72 on 14). A flat series gives 0 in both.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::DefinedDifference,
        make: make_adx,
    },
    FnDef {
        name: "adxr",
        family: Family::Momentum,
        inputs: HLC,
        params: PERIOD2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_adxr,
        memory: Memory::Infinite { settle: settle_wilder2 },
        legacy: None,
        talib_name: Some("ADXR"),
        doc: "Average directional index rating: (adx + adx `period - 1` bars earlier) / 2, TA-Lib's seeding. First value on bar 3 * period - 2. A flat series gives 0.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_adxr,
    },
    FnDef {
        name: "cci",
        family: Family::Momentum,
        inputs: HLC,
        params: WALKED2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p_minus_1,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("CCI"),
        doc: "Commodity channel index: (typical price - its mean over `period` bars) / (0.015 * mean absolute deviation), typical price (high + low + close) / 3. A window with no deviation (flat) gives 0. Period at most 100000 (the window is walked on every bar).",
        refs: &["Lambert, Commodity Channel Index, Commodities, 1980"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_cci,
    },
    FnDef {
        name: "mfi",
        family: Family::Momentum,
        inputs: HLCV,
        params: WALKED2_COMPAT,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_p,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::volume::mfi",
            tolerance: 0.0,
        }),
        talib_name: Some("MFI"),
        doc: "Money flow index: typical price (high + low + close) / 3 times volume, summed over the last `period` bars as positive flow when the typical price rose and negative when it fell; first value on bar `period`. compat=\"native\" (default): 100 when there is no negative flow, so a flat series, or zero volume, reads 100, else 100 - 100 / (1 + positive / negative). compat=\"talib\": 100 * positive / (positive + negative), and 0 when the two sum to less than 1 (a flat series, zero volume, or a very thin market in price * volume units reads 0). The two agree to rounding wherever the total flow is at least 1 and some flow is negative. Period at most 100000 (the window is summed afresh on every bar).",
        refs: &["Quong, Soudack, Volume-Weighted RSI: Money Flow, Technical Analysis of Stocks & Commodities, 1989"],
        rev: 1,
        // The talib mode zeroes a total flow under 1, which is not scale-free.
        invariance: &[],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::DefinedDifference,
        make: make_mfi,
    },
    FnDef {
        name: "ultosc",
        family: Family::Momentum,
        inputs: HLC,
        params: ULTOSC_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_ultosc,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("ULTOSC"),
        doc: "Ultimate oscillator: 100 * (4 * A1 + 2 * A2 + A3) / 7, Ai = sum of buying pressure (close - min(low, previous close)) over sum of true range (max(high, previous close) - min(low, previous close)) for the periods in ascending order (they are sorted, as TA-Lib does). First value on bar max(period). A window with no range contributes 0, so a flat series gives 0. When all three periods are 1 TA-Lib prints a spurious value on bar 0; this is NaN there.",
        refs: &["Williams, The Ultimate Oscillator, Technical Analysis of Stocks & Commodities, 1985"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_ultosc,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::momentum as legacy_momentum;
    use crate::indicators::streaming::{
        StreamingIndicator, StreamingMacd, StreamingRoc, StreamingRsi,
    };
    use crate::indicators::{strength, volume};
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::lookup;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    const NAN: f64 = f64::NAN;

    fn i(v: i64) -> ParamVal {
        ParamVal::Int(v)
    }
    fn talib() -> ParamVal {
        ParamVal::Enum("talib")
    }
    fn native() -> ParamVal {
        ParamVal::Enum("native")
    }

    fn runner(seed: u64) -> TestRunner {
        TestRunner::new(Config {
            cases: 100,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(seed),
            ..Config::default()
        })
    }

    fn node(name: &str, params: &[ParamVal], n_in: usize, n_out: usize) -> Box<dyn Node> {
        let ins: Vec<usize> = (0..n_in).collect();
        let outs: Vec<usize> = (n_in..n_in + n_out).collect();
        (lookup(name).unwrap().make)(params, &ins, &outs).unwrap()
    }

    /// Batch: one `Vec` per output.
    fn batch(name: &str, params: &[ParamVal], cols: &[&[f64]], n_out: usize) -> Vec<Vec<f64>> {
        let n = cols[0].len();
        let mut outs = vec![vec![0.0; n]; n_out];
        let mut refs: Vec<&mut [f64]> = outs.iter_mut().map(|o| &mut o[..]).collect();
        node(name, params, cols.len(), n_out).run_batch(cols, &mut refs);
        outs
    }

    fn stream(name: &str, params: &[ParamVal], cols: &[&[f64]], n_out: usize) -> Vec<Vec<f64>> {
        let n_in = cols.len();
        let mut nd = node(name, params, n_in, n_out);
        let mut regs = Regs::new(n_in + n_out);
        let mut outs = vec![Vec::new(); n_out];
        for t in 0..cols[0].len() {
            for (j, c) in cols.iter().enumerate() {
                regs.set(j, c[t]);
            }
            nd.step(&mut regs);
            for (o, v) in outs.iter_mut().enumerate() {
                v.push(regs.get(n_in + o));
            }
        }
        outs
    }

    fn one(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<f64> {
        batch(name, params, cols, 1).remove(0)
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|&x| canon_bits(x)).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1.0)
    }

    type Bars = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    /// A deterministic bar series with trends and reversals and a range on every bar.
    fn bars(n: usize) -> Bars {
        let (mut o, mut h, mut l, mut c, mut v) = (vec![], vec![], vec![], vec![], vec![]);
        let mut px = 100.0;
        for t in 0..n {
            let step = ((t * 37 % 19) as f64 - 9.0) * 0.173 + ((t * 11 % 7) as f64 - 3.0) * 0.05;
            let open = px;
            px += step;
            o.push(open);
            c.push(px);
            h.push(open.max(px) + 0.3 + (t % 5) as f64 * 0.11);
            l.push(open.min(px) - 0.25 - (t % 3) as f64 * 0.13);
            v.push(1000.0 + ((t * 53) % 400) as f64);
        }
        (o, h, l, c, v)
    }

    const THIRTY: [&str; 30] = [
        "adx", "adxr", "apo", "aroon", "aroonosc", "bop", "cci", "cmo", "dx", "macd", "macdext",
        "macdfix", "mfi", "minus_di", "minus_dm", "mom", "plus_di", "plus_dm", "ppo", "roc",
        "rocp", "rocr", "rocr100", "rsi", "stoch", "stochf", "stochrsi", "trix", "ultosc", "willr",
    ];

    #[test]
    fn the_thirty_talib_functions_and_stochastic_are_registered_with_talib_names() {
        for n in THIRTY {
            let d = lookup(n).unwrap_or_else(|| panic!("{n} missing"));
            assert_eq!(d.talib_name.map(str::to_lowercase).as_deref(), Some(n));
        }
        assert_eq!(DEFS.len(), THIRTY.len() + 1);
        assert!(lookup("stochastic").unwrap().talib_name.is_none());
    }

    #[test]
    fn parameter_and_output_names_follow_talib() {
        let table: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/oracle/talib_functions.json"))
                .unwrap();
        for n in THIRTY {
            let t = &table["functions"][n.to_uppercase()];
            let mut want: Vec<String> = t["parameters"]
                .as_object()
                .unwrap()
                .keys()
                .map(|k| if k == "timeperiod" { "period".to_string() } else { k.clone() })
                .collect();
            want.sort();
            let d = lookup(n).unwrap();
            let mut got: Vec<String> = d
                .params
                .iter()
                .filter(|p| p.name != "compat")
                .map(|p| p.name.to_string())
                .collect();
            got.sort();
            assert_eq!(got, want, "{n} parameters");
            let outs: Vec<&str> =
                t["outputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            if outs.len() > 1 {
                assert_eq!(d.outputs, &outs[..], "{n} outputs");
            }
        }
        for n in ["rsi", "macd", "adx", "mfi", "plus_di", "minus_di"] {
            let last = lookup(n).unwrap().params.last().unwrap();
            assert_eq!(last.name, "compat", "{n}");
            assert_eq!(last.default, Some(ParamVal::Enum("native")));
        }
        for n in ["roc", "stoch", "stochf", "dx", "adxr", "cmo", "macdext", "macdfix"] {
            assert!(lookup(n).unwrap().params.iter().all(|p| p.name != "compat"), "{n}");
        }
    }

    fn cases() -> Vec<(&'static str, Vec<ParamVal>, usize)> {
        let mut v = vec![
            ("mom", vec![i(5)], 1),
            ("roc", vec![i(5)], 1),
            ("rocp", vec![i(5)], 1),
            ("rocr", vec![i(5)], 1),
            ("rocr100", vec![i(5)], 1),
            ("bop", vec![], 1),
            ("cmo", vec![i(7)], 1),
            ("macd", vec![i(5), i(12), i(4), native()], 3),
            ("macd", vec![i(5), i(12), i(4), talib()], 3),
            ("macd", vec![i(12), i(5), i(4), talib()], 3),
            ("macdfix", vec![i(4)], 3),
            ("trix", vec![i(4)], 1),
            ("stochastic", vec![i(6), i(3)], 2),
            ("stochf", vec![i(5), i(3), i(0)], 2),
            ("stoch", vec![i(5), i(3), i(0), i(3), i(0)], 2),
            ("stochrsi", vec![i(6), i(4), i(3), i(1)], 2),
            ("willr", vec![i(6)], 1),
            ("aroon", vec![i(6)], 2),
            ("aroonosc", vec![i(6)], 1),
            ("plus_dm", vec![i(5)], 1),
            ("minus_dm", vec![i(5)], 1),
            ("plus_dm", vec![i(1)], 1),
            ("dx", vec![i(5)], 1),
            ("adxr", vec![i(5)], 1),
            ("cci", vec![i(6)], 1),
            ("ultosc", vec![i(3), i(5), i(8)], 1),
            ("ultosc", vec![i(8), i(3), i(5)], 1),
        ];
        for c in [native(), talib()] {
            for p in [1, 5] {
                v.push(("plus_di", vec![i(p), c], 1));
                v.push(("minus_di", vec![i(p), c], 1));
            }
            v.push(("rsi", vec![i(7), c], 1));
            v.push(("adx", vec![i(5), c], 1));
            v.push(("mfi", vec![i(5), c], 1));
        }
        for m in [0, 1, 2, 3, 4, 5, 6, 8] {
            v.push(("apo", vec![i(4), i(9), i(m)], 1));
            v.push(("ppo", vec![i(4), i(9), i(m)], 1));
            v.push(("macdext", vec![i(4), i(m), i(9), i(m), i(3), i(1)], 3));
            v.push(("stochf", vec![i(4), i(3), i(m)], 2));
            v.push(("stoch", vec![i(4), i(3), i(m), i(2), i(1)], 2));
        }
        v
    }

    /// The columns a function reads, from the generated bars.
    fn columns<'a>(
        name: &str,
        o: &'a [f64],
        h: &'a [f64],
        l: &'a [f64],
        c: &'a [f64],
        v: &'a [f64],
    ) -> Vec<&'a [f64]> {
        match name {
            "bop" => vec![o, h, l, c],
            "mfi" => vec![h, l, c, v],
            "aroon" | "aroonosc" | "plus_dm" | "minus_dm" => vec![h, l],
            "stochastic" | "stochf" | "stoch" | "willr" | "dx" | "adxr" | "cci" | "ultosc"
            | "plus_di" | "minus_di" | "adx" => vec![h, l, c],
            _ => vec![c],
        }
    }

    #[test]
    fn warmup_formula_matches_first_non_nan_for_every_output() {
        let (o, h, l, c, v) = bars(120);
        for (name, params, n_out) in cases() {
            let cols = columns(name, &o, &h, &l, &c, &v);
            let outs = batch(name, &params, &cols, n_out);
            let w = (lookup(name).unwrap().warmup)(&params);
            for (k, out) in outs.iter().enumerate() {
                let first = out.iter().position(|x| !x.is_nan()).unwrap();
                assert_eq!(first, w, "{name} {params:?} output {k}");
            }
        }
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans_and_infinities() {
        let value = prop_oneof![8 => 50.0f64..150.0, 1 => Just(NAN), 1 => Just(f64::INFINITY)];
        let strategy = prop::collection::vec(
            (value.clone(), value.clone(), value.clone(), value.clone(), value),
            0..140,
        );
        runner(31)
            .run(&strategy, |rows| {
                let col = |k: usize| -> Vec<f64> {
                    rows.iter().map(|r| [r.0, r.1, r.2, r.3, r.4][k]).collect()
                };
                let (o, h, l, c, v) = (col(0), col(1), col(2), col(3), col(4));
                for (name, params, n_out) in cases() {
                    let cols = columns(name, &o, &h, &l, &c, &v);
                    let (x, y) =
                        (batch(name, &params, &cols, n_out), stream(name, &params, &cols, n_out));
                    for k in 0..n_out {
                        prop_assert_eq!(
                            bits(&x[k]),
                            bits(&y[k]),
                            "{} {:?} output {}",
                            name,
                            params,
                            k
                        );
                    }
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn mama_is_refused() {
        for (name, params) in [
            ("apo", vec![i(4), i(9), i(7)]),
            ("macdext", vec![i(4), i(7), i(9), i(0), i(3), i(0)]),
            ("stochf", vec![i(4), i(3), i(7)]),
        ] {
            let err = (lookup(name).unwrap().make)(&params, &[0, 1, 2], &[3, 4, 5]).err().unwrap();
            assert!(err.contains("MAMA"), "{name}: {err}");
        }
    }

    #[test]
    fn windowed_kernels_recover_and_recursive_ones_skip_a_nan_bar() {
        let (o, h, l, c, v) = bars(90);
        for (name, params, n_out) in cases() {
            let cols = columns(name, &o, &h, &l, &c, &v);
            let owned: Vec<Vec<f64>> = cols
                .iter()
                .map(|x| {
                    let mut y = x.to_vec();
                    y[40] = NAN;
                    y
                })
                .collect();
            let refs: Vec<&[f64]> = owned.iter().map(|y| &y[..]).collect();
            let out = batch(name, &params, &refs, n_out);
            for out in &out {
                assert!(out[40].is_nan(), "{name}");
                assert!(!out[89].is_nan(), "{name} {params:?} did not recover");
            }
            if lookup(name).unwrap().nan_rule == NanRule::Recursive {
                // The same stream with the bar removed.
                let gone: Vec<Vec<f64>> = cols
                    .iter()
                    .map(|x| {
                        let mut y = x.to_vec();
                        y.remove(40);
                        y
                    })
                    .collect();
                let refs: Vec<&[f64]> = gone.iter().map(|y| &y[..]).collect();
                let want = batch(name, &params, &refs, n_out);
                for (a, b) in out.iter().zip(&want) {
                    assert_eq!(bits(&a[41..]), bits(&b[40..]), "{name} {params:?}");
                }
            }
        }
    }

    // ------------------------------------------------------------------------ hand values

    #[test]
    fn change_family_hand_values() {
        // Period 2 on 2, 4, 5, 0, 8: bases 2 (bar 2), 4 (bar 3), 5 (bar 4).
        let x = [2.0, 4.0, 5.0, 0.0, 8.0];
        let mom = one("mom", &[i(2)], &[&x]);
        assert!(mom[0].is_nan() && mom[1].is_nan());
        assert_eq!(mom[2..], [3.0, -4.0, 3.0]);
        // roc: (5 - 2) / 2 * 100 = 150, (0 - 4) / 4 * 100 = -100, (8 - 5) / 5 * 100 = 60.
        let want = [
            ("roc", [150.0, -100.0, 60.0]),
            ("rocp", [1.5, -1.0, 0.6]),
            ("rocr", [2.5, 0.0, 1.6]),
            ("rocr100", [250.0, 0.0, 160.0]),
        ];
        for (name, w) in want {
            let out = one(name, &[i(2)], &[&x]);
            assert!(out[1].is_nan(), "{name}");
            for (a, b) in out[2..].iter().zip(w) {
                assert!(close(*a, b), "{name}: {a} {b}");
            }
        }
        // A zero base: rocp, rocr and rocr100 give 0; roc keeps the last value, as the streaming
        // core does (NaN while it has none).
        let z = [0.0, 1.0, 5.0, 2.0];
        assert_eq!(one("rocr", &[i(2)], &[&z])[2..], [0.0, 2.0]);
        assert_eq!(one("rocp", &[i(2)], &[&z])[2..], [0.0, 1.0]);
        assert!(one("roc", &[i(2)], &[&z])[2].is_nan());
        assert_eq!(one("roc", &[i(2)], &[&z])[3], 100.0);
    }

    #[test]
    fn roc_equals_the_streaming_core_bitwise_on_finite_data() {
        let strategy = (1i64..30, prop::collection::vec(-1e3f64..1e3, 0..200));
        runner(32)
            .run(&strategy, |(p, data)| {
                let mut core = StreamingRoc::new(p as usize);
                let legacy: Vec<f64> =
                    data.iter().map(|&x| core.update(x).unwrap_or(NAN)).collect();
                prop_assert_eq!(bits(&one("roc", &[i(p)], &[&data])), bits(&legacy));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn bop_hand_values_and_a_bar_without_range() {
        // (11 - 10) / (13 - 9) = 0.25; a bar with no range gives 0; (6.5 - 7) / (8 - 6).
        let o = [10.0, 5.0, 7.0];
        let h = [13.0, 5.0, 8.0];
        let l = [9.0, 5.0, 6.0];
        let c = [11.0, 5.0, 6.5];
        assert_eq!(one("bop", &[], &[&o, &h, &l, &c]), [0.25, 0.0, -0.25]);
    }

    #[test]
    fn rsi_hand_values_in_both_modes() {
        // Period 2 on 10, 11, 10, 12: changes +1, -1, +2.
        // Seed (bar 2): gain (1 + 0) / 2 = 0.5, loss (0 + 1) / 2 = 0.5, rsi = 50.
        // Bar 3: gain 0.5 * 2 + 0.5 * 0.5 = 1.25 (TA-Lib form (0.5 * 1 + 2) / 2), loss
        // 0.5 * 0 + 0.5 * 0.5 = 0.25, rsi = 100 * 1.25 / 1.5.
        let x = [10.0, 11.0, 10.0, 12.0];
        for mode in [native(), talib()] {
            let out = one("rsi", &[i(2), mode], &[&x]);
            assert!(out[0].is_nan() && out[1].is_nan());
            assert_eq!(out[2], 50.0);
            assert!(close(out[3], 100.0 * 1.25 / 1.5), "{mode:?} {}", out[3]);
        }
    }

    #[test]
    fn rsi_flat_is_100_native_and_0_talib() {
        let flat = [100.0; 12];
        let n = one("rsi", &[i(5), native()], &[&flat]);
        let t = one("rsi", &[i(5), talib()], &[&flat]);
        assert!(n[4].is_nan() && t[4].is_nan());
        assert_eq!((n[5], n[11]), (100.0, 100.0));
        assert_eq!((t[5], t[11]), (0.0, 0.0));
        // Rises only: no loss, both modes say 100. Falls only: 0 in both.
        let up: Vec<f64> = (0..12).map(f64::from).collect();
        let down: Vec<f64> = (0..12).map(|v| 100.0 - f64::from(v)).collect();
        for mode in [native(), talib()] {
            assert_eq!(one("rsi", &[i(5), mode], &[&up])[11], 100.0);
            assert_eq!(one("rsi", &[i(5), mode], &[&down])[11], 0.0);
        }
    }

    #[test]
    fn rsi_native_equals_the_streaming_core_bitwise_on_finite_data() {
        let strategy = (1i64..30, prop::collection::vec(-1e3f64..1e3, 0..200));
        runner(33)
            .run(&strategy, |(p, data)| {
                let mut core = StreamingRsi::new(p as usize);
                let legacy: Vec<f64> =
                    data.iter().map(|&x| core.update(x).unwrap_or(NAN)).collect();
                prop_assert_eq!(bits(&one("rsi", &[i(p), native()], &[&data])), bits(&legacy));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn rsi_modes_agree_to_rounding_on_a_moving_series() {
        let (_, _, _, c, _) = bars(200);
        let n = one("rsi", &[i(14), native()], &[&c]);
        let t = one("rsi", &[i(14), talib()], &[&c]);
        for (a, b) in n.iter().zip(&t).skip(14) {
            assert!((a - b).abs() < 1e-9, "{a} {b}");
        }
    }

    #[test]
    fn cmo_hand_value() {
        // Same seed as the rsi case: gain 0.5, loss 0.5 gives 0; then gain 1.25, loss 0.25 gives
        // 100 * (1.25 - 0.25) / 1.5.
        let x = [10.0, 11.0, 10.0, 12.0];
        let out = one("cmo", &[i(2)], &[&x]);
        assert_eq!(out[2], 0.0);
        assert!(close(out[3], 100.0 / 1.5));
        assert_eq!(one("cmo", &[i(2)], &[&[5.0; 6]])[5], 0.0);
    }

    #[test]
    fn macd_native_equals_the_streaming_core_bitwise_on_finite_data() {
        let strategy = (2i64..8, 9i64..20, 1i64..8, prop::collection::vec(-1e3f64..1e3, 0..200));
        runner(34)
            .run(&strategy, |(f, s, g, data)| {
                let mut core = StreamingMacd::new(f as usize, s as usize, g as usize);
                let legacy: Vec<(f64, f64, f64)> =
                    data.iter().map(|&x| core.update(x).unwrap_or((NAN, NAN, NAN))).collect();
                let o = batch("macd", &[i(f), i(s), i(g), native()], &[&data], 3);
                for (t, (m, sg, hs)) in legacy.iter().enumerate() {
                    prop_assert_eq!(canon_bits(o[0][t]), canon_bits(*m), "macd {}", t);
                    prop_assert_eq!(canon_bits(o[1][t]), canon_bits(*sg), "signal {}", t);
                    prop_assert_eq!(canon_bits(o[2][t]), canon_bits(*hs), "hist {}", t);
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn macd_hand_values_native_and_talib() {
        // fast 2, slow 3, signal 1 on 1, 2, 4, 5. Alphas: fast 2 / 3, slow 1 / 2, signal 1.
        // Native: fast seed on bars 0..=1 = 1.5; slow seed on bars 0..=2 = 7 / 3 (bar 2).
        // Bar 2: fast = 2 / 3 * 4 + 1 / 3 * 1.5 = 19 / 6, macd = 19 / 6 - 14 / 6 = 5 / 6.
        // Bar 3: fast = 2 / 3 * 5 + 1 / 3 * 19 / 6 = 79 / 18, slow = 0.5 * 5 + 0.5 * 7 / 3 =
        // 66 / 18, macd = 13 / 18. The signal (period 1) is the line, so the histogram is 0.
        let x = [1.0, 2.0, 4.0, 5.0];
        let n = batch("macd", &[i(2), i(3), i(1), native()], &[&x], 3);
        assert!(n[0][0].is_nan() && n[0][1].is_nan());
        assert!(close(n[0][2], 5.0 / 6.0) && close(n[0][3], 13.0 / 18.0));
        assert!(close(n[1][2], n[0][2]) && close(n[2][3], 0.0));
        // TA-Lib: the fast EMA starts on bar 2 with the mean of bars 1..=2 = 3, so the line is
        // 3 - 7 / 3 = 2 / 3 there; bar 3: fast = 2 / 3 * 5 + 1 / 3 * 3 = 13 / 3, slow = 11 / 3.
        let t = batch("macd", &[i(2), i(3), i(1), talib()], &[&x], 3);
        assert!(close(t[0][2], 2.0 / 3.0) && close(t[0][3], 2.0 / 3.0));
    }

    #[test]
    fn macd_talib_takes_the_shorter_period_as_fast() {
        let (_, _, _, c, _) = bars(80);
        let a = batch("macd", &[i(12), i(5), i(4), talib()], &[&c], 3);
        let b = batch("macd", &[i(5), i(12), i(4), talib()], &[&c], 3);
        for k in 0..3 {
            assert_eq!(bits(&a[k]), bits(&b[k]));
        }
    }

    #[test]
    fn macdext_of_emas_is_the_talib_macd_and_apo_is_its_line() {
        let (_, _, _, c, _) = bars(120);
        let ext = batch("macdext", &[i(5), i(1), i(12), i(1), i(4), i(1)], &[&c], 3);
        let mac = batch("macd", &[i(5), i(12), i(4), talib()], &[&c], 3);
        for k in 0..3 {
            for (a, b) in ext[k].iter().zip(&mac[k]) {
                assert!((a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-9, "{a} {b}");
            }
        }
        // APO runs both averages over the whole series (the fast EMA is seeded on its own first
        // values), so it differs from the MACD line by the fast seed, which decays.
        let apo = one("apo", &[i(5), i(12), i(1)], &[&c]);
        assert!(apo[10].is_nan() && apo[11].is_finite());
    }

    #[test]
    fn po_hand_values() {
        // SMA 2 and 3 on 1, 2, 4, 8: bar 2 fast 3, slow 7 / 3; bar 3 fast 6, slow 14 / 3.
        let x = [1.0, 2.0, 4.0, 8.0];
        let apo = one("apo", &[i(2), i(3), i(0)], &[&x]);
        assert!(apo[0].is_nan() && apo[1].is_nan());
        assert!(close(apo[2], 3.0 - 7.0 / 3.0) && close(apo[3], 6.0 - 14.0 / 3.0));
        let ppo = one("ppo", &[i(2), i(3), i(0)], &[&x]);
        assert!(close(ppo[2], (3.0 - 7.0 / 3.0) / (7.0 / 3.0) * 100.0));
        assert!(close(ppo[3], (6.0 - 14.0 / 3.0) / (14.0 / 3.0) * 100.0));
        // The slower period is the slow one whichever order they are given.
        assert_eq!(bits(&apo), bits(&one("apo", &[i(3), i(2), i(0)], &[&x])));
    }

    #[test]
    fn trix_hand_value() {
        // Period 2, alpha 2 / 3, on 1, 2, 3, 4, 5. e1: seed 1.5 (bar 1), then 2.5, 3.5, 4.5.
        // e2 (from e1's bar 1): seed (1.5 + 2.5) / 2 = 2 (bar 2), then 2 / 3 * 3.5 + 2 / 3 = 3
        // (bar 3), 2 / 3 * 4.5 + 1 = 4 (bar 4). e3: seed (2 + 3) / 2 = 2.5 (bar 3), then
        // 2 / 3 * 4 + 2.5 / 3 = 3.5 (bar 4). trix at bar 4 = (3.5 / 2.5 - 1) * 100 = 40.
        // Warm-up 3 * (2 - 1) + 1 = 4.
        let out = one("trix", &[i(2)], &[&[1.0, 2.0, 3.0, 4.0, 5.0]]);
        assert!(out[..4].iter().all(|v| v.is_nan()));
        assert!(close(out[4], 40.0), "{}", out[4]);
    }

    #[test]
    fn stochastic_hand_values() {
        // k_period 3, d_period 2. Highs 5, 6, 7, 6; lows 3, 4, 4, 3; closes 4, 5, 6, 4.
        // Bar 2: HH 7, LL 3, k = (6 - 3) / 4 * 100 = 75. Bar 3: HH 7, LL 3, k = (4 - 3) / 4 * 100
        // = 25, d = (75 + 25) / 2 = 50. Bar 2's k is held back until d is ready (bar 3).
        let h = [5.0, 6.0, 7.0, 6.0];
        let l = [3.0, 4.0, 4.0, 3.0];
        let c = [4.0, 5.0, 6.0, 4.0];
        let o = batch("stochastic", &[i(3), i(2)], &[&h, &l, &c], 2);
        assert!(o[0][..3].iter().all(|v| v.is_nan()));
        assert_eq!((o[0][3], o[1][3]), (25.0, 50.0));
        // A zero range reads 50; the TA-Lib fast stochastic reads 0.
        let flat = [9.0; 6];
        let s = batch("stochastic", &[i(3), i(2)], &[&flat, &flat, &flat], 2);
        assert_eq!((s[0][5], s[1][5]), (50.0, 50.0));
        let f = batch("stochf", &[i(3), i(2), i(0)], &[&flat, &flat, &flat], 2);
        assert_eq!((f[0][5], f[1][5]), (0.0, 0.0));
    }

    #[test]
    fn stochastic_equals_the_array_function_bitwise_where_both_are_defined() {
        let (_, h, l, c, _) = bars(200);
        for (k, d) in [(14usize, 3usize), (5, 3), (1, 1), (7, 5)] {
            let legacy = legacy_momentum::stochastic(&h, &l, &c, k, d).unwrap();
            let o = batch("stochastic", &[i(k as i64), i(d as i64)], &[&h, &l, &c], 2);
            let first = k - 1 + d - 1;
            for (t, (&ok, &od)) in o[0].iter().zip(&o[1]).enumerate() {
                if t < first {
                    assert!(ok.is_nan() && od.is_nan());
                } else {
                    assert_eq!(canon_bits(ok), canon_bits(legacy.k[t]), "k {t}");
                    assert_eq!(canon_bits(od), canon_bits(legacy.d[t]), "d {t}");
                }
            }
        }
    }

    #[test]
    fn stochf_stoch_and_stochrsi_hand_values() {
        // fastk_period 3 on the same bars: k = 75 (bar 2), 25 (bar 3).
        let h = [5.0, 6.0, 7.0, 6.0];
        let l = [3.0, 4.0, 4.0, 3.0];
        let c = [4.0, 5.0, 6.0, 4.0];
        let f = batch("stochf", &[i(3), i(2), i(0)], &[&h, &l, &c], 2);
        assert!(f[0][2].is_nan() && f[1][2].is_nan());
        assert_eq!((f[0][3], f[1][3]), (25.0, 50.0));
        // stoch with slowk_period 1 (identity) and slowd 2 is stochf's d.
        let s = batch("stoch", &[i(3), i(1), i(0), i(2), i(0)], &[&h, &l, &c], 2);
        assert_eq!((s[0][3], s[1][3]), (25.0, 50.0));
        assert!(s[0][2].is_nan());
        // stochrsi: the RSI(2) of 10, 11, 10, 12 is 50 (bar 2) then 100 * 1.25 / 1.5 (bar 3);
        // highs, lows and closes are all the RSI, so fastk over 2 bars is
        // 100 * (r3 - min) / (max - min) = 100 (the newest is the high), and fastd of period 1 is
        // fastk.
        let x = [10.0, 11.0, 10.0, 12.0];
        let r = one("rsi", &[i(2), talib()], &[&x]);
        let sr = batch("stochrsi", &[i(2), i(2), i(1), i(0)], &[&x], 2);
        assert!(sr[0][2].is_nan());
        assert!(close(sr[0][3], 100.0 * (r[3] - r[2]) / (r[3] - r[2])));
        assert!(close(sr[1][3], sr[0][3]));
    }

    #[test]
    fn willr_aroon_hand_values_and_flat() {
        // willr period 3: bar 2 HH 7, LL 3, close 6: -100 * (7 - 6) / 4 = -25; bar 3 close 4:
        // -100 * (7 - 4) / 4 = -75.
        let h = [5.0, 6.0, 7.0, 6.0];
        let l = [3.0, 4.0, 4.0, 3.0];
        let c = [4.0, 5.0, 6.0, 4.0];
        let w = one("willr", &[i(3)], &[&h, &l, &c]);
        assert!(w[1].is_nan());
        assert_eq!((w[2], w[3]), (-25.0, -75.0));
        // aroon period 2 over the last 3 bars. Bar 2: highs 5, 6, 7 -> highest 0 bars ago, up =
        // 100; lows 3, 4, 4 -> lowest 2 bars ago (bar 0), down = 100 * (2 - 2) / 2 = 0.
        // Bar 3: highs 6, 7, 6 -> highest 1 bar ago, up = 50; lows 4, 4, 3 -> 0 ago, down = 100.
        let a = batch("aroon", &[i(2)], &[&h, &l], 2);
        assert!(a[0][1].is_nan());
        assert_eq!((a[0][2], a[1][2]), (0.0, 100.0));
        assert_eq!((a[0][3], a[1][3]), (100.0, 50.0));
        let osc = one("aroonosc", &[i(2)], &[&h, &l]);
        assert_eq!((osc[2], osc[3]), (100.0, -50.0));
        // Flat: ties keep the latest bar, so up = down = 100 and the oscillator is 0.
        let flat = [4.0; 6];
        let af = batch("aroon", &[i(3)], &[&flat, &flat], 2);
        assert_eq!((af[0][5], af[1][5]), (100.0, 100.0));
        assert_eq!(one("aroonosc", &[i(3)], &[&flat, &flat])[5], 0.0);
        assert_eq!(one("willr", &[i(3)], &[&flat, &flat, &flat])[5], 0.0);
    }

    #[test]
    fn directional_movement_hand_values() {
        // Bars (high, low): (10, 8), (11, 9), (12, 9.5), (11.5, 8.5).
        // Bar 1: up 1, down -1 -> +DM 1. Bar 2: up 1, down -0.5 -> +DM 1. Bar 3: up -0.5, down 1
        // -> -DM 1.
        let h = [10.0, 11.0, 12.0, 11.5];
        let l = [8.0, 9.0, 9.5, 8.5];
        // period 3: sum of the first 2 movements on bar 2 (1 + 1 = 2); bar 3: 2 - 2 / 3 + 0.
        let plus = one("plus_dm", &[i(3)], &[&h, &l]);
        assert!(plus[0].is_nan() && plus[1].is_nan());
        assert_eq!(plus[2], 2.0);
        assert!(close(plus[3], 2.0 - 2.0 / 3.0));
        let minus = one("minus_dm", &[i(3)], &[&h, &l]);
        assert_eq!(minus[2], 0.0);
        assert_eq!(minus[3], 1.0);
        // Period 1 is the raw movement from bar 1; period 2 starts on bar 1 with the first one.
        assert_eq!(one("plus_dm", &[i(1)], &[&h, &l])[1..], [1.0, 1.0, 0.0]);
        let p2 = one("plus_dm", &[i(2)], &[&h, &l]);
        assert_eq!(p2[1], 1.0);
        assert_eq!(p2[2], 1.0 - 0.5 + 1.0);
    }

    #[test]
    fn di_dx_hand_values_the_period_1_quirk_and_flat() {
        // The bars above with closes 9, 10, 11, 9.5. True range: bar 1 max(2, |11 - 9|, |9 - 9|)
        // = 2, bar 2 max(2.5, |12 - 10|, |9.5 - 10|) = 2.5, bar 3 max(3, 0.5, 2.5) = 3.
        let h = [10.0, 11.0, 12.0, 11.5];
        let l = [8.0, 9.0, 9.5, 8.5];
        let c = [9.0, 10.0, 11.0, 9.5];
        // TA-Lib, period 2: bar 1 only accumulates (+DM 1, -DM 0, TR 2); bar 2 smooths:
        // +DM 1 - 0.5 + 1 = 1.5, -DM 0, TR 2 - 1 + 2.5 = 3.5. +DI = 100 * 1.5 / 3.5, -DI = 0,
        // DX = 100.
        let t = one("plus_di", &[i(2), talib()], &[&h, &l, &c]);
        assert!(t[0].is_nan() && t[1].is_nan());
        assert!(close(t[2], 100.0 * 1.5 / 3.5));
        assert_eq!(one("minus_di", &[i(2), talib()], &[&h, &l, &c])[2], 0.0);
        assert!(close(one("dx", &[i(2)], &[&h, &l, &c])[2], 100.0));
        // Native, period 2: the plain sums of bars 1 and 2: +DM 2, TR 4.5, +DI = 100 * 2 / 4.5.
        let n = one("plus_di", &[i(2), native()], &[&h, &l, &c]);
        assert!(close(n[2], 100.0 * 2.0 / 4.5));
        // Period 1: TA-Lib's value is the movement over the true range without the factor 100;
        // native multiplies by 100.
        let one_t = one("plus_di", &[i(1), talib()], &[&h, &l, &c]);
        let one_n = one("plus_di", &[i(1), native()], &[&h, &l, &c]);
        assert!(close(one_t[1], 0.5) && close(one_n[1], 50.0));
        // Flat: TA-Lib 0, native DI NaN, ADX 0 in both.
        let flat = [50.0; 12];
        assert_eq!(one("plus_di", &[i(3), talib()], &[&flat, &flat, &flat])[11], 0.0);
        assert!(one("plus_di", &[i(3), native()], &[&flat, &flat, &flat])[11].is_nan());
        for mode in [native(), talib()] {
            assert_eq!(one("adx", &[i(3), mode], &[&flat, &flat, &flat])[11], 0.0);
        }
        assert_eq!(one("adxr", &[i(3)], &[&flat, &flat, &flat])[11], 0.0);
        assert_eq!(one("dx", &[i(3)], &[&flat, &flat, &flat])[11], 0.0);
    }

    #[test]
    fn adx_hand_value() {
        // Period 2 on the bars of the previous test extended by one: TA-Lib's DX on bars 2 and 3
        // averaged. DX(2) = 100 (above). Bar 3: -DM 1 against +DM 0: +DM s = 1.5 - 0.75 + 0 =
        // 0.75, -DM s = 0 - 0 + 1 = 1, TR s = 3.5 - 1.75 + 3 = 4.75; +DI = 100 * 0.75 / 4.75,
        // -DI = 100 * 1 / 4.75, DX = 100 * 0.25 / 1.75. ADX (bar 3) = (100 + 100 * 0.25 / 1.75) / 2.
        let h = [10.0, 11.0, 12.0, 11.5];
        let l = [8.0, 9.0, 9.5, 8.5];
        let c = [9.0, 10.0, 11.0, 9.5];
        let out = one("adx", &[i(2), talib()], &[&h, &l, &c]);
        assert!(out[2].is_nan());
        assert!(close(out[3], (100.0 + 100.0 * 0.25 / 1.75) / 2.0), "{}", out[3]);
    }

    #[test]
    fn adx_native_equals_the_array_functions_bitwise_on_finite_data() {
        let (_, h, l, c, _) = bars(240);
        for p in [2usize, 5, 14] {
            let adx = strength::adx(&h, &l, &c, p).unwrap();
            let di = strength::directional_movement(&h, &l, &c, p).unwrap();
            let a = one("adx", &[i(p as i64), native()], &[&h, &l, &c]);
            let pl = one("plus_di", &[i(p as i64), native()], &[&h, &l, &c]);
            let mi = one("minus_di", &[i(p as i64), native()], &[&h, &l, &c]);
            assert_eq!(bits(&a), bits(&adx), "adx p={p}");
            assert_eq!(bits(&a), bits(&di.adx), "adx via directional_movement p={p}");
            assert_eq!(bits(&pl), bits(&di.plus_di), "plus_di p={p}");
            assert_eq!(bits(&mi), bits(&di.minus_di), "minus_di p={p}");
        }
    }

    #[test]
    fn adxr_averages_the_adx_p_minus_1_bars_apart() {
        let (_, h, l, c, _) = bars(120);
        let p = 5usize;
        let adx = one("adx", &[i(p as i64), talib()], &[&h, &l, &c]);
        let adxr = one("adxr", &[i(p as i64)], &[&h, &l, &c]);
        let w = 3 * p - 2;
        assert!(adxr[w - 1].is_nan());
        for t in w..120 {
            assert!(close(adxr[t], (adx[t] + adx[t - (p - 1)]) / 2.0), "{t}");
        }
    }

    #[test]
    fn mfi_hand_values_and_flat() {
        // Period 2. Typical prices 10, 11, 10.5, 12 (high = low = close); volumes 100, 200, 300,
        // 400. Bar 2: flows of bars 1 and 2: bar 1 rose (+2200), bar 2 fell (-3150): pos 2200,
        // neg 3150. Bar 3: bar 2 fell (-3150), bar 3 rose (+4800): pos 4800, neg 3150.
        let t = [10.0, 11.0, 10.5, 12.0];
        let v = [100.0, 200.0, 300.0, 400.0];
        for mode in [native(), talib()] {
            let m = one("mfi", &[i(2), mode], &[&t, &t, &t, &v]);
            assert!(m[0].is_nan() && m[1].is_nan());
            assert!(close(m[2], 100.0 * 2200.0 / 5350.0), "{mode:?} {}", m[2]);
            assert!(close(m[3], 100.0 * 4800.0 / 7950.0), "{mode:?} {}", m[3]);
        }
        // Flat (and zero volume): native 100, talib 0.
        let flat = [20.0; 8];
        let vol = [500.0; 8];
        assert_eq!(one("mfi", &[i(3), native()], &[&flat, &flat, &flat, &vol])[7], 100.0);
        assert_eq!(one("mfi", &[i(3), talib()], &[&flat, &flat, &flat, &vol])[7], 0.0);
        let moving = [20.0, 21.0, 20.0, 22.0, 21.0, 23.0, 22.0, 24.0];
        let zero = [0.0; 8];
        let cols: [&[f64]; 4] = [&moving, &moving, &moving, &zero];
        assert_eq!(one("mfi", &[i(3), native()], &cols)[7], 100.0);
        assert_eq!(one("mfi", &[i(3), talib()], &cols)[7], 0.0);
        // A total flow under 1 reads 0 in talib mode only.
        let thin = [1e-4; 8];
        let up: Vec<f64> = (0..8).map(|k| 1.0 + f64::from(k)).collect();
        assert_eq!(one("mfi", &[i(3), talib()], &[&up, &up, &up, &thin])[7], 0.0);
        assert_eq!(one("mfi", &[i(3), native()], &[&up, &up, &up, &thin])[7], 100.0);
    }

    #[test]
    fn mfi_native_equals_the_array_function_bitwise_on_finite_data() {
        let (_, h, l, c, v) = bars(200);
        for p in [2usize, 5, 14] {
            let legacy = volume::mfi(&h, &l, &c, &v, p).unwrap();
            let got = one("mfi", &[i(p as i64), native()], &[&h, &l, &c, &v]);
            assert_eq!(bits(&got), bits(&legacy), "p={p}");
        }
    }

    #[test]
    fn mfi_modes_agree_to_rounding_when_some_flow_is_negative() {
        let (_, h, l, c, v) = bars(200);
        let n = one("mfi", &[i(14), native()], &[&h, &l, &c, &v]);
        let t = one("mfi", &[i(14), talib()], &[&h, &l, &c, &v]);
        for (a, b) in n.iter().zip(&t).skip(14) {
            assert!((a - b).abs() < 1e-9, "{a} {b}");
        }
    }

    #[test]
    fn cci_hand_value_and_flat() {
        // Period 3 on typical prices 10, 11, 15 (high = low = close): mean 12, deviations 2, 1, 3
        // -> mean deviation 2, cci = (15 - 12) / (0.015 * 2) = 100.
        let t = [10.0, 11.0, 15.0];
        let out = one("cci", &[i(3)], &[&t, &t, &t]);
        assert!(out[1].is_nan());
        assert!(close(out[2], 100.0), "{}", out[2]);
        assert_eq!(one("cci", &[i(3)], &[&[7.0; 5], &[7.0; 5], &[7.0; 5]])[4], 0.0);
    }

    #[test]
    fn ultosc_hand_value_sorting_and_flat() {
        // Periods 1, 2, 3 on bars (high, low, close): (10, 8, 9), (11, 9, 10), (12, 9.5, 11.5),
        // (11.5, 8.5, 9). BP = close - min(low, previous close), TR = max(high, previous close)
        // - min(low, previous close): bar 1 (pc 9): BP 10 - 9 = 1, TR 11 - 9 = 2; bar 2 (pc 10):
        // BP 11.5 - 9.5 = 2, TR 12 - 9.5 = 2.5; bar 3 (pc 11.5): BP 9 - 8.5 = 0.5, TR 11.5 - 8.5
        // = 3. Bar 3: A1 = 0.5 / 3, A2 = (2 + 0.5) / (2.5 + 3), A3 = (1 + 2 + 0.5) / (2 + 2.5 + 3).
        let h = [10.0, 11.0, 12.0, 11.5];
        let l = [8.0, 9.0, 9.5, 8.5];
        let c = [9.0, 10.0, 11.5, 9.0];
        let want = 100.0 * (4.0 * (0.5 / 3.0) + 2.0 * (2.5 / 5.5) + 3.5 / 7.5) / 7.0;
        let out = one("ultosc", &[i(1), i(2), i(3)], &[&h, &l, &c]);
        assert!(out[2].is_nan());
        assert!(close(out[3], want), "{} {want}", out[3]);
        // The periods are sorted: any order gives the same numbers.
        assert_eq!(bits(&out), bits(&one("ultosc", &[i(3), i(1), i(2)], &[&h, &l, &c])));
        let flat = [30.0; 40];
        assert_eq!(one("ultosc", &[i(7), i(14), i(28)], &[&flat, &flat, &flat])[39], 0.0);
    }

    #[test]
    fn ultosc_window_with_no_range_is_exactly_zero_after_a_wide_bar_leaves() {
        // A wide bar, then flat bars: once it leaves every window the sums are exactly 0, not a
        // compensation residue, so the oscillator is 0 rather than a garbage ratio.
        let mut h = vec![10.0, 50.0];
        let mut l = vec![10.0, 3.3];
        let mut c = vec![10.0, 20.0];
        h.extend([20.0; 30]);
        l.extend([20.0; 30]);
        c.extend([20.0; 30]);
        let out = one("ultosc", &[i(2), i(3), i(4)], &[&h, &l, &c]);
        assert_eq!(out[31], 0.0);
    }

    #[test]
    fn talib_sma_order_and_nan_window() {
        // TA-Lib's SMA adds, reads, then subtracts the value leaving; same numbers as a plain
        // mean here, and a NaN spoils exactly the windows holding it.
        let mut s = TaSma::new(3);
        let out: Vec<f64> =
            [1.0, 2.0, 3.0, NAN, 5.0, 6.0, 7.0, 8.0].iter().map(|&x| s.step(x)).collect();
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(out[2], 2.0);
        assert!(out[3].is_nan() && out[4].is_nan() && out[5].is_nan());
        assert_eq!((out[6], out[7]), (6.0, 7.0));
    }
}
