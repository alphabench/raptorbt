//! Hilbert transform and cycle functions: `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`,
//! `ht_trendmode`, `ht_trendline` and `mama`. All seven read one shared state machine, [`Ht`],
//! which follows TA-Lib's port of Ehlers' homodyne discriminator bar for bar: a 4-bar weighted
//! smoothing, a detrender and quadrature built from the same four-tap Hilbert filter, odd and
//! even bars kept in separate delay lines, a period estimate with clamps, and (for the phase
//! family) a sum over a window as long as the estimated cycle. The algorithm is a port of
//! TA-Lib's (BSD 3-Clause, see NOTICE); the code is written for this kernel form.
//!
//! NaN policy: a non-finite input outputs NaN and leaves every part of the state alone, the
//! bar counter and the odd/even phase included, so a gap is as if the bar had not existed.
//!
//! TA-Lib's first output convention: the phase family (`ht_dcphase`, `ht_sine`,
//! `ht_trendmode`, `ht_trendline`) lets the smoothing run for 34 bars before the first Hilbert
//! step and outputs from bar 63; the others start the Hilbert steps after 9 and output from bar
//! 32. The two starts leave different state at the first output, so they are two modes.

use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    regs, Domain, Family, FnDef, InputDef, Invariance, Memory, NanRule, OutKind, ParamDef,
    ParamKind, ParamVal, Reference, Relation,
};

const A: f64 = 0.0962;
const B: f64 = 0.5769;
const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
/// Raw-price ring: the cycle window reaches 50 bars back and the smoothing 3.
const PX_LEN: usize = 64;
/// Smoothed-price ring read by the dominant cycle phase sum.
const SMOOTH_LEN: usize = 50;

/// One Hilbert filter with its delay line for one parity of bars.
#[derive(Clone, Copy, Default)]
struct Filter {
    delay: [f64; 3],
    prev: f64,
    prev_input: f64,
}

impl Filter {
    fn step(&mut self, input: f64, idx: usize, adjust: f64) -> f64 {
        let scaled = A * input;
        let mut v = -self.delay[idx];
        self.delay[idx] = scaled;
        v += scaled;
        v -= self.prev;
        self.prev = B * self.prev_input;
        v += self.prev;
        self.prev_input = input;
        v * adjust
    }
}

const DETRENDER: usize = 0;
const Q1: usize = 1;
const JI: usize = 2;
const JQ: usize = 3;

/// What the state machine is being run for: it fixes the first recorded bar and the first
/// output bar, and which of the optional blocks (the phase family, MAMA) are worth computing.
#[derive(Clone, Copy)]
pub(crate) enum HtMode {
    /// `ht_dcperiod`.
    Period,
    /// `ht_phasor`.
    Phasor,
    /// `ht_dcphase`, `ht_sine`, `ht_trendmode`, `ht_trendline`.
    Phase,
    /// `mama`.
    Mama { fast: f64, slow: f64 },
}

impl HtMode {
    /// First bar whose Hilbert step is recorded.
    fn start(self) -> u64 {
        match self {
            HtMode::Phase => 37,
            _ => 12,
        }
    }

    /// Leading NaN outputs on clean data.
    fn lookback(self) -> u64 {
        match self {
            HtMode::Phase => 63,
            _ => 32,
        }
    }
}

/// Everything the seven functions read, NaN before the first output bar.
#[derive(Clone, Copy)]
pub(crate) struct HtOut {
    pub(crate) period: f64,
    pub(crate) phase: f64,
    pub(crate) inphase: f64,
    pub(crate) quadrature: f64,
    pub(crate) sine: f64,
    pub(crate) leadsine: f64,
    pub(crate) trendline: f64,
    pub(crate) trend: f64,
    pub(crate) mama: f64,
    pub(crate) fama: f64,
}

impl HtOut {
    const NAN: HtOut = HtOut {
        period: f64::NAN,
        phase: f64::NAN,
        inphase: f64::NAN,
        quadrature: f64::NAN,
        sine: f64::NAN,
        leadsine: f64::NAN,
        trendline: f64::NAN,
        trend: f64::NAN,
        mama: f64::NAN,
        fama: f64::NAN,
    };
}

/// The shared state machine. `n` counts accepted (finite) bars: it is both the bar index TA-Lib
/// uses and the odd/even phase, so it must not move on a NaN.
pub(crate) struct Ht {
    mode: HtMode,
    n: u64,
    px: [f64; PX_LEN],
    wma_sub: f64,
    wma_sum: f64,
    wma_trail: f64,
    filters: [[Filter; 2]; 4],
    idx: usize,
    i1_even2: f64,
    i1_even3: f64,
    i1_odd2: f64,
    i1_odd3: f64,
    prev_q2: f64,
    prev_i2: f64,
    re: f64,
    im: f64,
    period: f64,
    smooth_period: f64,
    smooth: [f64; SMOOTH_LEN],
    smooth_idx: usize,
    dc_phase: f64,
    sine: f64,
    leadsine: f64,
    trend_hist: [f64; 3],
    days_in_trend: u32,
    prev_phase: f64,
    mama: f64,
    fama: f64,
    /// Detrender and quadrature of the latest recorded bar (read by the tests).
    detrender: f64,
    q1: f64,
}

impl Ht {
    pub(crate) fn new(mode: HtMode) -> Self {
        Self {
            mode,
            n: 0,
            px: [0.0; PX_LEN],
            wma_sub: 0.0,
            wma_sum: 0.0,
            wma_trail: 0.0,
            filters: [[Filter::default(); 2]; 4],
            idx: 0,
            i1_even2: 0.0,
            i1_even3: 0.0,
            i1_odd2: 0.0,
            i1_odd3: 0.0,
            prev_q2: 0.0,
            prev_i2: 0.0,
            re: 0.0,
            im: 0.0,
            period: 0.0,
            smooth_period: 0.0,
            smooth: [0.0; SMOOTH_LEN],
            smooth_idx: 0,
            dc_phase: 0.0,
            sine: 0.0,
            leadsine: 0.0,
            trend_hist: [0.0; 3],
            days_in_trend: 0,
            prev_phase: 0.0,
            mama: 0.0,
            fama: 0.0,
            detrender: 0.0,
            q1: 0.0,
        }
    }

    pub(crate) fn step(&mut self, x: f64) -> HtOut {
        if !x.is_finite() {
            return HtOut::NAN;
        }
        let n = self.n;
        self.n += 1;
        self.px[n as usize % PX_LEN] = x;

        // 4-bar weighted mean, 4 x newest .. 1 x oldest, as running sums.
        match n {
            0 => {
                self.wma_sub = x;
                self.wma_sum = x;
                return HtOut::NAN;
            }
            1 => {
                self.wma_sub += x;
                self.wma_sum += x * 2.0;
                return HtOut::NAN;
            }
            2 => {
                self.wma_sub += x;
                self.wma_sum += x * 3.0;
                return HtOut::NAN;
            }
            _ => {}
        }
        self.wma_sub += x;
        self.wma_sub -= self.wma_trail;
        self.wma_sum += x * 4.0;
        self.wma_trail = self.px[(n - 3) as usize % PX_LEN];
        let smoothed = self.wma_sum * 0.1;
        self.wma_sum -= self.wma_sub;
        if n < self.mode.start() {
            return HtOut::NAN;
        }

        let adjust = 0.075 * self.period + 0.54;
        let even = n.is_multiple_of(2);
        let k = usize::from(!even);
        let idx = self.idx;
        let detrender = self.filters[DETRENDER][k].step(smoothed, idx, adjust);
        let q1 = self.filters[Q1][k].step(detrender, idx, adjust);
        let i1 = if even { self.i1_even3 } else { self.i1_odd3 };
        let ji = self.filters[JI][k].step(i1, idx, adjust);
        let jq = self.filters[JQ][k].step(q1, idx, adjust);
        if even {
            self.idx = (idx + 1) % 3;
            self.i1_odd3 = self.i1_odd2;
            self.i1_odd2 = detrender;
        } else {
            self.i1_even3 = self.i1_even2;
            self.i1_even2 = detrender;
        }
        self.detrender = detrender;
        self.q1 = q1;
        let q2 = 0.2 * (q1 + ji) + 0.8 * self.prev_q2;
        let i2 = 0.2 * (i1 - jq) + 0.8 * self.prev_i2;

        // Homodyne discriminator.
        self.re = 0.2 * (i2 * self.prev_i2 + q2 * self.prev_q2) + 0.8 * self.re;
        self.im = 0.2 * (i2 * self.prev_q2 - q2 * self.prev_i2) + 0.8 * self.im;
        self.prev_q2 = q2;
        self.prev_i2 = i2;
        let before = self.period;
        if self.im != 0.0 && self.re != 0.0 {
            self.period = 360.0 / ((self.im / self.re).atan() * RAD_TO_DEG);
        }
        let hi = 1.5 * before;
        if self.period > hi {
            self.period = hi;
        }
        let lo = 0.67 * before;
        if self.period < lo {
            self.period = lo;
        }
        self.period = self.period.clamp(6.0, 50.0);
        self.period = 0.2 * self.period + 0.8 * before;
        self.smooth_period = 0.33 * self.period + 0.67 * self.smooth_period;

        let phase_family = match self.mode {
            HtMode::Phase => Some(self.phase_block(n, smoothed)),
            _ => None,
        };
        if let HtMode::Mama { fast, slow } = self.mode {
            let angle = if i1 != 0.0 { (q1 / i1).atan() * RAD_TO_DEG } else { 0.0 };
            let mut delta = self.prev_phase - angle;
            self.prev_phase = angle;
            if delta < 1.0 {
                delta = 1.0;
            }
            let alpha = if delta > 1.0 { (fast / delta).max(slow) } else { fast };
            self.mama = alpha * x + (1.0 - alpha) * self.mama;
            let half = 0.5 * alpha;
            self.fama = half * self.mama + (1.0 - half) * self.fama;
        }

        let mut out = HtOut::NAN;
        if n >= self.mode.lookback() {
            out.period = self.smooth_period;
            out.inphase = i1;
            out.quadrature = q1;
            out.mama = self.mama;
            out.fama = self.fama;
            if let Some(((phase, sine, lead), (trendline, trend))) = phase_family {
                out.phase = phase;
                out.sine = sine;
                out.leadsine = lead;
                out.trendline = trendline;
                out.trend = trend;
            }
        }
        out
    }

    /// Dominant cycle phase, sine pair, instantaneous trendline and trend/cycle mode for bar `n`.
    /// Returns ((phase, sine, leadsine), (trendline, trend)); runs on every recorded bar so the
    /// state it carries matches TA-Lib's at the first output.
    fn phase_block(&mut self, n: u64, smoothed: f64) -> ((f64, f64, f64), (f64, f64)) {
        self.smooth[self.smooth_idx] = smoothed;
        let cycle = (self.smooth_period + 0.5) as usize;
        let (mut real, mut imag) = (0.0, 0.0);
        let mut at = self.smooth_idx;
        for i in 0..cycle {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / cycle as f64;
            let v = self.smooth[at];
            real += angle.sin() * v;
            imag += angle.cos() * v;
            at = if at == 0 { SMOOTH_LEN - 1 } else { at - 1 };
        }
        let prev_phase = self.dc_phase;
        let mut phase = self.dc_phase;
        if imag.abs() > 0.0 {
            phase = (real / imag).atan() * RAD_TO_DEG;
        } else if imag.abs() <= 0.01 {
            if real < 0.0 {
                phase -= 90.0;
            } else if real > 0.0 {
                phase += 90.0;
            }
        }
        phase += 90.0;
        phase += 360.0 / self.smooth_period;
        if imag < 0.0 {
            phase += 180.0;
        }
        if phase > 315.0 {
            phase -= 360.0;
        }
        self.dc_phase = phase;
        let (prev_sine, prev_lead) = (self.sine, self.leadsine);
        self.sine = (phase * DEG_TO_RAD).sin();
        self.leadsine = ((phase + 45.0) * DEG_TO_RAD).sin();
        self.smooth_idx = (self.smooth_idx + 1) % SMOOTH_LEN;

        // Trendline: mean of the raw price over one cycle, then a 4-3-2-1 weighting.
        let mut mean = 0.0;
        for i in 0..cycle {
            mean += self.px[(n as usize - i) % PX_LEN];
        }
        if cycle > 0 {
            mean /= cycle as f64;
        }
        let h = self.trend_hist;
        let trendline = (4.0 * mean + 3.0 * h[0] + 2.0 * h[1] + h[2]) / 10.0;
        self.trend_hist = [mean, h[0], h[1]];

        let mut trend = 1.0;
        if (self.sine > self.leadsine && prev_sine <= prev_lead)
            || (self.sine < self.leadsine && prev_sine >= prev_lead)
        {
            self.days_in_trend = 0;
            trend = 0.0;
        }
        self.days_in_trend += 1;
        if f64::from(self.days_in_trend) < 0.5 * self.smooth_period {
            trend = 0.0;
        }
        let rate = phase - prev_phase;
        if self.smooth_period != 0.0
            && rate > 0.67 * 360.0 / self.smooth_period
            && rate < 1.5 * 360.0 / self.smooth_period
        {
            trend = 0.0;
        }
        if trendline != 0.0 && ((smoothed - trendline) / trendline).abs() >= 0.015 {
            trend = 1.0;
        }
        ((phase, self.sine, self.leadsine), (trendline, trend))
    }
}

/// `mama` inside [`crate::quant::fns::overlap::MaCore`]: TA-Lib's `MA` with matype 7 fixes the
/// limits at 0.5 and 0.05 whatever the period.
pub(crate) struct MamaSt {
    ht: Ht,
}

impl MamaSt {
    pub(crate) fn new(fast: f64, slow: f64) -> Self {
        Self { ht: Ht::new(HtMode::Mama { fast, slow }) }
    }

    pub(crate) fn step(&mut self, x: f64) -> f64 {
        self.ht.step(x).mama
    }
}

/// A selection of [`HtOut`] as a kernel with `O` outputs.
pub struct HtKernel<const O: usize> {
    ht: Ht,
    pick: fn(&HtOut) -> [f64; O],
}

impl<const O: usize> Kernel<1, O> for HtKernel<O> {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; O] {
        (self.pick)(&self.ht.step(x[0]))
    }
}

fn make_ht<const O: usize>(
    mode: HtMode,
    pick: fn(&HtOut) -> [f64; O],
    name: &str,
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = HtKernel { ht: Ht::new(mode), pick };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, &format!("{name} inputs"))?,
        regs::<O>(outs, &format!("{name} outputs"))?,
    )))
}

fn make_dcperiod(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ht(HtMode::Period, |o| [o.period], "ht_dcperiod", ins, outs)
}

fn make_dcphase(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ht(HtMode::Phase, |o| [o.phase], "ht_dcphase", ins, outs)
}

fn make_phasor(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ht(HtMode::Phasor, |o| [o.inphase, o.quadrature], "ht_phasor", ins, outs)
}

fn make_sine(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ht(HtMode::Phase, |o| [o.sine, o.leadsine], "ht_sine", ins, outs)
}

fn make_trendmode(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ht(HtMode::Phase, |o| [o.trend], "ht_trendmode", ins, outs)
}

fn make_trendline(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ht(HtMode::Phase, |o| [o.trendline], "ht_trendline", ins, outs)
}

fn float_param(p: &[ParamVal], i: usize) -> Result<f64, String> {
    match p.get(i) {
        Some(ParamVal::Float(v)) => Ok(*v),
        Some(ParamVal::Int(v)) => Ok(*v as f64),
        other => Err(format!("parameter {i} must be a number, got {other:?}")),
    }
}

fn make_mama(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let (fast, slow) = (float_param(p, 0)?, float_param(p, 1)?);
    make_ht(HtMode::Mama { fast, slow }, |o| [o.mama, o.fama], "mama", ins, outs)
}

fn warmup_32(_: &[ParamVal]) -> usize {
    HtMode::Period.lookback() as usize
}

fn warmup_63(_: &[ParamVal]) -> usize {
    HtMode::Phase.lookback() as usize
}

/// The seed's influence passes through 0.8-weighted smoothing at every stage, so it shrinks
/// roughly by `rho` a bar; the period clamps and the phase logic make that a guide, not a bound.
fn settle_after(start: usize, rho: f64, eps: f64) -> usize {
    start + (eps.max(1e-300).ln() / rho.ln()).ceil().max(0.0) as usize
}

fn settle_32(_: &[ParamVal], eps: f64) -> usize {
    settle_after(32, 0.8, eps)
}

fn settle_63(_: &[ParamVal], eps: f64) -> usize {
    settle_after(63, 0.8, eps)
}

/// MAMA's slowest smoothing is FAMA at half the slow limit.
fn settle_mama(p: &[ParamVal], eps: f64) -> usize {
    let slow = float_param(p, 1).unwrap_or(0.05);
    settle_after(32, (1.0 - 0.5 * slow).max(0.8), eps)
}

const INPUT_X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const MAMA_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "fastlimit",
        kind: ParamKind::Float { min: 0.01, max: 0.99 },
        default: Some(ParamVal::Float(0.5)),
    },
    ParamDef {
        name: "slowlimit",
        kind: ParamKind::Float { min: 0.01, max: 0.99 },
        default: Some(ParamVal::Float(0.05)),
    },
];

const REFS: &[&str] = &[
    "Ehlers, J. F. (2001). Rocket Science for Traders. Wiley. Chapters on the Hilbert transform and the homodyne discriminator.",
    "Ehlers, J. F. (2013). Cycle Analytics for Traders. Wiley.",
    "Ehlers, J. F. (2001). MESA Adaptive Moving Averages (MAMA and FAMA). Technical Analysis of Stocks & Commodities.",
    "TA-Lib (BSD 3-Clause), HT_* and MAMA, which this follows value for value.",
];

/// Shared caveat for the doc strings: the family is cheap to compute and weakly evidenced.
macro_rules! caveat {
    () => {
        " Empirical support for cycle indicators is weak: they fit the recent past well, move with noise on short samples, and rarely add edge once costs are charged; test them out of sample before trusting one. The state never fully forgets its start, so values depend on where the series began (see settle)."
    };
}

pub const DEFS: &[FnDef] = &[
    FnDef {
        name: "ht_dcperiod",
        family: Family::Hilbert,
        inputs: INPUT_X,
        params: &[],
        outputs: &["real"],
        out_kind: OutKind::Series,
        warmup: warmup_32,
        memory: Memory::Infinite { settle: settle_32 },
        legacy: None,
        talib_name: Some("HT_DCPERIOD"),
        doc: concat!(
            "Ehlers' dominant cycle period in bars, estimated with a Hilbert-transform homodyne discriminator on a 4-bar weighted mean of the series; clamped to 6..50 bars and smoothed. It says how long the current oscillation seems to be, not that one exists.",
            caveat!()
        ),
        refs: REFS,
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_dcperiod,
    },
    FnDef {
        name: "ht_dcphase",
        family: Family::Hilbert,
        inputs: INPUT_X,
        params: &[],
        outputs: &["real"],
        out_kind: OutKind::Series,
        warmup: warmup_63,
        memory: Memory::Infinite { settle: settle_63 },
        legacy: None,
        talib_name: Some("HT_DCPHASE"),
        doc: concat!(
            "Dominant cycle phase in degrees (about -45 to 315): where in its cycle the smoothed price is, from a Fourier sum over one estimated cycle length of the smoothed series.",
            caveat!()
        ),
        refs: REFS,
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_dcphase,
    },
    FnDef {
        name: "ht_phasor",
        family: Family::Hilbert,
        inputs: INPUT_X,
        params: &[],
        outputs: &["inphase", "quadrature"],
        out_kind: OutKind::Series,
        warmup: warmup_32,
        memory: Memory::Infinite { settle: settle_32 },
        legacy: None,
        talib_name: Some("HT_PHASOR"),
        doc: concat!(
            "The in-phase and quadrature components of the detrended, smoothed series: a rotating vector whose angle is the cycle phase and whose length is the cycle amplitude, in the units of the input.",
            caveat!()
        ),
        refs: REFS,
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_phasor,
    },
    FnDef {
        name: "ht_sine",
        family: Family::Hilbert,
        inputs: INPUT_X,
        params: &[],
        outputs: &["sine", "leadsine"],
        out_kind: OutKind::Series,
        warmup: warmup_63,
        memory: Memory::Infinite { settle: settle_63 },
        legacy: None,
        talib_name: Some("HT_SINE"),
        doc: concat!(
            "Ehlers' sine wave: the sine of the dominant cycle phase and of the phase plus 45 degrees (the lead). Crossings of the two are the classic cycle-turn signal; in a trend the two stay close together and the signal is meaningless.",
            caveat!()
        ),
        refs: REFS,
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_sine,
    },
    FnDef {
        name: "ht_trendmode",
        family: Family::Hilbert,
        inputs: INPUT_X,
        params: &[],
        outputs: &["trend"],
        out_kind: OutKind::Bool,
        warmup: warmup_63,
        memory: Memory::Infinite { settle: settle_63 },
        legacy: None,
        talib_name: Some("HT_TRENDMODE"),
        doc: concat!(
            "1 when the series is judged to be trending and 0 when it is in a cycle (TA-Lib's integer output, stored as a boolean). Cycle mode holds for half a cycle after a sine/leadsine crossing and while the phase advances at about the cycle rate; a price more than 1.5% from the instantaneous trendline always counts as a trend.",
            caveat!()
        ),
        refs: REFS,
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_trendmode,
    },
    FnDef {
        name: "ht_trendline",
        family: Family::Hilbert,
        inputs: INPUT_X,
        params: &[],
        outputs: &["real"],
        out_kind: OutKind::Series,
        warmup: warmup_63,
        memory: Memory::Infinite { settle: settle_63 },
        legacy: None,
        talib_name: Some("HT_TRENDLINE"),
        doc: concat!(
            "Ehlers' instantaneous trendline: the mean of the price over one estimated dominant cycle, then a 4-3-2-1 weighted smoothing. It lags by about a quarter cycle and its length changes as the cycle estimate moves.",
            caveat!()
        ),
        refs: REFS,
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_trendline,
    },
    FnDef {
        name: "mama",
        family: Family::Hilbert,
        inputs: INPUT_X,
        params: MAMA_PARAMS,
        outputs: &["mama", "fama"],
        out_kind: OutKind::Series,
        warmup: warmup_32,
        memory: Memory::Infinite { settle: settle_mama },
        legacy: None,
        talib_name: Some("MAMA"),
        doc: concat!(
            "MESA adaptive moving average and its follower (FAMA). The smoothing constant is the fast limit divided by the rate of change of the cycle phase, floored at the slow limit, so the average hugs price when the phase moves fast and slows when it does not. `ma(x, p, 7)` is this with limits 0.5 and 0.05.",
            caveat!()
        ),
        refs: REFS,
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_mama,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::lookup;

    fn node(name: &str, params: &[ParamVal], n_out: usize) -> Box<dyn Node> {
        let outs: Vec<usize> = (1..1 + n_out).collect();
        (lookup(name).unwrap().make)(params, &[0], &outs).unwrap()
    }

    fn batch(name: &str, params: &[ParamVal], x: &[f64], n_out: usize) -> Vec<Vec<f64>> {
        let mut outs = vec![vec![0.0; x.len()]; n_out];
        let mut refs: Vec<&mut [f64]> = outs.iter_mut().map(|o| &mut o[..]).collect();
        node(name, params, n_out).run_batch(&[x], &mut refs);
        outs
    }

    fn stream(name: &str, params: &[ParamVal], x: &[f64], n_out: usize) -> Vec<Vec<f64>> {
        let mut nd = node(name, params, n_out);
        let mut regs = Regs::new(1 + n_out);
        let mut outs = vec![Vec::new(); n_out];
        for &v in x {
            regs.set(0, v);
            nd.step(&mut regs);
            for (o, out) in outs.iter_mut().enumerate() {
                out.push(regs.get(1 + o));
            }
        }
        outs
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|&x| canon_bits(x)).collect()
    }

    fn defaults(name: &str) -> Vec<ParamVal> {
        lookup(name).unwrap().params.iter().map(|p| p.default.unwrap()).collect()
    }

    /// (name, outputs)
    const ALL: [(&str, usize); 7] = [
        ("ht_dcperiod", 1),
        ("ht_dcphase", 1),
        ("ht_phasor", 2),
        ("ht_sine", 2),
        ("ht_trendmode", 1),
        ("ht_trendline", 1),
        ("mama", 2),
    ];

    /// A cycle riding a drift, the same series the TA-Lib values below were taken on.
    fn series(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + 10.0 * (i as f64 * 0.3).sin() + 0.05 * i as f64).collect()
    }

    /// A rougher series that exercises the clamps and the trend/cycle switch.
    fn noisy(n: usize) -> Vec<f64> {
        let mut s = 12345u64;
        let mut v = 100.0;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
                v += u * 2.0 + 0.8 * (i as f64 * 0.21).sin();
                v
            })
            .collect()
    }

    #[test]
    fn first_hilbert_step_by_hand() {
        // A constant 10: the 4-bar mean is 10 (weights 4+3+2+1 over 10). At the first recorded
        // bar the period is 0, so the filter's scale is 0.075 * 0 + 0.54 = 0.54 and its delay
        // lines are empty: detrender = 0.0962 * 10 * 0.54 = 0.519480.
        // The quadrature filter then sees 0.51948 with the same scale and empty lines:
        // 0.0962 * 0.51948 * 0.54 = 0.026987...
        let mut ht = Ht::new(HtMode::Period);
        for _ in 0..13 {
            ht.step(10.0);
        }
        assert!((ht.detrender - 0.0962 * 10.0 * 0.54).abs() < 1e-12, "{}", ht.detrender);
        assert!((ht.q1 - 0.0962 * 0.0962 * 10.0 * 0.54 * 0.54).abs() < 1e-12, "{}", ht.q1);
    }

    #[test]
    fn constants_have_known_limits() {
        // The period of a constant is set by rounding noise in Re and Im, not by a clean
        // fixed point; 31.091268948999193 is TA-Lib 0.6.4's HT_DCPERIOD on a constant 100 at bar
        // 399 (measured). The trendline is the price; MAMA's delta phase is floored at 1 so
        // alpha = fast = 0.5 and both averages meet the price.
        let x = vec![100.0; 400];
        let p = &batch("ht_dcperiod", &[], &x, 1)[0];
        assert!((p[399] - 31.091268948999193).abs() < 1e-9, "{}", p[399]);
        let tl = &batch("ht_trendline", &[], &x, 1)[0];
        assert!((tl[399] - 100.0).abs() < 1e-9, "{}", tl[399]);
        let m = batch("mama", &defaults("mama"), &x, 2);
        assert!((m[0][399] - 100.0).abs() < 1e-9 && (m[1][399] - 100.0).abs() < 1e-9);
        // The phasor of a constant is zero.
        let ph = batch("ht_phasor", &[], &x, 2);
        assert!(ph[0][399].abs() < 1e-12 && ph[1][399].abs() < 1e-12);
    }

    #[test]
    fn matches_talib_values() {
        // Values produced by TA-Lib 0.6.4 on `series(200)`, bars 80 and 150.
        let x = series(200);
        let check = |name: &str, o: usize, want: (f64, f64)| {
            let n_out = ALL.iter().find(|a| a.0 == name).unwrap().1;
            let b = &batch(name, &defaults(name), &x, n_out)[o];
            for (got, want) in [(b[80], want.0), (b[150], want.1)] {
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "{name}: {got} vs {want}"
                );
            }
        };
        check("ht_dcperiod", 0, (21.029093051260965, 20.923706278970787));
        check("ht_dcphase", 0, (292.37417831422096, 59.66652051061842));
        check("ht_phasor", 0, (-9.692888492825759, 4.873855056561842));
        check("ht_phasor", 1, (-0.9645992768552898, 9.049252681299143));
        check("ht_sine", 0, (-0.9247176780114114, 0.8631005939502521));
        check("ht_sine", 1, (-0.38471134993928585, 0.9674158653363502));
        check("ht_trendline", 0, (103.14595314266494, 106.96943833875346));
        check("ht_trendmode", 0, (1.0, 1.0));
        check("mama", 0, (99.92119252409543, 109.48050538802075));
        check("mama", 1, (101.09219058943924, 106.31919199092863));
    }

    #[test]
    fn warmup_formula_matches_first_non_nan() {
        let x = series(150);
        for (name, n_out) in ALL {
            let params = defaults(name);
            for out in batch(name, &params, &x, n_out) {
                let first = out.iter().position(|v| !v.is_nan()).unwrap();
                assert_eq!(first, (lookup(name).unwrap().warmup)(&params), "{name}");
            }
        }
    }

    #[test]
    fn batch_and_streaming_are_bit_identical() {
        let mut x = noisy(400);
        x[150] = f64::NAN;
        x[151] = f64::INFINITY;
        x[300] = f64::NAN;
        for (name, n_out) in ALL {
            let params = defaults(name);
            let (b, s) = (batch(name, &params, &x, n_out), stream(name, &params, &x, n_out));
            for o in 0..n_out {
                assert_eq!(bits(&b[o]), bits(&s[o]), "{name} output {o}");
            }
        }
    }

    #[test]
    fn a_nan_gap_is_as_if_the_bars_were_absent() {
        // The bar counter and the odd/even phase do not move on a NaN, so removing the NaN
        // bars from the input must give the same values on the remaining bars, bit for bit.
        // An odd-length gap makes a wrongly advanced parity visible.
        let clean = noisy(300);
        for gap in [3usize, 4] {
            let mut gapped = clean[..120].to_vec();
            gapped.extend(std::iter::repeat_n(f64::NAN, gap));
            gapped.extend_from_slice(&clean[120..]);
            for (name, n_out) in ALL {
                let params = defaults(name);
                let a = batch(name, &params, &clean, n_out);
                let g = batch(name, &params, &gapped, n_out);
                for o in 0..n_out {
                    assert!(g[o][120..120 + gap].iter().all(|v| v.is_nan()), "{name}");
                    let mut kept = g[o][..120].to_vec();
                    kept.extend_from_slice(&g[o][120 + gap..]);
                    assert_eq!(bits(&kept), bits(&a[o]), "{name} output {o} gap {gap}");
                    assert!(g[o][299 + gap].is_finite(), "{name} did not resume");
                }
            }
        }
    }

    #[test]
    fn a_nan_inside_the_warmup_only_delays_the_start() {
        let clean = noisy(200);
        let mut gapped = clean[..5].to_vec();
        gapped.push(f64::NAN);
        gapped.extend_from_slice(&clean[5..]);
        let a = &batch("ht_dcperiod", &[], &clean, 1)[0];
        let g = &batch("ht_dcperiod", &[], &gapped, 1)[0];
        assert!(g[..33].iter().all(|v| v.is_nan()));
        assert_eq!(bits(&g[33..]), bits(&a[32..]));
    }

    #[test]
    fn trend_mode_is_zero_or_one_and_both_occur() {
        // TA-Lib gives both values on this series.
        let x = noisy(500);
        let t = &batch("ht_trendmode", &[], &x, 1)[0];
        assert!(t[63..].iter().all(|v| *v == 0.0 || *v == 1.0));
        assert!(t[63..].contains(&1.0) && t[63..].contains(&0.0));
    }
}
