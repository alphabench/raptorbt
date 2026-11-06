//! Volatility and price-range studies: TA-Lib's `atr`, `natr` and `trange`, and raptorbt's own
//! `keltner`, `donchian` and `supertrend`.
//!
//! `atr` exists in raptorbt already and differs from TA-Lib, so it carries a trailing `compat`
//! parameter: `native` replicates `indicators::streaming::StreamingAtr` (the true range of the
//! first bar is high - low, seeded with the mean of the first `period` ranges, first value on bar
//! `period - 1`); `talib` replicates TA-Lib (the first bar has no range, the seed is the mean of
//! ranges 1..=period, first value on bar `period`, and the smoothing is
//! `(prev * (period - 1) + tr) / period`). `natr`, `trange` and the rest follow TA-Lib or the
//! legacy function they are pinned to, with no `compat`.
//!
//! The recursive kernels skip a bar with any non-finite input (output NaN, state held, and the
//! previous close stays the last accepted one), so a stream with a gap equals the stream with the
//! bar removed.

use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Legacy, Memory, NanPolicy,
    NanRule, OutKind, ParamDef, ParamKind, ParamVal, Reference, Relation,
};
use crate::quant::window::RollExt;

const MAX_PERIOD: i64 = 1 << 24;

/// True range of a bar against the previous close. Same expression as the legacy
/// `volatility::true_range` and `StreamingAtr`.
pub(crate) fn true_range(h: f64, l: f64, pc: f64) -> f64 {
    (h - l).max((h - pc).abs()).max((l - pc).abs())
}

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

/// Wilder-smoothed true range in either convention. Inputs must be finite.
pub(crate) struct AtrCore {
    p: usize,
    talib: bool,
    alpha: f64,
    prev_close: Option<f64>,
    seed_sum: f64,
    seen: usize,
    seeded: bool,
    value: f64,
}

impl AtrCore {
    pub(crate) fn new(p: usize, talib: bool) -> Self {
        let p = p.max(1);
        Self {
            p,
            talib,
            alpha: 1.0 / p as f64,
            prev_close: None,
            seed_sum: 0.0,
            seen: 0,
            seeded: false,
            value: f64::NAN,
        }
    }

    /// Feed one accepted bar; returns the ATR, NaN until seeded.
    pub(crate) fn push(&mut self, h: f64, l: f64, c: f64) -> f64 {
        let tr = match self.prev_close {
            Some(pc) => true_range(h, l, pc),
            None if self.talib => {
                self.prev_close = Some(c);
                return f64::NAN;
            }
            None => h - l,
        };
        self.prev_close = Some(c);
        if self.seeded {
            self.value = if self.talib {
                (self.value * (self.p - 1) as f64 + tr) / self.p as f64
            } else {
                self.alpha * tr + (1.0 - self.alpha) * self.value
            };
        } else {
            self.seed_sum += tr;
            self.seen += 1;
            if self.seen == self.p {
                self.value = self.seed_sum / self.p as f64;
                self.seeded = true;
            }
        }
        self.value
    }
}

/// Seeded EMA (mean of the first `p` values), the arithmetic of `StreamingEma`. Finite input only.
struct EmaCore {
    p: usize,
    alpha: f64,
    seed_sum: f64,
    seen: usize,
    value: Option<f64>,
}

impl EmaCore {
    fn new(p: usize) -> Self {
        let p = p.max(1);
        Self { p, alpha: 2.0 / (p as f64 + 1.0), seed_sum: 0.0, seen: 0, value: None }
    }

    fn push(&mut self, x: f64) -> f64 {
        match self.value {
            Some(prev) => {
                let v = self.alpha * x + (1.0 - self.alpha) * prev;
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

/// ATR over (high, low, close) with the selected convention.
pub struct Atr {
    core: AtrCore,
}

impl Atr {
    pub fn new(p: usize, talib: bool) -> Self {
        Self { core: AtrCore::new(p, talib) }
    }
}

impl Kernel<3, 1> for Atr {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 1] {
        if !all_finite(&x) {
            return [f64::NAN];
        }
        [finite_or_nan(self.core.push(x[0], x[1], x[2]))]
    }
}

/// 100 * ATR / close, with TA-Lib's ATR.
pub struct Natr {
    core: AtrCore,
}

impl Natr {
    pub fn new(p: usize) -> Self {
        Self { core: AtrCore::new(p, true) }
    }
}

impl Kernel<3, 1> for Natr {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 1] {
        if !all_finite(&x) {
            return [f64::NAN];
        }
        let atr = self.core.push(x[0], x[1], x[2]);
        if x[2] == 0.0 {
            return [f64::NAN];
        }
        [finite_or_nan(atr / x[2] * 100.0)]
    }
}

/// True range against the previous bar's close; NaN on the first bar and after a NaN bar.
pub struct Trange;

impl Kernel<3, 1> for Trange {
    fn heap_bytes(&self) -> usize {
        0
    }

    const CLEAN: bool = true;

    fn run_clean(&mut self, cols: &[&[f64]; 3], outs: &mut [&mut [f64]; 1]) {
        let [h, l, c] = *cols;
        let out = &mut *outs[0];
        if let Some(first) = out.first_mut() {
            *first = f64::NAN;
        }
        for i in 1..out.len() {
            out[i] = finite_or_nan(true_range(h[i], l[i], c[i - 1]));
        }
    }

    fn lookback(&self) -> usize {
        1
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], h: &H) -> [f64; 1] {
        let pc = h.ago(2, 1);
        if !(all_finite(&x) && pc.is_finite()) {
            return [f64::NAN];
        }
        [finite_or_nan(true_range(x[0], x[1], pc))]
    }
}

/// EMA(close) and ATR bands: outputs middle, upper = middle + m * atr, lower = middle - m * atr.
/// All three are NaN until both the EMA and the ATR are seeded.
pub struct Keltner {
    ema: EmaCore,
    atr: AtrCore,
    m: f64,
}

impl Keltner {
    pub fn new(ema_p: usize, atr_p: usize, m: f64) -> Self {
        Self { ema: EmaCore::new(ema_p), atr: AtrCore::new(atr_p, false), m }
    }
}

impl Kernel<3, 3> for Keltner {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 3] {
        if !all_finite(&x) {
            return [f64::NAN; 3];
        }
        let middle = self.ema.push(x[2]);
        let atr = self.atr.push(x[0], x[1], x[2]);
        if middle.is_nan() || atr.is_nan() {
            return [f64::NAN; 3];
        }
        [middle, middle + self.m * atr, middle - self.m * atr].map(finite_or_nan)
    }
}

/// Highest high and lowest low over `p` bars, and their midpoint.
pub struct Donchian {
    hi: RollExt,
    lo: RollExt,
}

impl Donchian {
    pub fn new(p: usize) -> Self {
        Self { hi: RollExt::new(p.max(1), true), lo: RollExt::new(p.max(1), false) }
    }
}

impl Kernel<2, 3> for Donchian {
    fn heap_bytes(&self) -> usize {
        self.hi.heap_bytes() + self.lo.heap_bytes()
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 3] {
        let upper = self.hi.push(x[0]);
        let lower = self.lo.push(x[1]);
        [upper, (upper + lower) / 2.0, lower]
    }
}

/// Supertrend line and direction (1 bullish, -1 bearish), ATR in the native convention.
pub struct Supertrend {
    atr: AtrCore,
    m: f64,
    prev_close: f64,
    /// Final upper and lower band and direction of the last valid bar.
    state: Option<(f64, f64, f64)>,
}

impl Supertrend {
    pub fn new(p: usize, m: f64) -> Self {
        Self { atr: AtrCore::new(p, false), m, prev_close: f64::NAN, state: None }
    }
}

impl Kernel<3, 2> for Supertrend {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 2] {
        let [high, low, close] = x;
        if !all_finite(&x) {
            return [f64::NAN; 2];
        }
        let prev_close = self.prev_close;
        self.prev_close = close;
        let atr = self.atr.push(high, low, close);
        if atr.is_nan() {
            return [f64::NAN; 2];
        }
        let hl2 = (high + low) / 2.0;
        let upper_band = hl2 + self.m * atr;
        let lower_band = hl2 - self.m * atr;
        let (line, dir, fu, fl) = match self.state {
            None => {
                if close <= upper_band {
                    (upper_band, -1.0, upper_band, lower_band)
                } else {
                    (lower_band, 1.0, upper_band, lower_band)
                }
            }
            Some((pfu, pfl, pdir)) => {
                let fu = if prev_close <= pfu { upper_band.min(pfu) } else { upper_band };
                let fl = if prev_close >= pfl { lower_band.max(pfl) } else { lower_band };
                if pdir == 1.0 {
                    if close < fl {
                        (fu, -1.0, fu, fl)
                    } else {
                        (fl, 1.0, fu, fl)
                    }
                } else if close > fu {
                    (fl, 1.0, fu, fl)
                } else {
                    (fu, -1.0, fu, fl)
                }
            }
        };
        self.state = Some((fu, fl, dir));
        [finite_or_nan(line), dir]
    }
}

const HLC: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
];
const HL: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
];
const NO_PARAMS: &[ParamDef] = &[];
const PERIOD_DEF: ParamDef =
    ParamDef { name: "period", kind: ParamKind::Int { min: 1, max: MAX_PERIOD }, default: None };
const PERIOD: &[ParamDef] = &[PERIOD_DEF];
const ATR_PARAMS: &[ParamDef] = &[
    PERIOD_DEF,
    ParamDef {
        name: "compat",
        kind: ParamKind::Enum(&["native", "talib"]),
        default: Some(ParamVal::Enum("native")),
    },
];
const KELTNER_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "ema_period",
        kind: ParamKind::Int { min: 1, max: MAX_PERIOD },
        default: None,
    },
    ParamDef {
        name: "atr_period",
        kind: ParamKind::Int { min: 1, max: MAX_PERIOD },
        default: None,
    },
    ParamDef {
        name: "multiplier",
        kind: ParamKind::Float { min: 0.0, max: 1e9 },
        default: Some(ParamVal::Float(2.0)),
    },
];
const SUPERTREND_PARAMS: &[ParamDef] = &[
    PERIOD_DEF,
    ParamDef {
        name: "multiplier",
        kind: ParamKind::Float { min: 0.0, max: 1e9 },
        default: Some(ParamVal::Float(3.0)),
    },
];

fn is_talib(p: &[ParamVal]) -> bool {
    matches!(p.get(1), Some(ParamVal::Enum("talib")))
}

fn float_param(p: &[ParamVal], i: usize) -> Result<f64, String> {
    match p.get(i) {
        Some(ParamVal::Float(v)) if v.is_finite() => Ok(*v),
        Some(ParamVal::Int(v)) => Ok(*v as f64),
        other => Err(format!("parameter {i} must be a finite number, got {other:?}")),
    }
}

fn warmup_atr(p: &[ParamVal]) -> usize {
    let n = int_param(p, 0).unwrap_or(1);
    if is_talib(p) {
        n
    } else {
        n.saturating_sub(1)
    }
}

fn warmup_natr(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(1)
}

fn warmup_one(_: &[ParamVal]) -> usize {
    1
}

fn warmup_keltner(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(1).max(int_param(p, 1).unwrap_or(1)).saturating_sub(1)
}

fn warmup_window(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(1).saturating_sub(1)
}

/// Bars until a seed difference falls below `eps` under Wilder smoothing, (1 - 1/p)^k < eps.
fn wilder_settle_bars(period: usize, eps: f64) -> usize {
    if period <= 1 {
        return 0;
    }
    let rho = 1.0 - 1.0 / period as f64;
    (eps.ln() / rho.ln()).ceil().max(0.0) as usize
}

fn wilder_settle(p: &[ParamVal], eps: f64) -> usize {
    wilder_settle_bars(int_param(p, 0).unwrap_or(1), eps)
}

/// The slower of the EMA (1 - 2/(p + 1)) and the Wilder ATR (1 - 1/p) decay.
fn keltner_settle(p: &[ParamVal], eps: f64) -> usize {
    let ema_p = int_param(p, 0).unwrap_or(1) as f64;
    let rho = 1.0 - 2.0 / (ema_p + 1.0);
    let ema = if rho <= 0.0 { 0 } else { (eps.ln() / rho.ln()).ceil().max(0.0) as usize };
    ema.max(wilder_settle_bars(int_param(p, 1).unwrap_or(1), eps))
}

/// The bands carry over from bar to bar and the direction flips only on a cross, so the line
/// depends on its whole path: no bar count makes the start irrelevant.
fn never_settles(_: &[ParamVal], _: f64) -> usize {
    usize::MAX
}

fn make_atr(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Atr::new(int_param(p, 0)?, is_talib(p));
    Ok(Box::new(Adapter::new(k, regs::<3>(ins, "atr inputs")?, regs::<1>(outs, "atr outputs")?)))
}

fn make_natr(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Natr::new(int_param(p, 0)?);
    Ok(Box::new(Adapter::new(k, regs::<3>(ins, "natr inputs")?, regs::<1>(outs, "natr outputs")?)))
}

fn make_trange(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    Ok(Box::new(Adapter::new(
        Trange,
        regs::<3>(ins, "trange inputs")?,
        regs::<1>(outs, "trange outputs")?,
    )))
}

fn make_keltner(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Keltner::new(int_param(p, 0)?, int_param(p, 1)?, float_param(p, 2)?);
    Ok(Box::new(Adapter::new(
        k,
        regs::<3>(ins, "keltner inputs")?,
        regs::<3>(outs, "keltner outputs")?,
    )))
}

fn make_donchian(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Donchian::new(int_param(p, 0)?);
    Ok(Box::new(Adapter::new(
        k,
        regs::<2>(ins, "donchian inputs")?,
        regs::<3>(outs, "donchian outputs")?,
    )))
}

fn make_supertrend(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Supertrend::new(int_param(p, 0)?, float_param(p, 1)?);
    Ok(Box::new(Adapter::new(
        k,
        regs::<3>(ins, "supertrend inputs")?,
        regs::<2>(outs, "supertrend outputs")?,
    )))
}

pub const DEFS: &[FnDef] = &[
    FnDef {
        name: "atr",
        family: Family::VolPrice,
        inputs: HLC,
        params: ATR_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_atr,
        memory: Memory::Infinite { settle: wilder_settle },
        legacy: Some(Legacy {
            // f64::max ignores a NaN operand, so the legacy core substitutes the other terms of
            // the true range instead of poisoning; the expression skips the bar.
            nan: NanPolicy::Recover,
            pinned_to: "indicators::streaming::StreamingAtr",
            tolerance: 0.0,
        }),
        talib_name: Some("ATR"),
        doc: "Average true range, Wilder smoothing. compat=\"native\" (default): the first bar's range is high - low, the seed is the mean of the first `period` ranges, first value on bar period - 1. compat=\"talib\": the first bar has no range, the seed is the mean of ranges 1..=period, first value on bar `period`, matching TA-Lib. The two differ by the seed, which decays as (1 - 1/period) per bar.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Native against TA-Lib 0.6.4 (legacy_vs_talib.txt): first valid p - 1 against p, values
        // differ by up to 0.05. The talib path replays Exact.
        relation: Relation::DefinedDifference,
        make: make_atr,
    },
    FnDef {
        name: "natr",
        family: Family::VolPrice,
        inputs: HLC,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_natr,
        memory: Memory::Infinite { settle: wilder_settle },
        legacy: None,
        talib_name: Some("NATR"),
        doc: "Normalised ATR: 100 * ATR / close with TA-Lib's ATR (first value on bar `period`). NaN where the close is zero.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_natr,
    },
    FnDef {
        name: "trange",
        family: Family::VolPrice,
        inputs: HLC,
        params: NO_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_one,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::volatility::true_range",
            tolerance: 0.0,
        }),
        talib_name: Some("TRANGE"),
        doc: "True range: the greatest of high - low, |high - previous close| and |low - previous close|. NaN on the first bar, and on the bar after a NaN bar.",
        refs: &["Wilder, J. W. (1978), New Concepts in Technical Trading Systems"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_trange,
    },
    FnDef {
        name: "keltner",
        family: Family::VolPrice,
        inputs: HLC,
        params: KELTNER_PARAMS,
        outputs: &["middle", "upper", "lower"],
        out_kind: OutKind::Series,
        warmup: warmup_keltner,
        memory: Memory::Infinite { settle: keltner_settle },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::volatility::keltner_channels",
            tolerance: 0.0,
        }),
        talib_name: None,
        doc: "Keltner channel: middle = EMA(close, ema_period), upper and lower = middle +/- multiplier * ATR(atr_period) with the native ATR. All three outputs are NaN until both are seeded, so the middle is later than a bare EMA when atr_period is longer.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Hand("keltner_hand_values"),
        relation: Relation::NoLibrary,
        make: make_keltner,
    },
    FnDef {
        name: "donchian",
        family: Family::VolPrice,
        inputs: HL,
        params: PERIOD,
        outputs: &["upper", "middle", "lower"],
        out_kind: OutKind::Series,
        warmup: warmup_window,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::streaming::StreamingDonchian",
            tolerance: 0.0,
        }),
        talib_name: None,
        doc: "Donchian channel: upper = highest high and lower = lowest low of the last `period` bars, middle = their midpoint (the legacy core has no middle). NaN while a NaN is in the window.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Hand("donchian_hand_values"),
        relation: Relation::NoLibrary,
        make: make_donchian,
    },
    FnDef {
        name: "supertrend",
        family: Family::VolPrice,
        inputs: HLC,
        params: SUPERTREND_PARAMS,
        outputs: &["line", "direction"],
        out_kind: OutKind::Series,
        warmup: warmup_window,
        memory: Memory::Infinite { settle: never_settles },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::trend::supertrend",
            tolerance: 0.0,
        }),
        talib_name: None,
        doc: "Supertrend: bands (high + low) / 2 +/- multiplier * ATR(period) (native ATR) that only tighten while the close stays on their side, and a line that follows the lower band when the direction is 1 (bullish) and the upper band when it is -1 (bearish). The direction flips when the close crosses the active band; it is NaN, not 0, before the first value. Path-dependent: the result depends on where the series starts and never settles.",
        refs: &[],
        rev: 1,
        // `line` is price-like and `direction` is not, so no single identity covers both outputs.
        invariance: &[],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Hand("supertrend_hand_values"),
        relation: Relation::NoLibrary,
        make: make_supertrend,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::streaming::{StreamingAtr, StreamingDonchian};
    use crate::indicators::{trend, volatility};
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::lookup;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    fn runner(seed: u64) -> TestRunner {
        TestRunner::new(Config {
            cases: 300,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(seed),
            ..Config::default()
        })
    }

    fn make(name: &str, params: &[ParamVal]) -> (Box<dyn Node>, usize, usize) {
        let def = lookup(name).unwrap();
        let (i, o) = (def.inputs.len(), def.outputs.len());
        let ins: Vec<usize> = (0..i).collect();
        let outs: Vec<usize> = (i..i + o).collect();
        ((def.make)(params, &ins, &outs).unwrap(), i, o)
    }

    /// Runs `name` over the first `inputs` columns of `cols`.
    fn batch(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<Vec<f64>> {
        let (mut node, i, o) = make(name, params);
        let mut out = vec![vec![0.0; cols[0].len()]; o];
        let mut refs: Vec<&mut [f64]> = out.iter_mut().map(|v| &mut v[..]).collect();
        node.run_batch(&cols[..i], &mut refs);
        out
    }

    fn stream(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<Vec<f64>> {
        let (mut node, i, o) = make(name, params);
        let n = cols[0].len();
        let mut regs = Regs::new(i + o);
        let mut out = vec![vec![0.0; n]; o];
        for t in 0..n {
            for (j, col) in cols.iter().enumerate().take(i) {
                regs.set(j, col[t]);
            }
            node.step(&mut regs);
            for (k, col) in out.iter_mut().enumerate() {
                col[t] = regs.get(i + k);
            }
        }
        out
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|&x| canon_bits(x)).collect()
    }

    fn ints(v: &[i64]) -> Vec<ParamVal> {
        v.iter().map(|&x| ParamVal::Int(x)).collect()
    }

    fn atr_params(p: i64, compat: &'static str) -> Vec<ParamVal> {
        vec![ParamVal::Int(p), ParamVal::Enum(compat)]
    }

    fn refs(cols: &[Vec<f64>]) -> Vec<&[f64]> {
        cols.iter().map(|c| &c[..]).collect()
    }

    /// Rows (close, up, down) as high/low/close columns with high >= low.
    fn bars(rows: &[(f64, f64, f64)]) -> Vec<Vec<f64>> {
        vec![
            rows.iter().map(|r| r.0 + r.1).collect(),
            rows.iter().map(|r| r.0 - r.2).collect(),
            rows.iter().map(|r| r.0).collect(),
        ]
    }

    fn row() -> impl Strategy<Value = (f64, f64, f64)> {
        (10.0f64..1000.0, 0.0f64..20.0, 0.0f64..20.0)
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans() {
        let value = prop_oneof![8 => 1.0f64..1e3, 1 => Just(f64::NAN), 1 => Just(f64::INFINITY)];
        let col = prop::collection::vec(value, 0..200);
        let strategy = (2i64..20, col.clone(), col.clone(), col);
        runner(41)
            .run(&strategy, |(p, a, b, c)| {
                let n = [a.len(), b.len(), c.len()].into_iter().min().unwrap();
                let cols: [&[f64]; 3] = [&a[..n], &b[..n], &c[..n]];
                let cases: [(&str, Vec<ParamVal>); 9] = [
                    ("atr", atr_params(p, "native")),
                    ("atr", atr_params(p, "talib")),
                    ("natr", ints(&[p])),
                    ("trange", vec![]),
                    ("keltner", vec![ParamVal::Int(p), ParamVal::Int(p + 2), ParamVal::Float(1.5)]),
                    ("donchian", ints(&[p])),
                    ("supertrend", vec![ParamVal::Int(p), ParamVal::Float(2.0)]),
                    ("supertrend", vec![ParamVal::Int(1), ParamVal::Float(0.0)]),
                    ("donchian", ints(&[1])),
                ];
                for (name, params) in cases {
                    let (x, y) = (batch(name, &params, &cols), stream(name, &params, &cols));
                    for (xo, yo) in x.iter().zip(&y) {
                        prop_assert_eq!(bits(xo), bits(yo), "{} p={}", name, p);
                    }
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn native_atr_equals_the_streaming_core_bitwise_on_finite_data() {
        let strategy = (1i64..40, prop::collection::vec(row(), 0..300));
        runner(42)
            .run(&strategy, |(p, rows)| {
                let b = bars(&rows);
                let mut core = StreamingAtr::new(p as usize);
                let want: Vec<f64> = (0..rows.len())
                    .map(|i| core.update_bar(b[0][i], b[1][i], b[2][i]).unwrap_or(f64::NAN))
                    .collect();
                let params = atr_params(p, "native");
                prop_assert_eq!(bits(&batch("atr", &params, &refs(&b))[0]), bits(&want));
                prop_assert_eq!(bits(&stream("atr", &params, &refs(&b))[0]), bits(&want));
                // The default compat is the native one.
                prop_assert_eq!(
                    bits(&batch("atr", &atr_params(p, "native"), &refs(&b))[0]),
                    bits(&want)
                );
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn trange_equals_the_legacy_true_range_bitwise() {
        let strategy = prop::collection::vec(row(), 0..300);
        runner(43)
            .run(&strategy, |rows| {
                let b = bars(&rows);
                let want: Vec<f64> = (0..rows.len())
                    .map(|i| {
                        if i == 0 {
                            f64::NAN
                        } else {
                            volatility::true_range(b[0][i], b[1][i], b[2][i - 1])
                        }
                    })
                    .collect();
                prop_assert_eq!(bits(&batch("trange", &[], &refs(&b))[0]), bits(&want));
                prop_assert_eq!(bits(&stream("trange", &[], &refs(&b))[0]), bits(&want));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn keltner_equals_the_legacy_channels_bitwise_where_the_channel_exists() {
        let strategy = (1i64..25, 1i64..25, 0.0f64..4.0, prop::collection::vec(row(), 30..200));
        runner(44)
            .run(&strategy, |(ep, ap, m, rows)| {
                let b = bars(&rows);
                let (mid, up, lo) =
                    volatility::keltner_channels(&b[0], &b[1], &b[2], ep as usize, ap as usize, m)
                        .unwrap();
                let params = vec![ParamVal::Int(ep), ParamVal::Int(ap), ParamVal::Float(m)];
                for out in
                    [batch("keltner", &params, &refs(&b)), stream("keltner", &params, &refs(&b))]
                {
                    // The legacy upper and lower are NaN exactly where the channel is incomplete;
                    // the middle is also NaN there in the expression, but not in the legacy.
                    prop_assert_eq!(bits(&out[1]), bits(&up));
                    prop_assert_eq!(bits(&out[2]), bits(&lo));
                    let full: Vec<f64> = (0..mid.len())
                        .map(|i| if up[i].is_nan() { f64::NAN } else { mid[i] })
                        .collect();
                    prop_assert_eq!(bits(&out[0]), bits(&full));
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn donchian_upper_and_lower_equal_the_streaming_core_bitwise() {
        let strategy = (1i64..30, prop::collection::vec(row(), 0..300));
        runner(45)
            .run(&strategy, |(p, rows)| {
                let b = bars(&rows);
                let mut core = StreamingDonchian::new(p as usize);
                let mut up = vec![];
                let mut lo = vec![];
                for (&h, &lw) in b[0].iter().zip(&b[1]) {
                    let (u, l) = core.update_bar(h, lw).unwrap_or((f64::NAN, f64::NAN));
                    up.push(u);
                    lo.push(l);
                }
                let mid: Vec<f64> = up.iter().zip(&lo).map(|(u, l)| (u + l) / 2.0).collect();
                for out in [
                    batch("donchian", &ints(&[p]), &refs(&b)),
                    stream("donchian", &ints(&[p]), &refs(&b)),
                ] {
                    prop_assert_eq!(bits(&out[0]), bits(&up));
                    prop_assert_eq!(bits(&out[1]), bits(&mid));
                    prop_assert_eq!(bits(&out[2]), bits(&lo));
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn supertrend_equals_the_legacy_function_bitwise_where_it_has_a_value() {
        let strategy = (1i64..25, 0.5f64..5.0, prop::collection::vec(row(), 30..250));
        runner(46)
            .run(&strategy, |(p, m, rows)| {
                let b = bars(&rows);
                let want = trend::supertrend(&b[0], &b[1], &b[2], p as usize, m).unwrap();
                let dir: Vec<f64> = want
                    .direction
                    .iter()
                    .zip(&want.supertrend)
                    .map(|(d, s)| if s.is_nan() { f64::NAN } else { f64::from(*d) })
                    .collect();
                let params = vec![ParamVal::Int(p), ParamVal::Float(m)];
                for out in [
                    batch("supertrend", &params, &refs(&b)),
                    stream("supertrend", &params, &refs(&b)),
                ] {
                    prop_assert_eq!(bits(&out[0]), bits(&want.supertrend));
                    prop_assert_eq!(bits(&out[1]), bits(&dir));
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn warmup_formula_matches_first_non_nan() {
        let rows: Vec<(f64, f64, f64)> =
            (0..60).map(|i| (100.0 + i as f64 + (i as f64).sin(), 1.0, 1.5)).collect();
        let b = bars(&rows);
        let kp = |a, b| vec![ParamVal::Int(a), ParamVal::Int(b), ParamVal::Float(2.0)];
        let cases: Vec<(&str, Vec<ParamVal>)> = vec![
            ("atr", atr_params(1, "native")),
            ("atr", atr_params(14, "native")),
            ("atr", atr_params(1, "talib")),
            ("atr", atr_params(14, "talib")),
            ("natr", ints(&[1])),
            ("natr", ints(&[14])),
            ("trange", vec![]),
            ("keltner", kp(5, 9)),
            ("keltner", kp(12, 3)),
            ("donchian", ints(&[1])),
            ("donchian", ints(&[9])),
            ("supertrend", vec![ParamVal::Int(1), ParamVal::Float(3.0)]),
            ("supertrend", vec![ParamVal::Int(10), ParamVal::Float(3.0)]),
        ];
        for (name, params) in cases {
            let def = lookup(name).unwrap();
            let cols: Vec<&[f64]> = if name == "donchian" { vec![&b[0], &b[1]] } else { refs(&b) };
            let out = batch(name, &params, &cols);
            for o in &out {
                let first = o.iter().position(|x| !x.is_nan()).unwrap();
                assert_eq!(first, (def.warmup)(&params), "{name} {params:?}");
            }
        }
    }

    #[test]
    fn trange_hand_values() {
        // Bars (high, low, close): (10, 8, 9), (11, 9, 10), (13, 12, 12.5).
        // Bar 1: max(11 - 9 = 2, |11 - 9| = 2, |9 - 9| = 0) = 2.
        // Bar 2: max(13 - 12 = 1, |13 - 10| = 3, |12 - 10| = 2) = 3.
        let out =
            batch("trange", &[], &[&[10.0, 11.0, 13.0], &[8.0, 9.0, 12.0], &[9.0, 10.0, 12.5]]);
        assert!(out[0][0].is_nan());
        assert_eq!(out[0][1..], [2.0, 3.0]);
    }

    // Bars (high, low, close): (10, 7, 9), (11, 9, 10), (13, 12, 12.5), (13, 10, 11).
    // True ranges: bar 0 (native only) 10 - 7 = 3; bar 1 max(2, 2, 0) = 2; bar 2 max(1, 3, 2) = 3;
    // bar 3 max(3, |13 - 12.5| = 0.5, |10 - 12.5| = 2.5) = 3.
    const H: [f64; 4] = [10.0, 11.0, 13.0, 13.0];
    const L: [f64; 4] = [7.0, 9.0, 12.0, 10.0];
    const C: [f64; 4] = [9.0, 10.0, 12.5, 11.0];

    #[test]
    fn atr_hand_values_in_both_conventions() {
        // Native, period 2: seed on bar 1 = (3 + 2) / 2 = 2.5; bar 2 = 0.5 * 3 + 0.5 * 2.5 = 2.75;
        // bar 3 = 0.5 * 3 + 0.5 * 2.75 = 2.875.
        let native = batch("atr", &atr_params(2, "native"), &[&H, &L, &C]);
        assert!(native[0][0].is_nan());
        assert_eq!(native[0][1..], [2.5, 2.75, 2.875]);
        // TA-Lib, period 2: no range on bar 0; seed on bar 2 = (2 + 3) / 2 = 2.5;
        // bar 3 = (2.5 * 1 + 3) / 2 = 2.75.
        let talib = batch("atr", &atr_params(2, "talib"), &[&H, &L, &C]);
        assert!(talib[0][0].is_nan() && talib[0][1].is_nan());
        assert_eq!(talib[0][2..], [2.5, 2.75]);
        // natr is TA-Lib's ATR over the close: 2.75 / 11 * 100 = 0.25 * 100 = 25.
        let natr = batch("natr", &ints(&[2]), &[&H, &L, &C]);
        assert_eq!(natr[0][3], 25.0);
    }

    #[test]
    fn keltner_hand_values() {
        // Bars (high, low, close): (10, 7, 9), (11, 9, 10), (12, 10, 11), (13, 11, 12);
        // ema_period 3 (alpha 1/2), atr_period 2, multiplier 2.
        // EMA: seed on bar 2 = (9 + 10 + 11) / 3 = 10; bar 3 = 0.5 * 12 + 0.5 * 10 = 11.
        // Native ATR: ranges 3, 2, max(2, 2, 0) = 2, max(2, 2, 0) = 2; seed on bar 1 = 2.5;
        // bar 2 = 0.5 * 2 + 0.5 * 2.5 = 2.25; bar 3 = 0.5 * 2 + 0.5 * 2.25 = 2.125.
        // Bar 2: 10, 10 + 2 * 2.25 = 14.5, 10 - 4.5 = 5.5. Bar 3: 11, 11 + 4.25 = 15.25, 6.75.
        let out = batch(
            "keltner",
            &[ParamVal::Int(3), ParamVal::Int(2), ParamVal::Float(2.0)],
            &[&[10.0, 11.0, 12.0, 13.0], &[7.0, 9.0, 10.0, 11.0], &[9.0, 10.0, 11.0, 12.0]],
        );
        assert!(out.iter().all(|o| o[0].is_nan() && o[1].is_nan()));
        assert_eq!((out[0][2], out[1][2], out[2][2]), (10.0, 14.5, 5.5));
        assert_eq!((out[0][3], out[1][3], out[2][3]), (11.0, 15.25, 6.75));
    }

    #[test]
    fn donchian_hand_values() {
        // Period 2, highs 3, 5, 4, 2 and lows 1, 2, 0, 1:
        // bar 1: upper 5, lower 1, middle 3; bar 2: 5, 0, 2.5; bar 3: 4, 0, 2.
        let out = batch("donchian", &ints(&[2]), &[&[3.0, 5.0, 4.0, 2.0], &[1.0, 2.0, 0.0, 1.0]]);
        assert!(out.iter().all(|o| o[0].is_nan()));
        assert_eq!(out[0][1..], [5.0, 5.0, 4.0]);
        assert_eq!(out[1][1..], [3.0, 2.5, 2.0]);
        assert_eq!(out[2][1..], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn supertrend_hand_values() {
        // Bars (high, low, close): (10, 7, 9), (11, 9, 10), (12, 10, 11), (13, 11, 12),
        // (16, 13, 15.5); period 2, multiplier 1; native ATR 2.5, 2.25, 2.125, then for bar 4
        // max(3, |16 - 12| = 4, |13 - 12| = 1) = 4: 0.5 * 4 + 0.5 * 2.125 = 3.0625.
        // Bar 1: mid 10, bands 12.5 / 7.5; close 10 <= 12.5 -> line 12.5, direction -1.
        // Bar 2: bands 13.25 / 8.75, final upper min(13.25, 12.5) = 12.5; close 11 <= 12.5 stays -1.
        // Bar 3: bands 14.125 / 9.875, final upper 12.5; close 12 <= 12.5 stays -1, line 12.5.
        // Bar 4: mid 14.5, bands 17.5625 / 11.4375; final upper 12.5, final lower
        //   max(11.4375, 9.875) = 11.4375; close 15.5 > 12.5 flips to +1, line 11.4375.
        let out = batch(
            "supertrend",
            &[ParamVal::Int(2), ParamVal::Float(1.0)],
            &[
                &[10.0, 11.0, 12.0, 13.0, 16.0],
                &[7.0, 9.0, 10.0, 11.0, 13.0],
                &[9.0, 10.0, 11.0, 12.0, 15.5],
            ],
        );
        assert!(out[0][0].is_nan() && out[1][0].is_nan());
        assert_eq!(out[0][1..], [12.5, 12.5, 12.5, 11.4375]);
        assert_eq!(out[1][1..], [-1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn a_gap_equals_the_bar_removed_for_recursive_kernels() {
        let cols = [
            vec![12.0, 13.0, 12.5, 14.0, 13.0, 15.0, 14.0],
            vec![8.0, 9.0, 9.5, 10.0, 9.0, 11.0, 10.0],
            vec![11.0, 10.0, 12.0, 11.0, 12.0, 13.0, 12.0],
        ];
        let gapped: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| {
                let mut c = c.clone();
                c[3] = f64::NAN;
                c
            })
            .collect();
        let removed: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| c.iter().enumerate().filter(|(i, _)| *i != 3).map(|(_, v)| *v).collect())
            .collect();
        let cases: Vec<(&str, Vec<ParamVal>)> = vec![
            ("atr", atr_params(2, "native")),
            ("atr", atr_params(2, "talib")),
            ("natr", ints(&[2])),
            ("keltner", vec![ParamVal::Int(2), ParamVal::Int(2), ParamVal::Float(1.0)]),
            ("supertrend", vec![ParamVal::Int(2), ParamVal::Float(1.0)]),
        ];
        for (name, params) in cases {
            let g = batch(name, &params, &refs(&gapped));
            let r = batch(name, &params, &refs(&removed));
            for (go, ro) in g.iter().zip(&r) {
                assert!(go[3].is_nan(), "{name}");
                let kept: Vec<f64> = [0, 1, 2, 4, 5, 6].iter().map(|&i| go[i]).collect();
                assert_eq!(bits(&kept), bits(ro), "{name}");
            }
        }
    }

    #[test]
    fn settle_bars_cover_the_seed_difference() {
        // (1 - 1/14)^k < 1e-9 needs k = ceil(ln 1e-9 / ln(13/14)) = 280.
        assert_eq!(wilder_settle_bars(14, 1e-9), 280);
        assert_eq!(wilder_settle_bars(1, 1e-9), 0);
        assert_eq!(never_settles(&[], 1e-9), usize::MAX);
    }
}
