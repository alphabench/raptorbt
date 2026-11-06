//! Volume studies: TA-Lib's `ad`, `adosc` and `obv`, and raptorbt's own `vwap` and
//! `volume_roc`. `ad`, `obv`, `vwap` and `volume_roc` reproduce `indicators::volume` bit for bit
//! on finite data; `ad`, `adosc` and `obv` follow TA-Lib.
//!
//! A bar with any non-finite input is skipped by the cumulative kernels (output NaN, state held),
//! so a stream with a gap equals the same stream with the bar removed.

use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Legacy, Memory, NanPolicy,
    NanRule, OutKind, ParamDef, ParamKind, ParamVal, Reference, Relation,
};

const MAX_PERIOD: i64 = 1 << 24;

/// One bar's contribution to the Chaikin A/D line: money flow multiplier times volume, zero on a
/// bar with no range. Same operation order as TA-Lib and `indicators::volume::ad_line`.
pub(crate) fn ad_term(h: f64, l: f64, c: f64, v: f64) -> f64 {
    let range = h - l;
    if range > 0.0 {
        (((c - l) - (h - c)) / range) * v
    } else {
        0.0
    }
}

fn finite_or_nan(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NAN
    }
}

/// Chaikin A/D line: running sum of [`ad_term`] from the first bar the kernel sees.
pub struct Ad {
    ad: f64,
}

impl Ad {
    pub fn new() -> Self {
        Self { ad: 0.0 }
    }
}

impl Default for Ad {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel<4, 1> for Ad {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<4>>(&mut self, x: [f64; 4], _h: &H) -> [f64; 1] {
        if !x.iter().all(|v| v.is_finite()) {
            return [f64::NAN];
        }
        self.ad += ad_term(x[0], x[1], x[2], x[3]);
        [finite_or_nan(self.ad)]
    }
}

/// Chaikin oscillator: fast EMA minus slow EMA of the A/D line. As in TA-Lib both EMAs are
/// seeded with the first A/D value (not an SMA) and the first output is bar `max(fast, slow) - 1`.
pub struct Adosc {
    ad: f64,
    fast_k: f64,
    slow_k: f64,
    fast: f64,
    slow: f64,
    seen: usize,
    first: usize,
}

impl Adosc {
    pub fn new(fast: usize, slow: usize) -> Self {
        let (fast, slow) = (fast.max(1), slow.max(1));
        Self {
            ad: 0.0,
            fast_k: 2.0 / (fast as f64 + 1.0),
            slow_k: 2.0 / (slow as f64 + 1.0),
            fast: 0.0,
            slow: 0.0,
            seen: 0,
            first: fast.max(slow),
        }
    }
}

impl Kernel<4, 1> for Adosc {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<4>>(&mut self, x: [f64; 4], _h: &H) -> [f64; 1] {
        if !x.iter().all(|v| v.is_finite()) {
            return [f64::NAN];
        }
        self.ad += ad_term(x[0], x[1], x[2], x[3]);
        if self.seen == 0 {
            self.fast = self.ad;
            self.slow = self.ad;
        } else {
            self.fast = self.fast_k * self.ad + (1.0 - self.fast_k) * self.fast;
            self.slow = self.slow_k * self.ad + (1.0 - self.slow_k) * self.slow;
        }
        self.seen += 1;
        if self.seen < self.first {
            [f64::NAN]
        } else {
            [finite_or_nan(self.fast - self.slow)]
        }
    }
}

/// On-balance volume: the first bar's volume, then plus or minus the volume as the close rises or
/// falls against the last accepted close.
pub struct Obv {
    obv: f64,
    prev_close: Option<f64>,
}

impl Obv {
    pub fn new() -> Self {
        Self { obv: 0.0, prev_close: None }
    }
}

impl Default for Obv {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel<2, 1> for Obv {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        let [close, volume] = x;
        if !(close.is_finite() && volume.is_finite()) {
            return [f64::NAN];
        }
        match self.prev_close {
            None => self.obv = volume,
            Some(pc) => {
                if close > pc {
                    self.obv += volume;
                } else if close < pc {
                    self.obv -= volume;
                }
            }
        }
        self.prev_close = Some(close);
        [finite_or_nan(self.obv)]
    }
}

/// Cumulative volume-weighted average of the typical price (high + low + close) / 3, from the
/// first bar the kernel sees; NaN until the cumulative volume is positive.
pub struct Vwap {
    tp_vol: f64,
    vol: f64,
}

impl Vwap {
    pub fn new() -> Self {
        Self { tp_vol: 0.0, vol: 0.0 }
    }
}

impl Default for Vwap {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel<4, 1> for Vwap {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<4>>(&mut self, x: [f64; 4], _h: &H) -> [f64; 1] {
        if !x.iter().all(|v| v.is_finite()) {
            return [f64::NAN];
        }
        let tp = (x[0] + x[1] + x[2]) / 3.0;
        self.tp_vol += tp * x[3];
        self.vol += x[3];
        if self.vol > 0.0 {
            [finite_or_nan(self.tp_vol / self.vol)]
        } else {
            [f64::NAN]
        }
    }
}

/// Percent change of volume over `p` bars; NaN where the volume `p` bars ago is zero.
pub struct VolumeRoc {
    p: usize,
}

impl VolumeRoc {
    pub fn new(p: usize) -> Self {
        Self { p: p.max(1) }
    }
}

impl Kernel<1, 1> for VolumeRoc {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        self.p
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let old = h.ago(0, self.p);
        if x[0].is_finite() && old.is_finite() && old != 0.0 {
            [finite_or_nan((x[0] - old) / old * 100.0)]
        } else {
            [f64::NAN]
        }
    }
}

const HLCV: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
    InputDef { name: "volume", default_field: Some("volume") },
];
const CLOSE_VOLUME: &[InputDef] = &[
    InputDef { name: "close", default_field: Some("close") },
    InputDef { name: "volume", default_field: Some("volume") },
];
const VOLUME: &[InputDef] = &[InputDef { name: "volume", default_field: Some("volume") }];
const NO_PARAMS: &[ParamDef] = &[];
const PERIOD: &[ParamDef] =
    &[ParamDef { name: "period", kind: ParamKind::Int { min: 1, max: MAX_PERIOD }, default: None }];
const FAST_SLOW: &[ParamDef] = &[
    ParamDef {
        name: "fast",
        kind: ParamKind::Int { min: 2, max: MAX_PERIOD },
        default: Some(ParamVal::Int(3)),
    },
    ParamDef {
        name: "slow",
        kind: ParamKind::Int { min: 2, max: MAX_PERIOD },
        default: Some(ParamVal::Int(10)),
    },
];

fn warmup_zero(_: &[ParamVal]) -> usize {
    0
}

fn warmup_period(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(0)
}

fn warmup_adosc(p: &[ParamVal]) -> usize {
    let fast = int_param(p, 0).unwrap_or(1);
    let slow = int_param(p, 1).unwrap_or(1);
    fast.max(slow).saturating_sub(1)
}

/// A cumulative sum remembers where it started for ever: shifting the start shifts every value
/// by a constant, which never decays. Callers must add to this with `saturating_add`.
fn never_settles(_: &[ParamVal], _: f64) -> usize {
    usize::MAX
}

/// The oscillator is a difference of two EMAs of the same line, so an offset of the line cancels;
/// what remains of the start is the EMA seed, which decays at the slower rate.
fn adosc_settle(p: &[ParamVal], eps: f64) -> usize {
    let period = int_param(p, 0).unwrap_or(1).max(int_param(p, 1).unwrap_or(1)) as f64;
    let alpha = 2.0 / (period + 1.0);
    (eps.ln() / (1.0 - alpha).ln()).ceil().max(0.0) as usize
}

fn make_ad(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    Ok(Box::new(Adapter::new(
        Ad::new(),
        regs::<4>(ins, "ad inputs")?,
        regs::<1>(outs, "ad outputs")?,
    )))
}

fn make_adosc(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Adosc::new(int_param(p, 0)?, int_param(p, 1)?);
    Ok(Box::new(Adapter::new(
        k,
        regs::<4>(ins, "adosc inputs")?,
        regs::<1>(outs, "adosc outputs")?,
    )))
}

fn make_obv(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    Ok(Box::new(Adapter::new(
        Obv::new(),
        regs::<2>(ins, "obv inputs")?,
        regs::<1>(outs, "obv outputs")?,
    )))
}

fn make_vwap(_: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    Ok(Box::new(Adapter::new(
        Vwap::new(),
        regs::<4>(ins, "vwap inputs")?,
        regs::<1>(outs, "vwap outputs")?,
    )))
}

fn make_volume_roc(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = VolumeRoc::new(int_param(p, 0)?);
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "volume_roc inputs")?,
        regs::<1>(outs, "volume_roc outputs")?,
    )))
}

pub const DEFS: &[FnDef] = &[
    FnDef {
        name: "ad",
        family: Family::Volume,
        inputs: HLCV,
        params: NO_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_zero,
        memory: Memory::Infinite { settle: never_settles },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::volume::ad_line",
            tolerance: 0.0,
        }),
        talib_name: Some("AD"),
        doc: "Chaikin accumulation/distribution line: the running sum of ((close - low) - (high - close)) / (high - low) * volume, zero on a bar with no range. Cumulative: the value depends on where the series starts (the first bar the expression sees is the origin), so a number is comparable only across the same start; a bar with a non-finite input is skipped.",
        refs: &["Chaikin, M. (1970s), Accumulation/Distribution"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Measured against TA-Lib 0.6.4 on 500 random-walk bars: largest difference 5.7e-13 on
        // values up to 4.1e3 (1.4e-16 relative); the legacy function is bitwise ours.
        relation: Relation::Exact,
        make: make_ad,
    },
    FnDef {
        name: "adosc",
        family: Family::Volume,
        inputs: HLCV,
        params: FAST_SLOW,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_adosc,
        memory: Memory::Infinite { settle: adosc_settle },
        legacy: None,
        talib_name: Some("ADOSC"),
        doc: "Chaikin A/D oscillator: EMA(ad, fast) - EMA(ad, slow), both EMAs seeded with the first A/D value as TA-Lib does. A constant offset of the A/D line cancels, so the start matters only through the EMA seed, which decays at the slow rate.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_adosc,
    },
    FnDef {
        name: "obv",
        family: Family::Volume,
        inputs: CLOSE_VOLUME,
        params: NO_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_zero,
        memory: Memory::Infinite { settle: never_settles },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::volume::obv",
            tolerance: 0.0,
        }),
        talib_name: Some("OBV"),
        doc: "On-balance volume: the first bar's volume, then plus the volume when the close rises and minus it when the close falls. Cumulative: the value depends on where the series starts, so only changes in it are comparable across starts; a bar with a non-finite input is skipped and the next close is compared with the last accepted one.",
        refs: &["Granville, J. (1963), Granville's New Key to Stock Market Profits"],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant, Invariance::ShiftInvariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Measured against TA-Lib 0.6.4: identical (difference 0.0) on 500 random-walk bars.
        relation: Relation::Exact,
        make: make_obv,
    },
    FnDef {
        name: "vwap",
        family: Family::Volume,
        inputs: HLCV,
        params: NO_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_zero,
        memory: Memory::Infinite { settle: never_settles },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::volume::vwap",
            tolerance: 0.0,
        }),
        talib_name: None,
        doc: "Volume-weighted average of the typical price (high + low + close) / 3, cumulative from the first bar the expression sees and never reset (there is no session input); NaN until the cumulative volume is positive. The value depends on where the series starts.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Hand("vwap_hand_values"),
        relation: Relation::NoLibrary,
        make: make_vwap,
    },
    FnDef {
        name: "volume_roc",
        family: Family::Volume,
        inputs: VOLUME,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::volume::volume_roc",
            tolerance: 0.0,
        }),
        talib_name: None,
        doc: "Percent change of volume over `period` bars: (v - v[period]) / v[period] * 100; NaN where the earlier volume is zero.",
        refs: &[],
        rev: 1,
        invariance: &[],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Hand("volume_roc_hand_values"),
        relation: Relation::NoLibrary,
        make: make_volume_roc,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::volume as legacy;
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

    fn refs(cols: &[Vec<f64>]) -> Vec<&[f64]> {
        cols.iter().map(|c| &c[..]).collect()
    }

    /// Rows (close, up, down, volume) as high/low/close/volume columns with high >= low.
    fn bars(rows: &[(f64, f64, f64, f64)]) -> Vec<Vec<f64>> {
        vec![
            rows.iter().map(|r| r.0 + r.1).collect(),
            rows.iter().map(|r| r.0 - r.2).collect(),
            rows.iter().map(|r| r.0).collect(),
            rows.iter().map(|r| r.3).collect(),
        ]
    }

    fn row() -> impl Strategy<Value = (f64, f64, f64, f64)> {
        (10.0f64..1000.0, 0.0f64..20.0, 0.0f64..20.0, 1.0f64..1e6)
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans() {
        let value = prop_oneof![8 => 1.0f64..1e6, 1 => Just(f64::NAN), 1 => Just(f64::INFINITY)];
        let col = prop::collection::vec(value, 0..200);
        let strategy = (2i64..20, col.clone(), col.clone(), col.clone(), col);
        runner(31)
            .run(&strategy, |(p, a, b, c, d)| {
                let n = [a.len(), b.len(), c.len(), d.len()].into_iter().min().unwrap();
                let cols: [&[f64]; 4] = [&a[..n], &b[..n], &c[..n], &d[..n]];
                let cases: [(&str, Vec<ParamVal>); 5] = [
                    ("ad", vec![]),
                    ("adosc", ints(&[p, p + 3])),
                    ("obv", vec![]),
                    ("vwap", vec![]),
                    ("volume_roc", ints(&[p])),
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
    fn kernels_equal_the_legacy_functions_bitwise_on_finite_data() {
        let strategy = (1i64..30, prop::collection::vec(row(), 0..300));
        runner(32)
            .run(&strategy, |(p, rows)| {
                let b = bars(&rows);
                let (h, l, c, v) = (&b[0], &b[1], &b[2], &b[3]);
                let want_ad = legacy::ad_line(h, l, c, v).unwrap();
                let want_obv = legacy::obv(c, v).unwrap();
                let want_vwap = legacy::vwap(h, l, c, v).unwrap();
                let want_roc = legacy::volume_roc(v, p as usize).unwrap();
                let hlcv = refs(&b);
                let cv: [&[f64]; 2] = [c, v];
                let roc = ints(&[p]);
                for run in [batch, stream] {
                    prop_assert_eq!(bits(&run("ad", &[], &hlcv)[0]), bits(&want_ad));
                    prop_assert_eq!(bits(&run("obv", &[], &cv)[0]), bits(&want_obv));
                    prop_assert_eq!(bits(&run("vwap", &[], &hlcv)[0]), bits(&want_vwap));
                    prop_assert_eq!(bits(&run("volume_roc", &roc, &[v])[0]), bits(&want_roc));
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn warmup_formula_matches_first_non_nan() {
        let rows: Vec<(f64, f64, f64, f64)> =
            (0..40).map(|i| (100.0 + i as f64, 1.0, 1.0, 10.0 + i as f64)).collect();
        let b = bars(&rows);
        let cases: [(&str, Vec<ParamVal>); 7] = [
            ("ad", vec![]),
            ("adosc", ints(&[3, 10])),
            ("adosc", ints(&[12, 5])),
            ("obv", vec![]),
            ("vwap", vec![]),
            ("volume_roc", ints(&[1])),
            ("volume_roc", ints(&[7])),
        ];
        for (name, params) in cases {
            let first = |cols: &[&[f64]]| {
                batch(name, &params, cols)[0].iter().position(|x| !x.is_nan()).unwrap()
            };
            let got = match name {
                "obv" => first(&[&b[2], &b[3]]),
                "volume_roc" => first(&[&b[3]]),
                _ => first(&refs(&b)),
            };
            assert_eq!(got, (lookup(name).unwrap().warmup)(&params), "{name} {params:?}");
        }
    }

    #[test]
    fn ad_hand_values() {
        // Bars (high, low, close, volume): (12, 8, 11, 100), (11, 9, 9, 200), (10, 10, 10, 50).
        // Bar 0: ((11 - 8) - (12 - 11)) / (12 - 8) = 2 / 4 = 0.5, times 100 = 50.
        // Bar 1: ((9 - 9) - (11 - 9)) / (11 - 9) = -2 / 2 = -1, times 200 = -200, running -150.
        // Bar 2: high == low, no range, adds nothing.
        let out = batch(
            "ad",
            &[],
            &[&[12.0, 11.0, 10.0], &[8.0, 9.0, 10.0], &[11.0, 9.0, 10.0], &[100.0, 200.0, 50.0]],
        );
        assert_eq!(out[0], vec![50.0, -150.0, -150.0]);
    }

    #[test]
    fn obv_hand_values_and_start_dependence() {
        // Close 10, 11, 11, 9 with volume 100, 200, 300, 400: 100, +200, unchanged, -400.
        let (c, v) = ([10.0, 11.0, 11.0, 9.0], [100.0, 200.0, 300.0, 400.0]);
        assert_eq!(batch("obv", &[], &[&c, &v])[0], vec![100.0, 300.0, 300.0, -100.0]);
        // Starting at the second bar re-seeds with that bar's volume: 200, unchanged, -400.
        assert_eq!(batch("obv", &[], &[&c[1..], &v[1..]])[0], vec![200.0, 200.0, -200.0]);
    }

    #[test]
    fn adosc_hand_value() {
        // Seven bars with close == high: multiplier 1, volume 8, so the A/D line is 8, 16, ..., 56.
        // fast = 3 (k = 1/2), slow = 7 (k = 1/4), both seeded with 8:
        //   fast: 8, 12, 18, 25, 32.5, 40.25, 48.125
        //   slow: 8, 10, 13.5, 18.125, 23.59375, 29.6953125, 36.271484375
        // Every product is exact in binary; the first output is bar 6: 48.125 - 36.271484375.
        let (h, l, c, v) = ([10.0; 7], [9.0; 7], [10.0; 7], [8.0; 7]);
        let out = batch("adosc", &ints(&[3, 7]), &[&h, &l, &c, &v]);
        assert!(out[0][..6].iter().all(|x| x.is_nan()));
        assert_eq!(out[0][6], 11.853515625);
    }

    #[test]
    fn vwap_hand_values() {
        // Bar 0: typical price (12 + 6 + 9) / 3 = 9, volume 100. Bar 1: (15 + 9 + 12) / 3 = 12,
        // volume 300: (9 * 100 + 12 * 300) / 400 = 4500 / 400 = 11.25.
        let out = batch("vwap", &[], &[&[12.0, 15.0], &[6.0, 9.0], &[9.0, 12.0], &[100.0, 300.0]]);
        assert_eq!(out[0], vec![9.0, 11.25]);
        // No volume yet: NaN, not 0 / 0.
        let zero = batch("vwap", &[], &[&[1.0], &[1.0], &[1.0], &[0.0]]);
        assert!(zero[0][0].is_nan());
    }

    #[test]
    fn volume_roc_hand_values() {
        // Period 2: (300 - 100) / 100 * 100 = 200 and (150 - 150) / 150 * 100 = 0.
        let out = batch("volume_roc", &ints(&[2]), &[&[100.0, 150.0, 300.0, 150.0]]);
        assert!(out[0][0].is_nan() && out[0][1].is_nan());
        assert_eq!(out[0][2], 200.0);
        assert_eq!(out[0][3], 0.0);
        // A zero base is refused rather than divided by.
        let zero = batch("volume_roc", &ints(&[1]), &[&[0.0, 5.0]]);
        assert!(zero[0][1].is_nan());
    }

    #[test]
    fn a_gap_equals_the_bar_removed_for_cumulative_kernels() {
        let cols = [
            vec![12.0, 13.0, 12.5, 14.0, 13.0],
            vec![8.0, 9.0, 9.5, 10.0, 9.0],
            vec![11.0, 10.0, 12.0, 11.0, 12.0],
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
        ];
        let gapped: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| {
                let mut c = c.clone();
                c[2] = f64::NAN;
                c
            })
            .collect();
        let removed: Vec<Vec<f64>> = cols
            .iter()
            .map(|c| c.iter().enumerate().filter(|(i, _)| *i != 2).map(|(_, v)| *v).collect())
            .collect();
        for (name, params) in
            [("ad", vec![]), ("adosc", ints(&[2, 3])), ("obv", vec![]), ("vwap", vec![])]
        {
            let (g, r) = if name == "obv" {
                (
                    batch(name, &params, &[&gapped[2], &gapped[3]]),
                    batch(name, &params, &[&removed[2], &removed[3]]),
                )
            } else {
                (batch(name, &params, &refs(&gapped)), batch(name, &params, &refs(&removed)))
            };
            assert!(g[0][2].is_nan(), "{name}");
            let kept: Vec<f64> = [0, 1, 3, 4].iter().map(|&i| g[0][i]).collect();
            assert_eq!(bits(&kept), bits(&r[0]), "{name}");
        }
    }
}
