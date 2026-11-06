//! Overlap studies. `sma` and `ema` replicate the streaming cores in
//! `indicators::streaming` operation for operation, so digests and live agree.

use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Legacy, Memory, NanPolicy,
    NanRule, OutKind, ParamDef, ParamKind, ParamVal, Reference, Relation,
};

const MAX_PERIOD: i64 = 1 << 24;

/// Same arithmetic as `StreamingSma`: add the new value, then subtract the one leaving; a
/// non-finite value is counted, not summed.
pub struct Sma {
    p: usize,
    sum: f64,
    bad: usize,
    n: u64,
}

impl Sma {
    pub fn new(p: usize) -> Self {
        Self { p: p.max(1), sum: 0.0, bad: 0, n: 0 }
    }
}

impl Kernel<1, 1> for Sma {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let x = x[0];
        self.n += 1;
        if x.is_finite() {
            self.sum += x;
        } else {
            self.bad += 1;
        }
        if self.n > self.p as u64 {
            let old = h.ago(0, self.p);
            if old.is_finite() {
                self.sum -= old;
            } else {
                self.bad -= 1;
            }
        }
        if self.n < self.p as u64 || self.bad > 0 {
            [f64::NAN]
        } else {
            [self.sum / self.p as f64]
        }
    }
}

/// Same arithmetic as `StreamingEma`, SMA-seeded. Differs on non-finite input by the expression
/// NaN policy: a NaN neither seeds nor advances the state (the legacy core seeds NaN forever).
pub struct Ema {
    p: usize,
    alpha: f64,
    seeded: Option<f64>,
    seed_sum: f64,
    seen: usize,
}

impl Ema {
    pub fn new(p: usize) -> Self {
        let p = p.max(1);
        Self { p, alpha: 2.0 / (p as f64 + 1.0), seeded: None, seed_sum: 0.0, seen: 0 }
    }
}

impl Kernel<1, 1> for Ema {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        let x = x[0];
        if !x.is_finite() {
            return [f64::NAN];
        }
        match self.seeded {
            Some(prev) => {
                let v = self.alpha * x + (1.0 - self.alpha) * prev;
                self.seeded = Some(v);
                [v]
            }
            None => {
                self.seed_sum += x;
                self.seen += 1;
                if self.seen == self.p {
                    let v = self.seed_sum / self.p as f64;
                    self.seeded = Some(v);
                    [v]
                } else {
                    [f64::NAN]
                }
            }
        }
    }
}

const INPUT_X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const PERIOD: &[ParamDef] =
    &[ParamDef { name: "period", kind: ParamKind::Int { min: 1, max: MAX_PERIOD }, default: None }];

fn warmup_period(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| v.saturating_sub(1))
}

fn make_sma(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Sma::new(int_param(p, 0)?);
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "sma inputs")?, regs::<1>(outs, "sma outputs")?)))
}

fn make_ema(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Ema::new(int_param(p, 0)?);
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "ema inputs")?, regs::<1>(outs, "ema outputs")?)))
}

fn ema_settle(p: &[ParamVal], eps: f64) -> usize {
    // Seed influence decays as (1 - alpha)^k.
    let period = int_param(p, 0).unwrap_or(1) as f64;
    let alpha = 2.0 / (period + 1.0);
    (eps.ln() / (1.0 - alpha).ln()).ceil().max(0.0) as usize
}

pub const DEFS: &[FnDef] = &[
    FnDef {
        name: "sma",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::streaming::StreamingSma",
            tolerance: 0.0,
        }),
        talib_name: Some("SMA"),
        doc: "Simple moving average of the last `period` values.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Measured against TA-Lib 0.6.4 (legacy_vs_talib.txt): same first valid bar, largest
        // difference 1.4e-13 on prices near 100, i.e. ~1e-15 relative.
        relation: Relation::Exact,
        make: make_sma,
    },
    FnDef {
        name: "ema",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Infinite { settle: ema_settle },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::streaming::StreamingEma",
            tolerance: 0.0,
        }),
        talib_name: Some("EMA"),
        doc: "Exponential moving average, alpha = 2/(period+1), seeded with the mean of the first `period` values.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // SMA-seeded like TA-Lib's classic EMA: same first valid bar and, measured, largest
        // difference 5.3e-13 on prices near 100 at periods 14 to 200, so no seed gap to decay.
        relation: Relation::Exact,
        make: make_ema,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::streaming::{StreamingEma, StreamingIndicator, StreamingSma};
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

    fn node(name: &str, period: i64) -> Box<dyn Node> {
        (lookup(name).unwrap().make)(&[ParamVal::Int(period)], &[0], &[1]).unwrap()
    }

    fn batch(name: &str, period: i64, data: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; data.len()];
        node(name, period).run_batch(&[data], &mut [&mut out[..]]);
        out
    }

    fn stream(name: &str, period: i64, data: &[f64]) -> Vec<f64> {
        let mut n = node(name, period);
        let mut regs = Regs::new(2);
        data.iter()
            .map(|&x| {
                regs.set(0, x);
                n.step(&mut regs);
                regs.get(1)
            })
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|&x| canon_bits(x)).collect()
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans() {
        let value = prop_oneof![8 => -1e6f64..1e6, 1 => Just(f64::NAN), 1 => Just(f64::INFINITY)];
        let strategy = (1i64..40, prop::collection::vec(value, 0..300));
        runner(11)
            .run(&strategy, |(p, data)| {
                for name in ["sma", "ema"] {
                    prop_assert_eq!(
                        bits(&batch(name, p, &data)),
                        bits(&stream(name, p, &data)),
                        "{} p={}",
                        name,
                        p
                    );
                }
                Ok(())
            })
            .unwrap();
    }

    fn legacy(core: &mut dyn StreamingIndicator, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| core.update(x).unwrap_or(f64::NAN)).collect()
    }

    #[test]
    fn sma_and_ema_equal_the_streaming_cores_bitwise_on_finite_data() {
        let strategy = (1i64..40, prop::collection::vec(-1e6f64..1e6, 0..300));
        runner(12)
            .run(&strategy, |(p, data)| {
                let sma = legacy(&mut StreamingSma::new(p as usize), &data);
                let ema = legacy(&mut StreamingEma::new(p as usize), &data);
                prop_assert_eq!(bits(&batch("sma", p, &data)), bits(&sma));
                prop_assert_eq!(bits(&stream("sma", p, &data)), bits(&sma));
                prop_assert_eq!(bits(&batch("ema", p, &data)), bits(&ema));
                prop_assert_eq!(bits(&stream("ema", p, &data)), bits(&ema));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn warmup_formula_matches_first_non_nan() {
        let data: Vec<f64> = (1..=30).map(f64::from).collect();
        for name in ["sma", "ema"] {
            for p in [1i64, 2, 5, 14] {
                let first = batch(name, p, &data).iter().position(|v| !v.is_nan()).unwrap();
                let def = lookup(name).unwrap();
                assert_eq!(first, (def.warmup)(&[ParamVal::Int(p)]), "{name} p={p}");
            }
        }
    }

    #[test]
    fn sma_hand_value() {
        // (1 + 2 + 3) / 3 = 2 and (2 + 3 + 4) / 3 = 3.
        let out = batch("sma", 3, &[1.0, 2.0, 3.0, 4.0]);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!((out[2], out[3]), (2.0, 3.0));
    }

    #[test]
    fn windowed_recovers_after_the_nan_leaves() {
        let out = batch("sma", 3, &[1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, 8.0]);
        // The window holds the NaN at indices 3, 4, 5; index 6 is (5 + 6 + 7) / 3.
        assert!(out[3].is_nan() && out[4].is_nan() && out[5].is_nan());
        assert_eq!(out[6], 6.0);
        assert_eq!(out[7], 7.0);
    }

    #[test]
    fn recursive_holds_state_on_nan_without_poisoning() {
        // Leading NaNs (an upstream warm-up) do not disturb the seed.
        let nan = f64::NAN;
        let clean = batch("ema", 3, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let lead = batch("ema", 3, &[nan, nan, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(bits(&lead[2..]), bits(&clean));
        assert!(lead[0].is_nan() && lead[1].is_nan());
        // A NaN mid-stream outputs NaN, and the next value continues from the held state:
        // seed (1 + 2 + 3) / 3 = 2, alpha = 0.5, then 0.5 * 4 + 0.5 * 2 = 3.
        let gap = batch("ema", 3, &[1.0, 2.0, 3.0, nan, 4.0]);
        assert_eq!(gap[2], 2.0);
        assert!(gap[3].is_nan());
        assert_eq!(gap[4], 3.0);
    }
}
