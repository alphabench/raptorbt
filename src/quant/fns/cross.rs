//! Cross-sectional functions: one value per name per date, made from all the names of that
//! date and nothing from any other date.
//!
//! They exist only over a panel (`Expr.eval_panel`); the panel evaluator (`eval::panel`) runs
//! them, so the registry rows carry no kernel (`make` refuses). A row is the cross-section of
//! one date; `present` says which names exist on it.
//!
//! # Which names count
//!
//! A name takes part at a date when it is present and its input is finite (and, for the group
//! forms, its group is finite). The rest are excluded: they take no rank, enter no mean, and
//! their output is NaN. `min_names` (default 2) is the fewest participants a cross-section may
//! have: below it every output of that date is NaN, and for the group forms the rule applies to
//! each group on its own.
//!
//! `rank`, `zscore_xs` and `winsorize_xs` run the very row functions of
//! `portfolio::factor_panel` (`rank_panel`, `zscore_panel`, `winsorize_panel`), so the
//! definitions are one: `rank` is the average rank of the value among the participants, ties
//! sharing the mean rank, scaled to [0, 1] as `(rank - 1) / (n - 1)`; `zscore_xs` is
//! `(x - mean) / stdev` with the population stdev and NaN for a cross-section with no spread.
//!
//! `quantile_xs` and `count_xs` are statistics of the date: every present name gets the same
//! value (also a present name whose own input is NaN), so `x > quantile_xs(x, 0.8)` reads
//! naturally. Absent names get NaN.
//!
//! Nothing here looks at another date: the output at a date is a function of that date's inputs.

use crate::portfolio::factor_panel::{rank_row, winsorize_row, zscore_row, RowScratch};
use crate::quant::registry::{
    Domain, Family, FnDef, InputDef, Invariance, Memory, NanRule, OutKind, ParamDef, ParamKind,
    ParamVal, Reference, Relation,
};

/// Which cross-sectional operation a registry row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Rank,
    ZScore,
    Demean,
    Winsorize,
    Quantile,
    Count,
    GroupRank,
    GroupZScore,
    GroupDemean,
}

/// The operation a cross-family row is, by name.
pub fn kind(def: &FnDef) -> Option<Kind> {
    if def.family != Family::Cross {
        return None;
    }
    Some(match def.name {
        "rank" => Kind::Rank,
        "zscore_xs" => Kind::ZScore,
        "demean_xs" => Kind::Demean,
        "winsorize_xs" => Kind::Winsorize,
        "quantile_xs" => Kind::Quantile,
        "count_xs" => Kind::Count,
        "group_rank" => Kind::GroupRank,
        "group_zscore" => Kind::GroupZScore,
        "group_demean" => Kind::GroupDemean,
        _ => return None,
    })
}

/// Working memory of one worker, reused across dates.
#[derive(Debug, Default)]
pub struct Scratch {
    masked: Vec<f64>,
    sub: Vec<f64>,
    sub_out: Vec<f64>,
    idx: Vec<usize>,
    sorted: Vec<f64>,
    row: RowScratch,
}

impl Scratch {
    /// Room for cross-sections of up to `n` names: applying a row then allocates nothing.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            masked: Vec::with_capacity(n),
            sub: Vec::with_capacity(n),
            sub_out: Vec::with_capacity(n),
            idx: Vec::with_capacity(n),
            sorted: Vec::with_capacity(n),
            row: RowScratch::with_capacity(n),
        }
    }
}

fn min_names(params: &[ParamVal], at: usize) -> usize {
    match params.get(at) {
        Some(ParamVal::Int(v)) if *v >= 0 => *v as usize,
        _ => 2,
    }
}

fn float(params: &[ParamVal], at: usize, default: f64) -> f64 {
    match params.get(at) {
        Some(ParamVal::Float(v)) => *v,
        _ => default,
    }
}

/// Mean-removal over the finite values of `row`, summed in index order. NaN below `min`.
fn demean_row(row: &[f64], min: usize, out: &mut [f64]) {
    let n = row.iter().filter(|v| v.is_finite()).count();
    if n < min || n == 0 {
        return;
    }
    let mut sum = 0.0;
    for v in row.iter().filter(|v| v.is_finite()) {
        sum += *v;
    }
    let mean = sum / n as f64;
    for (o, v) in out.iter_mut().zip(row) {
        if v.is_finite() {
            *o = *v - mean;
        }
    }
}

/// Linear-interpolated quantile `q` of a sorted non-empty slice at position `q * (n - 1)`, as
/// the time-series `quantile` does.
fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    let pos = q * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if frac == 0.0 || lo + 1 >= s.len() {
        return s[lo.min(s.len() - 1)];
    }
    let v = s[lo] + (s[lo + 1] - s[lo]) * frac;
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

/// One date. `x` is the input row, `g` the group row for the group forms, `present` which names
/// exist. `out` is overwritten, same length as the rows.
pub fn apply_row(
    kind: Kind,
    params: &[ParamVal],
    x: &[f64],
    g: Option<&[f64]>,
    present: &[bool],
    out: &mut [f64],
    s: &mut Scratch,
) {
    let m = x.len();
    out.fill(f64::NAN);
    let grouped = matches!(kind, Kind::GroupRank | Kind::GroupZScore | Kind::GroupDemean);
    s.masked.clear();
    s.masked.extend((0..m).map(|i| {
        let in_group = !grouped || g.is_some_and(|g| g[i].is_finite());
        if present[i] && x[i].is_finite() && in_group {
            x[i]
        } else {
            f64::NAN
        }
    }));
    match kind {
        Kind::Rank => rank_row(&s.masked, min_names(params, 0), out, &mut s.row),
        Kind::ZScore => zscore_row(&s.masked, min_names(params, 0), out),
        Kind::Demean => demean_row(&s.masked, min_names(params, 0), out),
        Kind::Winsorize => {
            let min = min_names(params, 1);
            if s.masked.iter().filter(|v| v.is_finite()).count() >= min {
                out.copy_from_slice(&s.masked);
                winsorize_row(&s.masked, float(params, 0, 0.05), out, &mut s.row);
            }
        }
        Kind::Quantile | Kind::Count => {
            s.sorted.clear();
            s.sorted.extend(s.masked.iter().copied().filter(|v| v.is_finite()));
            let n = s.sorted.len();
            let value = if kind == Kind::Count {
                n as f64
            } else if n >= min_names(params, 1) && n > 0 {
                s.sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                quantile_sorted(&s.sorted, float(params, 0, 0.5))
            } else {
                f64::NAN
            };
            for (o, p) in out.iter_mut().zip(present) {
                if *p {
                    *o = value;
                }
            }
        }
        Kind::GroupRank | Kind::GroupZScore | Kind::GroupDemean => {
            let Some(g) = g else { return };
            let min = min_names(params, 0);
            s.idx.clear();
            s.idx.extend((0..m).filter(|&i| s.masked[i].is_finite()));
            // Ties on the label break by name index: members of a group stay in name order, so sums
            // run in name order.
            s.idx.sort_unstable_by(|&a, &b| g[a].partial_cmp(&g[b]).unwrap().then(a.cmp(&b)));
            let mut i = 0;
            while i < s.idx.len() {
                let mut j = i;
                while j + 1 < s.idx.len() && g[s.idx[j + 1]] == g[s.idx[i]] {
                    j += 1;
                }
                s.sub.clear();
                s.sub.extend(s.idx[i..=j].iter().map(|&k| s.masked[k]));
                s.sub_out.clear();
                s.sub_out.resize(s.sub.len(), f64::NAN);
                match kind {
                    Kind::GroupRank => rank_row(&s.sub, min, &mut s.sub_out, &mut s.row),
                    Kind::GroupZScore => zscore_row(&s.sub, min, &mut s.sub_out),
                    _ => demean_row(&s.sub, min, &mut s.sub_out),
                }
                for (k, v) in s.idx[i..=j].iter().zip(&s.sub_out) {
                    out[*k] = *v;
                }
                i = j + 1;
            }
        }
    }
}

// ---------------------------------------------------------------- the table

const IN_X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const IN_XG: &[InputDef] = &[
    InputDef { name: "x", default_field: Some("close") },
    InputDef { name: "group", default_field: None },
];

const MAX_NAMES: i64 = 1 << 24;

const MIN_NAMES: ParamDef = ParamDef {
    name: "min_names",
    kind: ParamKind::Int { min: 2, max: MAX_NAMES },
    default: Some(ParamVal::Int(2)),
};
const P_MIN: &[ParamDef] = &[MIN_NAMES];
const P_WINSOR: &[ParamDef] = &[
    ParamDef {
        name: "pct",
        kind: ParamKind::Float { min: 0.0, max: 0.499 },
        default: Some(ParamVal::Float(0.05)),
    },
    MIN_NAMES,
];
const P_QUANTILE: &[ParamDef] = &[
    ParamDef {
        name: "q",
        kind: ParamKind::Float { min: 0.0, max: 1.0 },
        default: Some(ParamVal::Float(0.5)),
    },
    MIN_NAMES,
];

fn warm_none(_: &[ParamVal]) -> usize {
    0
}

fn make_panel_only(
    _: &[ParamVal],
    _: &[usize],
    _: &[usize],
) -> Result<Box<dyn crate::quant::kernel::Node>, String> {
    Err("a cross-sectional function runs over a panel (Expr.eval_panel), not a series".to_string())
}

const INV_SCALE_SHIFT: &[Invariance] = &[Invariance::ScaleInvariant, Invariance::ShiftInvariant];

macro_rules! cross {
    ($name:literal, $inputs:expr, $params:expr, $doc:expr, $inv:expr) => {
        FnDef {
            name: $name,
            family: Family::Cross,
            inputs: $inputs,
            params: $params,
            outputs: &["value"],
            out_kind: OutKind::Series,
            warmup: warm_none,
            memory: Memory::Finite,
            legacy: None,
            talib_name: None,
            doc: $doc,
            refs: &[],
            rev: 1,
            invariance: $inv,
            causal: true,
            nan_rule: NanRule::CrossSection,
            domain: Domain::Any,
            reference: Reference::Hand("fns::cross::tests::hand_values"),
            relation: Relation::NoLibrary,
            make: make_panel_only,
        }
    };
}

pub const DEFS: &[FnDef] = &[
    cross!("rank", IN_X, P_MIN,
        "Cross-sectional rank at each date, in [0, 1]: the lowest name 0, the highest 1, ties sharing the mean rank ((rank - 1) / (n - 1)). Names whose input is NaN or that are absent at the date take no rank and give NaN; below `min_names` participants the whole date is NaN. Needs a panel.",
        INV_SCALE_SHIFT),
    cross!("zscore_xs", IN_X, P_MIN,
        "Cross-sectional z-score at each date: (x - mean) / stdev over the participating names, population stdev. NaN for a date with fewer than `min_names` participants or no spread. Needs a panel.",
        INV_SCALE_SHIFT),
    cross!("demean_xs", IN_X, P_MIN,
        "x minus the mean of the participating names at the same date. NaN below `min_names` participants. Needs a panel.",
        &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant]),
    cross!("winsorize_xs", IN_X, P_WINSOR,
        "Clip each date's cross-section to its k lowest and k highest values, k = floor(n * pct) for n participants (nothing clips while k is 0). `pct` in [0, 0.499]. NaN below `min_names` participants. Needs a panel.",
        &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant]),
    cross!("quantile_xs", IN_X, P_QUANTILE,
        "The cross-sectional quantile `q` of the participating names at each date, linear interpolation at position q * (n - 1), given to every present name (so `x > quantile_xs(x, 0.8)` selects the top fifth). NaN below `min_names` participants. Needs a panel.",
        &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant]),
    cross!("count_xs", IN_X, &[],
        "How many names take part at each date (present with a finite x), given to every present name. Use it to ask for breadth: `count_xs(x) >= 30`. Needs a panel.",
        INV_SCALE_SHIFT),
    cross!("group_rank", IN_XG, P_MIN,
        "`rank` within each group at each date. `group` is a numeric label per name (a name with a NaN label is excluded); `min_names` applies to each group. Needs a panel and a group series, e.g. `col.group(\"sector\")`.",
        INV_SCALE_SHIFT),
    cross!("group_zscore", IN_XG, P_MIN,
        "`zscore_xs` within each group at each date; `min_names` applies to each group. Needs a panel and a group series.",
        INV_SCALE_SHIFT),
    cross!("group_demean", IN_XG, P_MIN,
        "`demean_xs` within each group at each date (group neutralisation); `min_names` applies to each group. Needs a panel and a group series.",
        &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant]),
];

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    fn run(k: Kind, p: &[ParamVal], x: &[f64], g: Option<&[f64]>, present: &[bool]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        apply_row(k, p, x, g, present, &mut out, &mut Scratch::default());
        out
    }

    fn same(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x.is_nan() && y.is_nan()) || x == y, "{a:?} vs {b:?}");
        }
    }

    fn all(n: usize) -> Vec<bool> {
        vec![true; n]
    }

    /// Hand computations, one cross-section each.
    #[test]
    fn hand_values() {
        let min = [ParamVal::Int(2)];
        // rank: values 10, 30, 20, 30 -> sorted 10, 20, 30, 30; ranks 0, 1, 2.5, 2.5 over n - 1 = 3.
        let x = [10.0, 30.0, 20.0, 30.0];
        same(&run(Kind::Rank, &min, &x, None, &all(4)), &[0.0, 2.5 / 3.0, 1.0 / 3.0, 2.5 / 3.0]);
        // An absent name and a NaN name take no rank: 1, 3 over n = 2 -> 0, 1.
        same(
            &run(Kind::Rank, &min, &[1.0, 99.0, NAN, 3.0], None, &[true, false, true, true]),
            &[0.0, NAN, NAN, 1.0],
        );
        // Below min_names the date is NaN.
        same(&run(Kind::Rank, &[ParamVal::Int(3)], &[1.0, 2.0, NAN], None, &all(3)), &[NAN; 3]);
        // One name only: NaN at the default.
        same(&run(Kind::Rank, &min, &[5.0], None, &all(1)), &[NAN]);
        // zscore: 1, 2, 3 -> mean 2, population var 2/3.
        let s = (2.0f64 / 3.0).sqrt();
        same(&run(Kind::ZScore, &min, &[1.0, 2.0, 3.0], None, &all(3)), &[-1.0 / s, 0.0, 1.0 / s]);
        same(&run(Kind::ZScore, &min, &[4.0, 4.0], None, &all(2)), &[NAN, NAN]);
        // demean: 1, 2, 6 -> mean 3.
        same(&run(Kind::Demean, &min, &[1.0, 2.0, 6.0], None, &all(3)), &[-2.0, -1.0, 3.0]);
        // quantile: 1..5, q = 0.25 -> position 1 -> 2; q = 0.3 -> 1.2 -> 2.2; every present name.
        let five = [5.0, 1.0, 4.0, 2.0, 3.0];
        let q = |q: f64, pr: &[bool]| {
            run(Kind::Quantile, &[ParamVal::Float(q), ParamVal::Int(2)], &five, None, pr)
        };
        same(&q(0.25, &all(5)), &[2.0; 5]);
        assert!((q(0.3, &all(5))[0] - 2.2).abs() < 1e-15);
        // The absent name is out: 1, 2, 4, 5 -> position 1.5 -> 2 + (4 - 2) * 0.5 = 3.
        same(&q(0.5, &[true, true, true, true, false]), &[3.0, 3.0, 3.0, 3.0, NAN]);
        // count: NaN input is not counted, but the name still reads the count.
        same(&run(Kind::Count, &[], &[1.0, NAN, 3.0], None, &all(3)), &[2.0; 3]);
        // winsorize: n = 10, pct 0.1 -> k = 1: the lowest and highest clip to their neighbours.
        let ten: Vec<f64> = (1..=10).map(f64::from).collect();
        let p = [ParamVal::Float(0.1), ParamVal::Int(2)];
        let w = run(Kind::Winsorize, &p, &ten, None, &all(10));
        same(&w, &[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0]);
        // groups: {1, 3} and {10, 20, 60}; demean per group (means 2 and 30).
        let g = [1.0, 2.0, 1.0, 2.0, 2.0];
        let v = [1.0, 10.0, 3.0, 20.0, 60.0];
        same(
            &run(Kind::GroupDemean, &min, &v, Some(&g), &all(5)),
            &[-1.0, -20.0, 1.0, -10.0, 30.0],
        );
        // min_names applies per group: the group of one is NaN.
        same(
            &run(Kind::GroupRank, &min, &[1.0, 2.0, 3.0], Some(&[1.0, 1.0, 2.0]), &all(3)),
            &[0.0, 1.0, NAN],
        );
    }

    #[test]
    fn every_row_is_a_cross_family_row_with_an_operation() {
        for d in DEFS {
            assert!(kind(d).is_some(), "{}", d.name);
            assert!((d.make)(&[], &[], &[]).is_err());
        }
    }
}
