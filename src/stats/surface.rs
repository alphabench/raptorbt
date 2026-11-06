//! How a metric moves across a parameter grid: whether the best cell sits on a plateau of
//! similar cells or on a spike. Pure arithmetic on the grid the caller ran; nothing is re-run.
//!
//! Neighbours of a cell differ by one step along exactly one parameter (the steps are the sorted
//! distinct values of that parameter). A higher metric is better. Conventions the numbers keep:
//! the best cell is the first maximum in input order; the robust pick may be a different cell
//! from the best and need not be adjacent to it; a best cell with no neighbour has plateau share
//! 0, which reads as a spike.

use std::collections::{BTreeMap, HashMap};

use crate::math::fsum::fsum;

use super::error::StatsError;

pub const MAX_PARAMS: usize = 3;
pub const MAX_CELLS: usize = 2000;
pub const MIN_CELLS: usize = 9;
pub const MIN_AXIS_VALUES: usize = 2;
/// A neighbour within 80% of the best (of the grid's range) counts as holding the plateau.
pub const PLATEAU_FRACTION: f64 = 0.8;
pub const PLATEAU_SHARE: f64 = 0.7;
pub const SPIKE_SHARE: f64 = 0.3;
pub const SMOOTH_MIN: f64 = 0.8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Plateau,
    Spike,
    Mixed,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Plateau => "plateau",
            Verdict::Spike => "spike",
            Verdict::Mixed => "mixed",
        }
    }
}

/// A cell of the grid. `params` are in the order of the parameter names.
#[derive(Debug, Clone, PartialEq)]
pub struct CellRow {
    pub params: Vec<f64>,
    pub metric: f64,
    /// `None` for a cell with no neighbour.
    pub neighbour_mean: Option<f64>,
    pub neighbours: usize,
    pub lift_over_neighbours: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BestCell {
    pub params: Vec<f64>,
    pub metric: f64,
    pub neighbour_mean: Option<f64>,
    pub neighbours: usize,
    pub plateau_share: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustPick {
    pub params: Vec<f64>,
    pub metric: f64,
    pub neighbour_mean: f64,
}

/// The heat map of a two-parameter surface: `values[i][j]` is the cell at the `i`-th value of
/// the first parameter and the `j`-th of the second, `None` where it was not run.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub values: Vec<Vec<Option<f64>>>,
}

/// The result of [`sensitivity_surface`], full precision. `axes[i]` are the sorted distinct
/// values of parameter `i`; `cells` are in lexicographic order of their axis positions.
#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    pub axes: Vec<Vec<f64>>,
    pub n_cells: usize,
    pub grid_cells: usize,
    pub coverage: f64,
    pub metric_min: f64,
    pub metric_max: f64,
    pub smoothness: f64,
    pub best: BestCell,
    pub robust_pick: RobustPick,
    pub verdict: Verdict,
    pub cells: Vec<CellRow>,
    pub matrix: Option<Matrix>,
}

type Key = Vec<usize>;

fn neighbours(key: &Key, grid: &HashMap<Key, f64>) -> Vec<Key> {
    let mut out = Vec::new();
    for d in 0..key.len() {
        for step in [-1isize, 1] {
            let Some(pos) = key[d].checked_add_signed(step) else { continue };
            let mut nb = key.clone();
            nb[d] = pos;
            if grid.contains_key(&nb) {
                out.push(nb);
            }
        }
    }
    out
}

fn mean_of(values: impl Iterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = values.collect();
    (!v.is_empty()).then(|| fsum(v.iter().copied()) / v.len() as f64)
}

/// `cells` are (parameter values in `names` order, metric).
pub fn sensitivity_surface<N: AsRef<str>>(
    names: &[N],
    cells: &[(Vec<f64>, f64)],
) -> Result<Surface, StatsError> {
    let k = names.len();
    let mut distinct: Vec<&str> = names.iter().map(|n| n.as_ref()).collect();
    distinct.sort_unstable();
    distinct.dedup();
    if !(1..=MAX_PARAMS).contains(&k) || distinct.len() != k {
        return Err(StatsError::Invalid(format!(
            "give 1 to {MAX_PARAMS} distinct parameter names"
        )));
    }
    if !(MIN_CELLS..=MAX_CELLS).contains(&cells.len()) {
        return Err(StatsError::Invalid(format!(
            "{} cells, a surface needs {MIN_CELLS} to {MAX_CELLS}",
            cells.len()
        )));
    }
    for (vals, metric) in cells {
        if vals.len() != k || !metric.is_finite() || vals.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Invalid(
                "a cell lacks a parameter or holds a non-finite number".into(),
            ));
        }
    }
    let mut axes: Vec<Vec<f64>> = Vec::with_capacity(k);
    for i in 0..k {
        let mut a: Vec<f64> = cells.iter().map(|(v, _)| v[i]).collect();
        a.sort_by(f64::total_cmp);
        a.dedup_by(|x, y| *x == *y);
        if a.len() < MIN_AXIS_VALUES {
            return Err(StatsError::Invalid(format!(
                "parameter '{}' has one value: nothing to vary",
                names[i].as_ref()
            )));
        }
        axes.push(a);
    }
    let mut grid: HashMap<Key, f64> = HashMap::with_capacity(cells.len());
    let mut order: Vec<Key> = Vec::with_capacity(cells.len());
    for (vals, metric) in cells {
        let key: Key = (0..k)
            .map(|i| axes[i].iter().position(|a| *a == vals[i]).unwrap_or(usize::MAX))
            .collect();
        if grid.insert(key.clone(), *metric).is_some() {
            let shown: Vec<String> =
                names.iter().zip(vals).map(|(n, v)| format!("'{}': {v:?}", n.as_ref())).collect();
            return Err(StatsError::Invalid(format!(
                "duplicate cell {{{}}}",
                shown.join(", ")
            )));
        }
        order.push(key);
    }
    let full: usize = axes.iter().map(Vec::len).product();
    let coverage = grid.len() as f64 / full as f64;

    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for m in grid.values() {
        lo = lo.min(*m);
        hi = hi.max(*m);
    }
    let spread = hi - lo;
    if spread == 0.0 {
        return Err(StatsError::Invalid("the metric is the same in every cell".into()));
    }
    let mut diffs = Vec::new();
    for a in &order {
        for b in neighbours(a, &grid) {
            if *a < b {
                diffs.push((grid[a] - grid[&b]).abs());
            }
        }
    }
    if diffs.is_empty() {
        return Err(StatsError::Invalid(
            "no two cells are neighbours: the grid has holes everywhere".into(),
        ));
    }
    let roughness = fsum(diffs.iter().copied()) / diffs.len() as f64 / spread;
    let smoothness = 1.0 - roughness;

    let nb_of: HashMap<&Key, Vec<Key>> = order.iter().map(|c| (c, neighbours(c, &grid))).collect();
    let nb_mean = |c: &Key| mean_of(nb_of[c].iter().map(|n| grid[n]));

    let mut best = &order[0];
    for c in &order {
        if grid[c] > grid[best] {
            best = c;
        }
    }
    let best_val = grid[best];
    let best_nbs = &nb_of[best];
    // within a fifth of the grid's range below the best: scale- and sign-free
    let floor = best_val - (1.0 - PLATEAU_FRACTION) * spread;
    let share = if best_nbs.is_empty() {
        0.0
    } else {
        best_nbs.iter().filter(|n| grid[*n] >= floor).count() as f64 / best_nbs.len() as f64
    };

    let mut robust: Option<(&Key, f64, f64)> = None;
    for c in &order {
        let Some(nm) = nb_mean(c) else { continue };
        let deg = nb_of[c].len() as f64;
        let score = (grid[c] + nm * deg) / (1.0 + deg);
        if robust.is_none_or(|(_, s, _)| score > s) {
            robust = Some((c, score, nm));
        }
    }
    let (robust, _, robust_nm) = robust.expect("a neighbouring pair exists, so some cell has one");

    let verdict = if share >= PLATEAU_SHARE && smoothness >= SMOOTH_MIN {
        Verdict::Plateau
    } else if share <= SPIKE_SHARE {
        Verdict::Spike
    } else {
        Verdict::Mixed
    };

    let params_of = |c: &Key| -> Vec<f64> { (0..k).map(|i| axes[i][c[i]]).collect() };
    let sorted: BTreeMap<&Key, f64> = grid.iter().map(|(c, m)| (c, *m)).collect();
    let rows: Vec<CellRow> = sorted
        .iter()
        .map(|(c, m)| {
            let nm = nb_mean(c);
            CellRow {
                params: params_of(c),
                metric: *m,
                neighbour_mean: nm,
                neighbours: nb_of[*c].len(),
                lift_over_neighbours: nm.map(|x| m - x),
            }
        })
        .collect();
    let matrix = (k == 2).then(|| Matrix {
        values: (0..axes[0].len())
            .map(|i| (0..axes[1].len()).map(|j| grid.get(&vec![i, j]).copied()).collect())
            .collect(),
    });
    Ok(Surface {
        n_cells: grid.len(),
        grid_cells: full,
        coverage,
        metric_min: lo,
        metric_max: hi,
        smoothness,
        best: BestCell {
            params: params_of(best),
            metric: best_val,
            neighbour_mean: if best_nbs.is_empty() { None } else { nb_mean(best) },
            neighbours: best_nbs.len(),
            plateau_share: share,
        },
        robust_pick: RobustPick {
            params: params_of(robust),
            metric: grid[robust],
            neighbour_mean: robust_nm,
        },
        verdict,
        cells: rows,
        matrix,
        axes,
    })
}
