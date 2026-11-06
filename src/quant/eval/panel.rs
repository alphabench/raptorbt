//! Panel evaluation: one expression over many names at once.
//!
//! A panel is row-major `n_dates x n_assets`. The checked DAG is cut into phases that alternate:
//! a time-series phase (every node with no cross-sectional ancestor still to run) and a
//! cross-section phase (the `cross` family calls that are ready). Time-series phases run per
//! name, in parallel by column, through the ordinary batch evaluator; cross-section phases run
//! per date, in parallel by row. Every column and every row is computed on its own, so the
//! numbers do not depend on the number of threads.
//!
//! # Absent bars
//!
//! A name need not have a bar at every date (a later listing, a halt, a delisting). The input
//! says so with a presence mask. In a time-series phase a name's column is *compacted*: only its
//! present bars are fed to the kernels, in order, and the results are put back at those dates;
//! an absent bar does not step a kernel, so a name listed late counts its warm-up from its own
//! first bar, and its output is NaN at the dates it has no bar. A present bar whose value is NaN
//! does step the kernel (as in a series). In a cross-section phase an absent name is excluded.
//!
//! Without a mask, a cell is absent when every field the expression reads is NaN there; a mask
//! is the way to say otherwise (a present bar with NaN values, or a bar that exists with no
//! price yet).
//!
//! Each column's results equal what a stream fed the name's present bars gives, bit for bit, and
//! each date's cross-section is the plain function of that date's row (see `fns::cross`).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rayon::prelude::*;

use super::{eval, ClockCtx, EvalError, Inputs, Plan};
use crate::quant::fns::cross::{self, Kind, Scratch};
use crate::quant::lang::{Clock, Dag, Field, NodeId, Op, Program, Shape, Ty};
use crate::quant::registry::{Family, ParamVal};

type Key = (NodeId, u8);

/// Where a cross-section phase reads a panel from.
#[derive(Debug, Clone)]
enum Src {
    Field(Field),
    User(String),
    Prior(Key),
}

/// Name of a boundary handle in a phase's sub-program: the panel of an earlier phase.
fn boundary_name(id: NodeId) -> String {
    format!("#{id}")
}

fn parse_boundary(name: &str) -> Option<Key> {
    name.strip_prefix('#')?.parse().ok().map(|id| (id, 0))
}

#[derive(Debug)]
struct TsPhase {
    plan: Plan,
    /// The panels this phase produces, in the plan's output order.
    exports: Vec<Key>,
    fields: Vec<Field>,
    /// The plan's handles, in plan order.
    handles: Vec<Src>,
}

#[derive(Debug)]
struct CrossNode {
    key: Key,
    kind: Kind,
    params: Vec<ParamVal>,
    x: Src,
    group: Option<Src>,
}

#[derive(Debug)]
enum Phase {
    Ts(TsPhase),
    Cross(Vec<CrossNode>),
}

/// An executable panel program. Holds no run state; cheap to share across threads.
#[derive(Debug)]
pub struct PanelPlan {
    phases: Vec<Phase>,
    finals: Vec<Key>,
    fields: Vec<Field>,
    handles: Vec<String>,
    has_at: bool,
    /// The last phase that reads each panel; finals live to the end.
    last_use: BTreeMap<Key, usize>,
}

fn node_inputs(op: &Op) -> Vec<NodeId> {
    match op {
        Op::Const(_) | Op::Field(_) | Op::Handle(_) | Op::At { .. } => Vec::new(),
        Op::Unary(_, x) => vec![*x],
        Op::Binary(_, a, b) => vec![*a, *b],
        Op::Call { inputs, .. } => inputs.clone(),
        Op::Out { call, .. } => vec![*call],
    }
}

/// A node of the per-column sub-program has a column type, never a panel.
fn column_ty(t: &Ty) -> Ty {
    Ty { shape: if t.shape == Shape::Panel { Shape::Series } else { t.shape }, ..t.clone() }
}

struct Cutter<'a> {
    prog: &'a Program,
    phase: &'a [usize],
    member: &'a [bool],
    at: usize,
    dag: Dag,
    map: HashMap<NodeId, NodeId>,
}

impl Cutter<'_> {
    fn operand(&mut self, x: NodeId) -> NodeId {
        if let Some(&m) = self.map.get(&x) {
            return m;
        }
        let node = self.prog.dag.node(x);
        // Leaves and constants are copied wherever they are read; a computed node of an earlier
        // phase arrives as a boundary handle.
        if !self.member[x] || self.phase[x] == self.at {
            return self.copy(x);
        }
        let id = self.dag.handle(&boundary_name(x), column_ty(&node.ty), node.warm);
        self.map.insert(x, id);
        id
    }

    fn copy(&mut self, x: NodeId) -> NodeId {
        if let Some(&m) = self.map.get(&x) {
            return m;
        }
        let node = self.prog.dag.node(x);
        let ty = column_ty(&node.ty);
        let id = match &node.op {
            Op::Const(v) => self.dag.constant(*v),
            Op::Field(f) => self.dag.field(*f),
            Op::Handle(name) => self.dag.handle(name, ty, node.warm),
            Op::Unary(op, a) => {
                let a = self.operand(*a);
                self.dag.unary(*op, a, ty)
            }
            Op::Binary(op, a, b) => {
                let (a, b) = (self.operand(*a), self.operand(*b));
                self.dag.binary(*op, a, b, ty)
            }
            Op::Call { def, params, inputs } => {
                let ins: Vec<NodeId> = inputs.iter().map(|i| self.operand(*i)).collect();
                self.dag.call(def, params.clone(), ins, ty)
            }
            Op::Out { call, index } => {
                let c = self.operand(*call);
                self.dag.out(c, *index)
            }
            Op::At { spec, body } => self.dag.at(*spec, body.clone(), ty),
        };
        self.map.insert(x, id);
        id
    }
}

impl PanelPlan {
    /// Plan any checked program: one without a cross-sectional call is a single time-series
    /// phase (every name run on its own).
    pub fn new(program: &Program) -> Result<PanelPlan, EvalError> {
        let nodes = program.dag.nodes();
        let n = nodes.len();
        let is_cross =
            |i: NodeId| matches!(&nodes[i].op, Op::Call { def, .. } if def.family == Family::Cross);
        let leaf = |i: NodeId| matches!(nodes[i].op, Op::Field(_) | Op::Handle(_));
        if nodes[program.root].ty.shape == Shape::Scalar {
            return Err(EvalError::Unsupported(
                "a constant has no panel: the expression must read a field".to_string(),
            ));
        }
        if nodes.iter().any(|n| n.ty.clock != Clock::Base) {
            return Err(EvalError::Unsupported("a node on another clock".to_string()));
        }

        // Phase 2L - 1 runs the cross-sectional calls with L of them on their longest path, phase
        // 2L the time-series nodes below L such calls.
        let mut depth = vec![0usize; n];
        let mut phase = vec![0usize; n];
        for i in 0..n {
            let below = node_inputs(&nodes[i].op).iter().map(|x| depth[*x]).max().unwrap_or(0);
            if is_cross(i) {
                depth[i] = below + 1;
                phase[i] = 2 * depth[i] - 1;
            } else {
                depth[i] = below;
                phase[i] = 2 * below;
            }
        }
        // Constants and leaves are copied to where they are read (a leaf that is the whole
        // program is a member of phase 0, so something produces it).
        let member: Vec<bool> = (0..n)
            .map(|i| nodes[i].ty.shape != Shape::Scalar && (!leaf(i) || i == program.root))
            .collect();

        let finals: Vec<Key> = program.outputs();
        let mut needed: BTreeSet<Key> = finals.iter().copied().collect();
        for u in 0..n {
            if !member[u] {
                continue;
            }
            for x in node_inputs(&nodes[u].op) {
                if member[x] && phase[x] != phase[u] {
                    needed.insert((x, 0));
                }
            }
        }

        let last = (0..n).filter(|i| member[*i]).map(|i| phase[i]).max().unwrap_or(0);
        let mut phases = Vec::new();
        for p in 0..=last {
            let members: Vec<NodeId> = (0..n).filter(|i| member[*i] && phase[*i] == p).collect();
            if members.is_empty() {
                continue;
            }
            if p % 2 == 1 {
                let src = |x: NodeId| match &nodes[x].op {
                    Op::Field(f) => Src::Field(*f),
                    Op::Handle(h) => Src::User(h.clone()),
                    _ => Src::Prior((x, 0)),
                };
                let mut group = Vec::new();
                for &i in &members {
                    let Op::Call { def, params, inputs } = &nodes[i].op else { continue };
                    let Some(kind) = cross::kind(def) else { continue };
                    group.push(CrossNode {
                        key: (i, 0),
                        kind,
                        params: params.clone(),
                        x: src(inputs[0]),
                        group: inputs.get(1).map(|g| src(*g)),
                    });
                }
                phases.push(Phase::Cross(group));
            } else {
                let mut cut = Cutter {
                    prog: program,
                    phase: &phase,
                    member: &member,
                    at: p,
                    dag: Dag::new(),
                    map: HashMap::new(),
                };
                for &i in &members {
                    cut.copy(i);
                }
                let exports: Vec<Key> = needed
                    .iter()
                    .copied()
                    .filter(|(id, _)| member[*id] && phase[*id] == p)
                    .collect();
                let outs: Vec<(NodeId, u8)> =
                    exports.iter().map(|&(id, i)| (cut.map[&id], i)).collect();
                let root = outs.last().map_or(0, |o| o.0);
                let sub =
                    Program { dag: cut.dag, root, canonical: String::new(), hash: String::new() };
                let plan = Plan::with_outputs(&sub, &outs)?;
                let mut fields = plan.fields();
                fields.sort_by_key(|f| *f as usize);
                fields.dedup();
                let handles = plan
                    .handles()
                    .iter()
                    .map(|h| match parse_boundary(h) {
                        Some(k) => Src::Prior(k),
                        None => Src::User(h.clone()),
                    })
                    .collect();
                phases.push(Phase::Ts(TsPhase { plan, exports, fields, handles }));
            }
        }

        let mut last_use: BTreeMap<Key, usize> = BTreeMap::new();
        let mut fields: Vec<Field> = Vec::new();
        let mut handles: BTreeSet<String> = BTreeSet::new();
        let mut note = |s: &Src, at: usize, last_use: &mut BTreeMap<Key, usize>| match s {
            Src::Prior(k) => {
                last_use.insert(*k, at);
            }
            Src::Field(f) => fields.push(*f),
            Src::User(h) => {
                handles.insert(h.clone());
            }
        };
        for (at, ph) in phases.iter().enumerate() {
            match ph {
                Phase::Ts(t) => {
                    t.fields.iter().for_each(|f| note(&Src::Field(*f), at, &mut last_use));
                    t.handles.iter().for_each(|h| note(h, at, &mut last_use));
                }
                Phase::Cross(c) => {
                    for cn in c {
                        note(&cn.x, at, &mut last_use);
                        if let Some(g) = &cn.group {
                            note(g, at, &mut last_use);
                        }
                    }
                }
            }
        }
        for k in &finals {
            last_use.insert(*k, usize::MAX);
        }
        fields.sort_by_key(|f| *f as usize);
        fields.dedup();
        Ok(PanelPlan {
            phases,
            finals,
            fields,
            handles: handles.into_iter().collect(),
            has_at: program.has_at(),
            last_use,
        })
    }

    /// The fields the program reads.
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// The names of the supplied series it reads besides fields (group labels).
    pub fn handles(&self) -> &[String] {
        &self.handles
    }

    pub fn has_at(&self) -> bool {
        self.has_at
    }

    /// Panels returned: one per output.
    pub fn n_outputs(&self) -> usize {
        self.finals.len()
    }

    /// Number of phases (time-series and cross-section alternate).
    pub fn n_phases(&self) -> usize {
        self.phases.len()
    }
}

/// Named panels for evaluation. Every panel is row-major `n_dates x n_assets`.
#[derive(Debug, Clone)]
pub struct PanelInputs<'a> {
    n_dates: usize,
    n_assets: usize,
    fields: [Option<&'a [f64]>; 5],
    handles: BTreeMap<String, &'a [f64]>,
    present: Option<&'a [bool]>,
    timestamps: Option<&'a [i64]>,
    clock: Option<ClockCtx>,
}

impl<'a> PanelInputs<'a> {
    pub fn new(n_dates: usize, n_assets: usize) -> Self {
        Self {
            n_dates,
            n_assets,
            fields: [None; 5],
            handles: BTreeMap::new(),
            present: None,
            timestamps: None,
            clock: None,
        }
    }

    pub fn field(mut self, f: Field, panel: &'a [f64]) -> Self {
        self.fields[f as usize] = Some(panel);
        self
    }

    pub fn handle(mut self, name: &str, panel: &'a [f64]) -> Self {
        self.handles.insert(name.to_string(), panel);
        self
    }

    /// Which cells have a bar. Without it a cell is absent when every field the program reads
    /// is NaN there.
    pub fn present(mut self, mask: &'a [bool]) -> Self {
        self.present = Some(mask);
        self
    }

    /// One timestamp (nanoseconds) per date, shared by every name; read by `at(...)` only.
    pub fn timestamps(mut self, ts: &'a [i64]) -> Self {
        self.timestamps = Some(ts);
        self
    }

    pub fn clock(mut self, clock: ClockCtx) -> Self {
        self.clock = Some(clock);
        self
    }
}

struct Ctx<'a> {
    n: usize,
    m: usize,
    fields: [Option<&'a [f64]>; 5],
    handles: &'a BTreeMap<String, &'a [f64]>,
    present: &'a [bool],
    timestamps: Option<&'a [i64]>,
    clock: Option<&'a ClockCtx>,
}

fn panel_of<'a>(
    s: &Src,
    ctx: &'a Ctx<'a>,
    store: &'a BTreeMap<Key, Vec<f64>>,
) -> Result<&'a [f64], EvalError> {
    match s {
        Src::Field(f) => ctx.fields[*f as usize].ok_or(EvalError::MissingField(f.name())),
        Src::User(h) => {
            ctx.handles.get(h).copied().ok_or_else(|| EvalError::MissingHandle(h.clone()))
        }
        Src::Prior(k) => Ok(store[k].as_slice()),
    }
}

fn run_ts(
    ph: &TsPhase,
    ctx: &Ctx,
    store: &BTreeMap<Key, Vec<f64>>,
) -> Result<Vec<Vec<f64>>, EvalError> {
    let (n, m) = (ctx.n, ctx.m);
    let fields: Vec<(Field, &[f64])> = ph
        .fields
        .iter()
        .map(|f| panel_of(&Src::Field(*f), ctx, store).map(|p| (*f, p)))
        .collect::<Result<_, _>>()?;
    let handles: Vec<(String, &[f64])> = ph
        .plan
        .handles()
        .iter()
        .zip(&ph.handles)
        .map(|(name, s)| panel_of(s, ctx, store).map(|p| (name.clone(), p)))
        .collect::<Result<_, _>>()?;
    let use_at = ph.plan.has_at();

    let per_column: Vec<Result<Vec<Vec<f64>>, EvalError>> = (0..m)
        .into_par_iter()
        .map(|a| {
            // The name's own bars, in date order: an absent bar is simply not there.
            let rows: Vec<usize> = (0..n).filter(|&d| ctx.present[d * m + a]).collect();
            let gather = |p: &[f64]| -> Vec<f64> { rows.iter().map(|&d| p[d * m + a]).collect() };
            let f_cols: Vec<Vec<f64>> = fields.iter().map(|(_, p)| gather(p)).collect();
            let h_cols: Vec<Vec<f64>> = handles.iter().map(|(_, p)| gather(p)).collect();
            let ts: Vec<i64> = match (use_at, ctx.timestamps) {
                (true, Some(t)) => rows.iter().map(|&d| t[d]).collect(),
                _ => Vec::new(),
            };
            let mut inputs = Inputs::new();
            for ((f, _), c) in fields.iter().zip(&f_cols) {
                inputs = inputs.field(*f, c);
            }
            for ((name, _), c) in handles.iter().zip(&h_cols) {
                inputs = inputs.handle(name, c);
            }
            if use_at {
                inputs = inputs.timestamps(&ts);
                if let Some(c) = ctx.clock {
                    inputs = inputs.clock(c.clone());
                }
            }
            let compact = eval(&ph.plan, &inputs)?;
            Ok(compact
                .into_iter()
                .map(|c| {
                    let mut full = vec![f64::NAN; n];
                    for (v, &d) in c.into_iter().zip(&rows) {
                        full[d] = v;
                    }
                    full
                })
                .collect())
        })
        .collect();
    let mut cols = Vec::with_capacity(m);
    for r in per_column {
        cols.push(r?);
    }

    let mut panels: Vec<Vec<f64>> = vec![vec![f64::NAN; n * m]; ph.exports.len()];
    if n > 0 && m > 0 {
        for (k, panel) in panels.iter_mut().enumerate() {
            panel.par_chunks_mut(m).enumerate().for_each(|(d, row)| {
                for (a, cell) in row.iter_mut().enumerate() {
                    *cell = cols[a][k][d];
                }
            });
        }
    }
    Ok(panels)
}

fn run_cross(
    nodes: &[CrossNode],
    ctx: &Ctx,
    store: &mut BTreeMap<Key, Vec<f64>>,
) -> Result<(), EvalError> {
    let (n, m) = (ctx.n, ctx.m);
    for cn in nodes {
        let mut out = vec![f64::NAN; n * m];
        if n > 0 && m > 0 {
            let x = panel_of(&cn.x, ctx, store)?;
            let g = cn.group.as_ref().map(|g| panel_of(g, ctx, store)).transpose()?;
            out.par_chunks_mut(m).enumerate().for_each_init(Scratch::default, |s, (d, row)| {
                let span = d * m..(d + 1) * m;
                cross::apply_row(
                    cn.kind,
                    &cn.params,
                    &x[span.clone()],
                    g.map(|g| &g[span.clone()]),
                    &ctx.present[span],
                    row,
                    s,
                );
            });
        }
        store.insert(cn.key, out);
    }
    Ok(())
}

/// Evaluate over a panel. Returns one row-major `n_dates x n_assets` panel per output.
///
/// Refuses (typed) a panel that is not `n_dates x n_assets`, a field or group series the
/// program reads and the inputs lack, and `at(...)` without timestamps or a clock.
pub fn eval_panel(plan: &PanelPlan, inputs: &PanelInputs) -> Result<Vec<Vec<f64>>, EvalError> {
    let (n, m) = (inputs.n_dates, inputs.n_assets);
    let want = n * m;
    let check = |name: &str, got: usize, expected: usize| {
        if got == expected {
            Ok(())
        } else {
            Err(EvalError::LengthMismatch { name: name.to_string(), expected, got })
        }
    };
    for (f, p) in Field::ALL.iter().zip(&inputs.fields) {
        if let Some(p) = p {
            check(f.name(), p.len(), want)?;
        }
    }
    for (name, p) in &inputs.handles {
        check(name, p.len(), want)?;
    }
    if let Some(p) = inputs.present {
        check("present", p.len(), want)?;
    }
    for f in plan.fields() {
        if inputs.fields[*f as usize].is_none() {
            return Err(EvalError::MissingField(f.name()));
        }
    }
    for h in plan.handles() {
        if !inputs.handles.contains_key(h) {
            return Err(EvalError::MissingHandle(h.clone()));
        }
    }
    if plan.has_at {
        let ts = inputs.timestamps.ok_or(EvalError::NoTimestamps)?;
        check("timestamps", ts.len(), n)?;
        if inputs.clock.is_none() {
            return Err(EvalError::NoClock);
        }
    }

    let derived: Vec<bool>;
    let present: &[bool] = match inputs.present {
        Some(p) => p,
        None => {
            let read: Vec<&[f64]> =
                plan.fields().iter().filter_map(|f| inputs.fields[*f as usize]).collect();
            derived =
                (0..want).map(|i| read.is_empty() || read.iter().any(|p| !p[i].is_nan())).collect();
            &derived
        }
    };
    let ctx = Ctx {
        n,
        m,
        fields: inputs.fields,
        handles: &inputs.handles,
        present,
        timestamps: inputs.timestamps,
        clock: inputs.clock.as_ref(),
    };

    let mut store: BTreeMap<Key, Vec<f64>> = BTreeMap::new();
    for (at, ph) in plan.phases.iter().enumerate() {
        match ph {
            Phase::Ts(t) => {
                let panels = run_ts(t, &ctx, &store)?;
                for (k, p) in t.exports.iter().zip(panels) {
                    store.insert(*k, p);
                }
            }
            Phase::Cross(c) => run_cross(c, &ctx, &mut store)?,
        }
        store.retain(|k, _| plan.last_use.get(k).is_some_and(|l| *l > at));
    }
    plan.finals
        .iter()
        .map(|k| {
            store.remove(k).ok_or_else(|| EvalError::Node(format!("no panel for output {k:?}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::kernel::canon_bits;
    use crate::quant::lang::{compile, Env};

    const NAN: f64 = f64::NAN;

    fn plan(src: &str) -> PanelPlan {
        PanelPlan::new(&compile(src, &Env::default()).unwrap_or_else(|e| panic!("{src}: {e}")))
            .unwrap()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| canon_bits(*x)).collect()
    }

    /// n x m prices: name a at date d, with name 2 listed from date 3.
    fn close(n: usize, m: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(n * m);
        for d in 0..n {
            for a in 0..m {
                let x = 10.0 + a as f64 * 3.0 + ((d * 7 + a * 5) % 11) as f64 * 0.5;
                v.push(if a == 2 && d < 3 { NAN } else { x });
            }
        }
        v
    }

    fn run(src: &str, n: usize, m: usize, c: &[f64]) -> Vec<f64> {
        let p = plan(src);
        eval_panel(&p, &PanelInputs::new(n, m).field(Field::Close, c)).unwrap().remove(0)
    }

    #[test]
    fn a_series_expression_runs_per_column_and_a_late_listing_counts_from_its_own_first_bar() {
        let (n, m) = (8, 3);
        let c = close(n, m);
        let got = run("sma(close, 3)", n, m, &c);
        // Name 2 has bars at dates 3.. only: its first value is its third bar, date 5.
        let col2: Vec<f64> = (0..n).map(|d| got[d * m + 2]).collect();
        assert!(col2[..5].iter().all(|v| v.is_nan()) && !col2[5].is_nan());
        let want = (c[3 * m + 2] + c[4 * m + 2] + c[5 * m + 2]) / 3.0;
        assert!((col2[5] - want).abs() < 1e-12);
        // Name 0 is warm after two bars.
        assert!(got[m].is_nan() && !got[2 * m].is_nan());
    }

    #[test]
    fn rank_is_the_cross_section_of_the_time_series() {
        let (n, m) = (6, 4);
        let c = close(n, m);
        let r = run("rank(roc(close, 2))", n, m, &c);
        let roc = run("roc(close, 2)", n, m, &c);
        let mut s = Scratch::default();
        for d in 0..n {
            let mut want = vec![0.0; m];
            cross::apply_row(
                Kind::Rank,
                &[ParamVal::Int(2)],
                &roc[d * m..(d + 1) * m],
                None,
                &vec![true; m],
                &mut want,
                &mut s,
            );
            assert_eq!(bits(&r[d * m..(d + 1) * m]), bits(&want), "date {d}");
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let (n, m) = (40, 9);
        let c = close(n, m);
        let src = "ema(rank(roc(close, 3)), 4) - zscore_xs(sma(close, 5))";
        let p = plan(src);
        let go = |threads| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                eval_panel(&p, &PanelInputs::new(n, m).field(Field::Close, &c)).unwrap().remove(0)
            })
        };
        assert_eq!(bits(&go(1)), bits(&go(4)));
        assert!(p.n_phases() >= 3);
    }

    #[test]
    fn a_leaf_program_and_a_constant_program() {
        let (n, m) = (3, 2);
        let c = close(n, m);
        assert_eq!(bits(&run("close", n, m, &c)), bits(&c));
        let k = PanelPlan::new(&compile("1 + 2", &Env::default()).unwrap());
        assert!(matches!(k, Err(EvalError::Unsupported(_))));
    }

    #[test]
    fn inputs_are_checked() {
        let p = plan("rank(close)");
        let c = [1.0, 2.0, 3.0];
        let r = eval_panel(&p, &PanelInputs::new(2, 2).field(Field::Close, &c));
        assert!(matches!(r, Err(EvalError::LengthMismatch { .. })));
        let r = eval_panel(&p, &PanelInputs::new(1, 3));
        assert_eq!(r, Err(EvalError::MissingField("close")));
    }
}
