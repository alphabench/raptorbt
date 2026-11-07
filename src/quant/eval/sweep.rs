//! Parameter sweeps: one expression, many bindings of its `$name` constants, one joint DAG.
//!
//! Every binding is compiled on its own (so each is exactly the program a plain `expr(...)`
//! with those params would be) and its nodes are re-interned into one DAG. The DAG is
//! hash-consed, so a node no parameter touches, and a node two bindings spell the same, is one
//! node and computes once. Nothing is rewritten, so every binding's output is bit-identical to
//! evaluating it alone: no prefix-sum or lane shortcut is taken.
//!
//! The joint plan runs in waves: a wave is the steps whose inputs were all produced in earlier
//! waves, they run in parallel, and a buffer is reused once the wave of its last reader is over.
//! Each step is deterministic and writes only its own buffers, so the numbers do not depend on
//! the thread count. The size is checked before anything large is allocated: past the bound a
//! sweep is refused with the size it would have needed, never started and left to swap.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::mem;

use rayon::prelude::*;

use super::batch::{zip1, zip_bin, Opd};
use super::{ops, Bar, Columns, EvalError, Inputs, Loc, Plan, PlanData, Step};
use crate::quant::fns::at::AtState;
use crate::quant::kernel::Node as KernelNode;
use crate::quant::lang::parser::{self, Expr, ExprKind};
use crate::quant::lang::{compile, Dag, Env, ExprError, NodeId, Op, Program};

/// Largest number of bindings one sweep may try.
pub const MAX_COMBINATIONS: usize = 10_000;
/// Largest working set (buffers and result columns) one sweep may allocate, in bytes.
pub const MAX_BYTES: usize = 4 << 30;

/// The bound on a sweep.
#[derive(Debug, Clone, Copy)]
pub struct SweepLimits {
    pub max_combinations: usize,
    pub max_bytes: usize,
}

impl Default for SweepLimits {
    fn default() -> Self {
        Self { max_combinations: MAX_COMBINATIONS, max_bytes: MAX_BYTES }
    }
}

/// Why a sweep was refused or failed.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepError {
    /// A binding does not compile (an unbound `$name` left over, a value out of range).
    Compile(ExprError),
    Eval(EvalError),
    /// A parameter was given that the source never reads.
    Unused {
        name: String,
        valid: Vec<String>,
    },
    /// No binding to try: an empty axis, an empty list of points or no parameter at all.
    Empty,
    TooManyCombinations {
        combinations: usize,
        max: usize,
    },
    TooManyBytes {
        combinations: usize,
        bytes: usize,
        max: usize,
    },
}

impl fmt::Display for SweepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SweepError::Compile(e) => write!(f, "{e}"),
            SweepError::Eval(e) => write!(f, "{e}"),
            SweepError::Unused { name, valid } => write!(
                f,
                "parameter `{name}` is not read by the expression; it reads: {}",
                if valid.is_empty() { "no parameter".to_string() } else { valid.join(", ") }
            ),
            SweepError::Empty => write!(f, "the sweep has no binding to try"),
            SweepError::TooManyCombinations { combinations, max } => {
                write!(f, "the sweep has {combinations} combinations; at most {max} are allowed")
            }
            SweepError::TooManyBytes { combinations, bytes, max } => write!(
                f,
                "the sweep of {combinations} combinations needs about {bytes} bytes \
                 ({:.2} GiB); at most {max} bytes ({:.2} GiB) are allowed",
                *bytes as f64 / (1u64 << 30) as f64,
                *max as f64 / (1u64 << 30) as f64
            ),
        }
    }
}

impl std::error::Error for SweepError {}

impl From<EvalError> for SweepError {
    fn from(e: EvalError) -> Self {
        SweepError::Eval(e)
    }
}

/// What to try: every combination of the axes (the last axis varies fastest), or the listed
/// bindings as they are.
#[derive(Debug, Clone)]
pub enum Space {
    Grid(Vec<(String, Vec<f64>)>),
    Points(Vec<BTreeMap<String, f64>>),
}

impl Space {
    /// Number of bindings, `None` when it does not fit a `usize`.
    pub fn len(&self) -> Option<usize> {
        match self {
            Space::Grid(axes) if axes.is_empty() => Some(0),
            Space::Grid(axes) => axes.iter().try_fold(1usize, |n, (_, v)| n.checked_mul(v.len())),
            Space::Points(p) => Some(p.len()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Every name the space binds.
    fn names(&self) -> BTreeSet<&str> {
        match self {
            Space::Grid(axes) => axes.iter().map(|(n, _)| n.as_str()).collect(),
            Space::Points(p) => p.iter().flat_map(|m| m.keys().map(String::as_str)).collect(),
        }
    }

    /// The bindings in input order. Call only after `len` passed the bound.
    fn bindings(&self) -> Vec<BTreeMap<String, f64>> {
        match self {
            Space::Points(p) => p.clone(),
            Space::Grid(axes) => {
                let total = self.len().unwrap_or(0);
                let mut out = Vec::with_capacity(total);
                for mut k in 0..total {
                    let mut m = BTreeMap::new();
                    for (name, values) in axes.iter().rev() {
                        m.insert(name.clone(), values[k % values.len()]);
                        k /= values.len();
                    }
                    out.push(m);
                }
                out
            }
        }
    }
}

/// The `$name`s a source reads.
pub fn param_names(src: &str) -> Result<BTreeSet<String>, ExprError> {
    fn walk(e: &Expr, out: &mut BTreeSet<String>) {
        match &e.kind {
            ExprKind::Param(n) => {
                out.insert(n.clone());
            }
            ExprKind::Call { args, .. } => args.iter().for_each(|a| walk(a, out)),
            ExprKind::Unary(_, x) | ExprKind::Select(x, _, _) => walk(x, out),
            ExprKind::Binary(_, a, b) => {
                walk(a, out);
                walk(b, out);
            }
            ExprKind::Num(_) | ExprKind::Str(_) | ExprKind::Ident(_) => {}
        }
    }
    let mut out = BTreeSet::new();
    walk(&parser::parse(src)?, &mut out);
    Ok(out)
}

/// Re-intern `p`'s nodes into `dag`; the joint id of each of `p`'s nodes.
fn merge(dag: &mut Dag, p: &Program) -> Vec<NodeId> {
    let mut map: Vec<NodeId> = Vec::with_capacity(p.dag.len());
    for node in p.dag.nodes() {
        let id = match &node.op {
            Op::Const(v) => dag.constant(*v),
            Op::Field(f) => dag.field(*f),
            Op::Handle(name) => dag.handle(name, node.ty.clone(), node.warm),
            Op::Unary(op, x) => dag.unary(*op, map[*x], node.ty.clone()),
            Op::Binary(op, a, b) => dag.binary(*op, map[*a], map[*b], node.ty.clone()),
            Op::Call { def, params, inputs } => dag.call(
                def,
                params.clone(),
                inputs.iter().map(|i| map[*i]).collect(),
                node.ty.clone(),
            ),
            Op::Out { call, index } => dag.out(map[*call], *index),
            Op::At { spec, body } => dag.at(*spec, body.clone(), node.ty.clone()),
            Op::SessionBar => dag.session_bar(),
        };
        map.push(id);
    }
    map
}

/// The wave each step runs in and the buffer each register lives in.
#[derive(Debug)]
struct Schedule {
    /// Indices into the plan's steps, wave by wave.
    waves: Vec<Vec<usize>>,
    /// Per register: its buffer, or `usize::MAX` for a caller's column.
    slot: Vec<usize>,
    n_slots: usize,
}

fn schedule(d: &PlanData) -> Schedule {
    let n_regs = d.n_regs;
    // `ready[r]`: the first wave that may read register `r` (a caller's column: wave 0).
    let mut ready = vec![0usize; n_regs];
    let mut step_wave = vec![usize::MAX; d.steps.len()];
    let mut n_waves = 0;
    for (i, st) in d.steps.iter().enumerate() {
        if matches!(st, Step::Load { .. }) {
            continue;
        }
        let w = st.reads().iter().map(|r| ready[*r]).max().unwrap_or(0);
        step_wave[i] = w;
        for r in st.writes() {
            ready[r] = w + 1;
        }
        n_waves = n_waves.max(w + 1);
    }

    // The wave after which a register's buffer may be reused; results live to the end.
    let mut last = vec![0usize; n_regs];
    for (i, st) in d.steps.iter().enumerate() {
        if step_wave[i] == usize::MAX {
            continue;
        }
        for r in st.reads().into_iter().chain(st.writes()) {
            last[r] = last[r].max(step_wave[i]);
        }
    }
    for &o in &d.out_regs {
        last[o] = usize::MAX;
    }

    let has_slot = |r: usize| matches!(d.locs[r], Loc::Slot(_));
    let mut waves = vec![Vec::new(); n_waves];
    for (i, w) in step_wave.iter().enumerate() {
        if *w != usize::MAX {
            waves[*w].push(i);
        }
    }
    let mut dying = vec![Vec::new(); n_waves];
    for r in (0..n_regs).filter(|r| has_slot(*r)) {
        if last[r] != usize::MAX {
            dying[last[r]].push(r);
        }
    }

    let mut slot = vec![usize::MAX; n_regs];
    let mut free: Vec<usize> = Vec::new();
    let mut n_slots = 0;
    for (w, wave) in waves.iter().enumerate() {
        for &i in wave {
            for r in d.steps[i].writes() {
                slot[r] = free.pop().unwrap_or_else(|| {
                    n_slots += 1;
                    n_slots - 1
                });
            }
        }
        for &r in &dying[w] {
            free.push(slot[r]);
        }
    }
    Schedule { waves, slot, n_slots }
}

/// The joint plan of a set of bindings.
#[derive(Debug)]
pub struct SweepPlan {
    plan: Plan,
    sched: Schedule,
    /// Output columns of each binding.
    widths: Vec<usize>,
    /// Kernel steps the bindings would run one by one.
    independent_steps: usize,
}

impl SweepPlan {
    /// One plan over `programs`, one per binding, in order.
    pub fn new(programs: &[Program]) -> Result<SweepPlan, EvalError> {
        let mut dag = Dag::new();
        let mut outs: Vec<(NodeId, u8)> = Vec::new();
        let mut widths = Vec::with_capacity(programs.len());
        let mut independent_steps = 0;
        for p in programs {
            let map = merge(&mut dag, p);
            let o = p.outputs();
            widths.push(o.len());
            outs.extend(o.into_iter().map(|(id, i)| (map[id], i)));
            independent_steps +=
                p.dag.nodes().iter().filter(|n| matches!(n.op, Op::Call { .. })).count();
        }
        let root = outs.first().map_or(0, |o| o.0);
        let joint = Program { dag, root, canonical: String::new(), hash: String::new() };
        let plan = Plan::with_outputs(&joint, &outs)?;
        let sched = schedule(&plan.d);
        Ok(SweepPlan { plan, sched, widths, independent_steps })
    }

    /// An upper bound on the heap of the joint plan's kernels, from the parameters alone.
    pub fn state_estimate(&self) -> usize {
        self.plan.state_estimate()
    }

    pub fn n_bindings(&self) -> usize {
        self.widths.len()
    }

    /// Kernel nodes the joint plan runs: a node shared by bindings counts once.
    pub fn n_kernel_steps(&self) -> usize {
        self.plan.n_kernel_steps()
    }

    /// Kernel nodes the bindings would run evaluated one by one.
    pub fn independent_kernel_steps(&self) -> usize {
        self.independent_steps
    }

    /// Heap a run over `n` bars allocates: the buffers, plus a copy for each result that is a
    /// caller's column or repeats another binding's.
    pub fn bytes(&self, n: usize) -> usize {
        let d = &*self.plan.d;
        let distinct: BTreeSet<usize> =
            d.out_regs.iter().copied().filter(|r| matches!(d.locs[*r], Loc::Slot(_))).collect();
        let copies = d.out_regs.len() - distinct.len();
        (self.sched.n_slots + copies).saturating_mul(n).saturating_mul(8)
    }

    /// Evaluate every binding over `inputs`; one [`Columns`] per binding, in binding order.
    pub fn eval(&self, inputs: &Inputs, limits: &SweepLimits) -> Result<Vec<Columns>, SweepError> {
        let plan = &self.plan;
        let d = &*plan.d;
        let n = inputs.rows()?;
        let session = inputs.session_column(plan, n)?;
        let (n, cols) = inputs.resolve(plan, session.as_deref())?;
        let bytes = self.bytes(n);
        if bytes > limits.max_bytes {
            return Err(SweepError::TooManyBytes {
                combinations: self.n_bindings(),
                bytes,
                max: limits.max_bytes,
            });
        }
        let at_inputs = inputs.at_inputs(plan, n)?;
        let mut nodes: Vec<Option<Box<dyn KernelNode>>> =
            d.instantiate()?.into_iter().map(Some).collect();
        let mut ats: Vec<Option<AtState>> = match at_inputs {
            Some((_, clock)) => d.instantiate_ats(clock)?.into_iter().map(Some).collect(),
            None => Vec::new(),
        };
        let ts = at_inputs.map(|(ts, _)| ts);
        let mut pool: Vec<Vec<f64>> = vec![Vec::new(); self.sched.n_slots];

        for wave in &self.sched.waves {
            let mut tasks: Vec<Task> = wave
                .iter()
                .map(|&i| {
                    let step = &d.steps[i];
                    let writes = step.writes();
                    let bufs = writes
                        .iter()
                        .map(|r| {
                            let b = mem::take(&mut pool[self.sched.slot[*r]]);
                            if b.len() == n {
                                b
                            } else {
                                vec![0.0; n]
                            }
                        })
                        .collect();
                    let node = match step {
                        Step::Call { node, .. } => nodes[*node].take(),
                        _ => None,
                    };
                    let at = match step {
                        Step::At { at, .. } => ats[*at].take(),
                        _ => None,
                    };
                    Task { step, writes, bufs, node, at }
                })
                .collect();
            {
                let ctx = Ctx { d, slot: &self.sched.slot, pool: &pool, cols: &cols, ts };
                tasks.par_iter_mut().for_each(|t| run(t, &ctx));
            }
            for t in tasks {
                for (r, b) in t.writes.iter().zip(t.bufs) {
                    pool[self.sched.slot[*r]] = b;
                }
            }
        }

        // A result read twice is copied for all but its last reader.
        let mut left: BTreeMap<usize, usize> = BTreeMap::new();
        for r in &d.out_regs {
            *left.entry(*r).or_insert(0) += 1;
        }
        let mut flat: Vec<Vec<f64>> = Vec::with_capacity(d.out_regs.len());
        for r in &d.out_regs {
            let rest = left.get_mut(r).expect("counted above");
            *rest -= 1;
            flat.push(match d.locs[*r] {
                Loc::Input(k) => cols[k].to_vec(),
                Loc::Slot(_) if *rest == 0 => mem::take(&mut pool[self.sched.slot[*r]]),
                Loc::Slot(_) => pool[self.sched.slot[*r]].clone(),
            });
        }
        let mut flat = flat.into_iter();
        Ok(self.widths.iter().map(|w| flat.by_ref().take(*w).collect()).collect())
    }
}

struct Task<'p> {
    step: &'p Step,
    writes: Vec<usize>,
    bufs: Vec<Vec<f64>>,
    node: Option<Box<dyn KernelNode>>,
    at: Option<AtState>,
}

/// What a step reads: the buffers of earlier waves, which no task of this wave holds.
struct Ctx<'a> {
    d: &'a PlanData,
    slot: &'a [usize],
    pool: &'a [Vec<f64>],
    cols: &'a [&'a [f64]],
    ts: Option<&'a [i64]>,
}

impl<'a> Ctx<'a> {
    fn view(&self, reg: usize) -> &'a [f64] {
        match self.d.locs[reg] {
            Loc::Input(k) => self.cols[k],
            Loc::Slot(_) => &self.pool[self.slot[reg]],
        }
    }
}

fn run(t: &mut Task, cx: &Ctx) {
    match t.step {
        Step::Load { .. } => {}
        Step::Const { v, .. } => t.bufs[0].fill(*v),
        Step::Neg { a, .. } => zip1(&mut t.bufs[0], Opd::Slice(cx.view(*a)), ops::neg),
        Step::Not { a, .. } => zip1(&mut t.bufs[0], Opd::Slice(cx.view(*a)), ops::not),
        Step::Bin { op, a, b, .. } => {
            zip_bin(*op, &mut t.bufs[0], Opd::Slice(cx.view(*a)), Opd::Slice(cx.view(*b)))
        }
        Step::At { ins, fields, .. } => {
            // The same state machine a stream steps, bar by bar in order.
            let (Some(state), Some(ts)) = (t.at.as_mut(), cx.ts) else { return };
            let columns: Vec<&[f64]> = ins.iter().map(|r| cx.view(*r)).collect();
            for (i, out) in t.bufs[0].iter_mut().enumerate() {
                let mut bar = Bar::NAN;
                for (c, f) in columns.iter().zip(fields) {
                    bar.set(*f, c[i]);
                }
                *out = state.step(ts[i], &bar);
            }
        }
        Step::Call { ins, .. } => {
            let in_cols: Vec<&[f64]> = ins.iter().map(|r| cx.view(*r)).collect();
            let mut out_cols: Vec<&mut [f64]> =
                t.bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
            if let Some(node) = t.node.as_mut() {
                node.run_batch(&in_cols, &mut out_cols);
            }
        }
    }
}

/// A finished sweep.
#[derive(Debug)]
pub struct SweepOutput {
    /// The bindings tried, in order: the base params of the source overridden by the space.
    pub bindings: Vec<BTreeMap<String, f64>>,
    /// Output columns of each binding.
    pub columns: Vec<Columns>,
    pub n_kernel_steps: usize,
    pub independent_kernel_steps: usize,
}

/// Compile `src` under each binding of `space` (over `base`, whose params the space overrides)
/// and evaluate them as one joint plan.
pub fn sweep(
    src: &str,
    base: &Env,
    space: &Space,
    inputs: &Inputs,
    limits: &SweepLimits,
) -> Result<SweepOutput, SweepError> {
    let combinations = match space.len() {
        Some(0) => return Err(SweepError::Empty),
        Some(n) if n <= limits.max_combinations => n,
        other => {
            return Err(SweepError::TooManyCombinations {
                combinations: other.unwrap_or(usize::MAX),
                max: limits.max_combinations,
            })
        }
    };
    let used = param_names(src).map_err(SweepError::Compile)?;
    if let Some(name) = space.names().into_iter().find(|n| !used.contains(*n)) {
        return Err(SweepError::Unused {
            name: name.to_string(),
            valid: used.iter().cloned().collect(),
        });
    }
    let points = space.bindings();
    debug_assert_eq!(points.len(), combinations);
    let mut bindings = Vec::with_capacity(combinations);
    let mut programs = Vec::with_capacity(combinations);
    for point in points {
        let mut env = base.clone();
        env.params.extend(point);
        programs.push(compile(src, &env).map_err(SweepError::Compile)?);
        bindings.push(env.params);
    }
    let plan = SweepPlan::new(&programs)?;
    let columns = plan.eval(inputs, limits)?;
    Ok(SweepOutput {
        bindings,
        columns,
        n_kernel_steps: plan.n_kernel_steps(),
        independent_kernel_steps: plan.independent_kernel_steps(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::eval::eval;
    use crate::quant::kernel::canon_bits;
    use crate::quant::lang::Field;
    use proptest::prelude::*;

    const NAN: f64 = f64::NAN;

    fn series() -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(prop_oneof![8 => 1.0f64..200.0, 1 => Just(NAN)], 0..80)
    }

    fn grid(n: &[f64], m: &[f64]) -> Space {
        Space::Grid(vec![("n".into(), n.to_vec()), ("m".into(), m.to_vec())])
    }

    fn bits(c: &Columns) -> Vec<Vec<u64>> {
        c.iter().map(|col| col.iter().map(|x| canon_bits(*x)).collect()).collect()
    }

    fn alone(src: &str, params: &BTreeMap<String, f64>, close: &[f64]) -> Columns {
        let env = Env { params: params.clone(), ..Env::default() };
        let plan = Plan::new(&compile(src, &env).unwrap()).unwrap();
        eval(&plan, &Inputs::new().field(Field::Close, close)).unwrap()
    }

    const SRCS: [&str; 3] = [
        "sma(close, $n) > sma(close, $m)",
        "ema(close, $n) - sma(close, $m) / sma(close, $n)",
        "(rsi(close, $n) < 30) and (close > ema(close, $m))",
    ];

    proptest! {
        #![proptest_config(ProptestConfig { cases: 48, failure_persistence: None, ..ProptestConfig::default() })]

        #[test]
        fn every_binding_is_bit_identical_to_its_own_evaluation(
            close in series(),
            k in 0..SRCS.len(),
            n in prop::collection::vec(2i64..9, 1..4),
            m in prop::collection::vec(2i64..9, 1..4),
        ) {
            let (n, m): (Vec<f64>, Vec<f64>) =
                (n.iter().map(|x| *x as f64).collect(), m.iter().map(|x| *x as f64).collect());
            let inputs = Inputs::new().field(Field::Close, &close);
            let out = sweep(SRCS[k], &Env::default(), &grid(&n, &m), &inputs, &SweepLimits::default());
            if close.is_empty() {
                // A zero-length column gives zero-length outputs, as a plain evaluation does.
                prop_assert!(out.is_ok() || matches!(out, Err(SweepError::Eval(_))));
                return Ok(());
            }
            let out = out.unwrap();
            prop_assert_eq!(out.columns.len(), n.len() * m.len());
            for (b, cols) in out.bindings.iter().zip(&out.columns) {
                prop_assert_eq!(bits(cols), bits(&alone(SRCS[k], b, &close)));
            }
        }
    }

    #[test]
    fn unchanged_nodes_compute_once() {
        let close: Vec<f64> = (0..60).map(|i| 100.0 + (i % 7) as f64).collect();
        let inputs = Inputs::new().field(Field::Close, &close);
        let space = grid(&[5.0, 10.0, 20.0], &[50.0, 5.0]);
        let out =
            sweep(SRCS[0], &Env::default(), &space, &inputs, &SweepLimits::default()).unwrap();
        // sma(close, 5) is `n` and `m` at once: 4 distinct smas. Alone, the binding (5, 5) is one
        // node (`sma5 > sma5`) and the other five are two each: 11, not 12.
        assert_eq!(out.n_kernel_steps, 4);
        assert_eq!(out.independent_kernel_steps, 11);
    }

    #[test]
    fn the_bound_refuses_with_the_size() {
        let close = vec![1.0; 1000];
        let inputs = Inputs::new().field(Field::Close, &close);
        let space = grid(&[2.0, 3.0, 4.0], &[5.0, 6.0]);
        let few = SweepLimits { max_combinations: 5, ..SweepLimits::default() };
        assert_eq!(
            sweep(SRCS[0], &Env::default(), &space, &inputs, &few).unwrap_err(),
            SweepError::TooManyCombinations { combinations: 6, max: 5 }
        );
        let small = SweepLimits { max_bytes: 1000, ..SweepLimits::default() };
        assert!(matches!(
            sweep(SRCS[0], &Env::default(), &space, &inputs, &small).unwrap_err(),
            SweepError::TooManyBytes { combinations: 6, .. }
        ));
    }

    #[test]
    fn a_parameter_the_source_never_reads_is_refused() {
        let close = vec![1.0; 10];
        let inputs = Inputs::new().field(Field::Close, &close);
        let space = Space::Grid(vec![("z".into(), vec![1.0])]);
        let e = sweep("close", &Env::default(), &space, &inputs, &SweepLimits::default());
        assert!(matches!(e, Err(SweepError::Unused { .. })));
    }

    /// The joint plan's estimate covers its built kernels and is no more than the bindings run
    /// one by one.
    #[test]
    fn the_state_estimate_covers_the_joint_plan() {
        let env = Env::default();
        let programs: Vec<Program> = [10, 20, 30]
            .iter()
            .map(|n| compile(&format!("sma(close, {n}) + ema(close, 5)"), &env).unwrap())
            .collect();
        let plan = SweepPlan::new(&programs).unwrap();
        let measured: usize =
            plan.plan.d.instantiate().unwrap().iter().map(|n| n.state_bytes()).sum();
        let est = plan.state_estimate();
        let alone: usize = programs.iter().map(|p| Plan::new(p).unwrap().state_estimate()).sum();
        assert!(est >= measured && est <= alone, "{measured} <= {est} <= {alone}");
    }
}
