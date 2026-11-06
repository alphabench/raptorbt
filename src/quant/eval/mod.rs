//! Evaluation of a checked [`Program`]: one executable [`Plan`], two ways to run it.
//!
//! The plan assigns a register to every node output (an `Out` is an alias and computes
//! nothing), keeps one step per `Const`, `Field`, `Handle`, unary, binary and `Call` node in
//! schedule order, and assigns batch buffers by liveness. `batch::eval` runs whole columns,
//! `Stream` runs one bar at a time. Every stateless operator is defined once, in [`ops`], and
//! both modes call that definition, so they cannot differ. Kernels are the registry's and are
//! bit-identical across modes by their own contract.
//!
//! Values are finite or NaN. Booleans are `1.0`, `0.0` or NaN (unknown) with Kleene logic.

mod batch;
pub mod panel;
mod stream;
pub mod sweep;

pub use batch::{eval, eval_many};
pub use panel::{eval_panel, PanelInputs, PanelPlan, PanelStream};
pub use stream::Stream;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use super::fns::at::AtState;
use super::kernel::Node as KernelNode;
use super::lang::{BinOp, Clock, Field, NodeId, Op, Program, Shape, UnOp};
use super::registry::{FnDef, ParamVal};
use crate::core::session_schedule::SessionSchedule;
use crate::data::{BarSpec, BarStamp};

/// Output columns, one per program output, each as long as the input.
pub type Columns = Vec<Vec<f64>>;

/// Most heap one evaluation's kernels may hold. A window of `p` bars is allocated whole when its
/// kernel is built, whatever the data holds, so `sma(close, 16777216)` costs the same on 400
/// bars as on 400 million; the sum of the kernels' state is counted as they are built and the
/// plan is refused past this, with the bytes. The peak is this plus the last kernel built.
pub const STATE_LIMIT_BYTES: usize = 512 << 20;

/// The one definition of every stateless operator. Batch loops and the streaming step both
/// call these.
pub mod ops {
    use crate::quant::lang::BinOp;

    #[inline(always)]
    fn finite_or_nan(x: f64) -> f64 {
        if x.is_infinite() {
            f64::NAN
        } else {
            x
        }
    }

    #[inline(always)]
    fn flag(c: bool) -> f64 {
        if c {
            1.0
        } else {
            0.0
        }
    }

    /// Overflow is NaN, never inf: a series holds finite values or NaN.
    #[inline(always)]
    pub fn add(a: f64, b: f64) -> f64 {
        finite_or_nan(a + b)
    }

    #[inline(always)]
    pub fn sub(a: f64, b: f64) -> f64 {
        finite_or_nan(a - b)
    }

    #[inline(always)]
    pub fn mul(a: f64, b: f64) -> f64 {
        finite_or_nan(a * b)
    }

    /// Division by zero is NaN, never inf.
    #[inline(always)]
    pub fn div(a: f64, b: f64) -> f64 {
        finite_or_nan(a / b)
    }

    #[inline(always)]
    pub fn pow(a: f64, b: f64) -> f64 {
        finite_or_nan(a.powf(b))
    }

    /// Natural logarithm; zero and negative input give NaN (not `-inf`).
    #[inline(always)]
    pub fn ln(x: f64) -> f64 {
        finite_or_nan(x.ln())
    }

    #[inline(always)]
    pub fn neg(x: f64) -> f64 {
        -x
    }

    /// Kleene not: unknown stays unknown.
    #[inline(always)]
    pub fn not(x: f64) -> f64 {
        if x.is_nan() {
            f64::NAN
        } else if x == 0.0 {
            1.0
        } else {
            0.0
        }
    }

    // Every comparison with a NaN is unknown.
    macro_rules! cmp {
        ($name:ident, $op:tt) => {
            #[inline(always)]
            pub fn $name(a: f64, b: f64) -> f64 {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    flag(a $op b)
                }
            }
        };
    }
    cmp!(lt, <);
    cmp!(le, <=);
    cmp!(gt, >);
    cmp!(ge, >=);
    cmp!(eq, ==);
    cmp!(ne, !=);

    /// Kleene and: a false operand decides it whatever the other is.
    #[inline(always)]
    pub fn and(a: f64, b: f64) -> f64 {
        if a == 0.0 || b == 0.0 {
            0.0
        } else if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            1.0
        }
    }

    /// Kleene or: a true operand decides it whatever the other is.
    #[inline(always)]
    pub fn or(a: f64, b: f64) -> f64 {
        let truthy = |x: f64| !x.is_nan() && x != 0.0;
        if truthy(a) || truthy(b) {
            1.0
        } else if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn bin(op: BinOp, a: f64, b: f64) -> f64 {
        match op {
            BinOp::Add => add(a, b),
            BinOp::Sub => sub(a, b),
            BinOp::Mul => mul(a, b),
            BinOp::Div => div(a, b),
            BinOp::Pow => pow(a, b),
            BinOp::Lt => lt(a, b),
            BinOp::Le => le(a, b),
            BinOp::Gt => gt(a, b),
            BinOp::Ge => ge(a, b),
            BinOp::Eq => eq(a, b),
            BinOp::Ne => ne(a, b),
            BinOp::And => and(a, b),
            BinOp::Or => or(a, b),
        }
    }
}

/// Why a plan could not be built or run. Never a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// The program uses a construct this evaluator does not run yet (panels, other clocks).
    Unsupported(String),
    /// No column was supplied at all, so there is no length to evaluate.
    NoInput,
    MissingField(&'static str),
    MissingHandle(String),
    /// A column differs in length from the first one supplied.
    LengthMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    /// A bar carried a different number of handle values than the plan names.
    HandleCount {
        expected: usize,
        got: usize,
    },
    /// A kernel refused its parameters or registers.
    Node(String),
    /// The program uses `at(...)`, which cuts the bars by their timestamps, and none were given.
    NoTimestamps,
    /// The program uses `at(...)` and no bar stamp (and zone) was bound to it.
    NoClock,
    /// The kernels' state would exceed [`STATE_LIMIT_BYTES`]: a period that is long for the
    /// data. `bytes` is what the kernels built so far held when the limit was crossed.
    State {
        bytes: usize,
        limit: usize,
    },
    /// An `at(...)` timeframe that is not a coarser exact multiple of the base bars.
    Timeframe {
        tf: String,
        base_ns: i64,
        valid: Vec<String>,
    },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Unsupported(what) => write!(f, "not supported by the evaluator: {what}"),
            EvalError::NoInput => write!(f, "no input column was supplied"),
            EvalError::MissingField(n) => {
                write!(f, "the program reads `{n}` but it was not supplied")
            }
            EvalError::MissingHandle(n) => {
                write!(f, "the program reads handle `{n}` but it was not supplied")
            }
            EvalError::LengthMismatch { name, expected, got } => {
                write!(f, "column `{name}` has {got} values, expected {expected}")
            }
            EvalError::HandleCount { expected, got } => {
                write!(f, "expected {expected} handle values per bar, got {got}")
            }
            EvalError::Node(m) => write!(f, "kernel construction failed: {m}"),
            EvalError::NoTimestamps => write!(
                f,
                "the program uses at(...), which cuts bars by their timestamps; none were supplied"
            ),
            EvalError::NoClock => write!(
                f,
                "the program uses at(...), which needs the bar stamp (open or close) its \
                 timestamps mean"
            ),
            EvalError::State { bytes, limit } => write!(
                f,
                "the expression's windows hold {bytes} bytes, over the {limit}-byte limit on one \
                 evaluation: use shorter periods"
            ),
            EvalError::Timeframe { tf, base_ns, valid } => write!(
                f,
                "at(\"{tf}\") must be a coarser exact multiple of the {base_ns} ns base bars; \
                 valid: {}",
                valid.join(", ")
            ),
        }
    }
}

impl std::error::Error for EvalError {}

/// What an `at(...)` needs to cut the base bars into windows, bound to a run: the zone, what
/// the timestamps mean and the instrument's session. It is the hub's own cut (`AggKey`): a
/// daily window is one date in the session's zone, and the schedule's zone replaces
/// `tz_offset_ns` where there is a schedule.
#[derive(Debug, Clone)]
pub struct ClockCtx {
    /// Seconds-east-of-UTC of the civil day, in nanoseconds (IST is 19 800 000 000 000).
    pub tz_offset_ns: i64,
    /// Whether a timestamp is the start or the end of its bar.
    pub stamp: BarStamp,
    /// The instrument's trading calendar, if it has one.
    pub session: Option<Arc<SessionSchedule>>,
    /// The base bar width, when known; only used to refuse a timeframe that is not coarser.
    pub base_ns: Option<i64>,
    /// With no `base_ns`: work the base width out of the first two bars and refuse a timeframe
    /// that is not coarser than it (see `fns::at`). A live cell that was told no width.
    pub infer_width: bool,
}

impl ClockCtx {
    pub fn new(tz_offset_ns: i64, stamp: BarStamp, session: Option<SessionSchedule>) -> Self {
        Self {
            tz_offset_ns,
            stamp,
            session: session.map(Arc::new),
            base_ns: None,
            infer_width: false,
        }
    }

    /// Only for measuring a stream's size: UTC, close-stamped, no session.
    pub(crate) fn probe() -> Self {
        Self::new(0, BarStamp::Close, None)
    }

    /// The text of what decides the numbers, for the bound hash: the zone, the stamp and the
    /// schedule's content. The base width is not part of it: it can only refuse, never change.
    pub fn text(&self) -> String {
        let stamp = match self.stamp {
            BarStamp::Open => "open",
            BarStamp::Close => "close",
        };
        let session = match &self.session {
            None => "none".to_string(),
            Some(s) => crate::journal::sha256_hex(
                serde_json::to_string(&**s).unwrap_or_default().as_bytes(),
            ),
        };
        format!("tz={};stamp={stamp};session={session}", self.tz_offset_ns)
    }
}

/// Named input columns for batch evaluation. Every column supplied must be the same length,
/// whether or not the program reads it.
#[derive(Debug, Clone, Default)]
pub struct Inputs<'a> {
    fields: [Option<&'a [f64]>; 5],
    handles: BTreeMap<String, &'a [f64]>,
    timestamps: Option<&'a [i64]>,
    clock: Option<ClockCtx>,
}

impl<'a> Inputs<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn field(mut self, f: Field, col: &'a [f64]) -> Self {
        self.fields[f as usize] = Some(col);
        self
    }

    pub fn handle(mut self, name: &str, col: &'a [f64]) -> Self {
        self.handles.insert(name.to_string(), col);
        self
    }

    /// The bars' timestamps in nanoseconds. Read only by a program that uses `at(...)`; any other
    /// program ignores them.
    pub fn timestamps(mut self, ts: &'a [i64]) -> Self {
        self.timestamps = Some(ts);
        self
    }

    /// What the timestamps mean and how to cut windows from them. Read only by a program that
    /// uses `at(...)`.
    pub fn clock(mut self, clock: ClockCtx) -> Self {
        self.clock = Some(clock);
        self
    }

    /// The timestamps and clock an `at(...)` needs, checked against the `n` bars; none for a
    /// plan without one.
    fn at_inputs(
        &self,
        plan: &Plan,
        n: usize,
    ) -> Result<Option<(&'a [i64], &ClockCtx)>, EvalError> {
        if plan.d.ats.is_empty() {
            return Ok(None);
        }
        let ts = self.timestamps.ok_or(EvalError::NoTimestamps)?;
        if ts.len() != n {
            return Err(EvalError::LengthMismatch {
                name: "timestamps".to_string(),
                expected: n,
                got: ts.len(),
            });
        }
        let clock = self.clock.as_ref().ok_or(EvalError::NoClock)?;
        Ok(Some((ts, clock)))
    }

    /// The length and the plan's input columns in the plan's order.
    fn resolve(&self, plan: &Plan) -> Result<(usize, Vec<&'a [f64]>), EvalError> {
        let supplied: Vec<(&str, &'a [f64])> = Field::ALL
            .iter()
            .zip(&self.fields)
            .filter_map(|(f, c)| c.map(|c| (f.name(), c)))
            .chain(self.handles.iter().map(|(k, v)| (k.as_str(), *v)))
            .collect();
        let Some(&(_, first)) = supplied.first() else {
            return Err(EvalError::NoInput);
        };
        let n = first.len();
        if let Some((name, c)) = supplied.iter().find(|(_, c)| c.len() != n) {
            return Err(EvalError::LengthMismatch {
                name: name.to_string(),
                expected: n,
                got: c.len(),
            });
        }
        let cols = plan
            .d
            .inputs
            .iter()
            .map(|s| match s {
                Source::Field(f) => {
                    self.fields[*f as usize].ok_or(EvalError::MissingField(f.name()))
                }
                Source::Handle(k) => {
                    let name = &plan.d.handle_names[*k];
                    self.handles
                        .get(name)
                        .copied()
                        .ok_or_else(|| EvalError::MissingHandle(name.clone()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((n, cols))
    }
}

/// One bar for streaming evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    /// A bar with every field unknown.
    pub const NAN: Bar =
        Bar { open: f64::NAN, high: f64::NAN, low: f64::NAN, close: f64::NAN, volume: f64::NAN };

    #[inline(always)]
    pub fn set(&mut self, f: Field, v: f64) {
        match f {
            Field::Open => self.open = v,
            Field::High => self.high = v,
            Field::Low => self.low = v,
            Field::Close => self.close = v,
            Field::Volume => self.volume = v,
        }
    }

    #[inline(always)]
    pub fn get(&self, f: Field) -> f64 {
        match f {
            Field::Open => self.open,
            Field::High => self.high,
            Field::Low => self.low,
            Field::Close => self.close,
            Field::Volume => self.volume,
        }
    }
}

/// What a `Load` step reads.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Source {
    Field(Field),
    /// Index into the plan's handle names.
    Handle(usize),
}

#[derive(Debug)]
pub(crate) enum Step {
    Const {
        dst: usize,
        v: f64,
    },
    Load {
        dst: usize,
        input: usize,
    },
    Neg {
        dst: usize,
        a: usize,
    },
    Not {
        dst: usize,
        a: usize,
    },
    Bin {
        op: BinOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    /// An `at(...)`: `ins` are the registers of the body's `fields`; `at` indexes the plan's
    /// bodies and the states built for a run.
    At {
        dst: usize,
        ins: Vec<usize>,
        fields: Vec<Field>,
        at: usize,
    },
    /// `node` indexes the kernel nodes built for a run.
    Call {
        def: &'static FnDef,
        params: Vec<ParamVal>,
        ins: Vec<usize>,
        outs: Vec<usize>,
        node: usize,
    },
}

impl Step {
    fn reads(&self) -> Vec<usize> {
        match self {
            Step::Const { .. } | Step::Load { .. } => vec![],
            Step::Neg { a, .. } | Step::Not { a, .. } => vec![*a],
            Step::Bin { a, b, .. } => vec![*a, *b],
            Step::Call { ins, .. } | Step::At { ins, .. } => ins.clone(),
        }
    }

    fn writes(&self) -> Vec<usize> {
        match self {
            Step::Const { dst, .. }
            | Step::Load { dst, .. }
            | Step::Neg { dst, .. }
            | Step::Not { dst, .. }
            | Step::Bin { dst, .. }
            | Step::At { dst, .. } => vec![*dst],
            Step::Call { outs, .. } => outs.clone(),
        }
    }
}

/// Where a register lives in batch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Loc {
    /// A caller's column, read in place.
    Input(usize),
    /// A pooled buffer.
    Slot(usize),
}

#[derive(Debug)]
pub(crate) struct PlanData {
    pub steps: Vec<Step>,
    pub n_regs: usize,
    pub inputs: Vec<Source>,
    pub handle_names: Vec<String>,
    pub locs: Vec<Loc>,
    pub n_slots: usize,
    pub out_regs: Vec<usize>,
    pub warm: usize,
    pub n_calls: usize,
    /// The `at(...)` bodies, by `Step::At::at`.
    pub ats: Vec<AtSpec>,
}

/// One `at(...)` of a plan: its timeframe and the plan of its body.
#[derive(Debug)]
pub(crate) struct AtSpec {
    pub spec: BarSpec,
    pub body: Plan,
}

impl PlanData {
    /// Fresh `at(...)` states cut as `clock` says, in `ats` order.
    pub fn instantiate_ats(&self, clock: &ClockCtx) -> Result<Vec<AtState>, EvalError> {
        self.ats.iter().map(|a| AtState::new(a.spec, &a.body, clock)).collect()
    }

    /// Fresh kernel nodes, one per call step, in call order.
    pub fn instantiate(&self) -> Result<Vec<Box<dyn KernelNode>>, EvalError> {
        let mut nodes = Vec::with_capacity(self.n_calls);
        let mut bytes = 0usize;
        for step in &self.steps {
            if let Step::Call { def, params, ins, outs, .. } = step {
                let node = (def.make)(params, ins, outs).map_err(EvalError::Node)?;
                bytes = bytes.saturating_add(node.state_bytes());
                if bytes > STATE_LIMIT_BYTES {
                    return Err(EvalError::State { bytes, limit: STATE_LIMIT_BYTES });
                }
                nodes.push(node);
            }
        }
        Ok(nodes)
    }
}

/// An executable program. Cheap to clone and share across threads; holds no run state.
#[derive(Debug, Clone)]
pub struct Plan {
    pub(crate) d: Arc<PlanData>,
}

impl Plan {
    pub fn new(program: &Program) -> Result<Plan, EvalError> {
        Self::with_outputs(program, &program.outputs())
    }

    /// A plan whose outputs are `outs` (node, output index) rather than the program's root: the
    /// panel evaluator hands each time-series phase to the same planner this way.
    pub(crate) fn with_outputs(
        program: &Program,
        outs: &[(NodeId, u8)],
    ) -> Result<Plan, EvalError> {
        let nodes = program.dag.nodes();
        let mut reg_of = vec![0usize; nodes.len()];
        let mut n_regs = 0;
        let mut inputs: Vec<Source> = Vec::new();
        let mut handle_names: Vec<String> = Vec::new();
        let mut steps: Vec<Step> = Vec::new();
        let mut n_calls = 0;
        let mut ats: Vec<AtSpec> = Vec::new();

        for (id, node) in nodes.iter().enumerate() {
            if node.ty.shape == Shape::Panel || node.ty.clock != Clock::Base {
                return Err(EvalError::Unsupported(format!(
                    "node {id} is a panel or on another clock; a panel expression runs through \
                     Expr.eval_panel"
                )));
            }
            if let Op::Out { call, index } = &node.op {
                reg_of[id] = reg_of[*call] + *index as usize;
                continue;
            }
            let dst = n_regs;
            reg_of[id] = dst;
            n_regs += node.n_outputs();
            if let Op::At { spec, body } = &node.op {
                // The body reads the base bars' fields; they are loaded beside this node's own
                // register and cut into windows by the step.
                let body = Plan::new(body)?;
                let mut fields = body.fields();
                fields.sort_by_key(|f| *f as usize);
                fields.dedup();
                let mut ins = Vec::with_capacity(fields.len());
                for f in &fields {
                    inputs.push(Source::Field(*f));
                    steps.push(Step::Load { dst: n_regs, input: inputs.len() - 1 });
                    ins.push(n_regs);
                    n_regs += 1;
                }
                ats.push(AtSpec { spec: *spec, body });
                steps.push(Step::At { dst, ins, fields, at: ats.len() - 1 });
                continue;
            }
            steps.push(match &node.op {
                Op::Const(v) => Step::Const { dst, v: *v },
                Op::Field(f) => {
                    inputs.push(Source::Field(*f));
                    Step::Load { dst, input: inputs.len() - 1 }
                }
                Op::Handle(name) => {
                    let k = handle_names.iter().position(|h| h == name).unwrap_or_else(|| {
                        handle_names.push(name.clone());
                        handle_names.len() - 1
                    });
                    inputs.push(Source::Handle(k));
                    Step::Load { dst, input: inputs.len() - 1 }
                }
                Op::Unary(UnOp::Neg, x) => Step::Neg { dst, a: reg_of[*x] },
                Op::Unary(UnOp::Not, x) => Step::Not { dst, a: reg_of[*x] },
                Op::Binary(op, a, b) => Step::Bin { op: *op, dst, a: reg_of[*a], b: reg_of[*b] },
                Op::Call { def, params, inputs: ins } => {
                    n_calls += 1;
                    Step::Call {
                        def,
                        params: params.clone(),
                        ins: ins.iter().map(|i| reg_of[*i]).collect(),
                        outs: (dst..n_regs).collect(),
                        node: n_calls - 1,
                    }
                }
                Op::Out { .. } | Op::At { .. } => continue,
            });
        }

        let out_regs: Vec<usize> = outs.iter().map(|(id, i)| reg_of[*id] + *i as usize).collect();
        let (locs, n_slots) = assign_locs(&steps, n_regs, &out_regs);
        Ok(Plan {
            d: Arc::new(PlanData {
                steps,
                n_regs,
                inputs,
                handle_names,
                locs,
                n_slots,
                out_regs,
                warm: outs.iter().map(|(id, _)| nodes[*id].warm).max().unwrap_or(0),
                n_calls,
                ats,
            }),
        })
    }

    /// Leading bars on which the root may be unknown (static; the first known bar on clean
    /// data is at most this index, and exactly it unless a Kleene `and`/`or` decides early).
    pub fn warmup(&self) -> usize {
        self.d.warm
    }

    /// Heap one stream over this plan owns (see [`Stream::state_bytes`]), measured on a fresh
    /// stream.
    pub fn state_bytes(&self) -> Result<usize, EvalError> {
        Ok(Stream::bound(self, ClockCtx::probe())?.state_bytes())
    }

    /// Heap a batch run over `rows` bars holds besides the kernels' state: the buffer pool.
    pub fn batch_bytes(&self, rows: usize) -> usize {
        self.d.n_slots.saturating_mul(rows).saturating_mul(std::mem::size_of::<f64>())
    }

    /// Whether the plan runs an `at(...)`, so it needs timestamps and a [`ClockCtx`].
    pub fn has_at(&self) -> bool {
        !self.d.ats.is_empty()
    }

    /// Number of output columns.
    pub fn n_outputs(&self) -> usize {
        self.d.out_regs.len()
    }

    /// Kernel nodes the plan runs: a shared sub-expression counts once.
    pub fn n_kernel_steps(&self) -> usize {
        self.d.n_calls
    }

    /// Registers, one per node output.
    pub fn n_regs(&self) -> usize {
        self.d.n_regs
    }

    /// Buffers a batch run allocates.
    pub fn n_slots(&self) -> usize {
        self.d.n_slots
    }

    /// Fields the program reads.
    pub fn fields(&self) -> Vec<Field> {
        self.d
            .inputs
            .iter()
            .filter_map(|s| if let Source::Field(f) = s { Some(*f) } else { None })
            .collect()
    }

    /// Handle names in the order a stream expects their per-bar values.
    pub fn handles(&self) -> &[String] {
        &self.d.handle_names
    }
}

/// Liveness-based buffer assignment. A stateless step frees its dying inputs before taking its
/// buffer, so it computes in place; a kernel step takes its output buffers first, because the
/// kernel reads inputs and writes outputs at once.
fn assign_locs(steps: &[Step], n_regs: usize, out_regs: &[usize]) -> (Vec<Loc>, usize) {
    let mut last = vec![0usize; n_regs];
    for (s, st) in steps.iter().enumerate() {
        for w in st.writes() {
            last[w] = s;
        }
        for r in st.reads() {
            last[r] = s;
        }
    }
    for &o in out_regs {
        last[o] = usize::MAX;
    }

    let mut locs = vec![Loc::Slot(usize::MAX); n_regs];
    let mut free: Vec<usize> = Vec::new();
    let mut n_slots = 0;
    let mut take = |free: &mut Vec<usize>| {
        free.pop().unwrap_or_else(|| {
            n_slots += 1;
            n_slots - 1
        })
    };
    for (s, st) in steps.iter().enumerate() {
        let mut dying = st.reads();
        dying.sort_unstable();
        dying.dedup();
        dying.retain(|r| last[*r] == s);
        let release = |free: &mut Vec<usize>, locs: &[Loc]| {
            for r in &dying {
                if let Loc::Slot(k) = locs[*r] {
                    free.push(k);
                }
            }
        };
        match st {
            Step::Load { dst, input } => locs[*dst] = Loc::Input(*input),
            Step::Call { outs, .. } => {
                for w in outs {
                    locs[*w] = Loc::Slot(take(&mut free));
                }
                release(&mut free, &locs);
            }
            _ => {
                release(&mut free, &locs);
                for w in st.writes() {
                    locs[w] = Loc::Slot(take(&mut free));
                }
            }
        }
        // A result nobody reads (an unselected output of a multi-output call) frees at once.
        for w in st.writes() {
            if last[w] == s {
                if let Loc::Slot(k) = locs[w] {
                    free.push(k);
                }
            }
        }
    }
    (locs, n_slots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::kernel::canon_bits;
    use crate::quant::lang::{compile, Env, Handle, Ty};
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    const NAN: f64 = f64::NAN;

    fn runner(seed: u64, cases: u32) -> TestRunner {
        TestRunner::new(Config {
            cases,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(seed),
            ..Config::default()
        })
    }

    fn env() -> Env {
        let mut e = Env::default();
        e.series.insert("h".into(), Handle { ty: Ty::series(), warm: 0 });
        e
    }

    fn plan(src: &str) -> Plan {
        let p = compile(src, &env()).unwrap_or_else(|e| panic!("{src:?}: {e}"));
        Plan::new(&p).unwrap()
    }

    #[derive(Debug, Clone)]
    struct Data {
        open: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    }

    impl Data {
        fn of(open: &[f64], close: &[f64], volume: &[f64]) -> Data {
            Data { open: open.to_vec(), close: close.to_vec(), volume: volume.to_vec() }
        }

        fn inputs(&self) -> Inputs<'_> {
            Inputs::new()
                .field(Field::Open, &self.open)
                .field(Field::Close, &self.close)
                .field(Field::Volume, &self.volume)
        }

        fn len(&self) -> usize {
            self.close.len()
        }
    }

    fn run_batch(pl: &Plan, d: &Data) -> Columns {
        eval(pl, &d.inputs()).unwrap()
    }

    fn run_stream(pl: &Plan, d: &Data) -> Columns {
        let mut s = Stream::new(pl).unwrap();
        let mut cols = vec![Vec::new(); pl.n_outputs()];
        for i in 0..d.len() {
            let bar = Bar {
                open: d.open[i],
                high: NAN,
                low: NAN,
                close: d.close[i],
                volume: d.volume[i],
            };
            for (c, v) in cols.iter_mut().zip(s.update(&bar, &[]).unwrap()) {
                c.push(*v);
            }
        }
        cols
    }

    fn bits(c: &Columns) -> Vec<Vec<u64>> {
        c.iter().map(|col| col.iter().map(|x| canon_bits(*x)).collect()).collect()
    }

    fn value() -> impl Strategy<Value = f64> {
        prop_oneof![8 => -1e3f64..1e3, 1 => Just(NAN), 1 => Just(0.0)]
    }

    fn data() -> impl Strategy<Value = Data> {
        prop::collection::vec((value(), value(), value()), 0..60).prop_map(|rows| Data {
            open: rows.iter().map(|r| r.0).collect(),
            close: rows.iter().map(|r| r.1).collect(),
            volume: rows.iter().map(|r| r.2).collect(),
        })
    }

    /// Numeric expressions that always contain a field, so no sub-expression is a bare constant.
    fn series_expr() -> BoxedStrategy<String> {
        let leaf = prop::sample::select(vec!["open", "close", "volume"]).prop_map(String::from);
        let arith = prop::sample::select(vec!["+", "-", "*", "/"]);
        leaf.prop_recursive(3, 16, 3, move |inner| {
            prop_oneof![
                (inner.clone(), inner.clone(), arith.clone())
                    .prop_map(|(a, b, o)| format!("({a} {o} {b})")),
                (inner.clone(), prop::sample::select(vec!["0", "1", "2.5"]), arith.clone())
                    .prop_map(|(a, c, o)| format!("({a} {o} {c})")),
                (inner.clone(), 1u32..6).prop_map(|(a, p)| format!("sma({a}, {p})")),
                (inner.clone(), 1u32..6).prop_map(|(a, p)| format!("ema({a}, {p})")),
                inner.clone().prop_map(|a| format!("(-{a})")),
                inner.prop_map(|a| format!("({a} ** 2)")),
            ]
        })
        .boxed()
    }

    fn bool_expr() -> BoxedStrategy<String> {
        let cmp = prop::sample::select(vec!["<", "<=", ">", ">=", "==", "!="]);
        let leaf =
            (series_expr(), series_expr(), cmp).prop_map(|(a, b, o)| format!("({a} {o} {b})"));
        leaf.prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                inner.clone().prop_map(|a| format!("(not {a})")),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| format!("({a} and {b})")),
                (inner.clone(), inner).prop_map(|(a, b)| format!("({a} or {b})")),
            ]
        })
        .boxed()
    }

    fn any_expr() -> BoxedStrategy<String> {
        prop_oneof![series_expr(), bool_expr()].boxed()
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans() {
        runner(21, 300)
            .run(&(any_expr(), data()), |(src, d)| {
                let pl = plan(&src);
                prop_assert_eq!(bits(&run_batch(&pl, &d)), bits(&run_stream(&pl, &d)), "{}", src);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn prefix_identity_poisoned_future_does_not_move_the_past() {
        let strategy = (any_expr(), data(), 0usize..80);
        runner(22, 300)
            .run(&strategy, |(src, d, cut)| {
                let cut = cut.min(d.len());
                let poison = |v: &[f64]| -> Vec<f64> {
                    v.iter()
                        .enumerate()
                        .map(|(i, x)| match (i >= cut, i % 2) {
                            (false, _) => *x,
                            (true, 0) => NAN,
                            (true, _) => 1e12 * (i as f64 + 1.0),
                        })
                        .collect()
                };
                let p = Data {
                    open: poison(&d.open),
                    close: poison(&d.close),
                    volume: poison(&d.volume),
                };
                let pl = plan(&src);
                let (a, b) = (bits(&run_batch(&pl, &d)), bits(&run_batch(&pl, &p)));
                for (x, y) in a.iter().zip(&b) {
                    prop_assert_eq!(&x[..cut], &y[..cut], "{} cut={}", src, cut);
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn eval_many_equals_sequential_for_any_thread_count() {
        let strategy = (any_expr(), prop::collection::vec(data(), 0..6));
        runner(23, 60)
            .run(&strategy, |(src, sets)| {
                let pl = plan(&src);
                let ins: Vec<Inputs> = sets.iter().map(Data::inputs).collect();
                let seq: Vec<_> = ins.iter().map(|i| bits(&eval(&pl, i).unwrap())).collect();
                for threads in [1, 3] {
                    let pool =
                        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                    let par: Vec<_> = pool
                        .install(|| eval_many(&pl, &ins))
                        .into_iter()
                        .map(|r| bits(&r.unwrap()))
                        .collect();
                    prop_assert_eq!(&seq, &par, "{} threads={}", src, threads);
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn a_shared_subexpression_is_one_kernel_step() {
        assert_eq!(plan("sma(close, 3) + sma(close, 3) * 2").n_kernel_steps(), 1);
        assert_eq!(plan("sma(close, 3) + sma(close, 4)").n_kernel_steps(), 2);
        assert_eq!(plan("ema(sma(close, 3), 3) + sma(close, 3)").n_kernel_steps(), 2);
        assert_eq!(plan("(sma(close, 3) > 1) and (sma(close, 3) < 5)").n_kernel_steps(), 1);
        assert_eq!(plan("close + close").n_kernel_steps(), 0);
    }

    #[test]
    fn kleene_truth_tables_are_exact() {
        let v = [0.0, 1.0, NAN];
        let and = [[0.0, 0.0, 0.0], [0.0, 1.0, NAN], [0.0, NAN, NAN]];
        let or = [[0.0, 1.0, NAN], [1.0, 1.0, 1.0], [NAN, 1.0, NAN]];
        let not = [1.0, 0.0, NAN];
        for (i, a) in v.iter().enumerate() {
            assert_eq!(canon_bits(ops::not(*a)), canon_bits(not[i]), "not {a}");
            for (j, b) in v.iter().enumerate() {
                assert_eq!(canon_bits(ops::and(*a, *b)), canon_bits(and[i][j]), "{a} and {b}");
                assert_eq!(canon_bits(ops::or(*a, *b)), canon_bits(or[i][j]), "{a} or {b}");
            }
        }
        for f in [ops::lt, ops::le, ops::gt, ops::ge, ops::eq, ops::ne] {
            for (a, b) in [(NAN, 1.0), (1.0, NAN), (NAN, NAN)] {
                assert!(f(a, b).is_nan());
            }
        }
        assert_eq!((ops::lt(1.0, 2.0), ops::lt(2.0, 1.0), ops::lt(2.0, 2.0)), (1.0, 0.0, 0.0));
        assert_eq!((ops::le(2.0, 2.0), ops::gt(2.0, 1.0), ops::ge(1.0, 2.0)), (1.0, 1.0, 0.0));
        assert_eq!((ops::eq(2.0, 2.0), ops::ne(2.0, 2.0), ops::ne(1.0, 2.0)), (1.0, 0.0, 1.0));
    }

    #[test]
    fn kleene_logic_through_the_evaluator_in_both_modes() {
        // a = open > 0 and b = close > 0 run over every pair of {true, false, unknown}.
        let d = Data::of(
            &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, NAN, NAN, NAN],
            &[1.0, 0.0, NAN, 1.0, 0.0, NAN, 1.0, 0.0, NAN],
            &[0.0; 9],
        );
        for (src, want) in [
            ("(open > 0) and (close > 0)", [1.0, 0.0, NAN, 0.0, 0.0, 0.0, NAN, 0.0, NAN]),
            ("(open > 0) or (close > 0)", [1.0, 1.0, 1.0, 1.0, 0.0, NAN, 1.0, NAN, NAN]),
            ("not (open > 0)", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, NAN, NAN, NAN]),
        ] {
            let pl = plan(src);
            let want = vec![want.iter().map(|x| canon_bits(*x)).collect::<Vec<_>>()];
            assert_eq!(bits(&run_batch(&pl, &d)), want, "batch {src}");
            assert_eq!(bits(&run_stream(&pl, &d)), want, "stream {src}");
        }
    }

    #[test]
    fn a_live_input_is_not_overwritten_by_in_place_reuse() {
        // x = close + 1 is read by the product, by the sma and by the final subtraction.
        // x = [2, 3, 4, 5, 6]; sma(x, 3) = [_, _, 3, 4, 5]; x * sma = [_, _, 12, 20, 30];
        // minus x = [_, _, 8, 15, 24].
        let d = Data::of(&[0.0; 5], &[1.0, 2.0, 3.0, 4.0, 5.0], &[0.0; 5]);
        let pl = plan("(close + 1) * sma(close + 1, 3) - (close + 1)");
        let want =
            vec![[NAN, NAN, 8.0, 15.0, 24.0].iter().map(|x| canon_bits(*x)).collect::<Vec<_>>()];
        assert_eq!(bits(&run_batch(&pl, &d)), want);
        assert_eq!(bits(&run_stream(&pl, &d)), want);
        // A caller's column is copied out, never aliased, and never modified.
        let same = plan("close");
        assert_eq!(run_batch(&same, &d)[0], d.close);
    }

    #[test]
    fn buffers_are_reused_along_a_chain() {
        let pl = plan("((close + 1) * 2) - 3");
        assert!(pl.n_slots() < pl.n_regs(), "{} slots for {} registers", pl.n_slots(), pl.n_regs());
        let d = Data::of(&[0.0; 3], &[1.0, 2.0, 3.0], &[0.0; 3]);
        // (c + 1) * 2 - 3 = 2c - 1.
        assert_eq!(run_batch(&pl, &d)[0], vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn warm_up_matches_the_first_known_bar_and_ready() {
        let clean: Vec<f64> = (1..=30).map(f64::from).collect();
        let d = Data::of(&clean, &clean, &clean);
        for (src, warm) in [
            ("close", 0),
            ("sma(close, 3)", 2),
            ("ema(sma(close, 3), 4)", 5),
            ("sma(close, 3) > ema(close, 5)", 4),
            ("not (sma(close, 4) > 1)", 3),
        ] {
            let pl = plan(src);
            assert_eq!(pl.warmup(), warm, "{src}");
            let first = run_batch(&pl, &d)[0].iter().position(|v| !v.is_nan()).unwrap();
            assert_eq!(first, warm, "{src}");
            let mut s = Stream::new(&pl).unwrap();
            for i in 0..d.len() {
                let bar = Bar { open: 0.0, high: 0.0, low: 0.0, close: d.close[i], volume: 0.0 };
                let out = s.update(&bar, &[]).unwrap()[0];
                assert_eq!(s.ready(), i >= warm, "{src} bar {i}");
                assert_eq!(!out.is_nan(), i >= warm, "{src} bar {i}");
            }
        }
        // A decided Kleene `and` is known before the static warm-up: never later than it.
        let pl = plan("(close < 0) and (sma(close, 5) > 0)");
        let first = run_batch(&pl, &d)[0].iter().position(|v| !v.is_nan()).unwrap();
        assert!(first <= pl.warmup() && pl.warmup() == 4);
    }

    #[test]
    fn short_empty_missing_and_mismatched_inputs_are_typed() {
        let pl = plan("sma(close, 5)");
        // Shorter than the warm-up: all unknown, in both modes.
        let short = Data::of(&[0.0; 3], &[1.0, 2.0, 3.0], &[0.0; 3]);
        assert!(run_batch(&pl, &short)[0].iter().all(|v| v.is_nan()));
        let mut s = Stream::new(&pl).unwrap();
        for c in &short.close {
            let bar = Bar { open: 0.0, high: 0.0, low: 0.0, close: *c, volume: 0.0 };
            assert!(s.update(&bar, &[]).unwrap()[0].is_nan());
            assert!(!s.ready());
        }
        // Zero-length columns give zero-length outputs.
        let empty = Data::of(&[], &[], &[]);
        assert_eq!(run_batch(&pl, &empty), vec![Vec::<f64>::new()]);
        // No column at all.
        assert_eq!(eval(&pl, &Inputs::new()), Err(EvalError::NoInput));
        // A field the program reads is missing.
        let open = [1.0, 2.0];
        assert_eq!(
            eval(&pl, &Inputs::new().field(Field::Open, &open)),
            Err(EvalError::MissingField("close"))
        );
        // Columns of different lengths, whichever one the program reads.
        let (close, volume) = ([1.0, 2.0, 3.0], [1.0, 2.0]);
        let r = eval(&pl, &Inputs::new().field(Field::Close, &close).field(Field::Volume, &volume));
        assert_eq!(
            r,
            Err(EvalError::LengthMismatch { name: "volume".into(), expected: 3, got: 2 })
        );
        // A handle the program reads is missing.
        let hp = plan("h + 1");
        assert_eq!(
            eval(&hp, &Inputs::new().field(Field::Close, &close)),
            Err(EvalError::MissingHandle("h".into()))
        );
    }

    #[test]
    fn handles_flow_in_both_modes() {
        let hp = plan("h + close");
        assert_eq!(hp.handles(), ["h".to_string()]);
        let (close, h) = ([1.0, 2.0, NAN, 4.0], [10.0, 20.0, 30.0, 40.0]);
        let batch = eval(&hp, &Inputs::new().field(Field::Close, &close).handle("h", &h)).unwrap();
        let mut s = Stream::new(&hp).unwrap();
        let mut got = Vec::new();
        for i in 0..4 {
            let bar = Bar { open: 0.0, high: 0.0, low: 0.0, close: close[i], volume: 0.0 };
            got.push(s.update(&bar, &[h[i]]).unwrap()[0]);
        }
        assert_eq!(bits(&batch), bits(&vec![got]));
        assert_eq!(batch[0][0], 11.0);
        let bar = Bar { open: 0.0, high: 0.0, low: 0.0, close: 1.0, volume: 0.0 };
        assert_eq!(s.update(&bar, &[]), Err(EvalError::HandleCount { expected: 1, got: 0 }));
    }

    #[test]
    fn division_by_zero_and_log_of_non_positive_are_nan_never_inf() {
        let d = Data::of(&[0.0; 5], &[1.0, -1.0, 0.0, NAN, 1e200], &[0.0; 5]);
        for src in ["close / open", "open ** (-1)", "close ** 2", "close / 0"] {
            let pl = plan(src);
            for out in [run_batch(&pl, &d), run_stream(&pl, &d)] {
                assert!(out[0].iter().all(|v| !v.is_infinite()), "{src}: {:?}", out[0]);
            }
        }
        assert!(run_batch(&plan("close / open"), &d)[0].iter().all(|v| v.is_nan()));
        assert!(run_batch(&plan("open ** (-1)"), &d)[0].iter().all(|v| v.is_nan()));
        assert!(run_batch(&plan("close ** 2"), &d)[0][4].is_nan());
        for x in [0.0, -1.0, -0.0, f64::NEG_INFINITY, f64::INFINITY, NAN] {
            assert!(ops::ln(x).is_nan(), "ln({x})");
        }
        assert_eq!(ops::ln(1.0), 0.0);
        assert!(ops::div(1.0, 0.0).is_nan() && ops::div(-1.0, 0.0).is_nan());
        assert!(ops::pow(0.0, -1.0).is_nan());
    }
}
