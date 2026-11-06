//! Expression cells: a checked expression as a bank indicator.
//!
//! An expression reaches the bank as a [`ProgramId`] from a process-wide interner keyed on the
//! program's full canonical text (not a hash), so two registrations share a cell exactly when
//! their canonical expressions are equal. An id is never reused: it means one program for the
//! life of the process, or none.
//!
//! A program lives while something holds a lease on it (an `Arc<ProgramInfo>`: an `Indicator`,
//! a registry, a bank, a cell). When the table holds [`MAX_PROGRAMS`] programs the interner
//! drops those nobody holds, so a long-lived worker that makes many one-off expressions never
//! fills it; only [`MAX_PROGRAMS`] programs in use at once are refused. A dropped program has no
//! cell, so no sharing is lost: the same text interned again gets a fresh id.
//!
//! A cell runs the program's own [`Stream`] over the bars the bank feeds it, the same step the
//! batch evaluator runs, so an expression in a strategy gives the numbers `expr(...).eval` gives
//! on the same bars, backtest and live alike. A program that reads `volume` gets the bar's real
//! volume from every path into the bank (primary bars, trade-built bars and composite bars all
//! carry it); only a hand-fed `update_bar` can lack one, and it is refused there.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use crate::core::error::RaptorError;
use crate::core::Result;
use crate::quant::eval::{Bar, ClockCtx, EvalError, PanelPlan, PanelStream, Plan, Stream};
use crate::quant::lang::{Elem, Field, Program};

use super::indicators::{Ohlc, Reading};

/// Most programs in use (leased) at once in one process.
pub const MAX_PROGRAMS: usize = 1 << 16;

/// A checked expression, by identity of its canonical text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramId(u32);

impl ProgramId {
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The id a key's parameter holds.
    pub(crate) fn from_param(raw: u64) -> Self {
        Self(raw as u32)
    }
}

/// What the bank needs to know about an interned program.
#[derive(Debug)]
pub struct ProgramInfo {
    pub canonical: String,
    /// The series plan; none for a cross-sectional program, which has `panel` instead.
    plan: Option<Plan>,
    /// A cross-sectional program's panel plan: its cells share one [`PanelCore`] per universe.
    panel: Option<PanelPlan>,
    /// Output columns: more than one only for an unselected multi-output call.
    pub outputs: usize,
    /// Leading bars that are unknown on clean data.
    pub warmup: usize,
    /// The value is a Kleene boolean (1, 0, unknown), not a number.
    pub boolean: bool,
    pub reads_volume: bool,
    /// The program uses `at(...)`: a cell needs the clock of the stream it runs on, and the bar's
    /// timestamp.
    pub needs_clock: bool,
    /// Heap one cell's stream owns, measured on a probe stream at interning.
    pub state_bytes: usize,
}

#[derive(Default)]
struct Interner {
    by_canonical: HashMap<Box<str>, ProgramId>,
    programs: HashMap<u32, Arc<ProgramInfo>>,
    /// The next id; ids are never reused.
    next: u32,
}

impl Interner {
    /// Make room for one more program: drop the ones only the table holds.
    fn room(&mut self) -> Result<()> {
        if self.programs.len() >= MAX_PROGRAMS {
            self.programs.retain(|_, info| Arc::strong_count(info) > 1);
            let live = &self.programs;
            self.by_canonical.retain(|_, id| live.contains_key(&id.0));
        }
        if self.programs.len() >= MAX_PROGRAMS {
            return Err(RaptorError::invalid_parameter(format!(
                "expression table is full: {MAX_PROGRAMS} expressions in use in one process"
            )));
        }
        if self.next == u32::MAX {
            return Err(RaptorError::invalid_parameter(
                "expression ids are exhausted in this process; restart it",
            ));
        }
        Ok(())
    }

    fn add(&mut self, canonical: &str, info: ProgramInfo) -> (ProgramId, Arc<ProgramInfo>) {
        let id = ProgramId(self.next);
        self.next += 1;
        let info = Arc::new(info);
        self.programs.insert(id.0, info.clone());
        self.by_canonical.insert(canonical.into(), id);
        (id, info)
    }
}

fn interner() -> MutexGuard<'static, Interner> {
    static PROGRAMS: OnceLock<Mutex<Interner>> = OnceLock::new();
    // A panic elsewhere cannot leave the table half-edited: every edit is one insert or retain.
    PROGRAMS.get_or_init(Mutex::default).lock().unwrap_or_else(PoisonError::into_inner)
}

/// Intern `program` (run by `plan`). The same canonical text gives the same id.
///
/// Fails when the plan cannot be run as a stream or the interner is full; nothing about the
/// program is checked again when a cell is built, so every refusal is here. The id holds only
/// while something holds a lease on it: see [`intern_leased`].
pub fn intern(program: &Program, plan: &Plan) -> Result<ProgramId> {
    intern_leased(program, plan).map(|(id, _)| id)
}

/// [`intern`], and the lease that keeps the program interned while it is held.
pub fn intern_leased(program: &Program, plan: &Plan) -> Result<(ProgramId, Arc<ProgramInfo>)> {
    let mut table = interner();
    if let Some(&id) = table.by_canonical.get(program.canonical.as_str()) {
        if let Some(info) = table.programs.get(&id.0) {
            return Ok((id, info.clone()));
        }
    }
    table.room()?;
    if !plan.handles().is_empty() {
        return Err(RaptorError::invalid_parameter(
            "an indicator expression reads bar fields only, not named series",
        ));
    }
    let state_bytes = plan
        .state_bytes()
        .map_err(|e| RaptorError::invalid_parameter(format!("expression cannot stream: {e}")))?;
    let boolean = program.dag.node(program.root).ty.elem == Elem::Bool;
    let info = ProgramInfo {
        canonical: program.canonical.clone(),
        plan: Some(plan.clone()),
        panel: None,
        outputs: plan.n_outputs(),
        warmup: program.warmup(),
        boolean,
        reads_volume: plan.fields().contains(&Field::Volume),
        needs_clock: plan.has_at(),
        state_bytes,
    };
    Ok(table.add(&program.canonical, info))
}

/// Intern a cross-sectional `program` (run by `panel`). Its cells are one per member of a universe
/// (`IndInput::Universe`) and share one [`PanelCore`], which steps once per bar of the universe.
///
/// Fails for what the bank cannot feed: `at(...)`, a supplied series (a group label) or several
/// outputs.
pub fn intern_panel(program: &Program, panel: &PanelPlan) -> Result<ProgramId> {
    intern_panel_leased(program, panel).map(|(id, _)| id)
}

/// [`intern_panel`], and the lease that keeps the program interned while it is held.
pub fn intern_panel_leased(
    program: &Program,
    panel: &PanelPlan,
) -> Result<(ProgramId, Arc<ProgramInfo>)> {
    let mut table = interner();
    if let Some(&id) = table.by_canonical.get(program.canonical.as_str()) {
        if let Some(info) = table.programs.get(&id.0) {
            return Ok((id, info.clone()));
        }
    }
    table.room()?;
    let probe = PanelStream::new(panel, 1).map_err(|e| {
        RaptorError::invalid_parameter(format!("a cross-sectional expression on the bank: {e}"))
    })?;
    let boolean = program.dag.node(program.root).ty.elem == Elem::Bool;
    let info = ProgramInfo {
        canonical: program.canonical.clone(),
        plan: None,
        panel: Some(panel.clone()),
        outputs: 1,
        warmup: program.warmup(),
        boolean,
        reads_volume: panel.fields().contains(&Field::Volume),
        needs_clock: false,
        state_bytes: probe.state_bytes(),
    };
    Ok(table.add(&program.canonical, info))
}

/// The lease on `id`, or none when the program was dropped because nothing held it.
pub fn lease(id: ProgramId) -> Option<Arc<ProgramInfo>> {
    interner().programs.get(&id.0).cloned()
}

/// The program behind an id. Whoever holds an id holds a lease on it too (an `Indicator`, a
/// registry, a bank), so while it is in use it is present; an id with no program reads as an
/// expression that cannot stream, which no cell was ever built for.
pub fn info(id: ProgramId) -> Arc<ProgramInfo> {
    lease(id).unwrap_or_else(|| Arc::new(ProgramInfo::gone()))
}

impl ProgramInfo {
    /// What an id with no program reads as: no output, so [`IndSpec::expr`] refuses it.
    fn gone() -> Self {
        Self {
            canonical: String::new(),
            plan: None,
            panel: None,
            outputs: 0,
            warmup: 0,
            boolean: false,
            reads_volume: false,
            needs_clock: false,
            state_bytes: 0,
        }
    }
}

/// Refuse a clock the program's `at(...)` cannot be cut by (a timeframe that is not a coarser
/// exact multiple of the bars the cell runs on), before any cell is built.
pub fn check_clock(program: ProgramId, clock: &ClockCtx) -> std::result::Result<(), EvalError> {
    match &info(program).plan {
        Some(plan) if plan.has_at() => Stream::bound(plan, clock.clone()).map(|_| ()),
        _ => Ok(()),
    }
}

/// A cell's core: the program's stream and the output it reads.
pub struct ExprCore {
    /// None only for a program with `at(...)` built without a clock (a hand-fed indicator,
    /// which `update_bar` refuses before it feeds one) and for a cross-sectional program.
    stream: Option<Stream>,
    at: bool,
    output: usize,
    /// The base width is being inferred from the first bars; see [`ExprCore::refusal`].
    inferring: bool,
    /// Keeps the program interned while the cell lives.
    _lease: Arc<ProgramInfo>,
}

impl std::fmt::Debug for ExprCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExprCore")
            .field("bars", &self.stream.as_ref().map(Stream::bars))
            .field("output", &self.output)
            .finish()
    }
}

fn stream_refusal(e: EvalError) -> RaptorError {
    RaptorError::invalid_parameter(format!("expression cannot stream here: {e}"))
}

impl ExprCore {
    /// A fresh core for output `output` of `program`. The output index is checked where the
    /// spec is made (`IndSpec::expr`).
    pub fn new(program: ProgramId, output: u8) -> Result<Self> {
        Self::with_clock(program, output, None)
    }

    /// [`ExprCore::new`] on a stream whose zone, bar stamp and session are `clock`; a program
    /// with `at(...)` reads it, any other ignores it. Refuses a clock the program's `at(...)`
    /// cannot be cut by; [`check_clock`] says so beforehand with the typed error.
    pub fn with_clock(program: ProgramId, output: u8, clock: Option<ClockCtx>) -> Result<Self> {
        let info = info(program);
        let clock_infers = clock.as_ref().is_some_and(|c| c.infer_width && c.base_ns.is_none());
        // A cross-sectional program has no series stream: its cells are not fed bar by bar.
        let stream = match (&info.plan, info.needs_clock, clock) {
            (None, ..) => None,
            (Some(plan), false, _) => Some(Stream::new(plan).map_err(stream_refusal)?),
            (Some(plan), true, Some(clock)) => {
                Some(Stream::bound(plan, clock).map_err(stream_refusal)?)
            }
            (Some(_), true, None) => None,
        };
        let inferring = stream.is_some() && info.needs_clock && clock_infers;
        Ok(Self {
            stream,
            at: info.needs_clock,
            output: usize::from(output),
            inferring,
            _lease: info,
        })
    }

    /// Feed one bar; the value after it. Unknown until the program's warm-up has passed.
    pub fn update(&mut self, bar: Ohlc) -> Reading {
        let ts = bar.ts;
        let bar = Bar {
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        };
        let Some(stream) = self.stream.as_mut() else {
            // A program with `at(...)` and no clock: the runtime refuses to build such a cell
            // for a run, and `Indicator.update_bar` refuses the bar.
            return Reading::UNINIT;
        };
        let out = if self.at { stream.update_at(ts, &bar, &[]) } else { stream.update(&bar, &[]) };
        // Only a handle count or a missing timestamp can fail, and neither is possible here:
        // interning refuses a program that reads a handle, and `at` is fed through `update_at`.
        let Some(&value) = out.ok().and_then(|o| o.get(self.output)) else {
            return Reading::UNINIT;
        };
        if stream.ready() {
            Reading { initialized: true, values: [value, 0.0, 0.0] }
        } else {
            Reading::UNINIT
        }
    }

    /// Whether the cell is still working out its base width from the first bars.
    pub fn infers(&self) -> bool {
        self.inferring
    }

    /// After a bar: the refusal of an `at(...)` whose timeframe is no coarser than the bars,
    /// found from the first two stamps when no width was given. Cheap and once: it reads nothing
    /// again after the second bar.
    pub fn refusal(&mut self) -> Option<EvalError> {
        if !self.inferring {
            return None;
        }
        let stream = self.stream.as_ref()?;
        if let Some(e) = stream.width_refusal() {
            self.inferring = false;
            return Some(e.clone());
        }
        if stream.bars() >= 2 {
            self.inferring = false;
        }
        None
    }

    /// Heap this core owns beyond `size_of::<ExprCore>()` (the box it lives in).
    pub fn state_bytes(&self) -> usize {
        self.stream.as_ref().map_or(0, Stream::state_bytes)
    }
}

impl ProgramInfo {
    /// The program is cross-sectional: it runs on a universe, not on one instrument's bars.
    pub fn is_panel(&self) -> bool {
        self.panel.is_some()
    }
}

/// The shared run of a cross-sectional program over a universe: the bars stored for the date in
/// flight, the panel stream, and each member's bar count (for readiness).
pub struct PanelCore {
    stream: PanelStream,
    pending: Vec<Option<Bar>>,
    any: bool,
    bars: Vec<u64>,
    out: Vec<f64>,
    warmup: u64,
    _lease: Arc<ProgramInfo>,
}

impl std::fmt::Debug for PanelCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PanelCore").field("members", &self.bars.len()).finish()
    }
}

impl PanelCore {
    /// A fresh run of `program` (a cross-sectional one) over `members` names.
    pub fn new(program: ProgramId, members: usize) -> Result<Self> {
        let info = info(program);
        let panel = info.panel.as_ref().ok_or_else(|| {
            RaptorError::invalid_parameter("a universe cell runs a cross-sectional expression")
        })?;
        let stream = PanelStream::new(panel, members).map_err(|e| {
            RaptorError::invalid_parameter(format!(
                "a cross-sectional expression over {members} names: {e}"
            ))
        })?;
        Ok(Self {
            stream,
            pending: vec![None; members],
            any: false,
            bars: vec![0; members],
            out: vec![f64::NAN; members],
            warmup: info.warmup as u64,
            _lease: info.clone(),
        })
    }

    /// Keep `bar` as member `member`'s bar of the date in flight. Nothing runs until [`seal`].
    ///
    /// [`seal`]: PanelCore::seal
    pub fn store(&mut self, member: usize, bar: Ohlc) {
        self.pending[member] = Some(Bar {
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        });
        self.any = true;
    }

    /// Step the panel on the stored bars: present members step their time-series functions, the
    /// cross-sections run on the date's row. False (and nothing happens) if no member has a bar.
    pub fn seal(&mut self) -> bool {
        if !self.any {
            return false;
        }
        for (n, b) in self.bars.iter_mut().zip(&self.pending) {
            *n += u64::from(b.is_some());
        }
        // No handle values are given and none are read (interning refuses a named series), so a
        // refusal here is not reachable; the names would read unknown, never a stale value.
        if self.stream.step(&self.pending, &mut self.out).is_err() {
            self.out.fill(f64::NAN);
        }
        self.pending.fill(None);
        self.any = false;
        true
    }

    /// Member `member`'s value after the last seal. Ready once it has seen more bars than the
    /// program's warm-up; NaN when it had no bar at the seal or the cross-section was too thin.
    pub fn reading(&self, member: usize) -> Reading {
        Reading {
            initialized: self.bars[member] > self.warmup,
            values: [self.out[member], 0.0, 0.0],
        }
    }

    /// Heap this run owns beyond its own size.
    pub fn state_bytes(&self) -> usize {
        self.stream.state_bytes()
            + self.pending.capacity() * size_of::<Option<Bar>>()
            + (self.bars.capacity() + self.out.capacity()) * size_of::<f64>()
    }
}

/// Heap one expression cell owns: the box and the stream's state. Equal to what a freshly built
/// core measures.
pub(crate) fn cell_heap_bytes(program: ProgramId) -> usize {
    size_of::<ExprCore>() + info(program).state_bytes
}
