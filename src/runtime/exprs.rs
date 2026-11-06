//! Expression cells: a checked expression as a bank indicator.
//!
//! An expression reaches the bank as a [`ProgramId`] from a process-wide, append-only interner
//! keyed on the program's full canonical text (not a hash), so two registrations share a cell
//! exactly when their canonical expressions are equal, and an id means the same program for the
//! life of the process. The interner only grows; [`MAX_PROGRAMS`] bounds it.
//!
//! A cell runs the program's own [`Stream`] over the bars the bank feeds it, the same step the
//! batch evaluator runs, so an expression in a strategy gives the numbers `expr(...).eval` gives
//! on the same bars, backtest and live alike. A program that reads `volume` gets the bar's real
//! volume from every path into the bank (primary bars, trade-built bars and composite bars all
//! carry it); only a hand-fed `update_bar` can lack one, and it is refused there.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::core::error::RaptorError;
use crate::core::Result;
use crate::quant::eval::{Bar, ClockCtx, PanelPlan, PanelStream, Plan, Stream};
use crate::quant::lang::{Elem, Field, Program};

use super::indicators::{Ohlc, Reading};

/// Most distinct programs one process interns.
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
    programs: Vec<Arc<ProgramInfo>>,
}

fn interner() -> &'static Mutex<Interner> {
    static PROGRAMS: OnceLock<Mutex<Interner>> = OnceLock::new();
    PROGRAMS.get_or_init(Mutex::default)
}

/// Intern `program` (run by `plan`). The same canonical text gives the same id.
///
/// Fails when the plan cannot be run as a stream or the interner is full; nothing about the
/// program is checked again when a cell is built, so every refusal is here.
pub fn intern(program: &Program, plan: &Plan) -> Result<ProgramId> {
    let mut table = interner().lock().expect("program interner poisoned");
    if let Some(&id) = table.by_canonical.get(program.canonical.as_str()) {
        return Ok(id);
    }
    if table.programs.len() >= MAX_PROGRAMS {
        return Err(RaptorError::invalid_parameter(format!(
            "expression table is full: {MAX_PROGRAMS} distinct expressions in one process"
        )));
    }
    if !plan.handles().is_empty() {
        return Err(RaptorError::invalid_parameter(
            "an indicator expression reads bar fields only, not named series",
        ));
    }
    let state_bytes = plan
        .state_bytes()
        .map_err(|e| RaptorError::invalid_parameter(format!("expression cannot stream: {e}")))?;
    let boolean = program.dag.node(program.root).ty.elem == Elem::Bool;
    let id = ProgramId(table.programs.len() as u32);
    table.programs.push(Arc::new(ProgramInfo {
        canonical: program.canonical.clone(),
        plan: Some(plan.clone()),
        panel: None,
        outputs: plan.n_outputs(),
        warmup: program.warmup(),
        boolean,
        reads_volume: plan.fields().contains(&Field::Volume),
        needs_clock: plan.has_at(),
        state_bytes,
    }));
    table.by_canonical.insert(program.canonical.as_str().into(), id);
    Ok(id)
}

/// Intern a cross-sectional `program` (run by `panel`). Its cells are one per member of a universe
/// (`IndInput::Universe`) and share one [`PanelCore`], which steps once per bar of the universe.
///
/// Fails for what the bank cannot feed: `at(...)`, a supplied series (a group label) or several
/// outputs.
pub fn intern_panel(program: &Program, panel: &PanelPlan) -> Result<ProgramId> {
    let mut table = interner().lock().expect("program interner poisoned");
    if let Some(&id) = table.by_canonical.get(program.canonical.as_str()) {
        return Ok(id);
    }
    if table.programs.len() >= MAX_PROGRAMS {
        return Err(RaptorError::invalid_parameter(format!(
            "expression table is full: {MAX_PROGRAMS} distinct expressions in one process"
        )));
    }
    let probe = PanelStream::new(panel, 1).map_err(|e| {
        RaptorError::invalid_parameter(format!("a cross-sectional expression on the bank: {e}"))
    })?;
    let boolean = program.dag.node(program.root).ty.elem == Elem::Bool;
    let id = ProgramId(table.programs.len() as u32);
    table.programs.push(Arc::new(ProgramInfo {
        canonical: program.canonical.clone(),
        plan: None,
        panel: Some(panel.clone()),
        outputs: 1,
        warmup: program.warmup(),
        boolean,
        reads_volume: panel.fields().contains(&Field::Volume),
        needs_clock: false,
        state_bytes: probe.state_bytes(),
    }));
    table.by_canonical.insert(program.canonical.as_str().into(), id);
    Ok(id)
}

/// The program behind an id. An id comes only from [`intern`], so it is always present.
pub fn info(id: ProgramId) -> Arc<ProgramInfo> {
    interner().lock().expect("program interner poisoned").programs[id.index()].clone()
}

/// A cell's core: the program's stream and the output it reads.
pub struct ExprCore {
    /// None only for a program with `at(...)` built without a clock (a hand-fed indicator,
    /// which `update_bar` refuses before it feeds one).
    stream: Option<Stream>,
    at: bool,
    output: usize,
}

impl std::fmt::Debug for ExprCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExprCore")
            .field("bars", &self.stream.as_ref().map(Stream::bars))
            .field("output", &self.output)
            .finish()
    }
}

impl ExprCore {
    /// A fresh core for output `output` of `program`. The output index is checked where the
    /// spec is made (`IndSpec::expr`).
    pub fn new(program: ProgramId, output: u8) -> Self {
        Self::with_clock(program, output, None)
    }

    /// [`ExprCore::new`] on a stream whose zone, bar stamp and session are `clock`; a program
    /// with `at(...)` reads it, any other ignores it.
    pub fn with_clock(program: ProgramId, output: u8, clock: Option<ClockCtx>) -> Self {
        let info = info(program);
        // A cross-sectional program has no series stream: its cells are not fed bar by bar.
        let stream = match (&info.plan, info.needs_clock, clock) {
            (None, ..) => None,
            (Some(plan), false, _) => {
                Some(Stream::new(plan).expect("a program is probed as a stream when interned"))
            }
            (Some(plan), true, Some(clock)) => Some(
                Stream::bound(plan, clock).expect("a program is probed as a stream when interned"),
            ),
            (Some(_), true, None) => None,
        };
        Self { stream, at: info.needs_clock, output: usize::from(output) }
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
            // Unreachable: the runtime refuses to build a cell without its clock, and
            // `Indicator.update_bar` refuses a program that needs one.
            return Reading::UNINIT;
        };
        let out = if self.at { stream.update_at(ts, &bar, &[]) } else { stream.update(&bar, &[]) };
        let value =
            out.expect("a program with no named series takes no handle values")[self.output];
        if stream.ready() {
            Reading { initialized: true, values: [value, 0.0, 0.0] }
        } else {
            Reading::UNINIT
        }
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
}

impl std::fmt::Debug for PanelCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PanelCore").field("members", &self.bars.len()).finish()
    }
}

impl PanelCore {
    /// A fresh run of `program` (a cross-sectional one) over `members` names.
    pub fn new(program: ProgramId, members: usize) -> Self {
        let info = info(program);
        let panel = info.panel.as_ref().expect("a panel core is built for a panel program");
        Self {
            stream: PanelStream::new(panel, members)
                .expect("a program is probed as a panel stream when interned"),
            pending: vec![None; members],
            any: false,
            bars: vec![0; members],
            out: vec![f64::NAN; members],
            warmup: info.warmup as u64,
        }
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
        self.stream
            .step(&self.pending, &mut self.out)
            .expect("a program with no named series takes no handle values");
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
