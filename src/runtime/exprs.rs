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
use crate::quant::eval::{Bar, Plan, Stream};
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
    plan: Plan,
    /// Output columns: more than one only for an unselected multi-output call.
    pub outputs: usize,
    /// Leading bars that are unknown on clean data.
    pub warmup: usize,
    /// The value is a Kleene boolean (1, 0, unknown), not a number.
    pub boolean: bool,
    pub reads_volume: bool,
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
        plan: plan.clone(),
        outputs: plan.n_outputs(),
        warmup: program.warmup(),
        boolean,
        reads_volume: plan.fields().contains(&Field::Volume),
        state_bytes,
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
    stream: Stream,
    output: usize,
}

impl std::fmt::Debug for ExprCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExprCore")
            .field("bars", &self.stream.bars())
            .field("output", &self.output)
            .finish()
    }
}

impl ExprCore {
    /// A fresh core for output `output` of `program`. The output index is checked where the
    /// spec is made (`IndSpec::expr`).
    pub fn new(program: ProgramId, output: u8) -> Self {
        let info = info(program);
        let stream =
            Stream::new(&info.plan).expect("a program is probed as a stream when interned");
        Self { stream, output: usize::from(output) }
    }

    /// Feed one bar; the value after it. Unknown until the program's warm-up has passed.
    pub fn update(&mut self, bar: Ohlc) -> Reading {
        let bar = Bar {
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        };
        let value = self
            .stream
            .update(&bar, &[])
            .expect("a program with no named series takes no handle values")[self.output];
        if self.stream.ready() {
            Reading { initialized: true, values: [value, 0.0, 0.0] }
        } else {
            Reading::UNINIT
        }
    }

    /// Heap this core owns beyond `size_of::<ExprCore>()` (the box it lives in).
    pub fn state_bytes(&self) -> usize {
        self.stream.state_bytes()
    }
}

/// Heap one expression cell owns: the box and the stream's state. Equal to what a freshly built
/// core measures.
pub(crate) fn cell_heap_bytes(program: ProgramId) -> usize {
    size_of::<ExprCore>() + info(program).state_bytes
}
