//! Streaming evaluation: a flat register file, one pass over the plan's steps per bar.
//!
//! Each kernel is one virtual call per bar; stateless steps are inline matches over the same
//! [`ops`] the batch loops use. Nothing allocates after construction, except what an `at(...)`
//! window builder does with a session bound (see `fns::at`).

use super::{ops, Bar, ClockCtx, EvalError, Plan, Source, Step};
use crate::quant::fns::at::AtState;
use crate::quant::kernel::{Node, Regs};

pub struct Stream {
    plan: Plan,
    regs: Regs,
    nodes: Vec<Box<dyn Node>>,
    /// The program's `at(...)` states and the clock they were cut by.
    ats: Vec<AtState>,
    clock: Option<ClockCtx>,
    out: Vec<f64>,
    bars: usize,
}

impl Stream {
    /// A stream over a program with no `at(...)`. One that has one needs [`Stream::bound`]: it
    /// is refused here, never given a default zone.
    pub fn new(plan: &Plan) -> Result<Stream, EvalError> {
        if plan.has_at() {
            return Err(EvalError::NoClock);
        }
        Self::build(plan, None)
    }

    /// A stream whose `at(...)` windows are cut as `clock` says. Feed it with
    /// [`Stream::update_at`].
    pub fn bound(plan: &Plan, clock: ClockCtx) -> Result<Stream, EvalError> {
        Self::build(plan, Some(clock))
    }

    fn build(plan: &Plan, clock: Option<ClockCtx>) -> Result<Stream, EvalError> {
        let ats = match &clock {
            Some(c) => plan.d.instantiate_ats(c)?,
            None => Vec::new(),
        };
        let mut s = Stream {
            plan: plan.clone(),
            regs: Regs::new(plan.d.n_regs),
            nodes: plan.d.instantiate()?,
            ats,
            clock,
            out: vec![f64::NAN; plan.d.out_regs.len()],
            bars: 0,
        };
        s.seed_constants();
        Ok(s)
    }

    /// Constants are written once; no step overwrites their registers.
    fn seed_constants(&mut self) {
        for step in &self.plan.d.steps {
            if let Step::Const { dst, v } = step {
                self.regs.set(*dst, *v);
            }
        }
    }

    /// Back to the state after construction (allocates: not for the per-bar path). Used to
    /// prime a stream from history, then feed live bars.
    pub fn reset(&mut self) -> Result<(), EvalError> {
        self.regs = Regs::new(self.plan.d.n_regs);
        self.nodes = self.plan.d.instantiate()?;
        if let Some(clock) = &self.clock {
            self.ats = self.plan.d.instantiate_ats(clock)?;
        }
        self.out.fill(f64::NAN);
        self.bars = 0;
        self.seed_constants();
        Ok(())
    }

    /// Heap this stream owns, counted at capacity: the register file, the output row, the node
    /// list and each node's allocation, ring and kernel state. The plan is shared and not
    /// counted; a stream's own struct is not either (it sits wherever the caller put it).
    pub fn state_bytes(&self) -> usize {
        std::mem::size_of::<f64>() * (self.regs.capacity() + self.out.capacity())
            + self.nodes.capacity() * std::mem::size_of::<Box<dyn Node>>()
            + self.nodes.iter().map(|n| n.state_bytes()).sum::<usize>()
            + self.ats.capacity() * std::mem::size_of::<AtState>()
            + self.ats.iter().map(|a| a.state_bytes()).sum::<usize>()
    }

    /// Bars consumed so far.
    pub fn bars(&self) -> usize {
        self.bars
    }

    /// True once the root can be known: the bar count has passed the static warm-up and every
    /// `at(...)` body is past its own, so the latest output is the first non-NaN on clean data
    /// or later.
    pub fn ready(&self) -> bool {
        self.bars > self.plan.d.warm && self.ats.iter().all(AtState::ready)
    }

    /// Feed one bar (and one value per plan handle, in [`Plan::handles`] order) and read the
    /// program's outputs for it. A program with `at(...)` is fed through [`Stream::update_at`].
    pub fn update(&mut self, bar: &Bar, handles: &[f64]) -> Result<&[f64], EvalError> {
        self.step(None, bar, handles)
    }

    /// [`Stream::update`] for the bar stamped `ts`, as the stream's clock reads stamps.
    pub fn update_at(&mut self, ts: i64, bar: &Bar, handles: &[f64]) -> Result<&[f64], EvalError> {
        self.step(Some(ts), bar, handles)
    }

    fn step(&mut self, ts: Option<i64>, bar: &Bar, handles: &[f64]) -> Result<&[f64], EvalError> {
        let d = &*self.plan.d;
        if handles.len() != d.handle_names.len() {
            return Err(EvalError::HandleCount {
                expected: d.handle_names.len(),
                got: handles.len(),
            });
        }
        for step in &d.steps {
            match step {
                Step::Const { .. } => {}
                Step::Load { dst, input } => {
                    let v = match d.inputs[*input] {
                        Source::Field(f) => bar.get(f),
                        Source::Handle(k) => handles[k],
                    };
                    self.regs.set(*dst, v);
                }
                Step::Neg { dst, a } => self.regs.set(*dst, ops::neg(self.regs.get(*a))),
                Step::Not { dst, a } => self.regs.set(*dst, ops::not(self.regs.get(*a))),
                Step::Bin { op, dst, a, b } => {
                    let v = ops::bin(*op, self.regs.get(*a), self.regs.get(*b));
                    self.regs.set(*dst, v);
                }
                Step::At { dst, ins, fields, at } => {
                    let ts = ts.ok_or(EvalError::NoTimestamps)?;
                    let mut b = Bar::NAN;
                    for (r, f) in ins.iter().zip(fields) {
                        b.set(*f, self.regs.get(*r));
                    }
                    let v = self.ats[*at].step(ts, &b);
                    self.regs.set(*dst, v);
                }
                Step::Call { node, .. } => self.nodes[*node].step(&mut self.regs),
            }
        }
        for (o, r) in self.out.iter_mut().zip(&d.out_regs) {
            *o = self.regs.get(*r);
        }
        self.bars += 1;
        Ok(&self.out)
    }
}
