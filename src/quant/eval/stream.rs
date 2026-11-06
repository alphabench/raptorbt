//! Streaming evaluation: a flat register file, one pass over the plan's steps per bar.
//!
//! Each kernel is one virtual call per bar; stateless steps are inline matches over the same
//! [`ops`] the batch loops use. Nothing allocates after construction.

use super::{ops, Bar, EvalError, Plan, Source, Step};
use crate::quant::kernel::{Node, Regs};

pub struct Stream {
    plan: Plan,
    regs: Regs,
    nodes: Vec<Box<dyn Node>>,
    out: Vec<f64>,
    bars: usize,
}

impl Stream {
    pub fn new(plan: &Plan) -> Result<Stream, EvalError> {
        let mut s = Stream {
            plan: plan.clone(),
            regs: Regs::new(plan.d.n_regs),
            nodes: plan.d.instantiate()?,
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
    }

    /// Bars consumed so far.
    pub fn bars(&self) -> usize {
        self.bars
    }

    /// True once the root can be known: the bar count has passed the static warm-up, so the
    /// latest output is the first non-NaN on clean data or later.
    pub fn ready(&self) -> bool {
        self.bars > self.plan.d.warm
    }

    /// Feed one bar (and one value per plan handle, in [`Plan::handles`] order) and read the
    /// program's outputs for it.
    pub fn update(&mut self, bar: &Bar, handles: &[f64]) -> Result<&[f64], EvalError> {
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
