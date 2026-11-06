//! Batch evaluation: the plan over whole columns, buffers from the plan's liveness pool.
//!
//! Elementwise loops are plain zipped slices over one scalar function from [`ops`], so the
//! compiler vectorises them and the streaming step computes the same bits.

use std::mem;

use rayon::prelude::*;

use super::{ops, Bar, Columns, EvalError, Inputs, Loc, Plan, Step};
use crate::quant::lang::BinOp;

/// An operand of an elementwise step: another buffer, or the destination itself (the step
/// computes in place because that input dies here).
enum Opd<'a> {
    Slice(&'a [f64]),
    InPlace,
}

fn operand<'a>(loc: Loc, dst: usize, pool: &'a [Vec<f64>], cols: &[&'a [f64]]) -> Opd<'a> {
    match loc {
        Loc::Slot(s) if s == dst => Opd::InPlace,
        Loc::Slot(s) => Opd::Slice(&pool[s]),
        Loc::Input(k) => Opd::Slice(cols[k]),
    }
}

#[inline(always)]
fn zip1(dst: &mut [f64], a: Opd, f: impl Fn(f64) -> f64) {
    match a {
        Opd::Slice(a) => {
            for (d, x) in dst.iter_mut().zip(a) {
                *d = f(*x);
            }
        }
        Opd::InPlace => {
            for d in dst.iter_mut() {
                *d = f(*d);
            }
        }
    }
}

#[inline(always)]
fn zip2(dst: &mut [f64], a: Opd, b: Opd, f: impl Fn(f64, f64) -> f64) {
    match (a, b) {
        (Opd::Slice(a), Opd::Slice(b)) => {
            for ((d, x), y) in dst.iter_mut().zip(a).zip(b) {
                *d = f(*x, *y);
            }
        }
        (Opd::InPlace, Opd::Slice(b)) => {
            for (d, y) in dst.iter_mut().zip(b) {
                *d = f(*d, *y);
            }
        }
        (Opd::Slice(a), Opd::InPlace) => {
            for (d, x) in dst.iter_mut().zip(a) {
                *d = f(*x, *d);
            }
        }
        (Opd::InPlace, Opd::InPlace) => {
            for d in dst.iter_mut() {
                *d = f(*d, *d);
            }
        }
    }
}

fn zip_bin(op: BinOp, dst: &mut [f64], a: Opd, b: Opd) {
    match op {
        BinOp::Add => zip2(dst, a, b, ops::add),
        BinOp::Sub => zip2(dst, a, b, ops::sub),
        BinOp::Mul => zip2(dst, a, b, ops::mul),
        BinOp::Div => zip2(dst, a, b, ops::div),
        BinOp::Pow => zip2(dst, a, b, ops::pow),
        BinOp::Lt => zip2(dst, a, b, ops::lt),
        BinOp::Le => zip2(dst, a, b, ops::le),
        BinOp::Gt => zip2(dst, a, b, ops::gt),
        BinOp::Ge => zip2(dst, a, b, ops::ge),
        BinOp::Eq => zip2(dst, a, b, ops::eq),
        BinOp::Ne => zip2(dst, a, b, ops::ne),
        BinOp::And => zip2(dst, a, b, ops::and),
        BinOp::Or => zip2(dst, a, b, ops::or),
    }
}

fn slot_of(plan: &Plan, reg: usize) -> usize {
    match plan.d.locs[reg] {
        Loc::Slot(s) => s,
        Loc::Input(_) => unreachable!("register {reg} is a caller's column, not a computed result"),
    }
}

/// Evaluate the plan over whole columns. Returns one column per program output.
///
/// A zero-length column gives zero-length outputs; a program run on fewer bars than its
/// warm-up gives all-NaN outputs. No column at all, a column the program reads that is missing,
/// or columns of different lengths are typed errors.
pub fn eval(plan: &Plan, inputs: &Inputs) -> Result<Columns, EvalError> {
    let (n, cols) = inputs.resolve(plan)?;
    let at_inputs = inputs.at_inputs(plan, n)?;
    let d = &*plan.d;
    let mut nodes = d.instantiate()?;
    let mut ats = match at_inputs {
        Some((_, clock)) => d.instantiate_ats(clock)?,
        None => Vec::new(),
    };
    let mut pool: Vec<Vec<f64>> = vec![vec![0.0; n]; d.n_slots];

    for step in &d.steps {
        match step {
            Step::Load { .. } => {}
            Step::Const { dst, v } => pool[slot_of(plan, *dst)].fill(*v),
            Step::Neg { dst, a } | Step::Not { dst, a } => {
                let ds = slot_of(plan, *dst);
                let mut buf = mem::take(&mut pool[ds]);
                let opd = operand(d.locs[*a], ds, &pool, &cols);
                if matches!(step, Step::Neg { .. }) {
                    zip1(&mut buf, opd, ops::neg);
                } else {
                    zip1(&mut buf, opd, ops::not);
                }
                pool[ds] = buf;
            }
            Step::Bin { op, dst, a, b } => {
                let ds = slot_of(plan, *dst);
                let mut buf = mem::take(&mut pool[ds]);
                let (oa, ob) =
                    (operand(d.locs[*a], ds, &pool, &cols), operand(d.locs[*b], ds, &pool, &cols));
                zip_bin(*op, &mut buf, oa, ob);
                pool[ds] = buf;
            }
            Step::At { dst, ins, fields, at } => {
                // The same state machine a stream steps, bar by bar in order: a value at bar i
                // is made of windows that closed before it, so there is nothing to look ahead
                // to. `ats` is non-empty exactly when `at_inputs` returned the timestamps.
                let ds = slot_of(plan, *dst);
                let mut buf = mem::take(&mut pool[ds]);
                if let Some((ts, _)) = at_inputs {
                    let columns: Vec<&[f64]> = ins
                        .iter()
                        .map(|r| match d.locs[*r] {
                            Loc::Input(k) => cols[k],
                            Loc::Slot(_) => {
                                unreachable!("an at(...) field is loaded, not computed")
                            }
                        })
                        .collect();
                    let state = &mut ats[*at];
                    for (i, out) in buf.iter_mut().enumerate() {
                        let mut bar = Bar::NAN;
                        for (c, f) in columns.iter().zip(fields) {
                            bar.set(*f, c[i]);
                        }
                        *out = state.step(ts[i], &bar);
                    }
                }
                pool[ds] = buf;
            }
            Step::Call { ins, outs, node, .. } => {
                let out_slots: Vec<usize> = outs.iter().map(|r| slot_of(plan, *r)).collect();
                let mut bufs: Vec<Vec<f64>> =
                    out_slots.iter().map(|s| mem::take(&mut pool[*s])).collect();
                {
                    let in_cols: Vec<&[f64]> = ins
                        .iter()
                        .map(|r| match d.locs[*r] {
                            Loc::Slot(s) => pool[s].as_slice(),
                            Loc::Input(k) => cols[k],
                        })
                        .collect();
                    let mut out_cols: Vec<&mut [f64]> =
                        bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
                    nodes[*node].run_batch(&in_cols, &mut out_cols);
                }
                for (s, b) in out_slots.iter().zip(bufs) {
                    pool[*s] = b;
                }
            }
        }
    }

    Ok(d.out_regs
        .iter()
        .map(|r| match d.locs[*r] {
            Loc::Input(k) => cols[k].to_vec(),
            Loc::Slot(s) => mem::take(&mut pool[s]),
        })
        .collect())
}

/// Evaluate the plan over many instruments in parallel. Each instrument is independent and
/// results come back in input order, so the output does not depend on the thread count.
pub fn eval_many(plan: &Plan, inputs: &[Inputs]) -> Vec<Result<Columns, EvalError>> {
    inputs.par_iter().map(|i| eval(plan, i)).collect()
}
