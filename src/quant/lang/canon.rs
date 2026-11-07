//! The canonical form, printed from the DAG alone, and the program hash.
//!
//! Spelling never reaches here: parentheses are re-derived from precedence, defaults are written
//! out, commutative operands come in digest order, floats print shortest-round-trip. The output
//! parses back to a DAG that prints the same string.

use super::dag::{BinOp, Dag, NodeId, Op, UnOp};
use crate::journal::sha256_hex;
use crate::quant::registry::ParamVal;

// Binding strength, loosest first; `ATOM` also covers calls and `.output`.
const OR: u8 = 1;
const AND: u8 = 2;
const NOT: u8 = 3;
const CMP: u8 = 4;
const ADD: u8 = 5;
const MUL: u8 = 6;
const UNARY: u8 = 7;
const POW: u8 = 8;
const ATOM: u8 = 9;

pub fn float(v: f64) -> String {
    format!("{v:?}")
}

/// The binding strength of a node as printed, and the least strength its left and right
/// operands may have without parentheses.
fn shape(dag: &Dag, id: NodeId) -> (u8, u8, u8) {
    match &dag.node(id).op {
        Op::Const(v) if v.is_sign_negative() => (UNARY, 0, 0),
        Op::Const(_)
        | Op::Field(_)
        | Op::Handle(_)
        | Op::Call { .. }
        | Op::Out { .. }
        | Op::At { .. }
        | Op::SessionBar => (ATOM, 0, 0),
        Op::Unary(UnOp::Neg, _) => (UNARY, UNARY, 0),
        Op::Unary(UnOp::Not, _) => (NOT, NOT, 0),
        Op::Binary(op, _, _) => match op {
            BinOp::Or => (OR, OR, OR + 1),
            BinOp::And => (AND, AND, AND + 1),
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
                (CMP, ADD, ADD)
            }
            BinOp::Add | BinOp::Sub => (ADD, ADD, ADD + 1),
            BinOp::Mul | BinOp::Div => (MUL, MUL, MUL + 1),
            BinOp::Pow => (POW, ATOM, UNARY),
        },
    }
}

fn print(dag: &Dag, id: NodeId, min: u8, out: &mut String) {
    let (prec, lmin, rmin) = shape(dag, id);
    let paren = prec < min;
    if paren {
        out.push('(');
    }
    match &dag.node(id).op {
        Op::Const(v) => out.push_str(&float(*v)),
        Op::Field(f) => out.push_str(f.name()),
        Op::Handle(n) => out.push_str(n),
        Op::Unary(UnOp::Neg, x) => {
            out.push('-');
            print(dag, *x, lmin, out);
        }
        Op::Unary(UnOp::Not, x) => {
            out.push_str("not ");
            print(dag, *x, lmin, out);
        }
        Op::Binary(op, a, b) => {
            print(dag, *a, lmin, out);
            out.push(' ');
            out.push_str(op.symbol());
            out.push(' ');
            print(dag, *b, rmin, out);
        }
        Op::Call { def, params, inputs } => {
            out.push_str(def.name);
            out.push('(');
            let mut first = true;
            for i in inputs {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                print(dag, *i, 0, out);
            }
            for p in params {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                match p {
                    ParamVal::Int(v) => out.push_str(&v.to_string()),
                    ParamVal::Float(v) => out.push_str(&float(*v)),
                    ParamVal::Enum(s) => {
                        out.push('"');
                        out.push_str(s);
                        out.push('"');
                    }
                }
            }
            out.push(')');
        }
        Op::SessionBar => out.push_str("session_bar()"),
        Op::At { spec, body } => {
            out.push_str("at(\"");
            out.push_str(&crate::quant::fns::at::timeframe_text(*spec));
            out.push_str("\", ");
            out.push_str(&body.canonical);
            out.push(')');
        }
        Op::Out { call, index } => {
            print(dag, *call, ATOM, out);
            if let Op::Call { def, .. } = &dag.node(*call).op {
                out.push('.');
                out.push_str(def.outputs[*index as usize]);
            }
        }
    }
    if paren {
        out.push(')');
    }
}

pub fn canonical(dag: &Dag, root: NodeId) -> String {
    let mut s = String::new();
    print(dag, root, 0, &mut s);
    s
}

/// `sha256("raptorbt-expr-v1\n" + canonical + function revs)`, lowercase hex. The revs are one
/// `name@rev` line per distinct function called, sorted by name, after a newline.
pub fn program_hash(dag: &Dag, _root: NodeId, canonical: &str) -> String {
    let mut s = String::from("raptorbt-expr-v1\n");
    s.push_str(canonical);
    s.push('\n');
    for (name, rev) in dag.function_revs() {
        s.push_str(&format!("{name}@{rev}\n"));
    }
    sha256_hex(s.as_bytes())
}
