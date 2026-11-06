//! Types and the checker: AST against the registry, building the DAG as it goes.
//!
//! `Ty` carries shape, element and clock from the start so the cross-sectional (`Panel`) and
//! multi-timeframe (`Clock::Tf`) slices add rules, not fields. Today only `Scalar`/`Series`,
//! `F64`/`Bool` and `Base` are produced; a `Panel` input is accepted from an environment
//! handle and propagates; `at(...)` is refused as unsupported.

use std::collections::BTreeMap;

use super::dag::{BinOp, Dag, Field, NodeId, UnOp};
use super::parser::{Expr, ExprKind};
use super::{suggest, ErrorCode, ExprError, MAX_PERIOD};
use crate::quant::registry::{self, FnDef, OutKind, ParamDef, ParamKind, ParamVal};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Shape {
    Scalar,
    Series,
    Panel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elem {
    F64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clock {
    Base,
    /// A coarser timeframe, by its spec (`"1d"`).
    Tf(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ty {
    pub shape: Shape,
    pub elem: Elem,
    pub clock: Clock,
}

impl Ty {
    pub fn scalar() -> Ty {
        Ty { shape: Shape::Scalar, elem: Elem::F64, clock: Clock::Base }
    }

    pub fn series() -> Ty {
        Ty { shape: Shape::Series, elem: Elem::F64, clock: Clock::Base }
    }

    fn describe(&self) -> &'static str {
        match (self.shape, self.elem) {
            (Shape::Scalar, Elem::F64) => "a number",
            (Shape::Scalar, Elem::Bool) => "a boolean constant",
            (Shape::Series, Elem::F64) => "a numeric series",
            (Shape::Series, Elem::Bool) => "a boolean series",
            (Shape::Panel, Elem::F64) => "a numeric panel",
            (Shape::Panel, Elem::Bool) => "a boolean panel",
        }
    }
}

/// A named series the caller supplies (how `h_ab12` chains reference an earlier program).
#[derive(Debug, Clone)]
pub struct Handle {
    pub ty: Ty,
    pub warm: usize,
}

/// What a name or `$n` in the source means.
#[derive(Debug, Clone, Default)]
pub struct Env {
    pub params: BTreeMap<String, f64>,
    pub series: BTreeMap<String, Handle>,
}

struct Checked {
    id: NodeId,
    /// The call this value is, before any `.output` selection.
    call: Option<&'static FnDef>,
}

fn plain(id: NodeId) -> Result<Checked, ExprError> {
    Ok(Checked { id, call: None })
}

struct Checker<'a> {
    src: &'a str,
    env: &'a Env,
    dag: Dag,
}

/// The constant a parameter argument denotes: a number, `$n`, or a negated one.
enum Arg<'e> {
    Const(f64),
    Str(&'e str),
    Value(&'e Expr),
}

pub fn check(ast: &Expr, src: &str, env: &Env) -> Result<(Dag, NodeId), ExprError> {
    let mut c = Checker { src, env, dag: Dag::new() };
    let root = c.expr(ast)?;
    Ok((c.dag, root.id))
}

impl Checker<'_> {
    fn err(&self, code: ErrorCode, pos: usize, msg: impl Into<String>) -> ExprError {
        ExprError::new(self.src, code, pos, msg)
    }

    fn type_err(&self, pos: usize, msg: impl Into<String>) -> ExprError {
        self.err(ErrorCode::Type, pos, msg)
    }

    fn param_value(&self, name: &str, pos: usize) -> Result<f64, ExprError> {
        self.env.params.get(name).copied().ok_or_else(|| {
            self.err(ErrorCode::UnboundParameter, pos, format!("parameter `${name}` has no value"))
                .with_valid(suggest(name, self.env.params.keys().map(String::as_str)))
        })
    }

    fn const_of(&self, e: &Expr) -> Result<Option<f64>, ExprError> {
        Ok(match &e.kind {
            ExprKind::Num(v) => Some(*v),
            ExprKind::Param(n) => Some(self.param_value(n, e.start)?),
            ExprKind::Unary(UnOp::Neg, x) => self.const_of(x)?.map(|v| -v),
            _ => None,
        })
    }

    fn arg<'e>(&self, e: &'e Expr) -> Result<Arg<'e>, ExprError> {
        Ok(match (&e.kind, self.const_of(e)?) {
            (_, Some(v)) => Arg::Const(v),
            (ExprKind::Str(s), _) => Arg::Str(s),
            _ => Arg::Value(e),
        })
    }

    /// A value usable as an operand: never an unselected multi-output call.
    fn value(&mut self, e: &Expr) -> Result<NodeId, ExprError> {
        let c = self.expr(e)?;
        if let Some(def) = c.call.filter(|d| d.outputs.len() > 1) {
            return Err(self
                .type_err(
                    e.start,
                    format!(
                        "`{}` has several outputs; pick one with `.name` (a bare multi-output call is only valid as the whole expression)",
                        def.name
                    ),
                )
                .with_valid(def.outputs.iter().map(|s| s.to_string()).collect()));
        }
        Ok(c.id)
    }

    fn expr(&mut self, e: &Expr) -> Result<Checked, ExprError> {
        match &e.kind {
            ExprKind::Num(v) => plain(self.dag.constant(*v)),
            ExprKind::Str(_) => {
                Err(self.type_err(e.start, "a string is only valid as an enum parameter"))
            }
            ExprKind::Param(n) => {
                let v = self.param_value(n, e.start)?;
                plain(self.dag.constant(v))
            }
            ExprKind::Ident(name) => self.ident(name, e.start).and_then(plain),
            ExprKind::Unary(UnOp::Neg, x) if self.const_of(x)?.is_some() => {
                // A negated constant is one constant (exact), with no orphan node behind it.
                let v = self.const_of(x)?.unwrap_or(0.0);
                plain(self.dag.constant(-v))
            }
            ExprKind::Unary(op, x) => {
                let id = self.value(x)?;
                let ty = self.dag.node(id).ty.clone();
                let want = if *op == UnOp::Not { Elem::Bool } else { Elem::F64 };
                if ty.elem != want {
                    let sym = if *op == UnOp::Not { "not" } else { "-" };
                    return Err(self.type_err(
                        x.start,
                        format!(
                            "`{sym}` needs a {} operand, found {}",
                            elem_name(want),
                            ty.describe()
                        ),
                    ));
                }
                plain(self.dag.unary(*op, id, ty))
            }
            ExprKind::Binary(op, l, r) => self.binary(*op, l, r).and_then(plain),
            ExprKind::Call { name, args } => self.call(e, name, args),
            ExprKind::Select(base, name, name_pos) => {
                let b = self.expr(base)?;
                let Some(def) = b.call else {
                    return Err(
                        self.type_err(*name_pos, "`.name` selects an output of a function call")
                    );
                };
                let Some(idx) = def.outputs.iter().position(|o| o == name) else {
                    return Err(self
                        .type_err(*name_pos, format!("`{}` has no output `{name}`", def.name))
                        .with_valid(def.outputs.iter().map(|s| s.to_string()).collect()));
                };
                // A single-output call's `.value` is the call itself.
                let id = if def.outputs.len() == 1 { b.id } else { self.dag.out(b.id, idx as u8) };
                plain(id)
            }
        }
    }

    fn ident(&mut self, name: &str, pos: usize) -> Result<NodeId, ExprError> {
        if let Some(f) = Field::from_name(name) {
            return Ok(self.dag.field(f));
        }
        if let Some(h) = self.env.series.get(name) {
            return Ok(self.dag.handle(name, h.ty.clone(), h.warm));
        }
        if registry::lookup(name).is_some() {
            return Err(
                self.type_err(pos, format!("`{name}` is a function; call it as `{name}(...)`"))
            );
        }
        let known =
            Field::ALL.iter().map(|f| f.name()).chain(self.env.series.keys().map(String::as_str));
        Err(self.type_err(pos, format!("unknown name `{name}`")).with_valid(suggest(name, known)))
    }

    fn binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<NodeId, ExprError> {
        let a = self.value(l)?;
        let b = self.value(r)?;
        let (ta, tb) = (self.dag.node(a).ty.clone(), self.dag.node(b).ty.clone());
        let (operand, result) = match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow => {
                (Elem::F64, Elem::F64)
            }
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => (Elem::F64, Elem::Bool),
            BinOp::Eq | BinOp::Ne => (ta.elem, Elem::Bool),
            BinOp::And | BinOp::Or => (Elem::Bool, Elem::Bool),
        };
        for (t, e) in [(&ta, l), (&tb, r)] {
            if t.elem != operand {
                return Err(self.type_err(
                    e.start,
                    format!(
                        "`{}` needs {} operands, found {}",
                        op.symbol(),
                        elem_name(operand),
                        t.describe()
                    ),
                ));
            }
        }
        let ty = self.join(&ta, &tb, r.start)?;
        Ok(self.dag.binary(op, a, b, Ty { elem: result, ..ty }))
    }

    /// Shape is the larger of the two; the clock of whichever side is not a constant, and equal
    /// when both are not.
    fn join(&self, a: &Ty, b: &Ty, pos: usize) -> Result<Ty, ExprError> {
        let clock = match (a.shape, b.shape) {
            (Shape::Scalar, _) => b.clock.clone(),
            (_, Shape::Scalar) => a.clock.clone(),
            _ if a.clock == b.clock => a.clock.clone(),
            _ => {
                return Err(self.type_err(
                    pos,
                    "operands are on different clocks; bring one across with `at(...)`",
                ))
            }
        };
        Ok(Ty { shape: a.shape.max(b.shape), elem: a.elem, clock })
    }

    fn call(&mut self, e: &Expr, name: &str, args: &[Expr]) -> Result<Checked, ExprError> {
        let Some(def) = registry::lookup(name) else {
            if name == "at" {
                return Err(self.err(
                    ErrorCode::UnsupportedHere,
                    e.start,
                    "`at(...)` (multi-timeframe) is not available yet",
                ));
            }
            return Err(self
                .err(ErrorCode::UnknownFunction, e.start, format!("unknown function `{name}`"))
                .with_valid(suggest(name, registry::all().map(|d| d.name))));
        };
        let sig = signature(def);
        let classified: Vec<Arg> = args.iter().map(|a| self.arg(a)).collect::<Result<_, _>>()?;
        let (n_in, n_p) = (def.inputs.len(), def.params.len());

        // Leading series arguments are inputs; a constant in input position is promoted only
        // when there are more arguments than parameters.
        let lead = classified.iter().take_while(|a| matches!(a, Arg::Value(_))).count().min(n_in);
        let k = lead.max(args.len().saturating_sub(n_p));
        if k > n_in || args.len() - k > n_p {
            return Err(self
                .type_err(e.start, format!("wrong number of arguments; expected {sig}"))
                .with_valid(def.params.iter().map(|p| p.name.to_string()).collect()));
        }

        let mut inputs = Vec::with_capacity(n_in);
        for (i, inp) in def.inputs.iter().enumerate() {
            if i < k {
                let id = self.value(&args[i])?;
                let t = &self.dag.node(id).ty;
                let want = registry::input_elem(def, i);
                if t.elem != want {
                    return Err(self.type_err(
                        args[i].start,
                        format!(
                            "input `{}` of `{name}` must be {}, found {}",
                            inp.name,
                            elem_name(want),
                            t.describe()
                        ),
                    ));
                }
                inputs.push(id);
            } else if let Some(f) = inp.default_field.and_then(Field::from_name) {
                inputs.push(self.dag.field(f));
            } else {
                return Err(self.type_err(
                    e.start,
                    format!("`{name}` needs input `{}`; expected {sig}", inp.name),
                ));
            }
        }

        let mut params = Vec::with_capacity(n_p);
        for (j, p) in def.params.iter().enumerate() {
            match classified.get(k + j) {
                Some(a) => params.push(self.param(def, p, a, &args[k + j])?),
                None => match p.default {
                    Some(v) => params.push(v),
                    None => {
                        return Err(self.type_err(
                            e.end,
                            format!("`{name}` needs parameter `{}`; expected {sig}", p.name),
                        ))
                    }
                },
            }
        }

        let mut joined: Option<Ty> = None;
        for &i in &inputs {
            let t = self.dag.node(i).ty.clone();
            joined = Some(match joined {
                None => t,
                Some(a) => self.join(&a, &t, e.start)?,
            });
        }
        let mut ty = joined.unwrap_or_else(Ty::series);
        ty.shape = ty.shape.max(Shape::Series);
        ty.elem = if def.out_kind == OutKind::Bool { Elem::Bool } else { Elem::F64 };
        let id = self.dag.call(def, params, inputs, ty);
        Ok(Checked { id, call: Some(def) })
    }

    fn param(&self, def: &FnDef, p: &ParamDef, a: &Arg, at: &Expr) -> Result<ParamVal, ExprError> {
        let who = format!("parameter `{}` of `{}`", p.name, def.name);
        match (a, p.kind) {
            (Arg::Value(v), _) => Err(self.type_err(
                v.start,
                format!("{who} must be a constant or a `$name`; a series is not a {}", p.name),
            )),
            (Arg::Str(s), ParamKind::Enum(values)) => match values.iter().find(|v| **v == *s) {
                Some(v) => Ok(ParamVal::Enum(v)),
                None => Err(self
                    .type_err(at.start, format!("{who} has no value \"{s}\""))
                    .with_valid(suggest(s, values.iter().copied()))),
            },
            (Arg::Str(_), _) | (Arg::Const(_), ParamKind::Enum(_)) => {
                Err(self.type_err(at.start, format!("{who} has the wrong kind of value")))
            }
            (Arg::Const(v), ParamKind::Int { min, max }) => {
                if v.fract() != 0.0 {
                    return Err(
                        self.type_err(at.start, format!("{who} must be an integer, found {v}"))
                    );
                }
                let max = max.min(MAX_PERIOD);
                if *v < min as f64 || *v > max as f64 {
                    return Err(self.type_err(
                        at.start,
                        format!("{who} must be between {min} and {max}, found {v}"),
                    ));
                }
                Ok(ParamVal::Int(*v as i64))
            }
            (Arg::Const(v), ParamKind::Float { min, max }) => {
                if *v < min || *v > max {
                    return Err(self.type_err(
                        at.start,
                        format!("{who} must be between {min} and {max}, found {v}"),
                    ));
                }
                Ok(ParamVal::Float(*v))
            }
        }
    }
}

fn elem_name(e: Elem) -> &'static str {
    match e {
        Elem::F64 => "numeric",
        Elem::Bool => "boolean",
    }
}

/// `name(input, ..., param, ...)` for messages.
fn signature(def: &FnDef) -> String {
    let parts: Vec<String> =
        def.inputs
            .iter()
            .map(|i| {
                if i.default_field.is_some() {
                    format!("[{}]", i.name)
                } else {
                    i.name.to_string()
                }
            })
            .chain(def.params.iter().map(|p| p.name.to_string()))
            .collect();
    format!("{}({})", def.name, parts.join(", "))
}
