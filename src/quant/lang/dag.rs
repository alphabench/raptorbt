//! The hash-consed DAG. A node's identity is a sha256 of its structure, so equal sub-expressions
//! are one node however they were spelled, and commutative operands sort by that digest rather
//! than by creation order. Children are always created before their parents: node ids are a
//! valid topological schedule, and every node is reachable from the root.
//!
//! No rewrite here changes a value: constants are never folded across operations, addition and
//! multiplication are never re-associated, and the only simplification is `-(c)` to the constant
//! `-c`, which is exact.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::types::Ty;
use super::Program;
use crate::data::BarSpec;
use crate::quant::registry::{FnDef, Memory, ParamVal};

pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Open,
    High,
    Low,
    Close,
    Volume,
}

impl Field {
    pub const ALL: [Field; 5] = [Field::Open, Field::High, Field::Low, Field::Close, Field::Volume];

    pub fn name(self) -> &'static str {
        match self {
            Field::Open => "open",
            Field::High => "high",
            Field::Low => "low",
            Field::Close => "close",
            Field::Volume => "volume",
        }
    }

    pub fn from_name(name: &str) -> Option<Field> {
        Field::ALL.into_iter().find(|f| f.name() == name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Pow => "**",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::And => "and",
            BinOp::Or => "or",
        }
    }

    pub fn is_commutative(self) -> bool {
        matches!(self, BinOp::Add | BinOp::Mul | BinOp::Eq | BinOp::Ne | BinOp::And | BinOp::Or)
    }
}

#[derive(Debug, Clone)]
pub enum Op {
    Const(f64),
    Field(Field),
    /// A named series the caller supplies (a program built earlier, referenced by its handle).
    Handle(String),
    Unary(UnOp, NodeId),
    Binary(BinOp, NodeId, NodeId),
    /// `inputs` are in the function's input order, defaults materialised; `params` in its
    /// parameter order, defaults materialised.
    Call {
        def: &'static FnDef,
        params: Vec<ParamVal>,
        inputs: Vec<NodeId>,
    },
    /// Output `index` of a multi-output call; it aliases that output, no computation.
    Out {
        call: NodeId,
        index: u8,
    },
    /// `at(tf, body)`: the program `body` evaluated on `tf` bars of the base clock and held on
    /// the base clock. The body is a program of its own with its own DAG: nothing in it is
    /// shared with this one.
    At {
        spec: BarSpec,
        body: Arc<Program>,
    },
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub ty: Ty,
    /// Leading NaN bars on clean data: `max(warm(inputs)) + warmup(params)`.
    pub warm: usize,
    pub digest: [u8; 32],
}

impl Node {
    /// Number of output registers the node writes: a call's outputs; one otherwise.
    pub fn n_outputs(&self) -> usize {
        match &self.op {
            Op::Call { def, .. } => def.outputs.len(),
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dag {
    nodes: Vec<Node>,
    index: HashMap<[u8; 32], NodeId>,
}

fn digest_of(op: &Op, nodes: &[Node]) -> [u8; 32] {
    let mut h = Sha256::new();
    match op {
        Op::Const(v) => {
            h.update([1]);
            h.update(v.to_bits().to_le_bytes());
        }
        Op::Field(f) => {
            h.update([2]);
            h.update(f.name());
        }
        Op::Handle(name) => {
            h.update([3]);
            h.update(name);
        }
        Op::Unary(op, x) => {
            h.update([4, *op as u8]);
            h.update(nodes[*x].digest);
        }
        Op::Binary(op, a, b) => {
            h.update([5, *op as u8]);
            h.update(nodes[*a].digest);
            h.update(nodes[*b].digest);
        }
        Op::Call { def, params, inputs } => {
            h.update([6]);
            h.update(def.name);
            h.update([0]);
            h.update(def.rev.to_le_bytes());
            for p in params {
                match p {
                    ParamVal::Int(v) => {
                        h.update([0]);
                        h.update(v.to_le_bytes());
                    }
                    ParamVal::Float(v) => {
                        h.update([1]);
                        h.update(v.to_bits().to_le_bytes());
                    }
                    ParamVal::Enum(s) => {
                        h.update([2]);
                        h.update(s);
                        h.update([0]);
                    }
                }
            }
            for i in inputs {
                h.update(nodes[*i].digest);
            }
        }
        Op::Out { call, index } => {
            h.update([7, *index]);
            h.update(nodes[*call].digest);
        }
        Op::At { spec, body } => {
            h.update([8]);
            h.update(crate::quant::fns::at::timeframe_text(*spec));
            h.update([0]);
            h.update(body.dag.node(body.root).digest);
        }
    }
    h.finalize().into()
}

impl Dag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Nodes in schedule order (every input precedes its users).
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    fn intern(&mut self, op: Op, ty: Ty, warm: usize) -> NodeId {
        let digest = digest_of(&op, &self.nodes);
        if let Some(&id) = self.index.get(&digest) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(Node { op, ty, warm, digest });
        self.index.insert(digest, id);
        id
    }

    pub fn constant(&mut self, v: f64) -> NodeId {
        self.intern(Op::Const(v), Ty::scalar(), 0)
    }

    pub fn field(&mut self, f: Field) -> NodeId {
        self.intern(Op::Field(f), Ty::series(), 0)
    }

    pub fn handle(&mut self, name: &str, ty: Ty, warm: usize) -> NodeId {
        self.intern(Op::Handle(name.to_string()), ty, warm)
    }

    pub fn unary(&mut self, op: UnOp, x: NodeId, ty: Ty) -> NodeId {
        if let (UnOp::Neg, Op::Const(c)) = (op, &self.nodes[x].op) {
            let c = -*c;
            return self.constant(c);
        }
        let warm = self.nodes[x].warm;
        self.intern(Op::Unary(op, x), ty, warm)
    }

    pub fn binary(&mut self, op: BinOp, mut a: NodeId, mut b: NodeId, ty: Ty) -> NodeId {
        if op.is_commutative() && self.nodes[b].digest < self.nodes[a].digest {
            std::mem::swap(&mut a, &mut b);
        }
        let warm = self.nodes[a].warm.max(self.nodes[b].warm);
        self.intern(Op::Binary(op, a, b), ty, warm)
    }

    pub fn call(
        &mut self,
        def: &'static FnDef,
        params: Vec<ParamVal>,
        inputs: Vec<NodeId>,
        ty: Ty,
    ) -> NodeId {
        let warm =
            inputs.iter().map(|i| self.nodes[*i].warm).max().unwrap_or(0) + (def.warmup)(&params);
        self.intern(Op::Call { def, params, inputs }, ty, warm)
    }

    /// An `at` node over `body`. It sits on the base clock and carries no warm-up of its own:
    /// how many base bars the body needs depends on how many bars a window holds, which the
    /// DAG does not know (see [`Program::clock_warmups`](super::Program::clock_warmups)).
    pub fn at(&mut self, spec: BarSpec, body: Arc<Program>, ty: Ty) -> NodeId {
        self.intern(Op::At { spec, body }, ty, 0)
    }

    pub fn out(&mut self, call: NodeId, index: u8) -> NodeId {
        let (ty, warm) = (self.nodes[call].ty.clone(), self.nodes[call].warm);
        self.intern(Op::Out { call, index }, ty, warm)
    }

    /// Bars until every infinite-memory kernel at or below `id` has forgotten its seed to within
    /// `eps`: `warm`, plus each such kernel's own settle time along the longest path.
    pub fn settle(&self, id: NodeId, eps: f64) -> usize {
        let mut s = vec![0usize; id + 1];
        for (i, n) in self.nodes.iter().enumerate().take(id + 1) {
            s[i] = match &n.op {
                Op::Const(_) | Op::Field(_) => 0,
                Op::Handle(_) => n.warm,
                Op::Unary(_, x) => s[*x],
                Op::Binary(_, a, b) => s[*a].max(s[*b]),
                Op::Out { call, .. } => s[*call],
                Op::At { .. } => 0,
                Op::Call { def, params, inputs } => {
                    let extra = match def.memory {
                        Memory::Finite => 0,
                        Memory::Infinite { settle } => settle(params, eps),
                    };
                    inputs.iter().map(|i| s[*i]).max().unwrap_or(0) + (def.warmup)(params) + extra
                }
            };
        }
        s[id]
    }

    /// Every function the DAG calls, with its revision, sorted by name.
    ///
    /// A program that calls `at` also lists `at` and every function its bodies call; one that
    /// does not is unchanged.
    pub fn function_revs(&self) -> Vec<(&'static str, u32)> {
        let mut m = BTreeMap::new();
        self.collect_revs(&mut m);
        m.into_iter().collect()
    }

    fn collect_revs(&self, m: &mut BTreeMap<&'static str, u32>) {
        for n in &self.nodes {
            match &n.op {
                Op::Call { def, .. } => {
                    m.insert(def.name, def.rev);
                }
                Op::At { body, .. } => {
                    m.insert("at", crate::quant::fns::at::REV);
                    body.dag.collect_revs(m);
                }
                _ => {}
            }
        }
    }

    /// The warm-up of `id` in base bars when every `at` costs `at_bars(spec, body_warmup)`
    /// base bars; none when `at_bars` cannot say. Mirrors how `warm` is built node by node.
    pub fn warm_with(
        &self,
        id: NodeId,
        at_bars: &dyn Fn(BarSpec, usize) -> Option<usize>,
    ) -> Option<usize> {
        let mut w = vec![0usize; id + 1];
        for (i, n) in self.nodes.iter().enumerate().take(id + 1) {
            w[i] = match &n.op {
                Op::Const(_) | Op::Field(_) => 0,
                Op::Handle(_) => n.warm,
                Op::Unary(_, x) => w[*x],
                Op::Binary(_, a, b) => w[*a].max(w[*b]),
                Op::Out { call, .. } => w[*call],
                Op::Call { def, params, inputs } => {
                    inputs.iter().map(|i| w[*i]).max().unwrap_or(0) + (def.warmup)(params)
                }
                Op::At { spec, body } => at_bars(*spec, body.warmup())?,
            };
        }
        Some(w[id])
    }
}
