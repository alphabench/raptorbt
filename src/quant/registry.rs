//! The function registry: what an expression can call. Every table is static data; capabilities,
//! the Python surface, the docs and the conformance tests are generated from it.

use super::fns;
use super::kernel::Node;
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Math,
    Rolling,
    Overlap,
    Momentum,
    Volume,
    VolPrice,
    Candles,
    Hilbert,
    Quant,
    Cross,
    At,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Math => "math",
            Family::Rolling => "rolling",
            Family::Overlap => "overlap",
            Family::Momentum => "momentum",
            Family::Volume => "volume",
            Family::VolPrice => "vol_price",
            Family::Candles => "candles",
            Family::Hilbert => "hilbert",
            Family::Quant => "quant",
            Family::Cross => "cross",
            Family::At => "at",
        }
    }
}

/// How a function treats a NaN input compared with the legacy array API it is pinned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanPolicy {
    Recover,
    Poison,
    Skip,
}

impl NanPolicy {
    fn name(self) -> &'static str {
        match self {
            NanPolicy::Recover => "recover",
            NanPolicy::Poison => "poison",
            NanPolicy::Skip => "skip",
        }
    }
}

/// The existing indicator an expression kernel is pinned to.
#[derive(Debug, Clone, Copy)]
pub struct Legacy {
    pub nan: NanPolicy,
    pub pinned_to: &'static str,
    /// 0.0 means bitwise equal on finite input.
    pub tolerance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamVal {
    Int(i64),
    Float(f64),
    Enum(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub enum ParamKind {
    Int { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Enum(&'static [&'static str]),
}

#[derive(Debug, Clone, Copy)]
pub struct ParamDef {
    pub name: &'static str,
    pub kind: ParamKind,
    pub default: Option<ParamVal>,
}

#[derive(Debug, Clone, Copy)]
pub struct InputDef {
    pub name: &'static str,
    /// The field read when the call omits this input (`atr(14)` reads high, low, close).
    pub default_field: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutKind {
    Series,
    Bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Memory {
    Finite,
    /// Infinite memory: bars until the influence of the seed falls below `eps`.
    Infinite {
        settle: fn(&[ParamVal], f64) -> usize,
    },
}

pub type MakeFn = fn(&[ParamVal], &[usize], &[usize]) -> Result<Box<dyn Node>, String>;

#[derive(Debug, Clone, Copy)]
pub struct FnDef {
    pub name: &'static str,
    pub family: Family,
    pub inputs: &'static [InputDef],
    pub params: &'static [ParamDef],
    /// At most three.
    pub outputs: &'static [&'static str],
    pub out_kind: OutKind,
    /// Leading NaN outputs on clean data.
    pub warmup: fn(&[ParamVal]) -> usize,
    pub memory: Memory,
    pub legacy: Option<Legacy>,
    pub talib_name: Option<&'static str>,
    pub doc: &'static str,
    pub refs: &'static [&'static str],
    /// Bumped when the numbers change; part of the program hash.
    pub rev: u32,
    pub invariance: &'static str,
    pub causal: bool,
    /// Builds the node on the given input and output registers.
    pub make: MakeFn,
}

/// Every table, one per family.
pub fn tables() -> [&'static [FnDef]; 11] {
    [
        fns::math::DEFS,
        fns::rolling::DEFS,
        fns::overlap::DEFS,
        fns::momentum::DEFS,
        fns::volume::DEFS,
        fns::vol_price::DEFS,
        fns::candles::DEFS,
        fns::hilbert::DEFS,
        fns::quant::DEFS,
        fns::cross::DEFS,
        fns::at::DEFS,
    ]
}

pub fn all() -> impl Iterator<Item = &'static FnDef> {
    tables().into_iter().flat_map(|t| t.iter())
}

pub fn lookup(name: &str) -> Option<&'static FnDef> {
    all().find(|d| d.name == name)
}

pub fn int_param(params: &[ParamVal], i: usize) -> Result<usize, String> {
    match params.get(i) {
        Some(ParamVal::Int(v)) if *v >= 0 => Ok(*v as usize),
        other => Err(format!("parameter {i} must be a non-negative integer, got {other:?}")),
    }
}

pub fn regs<const N: usize>(slice: &[usize], what: &str) -> Result<[usize; N], String> {
    slice.try_into().map_err(|_| format!("{what}: expected {N} registers, got {}", slice.len()))
}

fn param_json(p: &ParamDef) -> Value {
    let (kind, extra) = match p.kind {
        ParamKind::Int { min, max } => ("int", json!({ "min": min, "max": max })),
        ParamKind::Float { min, max } => ("float", json!({ "min": min, "max": max })),
        ParamKind::Enum(v) => ("enum", json!({ "values": v })),
    };
    let default = match p.default {
        None => Value::Null,
        Some(ParamVal::Int(v)) => json!(v),
        Some(ParamVal::Float(v)) => json!(v),
        Some(ParamVal::Enum(v)) => json!(v),
    };
    json!({ "name": p.name, "kind": kind, "range": extra, "default": default })
}

fn def_json(d: &FnDef) -> Value {
    let defaults: Option<Vec<ParamVal>> = d.params.iter().map(|p| p.default).collect();
    let warmup = defaults.map(|v| (d.warmup)(&v));
    json!({
        "name": d.name,
        "family": d.family.name(),
        "inputs": d.inputs.iter().map(|i| json!({"name": i.name, "default_field": i.default_field})).collect::<Vec<_>>(),
        "params": d.params.iter().map(param_json).collect::<Vec<_>>(),
        "outputs": d.outputs,
        "out_kind": match d.out_kind { OutKind::Series => "series", OutKind::Bool => "bool" },
        "warmup_at_defaults": warmup,
        "memory": match d.memory { Memory::Finite => "finite", Memory::Infinite { .. } => "infinite" },
        "legacy": d.legacy.map(|l| json!({
            "nan": l.nan.name(), "pinned_to": l.pinned_to, "tolerance": l.tolerance,
        })),
        "talib_name": d.talib_name,
        "doc": d.doc,
        "refs": d.refs,
        "rev": d.rev,
        "invariance": d.invariance,
        "causal": d.causal,
    })
}

/// The registry as JSON, sorted by name.
pub fn expr_registry_json() -> Value {
    let mut defs: Vec<&FnDef> = all().collect();
    defs.sort_by_key(|d| d.name);
    json!({ "functions": defs.into_iter().map(def_json).collect::<Vec<_>>() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn names_are_unique_and_looked_up() {
        let mut seen = HashSet::new();
        for d in all() {
            assert!(seen.insert(d.name), "duplicate function {}", d.name);
            assert!(d.outputs.len() <= 3 && !d.outputs.is_empty(), "{}", d.name);
            assert!(std::ptr::eq(lookup(d.name).unwrap(), d));
        }
        assert!(lookup("sma").is_some() && lookup("ema").is_some());
        assert!(lookup("nope").is_none());
    }

    #[test]
    fn json_lists_every_function_with_its_legacy_pin() {
        let v = expr_registry_json();
        let fns = v["functions"].as_array().unwrap();
        assert_eq!(fns.len(), all().count());
        let sma = fns.iter().find(|f| f["name"] == "sma").unwrap();
        assert_eq!(sma["family"], "overlap");
        assert_eq!(sma["legacy"]["pinned_to"], "indicators::streaming::StreamingSma");
        assert_eq!(sma["legacy"]["tolerance"], 0.0);
    }
}
