//! The expression language front end: source text to a typed, hash-consed DAG and a canonical
//! program hash. No evaluation happens here; the evaluator reads the DAG.
//!
//! Pipeline: [`lexer`] and [`parser`] (hand-written Pratt parser) give an AST; [`types`]
//! checks it against the registry while building the [`dag`]; [`canon`] prints the canonical
//! form and the hash from the DAG alone.

pub mod canon;
pub mod dag;
pub mod lexer;
pub mod parser;
pub mod types;

use std::fmt;

pub use dag::{BinOp, Dag, Field, Node, NodeId, Op, UnOp};
pub use types::{Clock, Elem, Env, Handle, Shape, Ty};

use crate::quant::registry::FnDef;

/// Deepest nesting of the syntax tree.
pub const MAX_DEPTH: usize = 64;
/// Most syntax nodes in one expression.
pub const MAX_NODES: usize = 4096;
/// Largest integer parameter (windows allocate).
pub const MAX_PERIOD: i64 = 1 << 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Syntax,
    UnknownFunction,
    Type,
    UnboundParameter,
    UnsupportedHere,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Syntax => "expr_syntax",
            ErrorCode::UnknownFunction => "expr_unknown_function",
            ErrorCode::Type => "expr_type",
            ErrorCode::UnboundParameter => "expr_unbound_parameter",
            ErrorCode::UnsupportedHere => "expr_unsupported_here",
        }
    }
}

/// A refusal of an expression. `pos` is a byte offset into the source (always on a char
/// boundary); `valid` holds close alternatives where a name was wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct ExprError {
    pub code: ErrorCode,
    pub pos: usize,
    pub snippet: String,
    pub message: String,
    pub valid: Vec<String>,
}

impl ExprError {
    pub fn new(src: &str, code: ErrorCode, pos: usize, message: impl Into<String>) -> Self {
        let pos = floor_boundary(src, pos);
        Self { code, pos, snippet: snippet(src, pos), message: message.into(), valid: Vec::new() }
    }

    pub fn with_valid(mut self, valid: Vec<String>) -> Self {
        self.valid = valid;
        self
    }
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} (at byte {})\n{}",
            self.code.as_str(),
            self.message,
            self.pos,
            self.snippet
        )?;
        if !self.valid.is_empty() {
            write!(f, "\nvalid: {}", self.valid.join(", "))?;
        }
        Ok(())
    }
}

impl std::error::Error for ExprError {}

fn floor_boundary(src: &str, mut pos: usize) -> usize {
    pos = pos.min(src.len());
    while !src.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// The source line holding `pos`, trimmed to a window around it, and a caret line below.
fn snippet(src: &str, pos: usize) -> String {
    const HALF: usize = 40;
    let line_start = src[..pos].rfind('\n').map_or(0, |i| i + 1);
    let line_end = src[pos..].find('\n').map_or(src.len(), |i| pos + i);
    let col = src[line_start..pos].chars().count();
    let chars: Vec<char> = src[line_start..line_end].chars().collect();
    let start = col.saturating_sub(HALF);
    let end = chars.len().min(col + HALF);
    let text: String =
        chars[start..end].iter().map(|c| if c.is_control() { ' ' } else { *c }).collect();
    format!("{text}\n{}^", " ".repeat(col - start))
}

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Up to five candidates within a small edit distance of `name`, closest first.
pub(crate) fn suggest<'a>(
    name: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    let a: Vec<char> = name.chars().take(64).collect();
    let limit = (a.len() / 3).max(2).min(a.len().saturating_sub(1).max(1));
    let mut hits: Vec<(usize, &str)> = candidates
        .into_iter()
        .filter_map(|c| {
            let b: Vec<char> = c.chars().collect();
            let d = edit_distance(&a, &b);
            (d <= limit).then_some((d, c))
        })
        .collect();
    hits.sort();
    hits.dedup();
    hits.into_iter().take(5).map(|(_, c)| c.to_string()).collect()
}

/// A checked expression: the DAG, its root and the canonical identity.
#[derive(Debug, Clone)]
pub struct Program {
    pub dag: Dag,
    pub root: NodeId,
    pub canonical: String,
    /// Lowercase hex sha256, see [`canon::program_hash`].
    pub hash: String,
}

impl Program {
    /// Leading NaN bars of the root on clean data, on the base clock. An `at(...)` counts as
    /// zero here: its cost in base bars depends on how many bars a window holds. See
    /// [`Program::clock_warmups`] and [`Program::warmup_bars`].
    pub fn warmup(&self) -> usize {
        self.dag.node(self.root).warm
    }

    /// Whether the program calls `at`, so it needs timestamps, a bar stamp and a zone to run.
    pub fn has_at(&self) -> bool {
        self.dag.nodes().iter().any(|n| matches!(n.op, Op::At { .. }))
    }

    /// Whether the program has a cross-sectional call, so it has a value only over a panel
    /// (`Expr.eval_panel`): its root, or something under it, is a `Panel`.
    pub fn is_panel(&self) -> bool {
        self.dag.nodes().iter().any(|n| n.ty.shape == Shape::Panel)
    }

    /// The warm-up of each clock: `("base", n)` first, then one entry per `at` timeframe with
    /// the most bars of that timeframe any body of it needs before it is known.
    pub fn clock_warmups(&self) -> Vec<(String, usize)> {
        let mut by_tf = std::collections::BTreeMap::new();
        for n in self.dag.nodes() {
            if let Op::At { spec, body } = &n.op {
                let w = by_tf.entry(crate::quant::fns::at::timeframe_text(*spec)).or_insert(0);
                *w = (*w).max(body.warmup());
            }
        }
        std::iter::once(("base".to_string(), self.warmup())).chain(by_tf).collect()
    }

    /// The identity of this program run under a clock (`ClockCtx::text`): the zone, the bar stamp
    /// and the session decide what an `at(...)` computes, so they belong to its hash. A program
    /// without `at` runs the same under every clock, so its hash is returned unchanged.
    pub fn bound_hash(&self, clock_text: &str) -> String {
        if !self.has_at() {
            return self.hash.clone();
        }
        crate::journal::sha256_hex(
            format!("raptorbt-expr-bound-v1\n{}\n{clock_text}\n", self.hash).as_bytes(),
        )
    }

    /// The most leading unknown base bars the root can have, when that can be known. Without
    /// `at` it is [`Program::warmup`]. With `at` it needs the base bar width `base_ns`: a window
    /// of width `w` holds at most `ceil(w / base_ns)` bars, the body's first value comes with
    /// emission number `warmup + 1`, which is the first bar of window `warmup + 1`, so an `at`
    /// costs at most `(warmup + 1) * ceil(w / base_ns)` bars. A month or a year has no fixed
    /// width, and without `base_ns` there is nothing to divide by: none then.
    pub fn warmup_bars(&self, base_ns: Option<i64>) -> Option<usize> {
        if !self.has_at() {
            return Some(self.warmup());
        }
        let base = base_ns.filter(|b| *b > 0)?;
        self.dag.warm_with(self.root, &|spec, body_warmup| {
            let w = crate::quant::fns::at::width_ns(spec)?;
            let per_window = usize::try_from((w + base - 1) / base).ok()?.max(1);
            body_warmup.checked_add(1)?.checked_mul(per_window)
        })
    }

    /// Bars until infinite-memory kernels have forgotten their seed to within `eps`.
    pub fn settle(&self, eps: f64) -> usize {
        self.dag.settle(self.root, eps)
    }

    /// The columns the program yields: the selected output, or every output of an unselected
    /// multi-output call at the root.
    pub fn outputs(&self) -> Vec<(NodeId, u8)> {
        match &self.dag.node(self.root).op {
            Op::Call { def, .. } => (0..def.outputs.len() as u8).map(|i| (self.root, i)).collect(),
            _ => vec![(self.root, 0)],
        }
    }

    /// The function a node calls, if it is a call.
    pub fn def(&self, id: NodeId) -> Option<&'static FnDef> {
        match &self.dag.node(id).op {
            Op::Call { def, .. } => Some(def),
            _ => None,
        }
    }
}

/// Parse, typecheck and canonicalise `src`.
pub fn compile(src: &str, env: &Env) -> Result<Program, ExprError> {
    let ast = parser::parse(src)?;
    let (dag, root) = types::check(&ast, src, env)?;
    let canonical = canon::canonical(&dag, root);
    let hash = canon::program_hash(&dag, root, &canonical);
    Ok(Program { dag, root, canonical, hash })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::sha256_hex;
    use crate::quant::registry::lookup;
    use proptest::prelude::*;

    fn env() -> Env {
        let mut e = Env::default();
        e.params.insert("n".into(), 5.0);
        e.series.insert("h_ab12".into(), Handle { ty: Ty::series(), warm: 10 });
        e
    }

    fn prog(src: &str) -> Program {
        compile(src, &env()).unwrap_or_else(|e| panic!("{src:?}: {e}"))
    }

    fn canon(src: &str) -> String {
        prog(src).canonical
    }

    fn err(src: &str) -> ExprError {
        match compile(src, &env()) {
            Err(e) => e,
            Ok(p) => panic!("{src:?} should be refused, got {}", p.canonical),
        }
    }

    fn same(a: &str, b: &str) {
        assert_eq!(prog(a).hash, prog(b).hash, "{a:?} vs {b:?}");
        assert_eq!(canon(a), canon(b));
    }

    fn differ(a: &str, b: &str) {
        assert_ne!(prog(a).hash, prog(b).hash, "{a:?} vs {b:?}");
    }

    fn small_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new().stack_size(1 << 21).spawn(f).unwrap().join().unwrap();
    }

    #[test]
    fn precedence_table() {
        // Each pair spells one tree two ways.
        for (loose, explicit) in [
            ("1 + 2 * 3", "1 + (2 * 3)"),
            ("close - 1 - 2", "(close - 1) - 2"),
            ("close / 2 * 3", "(close / 2) * 3"),
            ("-close ** 2", "-(close ** 2)"),
            ("2 ** -1", "2 ** (-1)"),
            ("close > 1 and open < 2 or low > 3", "((close > 1) and (open < 2)) or (low > 3)"),
            ("close > 1 or open < 2 and low > 3", "(close > 1) or ((open < 2) and (low > 3))"),
            ("not close > 1 and open < 2", "(not (close > 1)) and (open < 2)"),
            ("close > 1 | open < 2 & low > 3", "(close > 1) or ((open < 2) and (low > 3))"),
            ("close + 1 > open * 2", "(close + 1) > (open * 2)"),
            ("+close", "close"),
        ] {
            same(loose, explicit);
        }
        differ("1 + 2 * 3", "(1 + 2) * 3");
        differ("-close ** 2", "(-close) ** 2");
        differ("close - 1 - 2", "close - (1 - 2)");
        differ("not close > 1 and open < 2", "not (close > 1 and open < 2)");
    }

    #[test]
    fn pow_is_right_associative() {
        assert_eq!(canon("close ** 2 ** 3"), "close ** 2.0 ** 3.0");
        assert_eq!(canon("close ** (2 ** 3)"), "close ** 2.0 ** 3.0");
        assert_eq!(canon("(close ** 2) ** 3"), "(close ** 2.0) ** 3.0");
        assert_eq!(canon("-5 ** 2"), "-5.0 ** 2.0");
        assert_eq!(canon("(-5) ** 2"), "(-5.0) ** 2.0");
    }

    #[test]
    fn chained_comparison_is_refused_at_the_second_operator() {
        let e = err("1 < close < 3");
        assert_eq!((e.code, e.pos), (ErrorCode::Syntax, 10));
        assert!(e.message.contains("chain"), "{e}");
        assert_eq!(err("close == open != low").pos, 14);
    }

    #[test]
    fn unsupported_operators_are_refused_with_a_position() {
        for (src, pos) in
            [("close % 2", 6), ("close ^ 2", 6), ("close ! 1", 6), ("x = 1", 2), ("close ~ 1", 6)]
        {
            let e = err(src);
            assert_eq!((e.code, e.pos), (ErrorCode::Syntax, pos), "{src}");
        }
        let e = err("close +");
        assert_eq!((e.pos, e.snippet.as_str()), (7, "close +\n       ^"));
        assert_eq!(err("close close").pos, 6);
        assert_eq!(err("sma(close, 5").pos, 12);
        assert_eq!(err("").pos, 0);
    }

    #[test]
    fn bad_names_get_a_did_you_mean_at_the_right_position() {
        let e = err("1 + smaa(close, 5)");
        assert_eq!((e.code, e.pos), (ErrorCode::UnknownFunction, 4));
        assert!(e.valid.contains(&"sma".to_string()), "{e}");
        assert_eq!(e.snippet, "1 + smaa(close, 5)\n    ^");

        let e = err("clse > 1");
        assert_eq!((e.code, e.pos), (ErrorCode::Type, 0));
        assert_eq!(e.valid, vec!["close".to_string()]);

        let e = err("sma(close, 5).valeu");
        assert_eq!((e.code, e.pos), (ErrorCode::Type, 14));
        assert_eq!(e.valid, vec!["value".to_string()]);

        let e = err("sma(close, $m)");
        assert_eq!((e.code, e.pos), (ErrorCode::UnboundParameter, 11));
        assert_eq!(e.valid, vec!["n".to_string()]);

        assert_eq!(err("sma").code, ErrorCode::Type);
        assert_eq!(err("at(\"1d\", at(\"1d\", close))").code, ErrorCode::UnsupportedHere);
    }

    #[test]
    fn a_series_is_not_a_period() {
        for (src, pos) in [
            ("sma(close, close)", 11),
            ("sma(close, sma(close, 3))", 11),
            ("sma(close, 2 * 3)", 11),
            ("ema(open > 1, 3)", 4),
        ] {
            let e = err(src);
            assert_eq!((e.code, e.pos), (ErrorCode::Type, pos), "{src}: {e}");
        }
        assert!(err("sma(close, close)").message.contains("series"));
    }

    #[test]
    fn period_limits() {
        assert_eq!(canon("sma(close, 16777216)"), "sma(close, 16777216)");
        for src in [
            "sma(close, 16777217)",
            "sma(close, 0)",
            "sma(close, -3)",
            "sma(close, 2.5)",
            "sma(close, 1e300)",
        ] {
            assert_eq!(err(src).code, ErrorCode::Type, "{src}");
        }
    }

    #[test]
    fn inputs_default_and_can_be_overridden() {
        assert_eq!(canon("sma(5)"), "sma(close, 5)");
        assert_eq!(canon("sma(close, 5.0)"), "sma(close, 5)");
        assert_eq!(canon("sma(open, 5)"), "sma(open, 5)");
        assert_eq!(canon("sma(sma(close, 3), 5)"), "sma(sma(close, 3), 5)");
        assert_eq!(canon("sma(close, $n)"), "sma(close, 5)");
        assert_eq!(canon("sma(close, 5).value"), "sma(close, 5)");
        same("sma(5)", "sma(close, 5)");
        differ("sma(open, 5)", "sma(close, 5)");
        assert_eq!(err("sma(close, 5, 6)").code, ErrorCode::Type);
    }

    #[test]
    fn type_rules() {
        for src in [
            "close and open",
            "not close",
            "close + (close > 1)",
            "sma(close > 1, 3)",
            "-(close > 1)",
            "\"x\"",
            "close.value",
            "sma",
        ] {
            assert_eq!(err(src).code, ErrorCode::Type, "{src}");
        }
        assert_eq!(err("sma(close, 5).value.value").code, ErrorCode::Type);
        let p = prog("close > sma(20)");
        let ty = &p.dag.node(p.root).ty;
        assert_eq!((ty.shape, ty.elem, ty.clock.clone()), (Shape::Series, Elem::Bool, Clock::Base));
        let p = prog("1 + 2");
        assert_eq!(p.dag.node(p.root).ty.shape, Shape::Scalar);
    }

    #[test]
    fn handles_are_series_from_the_environment() {
        assert_eq!(canon("sma(h_ab12, 3) > h_ab12"), canon(&canon("sma(h_ab12, 3) > h_ab12")));
        assert_eq!(prog("sma(h_ab12, 3)").warmup(), 12);
        let e = err("h_ab13 + 1");
        assert_eq!(e.valid, vec!["h_ab12".to_string()]);
        assert!(compile("h_ab12", &Env::default()).is_err());
    }

    #[test]
    fn warmup_and_settle() {
        assert_eq!(prog("close").warmup(), 0);
        assert_eq!(prog("sma(close, 5)").warmup(), 4);
        assert_eq!(prog("sma(sma(close, 5), 5) + 1").warmup(), 8);
        assert_eq!(prog("sma(close, 5) - sma(close, 20)").warmup(), 19);
        assert_eq!(prog("sma(close, 5)").settle(1e-6), 4);
        // ema(10): 9 leading NaNs, then ceil(ln 1e-6 / ln(9/11)) = ceil(68.84) = 69 bars.
        assert_eq!(prog("ema(close, 10)").settle(1e-6), 78);
    }

    #[test]
    fn hash_is_stable_across_spellings_and_moves_with_content() {
        same("close + open", "open + close");
        same("(open + close) + low", "low + (close + open)");
        same("close * 2 == open", "open == 2 * close");
        same("close > 1 and open > 1", "open > 1 and close > 1");
        same("sma(close, 5) + 1", "1 + sma( close , 5.0 )");
        differ("(open + close) + low", "open + (close + low)");
        differ("close - open", "open - close");
        differ("sma(close, 5)", "sma(close, 6)");
        differ("sma(close, 5)", "ema(close, 5)");
        differ("close * 2", "close * 2.5");
        differ("close / open", "open / close");
        assert_eq!(prog("sma(close, $n)").hash, prog("sma(close, 5)").hash);

        let rev = lookup("sma").unwrap().rev;
        assert_eq!(prog("close").hash, sha256_hex(b"raptorbt-expr-v1\nclose\n"));
        assert_eq!(
            prog("sma(5)").hash,
            sha256_hex(format!("raptorbt-expr-v1\nsma(close, 5)\nsma@{rev}\n").as_bytes())
        );
    }

    /// Hashes of programs without `at`, computed from the hash definition before `at` existed
    /// (sha256 of `raptorbt-expr-v1`, the canonical text and one `name@rev` line per function).
    /// `rsi` is pinned on its canonical text with the `compat` parameter the matype work added,
    /// `rsi(close, 14, "native")`; it was pinned without it before that commit.
    /// They must not move: a stored hash and a shared cell key stay valid.
    #[test]
    fn programs_without_at_keep_their_hashes() {
        for (src, hash) in [
            ("close", "63a14bbe53bfdf48e2086a4f41586de9d16b13d54f8f5998d4acd3d5eae5cb6a"),
            ("sma(5)", "b2d173d37680ccf900eede9e5c7f1a23f94fa1c067e51e25a3efb35a86ad0ce3"),
            ("close > sma(20)", "2abfa0c16f9965d7f2dd3a0653be7f772b8f0da32bbae53281c5ff5d9ca84c3e"),
            ("rsi(14) < 30", "46200ab880ab4e63dd8a7f70d773326680ff6de4cd10b35193f6d377948c9d55"),
        ] {
            assert_eq!(prog(src).hash, hash, "{src}");
        }
    }

    #[test]
    fn at_is_one_node_on_the_base_clock_with_its_own_body() {
        let p = prog("close > at(\"1d\", sma(close, 5))");
        assert_eq!(p.canonical, "close > at(\"1d\", sma(close, 5))");
        assert!(p.has_at());
        let ty = &p.dag.node(p.root).ty;
        assert_eq!((ty.shape, ty.elem, ty.clock.clone()), (Shape::Series, Elem::Bool, Clock::Base));
        // Outer: close, at, >. The body's `close` and `sma` live in the body's own DAG.
        assert_eq!(p.dag.len(), 3);
        let at = p.dag.nodes().iter().find_map(|n| match &n.op {
            Op::At { body, .. } => Some(body.clone()),
            _ => None,
        });
        assert_eq!(at.unwrap().dag.len(), 2);
        // The same `at` spelled twice is one node; a different timeframe or body is another.
        assert_eq!(prog("at(\"1d\", close) + at(\"1d\", close)").dag.len(), 2);
        assert_eq!(prog("at(\"1d\", close) + at(\"1w\", close)").dag.len(), 3);
        same("at(\"1d\", sma(5))", "at(\"1d\", sma(close, 5.0))");
        differ("at(\"1d\", sma(5))", "at(\"1w\", sma(5))");
        differ("at(\"1d\", sma(5))", "at(\"1d\", sma(6))");
        differ("at(\"1d\", sma(5))", "sma(5)");
        differ("at(\"1d\", close)", "close");
    }

    #[test]
    fn at_hash_names_at_and_the_functions_of_its_body() {
        let p = prog("at(\"1d\", sma(5))");
        let want = sha256_hex(
            format!(
                "raptorbt-expr-v1\nat(\"1d\", sma(close, 5))\nat@{}\nsma@{}\n",
                crate::quant::fns::at::REV,
                lookup("sma").unwrap().rev
            )
            .as_bytes(),
        );
        assert_eq!(p.hash, want);
        // Folding a clock in moves the hash of an `at` program and no other.
        assert_ne!(p.bound_hash("tz=0;stamp=close;session=none"), p.hash);
        assert_ne!(
            p.bound_hash("tz=0;stamp=close;session=none"),
            p.bound_hash("tz=0;stamp=open;session=none")
        );
        let plain = prog("sma(5)");
        assert_eq!(plain.bound_hash("tz=0;stamp=close;session=none"), plain.hash);
    }

    #[test]
    fn at_round_trips_and_mixes_clocks() {
        for src in [
            "at(\"1d\", close)",
            "at(\"15m\", sma(close, 5) > open)",
            "close > at(\"1d\", sma(close, 200)) and at(\"1h\", rsi(14)) < 30",
            "-at(\"1w\", macd(close).macdsignal)",
            "at(\"1d\", close) ** 2",
            "sma(at(\"1d\", close), 3)",
        ] {
            let p = prog(src);
            let q = prog(&p.canonical);
            assert_eq!((&q.canonical, &q.hash), (&p.canonical, &p.hash), "{src}");
        }
    }

    #[test]
    fn at_refusals_are_typed_and_positioned() {
        let e = err("at(\"1d\", at(\"1w\", close))");
        assert_eq!((e.code, e.pos), (ErrorCode::UnsupportedHere, 9));
        for (src, pos) in [
            ("at(\"2x\", close)", 3),
            ("at(\"d\", close)", 3),
            ("at(\"0m\", close)", 3),
            ("at(\"100tick\", close)", 3),
            ("at(close, close)", 3),
            ("at(\"1d\")", 0),
            ("at(\"1d\", close, close)", 0),
            ("at(\"1d\", 5)", 9),
            ("at(\"1d\", macd(close))", 9),
            ("at(\"1d\", h_ab12)", 9),
        ] {
            let e = err(src);
            assert_eq!((e.code, e.pos), (ErrorCode::Type, pos), "{src}: {e}");
        }
        assert!(err("at(\"2x\", close)").valid.contains(&"1d".to_string()));
        assert!(err("at(\"1d\", h_ab12)").message.contains("base clock"));
    }

    #[test]
    fn warmup_is_reported_per_clock() {
        let p = prog("close > at(\"1d\", sma(close, 200)) and sma(close, 3) > 0");
        assert_eq!(p.warmup(), 2);
        assert_eq!(p.clock_warmups(), vec![("base".to_string(), 2), ("1d".to_string(), 199)]);
        // Unknowable without the base width; with it, (199 + 1) windows of at most 288 5-minute
        // bars.
        let five_min = 300_000_000_000;
        assert_eq!(p.warmup_bars(None), None);
        assert_eq!(p.warmup_bars(Some(five_min)), Some(200 * 288));
        // The deeper of the two clocks decides: (2 + 1) * 288 against the base sma(10)'s 9.
        assert_eq!(
            prog("at(\"1d\", sma(close, 3)) + sma(close, 10)").warmup_bars(Some(five_min)),
            Some(3 * 288)
        );
        assert_eq!(prog("sma(5)").warmup_bars(None), Some(4));
        assert_eq!(prog("at(\"1month\", close)").warmup_bars(Some(five_min)), None);
        assert_eq!(prog("at(\"1d\", close)").clock_warmups()[1], ("1d".to_string(), 0));
    }

    #[test]
    fn shared_subexpressions_are_one_node() {
        assert_eq!(prog("sma(close, 5) + sma(close, 5) * sma(5)").dag.len(), 4); // close, sma, *, +
        assert_eq!(prog("close * 2 + close * 2").dag.len(), 4); // close, 2, *, +
        assert_eq!(prog("sma(close, 5) > sma(close, 5)").dag.len(), 3);
        let p = prog("sma(sma(close, 3), 5)");
        for (i, n) in p.dag.nodes().iter().enumerate() {
            if let Op::Call { inputs, .. } = &n.op {
                assert!(inputs.iter().all(|&x| x < i), "schedule order");
            }
        }
    }

    #[test]
    fn float_literals_round_trip() {
        for lit in [
            "0.1",
            "1e-7",
            "1.7976931348623157e308",
            "5e-324",
            "123456789.12345678",
            "1e16",
            "0.30000000000000004",
            "2.2250738585072014e-308",
        ] {
            let want = lit.parse::<f64>().unwrap();
            let p = prog(lit);
            let Op::Const(v) = p.dag.node(p.root).op else { panic!("{lit}") };
            assert_eq!(v.to_bits(), want.to_bits());
            let Op::Const(w) = prog(&p.canonical).dag.node(p.root).op else { panic!() };
            assert_eq!(w.to_bits(), want.to_bits(), "{lit} -> {}", p.canonical);
        }
        assert_eq!(canon("-0"), "-0.0");
        assert_eq!(canon("-5"), "-5.0");
        assert_eq!(err("1e999").code, ErrorCode::Syntax);
        assert_eq!(err("5x").code, ErrorCode::Syntax);
    }

    #[test]
    fn non_ascii_is_refused_cleanly() {
        for (src, pos) in [
            ("close é 1", 6),
            ("日本語", 0),
            ("sma(close, 5) + ü", 16),
            ("é close", 0),
            ("sma(close, \"é\")", 12),
            ("ééé + close", 0),
        ] {
            let e = err(src);
            assert_eq!((e.code, e.pos), (ErrorCode::Syntax, pos), "{src}");
            assert!(src.is_char_boundary(e.pos));
            assert!(!e.to_string().is_empty());
        }
        assert_eq!(err("close é 1").snippet, "close é 1\n      ^");
    }

    #[test]
    fn hostile_nesting_is_refused_without_overflowing_the_stack() {
        small_stack(|| {
            let n = 100_000;
            let bombs = [
                "(".repeat(n),
                format!("{}close{}", "(".repeat(n), ")".repeat(n)),
                format!("{}close", "-".repeat(n)),
                format!("{}close", "not ".repeat(n)),
                format!("{}2", "2 ** ".repeat(n)),
                format!("{}close{}", "sma(".repeat(n), ", 5)".repeat(n)),
                format!("close{}", ".value".repeat(n)),
                format!("close{}", " + 1".repeat(n)),
            ];
            for b in bombs {
                let e = compile(&b, &Env::default()).unwrap_err();
                assert!(e.pos <= b.len());
            }
            let ok = format!("{}close{}", "(".repeat(60), ")".repeat(60));
            assert_eq!(compile(&ok, &Env::default()).unwrap().canonical, "close");
            let deep = format!("{}close{}", "(".repeat(100), ")".repeat(100));
            assert!(compile(&deep, &Env::default())
                .unwrap_err()
                .message
                .contains("deeper than 64"));
        });
    }

    #[test]
    fn node_limit() {
        let many = |n: usize| format!("sma({})", vec!["1"; n].join(","));
        let e = err(&many(5000));
        assert_eq!(e.code, ErrorCode::Syntax);
        assert!(e.message.contains("4096 nodes"), "{e}");
        // 4095 arguments and the call are 4096 nodes: past the parser, refused on arity.
        assert_eq!(err(&many(4095)).code, ErrorCode::Type);
    }

    fn atom() -> impl Strategy<Value = String> {
        prop::sample::select(vec![
            "close", "open", "high", "low", "volume", "1", "2.5", "0.1", "1e-7", "3", "h_ab12",
            "$n",
        ])
        .prop_map(String::from)
    }

    fn num_expr() -> impl Strategy<Value = String> {
        atom().prop_recursive(4, 32, 4, |inner| {
            prop_oneof![
                (
                    inner.clone(),
                    inner.clone(),
                    prop::sample::select(vec!["+", "-", "*", "/", "**"])
                )
                    .prop_map(|(a, b, o)| format!("({a}) {o} ({b})")),
                inner.clone().prop_map(|a| format!("-({a})")),
                inner.clone().prop_map(|a| format!("+{a}")),
                (inner.clone(), 1..40u32).prop_map(|(a, p)| format!("sma({a}, {p})")),
                (inner.clone(), 1..40u32).prop_map(|(a, p)| format!("ema({a}, {p})")),
                (1..40u32).prop_map(|p| format!("sma({p})")),
                inner.prop_map(|a| format!("(({a}))")),
            ]
        })
    }

    fn bool_expr() -> impl Strategy<Value = String> {
        let cmp =
            (num_expr(), num_expr(), prop::sample::select(vec!["<", "<=", ">", ">=", "==", "!="]))
                .prop_map(|(a, b, o)| format!("({a}) {o} ({b})"));
        cmp.prop_recursive(3, 16, 2, |inner| {
            prop_oneof![
                inner.clone().prop_map(|a| format!("not ({a})")),
                (inner.clone(), inner, prop::sample::select(vec!["and", "or", "&", "|"]))
                    .prop_map(|(a, b, o)| format!("({a}) {o} ({b})")),
            ]
        })
    }

    fn assert_round_trip(src: &str) -> Result<(), TestCaseError> {
        let p = prog(src);
        let q = prog(&p.canonical);
        prop_assert_eq!(&q.canonical, &p.canonical);
        prop_assert_eq!(&q.hash, &p.hash);
        prop_assert_eq!(q.dag.len(), p.dag.len());
        // Every node feeds the root.
        let mut live = vec![false; p.dag.len()];
        live[p.root] = true;
        for i in (0..p.dag.len()).rev() {
            if !live[i] {
                continue;
            }
            let kids: Vec<NodeId> = match &p.dag.node(i).op {
                Op::Unary(_, x) => vec![*x],
                Op::Binary(_, a, b) => vec![*a, *b],
                Op::Call { inputs, .. } => inputs.clone(),
                Op::Out { call, .. } => vec![*call],
                _ => vec![],
            };
            for k in kids {
                live[k] = true;
            }
        }
        prop_assert!(live.iter().all(|l| *l), "orphan node in {:?}", p.canonical);
        Ok(())
    }

    proptest! {
        #[test]
        fn canonical_round_trips_on_numeric_expressions(src in num_expr()) {
            assert_round_trip(&src)?;
        }

        #[test]
        fn canonical_round_trips_on_boolean_expressions(src in bool_expr()) {
            assert_round_trip(&src)?;
        }

        #[test]
        fn every_finite_float_survives_its_canonical_form(bits in any::<u64>()) {
            let v = f64::from_bits(bits);
            prop_assume!(v.is_finite());
            let p = prog(&format!("{v:?}"));
            let Op::Const(c) = p.dag.node(p.root).op else { panic!() };
            prop_assert_eq!(c.to_bits(), v.to_bits());
            let Op::Const(c) = prog(&p.canonical).dag.node(p.root).op else { panic!() };
            prop_assert_eq!(c.to_bits(), v.to_bits());
        }

        #[test]
        fn arbitrary_strings_never_panic(src in any::<String>()) {
            check_refusal(&src)?;
        }

        #[test]
        fn expression_like_noise_never_panics(src in "[ -~éß日\n\t]{0,120}") {
            check_refusal(&src)?;
        }

        #[test]
        fn token_soup_never_panics(toks in prop::collection::vec(prop::sample::select(vec![
            "(", ")", ",", ".", "$n", "$", "close", "sma", "ema", "5", "1e3", "\"x\"", "and", "or", "not",
            "+", "-", "*", "**", "/", "<", ">=", "==", "!=", "&", "|", "value", "é",
        ]), 0..40)) {
            check_refusal(&toks.join(" "))?;
        }
    }

    fn check_refusal(src: &str) -> Result<(), TestCaseError> {
        match compile(src, &env()) {
            Ok(p) => {
                let q = compile(&p.canonical, &env());
                prop_assert!(q.is_ok(), "canonical {:?} of {:?} does not parse", p.canonical, src);
                prop_assert_eq!(q.unwrap().canonical, p.canonical);
            }
            Err(e) => {
                prop_assert!(e.pos <= src.len() && src.is_char_boundary(e.pos));
                prop_assert!(e.snippet.contains('^'));
                let _ = e.to_string();
            }
        }
        Ok(())
    }
}
