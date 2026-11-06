//! Pratt-style recursive descent over the grammar, with the depth and size limits enforced as
//! nodes are built so a hostile input is refused before recursion can exhaust the stack.
//!
//! ```text
//! expr := or ; or := and ("or" and)* ; and := not ("and" not)* ; not := "not" not | cmp
//! cmp  := add (("<"|"<="|">"|">="|"=="|"!=") add)?   -- never chained
//! add := mul (("+"|"-") mul)* ; mul := unary (("*"|"/") unary)*
//! unary := ("-"|"+") unary | pow ; pow := postfix ("**" unary)?      -- right associative
//! postfix := atom ("." IDENT)*
//! atom := NUMBER | STRING | "$" IDENT | IDENT | IDENT "(" args ")" | "(" expr ")"
//! ```
//! `&` and `|` are the tokens `and` and `or`.

use super::dag::{BinOp, UnOp};
use super::lexer::{Lexer, Tok, Token};
use super::{ErrorCode, ExprError, MAX_DEPTH, MAX_NODES};

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Num(f64),
    Str(String),
    Param(String),
    Ident(String),
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// `base.name`, with the byte offset of `name`.
    Select(Box<Expr>, String, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub start: usize,
    pub end: usize,
    depth: usize,
}

struct Parser<'a> {
    src: &'a str,
    lex: Lexer<'a>,
    cur: Token,
    nesting: usize,
    nodes: usize,
}

pub fn parse(src: &str) -> Result<Expr, ExprError> {
    let mut lex = Lexer::new(src);
    let cur = lex.next_token()?;
    let mut p = Parser { src, lex, cur, nesting: 0, nodes: 0 };
    let e = p.expr()?;
    if p.cur.tok != Tok::Eof {
        return Err(p.err_here(format!("unexpected {}", p.cur.tok.describe())));
    }
    Ok(e)
}

impl<'a> Parser<'a> {
    fn err_here(&self, msg: impl Into<String>) -> ExprError {
        ExprError::new(self.src, ErrorCode::Syntax, self.cur.start, msg)
    }

    fn bump(&mut self) -> Result<Token, ExprError> {
        let next = self.lex.next_token()?;
        Ok(std::mem::replace(&mut self.cur, next))
    }

    fn expect(&mut self, tok: Tok, what: &str) -> Result<Token, ExprError> {
        if self.cur.tok == tok {
            self.bump()
        } else {
            Err(self.err_here(format!("expected {what}, found {}", self.cur.tok.describe())))
        }
    }

    /// Guards recursion that adds no node (parentheses, unary chains, a power's exponent).
    fn enter(&mut self) -> Result<(), ExprError> {
        self.nesting += 1;
        if self.nesting > MAX_DEPTH {
            return Err(self.err_here(format!("expression is nested deeper than {MAX_DEPTH}")));
        }
        Ok(())
    }

    fn exit(&mut self) {
        self.nesting -= 1;
    }

    fn mk(
        &mut self,
        kind: ExprKind,
        start: usize,
        end: usize,
        child_depth: usize,
    ) -> Result<Expr, ExprError> {
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(ExprError::new(
                self.src,
                ErrorCode::Syntax,
                start,
                format!("expression has more than {MAX_NODES} nodes"),
            ));
        }
        let depth = child_depth + 1;
        if depth > MAX_DEPTH {
            return Err(ExprError::new(
                self.src,
                ErrorCode::Syntax,
                start,
                format!("expression is nested deeper than {MAX_DEPTH}"),
            ));
        }
        Ok(Expr { kind, start, end, depth })
    }

    fn binary(&mut self, op: BinOp, l: Expr, r: Expr) -> Result<Expr, ExprError> {
        let (start, end, d) = (l.start, r.end, l.depth.max(r.depth));
        self.mk(ExprKind::Binary(op, Box::new(l), Box::new(r)), start, end, d)
    }

    fn expr(&mut self) -> Result<Expr, ExprError> {
        self.enter()?;
        let e = self.or()?;
        self.exit();
        Ok(e)
    }

    fn or(&mut self) -> Result<Expr, ExprError> {
        let mut l = self.and()?;
        while self.cur.tok == Tok::Or {
            self.bump()?;
            let r = self.and()?;
            l = self.binary(BinOp::Or, l, r)?;
        }
        Ok(l)
    }

    fn and(&mut self) -> Result<Expr, ExprError> {
        let mut l = self.not()?;
        while self.cur.tok == Tok::And {
            self.bump()?;
            let r = self.not()?;
            l = self.binary(BinOp::And, l, r)?;
        }
        Ok(l)
    }

    fn not(&mut self) -> Result<Expr, ExprError> {
        if self.cur.tok != Tok::Not {
            return self.cmp();
        }
        let start = self.bump()?.start;
        self.enter()?;
        let operand = self.not()?;
        self.exit();
        let (end, d) = (operand.end, operand.depth);
        self.mk(ExprKind::Unary(UnOp::Not, Box::new(operand)), start, end, d)
    }

    fn cmp(&mut self) -> Result<Expr, ExprError> {
        let l = self.add()?;
        let op = match self.cur.tok {
            Tok::Lt => BinOp::Lt,
            Tok::Le => BinOp::Le,
            Tok::Gt => BinOp::Gt,
            Tok::Ge => BinOp::Ge,
            Tok::EqEq => BinOp::Eq,
            Tok::Ne => BinOp::Ne,
            _ => return Ok(l),
        };
        self.bump()?;
        let r = self.add()?;
        if self.cur.tok.is_cmp() {
            return Err(self.err_here("comparisons do not chain; join them with `and`"));
        }
        self.binary(op, l, r)
    }

    fn add(&mut self) -> Result<Expr, ExprError> {
        let mut l = self.mul()?;
        loop {
            let op = match self.cur.tok {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => return Ok(l),
            };
            self.bump()?;
            let r = self.mul()?;
            l = self.binary(op, l, r)?;
        }
    }

    fn mul(&mut self) -> Result<Expr, ExprError> {
        let mut l = self.unary()?;
        loop {
            let op = match self.cur.tok {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                _ => return Ok(l),
            };
            self.bump()?;
            let r = self.unary()?;
            l = self.binary(op, l, r)?;
        }
    }

    fn unary(&mut self) -> Result<Expr, ExprError> {
        let neg = match self.cur.tok {
            Tok::Minus => true,
            Tok::Plus => false,
            _ => return self.pow(),
        };
        let start = self.bump()?.start;
        self.enter()?;
        let operand = self.unary()?;
        self.exit();
        if !neg {
            // `+x` is `x` exactly.
            return Ok(operand);
        }
        let (end, d) = (operand.end, operand.depth);
        self.mk(ExprKind::Unary(UnOp::Neg, Box::new(operand)), start, end, d)
    }

    fn pow(&mut self) -> Result<Expr, ExprError> {
        let base = self.postfix()?;
        if self.cur.tok != Tok::Pow {
            return Ok(base);
        }
        self.bump()?;
        self.enter()?;
        let exponent = self.unary()?;
        self.exit();
        self.binary(BinOp::Pow, base, exponent)
    }

    fn postfix(&mut self) -> Result<Expr, ExprError> {
        let mut e = self.atom()?;
        while self.cur.tok == Tok::Dot {
            self.bump()?;
            let name_tok = self.cur.clone();
            let Tok::Ident(name) = name_tok.tok else {
                return Err(self.err_here(format!(
                    "expected an output name after `.`, found {}",
                    self.cur.tok.describe()
                )));
            };
            self.bump()?;
            let (start, d) = (e.start, e.depth);
            e = self.mk(
                ExprKind::Select(Box::new(e), name, name_tok.start),
                start,
                name_tok.end,
                d,
            )?;
        }
        Ok(e)
    }

    fn atom(&mut self) -> Result<Expr, ExprError> {
        let t = self.cur.clone();
        match t.tok {
            Tok::Num(v) => {
                self.bump()?;
                self.mk(ExprKind::Num(v), t.start, t.end, 0)
            }
            Tok::Str(s) => {
                self.bump()?;
                self.mk(ExprKind::Str(s), t.start, t.end, 0)
            }
            Tok::Param(s) => {
                self.bump()?;
                self.mk(ExprKind::Param(s), t.start, t.end, 0)
            }
            Tok::Ident(name) => {
                self.bump()?;
                if self.cur.tok != Tok::LParen {
                    return self.mk(ExprKind::Ident(name), t.start, t.end, 0);
                }
                self.bump()?;
                let mut args = Vec::new();
                let mut d = 0;
                if self.cur.tok != Tok::RParen {
                    loop {
                        let a = self.expr()?;
                        d = d.max(a.depth);
                        args.push(a);
                        if self.cur.tok == Tok::Comma {
                            self.bump()?;
                        } else {
                            break;
                        }
                    }
                }
                let close = self.expect(Tok::RParen, "`,` or `)`")?;
                self.mk(ExprKind::Call { name, args }, t.start, close.end, d)
            }
            Tok::LParen => {
                self.bump()?;
                let e = self.expr()?;
                self.expect(Tok::RParen, "`)`")?;
                Ok(e)
            }
            other => {
                Err(self.err_here(format!("expected an expression, found {}", other.describe())))
            }
        }
    }
}
