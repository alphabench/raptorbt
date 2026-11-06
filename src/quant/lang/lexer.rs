//! Lexer: one token of lookahead, byte positions, no allocation beyond the token text.

use super::{ErrorCode, ExprError};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    /// `$name`
    Param(String),
    And,
    Or,
    Not,
    Plus,
    Minus,
    Star,
    Pow,
    Slash,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    LParen,
    RParen,
    Comma,
    Dot,
    Eof,
}

impl Tok {
    pub fn describe(&self) -> String {
        match self {
            Tok::Num(v) => format!("number {v}"),
            Tok::Str(s) => format!("string \"{s}\""),
            Tok::Ident(s) => format!("name `{s}`"),
            Tok::Param(s) => format!("`${s}`"),
            Tok::And => "`and`".into(),
            Tok::Or => "`or`".into(),
            Tok::Not => "`not`".into(),
            Tok::Plus => "`+`".into(),
            Tok::Minus => "`-`".into(),
            Tok::Star => "`*`".into(),
            Tok::Pow => "`**`".into(),
            Tok::Slash => "`/`".into(),
            Tok::Lt => "`<`".into(),
            Tok::Le => "`<=`".into(),
            Tok::Gt => "`>`".into(),
            Tok::Ge => "`>=`".into(),
            Tok::EqEq => "`==`".into(),
            Tok::Ne => "`!=`".into(),
            Tok::LParen => "`(`".into(),
            Tok::RParen => "`)`".into(),
            Tok::Comma => "`,`".into(),
            Tok::Dot => "`.`".into(),
            Tok::Eof => "end of input".into(),
        }
    }

    pub fn is_cmp(&self) -> bool {
        matches!(self, Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge | Tok::EqEq | Tok::Ne)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub start: usize,
    pub end: usize,
}

pub struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn err(&self, pos: usize, msg: impl Into<String>) -> ExprError {
        ExprError::new(self.src, ErrorCode::Syntax, pos, msg)
    }

    fn byte(&self, at: usize) -> Option<u8> {
        self.src.as_bytes().get(at).copied()
    }

    fn digits(&mut self) {
        while self.byte(self.pos).is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
    }

    fn word(&mut self) -> &'a str {
        let start = self.pos;
        while self.byte(self.pos).is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_') {
            self.pos += 1;
        }
        &self.src[start..self.pos]
    }

    pub fn next_token(&mut self) -> Result<Token, ExprError> {
        while self.byte(self.pos).is_some_and(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
        let start = self.pos;
        let Some(b) = self.byte(start) else {
            return Ok(Token { tok: Tok::Eof, start, end: start });
        };
        let two = |s: &Self, second: u8| s.byte(start + 1) == Some(second);
        let (tok, len) = match b {
            b'0'..=b'9' => return self.number(),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let w = self.word();
                let tok = match w {
                    "and" => Tok::And,
                    "or" => Tok::Or,
                    "not" => Tok::Not,
                    _ => Tok::Ident(w.to_string()),
                };
                return Ok(Token { tok, start, end: self.pos });
            }
            b'$' => {
                self.pos += 1;
                if !self.byte(self.pos).is_some_and(|b| b.is_ascii_alphabetic() || b == b'_') {
                    return Err(self.err(start, "expected a parameter name after `$`"));
                }
                let w = self.word().to_string();
                return Ok(Token { tok: Tok::Param(w), start, end: self.pos });
            }
            b'"' | b'\'' => return self.string(b),
            b'+' => (Tok::Plus, 1),
            b'-' => (Tok::Minus, 1),
            b'*' if two(self, b'*') => (Tok::Pow, 2),
            b'*' => (Tok::Star, 1),
            b'/' => (Tok::Slash, 1),
            b'<' if two(self, b'=') => (Tok::Le, 2),
            b'<' => (Tok::Lt, 1),
            b'>' if two(self, b'=') => (Tok::Ge, 2),
            b'>' => (Tok::Gt, 1),
            b'=' if two(self, b'=') => (Tok::EqEq, 2),
            b'=' => return Err(self.err(start, "there are no assignments; use `==` to compare")),
            b'!' if two(self, b'=') => (Tok::Ne, 2),
            b'!' => return Err(self.err(start, "use `not` for negation and `!=` for inequality")),
            b'&' => (Tok::And, 1),
            b'|' => (Tok::Or, 1),
            b'(' => (Tok::LParen, 1),
            b')' => (Tok::RParen, 1),
            b',' => (Tok::Comma, 1),
            b'.' => (Tok::Dot, 1),
            _ => {
                // Only ASCII bytes are ever stepped over, so `start` is a char boundary.
                let ch = self.src[start..].chars().next().unwrap_or('\u{fffd}');
                return Err(self.err(start, format!("unexpected character {ch:?}")));
            }
        };
        self.pos = start + len;
        Ok(Token { tok, start, end: self.pos })
    }

    fn number(&mut self) -> Result<Token, ExprError> {
        let start = self.pos;
        self.digits();
        if self.byte(self.pos) == Some(b'.')
            && self.byte(self.pos + 1).is_some_and(|b| b.is_ascii_digit())
        {
            self.pos += 1;
            self.digits();
        }
        if matches!(self.byte(self.pos), Some(b'e' | b'E')) {
            let mut j = self.pos + 1;
            if matches!(self.byte(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if self.byte(j).is_some_and(|b| b.is_ascii_digit()) {
                self.pos = j;
                self.digits();
            }
        }
        if self.byte(self.pos).is_some_and(|b| b.is_ascii_alphabetic() || b == b'_') {
            return Err(self.err(start, "malformed number"));
        }
        let text = &self.src[start..self.pos];
        match text.parse::<f64>() {
            Ok(v) if v.is_finite() => Ok(Token { tok: Tok::Num(v), start, end: self.pos }),
            _ => Err(self.err(start, format!("number `{text}` is out of range"))),
        }
    }

    fn string(&mut self, quote: u8) -> Result<Token, ExprError> {
        let start = self.pos;
        self.pos += 1;
        let body = self.pos;
        loop {
            match self.byte(self.pos) {
                None => return Err(self.err(start, "unterminated string")),
                Some(b) if b == quote => break,
                Some(b) if (0x20..0x7f).contains(&b) && b != b'\\' => self.pos += 1,
                Some(_) => {
                    return Err(self.err(self.pos, "strings are printable ASCII without escapes"))
                }
            }
        }
        let text = self.src[body..self.pos].to_string();
        self.pos += 1;
        Ok(Token { tok: Tok::Str(text), start, end: self.pos })
    }
}
