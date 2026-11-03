//! The one shape a journal record's bytes come from.
//!
//! A hash chain is only worth something if two runs that did the same things
//! write the same bytes, so a record is encoded as canonical JSON: keys sorted
//! by code point, no whitespace, integers as integers, strings escaped to
//! ASCII (`\uXXXX`, surrogate pairs above the basic plane). This is byte for
//! byte what Python's `json.dumps(value, sort_keys=True, separators=(",",
//! ":"))` writes for the same tree, so a host can re-encode a record it read
//! and get the bytes the chain was taken over.
//!
//! There is **no float in the encoding**. A float is written as a JSON string
//! holding its shortest round-trip decimal text (`"100.5"`, `"NaN"`, `"inf"`),
//! so binary rounding can never differ between a writer and a reader, and
//! parsing the text gives back exactly the same number.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A value that can be written into a record.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    /// Written as a string of its shortest round-trip text.
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn map<const N: usize>(entries: [(&str, Value); N]) -> Value {
        Value::Map(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    pub fn opt_str(s: Option<&str>) -> Value {
        s.map_or(Value::Null, Value::str)
    }

    pub fn opt_float(f: Option<f64>) -> Value {
        f.map_or(Value::Null, Value::Float)
    }

    /// The canonical bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = String::with_capacity(128);
        self.write(&mut out);
        out.into_bytes()
    }

    pub fn write(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Value::Float(f) => write_float(out, *f),
            Value::Str(s) => write_str(out, s),
            Value::List(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Value::Map(entries) => {
                out.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(out, key);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }

    /// Parse canonical JSON back. A number with a fraction or an exponent is
    /// refused: the encoding has no float, so one here means the bytes were
    /// not written by this module. Floats come back as the strings they were
    /// written as.
    pub fn from_json(text: &str) -> Result<Value, String> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
        Self::convert(&value)
    }

    fn convert(value: &serde_json::Value) -> Result<Value, String> {
        use serde_json::Value as J;
        Ok(match value {
            J::Null => Value::Null,
            J::Bool(b) => Value::Bool(*b),
            J::Number(n) => match n.as_i64() {
                Some(i) => Value::Int(i),
                None => return Err(format!("{n} is not a 64-bit integer")),
            },
            J::String(s) => Value::Str(s.clone()),
            J::Array(items) => {
                Value::List(items.iter().map(Self::convert).collect::<Result<_, _>>()?)
            }
            J::Object(entries) => Value::Map(
                entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), Self::convert(v)?)))
                    .collect::<Result<_, String>>()?,
            ),
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
}

/// A float as the text the encoding stores.
pub fn float_text(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_string()
    } else if f.is_infinite() {
        if f > 0.0 { "inf" } else { "-inf" }.to_string()
    } else if f == 0.0 && f.is_sign_negative() {
        "-0".to_string()
    } else {
        // Rust's Display for f64 is the shortest text that parses back to the
        // same bits, and never uses an exponent.
        format!("{f}")
    }
}

fn write_float(out: &mut String, f: f64) {
    out.push('"');
    out.push_str(&float_text(f));
    out.push('"');
}

/// A JSON string as Python's `json.dumps(..., ensure_ascii=True)` writes it.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_sort_and_nothing_pads() {
        let v = Value::map([("b", Value::Int(1)), ("a", Value::str("x"))]);
        assert_eq!(v.to_bytes(), br#"{"a":"x","b":1}"#);
    }

    #[test]
    fn floats_are_exact_text_and_round_trip() {
        for f in [0.1, 100.5, 1e21, 1e-7, -3.0, 123_456_789.123_456_79, f64::MIN_POSITIVE] {
            let text = float_text(f);
            assert_eq!(text.parse::<f64>().unwrap().to_bits(), f.to_bits(), "{text}");
        }
        assert_eq!(float_text(f64::NAN), "NaN");
        assert_eq!(float_text(-0.0), "-0");
        assert_eq!("-0".parse::<f64>().unwrap().to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn strings_escape_like_python_ensure_ascii() {
        let mut out = String::new();
        write_str(&mut out, "a\"b\\c\n\u{e9}\u{1f600}");
        assert_eq!(out, "\"a\\\"b\\\\c\\n\\u00e9\\ud83d\\ude00\"");
    }

    #[test]
    fn parse_back_refuses_json_floats() {
        assert!(Value::from_json(r#"{"a":1.5}"#).is_err());
        assert_eq!(
            Value::from_json(r#"{"a":"1.5"}"#).unwrap().get("a").unwrap().as_str(),
            Some("1.5")
        );
    }
}
