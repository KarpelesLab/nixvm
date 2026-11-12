//! A small JSON reader: enough for registry manifests, image configs and
//! token responses, without a dependency.

use std::fmt::Write as _;

/// A parsed JSON value. Objects keep their keys in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// The member `key` of an object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_array(&self) -> &[Value] {
        match self {
            Self::Arr(a) => a,
            _ => &[],
        }
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    /// The member `key` as a string.
    #[must_use]
    pub fn str_at(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// The member `key` as a list of strings (a missing or `null` one is
    /// empty, as an image config's `Cmd` may be).
    #[must_use]
    pub fn strings_at(&self, key: &str) -> Vec<String> {
        self.get(key)
            .map(|v| {
                v.as_array()
                    .iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Parse a JSON document.
///
/// # Errors
/// A message naming the byte offset of the first thing that isn't JSON.
pub fn parse(text: &[u8]) -> Result<Value, String> {
    let mut p = Parser { s: text, i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

/// `s` as a JSON string literal.
#[must_use]
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Nesting deeper than this is refused rather than recursed into.
const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn eat(&mut self, lit: &[u8]) -> bool {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return self.err("nested too deep");
        }
        self.ws();
        match self.s.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'{') => {
                self.i += 1;
                let mut members = Vec::new();
                self.ws();
                if self.eat(b"}") {
                    return Ok(Value::Obj(members));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return self.err("expected a key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if !self.eat(b":") {
                        return self.err("expected ':'");
                    }
                    let v = self.value(depth + 1)?;
                    members.push((k, v));
                    self.ws();
                    if self.eat(b",") {
                        continue;
                    }
                    if self.eat(b"}") {
                        return Ok(Value::Obj(members));
                    }
                    return self.err("expected ',' or '}'");
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat(b"]") {
                    return Ok(Value::Arr(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    if self.eat(b",") {
                        continue;
                    }
                    if self.eat(b"]") {
                        return Ok(Value::Arr(items));
                    }
                    return self.err("expected ',' or ']'");
                }
            }
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') if self.eat(b"true") => Ok(Value::Bool(true)),
            Some(b'f') if self.eat(b"false") => Ok(Value::Bool(false)),
            Some(b'n') if self.eat(b"null") => Ok(Value::Null),
            Some(b'-' | b'0'..=b'9') => {
                let start = self.i;
                while self.i < self.s.len()
                    && matches!(
                        self.s[self.i],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                    )
                {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
                text.parse()
                    .map(Value::Num)
                    .or_else(|_| self.err("bad number"))
            }
            Some(_) => self.err("unexpected character"),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let Some(h) = self.s.get(self.i..self.i + 4) else {
            return self.err("short \\u escape");
        };
        let h = std::str::from_utf8(h).unwrap_or("");
        let v = u32::from_str_radix(h, 16).or_else(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out = Vec::new();
        loop {
            let Some(&b) = self.s.get(self.i) else {
                return self.err("unterminated string");
            };
            self.i += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        return self.err("unterminated escape");
                    };
                    self.i += 1;
                    let c = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xd800..0xdc00).contains(&hi) && self.eat(b"\\u") {
                                let lo = self.hex4()?;
                                0x10000 + ((hi - 0xd800) << 10) + (lo.wrapping_sub(0xdc00) & 0x3ff)
                            } else {
                                hi
                            };
                            char::from_u32(cp).unwrap_or('\u{fffd}')
                        }
                        _ => return self.err("bad escape"),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                _ => out.push(b),
            }
        }
        String::from_utf8(out).or_else(|_| self.err("string isn't UTF-8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_what_registries_send() {
        let v = parse(
            br#"{"schemaVersion":2,"manifests":[{"digest":"sha256:ab","platform":{"architecture":"amd64","os":"linux"}}],
               "Cmd":null,"Env":["A=\"b\"","\u00e9\ud83d\ude00"],"n":-1.5e2,"t":true}"#,
        )
        .unwrap();
        assert_eq!(v.get("schemaVersion").and_then(Value::as_u64), Some(2));
        let m = &v.get("manifests").unwrap().as_array()[0];
        assert_eq!(m.get("platform").unwrap().str_at("os"), Some("linux"));
        assert_eq!(v.strings_at("Cmd"), Vec::<String>::new());
        assert_eq!(v.strings_at("Env"), vec!["A=\"b\"", "é😀"]);
        assert_eq!(v.get("n"), Some(&Value::Num(-150.0)));
        assert!(parse(b"{\"a\":1,}").is_err());
        assert!(parse(b"[1] x").is_err());
        assert_eq!(quote("a\"b\n"), r#""a\"b\n""#);
    }
}
