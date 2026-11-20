use crate::costs::Costs;
use crate::field::FieldSize;
use crate::parameters::{Layout, Parameters, Round, Stage, StageParameters, Variant};
use crate::soundness::{CodeBounds, RoundError, Soundness};
use crate::tuning::Analysis;
use std::fmt;
use std::ops::{Index, RangeInclusive};

pub const DEPTH_MAX: usize = 128;

const EXPONENT_FIXED_RANGE: RangeInclusive<i32> = -5..=15;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!message.is_empty());
        Self { message }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub trait ToJson {
    fn to_json(&self) -> Value;
}

pub trait FromJson: Sized {
    fn from_json(value: &Value) -> Result<Self>;
}

pub fn to_string<T: ToJson + ?Sized>(value: &T) -> String {
    value.to_json().to_string_compact()
}

pub fn to_string_pretty<T: ToJson + ?Sized>(value: &T) -> String {
    value.to_json().to_string_pretty()
}

pub fn from_slice<T: FromJson>(bytes: &[u8]) -> Result<T> {
    T::from_json(&parse(bytes)?)
}

static NULL: Value = Value::Null;

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries.iter().find(|(name, _)| name == key).map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::Unsigned(value) => Some(value as f64),
            Value::Signed(value) => Some(value as f64),
            Value::Float(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::Unsigned(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn sorted(self) -> Value {
        match self {
            Value::Array(items) => Value::Array(items.into_iter().map(Value::sorted).collect()),
            Value::Object(entries) => {
                let mut entries: Vec<(String, Value)> =
                    entries.into_iter().map(|(key, value)| (key, value.sorted())).collect();
                let count = entries.len();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                assert_eq!(entries.len(), count);
                assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0), "duplicate keys");
                Value::Object(entries)
            }
            other => other,
        }
    }

    pub fn to_string_compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, None);
        assert!(!out.is_empty());
        out
    }

    pub fn to_string_pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, Some(0));
        assert!(!out.is_empty());
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>) {
        assert!(indent.is_none_or(|level| level <= DEPTH_MAX));
        let len_before = out.len();
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Value::Unsigned(value) => out.push_str(&value.to_string()),
            Value::Signed(value) => out.push_str(&value.to_string()),
            Value::Float(value) => write_f64(*value, out),
            Value::String(text) => write_string(text, out),
            Value::Array(items) => {
                let count = items.len();
                write_sequence(out, indent, b"[]", count, |out, index, inner| items[index].write(out, inner));
            }
            Value::Object(entries) => {
                let count = entries.len();
                write_sequence(out, indent, b"{}", count, |out, index, inner| {
                    let (key, value) = &entries[index];
                    write_string(key, out);
                    out.push_str(if inner.is_some() { ": " } else { ":" });
                    value.write(out, inner);
                });
            }
        }
        assert!(out.len() > len_before);
    }
}

impl Index<&str> for Value {
    type Output = Value;

    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&NULL)
    }
}

impl Index<usize> for Value {
    type Output = Value;

    fn index(&self, index: usize) -> &Value {
        match self {
            Value::Array(items) => items.get(index).unwrap_or(&NULL),
            _ => &NULL,
        }
    }
}

fn write_sequence(
    out: &mut String,
    indent: Option<usize>,
    brackets: &[u8; 2],
    count: usize,
    mut write_item: impl FnMut(&mut String, usize, Option<usize>),
) {
    assert!(brackets == b"[]" || brackets == b"{}");
    let len_before = out.len();
    out.push(char::from(brackets[0]));
    let inner = indent.map(|level| level + 1);
    for index in 0..count {
        if index > 0 {
            out.push(',');
        }
        if let Some(level) = inner {
            out.push('\n');
            push_indent(out, level);
        }
        write_item(out, index, inner);
    }
    if let (Some(level), true) = (indent, count > 0) {
        out.push('\n');
        push_indent(out, level);
    }
    out.push(char::from(brackets[1]));
    assert!(out.len() >= len_before + 2);
}

fn push_indent(out: &mut String, level: usize) {
    assert!(level <= DEPTH_MAX * 4);
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_string(text: &str, out: &mut String) {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let len_before = out.len();
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0}'..='\u{1f}' => {
                let code = character as usize;
                out.push_str("\\u00");
                out.push(char::from(HEX_DIGITS[code >> 4]));
                out.push(char::from(HEX_DIGITS[code & 0xf]));
            }
            _ => out.push(character),
        }
    }
    out.push('"');
    assert!(out.len() >= len_before + 2 + text.len());
}

pub fn format_f64(value: f64) -> String {
    let mut out = String::new();
    write_f64(value, &mut out);
    out
}

fn write_f64(value: f64, out: &mut String) {
    if !value.is_finite() {
        out.push_str("null");
        return;
    }
    let len_before = out.len();
    let (digits, exponent) = shortest_digits(value.abs());
    if value.is_sign_negative() {
        out.push('-');
    }
    if EXPONENT_FIXED_RANGE.contains(&exponent) {
        write_fixed(&digits, exponent, out);
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        out.push_str(&exponent.unsigned_abs().to_string());
    }
    assert!(out.len() > len_before + digits.len());
}

fn scientific_digits(text: &str) -> (String, i32) {
    let (mantissa, exponent) = text.split_once('e').expect("an exponent");
    let exponent: i32 = exponent.parse().expect("a decimal exponent");
    let digits: String = mantissa.chars().filter(|character| *character != '.').collect();
    assert!(!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()));
    assert!(!digits.starts_with('0') || digits.bytes().all(|byte| byte == b'0'));
    (digits, exponent)
}

fn shortest_digits(magnitude: f64) -> (String, i32) {
    const EXACT_DIGITS_MAX: usize = 800;
    assert!(magnitude.is_finite() && magnitude.is_sign_positive());
    let (digits, exponent) = scientific_digits(&format!("{magnitude:e}"));
    let len = digits.len();
    let (rounded, _) = scientific_digits(&format!("{magnitude:.len$e}"));
    if !rounded.ends_with('5') {
        return (digits, exponent);
    }
    let (exact, exact_exponent) = scientific_digits(&format!("{magnitude:.EXACT_DIGITS_MAX$e}"));
    let exact = exact.trim_end_matches('0');
    if exact.len() != len + 1 || exact_exponent != exponent || !digits.ends_with(['1', '3', '5', '7', '9']) {
        return (digits, exponent);
    }
    let last = digits.as_bytes()[len - 1];
    let other_last = if exact.as_bytes()[len - 1] == last { last + 1 } else { last - 1 };
    let other = format!("{}{}", &digits[..len - 1], char::from(other_last));
    let other_value: f64 =
        format!("{other}e{}", exponent - len as i32 + 1).parse().expect("a decimal number");
    assert!(other_last.is_ascii_digit() && other_last % 2 == 0);
    if other_value == magnitude { (other, exponent) } else { (digits, exponent) }
}

fn write_fixed(digits: &str, exponent: i32, out: &mut String) {
    assert!(EXPONENT_FIXED_RANGE.contains(&exponent));
    assert!(digits.len() <= 17);
    let point_at = exponent + 1;
    let len = digits.len() as i32;
    if point_at >= len {
        out.push_str(digits);
        for _ in len..point_at {
            out.push('0');
        }
        out.push_str(".0");
    } else if point_at > 0 {
        out.push_str(&digits[..point_at as usize]);
        out.push('.');
        out.push_str(&digits[point_at as usize..]);
    } else {
        out.push_str("0.");
        for _ in point_at..0 {
            out.push('0');
        }
        out.push_str(digits);
    }
}

pub fn parse(bytes: &[u8]) -> Result<Value> {
    let mut parser = Parser { bytes, index: 0, depth: 0 };
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.index != bytes.len() {
        return Err(parser.error("trailing characters"));
    }
    assert_eq!(parser.depth, 0);
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    index: usize,
    depth: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.index).copied()
    }

    fn error(&self, message: &str) -> Error {
        assert!(self.index <= self.bytes.len());
        let consumed = &self.bytes[..self.index];
        let line = 1 + consumed.iter().filter(|&&byte| byte == b'\n').count();
        let line_start = consumed.iter().rposition(|&byte| byte == b'\n').map_or(0, |at| at + 1);
        let column = self.index - line_start;
        Error::new(format!("{message} at line {line} column {column}"))
    }

    fn skip_whitespace(&mut self) {
        let index_before = self.index;
        while let Some(b' ' | b'\n' | b'\t' | b'\r') = self.peek() {
            self.index += 1;
        }
        assert!(self.index >= index_before && self.index <= self.bytes.len());
    }

    fn value(&mut self) -> Result<Value> {
        assert!(self.depth < DEPTH_MAX);
        self.skip_whitespace();
        let index_before = self.index;
        let value = match self.peek() {
            None => return Err(self.error("EOF while parsing a value")),
            Some(b'n') => self.literal(b"null", Value::Null)?,
            Some(b't') => self.literal(b"true", Value::Bool(true))?,
            Some(b'f') => self.literal(b"false", Value::Bool(false))?,
            Some(b'"') => Value::String(self.string()?),
            Some(b'-' | b'0'..=b'9') => self.number()?,
            Some(b'[') => self.array()?,
            Some(b'{') => self.object()?,
            Some(_) => return Err(self.error("expected value")),
        };
        assert!(self.index > index_before);
        Ok(value)
    }

    fn literal(&mut self, text: &[u8], value: Value) -> Result<Value> {
        assert!(!text.is_empty());
        for &expected in text {
            match self.peek() {
                Some(byte) if byte == expected => self.index += 1,
                Some(_) => return Err(self.error("expected ident")),
                None => return Err(self.error("EOF while parsing a value")),
            }
        }
        assert!(self.index <= self.bytes.len());
        Ok(value)
    }

    fn enter(&mut self) -> Result<()> {
        assert!(self.depth < DEPTH_MAX);
        self.depth += 1;
        if self.depth >= DEPTH_MAX {
            return Err(self.error("recursion limit exceeded"));
        }
        self.index += 1;
        Ok(())
    }

    fn leave(&mut self) {
        assert!(self.depth > 0);
        self.depth -= 1;
        self.index += 1;
    }

    fn array(&mut self) -> Result<Value> {
        assert_eq!(self.peek(), Some(b'['));
        self.enter()?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.leave();
            return Ok(Value::Array(items));
        }
        for _ in 0..self.bytes.len() {
            items.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.index += 1,
                Some(b']') => {
                    self.leave();
                    assert!(!items.is_empty());
                    return Ok(Value::Array(items));
                }
                Some(_) => return Err(self.error("expected `,` or `]`")),
                None => return Err(self.error("EOF while parsing a list")),
            }
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                return Err(self.error("trailing comma"));
            }
        }
        Err(self.error("EOF while parsing a list"))
    }

    fn object(&mut self) -> Result<Value> {
        assert_eq!(self.peek(), Some(b'{'));
        self.enter()?;
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.leave();
            return Ok(Value::Object(entries));
        }
        for _ in 0..self.bytes.len() {
            entries.push(self.entry()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.index += 1,
                Some(b'}') => {
                    self.leave();
                    assert!(!entries.is_empty());
                    return Ok(Value::Object(entries));
                }
                Some(_) => return Err(self.error("expected `,` or `}`")),
                None => return Err(self.error("EOF while parsing an object")),
            }
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                return Err(self.error("trailing comma"));
            }
        }
        Err(self.error("EOF while parsing an object"))
    }

    fn entry(&mut self) -> Result<(String, Value)> {
        assert!(self.depth > 0);
        self.skip_whitespace();
        match self.peek() {
            Some(b'"') => {}
            Some(_) => return Err(self.error("key must be a string")),
            None => return Err(self.error("EOF while parsing an object")),
        }
        let key = self.string()?;
        assert_eq!(self.bytes[self.index - 1], b'"');
        self.skip_whitespace();
        match self.peek() {
            Some(b':') => self.index += 1,
            Some(_) => return Err(self.error("expected `:`")),
            None => return Err(self.error("EOF while parsing an object")),
        }
        let value = self.value()?;
        assert!(self.index <= self.bytes.len());
        Ok((key, value))
    }

    fn string(&mut self) -> Result<String> {
        assert_eq!(self.peek(), Some(b'"'));
        self.index += 1;
        let mut text = Vec::new();
        assert!(self.index <= self.bytes.len());
        for _ in 0..self.bytes.len() {
            let start = self.index;
            while self.peek().is_some_and(|byte| byte != b'"' && byte != b'\\' && byte >= 0x20) {
                self.index += 1;
            }
            text.extend_from_slice(&self.bytes[start..self.index]);
            match self.peek() {
                Some(b'"') => {
                    self.index += 1;
                    return String::from_utf8(text).map_err(|_| self.error("invalid unicode code point"));
                }
                Some(b'\\') => {
                    self.index += 1;
                    self.escape(&mut text)?;
                }
                Some(_) => {
                    return Err(
                        self.error("control character (\\u0000-\\u001F) found while parsing a string")
                    );
                }
                None => return Err(self.error("EOF while parsing a string")),
            }
        }
        Err(self.error("EOF while parsing a string"))
    }

    fn escape(&mut self, text: &mut Vec<u8>) -> Result<()> {
        assert_eq!(self.bytes[self.index - 1], b'\\');
        let len_before = text.len();
        let Some(byte) = self.peek() else {
            return Err(self.error("EOF while parsing a string"));
        };
        self.index += 1;
        let character = match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => self.unicode_escape()?,
            _ => return Err(self.error("invalid escape")),
        };
        let mut buffer = [0; 4];
        text.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
        assert!(text.len() > len_before);
        Ok(())
    }

    fn unicode_escape(&mut self) -> Result<char> {
        assert_eq!(self.bytes[self.index - 1], b'u');
        let high = self.hex_escape()?;
        if (0xDC00..=0xDFFF).contains(&high) {
            return Err(self.error("lone leading surrogate in hex escape"));
        }
        if !(0xD800..=0xDBFF).contains(&high) {
            return char::from_u32(high).ok_or_else(|| self.error("invalid unicode code point"));
        }
        if self.peek() != Some(b'\\') || self.bytes.get(self.index + 1) != Some(&b'u') {
            return Err(self.error("unexpected end of hex escape"));
        }
        self.index += 2;
        let low = self.hex_escape()?;
        if !(0xDC00..=0xDFFF).contains(&low) {
            return Err(self.error("lone leading surrogate in hex escape"));
        }
        let code = 0x1_0000 + ((high - 0xD800) << 10) + (low - 0xDC00);
        assert!((0x1_0000..=0x10_FFFF).contains(&code));
        char::from_u32(code).ok_or_else(|| self.error("invalid unicode code point"))
    }

    fn hex_escape(&mut self) -> Result<u32> {
        assert!(self.index <= self.bytes.len());
        let Some(digits) = self.bytes.get(self.index..self.index + 4) else {
            self.index = self.bytes.len();
            return Err(self.error("EOF while parsing a string"));
        };
        let mut code = 0;
        for &digit in digits {
            let Some(value) = char::from(digit).to_digit(16) else {
                return Err(self.error("invalid escape"));
            };
            code = code * 16 + value;
        }
        self.index += 4;
        assert!(code <= 0xFFFF);
        Ok(code)
    }

    fn digits(&mut self) -> usize {
        let start = self.index;
        while let Some(b'0'..=b'9') = self.peek() {
            self.index += 1;
        }
        assert!(self.index >= start);
        self.index - start
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.index;
        assert!(matches!(self.peek(), Some(b'-' | b'0'..=b'9')));
        let negative = self.peek() == Some(b'-');
        if negative {
            self.index += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.index += 1;
                if let Some(b'0'..=b'9') = self.peek() {
                    return Err(self.error("invalid number"));
                }
            }
            Some(b'1'..=b'9') => {
                self.digits();
            }
            Some(_) => return Err(self.error("invalid number")),
            None => return Err(self.error("EOF while parsing a value")),
        }
        let integer_end = self.index;
        let fraction = self.peek() == Some(b'.');
        if fraction {
            self.index += 1;
            if self.digits() == 0 {
                return Err(self.error("invalid number"));
            }
        }
        let exponent = matches!(self.peek(), Some(b'e' | b'E'));
        if exponent {
            self.index += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.index += 1;
            }
            if self.digits() == 0 {
                return Err(self.error("invalid number"));
            }
        }
        assert!(integer_end > start && self.index >= integer_end);
        self.number_value(start, negative, fraction || exponent)
    }

    fn number_value(&self, start: usize, negative: bool, is_float: bool) -> Result<Value> {
        let text =
            std::str::from_utf8(&self.bytes[start..self.index]).map_err(|_| self.error("invalid number"))?;
        assert!(!text.is_empty());
        if !is_float {
            let magnitude = text.strip_prefix('-').unwrap_or(text);
            if let Ok(magnitude) = magnitude.parse::<u64>() {
                return Ok(integer_value(negative, magnitude));
            }
        }
        let value: f64 = text.parse().map_err(|_| self.error("invalid number"))?;
        if !value.is_finite() {
            return Err(self.error("number out of range"));
        }
        assert_eq!(value.is_sign_negative(), negative);
        Ok(Value::Float(value))
    }
}

fn integer_value(negative: bool, magnitude: u64) -> Value {
    const SIGNED_MAGNITUDE_MAX: u64 = 1 << 63;
    if !negative {
        return Value::Unsigned(magnitude);
    }
    if magnitude == 0 {
        return Value::Float(-0.0);
    }
    assert!(negative && magnitude > 0);
    if magnitude <= SIGNED_MAGNITUDE_MAX {
        let value = (-i128::from(magnitude)) as i64;
        assert!(value < 0);
        return Value::Signed(value);
    }
    Value::Float(-(magnitude as f64))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unknown {
    Deny,
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    ObjectOrArray,
    Object,
}

pub fn fields<'a, const N: usize>(
    value: &'a Value,
    names: [&str; N],
    unknown: Unknown,
    shape: Shape,
) -> Result<[Option<&'a Value>; N]> {
    assert!(N > 0);
    assert!(names.iter().all(|name| !name.is_empty()));
    let mut slots = [None; N];
    match value {
        Value::Object(entries) => {
            for (key, item) in entries {
                match names.iter().position(|name| name == key) {
                    Some(index) if slots[index].is_some() => {
                        return Err(Error::new(format!("duplicate field `{key}`")));
                    }
                    Some(index) => slots[index] = Some(item),
                    None if unknown == Unknown::Deny => {
                        let expected = names.map(|name| format!("`{name}`")).join(", ");
                        return Err(Error::new(format!("unknown field `{key}`, expected one of {expected}")));
                    }
                    None => {}
                }
            }
        }
        Value::Array(items) if shape == Shape::ObjectOrArray => {
            if items.len() > N {
                return Err(Error::new(format!("invalid length {}, expected {N} elements", items.len())));
            }
            for (slot, item) in slots.iter_mut().zip(items) {
                *slot = Some(item);
            }
        }
        other => return Err(invalid_type(other, "a struct")),
    }
    Ok(slots)
}

pub fn required<T: FromJson>(slot: Option<&Value>, name: &str) -> Result<T> {
    assert!(!name.is_empty());
    match slot {
        Some(value) => T::from_json(value),
        None => Err(Error::new(format!("missing field `{name}`"))),
    }
}

pub fn defaulted<T: FromJson + Default>(slot: Option<&Value>) -> Result<T> {
    match slot {
        Some(value) => T::from_json(value),
        None => Ok(T::default()),
    }
}

pub fn invalid_type(value: &Value, expected: &str) -> Error {
    assert!(!expected.is_empty());
    let found = match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => format!("boolean `{value}`"),
        Value::Unsigned(value) => format!("integer `{value}`"),
        Value::Signed(value) => format!("integer `{value}`"),
        Value::Float(value) => format!("floating point `{}`", format_f64(*value)),
        Value::String(text) => format!("string {text:?}"),
        Value::Array(_) => "sequence".to_string(),
        Value::Object(_) => "map".to_string(),
    };
    assert!(!found.is_empty());
    Error::new(format!("invalid type: {found}, expected {expected}"))
}

impl FromJson for Value {
    fn from_json(value: &Value) -> Result<Self> {
        Ok(value.clone())
    }
}

impl FromJson for u64 {
    fn from_json(value: &Value) -> Result<Self> {
        match *value {
            Value::Unsigned(value) => Ok(value),
            Value::Signed(value) => {
                assert!(value < 0);
                Err(Error::new(format!("invalid value: integer `{value}`, expected an unsigned integer")))
            }
            _ => Err(invalid_type(value, "an unsigned integer")),
        }
    }
}

impl FromJson for usize {
    fn from_json(value: &Value) -> Result<Self> {
        let value = u64::from_json(value)?;
        let result =
            usize::try_from(value).map_err(|_| Error::new(format!("integer `{value}` too large")))?;
        assert_eq!(result as u64, value);
        Ok(result)
    }
}

impl FromJson for u32 {
    fn from_json(value: &Value) -> Result<Self> {
        let value = u64::from_json(value)?;
        let result = u32::try_from(value)
            .map_err(|_| Error::new(format!("invalid value: integer `{value}`, expected u32")))?;
        assert_eq!(u64::from(result), value);
        Ok(result)
    }
}

impl FromJson for f64 {
    fn from_json(value: &Value) -> Result<Self> {
        let result = value.as_f64().ok_or_else(|| invalid_type(value, "f64"))?;
        if !result.is_finite() {
            return Err(Error::new("a number is not finite"));
        }
        assert!(matches!(value, Value::Unsigned(_) | Value::Signed(_) | Value::Float(_)));
        Ok(result)
    }
}

impl FromJson for String {
    fn from_json(value: &Value) -> Result<Self> {
        match value {
            Value::String(text) => Ok(text.clone()),
            other => Err(invalid_type(other, "a string")),
        }
    }
}

impl<T: FromJson> FromJson for Vec<T> {
    fn from_json(value: &Value) -> Result<Self> {
        match value {
            Value::Array(items) => items.iter().map(T::from_json).collect(),
            other => Err(invalid_type(other, "a sequence")),
        }
    }
}

impl<T: FromJson> FromJson for std::collections::BTreeMap<String, T> {
    fn from_json(value: &Value) -> Result<Self> {
        let Value::Object(entries) = value else {
            return Err(invalid_type(value, "a map"));
        };
        let mut map = std::collections::BTreeMap::new();
        for (key, item) in entries {
            map.insert(key.clone(), T::from_json(item)?);
        }
        assert!(map.len() <= entries.len());
        Ok(map)
    }
}

impl ToJson for u64 {
    fn to_json(&self) -> Value {
        Value::Unsigned(*self)
    }
}

impl ToJson for usize {
    fn to_json(&self) -> Value {
        Value::Unsigned(*self as u64)
    }
}

impl ToJson for u32 {
    fn to_json(&self) -> Value {
        Value::Unsigned(u64::from(*self))
    }
}

impl ToJson for f64 {
    fn to_json(&self) -> Value {
        Value::Float(*self)
    }
}

impl ToJson for str {
    fn to_json(&self) -> Value {
        Value::String(self.to_string())
    }
}

impl ToJson for String {
    fn to_json(&self) -> Value {
        Value::String(self.clone())
    }
}

impl<T: ToJson> ToJson for [T] {
    fn to_json(&self) -> Value {
        Value::Array(self.iter().map(T::to_json).collect())
    }
}

impl<T: ToJson> ToJson for Vec<T> {
    fn to_json(&self) -> Value {
        self.as_slice().to_json()
    }
}

pub fn object<const N: usize>(entries: [(&str, Value); N]) -> Value {
    let entries: Vec<(String, Value)> =
        entries.into_iter().map(|(key, value)| (key.to_string(), value)).collect();
    assert!(entries.iter().all(|(key, _)| !key.is_empty()));
    assert_eq!(entries.len(), N);
    Value::Object(entries)
}

fn without_zeros(value: Value, keys: &[&str]) -> Value {
    let Value::Object(entries) = value else {
        unreachable!("an object");
    };
    let kept = entries.into_iter().filter(|(key, value)| {
        let is_zero = *value == Value::Unsigned(0);
        !(is_zero && keys.contains(&key.as_str()))
    });
    Value::Object(kept.collect())
}

const PARAMETERS_FIELDS: [&str; 6] =
    ["num_variables", "stages", "security_bits", "pow_bits", "base_case_pow_bits", "johnson_slack"];

const STAGE_PARAMETERS_FIELDS: [&str; 5] =
    ["folding", "domain_size", "queries", "sumcheck_pow_bits", "ood_pow_bits"];

impl ToJson for Parameters {
    fn to_json(&self) -> Value {
        let value = object([
            ("num_variables", self.num_variables.to_json()),
            ("stages", self.stages.to_json()),
            ("security_bits", self.security_bits.to_json()),
            ("pow_bits", self.pow_bits.to_json()),
            ("base_case_pow_bits", self.base_case_pow_bits.to_json()),
            ("johnson_slack", self.johnson_slack.to_json()),
        ]);
        without_zeros(value, &["base_case_pow_bits"])
    }
}

impl FromJson for Parameters {
    fn from_json(value: &Value) -> Result<Self> {
        let [num_variables, stages, security_bits, pow_bits, base_case_pow_bits, johnson_slack] =
            fields(value, PARAMETERS_FIELDS, Unknown::Deny, Shape::ObjectOrArray)?;
        Ok(Parameters {
            num_variables: required(num_variables, PARAMETERS_FIELDS[0])?,
            stages: required(stages, PARAMETERS_FIELDS[1])?,
            security_bits: required(security_bits, PARAMETERS_FIELDS[2])?,
            pow_bits: required(pow_bits, PARAMETERS_FIELDS[3])?,
            base_case_pow_bits: defaulted(base_case_pow_bits)?,
            johnson_slack: required(johnson_slack, PARAMETERS_FIELDS[5])?,
        })
    }
}

impl ToJson for StageParameters {
    fn to_json(&self) -> Value {
        let value = object([
            ("folding", self.folding.to_json()),
            ("domain_size", self.domain_size.to_json()),
            ("queries", self.queries.to_json()),
            ("sumcheck_pow_bits", self.sumcheck_pow_bits.to_json()),
            ("ood_pow_bits", self.ood_pow_bits.to_json()),
        ]);
        without_zeros(value, &["sumcheck_pow_bits", "ood_pow_bits"])
    }
}

impl FromJson for StageParameters {
    fn from_json(value: &Value) -> Result<Self> {
        let [folding, domain_size, queries, sumcheck_pow_bits, ood_pow_bits] =
            fields(value, STAGE_PARAMETERS_FIELDS, Unknown::Deny, Shape::ObjectOrArray)?;
        Ok(StageParameters {
            folding: required(folding, STAGE_PARAMETERS_FIELDS[0])?,
            domain_size: required(domain_size, STAGE_PARAMETERS_FIELDS[1])?,
            queries: required(queries, STAGE_PARAMETERS_FIELDS[2])?,
            sumcheck_pow_bits: defaulted(sumcheck_pow_bits)?,
            ood_pow_bits: defaulted(ood_pow_bits)?,
        })
    }
}

impl ToJson for Round {
    fn to_json(&self) -> Value {
        assert!(!matches!(self, Round::Fold(0)));
        let name = match *self {
            Round::SumcheckBatching => "sumcheck_batching",
            Round::Fold(index) => return object([("fold", index.to_json())]),
            Round::OutOfDomain => "out_of_domain",
            Round::QueryAndBatch => "query_and_batch",
            Round::BaseCaseCombination => "base_case_combination",
            Round::BaseCaseQuery => "base_case_query",
        };
        assert!(name.bytes().all(|byte| byte == b'_' || byte.is_ascii_lowercase()));
        name.to_json()
    }
}

const VARIANT_NAMES: [(Variant, &str); 2] =
    [(Variant::ZeroKnowledge, "zero_knowledge"), (Variant::Baseline, "baseline")];

impl ToJson for Variant {
    fn to_json(&self) -> Value {
        let (_, name) = VARIANT_NAMES.iter().find(|(variant, _)| variant == self).expect("a named variant");
        name.to_json()
    }
}

impl FromJson for Variant {
    fn from_json(value: &Value) -> Result<Self> {
        let name = match value {
            Value::String(name) => name,
            Value::Object(entries) if entries.len() == 1 && entries[0].1 == Value::Null => &entries[0].0,
            other => return Err(invalid_type(other, "enum Variant")),
        };
        let found = VARIANT_NAMES.iter().find(|(_, variant_name)| variant_name == name);
        let (variant, _) = found.ok_or_else(|| {
            Error::new(format!("unknown variant `{name}`, expected `zero_knowledge` or `baseline`"))
        })?;
        Ok(*variant)
    }
}

impl ToJson for Stage {
    fn to_json(&self) -> Value {
        object([
            ("folding", self.folding.to_json()),
            ("main_len", self.main_len.to_json()),
            ("aux_len", self.aux_len.to_json()),
            ("mask_len", self.mask_len.to_json()),
            ("seed_len", self.seed_len.to_json()),
            ("pad_len", self.pad_len.to_json()),
            ("domain_size", self.domain_size.to_json()),
            ("queries", self.queries.to_json()),
        ])
    }
}

impl ToJson for Layout {
    fn to_json(&self) -> Value {
        object([("variant", self.variant.to_json()), ("stages", self.stages.to_json())])
    }
}

impl ToJson for FieldSize {
    fn to_json(&self) -> Value {
        object([("bits", self.bits.to_json()), ("bytes", self.bytes.to_json())])
    }
}

impl ToJson for Costs {
    fn to_json(&self) -> Value {
        object([
            ("oracle_bytes", self.oracle_bytes.to_json()),
            ("message_bytes", self.message_bytes.to_json()),
            ("pow_nonce_bytes", self.pow_nonce_bytes.to_json()),
            ("iop_bytes", self.iop_bytes.to_json()),
            ("root_bytes", self.root_bytes.to_json()),
            ("expected_opening_bytes", self.expected_opening_bytes.to_json()),
            ("expected_argument_bytes", self.expected_argument_bytes.to_json()),
        ])
    }
}

impl ToJson for CodeBounds {
    fn to_json(&self) -> Value {
        object([
            ("dimension", self.dimension.to_json()),
            ("rate", self.rate.to_json()),
            ("delta", self.delta.to_json()),
            ("list_size", self.list_size.to_json()),
            ("mca_numerator", self.mca_numerator.to_json()),
        ])
    }
}

impl ToJson for RoundError {
    fn to_json(&self) -> Value {
        object([
            ("stage", self.stage.to_json()),
            ("round", self.round.to_json()),
            ("pow_bits", self.pow_bits.to_json()),
            ("algebraic", self.algebraic.to_json()),
            ("query", self.query.to_json()),
        ])
    }
}

impl ToJson for Soundness {
    fn to_json(&self) -> Value {
        object([
            ("codes", self.codes.to_json()),
            ("rounds", self.rounds.to_json()),
            ("algebraic_bits", self.algebraic_bits.to_json()),
            ("query_bits", self.query_bits.to_json()),
            ("total_bits", self.total_bits.to_json()),
        ])
    }
}

impl ToJson for Analysis {
    fn to_json(&self) -> Value {
        object([
            ("parameters", self.parameters.to_json()),
            ("field", self.field.to_json()),
            ("layout", self.layout.to_json()),
            ("soundness", self.soundness.to_json()),
            ("costs", self.costs.to_json()),
        ])
    }
}
