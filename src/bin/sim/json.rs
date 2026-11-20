#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

const DEPTH_MAX: usize = 64;

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut parser = Parser { bytes: text.as_bytes(), at: 0 };
    let value = parser.value(0)?;
    parser.skip_space();
    if parser.at != parser.bytes.len() {
        return Err(format!("trailing bytes at {}", parser.at));
    }
    assert!(parser.at <= parser.bytes.len());
    Ok(value)
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while self.at < self.bytes.len() && self.bytes[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
        assert!(self.at <= self.bytes.len());
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        self.skip_space();
        if self.peek() != Some(byte) {
            return Err(format!("expected {:?} at {}", byte as char, self.at));
        }
        self.at += 1;
        Ok(())
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > DEPTH_MAX {
            return Err("nested too deeply".into());
        }
        self.skip_space();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Text(self.text()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(_) => self.number(),
            None => Err("unexpected end".into()),
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        assert!(!word.is_empty());
        if !self.bytes[self.at..].starts_with(word.as_bytes()) {
            return Err(format!("expected {word} at {}", self.at));
        }
        self.at += word.len();
        Ok(value)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while self.peek().is_some_and(|b| b.is_ascii_digit() || b"+-.eE".contains(&b)) {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|e| e.to_string())?;
        let number: f64 = text.parse().map_err(|_| format!("not a number at {start}: {text:?}"))?;
        assert!(self.at > start);
        Ok(Json::Number(number))
    }

    fn text(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut text = Vec::new();
        while let Some(byte) = self.peek() {
            self.at += 1;
            match byte {
                b'"' => return String::from_utf8(text).map_err(|e| e.to_string()),
                b'\\' => {
                    let escaped = self.peek().ok_or("unterminated escape")?;
                    self.at += 1;
                    match escaped {
                        b'n' => text.push(b'\n'),
                        b't' => text.push(b'\t'),
                        b'r' => text.push(b'\r'),
                        b'u' => {
                            let digits =
                                self.bytes.get(self.at..self.at + 4).ok_or("short unicode escape")?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(digits).map_err(|e| e.to_string())?,
                                16,
                            )
                            .map_err(|e| e.to_string())?;
                            let character = char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER);
                            text.extend(character.to_string().as_bytes());
                            self.at += 4;
                        }
                        other => text.push(other),
                    }
                }
                other => text.push(other),
            }
        }
        Err("unterminated string".into())
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        while self.at < self.bytes.len() {
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(format!("expected , or ] at {}", self.at)),
            }
        }
        Err("unterminated array".into())
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut entries = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(entries));
        }
        while self.at < self.bytes.len() {
            self.skip_space();
            let key = self.text()?;
            self.expect(b':')?;
            entries.push((key, self.value(depth + 1)?));
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(entries));
                }
                _ => return Err(format!("expected , or }} at {}", self.at)),
            }
        }
        Err("unterminated object".into())
    }
}

impl Json {
    pub fn path(&self, path: &[&str]) -> Option<&Json> {
        let mut current = self;
        for segment in path {
            current = match current {
                Json::Object(entries) => &entries.iter().find(|(key, _)| key == segment)?.1,
                Json::Array(items) => items.get(segment.parse::<usize>().ok()?)?,
                _ => return None,
            };
        }
        Some(current)
    }

    pub fn number(&self, path: &[&str]) -> Result<f64, String> {
        match self.path(path) {
            Some(Json::Number(number)) => Ok(*number),
            other => Err(format!("{} is not a number: {other:?}", path.join("."))),
        }
    }

    pub fn len(&self, path: &[&str]) -> Result<usize, String> {
        match self.path(path) {
            Some(Json::Array(items)) => Ok(items.len()),
            other => Err(format!("{} is not an array: {other:?}", path.join("."))),
        }
    }
}

pub fn self_test() -> Result<(), String> {
    let document = parse(r#"{"a": [1, 2.5e1, {"b": "x\"y"}], "c": true, "d": null, "e": -3}"#)?;
    let checks = [
        document.number(&["a", "1"])? == 25.0,
        document.len(&["a"])? == 3,
        document.path(&["a", "2", "b"]) == Some(&Json::Text("x\"y".into())),
        document.number(&["e"])? == -3.0,
        document.path(&["c"]) == Some(&Json::Bool(true)),
        parse("{").is_err() && parse("[1,]").is_err() && parse("{} x").is_err(),
    ];
    if checks.iter().all(|&ok| ok) { Ok(()) } else { Err(format!("the JSON reader fails {checks:?}")) }
}
