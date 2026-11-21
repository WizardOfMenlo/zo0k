#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Whitespace,
    Comment,
    Literal,
    Lifetime,
    Ident,
    Number,
    Punct,
    Shebang,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Whitespace => "whitespace",
            Kind::Comment => "comment",
            Kind::Literal => "literal",
            Kind::Lifetime => "lifetime",
            Kind::Ident => "ident",
            Kind::Number => "number",
            Kind::Punct => "punct",
            Kind::Shebang => "shebang",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: Kind,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LexError {
    pub offset: usize,
    pub message: &'static str,
}

type Step = Result<(Kind, usize), LexError>;

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut pos = 0;
    if let Some(end) = shebang(bytes) {
        tokens.push(Token { kind: Kind::Shebang, start: 0, end });
        pos = end;
    }
    for _ in 0..bytes.len() {
        if pos == bytes.len() {
            break;
        }
        let (kind, end) = token(bytes, pos)?;
        assert!(end > pos && end <= bytes.len());
        assert!(source.is_char_boundary(end));
        tokens.push(Token { kind, start: pos, end });
        pos = end;
    }
    assert_eq!(pos, bytes.len());
    Ok(tokens)
}

pub fn significant<'a>(source: &'a str, tokens: &[Token]) -> Vec<(Kind, &'a str)> {
    tokens
        .iter()
        .filter(|token| !matches!(token.kind, Kind::Whitespace | Kind::Comment))
        .map(|token| (token.kind, &source[token.start..token.end]))
        .collect()
}

fn shebang(bytes: &[u8]) -> Option<usize> {
    if !bytes.starts_with(b"#!") {
        return None;
    }
    let mut pos = 2;
    for _ in 0..bytes.len() {
        if pos >= bytes.len() {
            break;
        }
        match token(bytes, pos) {
            Ok((Kind::Whitespace | Kind::Comment, end)) => pos = end,
            Ok(_) | Err(_) => break,
        }
    }
    if bytes.get(pos) == Some(&b'[') {
        return None;
    }
    Some(bytes.iter().position(|&byte| byte == b'\n').unwrap_or(bytes.len()))
}

fn token(bytes: &[u8], pos: usize) -> Step {
    let byte = bytes[pos];
    let next = bytes.get(pos + 1).copied();
    match byte {
        b'/' if next == Some(b'/') => Ok((Kind::Comment, line_end(bytes, pos))),
        b'/' if next == Some(b'*') => block_comment(bytes, pos),
        b'"' => literal(bytes, string_end(bytes, pos)?),
        b'\'' => quote(bytes, pos),
        b'0'..=b'9' => Ok((Kind::Number, number(bytes, pos))),
        _ if is_whitespace(byte) => Ok((Kind::Whitespace, run(bytes, pos, is_whitespace))),
        _ if is_ident_start(byte) => word(bytes, pos),
        _ => Ok((Kind::Punct, pos + 1)),
    }
}

fn is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_' || byte >= 0x80
}

fn is_ident_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte >= 0x80
}

fn run(bytes: &[u8], start: usize, accept: fn(u8) -> bool) -> usize {
    start + bytes[start..].iter().take_while(|&&byte| accept(byte)).count()
}

fn line_end(bytes: &[u8], start: usize) -> usize {
    start + bytes[start..].iter().position(|&byte| byte == b'\n').unwrap_or(bytes.len() - start)
}

fn utf8_len(byte: u8) -> usize {
    match byte {
        0xf0..=0xff => 4,
        0xe0..=0xef => 3,
        0xc0..=0xdf => 2,
        _ => 1,
    }
}

fn block_comment(bytes: &[u8], start: usize) -> Step {
    assert!(bytes[start..].starts_with(b"/*"));
    let mut depth = 0usize;
    let mut pos = start;
    for _ in 0..bytes.len() {
        match (bytes.get(pos), bytes.get(pos + 1)) {
            (Some(b'/'), Some(b'*')) => {
                depth += 1;
                pos += 2;
            }
            (Some(b'*'), Some(b'/')) => {
                assert!(depth > 0);
                depth -= 1;
                pos += 2;
                if depth == 0 {
                    return Ok((Kind::Comment, pos));
                }
            }
            (Some(_), _) => pos += 1,
            (None, _) => break,
        }
    }
    Err(LexError { offset: start, message: "unterminated block comment" })
}

fn string_end(bytes: &[u8], open: usize) -> Result<usize, LexError> {
    assert_eq!(bytes[open], b'"');
    let mut pos = open + 1;
    for _ in 0..bytes.len() {
        match bytes.get(pos) {
            Some(b'\\') => pos += 2,
            Some(b'"') => return Ok(pos + 1),
            Some(_) => pos += 1,
            None => break,
        }
    }
    Err(LexError { offset: open, message: "unterminated string literal" })
}

fn raw_string_end(bytes: &[u8], start: usize) -> Option<Result<usize, LexError>> {
    let hashes = run(bytes, start, |byte| byte == b'#') - start;
    let open = start + hashes;
    if bytes.get(open) != Some(&b'"') {
        return None;
    }
    for pos in open + 1..bytes.len() {
        let closing = bytes.get(pos + 1..pos + 1 + hashes);
        if bytes[pos] == b'"' && closing.is_some_and(|tail| tail.iter().all(|&byte| byte == b'#')) {
            return Some(Ok(pos + 1 + hashes));
        }
    }
    Some(Err(LexError { offset: start, message: "unterminated raw string literal" }))
}

fn literal(bytes: &[u8], end: usize) -> Step {
    assert!(end <= bytes.len());
    match bytes.get(end) {
        Some(&byte) if is_ident_start(byte) => Ok((Kind::Literal, run(bytes, end, is_ident_continue))),
        _ => Ok((Kind::Literal, end)),
    }
}

fn word(bytes: &[u8], start: usize) -> Step {
    let end = run(bytes, start, is_ident_continue);
    assert!(end > start);
    match (&bytes[start..end], bytes.get(end)) {
        (b"b" | b"c", Some(b'"')) => literal(bytes, string_end(bytes, end)?),
        (b"r" | b"br" | b"cr", Some(b'"' | b'#')) => raw_or_ident(bytes, start, end),
        (b"b", Some(b'\'')) => char_literal(bytes, end),
        _ => Ok((Kind::Ident, end)),
    }
}

fn raw_or_ident(bytes: &[u8], start: usize, end: usize) -> Step {
    if let Some(result) = raw_string_end(bytes, end) {
        return literal(bytes, result?);
    }
    Ok((Kind::Ident, raw_ident_end(bytes, start, end)))
}

fn raw_ident_end(bytes: &[u8], start: usize, end: usize) -> usize {
    let raw = &bytes[start..end] == b"r" && bytes.get(end) == Some(&b'#');
    if raw && bytes.get(end + 1).is_some_and(|&byte| is_ident_start(byte)) {
        return run(bytes, end + 1, is_ident_continue);
    }
    end
}

fn quote(bytes: &[u8], start: usize) -> Step {
    assert_eq!(bytes[start], b'\'');
    match bytes.get(start + 1) {
        Some(b'\\') => char_literal(bytes, start),
        Some(&byte) if bytes.get(start + 1 + utf8_len(byte)) == Some(&b'\'') => char_literal(bytes, start),
        Some(&byte) if is_ident_start(byte) => {
            let end = run(bytes, start + 1, is_ident_continue);
            Ok((Kind::Lifetime, raw_ident_end(bytes, start + 1, end)))
        }
        _ => Ok((Kind::Punct, start + 1)),
    }
}

fn char_literal(bytes: &[u8], start: usize) -> Step {
    assert_eq!(bytes[start], b'\'');
    let mut pos = start + 1;
    for _ in 0..bytes.len() {
        match bytes.get(pos) {
            Some(b'\\') => pos += 2,
            Some(b'\'') if pos > start + 1 => return literal(bytes, pos + 1),
            Some(b'\n') | None => break,
            Some(_) => pos += 1,
        }
    }
    Err(LexError { offset: start, message: "unterminated character literal" })
}

fn number(bytes: &[u8], start: usize) -> usize {
    let mut end = run(bytes, start, is_ident_continue);
    let radix = [b"0x", b"0o", b"0b"].iter().any(|prefix| bytes[start..end].starts_with(*prefix));
    if radix {
        return end;
    }
    if bytes.get(end) == Some(&b'.') && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
        end = run(bytes, end + 1, is_ident_continue);
    }
    let exponent = matches!(bytes[end - 1], b'e' | b'E') && matches!(bytes.get(end), Some(b'+' | b'-'));
    if exponent && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
        end = run(bytes, end + 1, is_ident_continue);
    }
    end
}
