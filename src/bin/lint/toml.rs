use crate::rules::{Finding, Lines};
use crate::strip::{Class, Piece};

const DEPENDENCY_TABLES: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];
const TEST_TARGETS: [&str; 2] = ["test", "bench"];

pub fn lex(source: &str) -> Result<Vec<Piece>, String> {
    let bytes = source.as_bytes();
    let mut pieces = Vec::new();
    let mut pos = 0;
    for _ in 0..bytes.len() {
        if pos == bytes.len() {
            break;
        }
        let (class, end) = piece(bytes, pos)?;
        assert!(end > pos && end <= bytes.len());
        assert!(source.is_char_boundary(end));
        pieces.push(Piece { class, start: pos, end });
        pos = end;
    }
    assert_eq!(pos, bytes.len());
    Ok(pieces)
}

pub fn content(source: &str, pieces: &[Piece]) -> Vec<String> {
    let mut content = Vec::new();
    let mut code = String::new();
    for piece in pieces {
        let text = &source[piece.start..piece.end];
        match piece.class {
            Class::Comment => {}
            Class::Code => code.extend(text.chars().filter(|c| !c.is_whitespace())),
            Class::Protected => {
                content.push(std::mem::take(&mut code));
                content.push(text.to_string());
            }
        }
    }
    content.push(code);
    content
}

fn piece(bytes: &[u8], pos: usize) -> Result<(Class, usize), String> {
    let rest = &bytes[pos..];
    assert!(!rest.is_empty());
    if rest[0] == b'#' {
        let length = rest.iter().position(|&byte| byte == b'\n').unwrap_or(rest.len());
        return Ok((Class::Comment, pos + length));
    }
    if rest.starts_with(b"\"\"\"") || rest.starts_with(b"'''") {
        return multiline(bytes, pos, rest[0]);
    }
    if rest[0] == b'"' || rest[0] == b'\'' {
        return single(bytes, pos, rest[0]);
    }
    let length = rest.iter().take_while(|&&byte| !matches!(byte, b'#' | b'"' | b'\'')).count();
    Ok((Class::Code, pos + length))
}

fn single(bytes: &[u8], start: usize, quote: u8) -> Result<(Class, usize), String> {
    let mut pos = start + 1;
    for _ in 0..bytes.len() {
        match bytes.get(pos) {
            Some(b'\\') if quote == b'"' => pos += 2,
            Some(&byte) if byte == quote => return Ok((Class::Protected, pos + 1)),
            Some(b'\n') | None => break,
            Some(_) => pos += 1,
        }
    }
    Err(format!("unterminated string at byte {start}"))
}

fn multiline(bytes: &[u8], start: usize, quote: u8) -> Result<(Class, usize), String> {
    let delimiter = [quote; 3];
    let mut pos = start + 3;
    for _ in 0..bytes.len() {
        if pos >= bytes.len() {
            break;
        }
        if quote == b'"' && bytes[pos] == b'\\' {
            pos += 2;
            continue;
        }
        if bytes[pos..].starts_with(&delimiter) {
            let extra = bytes[pos + 3..].iter().take(2).take_while(|&&byte| byte == quote).count();
            return Ok((Class::Protected, pos + 3 + extra));
        }
        pos += 1;
    }
    Err(format!("unterminated multi-line string at byte {start}"))
}

pub fn check(source: &str, manifest: bool) -> Vec<Finding> {
    let lines = Lines::new(source);
    let pieces = match lex(source) {
        Ok(pieces) => pieces,
        Err(message) => return vec![Finding { line: 1, message: format!("cannot lex: {message}") }],
    };
    let mut findings: Vec<Finding> = pieces
        .iter()
        .filter(|piece| piece.class == Class::Comment)
        .map(|piece| Finding { line: lines.line(piece.start), message: "comment".to_string() })
        .collect();
    if manifest {
        findings.extend(manifest_findings(&view(source, &pieces)));
    }
    findings
}

fn view(source: &str, pieces: &[Piece]) -> String {
    let mut view = String::with_capacity(source.len());
    for piece in pieces {
        let text = &source[piece.start..piece.end];
        match piece.class {
            Class::Code => view.push_str(text),
            Class::Comment => {}
            Class::Protected => {
                view.push('S');
                view.extend(text.chars().filter(|&c| c == '\n'));
            }
        }
    }
    assert_eq!(view.matches('\n').count(), source.matches('\n').count());
    view
}

fn manifest_findings(view: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut depth = 0i64;
    let mut table: Option<String> = None;
    for (index, line) in view.lines().enumerate() {
        let trimmed = line.trim();
        let mut found = None;
        if depth == 0 && trimmed.starts_with('[') {
            (table, found) = header(trimmed);
        } else if depth == 0 && !trimmed.is_empty() {
            found = key(trimmed, table.as_deref());
            depth += nesting(trimmed);
        } else {
            depth += nesting(trimmed);
        }
        if let Some(message) = found {
            findings.push(Finding { line: index + 1, message });
        }
    }
    findings
}

fn nesting(line: &str) -> i64 {
    let opened = line.chars().filter(|&c| c == '[' || c == '{').count();
    let closed = line.chars().filter(|&c| c == ']' || c == '}').count();
    i64::try_from(opened).unwrap_or(i64::MAX) - i64::try_from(closed).unwrap_or(i64::MAX)
}

fn header(line: &str) -> (Option<String>, Option<String>) {
    assert!(line.starts_with('['));
    let array = line.starts_with("[[");
    let name = line.trim_start_matches('[').split(']').next().unwrap_or("").trim();
    let segments: Vec<&str> = name.split('.').map(str::trim).collect();
    let last = segments.len() - 1;
    if array && last == 0 && TEST_TARGETS.contains(&segments[0]) {
        return (None, Some(format!("[[{name}]] target")));
    }
    if segments[..last].iter().any(|segment| DEPENDENCY_TABLES.contains(segment)) {
        return (None, Some(format!("dependency in [{name}]")));
    }
    if DEPENDENCY_TABLES.contains(&segments[last]) {
        return (Some(name.to_string()), None);
    }
    (None, None)
}

fn key(line: &str, table: Option<&str>) -> Option<String> {
    let key = line.split('=').next()?.trim();
    let first = key.split('.').next()?.trim();
    match table {
        Some(name) => Some(format!("dependency in [{name}]")),
        None if DEPENDENCY_TABLES.contains(&first) => Some(format!("dependency in {first}")),
        None => None,
    }
}
