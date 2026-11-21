use crate::lexer::{self, Kind};

const TEST_ATTRIBUTES: [&str; 4] = ["test", "bench", "should_panic", "ignore"];
const TEST_CONFIGURATIONS: [&str; 3] = ["test", "doctest", "doc"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub line: usize,
    pub message: String,
}

pub struct Lines {
    starts: Vec<usize>,
}

impl Lines {
    pub fn new(source: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(source.match_indices('\n').map(|(index, _)| index + 1));
        Lines { starts }
    }

    pub fn line(&self, offset: usize) -> usize {
        let line = self.starts.partition_point(|&start| start <= offset);
        assert!(line >= 1);
        line
    }
}

type Significant<'a> = (Kind, &'a str, usize);

pub fn check_rust(source: &str) -> Vec<Finding> {
    let lines = Lines::new(source);
    let tokens = match lexer::lex(source) {
        Ok(tokens) => tokens,
        Err(error) => {
            let message = format!("cannot lex: {}", error.message);
            return vec![Finding { line: lines.line(error.offset), message }];
        }
    };
    let mut findings: Vec<Finding> = tokens
        .iter()
        .filter(|token| token.kind == Kind::Comment)
        .map(|token| Finding {
            line: lines.line(token.start),
            message: comment_kind(&source[token.start..token.end]).to_string(),
        })
        .collect();
    let significant: Vec<Significant> = tokens
        .iter()
        .filter(|token| !matches!(token.kind, Kind::Whitespace | Kind::Comment))
        .map(|token| (token.kind, &source[token.start..token.end], token.start))
        .collect();
    for index in 0..significant.len() {
        let found = attribute(&significant, index).or_else(|| configuration_macro(&significant, index));
        if let Some(message) = found {
            findings.push(Finding { line: lines.line(significant[index].2), message });
        }
    }
    findings
}

fn comment_kind(text: &str) -> &'static str {
    assert!(text.starts_with("//") || text.starts_with("/*"));
    let outer_line = text.starts_with("///") && !text.starts_with("////");
    let outer_block = text.starts_with("/**") && !text.starts_with("/***") && text != "/**/";
    if outer_line || outer_block || text.starts_with("//!") || text.starts_with("/*!") {
        "doc comment"
    } else {
        "comment"
    }
}

fn is_punct(token: Option<&Significant>, text: &str) -> bool {
    token.is_some_and(|token| token.0 == Kind::Punct && token.1 == text)
}

fn attribute(tokens: &[Significant], index: usize) -> Option<String> {
    if !is_punct(tokens.get(index), "#") {
        return None;
    }
    let mut open = index + 1;
    if is_punct(tokens.get(open), "!") {
        open += 1;
    }
    if !is_punct(tokens.get(open), "[") {
        return None;
    }
    let close = matching(tokens, open)?;
    let inner = &tokens[open + 1..close];
    let first = inner.first()?.1;
    let idents: Vec<&str> =
        inner.iter().filter(|token| token.0 == Kind::Ident).map(|token| token.1).collect();
    if TEST_ATTRIBUTES.contains(&first) {
        return Some(format!("#[{first}] attribute"));
    }
    if first == "doc" {
        return Some("doc attribute".to_string());
    }
    if idents.contains(&"missing_docs") {
        return Some("missing_docs lint attribute".to_string());
    }
    let configuration = matches!(first, "cfg" | "cfg_attr");
    if configuration && idents.iter().any(|ident| TEST_CONFIGURATIONS.contains(ident)) {
        return Some(format!("{first} attribute on a test or doc configuration"));
    }
    None
}

fn configuration_macro(tokens: &[Significant], index: usize) -> Option<String> {
    let name = tokens.get(index)?;
    if name.0 != Kind::Ident || name.1 != "cfg" || !is_punct(tokens.get(index + 1), "!") {
        return None;
    }
    if !is_punct(tokens.get(index + 2), "(") {
        return None;
    }
    let close = matching(tokens, index + 2)?;
    let inner = &tokens[index + 3..close];
    let test = inner.iter().any(|token| token.0 == Kind::Ident && TEST_CONFIGURATIONS.contains(&token.1));
    test.then(|| "cfg! on a test or doc configuration".to_string())
}

fn matching(tokens: &[Significant], open: usize) -> Option<usize> {
    assert!(open < tokens.len());
    let mut depth = 0usize;
    for (offset, token) in tokens[open..].iter().enumerate() {
        if token.0 != Kind::Punct {
            continue;
        }
        match token.1 {
            "[" | "(" | "{" => depth += 1,
            "]" | ")" | "}" => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}
