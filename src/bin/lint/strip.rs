#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Code,
    Comment,
    Protected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub class: Class,
    pub start: usize,
    pub end: usize,
}

#[derive(Default)]
struct Line {
    text: String,
    protected_len: usize,
    had_comment: bool,
}

pub fn strip(source: &str, pieces: &[Piece]) -> String {
    assert_cover(source, pieces);
    let lines = split(source, pieces);
    let kept: Vec<String> = lines.into_iter().filter_map(finish).collect();
    join(&kept)
}

fn assert_cover(source: &str, pieces: &[Piece]) {
    let mut end = 0;
    for piece in pieces {
        assert_eq!(piece.start, end);
        assert!(piece.end > piece.start);
        end = piece.end;
    }
    assert_eq!(end, source.len());
}

fn split(source: &str, pieces: &[Piece]) -> Vec<Line> {
    let mut lines = Vec::new();
    let mut line = Line::default();
    for piece in pieces {
        let text = &source[piece.start..piece.end];
        match piece.class {
            Class::Comment => {
                line.had_comment = true;
                if line.text.ends_with(|c: char| !c.is_whitespace()) {
                    line.text.push(' ');
                }
            }
            Class::Protected => {
                line.text.push_str(text);
                line.protected_len = line.text.len();
            }
            Class::Code => {
                for part in text.split_inclusive('\n') {
                    match part.strip_suffix('\n') {
                        Some(body) => {
                            line.text.push_str(body);
                            lines.push(std::mem::take(&mut line));
                        }
                        None => line.text.push_str(part),
                    }
                }
            }
        }
    }
    if !line.text.is_empty() || line.had_comment {
        lines.push(line);
    }
    lines
}

fn finish(mut line: Line) -> Option<String> {
    let trimmed = line.text.trim_end_matches([' ', '\t']).len();
    line.text.truncate(trimmed.max(line.protected_len));
    if line.had_comment && line.text.trim().is_empty() {
        return None;
    }
    Some(line.text)
}

fn join(lines: &[String]) -> String {
    let mut out = String::new();
    let mut previous_blank = true;
    let mut pending_blank = false;
    for line in lines {
        let blank = line.trim().is_empty();
        if blank {
            pending_blank = !previous_blank;
            continue;
        }
        if pending_blank {
            out.push('\n');
        }
        out.push_str(line);
        out.push('\n');
        previous_blank = false;
        pending_blank = false;
    }
    out
}
