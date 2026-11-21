mod lexer;
mod rules;
mod strip;
mod toml;
mod walk;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lexer::Kind;
use rules::Finding;
use strip::{Class, Piece};

const USAGE: &str = "usage: lint [--fix] [ROOT]\n       lint --tokens FILE\n";
const MAX_ARGUMENTS: usize = 16;

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    Check,
    Fix,
    Tokens(PathBuf),
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (mode, root) = match parse(&arguments) {
        Ok(parsed) => parsed,
        Err(message) if message.is_empty() => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprint!("lint: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = match mode {
        Mode::Check => check(&root),
        Mode::Fix => fix(&root).and_then(|()| check(&root)),
        Mode::Tokens(file) => tokens(&file),
    };
    result.unwrap_or_else(|message| {
        eprintln!("lint: {message}");
        ExitCode::from(2)
    })
}

fn parse(arguments: &[String]) -> Result<(Mode, PathBuf), String> {
    if arguments.len() > MAX_ARGUMENTS {
        return Err("too many arguments".to_string());
    }
    let mut mode = Mode::Check;
    let mut root = None;
    let mut index = 0;
    for _ in 0..arguments.len() {
        let Some(argument) = arguments.get(index) else { break };
        index += 1;
        match argument.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--fix" if mode == Mode::Check => mode = Mode::Fix,
            "--tokens" if mode == Mode::Check => {
                let file = arguments.get(index).ok_or("--tokens needs a file")?;
                index += 1;
                mode = Mode::Tokens(PathBuf::from(file));
            }
            other if other.starts_with('-') => return Err(format!("unexpected option {other}")),
            other if root.is_none() => root = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if matches!(mode, Mode::Tokens(_)) && root.is_some() {
        return Err("--tokens takes no root".to_string());
    }
    Ok((mode, root.unwrap_or_else(|| PathBuf::from("."))))
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn shown(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).display().to_string()
}

fn check(root: &Path) -> Result<ExitCode, String> {
    let files = walk::collect(root)?;
    let mut findings: Vec<(String, Finding)> = Vec::new();
    for path in &files.rust {
        let found = rules::check_rust(&read(path)?);
        findings.extend(found.into_iter().map(|finding| (shown(root, path), finding)));
    }
    for (path, manifest) in &files.toml {
        let found = toml::check(&read(path)?, *manifest);
        findings.extend(found.into_iter().map(|finding| (shown(root, path), finding)));
    }
    for path in &files.test_directories {
        let finding = Finding { line: 0, message: "test directory".to_string() };
        findings.push((shown(root, path), finding));
    }
    findings.sort_by(|a, b| (&a.0, a.1.line).cmp(&(&b.0, b.1.line)));
    for (path, finding) in &findings {
        if finding.line == 0 {
            println!("{path}: {}", finding.message);
        } else {
            println!("{path}:{}: {}", finding.line, finding.message);
        }
    }
    if findings.is_empty() {
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!("lint: {} findings", findings.len());
    Ok(ExitCode::FAILURE)
}

fn fix(root: &Path) -> Result<(), String> {
    let files = walk::collect(root)?;
    for path in &files.rust {
        let source = read(path)?;
        let fixed = fix_rust(&source).map_err(|message| format!("{}: {message}", path.display()))?;
        write_if_changed(root, path, &source, &fixed)?;
    }
    for (path, _) in &files.toml {
        let source = read(path)?;
        let fixed = fix_toml(&source).map_err(|message| format!("{}: {message}", path.display()))?;
        write_if_changed(root, path, &source, &fixed)?;
    }
    Ok(())
}

fn write_if_changed(root: &Path, path: &Path, source: &str, fixed: &str) -> Result<(), String> {
    if source == fixed {
        return Ok(());
    }
    fs::write(path, fixed).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("fixed {}", shown(root, path));
    Ok(())
}

fn class(kind: Kind) -> Class {
    match kind {
        Kind::Comment => Class::Comment,
        Kind::Literal | Kind::Shebang => Class::Protected,
        _ => Class::Code,
    }
}

fn fix_rust(source: &str) -> Result<String, String> {
    let lex = |text: &str| {
        lexer::lex(text).map_err(|error| format!("cannot lex at byte {}: {}", error.offset, error.message))
    };
    let tokens = lex(source)?;
    let pieces: Vec<Piece> = tokens
        .iter()
        .map(|token| Piece { class: class(token.kind), start: token.start, end: token.end })
        .collect();
    let fixed = strip::strip(source, &pieces);
    let fixed_tokens = lex(&fixed)?;
    assert!(lexer::significant(source, &tokens) == lexer::significant(&fixed, &fixed_tokens));
    assert!(fixed_tokens.iter().all(|token| token.kind != Kind::Comment));
    let refixed: Vec<Piece> = fixed_tokens
        .iter()
        .map(|token| Piece { class: class(token.kind), start: token.start, end: token.end })
        .collect();
    assert_eq!(strip::strip(&fixed, &refixed), fixed);
    Ok(fixed)
}

fn fix_toml(source: &str) -> Result<String, String> {
    let pieces = toml::lex(source)?;
    let fixed = strip::strip(source, &pieces);
    let fixed_pieces = toml::lex(&fixed)?;
    assert!(toml::content(source, &pieces) == toml::content(&fixed, &fixed_pieces));
    assert!(fixed_pieces.iter().all(|piece| piece.class != Class::Comment));
    assert_eq!(strip::strip(&fixed, &fixed_pieces), fixed);
    Ok(fixed)
}

fn tokens(file: &Path) -> Result<ExitCode, String> {
    let source = read(file)?;
    let tokens = lexer::lex(&source).map_err(|error| {
        format!("{}: cannot lex at byte {}: {}", file.display(), error.offset, error.message)
    })?;
    let mut out = String::new();
    for (kind, text) in lexer::significant(&source, &tokens) {
        out.push_str(&format!("{}\t{text:?}\n", kind.name()));
    }
    print!("{out}");
    Ok(ExitCode::SUCCESS)
}
