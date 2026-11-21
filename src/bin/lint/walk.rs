use std::fs;
use std::path::{Path, PathBuf};

const SKIPPED: [&str; 3] = ["target", ".git", ".worktrees"];
const MAX_DEPTH: usize = 32;
const MAX_DIRECTORIES: usize = 100_000;

#[derive(Debug, Default)]
pub struct Files {
    pub rust: Vec<PathBuf>,
    pub toml: Vec<(PathBuf, bool)>,
    pub test_directories: Vec<PathBuf>,
}

pub fn collect(root: &Path) -> Result<Files, String> {
    let mut files = Files::default();
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    for _ in 0..MAX_DIRECTORIES {
        let Some((directory, depth)) = pending.pop() else { break };
        assert!(depth <= MAX_DEPTH);
        visit(&directory, depth, &mut files, &mut pending)?;
    }
    if !pending.is_empty() {
        return Err(format!("more than {MAX_DIRECTORIES} directories"));
    }
    files.rust.sort();
    files.toml.sort();
    files.test_directories.sort();
    Ok(files)
}

fn visit(
    directory: &Path,
    depth: usize,
    files: &mut Files,
    pending: &mut Vec<(PathBuf, usize)>,
) -> Result<(), String> {
    let error = |error: std::io::Error| format!("{}: {error}", directory.display());
    let mut entries =
        fs::read_dir(directory).map_err(error)?.collect::<Result<Vec<_>, _>>().map_err(error)?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = entry.file_type().map_err(error)?;
        let path = entry.path();
        if kind.is_dir() && !SKIPPED.contains(&name.as_str()) {
            if depth + 1 > MAX_DEPTH {
                return Err(format!("{}: deeper than {MAX_DEPTH} directories", path.display()));
            }
            if name == "tests" {
                files.test_directories.push(path.clone());
            }
            pending.push((path, depth + 1));
        } else if kind.is_file() && name.ends_with(".rs") {
            files.rust.push(path);
        } else if kind.is_file() && is_toml(directory, &name) {
            files.toml.push((path, name == "Cargo.toml"));
        }
    }
    Ok(())
}

fn is_toml(directory: &Path, name: &str) -> bool {
    let cargo_config =
        name == "config.toml" && directory.file_name().is_some_and(|parent| parent == ".cargo");
    cargo_config || matches!(name, "Cargo.toml" | "rustfmt.toml" | ".rustfmt.toml")
}
