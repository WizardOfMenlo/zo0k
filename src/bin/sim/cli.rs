use crate::json::{Json, parse};
use crate::report::ensure;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::SystemTime;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn binary() -> Result<PathBuf, String> {
    let current = std::env::current_exe().map_err(|e| format!("no current executable: {e}"))?;
    let directory = current.parent().ok_or("the executable has no directory")?;
    let binary = directory.join(format!("zo0k{}", std::env::consts::EXE_SUFFIX));
    ensure(binary.is_file(), || {
        format!("{} is missing: build it with cargo build --release", binary.display())
    })?;
    Ok(binary)
}

fn newest_source(directory: &Path, depth: usize) -> SystemTime {
    let mut newest = SystemTime::UNIX_EPOCH;
    let Ok(entries) = fs::read_dir(directory) else {
        return newest;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let skipped = path.ends_with("sim");
        if path.is_dir() && depth > 0 && !skipped {
            newest = newest.max(newest_source(&path, depth - 1));
        } else if path.extension().is_some_and(|e| e == "rs") {
            newest = newest.max(entry.metadata().and_then(|m| m.modified()).unwrap_or(newest));
        }
    }
    newest
}

pub fn binary_is_fresh() -> Result<(), String> {
    let binary = binary()?;
    let built = fs::metadata(&binary).and_then(|m| m.modified()).map_err(|e| e.to_string())?;
    let newest = newest_source(&Path::new(ROOT).join("src"), 4);
    ensure(newest <= built, || {
        format!("{} is older than the sources: rebuild it with cargo build --release", binary.display())
    })
}

fn zo0k(arguments: &[&str]) -> Result<Output, String> {
    let binary = binary()?;
    let output = Command::new(&binary)
        .args(arguments)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", binary.display()))?;
    Ok(output)
}

fn succeeds(arguments: &[&str]) -> Result<Output, String> {
    let output = zo0k(arguments)?;
    let stderr = String::from_utf8_lossy(&output.stderr).replace('\n', " ");
    ensure(output.status.success(), || format!("zo0k {} fails: {stderr}", arguments.join(" ")))?;
    Ok(output)
}

fn fails(arguments: &[&str]) -> Result<Output, String> {
    let output = zo0k(arguments)?;
    ensure(!output.status.success(), || format!("zo0k {} succeeds", arguments.join(" ")))?;
    Ok(output)
}

fn document(output: &Output) -> Result<Json, String> {
    let text = std::str::from_utf8(&output.stdout).map_err(|e| e.to_string())?;
    parse(text).map_err(|e| format!("the output is not JSON ({e}): {text}"))
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let directory = std::env::temp_dir().join(format!("zo0k-sim-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        assert!(!directory.exists());
        Self(directory)
    }

    fn text(&self) -> &str {
        self.0.to_str().expect("a temporary path is UTF-8")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn bench_and_verify() -> Result<(), String> {
    let scratch = Scratch::new("bench");
    let out = scratch.text();
    let bench =
        succeeds(&["bench", "--num-variables", "8", "--security", "30", "--repeats", "2", "--out", out])?;
    let report = document(&bench)?;
    ensure(report.number(&["soundness_bits"])? >= 30.0, || "bench reports fewer than 30 bits".into())?;
    ensure(report.number(&["argument_bytes"])? > 0.0, || "bench reports an empty argument".into())?;
    let verify = succeeds(&["verify", out])?;
    let printed = String::from_utf8_lossy(&verify.stdout);
    ensure(printed.contains("soundness error of 2^-3"), || format!("verify prints {printed:?}"))?;
    fails(&["bench", "--num-variables", "8", "--security", "30", "--out", out])?;
    let statement_file = scratch.0.join("statement.json");
    let statement = fs::read_to_string(&statement_file).map_err(|e| e.to_string())?;
    let commitment_at = statement.find("\"commitment\": \"").ok_or("no commitment in the statement")? + 15;
    let mut signed = statement.clone();
    signed.replace_range(commitment_at..commitment_at + 1, "+");
    let unknown_field = statement.replacen("\"gf2_256\"", "\"gf2_999\"", 1);
    for altered in [signed, statement.replacen('{', "{\"extra\": 1,", 1), unknown_field, String::new()] {
        fs::write(&statement_file, &altered).map_err(|e| e.to_string())?;
        fails(&["verify", out])?;
    }
    fs::write(&statement_file, &statement).map_err(|e| e.to_string())?;
    succeeds(&["verify", out])?;
    let proof_file = scratch.0.join("proof.bin");
    let proof = fs::read(&proof_file).map_err(|e| e.to_string())?;
    ensure(proof.len() > 100, || "the saved proof is short".into())?;
    let mut flipped = proof.clone();
    flipped[100] ^= 1;
    let truncated = proof[..proof.len() - 1].to_vec();
    for altered in [flipped, truncated] {
        fs::write(&proof_file, altered).map_err(|e| e.to_string())?;
        fails(&["verify", out])?;
    }
    Ok(())
}

pub fn bench_with_keccak() -> Result<(), String> {
    let scratch = Scratch::new("keccak");
    let out = scratch.text();
    let arguments = [
        "bench",
        "--num-variables",
        "6",
        "--security",
        "24",
        "--repeats",
        "1",
        "--field",
        "gf2_128",
        "--hash",
        "keccak256",
        "--out",
        out,
    ];
    let report = document(&succeeds(&arguments)?)?;
    ensure(report.number(&["soundness_bits"])? >= 24.0, || "bench over gf2_128 misses its target".into())?;
    let verify = succeeds(&["verify", out])?;
    let printed = String::from_utf8_lossy(&verify.stdout);
    ensure(printed.contains("keccak256") && printed.contains("gf2_128"), || {
        format!("verify prints {printed:?}")
    })?;
    fails(&["verify", &scratch.0.join("missing").to_string_lossy()])?;
    Ok(())
}

pub fn search_is_admissible() -> Result<(), String> {
    let scratch = Scratch::new("search");
    fs::create_dir_all(&scratch.0).map_err(|e| e.to_string())?;
    let file = scratch.0.join("parameters.json");
    let file_text = file.to_str().ok_or("path")?.to_string();
    for (stages, variant) in [("1", None), ("2", None), ("2", Some("--baseline"))] {
        let mut arguments = vec![
            "search",
            "--num-variables",
            "12",
            "--security",
            "40",
            "--pow-bits",
            "8",
            "--stages",
            stages,
        ];
        arguments.extend(variant);
        let search = succeeds(&arguments)?;
        fs::write(&file, &search.stdout).map_err(|e| e.to_string())?;
        let mut arguments = vec!["analyze", "--params", &file_text];
        arguments.extend(variant);
        let analysis = document(&succeeds(&arguments)?)?;
        let count: usize = stages.parse().map_err(|_| "stage count")?;
        ensure(analysis.len(&["layout", "stages"])? == count, || {
            format!("the analysis has another stage count than {count}")
        })?;
        ensure(analysis.number(&["soundness", "total_bits"])? >= 40.0, || {
            "a searched schedule misses 40 bits".into()
        })?;
        ensure(analysis.number(&["costs", "iop_bytes"])? <= (5 * 32 * 4096) as f64, || {
            "a searched schedule exceeds its budget".into()
        })?;
        ensure(analysis.number(&["parameters", "num_variables"])? == 12.0, || {
            "parameters do not round trip".into()
        })?;
    }
    let text = fs::read_to_string(&file).map_err(|e| e.to_string())?;
    ensure(text.contains("\"pow_bits\""), || "searched parameters have no pow_bits".into())?;
    fs::write(&file, text.replacen("\"pow_bits\"", "\"pow\"", 1)).map_err(|e| e.to_string())?;
    fails(&["analyze", "--params", &file_text, "--baseline"])?;
    fs::write(&file, text.replacen('{', "{\"extra\": 1,", 1)).map_err(|e| e.to_string())?;
    fails(&["analyze", "--params", &file_text, "--baseline"])?;
    Ok(())
}

pub fn paper_rows() -> Result<(), String> {
    let analysis = document(&succeeds(&["analyze", "--paper", "polylog-26"])?)?;
    ensure(analysis.number(&["costs", "iop_bytes"])? == 10_737_129_704.0, || {
        "polylog-26 has another proof length".into()
    })?;
    fails(&["analyze", "--paper", "polylog-27"])?;
    fails(&["analyze"])?;
    fails(&["analyze", "--paper", "polylog-26", "--security", "80"])?;
    fails(&["analyze", "--num-variables", "34"])?;
    fails(&["analyze", "--num-variables", "10", "--security", "4294967295"])?;
    for num_variables in ["2", "8", "12"] {
        succeeds(&["analyze", "--num-variables", num_variables, "--baseline"])?;
    }
    let missed = fails(&["analyze", "--paper", "polylog-20", "--field", "gf2_128"])?;
    let stderr = String::from_utf8_lossy(&missed.stderr);
    ensure(stderr.contains("above the target 2^-100"), || format!("a missed target prints {stderr:?}"))?;
    let bits = document(&missed)?.number(&["soundness", "total_bits"])?;
    ensure((47.0..48.0).contains(&bits), || format!("polylog-20 over gf2_128 has {bits} bits"))?;
    fails(&["bench", "--paper", "polylog-20", "--field", "gf2_128"])?;
    Ok(())
}

pub fn small_field_search() -> Result<(), String> {
    let scratch = Scratch::new("search-gf2-128");
    fs::create_dir_all(&scratch.0).map_err(|e| e.to_string())?;
    let options = [
        "search",
        "--field",
        "gf2_128",
        "--num-variables",
        "12",
        "--stages",
        "2",
        "--pow-bits",
        "8",
        "--max-pow-bits",
    ];
    fails(&[&options[..], &["0"]].concat())?;
    let found = succeeds(&[&options[..], &["16"]].concat())?;
    let file = scratch.0.join("parameters.json");
    fs::write(&file, &found.stdout).map_err(|e| e.to_string())?;
    let parameters = document(&found)?;
    ensure(parameters.number(&["johnson_slack"])? > 0.0002, || "the search keeps the smallest slack".into())?;
    let bits = parameters.number(&["stages", "0", "sumcheck_pow_bits"])?;
    ensure((1.0..=16.0).contains(&bits), || format!("the first sumcheck grinds {bits} bits"))?;
    let file = file.to_str().ok_or("path")?;
    let analysis = document(&succeeds(&["analyze", "--params", file, "--field", "gf2_128"])?)?;
    ensure(analysis.number(&["soundness", "total_bits"])? >= 100.0, || "the search misses 100 bits".into())?;
    succeeds(&["bench", "--params", file, "--repeats", "1", "--field", "gf2_128"])?;
    succeeds(&["analyze", "--params", file])?;
    Ok(())
}

pub fn paper_reproduction() -> Result<(), String> {
    let root = Path::new(ROOT);
    let scratch = Scratch::new("paper");
    let configurations = root.join("paper");
    succeeds(&["paper", "--configurations", &configurations.to_string_lossy(), "--out", scratch.text()])?;
    let read = |directory: &Path, file: &str| {
        fs::read_to_string(directory.join(file)).map_err(|e| format!("{file}: {e}"))
    };
    let file = "scaling-tables.tex";
    ensure(read(&scratch.0, file)? == read(&configurations, file)?, || {
        format!("paper/{file} is out of date")
    })?;
    let file = "frontiers.csv";
    let (actual, expected) = (read(&scratch.0, file)?, read(&configurations, file)?);
    ensure(actual.lines().count() == expected.lines().count(), || {
        "frontiers.csv has another line count".into()
    })?;
    ensure(actual.lines().next() == expected.lines().next(), || "frontiers.csv has another header".into())?;
    for (line, (actual, expected)) in actual.lines().zip(expected.lines()).enumerate().skip(1) {
        let (actual, expected): (Vec<&str>, Vec<&str>) =
            (actual.split(',').collect(), expected.split(',').collect());
        ensure(actual.len() == 5 && expected.len() == 5, || {
            format!("frontiers.csv line {line} has another shape")
        })?;
        ensure((&actual[..3], actual[4]) == (&expected[..3], expected[4]), || {
            format!("frontiers.csv line {line}: {actual:?}")
        })?;
        let size = |fields: &[&str]| fields[3].parse::<f64>().map_err(|e| e.to_string());
        let difference = (size(&actual)? - size(&expected)?).abs();
        ensure(difference < 0.002, || format!("frontiers.csv line {line}: {actual:?} and {expected:?}"))?;
    }
    Ok(())
}
