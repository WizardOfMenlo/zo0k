//! The command-line interface, end to end on small instances.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn zo0k(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_zo0k")).args(arguments).output().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("zo0k-cli-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&directory);
    directory
}

#[test]
fn a_saved_proof_verifies_and_an_altered_one_does_not() {
    let directory = scratch("bench");
    let out = directory.to_str().unwrap();
    let bench = zo0k(&["bench", "--num-variables", "8", "--security", "30", "--repeats", "2", "--out", out]);
    assert!(bench.status.success(), "{}", String::from_utf8_lossy(&bench.stderr));
    let report: zo0k::json::Value = zo0k::json::parse(&bench.stdout).unwrap();
    assert!(report["soundness_bits"].as_f64().unwrap() >= 30.0);
    assert!(report["argument_bytes"].as_f64().unwrap() > 0.0);
    let verify = zo0k(&["verify", out]);
    assert!(verify.status.success());
    assert!(String::from_utf8_lossy(&verify.stdout).contains("soundness error of 2^-3"));

    // An existing directory is not overwritten.
    assert!(!zo0k(&["bench", "--num-variables", "8", "--security", "30", "--out", out]).status.success());

    // A statement is parsed strictly.
    let statement_file = directory.join("statement.json");
    let statement = fs::read_to_string(&statement_file).unwrap();
    let commitment_at = statement.find("\"commitment\": \"").unwrap() + 15;
    let mut signed = statement.clone();
    signed.replace_range(commitment_at..commitment_at + 1, "+");
    for altered in [signed, statement.replacen('{', "{\"extra\": 1,", 1)] {
        fs::write(&statement_file, altered).unwrap();
        assert!(!zo0k(&["verify", out]).status.success());
    }
    fs::write(&statement_file, statement).unwrap();
    assert!(zo0k(&["verify", out]).status.success());

    let proof = directory.join("proof.bin");
    let mut bytes = fs::read(&proof).unwrap();
    bytes[100] ^= 1;
    fs::write(&proof, bytes).unwrap();
    assert!(!zo0k(&["verify", out]).status.success());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn a_searched_schedule_is_admissible_and_within_its_budget() {
    let directory = scratch("search");
    fs::create_dir_all(&directory).unwrap();
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
        let search = zo0k(&arguments);
        assert!(search.status.success(), "{}", String::from_utf8_lossy(&search.stderr));
        let file = directory.join("parameters.json");
        fs::write(&file, &search.stdout).unwrap();

        let mut arguments = vec!["analyze", "--params", file.to_str().unwrap()];
        arguments.extend(variant);
        let analyze = zo0k(&arguments);
        assert!(analyze.status.success(), "{}", String::from_utf8_lossy(&analyze.stderr));
        let analysis: zo0k::json::Value = zo0k::json::parse(&analyze.stdout).unwrap();
        assert_eq!(analysis["layout"]["stages"].as_array().unwrap().len(), stages.parse::<usize>().unwrap());
        assert!(analysis["soundness"]["total_bits"].as_f64().unwrap() >= 40.0);
        // Five times a table of 2^12 elements of 32 bytes.
        assert!(analysis["costs"]["iop_bytes"].as_u64().unwrap() <= 5 * 32 * 4096);
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn rows_of_the_paper_are_available_by_name() {
    let analyze = zo0k(&["analyze", "--paper", "polylog-26"]);
    assert!(analyze.status.success());
    let analysis: zo0k::json::Value = zo0k::json::parse(&analyze.stdout).unwrap();
    assert_eq!(analysis["costs"]["iop_bytes"].as_u64().unwrap(), 10_737_129_704);
    assert!(!zo0k(&["analyze", "--paper", "polylog-27"]).status.success());
    assert!(!zo0k(&["analyze"]).status.success());
    // The security option belongs to the standard parameters.
    assert!(!zo0k(&["analyze", "--paper", "polylog-26", "--security", "80"]).status.success());
    assert!(!zo0k(&["analyze", "--num-variables", "34"]).status.success());
    assert!(!zo0k(&["analyze", "--num-variables", "10", "--security", "4294967295"]).status.success());
    // Standard parameters serve both variants.
    for num_variables in ["2", "8", "12"] {
        assert!(zo0k(&["analyze", "--num-variables", num_variables, "--baseline"]).status.success());
    }
}

/// Over the smaller field the search reaches 100 bits with proofs of work
/// before the sumcheck challenges and a larger Johnson slack.
#[test]
fn a_search_over_the_smaller_field_uses_proofs_of_work() {
    let directory = scratch("search-gf2-128");
    fs::create_dir_all(&directory).unwrap();
    let field = ["--field", "gf2_128"];
    let options = ["--num-variables", "12", "--stages", "2", "--pow-bits", "8"];
    let search =
        |limit: &str| zo0k(&[&["search"][..], &field, &options, &["--max-pow-bits", limit]].concat());
    // Without such proofs of work no schedule reaches the target.
    assert!(!search("0").status.success());
    let found = search("16");
    assert!(found.status.success(), "{}", String::from_utf8_lossy(&found.stderr));
    let file = directory.join("parameters.json");
    fs::write(&file, &found.stdout).unwrap();
    let parameters: zo0k::json::Value = zo0k::json::parse(&found.stdout).unwrap();
    assert!(parameters["johnson_slack"].as_f64().unwrap() > 0.0002);
    let bits = parameters["stages"][0]["sumcheck_pow_bits"].as_u64().unwrap();
    assert!((1..=16).contains(&bits));

    let file = file.to_str().unwrap();
    let analyze = zo0k(&[&["analyze", "--params", file][..], &field].concat());
    assert!(analyze.status.success(), "{}", String::from_utf8_lossy(&analyze.stderr));
    let analysis: zo0k::json::Value = zo0k::json::parse(&analyze.stdout).unwrap();
    assert!(analysis["soundness"]["total_bits"].as_f64().unwrap() >= 100.0);
    let bench = zo0k(&[&["bench", "--params", file, "--repeats", "1"][..], &field].concat());
    assert!(bench.status.success(), "{}", String::from_utf8_lossy(&bench.stderr));
    // The same parameters do not need the proofs of work over the larger
    // field, and are valid there too.
    assert!(zo0k(&["analyze", "--params", file]).status.success());
    fs::remove_dir_all(directory).unwrap();
}

/// The analysis of parameters that miss their target is printed, and the
/// command fails.
#[test]
fn a_smaller_field_misses_the_target_of_a_row_of_the_paper() {
    let analyze = zo0k(&["analyze", "--paper", "polylog-20", "--field", "gf2_128"]);
    assert!(!analyze.status.success());
    assert!(String::from_utf8_lossy(&analyze.stderr).contains("above the target 2^-100"));
    let analysis: zo0k::json::Value = zo0k::json::parse(&analyze.stdout).unwrap();
    let bits = analysis["soundness"]["total_bits"].as_f64().unwrap();
    assert!((47.0..48.0).contains(&bits));
    // A proof needs parameters that meet their target.
    assert!(!zo0k(&["bench", "--paper", "polylog-20", "--field", "gf2_128"]).status.success());
}
