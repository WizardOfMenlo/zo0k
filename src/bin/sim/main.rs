mod algebra;
mod analysis;
mod cli;
mod composed;
mod fields;
mod fixed;
mod fixed_protocol;
mod json;
mod primitives;
mod prng;
mod protocol;
mod report;
mod rounds;
mod zk;

use prng::mix;
use report::{Ctx, Failure, guarded};
use std::fs::OpenOptions;
use std::io::Write;
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GATE_FRESH_SEEDS: usize = 1000;
const HUNT_SECONDS_DEFAULT: u64 = 300;
const HUNT_SEEDS_MAX: u64 = 1 << 32;
const SEEDS_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/sim/seeds.txt");
const USAGE: &str = "usage: sim [--fresh N] | sim --seed S | sim --hunt [--seconds T]";

type OneOff = (&'static str, fn() -> Result<(), String>);

fn one_off_checks() -> Vec<OneOff> {
    vec![
        ("json_reader", json::self_test),
        ("field_moduli_irreducible", fields::moduli_are_irreducible),
        ("field_edge_products", fields::edge_products),
        ("transcript_malformed_proofs", primitives::malformed_transcripts),
        ("transcript_uniform_positions", primitives::uniform_positions),
        ("merkle_every_shape_and_subset", primitives::fixed_openings),
        ("oracle_wide_openings", primitives::wide_oracle),
        ("polynomial_large_tables", algebra::large_polynomials),
        ("reed_solomon_cantor_basis", algebra::cantor_basis),
        ("reed_solomon_shapes_and_blocks", algebra::encodings),
        ("zero_knowledge_rank_conditions", zk::zero_knowledge_fixed),
        ("parameters_recorded_layout", fixed::recorded_layout),
        ("parameters_invalid_rejected", fixed::invalid_parameters),
        ("soundness_recorded_code_bounds", fixed::recorded_code_bounds),
        ("soundness_recorded_values", fixed::recorded_soundness),
        ("soundness_targets_and_grinding", fixed::soundness_targets),
        ("costs_expected_openings", fixed::expected_openings),
        ("tuning_queries", fixed::queries_are_tuned),
        ("tuning_standard_parameters", fixed::standard_parameters),
        ("tuning_grinding_assignment", fixed::grinding_is_assigned),
        ("protocol_schedules_gf2_256_blake3", fixed_protocol::schedules_gf2_256_blake3),
        ("protocol_schedules_gf2_128_blake3", fixed_protocol::schedules_gf2_128_blake3),
        ("protocol_schedules_gf2_256_keccak256", fixed_protocol::schedules_gf2_256_keccak256),
        ("protocol_schedules_gf2_128_keccak256", fixed_protocol::schedules_gf2_128_keccak256),
        ("protocol_fields_and_hashes_bind", fixed_protocol::fields_and_hashes_bind),
        ("protocol_proofs_independent_of_threads", fixed_protocol::threads_agree),
        ("cli_binary_is_fresh", cli::binary_is_fresh),
        ("cli_bench_and_verify", cli::bench_and_verify),
        ("cli_bench_with_keccak", cli::bench_with_keccak),
        ("cli_search_is_admissible", cli::search_is_admissible),
        ("cli_paper_rows", cli::paper_rows),
        ("cli_small_field_search", cli::small_field_search),
        ("cli_paper_reproduction", cli::paper_reproduction),
    ]
}

enum Mode {
    Fingerprint,
    Gate { fresh: usize },
    Seed(u64),
    Hunt { seconds: u64 },
}

fn parse_seed(text: &str) -> Result<u64, String> {
    let parsed = match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => text.parse(),
    };
    parsed.map_err(|_| format!("not a seed: {text:?}"))
}

fn parse_mode(arguments: &[String]) -> Result<Mode, String> {
    let words: Vec<&str> = arguments.iter().map(String::as_str).collect();
    let number = |text: &str| text.parse::<u64>().map_err(|_| format!("not a number: {text:?}"));
    match words.as_slice() {
        [] => Ok(Mode::Gate { fresh: GATE_FRESH_SEEDS }),
        ["--fresh", count] => Ok(Mode::Gate { fresh: number(count)? as usize }),
        ["--proof-fingerprint"] => Ok(Mode::Fingerprint),
        ["--seed", seed] => Ok(Mode::Seed(parse_seed(seed)?)),
        ["--hunt"] => Ok(Mode::Hunt { seconds: HUNT_SECONDS_DEFAULT }),
        ["--hunt", "--seconds", seconds] => Ok(Mode::Hunt { seconds: number(seconds)? }),
        _ => Err(format!("unknown arguments {words:?}")),
    }
}

fn entropy_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    hasher.write_u128(nanos);
    hasher.write_u32(std::process::id());
    let seed = hasher.finish();
    assert!(nanos > 0 || seed != 0);
    seed
}

pub fn run_scenario(seed: u64, verbose: bool) -> Result<usize, Failure> {
    type Module = fn(&mut Ctx) -> Result<(), Failure>;
    let modules: [(u64, Module); 7] = [
        (1, fields::scenario),
        (2, primitives::scenario),
        (3, algebra::scenario),
        (4, rounds::scenario),
        (5, analysis::scenario),
        (6, zk::scenario),
        (7, protocol::scenario),
    ];
    let mut checks = 0;
    for (stream, module) in modules {
        let mut ctx = Ctx::new(mix(seed, stream), verbose);
        let outcome = module(&mut ctx);
        checks += ctx.checks_run;
        outcome?;
    }
    assert!(checks > 0);
    Ok(checks)
}

fn repro(seed: u64) -> String {
    format!("cargo run --release --bin sim -- --seed {seed:#018x}")
}

fn report_seed_failure(seed: u64, failure: &Failure) {
    assert!(!failure.check.is_empty());
    println!("FAIL seed={seed:#018x} check={}: {} | repro: {}", failure.check, failure.detail, repro(seed));
}

fn load_corpus() -> Result<Vec<u64>, String> {
    let text = std::fs::read_to_string(SEEDS_FILE).map_err(|e| format!("{SEEDS_FILE}: {e}"))?;
    let mut seeds = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty() && !line.starts_with('#')) {
        let token = line.split_whitespace().next().expect("a non-empty line has a token");
        seeds.push(parse_seed(token)?);
    }
    assert!(seeds.len() <= text.lines().count());
    Ok(seeds)
}

fn append_corpus(seed: u64, check: &str) -> Result<(), String> {
    let known = load_corpus()?;
    if known.contains(&seed) {
        return Ok(());
    }
    let mut file = OpenOptions::new().append(true).open(SEEDS_FILE).map_err(|e| e.to_string())?;
    writeln!(file, "{seed:#018x} {check}").map_err(|e| e.to_string())?;
    assert!(!check.is_empty());
    Ok(())
}

fn run_one_offs() -> (usize, usize) {
    let checks = one_off_checks();
    let mut failures = 0;
    for (name, check) in &checks {
        if let Err(detail) = guarded(check) {
            println!("FAIL one-off={name}: {detail} | repro: cargo build --release && target/release/sim");
            failures += 1;
        }
    }
    assert!(failures <= checks.len());
    (checks.len(), failures)
}

fn run_seeds(seeds: impl Iterator<Item = u64>) -> (usize, usize) {
    let (mut count, mut failures) = (0, 0);
    for seed in seeds {
        count += 1;
        if let Err(failure) = run_scenario(seed, false) {
            report_seed_failure(seed, &failure);
            failures += 1;
        }
    }
    assert!(failures <= count);
    (count, failures)
}

fn arithmetic() -> &'static str {
    if cfg!(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    )) {
        "carry-less"
    } else {
        "portable"
    }
}

fn run_gate(fresh: usize) -> usize {
    let start = Instant::now();
    let (one_offs, one_off_failures) = run_one_offs();
    let corpus = load_corpus().unwrap_or_else(|error| {
        println!("FAIL corpus: {error}");
        Vec::new()
    });
    let (corpus_count, corpus_failures) = run_seeds(corpus.into_iter());
    let base = entropy_seed();
    let (fresh_count, fresh_failures) = run_seeds((0..fresh as u64).map(|i| base.wrapping_add(i)));
    let failures = one_off_failures + corpus_failures + fresh_failures;
    let verdict = if failures == 0 { "PASS" } else { "FAIL" };
    println!(
        "sim {verdict}: one-offs {}/{one_offs}, corpus {}/{corpus_count}, fresh {}/{fresh_count} (seeds {base:#018x}..+{fresh}), {:.1} s, {} arithmetic",
        one_offs - one_off_failures,
        corpus_count - corpus_failures,
        fresh_count - fresh_failures,
        start.elapsed().as_secs_f64(),
        arithmetic(),
    );
    failures
}

fn run_single(seed: u64) -> usize {
    println!("sim seed {seed:#018x}");
    let start = Instant::now();
    match run_scenario(seed, true) {
        Ok(checks) => {
            println!("sim PASS: seed {seed:#018x}, {checks} checks, {:.1} s", start.elapsed().as_secs_f64());
            0
        }
        Err(failure) => {
            report_seed_failure(seed, &failure);
            1
        }
    }
}

fn run_hunt(seconds: u64) -> usize {
    let start = Instant::now();
    let budget = Duration::from_secs(seconds);
    let base = entropy_seed();
    println!("sim hunt: seeds from {base:#018x} for {seconds} s");
    for i in 0..HUNT_SEEDS_MAX {
        if start.elapsed() >= budget {
            println!(
                "sim PASS: hunt of {i} seeds {base:#018x}..+{i}, {:.1} s",
                start.elapsed().as_secs_f64()
            );
            return 0;
        }
        let seed = base.wrapping_add(i);
        if let Err(failure) = run_scenario(seed, false) {
            report_seed_failure(seed, &failure);
            if let Err(error) = append_corpus(seed, &failure.check) {
                println!("FAIL corpus: cannot record {seed:#018x}: {error}");
            }
            return 1;
        }
    }
    0
}

fn run_fingerprint() -> usize {
    match fixed_protocol::reference_proof() {
        Ok(bytes) => {
            println!("{}", fixed_protocol::fingerprint(&bytes));
            0
        }
        Err(error) => {
            println!("FAIL fingerprint: {error}");
            1
        }
    }
}

fn main() -> ExitCode {
    report::install_panic_hook();
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mode = match parse_mode(&arguments) {
        Ok(mode) => mode,
        Err(error) => {
            eprintln!("error: {error}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let failures = match mode {
        Mode::Fingerprint => run_fingerprint(),
        Mode::Gate { fresh } => run_gate(fresh),
        Mode::Seed(seed) => run_single(seed),
        Mode::Hunt { seconds } => run_hunt(seconds),
    };
    if failures == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
