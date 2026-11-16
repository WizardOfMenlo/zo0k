//! Benchmarks of real proofs, and verification of saved ones.

use crate::{FieldChoice, HashChoice, Result, Source, with_types};
use clap::Args;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zo0k::tuning::Analysis;
use zo0k::{Commitment, Hasher, Parameters, Proof, Protocol, Rng, RsField, Variant};

#[derive(Args)]
pub struct Bench {
    #[command(flatten)]
    source: Source,
    /// Number of proofs to generate; the report has the median times.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Save the statement and the last proof to this new directory.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// A public statement, as saved next to a proof.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Statement {
    /// The names of the field and of the hash function.
    field: String,
    hash: String,
    parameters: Parameters,
    /// Hexadecimal encodings of the commitment, the coordinates of the
    /// evaluation point, and the claimed value.
    commitment: String,
    point: Vec<String>,
    value: String,
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn from_hex(text: &str, len: usize) -> Result<Vec<u8>> {
    if text.len() != 2 * len || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("expected {} hexadecimal digits", 2 * len).into());
    }
    let digits = text.as_bytes().chunks(2);
    Ok(digits.map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap()).collect())
}

fn field_to_hex<F: RsField>(element: F) -> String {
    let mut bytes = vec![0; F::BYTES];
    element.write_bytes(&mut bytes);
    to_hex(&bytes)
}

fn field_from_hex<F: RsField>(text: &str) -> Result<F> {
    Ok(F::from_bytes(&from_hex(text, F::BYTES)?).ok_or("a field element is not encoded canonically")?)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 { values[middle] } else { (values[middle - 1] + values[middle]) / 2.0 }
}

pub fn bench<F: RsField, H: Hasher>(arguments: &Bench) -> Result<()> {
    let parameters = arguments.source.parameters(Variant::ZeroKnowledge, F::SIZE)?;
    let analysis = Analysis::new(&parameters, Variant::ZeroKnowledge, F::SIZE)?;
    let protocol = Protocol::<F, H>::new(parameters.clone())?;
    if arguments.repeats == 0 {
        return Err("--repeats must be positive".into());
    }
    if let Some(directory) = arguments.out.as_ref().filter(|directory| directory.exists()) {
        return Err(format!("{} exists already", directory.display()).into());
    }

    // A fixed table and point make runs comparable. The prover's randomness
    // is fresh.
    let mut fixture = Rng::from_seed(*blake3::hash(b"zo0k benchmark fixture").as_bytes());
    let table: Vec<F> = fixture.fields(1 << parameters.num_variables);
    let point: Vec<F> = fixture.fields(parameters.num_variables);
    let mut rng = Rng::from_entropy();

    let (mut commit_times, mut prove_times, mut verify_times, mut sizes) = (vec![], vec![], vec![], vec![]);
    let mut last = None;
    for _ in 0..arguments.repeats {
        let table = table.clone();
        let start = Instant::now();
        let (commitment, witness) = protocol.commit(table, &mut rng)?;
        commit_times.push(start.elapsed().as_secs_f64());
        let start = Instant::now();
        let (value, proof) = protocol.prove(&commitment, witness, &point, &mut rng)?;
        prove_times.push(start.elapsed().as_secs_f64());
        let start = Instant::now();
        protocol.verify(&commitment, &point, value, &proof)?;
        verify_times.push(start.elapsed().as_secs_f64());
        // The argument consists of the commitment and the proof.
        sizes.push((proof.0.len() + commitment.0.len()) as f64);
        last = Some((commitment, value, proof));
    }

    let (commitment, value, proof) = last.expect("at least one repeat");
    if let Some(directory) = &arguments.out {
        let statement = Statement {
            field: F::NAME.into(),
            hash: H::NAME.into(),
            parameters: parameters.clone(),
            commitment: to_hex(&commitment.0),
            point: point.iter().map(|&coordinate| field_to_hex(coordinate)).collect(),
            value: field_to_hex(value),
        };
        fs::create_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
        fs::write(directory.join("statement.json"), serde_json::to_string_pretty(&statement)?)?;
        fs::write(directory.join("proof.bin"), &proof.0)?;
    }
    let costs = &analysis.costs;
    let report = serde_json::json!({
        "field": F::NAME,
        "hash": H::NAME,
        "parameters": parameters,
        "threads": rayon::current_num_threads(),
        "repeats": arguments.repeats,
        "commit_seconds": median(commit_times),
        "prove_seconds": median(prove_times),
        "verify_seconds": median(verify_times),
        "argument_bytes": median(sizes),
        "expected_argument_bytes": costs.expected_argument_bytes,
        "iop_bytes": costs.iop_bytes,
        "soundness_bits": analysis.soundness.total_bits,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub fn verify(directory: &Path) -> Result<()> {
    let read = |file: &str| {
        let path = directory.join(file);
        fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))
    };
    let statement: Statement = serde_json::from_slice(&read("statement.json")?)?;
    let proof = Proof(read("proof.bin")?);
    let field =
        FieldChoice::from_str(&statement.field, false).map_err(|_| "the statement has an unknown field")?;
    let hash =
        HashChoice::from_str(&statement.hash, false).map_err(|_| "the statement has an unknown hash")?;
    with_types!(field, hash, verify_statement, &statement, &proof)
}

fn verify_statement<F: RsField, H: Hasher>(statement: &Statement, proof: &Proof) -> Result<()> {
    let point =
        statement.point.iter().map(|coordinate| field_from_hex(coordinate)).collect::<Result<Vec<F>>>()?;
    let value: F = field_from_hex(&statement.value)?;
    let commitment = Commitment(from_hex(&statement.commitment, 32)?.try_into().unwrap());
    let protocol = Protocol::<F, H>::new(statement.parameters.clone())?;
    protocol.verify(&commitment, &point, value, proof)?;
    // The parameters come from the statement, so say what they guarantee.
    let soundness = Analysis::new(&statement.parameters, Variant::ZeroKnowledge, F::SIZE)?.soundness;
    println!(
        "The proof is valid. The parameters of the statement have a soundness error of 2^-{:.1} over {} with {}.",
        soundness.total_bits,
        F::NAME,
        H::NAME
    );
    Ok(())
}
