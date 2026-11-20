use crate::algebra::naive_multilinear;
use crate::composed::compose;
use crate::fields::uniforms;
use crate::prng::Prng;
use crate::protocol::{Instance, prove, rejected, verify};
use crate::report::ensure;
use zo0k::parameters::{Parameters, StageParameters, Variant};
use zo0k::tuning::{Analysis, optimize_queries};
use zo0k::{Blake3, Error, Field, Gf2_128, Gf2_256, Hasher, Keccak256, Proof, Protocol, RsField};

fn parameters<F: Field>(
    num_variables: usize,
    schedule: &[(usize, usize)],
    pow_bits: u32,
) -> Result<Parameters, String> {
    let mut parameters = Parameters {
        num_variables,
        stages: schedule
            .iter()
            .map(|&(folding, domain_size)| StageParameters::new(folding, domain_size, 0))
            .collect(),
        security_bits: 24,
        pow_bits,
        base_case_pow_bits: 0,
        johnson_slack: 0.02,
    };
    optimize_queries(&mut parameters, Variant::ZeroKnowledge, F::SIZE).map_err(|e| e.to_string())?;
    Ok(parameters)
}

fn ground<F: Field>() -> Result<Parameters, String> {
    let mut parameters = parameters::<F>(9, &[(2, 1100), (3, 300)], 3)?;
    parameters.stages[0].sumcheck_pow_bits = 4;
    parameters.stages[0].ood_pow_bits = 2;
    parameters.stages[1].sumcheck_pow_bits = 1;
    parameters.base_case_pow_bits = 5;
    Ok(parameters)
}

fn schedules<F: Field>() -> Result<Vec<Parameters>, String> {
    Ok(vec![
        ground::<F>()?,
        parameters::<F>(6, &[(3, 256)], 0)?,
        parameters::<F>(5, &[(5, 200)], 4)?,
        parameters::<F>(9, &[(2, 1100), (3, 300)], 3)?,
        parameters::<F>(9, &[(4, 512), (2, 256)], 0)?,
        parameters::<F>(11, &[(3, 1500), (2, 700), (1, 500), (3, 300)], 2)?,
        parameters::<F>(8, &[(1, 700), (7, 257)], 0)?,
    ])
}

fn instance<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    prng: &mut Prng,
) -> Result<Instance<F>, String> {
    let n = protocol.parameters().num_variables;
    let table: Vec<F> = uniforms(prng, 1 << n);
    let point: Vec<F> = uniforms(prng, n);
    let expected = naive_multilinear(&table, &point);
    let instance = prove(protocol, table, &point, prng.seed_bytes()).map_err(|e| e.to_string())?;
    ensure(instance.value == expected, || "the value differs from the naive evaluation".into())?;
    Ok(instance)
}

fn honest_sizes<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    for parameters in schedules::<F>()? {
        let analysis =
            Analysis::new(&parameters, Variant::ZeroKnowledge, F::SIZE).map_err(|e| e.to_string())?;
        let protocol = Protocol::<F, H>::new(parameters.clone()).map_err(|e| e.to_string())?;
        let repetitions = 30;
        let mut opening_bytes = 0.0;
        for _ in 0..repetitions {
            let instance = instance(&protocol, prng)?;
            verify(&protocol, &instance).map_err(|e| format!("{parameters:?}: {e}"))?;
            let costs = &analysis.costs;
            let fixed = costs.root_bytes + costs.message_bytes + costs.pow_nonce_bytes;
            opening_bytes += (instance.proof.0.len() + 32 - fixed) as f64 / repetitions as f64;
            let mut other = instance.clone();
            other.value += F::ONE;
            rejected(&protocol, &other).map_err(|e| format!("{parameters:?} another value: {e}"))?;
        }
        let relative = opening_bytes / analysis.costs.expected_opening_bytes - 1.0;
        ensure(relative.abs() < 0.04, || format!("{parameters:?}: openings are off by {relative}"))?;
    }
    Ok(())
}

fn special_tables<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let n = 9;
    let protocol =
        Protocol::<F, H>::new(parameters::<F>(n, &[(2, 1100), (3, 300)], 0)?).map_err(|e| e.to_string())?;
    let cases: Vec<(Vec<F>, Vec<F>)> = vec![
        (vec![F::ZERO; 1 << n], uniforms(prng, n)),
        (vec![F::ONE; 1 << n], uniforms(prng, n)),
        (uniforms(prng, 1 << n), vec![F::ZERO; n]),
        (uniforms(prng, 1 << n), vec![F::ONE; n]),
    ];
    for (table, point) in cases {
        let expected = naive_multilinear(&table, &point);
        let instance = prove(&protocol, table, &point, prng.seed_bytes()).map_err(|e| e.to_string())?;
        ensure(instance.value == expected, || "a special value differs".into())?;
        verify(&protocol, &instance).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn every_byte<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let schedules =
        [parameters::<F>(6, &[(3, 256)], 3)?, parameters::<F>(9, &[(2, 1100), (3, 300)], 3)?, ground::<F>()?];
    for parameters in schedules {
        let protocol = Protocol::<F, H>::new(parameters).map_err(|e| e.to_string())?;
        let instance = instance(&protocol, prng)?;
        let len = instance.proof.0.len();
        let step = if len < 20_000 { 1 } else { 13 };
        let mut altered = instance.clone();
        for position in (0..len).step_by(step) {
            altered.proof.0[position] ^= 1 << (position % 8);
            ensure(verify(&protocol, &altered).is_err(), || {
                format!("byte {position} of {len} is not bound")
            })?;
            altered.proof.0[position] ^= 1 << (position % 8);
        }
        for proof in [
            Proof(instance.proof.0[..len - 1].to_vec()),
            Proof([&instance.proof.0[..], &[0]].concat()),
            Proof(Vec::new()),
        ] {
            let other = Instance { proof, ..instance.clone() };
            ensure(verify(&protocol, &other).is_err(), || {
                format!("a proof of {} bytes for {len}", other.proof.0.len())
            })?;
        }
    }
    Ok(())
}

fn statement_bound<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let parameters = parameters::<F>(9, &[(2, 1100), (3, 300)], 3)?;
    let protocol = Protocol::<F, H>::new(parameters.clone()).map_err(|e| e.to_string())?;
    let instance = instance(&protocol, prng)?;
    verify(&protocol, &instance).map_err(|e| e.to_string())?;
    let mut other = instance.clone();
    other.point[4] += F::ONE;
    ensure(verify(&protocol, &other).is_err(), || "another point".into())?;
    let mut other = instance.clone();
    other.commitment.0[7] ^= 1;
    ensure(verify(&protocol, &other).is_err(), || "another commitment".into())?;
    let edits: [fn(&mut Parameters); 8] = [
        |p| p.security_bits -= 1,
        |p| p.johnson_slack = 0.019,
        |p| p.pow_bits += 1,
        |p| p.base_case_pow_bits += 1,
        |p| p.stages[0].sumcheck_pow_bits += 1,
        |p| p.stages[0].ood_pow_bits += 1,
        |p| p.stages[0].queries += 1,
        |p| p.stages[1].domain_size += 1,
    ];
    for (index, edit) in edits.iter().enumerate() {
        let mut changed = parameters.clone();
        edit(&mut changed);
        let changed = Protocol::<F, H>::new(changed).map_err(|e| format!("edit {index}: {e}"))?;
        rejected(&changed, &instance).map_err(|e| format!("parameter edit {index}: {e}"))?;
    }
    ensure(
        instance.proof
            != prove(&protocol, vec![F::ZERO; 1 << 9], &instance.point, prng.seed_bytes())
                .map_err(|e| e.to_string())?
                .proof,
        || "proofs repeat".into(),
    )
}

fn invalid_inputs<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let other_folding = parameters::<F>(6, &[(2, 256)], 0)?;
    let parameters = parameters::<F>(6, &[(3, 256)], 0)?;
    let protocol = Protocol::<F, H>::new(parameters.clone()).map_err(|e| e.to_string())?;
    let mut rng = zo0k::Rng::from_seed(prng.seed_bytes());
    ensure(matches!(protocol.commit(uniforms(prng, 63), &mut rng), Err(Error::InvalidInput(_))), || {
        "a short table".into()
    })?;
    let (commitment, witness) = protocol.commit(uniforms(prng, 64), &mut rng).map_err(|e| e.to_string())?;
    let short: Vec<F> = uniforms(prng, 5);
    ensure(
        matches!(protocol.prove(&commitment, witness, &short, &mut rng), Err(Error::InvalidInput(_))),
        || "a short point".into(),
    )?;
    ensure(protocol.verify(&commitment, &short, F::ZERO, &Proof(Vec::new())).is_err(), || {
        "verifying a short point".into()
    })?;
    let point: Vec<F> = uniforms(prng, 6);
    let mut longer = parameters.clone();
    longer.stages[0].domain_size = 300;
    for other in [other_folding, parameters.clone(), longer] {
        let other = Protocol::<F, H>::new(other).map_err(|e| e.to_string())?;
        let (_, foreign) = other.commit(uniforms(prng, 64), &mut rng).map_err(|e| e.to_string())?;
        let proved = protocol.prove(&commitment, foreign, &point, &mut rng);
        ensure(matches!(proved, Err(Error::InvalidInput(_))), || "a foreign witness".into())?;
    }
    let edits: [fn(&mut Parameters); 9] = [
        |p| p.stages.clear(),
        |p| p.stages[0].folding = usize::MAX,
        |p| p.stages.push(StageParameters::new(usize::MAX, 256, 20)),
        |p| p.num_variables = usize::MAX,
        |p| p.security_bits = 3_000_000_000,
        |p| p.security_bits = 1 << 31,
        |p| p.stages[0].domain_size = usize::MAX,
        |p| p.johnson_slack = f64::NAN,
        |p| p.stages[0].queries -= 1,
    ];
    for (index, edit) in edits.iter().enumerate() {
        let mut changed = parameters.clone();
        edit(&mut changed);
        ensure(matches!(Protocol::<F, H>::new(changed), Err(Error::InvalidParameters(_))), || {
            format!("adversarial edit {index}")
        })?;
    }
    Ok(())
}

fn false_claims<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    for (num_variables, schedule) in [(6, vec![(3, 256)]), (9, vec![(2, 1100), (3, 300)])] {
        let protocol = Protocol::<F, H>::new(parameters::<F>(num_variables, &schedule, 0)?)
            .map_err(|e| e.to_string())?;
        let point: Vec<F> = uniforms(prng, num_variables);
        for error in [F::ZERO, F::ONE] {
            let seed = prng.seed_bytes();
            let switch_seeds: Vec<[u8; 32]> = (1..schedule.len()).map(|_| prng.seed_bytes()).collect();
            let table = uniforms(prng, 1 << num_variables);
            let composed = compose::<F, H>(
                protocol.parameters(),
                protocol.layout(),
                table,
                &point,
                error,
                (&seed, &switch_seeds),
            );
            let result = protocol.verify(&composed.commitment, &point, composed.value, &composed.proof);
            ensure(result.is_ok() == (error == F::ZERO), || {
                format!("a claim off by {error:?} gives {result:?}")
            })?;
            ensure(matches!(result, Ok(()) | Err(Error::InvalidProof(_))), || {
                "a false claim gives another error".into()
            })?;
        }
    }
    Ok(())
}

fn per_types<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    honest_sizes::<F, H>(prng).map_err(|e| format!("{}/{} honest sizes: {e}", F::NAME, H::NAME))?;
    special_tables::<F, H>(prng).map_err(|e| format!("{}/{} special tables: {e}", F::NAME, H::NAME))?;
    every_byte::<F, H>(prng).map_err(|e| format!("{}/{} every byte: {e}", F::NAME, H::NAME))?;
    statement_bound::<F, H>(prng).map_err(|e| format!("{}/{} statement: {e}", F::NAME, H::NAME))?;
    invalid_inputs::<F, H>(prng).map_err(|e| format!("{}/{} invalid inputs: {e}", F::NAME, H::NAME))?;
    false_claims::<F, H>(prng).map_err(|e| format!("{}/{} false claims: {e}", F::NAME, H::NAME))
}

pub fn schedules_gf2_256_blake3() -> Result<(), String> {
    per_types::<Gf2_256, Blake3>(&mut Prng::new(0x5eed_0101))
}

pub fn schedules_gf2_128_blake3() -> Result<(), String> {
    per_types::<Gf2_128, Blake3>(&mut Prng::new(0x5eed_0102))
}

pub fn schedules_gf2_256_keccak256() -> Result<(), String> {
    per_types::<Gf2_256, Keccak256>(&mut Prng::new(0x5eed_0103))
}

pub fn schedules_gf2_128_keccak256() -> Result<(), String> {
    per_types::<Gf2_128, Keccak256>(&mut Prng::new(0x5eed_0104))
}

pub fn fields_and_hashes_bind() -> Result<(), String> {
    let mut prng = Prng::new(0x5eed_0105);
    let parameters = parameters::<Gf2_256>(6, &[(3, 256)], 0)?;
    let blake = Protocol::<Gf2_256, Blake3>::new(parameters.clone()).map_err(|e| e.to_string())?;
    let keccak = Protocol::<Gf2_256, Keccak256>::new(parameters).map_err(|e| e.to_string())?;
    let blake_instance = instance(&blake, &mut prng)?;
    verify(&blake, &blake_instance).map_err(|e| e.to_string())?;
    rejected(&keccak, &blake_instance).map_err(|e| format!("under keccak256: {e}"))?;
    let standard = Parameters::standard(12, 100, Gf2_256::SIZE).map_err(|e| e.to_string())?;
    ensure(Protocol::<Gf2_256, Blake3>::new(standard.clone()).is_ok(), || {
        "standard parameters over gf2_256".into()
    })?;
    let small = Protocol::<Gf2_128, Blake3>::new(standard);
    ensure(matches!(small, Err(Error::InvalidParameters(_))), || {
        "gf2_256 parameters suffice over gf2_128".into()
    })?;
    let parameters = Parameters::standard(13, 100, Gf2_128::SIZE).map_err(|e| e.to_string())?;
    ensure(parameters.stages.iter().all(|stage| stage.sumcheck_pow_bits > 0), || {
        "no sumcheck grinding".into()
    })?;
    ensure(parameters.stages[0].ood_pow_bits > 0 && parameters.base_case_pow_bits > 0, || {
        "no grinding".into()
    })?;
    let analysis =
        Analysis::new(&parameters, Variant::ZeroKnowledge, Gf2_128::SIZE).map_err(|e| e.to_string())?;
    ensure(analysis.soundness.total_bits >= 100.0, || "standard gf2_128 misses 100 bits".into())?;
    let protocol = Protocol::<Gf2_128, Blake3>::new(parameters.clone()).map_err(|e| e.to_string())?;
    let instance = instance(&protocol, &mut prng)?;
    verify(&protocol, &instance).map_err(|e| e.to_string())?;
    let mut other = instance.clone();
    other.value += Gf2_128::ONE;
    rejected(&protocol, &other).map_err(|e| format!("another value: {e}"))?;
    let mut weaker = parameters;
    weaker.stages[0].sumcheck_pow_bits = 0;
    ensure(matches!(Protocol::<Gf2_128, Blake3>::new(weaker), Err(Error::InvalidParameters(_))), || {
        "grinding is not needed".into()
    })
}

pub const THREADS_ENV: &str = "RAYON_NUM_THREADS";

pub fn reference_proof() -> Result<Vec<u8>, String> {
    let parameters = parameters::<Gf2_256>(9, &[(2, 1100), (3, 300)], 6)?;
    let protocol = Protocol::<Gf2_256, Blake3>::new(parameters).map_err(|e| e.to_string())?;
    let mut prng = Prng::new(0x5eed_0106);
    let table: Vec<Gf2_256> = uniforms(&mut prng, 1 << 9);
    let point: Vec<Gf2_256> = uniforms(&mut prng, 9);
    let instance = prove(&protocol, table, &point, [7; 32]).map_err(|e| e.to_string())?;
    let bytes = [&instance.commitment.0[..], &instance.proof.0].concat();
    ensure(bytes.len() > 32, || "the reference proof is empty".into())?;
    Ok(bytes)
}

pub fn fingerprint(bytes: &[u8]) -> String {
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        state = (state ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
    let text = format!("{}:{state:016x}", bytes.len());
    assert!(text.len() > 17);
    text
}

pub fn threads_agree() -> Result<(), String> {
    let expected = fingerprint(&reference_proof()?);
    let concurrent: Vec<Result<Vec<u8>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4).map(|_| scope.spawn(reference_proof)).collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or_else(|_| Err("a prover thread panics".into())))
            .collect()
    });
    for proof in concurrent {
        ensure(fingerprint(&proof?) == expected, || "concurrent provers give other proofs".into())?;
    }
    let current = std::env::current_exe().map_err(|e| e.to_string())?;
    for threads in ["1", "2", "5"] {
        let output = std::process::Command::new(&current)
            .arg("--proof-fingerprint")
            .env(THREADS_ENV, threads)
            .output()
            .map_err(|e| format!("cannot run {}: {e}", current.display()))?;
        let printed = String::from_utf8_lossy(&output.stdout);
        ensure(output.status.success(), || format!("the fingerprint run with {threads} threads fails"))?;
        ensure(printed.trim() == expected, || {
            format!("{threads} threads give proof {} instead of {expected}", printed.trim())
        })?;
    }
    Ok(())
}
