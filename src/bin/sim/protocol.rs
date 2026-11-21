use crate::algebra::naive_multilinear;
use crate::composed::{compose, frame, statement};
use crate::fields::{convert, nonzero, special, uniform, uniforms};
use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure, guarded};
use zo0k::costs::Costs;
use zo0k::field::BinaryField;
use zo0k::hash::DIGEST_BYTES;
use zo0k::parameters::{Layout, MAX_QUERIES, Parameters, StageParameters, Variant};
use zo0k::soundness::Soundness;
use zo0k::tuning::{Analysis, assign_pow_bits, optimize_queries, uniform_queries};
use zo0k::{
    Blake3, Commitment, Error, Field, FieldSize, Gf2_128, Gf2_256, Hasher, Keccak256, Proof, Protocol, Rng,
    RsField,
};

pub const NUM_VARIABLES_MAX: usize = 12;
const STAGES_MAX: usize = 4;
const FOLDING_MAX: usize = 6;
const GENERATE_ATTEMPTS: usize = 6;
const FLIPS: usize = 40;
const SWEEP_VERIFICATIONS_MAX: usize = 120;
const ADVERSARIAL_TRIALS: usize = 24;
const SLACKS: [f64; 5] = [0.0002, 0.005, 0.02, 0.05, 0.1];

fn candidate(prng: &mut Prng, attempt: usize) -> Parameters {
    let num_variables = if prng.chance(75) { prng.range(3, 10) } else { prng.range(1, NUM_VARIABLES_MAX) };
    let count = prng.range(1, num_variables.min(STAGES_MAX));
    let mut foldings = vec![1usize; count];
    for _ in 0..prng.range(0, num_variables - count) {
        let index = prng.below(count);
        if foldings[index] < FOLDING_MAX {
            foldings[index] += 1;
        }
    }
    let security_bits = prng.range(8, 40) as u32;
    let pow_bits = prng.range(0, 6) as u32;
    let reserve = 2 * security_bits as usize + 64;
    let mut remaining = num_variables;
    let mut stages = Vec::with_capacity(count);
    for &folding in &foldings {
        remaining -= folding;
        let rate = prng.pick(&[2usize, 3, 4, 8]) << attempt.min(3);
        let domain_size = rate * ((1 << remaining) + reserve) + prng.range(0, 64);
        stages.push(StageParameters::new(folding, domain_size, 0));
    }
    let johnson_slack = prng.pick(&SLACKS);
    assert!(foldings.iter().sum::<usize>() <= num_variables);
    Parameters { num_variables, stages, security_bits, pow_bits, base_case_pow_bits: 0, johnson_slack }
}

fn add_grinding(prng: &mut Prng, parameters: &mut Parameters) {
    if prng.chance(50) {
        return;
    }
    let count = parameters.stages.len();
    for (index, stage) in parameters.stages.iter_mut().enumerate() {
        stage.sumcheck_pow_bits = prng.pick(&[0, 0, 1, 2, 3]);
        if index + 1 < count {
            stage.ood_pow_bits = prng.pick(&[0, 0, 1, 2, 3]);
        }
    }
    parameters.base_case_pow_bits = prng.pick(&[0, 0, 1, 2, 4]);
    assert_eq!(parameters.stages[count - 1].ood_pow_bits, 0);
}

pub fn generate(prng: &mut Prng, field: FieldSize) -> Result<Parameters, String> {
    let mut last = String::new();
    for attempt in 0..GENERATE_ATTEMPTS {
        let mut parameters = candidate(prng, attempt);
        match optimize_queries(&mut parameters, Variant::ZeroKnowledge, field) {
            Ok(_) => {
                add_grinding(prng, &mut parameters);
                let analysis = Analysis::new(&parameters, Variant::ZeroKnowledge, field);
                ensure(analysis.is_ok(), || format!("grinding breaks admissible parameters {parameters:?}"))?;
                return Ok(parameters);
            }
            Err(error) => last = format!("{parameters:?}: {error}"),
        }
    }
    Err(format!("no admissible schedule in {GENERATE_ATTEMPTS} attempts, last {last}"))
}

pub struct Input<F> {
    pub table: Vec<F>,
    pub point: Vec<F>,
    pub kinds: (&'static str, &'static str),
}

fn random_table<F: Field>(prng: &mut Prng, num_variables: usize) -> (Vec<F>, &'static str) {
    let len = 1usize << num_variables;
    let table = match prng.below(10) {
        0 => (vec![F::ZERO; len], "zero"),
        1 => (vec![nonzero::<F>(prng); len], "constant"),
        2 => {
            let mut table = vec![F::ZERO; len];
            table[prng.below(len)] = nonzero(prng);
            (table, "one-hot")
        }
        3 => ((0..len).map(|_| if prng.chance(6) { uniform(prng) } else { F::ZERO }).collect(), "sparse"),
        4 => ((0..len).map(|_| if prng.chance(50) { F::ONE } else { F::ZERO }).collect(), "boolean"),
        5 => ((0..len).map(|_| special(prng)).collect(), "special"),
        _ => (uniforms(prng, len), "uniform"),
    };
    assert_eq!(table.0.len(), len);
    table
}

fn random_point<F: Field>(prng: &mut Prng, num_variables: usize) -> (Vec<F>, &'static str) {
    let z: F = uniform(prng);
    let point = match prng.below(8) {
        0 => (vec![F::ZERO; num_variables], "zero"),
        1 => (vec![F::ONE; num_variables], "ones"),
        2 => {
            ((0..num_variables).map(|_| if prng.chance(50) { F::ONE } else { F::ZERO }).collect(), "boolean")
        }
        3 => (vec![z; num_variables], "repeated"),
        4 => (
            (0..num_variables)
                .map(|_| {
                    let w = uniform(prng);
                    prng.pick(&[F::ZERO, F::ONE, z, w])
                })
                .collect(),
            "mixed",
        ),
        5 => ((0..num_variables).map(|_| special(prng)).collect(), "special"),
        _ => (uniforms(prng, num_variables), "uniform"),
    };
    assert_eq!(point.0.len(), num_variables);
    point
}

pub fn random_input<F: Field>(prng: &mut Prng, num_variables: usize) -> Input<F> {
    let (table, table_kind) = random_table(prng, num_variables);
    let (point, point_kind) = random_point(prng, num_variables);
    let input = Input { table, point, kinds: (table_kind, point_kind) };
    assert_eq!(input.table.len(), 1 << input.point.len());
    input
}

#[derive(Clone)]
pub struct Instance<F> {
    pub commitment: Commitment,
    pub point: Vec<F>,
    pub value: F,
    pub proof: Proof,
}

pub fn prove<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    table: Vec<F>,
    point: &[F],
    seed: [u8; 32],
) -> zo0k::Result<Instance<F>> {
    let mut rng = Rng::from_seed(seed);
    let (commitment, witness) = protocol.commit(table, &mut rng)?;
    let (value, proof) = protocol.prove(&commitment, witness, point, &mut rng)?;
    Ok(Instance { commitment, point: point.to_vec(), value, proof })
}

pub fn verify<F: RsField, H: Hasher>(protocol: &Protocol<F, H>, instance: &Instance<F>) -> zo0k::Result<()> {
    protocol.verify(&instance.commitment, &instance.point, instance.value, &instance.proof)
}

pub fn rejected<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    instance: &Instance<F>,
) -> Result<(), String> {
    match verify(protocol, instance) {
        Err(Error::InvalidProof(_)) => Ok(()),
        Ok(()) => Err("accepted".into()),
        Err(other) => Err(format!("rejected with {other} instead of an invalid proof")),
    }
}

fn completeness<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    input: &Input<F>,
    seed: [u8; 32],
) -> Result<Instance<F>, String> {
    let instance = prove(protocol, input.table.clone(), &input.point, seed)
        .map_err(|e| format!("proving fails: {e}"))?;
    verify(protocol, &instance).map_err(|e| format!("an honest proof is rejected: {e}"))?;
    ensure(instance.point == input.point, || "the instance has another point".into())?;
    Ok(instance)
}

fn value_matches<F: RsField>(input: &Input<F>, instance: &Instance<F>) -> Result<(), String> {
    let expected = naive_multilinear(&input.table, &input.point);
    ensure(instance.value == expected, || {
        format!("value {:?} differs from the naive evaluation {expected:?}", instance.value)
    })?;
    let boolean = input.point.iter().all(|&z| z == F::ZERO || z == F::ONE);
    if boolean {
        let index = input.point.iter().fold(0usize, |index, &z| 2 * index + usize::from(z == F::ONE));
        ensure(instance.value == input.table[index], || {
            format!("the value at vertex {index} is not the table entry")
        })?;
    }
    Ok(())
}

fn proof_size<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    instance: &Instance<F>,
) -> Result<(), String> {
    let (parameters, layout) = (protocol.parameters(), protocol.layout());
    let bytes = statement::<F, H>(parameters, &instance.commitment, &instance.point, instance.value);
    let framing = frame::<F, H>(parameters, layout, &bytes, &instance.proof.0)
        .map_err(|e| format!("the proof does not parse: {e}"))?;
    let costs = Costs::new(layout, parameters, F::BYTES);
    let predicted =
        costs.root_bytes - DIGEST_BYTES + costs.message_bytes + costs.pow_nonce_bytes + framing.opening_bytes;
    let actual = instance.proof.0.len();
    ensure(actual == predicted, || {
        format!("the proof has {actual} bytes, the cost model predicts {predicted}")
    })?;
    let analysis = Analysis::new(parameters, Variant::ZeroKnowledge, F::SIZE).map_err(|e| e.to_string())?;
    ensure(analysis.costs.iop_bytes == costs.iop_bytes, || "the analysis has other costs".into())?;
    let columns: usize = layout.stages.iter().map(|stage| stage.rows()).sum();
    ensure(actual >= F::BYTES * (layout.message_len() + columns), || {
        "the proof is below the minimum length".into()
    })?;
    ensure(framing.opening_bytes >= columns * F::BYTES, || "an opening has no column".into())?;
    let within =
        framing.positions.iter().zip(&layout.stages).all(|(p, s)| !p.is_empty() && p.len() <= s.queries);
    ensure(within, || "an opening has more positions than queries".into())
}

fn reproducible<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    input: &Input<F>,
    seed: [u8; 32],
    instance: &Instance<F>,
) -> Result<(), String> {
    let again = prove(protocol, input.table.clone(), &input.point, seed).map_err(|e| e.to_string())?;
    ensure(again.commitment == instance.commitment, || "the same seed gives another commitment".into())?;
    ensure(again.proof == instance.proof, || "the same seed gives another proof".into())?;
    let mut other_seed = seed;
    other_seed[0] ^= 1;
    let other = prove(protocol, input.table.clone(), &input.point, other_seed).map_err(|e| e.to_string())?;
    ensure(other.value == instance.value, || "another seed gives another value".into())?;
    ensure(other.commitment != instance.commitment, || "the commitment is not randomized".into())?;
    ensure(other.proof != instance.proof, || "the proof is not randomized".into())?;
    verify(protocol, &other).map_err(|e| format!("a proof with another seed fails: {e}"))
}

fn reproducible_across_threads<F: RsField, H: Hasher>(
    prng: &mut Prng,
    protocol: &Protocol<F, H>,
    input: &Input<F>,
    seed: [u8; 32],
    instance: &Instance<F>,
) -> Result<(), String> {
    let threads = prng.pick(&crate::threads::THREAD_CHOICES[..4]);
    let _guard = crate::threads::ThreadsGuard::set(threads);
    let again = prove(protocol, input.table.clone(), &input.point, seed).map_err(|e| e.to_string())?;
    ensure(again.commitment == instance.commitment, || format!("{threads} threads give another commitment"))?;
    ensure(again.proof == instance.proof, || format!("{threads} threads give another proof"))?;
    verify(protocol, instance).map_err(|e| format!("verifying with {threads} threads: {e}"))
}

fn new_agrees_with_analysis<F: RsField, H: Hasher>(prng: &mut Prng, base: &Parameters) -> Result<(), String> {
    for _ in 0..4 {
        let mut parameters = base.clone();
        let stage = prng.below(parameters.stages.len());
        let queries = parameters.stages[stage].queries;
        parameters.stages[stage].queries = prng.range(1, queries);
        if prng.chance(30) {
            parameters.security_bits += prng.range(1, 8) as u32;
        }
        let analysis = Analysis::new(&parameters, Variant::ZeroKnowledge, F::SIZE);
        let created = Protocol::<F, H>::new(parameters.clone());
        let agree = created.is_ok() == analysis.is_ok();
        ensure(agree, || {
            format!(
                "Protocol::new is {} but the analysis is {} for {parameters:?}",
                created.is_ok(),
                analysis.is_ok()
            )
        })?;
    }
    Ok(())
}

fn rejects_wrong_statement<F: RsField, H: Hasher>(
    prng: &mut Prng,
    protocol: &Protocol<F, H>,
    instance: &Instance<F>,
) -> Result<(), String> {
    for delta in [F::ONE, nonzero(prng)] {
        let mut other = instance.clone();
        other.value += delta;
        rejected(protocol, &other).map_err(|e| format!("a value off by {delta:?}: {e}"))?;
    }
    let n = instance.point.len();
    if n > 0 {
        let mut other = instance.clone();
        let coordinate = prng.below(n);
        other.point[coordinate] += nonzero::<F>(prng);
        ensure(verify(protocol, &other).is_err(), || {
            format!("a point changed at coordinate {coordinate} is accepted")
        })?;
        let mut shifted = instance.clone();
        shifted.point.iter_mut().for_each(|z| *z += F::ONE);
        ensure(verify(protocol, &shifted).is_err(), || "a shifted point is accepted".into())?;
    }
    let mut other = instance.clone();
    let bit = prng.below(8 * DIGEST_BYTES);
    other.commitment.0[bit / 8] ^= 1 << (bit % 8);
    ensure(verify(protocol, &other).is_err(), || format!("a commitment with bit {bit} flipped is accepted"))
}

fn parameter_edits() -> [fn(&mut Parameters); 9] {
    [
        |p| p.security_bits -= 1,
        |p| p.johnson_slack *= 0.95,
        |p| p.pow_bits += 1,
        |p| p.base_case_pow_bits += 1,
        |p| p.stages[0].sumcheck_pow_bits += 1,
        |p| p.stages[0].ood_pow_bits += 1,
        |p| p.stages[0].queries += 1,
        |p| {
            let last = p.stages.len() - 1;
            p.stages[last].domain_size += 1
        },
        |p| {
            let last = p.stages.len() - 1;
            p.stages[last].queries += 1
        },
    ]
}

fn rejects_other_parameters<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    instance: &Instance<F>,
) -> Result<(), String> {
    let mut tested = 0;
    for (index, edit) in parameter_edits().iter().enumerate() {
        let mut changed = protocol.parameters().clone();
        if changed.security_bits <= 1 {
            continue;
        }
        edit(&mut changed);
        let Ok(other) = Protocol::<F, H>::new(changed) else {
            continue;
        };
        rejected(&other, instance).map_err(|e| format!("parameter edit {index}: {e}"))?;
        tested += 1;
    }
    ensure(tested >= 3, || format!("only {tested} parameter edits are valid"))
}

fn verify_under<F: RsField, G: Hasher>(
    parameters: &Parameters,
    instance: &Instance<F>,
) -> Option<zo0k::Result<()>> {
    let protocol = Protocol::<F, G>::new(parameters.clone()).ok()?;
    Some(verify(&protocol, instance))
}

fn verify_over<F: RsField, G: RsField, H: Hasher>(
    parameters: &Parameters,
    instance: &Instance<F>,
) -> Option<zo0k::Result<()>> {
    let protocol = Protocol::<G, H>::new(parameters.clone()).ok()?;
    let point: Vec<G> = instance.point.iter().map(|&z| convert(z)).collect();
    Some(protocol.verify(&instance.commitment, &point, convert(instance.value), &instance.proof))
}

fn rejects_other_types<F: RsField, H: Hasher>(
    protocol: &Protocol<F, H>,
    instance: &Instance<F>,
) -> Result<(), String> {
    let parameters = protocol.parameters();
    let other_hash = if H::NAME == Blake3::NAME {
        verify_under::<F, Keccak256>(parameters, instance)
    } else {
        verify_under::<F, Blake3>(parameters, instance)
    };
    let hash_rejects = matches!(other_hash, Some(Err(Error::InvalidProof(_))));
    ensure(hash_rejects, || format!("under another hash function: {other_hash:?}"))?;
    let other_field = if F::BYTES == 32 {
        verify_over::<F, Gf2_128, H>(parameters, instance)
    } else {
        verify_over::<F, Gf2_256, H>(parameters, instance)
    };
    ensure(!matches!(other_field, Some(Ok(()))), || "the proof verifies over another field".into())
}

fn sweep_positions(prng: &mut Prng, len: usize) -> Vec<usize> {
    assert!(len > 0);
    let step = len.div_ceil(SWEEP_VERIFICATIONS_MAX).max(1);
    let offset = prng.below(step);
    let positions: Vec<usize> = (offset..len).step_by(step).collect();
    assert!(positions.len() <= SWEEP_VERIFICATIONS_MAX + 1);
    positions
}

fn rejects_tampering<F: RsField, H: Hasher>(
    prng: &mut Prng,
    protocol: &Protocol<F, H>,
    instance: &Instance<F>,
) -> Result<(), String> {
    let len = instance.proof.0.len();
    let mut altered = instance.clone();
    for _ in 0..FLIPS {
        let (byte, bit) = (prng.below(len), prng.below(8));
        altered.proof.0[byte] ^= 1 << bit;
        rejected(protocol, &altered)
            .map_err(|e| format!("bit {bit} of byte {byte} of {len} flipped: {e}"))?;
        altered.proof.0[byte] ^= 1 << bit;
    }
    for byte in sweep_positions(prng, len) {
        let replacement = (prng.range(1, 255) as u8) ^ altered.proof.0[byte];
        let original = std::mem::replace(&mut altered.proof.0[byte], replacement);
        rejected(protocol, &altered).map_err(|e| format!("byte {byte} of {len} replaced: {e}"))?;
        altered.proof.0[byte] = original;
    }
    let (first, second) = (prng.below(len), prng.below(len));
    if altered.proof.0[first] != altered.proof.0[second] {
        altered.proof.0.swap(first, second);
        rejected(protocol, &altered).map_err(|e| format!("bytes {first} and {second} swapped: {e}"))?;
        altered.proof.0.swap(first, second);
    }
    ensure(altered.proof == instance.proof, || "the sim did not restore the proof".into())
}

fn rejects_resizing<F: RsField, H: Hasher>(
    prng: &mut Prng,
    protocol: &Protocol<F, H>,
    instance: &Instance<F>,
) -> Result<(), String> {
    let len = instance.proof.0.len();
    for cut in [len - 1, prng.below(len), 0, len.saturating_sub(DIGEST_BYTES)] {
        let mut other = instance.clone();
        other.proof.0.truncate(cut);
        rejected(protocol, &other).map_err(|e| format!("a proof cut to {cut} of {len}: {e}"))?;
    }
    let extra_len = prng.range(1, 64);
    for extra in [vec![0u8], prng.bytes(extra_len), instance.proof.0[..DIGEST_BYTES.min(len)].to_vec()] {
        let mut other = instance.clone();
        other.proof.0.extend_from_slice(&extra);
        rejected(protocol, &other).map_err(|e| format!("a proof extended by {} bytes: {e}", extra.len()))?;
    }
    let mut garbage = instance.clone();
    garbage.proof = Proof(prng.bytes(len));
    rejected(protocol, &garbage).map_err(|e| format!("random bytes of the proof length: {e}"))
}

fn rejects_foreign_proofs<F: RsField, H: Hasher>(
    prng: &mut Prng,
    protocol: &Protocol<F, H>,
    input: &Input<F>,
    instance: &Instance<F>,
) -> Result<(), String> {
    let n = input.point.len();
    let mut table = input.table.clone();
    let index = prng.below(table.len());
    table[index] += nonzero::<F>(prng);
    let other = prove(protocol, table, &input.point, prng.seed_bytes()).map_err(|e| e.to_string())?;
    let mut swapped = instance.clone();
    swapped.proof = other.proof.clone();
    rejected(protocol, &swapped).map_err(|e| format!("the proof of another table: {e}"))?;
    let mut mixed = other.clone();
    mixed.commitment = instance.commitment;
    ensure(verify(protocol, &mixed).is_err(), || "a proof verifies for another commitment".into())?;
    let point: Vec<F> = uniforms(prng, n);
    let moved = prove(protocol, input.table.clone(), &point, prng.seed_bytes()).map_err(|e| e.to_string())?;
    let mut swapped = instance.clone();
    swapped.proof = moved.proof;
    ensure(verify(protocol, &swapped).is_err() || point == instance.point, || {
        "the proof of another point is accepted".into()
    })
}

fn invalid_inputs<F: RsField, H: Hasher>(
    prng: &mut Prng,
    protocol: &Protocol<F, H>,
    input: &Input<F>,
) -> Result<(), String> {
    let n = input.point.len();
    let len = input.table.len();
    let mut rng = Rng::from_seed(prng.seed_bytes());
    for wrong in [len - 1, len + 1, 2 * len] {
        let result = protocol.commit(uniforms(prng, wrong), &mut rng);
        ensure(matches!(result, Err(Error::InvalidInput(_))), || {
            format!("a table of {wrong} entries for {len}")
        })?;
    }
    let (commitment, witness) = protocol.commit(input.table.clone(), &mut rng).map_err(|e| e.to_string())?;
    for wrong in [n.saturating_sub(1), n + 1] {
        if wrong == n {
            continue;
        }
        let point: Vec<F> = uniforms(prng, wrong);
        let verified = protocol.verify(&commitment, &point, F::ZERO, &Proof(prng.bytes(64)));
        ensure(matches!(verified, Err(Error::InvalidInput(_))), || {
            format!("verifying a point of {wrong} coordinates for {n}")
        })?;
        let (_, witness) = protocol.commit(input.table.clone(), &mut rng).map_err(|e| e.to_string())?;
        let proved = protocol.prove(&commitment, witness, &point, &mut rng);
        ensure(matches!(proved, Err(Error::InvalidInput(_))), || {
            format!("proving a point of {wrong} coordinates for {n}")
        })?;
    }
    drop(witness);
    let (_, foreign) = protocol.commit(input.table.clone(), &mut rng).map_err(|e| e.to_string())?;
    let proved = protocol.prove(&commitment, foreign, &input.point, &mut rng);
    ensure(matches!(proved, Err(Error::InvalidInput(_))), || {
        "a witness of another commitment is accepted".into()
    })?;
    for other in foreign_parameters::<F>(protocol.parameters()) {
        let Ok(other) = Protocol::<F, H>::new(other) else {
            continue;
        };
        let (_, foreign) = other.commit(input.table.clone(), &mut rng).map_err(|e| e.to_string())?;
        let proved = protocol.prove(&commitment, foreign, &input.point, &mut rng);
        ensure(matches!(proved, Err(Error::InvalidInput(_))), || {
            format!("a witness of {:?} is accepted", other.parameters())
        })?;
    }
    Ok(())
}

fn foreign_parameters<F: Field>(parameters: &Parameters) -> Vec<Parameters> {
    let mut longer = parameters.clone();
    longer.stages[0].domain_size += 1;
    let mut others = vec![longer];
    let n = parameters.num_variables;
    let first = parameters.stages[0].folding;
    let folding = if first > 1 { first - 1 } else { first + 1 };
    if folding <= n {
        let main_len = 1usize << (n - folding);
        let domain_size = 8 * (main_len + 2 * parameters.security_bits as usize + 64);
        let mut single =
            Parameters { stages: vec![StageParameters::new(folding, domain_size, 0)], ..parameters.clone() };
        single.base_case_pow_bits = 0;
        if optimize_queries(&mut single, Variant::ZeroKnowledge, F::SIZE).is_ok() {
            others.push(single);
        }
    }
    assert!(!others.is_empty());
    others
}

fn composed_prover<F: RsField, H: Hasher>(
    prng: &mut Prng,
    protocol: &Protocol<F, H>,
    input: &Input<F>,
) -> Result<(), String> {
    let (parameters, layout) = (protocol.parameters(), protocol.layout());
    let seed = prng.seed_bytes();
    let switch_seeds: Vec<[u8; 32]> = (1..layout.stages.len()).map(|_| prng.seed_bytes()).collect();
    let seeds = (&seed, &switch_seeds[..]);
    let honest = compose::<F, H>(parameters, layout, input.table.clone(), &input.point, F::ZERO, seeds);
    let verified = protocol.verify(&honest.commitment, &input.point, honest.value, &honest.proof);
    ensure(verified.is_ok(), || {
        format!(
            "the composed honest prover is rejected ({verified:?}): the sim is out of step with the library"
        )
    })?;
    if layout.stages.len() == 1 {
        let library = prove(protocol, input.table.clone(), &input.point, seed).map_err(|e| e.to_string())?;
        ensure(library.commitment == honest.commitment && library.proof == honest.proof, || {
            "the library prover differs from the composed prover".into()
        })?;
    }
    let error = if prng.chance(50) { F::ONE } else { nonzero(prng) };
    let lying = compose::<F, H>(parameters, layout, input.table.clone(), &input.point, error, seeds);
    let instance = Instance {
        commitment: lying.commitment,
        point: input.point.clone(),
        value: lying.value,
        proof: lying.proof,
    };
    ensure(instance.value == honest.value + error, || "the false claim is not off by the error".into())?;
    rejected(protocol, &instance).map_err(|e| format!("a false claim off by {error:?}: {e}"))
}

fn wild_parameters(prng: &mut Prng) -> Parameters {
    let num_variables =
        if prng.chance(50) { prng.range(0, 14) } else { prng.pick(&[0, 1, 33, 40, 41, 64, usize::MAX]) };
    let count = prng.range(0, 5);
    let stages = (0..count).map(|_| wild_stage(prng)).collect();
    let parameters = Parameters {
        num_variables,
        stages,
        security_bits: prng.pick(&[0, 1, 4, 8, 16, 24, 100, 256, 257, 1 << 31, 3_000_000_000, u32::MAX]),
        pow_bits: wild_bits(prng),
        base_case_pow_bits: wild_bits(prng),
        johnson_slack: wild_slack(prng),
    };
    assert!(parameters.stages.len() <= 5);
    parameters
}

fn wild_bits(prng: &mut Prng) -> u32 {
    prng.pick(&[0, 0, 0, 1, 2, 8, 40, 41, 64, u32::MAX])
}

fn wild_slack(prng: &mut Prng) -> f64 {
    let slacks = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        5e-324,
        0.0001,
        0.0002,
        0.02,
        0.1,
        0.1000001,
        1.0,
        -0.02,
    ];
    prng.pick(&slacks)
}

fn wild_stage(prng: &mut Prng) -> StageParameters {
    let folding = if prng.chance(50) {
        prng.range(1, 4)
    } else {
        prng.pick(&[0, 12, 40, 41, 63, 64, 1 << 62, usize::MAX])
    };
    let domain_size = if prng.chance(50) {
        prng.range(1, 5000)
    } else {
        prng.pick(&[0, 1, 2, 64, 256, 1 << 32, (1 << 32) + 1, usize::MAX])
    };
    let queries = if prng.chance(50) {
        prng.range(1, 60)
    } else {
        prng.pick(&[0, 1, MAX_QUERIES, MAX_QUERIES + 1, usize::MAX])
    };
    let mut stage = StageParameters::new(folding, domain_size, queries);
    stage.sumcheck_pow_bits = wild_bits(prng);
    stage.ood_pow_bits = wild_bits(prng);
    stage
}

fn mutate(prng: &mut Prng, base: &Parameters) -> Parameters {
    let mut parameters = base.clone();
    for _ in 0..prng.range(1, 3) {
        let index = prng.below(parameters.stages.len().max(1));
        match prng.below(9) {
            0 => {
                parameters.num_variables =
                    parameters.num_variables.wrapping_add(prng.pick(&[1, usize::MAX, 30]))
            }
            1 => parameters.security_bits = prng.pick(&[0, 1, 2, 300, u32::MAX]),
            2 => parameters.pow_bits = wild_bits(prng),
            3 => parameters.johnson_slack = wild_slack(prng),
            4 => parameters.stages.truncate(index),
            5 if index < parameters.stages.len() => parameters.stages[index] = wild_stage(prng),
            6 if index < parameters.stages.len() => {
                parameters.stages[index].queries = prng.pick(&[0, 1, MAX_QUERIES + 1, usize::MAX])
            }
            7 if index < parameters.stages.len() => parameters.stages[index].domain_size /= 2,
            _ => parameters.stages.push(wild_stage(prng)),
        }
    }
    assert!(parameters.stages.len() <= base.stages.len() + 3);
    parameters
}

fn analyses_never_panic(parameters: &Parameters, field: FieldSize) -> Result<(), String> {
    for variant in [Variant::ZeroKnowledge, Variant::Baseline] {
        if let Ok(layout) = Layout::new(parameters, variant) {
            let soundness = Soundness::new(&layout, parameters, field.bits);
            ensure(soundness.is_ok() || matches!(soundness, Err(Error::InvalidParameters(_))), || {
                "soundness error kind".into()
            })?;
            let costs = Costs::new(&layout, parameters, field.bytes);
            ensure(costs.iop_bytes >= costs.message_bytes, || "costs".into())?;
        }
        let analysis = Analysis::ignoring_target(parameters, variant, field);
        ensure(analysis.is_ok() || matches!(analysis, Err(Error::InvalidParameters(_))), || {
            "analysis error kind".into()
        })?;
        let _ = uniform_queries(parameters, variant);
        let _ = assign_pow_bits(&mut parameters.clone(), variant, field, 8);
    }
    Ok(())
}

fn cheap_enough(protocol_parameters: &Parameters, layout: &Layout) -> bool {
    let bits = protocol_parameters.stages.iter().flat_map(|s| [s.sumcheck_pow_bits, s.ood_pow_bits]);
    let grinding =
        bits.chain([protocol_parameters.pow_bits, protocol_parameters.base_case_pow_bits]).all(|b| b <= 10);
    grinding && protocol_parameters.num_variables <= 14 && layout.oracle_len() <= 1 << 17
}

fn adversarial_parameters<F: RsField, H: Hasher>(prng: &mut Prng, base: &Parameters) -> Result<(), String> {
    for trial in 0..ADVERSARIAL_TRIALS {
        let parameters = if trial % 2 == 0 { wild_parameters(prng) } else { mutate(prng, base) };
        let outcome = guarded(|| Ok(Protocol::<F, H>::new(parameters.clone())));
        let protocol = match outcome {
            Err(panic) => return Err(format!("Protocol::new panics on {parameters:?}: {panic}")),
            Ok(Err(Error::InvalidParameters(_))) => None,
            Ok(Err(other)) => return Err(format!("Protocol::new fails with {other} on {parameters:?}")),
            Ok(Ok(protocol)) => Some(protocol),
        };
        guarded(|| analyses_never_panic(&parameters, F::SIZE))
            .map_err(|e| format!("analysis of {parameters:?}: {e}"))?;
        let Some(protocol) = protocol.filter(|p| cheap_enough(&parameters, p.layout())) else {
            continue;
        };
        let input = random_input::<F>(prng, parameters.num_variables);
        let seed = prng.seed_bytes();
        let instance = guarded(|| completeness(&protocol, &input, seed))
            .map_err(|e| format!("accepted {parameters:?}: {e}"))?;
        value_matches(&input, &instance).map_err(|e| format!("accepted {parameters:?}: {e}"))?;
    }
    Ok(())
}

fn run<F: RsField + BinaryField, H: Hasher>(ctx: &mut Ctx) -> Result<(), Failure> {
    let parameters = ctx.check("generate_parameters", |prng| generate(prng, F::SIZE))?;
    ctx.note(|| format!("protocol over {} with {}: {parameters:?}", F::NAME, H::NAME));
    let protocol = ctx.check("protocol_accepts_admissible_parameters", |_| {
        Protocol::<F, H>::new(parameters.clone()).map_err(|e| e.to_string())
    })?;
    let input = ctx.check("generate_input", |prng| Ok(random_input::<F>(prng, parameters.num_variables)))?;
    ctx.note(|| format!("table {} and point {}", input.kinds.0, input.kinds.1));
    let seed = ctx.prng.seed_bytes();
    let instance = ctx.check("completeness", |_| completeness(&protocol, &input, seed))?;
    ctx.note(|| format!("proof of {} bytes", instance.proof.0.len()));
    ctx.check("value_matches_naive_evaluation", |_| value_matches(&input, &instance))?;
    ctx.check("proof_size_matches_cost_model", |_| proof_size(&protocol, &instance))?;
    ctx.check("reproducible_from_prover_seed", |_| reproducible(&protocol, &input, seed, &instance))?;
    ctx.check("reproducible_across_threads", |prng| {
        reproducible_across_threads(prng, &protocol, &input, seed, &instance)
    })?;
    ctx.check("protocol_new_agrees_with_analysis", |prng| {
        new_agrees_with_analysis::<F, H>(prng, &parameters)
    })?;
    ctx.check("rejects_wrong_value_point_commitment", |prng| {
        rejects_wrong_statement(prng, &protocol, &instance)
    })?;
    ctx.check("rejects_other_parameters", |_| rejects_other_parameters(&protocol, &instance))?;
    ctx.check("rejects_other_hash_and_field", |_| rejects_other_types(&protocol, &instance))?;
    ctx.check("rejects_tampered_proofs", |prng| rejects_tampering(prng, &protocol, &instance))?;
    ctx.check("rejects_truncated_and_extended_proofs", |prng| rejects_resizing(prng, &protocol, &instance))?;
    ctx.check("rejects_foreign_proofs", |prng| rejects_foreign_proofs(prng, &protocol, &input, &instance))?;
    ctx.check("invalid_inputs_are_errors", |prng| invalid_inputs(prng, &protocol, &input))?;
    ctx.check("composed_prover_and_false_claims", |prng| composed_prover(prng, &protocol, &input))?;
    ctx.check("adversarial_parameters_never_panic", |prng| {
        adversarial_parameters::<F, H>(prng, &parameters)
    })?;
    Ok(())
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    match ctx.prng.below(4) {
        0 => run::<Gf2_256, Blake3>(ctx),
        1 => run::<Gf2_128, Blake3>(ctx),
        2 => run::<Gf2_256, Keccak256>(ctx),
        _ => run::<Gf2_128, Keccak256>(ctx),
    }
}
