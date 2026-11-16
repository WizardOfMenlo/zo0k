//! End-to-end tests of commitment, proof and verification, for every field
//! and hash function of the crate.

use zo0k::polynomial::evaluate_multilinear;
use zo0k::tuning::{Analysis, optimize_queries};
use zo0k::{
    Blake3, Commitment, Error, Field, Gf2_128, Gf2_256, Hasher, Keccak256, Parameters, Proof, Protocol, Rng,
    RsField, StageParameters, Variant,
};

/// Parameters with a low security target, so that proofs are small.
fn parameters<F: Field>(num_variables: usize, schedule: &[(usize, usize)], pow_bits: u32) -> Parameters {
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
    optimize_queries(&mut parameters, Variant::ZeroKnowledge, F::SIZE).unwrap();
    parameters
}

struct Instance<F> {
    commitment: Commitment,
    point: Vec<F>,
    value: F,
    proof: Proof,
}

impl<F: Field> Clone for Instance<F> {
    fn clone(&self) -> Self {
        Self {
            commitment: self.commitment,
            point: self.point.clone(),
            value: self.value,
            proof: self.proof.clone(),
        }
    }
}

fn prove<F: RsField, H: Hasher>(protocol: &Protocol<F, H>, rng: &mut Rng) -> Instance<F> {
    let num_variables = protocol.parameters().num_variables;
    let table: Vec<F> = rng.fields(1 << num_variables);
    let point: Vec<F> = rng.fields(num_variables);
    let expected = evaluate_multilinear(&table, &point);
    let (commitment, witness) = protocol.commit(table, rng).unwrap();
    let (value, proof) = protocol.prove(&commitment, witness, &point, rng).unwrap();
    assert_eq!(value, expected);
    Instance { commitment, point, value, proof }
}

fn verify<F: RsField, H: Hasher>(protocol: &Protocol<F, H>, instance: &Instance<F>) -> Result<(), Error> {
    protocol.verify(&instance.commitment, &instance.point, instance.value, &instance.proof)
}

/// A schedule with a proof of work before every verifier message that can
/// have one.
fn ground<F: Field>() -> Parameters {
    let mut parameters = parameters::<F>(9, &[(2, 1100), (3, 300)], 3);
    parameters.stages[0].sumcheck_pow_bits = 4;
    parameters.stages[0].ood_pow_bits = 2;
    parameters.stages[1].sumcheck_pow_bits = 1;
    parameters.base_case_pow_bits = 5;
    parameters
}

/// One, two and four stages, domains that are and are not powers of two,
/// last messages of one and of several entries, and proofs of work.
fn schedules<F: Field>() -> Vec<Parameters> {
    vec![
        ground::<F>(),
        parameters::<F>(6, &[(3, 256)], 0),
        parameters::<F>(5, &[(5, 200)], 4),
        parameters::<F>(9, &[(2, 1100), (3, 300)], 3),
        parameters::<F>(9, &[(4, 512), (2, 256)], 0),
        parameters::<F>(11, &[(3, 1500), (2, 700), (1, 500), (3, 300)], 2),
        parameters::<F>(8, &[(1, 700), (7, 257)], 0),
    ]
}

fn honest_proofs_verify_and_have_the_predicted_size<F: RsField, H: Hasher>() {
    let mut rng = Rng::from_seed([1; 32]);
    for parameters in schedules::<F>() {
        let analysis = Analysis::new(&parameters, Variant::ZeroKnowledge, F::SIZE).unwrap();
        let protocol = Protocol::<F, H>::new(parameters.clone()).unwrap();
        let mut opening_bytes = 0.0;
        let repetitions = 30;
        for _ in 0..repetitions {
            let instance = prove(&protocol, &mut rng);
            verify(&protocol, &instance).unwrap();
            let costs = &analysis.costs;
            let fixed = costs.root_bytes + costs.message_bytes + costs.pow_nonce_bytes;
            // The commitment is the first root; the proof holds the others.
            opening_bytes += (instance.proof.0.len() + 32 - fixed) as f64 / repetitions as f64;
        }
        let relative = opening_bytes / analysis.costs.expected_opening_bytes - 1.0;
        assert!(relative.abs() < 0.04, "{parameters:?}: openings are off by {relative}");
    }
}

fn special_tables_and_points_verify<F: RsField, H: Hasher>() {
    let mut rng = Rng::from_seed([2; 32]);
    let parameters = parameters::<F>(9, &[(2, 1100), (3, 300)], 0);
    let protocol = Protocol::<F, H>::new(parameters).unwrap();
    let n = 9;
    let cases: Vec<(Vec<F>, Vec<F>)> = vec![
        (vec![F::ZERO; 1 << n], rng.fields(n)),
        (vec![F::ONE; 1 << n], rng.fields(n)),
        (rng.fields(1 << n), vec![F::ZERO; n]),
        (rng.fields(1 << n), vec![F::ONE; n]),
    ];
    for (table, point) in cases {
        let expected = evaluate_multilinear(&table, &point);
        let (commitment, witness) = protocol.commit(table, &mut rng).unwrap();
        let (value, proof) = protocol.prove(&commitment, witness, &point, &mut rng).unwrap();
        assert_eq!(value, expected);
        protocol.verify(&commitment, &point, value, &proof).unwrap();
    }
}

fn proofs_do_not_verify_for_another_value<F: RsField, H: Hasher>() {
    let mut rng = Rng::from_seed([3; 32]);
    for parameters in schedules::<F>() {
        let protocol = Protocol::<F, H>::new(parameters.clone()).unwrap();
        let mut instance = prove(&protocol, &mut rng);
        instance.value += F::ONE;
        assert!(matches!(verify(&protocol, &instance), Err(Error::InvalidProof(_))), "{parameters:?}");
    }
}

fn the_statement_is_bound<F: RsField, H: Hasher>() {
    let mut rng = Rng::from_seed([4; 32]);
    let parameters = parameters::<F>(9, &[(2, 1100), (3, 300)], 3);
    let protocol = Protocol::<F, H>::new(parameters.clone()).unwrap();
    let instance = prove(&protocol, &mut rng);
    verify(&protocol, &instance).unwrap();

    let mut other = instance.clone();
    other.point[4] += F::ONE;
    assert!(verify(&protocol, &other).is_err());
    other = instance.clone();
    other.commitment.0[7] ^= 1;
    assert!(verify(&protocol, &other).is_err());

    // Every parameter is bound, also those that leave the layout unchanged.
    // The edited parameters are valid, so it is the proof that is rejected.
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
    for edit in edits {
        let mut changed = parameters.clone();
        edit(&mut changed);
        let changed = Protocol::<F, H>::new(changed).unwrap();
        assert!(matches!(verify(&changed, &instance), Err(Error::InvalidProof(_))));
    }
    // A second proof for the same statement is different: it is randomized.
    assert_ne!(prove(&protocol, &mut rng).proof, instance.proof);
}

/// A proof is a function of the statement and of the prover's randomness,
/// whatever the number of threads.
fn proofs_are_reproducible_from_the_seed_of_the_generator<F: RsField, H: Hasher>() {
    let protocol = Protocol::<F, H>::new(parameters::<F>(9, &[(2, 1100), (3, 300)], 6)).unwrap();
    let prove_with = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| prove(&protocol, &mut Rng::from_seed([7; 32])).proof)
    };
    let proof = prove_with(1);
    assert_eq!(prove_with(2), proof);
    assert_eq!(prove_with(5), proof);
}

fn every_altered_proof_is_rejected<F: RsField, H: Hasher>() {
    let mut rng = Rng::from_seed([5; 32]);
    let schedules =
        [parameters::<F>(6, &[(3, 256)], 3), parameters::<F>(9, &[(2, 1100), (3, 300)], 3), ground::<F>()];
    for parameters in schedules {
        let protocol = Protocol::<F, H>::new(parameters).unwrap();
        let instance = prove(&protocol, &mut rng);
        let bytes = &instance.proof.0;
        // Every byte of a short proof, every 13th of a long one.
        let step = if bytes.len() < 20_000 { 1 } else { 13 };
        for position in (0..bytes.len()).step_by(step) {
            let mut altered = instance.clone();
            altered.proof.0[position] ^= 1 << (position % 8);
            assert!(verify(&protocol, &altered).is_err(), "byte {position} of {}", bytes.len());
        }
        let mut truncated = instance.clone();
        truncated.proof.0.pop();
        assert!(verify(&protocol, &truncated).is_err());
        let mut extended = instance.clone();
        extended.proof.0.push(0);
        assert!(verify(&protocol, &extended).is_err());
        let empty = Instance { proof: Proof(Vec::new()), ..instance.clone() };
        assert!(verify(&protocol, &empty).is_err());
    }
}

fn invalid_inputs_are_reported<F: RsField, H: Hasher>() {
    let mut rng = Rng::from_seed([6; 32]);
    let other_folding = parameters::<F>(6, &[(2, 256)], 0);
    let parameters = parameters::<F>(6, &[(3, 256)], 0);
    let protocol = Protocol::<F, H>::new(parameters.clone()).unwrap();
    assert!(matches!(protocol.commit(rng.fields(63), &mut rng), Err(Error::InvalidInput(_))));
    let (commitment, witness) = protocol.commit(rng.fields(64), &mut rng).unwrap();
    let short: Vec<F> = rng.fields(5);
    assert!(matches!(protocol.prove(&commitment, witness, &short, &mut rng), Err(Error::InvalidInput(_))));
    assert!(protocol.verify(&commitment, &short, F::ZERO, &Proof(Vec::new())).is_err());

    // A witness of another protocol, of another commitment, or for a code of
    // another length.
    let point: Vec<F> = rng.fields(6);
    let mut longer = parameters.clone();
    longer.stages[0].domain_size = 300;
    for other in [other_folding, parameters.clone(), longer] {
        let other = Protocol::<F, H>::new(other).unwrap();
        let (_, foreign) = other.commit(rng.fields(64), &mut rng).unwrap();
        assert!(matches!(
            protocol.prove(&commitment, foreign, &point, &mut rng),
            Err(Error::InvalidInput(_))
        ));
    }
}

/// Parameters that are not valid are rejected without computing with them.
fn adversarial_parameters_are_rejected<F: RsField, H: Hasher>() {
    let parameters = parameters::<F>(6, &[(3, 256)], 0);
    let edits: [fn(&mut Parameters); 9] = [
        |p| p.stages.clear(),
        |p| p.stages[0].folding = usize::MAX,
        |p| p.stages.push(StageParameters::new(usize::MAX, 256, 20)),
        |p| p.num_variables = usize::MAX,
        |p| p.security_bits = 3_000_000_000,
        |p| p.security_bits = 1 << 31,
        |p| p.stages[0].domain_size = usize::MAX,
        |p| p.johnson_slack = f64::NAN,
        // Too few queries for the target.
        |p| p.stages[0].queries -= 1,
    ];
    for edit in edits {
        let mut changed = parameters.clone();
        edit(&mut changed);
        assert!(matches!(Protocol::<F, H>::new(changed), Err(Error::InvalidParameters(_))));
    }
}

macro_rules! tests {
    ($module:ident, $field:ty, $hash:ty, $($test:ident),*) => {
        mod $module {
            use super::*;
            $(
                #[test]
                fn $test() {
                    super::$test::<$field, $hash>();
                }
            )*
        }
    };
}

macro_rules! all_tests {
    ($module:ident, $field:ty, $hash:ty) => {
        tests!(
            $module,
            $field,
            $hash,
            honest_proofs_verify_and_have_the_predicted_size,
            special_tables_and_points_verify,
            proofs_do_not_verify_for_another_value,
            the_statement_is_bound,
            proofs_are_reproducible_from_the_seed_of_the_generator,
            every_altered_proof_is_rejected,
            invalid_inputs_are_reported,
            adversarial_parameters_are_rejected
        );
    };
}

all_tests!(gf2_256_blake3, Gf2_256, Blake3);
all_tests!(gf2_128_blake3, Gf2_128, Blake3);
all_tests!(gf2_256_keccak256, Gf2_256, Keccak256);
all_tests!(gf2_128_keccak256, Gf2_128, Keccak256);

/// A proof is bound to its field and to its hash function.
#[test]
fn proofs_do_not_verify_under_another_hash_function() {
    let mut rng = Rng::from_seed([8; 32]);
    let parameters = parameters::<Gf2_256>(6, &[(3, 256)], 0);
    let blake = Protocol::<Gf2_256, Blake3>::new(parameters.clone()).unwrap();
    let keccak = Protocol::<Gf2_256, Keccak256>::new(parameters).unwrap();
    let instance = prove(&blake, &mut rng);
    verify(&blake, &instance).unwrap();
    assert!(matches!(verify(&keccak, &instance), Err(Error::InvalidProof(_))));
}

/// The smaller field has a larger soundness error: parameters that reach 100
/// bits over the larger field do not over the smaller one.
#[test]
fn the_security_target_depends_on_the_field() {
    let parameters = Parameters::standard(12, 100, Gf2_256::SIZE).unwrap();
    assert!(Protocol::<Gf2_256, Blake3>::new(parameters.clone()).is_ok());
    assert!(matches!(Protocol::<Gf2_128, Blake3>::new(parameters), Err(Error::InvalidParameters(_))));
}

/// Over the smaller field the standard parameters reach 100 bits with proofs
/// of work before the sumcheck challenges, the out-of-domain samples and the
/// combination challenge of the base case.
#[test]
fn proofs_of_work_reach_the_target_over_the_smaller_field() {
    let mut rng = Rng::from_seed([10; 32]);
    let parameters = Parameters::standard(13, 100, Gf2_128::SIZE).unwrap();
    assert!(parameters.stages.iter().all(|stage| stage.sumcheck_pow_bits > 0));
    assert!(parameters.stages[0].ood_pow_bits > 0 && parameters.base_case_pow_bits > 0);
    let analysis = Analysis::new(&parameters, Variant::ZeroKnowledge, Gf2_128::SIZE).unwrap();
    assert!(analysis.soundness.total_bits >= 100.0);

    let protocol = Protocol::<Gf2_128, Blake3>::new(parameters.clone()).unwrap();
    let instance = prove(&protocol, &mut rng);
    verify(&protocol, &instance).unwrap();
    let mut other = instance.clone();
    other.value += Gf2_128::ONE;
    assert!(matches!(verify(&protocol, &other), Err(Error::InvalidProof(_))));

    // Without the proofs of work of the first sumcheck the parameters miss
    // the target.
    let mut weaker = parameters.clone();
    weaker.stages[0].sumcheck_pow_bits = 0;
    assert!(matches!(Protocol::<Gf2_128, Blake3>::new(weaker), Err(Error::InvalidParameters(_))));
}
