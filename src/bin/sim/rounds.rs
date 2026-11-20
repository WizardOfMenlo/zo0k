use crate::algebra::bits;
use crate::fields::{nonzero, uniform, uniforms};
use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure};
use zo0k::claim::{
    FoldedClaim, LinearForm, ProverClaim, ProverFoldedClaim, VerifierClaim, VerifierFoldedClaim,
};
use zo0k::code_switch::CodeSwitch;
use zo0k::field::BinaryField;
use zo0k::oracle::Oracle;
use zo0k::parameters::{Layout, MASK_LEN, Parameters, Stage, StageParameters, Variant};
use zo0k::polynomial::{add_powers, eq_table, evaluate, fold_rows, inner_product};
use zo0k::transcript::{Challenges, NONCE_BYTES, ProverTranscript, VerifierTranscript};
use zo0k::witness::{Message, Row};
use zo0k::{Blake3, Error, Gf2_128, Gf2_256, Hasher, Keccak256, Rng, RsField, sumcheck};

pub fn random_stage(prng: &mut Prng, folding: usize) -> Stage {
    assert!(folding >= 1);
    let mut stage = Stage {
        folding,
        main_len: 1 << prng.range(0, 3),
        aux_len: prng.range(0, 4),
        mask_len: MASK_LEN,
        seed_len: prng.range(1, 6),
        pad_len: 0,
        domain_size: 0,
        queries: prng.range(1, 6),
    };
    stage.domain_size = stage.dimension() + prng.range(1, 40);
    assert!(stage.domain_size > stage.dimension());
    stage
}

struct SumcheckInstance<F> {
    prover: ProverClaim<F>,
    verifier: VerifierClaim<F>,
    message: Message<F>,
}

fn sumcheck_instance<F: RsField>(stage: &Stage, error: F, prng: &mut Prng) -> SumcheckInstance<F> {
    assert!(stage.main_len.is_power_of_two());
    let rows = stage.rows();
    let mut rng = Rng::from_seed(prng.seed_bytes());
    let main = uniforms(prng, rows * stage.main_len);
    let aux = uniforms(prng, rows * stage.aux_len);
    let message = Message::new(stage, main, aux, &mut rng);
    let point: Vec<F> = uniforms(prng, stage.folding + stage.main_len.trailing_zeros() as usize);
    let mut prover = ProverClaim::evaluation(&point, F::ZERO);
    prover.aux = uniforms(prng, rows * stage.aux_len);
    prover.target =
        inner_product(&prover.main, &message.main) + inner_product(&prover.aux, &message.aux) + error;
    let verifier = VerifierClaim {
        main: LinearForm::evaluation(&point),
        aux: prover.aux.clone(),
        target: prover.target,
    };
    assert_eq!(prover.main.len(), rows * stage.main_len);
    SumcheckInstance { prover, verifier, message }
}

pub fn sumchecks<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let folding = prng.range(1, 5);
    let stage = random_stage(prng, folding);
    let error = if prng.chance(50) { F::ZERO } else { nonzero(prng) };
    let pow_bits = if prng.chance(50) { 0 } else { prng.range(1, 6) as u32 };
    let SumcheckInstance { prover, verifier, message } = sumcheck_instance::<F>(&stage, error, prng);
    let mut transcript = ProverTranscript::<H>::new(b"sim-sumcheck");
    let (folded, row, challenges) = sumcheck::prove(&mut transcript, prover, message.clone(), pow_bits);
    let proof = transcript.finish();
    let nonces = if pow_bits == 0 { 0 } else { folding };
    ensure(proof.len() == F::BYTES * (1 + 2 * folding) + NONCE_BYTES * nonces, || {
        format!("a sumcheck of {folding} rounds has {} bytes", proof.len())
    })?;
    let mut harder = VerifierTranscript::<H>::new(b"sim-sumcheck", &proof);
    let other_bits = if pow_bits == 0 { 6 } else { 0 };
    let read = sumcheck::verify(&mut harder, verifier.clone(), folding, other_bits);
    ensure(read.is_err() || harder.finish().is_err(), || {
        "a sumcheck verifies with other proofs of work".into()
    })?;
    let mut transcript = VerifierTranscript::<H>::new(b"sim-sumcheck", &proof);
    let (verified, verifier_challenges) =
        sumcheck::verify(&mut transcript, verifier, folding, pow_bits).map_err(|e| e.to_string())?;
    transcript.finish().map_err(|e| e.to_string())?;
    ensure(verifier_challenges == challenges, || "the sumcheck challenges differ".into())?;
    ensure(verified.main.to_dense() == folded.main && verified.aux == folded.aux, || {
        "the folded forms differ".into()
    })?;
    ensure(verified.mask == folded.mask && verified.target == folded.target, || {
        "the folded targets differ".into()
    })?;
    let weights = eq_table(&challenges);
    ensure(
        row.main == fold_rows(&message.main, &weights) && row.aux == fold_rows(&message.aux, &weights),
        || "the folded row".into(),
    )?;
    ensure(
        row.seed == fold_rows(&message.seeds, &weights) && row.mask == message.mask.fold(&challenges),
        || "the folded seeds or mask".into(),
    )?;
    ensure(
        (row.main.len(), row.aux.len(), row.seed.len()) == (stage.main_len, stage.aux_len, stage.seed_len),
        || "folded lengths".into(),
    )?;
    ensure((folded.evaluate(&row) == folded.target) == (error == F::ZERO), || {
        format!("the output claim holds {} with error {error:?}", folded.evaluate(&row) == folded.target)
    })
}

pub fn sumcheck_rounds<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let folding = prng.range(1, 4);
    let stage = random_stage(prng, folding);
    let SumcheckInstance { prover: claim, message, .. } = sumcheck_instance::<F>(&stage, F::ZERO, prng);
    let mut transcript = ProverTranscript::<H>::new(b"sim-rounds");
    sumcheck::prove(&mut transcript, claim.clone(), message.clone(), 0);
    let proof = transcript.finish();
    let mut replay = VerifierTranscript::<H>::new(b"sim-rounds", &proof);
    let mask_sum: F = replay.read_field().map_err(|e| e.to_string())?;
    let batching: F = replay.challenge_field();
    let mask_rows = message.mask.rows();
    let h = |prefix: &[F], x: F, rest: usize| -> F {
        let point = [prefix, &[x], &bits::<F>(rest, folding - prefix.len() - 1)].concat();
        let weights = eq_table(&point);
        let s0: Vec<F> = mask_rows.iter().map(|row| row[0]).collect();
        let s1: Vec<F> = mask_rows.iter().map(|row| row[1]).collect();
        let mask = inner_product(&s0, &weights) + inner_product(&s1, &weights) * point.iter().sum::<F>();
        let product =
            |form: &[F], table: &[F]| inner_product(&fold_rows(form, &weights), &fold_rows(table, &weights));
        mask + (product(&claim.main, &message.main) + product(&claim.aux, &message.aux)) * batching
    };
    let mut target = mask_sum + claim.target * batching;
    let mut prefix = Vec::new();
    for round in 0..folding {
        let coefficients: Vec<F> = replay.read_fields(2).map_err(|e| e.to_string())?;
        let sum = |x: F| -> F { (0..1 << (folding - round - 1)).map(|rest| h(&prefix, x, rest)).sum() };
        ensure(sum(F::ZERO) + sum(F::ONE) == target, || format!("round {round} does not sum to the target"))?;
        ensure(sum(F::ZERO) == coefficients[0], || format!("round {round} has a wrong constant"))?;
        let challenge: F = replay.challenge_field();
        let linear = target - coefficients[0] - coefficients[0] - coefficients[1];
        target = coefficients[0] + (linear + coefficients[1] * challenge) * challenge;
        ensure(sum(challenge) == target, || format!("round {round} is not the partial sum"))?;
        prefix.push(challenge);
    }
    Ok(())
}

pub fn witnesses<F: RsField>(prng: &mut Prng) -> Result<(), String> {
    let folding = prng.range(1, 3);
    let stage = random_stage(prng, folding);
    let mut rng = Rng::from_seed(prng.seed_bytes());
    let row: Row<F> = Row::sample(&stage, &mut rng);
    ensure(Row::from_fields(&stage, &row.to_fields()) == row, || "a row does not round trip".into())?;
    let polynomial = row.polynomial();
    ensure(polynomial.len() == stage.dimension(), || "the row polynomial has the wrong degree".into())?;
    ensure(
        polynomial[..stage.seed_len] == row.seed
            && polynomial[stage.mask_degree()..stage.main_degree()] == row.mask,
        || "row layout of seed and mask".into(),
    )?;
    ensure(
        polynomial[stage.main_degree()..stage.aux_degree()] == row.main
            && polynomial[stage.aux_degree()..] == row.aux,
        || "row layout of main and aux".into(),
    )?;
    let other = Row::sample(&stage, &mut rng);
    let scale: F = uniform(prng);
    let sum: Vec<F> = row.to_fields().iter().zip(other.to_fields()).map(|(&x, y)| x + y * scale).collect();
    ensure(row.add_scaled(&other, scale).to_fields() == sum, || "add_scaled".into())?;
    let rows = stage.rows();
    let message: Message<F> = Message::new(
        &stage,
        uniforms(prng, rows * stage.main_len),
        uniforms(prng, rows * stage.aux_len),
        &mut rng,
    );
    let extra = Row::sample(&stage, &mut rng);
    let codewords = message.encode(&stage, Some(&extra));
    ensure(codewords.len() == rows + 1, || "an extra row is not encoded".into())?;
    ensure(codewords[rows] == F::encode(&extra.polynomial(), stage.domain_size), || {
        "the extra row encoding".into()
    })?;
    ensure(message.encode(&stage, None).len() == rows, || "rows without an extra row".into())?;
    let challenges: Vec<F> = uniforms(prng, folding);
    let weights = eq_table(&challenges);
    let folded = Row {
        main: fold_rows(&message.main, &weights),
        aux: fold_rows(&message.aux, &weights),
        mask: message.mask.fold(&challenges),
        seed: fold_rows(&message.seeds, &weights),
    };
    let expected = F::encode(&folded.polynomial(), stage.domain_size);
    for position in 0..stage.domain_size {
        let column: Vec<F> = codewords[..rows].iter().map(|codeword| codeword[position]).collect();
        ensure(fold_rows(&column, &weights) == [expected[position]], || {
            format!("folding does not commute with encoding at {position}")
        })?;
    }
    Ok(())
}

const OOD_POW_BITS: u32 = 4;
const QUERY_POW_BITS: u32 = 3;

fn switch_layout(prng: &mut Prng) -> Layout {
    let (first, second) = (prng.range(1, 3), prng.range(1, 3));
    let num_variables = first + second + prng.range(0, 3);
    let queries = [prng.range(1, 12), prng.range(1, 12)];
    let main = [1usize << (num_variables - first), 1 << (num_variables - first - second)];
    let mut parameters = Parameters {
        num_variables,
        stages: vec![
            StageParameters::new(first, 2 * (main[0] + queries[0] + 8) + prng.range(0, 64), queries[0]),
            StageParameters::new(second, 2 * (main[1] + queries[1] + 32) + prng.range(0, 64), queries[1]),
        ],
        security_bits: 8,
        pow_bits: QUERY_POW_BITS,
        base_case_pow_bits: 0,
        johnson_slack: 0.02,
    };
    parameters.stages[0].ood_pow_bits = OOD_POW_BITS;
    let layout = Layout::new(&parameters, Variant::ZeroKnowledge).expect("the switch layout is valid");
    assert_eq!(layout.stages.len(), 2);
    layout
}

pub struct SwitchInstance<F> {
    pub oracle: Oracle<F>,
    pub fold_challenges: Vec<F>,
    pub row: Row<F>,
    pub claim: ProverFoldedClaim<F>,
    pub verifier_claim: VerifierFoldedClaim<F>,
}

fn switch_instance<F: RsField, H: Hasher>(source: &Stage, error: F, prng: &mut Prng) -> SwitchInstance<F> {
    let mut rng = Rng::from_seed(prng.seed_bytes());
    let message = Message::new(source, uniforms(prng, source.rows() * source.main_len), vec![], &mut rng);
    let oracle = Oracle::commit::<H>(message.encode(source, None));
    let fold_challenges: Vec<F> = uniforms(prng, source.folding);
    let weights = eq_table(&fold_challenges);
    let row = Row {
        main: fold_rows(&message.main, &weights),
        aux: vec![],
        mask: message.mask.fold(&fold_challenges),
        seed: fold_rows(&message.seeds, &weights),
    };
    let point: Vec<F> = uniforms(prng, source.main_len.trailing_zeros() as usize);
    let mask = [uniform(prng), uniform(prng)];
    let mut claim = FoldedClaim { main: eq_table(&point), aux: vec![], mask, target: F::ZERO };
    claim.target = claim.evaluate(&row) + error;
    let verifier_claim =
        FoldedClaim { main: LinearForm::evaluation(&point), aux: vec![], mask, target: claim.target };
    assert_eq!(row.main.len(), source.main_len);
    SwitchInstance { oracle, fold_challenges, row, claim, verifier_claim }
}

struct SwitchReplay<F> {
    x: F,
    beta: Vec<F>,
    answer: F,
    positions: Vec<usize>,
    rho: F,
}

fn replay_switch<F: RsField, H: Hasher>(
    proof: &[u8],
    source: &Stage,
) -> zo0k::Result<(SwitchReplay<F>, zo0k::Digest)> {
    let mut replay = VerifierTranscript::<H>::new(b"sim-switch", proof);
    let root = replay.read_digest()?;
    replay.check_work(OOD_POW_BITS)?;
    let x: F = replay.challenge_field();
    let beta: Vec<F> = replay.challenge_nonzero_vector(source.pad_len);
    let answer: F = replay.read_field()?;
    replay.check_work(QUERY_POW_BITS)?;
    let positions = replay.challenge_positions(source.domain_size, source.queries);
    let rho: F = replay.challenge_field();
    Ok((SwitchReplay { x, beta, answer, positions, rho }, root))
}

fn expected_switch_claim<F: RsField>(
    source: &Stage,
    instance: &SwitchInstance<F>,
    replay: &SwitchReplay<F>,
) -> (Vec<F>, F) {
    let SwitchInstance { row, claim, .. } = instance;
    let zeros = vec![F::ZERO; source.seed_len + source.pad_len];
    let mut coefficients = [&claim.main[..], &claim.aux, &claim.mask, &zeros].concat();
    let mut target = claim.target + replay.rho * replay.answer;
    let pad_at = coefficients.len() - source.pad_len;
    add_powers(&mut coefficients[..pad_at], replay.x, replay.rho);
    for (entry, &b) in coefficients[pad_at..].iter_mut().zip(&replay.beta) {
        *entry += replay.rho * b;
    }
    let folded_codeword = F::encode(&row.polynomial(), source.domain_size);
    let mut coefficient = replay.rho;
    for &position in &replay.positions {
        coefficient *= replay.rho;
        target += coefficient * folded_codeword[position];
        let mut generator = vec![F::ZERO; source.dimension()];
        add_powers(&mut generator, F::point(position), coefficient);
        let (r, rest) = generator.split_at(source.seed_len);
        let (s, rest) = rest.split_at(MASK_LEN);
        let (f, v) = rest.split_at(source.main_len);
        for (entry, &g) in coefficients.iter_mut().zip(f.iter().chain(v).chain(s).chain(r)) {
            *entry += g;
        }
    }
    assert_eq!(
        coefficients.len(),
        claim.main.len() + source.aux_len + MASK_LEN + source.seed_len + source.pad_len
    );
    (coefficients, target)
}

pub fn code_switches<F: RsField, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let layout = switch_layout(prng);
    let (source, next) = (&layout.stages[0], &layout.stages[1]);
    let switch = CodeSwitch {
        source,
        next,
        next_is_last: true,
        ood_pow_bits: OOD_POW_BITS,
        query_pow_bits: QUERY_POW_BITS,
    };
    let error = if prng.chance(50) { F::ZERO } else { nonzero(prng) };
    let instance = switch_instance::<F, H>(source, error, prng);
    let switch_seed = prng.seed_bytes();
    let mut prover = ProverTranscript::<H>::new(b"sim-switch");
    let (next_claim, _) = switch.prove(
        &mut prover,
        instance.claim.clone(),
        instance.row.clone(),
        &instance.oracle,
        &instance.fold_challenges,
        &mut Rng::from_seed(switch_seed),
    );
    let proof = prover.finish();
    let pad: Vec<F> = Rng::from_seed(switch_seed).fields(source.pad_len);
    let row = &instance.row;
    let carried = [&row.aux[..], &row.mask, &row.seed].concat();
    let (replay, next_root) = replay_switch::<F, H>(&proof, source).map_err(|e| e.to_string())?;
    let expected_answer =
        evaluate(&[&row.main[..], &carried].concat(), replay.x) + inner_product(&replay.beta, &pad);
    ensure(replay.answer == expected_answer, || {
        "the out-of-domain answer is not the evaluation of the next message".into()
    })?;
    let (coefficients, target) = expected_switch_claim(source, &instance, &replay);
    ensure([&next_claim.main[..], &next_claim.aux].concat() == coefficients, || {
        "the new claim is not the batch of the stated equations".into()
    })?;
    ensure(next_claim.target == target, || "the new target is not the batch of the stated values".into())?;
    let aux = [&carried[..], &pad].concat();
    let value = inner_product(&next_claim.main, &row.main) + inner_product(&next_claim.aux, &aux);
    ensure((value == next_claim.target) == (error == F::ZERO), || {
        format!("the new claim holds {} for error {error:?}", value == next_claim.target)
    })?;
    let verify = |proof: &[u8]| {
        let mut verifier = VerifierTranscript::<H>::new(b"sim-switch", proof);
        let result = switch.verify(
            &mut verifier,
            instance.verifier_claim.clone(),
            &instance.oracle.root(),
            &instance.fold_challenges,
        )?;
        verifier.finish()?;
        Ok::<_, Error>(result)
    };
    let (verified, verified_root) = verify(&proof).map_err(|e| format!("an honest switch fails: {e}"))?;
    ensure(verified_root == next_root, || "the verifier reads another root".into())?;
    ensure(verified.main.to_dense() == next_claim.main && verified.aux == next_claim.aux, || {
        "the verifier derives another claim".into()
    })?;
    ensure(verified.target == next_claim.target, || "the verifier derives another target".into())?;
    let opening_at = 32 + 8 + F::BYTES + 8;
    let flips: Vec<usize> = (0..48).map(|_| prng.range(opening_at, proof.len() - 1)).collect();
    for byte in flips {
        let mut altered = proof.clone();
        altered[byte] ^= 1 << prng.below(8);
        ensure(matches!(verify(&altered), Err(Error::InvalidProof(_))), || {
            format!("a switch altered at byte {byte} is accepted")
        })?;
    }
    ensure(verify(&proof[..proof.len() - 1]).is_err(), || "a truncated switch is accepted".into())
}

fn per_types<F: RsField + BinaryField, H: Hasher>(ctx: &mut Ctx) -> Result<(), Failure> {
    ctx.check("sumcheck_output_claim_iff_input_claim", sumchecks::<F, H>)?;
    ctx.check("sumcheck_round_polynomials", sumcheck_rounds::<F, H>)?;
    ctx.check("witness_rows_and_encoding", witnesses::<F>)?;
    ctx.check("code_switch_batches_equations", code_switches::<F, H>)?;
    Ok(())
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    match ctx.prng.below(4) {
        0 => per_types::<Gf2_256, Blake3>(ctx),
        1 => per_types::<Gf2_128, Blake3>(ctx),
        2 => per_types::<Gf2_256, Keccak256>(ctx),
        _ => per_types::<Gf2_128, Keccak256>(ctx),
    }
}
