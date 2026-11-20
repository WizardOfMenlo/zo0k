use zo0k::claim::ProverClaim;
use zo0k::code_switch::CodeSwitch;
use zo0k::hash::DIGEST_BYTES;
use zo0k::oracle::Oracle;
use zo0k::parameters::{Layout, Parameters, Stage, StageParameters};
use zo0k::polynomial::inner_product;
use zo0k::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use zo0k::witness::{Message, Row};
use zo0k::{Commitment, Field, Hasher, Proof, Rng, RsField, base_case, sumcheck};

use crate::primitives::missing_siblings;

pub fn statement<F: Field, H: Hasher>(
    parameters: &Parameters,
    commitment: &Commitment,
    point: &[F],
    value: F,
) -> Vec<u8> {
    let mut bytes = [F::NAME.as_bytes(), b"/", H::NAME.as_bytes(), b"/"].concat();
    bytes.extend(parameters.to_bytes());
    bytes.extend(commitment.0);
    let mut encoding = vec![0u8; F::BYTES];
    for coordinate in point.iter().chain([&value]) {
        coordinate.write_bytes(&mut encoding);
        bytes.extend_from_slice(&encoding);
    }
    assert!(bytes.len() >= DIGEST_BYTES + F::BYTES);
    bytes
}

pub struct Composed<F> {
    pub commitment: Commitment,
    pub value: F,
    pub proof: Proof,
}

struct StageWitness<F> {
    message: Message<F>,
    oracle: Oracle<F>,
    mask: Option<Row<F>>,
}

fn commit_stage<F: RsField, H: Hasher>(
    stage: &Stage,
    message: Message<F>,
    is_last: bool,
    rng: &mut Rng,
) -> StageWitness<F> {
    let mask = is_last.then(|| Row::sample(stage, rng));
    let rows = message.encode(stage, mask.as_ref());
    assert_eq!(rows.len(), stage.rows() + usize::from(is_last));
    let witness = StageWitness { oracle: Oracle::commit::<H>(rows), message, mask };
    assert_eq!(witness.oracle.shape().1, stage.domain_size);
    witness
}

pub fn compose<F: RsField, H: Hasher>(
    parameters: &Parameters,
    layout: &Layout,
    table: Vec<F>,
    point: &[F],
    error: F,
    seeds: (&[u8; 32], &[[u8; 32]]),
) -> Composed<F> {
    let stages = &layout.stages;
    assert_eq!(table.len(), 1 << parameters.num_variables);
    assert_eq!(seeds.1.len() + 1, stages.len());
    let mut rng = Rng::from_seed(*seeds.0);
    let message = Message::new(&stages[0], table, Vec::new(), &mut rng);
    let mut witness = commit_stage::<F, H>(&stages[0], message, stages.len() == 1, &mut rng);
    let commitment = Commitment(witness.oracle.root());
    let mut claim = ProverClaim::evaluation(point, F::ZERO);
    claim.target = inner_product(&claim.main, &witness.message.main) + error;
    let value = claim.target;
    let mut transcript =
        ProverTranscript::<H>::new(&statement::<F, H>(parameters, &commitment, point, value));
    for (index, stage) in stages.iter().enumerate() {
        let StageParameters { sumcheck_pow_bits, ood_pow_bits, .. } = parameters.stages[index];
        let StageWitness { message, oracle, mask } = witness;
        let (folded, row, challenges) = sumcheck::prove(&mut transcript, claim, message, sumcheck_pow_bits);
        let Some(next) = stages.get(index + 1) else {
            let mask = mask.expect("the last oracle carries a mask");
            base_case::prove(&mut transcript, parameters, stage, &folded, &row, &mask, &oracle);
            break;
        };
        let next_is_last = index + 2 == stages.len();
        let switch = CodeSwitch {
            source: stage,
            next,
            next_is_last,
            ood_pow_bits,
            query_pow_bits: parameters.pow_bits,
        };
        let carried = [&row.aux[..], &row.mask, &row.seed].concat();
        let main = row.main.clone();
        let seed = seeds.1[index];
        let (next_claim, _) =
            switch.prove(&mut transcript, folded, row, &oracle, &challenges, &mut Rng::from_seed(seed));
        let mut replay = Rng::from_seed(seed);
        let pad: Vec<F> = replay.fields(stage.pad_len);
        let message = Message::new(next, main, [&carried[..], &pad].concat(), &mut replay);
        witness = commit_stage::<F, H>(next, message, next_is_last, &mut replay);
        claim = next_claim;
    }
    Composed { commitment, value, proof: Proof(transcript.finish()) }
}

pub struct Framing {
    pub opening_bytes: usize,
    pub positions: Vec<Vec<usize>>,
}

fn read_opening<H: Hasher, F: Field>(
    transcript: &mut VerifierTranscript<H>,
    stage: &Stage,
    mut positions: Vec<usize>,
    framing: &mut Framing,
) -> zo0k::Result<()> {
    assert!(positions.iter().all(|&position| position < stage.domain_size));
    positions.sort_unstable();
    positions.dedup();
    for _ in 0..positions.len() {
        transcript.read_fields::<F>(stage.rows())?;
    }
    let digests = missing_siblings(stage.domain_size, &positions);
    for _ in 0..digests {
        transcript.read_digest()?;
    }
    framing.opening_bytes += positions.len() * stage.rows() * F::BYTES + digests * DIGEST_BYTES;
    framing.positions.push(positions);
    Ok(())
}

pub fn frame<F: RsField, H: Hasher>(
    parameters: &Parameters,
    layout: &Layout,
    statement: &[u8],
    proof: &[u8],
) -> zo0k::Result<Framing> {
    let mut transcript = VerifierTranscript::<H>::new(statement, proof);
    let mut framing = Framing { opening_bytes: 0, positions: Vec::new() };
    let count = layout.stages.len();
    for (index, stage) in layout.stages.iter().enumerate() {
        let given = &parameters.stages[index];
        transcript.read_field::<F>()?;
        let _: F = transcript.challenge_field();
        for _ in 0..stage.folding {
            transcript.read_fields::<F>(2)?;
            transcript.check_work(given.sumcheck_pow_bits)?;
            let _: F = transcript.challenge_field();
        }
        if index + 1 < count {
            transcript.read_digest()?;
            transcript.check_work(given.ood_pow_bits)?;
            let _: F = transcript.challenge_field();
            let _: Vec<F> = transcript.challenge_nonzero_vector(stage.pad_len);
            transcript.read_field::<F>()?;
            transcript.check_work(parameters.pow_bits)?;
            let positions = transcript.challenge_positions(stage.domain_size, stage.queries);
            let _: F = transcript.challenge_field();
            read_opening::<H, F>(&mut transcript, stage, positions, &mut framing)?;
            continue;
        }
        transcript.read_field::<F>()?;
        transcript.check_work(parameters.base_case_pow_bits)?;
        let _: F = transcript.challenge_field();
        transcript.read_fields::<F>(stage.dimension())?;
        transcript.check_work(parameters.pow_bits)?;
        let positions = transcript.challenge_positions(stage.domain_size, stage.queries);
        read_opening::<H, F>(&mut transcript, stage, positions, &mut framing)?;
    }
    transcript.finish()?;
    assert_eq!(framing.positions.len(), count);
    Ok(framing)
}
