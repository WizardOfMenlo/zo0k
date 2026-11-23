use crate::claim::{FoldedClaim, ProverFoldedClaim, VerifierFoldedClaim};
use crate::error::{Error, Result};
use crate::hash::{Digest, Hasher};
use crate::oracle::{Opening, Oracle};
use crate::parameters::{Parameters, Stage};
use crate::polynomial::{eq_table, inner_product};
use crate::reed_solomon::RsField;
use crate::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use crate::witness::Row;

pub fn prove<F: RsField, H: Hasher>(
    transcript: &mut ProverTranscript<H>,
    parameters: &Parameters,
    stage: &Stage,
    claim: &ProverFoldedClaim<F>,
    row: &Row<F>,
    mask: &Row<F>,
    oracle: &Oracle<F>,
) {
    transcript.send_field(claim.evaluate(mask));
    transcript.prove_work(parameters.base_case_pow_bits);
    let combination = transcript.challenge_field();
    transcript.send_fields(&mask.add_scaled(row, combination).to_fields());

    transcript.prove_work(parameters.pow_bits);
    let positions = transcript.challenge_positions(stage.domain_size, stage.queries);
    oracle.open(transcript, &positions, stage.rows());
}

pub fn verify<F: RsField, H: Hasher>(
    transcript: &mut VerifierTranscript<H>,
    parameters: &Parameters,
    stage: &Stage,
    claim: VerifierFoldedClaim<F>,
    root: &Digest,
    fold_challenges: &[F],
) -> Result<()> {
    let mask_target: F = transcript.read_field()?;
    transcript.check_work(parameters.base_case_pow_bits)?;
    let combination = transcript.challenge_field();
    let disclosed = Row::from_fields(stage, &transcript.read_fields(stage.dimension())?);

    let FoldedClaim { main, aux, mask, target } = claim;
    let evaluated = main.evaluate_rs(&disclosed.main)
        + inner_product(&aux, &disclosed.aux)
        + inner_product(&mask, &disclosed.mask);
    if evaluated != mask_target + target * combination {
        return Err(Error::InvalidProof("the disclosed row does not satisfy the claim"));
    }

    transcript.check_work(parameters.pow_bits)?;
    let positions = transcript.challenge_positions(stage.domain_size, stage.queries);
    let polynomial = disclosed.polynomial();
    let weights = eq_table(fold_challenges);
    let mask_symbol = |position: usize, column: &[F]| {
        let folded = inner_product(column, &weights);
        vec![F::evaluate_at(&polynomial, position) - folded * combination]
    };
    Opening::read(transcript, root, stage.domain_size, &positions, stage.rows(), mask_symbol)?;
    Ok(())
}
