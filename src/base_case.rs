//! The base case: step 5 of the last stage.
//!
//! This is the paper's construction "non-succinct zero-knowledge protocol",
//! applied to the row `w = (f, v, s, r)` that the last oracle folds to. The
//! prover has committed, together with that oracle, to the codeword `g` of a
//! uniform row `w_g` of the same shape. It sends the value of the claim's
//! linear form on `w_g`, receives a combination challenge `γ`, and discloses
//! the row `w* = w_g + w·γ`. The verifier checks that
//!
//! * the linear form takes the value `target_g + target·γ` on `w*`;
//! * the codeword of `w*` equals `g + (folded oracle)·γ` at uniform positions.
//!
//! The second check determines the symbol of `g` at a queried position from
//! the disclosed row and the column of the oracle. The prover therefore does
//! not send that symbol: the verifier computes it, and the Merkle opening of
//! the column, whose last entry it is, authenticates it.
//!
//! A proof of work can precede the combination challenge, and another the
//! positions.

use crate::claim::{FoldedClaim, ProverFoldedClaim, VerifierFoldedClaim};
use crate::error::{Error, Result};
use crate::hash::{Digest, Hasher};
use crate::oracle::{Opening, Oracle};
use crate::parameters::{Parameters, Stage};
use crate::polynomial::{eq_table, inner_product};
use crate::reed_solomon::RsField;
use crate::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use crate::witness::Row;

/// Prove a claim on the row `row` that the last oracle folds to, where `mask`
/// is the row whose codeword was committed as the last row of `oracle`.
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

/// The verifier's side of [`prove`], for a last oracle with commitment `root`
/// that was folded at `fold_challenges`.
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
    let claim = FoldedClaim { main: main.to_dense(), aux, mask, target };
    if claim.evaluate(&disclosed) != mask_target + target * combination {
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
