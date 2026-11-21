//! The zero-knowledge sumcheck: steps 1 to 3 of a stage.
//!
//! This is the paper's construction "zero-knowledge sumcheck IOR" for the
//! claim of a stage. With `f` and `v` the main and auxiliary messages of the
//! oracle, `c` and `d` their coefficients in the claim, and `U` the mask
//! polynomial of the oracle, the parties run a sumcheck on
//!
//! ```text
//! H(X) = U(X) + (<c(X, ·), f(X, ·)> + <d(X, ·), v(X, ·)>)·α
//! ```
//!
//! for a batching challenge `α`, where `X = (X_1, …, X_k)` are the variables
//! that index the rows. Each round polynomial has degree two. The prover
//! sends its constant and quadratic coefficients, and the verifier recovers
//! the linear one from the current target, which is the sum of the values of
//! the polynomial at zero and one. A proof of work can precede the challenge
//! of every round.
//!
//! The output is a claim on the fold of the oracle at the challenges.

use crate::claim::{Claim, FoldedClaim, ProverClaim, ProverFoldedClaim, VerifierClaim, VerifierFoldedClaim};
use crate::error::Result;
use crate::field::Field;
use crate::hash::Hasher;
use crate::mask;
use crate::polynomial::{eq_table, fold, fold_rows, round_coefficients};
use crate::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use crate::witness::{Message, Row};

/// The value at `challenge` of the polynomial of degree two with the given
/// constant and quadratic coefficients whose values at zero and one sum to
/// `target`.
fn evaluate_round<F: Field>(target: F, constant: F, quadratic: F, challenge: F) -> F {
    let linear = target - F::from_integer(2) * constant - quadratic;
    constant + (linear + quadratic * challenge) * challenge
}

/// Reduce a claim on the message of an oracle to a claim on the row that the
/// oracle folds to. Returns the folded claim, that row, and the challenges.
///
/// A proof of work of `pow_bits` bits precedes every folding challenge.
pub fn prove<F: Field, H: Hasher>(
    transcript: &mut ProverTranscript<H>,
    claim: ProverClaim<F>,
    message: Message<F>,
    pow_bits: u32,
) -> (ProverFoldedClaim<F>, Row<F>, Vec<F>) {
    let Claim { main: mut main_form, aux: mut aux_form, target } = claim;
    let Message { mut main, mut aux, mask, seeds } = message;

    // Step 1: the sum of the mask polynomial over the cube.
    let mut mask_sumcheck = mask.sumcheck();
    let mask_sum = mask_sumcheck.sum();
    transcript.send_field(mask_sum);

    // Step 2: the batching challenge.
    let batching = transcript.challenge_field();

    // Step 3: the rounds, with target `mask_sum + target·batching`.
    let mut target = mask_sum + target * batching;
    let mut challenges = Vec::with_capacity(mask.folding());
    for _ in 0..mask.folding() {
        let (mask_constant, mask_quadratic) = mask_sumcheck.round_coefficients();
        let (main_constant, main_quadratic) = round_coefficients(&main_form, &main);
        let (aux_constant, aux_quadratic) = round_coefficients(&aux_form, &aux);
        let constant = mask_constant + (main_constant + aux_constant) * batching;
        let quadratic = mask_quadratic + (main_quadratic + aux_quadratic) * batching;
        transcript.send_fields(&[constant, quadratic]);

        transcript.prove_work(pow_bits);
        let challenge = transcript.challenge_field();
        target = evaluate_round(target, constant, quadratic, challenge);
        mask_sumcheck.bind(challenge);
        for table in [&mut main_form, &mut main, &mut aux_form, &mut aux] {
            fold(table, challenge);
        }
        challenges.push(challenge);
    }

    for coefficient in main_form.iter_mut().chain(&mut aux_form) {
        *coefficient *= batching;
    }
    // Folding in place keeps the memory of the unfolded tables.
    main_form.shrink_to_fit();
    main.shrink_to_fit();
    let folded_claim =
        FoldedClaim { main: main_form, aux: aux_form, mask: mask::linear_form(&challenges), target };
    let folded_row =
        Row { main, aux, mask: mask.fold(&challenges), seed: fold_rows(&seeds, &eq_table(&challenges)) };
    (folded_claim, folded_row, challenges)
}

/// The verifier's side of [`prove`] for an oracle with folding parameter
/// `folding`. Returns the folded claim and the challenges.
pub fn verify<F: Field, H: Hasher>(
    transcript: &mut VerifierTranscript<H>,
    claim: VerifierClaim<F>,
    folding: usize,
    pow_bits: u32,
) -> Result<(VerifierFoldedClaim<F>, Vec<F>)> {
    let Claim { mut main, aux, target } = claim;
    let mask_sum: F = transcript.read_field()?;
    let batching = transcript.challenge_field();
    let mut target = mask_sum + target * batching;
    let mut challenges = Vec::with_capacity(folding);
    for _ in 0..folding {
        let coefficients = transcript.read_fields(2)?;
        transcript.check_work(pow_bits)?;
        let challenge = transcript.challenge_field();
        target = evaluate_round(target, coefficients[0], coefficients[1], challenge);
        challenges.push(challenge);
    }
    main.fix(&challenges);
    main.scale(batching);
    let aux = fold_rows(&aux, &eq_table(&challenges))
        .into_iter()
        .map(|coefficient| coefficient * batching)
        .collect();
    Ok((FoldedClaim { main, aux, mask: mask::linear_form(&challenges), target }, challenges))
}
