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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;

    type F = crate::field::Gf2_256;
    type H = crate::hash::Blake3;
    use crate::parameters::Stage;
    use crate::polynomial::inner_product;
    use crate::rng::Rng;
    use crate::transcript::NONCE_BYTES;

    fn stage(folding: usize) -> Stage {
        Stage {
            folding,
            main_len: 4,
            aux_len: 3,
            mask_len: 2,
            seed_len: 5,
            pad_len: 0,
            domain_size: 64,
            queries: 4,
        }
    }

    /// A random claim on a random message, off by `error`.
    fn instance(stage: &Stage, error: F, rng: &mut Rng) -> (ProverClaim<F>, VerifierClaim<F>, Message<F>) {
        let rows = stage.rows();
        let message =
            Message::new(stage, rng.fields(rows * stage.main_len), rng.fields(rows * stage.aux_len), rng);
        let point: Vec<F> = rng.fields(stage.folding + 2);
        let aux_form: Vec<F> = rng.fields(rows * stage.aux_len);
        let mut prover = ProverClaim::evaluation(&point, F::ZERO);
        prover.aux = aux_form.clone();
        prover.target =
            inner_product(&prover.main, &message.main) + inner_product(&aux_form, &message.aux) + error;
        let mut verifier = VerifierClaim::evaluation(&point, prover.target);
        verifier.aux = aux_form;
        (prover, verifier, message)
    }

    #[test]
    fn the_output_claim_holds_exactly_when_the_input_claim_does() {
        let mut rng = Rng::from_seed([80; 32]);
        for folding in 1..=5 {
            for error in [F::ZERO, F::ONE] {
                let stage = stage(folding);
                // Every other instance has a proof of work in each round.
                let pow_bits = if folding % 2 == 0 { 6 } else { 0 };
                let (claim, verifier_claim, message) = instance(&stage, error, &mut rng);
                let mut prover = ProverTranscript::<H>::new(b"sumcheck");
                let (folded_claim, row, challenges) = prove(&mut prover, claim, message.clone(), pow_bits);
                let proof = prover.finish();
                let nonces = if pow_bits == 0 { 0 } else { folding };
                assert_eq!(proof.len(), F::BYTES * (1 + 2 * folding) + NONCE_BYTES * nonces);

                // A verifier that expects another proof of work reads another
                // transcript.
                let mut other = VerifierTranscript::<H>::new(b"sumcheck", &proof);
                let read = verify(&mut other, verifier_claim.clone(), folding, 6 - pow_bits);
                assert!(read.is_err() || other.finish().is_err());

                let mut verifier = VerifierTranscript::<H>::new(b"sumcheck", &proof);
                let (verifier_folded, verifier_challenges) =
                    verify(&mut verifier, verifier_claim, folding, pow_bits).unwrap();
                verifier.finish().unwrap();
                assert_eq!(verifier_challenges, challenges);
                assert_eq!(verifier_folded.main.to_dense(), folded_claim.main);
                assert_eq!(verifier_folded.aux, folded_claim.aux);
                assert_eq!(verifier_folded.mask, folded_claim.mask);
                assert_eq!(verifier_folded.target, folded_claim.target);

                // The row is the fold of the message, mask and seeds.
                let weights = eq_table(&challenges);
                assert_eq!(row.main, fold_rows(&message.main, &weights));
                assert_eq!(row.aux, fold_rows(&message.aux, &weights));
                assert_eq!(row.seed, fold_rows(&message.seeds, &weights));
                assert_eq!(row.mask, message.mask.fold(&challenges));
                assert_eq!((row.main.len(), row.aux.len(), row.seed.len()), (4, 3, 5));

                assert_eq!(folded_claim.evaluate(&row) == folded_claim.target, error == F::ZERO);
            }
        }
    }

    #[test]
    fn round_polynomials_sum_to_the_target() {
        // Replay the transcript and compare each round with the definition of H.
        let mut rng = Rng::from_seed([81; 32]);
        let stage = stage(3);
        let (claim, _, message) = instance(&stage, F::ZERO, &mut rng);
        let mut prover = ProverTranscript::<H>::new(b"sumcheck");
        prove(&mut prover, claim.clone(), message.clone(), 0);
        let proof = prover.finish();

        let mut replay = VerifierTranscript::<H>::new(b"sumcheck", &proof);
        let mask_sum: F = replay.read_field().unwrap();
        let batching: F = replay.challenge_field();
        let mask_rows = message.mask.rows();
        // H on a point of the cube extended by the challenges so far: the
        // tables are folded at the challenges and then read at the bits.
        let h = |prefix: &[F], x: F, rest: usize| -> F {
            let mut point = prefix.to_vec();
            point.push(x);
            point.extend((0..stage.folding - prefix.len() - 1).rev().map(|i| F::from_integer(rest >> i)));
            let weights = eq_table(&point);
            let s0: Vec<F> = mask_rows.iter().map(|row| row[0]).collect();
            let s1: Vec<F> = mask_rows.iter().map(|row| row[1]).collect();
            let mask = inner_product(&s0, &weights) + inner_product(&s1, &weights) * point.iter().sum::<F>();
            let product = |form: &[F], table: &[F]| {
                inner_product(&fold_rows(form, &weights), &fold_rows(table, &weights))
            };
            mask + (product(&claim.main, &message.main) + product(&claim.aux, &message.aux)) * batching
        };
        let mut target = mask_sum + claim.target * batching;
        let mut prefix = Vec::new();
        for round in 0..stage.folding {
            let coefficients: Vec<F> = replay.read_fields(2).unwrap();
            let sum =
                |x: F| -> F { (0..1 << (stage.folding - round - 1)).map(|rest| h(&prefix, x, rest)).sum() };
            assert_eq!(sum(F::ZERO) + sum(F::ONE), target);
            assert_eq!(sum(F::ZERO), coefficients[0]);
            let challenge: F = replay.challenge_field();
            target = evaluate_round(target, coefficients[0], coefficients[1], challenge);
            assert_eq!(sum(challenge), target);
            prefix.push(challenge);
        }
    }
}
