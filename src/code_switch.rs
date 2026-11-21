//! The zero-knowledge code switch: step 4 of every stage but the last.
//!
//! This is the paper's construction "code-switching IOR". The input is a
//! claim on the row `(f, v, s, r)` that the oracle of a stage folds to: its
//! main block, auxiliary block, mask and seed. The prover sends the oracle of
//! the next stage, whose main message is `f` and whose auxiliary message is
//! `(v, s, r, pad)` for a fresh uniform pad. The verifier then
//!
//! * samples an out-of-domain point `x` and a nonzero vector `β`, to which
//!   the prover answers with `Σ_j w_j·x^j + <β, pad>` for `w = (f, v, s, r)`;
//! * queries the folded oracle at uniform positions. The symbol at a position
//!   with domain point `z` is the value at `z` of the polynomial with
//!   coefficients `(r, s, f, v)`, a linear function of the next message.
//!
//! The claim on the next message is the combination of the input claim, the
//! out-of-domain equation and the query equations with the coefficients
//! `1, ρ, ρ², …` for a batching challenge `ρ`.
//!
//! A proof of work can precede the out-of-domain sample, and another the
//! queries.

use crate::claim::{
    Claim, FoldedClaim, MainCoefficients, PowerRow, ProverClaim, ProverFoldedClaim, VerifierClaim,
    VerifierFoldedClaim,
};
use crate::error::Result;
use crate::hash::{Digest, Hasher};
use crate::oracle::{Opening, Oracle};
use crate::parameters::{MASK_LEN, Stage};
use crate::polynomial::{add_powers, eq_table, evaluate, inner_product};
use crate::reed_solomon::RsField;
use crate::rng::Rng;
use crate::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use crate::witness::{Message, Row, Witness};

/// The public description of a code switch.
pub struct CodeSwitch<'a> {
    /// The stage whose folded oracle is re-encoded.
    pub source: &'a Stage,
    /// The stage of the new oracle.
    pub next: &'a Stage,
    /// Whether `next` is the last stage, whose oracle carries the mask of the
    /// base case.
    pub next_is_last: bool,
    /// Bits of proof of work before the out-of-domain sample.
    pub ood_pow_bits: u32,
    /// Bits of proof of work before the queries.
    pub query_pow_bits: u32,
}

/// The verifier's messages of a code switch.
struct SwitchChallenges<F> {
    ood_point: F,
    pad_coefficients: Vec<F>,
    positions: Vec<usize>,
    batching: F,
}

impl CodeSwitch<'_> {
    /// Reduce a claim on the folded row of the source oracle to a claim on
    /// the message of a new oracle. Returns the claim and the new oracle.
    ///
    /// `fold_challenges` are the challenges at which `oracle` was folded.
    pub fn prove<F: RsField, H: Hasher>(
        &self,
        transcript: &mut ProverTranscript<H>,
        claim: ProverFoldedClaim<F>,
        row: Row<F>,
        oracle: &Oracle<F>,
        fold_challenges: &[F],
        rng: &mut Rng,
    ) -> (ProverClaim<F>, Witness<F>) {
        // The new oracle.
        let pad = rng.fields(self.source.pad_len);
        let carried = [&row.aux[..], &row.mask, &row.seed].concat();
        let message = Message::new(self.next, row.main, [&carried[..], &pad].concat(), rng);
        let witness = Witness::commit::<H>(self.next, message, self.next_is_last, rng);
        transcript.send_digest(&witness.oracle.root());

        // The out-of-domain sample and its answer.
        transcript.prove_work(self.ood_pow_bits);
        let ood_point = transcript.challenge_field();
        let pad_coefficients = transcript.challenge_nonzero_vector(self.source.pad_len);
        let ood_answer = evaluate(&witness.message.main, ood_point)
            + ood_point.pow(self.source.main_len as u64) * evaluate(&carried, ood_point)
            + inner_product(&pad_coefficients, &pad);
        transcript.send_field(ood_answer);

        // The queries to the source oracle, and the batching challenge.
        transcript.prove_work(self.query_pow_bits);
        let positions = transcript.challenge_positions(self.source.domain_size, self.source.queries);
        let batching = transcript.challenge_field();
        let opening = oracle.open(transcript, &positions, self.source.rows());

        let challenges = SwitchChallenges { ood_point, pad_coefficients, positions, batching };
        (self.next_claim(claim, &challenges, ood_answer, &opening, fold_challenges), witness)
    }

    /// The verifier's side of [`CodeSwitch::prove`], for a source oracle with
    /// commitment `root`. Returns the claim and the commitment of the new
    /// oracle.
    pub fn verify<F: RsField, H: Hasher>(
        &self,
        transcript: &mut VerifierTranscript<H>,
        claim: VerifierFoldedClaim<F>,
        root: &Digest,
        fold_challenges: &[F],
    ) -> Result<(VerifierClaim<F>, Digest)> {
        let next_root = transcript.read_digest()?;

        transcript.check_work(self.ood_pow_bits)?;
        let ood_point = transcript.challenge_field();
        let pad_coefficients = transcript.challenge_nonzero_vector(self.source.pad_len);
        let ood_answer: F = transcript.read_field()?;

        transcript.check_work(self.query_pow_bits)?;
        let positions = transcript.challenge_positions(self.source.domain_size, self.source.queries);
        let batching = transcript.challenge_field();
        let opening = Opening::read(
            transcript,
            root,
            self.source.domain_size,
            &positions,
            self.source.rows(),
            |_, _| Vec::new(),
        )?;

        let challenges = SwitchChallenges { ood_point, pad_coefficients, positions, batching };
        Ok((self.next_claim(claim, &challenges, ood_answer, &opening, fold_challenges), next_root))
    }

    /// The claim on the message of the new oracle: the batch of the input
    /// claim, the out-of-domain equation, and one equation per query.
    ///
    /// The new main message is the main block `f` of the folded row, and the
    /// new auxiliary message is `(v, s, r, pad)`.
    fn next_claim<F: RsField, M: MainCoefficients<F>>(
        &self,
        claim: FoldedClaim<F, M>,
        challenges: &SwitchChallenges<F>,
        ood_answer: F,
        opening: &Opening<F>,
        fold_challenges: &[F],
    ) -> Claim<F, M> {
        let source = self.source;
        // Where the mask, the seed and the pad begin in the auxiliary message.
        let mask_at = source.aux_len;
        let seed_at = mask_at + MASK_LEN;
        let pad_at = seed_at + source.seed_len;

        // The input claim, with coefficient one.
        let FoldedClaim { mut main, aux: aux_form, mask: mask_form, mut target } = claim;
        let mut aux = vec![F::ZERO; self.next.rows() * self.next.aux_len];
        aux[..mask_at].copy_from_slice(&aux_form);
        aux[mask_at..seed_at].copy_from_slice(&mask_form);

        // The out-of-domain equation: the powers of the point on (f, v, s, r)
        // and the vector β on the pad.
        let mut coefficient = challenges.batching;
        let x = challenges.ood_point;
        let mut main_rows = vec![PowerRow { scale: coefficient, base: x }];
        add_powers(&mut aux[..pad_at], x, coefficient * x.pow(source.main_len as u64));
        for (entry, &beta) in aux[pad_at..].iter_mut().zip(&challenges.pad_coefficients) {
            *entry += coefficient * beta;
        }
        target += coefficient * ood_answer;

        // One equation per query: the symbol of the folded oracle at the
        // domain point z is the value at z of the polynomial with
        // coefficients (r, s, f, v).
        let weights = eq_table(fold_challenges);
        for (column, &position) in opening.columns(&challenges.positions).zip(&challenges.positions) {
            coefficient *= challenges.batching;
            let z = F::point(position);
            let power = |degree: usize| coefficient * z.pow(degree as u64);
            main_rows.push(PowerRow { scale: power(source.main_degree()), base: z });
            add_powers(&mut aux[..mask_at], z, power(source.aux_degree()));
            add_powers(&mut aux[mask_at..seed_at], z, power(source.mask_degree()));
            add_powers(&mut aux[seed_at..pad_at], z, coefficient);
            target += coefficient * inner_product(column, &weights);
        }
        main.add_power_rows(&main_rows);
        Claim { main, aux, target }
    }
}
