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

pub struct CodeSwitch<'a> {
    pub source: &'a Stage,
    pub next: &'a Stage,
    pub next_is_last: bool,
    pub ood_pow_bits: u32,
    pub query_pow_bits: u32,
}

struct SwitchChallenges<F> {
    ood_point: F,
    pad_coefficients: Vec<F>,
    positions: Vec<usize>,
    batching: F,
}

impl CodeSwitch<'_> {
    pub fn prove<F: RsField, H: Hasher>(
        &self,
        transcript: &mut ProverTranscript<H>,
        claim: ProverFoldedClaim<F>,
        row: Row<F>,
        oracle: &Oracle<F>,
        fold_challenges: &[F],
        rng: &mut Rng,
    ) -> (ProverClaim<F>, Witness<F>) {
        let pad = rng.fields(self.source.pad_len);
        let carried = [&row.aux[..], &row.mask, &row.seed].concat();
        let message = Message::new(self.next, row.main, [&carried[..], &pad].concat(), rng);
        let witness = Witness::commit::<H>(self.next, message, self.next_is_last, rng);
        transcript.send_digest(&witness.oracle.root());

        transcript.prove_work(self.ood_pow_bits);
        let ood_point = transcript.challenge_field();
        let pad_coefficients = transcript.challenge_nonzero_vector(self.source.pad_len);
        let ood_answer = evaluate(&witness.message.main, ood_point)
            + ood_point.pow(self.source.main_len as u64) * evaluate(&carried, ood_point)
            + inner_product(&pad_coefficients, &pad);
        transcript.send_field(ood_answer);

        transcript.prove_work(self.query_pow_bits);
        let positions = transcript.challenge_positions(self.source.domain_size, self.source.queries);
        let batching = transcript.challenge_field();
        let opening = oracle.open(transcript, &positions, self.source.rows());

        let challenges = SwitchChallenges { ood_point, pad_coefficients, positions, batching };
        (self.next_claim(claim, &challenges, ood_answer, &opening, fold_challenges), witness)
    }

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

    fn next_claim<F: RsField, M: MainCoefficients<F>>(
        &self,
        claim: FoldedClaim<F, M>,
        challenges: &SwitchChallenges<F>,
        ood_answer: F,
        opening: &Opening<F>,
        fold_challenges: &[F],
    ) -> Claim<F, M> {
        let source = self.source;
        let mask_at = source.aux_len;
        let seed_at = mask_at + MASK_LEN;
        let pad_at = seed_at + source.seed_len;

        let FoldedClaim { mut main, aux: aux_form, mask: mask_form, mut target } = claim;
        let mut aux = vec![F::ZERO; self.next.rows() * self.next.aux_len];
        aux[..mask_at].copy_from_slice(&aux_form);
        aux[mask_at..seed_at].copy_from_slice(&mask_form);

        let mut coefficient = challenges.batching;
        let x = challenges.ood_point;
        let mut main_rows = vec![PowerRow { scale: coefficient, base: x }];
        add_powers(&mut aux[..pad_at], x, coefficient * x.pow(source.main_len as u64));
        for (entry, &beta) in aux[pad_at..].iter_mut().zip(&challenges.pad_coefficients) {
            *entry += coefficient * beta;
        }
        target += coefficient * ood_answer;

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
