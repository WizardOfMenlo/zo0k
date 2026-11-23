use crate::claim::{
    Claim, FoldedClaim, MainCoefficients, PowerRow, ProverClaim, ProverFoldedClaim, VerifierClaim,
    VerifierFoldedClaim,
};
use crate::error::Result;
use crate::field::Field;
use crate::hash::{Digest, Hasher};
use crate::oracle::{Opening, Oracle};
use crate::parameters::{MASK_LEN, Stage};
use crate::polynomial::{eq_table, evaluate, inner_product};
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

fn square_pow2<F: Field>(mut value: F, log_exp: u32) -> F {
    assert!(log_exp <= 63);
    for _ in 0..log_exp {
        value = value.square();
    }
    assert!(log_exp < 64);
    value
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
        let carried_len = row.aux.len() + row.mask.len() + row.seed.len();
        let mut next_aux = Vec::with_capacity(carried_len + pad.len());
        next_aux.extend_from_slice(&row.aux);
        next_aux.extend_from_slice(&row.mask);
        next_aux.extend_from_slice(&row.seed);
        next_aux.extend_from_slice(&pad);
        assert_eq!(next_aux.len(), carried_len + self.source.pad_len);
        assert!(self.source.main_len.is_power_of_two());
        let message = Message::new(self.next, row.main, next_aux, rng);
        let witness = Witness::commit::<H>(self.next, message, self.next_is_last, rng);
        transcript.send_digest(&witness.oracle.root());

        transcript.prove_work(self.ood_pow_bits);
        let ood_point = transcript.challenge_field();
        let pad_coefficients = transcript.challenge_nonzero_vector(self.source.pad_len);
        let main_eval = evaluate(&witness.message.main, ood_point);
        let carried_eval = F::eval_horner_slice(&witness.message.aux[..carried_len], ood_point);
        let pad_eval = inner_product(&pad_coefficients, &witness.message.aux[carried_len..]);
        let ood_answer = main_eval
            + square_pow2(ood_point, self.source.main_len.trailing_zeros()) * carried_eval
            + pad_eval;
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
        assert_eq!(source.mask_len, MASK_LEN);
        assert!(source.main_len.is_power_of_two());
        let mask_at = source.aux_len;
        let seed_at = mask_at + MASK_LEN;
        let pad_at = seed_at + source.seed_len;
        let main_log = source.main_len.trailing_zeros();

        let FoldedClaim { mut main, aux: aux_form, mask: mask_form, mut target } = claim;
        let mut aux = vec![F::ZERO; self.next.rows() * self.next.aux_len];
        aux[..mask_at].copy_from_slice(&aux_form);
        aux[mask_at..seed_at].copy_from_slice(&mask_form);

        let mut coefficient = challenges.batching;
        let x = challenges.ood_point;
        let mut main_rows = Vec::with_capacity(1 + challenges.positions.len());
        main_rows.push((PowerRow { scale: coefficient, base: x }, None));
        F::add_powers_slice(&mut aux[..pad_at], x, coefficient * square_pow2(x, main_log));
        F::add_scaled_slice(&mut aux[pad_at..], &challenges.pad_coefficients, coefficient);
        target += coefficient * ood_answer;

        let weights = eq_table(fold_challenges);
        for (column, &position) in opening.columns(&challenges.positions).zip(&challenges.positions) {
            coefficient *= challenges.batching;
            let z = F::point(position);
            F::add_powers_slice(&mut aux[seed_at..pad_at], z, coefficient);
            let after_seed = coefficient * z.pow(source.seed_len as u64);
            aux[mask_at] += after_seed;
            let after_mask_one = after_seed * z;
            aux[mask_at + 1] += after_mask_one;
            let after_mask = after_mask_one * z;
            main_rows.push((PowerRow { scale: after_mask, base: z }, Some(position)));
            if mask_at > 0 {
                let aux_scale = after_mask * square_pow2(z, main_log);
                F::add_powers_slice(&mut aux[..mask_at], z, aux_scale);
            }
            target += coefficient * F::inner_product_slice(column, &weights);
        }
        main.add_power_rows_positioned(&main_rows);
        Claim { main, aux, target }
    }
}
