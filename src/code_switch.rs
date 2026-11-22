use crate::claim::{
    Claim, FoldedClaim, MainCoefficients, PowerRow, ProverClaim, ProverFoldedClaim, VerifierClaim,
    VerifierFoldedClaim,
};
use crate::error::Result;
use crate::field::Field;
use crate::hash::{Digest, Hasher};
use crate::oracle::{Opening, Oracle};
use crate::parallel;
use crate::parameters::{MASK_LEN, Stage};
use crate::polynomial::{eq_table, inner_product};
use crate::reed_solomon::RsField;
use crate::rng::Rng;
use crate::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use crate::witness::{Message, Row, Witness};

const EVAL_PARALLEL_THRESHOLD: usize = 1 << 14;
const EVAL_PIECE_MIN: usize = 1 << 12;

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

fn eval_horner_4<F: Field>(coefficients: &[F], x: F) -> F {
    let (chunks, remainder) = coefficients.as_chunks::<4>();
    let mut a0 = remainder.iter().rev().fold(F::ZERO, |acc, &c| acc * x + c);
    if chunks.is_empty() {
        assert_eq!(remainder.len(), coefficients.len());
        return a0;
    }
    let x2 = x.square();
    let x3 = x2 * x;
    let x4 = x2.square();
    let (mut a1, mut a2, mut a3) = (F::ZERO, F::ZERO, F::ZERO);
    for chunk in chunks.iter().rev() {
        a0 = a0 * x4 + chunk[0];
        a1 = a1 * x4 + chunk[1];
        a2 = a2 * x4 + chunk[2];
        a3 = a3 * x4 + chunk[3];
    }
    assert!(remainder.len() < 4);
    a0 + x * a1 + x2 * a2 + x3 * a3
}

fn evaluate_parallel<F: Field>(coefficients: &[F], x: F) -> F {
    if coefficients.len() < EVAL_PARALLEL_THRESHOLD {
        return eval_horner_4(coefficients, x);
    }
    assert!(coefficients.len() >= EVAL_PARALLEL_THRESHOLD);
    let pieces = parallel::map_pieces(coefficients.len(), EVAL_PIECE_MIN, |range| {
        (range.len(), eval_horner_4(&coefficients[range], x))
    });
    assert!(!pieces.is_empty());
    pieces.into_iter().rev().fold(F::ZERO, |acc, (len, value)| acc * x.pow(len as u64) + value)
}

fn add_powers_returning_next<F: Field>(target: &mut [F], x: F, mut scale: F) -> F {
    assert!(target.len() <= isize::MAX as usize);
    let (chunks, remainder) = target.as_chunks_mut::<4>();
    if !chunks.is_empty() {
        let x2 = x.square();
        let x4 = x2.square();
        let mut s0 = scale;
        let mut s1 = s0 * x;
        let mut s2 = s0 * x2;
        let mut s3 = s1 * x2;
        for chunk in chunks {
            chunk[0] += s0;
            chunk[1] += s1;
            chunk[2] += s2;
            chunk[3] += s3;
            s0 *= x4;
            s1 *= x4;
            s2 *= x4;
            s3 *= x4;
        }
        scale = s0;
    }
    for entry in remainder {
        *entry += scale;
        scale *= x;
    }
    assert!(target.len() < usize::MAX);
    scale
}

fn inner_product_4<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (F::ZERO, F::ZERO, F::ZERO, F::ZERO);
    let (a_chunks, a_rem) = a.as_chunks::<4>();
    let (b_chunks, b_rem) = b.as_chunks::<4>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
        s0 += ac[0] * bc[0];
        s1 += ac[1] * bc[1];
        s2 += ac[2] * bc[2];
        s3 += ac[3] * bc[3];
    }
    let tail: F = a_rem.iter().zip(b_rem).map(|(&x, &y)| x * y).sum();
    assert_eq!(a_rem.len(), b_rem.len());
    (s0 + s1) + (s2 + s3) + tail
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
        let main_eval = evaluate_parallel(&witness.message.main, ood_point);
        let carried_eval = eval_horner_4(&witness.message.aux[..carried_len], ood_point);
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
        main_rows.push(PowerRow { scale: coefficient, base: x });
        add_powers_returning_next(&mut aux[..pad_at], x, coefficient * square_pow2(x, main_log));
        for (entry, &beta) in aux[pad_at..].iter_mut().zip(&challenges.pad_coefficients) {
            *entry += coefficient * beta;
        }
        target += coefficient * ood_answer;

        let weights = eq_table(fold_challenges);
        for (column, &position) in opening.columns(&challenges.positions).zip(&challenges.positions) {
            coefficient *= challenges.batching;
            let z = F::point(position);
            let after_seed = add_powers_returning_next(&mut aux[seed_at..pad_at], z, coefficient);
            let after_mask = add_powers_returning_next(&mut aux[mask_at..seed_at], z, after_seed);
            main_rows.push(PowerRow { scale: after_mask, base: z });
            if mask_at > 0 {
                let aux_scale = after_mask * square_pow2(z, main_log);
                add_powers_returning_next(&mut aux[..mask_at], z, aux_scale);
            }
            target += coefficient * inner_product_4(column, &weights);
        }
        main.add_power_rows(&main_rows);
        Claim { main, aux, target }
    }
}
