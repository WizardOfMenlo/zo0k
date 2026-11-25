use crate::claim::{
    self, Claim, EqFoldedClaim, FoldedClaim, MainCoefficients, PowerRow, ProverClaim, ProverFoldedClaim,
    VerifierClaim, VerifierFoldedClaim,
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

pub(crate) struct OpenedSwitch<F> {
    challenges: SwitchChallenges<F>,
    ood_answer: F,
    opening: Opening<F>,
}

struct ClaimParts<F> {
    aux: Vec<F>,
    main_rows: Vec<(PowerRow<F>, Option<usize>)>,
    target: F,
}

fn square_pow2<F: Field>(mut value: F, log_exp: u32) -> F {
    assert!(log_exp <= 63);
    for _ in 0..log_exp {
        value = value.square();
    }
    assert!(log_exp < 64);
    value
}

fn merged_queries<'a, F: Field>(
    positions: &'a [usize],
    batching: F,
    opening: &'a Opening<F>,
) -> Vec<(usize, F, &'a [F])> {
    assert!(!positions.is_empty());
    let mut coefficient = batching;
    let mut entries: Vec<(usize, F, &'a [F])> = Vec::with_capacity(positions.len());
    for (column, &position) in opening.columns(positions).zip(positions) {
        coefficient *= batching;
        entries.push((position, coefficient, column));
    }
    entries.sort_unstable_by_key(|entry| entry.0);
    let mut write = 0usize;
    for read in 0..entries.len() {
        let (pos, coeff, col) = entries[read];
        if write > 0 && entries[write - 1].0 == pos {
            entries[write - 1].1 += coeff;
        } else {
            entries[write] = (pos, coeff, col);
            write += 1;
        }
    }
    entries.truncate(write);
    entries.retain(|entry| entry.1 != F::ZERO);
    assert!(entries.len() <= positions.len());
    entries
}

impl CodeSwitch<'_> {
    pub(crate) fn commit_and_open<F: RsField, H: Hasher>(
        &self,
        transcript: &mut ProverTranscript<H>,
        row: Row<F>,
        oracle: &Oracle<F>,
        rng: &mut Rng,
    ) -> (OpenedSwitch<F>, Witness<F>) {
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
        (OpenedSwitch { challenges, ood_answer, opening }, witness)
    }

    pub fn prove<F: RsField, H: Hasher>(
        &self,
        transcript: &mut ProverTranscript<H>,
        claim: ProverFoldedClaim<F>,
        row: Row<F>,
        oracle: &Oracle<F>,
        fold_challenges: &[F],
        rng: &mut Rng,
    ) -> (ProverClaim<F>, Witness<F>) {
        let (opened, witness) = self.commit_and_open(transcript, row, oracle, rng);
        let next =
            self.next_claim(claim, &opened.challenges, opened.ood_answer, &opened.opening, fold_challenges);
        (next, witness)
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

    fn build_aux_and_rows<F: RsField>(
        &self,
        aux_form: &[F],
        mask_form: &[F; MASK_LEN],
        mut target: F,
        opened: (&SwitchChallenges<F>, F, &Opening<F>),
        fold_challenges: &[F],
    ) -> ClaimParts<F> {
        let (challenges, ood_answer, opening) = opened;
        let source = self.source;
        assert_eq!(source.mask_len, MASK_LEN);
        assert!(source.main_len.is_power_of_two());
        let mask_at = source.aux_len;
        let seed_at = mask_at + MASK_LEN;
        let pad_at = seed_at + source.seed_len;
        let main_log = source.main_len.trailing_zeros();

        let mut aux = vec![F::ZERO; self.next.rows() * self.next.aux_len];
        aux[..mask_at].copy_from_slice(aux_form);
        aux[mask_at..seed_at].copy_from_slice(mask_form);

        let ood_coeff = challenges.batching;
        let x = challenges.ood_point;
        let queries = merged_queries(&challenges.positions, ood_coeff, opening);
        let mut main_rows = Vec::with_capacity(1 + queries.len());
        main_rows.push((PowerRow { scale: ood_coeff, base: x }, None));
        F::add_powers_slice(&mut aux[..pad_at], x, ood_coeff * square_pow2(x, main_log));
        F::add_scaled_slice(&mut aux[pad_at..], &challenges.pad_coefficients, ood_coeff);
        target += ood_coeff * ood_answer;

        let weights = eq_table(fold_challenges);
        for (position, coefficient, column) in queries {
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
        ClaimParts { aux, main_rows, target }
    }

    fn next_claim<F: RsField, M: MainCoefficients<F>>(
        &self,
        claim: FoldedClaim<F, M>,
        challenges: &SwitchChallenges<F>,
        ood_answer: F,
        opening: &Opening<F>,
        fold_challenges: &[F],
    ) -> Claim<F, M> {
        let FoldedClaim { mut main, aux: aux_form, mask: mask_form, target } = claim;
        let parts = self.build_aux_and_rows(
            &aux_form,
            &mask_form,
            target,
            (challenges, ood_answer, opening),
            fold_challenges,
        );
        main.add_power_rows_positioned(&parts.main_rows);
        Claim { main, aux: parts.aux, target: parts.target }
    }

    pub(crate) fn finish_from_eq<F: RsField>(
        &self,
        claim: &EqFoldedClaim<F>,
        opened: &OpenedSwitch<F>,
        fold_challenges: &[F],
        next_main: &[F],
    ) -> (ProverClaim<F>, (F, F)) {
        assert_eq!(claim.eq.len(), next_main.len());
        let parts = self.build_aux_and_rows(
            &[],
            &claim.mask,
            claim.target,
            (&opened.challenges, opened.ood_answer, &opened.opening),
            fold_challenges,
        );
        let rows: Vec<PowerRow<F>> = parts.main_rows.into_iter().map(|(row, _)| row).collect();
        let (main, round0) = claim::materialize_and_round_from_eq(&claim.eq, claim.scale, &rows, next_main);
        assert_eq!(main.len(), next_main.len());
        (Claim { main, aux: parts.aux, target: parts.target }, round0)
    }

    pub(crate) fn finish_from_dense<F: RsField>(
        &self,
        claim: ProverFoldedClaim<F>,
        main_scale: F,
        opened: &OpenedSwitch<F>,
        fold_challenges: &[F],
        next_main: &[F],
    ) -> (ProverClaim<F>, (F, F)) {
        let FoldedClaim { mut main, aux: aux_form, mask: mask_form, target } = claim;
        assert_eq!(main.len(), next_main.len());
        let parts = self.build_aux_and_rows(
            &aux_form,
            &mask_form,
            target,
            (&opened.challenges, opened.ood_answer, &opened.opening),
            fold_challenges,
        );
        let rows: Vec<PowerRow<F>> = parts.main_rows.into_iter().map(|(row, _)| row).collect();
        let round0 = claim::scale_add_powers_and_round(&mut main, main_scale, &rows, next_main);
        assert_eq!(main.len(), next_main.len());
        (Claim { main, aux: parts.aux, target: parts.target }, round0)
    }
}
