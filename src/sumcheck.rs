use crate::claim::{
    Claim, EqFactor, EqFoldedClaim, FoldedClaim, ProverClaim, ProverFoldedClaim, VerifierClaim,
    VerifierFoldedClaim, l1_tile_len,
};
use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::Hasher;
use crate::mask;
use crate::parallel;
use crate::polynomial::{eq_table, fold, fold_rows, round_coefficients};
use crate::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use crate::witness::{Message, Row};
use std::mem::MaybeUninit;

const PARALLEL_THRESHOLD: usize = 1 << 12;
const PIECE_MIN: usize = 1 << 11;
const FUSED_TILE_BYTES: usize = 8192;
const FUSED_TILE_MAX: usize = 512;

const fn fused_tile_len<F: Field>() -> usize {
    let elems = FUSED_TILE_BYTES / F::BYTES;
    if elems > FUSED_TILE_MAX {
        FUSED_TILE_MAX
    } else if elems < 64 {
        64
    } else {
        elems
    }
}

fn evaluate_round<F: Field>(target: F, constant: F, quadratic: F, challenge: F) -> F {
    let linear = target - F::from_integer(2) * constant - quadratic;
    constant + (linear + quadratic * challenge) * challenge
}

pub(crate) fn row_inner_products_factored<F: Field>(table: &[F], eq: &EqFactor<F>, rows: usize) -> Vec<F> {
    let tile_len = eq.lo.len();
    let tiles_per_row = eq.hi.len();
    let width = tile_len * tiles_per_row;
    assert_eq!(table.len(), rows * width);
    assert!(rows >= 1 && width >= 1);
    let total_tiles = rows * tiles_per_row;
    if table.len() < PARALLEL_THRESHOLD || total_tiles < 2 {
        let mut dots = vec![F::ZERO; rows];
        for (r, dot) in dots.iter_mut().enumerate() {
            let row = &table[r * width..(r + 1) * width];
            let mut sum = F::ZERO;
            for (t, &w) in eq.hi.iter().enumerate() {
                sum += w * F::inner_product_slice(&eq.lo, &row[t * tile_len..(t + 1) * tile_len]);
            }
            *dot = sum;
        }
        return dots;
    }
    let tiles_min = PIECE_MIN.div_ceil(tile_len).max(1);
    let partials = parallel::fold_pieces(
        total_tiles,
        tiles_min,
        || vec![F::ZERO; rows],
        |dots, range| {
            for g in range {
                let r = g / tiles_per_row;
                let t = g % tiles_per_row;
                let off = g * tile_len;
                dots[r] += eq.hi[t] * F::inner_product_slice(&eq.lo, &table[off..off + tile_len]);
            }
        },
    );
    let mut dots = vec![F::ZERO; rows];
    for partial in partials {
        for (d, p) in dots.iter_mut().zip(partial) {
            *d += p;
        }
    }
    dots
}

fn fold_rows_tile_uninit<F: Field>(
    table: &[F],
    weights: &[F],
    width: usize,
    offset: usize,
    out_tile: &mut [MaybeUninit<F>],
) {
    let rows = weights.len();
    let len = out_tile.len();
    assert!(rows >= 1 && offset + len <= width);
    assert_eq!(table.len(), rows * width);
    let stop = offset + len;
    let start_row = if rows >= 4 && rows.is_multiple_of(4) {
        for slot in out_tile.iter_mut() {
            slot.write(F::ZERO);
        }
        0
    } else {
        F::scale_to_uninit(out_tile, &table[offset..stop], weights[0]);
        1
    };
    let out: &mut [F] = unsafe { std::slice::from_raw_parts_mut(out_tile.as_mut_ptr().cast(), len) };
    let groups = (rows - start_row) / 4;
    for group in 0..groups {
        let r = start_row + group * 4;
        F::add_scaled_4_slice(
            out,
            &table[r * width + offset..r * width + stop],
            weights[r],
            &table[(r + 1) * width + offset..(r + 1) * width + stop],
            weights[r + 1],
            &table[(r + 2) * width + offset..(r + 2) * width + stop],
            weights[r + 2],
            &table[(r + 3) * width + offset..(r + 3) * width + stop],
            weights[r + 3],
        );
    }
    for row in (start_row + groups * 4)..rows {
        F::add_scaled_slice(out, &table[row * width + offset..row * width + stop], weights[row]);
    }
}

fn fold_rows_parallel<F: Field>(table: &[F], weights: &[F]) -> Vec<F> {
    assert!(!weights.is_empty());
    assert!(table.len().is_multiple_of(weights.len()));
    let width = table.len() / weights.len();
    if width == 0 {
        return Vec::new();
    }
    let tile_len = l1_tile_len::<F>();
    let mut folded: Vec<F> = Vec::with_capacity(width);
    let spare = &mut folded.spare_capacity_mut()[..width];
    let process = |piece_start: usize, piece: &mut [MaybeUninit<F>]| {
        for (index, tile) in piece.chunks_mut(tile_len).enumerate() {
            fold_rows_tile_uninit(table, weights, width, piece_start + index * tile_len, tile);
        }
    };
    if table.len() < PARALLEL_THRESHOLD || width < PIECE_MIN {
        process(0, spare);
    } else {
        parallel::for_each_piece_mut(spare, PIECE_MIN, process);
    }
    unsafe { folded.set_len(width) };
    assert_eq!(folded.len(), width);
    folded
}

fn fused_fold_and_round<F: Field>(a: &mut Vec<F>, b: &mut Vec<F>, challenge: F) -> (F, F) {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(4));
    if a.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    let half = a.len() / 2;
    let quarter = half / 2;
    let tile = fused_tile_len::<F>();
    let (a_dst, a_src) = a.split_at_mut(half);
    let (b_dst, b_src) = b.split_at_mut(half);
    let (a0, a1) = a_dst.split_at_mut(quarter);
    let (b0, b1) = b_dst.split_at_mut(quarter);
    let (a2, a3) = a_src.split_at(quarter);
    let (b2, b3) = b_src.split_at(quarter);
    let kernel =
        |acc: &mut (F, F), start: usize, s_a0: &mut [F], s_a1: &mut [F], s_b0: &mut [F], s_b1: &mut [F]| {
            let len = s_a0.len();
            let mut offset = 0;
            while offset < len {
                let stop = (offset + tile).min(len);
                let src = start + offset..start + stop;
                F::fold_slice(&mut s_a0[offset..stop], &a2[src.clone()], challenge);
                F::fold_slice(&mut s_a1[offset..stop], &a3[src.clone()], challenge);
                F::fold_slice(&mut s_b0[offset..stop], &b2[src.clone()], challenge);
                F::fold_slice(&mut s_b1[offset..stop], &b3[src], challenge);
                let (tc, tq) = F::round_coefficients_slice(
                    &s_a0[offset..stop],
                    &s_a1[offset..stop],
                    &s_b0[offset..stop],
                    &s_b1[offset..stop],
                );
                acc.0 += tc;
                acc.1 += tq;
                offset = stop;
            }
        };
    let result = if quarter < PARALLEL_THRESHOLD {
        let mut acc = (F::ZERO, F::ZERO);
        kernel(&mut acc, 0, a0, a1, b0, b1);
        acc
    } else {
        parallel::fold_four_mut(a0, a1, b0, b1, PIECE_MIN, || (F::ZERO, F::ZERO), kernel)
            .into_iter()
            .fold((F::ZERO, F::ZERO), |x, y| (x.0 + y.0, x.1 + y.1))
    };
    a.truncate(half);
    b.truncate(half);
    result
}

fn fold_pair<F: Field>(form: &mut Vec<F>, values: &mut Vec<F>, challenge: F, scale_form: Option<F>) {
    assert_eq!(form.len(), values.len());
    assert!(form.len().is_multiple_of(2));
    if form.is_empty() {
        return;
    }
    let half = form.len() / 2;
    let tile = fused_tile_len::<F>();
    let (f_lo, f_hi) = form.split_at_mut(half);
    let (v_lo, v_hi) = values.split_at_mut(half);
    let f_hi = &*f_hi;
    let v_hi = &*v_hi;
    let kernel = |start: usize, fl: &mut [F], vl: &mut [F]| {
        let len = fl.len();
        let mut offset = 0;
        while offset < len {
            let stop = (offset + tile).min(len);
            let src = start + offset..start + stop;
            F::fold_slice(&mut fl[offset..stop], &f_hi[src.clone()], challenge);
            if let Some(batching) = scale_form {
                F::scale_slice(&mut fl[offset..stop], batching);
            }
            F::fold_slice(&mut vl[offset..stop], &v_hi[src], challenge);
            offset = stop;
        }
    };
    if half < PARALLEL_THRESHOLD {
        kernel(0, f_lo, v_lo);
    } else {
        parallel::for_each_pair_indexed_mut(f_lo, v_lo, PIECE_MIN, kernel);
    }
    form.truncate(half);
    values.truncate(half);
}

pub(crate) fn prove_evaluation<F: Field, H: Hasher>(
    transcript: &mut ProverTranscript<H>,
    mut eq_prefix: Vec<F>,
    eq_suffix: EqFactor<F>,
    mut row_dots: Vec<F>,
    target_value: F,
    message: Message<F>,
    pow_bits: u32,
) -> (EqFoldedClaim<F>, Row<F>, Vec<F>, Vec<F>) {
    let Message { main, aux, mask, seeds } = message;
    let folding = mask.folding();
    assert!(aux.is_empty());
    assert_eq!(eq_prefix.len(), 1usize << folding);
    assert_eq!(row_dots.len(), eq_prefix.len());
    assert_eq!(main.len(), eq_prefix.len() * eq_suffix.len());

    let mut mask_sumcheck = mask.sumcheck();
    let mask_sum = mask_sumcheck.sum();
    transcript.send_field(mask_sum);
    let batching = transcript.challenge_field();

    let mut target = mask_sum + target_value * batching;
    let mut challenges = Vec::with_capacity(folding);
    for _ in 0..folding {
        let (mask_constant, mask_quadratic) = mask_sumcheck.round_coefficients();
        let (main_constant, main_quadratic) = round_coefficients(&eq_prefix, &row_dots);
        let constant = mask_constant + main_constant * batching;
        let quadratic = mask_quadratic + main_quadratic * batching;
        transcript.send_fields(&[constant, quadratic]);

        transcript.prove_work(pow_bits);
        let challenge = transcript.challenge_field();
        target = evaluate_round(target, constant, quadratic, challenge);
        mask_sumcheck.bind(challenge);
        fold(&mut eq_prefix, challenge);
        fold(&mut row_dots, challenge);
        challenges.push(challenge);
    }

    assert_eq!(eq_prefix.len(), 1);
    let weights = eq_table(&challenges);
    let folded_main = fold_rows_parallel(&main, &weights);
    let folded_claim = EqFoldedClaim {
        eq: eq_suffix,
        scale: eq_prefix[0] * batching,
        mask: mask::linear_form(&challenges),
        target,
    };
    let folded_row = Row {
        main: folded_main,
        aux: Vec::new(),
        mask: mask.fold(&challenges),
        seed: fold_rows(&seeds, &weights),
    };
    (folded_claim, folded_row, challenges, main)
}

pub(crate) fn prove_inner<F: Field, H: Hasher>(
    transcript: &mut ProverTranscript<H>,
    claim: ProverClaim<F>,
    message: Message<F>,
    pow_bits: u32,
    precomputed_main_r0: Option<(F, F)>,
    scale_final_main: bool,
) -> (ProverFoldedClaim<F>, F, Row<F>, Vec<F>) {
    let Claim { main: mut main_form, aux: mut aux_form, target } = claim;
    let Message { mut main, mut aux, mask, seeds } = message;
    let folding = mask.folding();
    assert!(folding >= 1);
    assert_eq!(main_form.len(), main.len());

    let mut mask_sumcheck = mask.sumcheck();
    let mask_sum = mask_sumcheck.sum();
    transcript.send_field(mask_sum);
    let batching = transcript.challenge_field();

    let mut target = mask_sum + target * batching;
    let mut challenges = Vec::with_capacity(folding);
    let (mut main_constant, mut main_quadratic) =
        precomputed_main_r0.unwrap_or_else(|| round_coefficients(&main_form, &main));
    let (mut aux_constant, mut aux_quadratic) = round_coefficients(&aux_form, &aux);
    for round in 0..folding {
        let (mask_constant, mask_quadratic) = mask_sumcheck.round_coefficients();
        let constant = mask_constant + (main_constant + aux_constant) * batching;
        let quadratic = mask_quadratic + (main_quadratic + aux_quadratic) * batching;
        transcript.send_fields(&[constant, quadratic]);

        transcript.prove_work(pow_bits);
        let challenge = transcript.challenge_field();
        target = evaluate_round(target, constant, quadratic, challenge);
        mask_sumcheck.bind(challenge);
        if round + 1 < folding {
            (main_constant, main_quadratic) = fused_fold_and_round(&mut main_form, &mut main, challenge);
            (aux_constant, aux_quadratic) = fused_fold_and_round(&mut aux_form, &mut aux, challenge);
        } else {
            let main_scale = scale_final_main.then_some(batching);
            fold_pair(&mut main_form, &mut main, challenge, main_scale);
            fold_pair(&mut aux_form, &mut aux, challenge, Some(batching));
        }
        challenges.push(challenge);
    }

    let folded_claim =
        FoldedClaim { main: main_form, aux: aux_form, mask: mask::linear_form(&challenges), target };
    let folded_row =
        Row { main, aux, mask: mask.fold(&challenges), seed: fold_rows(&seeds, &eq_table(&challenges)) };
    let deferred_scale = if scale_final_main { F::ONE } else { batching };
    (folded_claim, deferred_scale, folded_row, challenges)
}

pub fn prove<F: Field, H: Hasher>(
    transcript: &mut ProverTranscript<H>,
    claim: ProverClaim<F>,
    message: Message<F>,
    pow_bits: u32,
) -> (ProverFoldedClaim<F>, Row<F>, Vec<F>) {
    let (folded_claim, _, folded_row, challenges) =
        prove_inner(transcript, claim, message, pow_bits, None, true);
    (folded_claim, folded_row, challenges)
}

fn read_field_pair<F: Field, H: Hasher>(transcript: &mut VerifierTranscript<H>) -> Result<(F, F)> {
    assert!(F::BYTES <= 32);
    let bytes = transcript.read_bytes(2 * F::BYTES)?;
    let error = Error::InvalidProof("a field element is not encoded canonically");
    let first = F::from_bytes(&bytes[..F::BYTES]).ok_or(error.clone())?;
    let second = F::from_bytes(&bytes[F::BYTES..]).ok_or(error)?;
    assert_eq!(bytes.len(), 2 * F::BYTES);
    Ok((first, second))
}

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
        let (constant, quadratic) = read_field_pair::<F, H>(transcript)?;
        transcript.check_work(pow_bits)?;
        let challenge = transcript.challenge_field();
        target = evaluate_round(target, constant, quadratic, challenge);
        challenges.push(challenge);
    }
    main.fix_and_scale(&challenges, batching);
    let aux = if aux.is_empty() {
        Vec::new()
    } else {
        let mut weights = eq_table(&challenges);
        for weight in &mut weights {
            *weight *= batching;
        }
        fold_rows(&aux, &weights)
    };
    Ok((FoldedClaim { main, aux, mask: mask::linear_form(&challenges), target }, challenges))
}
