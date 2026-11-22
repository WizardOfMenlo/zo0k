use crate::claim::{Claim, FoldedClaim, ProverClaim, ProverFoldedClaim, VerifierClaim, VerifierFoldedClaim};
use crate::error::Result;
use crate::field::Field;
use crate::hash::Hasher;
use crate::mask;
use crate::parallel;
use crate::polynomial::{eq_table, fold, fold_rows, round_coefficients};
use crate::transcript::{Challenges, ProverTranscript, VerifierTranscript};
use crate::witness::{Message, Row};

const PARALLEL_THRESHOLD: usize = 1 << 14;
const PIECE_MIN: usize = 1 << 12;
const L1_TILE: usize = 1 << 10;

fn evaluate_round<F: Field>(target: F, constant: F, quadratic: F, challenge: F) -> F {
    let linear = target - F::from_integer(2) * constant - quadratic;
    constant + (linear + quadratic * challenge) * challenge
}

fn dot_one_row<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (F::ZERO, F::ZERO, F::ZERO, F::ZERO);
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    for (x, y) in ac.iter().zip(bc.iter()) {
        s0 += x[0] * y[0];
        s1 += x[1] * y[1];
        s2 += x[2] * y[2];
        s3 += x[3] * y[3];
    }
    let tail: F = ar.iter().zip(br).map(|(&x, &y)| x * y).sum();
    assert_eq!(ar.len(), br.len());
    (s0 + s1) + (s2 + s3) + tail
}

fn accumulate_row_dots_range<F: Field>(
    table: &[F],
    eq_suffix: &[F],
    rows: usize,
    start: usize,
    end: usize,
) -> Vec<F> {
    let width = eq_suffix.len();
    assert!(start <= end && end <= width);
    assert_eq!(table.len(), rows * width);
    let mut dots = vec![F::ZERO; rows];
    let mut offset = start;
    while offset < end {
        let stop = (offset + L1_TILE).min(end);
        let eq_tile = &eq_suffix[offset..stop];
        for (r, dot) in dots.iter_mut().enumerate() {
            *dot += dot_one_row(eq_tile, &table[r * width + offset..r * width + stop]);
        }
        offset = stop;
    }
    dots
}

pub(crate) fn row_inner_products<F: Field>(table: &[F], eq_suffix: &[F], rows: usize) -> Vec<F> {
    let width = eq_suffix.len();
    assert_eq!(table.len(), rows * width);
    assert!(rows >= 1);
    if width == 0 {
        return vec![F::ZERO; rows];
    }
    if width >= PIECE_MIN {
        let partials = parallel::map_pieces(width, PIECE_MIN, |range| {
            accumulate_row_dots_range(table, eq_suffix, rows, range.start, range.end)
        });
        let mut dots = vec![F::ZERO; rows];
        for partial in partials {
            for (d, p) in dots.iter_mut().zip(partial) {
                *d += p;
            }
        }
        return dots;
    }
    if table.len() >= PARALLEL_THRESHOLD {
        return parallel::map_range(rows, 1, |r| dot_one_row(eq_suffix, &table[r * width..(r + 1) * width]));
    }
    accumulate_row_dots_range(table, eq_suffix, rows, 0, width)
}

fn fold_rows_tile<F: Field>(table: &[F], weights: &[F], width: usize, offset: usize, out_tile: &mut [F]) {
    let rows = weights.len();
    let len = out_tile.len();
    assert_eq!(table.len(), rows * width);
    assert!(offset + len <= width);
    let stop = offset + len;
    let (out_chunks, out_rem) = out_tile.as_chunks_mut::<4>();
    for (r, &w) in weights.iter().enumerate() {
        let (row_chunks, row_rem) = table[r * width + offset..r * width + stop].as_chunks::<4>();
        if r == 0 {
            for (out, src) in out_chunks.iter_mut().zip(row_chunks.iter()) {
                out[0] = src[0] * w;
                out[1] = src[1] * w;
                out[2] = src[2] * w;
                out[3] = src[3] * w;
            }
            for (out, &src) in out_rem.iter_mut().zip(row_rem) {
                *out = src * w;
            }
        } else {
            for (out, src) in out_chunks.iter_mut().zip(row_chunks.iter()) {
                out[0] += src[0] * w;
                out[1] += src[1] * w;
                out[2] += src[2] * w;
                out[3] += src[3] * w;
            }
            for (out, &src) in out_rem.iter_mut().zip(row_rem) {
                *out += src * w;
            }
        }
    }
}

fn fold_rows_parallel<F: Field>(table: &[F], weights: &[F]) -> Vec<F> {
    assert!(!weights.is_empty());
    assert!(table.len().is_multiple_of(weights.len()));
    let width = table.len() / weights.len();
    let mut folded = vec![F::ZERO; width];
    if width == 0 {
        return folded;
    }
    let process = |piece_start: usize, piece: &mut [F]| {
        for (index, tile) in piece.chunks_mut(L1_TILE).enumerate() {
            fold_rows_tile(table, weights, width, piece_start + index * L1_TILE, tile);
        }
    };
    if table.len() < PARALLEL_THRESHOLD || width < PIECE_MIN {
        process(0, &mut folded);
    } else {
        parallel::for_each_piece_mut(&mut folded, PIECE_MIN, process);
    }
    folded
}

fn scale_slice_in_place<F: Field>(slice: &mut [F], scale: F) {
    assert!(slice.len() <= isize::MAX as usize);
    let apply = |_: usize, piece: &mut [F]| {
        let (chunks, remainder) = piece.as_chunks_mut::<4>();
        for chunk in chunks {
            chunk[0] *= scale;
            chunk[1] *= scale;
            chunk[2] *= scale;
            chunk[3] *= scale;
        }
        for entry in remainder {
            *entry *= scale;
        }
    };
    if slice.len() < PARALLEL_THRESHOLD {
        apply(0, slice);
    } else {
        parallel::for_each_piece_mut(slice, PIECE_MIN, apply);
    }
    assert!(slice.len() < usize::MAX);
}

fn fused_fold_and_round<F: Field>(a: &mut Vec<F>, b: &mut Vec<F>, challenge: F) -> (F, F) {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(4));
    if a.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    let half = a.len() / 2;
    let quarter = half / 2;
    let (a_dst, a_src) = a.split_at_mut(half);
    let (b_dst, b_src) = b.split_at_mut(half);
    let (a0, a1) = a_dst.split_at_mut(quarter);
    let (b0, b1) = b_dst.split_at_mut(quarter);
    let (a2, a3) = a_src.split_at(quarter);
    let (b2, b3) = b_src.split_at(quarter);
    let kernel = |start: usize, s_a0: &mut [F], s_a1: &mut [F], s_b0: &mut [F], s_b1: &mut [F]| {
        let (mut c, mut q) = (F::ZERO, F::ZERO);
        let end = start + s_a0.len();
        let (r_a2, r_a3) = (&a2[start..end], &a3[start..end]);
        let (r_b2, r_b3) = (&b2[start..end], &b3[start..end]);
        for i in 0..s_a0.len() {
            let al = s_a0[i] + (r_a2[i] - s_a0[i]) * challenge;
            let ah = s_a1[i] + (r_a3[i] - s_a1[i]) * challenge;
            let bl = s_b0[i] + (r_b2[i] - s_b0[i]) * challenge;
            let bh = s_b1[i] + (r_b3[i] - s_b1[i]) * challenge;
            s_a0[i] = al;
            s_a1[i] = ah;
            s_b0[i] = bl;
            s_b1[i] = bh;
            c += al * bl;
            q += (ah - al) * (bh - bl);
        }
        (c, q)
    };
    let result = if quarter < PARALLEL_THRESHOLD {
        kernel(0, a0, a1, b0, b1)
    } else {
        parallel::map_four_mut(a0, a1, b0, b1, PIECE_MIN, kernel)
            .into_iter()
            .fold((F::ZERO, F::ZERO), |x, y| (x.0 + y.0, x.1 + y.1))
    };
    a.truncate(half);
    b.truncate(half);
    result
}

fn fold_pair_scaled<F: Field>(form: &mut Vec<F>, values: &mut Vec<F>, challenge: F, batching: F) {
    assert_eq!(form.len(), values.len());
    assert!(form.len().is_multiple_of(2));
    if form.is_empty() {
        return;
    }
    let half = form.len() / 2;
    let (f_lo, f_hi) = form.split_at_mut(half);
    let (v_lo, v_hi) = values.split_at_mut(half);
    let f_hi = &*f_hi;
    let v_hi = &*v_hi;
    let kernel = |start: usize, fl: &mut [F], vl: &mut [F]| {
        let end = start + fl.len();
        let (fh, vh) = (&f_hi[start..end], &v_hi[start..end]);
        for i in 0..fl.len() {
            fl[i] = (fl[i] + (fh[i] - fl[i]) * challenge) * batching;
            vl[i] += (vh[i] - vl[i]) * challenge;
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
    mut eq_suffix: Vec<F>,
    mut row_dots: Vec<F>,
    target_value: F,
    message: Message<F>,
    pow_bits: u32,
) -> (ProverFoldedClaim<F>, Row<F>, Vec<F>) {
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
    scale_slice_in_place(&mut eq_suffix, eq_prefix[0] * batching);
    let weights = eq_table(&challenges);
    let folded_main = fold_rows_parallel(&main, &weights);
    let folded_claim =
        FoldedClaim { main: eq_suffix, aux: Vec::new(), mask: mask::linear_form(&challenges), target };
    let folded_row = Row {
        main: folded_main,
        aux: Vec::new(),
        mask: mask.fold(&challenges),
        seed: fold_rows(&seeds, &weights),
    };
    (folded_claim, folded_row, challenges)
}

pub fn prove<F: Field, H: Hasher>(
    transcript: &mut ProverTranscript<H>,
    claim: ProverClaim<F>,
    message: Message<F>,
    pow_bits: u32,
) -> (ProverFoldedClaim<F>, Row<F>, Vec<F>) {
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
    let (mut main_constant, mut main_quadratic) = round_coefficients(&main_form, &main);
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
            fold_pair_scaled(&mut main_form, &mut main, challenge, batching);
            fold_pair_scaled(&mut aux_form, &mut aux, challenge, batching);
        }
        challenges.push(challenge);
    }

    main_form.shrink_to_fit();
    main.shrink_to_fit();
    let folded_claim =
        FoldedClaim { main: main_form, aux: aux_form, mask: mask::linear_form(&challenges), target };
    let folded_row =
        Row { main, aux, mask: mask.fold(&challenges), seed: fold_rows(&seeds, &eq_table(&challenges)) };
    (folded_claim, folded_row, challenges)
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
        let coefficients = transcript.read_fields(2)?;
        transcript.check_work(pow_bits)?;
        let challenge = transcript.challenge_field();
        target = evaluate_round(target, coefficients[0], coefficients[1], challenge);
        challenges.push(challenge);
    }
    main.fix(&challenges);
    main.scale(batching);
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
