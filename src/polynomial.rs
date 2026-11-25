use crate::field::Field;
use crate::parallel;
use std::mem::MaybeUninit;

const PARALLEL_THRESHOLD: usize = 1 << 15;
const MIN_PIECE: usize = 1 << 13;
const EQ_OUTER_MIN_VARS: usize = 14;
const EQ_LO_VARS: usize = 10;
const FOLD_TILE_BYTES: usize = 16384;

pub fn evaluate<F: Field>(coefficients: &[F], x: F) -> F {
    let len = coefficients.len();
    if len < PARALLEL_THRESHOLD {
        return F::eval_horner_slice(coefficients, x);
    }
    let pieces = parallel::map_pieces(len, MIN_PIECE, |range| {
        (range.len(), F::eval_horner_slice(&coefficients[range], x))
    });
    assert!(!pieces.is_empty());
    let mut value = F::ZERO;
    for (piece_len, piece_val) in pieces.into_iter().rev() {
        value = value * x.pow(piece_len as u64) + piece_val;
    }
    assert!(len >= PARALLEL_THRESHOLD);
    value
}

pub fn inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    if a.len() < PARALLEL_THRESHOLD {
        F::inner_product_slice(a, b)
    } else {
        let pieces = parallel::map_pieces(a.len(), MIN_PIECE, |range| {
            F::inner_product_slice(&a[range.clone()], &b[range])
        });
        assert!(!pieces.is_empty());
        pieces.into_iter().fold(F::ZERO, |x, y| x + y)
    }
}

pub fn add_powers<F: Field>(target: &mut [F], x: F, scale: F) {
    F::add_powers_slice(target, x, scale);
}

pub fn fold<F: Field>(values: &mut Vec<F>, challenge: F) {
    assert!(values.len().is_multiple_of(2));
    let half = values.len() / 2;
    let (low, high) = values.split_at_mut(half);
    if half < PARALLEL_THRESHOLD {
        F::fold_slice(low, high, challenge);
    } else {
        let high = &*high;
        parallel::for_each_piece_mut(low, MIN_PIECE, |start, piece| {
            F::fold_slice(piece, &high[start..start + piece.len()], challenge);
        });
    }
    values.truncate(half);
    assert_eq!(values.len(), half);
}

fn eq_table_doubling<F: Field>(point: &[F]) -> Vec<F> {
    let total = 1usize << point.len();
    let mut table = Vec::with_capacity(total);
    table.push(F::ONE);
    for &coordinate in point.iter().rev() {
        let half = table.len();
        assert!(2 * half <= table.capacity());
        let (low, high) = {
            let ptr = table.as_mut_ptr();
            unsafe {
                (
                    std::slice::from_raw_parts_mut(ptr, half),
                    std::slice::from_raw_parts_mut(ptr.add(half).cast(), half),
                )
            }
        };
        if half < PARALLEL_THRESHOLD {
            F::eq_extend_uninit(low, high, coordinate);
        } else {
            parallel::for_each_pair_mut(low, high, MIN_PIECE, |low_piece, high_piece| {
                F::eq_extend_uninit(low_piece, high_piece, coordinate);
            });
        }
        unsafe { table.set_len(2 * half) };
    }
    assert_eq!(table.len(), total);
    table
}

pub fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    if point.len() < EQ_OUTER_MIN_VARS {
        return eq_table_doubling(point);
    }
    let total = 1usize << point.len();
    let (hi_point, lo_point) = point.split_at(point.len() - EQ_LO_VARS);
    let eq_hi = eq_table_doubling(hi_point);
    let eq_lo = eq_table_doubling(lo_point);
    let lo_len = eq_lo.len();
    assert_eq!(eq_hi.len() * lo_len, total);
    let mut table: Vec<F> = Vec::with_capacity(total);
    let spare = &mut table.spare_capacity_mut()[..total];
    parallel::for_each_chunk_mut(spare, lo_len, MIN_PIECE, |idx, dst| {
        F::scale_to_uninit(dst, &eq_lo, eq_hi[idx]);
    });
    unsafe { table.set_len(total) };
    assert_eq!(table.len(), total);
    table
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
    F::scale_to_uninit(out_tile, &table[offset..stop], weights[0]);
    let out: &mut [F] = unsafe { std::slice::from_raw_parts_mut(out_tile.as_mut_ptr().cast(), len) };
    let groups = (rows - 1) / 4;
    for group in 0..groups {
        let r = 1 + group * 4;
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
    for row in (1 + groups * 4)..rows {
        F::add_scaled_slice(out, &table[row * width + offset..row * width + stop], weights[row]);
    }
}

pub fn fold_rows<F: Field>(table: &[F], weights: &[F]) -> Vec<F> {
    assert!(!weights.is_empty() && table.len().is_multiple_of(weights.len()));
    let width = table.len() / weights.len();
    if width == 0 {
        return Vec::new();
    }
    if width == 1 {
        return vec![inner_product(table, weights)];
    }
    let tile_len = (FOLD_TILE_BYTES / F::BYTES).max(256);
    let mut folded: Vec<F> = Vec::with_capacity(width);
    let spare = &mut folded.spare_capacity_mut()[..width];
    let process = |piece_start: usize, piece: &mut [MaybeUninit<F>]| {
        for (index, tile) in piece.chunks_mut(tile_len).enumerate() {
            fold_rows_tile_uninit(table, weights, width, piece_start + index * tile_len, tile);
        }
    };
    if width < PARALLEL_THRESHOLD {
        process(0, spare);
    } else {
        parallel::for_each_piece_mut(spare, MIN_PIECE, process);
    }
    unsafe { folded.set_len(width) };
    assert_eq!(folded.len(), width);
    folded
}

pub fn evaluate_multilinear<F: Field>(table: &[F], point: &[F]) -> F {
    assert_eq!(table.len(), 1 << point.len());
    let mut values = table.to_vec();
    for &coordinate in point {
        fold(&mut values, coordinate);
    }
    assert_eq!(values.len(), 1);
    values[0]
}

pub fn round_coefficients<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(2));
    let half = a.len() / 2;
    let (a_low, a_high) = a.split_at(half);
    let (b_low, b_high) = b.split_at(half);
    if half < PARALLEL_THRESHOLD {
        F::round_coefficients_slice(a_low, a_high, b_low, b_high)
    } else {
        let pieces = parallel::map_pieces(half, MIN_PIECE, |range| {
            F::round_coefficients_slice(
                &a_low[range.clone()],
                &a_high[range.clone()],
                &b_low[range.clone()],
                &b_high[range],
            )
        });
        assert!(!pieces.is_empty());
        pieces.into_iter().fold((F::ZERO, F::ZERO), |x, y| (x.0 + y.0, x.1 + y.1))
    }
}
