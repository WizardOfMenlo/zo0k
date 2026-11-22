use crate::field::Field;
use crate::parallel;

const PARALLEL_THRESHOLD: usize = 1 << 15;
const MIN_PIECE: usize = 1 << 13;

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

pub fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
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

pub fn fold_rows<F: Field>(table: &[F], weights: &[F]) -> Vec<F> {
    assert!(table.len().is_multiple_of(weights.len()));
    let width = table.len() / weights.len();
    if width == 0 {
        return Vec::new();
    }
    if width == 1 {
        return vec![inner_product(table, weights)];
    }
    let mut folded = vec![F::ZERO; width];
    if width < PARALLEL_THRESHOLD {
        for (row, &weight) in table.chunks_exact(width).zip(weights) {
            F::add_scaled_slice(&mut folded, row, weight);
        }
    } else {
        parallel::for_each_piece_mut(&mut folded, MIN_PIECE, |start, piece| {
            let len = piece.len();
            for (row, &weight) in table.chunks_exact(width).zip(weights) {
                F::add_scaled_slice(piece, &row[start..start + len], weight);
            }
        });
    }
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
