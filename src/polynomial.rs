use crate::field::Field;
use crate::parallel;

const PARALLEL_THRESHOLD: usize = 1 << 15;
const MIN_PIECE: usize = 1 << 13;

pub fn evaluate<F: Field>(coefficients: &[F], x: F) -> F {
    coefficients.iter().rev().fold(F::ZERO, |value, &coefficient| value * x + coefficient)
}

pub fn inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    if a.len() < PARALLEL_THRESHOLD {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    } else {
        let piece_product = |range: std::ops::Range<usize>| {
            a[range.clone()].iter().zip(&b[range]).map(|(&x, &y)| x * y).fold(F::ZERO, |x, y| x + y)
        };
        parallel::map_pieces(a.len(), MIN_PIECE, piece_product).into_iter().fold(F::ZERO, |x, y| x + y)
    }
}

pub fn add_powers<F: Field>(target: &mut [F], x: F, mut scale: F) {
    for entry in target {
        *entry += scale;
        scale *= x;
    }
}

pub fn fold<F: Field>(values: &mut Vec<F>, challenge: F) {
    assert!(values.len().is_multiple_of(2));
    let half = values.len() / 2;
    let (low, high) = values.split_at_mut(half);
    let fold_pair = |(low, high): (&mut F, &F)| *low += (*high - *low) * challenge;
    if half < PARALLEL_THRESHOLD {
        low.iter_mut().zip(&*high).for_each(fold_pair);
    } else {
        let high = &*high;
        parallel::for_each_piece_mut(low, MIN_PIECE, |start, piece| {
            piece.iter_mut().zip(&high[start..]).for_each(fold_pair);
        });
    }
    values.truncate(half);
}

pub fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    let mut table = Vec::with_capacity(1 << point.len());
    table.push(F::ONE);
    for &coordinate in point.iter().rev() {
        let half = table.len();
        table.resize(2 * half, F::ZERO);
        let (low, high) = table.split_at_mut(half);
        let extend = |(low, high): (&mut F, &mut F)| {
            *high = *low * coordinate;
            *low -= *high;
        };
        if half < PARALLEL_THRESHOLD {
            low.iter_mut().zip(high).for_each(extend);
        } else {
            parallel::for_each_pair_mut(low, high, MIN_PIECE, |low, high| {
                low.iter_mut().zip(high).for_each(extend);
            });
        }
    }
    table
}

pub fn fold_rows<F: Field>(table: &[F], weights: &[F]) -> Vec<F> {
    assert!(table.len().is_multiple_of(weights.len()));
    let width = table.len() / weights.len();
    let mut folded = vec![F::ZERO; width];
    if width > 0 {
        for (row, &weight) in table.chunks_exact(width).zip(weights) {
            folded.iter_mut().zip(row).for_each(|(entry, &value)| *entry += value * weight);
        }
    }
    folded
}

pub fn evaluate_multilinear<F: Field>(table: &[F], point: &[F]) -> F {
    assert_eq!(table.len(), 1 << point.len());
    let mut values = table.to_vec();
    for &coordinate in point {
        fold(&mut values, coordinate);
    }
    values[0]
}

pub fn round_coefficients<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(2));
    let half = a.len() / 2;
    let (a_low, a_high) = a.split_at(half);
    let (b_low, b_high) = b.split_at(half);
    let term = |i: usize| (a_low[i] * b_low[i], (a_high[i] - a_low[i]) * (b_high[i] - b_low[i]));
    let add = |x: (F, F), y: (F, F)| (x.0 + y.0, x.1 + y.1);
    let zero = (F::ZERO, F::ZERO);
    if half < PARALLEL_THRESHOLD {
        (0..half).map(term).fold(zero, add)
    } else {
        let pieces = parallel::map_pieces(half, MIN_PIECE, |range| range.map(term).fold(zero, add));
        pieces.into_iter().fold(zero, add)
    }
}
