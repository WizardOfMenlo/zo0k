//! Univariate and multilinear polynomials over the field.
//!
//! A table of `2^k` rows of `w` entries each is stored row by row: entry
//! `b·w + j` is entry `j` of row `b`. A row index `b` in `{0,1}^k` is read
//! with its first variable in the highest bit, so that tables are in
//! lexicographic order of `(X_1, …, X_k)`. Folding a table at a challenge
//! fixes `X_1`, which combines its two halves, so folding `k` times leaves
//! one row. An evaluation table of a multilinear polynomial is the case
//! `w = 1`.

use crate::field::Field;
use crate::parallel;

/// Tables shorter than this are processed on one thread, and longer ones in
/// pieces of at least [`MIN_PIECE`] entries: a piece should take tens of
/// microseconds, well above the cost of handing it to another thread.
const PARALLEL_THRESHOLD: usize = 1 << 15;
const MIN_PIECE: usize = 1 << 13;

/// `Σ_j coefficients[j]·x^j`.
pub fn evaluate<F: Field>(coefficients: &[F], x: F) -> F {
    coefficients.iter().rev().fold(F::ZERO, |value, &coefficient| value * x + coefficient)
}

/// `Σ_j a[j]·b[j]`.
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

/// Add `scale·x^j` to `target[j]` for every `j`.
pub fn add_powers<F: Field>(target: &mut [F], x: F, mut scale: F) {
    for entry in target {
        *entry += scale;
        scale *= x;
    }
}

/// Fix the first variable of a table to `challenge`: the result is
/// `(1 - challenge)·low + challenge·high` for the two halves of the table.
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

/// The table of `eq(b, point) = Π_i (b_i·point_i + (1 - b_i)(1 - point_i))`
/// over `b` in `{0,1}^k`, where `k` is the length of `point`.
pub fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    let mut table = Vec::with_capacity(1 << point.len());
    table.push(F::ONE);
    // Each coordinate doubles the table and becomes the highest bit so far,
    // so the first coordinate has to come last.
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

/// Fix all row variables of a table at once: with `weights = eq_table(γ)`,
/// the result is the combination `Σ_b weights[b]·row_b` of the rows.
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

/// The value at `point` of the multilinear polynomial with evaluation table
/// `table`: `Σ_b table[b]·eq(b, point)`.
///
/// # Panics
/// Panics unless the table has `2^k` entries for `k` the length of `point`.
pub fn evaluate_multilinear<F: Field>(table: &[F], point: &[F]) -> F {
    assert_eq!(table.len(), 1 << point.len());
    let mut values = table.to_vec();
    for &coordinate in point {
        fold(&mut values, coordinate);
    }
    values[0]
}

/// The constant and quadratic coefficients of `Σ_i a_i(X)·b_i(X)`, where
/// `a_i(X) = a_low[i] + (a_high[i] - a_low[i])·X` for the two halves of `a`,
/// and likewise for `b`: the round polynomial of a sumcheck for the inner
/// product of two tables.
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

#[cfg(test)]
mod tests {
    use super::*;
    type F = crate::field::Gf2_256;
    use crate::rng::Rng;

    /// The point of the cube with the given index: first coordinate in
    /// the highest bit.
    fn bits(index: usize, count: usize) -> Vec<F> {
        (0..count).rev().map(|i| F::from_integer(index >> i)).collect()
    }

    #[test]
    fn eq_table_is_the_indicator_on_the_cube_and_sums_to_one() {
        let mut rng = Rng::from_seed([10; 32]);
        for k in [0, 1, 3, 13] {
            let point = rng.fields(k);
            let table = eq_table(&point);
            assert_eq!(table.iter().sum::<F>(), F::ONE);
            for index in [0, (1 << k) / 3, (1 << k) - 1] {
                let literal: F = point
                    .iter()
                    .zip(bits(index, k))
                    .map(|(&x, b)| b * x + (F::ONE - b) * (F::ONE - x))
                    .fold(F::ONE, |product, factor| product * factor);
                assert_eq!(table[index], literal);
            }
        }
        assert_eq!(eq_table(&bits(5, 3))[5], F::ONE);
    }

    #[test]
    fn folding_fixes_the_first_variable() {
        let mut rng = Rng::from_seed([11; 32]);
        for log in [1, 4, 13] {
            let table = rng.fields(1 << log);
            let point = rng.fields(log);
            let expected = inner_product(&table, &eq_table(&point));
            assert_eq!(evaluate_multilinear(&table, &point), expected);
            for index in [0, 1, (1 << log) - 1] {
                assert_eq!(evaluate_multilinear(&table, &bits(index, log)), table[index]);
            }
            // The first variable selects a half of the table.
            let mut half = table.clone();
            fold(&mut half, F::ONE);
            assert_eq!(half, table[1 << (log - 1)..]);
        }
    }

    #[test]
    fn fold_rows_agrees_with_repeated_folding() {
        let mut rng = Rng::from_seed([12; 32]);
        let (k, width) = (3, 5);
        let table: Vec<F> = rng.fields(width << k);
        let challenges = rng.fields(k);
        let mut folded = table.clone();
        for &challenge in &challenges {
            fold(&mut folded, challenge);
        }
        assert_eq!(fold_rows(&table, &eq_table(&challenges)), folded);
    }

    #[test]
    fn round_coefficients_match_the_folded_inner_product() {
        let mut rng = Rng::from_seed([13; 32]);
        for len in [2, 64, 1 << 13] {
            let (mut a, mut b): (Vec<F>, Vec<F>) = (rng.fields(len), rng.fields(len));
            let (constant, quadratic) = round_coefficients(&a, &b);
            // The polynomial takes the values p(0) + p(1) = <a, b>.
            let linear = inner_product(&a, &b) - constant - constant - quadratic;
            let x: F = rng.field();
            fold(&mut a, x);
            fold(&mut b, x);
            assert_eq!(inner_product(&a, &b), constant + linear * x + quadratic * x * x);
        }
    }

    #[test]
    fn univariate_helpers() {
        let mut rng = Rng::from_seed([14; 32]);
        let (coefficients, x, scale) = (rng.fields(9), rng.field(), rng.field());
        let mut powers = vec![F::ZERO; 9];
        add_powers(&mut powers, x, scale);
        assert_eq!(inner_product(&coefficients, &powers), evaluate(&coefficients, x) * scale);
        assert_eq!(evaluate(&[], x), F::ZERO);
    }
}
