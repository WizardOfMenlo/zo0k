//! Reed–Solomon codes, and their encoding over binary fields by an additive
//! fast Fourier transform.
//!
//! The protocol needs a field with Reed–Solomon codes of every length: an
//! [`RsField`]. Every [`BinaryField`] is one, with the evaluation domain and
//! the algorithm below.
//!
//! # Evaluation domain
//!
//! The code of length `n` evaluates polynomials at the points
//! `point(0), …, point(n - 1)`, where `point(i) = Σ_j i_j·β_j` for the bits
//! `i_j` of `i` and a fixed sequence `β_0, β_1, …, β_31` of field elements
//! that are linearly independent over `GF(2)`. The points are distinct, and
//! the first `2^m` of them form the subspace `V_m` spanned by
//! `β_0, …, β_{m-1}`.
//!
//! The sequence is a *Cantor basis*: `β_0 = 1` and `β_{i+1}^2 + β_{i+1} = β_i`.
//! Writing `L(X) = X^2 + X`, the vanishing polynomial of `V_i` is then
//! `S_i = L^i`, the `i`-fold composition, which has coefficients in `GF(2)`
//! and satisfies `S_i(β_j) = β_{j-i}` for `j ≥ i`. In particular
//! `S_i(point(t)) = point(t >> i)`. All points lie in `GF(2^32)`, a subfield
//! of [`Gf2_128`] and hence of every [`BinaryField`].
//!
//! # Algorithm
//!
//! The transform is that of Lin, Chung and Han (FOCS 2014). The *novel basis*
//! consists of the polynomials `X_j = Π_{i : j_i = 1} S_i`, with `deg X_j = j`.
//!
//! 1. `to_novel_basis` rewrites a polynomial `P` of degree below `2^(m+1)`
//!    as `R + S_m·Q` by dividing by the sparse polynomial `S_m`, and recurses
//!    on `R` and `Q`. Only additions are needed.
//! 2. `evaluate` uses that `S_m` is constant on a coset `ω + V_m`: on the
//!    block of `2^(m+1)` points starting at index `t`, the polynomial equals
//!    `R + S_m(ω)·Q` on the first half and `R + (S_m(ω) + 1)·Q` on the second,
//!    where `S_m(ω) = point(t >> m)`. This is one multiplication per pair of
//!    coefficients, followed by a recursion on both halves.
//!
//! For `c` coefficients and `n` points this takes `O(c·log² c)` additions and
//! `O(n·log c)` multiplications.

use crate::field::{BinaryField, Field, Gf2_128};
use rayon::prelude::*;
use std::sync::OnceLock;

/// The largest evaluation domain has `2^32` points.
pub const MAX_DOMAIN_SIZE: usize = 1 << BASIS_SIZE;

/// A field with Reed–Solomon codes of every length up to
/// [`MAX_DOMAIN_SIZE`]: the code of length `n` consists of the values of the
/// polynomials of a given degree bound at the distinct points
/// `point(0), …, point(n - 1)`.
pub trait RsField: Field {
    /// The `index`-th evaluation point.
    ///
    /// # Panics
    /// Panics if `index` is not below [`MAX_DOMAIN_SIZE`].
    fn point(index: usize) -> Self;

    /// The codeword of a polynomial: its values at the first `len` points,
    /// given its coefficients from the constant term upwards.
    ///
    /// # Panics
    /// Panics if there are more coefficients than points, or more points
    /// than [`MAX_DOMAIN_SIZE`].
    fn encode(coefficients: &[Self], len: usize) -> Vec<Self>;

    /// The value at `point(index)` of the polynomial with these coefficients.
    fn evaluate_at(coefficients: &[Self], index: usize) -> Self;
}

impl<F: BinaryField> RsField for F {
    fn point(index: usize) -> Self {
        assert!(index < MAX_DOMAIN_SIZE);
        F::from_base(Domain::get().point(index))
    }

    fn encode(coefficients: &[Self], len: usize) -> Vec<Self> {
        encode(coefficients, len)
    }

    fn evaluate_at(coefficients: &[Self], index: usize) -> Self {
        assert!(index < MAX_DOMAIN_SIZE);
        let x = Domain::get().point(index);
        coefficients.iter().rev().fold(F::ZERO, |value, &coefficient| value.mul_base(x) + coefficient)
    }
}

const BASIS_SIZE: usize = 32;

/// Blocks at least this long are split between threads, and loops over that
/// many entries are run in pieces of at least [`MIN_PIECE`] entries: a piece
/// should take tens of microseconds, well above the cost of handing it to
/// another thread.
const PARALLEL_THRESHOLD: usize = 1 << 14;
const MIN_PIECE: usize = 1 << 13;

/// Dense blocks up to this length are transformed without recursion.
const SMALL_BLOCK: usize = 64;

/// For each of the four bytes of an index, the sum of the elements of the
/// Cantor basis `β_0, …, β_31` that the byte selects.
struct Domain {
    byte_tables: Box<[[u128; 256]; BASIS_SIZE / 8]>,
}

impl Domain {
    fn get() -> &'static Self {
        static DOMAIN: OnceLock<Domain> = OnceLock::new();
        DOMAIN.get_or_init(Self::new)
    }

    fn new() -> Self {
        // On K = GF(2^128), L is nilpotent of index 128 and L^127 is the
        // trace. For w of trace one, the elements L^127(w) = 1, L^126(w), …
        // form a Cantor basis of K, and the first 32 of them span GF(2^32).
        // We take the first power of x that has trace one.
        let apply = |v: u128| {
            let v = Gf2_128::new(v);
            (v.square() + v).limb()
        };
        let chain = (0..128)
            .map(|exponent| {
                let mut chain = vec![1u128 << exponent];
                for _ in 0..127 {
                    chain.push(apply(*chain.last().unwrap()));
                }
                chain
            })
            .find(|chain| chain[127] == 1)
            .expect("some power of x has trace one");
        let basis: [u128; BASIS_SIZE] = std::array::from_fn(|i| chain[127 - i]);
        let mut byte_tables = Box::new([[0; 256]; BASIS_SIZE / 8]);
        for (byte, table) in byte_tables.iter_mut().enumerate() {
            for (value, entry) in table.iter_mut().enumerate() {
                *entry = (0..8)
                    .filter(|bit| value >> bit & 1 == 1)
                    .fold(0, |sum, bit| sum ^ basis[8 * byte + bit]);
            }
        }
        Self { byte_tables }
    }

    /// `point(index)` as an element of `K`.
    #[inline]
    fn point(&self, index: usize) -> u128 {
        debug_assert!(index < MAX_DOMAIN_SIZE);
        let t = &self.byte_tables;
        t[0][index & 255] ^ t[1][index >> 8 & 255] ^ t[2][index >> 16 & 255] ^ t[3][index >> 24 & 255]
    }
}

/// [`RsField::encode`] for a binary field.
fn encode<F: BinaryField>(coefficients: &[F], len: usize) -> Vec<F> {
    let count = coefficients.len();
    assert!(count <= len && len <= MAX_DOMAIN_SIZE);
    if count == 0 {
        return vec![F::ZERO; len];
    }
    // The domain is covered by cosets of the smallest subspace that
    // determines the polynomial. The last coset may reach past the domain.
    let block = count.next_power_of_two();
    let mut values = vec![F::ZERO; len.div_ceil(block) * block];
    values[..count].copy_from_slice(coefficients);
    let (first, rest) = values.split_at_mut(block);
    to_novel_basis(first, count);
    rest.par_chunks_mut(block).for_each(|coset| coset[..count].copy_from_slice(&first[..count]));
    values
        .par_chunks_mut(block)
        .enumerate()
        .for_each(|(coset, values)| evaluate(values, coset * block, count, len));
    values.truncate(len);
    values
}

/// The exponents `j < i` such that `X^(2^j)` occurs in `S_i = L^i`: those
/// whose binary digits are among the digits of `i`, as the coefficient of
/// `X^(2^j)` is the binomial coefficient of `i` and `j` modulo two.
fn lower_exponents(i: usize) -> impl Iterator<Item = usize> {
    (0..i).filter(move |j| j & i == *j)
}

/// Replace the monomial coefficients of a polynomial by its coefficients in
/// the novel basis. The slice has a power-of-two length, and only its first
/// `count` entries are nonzero, before and after.
fn to_novel_basis<F: Field>(coefficients: &mut [F], count: usize) {
    let half = coefficients.len() / 2;
    if half == 0 {
        return;
    }
    if count <= half {
        return to_novel_basis(&mut coefficients[..half], count);
    }
    // Divide by S_m = X^half + (lower terms), with m = log2(half): in order
    // of decreasing degree d ≥ half, the coefficient of X^d stays in place as
    // a quotient coefficient and is added to the coefficient of
    // X^(d - half + 2^j) for every lower term X^(2^j). These targets lie at
    // least `gap` below d, so `gap` consecutive degrees are handled at once.
    let m = half.trailing_zeros() as usize;
    let offsets: Vec<usize> = lower_exponents(m).map(|j| 1 << j).collect();
    let gap = half - offsets.last().copied().unwrap_or(0);
    let mut end = count;
    while end > half {
        let start = half.max(end.saturating_sub(gap));
        for &offset in &offsets {
            let (target, source) = coefficients.split_at_mut(start);
            let target = &mut target[start - half + offset..][..end - start];
            let source = &source[..end - start];
            // Additions are an order of magnitude cheaper than products.
            if source.len() < 8 * PARALLEL_THRESHOLD {
                target.iter_mut().zip(source).for_each(|(t, s)| *t += *s);
            } else {
                target.par_iter_mut().zip(source).with_min_len(8 * MIN_PIECE).for_each(|(t, s)| *t += *s);
            }
        }
        end = start;
    }
    let (remainder, quotient) = coefficients.split_at_mut(half);
    if half < PARALLEL_THRESHOLD {
        to_novel_basis(remainder, half);
        to_novel_basis(quotient, count - half);
    } else {
        rayon::join(|| to_novel_basis(remainder, half), || to_novel_basis(quotient, count - half));
    }
}

/// Replace the novel-basis coefficients of a polynomial by its values at the
/// points with indices `start, start + 1, …`, one per entry of `values`.
///
/// The slice has a power-of-two length that divides `start`, and only its
/// first `count` entries are nonzero. Values at indices from `len` on, which
/// are outside the domain, are left unspecified.
fn evaluate<F: BinaryField>(values: &mut [F], start: usize, count: usize, len: usize) {
    let size = values.len();
    if size == 1 {
        return;
    }
    if size <= SMALL_BLOCK && count == size && start + size <= len {
        return evaluate_small(values, start);
    }
    let half = size / 2;
    let twiddle = Domain::get().point(start / half);
    let (low, high) = values.split_at_mut(half);
    // With R and Q the polynomials of the two halves, the first half becomes
    // R + twiddle·Q and the second R + (twiddle + 1)·Q.
    let quotient_count = count.saturating_sub(half);
    // The twiddle of the first block of the domain is zero at every size.
    let butterfly = |(r, q): (&mut F, &mut F)| {
        if twiddle != 0 {
            *r += q.mul_base(twiddle);
        }
        *q += *r;
    };
    let pairs = (&mut low[..quotient_count], &mut high[..quotient_count]);
    if quotient_count < 2 * PARALLEL_THRESHOLD {
        pairs.0.iter_mut().zip(pairs.1).for_each(butterfly);
    } else {
        pairs.0.par_iter_mut().zip(pairs.1).with_min_len(MIN_PIECE).for_each(butterfly);
    }
    // Where Q has no coefficient, both halves hold R.
    let count = count.min(half);
    let needs_high = start + half < len;
    if needs_high {
        high[quotient_count..count].copy_from_slice(&low[quotient_count..count]);
    }
    if half < PARALLEL_THRESHOLD || !needs_high {
        evaluate(low, start, count, len);
        if needs_high {
            evaluate(high, start + half, count, len);
        }
    } else {
        rayon::join(|| evaluate(low, start, count, len), || evaluate(high, start + half, count, len));
    }
}

/// [`evaluate`] for a short block that is dense and inside the domain.
fn evaluate_small<F: BinaryField>(values: &mut [F], start: usize) {
    let domain = Domain::get();
    let mut half = values.len() / 2;
    while half > 0 {
        // The block of 2·half points number i has twiddle
        // point(start / half + 2i) = point(start / half) + point(2i).
        let base = domain.point(start / half);
        for (i, block) in values.chunks_exact_mut(2 * half).enumerate() {
            let twiddle = base ^ domain.byte_tables[0][2 * i];
            let (low, high) = block.split_at_mut(half);
            for (r, q) in low.iter_mut().zip(high) {
                *r += q.mul_base(twiddle);
                *q += *r;
            }
        }
        half /= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Gf2_256;
    use crate::polynomial;
    use crate::rng::Rng;

    #[test]
    fn the_basis_is_a_cantor_basis_of_independent_elements() {
        let basis: Vec<u128> = (0..BASIS_SIZE).map(|i| Domain::get().point(1 << i)).collect();
        assert_eq!(basis[0], 1);
        for pair in basis.windows(2) {
            let next = Gf2_128::new(pair[1]);
            assert_eq!(next.square() + next, Gf2_128::new(pair[0]));
        }
        // Gaussian elimination over GF(2): the 32 elements have rank 32.
        let mut rows = basis;
        let mut rank = 0;
        for bit in 0..128 {
            if let Some(pivot) = (rank..rows.len()).find(|&i| rows[i] >> bit & 1 == 1) {
                rows.swap(rank, pivot);
                let pivot = rows[rank];
                for row in rows.iter_mut().skip(rank + 1) {
                    if *row >> bit & 1 == 1 {
                        *row ^= pivot;
                    }
                }
                rank += 1;
            }
        }
        assert_eq!(rank, BASIS_SIZE);
    }

    fn points<F: RsField>() {
        for index in [0usize, 1, 2, 255, 256, 0x1234_5678, 0xffff_ffff] {
            let expected: F =
                (0..BASIS_SIZE).filter(|bit| index >> bit & 1 == 1).map(|bit| F::point(1 << bit)).sum();
            assert_eq!(F::point(index), expected);
        }
        // S_i = L^i maps point(t) to point(t >> i).
        let t = 0x00c0_ffee;
        let mut value = F::point(t);
        for i in 1..=24 {
            value = value.square() + value;
            assert_eq!(value, F::point(t >> i));
        }
    }

    #[test]
    fn points_are_sums_of_basis_elements() {
        points::<Gf2_128>();
        points::<Gf2_256>();
        // The two fields share their points.
        assert_eq!(Gf2_256::point(12345), Gf2_256::from_base(Gf2_128::point(12345).limb()));
    }

    fn check<F: RsField>(count: usize, len: usize, rng: &mut Rng) {
        let coefficients: Vec<F> = rng.fields(count);
        let codeword = F::encode(&coefficients, len);
        assert_eq!(codeword.len(), len);
        // All positions of short codewords, the ends and random positions of
        // long ones.
        let positions: Vec<usize> = if len <= 512 {
            (0..len).collect()
        } else {
            [0, 1, len - 2, len - 1].into_iter().chain((0..300).map(|_| rng.below(len))).collect()
        };
        for index in positions {
            let expected = polynomial::evaluate(&coefficients, F::point(index));
            assert_eq!(codeword[index], expected, "count {count}, len {len}, index {index}");
            assert_eq!(F::evaluate_at(&coefficients, index), expected);
        }
    }

    fn small_shapes<F: RsField>() {
        let mut rng = Rng::from_seed([20; 32]);
        for len in 1..=70 {
            for count in 0..=len {
                check::<F>(count, len, &mut rng);
            }
        }
    }

    #[test]
    fn encoding_matches_evaluation_for_every_small_shape() {
        small_shapes::<Gf2_128>();
        small_shapes::<Gf2_256>();
    }

    fn block_boundaries<F: RsField>() {
        let mut rng = Rng::from_seed([21; 32]);
        for (count, len) in [
            (1, 1000),
            (63, 64),
            (64, 64),
            (65, 129),
            (100, 4097),
            (1000, 5000),
            (1024, 4096),
            (1025, 4096),
            (3000, 9792),
            (4097, 8191),
            (5000, 5000),
            (8192, 40000),
            (9000, 40001),
            (3, 70000),
            (40000, 40001),
            (5000, 150001),
        ] {
            check::<F>(count, len, &mut rng);
        }
    }

    #[test]
    fn encoding_matches_evaluation_across_block_boundaries() {
        block_boundaries::<Gf2_128>();
        block_boundaries::<Gf2_256>();
    }

    #[test]
    fn a_shorter_code_is_a_prefix_of_a_longer_one() {
        let mut rng = Rng::from_seed([22; 32]);
        let coefficients: Vec<Gf2_256> = rng.fields(300);
        let long = Gf2_256::encode(&coefficients, 5000);
        for len in [300, 512, 513, 2049, 4999] {
            assert_eq!(Gf2_256::encode(&coefficients, len), long[..len]);
        }
    }

    #[test]
    fn encoding_is_linear() {
        type F = Gf2_128;
        let mut rng = Rng::from_seed([23; 32]);
        let (a, b, scale): (Vec<F>, Vec<F>, F) = (rng.fields(40), rng.fields(40), rng.field());
        let combined: Vec<F> = a.iter().zip(&b).map(|(&x, &y)| x + y * scale).collect();
        let expected: Vec<F> =
            F::encode(&a, 200).iter().zip(F::encode(&b, 200)).map(|(&x, y)| x + y * scale).collect();
        assert_eq!(F::encode(&combined, 200), expected);
    }
}
