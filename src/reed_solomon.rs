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
use crate::parallel;
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
    let first = &*first;
    parallel::for_each_chunk_mut(rest, block, MIN_PIECE, |_, coset| {
        coset[..count].copy_from_slice(&first[..count])
    });
    parallel::for_each_chunk_mut(&mut values, block, MIN_PIECE, |coset, values| {
        evaluate(values, coset * block, count, len);
    });
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
                parallel::for_each_piece_mut(target, 8 * MIN_PIECE, |start, piece| {
                    piece.iter_mut().zip(&source[start..]).for_each(|(t, s)| *t += *s);
                });
            }
        }
        end = start;
    }
    let (remainder, quotient) = coefficients.split_at_mut(half);
    if half < PARALLEL_THRESHOLD {
        to_novel_basis(remainder, half);
        to_novel_basis(quotient, count - half);
    } else {
        parallel::join(|| to_novel_basis(remainder, half), || to_novel_basis(quotient, count - half));
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
        parallel::for_each_pair_mut(pairs.0, pairs.1, MIN_PIECE, |low, high| {
            low.iter_mut().zip(high).for_each(butterfly);
        });
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
        parallel::join(|| evaluate(low, start, count, len), || evaluate(high, start + half, count, len));
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
