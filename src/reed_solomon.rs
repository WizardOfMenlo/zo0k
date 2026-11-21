use crate::field::{BinaryField, Field, Gf2_128};
use crate::parallel;
use std::sync::OnceLock;

pub const MAX_DOMAIN_SIZE: usize = 1 << BASIS_SIZE;

pub trait RsField: Field {
    fn point(index: usize) -> Self;

    fn encode(coefficients: &[Self], len: usize) -> Vec<Self>;

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

const PARALLEL_THRESHOLD: usize = 1 << 14;
const MIN_PIECE: usize = 1 << 13;

const SMALL_BLOCK: usize = 64;

struct Domain {
    byte_tables: Box<[[u128; 256]; BASIS_SIZE / 8]>,
}

impl Domain {
    fn get() -> &'static Self {
        static DOMAIN: OnceLock<Domain> = OnceLock::new();
        DOMAIN.get_or_init(Self::new)
    }

    fn new() -> Self {
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

    #[inline]
    fn point(&self, index: usize) -> u128 {
        debug_assert!(index < MAX_DOMAIN_SIZE);
        let t = &self.byte_tables;
        t[0][index & 255] ^ t[1][index >> 8 & 255] ^ t[2][index >> 16 & 255] ^ t[3][index >> 24 & 255]
    }
}

fn encode<F: BinaryField>(coefficients: &[F], len: usize) -> Vec<F> {
    let count = coefficients.len();
    assert!(count <= len && len <= MAX_DOMAIN_SIZE);
    if count == 0 {
        return vec![F::ZERO; len];
    }
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

fn lower_exponents(i: usize) -> impl Iterator<Item = usize> {
    (0..i).filter(move |j| j & i == *j)
}

fn to_novel_basis<F: Field>(coefficients: &mut [F], count: usize) {
    let half = coefficients.len() / 2;
    if half == 0 {
        return;
    }
    if count <= half {
        return to_novel_basis(&mut coefficients[..half], count);
    }
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
    let quotient_count = count.saturating_sub(half);
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

fn evaluate_small<F: BinaryField>(values: &mut [F], start: usize) {
    let domain = Domain::get();
    let mut half = values.len() / 2;
    while half > 0 {
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
