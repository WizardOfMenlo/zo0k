use crate::field::{BinaryField, Field, Gf2_128, precompute_mul_pair};
use crate::parallel;
use std::mem::MaybeUninit;
use std::slice;
use std::sync::OnceLock;

pub const MAX_DOMAIN_SIZE: usize = 1 << BASIS_SIZE;

pub trait RsField: Field {
    fn point(index: usize) -> Self;

    fn encode(coefficients: &[Self], len: usize) -> Vec<Self>;

    fn encode_parts(parts: &[&[Self]], len: usize, _allow_parallel: bool) -> Vec<Self> {
        let total: usize = parts.iter().map(|part| part.len()).sum();
        let mut combined = Vec::with_capacity(total);
        for part in parts {
            combined.extend_from_slice(part);
        }
        Self::encode(&combined, len)
    }

    fn encode_batch<const P: usize>(rows: &[[&[Self]; P]], len: usize) -> Vec<Vec<Self>> {
        let allow_inner = rows.len() < parallel::threads();
        parallel::map_range(rows.len(), 1, |row| Self::encode_parts(&rows[row], len, allow_inner))
    }

    fn evaluate_at(coefficients: &[Self], index: usize) -> Self;
}

impl<F: BinaryField> RsField for F {
    fn point(index: usize) -> Self {
        assert!(index < MAX_DOMAIN_SIZE);
        F::from_base(Domain::get().point(index))
    }

    fn encode(coefficients: &[Self], len: usize) -> Vec<Self> {
        encode_parts(&[coefficients], len, true)
    }

    fn encode_parts(parts: &[&[Self]], len: usize, allow_parallel: bool) -> Vec<Self> {
        encode_parts(parts, len, allow_parallel)
    }

    fn encode_batch<const P: usize>(rows: &[[&[Self]; P]], len: usize) -> Vec<Vec<Self>> {
        encode_batch(rows, len)
    }

    fn evaluate_at(coefficients: &[Self], index: usize) -> Self {
        assert!(index < MAX_DOMAIN_SIZE);
        let x = Domain::get().point(index);
        F::eval_horner_base(coefficients, x)
    }
}

const BASIS_SIZE: usize = 32;
const PARALLEL_THRESHOLD: usize = 1 << 14;
const MIN_PIECE: usize = 1 << 13;
const L1_BLOCK: usize = 2048;
const NOVEL_TILE: usize = 1024;

struct Domain {
    byte_tables: Box<[[u128; 256]; BASIS_SIZE / 8]>,
    mul_tables: Box<[[(u128, u128); 256]; 2]>,
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
        let mut mul_tables = Box::new([[(0u128, 0u128); 256]; 2]);
        for byte in 0..2 {
            for value in 0..256 {
                mul_tables[byte][value] = precompute_mul_pair(byte_tables[byte][value]);
            }
        }
        Self { byte_tables, mul_tables }
    }

    #[inline(always)]
    fn point(&self, index: usize) -> u128 {
        debug_assert!(index < MAX_DOMAIN_SIZE);
        let t = &self.byte_tables;
        t[0][index & 255] ^ t[1][index >> 8 & 255] ^ t[2][index >> 16 & 255] ^ t[3][index >> 24 & 255]
    }

    #[inline(always)]
    fn point_small(&self, index: usize) -> u128 {
        debug_assert!(index < 65536);
        let t = &self.byte_tables;
        t[0][index & 255] ^ t[1][index >> 8]
    }
}

fn copy_to_uninit<T: Copy>(dst: &mut [MaybeUninit<T>], src: &[T]) {
    assert_eq!(dst.len(), src.len());
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<T>(), src.len());
    }
    assert_eq!(dst.len(), src.len());
}

fn evaluate_half_from_novel<F: BinaryField>(
    novel_low: &[F],
    novel_high: &[F],
    dst: &mut [MaybeUninit<F>],
    half_index: usize,
    len: usize,
    allow_parallel: bool,
) {
    let half = dst.len();
    let start = half_index * half;
    assert!(half >= 1 && half.is_power_of_two() && start < len);
    assert!(novel_low.len() == half && novel_high.len() <= half);
    let domain = Domain::get();
    let w_eff = domain.point(half_index);
    if half >= 4 && start + half <= len && (!allow_parallel || half < 4 * PARALLEL_THRESHOLD) {
        let q = half / 4;
        let w = domain.point(2 * half_index);
        let w0 = domain.point(4 * half_index);
        let w1 = w0 ^ domain.byte_tables[0][2];
        F::copy_half_radix4_uninit(novel_low, novel_high, dst, w_eff, w, w0, w1);
        let full = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), half) };
        let (low, high) = full.split_at_mut(2 * q);
        let (v0, v1) = low.split_at_mut(q);
        let (v2, v3) = high.split_at_mut(q);
        if !allow_parallel || 2 * q < PARALLEL_THRESHOLD {
            evaluate_full(v0, start, len, allow_parallel);
            evaluate_full(v1, start + q, len, allow_parallel);
            evaluate_full(v2, start + 2 * q, len, allow_parallel);
            evaluate_full(v3, start + 3 * q, len, allow_parallel);
        } else {
            parallel::join(
                || {
                    evaluate_full(v0, start, len, true);
                    evaluate_full(v1, start + q, len, true);
                },
                || {
                    evaluate_full(v2, start + 2 * q, len, true);
                    evaluate_full(v3, start + 3 * q, len, true);
                },
            );
        }
        return;
    }
    F::copy_half_uninit(novel_low, novel_high, dst, w_eff);
    let full = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), half) };
    evaluate_full(full, start, len, allow_parallel);
}

fn prepare_novel_buffer<F: BinaryField>(
    parts: &[&[F]],
    count: usize,
    block: usize,
    padded: usize,
    allow_parallel: bool,
) -> (Vec<F>, Vec<F>) {
    assert!(count >= 2 && block >= 2 && padded >= block);
    let half = block / 2;
    let mut values: Vec<F> = Vec::with_capacity(padded);
    let spare = &mut values.spare_capacity_mut()[..padded];
    let mut offset = 0;
    for part in parts {
        copy_to_uninit(&mut spare[offset..offset + part.len()], part);
        offset += part.len();
    }
    assert_eq!(offset, count);
    let first_init = unsafe { slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<F>(), count) };
    to_novel_basis_slice(first_init, block, allow_parallel);
    let quotient = first_init[half..count].to_vec();
    (values, quotient)
}

pub(crate) fn encode_parts<F: BinaryField>(parts: &[&[F]], len: usize, allow_parallel: bool) -> Vec<F> {
    let count: usize = parts.iter().map(|p| p.len()).sum();
    assert!(count <= len && len <= MAX_DOMAIN_SIZE);
    if count == 0 {
        return vec![F::ZERO; len];
    }
    if count == 1 {
        let single = parts.iter().find_map(|p| p.first().copied()).expect("count is one");
        return vec![single; len];
    }
    let block = count.next_power_of_two();
    let half = block / 2;
    let num_halves = len.div_ceil(half);
    let padded = num_halves * half;
    let (mut values, quotient) = prepare_novel_buffer(parts, count, block, padded, allow_parallel);
    let spare = &mut values.spare_capacity_mut()[..padded];
    let (first_half, rest_halves) = spare.split_at_mut(half);
    let novel_low = unsafe { slice::from_raw_parts(first_half.as_ptr().cast::<F>(), half) };
    let inner_par = allow_parallel && num_halves < parallel::threads();
    if allow_parallel && !rest_halves.is_empty() {
        parallel::for_each_chunk_mut(rest_halves, half, MIN_PIECE.min(half), |idx, dst| {
            evaluate_half_from_novel(novel_low, &quotient, dst, idx + 1, len, inner_par);
        });
    } else {
        for (idx, dst) in rest_halves.chunks_exact_mut(half).enumerate() {
            evaluate_half_from_novel(novel_low, &quotient, dst, idx + 1, len, false);
        }
    }
    let half0 = unsafe { slice::from_raw_parts_mut(first_half.as_mut_ptr().cast::<F>(), half) };
    evaluate_full(half0, 0, len, allow_parallel);
    assert!(values.capacity() >= len);
    unsafe { values.set_len(len) };
    values
}

fn evaluate_batch_half0<F: BinaryField>(buffers: &mut [Vec<F>], half: usize, len: usize) {
    assert!(!buffers.is_empty() && half >= 1 && half.is_power_of_two());
    if half >= 2 * PARALLEL_THRESHOLD && half <= len {
        let q = half / 2;
        let mut tasks: Vec<(usize, &mut [F])> = Vec::with_capacity(buffers.len() * 2);
        for buf in buffers.iter_mut() {
            let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
            let (lo, hi) = h0.split_at_mut(q);
            F::xor_slice(hi, lo);
            tasks.push((0, lo));
            tasks.push((q, hi));
        }
        assert_eq!(tasks.len(), buffers.len() * 2);
        parallel::for_each_piece_mut(&mut tasks, 1, |_, piece| {
            for (start, slice) in piece {
                evaluate_full(slice, *start, len, false);
            }
        });
        return;
    }
    parallel::for_each_piece_mut(buffers, 1, |_, piece| {
        for buf in piece {
            let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
            evaluate_full(h0, 0, len, false);
        }
    });
}

fn encode_batch<F: BinaryField, const P: usize>(rows: &[[&[F]; P]], len: usize) -> Vec<Vec<F>> {
    assert!(!rows.is_empty() && len <= MAX_DOMAIN_SIZE);
    let count: usize = rows[0].iter().map(|part| part.len()).sum();
    let threads = parallel::threads();
    if threads == 1 || rows.len() <= 1 || count < 2 || rows.len() * len < 4 * PARALLEL_THRESHOLD {
        let allow_inner = rows.len() < threads;
        return parallel::map_range(rows.len(), 1, |row| encode_parts(&rows[row], len, allow_inner));
    }
    let block = count.next_power_of_two();
    let half = block / 2;
    let num_halves = len.div_ceil(half);
    if num_halves < 2 || rows.len() * num_halves < threads {
        let allow_inner = rows.len() < threads;
        return parallel::map_range(rows.len(), 1, |row| encode_parts(&rows[row], len, allow_inner));
    }
    let padded = num_halves * half;
    let prepared: Vec<(Vec<F>, Vec<F>)> = parallel::map_range(rows.len(), 1, |row| {
        let row_count: usize = rows[row].iter().map(|part| part.len()).sum();
        assert_eq!(row_count, count);
        prepare_novel_buffer(&rows[row], count, block, padded, false)
    });
    let (mut buffers, quotients): (Vec<Vec<F>>, Vec<Vec<F>>) = prepared.into_iter().unzip();
    {
        let mut novel_lows: Vec<&[F]> = Vec::with_capacity(buffers.len());
        let mut half_tasks: Vec<(usize, usize, &mut [MaybeUninit<F>])> =
            Vec::with_capacity(buffers.len() * (num_halves - 1));
        for (row_idx, buf) in buffers.iter_mut().enumerate() {
            let spare = &mut buf.spare_capacity_mut()[..padded];
            let (first_half, rest_halves) = spare.split_at_mut(half);
            let novel_low = unsafe { slice::from_raw_parts(first_half.as_ptr().cast::<F>(), half) };
            novel_lows.push(novel_low);
            for (idx, dst) in rest_halves.chunks_exact_mut(half).enumerate() {
                half_tasks.push((row_idx, idx + 1, dst));
            }
        }
        let novel_lows = &novel_lows;
        let quotients = &quotients;
        parallel::for_each_piece_mut(&mut half_tasks, 1, |_, piece| {
            for (row_idx, half_idx, dst) in piece {
                evaluate_half_from_novel(
                    novel_lows[*row_idx],
                    &quotients[*row_idx],
                    dst,
                    *half_idx,
                    len,
                    false,
                );
            }
        });
    }
    evaluate_batch_half0(&mut buffers, half, len);
    for buf in &mut buffers {
        assert!(buf.capacity() >= len);
        unsafe { buf.set_len(len) };
    }
    assert_eq!(buffers.len(), rows.len());
    buffers
}

fn compute_offsets(m: usize) -> ([usize; BASIS_SIZE], usize) {
    assert!(m < BASIS_SIZE);
    let mut offsets = [0usize; BASIS_SIZE];
    let mut len = 0usize;
    for j in 0..m {
        if j & m == j {
            offsets[len] = 1 << j;
            len += 1;
        }
    }
    assert!(len <= m);
    (offsets, len)
}

fn xor_offsets_range<F: Field>(
    coefficients: &mut [F],
    start: usize,
    end: usize,
    half: usize,
    offsets: &[usize],
    allow_parallel: bool,
) {
    let span = end - start;
    assert!(start >= half && span > 0);
    if allow_parallel && span >= 8 * PARALLEL_THRESHOLD {
        for &offset in offsets {
            let (target, source) = coefficients.split_at_mut(start);
            let target = &mut target[start - half + offset..][..span];
            let source = &source[..span];
            parallel::for_each_piece_mut(target, 8 * MIN_PIECE, |t0, piece| {
                F::xor_slice(piece, &source[t0..t0 + piece.len()]);
            });
        }
        return;
    }
    let (target_all, source_all) = coefficients.split_at_mut(start);
    let source = &source_all[..span];
    let base = start - half;
    let tiles = span.div_ceil(NOVEL_TILE);
    for tile in 0..tiles {
        let t0 = tile * NOVEL_TILE;
        let t1 = (t0 + NOVEL_TILE).min(span);
        let src = &source[t0..t1];
        for &offset in offsets {
            let dst = &mut target_all[base + offset + t0..base + offset + t1];
            F::xor_slice(dst, src);
        }
    }
    assert!(tiles * NOVEL_TILE >= span);
}

fn to_novel_basis_slice<F: Field>(coefficients: &mut [F], block: usize, allow_parallel: bool) {
    let count = coefficients.len();
    assert!(block.is_power_of_two() && count <= block);
    if block <= 2 || count <= 2 {
        return;
    }
    if block == 4 && count == 4 {
        coefficients[2] += coefficients[3];
        coefficients[1] += coefficients[2];
        return;
    }
    if block == 8 && count == 8 {
        coefficients[2] += coefficients[5];
        coefficients[3] += coefficients[6];
        coefficients[4] += coefficients[7];
        coefficients[1] += coefficients[4];
        coefficients[2] += coefficients[3];
        coefficients[1] += coefficients[2];
        coefficients[6] += coefficients[7];
        coefficients[5] += coefficients[6];
        return;
    }
    let half = block / 2;
    if count <= half {
        return to_novel_basis_slice(coefficients, half, allow_parallel);
    }
    let m = half.trailing_zeros() as usize;
    let (offsets, offsets_len) = compute_offsets(m);
    let offsets = &offsets[..offsets_len];
    let gap = half - offsets.last().copied().unwrap_or(0);
    assert!(gap >= 1);
    let mut end = count;
    for _ in 0..count {
        if end <= half {
            break;
        }
        let start = half.max(end.saturating_sub(gap));
        xor_offsets_range(coefficients, start, end, half, offsets, allow_parallel);
        end = start;
    }
    assert_eq!(end, half);
    let (remainder, quotient) = coefficients.split_at_mut(half);
    if !allow_parallel || half < PARALLEL_THRESHOLD {
        to_novel_basis_slice(remainder, half, allow_parallel);
        to_novel_basis_slice(quotient, half, allow_parallel);
    } else {
        parallel::join(
            || to_novel_basis_slice(remainder, half, true),
            || to_novel_basis_slice(quotient, half, true),
        );
    }
}

fn evaluate_full<F: BinaryField>(values: &mut [F], start: usize, len: usize, allow_parallel: bool) {
    let size = values.len();
    assert!(size.is_power_of_two() && start < len);
    if size == 1 {
        return;
    }
    if size <= L1_BLOCK && start + size <= len {
        return evaluate_l1(values, start);
    }
    if start + size <= len && size >= 4 && (!allow_parallel || size < 4 * PARALLEL_THRESHOLD) {
        let quarter = size / 4;
        let domain = Domain::get();
        let w = domain.point(start / (2 * quarter));
        let w0 = domain.point(start / quarter);
        let w1 = w0 ^ domain.byte_tables[0][2];
        let (low, high) = values.split_at_mut(2 * quarter);
        let (v0, v1) = low.split_at_mut(quarter);
        let (v2, v3) = high.split_at_mut(quarter);
        F::butterfly_radix4(v0, v1, v2, v3, w, w0, w1);
        if !allow_parallel || 2 * quarter < PARALLEL_THRESHOLD {
            evaluate_full(v0, start, len, allow_parallel);
            evaluate_full(v1, start + quarter, len, allow_parallel);
            evaluate_full(v2, start + 2 * quarter, len, allow_parallel);
            evaluate_full(v3, start + 3 * quarter, len, allow_parallel);
        } else {
            parallel::join(
                || {
                    evaluate_full(v0, start, len, true);
                    evaluate_full(v1, start + quarter, len, true);
                },
                || {
                    evaluate_full(v2, start + 2 * quarter, len, true);
                    evaluate_full(v3, start + 3 * quarter, len, true);
                },
            );
        }
        return;
    }
    let half = size / 2;
    let twiddle = Domain::get().point(start / half);
    let (low, high) = values.split_at_mut(half);
    if !allow_parallel || half < 2 * PARALLEL_THRESHOLD {
        F::butterfly_slice(low, high, twiddle);
    } else {
        parallel::for_each_pair_mut(low, high, MIN_PIECE, |l, h| F::butterfly_slice(l, h, twiddle));
    }
    let needs_high = start + half < len;
    if !allow_parallel || half < PARALLEL_THRESHOLD || !needs_high {
        evaluate_full(low, start, len, allow_parallel);
        if needs_high {
            evaluate_full(high, start + half, len, allow_parallel);
        }
    } else {
        parallel::join(
            || evaluate_full(low, start, len, true),
            || evaluate_full(high, start + half, len, true),
        );
    }
}

fn evaluate_l1<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!((2..=L1_BLOCK).contains(&size) && size.is_power_of_two());
    let domain = Domain::get();
    let p2 = domain.byte_tables[0][2];
    let mut half = size / 2;
    if half == 1 {
        let (low, high) = values.split_at_mut(1);
        F::butterfly_slice(low, high, domain.point(start));
        return;
    }
    if half == 2 {
        let w = domain.point(start / 2);
        let w0 = domain.point(start);
        let (low, high) = values.split_at_mut(2);
        let (v0, v1) = low.split_at_mut(1);
        let (v2, v3) = high.split_at_mut(1);
        F::butterfly_radix4(v0, v1, v2, v3, w, w0, w0 ^ p2);
        return;
    }
    if half >= 8 && !half.trailing_zeros().is_multiple_of(2) {
        let base = domain.point(start / half);
        for (i, block) in values.chunks_exact_mut(2 * half).enumerate() {
            let twiddle = base ^ domain.point_small(2 * i);
            let (low, high) = block.split_at_mut(half);
            F::butterfly_slice(low, high, twiddle);
        }
        half /= 2;
    }
    for _ in 0..16 {
        if half < 8 {
            break;
        }
        let quarter = half / 2;
        let base_w = domain.point(start / half);
        let base_w0 = domain.point(start / quarter);
        for (i, block) in values.chunks_exact_mut(2 * half).enumerate() {
            let w = base_w ^ domain.point_small(2 * i);
            let w0 = base_w0 ^ domain.point_small(4 * i);
            let (low, high) = block.split_at_mut(half);
            let (v0, v1) = low.split_at_mut(quarter);
            let (v2, v3) = high.split_at_mut(quarter);
            F::butterfly_radix4(v0, v1, v2, v3, w, w0, w0 ^ p2);
        }
        half /= 4;
    }
    assert_eq!(half, 4);
    F::butterfly_radix8_pass(
        values,
        domain.point(start / 4),
        domain.point(start / 2),
        domain.point(start),
        &domain.mul_tables[0],
        &domain.mul_tables[1],
    );
}
