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
const RADIX8_OUTER_MIN: usize = 1 << 17;
const NOVEL_SMALL: usize = 256;
const NOVEL_TILE: usize = 1024;

const fn build_novel_offsets() -> [([usize; BASIS_SIZE], usize); BASIS_SIZE] {
    let mut table = [([0usize; BASIS_SIZE], 0usize); BASIS_SIZE];
    let mut m = 0usize;
    while m < BASIS_SIZE {
        let mut len = 0usize;
        let mut j = 0usize;
        while j < m {
            if j & m == j {
                table[m].0[len] = 1usize << j;
                len += 1;
            }
            j += 1;
        }
        table[m].1 = len;
        m += 1;
    }
    table
}

const NOVEL_OFFSETS: [([usize; BASIS_SIZE], usize); BASIS_SIZE] = build_novel_offsets();

#[inline(always)]
fn novel_offsets(m: usize) -> &'static [usize] {
    debug_assert!(m < BASIS_SIZE);
    let (ref arr, len) = NOVEL_OFFSETS[m];
    &arr[..len]
}

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

fn evaluate_octants<F: BinaryField>(full: &mut [F], start: usize, len: usize, allow_parallel: bool) {
    let size = full.len();
    assert!(size >= 8 && size.is_power_of_two() && start + size <= len);
    let e = size / 8;
    if !allow_parallel || size < 4 * PARALLEL_THRESHOLD {
        for (k, chunk) in full.chunks_exact_mut(e).enumerate() {
            evaluate_full(chunk, start + k * e, len, allow_parallel);
        }
        return;
    }
    let half = size / 2;
    let (lo, hi) = full.split_at_mut(half);
    parallel::join(
        || {
            for (k, chunk) in lo.chunks_exact_mut(e).enumerate() {
                evaluate_full(chunk, start + k * e, len, true);
            }
        },
        || {
            for (k, chunk) in hi.chunks_exact_mut(e).enumerate() {
                evaluate_full(chunk, start + half + k * e, len, true);
            }
        },
    );
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
    let p2 = domain.byte_tables[0][2];
    let p4 = domain.byte_tables[0][4];
    if half >= RADIX8_OUTER_MIN && start + half <= len && (!allow_parallel || half < 8 * PARALLEL_THRESHOLD) {
        let w4 = domain.point(2 * half_index);
        let w2_0 = domain.point(4 * half_index);
        let w1_0 = domain.point(8 * half_index);
        let w1 = [w1_0, w1_0 ^ p2, w1_0 ^ p4, w1_0 ^ p2 ^ p4];
        F::copy_half_radix8_uninit(novel_low, novel_high, dst, w_eff, w4, [w2_0, w2_0 ^ p2], w1);
        let full = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), half) };
        evaluate_octants(full, start, len, allow_parallel);
        return;
    }
    if half >= 4 && start + half <= len && (!allow_parallel || half < 4 * PARALLEL_THRESHOLD) {
        let q = half / 4;
        let w = domain.point(2 * half_index);
        let w0 = domain.point(4 * half_index);
        F::copy_half_radix4_uninit(novel_low, novel_high, dst, w_eff, w, w0, w0 ^ p2);
        let full = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), half) };
        for (k, chunk) in full.chunks_exact_mut(q).enumerate() {
            evaluate_full(chunk, start + k * q, len, allow_parallel);
        }
        return;
    }
    F::copy_half_uninit(novel_low, novel_high, dst, w_eff);
    let full = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), half) };
    evaluate_full(full, start, len, allow_parallel);
}

fn prepare_novel_top<F: BinaryField>(
    parts: &[&[F]],
    count: usize,
    block: usize,
    padded: usize,
) -> (Vec<F>, Vec<F>) {
    assert!(count >= 2 && block >= 4 && padded >= block);
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
    novel_top_step(first_init, half, false);
    let mut quotient = first_init[half..count].to_vec();
    to_novel_basis_slice(&mut quotient, half, false);
    novel_top_step(&mut first_init[..half], half / 2, false);
    (values, quotient)
}

fn finish_batch_novel<F: BinaryField>(buffers: &mut [Vec<F>], half: usize) {
    assert!(!buffers.is_empty() && half >= 4 && half.is_power_of_two());
    let quarter = half / 2;
    let mut tasks: Vec<&mut [F]> = Vec::with_capacity(buffers.len() * 2);
    for buf in buffers.iter_mut() {
        let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
        let (lo, hi) = h0.split_at_mut(quarter);
        tasks.push(lo);
        tasks.push(hi);
    }
    assert_eq!(tasks.len(), buffers.len() * 2);
    parallel::for_each_piece_mut(&mut tasks, 1, |_, piece| {
        for slice in piece {
            to_novel_basis_slice(slice, quarter, false);
        }
    });
}

fn novel_top_step<F: Field>(coefficients: &mut [F], half: usize, allow_parallel: bool) {
    let count = coefficients.len();
    assert!(half >= 1 && half.is_power_of_two() && count > half && count <= 2 * half);
    let offsets = novel_offsets(half.trailing_zeros() as usize);
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
    let split_novel = half >= 2 * PARALLEL_THRESHOLD;
    let prepared: Vec<(Vec<F>, Vec<F>)> = parallel::map_range(rows.len(), 1, |row| {
        let row_count: usize = rows[row].iter().map(|part| part.len()).sum();
        assert_eq!(row_count, count);
        if split_novel {
            prepare_novel_top(&rows[row], count, block, padded)
        } else {
            prepare_novel_buffer(&rows[row], count, block, padded, false)
        }
    });
    let (mut buffers, quotients): (Vec<Vec<F>>, Vec<Vec<F>>) = prepared.into_iter().unzip();
    if split_novel {
        finish_batch_novel(&mut buffers, half);
    }
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

#[inline(always)]
fn to_novel_8<F: Field>(c: &mut [F; 8]) {
    let (lo, hi) = c.split_at_mut(5);
    F::xor_slice(&mut lo[2..5], hi);
    c[1] += c[4];
    c[2] += c[3];
    c[1] += c[2];
    c[6] += c[7];
    c[5] += c[6];
}

fn to_novel_basis_small<F: Field>(coefficients: &mut [F]) {
    let total = coefficients.len();
    assert!((4..=NOVEL_SMALL).contains(&total) && total.is_power_of_two());
    if total == 4 {
        coefficients[2] += coefficients[3];
        coefficients[1] += coefficients[2];
        return;
    }
    let mut size = total;
    for _ in 0..10 {
        if size <= 8 {
            break;
        }
        let half = size / 2;
        let offsets = novel_offsets(half.trailing_zeros() as usize);
        let gap = half - offsets.last().copied().unwrap_or(0);
        for chunk in coefficients.chunks_exact_mut(size) {
            let mut end = size;
            for _ in 0..size {
                if end <= half {
                    break;
                }
                let start = half.max(end.saturating_sub(gap));
                let span = end - start;
                let (target_all, source_all) = chunk.split_at_mut(start);
                let src = &source_all[..span];
                let base = start - half;
                for &offset in offsets {
                    F::xor_slice(&mut target_all[base + offset..base + offset + span], src);
                }
                end = start;
            }
        }
        size = half;
    }
    assert_eq!(size, 8);
    let (blocks, remainder) = coefficients.as_chunks_mut::<8>();
    assert!(remainder.is_empty());
    for block in blocks {
        to_novel_8(block);
    }
}

fn to_novel_basis_slice<F: Field>(coefficients: &mut [F], block: usize, allow_parallel: bool) {
    let count = coefficients.len();
    assert!(block.is_power_of_two() && count <= block);
    if block <= 2 || count <= 2 {
        return;
    }
    if count == block && block <= NOVEL_SMALL {
        return to_novel_basis_small(coefficients);
    }
    let half = block / 2;
    if count <= half {
        return to_novel_basis_slice(coefficients, half, allow_parallel);
    }
    let offsets = novel_offsets(half.trailing_zeros() as usize);
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

fn evaluate_quarters<F: BinaryField>(full: &mut [F], start: usize, len: usize, allow_parallel: bool) {
    let size = full.len();
    assert!(size >= 4 && size.is_power_of_two() && start + size <= len);
    let q = size / 4;
    if !allow_parallel || 2 * q < PARALLEL_THRESHOLD {
        for (k, chunk) in full.chunks_exact_mut(q).enumerate() {
            evaluate_full(chunk, start + k * q, len, allow_parallel);
        }
        return;
    }
    let (lo, hi) = full.split_at_mut(2 * q);
    let (v0, v1) = lo.split_at_mut(q);
    let (v2, v3) = hi.split_at_mut(q);
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

fn evaluate_full<F: BinaryField>(values: &mut [F], start: usize, len: usize, allow_parallel: bool) {
    let size = values.len();
    assert!(size.is_power_of_two() && start < len);
    if size == 1 {
        return;
    }
    if size <= L1_BLOCK && start + size <= len {
        return evaluate_l1(values, start);
    }
    if start + size <= len && size >= RADIX8_OUTER_MIN && (!allow_parallel || size < 8 * PARALLEL_THRESHOLD) {
        let e = size / 8;
        let domain = Domain::get();
        let p2 = domain.byte_tables[0][2];
        let p4 = domain.byte_tables[0][4];
        let w4 = domain.point(start / (4 * e));
        let w2_0 = domain.point(start / (2 * e));
        let w1_0 = domain.point(start / e);
        let w1 = [w1_0, w1_0 ^ p2, w1_0 ^ p4, w1_0 ^ p2 ^ p4];
        F::butterfly_radix8_outer(values, w4, w2_0, w2_0 ^ p2, w1);
        evaluate_octants(values, start, len, allow_parallel);
        return;
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
        evaluate_quarters(values, start, len, allow_parallel);
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

fn evaluate_l1_radix4<F: BinaryField>(values: &mut [F], start: usize, half: usize) {
    assert!(half >= 2 && half.is_power_of_two() && values.len().is_multiple_of(2 * half));
    let domain = Domain::get();
    let p2 = domain.byte_tables[0][2];
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
    assert!(quarter >= 1);
}

fn evaluate_l1<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!((2..=L1_BLOCK).contains(&size) && size.is_power_of_two());
    let domain = Domain::get();
    let mut half = size / 2;
    if half == 1 {
        let (low, high) = values.split_at_mut(1);
        F::butterfly_slice(low, high, domain.point(start));
        return;
    }
    if half == 2 {
        evaluate_l1_radix4(values, start, 2);
        return;
    }
    if half >= 8 && !half.trailing_zeros().is_multiple_of(2) {
        let base = domain.point(start / half);
        for (i, block) in values.chunks_exact_mut(2 * half).enumerate() {
            let (low, high) = block.split_at_mut(half);
            F::butterfly_slice(low, high, base ^ domain.point_small(2 * i));
        }
        half /= 2;
    }
    for _ in 0..16 {
        if half < 8 {
            break;
        }
        evaluate_l1_radix4(values, start, half);
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
