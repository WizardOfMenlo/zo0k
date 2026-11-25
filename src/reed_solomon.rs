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

    fn add_transposed_queries(dst: &mut [Self], queries: &[(usize, Self)]) {
        for &(pos, scale) in queries {
            Self::add_powers_slice(dst, Self::point(pos), scale);
        }
    }
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

    fn add_transposed_queries(dst: &mut [Self], queries: &[(usize, Self)]) {
        add_transposed_queries(dst, queries);
    }
}

const BASIS_SIZE: usize = 32;
const PARALLEL_THRESHOLD: usize = 1 << 14;
const MIN_PIECE: usize = 1 << 13;
const L1_BYTES: usize = 65536;
const RADIX8_OUTER_MIN: usize = 64;
const SLICE_ELEMS: usize = 1 << 14;
const SPLIT_HALF_BYTES: usize = 4 << 20;
const FUSED64_TILE_BYTES: usize = 4096;
const FUSED64_MIN_HALF_BYTES: usize = 2 << 20;
const FUSED64_SPLIT_ELEMS: usize = 4096;
const NOVEL_TILE: usize = 512;
const TRANSPOSE_MIN_LEN: usize = 4096;
const TRANSPOSE_MIN_QUERIES: usize = 12;

#[inline(always)]
fn l1_block_len<F: Field>() -> usize {
    (L1_BYTES / F::BYTES).max(2048)
}

#[inline(always)]
fn fused64_tile<F: Field>() -> usize {
    (FUSED64_TILE_BYTES / F::BYTES).max(16)
}

#[inline(always)]
fn fused64_oct_stride<F: Field>(tile: usize) -> usize {
    8 * tile + (128 / F::BYTES).max(1)
}

#[inline(always)]
fn use_fused64<F: Field>(half: usize) -> bool {
    let tile = fused64_tile::<F>();
    half >= 64 * tile && half * F::BYTES >= FUSED64_MIN_HALF_BYTES
}

#[inline(always)]
fn as_uninit_slice_mut<T: Copy>(values: &mut [T]) -> &mut [MaybeUninit<T>] {
    unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast::<MaybeUninit<T>>(), values.len()) }
}

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
    step_mul: Box<[[(u128, u128); 16]; 3]>,
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
        let point_of = |index: usize| -> u128 {
            byte_tables[0][index & 255]
                ^ byte_tables[1][(index >> 8) & 255]
                ^ byte_tables[2][(index >> 16) & 255]
                ^ byte_tables[3][(index >> 24) & 255]
        };
        let mut step_mul = Box::new([[(0u128, 0u128); 16]; 3]);
        for (s, row) in step_mul.iter_mut().enumerate() {
            for (tz, entry) in row.iter_mut().enumerate() {
                let idx = ((2usize << tz) - 1) << (s + 1);
                *entry = precompute_mul_pair(point_of(idx));
            }
        }
        Self { byte_tables, step_mul }
    }

    #[inline(always)]
    fn point(&self, index: usize) -> u128 {
        debug_assert!(index < MAX_DOMAIN_SIZE);
        let t = &self.byte_tables;
        t[0][index & 255] ^ t[1][index >> 8 & 255] ^ t[2][index >> 16 & 255] ^ t[3][index >> 24 & 255]
    }

    #[inline(always)]
    fn radix8_twiddles(&self, start: usize, e: usize) -> (u128, [u128; 2], [u128; 4]) {
        let p2 = self.byte_tables[0][2];
        let p4 = self.byte_tables[0][4];
        let w4 = self.point(start / (4 * e));
        let w2_0 = self.point(start / (2 * e));
        let w1_0 = self.point(start / e);
        (w4, [w2_0, w2_0 ^ p2], [w1_0, w1_0 ^ p2, w1_0 ^ p4, w1_0 ^ p2 ^ p4])
    }

    #[inline(always)]
    fn radix8_twiddle_pairs(&self, start: usize, e: usize) -> ([(u128, u128); 5], bool) {
        let w4 = self.point(start / (4 * e));
        let w2_0 = self.point(start / (2 * e));
        let w1_0 = self.point(start / e);
        let pairs = [
            precompute_mul_pair(w4),
            precompute_mul_pair(w2_0),
            precompute_mul_pair(w1_0),
            self.step_mul[0][0],
            self.step_mul[1][0],
        ];
        (pairs, w4 == 0 && w2_0 == 0 && w1_0 == 0)
    }
}

fn copy_to_uninit<T: Copy>(dst: &mut [MaybeUninit<T>], src: &[T]) {
    assert_eq!(dst.len(), src.len());
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().cast::<T>(), src.len());
    }
    assert_eq!(dst.len(), src.len());
}

fn alloc_len_for(half: usize, num_halves: usize, len: usize) -> (usize, usize) {
    assert!(half >= 1 && half.is_power_of_two() && num_halves >= 2);
    let last_start = (num_halves - 1) * half;
    let last_need = len - last_start;
    assert!(last_need >= 1 && last_need <= half);
    if half >= RADIX8_OUTER_MIN {
        let e = half / 8;
        let last_octants = last_need.div_ceil(e);
        assert!((1..=8).contains(&last_octants));
        (last_start + last_octants * e, last_octants)
    } else {
        (num_halves * half, 8)
    }
}

fn evaluate_octants<F: BinaryField>(
    full: &mut [F],
    start: usize,
    e: usize,
    octants: usize,
    len: usize,
    allow_parallel: bool,
) {
    assert!((1..=8).contains(&octants) && full.len() == octants * e);
    assert!(e.is_power_of_two() && start < len);
    if !allow_parallel || full.len() < 4 * PARALLEL_THRESHOLD || octants < 2 {
        for (k, chunk) in full.chunks_exact_mut(e).enumerate() {
            if start + k * e < len {
                evaluate_full(chunk, start + k * e, len, allow_parallel);
            }
        }
        return;
    }
    let split = octants / 2;
    let (lo, hi) = full.split_at_mut(split * e);
    parallel::join(
        || evaluate_octants(lo, start, e, split, len, true),
        || evaluate_octants(hi, start + split * e, e, octants - split, len, true),
    );
}

#[allow(clippy::too_many_arguments)]
fn copy_half_radix8_parallel<F: BinaryField>(
    novel_low: &[F],
    novel_high: &[F],
    dst: &mut [MaybeUninit<F>],
    octants: usize,
    w_eff: u128,
    w4: u128,
    w2: [u128; 2],
    w1: [u128; 4],
) {
    let e = novel_low.len() / 8;
    let slice_len = SLICE_ELEMS.min(e);
    let slices = e / slice_len;
    assert!(dst.len() == octants * e && slices * slice_len == e);
    assert!((1..=8).contains(&octants));
    let mut oct_iters: Vec<_> = dst.chunks_exact_mut(e).map(|o| o.chunks_exact_mut(slice_len)).collect();
    let mut tasks: Vec<(usize, Vec<&mut [MaybeUninit<F>]>)> = Vec::with_capacity(slices);
    for s in 0..slices {
        let mut out = Vec::with_capacity(octants);
        for iter in &mut oct_iters {
            out.push(iter.next().expect("slice"));
        }
        tasks.push((s * slice_len, out));
    }
    parallel::for_each_piece_mut(&mut tasks, 1, |_, piece| {
        for (off, dst_slices) in piece {
            let low_oct: [&[F]; 8] =
                std::array::from_fn(|k| &novel_low[k * e + *off..k * e + *off + slice_len]);
            let high_oct: [&[F]; 8] = std::array::from_fn(|k| {
                let st = k * e + *off;
                if st < novel_high.len() {
                    &novel_high[st..novel_high.len().min(st + slice_len)]
                } else {
                    &[]
                }
            });
            F::copy_half_radix8_octants_uninit(low_oct, high_oct, dst_slices, w_eff, w4, w2, w1);
        }
    });
}

fn evaluate_sub_octants<F: BinaryField>(
    full: &mut [F],
    start: usize,
    e2: usize,
    sub_octants: usize,
    len: usize,
    allow_parallel: bool,
) {
    assert!((1..=64).contains(&sub_octants) && full.len() == sub_octants * e2);
    assert!(e2.is_power_of_two() && start < len);
    if allow_parallel && full.len() >= 4 * PARALLEL_THRESHOLD && sub_octants >= 2 {
        parallel::for_each_chunk_mut(full, e2, e2, |idx, chunk| {
            let sub_start = start + idx * e2;
            if sub_start < len {
                evaluate_full(chunk, sub_start, len, false);
            }
        });
        return;
    }
    for (idx, chunk) in full.chunks_exact_mut(e2).enumerate() {
        let sub_start = start + idx * e2;
        if sub_start < len {
            evaluate_full(chunk, sub_start, len, allow_parallel);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_radix64_step1_tile<F: BinaryField>(
    novel_low: &[F],
    novel_high: &[F],
    scratch: &mut [MaybeUninit<F>],
    octants: usize,
    e: usize,
    e2: usize,
    off: usize,
    tile: usize,
    oct_stride: usize,
    w_eff: u128,
    a_pairs: &[(u128, u128); 5],
    a_zero: bool,
    a_raw: (u128, [u128; 2], [u128; 4]),
) {
    assert!((1..=8).contains(&octants) && off + tile <= e2);
    assert!(scratch.len() >= 8 * oct_stride && 8 * tile <= oct_stride);
    let (aw4, aw2, aw1) = a_raw;
    for s in 0..8 {
        let pos = s * e2 + off;
        if pos >= novel_high.len() {
            F::copy_radix8_strided_uninit(
                &novel_low[pos..],
                e,
                &mut scratch[s * tile..],
                oct_stride,
                tile,
                a_pairs,
                a_zero,
            );
        } else {
            let low_oct: [&[F]; 8] = std::array::from_fn(|k| &novel_low[k * e + pos..k * e + pos + tile]);
            let high_oct: [&[F]; 8] = std::array::from_fn(|k| {
                let st = k * e + pos;
                if st < novel_high.len() { &novel_high[st..novel_high.len().min(st + tile)] } else { &[] }
            });
            let mut dst_vec: [&mut [MaybeUninit<F>]; 8] = Default::default();
            let mut rest = &mut *scratch;
            for slot in dst_vec.iter_mut().take(octants) {
                let (head, tail) = rest.split_at_mut(oct_stride);
                *slot = &mut head[s * tile..s * tile + tile];
                rest = tail;
            }
            F::copy_half_radix8_octants_uninit(
                low_oct,
                high_oct,
                &mut dst_vec[..octants],
                w_eff,
                aw4,
                aw2,
                aw1,
            );
        }
    }
}

fn evaluate_half_fused64<F: BinaryField>(
    novel_low: &[F],
    novel_high: &[F],
    dst: &mut [MaybeUninit<F>],
    scratch: &mut [MaybeUninit<F>],
    half_index: usize,
    len: usize,
    allow_parallel: bool,
) {
    let half = novel_low.len();
    let e = half / 8;
    let e2 = e / 8;
    let octants = dst.len() / e;
    let tile = fused64_tile::<F>();
    let oct_stride = fused64_oct_stride::<F>(tile);
    assert!(e2 >= tile && e2.is_multiple_of(tile) && (1..=8).contains(&octants));
    assert!(dst.len() == octants * e && scratch.len() >= 8 * oct_stride);
    let start = half_index * half;
    let domain = Domain::get();
    let w_eff = domain.point(half_index);
    let a_raw = domain.radix8_twiddles(start, e);
    let (a_pairs, a_zero) = domain.radix8_twiddle_pairs(start, e);
    let b_tw: [([(u128, u128); 5], bool); 8] =
        std::array::from_fn(|k| domain.radix8_twiddle_pairs(start + k * e, e2));
    let blocks = e2 / tile;
    for block in 0..blocks {
        let off = block * tile;
        copy_radix64_step1_tile(
            novel_low, novel_high, scratch, octants, e, e2, off, tile, oct_stride, w_eff, &a_pairs, a_zero,
            a_raw,
        );
        for (k, (b_pairs, b_zero)) in b_tw.iter().enumerate().take(octants) {
            let src_k =
                unsafe { slice::from_raw_parts(scratch.as_ptr().add(k * oct_stride).cast::<F>(), 8 * tile) };
            F::copy_radix8_strided_uninit(src_k, tile, &mut dst[k * e + off..], e2, tile, b_pairs, *b_zero);
        }
    }
    let full = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), dst.len()) };
    evaluate_sub_octants(full, start, e2, octants * 8, len, allow_parallel);
}

fn evaluate_half_from_novel<F: BinaryField>(
    novel_low: &[F],
    novel_high: &[F],
    dst: &mut [MaybeUninit<F>],
    half_index: usize,
    len: usize,
    allow_parallel: bool,
) {
    let half = novel_low.len();
    let start = half_index * half;
    assert!(half >= 1 && half.is_power_of_two() && start < len);
    assert!(novel_high.len() <= half);
    if !allow_parallel && use_fused64::<F>(half) {
        let oct_stride = fused64_oct_stride::<F>(fused64_tile::<F>());
        let mut scratch = vec![MaybeUninit::uninit(); 8 * oct_stride];
        evaluate_half_fused64(novel_low, novel_high, dst, &mut scratch, half_index, len, false);
        return;
    }
    let domain = Domain::get();
    let w_eff = domain.point(half_index);
    if half >= RADIX8_OUTER_MIN {
        let e = half / 8;
        let octants = dst.len() / e;
        assert!(dst.len() == octants * e && (1..=8).contains(&octants));
        let (w4, w2, w1) = domain.radix8_twiddles(start, e);
        if allow_parallel && e >= 2 * SLICE_ELEMS {
            copy_half_radix8_parallel(novel_low, novel_high, dst, octants, w_eff, w4, w2, w1);
        } else {
            F::copy_half_radix8_partial_uninit(novel_low, novel_high, dst, octants, w_eff, w4, w2, w1);
        }
        let full = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), dst.len()) };
        evaluate_octants(full, start, e, octants, len, allow_parallel);
        return;
    }
    assert_eq!(dst.len(), half);
    if half >= 4 && start + half <= len && (!allow_parallel || half < 4 * PARALLEL_THRESHOLD) {
        let q = half / 4;
        let w = domain.point(2 * half_index);
        let w0 = domain.point(4 * half_index);
        let p2 = domain.byte_tables[0][2];
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

#[inline(always)]
fn split_p(k: usize) -> usize {
    assert!(k >= 4);
    let p = 1usize << (usize::BITS - 1 - (k - 1).leading_zeros());
    assert!(p < k && p.is_power_of_two());
    p
}

fn xor_block_stripe<F: Field>(
    c: &mut [F],
    base: usize,
    dst_idx: usize,
    src_idx: usize,
    stride: usize,
    len: usize,
) {
    assert_ne!(dst_idx, src_idx);
    assert!(len <= stride);
    let dst = base + dst_idx * stride;
    let src = base + src_idx * stride;
    if dst < src {
        let (left, right) = c.split_at_mut(src);
        F::xor_slice(&mut left[dst..dst + len], &right[..len]);
    } else {
        let (left, right) = c.split_at_mut(dst);
        F::xor_slice(&mut right[..len], &left[src..src + len]);
    }
}

fn blocks_novel_stripe_forward<F: Field>(c: &mut [F], base: usize, m: usize, stride: usize, len: usize) {
    assert!(m.is_power_of_two() && len > 0);
    assert!(len <= stride && base + (m - 1) * stride + len <= c.len());
    if m <= 2 {
        return;
    }
    if m == 4 {
        xor_block_stripe(c, base, 2, 3, stride, len);
        xor_block_stripe(c, base, 1, 2, stride, len);
        return;
    }
    if m == 8 {
        xor_block_stripe(c, base, 2, 5, stride, len);
        xor_block_stripe(c, base, 3, 6, stride, len);
        xor_block_stripe(c, base, 4, 7, stride, len);
        xor_block_stripe(c, base, 1, 4, stride, len);
        xor_block_stripe(c, base, 2, 3, stride, len);
        xor_block_stripe(c, base, 1, 2, stride, len);
        xor_block_stripe(c, base, 6, 7, stride, len);
        xor_block_stripe(c, base, 5, 6, stride, len);
        return;
    }
    let km = m.trailing_zeros() as usize;
    let pm = split_p(km);
    let bm = 1usize << pm;
    let mm = m / bm;
    for r in (0..(km - pm)).rev() {
        let sh = bm << r;
        let shf = 1usize << r;
        let groups = m / (2 * sh);
        for g in 0..groups {
            let b0 = g * 2 * sh;
            for j in 0..(sh - shf) {
                xor_block_stripe(c, base, b0 + 2 * shf + j, b0 + sh + shf + j, stride, len);
            }
            for j in 0..shf {
                xor_block_stripe(c, base, b0 + shf + j, b0 + sh + j, stride, len);
            }
        }
    }
    if mm > 2 {
        for r_idx in 0..bm {
            blocks_novel_stripe_forward(c, base + r_idx * stride, mm, bm * stride, len);
        }
    }
    for g in 0..mm {
        blocks_novel_stripe_forward(c, base + g * bm * stride, bm, stride, len);
    }
}

fn blocks_novel_stripe_transpose<F: Field>(c: &mut [F], base: usize, m: usize, stride: usize, len: usize) {
    assert!(m.is_power_of_two() && len > 0);
    assert!(len <= stride && base + (m - 1) * stride + len <= c.len());
    if m <= 2 {
        return;
    }
    if m == 4 {
        xor_block_stripe(c, base, 2, 1, stride, len);
        xor_block_stripe(c, base, 3, 2, stride, len);
        return;
    }
    if m == 8 {
        xor_block_stripe(c, base, 6, 5, stride, len);
        xor_block_stripe(c, base, 7, 6, stride, len);
        xor_block_stripe(c, base, 2, 1, stride, len);
        xor_block_stripe(c, base, 3, 2, stride, len);
        xor_block_stripe(c, base, 4, 1, stride, len);
        xor_block_stripe(c, base, 5, 2, stride, len);
        xor_block_stripe(c, base, 6, 3, stride, len);
        xor_block_stripe(c, base, 7, 4, stride, len);
        return;
    }
    let km = m.trailing_zeros() as usize;
    let pm = split_p(km);
    let bm = 1usize << pm;
    let mm = m / bm;
    for g in 0..mm {
        blocks_novel_stripe_transpose(c, base + g * bm * stride, bm, stride, len);
    }
    if mm > 2 {
        for r_idx in 0..bm {
            blocks_novel_stripe_transpose(c, base + r_idx * stride, mm, bm * stride, len);
        }
    }
    for r in 0..(km - pm) {
        let sh = bm << r;
        let shf = 1usize << r;
        let groups = m / (2 * sh);
        for g in 0..groups {
            let b0 = g * 2 * sh;
            for j in 0..shf {
                xor_block_stripe(c, base, b0 + sh + j, b0 + shf + j, stride, len);
            }
            for j in 0..(sh - shf) {
                xor_block_stripe(c, base, b0 + sh + shf + j, b0 + 2 * shf + j, stride, len);
            }
        }
    }
}

fn xor_slice_maybe_parallel<F: Field>(dst: &mut [F], src: &[F], allow_parallel: bool) {
    assert_eq!(dst.len(), src.len());
    if allow_parallel && dst.len() >= 4 * PARALLEL_THRESHOLD {
        parallel::for_each_piece_mut(dst, MIN_PIECE, |t0, piece| {
            F::xor_slice(piece, &src[t0..t0 + piece.len()]);
        });
    } else {
        F::xor_slice(dst, src);
    }
}

fn taylor_binom_forward<F: Field>(c: &mut [F], k: usize, p: usize, allow_parallel: bool) {
    let b = 1usize << p;
    assert_eq!(c.len(), 1usize << k);
    assert!(p < k && b >= 4);
    for r in (0..(k - p)).rev() {
        let step_half = b << r;
        let shift = 1usize << r;
        let chunk_len = 2 * step_half;
        if allow_parallel && c.len() >= 4 * PARALLEL_THRESHOLD && c.len() / chunk_len >= 4 {
            parallel::for_each_chunk_mut(c, chunk_len, chunk_len, |_, chunk| {
                let (lo, hi) = chunk.split_at_mut(step_half + shift);
                F::xor_slice(&mut lo[2 * shift..], hi);
                let (lo2, hi2) = chunk.split_at_mut(step_half);
                F::xor_slice(&mut lo2[shift..2 * shift], &hi2[..shift]);
            });
        } else {
            for chunk in c.chunks_exact_mut(chunk_len) {
                let (lo, hi) = chunk.split_at_mut(step_half + shift);
                xor_slice_maybe_parallel(&mut lo[2 * shift..], hi, allow_parallel);
                let (lo2, hi2) = chunk.split_at_mut(step_half);
                F::xor_slice(&mut lo2[shift..2 * shift], &hi2[..shift]);
            }
        }
    }
}

fn taylor_binom_transpose<F: Field>(c: &mut [F], k: usize, p: usize, allow_parallel: bool) {
    let b = 1usize << p;
    assert_eq!(c.len(), 1usize << k);
    assert!(p < k && b >= 4);
    for r in 0..(k - p) {
        let step_half = b << r;
        let shift = 1usize << r;
        let chunk_len = 2 * step_half;
        if allow_parallel && c.len() >= 4 * PARALLEL_THRESHOLD && c.len() / chunk_len >= 4 {
            parallel::for_each_chunk_mut(c, chunk_len, chunk_len, |_, chunk| {
                let (lo2, hi2) = chunk.split_at_mut(step_half);
                F::xor_slice(&mut hi2[..shift], &lo2[shift..2 * shift]);
                let (lo, hi) = chunk.split_at_mut(step_half + shift);
                F::xor_slice(hi, &lo[2 * shift..]);
            });
        } else {
            for chunk in c.chunks_exact_mut(chunk_len) {
                let (lo2, hi2) = chunk.split_at_mut(step_half);
                F::xor_slice(&mut hi2[..shift], &lo2[shift..2 * shift]);
                let (lo, hi) = chunk.split_at_mut(step_half + shift);
                xor_slice_maybe_parallel(hi, &lo[2 * shift..], allow_parallel);
            }
        }
    }
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

#[inline(always)]
fn to_novel_8_transpose<F: Field>(c: &mut [F; 8]) {
    c[6] += c[5];
    c[7] += c[6];
    c[2] += c[1];
    c[3] += c[2];
    c[4] += c[1];
    let (lo, hi) = c.split_at_mut(5);
    F::xor_slice(hi, &lo[2..5]);
}

fn to_novel_pow2_top<F: Field>(c: &mut [F], allow_parallel: bool) -> usize {
    let n = c.len();
    assert!(n >= 16 && n.is_power_of_two());
    let k = n.trailing_zeros() as usize;
    let p = split_p(k);
    let b = 1usize << p;
    let m = n / b;
    taylor_binom_forward(c, k, p, allow_parallel);
    if m > 2 {
        let tile = NOVEL_TILE.min(b);
        let tiles = b / tile;
        assert_eq!(tiles * tile, b);
        for t in 0..tiles {
            blocks_novel_stripe_forward(c, t * tile, m, b, tile);
        }
    }
    b
}

fn to_novel_basis_pow2<F: Field>(c: &mut [F], allow_parallel: bool) {
    let n = c.len();
    assert!(n >= 1 && n.is_power_of_two());
    if n <= 2 {
        return;
    }
    if n == 4 {
        c[2] += c[3];
        c[1] += c[2];
        return;
    }
    if n == 8 {
        let arr: &mut [F; 8] = c.try_into().expect("eight elements");
        to_novel_8(arr);
        return;
    }
    let b = to_novel_pow2_top(c, allow_parallel);
    if allow_parallel && n >= 2 * PARALLEL_THRESHOLD {
        parallel::for_each_chunk_mut(c, b, b, |_, chunk| {
            to_novel_basis_pow2(chunk, false);
        });
    } else {
        for chunk in c.chunks_exact_mut(b) {
            to_novel_basis_pow2(chunk, false);
        }
    }
}

fn to_novel_basis_pow2_transpose<F: Field>(c: &mut [F], allow_parallel: bool) {
    let n = c.len();
    assert!(n >= 1 && n.is_power_of_two());
    if n <= 2 {
        return;
    }
    if n == 4 {
        c[2] += c[1];
        c[3] += c[2];
        return;
    }
    if n == 8 {
        let arr: &mut [F; 8] = c.try_into().expect("eight elements");
        to_novel_8_transpose(arr);
        return;
    }
    let k = n.trailing_zeros() as usize;
    let p = split_p(k);
    let b = 1usize << p;
    let m = n / b;
    if allow_parallel && n >= 2 * PARALLEL_THRESHOLD {
        parallel::for_each_chunk_mut(c, b, b, |_, chunk| {
            to_novel_basis_pow2_transpose(chunk, false);
        });
    } else {
        for chunk in c.chunks_exact_mut(b) {
            to_novel_basis_pow2_transpose(chunk, false);
        }
    }
    if m > 2 {
        let tile = NOVEL_TILE.min(b);
        let tiles = b / tile;
        assert_eq!(tiles * tile, b);
        for t in 0..tiles {
            blocks_novel_stripe_transpose(c, t * tile, m, b, tile);
        }
    }
    taylor_binom_transpose(c, k, p, allow_parallel);
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

fn to_novel_basis_slice<F: Field>(coefficients: &mut [F], block: usize, allow_parallel: bool) {
    let count = coefficients.len();
    assert!(block.is_power_of_two() && count <= block);
    if block <= 2 || count <= 2 {
        return;
    }
    if count == block {
        return to_novel_basis_pow2(coefficients, allow_parallel);
    }
    let half = block / 2;
    if count <= half {
        return to_novel_basis_slice(coefficients, half, allow_parallel);
    }
    novel_top_step(coefficients, half, allow_parallel);
    let (remainder, quotient) = coefficients.split_at_mut(half);
    if !allow_parallel || half < PARALLEL_THRESHOLD {
        to_novel_basis_pow2(remainder, allow_parallel);
        to_novel_basis_slice(quotient, half, allow_parallel);
    } else {
        parallel::join(
            || to_novel_basis_pow2(remainder, true),
            || to_novel_basis_slice(quotient, half, true),
        );
    }
}

fn prepare_novel_buffer<F: BinaryField>(
    parts: &[&[F]],
    count: usize,
    block: usize,
    alloc_len: usize,
    allow_parallel: bool,
) -> (Vec<F>, Vec<F>) {
    assert!(count >= 2 && block >= 2 && alloc_len >= count);
    let half = block / 2;
    let mut values: Vec<F> = Vec::with_capacity(alloc_len);
    let spare = &mut values.spare_capacity_mut()[..alloc_len];
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

fn prepare_novel_factored<F: BinaryField>(
    parts: &[&[F]],
    count: usize,
    block: usize,
    alloc_len: usize,
) -> (Vec<F>, Vec<F>, usize) {
    assert!(count >= 2 && block >= 32 && alloc_len >= count);
    let half = block / 2;
    let mut values: Vec<F> = Vec::with_capacity(alloc_len);
    let spare = &mut values.spare_capacity_mut()[..alloc_len];
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
    let inner_b = to_novel_pow2_top(&mut first_init[..half], false);
    (values, quotient, inner_b)
}

fn finish_batch_novel_blocks<F: BinaryField>(buffers: &mut [Vec<F>], half: usize, inner_b: usize) {
    assert!(!buffers.is_empty() && inner_b >= 4 && half.is_multiple_of(inner_b));
    let per_row = half / inner_b;
    let mut tasks: Vec<&mut [F]> = Vec::with_capacity(buffers.len() * per_row);
    for buf in buffers.iter_mut() {
        let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
        for chunk in h0.chunks_exact_mut(inner_b) {
            tasks.push(chunk);
        }
    }
    assert_eq!(tasks.len(), buffers.len() * per_row);
    parallel::for_each_piece_mut(&mut tasks, 1, |_, piece| {
        for slice in piece {
            to_novel_basis_pow2(slice, false);
        }
    });
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
    let (alloc_len, _) = alloc_len_for(half, num_halves, len);
    let (mut values, quotient) = prepare_novel_buffer(parts, count, block, alloc_len, allow_parallel);
    let spare = &mut values.spare_capacity_mut()[..alloc_len];
    let (first_half, rest_halves) = spare.split_at_mut(half);
    let novel_low = unsafe { slice::from_raw_parts(first_half.as_ptr().cast::<F>(), half) };
    let inner_par = allow_parallel && num_halves < parallel::threads();
    if allow_parallel && !rest_halves.is_empty() {
        parallel::for_each_chunk_mut(rest_halves, half, MIN_PIECE.min(half), |idx, dst| {
            evaluate_half_from_novel(novel_low, &quotient, dst, idx + 1, len, inner_par);
        });
    } else {
        for (idx, dst) in rest_halves.chunks_mut(half).enumerate() {
            evaluate_half_from_novel(novel_low, &quotient, dst, idx + 1, len, false);
        }
    }
    let half0 = unsafe { slice::from_raw_parts_mut(first_half.as_mut_ptr().cast::<F>(), half) };
    evaluate_full(half0, 0, len, allow_parallel);
    assert!(values.capacity() >= len);
    unsafe { values.set_len(len) };
    values
}

fn run_half0_fused64_task<F: BinaryField>(
    subs: &mut [[&mut [F]; 8]; 8],
    scratch: &mut [MaybeUninit<F>],
    slice_len: usize,
    a_tw: &([(u128, u128); 5], bool),
    b_tw: &[([(u128, u128); 5], bool); 8],
) {
    let tile = fused64_tile::<F>();
    let oct_stride = fused64_oct_stride::<F>(tile);
    assert!(slice_len.is_multiple_of(tile) && scratch.len() >= 8 * oct_stride);
    assert_eq!(subs[0][0].len(), slice_len);
    let (a_pairs, a_zero) = a_tw;
    let blocks = slice_len / tile;
    for block in 0..blocks {
        let local = block * tile;
        for s in 0..8 {
            let src8: [&[F]; 8] = std::array::from_fn(|k| &subs[k][s][local..local + tile]);
            F::copy_radix8_to_strided_uninit(src8, &mut scratch[s * tile..], oct_stride, a_pairs, *a_zero);
        }
        for (k, (b_pairs, b_zero)) in b_tw.iter().enumerate() {
            let src_k =
                unsafe { slice::from_raw_parts(scratch.as_ptr().add(k * oct_stride).cast::<F>(), 8 * tile) };
            let [d0, d1, d2, d3, d4, d5, d6, d7] = &mut subs[k];
            let out8 = [
                as_uninit_slice_mut(&mut d0[local..local + tile]),
                as_uninit_slice_mut(&mut d1[local..local + tile]),
                as_uninit_slice_mut(&mut d2[local..local + tile]),
                as_uninit_slice_mut(&mut d3[local..local + tile]),
                as_uninit_slice_mut(&mut d4[local..local + tile]),
                as_uninit_slice_mut(&mut d5[local..local + tile]),
                as_uninit_slice_mut(&mut d6[local..local + tile]),
                as_uninit_slice_mut(&mut d7[local..local + tile]),
            ];
            F::copy_radix8_from_strided_uninit(src_k, tile, out8, b_pairs, *b_zero);
        }
    }
}

fn evaluate_batch_half0_fused64<F: BinaryField>(buffers: &mut [Vec<F>], half: usize, len: usize) {
    let e = half / 8;
    let e2 = e / 8;
    let slice_len = FUSED64_SPLIT_ELEMS.min(e2);
    let slices = e2 / slice_len;
    let oct_stride = fused64_oct_stride::<F>(fused64_tile::<F>());
    assert!(!buffers.is_empty() && slices >= 1 && slices * slice_len == e2);
    assert!(half <= len && half.is_multiple_of(64));
    let domain = Domain::get();
    let a_tw = domain.radix8_twiddle_pairs(0, e);
    let b_tw: [([(u128, u128); 5], bool); 8] =
        std::array::from_fn(|k| domain.radix8_twiddle_pairs(k * e, e2));
    {
        let mut tasks: Vec<[[&mut [F]; 8]; 8]> = Vec::with_capacity(buffers.len() * slices);
        for buf in buffers.iter_mut() {
            let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
            let mut iters: [[_; 8]; 8] = split_octants_mut(h0).map(|oct| {
                let mut subs = oct.chunks_exact_mut(e2);
                std::array::from_fn(|_| subs.next().expect("sub").chunks_exact_mut(slice_len))
            });
            for _ in 0..slices {
                tasks.push(std::array::from_fn(|k| {
                    std::array::from_fn(|s| iters[k][s].next().expect("slice"))
                }));
            }
        }
        parallel::for_each_piece_mut_with(
            &mut tasks,
            1,
            || vec![MaybeUninit::uninit(); 8 * oct_stride],
            |scratch, _, piece| {
                for subs in piece {
                    run_half0_fused64_task(subs, scratch, slice_len, &a_tw, &b_tw);
                }
            },
        );
    }
    let mut sub_tasks: Vec<(usize, &mut [F])> = Vec::with_capacity(buffers.len() * 64);
    for buf in buffers.iter_mut() {
        let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
        for (idx, chunk) in h0.chunks_exact_mut(e2).enumerate() {
            sub_tasks.push((idx * e2, chunk));
        }
    }
    parallel::for_each_piece_mut(&mut sub_tasks, 1, |_, piece| {
        for (start, chunk) in piece {
            evaluate_full(chunk, *start, len, false);
        }
    });
}

fn evaluate_batch_half0<F: BinaryField>(buffers: &mut [Vec<F>], half: usize, len: usize) {
    assert!(!buffers.is_empty() && half >= 1 && half.is_power_of_two());
    if use_fused64::<F>(half) && half <= len {
        return evaluate_batch_half0_fused64(buffers, half, len);
    }
    if buffers.len() < 2 * parallel::threads() && half >= RADIX8_OUTER_MIN && half <= len {
        let e = half / 8;
        let domain = Domain::get();
        let (w4, w2, w1) = domain.radix8_twiddles(0, e);
        let slice_len = if e >= 2 * SLICE_ELEMS { SLICE_ELEMS } else { e };
        let slices_per_row = e / slice_len;
        assert_eq!(slices_per_row * slice_len, e);
        {
            let mut r8_tasks: Vec<[&mut [F]; 8]> = Vec::with_capacity(buffers.len() * slices_per_row);
            for buf in buffers.iter_mut() {
                let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
                let mut iters: [_; 8] = split_octants_mut(h0).map(|s| s.chunks_exact_mut(slice_len));
                for _ in 0..slices_per_row {
                    r8_tasks.push(std::array::from_fn(|k| iters[k].next().expect("slice")));
                }
            }
            parallel::for_each_piece_mut(&mut r8_tasks, 1, |_, piece| {
                for task in piece {
                    let [o0, o1, o2, o3, o4, o5, o6, o7] = task;
                    F::butterfly_radix8_octants([o0, o1, o2, o3, o4, o5, o6, o7], w4, w2, w1);
                }
            });
        }
        let mut oct_tasks: Vec<(usize, &mut [F])> = Vec::with_capacity(buffers.len() * 8);
        for buf in buffers.iter_mut() {
            let h0 = unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr(), half) };
            for (k, chunk) in h0.chunks_exact_mut(e).enumerate() {
                oct_tasks.push((k * e, chunk));
            }
        }
        parallel::for_each_piece_mut(&mut oct_tasks, 1, |_, piece| {
            for (start, chunk) in piece {
                evaluate_full(chunk, *start, len, false);
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

fn split_octants_mut<F>(values: &mut [F]) -> [&mut [F]; 8] {
    let e = values.len() / 8;
    assert_eq!(e * 8, values.len());
    let (a, b) = values.split_at_mut(4 * e);
    let (a0, a1) = a.split_at_mut(2 * e);
    let (b0, b1) = b.split_at_mut(2 * e);
    let (v0, v1) = a0.split_at_mut(e);
    let (v2, v3) = a1.split_at_mut(e);
    let (v4, v5) = b0.split_at_mut(e);
    let (v6, v7) = b1.split_at_mut(e);
    [v0, v1, v2, v3, v4, v5, v6, v7]
}

type CopySliceTask<'a, F> = (usize, usize, usize, Vec<&'a mut [MaybeUninit<F>]>);
type SplitFused64Task<'a, F> = (usize, usize, usize, Vec<[&'a mut [MaybeUninit<F>]; 8]>);

fn push_split_fused64_tasks<'a, F: BinaryField>(
    tasks: &mut Vec<SplitFused64Task<'a, F>>,
    row_idx: usize,
    rest_halves: &'a mut [MaybeUninit<F>],
    half: usize,
    slice_len: usize,
) {
    let e = half / 8;
    let e2 = e / 8;
    let slices = e2 / slice_len;
    assert!(slices >= 1 && slices * slice_len == e2);
    assert!(half.is_multiple_of(64));
    for (idx, dst) in rest_halves.chunks_mut(half).enumerate() {
        let octants = dst.len() / e;
        let mut oct_iters: Vec<[_; 8]> = dst
            .chunks_exact_mut(e)
            .map(|oct| {
                let mut subs = oct.chunks_exact_mut(e2);
                std::array::from_fn(|_| subs.next().expect("sub").chunks_exact_mut(slice_len))
            })
            .collect();
        for s in 0..slices {
            let mut out = Vec::with_capacity(octants);
            for iters in &mut oct_iters {
                out.push(std::array::from_fn(|j| iters[j].next().expect("slice")));
            }
            tasks.push((row_idx, idx + 1, s * slice_len, out));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_split_fused64_task<F: BinaryField>(
    novel_low: &[F],
    novel_high: &[F],
    scratch: &mut [MaybeUninit<F>],
    half_idx: usize,
    off: usize,
    dst_subs: &mut [[&mut [MaybeUninit<F>]; 8]],
    half: usize,
    slice_len: usize,
) {
    let e = half / 8;
    let e2 = e / 8;
    let octants = dst_subs.len();
    let tile = fused64_tile::<F>();
    let oct_stride = fused64_oct_stride::<F>(tile);
    assert!((1..=8).contains(&octants) && slice_len.is_multiple_of(tile));
    assert!(off + slice_len <= e2 && scratch.len() >= 8 * oct_stride);
    let start = half_idx * half;
    let domain = Domain::get();
    let w_eff = domain.point(half_idx);
    let a_raw = domain.radix8_twiddles(start, e);
    let (a_pairs, a_zero) = domain.radix8_twiddle_pairs(start, e);
    let b_tw: [([(u128, u128); 5], bool); 8] =
        std::array::from_fn(|k| domain.radix8_twiddle_pairs(start + k * e, e2));
    let blocks = slice_len / tile;
    for block in 0..blocks {
        let local = block * tile;
        copy_radix64_step1_tile(
            novel_low,
            novel_high,
            scratch,
            octants,
            e,
            e2,
            off + local,
            tile,
            oct_stride,
            w_eff,
            &a_pairs,
            a_zero,
            a_raw,
        );
        for (k, sub8) in dst_subs.iter_mut().enumerate() {
            let (b_pairs, b_zero) = &b_tw[k];
            let src_k =
                unsafe { slice::from_raw_parts(scratch.as_ptr().add(k * oct_stride).cast::<F>(), 8 * tile) };
            let [d0, d1, d2, d3, d4, d5, d6, d7] = sub8;
            let out8 = [
                &mut d0[local..local + tile],
                &mut d1[local..local + tile],
                &mut d2[local..local + tile],
                &mut d3[local..local + tile],
                &mut d4[local..local + tile],
                &mut d5[local..local + tile],
                &mut d6[local..local + tile],
                &mut d7[local..local + tile],
            ];
            F::copy_radix8_from_strided_uninit(src_k, tile, out8, b_pairs, *b_zero);
        }
    }
}

fn evaluate_batch_rest_split_fused64<F: BinaryField>(
    buffers: &mut [Vec<F>],
    quotients: &[Vec<F>],
    half: usize,
    alloc_len: usize,
    len: usize,
) {
    let e2 = half / 64;
    let slice_len = FUSED64_SPLIT_ELEMS.min(e2);
    let oct_stride = fused64_oct_stride::<F>(fused64_tile::<F>());
    assert!(!buffers.is_empty() && e2.is_multiple_of(slice_len));
    assert_eq!(buffers.len(), quotients.len());
    {
        let mut tasks: Vec<SplitFused64Task<'_, F>> = Vec::new();
        let mut novel_lows: Vec<&[F]> = Vec::with_capacity(buffers.len());
        for (row_idx, buf) in buffers.iter_mut().enumerate() {
            let spare = &mut buf.spare_capacity_mut()[..alloc_len];
            let (first_half, rest_halves) = spare.split_at_mut(half);
            novel_lows.push(unsafe { slice::from_raw_parts(first_half.as_ptr().cast::<F>(), half) });
            push_split_fused64_tasks(&mut tasks, row_idx, rest_halves, half, slice_len);
        }
        let novel_lows = &novel_lows;
        parallel::for_each_piece_mut_with(
            &mut tasks,
            1,
            || vec![MaybeUninit::uninit(); 8 * oct_stride],
            |scratch, _, piece| {
                for (row_idx, half_idx, off, dst_subs) in piece {
                    run_split_fused64_task(
                        novel_lows[*row_idx],
                        &quotients[*row_idx],
                        scratch,
                        *half_idx,
                        *off,
                        dst_subs,
                        half,
                        slice_len,
                    );
                }
            },
        );
    }
    let mut sub_tasks: Vec<(usize, &mut [F])> = Vec::new();
    for buf in buffers.iter_mut() {
        let spare = &mut buf.spare_capacity_mut()[..alloc_len];
        let rest = &mut spare[half..];
        let full = unsafe { slice::from_raw_parts_mut(rest.as_mut_ptr().cast::<F>(), rest.len()) };
        for (idx, chunk) in full.chunks_exact_mut(e2).enumerate() {
            let start = half + idx * e2;
            if start < len {
                sub_tasks.push((start, chunk));
            }
        }
    }
    parallel::for_each_piece_mut(&mut sub_tasks, 1, |_, piece| {
        for (start, chunk) in piece {
            evaluate_full(chunk, *start, len, false);
        }
    });
}

fn evaluate_batch_rest_split<F: BinaryField>(
    buffers: &mut [Vec<F>],
    quotients: &[Vec<F>],
    half: usize,
    alloc_len: usize,
    len: usize,
) {
    if use_fused64::<F>(half) {
        return evaluate_batch_rest_split_fused64(buffers, quotients, half, alloc_len, len);
    }
    let e = half / 8;
    let slice_len = if e >= 2 * SLICE_ELEMS { SLICE_ELEMS } else { e };
    let slices_per = e / slice_len;
    assert!(half >= RADIX8_OUTER_MIN && !buffers.is_empty());
    assert_eq!(slices_per * slice_len, e);
    let domain = Domain::get();
    {
        let mut copy_tasks: Vec<CopySliceTask<'_, F>> = Vec::new();
        let mut novel_lows: Vec<&[F]> = Vec::with_capacity(buffers.len());
        for (row_idx, buf) in buffers.iter_mut().enumerate() {
            let spare = &mut buf.spare_capacity_mut()[..alloc_len];
            let (first_half, rest_halves) = spare.split_at_mut(half);
            novel_lows.push(unsafe { slice::from_raw_parts(first_half.as_ptr().cast::<F>(), half) });
            for (idx, dst) in rest_halves.chunks_mut(half).enumerate() {
                let octants = dst.len() / e;
                let mut oct_iters: Vec<_> =
                    dst.chunks_exact_mut(e).map(|o| o.chunks_exact_mut(slice_len)).collect();
                for s in 0..slices_per {
                    let mut out_slices = Vec::with_capacity(octants);
                    for iter in &mut oct_iters {
                        out_slices.push(iter.next().expect("slice"));
                    }
                    copy_tasks.push((row_idx, idx + 1, s * slice_len, out_slices));
                }
            }
        }
        let novel_lows = &novel_lows;
        parallel::for_each_piece_mut(&mut copy_tasks, 1, |_, piece| {
            for (row_idx, half_idx, off, dst_slices) in piece {
                let nl = novel_lows[*row_idx];
                let nh = &quotients[*row_idx];
                let low_oct: [&[F]; 8] = std::array::from_fn(|k| &nl[k * e + *off..k * e + *off + slice_len]);
                let high_oct: [&[F]; 8] = std::array::from_fn(|k| {
                    let start = k * e + *off;
                    if start < nh.len() { &nh[start..nh.len().min(start + slice_len)] } else { &[] }
                });
                let w_eff = domain.point(*half_idx);
                let (w4, w2, w1) = domain.radix8_twiddles(*half_idx * half, e);
                F::copy_half_radix8_octants_uninit(low_oct, high_oct, dst_slices, w_eff, w4, w2, w1);
            }
        });
    }
    let mut oct_tasks: Vec<(usize, &mut [F])> = Vec::new();
    for buf in buffers.iter_mut() {
        let spare = &mut buf.spare_capacity_mut()[..alloc_len];
        let rest = &mut spare[half..];
        let full = unsafe { slice::from_raw_parts_mut(rest.as_mut_ptr().cast::<F>(), rest.len()) };
        for (idx, chunk) in full.chunks_exact_mut(e).enumerate() {
            let start = half + idx * e;
            if start < len {
                oct_tasks.push((start, chunk));
            }
        }
    }
    parallel::for_each_piece_mut(&mut oct_tasks, 1, |_, piece| {
        for (start, chunk) in piece {
            evaluate_full(chunk, *start, len, false);
        }
    });
}

fn evaluate_batch_rest_fused<F: BinaryField>(
    buffers: &mut [Vec<F>],
    quotients: &[Vec<F>],
    half: usize,
    alloc_len: usize,
    len: usize,
) {
    assert_eq!(buffers.len(), quotients.len());
    let mut novel_lows: Vec<&[F]> = Vec::with_capacity(buffers.len());
    let mut half_tasks: Vec<(usize, usize, &mut [MaybeUninit<F>])> = Vec::new();
    for (row_idx, buf) in buffers.iter_mut().enumerate() {
        let spare = &mut buf.spare_capacity_mut()[..alloc_len];
        let (first_half, rest_halves) = spare.split_at_mut(half);
        novel_lows.push(unsafe { slice::from_raw_parts(first_half.as_ptr().cast::<F>(), half) });
        for (idx, dst) in rest_halves.chunks_mut(half).enumerate() {
            half_tasks.push((row_idx, idx + 1, dst));
        }
    }
    assert!(!half_tasks.is_empty());
    let novel_lows = &novel_lows;
    let fused = use_fused64::<F>(half);
    let scratch_len = if fused { 8 * fused64_oct_stride::<F>(fused64_tile::<F>()) } else { 0 };
    parallel::for_each_piece_mut_with(
        &mut half_tasks,
        1,
        || vec![MaybeUninit::uninit(); scratch_len],
        |scratch, _, piece| {
            for (row_idx, half_idx, dst) in piece {
                if fused {
                    evaluate_half_fused64(
                        novel_lows[*row_idx],
                        &quotients[*row_idx],
                        dst,
                        scratch,
                        *half_idx,
                        len,
                        false,
                    );
                } else {
                    evaluate_half_from_novel(
                        novel_lows[*row_idx],
                        &quotients[*row_idx],
                        dst,
                        *half_idx,
                        len,
                        false,
                    );
                }
            }
        },
    );
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
    let (alloc_len, _) = alloc_len_for(half, num_halves, len);
    let split_novel = half >= 2 * PARALLEL_THRESHOLD;
    let mut inner_b = 0usize;
    let (mut buffers, quotients): (Vec<Vec<F>>, Vec<Vec<F>>) = if split_novel {
        let prepared: Vec<(Vec<F>, Vec<F>, usize)> = parallel::map_range(rows.len(), 1, |row| {
            let row_count: usize = rows[row].iter().map(|part| part.len()).sum();
            assert_eq!(row_count, count);
            prepare_novel_factored(&rows[row], count, block, alloc_len)
        });
        inner_b = prepared[0].2;
        prepared.into_iter().map(|(v, q, _)| (v, q)).unzip()
    } else {
        let prepared: Vec<(Vec<F>, Vec<F>)> = parallel::map_range(rows.len(), 1, |row| {
            let row_count: usize = rows[row].iter().map(|part| part.len()).sum();
            assert_eq!(row_count, count);
            prepare_novel_buffer(&rows[row], count, block, alloc_len, false)
        });
        prepared.into_iter().unzip()
    };
    if split_novel {
        finish_batch_novel_blocks(&mut buffers, half, inner_b);
    }
    let split_rest = rows.len() * (num_halves - 1) < 4 * threads
        && half >= RADIX8_OUTER_MIN
        && half * F::BYTES >= SPLIT_HALF_BYTES;
    if split_rest {
        evaluate_batch_rest_split(&mut buffers, &quotients, half, alloc_len, len);
    } else {
        evaluate_batch_rest_fused(&mut buffers, &quotients, half, alloc_len, len);
    }
    evaluate_batch_half0(&mut buffers, half, len);
    for buf in &mut buffers {
        assert!(buf.capacity() >= len);
        unsafe { buf.set_len(len) };
    }
    assert_eq!(buffers.len(), rows.len());
    buffers
}

fn radix8_outer_at<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!(size >= 8 && size.is_power_of_two());
    let e = size / 8;
    let (w4, w2, w1) = Domain::get().radix8_twiddles(start, e);
    F::butterfly_radix8_outer(values, w4, w2[0], w2[1], w1);
}

fn evaluate_radix4_one<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!(size >= 4 && size.is_power_of_two());
    let q = size / 4;
    let domain = Domain::get();
    let w = domain.point(start / (2 * q));
    let w0 = domain.point(start / q);
    let w1 = w0 ^ domain.byte_tables[0][2];
    let (lo, hi) = values.split_at_mut(2 * q);
    let (v0, v1) = lo.split_at_mut(q);
    let (v2, v3) = hi.split_at_mut(q);
    F::butterfly_radix4(v0, v1, v2, v3, w, w0, w1);
}

fn evaluate_l1_pow8<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!(size >= 8 && size.is_power_of_two());
    assert!(size.trailing_zeros().is_multiple_of(3));
    let domain = Domain::get();
    if size == 8 {
        F::butterfly_radix8_steps(
            values,
            domain.point(start / 4),
            domain.point(start / 2),
            domain.point(start),
            &domain.step_mul,
        );
        return;
    }
    if size == 64 {
        radix8_outer_at(values, start);
        F::butterfly_radix8_steps(
            values,
            domain.point(start / 4),
            domain.point(start / 2),
            domain.point(start),
            &domain.step_mul,
        );
        return;
    }
    if size == 512 {
        radix8_outer_at(values, start);
        F::butterfly_radix8_stride8(
            values,
            domain.point(start / 32),
            domain.point(start / 16),
            domain.point(start / 8),
            &domain.step_mul,
        );
        F::butterfly_radix8_steps(
            values,
            domain.point(start / 4),
            domain.point(start / 2),
            domain.point(start),
            &domain.step_mul,
        );
        return;
    }
    radix8_outer_at(values, start);
    let e = size / 8;
    for (k, chunk) in values.chunks_exact_mut(e).enumerate() {
        evaluate_l1_pow8(chunk, start + k * e);
    }
}

fn evaluate_l1<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!(size >= 2 && size <= l1_block_len::<F>());
    assert!(size.is_power_of_two());
    let domain = Domain::get();
    if size == 2 {
        let (lo, hi) = values.split_at_mut(1);
        F::butterfly_slice(lo, hi, domain.point(start));
        return;
    }
    if size == 4 {
        evaluate_radix4_one(values, start);
        return;
    }
    if size == 128 || size == 256 {
        if size == 128 {
            let (lo, hi) = values.split_at_mut(64);
            F::butterfly_slice(lo, hi, domain.point(start / 64));
        } else {
            evaluate_radix4_one(values, start);
        }
        F::butterfly_radix8_stride8(
            values,
            domain.point(start / 32),
            domain.point(start / 16),
            domain.point(start / 8),
            &domain.step_mul,
        );
        F::butterfly_radix8_steps(
            values,
            domain.point(start / 4),
            domain.point(start / 2),
            domain.point(start),
            &domain.step_mul,
        );
        return;
    }
    match size.trailing_zeros() % 3 {
        0 => evaluate_l1_pow8(values, start),
        1 => {
            let half = size / 2;
            let (lo, hi) = values.split_at_mut(half);
            F::butterfly_slice(lo, hi, domain.point(start / half));
            evaluate_l1_pow8(lo, start);
            evaluate_l1_pow8(hi, start + half);
        }
        _ => {
            let q = size / 4;
            evaluate_radix4_one(values, start);
            for (k, chunk) in values.chunks_exact_mut(q).enumerate() {
                evaluate_l1_pow8(chunk, start + k * q);
            }
        }
    }
}

fn radix64_in_place<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    let e = size / 8;
    let e2 = e / 8;
    let tile = fused64_tile::<F>();
    let oct_stride = fused64_oct_stride::<F>(tile);
    assert!(size.is_power_of_two() && e2 >= tile && e2.is_multiple_of(tile));
    let domain = Domain::get();
    let (a_pairs, a_zero) = domain.radix8_twiddle_pairs(start, e);
    let b_tw: [([(u128, u128); 5], bool); 8] =
        std::array::from_fn(|k| domain.radix8_twiddle_pairs(start + k * e, e2));
    let mut scratch = vec![MaybeUninit::<F>::uninit(); 8 * oct_stride];
    let blocks = e2 / tile;
    assert!(blocks >= 1);
    for block in 0..blocks {
        let off = block * tile;
        for s in 0..8 {
            let pos = s * e2 + off;
            F::copy_radix8_strided_uninit(
                &values[pos..],
                e,
                &mut scratch[s * tile..],
                oct_stride,
                tile,
                &a_pairs,
                a_zero,
            );
        }
        for (k, (b_pairs, b_zero)) in b_tw.iter().enumerate() {
            let src_k =
                unsafe { slice::from_raw_parts(scratch.as_ptr().add(k * oct_stride).cast::<F>(), 8 * tile) };
            let dst_k = as_uninit_slice_mut(&mut values[k * e + off..]);
            F::copy_radix8_strided_uninit(src_k, tile, dst_k, e2, tile, b_pairs, *b_zero);
        }
    }
}

fn evaluate_full<F: BinaryField>(values: &mut [F], start: usize, len: usize, allow_parallel: bool) {
    let size = values.len();
    assert!(size.is_power_of_two() && start < len);
    if size == 1 {
        return;
    }
    if size <= l1_block_len::<F>() && start + size <= len {
        return evaluate_l1(values, start);
    }
    if !allow_parallel && start + size <= len && use_fused64::<F>(size) {
        radix64_in_place(values, start);
        evaluate_sub_octants(values, start, size / 64, 64, len, false);
        return;
    }
    if start + size <= len && size >= RADIX8_OUTER_MIN && (!allow_parallel || size < 8 * PARALLEL_THRESHOLD) {
        radix8_outer_at(values, start);
        evaluate_octants(values, start, size / 8, 8, len, allow_parallel);
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

fn transpose_l1_pow8<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!(size >= 8 && size.is_power_of_two());
    assert!(size.trailing_zeros().is_multiple_of(3));
    let domain = Domain::get();
    if size == 8 {
        F::transpose_radix8_steps(
            values,
            domain.point(start / 4),
            domain.point(start / 2),
            domain.point(start),
            &domain.step_mul,
        );
        return;
    }
    if size == 64 {
        F::transpose_radix8_steps(
            values,
            domain.point(start / 4),
            domain.point(start / 2),
            domain.point(start),
            &domain.step_mul,
        );
        let (w4, w2, w1) = domain.radix8_twiddles(start, 8);
        F::transpose_radix8_outer(values, w4, w2, w1);
        return;
    }
    if size == 512 {
        F::transpose_radix8_steps(
            values,
            domain.point(start / 4),
            domain.point(start / 2),
            domain.point(start),
            &domain.step_mul,
        );
        F::transpose_radix8_stride8(
            values,
            domain.point(start / 32),
            domain.point(start / 16),
            domain.point(start / 8),
            &domain.step_mul,
        );
        let (w4, w2, w1) = domain.radix8_twiddles(start, 64);
        F::transpose_radix8_outer(values, w4, w2, w1);
        return;
    }
    let e = size / 8;
    for (k, chunk) in values.chunks_exact_mut(e).enumerate() {
        transpose_l1_pow8(chunk, start + k * e);
    }
    let (w4, w2, w1) = domain.radix8_twiddles(start, e);
    F::transpose_radix8_outer(values, w4, w2, w1);
}

fn transpose_l1<F: BinaryField>(values: &mut [F], start: usize) {
    let size = values.len();
    assert!(size >= 2 && size <= l1_block_len::<F>());
    assert!(size.is_power_of_two());
    if size == 2 {
        let (lo, hi) = values.split_at_mut(1);
        F::transpose_butterfly_slice(lo, hi, Domain::get().point(start));
        return;
    }
    if size == 4 {
        let domain = Domain::get();
        let w = domain.point(start / 2);
        let w0 = domain.point(start);
        let w1 = w0 ^ domain.byte_tables[0][2];
        let (lo, hi) = values.split_at_mut(2);
        let (v0, v1) = lo.split_at_mut(1);
        let (v2, v3) = hi.split_at_mut(1);
        F::transpose_butterfly_slice(v0, v1, w0);
        F::transpose_butterfly_slice(v2, v3, w1);
        F::transpose_butterfly_slice(lo, hi, w);
        return;
    }
    match size.trailing_zeros() % 3 {
        0 => transpose_l1_pow8(values, start),
        1 => {
            let half = size / 2;
            let (lo, hi) = values.split_at_mut(half);
            transpose_l1_pow8(lo, start);
            transpose_l1_pow8(hi, start + half);
            F::transpose_butterfly_slice(lo, hi, Domain::get().point(start / half));
        }
        _ => {
            let q = size / 4;
            for (k, chunk) in values.chunks_exact_mut(q).enumerate() {
                transpose_l1_pow8(chunk, start + k * q);
            }
            let domain = Domain::get();
            let w = domain.point(start / (2 * q));
            let w0 = domain.point(start / q);
            let w1 = w0 ^ domain.byte_tables[0][2];
            let (lo, hi) = values.split_at_mut(2 * q);
            let (v0, v1) = lo.split_at_mut(q);
            let (v2, v3) = hi.split_at_mut(q);
            F::transpose_butterfly_slice(v0, v1, w0);
            F::transpose_butterfly_slice(v2, v3, w1);
            F::transpose_butterfly_slice(lo, hi, w);
        }
    }
}

fn transpose_combine_halves<F: BinaryField>(
    lo: &mut [F],
    hi: &mut [F],
    has_lo: bool,
    has_hi: bool,
    twiddle: u128,
    allow_parallel: bool,
) -> bool {
    let half = lo.len();
    assert_eq!(half, hi.len());
    assert!(half >= 1 && half.is_power_of_two());
    let par = allow_parallel && half >= 2 * PARALLEL_THRESHOLD;
    match (has_lo, has_hi) {
        (false, false) => false,
        (true, false) => {
            if par {
                parallel::for_each_pair_mut(lo, hi, MIN_PIECE, |l, h| F::transpose_from_low(l, h, twiddle));
            } else {
                F::transpose_from_low(lo, hi, twiddle);
            }
            true
        }
        (false, true) => {
            if par {
                parallel::for_each_pair_mut(lo, hi, MIN_PIECE, |l, h| F::transpose_from_high(l, h, twiddle));
            } else {
                F::transpose_from_high(lo, hi, twiddle);
            }
            true
        }
        (true, true) => {
            if par {
                parallel::for_each_pair_mut(lo, hi, MIN_PIECE, |l, h| {
                    F::transpose_butterfly_slice(l, h, twiddle)
                });
            } else {
                F::transpose_butterfly_slice(lo, hi, twiddle);
            }
            true
        }
    }
}

fn transpose_fft_sparse<F: BinaryField>(
    out: &mut [F],
    start: usize,
    queries: &[(usize, F)],
    allow_parallel: bool,
) -> bool {
    let size = out.len();
    assert!(size >= 1 && size.is_power_of_two());
    if queries.is_empty() {
        return false;
    }
    assert!(queries[0].0 >= start && queries[queries.len() - 1].0 < start + size);
    if size == 1 {
        let mut sum = F::ZERO;
        for &(_, scale) in queries {
            sum += scale;
        }
        out[0] = sum;
        return true;
    }
    if size <= 64 || (size <= l1_block_len::<F>() && queries.len() * 8 >= size) {
        out.fill(F::ZERO);
        for &(pos, scale) in queries {
            out[pos - start] += scale;
        }
        transpose_l1(out, start);
        return true;
    }
    let half = size / 2;
    let mid = start + half;
    let split = queries.partition_point(|&(p, _)| p < mid);
    let (lo, hi) = out.split_at_mut(half);
    let (has_lo, has_hi) =
        if allow_parallel && half >= PARALLEL_THRESHOLD && split > 0 && split < queries.len() {
            parallel::join(
                || transpose_fft_sparse(lo, start, &queries[..split], true),
                || transpose_fft_sparse(hi, mid, &queries[split..], true),
            )
        } else {
            (
                transpose_fft_sparse(lo, start, &queries[..split], allow_parallel),
                transpose_fft_sparse(hi, mid, &queries[split..], allow_parallel),
            )
        };
    let twiddle = Domain::get().point(start / half);
    transpose_combine_halves(lo, hi, has_lo, has_hi, twiddle, allow_parallel)
}

fn group_coset_ranges<F>(sorted: &[(usize, F)], n: usize) -> Vec<(usize, usize, usize)> {
    assert!(!sorted.is_empty() && n > 0);
    let mut coset_ranges: Vec<(usize, usize, usize)> = Vec::new();
    let mut i = 0;
    for _ in 0..sorted.len() {
        if i >= sorted.len() {
            break;
        }
        let coset = sorted[i].0 / n;
        let mut j = i + 1;
        for _ in i + 1..sorted.len() {
            if j >= sorted.len() || sorted[j].0 / n != coset {
                break;
            }
            j += 1;
        }
        coset_ranges.push((coset, i, j));
        i = j;
    }
    assert_eq!(i, sorted.len());
    coset_ranges
}

fn add_transposed_queries<F: BinaryField>(dst: &mut [F], queries: &[(usize, F)]) {
    let n = dst.len();
    if n == 0 || queries.is_empty() {
        return;
    }
    if n < TRANSPOSE_MIN_LEN || !n.is_power_of_two() || queries.len() < TRANSPOSE_MIN_QUERIES {
        for &(pos, scale) in queries {
            F::add_powers_slice(dst, F::point(pos), scale);
        }
        return;
    }
    let mut sorted: Vec<(usize, F)> = queries.to_vec();
    sorted.sort_unstable_by_key(|&(pos, _)| pos);
    let coset_ranges = group_coset_ranges(&sorted, n);
    let allow_inner = coset_ranges.len() < parallel::threads();
    let mut partials: Vec<Vec<F>> = parallel::fold_pieces(
        coset_ranges.len(),
        1,
        || None::<(Vec<F>, Option<Vec<F>>)>,
        |state, range| {
            for idx in range {
                let (coset, s0, s1) = coset_ranges[idx];
                if let Some((acc, scratch_slot)) = state.as_mut() {
                    let buf = scratch_slot.get_or_insert_with(|| vec![F::ZERO; n]);
                    let _ = transpose_fft_sparse(buf, coset * n, &sorted[s0..s1], allow_inner);
                    F::xor_slice(acc, buf);
                } else {
                    let mut buf = vec![F::ZERO; n];
                    let _ = transpose_fft_sparse(&mut buf, coset * n, &sorted[s0..s1], allow_inner);
                    *state = Some((buf, None));
                }
            }
        },
    )
    .into_iter()
    .flatten()
    .map(|(acc, _)| acc)
    .collect();
    assert!(!partials.is_empty());
    let mut combined = partials.swap_remove(0);
    for other in &partials {
        if n < PARALLEL_THRESHOLD {
            F::xor_slice(&mut combined, other);
        } else {
            parallel::for_each_piece_mut(&mut combined, MIN_PIECE, |start, piece| {
                F::xor_slice(piece, &other[start..start + piece.len()]);
            });
        }
    }
    to_novel_basis_pow2_transpose(&mut combined, true);
    if n < PARALLEL_THRESHOLD {
        F::xor_slice(dst, &combined);
    } else {
        parallel::for_each_piece_mut(dst, MIN_PIECE, |start, piece| {
            F::xor_slice(piece, &combined[start..start + piece.len()]);
        });
    }
    assert_eq!(combined.len(), n);
}
