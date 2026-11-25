pub const KEY_LEN: usize = 32;
pub const OUT_LEN: usize = 32;
pub const BLOCK_LEN: usize = 64;
pub const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const CV_STACK_LEN_MAX: usize = 54;

const IV: [u32; 8] =
    [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

const MSG_SCHEDULE: [[usize; 16]; 7] = {
    let mut table = [[0usize; 16]; 7];
    let mut word = 0;
    while word < 16 {
        table[0][word] = word;
        word += 1;
    }
    let mut round = 1;
    while round < 7 {
        let mut word = 0;
        while word < 16 {
            table[round][word] = table[round - 1][MSG_PERMUTATION[word]];
            word += 1;
        }
        round += 1;
    }
    table
};

#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline(always)]
fn round(state: &mut [u32; 16], m: &[u32; 16], schedule: &[usize; 16]) {
    g(state, 0, 4, 8, 12, m[schedule[0]], m[schedule[1]]);
    g(state, 1, 5, 9, 13, m[schedule[2]], m[schedule[3]]);
    g(state, 2, 6, 10, 14, m[schedule[4]], m[schedule[5]]);
    g(state, 3, 7, 11, 15, m[schedule[6]], m[schedule[7]]);
    g(state, 0, 5, 10, 15, m[schedule[8]], m[schedule[9]]);
    g(state, 1, 6, 11, 12, m[schedule[10]], m[schedule[11]]);
    g(state, 2, 7, 8, 13, m[schedule[12]], m[schedule[13]]);
    g(state, 3, 4, 9, 14, m[schedule[14]], m[schedule[15]]);
}

fn compress(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    assert!(block_len <= BLOCK_LEN as u32);
    const { assert!(MSG_SCHEDULE.len() == 7) };
    let mut state = [
        chaining_value[0],
        chaining_value[1],
        chaining_value[2],
        chaining_value[3],
        chaining_value[4],
        chaining_value[5],
        chaining_value[6],
        chaining_value[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    for schedule in &MSG_SCHEDULE {
        round(&mut state, block_words, schedule);
    }
    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    state
}

#[inline(always)]
fn first_8_words(compression_output: [u32; 16]) -> [u32; 8] {
    compression_output[0..8].try_into().expect("eight words")
}

#[inline(always)]
fn words_from_block(block: &[u8; BLOCK_LEN]) -> [u32; 16] {
    let mut words = [0u32; 16];
    for (word, chunk) in words.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*chunk);
    }
    words
}

#[inline(always)]
fn words_from_partial_block(block: &[u8]) -> [u32; 16] {
    assert!(block.len() <= BLOCK_LEN);
    let mut padded = [0u8; BLOCK_LEN];
    padded[..block.len()].copy_from_slice(block);
    words_from_block(&padded)
}

pub(crate) const fn words_from_key(key: &[u8; KEY_LEN]) -> [u32; 8] {
    let mut words = [0u32; 8];
    let mut index = 0;
    while index < 8 {
        let base = index * 4;
        words[index] = u32::from_le_bytes([key[base], key[base + 1], key[base + 2], key[base + 3]]);
        index += 1;
    }
    words
}

#[inline(always)]
fn bytes_from_words(words: &[u32; 8]) -> [u8; OUT_LEN] {
    let mut out = [0u8; OUT_LEN];
    for (dst, &word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *dst = word.to_le_bytes();
    }
    out
}

#[inline(always)]
fn compress_chunk_scalar(
    key: &[u32; 8],
    chunk: &[u8],
    chunk_counter: u64,
    flags: u32,
    extra_end_flags: u32,
) -> [u32; 8] {
    assert!(chunk.len() <= CHUNK_LEN);
    let mut cv = *key;
    let mut offset = 0;
    let full_len = chunk.len().saturating_sub(1) / BLOCK_LEN * BLOCK_LEN;
    while offset < full_len {
        let block: &[u8; BLOCK_LEN] = chunk[offset..offset + BLOCK_LEN].try_into().expect("full block");
        let block_flags = flags | if offset == 0 { CHUNK_START } else { 0 };
        cv = first_8_words(compress(
            &cv,
            &words_from_block(block),
            chunk_counter,
            BLOCK_LEN as u32,
            block_flags,
        ));
        offset += BLOCK_LEN;
    }
    let tail = &chunk[offset..];
    assert!(tail.len() <= BLOCK_LEN);
    let end_flags = flags | if offset == 0 { CHUNK_START } else { 0 } | CHUNK_END | extra_end_flags;
    first_8_words(compress(&cv, &words_from_partial_block(tail), chunk_counter, tail.len() as u32, end_flags))
}

#[inline(always)]
fn parent_cv(left_child_cv: &[u32; 8], right_child_cv: &[u32; 8], key: &[u32; 8], flags: u32) -> [u32; 8] {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(left_child_cv);
    block_words[8..].copy_from_slice(right_child_cv);
    first_8_words(compress(key, &block_words, 0, BLOCK_LEN as u32, flags | PARENT))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
mod neon {
    use super::{
        BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, CV_STACK_LEN_MAX, IV, KEYED_HASH, MSG_SCHEDULE,
        OUT_LEN, PARENT, ROOT,
    };
    use std::arch::aarch64::*;

    type V = uint32x4_t;

    const ROT8_MASK: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
    const CHUNK_BLOCKS: usize = CHUNK_LEN / BLOCK_LEN;

    #[inline(always)]
    unsafe fn rotr16(x: V) -> V {
        unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
    }

    #[inline(always)]
    unsafe fn rotr12(x: V) -> V {
        unsafe { vsliq_n_u32::<20>(vshrq_n_u32::<12>(x), x) }
    }

    #[inline(always)]
    unsafe fn rotr8(x: V, mask: uint8x16_t) -> V {
        unsafe { vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), mask)) }
    }

    #[inline(always)]
    unsafe fn rotr7(x: V) -> V {
        unsafe { vsliq_n_u32::<25>(vshrq_n_u32::<7>(x), x) }
    }

    #[inline(always)]
    unsafe fn transpose_4x4(a: V, b: V, c: V, d: V) -> [V; 4] {
        unsafe {
            let ab_lo = vreinterpretq_u64_u32(vzip1q_u32(a, b));
            let ab_hi = vreinterpretq_u64_u32(vzip2q_u32(a, b));
            let cd_lo = vreinterpretq_u64_u32(vzip1q_u32(c, d));
            let cd_hi = vreinterpretq_u64_u32(vzip2q_u32(c, d));
            [
                vreinterpretq_u32_u64(vzip1q_u64(ab_lo, cd_lo)),
                vreinterpretq_u32_u64(vzip2q_u64(ab_lo, cd_lo)),
                vreinterpretq_u32_u64(vzip1q_u64(ab_hi, cd_hi)),
                vreinterpretq_u32_u64(vzip2q_u64(ab_hi, cd_hi)),
            ]
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn g4(s: &mut [V; 16], a: usize, b: usize, c: usize, d: usize, mx: V, my: V, rot8: uint8x16_t) {
        unsafe {
            s[a] = vaddq_u32(vaddq_u32(s[a], mx), s[b]);
            s[d] = rotr16(veorq_u32(s[d], s[a]));
            s[c] = vaddq_u32(s[c], s[d]);
            s[b] = rotr12(veorq_u32(s[b], s[c]));
            s[a] = vaddq_u32(vaddq_u32(s[a], my), s[b]);
            s[d] = rotr8(veorq_u32(s[d], s[a]), rot8);
            s[c] = vaddq_u32(s[c], s[d]);
            s[b] = rotr7(veorq_u32(s[b], s[c]));
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn g8(
        s0: &mut [V; 16],
        s1: &mut [V; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x0: V,
        y0: V,
        x1: V,
        y1: V,
        rot8: uint8x16_t,
    ) {
        unsafe {
            s0[a] = vaddq_u32(vaddq_u32(s0[a], x0), s0[b]);
            s1[a] = vaddq_u32(vaddq_u32(s1[a], x1), s1[b]);
            s0[d] = rotr16(veorq_u32(s0[d], s0[a]));
            s1[d] = rotr16(veorq_u32(s1[d], s1[a]));
            s0[c] = vaddq_u32(s0[c], s0[d]);
            s1[c] = vaddq_u32(s1[c], s1[d]);
            s0[b] = rotr12(veorq_u32(s0[b], s0[c]));
            s1[b] = rotr12(veorq_u32(s1[b], s1[c]));
            s0[a] = vaddq_u32(vaddq_u32(s0[a], y0), s0[b]);
            s1[a] = vaddq_u32(vaddq_u32(s1[a], y1), s1[b]);
            s0[d] = rotr8(veorq_u32(s0[d], s0[a]), rot8);
            s1[d] = rotr8(veorq_u32(s1[d], s1[a]), rot8);
            s0[c] = vaddq_u32(s0[c], s0[d]);
            s1[c] = vaddq_u32(s1[c], s1[d]);
            s0[b] = rotr7(veorq_u32(s0[b], s0[c]));
            s1[b] = rotr7(veorq_u32(s1[b], s1[c]));
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn g16(
        s: &mut [[V; 16]; 4],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        m: &[[V; 16]; 4],
        wx: usize,
        wy: usize,
        rot8: uint8x16_t,
    ) {
        unsafe {
            s[0][a] = vaddq_u32(vaddq_u32(s[0][a], m[0][wx]), s[0][b]);
            s[1][a] = vaddq_u32(vaddq_u32(s[1][a], m[1][wx]), s[1][b]);
            s[2][a] = vaddq_u32(vaddq_u32(s[2][a], m[2][wx]), s[2][b]);
            s[3][a] = vaddq_u32(vaddq_u32(s[3][a], m[3][wx]), s[3][b]);
            s[0][d] = rotr16(veorq_u32(s[0][d], s[0][a]));
            s[1][d] = rotr16(veorq_u32(s[1][d], s[1][a]));
            s[2][d] = rotr16(veorq_u32(s[2][d], s[2][a]));
            s[3][d] = rotr16(veorq_u32(s[3][d], s[3][a]));
            s[0][c] = vaddq_u32(s[0][c], s[0][d]);
            s[1][c] = vaddq_u32(s[1][c], s[1][d]);
            s[2][c] = vaddq_u32(s[2][c], s[2][d]);
            s[3][c] = vaddq_u32(s[3][c], s[3][d]);
            s[0][b] = rotr12(veorq_u32(s[0][b], s[0][c]));
            s[1][b] = rotr12(veorq_u32(s[1][b], s[1][c]));
            s[2][b] = rotr12(veorq_u32(s[2][b], s[2][c]));
            s[3][b] = rotr12(veorq_u32(s[3][b], s[3][c]));
            s[0][a] = vaddq_u32(vaddq_u32(s[0][a], m[0][wy]), s[0][b]);
            s[1][a] = vaddq_u32(vaddq_u32(s[1][a], m[1][wy]), s[1][b]);
            s[2][a] = vaddq_u32(vaddq_u32(s[2][a], m[2][wy]), s[2][b]);
            s[3][a] = vaddq_u32(vaddq_u32(s[3][a], m[3][wy]), s[3][b]);
            s[0][d] = rotr8(veorq_u32(s[0][d], s[0][a]), rot8);
            s[1][d] = rotr8(veorq_u32(s[1][d], s[1][a]), rot8);
            s[2][d] = rotr8(veorq_u32(s[2][d], s[2][a]), rot8);
            s[3][d] = rotr8(veorq_u32(s[3][d], s[3][a]), rot8);
            s[0][c] = vaddq_u32(s[0][c], s[0][d]);
            s[1][c] = vaddq_u32(s[1][c], s[1][d]);
            s[2][c] = vaddq_u32(s[2][c], s[2][d]);
            s[3][c] = vaddq_u32(s[3][c], s[3][d]);
            s[0][b] = rotr7(veorq_u32(s[0][b], s[0][c]));
            s[1][b] = rotr7(veorq_u32(s[1][b], s[1][c]));
            s[2][b] = rotr7(veorq_u32(s[2][b], s[2][c]));
            s[3][b] = rotr7(veorq_u32(s[3][b], s[3][c]));
        }
    }

    #[inline(always)]
    unsafe fn init_state(cv: &[V; 8], t_lo: V, t_hi: V, len: u32, flags: u32) -> [V; 16] {
        unsafe {
            [
                cv[0],
                cv[1],
                cv[2],
                cv[3],
                cv[4],
                cv[5],
                cv[6],
                cv[7],
                vdupq_n_u32(IV[0]),
                vdupq_n_u32(IV[1]),
                vdupq_n_u32(IV[2]),
                vdupq_n_u32(IV[3]),
                t_lo,
                t_hi,
                vdupq_n_u32(len),
                vdupq_n_u32(flags),
            ]
        }
    }

    #[inline(always)]
    unsafe fn broadcast_key(key: &[u32; 8]) -> [V; 8] {
        unsafe {
            [
                vdupq_n_u32(key[0]),
                vdupq_n_u32(key[1]),
                vdupq_n_u32(key[2]),
                vdupq_n_u32(key[3]),
                vdupq_n_u32(key[4]),
                vdupq_n_u32(key[5]),
                vdupq_n_u32(key[6]),
                vdupq_n_u32(key[7]),
            ]
        }
    }

    #[inline(always)]
    unsafe fn compress_cv_4x(
        cv: &mut [V; 8],
        m: &[V; 16],
        counter: u64,
        len: u32,
        flags: u32,
        rot8: uint8x16_t,
    ) {
        assert!(len <= BLOCK_LEN as u32);
        const { assert!(MSG_SCHEDULE.len() == 7) };
        unsafe {
            let mut s =
                init_state(cv, vdupq_n_u32(counter as u32), vdupq_n_u32((counter >> 32) as u32), len, flags);
            for sc in &MSG_SCHEDULE {
                g4(&mut s, 0, 4, 8, 12, m[sc[0]], m[sc[1]], rot8);
                g4(&mut s, 1, 5, 9, 13, m[sc[2]], m[sc[3]], rot8);
                g4(&mut s, 2, 6, 10, 14, m[sc[4]], m[sc[5]], rot8);
                g4(&mut s, 3, 7, 11, 15, m[sc[6]], m[sc[7]], rot8);
                g4(&mut s, 0, 5, 10, 15, m[sc[8]], m[sc[9]], rot8);
                g4(&mut s, 1, 6, 11, 12, m[sc[10]], m[sc[11]], rot8);
                g4(&mut s, 2, 7, 8, 13, m[sc[12]], m[sc[13]], rot8);
                g4(&mut s, 3, 4, 9, 14, m[sc[14]], m[sc[15]], rot8);
            }
            for i in 0..8 {
                cv[i] = veorq_u32(s[i], s[i + 8]);
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn compress_cv_8x(
        cv0: &mut [V; 8],
        cv1: &mut [V; 8],
        m0: &[V; 16],
        m1: &[V; 16],
        counter: u64,
        len: u32,
        flags: u32,
        rot8: uint8x16_t,
    ) {
        assert!(len <= BLOCK_LEN as u32);
        const { assert!(MSG_SCHEDULE.len() == 7) };
        unsafe {
            let t_lo = vdupq_n_u32(counter as u32);
            let t_hi = vdupq_n_u32((counter >> 32) as u32);
            let mut s0 = init_state(cv0, t_lo, t_hi, len, flags);
            let mut s1 = init_state(cv1, t_lo, t_hi, len, flags);
            for sc in &MSG_SCHEDULE {
                g8(&mut s0, &mut s1, 0, 4, 8, 12, m0[sc[0]], m0[sc[1]], m1[sc[0]], m1[sc[1]], rot8);
                g8(&mut s0, &mut s1, 1, 5, 9, 13, m0[sc[2]], m0[sc[3]], m1[sc[2]], m1[sc[3]], rot8);
                g8(&mut s0, &mut s1, 2, 6, 10, 14, m0[sc[4]], m0[sc[5]], m1[sc[4]], m1[sc[5]], rot8);
                g8(&mut s0, &mut s1, 3, 7, 11, 15, m0[sc[6]], m0[sc[7]], m1[sc[6]], m1[sc[7]], rot8);
                g8(&mut s0, &mut s1, 0, 5, 10, 15, m0[sc[8]], m0[sc[9]], m1[sc[8]], m1[sc[9]], rot8);
                g8(&mut s0, &mut s1, 1, 6, 11, 12, m0[sc[10]], m0[sc[11]], m1[sc[10]], m1[sc[11]], rot8);
                g8(&mut s0, &mut s1, 2, 7, 8, 13, m0[sc[12]], m0[sc[13]], m1[sc[12]], m1[sc[13]], rot8);
                g8(&mut s0, &mut s1, 3, 4, 9, 14, m0[sc[14]], m0[sc[15]], m1[sc[14]], m1[sc[15]], rot8);
            }
            for i in 0..8 {
                cv0[i] = veorq_u32(s0[i], s0[i + 8]);
                cv1[i] = veorq_u32(s1[i], s1[i + 8]);
            }
        }
    }

    #[inline(always)]
    unsafe fn compress_cv_16x(
        cv: &mut [[V; 8]; 4],
        m: &[[V; 16]; 4],
        counter: u64,
        len: u32,
        flags: u32,
        rot8: uint8x16_t,
    ) {
        assert!(len <= BLOCK_LEN as u32);
        const { assert!(MSG_SCHEDULE.len() == 7) };
        unsafe {
            let t_lo = vdupq_n_u32(counter as u32);
            let t_hi = vdupq_n_u32((counter >> 32) as u32);
            let mut s = [
                init_state(&cv[0], t_lo, t_hi, len, flags),
                init_state(&cv[1], t_lo, t_hi, len, flags),
                init_state(&cv[2], t_lo, t_hi, len, flags),
                init_state(&cv[3], t_lo, t_hi, len, flags),
            ];
            for sc in &MSG_SCHEDULE {
                g16(&mut s, 0, 4, 8, 12, m, sc[0], sc[1], rot8);
                g16(&mut s, 1, 5, 9, 13, m, sc[2], sc[3], rot8);
                g16(&mut s, 2, 6, 10, 14, m, sc[4], sc[5], rot8);
                g16(&mut s, 3, 7, 11, 15, m, sc[6], sc[7], rot8);
                g16(&mut s, 0, 5, 10, 15, m, sc[8], sc[9], rot8);
                g16(&mut s, 1, 6, 11, 12, m, sc[10], sc[11], rot8);
                g16(&mut s, 2, 7, 8, 13, m, sc[12], sc[13], rot8);
                g16(&mut s, 3, 4, 9, 14, m, sc[14], sc[15], rot8);
            }
            for i in 0..8 {
                cv[0][i] = veorq_u32(s[0][i], s[0][i + 8]);
                cv[1][i] = veorq_u32(s[1][i], s[1][i + 8]);
                cv[2][i] = veorq_u32(s[2][i], s[2][i + 8]);
                cv[3][i] = veorq_u32(s[3][i], s[3][i + 8]);
            }
        }
    }

    #[inline(always)]
    unsafe fn load_u32x4(ptr: *const u8) -> V {
        unsafe { vreinterpretq_u32_u8(vld1q_u8(ptr)) }
    }

    #[inline(always)]
    unsafe fn store_u32x4(ptr: *mut u8, v: V) {
        unsafe { vst1q_u8(ptr, vreinterpretq_u8_u32(v)) }
    }

    #[inline(always)]
    unsafe fn load_4_blocks(p0: *const u8, p1: *const u8, p2: *const u8, p3: *const u8) -> [V; 16] {
        unsafe {
            let mut m = [vdupq_n_u32(0); 16];
            for group in 0..4 {
                let off = group * 16;
                let [w0, w1, w2, w3] = transpose_4x4(
                    load_u32x4(p0.add(off)),
                    load_u32x4(p1.add(off)),
                    load_u32x4(p2.add(off)),
                    load_u32x4(p3.add(off)),
                );
                m[group * 4] = w0;
                m[group * 4 + 1] = w1;
                m[group * 4 + 2] = w2;
                m[group * 4 + 3] = w3;
            }
            m
        }
    }

    #[inline(always)]
    unsafe fn load_4_partial_blocks(b0: &[u8], b1: &[u8], b2: &[u8], b3: &[u8]) -> [V; 16] {
        let len = b0.len();
        assert!(len <= BLOCK_LEN && b1.len() == len && b2.len() == len && b3.len() == len);
        let mut p0 = [0u8; BLOCK_LEN];
        let mut p1 = [0u8; BLOCK_LEN];
        let mut p2 = [0u8; BLOCK_LEN];
        let mut p3 = [0u8; BLOCK_LEN];
        p0[..len].copy_from_slice(b0);
        p1[..len].copy_from_slice(b1);
        p2[..len].copy_from_slice(b2);
        p3[..len].copy_from_slice(b3);
        unsafe { load_4_blocks(p0.as_ptr(), p1.as_ptr(), p2.as_ptr(), p3.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_4_digests(cv: &[V; 8], out: &mut [[u8; OUT_LEN]; 4]) {
        unsafe {
            let lo = transpose_4x4(cv[0], cv[1], cv[2], cv[3]);
            let hi = transpose_4x4(cv[4], cv[5], cv[6], cv[7]);
            for lane in 0..4 {
                store_u32x4(out[lane].as_mut_ptr(), lo[lane]);
                store_u32x4(out[lane].as_mut_ptr().add(16), hi[lane]);
            }
        }
    }

    #[inline(always)]
    unsafe fn store_16_words_4x(s: &[V; 16], base_cv: &[V; 8], out: &mut [u8]) {
        assert!(out.len() == 4 * BLOCK_LEN);
        unsafe {
            let mut w = [vdupq_n_u32(0); 16];
            for i in 0..8 {
                w[i] = veorq_u32(s[i], s[i + 8]);
                w[i + 8] = veorq_u32(s[i + 8], base_cv[i]);
            }
            for group in 0..4 {
                let q = transpose_4x4(w[group * 4], w[group * 4 + 1], w[group * 4 + 2], w[group * 4 + 3]);
                for (lane, &vec) in q.iter().enumerate() {
                    store_u32x4(out.as_mut_ptr().add(lane * BLOCK_LEN + group * 16), vec);
                }
            }
        }
    }

    pub fn root_blocks_16x(
        cv_words: &[u32; 8],
        block_words: &[u32; 16],
        start_counter: u64,
        block_len: u32,
        flags: u32,
        out: &mut [u8],
    ) {
        assert!(out.len() == 16 * BLOCK_LEN);
        assert!(block_len <= BLOCK_LEN as u32);
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let base_cv = broadcast_key(cv_words);
            let mut m0 = [vdupq_n_u32(0); 16];
            for (dst, &word) in m0.iter_mut().zip(block_words) {
                *dst = vdupq_n_u32(word);
            }
            let m = [m0, m0, m0, m0];
            let mut s = [[vdupq_n_u32(0); 16]; 4];
            for (grp, st) in s.iter_mut().enumerate() {
                let base = start_counter.wrapping_add((grp * 4) as u64);
                let lo = [
                    base as u32,
                    base.wrapping_add(1) as u32,
                    base.wrapping_add(2) as u32,
                    base.wrapping_add(3) as u32,
                ];
                let hi = [
                    (base >> 32) as u32,
                    (base.wrapping_add(1) >> 32) as u32,
                    (base.wrapping_add(2) >> 32) as u32,
                    (base.wrapping_add(3) >> 32) as u32,
                ];
                *st = init_state(&base_cv, vld1q_u32(lo.as_ptr()), vld1q_u32(hi.as_ptr()), block_len, flags);
            }
            for sc in &MSG_SCHEDULE {
                g16(&mut s, 0, 4, 8, 12, &m, sc[0], sc[1], rot8);
                g16(&mut s, 1, 5, 9, 13, &m, sc[2], sc[3], rot8);
                g16(&mut s, 2, 6, 10, 14, &m, sc[4], sc[5], rot8);
                g16(&mut s, 3, 7, 11, 15, &m, sc[6], sc[7], rot8);
                g16(&mut s, 0, 5, 10, 15, &m, sc[8], sc[9], rot8);
                g16(&mut s, 1, 6, 11, 12, &m, sc[10], sc[11], rot8);
                g16(&mut s, 2, 7, 8, 13, &m, sc[12], sc[13], rot8);
                g16(&mut s, 3, 4, 9, 14, &m, sc[14], sc[15], rot8);
            }
            for (grp, st) in s.iter().enumerate() {
                store_16_words_4x(st, &base_cv, &mut out[grp * 4 * BLOCK_LEN..(grp + 1) * 4 * BLOCK_LEN]);
            }
        }
    }

    #[inline(always)]
    unsafe fn load_16_chunk_blocks(ptr: *const u8, off: usize) -> [[V; 16]; 4] {
        unsafe {
            [
                load_4_blocks(
                    ptr.add(off),
                    ptr.add(CHUNK_LEN + off),
                    ptr.add(2 * CHUNK_LEN + off),
                    ptr.add(3 * CHUNK_LEN + off),
                ),
                load_4_blocks(
                    ptr.add(4 * CHUNK_LEN + off),
                    ptr.add(5 * CHUNK_LEN + off),
                    ptr.add(6 * CHUNK_LEN + off),
                    ptr.add(7 * CHUNK_LEN + off),
                ),
                load_4_blocks(
                    ptr.add(8 * CHUNK_LEN + off),
                    ptr.add(9 * CHUNK_LEN + off),
                    ptr.add(10 * CHUNK_LEN + off),
                    ptr.add(11 * CHUNK_LEN + off),
                ),
                load_4_blocks(
                    ptr.add(12 * CHUNK_LEN + off),
                    ptr.add(13 * CHUNK_LEN + off),
                    ptr.add(14 * CHUNK_LEN + off),
                    ptr.add(15 * CHUNK_LEN + off),
                ),
            ]
        }
    }

    pub fn compress_16_chunks(
        key: &[u32; 8],
        chunks: &[u8],
        start_counter: u64,
        flags: u32,
        out: &mut [[u32; 8]; 16],
    ) {
        assert!(chunks.len() == 16 * CHUNK_LEN);
        const { assert!(CHUNK_BLOCKS == 16) };
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let mut cv = [broadcast_key(key); 4];
            let mut t_lo = [vdupq_n_u32(0); 4];
            let mut t_hi = [vdupq_n_u32(0); 4];
            for grp in 0..4 {
                let b = start_counter + (grp * 4) as u64;
                let lo = [b as u32, (b + 1) as u32, (b + 2) as u32, (b + 3) as u32];
                let hi = [
                    (b >> 32) as u32,
                    ((b + 1) >> 32) as u32,
                    ((b + 2) >> 32) as u32,
                    ((b + 3) >> 32) as u32,
                ];
                t_lo[grp] = vld1q_u32(lo.as_ptr());
                t_hi[grp] = vld1q_u32(hi.as_ptr());
            }
            let ptr = chunks.as_ptr();
            for block in 0..CHUNK_BLOCKS {
                let m = load_16_chunk_blocks(ptr, block * BLOCK_LEN);
                let mut bflags = flags;
                if block == 0 {
                    bflags |= CHUNK_START;
                }
                if block == CHUNK_BLOCKS - 1 {
                    bflags |= CHUNK_END;
                }
                let mut s = [
                    init_state(&cv[0], t_lo[0], t_hi[0], BLOCK_LEN as u32, bflags),
                    init_state(&cv[1], t_lo[1], t_hi[1], BLOCK_LEN as u32, bflags),
                    init_state(&cv[2], t_lo[2], t_hi[2], BLOCK_LEN as u32, bflags),
                    init_state(&cv[3], t_lo[3], t_hi[3], BLOCK_LEN as u32, bflags),
                ];
                for sc in &MSG_SCHEDULE {
                    g16(&mut s, 0, 4, 8, 12, &m, sc[0], sc[1], rot8);
                    g16(&mut s, 1, 5, 9, 13, &m, sc[2], sc[3], rot8);
                    g16(&mut s, 2, 6, 10, 14, &m, sc[4], sc[5], rot8);
                    g16(&mut s, 3, 7, 11, 15, &m, sc[6], sc[7], rot8);
                    g16(&mut s, 0, 5, 10, 15, &m, sc[8], sc[9], rot8);
                    g16(&mut s, 1, 6, 11, 12, &m, sc[10], sc[11], rot8);
                    g16(&mut s, 2, 7, 8, 13, &m, sc[12], sc[13], rot8);
                    g16(&mut s, 3, 4, 9, 14, &m, sc[14], sc[15], rot8);
                }
                for (c, st) in cv.iter_mut().zip(&s) {
                    for i in 0..8 {
                        c[i] = veorq_u32(st[i], st[i + 8]);
                    }
                }
            }
            for (grp, c) in cv.iter().enumerate() {
                let lo = transpose_4x4(c[0], c[1], c[2], c[3]);
                let hi = transpose_4x4(c[4], c[5], c[6], c[7]);
                for lane in 0..4 {
                    vst1q_u32(out[grp * 4 + lane].as_mut_ptr(), lo[lane]);
                    vst1q_u32(out[grp * 4 + lane].as_mut_ptr().add(4), hi[lane]);
                }
            }
        }
    }

    pub fn keyed_hash_parent_16x(key: &[u32; 8], children: &[[u8; 32]; 32], out: &mut [[u8; OUT_LEN]; 16]) {
        const { assert!(BLOCK_LEN == 64 && OUT_LEN == 32) };
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let key_v = broadcast_key(key);
            let mut cv = [key_v; 4];
            let m = [
                load_4_blocks(
                    children[0].as_ptr(),
                    children[2].as_ptr(),
                    children[4].as_ptr(),
                    children[6].as_ptr(),
                ),
                load_4_blocks(
                    children[8].as_ptr(),
                    children[10].as_ptr(),
                    children[12].as_ptr(),
                    children[14].as_ptr(),
                ),
                load_4_blocks(
                    children[16].as_ptr(),
                    children[18].as_ptr(),
                    children[20].as_ptr(),
                    children[22].as_ptr(),
                ),
                load_4_blocks(
                    children[24].as_ptr(),
                    children[26].as_ptr(),
                    children[28].as_ptr(),
                    children[30].as_ptr(),
                ),
            ];
            compress_cv_16x(
                &mut cv,
                &m,
                0,
                BLOCK_LEN as u32,
                KEYED_HASH | CHUNK_START | CHUNK_END | ROOT,
                rot8,
            );
            let chunks = out.as_chunks_mut::<4>().0;
            for grp in 0..4 {
                store_4_digests(&cv[grp], &mut chunks[grp]);
            }
        }
    }

    pub fn keyed_hash_parent_8x(key: &[u32; 8], children: &[[u8; 32]; 16], out: &mut [[u8; OUT_LEN]; 8]) {
        const { assert!(BLOCK_LEN == 64 && OUT_LEN == 32) };
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let key_v = broadcast_key(key);
            let mut cv0 = key_v;
            let mut cv1 = key_v;
            let m0 = load_4_blocks(
                children[0].as_ptr(),
                children[2].as_ptr(),
                children[4].as_ptr(),
                children[6].as_ptr(),
            );
            let m1 = load_4_blocks(
                children[8].as_ptr(),
                children[10].as_ptr(),
                children[12].as_ptr(),
                children[14].as_ptr(),
            );
            compress_cv_8x(
                &mut cv0,
                &mut cv1,
                &m0,
                &m1,
                0,
                BLOCK_LEN as u32,
                KEYED_HASH | CHUNK_START | CHUNK_END | ROOT,
                rot8,
            );
            let chunks = out.as_chunks_mut::<4>().0;
            store_4_digests(&cv0, &mut chunks[0]);
            store_4_digests(&cv1, &mut chunks[1]);
        }
    }

    pub fn keyed_hash_parent_4x(key: &[u32; 8], children: &[[u8; 32]; 8], out: &mut [[u8; OUT_LEN]; 4]) {
        const { assert!(BLOCK_LEN == 64 && OUT_LEN == 32) };
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let mut cv = broadcast_key(key);
            let m = load_4_blocks(
                children[0].as_ptr(),
                children[2].as_ptr(),
                children[4].as_ptr(),
                children[6].as_ptr(),
            );
            compress_cv_4x(
                &mut cv,
                &m,
                0,
                BLOCK_LEN as u32,
                KEYED_HASH | CHUNK_START | CHUNK_END | ROOT,
                rot8,
            );
            store_4_digests(&cv, out);
        }
    }

    #[inline(always)]
    unsafe fn parent_cv_16x(
        left: &[[V; 8]; 4],
        right: &[[V; 8]; 4],
        key_v: &[V; 8],
        flags: u32,
        rot8: uint8x16_t,
    ) -> [[V; 8]; 4] {
        unsafe {
            let mut m = [[vdupq_n_u32(0); 16]; 4];
            for grp in 0..4 {
                m[grp][..8].copy_from_slice(&left[grp]);
                m[grp][8..].copy_from_slice(&right[grp]);
            }
            let mut cv = [*key_v; 4];
            compress_cv_16x(&mut cv, &m, 0, BLOCK_LEN as u32, flags | PARENT, rot8);
            cv
        }
    }

    #[inline(always)]
    unsafe fn push_chunk_cv_16x(
        stack: &mut [[[V; 8]; 4]; CV_STACK_LEN_MAX],
        stack_len: &mut usize,
        mut cv: [[V; 8]; 4],
        completed_chunks: u64,
        key_v: &[V; 8],
        rot8: uint8x16_t,
    ) {
        let mut merges = completed_chunks.trailing_zeros();
        while merges > 0 {
            assert!(*stack_len > 0);
            *stack_len -= 1;
            cv = unsafe { parent_cv_16x(&stack[*stack_len], &cv, key_v, KEYED_HASH, rot8) };
            merges -= 1;
        }
        assert!(*stack_len < CV_STACK_LEN_MAX);
        stack[*stack_len] = cv;
        *stack_len += 1;
    }

    #[inline(always)]
    unsafe fn finish_stack_16x(
        stack: &mut [[[V; 8]; 4]; CV_STACK_LEN_MAX],
        mut stack_len: usize,
        key_v: &[V; 8],
        rot8: uint8x16_t,
    ) -> [[V; 8]; 4] {
        assert!((1..=CV_STACK_LEN_MAX).contains(&stack_len));
        let mut right = stack[stack_len - 1];
        stack_len -= 1;
        while stack_len > 1 {
            stack_len -= 1;
            right = unsafe { parent_cv_16x(&stack[stack_len], &right, key_v, KEYED_HASH, rot8) };
        }
        assert!(stack_len == 1);
        unsafe { parent_cv_16x(&stack[0], &right, key_v, KEYED_HASH | ROOT, rot8) }
    }

    #[inline(always)]
    unsafe fn load_16_blocks_at(inputs: &[&[u8]; 16], off: usize) -> [[V; 16]; 4] {
        unsafe {
            [
                load_4_blocks(
                    inputs[0].as_ptr().add(off),
                    inputs[1].as_ptr().add(off),
                    inputs[2].as_ptr().add(off),
                    inputs[3].as_ptr().add(off),
                ),
                load_4_blocks(
                    inputs[4].as_ptr().add(off),
                    inputs[5].as_ptr().add(off),
                    inputs[6].as_ptr().add(off),
                    inputs[7].as_ptr().add(off),
                ),
                load_4_blocks(
                    inputs[8].as_ptr().add(off),
                    inputs[9].as_ptr().add(off),
                    inputs[10].as_ptr().add(off),
                    inputs[11].as_ptr().add(off),
                ),
                load_4_blocks(
                    inputs[12].as_ptr().add(off),
                    inputs[13].as_ptr().add(off),
                    inputs[14].as_ptr().add(off),
                    inputs[15].as_ptr().add(off),
                ),
            ]
        }
    }

    #[inline(always)]
    unsafe fn load_16_partial_blocks(inputs: &[&[u8]; 16], start: usize, end: usize) -> [[V; 16]; 4] {
        unsafe {
            [
                load_4_partial_blocks(
                    &inputs[0][start..end],
                    &inputs[1][start..end],
                    &inputs[2][start..end],
                    &inputs[3][start..end],
                ),
                load_4_partial_blocks(
                    &inputs[4][start..end],
                    &inputs[5][start..end],
                    &inputs[6][start..end],
                    &inputs[7][start..end],
                ),
                load_4_partial_blocks(
                    &inputs[8][start..end],
                    &inputs[9][start..end],
                    &inputs[10][start..end],
                    &inputs[11][start..end],
                ),
                load_4_partial_blocks(
                    &inputs[12][start..end],
                    &inputs[13][start..end],
                    &inputs[14][start..end],
                    &inputs[15][start..end],
                ),
            ]
        }
    }

    #[inline(always)]
    unsafe fn compress_chunk_16x(
        key_v: &[V; 8],
        inputs: &[&[u8]; 16],
        cstart: usize,
        cend: usize,
        chunk_idx: u64,
        extra_end_flags: u32,
        rot8: uint8x16_t,
    ) -> [[V; 8]; 4] {
        let clen = cend - cstart;
        assert!(clen <= CHUNK_LEN);
        unsafe {
            let mut cv = [*key_v; 4];
            let full_len = clen.saturating_sub(1) / BLOCK_LEN * BLOCK_LEN;
            let mut off = 0;
            while off < full_len {
                let m = load_16_blocks_at(inputs, cstart + off);
                let flags = KEYED_HASH | if off == 0 { CHUNK_START } else { 0 };
                compress_cv_16x(&mut cv, &m, chunk_idx, BLOCK_LEN as u32, flags, rot8);
                off += BLOCK_LEN;
            }
            let m = load_16_partial_blocks(inputs, cstart + off, cend);
            let end_flags = KEYED_HASH | if off == 0 { CHUNK_START } else { 0 } | CHUNK_END | extra_end_flags;
            compress_cv_16x(&mut cv, &m, chunk_idx, (clen - off) as u32, end_flags, rot8);
            cv
        }
    }

    #[cold]
    #[inline(never)]
    fn keyed_hash_16x_multi(key: &[u32; 8], inputs: &[&[u8]; 16], out: &mut [[u8; OUT_LEN]; 16]) {
        let len = inputs[0].len();
        assert!(len > CHUNK_LEN && inputs.iter().all(|s| s.len() == len));
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let key_v = broadcast_key(key);
            let total_chunks = len.div_ceil(CHUNK_LEN);
            let mut stack = [[[vdupq_n_u32(0); 8]; 4]; CV_STACK_LEN_MAX];
            let mut stack_len = 0usize;
            for chunk_idx in 0..total_chunks {
                let cstart = chunk_idx * CHUNK_LEN;
                let cend = (cstart + CHUNK_LEN).min(len);
                let cv = compress_chunk_16x(&key_v, inputs, cstart, cend, chunk_idx as u64, 0, rot8);
                if chunk_idx + 1 < total_chunks {
                    push_chunk_cv_16x(&mut stack, &mut stack_len, cv, (chunk_idx + 1) as u64, &key_v, rot8);
                } else {
                    stack[stack_len] = cv;
                    stack_len += 1;
                }
            }
            let root = finish_stack_16x(&mut stack, stack_len, &key_v, rot8);
            let chunks = out.as_chunks_mut::<4>().0;
            for (c, dst) in root.iter().zip(chunks.iter_mut()) {
                store_4_digests(c, dst);
            }
        }
    }

    pub fn keyed_hash_16x(key: &[u32; 8], inputs: &[&[u8]; 16], out: &mut [[u8; OUT_LEN]; 16]) {
        let len = inputs[0].len();
        assert!(inputs.iter().all(|s| s.len() == len));
        if len > CHUNK_LEN {
            keyed_hash_16x_multi(key, inputs, out);
            return;
        }
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let key_v = broadcast_key(key);
            let root = compress_chunk_16x(&key_v, inputs, 0, len, 0, ROOT, rot8);
            let chunks = out.as_chunks_mut::<4>().0;
            for (c, dst) in root.iter().zip(chunks.iter_mut()) {
                store_4_digests(c, dst);
            }
        }
    }

    pub fn keyed_hash_8x(key: &[u32; 8], inputs: [&[u8]; 8], out: &mut [[u8; OUT_LEN]; 8]) {
        let len = inputs[0].len();
        assert!(inputs.iter().all(|s| s.len() == len));
        if len > CHUNK_LEN {
            for (dst, input) in out.iter_mut().zip(inputs) {
                *dst = super::keyed_hash_words(key, input);
            }
            return;
        }
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let key_v = broadcast_key(key);
            let mut cv0 = key_v;
            let mut cv1 = key_v;
            let full_len = len.saturating_sub(1) / BLOCK_LEN * BLOCK_LEN;
            let mut off = 0;
            while off < full_len {
                let m0 = load_4_blocks(
                    inputs[0].as_ptr().add(off),
                    inputs[1].as_ptr().add(off),
                    inputs[2].as_ptr().add(off),
                    inputs[3].as_ptr().add(off),
                );
                let m1 = load_4_blocks(
                    inputs[4].as_ptr().add(off),
                    inputs[5].as_ptr().add(off),
                    inputs[6].as_ptr().add(off),
                    inputs[7].as_ptr().add(off),
                );
                let flags = KEYED_HASH | if off == 0 { CHUNK_START } else { 0 };
                compress_cv_8x(&mut cv0, &mut cv1, &m0, &m1, 0, BLOCK_LEN as u32, flags, rot8);
                off += BLOCK_LEN;
            }
            let m0 = load_4_partial_blocks(
                &inputs[0][off..],
                &inputs[1][off..],
                &inputs[2][off..],
                &inputs[3][off..],
            );
            let m1 = load_4_partial_blocks(
                &inputs[4][off..],
                &inputs[5][off..],
                &inputs[6][off..],
                &inputs[7][off..],
            );
            let end_flags = KEYED_HASH | if off == 0 { CHUNK_START } else { 0 } | CHUNK_END | ROOT;
            compress_cv_8x(&mut cv0, &mut cv1, &m0, &m1, 0, (len - off) as u32, end_flags, rot8);
            let chunks = out.as_chunks_mut::<4>().0;
            store_4_digests(&cv0, &mut chunks[0]);
            store_4_digests(&cv1, &mut chunks[1]);
        }
    }

    pub fn keyed_hash_4x(key: &[u32; 8], inputs: [&[u8]; 4], out: &mut [[u8; OUT_LEN]; 4]) {
        let len = inputs[0].len();
        assert!(inputs.iter().all(|s| s.len() == len));
        if len > CHUNK_LEN {
            for (dst, input) in out.iter_mut().zip(inputs) {
                *dst = super::keyed_hash_words(key, input);
            }
            return;
        }
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let mut cv = broadcast_key(key);
            let full_len = len.saturating_sub(1) / BLOCK_LEN * BLOCK_LEN;
            let mut off = 0;
            while off < full_len {
                let m = load_4_blocks(
                    inputs[0].as_ptr().add(off),
                    inputs[1].as_ptr().add(off),
                    inputs[2].as_ptr().add(off),
                    inputs[3].as_ptr().add(off),
                );
                let flags = KEYED_HASH | if off == 0 { CHUNK_START } else { 0 };
                compress_cv_4x(&mut cv, &m, 0, BLOCK_LEN as u32, flags, rot8);
                off += BLOCK_LEN;
            }
            let m = load_4_partial_blocks(
                &inputs[0][off..],
                &inputs[1][off..],
                &inputs[2][off..],
                &inputs[3][off..],
            );
            let end_flags = KEYED_HASH | if off == 0 { CHUNK_START } else { 0 } | CHUNK_END | ROOT;
            compress_cv_4x(&mut cv, &m, 0, (len - off) as u32, end_flags, rot8);
            store_4_digests(&cv, out);
        }
    }

    #[inline(always)]
    unsafe fn load_row_4x_32<F: crate::field::Field>(row: &[F], base: usize) -> [V; 8] {
        assert!(F::BYTES == 32 && base + 4 <= row.len());
        let mut raw = [0u8; 128];
        for lane in 0..4 {
            row[base + lane].write_bytes(&mut raw[lane * 32..lane * 32 + 32]);
        }
        unsafe {
            let ptr = raw.as_ptr();
            let lo = transpose_4x4(
                load_u32x4(ptr),
                load_u32x4(ptr.add(32)),
                load_u32x4(ptr.add(64)),
                load_u32x4(ptr.add(96)),
            );
            let hi = transpose_4x4(
                load_u32x4(ptr.add(16)),
                load_u32x4(ptr.add(48)),
                load_u32x4(ptr.add(80)),
                load_u32x4(ptr.add(112)),
            );
            [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
        }
    }

    #[inline(always)]
    unsafe fn load_row_4x_16<F: crate::field::Field>(row: &[F], base: usize) -> [V; 4] {
        assert!(F::BYTES == 16 && base + 4 <= row.len());
        let mut raw = [0u8; 64];
        for lane in 0..4 {
            row[base + lane].write_bytes(&mut raw[lane * 16..lane * 16 + 16]);
        }
        unsafe {
            let ptr = raw.as_ptr();
            transpose_4x4(
                load_u32x4(ptr),
                load_u32x4(ptr.add(16)),
                load_u32x4(ptr.add(32)),
                load_u32x4(ptr.add(48)),
            )
        }
    }

    #[inline(always)]
    unsafe fn init_pos_carry_16x(base: usize) -> [[V; 2]; 4] {
        unsafe {
            let mut carry = [[vdupq_n_u32(0); 2]; 4];
            for (grp, slot) in carry.iter_mut().enumerate() {
                let b = (base + grp * 4) as u64;
                let lo = [b as u32, (b + 1) as u32, (b + 2) as u32, (b + 3) as u32];
                let hi = [
                    (b >> 32) as u32,
                    ((b + 1) >> 32) as u32,
                    ((b + 2) >> 32) as u32,
                    ((b + 3) >> 32) as u32,
                ];
                *slot = [vld1q_u32(lo.as_ptr()), vld1q_u32(hi.as_ptr())];
            }
            carry
        }
    }

    #[inline(always)]
    unsafe fn init_pos_carry_8x(base: usize) -> [[V; 2]; 2] {
        unsafe {
            let b0 = base as u64;
            let b1 = (base + 4) as u64;
            let lo0 = [b0 as u32, (b0 + 1) as u32, (b0 + 2) as u32, (b0 + 3) as u32];
            let hi0 = [
                (b0 >> 32) as u32,
                ((b0 + 1) >> 32) as u32,
                ((b0 + 2) >> 32) as u32,
                ((b0 + 3) >> 32) as u32,
            ];
            let lo1 = [b1 as u32, (b1 + 1) as u32, (b1 + 2) as u32, (b1 + 3) as u32];
            let hi1 = [
                (b1 >> 32) as u32,
                ((b1 + 1) >> 32) as u32,
                ((b1 + 2) >> 32) as u32,
                ((b1 + 3) >> 32) as u32,
            ];
            [
                [vld1q_u32(lo0.as_ptr()), vld1q_u32(hi0.as_ptr())],
                [vld1q_u32(lo1.as_ptr()), vld1q_u32(hi1.as_ptr())],
            ]
        }
    }

    #[inline(always)]
    unsafe fn build_block_f32<F: crate::field::Field>(
        r0: &[F],
        r1: &[F],
        base: usize,
        carry: &mut [V; 2],
    ) -> [V; 16] {
        unsafe {
            let a = load_row_4x_32(r0, base);
            let b = load_row_4x_32(r1, base);
            let m = [
                carry[0], carry[1], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0], b[1], b[2], b[3],
                b[4], b[5],
            ];
            *carry = [b[6], b[7]];
            m
        }
    }

    #[inline(always)]
    unsafe fn build_block_f16<F: crate::field::Field>(
        rows: &[Vec<F>],
        base: usize,
        carry: &mut [V; 2],
    ) -> [V; 16] {
        assert!(rows.len() >= 4);
        unsafe {
            let r0 = load_row_4x_16(&rows[0], base);
            let r1 = load_row_4x_16(&rows[1], base);
            let r2 = load_row_4x_16(&rows[2], base);
            let r3 = load_row_4x_16(&rows[3], base);
            let m = [
                carry[0], carry[1], r0[0], r0[1], r0[2], r0[3], r1[0], r1[1], r1[2], r1[3], r2[0], r2[1],
                r2[2], r2[3], r3[0], r3[1],
            ];
            *carry = [r3[2], r3[3]];
            m
        }
    }

    struct Rows16State {
        cv: [[V; 8]; 4],
        stack: [[[V; 8]; 4]; CV_STACK_LEN_MAX],
        stack_len: usize,
        block_in_chunk: usize,
        chunk_counter: u64,
        key_v: [V; 8],
        rot8: uint8x16_t,
    }

    impl Rows16State {
        #[inline(always)]
        unsafe fn new(key: &[u32; 8]) -> Self {
            unsafe {
                let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
                let key_v = broadcast_key(key);
                Self {
                    cv: [key_v; 4],
                    stack: [[[vdupq_n_u32(0); 8]; 4]; CV_STACK_LEN_MAX],
                    stack_len: 0,
                    block_in_chunk: 0,
                    chunk_counter: 0,
                    key_v,
                    rot8,
                }
            }
        }

        #[inline(always)]
        unsafe fn push_full_block(&mut self, m: &[[V; 16]; 4]) {
            let mut flags = KEYED_HASH;
            if self.block_in_chunk == 0 {
                flags |= CHUNK_START;
            }
            let is_end = self.block_in_chunk == CHUNK_BLOCKS - 1;
            if is_end {
                flags |= CHUNK_END;
            }
            unsafe {
                compress_cv_16x(&mut self.cv, m, self.chunk_counter, BLOCK_LEN as u32, flags, self.rot8)
            };
            if is_end {
                self.chunk_counter += 1;
                unsafe {
                    push_chunk_cv_16x(
                        &mut self.stack,
                        &mut self.stack_len,
                        self.cv,
                        self.chunk_counter,
                        &self.key_v,
                        self.rot8,
                    );
                }
                self.cv = [self.key_v; 4];
                self.block_in_chunk = 0;
            } else {
                self.block_in_chunk += 1;
            }
        }

        #[inline(always)]
        fn reset(&mut self) {
            self.cv = [self.key_v; 4];
            self.stack_len = 0;
            self.block_in_chunk = 0;
            self.chunk_counter = 0;
        }

        #[inline(always)]
        unsafe fn finish(&mut self, m: &[[V; 16]; 4], tail_len: u32, out: &mut [[u8; OUT_LEN]; 16]) {
            let mut flags = KEYED_HASH | CHUNK_END;
            if self.block_in_chunk == 0 {
                flags |= CHUNK_START;
            }
            if self.stack_len == 0 {
                flags |= ROOT;
            }
            unsafe {
                compress_cv_16x(&mut self.cv, m, self.chunk_counter, tail_len, flags, self.rot8);
                let final_cv = if self.stack_len == 0 {
                    self.cv
                } else {
                    self.stack[self.stack_len] = self.cv;
                    finish_stack_16x(&mut self.stack, self.stack_len + 1, &self.key_v, self.rot8)
                };
                let chunks = out.as_chunks_mut::<4>().0;
                for (c, dst) in final_cv.iter().zip(chunks.iter_mut()) {
                    store_4_digests(c, dst);
                }
            }
        }
    }

    #[inline(never)]
    fn hash_rows_16x_f32_single<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        start: usize,
        out: &mut [[[u8; OUT_LEN]; 16]],
    ) {
        assert!(F::BYTES == 32 && !rows.is_empty());
        assert!(8 + rows.len() * 32 <= CHUNK_LEN && start + out.len() * 16 <= rows[0].len());
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let zero = vdupq_n_u32(0);
            let key_v = broadcast_key(key);
            let pairs = rows.len() / 2;
            let odd = rows.len() % 2 == 1;
            for (group, dst) in out.iter_mut().enumerate() {
                let base = start + group * 16;
                let mut cv = [key_v; 4];
                let mut carry = init_pos_carry_16x(base);
                for p in 0..pairs {
                    let m = [
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base, &mut carry[0]),
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base + 4, &mut carry[1]),
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base + 8, &mut carry[2]),
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base + 12, &mut carry[3]),
                    ];
                    let flags = KEYED_HASH | if p == 0 { CHUNK_START } else { 0 };
                    compress_cv_16x(&mut cv, &m, 0, BLOCK_LEN as u32, flags, rot8);
                }
                let mut m = [[zero; 16]; 4];
                let tail_len = if odd {
                    let last = &rows[rows.len() - 1];
                    for grp in 0..4 {
                        m[grp][0] = carry[grp][0];
                        m[grp][1] = carry[grp][1];
                        m[grp][2..10].copy_from_slice(&load_row_4x_32(last, base + grp * 4));
                    }
                    40u32
                } else {
                    for grp in 0..4 {
                        m[grp][0] = carry[grp][0];
                        m[grp][1] = carry[grp][1];
                    }
                    8u32
                };
                let flags = KEYED_HASH | if pairs == 0 { CHUNK_START } else { 0 } | CHUNK_END | ROOT;
                compress_cv_16x(&mut cv, &m, 0, tail_len, flags, rot8);
                let chunks = dst.as_chunks_mut::<4>().0;
                for (c, d) in cv.iter().zip(chunks.iter_mut()) {
                    store_4_digests(c, d);
                }
            }
        }
    }

    #[inline(never)]
    fn hash_rows_16x_f16_single<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        start: usize,
        out: &mut [[[u8; OUT_LEN]; 16]],
    ) {
        assert!(F::BYTES == 16 && !rows.is_empty());
        assert!(8 + rows.len() * 16 <= CHUNK_LEN && start + out.len() * 16 <= rows[0].len());
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let zero = vdupq_n_u32(0);
            let key_v = broadcast_key(key);
            let quads = rows.len() / 4;
            let rem = rows.len() % 4;
            for (group, dst) in out.iter_mut().enumerate() {
                let base = start + group * 16;
                let mut cv = [key_v; 4];
                let mut carry = init_pos_carry_16x(base);
                for q in 0..quads {
                    let slice = &rows[q * 4..q * 4 + 4];
                    let m = [
                        build_block_f16(slice, base, &mut carry[0]),
                        build_block_f16(slice, base + 4, &mut carry[1]),
                        build_block_f16(slice, base + 8, &mut carry[2]),
                        build_block_f16(slice, base + 12, &mut carry[3]),
                    ];
                    let flags = KEYED_HASH | if q == 0 { CHUNK_START } else { 0 };
                    compress_cv_16x(&mut cv, &m, 0, BLOCK_LEN as u32, flags, rot8);
                }
                let mut m = [[zero; 16]; 4];
                for grp in 0..4 {
                    m[grp][0] = carry[grp][0];
                    m[grp][1] = carry[grp][1];
                    for r in 0..rem {
                        let v = load_row_4x_16(&rows[quads * 4 + r], base + grp * 4);
                        m[grp][2 + r * 4..6 + r * 4].copy_from_slice(&v);
                    }
                }
                let flags = KEYED_HASH | if quads == 0 { CHUNK_START } else { 0 } | CHUNK_END | ROOT;
                compress_cv_16x(&mut cv, &m, 0, (8 + rem * 16) as u32, flags, rot8);
                let chunks = dst.as_chunks_mut::<4>().0;
                for (c, d) in cv.iter().zip(chunks.iter_mut()) {
                    store_4_digests(c, d);
                }
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn hash_rows_16x_f32_multi<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        start: usize,
        out: &mut [[[u8; OUT_LEN]; 16]],
    ) {
        assert!(F::BYTES == 32 && 8 + rows.len() * 32 > CHUNK_LEN);
        assert!(start + out.len() * 16 <= rows[0].len());
        unsafe {
            let mut st = Rows16State::new(key);
            let zero = vdupq_n_u32(0);
            for (group, dst) in out.iter_mut().enumerate() {
                st.reset();
                let base = start + group * 16;
                let mut carry = init_pos_carry_16x(base);
                for p in 0..(rows.len() / 2) {
                    let m = [
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base, &mut carry[0]),
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base + 4, &mut carry[1]),
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base + 8, &mut carry[2]),
                        build_block_f32(&rows[2 * p], &rows[2 * p + 1], base + 12, &mut carry[3]),
                    ];
                    st.push_full_block(&m);
                }
                let mut m = [[zero; 16]; 4];
                let tail_len = if rows.len() % 2 == 1 {
                    let last = &rows[rows.len() - 1];
                    for grp in 0..4 {
                        m[grp][0] = carry[grp][0];
                        m[grp][1] = carry[grp][1];
                        m[grp][2..10].copy_from_slice(&load_row_4x_32(last, base + grp * 4));
                    }
                    40u32
                } else {
                    for grp in 0..4 {
                        m[grp][0] = carry[grp][0];
                        m[grp][1] = carry[grp][1];
                    }
                    8u32
                };
                st.finish(&m, tail_len, dst);
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn hash_rows_16x_f16_multi<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        start: usize,
        out: &mut [[[u8; OUT_LEN]; 16]],
    ) {
        assert!(F::BYTES == 16 && 8 + rows.len() * 16 > CHUNK_LEN);
        assert!(start + out.len() * 16 <= rows[0].len());
        unsafe {
            let mut st = Rows16State::new(key);
            let zero = vdupq_n_u32(0);
            let quads = rows.len() / 4;
            let rem = rows.len() % 4;
            for (group, dst) in out.iter_mut().enumerate() {
                st.reset();
                let base = start + group * 16;
                let mut carry = init_pos_carry_16x(base);
                for q in 0..quads {
                    let slice = &rows[q * 4..q * 4 + 4];
                    let m = [
                        build_block_f16(slice, base, &mut carry[0]),
                        build_block_f16(slice, base + 4, &mut carry[1]),
                        build_block_f16(slice, base + 8, &mut carry[2]),
                        build_block_f16(slice, base + 12, &mut carry[3]),
                    ];
                    st.push_full_block(&m);
                }
                let mut m = [[zero; 16]; 4];
                for grp in 0..4 {
                    m[grp][0] = carry[grp][0];
                    m[grp][1] = carry[grp][1];
                    for r in 0..rem {
                        let v = load_row_4x_16(&rows[quads * 4 + r], base + grp * 4);
                        m[grp][2 + r * 4..6 + r * 4].copy_from_slice(&v);
                    }
                }
                st.finish(&m, (8 + rem * 16) as u32, dst);
            }
        }
    }

    pub fn hash_rows_16x_direct<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        start: usize,
        out: &mut [[[u8; OUT_LEN]; 16]],
    ) {
        assert!(F::BYTES == 16 || F::BYTES == 32);
        assert!(!rows.is_empty());
        let single = 8 + rows.len() * F::BYTES <= CHUNK_LEN;
        if F::BYTES == 32 {
            if single {
                hash_rows_16x_f32_single(key, rows, start, out);
            } else {
                hash_rows_16x_f32_multi(key, rows, start, out);
            }
        } else if single {
            hash_rows_16x_f16_single(key, rows, start, out);
        } else {
            hash_rows_16x_f16_multi(key, rows, start, out);
        }
    }

    fn hash_rows_8x_f32<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 8],
    ) {
        assert!(F::BYTES == 32 && 8 + rows.len() * 32 <= CHUNK_LEN);
        assert!(base + 8 <= rows[0].len());
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let zero = vdupq_n_u32(0);
            let mut cv0 = broadcast_key(key);
            let mut cv1 = cv0;
            let mut carry = init_pos_carry_8x(base);
            let mut first = true;
            for p in 0..(rows.len() / 2) {
                let m0 = build_block_f32(&rows[2 * p], &rows[2 * p + 1], base, &mut carry[0]);
                let m1 = build_block_f32(&rows[2 * p], &rows[2 * p + 1], base + 4, &mut carry[1]);
                let flags = KEYED_HASH | if first { CHUNK_START } else { 0 };
                compress_cv_8x(&mut cv0, &mut cv1, &m0, &m1, 0, BLOCK_LEN as u32, flags, rot8);
                first = false;
            }
            let (mut m0, mut m1) = ([zero; 16], [zero; 16]);
            m0[0] = carry[0][0];
            m0[1] = carry[0][1];
            m1[0] = carry[1][0];
            m1[1] = carry[1][1];
            let tail_len = if rows.len() % 2 == 1 {
                let last = &rows[rows.len() - 1];
                m0[2..10].copy_from_slice(&load_row_4x_32(last, base));
                m1[2..10].copy_from_slice(&load_row_4x_32(last, base + 4));
                40u32
            } else {
                8u32
            };
            let flags = KEYED_HASH | if first { CHUNK_START } else { 0 } | CHUNK_END | ROOT;
            compress_cv_8x(&mut cv0, &mut cv1, &m0, &m1, 0, tail_len, flags, rot8);
            let chunks = out.as_chunks_mut::<4>().0;
            store_4_digests(&cv0, &mut chunks[0]);
            store_4_digests(&cv1, &mut chunks[1]);
        }
    }

    fn hash_rows_8x_f16<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 8],
    ) {
        assert!(F::BYTES == 16 && 8 + rows.len() * 16 <= CHUNK_LEN);
        assert!(base + 8 <= rows[0].len());
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let zero = vdupq_n_u32(0);
            let mut cv0 = broadcast_key(key);
            let mut cv1 = cv0;
            let mut carry = init_pos_carry_8x(base);
            let mut first = true;
            let quads = rows.len() / 4;
            for q in 0..quads {
                let slice = &rows[q * 4..q * 4 + 4];
                let m0 = build_block_f16(slice, base, &mut carry[0]);
                let m1 = build_block_f16(slice, base + 4, &mut carry[1]);
                let flags = KEYED_HASH | if first { CHUNK_START } else { 0 };
                compress_cv_8x(&mut cv0, &mut cv1, &m0, &m1, 0, BLOCK_LEN as u32, flags, rot8);
                first = false;
            }
            let rem = rows.len() % 4;
            let (mut m0, mut m1) = ([zero; 16], [zero; 16]);
            m0[0] = carry[0][0];
            m0[1] = carry[0][1];
            m1[0] = carry[1][0];
            m1[1] = carry[1][1];
            for r in 0..rem {
                let row = &rows[quads * 4 + r];
                m0[2 + r * 4..6 + r * 4].copy_from_slice(&load_row_4x_16(row, base));
                m1[2 + r * 4..6 + r * 4].copy_from_slice(&load_row_4x_16(row, base + 4));
            }
            let flags = KEYED_HASH | if first { CHUNK_START } else { 0 } | CHUNK_END | ROOT;
            compress_cv_8x(&mut cv0, &mut cv1, &m0, &m1, 0, (8 + rem * 16) as u32, flags, rot8);
            let chunks = out.as_chunks_mut::<4>().0;
            store_4_digests(&cv0, &mut chunks[0]);
            store_4_digests(&cv1, &mut chunks[1]);
        }
    }

    pub fn hash_rows_8x_direct<F: crate::field::Field>(
        key: &[u32; 8],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 8],
    ) {
        assert!(F::BYTES == 16 || F::BYTES == 32);
        assert!(8 + rows.len() * F::BYTES <= CHUNK_LEN);
        if F::BYTES == 32 {
            hash_rows_8x_f32(key, rows, base, out);
        } else {
            hash_rows_8x_f16(key, rows, base, out);
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn g16_pow<const WX: usize, const WY: usize>(
        s: &mut [[V; 16]; 4],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        seed: &[V; 8],
        nonces: &[V; 4],
        rot8: uint8x16_t,
    ) {
        unsafe {
            for (st, &nn) in s.iter_mut().zip(nonces) {
                st[a] = if WX < 8 {
                    vaddq_u32(vaddq_u32(st[a], seed[WX]), st[b])
                } else if WX == 8 {
                    vaddq_u32(vaddq_u32(st[a], nn), st[b])
                } else {
                    vaddq_u32(st[a], st[b])
                };
            }
            for st in s.iter_mut() {
                st[d] = rotr16(veorq_u32(st[d], st[a]));
            }
            for st in s.iter_mut() {
                st[c] = vaddq_u32(st[c], st[d]);
            }
            for st in s.iter_mut() {
                st[b] = rotr12(veorq_u32(st[b], st[c]));
            }
            for (st, &nn) in s.iter_mut().zip(nonces) {
                st[a] = if WY < 8 {
                    vaddq_u32(vaddq_u32(st[a], seed[WY]), st[b])
                } else if WY == 8 {
                    vaddq_u32(vaddq_u32(st[a], nn), st[b])
                } else {
                    vaddq_u32(st[a], st[b])
                };
            }
            for st in s.iter_mut() {
                st[d] = rotr8(veorq_u32(st[d], st[a]), rot8);
            }
            for st in s.iter_mut() {
                st[c] = vaddq_u32(st[c], st[d]);
            }
            for st in s.iter_mut() {
                st[b] = rotr7(veorq_u32(st[b], st[c]));
            }
        }
    }

    macro_rules! round16_pow {
        ($r:expr, $s:expr, $seed:expr, $n:expr, $rot8:expr) => {
            g16_pow::<{ MSG_SCHEDULE[$r][0] }, { MSG_SCHEDULE[$r][1] }>($s, 0, 4, 8, 12, $seed, $n, $rot8);
            g16_pow::<{ MSG_SCHEDULE[$r][2] }, { MSG_SCHEDULE[$r][3] }>($s, 1, 5, 9, 13, $seed, $n, $rot8);
            g16_pow::<{ MSG_SCHEDULE[$r][4] }, { MSG_SCHEDULE[$r][5] }>($s, 2, 6, 10, 14, $seed, $n, $rot8);
            g16_pow::<{ MSG_SCHEDULE[$r][6] }, { MSG_SCHEDULE[$r][7] }>($s, 3, 7, 11, 15, $seed, $n, $rot8);
            g16_pow::<{ MSG_SCHEDULE[$r][8] }, { MSG_SCHEDULE[$r][9] }>($s, 0, 5, 10, 15, $seed, $n, $rot8);
            g16_pow::<{ MSG_SCHEDULE[$r][10] }, { MSG_SCHEDULE[$r][11] }>($s, 1, 6, 11, 12, $seed, $n, $rot8);
            g16_pow::<{ MSG_SCHEDULE[$r][12] }, { MSG_SCHEDULE[$r][13] }>($s, 2, 7, 8, 13, $seed, $n, $rot8);
            g16_pow::<{ MSG_SCHEDULE[$r][14] }, { MSG_SCHEDULE[$r][15] }>($s, 3, 4, 9, 14, $seed, $n, $rot8);
        };
    }

    fn precompute_pow_r0(key: &[u32; 8], seed_words: &[u32; 8]) -> [u32; 16] {
        let mut s = [
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
            IV[0],
            IV[1],
            IV[2],
            IV[3],
            0,
            0,
            40,
            KEYED_HASH | CHUNK_START | CHUNK_END | ROOT,
        ];
        super::g(&mut s, 0, 4, 8, 12, seed_words[0], seed_words[1]);
        super::g(&mut s, 1, 5, 9, 13, seed_words[2], seed_words[3]);
        super::g(&mut s, 2, 6, 10, 14, seed_words[4], seed_words[5]);
        super::g(&mut s, 3, 7, 11, 15, seed_words[6], seed_words[7]);
        super::g(&mut s, 1, 6, 11, 12, 0, 0);
        super::g(&mut s, 2, 7, 8, 13, 0, 0);
        super::g(&mut s, 3, 4, 9, 14, 0, 0);
        s
    }

    #[inline(always)]
    unsafe fn round6_pow_16x(s: &mut [[V; 16]; 4], seed: &[V; 8], nonces: &[V; 4], rot8: uint8x16_t) {
        const { assert!(MSG_SCHEDULE[6][8] == 14 && MSG_SCHEDULE[6][9] == 10) };
        const { assert!(MSG_SCHEDULE[6][12] == 3 && MSG_SCHEDULE[6][13] == 4) };
        unsafe {
            g16_pow::<{ MSG_SCHEDULE[6][0] }, { MSG_SCHEDULE[6][1] }>(s, 0, 4, 8, 12, seed, nonces, rot8);
            g16_pow::<{ MSG_SCHEDULE[6][2] }, { MSG_SCHEDULE[6][3] }>(s, 1, 5, 9, 13, seed, nonces, rot8);
            g16_pow::<{ MSG_SCHEDULE[6][4] }, { MSG_SCHEDULE[6][5] }>(s, 2, 6, 10, 14, seed, nonces, rot8);
            g16_pow::<{ MSG_SCHEDULE[6][6] }, { MSG_SCHEDULE[6][7] }>(s, 3, 7, 11, 15, seed, nonces, rot8);
            for st in s.iter_mut() {
                st[0] = vaddq_u32(st[0], st[5]);
            }
            for st in s.iter_mut() {
                st[15] = rotr16(veorq_u32(st[15], st[0]));
            }
            for st in s.iter_mut() {
                st[10] = vaddq_u32(st[10], st[15]);
            }
            for st in s.iter_mut() {
                st[5] = rotr12(veorq_u32(st[5], st[10]));
            }
            for st in s.iter_mut() {
                st[0] = vaddq_u32(st[0], st[5]);
            }
            for st in s.iter_mut() {
                st[2] = vaddq_u32(vaddq_u32(st[2], seed[3]), st[7]);
            }
            for st in s.iter_mut() {
                st[13] = rotr16(veorq_u32(st[13], st[2]));
            }
            for st in s.iter_mut() {
                st[8] = vaddq_u32(st[8], st[13]);
            }
            for st in s.iter_mut() {
                st[7] = rotr12(veorq_u32(st[7], st[8]));
            }
            for st in s.iter_mut() {
                st[2] = vaddq_u32(vaddq_u32(st[2], seed[4]), st[7]);
            }
            for st in s.iter_mut() {
                st[13] = rotr8(veorq_u32(st[13], st[2]), rot8);
            }
            for st in s.iter_mut() {
                st[8] = vaddq_u32(st[8], st[13]);
            }
        }
    }

    #[inline(always)]
    unsafe fn init_pow_nonces_32(start: u32) -> [V; 4] {
        unsafe {
            let mut nonces = [vdupq_n_u32(0); 4];
            for (grp, slot) in nonces.iter_mut().enumerate() {
                let b = start.wrapping_add((grp * 4) as u32);
                let arr = [b, b.wrapping_add(1), b.wrapping_add(2), b.wrapping_add(3)];
                *slot = vld1q_u32(arr.as_ptr());
            }
            nonces
        }
    }

    fn find_pow_nonce_16x_fast32(
        key: &[u32; 8],
        seed: &[u8; 32],
        bits: u32,
        start: u32,
        groups: usize,
    ) -> Option<u64> {
        assert!(bits > 0 && bits <= 32);
        let seed_words = super::words_from_key(seed);
        let r0_scalar = precompute_pow_r0(key, &seed_words);
        let mask_be = if bits == 32 { !0u32 } else { !0u32 << (32 - bits) };
        let mask_le = mask_be.swap_bytes();
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let mut r0 = [vdupq_n_u32(0); 16];
            for (dst, &word) in r0.iter_mut().zip(&r0_scalar) {
                *dst = vdupq_n_u32(word);
            }
            let seed_v = broadcast_key(&seed_words);
            let mask_v = vdupq_n_u32(mask_le);
            let step_v = vdupq_n_u32(16);
            let mut nonces = init_pow_nonces_32(start);
            for grp in 0..groups {
                let mut s = [r0; 4];
                g16_pow::<8, 9>(&mut s, 0, 5, 10, 15, &seed_v, &nonces, rot8);
                round16_pow!(1, &mut s, &seed_v, &nonces, rot8);
                round16_pow!(2, &mut s, &seed_v, &nonces, rot8);
                round16_pow!(3, &mut s, &seed_v, &nonces, rot8);
                round16_pow!(4, &mut s, &seed_v, &nonces, rot8);
                round16_pow!(5, &mut s, &seed_v, &nonces, rot8);
                round6_pow_16x(&mut s, &seed_v, &nonces, rot8);
                let w0 = vandq_u32(veorq_u32(s[0][0], s[0][8]), mask_v);
                let w1 = vandq_u32(veorq_u32(s[1][0], s[1][8]), mask_v);
                let w2 = vandq_u32(veorq_u32(s[2][0], s[2][8]), mask_v);
                let w3 = vandq_u32(veorq_u32(s[3][0], s[3][8]), mask_v);
                if vminvq_u32(vminq_u32(vminq_u32(w0, w1), vminq_u32(w2, w3))) == 0 {
                    let base = start as u64 + (grp as u64) * 16;
                    for lane in 0..16u64 {
                        if super::pow_bits_with_key_words(key, seed, base + lane) >= bits {
                            return Some(base + lane);
                        }
                    }
                }
                for n in &mut nonces {
                    *n = vaddq_u32(*n, step_v);
                }
            }
        }
        None
    }

    fn find_pow_nonce_16x_general(
        key: &[u32; 8],
        seed: &[u8; 32],
        bits: u32,
        start: u64,
        groups: usize,
    ) -> Option<u64> {
        assert!(bits > 0 && bits <= 64);
        let seed_words = super::words_from_key(seed);
        let mask_lo = if bits >= 32 { !0u32 } else { (!0u32 << (32 - bits)).swap_bytes() };
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let key_v = broadcast_key(key);
            let mut m_base = [vdupq_n_u32(0); 16];
            for (dst, &word) in m_base[..8].iter_mut().zip(&seed_words) {
                *dst = vdupq_n_u32(word);
            }
            let mask_v = vdupq_n_u32(mask_lo);
            for grp in 0..groups {
                let base = start + (grp as u64) * 16;
                let mut m = [m_base; 4];
                for (k, block) in m.iter_mut().enumerate() {
                    let b = base + (k as u64) * 4;
                    let lo = [
                        b as u32,
                        b.wrapping_add(1) as u32,
                        b.wrapping_add(2) as u32,
                        b.wrapping_add(3) as u32,
                    ];
                    let hi = [
                        (b >> 32) as u32,
                        (b.wrapping_add(1) >> 32) as u32,
                        (b.wrapping_add(2) >> 32) as u32,
                        (b.wrapping_add(3) >> 32) as u32,
                    ];
                    block[8] = vld1q_u32(lo.as_ptr());
                    block[9] = vld1q_u32(hi.as_ptr());
                }
                let mut cv = [key_v; 4];
                compress_cv_16x(&mut cv, &m, 0, 40, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT, rot8);
                let w0 = vandq_u32(cv[0][0], mask_v);
                let w1 = vandq_u32(cv[1][0], mask_v);
                let w2 = vandq_u32(cv[2][0], mask_v);
                let w3 = vandq_u32(cv[3][0], mask_v);
                if vminvq_u32(vminq_u32(vminq_u32(w0, w1), vminq_u32(w2, w3))) == 0 {
                    for lane in 0..16u64 {
                        if super::pow_bits_with_key_words(key, seed, base + lane) >= bits {
                            return Some(base + lane);
                        }
                    }
                }
            }
        }
        None
    }

    pub fn find_pow_nonce_16x(
        key: &[u32; 8],
        seed: &[u8; 32],
        bits: u32,
        start: u64,
        count: usize,
    ) -> Option<u64> {
        assert!(bits <= 64);
        assert!(count == 0 || start <= u64::MAX - (count as u64 - 1));
        if count == 0 {
            return None;
        }
        if bits == 0 {
            return Some(start);
        }
        let end = start + (count as u64 - 1);
        let groups = count / 16;
        let found = if bits <= 32 && end <= u32::MAX as u64 {
            find_pow_nonce_16x_fast32(key, seed, bits, start as u32, groups)
        } else {
            find_pow_nonce_16x_general(key, seed, bits, start, groups)
        };
        if found.is_some() {
            return found;
        }
        for offset in (groups * 16)..count {
            let nonce = start + offset as u64;
            if super::pow_bits_with_key_words(key, seed, nonce) >= bits {
                return Some(nonce);
            }
        }
        None
    }
}

#[derive(Clone)]
struct Output {
    input_chaining_value: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        first_8_words(compress(
            &self.input_chaining_value,
            &self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        ))
    }

    fn root_output_bytes(&self, out_slice: &mut [u8]) {
        for (output_block_counter, out_block) in out_slice.chunks_mut(2 * OUT_LEN).enumerate() {
            let words = compress(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter as u64,
                self.block_len,
                self.flags | ROOT,
            );
            for (word, out_word) in words.iter().zip(out_block.chunks_mut(4)) {
                out_word.copy_from_slice(&word.to_le_bytes()[..out_word.len()]);
            }
        }
    }
}

#[derive(Clone)]
struct ChunkState {
    chaining_value: [u32; 8],
    chunk_counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: u8,
    blocks_compressed: u8,
    flags: u32,
}

impl ChunkState {
    fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32) -> Self {
        Self {
            chaining_value: key_words,
            chunk_counter,
            block: [0; BLOCK_LEN],
            block_len: 0,
            blocks_compressed: 0,
            flags,
        }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed as usize + self.block_len as usize
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.block_len as usize == BLOCK_LEN {
                let block_words = words_from_block(&self.block);
                self.chaining_value = first_8_words(compress(
                    &self.chaining_value,
                    &block_words,
                    self.chunk_counter,
                    BLOCK_LEN as u32,
                    self.flags | self.start_flag(),
                ));
                self.blocks_compressed += 1;
                self.block = [0; BLOCK_LEN];
                self.block_len = 0;
            }
            let take = (BLOCK_LEN - self.block_len as usize).min(input.len());
            self.block[self.block_len as usize..self.block_len as usize + take]
                .copy_from_slice(&input[..take]);
            self.block_len += take as u8;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        let block_words = words_from_block(&self.block);
        Output {
            input_chaining_value: self.chaining_value,
            block_words,
            counter: self.chunk_counter,
            block_len: self.block_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

fn parent_output(
    left_child_cv: [u32; 8],
    right_child_cv: [u32; 8],
    key_words: [u32; 8],
    flags: u32,
) -> Output {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&left_child_cv);
    block_words[8..].copy_from_slice(&right_child_cv);
    Output {
        input_chaining_value: key_words,
        block_words,
        counter: 0,
        block_len: BLOCK_LEN as u32,
        flags: PARENT | flags,
    }
}

#[derive(Clone)]
pub struct Hasher {
    chunk_state: ChunkState,
    key_words: [u32; 8],
    cv_stack: [[u32; 8]; CV_STACK_LEN_MAX],
    cv_stack_len: u8,
    flags: u32,
}

impl Hasher {
    fn new_internal(key_words: [u32; 8], flags: u32) -> Self {
        Self {
            chunk_state: ChunkState::new(key_words, 0, flags),
            key_words,
            cv_stack: [[0; 8]; CV_STACK_LEN_MAX],
            cv_stack_len: 0,
            flags,
        }
    }

    pub fn new() -> Self {
        Self::new_internal(IV, 0)
    }

    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        Self::new_internal(words_from_key(key), KEYED_HASH)
    }

    pub(crate) fn new_keyed_words(key_words: [u32; 8]) -> Self {
        Self::new_internal(key_words, KEYED_HASH)
    }

    fn push_stack(&mut self, cv: [u32; 8]) {
        assert!((self.cv_stack_len as usize) < CV_STACK_LEN_MAX);
        self.cv_stack[self.cv_stack_len as usize] = cv;
        self.cv_stack_len += 1;
    }

    fn pop_stack(&mut self) -> [u32; 8] {
        assert!(self.cv_stack_len > 0);
        self.cv_stack_len -= 1;
        self.cv_stack[self.cv_stack_len as usize]
    }

    fn add_chunk_chaining_value(&mut self, mut new_cv: [u32; 8], mut total_chunks: u64) {
        while total_chunks & 1 == 0 {
            let left = self.pop_stack();
            new_cv = parent_cv(&left, &new_cv, &self.key_words, self.flags);
            total_chunks >>= 1;
        }
        self.push_stack(new_cv);
    }

    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            if self.chunk_state.len() == CHUNK_LEN {
                let chunk_cv = self.chunk_state.output().chaining_value();
                let total_chunks = self.chunk_state.chunk_counter + 1;
                self.add_chunk_chaining_value(chunk_cv, total_chunks);
                self.chunk_state = ChunkState::new(self.key_words, total_chunks, self.flags);
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
            if self.chunk_state.len() == 0 && input.len() > 16 * CHUNK_LEN {
                let mut cvs = [[0u32; 8]; 16];
                let start = self.chunk_state.chunk_counter;
                neon::compress_16_chunks(
                    &self.key_words,
                    &input[..16 * CHUNK_LEN],
                    start,
                    self.flags,
                    &mut cvs,
                );
                for (i, cv) in cvs.into_iter().enumerate() {
                    self.add_chunk_chaining_value(cv, start + i as u64 + 1);
                }
                self.chunk_state = ChunkState::new(self.key_words, start + 16, self.flags);
                input = &input[16 * CHUNK_LEN..];
                continue;
            }
            let take = (CHUNK_LEN - self.chunk_state.len()).min(input.len());
            self.chunk_state.update(&input[..take]);
            input = &input[take..];
        }
        self
    }

    fn root_output(&self) -> Output {
        let mut output = self.chunk_state.output();
        let mut parent_nodes_remaining = self.cv_stack_len as usize;
        while parent_nodes_remaining > 0 {
            parent_nodes_remaining -= 1;
            output = parent_output(
                self.cv_stack[parent_nodes_remaining],
                output.chaining_value(),
                self.key_words,
                self.flags,
            );
        }
        output
    }

    pub fn finalize(&self) -> [u8; OUT_LEN] {
        let mut out = [0u8; OUT_LEN];
        self.root_output().root_output_bytes(&mut out);
        out
    }

    pub fn finalize_xof(&self) -> OutputReader {
        OutputReader::new(self.root_output())
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn hash_with_key_words(key: [u32; 8], flags: u32, input: &[u8]) -> [u8; OUT_LEN] {
    if input.len() <= CHUNK_LEN {
        return bytes_from_words(&compress_chunk_scalar(&key, input, 0, flags, ROOT));
    }
    let mut hasher = Hasher::new_internal(key, flags);
    hasher.update(input);
    hasher.finalize()
}

pub(crate) fn keyed_hash_words(key: &[u32; 8], input: &[u8]) -> [u8; OUT_LEN] {
    hash_with_key_words(*key, KEYED_HASH, input)
}

pub(crate) fn keyed_hash_block(key: &[u32; 8], block: &[u8; BLOCK_LEN]) -> [u8; OUT_LEN] {
    const { assert!(BLOCK_LEN == 64 && OUT_LEN == 32) };
    let words = words_from_block(block);
    let out = compress(key, &words, 0, BLOCK_LEN as u32, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);
    bytes_from_words(&first_8_words(out))
}

pub(crate) fn keyed_hash_parent_16x(
    key: &[u32; 8],
    children: &[[u8; 32]; 32],
    out: &mut [[u8; OUT_LEN]; 16],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_parent_16x(key, children, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for (pair, dst) in children.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
            let mut block = [0u8; BLOCK_LEN];
            block[..32].copy_from_slice(&pair[0]);
            block[32..].copy_from_slice(&pair[1]);
            *dst = keyed_hash_block(key, &block);
        }
    }
}

pub(crate) fn keyed_hash_parent_8x(key: &[u32; 8], children: &[[u8; 32]; 16], out: &mut [[u8; OUT_LEN]; 8]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_parent_8x(key, children, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for (pair, dst) in children.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
            let mut block = [0u8; BLOCK_LEN];
            block[..32].copy_from_slice(&pair[0]);
            block[32..].copy_from_slice(&pair[1]);
            *dst = keyed_hash_block(key, &block);
        }
    }
}

pub(crate) fn keyed_hash_parent_4x(key: &[u32; 8], children: &[[u8; 32]; 8], out: &mut [[u8; OUT_LEN]; 4]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_parent_4x(key, children, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for (pair, dst) in children.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
            let mut block = [0u8; BLOCK_LEN];
            block[..32].copy_from_slice(&pair[0]);
            block[32..].copy_from_slice(&pair[1]);
            *dst = keyed_hash_block(key, &block);
        }
    }
}

pub(crate) fn keyed_hash_16x(key: &[u32; 8], inputs: &[&[u8]; 16], out: &mut [[u8; OUT_LEN]; 16]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_16x(key, inputs, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for (dst, input) in out.iter_mut().zip(inputs.iter()) {
            *dst = keyed_hash_words(key, input);
        }
    }
}

pub(crate) fn keyed_hash_8x(key: &[u32; 8], inputs: [&[u8]; 8], out: &mut [[u8; OUT_LEN]; 8]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_8x(key, inputs, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for (dst, input) in out.iter_mut().zip(inputs) {
            *dst = keyed_hash_words(key, input);
        }
    }
}

pub(crate) fn keyed_hash_4x(key: &[u32; 8], inputs: [&[u8]; 4], out: &mut [[u8; OUT_LEN]; 4]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_4x(key, inputs, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for (dst, input) in out.iter_mut().zip(inputs) {
            *dst = keyed_hash_words(key, input);
        }
    }
}

pub(crate) fn hash_challenge_words(key: &[u32; 8], seed: &[u8; 32], counter: u64) -> [u8; OUT_LEN] {
    const { assert!(OUT_LEN == 32) };
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&words_from_key(seed));
    block_words[8] = counter as u32;
    block_words[9] = (counter >> 32) as u32;
    let out = compress(key, &block_words, 0, 40, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);
    bytes_from_words(&first_8_words(out))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
pub(crate) fn hash_rows_16x_direct<F: crate::field::Field>(
    key: &[u32; 8],
    rows: &[Vec<F>],
    start: usize,
    out: &mut [[[u8; OUT_LEN]; 16]],
) {
    neon::hash_rows_16x_direct(key, rows, start, out);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
#[inline(never)]
pub(crate) fn hash_rows_8x_direct<F: crate::field::Field>(
    key: &[u32; 8],
    rows: &[Vec<F>],
    base: usize,
    out: &mut [[u8; OUT_LEN]; 8],
) {
    neon::hash_rows_8x_direct(key, rows, base, out);
}

pub(crate) fn pow_bits_with_key_words(key: &[u32; 8], seed: &[u8; 32], nonce: u64) -> u32 {
    const { assert!(OUT_LEN == 32) };
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&words_from_key(seed));
    block_words[8] = nonce as u32;
    block_words[9] = (nonce >> 32) as u32;
    let out = compress(key, &block_words, 0, 40, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT);
    let hi = (out[0].swap_bytes() as u64) << 32 | (out[1].swap_bytes() as u64);
    let bits = hi.leading_zeros();
    assert!(bits <= 64);
    bits
}

pub(crate) fn find_pow_nonce_with_key_words(
    key: &[u32; 8],
    seed: &[u8; 32],
    bits: u32,
    start: u64,
    count: usize,
) -> Option<u64> {
    assert!(bits <= 64);
    assert!(count == 0 || start <= u64::MAX - (count as u64 - 1));
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::find_pow_nonce_16x(key, seed, bits, start, count)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for offset in 0..count {
            let nonce = start + offset as u64;
            if pow_bits_with_key_words(key, seed, nonce) >= bits {
                return Some(nonce);
            }
        }
        None
    }
}

pub fn hash(input: &[u8]) -> [u8; OUT_LEN] {
    hash_with_key_words(IV, 0, input)
}

pub fn keyed_hash(key: &[u8; KEY_LEN], input: &[u8]) -> [u8; OUT_LEN] {
    hash_with_key_words(words_from_key(key), KEYED_HASH, input)
}

#[derive(Clone)]
pub struct OutputReader {
    output: Output,
    block_counter: u64,
    buffer: [u8; 16 * BLOCK_LEN],
    buffer_position: usize,
    buffer_len: usize,
}

impl OutputReader {
    fn new(output: Output) -> Self {
        assert!(output.flags & ROOT == 0);
        Self { output, block_counter: 0, buffer: [0; 16 * BLOCK_LEN], buffer_position: 0, buffer_len: 0 }
    }

    pub fn position(&self) -> u64 {
        assert!(self.buffer_position <= self.buffer_len);
        assert!(self.buffer_len <= 16 * BLOCK_LEN);
        self.block_counter * BLOCK_LEN as u64 - (self.buffer_len - self.buffer_position) as u64
    }

    fn refill(&mut self) {
        assert!(self.buffer_position == self.buffer_len);
        assert!(self.block_counter < u64::MAX);
        #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
        if self.block_counter <= u64::MAX - 16 {
            neon::root_blocks_16x(
                &self.output.input_chaining_value,
                &self.output.block_words,
                self.block_counter,
                self.output.block_len,
                self.output.flags | ROOT,
                &mut self.buffer,
            );
            self.block_counter += 16;
            self.buffer_position = 0;
            self.buffer_len = 16 * BLOCK_LEN;
            return;
        }
        let words = compress(
            &self.output.input_chaining_value,
            &self.output.block_words,
            self.block_counter,
            self.output.block_len,
            self.output.flags | ROOT,
        );
        for (dst, &word) in self.buffer[..BLOCK_LEN].as_chunks_mut::<4>().0.iter_mut().zip(&words) {
            *dst = word.to_le_bytes();
        }
        self.block_counter += 1;
        self.buffer_position = 0;
        self.buffer_len = BLOCK_LEN;
    }

    pub fn fill(&mut self, mut dst: &mut [u8]) {
        assert!(self.buffer_position <= self.buffer_len);
        if self.buffer_position < self.buffer_len {
            let take = (self.buffer_len - self.buffer_position).min(dst.len());
            dst[..take].copy_from_slice(&self.buffer[self.buffer_position..self.buffer_position + take]);
            self.buffer_position += take;
            dst = &mut dst[take..];
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
        while dst.len() >= 16 * BLOCK_LEN && self.block_counter <= u64::MAX - 16 {
            let (batch, rest) = dst.split_at_mut(16 * BLOCK_LEN);
            neon::root_blocks_16x(
                &self.output.input_chaining_value,
                &self.output.block_words,
                self.block_counter,
                self.output.block_len,
                self.output.flags | ROOT,
                batch,
            );
            self.block_counter += 16;
            dst = rest;
        }
        while !dst.is_empty() {
            if self.buffer_position == self.buffer_len {
                self.refill();
            }
            let take = (self.buffer_len - self.buffer_position).min(dst.len());
            dst[..take].copy_from_slice(&self.buffer[self.buffer_position..self.buffer_position + take]);
            self.buffer_position += take;
            dst = &mut dst[take..];
        }
        assert!(self.buffer_position <= self.buffer_len);
    }
}
