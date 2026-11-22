pub const KEY_LEN: usize = 32;
pub const OUT_LEN: usize = 32;
pub const BLOCK_LEN: usize = 64;
pub const CHUNK_LEN: usize = 1024;

const CV_STACK_LEN_MAX: usize = 54;
const CV_STACK_INLINE_LEN_MAX: usize = 8;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const FLAGS_ALL: u32 = CHUNK_START | CHUNK_END | PARENT | ROOT | KEYED_HASH;

const IV: [u32; 8] =
    [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

const ROUNDS: usize = 7;

const MSG_SCHEDULE: [[usize; 16]; ROUNDS] = message_schedule();

const fn message_schedule() -> [[usize; 16]; ROUNDS] {
    let mut schedule = [[0; 16]; ROUNDS];
    let mut word = 0;
    while word < 16 {
        schedule[0][word] = word;
        word += 1;
    }
    let mut round = 1;
    while round < ROUNDS {
        let mut word = 0;
        while word < 16 {
            schedule[round][word] = schedule[round - 1][MSG_PERMUTATION[word]];
            word += 1;
        }
        round += 1;
    }
    schedule
}

#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    state[a] = state[a].wrapping_add(x).wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(y).wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline(always)]
fn round(state: &mut [u32; 16], message: &[u32; 16], schedule: &[usize; 16]) {
    g(state, 0, 4, 8, 12, message[schedule[0]], message[schedule[1]]);
    g(state, 1, 5, 9, 13, message[schedule[2]], message[schedule[3]]);
    g(state, 2, 6, 10, 14, message[schedule[4]], message[schedule[5]]);
    g(state, 3, 7, 11, 15, message[schedule[6]], message[schedule[7]]);
    g(state, 0, 5, 10, 15, message[schedule[8]], message[schedule[9]]);
    g(state, 1, 6, 11, 12, message[schedule[10]], message[schedule[11]]);
    g(state, 2, 7, 8, 13, message[schedule[12]], message[schedule[13]]);
    g(state, 3, 4, 9, 14, message[schedule[14]], message[schedule[15]]);
}

#[inline(always)]
fn compress(cv: &[u32; 8], message: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
    assert!(block_len as usize <= BLOCK_LEN);
    assert!(flags & !FLAGS_ALL == 0);
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    round(&mut state, message, &MSG_SCHEDULE[0]);
    round(&mut state, message, &MSG_SCHEDULE[1]);
    round(&mut state, message, &MSG_SCHEDULE[2]);
    round(&mut state, message, &MSG_SCHEDULE[3]);
    round(&mut state, message, &MSG_SCHEDULE[4]);
    round(&mut state, message, &MSG_SCHEDULE[5]);
    round(&mut state, message, &MSG_SCHEDULE[6]);
    for word in 0..8 {
        state[word] ^= state[word + 8];
        state[word + 8] ^= cv[word];
    }
    state
}

#[inline(always)]
fn compress_cv(cv: &[u32; 8], message: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 8] {
    let state = compress(cv, message, counter, block_len, flags);
    let mut cv_next = [0; 8];
    cv_next.copy_from_slice(&state[..8]);
    cv_next
}

#[inline(always)]
fn words_from_block(block: &[u8; BLOCK_LEN]) -> [u32; 16] {
    let mut words = [0; 16];
    for (word, bytes) in words.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    words
}

pub(crate) const fn words_from_key(key: &[u8; KEY_LEN]) -> [u32; 8] {
    const { assert!(KEY_LEN == 32) };
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
fn hash_from_cv(cv: [u32; 8]) -> [u8; OUT_LEN] {
    let mut hash = [0; OUT_LEN];
    for (bytes, word) in hash.as_chunks_mut::<4>().0.iter_mut().zip(cv) {
        *bytes = word.to_le_bytes();
    }
    hash
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
mod neon {
    use super::{
        BLOCK_LEN, CHUNK_END, CHUNK_LEN, CHUNK_START, CV_STACK_LEN_MAX, FLAGS_ALL, IV, KEYED_HASH,
        MSG_SCHEDULE, OUT_LEN, Output, PARENT, ROOT,
    };
    use std::arch::aarch64::*;

    type V = uint32x4_t;

    const ROT8_MASK: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];

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
    unsafe fn transpose4(r0: V, r1: V, r2: V, r3: V) -> [V; 4] {
        unsafe {
            let t0 = vtrn1q_u32(r0, r1);
            let t1 = vtrn2q_u32(r0, r1);
            let t2 = vtrn1q_u32(r2, r3);
            let t3 = vtrn2q_u32(r2, r3);
            let c0 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
            let c1 = vreinterpretq_u32_u64(vzip1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
            let c2 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
            let c3 = vreinterpretq_u32_u64(vzip2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
            [c0, c1, c2, c3]
        }
    }

    #[inline(always)]
    unsafe fn g_4x(s: &mut [V; 16], idx: [usize; 4], x: V, y: V, rot8: uint8x16_t) {
        let [a, b, c, d] = idx;
        unsafe {
            s[a] = vaddq_u32(vaddq_u32(s[a], x), s[b]);
            s[d] = rotr16(veorq_u32(s[d], s[a]));
            s[c] = vaddq_u32(s[c], s[d]);
            s[b] = rotr12(veorq_u32(s[b], s[c]));
            s[a] = vaddq_u32(vaddq_u32(s[a], y), s[b]);
            s[d] = rotr8(veorq_u32(s[d], s[a]), rot8);
            s[c] = vaddq_u32(s[c], s[d]);
            s[b] = rotr7(veorq_u32(s[b], s[c]));
        }
    }

    #[inline(always)]
    unsafe fn round_4x(s: &mut [V; 16], m: &[V; 16], sched: &[usize; 16], rot8: uint8x16_t) {
        unsafe {
            g_4x(s, [0, 4, 8, 12], m[sched[0]], m[sched[1]], rot8);
            g_4x(s, [1, 5, 9, 13], m[sched[2]], m[sched[3]], rot8);
            g_4x(s, [2, 6, 10, 14], m[sched[4]], m[sched[5]], rot8);
            g_4x(s, [3, 7, 11, 15], m[sched[6]], m[sched[7]], rot8);
            g_4x(s, [0, 5, 10, 15], m[sched[8]], m[sched[9]], rot8);
            g_4x(s, [1, 6, 11, 12], m[sched[10]], m[sched[11]], rot8);
            g_4x(s, [2, 7, 8, 13], m[sched[12]], m[sched[13]], rot8);
            g_4x(s, [3, 4, 9, 14], m[sched[14]], m[sched[15]], rot8);
        }
    }

    #[inline(always)]
    unsafe fn compress_core_4x(
        cv: &[V; 8],
        m: &[V; 16],
        counter_lo: V,
        counter_hi: V,
        block_len: u32,
        flags: u32,
    ) -> [V; 16] {
        assert!(block_len as usize <= BLOCK_LEN);
        assert!(flags & !FLAGS_ALL == 0);
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let mut s = [
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
                counter_lo,
                counter_hi,
                vdupq_n_u32(block_len),
                vdupq_n_u32(flags),
            ];
            round_4x(&mut s, m, &MSG_SCHEDULE[0], rot8);
            round_4x(&mut s, m, &MSG_SCHEDULE[1], rot8);
            round_4x(&mut s, m, &MSG_SCHEDULE[2], rot8);
            round_4x(&mut s, m, &MSG_SCHEDULE[3], rot8);
            round_4x(&mut s, m, &MSG_SCHEDULE[4], rot8);
            round_4x(&mut s, m, &MSG_SCHEDULE[5], rot8);
            round_4x(&mut s, m, &MSG_SCHEDULE[6], rot8);
            s
        }
    }

    #[inline(always)]
    unsafe fn compress_cv_4x(
        cv: &[V; 8],
        m: &[V; 16],
        counter_lo: V,
        counter_hi: V,
        block_len: u32,
        flags: u32,
    ) -> [V; 8] {
        unsafe {
            let s = compress_core_4x(cv, m, counter_lo, counter_hi, block_len, flags);
            [
                veorq_u32(s[0], s[8]),
                veorq_u32(s[1], s[9]),
                veorq_u32(s[2], s[10]),
                veorq_u32(s[3], s[11]),
                veorq_u32(s[4], s[12]),
                veorq_u32(s[5], s[13]),
                veorq_u32(s[6], s[14]),
                veorq_u32(s[7], s[15]),
            ]
        }
    }

    #[inline(always)]
    unsafe fn load_full_block_4x(ptrs: [*const u8; 4]) -> [V; 16] {
        unsafe {
            let mut m = [vdupq_n_u32(0); 16];
            for group in 0..4 {
                let off = group * 16;
                let r0 = vreinterpretq_u32_u8(vld1q_u8(ptrs[0].add(off)));
                let r1 = vreinterpretq_u32_u8(vld1q_u8(ptrs[1].add(off)));
                let r2 = vreinterpretq_u32_u8(vld1q_u8(ptrs[2].add(off)));
                let r3 = vreinterpretq_u32_u8(vld1q_u8(ptrs[3].add(off)));
                let cols = transpose4(r0, r1, r2, r3);
                m[group * 4] = cols[0];
                m[group * 4 + 1] = cols[1];
                m[group * 4 + 2] = cols[2];
                m[group * 4 + 3] = cols[3];
            }
            m
        }
    }

    #[inline(always)]
    unsafe fn load_partial_block_4x(inputs: [&[u8]; 4], offset: usize, len_bytes: usize) -> [V; 16] {
        assert!(len_bytes <= BLOCK_LEN);
        assert!(inputs.iter().all(|slice| offset + len_bytes <= slice.len()));
        let mut padded = [[0u8; BLOCK_LEN]; 4];
        for lane in 0..4 {
            padded[lane][..len_bytes].copy_from_slice(&inputs[lane][offset..offset + len_bytes]);
        }
        unsafe {
            load_full_block_4x([
                padded[0].as_ptr(),
                padded[1].as_ptr(),
                padded[2].as_ptr(),
                padded[3].as_ptr(),
            ])
        }
    }

    #[inline(always)]
    unsafe fn store_cv_4x(cv: &[V; 8], out: &mut [[u8; OUT_LEN]; 4]) {
        unsafe {
            let lo = transpose4(cv[0], cv[1], cv[2], cv[3]);
            let hi = transpose4(cv[4], cv[5], cv[6], cv[7]);
            for lane in 0..4 {
                vst1q_u8(out[lane].as_mut_ptr(), vreinterpretq_u8_u32(lo[lane]));
                vst1q_u8(out[lane].as_mut_ptr().add(16), vreinterpretq_u8_u32(hi[lane]));
            }
        }
    }

    #[inline(always)]
    unsafe fn key_splat(key: &[u32; 8]) -> [V; 8] {
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
    unsafe fn parent_cv_4x(key_v: &[V; 8], left: &[V; 8], right: &[V; 8], flags: u32) -> [V; 8] {
        unsafe {
            let m = [
                left[0], left[1], left[2], left[3], left[4], left[5], left[6], left[7], right[0], right[1],
                right[2], right[3], right[4], right[5], right[6], right[7],
            ];
            let zero = vdupq_n_u32(0);
            compress_cv_4x(key_v, &m, zero, zero, BLOCK_LEN as u32, flags | PARENT)
        }
    }

    #[inline(always)]
    unsafe fn compress_chunk_4x(
        key_v: &[V; 8],
        inputs: [&[u8]; 4],
        base_offset: usize,
        chunk_len: usize,
        chunk_counter: u64,
        base_flags: u32,
        root_flag: u32,
    ) -> [V; 8] {
        assert!(chunk_len <= CHUNK_LEN);
        assert!(inputs.iter().all(|slice| base_offset + chunk_len <= slice.len()));
        unsafe {
            let counter_lo = vdupq_n_u32(chunk_counter as u32);
            let counter_hi = vdupq_n_u32((chunk_counter >> 32) as u32);
            let mut cv = *key_v;
            if chunk_len == 0 {
                let m = [vdupq_n_u32(0); 16];
                let flags = base_flags | CHUNK_START | CHUNK_END | root_flag;
                return compress_cv_4x(&cv, &m, counter_lo, counter_hi, 0, flags);
            }
            let full_blocks = (chunk_len - 1) / BLOCK_LEN;
            for block in 0..full_blocks {
                let off = base_offset + block * BLOCK_LEN;
                let ptrs = [
                    inputs[0].as_ptr().add(off),
                    inputs[1].as_ptr().add(off),
                    inputs[2].as_ptr().add(off),
                    inputs[3].as_ptr().add(off),
                ];
                let m = load_full_block_4x(ptrs);
                let start = if block == 0 { CHUNK_START } else { 0 };
                cv = compress_cv_4x(&cv, &m, counter_lo, counter_hi, BLOCK_LEN as u32, base_flags | start);
            }
            let last_off = base_offset + full_blocks * BLOCK_LEN;
            let last_len = chunk_len - full_blocks * BLOCK_LEN;
            let start = if full_blocks == 0 { CHUNK_START } else { 0 };
            let flags = base_flags | start | CHUNK_END | root_flag;
            if last_len == 8 {
                let mut m = [vdupq_n_u32(0); 16];
                let mut w0 = [0u32; 4];
                let mut w1 = [0u32; 4];
                for lane in 0..4 {
                    let s = &inputs[lane][last_off..last_off + 8];
                    w0[lane] = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                    w1[lane] = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
                }
                m[0] = vld1q_u32(w0.as_ptr());
                m[1] = vld1q_u32(w1.as_ptr());
                return compress_cv_4x(&cv, &m, counter_lo, counter_hi, 8, flags);
            }
            let m = if last_len == BLOCK_LEN {
                load_full_block_4x([
                    inputs[0].as_ptr().add(last_off),
                    inputs[1].as_ptr().add(last_off),
                    inputs[2].as_ptr().add(last_off),
                    inputs[3].as_ptr().add(last_off),
                ])
            } else {
                load_partial_block_4x(inputs, last_off, last_len)
            };
            compress_cv_4x(&cv, &m, counter_lo, counter_hi, last_len as u32, flags)
        }
    }

    pub fn keyed_hash_parent_4x(key: &[u32; 8], children: &[[u8; 32]; 8], out: &mut [[u8; OUT_LEN]; 4]) {
        const { assert!(OUT_LEN == 32 && BLOCK_LEN == 64) };
        unsafe {
            let key_v = key_splat(key);
            let base = children.as_ptr().cast::<u8>();
            let m = load_full_block_4x([base, base.add(64), base.add(128), base.add(192)]);
            let zero = vdupq_n_u32(0);
            let flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
            let cv = compress_cv_4x(&key_v, &m, zero, zero, BLOCK_LEN as u32, flags);
            store_cv_4x(&cv, out);
        }
    }

    #[cold]
    #[inline(never)]
    fn keyed_hash_4x_multi_chunk(key: &[u32; 8], inputs: [&[u8]; 4], out: &mut [[u8; OUT_LEN]; 4]) {
        let len_bytes = inputs[0].len();
        assert!(len_bytes > CHUNK_LEN);
        assert!(inputs.iter().all(|slice| slice.len() == len_bytes));
        unsafe {
            let key_v = key_splat(key);
            let chunks_total = len_bytes.div_ceil(CHUNK_LEN);
            assert!(chunks_total >= 2);
            let mut stack = [[vdupq_n_u32(0); 8]; CV_STACK_LEN_MAX];
            let mut stack_len = 0usize;
            for chunk_index in 0..chunks_total - 1 {
                let off = chunk_index * CHUNK_LEN;
                let mut cv =
                    compress_chunk_4x(&key_v, inputs, off, CHUNK_LEN, chunk_index as u64, KEYED_HASH, 0);
                let mut total = (chunk_index + 1) as u64;
                while total & 1 == 0 {
                    stack_len -= 1;
                    cv = parent_cv_4x(&key_v, &stack[stack_len], &cv, KEYED_HASH);
                    total >>= 1;
                }
                assert!(stack_len < CV_STACK_LEN_MAX);
                stack[stack_len] = cv;
                stack_len += 1;
            }
            let last_index = chunks_total - 1;
            let last_off = last_index * CHUNK_LEN;
            let last_len = len_bytes - last_off;
            let mut cv =
                compress_chunk_4x(&key_v, inputs, last_off, last_len, last_index as u64, KEYED_HASH, 0);
            while stack_len > 1 {
                stack_len -= 1;
                cv = parent_cv_4x(&key_v, &stack[stack_len], &cv, KEYED_HASH);
            }
            assert!(stack_len == 1);
            cv = parent_cv_4x(&key_v, &stack[0], &cv, KEYED_HASH | ROOT);
            store_cv_4x(&cv, out);
        }
    }

    pub fn keyed_hash_4x(key: &[u32; 8], inputs: [&[u8]; 4], out: &mut [[u8; OUT_LEN]; 4]) {
        let len_bytes = inputs[0].len();
        assert!(inputs.iter().all(|slice| slice.len() == len_bytes));
        assert!(len_bytes <= usize::MAX - CHUNK_LEN);
        if len_bytes > CHUNK_LEN {
            keyed_hash_4x_multi_chunk(key, inputs, out);
            return;
        }
        unsafe {
            let key_v = key_splat(key);
            let cv = compress_chunk_4x(&key_v, inputs, 0, len_bytes, 0, KEYED_HASH, ROOT);
            store_cv_4x(&cv, out);
        }
    }

    #[inline(always)]
    unsafe fn counter_vecs(first: u64) -> (V, V) {
        let lo = [first as u32, (first + 1) as u32, (first + 2) as u32, (first + 3) as u32];
        let hi = [
            (first >> 32) as u32,
            ((first + 1) >> 32) as u32,
            ((first + 2) >> 32) as u32,
            ((first + 3) >> 32) as u32,
        ];
        unsafe { (vld1q_u32(lo.as_ptr()), vld1q_u32(hi.as_ptr())) }
    }

    #[inline(always)]
    unsafe fn load_block_at(inputs: [&[u8]; 4], off: usize) -> [V; 16] {
        unsafe {
            load_full_block_4x([
                inputs[0].as_ptr().add(off),
                inputs[1].as_ptr().add(off),
                inputs[2].as_ptr().add(off),
                inputs[3].as_ptr().add(off),
            ])
        }
    }

    #[inline(always)]
    unsafe fn unpack_cv_4x(cv: &[V; 8], out: &mut [[u32; 8]; 4]) {
        unsafe {
            let lo_words = transpose4(cv[0], cv[1], cv[2], cv[3]);
            let hi_words = transpose4(cv[4], cv[5], cv[6], cv[7]);
            for lane in 0..4 {
                vst1q_u32(out[lane].as_mut_ptr(), lo_words[lane]);
                vst1q_u32(out[lane].as_mut_ptr().add(4), hi_words[lane]);
            }
        }
    }

    pub fn compress_4_chunks(
        key: &[u32; 8],
        input: &[u8; 4 * CHUNK_LEN],
        first_counter: u64,
        flags: u32,
    ) -> [[u32; 8]; 4] {
        assert!(flags & !KEYED_HASH == 0);
        assert!(first_counter <= u64::MAX - 4);
        unsafe {
            let mut cv = key_splat(key);
            let (counter_lo, counter_hi) = counter_vecs(first_counter);
            let base = input.as_ptr();
            for block in 0..(CHUNK_LEN / BLOCK_LEN) {
                let off = block * BLOCK_LEN;
                let ptrs = [
                    base.add(off),
                    base.add(CHUNK_LEN + off),
                    base.add(2 * CHUNK_LEN + off),
                    base.add(3 * CHUNK_LEN + off),
                ];
                let m = load_full_block_4x(ptrs);
                let edge = if block == 0 {
                    CHUNK_START
                } else if block + 1 == CHUNK_LEN / BLOCK_LEN {
                    CHUNK_END
                } else {
                    0
                };
                cv = compress_cv_4x(&cv, &m, counter_lo, counter_hi, BLOCK_LEN as u32, flags | edge);
            }
            let mut out = [[0u32; 8]; 4];
            unpack_cv_4x(&cv, &mut out);
            out
        }
    }

    pub fn compress_8_chunks(
        key: &[u32; 8],
        input: &[u8; 8 * CHUNK_LEN],
        first_counter: u64,
        flags: u32,
    ) -> [[u32; 8]; 8] {
        assert!(flags & !KEYED_HASH == 0);
        assert!(first_counter <= u64::MAX - 8);
        unsafe {
            let mut cv0 = key_splat(key);
            let mut cv1 = cv0;
            let (lo0, hi0) = counter_vecs(first_counter);
            let (lo1, hi1) = counter_vecs(first_counter + 4);
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let base = input.as_ptr();
            for block in 0..(CHUNK_LEN / BLOCK_LEN) {
                let off = block * BLOCK_LEN;
                let m0 = load_full_block_4x([
                    base.add(off),
                    base.add(CHUNK_LEN + off),
                    base.add(2 * CHUNK_LEN + off),
                    base.add(3 * CHUNK_LEN + off),
                ]);
                let m1 = load_full_block_4x([
                    base.add(4 * CHUNK_LEN + off),
                    base.add(5 * CHUNK_LEN + off),
                    base.add(6 * CHUNK_LEN + off),
                    base.add(7 * CHUNK_LEN + off),
                ]);
                let edge = if block == 0 {
                    CHUNK_START
                } else if block + 1 == CHUNK_LEN / BLOCK_LEN {
                    CHUNK_END
                } else {
                    0
                };
                let mut s0 = init_state_4x(&cv0, BLOCK_LEN as u32, flags | edge);
                let mut s1 = init_state_4x(&cv1, BLOCK_LEN as u32, flags | edge);
                s0[12] = lo0;
                s0[13] = hi0;
                s1[12] = lo1;
                s1[13] = hi1;
                round_8x(&mut s0, &mut s1, &m0, &m1, &MSG_SCHEDULE[0], rot8);
                round_8x(&mut s0, &mut s1, &m0, &m1, &MSG_SCHEDULE[1], rot8);
                round_8x(&mut s0, &mut s1, &m0, &m1, &MSG_SCHEDULE[2], rot8);
                round_8x(&mut s0, &mut s1, &m0, &m1, &MSG_SCHEDULE[3], rot8);
                round_8x(&mut s0, &mut s1, &m0, &m1, &MSG_SCHEDULE[4], rot8);
                round_8x(&mut s0, &mut s1, &m0, &m1, &MSG_SCHEDULE[5], rot8);
                round_8x(&mut s0, &mut s1, &m0, &m1, &MSG_SCHEDULE[6], rot8);
                cv0 = finish_cv(&s0);
                cv1 = finish_cv(&s1);
            }
            let mut out = [[0u32; 8]; 8];
            let (out0, out1) = out.split_at_mut(4);
            unpack_cv_4x(&cv0, out0.try_into().expect("four cvs"));
            unpack_cv_4x(&cv1, out1.try_into().expect("four cvs"));
            out
        }
    }

    #[inline(always)]
    fn pow_check_4(w0: V, w1: V, base_nonce: u64, bits: u32) -> Option<u64> {
        let mut a0 = [0u32; 4];
        let mut a1 = [0u32; 4];
        unsafe {
            vst1q_u32(a0.as_mut_ptr(), w0);
            vst1q_u32(a1.as_mut_ptr(), w1);
        }
        for lane in 0..4 {
            let head = ((a0[lane].swap_bytes() as u64) << 32) | (a1[lane].swap_bytes() as u64);
            if head.leading_zeros() >= bits {
                return Some(base_nonce + lane as u64);
            }
        }
        None
    }

    #[inline(always)]
    unsafe fn g_8x(
        s0: &mut [V; 16],
        s1: &mut [V; 16],
        idx: [usize; 4],
        m0: [V; 2],
        m1: [V; 2],
        rot8: uint8x16_t,
    ) {
        let [a, b, c, d] = idx;
        unsafe {
            s0[a] = vaddq_u32(vaddq_u32(s0[a], m0[0]), s0[b]);
            s1[a] = vaddq_u32(vaddq_u32(s1[a], m1[0]), s1[b]);
            s0[d] = rotr16(veorq_u32(s0[d], s0[a]));
            s1[d] = rotr16(veorq_u32(s1[d], s1[a]));
            s0[c] = vaddq_u32(s0[c], s0[d]);
            s1[c] = vaddq_u32(s1[c], s1[d]);
            s0[b] = rotr12(veorq_u32(s0[b], s0[c]));
            s1[b] = rotr12(veorq_u32(s1[b], s1[c]));
            s0[a] = vaddq_u32(vaddq_u32(s0[a], m0[1]), s0[b]);
            s1[a] = vaddq_u32(vaddq_u32(s1[a], m1[1]), s1[b]);
            s0[d] = rotr8(veorq_u32(s0[d], s0[a]), rot8);
            s1[d] = rotr8(veorq_u32(s1[d], s1[a]), rot8);
            s0[c] = vaddq_u32(s0[c], s0[d]);
            s1[c] = vaddq_u32(s1[c], s1[d]);
            s0[b] = rotr7(veorq_u32(s0[b], s0[c]));
            s1[b] = rotr7(veorq_u32(s1[b], s1[c]));
        }
    }

    #[inline(always)]
    unsafe fn round_8x(
        s0: &mut [V; 16],
        s1: &mut [V; 16],
        m0: &[V; 16],
        m1: &[V; 16],
        sched: &[usize; 16],
        rot8: uint8x16_t,
    ) {
        unsafe {
            g_8x(s0, s1, [0, 4, 8, 12], [m0[sched[0]], m0[sched[1]]], [m1[sched[0]], m1[sched[1]]], rot8);
            g_8x(s0, s1, [1, 5, 9, 13], [m0[sched[2]], m0[sched[3]]], [m1[sched[2]], m1[sched[3]]], rot8);
            g_8x(s0, s1, [2, 6, 10, 14], [m0[sched[4]], m0[sched[5]]], [m1[sched[4]], m1[sched[5]]], rot8);
            g_8x(s0, s1, [3, 7, 11, 15], [m0[sched[6]], m0[sched[7]]], [m1[sched[6]], m1[sched[7]]], rot8);
            g_8x(s0, s1, [0, 5, 10, 15], [m0[sched[8]], m0[sched[9]]], [m1[sched[8]], m1[sched[9]]], rot8);
            g_8x(
                s0,
                s1,
                [1, 6, 11, 12],
                [m0[sched[10]], m0[sched[11]]],
                [m1[sched[10]], m1[sched[11]]],
                rot8,
            );
            g_8x(s0, s1, [2, 7, 8, 13], [m0[sched[12]], m0[sched[13]]], [m1[sched[12]], m1[sched[13]]], rot8);
            g_8x(s0, s1, [3, 4, 9, 14], [m0[sched[14]], m0[sched[15]]], [m1[sched[14]], m1[sched[15]]], rot8);
        }
    }

    #[inline(always)]
    unsafe fn init_state_4x(cv: &[V; 8], block_len: u32, flags: u32) -> [V; 16] {
        unsafe {
            let zero = vdupq_n_u32(0);
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
                zero,
                zero,
                vdupq_n_u32(block_len),
                vdupq_n_u32(flags),
            ]
        }
    }

    #[inline(always)]
    unsafe fn finish_cv(s: &[V; 16]) -> [V; 8] {
        unsafe {
            [
                veorq_u32(s[0], s[8]),
                veorq_u32(s[1], s[9]),
                veorq_u32(s[2], s[10]),
                veorq_u32(s[3], s[11]),
                veorq_u32(s[4], s[12]),
                veorq_u32(s[5], s[13]),
                veorq_u32(s[6], s[14]),
                veorq_u32(s[7], s[15]),
            ]
        }
    }

    #[inline(always)]
    unsafe fn compress_cv_8x(
        cv0: &[V; 8],
        cv1: &[V; 8],
        m0: &[V; 16],
        m1: &[V; 16],
        block_len: u32,
        flags: u32,
    ) -> ([V; 8], [V; 8]) {
        assert!(block_len as usize <= BLOCK_LEN);
        assert!(flags & !FLAGS_ALL == 0);
        unsafe {
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let mut s0 = init_state_4x(cv0, block_len, flags);
            let mut s1 = init_state_4x(cv1, block_len, flags);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[0], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[1], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[2], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[3], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[4], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[5], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[6], rot8);
            (finish_cv(&s0), finish_cv(&s1))
        }
    }

    pub fn keyed_hash_parent_8x(key: &[u32; 8], children: &[[u8; 32]; 16], out: &mut [[u8; OUT_LEN]; 8]) {
        const { assert!(OUT_LEN == 32 && BLOCK_LEN == 64) };
        unsafe {
            let key_v = key_splat(key);
            let base = children.as_ptr().cast::<u8>();
            let m0 = load_full_block_4x([base, base.add(64), base.add(128), base.add(192)]);
            let m1 = load_full_block_4x([base.add(256), base.add(320), base.add(384), base.add(448)]);
            let flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
            let (cv0, cv1) = compress_cv_8x(&key_v, &key_v, &m0, &m1, BLOCK_LEN as u32, flags);
            let (out0, out1) = out.split_at_mut(4);
            store_cv_4x(&cv0, out0.try_into().expect("four digests"));
            store_cv_4x(&cv1, out1.try_into().expect("four digests"));
        }
    }

    #[inline(always)]
    unsafe fn load_8byte_block_4x(inputs: [&[u8]; 4], offset: usize) -> [V; 16] {
        assert!(inputs.iter().all(|slice| offset + 8 <= slice.len()));
        let mut w0 = [0u32; 4];
        let mut w1 = [0u32; 4];
        for lane in 0..4 {
            let s = &inputs[lane][offset..offset + 8];
            w0[lane] = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            w1[lane] = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
        }
        unsafe {
            let mut m = [vdupq_n_u32(0); 16];
            m[0] = vld1q_u32(w0.as_ptr());
            m[1] = vld1q_u32(w1.as_ptr());
            m
        }
    }

    pub fn keyed_hash_8x(key: &[u32; 8], inputs: [&[u8]; 8], out: &mut [[u8; OUT_LEN]; 8]) {
        let len_bytes = inputs[0].len();
        assert!(inputs.iter().all(|slice| slice.len() == len_bytes));
        assert!(len_bytes <= usize::MAX - CHUNK_LEN);
        let in0 = [inputs[0], inputs[1], inputs[2], inputs[3]];
        let in1 = [inputs[4], inputs[5], inputs[6], inputs[7]];
        if len_bytes == 0 || len_bytes > CHUNK_LEN {
            let (out0, out1) = out.split_at_mut(4);
            keyed_hash_4x(key, in0, out0.try_into().expect("four digests"));
            keyed_hash_4x(key, in1, out1.try_into().expect("four digests"));
            return;
        }
        unsafe {
            let mut cv0 = key_splat(key);
            let mut cv1 = cv0;
            let full_blocks = (len_bytes - 1) / BLOCK_LEN;
            for block in 0..full_blocks {
                let off = block * BLOCK_LEN;
                let m0 = load_block_at(in0, off);
                let m1 = load_block_at(in1, off);
                let start = if block == 0 { CHUNK_START } else { 0 };
                (cv0, cv1) = compress_cv_8x(&cv0, &cv1, &m0, &m1, BLOCK_LEN as u32, KEYED_HASH | start);
            }
            let last_off = full_blocks * BLOCK_LEN;
            let last_len = len_bytes - last_off;
            let start = if full_blocks == 0 { CHUNK_START } else { 0 };
            let flags = KEYED_HASH | start | CHUNK_END | ROOT;
            let (cv0, cv1) = if last_len == 8 {
                let m0 = load_8byte_block_4x(in0, last_off);
                let m1 = load_8byte_block_4x(in1, last_off);
                compress_cv_8x(&cv0, &cv1, &m0, &m1, 8, flags)
            } else if last_len == BLOCK_LEN {
                let m0 = load_block_at(in0, last_off);
                let m1 = load_block_at(in1, last_off);
                compress_cv_8x(&cv0, &cv1, &m0, &m1, BLOCK_LEN as u32, flags)
            } else {
                let m0 = load_partial_block_4x(in0, last_off, last_len);
                let m1 = load_partial_block_4x(in1, last_off, last_len);
                compress_cv_8x(&cv0, &cv1, &m0, &m1, last_len as u32, flags)
            };
            let (out0, out1) = out.split_at_mut(4);
            store_cv_4x(&cv0, out0.try_into().expect("four digests"));
            store_cv_4x(&cv1, out1.try_into().expect("four digests"));
        }
    }

    #[inline(always)]
    unsafe fn store_root_block_4x(mut s: [V; 16], cv: &[V; 8], dst: *mut u8) {
        unsafe {
            for word in 0..8 {
                s[word] = veorq_u32(s[word], s[word + 8]);
                s[word + 8] = veorq_u32(s[word + 8], cv[word]);
            }
            for group in 0..4 {
                let cols = transpose4(s[group * 4], s[group * 4 + 1], s[group * 4 + 2], s[group * 4 + 3]);
                for (lane, &col) in cols.iter().enumerate() {
                    vst1q_u8(dst.add(lane * BLOCK_LEN + group * 16), vreinterpretq_u8_u32(col));
                }
            }
        }
    }

    pub fn root_blocks_8x(output: &Output, first_counter: u64, dst: &mut [u8; 8 * BLOCK_LEN]) {
        assert!(output.flags & ROOT == 0);
        assert!(first_counter <= u64::MAX - 7);
        unsafe {
            let cv = key_splat(&output.cv_input);
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let mut m = [vdupq_n_u32(0); 16];
            for word in 0..16 {
                m[word] = vdupq_n_u32(output.message[word]);
            }
            let (lo0, hi0) = counter_vecs(first_counter);
            let (lo1, hi1) = counter_vecs(first_counter + 4);
            let mut s0 = init_state_4x(&cv, output.block_len, output.flags | ROOT);
            let mut s1 = s0;
            s0[12] = lo0;
            s0[13] = hi0;
            s1[12] = lo1;
            s1[13] = hi1;
            round_8x(&mut s0, &mut s1, &m, &m, &MSG_SCHEDULE[0], rot8);
            round_8x(&mut s0, &mut s1, &m, &m, &MSG_SCHEDULE[1], rot8);
            round_8x(&mut s0, &mut s1, &m, &m, &MSG_SCHEDULE[2], rot8);
            round_8x(&mut s0, &mut s1, &m, &m, &MSG_SCHEDULE[3], rot8);
            round_8x(&mut s0, &mut s1, &m, &m, &MSG_SCHEDULE[4], rot8);
            round_8x(&mut s0, &mut s1, &m, &m, &MSG_SCHEDULE[5], rot8);
            round_8x(&mut s0, &mut s1, &m, &m, &MSG_SCHEDULE[6], rot8);
            store_root_block_4x(s0, &cv, dst.as_mut_ptr());
            store_root_block_4x(s1, &cv, dst.as_mut_ptr().add(4 * BLOCK_LEN));
        }
    }

    #[inline(always)]
    unsafe fn compress_pow_8x(
        init: &[V; 16],
        m0: &[V; 16],
        m1: &[V; 16],
        rot8: uint8x16_t,
    ) -> ([V; 16], [V; 16]) {
        let mut s0 = *init;
        let mut s1 = *init;
        unsafe {
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[0], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[1], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[2], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[3], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[4], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[5], rot8);
            round_8x(&mut s0, &mut s1, m0, m1, &MSG_SCHEDULE[6], rot8);
        }
        (s0, s1)
    }

    pub fn find_pow_nonce_4x(
        key: &[u32; 8],
        seed: &[u8; 32],
        bits: u32,
        start: u64,
        count: usize,
    ) -> Option<u64> {
        assert!(bits <= 64);
        assert!(count == 0 || start <= u64::MAX - (count as u64 - 1));
        if bits == 0 {
            return (count > 0).then_some(start);
        }
        let seed_words = super::words_from_key(seed);
        let hi_mask = if bits >= 32 { u32::MAX } else { u32::MAX << (32 - bits) };
        unsafe {
            let cv = key_splat(key);
            let rot8 = vld1q_u8(ROT8_MASK.as_ptr());
            let zero = vdupq_n_u32(0);
            let mut m0 = [zero; 16];
            for word in 0..8 {
                m0[word] = vdupq_n_u32(seed_words[word]);
            }
            let mut m1 = m0;
            let hi_mask_v = vdupq_n_u32(hi_mask);
            let flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
            let init = init_state_4x(&cv, 40, flags);
            let groups = count / 8;
            for group in 0..groups {
                let n0 = start + (group as u64) * 8;
                let n1 = n0 + 4;
                (m0[8], m0[9]) = counter_vecs(n0);
                (m1[8], m1[9]) = counter_vecs(n1);
                let (s0, s1) = compress_pow_8x(&init, &m0, &m1, rot8);
                let w0 = veorq_u32(s0[0], s0[8]);
                let w0_be = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(w0)));
                let w0b = veorq_u32(s1[0], s1[8]);
                let w0b_be = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(w0b)));
                let hit0 = vandq_u32(w0_be, hi_mask_v);
                let hit1 = vandq_u32(w0b_be, hi_mask_v);
                if vminvq_u32(vminq_u32(hit0, hit1)) == 0 {
                    if let Some(found) = pow_check_4(w0, veorq_u32(s0[1], s0[9]), n0, bits) {
                        return Some(found);
                    }
                    if let Some(found) = pow_check_4(w0b, veorq_u32(s1[1], s1[9]), n1, bits) {
                        return Some(found);
                    }
                }
            }
        }
        let done = (count / 8) * 8;
        for offset in done..count {
            let nonce = start + offset as u64;
            if super::pow_bits_with_key_words(key, seed, nonce) >= bits {
                return Some(nonce);
            }
        }
        None
    }
}

#[derive(Clone, Copy)]
struct Output {
    cv_input: [u32; 8],
    message: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        assert!(self.flags & ROOT == 0);
        compress_cv(&self.cv_input, &self.message, self.counter, self.block_len, self.flags)
    }

    fn root_block(&self, output_block_counter: u64) -> [u8; BLOCK_LEN] {
        assert!(self.flags & ROOT == 0);
        let state =
            compress(&self.cv_input, &self.message, output_block_counter, self.block_len, self.flags | ROOT);
        let mut block = [0; BLOCK_LEN];
        for (bytes, word) in block.as_chunks_mut::<4>().0.iter_mut().zip(state) {
            *bytes = word.to_le_bytes();
        }
        block
    }

    fn root_hash(&self) -> [u8; OUT_LEN] {
        assert!(self.flags & ROOT == 0);
        let cv = compress_cv(&self.cv_input, &self.message, 0, self.block_len, self.flags | ROOT);
        hash_from_cv(cv)
    }
}

fn parent_output(left: &[u32; 8], right: &[u32; 8], key: &[u32; 8], flags: u32) -> Output {
    assert!(flags & !KEYED_HASH == 0);
    let mut message = [0; 16];
    message[..8].copy_from_slice(left);
    message[8..].copy_from_slice(right);
    Output { cv_input: *key, message, counter: 0, block_len: BLOCK_LEN as u32, flags: flags | PARENT }
}

#[derive(Clone)]
struct ChunkState {
    cv: [u32; 8],
    chunk_counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: usize,
    blocks_compressed: usize,
    flags: u32,
}

impl ChunkState {
    fn new(key: &[u32; 8], chunk_counter: u64, flags: u32) -> Self {
        assert!(flags & !KEYED_HASH == 0);
        Self { cv: *key, chunk_counter, block: [0; BLOCK_LEN], block_len: 0, blocks_compressed: 0, flags }
    }

    fn len(&self) -> usize {
        let len = self.blocks_compressed * BLOCK_LEN + self.block_len;
        assert!(len <= CHUNK_LEN);
        len
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
    }

    #[inline(always)]
    fn compress_block(&mut self, block: &[u8; BLOCK_LEN]) {
        assert!(self.blocks_compressed < CHUNK_LEN / BLOCK_LEN - 1);
        let message = words_from_block(block);
        let flags = self.flags | self.start_flag();
        self.cv = compress_cv(&self.cv, &message, self.chunk_counter, BLOCK_LEN as u32, flags);
        self.blocks_compressed += 1;
    }

    fn update(&mut self, mut input: &[u8]) {
        assert!(self.len() + input.len() <= CHUNK_LEN);
        if self.block_len > 0 {
            let take = (BLOCK_LEN - self.block_len).min(input.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&input[..take]);
            self.block_len += take;
            input = &input[take..];
            if input.is_empty() {
                return;
            }
            let block = self.block;
            self.compress_block(&block);
            self.block_len = 0;
        }
        while input.len() > BLOCK_LEN {
            let (block, rest) = input.split_first_chunk::<BLOCK_LEN>().expect("a block remains");
            self.compress_block(block);
            input = rest;
        }
        if !input.is_empty() {
            assert!(self.block_len == 0);
            self.block = [0; BLOCK_LEN];
            self.block[..input.len()].copy_from_slice(input);
            self.block_len = input.len();
        }
    }

    fn output(&self) -> Output {
        assert!(self.block_len <= BLOCK_LEN);
        Output {
            cv_input: self.cv,
            message: words_from_block(&self.block),
            counter: self.chunk_counter,
            block_len: self.block_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

#[derive(Clone)]
pub struct Hasher {
    key: [u32; 8],
    flags: u32,
    chunk: ChunkState,
    cv_stack_inline: [[u32; 8]; CV_STACK_INLINE_LEN_MAX],
    cv_stack_spill: Vec<[u32; 8]>,
    cv_stack_len: usize,
}

impl Hasher {
    pub(crate) fn with_key_words(key: [u32; 8], flags: u32) -> Self {
        assert!(flags & !KEYED_HASH == 0);
        let chunk = ChunkState::new(&key, 0, flags);
        let cv_stack_inline = [[0; 8]; CV_STACK_INLINE_LEN_MAX];
        Self { key, flags, chunk, cv_stack_inline, cv_stack_spill: Vec::new(), cv_stack_len: 0 }
    }

    pub fn new() -> Self {
        Self::with_key_words(IV, 0)
    }

    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        Self::with_key_words(words_from_key(key), KEYED_HASH)
    }

    pub(crate) fn new_keyed_words(key: [u32; 8]) -> Self {
        Self::with_key_words(key, KEYED_HASH)
    }

    fn cv_stack_get(&self, index: usize) -> [u32; 8] {
        assert!(index < self.cv_stack_len);
        if index < CV_STACK_INLINE_LEN_MAX {
            self.cv_stack_inline[index]
        } else {
            self.cv_stack_spill[index - CV_STACK_INLINE_LEN_MAX]
        }
    }

    fn cv_stack_push(&mut self, cv: [u32; 8]) {
        assert!(self.cv_stack_len < CV_STACK_LEN_MAX);
        if self.cv_stack_len < CV_STACK_INLINE_LEN_MAX {
            self.cv_stack_inline[self.cv_stack_len] = cv;
        } else {
            self.cv_stack_spill.push(cv);
        }
        self.cv_stack_len += 1;
        assert!(self.cv_stack_spill.len() == self.cv_stack_len.saturating_sub(CV_STACK_INLINE_LEN_MAX));
    }

    fn cv_stack_pop(&mut self) -> [u32; 8] {
        let cv = self.cv_stack_get(self.cv_stack_len - 1);
        if self.cv_stack_len > CV_STACK_INLINE_LEN_MAX {
            self.cv_stack_spill.pop();
        }
        self.cv_stack_len -= 1;
        assert!(self.cv_stack_spill.len() == self.cv_stack_len.saturating_sub(CV_STACK_INLINE_LEN_MAX));
        cv
    }

    fn push_chunk_cv(&mut self, cv_chunk: [u32; 8], chunks_total: u64) {
        assert!(chunks_total > 0);
        let mut cv = cv_chunk;
        let mut chunks = chunks_total;
        while chunks & 1 == 0 {
            let left = self.cv_stack_pop();
            cv = parent_output(&left, &cv, &self.key, self.flags).chaining_value();
            chunks >>= 1;
        }
        self.cv_stack_push(cv);
        assert!(self.cv_stack_len as u32 == chunks_total.count_ones());
    }

    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            if self.chunk.len() == CHUNK_LEN {
                let cv_chunk = self.chunk.output().chaining_value();
                let chunks_total = self.chunk.chunk_counter + 1;
                self.push_chunk_cv(cv_chunk, chunks_total);
                self.chunk = ChunkState::new(&self.key, chunks_total, self.flags);
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
            if self.chunk.len() == 0 && input.len() > 4 * CHUNK_LEN {
                while input.len() > 8 * CHUNK_LEN {
                    let (eight, rest) =
                        input.split_first_chunk::<{ 8 * CHUNK_LEN }>().expect("eight chunks remain");
                    let first = self.chunk.chunk_counter;
                    let cvs = neon::compress_8_chunks(&self.key, eight, first, self.flags);
                    for (offset, cv) in cvs.into_iter().enumerate() {
                        self.push_chunk_cv(cv, first + offset as u64 + 1);
                    }
                    self.chunk = ChunkState::new(&self.key, first + 8, self.flags);
                    input = rest;
                }
                while input.len() > 4 * CHUNK_LEN {
                    let (four, rest) =
                        input.split_first_chunk::<{ 4 * CHUNK_LEN }>().expect("four chunks remain");
                    let first = self.chunk.chunk_counter;
                    let cvs = neon::compress_4_chunks(&self.key, four, first, self.flags);
                    for (offset, cv) in cvs.into_iter().enumerate() {
                        self.push_chunk_cv(cv, first + offset as u64 + 1);
                    }
                    self.chunk = ChunkState::new(&self.key, first + 4, self.flags);
                    input = rest;
                }
                continue;
            }
            let take = (CHUNK_LEN - self.chunk.len()).min(input.len());
            assert!(take > 0);
            self.chunk.update(&input[..take]);
            input = &input[take..];
        }
        self
    }

    fn root_output(&self) -> Output {
        assert!(self.cv_stack_len <= CV_STACK_LEN_MAX);
        let mut output = self.chunk.output();
        let mut remaining = self.cv_stack_len;
        while remaining > 0 {
            remaining -= 1;
            let left = self.cv_stack_get(remaining);
            output = parent_output(&left, &output.chaining_value(), &self.key, self.flags);
        }
        output
    }

    pub fn finalize(&self) -> [u8; OUT_LEN] {
        self.root_output().root_hash()
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
        let mut chunk = ChunkState::new(&key, 0, flags);
        chunk.update(input);
        return chunk.output().root_hash();
    }
    let mut hasher = Hasher::with_key_words(key, flags);
    hasher.update(input);
    hasher.finalize()
}

pub(crate) fn keyed_hash_words(key: &[u32; 8], input: &[u8]) -> [u8; OUT_LEN] {
    hash_with_key_words(*key, KEYED_HASH, input)
}

pub(crate) fn keyed_hash_block(key: &[u32; 8], block: &[u8; BLOCK_LEN]) -> [u8; OUT_LEN] {
    const { assert!(BLOCK_LEN == 64 && OUT_LEN == 32) };
    let message = words_from_block(block);
    let flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
    hash_from_cv(compress_cv(key, &message, 0, BLOCK_LEN as u32, flags))
}

pub(crate) fn keyed_hash_parent_8x(key: &[u32; 8], children: &[[u8; 32]; 16], out: &mut [[u8; OUT_LEN]; 8]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_parent_8x(key, children, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for lane in 0..8 {
            let mut block = [0u8; BLOCK_LEN];
            block[..32].copy_from_slice(&children[2 * lane]);
            block[32..].copy_from_slice(&children[2 * lane + 1]);
            out[lane] = keyed_hash_block(key, &block);
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
        for lane in 0..4 {
            let mut block = [0u8; BLOCK_LEN];
            block[..32].copy_from_slice(&children[2 * lane]);
            block[32..].copy_from_slice(&children[2 * lane + 1]);
            out[lane] = keyed_hash_block(key, &block);
        }
    }
}

pub(crate) fn keyed_hash_8x(key: &[u32; 8], inputs: [&[u8]; 8], out: &mut [[u8; OUT_LEN]; 8]) {
    let len_bytes = inputs[0].len();
    assert!(inputs.iter().all(|slice| slice.len() == len_bytes));
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_8x(key, inputs, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for lane in 0..8 {
            out[lane] = keyed_hash_words(key, inputs[lane]);
        }
    }
}

pub(crate) fn keyed_hash_4x(key: &[u32; 8], inputs: [&[u8]; 4], out: &mut [[u8; OUT_LEN]; 4]) {
    let len_bytes = inputs[0].len();
    assert!(inputs.iter().all(|slice| slice.len() == len_bytes));
    #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
    {
        neon::keyed_hash_4x(key, inputs, out);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little")))]
    {
        for lane in 0..4 {
            out[lane] = keyed_hash_words(key, inputs[lane]);
        }
    }
}

pub(crate) fn pow_bits_with_key_words(key: &[u32; 8], seed: &[u8; 32], nonce: u64) -> u32 {
    const { assert!(KEY_LEN == 32 && BLOCK_LEN == 64) };
    let seed_words = words_from_key(seed);
    let mut message = [0u32; 16];
    message[..8].copy_from_slice(&seed_words);
    message[8] = nonce as u32;
    message[9] = (nonce >> 32) as u32;
    let flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;
    let state = compress(key, &message, 0, 40, flags);
    let head = ((state[0].swap_bytes() as u64) << 32) | (state[1].swap_bytes() as u64);
    let bits = head.leading_zeros();
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
        neon::find_pow_nonce_4x(key, seed, bits, start, count)
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
    buffer: [u8; 8 * BLOCK_LEN],
    buffer_position: usize,
    buffer_len: usize,
}

impl OutputReader {
    fn new(output: Output) -> Self {
        assert!(output.flags & ROOT == 0);
        Self { output, block_counter: 0, buffer: [0; 8 * BLOCK_LEN], buffer_position: 0, buffer_len: 0 }
    }

    pub fn position(&self) -> u64 {
        assert!(self.buffer_position <= self.buffer_len);
        assert!(self.buffer_len <= 8 * BLOCK_LEN);
        self.block_counter * BLOCK_LEN as u64 - (self.buffer_len - self.buffer_position) as u64
    }

    fn refill(&mut self) {
        assert!(self.buffer_position == self.buffer_len);
        assert!(self.block_counter < u64::MAX);
        #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
        if self.block_counter <= u64::MAX - 8 {
            neon::root_blocks_8x(&self.output, self.block_counter, &mut self.buffer);
            self.block_counter += 8;
            self.buffer_position = 0;
            self.buffer_len = 8 * BLOCK_LEN;
            return;
        }
        let block = self.output.root_block(self.block_counter);
        self.buffer[..BLOCK_LEN].copy_from_slice(&block);
        self.block_counter += 1;
        self.buffer_position = 0;
        self.buffer_len = BLOCK_LEN;
    }

    pub fn fill(&mut self, buffer: &mut [u8]) {
        let mut filled = 0;
        while filled < buffer.len() {
            if self.buffer_position == self.buffer_len {
                self.refill();
            }
            let take = (self.buffer_len - self.buffer_position).min(buffer.len() - filled);
            buffer[filled..filled + take]
                .copy_from_slice(&self.buffer[self.buffer_position..self.buffer_position + take]);
            self.buffer_position += take;
            filled += take;
        }
        assert!(filled == buffer.len());
        assert!(self.buffer_position <= self.buffer_len);
    }
}
