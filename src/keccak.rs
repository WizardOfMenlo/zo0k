pub const OUT_LEN: usize = 32;
pub const RATE_BYTES: usize = 136;

const RATE_LANES: usize = RATE_BYTES / 8;
pub const LANES: usize = 25;
const ROUNDS: usize = 24;

const ROUND_CONSTANTS: [u64; ROUNDS] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

#[inline(always)]
fn round_scalar_inplace(a: &mut [u64; LANES], round_constant: u64) {
    let c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    let c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    let c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    let c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    let c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    let d = [
        c4 ^ c1.rotate_left(1),
        c0 ^ c2.rotate_left(1),
        c1 ^ c3.rotate_left(1),
        c2 ^ c4.rotate_left(1),
        c3 ^ c0.rotate_left(1),
    ];
    for x in 0..5 {
        a[x] ^= d[x];
        a[x + 5] ^= d[x];
        a[x + 10] ^= d[x];
        a[x + 15] ^= d[x];
        a[x + 20] ^= d[x];
    }
    let mut t = a[1];
    macro_rules! rho_pi {
        ($i:expr, $r:expr) => {
            let next = a[$i];
            a[$i] = t.rotate_left($r);
            t = next;
        };
    }
    rho_pi!(10, 1);
    rho_pi!(7, 3);
    rho_pi!(11, 6);
    rho_pi!(17, 10);
    rho_pi!(18, 15);
    rho_pi!(3, 21);
    rho_pi!(5, 28);
    rho_pi!(16, 36);
    rho_pi!(8, 45);
    rho_pi!(21, 55);
    rho_pi!(24, 2);
    rho_pi!(4, 14);
    rho_pi!(15, 27);
    rho_pi!(23, 41);
    rho_pi!(19, 56);
    rho_pi!(13, 8);
    rho_pi!(12, 25);
    rho_pi!(2, 43);
    rho_pi!(20, 62);
    rho_pi!(14, 18);
    rho_pi!(22, 39);
    rho_pi!(9, 61);
    rho_pi!(6, 20);
    a[1] = t.rotate_left(44);
    for row in 0..5 {
        let y = row * 5;
        let (b0, b1, b2, b3, b4) = (a[y], a[y + 1], a[y + 2], a[y + 3], a[y + 4]);
        a[y] = b0 ^ (!b1 & b2);
        a[y + 1] = b1 ^ (!b2 & b3);
        a[y + 2] = b2 ^ (!b3 & b4);
        a[y + 3] = b3 ^ (!b4 & b0);
        a[y + 4] = b4 ^ (!b0 & b1);
    }
    a[0] ^= round_constant;
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
#[inline(always)]
fn final_round_row0_scalar(a: &[u64; LANES]) -> [u64; 4] {
    let c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    let c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    let c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    let c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    let c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    let d0 = c4 ^ c1.rotate_left(1);
    let d1 = c0 ^ c2.rotate_left(1);
    let d2 = c1 ^ c3.rotate_left(1);
    let d3 = c2 ^ c4.rotate_left(1);
    let d4 = c3 ^ c0.rotate_left(1);
    let b0 = a[0] ^ d0;
    let b1 = (a[6] ^ d1).rotate_left(44);
    let b2 = (a[12] ^ d2).rotate_left(43);
    let b3 = (a[18] ^ d3).rotate_left(21);
    let b4 = (a[24] ^ d4).rotate_left(14);
    [(b0 ^ (!b1 & b2)) ^ ROUND_CONSTANTS[23], b1 ^ (!b2 & b3), b2 ^ (!b3 & b4), b3 ^ (!b4 & b0)]
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
#[inline(always)]
fn final_round_lane0_scalar(a: &[u64; LANES]) -> u64 {
    let c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    let c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    let c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    let c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    let c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    let d0 = c4 ^ c1.rotate_left(1);
    let d1 = c0 ^ c2.rotate_left(1);
    let d2 = c1 ^ c3.rotate_left(1);
    let b0 = a[0] ^ d0;
    let b1 = (a[6] ^ d1).rotate_left(44);
    let b2 = (a[12] ^ d2).rotate_left(43);
    (b0 ^ (!b1 & b2)) ^ ROUND_CONSTANTS[23]
}

pub fn keccak_f1600_portable(state: &mut [u64; LANES]) {
    const { assert!(ROUND_CONSTANTS.len() == ROUNDS) };
    for &round_constant in &ROUND_CONSTANTS {
        round_scalar_inplace(state, round_constant);
    }
}

pub(crate) const fn lanes_from_32(bytes: &[u8; 32]) -> [u64; 4] {
    let mut lanes = [0u64; 4];
    let mut index = 0;
    while index < 4 {
        let base = index * 8;
        lanes[index] = u64::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
            bytes[base + 4],
            bytes[base + 5],
            bytes[base + 6],
            bytes[base + 7],
        ]);
        index += 1;
    }
    lanes
}

#[inline(always)]
fn digest_from_words(words: &[u64; 4]) -> [u8; OUT_LEN] {
    let mut digest = [0u8; OUT_LEN];
    for (bytes, &lane) in digest.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        *bytes = lane.to_le_bytes();
    }
    digest
}

#[inline(always)]
fn digest_from_state(state: &[u64; LANES]) -> [u8; OUT_LEN] {
    digest_from_words(&[state[0], state[1], state[2], state[3]])
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
mod sha3_extension {
    use super::{
        LANES, OUT_LEN, RATE_BYTES, RATE_LANES, ROUND_CONSTANTS, digest_from_words, final_round_lane0_scalar,
        final_round_row0_scalar, lanes_from_32, round_scalar_inplace,
    };
    use std::arch::aarch64::*;

    type V = uint64x2_t;

    #[inline(always)]
    unsafe fn pair(x: u64, y: u64) -> V {
        unsafe { vsetq_lane_u64::<1>(y, vdupq_n_u64(x)) }
    }

    #[inline(always)]
    unsafe fn round(a: &mut [V; LANES], round_constant: u64) {
        unsafe {
            let c = [
                veor3q_u64(veor3q_u64(a[0], a[5], a[10]), a[15], a[20]),
                veor3q_u64(veor3q_u64(a[1], a[6], a[11]), a[16], a[21]),
                veor3q_u64(veor3q_u64(a[2], a[7], a[12]), a[17], a[22]),
                veor3q_u64(veor3q_u64(a[3], a[8], a[13]), a[18], a[23]),
                veor3q_u64(veor3q_u64(a[4], a[9], a[14]), a[19], a[24]),
            ];
            let d = [
                vrax1q_u64(c[4], c[1]),
                vrax1q_u64(c[0], c[2]),
                vrax1q_u64(c[1], c[3]),
                vrax1q_u64(c[2], c[4]),
                vrax1q_u64(c[3], c[0]),
            ];
            let b = [
                veorq_u64(a[0], d[0]),
                vxarq_u64::<20>(a[6], d[1]),
                vxarq_u64::<21>(a[12], d[2]),
                vxarq_u64::<43>(a[18], d[3]),
                vxarq_u64::<50>(a[24], d[4]),
                vxarq_u64::<36>(a[3], d[3]),
                vxarq_u64::<44>(a[9], d[4]),
                vxarq_u64::<61>(a[10], d[0]),
                vxarq_u64::<19>(a[16], d[1]),
                vxarq_u64::<3>(a[22], d[2]),
                vxarq_u64::<63>(a[1], d[1]),
                vxarq_u64::<58>(a[7], d[2]),
                vxarq_u64::<39>(a[13], d[3]),
                vxarq_u64::<56>(a[19], d[4]),
                vxarq_u64::<46>(a[20], d[0]),
                vxarq_u64::<37>(a[4], d[4]),
                vxarq_u64::<28>(a[5], d[0]),
                vxarq_u64::<54>(a[11], d[1]),
                vxarq_u64::<49>(a[17], d[2]),
                vxarq_u64::<8>(a[23], d[3]),
                vxarq_u64::<2>(a[2], d[2]),
                vxarq_u64::<9>(a[8], d[3]),
                vxarq_u64::<25>(a[14], d[4]),
                vxarq_u64::<23>(a[15], d[0]),
                vxarq_u64::<62>(a[21], d[1]),
            ];
            for y in 0..5 {
                for x in 0..5 {
                    a[5 * y + x] = vbcaxq_u64(b[5 * y + x], b[5 * y + (x + 2) % 5], b[5 * y + (x + 1) % 5]);
                }
            }
            a[0] = veorq_u64(a[0], vdupq_n_u64(round_constant));
        }
    }

    #[inline(always)]
    unsafe fn final_round_row0_2x(a: &[V; LANES]) -> [V; 4] {
        unsafe {
            let c0 = veor3q_u64(veor3q_u64(a[0], a[5], a[10]), a[15], a[20]);
            let c1 = veor3q_u64(veor3q_u64(a[1], a[6], a[11]), a[16], a[21]);
            let c2 = veor3q_u64(veor3q_u64(a[2], a[7], a[12]), a[17], a[22]);
            let c3 = veor3q_u64(veor3q_u64(a[3], a[8], a[13]), a[18], a[23]);
            let c4 = veor3q_u64(veor3q_u64(a[4], a[9], a[14]), a[19], a[24]);
            let d0 = vrax1q_u64(c4, c1);
            let d1 = vrax1q_u64(c0, c2);
            let d2 = vrax1q_u64(c1, c3);
            let d3 = vrax1q_u64(c2, c4);
            let d4 = vrax1q_u64(c3, c0);
            let b0 = veorq_u64(a[0], d0);
            let b1 = vxarq_u64::<20>(a[6], d1);
            let b2 = vxarq_u64::<21>(a[12], d2);
            let b3 = vxarq_u64::<43>(a[18], d3);
            let b4 = vxarq_u64::<50>(a[24], d4);
            [
                veorq_u64(vbcaxq_u64(b0, b2, b1), vdupq_n_u64(ROUND_CONSTANTS[23])),
                vbcaxq_u64(b1, b3, b2),
                vbcaxq_u64(b2, b4, b3),
                vbcaxq_u64(b3, b0, b4),
            ]
        }
    }

    #[inline(always)]
    unsafe fn final_round_lane0_2x(a: &[V; LANES]) -> V {
        unsafe {
            let c0 = veor3q_u64(veor3q_u64(a[0], a[5], a[10]), a[15], a[20]);
            let c1 = veor3q_u64(veor3q_u64(a[1], a[6], a[11]), a[16], a[21]);
            let c2 = veor3q_u64(veor3q_u64(a[2], a[7], a[12]), a[17], a[22]);
            let c3 = veor3q_u64(veor3q_u64(a[3], a[8], a[13]), a[18], a[23]);
            let c4 = veor3q_u64(veor3q_u64(a[4], a[9], a[14]), a[19], a[24]);
            let d0 = vrax1q_u64(c4, c1);
            let d1 = vrax1q_u64(c0, c2);
            let d2 = vrax1q_u64(c1, c3);
            let b0 = veorq_u64(a[0], d0);
            let b1 = vxarq_u64::<20>(a[6], d1);
            let b2 = vxarq_u64::<21>(a[12], d2);
            veorq_u64(vbcaxq_u64(b0, b2, b1), vdupq_n_u64(ROUND_CONSTANTS[23]))
        }
    }

    #[inline(always)]
    unsafe fn extract_2_digests(row0: &[V; 4]) -> [[u8; OUT_LEN]; 2] {
        let mut out = [[0u8; OUT_LEN]; 2];
        unsafe {
            let lo0 = vzip1q_u64(row0[0], row0[1]);
            let lo1 = vzip2q_u64(row0[0], row0[1]);
            let hi0 = vzip1q_u64(row0[2], row0[3]);
            let hi1 = vzip2q_u64(row0[2], row0[3]);
            vst1q_u8(out[0].as_mut_ptr(), vreinterpretq_u8_u64(lo0));
            vst1q_u8(out[0].as_mut_ptr().add(16), vreinterpretq_u8_u64(hi0));
            vst1q_u8(out[1].as_mut_ptr(), vreinterpretq_u8_u64(lo1));
            vst1q_u8(out[1].as_mut_ptr().add(16), vreinterpretq_u8_u64(hi1));
        }
        out
    }

    #[inline(always)]
    unsafe fn permute_2x(a: &mut [V; LANES]) {
        for &rc in &ROUND_CONSTANTS {
            unsafe { round(a, rc) };
        }
    }

    #[inline(always)]
    unsafe fn permute_3x(a: &mut [V; LANES], s: &mut [u64; LANES]) {
        for &rc in &ROUND_CONSTANTS {
            unsafe { round(a, rc) };
            round_scalar_inplace(s, rc);
        }
    }

    #[inline(always)]
    unsafe fn permute_to_2_digests(a: &mut [V; LANES]) -> [[u8; OUT_LEN]; 2] {
        for &rc in &ROUND_CONSTANTS[..23] {
            unsafe { round(a, rc) };
        }
        unsafe { extract_2_digests(&final_round_row0_2x(a)) }
    }

    #[inline(always)]
    unsafe fn permute_to_3_digests(a: &mut [V; LANES], s: &mut [u64; LANES]) -> [[u8; OUT_LEN]; 3] {
        for &rc in &ROUND_CONSTANTS[..23] {
            unsafe { round(a, rc) };
            round_scalar_inplace(s, rc);
        }
        let d01 = unsafe { extract_2_digests(&final_round_row0_2x(a)) };
        let d2 = digest_from_words(&final_round_row0_scalar(s));
        [d01[0], d01[1], d2]
    }

    #[inline(always)]
    unsafe fn load_pair2(p0: *const u8, p1: *const u8) -> (V, V) {
        unsafe {
            let a = vreinterpretq_u64_u8(vld1q_u8(p0));
            let b = vreinterpretq_u64_u8(vld1q_u8(p1));
            (vzip1q_u64(a, b), vzip2q_u64(a, b))
        }
    }

    pub fn keccak_f1600(state: &mut [u64; LANES]) {
        const { assert!(ROUND_CONSTANTS.len() == 24) };
        let mut lanes = unsafe { [vdupq_n_u64(0); LANES] };
        for (lane, &word) in lanes.iter_mut().zip(state.iter()) {
            *lane = unsafe { vdupq_n_u64(word) };
        }
        unsafe { permute_2x(&mut lanes) };
        for (word, &lane) in state.iter_mut().zip(lanes.iter()) {
            *word = unsafe { vgetq_lane_u64::<0>(lane) };
        }
    }

    pub fn keccak256_node_pair_3x(key_lanes: &[u64; 4], children: &[[u8; 32]; 6]) -> [[u8; OUT_LEN]; 3] {
        const { assert!(RATE_BYTES == 136 && OUT_LEN == 32) };
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut a = [zero; LANES];
            let mut s = [0u64; LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
                s[i] = key_lanes[i];
            }
            (a[4], a[5]) = load_pair2(children[0].as_ptr(), children[2].as_ptr());
            (a[6], a[7]) = load_pair2(children[0].as_ptr().add(16), children[2].as_ptr().add(16));
            (a[8], a[9]) = load_pair2(children[1].as_ptr(), children[3].as_ptr());
            (a[10], a[11]) = load_pair2(children[1].as_ptr().add(16), children[3].as_ptr().add(16));
            a[12] = vdupq_n_u64(0x01);
            a[16] = vdupq_n_u64(0x8000_0000_0000_0000);
            s[4..8].copy_from_slice(&lanes_from_32(&children[4]));
            s[8..12].copy_from_slice(&lanes_from_32(&children[5]));
            s[12] = 0x01;
            s[16] = 0x8000_0000_0000_0000;
            permute_to_3_digests(&mut a, &mut s)
        }
    }

    pub fn keccak256_node_pair_2x(key_lanes: &[u64; 4], children: &[[u8; 32]; 4]) -> [[u8; OUT_LEN]; 2] {
        const { assert!(RATE_BYTES == 136 && OUT_LEN == 32) };
        unsafe {
            let mut a = [vdupq_n_u64(0); LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
            }
            (a[4], a[5]) = load_pair2(children[0].as_ptr(), children[2].as_ptr());
            (a[6], a[7]) = load_pair2(children[0].as_ptr().add(16), children[2].as_ptr().add(16));
            (a[8], a[9]) = load_pair2(children[1].as_ptr(), children[3].as_ptr());
            (a[10], a[11]) = load_pair2(children[1].as_ptr().add(16), children[3].as_ptr().add(16));
            a[12] = vdupq_n_u64(0x01);
            a[16] = vdupq_n_u64(0x8000_0000_0000_0000);
            permute_to_2_digests(&mut a)
        }
    }

    #[inline(always)]
    unsafe fn load_row_2x_32<F: crate::field::Field>(row: &[F], base: usize) -> [V; 4] {
        assert!(F::BYTES == 32 && base + 2 <= row.len());
        let mut raw0 = [0u8; 32];
        let mut raw1 = [0u8; 32];
        row[base].write_bytes(&mut raw0);
        row[base + 1].write_bytes(&mut raw1);
        unsafe {
            let (v0, v1) = load_pair2(raw0.as_ptr(), raw1.as_ptr());
            let (v2, v3) = load_pair2(raw0.as_ptr().add(16), raw1.as_ptr().add(16));
            [v0, v1, v2, v3]
        }
    }

    #[inline(always)]
    fn load_row_1x_32<F: crate::field::Field>(row: &[F], index: usize) -> [u64; 4] {
        assert!(F::BYTES == 32 && index < row.len());
        let mut raw = [0u8; 32];
        row[index].write_bytes(&mut raw);
        lanes_from_32(&raw)
    }

    #[inline(always)]
    unsafe fn load_row_2x_16<F: crate::field::Field>(row: &[F], base: usize) -> [V; 2] {
        assert!(F::BYTES == 16 && base + 2 <= row.len());
        let mut raw0 = [0u8; 16];
        let mut raw1 = [0u8; 16];
        row[base].write_bytes(&mut raw0);
        row[base + 1].write_bytes(&mut raw1);
        unsafe {
            let (v0, v1) = load_pair2(raw0.as_ptr(), raw1.as_ptr());
            [v0, v1]
        }
    }

    #[inline(always)]
    fn load_row_1x_16<F: crate::field::Field>(row: &[F], index: usize) -> [u64; 2] {
        assert!(F::BYTES == 16 && index < row.len());
        let mut raw = [0u8; 16];
        row[index].write_bytes(&mut raw);
        [
            u64::from_le_bytes(raw[0..8].try_into().expect("eight bytes")),
            u64::from_le_bytes(raw[8..16].try_into().expect("eight bytes")),
        ]
    }

    #[inline(always)]
    unsafe fn xor4_2x(a: &mut [V; LANES], start: usize, v: [V; 4]) {
        assert!(start + 4 <= RATE_LANES);
        unsafe {
            a[start] = veorq_u64(a[start], v[0]);
            a[start + 1] = veorq_u64(a[start + 1], v[1]);
            a[start + 2] = veorq_u64(a[start + 2], v[2]);
            a[start + 3] = veorq_u64(a[start + 3], v[3]);
        }
    }

    #[inline(always)]
    fn xor4_1x(s: &mut [u64; LANES], start: usize, v: [u64; 4]) {
        assert!(start + 4 <= RATE_LANES);
        s[start] ^= v[0];
        s[start + 1] ^= v[1];
        s[start + 2] ^= v[2];
        s[start + 3] ^= v[3];
    }

    #[inline(always)]
    unsafe fn xor2_2x(a: &mut [V; LANES], start: usize, v: [V; 2]) {
        assert!(start + 2 <= RATE_LANES);
        unsafe {
            a[start] = veorq_u64(a[start], v[0]);
            a[start + 1] = veorq_u64(a[start + 1], v[1]);
        }
    }

    #[inline(always)]
    unsafe fn absorb_f32_phase_3x<F: crate::field::Field>(
        a: &mut [V; LANES],
        s: &mut [u64; LANES],
        rows: &[Vec<F>],
        base: usize,
        phase: usize,
        carry_v: &mut [V; 3],
        carry_s: &mut [u64; 3],
    ) -> usize {
        assert!(phase < 4 && rows.len() >= 4 + usize::from(phase == 0));
        unsafe {
            match phase {
                0 => {
                    for (i, row) in rows.iter().enumerate().take(4) {
                        xor4_2x(a, 4 * i, load_row_2x_32(row, base));
                        xor4_1x(s, 4 * i, load_row_1x_32(row, base + 2));
                    }
                    let rv = load_row_2x_32(&rows[4], base);
                    let rs = load_row_1x_32(&rows[4], base + 2);
                    a[16] = veorq_u64(a[16], rv[0]);
                    s[16] ^= rs[0];
                    *carry_v = [rv[1], rv[2], rv[3]];
                    *carry_s = [rs[1], rs[2], rs[3]];
                    5
                }
                1 => {
                    for (i, row) in rows.iter().enumerate().take(3) {
                        xor4_2x(a, 3 + 4 * i, load_row_2x_32(row, base));
                        xor4_1x(s, 3 + 4 * i, load_row_1x_32(row, base + 2));
                    }
                    let rv = load_row_2x_32(&rows[3], base);
                    let rs = load_row_1x_32(&rows[3], base + 2);
                    a[15] = veorq_u64(a[15], rv[0]);
                    a[16] = veorq_u64(a[16], rv[1]);
                    s[15] ^= rs[0];
                    s[16] ^= rs[1];
                    carry_v[0] = rv[2];
                    carry_v[1] = rv[3];
                    carry_s[0] = rs[2];
                    carry_s[1] = rs[3];
                    4
                }
                2 => {
                    for (i, row) in rows.iter().enumerate().take(3) {
                        xor4_2x(a, 2 + 4 * i, load_row_2x_32(row, base));
                        xor4_1x(s, 2 + 4 * i, load_row_1x_32(row, base + 2));
                    }
                    let rv = load_row_2x_32(&rows[3], base);
                    let rs = load_row_1x_32(&rows[3], base + 2);
                    a[14] = veorq_u64(a[14], rv[0]);
                    a[15] = veorq_u64(a[15], rv[1]);
                    a[16] = veorq_u64(a[16], rv[2]);
                    s[14] ^= rs[0];
                    s[15] ^= rs[1];
                    s[16] ^= rs[2];
                    carry_v[0] = rv[3];
                    carry_s[0] = rs[3];
                    4
                }
                _ => {
                    for (i, row) in rows.iter().enumerate().take(4) {
                        xor4_2x(a, 1 + 4 * i, load_row_2x_32(row, base));
                        xor4_1x(s, 1 + 4 * i, load_row_1x_32(row, base + 2));
                    }
                    4
                }
            }
        }
    }

    fn hash_rows_3x_f32<F: crate::field::Field>(
        key_lanes: &[u64; 4],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 3],
    ) {
        assert!(F::BYTES == 32 && !rows.is_empty());
        assert!(base + 3 <= rows[0].len());
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut a = [zero; LANES];
            let mut s = [0u64; LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
                s[i] = key_lanes[i];
            }
            a[4] = pair(base as u64, (base + 1) as u64);
            s[4] = (base + 2) as u64;
            let mut idx = 0usize;
            let mut pos = 5usize;
            if rows.len() >= 3 {
                for (i, row) in rows.iter().enumerate().take(3) {
                    let rv = load_row_2x_32(row, base);
                    let rs = load_row_1x_32(row, base + 2);
                    a[5 + 4 * i..9 + 4 * i].copy_from_slice(&rv);
                    s[5 + 4 * i..9 + 4 * i].copy_from_slice(&rs);
                }
                permute_3x(&mut a, &mut s);
                idx = 3;
                let mut phase = 0usize;
                let mut carry_v = [zero; 3];
                let mut carry_s = [0u64; 3];
                let max_blocks = rows.len() / 4 + 2;
                for _ in 0..max_blocks {
                    let carry_len = (4 - phase) & 3;
                    for i in 0..carry_len {
                        a[i] = veorq_u64(a[i], carry_v[i]);
                        s[i] ^= carry_s[i];
                    }
                    let need = 4 + usize::from(phase == 0);
                    if idx + need > rows.len() {
                        pos = carry_len;
                        break;
                    }
                    idx += absorb_f32_phase_3x(
                        &mut a,
                        &mut s,
                        &rows[idx..],
                        base,
                        phase,
                        &mut carry_v,
                        &mut carry_s,
                    );
                    permute_3x(&mut a, &mut s);
                    phase = (phase + 1) & 3;
                }
            }
            for row in &rows[idx..] {
                xor4_2x(&mut a, pos, load_row_2x_32(row, base));
                xor4_1x(&mut s, pos, load_row_1x_32(row, base + 2));
                pos += 4;
            }
            a[pos] = veorq_u64(a[pos], vdupq_n_u64(0x01));
            a[RATE_LANES - 1] = veorq_u64(a[RATE_LANES - 1], vdupq_n_u64(0x8000_0000_0000_0000));
            s[pos] ^= 0x01;
            s[RATE_LANES - 1] ^= 0x8000_0000_0000_0000;
            *out = permute_to_3_digests(&mut a, &mut s);
        }
    }

    #[inline(always)]
    unsafe fn absorb_f16_blocks_3x<F: crate::field::Field>(
        a: &mut [V; LANES],
        s: &mut [u64; LANES],
        rows: &[Vec<F>],
        base: usize,
        zero: V,
    ) -> (usize, usize) {
        assert!(rows.len() >= 6);
        unsafe {
            for (i, row) in rows.iter().enumerate().take(6) {
                let rv = load_row_2x_16(row, base);
                let rs = load_row_1x_16(row, base + 2);
                a[5 + 2 * i] = rv[0];
                a[6 + 2 * i] = rv[1];
                s[5 + 2 * i] = rs[0];
                s[6 + 2 * i] = rs[1];
            }
            permute_3x(a, s);
            let mut idx = 6usize;
            let mut phase = 0usize;
            let mut carry_v = zero;
            let mut carry_s = 0u64;
            for _ in 0..(rows.len() / 8 + 2) {
                if phase == 1 {
                    a[0] = veorq_u64(a[0], carry_v);
                    s[0] ^= carry_s;
                }
                let need = 9 - phase;
                if idx + need > rows.len() {
                    return (idx, phase);
                }
                for i in 0..8 {
                    let rv = load_row_2x_16(&rows[idx + i], base);
                    let rs = load_row_1x_16(&rows[idx + i], base + 2);
                    xor2_2x(a, phase + 2 * i, rv);
                    s[phase + 2 * i] ^= rs[0];
                    s[phase + 2 * i + 1] ^= rs[1];
                }
                if phase == 0 {
                    let rv = load_row_2x_16(&rows[idx + 8], base);
                    let rs = load_row_1x_16(&rows[idx + 8], base + 2);
                    a[16] = veorq_u64(a[16], rv[0]);
                    s[16] ^= rs[0];
                    carry_v = rv[1];
                    carry_s = rs[1];
                }
                idx += need;
                permute_3x(a, s);
                phase ^= 1;
            }
            unreachable!("bounded loop finishes all rows")
        }
    }

    fn hash_rows_3x_f16<F: crate::field::Field>(
        key_lanes: &[u64; 4],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 3],
    ) {
        assert!(F::BYTES == 16 && !rows.is_empty());
        assert!(base + 3 <= rows[0].len());
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut a = [zero; LANES];
            let mut s = [0u64; LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
                s[i] = key_lanes[i];
            }
            a[4] = pair(base as u64, (base + 1) as u64);
            s[4] = (base + 2) as u64;
            let (idx, mut pos) =
                if rows.len() >= 6 { absorb_f16_blocks_3x(&mut a, &mut s, rows, base, zero) } else { (0, 5) };
            for row in &rows[idx..] {
                let rv = load_row_2x_16(row, base);
                let rs = load_row_1x_16(row, base + 2);
                xor2_2x(&mut a, pos, rv);
                s[pos] ^= rs[0];
                s[pos + 1] ^= rs[1];
                pos += 2;
            }
            a[pos] = veorq_u64(a[pos], vdupq_n_u64(0x01));
            a[RATE_LANES - 1] = veorq_u64(a[RATE_LANES - 1], vdupq_n_u64(0x8000_0000_0000_0000));
            s[pos] ^= 0x01;
            s[RATE_LANES - 1] ^= 0x8000_0000_0000_0000;
            *out = permute_to_3_digests(&mut a, &mut s);
        }
    }

    pub fn hash_rows_3x_direct<F: crate::field::Field>(
        key_lanes: &[u64; 4],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 3],
    ) {
        assert!(F::BYTES == 16 || F::BYTES == 32);
        assert!(!rows.is_empty());
        if F::BYTES == 32 {
            hash_rows_3x_f32(key_lanes, rows, base, out);
        } else {
            hash_rows_3x_f16(key_lanes, rows, base, out);
        }
    }

    #[inline(always)]
    unsafe fn absorb_f32_phase_2x<F: crate::field::Field>(
        a: &mut [V; LANES],
        rows: &[Vec<F>],
        base: usize,
        phase: usize,
        carry: &mut [V; 3],
    ) -> usize {
        assert!(phase < 4 && rows.len() >= 4 + usize::from(phase == 0));
        unsafe {
            match phase {
                0 => {
                    for (i, row) in rows.iter().enumerate().take(4) {
                        xor4_2x(a, 4 * i, load_row_2x_32(row, base));
                    }
                    let rv = load_row_2x_32(&rows[4], base);
                    a[16] = veorq_u64(a[16], rv[0]);
                    *carry = [rv[1], rv[2], rv[3]];
                    5
                }
                1 => {
                    for (i, row) in rows.iter().enumerate().take(3) {
                        xor4_2x(a, 3 + 4 * i, load_row_2x_32(row, base));
                    }
                    let rv = load_row_2x_32(&rows[3], base);
                    a[15] = veorq_u64(a[15], rv[0]);
                    a[16] = veorq_u64(a[16], rv[1]);
                    carry[0] = rv[2];
                    carry[1] = rv[3];
                    4
                }
                2 => {
                    for (i, row) in rows.iter().enumerate().take(3) {
                        xor4_2x(a, 2 + 4 * i, load_row_2x_32(row, base));
                    }
                    let rv = load_row_2x_32(&rows[3], base);
                    a[14] = veorq_u64(a[14], rv[0]);
                    a[15] = veorq_u64(a[15], rv[1]);
                    a[16] = veorq_u64(a[16], rv[2]);
                    carry[0] = rv[3];
                    4
                }
                _ => {
                    for (i, row) in rows.iter().enumerate().take(4) {
                        xor4_2x(a, 1 + 4 * i, load_row_2x_32(row, base));
                    }
                    4
                }
            }
        }
    }

    fn hash_rows_2x_f32<F: crate::field::Field>(
        key_lanes: &[u64; 4],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 2],
    ) {
        assert!(F::BYTES == 32 && base + 2 <= rows[0].len());
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut a = [zero; LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
            }
            a[4] = pair(base as u64, (base + 1) as u64);
            let mut idx = 0usize;
            let mut pos = 5usize;
            if rows.len() >= 3 {
                for (i, row) in rows.iter().enumerate().take(3) {
                    a[5 + 4 * i..9 + 4 * i].copy_from_slice(&load_row_2x_32(row, base));
                }
                permute_2x(&mut a);
                idx = 3;
                let mut phase = 0usize;
                let mut carry = [zero; 3];
                for _ in 0..(rows.len() / 4 + 2) {
                    let carry_len = (4 - phase) & 3;
                    for i in 0..carry_len {
                        a[i] = veorq_u64(a[i], carry[i]);
                    }
                    if idx + 4 + usize::from(phase == 0) > rows.len() {
                        pos = carry_len;
                        break;
                    }
                    idx += absorb_f32_phase_2x(&mut a, &rows[idx..], base, phase, &mut carry);
                    permute_2x(&mut a);
                    phase = (phase + 1) & 3;
                }
            }
            for row in &rows[idx..] {
                xor4_2x(&mut a, pos, load_row_2x_32(row, base));
                pos += 4;
            }
            a[pos] = veorq_u64(a[pos], vdupq_n_u64(0x01));
            a[RATE_LANES - 1] = veorq_u64(a[RATE_LANES - 1], vdupq_n_u64(0x8000_0000_0000_0000));
            *out = permute_to_2_digests(&mut a);
        }
    }

    fn hash_rows_2x_f16<F: crate::field::Field>(
        key_lanes: &[u64; 4],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 2],
    ) {
        assert!(F::BYTES == 16 && base + 2 <= rows[0].len());
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut a = [zero; LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
            }
            a[4] = pair(base as u64, (base + 1) as u64);
            let mut idx = 0usize;
            let mut pos = 5usize;
            if rows.len() >= 6 {
                for (i, row) in rows.iter().enumerate().take(6) {
                    let rv = load_row_2x_16(row, base);
                    a[5 + 2 * i] = rv[0];
                    a[6 + 2 * i] = rv[1];
                }
                permute_2x(&mut a);
                idx = 6;
                let mut phase = 0usize;
                let mut carry = zero;
                for _ in 0..(rows.len() / 8 + 2) {
                    if phase == 1 {
                        a[0] = veorq_u64(a[0], carry);
                    }
                    let need = 9 - phase;
                    if idx + need > rows.len() {
                        pos = phase;
                        break;
                    }
                    for i in 0..8 {
                        xor2_2x(&mut a, phase + 2 * i, load_row_2x_16(&rows[idx + i], base));
                    }
                    if phase == 0 {
                        let rv = load_row_2x_16(&rows[idx + 8], base);
                        a[16] = veorq_u64(a[16], rv[0]);
                        carry = rv[1];
                    }
                    idx += need;
                    permute_2x(&mut a);
                    phase ^= 1;
                }
            }
            for row in &rows[idx..] {
                xor2_2x(&mut a, pos, load_row_2x_16(row, base));
                pos += 2;
            }
            a[pos] = veorq_u64(a[pos], vdupq_n_u64(0x01));
            a[RATE_LANES - 1] = veorq_u64(a[RATE_LANES - 1], vdupq_n_u64(0x8000_0000_0000_0000));
            *out = permute_to_2_digests(&mut a);
        }
    }

    pub fn hash_rows_2x_direct<F: crate::field::Field>(
        key_lanes: &[u64; 4],
        rows: &[Vec<F>],
        base: usize,
        out: &mut [[u8; OUT_LEN]; 2],
    ) {
        assert!(F::BYTES == 16 || F::BYTES == 32);
        assert!(!rows.is_empty());
        if F::BYTES == 32 {
            hash_rows_2x_f32(key_lanes, rows, base, out);
        } else {
            hash_rows_2x_f16(key_lanes, rows, base, out);
        }
    }

    #[inline(always)]
    unsafe fn xor_full_block_2x(a: &mut [V; LANES], b0: &[u8; RATE_BYTES], b1: &[u8; RATE_BYTES]) {
        unsafe {
            let p0 = b0.as_ptr();
            let p1 = b1.as_ptr();
            for pair_idx in 0..8 {
                let off = pair_idx * 16;
                let (v0, v1) = load_pair2(p0.add(off), p1.add(off));
                a[2 * pair_idx] = veorq_u64(a[2 * pair_idx], v0);
                a[2 * pair_idx + 1] = veorq_u64(a[2 * pair_idx + 1], v1);
            }
            let w0 = u64::from_le_bytes(b0[128..136].try_into().expect("last lane"));
            let w1 = u64::from_le_bytes(b1[128..136].try_into().expect("last lane"));
            a[16] = veorq_u64(a[16], pair(w0, w1));
        }
    }

    #[inline(always)]
    fn xor_full_block_1x(s: &mut [u64; LANES], b: &[u8; RATE_BYTES]) {
        for (lane, chunk) in s[..RATE_LANES].iter_mut().zip(b.as_chunks::<8>().0) {
            *lane ^= u64::from_le_bytes(*chunk);
        }
    }

    #[inline(always)]
    unsafe fn load_first_payload_2x(a: &mut [V; LANES], b0: &[u8; 104], b1: &[u8; 104]) {
        unsafe {
            let p0 = b0.as_ptr();
            let p1 = b1.as_ptr();
            for pair_idx in 0..6 {
                let off = pair_idx * 16;
                (a[4 + 2 * pair_idx], a[5 + 2 * pair_idx]) = load_pair2(p0.add(off), p1.add(off));
            }
            let w0 = u64::from_le_bytes(b0[96..104].try_into().expect("last lane"));
            let w1 = u64::from_le_bytes(b1[96..104].try_into().expect("last lane"));
            a[16] = pair(w0, w1);
        }
    }

    #[inline(always)]
    fn load_first_payload_1x(s: &mut [u64; LANES], b: &[u8; 104]) {
        for (lane, chunk) in s[4..RATE_LANES].iter_mut().zip(b.as_chunks::<8>().0) {
            *lane = u64::from_le_bytes(*chunk);
        }
    }

    pub fn keccak256_domain_3x(
        key_lanes: &[u64; 4],
        in0: &[u8],
        in1: &[u8],
        in2: &[u8],
    ) -> [[u8; OUT_LEN]; 3] {
        assert!(in0.len() == in1.len() && in0.len() == in2.len());
        const FIRST_PAYLOAD: usize = RATE_BYTES - 32;
        const { assert!(FIRST_PAYLOAD == 104) };
        unsafe {
            let mut a = [vdupq_n_u64(0); LANES];
            let mut s = [0u64; LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
                s[i] = key_lanes[i];
            }
            if in0.len() < FIRST_PAYLOAD {
                let (mut p0, mut p1, mut p2) =
                    ([0u8; FIRST_PAYLOAD], [0u8; FIRST_PAYLOAD], [0u8; FIRST_PAYLOAD]);
                p0[..in0.len()].copy_from_slice(in0);
                p1[..in1.len()].copy_from_slice(in1);
                p2[..in2.len()].copy_from_slice(in2);
                p0[in0.len()] ^= 0x01;
                p1[in1.len()] ^= 0x01;
                p2[in2.len()] ^= 0x01;
                p0[FIRST_PAYLOAD - 1] ^= 0x80;
                p1[FIRST_PAYLOAD - 1] ^= 0x80;
                p2[FIRST_PAYLOAD - 1] ^= 0x80;
                load_first_payload_2x(&mut a, &p0, &p1);
                load_first_payload_1x(&mut s, &p2);
                return permute_to_3_digests(&mut a, &mut s);
            }
            let (h0, mut r0) = in0.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload");
            let (h1, mut r1) = in1.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload");
            let (h2, mut r2) = in2.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload");
            load_first_payload_2x(&mut a, h0, h1);
            load_first_payload_1x(&mut s, h2);
            permute_3x(&mut a, &mut s);
            while let (Some((b0, t0)), Some((b1, t1)), Some((b2, t2))) = (
                r0.split_first_chunk::<RATE_BYTES>(),
                r1.split_first_chunk::<RATE_BYTES>(),
                r2.split_first_chunk::<RATE_BYTES>(),
            ) {
                xor_full_block_2x(&mut a, b0, b1);
                xor_full_block_1x(&mut s, b2);
                permute_3x(&mut a, &mut s);
                r0 = t0;
                r1 = t1;
                r2 = t2;
            }
            assert!(r0.len() < RATE_BYTES);
            let (mut b0, mut b1, mut b2) = ([0u8; RATE_BYTES], [0u8; RATE_BYTES], [0u8; RATE_BYTES]);
            b0[..r0.len()].copy_from_slice(r0);
            b1[..r1.len()].copy_from_slice(r1);
            b2[..r2.len()].copy_from_slice(r2);
            b0[r0.len()] ^= 0x01;
            b1[r1.len()] ^= 0x01;
            b2[r2.len()] ^= 0x01;
            b0[RATE_BYTES - 1] ^= 0x80;
            b1[RATE_BYTES - 1] ^= 0x80;
            b2[RATE_BYTES - 1] ^= 0x80;
            xor_full_block_2x(&mut a, &b0, &b1);
            xor_full_block_1x(&mut s, &b2);
            permute_to_3_digests(&mut a, &mut s)
        }
    }

    pub fn keccak256_domain_2x(key_lanes: &[u64; 4], in0: &[u8], in1: &[u8]) -> [[u8; OUT_LEN]; 2] {
        assert!(in0.len() == in1.len());
        const FIRST_PAYLOAD: usize = RATE_BYTES - 32;
        const { assert!(FIRST_PAYLOAD == 104) };
        unsafe {
            let mut a = [vdupq_n_u64(0); LANES];
            for i in 0..4 {
                a[i] = vdupq_n_u64(key_lanes[i]);
            }
            if in0.len() < FIRST_PAYLOAD {
                let (mut p0, mut p1) = ([0u8; FIRST_PAYLOAD], [0u8; FIRST_PAYLOAD]);
                p0[..in0.len()].copy_from_slice(in0);
                p1[..in1.len()].copy_from_slice(in1);
                p0[in0.len()] ^= 0x01;
                p1[in1.len()] ^= 0x01;
                p0[FIRST_PAYLOAD - 1] ^= 0x80;
                p1[FIRST_PAYLOAD - 1] ^= 0x80;
                load_first_payload_2x(&mut a, &p0, &p1);
                return permute_to_2_digests(&mut a);
            }
            let (h0, mut r0) = in0.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload");
            let (h1, mut r1) = in1.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload");
            load_first_payload_2x(&mut a, h0, h1);
            permute_2x(&mut a);
            while let (Some((b0, t0)), Some((b1, t1))) =
                (r0.split_first_chunk::<RATE_BYTES>(), r1.split_first_chunk::<RATE_BYTES>())
            {
                xor_full_block_2x(&mut a, b0, b1);
                permute_2x(&mut a);
                r0 = t0;
                r1 = t1;
            }
            assert!(r0.len() < RATE_BYTES);
            let (mut b0, mut b1) = ([0u8; RATE_BYTES], [0u8; RATE_BYTES]);
            b0[..r0.len()].copy_from_slice(r0);
            b1[..r1.len()].copy_from_slice(r1);
            b0[r0.len()] ^= 0x01;
            b1[r1.len()] ^= 0x01;
            b0[RATE_BYTES - 1] ^= 0x80;
            b1[RATE_BYTES - 1] ^= 0x80;
            xor_full_block_2x(&mut a, &b0, &b1);
            permute_to_2_digests(&mut a)
        }
    }

    struct PowPrep {
        bs: [u64; LANES],
        bv: [V; LANES],
        d2_s: u64,
        d3_s: u64,
        d4_s: u64,
        a2_s: u64,
        a4_s: u64,
        a7_s: u64,
        d2_v: V,
        d3_v: V,
        d4_v: V,
        a2_v: V,
        a4_v: V,
        a7_v: V,
        one_v: V,
    }

    unsafe fn build_pow_prep(key_lanes: &[u64; 4], seed_lanes: &[u64; 4]) -> PowPrep {
        let (a0, a1, a2, a3) = (key_lanes[0], key_lanes[1], key_lanes[2], key_lanes[3]);
        let (a4, a5, a6, a7) = (seed_lanes[0], seed_lanes[1], seed_lanes[2], seed_lanes[3]);
        let a16 = 0x8000_0000_0000_0000u64;
        let c0 = a0 ^ a5;
        let c1 = a1 ^ a6 ^ a16;
        let c2 = a2 ^ a7;
        let c3 = a3;
        let c4 = a4 ^ 1;
        let d0 = c4 ^ c1.rotate_left(1);
        let d1 = c0 ^ c2.rotate_left(1);
        let d2_s = c1 ^ c3.rotate_left(1);
        let d3_s = c2 ^ c4.rotate_left(1);
        let d4_s = c3 ^ c0.rotate_left(1);
        let bs = [
            a0 ^ d0,
            (a6 ^ d1).rotate_left(44),
            0,
            d3_s.rotate_left(21),
            0,
            (a3 ^ d3_s).rotate_left(28),
            0,
            d0.rotate_left(3),
            (a16 ^ d1).rotate_left(45),
            0,
            (a1 ^ d1).rotate_left(1),
            0,
            d3_s.rotate_left(25),
            0,
            d0.rotate_left(18),
            0,
            (a5 ^ d0).rotate_left(36),
            d1.rotate_left(10),
            0,
            d3_s.rotate_left(56),
            0,
            0,
            0,
            d0.rotate_left(41),
            d1.rotate_left(2),
        ];
        unsafe {
            let mut bv = [vdupq_n_u64(0); LANES];
            for i in 0..LANES {
                bv[i] = vdupq_n_u64(bs[i]);
            }
            PowPrep {
                bs,
                bv,
                d2_s,
                d3_s,
                d4_s,
                a2_s: a2,
                a4_s: a4,
                a7_s: a7,
                d2_v: vdupq_n_u64(d2_s),
                d3_v: vdupq_n_u64(d3_s),
                d4_v: vdupq_n_u64(d4_s),
                a2_v: vdupq_n_u64(a2),
                a4_v: vdupq_n_u64(a4),
                a7_v: vdupq_n_u64(a7),
                one_v: vdupq_n_u64(1),
            }
        }
    }

    #[inline(always)]
    unsafe fn pow_r0_2x(p: &PowPrep, nv: V, zero: V, rc0: V) -> [V; LANES] {
        unsafe {
            let d2 = vrax1q_u64(p.d2_v, nv);
            let d4 = veorq_u64(p.d4_v, nv);
            let mut b = p.bv;
            b[2] = vxarq_u64::<21>(zero, d2);
            b[4] = vxarq_u64::<50>(zero, d4);
            b[6] = vxarq_u64::<44>(p.one_v, d4);
            b[9] = vxarq_u64::<3>(zero, d2);
            b[11] = vxarq_u64::<58>(p.a7_v, d2);
            b[13] = vxarq_u64::<56>(zero, d4);
            b[15] = vxarq_u64::<37>(p.a4_v, d4);
            b[18] = vxarq_u64::<49>(zero, d2);
            b[20] = vxarq_u64::<2>(p.a2_v, d2);
            b[21] = vxarq_u64::<9>(nv, p.d3_v);
            b[22] = vxarq_u64::<25>(zero, d4);
            let mut a = [zero; LANES];
            for y in 0..5 {
                for x in 0..5 {
                    a[5 * y + x] = vbcaxq_u64(b[5 * y + x], b[5 * y + (x + 2) % 5], b[5 * y + (x + 1) % 5]);
                }
            }
            a[0] = veorq_u64(a[0], rc0);
            a
        }
    }

    #[inline(always)]
    fn pow_r0_1x(p: &PowPrep, ns: u64) -> [u64; LANES] {
        let d2 = p.d2_s ^ ns.rotate_left(1);
        let d4 = p.d4_s ^ ns;
        let mut b = p.bs;
        b[2] = d2.rotate_left(43);
        b[4] = d4.rotate_left(14);
        b[6] = (1 ^ d4).rotate_left(20);
        b[9] = d2.rotate_left(61);
        b[11] = (p.a7_s ^ d2).rotate_left(6);
        b[13] = d4.rotate_left(8);
        b[15] = (p.a4_s ^ d4).rotate_left(27);
        b[18] = d2.rotate_left(15);
        b[20] = (p.a2_s ^ d2).rotate_left(62);
        b[21] = (ns ^ p.d3_s).rotate_left(55);
        b[22] = d4.rotate_left(39);
        let mut s = [0u64; LANES];
        for y in 0..5 {
            for x in 0..5 {
                s[5 * y + x] = b[5 * y + x] ^ (!b[5 * y + (x + 1) % 5] & b[5 * y + (x + 2) % 5]);
            }
        }
        s[0] ^= ROUND_CONSTANTS[0];
        s
    }

    pub fn find_pow_nonce_3x(
        key_lanes: &[u64; 4],
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
        let seed_lanes = lanes_from_32(seed);
        let mask_be = if bits == 64 { !0u64 } else { !0u64 << (64 - bits) };
        let mask_le = mask_be.swap_bytes();
        let triples = count / 3;
        unsafe {
            let prep = build_pow_prep(key_lanes, &seed_lanes);
            let zero = vdupq_n_u64(0);
            let rc0 = vdupq_n_u64(ROUND_CONSTANTS[0]);
            let mask_v = vdupq_n_u64(mask_le);
            let step_v = vdupq_n_u64(3);
            let mut nv = pair(start, start.wrapping_add(1));
            let mut ns = start.wrapping_add(2);
            for step in 0..triples {
                let mut a = pow_r0_2x(&prep, nv, zero, rc0);
                let mut s = pow_r0_1x(&prep, ns);
                for &rc in &ROUND_CONSTANTS[1..23] {
                    round(&mut a, rc);
                    round_scalar_inplace(&mut s, rc);
                }
                let a0 = vandq_u64(final_round_lane0_2x(&a), mask_v);
                let s0 = final_round_lane0_scalar(&s) & mask_le;
                if vgetq_lane_u64::<0>(a0) == 0 {
                    return Some(start + (step as u64) * 3);
                }
                if vgetq_lane_u64::<1>(a0) == 0 {
                    return Some(start + (step as u64) * 3 + 1);
                }
                if s0 == 0 {
                    return Some(start + (step as u64) * 3 + 2);
                }
                nv = vaddq_u64(nv, step_v);
                ns = ns.wrapping_add(3);
            }
        }
        for offset in (triples * 3)..count {
            let nonce = start + offset as u64;
            if super::pow_bits(key_lanes, seed, nonce) >= bits {
                return Some(nonce);
            }
        }
        None
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
pub fn keccak_f1600(state: &mut [u64; LANES]) {
    sha3_extension::keccak_f1600(state);
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little")))]
pub fn keccak_f1600(state: &mut [u64; LANES]) {
    keccak_f1600_portable(state);
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
pub(crate) fn hash_rows_3x_direct<F: crate::field::Field>(
    key_lanes: &[u64; 4],
    rows: &[Vec<F>],
    base: usize,
    out: &mut [[u8; OUT_LEN]; 3],
) {
    sha3_extension::hash_rows_3x_direct(key_lanes, rows, base, out);
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
pub(crate) fn hash_rows_2x_direct<F: crate::field::Field>(
    key_lanes: &[u64; 4],
    rows: &[Vec<F>],
    base: usize,
    out: &mut [[u8; OUT_LEN]; 2],
) {
    sha3_extension::hash_rows_2x_direct(key_lanes, rows, base, out);
}

#[inline(always)]
fn absorb_block(state: &mut [u64; LANES], block: &[u8; RATE_BYTES]) {
    const { assert!(RATE_LANES * 8 == RATE_BYTES) };
    for (lane, bytes) in state[..RATE_LANES].iter_mut().zip(block.as_chunks::<8>().0) {
        *lane ^= u64::from_le_bytes(*bytes);
    }
    keccak_f1600(state);
}

pub(crate) fn keccak256_with_key_lanes(key_lanes: &[u64; 4], mut input: &[u8]) -> [u8; OUT_LEN] {
    const FIRST_PAYLOAD: usize = RATE_BYTES - 32;
    const { assert!(FIRST_PAYLOAD == 104) };
    let mut state = [0u64; LANES];
    state[..4].copy_from_slice(key_lanes);
    if input.len() < FIRST_PAYLOAD {
        let mut tail = [0u8; FIRST_PAYLOAD];
        tail[..input.len()].copy_from_slice(input);
        tail[input.len()] ^= 0x01;
        tail[FIRST_PAYLOAD - 1] ^= 0x80;
        for (lane, bytes) in state[4..RATE_LANES].iter_mut().zip(tail.as_chunks::<8>().0) {
            *lane ^= u64::from_le_bytes(*bytes);
        }
        keccak_f1600(&mut state);
        return digest_from_state(&state);
    }
    let (head, rest) = input.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload fits");
    for (lane, bytes) in state[4..RATE_LANES].iter_mut().zip(head.as_chunks::<8>().0) {
        *lane ^= u64::from_le_bytes(*bytes);
    }
    keccak_f1600(&mut state);
    input = rest;
    while let Some((block, rest)) = input.split_first_chunk::<RATE_BYTES>() {
        absorb_block(&mut state, block);
        input = rest;
    }
    assert!(input.len() < RATE_BYTES);
    let mut block = [0u8; RATE_BYTES];
    block[..input.len()].copy_from_slice(input);
    block[input.len()] ^= 0x01;
    block[RATE_BYTES - 1] ^= 0x80;
    absorb_block(&mut state, &block);
    digest_from_state(&state)
}

pub(crate) fn keccak256_node_pair(key_lanes: &[u64; 4], left: &[u8; 32], right: &[u8; 32]) -> [u8; OUT_LEN] {
    const { assert!(RATE_BYTES == 136 && OUT_LEN == 32) };
    let mut state = [0u64; LANES];
    state[..4].copy_from_slice(key_lanes);
    state[4..8].copy_from_slice(&lanes_from_32(left));
    state[8..12].copy_from_slice(&lanes_from_32(right));
    state[12] = 0x01;
    state[16] = 0x8000_0000_0000_0000;
    keccak_f1600(&mut state);
    digest_from_state(&state)
}

pub(crate) fn keccak256_node_pair_3x(key_lanes: &[u64; 4], children: &[[u8; 32]; 6]) -> [[u8; OUT_LEN]; 3] {
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
    {
        sha3_extension::keccak256_node_pair_3x(key_lanes, children)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little")))]
    {
        [
            keccak256_node_pair(key_lanes, &children[0], &children[1]),
            keccak256_node_pair(key_lanes, &children[2], &children[3]),
            keccak256_node_pair(key_lanes, &children[4], &children[5]),
        ]
    }
}

pub(crate) fn keccak256_node_pair_2x(key_lanes: &[u64; 4], children: &[[u8; 32]; 4]) -> [[u8; OUT_LEN]; 2] {
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
    {
        sha3_extension::keccak256_node_pair_2x(key_lanes, children)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little")))]
    {
        [
            keccak256_node_pair(key_lanes, &children[0], &children[1]),
            keccak256_node_pair(key_lanes, &children[2], &children[3]),
        ]
    }
}

pub(crate) fn keccak256_domain_3x(
    key_lanes: &[u64; 4],
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
) -> [[u8; OUT_LEN]; 3] {
    assert!(in0.len() == in1.len() && in0.len() == in2.len());
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
    {
        sha3_extension::keccak256_domain_3x(key_lanes, in0, in1, in2)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little")))]
    {
        [
            keccak256_with_key_lanes(key_lanes, in0),
            keccak256_with_key_lanes(key_lanes, in1),
            keccak256_with_key_lanes(key_lanes, in2),
        ]
    }
}

pub(crate) fn keccak256_domain_2x(key_lanes: &[u64; 4], in0: &[u8], in1: &[u8]) -> [[u8; OUT_LEN]; 2] {
    assert!(in0.len() == in1.len());
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
    {
        sha3_extension::keccak256_domain_2x(key_lanes, in0, in1)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little")))]
    {
        [keccak256_with_key_lanes(key_lanes, in0), keccak256_with_key_lanes(key_lanes, in1)]
    }
}

pub(crate) fn hash_challenge_lanes(key_lanes: &[u64; 4], seed: &[u8; 32], counter: u64) -> [u8; OUT_LEN] {
    const { assert!(RATE_BYTES == 136 && OUT_LEN == 32) };
    let mut state = [0u64; LANES];
    state[..4].copy_from_slice(key_lanes);
    state[4..8].copy_from_slice(&lanes_from_32(seed));
    state[8] = counter;
    state[9] = 0x01;
    state[16] = 0x8000_0000_0000_0000;
    keccak_f1600(&mut state);
    digest_from_state(&state)
}

pub(crate) fn pow_bits(key_lanes: &[u64; 4], seed: &[u8; 32], nonce: u64) -> u32 {
    const { assert!(RATE_BYTES == 136 && OUT_LEN == 32) };
    let mut state = [0u64; LANES];
    state[..4].copy_from_slice(key_lanes);
    state[4..8].copy_from_slice(&lanes_from_32(seed));
    state[8] = nonce;
    state[9] = 0x01;
    state[16] = 0x8000_0000_0000_0000;
    keccak_f1600(&mut state);
    let bits = state[0].swap_bytes().leading_zeros();
    assert!(bits <= 64);
    bits
}

pub(crate) fn find_pow_nonce(
    key_lanes: &[u64; 4],
    seed: &[u8; 32],
    bits: u32,
    start: u64,
    count: usize,
) -> Option<u64> {
    assert!(bits <= 64);
    assert!(count == 0 || start <= u64::MAX - (count as u64 - 1));
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
    {
        sha3_extension::find_pow_nonce_3x(key_lanes, seed, bits, start, count)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little")))]
    {
        for offset in 0..count {
            let nonce = start + offset as u64;
            if pow_bits(key_lanes, seed, nonce) >= bits {
                return Some(nonce);
            }
        }
        None
    }
}

#[derive(Clone)]
pub struct Keccak256 {
    state: [u64; LANES],
    buffer: [u8; RATE_BYTES],
    buffer_len: usize,
}

impl Keccak256 {
    pub fn new() -> Self {
        Self { state: [0; LANES], buffer: [0; RATE_BYTES], buffer_len: 0 }
    }

    pub(crate) fn new_with_prefix(prefix: &[u8; 32]) -> Self {
        const { assert!(32 < RATE_BYTES) };
        let mut buffer = [0; RATE_BYTES];
        buffer[..32].copy_from_slice(prefix);
        Self { state: [0; LANES], buffer, buffer_len: 32 }
    }

    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        assert!(self.buffer_len < RATE_BYTES);
        if self.buffer_len > 0 {
            let take = (RATE_BYTES - self.buffer_len).min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&input[..take]);
            self.buffer_len += take;
            input = &input[take..];
            if self.buffer_len < RATE_BYTES {
                return self;
            }
            let block = self.buffer;
            absorb_block(&mut self.state, &block);
            self.buffer_len = 0;
        }
        while let Some((block, rest)) = input.split_first_chunk::<RATE_BYTES>() {
            absorb_block(&mut self.state, block);
            input = rest;
        }
        assert!(input.len() < RATE_BYTES);
        self.buffer[..input.len()].copy_from_slice(input);
        self.buffer_len = input.len();
        assert!(self.buffer_len < RATE_BYTES);
        self
    }

    pub fn finalize(&self) -> [u8; OUT_LEN] {
        assert!(self.buffer_len < RATE_BYTES);
        let mut state = self.state;
        let mut block = [0; RATE_BYTES];
        block[..self.buffer_len].copy_from_slice(&self.buffer[..self.buffer_len]);
        block[self.buffer_len] ^= 0x01;
        block[RATE_BYTES - 1] ^= 0x80;
        absorb_block(&mut state, &block);
        digest_from_state(&state)
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

pub fn keccak256(input: &[u8]) -> [u8; OUT_LEN] {
    let mut hasher = Keccak256::new();
    hasher.update(input);
    hasher.finalize()
}
