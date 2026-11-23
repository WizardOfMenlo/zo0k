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

const RHO: [u32; 24] =
    [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];

const PI: [usize; 24] =
    [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

#[inline(always)]
fn theta(state: &mut [u64; LANES]) {
    let mut parity = [0u64; 5];
    for x in 0..5 {
        parity[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    for x in 0..5 {
        let d = parity[(x + 4) % 5] ^ parity[(x + 1) % 5].rotate_left(1);
        for y in 0..5 {
            state[5 * y + x] ^= d;
        }
    }
}

#[inline(always)]
fn rho_pi(state: &mut [u64; LANES]) {
    let mut carried = state[1];
    for step in 0..24 {
        let lane = PI[step];
        let next = state[lane];
        state[lane] = carried.rotate_left(RHO[step]);
        carried = next;
    }
}

#[inline(always)]
fn chi(state: &mut [u64; LANES]) {
    for y in 0..5 {
        let row = [state[5 * y], state[5 * y + 1], state[5 * y + 2], state[5 * y + 3], state[5 * y + 4]];
        for x in 0..5 {
            state[5 * y + x] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
        }
    }
}

pub fn keccak_f1600_portable(state: &mut [u64; LANES]) {
    const { assert!(ROUND_CONSTANTS.len() == ROUNDS) };
    for round_constant in ROUND_CONSTANTS {
        theta(state);
        rho_pi(state);
        chi(state);
        state[0] ^= round_constant;
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
fn digest_from_state(state: &[u64; LANES]) -> [u8; OUT_LEN] {
    let mut digest = [0u8; OUT_LEN];
    for (bytes, &lane) in digest.as_chunks_mut::<8>().0.iter_mut().zip(&state[..4]) {
        *bytes = lane.to_le_bytes();
    }
    digest
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
mod sha3_extension {
    use super::{LANES, OUT_LEN, RATE_BYTES, RATE_LANES, ROUND_CONSTANTS, lanes_from_32};
    use std::arch::aarch64::*;

    type V = uint64x2_t;

    #[inline(always)]
    unsafe fn pair(x: u64, y: u64) -> V {
        unsafe { vsetq_lane_u64::<1>(y, vdupq_n_u64(x)) }
    }

    #[inline(always)]
    fn round(a: &mut [V; LANES], round_constant: u64) {
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
    fn permute_lanes(lanes: &mut [V; LANES]) {
        const { assert!(ROUND_CONSTANTS.len() == 24) };
        for round_constant in ROUND_CONSTANTS {
            round(lanes, round_constant);
        }
    }

    #[inline(always)]
    fn final_round_lane0(a: &[V; LANES], round_constant: u64) -> V {
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
            veorq_u64(vbcaxq_u64(b0, b2, b1), vdupq_n_u64(round_constant))
        }
    }

    #[inline(always)]
    unsafe fn extract_2_digests(lanes: &[V; LANES]) -> [[u8; OUT_LEN]; 2] {
        let mut out = [[0u8; OUT_LEN]; 2];
        unsafe {
            for (word, &lane) in lanes[..4].iter().enumerate() {
                let off = word * 8;
                out[0][off..off + 8].copy_from_slice(&vgetq_lane_u64::<0>(lane).to_le_bytes());
                out[1][off..off + 8].copy_from_slice(&vgetq_lane_u64::<1>(lane).to_le_bytes());
            }
        }
        out
    }

    pub fn keccak_f1600(state: &mut [u64; LANES]) {
        const { assert!(ROUND_CONSTANTS.len() == 24) };
        let mut lanes = unsafe { [vdupq_n_u64(0); LANES] };
        for (lane, &word) in lanes.iter_mut().zip(state.iter()) {
            *lane = unsafe { vdupq_n_u64(word) };
        }
        permute_lanes(&mut lanes);
        for (word, &lane) in state.iter_mut().zip(lanes.iter()) {
            *word = unsafe { vgetq_lane_u64::<0>(lane) };
        }
    }

    #[inline(always)]
    unsafe fn load_pair2(p0: *const u8, p1: *const u8) -> (V, V) {
        unsafe {
            let a = vreinterpretq_u64_u8(vld1q_u8(p0));
            let b = vreinterpretq_u64_u8(vld1q_u8(p1));
            (vzip1q_u64(a, b), vzip2q_u64(a, b))
        }
    }

    pub fn keccak256_node_pair_2x(key_lanes: &[u64; 4], children: &[[u8; 32]; 4]) -> [[u8; OUT_LEN]; 2] {
        const { assert!(RATE_BYTES == 136 && OUT_LEN == 32) };
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut lanes = [zero; LANES];
            for i in 0..4 {
                lanes[i] = vdupq_n_u64(key_lanes[i]);
            }
            let p0 = children[0].as_ptr();
            let p1 = children[2].as_ptr();
            (lanes[4], lanes[5]) = load_pair2(p0, p1);
            (lanes[6], lanes[7]) = load_pair2(p0.add(16), p1.add(16));
            let q0 = children[1].as_ptr();
            let q1 = children[3].as_ptr();
            (lanes[8], lanes[9]) = load_pair2(q0, q1);
            (lanes[10], lanes[11]) = load_pair2(q0.add(16), q1.add(16));
            lanes[12] = vdupq_n_u64(0x01);
            lanes[16] = vdupq_n_u64(0x8000_0000_0000_0000);
            permute_lanes(&mut lanes);
            extract_2_digests(&lanes)
        }
    }

    #[inline(always)]
    unsafe fn absorb_lane_2x(lanes: &mut [V; LANES], pos: &mut usize, val: V) {
        assert!(*pos < RATE_LANES);
        unsafe {
            lanes[*pos] = veorq_u64(lanes[*pos], val);
        }
        *pos += 1;
        if *pos == RATE_LANES {
            permute_lanes(lanes);
            *pos = 0;
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
        unsafe {
            let mut lanes = [vdupq_n_u64(0); LANES];
            for i in 0..4 {
                lanes[i] = vdupq_n_u64(key_lanes[i]);
            }
            lanes[4] = pair(base as u64, (base + 1) as u64);
            let mut pos = 5usize;
            if F::BYTES == 32 {
                let mut raw0 = [0u8; 32];
                let mut raw1 = [0u8; 32];
                for row in rows {
                    row[base].write_bytes(&mut raw0);
                    row[base + 1].write_bytes(&mut raw1);
                    let (v0, v1) = load_pair2(raw0.as_ptr(), raw1.as_ptr());
                    let (v2, v3) = load_pair2(raw0.as_ptr().add(16), raw1.as_ptr().add(16));
                    absorb_lane_2x(&mut lanes, &mut pos, v0);
                    absorb_lane_2x(&mut lanes, &mut pos, v1);
                    absorb_lane_2x(&mut lanes, &mut pos, v2);
                    absorb_lane_2x(&mut lanes, &mut pos, v3);
                }
            } else {
                let mut raw0 = [0u8; 16];
                let mut raw1 = [0u8; 16];
                for row in rows {
                    row[base].write_bytes(&mut raw0);
                    row[base + 1].write_bytes(&mut raw1);
                    let (v0, v1) = load_pair2(raw0.as_ptr(), raw1.as_ptr());
                    absorb_lane_2x(&mut lanes, &mut pos, v0);
                    absorb_lane_2x(&mut lanes, &mut pos, v1);
                }
            }
            assert!(pos < RATE_LANES);
            lanes[pos] = veorq_u64(lanes[pos], vdupq_n_u64(0x01));
            lanes[RATE_LANES - 1] = veorq_u64(lanes[RATE_LANES - 1], vdupq_n_u64(0x8000_0000_0000_0000));
            permute_lanes(&mut lanes);
            *out = extract_2_digests(&lanes);
        }
    }

    #[inline(always)]
    unsafe fn xor_full_block_2x(lanes: &mut [V; LANES], b0: &[u8; RATE_BYTES], b1: &[u8; RATE_BYTES]) {
        unsafe {
            let p0 = b0.as_ptr();
            let p1 = b1.as_ptr();
            for pair_idx in 0..8 {
                let off = pair_idx * 16;
                let (v0, v1) = load_pair2(p0.add(off), p1.add(off));
                lanes[2 * pair_idx] = veorq_u64(lanes[2 * pair_idx], v0);
                lanes[2 * pair_idx + 1] = veorq_u64(lanes[2 * pair_idx + 1], v1);
            }
            let w0 = u64::from_le_bytes(b0[128..136].try_into().expect("last lane"));
            let w1 = u64::from_le_bytes(b1[128..136].try_into().expect("last lane"));
            lanes[16] = veorq_u64(lanes[16], pair(w0, w1));
        }
    }

    #[inline(always)]
    unsafe fn load_first_payload_2x(lanes: &mut [V; LANES], b0: &[u8; 104], b1: &[u8; 104]) {
        unsafe {
            let p0 = b0.as_ptr();
            let p1 = b1.as_ptr();
            for pair_idx in 0..6 {
                let off = pair_idx * 16;
                (lanes[4 + 2 * pair_idx], lanes[5 + 2 * pair_idx]) = load_pair2(p0.add(off), p1.add(off));
            }
            let w0 = u64::from_le_bytes(b0[96..104].try_into().expect("last lane"));
            let w1 = u64::from_le_bytes(b1[96..104].try_into().expect("last lane"));
            lanes[16] = pair(w0, w1);
        }
    }

    pub fn keccak256_domain_2x(key_lanes: &[u64; 4], in0: &[u8], in1: &[u8]) -> [[u8; OUT_LEN]; 2] {
        assert!(in0.len() == in1.len());
        const FIRST_PAYLOAD: usize = RATE_BYTES - 32;
        const { assert!(FIRST_PAYLOAD == 104) };
        unsafe {
            let mut lanes = [vdupq_n_u64(0); LANES];
            for i in 0..4 {
                lanes[i] = vdupq_n_u64(key_lanes[i]);
            }
            let (mut r0, mut r1) = (in0, in1);
            if r0.len() < FIRST_PAYLOAD {
                let mut p0 = [0u8; FIRST_PAYLOAD];
                let mut p1 = [0u8; FIRST_PAYLOAD];
                p0[..r0.len()].copy_from_slice(r0);
                p1[..r1.len()].copy_from_slice(r1);
                p0[r0.len()] ^= 0x01;
                p1[r1.len()] ^= 0x01;
                p0[FIRST_PAYLOAD - 1] ^= 0x80;
                p1[FIRST_PAYLOAD - 1] ^= 0x80;
                load_first_payload_2x(&mut lanes, &p0, &p1);
                permute_lanes(&mut lanes);
                return extract_2_digests(&lanes);
            }
            let (head0, tail0) = r0.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload fits");
            let (head1, tail1) = r1.split_first_chunk::<FIRST_PAYLOAD>().expect("first payload fits");
            load_first_payload_2x(&mut lanes, head0, head1);
            permute_lanes(&mut lanes);
            r0 = tail0;
            r1 = tail1;
            while let (Some((b0, t0)), Some((b1, t1))) =
                (r0.split_first_chunk::<RATE_BYTES>(), r1.split_first_chunk::<RATE_BYTES>())
            {
                xor_full_block_2x(&mut lanes, b0, b1);
                permute_lanes(&mut lanes);
                r0 = t0;
                r1 = t1;
            }
            assert!(r0.len() < RATE_BYTES && r0.len() == r1.len());
            let mut b0 = [0u8; RATE_BYTES];
            let mut b1 = [0u8; RATE_BYTES];
            b0[..r0.len()].copy_from_slice(r0);
            b1[..r1.len()].copy_from_slice(r1);
            b0[r0.len()] ^= 0x01;
            b1[r1.len()] ^= 0x01;
            b0[RATE_BYTES - 1] ^= 0x80;
            b1[RATE_BYTES - 1] ^= 0x80;
            xor_full_block_2x(&mut lanes, &b0, &b1);
            permute_lanes(&mut lanes);
            extract_2_digests(&lanes)
        }
    }

    pub fn find_pow_nonce_2x(
        key_lanes: &[u64; 4],
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
        let seed_lanes = lanes_from_32(seed);
        unsafe {
            let zero = vdupq_n_u64(0);
            let mut template = [zero; LANES];
            for i in 0..4 {
                template[i] = vdupq_n_u64(key_lanes[i]);
                template[4 + i] = vdupq_n_u64(seed_lanes[i]);
            }
            template[9] = vdupq_n_u64(0x01);
            template[16] = vdupq_n_u64(0x8000_0000_0000_0000);
            let pairs = count / 2;
            for step in 0..pairs {
                let nonce = start + (step as u64) * 2;
                let mut lanes = template;
                lanes[8] = pair(nonce, nonce + 1);
                for &rc in &ROUND_CONSTANTS[..23] {
                    round(&mut lanes, rc);
                }
                let a0 = final_round_lane0(&lanes, ROUND_CONSTANTS[23]);
                if vgetq_lane_u64::<0>(a0).swap_bytes().leading_zeros() >= bits {
                    return Some(nonce);
                }
                if vgetq_lane_u64::<1>(a0).swap_bytes().leading_zeros() >= bits {
                    return Some(nonce + 1);
                }
            }
        }
        if count % 2 == 1 {
            let nonce = start + (count as u64 - 1);
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
        sha3_extension::find_pow_nonce_2x(key_lanes, seed, bits, start, count)
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
