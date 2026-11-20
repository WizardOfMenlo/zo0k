#![allow(missing_docs)]

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

#[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
mod sha3_extension {
    use super::{LANES, ROUND_CONSTANTS};
    use std::arch::aarch64::*;

    type V = uint64x2_t;

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
                vxarq_u64::<0>(a[0], d[0]),
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

    pub fn keccak_f1600(state: &mut [u64; LANES]) {
        const { assert!(ROUND_CONSTANTS.len() == 24) };
        let mut lanes = unsafe { [vdupq_n_u64(0); LANES] };
        for (lane, &word) in lanes.iter_mut().zip(state.iter()) {
            *lane = unsafe { vdupq_n_u64(word) };
        }
        for round_constant in ROUND_CONSTANTS {
            round(&mut lanes, round_constant);
        }
        for (word, &lane) in state.iter_mut().zip(lanes.iter()) {
            *word = unsafe { vgetq_lane_u64::<0>(lane) };
        }
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

#[inline(always)]
fn absorb_block(state: &mut [u64; LANES], block: &[u8; RATE_BYTES]) {
    const { assert!(RATE_LANES * 8 == RATE_BYTES) };
    for (lane, bytes) in state[..RATE_LANES].iter_mut().zip(block.as_chunks::<8>().0) {
        *lane ^= u64::from_le_bytes(*bytes);
    }
    keccak_f1600(state);
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
        let mut digest = [0; OUT_LEN];
        for (bytes, lane) in digest.as_chunks_mut::<8>().0.iter_mut().zip(state) {
            *bytes = lane.to_le_bytes();
        }
        digest
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
