#[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
pub use neon::*;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
)))]
pub use portable::*;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub use sse::*;

#[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
mod neon {
    use std::arch::aarch64::*;

    const MODULUS_LOW: u64 = 0x87;

    pub type V = uint64x2_t;

    #[inline(always)]
    pub fn pack_v(mut v: V) -> V {
        unsafe {
            std::arch::asm!(
                "/* {0:v} */",
                inout(vreg) v,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        v
    }
    #[inline(always)]
    pub fn zero() -> V {
        unsafe { vdupq_n_u64(0) }
    }
    #[inline(always)]
    pub fn modulus() -> V {
        pack_v(unsafe { vdupq_n_u64(MODULUS_LOW) })
    }
    #[inline(always)]
    pub fn load(a: u128) -> V {
        unsafe { vreinterpretq_u64_p128(a) }
    }
    #[inline(always)]
    pub fn store(a: V) -> u128 {
        unsafe { vreinterpretq_p128_u64(a) }
    }
    #[inline(always)]
    pub fn xor(a: V, b: V) -> V {
        unsafe { veorq_u64(a, b) }
    }
    #[inline(always)]
    pub fn xor3(a: V, b: V, c: V) -> V {
        #[cfg(target_feature = "sha3")]
        unsafe {
            veor3q_u64(a, b, c)
        }
        #[cfg(not(target_feature = "sha3"))]
        unsafe {
            veorq_u64(veorq_u64(a, b), c)
        }
    }
    #[inline(always)]
    pub fn swap_lanes(a: V) -> V {
        unsafe { vextq_u64::<1>(a, a) }
    }
    #[inline(always)]
    pub fn zip_lo(a: V, b: V) -> V {
        unsafe { vzip1q_u64(a, b) }
    }
    #[inline(always)]
    pub fn zip_hi(a: V, b: V) -> V {
        unsafe { vzip2q_u64(a, b) }
    }
    #[inline(always)]
    pub fn mul_low(a: V, b: V) -> V {
        unsafe { vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b))) }
    }
    #[inline(always)]
    pub fn mul_high(a: V, b: V) -> V {
        unsafe { vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(b))) }
    }
    #[inline(always)]
    pub fn shift_up(a: V) -> V {
        unsafe { vextq_u64::<1>(vdupq_n_u64(0), a) }
    }
    #[inline(always)]
    pub fn shift_down(a: V) -> V {
        unsafe { vextq_u64::<1>(a, vdupq_n_u64(0)) }
    }
    #[inline(always)]
    pub fn split_x7(a: V) -> (V, V) {
        unsafe {
            let carried = vshlq_n_u64::<57>(a);
            (shift_up(carried), veorq_u64(vshrq_n_u64::<7>(a), shift_down(carried)))
        }
    }
    #[inline(always)]
    pub fn reduce(low: V, high: V) -> V {
        let m = modulus();
        let h1 = mul_high(high, m);
        let h0 = mul_low(xor(high, shift_down(h1)), m);
        xor3(low, h0, shift_up(h1))
    }
    #[inline(always)]
    pub fn reduce_karatsuba(low_raw: V, middle: V, high_raw: V) -> V {
        let m = modulus();
        let h1 = mul_high(high_raw, m);
        let mh = xor(middle, h1);
        let h0 = mul_low(xor(high_raw, shift_down(mh)), m);
        xor3(low_raw, h0, shift_up(mh))
    }
    #[inline(always)]
    pub fn mul_x121(a: V) -> V {
        let (low, high) = split_x7(a);
        let m = modulus();
        xor3(low, mul_low(high, m), shift_up(mul_high(high, m)))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
mod sse {
    use std::arch::x86_64::*;

    const MODULUS_LOW: i64 = 0x87;

    pub type V = __m128i;

    #[inline(always)]
    pub fn pack_v(v: V) -> V {
        v
    }
    #[inline(always)]
    pub fn zero() -> V {
        unsafe { _mm_setzero_si128() }
    }
    #[inline(always)]
    pub fn modulus() -> V {
        unsafe { _mm_set1_epi64x(MODULUS_LOW) }
    }
    #[inline(always)]
    pub fn load(a: u128) -> V {
        unsafe { std::mem::transmute(a) }
    }
    #[inline(always)]
    pub fn store(a: V) -> u128 {
        unsafe { std::mem::transmute(a) }
    }
    #[inline(always)]
    pub fn xor(a: V, b: V) -> V {
        unsafe { _mm_xor_si128(a, b) }
    }
    #[inline(always)]
    pub fn xor3(a: V, b: V, c: V) -> V {
        unsafe { _mm_xor_si128(_mm_xor_si128(a, b), c) }
    }
    #[inline(always)]
    pub fn swap_lanes(a: V) -> V {
        unsafe { _mm_shuffle_epi32::<0x4e>(a) }
    }
    #[inline(always)]
    pub fn zip_lo(a: V, b: V) -> V {
        unsafe { _mm_unpacklo_epi64(a, b) }
    }
    #[inline(always)]
    pub fn zip_hi(a: V, b: V) -> V {
        unsafe { _mm_unpackhi_epi64(a, b) }
    }
    #[inline(always)]
    pub fn mul_low(a: V, b: V) -> V {
        unsafe { _mm_clmulepi64_si128::<0x00>(a, b) }
    }
    #[inline(always)]
    pub fn mul_high(a: V, b: V) -> V {
        unsafe { _mm_clmulepi64_si128::<0x11>(a, b) }
    }
    #[inline(always)]
    pub fn shift_up(a: V) -> V {
        unsafe { _mm_slli_si128::<8>(a) }
    }
    #[inline(always)]
    pub fn shift_down(a: V) -> V {
        unsafe { _mm_srli_si128::<8>(a) }
    }
    #[inline(always)]
    pub fn split_x7(a: V) -> (V, V) {
        unsafe {
            let carried = _mm_slli_epi64::<57>(a);
            (shift_up(carried), _mm_xor_si128(_mm_srli_epi64::<7>(a), shift_down(carried)))
        }
    }
    #[inline(always)]
    pub fn reduce(low: V, high: V) -> V {
        let m = modulus();
        let h1 = mul_high(high, m);
        let h0 = mul_low(xor(high, shift_down(h1)), m);
        xor3(low, h0, shift_up(h1))
    }
    #[inline(always)]
    pub fn reduce_karatsuba(low_raw: V, middle: V, high_raw: V) -> V {
        let m = modulus();
        let h1 = mul_high(high_raw, m);
        let mh = xor(middle, h1);
        let h0 = mul_low(xor(high_raw, shift_down(mh)), m);
        xor3(low_raw, h0, shift_up(mh))
    }
    #[inline(always)]
    pub fn mul_x121(a: V) -> V {
        let (low, high) = split_x7(a);
        let m = modulus();
        xor3(low, mul_low(high, m), shift_up(mul_high(high, m)))
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
)))]
mod portable {
    const MODULUS_LOW: u128 = 0x87;

    pub type V = u128;

    fn clmul(a: u64, mut b: u64) -> u128 {
        let mut product = 0;
        let mut shifted = a as u128;
        for _ in 0..64 {
            if b == 0 {
                break;
            }
            if b & 1 == 1 {
                product ^= shifted;
            }
            shifted <<= 1;
            b >>= 1;
        }
        product
    }

    #[inline(always)]
    pub fn pack_v(v: V) -> V {
        v
    }
    #[inline(always)]
    pub fn zero() -> V {
        0
    }
    #[inline(always)]
    pub fn modulus() -> V {
        (MODULUS_LOW << 64) | MODULUS_LOW
    }
    #[inline(always)]
    pub fn load(a: u128) -> V {
        a
    }
    #[inline(always)]
    pub fn store(a: V) -> u128 {
        a
    }
    #[inline(always)]
    pub fn xor(a: V, b: V) -> V {
        a ^ b
    }
    #[inline(always)]
    pub fn xor3(a: V, b: V, c: V) -> V {
        a ^ b ^ c
    }
    #[inline(always)]
    pub fn swap_lanes(a: V) -> V {
        a.rotate_left(64)
    }
    #[inline(always)]
    pub fn zip_lo(a: V, b: V) -> V {
        (a as u64 as u128) | ((b as u64 as u128) << 64)
    }
    #[inline(always)]
    pub fn zip_hi(a: V, b: V) -> V {
        (a >> 64) | (b & !((u64::MAX) as u128))
    }
    #[inline(always)]
    pub fn mul_low(a: V, b: V) -> V {
        clmul(a as u64, b as u64)
    }
    #[inline(always)]
    pub fn mul_high(a: V, b: V) -> V {
        clmul((a >> 64) as u64, (b >> 64) as u64)
    }
    #[inline(always)]
    pub fn shift_up(a: V) -> V {
        a << 64
    }
    #[inline(always)]
    pub fn shift_down(a: V) -> V {
        a >> 64
    }
    #[inline(always)]
    pub fn split_x7(a: V) -> (V, V) {
        (a << 121, a >> 7)
    }
    #[inline(always)]
    pub fn reduce(low: V, high: V) -> V {
        let overflow = (high >> 121) ^ (high >> 126) ^ (high >> 127);
        let folded = high ^ overflow;
        low ^ folded ^ (folded << 1) ^ (folded << 2) ^ (folded << 7)
    }
    #[inline(always)]
    pub fn reduce_karatsuba(low_raw: V, middle: V, high_raw: V) -> V {
        reduce(xor(low_raw, shift_up(middle)), xor(high_raw, shift_down(middle)))
    }
    #[inline(always)]
    pub fn mul_x121(a: V) -> V {
        let high = a >> 7;
        (a << 121) ^ (high << 7) ^ (high << 2) ^ (high << 1) ^ high
    }
}

#[derive(Clone, Copy)]
pub struct Mul128 {
    b_lo: V,
    b_hi: V,
}

impl Mul128 {
    #[inline(always)]
    pub fn new(b: u128) -> Self {
        let b = load(b);
        let b_x64 = xor(shift_up(b), mul_high(b, modulus()));
        Self { b_lo: pack_v(zip_lo(b, b_x64)), b_hi: pack_v(zip_hi(b, b_x64)) }
    }

    #[inline(always)]
    pub fn from_pair(lo: u128, hi: u128) -> Self {
        Self { b_lo: load(lo), b_hi: load(hi) }
    }

    #[inline(always)]
    pub fn to_pair(self) -> (u128, u128) {
        (store(self.b_lo), store(self.b_hi))
    }

    #[inline(always)]
    pub fn xor(self, other: Self) -> Self {
        Self { b_lo: xor(self.b_lo, other.b_lo), b_hi: xor(self.b_hi, other.b_hi) }
    }

    #[inline(always)]
    pub fn xor3(self, a: Self, b: Self) -> Self {
        Self { b_lo: xor3(self.b_lo, a.b_lo, b.b_lo), b_hi: xor3(self.b_hi, a.b_hi, b.b_hi) }
    }

    #[inline(always)]
    pub fn mul_lm(self, a: V) -> (V, V) {
        (
            xor(mul_low(a, self.b_lo), mul_high(a, self.b_lo)),
            xor(mul_low(a, self.b_hi), mul_high(a, self.b_hi)),
        )
    }

    #[inline(always)]
    pub fn mul_v(self, a: V) -> V {
        let (l, m) = self.mul_lm(a);
        reduce_lm(l, m)
    }

    #[inline(always)]
    pub fn mul_xor_v(self, r: V, a: V) -> V {
        let l = xor3(r, mul_low(a, self.b_lo), mul_high(a, self.b_lo));
        let m = xor(mul_low(a, self.b_hi), mul_high(a, self.b_hi));
        reduce_lm(l, m)
    }

    #[inline(always)]
    pub fn mul(self, a: u128) -> u128 {
        store(self.mul_v(load(a)))
    }
}

#[inline(always)]
pub fn reduce_lm(l: V, m: V) -> V {
    xor3(l, shift_up(m), mul_high(m, modulus()))
}

#[derive(Clone, Copy)]
pub struct Mul256 {
    c: Mul128,
    dx: Mul128,
    cd: Mul128,
}

impl Mul256 {
    #[inline(always)]
    pub fn new(c: u128, d: u128) -> Self {
        let dx = store(mul_x121(load(d)));
        Self { c: Mul128::new(c), dx: Mul128::new(dx), cd: Mul128::new(c ^ d) }
    }

    #[inline(always)]
    pub fn mul_lm_pair(self, a: V, b: V) -> ((V, V), (V, V)) {
        let (l0, m0) = self.c.mul_lm(a);
        let (l1, m1) = self.dx.mul_lm(b);
        let (l2, m2) = self.cd.mul_lm(xor(a, b));
        ((xor(l0, l1), xor(m0, m1)), (xor(l0, l2), xor(m0, m2)))
    }

    #[inline(always)]
    pub fn mul_pair(self, a: V, b: V) -> (V, V) {
        let ((l_lo, m_lo), (l_hi, m_hi)) = self.mul_lm_pair(a, b);
        (reduce_lm(l_lo, m_lo), reduce_lm(l_hi, m_hi))
    }

    #[inline(always)]
    pub fn mul_xor_pair(self, r0: V, r1: V, a: V, b: V) -> (V, V) {
        let (l0, m0) = self.c.mul_lm(a);
        let (l1, m1) = self.dx.mul_lm(b);
        let (l2, m2) = self.cd.mul_lm(xor(a, b));
        let lo = reduce_lm(xor3(r0, l0, l1), xor(m0, m1));
        let hi = reduce_lm(xor3(r1, l0, l2), xor(m0, m2));
        (lo, hi)
    }
}

#[inline(always)]
pub fn butterfly_step_v(r: V, q: V, p: Mul128) -> (V, V) {
    let nr = p.mul_xor_v(r, q);
    (nr, xor(q, nr))
}

#[inline(always)]
pub fn transpose_butterfly_step_v(u: V, v: V, p: Mul128) -> (V, V) {
    let nu = xor(u, v);
    (nu, p.mul_xor_v(v, nu))
}

#[inline(always)]
pub fn radix8_8lanes_zero_prefix(v: &mut [V; 8], m2_1: Mul128, m1_1: Mul128, m1_2: Mul128, m1_3: Mul128) {
    for j in 0..4 {
        v[j + 4] = xor(v[j + 4], v[j]);
    }
    for j in 0..2 {
        v[j + 2] = xor(v[j + 2], v[j]);
        (v[j + 4], v[j + 6]) = butterfly_step_v(v[j + 4], v[j + 6], m2_1);
    }
    v[1] = xor(v[1], v[0]);
    (v[2], v[3]) = butterfly_step_v(v[2], v[3], m1_1);
    (v[4], v[5]) = butterfly_step_v(v[4], v[5], m1_2);
    (v[6], v[7]) = butterfly_step_v(v[6], v[7], m1_3);
}

#[inline(always)]
pub fn radix8_8lanes_partial(
    v: &mut [V; 8],
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
    octants: usize,
) {
    assert!((1..=8).contains(&octants));
    if octants <= 4 {
        for j in 0..4 {
            v[j] = m4.mul_xor_v(v[j], v[j + 4]);
        }
    } else {
        for j in 0..4 {
            (v[j], v[j + 4]) = butterfly_step_v(v[j], v[j + 4], m4);
        }
    }
    if octants <= 2 {
        for j in 0..2 {
            v[j] = m2_0.mul_xor_v(v[j], v[j + 2]);
        }
    } else {
        for j in 0..2 {
            (v[j], v[j + 2]) = butterfly_step_v(v[j], v[j + 2], m2_0);
        }
    }
    if octants > 4 {
        let m2_1 = m2_0.xor(p2);
        if octants <= 6 {
            for j in 0..2 {
                v[j + 4] = m2_1.mul_xor_v(v[j + 4], v[j + 6]);
            }
        } else {
            for j in 0..2 {
                (v[j + 4], v[j + 6]) = butterfly_step_v(v[j + 4], v[j + 6], m2_1);
            }
        }
    }
    let m1 = [m1_0, m1_0.xor(p2), m1_0.xor(p4), m1_0.xor3(p2, p4)];
    let pairs = octants.div_ceil(2);
    assert!((1..=4).contains(&pairs));
    for j in 0..pairs {
        if 2 * j + 1 < octants {
            (v[2 * j], v[2 * j + 1]) = butterfly_step_v(v[2 * j], v[2 * j + 1], m1[j]);
        } else {
            v[2 * j] = m1[j].mul_xor_v(v[2 * j], v[2 * j + 1]);
        }
    }
}

#[inline(always)]
pub fn radix8_8lanes_twiddles(v: &mut [V; 8], m4: Mul128, m2: [Mul128; 2], m1: [Mul128; 4]) {
    for j in 0..4 {
        (v[j], v[j + 4]) = butterfly_step_v(v[j], v[j + 4], m4);
    }
    for j in 0..2 {
        (v[j], v[j + 2]) = butterfly_step_v(v[j], v[j + 2], m2[0]);
        (v[j + 4], v[j + 6]) = butterfly_step_v(v[j + 4], v[j + 6], m2[1]);
    }
    for j in 0..4 {
        (v[2 * j], v[2 * j + 1]) = butterfly_step_v(v[2 * j], v[2 * j + 1], m1[j]);
    }
}

#[inline(always)]
pub fn transpose_radix8_8lanes_twiddles(v: &mut [V; 8], m4: Mul128, m2: [Mul128; 2], m1: [Mul128; 4]) {
    for j in 0..4 {
        (v[2 * j], v[2 * j + 1]) = transpose_butterfly_step_v(v[2 * j], v[2 * j + 1], m1[j]);
    }
    for j in 0..2 {
        (v[j], v[j + 2]) = transpose_butterfly_step_v(v[j], v[j + 2], m2[0]);
        (v[j + 4], v[j + 6]) = transpose_butterfly_step_v(v[j + 4], v[j + 6], m2[1]);
    }
    for j in 0..4 {
        (v[j], v[j + 4]) = transpose_butterfly_step_v(v[j], v[j + 4], m4);
    }
}

#[inline(never)]
pub fn radix8_rows_compact(
    rows: &mut [[V; 8]],
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
) {
    let len = rows.len();
    assert!(len >= 1);
    let m2_1 = m2_0.xor(p2);
    let m1_1 = m1_0.xor(p2);
    let m1_2 = m1_0.xor(p4);
    let m1_3 = m1_1.xor(p4);
    let m2 = [m2_0, m2_1];
    let m1 = [m1_0, m1_1, m1_2, m1_3];
    for row in rows.iter_mut() {
        radix8_8lanes_twiddles(row, m4, m2, m1);
    }
    assert_eq!(rows.len(), len);
}

#[inline(never)]
pub fn radix8_rows_partial(
    rows: &mut [[V; 8]],
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
    octants: usize,
) {
    let len = rows.len();
    assert!(len >= 1 && (1..=8).contains(&octants));
    for row in rows.iter_mut() {
        radix8_8lanes_partial(row, m4, m2_0, m1_0, p2, p4, octants);
    }
    assert_eq!(rows.len(), len);
}

#[inline(always)]
pub fn radix8_8lanes_compact(v: &mut [V; 8], m4: Mul128, m2_0: Mul128, m1_0: Mul128, p2: Mul128, p4: Mul128) {
    for j in 0..4 {
        (v[j], v[j + 4]) = butterfly_step_v(v[j], v[j + 4], m4);
    }
    let m2_1 = m2_0.xor(p2);
    for j in 0..2 {
        (v[j], v[j + 2]) = butterfly_step_v(v[j], v[j + 2], m2_0);
        (v[j + 4], v[j + 6]) = butterfly_step_v(v[j + 4], v[j + 6], m2_1);
    }
    (v[0], v[1]) = butterfly_step_v(v[0], v[1], m1_0);
    let m1_1 = m1_0.xor(p2);
    (v[2], v[3]) = butterfly_step_v(v[2], v[3], m1_1);
    let m1_3 = m1_1.xor(p4);
    (v[6], v[7]) = butterfly_step_v(v[6], v[7], m1_3);
    let m1_2 = m1_0.xor(p4);
    (v[4], v[5]) = butterfly_step_v(v[4], v[5], m1_2);
}

#[inline(always)]
pub fn transpose_radix8_8lanes_compact(
    v: &mut [V; 8],
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
) {
    (v[0], v[1]) = transpose_butterfly_step_v(v[0], v[1], m1_0);
    let m1_1 = m1_0.xor(p2);
    (v[2], v[3]) = transpose_butterfly_step_v(v[2], v[3], m1_1);
    let m1_3 = m1_1.xor(p4);
    (v[6], v[7]) = transpose_butterfly_step_v(v[6], v[7], m1_3);
    let m1_2 = m1_0.xor(p4);
    (v[4], v[5]) = transpose_butterfly_step_v(v[4], v[5], m1_2);
    let m2_1 = m2_0.xor(p2);
    for j in 0..2 {
        (v[j], v[j + 2]) = transpose_butterfly_step_v(v[j], v[j + 2], m2_0);
        (v[j + 4], v[j + 6]) = transpose_butterfly_step_v(v[j + 4], v[j + 6], m2_1);
    }
    for j in 0..4 {
        (v[j], v[j + 4]) = transpose_butterfly_step_v(v[j], v[j + 4], m4);
    }
}

#[derive(Clone, Copy)]
pub struct Acc128 {
    lo: V,
    mid: V,
    hi: V,
}

impl Acc128 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self { lo: zero(), mid: zero(), hi: zero() }
    }

    #[inline(always)]
    pub fn add_product_pre(&mut self, a: V, b: V, b_aux: V) {
        self.lo = xor(self.lo, mul_low(a, b));
        self.hi = xor(self.hi, mul_high(a, b));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            self.mid = xor3(self.mid, mul_low(a, b_aux), mul_high(a, b_aux));
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        )))]
        {
            self.mid = xor(self.mid, mul_low(xor(a, swap_lanes(a)), b_aux));
        }
    }

    #[inline(always)]
    pub fn precompute_aux(b: V) -> V {
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            swap_lanes(b)
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        )))]
        {
            xor(b, swap_lanes(b))
        }
    }

    #[inline(always)]
    pub fn add_product(&mut self, a: V, b: V) {
        self.add_product_pre(a, b, Self::precompute_aux(b));
    }

    #[inline(always)]
    pub fn merge(self, other: Self) -> Self {
        Self { lo: xor(self.lo, other.lo), mid: xor(self.mid, other.mid), hi: xor(self.hi, other.hi) }
    }

    #[inline(always)]
    fn resolved_mid(self) -> V {
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        ))]
        {
            self.mid
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq")
        )))]
        {
            xor3(self.mid, self.lo, self.hi)
        }
    }

    #[inline(always)]
    pub fn to_wide(self) -> (V, V) {
        let mid = self.resolved_mid();
        (xor(self.lo, shift_up(mid)), xor(self.hi, shift_down(mid)))
    }

    #[inline(always)]
    pub fn reduce(self) -> V {
        reduce_karatsuba(self.lo, self.resolved_mid(), self.hi)
    }
}

#[inline(always)]
pub fn mul_wide(a: V, b: V) -> (V, V) {
    let low = mul_low(a, b);
    let high = mul_high(a, b);
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    let middle = {
        let bs = swap_lanes(b);
        xor(mul_low(a, bs), mul_high(a, bs))
    };
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    )))]
    let middle = xor3(mul_low(xor(a, swap_lanes(a)), xor(b, swap_lanes(b))), low, high);
    (xor(low, shift_up(middle)), xor(high, shift_down(middle)))
}

#[inline(always)]
pub fn reduce_256(ac: (V, V), bd: (V, V), cross: (V, V)) -> (V, V) {
    let (bd0_lo, bd0_hi) = split_x7(bd.0);
    let low = reduce(xor(ac.0, bd0_lo), xor3(ac.1, bd0_hi, mul_x121(bd.1)));
    let high = reduce(xor(ac.0, cross.0), xor(ac.1, cross.1));
    (low, high)
}

#[inline(always)]
pub fn mul(a: u128, b: u128) -> u128 {
    let (a, b) = (load(a), load(b));
    let low_raw = mul_low(a, b);
    let high_raw = mul_high(a, b);
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    let middle = {
        let bs = swap_lanes(b);
        xor(mul_low(a, bs), mul_high(a, bs))
    };
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    )))]
    let middle = xor3(mul_low(xor(a, swap_lanes(a)), xor(b, swap_lanes(b))), low_raw, high_raw);
    store(reduce_karatsuba(low_raw, middle, high_raw))
}

#[inline(always)]
pub fn square(a: u128) -> u128 {
    let a = load(a);
    store(reduce(mul_low(a, a), mul_high(a, a)))
}
