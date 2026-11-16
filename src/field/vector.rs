//! Arithmetic in `K = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)` on 128-bit
//! vectors, with one implementation per instruction set. A vector holds a
//! polynomial of degree below 128 as two 64-bit halves.

#[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
pub use neon::*;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq")
)))]
pub use portable::*;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub use sse::*;

/// The low part `x^7 + x^2 + x + 1` of the modulus of `K`.
const MODULUS_LOW: u64 = 0x87;

// Every backend defines the type `V` and the following functions:
//
// load(u128) -> V, store(V) -> u128, xor(V, V) -> V
// mul_low(V, V) -> V       product of the low halves
// mul_high(V, V) -> V      product of the high halves
// mul_sums(V, V) -> V      product of the sums of the two halves
// shift_up(V) -> V         multiply by x^64, dropping the overflow
// shift_down(V) -> V       divide by x^64
// split_x7(V) -> (V, V)    for v, the pair (v mod x^7)·x^121, v div x^7
// modulus() -> V           MODULUS_LOW in both halves

#[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
mod neon {
    use std::arch::aarch64::*;

    // SAFETY (whole module): the intrinsics need the `neon` and `aes`
    // target features, which are enabled for this build.

    pub type V = uint64x2_t;

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
    pub fn mul_low(a: V, b: V) -> V {
        unsafe { vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b))) }
    }
    #[inline(always)]
    pub fn mul_high(a: V, b: V) -> V {
        unsafe { vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(b))) }
    }
    #[inline(always)]
    pub fn mul_sums(a: V, b: V) -> V {
        unsafe { mul_low(veorq_u64(a, vextq_u64::<1>(a, a)), veorq_u64(b, vextq_u64::<1>(b, b))) }
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
    pub fn modulus() -> V {
        unsafe { vdupq_n_u64(super::MODULUS_LOW) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
mod sse {
    use std::arch::x86_64::*;

    // SAFETY (whole module): the intrinsics need the `sse2` and
    // `pclmulqdq` target features, which are enabled for this build, and
    // a 128-bit vector has the layout of `u128`.

    pub type V = __m128i;

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
    pub fn mul_low(a: V, b: V) -> V {
        unsafe { _mm_clmulepi64_si128::<0x00>(a, b) }
    }
    #[inline(always)]
    pub fn mul_high(a: V, b: V) -> V {
        unsafe { _mm_clmulepi64_si128::<0x11>(a, b) }
    }
    #[inline(always)]
    pub fn mul_sums(a: V, b: V) -> V {
        unsafe {
            mul_low(
                _mm_xor_si128(a, _mm_shuffle_epi32::<0x4e>(a)),
                _mm_xor_si128(b, _mm_shuffle_epi32::<0x4e>(b)),
            )
        }
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
    pub fn modulus() -> V {
        unsafe { _mm_set1_epi64x(super::MODULUS_LOW as i64) }
    }
}

#[cfg_attr(
    any(
        all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ),
    allow(dead_code)
)]
pub mod portable {
    pub type V = u128;

    fn clmul(a: u64, mut b: u64) -> u128 {
        let mut product = 0;
        let mut shifted = a as u128;
        while b != 0 {
            if b & 1 == 1 {
                product ^= shifted;
            }
            shifted <<= 1;
            b >>= 1;
        }
        product
    }

    pub fn load(a: u128) -> V {
        a
    }
    pub fn store(a: V) -> u128 {
        a
    }
    pub fn xor(a: V, b: V) -> V {
        a ^ b
    }
    pub fn mul_low(a: V, b: V) -> V {
        clmul(a as u64, b as u64)
    }
    pub fn mul_high(a: V, b: V) -> V {
        clmul((a >> 64) as u64, (b >> 64) as u64)
    }
    pub fn mul_sums(a: V, b: V) -> V {
        clmul((a ^ (a >> 64)) as u64, (b ^ (b >> 64)) as u64)
    }
    pub fn shift_up(a: V) -> V {
        a << 64
    }
    pub fn shift_down(a: V) -> V {
        a >> 64
    }
    pub fn split_x7(a: V) -> (V, V) {
        (a << 121, a >> 7)
    }
    pub fn modulus() -> V {
        (super::MODULUS_LOW as u128) << 64 | super::MODULUS_LOW as u128
    }
}

/// The 255-bit product of two polynomials of degree below 128, as (low, high).
#[inline(always)]
pub fn mul_wide(a: V, b: V) -> (V, V) {
    let low = mul_low(a, b);
    let high = mul_high(a, b);
    let middle = xor(mul_sums(a, b), xor(low, high));
    (xor(low, shift_up(middle)), xor(high, shift_down(middle)))
}

/// Reduce `low + high·x^128` modulo `x^128 + x^7 + x^2 + x + 1`.
#[inline(always)]
pub fn reduce(low: V, high: V) -> V {
    // x^128 = x^7 + x^2 + x + 1 =: m, so the result is low + high·m. With
    // high = h_0 + h_1·x^64, the product h_1·m·x^64 passes degree 127 by at
    // most seven bits c, and c·x^128 = c·m.
    let m = modulus();
    let h0 = mul_low(high, m);
    let h1 = mul_high(high, m);
    let carry = mul_high(h1, m);
    xor(xor(low, h0), xor(shift_up(h1), carry))
}

/// Product with `x^121` in `K`.
#[inline(always)]
pub fn mul_x121(a: V) -> V {
    // a·x^121 = (a mod x^7)·x^121 + (a div x^7)·x^128, and the reduction of
    // the second term does not carry, as its factor has degree below 121.
    let (low, high) = split_x7(a);
    let m = modulus();
    xor(low, xor(mul_low(high, m), shift_up(mul_high(high, m))))
}

/// Product in `K`.
#[inline(always)]
pub fn mul(a: u128, b: u128) -> u128 {
    let (low, high) = mul_wide(load(a), load(b));
    store(reduce(low, high))
}

/// Square in `K`.
#[inline(always)]
pub fn square(a: u128) -> u128 {
    // Squaring spreads the bits of each half of a polynomial.
    let a = load(a);
    store(reduce(mul_low(a, a), mul_high(a, a)))
}
