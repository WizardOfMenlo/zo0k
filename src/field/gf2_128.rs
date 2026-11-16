//! The binary field `GF(2^128)`.

use super::{BinaryField, Field, binary_inverse, impl_binary_field_operators, vector};
use std::fmt;
use std::ops::{Add, Mul};

/// An element of `K = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`: a polynomial
/// in `x` of degree below 128, stored in one limb with the constant term in
/// the lowest bit.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gf2_128(u128);

impl Gf2_128 {
    /// The element with the given limb.
    pub const fn new(limb: u128) -> Self {
        Self(limb)
    }

    /// The limb of the element.
    pub const fn limb(self) -> u128 {
        self.0
    }
}

impl Field for Gf2_128 {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const BITS: u32 = 128;
    const BYTES: usize = 16;
    const NAME: &'static str = "gf2_128";

    fn from_integer(n: usize) -> Self {
        Self((n & 1) as u128)
    }

    fn inverse(self) -> Self {
        binary_inverse(self)
    }

    fn write_bytes(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.0.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Some(Self(u128::from_le_bytes(bytes.try_into().ok()?)))
    }

    fn sample(mut fill: impl FnMut(&mut [u8])) -> Self {
        let mut bytes = [0; 16];
        fill(&mut bytes);
        Self(u128::from_le_bytes(bytes))
    }

    #[inline(always)]
    fn square(self) -> Self {
        Self(vector::square(self.0))
    }
}

impl BinaryField for Gf2_128 {
    fn from_base(limb: u128) -> Self {
        Self(limb)
    }

    #[inline(always)]
    fn mul_base(self, w: u128) -> Self {
        Self(vector::mul(self.0, w))
    }
}

impl fmt::Debug for Gf2_128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gf2_128({:#x})", self.0)
    }
}

impl Add for Gf2_128 {
    type Output = Self;
    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Mul for Gf2_128 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(vector::mul(self.0, rhs.0))
    }
}

impl_binary_field_operators!(Gf2_128);

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Bit-by-bit product in `K`, independent of the vector arithmetic.
    pub fn mul_reference(mut a: u128, mut b: u128) -> u128 {
        let mut product = 0;
        for _ in 0..128 {
            if b & 1 == 1 {
                product ^= a;
            }
            b >>= 1;
            let carry = a >> 127;
            a <<= 1;
            if carry == 1 {
                a ^= 0b1000_0111;
            }
        }
        product
    }

    #[test]
    fn vector_operations_match_the_portable_ones() {
        use vector::portable;
        let mut rng = Rng::from_seed([1; 32]);
        let mut cases: Vec<(u128, u128)> =
            (0..500).map(|_| (rng.field::<Gf2_128>().0, rng.field::<Gf2_128>().0)).collect();
        cases.extend([(u128::MAX, u128::MAX), (0, u128::MAX), (1 << 127, 1 << 64)]);
        for (a, b) in cases {
            let (va, vb) = (vector::load(a), vector::load(b));
            assert_eq!(vector::store(vector::mul_low(va, vb)), portable::mul_low(a, b));
            assert_eq!(vector::store(vector::mul_high(va, vb)), portable::mul_high(a, b));
            assert_eq!(vector::store(vector::mul_sums(va, vb)), portable::mul_sums(a, b));
            assert_eq!(vector::store(vector::shift_up(va)), portable::shift_up(a));
            assert_eq!(vector::store(vector::shift_down(va)), portable::shift_down(a));
            let (low, high) = vector::split_x7(va);
            assert_eq!((vector::store(low), vector::store(high)), portable::split_x7(a));
            assert_eq!(vector::store(vector::xor(va, vb)), a ^ b);
        }
        assert_eq!(vector::store(vector::modulus()), portable::modulus());
    }

    #[test]
    fn products_match_the_reference() {
        let mut rng = Rng::from_seed([2; 32]);
        let edge = [0, 1, u128::MAX, 1 << 127, 1 << 121, (1 << 7) - 1];
        for &a in &edge {
            for &b in &edge {
                assert_eq!(vector::mul(a, b), mul_reference(a, b));
                assert_eq!(vector::store(vector::mul_x121(vector::load(a))), mul_reference(a, 1 << 121));
            }
        }
        for _ in 0..500 {
            let (p, q): (Gf2_128, Gf2_128) = (rng.field(), rng.field());
            assert_eq!((p * q).0, mul_reference(p.0, q.0));
            assert_eq!(p.square().0, mul_reference(p.0, p.0));
        }
    }

    #[test]
    fn the_modulus_is_irreducible() {
        // Rabin's test for degree 128: x^(2^128) = x, and x^(2^64) - x is
        // coprime to the modulus, that is, invertible modulo it.
        let mut power = 2;
        for _ in 0..64 {
            power = mul_reference(power, power);
        }
        let difference = Gf2_128(power ^ 2);
        assert_eq!(difference * difference.inverse(), Gf2_128::ONE);
        for _ in 0..64 {
            power = mul_reference(power, power);
        }
        assert_eq!(power, 2);
    }
}
