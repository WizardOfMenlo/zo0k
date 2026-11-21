//! The binary field `GF(2^256)`.

use super::{BinaryField, Field, binary_inverse, impl_binary_field_operators, vector};
use std::fmt;
use std::ops::{Add, Mul};

/// An element of `K[u] / (u^2 + u + x^121)` over `K`, the field of
/// [`super::Gf2_128`]. The element `a + b·u` is stored as the limbs of `a`
/// and `b`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gf2_256(u128, u128);

impl Gf2_256 {
    /// The element `a + b·u`.
    pub const fn new(a: u128, b: u128) -> Self {
        Self(a, b)
    }
}

impl Field for Gf2_256 {
    const ZERO: Self = Self(0, 0);
    const ONE: Self = Self(1, 0);
    const BITS: u32 = 256;
    const BYTES: usize = 32;
    const NAME: &'static str = "gf2_256";

    fn from_integer(n: usize) -> Self {
        Self((n & 1) as u128, 0)
    }

    fn inverse(self) -> Self {
        binary_inverse(self)
    }

    fn write_bytes(self, bytes: &mut [u8]) {
        let (a, b) = bytes.split_at_mut(16);
        a.copy_from_slice(&self.0.to_le_bytes());
        b.copy_from_slice(&self.1.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (a, b) = bytes.split_at_checked(16)?;
        Some(Self(u128::from_le_bytes(a.try_into().ok()?), u128::from_le_bytes(b.try_into().ok()?)))
    }

    fn sample(mut fill: impl FnMut(&mut [u8])) -> Self {
        let mut bytes = [0; 32];
        fill(&mut bytes);
        Self::from_bytes(&bytes).expect("every string of 32 bytes encodes an element")
    }

    #[inline(always)]
    fn square(self) -> Self {
        // (a + b·u)^2 = a^2 + b^2·(u + x^121).
        use vector::{load, store, xor};
        let (a, b) = (vector::square(self.0), vector::square(self.1));
        Self(store(xor(load(a), vector::mul_x121(load(b)))), b)
    }
}

impl BinaryField for Gf2_256 {
    fn from_base(limb: u128) -> Self {
        Self(limb, 0)
    }

    /// Two thirds of the cost of a general product.
    #[inline(always)]
    fn mul_base(self, w: u128) -> Self {
        Self(vector::mul(self.0, w), vector::mul(self.1, w))
    }
}

impl fmt::Debug for Gf2_256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gf2_256({:#x}, {:#x})", self.0, self.1)
    }
}

impl Add for Gf2_256 {
    type Output = Self;
    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0, self.1 ^ rhs.1)
    }
}

impl Mul for Gf2_256 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // (a + b·u)(c + d·u) = (ac + bd·x^121) + (ac + (a + b)(c + d))·u.
        use vector::{load, mul_wide, mul_x121, reduce, store, xor};
        let (a, b, c, d) = (load(self.0), load(self.1), load(rhs.0), load(rhs.1));
        let ac = mul_wide(a, c);
        let bd = mul_wide(b, d);
        let cross = mul_wide(xor(a, b), xor(c, d));
        Self(
            store(xor(reduce(ac.0, ac.1), mul_x121(reduce(bd.0, bd.1)))),
            store(reduce(xor(ac.0, cross.0), xor(ac.1, cross.1))),
        )
    }
}

impl_binary_field_operators!(Gf2_256);
