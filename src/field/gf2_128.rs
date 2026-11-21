use super::{BinaryField, Field, binary_inverse, impl_binary_field_operators, vector};
use std::fmt;
use std::ops::{Add, Mul};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gf2_128(u128);

impl Gf2_128 {
    pub const fn new(limb: u128) -> Self {
        Self(limb)
    }

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
