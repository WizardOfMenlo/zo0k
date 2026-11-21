mod gf2_128;
mod gf2_256;
mod vector;

pub use gf2_128::Gf2_128;
pub use gf2_256::Gf2_256;

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Field:
    Copy
    + Debug
    + Default
    + Eq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Sum
    + for<'a> Sum<&'a Self>
{
    const ZERO: Self;
    const ONE: Self;
    const BITS: u32;
    const BYTES: usize;
    const NAME: &'static str;
    const SIZE: FieldSize = FieldSize { bits: Self::BITS, bytes: Self::BYTES };

    fn from_integer(n: usize) -> Self;

    fn inverse(self) -> Self;

    fn write_bytes(self, bytes: &mut [u8]);

    fn from_bytes(bytes: &[u8]) -> Option<Self>;

    fn sample(fill: impl FnMut(&mut [u8])) -> Self;

    fn square(self) -> Self {
        self * self
    }

    fn pow(mut self, mut exponent: u64) -> Self {
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result *= self;
            }
            self = self.square();
            exponent >>= 1;
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldSize {
    pub bits: u32,
    pub bytes: usize,
}

pub trait BinaryField: Field {
    fn from_base(limb: u128) -> Self;

    fn mul_base(self, w: u128) -> Self;
}

fn binary_inverse<F: Field>(x: F) -> F {
    assert_ne!(x, F::ZERO, "zero has no inverse");
    let mut result = x;
    for _ in 2..F::BITS {
        result = result.square() * x;
    }
    result.square()
}

macro_rules! impl_binary_field_operators {
    ($field:ty) => {
        impl std::ops::Sub for $field {
            type Output = Self;
            #[inline(always)]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn sub(self, rhs: Self) -> Self {
                self + rhs
            }
        }
        impl std::ops::Neg for $field {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                self
            }
        }
        impl std::ops::AddAssign for $field {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl std::ops::SubAssign for $field {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl std::ops::MulAssign for $field {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl std::iter::Sum for $field {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(<Self as $crate::field::Field>::ZERO, std::ops::Add::add)
            }
        }
        impl<'a> std::iter::Sum<&'a $field> for $field {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.copied().sum()
            }
        }
    };
}
use impl_binary_field_operators;
