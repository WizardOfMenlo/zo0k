//! Finite fields.
//!
//! The protocol is generic over a [`Field`] with Reed–Solomon codes
//! ([`crate::reed_solomon::RsField`]). The field is at once the ring, the
//! module and the challenge set of the paper. This crate provides two binary
//! fields, [`Gf2_128`] and its quadratic extension [`Gf2_256`].
//!
//! Arithmetic is not constant time.

mod gf2_128;
mod gf2_256;
mod vector;

pub use gf2_128::Gf2_128;
pub use gf2_256::Gf2_256;

use serde::Serialize;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A finite field.
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
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// Base-two logarithm of the number of elements, rounded down.
    const BITS: u32;
    /// Length of the encoding of an element.
    const BYTES: usize;
    /// A name of the field, which transcripts bind.
    const NAME: &'static str;
    /// The sizes on which the soundness error and the communication depend.
    const SIZE: FieldSize = FieldSize { bits: Self::BITS, bytes: Self::BYTES };

    /// The image of the integer `n`.
    fn from_integer(n: usize) -> Self;

    /// The multiplicative inverse.
    ///
    /// # Panics
    /// Panics if `self` is zero.
    fn inverse(self) -> Self;

    /// Write the encoding of the element to `bytes`, of length
    /// [`Field::BYTES`].
    fn write_bytes(self, bytes: &mut [u8]);

    /// The element with the given encoding of [`Field::BYTES`] bytes, if
    /// there is one.
    fn from_bytes(bytes: &[u8]) -> Option<Self>;

    /// A uniform element, from a source of uniform bytes.
    fn sample(fill: impl FnMut(&mut [u8])) -> Self;

    /// The square of the element.
    fn square(self) -> Self {
        self * self
    }

    /// The power of the element with the given exponent.
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

/// The sizes of a field that determine the soundness error and the
/// communication of the protocol over it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct FieldSize {
    /// Base-two logarithm of the number of elements.
    pub bits: u32,
    /// Length of the encoding of an element.
    pub bytes: usize,
}

/// A field of `2^BITS` elements that contains [`Gf2_128`]. Its Reed–Solomon
/// codes are encoded by the additive transform of [`crate::reed_solomon`],
/// whose evaluation points lie in that subfield.
pub trait BinaryField: Field {
    /// The element of the subfield with the given limb (see [`Gf2_128`]).
    fn from_base(limb: u128) -> Self;

    /// The product with the element `w` of the subfield.
    fn mul_base(self, w: u128) -> Self;
}

/// The inverse in a field of `2^BITS` elements, as `x^(2^BITS - 2)`.
fn binary_inverse<F: Field>(x: F) -> F {
    assert_ne!(x, F::ZERO, "zero has no inverse");
    let mut result = x;
    for _ in 2..F::BITS {
        result = result.square() * x;
    }
    result.square()
}

/// The operators that follow from `Add` and `Mul` in characteristic two.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn axioms<F: Field>() {
        let mut rng = Rng::from_seed([3; 32]);
        for _ in 0..50 {
            let (p, q, r): (F, F, F) = (rng.field(), rng.field(), rng.field());
            assert_eq!(p * (q + r), p * q + p * r);
            assert_eq!((p * q) * r, p * (q * r));
            assert_eq!(p * q, q * p);
            assert_eq!(p - q, p + (-q));
            assert_eq!(p * p.inverse(), F::ONE);
            assert_eq!(p.square(), p * p);
            assert_eq!(p.pow(5), p * p * p * p * p);
            let mut bytes = vec![0; F::BYTES];
            p.write_bytes(&mut bytes);
            assert_eq!(F::from_bytes(&bytes), Some(p));
        }
        assert_eq!(F::from_integer(6), F::ZERO);
        assert_eq!(F::from_integer(7), F::ONE);
        assert_eq!([F::ONE, F::ONE, F::ONE].iter().sum::<F>(), F::ONE);
        assert_eq!(F::SIZE, FieldSize { bits: F::BITS, bytes: F::BYTES });
    }

    fn subfield<F: BinaryField>() {
        let mut rng = Rng::from_seed([4; 32]);
        for _ in 0..50 {
            let (p, w): (F, Gf2_128) = (rng.field(), rng.field());
            assert_eq!(p.mul_base(w.limb()), p * F::from_base(w.limb()));
            let v: Gf2_128 = rng.field();
            // The embedding of the subfield is a homomorphism.
            assert_eq!(F::from_base((v * w).limb()), F::from_base(v.limb()) * F::from_base(w.limb()));
        }
        assert_eq!(F::from_base(1), F::ONE);
    }

    #[test]
    fn field_axioms_hold_on_random_elements() {
        axioms::<Gf2_128>();
        axioms::<Gf2_256>();
    }

    #[test]
    fn both_fields_contain_the_subfield() {
        subfield::<Gf2_128>();
        subfield::<Gf2_256>();
    }
}
