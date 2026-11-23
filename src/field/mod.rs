mod gf2_128;
mod gf2_256;
mod vector;

pub use gf2_128::Gf2_128;
pub use gf2_256::Gf2_256;

#[inline(always)]
pub(crate) fn precompute_mul_pair(w: u128) -> (u128, u128) {
    vector::Mul128::new(w).to_pair()
}

use std::fmt::Debug;
use std::iter::Sum;
use std::mem::MaybeUninit;
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
        for _ in 0..64 {
            if exponent == 0 {
                break;
            }
            if exponent & 1 == 1 {
                result *= self;
            }
            self = self.square();
            exponent >>= 1;
        }
        assert_eq!(exponent, 0);
        result
    }

    fn inner_product_slice(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(Self::ZERO, |sum, (&x, &y)| sum + x * y)
    }

    fn round_coefficients_slice(
        a_low: &[Self],
        a_high: &[Self],
        b_low: &[Self],
        b_high: &[Self],
    ) -> (Self, Self) {
        assert_eq!(a_low.len(), a_high.len());
        assert_eq!(a_low.len(), b_low.len());
        assert_eq!(a_low.len(), b_high.len());
        let mut constant = Self::ZERO;
        let mut quadratic = Self::ZERO;
        for i in 0..a_low.len() {
            constant += a_low[i] * b_low[i];
            quadratic += (a_high[i] - a_low[i]) * (b_high[i] - b_low[i]);
        }
        (constant, quadratic)
    }

    fn fold_slice(low: &mut [Self], high: &[Self], challenge: Self) {
        assert_eq!(low.len(), high.len());
        for (l, &h) in low.iter_mut().zip(high) {
            *l += (h - *l) * challenge;
        }
    }

    fn eq_extend_uninit(low: &mut [Self], high: &mut [MaybeUninit<Self>], coordinate: Self) {
        assert_eq!(low.len(), high.len());
        for (l, h) in low.iter_mut().zip(high) {
            let scaled = *l * coordinate;
            h.write(scaled);
            *l -= scaled;
        }
    }

    fn add_scaled_slice(dst: &mut [Self], src: &[Self], scale: Self) {
        assert_eq!(dst.len(), src.len());
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += s * scale;
        }
    }

    fn eval_horner_slice(coefficients: &[Self], x: Self) -> Self {
        coefficients.iter().rev().fold(Self::ZERO, |value, &c| value * x + c)
    }

    fn add_powers_slice(target: &mut [Self], x: Self, mut scale: Self) {
        for entry in target {
            *entry += scale;
            scale *= x;
        }
    }

    fn xor_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len());
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += s;
        }
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

    fn butterfly_slice(low: &mut [Self], high: &mut [Self], twiddle: u128) {
        assert_eq!(low.len(), high.len());
        if twiddle == 0 {
            for (r, q) in low.iter_mut().zip(high) {
                *q += *r;
            }
        } else {
            for (r, q) in low.iter_mut().zip(high) {
                *r += q.mul_base(twiddle);
                *q += *r;
            }
        }
    }

    fn butterfly_radix4(
        v0: &mut [Self],
        v1: &mut [Self],
        v2: &mut [Self],
        v3: &mut [Self],
        w: u128,
        w0: u128,
        w1: u128,
    ) {
        let len = v0.len();
        assert!(v1.len() == len && v2.len() == len && v3.len() == len);
        for i in 0..len {
            let (mut a0, mut a1, mut a2, mut a3) = (v0[i], v1[i], v2[i], v3[i]);
            if w != 0 {
                a0 += a2.mul_base(w);
                a1 += a3.mul_base(w);
            }
            a2 += a0;
            a3 += a1;
            if w0 != 0 {
                a0 += a1.mul_base(w0);
            }
            a1 += a0;
            if w1 != 0 {
                a2 += a3.mul_base(w1);
            }
            a3 += a2;
            (v0[i], v1[i], v2[i], v3[i]) = (a0, a1, a2, a3);
        }
    }

    fn butterfly_radix8_pass(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        table0: &[(u128, u128); 256],
        table1: &[(u128, u128); 256],
    ) {
        let p2 = table0[2].0;
        let p4 = table0[4].0;
        let _ = (p2, p4, table1);
        let _ = (values, base4, base2, base1);
        unreachable!("implemented for concrete binary fields");
    }

    fn butterfly_radix8_outer(values: &mut [Self], w4: u128, w2_0: u128, w2_1: u128, w1: [u128; 4]) {
        let len = values.len();
        assert!(len >= 8 && len.is_multiple_of(8));
        let half = len / 2;
        let q = len / 4;
        let e = len / 8;
        let (lo, hi) = values.split_at_mut(half);
        Self::butterfly_slice(lo, hi, w4);
        let (v01, v23) = lo.split_at_mut(q);
        let (v45, v67) = hi.split_at_mut(q);
        Self::butterfly_slice(v01, v23, w2_0);
        Self::butterfly_slice(v45, v67, w2_1);
        let (v0, v1) = v01.split_at_mut(e);
        let (v2, v3) = v23.split_at_mut(e);
        let (v4, v5) = v45.split_at_mut(e);
        let (v6, v7) = v67.split_at_mut(e);
        Self::butterfly_slice(v0, v1, w1[0]);
        Self::butterfly_slice(v2, v3, w1[1]);
        Self::butterfly_slice(v4, v5, w1[2]);
        Self::butterfly_slice(v6, v7, w1[3]);
        assert_eq!(v7.len(), e);
    }

    fn copy_half_radix8_uninit(
        novel_low: &[Self],
        novel_high: &[Self],
        dst: &mut [MaybeUninit<Self>],
        w_eff: u128,
        w4: u128,
        w2: [u128; 2],
        w1: [u128; 4],
    ) {
        assert!(dst.len() >= 8 && dst.len().is_multiple_of(8));
        Self::copy_half_uninit(novel_low, novel_high, dst, w_eff);
        let full = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<Self>(), dst.len()) };
        Self::butterfly_radix8_outer(full, w4, w2[0], w2[1], w1);
        assert_eq!(full.len(), novel_low.len());
    }

    fn copy_half_radix4_uninit(
        novel_low: &[Self],
        novel_high: &[Self],
        dst: &mut [MaybeUninit<Self>],
        w_eff: u128,
        w: u128,
        w0: u128,
        w1: u128,
    ) {
        let half = dst.len();
        assert!(half >= 4 && half.is_multiple_of(4) && novel_low.len() == half);
        assert!(novel_high.len() <= half);
        let q = half / 4;
        let qcount = novel_high.len();
        let (d01, d23) = dst.split_at_mut(2 * q);
        let (d0, d1) = d01.split_at_mut(q);
        let (d2, d3) = d23.split_at_mut(q);
        for i in 0..q {
            let mut a0 = novel_low[i];
            let mut a1 = novel_low[q + i];
            let mut a2 = novel_low[2 * q + i];
            let mut a3 = novel_low[3 * q + i];
            if i < qcount {
                a0 += novel_high[i].mul_base(w_eff);
            }
            if q + i < qcount {
                a1 += novel_high[q + i].mul_base(w_eff);
            }
            if 2 * q + i < qcount {
                a2 += novel_high[2 * q + i].mul_base(w_eff);
            }
            if 3 * q + i < qcount {
                a3 += novel_high[3 * q + i].mul_base(w_eff);
            }
            a0 += a2.mul_base(w);
            a1 += a3.mul_base(w);
            a2 += a0;
            a3 += a1;
            a0 += a1.mul_base(w0);
            a1 += a0;
            a2 += a3.mul_base(w1);
            a3 += a2;
            d0[i].write(a0);
            d1[i].write(a1);
            d2[i].write(a2);
            d3[i].write(a3);
        }
    }

    fn copy_half_uninit(novel_low: &[Self], novel_high: &[Self], dst: &mut [MaybeUninit<Self>], w_eff: u128) {
        let half = dst.len();
        assert!(novel_low.len() == half && novel_high.len() <= half);
        let qcount = novel_high.len();
        for i in 0..qcount {
            dst[i].write(novel_low[i] + novel_high[i].mul_base(w_eff));
        }
        for i in qcount..half {
            dst[i].write(novel_low[i]);
        }
    }

    fn eval_horner_base(coefficients: &[Self], x: u128) -> Self {
        coefficients.iter().rev().fold(Self::ZERO, |value, &c| value.mul_base(x) + c)
    }
}

fn square_times<F: Field>(mut value: F, times: u32) -> F {
    assert!(times <= 256);
    for _ in 0..times {
        value = value.square();
    }
    value
}

fn binary_inverse<F: Field>(x: F) -> F {
    assert_ne!(x, F::ZERO, "zero has no inverse");
    let exponent = F::BITS - 1;
    assert!((1..512).contains(&exponent));
    let top_bit = u32::BITS - 1 - exponent.leading_zeros();
    let mut acc = x;
    let mut width = 1u32;
    for bit in (0..top_bit).rev() {
        acc = square_times(acc, width) * acc;
        width *= 2;
        if (exponent >> bit) & 1 == 1 {
            acc = acc.square() * x;
            width += 1;
        }
    }
    assert_eq!(width, exponent);
    acc.square()
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
