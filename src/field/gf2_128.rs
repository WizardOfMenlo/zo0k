use super::vector::{self, Acc128, Mul128, V};
use super::{BinaryField, Field, binary_inverse, impl_binary_field_operators};
use std::fmt;
use std::mem::MaybeUninit;
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

    fn inner_product_slice(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(a.len(), b.len());
        let mut acc0 = Acc128::zero();
        let mut acc1 = Acc128::zero();
        let mut acc2 = Acc128::zero();
        let mut acc3 = Acc128::zero();
        let chunks = a.len() / 4;
        for c in 0..chunks {
            let i = c * 4;
            acc0.add_product(vector::load(a[i].0), vector::load(b[i].0));
            acc1.add_product(vector::load(a[i + 1].0), vector::load(b[i + 1].0));
            acc2.add_product(vector::load(a[i + 2].0), vector::load(b[i + 2].0));
            acc3.add_product(vector::load(a[i + 3].0), vector::load(b[i + 3].0));
        }
        let mut acc = acc0.merge(acc1).merge(acc2.merge(acc3));
        for i in (chunks * 4)..a.len() {
            acc.add_product(vector::load(a[i].0), vector::load(b[i].0));
        }
        assert!(chunks * 4 <= a.len());
        Self(vector::store(acc.reduce()))
    }

    fn round_coefficients_slice(
        a_low: &[Self],
        a_high: &[Self],
        b_low: &[Self],
        b_high: &[Self],
    ) -> (Self, Self) {
        let len = a_low.len();
        assert!(a_high.len() == len && b_low.len() == len && b_high.len() == len);
        let mut c0 = Acc128::zero();
        let mut c1 = Acc128::zero();
        let mut q0 = Acc128::zero();
        let mut q1 = Acc128::zero();
        let chunks = len / 2;
        for c in 0..chunks {
            let i = c * 2;
            let (al0, ah0) = (vector::load(a_low[i].0), vector::load(a_high[i].0));
            let (bl0, bh0) = (vector::load(b_low[i].0), vector::load(b_high[i].0));
            let (al1, ah1) = (vector::load(a_low[i + 1].0), vector::load(a_high[i + 1].0));
            let (bl1, bh1) = (vector::load(b_low[i + 1].0), vector::load(b_high[i + 1].0));
            c0.add_product(al0, bl0);
            q0.add_product(vector::xor(ah0, al0), vector::xor(bh0, bl0));
            c1.add_product(al1, bl1);
            q1.add_product(vector::xor(ah1, al1), vector::xor(bh1, bl1));
        }
        let mut c_acc = c0.merge(c1);
        let mut q_acc = q0.merge(q1);
        for i in (chunks * 2)..len {
            let (al, ah) = (vector::load(a_low[i].0), vector::load(a_high[i].0));
            let (bl, bh) = (vector::load(b_low[i].0), vector::load(b_high[i].0));
            c_acc.add_product(al, bl);
            q_acc.add_product(vector::xor(ah, al), vector::xor(bh, bl));
        }
        assert!(chunks * 2 <= len);
        (Self(vector::store(c_acc.reduce())), Self(vector::store(q_acc.reduce())))
    }

    fn fold_slice(low: &mut [Self], high: &[Self], challenge: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        if challenge.0 == 0 {
            return;
        }
        let p = Mul128::new(challenge.0);
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            for k in 0..4 {
                let l = vector::load(low[i + k].0);
                let h = vector::load(high[i + k].0);
                low[i + k] = Self(vector::store(p.mul_xor_v(l, vector::xor(h, l))));
            }
        }
        for i in (chunks * 4)..len {
            let l = vector::load(low[i].0);
            let h = vector::load(high[i].0);
            low[i] = Self(vector::store(p.mul_xor_v(l, vector::xor(h, l))));
        }
        assert!(chunks * 4 <= len);
    }

    fn eq_extend_uninit(low: &mut [Self], high: &mut [MaybeUninit<Self>], coordinate: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        let p = Mul128::new(coordinate.0);
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            for k in 0..4 {
                let l = vector::load(low[i + k].0);
                let h = p.mul_v(l);
                high[i + k].write(Self(vector::store(h)));
                low[i + k] = Self(vector::store(vector::xor(l, h)));
            }
        }
        for i in (chunks * 4)..len {
            let l = vector::load(low[i].0);
            let h = p.mul_v(l);
            high[i].write(Self(vector::store(h)));
            low[i] = Self(vector::store(vector::xor(l, h)));
        }
        assert!(chunks * 4 <= len);
    }

    fn add_scaled_slice(dst: &mut [Self], src: &[Self], scale: Self) {
        let len = dst.len();
        assert_eq!(len, src.len());
        if scale.0 == 0 {
            return;
        }
        let p = Mul128::new(scale.0);
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            for k in 0..4 {
                let d = vector::load(dst[i + k].0);
                let s = vector::load(src[i + k].0);
                dst[i + k] = Self(vector::store(p.mul_xor_v(d, s)));
            }
        }
        for i in (chunks * 4)..len {
            let d = vector::load(dst[i].0);
            let s = vector::load(src[i].0);
            dst[i] = Self(vector::store(p.mul_xor_v(d, s)));
        }
        assert!(chunks * 4 <= len);
    }

    fn eval_horner_slice(coefficients: &[Self], x: Self) -> Self {
        Self::eval_horner_base(coefficients, x.0)
    }

    fn add_powers_slice(target: &mut [Self], x: Self, scale: Self) {
        let len = target.len();
        if len == 0 {
            return;
        }
        let px = Mul128::new(x.0);
        let mut s = [vector::load(scale.0); 4];
        for k in 1..4 {
            s[k] = px.mul_v(s[k - 1]);
        }
        let px4 = Mul128::new(vector::square(vector::square(x.0)));
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            for k in 0..4 {
                target[i + k].0 ^= vector::store(s[k]);
                s[k] = px4.mul_v(s[k]);
            }
        }
        let mut cur = s[0];
        for entry in &mut target[(chunks * 4)..] {
            entry.0 ^= vector::store(cur);
            cur = px.mul_v(cur);
        }
        assert!(chunks * 4 <= len);
    }

    fn xor_slice(dst: &mut [Self], src: &[Self]) {
        let len = dst.len();
        assert_eq!(len, src.len());
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            for k in 0..4 {
                let v = vector::xor(vector::load(dst[i + k].0), vector::load(src[i + k].0));
                dst[i + k] = Self(vector::store(v));
            }
        }
        for i in (chunks * 4)..len {
            let v = vector::xor(vector::load(dst[i].0), vector::load(src[i].0));
            dst[i] = Self(vector::store(v));
        }
        assert!(chunks * 4 <= len);
    }
}

#[inline(always)]
fn butterfly_step_v(r: V, q: V, p: Mul128) -> (V, V) {
    let nr = p.mul_xor_v(r, q);
    (nr, vector::xor(q, nr))
}

#[inline(always)]
fn radix8_8lanes(v: &mut [V; 8], p4: Mul128, p2_0: Mul128, p2_1: Mul128, p1: [Mul128; 4]) {
    for j in 0..4 {
        (v[j], v[j + 4]) = butterfly_step_v(v[j], v[j + 4], p4);
    }
    for j in 0..2 {
        (v[j], v[j + 2]) = butterfly_step_v(v[j], v[j + 2], p2_0);
        (v[j + 4], v[j + 6]) = butterfly_step_v(v[j + 4], v[j + 6], p2_1);
    }
    for j in 0..4 {
        (v[2 * j], v[2 * j + 1]) = butterfly_step_v(v[2 * j], v[2 * j + 1], p1[j]);
    }
}

fn butterfly_radix4_nonzero(
    v0: &mut [Gf2_128],
    v1: &mut [Gf2_128],
    v2: &mut [Gf2_128],
    v3: &mut [Gf2_128],
    pw: Mul128,
    pw0: Mul128,
    pw1: Mul128,
) {
    let len = v0.len();
    assert!(v1.len() == len && v2.len() == len && v3.len() == len);
    let chunks = len / 2;
    for c in 0..chunks {
        let i = c * 2;
        for k in 0..2 {
            let (a0, a2) = butterfly_step_v(vector::load(v0[i + k].0), vector::load(v2[i + k].0), pw);
            let (a1, a3) = butterfly_step_v(vector::load(v1[i + k].0), vector::load(v3[i + k].0), pw);
            let (b0, b1) = butterfly_step_v(a0, a1, pw0);
            let (b2, b3) = butterfly_step_v(a2, a3, pw1);
            v0[i + k] = Gf2_128(vector::store(b0));
            v1[i + k] = Gf2_128(vector::store(b1));
            v2[i + k] = Gf2_128(vector::store(b2));
            v3[i + k] = Gf2_128(vector::store(b3));
        }
    }
    for i in (chunks * 2)..len {
        let (a0, a2) = butterfly_step_v(vector::load(v0[i].0), vector::load(v2[i].0), pw);
        let (a1, a3) = butterfly_step_v(vector::load(v1[i].0), vector::load(v3[i].0), pw);
        let (b0, b1) = butterfly_step_v(a0, a1, pw0);
        let (b2, b3) = butterfly_step_v(a2, a3, pw1);
        v0[i] = Gf2_128(vector::store(b0));
        v1[i] = Gf2_128(vector::store(b1));
        v2[i] = Gf2_128(vector::store(b2));
        v3[i] = Gf2_128(vector::store(b3));
    }
    assert!(chunks * 2 <= len);
}

fn butterfly_radix4_zero_prefix(
    v0: &mut [Gf2_128],
    v1: &mut [Gf2_128],
    v2: &mut [Gf2_128],
    v3: &mut [Gf2_128],
    pw1: Mul128,
) {
    let len = v0.len();
    assert!(v1.len() == len && v2.len() == len && v3.len() == len);
    for i in 0..len {
        let a0 = vector::load(v0[i].0);
        let a1 = vector::load(v1[i].0);
        let a2 = vector::xor(vector::load(v2[i].0), a0);
        let a3 = vector::xor(vector::load(v3[i].0), a1);
        let (b2, b3) = butterfly_step_v(a2, a3, pw1);
        v1[i] = Gf2_128(vector::store(vector::xor(a1, a0)));
        v2[i] = Gf2_128(vector::store(b2));
        v3[i] = Gf2_128(vector::store(b3));
    }
    assert_eq!(v0.len(), len);
}

#[inline(always)]
fn write_radix4_128(
    d0: &mut [MaybeUninit<Gf2_128>],
    d1: &mut [MaybeUninit<Gf2_128>],
    d2: &mut [MaybeUninit<Gf2_128>],
    d3: &mut [MaybeUninit<Gf2_128>],
    i: usize,
    a: [V; 4],
    p: [Mul128; 3],
) {
    let (t0, t2) = butterfly_step_v(a[0], a[2], p[0]);
    let (t1, t3) = butterfly_step_v(a[1], a[3], p[0]);
    let (b0, b1) = butterfly_step_v(t0, t1, p[1]);
    let (b2, b3) = butterfly_step_v(t2, t3, p[2]);
    d0[i].write(Gf2_128(vector::store(b0)));
    d1[i].write(Gf2_128(vector::store(b1)));
    d2[i].write(Gf2_128(vector::store(b2)));
    d3[i].write(Gf2_128(vector::store(b3)));
}

#[inline(always)]
fn radix8_zero_prefix_8lanes(v: &mut [V; 8], p2_1: Mul128, p1: [Mul128; 4]) {
    for j in 0..4 {
        v[j + 4] = vector::xor(v[j + 4], v[j]);
    }
    for j in 0..2 {
        v[j + 2] = vector::xor(v[j + 2], v[j]);
        (v[j + 4], v[j + 6]) = butterfly_step_v(v[j + 4], v[j + 6], p2_1);
    }
    v[1] = vector::xor(v[1], v[0]);
    for j in 1..4 {
        (v[2 * j], v[2 * j + 1]) = butterfly_step_v(v[2 * j], v[2 * j + 1], p1[j]);
    }
}

fn radix8_outer_128(values: &mut [Gf2_128], m4: Mul128, m2: [Mul128; 2], m1: [Mul128; 4], zero_prefix: bool) {
    let len = values.len();
    assert!(len >= 8 && len.is_multiple_of(8));
    let e = len / 8;
    let ptr = values.as_mut_ptr();
    if zero_prefix {
        for i in 0..e {
            let mut v = [vector::zero(); 8];
            for (k, slot) in v.iter_mut().enumerate() {
                *slot = vector::load(unsafe { (*ptr.add(k * e + i)).0 });
            }
            radix8_zero_prefix_8lanes(&mut v, m2[1], m1);
            for (k, &val) in v.iter().enumerate() {
                unsafe { (*ptr.add(k * e + i)).0 = vector::store(val) };
            }
        }
    } else {
        for i in 0..e {
            let mut v = [vector::zero(); 8];
            for (k, slot) in v.iter_mut().enumerate() {
                *slot = vector::load(unsafe { (*ptr.add(k * e + i)).0 });
            }
            radix8_8lanes(&mut v, m4, m2[0], m2[1], m1);
            for (k, &val) in v.iter().enumerate() {
                unsafe { (*ptr.add(k * e + i)).0 = vector::store(val) };
            }
        }
    }
    assert_eq!(e * 8, len);
}

fn copy_radix8_128_short(
    novel_low: &[Gf2_128],
    novel_high: &[Gf2_128],
    dst: &mut [MaybeUninit<Gf2_128>],
    pw_eff: Mul128,
    m4: Mul128,
    m2: [Mul128; 2],
    m1: [Mul128; 4],
) {
    let e = dst.len() / 8;
    let qcount = novel_high.len();
    assert!(dst.len() == 8 * e && novel_low.len() == 8 * e && qcount <= e);
    let nl = novel_low.as_ptr();
    let nh = novel_high.as_ptr();
    let dp = dst.as_mut_ptr();
    for i in 0..qcount {
        let mut v = [vector::zero(); 8];
        v[0] = pw_eff
            .mul_xor_v(vector::load(unsafe { (*nl.add(i)).0 }), vector::load(unsafe { (*nh.add(i)).0 }));
        for (k, slot) in v.iter_mut().enumerate().skip(1) {
            *slot = vector::load(unsafe { (*nl.add(k * e + i)).0 });
        }
        radix8_8lanes(&mut v, m4, m2[0], m2[1], m1);
        for (k, &val) in v.iter().enumerate() {
            unsafe { (*dp.add(k * e + i)).write(Gf2_128(vector::store(val))) };
        }
    }
    for i in qcount..e {
        let mut v = [vector::zero(); 8];
        for (k, slot) in v.iter_mut().enumerate() {
            *slot = vector::load(unsafe { (*nl.add(k * e + i)).0 });
        }
        radix8_8lanes(&mut v, m4, m2[0], m2[1], m1);
        for (k, &val) in v.iter().enumerate() {
            unsafe { (*dp.add(k * e + i)).write(Gf2_128(vector::store(val))) };
        }
    }
    assert!(qcount <= e);
}

impl BinaryField for Gf2_128 {
    fn from_base(limb: u128) -> Self {
        Self(limb)
    }

    #[inline(always)]
    fn mul_base(self, w: u128) -> Self {
        Self(vector::mul(self.0, w))
    }

    fn butterfly_slice(low: &mut [Self], high: &mut [Self], twiddle: u128) {
        let len = low.len();
        assert_eq!(len, high.len());
        if twiddle == 0 {
            for i in 0..len {
                high[i].0 ^= low[i].0;
            }
            return;
        }
        let p = Mul128::new(twiddle);
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            for k in 0..4 {
                let (nr, nq) = butterfly_step_v(vector::load(low[i + k].0), vector::load(high[i + k].0), p);
                low[i + k] = Self(vector::store(nr));
                high[i + k] = Self(vector::store(nq));
            }
        }
        for i in (chunks * 4)..len {
            let (nr, nq) = butterfly_step_v(vector::load(low[i].0), vector::load(high[i].0), p);
            low[i] = Self(vector::store(nr));
            high[i] = Self(vector::store(nq));
        }
        assert!(chunks * 4 <= len);
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
        assert!(v0.len() == v1.len() && v0.len() == v2.len() && v0.len() == v3.len());
        if w != 0 && w0 != 0 && w1 != 0 {
            butterfly_radix4_nonzero(v0, v1, v2, v3, Mul128::new(w), Mul128::new(w0), Mul128::new(w1));
        } else if w == 0 && w0 == 0 && w1 != 0 {
            butterfly_radix4_zero_prefix(v0, v1, v2, v3, Mul128::new(w1));
        } else {
            Self::butterfly_slice(v0, v2, w);
            Self::butterfly_slice(v1, v3, w);
            Self::butterfly_slice(v0, v1, w0);
            Self::butterfly_slice(v2, v3, w1);
        }
        assert_eq!(v0.len(), v3.len());
    }

    fn butterfly_radix8_pass(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        table0: &[(u128, u128); 256],
        table1: &[(u128, u128); 256],
    ) {
        let load_m = |pair: (u128, u128)| Mul128::from_pair(pair.0, pair.1);
        let p2 = load_m(table0[2]);
        let p4 = load_m(table0[4]);
        let base4_m = Mul128::new(base4);
        let base2_m = Mul128::new(base2);
        let base1_m = Mul128::new(base1);
        let (blocks, remainder) = values.as_chunks_mut::<8>();
        assert!(remainder.is_empty());
        assert!(!blocks.is_empty());
        for (i, block) in blocks.iter_mut().enumerate() {
            let m4 = base4_m.xor(load_m(table0[(2 * i) & 255])).xor(load_m(table1[(2 * i) >> 8]));
            let m2_0 = base2_m.xor(load_m(table0[(4 * i) & 255])).xor(load_m(table1[(4 * i) >> 8]));
            let m1_0 = base1_m.xor(load_m(table0[(8 * i) & 255])).xor(load_m(table1[(8 * i) >> 8]));
            let m1_1 = m1_0.xor(p2);
            let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
            let mut v = [
                vector::load(block[0].0),
                vector::load(block[1].0),
                vector::load(block[2].0),
                vector::load(block[3].0),
                vector::load(block[4].0),
                vector::load(block[5].0),
                vector::load(block[6].0),
                vector::load(block[7].0),
            ];
            radix8_8lanes(&mut v, m4, m2_0, m2_0.xor(p2), m1);
            for j in 0..8 {
                block[j] = Self(vector::store(v[j]));
            }
        }
    }

    fn butterfly_radix8_outer(values: &mut [Self], w4: u128, w2_0: u128, w2_1: u128, w1: [u128; 4]) {
        let len = values.len();
        assert!(len >= 8 && len.is_multiple_of(8));
        let m4 = Mul128::new(w4);
        let m2 = [Mul128::new(w2_0), Mul128::new(w2_1)];
        let m1 = [Mul128::new(w1[0]), Mul128::new(w1[1]), Mul128::new(w1[2]), Mul128::new(w1[3])];
        radix8_outer_128(values, m4, m2, m1, w4 == 0 && w2_0 == 0 && w1[0] == 0);
        assert_eq!(values.len(), len);
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
        let half = dst.len();
        assert!(half >= 8 && half.is_multiple_of(8) && novel_low.len() == half);
        assert!(novel_high.len() <= half);
        let e = half / 8;
        let qcount = novel_high.len();
        let pw_eff = Mul128::new(w_eff);
        let m4 = Mul128::new(w4);
        let m2 = [Mul128::new(w2[0]), Mul128::new(w2[1])];
        let m1 = [Mul128::new(w1[0]), Mul128::new(w1[1]), Mul128::new(w1[2]), Mul128::new(w1[3])];
        if qcount <= e {
            return copy_radix8_128_short(novel_low, novel_high, dst, pw_eff, m4, m2, m1);
        }
        let nl = novel_low.as_ptr();
        let nh = novel_high.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..e {
            let mut v = [vector::zero(); 8];
            for (k, slot) in v.iter_mut().enumerate() {
                let idx = k * e + i;
                let lo = vector::load(unsafe { (*nl.add(idx)).0 });
                *slot = if idx < qcount {
                    pw_eff.mul_xor_v(lo, vector::load(unsafe { (*nh.add(idx)).0 }))
                } else {
                    lo
                };
            }
            radix8_8lanes(&mut v, m4, m2[0], m2[1], m1);
            for (k, &val) in v.iter().enumerate() {
                unsafe { (*dp.add(k * e + i)).write(Self(vector::store(val))) };
            }
        }
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
        let pw_eff = Mul128::new(w_eff);
        let p = [Mul128::new(w), Mul128::new(w0), Mul128::new(w1)];
        let (d01, d23) = dst.split_at_mut(2 * q);
        let (d0, d1) = d01.split_at_mut(q);
        let (d2, d3) = d23.split_at_mut(q);
        if qcount <= q {
            for i in 0..qcount {
                let a0 = pw_eff.mul_xor_v(vector::load(novel_low[i].0), vector::load(novel_high[i].0));
                let a = [
                    a0,
                    vector::load(novel_low[q + i].0),
                    vector::load(novel_low[2 * q + i].0),
                    vector::load(novel_low[3 * q + i].0),
                ];
                write_radix4_128(d0, d1, d2, d3, i, a, p);
            }
            for i in qcount..q {
                let a = [
                    vector::load(novel_low[i].0),
                    vector::load(novel_low[q + i].0),
                    vector::load(novel_low[2 * q + i].0),
                    vector::load(novel_low[3 * q + i].0),
                ];
                write_radix4_128(d0, d1, d2, d3, i, a, p);
            }
            return;
        }
        for i in 0..q {
            let load_at = |idx: usize| -> V {
                let lo = vector::load(novel_low[idx].0);
                if idx < qcount { pw_eff.mul_xor_v(lo, vector::load(novel_high[idx].0)) } else { lo }
            };
            let a = [load_at(i), load_at(q + i), load_at(2 * q + i), load_at(3 * q + i)];
            write_radix4_128(d0, d1, d2, d3, i, a, p);
        }
    }

    fn copy_half_uninit(novel_low: &[Self], novel_high: &[Self], dst: &mut [MaybeUninit<Self>], w_eff: u128) {
        let half = dst.len();
        assert!(novel_low.len() == half && novel_high.len() <= half);
        let qcount = novel_high.len();
        let p = Mul128::new(w_eff);
        for i in 0..qcount {
            let v = p.mul_xor_v(vector::load(novel_low[i].0), vector::load(novel_high[i].0));
            dst[i].write(Self(vector::store(v)));
        }
        for i in qcount..half {
            dst[i].write(novel_low[i]);
        }
        assert_eq!(dst.len(), half);
    }

    fn eval_horner_base(coefficients: &[Self], x: u128) -> Self {
        let len = coefficients.len();
        if len < 16 {
            let px = Mul128::new(x);
            let acc =
                coefficients.iter().rev().fold(vector::zero(), |acc, c| px.mul_xor_v(vector::load(c.0), acc));
            return Self(vector::store(acc));
        }
        let x2 = vector::square(x);
        let x4 = vector::square(x2);
        let x8 = vector::square(x4);
        let px8 = Mul128::new(x8);
        let mut acc = [vector::zero(); 8];
        let rem = len % 8;
        let full = len - rem;
        for k in 0..rem {
            acc[k] = vector::load(coefficients[full + k].0);
        }
        let mut pos = full;
        for _ in 0..(full / 8) {
            pos -= 8;
            for k in 0..8 {
                acc[k] = px8.mul_xor_v(vector::load(coefficients[pos + k].0), acc[k]);
            }
        }
        assert_eq!(pos, 0);
        let px = Mul128::new(x);
        let px2 = Mul128::new(x2);
        let px4 = Mul128::new(x4);
        let a0 = px.mul_xor_v(acc[0], acc[1]);
        let a1 = px.mul_xor_v(acc[2], acc[3]);
        let a2 = px.mul_xor_v(acc[4], acc[5]);
        let a3 = px.mul_xor_v(acc[6], acc[7]);
        let b0 = px2.mul_xor_v(a0, a1);
        let b1 = px2.mul_xor_v(a2, a3);
        assert!(len >= 16);
        Self(vector::store(px4.mul_xor_v(b0, b1)))
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
