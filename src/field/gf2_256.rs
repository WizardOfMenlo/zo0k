use super::vector::{self, Acc128, Mul128, Mul256, V};
use super::{BinaryField, Field, binary_inverse, impl_binary_field_operators};
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gf2_256(u128, u128);

impl Gf2_256 {
    pub const fn new(a: u128, b: u128) -> Self {
        Self(a, b)
    }
}

#[derive(Clone, Copy)]
struct WideAcc256 {
    ac: Acc128,
    bd: Acc128,
    cross: Acc128,
}

impl WideAcc256 {
    #[inline(always)]
    fn zero() -> Self {
        Self { ac: Acc128::zero(), bd: Acc128::zero(), cross: Acc128::zero() }
    }

    #[inline(always)]
    fn add_product(&mut self, x: Gf2_256, y: Gf2_256) {
        let (a, b) = (vector::load(x.0), vector::load(x.1));
        let (c, d) = (vector::load(y.0), vector::load(y.1));
        self.ac.add_product(a, c);
        self.bd.add_product(b, d);
        self.cross.add_product(vector::xor(a, b), vector::xor(c, d));
    }

    #[inline(always)]
    fn add_product_v(&mut self, a: V, b: V, c: V, d: V) {
        self.ac.add_product(a, c);
        self.bd.add_product(b, d);
        self.cross.add_product(vector::xor(a, b), vector::xor(c, d));
    }

    #[inline(always)]
    fn merge(self, other: Self) -> Self {
        Self {
            ac: self.ac.merge(other.ac),
            bd: self.bd.merge(other.bd),
            cross: self.cross.merge(other.cross),
        }
    }

    #[inline(always)]
    fn finish(self) -> Gf2_256 {
        let (lo, hi) = vector::reduce_256(self.ac.to_wide(), self.bd.to_wide(), self.cross.to_wide());
        Gf2_256(vector::store(lo), vector::store(hi))
    }
}

#[inline(always)]
fn load_256(x: Gf2_256) -> (V, V) {
    (vector::load(x.0), vector::load(x.1))
}

#[inline(always)]
fn store_256(x: (V, V)) -> Gf2_256 {
    Gf2_256(vector::store(x.0), vector::store(x.1))
}

fn eval_horner_blocked_base_256(coefficients: &[Gf2_256], x: u128) -> Gf2_256 {
    const BLOCK: usize = 32;
    let len = coefficients.len();
    assert!(len >= 2 * BLOCK);
    let px = Mul128::new(x);
    let mut p = [vector::zero(); BLOCK];
    let mut ps = [vector::zero(); BLOCK];
    let mut cur = vector::load(1);
    for k in 0..BLOCK {
        p[k] = cur;
        ps[k] = Acc128::precompute_aux(cur);
        cur = px.mul_v(cur);
    }
    let px32 = Mul128::new(vector::store(cur));
    let rem = len % BLOCK;
    let full = len - rem;
    let mut t0 = Acc128::zero();
    let mut t1 = Acc128::zero();
    for k in 0..rem {
        t0.add_product_pre(vector::load(coefficients[full + k].0), p[k], ps[k]);
        t1.add_product_pre(vector::load(coefficients[full + k].1), p[k], ps[k]);
    }
    let mut acc0 = t0.reduce();
    let mut acc1 = t1.reduce();
    let (blocks, _) = coefficients[..full].as_chunks::<BLOCK>();
    assert_eq!(blocks.len() * BLOCK, full);
    for chunk in blocks.iter().rev() {
        let mut a0 = Acc128::zero();
        let mut b0 = Acc128::zero();
        let mut a1 = Acc128::zero();
        let mut b1 = Acc128::zero();
        for j in 0..(BLOCK / 4) {
            let k = 4 * j;
            a0.add_product_pre(vector::load(chunk[k].0), p[k], ps[k]);
            b0.add_product_pre(vector::load(chunk[k].1), p[k], ps[k]);
            a1.add_product_pre(vector::load(chunk[k + 1].0), p[k + 1], ps[k + 1]);
            b1.add_product_pre(vector::load(chunk[k + 1].1), p[k + 1], ps[k + 1]);
            a0.add_product_pre(vector::load(chunk[k + 2].0), p[k + 2], ps[k + 2]);
            b0.add_product_pre(vector::load(chunk[k + 2].1), p[k + 2], ps[k + 2]);
            a1.add_product_pre(vector::load(chunk[k + 3].0), p[k + 3], ps[k + 3]);
            b1.add_product_pre(vector::load(chunk[k + 3].1), p[k + 3], ps[k + 3]);
        }
        acc0 = px32.mul_xor_v(a0.merge(a1).reduce(), acc0);
        acc1 = px32.mul_xor_v(b0.merge(b1).reduce(), acc1);
    }
    store_256((acc0, acc1))
}

fn eval_horner_blocked_full_256(coefficients: &[Gf2_256], x: Gf2_256) -> Gf2_256 {
    const BLOCK: usize = 32;
    let len = coefficients.len();
    assert!(len >= 2 * BLOCK);
    let px = Mul256::new(x.0, x.1);
    let mut p0 = [vector::zero(); BLOCK];
    let mut p1 = [vector::zero(); BLOCK];
    let mut p01 = [vector::zero(); BLOCK];
    let mut p0s = [vector::zero(); BLOCK];
    let mut p1s = [vector::zero(); BLOCK];
    let mut p01s = [vector::zero(); BLOCK];
    let mut cur = (vector::load(1), vector::zero());
    for k in 0..BLOCK {
        let sum = vector::xor(cur.0, cur.1);
        p0[k] = cur.0;
        p1[k] = cur.1;
        p01[k] = sum;
        p0s[k] = Acc128::precompute_aux(cur.0);
        p1s[k] = Acc128::precompute_aux(cur.1);
        p01s[k] = Acc128::precompute_aux(sum);
        cur = px.mul_pair(cur.0, cur.1);
    }
    let px32 = Mul256::new(vector::store(cur.0), vector::store(cur.1));
    let rem = len % BLOCK;
    let full = len - rem;
    let mut tac = Acc128::zero();
    let mut tbd = Acc128::zero();
    let mut tcr = Acc128::zero();
    for k in 0..rem {
        let (a, b) = load_256(coefficients[full + k]);
        tac.add_product_pre(a, p0[k], p0s[k]);
        tbd.add_product_pre(b, p1[k], p1s[k]);
        tcr.add_product_pre(vector::xor(a, b), p01[k], p01s[k]);
    }
    let mut acc = vector::reduce_256(tac.to_wide(), tbd.to_wide(), tcr.to_wide());
    let (blocks, _) = coefficients[..full].as_chunks::<BLOCK>();
    assert_eq!(blocks.len() * BLOCK, full);
    for chunk in blocks.iter().rev() {
        let mut ac = Acc128::zero();
        let mut bd = Acc128::zero();
        let mut cr = Acc128::zero();
        for k in 0..BLOCK {
            let (a, b) = load_256(chunk[k]);
            ac.add_product_pre(a, p0[k], p0s[k]);
            bd.add_product_pre(b, p1[k], p1s[k]);
            cr.add_product_pre(vector::xor(a, b), p01[k], p01s[k]);
        }
        let (b_lo, b_hi) = vector::reduce_256(ac.to_wide(), bd.to_wide(), cr.to_wide());
        acc = px32.mul_xor_pair(b_lo, b_hi, acc.0, acc.1);
    }
    store_256(acc)
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

    #[inline(always)]
    fn write_bytes(self, bytes: &mut [u8]) {
        let (a, b) = bytes.split_at_mut(16);
        a.copy_from_slice(&self.0.to_le_bytes());
        b.copy_from_slice(&self.1.to_le_bytes());
    }

    #[inline(always)]
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
        if self.1 == 0 {
            return Self(vector::square(self.0), 0);
        }
        use vector::{load, store, xor};
        let (a, b) = (vector::square(self.0), vector::square(self.1));
        Self(store(xor(load(a), vector::mul_x121(load(b)))), b)
    }

    fn pow(mut self, mut exponent: u64) -> Self {
        if self.1 == 0 {
            let mut base = self.0;
            let mut res = 1u128;
            for _ in 0..64 {
                if exponent == 0 {
                    break;
                }
                if exponent & 1 == 1 {
                    res = vector::mul(res, base);
                }
                base = vector::square(base);
                exponent >>= 1;
            }
            assert_eq!(exponent, 0);
            return Self(res, 0);
        }
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
        let mut acc0 = WideAcc256::zero();
        let mut acc1 = WideAcc256::zero();
        let (ac, ar) = a.as_chunks::<2>();
        let (bc, br) = b.as_chunks::<2>();
        for (x, y) in ac.iter().zip(bc) {
            acc0.add_product(x[0], y[0]);
            acc1.add_product(x[1], y[1]);
        }
        let mut acc = acc0.merge(acc1);
        for (x, y) in ar.iter().zip(br) {
            acc.add_product(*x, *y);
        }
        assert_eq!(ar.len(), br.len());
        acc.finish()
    }

    fn round_coefficients_slice(
        a_low: &[Self],
        a_high: &[Self],
        b_low: &[Self],
        b_high: &[Self],
    ) -> (Self, Self) {
        let len = a_low.len();
        assert!(a_high.len() == len && b_low.len() == len && b_high.len() == len);
        let mut c_acc = WideAcc256::zero();
        let mut q_acc = WideAcc256::zero();
        let (al_c, al_r) = a_low.as_chunks::<2>();
        let (ah_c, ah_r) = a_high.as_chunks::<2>();
        let (bl_c, bl_r) = b_low.as_chunks::<2>();
        let (bh_c, bh_r) = b_high.as_chunks::<2>();
        for (((al, ah), bl), bh) in al_c.iter().zip(ah_c).zip(bl_c).zip(bh_c) {
            for k in 0..2 {
                let (al0, al1) = load_256(al[k]);
                let (ah0, ah1) = load_256(ah[k]);
                let (bl0, bl1) = load_256(bl[k]);
                let (bh0, bh1) = load_256(bh[k]);
                c_acc.add_product_v(al0, al1, bl0, bl1);
                q_acc.add_product_v(
                    vector::xor(ah0, al0),
                    vector::xor(ah1, al1),
                    vector::xor(bh0, bl0),
                    vector::xor(bh1, bl1),
                );
            }
        }
        for i in 0..al_r.len() {
            let (al0, al1) = load_256(al_r[i]);
            let (ah0, ah1) = load_256(ah_r[i]);
            let (bl0, bl1) = load_256(bl_r[i]);
            let (bh0, bh1) = load_256(bh_r[i]);
            c_acc.add_product_v(al0, al1, bl0, bl1);
            q_acc.add_product_v(
                vector::xor(ah0, al0),
                vector::xor(ah1, al1),
                vector::xor(bh0, bl0),
                vector::xor(bh1, bl1),
            );
        }
        assert_eq!(al_r.len(), bh_r.len());
        (c_acc.finish(), q_acc.finish())
    }

    fn fold_slice(low: &mut [Self], high: &[Self], challenge: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        if challenge == Self::ZERO {
            return;
        }
        let p = Mul256::new(challenge.0, challenge.1);
        let (lc, lr) = low.as_chunks_mut::<2>();
        let (hc, hr) = high.as_chunks::<2>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..2 {
                let (l0, l1) = load_256(lo[k]);
                let (h0, h1) = load_256(hi[k]);
                lo[k] = store_256(p.mul_xor_pair(l0, l1, vector::xor(h0, l0), vector::xor(h1, l1)));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(hr) {
            let (l0, l1) = load_256(*lo);
            let (h0, h1) = load_256(*hi);
            *lo = store_256(p.mul_xor_pair(l0, l1, vector::xor(h0, l0), vector::xor(h1, l1)));
        }
        assert_eq!(lr.len(), hr.len());
    }

    fn eq_extend_uninit(low: &mut [Self], high: &mut [MaybeUninit<Self>], coordinate: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        let p = Mul256::new(coordinate.0, coordinate.1);
        let (lc, lr) = low.as_chunks_mut::<2>();
        let (hc, hr) = high.as_chunks_mut::<2>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..2 {
                let (l0, l1) = load_256(lo[k]);
                let (h0, h1) = p.mul_pair(l0, l1);
                hi[k].write(store_256((h0, h1)));
                lo[k] = store_256((vector::xor(l0, h0), vector::xor(l1, h1)));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let (l0, l1) = load_256(*lo);
            let (h0, h1) = p.mul_pair(l0, l1);
            hi.write(store_256((h0, h1)));
            *lo = store_256((vector::xor(l0, h0), vector::xor(l1, h1)));
        }
        assert_eq!(lr.len(), hr.len());
    }

    fn add_scaled_slice(dst: &mut [Self], src: &[Self], scale: Self) {
        let len = dst.len();
        assert_eq!(len, src.len());
        if scale == Self::ZERO {
            return;
        }
        let p = Mul256::new(scale.0, scale.1);
        let (dc, dr) = dst.as_chunks_mut::<2>();
        let (sc, sr) = src.as_chunks::<2>();
        for (d, s) in dc.iter_mut().zip(sc) {
            for k in 0..2 {
                let (d0, d1) = load_256(d[k]);
                let (s0, s1) = load_256(s[k]);
                d[k] = store_256(p.mul_xor_pair(d0, d1, s0, s1));
            }
        }
        for (d, s) in dr.iter_mut().zip(sr) {
            let (d0, d1) = load_256(*d);
            let (s0, s1) = load_256(*s);
            *d = store_256(p.mul_xor_pair(d0, d1, s0, s1));
        }
        assert_eq!(dr.len(), sr.len());
    }

    fn scale_slice(dst: &mut [Self], scale: Self) {
        let len = dst.len();
        if scale == Self::ONE {
            return;
        }
        if scale == Self::ZERO {
            dst.fill(Self::ZERO);
            return;
        }
        let p = Mul256::new(scale.0, scale.1);
        let (dc, dr) = dst.as_chunks_mut::<2>();
        for d in dc {
            for entry in d.iter_mut() {
                let (v0, v1) = load_256(*entry);
                *entry = store_256(p.mul_pair(v0, v1));
            }
        }
        for d in &mut *dr {
            let (v0, v1) = load_256(*d);
            *d = store_256(p.mul_pair(v0, v1));
        }
        assert!(len >= dr.len());
        assert!(dr.len() < 2);
    }

    fn scale_to_uninit(dst: &mut [MaybeUninit<Self>], src: &[Self], scale: Self) {
        let len = dst.len();
        assert_eq!(len, src.len());
        if scale == Self::ZERO {
            for d in dst {
                d.write(Self::ZERO);
            }
            return;
        }
        let p = Mul256::new(scale.0, scale.1);
        let (dc, dr) = dst.as_chunks_mut::<2>();
        let (sc, sr) = src.as_chunks::<2>();
        for (d, s) in dc.iter_mut().zip(sc) {
            for k in 0..2 {
                let (s0, s1) = load_256(s[k]);
                d[k].write(store_256(p.mul_pair(s0, s1)));
            }
        }
        for (d, s) in dr.iter_mut().zip(sr) {
            let (s0, s1) = load_256(*s);
            d.write(store_256(p.mul_pair(s0, s1)));
        }
        assert_eq!(dr.len(), sr.len());
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_and_round_slice(
        a0: &mut [Self],
        a1: &mut [Self],
        a2: &[Self],
        a3: &[Self],
        b0: &mut [Self],
        b1: &mut [Self],
        b2: &[Self],
        b3: &[Self],
        challenge: Self,
    ) -> (Self, Self) {
        let len = a0.len();
        assert!(a1.len() == len && a2.len() == len && a3.len() == len);
        assert!(b0.len() == len && b1.len() == len && b2.len() == len && b3.len() == len);
        if challenge == Self::ZERO {
            return Self::round_coefficients_slice(a0, a1, b0, b1);
        }
        let p = Mul256::new(challenge.0, challenge.1);
        let mut c_acc = WideAcc256::zero();
        let mut q_acc = WideAcc256::zero();
        for i in 0..len {
            let fa0 = fold_one_256(a0[i], a2[i], p);
            let fa1 = fold_one_256(a1[i], a3[i], p);
            let fb0 = fold_one_256(b0[i], b2[i], p);
            let fb1 = fold_one_256(b1[i], b3[i], p);
            a0[i] = store_256(fa0);
            a1[i] = store_256(fa1);
            b0[i] = store_256(fb0);
            b1[i] = store_256(fb1);
            c_acc.add_product_v(fa0.0, fa0.1, fb0.0, fb0.1);
            q_acc.add_product_v(
                vector::xor(fa1.0, fa0.0),
                vector::xor(fa1.1, fa0.1),
                vector::xor(fb1.0, fb0.0),
                vector::xor(fb1.1, fb0.1),
            );
        }
        (c_acc.finish(), q_acc.finish())
    }

    #[allow(clippy::too_many_arguments)]
    fn add_scaled_4_slice(
        dst: &mut [Self],
        s0: &[Self],
        w0: Self,
        s1: &[Self],
        w1: Self,
        s2: &[Self],
        w2: Self,
        s3: &[Self],
        w3: Self,
    ) {
        let len = dst.len();
        assert!(s0.len() == len && s1.len() == len && s2.len() == len && s3.len() == len);
        let p0 = Mul256::new(w0.0, w0.1);
        let p1 = Mul256::new(w1.0, w1.1);
        let p2 = Mul256::new(w2.0, w2.1);
        let p3 = Mul256::new(w3.0, w3.1);
        for i in 0..len {
            let (d0, d1) = load_256(dst[i]);
            let (a0, a1) = load_256(s0[i]);
            let (b0, b1) = load_256(s1[i]);
            let (c0, c1) = load_256(s2[i]);
            let (e0, e1) = load_256(s3[i]);
            let (r0_lo, r0_hi) = p0.mul_lm_pair(a0, a1);
            let (r1_lo, r1_hi) = p1.mul_lm_pair(b0, b1);
            let (r2_lo, r2_hi) = p2.mul_lm_pair(c0, c1);
            let (r3_lo, r3_hi) = p3.mul_lm_pair(e0, e1);
            let lo_l = vector::xor3(d0, vector::xor(r0_lo.0, r1_lo.0), vector::xor(r2_lo.0, r3_lo.0));
            let lo_m = vector::xor(vector::xor(r0_lo.1, r1_lo.1), vector::xor(r2_lo.1, r3_lo.1));
            let hi_l = vector::xor3(d1, vector::xor(r0_hi.0, r1_hi.0), vector::xor(r2_hi.0, r3_hi.0));
            let hi_m = vector::xor(vector::xor(r0_hi.1, r1_hi.1), vector::xor(r2_hi.1, r3_hi.1));
            dst[i] = store_256((vector::reduce_lm(lo_l, lo_m), vector::reduce_lm(hi_l, hi_m)));
        }
        assert_eq!(dst.len(), s3.len());
    }

    fn eval_horner_slice(coefficients: &[Self], x: Self) -> Self {
        if x.1 == 0 {
            return Self::eval_horner_base(coefficients, x.0);
        }
        let len = coefficients.len();
        if len >= 64 {
            return eval_horner_blocked_full_256(coefficients, x);
        }
        let px = Mul256::new(x.0, x.1);
        let acc = coefficients.iter().rev().fold((vector::zero(), vector::zero()), |acc, c| {
            let (c0, c1) = load_256(*c);
            px.mul_xor_pair(c0, c1, acc.0, acc.1)
        });
        assert!(len < 64);
        store_256(acc)
    }

    fn add_powers_slice(target: &mut [Self], x: Self, scale: Self) {
        let len = target.len();
        if len == 0 {
            return;
        }
        if x.1 == 0 {
            return add_powers_base_256(target, x.0, scale);
        }
        let px = Mul256::new(x.0, x.1);
        let mut s = [load_256(scale); 4];
        for k in 1..4 {
            s[k] = px.mul_pair(s[k - 1].0, s[k - 1].1);
        }
        let x4 = x.square().square();
        let px4 = Mul256::new(x4.0, x4.1);
        let (chunks, rem) = target.as_chunks_mut::<4>();
        for chunk in chunks {
            for k in 0..4 {
                let (t0, t1) = load_256(chunk[k]);
                chunk[k] = store_256((vector::xor(t0, s[k].0), vector::xor(t1, s[k].1)));
                s[k] = px4.mul_pair(s[k].0, s[k].1);
            }
        }
        let mut cur = s[0];
        for entry in &mut *rem {
            let (t0, t1) = load_256(*entry);
            *entry = store_256((vector::xor(t0, cur.0), vector::xor(t1, cur.1)));
            cur = px.mul_pair(cur.0, cur.1);
        }
        assert!(rem.len() < 4);
    }

    fn xor_slice(dst: &mut [Self], src: &[Self]) {
        let len = dst.len();
        assert_eq!(len, src.len());
        let (dc, dr) = dst.as_chunks_mut::<4>();
        let (sc, sr) = src.as_chunks::<4>();
        for (d, s) in dc.iter_mut().zip(sc) {
            for k in 0..4 {
                let (d0, d1) = load_256(d[k]);
                let (s0, s1) = load_256(s[k]);
                d[k] = store_256((vector::xor(d0, s0), vector::xor(d1, s1)));
            }
        }
        for (d, s) in dr.iter_mut().zip(sr) {
            let (d0, d1) = load_256(*d);
            let (s0, s1) = load_256(*s);
            *d = store_256((vector::xor(d0, s0), vector::xor(d1, s1)));
        }
        assert_eq!(dr.len(), sr.len());
    }
}

#[inline(always)]
fn fold_one_256(lo: Gf2_256, hi: Gf2_256, p: Mul256) -> (V, V) {
    let (l0, l1) = load_256(lo);
    let (h0, h1) = load_256(hi);
    p.mul_xor_pair(l0, l1, vector::xor(h0, l0), vector::xor(h1, l1))
}

fn add_powers_base_256(target: &mut [Gf2_256], x: u128, scale: Gf2_256) {
    let len = target.len();
    assert!(len >= 1);
    let px = Mul128::new(x);
    let mut s = [load_256(scale); 4];
    for k in 1..4 {
        s[k] = (px.mul_v(s[k - 1].0), px.mul_v(s[k - 1].1));
    }
    let px4 = Mul128::new(vector::square(vector::square(x)));
    let (chunks, rem) = target.as_chunks_mut::<4>();
    for chunk in chunks {
        for k in 0..4 {
            let (t0, t1) = load_256(chunk[k]);
            chunk[k] = store_256((vector::xor(t0, s[k].0), vector::xor(t1, s[k].1)));
            s[k] = (px4.mul_v(s[k].0), px4.mul_v(s[k].1));
        }
    }
    let mut cur = s[0];
    for entry in &mut *rem {
        let (t0, t1) = load_256(*entry);
        *entry = store_256((vector::xor(t0, cur.0), vector::xor(t1, cur.1)));
        cur = (px.mul_v(cur.0), px.mul_v(cur.1));
    }
    assert!(rem.len() < 4);
}

#[inline(always)]
fn butterfly_step_256(r: (V, V), q: (V, V), p: Mul128) -> ((V, V), (V, V)) {
    let nr0 = p.mul_xor_v(r.0, q.0);
    let nr1 = p.mul_xor_v(r.1, q.1);
    ((nr0, nr1), (vector::xor(q.0, nr0), vector::xor(q.1, nr1)))
}

#[inline(always)]
fn transpose_butterfly_step_256(u: (V, V), v: (V, V), p: Mul128) -> ((V, V), (V, V)) {
    let (nu0, nv0) = vector::transpose_butterfly_step_v(u.0, v.0, p);
    let (nu1, nv1) = vector::transpose_butterfly_step_v(u.1, v.1, p);
    ((nu0, nu1), (nv0, nv1))
}

fn butterfly_radix4_nonzero(
    v0: &mut [Gf2_256],
    v1: &mut [Gf2_256],
    v2: &mut [Gf2_256],
    v3: &mut [Gf2_256],
    pw: Mul128,
    pw0: Mul128,
    pw1: Mul128,
) {
    let len = v0.len();
    assert!(v1.len() == len && v2.len() == len && v3.len() == len);
    for i in 0..len {
        let (a0, a2) = butterfly_step_256(load_256(v0[i]), load_256(v2[i]), pw);
        let (a1, a3) = butterfly_step_256(load_256(v1[i]), load_256(v3[i]), pw);
        let (b0, b1) = butterfly_step_256(a0, a1, pw0);
        let (b2, b3) = butterfly_step_256(a2, a3, pw1);
        v0[i] = store_256(b0);
        v1[i] = store_256(b1);
        v2[i] = store_256(b2);
        v3[i] = store_256(b3);
    }
    assert_eq!(v0.len(), len);
}

fn butterfly_radix4_zero_prefix(
    v0: &mut [Gf2_256],
    v1: &mut [Gf2_256],
    v2: &mut [Gf2_256],
    v3: &mut [Gf2_256],
    pw1: Mul128,
) {
    let len = v0.len();
    assert!(v1.len() == len && v2.len() == len && v3.len() == len);
    let (c0, r0) = v0.as_chunks_mut::<2>();
    let (c1, r1) = v1.as_chunks_mut::<2>();
    let (c2, r2) = v2.as_chunks_mut::<2>();
    let (c3, r3) = v3.as_chunks_mut::<2>();
    for i in 0..c0.len() {
        for k in 0..2 {
            let a0 = load_256(c0[i][k]);
            let a1 = load_256(c1[i][k]);
            let t2 = load_256(c2[i][k]);
            let t3 = load_256(c3[i][k]);
            let a2 = (vector::xor(t2.0, a0.0), vector::xor(t2.1, a0.1));
            let a3 = (vector::xor(t3.0, a1.0), vector::xor(t3.1, a1.1));
            let (b2, b3) = butterfly_step_256(a2, a3, pw1);
            c1[i][k] = store_256((vector::xor(a1.0, a0.0), vector::xor(a1.1, a0.1)));
            c2[i][k] = store_256(b2);
            c3[i][k] = store_256(b3);
        }
    }
    for i in 0..r0.len() {
        let a0 = load_256(r0[i]);
        let a1 = load_256(r1[i]);
        let t2 = load_256(r2[i]);
        let t3 = load_256(r3[i]);
        let a2 = (vector::xor(t2.0, a0.0), vector::xor(t2.1, a0.1));
        let a3 = (vector::xor(t3.0, a1.0), vector::xor(t3.1, a1.1));
        let (b2, b3) = butterfly_step_256(a2, a3, pw1);
        r1[i] = store_256((vector::xor(a1.0, a0.0), vector::xor(a1.1, a0.1)));
        r2[i] = store_256(b2);
        r3[i] = store_256(b3);
    }
    assert_eq!(r0.len(), r3.len());
}

fn radix8_octants_256(
    octants: [&mut [Gf2_256]; 8],
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
    zero_prefix: bool,
) {
    let e = octants[0].len();
    assert!(octants.iter().all(|s| s.len() == e));
    if zero_prefix {
        let m2_1 = m2_0.xor(p2);
        let m1_1 = m1_0.xor(p2);
        let m1_2 = m1_0.xor(p4);
        let m1_3 = m1_1.xor(p4);
        for i in 0..e {
            let mut v0: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].0));
            vector::radix8_8lanes_zero_prefix(&mut v0, m2_1, m1_1, m1_2, m1_3);
            for k in 1..8 {
                octants[k][i].0 = vector::store(v0[k]);
            }
            let mut v1: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].1));
            vector::radix8_8lanes_zero_prefix(&mut v1, m2_1, m1_1, m1_2, m1_3);
            for k in 1..8 {
                octants[k][i].1 = vector::store(v1[k]);
            }
        }
    } else {
        for i in 0..e {
            let mut v0: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].0));
            vector::radix8_8lanes_compact(&mut v0, m4, m2_0, m1_0, p2, p4);
            for k in 0..8 {
                octants[k][i].0 = vector::store(v0[k]);
            }
            let mut v1: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].1));
            vector::radix8_8lanes_compact(&mut v1, m4, m2_0, m1_0, p2, p4);
            for k in 0..8 {
                octants[k][i].1 = vector::store(v1[k]);
            }
        }
    }
    assert_eq!(octants[7].len(), e);
}

#[allow(clippy::too_many_arguments)]
fn copy_radix8_full_256(
    novel_low: [&[Gf2_256]; 8],
    novel_high: [&[Gf2_256]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_256>]],
    len: usize,
    pw_eff: Mul128,
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
) {
    let octants = dst.len();
    assert!((1..=8).contains(&octants));
    let tiles = len / 8;
    let mut tile0 = [[vector::zero(); 8]; 8];
    let mut tile1 = [[vector::zero(); 8]; 8];
    for tile in 0..tiles {
        let base = tile * 8;
        for k in 0..8 {
            let lo = &novel_low[k][base..base + 8];
            let hi = &novel_high[k][base..base + 8];
            for t in 0..8 {
                let (l0, l1) = load_256(lo[t]);
                let (h0, h1) = load_256(hi[t]);
                tile0[t][k] = pw_eff.mul_xor_v(l0, h0);
                tile1[t][k] = pw_eff.mul_xor_v(l1, h1);
            }
        }
        if octants == 8 {
            vector::radix8_rows_compact(&mut tile0, m4, m2_0, m1_0, p2, p4);
            vector::radix8_rows_compact(&mut tile1, m4, m2_0, m1_0, p2, p4);
        } else {
            vector::radix8_rows_partial(&mut tile0, m4, m2_0, m1_0, p2, p4, octants);
            vector::radix8_rows_partial(&mut tile1, m4, m2_0, m1_0, p2, p4, octants);
        }
        for (k, out_oct) in dst.iter_mut().enumerate() {
            let out = &mut out_oct[base..base + 8];
            for t in 0..8 {
                out[t].write(store_256((tile0[t][k], tile1[t][k])));
            }
        }
    }
    let rem_start = tiles * 8;
    for i in rem_start..len {
        let mut v0: [V; 8] = std::array::from_fn(|k| {
            pw_eff.mul_xor_v(vector::load(novel_low[k][i].0), vector::load(novel_high[k][i].0))
        });
        let mut v1: [V; 8] = std::array::from_fn(|k| {
            pw_eff.mul_xor_v(vector::load(novel_low[k][i].1), vector::load(novel_high[k][i].1))
        });
        vector::radix8_8lanes_partial(&mut v0, m4, m2_0, m1_0, p2, p4, octants);
        vector::radix8_8lanes_partial(&mut v1, m4, m2_0, m1_0, p2, p4, octants);
        for (k, out_oct) in dst.iter_mut().enumerate() {
            out_oct[i].write(store_256((v0[k], v1[k])));
        }
    }
    assert!(rem_start <= len);
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn copy_radix8_pure_256(
    novel_low: [&[Gf2_256]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_256>]],
    start: usize,
    end: usize,
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
) {
    let octants = dst.len();
    assert!((1..=8).contains(&octants) && start <= end);
    assert!(novel_low.iter().all(|s| s.len() >= end) && dst.iter().all(|s| s.len() >= end));
    if octants == 8 {
        let m2 = [m2_0, m2_0.xor(p2)];
        let m1_1 = m1_0.xor(p2);
        let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
        let dp: [*mut Gf2_256; 8] = std::array::from_fn(|k| dst[k].as_mut_ptr().cast::<Gf2_256>());
        for i in start..end {
            let mut v0: [V; 8] = std::array::from_fn(|k| vector::load(novel_low[k][i].0));
            vector::radix8_8lanes_twiddles(&mut v0, m4, m2, m1);
            for k in 0..8 {
                unsafe { std::ptr::write(&raw mut (*dp[k].add(i)).0, vector::store(v0[k])) };
            }
            let mut v1: [V; 8] = std::array::from_fn(|k| vector::load(novel_low[k][i].1));
            vector::radix8_8lanes_twiddles(&mut v1, m4, m2, m1);
            for k in 0..8 {
                unsafe { std::ptr::write(&raw mut (*dp[k].add(i)).1, vector::store(v1[k])) };
            }
        }
        return;
    }
    for i in start..end {
        let mut v0: [V; 8] = std::array::from_fn(|k| vector::load(novel_low[k][i].0));
        vector::radix8_8lanes_partial(&mut v0, m4, m2_0, m1_0, p2, p4, octants);
        for (k, out_oct) in dst.iter_mut().enumerate() {
            let ptr = out_oct.as_mut_ptr().cast::<Gf2_256>();
            unsafe { std::ptr::write(&raw mut (*ptr.add(i)).0, vector::store(v0[k])) };
        }
        let mut v1: [V; 8] = std::array::from_fn(|k| vector::load(novel_low[k][i].1));
        vector::radix8_8lanes_partial(&mut v1, m4, m2_0, m1_0, p2, p4, octants);
        for (k, out_oct) in dst.iter_mut().enumerate() {
            let ptr = out_oct.as_mut_ptr().cast::<Gf2_256>();
            unsafe { std::ptr::write(&raw mut (*ptr.add(i)).1, vector::store(v1[k])) };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_radix8_octants_256(
    novel_low: [&[Gf2_256]; 8],
    novel_high: [&[Gf2_256]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_256>]],
    pw_eff: Mul128,
    m4: Mul128,
    m2_0: Mul128,
    m1_0: Mul128,
    p2: Mul128,
    p4: Mul128,
) {
    let octants = dst.len();
    let e = novel_low[0].len();
    assert!((1..=8).contains(&octants));
    assert!(novel_low.iter().all(|s| s.len() == e) && dst.iter().all(|s| s.len() == e));
    let min_hi = novel_high.iter().map(|s| s.len()).min().unwrap_or(0);
    let max_hi = novel_high.iter().map(|s| s.len()).max().unwrap_or(0);
    if min_hi > 0 {
        copy_radix8_full_256(novel_low, novel_high, dst, min_hi, pw_eff, m4, m2_0, m1_0, p2, p4);
    }
    for i in min_hi..max_hi {
        let mut v0 = [vector::zero(); 8];
        let mut v1 = [vector::zero(); 8];
        for k in 0..8 {
            let lo = load_256(novel_low[k][i]);
            if i < novel_high[k].len() {
                let hi = load_256(novel_high[k][i]);
                v0[k] = pw_eff.mul_xor_v(lo.0, hi.0);
                v1[k] = pw_eff.mul_xor_v(lo.1, hi.1);
            } else {
                v0[k] = lo.0;
                v1[k] = lo.1;
            }
        }
        vector::radix8_8lanes_partial(&mut v0, m4, m2_0, m1_0, p2, p4, octants);
        vector::radix8_8lanes_partial(&mut v1, m4, m2_0, m1_0, p2, p4, octants);
        for (k, out_oct) in dst.iter_mut().enumerate() {
            out_oct[i].write(store_256((v0[k], v1[k])));
        }
    }
    if max_hi < e {
        copy_radix8_pure_256(novel_low, dst, max_hi, e, m4, m2_0, m1_0, p2, p4);
    }
}

#[inline(always)]
fn split_octants_mut_256(values: &mut [Gf2_256]) -> [&mut [Gf2_256]; 8] {
    let e = values.len() / 8;
    let (a, b) = values.split_at_mut(4 * e);
    let (a0, a1) = a.split_at_mut(2 * e);
    let (b0, b1) = b.split_at_mut(2 * e);
    let (v0, v1) = a0.split_at_mut(e);
    let (v2, v3) = a1.split_at_mut(e);
    let (v4, v5) = b0.split_at_mut(e);
    let (v6, v7) = b1.split_at_mut(e);
    [v0, v1, v2, v3, v4, v5, v6, v7]
}

#[inline(always)]
fn split_octants_ref_256(values: &[Gf2_256]) -> [&[Gf2_256]; 8] {
    let e = values.len() / 8;
    [
        &values[0..e],
        &values[e..2 * e],
        &values[2 * e..3 * e],
        &values[3 * e..4 * e],
        &values[4 * e..5 * e],
        &values[5 * e..6 * e],
        &values[6 * e..7 * e],
        &values[7 * e..8 * e],
    ]
}

#[inline(always)]
fn split_high_octants_256(novel_high: &[Gf2_256], e: usize) -> [&[Gf2_256]; 8] {
    std::array::from_fn(|k| {
        let start = k * e;
        if start < novel_high.len() { &novel_high[start..novel_high.len().min(start + e)] } else { &[] }
    })
}

#[inline(always)]
fn write_radix4_256(
    d0: &mut [MaybeUninit<Gf2_256>],
    d1: &mut [MaybeUninit<Gf2_256>],
    d2: &mut [MaybeUninit<Gf2_256>],
    d3: &mut [MaybeUninit<Gf2_256>],
    i: usize,
    a: [(V, V); 4],
    p: [Mul128; 3],
) {
    let (t0, t2) = butterfly_step_256(a[0], a[2], p[0]);
    let (t1, t3) = butterfly_step_256(a[1], a[3], p[0]);
    let (b0, b1) = butterfly_step_256(t0, t1, p[1]);
    let (b2, b3) = butterfly_step_256(t2, t3, p[2]);
    d0[i].write(store_256(b0));
    d1[i].write(store_256(b1));
    d2[i].write(store_256(b2));
    d3[i].write(store_256(b3));
}

#[inline(always)]
fn apply_radix8_block_256<const TRANSPOSE: bool>(
    block: &mut [Gf2_256; 8],
    m4: Mul128,
    m2: [Mul128; 2],
    m1: [Mul128; 4],
) {
    let mut v0 = std::array::from_fn(|j| vector::load(block[j].0));
    if TRANSPOSE {
        vector::transpose_radix8_8lanes_twiddles(&mut v0, m4, m2, m1);
    } else {
        vector::radix8_8lanes_twiddles(&mut v0, m4, m2, m1);
    }
    for j in 0..8 {
        block[j].0 = vector::store(v0[j]);
    }
    let mut v1 = std::array::from_fn(|j| vector::load(block[j].1));
    if TRANSPOSE {
        vector::transpose_radix8_8lanes_twiddles(&mut v1, m4, m2, m1);
    } else {
        vector::radix8_8lanes_twiddles(&mut v1, m4, m2, m1);
    }
    for j in 0..8 {
        block[j].1 = vector::store(v1[j]);
    }
}

fn radix8_steps_256<const TRANSPOSE: bool>(
    values: &mut [Gf2_256],
    base4: u128,
    base2: u128,
    base1: u128,
    steps: &[[(u128, u128); 16]; 3],
) {
    let load_m = |pair: (u128, u128)| Mul128::from_pair(pair.0, pair.1);
    let p2 = load_m(steps[0][0]);
    let p4 = load_m(steps[1][0]);
    let mut m4 = Mul128::new(base4);
    let m2_0 = Mul128::new(base2);
    let mut m2 = [m2_0, m2_0.xor(p2)];
    let m1_0 = Mul128::new(base1);
    let m1_1 = m1_0.xor(p2);
    let mut m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
    let (blocks, remainder) = values.as_chunks_mut::<8>();
    let n = blocks.len();
    assert!(remainder.is_empty());
    assert!((1..=65536).contains(&n));
    for (i, block) in blocks.iter_mut().enumerate() {
        apply_radix8_block_256::<TRANSPOSE>(block, m4, m2, m1);
        let next = i + 1;
        if next < n {
            let tz = (next.trailing_zeros() as usize) & 15;
            let d4 = load_m(steps[0][tz]);
            let d2 = load_m(steps[1][tz]);
            let d1 = load_m(steps[2][tz]);
            m4 = m4.xor(d4);
            m2 = [m2[0].xor(d2), m2[1].xor(d2)];
            m1 = [m1[0].xor(d1), m1[1].xor(d1), m1[2].xor(d1), m1[3].xor(d1)];
        }
    }
}

#[inline(never)]
fn radix8_stride8_chunk_256<const TRANSPOSE: bool>(chunk: &mut [Gf2_256; 64], twiddles: &[Mul128; 5]) {
    assert_eq!(chunk.len(), 64);
    let [m4, m2_0, m1_0, p2, p4] = *twiddles;
    let m2 = [m2_0, m2_0.xor(p2)];
    let m1_1 = m1_0.xor(p2);
    let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
    for off in 0..8 {
        let mut v0: [V; 8] = std::array::from_fn(|k| vector::load(chunk[k * 8 + off].0));
        if TRANSPOSE {
            vector::transpose_radix8_8lanes_twiddles(&mut v0, m4, m2, m1);
        } else {
            vector::radix8_8lanes_twiddles(&mut v0, m4, m2, m1);
        }
        for k in 0..8 {
            chunk[k * 8 + off].0 = vector::store(v0[k]);
        }
    }
    for off in 0..8 {
        let mut v1: [V; 8] = std::array::from_fn(|k| vector::load(chunk[k * 8 + off].1));
        if TRANSPOSE {
            vector::transpose_radix8_8lanes_twiddles(&mut v1, m4, m2, m1);
        } else {
            vector::radix8_8lanes_twiddles(&mut v1, m4, m2, m1);
        }
        for k in 0..8 {
            chunk[k * 8 + off].1 = vector::store(v1[k]);
        }
    }
    assert_eq!(twiddles.len(), 5);
}

fn radix8_stride8_256<const TRANSPOSE: bool>(
    values: &mut [Gf2_256],
    base4: u128,
    base2: u128,
    base1: u128,
    steps: &[[(u128, u128); 16]; 3],
) {
    let load_m = |pair: (u128, u128)| Mul128::from_pair(pair.0, pair.1);
    let mut twiddles = [
        Mul128::new(base4),
        Mul128::new(base2),
        Mul128::new(base1),
        load_m(steps[0][0]),
        load_m(steps[1][0]),
    ];
    let (chunks, remainder) = values.as_chunks_mut::<64>();
    let n = chunks.len();
    assert!(remainder.is_empty());
    assert!((1..=65536).contains(&n));
    for (i, chunk) in chunks.iter_mut().enumerate() {
        radix8_stride8_chunk_256::<TRANSPOSE>(chunk, &twiddles);
        let next = i + 1;
        if next < n {
            let tz = (next.trailing_zeros() as usize) & 15;
            twiddles[0] = twiddles[0].xor(load_m(steps[0][tz]));
            twiddles[1] = twiddles[1].xor(load_m(steps[1][tz]));
            twiddles[2] = twiddles[2].xor(load_m(steps[2][tz]));
        }
    }
}

impl BinaryField for Gf2_256 {
    fn from_base(limb: u128) -> Self {
        Self(limb, 0)
    }

    #[inline(always)]
    fn mul_base(self, w: u128) -> Self {
        let p = Mul128::new(w);
        Self(p.mul(self.0), p.mul(self.1))
    }

    fn butterfly_slice(low: &mut [Self], high: &mut [Self], twiddle: u128) {
        let len = low.len();
        assert_eq!(len, high.len());
        if twiddle == 0 {
            Self::xor_slice(high, low);
            return;
        }
        let p = Mul128::new(twiddle);
        let (lc, lr) = low.as_chunks_mut::<2>();
        let (hc, hr) = high.as_chunks_mut::<2>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..2 {
                let (nr, nq) = butterfly_step_256(load_256(lo[k]), load_256(hi[k]), p);
                lo[k] = store_256(nr);
                hi[k] = store_256(nq);
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let (nr, nq) = butterfly_step_256(load_256(*lo), load_256(*hi), p);
            *lo = store_256(nr);
            *hi = store_256(nq);
        }
        assert_eq!(lr.len(), hr.len());
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
        assert!(remainder.is_empty() && !blocks.is_empty());
        for (i, block) in blocks.iter_mut().enumerate() {
            let m4 = base4_m.xor3(load_m(table0[(2 * i) & 255]), load_m(table1[(2 * i) >> 8]));
            let m2_0 = base2_m.xor3(load_m(table0[(4 * i) & 255]), load_m(table1[(4 * i) >> 8]));
            let m1_0 = base1_m.xor3(load_m(table0[(8 * i) & 255]), load_m(table1[(8 * i) >> 8]));
            let m1_1 = m1_0.xor(p2);
            apply_radix8_block_256::<false>(
                block,
                m4,
                [m2_0, m2_0.xor(p2)],
                [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)],
            );
        }
    }

    fn butterfly_radix8_steps(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        steps: &[[(u128, u128); 16]; 3],
    ) {
        radix8_steps_256::<false>(values, base4, base2, base1, steps);
    }

    fn butterfly_radix8_stride8(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        steps: &[[(u128, u128); 16]; 3],
    ) {
        radix8_stride8_256::<false>(values, base4, base2, base1, steps);
    }

    fn butterfly_radix8_octants(octants: [&mut [Self]; 8], w4: u128, w2: [u128; 2], w1: [u128; 4]) {
        let p2_w = w2[0] ^ w2[1];
        let p4_w = w1[0] ^ w1[2];
        assert!(w1[1] == (w1[0] ^ p2_w) && w1[3] == (w1[1] ^ p4_w));
        let m4 = Mul128::new(w4);
        let m2_0 = Mul128::new(w2[0]);
        let m1_0 = Mul128::new(w1[0]);
        let p2 = Mul128::new(p2_w);
        let p4 = Mul128::new(p4_w);
        radix8_octants_256(octants, m4, m2_0, m1_0, p2, p4, w4 == 0 && w2[0] == 0 && w1[0] == 0);
    }

    fn butterfly_radix8_outer(values: &mut [Self], w4: u128, w2_0: u128, w2_1: u128, w1: [u128; 4]) {
        let len = values.len();
        assert!(len >= 8 && len.is_multiple_of(8));
        Self::butterfly_radix8_octants(split_octants_mut_256(values), w4, [w2_0, w2_1], w1);
        assert_eq!(values.len(), len);
    }

    fn copy_half_radix8_octants_uninit(
        novel_low: [&[Self]; 8],
        novel_high: [&[Self]; 8],
        dst: &mut [&mut [MaybeUninit<Self>]],
        w_eff: u128,
        w4: u128,
        w2: [u128; 2],
        w1: [u128; 4],
    ) {
        assert!((1..=8).contains(&dst.len()));
        let p2_w = w2[0] ^ w2[1];
        let p4_w = w1[0] ^ w1[2];
        assert!(w1[1] == (w1[0] ^ p2_w) && w1[3] == (w1[1] ^ p4_w));
        let pw_eff = Mul128::new(w_eff);
        let m4 = Mul128::new(w4);
        let m2_0 = Mul128::new(w2[0]);
        let m1_0 = Mul128::new(w1[0]);
        let p2 = Mul128::new(p2_w);
        let p4 = Mul128::new(p4_w);
        copy_radix8_octants_256(novel_low, novel_high, dst, pw_eff, m4, m2_0, m1_0, p2, p4);
    }

    fn copy_half_radix8_partial_uninit(
        novel_low: &[Self],
        novel_high: &[Self],
        dst: &mut [MaybeUninit<Self>],
        octants: usize,
        w_eff: u128,
        w4: u128,
        w2: [u128; 2],
        w1: [u128; 4],
    ) {
        let half = novel_low.len();
        assert!(half >= 8 && half.is_multiple_of(8) && (1..=8).contains(&octants));
        let e = half / 8;
        assert!(dst.len() == octants * e && novel_high.len() <= half);
        let low_oct = split_octants_ref_256(novel_low);
        let high_oct = split_high_octants_256(novel_high, e);
        let mut dst_vec: [&mut [MaybeUninit<Self>]; 8] = Default::default();
        let mut rest = dst;
        for slot in dst_vec.iter_mut().take(octants) {
            let (head, tail) = rest.split_at_mut(e);
            *slot = head;
            rest = tail;
        }
        Self::copy_half_radix8_octants_uninit(low_oct, high_oct, &mut dst_vec[..octants], w_eff, w4, w2, w1);
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
        Self::copy_half_radix8_partial_uninit(novel_low, novel_high, dst, 8, w_eff, w4, w2, w1);
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
                let (lo, hi) = (load_256(novel_low[i]), load_256(novel_high[i]));
                let a = [
                    (pw_eff.mul_xor_v(lo.0, hi.0), pw_eff.mul_xor_v(lo.1, hi.1)),
                    load_256(novel_low[q + i]),
                    load_256(novel_low[2 * q + i]),
                    load_256(novel_low[3 * q + i]),
                ];
                write_radix4_256(d0, d1, d2, d3, i, a, p);
            }
            for i in qcount..q {
                let a = [
                    load_256(novel_low[i]),
                    load_256(novel_low[q + i]),
                    load_256(novel_low[2 * q + i]),
                    load_256(novel_low[3 * q + i]),
                ];
                write_radix4_256(d0, d1, d2, d3, i, a, p);
            }
            return;
        }
        for i in 0..q {
            let load_at = |idx: usize| -> (V, V) {
                let lo = load_256(novel_low[idx]);
                if idx < qcount {
                    let hi = load_256(novel_high[idx]);
                    (pw_eff.mul_xor_v(lo.0, hi.0), pw_eff.mul_xor_v(lo.1, hi.1))
                } else {
                    lo
                }
            };
            let a = [load_at(i), load_at(q + i), load_at(2 * q + i), load_at(3 * q + i)];
            write_radix4_256(d0, d1, d2, d3, i, a, p);
        }
    }

    fn copy_half_uninit(novel_low: &[Self], novel_high: &[Self], dst: &mut [MaybeUninit<Self>], w_eff: u128) {
        let half = dst.len();
        assert!(novel_low.len() == half && novel_high.len() <= half);
        let qcount = novel_high.len();
        let p = Mul128::new(w_eff);
        let (dc, dr) = dst[..qcount].as_chunks_mut::<2>();
        let (lc, lr) = novel_low[..qcount].as_chunks::<2>();
        let (hc, hr) = novel_high.as_chunks::<2>();
        for ((d, l), h) in dc.iter_mut().zip(lc).zip(hc) {
            for k in 0..2 {
                let lo = load_256(l[k]);
                let hi = load_256(h[k]);
                d[k].write(store_256((p.mul_xor_v(lo.0, hi.0), p.mul_xor_v(lo.1, hi.1))));
            }
        }
        for ((d, l), h) in dr.iter_mut().zip(lr).zip(hr) {
            let lo = load_256(*l);
            let hi = load_256(*h);
            d.write(store_256((p.mul_xor_v(lo.0, hi.0), p.mul_xor_v(lo.1, hi.1))));
        }
        if qcount < half {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    novel_low.as_ptr().add(qcount),
                    dst.as_mut_ptr().add(qcount).cast::<Self>(),
                    half - qcount,
                );
            }
        }
        assert_eq!(dst.len(), half);
    }

    fn eval_horner_base(coefficients: &[Self], x: u128) -> Self {
        let len = coefficients.len();
        if len >= 64 {
            return eval_horner_blocked_base_256(coefficients, x);
        }
        let px = Mul128::new(x);
        let acc = coefficients.iter().rev().fold((vector::zero(), vector::zero()), |acc, c| {
            (px.mul_xor_v(vector::load(c.0), acc.0), px.mul_xor_v(vector::load(c.1), acc.1))
        });
        assert!(len < 64);
        store_256(acc)
    }

    fn transpose_butterfly_slice(low: &mut [Self], high: &mut [Self], twiddle: u128) {
        let len = low.len();
        assert_eq!(len, high.len());
        if twiddle == 0 {
            Self::xor_slice(low, high);
            return;
        }
        let p = Mul128::new(twiddle);
        let (lc, lr) = low.as_chunks_mut::<2>();
        let (hc, hr) = high.as_chunks_mut::<2>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..2 {
                let (nu, nv) = transpose_butterfly_step_256(load_256(lo[k]), load_256(hi[k]), p);
                lo[k] = store_256(nu);
                hi[k] = store_256(nv);
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let (nu, nv) = transpose_butterfly_step_256(load_256(*lo), load_256(*hi), p);
            *lo = store_256(nu);
            *hi = store_256(nv);
        }
        assert_eq!(lr.len(), hr.len());
    }

    fn transpose_from_low(low: &[Self], high: &mut [Self], twiddle: u128) {
        let len = low.len();
        assert_eq!(len, high.len());
        if twiddle == 0 {
            high.fill(Self::ZERO);
            return;
        }
        let p = Mul128::new(twiddle);
        let (lc, lr) = low.as_chunks::<2>();
        let (hc, hr) = high.as_chunks_mut::<2>();
        for (lo, hi) in lc.iter().zip(hc) {
            for k in 0..2 {
                let (v0, v1) = load_256(lo[k]);
                hi[k] = store_256((p.mul_v(v0), p.mul_v(v1)));
            }
        }
        for (lo, hi) in lr.iter().zip(&mut *hr) {
            let (v0, v1) = load_256(*lo);
            *hi = store_256((p.mul_v(v0), p.mul_v(v1)));
        }
        assert_eq!(lr.len(), hr.len());
    }

    fn transpose_from_high(low: &mut [Self], high: &mut [Self], twiddle: u128) {
        let len = low.len();
        assert_eq!(len, high.len());
        if twiddle == 0 {
            low.copy_from_slice(high);
            return;
        }
        let p = Mul128::new(twiddle ^ 1);
        let (lc, lr) = low.as_chunks_mut::<2>();
        let (hc, hr) = high.as_chunks_mut::<2>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..2 {
                let (v0, v1) = load_256(hi[k]);
                lo[k] = store_256((v0, v1));
                hi[k] = store_256((p.mul_v(v0), p.mul_v(v1)));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let (v0, v1) = load_256(*hi);
            *lo = store_256((v0, v1));
            *hi = store_256((p.mul_v(v0), p.mul_v(v1)));
        }
        assert_eq!(lr.len(), hr.len());
    }

    fn transpose_radix8_steps(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        steps: &[[(u128, u128); 16]; 3],
    ) {
        radix8_steps_256::<true>(values, base4, base2, base1, steps);
    }

    fn transpose_radix8_stride8(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        steps: &[[(u128, u128); 16]; 3],
    ) {
        radix8_stride8_256::<true>(values, base4, base2, base1, steps);
    }

    fn transpose_radix8_outer(values: &mut [Self], w4: u128, w2: [u128; 2], w1: [u128; 4]) {
        let len = values.len();
        assert!(len >= 8 && len.is_multiple_of(8));
        let p2_w = w2[0] ^ w2[1];
        let p4_w = w1[0] ^ w1[2];
        assert!(w1[1] == (w1[0] ^ p2_w) && w1[3] == (w1[1] ^ p4_w));
        let (m4, m2_0, m1_0) = (Mul128::new(w4), Mul128::new(w2[0]), Mul128::new(w1[0]));
        let (p2, p4) = (Mul128::new(p2_w), Mul128::new(p4_w));
        let octants = split_octants_mut_256(values);
        let e = len / 8;
        for i in 0..e {
            let mut v0: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].0));
            vector::transpose_radix8_8lanes_compact(&mut v0, m4, m2_0, m1_0, p2, p4);
            for k in 0..8 {
                octants[k][i].0 = vector::store(v0[k]);
            }
            let mut v1: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].1));
            vector::transpose_radix8_8lanes_compact(&mut v1, m4, m2_0, m1_0, p2, p4);
            for k in 0..8 {
                octants[k][i].1 = vector::store(v1[k]);
            }
        }
        assert_eq!(e * 8, len);
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
        use vector::{load, mul_wide, reduce_256, store, xor};
        let (a, b, c, d) = (load(self.0), load(self.1), load(rhs.0), load(rhs.1));
        let ac = mul_wide(a, c);
        let bd = mul_wide(b, d);
        let cross = mul_wide(xor(a, b), xor(c, d));
        let (lo, hi) = reduce_256(ac, bd, cross);
        Self(store(lo), store(hi))
    }
}

impl_binary_field_operators!(Gf2_256);
