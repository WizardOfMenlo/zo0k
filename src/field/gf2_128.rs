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

#[inline(always)]
fn load_tile_8(src: &[Gf2_128; 8]) -> [V; 8] {
    [
        vector::load(src[0].0),
        vector::load(src[1].0),
        vector::load(src[2].0),
        vector::load(src[3].0),
        vector::load(src[4].0),
        vector::load(src[5].0),
        vector::load(src[6].0),
        vector::load(src[7].0),
    ]
}

#[inline(always)]
fn store_tile_8(dst: &mut [Gf2_128; 8], vals: [V; 8]) {
    for t in 0..8 {
        dst[t] = Gf2_128(vector::store(vals[t]));
    }
}

fn eval_horner_8way_128(coefficients: &[Gf2_128], x: u128) -> Gf2_128 {
    let len = coefficients.len();
    assert!(len >= 8);
    let px = Mul128::new(x);
    let x2 = vector::square(x);
    let x4 = vector::square(x2);
    let px8 = Mul128::new(vector::square(x4));
    let rem = len & 7;
    let full = len - rem;
    let mut a = [vector::zero(); 8];
    for k in 0..rem {
        a[k] = vector::load(coefficients[full + k].0);
    }
    let (chunks, _) = coefficients[..full].as_chunks::<8>();
    assert_eq!(chunks.len() * 8, full);
    for chunk in chunks.iter().rev() {
        for k in 0..8 {
            a[k] = px8.mul_xor_v(vector::load(chunk[k].0), a[k]);
        }
    }
    let mut acc = vector::zero();
    for k in (0..8).rev() {
        acc = px.mul_xor_v(a[k], acc);
    }
    Gf2_128(vector::store(acc))
}

#[inline(never)]
fn eval_horner_blocked_128(coefficients: &[Gf2_128], x: u128) -> Gf2_128 {
    const BLOCK: usize = 32;
    let len = coefficients.len();
    assert!(len >= 2 * BLOCK);
    let px = Mul128::new(x);
    let mut p = [vector::zero(); BLOCK];
    let mut ps = [vector::zero(); BLOCK];
    let mut cur = vector::load(1);
    for k in 0..BLOCK {
        p[k] = vector::pack_v(cur);
        ps[k] = vector::pack_v(Acc128::precompute_aux(cur));
        cur = px.mul_v(cur);
    }
    let px32 = Mul128::new(vector::store(cur));
    let rem = len % BLOCK;
    let full = len - rem;
    let mut tail_acc = Acc128::zero();
    for k in 0..rem {
        tail_acc.add_product_pre(vector::load(coefficients[full + k].0), p[k], ps[k]);
    }
    let mut acc = tail_acc.reduce();
    let (blocks, _) = coefficients[..full].as_chunks::<BLOCK>();
    assert_eq!(blocks.len() * BLOCK, full);
    for chunk in blocks.iter().rev() {
        let mut a0 = Acc128::zero();
        let mut a1 = Acc128::zero();
        let mut a2 = Acc128::zero();
        let mut a3 = Acc128::zero();
        for j in 0..(BLOCK / 4) {
            let k = 4 * j;
            a0.add_product_pre(vector::load(chunk[k].0), p[k], ps[k]);
            a1.add_product_pre(vector::load(chunk[k + 1].0), p[k + 1], ps[k + 1]);
            a2.add_product_pre(vector::load(chunk[k + 2].0), p[k + 2], ps[k + 2]);
            a3.add_product_pre(vector::load(chunk[k + 3].0), p[k + 3], ps[k + 3]);
        }
        let block_sum = a0.merge(a1).merge(a2.merge(a3)).reduce();
        acc = px32.mul_xor_v(block_sum, acc);
    }
    Gf2_128(vector::store(acc))
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

    #[inline(always)]
    fn write_bytes(self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.0.to_le_bytes());
    }

    #[inline(always)]
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
        let (ac, ar) = a.as_chunks::<4>();
        let (bc, br) = b.as_chunks::<4>();
        for (x, y) in ac.iter().zip(bc) {
            acc0.add_product(vector::load(x[0].0), vector::load(y[0].0));
            acc1.add_product(vector::load(x[1].0), vector::load(y[1].0));
            acc2.add_product(vector::load(x[2].0), vector::load(y[2].0));
            acc3.add_product(vector::load(x[3].0), vector::load(y[3].0));
        }
        let mut acc = acc0.merge(acc1).merge(acc2.merge(acc3));
        for (x, y) in ar.iter().zip(br) {
            acc.add_product(vector::load(x.0), vector::load(y.0));
        }
        assert_eq!(ar.len(), br.len());
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
        let (al_c, al_r) = a_low.as_chunks::<4>();
        let (ah_c, ah_r) = a_high.as_chunks::<4>();
        let (bl_c, bl_r) = b_low.as_chunks::<4>();
        let (bh_c, bh_r) = b_high.as_chunks::<4>();
        for (((al, ah), bl), bh) in al_c.iter().zip(ah_c).zip(bl_c).zip(bh_c) {
            for k in 0..2 {
                let (x0, y0) = (vector::load(al[2 * k].0), vector::load(bl[2 * k].0));
                let (x1, y1) = (vector::load(al[2 * k + 1].0), vector::load(bl[2 * k + 1].0));
                c0.add_product(x0, y0);
                q0.add_product(
                    vector::xor(vector::load(ah[2 * k].0), x0),
                    vector::xor(vector::load(bh[2 * k].0), y0),
                );
                c1.add_product(x1, y1);
                q1.add_product(
                    vector::xor(vector::load(ah[2 * k + 1].0), x1),
                    vector::xor(vector::load(bh[2 * k + 1].0), y1),
                );
            }
        }
        let mut c_acc = c0.merge(c1);
        let mut q_acc = q0.merge(q1);
        for i in 0..al_r.len() {
            let (al, ah) = (vector::load(al_r[i].0), vector::load(ah_r[i].0));
            let (bl, bh) = (vector::load(bl_r[i].0), vector::load(bh_r[i].0));
            c_acc.add_product(al, bl);
            q_acc.add_product(vector::xor(ah, al), vector::xor(bh, bl));
        }
        assert_eq!(al_r.len(), bh_r.len());
        (Self(vector::store(c_acc.reduce())), Self(vector::store(q_acc.reduce())))
    }

    fn fold_slice(low: &mut [Self], high: &[Self], challenge: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        if challenge.0 == 0 {
            return;
        }
        let p = Mul128::new(challenge.0);
        let (lc, lr) = low.as_chunks_mut::<4>();
        let (hc, hr) = high.as_chunks::<4>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..4 {
                let l = vector::load(lo[k].0);
                let h = vector::load(hi[k].0);
                lo[k] = Self(vector::store(p.mul_xor_v(l, vector::xor(h, l))));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(hr) {
            let l = vector::load(lo.0);
            let h = vector::load(hi.0);
            *lo = Self(vector::store(p.mul_xor_v(l, vector::xor(h, l))));
        }
        assert_eq!(lr.len(), hr.len());
    }

    fn eq_extend_uninit(low: &mut [Self], high: &mut [MaybeUninit<Self>], coordinate: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        let p = Mul128::new(coordinate.0);
        let (lc, lr) = low.as_chunks_mut::<4>();
        let (hc, hr) = high.as_chunks_mut::<4>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..4 {
                let l = vector::load(lo[k].0);
                let h = p.mul_v(l);
                hi[k].write(Self(vector::store(h)));
                lo[k] = Self(vector::store(vector::xor(l, h)));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let l = vector::load(lo.0);
            let h = p.mul_v(l);
            hi.write(Self(vector::store(h)));
            *lo = Self(vector::store(vector::xor(l, h)));
        }
        assert_eq!(lr.len(), hr.len());
    }

    fn add_scaled_slice(dst: &mut [Self], src: &[Self], scale: Self) {
        let len = dst.len();
        assert_eq!(len, src.len());
        if scale.0 == 0 {
            return;
        }
        let p = Mul128::new(scale.0);
        let (dc, dr) = dst.as_chunks_mut::<4>();
        let (sc, sr) = src.as_chunks::<4>();
        for (d, s) in dc.iter_mut().zip(sc) {
            for k in 0..4 {
                d[k] = Self(vector::store(p.mul_xor_v(vector::load(d[k].0), vector::load(s[k].0))));
            }
        }
        for (d, s) in dr.iter_mut().zip(sr) {
            *d = Self(vector::store(p.mul_xor_v(vector::load(d.0), vector::load(s.0))));
        }
        assert_eq!(dr.len(), sr.len());
    }

    fn scale_slice(dst: &mut [Self], scale: Self) {
        let len = dst.len();
        if scale.0 == 1 {
            return;
        }
        if scale.0 == 0 {
            dst.fill(Self::ZERO);
            return;
        }
        let p = Mul128::new(scale.0);
        let (dc, dr) = dst.as_chunks_mut::<4>();
        for d in dc {
            for entry in d.iter_mut() {
                *entry = Self(vector::store(p.mul_v(vector::load(entry.0))));
            }
        }
        for d in &mut *dr {
            *d = Self(vector::store(p.mul_v(vector::load(d.0))));
        }
        assert!(len >= dr.len());
        assert!(dr.len() < 4);
    }

    fn scale_to_uninit(dst: &mut [MaybeUninit<Self>], src: &[Self], scale: Self) {
        let len = dst.len();
        assert_eq!(len, src.len());
        if scale.0 == 0 {
            for d in dst {
                d.write(Self::ZERO);
            }
            return;
        }
        let p = Mul128::new(scale.0);
        let (dc, dr) = dst.as_chunks_mut::<4>();
        let (sc, sr) = src.as_chunks::<4>();
        for (d, s) in dc.iter_mut().zip(sc) {
            for k in 0..4 {
                d[k].write(Self(vector::store(p.mul_v(vector::load(s[k].0)))));
            }
        }
        for (d, s) in dr.iter_mut().zip(sr) {
            d.write(Self(vector::store(p.mul_v(vector::load(s.0)))));
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
        if challenge.0 == 0 {
            return Self::round_coefficients_slice(a0, a1, b0, b1);
        }
        let p = Mul128::new(challenge.0);
        let mut c0 = Acc128::zero();
        let mut c1 = Acc128::zero();
        let mut q0 = Acc128::zero();
        let mut q1 = Acc128::zero();
        let (a0c, a0r) = a0.as_chunks_mut::<2>();
        let (a1c, a1r) = a1.as_chunks_mut::<2>();
        let (a2c, a2r) = a2.as_chunks::<2>();
        let (a3c, a3r) = a3.as_chunks::<2>();
        let (b0c, b0r) = b0.as_chunks_mut::<2>();
        let (b1c, b1r) = b1.as_chunks_mut::<2>();
        let (b2c, b2r) = b2.as_chunks::<2>();
        let (b3c, b3r) = b3.as_chunks::<2>();
        for i in 0..a0c.len() {
            for k in 0..2 {
                let fa0 = fold_one_128(a0c[i][k], a2c[i][k], p);
                let fa1 = fold_one_128(a1c[i][k], a3c[i][k], p);
                let fb0 = fold_one_128(b0c[i][k], b2c[i][k], p);
                let fb1 = fold_one_128(b1c[i][k], b3c[i][k], p);
                a0c[i][k] = Self(vector::store(fa0));
                a1c[i][k] = Self(vector::store(fa1));
                b0c[i][k] = Self(vector::store(fb0));
                b1c[i][k] = Self(vector::store(fb1));
                if k == 0 {
                    c0.add_product(fa0, fb0);
                    q0.add_product(vector::xor(fa1, fa0), vector::xor(fb1, fb0));
                } else {
                    c1.add_product(fa0, fb0);
                    q1.add_product(vector::xor(fa1, fa0), vector::xor(fb1, fb0));
                }
            }
        }
        let mut c_acc = c0.merge(c1);
        let mut q_acc = q0.merge(q1);
        for i in 0..a0r.len() {
            let fa0 = fold_one_128(a0r[i], a2r[i], p);
            let fa1 = fold_one_128(a1r[i], a3r[i], p);
            let fb0 = fold_one_128(b0r[i], b2r[i], p);
            let fb1 = fold_one_128(b1r[i], b3r[i], p);
            a0r[i] = Self(vector::store(fa0));
            a1r[i] = Self(vector::store(fa1));
            b0r[i] = Self(vector::store(fb0));
            b1r[i] = Self(vector::store(fb1));
            c_acc.add_product(fa0, fb0);
            q_acc.add_product(vector::xor(fa1, fa0), vector::xor(fb1, fb0));
        }
        (Self(vector::store(c_acc.reduce())), Self(vector::store(q_acc.reduce())))
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
        let (p0, p1, p2, p3) = (Mul128::new(w0.0), Mul128::new(w1.0), Mul128::new(w2.0), Mul128::new(w3.0));
        let (dc, dr) = dst.as_chunks_mut::<2>();
        let (c0, r0) = s0.as_chunks::<2>();
        let (c1, r1) = s1.as_chunks::<2>();
        let (c2, r2) = s2.as_chunks::<2>();
        let (c3, r3) = s3.as_chunks::<2>();
        for i in 0..dc.len() {
            for k in 0..2 {
                dc[i][k] =
                    add_scaled_4_one_128(dc[i][k], c0[i][k], c1[i][k], c2[i][k], c3[i][k], p0, p1, p2, p3);
            }
        }
        for i in 0..dr.len() {
            dr[i] = add_scaled_4_one_128(dr[i], r0[i], r1[i], r2[i], r3[i], p0, p1, p2, p3);
        }
        assert_eq!(dr.len(), r3.len());
    }

    fn eval_horner_slice(coefficients: &[Self], x: Self) -> Self {
        Self::eval_horner_base(coefficients, x.0)
    }

    fn add_powers_slice(target: &mut [Self], x: Self, scale: Self) {
        let len = target.len();
        if len == 0 {
            return;
        }
        if len == 1 {
            target[0] += scale;
            return;
        }
        let px = Mul128::new(x.0);
        if len < 16 {
            let mut cur = vector::load(scale.0);
            for entry in target {
                *entry = Self(vector::store(vector::xor(vector::load(entry.0), cur)));
                cur = px.mul_v(cur);
            }
            assert!(len < 16);
            return;
        }
        let mut s = [vector::load(scale.0); 8];
        for k in 1..8 {
            s[k] = px.mul_v(s[k - 1]);
        }
        let x2 = vector::square(x.0);
        let x4 = vector::square(x2);
        let px8 = Mul128::new(vector::square(x4));
        let (chunks, rem) = target.as_chunks_mut::<8>();
        for chunk in chunks {
            for k in 0..8 {
                chunk[k] = Self(vector::store(vector::xor(vector::load(chunk[k].0), s[k])));
                s[k] = px8.mul_v(s[k]);
            }
        }
        let mut cur = s[0];
        for entry in &mut *rem {
            *entry = Self(vector::store(vector::xor(vector::load(entry.0), cur)));
            cur = px.mul_v(cur);
        }
        assert!(rem.len() < 8);
    }

    fn xor_slice(dst: &mut [Self], src: &[Self]) {
        let len = dst.len();
        assert_eq!(len, src.len());
        let (dc, dr) = dst.as_chunks_mut::<8>();
        let (sc, sr) = src.as_chunks::<8>();
        for (d, s) in dc.iter_mut().zip(sc) {
            for k in 0..8 {
                let v = vector::xor(vector::load(d[k].0), vector::load(s[k].0));
                d[k] = Self(vector::store(v));
            }
        }
        for (d, s) in dr.iter_mut().zip(sr) {
            let v = vector::xor(vector::load(d.0), vector::load(s.0));
            *d = Self(vector::store(v));
        }
        assert_eq!(dr.len(), sr.len());
    }
}

#[inline(always)]
fn fold_one_128(lo: Gf2_128, hi: Gf2_128, p: Mul128) -> V {
    let l = vector::load(lo.0);
    let h = vector::load(hi.0);
    p.mul_xor_v(l, vector::xor(h, l))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn add_scaled_4_one_128(
    d: Gf2_128,
    s0: Gf2_128,
    s1: Gf2_128,
    s2: Gf2_128,
    s3: Gf2_128,
    p0: Mul128,
    p1: Mul128,
    p2: Mul128,
    p3: Mul128,
) -> Gf2_128 {
    let (l0, m0) = p0.mul_lm(vector::load(s0.0));
    let (l1, m1) = p1.mul_lm(vector::load(s1.0));
    let (l2, m2) = p2.mul_lm(vector::load(s2.0));
    let (l3, m3) = p3.mul_lm(vector::load(s3.0));
    let l = vector::xor3(vector::load(d.0), vector::xor(l0, l1), vector::xor(l2, l3));
    let m = vector::xor(vector::xor(m0, m1), vector::xor(m2, m3));
    Gf2_128(vector::store(vector::reduce_lm(l, m)))
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
    let (c0, r0) = v0.as_chunks_mut::<2>();
    let (c1, r1) = v1.as_chunks_mut::<2>();
    let (c2, r2) = v2.as_chunks_mut::<2>();
    let (c3, r3) = v3.as_chunks_mut::<2>();
    for i in 0..c0.len() {
        for k in 0..2 {
            let (a0, a2) = vector::butterfly_step_v(vector::load(c0[i][k].0), vector::load(c2[i][k].0), pw);
            let (a1, a3) = vector::butterfly_step_v(vector::load(c1[i][k].0), vector::load(c3[i][k].0), pw);
            let (b0, b1) = vector::butterfly_step_v(a0, a1, pw0);
            let (b2, b3) = vector::butterfly_step_v(a2, a3, pw1);
            c0[i][k] = Gf2_128(vector::store(b0));
            c1[i][k] = Gf2_128(vector::store(b1));
            c2[i][k] = Gf2_128(vector::store(b2));
            c3[i][k] = Gf2_128(vector::store(b3));
        }
    }
    for i in 0..r0.len() {
        let (a0, a2) = vector::butterfly_step_v(vector::load(r0[i].0), vector::load(r2[i].0), pw);
        let (a1, a3) = vector::butterfly_step_v(vector::load(r1[i].0), vector::load(r3[i].0), pw);
        let (b0, b1) = vector::butterfly_step_v(a0, a1, pw0);
        let (b2, b3) = vector::butterfly_step_v(a2, a3, pw1);
        r0[i] = Gf2_128(vector::store(b0));
        r1[i] = Gf2_128(vector::store(b1));
        r2[i] = Gf2_128(vector::store(b2));
        r3[i] = Gf2_128(vector::store(b3));
    }
    assert_eq!(r0.len(), r3.len());
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
    let (c0, r0) = v0.as_chunks_mut::<4>();
    let (c1, r1) = v1.as_chunks_mut::<4>();
    let (c2, r2) = v2.as_chunks_mut::<4>();
    let (c3, r3) = v3.as_chunks_mut::<4>();
    for i in 0..c0.len() {
        for k in 0..4 {
            let a0 = vector::load(c0[i][k].0);
            let a1 = vector::load(c1[i][k].0);
            let a2 = vector::xor(vector::load(c2[i][k].0), a0);
            let a3 = vector::xor(vector::load(c3[i][k].0), a1);
            let (b2, b3) = vector::butterfly_step_v(a2, a3, pw1);
            c1[i][k] = Gf2_128(vector::store(vector::xor(a1, a0)));
            c2[i][k] = Gf2_128(vector::store(b2));
            c3[i][k] = Gf2_128(vector::store(b3));
        }
    }
    for i in 0..r0.len() {
        let a0 = vector::load(r0[i].0);
        let a1 = vector::load(r1[i].0);
        let a2 = vector::xor(vector::load(r2[i].0), a0);
        let a3 = vector::xor(vector::load(r3[i].0), a1);
        let (b2, b3) = vector::butterfly_step_v(a2, a3, pw1);
        r1[i] = Gf2_128(vector::store(vector::xor(a1, a0)));
        r2[i] = Gf2_128(vector::store(b2));
        r3[i] = Gf2_128(vector::store(b3));
    }
    assert_eq!(r0.len(), r3.len());
}

fn radix8_octants_128(
    octants: [&mut [Gf2_128]; 8],
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
            let mut v: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].0));
            vector::radix8_8lanes_zero_prefix(&mut v, m2_1, m1_1, m1_2, m1_3);
            for k in 1..8 {
                octants[k][i] = Gf2_128(vector::store(v[k]));
            }
        }
    } else {
        for i in 0..e {
            let mut v: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].0));
            vector::radix8_8lanes_compact(&mut v, m4, m2_0, m1_0, p2, p4);
            for k in 0..8 {
                octants[k][i] = Gf2_128(vector::store(v[k]));
            }
        }
    }
    assert_eq!(octants[7].len(), e);
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn copy_radix8_full_128(
    novel_low: [&[Gf2_128]; 8],
    novel_high: [&[Gf2_128]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_128>]],
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
    let mut tile_buf = [[vector::zero(); 8]; 8];
    for tile in 0..tiles {
        let base = tile * 8;
        for k in 0..8 {
            let lo = &novel_low[k][base..base + 8];
            let hi = &novel_high[k][base..base + 8];
            for t in 0..8 {
                tile_buf[t][k] = pw_eff.mul_xor_v(vector::load(lo[t].0), vector::load(hi[t].0));
            }
        }
        if octants == 8 {
            vector::radix8_rows_compact(&mut tile_buf, m4, m2_0, m1_0, p2, p4);
        } else {
            vector::radix8_rows_partial(&mut tile_buf, m4, m2_0, m1_0, p2, p4, octants);
        }
        for (k, out_oct) in dst.iter_mut().enumerate() {
            let out = &mut out_oct[base..base + 8];
            for t in 0..8 {
                out[t].write(Gf2_128(vector::store(tile_buf[t][k])));
            }
        }
    }
    let rem_start = tiles * 8;
    for i in rem_start..len {
        let mut v: [V; 8] = std::array::from_fn(|k| {
            pw_eff.mul_xor_v(vector::load(novel_low[k][i].0), vector::load(novel_high[k][i].0))
        });
        if octants == 8 {
            vector::radix8_8lanes_compact(&mut v, m4, m2_0, m1_0, p2, p4);
        } else {
            vector::radix8_8lanes_partial(&mut v, m4, m2_0, m1_0, p2, p4, octants);
        }
        for (k, out_oct) in dst.iter_mut().enumerate() {
            out_oct[i].write(Gf2_128(vector::store(v[k])));
        }
    }
    assert!(rem_start <= len);
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn copy_radix8_ragged_128(
    novel_low: [&[Gf2_128]; 8],
    novel_high: [&[Gf2_128]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_128>]],
    start: usize,
    end: usize,
    pw_eff: Mul128,
    tw: &[Mul128; 5],
) {
    let octants = dst.len();
    assert!((1..=8).contains(&octants) && start <= end);
    assert!(novel_low.iter().all(|s| s.len() >= end) && dst.iter().all(|s| s.len() >= end));
    let [m4, m2_0, m1_0, p2, p4] = *tw;
    for i in start..end {
        let mut v = [vector::zero(); 8];
        for k in 0..8 {
            let lo = vector::load(novel_low[k][i].0);
            v[k] = if i < novel_high[k].len() {
                pw_eff.mul_xor_v(lo, vector::load(novel_high[k][i].0))
            } else {
                lo
            };
        }
        vector::radix8_8lanes_partial(&mut v, m4, m2_0, m1_0, p2, p4, octants);
        for (k, out_oct) in dst.iter_mut().enumerate() {
            out_oct[i].write(Gf2_128(vector::store(v[k])));
        }
    }
}

#[inline(always)]
fn load_tw5(tw: &[(u128, u128); 5]) -> [Mul128; 5] {
    [
        Mul128::from_pair(tw[0].0, tw[0].1),
        Mul128::from_pair(tw[1].0, tw[1].1),
        Mul128::from_pair(tw[2].0, tw[2].1),
        Mul128::from_pair(tw[3].0, tw[3].1),
        Mul128::from_pair(tw[4].0, tw[4].1),
    ]
}

#[inline(never)]
fn copy_radix8_strided_8_zero_128(
    src: &[Gf2_128],
    src_stride: usize,
    dst: &mut [MaybeUninit<Gf2_128>],
    dst_stride: usize,
    len: usize,
    tw: &[(u128, u128); 5],
) {
    assert!(len == 0 || 7 * src_stride + len <= src.len());
    assert!(len == 0 || 7 * dst_stride + len <= dst.len());
    let [_, m2_0, m1_0, p2, p4] = load_tw5(tw);
    let m2_1 = m2_0.xor(p2);
    let m1_1 = m1_0.xor(p2);
    let m1_2 = m1_0.xor(p4);
    let m1_3 = m1_1.xor(p4);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr().cast::<Gf2_128>();
    for i in 0..len {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(unsafe { (*sp.add(k * src_stride + i)).0 }));
        vector::radix8_8lanes_zero_prefix(&mut v, m2_1, m1_1, m1_2, m1_3);
        for (k, &val) in v.iter().enumerate() {
            unsafe { std::ptr::write(dp.add(k * dst_stride + i), Gf2_128(vector::store(val))) };
        }
    }
}

#[inline(never)]
fn copy_radix8_strided_8_nonzero_128(
    src: &[Gf2_128],
    src_stride: usize,
    dst: &mut [MaybeUninit<Gf2_128>],
    dst_stride: usize,
    len: usize,
    tw: &[(u128, u128); 5],
) {
    assert!(len == 0 || 7 * src_stride + len <= src.len());
    assert!(len == 0 || 7 * dst_stride + len <= dst.len());
    let [m4, m2_0, m1_0, p2, p4] = load_tw5(tw);
    let m2 = [m2_0, m2_0.xor(p2)];
    let m1_1 = m1_0.xor(p2);
    let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr().cast::<Gf2_128>();
    for i in 0..len {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(unsafe { (*sp.add(k * src_stride + i)).0 }));
        vector::radix8_8lanes_twiddles(&mut v, m4, m2, m1);
        for (k, &val) in v.iter().enumerate() {
            unsafe { std::ptr::write(dp.add(k * dst_stride + i), Gf2_128(vector::store(val))) };
        }
    }
}

#[inline(always)]
fn copy_radix8_strided_8_128(
    src: &[Gf2_128],
    src_stride: usize,
    dst: &mut [MaybeUninit<Gf2_128>],
    dst_stride: usize,
    len: usize,
    tw: &[(u128, u128); 5],
    zero_prefix: bool,
) {
    if zero_prefix {
        copy_radix8_strided_8_zero_128(src, src_stride, dst, dst_stride, len, tw);
    } else {
        copy_radix8_strided_8_nonzero_128(src, src_stride, dst, dst_stride, len, tw);
    }
}

#[inline(never)]
fn copy_radix8_from_strided_zero_128(
    src: &[Gf2_128],
    src_stride: usize,
    dst: [&mut [MaybeUninit<Gf2_128>]; 8],
    tw: &[(u128, u128); 5],
) {
    let len = dst[0].len();
    assert!(dst.iter().all(|s| s.len() == len));
    assert!(len == 0 || 7 * src_stride + len <= src.len());
    let [_, m2_0, m1_0, p2, p4] = load_tw5(tw);
    let m2_1 = m2_0.xor(p2);
    let m1_1 = m1_0.xor(p2);
    let m1_2 = m1_0.xor(p4);
    let m1_3 = m1_1.xor(p4);
    let sp = src.as_ptr();
    let dp: [*mut Gf2_128; 8] = std::array::from_fn(|k| dst[k].as_mut_ptr().cast());
    for i in 0..len {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(unsafe { (*sp.add(k * src_stride + i)).0 }));
        vector::radix8_8lanes_zero_prefix(&mut v, m2_1, m1_1, m1_2, m1_3);
        for (k, &val) in v.iter().enumerate() {
            unsafe { std::ptr::write(dp[k].add(i), Gf2_128(vector::store(val))) };
        }
    }
}

#[inline(never)]
fn copy_radix8_from_strided_nonzero_128(
    src: &[Gf2_128],
    src_stride: usize,
    dst: [&mut [MaybeUninit<Gf2_128>]; 8],
    tw: &[(u128, u128); 5],
) {
    let len = dst[0].len();
    assert!(dst.iter().all(|s| s.len() == len));
    assert!(len == 0 || 7 * src_stride + len <= src.len());
    let [m4, m2_0, m1_0, p2, p4] = load_tw5(tw);
    let m2 = [m2_0, m2_0.xor(p2)];
    let m1_1 = m1_0.xor(p2);
    let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
    let sp = src.as_ptr();
    let dp: [*mut Gf2_128; 8] = std::array::from_fn(|k| dst[k].as_mut_ptr().cast());
    for i in 0..len {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(unsafe { (*sp.add(k * src_stride + i)).0 }));
        vector::radix8_8lanes_twiddles(&mut v, m4, m2, m1);
        for (k, &val) in v.iter().enumerate() {
            unsafe { std::ptr::write(dp[k].add(i), Gf2_128(vector::store(val))) };
        }
    }
}

#[inline(never)]
fn copy_radix8_to_strided_zero_128(
    src: [&[Gf2_128]; 8],
    dst: &mut [MaybeUninit<Gf2_128>],
    dst_stride: usize,
    tw: &[(u128, u128); 5],
) {
    let len = src[0].len();
    assert!(src.iter().all(|s| s.len() == len));
    assert!(len == 0 || 7 * dst_stride + len <= dst.len());
    let [_, m2_0, m1_0, p2, p4] = load_tw5(tw);
    let m2_1 = m2_0.xor(p2);
    let m1_1 = m1_0.xor(p2);
    let m1_2 = m1_0.xor(p4);
    let m1_3 = m1_1.xor(p4);
    let sp: [*const Gf2_128; 8] = std::array::from_fn(|k| src[k].as_ptr());
    let dp = dst.as_mut_ptr().cast::<Gf2_128>();
    for i in 0..len {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(unsafe { (*sp[k].add(i)).0 }));
        vector::radix8_8lanes_zero_prefix(&mut v, m2_1, m1_1, m1_2, m1_3);
        for (k, &val) in v.iter().enumerate() {
            unsafe { std::ptr::write(dp.add(k * dst_stride + i), Gf2_128(vector::store(val))) };
        }
    }
}

#[inline(never)]
fn copy_radix8_to_strided_nonzero_128(
    src: [&[Gf2_128]; 8],
    dst: &mut [MaybeUninit<Gf2_128>],
    dst_stride: usize,
    tw: &[(u128, u128); 5],
) {
    let len = src[0].len();
    assert!(src.iter().all(|s| s.len() == len));
    assert!(len == 0 || 7 * dst_stride + len <= dst.len());
    let [m4, m2_0, m1_0, p2, p4] = load_tw5(tw);
    let m2 = [m2_0, m2_0.xor(p2)];
    let m1_1 = m1_0.xor(p2);
    let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
    let sp: [*const Gf2_128; 8] = std::array::from_fn(|k| src[k].as_ptr());
    let dp = dst.as_mut_ptr().cast::<Gf2_128>();
    for i in 0..len {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(unsafe { (*sp[k].add(i)).0 }));
        vector::radix8_8lanes_twiddles(&mut v, m4, m2, m1);
        for (k, &val) in v.iter().enumerate() {
            unsafe { std::ptr::write(dp.add(k * dst_stride + i), Gf2_128(vector::store(val))) };
        }
    }
}

#[inline(never)]
fn copy_radix8_pure_8_128(
    novel_low: [&[Gf2_128]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_128>]],
    start: usize,
    end: usize,
    tw: &[Mul128; 5],
) {
    assert!(dst.len() == 8 && start <= end);
    assert!(novel_low.iter().all(|s| s.len() >= end) && dst.iter().all(|s| s.len() >= end));
    let [m4, m2_0, m1_0, p2, p4] = *tw;
    let m2 = [m2_0, m2_0.xor(p2)];
    let m1_1 = m1_0.xor(p2);
    let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
    let len = end - start;
    let sp: [*const Gf2_128; 8] = std::array::from_fn(|k| unsafe { novel_low[k].as_ptr().add(start) });
    let dp: [*mut Gf2_128; 8] = std::array::from_fn(|k| unsafe { dst[k].as_mut_ptr().add(start).cast() });
    for i in 0..len {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(unsafe { (*sp[k].add(i)).0 }));
        vector::radix8_8lanes_twiddles(&mut v, m4, m2, m1);
        for (k, &val) in v.iter().enumerate() {
            unsafe { std::ptr::write(dp[k].add(i), Gf2_128(vector::store(val))) };
        }
    }
}

#[inline(never)]
fn copy_radix8_pure_partial_128(
    novel_low: [&[Gf2_128]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_128>]],
    start: usize,
    end: usize,
    tw: &[Mul128; 5],
) {
    let octants = dst.len();
    assert!((1..8).contains(&octants) && start <= end);
    assert!(novel_low.iter().all(|s| s.len() >= end) && dst.iter().all(|s| s.len() >= end));
    let [m4, m2_0, m1_0, p2, p4] = *tw;
    for i in start..end {
        let mut v: [V; 8] = std::array::from_fn(|k| vector::load(novel_low[k][i].0));
        vector::radix8_8lanes_partial(&mut v, m4, m2_0, m1_0, p2, p4, octants);
        for (k, out_oct) in dst.iter_mut().enumerate() {
            out_oct[i].write(Gf2_128(vector::store(v[k])));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_radix8_octants_128(
    novel_low: [&[Gf2_128]; 8],
    novel_high: [&[Gf2_128]; 8],
    dst: &mut [&mut [MaybeUninit<Gf2_128>]],
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
    let tw = [m4, m2_0, m1_0, p2, p4];
    let min_hi = novel_high.iter().map(|s| s.len()).min().unwrap_or(0);
    let max_hi = novel_high.iter().map(|s| s.len()).max().unwrap_or(0);
    if min_hi > 0 {
        copy_radix8_full_128(novel_low, novel_high, dst, min_hi, pw_eff, m4, m2_0, m1_0, p2, p4);
    }
    if min_hi < max_hi {
        copy_radix8_ragged_128(novel_low, novel_high, dst, min_hi, max_hi, pw_eff, &tw);
    }
    if max_hi < e {
        if octants == 8 {
            copy_radix8_pure_8_128(novel_low, dst, max_hi, e, &tw);
        } else {
            copy_radix8_pure_partial_128(novel_low, dst, max_hi, e, &tw);
        }
    }
}

#[inline(always)]
fn split_octants_mut_128(values: &mut [Gf2_128]) -> [&mut [Gf2_128]; 8] {
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
fn split_octants_ref_128(values: &[Gf2_128]) -> [&[Gf2_128]; 8] {
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
fn split_high_octants_128(novel_high: &[Gf2_128], e: usize) -> [&[Gf2_128]; 8] {
    std::array::from_fn(|k| {
        let start = k * e;
        if start < novel_high.len() { &novel_high[start..novel_high.len().min(start + e)] } else { &[] }
    })
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
    let (t0, t2) = vector::butterfly_step_v(a[0], a[2], p[0]);
    let (t1, t3) = vector::butterfly_step_v(a[1], a[3], p[0]);
    let (b0, b1) = vector::butterfly_step_v(t0, t1, p[1]);
    let (b2, b3) = vector::butterfly_step_v(t2, t3, p[2]);
    d0[i].write(Gf2_128(vector::store(b0)));
    d1[i].write(Gf2_128(vector::store(b1)));
    d2[i].write(Gf2_128(vector::store(b2)));
    d3[i].write(Gf2_128(vector::store(b3)));
}

#[inline(always)]
fn apply_radix8_block_128<const TRANSPOSE: bool>(
    block: &mut [Gf2_128; 8],
    m4: Mul128,
    m2: [Mul128; 2],
    m1: [Mul128; 4],
) {
    let mut v = load_tile_8(block);
    if TRANSPOSE {
        vector::transpose_radix8_8lanes_twiddles(&mut v, m4, m2, m1);
    } else {
        vector::radix8_8lanes_twiddles(&mut v, m4, m2, m1);
    }
    store_tile_8(block, v);
}

fn radix8_steps_128<const TRANSPOSE: bool>(
    values: &mut [Gf2_128],
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
        apply_radix8_block_128::<TRANSPOSE>(block, m4, m2, m1);
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

fn radix8_stride8_128<const TRANSPOSE: bool>(
    values: &mut [Gf2_128],
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
    let (chunks, remainder) = values.as_chunks_mut::<64>();
    let n = chunks.len();
    assert!(remainder.is_empty());
    assert!((1..=65536).contains(&n));
    for (i, chunk) in chunks.iter_mut().enumerate() {
        for off in 0..8 {
            let mut v: [V; 8] = std::array::from_fn(|k| vector::load(chunk[k * 8 + off].0));
            if TRANSPOSE {
                vector::transpose_radix8_8lanes_twiddles(&mut v, m4, m2, m1);
            } else {
                vector::radix8_8lanes_twiddles(&mut v, m4, m2, m1);
            }
            for k in 0..8 {
                chunk[k * 8 + off] = Gf2_128(vector::store(v[k]));
            }
        }
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
            Self::xor_slice(high, low);
            return;
        }
        let p = Mul128::new(twiddle);
        let (lc, lr) = low.as_chunks_mut::<4>();
        let (hc, hr) = high.as_chunks_mut::<4>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..4 {
                let (nr, nq) = vector::butterfly_step_v(vector::load(lo[k].0), vector::load(hi[k].0), p);
                lo[k] = Self(vector::store(nr));
                hi[k] = Self(vector::store(nq));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let (nr, nq) = vector::butterfly_step_v(vector::load(lo.0), vector::load(hi.0), p);
            *lo = Self(vector::store(nr));
            *hi = Self(vector::store(nq));
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
            apply_radix8_block_128::<false>(
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
        radix8_steps_128::<false>(values, base4, base2, base1, steps);
    }

    fn butterfly_radix8_stride8(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        steps: &[[(u128, u128); 16]; 3],
    ) {
        radix8_stride8_128::<false>(values, base4, base2, base1, steps);
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
        radix8_octants_128(octants, m4, m2_0, m1_0, p2, p4, w4 == 0 && w2[0] == 0 && w1[0] == 0);
    }

    fn butterfly_radix8_outer(values: &mut [Self], w4: u128, w2_0: u128, w2_1: u128, w1: [u128; 4]) {
        let len = values.len();
        assert!(len >= 8 && len.is_multiple_of(8));
        Self::butterfly_radix8_octants(split_octants_mut_128(values), w4, [w2_0, w2_1], w1);
        assert_eq!(values.len(), len);
    }

    #[inline(always)]
    fn copy_radix8_strided_uninit(
        src: &[Self],
        src_stride: usize,
        dst: &mut [MaybeUninit<Self>],
        dst_stride: usize,
        len: usize,
        tw: &[(u128, u128); 5],
        zero_prefix: bool,
    ) {
        copy_radix8_strided_8_128(src, src_stride, dst, dst_stride, len, tw, zero_prefix);
    }

    #[inline(always)]
    fn copy_radix8_from_strided_uninit(
        src: &[Self],
        src_stride: usize,
        dst: [&mut [MaybeUninit<Self>]; 8],
        tw: &[(u128, u128); 5],
        zero_prefix: bool,
    ) {
        if zero_prefix {
            copy_radix8_from_strided_zero_128(src, src_stride, dst, tw);
        } else {
            copy_radix8_from_strided_nonzero_128(src, src_stride, dst, tw);
        }
    }

    #[inline(always)]
    fn copy_radix8_to_strided_uninit(
        src: [&[Self]; 8],
        dst: &mut [MaybeUninit<Self>],
        dst_stride: usize,
        tw: &[(u128, u128); 5],
        zero_prefix: bool,
    ) {
        if zero_prefix {
            copy_radix8_to_strided_zero_128(src, dst, dst_stride, tw);
        } else {
            copy_radix8_to_strided_nonzero_128(src, dst, dst_stride, tw);
        }
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
        copy_radix8_octants_128(novel_low, novel_high, dst, pw_eff, m4, m2_0, m1_0, p2, p4);
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
        if octants == 8 && novel_high.len() < e {
            let p2_w = w2[0] ^ w2[1];
            let p4_w = w1[0] ^ w1[2];
            assert!(w1[1] == (w1[0] ^ p2_w) && w1[3] == (w1[1] ^ p4_w));
            let tw = [
                Mul128::new(w4),
                Mul128::new(w2[0]),
                Mul128::new(w1[0]),
                Mul128::new(p2_w),
                Mul128::new(p4_w),
            ];
            let max_hi = novel_high.len();
            if max_hi > 0 {
                let low_oct = split_octants_ref_128(novel_low);
                let high_oct = split_high_octants_128(novel_high, e);
                let mut dst_vec: [&mut [MaybeUninit<Self>]; 8] = Default::default();
                let mut rest = &mut *dst;
                for slot in &mut dst_vec {
                    let (head, tail) = rest.split_at_mut(e);
                    *slot = head;
                    rest = tail;
                }
                copy_radix8_ragged_128(low_oct, high_oct, &mut dst_vec, 0, max_hi, Mul128::new(w_eff), &tw);
            }
            let tw_pairs =
                [tw[0].to_pair(), tw[1].to_pair(), tw[2].to_pair(), tw[3].to_pair(), tw[4].to_pair()];
            copy_radix8_strided_8_128(
                &novel_low[max_hi..],
                e,
                &mut dst[max_hi..],
                e,
                e - max_hi,
                &tw_pairs,
                w4 == 0 && w2[0] == 0 && w1[0] == 0,
            );
            return;
        }
        let low_oct = split_octants_ref_128(novel_low);
        let high_oct = split_high_octants_128(novel_high, e);
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
                let a = [
                    pw_eff.mul_xor_v(vector::load(novel_low[i].0), vector::load(novel_high[i].0)),
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
        let (dc, dr) = dst[..qcount].as_chunks_mut::<4>();
        let (lc, lr) = novel_low[..qcount].as_chunks::<4>();
        let (hc, hr) = novel_high.as_chunks::<4>();
        for ((d, l), h) in dc.iter_mut().zip(lc).zip(hc) {
            for k in 0..4 {
                d[k].write(Self(vector::store(p.mul_xor_v(vector::load(l[k].0), vector::load(h[k].0)))));
            }
        }
        for ((d, l), h) in dr.iter_mut().zip(lr).zip(hr) {
            d.write(Self(vector::store(p.mul_xor_v(vector::load(l.0), vector::load(h.0)))));
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
        if len <= 1 {
            return coefficients.first().copied().unwrap_or(Self::ZERO);
        }
        if len >= 768 {
            return eval_horner_blocked_128(coefficients, x);
        }
        if len >= 8 {
            return eval_horner_8way_128(coefficients, x);
        }
        let px = Mul128::new(x);
        let acc =
            coefficients.iter().rev().fold(vector::zero(), |acc, c| px.mul_xor_v(vector::load(c.0), acc));
        assert!(len < 8);
        Self(vector::store(acc))
    }

    fn transpose_butterfly_slice(low: &mut [Self], high: &mut [Self], twiddle: u128) {
        let len = low.len();
        assert_eq!(len, high.len());
        if twiddle == 0 {
            Self::xor_slice(low, high);
            return;
        }
        let p = Mul128::new(twiddle);
        let (lc, lr) = low.as_chunks_mut::<4>();
        let (hc, hr) = high.as_chunks_mut::<4>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..4 {
                let (nu, nv) =
                    vector::transpose_butterfly_step_v(vector::load(lo[k].0), vector::load(hi[k].0), p);
                lo[k] = Self(vector::store(nu));
                hi[k] = Self(vector::store(nv));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let (nu, nv) = vector::transpose_butterfly_step_v(vector::load(lo.0), vector::load(hi.0), p);
            *lo = Self(vector::store(nu));
            *hi = Self(vector::store(nv));
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
        let (lc, lr) = low.as_chunks::<4>();
        let (hc, hr) = high.as_chunks_mut::<4>();
        for (lo, hi) in lc.iter().zip(hc) {
            for k in 0..4 {
                hi[k] = Self(vector::store(p.mul_v(vector::load(lo[k].0))));
            }
        }
        for (lo, hi) in lr.iter().zip(&mut *hr) {
            *hi = Self(vector::store(p.mul_v(vector::load(lo.0))));
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
        let (lc, lr) = low.as_chunks_mut::<4>();
        let (hc, hr) = high.as_chunks_mut::<4>();
        for (lo, hi) in lc.iter_mut().zip(hc) {
            for k in 0..4 {
                let v = vector::load(hi[k].0);
                lo[k] = Self(vector::store(v));
                hi[k] = Self(vector::store(p.mul_v(v)));
            }
        }
        for (lo, hi) in lr.iter_mut().zip(&mut *hr) {
            let v = vector::load(hi.0);
            *lo = Self(vector::store(v));
            *hi = Self(vector::store(p.mul_v(v)));
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
        radix8_steps_128::<true>(values, base4, base2, base1, steps);
    }

    fn transpose_radix8_stride8(
        values: &mut [Self],
        base4: u128,
        base2: u128,
        base1: u128,
        steps: &[[(u128, u128); 16]; 3],
    ) {
        radix8_stride8_128::<true>(values, base4, base2, base1, steps);
    }

    fn transpose_radix8_outer(values: &mut [Self], w4: u128, w2: [u128; 2], w1: [u128; 4]) {
        let len = values.len();
        assert!(len >= 8 && len.is_multiple_of(8));
        let p2_w = w2[0] ^ w2[1];
        let p4_w = w1[0] ^ w1[2];
        assert!(w1[1] == (w1[0] ^ p2_w) && w1[3] == (w1[1] ^ p4_w));
        let (m4, m2_0, m1_0) = (Mul128::new(w4), Mul128::new(w2[0]), Mul128::new(w1[0]));
        let (p2, p4) = (Mul128::new(p2_w), Mul128::new(p4_w));
        let octants = split_octants_mut_128(values);
        let e = len / 8;
        for i in 0..e {
            let mut v: [V; 8] = std::array::from_fn(|k| vector::load(octants[k][i].0));
            vector::transpose_radix8_8lanes_compact(&mut v, m4, m2_0, m1_0, p2, p4);
            for k in 0..8 {
                octants[k][i] = Self(vector::store(v[k]));
            }
        }
        assert_eq!(e * 8, len);
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
