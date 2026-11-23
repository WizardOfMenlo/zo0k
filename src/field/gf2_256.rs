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
        let len = a.len();
        assert_eq!(len, b.len());
        let mut acc0 = WideAcc256::zero();
        let mut acc1 = WideAcc256::zero();
        let chunks = len / 2;
        for c in 0..chunks {
            let i = c * 2;
            acc0.add_product(a[i], b[i]);
            acc1.add_product(a[i + 1], b[i + 1]);
        }
        let mut acc = acc0.merge(acc1);
        for i in (chunks * 2)..len {
            acc.add_product(a[i], b[i]);
        }
        assert!(chunks * 2 <= len);
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
        for i in 0..len {
            let (al, ah) = (a_low[i], a_high[i]);
            let (bl, bh) = (b_low[i], b_high[i]);
            c_acc.add_product(al, bl);
            q_acc.add_product(Self(ah.0 ^ al.0, ah.1 ^ al.1), Self(bh.0 ^ bl.0, bh.1 ^ bl.1));
        }
        assert_eq!(a_low.len(), len);
        (c_acc.finish(), q_acc.finish())
    }

    fn fold_slice(low: &mut [Self], high: &[Self], challenge: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        if challenge == Self::ZERO {
            return;
        }
        let p = Mul256::new(challenge.0, challenge.1);
        let chunks = len / 2;
        for c in 0..chunks {
            let i = c * 2;
            for k in 0..2 {
                let (l0, l1) = load_256(low[i + k]);
                let (h0, h1) = load_256(high[i + k]);
                low[i + k] = store_256(p.mul_xor_pair(l0, l1, vector::xor(h0, l0), vector::xor(h1, l1)));
            }
        }
        for i in (chunks * 2)..len {
            let (l0, l1) = load_256(low[i]);
            let (h0, h1) = load_256(high[i]);
            low[i] = store_256(p.mul_xor_pair(l0, l1, vector::xor(h0, l0), vector::xor(h1, l1)));
        }
        assert!(chunks * 2 <= len);
    }

    fn eq_extend_uninit(low: &mut [Self], high: &mut [MaybeUninit<Self>], coordinate: Self) {
        let len = low.len();
        assert_eq!(len, high.len());
        let p = Mul256::new(coordinate.0, coordinate.1);
        let chunks = len / 2;
        for c in 0..chunks {
            let i = c * 2;
            for k in 0..2 {
                let (l0, l1) = load_256(low[i + k]);
                let (h0, h1) = p.mul_pair(l0, l1);
                high[i + k].write(store_256((h0, h1)));
                low[i + k] = store_256((vector::xor(l0, h0), vector::xor(l1, h1)));
            }
        }
        for i in (chunks * 2)..len {
            let (l0, l1) = load_256(low[i]);
            let (h0, h1) = p.mul_pair(l0, l1);
            high[i].write(store_256((h0, h1)));
            low[i] = store_256((vector::xor(l0, h0), vector::xor(l1, h1)));
        }
        assert!(chunks * 2 <= len);
    }

    fn add_scaled_slice(dst: &mut [Self], src: &[Self], scale: Self) {
        let len = dst.len();
        assert_eq!(len, src.len());
        if scale == Self::ZERO {
            return;
        }
        let p = Mul256::new(scale.0, scale.1);
        let chunks = len / 2;
        for c in 0..chunks {
            let i = c * 2;
            for k in 0..2 {
                let (d0, d1) = load_256(dst[i + k]);
                let (s0, s1) = load_256(src[i + k]);
                dst[i + k] = store_256(p.mul_xor_pair(d0, d1, s0, s1));
            }
        }
        for i in (chunks * 2)..len {
            let (d0, d1) = load_256(dst[i]);
            let (s0, s1) = load_256(src[i]);
            dst[i] = store_256(p.mul_xor_pair(d0, d1, s0, s1));
        }
        assert!(chunks * 2 <= len);
    }

    fn eval_horner_slice(coefficients: &[Self], x: Self) -> Self {
        if x.1 == 0 {
            return Self::eval_horner_base(coefficients, x.0);
        }
        let len = coefficients.len();
        if len < 8 {
            let px = Mul256::new(x.0, x.1);
            let acc = coefficients.iter().rev().fold((vector::zero(), vector::zero()), |acc, c| {
                let (c0, c1) = load_256(*c);
                px.mul_xor_pair(c0, c1, acc.0, acc.1)
            });
            return store_256(acc);
        }
        let x2 = x.square();
        let x4 = x2.square();
        let px4 = Mul256::new(x4.0, x4.1);
        let mut acc = [(vector::zero(), vector::zero()); 4];
        let rem = len % 4;
        let full = len - rem;
        for k in 0..rem {
            acc[k] = load_256(coefficients[full + k]);
        }
        let mut pos = full;
        for _ in 0..(full / 4) {
            pos -= 4;
            for k in 0..4 {
                let (c0, c1) = load_256(coefficients[pos + k]);
                acc[k] = px4.mul_xor_pair(c0, c1, acc[k].0, acc[k].1);
            }
        }
        assert_eq!(pos, 0);
        let a0 = store_256(acc[0]) + store_256(acc[1]) * x;
        let a1 = store_256(acc[2]) + store_256(acc[3]) * x;
        assert!(len >= 8);
        a0 + a1 * x2
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
        let chunks = len / 4;
        for c in 0..chunks {
            let i = c * 4;
            for k in 0..4 {
                let (t0, t1) = load_256(target[i + k]);
                target[i + k] = store_256((vector::xor(t0, s[k].0), vector::xor(t1, s[k].1)));
                s[k] = px4.mul_pair(s[k].0, s[k].1);
            }
        }
        let mut cur = s[0];
        for entry in &mut target[(chunks * 4)..] {
            let (t0, t1) = load_256(*entry);
            *entry = store_256((vector::xor(t0, cur.0), vector::xor(t1, cur.1)));
            cur = px.mul_pair(cur.0, cur.1);
        }
        assert!(chunks * 4 <= len);
    }

    fn xor_slice(dst: &mut [Self], src: &[Self]) {
        let len = dst.len();
        assert_eq!(len, src.len());
        let chunks = len / 2;
        for c in 0..chunks {
            let i = c * 2;
            for k in 0..2 {
                let d = load_256(dst[i + k]);
                let s = load_256(src[i + k]);
                dst[i + k] = store_256((vector::xor(d.0, s.0), vector::xor(d.1, s.1)));
            }
        }
        for i in (chunks * 2)..len {
            let d = load_256(dst[i]);
            let s = load_256(src[i]);
            dst[i] = store_256((vector::xor(d.0, s.0), vector::xor(d.1, s.1)));
        }
        assert!(chunks * 2 <= len);
    }
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
    let chunks = len / 4;
    for c in 0..chunks {
        let i = c * 4;
        for k in 0..4 {
            let (t0, t1) = load_256(target[i + k]);
            target[i + k] = store_256((vector::xor(t0, s[k].0), vector::xor(t1, s[k].1)));
            s[k] = (px4.mul_v(s[k].0), px4.mul_v(s[k].1));
        }
    }
    let mut cur = s[0];
    for entry in &mut target[(chunks * 4)..] {
        let (t0, t1) = load_256(*entry);
        *entry = store_256((vector::xor(t0, cur.0), vector::xor(t1, cur.1)));
        cur = (px.mul_v(cur.0), px.mul_v(cur.1));
    }
    assert!(chunks * 4 <= len);
}

#[inline(always)]
fn butterfly_step_v(r: V, q: V, p: Mul128) -> (V, V) {
    let nr = p.mul_xor_v(r, q);
    (nr, vector::xor(q, nr))
}

#[inline(always)]
fn butterfly_step_256(r: (V, V), q: (V, V), p: Mul128) -> ((V, V), (V, V)) {
    let nr0 = p.mul_xor_v(r.0, q.0);
    let nr1 = p.mul_xor_v(r.1, q.1);
    ((nr0, nr1), (vector::xor(q.0, nr0), vector::xor(q.1, nr1)))
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

#[inline(always)]
fn load_256(x: Gf2_256) -> (V, V) {
    (vector::load(x.0), vector::load(x.1))
}

#[inline(always)]
fn store_256(x: (V, V)) -> Gf2_256 {
    Gf2_256(vector::store(x.0), vector::store(x.1))
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
    for i in 0..len {
        let a0 = load_256(v0[i]);
        let a1 = load_256(v1[i]);
        let r2 = load_256(v2[i]);
        let r3 = load_256(v3[i]);
        let a2 = (vector::xor(r2.0, a0.0), vector::xor(r2.1, a0.1));
        let a3 = (vector::xor(r3.0, a1.0), vector::xor(r3.1, a1.1));
        let (b2, b3) = butterfly_step_256(a2, a3, pw1);
        v1[i] = store_256((vector::xor(a1.0, a0.0), vector::xor(a1.1, a0.1)));
        v2[i] = store_256(b2);
        v3[i] = store_256(b3);
    }
    assert_eq!(v0.len(), len);
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

fn radix8_outer_256(values: &mut [Gf2_256], m4: Mul128, m2: [Mul128; 2], m1: [Mul128; 4], zero_prefix: bool) {
    let len = values.len();
    assert!(len >= 8 && len.is_multiple_of(8));
    let e = len / 8;
    let ptr = values.as_mut_ptr();
    if zero_prefix {
        for i in 0..e {
            let mut v0 = [vector::zero(); 8];
            for (k, slot) in v0.iter_mut().enumerate() {
                *slot = vector::load(unsafe { (*ptr.add(k * e + i)).0 });
            }
            radix8_zero_prefix_8lanes(&mut v0, m2[1], m1);
            for (k, &val) in v0.iter().enumerate() {
                unsafe { (*ptr.add(k * e + i)).0 = vector::store(val) };
            }
            let mut v1 = [vector::zero(); 8];
            for (k, slot) in v1.iter_mut().enumerate() {
                *slot = vector::load(unsafe { (*ptr.add(k * e + i)).1 });
            }
            radix8_zero_prefix_8lanes(&mut v1, m2[1], m1);
            for (k, &val) in v1.iter().enumerate() {
                unsafe { (*ptr.add(k * e + i)).1 = vector::store(val) };
            }
        }
    } else {
        for i in 0..e {
            let mut v0 = [vector::zero(); 8];
            for (k, slot) in v0.iter_mut().enumerate() {
                *slot = vector::load(unsafe { (*ptr.add(k * e + i)).0 });
            }
            radix8_8lanes(&mut v0, m4, m2[0], m2[1], m1);
            for (k, &val) in v0.iter().enumerate() {
                unsafe { (*ptr.add(k * e + i)).0 = vector::store(val) };
            }
            let mut v1 = [vector::zero(); 8];
            for (k, slot) in v1.iter_mut().enumerate() {
                *slot = vector::load(unsafe { (*ptr.add(k * e + i)).1 });
            }
            radix8_8lanes(&mut v1, m4, m2[0], m2[1], m1);
            for (k, &val) in v1.iter().enumerate() {
                unsafe { (*ptr.add(k * e + i)).1 = vector::store(val) };
            }
        }
    }
    assert_eq!(e * 8, len);
}

fn copy_radix8_256_short(
    novel_low: &[Gf2_256],
    novel_high: &[Gf2_256],
    dst: &mut [MaybeUninit<Gf2_256>],
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
    let dp = dst.as_mut_ptr().cast::<Gf2_256>();
    for i in 0..qcount {
        let mut v0 = [vector::zero(); 8];
        v0[0] = pw_eff
            .mul_xor_v(vector::load(unsafe { (*nl.add(i)).0 }), vector::load(unsafe { (*nh.add(i)).0 }));
        for (k, slot) in v0.iter_mut().enumerate().skip(1) {
            *slot = vector::load(unsafe { (*nl.add(k * e + i)).0 });
        }
        radix8_8lanes(&mut v0, m4, m2[0], m2[1], m1);
        for (k, &val) in v0.iter().enumerate() {
            unsafe { std::ptr::write(&raw mut (*dp.add(k * e + i)).0, vector::store(val)) };
        }
        let mut v1 = [vector::zero(); 8];
        v1[0] = pw_eff
            .mul_xor_v(vector::load(unsafe { (*nl.add(i)).1 }), vector::load(unsafe { (*nh.add(i)).1 }));
        for (k, slot) in v1.iter_mut().enumerate().skip(1) {
            *slot = vector::load(unsafe { (*nl.add(k * e + i)).1 });
        }
        radix8_8lanes(&mut v1, m4, m2[0], m2[1], m1);
        for (k, &val) in v1.iter().enumerate() {
            unsafe { std::ptr::write(&raw mut (*dp.add(k * e + i)).1, vector::store(val)) };
        }
    }
    for i in qcount..e {
        let mut v0 = [vector::zero(); 8];
        for (k, slot) in v0.iter_mut().enumerate() {
            *slot = vector::load(unsafe { (*nl.add(k * e + i)).0 });
        }
        radix8_8lanes(&mut v0, m4, m2[0], m2[1], m1);
        for (k, &val) in v0.iter().enumerate() {
            unsafe { std::ptr::write(&raw mut (*dp.add(k * e + i)).0, vector::store(val)) };
        }
        let mut v1 = [vector::zero(); 8];
        for (k, slot) in v1.iter_mut().enumerate() {
            *slot = vector::load(unsafe { (*nl.add(k * e + i)).1 });
        }
        radix8_8lanes(&mut v1, m4, m2[0], m2[1], m1);
        for (k, &val) in v1.iter().enumerate() {
            unsafe { std::ptr::write(&raw mut (*dp.add(k * e + i)).1, vector::store(val)) };
        }
    }
    assert!(qcount <= e);
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
            for i in 0..len {
                high[i].0 ^= low[i].0;
                high[i].1 ^= low[i].1;
            }
            return;
        }
        let p = Mul128::new(twiddle);
        let chunks = len / 2;
        for c in 0..chunks {
            let i = c * 2;
            for k in 0..2 {
                let (nr, nq) = butterfly_step_256(load_256(low[i + k]), load_256(high[i + k]), p);
                low[i + k] = store_256(nr);
                high[i + k] = store_256(nq);
            }
        }
        for i in (chunks * 2)..len {
            let (nr, nq) = butterfly_step_256(load_256(low[i]), load_256(high[i]), p);
            low[i] = store_256(nr);
            high[i] = store_256(nq);
        }
        assert!(chunks * 2 <= len);
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
            let m2_1 = m2_0.xor(p2);
            let m1_0 = base1_m.xor(load_m(table0[(8 * i) & 255])).xor(load_m(table1[(8 * i) >> 8]));
            let m1_1 = m1_0.xor(p2);
            let m1 = [m1_0, m1_1, m1_0.xor(p4), m1_1.xor(p4)];
            let mut v0 = [
                vector::load(block[0].0),
                vector::load(block[1].0),
                vector::load(block[2].0),
                vector::load(block[3].0),
                vector::load(block[4].0),
                vector::load(block[5].0),
                vector::load(block[6].0),
                vector::load(block[7].0),
            ];
            radix8_8lanes(&mut v0, m4, m2_0, m2_1, m1);
            for j in 0..8 {
                block[j].0 = vector::store(v0[j]);
            }
            let mut v1 = [
                vector::load(block[0].1),
                vector::load(block[1].1),
                vector::load(block[2].1),
                vector::load(block[3].1),
                vector::load(block[4].1),
                vector::load(block[5].1),
                vector::load(block[6].1),
                vector::load(block[7].1),
            ];
            radix8_8lanes(&mut v1, m4, m2_0, m2_1, m1);
            for j in 0..8 {
                block[j].1 = vector::store(v1[j]);
            }
        }
    }

    fn butterfly_radix8_outer(values: &mut [Self], w4: u128, w2_0: u128, w2_1: u128, w1: [u128; 4]) {
        let len = values.len();
        assert!(len >= 8 && len.is_multiple_of(8));
        let m4 = Mul128::new(w4);
        let m2 = [Mul128::new(w2_0), Mul128::new(w2_1)];
        let m1 = [Mul128::new(w1[0]), Mul128::new(w1[1]), Mul128::new(w1[2]), Mul128::new(w1[3])];
        radix8_outer_256(values, m4, m2, m1, w4 == 0 && w2_0 == 0 && w1[0] == 0);
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
            return copy_radix8_256_short(novel_low, novel_high, dst, pw_eff, m4, m2, m1);
        }
        let nl = novel_low.as_ptr();
        let nh = novel_high.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..e {
            let mut v0 = [vector::zero(); 8];
            let mut v1 = [vector::zero(); 8];
            for k in 0..8 {
                let idx = k * e + i;
                let lo = load_256(unsafe { *nl.add(idx) });
                if idx < qcount {
                    let hi = load_256(unsafe { *nh.add(idx) });
                    v0[k] = pw_eff.mul_xor_v(lo.0, hi.0);
                    v1[k] = pw_eff.mul_xor_v(lo.1, hi.1);
                } else {
                    v0[k] = lo.0;
                    v1[k] = lo.1;
                }
            }
            radix8_8lanes(&mut v0, m4, m2[0], m2[1], m1);
            radix8_8lanes(&mut v1, m4, m2[0], m2[1], m1);
            for k in 0..8 {
                unsafe { (*dp.add(k * e + i)).write(store_256((v0[k], v1[k]))) };
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
                let (lo, hi) = (load_256(novel_low[i]), load_256(novel_high[i]));
                let a0 = (pw_eff.mul_xor_v(lo.0, hi.0), pw_eff.mul_xor_v(lo.1, hi.1));
                let a = [
                    a0,
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
        for i in 0..qcount {
            let lo = load_256(novel_low[i]);
            let hi = load_256(novel_high[i]);
            dst[i].write(store_256((p.mul_xor_v(lo.0, hi.0), p.mul_xor_v(lo.1, hi.1))));
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
            let acc = coefficients.iter().rev().fold((vector::zero(), vector::zero()), |acc, c| {
                (px.mul_xor_v(vector::load(c.0), acc.0), px.mul_xor_v(vector::load(c.1), acc.1))
            });
            return store_256(acc);
        }
        let x2 = vector::square(x);
        let x4 = vector::square(x2);
        let x8 = vector::square(x4);
        let px8 = Mul128::new(x8);
        let mut acc0 = [vector::zero(); 8];
        let mut acc1 = [vector::zero(); 8];
        let rem = len % 8;
        let full = len - rem;
        for k in 0..rem {
            acc0[k] = vector::load(coefficients[full + k].0);
            acc1[k] = vector::load(coefficients[full + k].1);
        }
        let mut pos = full;
        for _ in 0..(full / 8) {
            pos -= 8;
            for k in 0..8 {
                acc0[k] = px8.mul_xor_v(vector::load(coefficients[pos + k].0), acc0[k]);
                acc1[k] = px8.mul_xor_v(vector::load(coefficients[pos + k].1), acc1[k]);
            }
        }
        assert_eq!(pos, 0);
        let px = Mul128::new(x);
        let px2 = Mul128::new(x2);
        let px4 = Mul128::new(x4);
        let reduce_8 = |acc: [V; 8]| -> V {
            let a0 = px.mul_xor_v(acc[0], acc[1]);
            let a1 = px.mul_xor_v(acc[2], acc[3]);
            let a2 = px.mul_xor_v(acc[4], acc[5]);
            let a3 = px.mul_xor_v(acc[6], acc[7]);
            let b0 = px2.mul_xor_v(a0, a1);
            let b1 = px2.mul_xor_v(a2, a3);
            px4.mul_xor_v(b0, b1)
        };
        assert!(len >= 16);
        store_256((reduce_8(acc0), reduce_8(acc1)))
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
