use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure};
use zo0k::field::BinaryField;
use zo0k::{Field, FieldSize, Gf2_128, Gf2_256};

const SAMPLES: usize = 48;

pub fn mul_reference_128(mut a: u128, mut b: u128) -> u128 {
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
    assert_eq!(b, 0);
    product
}

pub fn limbs<F: Field>(element: F) -> (u128, u128) {
    assert!(F::BYTES == 16 || F::BYTES == 32);
    let mut bytes = vec![0u8; F::BYTES];
    element.write_bytes(&mut bytes);
    let low = u128::from_le_bytes(bytes[..16].try_into().expect("sixteen bytes"));
    let high =
        if F::BYTES == 32 { u128::from_le_bytes(bytes[16..].try_into().expect("sixteen bytes")) } else { 0 };
    (low, high)
}

pub fn from_limbs<F: Field>(low: u128, high: u128) -> F {
    assert!(F::BYTES == 16 || F::BYTES == 32);
    assert!(F::BYTES == 32 || high == 0);
    let mut bytes = low.to_le_bytes().to_vec();
    if F::BYTES == 32 {
        bytes.extend(high.to_le_bytes());
    }
    F::from_bytes(&bytes).expect("every string of the field length encodes an element")
}

pub fn mul_reference<F: Field>(p: F, q: F) -> F {
    let ((a, b), (c, d)) = (limbs(p), limbs(q));
    let m = mul_reference_128;
    if F::BYTES == 16 {
        assert_eq!((b, d), (0, 0));
        return from_limbs(m(a, c), 0);
    }
    let bd = m(b, d);
    let product = from_limbs(m(a, c) ^ m(bd, 1 << 121), m(a, d) ^ m(b, c) ^ bd);
    assert_eq!(F::BYTES, 32);
    product
}

pub fn convert<F: Field, G: Field>(element: F) -> G {
    let mut bytes = vec![0u8; F::BYTES];
    element.write_bytes(&mut bytes);
    bytes.resize(G::BYTES, 0);
    let converted = G::from_bytes(&bytes).expect("every string of the field length encodes an element");
    assert_eq!(bytes.len(), G::BYTES);
    converted
}

pub fn uniform<F: Field>(prng: &mut Prng) -> F {
    let bytes = prng.bytes(F::BYTES);
    let element = F::from_bytes(&bytes).expect("every string of the field length encodes an element");
    assert_eq!(bytes.len(), F::BYTES);
    element
}

pub fn uniforms<F: Field>(prng: &mut Prng, count: usize) -> Vec<F> {
    let elements: Vec<F> = (0..count).map(|_| uniform(prng)).collect();
    assert_eq!(elements.len(), count);
    elements
}

pub fn nonzero<F: Field>(prng: &mut Prng) -> F {
    for _ in 0..64 {
        let element = uniform(prng);
        if element != F::ZERO {
            return element;
        }
    }
    F::ONE
}

pub fn special<F: Field>(prng: &mut Prng) -> F {
    let mut bytes = vec![0u8; F::BYTES];
    match prng.below(8) {
        0 => {}
        1 => bytes[0] = 1,
        2 => bytes.fill(0xff),
        3 => {
            let bit = prng.below(8 * F::BYTES);
            bytes[bit / 8] = 1 << (bit % 8);
        }
        4 => bytes[F::BYTES - 1] = 0x80,
        _ => bytes = prng.bytes(F::BYTES),
    }
    assert_eq!(bytes.len(), F::BYTES);
    F::from_bytes(&bytes).expect("every string of the field length encodes an element")
}

fn ring_axioms<F: Field>(p: F, q: F, r: F) -> Result<(), String> {
    let show = || format!("{} on {p:?}, {q:?}, {r:?}", F::NAME);
    ensure(p * (q + r) == p * q + p * r, || format!("distributivity fails {}", show()))?;
    ensure((p * q) * r == p * (q * r), || format!("multiplication is not associative {}", show()))?;
    ensure((p + q) + r == p + (q + r), || format!("addition is not associative {}", show()))?;
    ensure(p * q == q * p && p + q == q + p, || format!("commutativity fails {}", show()))?;
    ensure(p - q == p + (-q) && -p == p && p + p == F::ZERO, || format!("negation fails {}", show()))?;
    ensure(p + F::ZERO == p && p * F::ONE == p && p * F::ZERO == F::ZERO, || {
        format!("identities fail {}", show())
    })?;
    ensure(p.square() == p * p, || format!("square differs from product {}", show()))?;
    ensure(p.pow(5) == p * p * p * p * p && p.pow(0) == F::ONE && p.pow(1) == p, || {
        format!("pow fails {}", show())
    })?;
    let (mut sum, mut difference, mut product) = (p, p, p);
    sum += q;
    difference -= q;
    product *= q;
    ensure(sum == p + q && difference == p - q && product == p * q, || {
        format!("compound assignment {}", show())
    })?;
    if p != F::ZERO {
        let inverse = p.inverse();
        ensure(p * inverse == F::ONE && inverse.inverse() == p, || format!("inverse fails {}", show()))?;
    }
    Ok(())
}

fn reference_products<F: BinaryField>(p: F, q: F, w: u128) -> Result<(), String> {
    let show = || format!("{} on {p:?}, {q:?}, {w:#x}", F::NAME);
    ensure(p * q == mul_reference(p, q), || format!("product differs from the reference {}", show()))?;
    ensure(p.square() == mul_reference(p, p), || format!("square differs from the reference {}", show()))?;
    let base = F::from_base(w);
    ensure(p.mul_base(w) == p * base, || format!("mul_base differs from the product {}", show()))?;
    ensure(p.mul_base(w) == mul_reference(p, base), || {
        format!("mul_base differs from the reference {}", show())
    })?;
    let (v, _) = limbs(q);
    let embedded = F::from_base(mul_reference_128(v, w));
    ensure(embedded == F::from_base(v) * base, || {
        format!("the subfield embedding is not multiplicative {}", show())
    })?;
    ensure(F::from_base(v ^ w) == F::from_base(v) + base, || {
        format!("the subfield embedding is not additive {}", show())
    })?;
    Ok(())
}

fn encodings<F: Field>(p: F, prng: &mut Prng) -> Result<(), String> {
    let mut bytes = vec![0u8; F::BYTES];
    p.write_bytes(&mut bytes);
    ensure(F::from_bytes(&bytes) == Some(p), || {
        format!("{} encoding does not round trip for {p:?}", F::NAME)
    })?;
    for len in [0, F::BYTES - 1, F::BYTES + 1, 2 * F::BYTES] {
        let other = vec![0u8; len];
        ensure(F::from_bytes(&other).is_none(), || {
            format!("{} accepts an encoding of {len} bytes", F::NAME)
        })?;
    }
    let fresh = prng.bytes(F::BYTES);
    let mut source = fresh.clone();
    let sampled = F::sample(|target| {
        let take = target.len().min(source.len());
        target[..take].copy_from_slice(&source[..take]);
        source.drain(..take);
    });
    ensure(Some(sampled) == F::from_bytes(&fresh), || format!("{} sample differs from decoding", F::NAME))?;
    Ok(())
}

fn constants<F: Field>() -> Result<(), String> {
    ensure(F::from_integer(6) == F::ZERO && F::from_integer(7) == F::ONE, || {
        format!("{} from_integer", F::NAME)
    })?;
    ensure([F::ONE, F::ONE, F::ONE].iter().sum::<F>() == F::ONE, || {
        format!("{} sum of references", F::NAME)
    })?;
    ensure([F::ONE, F::ONE].into_iter().sum::<F>() == F::ZERO, || format!("{} sum of values", F::NAME))?;
    ensure(F::SIZE == FieldSize { bits: F::BITS, bytes: F::BYTES }, || format!("{} size", F::NAME))?;
    ensure(F::BITS as usize == 8 * F::BYTES, || {
        format!("{} has {} bits in {} bytes", F::NAME, F::BITS, F::BYTES)
    })?;
    ensure(F::default() == F::ZERO && F::ONE != F::ZERO, || format!("{} default or one", F::NAME))?;
    Ok(())
}

pub fn axioms<F: BinaryField>(prng: &mut Prng) -> Result<(), String> {
    constants::<F>()?;
    let mut total = F::ZERO;
    let mut elements = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let (p, q, r): (F, F, F) = (special(prng), special(prng), uniform(prng));
        let w = limbs(special::<Gf2_128>(prng)).0;
        ring_axioms(p, q, r)?;
        reference_products(p, q, w)?;
        encodings(p, prng)?;
        total += p;
        elements.push(p);
    }
    ensure(elements.iter().sum::<F>() == total, || format!("{} sum of a list", F::NAME))?;
    ensure(elements.len() == SAMPLES, || "sample count".to_string())
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    ctx.check("field_axioms_gf2_128", axioms::<Gf2_128>)?;
    ctx.check("field_axioms_gf2_256", axioms::<Gf2_256>)?;
    Ok(())
}

pub fn moduli_are_irreducible() -> Result<(), String> {
    let mut power = 2u128;
    for _ in 0..64 {
        power = mul_reference_128(power, power);
    }
    let difference = Gf2_128::new(power ^ 2);
    ensure(difference * difference.inverse() == Gf2_128::ONE, || "x^(2^64) - x is not invertible".into())?;
    for _ in 0..64 {
        power = mul_reference_128(power, power);
    }
    ensure(power == 2, || "x^(2^128) differs from x".into())?;
    let mut conjugate = 1u128 << 121;
    let mut trace = conjugate;
    for _ in 1..128 {
        conjugate = mul_reference_128(conjugate, conjugate);
        trace ^= conjugate;
    }
    ensure(trace == 1, || "the extension constant has trace zero".into())
}

pub fn edge_products() -> Result<(), String> {
    let edge = [0u128, 1, u128::MAX, 1 << 127, 1 << 121, (1 << 7) - 1, 1 << 64, (1 << 64) - 1];
    for &a in &edge {
        for &b in &edge {
            let product = Gf2_128::new(a) * Gf2_128::new(b);
            ensure(product.limb() == mul_reference_128(a, b), || {
                format!("gf2_128 product of {a:#x} and {b:#x}")
            })?;
            for &c in &edge {
                let (p, q) = (Gf2_256::new(a, b), Gf2_256::new(c, a ^ b));
                ensure(p * q == mul_reference(p, q), || format!("gf2_256 product of {p:?} and {q:?}"))?;
                ensure(p.square() == mul_reference(p, p), || format!("gf2_256 square of {p:?}"))?;
            }
        }
    }
    Ok(())
}
