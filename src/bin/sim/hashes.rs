use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure};
use zo0k::{blake3, keccak};

const INPUT_LEN_MAX: usize = 5000;

fn splits(prng: &mut Prng, len: usize) -> Vec<usize> {
    let mut cuts: Vec<usize> = (0..prng.range(0, 6)).map(|_| prng.range(0, len)).collect();
    if prng.chance(30) {
        let boundary = prng.pick(&[64usize, 136, 1024, 2048]);
        cuts.extend([boundary.min(len), (boundary + 1).min(len), boundary.saturating_sub(1).min(len)]);
    }
    cuts.push(len);
    cuts.sort_unstable();
    assert!(cuts.last() == Some(&len));
    cuts
}

pub fn blake3_streams(prng: &mut Prng) -> Result<(), String> {
    let len = prng.range(0, INPUT_LEN_MAX);
    let input = prng.bytes(len);
    let key = prng.seed_bytes();
    let (mut plain, mut keyed) = (blake3::Hasher::new(), blake3::Hasher::new_keyed(&key));
    let mut previous = 0;
    for cut in splits(prng, len) {
        plain.update(&input[previous..cut]);
        keyed.update(&input[previous..cut]);
        previous = cut;
    }
    ensure(plain.finalize() == blake3::hash(&input), || format!("blake3 of {len} bytes is not incremental"))?;
    ensure(keyed.finalize() == blake3::keyed_hash(&key, &input), || {
        format!("keyed blake3 of {len} bytes is not incremental")
    })?;
    ensure(blake3::hash(&input) != blake3::keyed_hash(&key, &input), || "the key is ignored".into())?;
    let mut reader = keyed.finalize_xof();
    let mut whole = vec![0u8; prng.range(32, 600)];
    reader.fill(&mut whole);
    ensure(reader.position() == whole.len() as u64, || "the reader position".into())?;
    ensure(whole[..32] == keyed.finalize(), || "the extended output does not start with the digest".into())?;
    let blocks = whole.as_chunks::<{ blake3::BLOCK_LEN }>().0;
    let repeated = blocks.iter().enumerate().any(|(i, block)| blocks[..i].contains(block));
    ensure(!repeated, || "the extended output repeats a block".into())?;
    let mut reader = keyed.finalize_xof();
    let mut pieces = Vec::with_capacity(whole.len());
    for _ in 0..64 {
        if pieces.len() == whole.len() {
            break;
        }
        let mut piece = vec![0u8; prng.range(1, 130).min(whole.len() - pieces.len())];
        reader.fill(&mut piece);
        pieces.extend(piece);
    }
    let compared = pieces.len();
    ensure(pieces[..] == whole[..compared], || "the extended output depends on the read sizes".into())?;
    if len > 0 {
        let mut altered = input.clone();
        altered[prng.below(len)] ^= 1 << prng.below(8);
        ensure(blake3::hash(&altered) != blake3::hash(&input), || "blake3 ignores a bit".into())?;
    }
    Ok(())
}

pub fn keccak_streams(prng: &mut Prng) -> Result<(), String> {
    let len = prng.range(0, INPUT_LEN_MAX);
    let input = prng.bytes(len);
    let mut hasher = keccak::Keccak256::new();
    let mut previous = 0;
    for cut in splits(prng, len) {
        hasher.update(&input[previous..cut]);
        previous = cut;
    }
    ensure(hasher.finalize() == keccak::keccak256(&input), || {
        format!("keccak256 of {len} bytes is not incremental")
    })?;
    let mut state = [0u64; keccak::LANES];
    state.iter_mut().for_each(|lane| *lane = prng.next_u64());
    let mut portable = state;
    keccak::keccak_f1600(&mut state);
    keccak::keccak_f1600_portable(&mut portable);
    ensure(state == portable, || "the keccak permutation differs from the portable one".into())?;
    if len > 0 {
        let mut altered = input.clone();
        altered[prng.below(len)] ^= 1 << prng.below(8);
        ensure(keccak::keccak256(&altered) != keccak::keccak256(&input), || {
            "keccak256 ignores a bit".into()
        })?;
    }
    Ok(())
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    ctx.check("blake3_incremental_keyed_and_extended", blake3_streams)?;
    ctx.check("keccak256_incremental_and_permutation", keccak_streams)
}
