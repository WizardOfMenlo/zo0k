use crate::fields::uniforms;
use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure};
use zo0k::hash::{DIGEST_BYTES, Domain};
use zo0k::merkle::{self, MerkleTree};
use zo0k::oracle::{Opening, Oracle};
use zo0k::transcript::{Challenges, NONCE_BYTES, ProverTranscript, VerifierTranscript};
use zo0k::{Blake3, Digest, Error, Field, Gf2_128, Gf2_256, Hasher, Keccak256};

pub const DOMAINS: [Domain; 5] =
    [Domain::MerkleLeaf, Domain::MerkleNode, Domain::Transcript, Domain::Challenge, Domain::ProofOfWork];

pub fn hashing<H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let len = prng.range(0, 700);
    let bytes = prng.bytes(len);
    let (first, second) = (prng.range(0, len), prng.range(0, len));
    let (first, second) = (first.min(second), first.max(second));
    let mut digests: Vec<Digest> = Vec::with_capacity(DOMAINS.len());
    for domain in DOMAINS {
        let mut hasher = H::new(domain);
        hasher.update(&bytes[..first]);
        let mut other = hasher.clone();
        hasher.update(&bytes[first..second]);
        hasher.update(&bytes[second..]);
        let digest = hasher.finalize();
        ensure(digest == H::hash(domain, &bytes), || {
            format!("{} is not incremental at {first}, {second} of {len}", H::NAME)
        })?;
        other.update(&bytes[first..]);
        other.update(b"x");
        ensure(other.finalize() != digest, || format!("{} ignores an appended byte", H::NAME))?;
        ensure(!digests.contains(&digest), || format!("{} does not separate {domain:?}", H::NAME))?;
        digests.push(digest);
    }
    if len > 0 {
        let mut altered = bytes.clone();
        let position = prng.below(len);
        altered[position] ^= 1 << prng.below(8);
        ensure(H::hash(Domain::Transcript, &altered) != digests[2], || {
            format!("{} ignores byte {position}", H::NAME)
        })?;
    }
    Ok(())
}

pub fn hash_functions_differ(prng: &mut Prng) -> Result<(), String> {
    let len = prng.range(0, 200);
    let bytes = prng.bytes(len);
    for domain in DOMAINS {
        ensure(Blake3::hash(domain, &bytes) != Keccak256::hash(domain, &bytes), || {
            format!("blake3 equals keccak256 on {domain:?}")
        })?;
    }
    ensure(Blake3::NAME != Keccak256::NAME, || "the hash functions share a name".into())
}

pub fn reference_leaf<H: Hasher, F: Field>(position: usize, column: &[F]) -> Digest {
    let mut bytes = (position as u64).to_le_bytes().to_vec();
    let mut encoding = vec![0u8; F::BYTES];
    for element in column {
        element.write_bytes(&mut encoding);
        bytes.extend_from_slice(&encoding);
    }
    assert_eq!(bytes.len(), 8 + F::BYTES * column.len());
    H::hash(Domain::MerkleLeaf, &bytes)
}

pub fn reference_root<H: Hasher>(leaves: &[Digest]) -> Digest {
    assert!(!leaves.is_empty());
    let mut level = leaves.to_vec();
    for _ in 0..64 {
        if level.len() == 1 {
            break;
        }
        level = level.chunks(2).map(|children| H::hash(Domain::MerkleNode, &children.concat())).collect();
    }
    assert_eq!(level.len(), 1);
    level[0]
}

pub fn missing_siblings(num_leaves: usize, positions: &[usize]) -> usize {
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(positions.last().is_none_or(|&last| last < num_leaves));
    let mut known = positions.to_vec();
    let mut width = num_leaves;
    let mut missing = 0;
    for _ in 0..64 {
        if width <= 1 {
            break;
        }
        let mut parents: Vec<usize> = known.iter().map(|node| node / 2).collect();
        parents.dedup();
        for &parent in &parents {
            let children = (2 * parent..(2 * parent + 2).min(width)).filter(|child| !known.contains(child));
            missing += children.count();
        }
        known = parents;
        width = width.div_ceil(2);
    }
    missing
}

fn verify_opening<H: Hasher>(
    count: usize,
    opened: &[(usize, Digest)],
    opening: &[Digest],
) -> zo0k::Result<(Digest, bool)> {
    let mut digests = opening.iter();
    let root = merkle::root_from_opening::<H>(count, opened, || {
        digests.next().copied().ok_or(Error::InvalidProof("the opening is too short"))
    })?;
    Ok((root, digests.next().is_none()))
}

pub fn merkle_trees<H: Hasher, F: Field>(prng: &mut Prng) -> Result<(), String> {
    let count = prng.range(1, 40);
    let mut columns: Vec<Vec<F>> = Vec::with_capacity(count);
    for _ in 0..count {
        let width = prng.range(0, 70);
        columns.push(uniforms(prng, width));
    }
    let leaves: Vec<Digest> =
        columns.iter().enumerate().map(|(i, column)| reference_leaf::<H, F>(i, column)).collect();
    for (i, column) in columns.iter().enumerate() {
        let digest = merkle::leaf_digest::<H, F>(i, column.iter().copied());
        ensure(digest == leaves[i], || {
            format!("leaf digest {i} of {} elements differs from the reference", column.len())
        })?;
    }
    let tree = MerkleTree::new::<H>(leaves.clone());
    ensure(tree.root() == reference_root::<H>(&leaves), || {
        format!("root of {count} leaves differs from the reference")
    })?;
    let positions = prng.subset(count, 8);
    let opening = tree.open(&positions);
    let expected = missing_siblings(count, &positions);
    ensure(opening.len() == expected, || {
        format!("{count} leaves at {positions:?}: {} digests, not {expected}", opening.len())
    })?;
    let opened: Vec<(usize, Digest)> = positions.iter().map(|&p| (p, leaves[p])).collect();
    let verified = verify_opening::<H>(count, &opened, &opening).map_err(|e| e.to_string())?;
    ensure(verified == (tree.root(), true), || {
        format!("an honest opening of {positions:?} in {count} fails")
    })?;
    altered_openings::<H>(prng, &tree, count, &opened, &opening)
}

fn altered_openings<H: Hasher>(
    prng: &mut Prng,
    tree: &MerkleTree,
    count: usize,
    opened: &[(usize, Digest)],
    opening: &[Digest],
) -> Result<(), String> {
    assert!(!opened.is_empty() && count > 0);
    let accepts = |opened: &[(usize, Digest)], opening: &[Digest]| {
        verify_opening::<H>(count, opened, opening).is_ok_and(|(root, used)| root == tree.root() && used)
    };
    if !opening.is_empty() {
        let mut bad = opening.to_vec();
        let index = prng.below(bad.len());
        bad[index][prng.below(DIGEST_BYTES)] ^= 1 << prng.below(8);
        ensure(!accepts(opened, &bad), || format!("an altered digest {index} is accepted"))?;
        ensure(verify_opening::<H>(count, opened, &opening[1..]).is_err(), || {
            "a short opening is accepted".into()
        })?;
    }
    let mut longer = opening.to_vec();
    longer.push([0; DIGEST_BYTES]);
    ensure(!accepts(opened, &longer), || "a long opening is accepted".into())?;
    let mut bad_leaf = opened.to_vec();
    bad_leaf[0].1[0] ^= 1;
    ensure(!accepts(&bad_leaf, opening), || "an altered leaf is accepted".into())?;
    let invalid: [Vec<(usize, Digest)>; 3] =
        [vec![], vec![(count, opened[0].1)], [opened, &opened[..1]].concat()];
    for bad in invalid {
        ensure(merkle::root_from_opening::<H>(count, &bad, || Ok([0; 32])).is_err(), || {
            format!("invalid positions {bad:?} are accepted")
        })?;
    }
    if opened.len() >= 2 {
        let reversed: Vec<(usize, Digest)> = opened.iter().rev().copied().collect();
        ensure(merkle::root_from_opening::<H>(count, &reversed, || Ok([0; 32])).is_err(), || {
            "unsorted positions are accepted".into()
        })?;
    }
    let moved = (opened[0].0 + 1..count).find(|p| opened.iter().all(|leaf| leaf.0 != *p));
    if let Some(position) = moved.filter(|_| opened.len() == 1) {
        let shifted = [(position, opened[0].1)];
        ensure(!accepts(&shifted, opening), || format!("a leaf moved to {position} is accepted"))?;
    }
    Ok(())
}

pub fn node_digests<H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let (a, b, c) = (prng.seed_bytes(), prng.seed_bytes(), prng.seed_bytes());
    let pair = MerkleTree::new::<H>(vec![a, b]).root();
    ensure(pair == H::hash(Domain::MerkleNode, &[a, b].concat()), || {
        "a node is not the hash of its children".into()
    })?;
    ensure(pair != H::hash(Domain::MerkleLeaf, &[a, b].concat()), || {
        "nodes and leaves share a domain".into()
    })?;
    let ragged = MerkleTree::new::<H>(vec![a, b, c]).root();
    let expected = H::hash(Domain::MerkleNode, &[pair, H::hash(Domain::MerkleNode, &c)].concat());
    ensure(ragged == expected, || "a ragged tree hashes its last node alone".into())?;
    ensure(MerkleTree::new::<H>(vec![a]).root() == a, || "a single leaf is the root".into())
}

pub fn oracles<F: Field, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let (len, width) = (prng.range(1, 50), prng.range(1, 6));
    let rows: Vec<Vec<F>> = (0..width).map(|_| uniforms(prng, len)).collect();
    let positions: Vec<usize> = (0..prng.range(1, 8)).map(|_| prng.below(len)).collect();
    let sent = prng.range(1, width);
    let oracle = Oracle::commit::<H>(rows.clone());
    ensure(oracle.shape() == (width, len), || format!("shape {:?} is not {width} x {len}", oracle.shape()))?;
    let mut prover = ProverTranscript::<H>::new(b"sim-oracle");
    let opened = oracle.open(&mut prover, &positions, sent);
    let end: F = prover.challenge_field();
    let proof = prover.finish();
    let complete = |position: usize, _: &[F]| rows[sent..].iter().map(|row| row[position]).collect();
    let read = |proof: &[u8]| {
        let mut verifier = VerifierTranscript::<H>::new(b"sim-oracle", proof);
        let opening = Opening::read(&mut verifier, &oracle.root(), len, &positions, sent, complete)?;
        let end: F = verifier.challenge_field();
        verifier.finish()?;
        Ok::<_, Error>((opening, end))
    };
    let (opening, verifier_end) = read(&proof).map_err(|e| format!("an honest opening fails: {e}"))?;
    ensure(verifier_end == end, || "the transcripts diverge after an opening".into())?;
    for (column, &position) in opening.columns(&positions).zip(&positions) {
        let expected: Vec<F> = rows[..sent].iter().map(|row| row[position]).collect();
        ensure(column == expected, || format!("column {position} differs"))?;
    }
    ensure(opened.columns(&positions).eq(opening.columns(&positions)), || {
        "prover and verifier columns differ".into()
    })?;
    let flips: Vec<usize> = if proof.len() <= 1500 {
        (0..proof.len()).collect()
    } else {
        (0..200).map(|_| prng.below(proof.len())).collect()
    };
    for byte in flips {
        let mut altered = proof.clone();
        altered[byte] ^= 1 << (byte % 8);
        ensure(read(&altered).is_err(), || {
            format!("an opening altered at byte {byte} of {} is accepted", proof.len())
        })?;
    }
    ensure(read(&proof[..proof.len() - 1]).is_err(), || "a truncated opening is accepted".into())?;
    let wrong = |_: usize, _: &[F]| vec![F::ZERO; width - sent];
    let mut verifier = VerifierTranscript::<H>::new(b"sim-oracle", &proof);
    let result = Opening::read(&mut verifier, &oracle.root(), len, &positions, sent, wrong);
    ensure(result.is_err() == (sent < width), || {
        format!("a wrong completion of {} rows is not detected", width - sent)
    })
}

#[derive(Clone, Copy, Debug)]
enum Step {
    Bytes(usize),
    Digest,
    Field,
    Fields(usize),
    Work(u32),
    Challenge,
    Challenges(usize),
    Positions(usize, usize),
    Nonzero(usize),
}

fn random_steps(prng: &mut Prng) -> Vec<Step> {
    let count = prng.range(1, 12);
    let steps: Vec<Step> = (0..count)
        .map(|_| match prng.below(9) {
            0 => Step::Bytes(prng.range(0, 80)),
            1 => Step::Digest,
            2 => Step::Field,
            3 => Step::Fields(prng.range(0, 5)),
            4 => Step::Work(prng.range(0, 8) as u32),
            5 => Step::Challenge,
            6 => Step::Challenges(prng.range(1, 4)),
            7 => Step::Positions(prng.range(1, 1000), prng.range(0, 40)),
            _ => Step::Nonzero(prng.range(1, 4)),
        })
        .collect();
    assert_eq!(steps.len(), count);
    steps
}

fn run_prover<F: Field, H: Hasher>(
    statement: &[u8],
    steps: &[Step],
    prng: &mut Prng,
) -> (Vec<u8>, Vec<Vec<u64>>, usize) {
    let mut prover = ProverTranscript::<H>::new(statement);
    let mut seen = Vec::new();
    let mut expected_len = 0;
    for step in steps {
        match *step {
            Step::Bytes(len) => {
                prover.send_bytes(&prng.bytes(len));
                expected_len += len;
            }
            Step::Digest => {
                prover.send_digest(&prng.seed_bytes());
                expected_len += DIGEST_BYTES;
            }
            Step::Field => {
                prover.send_field(uniforms::<F>(prng, 1)[0]);
                expected_len += F::BYTES;
            }
            Step::Fields(count) => {
                prover.send_fields(&uniforms::<F>(prng, count));
                expected_len += count * F::BYTES;
            }
            Step::Work(bits) => {
                prover.prove_work(bits);
                expected_len += if bits > 0 { NONCE_BYTES } else { 0 };
            }
            Step::Challenge => seen.push(fingerprints(&[prover.challenge_field::<F>()])),
            Step::Challenges(count) => seen.push(fingerprints(&prover.challenge_fields::<F>(count))),
            Step::Positions(bound, count) => {
                seen.push(prover.challenge_positions(bound, count).iter().map(|&p| p as u64).collect())
            }
            Step::Nonzero(count) => seen.push(fingerprints(&prover.challenge_nonzero_vector::<F>(count))),
        }
    }
    let proof = prover.finish();
    assert!(seen.len() <= steps.len());
    (proof, seen, expected_len)
}

fn fingerprints<F: Field>(elements: &[F]) -> Vec<u64> {
    let mut bytes = vec![0u8; F::BYTES];
    let prints: Vec<u64> = elements
        .iter()
        .map(|element| {
            element.write_bytes(&mut bytes);
            bytes.chunks(8).fold(0u64, |acc, chunk| {
                acc.rotate_left(13) ^ u64::from_le_bytes(chunk.try_into().expect("eight bytes"))
            })
        })
        .collect();
    assert_eq!(prints.len(), elements.len());
    prints
}

fn run_verifier<F: Field, H: Hasher>(
    statement: &[u8],
    steps: &[Step],
    proof: &[u8],
) -> zo0k::Result<Vec<Vec<u64>>> {
    let mut verifier = VerifierTranscript::<H>::new(statement, proof);
    let mut seen = Vec::new();
    for step in steps {
        match *step {
            Step::Bytes(len) => {
                verifier.read_bytes(len)?;
            }
            Step::Digest => {
                verifier.read_digest()?;
            }
            Step::Field => {
                verifier.read_field::<F>()?;
            }
            Step::Fields(count) => {
                verifier.read_fields::<F>(count)?;
            }
            Step::Work(bits) => verifier.check_work(bits)?,
            Step::Challenge => seen.push(fingerprints(&[verifier.challenge_field::<F>()])),
            Step::Challenges(count) => seen.push(fingerprints(&verifier.challenge_fields::<F>(count))),
            Step::Positions(bound, count) => {
                seen.push(verifier.challenge_positions(bound, count).iter().map(|&p| p as u64).collect())
            }
            Step::Nonzero(count) => seen.push(fingerprints(&verifier.challenge_nonzero_vector::<F>(count))),
        }
    }
    verifier.finish()?;
    Ok(seen)
}

pub fn transcripts<F: Field, H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let steps = random_steps(prng);
    let statement_len = prng.range(0, 40);
    let statement = prng.bytes(statement_len);
    let (proof, seen, expected_len) = run_prover::<F, H>(&statement, &steps, prng);
    ensure(proof.len() == expected_len, || {
        format!("{steps:?}: proof of {} bytes, not {expected_len}", proof.len())
    })?;
    let replayed = run_verifier::<F, H>(&statement, &steps, &proof).map_err(|e| format!("{steps:?}: {e}"))?;
    ensure(replayed == seen, || format!("{steps:?}: the verifier derives other challenges"))?;
    for (step, values) in steps
        .iter()
        .filter(|s| {
            !matches!(s, Step::Bytes(_) | Step::Digest | Step::Field | Step::Fields(_) | Step::Work(_))
        })
        .zip(&seen)
    {
        if let Step::Positions(bound, count) = *step {
            ensure(values.len() == count && values.iter().all(|&p| (p as usize) < bound), || {
                format!("positions out of range {bound}")
            })?;
        }
        if let Step::Nonzero(_) = *step {
            ensure(values.iter().any(|&v| v != 0), || "a nonzero vector is zero".into())?;
        }
    }
    let mut other = statement.clone();
    other.push(1);
    let has_field_challenge =
        steps.iter().any(|step| matches!(step, Step::Challenge | Step::Challenges(_) | Step::Nonzero(_)));
    if has_field_challenge {
        let diverged = run_verifier::<F, H>(&other, &steps, &proof).map_or(true, |values| values != seen);
        ensure(diverged, || format!("{steps:?}: the challenges do not depend on the statement"))?;
    }
    if !proof.is_empty() {
        ensure(run_verifier::<F, H>(&statement, &steps, &proof[..proof.len() - 1]).is_err(), || {
            "a short transcript is accepted".into()
        })?;
    }
    let mut longer = proof.clone();
    longer.push(0);
    ensure(run_verifier::<F, H>(&statement, &steps, &longer).is_err(), || {
        "a long transcript is accepted".into()
    })
}

pub fn proof_of_work_bits<H: Hasher>(seed: &Digest, nonce: u64) -> u32 {
    let mut input = [0u8; DIGEST_BYTES + NONCE_BYTES];
    input[..DIGEST_BYTES].copy_from_slice(seed);
    input[DIGEST_BYTES..].copy_from_slice(&nonce.to_le_bytes());
    let digest = H::hash(Domain::ProofOfWork, &input);
    let bits = u64::from_be_bytes(digest[..8].try_into().expect("eight bytes")).leading_zeros();
    assert!(bits <= 64);
    bits
}

pub fn proofs_of_work<H: Hasher>(prng: &mut Prng) -> Result<(), String> {
    let bits = prng.range(1, 10) as u32;
    let statement_len = prng.range(0, 40);
    let statement = prng.bytes(statement_len);
    let mut prover = ProverTranscript::<H>::new(&statement);
    prover.prove_work(bits);
    let proof = prover.finish();
    ensure(proof.len() == NONCE_BYTES, || format!("a proof of work has {} bytes", proof.len()))?;
    let nonce = u64::from_le_bytes(proof[..].try_into().expect("eight bytes"));
    let mut twin = ProverTranscript::<H>::new(&statement);
    let mut seed = [0u8; DIGEST_BYTES];
    twin.challenge_stream().fill(&mut seed);
    ensure(proof_of_work_bits::<H>(&seed, nonce) >= bits, || {
        format!("nonce {nonce} does not meet {bits} bits")
    })?;
    ensure(nonce < 1 << 20, || format!("nonce {nonce} for {bits} bits is implausibly large"))?;
    let smaller = (0..nonce).find(|&n| proof_of_work_bits::<H>(&seed, n) >= bits);
    ensure(smaller.is_none(), || format!("nonce {nonce} is not the smallest for {bits} bits: {smaller:?}"))?;
    let check = |proof: &[u8], bits: u32| VerifierTranscript::<H>::new(&statement, proof).check_work(bits);
    ensure(check(&proof, bits).is_ok(), || "an honest proof of work is rejected".into())?;
    let achieved = proof_of_work_bits::<H>(&seed, nonce);
    ensure(check(&proof, achieved + 1).is_err(), || {
        format!("a proof of work of {achieved} bits passes for more")
    })?;
    let weak =
        (0..1u64 << 16).find(|&n| proof_of_work_bits::<H>(&seed, n) < bits).expect("some nonce is weak");
    ensure(check(&weak.to_le_bytes(), bits).is_err(), || format!("weak nonce {weak} passes {bits} bits"))?;
    ensure(check(&proof[..NONCE_BYTES - 1], bits).is_err(), || "a short nonce is accepted".into())?;
    ensure(check(&[], 0).is_ok(), || "a proof of work of zero bits reads bytes".into())
}

pub fn framing<H: Hasher>() -> Result<(), String> {
    let challenge = |statement: &[u8], messages: &[&[u8]]| -> Gf2_256 {
        let mut prover = ProverTranscript::<H>::new(statement);
        for message in messages {
            prover.send_bytes(message);
        }
        prover.challenge_field()
    };
    let base = challenge(b"s", &[b"ab", b"c"]);
    ensure(base == challenge(b"s", &[b"ab", b"c"]), || "challenges are not deterministic".into())?;
    ensure(base != challenge(b"t", &[b"ab", b"c"]), || "the statement is not absorbed".into())?;
    ensure(base != challenge(b"s", &[b"a", b"bc"]), || "message boundaries are not absorbed".into())?;
    ensure(base != challenge(b"s", &[b"abc"]), || "message count is not absorbed".into())?;
    let mut prover = ProverTranscript::<H>::new(b"s");
    let batch: Vec<Gf2_256> = prover.challenge_fields(3);
    let next: Gf2_256 = prover.challenge_field();
    ensure(batch[0] != batch[1] && batch[1] != batch[2] && !batch.contains(&next), || {
        "repeated challenges".into()
    })
}

pub fn malformed_transcripts() -> Result<(), String> {
    type F = Gf2_256;
    let mut prover = ProverTranscript::<Blake3>::new(b"s");
    prover.send_field(F::ONE);
    prover.prove_work(10);
    let proof = prover.finish();
    let after_field = |statement: &[u8], proof: &[u8], bits: u32| {
        let mut verifier = VerifierTranscript::<Blake3>::new(statement, proof);
        verifier.read_field::<F>().map_err(|e| e.to_string())?;
        verifier.check_work(bits).map_err(|e| e.to_string())?;
        verifier.finish().map_err(|e| e.to_string())
    };
    ensure(after_field(b"s", &proof, 10).is_ok(), || "an honest transcript fails".into())?;
    ensure(after_field(b"s", &proof[..39], 10).is_err(), || "a short nonce passes".into())?;
    ensure(after_field(b"s", &proof, 40).is_err(), || "a harder proof of work passes".into())?;
    ensure(after_field(b"other", &proof, 10).is_err(), || {
        "a proof of work of another statement passes".into()
    })?;
    let mut keccak = VerifierTranscript::<Keccak256>::new(b"s", &proof);
    keccak.read_field::<F>().map_err(|e| e.to_string())?;
    ensure(keccak.check_work(10).is_err(), || "a proof of work of another hash passes".into())?;
    let mut long = VerifierTranscript::<Blake3>::new(b"s", &proof);
    long.read_field::<F>().map_err(|e| e.to_string())?;
    ensure(long.finish().is_err(), || "unread bytes are accepted".into())?;
    let mut huge = VerifierTranscript::<Blake3>::new(b"s", &proof);
    ensure(huge.read_fields::<F>(usize::MAX).is_err(), || "an overflowing read is accepted".into())
}

pub fn uniform_positions() -> Result<(), String> {
    let mut prover = ProverTranscript::<Blake3>::new(b"s");
    let bound = 5;
    let mut counts = vec![0; bound];
    for position in prover.challenge_positions(bound, 5000) {
        counts[position] += 1;
    }
    ensure(counts.iter().all(|&count| (850..1150).contains(&count)), || {
        format!("positions are not uniform: {counts:?}")
    })?;
    ensure(prover.challenge_positions(1, 3) == [0, 0, 0], || "positions below one are not zero".into())
}

pub fn wide_oracle() -> Result<(), String> {
    let mut prng = Prng::new(0x5eed_0040);
    let (len, width, sent) = (3000, 72, 64);
    let rows: Vec<Vec<Gf2_128>> = (0..width).map(|_| uniforms(&mut prng, len)).collect();
    let positions: Vec<usize> = (0..120).map(|_| prng.below(len)).collect();
    let oracle = Oracle::commit::<Blake3>(rows.clone());
    let mut prover = ProverTranscript::<Blake3>::new(b"sim-wide");
    oracle.open(&mut prover, &positions, sent);
    let proof = prover.finish();
    let complete = |position: usize, _: &[Gf2_128]| rows[sent..].iter().map(|row| row[position]).collect();
    let read = |proof: &[u8]| {
        let mut verifier = VerifierTranscript::<Blake3>::new(b"sim-wide", proof);
        Opening::read(&mut verifier, &oracle.root(), len, &positions, sent, complete)?;
        verifier.finish()
    };
    read(&proof).map_err(|e| format!("a wide opening fails: {e}"))?;
    for _ in 0..16 {
        let mut altered = proof.clone();
        let byte = prng.below(proof.len());
        altered[byte] ^= 1 << prng.below(8);
        ensure(read(&altered).is_err(), || format!("a wide opening altered at byte {byte} is accepted"))?;
    }
    Ok(())
}

pub fn fixed_openings() -> Result<(), String> {
    let leaves = |count: usize| -> Vec<Digest> {
        (0..count).map(|i| merkle::leaf_digest::<Blake3, Gf2_128>(i, [Gf2_128::new(i as u128)])).collect()
    };
    let tree = MerkleTree::new::<Blake3>(leaves(8));
    let sizes = [
        tree.open(&[0]).len(),
        tree.open(&[0, 1]).len(),
        tree.open(&[0, 7]).len(),
        tree.open(&(0..8).collect::<Vec<_>>()).len(),
    ];
    ensure(sizes == [3, 2, 4, 0], || format!("openings of 8 leaves have {sizes:?} digests"))?;
    let ragged = MerkleTree::new::<Blake3>(leaves(5));
    ensure(ragged.open(&[4]).len() == 1 && ragged.open(&[0]).len() == 3, || "openings of 5 leaves".into())?;
    for count in 1..=20usize {
        let all = leaves(count);
        let tree = MerkleTree::new::<Keccak256>(all.clone());
        let subsets: Vec<Vec<usize>> = if count <= 10 {
            (1..1usize << count).map(|mask| (0..count).filter(|i| mask >> i & 1 == 1).collect()).collect()
        } else {
            (0..count - 1).map(|i| vec![i, count - 1]).collect()
        };
        for positions in subsets {
            let opened: Vec<(usize, Digest)> = positions.iter().map(|&p| (p, all[p])).collect();
            let opening = tree.open(&positions);
            let verified =
                verify_opening::<Keccak256>(count, &opened, &opening).map_err(|e| e.to_string())?;
            ensure(verified == (tree.root(), true), || format!("{count} leaves at {positions:?}"))?;
            ensure(opening.len() == missing_siblings(count, &positions), || {
                format!("size of {count} leaves at {positions:?}")
            })?;
        }
    }
    Ok(())
}

fn per_hash<H: Hasher>(ctx: &mut Ctx) -> Result<(), Failure> {
    ctx.check("hash_incremental_and_domain_separated", hashing::<H>)?;
    ctx.check("merkle_node_digests", node_digests::<H>)?;
    ctx.check("merkle_openings_gf2_128", merkle_trees::<H, Gf2_128>)?;
    ctx.check("merkle_openings_gf2_256", merkle_trees::<H, Gf2_256>)?;
    ctx.check("oracle_openings_gf2_128", oracles::<Gf2_128, H>)?;
    ctx.check("oracle_openings_gf2_256", oracles::<Gf2_256, H>)?;
    ctx.check("transcript_round_trip_gf2_128", transcripts::<Gf2_128, H>)?;
    ctx.check("transcript_round_trip_gf2_256", transcripts::<Gf2_256, H>)?;
    ctx.check("proof_of_work_minimal_nonce", proofs_of_work::<H>)?;
    ctx.check("transcript_framing", |_| framing::<H>())?;
    Ok(())
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    ctx.check("hash_functions_differ", hash_functions_differ)?;
    if ctx.prng.chance(50) { per_hash::<Blake3>(ctx) } else { per_hash::<Keccak256>(ctx) }
}
