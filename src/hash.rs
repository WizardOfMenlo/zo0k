pub type Digest = [u8; DIGEST_BYTES];

pub const DIGEST_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    MerkleLeaf,
    MerkleNode,
    Transcript,
    Challenge,
    ProofOfWork,
}

impl Domain {
    const fn key(self) -> [u8; 32] {
        let label = match self {
            Self::MerkleLeaf => "zo0k/merkle-leaf",
            Self::MerkleNode => "zo0k/merkle-node",
            Self::Transcript => "zo0k/transcript",
            Self::Challenge => "zo0k/challenge",
            Self::ProofOfWork => "zo0k/proof-of-work",
        }
        .as_bytes();
        let mut key = [b'.'; 32];
        let mut i = 0;
        while i < label.len() {
            key[i] = label[i];
            i += 1;
        }
        key
    }

    const fn key_words(self) -> [u32; 8] {
        crate::blake3::words_from_key(&self.key())
    }

    const fn key_lanes(self) -> [u64; 4] {
        crate::keccak::lanes_from_32(&self.key())
    }
}

pub trait Hasher: Clone + Send + Sync + 'static {
    const NAME: &'static str;

    fn new(domain: Domain) -> Self;

    fn update(&mut self, bytes: &[u8]);

    fn finalize(self) -> Digest;

    fn hash(domain: Domain, bytes: &[u8]) -> Digest {
        let mut hasher = Self::new(domain);
        hasher.update(bytes);
        hasher.finalize()
    }

    fn hash_nodes(children: &[Digest], out: &mut [Digest]) {
        assert!(out.len() == children.len().div_ceil(2));
        for (dst, pair) in out.iter_mut().zip(children.chunks(2)) {
            *dst = Self::hash(Domain::MerkleNode, pair.as_flattened());
        }
    }

    fn hash_equal_len_batch(domain: Domain, inputs: &[u8], input_len: usize, out: &mut [Digest]) {
        assert!(inputs.len() == input_len * out.len());
        if input_len == 0 {
            let digest = Self::hash(domain, &[]);
            out.fill(digest);
            return;
        }
        for (dst, chunk) in out.iter_mut().zip(inputs.chunks_exact(input_len)) {
            *dst = Self::hash(domain, chunk);
        }
    }

    fn pow_bits(seed: &Digest, nonce: u64) -> u32 {
        let mut input = [0u8; DIGEST_BYTES + 8];
        input[..DIGEST_BYTES].copy_from_slice(seed);
        input[DIGEST_BYTES..].copy_from_slice(&nonce.to_le_bytes());
        let digest = Self::hash(Domain::ProofOfWork, &input);
        let bits = u64::from_be_bytes(digest[..8].try_into().expect("eight bytes")).leading_zeros();
        assert!(bits <= 64);
        bits
    }

    fn find_pow_nonce(seed: &Digest, bits: u32, start: u64, count: usize) -> Option<u64> {
        assert!(bits <= 64);
        assert!(count == 0 || start <= u64::MAX - (count as u64 - 1));
        for offset in 0..count {
            let nonce = start + offset as u64;
            if Self::pow_bits(seed, nonce) >= bits {
                return Some(nonce);
            }
        }
        None
    }
}

#[derive(Clone)]
pub struct Blake3(crate::blake3::Hasher);

impl Hasher for Blake3 {
    const NAME: &'static str = "blake3";

    fn new(domain: Domain) -> Self {
        Self(crate::blake3::Hasher::new_keyed_words(domain.key_words()))
    }

    fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finalize(self) -> Digest {
        self.0.finalize()
    }

    fn hash(domain: Domain, bytes: &[u8]) -> Digest {
        crate::blake3::keyed_hash_words(&domain.key_words(), bytes)
    }

    fn hash_nodes(children: &[Digest], out: &mut [Digest]) {
        assert!(out.len() == children.len().div_ceil(2));
        const KEY: [u32; 8] = Domain::MerkleNode.key_words();
        let (child_octs, child_after_oct) = children.as_chunks::<16>();
        let (out_octs, out_after_oct) = out.split_at_mut(child_octs.len() * 8);
        for (oct, dst) in child_octs.iter().zip(out_octs.as_chunks_mut::<8>().0.iter_mut()) {
            crate::blake3::keyed_hash_parent_8x(&KEY, oct, dst);
        }
        let (child_quads, child_rem) = child_after_oct.as_chunks::<8>();
        let (out_quads, out_rem) = out_after_oct.split_at_mut(child_quads.len() * 4);
        for (quad, dst) in child_quads.iter().zip(out_quads.as_chunks_mut::<4>().0.iter_mut()) {
            crate::blake3::keyed_hash_parent_4x(&KEY, quad, dst);
        }
        let (pairs, single) = child_rem.as_chunks::<2>();
        assert!(out_rem.len() == pairs.len() + single.len());
        for (pair, dst) in pairs.iter().zip(out_rem.iter_mut()) {
            let mut block = [0u8; 64];
            block[..32].copy_from_slice(&pair[0]);
            block[32..].copy_from_slice(&pair[1]);
            *dst = crate::blake3::keyed_hash_block(&KEY, &block);
        }
        if let Some(last) = single.first() {
            out_rem[pairs.len()] = crate::blake3::keyed_hash_words(&KEY, last);
        }
    }

    fn hash_equal_len_batch(domain: Domain, inputs: &[u8], input_len: usize, out: &mut [Digest]) {
        assert!(inputs.len() == input_len * out.len());
        let key = domain.key_words();
        let (out_octs, out_after_oct) = out.as_chunks_mut::<8>();
        let mut offset = 0;
        for dst in out_octs {
            let s = [
                &inputs[offset..offset + input_len],
                &inputs[offset + input_len..offset + 2 * input_len],
                &inputs[offset + 2 * input_len..offset + 3 * input_len],
                &inputs[offset + 3 * input_len..offset + 4 * input_len],
                &inputs[offset + 4 * input_len..offset + 5 * input_len],
                &inputs[offset + 5 * input_len..offset + 6 * input_len],
                &inputs[offset + 6 * input_len..offset + 7 * input_len],
                &inputs[offset + 7 * input_len..offset + 8 * input_len],
            ];
            crate::blake3::keyed_hash_8x(&key, s, dst);
            offset += 8 * input_len;
        }
        let (out_quads, out_rem) = out_after_oct.as_chunks_mut::<4>();
        for dst in out_quads {
            let s0 = &inputs[offset..offset + input_len];
            let s1 = &inputs[offset + input_len..offset + 2 * input_len];
            let s2 = &inputs[offset + 2 * input_len..offset + 3 * input_len];
            let s3 = &inputs[offset + 3 * input_len..offset + 4 * input_len];
            crate::blake3::keyed_hash_4x(&key, [s0, s1, s2, s3], dst);
            offset += 4 * input_len;
        }
        for dst in out_rem {
            *dst = crate::blake3::keyed_hash_words(&key, &inputs[offset..offset + input_len]);
            offset += input_len;
        }
        assert!(offset == inputs.len());
    }

    fn pow_bits(seed: &Digest, nonce: u64) -> u32 {
        const KEY: [u32; 8] = Domain::ProofOfWork.key_words();
        crate::blake3::pow_bits_with_key_words(&KEY, seed, nonce)
    }

    fn find_pow_nonce(seed: &Digest, bits: u32, start: u64, count: usize) -> Option<u64> {
        const KEY: [u32; 8] = Domain::ProofOfWork.key_words();
        crate::blake3::find_pow_nonce_with_key_words(&KEY, seed, bits, start, count)
    }
}

#[derive(Clone)]
pub struct Keccak256(crate::keccak::Keccak256);

impl Hasher for Keccak256 {
    const NAME: &'static str = "keccak256";

    fn new(domain: Domain) -> Self {
        Self(crate::keccak::Keccak256::new_with_prefix(&domain.key()))
    }

    fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finalize(self) -> Digest {
        self.0.finalize()
    }

    fn hash(domain: Domain, bytes: &[u8]) -> Digest {
        crate::keccak::keccak256_with_key_lanes(&domain.key_lanes(), bytes)
    }

    fn hash_nodes(children: &[Digest], out: &mut [Digest]) {
        assert!(out.len() == children.len().div_ceil(2));
        const KEY: [u64; 4] = Domain::MerkleNode.key_lanes();
        let (child_quads, child_rem) = children.as_chunks::<4>();
        let (out_pairs, out_rem) = out.split_at_mut(child_quads.len() * 2);
        for (quad, dst) in child_quads.iter().zip(out_pairs.as_chunks_mut::<2>().0.iter_mut()) {
            *dst = crate::keccak::keccak256_node_pair_2x(&KEY, quad);
        }
        let (pairs, single) = child_rem.as_chunks::<2>();
        assert!(out_rem.len() == pairs.len() + single.len());
        for (pair, dst) in pairs.iter().zip(out_rem.iter_mut()) {
            *dst = crate::keccak::keccak256_node_pair(&KEY, &pair[0], &pair[1]);
        }
        if let Some(last) = single.first() {
            out_rem[pairs.len()] = crate::keccak::keccak256_with_key_lanes(&KEY, last);
        }
    }

    fn hash_equal_len_batch(domain: Domain, inputs: &[u8], input_len: usize, out: &mut [Digest]) {
        assert!(inputs.len() == input_len * out.len());
        let key = domain.key_lanes();
        let (out_pairs, out_rem) = out.as_chunks_mut::<2>();
        let mut offset = 0;
        for dst in out_pairs {
            let s0 = &inputs[offset..offset + input_len];
            let s1 = &inputs[offset + input_len..offset + 2 * input_len];
            *dst = crate::keccak::keccak256_domain_2x(&key, s0, s1);
            offset += 2 * input_len;
        }
        for dst in out_rem {
            *dst = crate::keccak::keccak256_with_key_lanes(&key, &inputs[offset..offset + input_len]);
            offset += input_len;
        }
        assert!(offset == inputs.len());
    }

    fn pow_bits(seed: &Digest, nonce: u64) -> u32 {
        const KEY: [u64; 4] = Domain::ProofOfWork.key_lanes();
        crate::keccak::pow_bits(&KEY, seed, nonce)
    }

    fn find_pow_nonce(seed: &Digest, bits: u32, start: u64, count: usize) -> Option<u64> {
        const KEY: [u64; 4] = Domain::ProofOfWork.key_lanes();
        crate::keccak::find_pow_nonce(&KEY, seed, bits, start, count)
    }
}
