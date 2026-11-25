use crate::field::Field;

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
    const POW_CHUNK_NONCES: usize = 1024;

    fn new(domain: Domain) -> Self;

    fn update(&mut self, bytes: &[u8]);

    fn finalize(self) -> Digest;

    fn hash(domain: Domain, bytes: &[u8]) -> Digest {
        let mut hasher = Self::new(domain);
        hasher.update(bytes);
        hasher.finalize()
    }

    fn hash_challenge(seed: &Digest, counter: u64) -> Digest {
        let mut input = [0u8; DIGEST_BYTES + 8];
        input[..DIGEST_BYTES].copy_from_slice(seed);
        input[DIGEST_BYTES..].copy_from_slice(&counter.to_le_bytes());
        Self::hash(Domain::Challenge, &input)
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

    fn hash_rows_range<F: Field>(
        rows: &[Vec<F>],
        start: usize,
        count: usize,
        scratch: &mut Vec<u8>,
        out: &mut [Digest],
    ) {
        crate::merkle::hash_rows_scratch::<Self, F>(rows, start, count, scratch, out);
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
    const POW_CHUNK_NONCES: usize = 4096;

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

    fn hash_challenge(seed: &Digest, counter: u64) -> Digest {
        const KEY: [u32; 8] = Domain::Challenge.key_words();
        crate::blake3::hash_challenge_words(&KEY, seed, counter)
    }

    fn hash_nodes(children: &[Digest], out: &mut [Digest]) {
        assert!(out.len() == children.len().div_ceil(2));
        const KEY: [u32; 8] = Domain::MerkleNode.key_words();
        let (c8, c_rem8) = children.as_chunks::<16>();
        let (o8, o_rem8) = out.split_at_mut(c8.len() * 8);
        for (grp, dst) in c8.iter().zip(o8.as_chunks_mut::<8>().0.iter_mut()) {
            crate::blake3::keyed_hash_parent_8x(&KEY, grp, dst);
        }
        let (c4, c_rem4) = c_rem8.as_chunks::<8>();
        let (o4, o_rem4) = o_rem8.split_at_mut(c4.len() * 4);
        for (grp, dst) in c4.iter().zip(o4.as_chunks_mut::<4>().0.iter_mut()) {
            crate::blake3::keyed_hash_parent_4x(&KEY, grp, dst);
        }
        let (pairs, single) = c_rem4.as_chunks::<2>();
        assert!(o_rem4.len() == pairs.len() + single.len());
        for (pair, dst) in pairs.iter().zip(o_rem4.iter_mut()) {
            let mut block = [0u8; 64];
            block[..32].copy_from_slice(&pair[0]);
            block[32..].copy_from_slice(&pair[1]);
            *dst = crate::blake3::keyed_hash_block(&KEY, &block);
        }
        if let Some(last) = single.first() {
            o_rem4[pairs.len()] = crate::blake3::keyed_hash_words(&KEY, last);
        }
    }

    fn hash_equal_len_batch(domain: Domain, inputs: &[u8], input_len: usize, out: &mut [Digest]) {
        assert!(inputs.len() == input_len * out.len());
        let key = domain.key_words();
        let (o8, o_rem8) = out.as_chunks_mut::<8>();
        let mut offset = 0;
        for dst in o8 {
            let slices = std::array::from_fn(|i| {
                let s = offset + i * input_len;
                &inputs[s..s + input_len]
            });
            crate::blake3::keyed_hash_8x(&key, slices, dst);
            offset += 8 * input_len;
        }
        let (o4, o_rem4) = o_rem8.as_chunks_mut::<4>();
        for dst in o4 {
            let slices = std::array::from_fn(|i| {
                let s = offset + i * input_len;
                &inputs[s..s + input_len]
            });
            crate::blake3::keyed_hash_4x(&key, slices, dst);
            offset += 4 * input_len;
        }
        for dst in o_rem4 {
            *dst = crate::blake3::keyed_hash_words(&key, &inputs[offset..offset + input_len]);
            offset += input_len;
        }
        assert!(offset == inputs.len());
    }

    fn hash_rows_range<F: Field>(
        rows: &[Vec<F>],
        start: usize,
        count: usize,
        scratch: &mut Vec<u8>,
        out: &mut [Digest],
    ) {
        assert!(!rows.is_empty());
        assert!(out.len() == count);
        #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"))]
        if F::BYTES == 16 || F::BYTES == 32 {
            const KEY: [u32; 8] = Domain::MerkleLeaf.key_words();
            let (o8, rem8) = out.as_chunks_mut::<8>();
            if !o8.is_empty() {
                crate::blake3::hash_rows_8x_direct(&KEY, rows, start, o8);
            }
            if !rem8.is_empty() {
                let done = o8.len() * 8;
                crate::merkle::hash_rows_scratch::<Self, F>(rows, start + done, rem8.len(), scratch, rem8);
            }
            return;
        }
        crate::merkle::hash_rows_scratch::<Self, F>(rows, start, count, scratch, out);
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
    const POW_CHUNK_NONCES: usize = 768;

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

    fn hash_challenge(seed: &Digest, counter: u64) -> Digest {
        const KEY: [u64; 4] = Domain::Challenge.key_lanes();
        crate::keccak::hash_challenge_lanes(&KEY, seed, counter)
    }

    fn hash_nodes(children: &[Digest], out: &mut [Digest]) {
        assert!(out.len() == children.len().div_ceil(2));
        const KEY: [u64; 4] = Domain::MerkleNode.key_lanes();
        let (c2, c_rem2) = children.as_chunks::<4>();
        let (o2, o_rem2) = out.split_at_mut(c2.len() * 2);
        for (grp, dst) in c2.iter().zip(o2.as_chunks_mut::<2>().0.iter_mut()) {
            *dst = crate::keccak::keccak256_node_pair_2x(&KEY, grp);
        }
        let (pairs, single) = c_rem2.as_chunks::<2>();
        assert!(o_rem2.len() == pairs.len() + single.len());
        for (pair, dst) in pairs.iter().zip(o_rem2.iter_mut()) {
            *dst = crate::keccak::keccak256_node_pair(&KEY, &pair[0], &pair[1]);
        }
        if let Some(last) = single.first() {
            o_rem2[pairs.len()] = crate::keccak::keccak256_with_key_lanes(&KEY, last);
        }
    }

    fn hash_equal_len_batch(domain: Domain, inputs: &[u8], input_len: usize, out: &mut [Digest]) {
        assert!(inputs.len() == input_len * out.len());
        let key = domain.key_lanes();
        let (o2, o_rem2) = out.as_chunks_mut::<2>();
        let mut offset = 0;
        for dst in o2 {
            let s0 = &inputs[offset..offset + input_len];
            let s1 = &inputs[offset + input_len..offset + 2 * input_len];
            *dst = crate::keccak::keccak256_domain_2x(&key, s0, s1);
            offset += 2 * input_len;
        }
        for dst in o_rem2 {
            *dst = crate::keccak::keccak256_with_key_lanes(&key, &inputs[offset..offset + input_len]);
            offset += input_len;
        }
        assert!(offset == inputs.len());
    }

    fn hash_rows_range<F: Field>(
        rows: &[Vec<F>],
        start: usize,
        count: usize,
        scratch: &mut Vec<u8>,
        out: &mut [Digest],
    ) {
        assert!(!rows.is_empty());
        assert!(out.len() == count);
        #[cfg(all(target_arch = "aarch64", target_feature = "sha3", target_endian = "little"))]
        if F::BYTES == 16 || F::BYTES == 32 {
            const KEY: [u64; 4] = Domain::MerkleLeaf.key_lanes();
            let (o3, rem3) = out.as_chunks_mut::<3>();
            if !o3.is_empty() {
                crate::keccak::hash_rows_3x_direct(&KEY, rows, start, o3);
            }
            let d3 = o3.len() * 3;
            let (o2, rem2) = rem3.as_chunks_mut::<2>();
            if !o2.is_empty() {
                crate::keccak::hash_rows_2x_direct(&KEY, rows, start + d3, o2);
            }
            if !rem2.is_empty() {
                let done = d3 + o2.len() * 2;
                crate::merkle::hash_rows_scratch::<Self, F>(rows, start + done, rem2.len(), scratch, rem2);
            }
            return;
        }
        crate::merkle::hash_rows_scratch::<Self, F>(rows, start, count, scratch, out);
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
