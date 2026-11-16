//! Hash functions.
//!
//! The protocol is generic over a [`Hasher`], which it uses for Merkle trees,
//! for the Fiat–Shamir transformation and for the proof of work. This crate
//! provides [`Blake3`] and [`Keccak256`].

use sha3::Digest as _;

/// The output of a hash function.
pub type Digest = [u8; DIGEST_BYTES];

/// Number of bytes of a digest.
pub const DIGEST_BYTES: usize = 32;

/// What a hash is used for. A [`Hasher`] is an independent function in every
/// domain, so that hashes made for different purposes cannot be confused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// A leaf of a Merkle tree: a position and a column of an oracle.
    MerkleLeaf,
    /// An inner node of a Merkle tree: its children.
    MerkleNode,
    /// The messages of a transcript.
    Transcript,
    /// The verifier's challenges, derived from a transcript.
    Challenge,
    /// A proof-of-work nonce.
    ProofOfWork,
}

impl Domain {
    /// A key for the domain: a label, padded to 32 bytes.
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
}

/// A hash function with 32-byte digests, modelled as a random oracle.
pub trait Hasher: Clone + Send + Sync + 'static {
    /// A name of the hash function, which transcripts bind.
    const NAME: &'static str;

    /// A hasher for the given domain.
    fn new(domain: Domain) -> Self;

    /// Absorb bytes. Only the concatenation of all absorbed bytes matters.
    fn update(&mut self, bytes: &[u8]);

    /// The digest of the absorbed bytes.
    fn finalize(self) -> Digest;

    /// The digest of `bytes` in the given domain.
    fn hash(domain: Domain, bytes: &[u8]) -> Digest {
        let mut hasher = Self::new(domain);
        hasher.update(bytes);
        hasher.finalize()
    }
}

/// BLAKE3, keyed by the domain.
#[derive(Clone)]
pub struct Blake3(blake3::Hasher);

impl Hasher for Blake3 {
    const NAME: &'static str = "blake3";

    fn new(domain: Domain) -> Self {
        Self(blake3::Hasher::new_keyed(&domain.key()))
    }

    fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finalize(self) -> Digest {
        *self.0.finalize().as_bytes()
    }

    fn hash(domain: Domain, bytes: &[u8]) -> Digest {
        *blake3::keyed_hash(&domain.key(), bytes).as_bytes()
    }
}

/// Keccak-256 as used by Ethereum, with the key of the domain as a prefix.
#[derive(Clone)]
pub struct Keccak256(sha3::Keccak256);

impl Hasher for Keccak256 {
    const NAME: &'static str = "keccak256";

    fn new(domain: Domain) -> Self {
        let mut hasher = sha3::Keccak256::new();
        hasher.update(domain.key());
        Self(hasher)
    }

    fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finalize(self) -> Digest {
        self.0.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOMAINS: [Domain; 5] =
        [Domain::MerkleLeaf, Domain::MerkleNode, Domain::Transcript, Domain::Challenge, Domain::ProofOfWork];

    fn check<H: Hasher>() {
        // Splitting the input does not matter, the domain and the input do.
        let digests: Vec<Digest> = DOMAINS
            .into_iter()
            .map(|domain| {
                let mut hasher = H::new(domain);
                hasher.update(b"ab");
                let mut other = hasher.clone();
                hasher.update(b"c");
                other.update(b"d");
                let digest = hasher.finalize();
                assert_eq!(digest, H::hash(domain, b"abc"));
                assert_ne!(digest, other.finalize());
                digest
            })
            .collect();
        for (i, digest) in digests.iter().enumerate() {
            assert!(!digests[..i].contains(digest));
        }
    }

    #[test]
    fn hashers_are_incremental_and_separate_domains() {
        check::<Blake3>();
        check::<Keccak256>();
        assert_ne!(Blake3::hash(Domain::MerkleLeaf, b""), Keccak256::hash(Domain::MerkleLeaf, b""));
    }

    #[test]
    fn keccak_is_the_ethereum_hash_of_the_prefixed_input() {
        // Keccak-256 of the empty string, a well-known constant.
        let empty: [u8; 32] = sha3::Keccak256::new().finalize().into();
        assert_eq!(empty[..4], [0xc5, 0xd2, 0x46, 0x01]);
        let mut plain = sha3::Keccak256::new();
        plain.update(Domain::MerkleNode.key());
        plain.update(b"xyz");
        let expected: [u8; 32] = plain.finalize().into();
        assert_eq!(Keccak256::hash(Domain::MerkleNode, b"xyz"), expected);
    }

    #[test]
    fn domain_keys_are_distinct_labels() {
        for (i, domain) in DOMAINS.into_iter().enumerate() {
            assert!(domain.key().starts_with(b"zo0k/"));
            assert!(DOMAINS[..i].iter().all(|other| other.key() != domain.key()));
        }
    }
}
