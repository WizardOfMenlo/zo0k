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
}

#[derive(Clone)]
pub struct Blake3(crate::blake3::Hasher);

impl Hasher for Blake3 {
    const NAME: &'static str = "blake3";

    fn new(domain: Domain) -> Self {
        Self(crate::blake3::Hasher::new_keyed(&domain.key()))
    }

    fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finalize(self) -> Digest {
        self.0.finalize()
    }

    fn hash(domain: Domain, bytes: &[u8]) -> Digest {
        crate::blake3::keyed_hash(&domain.key(), bytes)
    }
}

#[derive(Clone)]
pub struct Keccak256(crate::keccak::Keccak256);

impl Hasher for Keccak256 {
    const NAME: &'static str = "keccak256";

    fn new(domain: Domain) -> Self {
        let mut hasher = crate::keccak::Keccak256::new();
        hasher.update(&domain.key());
        Self(hasher)
    }

    fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finalize(self) -> Digest {
        self.0.finalize()
    }
}
