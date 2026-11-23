use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{DIGEST_BYTES, Digest, Domain, Hasher};
use crate::parallel;
use std::marker::PhantomData;

pub const NONCE_BYTES: usize = 8;

const PROTOCOL_TAG: &[u8] = b"zo0k/v1";
const SPONGE_FLUSH_BYTES: usize = 64 * 1024 + 64;
const POW_BATCH: usize = 256;
const POW_SEQUENTIAL_BITS_MAX: u32 = 10;
const POW_SEQUENTIAL_PREFIX: usize = 1024;

struct Sponge<H> {
    hasher: Option<H>,
    buffer: Vec<u8>,
}

impl<H: Hasher> Sponge<H> {
    fn new(statement: &[u8]) -> Self {
        let mut sponge = Self { hasher: None, buffer: Vec::new() };
        sponge.absorb(PROTOCOL_TAG);
        sponge.absorb(statement);
        sponge
    }

    fn flush_into_hasher(&mut self) {
        let hasher = self.hasher.get_or_insert_with(|| H::new(Domain::Transcript));
        if !self.buffer.is_empty() {
            hasher.update(&self.buffer);
            self.buffer.clear();
        }
        assert!(self.buffer.is_empty());
    }

    fn absorb(&mut self, message: &[u8]) {
        self.buffer.extend_from_slice(&(message.len() as u64).to_le_bytes());
        if self.buffer.len() + message.len() >= SPONGE_FLUSH_BYTES {
            self.flush_into_hasher();
            if message.len() >= SPONGE_FLUSH_BYTES {
                self.hasher.as_mut().expect("hasher initialized").update(message);
                return;
            }
        }
        self.buffer.extend_from_slice(message);
    }

    fn squeeze(&mut self) -> ChallengeStream<H> {
        let seed = if let Some(mut hasher) = self.hasher.take() {
            if !self.buffer.is_empty() {
                hasher.update(&self.buffer);
                self.buffer.clear();
            }
            hasher.finalize()
        } else {
            let digest = H::hash(Domain::Transcript, &self.buffer);
            self.buffer.clear();
            digest
        };
        assert!(self.hasher.is_none() && self.buffer.is_empty());
        self.buffer.extend_from_slice(&seed);
        ChallengeStream {
            seed,
            counter: 0,
            block: [0; DIGEST_BYTES],
            used: DIGEST_BYTES,
            hasher: PhantomData,
        }
    }
}

pub struct ChallengeStream<H> {
    seed: Digest,
    counter: u64,
    block: Digest,
    used: usize,
    hasher: PhantomData<H>,
}

impl<H: Hasher> ChallengeStream<H> {
    pub fn fill(&mut self, bytes: &mut [u8]) {
        let mut filled = 0;
        while filled < bytes.len() {
            if self.used == DIGEST_BYTES {
                self.block = H::hash_challenge(&self.seed, self.counter);
                self.counter += 1;
                self.used = 0;
            }
            let take = (DIGEST_BYTES - self.used).min(bytes.len() - filled);
            bytes[filled..filled + take].copy_from_slice(&self.block[self.used..self.used + take]);
            self.used += take;
            filled += take;
        }
        assert!(filled == bytes.len());
        assert!(self.used <= DIGEST_BYTES);
    }
}

pub trait Challenges {
    type Hasher: Hasher;

    fn challenge_stream(&mut self) -> ChallengeStream<Self::Hasher>;

    fn challenge_field<F: Field>(&mut self) -> F {
        let mut stream = self.challenge_stream();
        F::sample(|bytes| stream.fill(bytes))
    }

    fn challenge_fields<F: Field>(&mut self, count: usize) -> Vec<F> {
        let mut stream = self.challenge_stream();
        (0..count).map(|_| F::sample(|bytes| stream.fill(bytes))).collect()
    }

    fn challenge_nonzero_vector<F: Field>(&mut self, count: usize) -> Vec<F> {
        assert!(count > 0);
        let mut stream = self.challenge_stream();
        loop {
            let vector: Vec<F> = (0..count).map(|_| F::sample(|bytes| stream.fill(bytes))).collect();
            if vector.iter().any(|&entry| entry != F::ZERO) {
                return vector;
            }
        }
    }

    fn challenge_positions(&mut self, bound: usize, count: usize) -> Vec<usize> {
        assert!(bound > 0);
        let mask = (bound as u64).next_power_of_two() - 1;
        let mut stream = self.challenge_stream();
        (0..count)
            .map(|_| {
                loop {
                    let mut bytes = [0; 8];
                    stream.fill(&mut bytes);
                    let draw = u64::from_le_bytes(bytes) & mask;
                    if draw < bound as u64 {
                        break draw as usize;
                    }
                }
            })
            .collect()
    }
}

fn search_nonce<H: Hasher>(seed: &Digest, bits: u32) -> u64 {
    assert!(bits <= 64, "a proof-of-work hash has 64 bits");
    if bits == 0 {
        return 0;
    }
    let mut start = 0u64;
    if bits <= POW_SEQUENTIAL_BITS_MAX {
        if let Some(nonce) = H::find_pow_nonce(seed, bits, 0, POW_SEQUENTIAL_PREFIX) {
            return nonce;
        }
        start = POW_SEQUENTIAL_PREFIX as u64;
    }
    let window_batches: usize = if bits <= 16 { 256 } else { 4096 };
    let window_nonces = window_batches * POW_BATCH;
    assert!(window_batches >= 4 && window_nonces == window_batches * POW_BATCH);
    let max_windows = u64::MAX / window_nonces as u64;
    for _ in 0..max_windows {
        let found = parallel::find_first(0..window_batches, 4, |batch| {
            let batch_start = start + (batch as u64) * (POW_BATCH as u64);
            H::find_pow_nonce(seed, bits, batch_start, POW_BATCH).is_some()
        });
        if let Some(batch) = found {
            let batch_start = start + (batch as u64) * (POW_BATCH as u64);
            let nonce = H::find_pow_nonce(seed, bits, batch_start, POW_BATCH)
                .expect("the winning batch contains a valid nonce");
            assert!(H::pow_bits(seed, nonce) >= bits);
            return nonce;
        }
        start += window_nonces as u64;
    }
    panic!("some nonce meets the difficulty")
}

pub struct ProverTranscript<H> {
    sponge: Sponge<H>,
    proof: Vec<u8>,
}

impl<H: Hasher> ProverTranscript<H> {
    pub fn new(statement: &[u8]) -> Self {
        Self { sponge: Sponge::new(statement), proof: Vec::new() }
    }

    pub fn send_bytes(&mut self, message: &[u8]) {
        self.sponge.absorb(message);
        self.proof.extend_from_slice(message);
    }

    pub fn send_digest(&mut self, digest: &Digest) {
        self.send_bytes(digest);
    }

    pub fn send_field<F: Field>(&mut self, element: F) {
        self.send_fields(&[element]);
    }

    pub fn send_fields<F: Field>(&mut self, elements: &[F]) {
        let start = self.proof.len();
        let len_bytes = elements.len().checked_mul(F::BYTES).expect("encoded fields fit in memory");
        self.proof.resize(start + len_bytes, 0);
        for (chunk, element) in self.proof[start..].chunks_exact_mut(F::BYTES).zip(elements) {
            element.write_bytes(chunk);
        }
        self.sponge.absorb(&self.proof[start..]);
        assert!(self.proof.len() == start + len_bytes);
    }

    pub fn prove_work(&mut self, bits: u32) {
        if bits == 0 {
            return;
        }
        let mut seed = [0; DIGEST_BYTES];
        self.sponge.squeeze().fill(&mut seed);
        self.send_bytes(&search_nonce::<H>(&seed, bits).to_le_bytes());
    }

    pub fn finish(self) -> Vec<u8> {
        self.proof
    }
}

impl<H: Hasher> Challenges for ProverTranscript<H> {
    type Hasher = H;

    fn challenge_stream(&mut self) -> ChallengeStream<H> {
        self.sponge.squeeze()
    }
}

pub struct VerifierTranscript<'a, H> {
    sponge: Sponge<H>,
    proof: &'a [u8],
}

impl<'a, H: Hasher> VerifierTranscript<'a, H> {
    pub fn new(statement: &[u8], proof: &'a [u8]) -> Self {
        Self { sponge: Sponge::new(statement), proof }
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let (message, rest) =
            self.proof.split_at_checked(len).ok_or(Error::InvalidProof("the proof is too short"))?;
        self.proof = rest;
        self.sponge.absorb(message);
        Ok(message)
    }

    pub fn read_digest(&mut self) -> Result<Digest> {
        Ok(self.read_bytes(DIGEST_BYTES)?.try_into().unwrap())
    }

    pub fn read_field<F: Field>(&mut self) -> Result<F> {
        let bytes = self.read_bytes(F::BYTES)?;
        F::from_bytes(bytes).ok_or(Error::InvalidProof("a field element is not encoded canonically"))
    }

    pub fn read_fields<F: Field>(&mut self, count: usize) -> Result<Vec<F>> {
        let len = count.checked_mul(F::BYTES).ok_or(Error::InvalidProof("the proof is too short"))?;
        self.read_bytes(len)?
            .chunks_exact(F::BYTES)
            .map(|chunk| {
                F::from_bytes(chunk).ok_or(Error::InvalidProof("a field element is not encoded canonically"))
            })
            .collect()
    }

    pub fn check_work(&mut self, bits: u32) -> Result<()> {
        if bits == 0 {
            return Ok(());
        }
        let mut seed = [0; DIGEST_BYTES];
        self.sponge.squeeze().fill(&mut seed);
        let nonce = u64::from_le_bytes(self.read_bytes(NONCE_BYTES)?.try_into().unwrap());
        if H::pow_bits(&seed, nonce) < bits {
            return Err(Error::InvalidProof("the proof of work is invalid"));
        }
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        if self.proof.is_empty() { Ok(()) } else { Err(Error::InvalidProof("the proof is too long")) }
    }
}

impl<H: Hasher> Challenges for VerifierTranscript<'_, H> {
    type Hasher = H;

    fn challenge_stream(&mut self) -> ChallengeStream<H> {
        self.sponge.squeeze()
    }
}
