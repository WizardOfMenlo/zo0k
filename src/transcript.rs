//! The Fiat–Shamir transformation.
//!
//! The prover writes its messages into the proof with a [`ProverTranscript`]
//! and the verifier reads them back with a [`VerifierTranscript`]. Both absorb
//! every message into the same hash state, from which the verifier's
//! challenges are derived, so the two parties obtain the same challenges
//! exactly when they have seen the same messages.
//!
//! The hash state is a [`Hasher`] that absorbs every message with its length.
//! A batch of challenges is a stream of digests of a seed and a counter, where
//! the seed is the digest of the state. The state then starts afresh from the
//! seed.
//!
//! A proof of work is a nonce whose digest together with such a seed has a
//! given number of leading zero bits. The nonce is a message of the prover:
//! the challenges that follow depend on it.

use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{DIGEST_BYTES, Digest, Domain, Hasher};
use crate::parallel;
use std::marker::PhantomData;

/// Number of bytes of a proof-of-work nonce.
pub const NONCE_BYTES: usize = 8;

const PROTOCOL_TAG: &[u8] = b"zo0k/v1";

/// The hash state shared by the prover and the verifier.
struct Sponge<H>(H);

impl<H: Hasher> Sponge<H> {
    fn new(statement: &[u8]) -> Self {
        let mut sponge = Self(H::new(Domain::Transcript));
        sponge.absorb(PROTOCOL_TAG);
        sponge.absorb(statement);
        sponge
    }

    fn absorb(&mut self, message: &[u8]) {
        self.0.update(&(message.len() as u64).to_le_bytes());
        self.0.update(message);
    }

    fn squeeze(&mut self) -> ChallengeStream<H> {
        let seed = self.0.clone().finalize();
        self.0 = H::new(Domain::Transcript);
        self.0.update(&seed);
        ChallengeStream {
            seed,
            counter: 0,
            block: [0; DIGEST_BYTES],
            used: DIGEST_BYTES,
            hasher: PhantomData,
        }
    }
}

/// A stream of uniform bytes: the digests of a seed followed by 0, 1, 2, ….
pub struct ChallengeStream<H> {
    seed: Digest,
    counter: u64,
    /// The current digest, and how many of its bytes are used up.
    block: Digest,
    used: usize,
    hasher: PhantomData<H>,
}

impl<H: Hasher> ChallengeStream<H> {
    /// Fill `bytes` with the next bytes of the stream.
    pub fn fill(&mut self, bytes: &mut [u8]) {
        for byte in bytes {
            if self.used == DIGEST_BYTES {
                let mut input = [0; DIGEST_BYTES + 8];
                input[..DIGEST_BYTES].copy_from_slice(&self.seed);
                input[DIGEST_BYTES..].copy_from_slice(&self.counter.to_le_bytes());
                self.block = H::hash(Domain::Challenge, &input);
                self.counter += 1;
                self.used = 0;
            }
            *byte = self.block[self.used];
            self.used += 1;
        }
    }
}

/// The verifier's random choices, available to both parties.
pub trait Challenges {
    /// The hash function of the transcript.
    type Hasher: Hasher;

    /// A stream of uniform bytes that depends on all messages so far. Each
    /// call returns an independent stream.
    fn challenge_stream(&mut self) -> ChallengeStream<Self::Hasher>;

    /// A uniform field element.
    fn challenge_field<F: Field>(&mut self) -> F {
        self.challenge_fields(1)[0]
    }

    /// Independent uniform field elements.
    fn challenge_fields<F: Field>(&mut self, count: usize) -> Vec<F> {
        let mut stream = self.challenge_stream();
        (0..count).map(|_| F::sample(|bytes| stream.fill(bytes))).collect()
    }

    /// A uniform vector of `count` field elements that is not zero.
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

    /// Independent uniform positions below `bound`, possibly repeated.
    fn challenge_positions(&mut self, bound: usize, count: usize) -> Vec<usize> {
        assert!(bound > 0);
        // Draw as many bits as the largest position has and reject draws
        // outside the range, which leaves the accepted ones uniform.
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

/// Leading zero bits of the proof-of-work hash of a nonce: the hash of a
/// seed, which the transcript supplies, and the nonce.
fn proof_of_work_bits<H: Hasher>(seed: &Digest, nonce: u64) -> u32 {
    let mut input = [0; DIGEST_BYTES + NONCE_BYTES];
    input[..DIGEST_BYTES].copy_from_slice(seed);
    input[DIGEST_BYTES..].copy_from_slice(&nonce.to_le_bytes());
    let digest = H::hash(Domain::ProofOfWork, &input);
    u64::from_be_bytes(digest[..8].try_into().unwrap()).leading_zeros()
}

/// The smallest nonce whose proof-of-work hash has at least `bits` leading
/// zero bits.
fn search_nonce<H: Hasher>(seed: &Digest, bits: u32) -> u64 {
    assert!(bits <= 64, "a proof-of-work hash has 64 bits");
    // Search one window of nonces after another, each in parallel. The result
    // does not depend on the number of threads.
    const WINDOW: usize = 1 << 18;
    (0..)
        .map(|window| window * WINDOW)
        .find_map(|start| {
            parallel::find_first(start..start + WINDOW, 1 << 10, |nonce| {
                proof_of_work_bits::<H>(seed, nonce as u64) >= bits
            })
        })
        .expect("some nonce meets the difficulty") as u64
}

/// The prover's side: messages are appended to the proof.
pub struct ProverTranscript<H> {
    sponge: Sponge<H>,
    proof: Vec<u8>,
}

impl<H: Hasher> ProverTranscript<H> {
    /// A transcript for the given public statement.
    pub fn new(statement: &[u8]) -> Self {
        Self { sponge: Sponge::new(statement), proof: Vec::new() }
    }

    /// Send a message.
    pub fn send_bytes(&mut self, message: &[u8]) {
        self.sponge.absorb(message);
        self.proof.extend_from_slice(message);
    }

    /// Send a digest.
    pub fn send_digest(&mut self, digest: &Digest) {
        self.send_bytes(digest);
    }

    /// Send a field element.
    pub fn send_field<F: Field>(&mut self, element: F) {
        self.send_fields(&[element]);
    }

    /// Send field elements as one message.
    pub fn send_fields<F: Field>(&mut self, elements: &[F]) {
        let mut bytes = vec![0; elements.len() * F::BYTES];
        for (chunk, element) in bytes.chunks_exact_mut(F::BYTES).zip(elements) {
            element.write_bytes(chunk);
        }
        self.send_bytes(&bytes);
    }

    /// Find and send a nonce whose hash with the current state has `bits`
    /// leading zero bits. Nothing is sent for zero bits.
    ///
    /// # Panics
    /// Panics if `bits` is above 64.
    pub fn prove_work(&mut self, bits: u32) {
        if bits == 0 {
            return;
        }
        let mut seed = [0; DIGEST_BYTES];
        self.sponge.squeeze().fill(&mut seed);
        self.send_bytes(&search_nonce::<H>(&seed, bits).to_le_bytes());
    }

    /// The proof: all messages in the order in which they were sent.
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

/// The verifier's side: messages are read from the proof.
pub struct VerifierTranscript<'a, H> {
    sponge: Sponge<H>,
    proof: &'a [u8],
}

impl<'a, H: Hasher> VerifierTranscript<'a, H> {
    /// A transcript for the given public statement and proof.
    pub fn new(statement: &[u8], proof: &'a [u8]) -> Self {
        Self { sponge: Sponge::new(statement), proof }
    }

    /// Read the next message of the prover, of `len` bytes.
    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let (message, rest) =
            self.proof.split_at_checked(len).ok_or(Error::InvalidProof("the proof is too short"))?;
        self.proof = rest;
        self.sponge.absorb(message);
        Ok(message)
    }

    /// Read a digest.
    pub fn read_digest(&mut self) -> Result<Digest> {
        Ok(self.read_bytes(DIGEST_BYTES)?.try_into().unwrap())
    }

    /// Read a field element. Fails if the encoding is not canonical.
    pub fn read_field<F: Field>(&mut self) -> Result<F> {
        Ok(self.read_fields(1)?[0])
    }

    /// Read a message of `count` field elements. Fails if an encoding is not
    /// canonical.
    pub fn read_fields<F: Field>(&mut self, count: usize) -> Result<Vec<F>> {
        let len = count.checked_mul(F::BYTES).ok_or(Error::InvalidProof("the proof is too short"))?;
        self.read_bytes(len)?
            .chunks_exact(F::BYTES)
            .map(|chunk| {
                F::from_bytes(chunk).ok_or(Error::InvalidProof("a field element is not encoded canonically"))
            })
            .collect()
    }

    /// Read and check the nonce of [`ProverTranscript::prove_work`].
    pub fn check_work(&mut self, bits: u32) -> Result<()> {
        if bits == 0 {
            return Ok(());
        }
        let mut seed = [0; DIGEST_BYTES];
        self.sponge.squeeze().fill(&mut seed);
        let nonce = u64::from_le_bytes(self.read_bytes(NONCE_BYTES)?.try_into().unwrap());
        if proof_of_work_bits::<H>(&seed, nonce) < bits {
            return Err(Error::InvalidProof("the proof of work is invalid"));
        }
        Ok(())
    }

    /// Check that the whole proof was read.
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
