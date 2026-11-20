//! A zero-knowledge IOPP for constrained interleaved Reed–Solomon codes,
//! compiled into a non-interactive argument for evaluations of a committed
//! multilinear polynomial.
//!
//! This crate implements the construction of the section "Full construction
//! of the zero-knowledge IOPP" of the paper *Zero-Knowledge IOPPs for
//! Constrained Interleaved Codes*, with Reed–Solomon codes, Merkle trees
//! without salts, and the Fiat–Shamir transformation. It is generic over the
//! field ([`RsField`]) and the hash function ([`Hasher`]).
//!
//! # Use
//!
//! ```
//! use zo0k::{Blake3, Field, Gf2_256, Parameters, Protocol, Rng};
//!
//! let parameters = Parameters::standard(10, 100, Gf2_256::SIZE)?;
//! let protocol = Protocol::<Gf2_256, Blake3>::new(parameters)?;
//! let mut rng = Rng::from_entropy();
//! let table = rng.fields(1 << 10);
//! let point = rng.fields(10);
//!
//! let (commitment, witness) = protocol.commit(table, &mut rng)?;
//! let (value, proof) = protocol.prove(&commitment, witness, &point, &mut rng)?;
//! protocol.verify(&commitment, &point, value, &proof)?;
//! # Ok::<(), zo0k::Error>(())
//! ```
//!
//! A commitment can be opened once: [`Protocol::prove`] consumes the witness,
//! because the randomness of the commitment hides one proof only.
//!
//! # Structure
//!
//! | Paper | Module |
//! |---|---|
//! | Layout and encodings, communication | [`parameters`], [`costs`] |
//! | The mask polynomial | [`mask`] |
//! | Steps 1–3: mask sum, batching, sumcheck rounds | [`sumcheck`] |
//! | Step 4: code switch | [`code_switch`] |
//! | Step 5: base case | [`base_case`] |
//! | Claims and succinct linear forms | [`claim`] |
//! | Fields, Reed–Solomon codes | [`field`], [`reed_solomon`], [`witness`] |
//! | Hash functions, Merkle commitments | [`hash`], [`merkle`], [`oracle`] |
//! | Round-by-round soundness | [`soundness`] |
//! | Fiat–Shamir transformation, proofs of work | [`transcript`] |
//! | Choice of query counts and proofs of work | [`tuning`] |

#![warn(missing_docs)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("zo0k supports 64-bit targets only");

pub mod base_case;
pub mod blake3;
pub mod claim;
pub mod code_switch;
pub mod costs;
mod entropy;
pub mod error;
pub mod field;
pub mod hash;
pub mod keccak;
pub mod mask;
pub mod merkle;
pub mod oracle;
pub mod parameters;
pub mod polynomial;
pub mod reed_solomon;
pub mod rng;
pub mod soundness;
pub mod sumcheck;
pub mod transcript;
pub mod tuning;
pub mod witness;

pub use error::{Error, Result};
pub use field::{Field, FieldSize, Gf2_128, Gf2_256};
pub use hash::{Blake3, Digest, Hasher, Keccak256};
pub use parameters::{Layout, Parameters, StageParameters, Variant};
pub use reed_solomon::RsField;
pub use rng::Rng;
pub use witness::Witness;

use claim::{ProverClaim, VerifierClaim};
use code_switch::CodeSwitch;
use std::marker::PhantomData;
use transcript::{ProverTranscript, VerifierTranscript};
use witness::Message;

/// The Rust examples of the README, which the documentation tests run.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct Readme;

/// A commitment to a table: the Merkle root of its encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment(pub Digest);

/// A proof that a committed table has a given multilinear evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof(pub Vec<u8>);

/// The protocol with given parameters over the field `F`, compiled with the
/// hash function `H`.
pub struct Protocol<F, H> {
    parameters: Parameters,
    layout: Layout,
    types: PhantomData<fn() -> (F, H)>,
}

impl<F: RsField, H: Hasher> Protocol<F, H> {
    /// The protocol for the given parameters. Fails if they are invalid, or
    /// if the soundness error over `F` is above their target.
    pub fn new(parameters: Parameters) -> Result<Self> {
        let layout = Layout::new(&parameters, Variant::ZeroKnowledge)?;
        soundness::analyze(&layout, &parameters, F::BITS)?;
        Ok(Self { parameters, layout, types: PhantomData })
    }

    /// The parameters of the protocol.
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// The shapes of the oracles of the protocol.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// What the transcript binds before the first message: the field, the
    /// hash function, the parameters, the commitment and the claim.
    fn statement(&self, commitment: &Commitment, point: &[F], value: F) -> Vec<u8> {
        let mut bytes = [F::NAME.as_bytes(), b"/", H::NAME.as_bytes(), b"/"].concat();
        bytes.extend(self.parameters.to_bytes());
        bytes.extend(commitment.0);
        let mut encoding = vec![0; F::BYTES];
        for coordinate in point.iter().chain([&value]) {
            coordinate.write_bytes(&mut encoding);
            bytes.extend(&encoding);
        }
        bytes
    }

    /// Commit to a table of `2^num_variables` field elements.
    ///
    /// The table is the list of values of a multilinear polynomial on the
    /// Boolean cube in lexicographic order: the first variable is the highest
    /// bit of an index. The commitment is the input oracle of the protocol:
    /// the encoding of the table as the `2^k` rows of the first stage, with
    /// fresh uniform randomness.
    pub fn commit(&self, table: Vec<F>, rng: &mut Rng) -> Result<(Commitment, Witness<F>)> {
        if table.len() != 1 << self.parameters.num_variables {
            return Err(Error::InvalidInput("the table must have 2^num_variables entries"));
        }
        let first = &self.layout.stages[0];
        let message = Message::new(first, table, Vec::new(), rng);
        let witness = Witness::commit::<H>(first, message, self.layout.stages.len() == 1, rng);
        Ok((Commitment(witness.oracle.root()), witness))
    }

    /// Evaluate the multilinear polynomial of a committed table at `point`,
    /// and prove the evaluation. Returns the value and the proof.
    ///
    /// `witness` is the witness that [`Protocol::commit`] returned with
    /// `commitment`.
    pub fn prove(
        &self,
        commitment: &Commitment,
        witness: Witness<F>,
        point: &[F],
        rng: &mut Rng,
    ) -> Result<(F, Proof)> {
        if point.len() != self.parameters.num_variables {
            return Err(Error::InvalidInput("the point must have num_variables coordinates"));
        }
        let stages = &self.layout.stages;
        if !witness.fits(&stages[0], stages.len() == 1) || witness.oracle.root() != commitment.0 {
            return Err(Error::InvalidInput(
                "the witness does not belong to this protocol and this commitment",
            ));
        }
        let mut claim = ProverClaim::evaluation(point, F::ZERO);
        claim.target = polynomial::inner_product(&claim.main, &witness.message.main);
        let value = claim.target;
        let transcript = ProverTranscript::new(&self.statement(commitment, point, value));
        Ok((value, self.prove_claim(transcript, claim, witness, rng)))
    }

    /// The prover of the protocol for a claim on the message of the first
    /// oracle. A proof for a false claim does not verify, and is not zero
    /// knowledge.
    fn prove_claim(
        &self,
        mut transcript: ProverTranscript<H>,
        mut claim: ProverClaim<F>,
        mut witness: Witness<F>,
        rng: &mut Rng,
    ) -> Proof {
        let parameters = &self.parameters;
        let stages = &self.layout.stages;
        for (index, stage) in stages.iter().enumerate() {
            let Witness { message, oracle, base_case_mask } = witness;
            let StageParameters { sumcheck_pow_bits, ood_pow_bits, .. } = parameters.stages[index];
            let (folded_claim, row, challenges) =
                sumcheck::prove(&mut transcript, claim, message, sumcheck_pow_bits);
            let Some(next) = stages.get(index + 1) else {
                let mask = base_case_mask.expect("the last oracle carries the mask of the base case");
                base_case::prove(&mut transcript, parameters, stage, &folded_claim, &row, &mask, &oracle);
                break;
            };
            let next_is_last = index + 2 == stages.len();
            let query_pow_bits = parameters.pow_bits;
            let switch = CodeSwitch { source: stage, next, next_is_last, ood_pow_bits, query_pow_bits };
            (claim, witness) = switch.prove(&mut transcript, folded_claim, row, &oracle, &challenges, rng);
        }
        Proof(transcript.finish())
    }

    /// Verify a proof that the multilinear polynomial of the committed table
    /// takes the value `value` at `point`.
    pub fn verify(&self, commitment: &Commitment, point: &[F], value: F, proof: &Proof) -> Result<()> {
        if point.len() != self.parameters.num_variables {
            return Err(Error::InvalidInput("the point must have num_variables coordinates"));
        }
        let parameters = &self.parameters;
        let stages = &self.layout.stages;
        // A proof holds the direct messages and at least one column per
        // stage. Checking this first bounds what the verifier reads and
        // allocates by the proof length. The rest of its work depends on the
        // parameters alone, which bound the query counts.
        let columns: usize = stages.iter().map(|stage| stage.rows()).sum();
        if proof.0.len() < F::BYTES * (self.layout.message_len() + columns) {
            return Err(Error::InvalidProof("the proof is too short"));
        }
        let statement = self.statement(commitment, point, value);
        let mut transcript = VerifierTranscript::<H>::new(&statement, &proof.0);
        let mut claim = VerifierClaim::evaluation(point, value);
        let mut root = commitment.0;
        for (index, stage) in stages.iter().enumerate() {
            let StageParameters { sumcheck_pow_bits, ood_pow_bits, .. } = parameters.stages[index];
            let (folded_claim, challenges) =
                sumcheck::verify(&mut transcript, claim, stage.folding, sumcheck_pow_bits)?;
            let Some(next) = stages.get(index + 1) else {
                base_case::verify(&mut transcript, parameters, stage, folded_claim, &root, &challenges)?;
                break;
            };
            let next_is_last = index + 2 == stages.len();
            let query_pow_bits = parameters.pow_bits;
            let switch = CodeSwitch { source: stage, next, next_is_last, ood_pow_bits, query_pow_bits };
            (claim, root) = switch.verify(&mut transcript, folded_claim, &root, &challenges)?;
        }
        transcript.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The honest prover algorithm on a false claim: the verifier's checks
    /// reject it, and not only the binding of the statement.
    fn false_claims<F: RsField, H: Hasher>() {
        let mut rng = Rng::from_seed([9; 32]);
        for (num_variables, schedule) in [(6, vec![(3, 256)]), (9, vec![(2, 1100), (3, 300)])] {
            let stages =
                schedule.iter().map(|&(folding, domain_size)| StageParameters::new(folding, domain_size, 0));
            let mut parameters = Parameters {
                num_variables,
                stages: stages.collect(),
                security_bits: 24,
                pow_bits: 0,
                base_case_pow_bits: 0,
                johnson_slack: 0.02,
            };
            tuning::optimize_queries(&mut parameters, Variant::ZeroKnowledge, F::SIZE).unwrap();
            let protocol = Protocol::<F, H>::new(parameters).unwrap();
            let point: Vec<F> = rng.fields(num_variables);
            for error in [F::ZERO, F::ONE] {
                let (commitment, witness) =
                    protocol.commit(rng.fields(1 << num_variables), &mut rng).unwrap();
                let mut claim = ProverClaim::evaluation(&point, F::ZERO);
                claim.target = polynomial::inner_product(&claim.main, &witness.message.main) + error;
                let value = claim.target;
                let transcript = ProverTranscript::new(&protocol.statement(&commitment, &point, value));
                let proof = protocol.prove_claim(transcript, claim, witness, &mut rng);
                let result = protocol.verify(&commitment, &point, value, &proof);
                assert_eq!(result.is_ok(), error == F::ZERO);
                assert!(matches!(result, Ok(()) | Err(Error::InvalidProof(_))));
            }
        }
    }

    #[test]
    fn the_honest_algorithm_cannot_prove_a_false_claim() {
        false_claims::<Gf2_256, Blake3>();
        false_claims::<Gf2_128, Blake3>();
        false_claims::<Gf2_256, Keccak256>();
        false_claims::<Gf2_128, Keccak256>();
    }
}
