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
pub mod json;
pub mod keccak;
pub mod mask;
pub mod merkle;
pub mod oracle;
pub mod parallel;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment(pub Digest);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof(pub Vec<u8>);

pub struct Protocol<F, H> {
    parameters: Parameters,
    layout: Layout,
    types: PhantomData<fn() -> (F, H)>,
}

impl<F: RsField, H: Hasher> Protocol<F, H> {
    pub fn new(parameters: Parameters) -> Result<Self> {
        let layout = Layout::new(&parameters, Variant::ZeroKnowledge)?;
        soundness::analyze(&layout, &parameters, F::BITS)?;
        Ok(Self { parameters, layout, types: PhantomData })
    }

    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    fn statement(&self, commitment: &Commitment, point: &[F], value: F) -> Vec<u8> {
        let mut bytes = [F::NAME.as_bytes(), b"/", H::NAME.as_bytes(), b"/"].concat();
        bytes.extend(self.parameters.to_bytes());
        bytes.extend(commitment.0);
        assert!(F::BYTES <= 32);
        let mut encoding = [0u8; 32];
        for coordinate in point.iter().chain([&value]) {
            coordinate.write_bytes(&mut encoding[..F::BYTES]);
            bytes.extend_from_slice(&encoding[..F::BYTES]);
        }
        assert!(bytes.len() > commitment.0.len());
        bytes
    }

    pub fn commit(&self, table: Vec<F>, rng: &mut Rng) -> Result<(Commitment, Witness<F>)> {
        if table.len() != 1 << self.parameters.num_variables {
            return Err(Error::InvalidInput("the table must have 2^num_variables entries"));
        }
        let first = &self.layout.stages[0];
        let message = Message::new(first, table, Vec::new(), rng);
        let witness = Witness::commit::<H>(first, message, self.layout.stages.len() == 1, rng);
        Ok((Commitment(witness.oracle.root()), witness))
    }

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
        let first = &stages[0];
        let (prefix, suffix) = point.split_at(first.folding);
        let eq_prefix = polynomial::eq_table(prefix);
        let eq_suffix = polynomial::eq_table(suffix);
        let row_dots = sumcheck::row_inner_products(&witness.message.main, &eq_suffix, first.rows());
        let value = polynomial::inner_product(&eq_prefix, &row_dots);
        let mut transcript = ProverTranscript::new(&self.statement(commitment, point, value));
        let Witness { message, oracle, base_case_mask } = witness;
        let StageParameters { sumcheck_pow_bits, ood_pow_bits, .. } = self.parameters.stages[0];
        let (folded_claim, row, challenges) = sumcheck::prove_evaluation(
            &mut transcript,
            eq_prefix,
            eq_suffix,
            row_dots,
            value,
            message,
            sumcheck_pow_bits,
        );
        let Some(next) = stages.get(1) else {
            let mask = base_case_mask.expect("the last oracle carries the mask of the base case");
            base_case::prove(&mut transcript, &self.parameters, first, &folded_claim, &row, &mask, &oracle);
            return Ok((value, Proof(transcript.finish())));
        };
        let next_is_last = stages.len() == 2;
        let switch = CodeSwitch {
            source: first,
            next,
            next_is_last,
            ood_pow_bits,
            query_pow_bits: self.parameters.pow_bits,
        };
        let (claim, witness) = switch.prove(&mut transcript, folded_claim, row, &oracle, &challenges, rng);
        Ok((value, self.prove_remaining(transcript, claim, witness, 1, rng)))
    }

    fn prove_remaining(
        &self,
        mut transcript: ProverTranscript<H>,
        mut claim: ProverClaim<F>,
        mut witness: Witness<F>,
        start_stage: usize,
        rng: &mut Rng,
    ) -> Proof {
        let parameters = &self.parameters;
        let stages = &self.layout.stages;
        assert!(start_stage < stages.len());
        assert_eq!(stages.len(), parameters.stages.len());
        for (index, stage) in stages.iter().enumerate().skip(start_stage) {
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

    pub fn verify(&self, commitment: &Commitment, point: &[F], value: F, proof: &Proof) -> Result<()> {
        if point.len() != self.parameters.num_variables {
            return Err(Error::InvalidInput("the point must have num_variables coordinates"));
        }
        let parameters = &self.parameters;
        let stages = &self.layout.stages;
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
