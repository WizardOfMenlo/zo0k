//! Parameters of the protocol and the layout of its oracles.
//!
//! [`Parameters`] is what a user chooses: the schedule of stages and the
//! security target. [`Layout`] is what follows from it: the lengths of the
//! blocks of every row, in the notation of the paper's section "Full
//! construction of the zero-knowledge IOPP".

use crate::error::{Error, Result};
use crate::reed_solomon::MAX_DOMAIN_SIZE;

/// Number of mask coordinates of a row. A mask has `D·(d - 1)` coordinates
/// for a module with `D` generators and masking degree `d`; here the module
/// is the field and the degree is three.
pub const MASK_LEN: usize = 2;

/// Admissible range of [`Parameters::johnson_slack`]. The lower end keeps the
/// rounding error of the floating-point soundness analysis negligible.
pub const JOHNSON_SLACK_RANGE: std::ops::RangeInclusive<f64> = 0.0002..=0.1;

/// Largest admissible number of bits of a proof of work.
pub const MAX_POW_BITS: u32 = 40;

/// Largest admissible query count of a stage. The verifier of a code switch
/// does a number of field operations that is quadratic in the query count,
/// so this bounds its work.
pub const MAX_QUERIES: usize = 1 << 14;

/// Largest admissible [`Parameters::security_bits`].
pub const MAX_SECURITY_BITS: u32 = 256;

/// Largest admissible [`Parameters::num_variables`].
pub const MAX_NUM_VARIABLES: usize = 40;

/// Largest admissible number of field elements in all oracles together.
const MAX_ORACLE_LEN: u128 = 1 << 48;

/// The parameters of the protocol for tables of `2^num_variables` entries.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameters {
    /// Base-two logarithm of the length of the committed table.
    pub num_variables: usize,
    /// The stages, in order. Every stage but the last is a sumcheck followed
    /// by a code switch; the last is a sumcheck followed by the base case.
    pub stages: Vec<StageParameters>,
    /// The sum of the round-by-round soundness errors is at most
    /// `2^-security_bits`.
    pub security_bits: u32,
    /// Bits of proof of work before every round of queries. The soundness
    /// analysis divides the error of a round by two to the number of bits of
    /// the proof of work that precedes it.
    pub pow_bits: u32,
    /// Bits of proof of work before the combination challenge of the base
    /// case.
    pub base_case_pow_bits: u32,
    /// Distance of every proximity parameter from the Johnson bound.
    pub johnson_slack: f64,
}

impl Parameters {
    /// A canonical encoding, which the transcript binds.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut integers = vec![self.num_variables, self.stages.len()];
        for stage in &self.stages {
            integers.extend([stage.folding, stage.domain_size, stage.queries]);
            integers.extend([stage.sumcheck_pow_bits as usize, stage.ood_pow_bits as usize]);
        }
        integers.extend([self.security_bits, self.pow_bits, self.base_case_pow_bits].map(|n| n as usize));
        let mut bytes: Vec<u8> = integers.iter().flat_map(|&n| (n as u64).to_le_bytes()).collect();
        bytes.extend(self.johnson_slack.to_bits().to_le_bytes());
        bytes
    }

    /// Bits of proof of work before the given verifier message of the given
    /// stage. Zero means that there is none.
    pub fn pow_bits_before(&self, stage: usize, round: Round) -> u32 {
        match round {
            // The error of this round is a list size over the field size.
            Round::SumcheckBatching => 0,
            Round::Fold(_) => self.stages[stage].sumcheck_pow_bits,
            Round::OutOfDomain => self.stages[stage].ood_pow_bits,
            Round::BaseCaseCombination => self.base_case_pow_bits,
            Round::QueryAndBatch | Round::BaseCaseQuery => self.pow_bits,
        }
    }
}

/// The choices that are specific to one stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageParameters {
    /// Folding parameter `k`: the oracle of the stage has `2^k` rows.
    pub folding: usize,
    /// Length `n` of the Reed–Solomon code of the stage.
    pub domain_size: usize,
    /// Number `t` of queries to the oracle of the stage.
    pub queries: usize,
    /// Bits of proof of work before every folding challenge of the sumcheck
    /// of the stage.
    pub sumcheck_pow_bits: u32,
    /// Bits of proof of work before the out-of-domain sample of the code
    /// switch out of the stage. Zero in the last stage, which has none.
    pub ood_pow_bits: u32,
}

impl StageParameters {
    /// A stage whose only proof of work is that of its queries.
    pub fn new(folding: usize, domain_size: usize, queries: usize) -> Self {
        Self { folding, domain_size, queries, sumcheck_pow_bits: 0, ood_pow_bits: 0 }
    }
}

/// A verifier message of a stage. Each has a round-by-round soundness error,
/// and each but the first can be preceded by a proof of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    /// The challenge that batches the mask with the claim in a sumcheck.
    SumcheckBatching,
    /// A folding challenge of a sumcheck, numbered from one within its stage.
    Fold(usize),
    /// The out-of-domain sample of a code switch.
    OutOfDomain,
    /// The queries and the batching challenge of a code switch.
    QueryAndBatch,
    /// The combination challenge of the base case.
    BaseCaseCombination,
    /// The queries of the base case.
    BaseCaseQuery,
}

/// Which protocol a layout describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The protocol of the paper, which this crate implements.
    ZeroKnowledge,
    /// The same protocol with everything removed that exists only for zero
    /// knowledge: seeds, masks, auxiliary blocks, pads, the mask sum and
    /// batching challenge of every sumcheck, and the mask of the base case.
    /// It is a size and soundness model for comparison, and is not executed.
    Baseline,
}

/// The shape of the oracle of one stage. A row is the codeword of the
/// polynomial whose coefficients are, from the constant term upwards, a seed,
/// a mask, a main block and an auxiliary block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    /// Folding parameter `k`.
    pub folding: usize,
    /// Length `ℓ` of the main block.
    pub main_len: usize,
    /// Length `a` of the auxiliary block.
    pub aux_len: usize,
    /// Length `p` of the mask.
    pub mask_len: usize,
    /// Length `r` of the seed: one more than the query count, which lets the
    /// oracle be committed without salts.
    pub seed_len: usize,
    /// Length of the pad that the code switch out of this stage appends to
    /// the next auxiliary message. Zero for the last stage.
    pub pad_len: usize,
    /// Code length `n`.
    pub domain_size: usize,
    /// Query count `t`.
    pub queries: usize,
}

impl Stage {
    /// Number of rows, `2^k`.
    pub fn rows(&self) -> usize {
        1 << self.folding
    }

    /// Dimension of the code: the number of coefficients of a row.
    pub fn dimension(&self) -> usize {
        self.seed_len + self.mask_len + self.main_len + self.aux_len
    }

    /// Degree of the first mask coefficient of a row.
    pub fn mask_degree(&self) -> usize {
        self.seed_len
    }

    /// Degree of the first main coefficient of a row.
    pub fn main_degree(&self) -> usize {
        self.seed_len + self.mask_len
    }

    /// Degree of the first auxiliary coefficient of a row.
    pub fn aux_degree(&self) -> usize {
        self.seed_len + self.mask_len + self.main_len
    }
}

/// The shapes of all oracles of a protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The variant of the protocol.
    pub variant: Variant,
    /// The shape of the oracle of every stage, in order.
    pub stages: Vec<Stage>,
}

impl Layout {
    /// The layout of the given variant of the protocol.
    ///
    /// This checks that the parameters describe a protocol. It does not check
    /// its soundness error; see [`crate::soundness::analyze`].
    pub fn new(parameters: &Parameters, variant: Variant) -> Result<Self> {
        let invalid = |reason: &str| Err(Error::InvalidParameters(reason.into()));
        if parameters.num_variables > MAX_NUM_VARIABLES {
            return invalid("num_variables must be at most 40");
        }
        if parameters.stages.is_empty() {
            return invalid("there must be at least one stage");
        }
        // Each folding parameter is bounded before they are added up.
        let mut unfolded_variables = parameters.num_variables;
        for stage in &parameters.stages {
            if stage.folding == 0 {
                return invalid("every stage must fold at least once");
            }
            let Some(rest) = unfolded_variables.checked_sub(stage.folding) else {
                return invalid("the folding parameters must sum to at most num_variables");
            };
            unfolded_variables = rest;
        }
        if !JOHNSON_SLACK_RANGE.contains(&parameters.johnson_slack) {
            return invalid("johnson_slack must be between 0.0002 and 0.1");
        }
        if !(1..=MAX_SECURITY_BITS).contains(&parameters.security_bits) {
            return invalid("security_bits must be between 1 and 256");
        }
        let stage_bits = parameters.stages.iter().flat_map(|s| [s.sumcheck_pow_bits, s.ood_pow_bits]);
        if stage_bits.chain([parameters.pow_bits, parameters.base_case_pow_bits]).any(|b| b > MAX_POW_BITS) {
            return invalid("a proof of work has at most 40 bits");
        }
        if parameters.stages.last().is_some_and(|last| last.ood_pow_bits != 0) {
            return invalid("the last stage has no out-of-domain sample: its ood_pow_bits must be zero");
        }

        let zero_knowledge = variant == Variant::ZeroKnowledge;
        let mask_len = if zero_knowledge { MASK_LEN } else { 0 };
        let mut stages = Vec::with_capacity(parameters.stages.len());
        let mut remaining_variables = parameters.num_variables;
        let mut aux_len = 0;
        for (index, stage) in parameters.stages.iter().enumerate() {
            if !(1..=MAX_QUERIES).contains(&stage.queries) {
                return invalid("every query count must be between 1 and 2^14");
            }
            remaining_variables -= stage.folding;
            let seed_len = if zero_knowledge { stage.queries + 1 } else { 0 };
            // The next auxiliary message holds this stage's folded auxiliary
            // block, mask and seed, and a pad that fills its rows. The pad
            // has at least one coordinate.
            let carried = aux_len + mask_len + seed_len;
            let (next_aux_len, pad_len) = match parameters.stages.get(index + 1) {
                Some(next) if zero_knowledge => {
                    let next_aux_len = (carried + 1).div_ceil(1 << next.folding);
                    (next_aux_len, (next_aux_len << next.folding) - carried)
                }
                _ => (0, 0),
            };
            let layout = Stage {
                folding: stage.folding,
                main_len: 1 << remaining_variables,
                aux_len,
                mask_len,
                seed_len,
                pad_len,
                domain_size: stage.domain_size,
                queries: stage.queries,
            };
            if stage.domain_size > MAX_DOMAIN_SIZE || layout.dimension() >= stage.domain_size {
                return Err(Error::InvalidParameters(format!(
                    "stage {index}: the domain size must be above the code dimension {} and at most 2^32",
                    layout.dimension()
                )));
            }
            stages.push(layout);
            aux_len = next_aux_len;
        }
        // Sizes in bytes fit a machine word.
        let oracles: u128 = stages.iter().map(|s| (s.rows() as u128 + 1) * s.domain_size as u128).sum();
        if oracles > MAX_ORACLE_LEN {
            return invalid("the oracles have more than 2^48 field elements");
        }
        Ok(Self { variant, stages })
    }

    /// Number of field elements in all oracles, including the input oracle
    /// and, with zero knowledge, the mask of the base case.
    pub fn oracle_len(&self) -> usize {
        let rows: usize = self.stages.iter().map(|s| s.rows() * s.domain_size).sum();
        match self.variant {
            Variant::ZeroKnowledge => rows + self.last().domain_size,
            Variant::Baseline => rows,
        }
    }

    /// Number of field elements that the prover sends outside of oracles.
    pub fn message_len(&self) -> usize {
        let count = self.stages.len();
        let total_folding: usize = self.stages.iter().map(|s| s.folding).sum();
        let last = self.last();
        // Two coefficients per sumcheck round, one out-of-domain answer per
        // code switch, and the row that the base case discloses.
        let common = 2 * total_folding + (count - 1) + last.dimension();
        match self.variant {
            // A mask sum per stage and the masked target of the base case.
            Variant::ZeroKnowledge => common + count + 1,
            Variant::Baseline => common,
        }
    }

    /// Number of verifier messages, which is the number of round-by-round
    /// soundness errors.
    pub fn round_count(&self) -> usize {
        let count = self.stages.len();
        let total_folding: usize = self.stages.iter().map(|s| s.folding).sum();
        match self.variant {
            Variant::ZeroKnowledge => total_folding + 3 * count,
            Variant::Baseline => total_folding + 2 * count - 1,
        }
    }

    /// The verifier messages in the order of the protocol, each with the
    /// index of its stage.
    pub fn rounds(&self) -> Vec<(usize, Round)> {
        let zero_knowledge = self.variant == Variant::ZeroKnowledge;
        let mut rounds = Vec::with_capacity(self.round_count());
        for (index, stage) in self.stages.iter().enumerate() {
            if zero_knowledge {
                rounds.push((index, Round::SumcheckBatching));
            }
            rounds.extend((1..=stage.folding).map(|round| (index, Round::Fold(round))));
            if index + 1 < self.stages.len() {
                rounds.extend([(index, Round::OutOfDomain), (index, Round::QueryAndBatch)]);
            } else {
                if zero_knowledge {
                    rounds.push((index, Round::BaseCaseCombination));
                }
                rounds.push((index, Round::BaseCaseQuery));
            }
        }
        rounds
    }

    /// The last stage.
    pub fn last(&self) -> &Stage {
        self.stages.last().expect("a layout has at least one stage")
    }
}
