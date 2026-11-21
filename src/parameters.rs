use crate::error::{Error, Result};
use crate::reed_solomon::MAX_DOMAIN_SIZE;

pub const MASK_LEN: usize = 2;

pub const JOHNSON_SLACK_RANGE: std::ops::RangeInclusive<f64> = 0.0002..=0.1;

pub const MAX_POW_BITS: u32 = 40;

pub const MAX_QUERIES: usize = 1 << 14;

pub const MAX_SECURITY_BITS: u32 = 256;

pub const MAX_NUM_VARIABLES: usize = 40;

const MAX_ORACLE_LEN: u128 = 1 << 48;

#[derive(Clone, Debug, PartialEq)]
pub struct Parameters {
    pub num_variables: usize,
    pub stages: Vec<StageParameters>,
    pub security_bits: u32,
    pub pow_bits: u32,
    pub base_case_pow_bits: u32,
    pub johnson_slack: f64,
}

impl Parameters {
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

    pub fn pow_bits_before(&self, stage: usize, round: Round) -> u32 {
        match round {
            Round::SumcheckBatching => 0,
            Round::Fold(_) => self.stages[stage].sumcheck_pow_bits,
            Round::OutOfDomain => self.stages[stage].ood_pow_bits,
            Round::BaseCaseCombination => self.base_case_pow_bits,
            Round::QueryAndBatch | Round::BaseCaseQuery => self.pow_bits,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageParameters {
    pub folding: usize,
    pub domain_size: usize,
    pub queries: usize,
    pub sumcheck_pow_bits: u32,
    pub ood_pow_bits: u32,
}

impl StageParameters {
    pub fn new(folding: usize, domain_size: usize, queries: usize) -> Self {
        Self { folding, domain_size, queries, sumcheck_pow_bits: 0, ood_pow_bits: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    SumcheckBatching,
    Fold(usize),
    OutOfDomain,
    QueryAndBatch,
    BaseCaseCombination,
    BaseCaseQuery,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    ZeroKnowledge,
    Baseline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    pub folding: usize,
    pub main_len: usize,
    pub aux_len: usize,
    pub mask_len: usize,
    pub seed_len: usize,
    pub pad_len: usize,
    pub domain_size: usize,
    pub queries: usize,
}

impl Stage {
    pub fn rows(&self) -> usize {
        1 << self.folding
    }

    pub fn dimension(&self) -> usize {
        self.seed_len + self.mask_len + self.main_len + self.aux_len
    }

    pub fn mask_degree(&self) -> usize {
        self.seed_len
    }

    pub fn main_degree(&self) -> usize {
        self.seed_len + self.mask_len
    }

    pub fn aux_degree(&self) -> usize {
        self.seed_len + self.mask_len + self.main_len
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub variant: Variant,
    pub stages: Vec<Stage>,
}

impl Layout {
    pub fn new(parameters: &Parameters, variant: Variant) -> Result<Self> {
        let invalid = |reason: &str| Err(Error::InvalidParameters(reason.into()));
        if parameters.num_variables > MAX_NUM_VARIABLES {
            return invalid("num_variables must be at most 40");
        }
        if parameters.stages.is_empty() {
            return invalid("there must be at least one stage");
        }
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
        let oracles: u128 = stages.iter().map(|s| (s.rows() as u128 + 1) * s.domain_size as u128).sum();
        if oracles > MAX_ORACLE_LEN {
            return invalid("the oracles have more than 2^48 field elements");
        }
        Ok(Self { variant, stages })
    }

    pub fn oracle_len(&self) -> usize {
        let rows: usize = self.stages.iter().map(|s| s.rows() * s.domain_size).sum();
        match self.variant {
            Variant::ZeroKnowledge => rows + self.last().domain_size,
            Variant::Baseline => rows,
        }
    }

    pub fn message_len(&self) -> usize {
        let count = self.stages.len();
        let total_folding: usize = self.stages.iter().map(|s| s.folding).sum();
        let last = self.last();
        let common = 2 * total_folding + (count - 1) + last.dimension();
        match self.variant {
            Variant::ZeroKnowledge => common + count + 1,
            Variant::Baseline => common,
        }
    }

    pub fn round_count(&self) -> usize {
        let count = self.stages.len();
        let total_folding: usize = self.stages.iter().map(|s| s.folding).sum();
        match self.variant {
            Variant::ZeroKnowledge => total_folding + 3 * count,
            Variant::Baseline => total_folding + 2 * count - 1,
        }
    }

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

    pub fn last(&self) -> &Stage {
        self.stages.last().expect("a layout has at least one stage")
    }
}
