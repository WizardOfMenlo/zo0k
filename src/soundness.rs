use crate::error::{Error, Result};
pub use crate::parameters::Round;
use crate::parameters::{Layout, Parameters, Variant};

const MARGIN: f64 = 1.0 + 1e-12;

const MASKING_DEGREE: f64 = 3.0;

#[derive(Clone, Debug)]
pub struct CodeBounds {
    pub dimension: usize,
    pub rate: f64,
    pub delta: f64,
    pub list_size: f64,
    pub mca_numerator: f64,
}

impl CodeBounds {
    pub fn new(dimension: usize, len: usize, johnson_slack: f64) -> Result<Self> {
        let n = len as f64;
        let rate = dimension as f64 / n;
        let reduced_rate = (dimension as f64 - 1.0) / n;
        let delta = (1.0 - ((dimension + 1) as f64 / n).sqrt() - johnson_slack).next_down();
        if dimension < 2 {
            return Err(Error::InvalidParameters("a code must have dimension at least two".into()));
        }
        if dimension >= len || delta <= 0.0 {
            return Err(Error::InvalidParameters(
                "a code has no positive proximity parameter: enlarge its domain".into(),
            ));
        }
        let distance = 1.0 - reduced_rate;
        let list_size = (MARGIN * distance / ((1.0 - delta).powi(2) - reduced_rate)).ceil();
        let gap = 1.0 - reduced_rate.sqrt() - delta;
        let m = (reduced_rate.sqrt() / gap * MARGIN).ceil().max(3.0) + 0.5;
        let mca_numerator =
            (2.0 * m.powi(5) + 3.0 * m * delta * reduced_rate) / (3.0 * reduced_rate.powf(1.5)) * n
                + m / reduced_rate.sqrt();
        Ok(Self { dimension, rate, delta, list_size, mca_numerator })
    }
}

#[derive(Clone, Debug)]
pub struct RoundError {
    pub stage: usize,
    pub round: Round,
    pub pow_bits: u32,
    pub algebraic: f64,
    pub query: f64,
}

#[derive(Clone, Debug)]
pub struct Soundness {
    pub codes: Vec<CodeBounds>,
    pub rounds: Vec<RoundError>,
    pub algebraic_bits: f64,
    pub query_bits: f64,
    pub total_bits: f64,
}

impl Soundness {
    pub fn new(layout: &Layout, parameters: &Parameters, field_bits: u32) -> Result<Self> {
        let codes = layout
            .stages
            .iter()
            .map(|stage| CodeBounds::new(stage.dimension(), stage.domain_size, parameters.johnson_slack))
            .collect::<Result<Vec<_>>>()?;
        let rounds = round_errors(layout, &codes, parameters, field_bits);
        let algebraic: f64 = rounds.iter().map(|round| round.algebraic).sum();
        let query: f64 = rounds.iter().map(|round| round.query).sum();
        let total = (algebraic + query) * MARGIN;
        Ok(Self {
            codes,
            rounds,
            algebraic_bits: -algebraic.log2(),
            query_bits: -query.log2(),
            total_bits: -total.log2(),
        })
    }

    pub fn check(&self, security_bits: u32) -> Result<()> {
        if self.total_bits.is_finite() && self.total_bits >= security_bits as f64 {
            return Ok(());
        }
        Err(Error::InvalidParameters(format!(
            "the soundness error is 2^-{:.2}, above the target 2^-{security_bits}",
            self.total_bits
        )))
    }
}

pub fn analyze(layout: &Layout, parameters: &Parameters, field_bits: u32) -> Result<Soundness> {
    let soundness = Soundness::new(layout, parameters, field_bits)?;
    soundness.check(parameters.security_bits)?;
    Ok(soundness)
}

fn two_to_the_minus(bits: u32) -> f64 {
    2f64.powi(-i32::try_from(bits).unwrap_or(i32::MAX))
}

fn round_errors(
    layout: &Layout,
    codes: &[CodeBounds],
    parameters: &Parameters,
    field_bits: u32,
) -> Vec<RoundError> {
    let zero_knowledge = layout.variant == Variant::ZeroKnowledge;
    let field_size = 2f64.powi(i32::try_from(field_bits).unwrap_or(i32::MAX));
    let rounds = layout.rounds().into_iter().map(|(index, round)| {
        let (stage, code) = (&layout.stages[index], &codes[index]);
        let query_error = || (1.0 - code.delta).powf(stage.queries as f64);
        let (numerator, query) = match round {
            Round::SumcheckBatching => (code.list_size, 0.0),
            Round::Fold(round) => {
                let interleaving = (1usize << (stage.folding - round)) as f64;
                (interleaving * code.mca_numerator + code.list_size * (MASKING_DEGREE - 1.0), 0.0)
            }
            Round::OutOfDomain => {
                let mut numerator = (stage.dimension() - 1) as f64;
                if zero_knowledge {
                    numerator = numerator.max(field_size / (field_size - 1.0));
                }
                (codes[index + 1].list_size.powi(2) / 2.0 * numerator, 0.0)
            }
            Round::QueryAndBatch => (codes[index + 1].list_size * (1 + stage.queries) as f64, query_error()),
            Round::BaseCaseCombination => (code.mca_numerator + code.list_size, 0.0),
            Round::BaseCaseQuery => (0.0, query_error()),
        };
        let pow_bits = parameters.pow_bits_before(index, round);
        let grinding = two_to_the_minus(pow_bits);
        RoundError {
            stage: index,
            round,
            pow_bits,
            algebraic: numerator / field_size * grinding,
            query: query * grinding,
        }
    });
    rounds.collect()
}
