//! Round-by-round soundness errors of the protocol.
//!
//! The analysis evaluates the errors of the paper's theorems for the
//! sumcheck, the code switch and the base case, for Reed–Solomon codes in the
//! list-decoding regime. For a code of dimension `d` and length `n`, with
//! `ρ = (d - 1)/n`, it uses:
//!
//! * the proximity parameter `δ = 1 - sqrt((d + 1)/n) - η`, where `η` is the
//!   Johnson slack of the parameters. It is below the Johnson radius
//!   `1 - sqrt(ρ)` by more than `η`;
//! * the Johnson bound `(1 - ρ) / ((1 - δ)² - ρ)` on the size of a list at
//!   distance `δ`. It depends only on the minimum distance `1 - ρ` of the
//!   code, and therefore also holds for its interleavings;
//! * the mutual correlated agreement error for a line of Ben-Sasson, Carmon,
//!   Haböck, Kopparty and Saraf, "On Proximity Gaps for Reed–Solomon Codes"
//!   (ECCC TR25-169, Theorem 4.6 with one challenge power), and `m` times
//!   that error for an interleaving of `m` rows.
//!
//! A proof of work before a verifier message raises the cost of an attempt
//! at that message. The analysis therefore divides the error of a round by
//! `2^bits` for the `bits` of the proof of work that precedes it, and the
//! security target bounds the sum of these quotients. Without proofs of work
//! that sum is the bound on the soundness error of the interactive protocol.
//!
//! All errors are computed in floating point and rounded up by a relative
//! margin that exceeds the rounding error of the computation.

use crate::error::{Error, Result};
pub use crate::parameters::Round;
use crate::parameters::{Layout, Parameters, Variant};

/// Relative margin by which bounds are rounded up.
const MARGIN: f64 = 1.0 + 1e-12;

/// The masking degree `d` of the sumcheck: round polynomials have degree
/// below three.
const MASKING_DEGREE: f64 = 3.0;

/// Decoding bounds of the Reed–Solomon code of one stage.
#[derive(Clone, Debug)]
pub struct CodeBounds {
    /// Dimension of the code.
    pub dimension: usize,
    /// Rate of the code.
    pub rate: f64,
    /// Proximity parameter `δ`.
    pub delta: f64,
    /// Bound on the size of every list at distance `δ`.
    pub list_size: f64,
    /// Mutual correlated agreement error at distance `δ`, times the field
    /// size.
    pub mca_numerator: f64,
}

impl CodeBounds {
    /// The bounds of the code of dimension `dimension` and length `len`.
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
        // The theorem's bound, with its parameter m = max(ceil(sqrt(ρ) / gap), 3)
        // for the gap between δ and the Johnson radius, here as m + 1/2.
        let gap = 1.0 - reduced_rate.sqrt() - delta;
        let m = (reduced_rate.sqrt() / gap * MARGIN).ceil().max(3.0) + 0.5;
        let mca_numerator =
            (2.0 * m.powi(5) + 3.0 * m * delta * reduced_rate) / (3.0 * reduced_rate.powf(1.5)) * n
                + m / reduced_rate.sqrt();
        Ok(Self { dimension, rate, delta, list_size, mca_numerator })
    }
}

/// The soundness error of one round divided by `2^pow_bits`, split into the
/// part that decreases with the field size and the part that decreases with
/// the number of queries.
#[derive(Clone, Debug)]
pub struct RoundError {
    /// Index of the stage of the round.
    pub stage: usize,
    /// The verifier message of the round.
    pub round: Round,
    /// Bits of the proof of work that precedes the round.
    pub pow_bits: u32,
    /// The part that decreases with the field size.
    pub algebraic: f64,
    /// The part that decreases with the number of queries.
    pub query: f64,
}

/// The soundness analysis of a layout.
#[derive(Clone, Debug)]
pub struct Soundness {
    /// The bounds of the code of every stage.
    pub codes: Vec<CodeBounds>,
    /// One error per verifier message, in the order of the protocol.
    pub rounds: Vec<RoundError>,
    /// `-log2` of the sum of the algebraic errors.
    pub algebraic_bits: f64,
    /// `-log2` of the sum of the query errors.
    pub query_bits: f64,
    /// `-log2` of the sum of all errors.
    pub total_bits: f64,
}

impl Soundness {
    /// The round-by-round errors of the layout of the given parameters over
    /// a field of `2^field_bits` elements. The security target of the
    /// parameters is not enforced; see [`Soundness::check`].
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

    /// Fail if the sum of the errors is above `2^-security_bits`.
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

/// The round-by-round errors of the layout of the given parameters over a
/// field of `2^field_bits` elements. Fails if their sum is above the security
/// target of the parameters.
pub fn analyze(layout: &Layout, parameters: &Parameters, field_bits: u32) -> Result<Soundness> {
    let soundness = Soundness::new(layout, parameters, field_bits)?;
    soundness.check(parameters.security_bits)?;
    Ok(soundness)
}

/// `2^-bits`, exactly.
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
        // The numerator of the algebraic error, and the query error.
        let (numerator, query) = match round {
            // The sumcheck (theorem "Zero-knowledge sumcheck IOR").
            Round::SumcheckBatching => (code.list_size, 0.0),
            Round::Fold(round) => {
                let interleaving = (1usize << (stage.folding - round)) as f64;
                (interleaving * code.mca_numerator + code.list_size * (MASKING_DEGREE - 1.0), 0.0)
            }
            // The code switch into the next stage (theorem "Code-switching
            // IOR"), with the list size of the next code.
            Round::OutOfDomain => {
                // The out-of-domain sample is a row of powers over the folded
                // row of this stage, which has as many entries as its code
                // has dimensions. With zero knowledge it is extended by a
                // nonzero vector over the pad, which has error 1/(|F| - 1).
                let mut numerator = (stage.dimension() - 1) as f64;
                if zero_knowledge {
                    numerator = numerator.max(field_size / (field_size - 1.0));
                }
                (codes[index + 1].list_size.powi(2) / 2.0 * numerator, 0.0)
            }
            // The batching coefficients are the powers of one challenge, for
            // the claim, the out-of-domain answer and every query.
            Round::QueryAndBatch => (codes[index + 1].list_size * (1 + stage.queries) as f64, query_error()),
            // The base case (theorem "non-succinct zero-knowledge protocol").
            Round::BaseCaseCombination => (code.mca_numerator + code.list_size, 0.0),
            Round::BaseCaseQuery => (0.0, query_error()),
        };
        // An attempt at the challenges of a round costs the proof of work
        // that precedes them.
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
