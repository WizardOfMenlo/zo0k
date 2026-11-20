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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::StageParameters;

    fn parameters(pow_bits: u32) -> Parameters {
        Parameters {
            num_variables: 20,
            stages: [(3, 448512, 87), (3, 133120, 52), (3, 50688, 34), (3, 19968, 26)]
                .map(|(folding, domain_size, queries)| StageParameters::new(folding, domain_size, queries))
                .to_vec(),
            security_bits: 100,
            pow_bits,
            base_case_pow_bits: 0,
            johnson_slack: 0.0002,
        }
    }

    #[test]
    fn code_bounds_follow_their_definitions() {
        let (d, n, slack) = (131162usize, 448512usize, 0.0002);
        let bounds = CodeBounds::new(d, n, slack).unwrap();
        let rho = (d - 1) as f64 / n as f64;
        let delta = 1.0 - ((d + 1) as f64 / n as f64).sqrt() - slack;
        assert!((bounds.delta - delta).abs() < 1e-15 && bounds.delta < delta);
        // Johnson bound for minimum distance 1 - rho.
        let johnson = (1.0 - rho) / ((1.0 - delta).powi(2) - rho);
        assert!(bounds.list_size >= johnson && bounds.list_size < johnson + 1.0 + 1e-6);
        // The proximity parameter is below the Johnson radius of the code.
        assert!(bounds.delta < 1.0 - rho.sqrt());
        let m = (rho.sqrt() / (1.0 - rho.sqrt() - delta)).ceil().max(3.0) + 0.5;
        let mca =
            (2.0 * m.powi(5) + 3.0 * m * delta * rho) / (3.0 * rho.powf(1.5)) * n as f64 + m / rho.sqrt();
        assert!((bounds.mca_numerator / mca - 1.0).abs() < 1e-9);
        assert!(CodeBounds::new(n, n, slack).is_err());
        assert!(CodeBounds::new(n - 2, n, slack).is_err());
        assert!(CodeBounds::new(1, n, slack).is_err());
        assert!(CodeBounds::new(0, n, slack).is_err());
    }

    /// Values recorded from the analysis that produced the paper's tables,
    /// which was implemented separately.
    #[test]
    fn the_analysis_reproduces_recorded_values() {
        let close = |actual: f64, expected: f64| (actual / expected - 1.0).abs() < 1e-12;
        let parameters = parameters(24);
        let layout = Layout::new(&parameters, Variant::ZeroKnowledge).unwrap();
        let soundness = analyze(&layout, &parameters, 256).unwrap();
        let codes = [
            (131162, 0.4590224708398836, 3205.0, 2.4732947342722602e23),
            (16451, 0.6482495194362331, 5630.0, 2.066351765865866e22),
            (2094, 0.7964990369780394, 7936.0, 6.096142117124786e20),
            (291, 0.8788727597441442, 6634.0, 2.2855128317013386e18),
        ];
        for (code, (dimension, delta, list_size, mca_numerator)) in soundness.codes.iter().zip(codes) {
            assert_eq!(code.dimension, dimension);
            assert_eq!(code.list_size, list_size);
            assert!(close(code.delta, delta) && close(code.mca_numerator, mca_numerator));
        }
        // The recorded algebraic errors are those before the proof of work.
        let rounds = [
            (Round::SumcheckBatching, 2.7678920219077695e-74, 0.0),
            (Round::Fold(1), 8.543916084641106e-54, 0.0),
            (Round::Fold(2), 4.271958042320553e-54, 0.0),
            (Round::Fold(3), 2.1359790211602764e-54, 0.0),
            (Round::OutOfDomain, 1.7951991056916693e-65, 0.0),
            (Round::QueryAndBatch, 4.2787033489359916e-72, 3.646471006694588e-31),
        ];
        for (round, (kind, algebraic, query)) in soundness.rounds.iter().zip(rounds) {
            assert_eq!((round.stage, round.round), (0, kind));
            assert_eq!(round.pow_bits, if kind == Round::QueryAndBatch { 24 } else { 0 });
            assert!(close(round.algebraic * 2f64.powi(round.pow_bits as i32), algebraic));
            assert!(round.query == query || close(round.query, query));
        }
        let last = &soundness.rounds[22..];
        assert_eq!((last[0].round, last[1].round), (Round::BaseCaseCombination, Round::BaseCaseQuery));
        assert!(
            close(last[0].algebraic, 1.973807404940402e-59) && close(last[1].query, 8.700966316775354e-32)
        );
        assert!(close(soundness.total_bits, 100.00237296826924));
        assert!(close(soundness.algebraic_bits, 175.36281156799046));
        // The baseline of the same row of the table.
        let mut baseline = parameters.clone();
        for (stage, (domain_size, queries)) in
            baseline.stages.iter_mut().zip([(452608, 86), (138240, 51), (46592, 35), (17664, 26)])
        {
            (stage.domain_size, stage.queries) = (domain_size, queries);
        }
        let layout = Layout::new(&baseline, Variant::Baseline).unwrap();
        let soundness = analyze(&layout, &baseline, 256).unwrap();
        assert!(close(soundness.total_bits, 100.00981160174943));
        assert!(close(soundness.algebraic_bits, 175.36369215618043));
    }

    #[test]
    fn there_is_one_error_per_verifier_message() {
        let parameters = parameters(24);
        for variant in [Variant::ZeroKnowledge, Variant::Baseline] {
            let layout = Layout::new(&parameters, variant).unwrap();
            let soundness = analyze(&layout, &parameters, 256).unwrap();
            assert_eq!(soundness.rounds.len(), layout.round_count());
            assert!(soundness.total_bits >= 100.0);
            let sum: f64 = soundness.rounds.iter().map(|r| r.algebraic + r.query).sum();
            assert!((soundness.total_bits + sum.log2()).abs() < 1e-9);
            // Stages appear in order, and each ends with its queries.
            let stages: Vec<usize> = soundness.rounds.iter().map(|r| r.stage).collect();
            assert!(stages.is_sorted());
            for (stage, layout_stage) in layout.stages.iter().enumerate() {
                let rounds: Vec<Round> =
                    soundness.rounds.iter().filter(|r| r.stage == stage).map(|r| r.round).collect();
                let folds = rounds.iter().filter(|r| matches!(r, Round::Fold(_))).count();
                assert_eq!(folds, layout_stage.folding);
                let last = stage + 1 == layout.stages.len();
                assert_eq!(
                    *rounds.last().unwrap(),
                    if last { Round::BaseCaseQuery } else { Round::QueryAndBatch }
                );
                assert_eq!(rounds.contains(&Round::SumcheckBatching), variant == Variant::ZeroKnowledge);
            }
        }
    }

    /// The algebraic errors scale with the size of the field.
    #[test]
    fn a_smaller_field_has_larger_algebraic_errors() {
        let parameters = parameters(24);
        let layout = Layout::new(&parameters, Variant::ZeroKnowledge).unwrap();
        let large = analyze(&layout, &parameters, 256).unwrap();
        assert!(analyze(&layout, &parameters, 128).is_err());
        let mut weaker = parameters.clone();
        weaker.security_bits = 40;
        let small = analyze(&layout, &weaker, 128).unwrap();
        assert!((large.algebraic_bits - small.algebraic_bits - 128.0).abs() < 1e-6);
        assert!((large.query_bits - small.query_bits).abs() < 1e-9);
    }

    /// A proof of work divides the error of the rounds that it precedes, and
    /// of no other round.
    #[test]
    fn every_proof_of_work_is_credited_to_its_rounds() {
        let plain = parameters(24);
        let mut ground = plain.clone();
        ground.stages[0].sumcheck_pow_bits = 20;
        ground.stages[1].ood_pow_bits = 6;
        ground.base_case_pow_bits = 11;
        let layout = Layout::new(&plain, Variant::ZeroKnowledge).unwrap();
        let before = Soundness::new(&layout, &plain, 128).unwrap();
        let after = Soundness::new(&layout, &ground, 128).unwrap();
        for (before, after) in before.rounds.iter().zip(&after.rounds) {
            let added = match (before.stage, before.round) {
                (0, Round::Fold(_)) => 20,
                (1, Round::OutOfDomain) => 6,
                (3, Round::BaseCaseCombination) => 11,
                _ => 0,
            };
            assert_eq!(after.pow_bits, before.pow_bits + added);
            assert_eq!(before.algebraic, after.algebraic * 2f64.powi(added as i32));
            assert_eq!(before.query, after.query);
        }
        // The first sumcheck has the largest errors of this schedule.
        assert!(after.algebraic_bits > before.algebraic_bits + 2.0);
        // The baseline has no combination challenge to precede.
        let layout = Layout::new(&ground, Variant::Baseline).unwrap();
        let baseline = Soundness::new(&layout, &ground, 128).unwrap();
        assert!(baseline.rounds.iter().all(|round| round.pow_bits != 11));
    }

    #[test]
    fn the_target_is_enforced_and_grinding_counts_towards_it() {
        // Without the proof of work the same queries fall short.
        let weak = parameters(0);
        let layout = Layout::new(&weak, Variant::ZeroKnowledge).unwrap();
        assert!(analyze(&layout, &weak, 256).is_err());
        let mut fewer = parameters(24);
        fewer.stages[0].queries -= 3;
        let layout = Layout::new(&fewer, Variant::ZeroKnowledge).unwrap();
        assert!(analyze(&layout, &fewer, 256).is_err());
        fewer.security_bits = 90;
        assert!(analyze(&layout, &fewer, 256).is_ok());
    }
}
