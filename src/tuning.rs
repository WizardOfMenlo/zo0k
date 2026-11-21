//! Choice of the query counts and of the proofs of work of a schedule.
//!
//! The security target bounds the sum of all round-by-round errors, so the
//! stages can trade queries with one another. [`optimize_queries`] searches
//! for the allocation with the smallest expected argument size.
//!
//! The errors of the other rounds do not depend on the query counts but on
//! the size of the field. Over a field that is too small for the security
//! target, [`assign_pow_bits`] gives those rounds proofs of work, and
//! [`assign_pow_bits_and_slack`] also enlarges the Johnson slack, which
//! lowers their errors and raises the query counts.

use crate::costs::Costs;
use crate::error::{Error, Result};
use crate::field::FieldSize;
use crate::parameters::{
    JOHNSON_SLACK_RANGE, Layout, MAX_POW_BITS, Parameters, Round, StageParameters, Variant,
};
use crate::soundness::{CodeBounds, Soundness};
use std::cmp::Reverse;

/// Largest number of bits that [`Parameters::standard`] gives a proof of
/// work.
pub const STANDARD_MAX_POW_BITS: u32 = 24;

/// Largest number of variables of [`Parameters::standard`]. Its first code
/// has length `2^(n - 1)` for `n` variables, and a code has length at most
/// `2^32`.
pub const STANDARD_MAX_NUM_VARIABLES: usize = 33;

/// Everything that follows from the parameters of a variant of the protocol.
#[derive(Clone, Debug)]
pub struct Analysis {
    /// The analyzed parameters.
    pub parameters: Parameters,
    /// The size of the field.
    pub field: FieldSize,
    /// The shapes of the oracles.
    pub layout: Layout,
    /// The round-by-round soundness errors.
    pub soundness: Soundness,
    /// The proof sizes.
    pub costs: Costs,
}

impl Analysis {
    /// Analyze the given variant over a field of the given size. Fails if the
    /// parameters are invalid or the soundness error is above their target.
    pub fn new(parameters: &Parameters, variant: Variant, field: FieldSize) -> Result<Self> {
        let analysis = Self::ignoring_target(parameters, variant, field)?;
        analysis.soundness.check(parameters.security_bits)?;
        Ok(analysis)
    }

    /// As [`Analysis::new`], but the soundness error may be above the target
    /// of the parameters.
    pub fn ignoring_target(parameters: &Parameters, variant: Variant, field: FieldSize) -> Result<Self> {
        let layout = Layout::new(parameters, variant)?;
        let soundness = Soundness::new(&layout, parameters, field.bits)?;
        let costs = Costs::new(&layout, parameters, field.bytes);
        Ok(Self { parameters: parameters.clone(), field, layout, soundness, costs })
    }
}

/// The smallest query counts for which the query error of every stage, after
/// the proof of work, is at most the security target divided by
/// `8·2^⌊log2(2s - 1)⌋`, for `s` stages. This leaves room for the algebraic
/// errors, and is the starting point of [`optimize_queries`].
///
/// The query counts of `parameters` are ignored.
pub fn uniform_queries(parameters: &Parameters, variant: Variant) -> Result<Vec<usize>> {
    // Check the parameters before computing with them.
    let mut trial = parameters.clone();
    for stage in &mut trial.stages {
        stage.queries = 1;
    }
    Layout::new(&trial, variant)?;
    let count = parameters.stages.len();
    let budget_bits = (parameters.security_bits.saturating_sub(parameters.pow_bits)
        + (2 * count as u32 - 1).ilog2()
        + 3) as f64;
    for index in 0..count {
        // With zero knowledge the dimension of the code grows with the query
        // count, which lowers the proximity parameter: iterate to the least
        // fixed point.
        let mut converged = false;
        for _ in 0..150 {
            let stage = Layout::new(&trial, variant)?.stages[index];
            let bounds = CodeBounds::new(stage.dimension(), stage.domain_size, parameters.johnson_slack)?;
            let needed = (budget_bits / -(1.0 - bounds.delta).log2()).ceil();
            if needed.is_nan() || needed >= stage.domain_size as f64 {
                return Err(Error::InvalidParameters(format!(
                    "stage {index}: the domain is too small for the query error"
                )));
            }
            if needed <= stage.queries as f64 {
                converged = true;
                break;
            }
            trial.stages[index].queries = needed as usize;
        }
        if !converged {
            return Err(Error::InvalidParameters(format!(
                "stage {index}: the query count does not converge"
            )));
        }
    }
    Ok(trial.stages.iter().map(|stage| stage.queries).collect())
}

/// The analysis of `parameters` with the given query counts.
fn with_queries(
    parameters: &Parameters,
    queries: &[usize],
    variant: Variant,
    field: FieldSize,
) -> Result<Analysis> {
    let mut trial = parameters.clone();
    for (stage, &queries) in trial.stages.iter_mut().zip(queries) {
        stage.queries = queries;
    }
    Analysis::new(&trial, variant, field)
}

/// The proofs of work that [`assign_pow_bits`] sets: one parameter serves all
/// rounds of a slot.
#[derive(PartialEq)]
enum Slot {
    /// The folding challenges of the sumcheck of a stage.
    Sumcheck(usize),
    /// The out-of-domain sample of the code switch out of a stage.
    OutOfDomain(usize),
    /// The combination challenge of the base case.
    BaseCase,
}

struct SlotError {
    slot: Slot,
    /// Sum of the errors of the rounds of the slot, without proof of work.
    error: f64,
    rounds: usize,
    bits: u32,
}

/// Set the proofs of work before the sumcheck challenges, the out-of-domain
/// samples and the combination challenge of the base case to few bits, at
/// most `max_bits` each, for which the errors of all rounds other than the
/// queries sum to at most half of the security target. The other half is
/// left to the queries. Fails if `max_bits` do not suffice.
///
/// The errors are those for the query counts of [`uniform_queries`], from
/// which [`optimize_queries`] starts. The query counts of `parameters` are
/// ignored and kept, and its proofs of work for these rounds are replaced.
/// Over a field that is large enough for the security target, all of them
/// become zero.
pub fn assign_pow_bits(
    parameters: &mut Parameters,
    variant: Variant,
    field: FieldSize,
    max_bits: u32,
) -> Result<()> {
    let max_bits = max_bits.min(MAX_POW_BITS);
    let mut trial = parameters.clone();
    trial.base_case_pow_bits = 0;
    for stage in &mut trial.stages {
        (stage.sumcheck_pow_bits, stage.ood_pow_bits) = (0, 0);
    }
    for (stage, queries) in trial.stages.iter_mut().zip(uniform_queries(parameters, variant)?) {
        stage.queries = queries;
    }
    let soundness = Soundness::new(&Layout::new(&trial, variant)?, &trial, field.bits)?;

    // The errors that no proof of work of a slot lowers, and those of the
    // slots. The rounds of a slot are consecutive.
    let mut fixed = 0.0;
    let mut slots: Vec<SlotError> = Vec::new();
    for round in &soundness.rounds {
        let slot = match round.round {
            Round::Fold(_) => Slot::Sumcheck(round.stage),
            Round::OutOfDomain => Slot::OutOfDomain(round.stage),
            Round::BaseCaseCombination => Slot::BaseCase,
            Round::SumcheckBatching | Round::QueryAndBatch | Round::BaseCaseQuery => {
                fixed += round.algebraic;
                continue;
            }
        };
        match slots.last_mut().filter(|last| last.slot == slot) {
            Some(last) => (last.error, last.rounds) = (last.error + round.algebraic, last.rounds + 1),
            None => slots.push(SlotError { slot, error: round.algebraic, rounds: 1, bits: 0 }),
        }
    }
    let budget = 0.5 * 2f64.powi(-(parameters.security_bits as i32));
    let total = |slots: &[SlotError]| {
        fixed + slots.iter().map(|slot| slot.error * 2f64.powi(-(slot.bits as i32))).sum::<f64>()
    };

    // The expected work of a slot is `rounds·2^bits` hashes. Add one bit at a
    // time where it removes the most error per hash that it adds.
    while total(&slots) > budget {
        let gain = |slot: &SlotError| slot.error * 4f64.powi(-(slot.bits as i32)) / slot.rounds as f64;
        let open = slots.iter_mut().filter(|slot| slot.bits < max_bits);
        let Some(next) = open.max_by(|a, b| gain(a).total_cmp(&gain(b))) else {
            return Err(Error::InvalidParameters(format!(
                "the errors of the sumcheck, out-of-domain and base-case rounds need more than \
                 {max_bits} bits of proof of work"
            )));
        };
        next.bits += 1;
    }
    // Remove the bits that later ones made unnecessary, the most expensive
    // first.
    let mut order: Vec<usize> = (0..slots.len()).collect();
    order.sort_by_key(|&index| Reverse((slots[index].rounds as u64) << slots[index].bits));
    for index in order {
        while slots[index].bits > 0 {
            slots[index].bits -= 1;
            if total(&slots) > budget {
                slots[index].bits += 1;
                break;
            }
        }
    }

    parameters.base_case_pow_bits = 0;
    for stage in &mut parameters.stages {
        (stage.sumcheck_pow_bits, stage.ood_pow_bits) = (0, 0);
    }
    for SlotError { slot, bits, .. } in slots {
        match slot {
            Slot::Sumcheck(stage) => parameters.stages[stage].sumcheck_pow_bits = bits,
            Slot::OutOfDomain(stage) => parameters.stages[stage].ood_pow_bits = bits,
            Slot::BaseCase => parameters.base_case_pow_bits = bits,
        }
    }
    Ok(())
}

/// The Johnson slacks that [`assign_pow_bits_and_slack`] moves through, in
/// increasing order: the admissible ones among 1, 1.2, 1.5, 2, 2.5, 3, 4, 5,
/// 6 and 8 times a power of ten.
fn slack_grid() -> Vec<f64> {
    let mantissas = [10, 12, 15, 20, 25, 30, 40, 50, 60, 80];
    let scales = [1e6, 1e5, 1e4, 1e3, 1e2];
    let values = scales.iter().flat_map(|scale| mantissas.map(|mantissa| mantissa as f64 / scale));
    values.filter(|slack| JOHNSON_SLACK_RANGE.contains(slack)).collect()
}

/// [`assign_pow_bits`] with the smallest Johnson slack for which it
/// succeeds, among the slack of `parameters` and the larger values of a
/// fixed grid. A larger slack lowers the errors that the proofs of work have
/// to cover, and raises the query counts.
///
/// Fails, and leaves `parameters` unchanged, if no slack succeeds.
pub fn assign_pow_bits_and_slack(
    parameters: &mut Parameters,
    variant: Variant,
    field: FieldSize,
    max_bits: u32,
) -> Result<()> {
    let attempt = |slack: f64| {
        let mut trial = parameters.clone();
        trial.johnson_slack = slack;
        assign_pow_bits(&mut trial, variant, field, max_bits).map(|()| trial)
    };
    let first = attempt(parameters.johnson_slack);
    let larger: Vec<f64> =
        slack_grid().into_iter().filter(|&slack| slack > parameters.johnson_slack).collect();
    let found = match (first, larger.last()) {
        (Ok(found), _) => Ok(found),
        (Err(error), None) => Err(error),
        (Err(error), Some(&largest)) => match attempt(largest) {
            // The errors decrease as the slack grows: bisect for the first
            // slack that succeeds.
            Ok(mut found) => {
                let (mut failing, mut succeeding) = (0, larger.len() - 1);
                while failing < succeeding {
                    let middle = (failing + succeeding) / 2;
                    match attempt(larger[middle]) {
                        Ok(trial) => (found, succeeding) = (trial, middle),
                        Err(_) => failing = middle + 1,
                    }
                }
                Ok(found)
            }
            // A slack this large can leave a code of high rate without a
            // proximity parameter. Try all slacks in order.
            Err(_) => larger.iter().find_map(|&slack| attempt(slack).ok()).ok_or(error),
        },
    };
    *parameters = found?;
    Ok(())
}

/// Replace the query counts of `parameters` by counts that meet the security
/// target with a small expected argument size, and return the analysis. The
/// proofs of work of `parameters` are kept.
///
/// The search is deterministic. It starts from [`uniform_queries`] and moves
/// to the best neighbouring allocation as long as one is smaller: first among
/// those with one query fewer in one stage, and when none is admissible,
/// among those that exchange queries between stages.
pub fn optimize_queries(parameters: &mut Parameters, variant: Variant, field: FieldSize) -> Result<Analysis> {
    let mut queries = uniform_queries(parameters, variant)?;
    let mut best = with_queries(parameters, &queries, variant, field)?;
    let neighbourhoods = [removals(queries.len()), exchanges(queries.len())];
    loop {
        let mut improved: Option<(Vec<usize>, Analysis)> = None;
        for moves in &neighbourhoods {
            for change in moves {
                let Some(trial) = apply(&queries, change) else { continue };
                let Ok(analysis) = with_queries(parameters, &trial, variant, field) else { continue };
                let current = improved.as_ref().map_or(&best, |(_, analysis)| analysis);
                if analysis.costs.expected_argument_bytes < current.costs.expected_argument_bytes {
                    improved = Some((trial, analysis));
                }
            }
            if improved.is_some() {
                break;
            }
        }
        match improved {
            Some((trial, analysis)) => (queries, best) = (trial, analysis),
            None => break,
        }
    }
    *parameters = best.parameters.clone();
    Ok(best)
}

/// Replace the query counts of `parameters` by the cheapest admissible counts
/// within two queries of each of them, among those whose IOP proof length is
/// at most `iop_limit` bytes. The window has `5^stages` allocations.
///
/// Fails if the given counts are not admissible themselves.
pub fn polish_queries(
    parameters: &mut Parameters,
    variant: Variant,
    field: FieldSize,
    iop_limit: Option<usize>,
) -> Result<Analysis> {
    let center: Vec<usize> = parameters.stages.iter().map(|stage| stage.queries).collect();
    let mut best = Analysis::new(parameters, variant, field)?;
    let key = |analysis: &Analysis| (analysis.costs.expected_argument_bytes, analysis.costs.iop_bytes);
    let mut change = vec![-2isize; center.len()];
    loop {
        if let Some(trial) = apply(&center, &change)
            && let Ok(analysis) = with_queries(parameters, &trial, variant, field)
            && iop_limit.is_none_or(|limit| analysis.costs.iop_bytes <= limit)
            && key(&analysis) < key(&best)
        {
            best = analysis;
        }
        // The next vector of changes in {-2, …, 2}^stages, last stage fastest.
        let Some(position) = change.iter().rposition(|&delta| delta < 2) else { break };
        change[position] += 1;
        change[position + 1..].fill(-2);
    }
    *parameters = best.parameters.clone();
    Ok(best)
}

/// The query counts after a change, unless a count would not be positive.
fn apply(queries: &[usize], change: &[isize]) -> Option<Vec<usize>> {
    queries
        .iter()
        .zip(change)
        .map(|(&count, &delta)| count.checked_add_signed(delta).filter(|&count| count > 0))
        .collect()
}

fn change(count: usize, entries: &[(usize, isize)]) -> Vec<isize> {
    let mut change = vec![0; count];
    for &(stage, delta) in entries {
        change[stage] = delta;
    }
    change
}

/// One query fewer in one stage.
fn removals(count: usize) -> Vec<Vec<isize>> {
    (0..count).map(|stage| change(count, &[(stage, -1)])).collect()
}

/// Moves that shift error between stages: one query fewer in a stage for one
/// or two more in another; one fewer in each of two stages for one or two
/// more in a third; two fewer in a stage for one more in another.
fn exchanges(count: usize) -> Vec<Vec<isize>> {
    let others = |excluded: &[usize]| {
        let excluded = excluded.to_vec();
        (0..count).filter(move |stage| !excluded.contains(stage))
    };
    let mut moves = Vec::new();
    for i in 0..count {
        for j in others(&[i]) {
            for added in 1..=2 {
                moves.push(change(count, &[(i, -1), (j, added)]));
            }
        }
    }
    for i in 0..count {
        for l in i + 1..count {
            for j in others(&[i, l]) {
                for added in 1..=2 {
                    moves.push(change(count, &[(i, -1), (l, -1), (j, added)]));
                }
            }
        }
    }
    for i in 0..count {
        for j in others(&[i]) {
            moves.push(change(count, &[(i, -2), (j, 1)]));
        }
    }
    moves
}

impl Parameters {
    /// Parameters for tables of `2^num_variables` entries with the given
    /// security target, for at most [`STANDARD_MAX_NUM_VARIABLES`] variables:
    /// stages that fold four times, codes of rate about one quarter, and
    /// optimized query counts.
    ///
    /// There is no proof of work before the queries. Over a field that is
    /// large enough for the security target there is none at all, and the
    /// Johnson slack is 0.02. Over a smaller field the other rounds get
    /// proofs of work of at most [`STANDARD_MAX_POW_BITS`] bits, and the
    /// slack is as large as these need; see [`assign_pow_bits_and_slack`].
    ///
    /// These are convenient defaults, not the configurations of the paper's
    /// evaluation, which trade proof length against argument size.
    pub fn standard(num_variables: usize, security_bits: u32, field: FieldSize) -> Result<Self> {
        if !(1..=STANDARD_MAX_NUM_VARIABLES).contains(&num_variables) {
            return Err(Error::InvalidParameters(
                "standard parameters exist for 1 to 33 variables: choose a schedule for more".into(),
            ));
        }
        // The last stage leaves a main block of at least two entries when
        // there are at least two variables.
        let mut foldings = Vec::new();
        let mut remaining = num_variables;
        while remaining > 6 {
            foldings.push(4);
            remaining -= 4;
        }
        foldings.push((remaining - 1).clamp(1, 4));
        let mut main_variables = num_variables;
        let stages = foldings
            .iter()
            .map(|&folding| {
                main_variables = main_variables.saturating_sub(folding);
                // Room for the seed and the auxiliary block next to the main block.
                let domain_size = (4 * ((1usize << main_variables) + 256)).next_power_of_two();
                StageParameters::new(folding, domain_size, 0)
            })
            .collect();
        let mut parameters = Self {
            num_variables,
            stages,
            security_bits,
            pow_bits: 0,
            base_case_pow_bits: 0,
            johnson_slack: 0.02,
        };
        let variant = Variant::ZeroKnowledge;
        assign_pow_bits_and_slack(&mut parameters, variant, field, STANDARD_MAX_POW_BITS)?;
        optimize_queries(&mut parameters, variant, field)?;
        Ok(parameters)
    }
}
