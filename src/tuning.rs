use crate::costs::Costs;
use crate::error::{Error, Result};
use crate::field::FieldSize;
use crate::parameters::{
    JOHNSON_SLACK_RANGE, Layout, MAX_POW_BITS, Parameters, Round, StageParameters, Variant,
};
use crate::soundness::{CodeBounds, Soundness};
use std::cmp::Reverse;

pub const STANDARD_MAX_POW_BITS: u32 = 24;

pub const STANDARD_MAX_NUM_VARIABLES: usize = 33;

#[derive(Clone, Debug)]
pub struct Analysis {
    pub parameters: Parameters,
    pub field: FieldSize,
    pub layout: Layout,
    pub soundness: Soundness,
    pub costs: Costs,
}

impl Analysis {
    pub fn new(parameters: &Parameters, variant: Variant, field: FieldSize) -> Result<Self> {
        let analysis = Self::ignoring_target(parameters, variant, field)?;
        analysis.soundness.check(parameters.security_bits)?;
        Ok(analysis)
    }

    pub fn ignoring_target(parameters: &Parameters, variant: Variant, field: FieldSize) -> Result<Self> {
        let layout = Layout::new(parameters, variant)?;
        let soundness = Soundness::new(&layout, parameters, field.bits)?;
        let costs = Costs::new(&layout, parameters, field.bytes);
        Ok(Self { parameters: parameters.clone(), field, layout, soundness, costs })
    }
}

pub fn uniform_queries(parameters: &Parameters, variant: Variant) -> Result<Vec<usize>> {
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

#[derive(PartialEq)]
enum Slot {
    Sumcheck(usize),
    OutOfDomain(usize),
    BaseCase,
}

struct SlotError {
    slot: Slot,
    error: f64,
    rounds: usize,
    bits: u32,
}

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

fn slack_grid() -> Vec<f64> {
    let mantissas = [10, 12, 15, 20, 25, 30, 40, 50, 60, 80];
    let scales = [1e6, 1e5, 1e4, 1e3, 1e2];
    let values = scales.iter().flat_map(|scale| mantissas.map(|mantissa| mantissa as f64 / scale));
    values.filter(|slack| JOHNSON_SLACK_RANGE.contains(slack)).collect()
}

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
            Err(_) => larger.iter().find_map(|&slack| attempt(slack).ok()).ok_or(error),
        },
    };
    *parameters = found?;
    Ok(())
}

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
        let Some(position) = change.iter().rposition(|&delta| delta < 2) else { break };
        change[position] += 1;
        change[position + 1..].fill(-2);
    }
    *parameters = best.parameters.clone();
    Ok(best)
}

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

fn removals(count: usize) -> Vec<Vec<isize>> {
    (0..count).map(|stage| change(count, &[(stage, -1)])).collect()
}

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
    pub fn standard(num_variables: usize, security_bits: u32, field: FieldSize) -> Result<Self> {
        if !(1..=STANDARD_MAX_NUM_VARIABLES).contains(&num_variables) {
            return Err(Error::InvalidParameters(
                "standard parameters exist for 1 to 33 variables: choose a schedule for more".into(),
            ));
        }
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
