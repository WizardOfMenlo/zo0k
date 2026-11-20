use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure};
use zo0k::costs::{Costs, ExpectedOpening};
use zo0k::hash::DIGEST_BYTES;
use zo0k::merkle::MerkleTree;
use zo0k::parameters::{
    JOHNSON_SLACK_RANGE, Layout, MASK_LEN, MAX_QUERIES, Parameters, Round, StageParameters, Variant,
};
use zo0k::soundness::{CodeBounds, Soundness, analyze};
use zo0k::transcript::NONCE_BYTES;
use zo0k::tuning::{Analysis, optimize_queries, polish_queries, uniform_queries};
use zo0k::{Digest, Field, FieldSize, Gf2_128, Gf2_256};

pub const VARIANTS: [Variant; 2] = [Variant::ZeroKnowledge, Variant::Baseline];

pub fn expected_rounds(parameters: &Parameters, variant: Variant) -> Vec<(usize, Round)> {
    let zero_knowledge = variant == Variant::ZeroKnowledge;
    let count = parameters.stages.len();
    let mut rounds = Vec::new();
    for (index, stage) in parameters.stages.iter().enumerate() {
        if zero_knowledge {
            rounds.push((index, Round::SumcheckBatching));
        }
        for round in 1..=stage.folding {
            rounds.push((index, Round::Fold(round)));
        }
        if index + 1 < count {
            rounds.extend([(index, Round::OutOfDomain), (index, Round::QueryAndBatch)]);
            continue;
        }
        if zero_knowledge {
            rounds.push((index, Round::BaseCaseCombination));
        }
        rounds.push((index, Round::BaseCaseQuery));
    }
    assert!(rounds.len() >= count);
    rounds
}

fn expected_pow_bits(parameters: &Parameters, stage: usize, round: Round) -> u32 {
    assert!(stage < parameters.stages.len());
    let bits = match round {
        Round::SumcheckBatching => 0,
        Round::Fold(_) => parameters.stages[stage].sumcheck_pow_bits,
        Round::OutOfDomain => parameters.stages[stage].ood_pow_bits,
        Round::BaseCaseCombination => parameters.base_case_pow_bits,
        Round::QueryAndBatch | Round::BaseCaseQuery => parameters.pow_bits,
    };
    assert!(bits <= 64);
    bits
}

fn stage_shapes(parameters: &Parameters, layout: &Layout, variant: Variant) -> Result<(), String> {
    let zero_knowledge = variant == Variant::ZeroKnowledge;
    let mut remaining = parameters.num_variables;
    for (index, (stage, given)) in layout.stages.iter().zip(&parameters.stages).enumerate() {
        remaining -= given.folding;
        let fields = (stage.folding, stage.domain_size, stage.queries, stage.main_len);
        ensure(fields == (given.folding, given.domain_size, given.queries, 1 << remaining), || {
            format!("stage {index} has shape {fields:?}")
        })?;
        let expected_seed = if zero_knowledge { given.queries + 1 } else { 0 };
        let expected_mask = if zero_knowledge { MASK_LEN } else { 0 };
        ensure((stage.seed_len, stage.mask_len) == (expected_seed, expected_mask), || {
            format!("stage {index} seed or mask length")
        })?;
        ensure(stage.dimension() == stage.aux_degree() + stage.aux_len, || {
            format!("stage {index} dimension")
        })?;
        ensure(stage.dimension() < stage.domain_size, || {
            format!("stage {index} has dimension {} for a domain of {}", stage.dimension(), stage.domain_size)
        })?;
        ensure(stage.rows() == 1 << stage.folding, || format!("stage {index} rows"))?;
        if !zero_knowledge {
            ensure((stage.aux_len, stage.pad_len, stage.dimension()) == (0, 0, stage.main_len), || {
                format!("baseline stage {index} has randomness")
            })?;
        }
    }
    ensure(layout.stages[0].aux_len == 0 && layout.last().pad_len == 0, || {
        "the first aux or last pad is not empty".into()
    })?;
    for pair in layout.stages.windows(2) {
        let (stage, next) = (&pair[0], &pair[1]);
        ensure(stage.main_len == next.rows() * next.main_len, || {
            "main lengths do not follow the folding".into()
        })?;
        if zero_knowledge {
            let carried = stage.aux_len + MASK_LEN + stage.seed_len + stage.pad_len;
            ensure(carried == next.rows() * next.aux_len, || {
                format!("{carried} carried entries do not fill the next aux")
            })?;
            ensure(next.aux_len == (stage.aux_len + stage.queries + 4).div_ceil(next.rows()), || {
                "the next aux length".into()
            })?;
            ensure((1..=next.rows()).contains(&stage.pad_len), || {
                format!("pad of {} for {} rows", stage.pad_len, next.rows())
            })?;
        }
    }
    Ok(())
}

pub fn layout_invariants(parameters: &Parameters, variant: Variant) -> Result<(), String> {
    let layout = Layout::new(parameters, variant).map_err(|e| format!("{variant:?}: {e}"))?;
    ensure(layout.variant == variant && layout.stages.len() == parameters.stages.len(), || {
        "layout stage count".into()
    })?;
    stage_shapes(parameters, &layout, variant)?;
    let zero_knowledge = variant == Variant::ZeroKnowledge;
    let count = layout.stages.len();
    let total_folding: usize = layout.stages.iter().map(|s| s.folding).sum();
    let rows: usize = layout.stages.iter().map(|s| s.rows() * s.domain_size).sum();
    let last = layout.last();
    let oracle_len = rows + if zero_knowledge { last.domain_size } else { 0 };
    ensure(layout.oracle_len() == oracle_len, || {
        format!("{variant:?} oracle length {} is not {oracle_len}", layout.oracle_len())
    })?;
    let message_len =
        2 * total_folding + (count - 1) + last.dimension() + if zero_knowledge { count + 1 } else { 0 };
    ensure(layout.message_len() == message_len, || {
        format!("{variant:?} message length {} is not {message_len}", layout.message_len())
    })?;
    let round_count = total_folding + if zero_knowledge { 3 * count } else { 2 * count - 1 };
    ensure(layout.round_count() == round_count, || {
        format!("{variant:?} round count {}", layout.round_count())
    })?;
    let rounds = layout.rounds();
    ensure(rounds == expected_rounds(parameters, variant), || format!("{variant:?} rounds {rounds:?}"))?;
    for &(stage, round) in &rounds {
        let bits = parameters.pow_bits_before(stage, round);
        ensure(bits == expected_pow_bits(parameters, stage, round), || {
            format!("pow bits before {round:?} of stage {stage}")
        })?;
    }
    Ok(())
}

pub fn soundness_invariants(
    parameters: &Parameters,
    variant: Variant,
    field: FieldSize,
) -> Result<(), String> {
    let layout = Layout::new(parameters, variant).map_err(|e| e.to_string())?;
    let codes =
        layout.stages.iter().map(|s| CodeBounds::new(s.dimension(), s.domain_size, parameters.johnson_slack));
    let codes_valid = codes.collect::<zo0k::Result<Vec<_>>>().is_ok();
    let soundness = match Soundness::new(&layout, parameters, field.bits) {
        Ok(soundness) => soundness,
        Err(_) if !codes_valid => return Ok(()),
        Err(error) => return Err(format!("{variant:?}: valid codes and {error}")),
    };
    ensure(codes_valid, || format!("{variant:?}: an invalid code has a soundness analysis"))?;
    ensure(soundness.rounds.len() == layout.round_count(), || "one error per verifier message".into())?;
    ensure(soundness.codes.len() == layout.stages.len(), || "one code per stage".into())?;
    let sum: f64 = soundness.rounds.iter().map(|r| r.algebraic + r.query).sum();
    ensure((soundness.total_bits + sum.log2()).abs() < 1e-9, || {
        format!("total bits {} for a sum of {sum}", soundness.total_bits)
    })?;
    let algebraic: f64 = soundness.rounds.iter().map(|r| r.algebraic).sum();
    ensure((soundness.algebraic_bits + algebraic.log2()).abs() < 1e-9, || "algebraic bits".into())?;
    let rounds: Vec<(usize, Round)> = soundness.rounds.iter().map(|r| (r.stage, r.round)).collect();
    ensure(rounds == layout.rounds(), || "soundness rounds differ from the layout".into())?;
    for (round, code) in
        soundness.rounds.iter().zip(soundness.rounds.iter().map(|r| &soundness.codes[r.stage]))
    {
        ensure(round.pow_bits == parameters.pow_bits_before(round.stage, round.round), || {
            format!("{:?} credits {} bits", round.round, round.pow_bits)
        })?;
        ensure(round.algebraic >= 0.0 && round.query >= 0.0, || {
            format!("{:?} has a negative error", round.round)
        })?;
        ensure(code.delta > 0.0 && code.delta < 1.0 && code.list_size >= 1.0, || {
            "a code bound is out of range".into()
        })?;
    }
    let target_met =
        soundness.total_bits.is_finite() && soundness.total_bits >= parameters.security_bits as f64;
    ensure(soundness.check(parameters.security_bits).is_ok() == target_met, || "the target check".into())?;
    ensure(analyze(&layout, parameters, field.bits).is_ok() == target_met, || {
        "analyze disagrees with check".into()
    })?;
    grinding_is_credited(parameters, &layout, &soundness, field)
}

fn grinding_is_credited(
    parameters: &Parameters,
    layout: &Layout,
    soundness: &Soundness,
    field: FieldSize,
) -> Result<(), String> {
    let mut plain = parameters.clone();
    (plain.pow_bits, plain.base_case_pow_bits) = (0, 0);
    for stage in &mut plain.stages {
        (stage.sumcheck_pow_bits, stage.ood_pow_bits) = (0, 0);
    }
    let before = Soundness::new(layout, &plain, field.bits).map_err(|e| e.to_string())?;
    for (before, after) in before.rounds.iter().zip(&soundness.rounds) {
        let scale = 2f64.powi(after.pow_bits as i32);
        ensure(before.algebraic == after.algebraic * scale && before.query == after.query * scale, || {
            format!("{:?} of stage {} does not credit {} bits", after.round, after.stage, after.pow_bits)
        })?;
    }
    if field.bits == 256 {
        let small = Soundness::new(layout, parameters, 128).map_err(|e| e.to_string())?;
        ensure((soundness.algebraic_bits - small.algebraic_bits - 128.0).abs() < 1e-6, || {
            "the field size does not scale the algebraic error".into()
        })?;
        ensure((soundness.query_bits - small.query_bits).abs() < 1e-9, || {
            "the field size changes the query error".into()
        })?;
    }
    Ok(())
}

pub fn costs_invariants(parameters: &Parameters, variant: Variant, field: FieldSize) -> Result<(), String> {
    let layout = Layout::new(parameters, variant).map_err(|e| e.to_string())?;
    let costs = Costs::new(&layout, parameters, field.bytes);
    let nonces = layout.rounds().iter().filter(|&&(s, r)| expected_pow_bits(parameters, s, r) > 0).count();
    ensure(costs.pow_nonce_bytes == NONCE_BYTES * nonces, || {
        format!("{nonces} proofs of work cost {} bytes", costs.pow_nonce_bytes)
    })?;
    ensure(costs.oracle_bytes == field.bytes * layout.oracle_len(), || "oracle bytes".into())?;
    ensure(costs.message_bytes == field.bytes * layout.message_len(), || "message bytes".into())?;
    ensure(costs.iop_bytes == costs.oracle_bytes + costs.message_bytes + costs.pow_nonce_bytes, || {
        "iop bytes".into()
    })?;
    ensure(costs.root_bytes == DIGEST_BYTES * layout.stages.len(), || "root bytes".into())?;
    let openings: f64 = layout
        .stages
        .iter()
        .map(|stage| {
            let opening = ExpectedOpening::new(stage.domain_size, stage.queries);
            (field.bytes * stage.rows()) as f64 * opening.leaves + DIGEST_BYTES as f64 * opening.digests
        })
        .sum();
    ensure((costs.expected_opening_bytes - openings).abs() <= 1e-9 * openings.max(1.0), || {
        "expected opening bytes".into()
    })?;
    let fixed = (costs.root_bytes + costs.message_bytes + costs.pow_nonce_bytes) as f64;
    ensure((costs.expected_argument_bytes - fixed - openings).abs() <= 1e-9 * fixed, || {
        "expected argument bytes".into()
    })?;
    for stage in &layout.stages {
        let opening = ExpectedOpening::new(stage.domain_size, stage.queries);
        let leaves_max = stage.queries.min(stage.domain_size) as f64;
        ensure(opening.leaves >= 1.0 - 1e-9 && opening.leaves <= leaves_max + 1e-9, || {
            format!("expected leaves {}", opening.leaves)
        })?;
        ensure(
            opening.digests >= -1e-9
                && opening.digests <= (stage.domain_size as f64).log2().ceil() * leaves_max + 1e-9,
            || format!("expected digests {}", opening.digests),
        )?;
    }
    Ok(())
}

pub fn encodings_bind(parameters: &Parameters) -> Result<(), String> {
    let edits: [fn(&mut Parameters); 12] = [
        |p| p.num_variables += 1,
        |p| p.security_bits += 1,
        |p| p.pow_bits += 1,
        |p| p.base_case_pow_bits += 1,
        |p| p.johnson_slack = p.johnson_slack.next_up(),
        |p| p.stages[0].folding += 1,
        |p| p.stages[0].domain_size += 1,
        |p| p.stages[0].queries += 1,
        |p| p.stages[0].sumcheck_pow_bits += 1,
        |p| p.stages[0].ood_pow_bits += 1,
        |p| {
            let last = p.stages.len() - 1;
            p.stages[last].queries += 1
        },
        |p| p.stages.push(StageParameters::new(1, 4, 1)),
    ];
    let mut encodings = vec![parameters.to_bytes()];
    for edit in edits {
        let mut changed = parameters.clone();
        edit(&mut changed);
        encodings.push(changed.to_bytes());
    }
    let count = encodings.len();
    encodings.sort();
    encodings.dedup();
    let distinct = encodings.len() + usize::from(parameters.stages.len() == 1);
    ensure(distinct >= count, || {
        format!("only {} of {count} parameter encodings are distinct", encodings.len())
    })
}

pub fn exact_expected_opening(num_leaves: usize, queries: u32) -> Result<(), String> {
    assert!(num_leaves >= 1 && queries >= 1);
    let tree = MerkleTree::new::<zo0k::Blake3>(vec![Digest::default(); num_leaves]);
    let (mut leaves, mut digests) = (0, 0);
    let total = num_leaves.pow(queries);
    for code in 0..total {
        let mut positions: Vec<usize> = (0..queries).map(|i| code / num_leaves.pow(i) % num_leaves).collect();
        positions.sort_unstable();
        positions.dedup();
        leaves += positions.len();
        digests += tree.open(&positions).len();
    }
    let expected = ExpectedOpening::new(num_leaves, queries as usize);
    let close = |a: f64, b: usize| (a - b as f64 / total as f64).abs() < 1e-9;
    ensure(close(expected.leaves, leaves), || {
        format!("expected leaves of {num_leaves} leaves and {queries} queries")
    })?;
    ensure(close(expected.digests, digests), || {
        format!("expected digests of {num_leaves} leaves and {queries} queries")
    })
}

pub fn tuning_invariants(parameters: &Parameters, variant: Variant, field: FieldSize) -> Result<(), String> {
    let mut tuned = parameters.clone();
    let Ok(best) = optimize_queries(&mut tuned, variant, field) else {
        return Ok(());
    };
    ensure(best.parameters == tuned, || "optimize_queries returns other parameters".into())?;
    ensure(best.soundness.total_bits >= parameters.security_bits as f64, || {
        "optimized queries miss the target".into()
    })?;
    for stage in 0..tuned.stages.len() {
        let mut fewer = tuned.clone();
        fewer.stages[stage].queries -= 1;
        let fails = fewer.stages[stage].queries == 0 || Analysis::new(&fewer, variant, field).is_err();
        let cheaper = Analysis::new(&fewer, variant, field)
            .is_ok_and(|a| a.costs.expected_argument_bytes < best.costs.expected_argument_bytes);
        ensure(fails || !cheaper, || {
            format!("{variant:?}: one query fewer at stage {stage} is admissible and cheaper")
        })?;
    }
    let mut again = tuned.clone();
    let repeated = optimize_queries(&mut again, variant, field).map_err(|e| e.to_string())?;
    ensure(repeated.costs.expected_argument_bytes <= best.costs.expected_argument_bytes, || {
        "optimizing again costs more".into()
    })?;
    let polished = polish_queries(&mut tuned.clone(), variant, field, None).map_err(|e| e.to_string())?;
    ensure(polished.costs.expected_argument_bytes <= best.costs.expected_argument_bytes, || {
        "polishing costs more".into()
    })?;
    let within = polish_queries(&mut tuned.clone(), variant, field, Some(best.costs.iop_bytes))
        .map_err(|e| e.to_string())?;
    ensure(within.costs.iop_bytes <= best.costs.iop_bytes, || {
        "polishing exceeds its proof length limit".into()
    })?;
    uniform_minimal(parameters, variant)
}

fn uniform_minimal(parameters: &Parameters, variant: Variant) -> Result<(), String> {
    let Ok(queries) = uniform_queries(parameters, variant) else {
        return Ok(());
    };
    let mut trial = parameters.clone();
    for (stage, &count) in trial.stages.iter_mut().zip(&queries) {
        stage.queries = count;
    }
    let count = trial.stages.len() as u32;
    let budget = (trial.security_bits.saturating_sub(trial.pow_bits) + (2 * count - 1).ilog2() + 3) as f64;
    let layout = Layout::new(&trial, variant).map_err(|e| e.to_string())?;
    for (index, stage) in layout.stages.iter().enumerate() {
        let code = CodeBounds::new(stage.dimension(), stage.domain_size, trial.johnson_slack)
            .map_err(|e| e.to_string())?;
        let error_bits = |queries: usize| -(queries as f64) * (1.0 - code.delta).log2();
        ensure(error_bits(stage.queries) >= budget, || {
            format!("stage {index}: uniform queries miss the budget")
        })?;
    }
    ensure(queries.iter().all(|&q| (1..=MAX_QUERIES).contains(&q)), || format!("uniform queries {queries:?}"))
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    let field = if ctx.prng.chance(50) { Gf2_256::SIZE } else { Gf2_128::SIZE };
    let parameters =
        ctx.check("analysis_generate_parameters", |prng| crate::protocol::generate(prng, field))?;
    ctx.note(|| format!("analysis over {} bits: {parameters:?}", field.bits));
    for variant in VARIANTS {
        ctx.check("layout_follows_the_recurrences", |_| layout_invariants(&parameters, variant))?;
        ctx.check("soundness_one_error_per_message", |_| soundness_invariants(&parameters, variant, field))?;
        ctx.check("costs_match_the_layout", |_| costs_invariants(&parameters, variant, field))?;
    }
    let variant = ctx.prng.pick(&VARIANTS);
    ctx.check("query_optimization_locally_minimal", |_| tuning_invariants(&parameters, variant, field))?;
    ctx.check("parameter_encoding_binds_every_field", |_| encodings_bind(&parameters))?;
    ctx.check("expected_opening_exact_on_small_trees", |prng: &mut Prng| {
        exact_expected_opening(prng.range(1, 12), prng.range(1, 3) as u32)
    })?;
    ctx.check("johnson_slack_range", |_| {
        ensure(JOHNSON_SLACK_RANGE.contains(&parameters.johnson_slack), || "the slack is out of range".into())
    })?;
    Ok(())
}
