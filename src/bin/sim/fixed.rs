use crate::analysis::{VARIANTS, exact_expected_opening, layout_invariants, soundness_invariants};
use crate::prng::Prng;
use crate::report::ensure;
use zo0k::costs::ExpectedOpening;
use zo0k::merkle::MerkleTree;
use zo0k::parameters::{Layout, MASK_LEN, MAX_QUERIES, Parameters, Round, StageParameters, Variant};
use zo0k::soundness::{CodeBounds, Soundness, analyze};
use zo0k::tuning::{
    Analysis, STANDARD_MAX_NUM_VARIABLES, assign_pow_bits, assign_pow_bits_and_slack, optimize_queries,
    polish_queries, uniform_queries,
};
use zo0k::{Blake3, Digest, Field, FieldSize, Gf2_128, Gf2_256};

const FIELD: FieldSize = Gf2_256::SIZE;

fn paper_parameters(pow_bits: u32) -> Parameters {
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

fn schedule(pow_bits: u32) -> Parameters {
    Parameters {
        num_variables: 20,
        stages: [(4, 448512), (3, 133120), (3, 50688), (3, 19968)]
            .map(|(folding, domain_size)| StageParameters::new(folding, domain_size, 0))
            .to_vec(),
        security_bits: 100,
        pow_bits,
        base_case_pow_bits: 0,
        johnson_slack: 0.0002,
    }
}

fn error(e: zo0k::Error) -> String {
    e.to_string()
}

pub fn recorded_layout() -> Result<(), String> {
    let parameters = paper_parameters(24);
    let layout = Layout::new(&parameters, Variant::ZeroKnowledge).map_err(error)?;
    let mains: Vec<usize> = layout.stages.iter().map(|s| s.main_len).collect();
    ensure(mains == [1 << 17, 1 << 14, 1 << 11, 1 << 8], || format!("main lengths {mains:?}"))?;
    ensure(layout.oracle_len() == 8 * (448512 + 133120 + 50688 + 19968) + 19968, || "oracle length".into())?;
    let last = layout.last();
    ensure(
        layout.message_len() == 2 * 12 + 2 * 4 + last.main_len + last.aux_len + last.seed_len + 2,
        || "message length".into(),
    )?;
    ensure(layout.round_count() == 12 + 3 * 4, || "round count".into())?;
    let baseline = Layout::new(&parameters, Variant::Baseline).map_err(error)?;
    ensure(baseline.oracle_len() == 8 * (448512 + 133120 + 50688 + 19968), || {
        "baseline oracle length".into()
    })?;
    ensure(baseline.message_len() == 2 * 12 + 3 + 256 && baseline.round_count() == 12 + 7, || {
        "baseline lengths".into()
    })?;
    for stage in &baseline.stages {
        ensure((stage.aux_len, stage.mask_len, stage.seed_len, stage.pad_len) == (0, 0, 0, 0), || {
            "baseline randomness".into()
        })?;
    }
    let mut ground = parameters.clone();
    ground.stages[1].sumcheck_pow_bits = 7;
    ground.stages[2].ood_pow_bits = 5;
    ground.base_case_pow_bits = 9;
    for variant in VARIANTS {
        layout_invariants(&parameters, variant)?;
        layout_invariants(&ground, variant)?;
    }
    ensure(MASK_LEN == 2, || "the mask has two entries".into())
}

pub fn invalid_parameters() -> Result<(), String> {
    let edits: [fn(&mut Parameters); 22] = [
        |p| p.stages.clear(),
        |p| p.stages[1].folding = 0,
        |p| p.stages[0].folding = 12,
        |p| p.stages[2].queries = 0,
        |p| p.stages[3].domain_size = 291,
        |p| p.stages[0].domain_size = (1 << 32) + 1,
        |p| p.stages[0].folding = usize::MAX,
        |p| p.stages.iter_mut().for_each(|stage| stage.folding = 1 << 62),
        |p| p.num_variables = 41,
        |p| p.security_bits = 0,
        |p| p.security_bits = 257,
        |p| p.security_bits = u32::MAX,
        |p| p.stages[0].queries = usize::MAX,
        |p| p.stages[0].queries = MAX_QUERIES + 1,
        |p| p.stages[0].domain_size = usize::MAX,
        |p| p.johnson_slack = f64::NAN,
        |p| p.johnson_slack = 0.0001,
        |p| p.pow_bits = 41,
        |p| p.base_case_pow_bits = 41,
        |p| p.stages[0].sumcheck_pow_bits = 41,
        |p| p.stages[0].ood_pow_bits = 41,
        |p| p.stages[3].ood_pow_bits = 1,
    ];
    for (index, edit) in edits.iter().enumerate() {
        let mut parameters = paper_parameters(24);
        edit(&mut parameters);
        ensure(Layout::new(&parameters, Variant::ZeroKnowledge).is_err(), || {
            format!("invalid edit {index} is accepted")
        })?;
    }
    let huge = Parameters {
        num_variables: 40,
        stages: vec![StageParameters::new(40, 1 << 32, 8)],
        security_bits: 1,
        pow_bits: 0,
        base_case_pow_bits: 0,
        johnson_slack: 0.1,
    };
    ensure(Layout::new(&huge, Variant::ZeroKnowledge).is_err(), || "oracles above 2^48 are accepted".into())?;
    let edits: [fn(&mut Parameters); 3] = [
        |p| p.stages[0].sumcheck_pow_bits = 20,
        |p| p.stages[0].ood_pow_bits = 6,
        |p| p.base_case_pow_bits = 15,
    ];
    let mut encodings = vec![paper_parameters(24).to_bytes()];
    for edit in edits {
        let mut parameters = paper_parameters(24);
        edit(&mut parameters);
        Layout::new(&parameters, Variant::ZeroKnowledge).map_err(error)?;
        encodings.push(parameters.to_bytes());
    }
    encodings.sort();
    encodings.dedup();
    ensure(encodings.len() == 4, || "proofs of work are not bound by the encoding".into())
}

pub fn recorded_code_bounds() -> Result<(), String> {
    let (d, n, slack) = (131162usize, 448512usize, 0.0002);
    let bounds = CodeBounds::new(d, n, slack).map_err(error)?;
    let rho = (d - 1) as f64 / n as f64;
    let delta = 1.0 - ((d + 1) as f64 / n as f64).sqrt() - slack;
    ensure((bounds.delta - delta).abs() < 1e-15 && bounds.delta < delta, || "delta".into())?;
    let johnson = (1.0 - rho) / ((1.0 - delta).powi(2) - rho);
    ensure(bounds.list_size >= johnson && bounds.list_size < johnson + 1.0 + 1e-6, || "list size".into())?;
    ensure(bounds.delta < 1.0 - rho.sqrt(), || "delta beyond Johnson".into())?;
    let m = (rho.sqrt() / (1.0 - rho.sqrt() - delta)).ceil().max(3.0) + 0.5;
    let mca = (2.0 * m.powi(5) + 3.0 * m * delta * rho) / (3.0 * rho.powf(1.5)) * n as f64 + m / rho.sqrt();
    ensure((bounds.mca_numerator / mca - 1.0).abs() < 1e-9, || "mca numerator".into())?;
    for dimension in [n, n - 2, 1, 0] {
        ensure(CodeBounds::new(dimension, n, slack).is_err(), || {
            format!("a code of dimension {dimension} is accepted")
        })?;
    }
    Ok(())
}

pub fn recorded_soundness() -> Result<(), String> {
    let close = |actual: f64, expected: f64| (actual / expected - 1.0).abs() < 1e-12;
    let parameters = paper_parameters(24);
    let layout = Layout::new(&parameters, Variant::ZeroKnowledge).map_err(error)?;
    let soundness = analyze(&layout, &parameters, 256).map_err(error)?;
    let codes = [
        (131162, 0.4590224708398836, 3205.0, 2.4732947342722602e23),
        (16451, 0.6482495194362331, 5630.0, 2.066351765865866e22),
        (2094, 0.7964990369780394, 7936.0, 6.096142117124786e20),
        (291, 0.8788727597441442, 6634.0, 2.2855128317013386e18),
    ];
    for (code, (dimension, delta, list_size, mca)) in soundness.codes.iter().zip(codes) {
        ensure(code.dimension == dimension && code.list_size == list_size, || {
            format!("code of dimension {dimension}")
        })?;
        ensure(close(code.delta, delta) && close(code.mca_numerator, mca), || {
            format!("bounds of dimension {dimension}")
        })?;
    }
    let rounds = [
        (Round::SumcheckBatching, 2.7678920219077695e-74, 0.0),
        (Round::Fold(1), 8.543916084641106e-54, 0.0),
        (Round::Fold(2), 4.271958042320553e-54, 0.0),
        (Round::Fold(3), 2.1359790211602764e-54, 0.0),
        (Round::OutOfDomain, 1.7951991056916693e-65, 0.0),
        (Round::QueryAndBatch, 4.2787033489359916e-72, 3.646471006694588e-31),
    ];
    for (round, (kind, algebraic, query)) in soundness.rounds.iter().zip(rounds) {
        ensure((round.stage, round.round) == (0, kind), || format!("round {kind:?}"))?;
        ensure(round.pow_bits == if kind == Round::QueryAndBatch { 24 } else { 0 }, || {
            format!("pow of {kind:?}")
        })?;
        ensure(close(round.algebraic * 2f64.powi(round.pow_bits as i32), algebraic), || {
            format!("algebraic error of {kind:?}")
        })?;
        ensure(round.query == query || close(round.query, query), || format!("query error of {kind:?}"))?;
    }
    let last = &soundness.rounds[22..];
    ensure((last[0].round, last[1].round) == (Round::BaseCaseCombination, Round::BaseCaseQuery), || {
        "last rounds".into()
    })?;
    ensure(
        close(last[0].algebraic, 1.973807404940402e-59) && close(last[1].query, 8.700966316775354e-32),
        || "base case errors".into(),
    )?;
    ensure(
        close(soundness.total_bits, 100.00237296826924)
            && close(soundness.algebraic_bits, 175.36281156799046),
        || "total bits".into(),
    )?;
    let mut baseline = parameters.clone();
    for (stage, (domain_size, queries)) in
        baseline.stages.iter_mut().zip([(452608, 86), (138240, 51), (46592, 35), (17664, 26)])
    {
        (stage.domain_size, stage.queries) = (domain_size, queries);
    }
    let layout = Layout::new(&baseline, Variant::Baseline).map_err(error)?;
    let soundness = analyze(&layout, &baseline, 256).map_err(error)?;
    ensure(
        close(soundness.total_bits, 100.00981160174943)
            && close(soundness.algebraic_bits, 175.36369215618043),
        || "baseline bits".into(),
    )
}

pub fn soundness_targets() -> Result<(), String> {
    let parameters = paper_parameters(24);
    for variant in VARIANTS {
        soundness_invariants(&parameters, variant, FIELD)?;
    }
    let layout = Layout::new(&parameters, Variant::ZeroKnowledge).map_err(error)?;
    ensure(analyze(&layout, &parameters, 128).is_err(), || "the smaller field meets 100 bits".into())?;
    let weak = paper_parameters(0);
    ensure(analyze(&layout, &weak, 256).is_err(), || "no grinding meets 100 bits".into())?;
    let mut fewer = paper_parameters(24);
    fewer.stages[0].queries -= 3;
    let layout = Layout::new(&fewer, Variant::ZeroKnowledge).map_err(error)?;
    ensure(analyze(&layout, &fewer, 256).is_err(), || "fewer queries meet the target".into())?;
    fewer.security_bits = 90;
    ensure(analyze(&layout, &fewer, 256).is_ok(), || "a lower target is not met".into())?;
    let mut ground = paper_parameters(24);
    ground.stages[0].sumcheck_pow_bits = 20;
    ground.stages[1].ood_pow_bits = 6;
    ground.base_case_pow_bits = 11;
    let layout = Layout::new(&ground, Variant::ZeroKnowledge).map_err(error)?;
    let before = Soundness::new(&layout, &paper_parameters(24), 128).map_err(error)?;
    let after = Soundness::new(&layout, &ground, 128).map_err(error)?;
    ensure(after.algebraic_bits > before.algebraic_bits + 2.0, || {
        "grinding does not improve the algebraic error".into()
    })?;
    let baseline = Layout::new(&ground, Variant::Baseline).map_err(error)?;
    let rounds = Soundness::new(&baseline, &ground, 128).map_err(error)?.rounds;
    ensure(rounds.iter().all(|round| round.pow_bits != 11), || {
        "the baseline credits a base case combination".into()
    })
}

pub fn expected_openings() -> Result<(), String> {
    for num_leaves in [1usize, 2, 3, 5, 8, 9, 11] {
        for queries in 1..=3u32 {
            exact_expected_opening(num_leaves, queries)?;
        }
    }
    let (num_leaves, queries) = (9792, 69);
    let tree = MerkleTree::new::<Blake3>(vec![Digest::default(); num_leaves]);
    let mut prng = Prng::new(0x5eed_0030);
    let trials = 300;
    let mut digests = 0;
    for _ in 0..trials {
        let mut positions: Vec<usize> = (0..queries).map(|_| prng.below(num_leaves)).collect();
        positions.sort_unstable();
        positions.dedup();
        digests += tree.open(&positions).len();
    }
    let expected = ExpectedOpening::new(num_leaves, queries).digests;
    ensure((digests as f64 / trials as f64 / expected - 1.0).abs() < 0.01, || {
        "sampled openings differ from the expectation".into()
    })
}

pub fn queries_are_tuned() -> Result<(), String> {
    for variant in VARIANTS {
        let mut parameters = schedule(24);
        let queries = uniform_queries(&parameters, variant).map_err(error)?;
        for (stage, &count) in parameters.stages.iter_mut().zip(&queries) {
            stage.queries = count;
        }
        let analysis = Analysis::new(&parameters, variant, FIELD).map_err(error)?;
        for (stage, code) in analysis.layout.stages.iter().zip(&analysis.soundness.codes) {
            let error_bits = |queries: usize| -(queries as f64) * (1.0 - code.delta).log2();
            ensure(error_bits(stage.queries) >= 81.0 && error_bits(stage.queries - 1) < 81.01, || {
                "uniform queries".into()
            })?;
        }
        let mut parameters = schedule(24);
        let best = optimize_queries(&mut parameters, variant, FIELD).map_err(error)?;
        let optimized: Vec<usize> = parameters.stages.iter().map(|stage| stage.queries).collect();
        ensure(optimized.iter().sum::<usize>() < queries.iter().sum::<usize>(), || {
            "optimized queries are not fewer".into()
        })?;
        let polished = polish_queries(&mut parameters.clone(), variant, FIELD, None).map_err(error)?;
        ensure(polished.costs.expected_argument_bytes <= best.costs.expected_argument_bytes, || {
            "polished".into()
        })?;
        let within = polish_queries(&mut parameters.clone(), variant, FIELD, Some(best.costs.iop_bytes))
            .map_err(error)?;
        ensure(within.costs.iop_bytes <= best.costs.iop_bytes, || "polished within a limit".into())?;
        ensure(
            best.soundness.total_bits >= 100.0
                && best.soundness.total_bits < 100.5
                && best.parameters == parameters,
            || "optimized".into(),
        )?;
        for stage in 0..optimized.len() {
            let mut fewer = parameters.clone();
            fewer.stages[stage].queries -= 1;
            ensure(Analysis::new(&fewer, variant, FIELD).is_err(), || {
                format!("one query fewer at stage {stage}")
            })?;
        }
        let mut again = schedule(24);
        optimize_queries(&mut again, variant, FIELD).map_err(error)?;
        ensure(again == parameters, || "optimization is not deterministic".into())?;
    }
    let mut small = schedule(24);
    small.stages[3].domain_size = 400;
    ensure(uniform_queries(&small, Variant::ZeroKnowledge).is_err(), || "a small domain has queries".into())?;
    small.stages.clear();
    ensure(uniform_queries(&small, Variant::ZeroKnowledge).is_err(), || "no stages have queries".into())
}

pub fn standard_parameters() -> Result<(), String> {
    for num_variables in (1..=22).chain([STANDARD_MAX_NUM_VARIABLES]) {
        let mut parameters = Parameters::standard(num_variables, 100, FIELD).map_err(error)?;
        let analysis = Analysis::new(&parameters, Variant::ZeroKnowledge, FIELD).map_err(error)?;
        ensure(analysis.soundness.total_bits >= 100.0, || {
            format!("standard {num_variables} misses 100 bits")
        })?;
        let folded: usize = parameters.stages.iter().map(|stage| stage.folding).sum();
        ensure(folded <= num_variables, || format!("standard {num_variables} folds {folded}"))?;
        let baseline = optimize_queries(&mut parameters, Variant::Baseline, FIELD).is_ok();
        ensure(baseline == (num_variables > 1), || format!("baseline queries for {num_variables}"))?;
    }
    for invalid in [0, STANDARD_MAX_NUM_VARIABLES + 1, 41, usize::MAX] {
        ensure(Parameters::standard(invalid, 100, FIELD).is_err(), || format!("standard {invalid} exists"))?;
    }
    ensure(
        Parameters::standard(10, 0, FIELD).is_err() && Parameters::standard(10, u32::MAX, FIELD).is_err(),
        || "standard targets".into(),
    )?;
    let field = Gf2_128::SIZE;
    for num_variables in [1, 2, 7, 13, 19] {
        let parameters = Parameters::standard(num_variables, 100, field).map_err(error)?;
        let analysis = Analysis::new(&parameters, Variant::ZeroKnowledge, field).map_err(error)?;
        ensure(analysis.soundness.total_bits >= 100.0 && parameters.pow_bits == 0, || {
            format!("small standard {num_variables}")
        })?;
        ensure(analysis.soundness.rounds.iter().all(|round| round.pow_bits <= 24), || {
            "standard grinding limit".into()
        })?;
        ensure(parameters.stages.iter().all(|stage| stage.sumcheck_pow_bits > 0), || {
            "standard sumcheck grinding".into()
        })?;
    }
    let parameters = Parameters::standard(12, 100, FIELD).map_err(error)?;
    ensure(Analysis::new(&parameters, Variant::ZeroKnowledge, field).is_err(), || {
        "the larger field's parameters suffice".into()
    })?;
    let small = Parameters::standard(12, 60, field).map_err(error)?;
    let analysis = Analysis::new(&small, Variant::ZeroKnowledge, field).map_err(error)?;
    let large = Analysis::new(&small, Variant::ZeroKnowledge, FIELD).map_err(error)?;
    ensure(
        analysis.soundness.total_bits >= 60.0 && 2 * analysis.costs.oracle_bytes == large.costs.oracle_bytes,
        || "field size in costs".into(),
    )
}

pub fn slack_grid() -> Vec<f64> {
    let mantissas = [10, 12, 15, 20, 25, 30, 40, 50, 60, 80];
    let scales = [1e6, 1e5, 1e4, 1e3, 1e2];
    let values = scales.iter().flat_map(|scale| mantissas.map(|mantissa| mantissa as f64 / scale));
    let grid: Vec<f64> =
        values.filter(|slack| zo0k::parameters::JOHNSON_SLACK_RANGE.contains(slack)).collect();
    assert!(grid.is_sorted_by(|a, b| a < b));
    grid
}

fn algebraic_bits(parameters: &Parameters, variant: Variant, field: FieldSize) -> Result<f64, String> {
    let mut trial = parameters.clone();
    for (stage, queries) in trial.stages.iter_mut().zip(uniform_queries(parameters, variant).map_err(error)?)
    {
        stage.queries = queries;
    }
    let layout = Layout::new(&trial, variant).map_err(error)?;
    Ok(Soundness::new(&layout, &trial, field.bits).map_err(error)?.algebraic_bits)
}

fn slots(parameters: &mut Parameters) -> Vec<&mut u32> {
    let mut slots = vec![&mut parameters.base_case_pow_bits];
    for stage in &mut parameters.stages {
        slots.extend([&mut stage.sumcheck_pow_bits, &mut stage.ood_pow_bits]);
    }
    slots
}

pub fn grinding_is_assigned() -> Result<(), String> {
    for variant in VARIANTS {
        let mut parameters = schedule(24);
        parameters.stages[0].sumcheck_pow_bits = 9;
        parameters.stages[1].ood_pow_bits = 2;
        parameters.base_case_pow_bits = 3;
        assign_pow_bits_and_slack(&mut parameters, variant, FIELD, 24).map_err(error)?;
        ensure(parameters == schedule(24), || "the larger field gets grinding".into())?;
        assign_pow_bits(&mut parameters, variant, FIELD, 0).map_err(error)?;
        ensure(parameters == schedule(24), || "the larger field gets grinding without a budget".into())?;
    }
    let grid = slack_grid();
    ensure(
        (grid[0], grid[grid.len() - 1]) == (0.0002, 0.1) && grid.contains(&0.025) && grid.contains(&0.0012),
        || "slack grid".into(),
    )?;
    let field = Gf2_128::SIZE;
    for variant in VARIANTS {
        small_field_grinding(variant, field, &grid)?;
    }
    let mut parameters = schedule(24);
    ensure(assign_pow_bits_and_slack(&mut parameters, Variant::ZeroKnowledge, field, 3).is_err(), || {
        "three bits suffice".into()
    })?;
    ensure(parameters == schedule(24), || "a failed assignment changes the parameters".into())
}

fn small_field_grinding(variant: Variant, field: FieldSize, grid: &[f64]) -> Result<(), String> {
    let limit = 24;
    let mut parameters = schedule(24);
    ensure(assign_pow_bits(&mut parameters.clone(), variant, field, limit).is_err(), || {
        "the smallest slack suffices".into()
    })?;
    assign_pow_bits_and_slack(&mut parameters, variant, field, limit).map_err(error)?;
    let position = grid
        .iter()
        .position(|&slack| slack == parameters.johnson_slack)
        .ok_or("the slack is not on the grid")?;
    let mut smaller = schedule(24);
    smaller.johnson_slack = grid[position - 1];
    ensure(assign_pow_bits(&mut smaller, variant, field, limit).is_err(), || {
        "a smaller slack suffices".into()
    })?;
    ensure(algebraic_bits(&parameters, variant, field)? >= 101.0, || {
        "the algebraic error misses its budget".into()
    })?;
    let count = slots(&mut parameters).len();
    let mut ground = 0;
    for index in 0..count {
        let mut fewer = parameters.clone();
        let bits = slots(&mut fewer).swap_remove(index);
        ensure(*bits <= limit, || "a slot exceeds the limit".into())?;
        if *bits > 0 {
            *bits -= 1;
            ground += 1;
            ensure(algebraic_bits(&fewer, variant, field)? < 101.0, || {
                format!("slot {index} is not minimal")
            })?;
        }
    }
    ensure(
        parameters.stages.iter().all(|stage| stage.sumcheck_pow_bits > 0)
            && parameters.stages[3].ood_pow_bits == 0,
        || "sumcheck grinding".into(),
    )?;
    ensure(
        (parameters.base_case_pow_bits > 0) == (variant == Variant::ZeroKnowledge) && ground >= 5,
        || "base case grinding".into(),
    )?;
    let analysis = optimize_queries(&mut parameters, variant, field).map_err(error)?;
    let nonces = analysis.soundness.rounds.iter().filter(|round| round.pow_bits > 0).count();
    ensure(analysis.soundness.total_bits >= 100.0 && analysis.costs.pow_nonce_bytes == 8 * nonces, || {
        "nonce bytes".into()
    })?;
    ensure(nonces > analysis.layout.stages.len() + 10, || "too few proofs of work".into())
}
