//! A heuristic search for schedules with a small argument.
//!
//! The search minimizes the expected argument size of a variant of the
//! protocol for tables of a given size, subject to a limit on the IOP proof
//! length. A schedule is a vector of folding parameters together with a
//! domain size per stage. Its proofs of work and Johnson slack follow from
//! [`zo0k::tuning::assign_pow_bits_and_slack`], and its query counts from
//! [`zo0k::tuning::optimize_queries`].
//!
//! Domain sizes move on a grid of 128 sizes per octave. For a fixed vector of
//! folding parameters the search descends: it moves to the best schedule
//! among those that change the size of one stage, or enlarge one stage and
//! shrink another, as long as that lowers the argument size. Over vectors of
//! folding parameters it does a local search from the best constant vector,
//! with a coarse descent for every neighbour and a full descent for the best
//! few vectors. The result is a local optimum and not provably the best
//! schedule.

use crate::Result;
use crate::args::Matches;
use std::collections::HashMap;
use std::sync::Mutex;
use zo0k::json::ToJson;
use zo0k::reed_solomon::MAX_DOMAIN_SIZE;
use zo0k::tuning::{Analysis, assign_pow_bits_and_slack, optimize_queries, polish_queries};
use zo0k::{FieldSize, Parameters, StageParameters, Variant};

pub struct Search {
    num_variables: usize,
    stages: Option<usize>,
    max_stages: usize,
    budget: f64,
    security: u32,
    pow_bits: u32,
    max_pow_bits: u32,
    johnson_slack: f64,
    baseline: bool,
}

impl Search {
    pub fn new(matches: &Matches) -> Self {
        let search = Self {
            num_variables: matches.count("num-variables").expect("a required number of variables"),
            stages: matches.count("stages"),
            max_stages: matches.count("max-stages").expect("a default number of stages"),
            budget: matches.float("budget").expect("a default budget"),
            security: matches.bits("security").expect("a default security target"),
            pow_bits: matches.bits("pow-bits").expect("a default number of bits"),
            max_pow_bits: matches.bits("max-pow-bits").expect("a default number of bits"),
            johnson_slack: matches.float("johnson-slack").expect("a default slack"),
            baseline: matches.flag("baseline"),
        };
        assert!((1..=40).contains(&search.num_variables) && search.max_pow_bits <= 40);
        assert!(
            (1..=8).contains(&search.max_stages)
                && search.stages.is_none_or(|count| (1..=8).contains(&count))
        );
        search
    }
}

/// Domain sizes are `j·2^t` with `GRID ≤ j < 2·GRID`.
const GRID: usize = 128;
/// Moves of a domain size on the grid, for a full and for a coarse descent.
const FINE_STEPS: [isize; 4] = [1, 4, 16, 64];
const COARSE_STEPS: [isize; 2] = [16, 64];
/// Rounds of a local search over vectors of folding parameters, number of
/// schedules from which such a search starts, number of vectors that get a
/// full descent afterwards, and repetitions of the whole.
const STRUCTURE_ROUNDS: usize = 8;
const CENTERS: usize = 3;
const FINALISTS: usize = 8;
const PASSES: usize = 4;
/// Largest number of stages for which the final query counts are polished.
const POLISH_STAGES: usize = 6;

/// The size `steps` positions after `size` on the grid, if there is one.
fn shift(size: usize, steps: isize) -> Option<usize> {
    let exponent = (size.max(1).ilog2() as usize).saturating_sub(GRID.ilog2() as usize);
    let index = (exponent * GRID + (size >> exponent)) as isize - GRID as isize + steps;
    let index = usize::try_from(index).ok()?;
    let size = (GRID + index % GRID) << (index / GRID);
    (size <= MAX_DOMAIN_SIZE).then_some(size)
}

/// Folding parameters and domain sizes, stage by stage.
type Schedule = (Vec<usize>, Vec<usize>);

#[derive(Clone)]
struct Candidate {
    schedule: Schedule,
    analysis: Analysis,
}

impl Candidate {
    fn argument_bytes(&self) -> f64 {
        self.analysis.costs.expected_argument_bytes
    }
}

struct Searcher<'a> {
    arguments: &'a Search,
    variant: Variant,
    field: FieldSize,
    iop_limit: usize,
    /// Every schedule analyzed so far; `None` for inadmissible ones.
    cache: Mutex<HashMap<Schedule, Option<Candidate>>>,
}

impl Searcher<'_> {
    /// The schedule with optimized query counts, if it is admissible.
    fn analyze(&self, schedule: &Schedule) -> Option<Candidate> {
        if let Some(known) = self.cache.lock().unwrap().get(schedule) {
            return known.clone();
        }
        let mut parameters = Parameters {
            num_variables: self.arguments.num_variables,
            stages: schedule
                .0
                .iter()
                .zip(&schedule.1)
                .map(|(&folding, &domain_size)| StageParameters::new(folding, domain_size, 0))
                .collect(),
            security_bits: self.arguments.security,
            pow_bits: self.arguments.pow_bits,
            base_case_pow_bits: 0,
            johnson_slack: self.arguments.johnson_slack,
        };
        let (variant, field) = (self.variant, self.field);
        let candidate =
            assign_pow_bits_and_slack(&mut parameters, variant, field, self.arguments.max_pow_bits)
                .and_then(|()| optimize_queries(&mut parameters, variant, field))
                .ok()
                .map(|analysis| Candidate { schedule: schedule.clone(), analysis });
        self.cache.lock().unwrap().insert(schedule.clone(), candidate.clone());
        candidate
    }

    /// [`Searcher::analyze`], if the proof length is within the limit.
    fn evaluate(&self, schedule: &Schedule) -> Option<Candidate> {
        self.analyze(schedule).filter(|candidate| candidate.analysis.costs.iop_bytes <= self.iop_limit)
    }

    /// The best of the given schedules, if it beats `incumbent`.
    fn best(&self, schedules: Vec<Schedule>, incumbent: Option<&Candidate>) -> Option<Candidate> {
        zo0k::parallel::filter_map(&schedules, |schedule| self.evaluate(schedule))
            .into_iter()
            .min_by(|a, b| {
                a.argument_bytes().total_cmp(&b.argument_bytes()).then(a.schedule.cmp(&b.schedule))
            })
            .filter(|best| {
                incumbent.is_none_or(|incumbent| best.argument_bytes() < incumbent.argument_bytes())
            })
    }

    /// Descend over domain sizes from `start` to a local optimum.
    fn descend(&self, start: Candidate, steps: &[isize]) -> Candidate {
        let mut current = start;
        loop {
            let (folds, sizes) = &current.schedule;
            let moved = |stage: usize, by: isize| shift(sizes[stage], by).map(|size| (stage, size));
            let mut neighbours = Vec::new();
            for stage in 0..sizes.len() {
                for &step in steps {
                    let (up, down) = (moved(stage, step), moved(stage, -step));
                    neighbours.extend([up, down].into_iter().flatten().map(|change| vec![change]));
                    // Enlarge this stage and shrink another: the trade that
                    // the limit on the proof length forces.
                    for other in (0..sizes.len()).filter(|&other| other != stage) {
                        for &other_step in steps {
                            if let (Some(up), Some(down)) = (up, moved(other, -other_step)) {
                                neighbours.push(vec![up, down]);
                            }
                        }
                    }
                }
            }
            let schedules = neighbours
                .into_iter()
                .map(|changes| {
                    let mut sizes = sizes.clone();
                    for (stage, size) in changes {
                        sizes[stage] = size;
                    }
                    (folds.clone(), sizes)
                })
                .collect();
            match self.best(schedules, Some(&current)) {
                Some(better) => current = better,
                None => return current,
            }
        }
    }

    /// A starting schedule for the given folding parameters: every stage at
    /// the given ratios of domain size to main block length, all shrunk
    /// together until the proof length is within the limit.
    fn start(&self, folds: &[usize], ratios: &[f64]) -> Option<Candidate> {
        let total: usize = folds.iter().sum();
        if folds.contains(&0) || total > self.arguments.num_variables {
            return None;
        }
        let mut remaining = self.arguments.num_variables;
        let mut sizes: Vec<usize> = folds
            .iter()
            .zip(ratios.iter().chain(std::iter::repeat(ratios.last()?)))
            .map(|(&folding, &ratio)| {
                remaining -= folding;
                // Room for the seed and the auxiliary block.
                let size = ratio * ((1usize << remaining) + 256) as f64;
                shift((size as usize).clamp(GRID, MAX_DOMAIN_SIZE), 0).expect("the size is on the grid")
            })
            .collect();
        // Shrinking the domains raises the rates, so a schedule that is not
        // admissible does not become so.
        for _ in 0..400 {
            let candidate = self.analyze(&(folds.to_vec(), sizes.clone()))?;
            if candidate.analysis.costs.iop_bytes <= self.iop_limit {
                return Some(candidate);
            }
            sizes = sizes.iter().map(|&size| shift(size, -2).unwrap_or(size)).collect();
        }
        None
    }

    /// The vectors of folding parameters next to `folds`: one fold more or
    /// fewer in a stage, one fold moved between stages, a further stage, and
    /// the last stage removed or merged into the one before it.
    fn fold_neighbours(&self, folds: &[usize]) -> Vec<Vec<usize>> {
        let count = folds.len();
        let mut vectors = Vec::new();
        for stage in 0..count {
            for delta in [-1isize, 1] {
                let mut vector = folds.to_vec();
                vector[stage] = vector[stage].saturating_add_signed(delta);
                vectors.push(vector);
            }
            for other in (0..count).filter(|&other| other != stage) {
                let mut vector = folds.to_vec();
                vector[stage] -= 1;
                vector[other] += 1;
                vectors.push(vector);
            }
        }
        if self.arguments.stages.is_none() {
            if count < self.arguments.max_stages {
                vectors.extend((1..=4).map(|last| [folds, &[last][..]].concat()));
            }
            if count > 1 {
                vectors.push(folds[..count - 1].to_vec());
                vectors.push([&folds[..count - 2], &[folds[count - 2] + folds[count - 1]][..]].concat());
            }
        }
        vectors.sort();
        vectors.dedup();
        vectors
    }

    /// The best schedule that the search finds.
    fn search(&self) -> Option<Candidate> {
        let ratios_of = |candidate: &Candidate| -> Vec<f64> {
            let stages = &candidate.analysis.layout.stages;
            stages.iter().map(|stage| stage.domain_size as f64 / stage.main_len as f64).collect()
        };
        let coarse = |folds: &Vec<usize>, ratios: &[f64]| {
            self.start(folds, ratios).map(|start| self.descend(start, &COARSE_STEPS))
        };
        let by_size = |a: &Candidate, b: &Candidate| a.argument_bytes().total_cmp(&b.argument_bytes());
        // The best schedule found for each vector of folding parameters.
        let mut pool: HashMap<Vec<usize>, Candidate> = HashMap::new();
        let keep = |pool: &mut HashMap<Vec<usize>, Candidate>, found: Vec<Candidate>| {
            for candidate in found {
                let entry = pool.entry(candidate.schedule.0.clone()).or_insert_with(|| candidate.clone());
                if candidate.argument_bytes() < entry.argument_bytes() {
                    *entry = candidate;
                }
            }
        };

        // Constant vectors of folding parameters. The first domain is four
        // times as long as its main block, and the ratio grows from stage to
        // stage by a constant factor: later oracles are small, so that low
        // rates are cheap for them.
        let counts = match self.arguments.stages {
            Some(count) => count..=count,
            None => 1..=self.arguments.max_stages,
        };
        let seeds: Vec<(Vec<usize>, Vec<f64>)> = counts
            .flat_map(|count| (1..=12).map(move |folding| vec![folding; count]))
            .filter(|folds| folds.iter().sum::<usize>() <= self.arguments.num_variables)
            .flat_map(|folds| {
                let ratios =
                    |growth: f64| (0..folds.len()).map(|stage| 4.0 * growth.powi(stage as i32)).collect();
                [1.0, 2.0, 3.0].map(|growth| (folds.clone(), ratios(growth)))
            })
            .collect();
        keep(&mut pool, zo0k::parallel::filter_map(&seeds, |(folds, ratios)| coarse(folds, ratios)));

        // A coarse descent is a poor guide to the vector that is best after a
        // full one. So alternate: local searches over vectors by coarse
        // descents from the best few schedules, then full descents from the
        // best few vectors, until the best vector no longer changes.
        let ranked = |pool: &HashMap<Vec<usize>, Candidate>, count: usize| {
            let mut candidates: Vec<Candidate> = pool.values().cloned().collect();
            candidates.sort_by(|a, b| by_size(a, b).then(a.schedule.cmp(&b.schedule)));
            candidates.truncate(count);
            candidates
        };
        let mut previous_best: Option<Vec<usize>> = None;
        for _ in 0..PASSES {
            let centers = ranked(&pool, CENTERS);
            if previous_best.as_ref() == Some(&centers.first()?.schedule.0) {
                break;
            }
            previous_best = Some(centers[0].schedule.0.clone());
            for mut best in centers {
                for _ in 0..STRUCTURE_ROUNDS {
                    let ratios = ratios_of(&best);
                    let neighbours = self.fold_neighbours(&best.schedule.0);
                    let found: Vec<Candidate> =
                        zo0k::parallel::filter_map(&neighbours, |folds| coarse(folds, &ratios));
                    let round_best = found.iter().min_by(|a, b| by_size(a, b)).cloned();
                    keep(&mut pool, found);
                    match round_best {
                        Some(found) if found.argument_bytes() < best.argument_bytes() => best = found,
                        _ => break,
                    }
                }
            }
            let refined = ranked(&pool, FINALISTS).into_iter().map(|start| self.descend(start, &FINE_STEPS));
            keep(&mut pool, refined.collect());
        }
        pool.into_values().min_by(|a, b| by_size(a, b).then(a.schedule.cmp(&b.schedule)))
    }
}

pub fn search(arguments: &Search, field: FieldSize) -> Result<()> {
    let variant = if arguments.baseline { Variant::Baseline } else { Variant::ZeroKnowledge };
    let table_bytes = (field.bytes << arguments.num_variables) as f64;
    let iop_limit = (arguments.budget * table_bytes) as usize;
    let searcher = Searcher { arguments, variant, field, iop_limit, cache: Mutex::new(HashMap::new()) };
    let best = searcher.search().ok_or("no admissible schedule was found within the limit")?;
    let mut parameters = best.analysis.parameters.clone();
    // The window of the polish has 5^stages allocations.
    let analysis = if parameters.stages.len() <= POLISH_STAGES {
        polish_queries(&mut parameters, variant, field, Some(iop_limit))?
    } else {
        best.analysis
    };
    eprintln!(
        "evaluated {} schedules; argument {:.2} KiB, IOP proof length {:.2}x the table",
        searcher.cache.lock().unwrap().len(),
        analysis.costs.expected_argument_bytes / 1024.0,
        analysis.costs.iop_bytes as f64 / table_bytes,
    );
    println!("{}", analysis.parameters.to_json().to_string_pretty());
    Ok(())
}
