use crate::algebra::rank;
use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure};
use zo0k::field::BinaryField;
use zo0k::parameters::Stage;
use zo0k::witness::Row;
use zo0k::{Gf2_128, Gf2_256, Rng, RsField};

fn stage(queries: usize, extra: usize) -> Stage {
    Stage {
        folding: 3,
        main_len: 4,
        aux_len: 3,
        mask_len: 2,
        seed_len: queries + extra,
        pad_len: 0,
        domain_size: 64,
        queries,
    }
}

fn symbol_of_seed<F: RsField>(stage: &Stage, position: usize) -> Vec<F> {
    assert!(position < stage.domain_size);
    let functional: Vec<F> = (0..stage.seed_len)
        .map(|coordinate| {
            let mut row = Row {
                main: vec![F::ZERO; stage.main_len],
                aux: vec![F::ZERO; stage.aux_len],
                mask: [F::ZERO; 2],
                seed: vec![F::ZERO; stage.seed_len],
            };
            row.seed[coordinate] = F::ONE;
            F::evaluate_at(&row.polynomial(), position)
        })
        .collect();
    assert_eq!(functional.len(), stage.seed_len);
    functional
}

fn opened_symbols_are_uniform<F: RsField>(positions: &[usize], extra: usize) -> Result<(), String> {
    let queries = positions.len();
    let stage = stage(queries, extra);
    let matrix: Vec<Vec<F>> = positions.iter().map(|&p| symbol_of_seed(&stage, p)).collect();
    let found = rank(matrix);
    ensure(found == queries, || {
        format!("{}: symbols at {positions:?} with {extra} extra have rank {found}", F::NAME)
    })
}

fn unopened_columns_unpredictable<F: RsField>(
    rows: usize,
    opened: &[usize],
    unopened: usize,
    rng: &mut Rng,
) -> Result<(), String> {
    assert!(!opened.contains(&unopened));
    let stage = stage(opened.len(), 1);
    let seed_len = stage.seed_len;
    let symbol = |row: usize, position: usize| {
        let mut functional = vec![F::ZERO; rows * seed_len];
        functional[row * seed_len..(row + 1) * seed_len].copy_from_slice(&symbol_of_seed(&stage, position));
        functional
    };
    let mut view: Vec<Vec<F>> = (0..rows)
        .flat_map(|row| opened.iter().map(move |&position| (row, position)))
        .map(|(r, p)| symbol(r, p))
        .collect();
    let weights: Vec<F> = rng.fields(rows);
    for coordinate in 0..seed_len {
        let mut functional = vec![F::ZERO; rows * seed_len];
        for (row, &weight) in weights.iter().enumerate() {
            functional[row * seed_len + coordinate] = weight;
        }
        view.push(functional);
    }
    let view_rank = rank(view.clone());
    ensure(view_rank == rows * opened.len() + 1, || {
        format!("{}: view of {rows} rows at {opened:?} has rank {view_rank}", F::NAME)
    })?;
    view.extend((0..rows).map(|row| symbol(row, unopened)));
    let found = rank(view);
    ensure(found == view_rank + rows - 1, || {
        format!("{}: column {unopened} of {rows} rows adds rank {}", F::NAME, found - view_rank)
    })
}

fn without_extra_coordinate_determined<F: RsField>(opened: &[usize], unopened: usize) -> Result<(), String> {
    assert!(!opened.contains(&unopened));
    let stage = stage(opened.len(), 0);
    let mut matrix: Vec<Vec<F>> = opened.iter().map(|&p| symbol_of_seed(&stage, p)).collect();
    ensure(rank(matrix.clone()) == opened.len(), || format!("{}: opened symbols are dependent", F::NAME))?;
    matrix.push(symbol_of_seed(&stage, unopened));
    ensure(rank(matrix) == opened.len(), || {
        format!("{}: an unopened symbol is not determined without salt", F::NAME)
    })
}

fn fixed<F: RsField>() -> Result<(), String> {
    for positions in [vec![0, 1, 7, 19], vec![3, 5, 11, 63], vec![0]] {
        for extra in [0, 1] {
            opened_symbols_are_uniform::<F>(&positions, extra)?;
        }
    }
    let mut rng = Rng::from_seed([1; 32]);
    let cases: [(usize, Vec<usize>, usize); 4] =
        [(2, vec![0, 1, 7, 19], 5), (4, vec![3, 5, 11], 0), (8, vec![2], 30), (9, vec![1, 4, 9], 2)];
    for (rows, opened, unopened) in cases {
        unopened_columns_unpredictable::<F>(rows, &opened, unopened, &mut rng)?;
    }
    without_extra_coordinate_determined::<F>(&[0, 1, 7, 19], 5)
}

pub fn zero_knowledge_fixed() -> Result<(), String> {
    fixed::<Gf2_128>()?;
    fixed::<Gf2_256>()
}

fn random_case(prng: &mut Prng) -> (Vec<usize>, usize) {
    let opened = prng.subset(64, 6);
    let free: Vec<usize> = (0..64).filter(|p| !opened.contains(p)).collect();
    let unopened = prng.pick(&free);
    assert!(!opened.contains(&unopened));
    (opened, unopened)
}

pub fn hiding<F: RsField + BinaryField>(prng: &mut Prng) -> Result<(), String> {
    let (opened, unopened) = random_case(prng);
    opened_symbols_are_uniform::<F>(&opened, prng.range(0, 1))?;
    let rows = prng.range(2, 9);
    let mut rng = Rng::from_seed(prng.seed_bytes());
    unopened_columns_unpredictable::<F>(rows, &opened, unopened, &mut rng)?;
    without_extra_coordinate_determined::<F>(&opened, unopened)
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    if ctx.prng.chance(50) {
        ctx.check("zero_knowledge_rank_conditions", hiding::<Gf2_128>)
    } else {
        ctx.check("zero_knowledge_rank_conditions", hiding::<Gf2_256>)
    }
}
