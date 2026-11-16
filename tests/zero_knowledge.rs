//! The hiding properties of the row encoding on which the paper's
//! zero-knowledge analysis and its appendix "Hiding Merkle commitments
//! without salts" rely, checked as rank conditions on the actual encoding.
//!
//! The symbols of a row are affine functions of its seed, with a linear part
//! that does not depend on the message or the mask. So the statements below
//! are about that linear part.

use zo0k::parameters::Stage;
use zo0k::witness::Row;
use zo0k::{Field, Gf2_128, Gf2_256, Rng, RsField};

/// A stage whose rows have `queries + extra` seed coordinates.
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

/// The linear map from the seed of a row to its symbol at `position`.
fn symbol_of_seed<F: RsField>(stage: &Stage, position: usize) -> Vec<F> {
    (0..stage.seed_len)
        .map(|coordinate| {
            let mut row = Row {
                main: vec![F::ZERO; 4],
                aux: vec![F::ZERO; 3],
                mask: [F::ZERO; 2],
                seed: vec![F::ZERO; stage.seed_len],
            };
            row.seed[coordinate] = F::ONE;
            F::evaluate_at(&row.polynomial(), position)
        })
        .collect()
}

/// The rank of a matrix, by elimination.
fn rank<F: Field>(mut rows: Vec<Vec<F>>) -> usize {
    let mut rank = 0;
    for column in 0..rows.first().map_or(0, Vec::len) {
        let Some(pivot) = (rank..rows.len()).find(|&i| rows[i][column] != F::ZERO) else {
            continue;
        };
        rows.swap(rank, pivot);
        let inverse = rows[rank][column].inverse();
        let pivot_row: Vec<F> = rows[rank].iter().map(|&entry| entry * inverse).collect();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != rank {
                let factor = row[column];
                row.iter_mut().zip(&pivot_row).for_each(|(entry, &p)| *entry -= factor * p);
            }
        }
        rank += 1;
    }
    rank
}

/// The lemma "Reed–Solomon is zero-knowledge", with the mask regarded as
/// message: the symbols at `t` distinct positions are a surjective function of
/// the seed, hence uniform for a uniform seed, whatever the message and mask.
fn opened_symbols_are_uniform_over<F: RsField>() {
    for positions in [vec![0, 1, 7, 19], vec![3, 5, 11, 63], vec![0]] {
        let queries = positions.len();
        for extra in [0, 1] {
            let stage = stage(queries, extra);
            let matrix: Vec<Vec<F>> = positions.iter().map(|&p| symbol_of_seed(&stage, p)).collect();
            assert_eq!(rank(matrix), queries);
        }
    }
}

/// The lemma "residual entropy": with `t + 1` seed coordinates, given the
/// `t` opened columns of an oracle and the fold of its seeds, an unopened
/// column of `B ≥ 2` rows is uniform over a coset of dimension `B - 1`.
fn one_more_seed_coordinate_keeps_unopened_columns_unpredictable_over<F: RsField>() {
    let mut rng = Rng::from_seed([1; 32]);
    for (rows, opened, unopened) in
        [(2, vec![0, 1, 7, 19], 5), (4, vec![3, 5, 11], 0), (8, vec![2], 30), (9, vec![1, 4, 9], 2)]
    {
        let stage = stage(opened.len(), 1);
        let seed_len = stage.seed_len;
        // The unknowns are the seeds of all rows. The symbol of a row reads
        // the seed of that row only.
        let symbol = |row: usize, position: usize| {
            let mut functional = vec![F::ZERO; rows * seed_len];
            functional[row * seed_len..(row + 1) * seed_len]
                .copy_from_slice(&symbol_of_seed(&stage, position));
            functional
        };
        let mut view: Vec<Vec<F>> = (0..rows)
            .flat_map(|row| opened.iter().map(move |&position| (row, position)))
            .map(|(r, p)| symbol(r, p))
            .collect();
        // A coordinate of the folded seed combines that coordinate of every row.
        let weights: Vec<F> = rng.fields(rows);
        for coordinate in 0..seed_len {
            let mut functional = vec![F::ZERO; rows * seed_len];
            for (row, &weight) in weights.iter().enumerate() {
                functional[row * seed_len + coordinate] = weight;
            }
            view.push(functional);
        }
        // The folded seed determines the fold of the opened columns, so it
        // adds one dimension to them.
        let view_rank = rank(view.clone());
        assert_eq!(view_rank, rows * opened.len() + 1);
        view.extend((0..rows).map(|row| symbol(row, unopened)));
        assert_eq!(rank(view), view_rank + rows - 1);
    }
}

/// With `t` seed coordinates, the opened symbols of a row determine its seed
/// and with it every other symbol: such leaves would need salts.
fn without_the_extra_coordinate_unopened_symbols_are_determined_over<F: RsField>() {
    let opened = [0, 1, 7, 19];
    let stage = stage(opened.len(), 0);
    let mut matrix: Vec<Vec<F>> = opened.iter().map(|&p| symbol_of_seed(&stage, p)).collect();
    assert_eq!(rank(matrix.clone()), opened.len());
    matrix.push(symbol_of_seed(&stage, 5));
    assert_eq!(rank(matrix), opened.len());
}

#[test]
fn opened_symbols_are_uniform() {
    opened_symbols_are_uniform_over::<Gf2_128>();
    opened_symbols_are_uniform_over::<Gf2_256>();
}

#[test]
fn one_more_seed_coordinate_keeps_unopened_columns_unpredictable() {
    one_more_seed_coordinate_keeps_unopened_columns_unpredictable_over::<Gf2_128>();
    one_more_seed_coordinate_keeps_unopened_columns_unpredictable_over::<Gf2_256>();
}

#[test]
fn without_the_extra_coordinate_unopened_symbols_are_determined() {
    without_the_extra_coordinate_unopened_symbols_are_determined_over::<Gf2_128>();
    without_the_extra_coordinate_unopened_symbols_are_determined_over::<Gf2_256>();
}
