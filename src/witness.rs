//! What the prover knows about an oracle: the messages and the randomness of
//! its rows.

use crate::field::Field;
use crate::hash::Hasher;
use crate::mask::Mask;
use crate::oracle::Oracle;
use crate::parameters::{MASK_LEN, Stage};
use crate::reed_solomon::RsField;
use crate::rng::Rng;
use rayon::prelude::*;

/// The message and the randomness of a single codeword: a main block, an
/// auxiliary block, a mask and a seed. This is a witness of the relation on a
/// folded oracle, and the form of the mask and of the disclosed row of the
/// base case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row<F> {
    /// The main block.
    pub main: Vec<F>,
    /// The auxiliary block.
    pub aux: Vec<F>,
    /// The mask.
    pub mask: [F; MASK_LEN],
    /// The seed.
    pub seed: Vec<F>,
}

impl<F: Field> Row<F> {
    /// A row of the shape of the given stage with uniform entries.
    pub fn sample(stage: &Stage, rng: &mut Rng) -> Self {
        Self {
            main: rng.fields(stage.main_len),
            aux: rng.fields(stage.aux_len),
            mask: [rng.field(), rng.field()],
            seed: rng.fields(stage.seed_len),
        }
    }

    /// The row `self + other·scale`.
    pub fn add_scaled(&self, other: &Self, scale: F) -> Self {
        let combine =
            |a: &[F], b: &[F]| -> Vec<F> { a.iter().zip(b).map(|(&x, &y)| x + y * scale).collect() };
        Self {
            main: combine(&self.main, &other.main),
            aux: combine(&self.aux, &other.aux),
            mask: [self.mask[0] + other.mask[0] * scale, self.mask[1] + other.mask[1] * scale],
            seed: combine(&self.seed, &other.seed),
        }
    }

    /// The coefficients of the polynomial that the row encodes, from the
    /// constant term upwards: the seed, the mask, the main block and the
    /// auxiliary block.
    pub fn polynomial(&self) -> Vec<F> {
        [&self.seed[..], &self.mask, &self.main, &self.aux].concat()
    }

    /// The entries in the order in which the base case sends them: the main
    /// block, the auxiliary block, the mask and the seed.
    pub fn to_fields(&self) -> Vec<F> {
        [&self.main[..], &self.aux, &self.mask, &self.seed].concat()
    }

    /// The inverse of [`Row::to_fields`] for a row of the given stage.
    ///
    /// # Panics
    /// Panics unless there are as many fields as the code of the stage has
    /// dimensions.
    pub fn from_fields(stage: &Stage, fields: &[F]) -> Self {
        assert_eq!(fields.len(), stage.dimension());
        let (main, rest) = fields.split_at(stage.main_len);
        let (aux, rest) = rest.split_at(stage.aux_len);
        let (mask, seed) = rest.split_at(MASK_LEN);
        Self { main: main.to_vec(), aux: aux.to_vec(), mask: [mask[0], mask[1]], seed: seed.to_vec() }
    }
}

/// The messages and the randomness of the `2^k` rows of an interleaved
/// oracle. The three tables are stored row by row, as in
/// [`crate::polynomial`].
#[derive(Clone, Debug)]
pub struct Message<F> {
    /// The main message, of `2^k·ℓ` entries.
    pub main: Vec<F>,
    /// The auxiliary message, of `2^k·a` entries.
    pub aux: Vec<F>,
    /// The mask, which determines the masks of all rows.
    pub mask: Mask<F>,
    /// The seeds of the rows, `2^k·r` entries in total.
    pub seeds: Vec<F>,
}

impl<F: RsField> Message<F> {
    /// A message with the given main and auxiliary parts and with fresh
    /// uniform randomness.
    pub fn new(stage: &Stage, main: Vec<F>, aux: Vec<F>, rng: &mut Rng) -> Self {
        assert_eq!(main.len(), stage.rows() * stage.main_len);
        assert_eq!(aux.len(), stage.rows() * stage.aux_len);
        Self {
            main,
            aux,
            mask: Mask::sample(stage.folding, rng),
            seeds: rng.fields(stage.rows() * stage.seed_len),
        }
    }

    /// The codewords of the rows, followed by the codeword of `extra_row` if
    /// there is one.
    pub fn encode(&self, stage: &Stage, extra_row: Option<&Row<F>>) -> Vec<Vec<F>> {
        let rows = stage.rows();
        let masks = self.mask.rows();
        fn of_row<F>(table: &[F], rows: usize, row: usize) -> &[F] {
            let width = table.len() / rows;
            &table[row * width..(row + 1) * width]
        }
        (0..rows + usize::from(extra_row.is_some()))
            .into_par_iter()
            .map(|row| {
                let polynomial = if row < rows {
                    let mut polynomial = Vec::with_capacity(stage.dimension());
                    polynomial.extend_from_slice(of_row(&self.seeds, rows, row));
                    polynomial.extend_from_slice(&masks[row]);
                    polynomial.extend_from_slice(of_row(&self.main, rows, row));
                    polynomial.extend_from_slice(of_row(&self.aux, rows, row));
                    polynomial
                } else {
                    extra_row.unwrap().polynomial()
                };
                F::encode(&polynomial, stage.domain_size)
            })
            .collect()
    }
}

/// What the prover knows about a committed oracle: its message and
/// randomness, and its codewords.
pub struct Witness<F> {
    pub(crate) message: Message<F>,
    pub(crate) oracle: Oracle<F>,
    /// The mask of the base case: a uniform row whose codeword is committed
    /// together with the oracle of the last stage.
    pub(crate) base_case_mask: Option<Row<F>>,
}

impl<F: RsField> Witness<F> {
    /// Whether this is a witness for an oracle of the given stage.
    pub(crate) fn fits(&self, stage: &Stage, is_last: bool) -> bool {
        let rows = stage.rows();
        self.message.mask.folding() == stage.folding
            && self.message.main.len() == rows * stage.main_len
            && self.message.aux.len() == rows * stage.aux_len
            && self.message.seeds.len() == rows * stage.seed_len
            && self.base_case_mask.is_some() == is_last
            && self.oracle.shape() == (rows + usize::from(is_last), stage.domain_size)
    }

    /// Encode and commit to the oracle of a stage with the given message.
    pub(crate) fn commit<H: Hasher>(
        stage: &Stage,
        message: Message<F>,
        is_last: bool,
        rng: &mut Rng,
    ) -> Self {
        let base_case_mask = is_last.then(|| Row::sample(stage, rng));
        let rows = message.encode(stage, base_case_mask.as_ref());
        Self { oracle: Oracle::commit::<H>(rows), message, base_case_mask }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Gf2_128, Gf2_256};
    use crate::polynomial::{eq_table, fold_rows};

    fn stage() -> Stage {
        Stage {
            folding: 2,
            main_len: 4,
            aux_len: 3,
            mask_len: 2,
            seed_len: 5,
            pad_len: 0,
            domain_size: 40,
            queries: 4,
        }
    }

    fn row_layouts<F: Field>() {
        let mut rng = Rng::from_seed([60; 32]);
        let stage = stage();
        let row: Row<F> = Row::sample(&stage, &mut rng);
        assert_eq!(Row::from_fields(&stage, &row.to_fields()), row);
        let polynomial = row.polynomial();
        assert_eq!(polynomial.len(), stage.dimension());
        assert_eq!(polynomial[..stage.seed_len], row.seed);
        assert_eq!(polynomial[stage.mask_degree()..stage.main_degree()], row.mask);
        assert_eq!(polynomial[stage.main_degree()..stage.aux_degree()], row.main);
        assert_eq!(polynomial[stage.aux_degree()..], row.aux);
        let other = Row::sample(&stage, &mut rng);
        let scale: F = rng.field();
        let sum: Vec<F> =
            row.to_fields().iter().zip(other.to_fields()).map(|(&x, y)| x + y * scale).collect();
        assert_eq!(row.add_scaled(&other, scale).to_fields(), sum);
    }

    #[test]
    fn row_layouts_are_consistent() {
        row_layouts::<Gf2_128>();
        row_layouts::<Gf2_256>();
    }

    /// The lemma "folding and interleaving": the fold of the codewords is the
    /// codeword of the folded message, mask and seed.
    fn folding_commutes<F: RsField>() {
        let mut rng = Rng::from_seed([61; 32]);
        let stage = stage();
        let message: Message<F> = Message::new(&stage, rng.fields(16), rng.fields(12), &mut rng);
        let extra = Row::sample(&stage, &mut rng);
        let codewords = message.encode(&stage, Some(&extra));
        assert_eq!(codewords.len(), 5);
        assert_eq!(codewords[4], F::encode(&extra.polynomial(), stage.domain_size));

        let challenges: Vec<F> = rng.fields(stage.folding);
        let weights = eq_table(&challenges);
        let folded = Row {
            main: fold_rows(&message.main, &weights),
            aux: fold_rows(&message.aux, &weights),
            mask: message.mask.fold(&challenges),
            seed: fold_rows(&message.seeds, &weights),
        };
        let expected = F::encode(&folded.polynomial(), stage.domain_size);
        for position in 0..stage.domain_size {
            let column: Vec<F> = codewords[..4].iter().map(|codeword| codeword[position]).collect();
            assert_eq!(fold_rows(&column, &weights), [expected[position]]);
        }
    }

    #[test]
    fn folding_commutes_with_encoding() {
        folding_commutes::<Gf2_128>();
        folding_commutes::<Gf2_256>();
    }
}
