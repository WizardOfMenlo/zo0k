use crate::field::Field;
use crate::hash::Hasher;
use crate::mask::Mask;
use crate::oracle::Oracle;
use crate::parallel;
use crate::parameters::{MASK_LEN, Stage};
use crate::reed_solomon::RsField;
use crate::rng::Rng;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row<F> {
    pub main: Vec<F>,
    pub aux: Vec<F>,
    pub mask: [F; MASK_LEN],
    pub seed: Vec<F>,
}

impl<F: Field> Row<F> {
    pub fn sample(stage: &Stage, rng: &mut Rng) -> Self {
        Self {
            main: rng.fields(stage.main_len),
            aux: rng.fields(stage.aux_len),
            mask: [rng.field(), rng.field()],
            seed: rng.fields(stage.seed_len),
        }
    }

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

    pub fn polynomial(&self) -> Vec<F> {
        [&self.seed[..], &self.mask, &self.main, &self.aux].concat()
    }

    pub fn to_fields(&self) -> Vec<F> {
        [&self.main[..], &self.aux, &self.mask, &self.seed].concat()
    }

    pub fn from_fields(stage: &Stage, fields: &[F]) -> Self {
        assert_eq!(fields.len(), stage.dimension());
        let (main, rest) = fields.split_at(stage.main_len);
        let (aux, rest) = rest.split_at(stage.aux_len);
        let (mask, seed) = rest.split_at(MASK_LEN);
        Self { main: main.to_vec(), aux: aux.to_vec(), mask: [mask[0], mask[1]], seed: seed.to_vec() }
    }
}

#[derive(Clone, Debug)]
pub struct Message<F> {
    pub main: Vec<F>,
    pub aux: Vec<F>,
    pub mask: Mask<F>,
    pub seeds: Vec<F>,
}

impl<F: RsField> Message<F> {
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

    pub fn encode(&self, stage: &Stage, extra_row: Option<&Row<F>>) -> Vec<Vec<F>> {
        let rows = stage.rows();
        let masks = self.mask.rows();
        fn of_row<F>(table: &[F], rows: usize, row: usize) -> &[F] {
            let width = table.len() / rows;
            &table[row * width..(row + 1) * width]
        }
        parallel::map_range(rows + usize::from(extra_row.is_some()), 1, |row| {
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
    }
}

pub struct Witness<F> {
    pub(crate) message: Message<F>,
    pub(crate) oracle: Oracle<F>,
    pub(crate) base_case_mask: Option<Row<F>>,
}

impl<F: RsField> Witness<F> {
    pub(crate) fn fits(&self, stage: &Stage, is_last: bool) -> bool {
        let rows = stage.rows();
        self.message.mask.folding() == stage.folding
            && self.message.main.len() == rows * stage.main_len
            && self.message.aux.len() == rows * stage.aux_len
            && self.message.seeds.len() == rows * stage.seed_len
            && self.base_case_mask.is_some() == is_last
            && self.oracle.shape() == (rows + usize::from(is_last), stage.domain_size)
    }

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
