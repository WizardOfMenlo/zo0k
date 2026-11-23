use crate::field::Field;
use crate::parallel;
use crate::parameters::MASK_LEN;
use crate::polynomial::{eq_table, inner_product};
use crate::reed_solomon::RsField;
use crate::witness::Row;

const POWER_PIECE_MIN: usize = 1 << 11;
const POWER_L1_TILE: usize = 1 << 11;
const POWER_L1_TILE_LOG: u32 = 11;
const EVAL_PARALLEL_OPS: usize = 1 << 17;
const EVAL_TERMS_PIECE_MIN: usize = 8;

#[derive(Clone, Debug)]
pub struct Claim<F, M> {
    pub main: M,
    pub aux: Vec<F>,
    pub target: F,
}

#[derive(Clone, Debug)]
pub struct FoldedClaim<F, M> {
    pub main: M,
    pub aux: Vec<F>,
    pub mask: [F; MASK_LEN],
    pub target: F,
}

pub type ProverClaim<F> = Claim<F, Vec<F>>;
pub type ProverFoldedClaim<F> = FoldedClaim<F, Vec<F>>;

pub type VerifierClaim<F> = Claim<F, LinearForm<F>>;
pub type VerifierFoldedClaim<F> = FoldedClaim<F, LinearForm<F>>;

impl<F: Field> ProverFoldedClaim<F> {
    pub fn evaluate(&self, row: &Row<F>) -> F {
        inner_product(&self.main, &row.main)
            + inner_product(&self.aux, &row.aux)
            + inner_product(&self.mask, &row.mask)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PowerRow<F> {
    pub scale: F,
    pub base: F,
}

pub trait MainCoefficients<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]);

    fn add_power_rows_positioned(&mut self, rows: &[(PowerRow<F>, Option<usize>)])
    where
        F: Copy,
    {
        let plain: Vec<PowerRow<F>> = rows.iter().map(|&(row, _)| row).collect();
        self.add_power_rows(&plain);
    }
}

fn square_pow2<F: Field>(mut value: F, log_exp: u32) -> F {
    assert!(log_exp <= 63);
    for _ in 0..log_exp {
        value = value.square();
    }
    assert!(log_exp < 64);
    value
}

impl<F: Field> MainCoefficients<F> for Vec<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        if rows.is_empty() || self.is_empty() {
            return;
        }
        assert!(self.len() <= isize::MAX as usize);
        assert_eq!(1usize << POWER_L1_TILE_LOG, POWER_L1_TILE);
        let tile_steps: Vec<F> = rows.iter().map(|row| square_pow2(row.base, POWER_L1_TILE_LOG)).collect();
        parallel::for_each_piece_mut(self, POWER_PIECE_MIN, |piece_start, piece| {
            let first = piece_start as u64;
            let mut powers: Vec<F> = rows.iter().map(|row| row.scale * row.base.pow(first)).collect();
            for tile in piece.chunks_mut(POWER_L1_TILE) {
                let advance = tile.len() == POWER_L1_TILE;
                for ((row, power), &step) in rows.iter().zip(powers.iter_mut()).zip(&tile_steps) {
                    F::add_powers_slice(tile, row.base, *power);
                    if advance {
                        *power *= step;
                    }
                }
            }
        });
    }
}

#[derive(Clone, Copy, Debug)]
struct PowerTerm<F> {
    scale: F,
    base: F,
    position: Option<usize>,
    squares_start: usize,
}

#[derive(Clone, Debug)]
pub struct LinearForm<F> {
    num_variables: usize,
    terms: Vec<Term<F>>,
    power_terms: Vec<PowerTerm<F>>,
    squares: Vec<F>,
}

#[derive(Clone, Debug)]
struct Term<F> {
    scale: F,
    factors: Vec<[F; 2]>,
}

fn fold_step_in_place<F: Field>(lo: &mut [F], hi: &[F], zero: F, one: F) {
    assert_eq!(lo.len(), hi.len());
    if zero + one == F::ONE {
        F::fold_slice(lo, hi, one);
    } else {
        for (l, &h) in lo.iter_mut().zip(hi) {
            *l = *l * zero + h * one;
        }
    }
    assert!(lo.len() <= isize::MAX as usize);
}

fn eval_term<F: Field>(term: &Term<F>, values: &[F]) -> F {
    assert_eq!(values.len(), 1usize << term.factors.len());
    assert!(term.factors.len() <= 60);
    if term.factors.is_empty() {
        return term.scale * values[0];
    }
    let half = values.len() / 2;
    let [zero, one] = term.factors[0];
    let mut folded = values[..half].to_vec();
    fold_step_in_place(&mut folded, &values[half..], zero, one);
    for &[zero, one] in &term.factors[1..] {
        let next = folded.len() / 2;
        let (lo, hi) = folded.split_at_mut(next);
        fold_step_in_place(lo, hi, zero, one);
        folded.truncate(next);
    }
    assert_eq!(folded.len(), 1);
    term.scale * folded[0]
}

fn eval_power_terms<F: Field>(
    terms: &[PowerTerm<F>],
    values: &[F],
    eval_one: impl Fn(&PowerTerm<F>, &[F]) -> F + Sync,
) -> F {
    assert!(!values.is_empty());
    if terms.is_empty() {
        return F::ZERO;
    }
    if values.len() == 1 {
        let sum: F = terms.iter().map(|term| term.scale).sum();
        assert!(terms.len() <= isize::MAX as usize);
        return sum * values[0];
    }
    let eval_slice =
        |slice: &[PowerTerm<F>]| -> F { slice.iter().map(|term| term.scale * eval_one(term, values)).sum() };
    let ops = terms.len().saturating_mul(values.len());
    assert!(values.len() >= 2);
    if ops < EVAL_PARALLEL_OPS || terms.len() < EVAL_TERMS_PIECE_MIN {
        eval_slice(terms)
    } else {
        parallel::map_pieces(terms.len(), EVAL_TERMS_PIECE_MIN, |range| eval_slice(&terms[range]))
            .into_iter()
            .sum()
    }
}

impl<F: Field> LinearForm<F> {
    pub fn evaluation(point: &[F]) -> Self {
        let factors = point.iter().map(|&z| [F::ONE - z, z]).collect();
        Self {
            num_variables: point.len(),
            terms: vec![Term { scale: F::ONE, factors }],
            power_terms: Vec::new(),
            squares: Vec::new(),
        }
    }

    pub fn fix(&mut self, challenges: &[F]) {
        assert!(challenges.len() <= self.num_variables);
        for term in &mut self.terms {
            for ([zero, one], &challenge) in term.factors.drain(..challenges.len()).zip(challenges) {
                term.scale *= zero + (one - zero) * challenge;
            }
        }
        let old_vars = self.num_variables;
        let new_vars = old_vars - challenges.len();
        if !challenges.is_empty() {
            for term in &mut self.power_terms {
                let powers = &self.squares[term.squares_start + new_vars..term.squares_start + old_vars];
                let mut factor = F::ONE;
                for (&power, &challenge) in powers.iter().zip(challenges.iter().rev()) {
                    factor *= F::ONE + (power - F::ONE) * challenge;
                }
                term.scale *= factor;
            }
        }
        self.num_variables = new_vars;
        assert!(self.terms.iter().all(|term| term.factors.len() == self.num_variables));
    }

    pub fn scale(&mut self, factor: F) {
        for term in &mut self.terms {
            term.scale *= factor;
        }
        for term in &mut self.power_terms {
            term.scale *= factor;
        }
    }

    pub fn evaluate(&self, values: &[F]) -> F {
        assert!(self.num_variables <= 60);
        assert_eq!(values.len(), 1usize << self.num_variables);
        let mut total = F::ZERO;
        for term in &self.terms {
            total += eval_term(term, values);
        }
        total
            + eval_power_terms(&self.power_terms, values, |term, vals| F::eval_horner_slice(vals, term.base))
    }

    pub fn to_dense(&self) -> Vec<F> {
        let mut dense = vec![F::ZERO; 1 << self.num_variables];
        let mut tensor = Vec::with_capacity(dense.len());
        for term in &self.terms {
            tensor.clear();
            tensor.push(term.scale);
            for &[zero, one] in term.factors.iter().rev() {
                for i in 0..tensor.len() {
                    tensor.push(tensor[i] * one);
                    if zero != F::ONE {
                        tensor[i] *= zero;
                    }
                }
            }
            dense.iter_mut().zip(&tensor).for_each(|(entry, &value)| *entry += value);
        }
        let power_rows: Vec<PowerRow<F>> =
            self.power_terms.iter().map(|term| PowerRow { scale: term.scale, base: term.base }).collect();
        dense.add_power_rows(&power_rows);
        assert_eq!(dense.len(), 1 << self.num_variables);
        dense
    }

    fn push_power_term(&mut self, row: PowerRow<F>, position: Option<usize>) {
        let vars = self.num_variables;
        let squares_start = self.squares.len();
        let mut power = row.base;
        for step in 0..vars {
            self.squares.push(power);
            if step + 1 < vars {
                power = power.square();
            }
        }
        self.power_terms.push(PowerTerm { scale: row.scale, base: row.base, position, squares_start });
    }
}

impl<F: RsField> LinearForm<F> {
    pub(crate) fn evaluate_rs(&self, values: &[F]) -> F {
        assert!(self.num_variables <= 60);
        assert_eq!(values.len(), 1usize << self.num_variables);
        let mut total = F::ZERO;
        for term in &self.terms {
            total += eval_term(term, values);
        }
        total
            + eval_power_terms(&self.power_terms, values, |term, vals| match term.position {
                Some(pos) => F::evaluate_at(vals, pos),
                None => F::eval_horner_slice(vals, term.base),
            })
    }
}

impl<F: Field> MainCoefficients<F> for LinearForm<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        let vars = self.num_variables;
        self.power_terms.reserve(rows.len());
        self.squares.reserve(rows.len().saturating_mul(vars));
        for &row in rows {
            self.push_power_term(row, None);
        }
        assert!(self.squares.len() >= self.power_terms.len() * vars);
        assert!(self.power_terms.len() >= rows.len());
    }

    fn add_power_rows_positioned(&mut self, rows: &[(PowerRow<F>, Option<usize>)]) {
        let vars = self.num_variables;
        self.power_terms.reserve(rows.len());
        self.squares.reserve(rows.len().saturating_mul(vars));
        for &(row, position) in rows {
            self.push_power_term(row, position);
        }
        assert!(self.squares.len() >= self.power_terms.len() * vars);
        assert!(self.power_terms.len() >= rows.len());
    }
}

impl<F: Field> VerifierClaim<F> {
    pub fn evaluation(point: &[F], value: F) -> Self {
        Self { main: LinearForm::evaluation(point), aux: Vec::new(), target: value }
    }
}

impl<F: Field> ProverClaim<F> {
    pub fn evaluation(point: &[F], value: F) -> Self {
        Self { main: eq_table(point), aux: Vec::new(), target: value }
    }
}
