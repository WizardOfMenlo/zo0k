use crate::field::Field;
use crate::parallel;
use crate::parameters::MASK_LEN;
use crate::polynomial::{eq_table, inner_product};
use crate::witness::Row;

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
}

impl<F: Field> MainCoefficients<F> for Vec<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        const BLOCK: usize = 1 << 10;
        parallel::for_each_chunk_mut(self, BLOCK, BLOCK, |block, entries| {
            let first = (block * BLOCK) as u64;
            let mut powers: Vec<F> = rows.iter().map(|row| row.scale * row.base.pow(first)).collect();
            for entry in entries {
                for (power, row) in powers.iter_mut().zip(rows) {
                    *entry += *power;
                    *power *= row.base;
                }
            }
        });
    }
}

#[derive(Clone, Debug)]
pub struct LinearForm<F> {
    num_variables: usize,
    terms: Vec<Term<F>>,
}

#[derive(Clone, Debug)]
struct Term<F> {
    scale: F,
    factors: Vec<[F; 2]>,
}

impl<F: Field> LinearForm<F> {
    pub fn evaluation(point: &[F]) -> Self {
        let factors = point.iter().map(|&z| [F::ONE - z, z]).collect();
        Self { num_variables: point.len(), terms: vec![Term { scale: F::ONE, factors }] }
    }

    pub fn fix(&mut self, challenges: &[F]) {
        assert!(challenges.len() <= self.num_variables);
        for term in &mut self.terms {
            for ([zero, one], &challenge) in term.factors.drain(..challenges.len()).zip(challenges) {
                term.scale *= zero + (one - zero) * challenge;
            }
        }
        self.num_variables -= challenges.len();
    }

    pub fn scale(&mut self, factor: F) {
        for term in &mut self.terms {
            term.scale *= factor;
        }
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
                    tensor[i] *= zero;
                }
            }
            dense.iter_mut().zip(&tensor).for_each(|(entry, &value)| *entry += value);
        }
        dense
    }
}

impl<F: Field> MainCoefficients<F> for LinearForm<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        for row in rows {
            let mut power = row.base;
            let mut factors = vec![[F::ONE; 2]; self.num_variables];
            for factor in factors.iter_mut().rev() {
                factor[1] = power;
                power = power.square();
            }
            self.terms.push(Term { scale: row.scale, factors });
        }
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
