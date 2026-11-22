use crate::field::Field;
use crate::parallel;
use crate::parameters::MASK_LEN;
use crate::polynomial::{eq_table, inner_product};
use crate::witness::Row;

const POWER_PIECE_MIN: usize = 1 << 12;
const POWER_L1_TILE: usize = 1 << 10;
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
}

fn add_power_tile_single<F: Field>(tile: &mut [F], base: F, mut power: F) -> F {
    assert!(!tile.is_empty());
    let (chunks, remainder) = tile.as_chunks_mut::<4>();
    if !chunks.is_empty() {
        let b2 = base.square();
        let b4 = b2.square();
        let mut p0 = power;
        let mut p1 = p0 * base;
        let mut p2 = p0 * b2;
        let mut p3 = p1 * b2;
        for chunk in chunks {
            chunk[0] += p0;
            chunk[1] += p1;
            chunk[2] += p2;
            chunk[3] += p3;
            p0 *= b4;
            p1 *= b4;
            p2 *= b4;
            p3 *= b4;
        }
        power = p0;
    }
    for entry in remainder {
        *entry += power;
        power *= base;
    }
    assert!(tile.len() <= isize::MAX as usize);
    power
}

impl<F: Field> MainCoefficients<F> for Vec<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        if rows.is_empty() || self.is_empty() {
            return;
        }
        assert!(self.len() <= isize::MAX as usize);
        assert!(!rows.is_empty());
        parallel::for_each_piece_mut(self, POWER_PIECE_MIN, |piece_start, piece| {
            let first = piece_start as u64;
            let mut powers: Vec<F> = rows.iter().map(|row| row.scale * row.base.pow(first)).collect();
            for tile in piece.chunks_mut(POWER_L1_TILE) {
                for (row, power) in rows.iter().zip(powers.iter_mut()) {
                    *power = add_power_tile_single(tile, row.base, *power);
                }
            }
        });
    }
}

#[derive(Clone, Copy, Debug)]
struct PowerTerm<F> {
    scale: F,
    base: F,
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

fn eval_horner_single<F: Field>(values: &[F], x: F) -> F {
    assert!(!values.is_empty());
    let (chunks, remainder) = values.as_chunks::<4>();
    let mut a0 = remainder.iter().rev().fold(F::ZERO, |acc, &c| acc * x + c);
    if chunks.is_empty() {
        assert_eq!(remainder.len(), values.len());
        return a0;
    }
    let x2 = x.square();
    let x3 = x2 * x;
    let x4 = x2.square();
    let (mut a1, mut a2, mut a3) = (F::ZERO, F::ZERO, F::ZERO);
    for chunk in chunks.iter().rev() {
        a0 = a0 * x4 + chunk[0];
        a1 = a1 * x4 + chunk[1];
        a2 = a2 * x4 + chunk[2];
        a3 = a3 * x4 + chunk[3];
    }
    assert!(remainder.len() < 4);
    a0 + x * a1 + x2 * a2 + x3 * a3
}

fn fold_step_in_place<F: Field>(lo: &mut [F], hi: &[F], zero: F, one: F) {
    assert_eq!(lo.len(), hi.len());
    if zero + one == F::ONE {
        let (lc, lr) = lo.as_chunks_mut::<4>();
        let (hc, hr) = hi.as_chunks::<4>();
        for (l, h) in lc.iter_mut().zip(hc.iter()) {
            l[0] += (h[0] - l[0]) * one;
            l[1] += (h[1] - l[1]) * one;
            l[2] += (h[2] - l[2]) * one;
            l[3] += (h[3] - l[3]) * one;
        }
        for (l, &h) in lr.iter_mut().zip(hr) {
            *l += (h - *l) * one;
        }
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

fn eval_power_terms<F: Field>(terms: &[PowerTerm<F>], values: &[F]) -> F {
    assert!(!values.is_empty());
    if terms.is_empty() {
        return F::ZERO;
    }
    if values.len() == 1 {
        let sum: F = terms.iter().map(|term| term.scale).sum();
        assert!(terms.len() <= isize::MAX as usize);
        return sum * values[0];
    }
    let eval_slice = |slice: &[PowerTerm<F>]| -> F {
        slice.iter().map(|term| term.scale * eval_horner_single(values, term.base)).sum()
    };
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
        total + eval_power_terms(&self.power_terms, values)
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
}

impl<F: Field> MainCoefficients<F> for LinearForm<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        let vars = self.num_variables;
        self.power_terms.reserve(rows.len());
        self.squares.reserve(rows.len().saturating_mul(vars));
        for row in rows {
            let squares_start = self.squares.len();
            let mut power = row.base;
            for step in 0..vars {
                self.squares.push(power);
                if step + 1 < vars {
                    power = power.square();
                }
            }
            self.power_terms.push(PowerTerm { scale: row.scale, base: row.base, squares_start });
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
