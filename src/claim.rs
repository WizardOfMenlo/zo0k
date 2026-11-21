//! The public linear claims that the protocol reduces.
//!
//! A [`Claim`] is the incoming claim of a stage, on the message of an
//! interleaved oracle. A sumcheck turns it into a [`FoldedClaim`], on the
//! message and mask of the folded oracle, and a code switch turns that into
//! the claim of the next stage.
//!
//! The coefficients on the main blocks are a vector `c` as long as the main
//! message. The prover holds it as a `Vec<F>`. The verifier holds it as a
//! [`LinearForm`], a succinct description that it only expands in the base
//! case, when the vector is short. The two representations implement
//! [`MainCoefficients`], so the code that updates a claim is shared.

use crate::field::Field;
use crate::parallel;
use crate::parameters::MASK_LEN;
use crate::polynomial::{eq_table, inner_product};
use crate::witness::Row;

/// The claim `target = <main, f> + <aux, v>` on the main message `f` and the
/// auxiliary message `v` of an interleaved oracle. `M` is the representation
/// of the coefficients on the main message.
#[derive(Clone, Debug)]
pub struct Claim<F, M> {
    /// The coefficients `c` on the main message.
    pub main: M,
    /// The coefficients `d` on the auxiliary message.
    pub aux: Vec<F>,
    /// The claimed value.
    pub target: F,
}

/// The claim `target = <main, f> + <aux, v> + <mask, s>` on the main block
/// `f`, the auxiliary block `v` and the mask `s` of a single row.
#[derive(Clone, Debug)]
pub struct FoldedClaim<F, M> {
    /// The coefficients on the main block.
    pub main: M,
    /// The coefficients on the auxiliary block.
    pub aux: Vec<F>,
    /// The coefficients on the mask.
    pub mask: [F; MASK_LEN],
    /// The claimed value.
    pub target: F,
}

/// A claim of the prover, with explicit coefficients.
pub type ProverClaim<F> = Claim<F, Vec<F>>;
/// A folded claim of the prover, with explicit coefficients.
pub type ProverFoldedClaim<F> = FoldedClaim<F, Vec<F>>;

/// A claim of the verifier, with succinct coefficients.
pub type VerifierClaim<F> = Claim<F, LinearForm<F>>;
/// A folded claim of the verifier, with succinct coefficients.
pub type VerifierFoldedClaim<F> = FoldedClaim<F, LinearForm<F>>;

impl<F: Field> ProverFoldedClaim<F> {
    /// The value of the linear form of the claim on a row.
    pub fn evaluate(&self, row: &Row<F>) -> F {
        inner_product(&self.main, &row.main)
            + inner_product(&self.aux, &row.aux)
            + inner_product(&self.mask, &row.mask)
    }
}

/// The vector `(scale·base^j)_j`: a row of a Reed–Solomon generator matrix,
/// or an out-of-domain sample.
#[derive(Clone, Copy, Debug)]
pub struct PowerRow<F> {
    /// The factor of every entry.
    pub scale: F,
    /// The base of the powers.
    pub base: F,
}

/// A representation of the coefficients on a main message.
pub trait MainCoefficients<F> {
    /// Add the given power rows to the coefficients.
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]);
}

impl<F: Field> MainCoefficients<F> for Vec<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        const BLOCK: usize = 1 << 10;
        parallel::for_each_chunk_mut(self, BLOCK, BLOCK, |block, entries| {
            // Each row advances by one power per entry of the block.
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

/// A vector of `2^m` field elements given as a sum of tensor products: entry
/// `j` is `Σ_terms scale·Π_i factors[i][j_i]`, where `j_1, …, j_m` are the
/// bits of `j` from the highest down.
///
/// This is the succinct linear form of the paper for the two kinds of rows
/// that occur: `eq(·, z)` has the factors `(1 - z_i, z_i)`, and the powers of
/// `x` have the factors `(1, x^(2^(m-i)))`.
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
    /// The form `eq(·, point)`, whose inner product with the evaluation table
    /// of a multilinear polynomial is the value of the polynomial at `point`.
    pub fn evaluation(point: &[F]) -> Self {
        let factors = point.iter().map(|&z| [F::ONE - z, z]).collect();
        Self { num_variables: point.len(), terms: vec![Term { scale: F::ONE, factors }] }
    }

    /// Fix the first variables to the given challenges, which leaves a form
    /// in the remaining variables: the fold of the vector, as in
    /// [`crate::polynomial::fold`].
    pub fn fix(&mut self, challenges: &[F]) {
        assert!(challenges.len() <= self.num_variables);
        for term in &mut self.terms {
            for ([zero, one], &challenge) in term.factors.drain(..challenges.len()).zip(challenges) {
                term.scale *= zero + (one - zero) * challenge;
            }
        }
        self.num_variables -= challenges.len();
    }

    /// Multiply the form by a scalar.
    pub fn scale(&mut self, factor: F) {
        for term in &mut self.terms {
            term.scale *= factor;
        }
    }

    /// The vector of the form.
    pub fn to_dense(&self) -> Vec<F> {
        let mut dense = vec![F::ZERO; 1 << self.num_variables];
        let mut tensor = Vec::with_capacity(dense.len());
        for term in &self.terms {
            tensor.clear();
            tensor.push(term.scale);
            // Each factor doubles the vector and becomes its highest bit.
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
            // The last variable is the lowest bit of the exponent.
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
    /// The claim that the multilinear polynomial with the committed table
    /// takes the value `value` at `point`. The first oracle has no auxiliary
    /// message.
    pub fn evaluation(point: &[F], value: F) -> Self {
        Self { main: LinearForm::evaluation(point), aux: Vec::new(), target: value }
    }
}

impl<F: Field> ProverClaim<F> {
    /// The prover's form of [`VerifierClaim::evaluation`].
    pub fn evaluation(point: &[F], value: F) -> Self {
        Self { main: eq_table(point), aux: Vec::new(), target: value }
    }
}
