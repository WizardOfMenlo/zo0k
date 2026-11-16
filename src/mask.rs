//! The mask polynomial, which hides the messages of the sumcheck.
//!
//! This is the paper's definition "mask polynomial" for the field and masking
//! degree three. A mask `σ` consists of `k + 1` pairs `σ_i = (σ_{i,0},
//! σ_{i,1})`. It defines two multilinear polynomials
//!
//! ```text
//! s_e(X_1, …, X_k) = Σ_{i=0}^{k} σ_{i,e}·X_{i+1}⋯X_k        for e = 0, 1,
//! ```
//!
//! and the mask polynomial
//!
//! ```text
//! U(X_1, …, X_k) = s_0 + s_1·(X_1 + ⋯ + X_k).
//! ```
//!
//! Row `b` of an oracle carries the pair `(s_0(b), s_1(b))` as the mask of its
//! encoding. Folding the rows at challenges `γ` folds these pairs to
//! `(s_0(γ), s_1(γ))`, so `U(γ)` is the inner product of the folded mask with
//! the linear form `(1, γ_1 + ⋯ + γ_k)`.

use crate::field::Field;
use crate::parameters::MASK_LEN;
use crate::rng::Rng;

/// The coefficients `σ` of a mask polynomial in `k` variables.
#[derive(Clone, Debug)]
pub struct Mask<F> {
    /// `coefficients[i]` is the pair `σ_i`, for `i = 0, …, k`.
    coefficients: Vec<[F; MASK_LEN]>,
}

impl<F: Field> Mask<F> {
    /// A uniform mask for folding parameter `folding`.
    pub fn sample(folding: usize, rng: &mut Rng) -> Self {
        Self { coefficients: (0..=folding).map(|_| [rng.field(), rng.field()]).collect() }
    }

    /// The folding parameter `k`.
    pub fn folding(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// The masks `(s_0(b), s_1(b))` of the `2^k` rows, where the row index
    /// `b` has the value of `X_1` in its highest bit.
    pub fn rows(&self) -> Vec<[F; MASK_LEN]> {
        let k = self.folding();
        // The monomial X_{i+1}⋯X_k is one on the rows whose k - i lowest
        // bits are set, so a row whose `ones` lowest bits are set, and no
        // more, has the value σ_{k-ones} + ⋯ + σ_k.
        let mut suffix_sums = Vec::with_capacity(k + 1);
        let mut sum = [F::ZERO; MASK_LEN];
        for coefficient in self.coefficients.iter().rev() {
            sum = [sum[0] + coefficient[0], sum[1] + coefficient[1]];
            suffix_sums.push(sum);
        }
        (0..1usize << k).map(|row| suffix_sums[row.trailing_ones() as usize]).collect()
    }

    /// The fold `(s_0(γ), s_1(γ))` of the masks of the rows at challenges `γ`.
    pub fn fold(&self, challenges: &[F]) -> [F; MASK_LEN] {
        assert_eq!(challenges.len(), self.folding());
        // Horner's rule for Σ_i σ_i·γ_{i+1}⋯γ_k.
        let mut value = self.coefficients[0];
        for (coefficient, &challenge) in self.coefficients[1..].iter().zip(challenges) {
            value = [value[0] * challenge + coefficient[0], value[1] * challenge + coefficient[1]];
        }
        value
    }

    /// The sumcheck messages of the mask polynomial.
    pub fn sumcheck(&self) -> MaskSumcheck<F> {
        MaskSumcheck::new(self)
    }
}

/// The linear form of the mask polynomial: `U(γ)` is the inner product of
/// `(1, γ_1 + ⋯ + γ_k)` with the folded mask.
pub fn linear_form<F: Field>(challenges: &[F]) -> [F; MASK_LEN] {
    [F::ONE, challenges.iter().sum()]
}

/// The prover's sumcheck for a mask polynomial alone.
///
/// For a polynomial `P = Σ_i a_i·X_{i+1}⋯X_k` and a round `j` with previous
/// challenges `γ_1, …, γ_{j-1}`, the paper defines
///
/// ```text
/// Σ_b     P(γ, X, b) = λ_j + λ'_j·X        Σ_b |b|·P(γ, X, b) = ν_j + ν'_j·X
/// ```
///
/// with sums over `b` in `{0,1}^(k-j)` and `|b|` the number of ones of `b`.
/// The values `λ_j` and `ν_j` do not depend on the challenges and are computed
/// up front by the recurrences
///
/// ```text
/// λ_k = a_k,   λ_j = a_j + 2·λ_{j+1},      ν_k = 0,   ν_j = (k - j)·a_j + 2·ν_{j+1} + λ_{j+1},
/// ```
///
/// which also hold for `j = 0`, the sums over the whole cube. The value `λ'_j`
/// follows the challenges: `λ'_1 = a_0` and `λ'_{j+1} = γ_j·λ'_j + a_j`.
pub struct MaskSumcheck<F> {
    /// The coefficients of `s_0` and `s_1`.
    coefficients: Vec<[F; MASK_LEN]>,
    /// `lambda[j]` is the pair `(λ_j(s_0), λ_j(s_1))`, for `j = 0, …, k`.
    lambda: Vec<[F; MASK_LEN]>,
    /// `nu[j]` is `ν_j(s_1)`, for `j = 0, …, k`.
    nu: Vec<F>,
    /// The current round `j`, from one.
    round: usize,
    /// `λ'_j(s_1)` for the current round.
    lambda_prime: F,
    /// `γ_1 + ⋯ + γ_{j-1}` for the current round.
    challenge_sum: F,
}

impl<F: Field> MaskSumcheck<F> {
    fn new(mask: &Mask<F>) -> Self {
        let k = mask.folding();
        let a = &mask.coefficients;
        let two = F::from_integer(2);
        let mut lambda = vec![a[k]; k + 1];
        let mut nu = vec![F::ZERO; k + 1];
        for j in (0..k).rev() {
            lambda[j] = [a[j][0] + two * lambda[j + 1][0], a[j][1] + two * lambda[j + 1][1]];
            nu[j] = F::from_integer(k - j) * a[j][1] + two * nu[j + 1] + lambda[j + 1][1];
        }
        Self { lambda_prime: a[0][1], coefficients: a.clone(), lambda, nu, round: 1, challenge_sum: F::ZERO }
    }

    /// The sum of the mask polynomial over the Boolean cube.
    pub fn sum(&self) -> F {
        self.lambda[0][0] + self.nu[0]
    }

    /// The constant and the quadratic coefficient of the polynomial of the
    /// current round, `Σ_b U(γ_1, …, γ_{j-1}, X, b)`.
    ///
    /// On the points of this sum, `U = s_0 + s_1·(γ_1 + ⋯ + γ_{j-1} + X + |b|)`.
    pub fn round_coefficients(&self) -> (F, F) {
        let j = self.round;
        let constant = self.lambda[j][0] + self.challenge_sum * self.lambda[j][1] + self.nu[j];
        (constant, self.lambda_prime)
    }

    /// Move to the next round with the challenge of the current one.
    pub fn bind(&mut self, challenge: F) {
        self.lambda_prime = challenge * self.lambda_prime + self.coefficients[self.round][1];
        self.challenge_sum += challenge;
        self.round += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Gf2_128, Gf2_256};
    use crate::polynomial::{eq_table, fold_rows, inner_product};

    fn bits<F: Field>(index: usize, count: usize) -> Vec<F> {
        (0..count).rev().map(|i| F::from_integer(index >> i)).collect()
    }

    /// The definition of the polynomials s_e, monomial by monomial.
    fn s<F: Field>(mask: &Mask<F>, point: &[F]) -> [F; MASK_LEN] {
        let mut value = [F::ZERO; MASK_LEN];
        for (i, coefficient) in mask.coefficients.iter().enumerate() {
            let monomial = point[i..].iter().fold(F::ONE, |product, &x| product * x);
            value = [value[0] + coefficient[0] * monomial, value[1] + coefficient[1] * monomial];
        }
        value
    }

    /// The definition of the mask polynomial.
    fn u<F: Field>(mask: &Mask<F>, point: &[F]) -> F {
        let [s0, s1] = s(mask, point);
        s0 + s1 * point.iter().sum::<F>()
    }

    fn rows_are_the_values_on_the_cube_and_fold_to_the_value_at_the_challenges_over<F: Field>() {
        let mut rng = Rng::from_seed([50; 32]);
        for k in 0..=6 {
            let mask: Mask<F> = Mask::sample(k, &mut rng);
            let rows = mask.rows();
            assert_eq!(rows.len(), 1 << k);
            for (row, value) in rows.iter().enumerate() {
                assert_eq!(*value, s(&mask, &bits(row, k)), "k {k}, row {row}");
            }
            // Item 1 of the lemma "mask polynomial".
            let challenges: Vec<F> = rng.fields(k);
            let weights = eq_table(&challenges);
            let folded = mask.fold(&challenges);
            for e in 0..MASK_LEN {
                let column: Vec<F> = rows.iter().map(|row| row[e]).collect();
                assert_eq!(fold_rows(&column, &weights), [folded[e]]);
            }
            assert_eq!(folded, s(&mask, &challenges));
            assert_eq!(inner_product(&linear_form(&challenges), &folded), u(&mask, &challenges));
        }
    }

    fn sumcheck_messages_match_sums_over_the_cube_over<F: Field>() {
        let mut rng = Rng::from_seed([51; 32]);
        for k in 1..=6 {
            let mask: Mask<F> = Mask::sample(k, &mut rng);
            let mut sumcheck = mask.sumcheck();
            let total: F = (0..1 << k).map(|b| u(&mask, &bits::<F>(b, k))).sum();
            assert_eq!(sumcheck.sum(), total);
            let mut prefix = Vec::new();
            for round in 1..=k {
                let partial = |x: F| -> F {
                    (0..1usize << (k - round))
                        .map(|b| {
                            let mut point = prefix.clone();
                            point.push(x);
                            point.extend(bits::<F>(b, k - round));
                            u(&mask, &point)
                        })
                        .sum()
                };
                // A polynomial of degree two is determined by three values.
                let (constant, quadratic) = sumcheck.round_coefficients();
                let x: F = rng.field();
                let linear = partial(F::ONE) - constant - quadratic;
                assert_eq!(partial(F::ZERO), constant, "k {k}, round {round}");
                assert_eq!(partial(x), constant + linear * x + quadratic * x * x, "k {k}, round {round}");
                let challenge: F = rng.field();
                sumcheck.bind(challenge);
                prefix.push(challenge);
            }
        }
    }

    /// Item 2 of the lemma: the map from a mask to the free entries of the
    /// transcript (the sum, and the constant and quadratic coefficients of
    /// every round) is onto, so a uniform mask gives a uniform transcript.
    fn the_transcript_map_is_onto_over<F: Field>() {
        let mut rng = Rng::from_seed([52; 32]);
        for k in 1..=6 {
            let challenge_sets: [Vec<F>; 3] = [vec![F::ZERO; k], vec![F::ONE; k], rng.fields(k)];
            for challenges in &challenge_sets {
                // One row of the matrix per coordinate of the mask.
                let mut matrix = Vec::new();
                for coordinate in 0..MASK_LEN * (k + 1) {
                    let mut mask = Mask { coefficients: vec![[F::ZERO; MASK_LEN]; k + 1] };
                    mask.coefficients[coordinate / MASK_LEN][coordinate % MASK_LEN] = F::ONE;
                    let mut sumcheck = mask.sumcheck();
                    let mut image = vec![sumcheck.sum()];
                    for &challenge in challenges {
                        let (constant, quadratic) = sumcheck.round_coefficients();
                        image.extend([constant, quadratic]);
                        sumcheck.bind(challenge);
                    }
                    matrix.push(image);
                }
                assert_eq!(rank(matrix), 1 + 2 * k, "k {k}, challenges {challenges:?}");
            }
        }
    }

    fn rank<F: Field>(mut matrix: Vec<Vec<F>>) -> usize {
        let mut rank = 0;
        for column in 0..matrix[0].len() {
            let Some(pivot) = (rank..matrix.len()).find(|&i| matrix[i][column] != F::ZERO) else {
                continue;
            };
            matrix.swap(rank, pivot);
            let inverse = matrix[rank][column].inverse();
            let pivot_row: Vec<F> = matrix[rank].iter().map(|&entry| entry * inverse).collect();
            for row in matrix.iter_mut().skip(rank + 1) {
                let factor = row[column];
                row.iter_mut().zip(&pivot_row).for_each(|(entry, &p)| *entry -= factor * p);
            }
            rank += 1;
        }
        rank
    }

    #[test]
    fn rows_are_the_values_on_the_cube_and_fold_to_the_value_at_the_challenges() {
        rows_are_the_values_on_the_cube_and_fold_to_the_value_at_the_challenges_over::<Gf2_128>();
        rows_are_the_values_on_the_cube_and_fold_to_the_value_at_the_challenges_over::<Gf2_256>();
    }

    #[test]
    fn sumcheck_messages_match_sums_over_the_cube() {
        sumcheck_messages_match_sums_over_the_cube_over::<Gf2_128>();
        sumcheck_messages_match_sums_over_the_cube_over::<Gf2_256>();
    }

    #[test]
    fn the_transcript_map_is_onto() {
        the_transcript_map_is_onto_over::<Gf2_128>();
        the_transcript_map_is_onto_over::<Gf2_256>();
    }
}
