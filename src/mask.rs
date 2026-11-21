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
