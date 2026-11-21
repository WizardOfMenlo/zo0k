use crate::field::Field;
use crate::parameters::MASK_LEN;
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Mask<F> {
    coefficients: Vec<[F; MASK_LEN]>,
}

impl<F: Field> Mask<F> {
    pub fn sample(folding: usize, rng: &mut Rng) -> Self {
        Self { coefficients: (0..=folding).map(|_| [rng.field(), rng.field()]).collect() }
    }

    pub fn folding(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub fn rows(&self) -> Vec<[F; MASK_LEN]> {
        let k = self.folding();
        let mut suffix_sums = Vec::with_capacity(k + 1);
        let mut sum = [F::ZERO; MASK_LEN];
        for coefficient in self.coefficients.iter().rev() {
            sum = [sum[0] + coefficient[0], sum[1] + coefficient[1]];
            suffix_sums.push(sum);
        }
        (0..1usize << k).map(|row| suffix_sums[row.trailing_ones() as usize]).collect()
    }

    pub fn fold(&self, challenges: &[F]) -> [F; MASK_LEN] {
        assert_eq!(challenges.len(), self.folding());
        let mut value = self.coefficients[0];
        for (coefficient, &challenge) in self.coefficients[1..].iter().zip(challenges) {
            value = [value[0] * challenge + coefficient[0], value[1] * challenge + coefficient[1]];
        }
        value
    }

    pub fn sumcheck(&self) -> MaskSumcheck<F> {
        MaskSumcheck::new(self)
    }
}

pub fn linear_form<F: Field>(challenges: &[F]) -> [F; MASK_LEN] {
    [F::ONE, challenges.iter().sum()]
}

pub struct MaskSumcheck<F> {
    coefficients: Vec<[F; MASK_LEN]>,
    lambda: Vec<[F; MASK_LEN]>,
    nu: Vec<F>,
    round: usize,
    lambda_prime: F,
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

    pub fn sum(&self) -> F {
        self.lambda[0][0] + self.nu[0]
    }

    pub fn round_coefficients(&self) -> (F, F) {
        let j = self.round;
        let constant = self.lambda[j][0] + self.challenge_sum * self.lambda[j][1] + self.nu[j];
        (constant, self.lambda_prime)
    }

    pub fn bind(&mut self, challenge: F) {
        self.lambda_prime = challenge * self.lambda_prime + self.coefficients[self.round][1];
        self.challenge_sum += challenge;
        self.round += 1;
    }
}
