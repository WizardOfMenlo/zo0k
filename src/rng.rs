//! The prover's source of randomness.

use crate::field::Field;

/// A cryptographic random generator: the BLAKE3 output stream of a seed.
pub struct Rng(crate::blake3::OutputReader);

impl Rng {
    /// A generator seeded by the operating system.
    ///
    /// # Panics
    /// Panics if the operating system cannot supply entropy.
    pub fn from_entropy() -> Self {
        let mut seed = [0; 32];
        assert!(crate::entropy::fill(&mut seed), "the operating system supplies entropy");
        assert!(seed != [0; 32]);
        Self::from_seed(seed)
    }

    /// A deterministic generator, for tests and reproducible benchmarks.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self(crate::blake3::Hasher::new_keyed(&seed).finalize_xof())
    }

    /// A uniform field element.
    pub fn field<F: Field>(&mut self) -> F {
        F::sample(|bytes| self.0.fill(bytes))
    }

    /// A uniform integer below `bound`.
    pub fn below(&mut self, bound: usize) -> usize {
        assert!(bound > 0);
        let mask = (bound as u64).next_power_of_two() - 1;
        loop {
            let mut bytes = [0; 8];
            self.0.fill(&mut bytes);
            let draw = u64::from_le_bytes(bytes) & mask;
            if draw < bound as u64 {
                return draw as usize;
            }
        }
    }

    /// Independent uniform field elements.
    pub fn fields<F: Field>(&mut self, count: usize) -> Vec<F> {
        (0..count).map(|_| self.field()).collect()
    }
}
