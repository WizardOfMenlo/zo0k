use crate::field::Field;

pub struct Rng(crate::blake3::OutputReader);

impl Rng {
    pub fn from_entropy() -> Self {
        let mut seed = [0; 32];
        assert!(crate::entropy::fill(&mut seed), "the operating system supplies entropy");
        assert!(seed != [0; 32]);
        Self::from_seed(seed)
    }

    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self(crate::blake3::Hasher::new_keyed(&seed).finalize_xof())
    }

    pub fn field<F: Field>(&mut self) -> F {
        F::sample(|bytes| self.0.fill(bytes))
    }

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

    pub fn fields<F: Field>(&mut self, count: usize) -> Vec<F> {
        (0..count).map(|_| self.field()).collect()
    }
}
