pub struct Prng {
    state: [u64; 4],
}

pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub fn mix(seed: u64, stream: u64) -> u64 {
    let mut state = seed ^ stream.wrapping_mul(0xd6e8_feb8_6659_fd93);
    let first = splitmix64(&mut state);
    let second = splitmix64(&mut state);
    first ^ second.rotate_left(17)
}

impl Prng {
    pub fn new(seed: u64) -> Self {
        let mut mixer = seed;
        let state =
            [splitmix64(&mut mixer), splitmix64(&mut mixer), splitmix64(&mut mixer), splitmix64(&mut mixer)];
        assert!(state.iter().any(|&word| word != 0));
        let prng = Self { state };
        assert_eq!(prng.state.len(), 4);
        prng
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    pub fn below(&mut self, bound: usize) -> usize {
        assert!(bound > 0);
        let draw = ((self.next_u64() as u128 * bound as u128) >> 64) as usize;
        assert!(draw < bound);
        draw
    }

    pub fn range(&mut self, low: usize, high: usize) -> usize {
        assert!(low <= high);
        let value = low + self.below(high - low + 1);
        assert!((low..=high).contains(&value));
        value
    }

    pub fn chance(&mut self, percent: usize) -> bool {
        assert!(percent <= 100);
        self.below(100) < percent
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        assert!(!items.is_empty());
        items[self.below(items.len())]
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            bytes.push(self.next_u64() as u8);
        }
        assert_eq!(bytes.len(), len);
        bytes
    }

    pub fn seed_bytes(&mut self) -> [u8; 32] {
        let mut seed = [0u8; 32];
        for word in 0..4 {
            seed[8 * word..8 * word + 8].copy_from_slice(&self.next_u64().to_le_bytes());
        }
        assert_eq!(seed.len(), 32);
        seed
    }

    pub fn subset(&mut self, len: usize, count_max: usize) -> Vec<usize> {
        assert!(len > 0 && count_max > 0);
        let count = self.range(1, count_max.min(len));
        let mut positions: Vec<usize> = (0..count).map(|_| self.below(len)).collect();
        positions.sort_unstable();
        positions.dedup();
        assert!(!positions.is_empty() && positions.iter().all(|&position| position < len));
        positions
    }
}
