#![allow(missing_docs)]

pub const KEY_LEN: usize = 32;
pub const OUT_LEN: usize = 32;
pub const BLOCK_LEN: usize = 64;
pub const CHUNK_LEN: usize = 1024;

const CV_STACK_LEN_MAX: usize = 54;
const CV_STACK_INLINE_LEN_MAX: usize = 8;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const FLAGS_ALL: u32 = CHUNK_START | CHUNK_END | PARENT | ROOT | KEYED_HASH;

const IV: [u32; 8] =
    [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

const ROUNDS: usize = 7;

const MSG_SCHEDULE: [[usize; 16]; ROUNDS] = message_schedule();

const fn message_schedule() -> [[usize; 16]; ROUNDS] {
    let mut schedule = [[0; 16]; ROUNDS];
    let mut word = 0;
    while word < 16 {
        schedule[0][word] = word;
        word += 1;
    }
    let mut round = 1;
    while round < ROUNDS {
        let mut word = 0;
        while word < 16 {
            schedule[round][word] = schedule[round - 1][MSG_PERMUTATION[word]];
            word += 1;
        }
        round += 1;
    }
    schedule
}

#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline(always)]
fn round(state: &mut [u32; 16], message: &[u32; 16], schedule: &[usize; 16]) {
    g(state, 0, 4, 8, 12, message[schedule[0]], message[schedule[1]]);
    g(state, 1, 5, 9, 13, message[schedule[2]], message[schedule[3]]);
    g(state, 2, 6, 10, 14, message[schedule[4]], message[schedule[5]]);
    g(state, 3, 7, 11, 15, message[schedule[6]], message[schedule[7]]);
    g(state, 0, 5, 10, 15, message[schedule[8]], message[schedule[9]]);
    g(state, 1, 6, 11, 12, message[schedule[10]], message[schedule[11]]);
    g(state, 2, 7, 8, 13, message[schedule[12]], message[schedule[13]]);
    g(state, 3, 4, 9, 14, message[schedule[14]], message[schedule[15]]);
}

#[inline(always)]
fn compress(cv: &[u32; 8], message: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
    assert!(block_len as usize <= BLOCK_LEN);
    assert!(flags & !FLAGS_ALL == 0);
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    round(&mut state, message, &MSG_SCHEDULE[0]);
    round(&mut state, message, &MSG_SCHEDULE[1]);
    round(&mut state, message, &MSG_SCHEDULE[2]);
    round(&mut state, message, &MSG_SCHEDULE[3]);
    round(&mut state, message, &MSG_SCHEDULE[4]);
    round(&mut state, message, &MSG_SCHEDULE[5]);
    round(&mut state, message, &MSG_SCHEDULE[6]);
    for word in 0..8 {
        state[word] ^= state[word + 8];
        state[word + 8] ^= cv[word];
    }
    state
}

#[inline(always)]
fn compress_cv(cv: &[u32; 8], message: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 8] {
    let state = compress(cv, message, counter, block_len, flags);
    let mut cv_next = [0; 8];
    cv_next.copy_from_slice(&state[..8]);
    cv_next
}

#[inline(always)]
fn words_from_block(block: &[u8; BLOCK_LEN]) -> [u32; 16] {
    let mut words = [0; 16];
    for (word, bytes) in words.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    words
}

fn words_from_key(key: &[u8; KEY_LEN]) -> [u32; 8] {
    let mut words = [0; 8];
    for (word, bytes) in words.iter_mut().zip(key.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    const { assert!(KEY_LEN == 32) };
    words
}

#[derive(Clone, Copy)]
struct Output {
    cv_input: [u32; 8],
    message: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        assert!(self.flags & ROOT == 0);
        compress_cv(&self.cv_input, &self.message, self.counter, self.block_len, self.flags)
    }

    fn root_block(&self, output_block_counter: u64) -> [u8; BLOCK_LEN] {
        assert!(self.flags & ROOT == 0);
        let state =
            compress(&self.cv_input, &self.message, output_block_counter, self.block_len, self.flags | ROOT);
        let mut block = [0; BLOCK_LEN];
        for (bytes, word) in block.as_chunks_mut::<4>().0.iter_mut().zip(state) {
            *bytes = word.to_le_bytes();
        }
        block
    }

    fn root_hash(&self) -> [u8; OUT_LEN] {
        assert!(self.flags & ROOT == 0);
        let cv = compress_cv(&self.cv_input, &self.message, 0, self.block_len, self.flags | ROOT);
        let mut hash = [0; OUT_LEN];
        for (bytes, word) in hash.as_chunks_mut::<4>().0.iter_mut().zip(cv) {
            *bytes = word.to_le_bytes();
        }
        hash
    }
}

fn parent_output(left: &[u32; 8], right: &[u32; 8], key: &[u32; 8], flags: u32) -> Output {
    assert!(flags & !KEYED_HASH == 0);
    let mut message = [0; 16];
    message[..8].copy_from_slice(left);
    message[8..].copy_from_slice(right);
    Output { cv_input: *key, message, counter: 0, block_len: BLOCK_LEN as u32, flags: flags | PARENT }
}

#[derive(Clone)]
struct ChunkState {
    cv: [u32; 8],
    chunk_counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: usize,
    blocks_compressed: usize,
    flags: u32,
}

impl ChunkState {
    fn new(key: &[u32; 8], chunk_counter: u64, flags: u32) -> Self {
        assert!(flags & !KEYED_HASH == 0);
        Self { cv: *key, chunk_counter, block: [0; BLOCK_LEN], block_len: 0, blocks_compressed: 0, flags }
    }

    fn len(&self) -> usize {
        let len = self.blocks_compressed * BLOCK_LEN + self.block_len;
        assert!(len <= CHUNK_LEN);
        len
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
    }

    #[inline(always)]
    fn compress_block(&mut self, block: &[u8; BLOCK_LEN]) {
        assert!(self.blocks_compressed < CHUNK_LEN / BLOCK_LEN - 1);
        let message = words_from_block(block);
        let flags = self.flags | self.start_flag();
        self.cv = compress_cv(&self.cv, &message, self.chunk_counter, BLOCK_LEN as u32, flags);
        self.blocks_compressed += 1;
    }

    fn update(&mut self, mut input: &[u8]) {
        assert!(self.len() + input.len() <= CHUNK_LEN);
        if self.block_len > 0 {
            let take = (BLOCK_LEN - self.block_len).min(input.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&input[..take]);
            self.block_len += take;
            input = &input[take..];
            if input.is_empty() {
                return;
            }
            let block = self.block;
            self.compress_block(&block);
            self.block_len = 0;
        }
        while input.len() > BLOCK_LEN {
            let (block, rest) = input.split_first_chunk::<BLOCK_LEN>().expect("a block remains");
            self.compress_block(block);
            input = rest;
        }
        if !input.is_empty() {
            assert!(self.block_len == 0);
            self.block = [0; BLOCK_LEN];
            self.block[..input.len()].copy_from_slice(input);
            self.block_len = input.len();
        }
    }

    fn output(&self) -> Output {
        assert!(self.block_len <= BLOCK_LEN);
        Output {
            cv_input: self.cv,
            message: words_from_block(&self.block),
            counter: self.chunk_counter,
            block_len: self.block_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

#[derive(Clone)]
pub struct Hasher {
    key: [u32; 8],
    flags: u32,
    chunk: ChunkState,
    cv_stack_inline: [[u32; 8]; CV_STACK_INLINE_LEN_MAX],
    cv_stack_spill: Vec<[u32; 8]>,
    cv_stack_len: usize,
}

impl Hasher {
    fn with_key_words(key: [u32; 8], flags: u32) -> Self {
        assert!(flags & !KEYED_HASH == 0);
        let chunk = ChunkState::new(&key, 0, flags);
        let cv_stack_inline = [[0; 8]; CV_STACK_INLINE_LEN_MAX];
        Self { key, flags, chunk, cv_stack_inline, cv_stack_spill: Vec::new(), cv_stack_len: 0 }
    }

    pub fn new() -> Self {
        Self::with_key_words(IV, 0)
    }

    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        Self::with_key_words(words_from_key(key), KEYED_HASH)
    }

    fn cv_stack_get(&self, index: usize) -> [u32; 8] {
        assert!(index < self.cv_stack_len);
        if index < CV_STACK_INLINE_LEN_MAX {
            self.cv_stack_inline[index]
        } else {
            self.cv_stack_spill[index - CV_STACK_INLINE_LEN_MAX]
        }
    }

    fn cv_stack_push(&mut self, cv: [u32; 8]) {
        assert!(self.cv_stack_len < CV_STACK_LEN_MAX);
        if self.cv_stack_len < CV_STACK_INLINE_LEN_MAX {
            self.cv_stack_inline[self.cv_stack_len] = cv;
        } else {
            self.cv_stack_spill.push(cv);
        }
        self.cv_stack_len += 1;
        assert!(self.cv_stack_spill.len() == self.cv_stack_len.saturating_sub(CV_STACK_INLINE_LEN_MAX));
    }

    fn cv_stack_pop(&mut self) -> [u32; 8] {
        let cv = self.cv_stack_get(self.cv_stack_len - 1);
        if self.cv_stack_len > CV_STACK_INLINE_LEN_MAX {
            self.cv_stack_spill.pop();
        }
        self.cv_stack_len -= 1;
        assert!(self.cv_stack_spill.len() == self.cv_stack_len.saturating_sub(CV_STACK_INLINE_LEN_MAX));
        cv
    }

    fn push_chunk_cv(&mut self, cv_chunk: [u32; 8], chunks_total: u64) {
        assert!(chunks_total > 0);
        let mut cv = cv_chunk;
        let mut chunks = chunks_total;
        while chunks & 1 == 0 {
            let left = self.cv_stack_pop();
            cv = parent_output(&left, &cv, &self.key, self.flags).chaining_value();
            chunks >>= 1;
        }
        self.cv_stack_push(cv);
        assert!(self.cv_stack_len as u32 == chunks_total.count_ones());
    }

    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            if self.chunk.len() == CHUNK_LEN {
                let cv_chunk = self.chunk.output().chaining_value();
                let chunks_total = self.chunk.chunk_counter + 1;
                self.push_chunk_cv(cv_chunk, chunks_total);
                self.chunk = ChunkState::new(&self.key, chunks_total, self.flags);
            }
            let take = (CHUNK_LEN - self.chunk.len()).min(input.len());
            assert!(take > 0);
            self.chunk.update(&input[..take]);
            input = &input[take..];
        }
        self
    }

    fn root_output(&self) -> Output {
        assert!(self.cv_stack_len <= CV_STACK_LEN_MAX);
        let mut output = self.chunk.output();
        let mut remaining = self.cv_stack_len;
        while remaining > 0 {
            remaining -= 1;
            let left = self.cv_stack_get(remaining);
            output = parent_output(&left, &output.chaining_value(), &self.key, self.flags);
        }
        output
    }

    pub fn finalize(&self) -> [u8; OUT_LEN] {
        self.root_output().root_hash()
    }

    pub fn finalize_xof(&self) -> OutputReader {
        OutputReader::new(self.root_output())
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

fn hash_with_key_words(key: [u32; 8], flags: u32, input: &[u8]) -> [u8; OUT_LEN] {
    if input.len() <= CHUNK_LEN {
        let mut chunk = ChunkState::new(&key, 0, flags);
        chunk.update(input);
        return chunk.output().root_hash();
    }
    let mut hasher = Hasher::with_key_words(key, flags);
    hasher.update(input);
    hasher.finalize()
}

pub fn hash(input: &[u8]) -> [u8; OUT_LEN] {
    hash_with_key_words(IV, 0, input)
}

pub fn keyed_hash(key: &[u8; KEY_LEN], input: &[u8]) -> [u8; OUT_LEN] {
    hash_with_key_words(words_from_key(key), KEYED_HASH, input)
}

#[derive(Clone)]
pub struct OutputReader {
    output: Output,
    block_counter: u64,
    block: [u8; BLOCK_LEN],
    block_position: usize,
}

impl OutputReader {
    fn new(output: Output) -> Self {
        assert!(output.flags & ROOT == 0);
        Self { output, block_counter: 0, block: [0; BLOCK_LEN], block_position: BLOCK_LEN }
    }

    pub fn position(&self) -> u64 {
        assert!(self.block_position <= BLOCK_LEN);
        self.block_counter * BLOCK_LEN as u64 - (BLOCK_LEN - self.block_position) as u64
    }

    pub fn fill(&mut self, buffer: &mut [u8]) {
        let mut filled = 0;
        while filled < buffer.len() {
            if self.block_position == BLOCK_LEN {
                assert!(self.block_counter < u64::MAX);
                self.block = self.output.root_block(self.block_counter);
                self.block_counter += 1;
                self.block_position = 0;
            }
            let take = (BLOCK_LEN - self.block_position).min(buffer.len() - filled);
            buffer[filled..filled + take]
                .copy_from_slice(&self.block[self.block_position..self.block_position + take]);
            self.block_position += take;
            filled += take;
        }
        assert!(filled == buffer.len());
        assert!(self.block_position <= BLOCK_LEN);
    }
}
