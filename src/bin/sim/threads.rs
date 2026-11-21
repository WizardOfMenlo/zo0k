use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure, guarded};
use std::sync::atomic::{AtomicUsize, Ordering};
use zo0k::parallel;

pub const THREAD_CHOICES: [usize; 5] = [1, 2, 3, 5, 0];
const LEN_MAX: usize = 5000;
const JOIN_DEPTH_MAX: usize = 7;

pub struct ThreadsGuard;

impl ThreadsGuard {
    pub fn set(threads: usize) -> Self {
        assert!(threads <= parallel::THREADS_MAX);
        parallel::set_threads(threads);
        assert!(threads == 0 || parallel::threads() == threads);
        ThreadsGuard
    }
}

impl Drop for ThreadsGuard {
    fn drop(&mut self) {
        parallel::set_threads(0);
        assert!(parallel::threads() >= 1);
    }
}

fn values(prng: &mut Prng, len: usize) -> Vec<u64> {
    let values: Vec<u64> = (0..len).map(|_| prng.next_u64()).collect();
    assert_eq!(values.len(), len);
    values
}

fn scramble(value: u64) -> u64 {
    let mut state = value;
    crate::prng::splitmix64(&mut state)
}

fn maps(prng: &mut Prng, items: &[u64], piece_len_min: usize) -> Result<(), String> {
    let len = items.len();
    let expected: Vec<u64> = items.iter().map(|&v| scramble(v)).collect();
    ensure(parallel::map(items, |&v| scramble(v)) == expected, || format!("map of {len} differs"))?;
    let keep = prng.range(1, 16) as u64;
    let filtered: Vec<u64> = items.iter().filter(|&&v| v % keep == 0).map(|&v| scramble(v)).collect();
    let found = parallel::filter_map(items, |&v| (v % keep == 0).then(|| scramble(v)));
    ensure(found == filtered, || format!("filter_map of {len} differs"))?;
    let ranged = parallel::map_range(len, piece_len_min, |i| scramble(items[i]));
    ensure(ranged == expected, || format!("map_range of {len} with pieces of {piece_len_min} differs"))?;
    let chunk_len = prng.range(1, 70);
    let chunks: Vec<u64> =
        items.chunks(chunk_len).map(|c| c.iter().fold(0, |a, &v| a ^ scramble(v))).collect();
    let mapped =
        parallel::map_chunks(items, chunk_len, piece_len_min, |c| c.iter().fold(0, |a, &v| a ^ scramble(v)));
    ensure(mapped == chunks, || format!("map_chunks of {len} by {chunk_len} differs"))?;
    let pieces = parallel::map_pieces(len, piece_len_min, |range| range);
    ensure(!pieces.is_empty() && pieces[0].start == 0, || {
        format!("map_pieces of {len} does not start at zero")
    })?;
    ensure(pieces.windows(2).all(|w| w[0].end == w[1].start), || format!("map_pieces of {len} has gaps"))?;
    ensure(pieces.last().is_some_and(|r| r.end == len), || format!("map_pieces of {len} drops the end"))?;
    ensure(pieces.iter().all(|r| r.len() >= piece_len_min.min(len)) || pieces.len() == 1, || {
        "a piece is too short".into()
    })
}

fn mutations(prng: &mut Prng, items: &[u64], piece_len_min: usize) -> Result<(), String> {
    let len = items.len();
    let mut indices = vec![usize::MAX; len];
    parallel::for_each_piece_mut(&mut indices, piece_len_min, |start, piece| {
        piece.iter_mut().enumerate().for_each(|(offset, slot)| *slot = start + offset);
    });
    ensure(indices.iter().enumerate().all(|(i, &v)| i == v), || {
        format!("for_each_piece_mut of {len} misplaces pieces")
    })?;
    let chunk_len = prng.range(1, 70);
    let mut chunks = vec![usize::MAX; len];
    parallel::for_each_chunk_mut(&mut chunks, chunk_len, piece_len_min, |chunk, slots| slots.fill(chunk));
    ensure(chunks.iter().enumerate().all(|(i, &c)| c == i / chunk_len), || {
        format!("for_each_chunk_mut of {len} by {chunk_len}")
    })?;
    let mut sums = items.to_vec();
    let others: Vec<u64> = values(prng, len);
    let mut copy = others.clone();
    parallel::for_each_pair_mut(&mut sums, &mut copy, piece_len_min, |a, b| {
        a.iter_mut().zip(b.iter_mut()).for_each(|(x, y)| {
            *x ^= *y;
            *y = 0;
        });
    });
    let paired = sums.iter().zip(items).zip(&others).all(|((&s, &x), &y)| s == x ^ y);
    ensure(paired && copy.iter().all(|&y| y == 0), || format!("for_each_pair_mut of {len} misaligns"))
}

fn searches(prng: &mut Prng, len: usize, piece_len_min: usize) -> Result<(), String> {
    let start = prng.range(0, 1000);
    let density = prng.pick(&[0usize, 1, 10, 500, 1000]);
    let marked: Vec<bool> = (0..len).map(|_| prng.below(1000) < density).collect();
    let calls = AtomicUsize::new(0);
    let found = parallel::find_first(start..start + len, piece_len_min, |i| {
        calls.fetch_add(1, Ordering::Relaxed);
        marked[i - start]
    });
    let expected = marked.iter().position(|&m| m).map(|i| start + i);
    ensure(found == expected, || {
        format!("find_first in {len} from {start} gives {found:?}, not {expected:?}")
    })?;
    ensure(calls.load(Ordering::Relaxed) <= len, || "find_first evaluates an index twice".into())
}

const SPIN_ROUNDS: usize = 200_000;

fn spin(rounds: usize) -> u64 {
    let mut state = rounds as u64;
    for _ in 0..rounds {
        state = std::hint::black_box(scramble(state));
    }
    state
}

fn racing_search(len: usize) -> Result<(), String> {
    let len = len.clamp(64, 2000);
    let start = 7;
    let found = parallel::find_first(start..start + len, 1, |i| {
        let rounds = if i == start { SPIN_ROUNDS } else { 2 * SPIN_ROUNDS };
        spin(rounds) != 0 || i >= start
    });
    ensure(found == Some(start), || format!("find_first with a slow first hit in {len} gives {found:?}"))
}

fn join_sum(items: &[u64], depth: usize) -> u64 {
    if depth == 0 || items.len() < 2 {
        return items.iter().fold(0u64, |a, &v| a.wrapping_add(scramble(v)));
    }
    let (left, right) = items.split_at(items.len() / 2);
    let (a, b) = parallel::join(|| join_sum(left, depth - 1), || join_sum(right, depth - 1));
    a.wrapping_add(b)
}

fn nesting(prng: &mut Prng, items: &[u64]) -> Result<(), String> {
    let depth = prng.range(0, JOIN_DEPTH_MAX);
    let expected = items.iter().fold(0u64, |a, &v| a.wrapping_add(scramble(v)));
    ensure(join_sum(items, depth) == expected, || format!("join recursion of depth {depth} differs"))?;
    let (first, second) = parallel::join(|| items.len(), || items.first().copied());
    ensure((first, second) == (items.len(), items.first().copied()), || "join swaps its results".into())?;
    let outer = prng.range(1, 40);
    let inner = prng.range(0, 200);
    let nested: Vec<Vec<u64>> =
        parallel::map_range(outer, 1, |i| parallel::map_range(inner, 1, |j| scramble((i * 1000 + j) as u64)));
    let flat = nested.iter().enumerate().all(|(i, row)| {
        row.len() == inner && row.iter().enumerate().all(|(j, &v)| v == scramble((i * 1000 + j) as u64))
    });
    ensure(flat, || format!("nested map of {outer} by {inner} differs"))
}

fn panics_propagate(len: usize) -> Result<(), String> {
    let poisoned = len.max(2) - 1;
    let outcome =
        guarded(|| {
            let mapped = parallel::map_range(len.max(2), 1, |i| {
                if i == poisoned { panic!("sim-injected panic") } else { i }
            });
            Ok(mapped.len())
        });
    ensure(outcome.is_err_and(|e| e.contains("sim-injected panic")), || {
        "a panic in map_range is lost".into()
    })?;
    let after = parallel::map_range(64, 1, |i| i * 2);
    ensure(after.iter().enumerate().all(|(i, &v)| v == 2 * i), || "the pool fails after a panic".into())
}

pub fn primitives(prng: &mut Prng) -> Result<(), String> {
    let threads = prng.pick(&THREAD_CHOICES);
    let _guard = ThreadsGuard::set(threads);
    let len = if prng.chance(20) { prng.range(0, 8) } else { prng.range(0, LEN_MAX) };
    let piece_len_min = prng.pick(&[1usize, 2, 7, 64, 1000]);
    let items = values(prng, len);
    maps(prng, &items, piece_len_min).map_err(|e| format!("{threads} threads: {e}"))?;
    mutations(prng, &items, piece_len_min).map_err(|e| format!("{threads} threads: {e}"))?;
    searches(prng, len, piece_len_min).map_err(|e| format!("{threads} threads: {e}"))?;
    racing_search(len).map_err(|e| format!("{threads} threads: {e}"))?;
    nesting(prng, &items).map_err(|e| format!("{threads} threads: {e}"))?;
    if prng.chance(10) {
        panics_propagate(len).map_err(|e| format!("{threads} threads: {e}"))?;
    }
    ensure(parallel::threads() >= 1, || "no threads".into())
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    ctx.check("parallel_matches_sequential", primitives)
}
