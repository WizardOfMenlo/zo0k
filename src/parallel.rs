use std::any::Any;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;

pub const THREADS_MAX: usize = 512;
const PIECES_PER_THREAD: usize = 16;
const COARSE_PIECES_PER_THREAD: usize = 64;
const BOUNDARY_ALIGN: usize = 16;
const SLOT_COUNT: usize = 64;
const SPIN_PAUSE_ROUNDS: usize = 512;
const SPIN_YIELD_ROUNDS: usize = 2;
const SPIN_CHECK_STRIDE: usize = 64;
const WORKER_STACK_BYTES: usize = 8 << 20;

const RESERVED_BIT: u64 = 1;
const OPEN_BIT: u64 = 2;
const REF_UNIT: u64 = 4;
const REF_MASK: u64 = 0xFFFF_FFFC;
const GEN_UNIT: u64 = 1u64 << 32;

static THREADS_CHOSEN: AtomicUsize = AtomicUsize::new(0);
static POOL: OnceLock<Box<Pool>> = OnceLock::new();

#[repr(align(128))]
struct CachePadded<T>(T);

struct SliceMut<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Sync for SliceMut<T> {}

impl<T> SliceMut<T> {
    fn new(slice: &mut [T]) -> Self {
        assert!(slice.len() <= isize::MAX as usize);
        let ptr = slice.as_mut_ptr();
        assert!(!ptr.is_null());
        Self { ptr, len: slice.len() }
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn range_mut(&self, range: Range<usize>) -> &mut [T] {
        assert!(range.start <= range.end && range.end <= self.len);
        let len = range.len();
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(range.start), len) }
    }
}

struct CellSlot<T>(UnsafeCell<Option<T>>);

unsafe impl<T: Send> Sync for CellSlot<T> {}

impl<T> CellSlot<T> {
    fn new(value: Option<T>) -> Self {
        Self(UnsafeCell::new(value))
    }

    unsafe fn take(&self) -> Option<T> {
        unsafe { (*self.0.get()).take() }
    }

    unsafe fn set(&self, value: T) {
        unsafe {
            *self.0.get() = Some(value);
        }
    }

    fn into_inner(self) -> Option<T> {
        self.0.into_inner()
    }
}

#[derive(Clone, Copy)]
struct PiecePlan {
    len: usize,
    align: usize,
    units: usize,
    pieces: usize,
}

impl PiecePlan {
    fn compute(len: usize, piece_len_min: usize) -> Self {
        assert!(piece_len_min >= 1);
        let threads = threads();
        if threads == 1 || len <= piece_len_min || len / piece_len_min < 2 {
            return Self { len, align: 1, units: len, pieces: 1 };
        }
        if piece_len_min == 1 {
            let pieces = len.min(threads * COARSE_PIECES_PER_THREAD);
            assert!((2..=len).contains(&pieces));
            return Self { len, align: 1, units: len, pieces };
        }
        let align = if piece_len_min >= BOUNDARY_ALIGN
            && (len / BOUNDARY_ALIGN) / 2 >= piece_len_min.div_ceil(BOUNDARY_ALIGN)
        {
            BOUNDARY_ALIGN
        } else {
            1
        };
        let units = len / align;
        let min_units = piece_len_min.div_ceil(align);
        assert!(units / 2 >= min_units);
        let pieces = (units / min_units).clamp(2, threads * PIECES_PER_THREAD);
        Self { len, align, units, pieces }
    }

    fn pieces(self) -> usize {
        self.pieces
    }

    fn range(self, piece: usize) -> Range<usize> {
        assert!(piece < self.pieces);
        let start = ((piece as u128 * self.units as u128) / self.pieces as u128) as usize * self.align;
        let end = if piece + 1 == self.pieces {
            self.len
        } else {
            (((piece + 1) as u128 * self.units as u128) / self.pieces as u128) as usize * self.align
        };
        assert!(start <= end && end <= self.len);
        start..end
    }
}

pub fn set_threads(threads: usize) {
    assert!(threads <= THREADS_MAX, "at most {THREADS_MAX} threads");
    let threads_resolved = if threads == 0 { threads_default() } else { threads };
    assert!((1..=THREADS_MAX).contains(&threads_resolved));
    THREADS_CHOSEN.store(threads_resolved, Ordering::SeqCst);
    if let Some(pool) = POOL.get() {
        pool.spawn_workers(threads_resolved - 1);
        pool.wake_all();
    }
}

pub fn threads() -> usize {
    let chosen = THREADS_CHOSEN.load(Ordering::Relaxed);
    let threads = if chosen == 0 { threads_default() } else { chosen };
    assert!((1..=THREADS_MAX).contains(&threads));
    threads
}

fn threads_default() -> usize {
    static DEFAULT: OnceLock<usize> = OnceLock::new();
    let available = || thread::available_parallelism().map_or(1, |threads| threads.get());
    let threads = *DEFAULT.get_or_init(|| available().clamp(1, THREADS_MAX));
    assert!((1..=THREADS_MAX).contains(&threads));
    threads
}

pub fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    assert!(items.len() < usize::MAX / THREADS_MAX);
    let results = map_range(items.len(), 1, |index| f(&items[index]));
    assert_eq!(results.len(), items.len());
    results
}

pub fn filter_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> Option<R> + Sync) -> Vec<R> {
    let options = map(items, f);
    assert_eq!(options.len(), items.len());
    let results: Vec<R> = options.into_iter().flatten().collect();
    assert!(results.len() <= items.len());
    results
}

pub fn map_range<R: Send>(len: usize, piece_len_min: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let plan = PiecePlan::compute(len, piece_len_min);
    let pieces = plan.pieces();
    if pieces == 1 {
        return (0..len).map(f).collect();
    }
    assert!((2..=len).contains(&pieces));
    let mut results: Vec<R> = Vec::with_capacity(len);
    let slots = SliceMut::new(&mut results.spare_capacity_mut()[..len]);
    run(pieces, &|piece| {
        let range = plan.range(piece);
        let start = range.start;
        let slot: &mut [MaybeUninit<R>] = unsafe { slots.range_mut(range) };
        for (offset, result) in slot.iter_mut().enumerate() {
            result.write(f(start + offset));
        }
    });
    assert!(results.capacity() >= len);
    unsafe { results.set_len(len) };
    results
}

pub fn map_chunks<T: Sync, R: Send>(
    items: &[T],
    chunk_len: usize,
    piece_len_min: usize,
    f: impl Fn(&[T]) -> R + Sync,
) -> Vec<R> {
    assert!(chunk_len >= 1);
    let chunks_len = items.len().div_ceil(chunk_len);
    let chunks_min = piece_len_min.div_ceil(chunk_len).max(1);
    let results = map_range(chunks_len, chunks_min, |chunk| {
        let start = chunk * chunk_len;
        f(&items[start..(start + chunk_len).min(items.len())])
    });
    assert_eq!(results.len(), chunks_len);
    results
}

pub fn map_pieces<R: Send>(len: usize, piece_len_min: usize, f: impl Fn(Range<usize>) -> R + Sync) -> Vec<R> {
    let plan = PiecePlan::compute(len, piece_len_min);
    let pieces = plan.pieces();
    if pieces == 1 {
        return vec![f(0..len)];
    }
    let mut results: Vec<R> = Vec::with_capacity(pieces);
    let out = SliceMut::new(&mut results.spare_capacity_mut()[..pieces]);
    run(pieces, &|piece| {
        let value = f(plan.range(piece));
        unsafe { out.range_mut(piece..piece + 1)[0].write(value) };
    });
    assert!(results.capacity() >= pieces);
    unsafe { results.set_len(pieces) };
    assert_eq!(results.len(), pieces);
    results
}

pub(crate) fn fold_pieces<S: Send>(
    len: usize,
    piece_len_min: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, Range<usize>) + Sync,
) -> Vec<S> {
    let plan = PiecePlan::compute(len, piece_len_min);
    let pieces = plan.pieces();
    assert!(pieces >= 1);
    if pieces == 1 {
        let mut acc = init();
        f(&mut acc, 0..len);
        return vec![acc];
    }
    let workers = threads().min(pieces);
    let next_piece = CachePadded(AtomicUsize::new(0));
    let mut partials: Vec<Option<S>> = Vec::with_capacity(workers);
    let out = SliceMut::new(&mut partials.spare_capacity_mut()[..workers]);
    run(workers, &|worker| {
        let mut state: Option<S> = None;
        for _ in 0..pieces {
            let piece = next_piece.0.fetch_add(1, Ordering::Relaxed);
            if piece >= pieces {
                break;
            }
            let acc = state.get_or_insert_with(&init);
            f(acc, plan.range(piece));
        }
        unsafe { out.range_mut(worker..worker + 1)[0].write(state) };
    });
    assert!(partials.capacity() >= workers);
    unsafe { partials.set_len(workers) };
    partials.into_iter().flatten().collect()
}

pub(crate) fn for_each_piece_mut_with<T: Send, S>(
    items: &mut [T],
    piece_len_min: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize, &mut [T]) + Sync,
) {
    let len = items.len();
    let plan = PiecePlan::compute(len, piece_len_min);
    let pieces = plan.pieces();
    assert!(pieces >= 1);
    if pieces == 1 {
        let mut state = init();
        f(&mut state, 0, items);
        return;
    }
    let workers = threads().min(pieces);
    let next_piece = CachePadded(AtomicUsize::new(0));
    let slots = SliceMut::new(items);
    run(workers, &|_| {
        let mut state: Option<S> = None;
        for _ in 0..pieces {
            let piece = next_piece.0.fetch_add(1, Ordering::Relaxed);
            if piece >= pieces {
                break;
            }
            let range = plan.range(piece);
            let start = range.start;
            let acc = state.get_or_insert_with(&init);
            f(acc, start, unsafe { slots.range_mut(range) });
        }
    });
    assert!(next_piece.0.load(Ordering::Relaxed) >= pieces);
}

pub(crate) fn fold_pair_chunks_mut_with<A: Send, B: Send, S: Send>(
    a: &mut [A],
    b: &mut [B],
    chunk_len: usize,
    chunks_min: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize, &mut [A], &mut [B]) + Sync,
) -> Vec<S> {
    assert_eq!(a.len(), b.len());
    assert!(chunk_len >= 1 && a.len().is_multiple_of(chunk_len));
    let chunks_len = a.len() / chunk_len;
    let plan = PiecePlan::compute(chunks_len, chunks_min.max(1));
    let pieces = plan.pieces();
    if pieces == 1 {
        let mut state = init();
        f(&mut state, 0, a, b);
        return vec![state];
    }
    let workers = threads().min(pieces);
    let next_piece = CachePadded(AtomicUsize::new(0));
    let slots_a = SliceMut::new(a);
    let slots_b = SliceMut::new(b);
    let mut partials: Vec<Option<S>> = Vec::with_capacity(workers);
    let out = SliceMut::new(&mut partials.spare_capacity_mut()[..workers]);
    run(workers, &|worker| {
        let mut state: Option<S> = None;
        for _ in 0..pieces {
            let piece = next_piece.0.fetch_add(1, Ordering::Relaxed);
            if piece >= pieces {
                break;
            }
            let units = plan.range(piece);
            let elem_range = (units.start * chunk_len)..(units.end * chunk_len);
            let acc = state.get_or_insert_with(&init);
            f(acc, units.start, unsafe { slots_a.range_mut(elem_range.clone()) }, unsafe {
                slots_b.range_mut(elem_range)
            });
        }
        unsafe { out.range_mut(worker..worker + 1)[0].write(state) };
    });
    assert!(partials.capacity() >= workers);
    unsafe { partials.set_len(workers) };
    partials.into_iter().flatten().collect()
}

pub fn for_each_piece_mut<T: Send>(
    items: &mut [T],
    piece_len_min: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let len = items.len();
    let plan = PiecePlan::compute(len, piece_len_min);
    let pieces = plan.pieces();
    assert!(pieces >= 1 && pieces <= len.max(1));
    if pieces == 1 {
        return f(0, items);
    }
    let slots = SliceMut::new(items);
    assert!(pieces >= 2);
    run(pieces, &|piece| {
        let range = plan.range(piece);
        f(range.start, unsafe { slots.range_mut(range) });
    });
}

pub fn for_each_chunk_mut<T: Send>(
    items: &mut [T],
    chunk_len: usize,
    piece_len_min: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    assert!(chunk_len >= 1);
    let len = items.len();
    let chunks_len = len.div_ceil(chunk_len);
    let plan = PiecePlan::compute(chunks_len, piece_len_min.div_ceil(chunk_len).max(1));
    let pieces = plan.pieces();
    let visit = |chunk_first: usize, piece: &mut [T]| {
        for (offset, chunk) in piece.chunks_mut(chunk_len).enumerate() {
            f(chunk_first + offset, chunk);
        }
    };
    if pieces == 1 {
        return visit(0, items);
    }
    let slots = SliceMut::new(items);
    assert!(pieces <= chunks_len);
    run(pieces, &|piece| {
        let units = plan.range(piece);
        let start = units.start.saturating_mul(chunk_len).min(len);
        let end = units.end.saturating_mul(chunk_len).min(len);
        visit(units.start, unsafe { slots.range_mut(start..end) });
    });
}

pub fn for_each_pair_mut<A: Send, B: Send>(
    a: &mut [A],
    b: &mut [B],
    piece_len_min: usize,
    f: impl Fn(&mut [A], &mut [B]) + Sync,
) {
    for_each_pair_indexed_mut(a, b, piece_len_min, |_, slice_a, slice_b| f(slice_a, slice_b));
}

pub(crate) fn for_each_pair_indexed_mut<A: Send, B: Send>(
    a: &mut [A],
    b: &mut [B],
    piece_len_min: usize,
    f: impl Fn(usize, &mut [A], &mut [B]) + Sync,
) {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let plan = PiecePlan::compute(len, piece_len_min);
    let pieces = plan.pieces();
    assert!(pieces >= 1);
    if pieces == 1 {
        return f(0, a, b);
    }
    let slots_a = SliceMut::new(a);
    let slots_b = SliceMut::new(b);
    run(pieces, &|piece| {
        let range = plan.range(piece);
        let start = range.start;
        f(start, unsafe { slots_a.range_mut(range.clone()) }, unsafe { slots_b.range_mut(range) });
    });
}

pub(crate) fn fold_four_mut<T: Send, S: Send>(
    a: &mut [T],
    b: &mut [T],
    c: &mut [T],
    d: &mut [T],
    piece_len_min: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize, &mut [T], &mut [T], &mut [T], &mut [T]) + Sync,
) -> Vec<S> {
    let len = a.len();
    assert!(b.len() == len && c.len() == len && d.len() == len);
    let plan = PiecePlan::compute(len, piece_len_min);
    let pieces = plan.pieces();
    assert!(pieces >= 1);
    if pieces == 1 {
        let mut state = init();
        f(&mut state, 0, a, b, c, d);
        return vec![state];
    }
    let workers = threads().min(pieces);
    let next_piece = CachePadded(AtomicUsize::new(0));
    let slots_a = SliceMut::new(a);
    let slots_b = SliceMut::new(b);
    let slots_c = SliceMut::new(c);
    let slots_d = SliceMut::new(d);
    let mut partials: Vec<Option<S>> = Vec::with_capacity(workers);
    let out = SliceMut::new(&mut partials.spare_capacity_mut()[..workers]);
    run(workers, &|worker| {
        let mut state: Option<S> = None;
        for _ in 0..pieces {
            let piece = next_piece.0.fetch_add(1, Ordering::Relaxed);
            if piece >= pieces {
                break;
            }
            let range = plan.range(piece);
            let acc = state.get_or_insert_with(&init);
            f(
                acc,
                range.start,
                unsafe { slots_a.range_mut(range.clone()) },
                unsafe { slots_b.range_mut(range.clone()) },
                unsafe { slots_c.range_mut(range.clone()) },
                unsafe { slots_d.range_mut(range) },
            );
        }
        unsafe { out.range_mut(worker..worker + 1)[0].write(state) };
    });
    assert!(partials.capacity() >= workers);
    unsafe { partials.set_len(workers) };
    partials.into_iter().flatten().collect()
}

pub fn join<RA: Send, RB: Send>(a: impl FnOnce() -> RA + Send, b: impl FnOnce() -> RB + Send) -> (RA, RB) {
    let threads = threads();
    assert!(threads >= 1);
    if threads == 1 {
        return (a(), b());
    }
    let tasks = (CellSlot::new(Some(a)), CellSlot::new(Some(b)));
    let results = (CellSlot::new(None), CellSlot::new(None));
    run(2, &|index| {
        assert!(index < 2);
        if index == 0 {
            let task = unsafe { tasks.0.take() }.expect("the first task runs once");
            unsafe { results.0.set(task()) };
        } else {
            let task = unsafe { tasks.1.take() }.expect("the second task runs once");
            unsafe { results.1.set(task()) };
        }
    });
    let first = results.0.into_inner().expect("the first task ran");
    let second = results.1.into_inner().expect("the second task ran");
    (first, second)
}

pub fn find_first(
    range: Range<usize>,
    piece_len_min: usize,
    predicate: impl Fn(usize) -> bool + Sync,
) -> Option<usize> {
    let len = range.len();
    let pieces = pieces_len(len, piece_len_min);
    let found = CachePadded(AtomicUsize::new(usize::MAX));
    let search = |piece: usize| {
        let sub = piece_range(len, pieces, piece);
        if sub.start >= found.0.load(Ordering::Relaxed) {
            return;
        }
        for offset in sub {
            if offset >= found.0.load(Ordering::Relaxed) {
                return;
            }
            if predicate(range.start + offset) {
                found.0.fetch_min(offset, Ordering::Relaxed);
                return;
            }
        }
    };
    run(pieces, &search);
    let offset = found.0.into_inner();
    assert!(offset == usize::MAX || offset < len);
    (offset < len).then(|| range.start + offset)
}

fn pieces_len(len: usize, piece_len_min: usize) -> usize {
    assert!(piece_len_min >= 1);
    let threads = threads();
    let pieces = if threads == 1 || len <= piece_len_min {
        1
    } else {
        (len / piece_len_min).clamp(1, threads * PIECES_PER_THREAD)
    };
    assert!(pieces == 1 || pieces <= len);
    pieces
}

fn piece_range(len: usize, pieces: usize, piece: usize) -> Range<usize> {
    assert!(piece < pieces);
    assert!(len <= usize::MAX / pieces);
    let start = piece * len / pieces;
    let end = (piece + 1) * len / pieces;
    assert!(start <= end && end <= len);
    start..end
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    assert!(std::mem::size_of::<MutexGuard<'_, T>>() > 0);
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar.wait(guard).unwrap_or_else(PoisonError::into_inner)
}

fn erase_task<'a>(task: &'a (dyn Fn(usize) + Sync + 'a)) -> *const (dyn Fn(usize) + Sync + 'static) {
    let raw: *const (dyn Fn(usize) + Sync + 'a) = task;
    let erased = unsafe {
        std::mem::transmute::<*const (dyn Fn(usize) + Sync + 'a), *const (dyn Fn(usize) + Sync + 'static)>(
            raw,
        )
    };
    assert!(!erased.is_null());
    assert!(std::mem::size_of_val(&erased) == 2 * std::mem::size_of::<usize>());
    erased
}

fn run(tasks_len: usize, task: &(dyn Fn(usize) + Sync)) {
    assert!(tasks_len >= 1);
    let active_threads = threads();
    assert!(active_threads >= 1);
    if tasks_len == 1 || active_threads == 1 {
        for index in 0..tasks_len {
            task(index);
        }
        return;
    }
    let pool = pool();
    if let Some(slot_idx) = pool.acquire_slot() {
        pool.run_slot(slot_idx, tasks_len, task);
    } else {
        pool.run_overflow(tasks_len, task);
    }
}

struct Slot {
    meta: CachePadded<AtomicU64>,
    claimed: CachePadded<AtomicUsize>,
    finished: CachePadded<AtomicUsize>,
    owner_sleeping: CachePadded<AtomicBool>,
    task: UnsafeCell<Option<*const (dyn Fn(usize) + Sync + 'static)>>,
    tasks_len: UnsafeCell<usize>,
    panicked: AtomicBool,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

impl Slot {
    fn new() -> Self {
        Self {
            meta: CachePadded(AtomicU64::new(0)),
            claimed: CachePadded(AtomicUsize::new(0)),
            finished: CachePadded(AtomicUsize::new(0)),
            owner_sleeping: CachePadded(AtomicBool::new(false)),
            task: UnsafeCell::new(None),
            tasks_len: UnsafeCell::new(0),
            panicked: AtomicBool::new(false),
            panic: Mutex::new(None),
        }
    }

    fn init(&self, task: &(dyn Fn(usize) + Sync), tasks_len: usize) {
        assert!(tasks_len >= 2);
        assert_eq!(self.meta.0.load(Ordering::Relaxed) & (RESERVED_BIT | OPEN_BIT | REF_MASK), RESERVED_BIT);
        unsafe {
            *self.task.get() = Some(erase_task(task));
            *self.tasks_len.get() = tasks_len;
        }
        self.claimed.0.store(0, Ordering::Relaxed);
        self.finished.0.store(0, Ordering::Relaxed);
        self.owner_sleeping.0.store(false, Ordering::Relaxed);
        self.panicked.store(false, Ordering::Relaxed);
    }

    fn close(&self, active_mask: &CachePadded<AtomicU64>, idx: usize) {
        assert!(idx < SLOT_COUNT);
        if self.meta.0.load(Ordering::Relaxed) & OPEN_BIT == 0 {
            return;
        }
        let prev = self.meta.0.fetch_and(!OPEN_BIT, Ordering::AcqRel);
        if prev & OPEN_BIT != 0 {
            active_mask.0.fetch_and(!(1u64 << idx), Ordering::AcqRel);
        }
        assert!(idx < 64);
    }

    fn record_panic(&self, payload: Box<dyn Any + Send>) {
        let mut slot = lock(&self.panic);
        if slot.is_none() {
            *slot = Some(payload);
            self.panicked.store(true, Ordering::Release);
        }
        assert!(self.panicked.load(Ordering::Relaxed));
    }

    fn execute_body(&self, active_mask: &CachePadded<AtomicU64>, idx: usize) {
        let tasks_len = unsafe { *self.tasks_len.get() };
        let task_ptr = unsafe { (*self.task.get()).expect("active slot has a task") };
        let task = unsafe { &*task_ptr };
        assert!(tasks_len >= 2);
        let mut completed = 0usize;
        for _ in 0..tasks_len {
            let index = self.claimed.0.fetch_add(1, Ordering::Relaxed);
            if index >= tasks_len {
                self.close(active_mask, idx);
                break;
            }
            if index + 1 == tasks_len {
                self.close(active_mask, idx);
            }
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| task(index))) {
                self.record_panic(payload);
            }
            completed += 1;
        }
        if completed > 0 {
            let finished = self.finished.0.fetch_add(completed, Ordering::Release) + completed;
            assert!(finished <= tasks_len);
        }
    }

    fn finish_and_release(
        &self,
        active_mask: &CachePadded<AtomicU64>,
        idx: usize,
        tasks_len: usize,
    ) -> Option<Box<dyn Any + Send>> {
        assert_eq!(self.meta.0.load(Ordering::Acquire) & (OPEN_BIT | REF_MASK), 0);
        assert_eq!(self.finished.0.load(Ordering::Acquire), tasks_len);
        if active_mask.0.load(Ordering::Relaxed) & (1u64 << idx) != 0 {
            active_mask.0.fetch_and(!(1u64 << idx), Ordering::Release);
        }
        let payload = if self.panicked.load(Ordering::Acquire) { lock(&self.panic).take() } else { None };
        unsafe {
            *self.task.get() = None;
        }
        self.meta.0.fetch_and(!RESERVED_BIT, Ordering::Release);
        payload
    }
}

struct Job {
    task: *const (dyn Fn(usize) + Sync + 'static),
    tasks_len: usize,
    panicked: AtomicBool,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    claimed: CachePadded<AtomicUsize>,
    finished: CachePadded<AtomicUsize>,
    waiter_sleeping: CachePadded<AtomicBool>,
}

unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    fn new(task: &(dyn Fn(usize) + Sync), tasks_len: usize) -> Self {
        assert!(tasks_len >= 2);
        let task = erase_task(task);
        assert!(!task.is_null());
        Self {
            task,
            tasks_len,
            panicked: AtomicBool::new(false),
            panic: Mutex::new(None),
            claimed: CachePadded(AtomicUsize::new(0)),
            finished: CachePadded(AtomicUsize::new(0)),
            waiter_sleeping: CachePadded(AtomicBool::new(false)),
        }
    }

    fn is_exhausted(&self) -> bool {
        self.claimed.0.load(Ordering::Relaxed) >= self.tasks_len
    }

    fn is_finished(&self) -> bool {
        let finished = self.finished.0.load(Ordering::SeqCst);
        assert!(finished <= self.tasks_len);
        finished == self.tasks_len
    }
}

struct Queue {
    jobs: Vec<Arc<Job>>,
    workers_len: usize,
}

struct Pool {
    slots: Box<[Slot; SLOT_COUNT]>,
    active_mask: CachePadded<AtomicU64>,
    sleepers: CachePadded<AtomicUsize>,
    overflow_len: CachePadded<AtomicUsize>,
    queue: Mutex<Queue>,
    wake: Condvar,
    parked: Condvar,
    workers_len: AtomicUsize,
}

fn pool() -> &'static Pool {
    let pool = POOL.get_or_init(|| {
        let slots: Box<[Slot; SLOT_COUNT]> = Box::new(std::array::from_fn(|_| Slot::new()));
        Box::new(Pool {
            slots,
            active_mask: CachePadded(AtomicU64::new(0)),
            sleepers: CachePadded(AtomicUsize::new(0)),
            overflow_len: CachePadded(AtomicUsize::new(0)),
            queue: Mutex::new(Queue { jobs: Vec::new(), workers_len: 0 }),
            wake: Condvar::new(),
            parked: Condvar::new(),
            workers_len: AtomicUsize::new(0),
        })
    });
    let workers_needed = threads() - 1;
    if pool.workers_len.load(Ordering::Acquire) < workers_needed {
        pool.spawn_workers(workers_needed);
    }
    assert!(pool.workers_len.load(Ordering::Acquire) >= workers_needed);
    pool
}

fn is_active(worker: usize) -> bool {
    worker + 1 < threads()
}

impl Pool {
    fn spawn_workers(&'static self, workers_needed: usize) {
        assert!(workers_needed < THREADS_MAX);
        let mut queue = lock(&self.queue);
        for worker in queue.workers_len..workers_needed {
            let builder =
                thread::Builder::new().name(format!("zo0k-{worker}")).stack_size(WORKER_STACK_BYTES);
            builder.spawn(move || self.work_forever(worker)).expect("a worker thread starts");
        }
        queue.workers_len = queue.workers_len.max(workers_needed);
        self.workers_len.store(queue.workers_len, Ordering::Release);
        assert!(queue.workers_len >= workers_needed);
    }

    fn wake_all(&self) {
        let queue = lock(&self.queue);
        assert!(queue.workers_len < THREADS_MAX);
        self.wake.notify_all();
        self.parked.notify_all();
    }

    fn wake_workers(&self, max_wakes: usize) {
        assert!(max_wakes >= 1);
        let sleepers = self.sleepers.0.load(Ordering::SeqCst);
        if sleepers == 0 {
            return;
        }
        let wakes = {
            let _queue = lock(&self.queue);
            let current = self.sleepers.0.load(Ordering::Relaxed);
            let wakes = current.min(max_wakes);
            (wakes, wakes > 1 && wakes == current)
        };
        assert!(wakes.0 <= max_wakes);
        if wakes.1 {
            self.wake.notify_all();
        } else {
            for _ in 0..wakes.0 {
                self.wake.notify_one();
            }
        }
    }

    fn acquire_slot(&self) -> Option<usize> {
        assert_eq!(self.slots.len(), SLOT_COUNT);
        for idx in 0..SLOT_COUNT {
            let slot = &self.slots[idx];
            let m = slot.meta.0.load(Ordering::Relaxed);
            if m & (RESERVED_BIT | OPEN_BIT | REF_MASK) == 0
                && slot
                    .meta
                    .0
                    .compare_exchange(m, m | RESERVED_BIT, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                return Some(idx);
            }
        }
        assert!(self.workers_len.load(Ordering::Relaxed) < THREADS_MAX);
        None
    }

    fn run_slot(&self, idx: usize, tasks_len: usize, task: &(dyn Fn(usize) + Sync)) {
        assert!(idx < SLOT_COUNT);
        assert!(tasks_len >= 2);
        let slot = &self.slots[idx];
        slot.init(task, tasks_len);
        slot.meta.0.fetch_add(OPEN_BIT + GEN_UNIT, Ordering::Release);
        self.active_mask.0.fetch_or(1u64 << idx, Ordering::SeqCst);
        if self.sleepers.0.load(Ordering::SeqCst) > 0 {
            self.wake_workers(tasks_len - 1);
        }
        slot.execute_body(&self.active_mask, idx);
        slot.close(&self.active_mask, idx);
        self.help_slot_until_done(idx);
        let payload = slot.finish_and_release(&self.active_mask, idx, tasks_len);
        if let Some(payload) = payload {
            panic::resume_unwind(payload);
        }
    }

    fn try_join_slot(&self, skip: Option<usize>) -> Option<usize> {
        assert!(skip.is_none_or(|s| s < SLOT_COUNT));
        let mut mask = self.active_mask.0.load(Ordering::Acquire);
        if let Some(skip_idx) = skip {
            mask &= !(1u64 << skip_idx);
        }
        for _ in 0..SLOT_COUNT {
            if mask == 0 {
                break;
            }
            let idx = 63 - mask.leading_zeros() as usize;
            mask &= !(1u64 << idx);
            let slot = &self.slots[idx];
            let mut m = slot.meta.0.load(Ordering::Acquire);
            for _ in 0..THREADS_MAX {
                if m & OPEN_BIT == 0 {
                    break;
                }
                match slot.meta.0.compare_exchange_weak(m, m + REF_UNIT, Ordering::Acquire, Ordering::Relaxed)
                {
                    Ok(_) => return Some(idx),
                    Err(actual) => m = actual,
                }
            }
        }
        assert!(mask == 0);
        None
    }

    fn execute_slot_worker(&self, idx: usize) {
        assert!(idx < SLOT_COUNT);
        let slot = &self.slots[idx];
        slot.execute_body(&self.active_mask, idx);
        let prev = slot.meta.0.fetch_sub(REF_UNIT, Ordering::SeqCst);
        assert!(prev & REF_MASK >= REF_UNIT);
        if (prev - REF_UNIT) & REF_MASK == 0 && slot.owner_sleeping.0.load(Ordering::SeqCst) {
            let _queue = lock(&self.queue);
            self.wake.notify_all();
        }
    }

    fn help_slot_until_done(&self, idx: usize) {
        assert!(idx < SLOT_COUNT);
        let slot = &self.slots[idx];
        let mut idle_rounds = 0usize;
        while slot.meta.0.load(Ordering::Acquire) & REF_MASK != 0 {
            if self.try_help_any(Some(idx)) {
                idle_rounds = 0;
            } else if idle_rounds < SPIN_PAUSE_ROUNDS {
                idle_rounds += 1;
                std::hint::spin_loop();
            } else if idle_rounds < SPIN_PAUSE_ROUNDS + SPIN_YIELD_ROUNDS {
                idle_rounds += 1;
                thread::yield_now();
            } else {
                self.sleep_slot_done(idx);
                idle_rounds = 0;
            }
        }
        assert_eq!(slot.meta.0.load(Ordering::Acquire) & REF_MASK, 0);
    }

    fn sleep_slot_done(&self, idx: usize) {
        assert!(idx < SLOT_COUNT);
        let slot = &self.slots[idx];
        let mut queue = lock(&self.queue);
        self.sleepers.0.fetch_add(1, Ordering::SeqCst);
        slot.owner_sleeping.0.store(true, Ordering::SeqCst);
        let done = slot.meta.0.load(Ordering::SeqCst) & REF_MASK == 0;
        let has_work = (self.active_mask.0.load(Ordering::SeqCst) & !(1u64 << idx)) != 0
            || self.take_overflow_job(&mut queue).is_some();
        if !done && !has_work {
            queue = wait(&self.wake, queue);
        }
        slot.owner_sleeping.0.store(false, Ordering::Relaxed);
        self.sleepers.0.fetch_sub(1, Ordering::SeqCst);
        drop(queue);
        assert!(idx < 64);
    }

    fn run_overflow(&self, tasks_len: usize, task: &(dyn Fn(usize) + Sync)) {
        assert!(tasks_len >= 2);
        let job = Arc::new(Job::new(task, tasks_len));
        {
            let mut queue = lock(&self.queue);
            queue.jobs.push(Arc::clone(&job));
            self.overflow_len.0.store(queue.jobs.len(), Ordering::SeqCst);
        }
        if self.sleepers.0.load(Ordering::SeqCst) > 0 {
            self.wake_workers(tasks_len - 1);
        }
        self.execute_overflow(&job);
        self.prune_overflow();
        self.help_overflow_until_finished(&job);
        assert!(job.is_finished());
        if job.panicked.load(Ordering::Acquire) {
            let payload = lock(&job.panic).take();
            if let Some(payload) = payload {
                panic::resume_unwind(payload);
            }
        }
    }

    fn take_overflow_job(&self, queue: &mut Queue) -> Option<Arc<Job>> {
        if queue.jobs.last().is_some_and(|job| job.is_exhausted()) {
            queue.jobs.retain(|job| !job.is_exhausted());
            self.overflow_len.0.store(queue.jobs.len(), Ordering::Release);
        }
        let job = queue.jobs.last().cloned();
        assert!(job.as_ref().is_none_or(|job| job.tasks_len >= 2));
        job
    }

    fn prune_overflow(&self) {
        if self.overflow_len.0.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut queue = lock(&self.queue);
        queue.jobs.retain(|job| !job.is_exhausted());
        self.overflow_len.0.store(queue.jobs.len(), Ordering::Release);
    }

    fn try_take_overflow(&self) -> Option<Arc<Job>> {
        if self.overflow_len.0.load(Ordering::Acquire) == 0 {
            return None;
        }
        self.take_overflow_job(&mut lock(&self.queue))
    }

    fn execute_overflow(&self, job: &Job) {
        assert!(job.tasks_len >= 2);
        let mut completed = 0usize;
        for _ in 0..job.tasks_len {
            let index = job.claimed.0.fetch_add(1, Ordering::Relaxed);
            if index >= job.tasks_len {
                break;
            }
            let task = unsafe { &*job.task };
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| task(index))) {
                let mut slot = lock(&job.panic);
                if slot.is_none() {
                    *slot = Some(payload);
                    job.panicked.store(true, Ordering::Release);
                }
            }
            completed += 1;
        }
        if completed > 0 {
            let finished = job.finished.0.fetch_add(completed, Ordering::SeqCst) + completed;
            assert!(finished <= job.tasks_len);
            if finished == job.tasks_len && job.waiter_sleeping.0.load(Ordering::SeqCst) {
                let _queue = lock(&self.queue);
                self.wake.notify_all();
            }
        }
    }

    fn sleep_overflow_done(&self, job: &Job) {
        assert!(job.tasks_len >= 2);
        let mut queue = lock(&self.queue);
        self.sleepers.0.fetch_add(1, Ordering::SeqCst);
        job.waiter_sleeping.0.store(true, Ordering::SeqCst);
        let done = job.is_finished();
        let has_work =
            self.active_mask.0.load(Ordering::SeqCst) != 0 || self.take_overflow_job(&mut queue).is_some();
        if !done && !has_work {
            queue = wait(&self.wake, queue);
        }
        job.waiter_sleeping.0.store(false, Ordering::Relaxed);
        self.sleepers.0.fetch_sub(1, Ordering::SeqCst);
        drop(queue);
        assert!(job.tasks_len >= 2);
    }

    fn help_overflow_until_finished(&self, job: &Job) {
        assert!(job.tasks_len >= 2);
        let mut idle_rounds = 0usize;
        while !job.is_finished() {
            if self.try_help_any(None) {
                idle_rounds = 0;
            } else if idle_rounds < SPIN_PAUSE_ROUNDS {
                idle_rounds += 1;
                std::hint::spin_loop();
            } else if idle_rounds < SPIN_PAUSE_ROUNDS + SPIN_YIELD_ROUNDS {
                idle_rounds += 1;
                thread::yield_now();
            } else {
                self.sleep_overflow_done(job);
                idle_rounds = 0;
            }
        }
        assert!(job.is_exhausted());
    }

    fn try_help_any(&self, skip: Option<usize>) -> bool {
        if let Some(idx) = self.try_join_slot(skip) {
            self.execute_slot_worker(idx);
            return true;
        }
        if let Some(job) = self.try_take_overflow() {
            self.execute_overflow(&job);
            return true;
        }
        false
    }

    fn spin_for_work(&self, worker: usize) -> bool {
        assert!(worker < THREADS_MAX);
        if !is_active(worker) {
            return false;
        }
        for round in 0..SPIN_PAUSE_ROUNDS {
            if self.try_help_any(None) {
                return true;
            }
            if round % SPIN_CHECK_STRIDE == SPIN_CHECK_STRIDE - 1 && !is_active(worker) {
                return false;
            }
            std::hint::spin_loop();
        }
        for _ in 0..SPIN_YIELD_ROUNDS {
            if !is_active(worker) {
                return false;
            }
            if self.try_help_any(None) {
                return true;
            }
            thread::yield_now();
        }
        assert!(worker < THREADS_MAX);
        false
    }

    fn work_forever(&self, worker: usize) {
        assert!(worker < THREADS_MAX);
        loop {
            if self.spin_for_work(worker) {
                continue;
            }
            let mut queue = lock(&self.queue);
            loop {
                if !is_active(worker) {
                    queue = wait(&self.parked, queue);
                    continue;
                }
                self.sleepers.0.fetch_add(1, Ordering::SeqCst);
                let has_work = self.active_mask.0.load(Ordering::SeqCst) != 0
                    || self.overflow_len.0.load(Ordering::SeqCst) != 0;
                if has_work {
                    self.sleepers.0.fetch_sub(1, Ordering::SeqCst);
                    drop(queue);
                    break;
                }
                queue = wait(&self.wake, queue);
                self.sleepers.0.fetch_sub(1, Ordering::SeqCst);
                if self.active_mask.0.load(Ordering::Acquire) != 0
                    || self.overflow_len.0.load(Ordering::Acquire) != 0
                {
                    drop(queue);
                    break;
                }
            }
        }
    }
}
