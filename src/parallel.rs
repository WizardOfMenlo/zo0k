use std::any::Any;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;

pub const THREADS_MAX: usize = 512;
const PIECES_PER_THREAD: usize = 8;
const SPIN_PAUSE_ROUNDS: usize = 2048;
const SPIN_YIELD_ROUNDS: usize = 8;
const WORKER_STACK_BYTES: usize = 8 << 20;

static THREADS_CHOSEN: AtomicUsize = AtomicUsize::new(0);
static POOL: OnceLock<Pool> = OnceLock::new();

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
    let pieces = pieces_len(len, piece_len_min);
    if pieces == 1 {
        return (0..len).map(f).collect();
    }
    assert!((2..=len).contains(&pieces));
    let mut results: Vec<R> = Vec::with_capacity(len);
    let slots = SliceMut::new(&mut results.spare_capacity_mut()[..len]);
    run(pieces, &|piece| {
        let range = piece_range(len, pieces, piece);
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
    let pieces = pieces_len(len, piece_len_min);
    let results = map_range(pieces, 1, |piece| f(piece_range(len, pieces, piece)));
    assert_eq!(results.len(), pieces);
    results
}

pub fn for_each_piece_mut<T: Send>(
    items: &mut [T],
    piece_len_min: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let len = items.len();
    let pieces = pieces_len(len, piece_len_min);
    assert!(pieces >= 1 && pieces <= len.max(1));
    if pieces == 1 {
        return f(0, items);
    }
    let slots = SliceMut::new(items);
    assert!(pieces >= 2);
    run(pieces, &|piece| {
        let range = piece_range(len, pieces, piece);
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
    let pieces = pieces_len(chunks_len, piece_len_min.div_ceil(chunk_len).max(1));
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
        let units = piece_range(chunks_len, pieces, piece);
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
    let pieces = pieces_len(len, piece_len_min);
    assert!(pieces >= 1);
    if pieces == 1 {
        return f(0, a, b);
    }
    let slots_a = SliceMut::new(a);
    let slots_b = SliceMut::new(b);
    run(pieces, &|piece| {
        let range = piece_range(len, pieces, piece);
        let start = range.start;
        f(start, unsafe { slots_a.range_mut(range.clone()) }, unsafe { slots_b.range_mut(range) });
    });
}

pub(crate) fn map_four_mut<T: Send, R: Send>(
    a: &mut [T],
    b: &mut [T],
    c: &mut [T],
    d: &mut [T],
    piece_len_min: usize,
    f: impl Fn(usize, &mut [T], &mut [T], &mut [T], &mut [T]) -> R + Sync,
) -> Vec<R> {
    let len = a.len();
    assert!(b.len() == len && c.len() == len && d.len() == len);
    let pieces = pieces_len(len, piece_len_min);
    assert!(pieces >= 1);
    if pieces == 1 {
        return vec![f(0, a, b, c, d)];
    }
    let slots_a = SliceMut::new(a);
    let slots_b = SliceMut::new(b);
    let slots_c = SliceMut::new(c);
    let slots_d = SliceMut::new(d);
    let mut results: Vec<R> = Vec::with_capacity(pieces);
    let out = SliceMut::new(&mut results.spare_capacity_mut()[..pieces]);
    run(pieces, &|piece| {
        let range = piece_range(len, pieces, piece);
        let value = f(
            range.start,
            unsafe { slots_a.range_mut(range.clone()) },
            unsafe { slots_b.range_mut(range.clone()) },
            unsafe { slots_c.range_mut(range.clone()) },
            unsafe { slots_d.range_mut(range) },
        );
        unsafe { out.range_mut(piece..piece + 1)[0].write(value) };
    });
    assert!(results.capacity() >= pieces);
    unsafe { results.set_len(pieces) };
    results
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

fn run(tasks_len: usize, task: &(dyn Fn(usize) + Sync)) {
    assert!(tasks_len >= 1);
    if tasks_len == 1 || threads() == 1 {
        for index in 0..tasks_len {
            task(index);
        }
        return;
    }
    let pool = pool();
    let job = Arc::new(Job::new(task, tasks_len));
    pool.push(&job);
    pool.execute(&job);
    pool.prune();
    pool.help_until_finished(&job);
    assert!(job.is_finished());
    if job.panicked.load(Ordering::Acquire) {
        let payload = lock(&job.panic).take();
        if let Some(payload) = payload {
            panic::resume_unwind(payload);
        }
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
    fn new<'a>(task: &'a (dyn Fn(usize) + Sync + 'a), tasks_len: usize) -> Self {
        assert!(tasks_len >= 2);
        let task: *const (dyn Fn(usize) + Sync + 'a) = task;
        let task = unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + 'a), *const (dyn Fn(usize) + Sync + 'static)>(
                task,
            )
        };
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
    sleepers: usize,
    workers_len: usize,
}

struct Pool {
    queue: Mutex<Queue>,
    wake: Condvar,
    parked: Condvar,
    jobs_len: CachePadded<AtomicUsize>,
    workers_len: AtomicUsize,
}

fn pool() -> &'static Pool {
    let pool = POOL.get_or_init(|| Pool {
        queue: Mutex::new(Queue { jobs: Vec::new(), sleepers: 0, workers_len: 0 }),
        wake: Condvar::new(),
        parked: Condvar::new(),
        jobs_len: CachePadded(AtomicUsize::new(0)),
        workers_len: AtomicUsize::new(0),
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

    fn push(&self, job: &Arc<Job>) {
        assert!(!job.is_exhausted());
        let (wakes, notify_all) = {
            let mut queue = lock(&self.queue);
            queue.jobs.push(Arc::clone(job));
            self.jobs_len.0.store(queue.jobs.len(), Ordering::Release);
            let wakes = queue.sleepers.min(job.tasks_len - 1);
            (wakes, wakes > 1 && wakes == queue.sleepers)
        };
        assert!(wakes < THREADS_MAX);
        if notify_all {
            self.wake.notify_all();
        } else {
            for _ in 0..wakes {
                self.wake.notify_one();
            }
        }
    }

    fn take_job(&self, queue: &mut Queue) -> Option<Arc<Job>> {
        if queue.jobs.first().is_some_and(|job| job.is_exhausted()) {
            queue.jobs.retain(|job| !job.is_exhausted());
            self.jobs_len.0.store(queue.jobs.len(), Ordering::Release);
        }
        let job = queue.jobs.first().cloned();
        assert!(job.as_ref().is_none_or(|job| job.tasks_len >= 2));
        job
    }

    fn prune(&self) {
        if self.jobs_len.0.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut queue = lock(&self.queue);
        queue.jobs.retain(|job| !job.is_exhausted());
        self.jobs_len.0.store(queue.jobs.len(), Ordering::Release);
    }

    fn try_take_job(&self) -> Option<Arc<Job>> {
        if self.jobs_len.0.load(Ordering::Acquire) == 0 {
            return None;
        }
        self.take_job(&mut lock(&self.queue))
    }

    fn execute(&self, job: &Job) {
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

    fn work_forever(&self, worker: usize) {
        assert!(worker < THREADS_MAX);
        loop {
            let job = self.next_job(worker);
            self.execute(&job);
        }
    }

    fn spin_for_job(&self, worker: usize) -> Option<Arc<Job>> {
        assert!(worker < THREADS_MAX);
        if !is_active(worker) {
            return None;
        }
        for _ in 0..SPIN_PAUSE_ROUNDS {
            if let Some(job) = self.try_take_job() {
                return Some(job);
            }
            std::hint::spin_loop();
        }
        for _ in 0..SPIN_YIELD_ROUNDS {
            if !is_active(worker) {
                return None;
            }
            if let Some(job) = self.try_take_job() {
                return Some(job);
            }
            thread::yield_now();
        }
        assert!(worker < THREADS_MAX);
        None
    }

    fn next_job(&self, worker: usize) -> Arc<Job> {
        assert!(worker < THREADS_MAX);
        if let Some(job) = self.spin_for_job(worker) {
            return job;
        }
        let mut queue = lock(&self.queue);
        loop {
            if !is_active(worker) {
                queue = wait(&self.parked, queue);
                continue;
            }
            if let Some(job) = self.take_job(&mut queue) {
                assert!(!job.is_exhausted() || job.tasks_len >= 2);
                return job;
            }
            assert!(queue.sleepers < THREADS_MAX);
            queue.sleepers += 1;
            queue = wait(&self.wake, queue);
            queue.sleepers -= 1;
        }
    }

    fn help_until_finished(&self, job: &Job) {
        assert!(job.tasks_len >= 2);
        let mut idle_rounds = 0;
        while !job.is_finished() {
            if let Some(other) = self.try_take_job() {
                self.execute(&other);
                idle_rounds = 0;
            } else if idle_rounds < SPIN_PAUSE_ROUNDS {
                idle_rounds += 1;
                std::hint::spin_loop();
            } else if idle_rounds < SPIN_PAUSE_ROUNDS + SPIN_YIELD_ROUNDS {
                idle_rounds += 1;
                thread::yield_now();
            } else {
                self.sleep_until_ready(job);
                idle_rounds = 0;
            }
        }
        assert!(job.is_exhausted());
    }

    fn sleep_until_ready(&self, job: &Job) {
        let mut queue = lock(&self.queue);
        if self.take_job(&mut queue).is_some() {
            return;
        }
        job.waiter_sleeping.0.store(true, Ordering::SeqCst);
        if job.is_finished() {
            return;
        }
        queue.sleepers += 1;
        queue = wait(&self.wake, queue);
        queue.sleepers -= 1;
        assert!(queue.workers_len < THREADS_MAX);
    }
}
