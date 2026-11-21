use std::any::Any;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;

pub const THREADS_MAX: usize = 512;
const PIECES_PER_THREAD: usize = 8;
const SPIN_ROUNDS: usize = 64;
const WORKER_STACK_BYTES: usize = 8 << 20;

static THREADS_CHOSEN: AtomicUsize = AtomicUsize::new(0);
static POOL: OnceLock<Pool> = OnceLock::new();

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
    let slots = split(&mut results.spare_capacity_mut()[..len], 1, pieces);
    run(pieces, &|piece| {
        let start = piece_range(len, pieces, piece).start;
        let slot: &mut [MaybeUninit<R>] = take(&slots, piece);
        for (offset, result) in slot.iter_mut().enumerate() {
            result.write(f(start + offset));
        }
    });
    drop(slots);
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
    let slots = split(items, 1, pieces);
    assert_eq!(slots.len(), pieces);
    run(pieces, &|piece| f(piece_range(len, pieces, piece).start, take(&slots, piece)));
}

pub fn for_each_chunk_mut<T: Send>(
    items: &mut [T],
    chunk_len: usize,
    piece_len_min: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    assert!(chunk_len >= 1);
    let chunks_len = items.len().div_ceil(chunk_len);
    let pieces = pieces_len(chunks_len, piece_len_min.div_ceil(chunk_len).max(1));
    let visit = |chunk_first: usize, piece: &mut [T]| {
        for (offset, chunk) in piece.chunks_mut(chunk_len).enumerate() {
            f(chunk_first + offset, chunk);
        }
    };
    if pieces == 1 {
        return visit(0, items);
    }
    let slots = split(items, chunk_len, pieces);
    assert_eq!(slots.len(), pieces);
    run(pieces, &|piece| visit(piece_range(chunks_len, pieces, piece).start, take(&slots, piece)));
}

pub fn for_each_pair_mut<A: Send, B: Send>(
    a: &mut [A],
    b: &mut [B],
    piece_len_min: usize,
    f: impl Fn(&mut [A], &mut [B]) + Sync,
) {
    assert_eq!(a.len(), b.len());
    let pieces = pieces_len(a.len(), piece_len_min);
    assert!(pieces >= 1);
    if pieces == 1 {
        return f(a, b);
    }
    let slots_a = split(a, 1, pieces);
    let slots_b = split(b, 1, pieces);
    assert_eq!(slots_a.len(), slots_b.len());
    run(pieces, &|piece| f(take(&slots_a, piece), take(&slots_b, piece)));
}

pub fn join<RA: Send, RB: Send>(a: impl FnOnce() -> RA + Send, b: impl FnOnce() -> RB + Send) -> (RA, RB) {
    let threads = threads();
    assert!(threads >= 1);
    if threads == 1 {
        return (a(), b());
    }
    let tasks = (Mutex::new(Some(a)), Mutex::new(Some(b)));
    let results = (Mutex::new(None), Mutex::new(None));
    run(2, &|index| {
        assert!(index < 2);
        if index == 0 {
            let task = lock(&tasks.0).take().expect("the first task runs once");
            *lock(&results.0) = Some(task());
        } else {
            let task = lock(&tasks.1).take().expect("the second task runs once");
            *lock(&results.1) = Some(task());
        }
    });
    let first = results.0.into_inner().unwrap_or_else(PoisonError::into_inner);
    let second = results.1.into_inner().unwrap_or_else(PoisonError::into_inner);
    (first.expect("the first task ran"), second.expect("the second task ran"))
}

pub fn find_first(
    range: Range<usize>,
    piece_len_min: usize,
    predicate: impl Fn(usize) -> bool + Sync,
) -> Option<usize> {
    let len = range.len();
    let pieces = pieces_len(len, piece_len_min);
    let found = AtomicUsize::new(usize::MAX);
    let search = |piece: usize| {
        for offset in piece_range(len, pieces, piece) {
            if offset >= found.load(Ordering::Relaxed) {
                return;
            }
            if predicate(range.start + offset) {
                found.fetch_min(offset, Ordering::Relaxed);
                return;
            }
        }
    };
    run(pieces, &search);
    let offset = found.into_inner();
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

type Slots<'a, T> = Vec<Mutex<Option<&'a mut [T]>>>;

fn split<T>(items: &mut [T], unit_len: usize, pieces: usize) -> Slots<'_, T> {
    assert!(unit_len >= 1 && pieces >= 1);
    let units_len = items.len().div_ceil(unit_len);
    let mut slots = Vec::with_capacity(pieces);
    let mut rest = items;
    for piece in 0..pieces {
        let piece_len = (piece_range(units_len, pieces, piece).len() * unit_len).min(rest.len());
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(piece_len);
        slots.push(Mutex::new(Some(head)));
        rest = tail;
    }
    assert!(rest.is_empty());
    assert_eq!(slots.len(), pieces);
    slots
}

fn take<'a, T>(slots: &[Mutex<Option<&'a mut [T]>>], piece: usize) -> &'a mut [T] {
    assert!(piece < slots.len());
    let slice = lock(&slots[piece]).take();
    slice.expect("every piece is taken once")
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
    let payload = lock(&job.panic).take();
    if let Some(payload) = payload {
        panic::resume_unwind(payload);
    }
}

struct Job {
    task: *const (dyn Fn(usize) + Sync + 'static),
    tasks_len: usize,
    claimed: AtomicUsize,
    finished: AtomicUsize,
    waiter_sleeping: AtomicBool,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
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
            claimed: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            waiter_sleeping: AtomicBool::new(false),
            panic: Mutex::new(None),
        }
    }

    fn is_exhausted(&self) -> bool {
        self.claimed.load(Ordering::Relaxed) >= self.tasks_len
    }

    fn is_finished(&self) -> bool {
        let finished = self.finished.load(Ordering::SeqCst);
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
    jobs_len: AtomicUsize,
    workers_len: AtomicUsize,
}

fn pool() -> &'static Pool {
    let pool = POOL.get_or_init(|| Pool {
        queue: Mutex::new(Queue { jobs: Vec::new(), sleepers: 0, workers_len: 0 }),
        wake: Condvar::new(),
        parked: Condvar::new(),
        jobs_len: AtomicUsize::new(0),
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
        let wakes = {
            let mut queue = lock(&self.queue);
            queue.jobs.push(Arc::clone(job));
            self.jobs_len.store(queue.jobs.len(), Ordering::Release);
            queue.sleepers.min(job.tasks_len - 1)
        };
        assert!(wakes < THREADS_MAX);
        for _ in 0..wakes {
            self.wake.notify_one();
        }
    }

    fn take_job(&self, queue: &mut Queue) -> Option<Arc<Job>> {
        queue.jobs.retain(|job| !job.is_exhausted());
        self.jobs_len.store(queue.jobs.len(), Ordering::Release);
        let job = queue.jobs.first().cloned();
        assert!(job.as_ref().is_none_or(|job| job.tasks_len >= 2));
        job
    }

    fn prune(&self) {
        let mut queue = lock(&self.queue);
        queue.jobs.retain(|job| !job.is_exhausted());
        self.jobs_len.store(queue.jobs.len(), Ordering::Release);
    }

    fn try_take_job(&self) -> Option<Arc<Job>> {
        if self.jobs_len.load(Ordering::Acquire) == 0 {
            return None;
        }
        self.take_job(&mut lock(&self.queue))
    }

    fn execute(&self, job: &Job) {
        assert!(job.tasks_len >= 2);
        for _ in 0..job.tasks_len {
            let index = job.claimed.fetch_add(1, Ordering::Relaxed);
            if index >= job.tasks_len {
                return;
            }
            let task = unsafe { &*job.task };
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| task(index))) {
                let mut slot = lock(&job.panic);
                if slot.is_none() {
                    *slot = Some(payload);
                }
            }
            let finished = job.finished.fetch_add(1, Ordering::SeqCst) + 1;
            assert!(finished <= job.tasks_len);
            if finished == job.tasks_len && job.waiter_sleeping.load(Ordering::SeqCst) {
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

    fn next_job(&self, worker: usize) -> Arc<Job> {
        assert!(worker < THREADS_MAX);
        for _ in 0..SPIN_ROUNDS {
            if is_active(worker)
                && let Some(job) = self.try_take_job()
            {
                return job;
            }
            thread::yield_now();
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
            } else if idle_rounds < SPIN_ROUNDS {
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
        job.waiter_sleeping.store(true, Ordering::SeqCst);
        if job.is_finished() {
            return;
        }
        queue.sleepers += 1;
        queue = wait(&self.wake, queue);
        queue.sleepers -= 1;
        assert!(queue.workers_len < THREADS_MAX);
    }
}
