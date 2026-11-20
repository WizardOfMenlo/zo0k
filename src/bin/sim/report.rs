use crate::prng::Prng;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::time::Instant;

static PANIC_MESSAGE: Mutex<Option<String>> = Mutex::new(None);

pub struct Failure {
    pub check: String,
    pub detail: String,
}

pub struct Ctx {
    pub prng: Prng,
    pub verbose: bool,
    pub checks_run: usize,
}

pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let message = info.to_string();
        if let Ok(mut slot) = PANIC_MESSAGE.lock() {
            *slot = Some(message);
        }
    }));
    assert!(PANIC_MESSAGE.lock().is_ok());
}

fn take_panic_message() -> String {
    let taken = PANIC_MESSAGE.lock().ok().and_then(|mut slot| slot.take());
    let message = taken.unwrap_or_else(|| "a panic without a message".to_string());
    assert!(!message.is_empty());
    message.replace('\n', " ")
}

pub fn guarded<T>(body: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let outcome = catch_unwind(AssertUnwindSafe(body));
    match outcome {
        Ok(result) => result,
        Err(_) => Err(format!("panic: {}", take_panic_message())),
    }
}

pub fn ensure(condition: bool, detail: impl FnOnce() -> String) -> Result<(), String> {
    if condition { Ok(()) } else { Err(detail()) }
}

impl Ctx {
    pub fn new(seed: u64, verbose: bool) -> Self {
        let ctx = Self { prng: Prng::new(seed), verbose, checks_run: 0 };
        assert_eq!(ctx.checks_run, 0);
        ctx
    }

    pub fn check<T>(
        &mut self,
        name: &'static str,
        body: impl FnOnce(&mut Prng) -> Result<T, String>,
    ) -> Result<T, Failure> {
        assert!(!name.is_empty());
        let start = Instant::now();
        let prng = &mut self.prng;
        let outcome = guarded(|| body(prng));
        self.checks_run += 1;
        if self.verbose {
            let status = if outcome.is_ok() { "ok  " } else { "FAIL" };
            println!("  {status} {name} ({:.1} ms)", start.elapsed().as_secs_f64() * 1e3);
        }
        outcome.map_err(|detail| Failure { check: name.to_string(), detail })
    }

    pub fn note(&self, text: impl FnOnce() -> String) {
        if self.verbose {
            println!("  {}", text());
        }
    }
}
