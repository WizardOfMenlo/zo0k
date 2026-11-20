use std::fs::File;
use std::io::Read;

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "illumos",
    target_os = "solaris"
))]
mod os {
    const CALL_LEN_MAX: usize = 256;

    unsafe extern "C" {
        fn getentropy(buffer: *mut std::ffi::c_void, len_bytes: usize) -> std::ffi::c_int;
    }

    pub fn fill(bytes: &mut [u8]) -> bool {
        for chunk in bytes.chunks_mut(CALL_LEN_MAX) {
            assert!(chunk.len() <= CALL_LEN_MAX);
            let status = unsafe { getentropy(chunk.as_mut_ptr().cast(), chunk.len()) };
            if status != 0 {
                return false;
            }
        }
        true
    }
}

#[cfg(windows)]
mod os {
    #[link(name = "bcryptprimitives", kind = "raw-dylib")]
    unsafe extern "system" {
        fn ProcessPrng(data: *mut u8, len_bytes: usize) -> i32;
    }

    pub fn fill(bytes: &mut [u8]) -> bool {
        assert!(!bytes.is_empty());
        let status = unsafe { ProcessPrng(bytes.as_mut_ptr(), bytes.len()) };
        status != 0
    }
}

#[cfg(not(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "illumos",
    target_os = "solaris"
)))]
mod os {
    pub fn fill(_bytes: &mut [u8]) -> bool {
        false
    }
}

fn fill_from_device(bytes: &mut [u8]) -> bool {
    assert!(!bytes.is_empty());
    if !cfg!(unix) {
        return false;
    }
    match File::open("/dev/urandom") {
        Ok(mut device) => device.read_exact(bytes).is_ok(),
        Err(_) => false,
    }
}

pub fn fill(bytes: &mut [u8]) -> bool {
    assert!(!bytes.is_empty());
    if os::fill(bytes) {
        return true;
    }
    fill_from_device(bytes)
}
