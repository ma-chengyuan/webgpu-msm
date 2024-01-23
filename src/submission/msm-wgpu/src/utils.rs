#![allow(dead_code)]

#[cfg(target = "wasm32-unknown-unknown")]
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[cfg(target = "wasm32-unknown-unknown")]
pub fn time_begin(label: &str) {
    use web_sys::console;
    console::time_with_label(label);
}

#[cfg(target = "wasm32-unknown-unknown")]
pub fn time_end(label: &str) {
    use web_sys::console;
    console::time_end_with_label(label);
}

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::Instant,
};

static START_TIMESTAMP: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();

#[cfg(not(target = "wasm32-unknown-unknown"))]
pub fn time_begin(label: &str) {
    let timestamps = START_TIMESTAMP.get_or_init(|| Mutex::new(HashMap::new()));
    let mut timestamps = timestamps.lock().unwrap();
    if timestamps.contains_key(label) {
        panic!("duplicate label: {}", label);
    }
    timestamps.insert(label.to_string(), Instant::now());
}

#[cfg(not(target = "wasm32-unknown-unknown"))]
pub fn time_end(label: &str) {
    let timestamps = START_TIMESTAMP.get_or_init(|| Mutex::new(HashMap::new()));
    let mut timestamps = timestamps.lock().unwrap();
    match timestamps.get(label) {
        None => panic!("unknown label: {}", label),
        Some(start) => {
            let elapsed = start.elapsed();
            log::info!("{}: {}ms", label, elapsed.as_millis());
            timestamps.remove(label);
        }
    }
}
