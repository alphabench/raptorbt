//! `Stream::update` allocates nothing per bar once the stream is built.
//!
//! A `#[global_allocator]` is per binary, so this lives in its own test file. It counts on the
//! calling thread only (a const thread-local, which itself never allocates), so the test
//! harness and other threads cannot add to the count. Run with `--no-default-features`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use raptorbt::quant::eval::{Bar, Plan, Stream};
use raptorbt::quant::lang::{compile, Env};

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

#[test]
fn update_allocates_nothing_after_construction() {
    let sources = [
        "ema(sma(close, 5) - sma(close, 20), 9) > 0 and not (volume < 100)",
        "(close - open) / (high - low) + sma(volume, 7) * 2",
        "close ** 2 - ema(close, 3)",
    ];
    for src in sources {
        let program = compile(src, &Env::default()).unwrap();
        let plan = Plan::new(&program).unwrap();
        let mut stream = Stream::new(&plan).unwrap();

        // The counter sees this thread's allocations; prove it counts before relying on it.
        let before = allocs();
        let probe = vec![0u8; 16];
        assert!(allocs() > before, "the counting allocator is not installed");
        drop(probe);

        let bars: Vec<Bar> = (0..2000)
            .map(|i| {
                let x = i as f64;
                // Every seventh bar carries NaN, every eleventh a zero range (division by 0).
                let close = if i % 7 == 3 { f64::NAN } else { 100.0 + (x * 0.37).sin() * 5.0 };
                let (high, low) =
                    if i % 11 == 5 { (close, close) } else { (close + 1.0, close - 1.0) };
                Bar { open: close - 0.5, high, low, close, volume: 50.0 + (i % 100) as f64 }
            })
            .collect();

        let start = allocs();
        let mut known = 0usize;
        for bar in &bars {
            if !stream.update(bar, &[]).unwrap()[0].is_nan() {
                known += 1;
            }
        }
        let used = allocs() - start;
        assert_eq!(used, 0, "{src}: {used} allocations over {} bars", bars.len());
        assert!(stream.ready() && known > 0, "{src}: the stream never produced a value");
    }
}
