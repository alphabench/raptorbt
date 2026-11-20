//! Appending a sample to an unbounded curve never moves the samples already
//! taken: a book on a feed that never stops pays the same for its millionth
//! event as for its first.
//!
//! A `#[global_allocator]` is per binary, so this lives in its own test file. It records, on the
//! calling thread only, the largest buffer handed to `realloc` (the bytes a reallocation may
//! copy). Run with `--no-default-features`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use raptorbt::accounts::AccountMode;
use raptorbt::core::types::{BacktestConfig, Direction};
use raptorbt::portfolio::session::{ScheduleData, ScheduleEntry};
use raptorbt::portfolio::{KernelBar, PortfolioBook, PositionPolicy, StepInput};

/// One block of a curve, in bytes (`BLOCK` samples of eight bytes in `portfolio/curve.rs`). A
/// curve smaller than that grows as a `Vec` does, so a reallocation may copy up to half of it.
const ONE_BLOCK: usize = 4096 * 8;

struct Counting;

thread_local! {
    /// Largest old size passed to `realloc` on this thread.
    static MOVED: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = MOVED.try_with(|m| m.set(m.get().max(layout.size())));
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Feed one book `n` bars at distinct instants and return the largest buffer
/// reallocated while it ran.
fn largest_move(window: Option<usize>, n: usize) -> usize {
    let config = BacktestConfig { fees: 0.0, ..BacktestConfig::default() };
    let mut book = PortfolioBook::new(config, AccountMode::Cash);
    book.add_instrument("AAA".into(), Direction::Long, None, None, PositionPolicy::Net);
    book.set_curve_window(window);
    MOVED.with(|m| m.set(0));
    for i in 0..n {
        let close = 100.0 + (i % 17) as f64 * 0.01;
        let bar = KernelBar {
            timestamp: i as i64 * 1_000,
            open: close,
            high: close,
            low: close,
            close,
            volume: 1.0,
        };
        let entry = ScheduleEntry { instrument: 0, local_idx: i, data: ScheduleData::Bar(bar) };
        book.apply(&entry, None, StepInput::default());
    }
    let moved = MOVED.with(Cell::get);
    assert!(book.applied() == n);
    moved
}

#[test]
fn an_unbounded_curve_never_reallocates_the_samples_it_holds() {
    let n = 300_000;
    // The same book with its curves held to one sample: whatever else the book
    // reallocates per event, the curves are not part of it.
    let control = largest_move(Some(1), n);
    let unbounded = largest_move(None, n);
    assert!(
        unbounded <= control.max(ONE_BLOCK),
        "a {n}-sample curve reallocated a {unbounded}-byte buffer (control {control})"
    );
}
