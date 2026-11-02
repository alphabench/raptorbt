//! Known defects that need Rust-level access. Each is `#[ignore]`d so a fix
//! surfaces (run with `--ignored` to see it fail until fixed).

use raptorbt::accounts::MarginBook;
use std::collections::BTreeSet;

/// `MarginBook::total_locked` sums `HashMap` values. Rust's
/// `HashMap` seeds `RandomState` per instance, so the same inserts iterate in
/// different orders and the f64 sum differs in the last bit.
///
/// Expected: identical books report identical totals (one bit pattern).
/// Observed: several distinct bit patterns across 64 identically-built books.
#[test]
fn total_locked_is_order_independent() {
    let mut patterns = BTreeSet::new();
    for _ in 0..64 {
        let mut book = MarginBook::default();
        for k in 1..60u64 {
            // Values with long mantissas so reordering changes rounding.
            book.lock(k, (k as f64 * 0.7919).fract() * 3.1 / 3.0 + 1.0 / (k as f64 + 0.3));
        }
        patterns.insert(book.total_locked().to_bits());
    }
    assert_eq!(patterns.len(), 1, "distinct total_locked bit patterns: {}", patterns.len());
}
