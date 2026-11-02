//! Heap accounting for the runtime's data structures.

/// Exact count of the heap bytes a value owns.
///
/// Counts requested payload sizes (element size times length) plus the
/// two-word reference-count header of every `Arc` allocation. Allocator
/// rounding and bookkeeping are not included. A value shared through several
/// `Arc` clones is counted in full by each holder, so sum over distinct
/// owners, not over readers.
pub trait HeapSize {
    fn heap_bytes(&self) -> usize;
}

/// Strong and weak counters that precede an `Arc<[T]>` payload.
pub(crate) const ARC_HEADER: usize = 2 * std::mem::size_of::<usize>();

/// Heap bytes of an `Arc<[T]>`: header plus elements.
pub(crate) fn arc_slice_bytes<T>(slice: &std::sync::Arc<[T]>) -> usize {
    ARC_HEADER + std::mem::size_of_val::<[T]>(slice)
}
