//! A sampled result curve that grows without moving the samples it holds.
//!
//! A `Vec` that doubles copies every sample it holds when it outgrows its
//! buffer, so on a feed that never stops one push in a power of two pays for
//! all the pushes before it. Past [`BLOCK`] samples a curve fills fixed-size
//! blocks instead and is made contiguous only when the result is read
//! ([`Curve::into_vec`]). A run that knows its length reserves it up front
//! ([`Curve::reserve`]) and stays one contiguous buffer, which `into_vec`
//! hands over without copying.

/// Samples per block once a curve outgrows its first buffer. While a curve is
/// smaller it grows as a `Vec` does, so a doubling copies at most half a block.
pub(crate) const BLOCK: usize = 4096;

/// Samples in order: every block of `full`, then `tail`.
#[derive(Debug, Clone, Default)]
pub(crate) struct Curve<T> {
    /// Earlier samples, a block each. None is empty.
    full: Vec<Vec<T>>,
    /// Samples held in `full`.
    held: usize,
    /// The block being filled.
    tail: Vec<T>,
}

impl<T: Copy> Curve<T> {
    /// Room for `samples` in one buffer, for a run that knows its length.
    /// Call before the first sample; it does nothing afterwards.
    pub(crate) fn reserve(&mut self, samples: usize) {
        if self.held == 0 && self.tail.is_empty() {
            self.tail.reserve_exact(samples);
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.held + self.tail.len()
    }

    #[inline]
    pub(crate) fn push(&mut self, value: T) {
        if self.tail.len() == self.tail.capacity() && self.tail.capacity() >= BLOCK {
            self.seal_tail();
        }
        self.tail.push(value);
    }

    #[cold]
    #[inline(never)]
    fn seal_tail(&mut self) {
        let sealed = std::mem::replace(&mut self.tail, Vec::with_capacity(BLOCK));
        self.held += sealed.len();
        self.full.push(sealed);
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<T> {
        if let Some(value) = self.tail.pop() {
            return Some(value);
        }
        let block = self.full.last_mut()?;
        let value = block.pop();
        self.held -= 1;
        if block.is_empty() {
            self.full.pop();
        }
        value
    }

    /// The sample `k` places from the end: `back(0)` is the last.
    #[inline]
    pub(crate) fn back(&self, k: usize) -> Option<T> {
        let t = self.tail.len();
        if k < t {
            return Some(self.tail[t - 1 - k]);
        }
        let mut k = k - t;
        for block in self.full.iter().rev() {
            if k < block.len() {
                return Some(block[block.len() - 1 - k]);
            }
            k -= block.len();
        }
        None
    }

    #[inline]
    pub(crate) fn last(&self) -> Option<T> {
        self.back(0)
    }

    #[inline]
    pub(crate) fn last_mut(&mut self) -> Option<&mut T> {
        match self.tail.last_mut() {
            Some(last) => Some(last),
            None => self.full.last_mut().and_then(|block| block.last_mut()),
        }
    }

    /// Drop the oldest `count` samples.
    pub(crate) fn drain_front(&mut self, mut count: usize) {
        while count > 0 {
            match self.full.first().map(Vec::len) {
                Some(first) if first <= count => {
                    self.full.remove(0);
                    self.held -= first;
                    count -= first;
                }
                Some(_) => {
                    self.full[0].drain(..count);
                    self.held -= count;
                    count = 0;
                }
                None => {
                    self.tail.drain(..count);
                    count = 0;
                }
            }
        }
    }

    /// Heap held, at capacity.
    pub(crate) fn heap_bytes(&self) -> usize {
        let samples: usize =
            self.full.iter().map(Vec::capacity).sum::<usize>() + self.tail.capacity();
        samples * std::mem::size_of::<T>() + self.full.capacity() * std::mem::size_of::<Vec<T>>()
    }

    /// The samples as one buffer. A curve that never left its first buffer
    /// is handed over as it is.
    pub(crate) fn into_vec(self) -> Vec<T> {
        if self.full.is_empty() {
            return self.tail;
        }
        let mut out = Vec::with_capacity(self.len());
        for block in self.full {
            out.extend_from_slice(&block);
        }
        out.extend_from_slice(&self.tail);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every operation applied to a curve and to a `Vec` alongside it.
    fn check(ops: impl IntoIterator<Item = (u8, usize)>) {
        let mut curve = Curve::default();
        let mut vec: Vec<u64> = Vec::new();
        let mut next = 0u64;
        for (op, arg) in ops {
            match op {
                0 => {
                    curve.push(next);
                    vec.push(next);
                    next += 1;
                }
                1 => assert_eq!(curve.pop(), vec.pop()),
                2 => {
                    if let (Some(a), Some(b)) = (curve.last_mut(), vec.last_mut()) {
                        *a += 1_000_000;
                        *b += 1_000_000;
                    }
                }
                _ => {
                    let count = arg.min(vec.len());
                    curve.drain_front(count);
                    vec.drain(..count);
                }
            }
            assert_eq!(curve.len(), vec.len());
            assert_eq!(curve.last(), vec.last().copied());
            for k in [0, 1, BLOCK - 1, BLOCK, BLOCK + 1] {
                let want = vec.len().checked_sub(k + 1).map(|i| vec[i]);
                assert_eq!(curve.back(k), want, "back({k})");
            }
        }
        assert_eq!(curve.into_vec(), vec);
    }

    #[test]
    fn a_curve_reads_as_the_vec_it_replaces() {
        // Pushes across several block boundaries, a pop and a restate at each,
        // the way a second event at one timestamp replaces the sample.
        let mut ops = Vec::new();
        for i in 0..3 * BLOCK + 17 {
            ops.push((0, 0));
            if i % (BLOCK - 1) == 0 || i % BLOCK == 0 {
                ops.extend([(1, 0), (0, 0), (2, 0)]);
            }
        }
        // Pops that empty the tail reach back into the last block.
        ops.extend(std::iter::repeat_n((1, 0), BLOCK + 5));
        ops.extend(std::iter::repeat_n((0, 0), BLOCK + 5));
        check(ops.clone());
        // A window: the front drained by part blocks, whole blocks and more.
        ops.extend([(3, 5), (3, BLOCK), (3, BLOCK + 3), (0, 0), (3, usize::MAX)]);
        check(ops);
    }

    #[test]
    fn a_reserved_curve_stays_one_buffer() {
        let mut curve = Curve::default();
        curve.reserve(3 * BLOCK);
        for i in 0..3 * BLOCK as u64 {
            curve.push(i);
        }
        assert!(curve.full.is_empty());
        let ptr = curve.tail.as_ptr();
        let out = curve.into_vec();
        assert_eq!(out.as_ptr(), ptr, "handed over, not copied");
        assert_eq!(out, (0..3 * BLOCK as u64).collect::<Vec<_>>());
    }
}
