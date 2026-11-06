//! One kernel, two modes. A kernel's `step` sees the current inputs and a read-only history of
//! the bars before them; history is held outside the kernel, so batch (a prefix slice) and
//! streaming (a preallocated ring) run the same `step` in the same operation order.

use std::array;

/// Read-only past of `I` input series. `ago(input, k)` is the value `k` bars before the
/// current one (`k >= 1`); a bar that does not exist, or `k == 0`, reads NaN. There is no way
/// to ask for the current bar or a later one.
pub trait Hist<const I: usize> {
    fn ago(&self, input: usize, k: usize) -> f64;
}

pub trait Kernel<const I: usize, const O: usize>: Send + Sync {
    /// The largest `k` this kernel passes to `ago`. Sizes the streaming ring.
    fn lookback(&self) -> usize;
    fn step<H: Hist<I>>(&mut self, x: [f64; I], h: &H) -> [f64; O];
    /// Heap the kernel owns beyond its own size, counted at capacity (windows, rings, deques).
    /// Required, so a new kernel cannot forget it: the bank's memory report reads it.
    fn heap_bytes(&self) -> usize;
}

/// The flat register file of a streaming program: one `f64` per node output.
#[derive(Debug, Clone, Default)]
pub struct Regs {
    v: Vec<f64>,
}

impl Regs {
    pub fn new(n: usize) -> Self {
        Self { v: vec![f64::NAN; n] }
    }

    pub fn get(&self, r: usize) -> f64 {
        self.v[r]
    }

    pub fn set(&mut self, r: usize, value: f64) {
        self.v[r] = value;
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.v
    }

    /// Registers allocated.
    pub fn capacity(&self) -> usize {
        self.v.capacity()
    }
}

/// The only dyn boundary: one virtual call per node, per column or per bar.
pub trait Node: Send + Sync {
    /// Run a whole column on a fresh node. `ins` and `outs` have the node's arity, all of one
    /// length.
    fn run_batch(&mut self, ins: &[&[f64]], outs: &mut [&mut [f64]]);
    /// Read the node's input registers, write its output registers.
    fn step(&mut self, regs: &mut Regs);
    /// Heap the node owns: its own allocation (the box it lives in), its history ring and its
    /// kernel's state.
    fn state_bytes(&self) -> usize;
}

/// Batch history: the prefix `data[..i]` of each input. The current bar and the future are not
/// in the slice, so they cannot be addressed.
#[derive(Debug, Clone, Copy)]
pub struct SliceHist<'a, const I: usize> {
    prefix: [&'a [f64]; I],
}

impl<'a, const I: usize> SliceHist<'a, I> {
    /// History as seen from bar `i`. Panics when `i` exceeds a column.
    pub fn new(cols: [&'a [f64]; I], i: usize) -> Self {
        Self { prefix: array::from_fn(|j| &cols[j][..i]) }
    }
}

impl<const I: usize> Hist<I> for SliceHist<'_, I> {
    #[inline(always)]
    fn ago(&self, input: usize, k: usize) -> f64 {
        let s = self.prefix[input];
        if k == 0 || k > s.len() {
            f64::NAN
        } else {
            s[s.len() - k]
        }
    }
}

/// Streaming history: a ring of the last `cap` bars of `I` inputs, allocated once. `cap` is a
/// power of two at least the kernel's lookback (the period is a runtime value, so the size is
/// fixed at construction rather than in the type).
#[derive(Debug, Clone)]
pub struct RingHist<const I: usize> {
    buf: Box<[f64]>,
    mask: usize,
    n: usize,
}

impl<const I: usize> RingHist<I> {
    pub fn new(lookback: usize) -> Self {
        let cap = lookback.max(1).next_power_of_two();
        Self { buf: vec![f64::NAN; cap * I].into_boxed_slice(), mask: cap - 1, n: 0 }
    }

    pub fn push(&mut self, x: [f64; I]) {
        let base = (self.n & self.mask) * I;
        self.buf[base..base + I].copy_from_slice(&x);
        self.n += 1;
    }

    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.buf)
    }
}

impl<const I: usize> Hist<I> for RingHist<I> {
    #[inline(always)]
    fn ago(&self, input: usize, k: usize) -> f64 {
        debug_assert!(k <= self.mask + 1, "ago({k}) beyond the ring");
        if k == 0 || k > self.n || k > self.mask + 1 {
            return f64::NAN;
        }
        self.buf[((self.n - k) & self.mask) * I + input]
    }
}

/// Binds a kernel to registers and gives it both modes.
pub struct Adapter<K, const I: usize, const O: usize> {
    kernel: K,
    in_regs: [usize; I],
    out_regs: [usize; O],
    ring: RingHist<I>,
}

impl<K: Kernel<I, O>, const I: usize, const O: usize> Adapter<K, I, O> {
    pub fn new(kernel: K, in_regs: [usize; I], out_regs: [usize; O]) -> Self {
        assert!(I >= 1, "a kernel reads at least one input");
        let ring = RingHist::new(kernel.lookback());
        Self { kernel, in_regs, out_regs, ring }
    }
}

impl<K: Kernel<I, O>, const I: usize, const O: usize> Node for Adapter<K, I, O> {
    fn run_batch(&mut self, ins: &[&[f64]], outs: &mut [&mut [f64]]) {
        assert_eq!(ins.len(), I, "input arity");
        assert_eq!(outs.len(), O, "output arity");
        let cols: [&[f64]; I] = array::from_fn(|j| ins[j]);
        let n = cols[0].len();
        assert!(cols.iter().all(|c| c.len() == n) && outs.iter().all(|o| o.len() == n));
        for i in 0..n {
            let h = SliceHist::new(cols, i);
            let y = self.kernel.step(array::from_fn(|j| cols[j][i]), &h);
            for (o, v) in outs.iter_mut().zip(y) {
                o[i] = v;
            }
        }
    }

    fn step(&mut self, regs: &mut Regs) {
        let x: [f64; I] = array::from_fn(|j| regs.get(self.in_regs[j]));
        let y = self.kernel.step(x, &self.ring);
        self.ring.push(x);
        for (r, v) in self.out_regs.iter().zip(y) {
            regs.set(*r, v);
        }
    }

    fn state_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.ring.heap_bytes() + self.kernel.heap_bytes()
    }
}

/// Bit pattern with every NaN made one NaN, for comparing outputs.
pub fn canon_bits(x: f64) -> u64 {
    if x.is_nan() {
        f64::NAN.to_bits()
    } else {
        x.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_hist_cannot_address_the_current_bar_or_later() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        for i in 0..data.len() {
            let h = SliceHist::<1>::new([&data], i);
            assert!(h.ago(0, 0).is_nan(), "k=0 would be bar {i}");
            for k in 1..=i {
                assert_eq!(h.ago(0, k), data[i - k]);
            }
            // Past the start reads NaN, never wraps to a later bar.
            assert!(h.ago(0, i + 1).is_nan());
            assert!(h.ago(0, i + 3).is_nan());
        }
    }

    #[test]
    fn ring_matches_slice_history() {
        let data: Vec<f64> = (0..40).map(|v| v as f64 * 1.5).collect();
        let mut ring = RingHist::<1>::new(7);
        for i in 0..data.len() {
            let s = SliceHist::<1>::new([&data], i);
            for k in 0..=8 {
                assert_eq!(canon_bits(ring.ago(0, k)), canon_bits(s.ago(0, k)), "i={i} k={k}");
            }
            ring.push([data[i]]);
        }
    }
}
