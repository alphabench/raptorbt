//! SplitMix64, draw for draw the standard generator.

/// Steele, Lea and Flood's SplitMix64. `uniform` is the top 53 bits over 2^53 and
/// `below(n)` is a plain modulo, so a seed gives the same stream in any implementation
/// that computes them the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// The seed is masked to 64 bits (`seed & (2**64 - 1)`); a caller with a
    /// negative seed passes its two's-complement bits.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// An integer in [0, n). `n` must be at least 1.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pinned reference draws.
    #[test]
    fn draws_match_reference() {
        let mut r = SplitMix64::new(0);
        assert_eq!(
            [r.next_u64(), r.next_u64(), r.next_u64()],
            [16294208416658607535, 7960286522194355700, 487617019471545679]
        );
        let mut r = SplitMix64::new(1234567);
        assert_eq!(
            [r.next_u64(), r.next_u64(), r.next_u64()],
            [6457827717110365317, 3203168211198807973, 9817491932198370423]
        );
        assert_eq!(SplitMix64::new(7).uniform(), 0.3898297483912715);
        // SplitMix64(-1): the seed is masked to 64 bits
        assert_eq!(SplitMix64::new(u64::MAX).next_u64(), 16490336266968443936);
        let mut r = SplitMix64::new(42);
        for _ in 0..1000 {
            assert!(r.below(7) < 7);
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
