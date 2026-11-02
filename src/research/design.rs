//! Points in parameter space: the full grid in key
//! order, or `n` random points drawn by SplitMix64 from a declared seed. The values
//! themselves stay with the caller; a design here is a list of draws per key.

use crate::core::error::{RaptorError, Result};

use super::SplitMix64;

/// What one parameter may take, keys already in sorted order.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamSpace {
    /// A list of `n` choices; a draw picks an index.
    Choices(usize),
    /// A `(lo, hi)` range. Both bounds integers: the draw is rounded half to even to an
    /// integer, as Python's `int(round(v))` does.
    IntRange { lo: i64, hi: i64 },
    /// A `(lo, hi)` range with at least one float bound.
    FloatRange { lo: f64, hi: f64 },
}

/// One parameter's value in one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamDraw {
    Choice(usize),
    Int(i64),
    Float(f64),
}

pub struct Design;

impl Design {
    /// The full product over sorted keys, the last key varying fastest. Each point holds one choice index per key.
    pub fn grid(sizes: &[usize]) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = vec![Vec::new()];
        for &n in sizes {
            let mut next = Vec::with_capacity(out.len() * n);
            for p in &out {
                for v in 0..n {
                    let mut q = p.clone();
                    q.push(v);
                    next.push(q);
                }
            }
            out = next;
        }
        out
    }

    /// `n` points; within a point the keys draw in order, one `uniform` per range and one
    /// `below` per choice list.
    pub fn random(space: &[ParamSpace], n: usize, seed: u64) -> Result<Vec<Vec<ParamDraw>>> {
        if n > 0 && space.iter().any(|s| matches!(s, ParamSpace::Choices(0))) {
            return Err(RaptorError::invalid_parameter("a parameter has no choices"));
        }
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut p = Vec::with_capacity(space.len());
            for s in space {
                p.push(match *s {
                    ParamSpace::Choices(k) => ParamDraw::Choice(rng.below(k as u64) as usize),
                    ParamSpace::IntRange { lo, hi } => {
                        let u = rng.uniform();
                        // Python: lo + (hi - lo) * u, the difference an exact integer
                        let v = lo as f64 + (hi as i128 - lo as i128) as f64 * u;
                        ParamDraw::Int(v.round_ties_even() as i64)
                    }
                    ParamSpace::FloatRange { lo, hi } => {
                        let u = rng.uniform();
                        ParamDraw::Float(lo + (hi - lo) * u)
                    }
                });
            }
            out.push(p);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_the_product_in_key_order() {
        let g = Design::grid(&[3, 2]);
        assert_eq!(g, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1], vec![2, 0], vec![2, 1]]);
        assert_eq!(Design::grid(&[]), vec![Vec::<usize>::new()]);
        assert!(Design::grid(&[2, 0]).is_empty());
    }

    // Pinned reference draws for `Design.random({"k": [1.0, 2.0], "n": (5, 20),
    // "x": (0.0, 1.0)}, 4, seed=7).points`.
    #[test]
    fn random_draws_match_reference() {
        let space = [
            ParamSpace::Choices(2),
            ParamSpace::IntRange { lo: 5, hi: 20 },
            ParamSpace::FloatRange { lo: 0.0, hi: 1.0 },
        ];
        let pts = Design::random(&space, 4, 7).unwrap();
        assert_eq!(pts, PINNED.iter().map(|p| p.to_vec()).collect::<Vec<_>>());
        assert_eq!(pts, Design::random(&space, 4, 7).unwrap());
        assert_ne!(pts, Design::random(&space, 4, 8).unwrap());
        assert!(Design::random(&[ParamSpace::Choices(0)], 1, 7).is_err());
    }

    const PINNED: [[ParamDraw; 3]; 4] = [
        [ParamDraw::Choice(1), ParamDraw::Int(5), ParamDraw::Float(0.9007606806068834)],
        [ParamDraw::Choice(1), ParamDraw::Int(12), ParamDraw::Float(0.24943152228274335)],
        [ParamDraw::Choice(0), ParamDraw::Int(10), ParamDraw::Float(0.13425829880844864)],
        [ParamDraw::Choice(1), ParamDraw::Int(7), ParamDraw::Float(0.9598740765730915)],
    ];
}
