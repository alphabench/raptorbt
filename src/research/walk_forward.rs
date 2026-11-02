//! Rolling in-sample / out-of-sample windows over a run of trading days.

use std::ops::Range;

use crate::core::error::{RaptorError, Result};

/// One window: index ranges into the day list for its training and test days.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowRange {
    pub index: usize,
    pub train: Range<usize>,
    pub test: Range<usize>,
}

/// `train` days then `test` days, stepped by `step` (the test length by default, so
/// the test windows tile). Counts are of days; the day labels stay with the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkForward {
    pub days: usize,
    pub train: usize,
    pub test: usize,
    pub step: usize,
}

impl WalkForward {
    pub fn new(days: usize, train: i64, test: i64, step: Option<i64>) -> Result<Self> {
        if train < 1 || test < 1 {
            return Err(RaptorError::invalid_parameter("train and test are at least one day"));
        }
        let step = step.unwrap_or(test);
        if step < 1 {
            return Err(RaptorError::invalid_parameter("step is at least one day"));
        }
        let size = |v: i64| usize::try_from(v).unwrap_or(usize::MAX);
        Ok(Self { days, train: size(train), test: size(test), step: size(step) })
    }

    /// One window: no training days, every day out of sample (a plain sweep).
    pub fn single(days: usize) -> Self {
        Self { days, train: 0, test: days, step: days }
    }

    pub fn windows(&self) -> Vec<WindowRange> {
        let mut out = Vec::new();
        let mut start = 0usize;
        while let Some(end) = start.checked_add(self.train).and_then(|v| v.checked_add(self.test)) {
            if end > self.days {
                break;
            }
            let mid = start + self.train;
            out.push(WindowRange { index: out.len(), train: start..mid, test: mid..end });
            if self.step == 0 {
                break; // `single` over no days: one empty window
            }
            start += self.step;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_tile_the_calendar() {
        let ws = WalkForward::new(20, 5, 3, None).unwrap().windows();
        assert_eq!(ws.len(), 5);
        assert_eq!((ws[0].train.clone(), ws[0].test.clone()), (0..5, 5..8));
        assert_eq!((ws[1].train.clone(), ws[1].test.clone()), (3..8, 8..11));
        assert_eq!(ws[4].test.end, 20);
        let single = WalkForward::single(20).windows();
        assert_eq!(single, vec![WindowRange { index: 0, train: 0..0, test: 0..20 }]);
        assert!(WalkForward::new(20, 0, 3, None).is_err());
        assert!(WalkForward::new(20, 5, 3, Some(0)).is_err());
        assert_eq!(WalkForward::new(20, 5, 3, Some(10)).unwrap().windows().len(), 2);
        assert!(WalkForward::new(4, 5, 3, None).unwrap().windows().is_empty());
    }
}
