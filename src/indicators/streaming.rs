//! Incremental (streaming) indicator cores.
//!
//! One `update` per record, O(1) or O(period) per step — no growing-window
//! recompute. Each core reproduces its batch counterpart's values exactly
//! (same seeding, same smoothing; equivalence-tested), so a strategy using
//! registered streaming indicators sees the same numbers it would have
//! precomputed with the array functions.

use std::collections::VecDeque;

/// Common streaming surface: feed values, read the latest output.
pub trait StreamingIndicator: std::fmt::Debug + Send {
    /// Feed one value; returns the indicator value once warm, else `None`.
    fn update(&mut self, value: f64) -> Option<f64>;
    /// Latest value, if warm.
    fn value(&self) -> Option<f64>;
    /// Whether the warmup period has completed.
    fn initialized(&self) -> bool {
        self.value().is_some()
    }
    /// Forget all state.
    fn reset(&mut self);
}

/// Rolling window shared by the windowed cores.
#[derive(Debug, Clone)]
struct Window {
    period: usize,
    values: VecDeque<f64>,
    sum: f64,
}

impl Window {
    fn new(period: usize) -> Self {
        Self { period, values: VecDeque::with_capacity(period + 1), sum: 0.0 }
    }

    /// Push a value; returns `true` once the window is full.
    fn push(&mut self, value: f64) -> bool {
        self.values.push_back(value);
        self.sum += value;
        if self.values.len() > self.period {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
            }
        }
        self.values.len() == self.period
    }

    fn mean(&self) -> f64 {
        self.sum / self.period as f64
    }

    fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }
}

/// Simple moving average.
#[derive(Debug)]
pub struct StreamingSma {
    window: Window,
    latest: Option<f64>,
}

impl StreamingSma {
    pub fn new(period: usize) -> Self {
        Self { window: Window::new(period.max(1)), latest: None }
    }
}

impl StreamingIndicator for StreamingSma {
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.window.push(value) {
            self.latest = Some(self.window.mean());
        }
        self.latest
    }

    fn value(&self) -> Option<f64> {
        self.latest
    }

    fn reset(&mut self) {
        self.window.reset();
        self.latest = None;
    }
}

/// Exponential moving average, SMA-seeded like the batch function.
#[derive(Debug)]
pub struct StreamingEma {
    period: usize,
    alpha: f64,
    seed: Window,
    latest: Option<f64>,
}

impl StreamingEma {
    pub fn new(period: usize) -> Self {
        let period = period.max(1);
        Self { period, alpha: 2.0 / (period as f64 + 1.0), seed: Window::new(period), latest: None }
    }

    /// Wilder-style smoothing (alpha = 1/period), SMA-seeded.
    pub fn wilder(period: usize) -> Self {
        let period = period.max(1);
        Self { period, alpha: 1.0 / period as f64, seed: Window::new(period), latest: None }
    }
}

impl StreamingIndicator for StreamingEma {
    fn update(&mut self, value: f64) -> Option<f64> {
        match self.latest {
            Some(prev) => {
                self.latest = Some(self.alpha * value + (1.0 - self.alpha) * prev);
            }
            None => {
                if self.seed.push(value) {
                    self.latest = Some(self.seed.mean());
                }
            }
        }
        self.latest
    }

    fn value(&self) -> Option<f64> {
        self.latest
    }

    fn reset(&mut self) {
        self.seed.reset();
        self.latest = None;
    }
}

/// Weighted moving average (linear weights, newest heaviest).
#[derive(Debug)]
pub struct StreamingWma {
    period: usize,
    values: VecDeque<f64>,
    latest: Option<f64>,
}

impl StreamingWma {
    pub fn new(period: usize) -> Self {
        let period = period.max(1);
        Self { period, values: VecDeque::with_capacity(period + 1), latest: None }
    }
}

impl StreamingIndicator for StreamingWma {
    fn update(&mut self, value: f64) -> Option<f64> {
        self.values.push_back(value);
        if self.values.len() > self.period {
            self.values.pop_front();
        }
        if self.values.len() == self.period {
            let denom = (self.period * (self.period + 1)) as f64 / 2.0;
            let weighted: f64 =
                self.values.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v).sum();
            self.latest = Some(weighted / denom);
        }
        self.latest
    }

    fn value(&self) -> Option<f64> {
        self.latest
    }

    fn reset(&mut self) {
        self.values.clear();
        self.latest = None;
    }
}

/// Rate of change over `period` steps, in percent.
#[derive(Debug)]
pub struct StreamingRoc {
    period: usize,
    values: VecDeque<f64>,
    latest: Option<f64>,
}

impl StreamingRoc {
    pub fn new(period: usize) -> Self {
        let period = period.max(1);
        Self { period, values: VecDeque::with_capacity(period + 2), latest: None }
    }
}

impl StreamingIndicator for StreamingRoc {
    fn update(&mut self, value: f64) -> Option<f64> {
        self.values.push_back(value);
        if self.values.len() > self.period + 1 {
            self.values.pop_front();
        }
        if self.values.len() == self.period + 1 {
            let base = self.values[0];
            if base != 0.0 {
                self.latest = Some((value - base) / base * 100.0);
            }
        }
        self.latest
    }

    fn value(&self) -> Option<f64> {
        self.latest
    }

    fn reset(&mut self) {
        self.values.clear();
        self.latest = None;
    }
}

/// Rolling population standard deviation.
#[derive(Debug)]
pub struct StreamingStdDev {
    window: Window,
    latest: Option<f64>,
}

impl StreamingStdDev {
    pub fn new(period: usize) -> Self {
        Self { window: Window::new(period.max(1)), latest: None }
    }
}

impl StreamingIndicator for StreamingStdDev {
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.window.push(value) {
            let mean = self.window.mean();
            let variance = self.window.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / self.window.period as f64;
            self.latest = Some(variance.sqrt());
        }
        self.latest
    }

    fn value(&self) -> Option<f64> {
        self.latest
    }

    fn reset(&mut self) {
        self.window.reset();
        self.latest = None;
    }
}

/// Wilder RSI, matching the batch seeding (SMA of the first `period`
/// changes) and continuation (alpha = 1/period).
#[derive(Debug)]
pub struct StreamingRsi {
    period: usize,
    prev: Option<f64>,
    seed_gains: Vec<f64>,
    seed_losses: Vec<f64>,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
    latest: Option<f64>,
}

impl StreamingRsi {
    pub fn new(period: usize) -> Self {
        let period = period.max(1);
        Self {
            period,
            prev: None,
            seed_gains: Vec::with_capacity(period),
            seed_losses: Vec::with_capacity(period),
            avg_gain: None,
            avg_loss: None,
            latest: None,
        }
    }

    fn rsi_of(avg_gain: f64, avg_loss: f64) -> f64 {
        if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        }
    }
}

impl StreamingIndicator for StreamingRsi {
    fn update(&mut self, value: f64) -> Option<f64> {
        let Some(prev) = self.prev.replace(value) else {
            return None;
        };
        let change = value - prev;
        let (gain, loss) = if change > 0.0 { (change, 0.0) } else { (0.0, -change) };

        match (self.avg_gain, self.avg_loss) {
            (Some(ag), Some(al)) => {
                let alpha = 1.0 / self.period as f64;
                let ag = alpha * gain + (1.0 - alpha) * ag;
                let al = alpha * loss + (1.0 - alpha) * al;
                self.avg_gain = Some(ag);
                self.avg_loss = Some(al);
                self.latest = Some(Self::rsi_of(ag, al));
            }
            _ => {
                self.seed_gains.push(gain);
                self.seed_losses.push(loss);
                if self.seed_gains.len() == self.period {
                    let ag = self.seed_gains.iter().sum::<f64>() / self.period as f64;
                    let al = self.seed_losses.iter().sum::<f64>() / self.period as f64;
                    self.avg_gain = Some(ag);
                    self.avg_loss = Some(al);
                    self.latest = Some(Self::rsi_of(ag, al));
                }
            }
        }
        self.latest
    }

    fn value(&self) -> Option<f64> {
        self.latest
    }

    fn reset(&mut self) {
        self.prev = None;
        self.seed_gains.clear();
        self.seed_losses.clear();
        self.avg_gain = None;
        self.avg_loss = None;
        self.latest = None;
    }
}

/// Wilder ATR over (high, low, close) bars, matching the batch function.
#[derive(Debug)]
pub struct StreamingAtr {
    period: usize,
    prev_close: Option<f64>,
    seed: Vec<f64>,
    latest: Option<f64>,
}

impl StreamingAtr {
    pub fn new(period: usize) -> Self {
        let period = period.max(1);
        Self { period, prev_close: None, seed: Vec::with_capacity(period), latest: None }
    }

    /// Feed one bar; returns ATR once warm.
    pub fn update_bar(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = match self.prev_close {
            None => high - low,
            Some(pc) => (high - low).max((high - pc).abs()).max((low - pc).abs()),
        };
        self.prev_close = Some(close);

        match self.latest {
            Some(prev) => {
                let alpha = 1.0 / self.period as f64;
                self.latest = Some(alpha * tr + (1.0 - alpha) * prev);
            }
            None => {
                self.seed.push(tr);
                if self.seed.len() == self.period {
                    self.latest = Some(self.seed.iter().sum::<f64>() / self.period as f64);
                }
            }
        }
        self.latest
    }

    pub fn value(&self) -> Option<f64> {
        self.latest
    }

    pub fn reset(&mut self) {
        self.prev_close = None;
        self.seed.clear();
        self.latest = None;
    }
}

/// Rolling highest-high / lowest-low channel.
#[derive(Debug)]
pub struct StreamingDonchian {
    period: usize,
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    latest: Option<(f64, f64)>,
}

impl StreamingDonchian {
    pub fn new(period: usize) -> Self {
        let period = period.max(1);
        Self {
            period,
            highs: VecDeque::with_capacity(period + 1),
            lows: VecDeque::with_capacity(period + 1),
            latest: None,
        }
    }

    /// Feed one bar; returns `(upper, lower)` once warm.
    pub fn update_bar(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        self.highs.push_back(high);
        self.lows.push_back(low);
        if self.highs.len() > self.period {
            self.highs.pop_front();
            self.lows.pop_front();
        }
        if self.highs.len() == self.period {
            let upper = self.highs.iter().copied().fold(f64::MIN, f64::max);
            let lower = self.lows.iter().copied().fold(f64::MAX, f64::min);
            self.latest = Some((upper, lower));
        }
        self.latest
    }

    pub fn value(&self) -> Option<(f64, f64)> {
        self.latest
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.latest = None;
    }
}

/// Bollinger bands: SMA middle ± k population standard deviations.
#[derive(Debug)]
pub struct StreamingBollinger {
    window: Window,
    k: f64,
    latest: Option<(f64, f64, f64)>,
}

impl StreamingBollinger {
    pub fn new(period: usize, k: f64) -> Self {
        Self { window: Window::new(period.max(1)), k, latest: None }
    }

    /// Feed one value; returns `(middle, upper, lower)` once warm.
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if self.window.push(value) {
            let mean = self.window.mean();
            let variance = self.window.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / self.window.period as f64;
            let band = self.k * variance.sqrt();
            self.latest = Some((mean, mean + band, mean - band));
        }
        self.latest
    }

    pub fn value(&self) -> Option<(f64, f64, f64)> {
        self.latest
    }

    pub fn reset(&mut self) {
        self.window.reset();
        self.latest = None;
    }
}

/// MACD line + signal + histogram, batch-matching EMA seeding.
#[derive(Debug)]
pub struct StreamingMacd {
    fast: StreamingEma,
    slow: StreamingEma,
    signal: StreamingEma,
    latest: Option<(f64, f64, f64)>,
}

impl StreamingMacd {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast: StreamingEma::new(fast),
            slow: StreamingEma::new(slow),
            signal: StreamingEma::new(signal),
            latest: None,
        }
    }

    /// Feed one value; returns `(macd, signal, histogram)` once all legs warm.
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        let fast = self.fast.update(value);
        let slow = self.slow.update(value);
        if let (Some(fast), Some(slow)) = (fast, slow) {
            let macd = fast - slow;
            if let Some(signal) = self.signal.update(macd) {
                self.latest = Some((macd, signal, macd - signal));
            }
        }
        self.latest
    }

    pub fn value(&self) -> Option<(f64, f64, f64)> {
        self.latest
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
        self.latest = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::{momentum, trend, volatility};

    fn series(n: usize) -> Vec<f64> {
        // Deterministic pseudo-random walk.
        let mut x: u64 = 42;
        let mut price = 100.0;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let step = ((x >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                price += step;
                price
            })
            .collect()
    }

    #[test]
    fn sma_matches_batch() {
        let data = series(200);
        let batch = trend::sma(&data, 14).unwrap();
        let mut streaming = StreamingSma::new(14);
        for (i, &v) in data.iter().enumerate() {
            let s = streaming.update(v);
            match s {
                Some(s) => assert!((s - batch[i]).abs() < 1e-9, "idx {i}"),
                None => assert!(batch[i].is_nan(), "idx {i}"),
            }
        }
    }

    #[test]
    fn ema_matches_batch() {
        let data = series(200);
        let batch = trend::ema(&data, 20).unwrap();
        let mut streaming = StreamingEma::new(20);
        for (i, &v) in data.iter().enumerate() {
            if let Some(s) = streaming.update(v) {
                assert!((s - batch[i]).abs() < 1e-9, "idx {i}");
            } else {
                assert!(batch[i].is_nan(), "idx {i}");
            }
        }
    }

    #[test]
    fn rsi_matches_batch() {
        let data = series(300);
        let batch = momentum::rsi(&data, 14).unwrap();
        let mut streaming = StreamingRsi::new(14);
        for (i, &v) in data.iter().enumerate() {
            if let Some(s) = streaming.update(v) {
                assert!((s - batch[i]).abs() < 1e-9, "idx {i}: {s} vs {}", batch[i]);
            } else {
                assert!(batch[i].is_nan(), "idx {i}");
            }
        }
    }

    #[test]
    fn atr_matches_batch() {
        let close = series(250);
        let high: Vec<f64> = close.iter().map(|c| c + 0.7).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.6).collect();
        let batch = volatility::atr(&high, &low, &close, 14).unwrap();
        let mut streaming = StreamingAtr::new(14);
        for i in 0..close.len() {
            if let Some(s) = streaming.update_bar(high[i], low[i], close[i]) {
                assert!((s - batch[i]).abs() < 1e-9, "idx {i}");
            } else {
                assert!(batch[i].is_nan(), "idx {i}");
            }
        }
    }

    #[test]
    fn bollinger_matches_batch() {
        let data = series(150);
        let batch = volatility::bollinger_bands(&data, 20, 2.0).unwrap();
        let mut streaming = StreamingBollinger::new(20, 2.0);
        for (i, &v) in data.iter().enumerate() {
            if let Some((mid, upper, lower)) = streaming.update(v) {
                assert!((mid - batch.middle[i]).abs() < 1e-9, "mid idx {i}");
                assert!((upper - batch.upper[i]).abs() < 1e-9, "upper idx {i}");
                assert!((lower - batch.lower[i]).abs() < 1e-9, "lower idx {i}");
            } else {
                assert!(batch.middle[i].is_nan(), "idx {i}");
            }
        }
    }

    #[test]
    fn windowed_cores_basic_sanity() {
        let mut wma = StreamingWma::new(3);
        assert!(wma.update(1.0).is_none());
        assert!(wma.update(2.0).is_none());
        // (1*1 + 2*2 + 3*3) / 6 = 14/6
        assert!((wma.update(3.0).unwrap() - 14.0 / 6.0).abs() < 1e-12);

        let mut roc = StreamingRoc::new(2);
        assert!(roc.update(100.0).is_none());
        assert!(roc.update(105.0).is_none());
        assert!((roc.update(110.0).unwrap() - 10.0).abs() < 1e-12);

        let mut donchian = StreamingDonchian::new(2);
        assert!(donchian.update_bar(10.0, 9.0).is_none());
        assert_eq!(donchian.update_bar(12.0, 8.5), Some((12.0, 8.5)));
        assert_eq!(donchian.update_bar(11.0, 9.5), Some((12.0, 8.5)));
    }
}
