//! Single instrument backtest implementation.

use crate::core::types::{BacktestConfig, BacktestResult, CompiledSignals, OhlcvData};
use crate::portfolio::engine::PortfolioEngine;

/// Single instrument backtest runner.
#[derive(Debug)]
pub struct SingleBacktest {
    /// Portfolio engine.
    engine: PortfolioEngine,
}

impl SingleBacktest {
    /// Create a new single instrument backtest.
    pub fn new(config: BacktestConfig) -> Self {
        Self { engine: PortfolioEngine::new(config) }
    }

    /// Run the backtest.
    ///
    /// # Arguments
    /// * `ohlcv` - OHLCV price data
    /// * `signals` - Compiled trading signals
    ///
    /// # Returns
    /// Backtest result with metrics, trades, and equity curve
    pub fn run(&self, ohlcv: &OhlcvData, signals: &CompiledSignals) -> BacktestResult {
        self.engine.run_single(ohlcv, signals)
    }

    /// Run backtest from raw arrays.
    ///
    /// # Arguments
    /// * `timestamps` - Timestamp array
    /// * `open` - Open prices
    /// * `high` - High prices
    /// * `low` - Low prices
    /// * `close` - Close prices
    /// * `volume` - Volume
    /// * `entries` - Entry signals
    /// * `exits` - Exit signals
    /// * `direction` - Trade direction (1 = long, -1 = short)
    /// * `symbol` - Symbol name
    ///
    /// # Returns
    /// Backtest result
    pub fn run_from_arrays(
        &self,
        timestamps: &[i64],
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        entries: &[bool],
        exits: &[bool],
        direction: i32,
        symbol: &str,
    ) -> BacktestResult {
        let ohlcv = OhlcvData {
            timestamps: timestamps.to_vec(),
            open: open.to_vec(),
            high: high.to_vec(),
            low: low.to_vec(),
            close: close.to_vec(),
            volume: volume.to_vec(),
        };

        let dir = crate::core::types::Direction::from_int(direction)
            .unwrap_or(crate::core::types::Direction::Long);

        let signals = CompiledSignals {
            symbol: symbol.to_string(),
            entries: entries.to_vec(),
            exits: exits.to_vec(),
            position_sizes: None,
            direction: dir,
            weight: 1.0,
        };

        self.run(&ohlcv, &signals)
    }

    /// Run backtest with position sizing.
    ///
    /// # Arguments
    /// * `ohlcv` - OHLCV price data
    /// * `signals` - Compiled trading signals
    /// * `position_sizes` - Position size for each bar (fraction of capital)
    ///
    /// # Returns
    /// Backtest result
    pub fn run_with_sizing(
        &self,
        ohlcv: &OhlcvData,
        signals: &CompiledSignals,
        position_sizes: Vec<f64>,
    ) -> BacktestResult {
        let mut signals_with_sizing = signals.clone();
        signals_with_sizing.position_sizes = Some(position_sizes);
        self.engine.run_single(ohlcv, &signals_with_sizing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Direction, StopConfig, TargetConfig};

    fn sample_data() -> (OhlcvData, CompiledSignals) {
        let ohlcv = OhlcvData {
            timestamps: (0..20).map(|i| i as i64).collect(),
            open: vec![
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 101.0,
                102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            ],
            high: vec![
                101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 102.0,
                103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
            ],
            low: vec![
                99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0,
                101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
            ],
            close: vec![
                100.5, 101.5, 102.5, 103.5, 104.5, 105.0, 104.0, 103.0, 102.0, 101.0, 100.5, 101.5,
                102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5,
            ],
            volume: vec![1000.0; 20],
        };

        let signals = CompiledSignals {
            symbol: "TEST".to_string(),
            entries: vec![
                false, true, false, false, false, false, false, false, false, false, false, true,
                false, false, false, false, false, false, false, false,
            ],
            exits: vec![
                false, false, false, false, false, true, false, false, false, false, false, false,
                false, false, false, true, false, false, false, false,
            ],
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        };

        (ohlcv, signals)
    }

    #[test]
    fn test_single_backtest() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.0,
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
        };

        let backtest = SingleBacktest::new(config);
        let (ohlcv, signals) = sample_data();

        let result = backtest.run(&ohlcv, &signals);

        assert_eq!(result.trades.len(), 2);
        assert!(result.metrics.total_return_pct > 0.0);
    }

    #[test]
    fn test_from_arrays() {
        let config = BacktestConfig::default();
        let backtest = SingleBacktest::new(config);

        let timestamps: Vec<i64> = (0..10).collect();
        let close: Vec<f64> = (100..110).map(|x| x as f64).collect();
        let open = close.clone();
        let high: Vec<f64> = close.iter().map(|x| x + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|x| x - 1.0).collect();
        let volume = vec![1000.0; 10];

        let entries = vec![false, true, false, false, false, false, false, false, false, false];
        let exits = vec![false, false, false, false, false, true, false, false, false, false];

        let result = backtest.run_from_arrays(
            &timestamps,
            &open,
            &high,
            &low,
            &close,
            &volume,
            &entries,
            &exits,
            1,
            "TEST",
        );

        assert_eq!(result.trades.len(), 1);
    }
}
