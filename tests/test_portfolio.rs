//! Integration tests for RaptorBT portfolio engine.

use raptorbt::core::types::{
    BacktestConfig, CompiledSignals, Direction, OhlcvData, StopConfig, TargetConfig,
};
use raptorbt::portfolio::engine::PortfolioEngine;

fn sample_ohlcv() -> OhlcvData {
    // Create trending sample data
    let n = 100;
    let mut close = vec![100.0];
    let mut open = vec![100.0];
    let mut high = vec![101.0];
    let mut low = vec![99.0];

    for i in 1..n {
        let trend = (i as f64) * 0.5; // Upward trend
        let noise = ((i as f64) * 0.3).sin() * 2.0;
        let new_close = 100.0 + trend + noise;
        close.push(new_close);
        open.push(close[i - 1]);
        high.push(new_close + 1.0);
        low.push(new_close - 1.0);
    }

    OhlcvData {
        timestamps: (0..n as i64).collect(),
        open,
        high,
        low,
        close,
        volume: vec![1000.0; n],
    }
}

fn simple_signals(n: usize) -> CompiledSignals {
    // Entry at bar 10, exit at bar 50
    let mut entries = vec![false; n];
    let mut exits = vec![false; n];
    entries[10] = true;
    exits[50] = true;

    CompiledSignals {
        symbol: "TEST".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    }
}

#[test]
fn test_basic_backtest() {
    let ohlcv = sample_ohlcv();
    let signals = simple_signals(ohlcv.len());

    let config = BacktestConfig::default();
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Should have 1 complete trade
    assert_eq!(result.trades.len(), 1);

    // Equity curve should have same length as data
    assert_eq!(result.equity_curve.len(), ohlcv.len());

    // In an uptrend, should have positive return
    assert!(result.metrics.total_return_pct > 0.0);
}

#[test]
fn test_multiple_trades() {
    let ohlcv = sample_ohlcv();
    let n = ohlcv.len();

    // Multiple trades
    let mut entries = vec![false; n];
    let mut exits = vec![false; n];
    entries[10] = true;
    exits[20] = true;
    entries[30] = true;
    exits[40] = true;
    entries[50] = true;
    exits[60] = true;

    let signals = CompiledSignals {
        symbol: "TEST".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    };

    let config = BacktestConfig::default();
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Should have 3 trades
    assert_eq!(result.trades.len(), 3);
}

#[test]
fn test_with_fees() {
    let ohlcv = sample_ohlcv();
    let signals = simple_signals(ohlcv.len());

    let config = BacktestConfig {
        fees: 0.01, // 1% fee
        ..Default::default()
    };
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Trade should have fees deducted
    assert!(result.trades[0].fees > 0.0);

    // Return should be lower due to fees
    let config_no_fees = BacktestConfig::default();
    let engine_no_fees = PortfolioEngine::new(config_no_fees);
    let result_no_fees = engine_no_fees.run_single(&ohlcv, &signals);

    assert!(result.metrics.end_value < result_no_fees.metrics.end_value);
}

#[test]
fn test_fixed_stop_loss() {
    let ohlcv = sample_ohlcv();
    let n = ohlcv.len();

    // Entry at bar 10
    let mut entries = vec![false; n];
    entries[10] = true;
    let exits = vec![false; n]; // No exit signal

    let signals = CompiledSignals {
        symbol: "TEST".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    };

    let config = BacktestConfig {
        stop: StopConfig::Fixed { percent: 0.02 }, // 2% stop
        ..Default::default()
    };
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Should have at least one trade (may exit on stop or end of data)
    assert!(!result.trades.is_empty());
}

#[test]
fn test_fixed_take_profit() {
    let ohlcv = sample_ohlcv();
    let n = ohlcv.len();

    // Entry at bar 10
    let mut entries = vec![false; n];
    entries[10] = true;
    let exits = vec![false; n]; // No exit signal

    let signals = CompiledSignals {
        symbol: "TEST".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    };

    let config = BacktestConfig {
        target: TargetConfig::Fixed { percent: 0.10 }, // 10% target
        ..Default::default()
    };
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Should have at least one trade
    assert!(!result.trades.is_empty());
}

#[test]
fn test_no_trades() {
    let ohlcv = sample_ohlcv();
    let n = ohlcv.len();

    // No entry signals
    let signals = CompiledSignals {
        symbol: "TEST".to_string(),
        entries: vec![false; n],
        exits: vec![false; n],
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    };

    let config = BacktestConfig::default();
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Should have no trades
    assert_eq!(result.trades.len(), 0);
    assert_eq!(result.metrics.total_trades, 0);

    // Equity should remain at initial capital
    assert!((result.metrics.end_value - result.metrics.start_value).abs() < 1e-10);
}

#[test]
fn test_drawdown_positive() {
    let ohlcv = sample_ohlcv();
    let signals = simple_signals(ohlcv.len());

    let config = BacktestConfig::default();
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // All drawdown values should be non-negative
    for dd in &result.drawdown_curve {
        assert!(*dd >= 0.0, "Drawdown should be non-negative");
    }
}

#[test]
fn test_short_direction() {
    // Create downtrend data
    let n = 100;
    let mut close = vec![100.0];
    for i in 1..n {
        close.push(100.0 - (i as f64) * 0.3); // Downward trend
    }

    let ohlcv = OhlcvData {
        timestamps: (0..n as i64).collect(),
        open: close.iter().skip(1).chain(std::iter::once(&close[n - 1])).cloned().collect(),
        high: close.iter().map(|c| c + 1.0).collect(),
        low: close.iter().map(|c| c - 1.0).collect(),
        close: close.clone(),
        volume: vec![1000.0; n],
    };

    // Entry at bar 10, exit at bar 50
    let mut entries = vec![false; n];
    let mut exits = vec![false; n];
    entries[10] = true;
    exits[50] = true;

    let signals = CompiledSignals {
        symbol: "TEST".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Short, // Short direction
        weight: 1.0,
    };

    let config = BacktestConfig::default();
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Short in a downtrend should be profitable
    assert!(result.trades[0].pnl > 0.0);
}

#[test]
fn test_metrics_consistency() {
    let ohlcv = sample_ohlcv();
    let n = ohlcv.len();

    // Multiple trades for statistics
    let mut entries = vec![false; n];
    let mut exits = vec![false; n];
    for i in (10..90).step_by(20) {
        entries[i] = true;
        exits[i + 10] = true;
    }

    let signals = CompiledSignals {
        symbol: "TEST".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    };

    let config = BacktestConfig::default();
    let engine = PortfolioEngine::new(config);
    let result = engine.run_single(&ohlcv, &signals);

    // Total trades should equal winning + losing
    assert_eq!(
        result.metrics.total_trades,
        result.metrics.winning_trades + result.metrics.losing_trades
    );

    // Win rate should be in [0, 100]
    assert!(result.metrics.win_rate_pct >= 0.0);
    assert!(result.metrics.win_rate_pct <= 100.0);

    // Exposure should be in [0, 100]
    assert!(result.metrics.exposure_pct >= 0.0);
    assert!(result.metrics.exposure_pct <= 100.0);
}
