//! Benchmark for RaptorBT backtesting performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use raptorbt::core::types::{BacktestConfig, CompiledSignals, Direction, OhlcvData};
use raptorbt::indicators::trend::{ema, sma};
use raptorbt::portfolio::engine::PortfolioEngine;

/// Generate sample OHLCV data.
fn generate_sample_data(n: usize) -> OhlcvData {
    let mut open = vec![100.0; n];
    let mut high = vec![101.0; n];
    let mut low = vec![99.0; n];
    let mut close = vec![100.0; n];

    // Create a trending pattern
    for i in 1..n {
        let change = (i as f64 * 0.1).sin() * 2.0;
        close[i] = close[i - 1] + change;
        open[i] = close[i - 1];
        high[i] = close[i].max(open[i]) + 1.0;
        low[i] = close[i].min(open[i]) - 1.0;
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

/// Generate sample trading signals based on SMA crossover.
fn generate_sample_signals(
    close: &[f64],
    fast_period: usize,
    slow_period: usize,
) -> CompiledSignals {
    let n = close.len();
    let fast_sma = sma(close, fast_period).unwrap_or_else(|_| vec![0.0; n]);
    let slow_sma = sma(close, slow_period).unwrap_or_else(|_| vec![0.0; n]);

    let mut entries = vec![false; n];
    let mut exits = vec![false; n];

    for i in 1..n {
        // Entry: fast crosses above slow
        if fast_sma[i] > slow_sma[i] && fast_sma[i - 1] <= slow_sma[i - 1] {
            entries[i] = true;
        }
        // Exit: fast crosses below slow
        if fast_sma[i] < slow_sma[i] && fast_sma[i - 1] >= slow_sma[i - 1] {
            exits[i] = true;
        }
    }

    CompiledSignals {
        symbol: "BENCH".to_string(),
        entries,
        exits,
        position_sizes: None,
        direction: Direction::Long,
        weight: 1.0,
    }
}

fn bench_single_backtest(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_backtest");

    for size in [1000, 5000, 10000, 50000].iter() {
        group.bench_with_input(BenchmarkId::new("bars", size), size, |b, &size| {
            let ohlcv = generate_sample_data(size);
            let signals = generate_sample_signals(&ohlcv.close, 10, 30);
            let config = BacktestConfig::default();
            let engine = PortfolioEngine::new(config);

            b.iter(|| {
                let result = engine.run_single(black_box(&ohlcv), black_box(&signals));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("sma");

    for size in [1000, 5000, 10000, 50000].iter() {
        group.bench_with_input(BenchmarkId::new("data_size", size), size, |b, &size| {
            let ohlcv = generate_sample_data(size);

            b.iter(|| {
                let result = sma(black_box(&ohlcv.close), black_box(20));
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("ema");

    for size in [1000, 5000, 10000, 50000].iter() {
        group.bench_with_input(BenchmarkId::new("data_size", size), size, |b, &size| {
            let ohlcv = generate_sample_data(size);

            b.iter(|| {
                let result = ema(black_box(&ohlcv.close), black_box(20));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_single_backtest, bench_sma, bench_ema);
criterion_main!(benches);
