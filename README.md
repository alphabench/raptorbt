# RaptorBT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/raptorbt.svg)](https://pypi.org/project/raptorbt/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-red.svg)](https://www.rust-lang.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/raptorbt?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=ORANGE&left_text=downloads)](https://pepy.tech/projects/raptorbt)

**Blazing-fast backtesting for the modern quant.**

RaptorBT is a high-performance backtesting engine written in Rust with Python bindings via PyO3. It serves as a drop-in replacement for VectorBT — delivering **HFT-grade compute efficiency** with full metric parity.

<p align="center">
  <strong>5,800x faster</strong> · <strong>45x smaller</strong> · <strong>100% deterministic</strong>
</p>

---

### Quick Install

```bash
pip install raptorbt
```

### 30-Second Example

```python
import numpy as np
import raptorbt

# Configure
config = raptorbt.PyBacktestConfig(initial_capital=100000, fees=0.001)

# Run backtest
result = raptorbt.run_single_backtest(
    timestamps=timestamps, open=open, high=high, low=low, close=close,
    volume=volume, entries=entries, exits=exits,
    direction=1, weight=1.0, symbol="AAPL", config=config,
)

# Results
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

---

Developed and maintained by the [Alphabench](https://alphabench.in) team.

## Table of Contents

- [Overview](#overview)
- [Performance](#performance)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Strategy Types](#strategy-types)
- [Metrics](#metrics)
- [Indicators](#indicators)
- [Stop-Loss & Take-Profit](#stop-loss--take-profit)
- [VectorBT Comparison](#vectorbt-comparison)
- [API Reference](#api-reference)
- [Building from Source](#building-from-source)
- [Testing](#testing)

---

## Overview

RaptorBT was built to address the performance limitations of VectorBT. Benchmarked by the Alphabench team:

| Metric                        | VectorBT            | RaptorBT     | Improvement               |
| ----------------------------- | ------------------- | ------------ | ------------------------- |
| **Disk Footprint**            | ~450MB              | <10MB        | **45x smaller**           |
| **Startup Latency**           | 200-600ms           | <10ms        | **20-60x faster**         |
| **Backtest Speed (1K bars)**  | 1460ms              | 0.25ms       | **5,800x faster**         |
| **Backtest Speed (50K bars)** | 43ms                | 1.7ms        | **25x faster**            |
| **Memory Usage**              | High (JIT + pandas) | Low (native) | **Significant reduction** |

### Key Features

- **6 Strategy Types**: Single instrument, basket/collective, pairs trading, options, spreads, and multi-strategy
- **Monte Carlo Simulation**: Correlated multi-asset forward projection via GBM + Cholesky decomposition
- **33 Metrics**: Full parity with VectorBT including Sharpe, Sortino, Calmar, Omega, SQN, Payoff Ratio, Recovery Factor, and more
- **12 Technical Indicators**: SMA, EMA, RSI, MACD, Stochastic, ATR, Bollinger Bands, ADX, VWAP, Supertrend, Rolling Min, Rolling Max
- **Stop/Target Management**: Fixed, ATR-based, and trailing stops with risk-reward targets
- **100% Deterministic**: No JIT compilation variance between runs
- **Native Parallelism**: Rayon-based parallel processing with explicit SIMD optimizations

---

## Performance

### Benchmark Results

Tested on Apple Silicon M-series with random walk price data and SMA crossover strategy:

```
┌─────────────┬────────────┬───────────┬──────────┐
│ Data Size   │ VectorBT   │ RaptorBT  │ Speedup  │
├─────────────┼────────────┼───────────┼──────────┤
│ 1,000 bars  │ 1,460 ms   │ 0.25 ms   │ 5,827x   │
│ 5,000 bars  │ 36 ms      │ 0.24 ms   │ 153x     │
│ 10,000 bars │ 37 ms      │ 0.46 ms   │ 80x      │
│ 50,000 bars │ 43 ms      │ 1.68 ms   │ 26x      │
└─────────────┴────────────┴───────────┴──────────┘
```

> **Note**: First VectorBT run includes Numba JIT compilation overhead. Subsequent runs are faster but still significantly slower than RaptorBT.

### Metric Accuracy

RaptorBT produces **identical results** to VectorBT:

```
VectorBT Total Return: 7.2764%
RaptorBT Total Return: 7.2764%
Difference: 0.0000% ✓
```

---

## Architecture

```
raptorbt/
├── src/
│   ├── core/              # Core types and error handling
│   │   ├── types.rs       # BacktestConfig, BacktestResult, Trade, Metrics
│   │   ├── error.rs       # RaptorError enum
│   │   ├── session.rs     # SessionTracker, SessionConfig (intraday sessions)
│   │   └── timeseries.rs  # Time series utilities
│   │
│   ├── strategies/        # Strategy implementations
│   │   ├── single.rs      # Single instrument backtest
│   │   ├── basket.rs      # Basket/collective strategies
│   │   ├── pairs.rs       # Pairs trading
│   │   ├── options.rs     # Options strategies
│   │   ├── spreads.rs     # Multi-leg spread strategies
│   │   └── multi.rs       # Multi-strategy combining
│   │
│   ├── indicators/        # Technical indicators
│   │   ├── trend.rs       # SMA, EMA, Supertrend
│   │   ├── momentum.rs    # RSI, MACD, Stochastic
│   │   ├── volatility.rs  # ATR, Bollinger Bands
│   │   ├── strength.rs    # ADX
│   │   ├── volume.rs      # VWAP
│   │   └── rolling.rs     # Rolling Min/Max (LLV/HHV)
│   │
│   ├── metrics/           # Performance metrics
│   │   ├── streaming.rs   # Streaming metric calculations
│   │   ├── drawdown.rs    # Drawdown analysis
│   │   └── trade_stats.rs # Trade statistics
│   │
│   ├── signals/           # Signal processing
│   │   ├── processor.rs   # Entry/exit signal processing
│   │   ├── synchronizer.rs # Multi-instrument sync
│   │   └── expression.rs  # Signal expressions
│   │
│   ├── stops/             # Stop-loss implementations
│   │   ├── fixed.rs       # Fixed percentage stops
│   │   ├── atr.rs         # ATR-based stops
│   │   └── trailing.rs    # Trailing stops
│   │
│   ├── portfolio/         # Portfolio-level analysis
│   │   ├── monte_carlo.rs # Monte Carlo forward simulation (GBM + Cholesky)
│   │   ├── allocation.rs  # Capital allocation
│   │   ├── engine.rs      # Portfolio engine
│   │   └── position.rs    # Position management
│   │
│   ├── python/            # PyO3 bindings
│   │   ├── bindings.rs    # Python function exports
│   │   └── numpy_bridge.rs # NumPy array conversion
│   │
│   └── lib.rs             # Library entry point
│
├── Cargo.toml             # Rust dependencies
└── pyproject.toml         # Python package config
```

---

## Installation

### From Pre-built Wheel

```bash
pip install raptorbt
```

### From Source

```bash
cd raptorbt
maturin develop --release
```

### Verify Installation

```python
import raptorbt
print("RaptorBT installed successfully!")
```

---

## Quick Start

### Basic Single Instrument Backtest

```python
import numpy as np
import pandas as pd
import raptorbt

# Prepare data
df = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Generate signals (SMA crossover example)
sma_fast = df['close'].rolling(10).mean()
sma_slow = df['close'].rolling(20).mean()
entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

# Configure backtest
config = raptorbt.PyBacktestConfig(
    initial_capital=100000,
    fees=0.001,        # 0.1% per trade
    slippage=0.0005,   # 0.05% slippage
    upon_bar_close=True
)

# Optional: Add stop-loss
config.set_fixed_stop(0.02)  # 2% stop-loss

# Optional: Add take-profit
config.set_fixed_target(0.04)  # 4% take-profit

# Run backtest
result = raptorbt.run_single_backtest(
    timestamps=df.index.astype('int64').values,
    open=df['open'].values,
    high=df['high'].values,
    low=df['low'].values,
    close=df['close'].values,
    volume=df['volume'].values,
    entries=entries.values,
    exits=exits.values,
    direction=1,       # 1 = Long, -1 = Short
    weight=1.0,
    symbol="AAPL",
    config=config,
)

# Access results
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
print(f"Win Rate: {result.metrics.win_rate_pct:.2f}%")
print(f"Total Trades: {result.metrics.total_trades}")

# Get equity curve
equity = result.equity_curve()  # Returns numpy array

# Get trades
trades = result.trades()  # Returns list of PyTrade objects
```

---

## Strategy Types

### 1. Single Instrument

Basic long or short strategy on a single instrument.

```python
# Optional: Instrument-specific configuration
inst_config = raptorbt.PyInstrumentConfig(lot_size=1.0)

result = raptorbt.run_single_backtest(
    timestamps=timestamps,
    open=open_prices, high=high_prices, low=low_prices,
    close=close_prices, volume=volume,
    entries=entries, exits=exits,
    direction=1,  # 1=Long, -1=Short
    weight=1.0,
    symbol="SYMBOL",
    config=config,
    instrument_config=inst_config,  # Optional: lot_size rounding, capital caps
)
```

### 2. Basket/Collective

Trade multiple instruments with synchronized signals.

```python
instruments = [
    (timestamps, open1, high1, low1, close1, volume1, entries1, exits1, 1, 0.33, "AAPL"),
    (timestamps, open2, high2, low2, close2, volume2, entries2, exits2, 1, 0.33, "GOOGL"),
    (timestamps, open3, high3, low3, close3, volume3, entries3, exits3, 1, 0.34, "MSFT"),
]

# Optional: Per-instrument configs for lot_size and capital allocation
instrument_configs = {
    "AAPL": raptorbt.PyInstrumentConfig(lot_size=1.0, alloted_capital=33000),
    "GOOGL": raptorbt.PyInstrumentConfig(lot_size=1.0, alloted_capital=33000),
    "MSFT": raptorbt.PyInstrumentConfig(lot_size=1.0, alloted_capital=34000),
}

result = raptorbt.run_basket_backtest(
    instruments=instruments,
    config=config,
    sync_mode="all",  # "all", "any", "majority", "master"
    instrument_configs=instrument_configs,  # Optional
)
```

**Sync Modes:**

- `all`: Enter only when ALL instruments signal
- `any`: Enter when ANY instrument signals
- `majority`: Enter when >50% of instruments signal
- `master`: Follow the first instrument's signals

### 3. Pairs Trading

Long one instrument, short another with optional hedge ratio.

```python
result = raptorbt.run_pairs_backtest(
    # Long leg
    leg1_timestamps=timestamps,
    leg1_open=long_open, leg1_high=long_high,
    leg1_low=long_low, leg1_close=long_close,
    leg1_volume=long_volume,
    # Short leg
    leg2_timestamps=timestamps,
    leg2_open=short_open, leg2_high=short_high,
    leg2_low=short_low, leg2_close=short_close,
    leg2_volume=short_volume,
    # Signals
    entries=entries, exits=exits,
    direction=1,
    symbol="TCS_INFY",
    config=config,
    hedge_ratio=1.5,      # Short 1.5x the long position
    dynamic_hedge=False,  # Use rolling hedge ratio
)
```

### 4. Options

Backtest options strategies with strike selection.

```python
result = raptorbt.run_options_backtest(
    timestamps=timestamps,
    open=underlying_open, high=underlying_high,
    low=underlying_low, close=underlying_close,
    volume=volume,
    option_prices=option_prices,  # Option premium series
    entries=entries, exits=exits,
    direction=1,
    symbol="NIFTY_CE",
    config=config,
    option_type="call",           # "call" or "put"
    strike_selection="atm",       # "atm", "otm1", "otm2", "itm1", "itm2"
    size_type="percent",          # "percent", "contracts", "notional", "risk"
    size_value=0.1,               # 10% of capital
    lot_size=50,                  # Options lot size
    strike_interval=50.0,         # Strike interval (e.g., 50 for NIFTY)
)
```

### 5. Multi-Strategy

Combine multiple strategies on the same instrument.

```python
strategies = [
    (entries_sma, exits_sma, 1, 0.4, "SMA_Crossover"),    # 40% weight
    (entries_rsi, exits_rsi, 1, 0.35, "RSI_MeanRev"),     # 35% weight
    (entries_bb, exits_bb, 1, 0.25, "BB_Breakout"),       # 25% weight
]

result = raptorbt.run_multi_backtest(
    timestamps=timestamps,
    open=open_prices, high=high_prices,
    low=low_prices, close=close_prices,
    volume=volume,
    strategies=strategies,
    config=config,
    combine_mode="any",  # "any", "all", "majority", "weighted", "independent"
)
```

**Combine Modes:**

- `any`: Enter when any strategy signals
- `all`: Enter only when all strategies signal
- `majority`: Enter when >50% of strategies signal
- `weighted`: Weight signals by strategy weight
- `independent`: Run strategies independently (aggregate PnL)

---

## Metrics

RaptorBT calculates 30+ performance metrics:

### Core Performance

| Metric             | Description                       |
| ------------------ | --------------------------------- |
| `total_return_pct` | Total return as percentage        |
| `sharpe_ratio`     | Risk-adjusted return (annualized) |
| `sortino_ratio`    | Downside risk-adjusted return     |
| `calmar_ratio`     | Return / Max Drawdown             |
| `omega_ratio`      | Probability-weighted gains/losses |

### Drawdown

| Metric                  | Description                    |
| ----------------------- | ------------------------------ |
| `max_drawdown_pct`      | Maximum peak-to-trough decline |
| `max_drawdown_duration` | Longest drawdown period (bars) |

### Trade Statistics

| Metric                | Description                  |
| --------------------- | ---------------------------- |
| `total_trades`        | Total number of trades       |
| `total_closed_trades` | Number of closed trades      |
| `total_open_trades`   | Number of open positions     |
| `winning_trades`      | Number of profitable trades  |
| `losing_trades`       | Number of losing trades      |
| `win_rate_pct`        | Percentage of winning trades |

### Trade Performance

| Metric                 | Description                       |
| ---------------------- | --------------------------------- |
| `profit_factor`        | Gross profit / Gross loss         |
| `expectancy`           | Average expected profit per trade |
| `sqn`                  | System Quality Number             |
| `avg_trade_return_pct` | Average trade return              |
| `avg_win_pct`          | Average winning trade return      |
| `avg_loss_pct`         | Average losing trade return       |
| `best_trade_pct`       | Best single trade return          |
| `worst_trade_pct`      | Worst single trade return         |

### Duration

| Metric                 | Description                    |
| ---------------------- | ------------------------------ |
| `avg_holding_period`   | Average trade duration (bars)  |
| `avg_winning_duration` | Average winning trade duration |
| `avg_losing_duration`  | Average losing trade duration  |

### Streaks

| Metric                   | Description            |
| ------------------------ | ---------------------- |
| `max_consecutive_wins`   | Longest winning streak |
| `max_consecutive_losses` | Longest losing streak  |

### Other

| Metric            | Description                        |
| ----------------- | ---------------------------------- |
| `start_value`     | Initial portfolio value            |
| `end_value`       | Final portfolio value              |
| `total_fees_paid` | Total transaction costs            |
| `open_trade_pnl`  | Unrealized PnL from open positions |
| `exposure_pct`    | Percentage of time in market       |

---

## Indicators

RaptorBT includes optimized technical indicators:

```python
import raptorbt

# Trend indicators
sma = raptorbt.sma(close, period=20)
ema = raptorbt.ema(close, period=20)
supertrend, direction = raptorbt.supertrend(high, low, close, period=10, multiplier=3.0)

# Momentum indicators
rsi = raptorbt.rsi(close, period=14)
macd_line, signal_line, histogram = raptorbt.macd(close, fast=12, slow=26, signal=9)
stoch_k, stoch_d = raptorbt.stochastic(high, low, close, k_period=14, d_period=3)

# Volatility indicators
atr = raptorbt.atr(high, low, close, period=14)
upper, middle, lower = raptorbt.bollinger_bands(close, period=20, std_dev=2.0)

# Strength indicators
adx = raptorbt.adx(high, low, close, period=14)

# Volume indicators
vwap = raptorbt.vwap(high, low, close, volume)
```

---

## Stop-Loss & Take-Profit

### Fixed Percentage

```python
config = raptorbt.PyBacktestConfig(initial_capital=100000, fees=0.001)
config.set_fixed_stop(0.02)    # 2% stop-loss
config.set_fixed_target(0.04)  # 4% take-profit
```

### ATR-Based

```python
config.set_atr_stop(multiplier=2.0, period=14)    # 2x ATR stop
config.set_atr_target(multiplier=3.0, period=14)  # 3x ATR target
```

### Trailing Stop

```python
config.set_trailing_stop(0.02)  # 2% trailing stop
```

### Risk-Reward Target

```python
config.set_risk_reward_target(ratio=2.0)  # 2:1 risk-reward ratio
```

---

## Monte Carlo Portfolio Simulation

RaptorBT includes a high-performance Monte Carlo forward simulation engine for portfolio risk analysis. It uses Geometric Brownian Motion (GBM) with Cholesky decomposition for correlated multi-asset simulation, parallelized via Rayon.

```python
import numpy as np
import raptorbt

# Historical daily returns per strategy/asset (numpy arrays)
returns = [
    np.array([0.001, -0.002, 0.003, ...]),  # Strategy 1 returns
    np.array([0.002, 0.001, -0.001, ...]),   # Strategy 2 returns
]

# Portfolio weights (must sum to 1.0)
weights = np.array([0.6, 0.4])

# Correlation matrix (N x N)
correlation_matrix = [
    np.array([1.0, 0.3]),
    np.array([0.3, 1.0]),
]

# Run simulation
result = raptorbt.simulate_portfolio_mc(
    returns=returns,
    weights=weights,
    correlation_matrix=correlation_matrix,
    initial_value=100000.0,
    n_simulations=10000,   # Number of Monte Carlo paths (default: 10,000)
    horizon_days=252,      # Forward projection horizon (default: 252)
    seed=42,               # Random seed for reproducibility (default: 42)
)

# Results
print(f"Expected Return: {result['expected_return']:.2f}%")
print(f"Probability of Loss: {result['probability_of_loss']:.2%}")
print(f"VaR (95%): {result['var_95']:.2f}%")
print(f"CVaR (95%): {result['cvar_95']:.2f}%")

# Percentile paths: list of (percentile, path_values)
# Percentiles: 5th, 25th, 50th, 75th, 95th
for pct, path in result['percentile_paths']:
    print(f"  P{pct:.0f} final value: {path[-1]:.2f}")

# Final values: numpy array of terminal values for all simulations
final_values = result['final_values']  # numpy array, length = n_simulations
```

### Result Fields

| Field                 | Type                       | Description                                                |
| --------------------- | -------------------------- | ---------------------------------------------------------- |
| `expected_return`     | `float`                    | Expected return as percentage over the horizon             |
| `probability_of_loss` | `float`                    | Probability that final value < initial value (0.0 to 1.0)  |
| `var_95`              | `float`                    | Value at Risk at 95% confidence (percentage)               |
| `cvar_95`             | `float`                    | Conditional VaR at 95% confidence (percentage)             |
| `percentile_paths`    | `List[Tuple[float, List]]` | Portfolio paths at 5th, 25th, 50th, 75th, 95th percentiles |
| `final_values`        | `numpy.ndarray`            | Terminal portfolio values for all simulations              |

---

## VectorBT Comparison

RaptorBT is designed as a drop-in replacement for VectorBT. Here's a side-by-side comparison:

### VectorBT (before)

```python
import vectorbt as vbt
import pandas as pd

# Run backtest
pf = vbt.Portfolio.from_signals(
    close=close_series,
    entries=entries,
    exits=exits,
    init_cash=100000,
    fees=0.001,
)

# Get metrics
print(pf.stats()["Total Return [%]"])
print(pf.stats()["Sharpe Ratio"])
print(pf.stats()["Max Drawdown [%]"])
```

### RaptorBT (after)

```python
import raptorbt
import numpy as np

# Configure backtest
config = raptorbt.PyBacktestConfig(
    initial_capital=100000,
    fees=0.001,
)

# Run backtest
result = raptorbt.run_single_backtest(
    timestamps=timestamps,
    open=open_prices, high=high_prices,
    low=low_prices, close=close_prices,
    volume=volume,
    entries=entries, exits=exits,
    direction=1, weight=1.0,
    symbol="SYMBOL",
    config=config,
)

# Get metrics
print(f"Total Return: {result.metrics.total_return_pct}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct}%")
```

### Metric Mapping

| VectorBT Key       | RaptorBT Attribute         |
| ------------------ | -------------------------- |
| `Total Return [%]` | `metrics.total_return_pct` |
| `Sharpe Ratio`     | `metrics.sharpe_ratio`     |
| `Sortino Ratio`    | `metrics.sortino_ratio`    |
| `Max Drawdown [%]` | `metrics.max_drawdown_pct` |
| `Win Rate [%]`     | `metrics.win_rate_pct`     |
| `Profit Factor`    | `metrics.profit_factor`    |
| `SQN`              | `metrics.sqn`              |
| `Omega Ratio`      | `metrics.omega_ratio`      |
| `Total Trades`     | `metrics.total_trades`     |
| `Expectancy`       | `metrics.expectancy`       |

---

## API Reference

### PyBacktestConfig

```python
config = raptorbt.PyBacktestConfig(
    initial_capital: float = 100000.0,
    fees: float = 0.001,
    slippage: float = 0.0,
    upon_bar_close: bool = True,
)

# Stop methods
config.set_fixed_stop(percent: float)
config.set_atr_stop(multiplier: float, period: int)
config.set_trailing_stop(percent: float)

# Target methods
config.set_fixed_target(percent: float)
config.set_atr_target(multiplier: float, period: int)
config.set_risk_reward_target(ratio: float)
```

### PyInstrumentConfig

Per-instrument configuration for position sizing and risk management.

```python
inst_config = raptorbt.PyInstrumentConfig(
    lot_size=1.0,              # Min tradeable quantity (1 for equity, 50 for NIFTY F&O)
    alloted_capital=50000.0,   # Capital allocated to this instrument (optional)
    existing_qty=None,         # Existing position quantity (future use)
    avg_price=None,            # Existing position avg price (future use)
)

# Optional: per-instrument stop/target overrides
inst_config.set_fixed_stop(0.02)
inst_config.set_trailing_stop(0.03)
inst_config.set_fixed_target(0.05)
```

**Fields:**

- `lot_size` - Minimum tradeable quantity. Position sizes are rounded down to nearest lot_size multiple. Use `1.0` for equities, `50.0` for NIFTY F&O, `0.01` for forex.
- `alloted_capital` - Per-instrument capital cap (capped at available cash).
- `existing_qty` / `avg_price` - Reserved for future live-to-backtest transitions.

### simulate_portfolio_mc

```python
result = raptorbt.simulate_portfolio_mc(
    returns: List[np.ndarray],               # Per-asset daily returns (N arrays)
    weights: np.ndarray,                     # Portfolio weights (length N, sum to 1)
    correlation_matrix: List[np.ndarray],    # N x N correlation matrix
    initial_value: float,                    # Starting portfolio value
    n_simulations: int = 10000,              # Number of Monte Carlo paths
    horizon_days: int = 252,                 # Forward projection horizon in days
    seed: int = 42,                          # Random seed for reproducibility
) -> dict
```

Returns a dictionary with keys: `expected_return`, `probability_of_loss`, `var_95`, `cvar_95`, `percentile_paths`, `final_values`.

### PyBacktestResult

```python
result = raptorbt.run_single_backtest(...)

# Attributes
result.metrics        # PyBacktestMetrics object

# Methods
result.equity_curve()    # numpy.ndarray
result.drawdown_curve()  # numpy.ndarray
result.returns()         # numpy.ndarray
result.trades()          # List[PyTrade]
```

### PyBacktestMetrics

```python
metrics = result.metrics

# All available metrics
metrics.total_return_pct
metrics.sharpe_ratio
metrics.sortino_ratio
metrics.calmar_ratio
metrics.omega_ratio
metrics.max_drawdown_pct
metrics.max_drawdown_duration
metrics.win_rate_pct
metrics.profit_factor
metrics.expectancy
metrics.sqn
metrics.total_trades
metrics.total_closed_trades
metrics.total_open_trades
metrics.winning_trades
metrics.losing_trades
metrics.start_value
metrics.end_value
metrics.total_fees_paid
metrics.best_trade_pct
metrics.worst_trade_pct
metrics.avg_trade_return_pct
metrics.avg_win_pct
metrics.avg_loss_pct
metrics.avg_holding_period
metrics.avg_winning_duration
metrics.avg_losing_duration
metrics.max_consecutive_wins
metrics.max_consecutive_losses
metrics.exposure_pct
metrics.open_trade_pnl
metrics.payoff_ratio            # avg win / avg loss (risk/reward per trade)
metrics.recovery_factor         # net profit / max drawdown (resilience)

# Convert to dictionary (VectorBT format)
stats_dict = metrics.to_dict()
```

### PyTrade

```python
for trade in result.trades():
    print(trade.id)           # Trade ID
    print(trade.symbol)       # Symbol
    print(trade.entry_idx)    # Entry bar index
    print(trade.exit_idx)     # Exit bar index
    print(trade.entry_price)  # Entry price
    print(trade.exit_price)   # Exit price
    print(trade.size)         # Position size
    print(trade.direction)    # 1=Long, -1=Short
    print(trade.pnl)          # Profit/Loss
    print(trade.return_pct)   # Return percentage
    print(trade.fees)         # Fees paid
    print(trade.exit_reason)  # "Signal", "StopLoss", "TakeProfit"
```

---

## Building from Source

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.10+
- maturin (`pip install maturin`)

### Development Build

```bash
cd raptorbt
maturin develop --release
```

### Production Build

```bash
cd raptorbt
maturin build --release
pip install target/wheels/raptorbt-*.whl
```

---

## Testing

### Rust Unit Tests

```bash
cd raptorbt
cargo test
```

### Python Integration Tests

```python
import raptorbt
import numpy as np

config = raptorbt.PyBacktestConfig(initial_capital=100000, fees=0.001)
result = raptorbt.run_single_backtest(
    timestamps=np.arange(100, dtype=np.int64),
    open=np.random.randn(100).cumsum() + 100,
    high=np.random.randn(100).cumsum() + 101,
    low=np.random.randn(100).cumsum() + 99,
    close=np.random.randn(100).cumsum() + 100,
    volume=np.ones(100),
    entries=np.array([i % 20 == 0 for i in range(100)]),
    exits=np.array([i % 20 == 10 for i in range(100)]),
    direction=1,
    weight=1.0,
    symbol='TEST',
    config=config,
)
print(f'Total Return: {result.metrics.total_return_pct:.2f}%')
print('RaptorBT is working correctly!')
```

### Comparison Test (VectorBT vs RaptorBT)

```python
import numpy as np
import pandas as pd
import vectorbt as vbt
import raptorbt

# Create test data
np.random.seed(42)
n = 500
dates = pd.date_range('2023-01-01', periods=n, freq='D')
close = np.cumprod(1 + np.random.randn(n) * 0.02) * 100
entries = np.zeros(n, dtype=bool)
exits = np.zeros(n, dtype=bool)
entries[::20] = True
exits[10::20] = True

# VectorBT
pf = vbt.Portfolio.from_signals(
    close=pd.Series(close, index=dates),
    entries=pd.Series(entries, index=dates),
    exits=pd.Series(exits, index=dates),
    init_cash=100000, fees=0.001
)

# RaptorBT
config = raptorbt.PyBacktestConfig(initial_capital=100000, fees=0.001)
result = raptorbt.run_single_backtest(
    timestamps=dates.astype('int64').values,
    open=close, high=close, low=close, close=close,
    volume=np.ones(n), entries=entries, exits=exits,
    direction=1, weight=1.0, symbol="TEST", config=config
)

print(f"VectorBT: {pf.stats()['Total Return [%]']:.4f}%")
print(f"RaptorBT: {result.metrics.total_return_pct:.4f}%")
# Results should match within 0.01%
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Changelog

### v0.3.2

- Add `payoff_ratio` metric to `BacktestMetrics` — average winning trade return divided by average losing trade return (absolute), measures risk/reward per trade
- Add `recovery_factor` metric to `BacktestMetrics` — net profit divided by maximum drawdown in absolute terms, measures how many times over the strategy recovered from its worst drawdown
- Both metrics computed in `StreamingMetrics::finalize()` (single-instrument backtest) and `PortfolioEngine` (multi-strategy aggregation)
- Both metrics exposed via PyO3 as `#[pyo3(get)]` attributes on `PyBacktestMetrics`
- Handles edge cases: returns `f64::INFINITY` when denominator is zero with positive numerator, `0.0` otherwise

### v0.3.1

- Add Monte Carlo portfolio simulation (`simulate_portfolio_mc`) for forward risk projection
- Geometric Brownian Motion (GBM) with Cholesky decomposition for correlated multi-asset simulation
- Rayon-parallelized simulation paths with deterministic seeding (xoshiro256\*\*)
- Returns percentile paths (P5/P25/P50/P75/P95), VaR, CVaR, expected return, and probability of loss
- GIL released during simulation for maximum Python concurrency

### v0.3.0

- Per-instrument configuration via `PyInstrumentConfig` (lot_size, alloted_capital, stop/target overrides)
- Position sizes now correctly rounded to lot_size multiples
- Support for per-instrument capital allocation in basket backtests
- Future-ready fields: existing_qty, avg_price for live-to-backtest transitions

### v0.2.2

- Export `run_spread_backtest` Python binding for multi-leg options spread strategies
- Export `rolling_min` and `rolling_max` indicator functions to Python

### v0.2.1

- Add `rolling_min` and `rolling_max` indicators for LLV (Lowest Low Value) and HHV (Highest High Value) support
- NaN handling for warmup period

### v0.2.0

- Add multi-leg spread backtesting (`run_spread_backtest`) supporting straddles, strangles, vertical spreads, iron condors, iron butterflies, butterfly spreads, calendar spreads, and diagonal spreads
- Coordinated entry/exit across all legs with net premium P&L calculation
- Max loss and target profit exit thresholds for spreads
- Add `SessionTracker` for intraday session management: market hours detection, squareoff time enforcement, session high/low/open tracking
- Pre-built session configs for NSE equity (9:15-15:30), MCX commodity (9:00-23:30), and CDS currency (9:00-17:00)
- Extend `StreamingMetrics` with equity/drawdown tracking, trade recording, and `finalize()` method

### v0.1.0

- Initial release
- 5 strategy types: single, basket, pairs, options, multi
- 30+ performance metrics with full VectorBT parity
- 10 technical indicators (SMA, EMA, RSI, MACD, Stochastic, ATR, Bollinger Bands, ADX, VWAP, Supertrend)
- Stop-loss management: fixed, ATR-based, and trailing stops
- Take-profit management: fixed, ATR-based, and risk-reward targets
- PyO3 Python bindings for seamless Python integration
