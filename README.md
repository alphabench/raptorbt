# RaptorBT

**RaptorBT** is a high-performance backtesting engine written in Rust with Python bindings via PyO3. It serves as a drop-in replacement for VectorBT, providing significant performance improvements while maintaining full metric parity.

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
- [Python Integration](#python-integration)
- [VectorBT Drop-in Replacement](#vectorbt-drop-in-replacement)
- [API Reference](#api-reference)
- [Building from Source](#building-from-source)
- [Testing](#testing)

---

## Overview

RaptorBT was built to address the performance limitations of VectorBT in production environments:

| Metric                        | VectorBT            | RaptorBT     | Improvement               |
| ----------------------------- | ------------------- | ------------ | ------------------------- |
| **Disk Footprint**            | ~450MB              | <10MB        | **45x smaller**           |
| **Startup Latency**           | 200-600ms           | <10ms        | **20-60x faster**         |
| **Backtest Speed (1K bars)**  | 1460ms              | 0.25ms       | **5,800x faster**         |
| **Backtest Speed (50K bars)** | 43ms                | 1.7ms        | **25x faster**            |
| **Memory Usage**              | High (JIT + pandas) | Low (native) | **Significant reduction** |

### Key Features

- **5 Strategy Types**: Single instrument, basket/collective, pairs trading, options, and multi-strategy
- **30+ Metrics**: Full parity with VectorBT including Sharpe, Sortino, Calmar, Omega, SQN, and more
- **10 Technical Indicators**: SMA, EMA, RSI, MACD, Stochastic, ATR, Bollinger Bands, ADX, VWAP, Supertrend
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
│   │   └── timeseries.rs  # Time series utilities
│   │
│   ├── strategies/        # Strategy implementations
│   │   ├── single.rs      # Single instrument backtest
│   │   ├── basket.rs      # Basket/collective strategies
│   │   ├── pairs.rs       # Pairs trading
│   │   ├── options.rs     # Options strategies
│   │   └── multi.rs       # Multi-strategy combining
│   │
│   ├── indicators/        # Technical indicators
│   │   ├── trend.rs       # SMA, EMA, Supertrend
│   │   ├── momentum.rs    # RSI, MACD, Stochastic
│   │   ├── volatility.rs  # ATR, Bollinger Bands
│   │   ├── strength.rs    # ADX
│   │   └── volume.rs      # VWAP
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
result = raptorbt.run_single_backtest(
    timestamps=timestamps,
    open=open_prices, high=high_prices, low=low_prices,
    close=close_prices, volume=volume,
    entries=entries, exits=exits,
    direction=1,  # 1=Long, -1=Short
    weight=1.0,
    symbol="SYMBOL",
    config=config,
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

result = raptorbt.run_basket_backtest(
    instruments=instruments,
    config=config,
    sync_mode="all",  # "all", "any", "majority", "master"
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

## Python Integration

RaptorBT integrates seamlessly with the Quant5 golf runner through `rpbt.py`.

### Enable RaptorBT

```bash
export USE_RAPTORBT=1
```

Or in Python:

```python
import os
os.environ["USE_RAPTORBT"] = "1"
```

### Integration Functions

```python
from app.engine.golf.rpbt import (
    is_raptorbt_enabled,
    RaptorBTConfig,
    RaptorBTPortfolioWrapper,
    run_single_backtest_raptorbt,
    run_basket_backtest_raptorbt,
    run_pairs_backtest_raptorbt,
    run_options_backtest_raptorbt,
    run_multi_backtest_raptorbt,
)

# Check if RaptorBT is enabled
if is_raptorbt_enabled():
    print("Using RaptorBT backend")
```

---

## VectorBT Drop-in Replacement

RaptorBT provides a `RaptorBTPortfolioWrapper` that mimics the VectorBT Portfolio interface:

```python
from app.engine.golf.rpbt import (
    RaptorBTPortfolioWrapper,
    run_single_backtest_raptorbt,
    RaptorBTConfig,
)

# Run backtest
result = run_single_backtest_raptorbt(compiled, ohlcv_df, config, symbol)

# Wrap result for VectorBT compatibility
portfolio = RaptorBTPortfolioWrapper(result)

# Use like VectorBT Portfolio
stats = portfolio.stats()           # Returns pd.Series with VectorBT-format keys
equity = portfolio.value()          # Returns equity curve as pd.Series
dd = portfolio.drawdown()           # Returns drawdown curve as pd.Series
trades_df = portfolio.trades()      # Returns trades as pd.DataFrame

# Access properties
print(portfolio.total_return)       # Total return percentage
print(portfolio.sharpe_ratio)       # Sharpe ratio
print(portfolio.max_drawdown)       # Max drawdown percentage
print(portfolio.win_rate)           # Win rate percentage
print(portfolio.profit_factor)      # Profit factor
print(portfolio.sqn)                # System Quality Number
print(portfolio.expectancy)         # Expected value per trade
print(portfolio.omega_ratio)        # Omega ratio
```

### Stats Format

The `stats()` method returns a pandas Series with VectorBT-compatible keys:

```python
stats = portfolio.stats()
print(stats["Total Return [%]"])
print(stats["Sharpe Ratio"])
print(stats["Max Drawdown [%]"])
print(stats["Win Rate [%]"])
print(stats["Profit Factor"])
print(stats["SQN"])
print(stats["Omega Ratio"])
# ... and 20+ more metrics
```

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

### Using the Build Script

```bash
./scripts/build-engine.sh --install
```

---

## Testing

### Rust Unit Tests

```bash
cd raptorbt
cargo test
```

### Python Integration Tests

```bash
# Test basic functionality
uv run python -c "
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
"
```

### Comparison Test (VectorBT vs RaptorBT)

```bash
USE_RAPTORBT=1 uv run python << 'EOF'
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
print(f"Match: {abs(pf.stats()['Total Return [%]'] - result.metrics.total_return_pct) < 0.01}")
EOF
```

---

## License

RaptorBT is proprietary software developed for the Quant5 platform.

---

## Changelog

### v0.1.0 (2024-01)

- Initial release
- 5 strategy types: single, basket, pairs, options, multi
- 30+ performance metrics
- 10 technical indicators
- Fixed, ATR, and trailing stops
- PyO3 Python bindings
- VectorBT-compatible wrapper
