---
name: raptorbt-core
description: >
    Engine concepts, the backtest lifecycle (data in, configuration, run,
    result) and the settings every run shares. Load this first.
version: 1.0
api_version: 1.x
depends_on: []
status: stable
---

# RaptorBT Core

## 1. When to use this skill

Load it for any RaptorBT task. It teaches the lifecycle every other domain
plugs into, the data layout the engine requires, and the configuration object
every runner takes. Then load only the skills your task needs, found through
[the registry](../INDEX.md) or `raptorbt.skills.for_task("<your task>")`.

## 2. Domain overview

RaptorBT is a Rust simulation engine with a Python API. A run has four steps:

1. **Data in.** Arrays of `int64` nanosecond timestamps, `float64` open, high,
   low, close and volume. Timestamps ascend; equal timestamps keep input order.
2. **Configuration.** One `BacktestConfig`: capital, fees, slippage, fill
   timing, session length, risk limits. Defaults are usable.
3. **Run.** A runner executes the strategy bar by bar, applies fills, fees and
   stops, and records every order and trade.
4. **Result.** A `BacktestResult` with metrics, an equity curve, a drawdown
   curve and the trades. See the results skill.

There are two ways to say what to trade. Class-based strategies (a `Strategy`
subclass run by `run_strategy_backtest`) decide bar by bar. Array runners
(`run_single_backtest` and relatives) take precomputed boolean entry and exit
arrays. Prefer class-based strategies for new work; use arrays for vectorised
research. Both run the same execution core and return the same result type.

Event time: a strategy sees a whole bar, close included, when the bar event
fires. A decision on bar `i` can execute no earlier than that bar's close
(see `fill_timing` below). Nothing a run does depends on wall-clock time, and
the same inputs give identical results.

## 3. API reference

Public names this skill covers (import them from `raptorbt`):

| Name | What it is |
| --- | --- |
| `BacktestConfig(...)` | Run settings. Read and write attributes after construction, except `fill_timing`, which is set in the constructor only. |
| `SESSION_NSE`, `SESSION_MCX`, `SESSION_CDS` | Session length in minutes (375, 870, 480) for `session_minutes`. |
| `SESSION_CONTINUOUS` | `0.0`: a market open around the clock. |
| `IST_OFFSET_NS` | UTC+05:30 as nanoseconds, for `session_tz_offset_ns`. |
| `raptorbt.skills` | This skill system: `index()`, `load(name)`, `manifest()`, `for_task(text)`. |

The settings you will touch most (all keyword arguments of `BacktestConfig`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `initial_capital` | `100000.0` | Starting cash, in your account currency. |
| `fees` | `0.001` | Flat fee as a fraction of traded value. Use `fee_schedule` (priced per instrument) or `fee_segment` (one row for all) for itemized statutory charges (backtesting skill). |
| `slippage` | `0.0` | Fraction of price, applied when `apply_slippage` is true. |
| `fill_timing` | `None` (means `same_bar_close`) | `"same_bar_close"`, `"next_bar_open"`; `"same_bar_open_lookahead"` exists only to reproduce old results and is not causally valid. |
| `session_minutes` | `None` (no session) | Trading minutes per session, for annualising intraday statistics. With none, intraday bars count against calendar time; name the session of a session market (`SESSION_NSE`, `SESSION_MCX`, ...). |
| `session_tz_offset_ns` | `None` (no zone named) | UTC offset of the session's local time: the zone of a trading day, DAY expiry and `squareoff_time`. `IST_OFFSET_NS`, or `0` for UTC. |
| `periods_per_year` | `None` | Explicit annualisation factor; `None` infers it from bar spacing. |
| `max_positions`, `max_drawdown_pct` | `None` | Optional limits; `None` is off. See the risk skill. |

`raptorbt.run_single_backtest(timestamps, open, high, low, close, volume,
entries, exits, direction=1, weight=1.0, symbol="UNKNOWN", config=None, ...)`
is the smallest runner. `direction` is `1` for long, `-1` for short.

## 4. Standard workflow

1. Build the arrays; check they are `float64`, `int64` timestamps, same length,
   ascending.
2. Build a `BacktestConfig`; set only what the task needs.
3. Choose a runner (backtesting skill) or a `Strategy` (strategy skill).
4. Run it. Errors are `RaptorBTError` subclasses carrying `.code`, `.valid`
   and `.skill` (troubleshooting skill).
5. Read `result.metrics`, `result.trades()` and `result.equity_curve()`.
6. Validate before trusting: see section 9.

## 5. Examples

The smallest complete run: an SMA crossover on synthetic bars, long only.
Indicator arrays carry `nan` for the warm-up bars, and comparisons with `nan`
are false, so no entry fires before both averages exist.

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, run_single_backtest

def make_bars(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return ts, open_, high, low, close, np.full(n, 1000.0)

ts, o, h, l, c, v = make_bars()
fast, slow = raptorbt.sma(c, 5), raptorbt.sma(c, 20)

config = BacktestConfig(initial_capital=100_000.0, fees=0.0005)
result = run_single_backtest(
    ts, o, h, l, c, v,
    entries=fast > slow, exits=fast < slow,
    symbol="AAPL", config=config,
)

assert result.metrics.total_trades > 0
assert len(result.equity_curve()) == len(c)
print(result.metrics.total_trades, round(result.metrics.total_return_pct, 2))
```

The same strategy as a class. `run_strategy_backtest` needs `bar_stamp` to say
whether your timestamps mark a bar's open or its close:

```python
import numpy as np
from raptorbt import BacktestConfig, Indicator, Strategy, run_strategy_backtest

def make_bars(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return ts, open_, high, low, close, np.full(n, 1000.0)

class SmaCross(Strategy):
    def on_start(self, ctx):
        self.fast = self.register_indicator(Indicator.sma(5))
        self.slow = self.register_indicator(Indicator.sma(20))

    def on_bar(self, ctx):
        fast, slow = self.fast.value, self.slow.value
        if fast is None or slow is None:  # still warming up
            return
        if ctx.is_flat and fast > slow:
            self.enter(size_frac=0.5)
        elif not ctx.is_flat and fast < slow:
            self.close_position()

ts, o, h, l, c, v = make_bars()
result = run_strategy_backtest(
    SmaCross, ts, o, h, l, c, v,
    symbol="AAPL", config=BacktestConfig(initial_capital=100_000.0),
    bar_stamp="close",
)
assert result.metrics.total_trades > 0
print(result.metrics.total_trades)
```

Fill timing changes when a signal executes, never which bars it uses:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, run_single_backtest

rng = np.random.default_rng(7)
n = 300
c = 100 + np.cumsum(rng.normal(0, 1, n))
o = np.concatenate([[c[0]], c[:-1]])
h, l = np.maximum(o, c) + 0.5, np.minimum(o, c) - 0.5
ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
v = np.full(n, 1000.0)
fast, slow = raptorbt.sma(c, 5), raptorbt.sma(c, 20)

entry_bar = {}
for timing in ("same_bar_close", "next_bar_open"):
    cfg = BacktestConfig(fill_timing=timing)
    r = run_single_backtest(ts, o, h, l, c, v, fast > slow, fast < slow, config=cfg)
    entry_bar[timing] = r.trades()[0].entry_idx

# The signal fires on one bar; the second policy fills one bar later.
assert entry_bar["next_bar_open"] == entry_bar["same_bar_close"] + 1
```

## 6. Composition

- Strategy decisions come from the [strategy](../strategy/SKILL.md) skill;
  indicators from [indicators](../indicators/SKILL.md).
- The runner is chosen in [backtesting](../backtesting/SKILL.md); several
  symbols need [instruments](../instruments/SKILL.md).
- Fees and slippage models live in [execution](../execution/SKILL.md); limits
  in [risk](../risk/SKILL.md).
- Reading a result is the [results](../results/SKILL.md) skill.
- Runnable examples: `examples/single-instrument/sma_cross.py`.

## 7. Constraints and correctness

- Timestamps are `int64` nanoseconds, never rounded or shifted by the engine.
  Descending timestamps are refused with an `InvalidParameter` naming the
  index; equal ones are allowed.
- `fees` and `slippage` are fractions (`0.001` is 0.1%), not money.
- `same_bar_close` fills at the close of the bar that produced the signal, so a
  strategy whose signal uses that close is acting on a price it just saw. That
  is causally valid for the close, but optimistic when your live system cannot
  act that fast. Use `next_bar_open` when it cannot.
- Statistics such as Sharpe are annualised from bar spacing and the session
  length; set `session_minutes` (or `periods_per_year`) when the market is not
  NSE equity hours.
- Results are deterministic: the same arrays and config give the same trades.

## 8. Common mistakes

An unknown setting value is refused, with the accepted values in the error:

<!-- skill: invalid -->
```python
from raptorbt import BacktestConfig

BacktestConfig(fill_timing="soon")  # InvalidParameter: supported are listed
```

- Passing `float32` or Python lists. Convert with `np.asarray(x, dtype=np.float64)`.
- Treating `fees=0.05` as five rupees. It is 5% of traded value.
- Reading `total_return_pct` from a run on future-peeking signals. Compute
  every signal from bars up to and including the current one.
- Forgetting `bar_stamp` on class-based runs: it is required, and omitting it
  raises `missing_bar_stamp`.
- Comparing runs with different `fill_timing`, `fees` or `session_minutes` as
  if only the strategy changed.

## 9. Validation

1. Run once with `fees=0.0` and `slippage=0.0`; the result is your strategy's
   gross edge. Then add realistic costs and compare.
2. Run with `fill_timing="next_bar_open"`. A strategy whose return collapses
   has been trading on prices it could not have reached.
3. Assert the shapes you rely on (`len(result.equity_curve()) == len(close)`).
4. Run it twice and compare trades: they must match exactly.
5. Check `result.metrics.total_trades` is not zero before reading ratios.

Before live deployment, also read the [live-trading](../live-trading/SKILL.md)
and [deployment](../deployment/SKILL.md) skills.

## 10. Related skills

- [Strategy](../strategy/SKILL.md)
- [Backtesting](../backtesting/SKILL.md)
- [Results](../results/SKILL.md)
- [Troubleshooting](../troubleshooting/SKILL.md)
