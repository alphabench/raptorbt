---
name: raptorbt-indicators
description: >
    Built-in vectorised and streaming indicators.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
status: stable
---

# RaptorBT Indicators

## 1. When to use this skill

Load it when a strategy or an array run needs a moving average, RSI, ATR,
MACD, Bollinger bands, ADX, Supertrend, VWAP or a rolling extreme, whether you
precompute it over arrays or read it bar by bar inside a hook. It also covers
readiness (when the first value exists), timeframe-specific indicators and
writing your own.

## 2. Domain overview

An indicator comes in three forms that give the same numbers:

1. **Vectorised.** `raptorbt.sma(close, 20)` takes arrays and returns an array of
   the same length. The first values are `nan` while the window fills. Use it
   for array runners and for research.
2. **Streaming.** `Indicator.sma(20)` holds state. Each `update_bar(open, high,
   low, close)` returns the new value, or `None` until the indicator is
   *initialized*. `value` and `initialized` read it back.
3. **Registered.** A strategy asks the runtime to feed an indicator for it.
   `ctx.indicators.sma(instrument=..., period=...)` returns a handle with
   `.ready` and `.value` (and `.values` for multi-output indicators). The
   runtime updates it **before your hook sees the bar** and shares work between
   strategies on the same data.

**Event time.** At `on_bar` for bar `i` the indicator already includes bar `i`'s
close and nothing later. For a timeframe (`timeframe="5m"`) the indicator is fed
composite bars, which arrive one primary bar after their stamp
([subscriptions](../subscriptions/SKILL.md)); a 3 period SMA of 5 minute bars
over 1 minute data is ready at the primary bar with index 15, not 14.

**Warm-up.** Until `ready`, `value` is `None` (streaming) or `nan` (vectorised).
Warm-up lengths, as the first index with a value:

| Indicator | First valid index | Output |
| --- | --- | --- |
| `sma(p)`, `ema(p)`, `rolling_min/max(w)` | `p - 1` | one array (the rolling functions read the series you pass, e.g. closes) |
| `rsi(p)` | `p` | one array, 0 to 100 |
| `atr(p)` | `p - 1` | one array |
| `bollinger_bands(p)` | `p - 1` | `(upper, middle, lower)` |
| `macd(fast, slow, signal)` | line `slow - 1`, signal and histogram `slow + signal - 2` | `(macd, signal, histogram)` |
| `stochastic(k, d)` | `%K` at `k - 1`, `%D` at `k + d - 2` | `(k, d)` |
| `adx(p)` | `2p - 1` | one array |
| `supertrend(p, m)` | line at `p - 1` | `(line, direction)`; direction is `0` while warming up, then `1` or `-1` |
| `vwap(...)` | `0` | one array: cumulative over the whole array, no session reset |

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.sma / ema / rsi / atr / adx / rolling_min / rolling_max` | Vectorised. `sma(data, period)`, `atr(high, low, close, period)`. |
| `raptorbt.macd(data, fast_period=12, slow_period=26, signal_period=9)` | `(macd, signal, histogram)`. |
| `raptorbt.stochastic(high, low, close, k_period=14, d_period=3)` | `(k, d)`. |
| `raptorbt.bollinger_bands(data, period=20, std_dev=2.0)` | A `raptorbt.BollingerBands(upper, middle, lower)` of arrays. |
| `raptorbt.supertrend(high, low, close, period=10, multiplier=3.0)` | `(line, direction)`. |
| `raptorbt.vwap(high, low, close, volume)` | Cumulative typical-price VWAP. |
| `raptorbt.Indicator.sma/ema/wma/wilder_ma/roc/stddev/rsi/atr(period)` | Streaming, single output. |
| `raptorbt.Indicator.donchian(period)`, `.bollinger(period, k=2.0)`, `.macd(fast, slow, signal)` | Streaming, multi output. |
| `indicator.update_bar(o, h, l, c)`, `.value`, `.initialized`, `.update_count`, `.reset()` | State. `.linked` is true once a runtime owns the indicator. |
| `ctx.indicators.sma/ema/wma/rsi/roc/stddev/atr/donchian/bollinger/macd(instrument=, period=, timeframe=None, field="close")` | Registered, shared; returns a handle with `.ready`, `.value`, `.values`. |
| `ctx.indicators.custom(obj, instrument=, timeframe=None)` | Your own indicator, fed closed bars. |
| `Strategy.register_indicator(indicator, stream_id=None, symbol=None)`, `.register_indicators(factory, symbols)` | The lower-level registration. Pass `symbol=` in a multi-symbol run. |
| `Strategy.indicators_initialized()` | True when every registered indicator is warm. |
| `raptorbt.resolve_atr_period(config, instrument_config)` | The ATR period the stops will use, or `None` (orders skill). |

Handle outputs: `donchian.values` is `(upper, lower)` from highs and lows;
`bollinger.values` is `(upper, middle, lower)`, the same `BollingerBands` the array form returns; `macd.values` is
`(macd, signal, histogram)`. Units: `roc` is a percent, `stddev` is the
population deviation (divisor `n`), `ema` is seeded with the SMA of the first
`period` values then smoothed with `2 / (period + 1)`, `atr` is the mean true
range of the first `period` bars then Wilder smoothing.

## 4. Standard workflow

1. Array run: compute the indicator, build boolean entries and exits from
   comparisons, and run an array runner. `nan` compares false, so no signal
   fires during warm-up.
2. Strategy run: register in `on_init` with `ctx.indicators.*`, with
   `timeframe=` for a higher timeframe.
3. In `on_bar`, check `handle.ready` (or `value is not None`) before using it.
4. For multi-output indicators use `handle.values` and its field names.
5. For something the library lacks, write a small class with `update(bar)`,
   `ready` and `value`, and register it with `ctx.indicators.custom`.
6. Confirm the streaming value equals the vectorised value at the same bar.

## 5. Examples

Vectorised indicators, shapes and ordering. The functions return `nan` for
warm-up bars, so a comparison is false there:

```python
import numpy as np
import raptorbt

rng = np.random.default_rng(2)
n = 60
close = 100.0 + np.cumsum(rng.normal(0, 1, n))
high = close + rng.uniform(0.2, 1.0, n)
low = close - rng.uniform(0.2, 1.0, n)

def first_valid(a):
    return int(np.argmax(~np.isnan(a)))

fast, slow = raptorbt.sma(close, 5), raptorbt.sma(close, 20)
assert len(fast) == n and first_valid(fast) == 4 and first_valid(slow) == 19
assert not (fast > slow)[:19].any()                      # nan compares false

upper, middle, lower = raptorbt.bollinger_bands(close, 20, 2.0)
assert np.isclose(middle[25], close[6:26].mean()) and upper[25] > middle[25] > lower[25]

line, signal, hist = raptorbt.macd(close)
assert first_valid(line) == 25 and first_valid(signal) == 33
assert np.isclose(hist[40], line[40] - signal[40])

k, d = raptorbt.stochastic(high, low, close, 14, 3)
assert first_valid(k) == 13 and first_valid(d) == 15
assert first_valid(raptorbt.rsi(close, 14)) == 14 and first_valid(raptorbt.atr(high, low, close, 14)) == 13
assert first_valid(raptorbt.adx(high, low, close, 14)) == 27

# A nan in the input poisons what follows: the SMA recovers, the EMA never does.
bad = close.copy()
bad[10] = np.nan
assert np.isnan(raptorbt.sma(bad, 5)).sum() == 9 and np.isnan(raptorbt.ema(bad, 5)).sum() == 54
```

Streaming gives the same numbers, and says when it is ready:

```python
import numpy as np
import raptorbt
from raptorbt import Indicator

rng = np.random.default_rng(2)
n = 60
close = 100.0 + np.cumsum(rng.normal(0, 1, n))
high, low = close + 0.5, close - 0.5

streams = {"sma": (Indicator.sma(5), raptorbt.sma(close, 5), 4),
           "ema": (Indicator.ema(5), raptorbt.ema(close, 5), 4),
           "rsi": (Indicator.rsi(14), raptorbt.rsi(close, 14), 14),
           "atr": (Indicator.atr(14), raptorbt.atr(high, low, close, 14), 13)}
for name, (ind, ref, ready_at) in streams.items():
    for i in range(n):
        value = ind.update_bar(close[i], high[i], low[i], close[i])
        assert (value is None) == (i < ready_at) == (not ind.initialized), (name, i)
        if value is not None:
            assert np.isclose(value, ref[i]), (name, i)
    assert ind.update_count == n

macd = Indicator.macd()
for i in range(n):
    macd.update_bar(close[i], high[i], low[i], close[i])
line, signal, hist = macd.value
assert np.isclose(hist, line - signal)
```

Indicators in a strategy. They update before `on_bar`, share work, and a
timeframe indicator is warm later than a primary one. The value at `on_bar(i)`
equals the vectorised value at `i`:

```python
import numpy as np
import raptorbt
from raptorbt import Strategy, run_strategies

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 60
rng = np.random.default_rng(5)
close = 100.0 + np.cumsum(rng.normal(0, 1, n))
open_ = np.concatenate([[close[0]], close[:-1]])
bars = dict(timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
            open=open_, high=np.maximum(open_, close) + 0.3,
            low=np.minimum(open_, close) - 0.3, close=close, volume=np.full(n, 1000.0))

class Signals(Strategy):
    def on_init(self, ctx):
        self.fast = ctx.indicators.sma(instrument="ASSET", period=5)
        self.slow = ctx.indicators.sma(instrument="ASSET", period=20)
        self.bands = ctx.indicators.bollinger(instrument="ASSET", period=20, k=2.0)
        self.five = ctx.indicators.sma(instrument="ASSET", period=3, timeframe="5m")

    def on_start(self, ctx):
        self.seen = []
        self.five_ready_at = None

    def on_bar(self, ctx, event):
        if event.timeframe is not None:
            return
        if self.five.ready and self.five_ready_at is None:
            self.five_ready_at = ctx.idx
        if self.slow.ready:
            self.seen.append((ctx.idx, self.fast.value, self.slow.value, self.bands.values))

s = Signals()
run_strategies([s], {"ASSET": bars}, bar_stamp="close")

fast, slow = raptorbt.sma(close, 5), raptorbt.sma(close, 20)
assert s.seen[0][0] == 19                                # warm at the 20th bar
for idx, f, sl, bands in s.seen:
    assert np.isclose(f, fast[idx]) and np.isclose(sl, slow[idx])   # includes bar idx, nothing later
assert bands.middle < bands.upper and bands.lower < bands.middle
assert s.five_ready_at == 15                             # 3 five-minute bars, delivered a bar late
```

Your own indicator. `update(bar)` runs once per closed bar before your hooks,
`ready` and `value` are attributes:

```python
import numpy as np
from raptorbt import Strategy, run_strategies

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 30
close = 100.0 + np.cumsum(np.random.default_rng(5).normal(0, 1, n))
bars = dict(timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
            open=close, high=close + 0.5, low=close - 0.5, close=close,
            volume=np.full(n, 1000.0))

class MeanRange:
    """Average high-low range of the last five bars."""
    def __init__(self):
        self.ranges = []

    def update(self, bar):
        self.ranges.append(bar.high - bar.low)

    @property
    def ready(self):
        return len(self.ranges) >= 5

    @property
    def value(self):
        return float(np.mean(self.ranges[-5:])) if self.ready else None

class UseIt(Strategy):
    def on_init(self, ctx):
        self.mean_range = ctx.indicators.custom(MeanRange(), instrument="ASSET")

    def on_start(self, ctx):
        self.rows = []

    def on_bar(self, ctx):
        self.rows.append((ctx.idx, self.mean_range.ready, self.mean_range.value))

s = UseIt()
run_strategies([s], {"ASSET": bars}, bar_stamp="close")
assert not s.rows[3][1] and s.rows[4][1] and abs(s.rows[4][2] - 1.0) < 1e-12
```

## 6. Composition

- Timeframes, and when a composite bar arrives: [subscriptions](../subscriptions/SKILL.md)
  and [timeframes](../timeframes/SKILL.md).
- Building entries and exits from these arrays for an array runner:
  [backtesting](../backtesting/SKILL.md). Using indicator values in `on_bar`:
  [strategy](../strategy/SKILL.md).
- ATR-based stops and targets are configuration, not indicators: [orders](../orders/SKILL.md).

## 7. Constraints and correctness

- **No future in the stream.** A registered indicator holds bars up to and
  including the current one. A vectorised array is safe the same way: element
  `i` uses bars `0..i`. Do not shift it backwards (`x[1:]`) to "align".
- **Compare a signal to the previous bar yourself.** A crossover is
  `fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]`; the first bar has no
  previous value.
- **`nan` and `None` are warm-up, not zero.** Test `handle.ready`; a streaming
  indicator is `None` until then.
- **Inputs must be finite.** A `nan` in the close poisons the EMA, RSI and ATR
  that follow it for the rest of the run; the SMA recovers after its window.
- **Bollinger has one order, with names.** The array `bollinger_bands`, the
  streaming `Indicator.bollinger` value and the registered handle's `.value` and
  `.values` are all `BollingerBands(upper, middle, lower)`, a named tuple; read
  `bands.middle` rather than by position. Before this release the streaming and
  registered form was `(middle, upper, lower)`, so a positional unpack written
  for that order now reads the wrong fields.
- **VWAP is cumulative** over the array you pass and does not reset per day.
  Slice by session first.
- **Rolling extremes read the series you give them.** `rolling_max(close, w)`
  is the highest close, not the highest high. `Indicator.donchian` uses highs
  and lows.
- **Registered indicators are read-only.** `update_bar` and `reset` on one the
  runtime owns raise `RuntimeError` (`.linked` is true); read `value` and
  `initialized`.
- **A registered indicator on another instrument is current as of the timestamp you
  decide at.** When several instruments print at one timestamp, every one of their
  indicators has taken that bar before any `on_bar` runs, so
  `self.fast["OTHER"].value` in the hook for `"THIS"` equals the value computed
  from OTHER's bars through that timestamp, whatever the symbols are called. Only
  your own event is revealed in steps: an indicator on the event's own primary
  stream shows the value before the bar in a composite-bar hook and the value after
  it in `on_bar`; one on an own composite stream shows the value after the bar
  the hook is handling (after brick `k` when an event completes several).
- **Pass `symbol=` to `register_indicator` in a multi-symbol run.** Without it
  the indicator is fed every symbol's bars interleaved and the runtime warns; it has
  no own stream, so it shows the value after the whole timestamp at every decision.
- **Only the close** can be the input of a registered indicator (`field="close"`);
  ATR and Donchian read the whole bar.
- **Periods start at 1.** A period below one raises `InvalidParameter`
  (`invalid_parameter`) from every vectorised function and every
  `Indicator.*` constructor alike; Bollinger needs `k > 0` and MACD needs
  `fast < slow`.

## 8. Common mistakes

An indicator input other than the close is refused:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategies

class HighAverage(Strategy):
    def on_init(self, ctx):
        ctx.indicators.sma(instrument="ASSET", period=5, field="high")  # invalid_indicator_input

n = 30
c = np.linspace(100.0, 130.0, n)
bars = dict(timestamps=np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9,
            open=c, high=c, low=c, close=c, volume=np.ones(n))
run_strategies([HighAverage()], {"ASSET": bars}, bar_stamp="close")
```

A custom indicator must provide `update`, `ready` and `value`:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategies

class Incomplete:
    def update(self, bar):
        pass

class UseIt(Strategy):
    def on_init(self, ctx):
        ctx.indicators.custom(Incomplete(), instrument="ASSET")   # TypeError

n = 30
c = np.linspace(100.0, 130.0, n)
bars = dict(timestamps=np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9,
            open=c, high=c, low=c, close=c, volume=np.ones(n))
run_strategies([UseIt()], {"ASSET": bars}, bar_stamp="close")
```

An indicator the runtime owns cannot be updated by hand:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Indicator, Strategy, run_strategies

class Owned(Strategy):
    def on_init(self, ctx):
        self.sma = Indicator.sma(3)
        self.register_indicator(self.sma, symbol="ASSET")

    def on_bar(self, ctx):
        self.sma.update_bar(1.0, 1.0, 1.0, 1.0)   # RuntimeError: linked to a shared runtime

n = 30
c = np.linspace(100.0, 130.0, n)
bars = dict(timestamps=np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9,
            open=c, high=c, low=c, close=c, volume=np.ones(n))
run_strategies([Owned()], {"ASSET": bars}, bar_stamp="close")
```

- Trading on `fast > slow` as a state when you meant the crossing event.
- Reading `.value` without checking `.ready` and comparing `None` to a number.
- Unpacking a Bollinger value positionally from code written for the old
  streaming order `(middle, upper, lower)`.
- Computing an indicator over the whole series in `on_start` and indexing it
  with `ctx.idx + 1`.
- Using `vwap` across several days as if it reset at the open.

## 9. Validation

1. Assert the first valid index against the warm-up table; a different
   index means a different period or input length.
2. Compare the registered value at `on_bar(i)` with the vectorised array at `i`
   for several `i`.
3. Poison test: replace every bar after a cut with garbage and confirm the
   indicator values up to the cut are identical.
4. Confirm no entry fires while any input indicator is not ready.
5. For a timeframe indicator, log `ctx.idx` at the first `ready` and confirm it
   is the delivery bar of the Nth composite bar.
6. Check `np.isfinite` on the inputs; a single `nan` explains an indicator that
   stays `nan`.

## 10. Related skills

- [Subscriptions](../subscriptions/SKILL.md)
- [Timeframes](../timeframes/SKILL.md)
- [Strategy](../strategy/SKILL.md)
- [Backtesting](../backtesting/SKILL.md)
