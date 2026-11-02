---
name: raptorbt-timeframes
description: >
    Multi-timeframe aggregation and composite bars.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
    - raptorbt-market-data
status: stable
---

# RaptorBT Timeframes

## 1. When to use this skill

Load it to turn 1 minute bars into 5, 15 or 60 minute bars, ticks into bars,
bars into daily bars, or to build bars by count, volume, value or Renko brick.
Use it when the **grid** matters: a 30 minute bar that starts at the 09:15
open, a daily bar that follows a trading calendar, or a feed that runs for days.
Declaring a composite stream inside a strategy is the
[subscriptions](../subscriptions/SKILL.md) skill; this one is the arithmetic
underneath.

## 2. Domain overview

There is one time-window bar builder in the engine. Three entry points reach it:

- **`BarAggregator`** (and the array form `aggregate_bars`, and
  `Strategy.subscribe`): fixed units (`ms s m h d w`) on a **wall-clock grid**
  (multiples of the width from local midnight), plus count units (`tick`,
  `volume`, `value`) and `renko`. This is what a strategy's composite bars use.
- **`Resampler`** (and `resample_bars`, `resample_trades`): the same builder
  with every choice exposed, for a grid at an arbitrary **anchor**, either edge
  closed, empty intervals carried, a clock that closes bars on time and
  session-anchored buckets.
- **`raptorbt.sampling.TimeBars` and `Resample`**: two names for the common
  right-closed `Resampler` declarations.

**Event time.** Every output bar is stamped with its window end, a close time,
whichever way the inputs were stamped. A bar contains only records at or before
its stamp (right-closed on close-stamped bars), so it never contains the record
that made it complete. A streaming builder hands a bar back when the first
record *past* the window arrives; the window still open at the end of the data
exists only if you ask for it (`flush`). An array call (`aggregate_bars`,
`resample_*`) includes that last partial bar by default, flagged
`complete=False` by the resampler.

`CompositeBar` is the bar a strategy receives: `stream_id`, `step`, `unit`,
`timestamp` (window end), `open`, `high`, `low`, `close`, `volume`, `symbol`,
and `.timeframe`, the label `"5m"`, `"1h"`, `"100tick"`.

## 3. API reference

| Name | What it is |
| --- | --- |
| `BarAggregator(step, unit, tz_offset_ns=0, brick_size=0.0, stamp="open", session=None)` | Streaming. `push_bar(ts, o, h, l, c, v)` and `push_trade(ts, price, size, signed_size=0)` return the bar they completed, or `None`. `next_pending()` drains Renko bricks, `flush()` returns the open bar. `stamp` says what pushed *bar* stamps mean (`"open"` or `"close"`). |
| `aggregate_bars(timestamps, open, high, low, close, volume, step, unit, tz_offset_ns=0, brick_size=0.0, stamp="open")` | Arrays in, six arrays out (`timestamps open high low close volume`). Includes the trailing partial bar. |
| `raptorbt.strategy.Timeframe(step, unit, brick_size=0.0)` | A bar spec; `.ticks(n)`, `.volume(n)`, `.value(n)`, `.renko(brick)`. |
| `raptorbt.strategy.CompositeBar` | The bar a strategy receives. |
| `Resampler(width_ns, *, source_ns=0, anchor_ns=0, tz_offset_ns=0, closed="left", gaps="skip", close_on="clock", min_source=0, session_anchored=False)` | Streaming. `push_trade`, `push_bar`, `on_clock(ts)`, `on_session(open, ts)`, `flush()` return lists of `ResampledBar`. `next_close()` gives the next due close. |
| `resample_trades(timestamps, price, qty, width_ns, *, anchor_ns, tz_offset_ns, closed, gaps, min_source, flush=True)` | Trades to bars. Returns a dict of arrays. |
| `resample_bars(timestamps, o, h, l, c, v, width_ns, *, source_ns, anchor_ns, tz_offset_ns, closed, gaps, min_source, flush=True)` | Bars to coarser bars. Returns a dict of arrays. |
| `raptorbt.ResampledBar` | `ts` (bucket end), `open high low close volume`, `n_source`, `complete`, `closed_by`. |
| `raptorbt.sampling.TimeBars(source, period, stream, anchor=, ...)`, `Resample(source, source_period, period, stream, ...)` | Right-closed declarations; `anchor="session"` restarts the grid at each session open. |

Resampler options:

| Option | Meaning |
| --- | --- |
| `anchor_ns`, `tz_offset_ns` | Boundaries at `anchor_ns + k * width_ns`, on the local clock `tz_offset_ns` ahead of UTC. A 30 minute grid from a 09:15 open is `anchor_ns=` 09:15 as nanoseconds since local midnight with `tz_offset_ns=IST_OFFSET_NS`. |
| `closed` | `"left"`: `[start, end)`, bars stamped at their open. `"right"`: `(start, end]`, bars stamped at their close. A record exactly on a boundary closes the bar it ends under `"right"` and opens the next under `"left"`. Use `"right"` for close-stamped bars. |
| `close_on` | `"clock"`: a bar closes when `on_clock(ts)` reaches its end. `"next_print"`: when a later record arrives. |
| `gaps` | `"skip"` or `"carry"` (a flat, zero-volume bar at the last close for each empty interval). |
| `min_source` | Drop a bar built from fewer source records; with the flag `complete`, how you find short buckets. |

`closed_by` of a bar is `clock`, `print`, `carry`, `session` or `flush`.

## 4. Standard workflow

1. Decide the **grid** first: wall-clock multiples of the width
   (`aggregate_bars`, `subscribe`), or a session-open anchor (`resample_bars`).
2. For a feed that runs live or for days, use the streaming object; for a
   dataset, use the array function.
3. State what your input stamps mean: `stamp="close"` or `"open"` for the
   aggregator, `closed="right"` for close-stamped bars in the resampler.
4. Decide about the last partial bar: drop it (`flush=False`, or never `flush()`)
   if it will feed a decision.
5. Check `complete`, `n_source` and the first and last stamps before using the
   bars.

## 5. Examples

Streaming and array form on 12 one-minute bars. The streaming object returns a
bar when the next window opens, so the trailing two-bar window comes only from
`flush()`, while the array function includes it:

```python
import numpy as np
import raptorbt

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 12
close = 100.0 + np.arange(n, dtype=float)
stamps = T0 + np.arange(1, n + 1, dtype=np.int64) * MIN       # close stamps

agg = raptorbt.BarAggregator(5, "m", stamp="close")
done = []
for i in range(n):
    bar = agg.push_bar(int(stamps[i]), close[i] - 0.5, close[i] + 0.25, close[i] - 0.75,
                       close[i], 1000.0)
    if bar:
        done.append(bar)
assert [(b[0] - T0) // MIN for b in done] == [5, 10]
assert done[0][1:] == (99.5, 104.25, 99.25, 104.0, 5000.0)    # open of bar 0, close of bar 4
assert agg.flush()[5] == 2000.0                               # the open window: two bars

ts, o, h, l, c, v = raptorbt.aggregate_bars(
    stamps, close - 0.5, close + 0.25, close - 0.75, close, np.full(n, 1000.0),
    5, "m", stamp="close",
)
assert list((ts - T0) // MIN) == [5, 10, 15] and v[-1] == 2000.0   # partial bar included
```

The wall-clock grid trap. NSE bars open at 09:15, but `aggregate_bars` cuts 30
minute buckets at 09:00, 09:30, ..., so the first bar covers only 15 minutes.
`resample_bars` anchored at the open gives whole 30 minute bars:

```python
import numpy as np
import raptorbt

S = 10**9
MIN = 60 * S
IST = raptorbt.IST_OFFSET_NS
midnight_utc = 1_767_225_600 * S                       # 2026-01-01 00:00 UTC
open_ns = (9 * 3600 + 15 * 60) * S                     # 09:15 local, from local midnight
local0 = midnight_utc - IST + open_ns                  # the open, in UTC nanoseconds

n = 90                                                 # 09:16 .. 10:45 close stamps
ts = local0 + np.arange(1, n + 1, dtype=np.int64) * MIN
px = 100.0 + np.arange(n, dtype=float)
ones = np.ones(n)

wall = raptorbt.aggregate_bars(ts, px, px + 1, px - 1, px, ones, 30, "m",
                               tz_offset_ns=IST, stamp="close")
assert wall[5].tolist() == [15.0, 30.0, 30.0, 15.0]    # a 15 minute first bar and last bar

anchored = raptorbt.resample_bars(
    ts, px, px + 1, px - 1, px, ones, 30 * MIN, source_ns=MIN,
    anchor_ns=open_ns, tz_offset_ns=IST, closed="right",
)
assert anchored["n_source"].tolist() == [30, 30, 30] and anchored["complete"].all()
assert anchored["timestamp"].tolist() == [local0 + k * 30 * MIN for k in (1, 2, 3)]
assert anchored["open"][1] == 130.0                    # the second bar opens at minute 30
```

Counted bars from a trade stream: every third print, every 25 units, and Renko
bricks (a brick can complete several at once; drain them):

```python
import raptorbt

S = 10**9
T0 = 1_767_225_600 * S

ticks = raptorbt.BarAggregator(3, "tick")
volume = raptorbt.BarAggregator(25, "volume")
bars = []
for i, price in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
    for agg, name in ((ticks, "tick"), (volume, "volume")):
        bar = agg.push_trade(T0 + i * S, price, 10.0)
        if bar:
            bars.append((name, bar[1:5]))
assert bars == [("tick", (1.0, 3.0, 1.0, 3.0)), ("volume", (1.0, 3.0, 1.0, 3.0)),
                ("tick", (4.0, 6.0, 4.0, 6.0)), ("volume", (4.0, 6.0, 4.0, 6.0))]

renko = raptorbt.BarAggregator(1, "renko", brick_size=2.0)
bricks = []
for i, price in enumerate([100.0, 101.0, 103.0, 107.0, 104.0, 99.0]):
    bar = renko.push_trade(T0 + i * S, price, 10.0)
    while bar:                                          # drain every brick this print made
        bricks.append((bar[1], bar[4]))
        bar = renko.next_pending()
assert bricks == [(100.0, 102.0), (102.0, 104.0), (104.0, 106.0),
                  (106.0, 104.0), (104.0, 102.0), (102.0, 100.0)]
```

A streaming resampler. Trades land in one-minute bars; with `close_on="clock"`
the bar is handed back as soon as the clock reaches its end, with no later trade
needed, and an exactly-on-boundary trade belongs to the bar it ends under
`closed="right"`:

```python
import numpy as np
import raptorbt

S = 10**9
M = 60 * S
D0 = 1_767_225_600 * S

rs = raptorbt.Resampler(M, closed="right", close_on="clock", gaps="carry")
assert rs.push_trade(D0 + 10 * S, 100.0, 1.0) == [] and rs.push_trade(D0 + 50 * S, 101.0, 2.0) == []
assert rs.next_close() == D0 + M
assert rs.on_clock(D0 + 59 * S) == []                  # not yet
(bar,) = rs.on_clock(D0 + M)
assert (bar.ts, bar.open, bar.close, bar.volume, bar.closed_by) == (D0 + M, 100.0, 101.0, 3.0, "clock")

ts = np.array([D0 + 10 * S, D0 + M, D0 + M + S], dtype=np.int64)    # the middle print is on the edge
px, qty = np.array([100.0, 101.0, 102.0]), np.ones(3)
left = raptorbt.resample_trades(ts, px, qty, M, closed="left")
right = raptorbt.resample_trades(ts, px, qty, M, closed="right")
assert left["close"].tolist() == [100.0, 102.0]         # the edge print opens the next bar
assert right["close"].tolist() == [101.0, 102.0]        # the edge print closes its own bar
```

## 6. Composition

- Subscribe a strategy to these bars, and read them back: [subscriptions](../subscriptions/SKILL.md).
- Streaming indicators over a timeframe take `timeframe=`: [indicators](../indicators/SKILL.md).
- Daily and longer bars follow a symbol's calendar when you give
  `BarAggregator(..., session=SessionSchedule)`; the schedule's holidays and
  weekends are skipped and the day is cut at the schedule's own midnight:
  [instruments](../instruments/SKILL.md).
- Bars from ticks with signed flow: `raptorbt.bars_from_ticks`
  ([market-data](../market-data/SKILL.md)).

## 7. Constraints and correctness

- **The wall-clock grid is not the session open.** `BarAggregator` buckets are
  multiples of the width from local midnight (`tz_offset_ns`). For widths that do
  not divide the time from midnight to the open (30 minutes from 09:15, 1 hour
  from 09:15), the first and last buckets of the day are partial. Anchor with
  `Resampler`.
- **Daily bars are stamped at the end of the calendar day** (UTC midnight with
  `tz_offset_ns=0`, local midnight otherwise, or the schedule's midnight with a
  `session`), not at the session close. Set `tz_offset_ns` for the market's zone.
- **The last window.** A streaming builder never delivers the window still open.
  An array call includes it; check `complete` (resampler) or the last volume
  before trading on it. Its stamp lies after the last record, so it is not
  something you knew at that time.
- **Source width.** `Resampler(source_ns=...)` requires the width to be a
  multiple of it and the anchor to sit on its grid; a bar off the grid, a late
  record, a non-finite price or a negative volume raises and leaves the
  resampler as it was. Nothing is repaired.
- **Gaps.** `gaps="skip"` leaves holes for quiet intervals; `"carry"` fills them
  with flat zero-volume bars at the last close. Carried bars have
  `n_source == 0`; do not read them as trading.
- **Count units** (`tick`, `volume`, `value`, `renko`) in a bar run count the
  primary bars: one bar is one tick, its volume is its volume, its value is
  price times volume. A Renko brick is `brick_size` tall and one print can
  complete several; drain `next_pending()`.
- **Open or close stamps.** `BarAggregator(stamp=...)` and
  `aggregate_bars(stamp=...)` must match the input. A default of `"open"` fed
  close-stamped bars moves every bucket edge by one source bar (windows of 4, 5, 3
  bars instead of 5, 5, 2).
- **Deterministic.** The same records give the same bars on any machine.

## 8. Common mistakes

An unknown unit is refused:

<!-- skill: invalid -->
```python
import raptorbt

raptorbt.BarAggregator(5, "fortnight")   # InvalidSubscription: invalid_bar_spec
```

A resampler built for bars refuses trades, and the reverse:

<!-- skill: invalid -->
```python
import raptorbt

MIN = 60 * 10**9
rs = raptorbt.Resampler(5 * MIN, source_ns=MIN, closed="right")
rs.push_trade(1_767_225_600 * 10**9, 100.0, 1.0)   # InvalidParameter: takes bars, not trades
```

A target width that is not a multiple of the source width is refused:

<!-- skill: invalid -->
```python
import numpy as np
import raptorbt

MIN = 60 * 10**9
n = 10
ts = 1_767_225_600 * 10**9 + np.arange(1, n + 1, dtype=np.int64) * MIN
px = np.arange(n, dtype=float) + 100.0
raptorbt.resample_bars(ts, px, px, px, px, np.ones(n), 7 * MIN + 1, source_ns=MIN)  # InvalidParameter
```

- Using `aggregate_bars` for a 09:15-anchored grid.
- Reading the trailing partial bar of an array call as a finished bar.
- Feeding close-stamped bars to an `"open"` aggregator.
- Using `closed="left"` on close-stamped bars: the boundary record lands in the
  wrong bar.
- Treating a carried bar as a trade.
- Forgetting `tz_offset_ns` for daily bars of a non-UTC market.

## 9. Validation

1. Count: `n_source.sum()` equals the number of input records (trades), or
   `volume.sum()` equals the input volume (bars), when nothing was dropped.
2. The first stamp, last stamp and width of every output bar: stamps differ by
   exactly `width_ns` on a grid without gaps.
3. Check `complete` and `n_source` on the first and last bars.
4. Round trip: streaming and array builders agree bar for bar on the same data
   (`flush=False` against the streaming list).
5. Compare against the strategy: bars a strategy receives through
   `subscribe(..., timeframe=...)` equal `aggregate_bars` of the same input,
   except the final partial one.
6. Sensitivity: shift the anchor by one source bar; if results change a lot, the
   strategy is sensitive to the grid.

## 10. Related skills

- [Subscriptions](../subscriptions/SKILL.md)
- [Market data](../market-data/SKILL.md)
- [Indicators](../indicators/SKILL.md)
- [Instruments](../instruments/SKILL.md)
