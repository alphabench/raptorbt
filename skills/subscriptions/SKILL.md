---
name: raptorbt-subscriptions
description: >
    Declaring which streams and bar histories a strategy receives.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
    - raptorbt-market-data
status: stable
---

# RaptorBT Subscriptions

## 1. When to use this skill

Load it when a strategy needs bars of a width other than the input's (5 minute
bars from a 1 minute feed), a history of those bars, only some symbols' trades,
quotes or books, or the last quote and book of a symbol. A single-timeframe
strategy that reads `on_bar` needs no subscription.

## 2. Domain overview

`Strategy.subscribe(instrument, data_type="bar", timeframe=None, history=None)`
declares interest in one instrument's data. It does two things.

1. **Builds streams.** A `timeframe` creates a composite bar stream, built from
   that instrument's own primary bars, with a buffer of the closed bars you can
   read back through `ctx.data`. Several subscriptions to the same timeframe
   share one stream. The returned handle is the stream id.
2. **Filters ticks.** Once a strategy calls `subscribe` at all, it receives
   trades, quotes and books only for what it subscribed to. A strategy that
   never calls it receives every event for every symbol in the run.

The call is a declaration, so it belongs in `on_init` (or `on_start`). After the
first bar it raises `StateError` (`declaration_outside_run`).

**Event time.** A composite bar is stamped at its window end and holds only
bars that closed at or before that stamp. It is delivered when the *next* primary
bar of the same instrument arrives, before that bar's own `on_bar`. So when a 5
minute bar ending 09:20 reaches you, the clock (`ctx.bar_close_ts`) already reads
the 09:21 bar's close, and `event.timestamp` is 09:20. The delay is deliberate:
the engine cannot know a window is over until a bar past it exists. The last
window of a run is therefore never delivered.

**Where bars arrive.** Define `on_bar(self, ctx, event)` (two required
parameters) to receive primary bars (`event.timeframe is None`) and composite
bars (`event.timeframe` is a label such as `"5m"`). Define `on_bar(self, ctx)`,
or give `event` a default, and it sees primary bars only. `on_market_data(ctx,
event)` sees bars, composite bars, trades, quotes and books, after the hook
specific to the event.

## 3. API reference

| Name | What it is |
| --- | --- |
| `self.subscribe(instrument, data_type="bar", timeframe=None, history=None)` | Declare interest. `data_type` is `"bar"`, `"trade"`, `"quote"` or `"book"`. `timeframe` (bars only) is `None` (primary), one of `"1m" "3m" "5m" "10m" "15m" "30m" "1h" "2h" "4h" "1d" "1w"`, or a `Timeframe` / `(step, unit)` for other widths. Returns the stream id for a composite timeframe, else `None`. |
| `raptorbt.strategy.Timeframe` | `Timeframe(step, unit, brick_size=0.0)`; `Timeframe.ticks(n)`, `.volume(n)`, `.value(n)`, `.renko(brick)`. Units: `ms s m h d w tick volume value renko`. |
| `history=` | Rows to keep: the buffer of a composite stream (default 1000) and, on a live node, the primary rows kept. |
| `ctx.data.latest(instrument, timeframe=None)` | The last closed bar, or `None` before the first. A composite timeframe must be subscribed. |
| `ctx.data.history(instrument, timeframe=None, count=None)` | `BarHistory`: read-only `timestamps open high low close volume`, oldest first. |
| `ctx.data.quote(instrument)`, `ctx.data.book(instrument)` | The last recorded quote or book, or `None`. Needs a `data_type="quote"` or `"book"` subscription. |
| `ctx.data.position(instrument)` | The earliest open position of that instrument, or `None`. |
| `raptorbt.strategy.BarHistory` | The history type; `len(h)` is the bar count. |
| `raptorbt.InvalidSubscription` | Raised for a bad timeframe, data type or unsubscribed read. `.code` and `.valid` say what to change. |
| `self.subscribe_bars(step, unit)` | The older spelling: a stream fed to **every** symbol, delivered as composite events. Prefer `subscribe`. |

Errors you can meet, by `.code`: `invalid_timeframe` (lists the accepted
strings), `invalid_data_type`, `invalid_history`, `unknown_symbol`,
`not_subscribed_timeframe`, `not_subscribed_data`, `declaration_outside_run`.

## 4. Standard workflow

1. In `on_init`, call `self.subscribe(symbol, timeframe=...)` for each
   instrument and width you will read, with `history=` at least as long as the
   longest window you compute.
2. Define `on_bar(self, ctx, event)`. Branch on `event.timeframe`.
3. On a composite event, read `ctx.data.history(symbol, "5m", count)` and update
   your state. On a primary event, act on that state.
4. For ticks, subscribe `"trade"`, `"quote"` or `"book"` for exactly the symbols
   you trade, then read `ctx.data.quote(symbol)` when needed.
5. Use indicators with `timeframe=` (indicators skill) instead of recomputing
   from history when a built-in fits.

## 5. Examples

Two symbols, one subscribed to a 5 minute stream. Only that symbol's composite
bars arrive, each before the primary bar that revealed it, and the buffer is
capped at `history`:

```python
import numpy as np
from raptorbt import Strategy, run_strategies

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9

def bars(base, n=30):
    close = base + np.arange(n, dtype=float)
    return dict(timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
                open=close - 0.5, high=close + 0.25, low=close - 0.75, close=close,
                volume=np.full(n, 1000.0))

class FiveMinute(Strategy):
    def on_init(self, ctx):
        self.stream = self.subscribe("AAA", timeframe="5m", history=3)

    def on_start(self, ctx):
        self.events = []

    def on_bar(self, ctx, event):
        if event.timeframe is None:
            self.events.append((ctx.symbol, ctx.idx, None))
            return
        h = ctx.data.history("AAA", "5m")
        assert len(h) <= 3                                   # the declared history
        assert ctx.data.latest("AAA", "5m").timestamp == event.timestamp == h.timestamps[-1]
        self.events.append((event.symbol, ctx.idx, event.timeframe))

s = FiveMinute()
run_strategies([s], {"AAA": bars(100.0), "BBB": bars(200.0)}, bar_stamp="close")

composites = [e for e in s.events if e[2] is not None]
assert {sym for sym, _, _ in composites} == {"AAA"}          # BBB never subscribed
assert [idx for _, idx, _ in composites] == [5, 10, 15, 20, 25]
assert s.events.index(("AAA", 5, None)) > s.events.index(("AAA", 5, "5m"))  # composite first
```

The first 5 minute bar holds primary bars 0 to 4 (open of bar 0, close of bar 4).
It is stamped at the end of bar 4 and arrives on bar 5:

```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 30
close = 100.0 + np.arange(n, dtype=float)

class First(Strategy):
    def on_init(self, ctx):
        self.subscribe("ASSET", timeframe="5m")

    def on_start(self, ctx):
        self.first = None

    def on_bar(self, ctx, event):
        if event.timeframe and self.first is None:
            self.first = (ctx.idx, ctx.bar_close_ts, event)

s = First()
run_strategy_backtest(
    s, timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
    open=close - 0.5, high=close + 0.25, low=close - 0.75, close=close,
    volume=np.full(n, 1000.0), symbol="ASSET", bar_stamp="close",
)
idx, now, bar = s.first
assert idx == 5 and bar.timestamp == T0 + 5 * MIN       # window end = close of bar 4
assert now == T0 + 6 * MIN                              # delivered with bar 5
assert (bar.open, bar.close, bar.volume) == (99.5, 104.0, 5000.0)
```

Trades, quotes and books are filtered by subscription. With two symbols but one
trade subscription, only that symbol's prints arrive:

```python
import numpy as np
from raptorbt import Strategy, run_tick_strategy

S = 10**9
T0 = 1_767_225_600 * S
n = 50

def tape(base):
    ts = T0 + np.arange(1, n + 1, dtype=np.int64) * S
    px = base + np.arange(n) * 0.01
    return dict(timestamps=ts, ltp=px, bid=px - 0.01, ask=px + 0.01)

class OneName(Strategy):
    def on_init(self, ctx):
        self.subscribe("AAA", data_type="trade")
        self.subscribe("AAA", data_type="quote")

    def on_start(self, ctx):
        self.trades, self.quotes, self.last_quote = [], 0, None

    def on_trade(self, ctx, tick):
        self.trades.append(tick.symbol)

    def on_quote(self, ctx, quote):
        self.quotes += 1
        self.last_quote = ctx.data.quote("AAA")

s = OneName()
run_tick_strategy(s, {"AAA": tape(100.0), "BBB": tape(200.0)})
assert set(s.trades) == {"AAA"} and len(s.trades) == n
assert s.quotes == n and s.last_quote.symbol == "AAA"
```

## 6. Composition

- What the arrays and ticks must look like, and how far `ctx.data` sees, is
  [market-data](../market-data/SKILL.md).
- Resampling grids, session-anchored bars and the aggregator's units are
  [timeframes](../timeframes/SKILL.md).
- Streaming indicators over a subscribed timeframe use
  `ctx.indicators.sma(instrument=..., period=..., timeframe="5m")`: see
  [indicators](../indicators/SKILL.md).
- The hooks themselves, and order placement from them, are the
  [strategy](../strategy/SKILL.md) skill.
- Runnable examples: `examples/multi-timeframe/trend_confirmation.py`.

## 7. Constraints and correctness

- **Composite bars are one primary bar late.** Act on a 5 minute bar from the
  primary bar that delivered it, never at its own stamp.
- **Closed bars only.** `ctx.data.latest` and `history` never include a bar
  still forming. The in-progress window is not readable.
- **Per instrument.** A composite stream is built from that instrument's bars
  only. In a portfolio run it arrives on `ctx.symbol` and carries
  `event.symbol`. A symbol you did not subscribe to has no stream:
  `ctx.data.history(other, "5m")` raises `not_subscribed_timeframe`.
- **Primary bars are not filtered.** `subscribe("AAA")` for primary bars does not
  stop `on_bar` firing for the run's other symbols.
- **Subscribing narrows ticks for the whole strategy.** Once any `subscribe` call
  has run, a trade, quote or book type you did not subscribe is dropped, for
  every symbol. Subscribe each type you handle.
- **Buffer size.** A composite buffer keeps `history` closed bars (1000 by
  default). Asking for more returns what is kept. Raise `history=` to the
  longest window you compute; repeated subscriptions use the largest.
- **Declare early.** `subscribe` and `ctx.indicators.*` work only in `on_init`
  and `on_start`.
- **One window short.** The window still open when the data ends is not
  delivered. A backtest on an exact multiple of the width ends one composite
  bar before the data does.
- **`subscribe_bars` fans out to every symbol** and is delivered even for
  symbols you did not name; use `subscribe` for per-instrument control.

## 8. Common mistakes

Reading a timeframe you did not subscribe to fails with the list of what you did:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

MIN = 60 * 10**9

class Wrong(Strategy):
    def on_init(self, ctx):
        self.subscribe("ASSET", timeframe="5m")

    def on_bar(self, ctx, event):
        ctx.data.history("ASSET", "15m")   # InvalidSubscription: not_subscribed_timeframe

c = np.linspace(100.0, 130.0, 30)
run_strategy_backtest(Wrong, timestamps=np.arange(1, 31, dtype=np.int64) * MIN,
                      open=c, high=c, low=c, close=c, volume=np.ones(30),
                      symbol="ASSET", bar_stamp="close")
```

A width that is not a supported string is refused; use a tuple for others:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

MIN = 60 * 10**9

class Odd(Strategy):
    def on_init(self, ctx):
        self.subscribe("ASSET", timeframe="7m")   # InvalidSubscription: invalid_timeframe

c = np.linspace(100.0, 130.0, 30)
run_strategy_backtest(Odd, timestamps=np.arange(1, 31, dtype=np.int64) * MIN,
                      open=c, high=c, low=c, close=c, volume=np.ones(30),
                      symbol="ASSET", bar_stamp="close")
```

A declaration after the run started is refused:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

MIN = 60 * 10**9

class Late(Strategy):
    def on_bar(self, ctx, event):
        self.subscribe("ASSET", timeframe="5m")   # StateError: declaration_outside_run

c = np.linspace(100.0, 130.0, 30)
run_strategy_backtest(Late, timestamps=np.arange(1, 31, dtype=np.int64) * MIN,
                      open=c, high=c, low=c, close=c, volume=np.ones(30),
                      symbol="ASSET", bar_stamp="close")
```

- Acting on a composite bar at its stamp and expecting the fill price to be that
  bar's close: it arrives one primary bar later.
- Defining `on_bar(self, ctx)` and wondering why no composite event arrives.
  Take the `event` parameter.
- Subscribing only `"trade"` and then reading `ctx.data.quote(...)`.
- Computing a 20 bar average of 5 minute closes from `history=10`.
- Forgetting that subscribing quotes for one symbol drops quotes for the others.

## 9. Validation

1. Log `(ctx.idx, event.timeframe, event.timestamp, ctx.bar_close_ts)` for the
   first few composite events and confirm `event.timestamp < ctx.bar_close_ts`.
2. Rebuild the composite bars from the primary arrays with
   `raptorbt.aggregate_bars(..., stamp="close")` and compare them with what the
   run delivered (the aggregate also holds the final partial window).
3. Assert `len(ctx.data.history(symbol, tf)) >= your_window` before using it.
4. Count trades per symbol and confirm only subscribed symbols appear.
5. Run the same strategy with `bar_stamp="open"` on re-stamped data; composite
   events must match exactly.

## 10. Related skills

- [Market data](../market-data/SKILL.md)
- [Timeframes](../timeframes/SKILL.md)
- [Indicators](../indicators/SKILL.md)
- [Strategy](../strategy/SKILL.md)
