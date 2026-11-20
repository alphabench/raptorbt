---
name: raptorbt-market-data
description: >
    Bars, trades, quotes and order-book inputs, and tick-level features.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
    - raptorbt-instruments
status: stable
---

# RaptorBT Market Data

## 1. When to use this skill

Load it when you prepare the data a run consumes (bars, trade prints, quotes,
depth), when you need to know exactly what a strategy may see at a given
moment, or when you compute tick features. It is where the no-lookahead rules
live. Which streams a strategy asks for, and bars of other widths, are the
[subscriptions](../subscriptions/SKILL.md) and [timeframes](../timeframes/SKILL.md)
skills.

## 2. Domain overview

A run consumes up to four kinds of market data per symbol:

| Kind | Input | Hook | Fills orders? |
| --- | --- | --- | --- |
| Bars | `{"timestamps", "open", "high", "low", "close", "volume"}` | `on_bar` | Yes |
| Trades | `ticks[symbol]` with `timestamps`, `ltp` (+ `ltq`, `oi`, flow deltas) | `on_trade` | Yes |
| Quotes | `ticks[symbol]` rows with `bid` and `ask` (+ `bid_qty`, `ask_qty`) | `on_quote` | **No** |
| Books | `depth[symbol]`: `bid_prices`, `bid_sizes`, `ask_prices`, `ask_sizes`, each `(n, levels)` | `on_book` | **No** |

Quotes and books are observation only. They never fill an order, move a
trailing stop or mark equity. An order placed from `on_quote` rests and matches
against the next trade print, the first evidence that something traded.

**Event time.** The engine's clock is the instant a bar *closed*
(`ctx.bar_close_ts`). A strategy sees a whole bar, close included, when that
bar's event fires, and nothing after it. `bar_stamp` tells the engine what your
timestamps mean: `"close"` (a bar is stamped when it ends) or `"open"` (when it
starts, with `bar_ns`, the bar width in nanoseconds, so the close is
`stamp + bar_ns`). `ctx.timestamp` and `ctx.bar.timestamp` stay your own stamps;
`ctx.bar_close_ts` is the clock. Ticks are instants and ignore both.

**What a strategy may read.** `ctx.data` is the safe view: everything up to and
including the current event, read-only.

- `ctx.data.latest(symbol)`: the last closed bar of that symbol, or `None`.
- `ctx.data.history(symbol, count=None)`: read-only arrays of the last closed
  bars, oldest first (`timestamps`, `open`, `high`, `low`, `close`, `volume`).
- `ctx.data.quote(symbol)` and `ctx.data.book(symbol)`: the last recorded
  quote or book, after a matching `subscribe` (subscriptions skill).

In a run with several symbols, a symbol's `latest` and `history` hold the bars
that had closed by the current event, in merged time order. Symbols may be
stamped on different clocks; each sees only what has closed.

**`ctx.series(symbol)`** is a symbol's bars once it has had a bar, otherwise its
tick rows (`timestamps`, `ltp`, `bid`, `ask`, ...). The bars `primary_bars` builds
from a symbol's prints join its bars from the symbol's first pushed or given bar on,
in the order `on_bar` hears them, the bar being delivered included, the same in a
batch run, a windowed run and a live node. A built bar is stamped in the series as a
given bar of the same close would be (its open under `bar_stamp="open"`); the
`bar.timestamp` that `on_bar` receives for a built bar is its close whatever
`bar_stamp` says. Bars built before the symbol's first given bar do not join, and a
symbol fed only ticks keeps its tick series.

**Ticks on a live node.** `LiveNode.push_tick(symbol, ts, ltp, bid, ask, ...)` is one
row, as in a batch tape: `ltp > 0` is a trade, `bid > 0` and `ask > 0` together are a
quote, heard after that row's trade. `bid` and `ask` not given are 0: the row is a
trade only, and `ctx.best_bid` / `ctx.best_ask` keep the last two-sided quote (`None`
before any). A one-sided book makes no quote; a row with neither a print nor a
two-sided book makes no event and is still journaled. The call returns the number of
events (0, 1 or 2). A `nan` or infinite `ltp`, `bid` or `ask` is refused
(`non_finite_price`) before it is journaled, and nothing changes.

- A tick must not be older than its own symbol's last tick (`invalid_parameter`).
  One older than another symbol's last push, or than the last `advance_time`, is
  accepted at its own time: its hook sees its own timestamp and it joins the bar of
  its own window.
- A bar built from ticks (`primary_bars`, and composite bars over them) closes on its
  symbol's first tick of a later window. `advance_time` never closes it, and a window
  with no tick makes no bar.
- A symbol warmed with closed bars (`warmup_bars=`, `warm_up`, `push_bar`,
  `push_group`) and then fed ticks builds its next bar from the window after the last
  bar given, and its indicators run on across the join. A tick stamped at that bar's
  close opens the next bar; an earlier one is refused.
- Bars and ticks of one symbol are fed one way at a time: bars as history up to its
  first tick, then ticks only. A bar pushed once its ticks have started (one stamped
  before its last tick is refused) is executed and heard beside the bar the engine
  builds from the same prints, and both are rows of its series (every bar twice).

## 3. API reference

| Name | What it is |
| --- | --- |
| `run_strategy_backtest(strategy, timestamps, open, high, low, close, volume, symbol=, bar_stamp=, bar_ns=, ...)` | One symbol, bars. |
| `run_strategies(strategies, data=, ticks=, depth=, primary_bars=, bar_stamp=, bar_ns=, ...)` | Many symbols or strategies; bars, ticks and depth together. |
| `raptorbt.run_tick_strategy(strategy, ticks, depth=, primary_bars=, ...)` | Ticks only; the same runtime with one strategy. `primary_bars=(step, unit)` builds bars *from the prints* as a view; orders still match on ticks. |
| `raptorbt.TickStrategyStream(strategy, symbols, ...)` | The same, fed one event at a time (`push_tick`, `push_bar`, `push_depth`). |
| `LiveNode.push_tick(symbol, ts, ltp, bid=0.0, ask=0.0, buy_qty_delta=0.0, sell_qty_delta=0.0, ltq=0.0, bid_qty=0.0, ask_qty=0.0, oi=0.0, *, arrival_ns=None)` | One tick row on a live node (see "Ticks on a live node"); returns how many events it made. The node itself is the [live-trading](../live-trading/SKILL.md) skill. |
| `raptorbt.strategy.TradeTick` | `timestamp`, `price`, `size`, `symbol`, `oi`. `size` is `ltq` when the feed has it, else the flow-delta proxy. |
| `raptorbt.strategy.QuoteTick` | `timestamp`, `bid`, `ask`, `symbol`, `bid_size`, `ask_size` (`nan` when unknown), `.mid`, `.spread`. |
| `raptorbt.bars_from_ticks(timestamps, ltp, buy_qty_delta, sell_qty_delta, step, unit)` | Ticks to bars; returns six arrays (stamps are window ends). |
| `raptorbt.run_tick_backtest(timestamps, ltp, bid, ask, buy_qty_delta, sell_qty_delta, oi, entries, exits, ...)` | Array runner over a tick tape with precomputed boolean signals. A row with `ltp` 0 is no trade: it fills nothing, triggers no stop or target, leaves equity as it was, and a signal on it is taken at the next print; a negative `ltp` is refused (`invalid_data`). |
| `raptorbt.tick_spread_pct(bid, ask)` | Spread as a percentage of mid; 0 where both sides are zero. |
| `raptorbt.buy_sell_imbalance_delta(buy_cumulative, sell_cumulative)` | Per-tick buy share in `[0, 1]` (0.5 neutral) from **cumulative** session totals. |
| `raptorbt.return_window(timestamps, ltp, window_seconds)` | Lookback return in percent; `nan` until the window has history. |
| `raptorbt.realized_vol_rolling(timestamps, ltp, window_seconds)` | Standard deviation of log returns over the window, in percent; `nan` until two points. |
| `raptorbt.tick_velocity(timestamps, window_seconds)` | Ticks per minute over the preceding window. |
| `raptorbt.oi_position_pct(oi, oi_day_high, oi_day_low)` | Open interest within the day's range, 0 to 100. |
| `raptorbt.compute_tick_entry_signals(...)`, `raptorbt.compute_tick_exit_signals(...)` | Gate precomputed features into entry and exit arrays. |

## 4. Standard workflow

1. Build arrays: `int64` nanosecond timestamps ascending (equal ones keep input
   order), `float64` prices and volume, one length per symbol.
2. Decide the stamp. Pass `bar_stamp="close"`, or `"open"` with `bar_ns`.
   Leaving it unset warns today and is an error later.
3. Hand the data to a runner under symbol names that match your specs.
4. Inside hooks read the market through `ctx.data`, or `ctx.history(n)` for a
   trailing window of closes. Do not index the arrays you passed in.
5. For ticks, give `ltq` when the feed has it, and `bid_qty` and `ask_qty` if
   you want queue-aware limit fills.
6. Precompute tick features with the functions above, gate them into signals,
   and run `run_tick_backtest`, or compute them inside a hook from
   `ctx.tick`, `ctx.best_bid` and `ctx.best_ask`.

## 5. Examples

`ctx.data` and the stamps. The same bars stamped by close and by open give the
same strategy view; only `ctx.timestamp` differs. In a bar hook `ctx.close` on the
single-symbol runner ends at the current bar (in `on_start` it holds the whole run);
`ctx.data` gives the same view with the symbol and the timeframes:

```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 30
close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n))
stamps_close = T0 + np.arange(1, n + 1, dtype=np.int64) * MIN
stamps_open = stamps_close - MIN

def bars(stamps):
    return dict(timestamps=stamps, open=close, high=close + 0.3, low=close - 0.3,
                close=close, volume=np.full(n, 1000.0))

class Spy(Strategy):
    def on_start(self, ctx):
        self.rows = []

    def on_bar(self, ctx):
        h = ctx.data.history("ASSET")
        assert len(ctx.close) == ctx.idx + 1    # ends at this bar; ctx.close[ctx.idx + 1] raises
        assert len(h) == ctx.idx + 1            # ctx.data stops at this bar
        assert h.close[-1] == ctx.bar.close == ctx.data.latest("ASSET").close
        self.rows.append((ctx.timestamp, ctx.bar_close_ts))

by_close, by_open = Spy(), Spy()
run_strategy_backtest(by_close, **bars(stamps_close), symbol="ASSET", bar_stamp="close")
run_strategy_backtest(by_open, **bars(stamps_open), symbol="ASSET",
                      bar_stamp="open", bar_ns=MIN)

assert [c for _, c in by_close.rows] == [c for _, c in by_open.rows]   # same clock
assert by_open.rows[0] == (T0, T0 + MIN)       # your stamp, and the bar's close
```

Several symbols on different clocks. Each sees the other's latest bar that has
already closed, never one that closes later:

```python
import numpy as np
from raptorbt import Strategy, run_strategies

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9

def bars(offset, base, n=6):
    close = base + np.arange(n, dtype=float)
    return dict(timestamps=T0 + offset + np.arange(1, n + 1, dtype=np.int64) * MIN,
                open=close, high=close + 0.5, low=close - 0.5, close=close,
                volume=np.full(n, 1000.0))

class Peek(Strategy):
    def on_start(self, ctx):
        self.rows = []

    def on_bar(self, ctx):
        other = "BBB" if ctx.symbol == "AAA" else "AAA"
        seen = ctx.data.latest(other)
        self.rows.append((ctx.symbol, ctx.bar_close_ts, None if seen is None else seen.timestamp))

s = Peek()
run_strategies([s], {"AAA": bars(0, 100.0), "BBB": bars(MIN // 2, 200.0)}, bar_stamp="close")
assert s.rows[0] == ("AAA", T0 + MIN, None)                  # BBB has not closed yet
for _, now, seen in s.rows:
    assert seen is None or seen <= now                       # never a later bar
```

Trades and quotes. A quote row is dispatched after the print of the same row,
and at a print `ctx.best_bid` is the book **before** that print:

```python
import numpy as np
from raptorbt import Strategy, run_tick_strategy

S = 10**9
T0 = 1_767_225_600 * S
n = 400
rng = np.random.default_rng(3)
ts = T0 + np.cumsum(rng.integers(1, 3, n)).astype(np.int64) * S
px = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
ticks = {"AAPL": dict(timestamps=ts, ltp=px, bid=px - 0.02, ask=px + 0.02,
                      ltq=np.full(n, 10.0), bid_qty=np.full(n, 100.0), ask_qty=np.full(n, 120.0))}

class Tape(Strategy):
    def on_start(self, ctx):
        self.seen = []

    def on_trade(self, ctx, tick):
        self.seen.append(("trade", tick.timestamp, tick.size, ctx.best_bid))

    def on_quote(self, ctx, quote):
        self.seen.append(("quote", quote.timestamp, quote.bid))

s = Tape()
run_tick_strategy(s, ticks)
assert [k for k, *_ in s.seen[:4]] == ["trade", "quote", "trade", "quote"]
assert s.seen[0][3] is None                    # no quote yet at the first print
assert s.seen[2][3] == s.seen[1][2]            # the previous row's bid
assert s.seen[0][2] == 10.0                    # ltq is the trade size

class Minutes(Strategy):
    def on_start(self, ctx):
        self.stamps = []

    def on_bar(self, ctx):
        self.stamps.append(ctx.bar.timestamp)

m = Minutes()
run_tick_strategy(m, ticks, primary_bars=(1, "m"))   # bars built from prints, a view only
assert all(t % (60 * S) == 0 for t in m.stamps)       # stamped at the window end
```

Tick rows pushed to a live node one at a time. A row with no book is a trade
only, a one-sided book with no print is nothing, `advance_time` closes no bar built
from ticks, and once the symbol has had bars the built bar joins `ctx.series()`:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy

S = 10**9
MIN = 60 * S


class Ear(Strategy):
    def on_start(self, ctx):
        self.heard = []

    def on_trade(self, ctx, tick):
        self.heard.append(("trade", tick.timestamp // S, ctx.best_bid))

    def on_quote(self, ctx, quote):
        self.heard.append(("quote", quote.timestamp // S, quote.bid))

    def on_bar(self, ctx, bar):
        closes = ctx.series()["close"]
        self.heard.append(("bar", bar.timestamp // S, bar.close, len(closes), closes[-1]))


ear = Ear()
node = LiveNode([ear], ["AAA"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                primary_bars=(1, "m"))
for m in (1, 2):                                          # history as closed 1-minute bars
    node.push_bar("AAA", m * MIN, 100.0, 101.0, 99.0, 100.0 + m, 10.0)
assert node.push_tick("AAA", 2 * MIN + 5 * S, 103.0) == 1             # no book: a trade only
assert node.push_tick("AAA", 2 * MIN + 9 * S, 103.5, 103.4, 103.6) == 2  # a trade, then its quote
assert node.push_tick("AAA", 2 * MIN + 20 * S, 0.0, 103.4, 0.0) == 0     # one-sided, no print: nothing
node.advance_time(4 * MIN)                                # past the window: no bar yet
assert [h[0] for h in ear.heard].count("bar") == 2
node.push_tick("AAA", 4 * MIN + 1 * S, 104.0)            # the first tick of a later window closes it
assert ear.heard[2:5] == [("trade", 125, None), ("trade", 129, None), ("quote", 129, 103.4)]
assert ear.heard[5] == ("bar", 180, 103.5, 3, 103.5)      # the built bar joins the series: 3 rows
```

Depth. A strategy that subscribes to `"book"` receives snapshots; they inform
queue position and never fill:

```python
import numpy as np
from raptorbt import Strategy, run_tick_strategy

S = 10**9
T0 = 1_767_225_600 * S
n = 50
ts = T0 + np.arange(1, n + 1, dtype=np.int64) * S
px = np.full(n, 100.0)
ticks = {"AAPL": dict(timestamps=ts, ltp=px, bid=px - 0.1, ask=px + 0.1)}
depth = {"AAPL": dict(
    timestamps=ts,
    bid_prices=np.tile([99.9, 99.8], (n, 1)), bid_sizes=np.tile([100.0, 200.0], (n, 1)),
    ask_prices=np.tile([100.1, 100.2], (n, 1)), ask_sizes=np.tile([100.0, 150.0], (n, 1)),
)}

class Books(Strategy):
    def on_init(self, ctx):
        self.subscribe("AAPL", data_type="book")

    def on_start(self, ctx):
        self.books = []

    def on_book(self, ctx, book):
        self.books.append(book)

s = Books()
run_tick_strategy(s, ticks, depth=depth)
b = s.books[0]
assert (b.best_bid, b.best_ask) == (99.9, 100.1) and b.imbalance == 0.5
```

Tick features into signals, then the array runner. Every feature looks back
only, and `nan` fails the return gate rather than passing it:

```python
import numpy as np
import raptorbt

S = 10**9
T0 = 1_767_225_600 * S
n = 2000
rng = np.random.default_rng(3)
ts = T0 + np.cumsum(rng.integers(1, 3, n)).astype(np.int64) * S
px = 100.0 + np.cumsum(rng.normal(0.01, 0.05, n))
bid, ask = px - 0.02, px + 0.02
buy, sell = rng.uniform(5, 15, n), rng.uniform(1, 10, n)      # per-tick traded quantity

spread = raptorbt.tick_spread_pct(bid, ask)
bsi = raptorbt.buy_sell_imbalance_delta(np.cumsum(buy), np.cumsum(sell))   # takes cumulative sums
ret = raptorbt.return_window(ts, px, 60.0)
entries = raptorbt.compute_tick_entry_signals(
    spread, bsi, ret, spread_pct_max=1.0, bsi_min=0.5,
    return_1m_min_abs=0.05, return_direction=1, cooldown_ticks=20,
)
exits = raptorbt.compute_tick_exit_signals(ts, eod_exit_time_ns=int(ts[-50]))

warmup = int(np.isnan(ret).sum())
assert warmup > 0 and not entries[:warmup].any()   # no entry before the window has history
result = raptorbt.run_tick_backtest(
    ts, px, bid, ask, buy, sell, np.zeros(n), entries, exits,
    symbol="AAPL", initial_capital=100_000.0, fees=0.0, quantity=10,
)
assert result.metrics.total_trades > 0
```

## 6. Composition

- Declare which trades, quotes, books and bar widths a strategy receives, and
  read a closed-bar history of a subscribed width: [subscriptions](../subscriptions/SKILL.md).
- Build coarser bars from bars or ticks, and set the grid they sit on:
  [timeframes](../timeframes/SKILL.md).
- Symbols need matching specs (lots, sessions): [instruments](../instruments/SKILL.md).
- Indicators over bars: [indicators](../indicators/SKILL.md).
- A live node, its journal and restore: [live-trading](../live-trading/SKILL.md).
- Runnable examples: `examples/tick-level/passive_quotes.py`, `examples/tick-level/tick_scalper.py`
  (a momentum scalper on prints, with its batch and live runs compared),
  `examples/tick-level/dynamic_exit.py`, `examples/multi-instrument/cross_instrument_factor.py`
  (a flow factor over several instruments' prints, read with five-minute bars built from them),
  `examples/multi-instrument/relative_value.py`.

## 7. Constraints and correctness

- **Timestamps** are `int64` nanoseconds and must not decrease; a descending
  pair raises `decreasing_timestamps` naming the index. Equal timestamps are
  allowed and keep input order. All arrays of one symbol share one length
  (`length_mismatch`).
- **Prices must be finite.** A `nan` or infinite open, high, low or close is
  refused before the run with `non_finite_price`, naming the column
  (`.parameter`) and the first index (`.value`), on the array runners, the
  strategy runners and `run_strategies`. Clean the data first (drop or fill the
  bad bars); the engine never fills them for you.
- **`float32` is coerced** by the strategy runners and refused by the array
  runners (`TypeError`). Convert once with `np.asarray(x, dtype=np.float64)`.
- **A bar hook sees up to the current bar.** On the single-symbol runner
  `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.volume` and
  `ctx.timestamps` end at `ctx.idx` inside a bar hook, so `ctx.close[-1]` is
  this bar's close and `ctx.close[ctx.idx + 1]` raises `IndexError`; in `on_start`
  they hold the whole run. On the multi-symbol runner `ctx.series(symbol)`
  is cut at the latest bar.
- **`on_start` precomputation.** The full arrays are visible in `on_start` for
  vectorised indicators. An array computed there from `close` is only safe if
  each element uses bars up to its own index, which the indicator functions
  (`raptorbt.sma`, ...) do. A `shift(-1)`, a full-sample mean or a z-score over
  the whole series leaks the future into every bar.
- **Quote timing.** A quote row follows its print; at a print the book is the
  one *before* it. Acting on `ctx.best_bid` inside `on_trade` reads a book the
  print has not yet moved.
- **Unknown size is `nan`**, never zero (`bid_size`, `ask_size`).
- **Flow deltas.** `buy_sell_imbalance_delta` takes cumulative session totals
  and differences them itself. `run_tick_backtest` and `bars_from_ticks` take
  per-tick deltas (`np.diff(cumulative, prepend=0).clip(min=0)`). Passing
  cumulative values where deltas are expected inflates every bar's flow.
- **`primary_bars` is a view.** Bars built from prints fire `on_bar` and feed
  indicators; fills still happen on ticks. A tick strategy with no
  `primary_bars` never gets `on_bar`. Once a symbol has had a given bar, the bars
  built after it are rows of `ctx.series(symbol)` too (section 2); before that its
  series is its ticks.
- **A built bar closes on a later tick.** Live, the last window of a quiet symbol
  stays open until that symbol prints again; `advance_time` fires timers and
  alerts, not bars.
- **One symbol's ticks never go back.** A tick older than its symbol's last tick is
  refused; a tick older than another symbol's last tick, or than the last
  `advance_time`, is taken at its own time (and the node's `health()["clock_ns"]`
  steps back to it).
- **Depth needs ticks** for the same symbol (`depth_without_ticks`).

## 8. Common mistakes

Bars that go backwards are refused:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

MIN = 60 * 10**9
ts = np.array([1, 3, 2, 4], dtype=np.int64) * MIN
c = np.array([100.0, 101.0, 102.0, 103.0])
run_strategy_backtest(Strategy, timestamps=ts, open=c, high=c, low=c, close=c,
                      volume=np.ones(4), bar_stamp="close")  # InvalidData
```

History handed to a strategy is read-only; copy it to modify it:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

MIN = 60 * 10**9

class Writer(Strategy):
    def on_bar(self, ctx):
        ctx.data.history("ASSET").close[0] = 0.0   # ValueError: read-only

c = np.linspace(100.0, 103.0, 4)
run_strategy_backtest(Writer, timestamps=np.arange(1, 5, dtype=np.int64) * MIN,
                      open=c, high=c, low=c, close=c, volume=np.ones(4),
                      symbol="ASSET", bar_stamp="close")
```

- Computing a full-array statistic (`ctx.close.mean()` over a run you precomputed in
  `on_start`) and using it inside `on_bar`: the statistic includes the future.
- Treating `ctx.timestamp` as the clock in a run stamped by bar open. Use
  `ctx.bar_close_ts`.
- Reading `ctx.data.quote(symbol)` without subscribing to quotes:
  `not_subscribed_data`.
- Expecting quotes or books to fill a resting order.
- Pushing a tick with only a bid (or only an ask) and waiting for `on_quote`: a
  quote needs both sides.
- Calling `advance_time` at a minute boundary to close the minute's bar on a
  tick-fed node: it closes on the symbol's next tick.
- Passing a data dict whose keys differ from the `symbol` or the specs.
- Mixing bar widths in one symbol's array and expecting aggregation to repair
  it; the engine refuses a bar off its grid in resampling and does not guess.
- Assigning `self.log = []` in a strategy: `Strategy.log` is a read-only
  logger property. Use another name.

## 9. Validation

1. Assert `np.isfinite` on every price array and `np.all(np.diff(ts) >= 0)`.
2. Run once with `bar_stamp="close"` on a copy re-stamped by open
   (`bar_stamp="open"`, `bar_ns`): trades must match exactly.
3. Poison test: replace every bar after a cut time with garbage and re-run.
   Everything the strategy saw or did up to the cut must be identical. If it
   is not, the strategy reads the future.
4. In a hook, assert `len(ctx.data.history(symbol)) == ctx.idx + 1` on the
   single-symbol runner.
5. For ticks, log the order of `on_trade` and `on_quote` for the first rows and
   confirm the print comes first.
6. Count `nan` in each feature array and confirm no signal fires inside that
   warm-up span.

## 10. Related skills

- [Subscriptions](../subscriptions/SKILL.md)
- [Timeframes](../timeframes/SKILL.md)
- [Indicators](../indicators/SKILL.md)
- [Instruments](../instruments/SKILL.md)
