# RaptorBT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/raptorbt.svg)](https://pypi.org/project/raptorbt/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-red.svg)](https://www.rust-lang.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/raptorbt?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=ORANGE&left_text=downloads)](https://pepy.tech/projects/raptorbt)

**Blazing-fast backtesting for the modern quant.**

RaptorBT is a high-performance backtesting engine written in Rust with Python bindings via PyO3. It runs single-instrument, basket, pairs, options, spread, multi-strategy, and tick-level backtests over any OHLCV or tick arrays — from any broker, market, or asset class — and returns a full performance report in sub-millisecond time.

<p align="center">
  <strong>Sub-millisecond backtests</strong> · <strong>&lt;1 MB compiled engine</strong> · <strong>Bit-for-bit deterministic</strong>
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
    timestamps=timestamps,
    open=open,
    high=high,
    low=low,
    close=close,
    volume=volume,
    entries=entries,
    exits=exits,
    direction=1,
    weight=1.0,
    symbol="AAPL",
    config=config,
)

# Results
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

RaptorBT is open source (MIT) and developed by the [Alphabench](https://alphabench.in) team.

---

## Table of Contents

- [Overview](#overview)
- [Performance](#performance)
- [Class-Based Strategies](#class-based-strategies)
- [Strategy Types](#strategy-types)
- [Metrics](#metrics)
- [Indicators](#indicators)
- [Stop-Loss & Take-Profit](#stop-loss--take-profit)
- [Monte Carlo Portfolio Simulation](#monte-carlo-portfolio-simulation)
- [API Reference](#api-reference)
- [Building from Source](#building-from-source)

---

## Overview

RaptorBT compiles to a single native extension and runs entirely in Rust, so a
full backtest with all 33 metrics finishes in well under a millisecond on
typical bar counts. Measured on an Apple M4 (raptorbt 0.4.0):

| Metric                        | RaptorBT     |
| ----------------------------- | ------------ |
| **Compiled engine size**      | <1 MB        |
| **Backtest speed (1K bars)**  | ~0.03 ms     |
| **Backtest speed (10K bars)** | ~0.25 ms     |
| **Backtest speed (50K bars)** | ~1.4 ms      |
| **Memory usage**              | Low (native) |

See [Performance](#performance) for the full method and how to reproduce these
numbers on your own hardware.

### Key Features

- **7 Strategy Types**: Single instrument, basket/collective, pairs trading, options, spreads, multi-strategy, and tick-level
- **Asset- and broker-agnostic**: Pass NumPy OHLCV or tick arrays from any source — equities, futures, FX, crypto, options — RaptorBT never assumes a market or data vendor
- **Tick-Level Simulation**: Full tick resolution for intraday options momentum, scalping, and microstructure strategies
- **Batch Spread Backtesting**: Run multiple spread backtests in parallel via Rayon with GIL released
- **Monte Carlo Simulation**: Correlated multi-asset forward projection via GBM + Cholesky decomposition
- **33 Metrics**: Sharpe, Sortino, Calmar, Omega, SQN, Payoff Ratio, Recovery Factor, and more
- **20 Indicator & Tick Functions**: 12 classic technical indicators (SMA, EMA, RSI, MACD, Stochastic, ATR, Bollinger Bands, ADX, VWAP, Supertrend, Rolling Min/Max) plus 8 tick microstructure/feature functions
- **Stop/Target Management**: Fixed, ATR-based, and trailing stops with risk-reward targets
- **Deterministic**: Identical inputs produce bit-for-bit identical results across runs — no JIT compilation variance
- **Native Parallelism**: Rayon-based parallel processing with explicit SIMD optimizations

---

## Performance

### Benchmark Results

Measured on an Apple M4 (raptorbt 0.4.0, Python 3.11) with random-walk price
data and an SMA-crossover strategy. Each figure is the fastest of several
hundred repetitions of `run_single_backtest` (so it reflects engine time, not
scheduler noise):

```
┌─────────────┬───────────┐
│ Data Size   │ RaptorBT  │
├─────────────┼───────────┤
│ 1,000 bars  │ 0.03 ms   │
│ 5,000 bars  │ 0.13 ms   │
│ 10,000 bars │ 0.25 ms   │
│ 50,000 bars │ 1.37 ms   │
└─────────────┴───────────┘
```

Timings scale roughly linearly with bar count and will vary with your CPU,
data, and signal density. Reproduce them with the [Verification Test](#verification-test)
below, swapping in your own array sizes.

### Determinism

RaptorBT is fully deterministic: the same inputs produce bit-for-bit identical
results across runs (no JIT warmup, no nondeterministic reductions). Running the
[Verification Test](#verification-test) five times in a row on this machine
produced the same total return every time, to the last decimal:

```
Total return:           -30.6192%  (seed=42, 500 bars, periodic entries/exits)
Max difference across 5 runs: 0.0000000000%
```

(The exact return depends on your data and signals — the point is that it does
not change between runs.)

---

## Class-Based Strategies

New in 0.5.0: strategies can be written as event-driven classes instead of
precomputed signal arrays. Subclass `raptorbt.Strategy`, override lifecycle
hooks, and emit order intents; the engine simulates fills and routes events
back into your hooks. Both paths share one execution core, so identical
decisions produce identical results — the class contract is the recommended
way to write new strategies, while the array runners remain the fast path
for vectorized workloads.

```python
import numpy as np
import raptorbt


class SmaCross(raptorbt.Strategy):
    def on_start(self, ctx):
        # Full OHLCV arrays are available for indicator precomputation.
        self.fast = raptorbt.sma(ctx.close, 10)
        self.slow = raptorbt.sma(ctx.close, 30)

    def on_bar(self, ctx):
        i = ctx.idx
        if i == 0 or np.isnan(self.slow[i]) or np.isnan(self.slow[i - 1]):
            return
        crossed_up = self.fast[i] > self.slow[i] and self.fast[i - 1] <= self.slow[i - 1]
        crossed_dn = self.fast[i] < self.slow[i] and self.fast[i - 1] >= self.slow[i - 1]
        if crossed_up and ctx.position is None:
            self.enter()                      # optional: size_frac=, stop_price=, target_price=
        elif crossed_dn and ctx.position is not None:
            self.close_position()

    def on_position_closed(self, ctx, event):
        self.log.info("closed: pnl=%.2f", event.trade.pnl)


result = raptorbt.run_strategy_backtest(
    SmaCross(), timestamps, open_, high, low, close, volume,
    symbol="EXAMPLE", config=raptorbt.PyBacktestConfig(fees=0.001),
)
print(result.metrics.total_return_pct, len(result.trades()))
```

Hooks: `on_start`, `on_bar`, `on_stop`, `on_order_filled`,
`on_order_rejected`, `on_position_opened`, `on_position_closed`. Inside
`on_bar`, `ctx` provides the current `bar`, `position` snapshot, `equity`,
`cash`, `history(n)`, and `set_stop_price()` / `set_target_price()` for
programmatic exits. Decision logic must only read array values at `ctx.idx`
or earlier — indexing past the current bar reads the future.

Engine-level stop/target/sizing configuration (`PyBacktestConfig`,
`PyInstrumentConfig`) applies to both paths. `run_strategy_backtest` returns
the same `PyBacktestResult` as `run_single_backtest`. For advanced drivers
(live feeds, custom loops), `PyKernelSession` exposes the per-bar engine
step directly.

Note: one Python hook call per bar makes the class path slower than the
array path — fine for typical bar counts, but prefer arrays for large
parameter sweeps.

### Instrument Definitions

New in 0.5.0: `InstrumentSpec` describes the market being traded — tick
size, lot size, contract multiplier, expiry — separately from the per-run
allocation knobs in `PyInstrumentConfig`. Attach one to a class-based run
via `run_strategy_backtest(..., instrument=...)` (or directly on
`PyKernelSession`):

```python
import raptorbt

# NIFTY monthly future: 50-unit lots, expiry settlement at the contract's
# expiration timestamp, entries refused before activation / after expiry.
fut = raptorbt.InstrumentSpec.futures_contract(
    "NIFTY24AUGFUT",
    expiration_ns=1724839200_000_000_000,
    lot_size=50.0,
    price_increment=0.05,
    underlying="NIFTY",
)

result = raptorbt.run_strategy_backtest(
    MyStrategy, ts, o, h, l, c, v, instrument=fut,
)
```

Constructors: `equity`, `futures_contract`, `perpetual`, `option` (vanilla
and binary; settles to intrinsic value when an underlying price is known),
`currency_pair`, and `index` (non-tradable reference). With a spec attached
the engine:

- scales notional by the contract `multiplier` — sizing, cash, PnL, and
  value-based fees charge on `price * size * multiplier`, while
  per-share/per-contract fee models keep charging per contract;
- floors sizes to `lot_size` / `size_increment` (an explicit
  `PyInstrumentConfig.lot_size` still wins — it is the per-run override);
- rounds engine-derived stop/target prices onto the `price_increment` grid,
  conservatively (never in the strategy's favor);
- force-settles open positions at expiry (`Settlement` exit reason) and
  rejects entries outside the activation/expiration window.

Without a spec, behavior is unchanged — existing results reproduce
bit-for-bit. `margin_init`/`margin_maint`/`maker_fee`/`taker_fee` are
carried on the spec for the account layer that consumes them in a later
0.5.x release.

### Typed Orders

New in 0.5.0: alongside the `enter()`/`close_position()` sugar, strategies
can submit typed orders that rest across bars and report a full lifecycle:

```python
from raptorbt.strategy import orders

class Breakout(raptorbt.Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 20 and ctx.position is None:
            # Buy stop above the market, protective stop attached.
            self.oid = self.submit_order(orders.StopMarket(
                side="buy",
                trigger=float(ctx.high[:20].max()),
                size_frac=0.5,
                stop_price=float(ctx.close[ctx.idx] * 0.97),
                tif="day",
            ))
        if ctx.position is not None:
            self.submit_order(orders.Limit(side="sell", price=ctx.position.entry_price * 1.1))
```

- **Kinds**: `orders.Market`, `orders.Limit` (with `post_only=`),
  `orders.StopMarket`, `orders.StopLimit` (trigger fires, then rests as a
  limit from the next bar), `orders.MarketIfTouched` / `orders.LimitIfTouched`
  (favorable-touch triggers — a buy fires when price *falls* to the
  trigger), `orders.MarketToLimit` (fills at the next bar's open), and
  `orders.TrailingStopMarket` / `orders.TrailingStopLimit` (trigger trails
  the running favorable extreme; `offset_kind` is `"price"`, `"bps"`, or
  `"ticks"` — ticks need an instrument `price_increment`).
- **Time-in-force**: `gtc` (default), `day` (UTC-date rollover), `gtd`
  (with `expire_ns`), `ioc`, `fok`, plus `at_open` / `at_close` for market
  orders queued to a bar phase.
- **Flags**: `post_only` (limit rejects if marketable at its first resting
  open), `reduce_only` (a fill may never increase exposure).
- **Brackets**: `self.submit_bracket(entry, stop_trigger=…, target_price=…,
  stop_limit_price=None)` — the protective legs are held until the entry
  fills (one-triggers-other), then linked one-cancels-other: the first leg
  to fill cancels its sibling, and both die if the entry never fills.
  Generic linkage: `submit_order(order, parent=other_id)` and
  `self.link_oco(id_a, id_b, …)`. One-updates-other reduces to
  one-cancels-other while fills are all-or-nothing (partial fills arrive
  with book depth). Netting policy only — under hedging every order opens,
  so protect positions with per-position `stop_price`/`target_price`
  attachments instead.
- **Sizing**: `units=` (explicit contracts; refused if it exceeds available
  capital) or `size_frac=` (fraction of capital, resolved at fill time);
  omit both on a closing-side order to close the full position.
- **Semantics**: market orders fill on the submission bar at the configured
  fill-price model — the same contract as `enter()`. Resting orders begin
  matching on the *next* bar (an order cannot rest into a bar that had
  already closed), with gap-throughs filling at the open.
- **Lifecycle hooks**: `on_order_accepted`, `on_order_triggered`,
  `on_order_filled`, `on_order_canceled`, `on_order_expired`,
  `on_order_rejected`, plus catch-all `on_order_event`. Events carry
  `client_order_id` (deterministic `"{order_id_tag}-{seq}"`).
- **Management**: `self.cancel_order(client_id)`,
  `self.cancel_all_orders()`, `self.modify_order(client_id, limit_price=…,
  trigger_price=…, units=…)`.
- Order-driven exits report `exit_reason == "Order"` on the trade record.
  One position at a time: an opening order while a position is open rejects
  with `"position_open"` (independent concurrent positions arrive in a
  later 0.5.x release).

The signal-array runners do not interact with the order book and are
unaffected.

### Bar Aggregation and Multi-Timeframe Strategies

New in 0.5.0. Streaming and batch aggregation of bars (and raw ticks) into
coarser bars — time (`"ms"`/`"s"`/`"m"`/`"h"`/`"d"`/`"w"`), `"tick"`,
`"volume"`, and `"value"` units. Time bars use left-open epoch-aligned
windows and are stamped with the window-*end* timestamp, so a bar labeled
`t` contains only data strictly before `t` — no look-ahead by construction.
Beyond time, tick, volume and value windows, two families sample on
something other than the clock:

**Renko** (`"renko"`) emits a brick per full brick-height price move and
ignores time and volume entirely — a quiet hour produces nothing, a fast
move produces several bricks at once. Set the height with `brick_size`;
without it, `step` reads as whole price units. Because one record can
complete several bricks, `push` returns only the first and the rest must be
drained:

```python
agg = raptorbt.BarAggregator(1, "renko", brick_size=0.05)
bar = agg.push_trade(ts, price, size)
while bar is not None:
    handle(bar)
    bar = agg.next_pending()      # drain, or bricks are silently lost
```

Bricks carry no wicks, and a partial brick is discarded at end of data
rather than flushed — an incomplete brick is not a brick.

**Signed-flow bars** (`"{tick,volume,value}_imbalance"` and
`"{tick,volume,value}_runs"`) sample by order-flow direction. *Imbalance*
closes on net signed flow, so balanced two-sided trading never closes a bar
however heavy it is; *runs* closes on the larger one-sided accumulation, so
the same tape does close bars. `step` is the threshold — fixed, rather than
the adaptive estimate in the literature, so runs stay reproducible.

Direction comes from the buy/sell quantity deltas when you supply them
(`bars_from_ticks`), and otherwise from the tick rule, which is what lets
these units work over plain OHLC bars.

```python
# Batch: 1-minute bars -> 5-minute bars (or ticks -> bars).
ts5, o5, h5, l5, c5, v5 = raptorbt.aggregate_bars(ts, o, h, l, c, v, 5, "m")
bts, bo, bh, bl, bc, bv = raptorbt.bars_from_ticks(ts, ltp, buys, sells, 1000, "volume")
# Signed flow: close a bar every 10,000 shares of net buying or selling.
its = raptorbt.bars_from_ticks(ts, ltp, buys, sells, 10_000, "volume_imbalance")

# In a strategy: a 5-minute trend filter gating 1-minute entries.
class TrendGated(raptorbt.Strategy):
    def on_start(self, ctx):
        self.h5 = self.subscribe_bars(5, "m")
        self.trend_up = False

    def on_composite_bar(self, ctx, bar):   # fires when a 5m bar completes
        self.trend_up = bar.close > bar.open

    def on_bar(self, ctx):                  # every 1m bar
        if self.trend_up and ctx.position is None:
            self.enter()
```

`on_composite_bar` dispatches *before* the `on_bar` of the primary bar that
completed the window — the composite closed strictly earlier. A partial
final window is not dispatched to strategies (it never closed); the batch
helpers do include it, flushed at end of data.

In a portfolio run one `subscribe_bars` declaration yields one aggregated
stream **per symbol**, each built only from that symbol's bars. The symbol
that completed a bar arrives as `bar.symbol` (and `ctx.symbol`); the
dispatch-before-`on_bar` guarantee holds per symbol, while ordering across
symbols follows the merged schedule.

Calendar `"month"`/`"year"` units aggregate on civil UTC dates. Passing
`tz_offset_ns` (e.g. `raptorbt.IST_OFFSET_NS`) aligns day/week/month/year
windows to that timezone's civil dates — an NSE day bar covers one IST
trading date (a 23:30 IST print stays on its trading date instead of
rolling into the next UTC day).

### Streaming Indicators, Clock, and Cache

New in 0.5.0:

- **Streaming indicators** — `raptorbt.Indicator.sma(14)`, `.ema`,
  `.wilder_ma`, `.wma`, `.roc`, `.stddev`, `.rsi`, `.atr`, `.donchian`
  (value `(upper, lower)`), `.bollinger` (`(middle, upper, lower)`),
  `.macd` (`(macd, signal, histogram)`). Rust incremental cores, O(1)-ish
  per bar, producing values identical to the batch array functions
  (equivalence-tested). Register for auto-update:

  ```python
  class Cross(raptorbt.Strategy):
      def on_start(self, ctx):
          self.fast = self.register_indicator(raptorbt.Indicator.ema(10))
          self.slow = self.register_indicator(raptorbt.Indicator.ema(30))
          # Or feed a subscribed higher timeframe instead:
          h5 = self.subscribe_bars(5, "m")
          self.trend = self.register_indicator(raptorbt.Indicator.sma(20), stream_id=h5)

      def on_bar(self, ctx):
          if not self.indicators_initialized():
              return
          if self.fast.value > self.slow.value and ctx.position is None:
              self.enter()
  ```

  In portfolio runs an indicator tracks one symbol, so register one per
  symbol with `symbol=` — an unrouted registration is fed every symbol's
  bars interleaved (and warns):

  ```python
  class Cross(raptorbt.Strategy):
      def on_start(self, ctx):
          self.fast = self.register_indicators(
              lambda: raptorbt.Indicator.ema(10), ctx.symbols
          )
          # Equivalently, explicit per symbol:
          self.slow = {
              s: self.register_indicator(raptorbt.Indicator.ema(30), symbol=s)
              for s in ctx.symbols
          }

      def on_bar(self, ctx):
          fast, slow = self.fast[ctx.symbol], self.slow[ctx.symbol]
          if self.indicators_initialized() and fast.value > slow.value:
              self.enter()
  ```

  Registered indicators update *before* handlers see the bar.
- **Clock** — `self.clock.set_time_alert(name, at_ns)` (one-shot) and
  `set_timer(name, interval_ns, start_ns=None, stop_ns=None)` (recurring;
  one firing per bar, gaps collapse); due events reach `on_time_event`
  *before* the bar's data handlers. Bar-granular by design: events carry
  `ts_scheduled` and `ts_fired`.
- **Cache** — `self.cache`, an event-sourced mirror (no per-query engine
  calls): `order(client_id)` / `orders_open()` / `is_order_open()`,
  `closed_trades()`, `realized_pnl(symbol=None)`.
- **Portfolio view** — `ctx.net_position` / `is_net_long` / `is_net_short`
  / `is_flat` (signed across hedged positions) — properties on every
  context; per-symbol lookups via `position_for(symbol)` /
  `positions(symbol)` / `net_position_for(symbol)` on the portfolio
  context.

### Multi-Instrument Strategies

New in 0.5.0: one class-based strategy trading N instruments against a
single shared cash pool. Bars from all instruments merge into one
deterministic schedule (by timestamp, then registration order); `on_bar`
fires once per event with `ctx.symbol` naming the instrument whose bar
closed. Capital committed to one symbol is unavailable to the rest.

```python
class Rotation(raptorbt.Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.enter(size_frac=0.4)          # routes to ctx.symbol
        if ctx.symbol == "INFY" and ctx.idx == 50:
            # Orders and closes can route across symbols explicitly.
            self.submit_order(orders.Limit(side="buy", price=2400.0,
                                           units=10.0), symbol="TCS")
            if ctx.position_for("RELIANCE") is not None:
                self.close_position(symbol="RELIANCE")

result = raptorbt.run_portfolio_strategy(
    Rotation,
    data={sym: dict(timestamps=..., open=..., high=..., low=..., close=...,
                    volume=...) for sym in symbols},
    instruments={...},        # optional per-symbol InstrumentSpec
    oms_type="netting",       # or "hedging", per instrument
    account_type="cash",      # or "margin", shared across all instruments
    leverage=1.0,             # portfolio-wide under account_type="margin"
)
result.result.equity_curve()  # portfolio curve, sampled per merged event
result.per_instrument         # per-symbol trades / pnl / rejections
```

`ctx` in portfolio runs is a `PortfolioContext`: `ctx.bar` / `ctx.symbol` /
`ctx.idx` (local to the symbol), `ctx.series(symbol)` for full arrays,
`ctx.position` / `ctx.is_flat` (properties for the current symbol, same as the
single-instrument context), `ctx.position_for(symbol)` /
`ctx.positions(symbol)`, and portfolio-level
`ctx.equity` / `ctx.cash`. Composite-bar subscriptions are single-instrument
for now, and one-cancels-other links cannot span symbols.

Risk limits on the config are portfolio-wide, matching
`run_portfolio_backtest`: `max_positions` counts open positions across all
symbols (including entries from resting orders), and `max_drawdown_pct`
trips on portfolio equity and halts entries on every symbol. Capital
*allocation* is the strategy's own — each entry is offered the full free
balance, so size it with `size_frac`; there is no `EqualWeight` budget on
this path yet.

With `account_type="margin"` the instruments share one account: `leverage`
applies portfolio-wide, sizing draws on the portfolio's free capital
(balance less every instrument's locked margin), and equity marks the
balance plus direction-aware unrealized PnL so winning shorts price upward.
The maintenance requirement is the **sum of each instrument's own**
requirement, so per-symbol `margin_maint` rates apply rather than one
blended rate. A breach fires `on_margin_call` once and halts new entries on
*every* instrument — subsequent entries are rejected with `MarginCall`,
including on symbols that never traded. Results carry `halted` and
`halted_at`; in portfolio runs `halted_at` is a **schedule-event ordinal**
(the session interleaves N streams), not the bar index the array runners
report. `result.rejected_entries` is the sum across instruments, whereas the
array portfolio runner reports its single shared risk gate's counter.

### Execution Realism Knobs

Four opt-in settings, all off by default:

```python
config.limit_slippage = 0.0005          # adverse adjustment on limit fills
config.liquidate_on_margin_call = True  # broker closes you out, vs only halting
spec.settlement_fee = 0.001             # charged on the settled notional at expiry
```

`limit_slippage` models adverse selection on a resting order — you tend to
be filled when the market is about to move through you. It is suppressed
when `queue_fill_model` granted the fill, since volume observed trading
ahead of your order is evidence you genuinely held that price.

`liquidate_on_margin_call` turns a margin call from a latching halt into a
forced close. Unlike expiry settlement or end-of-data finalization, which
close free, a liquidation prices through the fill model and pays exit
costs, reporting `exit_reason == "Liquidation"`.

Options settle at their own last close unless you supply an underlying —
an option's bars carry the option's price, so intrinsic value has to come
from somewhere else:

```python
class Hold(raptorbt.Strategy):
    def on_bar(self, ctx):
        # From whatever index series the strategy already tracks.
        ctx.set_underlying_price(spot_close[ctx.idx])
```

Without it a 100-strike call whose own price decayed to 0.50 settles at
0.50, even with spot at 112 — where intrinsic is 12.00.

### Execution Algorithms

`orders.Twap` slices an order into equal parts released at a fixed
interval, each an ordinary order with its own fill:

```python
class Accumulate(raptorbt.Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Twap(
                side="buy", units=1_000, slices=10,
                every=60_000_000_000,        # one slice per minute
            ))

    def on_order_filled(self, ctx, event):
        # client ids are "<parent>#0", "<parent>#1", ...
        ...
```

The interval is a duration rather than a bar count, because a bar index
means different things in bar and tick sessions — "every 1 bar" would
silently become "every 1 print" on a tick feed. Use `every_bars=N,
bar_ns=...` if you prefer to think in bars.

A schedule is not an order: cancelling it (`cancel_twap`) stops the
remaining slices but does not unwind the ones that already traded, and
`on_algo_completed` means fully *released*, not fully filled. Only explicit
`units` can be sliced — `size_frac` resolves against equity at fill time,
so each slice would size against a different account.

### Position Policies, Margin Accounts, and Fill Realism

New in 0.5.0, all default-off (defaults reproduce prior results bit-for-bit;
a committed golden-fixture suite enforces this):

- **Hedging** — `run_strategy_backtest(..., oms_type="hedging")`: every
  typed order opens an independent position in its own direction (buy →
  long, sell → short), so longs and shorts coexist, each with its own
  protective stop/target and trailing state. Inspect them via
  `ctx.positions` (each has a `position_id`) and close one with
  `self.close_position(position_id)`. The default `"netting"` keeps the
  one-position-at-a-time behavior.
- **Margin accounts** — `account_type="margin", leverage=N`: entries lock
  initial margin (the instrument's `margin_init`, else `1/leverage`)
  instead of full notional, equity marks balance plus direction-aware
  unrealized PnL (shorts price correctly), and an equity breach of the
  maintenance requirement (`margin_maint`, else half initial) fires
  `on_margin_call` and halts new entries — no forced liquidation.
  `ctx.free_capital` reports unlocked cash.
- **Stochastic fills** — `PyBacktestConfig(fill_prob_limit=0.9,
  fill_prob_slippage=0.1, fill_seed=42)`: a marketable resting limit may be
  passed over (it stays working and retries), and stop/market fills may
  slip one tick against the trader (needs an instrument `price_increment`).
  Seeded and fully deterministic: same seed, same fills.
- **Adaptive bar path** — `PyBacktestConfig(bar_path_adaptive=True)`: when
  a stop and target are both touched inside one bar, infer the traversal
  from candle geometry (up-candle: open→low→high→close) instead of the
  conservative stop-first default.

## Strategy Types

All strategy entrypoints take NumPy arrays directly. Signals (`entries` / `exits`)
are boolean arrays you compute however you like — pandas, the built-in
[indicators](#indicators), or your own model. The engine is asset- and
broker-agnostic: timestamps are `int64` (nanoseconds for tick data; any
monotonic int for bars), prices are `float64`.

### 1. Single Instrument

Long or short on one instrument. This is the canonical example — the other
strategy types follow the same shape.

```python
import numpy as np
import pandas as pd
import raptorbt

df = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Signals (SMA crossover) — any boolean arrays work here
sma_fast = df["close"].rolling(10).mean()
sma_slow = df["close"].rolling(20).mean()
entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

config = raptorbt.PyBacktestConfig(initial_capital=100000, fees=0.001, slippage=0.0005)
config.set_fixed_stop(0.02)    # optional 2% stop-loss
config.set_fixed_target(0.04)  # optional 4% take-profit

result = raptorbt.run_single_backtest(
    timestamps=df.index.astype("int64").values,
    open=df["open"].values,
    high=df["high"].values,
    low=df["low"].values,
    close=df["close"].values,
    volume=df["volume"].values,
    entries=entries.values,
    exits=exits.values,
    direction=1,   # 1 = long, -1 = short
    weight=1.0,
    symbol="AAPL",
    config=config,
    instrument_config=raptorbt.PyInstrumentConfig(lot_size=1.0),  # optional: lot rounding, capital cap
)

print(f"Return {result.metrics.total_return_pct:.2f}%  "
      f"Sharpe {result.metrics.sharpe_ratio:.2f}  "
      f"MaxDD {result.metrics.max_drawdown_pct:.2f}%  "
      f"Trades {result.metrics.total_trades}")

equity = result.equity_curve()  # np.ndarray
trades = result.trades()        # list[PyTrade]
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
    leg1_open=long_open,
    leg1_high=long_high,
    leg1_low=long_low,
    leg1_close=long_close,
    leg1_volume=long_volume,
    # Short leg
    leg2_timestamps=timestamps,
    leg2_open=short_open,
    leg2_high=short_high,
    leg2_low=short_low,
    leg2_close=short_close,
    leg2_volume=short_volume,
    # Signals
    entries=entries,
    exits=exits,
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
    open=underlying_open,
    high=underlying_high,
    low=underlying_low,
    close=underlying_close,
    volume=volume,
    option_prices=option_prices,  # Option premium series
    entries=entries,
    exits=exits,
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
    open=open_prices,
    high=high_prices,
    low=low_prices,
    close=close_prices,
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

### 6. Batch Spread Backtest

Run multiple spread backtests in parallel. Shared data (timestamps, underlying close) is converted once, then each item is backtested on its own Rayon thread with the GIL released for maximum throughput.

```python
import numpy as np
import raptorbt

config = raptorbt.PyBacktestConfig(initial_capital=100000, fees=0.001)

# Create batch items — one per strategy variation
items = [
    raptorbt.PyBatchSpreadItem(
        strategy_id="straddle_24000",
        legs_premiums=[call_24000_premiums, put_24000_premiums],
        leg_configs=[("CE", 24000.0, -1, 50), ("PE", 24000.0, -1, 50)],
        entries=entries,
        exits=exits,
        spread_type="straddle",
        max_loss=5000.0,
        target_profit=3000.0,
    ),
    raptorbt.PyBatchSpreadItem(
        strategy_id="strangle_23500_24500",
        legs_premiums=[call_24500_premiums, put_23500_premiums],
        leg_configs=[("CE", 24500.0, -1, 50), ("PE", 23500.0, -1, 50)],
        entries=entries,
        exits=exits,
        spread_type="strangle",
    ),
]

# Run all in parallel — returns list of (strategy_id, result) tuples
results = raptorbt.batch_spread_backtest(
    timestamps=timestamps,
    underlying_close=underlying_close,
    items=items,
    config=config,
)

for strategy_id, result in results:
    print(f"{strategy_id}: {result.metrics.total_return_pct:.2f}%")
```

### 7. Tick-Level Backtest

Simulate intraday strategies at full tick resolution — no bar resampling, no intra-bar path approximation. Designed for options momentum, scalping, and any setup where the exact fill tick matters.

```python
import numpy as np
import raptorbt

# Raw tick arrays (one element per tick, same length N)
# buy_qty_delta / sell_qty_delta must be per-tick deltas, NOT Zerodha cumulative sums
result = raptorbt.run_tick_backtest(
    timestamps=timestamps_ns,       # int64 nanoseconds-since-epoch
    ltp=ltp_arr,                    # last traded price
    bid=bid_arr,
    ask=ask_arr,
    buy_qty_delta=buy_delta,        # pre-converted from cumulative: np.diff(buy_cum).clip(0)
    sell_qty_delta=sell_delta,
    oi=oi_arr,
    entries=entry_signals,          # bool array — True where entry is allowed
    exits=exit_signals,             # bool array — True where position should exit
    symbol="NIFTY26APR24600PE",
    initial_capital=100_000.0,
    fees=0.001,
    slippage=0.0005,
    stop_loss_pct=5.0,
    take_profit_pct=10.0,
    max_hold_seconds=1800,          # 30-minute maximum hold
    entry_cooldown_ticks=10,        # minimum ticks between entries
    max_trades=50,
)

print(f"trades: {result.metrics.total_trades}")
print(f"profit_factor: {result.metrics.profit_factor:.2f}")
print(f"win_rate: {result.metrics.win_rate_pct:.1f}%")
```

#### Class-Contract Tick Strategies

`run_tick_backtest` above is the array runner: precomputed signal arrays, one
long-only position at a time. For the full class contract — typed orders,
multiple positions, margin accounts, portfolio risk gates — drive the event
session from ticks instead:

```python
class Scalper(raptorbt.Strategy):
    def on_quote(self, ctx, quote):
        # Quotes are observation only: nothing fills here.
        self.wide = quote.spread > 0.05

    def on_trade_tick(self, ctx, tick):
        # ctx.best_bid / ctx.best_ask are the book observed BEFORE this
        # print — the quote from the same feed row arrives next.
        if not self.wide and ctx.is_flat and ctx.best_bid is not None:
            self.submit_order(orders.Limit(side="buy", price=ctx.best_bid, units=10))

    def on_bar(self, ctx):
        # Only fires when primary_bars is set. A view, not a venue.
        ...

result = raptorbt.run_tick_strategy(
    Scalper,
    ticks={"NIFTY24600PE": dict(timestamps=ts, ltp=ltp, bid=bid, ask=ask)},
    primary_bars=(1, "m"),     # aggregate prints into 1m bars for on_bar
    account_type="margin", leverage=5.0,
)
```

Three semantics to know:

- **Quotes do not fill orders**, move trailing stops, or mark equity.
  Filling against a quote asserts a counterparty the engine has no evidence
  for; the print that follows is that evidence. Orders submitted from
  `on_quote` rest and match on the next print. Quotes also do not sample the
  equity curve, so metrics do not shift with how chatty the feed is.
- **`primary_bars` builds bars from prints as a view.** They fire `on_bar`
  and feed indicators registered without a `stream_id`; `subscribe_bars`
  composites work too. Order matching still happens against ticks only.
- **`AT_OPEN`/`AT_CLOSE` market orders keep resting** on a print — a print
  has no bar phase to queue against. Trailing stops ratchet off every print,
  so they resolve at tick resolution; a tick run and a bar run over the same
  data legitimately differ there, since a bar can trigger a stop against a
  low that preceded the high which set the watermark.

#### Order Book and Queue-Position Fills

Pass `depth=` to `run_tick_strategy` for five-level book snapshots, which
arrive via `on_order_book` and persist on `ctx.book`:

```python
depth = {"NIFTY24600PE": dict(
    timestamps=ts,                      # int64 ns, one row per snapshot
    bid_prices=bp, bid_sizes=bs,        # (n_snapshots, levels), best first
    ask_prices=ap, ask_sizes=asz,
)}

class Maker(raptorbt.Strategy):
    def on_order_book(self, ctx, book):
        if book.imbalance and book.imbalance > 0.7:      # bid-heavy
            self.submit_order(orders.Limit(side="buy", price=book.best_bid, units=10))

config.queue_fill_model = True          # opt-in
result = raptorbt.run_tick_strategy(Maker, ticks, config=config, depth=depth)
```

`queue_fill_model` replaces `fill_prob_limit`'s coin flip with the tape. The
size queued ahead is estimated once when the order rests, then consumed by
print volume at that price; a print *through* the level fills
unconditionally. Progress is monotone, so an order passed over repeatedly
genuinely advances — the probability model has no such memory.

It does not claim a real queue rank. Market-by-price data cannot tell you
where you stand in line, nor separate size that executed ahead of you from
size that was cancelled, so the model falls back to `fill_prob_limit` rather
than guessing: on bar events (a bar's volume is not volume *at* the limit
price) and on a quote-only book (a quote gives the price, not the size). A
level outside the visible five reads as unknown, never as empty.

Books, like quotes, are observation only — they never fill an order, move a
trailing stop, or mark equity. Displayed size is intent, not a trade.

#### Tick Signal & Feature Helpers

Precompute entry/exit signal arrays and tick microstructure features before calling `run_tick_backtest`:

```python
# Signal arrays
entries = raptorbt.compute_tick_entry_signals(
    spread_pct=raptorbt.tick_spread_pct(bid, ask),
    bsi_delta=raptorbt.buy_sell_imbalance_delta(buy_cum, sell_cum),  # pass raw cumulative
    return_1m=raptorbt.return_window(timestamps_ns, ltp, window_seconds=60.0),
    spread_pct_max=3.0,
    bsi_min=0.55,           # minimum buy-side delta fraction
    return_1m_min_abs=0.3,  # minimum 1-min return % (abs)
    return_direction=1,     # +1 long, -1 short
    cooldown_ticks=10,
)
exits = raptorbt.compute_tick_exit_signals(
    timestamps_ns=timestamps_ns,
    eod_exit_time_ns=eod_ns,   # force exit at/after this timestamp; 0 = disabled
)

# Feature arrays (all return Vec<f64> of same length as input)
spread   = raptorbt.tick_spread_pct(bid, ask)               # (ask-bid)/mid * 100
bsi      = raptorbt.buy_sell_imbalance_delta(buy_cum, sell_cum)  # delta BSI per tick
ret_1m   = raptorbt.return_window(ts_ns, ltp, 60.0)         # 1-min lookback return %
vol      = raptorbt.realized_vol_rolling(ts_ns, ltp, 300.0)  # 5-min realized vol %
oi_pos   = raptorbt.oi_position_pct(oi, oi_day_high, oi_day_low)  # [0, 100]
velocity = raptorbt.tick_velocity(ts_ns, 60.0)              # ticks/min over last 60s
```

**Important for Zerodha data:** `total_buy_qty` and `total_sell_qty` from KiteTicker are cumulative session running sums, not per-tick values. Pass them as-is to `buy_sell_imbalance_delta` (it computes deltas internally). For `run_tick_backtest`, convert first: `buy_delta = np.diff(buy_cum, prepend=0).clip(min=0)`.

---

## Metrics

Every backtest returns a `PyBacktestMetrics` object exposing **33 metric fields**
(listed in full under [PyBacktestMetrics](#pybacktestmetrics)). `metrics.to_dict()`
returns a subset of 24 of them under human-readable labels (e.g. `"Sharpe Ratio"`,
`"Total Return [%]"`) for quick display; read fields directly off the object to
access all 33. The most useful are grouped below.

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

RaptorBT exports **12 classic technical indicators**, computed in native Rust
and operating on (and returning) NumPy arrays:

```python
import raptorbt

# Trend indicators
sma = raptorbt.sma(close, period=20)
ema = raptorbt.ema(close, period=20)
supertrend, direction = raptorbt.supertrend(high, low, close, period=10, multiplier=3.0)

# Momentum indicators
rsi = raptorbt.rsi(close, period=14)
macd_line, signal_line, histogram = raptorbt.macd(close, 12, 26, 9)  # fast, slow, signal (positional)
stoch_k, stoch_d = raptorbt.stochastic(high, low, close, k_period=14, d_period=3)

# Volatility indicators
atr = raptorbt.atr(high, low, close, period=14)
upper, middle, lower = raptorbt.bollinger_bands(close, period=20, std_dev=2.0)

# Strength indicators
adx = raptorbt.adx(high, low, close, period=14)

# Volume indicators
vwap = raptorbt.vwap(high, low, close, volume)

# Rolling indicators (LLV / HHV)
rolling_low = raptorbt.rolling_min(low, period=20)    # Lowest Low Value
rolling_high = raptorbt.rolling_max(high, period=20)  # Highest High Value
```

In addition, **8 tick microstructure / feature functions** are available for
tick-level work (`tick_spread_pct`, `buy_sell_imbalance_delta`, `return_window`,
`realized_vol_rolling`, `oi_position_pct`, `tick_velocity`,
`compute_tick_entry_signals`, `compute_tick_exit_signals`) — see
[Tick-Level Backtest](#7-tick-level-backtest).

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

### PyBatchSpreadItem

```python
item = raptorbt.PyBatchSpreadItem(
    strategy_id: str,                    # Unique identifier for this backtest
    legs_premiums: List[np.ndarray],     # Premium series per leg
    leg_configs: List[Tuple[str, float, int, int]],  # (option_type, strike, quantity, lot_size)
    entries: np.ndarray,                 # bool entry signals
    exits: np.ndarray,                   # bool exit signals
    spread_type: str = "custom",         # Spread type string
    max_loss: float = None,              # Optional max loss exit
    target_profit: float = None,         # Optional target profit exit
)
```

### batch_spread_backtest

```python
results = raptorbt.batch_spread_backtest(
    timestamps: np.ndarray,              # int64 nanosecond timestamps (shared)
    underlying_close: np.ndarray,        # Underlying close prices (shared)
    items: List[PyBatchSpreadItem],      # List of spread backtest items
    config: PyBacktestConfig = None,     # Optional shared config
) -> List[Tuple[str, PyBacktestResult]]  # (strategy_id, result) pairs
```

Runs all spread backtests in parallel via Rayon. Timestamps and underlying close are shared across all items and converted once. The GIL is released during execution for maximum Python concurrency.

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

33 read-only fields — see the [Metrics](#metrics) section for the full table with
descriptions. `metrics.to_dict()` returns 24 of them under human-readable labels
(e.g. `"Sharpe Ratio"`) for quick display; read fields off the object directly
for the complete set.

```python
m = result.metrics
m.total_return_pct, m.sharpe_ratio, m.max_drawdown_pct   # etc. — 33 fields total
stats = m.to_dict()
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
    print(trade.exit_reason)  # "Signal", "StopLoss", "TakeProfit", "TrailingStop", "EndOfData", "Settlement", "TimeExit"
```

---

## Building from Source

Most users should `pip install raptorbt`. To build the engine yourself you need
Rust 1.70+, Python 3.10+, and `maturin`:

```bash
cd raptorbt
maturin develop --release   # editable install into the active venv
cargo test                  # run the Rust test suite
```

### Verification Test

A seeded smoke test — run it twice and the result is identical to the last
decimal (the determinism guarantee):

```python
import numpy as np
import raptorbt

np.random.seed(42)
n = 500
close = np.cumprod(1 + np.random.randn(n) * 0.02) * 100
entries = np.zeros(n, dtype=bool); entries[::20] = True
exits = np.zeros(n, dtype=bool);  exits[10::20] = True

config = raptorbt.PyBacktestConfig(initial_capital=100000, fees=0.001)
result = raptorbt.run_single_backtest(
    timestamps=np.arange(n, dtype=np.int64),
    open=close,
    high=close,
    low=close,
    close=close,
    volume=np.ones(n),
    entries=entries,
    exits=exits,
    direction=1,
    weight=1.0,
    symbol="TEST",
    config=config,
)
print(f"Total Return: {result.metrics.total_return_pct:.4f}%")  # -30.6192%
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.4f}")       # -0.9086
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Changelog

### v0.4.0

**Tick-level backtesting — full tick resolution, no bar resampling.**

- Add `TickData` struct — parallel arrays of `timestamps`, `ltp`, `bid`, `ask`, `buy_qty_delta`, `sell_qty_delta`, `oi` (one element per tick). Callers must pre-convert Zerodha cumulative session totals to per-tick deltas before passing.
- Add `ExitReason::TimeExit` — max hold-time exceeded exit for tick strategies.
- Add `run_tick_backtest` — tick-native simulation engine. Entry fills at ask+slippage; stop/target checked against ltp on every tick (not OHLC approximation); max-hold-seconds time exit; configurable cooldown between entries. Returns the same `PyBacktestResult` / `PyBacktestMetrics` (33 fields) as all other strategy types.
- Add `compute_tick_entry_signals` — compute momentum entry bool array from precomputed feature arrays (spread gate, delta BSI gate, 1-min return gate, cooldown enforcement). O(N) single pass.
- Add `compute_tick_exit_signals` — time-based (EOD) exit bool array from tick timestamps.
- Add `tick_spread_pct` — per-tick bid/ask spread as percentage of mid price.
- Add `buy_sell_imbalance_delta` — per-tick delta BSI from Zerodha cumulative running sums. Fixes the raw-cumulative BSI artefact (~0.95 all day regardless of order flow).
- Add `return_window` — per-tick lookback return over a configurable time window using binary search (O(N log N)). Returns NaN where history is insufficient — correctly gates the entry filter rather than silently passing.
- Add `realized_vol_rolling` — rolling realized volatility proxy (stddev of log-returns) over a time window.
- Add `oi_position_pct` — OI position within the day's high/low range, per tick: [0, 100].
- Add `tick_velocity` — rolling tick count per minute over a configurable time window.
- Expose `compute_backtest_metrics` as a public free function in `portfolio::engine` — non-OHLCV strategy types can produce identical metrics without duplicating the calculation logic.

### v0.3.4

- Add single-leg option spread types: `LongCall`, `LongPut`, `NakedCall`, `NakedPut` to `SpreadType` enum
- Add `ExitReason::Settlement` for option expiry settlement exits
- Add `leg_expiry_timestamps` parameter to `run_spread_backtest` for per-leg expiry tracking
- Positions are force-closed at settlement when any leg expires, with premiums replaced by intrinsic value
- Prevent re-entry after all legs have expired

### v0.3.3

- Add `batch_spread_backtest` function for running multiple spread backtests in parallel via Rayon
- Add `PyBatchSpreadItem` class for defining individual items in a batch spread backtest
- Shared data (timestamps, underlying close) is converted once and reused across all items
- GIL released during parallel execution for maximum Python concurrency
- Each item carries its own `strategy_id`, leg configs, signals, spread type, and optional max loss / target profit
- Returns a list of `(strategy_id, PyBacktestResult)` tuples preserving result-to-input mapping

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
- 30+ performance metrics: Sharpe, Sortino, Calmar, Omega, SQN, profit factor, drawdown duration, and more
- 10 technical indicators (SMA, EMA, RSI, MACD, Stochastic, ATR, Bollinger Bands, ADX, VWAP, Supertrend)
- Stop-loss management: fixed, ATR-based, and trailing stops
- Take-profit management: fixed, ATR-based, and risk-reward targets
- PyO3 Python bindings for seamless Python integration
