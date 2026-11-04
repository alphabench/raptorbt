# RaptorBT

[![PyPI](https://img.shields.io/pypi/v/raptorbt.svg)](https://pypi.org/project/raptorbt/)
[![Python](https://img.shields.io/pypi/pyversions/raptorbt.svg)](https://pypi.org/project/raptorbt/)
[![Rust](https://img.shields.io/badge/Rust-powered-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/pypi/l/raptorbt.svg)](https://opensource.org/licenses/MIT)
[![Support](https://img.shields.io/badge/Support-the%20project-ff69b4)](https://checkout.dodopayments.com/buy/pdt_0NmIpbPfM2KlwPVgZ6kMX?redirect_url=https%3A%2F%2Fwww.alphabench.in%2Fraptorbt%2Fthanks)

**One strategy class that backtests, papers and trades live, on one deterministic Rust engine.**

RaptorBT is a strategy runtime written in Rust with Python bindings (PyO3). You write a `Strategy` class once.
The same class runs over history in a batch (`run_strategies`), on a pushed feed in a `LiveNode`, against a
simulated venue or against a broker adapter, and decides the same thing in each, because they share one engine.
The priorities are strictly ordered: quantitative correctness, speed, memory, expressiveness, usability for AI
agents, production reliability. Speed is never bought with correctness.

It models orders and their lifecycle, fills, fee schedules by date, margin, option structures, expiries and
corporate actions, trading sessions, multi-timeframe bars, risk limits, and an append-only journal that lets a
crashed live node resume. It does not ship market data, a trading calendar or a broker account: you bring those.

## Install

```bash
pip install raptorbt
```

Python 3.10 to 3.13, on Linux and macOS. Bar data is NumPy arrays, so install `numpy` too. To build from
source see [Building from source](#building-from-source).

## A first strategy

A moving-average crossover on synthetic one-minute bars:

```python
import numpy as np
from raptorbt import BacktestConfig, Indicator, Strategy, run_strategies

n = 400
close = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, n))
bars = dict(timestamps=(np.arange(n, dtype=np.int64) + 1) * 60_000_000_000, open=np.r_[close[0], close[:-1]],
            high=close + 0.5, low=close - 0.5, close=close, volume=np.full(n, 1e3))

class SmaCross(Strategy):
    def on_start(self, ctx):
        self.fast = self.register_indicator(Indicator.sma(5), symbol="AAA")
        self.slow = self.register_indicator(Indicator.sma(20), symbol="AAA")

    def on_bar(self, ctx):
        if not self.indicators_initialized():
            return
        if ctx.is_flat and self.fast.value > self.slow.value:
            self.enter(size_frac=0.5)
        elif not ctx.is_flat and self.fast.value < self.slow.value:
            self.close_position()

(run,) = run_strategies([SmaCross()], {"AAA": bars}, BacktestConfig(initial_capital=100_000.0), bar_stamp="close")
print(run.result.metrics.total_trades, round(run.result.metrics.total_return_pct, 2))
```

Timestamps are `int64` nanoseconds. `bar_stamp="close"` says they mark the end of each bar; use
`bar_stamp="open"` (with `bar_ns=`) when they mark the start, as many broker candles do. Always pass it: it
decides what a strategy may know at each instant.

## The core ideas

**Strategy.** A subclass of `raptorbt.Strategy`. The engine owns the clock, the account and the fills; the
class owns decisions. Hooks (all optional) are `on_init`, `on_start`, `on_bar`, `on_fill`,
`on_order_event`, `on_position_event`, `on_timer`, `on_stop` and the tick hooks (`on_trade`, `on_quote`,
`on_book`). Parameters live in a frozen `StrategyConfig` dataclass with bounds (`param(...)`), and what a
strategy needs (instruments, timeframes, warm-up, holding style) can be declared as data
(`Strategy.requirements`) so a host can plan before it runs.
Skill: [strategy](skills/strategy/SKILL.md).

**ctx.** What a hook reads: the current bar, `ctx.is_flat`, positions, `equity`, `cash`, `ctx.history(n)`,
`ctx.data` (any subscribed instrument and timeframe) and, in a portfolio run, `ctx.symbol` and
`ctx.series(symbol)`. It shows only what had closed or arrived at the event: a strategy cannot read the
future through it. Decisions and events are separate channels: `self.enter()` queues an intent and returns
nothing; whether and when it filled arrives later in `on_fill`.

**Subscriptions.** `self.subscribe(symbol, timeframe="5m", history=20)` in `on_init` asks the engine to build
completed higher-timeframe bars from the primary feed (or to deliver trades, quotes or order books). They
arrive as `on_bar(ctx, event)` with `event.timeframe`, one primary bar after the window closes, so a
decision never uses a bar that was still forming. Skill: [subscriptions](skills/subscriptions/SKILL.md).

**Orders.** `enter()` and `close_position()` are market intents. Everything else is a typed order sent with
`self.submit_order(...)`: `Market`, `Limit`, `StopMarket`, `StopLimit`, trailing and if-touched variants,
`Twap`, brackets, one-cancels-other, and option structures (`submit_structure`, hedge-first: the bought legs
go out first, a sold leg when its hedges have filled). Orders have a lifecycle (submitted, accepted,
partially filled, filled, cancelled, expired, rejected) and a refused order says why, in a structured error.
Skill: [orders](skills/orders/SKILL.md).

```python
import numpy as np
from raptorbt import Strategy, run_strategies
from raptorbt.strategy import orders

n = 60
close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, n))
bars = dict(timestamps=(np.arange(n, dtype=np.int64) + 1) * 60_000_000_000, open=np.r_[close[0], close[:-1]],
            high=close + 0.5, low=close - 0.5, close=close, volume=np.full(n, 1e3))

class BuyTheDip(Strategy):
    def on_init(self, ctx):
        self.subscribe("AAA", timeframe="5m", history=20)        # 5-minute bars, built from the 1-minute feed

    def on_bar(self, ctx, event):
        if event.timeframe == "5m" and ctx.is_flat:
            low = float(ctx.data.history("AAA", "5m").low.min())
            self.submit_order(orders.Limit(side="buy", price=round(low, 2), quantity=10))

(run,) = run_strategies([BuyTheDip()], {"AAA": bars}, bar_stamp="close")
print(run.result.metrics.total_trades)
```

**run_strategies.** Many strategies over the same data in one call, each with the result it would get alone.
Data is held once and shared; strategies may share an account group (`AccountSpec`) or have their own.
`run_single_backtest`, `run_basket_backtest`, `run_spread_backtest` and the other array runners remain for
vectorised work (parameter sweeps, precomputed signals) and return the same result types.
Skills: [backtesting](skills/backtesting/SKILL.md), [optimization](skills/optimization/SKILL.md).

**LiveNode.** The same strategies on a feed you push one event at a time (`push_bar`, `push_tick`,
`push_depth`, `push_report`, `advance_time`). A node that is fed the groups a batch run saw reproduces the
batch run exactly. It is single-threaded, its memory does not grow with the feed, and it can adopt positions
the account already holds and warm up from history without trading.

**Adapters.** A node's `execution=` is where orders go: the in-process engine (the default), `PaperExecution`
(a simulated venue with broker-style reports) or a broker adapter. `raptorbt.adapters` holds the common layer
(outbox, rate limits, retries, duplicate-order guard) and mappings for Kite, Upstox and Dhan, plus a
conformance kit (`raptorbt.adapters.conformance`) that tests any adapter you write.
Skills: [execution](skills/execution/SKILL.md), [deployment](skills/deployment/SKILL.md).

**Journal.** `LiveNode(journal=Journal(...))` records everything the node is told, decides and sends,
before it acts, as a hash-chained append-only log. After a crash a fresh node replays it, checks that each
replayed record equals the stored one, and reconciles with the venue instead of re-sending an unconfirmed
order. A fencing epoch stops a stale process from sending.

```python
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt import journal as rj
from raptorbt.execution import PaperExecution

class BuyOnce(Strategy):
    def on_start(self, ctx):
        self.done = False

    def on_bar(self, ctx):
        if not self.done:
            self.enter(size_frac=0.1)
            self.done = True

node = LiveNode([BuyOnce()], ["AAA"], BacktestConfig(initial_capital=100_000.0, fees=0.0), bar_stamp="close",
                execution=PaperExecution(), journal=rj.Journal({"run": "demo"}, memory=True))
for i in range(5):
    node.push_bar("AAA", (i + 1) * 60_000_000_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0)

count, chain, header = node.journal.head()
assert rj.verify(node.journal.since(0), header) == chain          # the whole chain checks out
print(len(node.positions(0)["AAA"]), count)
```

Skill: [live-trading](skills/live-trading/SKILL.md). Contract:
[docs/architecture/journal-and-recovery.md](docs/architecture/journal-and-recovery.md).

## More examples

The examples in this section run as written against four names the test suite supplies: `bars`
(`{symbol: {"timestamps", "open", "high", "low", "close", "volume"}}` for three symbols of one-minute bars),
`ticks` and `depth` (the same shape for tick and order-book feeds) and `config` (a `BacktestConfig`). Swap in
your own data of the same shape. They are executed by `tests/python/test_strategy_api.py`, so they stay true.

**Several instruments and timeframes.** `ctx.data` only ever shows what has closed by the current event.

```python raptorbt-example
from raptorbt import Strategy, run_strategies


class MultiTimeframe(Strategy):
    def on_start(self, ctx):
        self.fast, self.slow = ctx.symbols[0], ctx.symbols[1]
        self.subscribe(self.fast, timeframe="5m", history=50)      # 5-minute bars of one instrument
        self.subscribe(self.slow, timeframe="15m", history=20)     # 15-minute bars of another

    def on_bar(self, ctx, event):
        if event.timeframe is not None:                            # a completed 5m / 15m bar: not a decision point
            return
        five = ctx.data.history(self.fast, "5m", count=10)         # read-only arrays
        quarter = ctx.data.latest(self.slow, "15m")                # last closed bar, or None
        if len(five) == 10 and quarter is not None and five.close[-1] > quarter.close:
            if ctx.data.position(self.fast) is None:
                self.enter(size_frac=0.1)


run_strategies([MultiTimeframe()], bars, config=config, bar_stamp="close")
```

**Indicators.** `ctx.indicators.<name>(instrument=, timeframe=, ...)` returns a streaming handle with `.ready`
and `.value`. Identical indicators are computed once however many strategies ask for them.

```python raptorbt-example
from raptorbt import Strategy, run_strategies


class Trend(Strategy):
    def on_start(self, ctx):
        s = ctx.symbols[0]
        self.rsi = ctx.indicators.rsi(instrument=s, period=4)
        self.bands = ctx.indicators.bollinger(instrument=s, period=5, k=2.0)

    def on_bar(self, ctx, event):
        if not (self.rsi.ready and self.bands.ready):
            return
        if event.timeframe is None and self.rsi.value < 30 and event.close < self.bands.values.lower:
            self.enter(size_frac=0.1)


run_strategies([Trend()], bars, config=config, bar_stamp="close")
```

**Orders and hooks.** A typed order names the instrument it routes to; a bracket attaches a stop and a target.
The engine reports what happened to `on_fill` and `on_position_event`, never to `on_bar`.

```python raptorbt-example
from raptorbt import Strategy, run_strategies
from raptorbt.strategy import orders


class Hooks(Strategy):
    def on_start(self, ctx):
        self.events = []

    def on_bar(self, ctx, event):
        if ctx.idx == 3 and ctx.is_flat:
            self.submit_bracket(
                orders.Market(side="buy", instrument=ctx.symbol, quantity=5),
                stop_trigger=event.close * 0.95,
                target_price=event.close * 1.10,
            )

    def on_fill(self, ctx, event):
        self.events.append(("fill", ctx.symbol, event.kind))

    def on_position_event(self, ctx, event):
        self.events.append(("position", ctx.symbol, event.kind))


strategy = Hooks()
run_strategies([strategy], bars, config=config, bar_stamp="close")
assert any(e[0] == "fill" for e in strategy.events)
```

**Ticks and order books.** Once a strategy calls `subscribe`, it receives trades, quotes and books only for the
instruments and types it subscribed to.

```python raptorbt-example
from raptorbt import Strategy, run_strategies


class Spread(Strategy):
    def on_start(self, ctx):
        s = ctx.symbols[0]
        self.subscribe(s, data_type="quote")
        self.subscribe(s, data_type="book")
        self.wide = 0

    def on_quote(self, ctx, quote):
        assert ctx.data.quote(quote.symbol) is quote
        if quote.spread > 1.0:
            self.wide += 1

    def on_book(self, ctx, book):
        assert ctx.data.book(book.symbol) is book


run_strategies([Spread()], ticks=ticks, depth=depth, config=config)
```

**A live node.** The same strategy class, fed one bar at a time. A node fed the bars of a batch run ends where
the batch run ends.

```python raptorbt-example
from raptorbt import LiveNode, Strategy, run_strategies


class Enter(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 2 and ctx.is_flat:
            self.enter(size_frac=0.25)


series = bars["AAA"]
batch = run_strategies([Enter()], {"AAA": series}, config=config, bar_stamp="close")[0]

node = LiveNode([Enter()], ["AAA"], config, bar_stamp="close")
for i in range(len(series["timestamps"])):
    node.push_bar("AAA", int(series["timestamps"][i]), *(float(series[k][i]) for k in ("open", "high", "low", "close", "volume")))
live = node.finish()[0]
assert live.result.metrics.total_trades == batch.result.metrics.total_trades
```

**Results.** Each strategy gets a `PortfolioResult`; `.result` is the `BacktestResult`.

```python raptorbt-example
from raptorbt import Strategy, run_strategies


class Enter(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 2 and ctx.is_flat:
            self.enter(size_frac=0.25)


(run,) = run_strategies([Enter()], {"AAA": bars["AAA"]}, config=config, bar_stamp="close")
result = run.result
trades = result.trades()
curve = result.equity_curve()
assert len(curve) > 0 and result.metrics.total_trades == len(trades)
print(result.metrics.total_return_pct, [t.pnl for t in trades])
```

## What you can rely on

- **Deterministic.** The same inputs and configuration give identical trades, curves and journals. Nothing
  depends on wall-clock time or on the order symbols are listed; a seeded Monte Carlo gives the same answer
  on any thread count.
- **No lookahead.** A hook sees a bar only once it has closed; an order decided on a bar cannot fill earlier
  than that bar's close (`fill_timing`: `same_bar_close` by default, or `next_bar_open`). Tests poison the
  future of every runner to prove it.
- **Fixed-point money.** Cash, fees, notionals and realised P&L are booked in integer minor units
  (`money_decimals`), so equity reconciles to the unit. Prices, quantities and indicators stay floats.
  See [docs/quantitative-correctness/numerics.md](docs/quantitative-correctness/numerics.md).
- **Fees by date.** Statutory exchange charges are dated and priced per instrument; a fill before the first
  encoded row is refused (`fee_schedule_gap`) rather than priced on a guess, and unverified rows are listed on
  the result. Brokerage comes from a broker's `RateCard`.
- **Structured errors.** Every error raised on purpose is a `raptorbt.errors.RaptorBTError` with a `code`,
  the accepted alternatives (`valid`) and the skill that covers it. See [docs/errors.md](docs/errors.md).
- **Honest numbers.** No performance figure is published here without a script that reproduces it: see
  `benches/python/README.md`.

## Crash reporting (optional)

Off by default. raptorbt sends nothing about crashes unless you turn this on and give it a collector: it has
no built-in destination. Enabled without an endpoint and key, it sends nothing and logs one line to the
`raptorbt` logger.

```bash
export RAPTORBT_CRASH_REPORTING=1
```

or in code, `raptorbt.enable_crash_reporting(endpoint=..., key=...)`. Point it at your own collector with
`RAPTORBT_CRASH_ENDPOINT` (base URL; reports go to `<endpoint>/batch`; an address already ending in `/batch` or `/e` is used as given) and `RAPTORBT_CRASH_KEY` (a
project key; the collector must accept the JSON batch described below), or pass `endpoint=` and `key=`. To disable, unset the variable (or
do not call the function); nothing else is needed.

Only an internal engine panic is reported, once per distinct (function, source line), at most five per
process. A `RaptorBTError`, an error you can fix, and an exception raised by your own strategy callback are
never reported. A report is one HTTPS POST from a background thread with a two second timeout; it never
raises into your program or delays it. It holds exactly these fields:

| Field | Value |
| --- | --- |
| `project`, `error_type` | `raptorbt`, `PanicException` |
| `version` | the raptorbt version |
| `operation` | the raptorbt function or constructor that crashed |
| `location` | the engine source `file:line` of the panic |
| `os`, `arch`, `python` | for example `Linux`, `x86_64`, `3.12` |
| `$exception_list` | one entry, `panic in <operation> at <location>` (so the collector can group issues) |
| `distinct_id` | a random id made when the process starts; not stored, not linked to you |
| `$process_person_profile`, `$geoip_disable` | `false`, `true`: ask the collector to keep no profile and no location |

Never sent: the panic message, argument values, data, symbols, prices, your code or file paths, local
variables, environment variables, the host name.

**Version check.** Separately from crash reporting, `import raptorbt` starts a background thread that asks
pypi.org, at most once a day, for the newest release and logs one line if yours is older. It sends no data
beyond an ordinary HTTPS request to pypi.org. Set `RAPTORBT_NO_VERSION_CHECK=1` to open no socket (CI
environments are skipped automatically).

## Documentation

| Where | What |
| --- | --- |
| [skills/INDEX.md](skills/INDEX.md) | One teaching document per domain (core, strategy, orders, subscriptions, instruments, execution, risk, live-trading, deployment, results, and more), each with runnable, tested examples. Also installed with the package: `import raptorbt.skills as skills; skills.for_task("...")`. |
| [examples/](examples/) | Complete runnable scripts, one folder per kind of strategy (single and multi-instrument, multi-timeframe, options, portfolio, tick-level, live trading), each on synthetic data and run by the tests. |
| [docs/api/README.md](docs/api/README.md) | The API reference, generated from each public module's `__all__`, signatures and docstrings. |
| [docs/architecture/event-semantics.md](docs/architecture/event-semantics.md) | Event ordering and visibility; the same folder holds the order lifecycle, execution adapters, journal and recovery, broker adapters and the embedding contract. |
| [docs/quantitative-correctness/numerics.md](docs/quantitative-correctness/numerics.md) | Every rounding and conversion boundary. |
| [docs/errors.md](docs/errors.md) | Every error code, its meaning and alternatives. |
| [CHANGELOG.md](CHANGELOG.md) | Every change, with the results that moved. |

## Keeping the docs true

Docs that drift are worse than none, so a machine checks them:

```bash
python -m raptorbt.skills check --strict    # every skill: examples run, names exist, nothing deprecated
python scripts/check_docs.py --api          # README and guides: examples run; docs/api is current
python -m pytest tests/python/test_strategy_api.py -k readme   # the `raptorbt-example` blocks above
python scripts/gen_api_reference.py         # regenerate docs/api after changing a signature or docstring
```

## Building from source

Needs a Rust toolchain and Python 3.10 to 3.13.

```bash
python -m venv .venv && .venv/bin/pip install maturin numpy scipy pytest
.venv/bin/python -m maturin develop --release      # a release build, always, when you measure or run examples
cargo test --no-default-features --all-targets      # Rust tests (the extension-module feature cannot link a test binary)
.venv/bin/python -m pytest tests/python
```

## Support

If RaptorBT helps your work, you can [support the project](https://checkout.dodopayments.com/buy/pdt_0NmIpbPfM2KlwPVgZ6kMX?redirect_url=https%3A%2F%2Fwww.alphabench.in%2Fraptorbt%2Fthanks).

## License

MIT. See [LICENSE](LICENSE).
