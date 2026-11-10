---
name: raptorbt-strategy
description: >
    Writing a class-based strategy: the Strategy class, its lifecycle hooks,
    StrategyConfig parameters, the requirements manifest, the context and
    state, and running one or many strategies over the same data.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
status: stable
---

# RaptorBT Strategy

## 1. When to use this skill

Load it before you write a `Strategy` subclass, choose hooks, give a strategy
parameters, declare what data it needs, or run it. A class-based strategy
decides bar by bar and can use any order type, any number of instruments, and
every event the engine emits. If your idea is two boolean arrays (entry, exit)
and nothing else, the array runners in [backtesting](../backtesting/SKILL.md)
are shorter; everything else belongs here.

## 2. Domain overview

A strategy is a Python class. The engine owns the clock, the account and the
fills; the class owns decisions. The two share state only through the context
(`ctx`) handed to each hook and the events the engine delivers.

**Lifecycle**, in the order it runs:

1. `on_init(ctx)`: once. Declare subscriptions and indicators. The account is
   not seeded yet.
2. `on_start(ctx)`: once, before the first bar. Positions and equity can be
   read. The full OHLCV arrays are on `ctx` for precomputing indicators.
3. For each bar, in time order: clock alerts (`on_timer`), completed
   higher-timeframe bars, then `on_bar(ctx)`. Whatever `on_bar` queued
   (`enter`, `close_position`, `submit_order`, ...) is applied by the engine
   within the same step.
4. The events that step produced reach the hooks: `on_order_accepted`,
   `on_fill`, `on_position_event`, `on_order_canceled`, `on_risk_event`, and so
   on. A fill that happens on bar `i` is reported after `on_bar` of bar `i`.
5. `on_stop(ctx)`: once, after the last bar, before the result is built.

**Decisions and events are different channels.** `on_bar` is where you decide;
it never receives a fill. A call such as `self.enter()` queues an intent and
returns nothing; whether it filled, when and at what price arrive later as an
event. Never assume a queued order is a position.

**The context** is the read surface: `ctx.bar`, `ctx.idx`, `ctx.equity`,
`ctx.cash`, `ctx.free_capital`, `ctx.position`, `ctx.positions()`, `ctx.is_flat`,
`ctx.history(n)`, `ctx.datetime`. Portfolio runs (`run_strategies`) add
`ctx.symbol`, `ctx.symbols`, `ctx.series(symbol)`, `ctx.position_for(symbol)`
and `ctx.open_orders()`. A single-instrument run has no `ctx.symbol`.
`ctx.positions(symbol=None)` is a method in every runner (the symbol is
ignored in a single-instrument run); the attribute used as a list is deprecated.

**State** is ordinary instance attributes. An instance run twice gets a fresh
engine account the second time, but your attributes keep their values; reset
them in `on_start`.

**Parameters** live in a frozen `StrategyConfig` dataclass. Declare bounds with
`param(default, min=, max=, choices=, description=)` so a host can validate and
sweep them without running the class.

**Requirements** (`Strategy.requirements`) declare, as data on the class, which
instruments, timeframes, data kinds, warm-up and holding style the strategy
needs. A host reads them without a run, plans data and capital from them, and
hashes them with the saved strategy.

**Event time.** A decision on bar `i` sees bar `i` complete, close included. It
can fill no earlier than that close (`fill_timing`, see core). Never read
`ctx.close[ctx.idx + 1]`: in a bar hook the arrays end at `ctx.idx` and that read
raises `IndexError`. In `on_start` they hold the whole run, so a statistic computed
there over all of it is the future too.

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.Strategy` | Base class. Override hooks. Defining a hook under both its current and its deprecated name raises `TypeError` at class creation. |
| `raptorbt.StrategyConfig` | Frozen dataclass base. `order_id_tag="O"` prefixes auto client ids (`"O-0"`); `params` is a free `dict`. |
| `raptorbt.StrategyContext`, `raptorbt.Bar` | The context, and the `Bar(timestamp, open, high, low, close, volume)` tuple. |
| `raptorbt.run_strategy_backtest(strategy, timestamps, open, high, low, close, volume, direction=1, symbol="ASSET", config=None, instrument_config=None, instrument=None, oms_type="netting", account_type="cash", leverage=1.0, bar_stamp=, bar_ns=None)` | One strategy, one instrument; returns a `BacktestResult`. Accepts the class or an instance. |
| `raptorbt.run_strategies(strategies, data, config=None, *, ticks=, depth=, directions=, instruments=, instrument_configs=, oms_type=, account_type=, leverage=, accounts=, bar_stamp=, bar_ns=, keep_curves=True, share=True, batch_size=None, corporate_actions=, adjustment_policy=)` | Many strategies over shared data; returns `list[PortfolioResult]` in input order (`.result` is the `BacktestResult`). |
| `raptorbt.run_strategies_windowed(strategies, windows, config=None, *, symbols=, history="declared", on_window=, ...)` | The same call over data fed in `raptorbt.Window`s (`bars=`, `ticks=`, `depth=`, `end_ns=`; ascending, none cutting a timestamp), so memory is bounded by the window. For a strategy whose logic reads at most the rows it declared, the results equal `run_strategies`'s, bit for bit. A strategy that declares `ctx.keep_history(N, symbol=)` sees exactly the last N rows in `ctx.series`; for any symbol it declares nothing for, every row is kept (memory of the run). `ctx.data.history` past the declared N is refused; `ctx.series` past it is your contract (it shows N rows and raises nothing). Declare in `on_start`: a `ctx.keep_history` after the first window that needs released rows is refused. A `subscribe(..., data_type="quote", history=)` is for ticks, not bars. `history="all"` or a row count overrides. Refuses ATR stops, a restating `adjustment_policy="adjusted"` and `at(...)` expressions without `bar_ns`. |
| `raptorbt.KernelSession`, `raptorbt.PortfolioSession`, `raptorbt.EngineEvent`, `raptorbt.PositionSnapshot` | The engine objects behind the runners. Read events (`EngineEvent.kind`, `.idx`, `.price`, `.size`, `.reject_reason`, `.trade`) and positions from them; do not drive a run by hand. |

Hooks (all default to no-ops):

| Hook | Fires |
| --- | --- |
| `on_init(ctx)`, `on_start(ctx)`, `on_stop(ctx)` | Once each, as above. |
| `on_bar(ctx)` or `on_bar(ctx, event)` | Each primary bar. The two-argument form also receives completed composite bars (`event.timeframe` is `None` for a primary bar, `"5m"` for a composite one). |
| `on_timer(ctx, event)` | A due `self.clock` alert or timer, before the bar's data. |
| `on_trade`, `on_quote`, `on_book`, `on_market_data` | Tick runs (see [market-data](../market-data/SKILL.md)). |
| `on_fill(ctx, event)` | An entry or exit filled. `event.kind` is `"entered"` or `"exited"` for an intent, `"order_filled"` for a typed order. |
| `on_position_event(ctx, event)` | After a position opened (`"entered"`) or closed (`"exited"`; `event.trade` is the round trip). |
| `on_order_accepted`, `on_order_triggered`, `on_order_canceled`, `on_order_expired`, `on_order_rejected` | Typed-order lifecycle. A refused entry intent also reaches `on_order_rejected`. |
| `on_order_event(ctx, event)` | Every typed-order event, after its specific hook. |
| `on_risk_event`, `on_margin_call`, `on_contract_event`, `on_corporate_action` | Risk, margin, expiry and corporate-action events (see [risk](../risk/SKILL.md) and [backtesting](../backtesting/SKILL.md)). |
| `on_algo_started`, `on_algo_completed` | A `Twap` schedule began, or released its last slice. |

Old hook names still work but are deprecated; write the current ones above.

Intents and orders (queued from any hook that runs before the step):

| Call | Effect |
| --- | --- |
| `enter(size_frac=None, stop_price=None, target_price=None, side=None)` | Market entry in the run's direction; `side="buy"/"sell"` overrides it. No `size_frac` means all available capital. `enter_long`, `enter_short` and `buy` are the same intent spelled out. |
| `close_position(position_id=None, symbol=None)` | Close the open position, one position (hedging) or one symbol (portfolio). |
| `submit_order(order, client_id=None, parent=None, symbol=None)` | A typed order; returns its client id. See [orders](../orders/SKILL.md). |
| `register_indicator(indicator, stream_id=None, symbol=None)`, `indicators_initialized()` | An auto-updated streaming indicator; true once every registered one has warmed up. |
| `subscribe(instrument, *, data_type="bar", timeframe=None, history=None)`, `subscribe_bars(step, unit)` | Declare data; see [subscriptions](../subscriptions/SKILL.md). |

Declarations, from `raptorbt.strategy`:

| Name | What it is |
| --- | --- |
| `Requirements(roles, timeframes, holding, data=("bar",), warmup=None, max_structure=None, params=None, max_position=None)` | The manifest. `.to_dict()`, `Requirements.from_dict(d)`. |
| `Underlying(asset=)`, `Instrument(symbol, asset=)`, `Universe(asset=, min=, max=, data=)`, `Future(of, contract=)`, `Chain(of, expiry, strikes, rights=)` | Roles: bound by the deployer to one symbol, fixed, bound by the deployer to a list of `min` to `max` symbols (a basket; `bindings={"basket": ["A", "B"]}`, read with `ctx.role("basket")`, a tuple), derived future, derived option chain. |
| `StrikeWindow(low, high)`, `StrikeWindow.around(n)`, `ByDelta(low, high)` | Which strikes of a chain. |
| `Holding.intraday(flatten_before="HH:MM")`, `Holding.overnight()` | Holding style. |
| `Bars(n, timeframe=None)`, `Duration.minutes/hours/days(n)` | Warm-up. |
| `Legs(OptionLeg(side=, right=, offset= or delta=+window=, expiry=, lots=1, chain=None), ...)` | The legs of the structure held at once: side (`buy`/`sell`), right, strike as listed strikes from the money (`offset`, up positive) or a `delta` target with the `window` of strikes it may sit in, expiry selector. Validated against the chain role. `Legs(4, lots=2)` is refused (`legs_need_shape`). |
| `MaxPosition(lots=None, notional=None, fraction=None)` | The largest position in a role (`Requirements(max_position={role: MaxPosition(...)})`), per instrument the role resolves to; at least one field; `fraction` is of the capital in (0, 1]. A host sizes capital from it without asking. |
| `param(default, min=, max=, choices=, description=)`, `params_schema(Config)`, `ParamSchema.validate(values)`, `ParamSpec`, `Issue` | Parameter declaration and validation; `validate` returns `list[Issue]` (`code`, `path`, `message`). |
| `check_requirements(StrategyClass, dry_run=True)` | Static checks, then a dry run on a synthetic tape with every `Underlying` and `Universe` role bound (so `ctx.role` works in it); returns `list[Issue]`. |
| `requirements_hash(StrategyClass or Requirements)` | SHA-256 of the canonical declaration plus the engine version. |
| `resolve(requirements, *, now_ns, bindings, specs, prices=None, vols=None, chain=None)` | A `SymbolPlan`: concrete symbols per role at a time. Gaps are `Issue`s, never guesses. |
| `resolve_legs(requirements, *, now_ns, bindings, specs, prices, chain=None)` | A `LegPlan`: the contracts each declared leg can mean, their units and signs, for subscribing and margining (`plan.lines(prices)` feeds `MarginModel.required_legs`). `Legs.structure(vol=)` builds the `Structure` of the legs. |
| `RequirementsError` | A declaration that cannot be built. |

## 4. Standard workflow

1. Write the config dataclass: every number the strategy tunes, with bounds.
2. Subclass `Strategy`. In `on_init` or `on_start`: create and register
   indicators, subscribe to extra data, reset your own state.
3. In `on_bar`, return early until `indicators_initialized()` is true, then
   read the context, decide, and queue at most one entry or close per bar.
4. React to fills in `on_fill` and `on_position_event`, not in `on_bar`.
5. Declare `requirements` if a host will deploy the class, and run
   `check_requirements` until it returns `[]`.
6. Run with `run_strategy_backtest` (one instrument) or `run_strategies`
   (several), always passing `bar_stamp`.
7. Validate (section 9).

## 5. Examples

A complete strategy: an SMA crossover with a typed config, a warm-up guard, and
a hook that records what the engine reports. With `next_bar_open` a decision
made at one bar's close fills one bar later, and the hook shows it:

```python
from dataclasses import dataclass

import numpy as np
from raptorbt import (BacktestConfig, Indicator, Strategy, StrategyConfig,
                      run_strategy_backtest)
from raptorbt.strategy import param


def make_bars(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return ts, open_, high, low, close, np.full(n, 1000.0)


@dataclass(frozen=True)
class CrossConfig(StrategyConfig):
    fast: int = param(5, min=2, max=50, description="fast SMA period")
    slow: int = param(20, min=5, max=200, description="slow SMA period")
    size: float = param(0.5, min=0.05, max=1.0, description="fraction of capital per entry")


class SmaCross(Strategy):
    def __init__(self, config=None):
        super().__init__(config or CrossConfig())
        self.fills = []

    def on_start(self, ctx):
        self.fills = []                                   # reset per-run state
        self.fast = self.register_indicator(Indicator.sma(self.config.fast))
        self.slow = self.register_indicator(Indicator.sma(self.config.slow))

    def on_bar(self, ctx):
        if not self.indicators_initialized():
            return
        if ctx.is_flat and self.fast.value > self.slow.value:
            self.enter(size_frac=self.config.size)
        elif not ctx.is_flat and self.fast.value < self.slow.value:
            self.close_position()

    def on_fill(self, ctx, event):
        self.fills.append((event.kind, event.idx))


ts, o, h, l, c, v = make_bars()
fills = {}
for timing in ("same_bar_close", "next_bar_open"):
    strategy = SmaCross()
    result = run_strategy_backtest(
        strategy, ts, o, h, l, c, v, symbol="TEST", bar_stamp="close",
        config=BacktestConfig(fees=0.0, fill_timing=timing),
    )
    fills[timing] = strategy.fills

assert fills["same_bar_close"][0][0] == "entered"
assert fills["next_bar_open"][0][1] == fills["same_bar_close"][0][1] + 1   # same decision, one bar later
assert result.metrics.total_trades > 0
```

Parameters are data. `params_schema` describes the config, and `validate`
returns every problem at once instead of raising on the first:

```python
from dataclasses import dataclass
from raptorbt import StrategyConfig
from raptorbt.strategy import param, params_schema


@dataclass(frozen=True)
class Config(StrategyConfig):
    fast: int = param(5, min=2, max=50, description="fast period")
    mode: str = param("long", choices=["long", "short"])


schema = params_schema(Config)
assert schema.validate({"fast": 10}) == []
issues = schema.validate({"fast": 1, "mode": "flat", "typo": 3})
assert {i.code for i in issues} == {"out_of_bounds", "not_a_choice", "unknown_param"}
assert Config(fast=8).fast == 8
```

A manifest. `check_requirements` dry-runs the class on a synthetic tape built
from the declaration and flags any subscription it did not declare. The hash
changes with the declaration and survives a JSON round trip:

```python

from raptorbt import Strategy
from raptorbt.strategy import (Bars, Holding, Requirements, Underlying,
                               check_requirements, requirements_hash)


class Declared(Strategy):
    requirements = Requirements(
        roles={"asset": Underlying(asset="equity")},
        timeframes=["1m", "5m"],
        holding=Holding.intraday(flatten_before="15:15"),
        warmup=Bars(30),
    )

    def on_init(self, ctx):
        self.subscribe("ASSET")                       # a role is bound to its upper-cased name
        self.subscribe("ASSET", timeframe="5m")


class Undeclared(Strategy):
    requirements = Declared.requirements

    def on_init(self, ctx):
        self.subscribe("ASSET")
        self.subscribe("ASSET", timeframe="15m")      # 15m is not in timeframes


assert check_requirements(Declared) == []
assert [i.code for i in check_requirements(Undeclared)] == ["undeclared_timeframe"]

again = Requirements.from_dict(Declared.requirements.to_dict())
assert requirements_hash(again) == requirements_hash(Declared)
fewer = Requirements(roles=Declared.requirements.roles, timeframes=["1m"], holding=Holding.overnight())
assert requirements_hash(fewer) != requirements_hash(Declared)
```

Several strategies over shared data. Each gets its own account and result, and
`ctx.symbol` says which instrument's bar is being processed:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategies


def bars(seed, base, n=120):
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
    return dict(timestamps=ts, open=open_, high=np.maximum(open_, close) + 0.5,
                low=np.minimum(open_, close) - 0.5, close=close, volume=np.full(n, 1000.0))


class BuyOnce(Strategy):
    def __init__(self, symbol):
        super().__init__()
        self.target = symbol

    def on_bar(self, ctx):
        if ctx.symbol == self.target and ctx.idx == 5:
            self.enter_long(size_frac=0.4)


data = {"AAA": bars(1, 100.0), "BBB": bars(2, 50.0)}
results = run_strategies(
    [(BuyOnce("AAA"), ["AAA", "BBB"]), (BuyOnce("BBB"), ["AAA", "BBB"])],
    data, BacktestConfig(fees=0.0), bar_stamp="close",
)
assert len(results) == 2
assert all(r.result.metrics.total_trades == 1 for r in results)
```

## 6. Composition

- Orders beyond `enter` and `close_position`: [orders](../orders/SKILL.md).
- The account the strategy trades against: [portfolio](../portfolio/SKILL.md).
  Limits and kill switches: [risk](../risk/SKILL.md).
- Data and timeframes the strategy subscribes to:
  [subscriptions](../subscriptions/SKILL.md),
  [timeframes](../timeframes/SKILL.md), [indicators](../indicators/SKILL.md).
- Option chains and multi-leg structures a manifest names:
  [instruments](../instruments/SKILL.md).
- The same class runs live through `LiveNode` ([live-trading](../live-trading/SKILL.md)).
  Write it so nothing in `on_bar` depends on the run being a backtest.
- Runnable examples: `examples/single-instrument/sma_cross.py`, `examples/multi-instrument/relative_value.py`.

## 7. Constraints and correctness

- One entry or close intent per bar. `enter()` and `close_position()` queued on
  the same bar while in position raise `InvalidOrder` (`invalid_order_intent`).
  Use typed orders if you need a reversal.
- An intent is sized when it fills, not when it is queued. `size_frac` is a
  fraction of available capital at fill time; `enter()` with no size uses all
  of it.
- `enter()` and `close_position()` report through `on_fill` and
  `on_position_event`; only typed orders raise `on_order_accepted` and
  `on_order_event`.
- `on_bar(self, ctx)` sees primary bars only. Define `on_bar(self, ctx, event)`
  to also see composite bars.
- A hook that raises aborts the run: the engine does not catch strategy errors.
- A run is deterministic. Do not read the wall clock, an unseeded `random`, or
  global mutable state: backtest and live must see the same decisions.
- **`on_bar(t)` sees the position bar `t` left.** Stops, targets, trailing and every resting order
  you placed on an earlier bar trade bar `t`'s range before `on_bar(t)` runs, as at a venue: a long
  whose stop was touched is flat in `on_bar(t)`, a limit touched is already filled. Size entries
  from `ctx.positions()` and branch on it freely. Orders you place in `on_bar(t)` never trade
  inside bar `t`. The fill hooks (`on_fill`, `on_order_event`) for those intra-bar fills run before
  `on_bar(t)` and see data up to bar `t - 1` only; what they send fills at your fill timing (bar
  `t`'s close under `same_bar_close`, bar `t + 1`'s open under `next_bar_open`), except a fill at
  the bar's open (a gap through your stop), which is answered at that open.
- `bar_stamp` says whether your timestamps are bar opens (`"open"`, which also
  needs `bar_ns`) or closes (`"close"`). It is required: omitting it raises
  `missing_bar_stamp`.
- `check_requirements` reads the imported class, not source text. Importing a
  module runs its top level, so a host that must not import untrusted code
  reads the JSON that `to_dict()` exported from a trusted build.

## 8. Common mistakes

Queueing an entry and a close on the same bar:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest


class Both(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.enter(size_frac=0.5)
        if ctx.idx == 3:
            self.enter(size_frac=0.5)
            self.close_position()          # InvalidOrder: invalid_order_intent


n = 10
close = 100.0 + np.arange(n)
run_strategy_backtest(Both, np.arange(n, dtype=np.int64) * 60_000_000_000, close, close + 1,
                      close - 1, close, np.full(n, 1000.0), bar_stamp="close")
```

Passing something that is not a strategy:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import run_strategy_backtest

x = np.ones(5)
run_strategy_backtest(object(), np.arange(5, dtype=np.int64), x, x, x, x, x, bar_stamp="close")
```

- Reading `self.fast.value` before warm-up: it is `None`, so a comparison
  raises `TypeError`. Guard with `indicators_initialized()`.
- Assuming `self.enter()` made a position, then sizing the next order from
  `ctx.position` in the same `on_bar`. The position appears after the step.
- Keeping run state in `__init__` only: a second run of the same instance sees
  the first run's values. Reset in `on_start`.
- Looking ahead in precomputation: an indicator computed over `ctx.close` in
  `on_start` must not use values after the bar it is read at.
- `register_indicator` without `symbol=` in a portfolio run feeds the indicator
  every symbol's bars interleaved, and `check_requirements` warns about it.
  Pass `symbol=`.
- Assuming an indicator on another symbol is a bar behind: when instruments share
  a timestamp every registered indicator, on every symbol, is current as of that
  timestamp before any `on_bar` runs, whatever the symbols are called. Your own
  symbol's primary-stream indicator is the one revealed late: it shows the value
  before the bar until `on_bar`.
- Sizing from indicators in a fill hook under `fill_timing="next_bar_open"`: a quantity or level you compute in a fill hook (`on_fill`, `on_position_event`) answers a fill at a bar's open, but an indicator or object you own and update yourself already includes that bar's close. Size from values you stored on an earlier bar (for example in `on_bar`), not from indicators read in the hook.

## 9. Validation

1. `check_requirements(MyStrategy) == []` before the class goes anywhere.
2. Run with `fees=0.0`, then with real costs, then with
   `fill_timing="next_bar_open"`; the result should degrade gracefully.
3. Run twice and compare `result.trades()`: identical.
4. Record the hooks (as in the first example) and assert the sequence you
   expect on a tape small enough to read by hand.
5. Where two runners overlap, run the strategy through both and compare.
6. Before live use, read [live-trading](../live-trading/SKILL.md): a backtest
   proves the logic, not the connection.

## 10. Related skills

- [Core](../core/SKILL.md)
- [Orders](../orders/SKILL.md)
- [Portfolio](../portfolio/SKILL.md)
- [Risk](../risk/SKILL.md)
- [Backtesting](../backtesting/SKILL.md)
- [Subscriptions](../subscriptions/SKILL.md)
