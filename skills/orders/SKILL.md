---
name: raptorbt-orders
description: >
    Order construction and lifecycle: the typed order kinds, sizing forms,
    time in force, brackets, OCO links, modify and cancel, option structures,
    and the reasons an order is refused.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
    - raptorbt-instruments
status: stable
---

# RaptorBT Orders

## 1. When to use this skill

Load it when a strategy needs more than `enter()` and `close_position()`: a
limit or stop, a stop-loss and target, a bracket, a trailing stop, a sliced
order, an order that expires, a multi-leg option position, or when you must
explain why an order did not trade. The simple intents are in
[strategy](../strategy/SKILL.md); position limits are in [risk](../risk/SKILL.md).

## 2. Domain overview

A typed order is a frozen dataclass from `raptorbt.strategy.orders`, sent with
`self.submit_order(order)` from any hook. The call queues it and returns a
**client id** (`"O-0"`, `"O-1"`, ...). Nothing has happened yet: the engine
applies the queue within the step, and the outcome arrives as events keyed by
that client id.

**Kinds.**

| Class | Behaviour |
| --- | --- |
| `Market` | Fills on the submission bar at the engine's fill price (see `fill_timing`). |
| `Limit(price, post_only=False)` | Rests; fills when the market trades at or through `price`, at `price`. A gap through it fills at the open. |
| `StopMarket(trigger)` | Rests; becomes a market order when the market trades at or through `trigger` (a sell stop triggers below, a buy stop above). Fills at the trigger, or at the open if the bar gapped past it. |
| `StopLimit(trigger, price)` | Once `trigger` fires, rests as a limit at `price` from the next bar. |
| `MarketIfTouched(trigger)`, `LimitIfTouched(trigger, price)` | The mirror of a stop: fires when the market trades *favorably* to `trigger` (a buy fires when price falls to it). |
| `MarketToLimit` | Fills at the next bar's open. |
| `TrailingStopMarket(offset, offset_kind="price")`, `TrailingStopLimit(..., limit_offset)` | A stop trailing the favorable extreme by `offset` (`"price"`, `"bps"` or `"ticks"`; ticks need an instrument with a `price_increment`). |
| `Twap(slices, every, every_bars=, bar_ns=)` | A schedule that releases `slices` equal market orders, the first at once. Needs an explicit `quantity`. Slice ids are `"<parent>#<n>"`. |

**Every order** also takes `side` (`"buy"` or `"sell"`, required), `instrument`
(routes it in a multi-instrument run), `tif`, `expire_ns`, `reduce_only`,
`stop_price`, `target_price` (protective levels attached to the position an
opening order creates) and `tags`.

**Sizing forms.** Give exactly one of `quantity` (units) or `size_frac`
(a fraction of available capital, resolved when the order fills). Giving both
raises `conflicting_order_args`. Giving neither means *close the whole
position*: on an order that would open a position it is rejected as
`invalid_qty`.

**Time in force.** `auto` (the default) is `day` on an instrument that has a
`SessionSchedule` (the order is cancelled at that session's close, as an
exchange does) and `gtc` on one that has none. `gtc` rests until filled or
cancelled and carries across sessions; ask for it explicitly, and only where
the venue supports it (a live adapter refuses `gtc` unless it was told to send
it as `day`). `day` expires at the end of the trading day in the instrument's
session zone (`session_tz_offset_ns` without a session), `gtd` (needs
`expire_ns`; expires at the first bar stamped at or after it), `ioc` and `fok`
(cancelled after their first bar if not filled), `at_open` and `at_close`
(market orders only: fill at the next bar's open, or at the next bar's close).

**Lifecycle.** An order is `Submitted`, then `Accepted` (it is working),
possibly `Triggered` (a stop fired), then `PartiallyFilled` and `Filled`. The
end states are `Filled`, `Canceled`, `Expired` and `Rejected`. In a backtest
each state arrives as an event: `order_accepted`, `order_triggered`,
`order_filled`, `order_canceled`, `order_expired`, `order_rejected`, each
reaching its own hook and then `on_order_event`. The model, including
the cancel-and-replace pending states a live venue adds, is in
`docs/architecture/order-lifecycle.md` and [execution](../execution/SKILL.md).

**Resting orders start matching on the next bar.** An order cannot rest into a
bar that already closed when it was placed.

**Exit rules.** `self.set_exit(ctx, ExitRule(...), symbol=)` (or `symbols=` with
`name=`, or a structure's `group=`) states when to close a position that already
exists: `stop_loss_pct` / `take_profit_pct` (percent of what the positions cost),
`stop_loss` / `take_profit` (amounts of P&L), `premium_at_or_above` /
`premium_at_or_below` (the sum of direction × last price × size; for a long of
`q` units, `premium_at_or_below=level * q` is a stop at the price `level`). The
first condition that holds fires, in that order. Mix `ExitEvents` in before
`Strategy` to have the rules evaluated after every `on_bar` and `on_trade`, or
call `self.update_exits(ctx)` yourself. The last price is the last bar close, else
the last traded `ltp` (`> 0`): a quote-only tick row is no trade and never marks a
rule, and a symbol with only quotes so far has no price, so its rule waits. To mark
at a quote, pass it: `self.update_exits(ctx, marks={sym: quote.mid})`. A rule that fires sends one reduce-only
market order per open position (client id `{rule id}:x{n}`). A rule has no
modify: to move a stop or a target, `cancel_exit(handle)` and `set_exit` a new
rule. The live behaviour (a venue that holds the rule, retries) is in
[live-trading](../live-trading/SKILL.md).

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.strategy.orders.{Market, Limit, StopMarket, StopLimit, MarketIfTouched, LimitIfTouched, MarketToLimit, TrailingStopMarket, TrailingStopLimit, Twap}` | The typed orders above. Construction validates side, tif, prices and sizing and raises `InvalidOrder` with a `code`. |
| `Strategy.submit_order(order, client_id=None, parent=None, symbol=None)` | Queue an order; returns its client id. `parent=<id>` holds this order until the parent fills and kills it if the parent dies. |
| `Strategy.submit_bracket(entry, stop_trigger, target_price, stop_limit_price=None)` | An entry plus a `reduce_only` stop and target that activate when it fills, linked so one cancels the other. Returns `(entry_id, stop_id, target_id)`. Netting only: under hedging it raises `bracket_under_hedging`. |
| `Strategy.link_oco(*client_ids)` | One-cancels-other among two or more submitted orders. |
| `Strategy.cancel_order(client_id)`, `cancel_all_orders()` | Cancel one working order (or a `Twap`'s unreleased slices), or all of them. A finished order is a no-op; an id never submitted raises `unknown_order`. |
| `Strategy.modify_order(client_id, units=None, size_frac=None, limit_price=None, trigger_price=None)` | Replace price or size of a working order. `units` is the quantity. |
| `Strategy.submit_structure(ctx, structure, *, chain, underlying, spot, lots=1, ...)`, `update_structures(ctx, marks)`, `close_structure(group)`, `structure_status(group)` | A multi-leg option position placed hedge-first in every mode: bought legs first, each sold leg when its hedges have filled, hedges closed if a sold leg fails; see `docs/structures.md`. |
| `Strategy.cache` | `order(id)`, `orders()`, `orders_open()`, `is_order_open(id)`, `realized_pnl()`: order states folded from events. |
| `raptorbt.MarketOrder`, `raptorbt.ClosePosition` | The legacy intents behind `enter()` and `close_position()`. Prefer the methods. |
| `raptorbt.StopConfig`, `raptorbt.TargetConfig` | Fixed, ATR, trailing and risk-reward stop and target models for `BacktestConfig` (`set_fixed_stop(0.02)` is 2 percent; fractions, not percents). |
| `raptorbt.strategy.ExitRule`, `ExitEvents`, `ExitHandle`; `Strategy.set_exit(ctx, rule, *, symbol= / symbols=, name= / group=)`, `update_exits(ctx, marks=None)`, `cancel_exit(handle)` | Exit rules on a position or a named group (see "Exit rules" above). `set_exit` returns the handle: `state` is `armed`, `venue`, `fired`, `done`, `canceled` or `failed`, `reason` the condition that fired, `close_ids` the orders it sent. |

Events carry `kind`, `idx`, `client_order_id`, `price`, `size` and
`reject_reason`. Order-level `reject_reason` values: `invalid_qty`,
`insufficient_capital`, `reduce_only`, `post_only`, `zero_size`,
`insufficient_margin`, `margin_unpriced`, `expired`, `inactive`,
`market_closed`, `squareoff`, `not_tradable` (an index takes no orders),
`warmup` (decided before `trade_from`), plus the risk reasons (`order_size`,
`max_position`, `max_exposure`, `max_positions`, `daily_loss`, `killed`,
`drawdown_halt`, `margin_call`, and `no_mark` for a market order on a real venue
that could not be priced against a notional or exposure limit; see
[risk](../risk/SKILL.md)).

## 4. Standard workflow

1. Pick the order kind from what you want at the venue, not from a habit.
2. Size it with `quantity` or `size_frac`, never both.
3. Choose the time in force. A resting order you forget about is a position
   you did not intend; give it `day`, `gtd` or `ioc` unless it should live
   indefinitely.
4. Keep the client id `submit_order` returns if you will cancel, modify or link
   it.
5. Handle the outcome in `on_fill`, `on_order_rejected` and
   `on_order_canceled`; read `event.reject_reason`.
6. Test on a tape small enough to read: assert the event kinds, bars and fill
   prices.

## 5. Examples

Every order kind on one tape, with where and at what price it fills. The tape
dips to 98 on bar 2 and spikes to 105 on bar 3:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

closes = np.array([100.0, 100.0, 98.0, 104.0, 103.0])
tape = dict(
    timestamps=np.arange(5, dtype=np.int64) * 60_000_000_000,
    open=np.array([100.0, 100.0, 99.0, 99.0, 104.0]),
    high=np.array([101.0, 101.0, 100.0, 105.0, 105.0]),
    low=np.array([99.0, 99.0, 97.0, 98.0, 102.0]),
    close=closes, volume=np.full(5, 1e6),
)


def first_fill(order):
    class Place(Strategy):
        fills = []

        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.submit_order(order)

        def on_order_event(self, ctx, event):
            if event.kind == "order_filled":
                self.fills.append((event.idx, event.price))

    s = Place()
    s.fills = []
    run_strategy_backtest(s, **tape, config=BacktestConfig(fees=0.0), bar_stamp="close")
    return s.fills[0]


q = dict(side="buy", quantity=10)
assert first_fill(orders.MarketToLimit(**q)) == (1, 100.0)                         # next bar's open
assert first_fill(orders.Limit(price=98.0, **q)) == (2, 98.0)                      # rests, then touched
assert first_fill(orders.StopMarket(trigger=103.0, **q)) == (3, 103.0)             # breakout buy
assert first_fill(orders.StopLimit(trigger=103.0, price=103.5, **q)) == (4, 103.5)  # triggers, then rests
assert first_fill(orders.MarketIfTouched(trigger=98.0, **q)) == (2, 98.0)          # buy the dip
assert first_fill(orders.LimitIfTouched(trigger=98.0, price=98.5, **q)) == (3, 98.5)
```

A bracket: an entry, then a protective stop and a target linked so that the
one that fills cancels the other. The events say exactly what happened:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

closes = np.array([100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 104.0])
tape = dict(timestamps=np.arange(7, dtype=np.int64) * 60_000_000_000, open=closes,
            high=closes + 1.0, low=closes - 1.0, close=closes, volume=np.full(7, 1e6))


class Bracketed(Strategy):
    def __init__(self):
        super().__init__()
        self.events = []

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.ids = self.submit_bracket(orders.Market(side="buy", quantity=10),
                                           stop_trigger=98.0, target_price=104.0)

    def on_order_event(self, ctx, event):
        self.events.append((event.kind, event.idx, event.client_order_id))


s = Bracketed()
result = run_strategy_backtest(s, **tape, config=BacktestConfig(fees=0.0), bar_stamp="close")
entry_id, stop_id, target_id = s.ids
assert ("order_filled", 1, entry_id) in s.events           # entry fills, both legs activate
assert ("order_filled", 4, target_id) in s.events          # the target is touched on bar 4
assert ("order_canceled", 4, stop_id) in s.events          # and the stop is cancelled with it
trade = result.trades()[0]
assert (trade.entry_price, trade.exit_price, trade.exit_reason) == (100.0, 104.0, "Order")
```

Time in force, cancel and modify. An `ioc` order that cannot fill is cancelled
after one bar; a `gtd` order expires at the first bar at or after its time; a
modify moves a resting limit:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

DAY = 86_400 * 10**9
close = np.full(6, 100.0)
tape = dict(timestamps=np.arange(6, dtype=np.int64) * DAY // 2, open=close, high=close + 1.0,
            low=close - 1.0, close=close, volume=np.full(6, 1e6))


class Script(Strategy):
    def __init__(self):
        super().__init__()
        self.kinds = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Limit(side="buy", price=90.0, quantity=1, tif="ioc"))
            self.submit_order(orders.Limit(side="buy", price=90.0, quantity=1, tif="gtd",
                                           expire_ns=int(DAY * 1.2)))
            self.keep = self.submit_order(orders.Limit(side="buy", price=90.0, quantity=1))
        if ctx.idx == 1:
            self.modify_order(self.keep, limit_price=100.5)    # raised above the market

    def on_order_event(self, ctx, event):
        self.kinds.append((event.kind, event.idx))


s = Script()
result = run_strategy_backtest(s, **tape, config=BacktestConfig(fees=0.0), bar_stamp="close")
assert ("order_canceled", 1) in s.kinds                   # ioc: one bar and gone
assert ("order_expired", 3) in s.kinds                    # gtd: first bar stamped at 1.5 days
assert ("order_filled", 2) in s.kinds                     # the modified limit, now marketable: bar 2 is the first to meet it
assert len(result.trades()) == 1
```

Refusals arrive as `order_rejected` with a reason; nothing is silently
resized. Here: no position to reduce, more than the cash allows, and a
post-only limit that would have crossed:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

close = np.array([100.0, 100.0, 101.0, 102.0, 103.0])
tape = dict(timestamps=np.arange(5, dtype=np.int64) * 60_000_000_000, open=close,
            high=close + 1.0, low=close - 1.0, close=close, volume=np.full(5, 1e6))


class Refused(Strategy):
    def __init__(self):
        super().__init__()
        self.reasons = {}

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="sell", quantity=5, reduce_only=True))
            self.submit_order(orders.Market(side="buy", quantity=1_000_000))
            self.submit_order(orders.Limit(side="buy", price=150.0, quantity=1, post_only=True))
            self.submit_order(orders.Market(side="buy"))          # no size on an opening order

    def on_order_rejected(self, ctx, event):
        self.reasons[event.client_order_id] = event.reject_reason


s = Refused()
result = run_strategy_backtest(s, **tape, config=BacktestConfig(fees=0.0), bar_stamp="close")
assert s.reasons == {"O-0": "reduce_only", "O-1": "insufficient_capital",
                     "O-2": "post_only", "O-3": "invalid_qty"}
assert result.trades() == []
```

A resting order (stop, target, limit, trailing, bracket leg) trades the range of every bar after the one
that placed it, and it does so before your `on_bar` for that bar runs: the strategy decides on a bar
knowing which of its resting orders that bar filled. An order never trades inside the bar on which
you placed it.

The marketable-limit rule: a limit that is already marketable fills at the
market, never at its own limit price. Under `same_bar_close` that market is the
price the order was decided at (the deciding bar's close, like a market order
placed there); under `next_bar_open` it is the next bar's open. `at_open` and `at_close` market orders fill at the next
bar's open and close:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

tape = dict(
    timestamps=np.arange(6, dtype=np.int64) * 60_000_000_000,
    open=np.array([100.0, 100.0, 100.5, 101.5, 103.0, 104.0]),
    high=np.array([101.0, 101.0, 103.0, 103.0, 104.0, 105.0]),
    low=np.array([99.0, 99.0, 100.0, 101.0, 102.0, 103.0]),
    close=np.array([100.0, 100.0, 102.0, 102.0, 103.0, 104.0]),
    volume=np.full(6, 1e6),
)


def fill_price(order, timing="same_bar_close"):
    class Once(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 1:
                self.submit_order(order)

    r = run_strategy_backtest(Once, **tape, bar_stamp="close",
                              config=BacktestConfig(fees=0.0, fill_timing=timing))
    return r.trades()[0].entry_price


assert fill_price(orders.Limit(side="buy", price=200.0, quantity=10)) == 100.0           # bar 1 close, not 200
assert fill_price(orders.Limit(side="buy", price=200.0, quantity=10), "next_bar_open") == 100.5   # the open, not 200
assert fill_price(orders.Market(side="buy", quantity=10, tif="at_open")) == 100.5            # bar 2 open
assert fill_price(orders.Market(side="buy", quantity=10, tif="at_close")) == 102.0           # bar 2 close
```

A short option structure, placed hedge-first the same way in a backtest, on
`PaperExecution` and on a broker. All checks run before any leg is sent. The
bought legs go out at once; each sold leg goes out when the hedges covering it
have filled, for the quantity they cover. A sold leg that is refused closes
the hedges that filled, with orders of their own (`group.failure` says which
order failed and why). The engine drives this from the order events: nothing
needs calling for it (`update_structures` is for exits). A sold leg needs a
margin model that groups option lines to be funded with its hedge held:

```python
import numpy as np
from raptorbt import BacktestConfig, InstrumentSpec, MarginModel, OptionChain, Strategy, run_strategies
from raptorbt import structures as st

STEP, T0, N, LOT = 60 * 10**9, 1_767_225_600 * 10**9, 20, 50.0
EXPIRY = T0 + 60 * STEP


def series(level):
    close = np.full(N, float(level))
    return dict(timestamps=T0 + STEP * np.arange(1, N + 1, dtype=np.int64), open=close,
                high=close + 0.05, low=close - 0.05, close=close, volume=np.full(N, 1000.0))


data, specs = {"IDX": series(100.0)}, []
for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
    for right in ("call", "put"):
        symbol = f"IDX{int(strike)}{right[0].upper()}E"
        specs.append(InstrumentSpec.option(symbol, strike, right, EXPIRY, LOT, underlying="IDX",
                                           span_pct=0.1, exposure_pct=0.02))
        data[symbol] = series(max(0.5, 3.0 - abs(strike - 100.0) / 5.0))
chain = OptionChain(specs)


class Condor(Strategy):
    def __init__(self):
        super().__init__()
        self.group, self.marks = None, {}

    def on_bar(self, ctx):
        self.marks[ctx.symbol] = ctx.bar.close
        self.update_structures(ctx, self.marks)
        if ctx.symbol == "IDX" and self.group is None and ctx.idx == 3:
            self.group = self.submit_structure(
                ctx, st.iron_condor(inner_steps=1, outer_steps=2), chain=chain,
                underlying="IDX", spot=ctx.bar.close, lots=1)


s = Condor()
res = run_strategies([s], data,
                     BacktestConfig(initial_capital=2_000_000.0, fees=0.0, margin_model=MarginModel.defined_risk()),
                     instruments={sp.symbol: sp for sp in specs}, account_type="margin",
                     bar_stamp="close")
assert s.structure_status(s.group) == "filled"
sides = {t.symbol: t.direction for t in res[0].result.trades()}
assert sides == {"IDX90PE": 1, "IDX95PE": -1, "IDX105CE": -1, "IDX110CE": 1}   # wings long, body short
```

**A stop on an option leg: the premium or the index.** Decide first what the
stop protects. A stop on the **premium** caps the money lost on the contract
and is an engine level, so it fills at the level (or the open after a gap);
it also trips on time decay and a volatility drop with the index unchanged.
A stop on the **index** says the thesis is wrong, but the premium loss at that
index level depends on delta and time left, so the loss is not fixed, and the
strategy must close the contract itself because the engine only watches the
contract's own price.

- Premium stop: put `stop_price` (and `target_price`) on the opening order,
  computed from the **contract's** bars, for example
  `raptorbt.atr(b["high"], b["low"], b["close"], 14)[-1]` over
  `ctx.series(contract_symbol)`. An ATR of the index is in index points and is
  the wrong size for a premium.
- Index stop: take the index ATR (`ctx.indicators.atr(instrument="NIFTY")`),
  remember the index level, and on a later index bar call
  `close_position(symbol=contract_symbol)`.
- `StopConfig` / `set_atr_stop` on `BacktestConfig` is a run-level setting for
  every position of the run, measured on the traded contract's own ATR at
  entry. Use it when you own the config (a research script); a strategy that
  must carry its stop to a host that builds the config places the stop on the
  order instead, as below.

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, InstrumentSpec, Strategy, pricing, run_strategies
from raptorbt.strategy import orders

MIN, DAY = 60 * 10**9, 86_400 * 10**9
T0 = 1_767_225_600 * 10**9
N, EXPIRY = 60, 1_767_225_600 * 10**9 + 7 * DAY
STAMPS = T0 + MIN * np.arange(1, N + 1, dtype=np.int64)


def bars(close, wick):
    close = np.asarray(close, dtype=float)
    return dict(timestamps=STAMPS, open=close, high=close + wick, low=close - wick,
                close=close, volume=np.full(N, 1_000.0))


# the index drifts down from bar 20, so a long call loses; its premium follows the index
index = 24_000.0 + np.cumsum(np.where(np.arange(N) < 20, 1.0, -4.0)) + np.tile([0.0, 3.0], N // 2)
premium = np.maximum(pricing.price("call", index, 24_000.0, (EXPIRY - STAMPS) / (365 * DAY), 0.14, rate=0.065), 0.05)
data = {"IDX": bars(index, 2.0), "IDX-C": bars(premium, 0.4)}
call = InstrumentSpec.option("IDX-C", 24_000.0, "call", EXPIRY, 50.0, underlying="IDX")


class LongCall(Strategy):
    def __init__(self, stop_on):
        super().__init__()
        self.stop_on, self.entered, self.index_stop = stop_on, False, None

    def on_start(self, ctx):
        self.index_atr = ctx.indicators.atr(instrument="IDX", period=14)

    def on_bar(self, ctx, event=None):
        if ctx.symbol != "IDX":
            return
        if not self.entered and ctx.idx == 20:
            self.entered = True
            if self.stop_on == "premium":
                # the contract's own ATR, from its own bars: the stop is a level on the premium
                b = ctx.series("IDX-C")
                atr = raptorbt.atr(b["high"], b["low"], b["close"], 14)[-1]
                px = float(b["close"][-1])
                self.submit_order(orders.Market(side="buy", quantity=50.0, stop_price=px - 2 * atr,
                                                target_price=px + 4 * atr), symbol="IDX-C")
            else:
                # the thesis is about the index: remember its level, stop when the index breaks it
                self.index_stop = ctx.bar.close - 2 * self.index_atr.value
                self.submit_order(orders.Market(side="buy", quantity=50.0), symbol="IDX-C")
        elif self.index_stop is not None and ctx.bar.close <= self.index_stop:
            self.close_position(symbol="IDX-C")
            self.index_stop = None


def run(stop_on):
    (r,) = run_strategies([LongCall(stop_on)], data, BacktestConfig(initial_capital=1_000_000.0, fees=0.0),
                          instruments={"IDX-C": call}, bar_stamp="close")
    return r.result.trades()[0]


assert run("premium").exit_reason == "StopLoss"   # the engine held the premium level
assert run("index").exit_reason == "Signal"       # the strategy closed the contract itself
```

Moving a stop with an exit rule. The first rule risks 15 of P&L; once the close
reaches 103 the strategy cancels it and sets a stop at the price 102, above the
entry. The rule fires on the bar that closes at 102 and closes the position with
a reduce-only market order:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategies
from raptorbt.strategy import orders
from raptorbt.strategy.exits import ExitEvents, ExitRule

MIN = 60 * 10**9
close = np.array([100, 100, 101, 102, 103, 102.5, 102, 101.5, 101, 100.5], dtype=float)
bars = {"AAA": dict(timestamps=np.arange(1, 11, dtype=np.int64) * MIN, open=close, high=close + 0.1,
                    low=close - 0.1, close=close, volume=np.full(10, 1000.0))}


class Raise(ExitEvents, Strategy):
    def on_start(self, ctx):
        self.handles, self.n = [], 0

    def on_bar(self, ctx):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", quantity=10))
        elif ctx.positions("AAA") and not self.handles:
            self.handles.append(self.set_exit(ctx, ExitRule(stop_loss=15.0), symbol="AAA"))
        elif ctx.bar.close >= 103 and self.handles[-1].state == "armed":
            self.cancel_exit(self.handles[-1])          # a rule has no modify: cancel it, set another
            self.handles.append(self.set_exit(ctx, ExitRule(premium_at_or_below=102.0 * 10), symbol="AAA"))


s = Raise()
(r,) = run_strategies([s], bars, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
first, moved = s.handles
assert (first.state, moved.state, moved.reason) == ("canceled", "done", "premium_at_or_below")
assert moved.close_ids == [f"{moved.id}:x0"]
(trade,) = r.result.trades()
assert (trade.entry_price, trade.exit_price, trade.pnl) == (100.0, 102.0, 20.0)
```

## 6. Composition

- A strategy sends orders; the account they trade against, with netting,
  hedging and margin, is [portfolio](../portfolio/SKILL.md).
- Limits that refuse an order (size, exposure, daily loss) and halts:
  [risk](../risk/SKILL.md).
- Fill timing, slippage and costs applied to the fill:
  [backtesting](../backtesting/SKILL.md).
- Option chains and structure definitions: [instruments](../instruments/SKILL.md).
- A live venue answers with execution reports instead of the simulated
  matcher: [execution](../execution/SKILL.md).
- Runnable examples: `examples/single-instrument/sma_cross.py`, `examples/tick-level/passive_quotes.py`,
  `examples/tick-level/tick_scalper.py` (a fixed stop on each entry and the configuration's
  trailing stop, on prints), `examples/tick-level/dynamic_exit.py` (exit rules moved while
  the position is open, batch and live).

## 7. Constraints and correctness

- **Netting vs hedging.** The default `oms_type="netting"` keeps one net
  position per instrument and moves it as a broker's does:
  - an order **on the side held adds**: the size adds and the entry price
    becomes the size-weighted average. Each fill pays its own fees, and each
    is a record of its own (the order log, the journal). A capital-fraction
    order adds too. It is gated like an entry (halts, the kill switch, daily
    loss, margin, size and position limits), but `max_positions` does not
    count it, since the position already counts;
  - an order **against the position reduces it**. The reduction is booked at
    the average entry price and is one trade (`size`, `entry_price` = the
    average, `pnl`, fees: the entry fees of the slice and the exit fees); the
    rest stays open at the same average. Indian brokers' net positions use the
    average price too;
  - an order **larger than the position flips it**: it closes what is held (a
    trade) and opens the rest on its own side, as one order and one fill
    price. The close and the open are two fills and are charged separately.
    A `reduce_only` order, an order that names a position, and an order sized
    as a capital fraction or as the whole position never flip: they stop at
    flat. With nothing open an order opens in its own side's direction.

  The levels of a net position are one pair. A level derived from the entry
  (a percent or ATR stop) follows the new average. A `stop_price` or
  `target_price` the added order names becomes the position's level, so the
  whole net position is protected at it. A bracket's legs close the full
  position when they trigger, so a bracket made with an added slice protects
  the net position, earlier slices included, and the earlier bracket's legs do
  too: the first leg to trigger closes it and the others are refused
  (`reduce_only`, nothing left). A leg with an explicit quantity closes at most
  that quantity (and never more than is held).

  `enter()` and `close_position()` are signals, "be in this position": with a
  position held `enter()` is a no-op in every mode (a simulated venue refuses to
  fill it as an add). Typed orders are how a strategy adds.

  A host's `order_check` sees typed orders only. With one installed, `enter()` and
  `close_position()` are refused (`order_check_intent`): they carry no order to show it.
  `order_check_scope="typed"` (on `run_strategies` and `LiveNode`) declares that the check
  judges typed orders only, and intents then pass without being shown to it. Use it when the
  check judges forms an intent cannot have (a kind, a validity, `post_only`); it does not make
  the check see intents.

  With `oms_type="hedging"` every typed order opens its own position in its own
  direction, so a buy and a sell coexist; close one with
  `close_position(position_id)`. Batch runs, a `LiveNode` and a paper or
  adapter venue all follow these rules.
- `reduce_only` orders can only reduce an open position; they are rejected
  with `reduce_only` when there is nothing to reduce. Bracket legs use it so a
  leg still working after the position closed can never open the other side.
- A `Twap` on a netting account builds one position: each slice adds to it.
  Cancelling a schedule stops unreleased slices but does not undo traded ones.
- **Stops and gaps.** A stop that the market gaps through fills at the open,
  not at the trigger. Within a bar where a stop and a target are both touched,
  the stop is taken first unless `bar_path_adaptive` is on.
- Cancelling or modifying an id that does not exist, or has finished, is
  silently ignored: no error, no event. Check `self.cache.is_order_open(id)` if
  you need to know.
- `size_frac` is resolved against *available* capital at fill time, so two
  orders queued on one bar size one after the other: the second gets its
  fraction of what the first left.
- Explicit quantities must be whole lots and size steps of the instrument
  (`off_lot_quantity` otherwise; round them yourself, or size with
  `raptorbt.sizing`, which returns whole lots), and are refused, not shrunk,
  if they exceed available funds. Capital-fraction sizes are floored to the lot
  and size step; one that rounds to zero is rejected as `zero_size`.
- Order ids are deterministic (`"{order_id_tag}-{n}"`, counted per strategy), so
  two identical runs produce identical ids.
- **What an exit rule costs.** On every event `update_exits` visits every handle
  the strategy has set; a cancelled or finished handle is skipped by its state but
  never removed. For each armed rule it reads the positions of each symbol the rule
  covers and that symbol's last price. On a live node and in a windowed run the last
  price (the last close, else the last `ltp > 0`) comes from the engine without
  building `ctx.series`; in a batch run it is read from `ctx.series`, a view of the
  arrays (a copy for a symbol whose built bars join its bars). No per-rule timing
  is published. On ticks, where this runs on every print: one rule per position,
  or one group rule (`symbols=`, `name=`) for legs that exit together, rather than
  a rule per leg; and move a rule on a step (a price moved by a set amount), not on
  every print, since every move leaves a cancelled handle behind.
- **Percents are percents here.** `ExitRule(stop_loss_pct=30)` is 30 percent of
  cost, unlike `StopConfig`'s fractions. Nothing fires until every symbol of the
  rule has a position and a last price; `premium` leaves out the instrument's
  multiplier, `pnl` and `cost` include it.

## 8. Common mistakes

Giving two sizing forms:

<!-- skill: invalid -->
```python
from raptorbt.strategy import orders

orders.Market(side="buy", quantity=10, size_frac=0.5)   # InvalidOrder: conflicting_order_args
```

A non-positive price:

<!-- skill: invalid -->
```python
from raptorbt.strategy import orders

orders.Limit(side="buy", price=0.0, quantity=1)         # InvalidOrder: invalid_order_value
```

- Treating `submit_order` as a fill: it returns an id, nothing more.
- Using `at_open` or `at_close` on a limit or stop: refused with
  `incompatible_order_options`; they are for market orders.
- Expecting a stop-limit to fill on the bar it triggers. It triggers, then
  rests as a limit from the next bar.
- `tif="gtd"` without `expire_ns`: refused with `missing_order_field`.
- Passing `size_frac` to a `Twap`: only an explicit `quantity` can be sliced.
- Using `submit_bracket` under hedging: it is refused with
  `bracket_under_hedging`, because its `reduce_only` legs could never close a
  position that hedging opens on its own. Attach `stop_price` and
  `target_price` to each entry instead.
- Cancelling or modifying a client id the run never submitted: refused with
  `unknown_order`. One that was submitted and has finished is left alone, with
  no error and no event.
- Assuming a limit buy at a price above the market fills at that price. It
  fills at the market, which is better for a buyer.
- Setting a new `ExitRule` to move a stop without cancelling the old one: both
  stay armed and the first to hold fires.
- Leaving a rule armed after its position was closed some other way: it waits,
  and covers the symbol's next position at that position's own entry price.
  Cancel it when you close.
- Sizing from indicators in a fill hook under `fill_timing="next_bar_open"`: an order you queue from a fill hook (`on_fill`, `on_position_event`) answers a fill at a bar's open, but an indicator or object you own and update yourself already includes that bar's close. Size from values you stored on an earlier bar (for example in `on_bar`), not from indicators read in the hook.

## 9. Validation

1. Reproduce the order on a tape of five to ten bars and assert the event
   sequence and the fill bar and price, as the examples do.
2. Assert that every order you send ends in one of `filled`, `canceled`,
   `expired` or `rejected`; list `self.cache.orders_open()` in `on_stop` and
   expect it to be empty or deliberate.
3. Run once with real fees and `fill_timing="next_bar_open"` and compare.
4. For brackets and OCO, test both outcomes: the target filling and the stop
   filling.
5. Before live use, run the same order flow through a paper venue
   ([execution](../execution/SKILL.md) and [live-trading](../live-trading/SKILL.md)).

## 10. Related skills

- [Strategy](../strategy/SKILL.md)
- [Portfolio](../portfolio/SKILL.md)
- [Risk](../risk/SKILL.md)
- [Backtesting](../backtesting/SKILL.md)
- [Instruments](../instruments/SKILL.md)
