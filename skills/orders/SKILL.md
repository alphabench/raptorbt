---
name: raptorbt-orders
description: >
    Order construction and lifecycle: the typed order kinds, sizing forms,
    time in force, brackets, OCO links, modify and cancel, option structures,
    and the reasons an order is refused.
version: 1.0
api_version: 2.x
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

**Time in force.** `gtc` (default; rests until filled or cancelled), `day`
(expires at the end of the trading day in `session_tz_offset_ns`), `gtd` (needs
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

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.strategy.orders.{Market, Limit, StopMarket, StopLimit, MarketIfTouched, LimitIfTouched, MarketToLimit, TrailingStopMarket, TrailingStopLimit, Twap}` | The typed orders above. Construction validates side, tif, prices and sizing and raises `InvalidOrder` with a `code`. |
| `Strategy.submit_order(order, client_id=None, parent=None, symbol=None)` | Queue an order; returns its client id. `parent=<id>` holds this order until the parent fills and kills it if the parent dies. |
| `Strategy.submit_bracket(entry, stop_trigger, target_price, stop_limit_price=None)` | An entry plus a `reduce_only` stop and target that activate when it fills, linked so one cancels the other. Returns `(entry_id, stop_id, target_id)`. Netting only: under hedging it raises `bracket_under_hedging`. |
| `Strategy.link_oco(*client_ids)` | One-cancels-other among two or more submitted orders. |
| `Strategy.cancel_order(client_id)`, `cancel_all_orders()` | Cancel one working order (or a `Twap`'s unreleased slices), or all of them. A finished order is a no-op; an id never submitted raises `unknown_order`. |
| `Strategy.modify_order(client_id, units=None, size_frac=None, limit_price=None, trigger_price=None)` | Replace price or size of a working order. `units` is the quantity. |
| `Strategy.submit_structure(ctx, structure, *, chain, underlying, spot, lots=1, ...)`, `update_structures(ctx, marks)`, `close_structure(group)`, `structure_status(group)` | A multi-leg option position as one group of orders; see `docs/structures.md`. |
| `Strategy.cache` | `order(id)`, `orders()`, `orders_open()`, `is_order_open(id)`, `realized_pnl()`: order states folded from events. |
| `raptorbt.MarketOrder`, `raptorbt.ClosePosition` | The legacy intents behind `enter()` and `close_position()`. Prefer the methods. |
| `raptorbt.StopConfig`, `raptorbt.TargetConfig` | Fixed, ATR, trailing and risk-reward stop and target models for `BacktestConfig` (`set_fixed_stop(0.02)` is 2 percent; fractions, not percents). |

Events carry `kind`, `idx`, `client_order_id`, `price`, `size` and
`reject_reason`. Order-level `reject_reason` values: `invalid_qty`,
`insufficient_capital`, `reduce_only`, `post_only`, `position_open`, `zero_size`,
`insufficient_margin`, `margin_unpriced`, `expired`, `inactive`,
`market_closed`, `squareoff`, plus the risk reasons (`order_size`,
`max_position`, `max_exposure`, `max_positions`, `daily_loss`, `killed`,
`drawdown_halt`, `margin_call`; see [risk](../risk/SKILL.md)).

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
assert ("order_filled", 1) in s.kinds                     # the modified limit, now marketable
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

The marketable-limit rule: a limit that is already marketable fills at the
market, never at its own limit price, and under either fill timing that market
is the next bar's open. `at_open` and `at_close` market orders fill at the next
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


for timing in ("same_bar_close", "next_bar_open"):
    assert fill_price(orders.Limit(side="buy", price=200.0, quantity=10), timing) == 100.5   # the open, not 200
assert fill_price(orders.Market(side="buy", quantity=10, tif="at_open")) == 100.5            # bar 2 open
assert fill_price(orders.Market(side="buy", quantity=10, tif="at_close")) == 102.0           # bar 2 close
```

A short option structure as one group of orders. All checks run before any
leg is sent; legs fill independently and `update_structures` must be called
every bar:

```python
import numpy as np
from raptorbt import BacktestConfig, InstrumentSpec, OptionChain, Strategy, run_strategies
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
res = run_strategies([s], data, BacktestConfig(initial_capital=2_000_000.0, fees=0.0),
                     instruments={sp.symbol: sp for sp in specs}, account_type="margin",
                     bar_stamp="close")
assert s.structure_status(s.group) == "filled"
sides = {t.symbol: t.direction for t in res[0].result.trades()}
assert sides == {"IDX90PE": 1, "IDX95PE": -1, "IDX105CE": -1, "IDX110CE": 1}   # wings long, body short
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

## 7. Constraints and correctness

- **Netting vs hedging.** The default `oms_type="netting"` keeps one position
  per instrument: a sell against a long reduces it and does not open a short,
  and a sell larger than the position closes it and stops (the fill is capped at
  the open size). A sell with nothing open opens a short. An order in the
  direction of the open position is rejected as `position_open`; a netting
  account does not add to a position through typed orders. With
  `oms_type="hedging"` every typed order opens its own position in its own
  direction, so a buy and a sell coexist; close one with
  `close_position(position_id)`.
- `reduce_only` orders can only reduce an open position; they are rejected
  with `reduce_only` when there is nothing to reduce. Bracket legs use it so a
  leg still working after the position closed can never open the other side.
- A `Twap` on a netting account rejects every slice after the first as
  `position_open`; use hedging for a schedule that builds a position.
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
- Explicit quantities are floored to the instrument's lot and size step, then
  refused, not shrunk, if they exceed available funds. Capital-fraction orders
  that round to zero are rejected as `zero_size`.
- Order ids are deterministic (`"{order_id_tag}-{n}"`, counted per strategy), so
  two identical runs produce identical ids.

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
