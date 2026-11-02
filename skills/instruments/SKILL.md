---
name: raptorbt-instruments
description: >
    Instrument definitions, asset classes, options and multi-instrument runs.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
status: stable
---

# RaptorBT Instruments

## 1. When to use this skill

Load it when a run needs more than a bare price series: a futures contract
with a lot size and an expiry, an option, a market that is only open at some
hours, or a chain from which a strategy picks contracts at the time it decides.
Skip it for a single stock traded as plain arrays; the core skill is enough.

## 2. Domain overview

An `InstrumentSpec` is the market definition of one tradable symbol: its kind
(`cash`, `contract`, `option`, `currency_pair`, `index`), lot size, price
increment, contract multiplier, expiry, margin and fee fields, and an optional
`SessionSchedule`. You pass specs to a runner (`instrument=` for one symbol,
`instruments={symbol: spec}` for many). With no spec a symbol trades at any
timestamp, one unit at a time, and never expires. A spec turns on, per symbol:

- **Lot floor.** Order size is rounded down to a whole number of lots.
- **Multiplier.** Notional is `price * units * multiplier`.
- **Tick grid.** Stops and targets derived from a percentage land on the
  `price_increment` grid.
- **Expiry.** An order after `expiration_ns` is rejected (`expired`); an open
  position is settled on the first bar at or after expiry (exit reason
  `Settlement`). Before `activation_ns` entries are rejected.
- **Session.** With a `SessionSchedule`, orders outside the windows are
  rejected (`market_closed`) and nothing matches outside them.

An `OptionChain` is a lookup over option specs. **Every query takes the event
time `now_ns`** and sees only contracts listed at that time: a contract is
visible from its `activation_ns` up to, not including, its `expiration_ns`.
There is no call that returns the unfiltered chain, so a chain query cannot
look ahead. `raptorbt.pricing` prices European options (Black-Scholes-Merton
and Black-76), and `raptorbt.structures` builds multi-leg positions from a
chain.

Event time: use the engine's clock, `ctx.bar_close_ts`, as `now_ns`. It is the
instant the bar you are looking at ended. `ctx.timestamp` is your own stamp,
and is the bar's open when the run says `bar_stamp="open"`.

## 3. API reference

| Name | What it is |
| --- | --- |
| `InstrumentSpec.equity(symbol, price_increment=, lot_size=, ...)` | A cash instrument (kind `cash`). |
| `InstrumentSpec.futures_contract(symbol, expiration_ns, lot_size, multiplier=, underlying=, activation_ns=, ...)` | A dated future (kind `contract`). |
| `InstrumentSpec.perpetual(symbol, ...)` | A future with no expiry (kind `contract`). |
| `InstrumentSpec.option(symbol, strike, right, expiration_ns, lot_size, underlying=, ...)` | A European option. `right` is `"call"` or `"put"` (`"CE"` and `"PE"` also work). `underlying` is required by `OptionChain`. |
| `InstrumentSpec.currency_pair(symbol, ...)` | A currency pair. |
| `InstrumentSpec.index(symbol, ...)` | A reference series. It is **not tradable**: ordering it raises. |
| `SessionSchedule(windows, tz_offset_ns=, holidays=, squareoff_minute=)` | Weekday windows in local minutes from midnight, half-open (`open <= t < close`), plus holiday dates as `yyyymmdd` ints. |
| `SessionSchedule.nse_equity()`, `.nse_fo()`, `.bse_equity()` | Presets: Monday to Friday, 09:15 to 15:30 IST (minutes 555 to 930). They carry **no holiday calendar**; pass the exchange's dates in `holidays=`. |
| `schedule.is_open(ts_ns)`, `.session_date(ts_ns)`, `.next_open_ns(ts_ns)`, `.session_close_ns(date)` | Ask the schedule. |
| `OptionChain(specs)` | Build once from every contract and expiry you know. |
| `chain.expiries / strikes / contract / contracts` | List what is visible at `now_ns`. |
| `chain.pick_expiry(underlying, now_ns, which, n=0, min_days=0)` | `which` is `"nearest"`, `"nth"`, `"weekly"`, `"monthly"` or `"date"`. |
| `chain.atm_strike / strike_at_offset / by_delta / closest_by` | Pick a strike or a contract. Ties go to the lower strike. |
| `raptorbt.pricing.price / greeks / implied_vol / implied_vol_status / year_fraction` | Pricing, on scalars or NumPy arrays. |
| `raptorbt.structures.*` | `iron_condor`, `straddle`, `Structure`, `Leg`, `Steps`, `Delta`, `Exit` and the rest. `docs/structures.md` in the package documents `submit_structure`. |
| `raptorbt.InstrumentConfig(lot_size=, alloted_capital=, existing_qty=, avg_price=)` | Per-symbol settings for the array runners. A spec is the richer form. |

Pricing units: `tau` in years (`pricing.year_fraction(now_ns, expiry_ns)`, a
365-day year), `vol`, `rate` and `dividend_yield` as annual decimals, `vega`
per 1.00 of volatility (divide by 100 for a point), `theta` per year (divide
by 365 for a day).

## 4. Standard workflow

1. Write one `InstrumentSpec` per tradable symbol. Give futures and options an
   `expiration_ns` and a `lot_size`, and options an `underlying`.
2. Attach a `SessionSchedule` if the market has hours, with its holidays.
3. Pass the specs: `instrument=spec` for one symbol, `instruments={...}` for
   a multi-symbol run.
4. For options, build the `OptionChain` from the **whole** expiry calendar,
   not only what is listed now. "Monthly" is the last expiry of its month in
   that calendar, so a partial calendar changes the answer.
5. In `on_bar`, query the chain with `now_ns=ctx.bar_close_ts`, price with
   `pricing`, and place orders on the contracts you got back.
6. If an option position can run to expiry, call `ctx.set_underlying_price(...)`
   each bar so settlement uses intrinsic value. Without it the contract settles
   at its own last close.

## 5. Examples

A future with a lot size: half the capital buys 5,000 units, which is 100 lots
of 50. Order size is floored to whole lots.

```python
import numpy as np
from raptorbt import BacktestConfig, InstrumentSpec, Strategy, run_strategy_backtest

DAY = 86_400 * 10**9
MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9  # 2026-01-01 00:00 UTC

n = 20
close = 100.0 + np.arange(n) * 0.1
bars = dict(
    timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
    open=close, high=close + 0.2, low=close - 0.2, close=close,
    volume=np.full(n, 1000.0),
)

class EnterOnce(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.enter(size_frac=0.5)

fut = InstrumentSpec.futures_contract(
    "NIFTY-FUT", expiration_ns=T0 + 30 * DAY, lot_size=50.0, underlying="NIFTY"
)
result = run_strategy_backtest(
    EnterOnce, **bars, symbol="NIFTY-FUT", instrument=fut,
    bar_stamp="close", config=BacktestConfig(initial_capital=1_000_000.0, fees=0.0),
)
trade = result.trades()[0]
assert trade.size == 5000.0  # 500_000 / 100 = 5000 units
assert trade.size % fut.lot_size == 0
```

A session schedule rejects an order while the market is closed. The bar at
index 1 closes at 09:11 IST, before the 09:15 open; the bar at index 12 closes
at 09:22:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, InstrumentSpec, SessionSchedule, Strategy, run_strategy_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9  # 2026-01-01 00:00 UTC, a Thursday
ist_midnight = T0 - raptorbt.IST_OFFSET_NS

n = 40
ts = ist_midnight + (9 * 60 + 10) * MIN + np.arange(n, dtype=np.int64) * MIN  # closes 09:10, 09:11, ...
close = 100.0 + np.arange(n) * 0.1
bars = dict(timestamps=ts, open=close, high=close + 0.2, low=close - 0.2,
            close=close, volume=np.full(n, 1000.0))

class TryEarlyAndLate(Strategy):
    def on_start(self, ctx):
        self.rejected = []

    def on_bar(self, ctx):
        if ctx.idx in (1, 12):
            self.enter(size_frac=0.1)
        if ctx.idx == 30 and not ctx.is_flat:
            self.close_position()

    def on_order_rejected(self, ctx, event):
        self.rejected.append((ctx.idx, event.reject_reason))

spec = InstrumentSpec.equity("AAPL", session=SessionSchedule.nse_equity())
strategy = TryEarlyAndLate()
result = run_strategy_backtest(
    strategy, **bars, symbol="AAPL", instrument=spec,
    bar_stamp="close", config=BacktestConfig(fees=0.0),
)
assert strategy.rejected == [(1, "market_closed")]
assert result.trades()[0].entry_idx == 12
```

Picking options from a chain, pricing them and resolving a structure. The
chain holds two expiries; every call states the event time:

```python
import numpy as np
from raptorbt import InstrumentSpec, OptionChain
from raptorbt import pricing, structures as st

DAY = 86_400 * 10**9
T0 = 1_767_225_600 * 10**9
exp_near, exp_far = T0 + 7 * DAY, T0 + 35 * DAY

specs = [
    InstrumentSpec.option(
        f"NIFTY-{int(k)}-{right[0].upper()}-{i}", float(k), right, expiry, 50.0,
        underlying="NIFTY",
    )
    for i, expiry in enumerate((exp_near, exp_far))
    for k in np.arange(23_500, 24_600, 100)
    for right in ("call", "put")
]
chain = OptionChain(specs)
now = T0                      # in a strategy: ctx.bar_close_ts
spot = 24_040.0

expiry = chain.pick_expiry("NIFTY", now, "nearest")
assert expiry == exp_near
assert chain.atm_strike("NIFTY", expiry, now, spot) == 24_000.0
assert chain.atm_strike("NIFTY", expiry, now, 24_050.0) == 24_000.0  # a tie goes down
assert chain.contract("NIFTY", expiry, 24_050.0, "call", now) is None  # not listed
wing = chain.by_delta("NIFTY", expiry, now, "put", -0.15, spot, vol=0.14, rate=0.065)
assert wing.strike < spot and wing.right == "put"

tau = pricing.year_fraction(now, expiry)
g = pricing.greeks("call", spot, 24_000.0, tau, 0.14, rate=0.065)
assert 0.5 < g.delta < 0.6 and g.theta < 0.0
iv = pricing.implied_vol(g.price, "call", spot, 24_000.0, tau, rate=0.065)
assert abs(iv - 0.14) < 1e-9

condor = st.iron_condor(inner_steps=2, outer_steps=4).resolve(
    chain, "NIFTY", spot, now_ns=now, lots=2
)
assert [(leg.side, leg.lots, leg.quantity) for leg in condor.legs] == [
    ("sell", 2, 100.0), ("sell", 2, 100.0), ("buy", 2, 100.0), ("buy", 2, 100.0),
]
```

Option settlement needs the underlying price. The same long call settles at
intrinsic value (108 - 100 = 8) when the strategy supplies it, and at its own
close (5) when it does not:

```python
import numpy as np
from raptorbt import BacktestConfig, InstrumentSpec, Strategy, run_strategy_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 10
close = np.full(n, 5.0)
bars = dict(timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
            open=close, high=close + 0.1, low=close - 0.1, close=close,
            volume=np.full(n, 1000.0))
call = InstrumentSpec.option("U-100-C", 100.0, "call", T0 + 6 * MIN, 10.0, underlying="U")

class Hold(Strategy):
    underlying = None

    def on_bar(self, ctx):
        if self.underlying is not None:
            ctx.set_underlying_price(self.underlying)
        if ctx.idx == 0:
            self.enter(size_frac=0.5)

def settle(underlying):
    strategy = Hold()
    strategy.underlying = underlying
    r = run_strategy_backtest(strategy, **bars, symbol="U-100-C", instrument=call,
                              bar_stamp="close", config=BacktestConfig(fees=0.0))
    t = r.trades()[0]
    return t.exit_reason, t.exit_price

assert settle(108.0) == ("Settlement", 8.0)
assert settle(None) == ("Settlement", 5.0)
```

## 6. Composition

- Bars for each symbol, and what a strategy may read of them, are in
  [market-data](../market-data/SKILL.md). Several symbols in one run need the
  portfolio runner.
- Placing the orders a structure resolves to, and group exits, are in
  [strategy](../strategy/SKILL.md) and [orders](../orders/SKILL.md).
- Margin for short options and spreads is [risk](../risk/SKILL.md); fees by
  segment are [execution](../execution/SKILL.md).
- Daily and longer bars follow a symbol's schedule; see
  [timeframes](../timeframes/SKILL.md).

## 7. Constraints and correctness

- **No lookahead.** A chain only shows contracts listed at `now_ns`, and
  `pick_expiry("nth", n=0)` is the first expiry listed *then*. Never pass a
  time later than the event you are handling.
- **A contract is gone at its expiry.** `chain.expiries(u, t)` omits an expiry
  once `t >= expiration_ns`.
- **Ties go to the lower strike**, in `atm_strike`, `strike_at_offset`,
  `by_delta` and `closest_by`. Lists come back ascending.
- **`by_delta` uses one flat volatility** for every contract (the `vol` you
  pass), not a smile. It picks the nearest delta under that assumption.
- **Spot versus forward.** `black_scholes` takes spot and carries it forward
  with `rate - dividend_yield`. `black76` takes a futures price and requires
  `dividend_yield == 0`. A futures price fed to `black_scholes` prices a
  different forward.
- **Implied volatility never guesses.** A price below intrinsic value, or one
  that needs a volatility outside `[0.0001, 5]`, raises `ImpliedVolError` (a
  scalar call) or returns `NaN` (array call). `implied_vol_status` returns
  `(iv, status)` and says why. Deep out-of-the-money prices carry little
  information about volatility; one tick can move the result a lot.
- **`NaN` means absent.** Array pricing marks an invalid element `NaN`, never
  zero. Check with `np.isnan` before using a result.
- **Schedules hold no holidays unless you give them.** The NSE presets know
  weekdays and hours only. A date in `holidays` is skipped by the schedule,
  by daily bars and by the square-off.
- **Index specs cannot be traded.** Use one as a reference and trade the
  future or option on it.
- **Constructor errors.** `OptionChain` refuses an option with no
  `underlying`. `InstrumentSpec` factories raise `InvalidParameter` naming the
  parameter: `invalid_option_right` for a `right` that is not call or put,
  `invalid_parameter` for a bad strike, multiplier, increment or an expiry
  before activation. `OptionChain.pick_expiry` raises `invalid_expiry_pick`
  for an unknown rule. `SessionSchedule` raises `InvalidParameter`
  (`invalid_session_schedule`). Each carries `.skill == "instruments"`.

## 8. Common mistakes

A `right` other than call or put is refused:

<!-- skill: invalid -->
```python
from raptorbt import InstrumentSpec

InstrumentSpec.option("X", 100.0, "straddle", 10**18, 50.0)  # InvalidParameter: invalid_option_right
```

A price below intrinsic value has no implied volatility; the call raises
instead of returning a number:

<!-- skill: invalid -->
```python
from raptorbt import pricing

# A call with spot 24040 and strike 24000 is worth at least 40; 1.0 is not.
pricing.implied_vol(1.0, "call", 24_040.0, 24_000.0, 7 / 365)  # ImpliedVolError
```

- Building the chain from today's listed contracts only. `monthly` then
  means "the last expiry I know of", which moves as contracts list.
- Passing `ctx.timestamp` as `now_ns` in a run stamped by bar open. Use
  `ctx.bar_close_ts`.
- Treating `lot_size` as your own quantity multiplier. The engine already
  floors orders to whole lots; compute lots, then multiply.
- Reading `vega` as per volatility point. It is per 1.00; divide by 100.
- Forgetting `underlying=` on an option spec.
- Expecting `nse_equity()` to skip exchange holidays.

## 9. Validation

1. Assert the contracts a structure resolved to (`[(leg.instrument, leg.side,
   leg.lots) for leg in resolved.legs]`) before sending orders.
2. Put-call parity: for one strike and expiry, `call - put` equals
   `spot * exp(-q * tau) - strike * exp(-rate * tau)` from `pricing.price`,
   to within 1e-9.
3. Round trip: `implied_vol(price(vol))` returns `vol`.
4. Query with a `now_ns` before a contract's `activation_ns` and confirm the
   chain does not list it.
5. For a session run, assert the reject reasons (`market_closed`, `expired`)
   you expect.
6. Settle an option both ways (with and without `set_underlying_price`) and
   check the exit price is the one you intend.

## 10. Related skills

- [Core](../core/SKILL.md)
- [Market data](../market-data/SKILL.md)
- [Timeframes](../timeframes/SKILL.md)
- [Risk](../risk/SKILL.md)
