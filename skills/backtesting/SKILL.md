---
name: raptorbt-backtesting
description: >
    Historical simulation: choosing a runner, configuring fills, fees,
    slippage and latency, expiry and corporate actions in a backtest, and
    keeping a run deterministic.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-instruments
status: stable
---

# RaptorBT Backtesting

## 1. When to use this skill

Load it to pick the runner for a historical simulation, to decide what the
simulation assumes about fills, costs and time, and to know what in a result
is realistic and what is not. Class-based strategies are in
[strategy](../strategy/SKILL.md); this skill covers the array runners, the
`BacktestConfig` that every runner shares, and the semantics of a backtest.

## 2. Domain overview

**Runners.** Every runner returns the same `BacktestResult` (a portfolio
runner wraps it as `PortfolioResult.result`).

| Runner | Input | Use it for |
| --- | --- | --- |
| `run_strategy_backtest` | A `Strategy`, one instrument's bars | One class-based strategy. |
| `run_strategies` | Many strategies, `{symbol: bars}` | Many strategies or instruments over shared data; accounts, ticks, corporate actions. |
| `run_single_backtest` | Bars and boolean `entries`, `exits` | A vectorised idea on one instrument; `position_sizes` and `instrument_config` size it. |
| `run_basket_backtest`, `run_portfolio_backtest` | A list of `(timestamps, open, high, low, close, volume, entries, exits, direction, weight, symbol)` | Several instruments from precomputed signals. |
| `run_multi_backtest` | Bars and a list of `(entries, exits, direction, weight, symbol)` | Several signal sets on one instrument, combined (`combine_mode`: `any`, `all`, `majority`, `independent`, `weighted`). |
| `run_pairs_backtest` | Two instruments' bars, `entries`, `exits`, `hedge_ratio` | A hedged pair. |
| `run_options_backtest` | Underlying bars, an `option_prices` series, signals | One option from a premium series. |
| `run_spread_backtest` | Underlying close, a premium series per leg, `leg_configs` | A multi-leg option spread; `max_loss`, `target_profit` and per-leg expiries. |

The array runners decide on a whole signal array computed beforehand; the
class runners decide bar by bar. The array runners cannot look ahead any
differently from your arrays: a signal array that used the future gives a
future-aware result.

**Fill timing** (`fill_timing`): `same_bar_close` (default; decide and fill at
bar `i`'s close) or `next_bar_open` (decide at `i`'s close, fill at `i+1`'s
open; a decision on the last bar never trades). `same_bar_open_lookahead` is
the pre-0.11 behaviour and not causal.

**The marketable-limit rule.** A limit order that is already marketable fills
at the market, the next bar's open, never at its own limit price, under either
fill timing. A resting limit touched inside a bar fills at its limit; a gap
through it fills at the open. A stop that the market gaps past fills at the
open. See [orders](../orders/SKILL.md).

**Slippage and latency.** `slippage` is a fraction of price applied against
you on market and stop fills (`apply_slippage`, default on); limit fills are
at their price unless `limit_slippage` models adverse selection.
`order_latency_ns` delays an order on the **tick** path only: bars are
unaffected. `fill_prob_limit` / `fill_prob_slippage` / `fill_seed` add seeded
stochastic fills; `queue_fill_model` and `partial_fills` model a queue and
partial executions on tick data. All are off by default, and the same seed
reproduces the same fills.

**Fees.** `fees` is a flat fraction of traded value. For exchange charges set
`fee_segment` (for example `"NSE-INTRADAY"`, `"NSE-DELIVERY"`, `"NFO-OPT"`):
brokerage, transaction tax, exchange charge, regulatory fee, stamp duty and
GST are itemised in `trade.fee_breakdown`. Your own rates are data: a
`RateCard` document becomes a `FeeSchedule`, passed as
`BacktestConfig(fee_schedule=..., fee_segment=...)`. A schedule without a row
for the fill's venue and segment is refused (`fee_schedule_gap`), never
charged zero.

**Expiry, settlement and corporate actions.** A contract's expiry and how it
settles are declared on its `InstrumentSpec` ([instruments](../instruments/SKILL.md));
at expiry the position closes with exit reason `Settlement` at a price that is
declared, the underlying's or the last price, never guessed. Corporate actions
are data you supply: `CorporateAction.split`, `.bonus`, `.dividend`, passed to
`run_strategies(corporate_actions=[...], adjustment_policy="raw" | "adjusted")`.
With `"raw"` the bars are as traded, a split shows as a jump and the engine
rescales positions on the ex-date, so cash stays exact. With `"adjusted"` the
engine restates every bar before the action, indicators see a continuous
series, and positions are not rescaled again.

**Determinism.** The same arrays, config and strategy give identical trades
and curves. The engine reads no clock and uses no unseeded randomness.
`bar_stamp` says whether timestamps mark a bar's open or its close; the engine
runs on close time.

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.run_single_backtest(timestamps, open, high, low, close, volume, entries, exits, direction=1, weight=1.0, symbol="UNKNOWN", config=None, position_sizes=None, instrument_config=None)` | One instrument from boolean signals. `position_sizes` is per-bar fractions of capital. |
| `raptorbt.run_basket_backtest(instruments, config=None, sync_mode=, instrument_configs=None)` | A list of instrument tuples; weights set each one's share of capital. |
| `raptorbt.run_multi_backtest(timestamps, open, high, low, close, volume, strategies, config=None, combine_mode="any")` | Combine signal sets. |
| `raptorbt.run_pairs_backtest(leg1_timestamps, leg1_open, ..., leg2_volume, entries, exits, direction=1, symbol="PAIR", config=None, hedge_ratio=1.0, dynamic_hedge=False)` | A pair trade. |
| `raptorbt.run_options_backtest(timestamps, open, high, low, close, volume, option_prices, entries, exits, direction=1, symbol=, config=None, option_type=, strike_selection=, size_type=, size_value=, lot_size=, strike_interval=, option_open_prices=None)` | An option from a premium series. |
| `raptorbt.run_spread_backtest(timestamps, underlying_close, legs_premiums, leg_configs, entries, exits, config=None, spread_type=, max_loss=None, target_profit=None, leg_expiry_timestamps=None, legs_open_premiums=None)` | A spread. `leg_configs` is `(option_type, strike, signed_quantity, lot_size)` per leg. |
| `raptorbt.run_strategy_backtest`, `raptorbt.run_strategies` | See [strategy](../strategy/SKILL.md). |
| `raptorbt.BacktestConfig` | Capital, fees, slippage, fill timing, session, risk limits and execution-realism settings (core, risk skills). Notable here: `fill_timing`, `slippage`, `apply_slippage`, `limit_slippage`, `fees`, `fee_segment`, `fee_schedule`, `money_decimals`, `order_latency_ns`, `fill_prob_limit`, `fill_prob_slippage`, `fill_seed`, `queue_fill_model`, `partial_fills`, `bar_path_adaptive`, `liquidate_on_margin_call`, `squareoff_time`. |
| `raptorbt.InstrumentConfig(lot_size=None, alloted_capital=None, existing_qty=None, avg_price=None)` | Per-instrument sizing and starting position for the array runners. |
| `raptorbt.costs.FeeSchedule`, `raptorbt.FeeSchedule` | `FeeSchedule.exchange_presets()`, `.from_json`, `.from_dict`, `.from_cards([...])`, `.rows()`, `.row(venue, segment, ...)`, `.charge(venue, segment, side, price, quantity, ...)`. |
| `raptorbt.costs.RateCard`, `raptorbt.RateCard` | A dated rate card: `RateCard.from_dict(doc)`, `.from_json`, `.fee_schedule()`, `.to_dict()`, `.backdated_fills(fill_ts)`. |
| `raptorbt.indian_cost_schedule(name)` | The legacy named cost schedule (`"equity_intraday"`, `"options_nfo"`, ...). Prefer `FeeSchedule`. |
| `raptorbt.CorporateAction.split/bonus/dividend`, `raptorbt.adjust_history` | Corporate actions as data, and the restatement used by the `"adjusted"` policy. |

## 4. Standard workflow

1. Choose the runner from the table; if the idea needs state, orders or more
   than one instrument's logic, use a class-based strategy.
2. Fix the assumptions in one `BacktestConfig`: capital, `fill_timing`,
   `slippage`, fees (`fee_segment` for exchange charges), session length.
3. Run once gross (`fees=0.0`, `slippage=0.0`), then with costs; the gap is
   what the strategy gives to the market.
4. Run with `fill_timing="next_bar_open"` if live execution cannot act on the
   closing price.
5. Read the result ([results](../results/SKILL.md)); check trades, not only
   the summary.
6. To search parameters, use [optimization](../optimization/SKILL.md).

## 5. Examples

One idea, three assumptions. The same signals, run gross, with costs, and with
the fill moved to the next open. Costs can only lower the final equity; the
later fill changes the trades:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, run_single_backtest

rng = np.random.default_rng(7)
n = 300
close = 100 + np.cumsum(rng.normal(0, 1, n))
open_ = np.concatenate([[close[0]], close[:-1]])
ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
high, low, volume = np.maximum(open_, close) + 0.5, np.minimum(open_, close) - 0.5, np.full(n, 1000.0)
fast, slow = raptorbt.sma(close, 5), raptorbt.sma(close, 20)
signals = (fast > slow, fast < slow)


def final_equity(**config):
    r = run_single_backtest(ts, open_, high, low, close, volume, *signals, config=BacktestConfig(**config))
    return r, float(r.equity_curve()[-1])


gross, gross_equity = final_equity(fees=0.0, slippage=0.0)
_, with_fees = final_equity(fees=0.001, slippage=0.0)
_, with_slippage = final_equity(fees=0.0, slippage=0.001)
later, _ = final_equity(fees=0.0, slippage=0.0, fill_timing="next_bar_open")
assert with_fees < gross_equity and with_slippage < gross_equity
assert later.trades()[0].entry_idx == gross.trades()[0].entry_idx + 1
again, again_equity = final_equity(fees=0.0, slippage=0.0)
assert again_equity == gross_equity                             # deterministic

# 25% of capital per trade instead of all of it
quarter = run_single_backtest(ts, open_, high, low, close, volume, *signals,
                              config=BacktestConfig(fees=0.0), position_sizes=np.full(n, 0.25))
t = quarter.trades()[0]
assert abs(t.size * t.entry_price - 25_000.0) < 1e-6
```

Exchange charges. A segment prices the run on the built-in exchange rows and
itemises each trade; your own card replaces them. A card that lacks the row
you asked for is refused, not treated as free:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, FeeSchedule, RateCard, run_single_backtest
from raptorbt.errors import InvalidParameter

rng = np.random.default_rng(7)
n = 300
close = 100 + np.cumsum(rng.normal(0, 1, n))
open_ = np.concatenate([[close[0]], close[:-1]])
ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
high, low, volume = np.maximum(open_, close) + 0.5, np.minimum(open_, close) - 0.5, np.full(n, 1000.0)
fast, slow = raptorbt.sma(close, 5), raptorbt.sma(close, 20)


def run(**config):
    return run_single_backtest(ts, open_, high, low, close, volume, fast > slow, fast < slow,
                               config=BacktestConfig(**config))


trade = run(fee_segment="NSE-INTRADAY").trades()[0]
b = trade.fee_breakdown
assert set(b) >= {"brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst", "total"}
assert abs(b["total"] - trade.fees) < 1e-9 and b["brokerage"] == 40.0       # 20 per side, the preset's cap

card = RateCard.from_dict({
    "name": "flat-fee-card", "effective_from": "2026-01-01T00:00:00+05:30", "source": "my broker's schedule",
    "rates": [{"exchange": "NSE", "segment": "INTRADAY", "underlying": None,
               "brokerage": {"per_order": "55"},
               "transaction_tax": {"kind": "STT", "percent": "0.025", "sides": ["SELL"]},
               "exchange_turnover_percent": "0.00307", "sebi_percent": "0.0001",
               "stamp_duty_percent": "0.003", "gst_percent": "18"}],
})
custom = run(fee_segment="NSE-INTRADAY", fee_schedule=card.fee_schedule()).trades()[0]
assert custom.fee_breakdown["brokerage"] == 110.0                          # 55 on each of two orders

try:
    BacktestConfig(fee_segment="MCX-FUT", fee_schedule=card.fee_schedule())
except InvalidParameter as e:
    assert e.code == "fee_schedule_gap"

quote = FeeSchedule.exchange_presets().charge("NFO", "OPTIONS", "SELL", price=100.0, quantity=75)
assert quote["stt"] == 11.25 and quote["brokerage"] == 20.0
```

Several signal sets on one instrument, and several instruments. `all`
needs every set to agree, so it trades less than `any`; `independent` runs
each set as its own position:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, run_basket_backtest, run_multi_backtest


def bars(seed, base, n=300):
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
    return ts, open_, np.maximum(open_, close) + 0.5, np.minimum(open_, close) - 0.5, close, np.full(n, 1000.0)


ts, o, h, l, c, v = bars(7, 100.0)
fast, slow, trend = raptorbt.sma(c, 5), raptorbt.sma(c, 20), raptorbt.sma(c, 50)
sets = [(fast > slow, fast < slow, 1, 1.0, "cross"), (fast > trend, fast < trend, 1, 1.0, "trend")]
cfg = BacktestConfig(fees=0.0)
counts = {m: run_multi_backtest(ts, o, h, l, c, v, sets, config=cfg, combine_mode=m).metrics.total_trades
          for m in ("any", "all", "independent")}
assert counts["all"] <= counts["any"] <= counts["independent"]

a, b = bars(7, 100.0), bars(9, 50.0)
legs = []
for data, name in ((a, "A"), (b, "B")):
    f, s = raptorbt.sma(data[4], 5), raptorbt.sma(data[4], 20)
    legs.append((*data, f > s, f < s, 1, 0.5, name))
basket = run_basket_backtest(legs, config=cfg)
assert {t.symbol for t in basket.trades()} == {"A", "B"}
```

A short call spread leg from its premium series. The premium falls from 90 at
entry to 60 at exit on a 75-lot, so a short position earns 30 x 75:

```python
import numpy as np
from raptorbt import BacktestConfig, run_spread_backtest

premium = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 60.0])
n = len(premium)
entries, exits = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
entries[1], exits[4] = True, True
result = run_spread_backtest(
    timestamps=np.arange(n, dtype=np.int64) * 300_000_000_000 + 1_786_000_000_000_000_000,
    underlying_close=np.full(n, 24_550.0),
    legs_premiums=[premium],
    leg_configs=[("CE", 24_800.0, -1, 75)],        # short one lot of 75
    entries=entries, exits=exits,
    config=BacktestConfig(initial_capital=500_000.0, fees=0.0), spread_type="custom",
)
assert result.trades()[0].pnl == 2_250.0
```

Corporate actions. A 2-for-1 split on raw bars: the engine doubles the
position and halves its price on the ex-date, so equity does not jump. On
adjusted bars the history is restated instead:

```python
import numpy as np
from raptorbt import BacktestConfig, CorporateAction, Strategy, run_strategies
from raptorbt.strategy import orders

DAY, T0, n = 86_400 * 10**9, 1_767_225_600 * 10**9, 8
close = np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 52.0, 52.0])      # the split shows as a jump
data = {"AAA": dict(timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * DAY, open=close.copy(),
                    high=close + 1.0, low=close - 1.0, close=close, volume=np.full(n, 1e6))}


class Hold(Strategy):
    def __init__(self):
        super().__init__()
        self.actions, self.sizes = [], []

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", quantity=100))
        if ctx.idx in (2, 3):
            self.sizes.append(ctx.position.size)

    def on_corporate_action(self, ctx, event):
        self.actions.append((event.outcome, event.quantity))


split = [CorporateAction.split("AAA", T0 + 4 * DAY, 2.0)]
s = Hold()
result = run_strategies([s], data, BacktestConfig(fees=0.0), corporate_actions=split,
                        adjustment_policy="raw", bar_stamp="close")[0].result
assert s.actions == [("split", 2.0)] and s.sizes == [100.0, 200.0]
trade = result.trades()[0]
assert (trade.size, trade.entry_price, trade.pnl) == (200.0, 50.0, 400.0)   # 200 x (52 - 50)

adjusted = Hold()
run_strategies([adjusted], data, BacktestConfig(fees=0.0), corporate_actions=split,
               adjustment_policy="adjusted", bar_stamp="close")
assert adjusted.sizes == [100.0, 100.0]                  # history restated; no second rescale
```

Bar stamps, and a seeded stochastic run. Timestamps that mark a bar's open
need the bar width, and the engine acts at the close; the same seed
reproduces the same stochastic fills:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import orders

MINUTE = 60_000_000_000
n = 10
c = np.full(n, 100.0)
tape = dict(timestamps=np.arange(n, dtype=np.int64) * MINUTE, open=c, high=c + 1.0, low=c - 1.0,
            close=c, volume=np.full(n, 1e6))


class Probe(Strategy):
    def __init__(self):
        super().__init__()
        self.stamps = []

    def on_bar(self, ctx):
        self.stamps.append((ctx.timestamp, ctx.bar_close_ts))


p = Probe()
run_strategy_backtest(p, **tape, bar_stamp="open", bar_ns=MINUTE, config=BacktestConfig(fees=0.0))
assert p.stamps[0] == (0, MINUTE)                       # stamped at its open; acted on at its close
try:
    run_strategy_backtest(Probe, **tape, bar_stamp="open")
except InvalidParameter as e:
    assert e.code == "missing_bar_ns"

rng = np.random.default_rng(1)
close = 100 + np.cumsum(rng.normal(0, 1, 300))
open_ = np.concatenate([[close[0]], close[:-1]])
bars = dict(timestamps=(np.arange(300, dtype=np.int64) + 1) * MINUTE, open=open_,
            high=np.maximum(open_, close) + 0.5, low=np.minimum(open_, close) - 0.5,
            close=close, volume=np.full(300, 1e6))


class Limits(Strategy):
    def on_bar(self, ctx):
        if ctx.idx % 20 == 1:
            side = "buy" if (ctx.idx // 20) % 2 == 0 else "sell"
            self.submit_order(orders.Limit(side=side, price=ctx.bar.close, quantity=10))


def final(seed):
    cfg = BacktestConfig(fees=0.0, fill_prob_limit=0.5, fill_seed=seed)
    return float(run_strategy_backtest(Limits, **bars, bar_stamp="close", config=cfg).equity_curve()[-1])


assert final(1) == final(1)                              # same seed, same fills
assert len({final(s) for s in (1, 2, 3, 4)}) > 1         # a different seed passes over different orders
```

## 6. Composition

- Class-based strategies and the hooks that react to fills:
  [strategy](../strategy/SKILL.md); the order types behind fills:
  [orders](../orders/SKILL.md).
- Accounts, margin and exact money: [portfolio](../portfolio/SKILL.md);
  limits and halts: [risk](../risk/SKILL.md).
- Expiry, settlement and sessions are declared on instruments:
  [instruments](../instruments/SKILL.md).
- Reading the output: [results](../results/SKILL.md); sweeping parameters
  honestly: [optimization](../optimization/SKILL.md).
- What changes when orders go to a venue instead of the simulator:
  [execution](../execution/SKILL.md).

## 7. Constraints and correctness

- Timestamps are `int64` nanoseconds and must not decrease; arrays must share
  one length; open, high, low and close must be finite. Each is refused with
  a structured error naming the first bad index.
- A backtest fills at prices the data contains. Slippage, fees and
  `next_bar_open` are your protection against assuming a fill you could not
  have had; a strategy whose edge disappears under them did not have one.
- Within a bar where both a stop and a target are touched, the stop is taken
  first unless `bar_path_adaptive` is on, which infers the path from the
  candle.
- `fees` is a fraction of traded value; `fee_segment` and `fee_schedule`
  replace it with itemised charges. A short owes the sell-side tax when it
  opens; a four-leg structure pays the per-order charge once per leg per side.
- `order_latency_ns` does not affect bars. `squareoff_time` (an `"HH:MM"`
  string with `session_tz_offset_ns`) closes positions in every runner that
  has a session: the first bar at or after it closes the position
  (`Squareoff`), and entries are refused as `squareoff` for the rest of that
  local day. An instrument's own `SessionSchedule` takes precedence over it,
  and also supplies the closed-market rejections and DAY expiry.
- `run_multi_backtest` refuses an unknown `combine_mode` with
  `invalid_combine_mode`; the modes are `any`, `all`, `majority`,
  `independent` and `weighted`.
- Premium-only runners (`run_options_backtest`, `run_spread_backtest`) never
  invent a price: with `next_bar_open` they fill at an open-premium series if
  you supply one, otherwise at the next bar's value.
- Corporate actions are supplied by you; the engine never infers a split from a
  price jump. `adjustment_policy` is required with them and is part of an
  indicator's identity.
- Determinism holds for the same inputs on the same engine version. Record the
  installed package version with a result you keep.

## 8. Common mistakes

A fee schedule without the row the run needs is refused rather than charged
nothing:

<!-- skill: invalid -->
```python
from raptorbt import BacktestConfig, RateCard

card = RateCard.from_dict({
    "name": "equity-only", "effective_from": "2026-01-01T00:00:00+05:30", "source": "example",
    "rates": [{"exchange": "NSE", "segment": "INTRADAY", "underlying": None,
               "brokerage": {"per_order": "20"},
               "transaction_tax": {"kind": "STT", "percent": "0.025", "sides": ["SELL"]},
               "exchange_turnover_percent": "0.00307", "sebi_percent": "0.0001",
               "stamp_duty_percent": "0.003", "gst_percent": "18"}],
})
BacktestConfig(fee_segment="MCX-FUT", fee_schedule=card.fee_schedule())   # InvalidParameter: fee_schedule_gap
```

Bars stamped at their open without saying how wide they are:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import Strategy, run_strategy_backtest

x = np.ones(5)
run_strategy_backtest(Strategy, np.arange(5, dtype=np.int64), x, x, x, x, x, bar_stamp="open")
```

- Comparing gross and net runs as if they were two strategies.
- Believing `slippage` applies to a limit that fills at its price: it does not
  (see `limit_slippage`).
- Feeding adjusted prices and also a `"raw"` policy, or the reverse: the
  position is rescaled twice or not at all.
- Letting an indicator warm up inside the first bar of data and then trading
  on it: guard with the indicator's own readiness.
- Reading `PortfolioResult` fields from a `BacktestResult` and vice versa:
  the portfolio wrapper exposes `.result`.
- Setting `squareoff_time` without `session_tz_offset_ns` and getting a
  square-off at the wrong local time: the cutoff is read in the offset you give.

## 9. Validation

1. Run gross, then with real costs, then with `next_bar_open`. Keep all three.
2. Run twice; assert identical trades and equity.
3. Pick three trades and check entry bar, price and exit reason by hand
   against the tape.
4. Check fees: sum of `trade.fees` against `metrics.total_fees_paid`, and a
   breakdown's `total` against the trade's `fees`.
5. Count rejected entries (`PortfolioResult.rejected_entries`) and read
   their reasons: a backtest that never could size or fund its trades proves
   little.
6. Before live use, see [execution](../execution/SKILL.md) and
   [live-trading](../live-trading/SKILL.md).

## 10. Related skills

- [Core](../core/SKILL.md)
- [Strategy](../strategy/SKILL.md)
- [Orders](../orders/SKILL.md)
- [Results](../results/SKILL.md)
- [Optimization](../optimization/SKILL.md)
- [Instruments](../instruments/SKILL.md)
