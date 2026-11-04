---
name: raptorbt-risk
description: >
    Risk limits, kill switches, margin models and position sizing helpers:
    what refuses an order, what halts a strategy, and how many lots to trade.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
    - raptorbt-portfolio
status: stable
---

# RaptorBT Risk

## 1. When to use this skill

Load it when you cap order size, position size, exposure, open positions,
daily loss or drawdown; when a strategy needs a kill switch; when you size a
position from capital, risk, volatility or margin; or when you measure how a
set of weights spreads risk across assets. Limits sit in the engine, below your
strategy: a strategy that forgets a check cannot trade past them.

## 2. Domain overview

**Limits are off by default and are `BacktestConfig` settings.** A non-positive
value counts as off. With none set a run is bit-identical to one without them.
A breach refuses the order or entry whole; it never shrinks it. The refusal
reaches `on_order_rejected` with a stable `reject_reason`.

| Setting | Refuses when | `reject_reason` |
| --- | --- | --- |
| `max_order_qty` | one order's units exceed it | `order_size` |
| `max_order_notional` | units x price x multiplier exceed it | `order_size` |
| `max_position_qty` | the instrument's net position after the fill would exceed it | `max_position` |
| `max_gross_exposure` | notional across instruments after the fill would exceed it | `max_exposure` |
| `max_positions` | already at this many open positions | `max_positions` |
| `max_daily_loss` (money), `max_daily_loss_pct` (percent of the day's starting equity) | the day's loss, realised plus unrealised, has reached the limit (the tighter of the two applies) | `daily_loss`, until the next day |
| `max_drawdown_pct` (percent) | equity has fallen that far from its peak; latches for the run | `drawdown_halt` |
| `ctx.halt(reason, flatten=False)` | the strategy halted itself; latches | `killed` |

An equal value passes; one step above is refused. A reducing order is not
blocked by a position cap it reduces.

**On a `LiveNode` with a real venue** (an adapter that is not `PaperExecution`)
the same limits are checked before the adapter sees the order, because the venue
knows none of them. A market order names no price, so it is priced at the last
price seen on the instrument. Orders the venue is working or has not answered
count as if filled: with `max_position_qty=7`, a second 5-unit buy is refused
while the first is still out, and `max_gross_exposure` counts the working orders
of every instrument. Reducing orders are never held back. A market order that
cannot be priced at all (no print yet on its instrument) under
`max_order_notional` or `max_gross_exposure` is refused as `no_mark`, never
sent unchecked.

**Events.** `on_risk_event(ctx, event)` hears the first six limits as
`"risk_rejected"` (`event.reject_reason` says which), and the two stop-trading
moments as `"risk_halted"`: the daily loss limit being reached
(`reject_reason == "daily_loss"`), and a halt (the reason you gave). Drawdown
halts and `max_positions` refusals reach only `on_order_rejected`. A refused
typed order reaches both hooks.

**Daily loss** measures from the day's starting equity, where a day is the
local calendar date at `session_tz_offset_ns` (set it for non-UTC markets).
It blocks new entries for the rest of the day and resets at the next one. With
`flatten_on_daily_loss=True` it also closes open positions (exit reason
`Liquidation`).

**Kill switch.** `ctx.halt(reason, flatten=False)` stops this strategy: new
entries are refused as `killed`, working orders are cancelled, and with
`flatten=True` positions are closed at the last mark. It cannot be undone, and
the result reports `halted` and `halted_at`. `LiveNode.halt` does the same
for a live strategy; `halt_account` halts a whole shared account.

**Margin** is a funding rule, not a limit: see
[portfolio](../portfolio/SKILL.md) for accounts and `MarginModel`. A model
that has no rule for an instrument refuses its entries as `margin_unpriced`;
it never reads a missing rule as zero margin. `MarginModel.defined_risk()` and
`spread_credit` price hedged option groups for less than their legs.

**Sizing** (`raptorbt.sizing`) turns a rule into a whole number of lots. Every
function rounds down, returns lots (multiply by the lot size for units), and
raises rather than return zero: `below_one_lot` when the rule comes to less
than one lot (`.value` is the fractional lots), `invalid_sizing_input` for a
non-positive or non-finite input.

**Portfolio risk.** `estimate_covariance` shrinks a sample covariance into a
`RiskModel`; `compute_risk_contributions` splits a book's volatility by asset.

## 3. API reference

| Name | What it is |
| --- | --- |
| `BacktestConfig(max_order_qty=, max_order_notional=, max_position_qty=, max_gross_exposure=, max_positions=, max_daily_loss=, max_daily_loss_pct=, flatten_on_daily_loss=False, max_drawdown_pct=, liquidate_on_margin_call=False)` | The limits above. All default off. |
| `StrategyContext.halt(reason="manual", flatten=False)` | Kill switch. |
| `Strategy.on_risk_event(ctx, event)`, `on_margin_call(ctx, event)` | Hooks for limit breaches and margin calls. |
| `raptorbt.margin.MarginModel` | `no_margin`, `cash_only`, `leverage({...})`, `fraction_of_notional(fraction)`, `option_simple`, `table`, `by_type`, `by_product`, `defined_risk`, `spread_credit`, `from_spec`; `.required(spec, qty, price)`, `.required_legs(...)`, `.describe()`. `raptorbt.margin.defined_risk(legs)` is the worst expiry loss of a set of option legs. |
| `raptorbt.sizing.lots_for_capital(capital, price, lot_size, *, fraction=1.0, multiplier=1.0)` | Lots whose cost fits a fraction of capital. |
| `raptorbt.sizing.lots_for_percent_of_equity(equity, percent, price, lot_size, *, multiplier=1.0)` | `percent` is 0 to 100. |
| `raptorbt.sizing.lots_for_risk(capital, risk_fraction, entry, stop, lot_size, *, multiplier=1.0)` | Lots that lose at most the fraction if price runs from entry to stop. A stop at the entry is refused. |
| `raptorbt.sizing.lots_for_volatility(capital, target_vol, asset_vol, price, lot_size, *, max_leverage=None, multiplier=1.0)` | Lots whose notional is `capital x target_vol / asset_vol`. |
| `raptorbt.sizing.lots_for_margin(available, margin_per_lot, *, buffer=0.0)`, `max_lots_by_margin(required, available, *, cap=100_000)` | Lots that fit the margin; the second takes a function for margin that is not linear in lots. |
| `raptorbt.sizing.units(lots, lot_size)`, `raptorbt.sizing.Sizer(ctx)` | Lots to units; the same rules bound to a context's equity and free capital. |
| `raptorbt.RiskModel`, `raptorbt.estimate_covariance(returns, asset_ids, periods_per_year)` | A shrunk covariance: `cov()`, `shrinkage_intensity`, `n_obs`, `asset_ids`. |
| `raptorbt.compute_risk_contributions(model, weights, asset_ids)` | `RiskContributions`: `marginal()`, `contribution()`, `pct_contribution()`, `total_vol_annualized`. |

## 4. Standard workflow

1. Decide the limits before the strategy: the largest order, position and
   exposure you would accept, and the loss that should stop the day.
2. Put them in `BacktestConfig`; set `session_tz_offset_ns` so a day is the
   market's day.
3. Size in the strategy with `raptorbt.sizing`, from the stop distance or the
   volatility, not from a constant.
4. Handle `on_order_rejected` and `on_risk_event`: log the reason, do not retry
   the same order into the same limit.
5. Test each limit at its boundary: equal passes, one step above is refused.
6. Run the same limits live; they are the same settings on a `LiveNode`, and
   on a real venue they are checked before each order is sent.

## 5. Examples

Order, position and exposure limits, each tested at the boundary. Under
hedging every order opens a position, so the caps do the refusing:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

n = 6
close = np.full(n, 100.0)
tape = dict(timestamps=np.arange(n, dtype=np.int64) * 3_600_000_000_000, open=close,
            high=close + 1.0, low=close - 1.0, close=close, volume=np.full(n, 1e6))


class Orders(Strategy):
    def __init__(self, *quantities):
        super().__init__()
        self.quantities, self.rejected, self.risk = quantities, [], []

    def on_bar(self, ctx):
        if 1 <= ctx.idx <= len(self.quantities):
            self.submit_order(orders.Market(side="buy", quantity=self.quantities[ctx.idx - 1]))

    def on_order_rejected(self, ctx, event):
        self.rejected.append(event.reject_reason)

    def on_risk_event(self, ctx, event):
        self.risk.append((event.kind, event.reject_reason))


def run(*quantities, **limits):
    s = Orders(*quantities)
    result = run_strategy_backtest(s, **tape, bar_stamp="close", oms_type="hedging",
                                   config=BacktestConfig(fees=0.0, **limits))
    return s, result


s, r = run(500, max_order_qty=500)                       # equal passes
assert s.rejected == [] and r.metrics.total_trades == 1
s, r = run(501, max_order_qty=500)                       # one above is refused whole
assert s.rejected == ["order_size"] and s.risk == [("risk_rejected", "order_size")]
assert r.metrics.total_trades == 0                       # not shrunk to 500

s, r = run(100, 100, max_position_qty=150)               # second would take the position to 200
assert s.rejected == ["max_position"] and r.metrics.total_trades == 1

s, r = run(100, 100, max_gross_exposure=15_000.0)        # 100 x 100 = 10,000 held, +10,000 > 15,000
assert s.rejected == ["max_exposure"]

s, r = run(10, 10, max_positions=1)                      # refused, but no risk event
assert s.rejected == ["max_positions"] and s.risk == []
```

Daily loss. The limit is reached on day one, new entries are refused for the
rest of that day and positions are flattened; the next day trades again:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

HOUR = 3_600_000_000_000
close = np.array([100.0, 100.0, 95.0, 90.0, 90.0] + [90.0] * 24)       # hourly: bars 0-23 are day one
tape = dict(timestamps=np.arange(len(close), dtype=np.int64) * HOUR, open=close,
            high=close + 1.0, low=close - 1.0, close=close, volume=np.full(len(close), 1e6))


class Loser(Strategy):
    def __init__(self):
        super().__init__()
        self.rejected, self.risk = [], []

    def on_bar(self, ctx):
        if ctx.idx in (1, 4, 26):
            self.submit_order(orders.Market(side="buy", quantity=1000 if ctx.idx == 1 else 10))

    def on_order_rejected(self, ctx, event):
        self.rejected.append((event.idx, event.reject_reason))

    def on_risk_event(self, ctx, event):
        self.risk.append((event.kind, event.idx))


s = Loser()
result = run_strategy_backtest(
    s, **tape, bar_stamp="close", oms_type="hedging",
    config=BacktestConfig(fees=0.0, max_daily_loss=5_000.0, flatten_on_daily_loss=True),
)
assert ("risk_halted", 2) in s.risk                       # 1000 units x 5 down = 5,000 reached on bar 2
assert s.rejected == [(4, "daily_loss")]                  # day one is closed to new entries
first, second = result.trades()
assert (first.exit_idx, first.exit_reason, first.pnl) == (2, "Liquidation", -5_000.0)
assert second.entry_idx == 26                              # day two trades again
```

Kill switches. `ctx.halt` refuses entries from then on and can flatten;
`max_drawdown_pct` halts the run when equity falls far enough from its peak:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders


def tape(closes):
    c = np.asarray(closes, dtype=float)
    return dict(timestamps=np.arange(len(c), dtype=np.int64) * 60_000_000_000, open=c,
                high=c + 1.0, low=c - 1.0, close=c, volume=np.full(len(c), 1e6))


class Halting(Strategy):
    def __init__(self):
        super().__init__()
        self.rejected = []

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", quantity=10))
        if ctx.idx == 2:
            ctx.halt("operator stop")
        if ctx.idx == 4:
            self.submit_order(orders.Market(side="buy", quantity=10))

    def on_order_rejected(self, ctx, event):
        self.rejected.append(event.reject_reason)


s = Halting()
result = run_strategy_backtest(s, **tape([100.0] * 6), bar_stamp="close", oms_type="hedging",
                               config=BacktestConfig(fees=0.0))
assert s.rejected == ["killed"] and result.halted and result.halted_at == 2


class AllIn(Strategy):
    def __init__(self):
        super().__init__()
        self.rejected = []

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.enter(size_frac=1.0)
        if ctx.idx == 6:
            self.submit_order(orders.Market(side="buy", quantity=1))

    def on_order_rejected(self, ctx, event):
        self.rejected.append(event.reject_reason)


s = AllIn()
result = run_strategy_backtest(s, **tape([100, 100, 90, 80, 70, 60, 60, 60]), bar_stamp="close",
                               oms_type="hedging", config=BacktestConfig(fees=0.0, max_drawdown_pct=15.0))
assert result.halted and s.rejected == ["drawdown_halt"]
```

Sizing helpers. Each answers "how many lots" and rounds down; a rule that
comes to less than one lot raises instead of returning zero:

```python
from raptorbt import sizing
from raptorbt.errors import InvalidParameter

# 1% of 1,000,000 at risk, entry 100, stop 95, lot of 50: 10,000 / (5 x 50) = 40 lots
assert sizing.lots_for_risk(1_000_000, 0.01, 100.0, 95.0, 50) == 40
# 20% of capital in a 2,500 x 25 lot: 200,000 / 62,500 = 3.2, rounded down
assert sizing.lots_for_capital(1_000_000, 2_500.0, 25, fraction=0.2) == 3
# target 10% vol on a 20% vol asset: half of capital, 500,000 / 62,500 = 8 lots (cap 2x is not reached)
assert sizing.lots_for_volatility(1_000_000, 0.10, 0.20, 2_500.0, 25, max_leverage=2) == 8
# 500,000 available less a 10% buffer, 120,000 margin per lot
assert sizing.lots_for_margin(500_000, 120_000, buffer=0.1) == 3
assert sizing.units(3, 50) == 150.0

try:
    sizing.lots_for_risk(1_000, 0.01, 100.0, 95.0, 50)       # risks 10 where one lot loses 250
except InvalidParameter as e:
    assert e.code == "below_one_lot" and abs(e.value - 0.04) < 1e-12
```

A covariance and where a book's risk comes from. Equal weights are not equal
risk; the contributions sum to one:

```python
import numpy as np
import raptorbt as rb

ids = ["A", "B", "C", "D"]
rng = np.random.default_rng(11)
returns = (rng.normal(0, 0.01, (400, 1)) * np.linspace(0.2, 1.0, 4)
           + rng.normal(0, 0.012, (400, 4)))
model = rb.estimate_covariance(returns, ids, 252.0)
assert 0.0 <= model.shrinkage_intensity <= 1.0 and model.n_obs == 400

risk = rb.compute_risk_contributions(model, np.full(4, 0.25), ids)
share = risk.pct_contribution()
assert abs(share.sum() - 1.0) < 1e-12
assert share[3] > share[0]                      # the most market-exposed asset carries the most risk
assert risk.total_vol_annualized > 0.0
```

## 6. Composition

- The account the limits protect: [portfolio](../portfolio/SKILL.md). Account
  groups apply `max_positions`, daily loss, drawdown and gross exposure to the
  whole account through `AccountSpec`.
- The orders that get refused: [orders](../orders/SKILL.md).
- Stop-loss and target levels are order features, not limits: see orders and
  `StopConfig`/`TargetConfig`.
- Event hooks are in [strategy](../strategy/SKILL.md).
- Runnable examples: `examples/portfolio/shared_account.py`, `examples/options/nifty_momentum.py`.

## 7. Constraints and correctness

- The limits apply the same way in `run_strategy_backtest`, `run_strategies`,
  the portfolio and tick runners and `LiveNode`.
- Under netting, an order in the direction of an open position adds to it, and
  the size caps (`max_position_qty`, `max_order_qty`, exposure) judge the
  position after the add, so a cap is tested the same with a position open and
  with a flat account (see orders).
- `max_drawdown_pct` and the percent forms are percents (`15.0` is 15%);
  `fees`, `slippage` and sizing `fraction` / `risk_fraction` are fractions.
- Daily loss counts unrealised loss, and the day boundary follows
  `session_tz_offset_ns`; with the default (UTC) a session that crosses UTC
  midnight is cut into two days.
- A halt latches for the rest of the run. Daily loss does not: it resets at
  the next day.
- A flatten closes at the latest mark the engine has, not at a price a later bar prints.
- The margin rule never reads a missing instrument rule as zero: it refuses.
- `ctx.halt(reason, flatten=True)` records the flattened trade once, in
  `run_strategy_backtest` and `run_strategies` alike (`metrics.total_trades`
  counts it once).
- Sizing never rounds up and never returns zero. Handle `below_one_lot`: the
  honest answer may be "do not trade".

## 8. Common mistakes

A stop at the entry price has infinite size, so it is refused:

<!-- skill: invalid -->
```python
from raptorbt import sizing

sizing.lots_for_risk(1_000_000, 0.01, 100.0, 100.0, 50)   # InvalidParameter: invalid_sizing_input
```

A fraction above one:

<!-- skill: invalid -->
```python
from raptorbt import sizing

sizing.lots_for_capital(1_000_000, 100.0, 1, fraction=1.5)   # InvalidParameter: invalid_sizing_input
```

- Expecting a cap to shrink an order. It refuses it; size under the cap
  yourself.
- Retrying a rejected order every bar into the same limit.
- Reading `max_daily_loss=0` as "no loss allowed". Zero means off.
- Forgetting `session_tz_offset_ns` and getting a "day" that cuts the session.
- Putting the kill switch only in backtest code: a live strategy needs its own
  path to `ctx.halt`.
- Using `percent_of_notional(0.2)` for 20 percent margin: it is 0.2 percent.
  Use `fraction_of_notional(0.2)`, which refuses `20`.
- Sizing from indicators in a fill hook under `fill_timing="next_bar_open"`: a size or stop distance you compute in a fill hook answers a fill at a bar's open, but an indicator or object you own and update yourself already includes that bar's close. Size from values you stored on an earlier bar (for example in `on_bar`), not from indicators read in the hook.

## 9. Validation

1. For each limit, assert the equal case passes and the next step up is
   refused with the reason you expect.
2. Confirm the same strategy with no limits is unchanged (the limits are off
   by default and exactly so).
3. Replay a bad day: a gap through your stop, a run of losers. Check the
   daily-loss halt fires on the bar you expect and that the next day trades.
4. Check `halted` and `halted_at` on the result after any kill switch.
5. Before live use, set the limits on the `LiveNode` config, rehearse
   `halt` and `halt_account`, and read [live-trading](../live-trading/SKILL.md).

## 10. Related skills

- [Portfolio](../portfolio/SKILL.md)
- [Orders](../orders/SKILL.md)
- [Strategy](../strategy/SKILL.md)
- [Backtesting](../backtesting/SKILL.md)
