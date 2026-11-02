# Option structures and position sizing

`raptorbt.structures` describes multi-leg option positions declaratively, binds them to
contracts from an `OptionChain` at the event time, and places them as one group of ordinary
orders. `raptorbt.sizing` turns capital, risk, volatility or margin into a whole number of lots.
Neither adds an order type or touches the engine: a structure is a set of `Market`/`Limit`
orders sent with `submit_order`, so it behaves the same in `run_strategies`, the portfolio and
tick runners and `LiveNode`.

## Describing a structure

```python
from raptorbt import structures as st

condor = st.iron_condor(inner_steps=2, outer_steps=4)             # sell 2 out, buy 4 out
spread = st.bull_call_spread(long_steps=0, short_steps=2)
custom = st.Structure(
    legs=[
        st.Leg("call", st.Steps(0), st.Expiry.nearest(), "sell", ratio=2),
        st.Leg("call", st.Delta(0.25, vol=0.14), st.Expiry.monthly(), "buy"),
    ],
    name="my_ratio",
)
```

Factories: `straddle`, `strangle`, `strangle_by_delta`, `vertical`, `bull_call_spread`,
`bear_call_spread`, `bull_put_spread`, `bear_put_spread`, `butterfly`, `condor`, `iron_condor`,
`iron_butterfly`, `calendar`, `ratio_spread`, `risk_reversal`, `covered_call`,
`protective_put`, `collar`. Long structures take `side="sell"` to flip every leg (and short
ones `side="buy"`).

A `Leg` is `Leg(right, strike, expiry, side, ratio)`:

| strike selector | picks |
|---|---|
| `Steps(n)` / `Atm()` | `n` listed strikes from the money strike, up positive, on the ladder of the structure's earliest expiry (a later-expiry leg takes the same strike value there) |
| `Strike(value)` | an explicit strike |
| `Delta(target, vol, rate, dividend_yield)` | the listed contract nearest `target` in magnitude, priced at a flat vol at the event time (`OptionChain.by_delta`) |
| `Premium(target, measure)` | the contract whose `measure(spec)` is nearest `target` (`OptionChain.closest_by`) |

Expiries are `Expiry.nearest(min_days)`, `nth(n)`, `monthly(n)`, `weekly(n)`, `on(date_ns)`
and `at(expiry_ns)`; they are `OptionChain.pick_expiry` at the event time. The money strike is
the listed strike nearest the spot, the lower on a tie. Steps are counted outward from it for
each right: a put "2 out" is two strikes down. A strike or expiry the chain does not list at
that time raises `InvalidParameter` with code `structure_contract_missing`.

`UnderlyingLeg(side, ratio)` (used by `covered_call`, `protective_put`, `collar`) holds shares
against the options: `ratio` option lots' worth of units. It needs
`underlying_spec=` on resolve, and the units must be a multiple of its lot size.

## Resolving

```python
r = condor.resolve(chain, "NIFTY", spot, now_ns=ctx.bar_close_ts, lots=2)
for leg in r.legs:
    leg.instrument, leg.side, leg.lots, leg.quantity   # symbol, "buy"/"sell", lots, units
r.net_premium(prices)          # debit positive, credit negative, per unit of the option legs
r.margin(MarginModel.spread_credit(...), prices)   # whole group, MarginModel.required_legs
```

Every query takes `now_ns`; nothing listed later is visible (no lookahead). A leg's lots are
`ratio x lots`; its quantity is lots x the contract's lot size.

## Placing a structure

```python
class Condor(Strategy):
    def on_bar(self, ctx):
        self.marks[ctx.symbol] = ctx.bar.close
        self.update_structures(ctx, self.marks)         # every bar, every symbol
        if ctx.symbol == "NIFTY" and self.group is None:
            self.group = self.submit_structure(
                ctx, st.iron_condor(2, 4), chain=chain, underlying="NIFTY",
                spot=ctx.bar.close, lots=1,
                exit=st.Exit(stop_loss=5_000, take_profit=3_000, days_before_expiry=1),
            )
```

`Strategy.submit_structure(ctx, structure, *, chain, underlying, spot, lots=1, now_ns=None,
underlying_spec=None, order_type="market", prices=None, net_limit=None, exit=None,
on_leg_failure="unwind", margin_model=None, available=None, tif="gtc", group_id=None)`

* `order_type="limit"` sends each leg as a `Limit` at `prices[symbol]`. `prices` is also
  the reference for `net_limit` and for the margin check.
* `net_limit`: the net premium (debit positive) at those prices must not exceed it, else
  `structure_limit_breached` and nothing is sent.
* `margin_model` and `available` (both or neither): `ResolvedStructure.margin(...)` must fit
  `available`, else `structure_margin_exceeded` and nothing is sent. The engine's own margin
  check still applies to each leg.
* All checks run before any order is queued, so a group is sent whole or not at all.
* Two legs resolving to the same contract are refused; give one leg a `ratio`.

Legs go out protective-first (shares and bought options, then sold options), each as its own
order with client id `"<group>:<leg index>"` and `tags={"structure": group}`. The group id
defaults to `"<order_id_tag>-st<n>"`.

### Fill semantics and leg failure

Legs fill independently. A leg is *failed* when its order is rejected, cancelled or expired.

| `on_leg_failure` | what happens on the first failed leg |
|---|---|
| `"unwind"` (default) | every leg still working is cancelled; every leg that filled is closed with a `reduce_only` market order; the group ends `closed`, reason `leg_failure` |
| `"keep"` | nothing is undone; `status` reads `broken` and the strategy decides (`close_structure`) |

The unwind runs in `update_structures` (call it from `on_bar`, for every symbol's bar), or in
the same event step as the failure if the strategy inherits `StructureEvents`
(`class S(StructureEvents, Strategy)`). Until then a filled leg is open. A leg that fills after
the unwind began (a cancel that lost a race with a fill) is closed on the next update; a close
the engine refuses is retried three times and then reported as `stuck`.

`group.status(cache)` / `strategy.structure_status(group)`: `submitted`, `working`, `filled`
(every leg), `broken`, `closing`, `closed`, `stuck`. `group.leg_status(cache, i)` is
`submitted`, `working`, `filled`, `rejected`, `canceled` or `expired`.

Fills are read from the order cache as whole-order fills. With `partial_fills` on, a leg that
has filled in part reads as filled; its close is sent for the planned quantity and is
`reduce_only`, so it cannot open the opposite position.

### Group exits

`Exit(stop_loss, take_profit, premium_at_or_above, premium_at_or_below, close_at_ns,
days_before_expiry, entry_timeout_ns)`: the first rule that fires closes the group like an
unwind (reason = the rule). `stop_loss`/`take_profit` are net P&L in currency against the legs'
fill prices (costs excluded); the premium rules use the net premium of one lot at the marks.
They are checked once every leg has filled and need `marks` (symbol to last price) in
`update_structures`; a missing mark raises `structure_missing_mark` rather than skipping the
rule. `close_at_ns`, `days_before_expiry` and `entry_timeout_ns` need no marks and apply to a
partly filled group too. `close_structure(group)` closes by hand.

### Order of events in a bar run

An order for another instrument, placed from the underlying's bar, fills on that instrument's
next bar (stamped with its last stepped bar), which for aligned bars is the same timestamp.

## Sizing

```python
from raptorbt import sizing
sizing.lots_for_risk(capital, 0.01, entry, stop, lot_size)        # 1% of capital at risk
sizing.lots_for_capital(capital, price, lot_size, fraction=0.2)
sizing.lots_for_percent_of_equity(equity, 10, price, lot_size)
sizing.lots_for_volatility(capital, 0.10, asset_vol, price, lot_size, max_leverage=2)
sizing.lots_for_margin(free_capital, margin_per_lot, buffer=0.1)
sizing.max_lots_by_margin(lambda n: structure.resolve(chain, U, spot, now, n).margin(model, px), free)
sizing.Sizer(ctx).risk(0.01, entry, stop, lot_size)               # ctx.equity / ctx.free_capital
```

All round down and return lots. Less than one lot raises `below_one_lot` (the fractional lots
are on `.value`); a non-positive or non-finite input raises `invalid_sizing_input`. Zero is
never returned.

## Differences from the reference implementation

Checked on a fixed chain by `tests/python/test_structures_differential.py` against
`golden/structures_reference.json` (numbers and strings only): the
same contracts, sides and lots for every structure above on 14 spots (ties included), and the
same units from the risk sizer. Intended differences:

* Structures are not intents. There is no `Structure` intent, `MultiLeg` or all-or-none
  placement by the engine; a group is ordinary orders and the failure policy above.
* `qty_from_risk` returned 0 for less than one lot; `lots_for_risk` raises `below_one_lot`.
* Missing contracts raise `InvalidParameter` (`structure_contract_missing`), not `KeyError`.
* Not ported: `roll`, `roll_intents`, `hedge_delta`, `flatten`/`flatten_group`/`cap_to_position`
  (intent-model helpers; `close_structure` replaces the group close).
* Delta-picked legs use `OptionChain.by_delta` (flat vol at the event time) rather than a
  `PricingConfig`; not compared against the reference picks.
