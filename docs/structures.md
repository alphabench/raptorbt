# Option structures and position sizing

`raptorbt.structures` describes multi-leg option positions declaratively, binds them to
contracts from an `OptionChain` at the event time, and places them as one group of ordinary
orders. `raptorbt.sizing` turns capital, risk, volatility or margin into a whole number of lots.
Neither adds an order type: a structure is a set of `Market`/`Limit` orders sent with
`submit_order`, hedge-first (below), so it behaves the same in `run_strategies`, the portfolio and
tick runners and `LiveNode` on the in-process engine, on `PaperExecution` and on a broker adapter.

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
on_leg_failure="unwind", margin_model=None, available=None, tif="auto", group_id=None)`

* `tif` is every leg's time in force. The default `"auto"` is the instrument-driven one: DAY on an
  instrument with a session, good-till-cancelled on one without. Pass `"gtc"` to keep legs working
  past the session.
* `order_type="limit"` sends each leg as a `Limit` at `prices[symbol]`. `prices` is also
  the reference for `net_limit` and for the margin check.
* `net_limit`: the net premium (debit positive) at those prices must not exceed it, else
  `structure_limit_breached` and nothing is sent.
* `margin_model` and `available` (both or neither): `ResolvedStructure.margin(...)` must fit
  `available`, else `structure_margin_exceeded` and nothing is sent. The engine's own margin
  check still applies to each leg.
* All checks run before any order is queued, so a group is sent whole or not at all.
* Two legs resolving to the same contract are refused; give one leg a `ratio`.

### Hedge-first

One model in every mode (a backtest, a node on `SimulatedExecution`, `PaperExecution`, a broker):

1. **Bought legs first.** The bought legs (shares and bought options: the hedges) go out at once.
2. **Sold legs follow the hedges.** A sold leg goes out when the hedges that cover it have filled
   (a sold call is covered by bought calls and bought shares, a sold put by bought puts, sold
   shares by bought calls). With partial fills a sold leg goes out for the share of its planned
   quantity that its least-filled covering hedge has filled, in whole lots, and again as more
   fills, so a sold quantity never runs ahead of its hedge cover. A sold leg nothing covers (a
   straddle, a strangle) goes out with the hedges. A ratio structure that sells more than it buys
   keeps its ratio: the uncovered excess is the structure's own.
3. **A failed sold leg closes the hedges.** See leg failure below. Every step is an order of its
   own in the journal: the sold legs, and the closes, tagged `close=<reason>`.
4. **Funding.** The sold leg is sent with the hedge held, so a margin model that groups option
   lines (`MarginModel.defined_risk()`, `spread_credit`, `from_spec`) gives the hedged
   requirement. A leg the account cannot fund is refused once, with the engine's reason (for
   example `insufficient_capital`), and that is the group's `failure`.
5. **Same step.** The sold leg and a close are sent in answer to a fill, so they meet the bar the
   fill came in (the close under `same_bar_close`, the open under `next_bar_open`), in a backtest
   and on a simulated venue alike; a real broker fills when it fills. The group is driven by the
   order events the strategy base class forwards to it: no call and no mixin is needed, and a
   subclass's own `on_order_event` runs after it.

Each order has the client id `"<group>:<leg index>"` (a sold leg sent in several parts has
`"<group>:<leg index>.<k>"` for the later ones) and `tags={"structure": group}`. The group id
defaults to `"<order_id_tag>-st<n>"`.

### Leg failure

A leg is *failed* when one of its orders was rejected, cancelled or expired (time in force).

| `on_leg_failure` | what happens on the first failed leg |
|---|---|
| `"unwind"` (default) | every order still working is cancelled; every leg that filled is closed with a `reduce_only` market order (sold legs first), tagged `close="leg_failure"`; the group ends `closed`, reason `leg_failure`, and `group.failure` is `(client id, status, engine reason)` |
| `"keep"` | nothing is undone; `status` reads `broken` and the strategy decides (`close_structure`) |

The close runs in the step of the failing order event. Every order still working is cancelled
first, including the remainder of a part-filled leg (an order is working until its fills reach its
size, whatever the venue says of the part that filled), and what filled is closed at once: the sold
legs, then the bought legs once no sold leg is open or could still fill, so a cover is never taken
off a position it covers. A fill that arrives after its leg was closed (a cancel that lost a race
with a fill), even after the group read `closed`, is closed in turn and the group reads `closing`
again. A close the venue refuses is tried again on a later step, never in the step it was refused
(an `update_structures` call or an order event on a later bar), at most three times, and then
reported as `stuck`; the cover stays on while a sold leg is stuck. A hedge that fails (for example one the account
cannot fund) fails the group the same way, and the sold legs it would have covered are never sent.

**A stop or target on a bought option of a margin group is the engine's, in every mode.** On an
account whose margin model margins option lines together (`MarginModel.defined_risk()`,
`spread_credit(...)`), the stop or target of a bought option is never sent to a
venue as a resting order (one would be refused, `insufficient_margin`, when it fired, leaving the
sold legs unprotected). When it fires, or when a square-off or liquidation reaches the leg, the
engine sends `reduce_only` market closes: the group's sold legs first, then the hedge once none is
left open. They are ordinary orders, priced by the run's fill timing (the bar's close under
`same_bar_close`; the next open under `next_bar_open`, or this bar's open when it opened through the
level), journaled, and the strategy hears an order event for each. A batch run, a node, the paper
venue and a broker send the same closes in the same order. A close the strategy itself asks for
(`close_position`) is not a forced exit: it is refused while it would leave a sold leg without a
hedge.

`group.status(cache)` / `strategy.structure_status(group)`: `submitted`, `working`, `filled`
(every leg), `broken`, `closing`, `closed`, `stuck`. `group.leg_status(cache, i)` is `waiting` (a
sold leg whose hedges have not filled), `submitted`, `working`, `filled`, `rejected`, `canceled`
or `expired`.

Fills are read from the order events as the quantity filled, so a part-filled hedge covers only
what it filled, and a close is for the quantity that filled. A structure group hears every order
event once, whether or not the strategy overrides `on_order_event` and calls `super()`. The order-event
hook is switched on when the strategy creates its first group (before any leg is queued); a strategy
that places no structures and does not override `on_order_event` is not called on order events at all.

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
