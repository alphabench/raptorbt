---
name: raptorbt-execution
description: >
    Where a strategy's orders go: the ExecutionAdapter protocol, requests and
    execution reports, the simulated and paper venues, how a venue's fills are
    booked as facts, the host's order check and order tags, and what differs
    between paper and the in-process matcher.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-orders
status: stable
---

# RaptorBT Execution

## 1. When to use this skill

Load it when you connect a strategy to a venue (a broker, an exchange, your
own simulator), when you must understand what a fill report does to the book,
when you validate on paper before real money, or when an order sent to a venue
behaves differently from the same order in a backtest. Running the node around
the venue (feeds, journal, recovery) and the broker mappings are
[live-trading](../live-trading/SKILL.md); order kinds are [orders](../orders/SKILL.md).

## 2. Domain overview

A strategy only queues orders. What happens to them is the **execution** of the
node that runs it. The same `Strategy` class runs under all three:

| execution | who matches | how a fill reaches the book |
| --- | --- | --- |
| `None` or `SimulatedExecution()` (a backtest; the default of `LiveNode`) | the engine's matcher, inside the step | booked directly |
| `PaperExecution()` | a shadow copy of the book, stepped with the same events | as an `ExecutionReport`, the route a broker's fill takes |
| your adapter (any object with `submit`, `cancel`, `modify`) | the venue | an `ExecutionReport` you hand to `LiveNode.push_report` |

**Requests are fire and forget.** `submit`, `cancel` and `modify` return when
the request is handed over, never when the venue acted. The outcome comes back
only as reports (`ExecutionReport.accepted`, `.fill`, `.rejected`,
`.cancel_ack`, `.replace_ack`, `.expired`, `.correction`, `.bust`, ...), which
may repeat, arrive late or out of order. The node applies them so that none of
that changes the outcome: status is derived from the facts reported (executions
keyed by `exec_id`, acknowledgements), never from whichever arrived last.

**An adapter never touches the engine.** It must not call the node from inside
`submit`, `cancel` or `modify`; collect what the venue says and push it after
the call returns.

**What an adapter can say about a request.** Raise `OrderRefused` when the venue
certainly will not take it (the order is rejected with your message as the
reason), `Unsupported` when it has no equivalent (rejected as
`unsupported: ...`). Any other exception means *it may or may not have arrived*:
it propagates unchanged out of the push that caused it, the order stays
`Submitted`, and a report or a reconciliation settles it. Never raise
`OrderRefused` for a timeout.

**Identity.** The venue-facing id is `s{strategy}-{n}` (`s0-0`, `s0-1`, ...),
counted per strategy, so a replay of the same decisions sends the same ids and
an id is never sent twice. The id the strategy chose (`"O-0"` or its own
`client_id=`) is in `request.tags["client_id"]`, and is the id its hooks see.
Reports are keyed by the venue-facing id.

**Fills are facts.** A fill report is booked at the quantity and price the
venue reported. No slippage is added, and no limit of the book can undo it: the
venue executed it. A fill that breaches a limit (`order_size`, `max_position`,
`max_exposure`, `insufficient_capital`, `position_open`, `reduce_only`) is
booked and announced as a risk event `fill_breach` with the limit named. A fill
the book cannot book yet (a reduce-only fill that arrives before the fill that
opens its position) is kept, announced as `fill_deferred`, journaled as
`unbooked_fill`, and booked automatically once the book can. The fee booked is
the book's own model (`BacktestConfig.fees` or `fee_schedule`), rounded to the
money scale; the fee on the report is kept on the order for reconciliation and
moves no cash.

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.execution.ExecutionAdapter` | The `Protocol`: `submit(request)`, `cancel(client_order_id)`, `modify(client_order_id, *, quantity=None, limit_price=None, trigger_price=None)` (`quantity` is the new total). `isinstance` checks the three methods. |
| `raptorbt.execution.OrderRequest` | What `submit` receives, immutable: `client_order_id`, `symbol`, `side`, `quantity` (units), `order_type`, `limit_price`, `trigger_price`, `trail_offset`, `trail_kind`, `trail_limit_offset`, `time_in_force`, `expire_ns`, `strategy_id`, `account_id`, `reduce_only`, `post_only`, `parent_client_order_id`, `tags`, `submitted_ns`. |
| `raptorbt.execution.ExecutionReport` | What a venue says, built with the static constructors above (`fill(exec_id, qty, price, ts_ns, fee=)`). Fields: `kind`, `ts_ns`, `exec_id`, `qty`, `price`, `fee`, `reason`, `venue_order_id`. |
| `raptorbt.execution.OrderRefused`, `Unsupported`, `Fenced`, `AdapterError` | The errors an adapter raises on purpose; the first three are subclasses of `AdapterError`. `Fenced` is for epochs ([live-trading](../live-trading/SKILL.md)). |
| `raptorbt.execution.SimulatedExecution`, `PaperExecution` | The in-process matcher and the simulated exchange. Neither takes arguments; they follow the node's configuration. |
| `LiveNode(..., execution=, venue_account=, protective="venue" or "local", order_check=, warmup=)` | Where orders go; the account name written on every request; who holds stops and targets; the host's check; what warm-up bars do. |
| `LiveNode.push_report(client_order_id, report, *, arrival_ns)` | Apply a report now, as an event in the node's stream. Returns `"applied"`, `"duplicate"` or `"late"`. |
| `raptorbt.strategy.order_check.OrderProposal`, `OrderRefusal` | What the check sees (`symbol`, `strategy_index`, `client_id`, `side`, `kind`, `units`, `size_frac`, `limit_price`, `trigger_price`, `tif`, `reduce_only`, `post_only`, `tags`, `ts_ns`) and a refusal with a code. |
| `raptorbt.FeeSchedule`, `raptorbt.RateCard`, `raptorbt.costs.FeeSchedule`, `raptorbt.costs.RateCard`, `raptorbt.indian_cost_schedule` | The fee model the book charges for a venue fill, as for a backtest (see [backtesting](../backtesting/SKILL.md)). |

Order events a strategy hears are those of [orders](../orders/SKILL.md)
(`order_accepted`, `order_filled`, `order_canceled`, `order_expired`,
`order_rejected`). `fill_breach` and `fill_deferred` arrive in `on_risk_event`
with `event.reject_reason` naming the limit (breach) or the cause (deferred).
What each report does:

| report | effect |
| --- | --- |
| `accepted` | records the venue's order id; a different id later is an error |
| `fill` | adds one execution keyed by `exec_id`; implies acceptance; refused if it would over-fill, or if the same `exec_id` repeats with different terms |
| `rejected` | `Rejected`; an error if any fill was ever reported for the order |
| `cancel_ack` | `Canceled`; `late` once the order filled or expired |
| `replace_ack` | applies new total quantity, limit and trigger |
| `cancel_rejected`, `replace_rejected` | clears the pending request |
| `expired` | `Expired` (outranks a cancel) |
| `correction`, `bust` | restates or removes one execution; [live-trading](../live-trading/SKILL.md) says what can be taken back |

## 4. Standard workflow

1. Develop against the default (no `execution=`). Everything the in-process
   matcher accepts is turned into real venue orders when you move to a venue.
2. Run the same feed through `PaperExecution()` and compare trades. They match
   except for the cases in section 7.
3. Write your adapter: three methods, a way to translate `OrderRequest` into
   the venue's call, and a place where the venue's messages become
   `ExecutionReport`s. Use `OrderRefused` and `Unsupported` for what is
   certainly not taken; let everything uncertain raise.
4. Deliver reports from the node's thread, after the call that caused them
   returned: `node.push_report(request.client_order_id, report, arrival_ns=now)`.
5. Install an `order_check` for the rules your platform owns (price bands,
   instruments you do not allow). Without a venue the same check runs in a
   backtest, so the rule is tested before it is live.
6. Before real money: [live-trading](../live-trading/SKILL.md) for the journal
   and recovery, [deployment](../deployment/SKILL.md) for the checklist.

## 5. Examples

A venue of your own. The venue accepts, fills in two parts, repeats one
execution and reports a fee. The node books the venue's prices exactly (no
slippage), ignores the repeat, and charges its own fee model:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import ExecutionAdapter, ExecutionReport
from raptorbt.strategy import orders

MIN = 60_000_000_000


class Venue:
    namespace = "N1"
    def __init__(self):
        self.requests = []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass


class Buyer(Strategy):
    def __init__(self):
        super().__init__()
        self.heard = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Limit(side="buy", price=99.0, quantity=10, tags={"basket": "b-1"}))
        if ctx.idx == 2:
            self.submit_order(orders.Market(side="sell", quantity=10, reduce_only=True))

    def on_order_event(self, ctx, event):
        self.heard.append((event.kind, event.client_order_id, event.price, event.size))


venue = Venue()
assert isinstance(venue, ExecutionAdapter)
strategy = Buyer()
node = LiveNode([strategy], ["AAA"], BacktestConfig(initial_capital=1_000_000.0, fees=0.001, slippage=0.01),
                bar_stamp="close", execution=venue)
node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)

(request,) = venue.requests
assert (request.client_order_id, request.order_type, request.quantity) == ("s0-0", "limit", 10.0)
assert request.tags == {"client_id": "O-0", "basket": "b-1"}   # the strategy's own id rides in the tags

cid = request.client_order_id
assert node.push_report(cid, ExecutionReport.accepted("V1", MIN), arrival_ns=MIN) == "applied"
assert node.push_report(cid, ExecutionReport.fill("X1", 4.0, 99.0, 2 * MIN), arrival_ns=2 * MIN) == "applied"
assert node.push_report(cid, ExecutionReport.fill("X1", 4.0, 99.0, 2 * MIN), arrival_ns=2 * MIN) == "duplicate"
assert node.push_report(cid, ExecutionReport.fill("X2", 6.0, 98.5, 2 * MIN), arrival_ns=2 * MIN) == "applied"
assert strategy.heard == [("order_accepted", "O-0", None, None),
                          ("order_filled", "O-0", 99.0, 4.0), ("order_filled", "O-0", 98.5, 6.0)]
(position,) = node.positions(0)["AAA"]
assert (position.size, round(position.entry_price, 4)) == (10.0, 98.7)    # the venue's average, not slipped

node.push_bar("AAA", 2 * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
node.push_bar("AAA", 3 * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
close_id = venue.requests[1].client_order_id
node.push_report(close_id, ExecutionReport.fill("X3", 10.0, 110.0, 3 * MIN, fee=99.0), arrival_ns=3 * MIN)
trade = node.finish()[0].result.trades()[0]
assert trade.exit_price == 110.0 and trade.exit_reason == "Order"
assert round(trade.fees, 2) == 2.09     # 0.1% of 987 and of 1100, each leg rounded to paise; not the reported 99.0
```

What a request can come to. Refusals are named, an unsupported kind is named,
and any other exception means unknown: it raises out of the push and the order
stays unconfirmed:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import OrderRefused, Unsupported
from raptorbt.strategy import orders

MIN = 60_000_000_000


class Venue:
    namespace = "N1"
    def __init__(self, answers):
        self.answers, self.taken = answers, []

    def submit(self, request):
        if request.client_order_id in self.answers:
            raise self.answers[request.client_order_id]
        self.taken.append(request.client_order_id)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass


class Three(Strategy):
    def __init__(self):
        super().__init__()
        self.rejected = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            for qty in (1, 2, 3):
                self.submit_order(orders.Market(side="buy", quantity=qty))

    def on_order_rejected(self, ctx, event):
        self.rejected.append((event.client_order_id, event.reject_reason))


strategy = Three()
venue = Venue({"s0-0": OrderRefused("price band"), "s0-1": Unsupported("no such type")})
node = LiveNode([strategy], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=venue)
node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
assert strategy.rejected == [("O-0", "price band"), ("O-1", "unsupported: no such type")]
assert venue.taken == ["s0-2"]                       # the third reached the venue and is waiting

broken = Venue({"s0-0": ConnectionError("connection reset")})
node = LiveNode([Three()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=broken)
try:
    node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
    raise AssertionError("the exception must propagate")
except ConnectionError as e:
    assert str(e) == "connection reset"
```

Fills are facts. A limit says no more than 5 units; the venue filled 10. The
position follows the venue and the breach is reported. A reduce-only fill that
arrives before the entry it reduces is kept, journaled, and booked when the
entry arrives:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders

MIN = 60_000_000_000


class Venue:
    namespace = "N1"
    def __init__(self):
        self.requests = []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass


class Trader(Strategy):
    def __init__(self, script):
        super().__init__()
        self.script, self.heard = script, []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.script(self)

    def on_order_event(self, ctx, event):
        self.heard.append((event.kind, event.client_order_id))

    def on_risk_event(self, ctx, event):
        self.heard.append((event.kind, event.reject_reason))


def node_for(script, **config):
    venue, journal, strategy = Venue(), rj.Journal({"run": "demo"}, memory=True), Trader(script)
    node = LiveNode([strategy], ["AAA"], BacktestConfig(fees=0.0, **config), bar_stamp="close",
                    execution=venue, journal=journal)
    node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
    return node, venue, journal, strategy


# The venue filled 10 although the book allows 5.
node, venue, _, strategy = node_for(
    lambda s: s.submit_order(orders.Market(side="buy", quantity=10), client_id="in"), max_position_qty=5.0)
cid = venue.requests[0].client_order_id
node.push_report(cid, ExecutionReport.fill("X1", 10.0, 101.0, MIN), arrival_ns=MIN)
assert strategy.heard == [("order_filled", "in"), ("fill_breach", "max_position")]
assert node.positions(0)["AAA"][0].size == 10.0                       # the venue's position, not the limit's


def entry_and_exit(s):
    s.submit_order(orders.Market(side="buy", quantity=5), client_id="in")
    s.submit_order(orders.Market(side="sell", quantity=5, reduce_only=True), client_id="out")


# The exit's fill arrives first.
node, venue, journal, strategy = node_for(entry_and_exit)
entry, exit_ = [r.client_order_id for r in venue.requests]
node.push_report(exit_, ExecutionReport.fill("X1", 5.0, 101.0, MIN), arrival_ns=MIN)
assert strategy.heard[0][0] == "fill_deferred"
assert node.positions(0)["AAA"] == []
(unbooked,) = [r for r in journal.since(0) if r.kind == "unbooked_fill"]
assert unbooked.client_order_id == exit_ and unbooked.payload["exec_id"] == "X1"
node.push_report(entry, ExecutionReport.fill("E1", 5.0, 100.0, MIN), arrival_ns=MIN)
assert [h for h in strategy.heard if h[0] == "order_filled"] == [("order_filled", "in"), ("order_filled", "out")]
assert node.positions(0)["AAA"] == []                                 # flat: both fills booked, in order
```

A cancel is a request, not a result. Until the venue acknowledges it the order
can still fill, and every such fill is real; a fill reported after the
acknowledgement is `late` but still counts, because the venue executed it:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders

MIN = 60_000_000_000


class Venue:
    namespace = "N1"
    def __init__(self):
        self.requests, self.cancels = [], []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        self.cancels.append(client_order_id)

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass


class Canceller(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Limit(side="buy", price=95.0, quantity=10), client_id="L")
        if ctx.idx == 1:
            self.cancel_order("L")


venue = Venue()
node = LiveNode([Canceller()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=venue)
node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
cid = venue.requests[0].client_order_id
node.push_report(cid, ExecutionReport.accepted("V1", MIN), arrival_ns=MIN)
node.push_bar("AAA", 2 * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
assert venue.cancels == [cid]                                         # asked, not yet done
assert node.push_report(cid, ExecutionReport.fill("X1", 4.0, 95.0, 2 * MIN), arrival_ns=2 * MIN) == "late"
assert node.push_report(cid, ExecutionReport.cancel_ack(2 * MIN + 1), arrival_ns=2 * MIN + 1) == "applied"
assert node.push_report(cid, ExecutionReport.fill("X2", 1.0, 95.0, 2 * MIN), arrival_ns=3 * MIN) == "late"
assert node.positions(0)["AAA"][0].size == 5.0                        # both fills count
```

The host's order check and tags. The check sees every typed order once, before
the engine and before any venue; a code refuses the order with that reason.
`enter()` and `close_position()` carry no order for it to see, so with a check
installed they are refused (use `submit_order`):

```python
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.errors import InvalidOrder
from raptorbt.strategy import orders

MIN = 60_000_000_000


def check(order, ctx):
    if order.units is not None and order.units > 100:
        return "too_big"
    if order.tags.get("basket") == "blocked":
        return "basket_blocked"
    return None


class Venue:
    namespace = "N1"
    def __init__(self):
        self.requests = []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass


class Sender(Strategy):
    def __init__(self):
        super().__init__()
        self.refused = []

    def on_bar(self, ctx):
        self.submit_order(orders.Market(side="buy", quantity=500))
        self.submit_order(orders.Market(side="buy", quantity=5, tags={"basket": "blocked"}))
        self.submit_order(orders.Market(side="buy", quantity=5, tags={"basket": "ok"}))

    def on_order_rejected(self, ctx, event):
        self.refused.append((event.client_order_id, event.reject_reason))


venue, strategy = Venue(), Sender()
node = LiveNode([strategy], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close",
                execution=venue, order_check=check)
node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
assert strategy.refused == [("O-0", "too_big"), ("O-1", "basket_blocked")]
assert [r.tags["basket"] for r in venue.requests] == ["ok"]           # only the third left the engine


class Intent(Strategy):
    def on_bar(self, ctx):
        self.enter()


node = LiveNode([Intent()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", order_check=check)
try:
    node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
    raise AssertionError("an intent has no order for the check to see")
except InvalidOrder as e:
    assert e.code == "order_check_intent"
```

Paper against in-process. The same strategy on the same feed makes the same
trades, prices, sizes and fees; the only visible difference in this run is that
a close is an order, so its trade ends with reason `Order` instead of `Signal`:

```python
import numpy as np
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders

MIN = 60_000_000_000
close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0.0, 0.5, 60))
bars = [((i + 1) * MIN, close[i] - 0.1, close[i] + 0.5, close[i] - 0.5, close[i], 1000.0) for i in range(60)]


class Plan(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 5:
            self.enter(size_frac=0.5)
        if ctx.idx == 12:
            self.close_position()
        if ctx.idx == 20:
            self.submit_order(orders.Limit(side="buy", price=close[20] - 0.3, quantity=10,
                                           stop_price=close[20] - 3.0, target_price=close[20] + 2.0))


def trades(execution):
    node = LiveNode([Plan()], ["AAA"], BacktestConfig(initial_capital=100_000.0, fees=0.0005, slippage=0.0003),
                    bar_stamp="close", execution=execution)
    for bar in bars:
        node.push_bar("AAA", *bar)
    return node.finish()[0].result.trades()


plain, paper = trades(None), trades(PaperExecution())
shape = lambda ts: [(t.entry_idx, t.entry_price, t.exit_idx, t.exit_price, t.size, t.fees) for t in ts]
assert shape(plain) == shape(paper) and len(plain) == 2
assert [t.exit_reason for t in plain] == ["Signal", "EndOfData"]
assert [t.exit_reason for t in paper] == ["Order", "EndOfData"]
```

## 6. Composition

- A real venue is a [live-trading](../live-trading/SKILL.md) node plus a broker
  mapping from `raptorbt.adapters`; the journal that makes a crash survivable
  is there too.
- The order kinds, time in force and refusal reasons are [orders](../orders/SKILL.md);
  the same limits act on a venue's orders ([risk](../risk/SKILL.md)).
- `warmup="muted"` (the default for your own adapter) keeps history from
  reaching a venue: warm-up bars update indicators and state, and every order
  they queue is discarded.
- What a host must provide and decide before it relies on any of this:
  [deployment](../deployment/SKILL.md). Errors by symptom:
  [troubleshooting](../troubleshooting/SKILL.md).

## 7. Constraints and correctness

- **Order validity is the exchange segment's.** `tif="auto"` is `day` on an instrument with a session or on a day-only segment (NSE equity derivatives, currency, BSE, MCX), `gtc` otherwise. NSE equity cash and NSE commodity hold `gtc`/`gtd` natively (a commodity `gtd` no later than the contract's maturity); an explicit `gtc`/`gtd` anywhere else is refused (`order_validity_unsupported`). A broker that resubmits an order each day to imitate `gtc` is the host's business, not the exchange's validity.

**Everything a backtest accepts becomes real orders.** Sizes given as
`size_frac`, or no size ("the whole position"), are resolved to units when the
order is sent, from the book on that event. `enter()` becomes a market order,
`close_position()` a reduce-only market order, stops and targets (on an order,
an entry, or configured on the position) become protective orders linked
one-cancels-other, and the session square-off, `Twap` slices and `halt` send
their orders. `protective="venue"` (default) has the venue hold them (a leg the
adapter answers `Unsupported` for is watched by the engine instead); `"local"`
has the engine watch the levels and send a reduce-only market order, which fills
at the event's price, not at the stop level.

**Paper is not identical to the in-process matcher.** Where it differs, a test in
`tests/python/test_execution.py` pins exactly what:

1. The excursions (`mae_*`, `mfe_*`) of an entry intent filled at the next open
   start from the fill's price on a paper node.
2. An `enter()` held for the next print and a typed market order made in the
   same step: in-process the entry (a signal) fills first, at a venue they fill
   in the order they were sent. Use one form for one instrument in one step.

Hedging (`oms_type="hedging"`) is equal to in-process: a close, protection or
flatten order names the position it ends (`OrderRequest.position_id`). A venue
that nets positions cannot do that: a `BrokerAdapter` has
`supports_hedging = False`, and a hedging node on one is refused when it is
built (`hedging_unsupported_by_venue`).

Also: closes and protection exit with reason `Order`, not `Signal`, `StopLoss`
or `TakeProfit`; the engine's own orders are extra order records; an order the
matcher would reject after it had partly filled is cancelled instead (a venue
does not reject what it has traded).

**Refused when the node is built:** ATR-based stops on a live node,
`liquidate_on_margin_call`, and shared account groups with a venue. Position
stops, targets and the square-off are venue-managed; a paper run that changes a
position without an order raises `paper_unsolicited_position_change`.

- `quantity` in an `OrderRequest` is always units. A mapping to a venue that
  counts lots converts at its edge.
- A report for an order the node never sent raises `unknown_client_order_id`.
  The orders an exit rule the venue holds will fire are the exception: the node
  registers them when `place_exit_rule` succeeds, and their fills are booked.
- Resting orders in a venue run work from the next event. A cancel asked twice
  is asked of the venue once; the same id with different terms is refused
  (`duplicate_client_order_id`).
- An adapter that raises `OrderRefused` from `cancel` or `modify` gets the
  request rejected and the order keeps its state.
- `order_check` is pure: it must not submit or cancel. An exception it raises
  stops the run, never reads as acceptance. A code is 1 to 64 bytes and at most
  1024 distinct codes exist per process, so codes are a small stable set, not
  ids. `Twap` slices are not shown to the check; their parent is.
- Tags: at most 16 pairs, keys 1 to 64 bytes, values up to 256, and the key
  `client_id` is reserved. In-process they are validated and ignored.

## 8. Common mistakes

Reporting a fill for an order the node did not send:

<!-- skill: invalid -->
```python
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import ExecutionReport


class Idle(Strategy):
    pass


class Venue:
    namespace = "N1"
    def submit(self, request): ...
    def cancel(self, client_order_id): ...
    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None): ...


node = LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=Venue())
node.push_bar("AAA", 60_000_000_000, 100.0, 101.0, 99.0, 100.0, 1000.0)
node.push_report("s0-9", ExecutionReport.accepted("V", 0), arrival_ns=0)   # unknown_client_order_id
```

- Raising `OrderRefused` for a timeout. A refusal says the order will never
  trade; if it did reach the venue you now hold an order the node thinks is
  dead. Let uncertain failures raise anything else.
- Calling `node.push_report` from inside `submit`. Collect the reply and push
  it after the call returns.
- Keying a report by the strategy's id (`"O-0"`) instead of the request's
  `client_order_id` (`"s0-0"`).
- Repeating an `exec_id` with different numbers to "correct" a fill: refused.
  Use `ExecutionReport.correction` or `bust`.
- Expecting the venue's fee to move cash. The book charges its own model; the
  reported fee is kept for reconciliation.
- Treating `PaperExecution` results as the backtest's to the digit when the
  strategy mixes `enter()` with a typed order of the same instrument in one step.

## 9. Validation

1. `isinstance(adapter, ExecutionAdapter)`.
2. Drive the adapter from a scripted feed with `LiveNode` and assert the
   request fields you depend on (`order_type`, `quantity`, `tags`).
3. Push every report kind your venue can send, each twice and in a different
   order, and assert the same orders, positions and cash.
4. Run the strategy through `PaperExecution()` and compare trades with the
   in-process run; any difference must be one listed in section 7.
5. For a broker mapping, `raptorbt.adapters.conformance.run_conformance`
   ([live-trading](../live-trading/SKILL.md)) drives it against a hostile
   scripted venue.
6. After your settle step in tests, assert `node.health()["unconfirmed"] == []`
   (needs a journal).

## 10. Related skills

- [Orders](../orders/SKILL.md)
- [Live trading](../live-trading/SKILL.md)
- [Deployment](../deployment/SKILL.md)
- [Risk](../risk/SKILL.md)
- [Troubleshooting](../troubleshooting/SKILL.md)
