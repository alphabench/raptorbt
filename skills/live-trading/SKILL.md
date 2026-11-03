---
name: raptorbt-live-trading
description: >
    Running strategies on a pushed feed with LiveNode: arrival times, warm-up,
    adopted positions and corporate actions, the journal, checkpoint, restore,
    reconciliation and fencing, broker adapters and their conformance kit,
    percent and group exits, continuous futures, and the instruments a context
    answers for.
version: 1.0
api_version: 2.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
    - raptorbt-market-data
    - raptorbt-risk
    - raptorbt-execution
status: stable
---

# RaptorBT Live Trading

## 1. When to use this skill

Load it when a strategy must run on a feed you push to it one event at a time,
on paper or against a broker; when a process that holds positions must survive
a crash; when you adopt an account that already holds positions; or when you
connect a retail broker. Where orders go and what a fill report does is
[execution](../execution/SKILL.md); what a host must decide before real money
is [deployment](../deployment/SKILL.md); a backtest of the same strategy is
[backtesting](../backtesting/SKILL.md).

## 2. Domain overview

A `LiveNode` hosts many strategies on one pushed feed, on the same Rust core as
`run_strategies`. A strategy sees what its own backtest would show it. The node
is **single-threaded**: pushes, reports and hooks happen in the order calls are
made, a push from inside a hook raises `runtime_busy`, and nothing is applied
from another thread. Memory does not grow with the feed (fixed rings of the
latest `history` rows per symbol; `curve_samples` bounds the equity curves).

**The feed.** `push_tick(symbol, ts, ltp, bid, ask, ..., arrival_ns=)`,
`push_bar(symbol, ts, o, h, l, c, v, arrival_ns=)`, `push_depth(...)`,
`advance_time(ts)` (fires due timers and alerts with no data) and
`push_report(...)`. Timestamps must not decrease per symbol. A pushed bar is
stamped as the node's `bar_stamp` says (`"open"` or `"close"`); always pass it.

**Three times.** Every event has the exchange's time (`ctx.event_ns`), the time
it reached the engine (`arrival_ns=`, read as `ctx.arrival_ns`; default the
event's own time) and the processing time (arrival plus
`processing_latency_ns`, read as `ctx.processing_ns`). Pass the real arrival
time: it decides what a strategy could have known and is journaled.

**Warm-up.** `warmup_bars={symbol: OHLCV arrays}` replays history at
construction; `node.warm_up(...)` does it later (always muted).
`warmup="muted"` runs the hooks so indicators and strategy state are primed but
discards every order and intent, holds no position, and sends nothing to a
venue. `warmup="live"` replays bars as ordinary bars (orders match). The default
is muted for your own `ExecutionAdapter`, live otherwise; a live warm-up that
makes an order against a real venue raises `unsupported_in_venue_run`.
`ctx.is_warmup` is true in every hook on a warm-up bar.

**Adopted positions.** `initial_positions={strategy: {symbol: seed}}`, with
`seed = {"quantity", "avg_price", "timestamp_ns"?, "margin_minor"?,
"cost_minor"?}`. `quantity` is signed (a short needs a margin account). There is
no order, fill, fee or entry event; the entry price is `avg_price`. On a margin
account `margin_minor` (an `int` in minor units of `money_decimals`) is the
margin the broker holds against the position, locked as stated: without it only
a fully funded account is adopted, because the margin a broker holds is not
derivable from quantity and price. On a cash account `cost_minor` is the cash
the position cost; longs only.

**Corporate actions.** A live node takes `adjustment_policy="raw"` only: prices
are as traded; at the ex-date a split or bonus rescales the position and a
dividend credits cash. Declare every action the node will see.

**Journal.** `LiveNode(journal=Journal(header, ...))` writes an append-only,
hash-chained record of everything the node is told, decides and sends, **before
it acts**: inputs (`tick`, `bar`, `depth`, `time`, `report`), decisions
(`command`, `refusal`, `halt`), requests (`adapter_request` before the adapter
is called, then `adapter_result`), and recovery (`reconcile`, `discrepancy`,
`resume`, `fence`, `finish`). Records are canonical JSON with no float (a float
is its shortest decimal string), chained by `sha256(prev || record)`. Sinks: a
file (frames of length, crc32 and bytes; a torn last frame is cut on reopen), a
bounded memory, and a callback that receives each record before the node acts,
so you can be the durable store. A sink that fails closes the journal and the
node refuses to go on (`journal_failed`) *before* the engine acts. `fsync` is
`"every"`, `"orders"` (default: everything on the path of an order), an integer
`N`, or `"never"`; a killed process loses nothing under any policy, a power loss
can take what the policy did not force.

**Recovery.** The engine is deterministic, so a node is rebuilt by replaying its
journal into a fresh node with the adapter muted, checking that every record the
replay makes equals the one stored; a difference raises `replay_diverged`. A
request that was journaled but has no result was never confirmed: recovery does
**not** send it again. It reconciles with the venue instead. The restored node
is in state `"live"`, or `"recovering"` when the venue and the node disagree
(`missing_order`, `unknown_order`, `order_mismatch`, `position_mismatch`,
`cash_mismatch`); in `recovering`, market data and `finish` are refused
(`recovering`) while reports, `resend`, `reconcile` and `annotate` are allowed.
Restore time grows with the journal since `seq_base`, so a long-running host
starts a new journal per session and carries strategy state with
`Strategy.state()`.

**Fencing.** `LiveNode(epoch=E)`. An adapter with `fence(epoch)` raises
`Fenced` when a newer epoch owns the account. The node checks before every
request and on `heartbeat()`; after a `Fenced` it writes a `fence` record,
sends nothing more, and every push raises `fenced`. The epoch is a number you
assign and the venue enforces; it stops this process sending stale work, it
cannot recall a request in flight and does not stop another process.

**Broker adapters.** `raptorbt.adapters` is the layer between the engine's
adapter contract and a retail broker's API: `KiteAdapter`, `UpstoxAdapter` and
`DhanAdapter` over a common `BrokerAdapter` (outbox, executor, rate limits,
retries, duplicate-order guard, fill deltas from cumulative figures). Details in
section 3 and 7.

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.LiveNode(strategies, symbols, config=None, *, bar_stamp=, primary_bars=, warmup_bars=, warmup=, initial_positions=, corporate_actions=, adjustment_policy=, execution=, venue_account=, protective=, order_check=, instruments=, bindings=, processing_latency_ns=, journal=, epoch=, strategy_states=, accounts=, history=, curve_samples=, ...)` | The node. Strategies are instances, classes, `(strategy, symbols)` pairs, numbered by position. |
| `push_tick`, `push_bar`, `push_depth`, `advance_time`, `warm_up`, `push_report` | The only ways in. All but `warm_up` take `arrival_ns=` (`push_report` requires it). |
| `positions(i, symbol=None)`, `equity(i)`, `ctx(i)`, `is_halted(i)`, `halt(i, flatten=, reason=)`, `halt_account(name, ...)`, `memory_report()`, `finish()` | State and control. `finish()` ends the node and returns one `PortfolioResult` per strategy (`.result.trades()`, `.result.equity_curve()`); a finished node raises `node_finished`. |
| `raptorbt.journal.Journal(header, *, seq_base=0, chain=None, file=None, fsync="orders", memory=None, sink=None)` | The journal. `since(seq)`, `head()` returns `(count, chain, header)`, `count`, `sync()`. |
| `raptorbt.journal.read(path)`, `verify(records, header, chain=None)`, `open_for_recovery(path, ...)`, `chain_step`, `chain_start`, `canonical`, `report_from_payload`, `Record`, `Discrepancy` | Read a file (`header, seq_base, chain_from, records, torn_bytes`, checksums and chain verified), recompute the chain, reopen to recover. |
| `LiveNode.checkpoint(include_log=True)`, `LiveNode.restore(checkpoint, strategies, symbols, config, *, journal=, records=, execution=, reconcile=True, accept=None, on_discrepancy=, **the node's arguments)` | A versioned snapshot (journal head, state digest, each strategy's `state()`, the log) and the rebuild. `checkpoint=None` restores from `journal=` and `records=`. `reconcile="skip"` skips the venue and says so in the journal. |
| `LiveNode.reconcile(cash_tolerance=0.0)`, `resume(accept=())`, `resend(client_order_id)`, `annotate(ts, kind, detail)`, `state`, `state_digest()` | Compare with the venue, leave `recovering`, send an unanswered order again under its first id, write a note in the journal. |
| `LiveNode.heartbeat()`, `health()`, `fenced`, `epoch` | Check the epoch now; a dict for a supervisor: `state`, `epoch`, `fenced`, `finished`, `journal_count`, `chain`, `unconfirmed` (orders with no confirmed outcome; needs a journal), `clock_ns`. |
| `Strategy.state()`, `Strategy.load_state(state)`, `Strategy.on_discrepancy(ctx, d)` | Your own JSON-serialisable state across runs (`strategy_states=` hands it back before `on_start`); the recovery hook. |
| `raptorbt.CorporateAction.split/bonus/dividend` | Corporate actions as data. |
| `raptorbt.adapters.{KiteAdapter, UpstoxAdapter, DhanAdapter}` (modules `kite`, `upstox`, `dhan`), `BrokerAdapter`, `Session`, `Secret`, `Instrument`, `new_namespace`, `Executor`, `InlineExecutor`, `ThreadExecutor`, `ManualExecutor`, `RateLimiter`, `TokenBucket`, `Backoff`, `reconnect_loop`, `IdCodec`, `AdapterEvent`, `QueuedReport`, `OrderStatusUpdate`, `TradeRow`, `VenueOrder`, `VenuePosition`, `VenueBalances`, `VenueRefusal`, `Transient`, `Ambiguous`, `AuthExpired`, `MalformedRow` | The broker layer. A mapping translates and reads; the base owns everything else. |
| `raptorbt.adapters.conformance.run_conformance(factory)`, `ReferenceBinding`, `ScriptedVenue`, `assert_conforms` | Drives any adapter against a hostile scripted venue. Returns failures (empty means conforming). |
| `raptorbt.strategy.exits.ExitRule`, `ExitEvents`, `ExitRuleRequest`, `ExitHandle`; `Strategy.set_exit(ctx, rule, symbol= / symbols=, name= / group=)`, `cancel_exit(handle)` | Percent and absolute exits on a position or a named group of legs. |
| `raptorbt.strategy.RollEvent`, `RollEvents`, `continuous_series(contracts, specs, roll_days=, adjust=)`, `Future(of=, contract="continuous", roll_days=)` | Continuous futures with a roll rule, roll events, optional auto-roll, and one adjusted history. |
| `ctx.instrument(key=None)`, `ctx.option_chain(underlying=None)`, `ctx.role(name)` | The instruments of the run (`instruments={symbol: InstrumentSpec}`) and role bindings (`bindings={role: symbol}`) a strategy reads through its context. |

Adapter construction (all three): `Adapter(Session(token, api_key=?, refresh=?),
{engine_symbol: Instrument(exchange, symbol, key=?, lot_size=?, tick_size=?,
quantity_in_lots=?)}, namespace=new_namespace(), gtc="day" or "refuse",
reduce_only="best_effort" or "refuse", default_product=, account_id=,
fill_source="order_updates" or "trades", verifier=, trusted=, executor=,
limiter=, retention_s=, resend_if_absent=, ...)`. `namespace` is required.
Brokers' support is summarised in section 7.

## 4. Standard workflow

1. Build the node on a paper venue first (`execution=PaperExecution()`) with a
   journal, and push the day's recorded feed with real arrival times.
2. Adopt what the account already holds with `initial_positions`, declare the
   corporate actions you know of, and warm up with enough history for your
   longest indicator.
3. Decide the strategy's part of recovery: make `state()` return only what the
   engine cannot rebuild (a regime flag an operator set), and implement
   `on_discrepancy` if the strategy must react.
4. Choose the journal's durability (file, `fsync`) and keep a copy off the
   machine. Start a new journal per session; continue the chain with
   `Journal(header, seq_base=count, chain=head)`.
5. For a broker, build its adapter with a fresh `namespace`, authenticate every
   push channel, run `run_conformance` for any adapter of your own, and wire
   `adapter.deliver_to(node)` on the node's thread.
6. Kill the process at awkward moments in a rehearsal and restore from the
   journal. Assert `state_digest()` matches and that no order was sent twice.
7. Go through the checklist in [deployment](../deployment/SKILL.md).

## 5. Examples

Warm-up, arrival times and processing latency. Muted warm-up primes the
strategy and holds nothing; live warm-up trades the history:

```python
import numpy as np
from raptorbt import BacktestConfig, LiveNode, Strategy

MIN = 60_000_000_000
n = 5
history = dict(timestamps=(np.arange(n) + 1) * MIN, open=np.full(n, 100.0), high=np.full(n, 101.0),
               low=np.full(n, 99.0), close=np.full(n, 100.0), volume=np.full(n, 1000.0))


class Probe(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx):
        self.seen.append((ctx.idx, ctx.is_warmup, ctx.event_ns, ctx.arrival_ns, ctx.processing_ns))
        if ctx.idx in (1, 4):
            self.enter(size_frac=0.1)


def build(warmup):
    s = Probe()
    node = LiveNode([s], ["AAA"], BacktestConfig(fees=0.0, initial_capital=100_000.0), bar_stamp="close",
                    warmup_bars={"AAA": history}, warmup=warmup, processing_latency_ns=500)
    return s, node


primed, muted = build("muted")
assert muted.positions(0)["AAA"] == []                         # hooks ran, nothing was held
assert primed.seen[0] == (0, True, MIN, MIN, MIN + 500)         # (idx, is_warmup, event, arrival, processing)
_, live = build("live")
assert live.positions(0)["AAA"][0].entry_idx == 1              # the history traded

muted.push_bar("AAA", 6 * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0, arrival_ns=6 * MIN + 7_000)
assert primed.seen[-1] == (5, False, 6 * MIN, 6 * MIN + 7_000, 6 * MIN + 7_500)
```

Adopted positions and corporate actions. A signed short with the margin the
broker holds, and a long that meets a 2-for-1 split and a dividend at their
ex-dates (prices stay as traded, the position is rescaled, cash is credited):

```python
from raptorbt import BacktestConfig, CorporateAction, LiveNode, Strategy
from raptorbt.errors import InvalidParameter

DAY = 86_400_000_000_000
T0 = 1_767_225_600 * 10**9


class Idle(Strategy):
    pass


short = LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0, initial_capital=100_000.0), account_type="margin",
                 leverage=5.0, bar_stamp="close",
                 initial_positions={0: {"AAA": {"quantity": -10.0, "avg_price": 100.0, "margin_minor": 200_000}}})
(position,) = short.positions(0)["AAA"]
assert (position.size, position.direction, position.entry_price) == (10.0, -1, 100.0)
short.push_bar("AAA", T0, 100.0, 101.0, 89.0, 90.0, 1000.0)
assert short.finish()[0].result.equity_curve()[-1] == 100_100.0       # the short gained as price fell from 100

try:
    LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0), account_type="margin", leverage=5.0, bar_stamp="close",
             initial_positions={0: {"AAA": {"quantity": -10.0, "avg_price": 100.0}}})
except ValueError as e:                        # no margin stated on a leveraged account: not derivable
    assert "posted margin" in str(e)

actions = [CorporateAction.split("AAA", T0 + 2 * DAY, 2.0), CorporateAction.dividend("AAA", T0 + 3 * DAY, 1.0)]
node = LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0, initial_capital=100_000.0), bar_stamp="close",
                corporate_actions=actions, adjustment_policy="raw",
                initial_positions={0: {"AAA": {"quantity": 10.0, "avg_price": 100.0}}})
for i, price in enumerate([100.0, 101.0, 51.0, 51.5, 52.0]):
    node.push_bar("AAA", T0 + i * DAY, price, price + 1, price - 1, price, 1000.0)
(position,) = node.positions(0)["AAA"]
assert (position.size, position.entry_price) == (20.0, 50.0)           # rescaled at the split's ex-date
assert list(node.finish()[0].result.equity_curve()) == [100_000.0, 100_010.0, 100_020.0, 100_050.0, 100_060.0]

try:
    LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", corporate_actions=actions)
except InvalidParameter as e:
    assert e.code == "invalid_adjustment_policy"      # actions need a declared policy; a live node takes "raw"
```

Crash and recover from a file journal. The process dies; a fresh process
reopens the file, replays it into a new node with the adapter muted, and ends
in the same account state without sending anything again. The chain verifies
end to end:

```python
import os
import tempfile

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders

MIN = 60_000_000_000
CONFIG = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
HEADER = {"run": "r-1", "algo": "demo"}


class Exchange:
    namespace = "N1"
    can_fence = True
    """A venue that outlives the node. It accepts orders and answers what it holds."""

    def __init__(self):
        self.orders, self.sent, self.outbox = {}, [], []

    def submit(self, request):
        self.sent.append(request.client_order_id)
        if request.client_order_id not in self.orders:        # idempotent on the client id, as venues are
            self.orders[request.client_order_id] = {"client_order_id": request.client_order_id, "open": True,
                                                    "symbol": request.symbol, "quantity": request.quantity}
            self.outbox.append((request.client_order_id, ExecutionReport.accepted("V" + request.client_order_id, request.submitted_ns)))

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def open_orders(self):
        return [dict(o) for o in self.orders.values() if o["open"]]

    def positions(self):
        return {}

    def balances(self):
        return {"cash": 1_000_000.0}

    def deliver(self, node):
        for cid, report in list(self.outbox):
            node.push_report(cid, report, arrival_ns=10**15)
            self.outbox.remove((cid, report))


class Trader(Strategy):
    def __init__(self):
        super().__init__()
        self.count = 0

    def on_bar(self, ctx):
        self.count += 1
        if self.count % 4 == 2:
            self.submit_order(orders.Limit(side="buy", price=90.0, quantity=5))


def push(node, i):
    node.push_bar("AAA", (i + 1) * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)


path = os.path.join(tempfile.mkdtemp(), "run.journal")
venue = Exchange()
node = LiveNode([Trader()], ["AAA"], CONFIG, bar_stamp="close", execution=venue, epoch=1,
                journal=rj.Journal(HEADER, file=path, fsync="orders"))
for i in range(10):
    push(node, i)
    venue.deliver(node)
want = node.state_digest()
assert venue.sent == ["s0-0", "s0-1", "s0-2"]
del node                                                   # the process is gone

journal, records = rj.open_for_recovery(path, fsync="orders", memory=True)
restored = LiveNode.restore(None, [Trader()], ["AAA"], CONFIG, journal=journal, records=records,
                            execution=venue, bar_stamp="close", epoch=2)   # a newer epoch: the old node is fenced
assert restored.state == "live" and restored.state_digest() == want
assert venue.sent == ["s0-0", "s0-1", "s0-2"]              # recovery sent nothing

for i in range(10, 14):                                    # and the run goes on, ids continuing
    push(restored, i)
assert venue.sent[-1] == "s0-3"
restored.journal.sync()
header, seq_base, chain_from, again, torn = rj.read(path)
assert torn == 0 and rj.verify(again, header) == restored.journal.head()[1]
```

A request with no confirmed outcome. The process died after the request was
durable and before the adapter answered. Recovery does not send it again: the
node comes back in `recovering`, names the order, and refuses market data until
you have asked the venue. Here the venue never got it, so it is sent again under
its first id and recovery is accepted:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.errors import StateError
from raptorbt.strategy import orders

MIN = 60_000_000_000
CONFIG = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
HEADER = {"run": "r-2"}


class Died(Exception):
    pass


class Exchange:
    namespace = "N1"
    can_fence = True
    def __init__(self):
        self.orders, self.sent = {}, []

    def submit(self, request):
        self.sent.append(request.client_order_id)
        self.orders[request.client_order_id] = {"client_order_id": request.client_order_id, "open": True}

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def open_orders(self):
        return [dict(o) for o in self.orders.values() if o["open"]]

    def positions(self):
        return {}

    def balances(self):
        return {"cash": 1_000_000.0}


class Buyer(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Limit(side="buy", price=90.0, quantity=5))


stored = []


def sink(record):                                  # the durable store; it dies as the result would be written
    if record.kind == "adapter_result":
        raise Died("killed")
    stored.append(record)


class Crashing(Exchange):
    def submit(self, request):
        raise Died("killed")


node = LiveNode([Buyer()], ["AAA"], CONFIG, bar_stamp="close", execution=Crashing(), journal=rj.Journal(HEADER, sink=sink))
try:
    for i in range(3):
        node.push_bar("AAA", (i + 1) * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
except Died:
    pass
assert [r.kind for r in stored][-1] == "adapter_request"      # the request is durable; its result is not

venue, found = Exchange(), []
node = LiveNode.restore(None, [Buyer()], ["AAA"], CONFIG, journal=rj.Journal(HEADER, memory=True), records=list(stored),
                        execution=venue, bar_stamp="close", on_discrepancy=found.append)
assert venue.sent == []                                        # recovery never sends
assert node.state == "recovering" and node.health()["unconfirmed"] == ["s0-0"]
assert [(d.kind, d.client_order_id) for d in found] == [("missing_order", "s0-0")]
try:
    node.push_bar("AAA", 9 * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
except StateError as e:
    assert e.code == "recovering"

node.resend("s0-0")                                            # the venue does not hold it: same id, once
assert venue.sent == ["s0-0"] and node.reconcile() == []
node.resume()
assert node.state == "live"
```

Fencing across workers. A newer epoch takes the account; the older node is
stopped before it sends anything:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.errors import StateError
from raptorbt.execution import Fenced

MIN = 60_000_000_000


class Exchange:
    namespace = "N1"
    can_fence = True
    """The venue enforces the epoch."""

    def __init__(self):
        self.epoch, self.sent = 0, []

    def submit(self, request):
        self.sent.append(request.client_order_id)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def fence(self, epoch):
        if epoch < self.epoch:
            raise Fenced(f"epoch {self.epoch} owns the account")
        self.epoch = epoch


class Idle(Strategy):
    pass


venue = Exchange()
old = LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=venue, epoch=1)
old.heartbeat()
new = LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=venue, epoch=2)
new.heartbeat()
assert venue.epoch == 2 and not new.fenced

try:
    old.heartbeat()
except StateError as e:
    assert e.code == "fenced"
assert old.fenced and old.health()["fenced"]
try:
    old.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
except StateError as e:
    assert e.code == "fenced"                      # every later call is refused; nothing is ever sent
assert venue.sent == []
```

Checkpoint, a damaged checkpoint, and the next session. A checkpoint carries
the journal head, a digest of the state to check the rebuild against, and each
strategy's `state()`. The next session continues the chain and numbering and
hands strategy state back:

```python
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.errors import InvalidData, StateError
from raptorbt.strategy import orders

MIN = 60_000_000_000
CONFIG = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
HEADER = {"run": "r-3"}


class Venue:
    namespace = "N1"
    def submit(self, request):
        pass

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def open_orders(self):
        return []


class Trader(Strategy):
    def __init__(self):
        super().__init__()
        self.count = 0

    def on_bar(self, ctx):
        self.count += 1
        if self.count % 4 == 2:
            self.submit_order(orders.Limit(side="buy", price=90.0, quantity=5))

    def state(self):                                   # only what the engine cannot rebuild
        return {"count": self.count}

    def load_state(self, state):
        self.count = state["count"]


def push(node, i):
    node.push_bar("AAA", (i + 1) * MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)


journal = rj.Journal(HEADER, memory=True)
node = LiveNode([Trader()], ["AAA"], CONFIG, bar_stamp="close", execution=Venue(), journal=journal)
for i in range(6):
    push(node, i)

blob = node.checkpoint()                                           # bytes, with the log
restored = LiveNode.restore(blob, [Trader()], ["AAA"], CONFIG, journal=rj.Journal(HEADER, memory=True),
                            execution=Venue(), bar_stamp="close", reconcile="skip")
assert restored.state_digest() == node.state_digest()

small = node.checkpoint(include_log=False)                         # the host supplies the records instead
again = LiveNode.restore(small, [Trader()], ["AAA"], CONFIG, journal=rj.Journal(HEADER, memory=True),
                         records=journal.since(0), execution=Venue(), bar_stamp="close", reconcile="skip")
assert again.state_digest() == node.state_digest()

try:
    LiveNode.restore(blob[:-3], [Trader()], ["AAA"], CONFIG, journal=rj.Journal(HEADER, memory=True),
                     execution=Venue(), bar_stamp="close", reconcile="skip")
except InvalidData as e:
    assert e.code == "invalid_checkpoint"                          # truncated: refused, nothing rebuilt


class Other(Trader):                                               # not the strategy that wrote the journal
    def on_bar(self, ctx):
        super().on_bar(ctx)
        if self.count == 3:
            self.submit_order(orders.Limit(side="buy", price=80.0, quantity=1))


try:
    LiveNode.restore(blob, [Other()], ["AAA"], CONFIG, journal=rj.Journal(HEADER, memory=True),
                     execution=Venue(), bar_stamp="close", reconcile="skip")
except StateError as e:
    assert e.code == "replay_diverged"

count, chain, header = journal.head()                              # tomorrow: a new journal on the same chain
state = node.strategies[0].state()
trader = Trader()
tomorrow = LiveNode([trader], ["AAA"], CONFIG, bar_stamp="close", execution=Venue(), strategy_states=[state],
                    journal=rj.Journal(HEADER, seq_base=count, chain=chain, memory=True))
assert trader.count == 6 and tomorrow.journal.since(count)[0].seq == count
```

A broker through its adapter. The adapter is built with a fresh `namespace`
(the engine's ids restart at `s0-0` every run, so a tag alone never identifies
an order of this run) and a verifier for push channels. The transport here is a
fake; in production it is the default HTTP client. Fills arrive as the broker's
cumulative figures, become deterministic fill deltas, and a payload that fails
verification is dropped and named. Then the conformance kit:

```python
import hashlib
import json

from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.adapters import Instrument, Session, new_namespace
from raptorbt.adapters.conformance import ReferenceBinding, run_conformance
from raptorbt.adapters.http import HttpResponse
from raptorbt.adapters.kite import KiteAdapter, postback_verifier
from raptorbt.execution import AdapterError
from raptorbt.strategy import orders

MIN = 60_000_000_000
SECRET = "api-secret"


class Transport:
    """What the adapter would send over HTTP, recorded, and the broker's answer."""

    def __init__(self):
        self.sent = []

    def __call__(self, method, url, *, headers=None, params=None, form=None, body=None, timeout=10.0):
        self.sent.append((method, url, form))
        return HttpResponse(200, json.dumps({"status": "success", "data": {"order_id": "2211"}}).encode())


class Buyer(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Limit(side="buy", price=99.0, quantity=10))


transport = Transport()
adapter = KiteAdapter(Session("ACCESS", api_key="KEY"), {"INFY": Instrument("NSE", "INFY")}, transport=transport,
                      namespace=new_namespace(), gtc="day", verifier=postback_verifier(SECRET))
node = LiveNode([Buyer()], ["INFY"], BacktestConfig(fees=0.0, initial_capital=1_000_000.0), bar_stamp="close",
                execution=adapter)
node.push_bar("INFY", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)

method, url, form = transport.sent[0]
assert (method, url.rsplit("/", 1)[1]) == ("POST", "regular")
assert form["tradingsymbol"] == "INFY" and form["transaction_type"] == "BUY" and form["quantity"] == 10
assert form["tag"].endswith("s0x0")                         # the engine id behind this run's namespace


def postback(status, filled, average, checksum=True):
    body = {"order_id": "2211", "status": status, "tag": form["tag"], "exchange": "NSE", "tradingsymbol": "INFY",
            "quantity": 10, "filled_quantity": filled, "average_price": average, "price": 99.0,
            "order_timestamp": "2026-11-02 09:15:01", "exchange_update_timestamp": "2026-11-02 09:15:02"}
    raw = f"2211{body['order_timestamp']}{SECRET}"
    body["checksum"] = hashlib.sha256(raw.encode()).hexdigest() if checksum else "0" * 64
    return body


for body in (postback("OPEN", 0, 0.0), postback("OPEN", 4, 99.0), postback("OPEN", 4, 99.0),   # a repeat
             postback("COMPLETE", 10, 98.7), postback("COMPLETE", 10, 50.0, checksum=False)):
    adapter.handle_payload(body)
assert [e.kind for e in adapter.drain_events()] == ["payload_rejected"]
assert adapter.deliver_to(node) == ["applied", "applied", "applied"]      # accepted, fill of 4, fill of 6
(position,) = node.positions(0)["INFY"]
assert (position.size, round(position.entry_price, 4)) == (10.0, 98.7)

unauthenticated = KiteAdapter(Session("ACCESS", api_key="KEY"), {"INFY": Instrument("NSE", "INFY")},
                              transport=transport, namespace="N1")
try:
    unauthenticated.handle_payload(postback("OPEN", 0, 0.0))
except AdapterError as e:
    assert "unauthenticated" in str(e)

try:
    KiteAdapter(Session("ACCESS", api_key="KEY"), {}, transport=transport, namespace="")
except ValueError as e:
    assert "namespace is required" in str(e)

assert not run_conformance(ReferenceBinding)       # for your own adapter, pass a binding for it
```

Percent and group exits, and continuous futures. A rule on a group of legs
fires once and sends reduce-only market orders with ids `{rule}:x{n}`; they are
ordinary orders. A continuous role names the contract that is front under the
roll rule, `on_roll` reports the switch, and `auto_roll` moves the position
with a reduce-only close and an open (ids `{role}:roll{n}:c{k}` / `:o{k}`);
`ctx.instrument(ctx.role("fut"))` is the spec of the contract the role names now:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import PaperExecution
from raptorbt.strategy import Future, Holding, Requirements, Underlying, orders
from raptorbt.strategy.continuous import RollEvents
from raptorbt.strategy.exits import ExitEvents, ExitRule

MIN = 60_000_000_000


class Pair(ExitEvents, Strategy):
    def __init__(self):
        super().__init__()
        self.n, self.handle = 0, None

    def on_bar(self, ctx):
        if ctx.symbol != ctx.symbols[0]:
            return
        self.n += 1
        if self.n == 1:
            for symbol in ctx.symbols:
                self.submit_order(orders.Market(side="buy", quantity=10, instrument=symbol))
        if self.n == 2:
            self.handle = self.set_exit(ctx, ExitRule(take_profit=25.0), symbols=["AAA", "BBB"], name="pair")


closes = {"AAA": [100, 100, 100, 101, 102, 103, 103, 103], "BBB": [50, 50, 50, 50, 51, 52, 52, 52]}
strategy = Pair()
node = LiveNode([strategy], ["AAA", "BBB"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                execution=PaperExecution())
for i in range(8):
    for symbol in ("AAA", "BBB"):
        price = float(closes[symbol][i])
        node.push_bar(symbol, (i + 1) * MIN, price, price + 0.1, price - 0.1, price, 1000.0)
assert strategy.handle.state == "done" and strategy.handle.reason == "take_profit"
assert len(strategy.handle.close_ids) == 2 and all(":x" in c for c in strategy.handle.close_ids)
assert node.positions(0) == {"AAA": [], "BBB": []}               # both legs closed together

T0 = 1_700_000_000 * 10**9
EXPIRY_A, EXPIRY_B = T0 + 20 * MIN, T0 + 40 * MIN
specs = {"IDX": raptorbt.InstrumentSpec.equity("IDX"),
         "FA": raptorbt.InstrumentSpec.futures_contract("FA", EXPIRY_A, 1.0, underlying="IDX"),
         "FB": raptorbt.InstrumentSpec.futures_contract("FB", EXPIRY_B, 1.0, underlying="IDX")}


class Carry(RollEvents, Strategy):
    requirements = Requirements(
        roles={"index": Underlying(asset="index"), "fut": Future(of="index", contract="continuous", roll_days=5 / 1440)},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar"])
    auto_roll = True

    def __init__(self):
        super().__init__()
        self.rolls, self.first_spec, self.n = [], None, 0

    def on_bar(self, ctx):
        if ctx.symbol != "IDX":
            return
        self.n += 1
        if self.n == 1:
            self.first_spec = ctx.instrument(ctx.role("fut"))
            self.submit_order(orders.Market(side="buy", quantity=3, instrument=ctx.role("fut")))

    def on_roll(self, ctx, roll):
        self.rolls.append(roll)


carry = Carry()
node = LiveNode([carry], ["IDX", "FA", "FB"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                execution=PaperExecution(), instruments=specs, bindings={"index": "IDX"})
for i in range(35):                                              # ends before FB expires
    for k, symbol in enumerate(("IDX", "FA", "FB")):
        price = 100.0 + 5 * k
        node.push_bar(symbol, T0 + (i + 1) * MIN, price, price + 1, price - 1, price, 100.0)
assert carry.first_spec.symbol == "FA"
(roll,) = carry.rolls
assert (roll.role, roll.from_symbol, roll.to_symbol, roll.ts_ns) == ("fut", "FA", "FB", EXPIRY_A - 5 * MIN)
positions = node.positions(0)
# the roll keeps the notional, not the lot count: 3 x 105 = 315 buys floor(315 / 110) = 2 of FB
assert positions["FA"] == [] and positions["FB"][0].size == 2.0  # moved by real orders, not by assumption
```

## 6. Composition

- Orders and fills on a venue: [execution](../execution/SKILL.md). Limits that
  refuse an order or halt trading: [risk](../risk/SKILL.md).
- The `requirements` a strategy declares are what a host resolves into symbols
  before building a node (`resolve`, `requirements_hash`): [strategy](../strategy/SKILL.md)
  and [deployment](../deployment/SKILL.md).
- Feeds: [market-data](../market-data/SKILL.md), [subscriptions](../subscriptions/SKILL.md)
  and [timeframes](../timeframes/SKILL.md) (`primary_bars=`, `bar_stamp`).
- The shipped mappings are for Kite Connect v3, Upstox v3 and DhanHQ v2 only.

## 7. Constraints and correctness

- **Journal before action.** Nothing the engine acts on is unrecorded. A node
  with no `journal=` and no `epoch=` writes nothing and pays one flag test per
  push.
- **Never resend blindly.** An order whose request was journaled without a
  result stays `Submitted` until the venue's report or a reconciliation settles
  it. `resend` is for an order reconcile reported `missing_order` after the
  venue was asked; it uses the first request's id, so a venue that deduplicates
  on the client id takes it once.
- **Determinism is a requirement.** A strategy that reads the wall clock, a
  random source or the environment fails the replay (`replay_diverged`). So does
  a different feed, symbol list or warm-up. Seed any randomness from the
  journal's header or from `state()`.
- **Replay cost.** Restore replays every record since `seq_base`; the heap is
  not snapshotted (indicator banks and Python objects are not serialisable).
  Bound sessions. A bounded `memory=N` journal raises `journal_evicted` for
  records it dropped; the durable sink holds them.
- **No float in a record** and no secrets: tokens never appear in a request
  record; adapters hold them in `Secret`, which prints as `Secret(****)` and
  refuses to pickle.
- **Corrections and busts** of an execution that opened or added to a position
  that has not been reduced since are applied exactly (cash, margin lock, cost
  and entry fees come off in the minor units they were booked in). An exit, or
  an entry whose position was since reduced, is refused with
  `correction_not_reversible` before anything changes: reconcile instead.
  Protective orders the venue holds for the original quantity are not resized.
- **Partial booking poisons.** A call that stops part way through a booking
  leaves books that cannot be trusted: the node raises `node_poisoned`; rebuild
  it from the venue's positions, not from the journal.
- **Venue-managed positions.** Stops, targets and the square-off are not acted
  on by the kernel; the runtime sends orders ([execution](../execution/SKILL.md)).
  ATR stops, `liquidate_on_margin_call` and shared account groups with a venue
  are refused at construction.
- **Percent exits** measure percent of the covered positions' entry value; a
  rule fires once and its orders pass `order_check`. A venue that defines
  `place_exit_rule(request)` is offered the rule first, for positions that
  exist; `Unsupported` or `OrderRefused` leaves it with the engine. The orders
  the venue fires are registered, so their fills are booked. A held rule is
  replaced (`replace_exit_rule`) or cancelled (`cancel_exit_rule`) when the
  position changes or goes, and a venue that cannot take it back is an error
  (`exit_rule_not_cancellable`), never a rule reported gone.
- **Continuous futures.** The old and the new contract must both be among the
  node's symbols; a roll switches which one a role names and does not add a
  contract to a running node (`roll_target_not_subscribed`).
  `continuous_series(..., adjust="raw" | "back" | "ratio")` has no default.
- **`ctx.option_chain()`** holds only contracts listed (`activation_ns <= now`)
  and unexpired at `ctx.event_ns`; pass `ctx.event_ns` as `now_ns`.
  `ctx.instrument(key)` accepts a symbol or a role bound in `bindings`; a derived
  role is resolved with `ctx.role(...)` first. A symbol with no spec raises
  `no_instrument_spec`.

**Broker support** ("no" means `Unsupported` at `submit`; nothing is dropped
silently):

| | Kite Connect v3 | Upstox v3 (writes) / v2 (reads) | DhanHQ v2 |
| --- | --- | --- | --- |
| market, limit, stop-limit, stop-market | yes | yes | yes |
| trailing, if-touched, market-to-limit | no | no | no |
| time in force | day, ioc | day, ioc | day, ioc |
| `gtc` (the default only for an instrument with no session; with one the default is `day`) | only with `gtc="day"` | only with `gtc="day"` | only with `gtc="day"` |
| gtd, fok, at-open, at-close, `post_only` | no | no | no |
| `reduce_only` | only with `reduce_only="best_effort"` | same | same |
| client id field | `tag` (20, letters and digits) | `tag` (40) | `correlationId` (30) |
| writes need | a valid token | the registered static IP | the registered static IP |

The mappings were written from the brokers' public documentation and tested on
synthetic payloads in those shapes only. Field names the pages do not state are
marked in each module's docstring; confirm them against a paper or small live
account before relying on them.

How an adapter treats a failed request: a refusal (`VenueRefusal`) becomes a
`rejected` report with the broker's code; a request certainly not processed
(`Transient`: 429, refused connection) is retried with backoff and then rejected
as unreachable; one that may have been processed (`Ambiguous`: timeout, reset,
gateway 5xx) is never repeated: the adapter looks the order up by its tag and
adopts it if found, otherwise raises an `unconfirmed` event and the order stays
`Submitted` for reconciliation (`resend_if_absent=True` sends it once more only
after `resend_delay` and a second lookup that again finds nothing). A token that
expired (`AuthExpired`) calls `Session(refresh=)` once, or pauses the outbox
with a `session_expired` event until `set_token()`. A cancel or modify that
cannot be confirmed becomes `cancel_rejected` or `replace_rejected`
("unconfirmed") so the strategy can act. Log in from one process only: a new
login revokes the previous token on some brokers.

## 8. Common mistakes

Restoring without the venue to ask, so the node cannot know what it holds:

<!-- skill: invalid -->
```python
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj

MIN = 60_000_000_000


class Idle(Strategy):
    pass


class Plain:                                   # no open_orders, positions or balances
    namespace = "N1"
    def submit(self, request): ...
    def cancel(self, client_order_id): ...
    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None): ...


journal = rj.Journal({"run": "x"}, memory=True)
node = LiveNode([Idle()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=Plain(), journal=journal)
node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
LiveNode.restore(None, [Idle()], ["AAA"], BacktestConfig(fees=0.0), journal=rj.Journal({"run": "x"}, memory=True),
                 records=journal.since(0), execution=Plain(), bar_stamp="close")   # no_reconcile
```

- Passing no `arrival_ns`. Arrival defaults to the event's own time, which makes
  every event look instant; the processing latency and what the strategy could
  have known are then wrong.
- Reusing a namespace across restarts of the same account, or passing none:
  two runs read each other's orders. A recovered run keeps its own
  (`export_state()` carries it).
- Opening the postback or websocket channel without authenticating it and
  calling `handle_payload(authenticated=True)` for everything. Use `verifier=`,
  or `trusted=True` only when the host opened the channel with the account's own
  token.
- Treating an exception from `push_bar` as a refusal. An exception from the
  adapter means unknown: check `node.health()["unconfirmed"]`.
- Calling `node.push_*` from the executor or feed thread. The node is
  single-threaded; queue reports and deliver them on the node's thread
  (`adapter.deliver_to(node)`).
- Putting derived or engine-owned data in `state()` (positions, orders,
  indicator values): it is rebuilt by replay and a copy only goes stale.
- Letting a journal grow for weeks. Restore time grows with it.
- Resuming with `accept="all"` without looking at the discrepancies.
- Doing the broker round trip inside `submit` (a custom adapter) and raising on
  a timeout: a node whose journal holds such a push cannot currently be
  restored (see [troubleshooting](../troubleshooting/SKILL.md)). Queue the
  request and return, as `BrokerAdapter` does, and raise only `OrderRefused` or
  `Unsupported`.

## 9. Validation

1. Run the day's feed through a paper node with a journal and assert the same
   `state_digest()` after a `restore` at a dozen cut points.
2. Truncate the file journal mid-record and reopen: the torn tail is cut
   (`torn_bytes`), the rest verifies.
3. Replay with a changed strategy and expect `replay_diverged`.
4. With a second node at a higher epoch, expect `fenced` on the first and no
   orders from it.
5. Start from a venue state with a position and an open order the node does not
   know: `reconcile()` returns `position_mismatch` and `unknown_order`.
6. `assert not run_conformance(...)` for every adapter you write.
7. `python -m raptorbt.skills check` after changing anything in a skill.

## 10. Related skills

- [Execution](../execution/SKILL.md)
- [Deployment](../deployment/SKILL.md)
- [Strategy](../strategy/SKILL.md)
- [Risk](../risk/SKILL.md)
- [Troubleshooting](../troubleshooting/SKILL.md)
