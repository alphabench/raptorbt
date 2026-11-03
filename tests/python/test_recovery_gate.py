"""The gate fixes of the execution and recovery work, each from a failing run.

* fencing: a successor restored with a newer epoch takes the account, the old node is fenced;
* recovery against a real ``BrokerAdapter`` (list of ``VenuePosition``, a ``VenueBalances``);
* the adapter's namespace is carried through the journal;
* a report for an order the route no longer holds is journaled, never silently dropped;
* hedging on a venue books against the position it opened and exits close what they target;
* a venue-held exit rule goes through the journal, the fence and the replay, and its fills book;
* an auto-roll opens in the new contract only once the close has filled.
"""

import json

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.adapters import Instrument, Session, VenueBalances, VenuePosition
from raptorbt.adapters.conformance import ReferenceBinding, ScriptedVenue
from raptorbt.errors import InvalidData, InvalidParameter, StateError
from raptorbt.execution import ExecutionReport, OrderRefused
from raptorbt.strategy import Future, Holding, Requirements, Underlying, orders
from raptorbt.strategy.continuous import RollEvents
from raptorbt.strategy.exits import ExitEvents, ExitRule
import raptorbt
from test_journal import CONFIG, HEADER, MIN, SYMBOLS, MockVenue, Trader, build, run_step, script


def records_of(node):
    j = node.journal
    return [r.canonical() for r in j.since(j.seq_base)]


def restore(store, venue, **kw):
    return LiveNode.restore(
        None, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
        records=list(store), execution=venue, bar_stamp="close", **kw,
    )


# -- fencing across a restore --------------------------------------------------------


def test_a_successor_restored_with_a_newer_epoch_takes_the_account_and_fences_the_old_node():
    venue = MockVenue()
    store = []
    old = build(venue, rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())), epoch=1)
    steps = script(12)
    for step in steps[:4]:
        run_step(old, venue, step)
    new = restore(store, venue, epoch=2)
    assert new.state == "live" and new.epoch == 2 and venue.epoch == 2 and not new.fenced
    epochs = [r.payload["epoch"] for r in new.journal.since(0) if r.kind == "epoch"]
    assert epochs == [1, 2]
    sent = list(venue.sent)
    # The old node's next request is refused; nothing leaves it.
    with pytest.raises(Exception) as e:
        for step in steps[4:]:
            run_step(old, venue, step)
    assert e.value.code == "fenced" and old.fenced
    assert venue.sent == sent
    # The new node sends.
    for step in steps[4:]:
        run_step(new, venue, step)
    assert len(venue.sent) > len(sent)
    # A node never goes back to an older epoch, and a third takeover replays both.
    with pytest.raises(StateError) as e:
        restore(records_of(new), venue, epoch=1)
    assert e.value.code == "stale_epoch"
    third = restore(records_of(new), venue, epoch=3)
    assert third.epoch == 3 and venue.epoch == 3


def test_restore_without_an_epoch_keeps_the_epoch_the_journal_holds():
    venue = MockVenue()
    store = []
    node = build(venue, rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())), epoch=4)
    for step in script(4):
        run_step(node, venue, step)
    again = restore(store, venue)
    assert again.epoch == 4
    assert [r.kind for r in again.journal.since(0)].count("epoch") == 1


def test_an_epoch_on_an_adapter_with_no_fence_is_refused_not_ignored():
    class Bare:
        namespace = "B1"

        def submit(self, r):
            pass

        def cancel(self, c):
            pass

        def modify(self, c, **k):
            pass

    with pytest.raises(InvalidParameter) as e:
        LiveNode([Trader()], SYMBOLS, CONFIG, bar_stamp="close", execution=Bare(),
                 journal=rj.Journal(HEADER, memory=True), epoch=3)
    assert e.value.code == "fence_unsupported"
    LiveNode([Trader()], SYMBOLS, CONFIG, bar_stamp="close", execution=Bare(),
             journal=rj.Journal(HEADER, memory=True))


# -- H-2 / H-6: recovery with a real BrokerAdapter, and the namespace -------------------------


class Buyer(Strategy):
    def __init__(self):
        super().__init__()
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=5.0))
        elif self.n == 2:
            self.submit_order(orders.Limit(side="sell", units=2.0, price=150.0))


def broker_adapter(venue, namespace="CF"):
    """A BrokerAdapter (the conformance kit's reference mapping) that also maps its books."""
    base = type(ReferenceBinding(venue, {}).adapter)

    class WithBooks(base):
        def fetch_positions(self):
            held = {}
            for o in venue.orders.values():
                sign = 1 if o.side == "buy" else -1
                held[o.symbol] = held.get(o.symbol, 0.0) + sign * o.cum
            return [VenuePosition(s, q, 100.0) for s, q in held.items() if q]

        def fetch_balances(self):
            return VenueBalances(available=1_000_000.0 - sum(o.notional for o in venue.orders.values()), currency="INR")

    clock = iter(range(1, 10**9))
    return WithBooks(
        Session("t"), lambda symbol: Instrument("X", symbol),
        sleep=lambda s: None, clock_ns=lambda: next(clock), namespace=namespace, trusted=True,
    )


def drive_buyer(adapter, venue, store):
    venue.subscribers.append(adapter.handle_payload)
    node = LiveNode([Buyer()], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close",
                    execution=adapter, journal=rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())))
    node.push_bar("A", MIN, 100, 100, 100, 100, 1, arrival_ns=MIN)
    adapter.pump()
    ref = adapter.ids.encode("s0-0")
    venue.accept(ref)
    venue.fill(ref, 5.0, 100.0)
    for cid, report, arrival in [(q.client_order_id, q.report, q.arrival_ns) for q in adapter.drain_reports()]:
        node.push_report(cid, report, arrival_ns=arrival)
    node.push_bar("A", 2 * MIN, 100, 100, 100, 100, 1, arrival_ns=2 * MIN)
    adapter.pump()
    venue.accept(adapter.ids.encode("s0-1"))
    for q in adapter.drain_reports():
        node.push_report(q.client_order_id, q.report, arrival_ns=q.arrival_ns)
    return node


def test_recovery_reconciles_and_resumes_against_a_real_broker_adapter_without_a_second_order():
    venue = ScriptedVenue()
    store = []
    node = drive_buyer(broker_adapter(venue), venue, store)
    assert node.reconcile() == []  # list[VenuePosition] and a VenueBalances, not dicts
    placed = dict(venue.place_calls)
    again = broker_adapter(venue)  # the process restarted: a fresh adapter, same namespace
    venue.subscribers.append(again.handle_payload)
    restored = LiveNode.restore(
        None, [Buyer()], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0),
        journal=rj.Journal(HEADER, memory=True), records=list(store), execution=again, bar_stamp="close",
    )
    # The working limit order is found at the venue by the namespace in its tag.
    assert restored.state == "live"
    assert dict(venue.place_calls) == placed and venue.duplicates == []
    rec = [r for r in restored.journal.since(0) if r.kind == "reconcile"][-1]
    assert rec.payload["venue"]["positions"] == {"A": "5"}  # floats are decimal strings in a record
    assert rec.payload["venue"]["balances"]["currency"] == "INR"


def test_a_restore_refuses_an_adapter_of_another_namespace():
    venue = ScriptedVenue()
    store = []
    drive_buyer(broker_adapter(venue), venue, store)
    other = broker_adapter(venue, namespace="ZZ")
    with pytest.raises(StateError) as e:
        LiveNode.restore(
            None, [Buyer()], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0),
            journal=rj.Journal(HEADER, memory=True), records=list(store), execution=other, bar_stamp="close",
        )
    assert e.value.code == "adapter_namespace_mismatch"


def test_an_adapter_without_a_namespace_is_refused():
    class NoNamespace:
        def submit(self, r):
            pass

        def cancel(self, c):
            pass

        def modify(self, c, **k):
            pass

    with pytest.raises(InvalidParameter) as e:
        LiveNode([Trader()], SYMBOLS, CONFIG, bar_stamp="close", execution=NoNamespace())
    assert e.value.code == "adapter_namespace_missing"


def test_reconcile_keys_each_foreign_order_by_its_venue_id():
    class Foreign(MockVenue):
        def open_orders(self):
            return [
                {"client_order_id": None, "venue_order_id": "V1", "symbol": "A", "quantity": 1.0},
                {"client_order_id": None, "venue_order_id": "V2", "symbol": "A", "quantity": 2.0},
            ]

    venue = Foreign()
    node = build(venue, rj.Journal(HEADER, memory=True))
    found = node.reconcile()
    assert sorted(d.client_order_id for d in found if d.kind == "unknown_order") == ["venue:V1", "venue:V2"]


# -- M-5: the checkpoint's chain head -----------------------------------------------------------


def test_restore_refuses_records_whose_chain_is_not_the_checkpoints():
    venue = MockVenue()
    node = build(venue, rj.Journal(HEADER, memory=True))
    for step in script(4):
        run_step(node, venue, step)
    ck = node.checkpoint(include_log=False)
    records = records_of(node)
    changed = records[:3] + [records[3].replace(b'"seq":3', b'"seq":3') ] + records[4:]
    # Same count, one record's bytes changed: the digest of the books may even agree, the chain cannot.
    tampered = list(records)
    tampered[2] = tampered[2].replace(b'"source":"feed"', b'"source":"FEED"') if b'"source":"feed"' in tampered[2] else tampered[2] + b" "
    with pytest.raises(Exception) as e:
        LiveNode.restore(ck, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
                         records=tampered, execution=venue, bar_stamp="close")
    assert e.value.code in ("replay_diverged", "invalid_checkpoint")
    assert changed  # (kept for the reader: the untampered records restore)
    LiveNode.restore(ck, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
                     records=records, execution=venue, bar_stamp="close")


# -- a report for an order the route no longer holds ---------------------------------------


class Flip(Strategy):
    def __init__(self):
        super().__init__()
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n % 2:
            self.submit_order(orders.Market(side="buy", units=1.0))
        else:
            self.submit_order(orders.Market(side="sell", units=1.0, reduce_only=True))


def test_a_fill_or_correction_on_an_evicted_order_is_journaled_and_never_silently_late():
    v = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([Flip()], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close",
                    execution=v, journal=j)
    for i in range(1, 4200):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    first = next(iter(v.orders))
    cnt = j.count
    cash = node.ctx(0).cash
    r = node.push_report(first, ExecutionReport.correction(f"X-{first}", 1.0, 50.0, 10**15), arrival_ns=10**15)
    assert r == "unbooked"
    kinds = [x.kind for x in j.since(cnt)]
    assert kinds == ["report", "unbooked_fill"]
    assert node.ctx(0).cash == cash
    r = node.push_report(first, ExecutionReport.fill("X-extra", 1.0, 100.0, 10**15), arrival_ns=10**15 + 1)
    assert r == "unbooked"
    # What does not move money is journaled too, and is late.
    r = node.push_report(first, ExecutionReport.cancel_ack(10**15 + 2), arrival_ns=10**15 + 2)
    assert r == "late"
    assert [x.kind for x in j.since(cnt)][-1] == "report"


def test_a_report_for_an_unknown_order_is_on_the_record_before_it_raises():
    v = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = build(v, j)
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    cnt = j.count
    with pytest.raises(Exception) as e:
        node.push_report("s0-99", ExecutionReport.accepted("V", MIN), arrival_ns=MIN)
    assert e.value.code == "unknown_client_order_id"
    assert [x.kind for x in j.since(cnt)][0] == "report"


# -- H-4 / M-1: hedging on a venue --------------------------------------------------------------


class Stack(Strategy):
    def __init__(self, plan):
        super().__init__()
        self.n, self.plan = 0, plan

    def on_bar(self, ctx, bar):
        self.n += 1
        step = self.plan.get(self.n)
        if step:
            side, units, reduce_only = step
            self.submit_order(orders.Market(side=side, units=units, reduce_only=reduce_only))


def hedged(plan):
    v = MockVenue()
    node = LiveNode([Stack(plan)], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close",
                    execution=v, oms_type="hedging", journal=rj.Journal(HEADER, memory=True))
    for i in range(1, max(plan) + 2):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    return node, v


def test_a_bust_on_a_hedged_venue_takes_back_the_position_that_execution_opened():
    node, v = hedged({1: ("buy", 5.0, False), 2: ("buy", 3.0, False)})
    ctx = node.ctx(0)
    assert sorted(p.size for p in ctx.positions("A")) == [3.0, 5.0]
    second = list(v.orders)[1]
    cash = ctx.cash
    assert node.push_report(second, ExecutionReport.bust(f"X-{second}", 3 * MIN), arrival_ns=3 * MIN) == "applied"
    assert [p.size for p in ctx.positions("A")] == [5.0]
    assert ctx.cash > cash


def test_exits_on_a_hedged_venue_close_what_they_target_oldest_first():
    node, v = hedged({1: ("buy", 5.0, False), 2: ("buy", 3.0, False), 3: ("sell", 6.0, True)})
    ctx = node.ctx(0)
    assert [p.size for p in ctx.positions("A")] == [2.0]  # the older 5 closed, 1 of the 3 taken
    assert [p.direction for p in ctx.positions("A")] == [1]
    assert "unbooked_fill" not in {r.kind for r in node.journal.since(0)}
    node2, _ = hedged({1: ("buy", 5.0, False), 2: ("buy", 3.0, False), 3: ("sell", 8.0, True)})
    assert node2.ctx(0).positions("A") == []


# -- M-2: a failing order check leaves no order behind --------------------------------------------


def test_an_order_check_that_raises_leaves_no_phantom_order_on_the_venue_path():
    v = MockVenue()
    calls = []

    def check(order, ctx):
        calls.append(order)
        raise RuntimeError("check is down")

    node = LiveNode([Stack({1: ("buy", 1.0, False)})], ["A"],
                    BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close",
                    execution=v, order_check=check)
    with pytest.raises(Exception):
        node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    assert calls and v.sent == []
    assert node._rt.venue_orders() == []
    assert node.ctx(0).open_orders("A") == []


# -- an exit rule the venue holds ----------------------------------------------------------------


class HoldingVenue(MockVenue):
    def __init__(self):
        super().__init__()
        self.rules = []

    def place_exit_rule(self, request):
        self.rules.append(request)


class Holder(ExitEvents, Strategy):
    def __init__(self):
        super().__init__()
        self.n, self.handle = 0, None

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=5.0))
        if self.n == 3 and self.handle is None:
            self.handle = self.set_exit(ctx, ExitRule(stop_loss_pct=10), symbol="A")


def held_node(venue, store, **kw):
    return LiveNode([Holder()], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close",
                    execution=venue, journal=rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())), **kw)


def run_holder(node, venue, bars=4):
    for i in range(1, bars + 1):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        venue.fill_all()
        venue.deliver(node, i * MIN + 5)


def test_a_venue_held_exit_rule_is_journaled_and_the_exit_the_venue_fires_is_booked():
    venue, store = HoldingVenue(), []
    node = held_node(venue, store)
    run_holder(node, venue)
    handle = node._strategies[0].handle
    assert handle.state == "venue" and len(venue.rules) == 1
    [req] = venue.rules
    assert req.strategy_index == 0 and req.legs == (("A", "sell", 5.0),)
    kept = [json.loads(c) for c in store]
    kinds = [(r["kind"], r["payload"].get("op")) for r in kept
             if r["kind"].startswith("adapter_") and r["payload"].get("op") == "place_exit_rule"]
    assert kinds == [("adapter_request", "place_exit_rule"), ("adapter_result", "place_exit_rule")]
    assert [r["payload"]["status"] for r in kept if r["kind"] == "adapter_result"
            and r["payload"].get("op") == "place_exit_rule"] == ["ok"]
    # The venue fires the exit under the id it was told; the node books it.
    cid = req.exit_client_ids[0]
    assert node.push_report(cid, ExecutionReport.accepted("V-x", 6 * MIN), arrival_ns=6 * MIN) == "applied"
    assert node.push_report(cid, ExecutionReport.fill("X-exit", 5.0, 90.0, 6 * MIN + 1), arrival_ns=6 * MIN + 1) == "applied"
    assert node.ctx(0).positions("A") == []


def test_a_replay_does_not_place_the_exit_rule_again_and_reaches_the_same_state():
    venue, store = HoldingVenue(), []
    node = held_node(venue, store)
    run_holder(node, venue)
    digest = node.state_digest()
    restored = LiveNode.restore(
        None, [Holder()], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0),
        journal=rj.Journal(HEADER, memory=True), records=list(store), execution=venue, bar_stamp="close",
        reconcile="skip",
    )
    assert len(venue.rules) == 1, "the venue was not asked again"
    assert restored.state_digest() == digest
    cid = venue.rules[0].exit_client_ids[0]
    restored.push_report(cid, ExecutionReport.accepted("V-x", 6 * MIN), arrival_ns=6 * MIN)
    restored.push_report(cid, ExecutionReport.fill("X-exit", 5.0, 90.0, 6 * MIN + 1), arrival_ns=6 * MIN + 1)
    assert restored.ctx(0).positions("A") == []


def test_a_fenced_node_does_not_place_an_exit_rule_at_the_venue():
    venue, store = HoldingVenue(), []
    node = held_node(venue, store, epoch=1)
    run_holder(node, venue, bars=2)
    venue.epoch = 9  # a newer node owns the account
    with pytest.raises(Exception) as e:
        run_holder_from(node, venue, 3, 4)
    assert e.value.code == "fenced"
    assert venue.rules == []


def run_holder_from(node, venue, first, last):
    for i in range(first, last + 1):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        venue.fill_all()
        venue.deliver(node, i * MIN + 5)


def test_warmup_bars_place_no_exit_rule_at_the_venue():
    class WarmHolder(ExitEvents, Strategy):
        def __init__(self):
            super().__init__()
            self.handles = []

        def on_bar(self, ctx, bar):
            self.handles.append(self.set_exit(ctx, ExitRule(stop_loss_pct=10), symbol="A"))

    venue = HoldingVenue()
    warm = {"A": {"timestamps": np.array([MIN, 2 * MIN], dtype=np.int64), "open": np.array([100.0, 100.0]),
                  "high": np.array([100.0, 100.0]), "low": np.array([100.0, 100.0]),
                  "close": np.array([100.0, 100.0]), "volume": np.array([1.0, 1.0])}}
    s = WarmHolder()
    LiveNode([s], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close",
              execution=venue, warmup_bars=warm)
    assert s.handles and venue.rules == []
    assert all(h.state == "armed" for h in s.handles)


# -- M-4: a refused exit close re-arms ---------------------------------------------------------------


class RefuseReduceOnly(MockVenue):
    def submit(self, request):
        if request.reduce_only:
            raise OrderRefused("reduce_only_refused")
        super().submit(request)


class Guard(ExitEvents, Strategy):
    def __init__(self):
        super().__init__()
        self.n, self.handle = 0, None

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=5.0))
        if self.n == 2:
            self.handle = self.set_exit(ctx, ExitRule(stop_loss_pct=10), symbol="A")


def test_an_exit_whose_close_is_refused_arms_again_and_is_never_left_without_its_stop():
    v = RefuseReduceOnly()
    node = LiveNode([Guard()], ["A"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close",
                    execution=v)
    px = [100, 100, 80, 80, 80, 80]
    for i, p in enumerate(px, start=1):
        node.push_bar("A", i * MIN, p, p, p, p, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    h = node._strategies[0].handle
    assert h.attempts >= 2, "the refused close fired the rule again"
    assert h.refused and h.refused[0][1] == "rejected"
    assert len(set(h.close_ids)) == len(h.close_ids), "each attempt has its own ids"
    assert node.ctx(0).positions("A"), "the position is still open, and still guarded"


# -- H-7 / M-8: auto-roll ----------------------------------------------------------------------------

T0C = 1_700_000_000 * 10**9
EXPA, EXPB = T0C + 20 * MIN, T0C + 40 * MIN


class Carry(RollEvents, Strategy):
    requirements = Requirements(
        roles={"index": Underlying(asset="index"),
               "fut": Future(of="index", contract="continuous", roll_days=5 / 1440)},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
    )
    auto_roll = True

    def __init__(self):
        super().__init__()
        self.n, self.failed = 0, []

    def on_bar(self, ctx, bar):
        if ctx.symbol != "IDX":
            return
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=3.0, instrument=ctx.role("fut")), client_id="open")

    def on_roll_failed(self, ctx, roll, close_id, status, reason):
        self.failed.append((close_id, status, reason))


def roll_node(venue):
    specs = {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", EXPA, 1.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", EXPB, 1.0, underlying="IDX"),
    }
    return LiveNode([Carry()], ["IDX", "FA", "FB"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                    execution=venue, instruments=specs, bindings={"index": "IDX"})


def drive_roll(node, venue, bars=24):
    for i in range(1, bars + 1):
        for k, s in enumerate(("IDX", "FA", "FB")):
            px = 100.0 + 5 * k
            node.push_bar(s, T0C + i * MIN, px, px + 1, px - 1, px, 100.0)
        venue.fill_all()
        venue.deliver(node, T0C + i * MIN + 500)


class SymbolVenue(MockVenue):
    pass


def test_auto_roll_opens_the_new_contract_only_after_the_close_has_filled():
    v = MockVenue()
    node = roll_node(v)
    drive_roll(node, v)
    by_symbol = [(v.orders[c]["symbol"], v.orders[c]["side"], v.orders[c]["reduce_only"]) for c in v.sent]
    assert by_symbol == [("FA", "buy", False), ("FA", "sell", True), ("FB", "buy", False)]
    assert [p.size for p in node.ctx(0).positions("FB")] == [3.0]
    assert node.ctx(0).positions("FA") == []


def test_auto_roll_does_not_double_the_exposure_when_the_close_is_refused():
    v = RefuseReduceOnly()
    node = roll_node(v)
    drive_roll(node, v, bars=17)  # the roll falls at bar 15; FA expires at 20
    s = node._strategies[0]
    assert [(c, st) for c, st, _ in s.failed] == [("fut:roll0:c0", "rejected")]
    assert [v.orders[c]["symbol"] for c in v.sent] == ["FA"], "no open was sent in the new contract"
    assert [p.size for p in node.ctx(0).positions("FA")] == [3.0]
    assert node.ctx(0).positions("FB") == []


def test_a_failed_roll_close_raises_by_default():
    class Strict(Carry):
        on_roll_failed = RollEvents.on_roll_failed

    v = RefuseReduceOnly()
    specs = {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", EXPA, 1.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", EXPB, 1.0, underlying="IDX"),
    }
    node = LiveNode([Strict()], ["IDX", "FA", "FB"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                    execution=v, instruments=specs, bindings={"index": "IDX"})
    with pytest.raises(Exception) as e:
        drive_roll(node, v)
    assert "still in the old contract" in str(e.value)
    assert [v.orders[c]["symbol"] for c in v.sent] == ["FA"]


def test_the_roll_is_part_of_the_strategys_state_so_a_restore_resumes_it():
    s = Carry()
    assert s.state() is None
    s._roll_state()[1]["fut"] = ("FA", 123)
    s._roll_state()[2][0] = 2
    s._roll_state()[3]["fut:roll1:c0"] = {"open_id": "fut:roll1:o0", "side": "buy", "role": "fut",
                                           "from": "FA", "to": "FB", "ts": 1, "left": 3.0}
    saved = s.state()
    import json

    saved = json.loads(json.dumps(saved))
    t = Carry()
    t.load_state(saved)
    assert t.state() == s.state()
    assert t._roll_state()[1]["fut"] == ("FA", 123) and t._roll_state()[2] == [2]


def test_a_strategy_state_of_its_own_carries_the_roll_alongside():
    class Own(RollEvents, Strategy):
        def state(self):
            return {"mine": 1}

        def load_state(self, state):
            self.mine = state["mine"]

    a = Own()
    a._roll_state()[2][0] = 4
    saved = a.state()
    assert saved["mine"] == 1 and "__rolls__" in saved
    b = Own()
    b.load_state(saved)
    assert b.mine == 1 and b._roll_state()[2] == [4]


# -- a request that raised: replay reproduces the outcome, and unconfirmed is tracked ------------------


def test_a_journal_with_a_push_whose_submit_raised_restores_without_calling_the_adapter():
    venue = MockVenue()
    store = []
    node = build(venue, rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())))
    steps = script(6)
    venue.fail_next = ConnectionError("link down")
    raised = 0
    for step in steps:
        try:
            run_step(node, venue, step)
        except Exception:
            raised += 1
    assert raised, "the first order raised"
    calls = len(venue.sent)

    class Cold(MockVenue):
        def submit(self, request):
            raise AssertionError("a replay must not call the adapter")

    cold = Cold()
    restored = restore(store, cold, reconcile="skip")
    assert restored.state == "live"
    assert restored.state_digest() == node.state_digest()
    assert len(venue.sent) == calls


def test_unconfirmed_requests_are_tracked_on_a_node_with_no_journal():
    venue = MockVenue()
    node = build(venue, None)
    venue.fail_next = ConnectionError("link down")
    for step in script(4):
        try:
            run_step(node, venue, step)
        except Exception:
            break
    assert node.health()["unconfirmed"] == ["s0-0"]
