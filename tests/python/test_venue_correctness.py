"""Venue, journal and fencing correctness: each test starts from the defect a reviewer reproduced.

* a leased BrokerAdapter fences the node, and a write dropped by the fence is a rejected report;
* a restore takes a newer epoch, and a fresh adapter is taught the orders the node holds;
* a bust of a fill that reversed a position is refused, a hedged one is applied;
* a venue-held exit rule is never handed zero legs, follows the position, is cancelled at the
  venue, and its orders are neither missing nor unknown;
* a position the broker keys by its own key reads as the engine's symbol, a cash balance is
  compared like with like;
* a rejected roll open is a failed roll, and an open is sized in the new contract's lot;
* a refused exit close backs off and gives up loudly;
* hedging closes name their position, and a venue that nets refuses a hedging node.
"""

import json

import pytest

import raptorbt
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.adapters import Instrument, ManualExecutor, Session, VenueBalances, VenuePosition
from raptorbt.adapters.base import OrderStatusUpdate
from raptorbt.adapters.conformance import ReferenceBinding, ScriptedVenue
from raptorbt.errors import InvalidParameter, StateError
from raptorbt.execution import ExecutionReport, Fenced, OrderRefused, Unsupported
from raptorbt.strategy import orders
from raptorbt.strategy.continuous import RollEvents
from raptorbt.strategy.exits import MAX_CLOSE_ATTEMPTS, ExitEvents, ExitRule
from raptorbt.strategy._recovery import venue_cash
from test_journal import CONFIG, HEADER, MIN, MockVenue
from test_recovery_gate import (
    Buyer,
    Carry,
    Guard,
    HoldingVenue,
    Holder,
    RefuseReduceOnly,
    Stack,
    broker_adapter,
    drive_buyer,
    drive_roll,
    held_node,
    hedged,
    roll_node,
    run_holder,
)

NO_FEES = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)


# -- 1. a leased BrokerAdapter fences the node ---------------------------------------------------


def _leased(holds, executor=None):
    venue = ScriptedVenue()
    base = type(ReferenceBinding(venue, {}).adapter)
    clock = iter(range(1, 10**9))
    kw = {"executor": executor} if executor is not None else {}
    ad = base(Session("t"), lambda s: Instrument("X", s), sleep=lambda s: None, clock_ns=lambda: next(clock),
              namespace="CF", trusted=True, lease=lambda e: holds["ok"], **kw)
    venue.subscribers.append(ad.handle_payload)
    return venue, ad


def test_a_node_on_a_leased_adapter_is_fenced_at_the_first_write_after_the_lease_is_lost():
    holds = {"ok": True}
    venue, ad = _leased(holds)
    store = []
    node = LiveNode([Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=ad, epoch=1,
                    journal=rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())))
    assert node.heartbeat()["fenced"] is False
    holds["ok"] = False  # the lease is lost
    with pytest.raises(StateError) as e:
        node.heartbeat()
    assert e.value.code == "fenced" and node.fenced
    assert dict(venue.place_calls) == {}
    # Nothing is journaled as taken that was never sent.
    kept = [json.loads(c) for c in store]
    assert not [r for r in kept if r["kind"] == "adapter_result" and r["payload"]["op"] == "submit"]
    assert [r["kind"] for r in kept][-1] == "fence"
    with pytest.raises(StateError) as e:
        node.push_bar("A", MIN, 100, 100, 100, 100, 1, arrival_ns=MIN)
    assert e.value.code == "fenced"
    assert dict(venue.place_calls) == {}


def test_the_first_order_after_the_lease_is_lost_is_refused_not_journaled_ok():
    holds = {"ok": True}
    venue, ad = _leased(holds)
    store = []
    node = LiveNode([Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=ad, epoch=1,
                    journal=rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())))
    holds["ok"] = False
    with pytest.raises(StateError) as e:
        node.push_bar("A", MIN, 100, 100, 100, 100, 1, arrival_ns=MIN)
    assert e.value.code == "fenced"
    kept = [json.loads(c) for c in store]
    assert not [r for r in kept if r["kind"] == "adapter_result" and r["payload"].get("status") == "ok"]
    assert dict(venue.place_calls) == {} and node.fenced


def test_the_adapters_fenced_is_the_executions_fenced():
    from raptorbt.adapters.base import Fenced as AdapterFenced

    assert AdapterFenced is Fenced


def test_writes_the_fence_drops_come_back_as_rejected_reports_so_the_node_never_holds_them_as_sent():
    holds = {"ok": True}
    venue, ad = _leased(holds, executor=ManualExecutor())
    node = LiveNode([Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=ad)
    node.push_bar("A", MIN, 100, 100, 100, 100, 1, arrival_ns=MIN)
    assert [o[6] for o in node._rt.venue_orders()] == ["submitted"]
    holds["ok"] = False  # lost between the hand-over and the send
    ad.pump()
    assert dict(venue.place_calls) == {}
    reports = ad.drain_reports()
    assert [(q.client_order_id, q.report.kind, q.report.reason) for q in reports] == [("s0-0", "rejected", "fenced")]
    for q in reports:
        node.push_report(q.client_order_id, q.report, arrival_ns=q.arrival_ns)
    assert [o[6] for o in node._rt.venue_orders()] == ["rejected"]
    assert node.ctx(0).open_orders("A") == []


# -- 2. restore takes a newer epoch ---------------------------------------------------------------


def _epoch_journal(epoch=4):
    venue = MockVenue()
    store = []
    from test_journal import build, run_step, script

    node = build(venue, rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())), epoch=epoch)
    for step in script(4):
        run_step(node, venue, step)
    return venue, store


def _restore(store, venue, **kw):
    from test_journal import SYMBOLS, Trader

    return LiveNode.restore(None, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
                            records=list(store), execution=venue, bar_stamp="close", **kw)


def test_restoring_a_journal_that_holds_an_epoch_needs_a_newer_one():
    venue, store = _epoch_journal(4)
    with pytest.raises(StateError) as e:
        _restore(store, venue)
    assert e.value.code == "epoch_required"
    with pytest.raises(StateError) as e:
        _restore(store, venue, epoch=4)
    assert e.value.code == "stale_epoch"
    with pytest.raises(StateError) as e:
        _restore(store, venue, epoch=3)
    assert e.value.code == "stale_epoch"
    again = _restore(store, venue, epoch=5)
    assert again.epoch == 5 and venue.epoch == 5
    assert [r.payload["epoch"] for r in again.journal.since(0) if r.kind == "epoch"] == [4, 5]


def test_a_journal_with_no_epoch_restores_without_one():
    venue = MockVenue()
    store = []
    from test_journal import build, run_step, script

    node = build(venue, rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())))
    for step in script(4):
        run_step(node, venue, step)
    assert _restore(store, venue).epoch is None


# -- 3. a restored node and a fresh adapter -----------------------------------------------------


def _restored_with_a_fresh_adapter():
    venue = ScriptedVenue()
    store = []
    drive_buyer(broker_adapter(venue), venue, store)
    again = broker_adapter(venue)  # the process restarted: a fresh adapter, same namespace
    venue.subscribers.append(again.handle_payload)
    restored = LiveNode.restore(
        None, [Buyer()], ["A"], NO_FEES, journal=rj.Journal(HEADER, memory=True), records=list(store),
        execution=again, bar_stamp="close",
    )
    return venue, again, restored, store


def test_a_fresh_adapter_follows_the_fill_of_an_order_the_restored_node_holds_as_working():
    venue, again, restored, _ = _restored_with_a_fresh_adapter()
    assert restored.state == "live"
    ref = again.ids.encode("s0-1")
    venue.fill(ref, 2.0, 150.0)
    reports = again.drain_reports()
    assert [(q.client_order_id, q.report.kind, q.report.qty, q.report.price) for q in reports] == [("s0-1", "fill", 2.0, 150.0)]
    assert again.ignored_updates == 0
    for q in reports:
        restored.push_report(q.client_order_id, q.report, arrival_ns=q.arrival_ns)
    assert [p.size for p in restored.ctx(0).positions("A")] == [3.0]  # 5 bought, 2 sold at the limit


def test_a_fresh_adapter_can_cancel_an_order_the_restored_node_holds_as_working():
    venue, again, restored, _ = _restored_with_a_fresh_adapter()
    again.cancel("s0-1")  # was OrderRefused("unknown order")
    assert venue.cancel_calls[again.ids.encode("s0-1")] == 1


def test_a_restored_order_resumes_from_the_fills_the_node_booked_not_from_zero():
    venue = ScriptedVenue()
    store = []
    first = broker_adapter(venue)
    node = drive_buyer(first, venue, store)
    ref = first.ids.encode("s0-1")
    venue.fill(ref, 1.0, 150.0)  # half of the limit order: the first process books it
    for q in first.drain_reports():
        node.push_report(q.client_order_id, q.report, arrival_ns=q.arrival_ns)
    again = broker_adapter(venue)
    venue.subscribers.append(again.handle_payload)
    LiveNode.restore(None, [Buyer()], ["A"], NO_FEES, journal=rj.Journal(HEADER, memory=True),
                     records=list(store), execution=again, bar_stamp="close")
    venue.fill(ref, 1.0, 160.0)  # the rest: cumulative 2 at the venue
    reports = again.drain_reports()
    assert [(q.report.kind, q.report.qty, q.report.price) for q in reports] == [("fill", 1.0, 160.0)], \
        "only what the node had not seen, at the price that makes the average right"


def test_an_adapter_that_cannot_account_for_a_working_order_refuses_the_restore():
    venue = ScriptedVenue()
    store = []
    drive_buyer(broker_adapter(venue), venue, store)
    blind = type(broker_adapter(venue))(
        Session("t"), lambda symbol: None, sleep=lambda s: None, namespace="CF", trusted=True,
    )
    with pytest.raises(StateError) as e:
        LiveNode.restore(None, [Buyer()], ["A"], NO_FEES, journal=rj.Journal(HEADER, memory=True),
                         records=list(store), execution=blind, bar_stamp="close")
    assert e.value.code == "adapter_cannot_adopt"


def test_a_venue_row_the_adapter_ignores_raises_an_event_once():
    venue = ScriptedVenue()
    ad = broker_adapter(venue)
    row = OrderStatusUpdate(venue_order_id="V99", status="open", tag="OTHER-s0x0", symbol="A", side="buy", quantity=1.0)
    ad.handle_update(row)
    ad.handle_update(row)
    assert ad.ignored_updates == 2
    events = [e for e in ad.drain_events() if e.kind == "ignored_update"]
    assert len(events) == 1 and "V99" in events[0].detail


# -- 4. a bust of a fill that reversed a position ---------------------------------------------------------


def test_a_bust_of_a_fill_that_reversed_a_position_is_refused_and_creates_no_money():
    v = MockVenue()
    node = LiveNode([Stack({1: ("buy", 5.0, False), 2: ("sell", 8.0, False)})], ["A"], NO_FEES,
                    bar_stamp="close", execution=v, journal=rj.Journal(HEADER, memory=True))
    for i in range(1, 4):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    ctx = node.ctx(0)
    before = ([(p.direction, p.size, p.entry_price) for p in ctx.positions("A")], ctx.cash)
    second = list(v.orders)[1]
    with pytest.raises(Exception) as e:
        node.push_report(second, ExecutionReport.bust(f"X-{second}", 4 * MIN), arrival_ns=4 * MIN)
    assert e.value.code == "correction_not_reversible"
    assert "more than one position change" in str(e.value)
    assert ([(p.direction, p.size, p.entry_price) for p in ctx.positions("A")], ctx.cash) == before


def test_under_hedging_the_same_fill_opens_a_position_of_its_own_and_its_bust_is_applied_exactly():
    node, v = hedged({1: ("buy", 5.0, False), 2: ("sell", 8.0, False)})
    ctx = node.ctx(0)
    assert sorted((p.direction, p.size) for p in ctx.positions("A")) == [(-1, 8.0), (1, 5.0)]
    cash_with_both = ctx.cash
    second = list(v.orders)[1]
    assert node.push_report(second, ExecutionReport.bust(f"X-{second}", 4 * MIN), arrival_ns=4 * MIN) == "applied"
    assert [(p.direction, p.size) for p in ctx.positions("A")] == [(1, 5.0)]
    assert ctx.cash > cash_with_both  # the short's proceeds came off, the purchase's cost stayed


# -- 5. venue-held exit rules ----------------------------------------------------------------------


class RuleVenue(HoldingVenue):
    """A venue that holds rules and can change or drop them."""

    def __init__(self):
        super().__init__()
        self.cancelled, self.replaced = [], []

    def cancel_exit_rule(self, request):
        self.cancelled.append(request)

    def replace_exit_rule(self, old, new):
        self.replaced.append((old, new))


class Trim(ExitEvents, Strategy):
    """Buys 5, holds a rule from the third bar, sells 2 on the fifth."""

    def __init__(self):
        super().__init__()
        self.n, self.handle = 0, None

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=5.0))
        if self.n == 3:
            self.handle = self.set_exit(ctx, ExitRule(stop_loss_pct=10), symbol="A")
        if self.n == 5:
            self.submit_order(orders.Market(side="sell", units=2.0, reduce_only=True))


def trim_node(venue):
    return LiveNode([Trim()], ["A"], NO_FEES, bar_stamp="close", execution=venue,
                    journal=rj.Journal(HEADER, memory=True))


def run_bars(node, venue, first, last, price=100):
    for i in range(first, last + 1):
        node.push_bar("A", i * MIN, price, price, price, price, 1)
        venue.fill_all()
        venue.deliver(node, i * MIN + 5)


def test_a_rule_set_before_its_entry_fills_has_no_legs_so_the_engine_watches_it():
    class Early(ExitEvents, Strategy):
        def __init__(self):
            super().__init__()
            self.n, self.h = 0, None

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Market(side="buy", units=5.0))
                self.h = self.set_exit(ctx, ExitRule(stop_loss_pct=10), symbol="A")

    v = RuleVenue()
    s = Early()
    node = LiveNode([s], ["A"], NO_FEES, bar_stamp="close", execution=v)
    run_bars(node, v, 1, 2)
    assert v.rules == [], "the venue was not handed a rule with no legs"
    assert s.h.state == "armed"
    run_bars(node, v, 3, 5, price=50)  # the stop is crossed; the engine closes
    assert node.ctx(0).positions("A") == []


def test_a_held_rule_is_not_a_missing_order_at_reconcile():
    v, store = HoldingVenue(), []
    node = held_node(v, store)
    run_holder(node, v)
    assert node._strategies[0].handle.state == "venue"
    assert node.reconcile() == []


def test_a_restore_with_a_held_rule_reconciles_clean_and_goes_live():
    v, store = HoldingVenue(), []
    node = held_node(v, store)
    run_holder(node, v)
    restored = LiveNode.restore(
        None, [Holder()], ["A"], NO_FEES, journal=rj.Journal(HEADER, memory=True), records=list(store),
        execution=v, bar_stamp="close",
    )
    assert restored.state == "live", "reconcile did not call the rule's exit order missing"
    assert len(v.rules) == 1


def test_cancelling_a_held_rule_reaches_the_venue_and_ends_its_exit_orders():
    v = RuleVenue()
    node = trim_node(v)
    run_bars(node, v, 1, 4)
    s = node._strategies[0]
    assert s.handle.state == "venue"
    [rule] = v.rules
    s.cancel_exit(s.handle)
    assert v.cancelled == [rule] and s.handle.state == "canceled"
    cid = rule.exit_client_ids[0]
    assert dict((o[0], o[6]) for o in node._rt.venue_orders())[cid] == "canceled"
    assert node.reconcile() == []


def test_a_held_rule_the_venue_cannot_take_back_is_never_reported_cancelled():
    v = HoldingVenue()  # has place_exit_rule only
    node = trim_node(v)
    run_bars(node, v, 1, 4)
    s = node._strategies[0]
    with pytest.raises(StateError) as e:
        s.cancel_exit(s.handle)
    assert e.value.code == "exit_rule_not_cancellable"
    assert s.handle.state == "venue", "the venue still holds it"


def test_a_held_rule_follows_the_position_when_its_size_changes():
    v = RuleVenue()
    node = trim_node(v)
    run_bars(node, v, 1, 6)  # the sell of 2 fills on bar 5; the rule is checked on the next price
    [old] = v.rules
    assert old.legs == (("A", "sell", 5.0),)
    [(was, now)] = v.replaced
    assert was is old and now.legs == (("A", "sell", 3.0),)
    assert now.exit_client_ids != old.exit_client_ids
    orders_ = dict((o[0], o[6]) for o in node._rt.venue_orders())
    assert orders_[old.exit_client_ids[0]] == "canceled" and orders_[now.exit_client_ids[0]] == "submitted"
    # The venue fires the new rule's exit under the id it was given; it is booked.
    cid = now.exit_client_ids[0]
    node.push_report(cid, ExecutionReport.accepted("V-x", 8 * MIN), arrival_ns=8 * MIN)
    node.push_report(cid, ExecutionReport.fill("X-exit", 3.0, 90.0, 8 * MIN + 1), arrival_ns=8 * MIN + 1)
    assert node.ctx(0).positions("A") == []


def test_a_venue_that_cannot_replace_a_rule_has_it_cancelled_and_watched_by_the_engine():
    class NoReplace(RuleVenue):
        def replace_exit_rule(self, old, new):
            raise Unsupported("not in place")

    v = NoReplace()
    node = trim_node(v)
    run_bars(node, v, 1, 6)
    assert len(v.cancelled) == 1
    assert node._strategies[0].handle.state == "armed"


def test_a_held_rule_is_cancelled_at_the_venue_when_its_position_is_gone():
    class CloseOut(Trim):
        def on_bar(self, ctx, bar):
            super().on_bar(ctx, bar)
            if self.n == 5:
                self.submit_order(orders.Market(side="sell", units=3.0, reduce_only=True))

    v = RuleVenue()
    node = LiveNode([CloseOut()], ["A"], NO_FEES, bar_stamp="close", execution=v)
    run_bars(node, v, 1, 7)
    assert node.ctx(0).positions("A") == []
    assert len(v.cancelled) == 1
    assert node._strategies[0].handle.state == "done"


def test_an_exit_rule_whose_placement_has_no_answer_registers_its_orders_and_is_reported():
    class Silent(HoldingVenue):
        def place_exit_rule(self, request):
            self.rules.append(request)
            raise RuntimeError("connection reset after the request left")

    v = Silent()
    node = trim_node(v)
    with pytest.raises(StateError) as e:
        run_bars(node, v, 1, 3)
    assert e.value.code == "exit_rule_unconfirmed"
    s = node._strategies[0]
    h = s._exit_book().handles[0]
    assert h.state == "venue" and h.unconfirmed, "the engine does not watch a rule the venue may hold"
    [rule] = v.rules
    cid = rule.exit_client_ids[0]
    assert cid in node.health()["unconfirmed"]
    found = node.reconcile()
    assert [(d.kind, d.client_order_id) for d in found] == [("exit_rule_unconfirmed", cid)]
    # A fill of the exit it may hold is booked, not refused as an unknown order.
    node.push_report(cid, ExecutionReport.accepted("V-x", 6 * MIN), arrival_ns=6 * MIN)
    node.push_report(cid, ExecutionReport.fill("X-exit", 5.0, 90.0, 6 * MIN + 1), arrival_ns=6 * MIN + 1)
    assert node.ctx(0).positions("A") == []


def test_a_replay_of_a_rule_whose_result_was_never_written_does_not_watch_it_twice():
    v, store = HoldingVenue(), []
    node = held_node(v, store)
    run_holder(node, v)
    kept = [json.loads(c) for c in store]
    cut = max(i for i, r in enumerate(kept) if r["kind"] == "adapter_result" and r["payload"].get("op") == "place_exit_rule")
    records = [c for c in store[:cut]]  # the process died between the call and the record
    restored = LiveNode.restore(
        None, [Holder()], ["A"], NO_FEES, journal=rj.Journal(HEADER, memory=True), records=records,
        execution=v, bar_stamp="close", reconcile="skip",
    )
    handle = restored._strategies[0].handle
    assert handle.state == "venue" and handle.unconfirmed
    assert len(v.rules) == 1, "the venue was not asked again"


# -- 6. positions keyed by the broker's key, and cash like with like -------------------------------------


def test_a_position_the_broker_names_by_its_own_key_reads_as_the_engines_symbol():
    venue = ScriptedVenue()
    base = type(broker_adapter(venue))

    class Keyed(base):
        def fetch_positions(self):
            return [VenuePosition("X:A", 5.0, 100.0)]

    clock = iter(range(1, 10**9))
    ad = Keyed(Session("t"), lambda symbol: Instrument("X", symbol) if symbol == "A" else None,
               sleep=lambda s: None, clock_ns=lambda: next(clock), namespace="CF", trusted=True)
    assert [p.symbol for p in ad.positions()] == ["X:A"], "nothing has said which symbol X:A is"
    ad.drain_events()
    LiveNode([Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=ad)  # a node tells its adapter its symbols
    assert [p.symbol for p in ad.positions()] == ["A"]


def test_a_position_that_maps_to_no_symbol_of_the_node_is_a_discrepancy_not_a_skipped_row():
    class Foreign(MockVenue):
        def positions(self):
            return {"X:ZZ": 3.0}

    v = Foreign()
    from test_journal import build

    node = build(v, rj.Journal(HEADER, memory=True))
    found = node.reconcile()
    assert [(d.kind, d.symbol, d.actual) for d in found if d.kind == "unmapped_position"] == [("unmapped_position", "X:ZZ", 3.0)]


def test_a_broker_balance_is_compared_with_the_field_that_means_the_nodes_cash():
    cash, snap = venue_cash(VenueBalances(available=60.0, currency="INR", used=40.0, total=100.0))
    assert (cash, snap["compared"]) == (100.0, "total")
    cash, snap = venue_cash(VenueBalances(available=60.0, currency="INR", used=40.0))
    assert (cash, snap["compared"]) == (100.0, "available + used")
    cash, snap = venue_cash(VenueBalances(available=60.0, currency="INR"))
    assert cash is None and "compared" not in snap, "available alone is margin-adjusted: never compared as cash"
    assert venue_cash({"cash": 7})[0] == 7.0


def test_a_cash_mismatch_names_the_field_and_the_tolerance():
    class Short(MockVenue):
        def balances(self):
            return VenueBalances(available=1.0, currency="INR", total=5.0)

    from test_journal import build

    node = build(Short(), rj.Journal(HEADER, memory=True))
    [d] = [d for d in node.reconcile(cash_tolerance=0.5) if d.kind == "cash_mismatch"]
    assert "total" in d.detail and "tolerance 0.5" in d.detail
    rec = [r for r in node.journal.since(0) if r.kind == "reconcile"][-1]
    assert rec.payload["venue"]["balances"]["compared"] == "total"


# -- 7. rolls ---------------------------------------------------------------------------------------------


class RefuseNewContract(MockVenue):
    def submit(self, request):
        if request.symbol == "FB":
            raise OrderRefused("no_new_contract")
        super().submit(request)


def test_a_rejected_roll_open_is_reported_as_a_failed_roll():
    v = RefuseNewContract()
    node = roll_node(v)
    drive_roll(node, v)
    s = node._strategies[0]
    assert [(c, st) for c, st, _ in s.failed] == [("fut:roll0:o0", "rejected")]
    assert node.ctx(0).positions("FA") == [] and node.ctx(0).positions("FB") == []


def test_an_accepted_roll_open_ends_the_job_and_reports_nothing():
    v = MockVenue()
    node = roll_node(v)
    drive_roll(node, v)
    s = node._strategies[0]
    assert s.failed == []
    assert s._roll_state()[3] == {}, "no job is left once the open filled"


def test_the_roll_open_is_sized_in_the_new_contracts_lot():
    class Lots(Carry):
        def on_bar(self, ctx, bar):
            if ctx.symbol != "IDX":
                return
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Market(side="buy", units=4.0, instrument=ctx.role("fut")), client_id="open")

    from test_recovery_gate import EXPA, EXPB, T0C

    specs = {
        "IDX": raptorbt.InstrumentSpec.equity("IDX"),
        "FA": raptorbt.InstrumentSpec.futures_contract("FA", EXPA, 2.0, underlying="IDX"),
        "FB": raptorbt.InstrumentSpec.futures_contract("FB", EXPB, 3.0, underlying="IDX"),
    }
    v = MockVenue()
    node = LiveNode([Lots()], ["IDX", "FA", "FB"], NO_FEES, bar_stamp="close", execution=v,
                    instruments=specs, bindings={"index": "IDX"})
    drive_roll(node, v)
    assert [p.size for p in node.ctx(0).positions("FB")] == [6.0], "two lots of 2 became two lots of 3"
    opened = [c for c in v.sent if v.orders[c]["symbol"] == "FB"]
    assert [v.orders[c]["quantity"] for c in opened] == [6.0]


# -- 8. a refused exit close backs off and gives up loudly ---------------------------------------------------


def test_a_refused_exit_close_gives_up_loudly_after_a_bounded_number_of_attempts():
    v = RefuseReduceOnly()
    node = LiveNode([Guard()], ["A"], NO_FEES, bar_stamp="close", execution=v)
    with pytest.raises(StateError) as e:
        for i, p in enumerate([100, 100] + [80] * 40, start=1):
            node.push_bar("A", i * MIN, p, p, p, p, 1)
            v.fill_all()
            v.deliver(node, i * MIN + 5)
    assert e.value.code == "exit_close_failed"
    h = node._strategies[0].handle
    assert h.attempts == MAX_CLOSE_ATTEMPTS and h.state == "failed"
    assert len(h.refused) == MAX_CLOSE_ATTEMPTS
    assert node.ctx(0).positions("A"), "the position is still open, and the strategy was told"


def test_a_strategy_hears_that_an_exit_failed_and_the_run_goes_on():
    heard = []

    class Told(Guard):
        def on_exit_failed(self, ctx, handle):
            heard.append((handle.id, handle.attempts))

    v = RefuseReduceOnly()
    node = LiveNode([Told()], ["A"], NO_FEES, bar_stamp="close", execution=v)
    for i, p in enumerate([100, 100] + [80] * 40, start=1):
        node.push_bar("A", i * MIN, p, p, p, p, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    assert heard == [(node._strategies[0].handle.id, MAX_CLOSE_ATTEMPTS)]
    assert len(node._strategies[0].handle.close_ids) == MAX_CLOSE_ATTEMPTS  # one close per attempt, no more


def test_refused_closes_back_off_instead_of_resending_on_every_price():
    v = RefuseReduceOnly()
    node = LiveNode([Guard()], ["A"], NO_FEES, bar_stamp="close", execution=v)
    ms = 10**6
    node.push_bar("A", 100 * ms, 100, 100, 100, 100, 1)
    v.fill_all(); v.deliver(node, 100 * ms + 5)
    node.push_bar("A", 200 * ms, 100, 100, 100, 100, 1)
    v.fill_all(); v.deliver(node, 200 * ms + 5)
    for i in range(30):  # 30 prices, 100 ms apart: three seconds
        t = 300 * ms + i * 100 * ms
        node.push_bar("A", t, 80, 80, 80, 80, 1)
        v.fill_all(); v.deliver(node, t + 5)
    h = node._strategies[0].handle
    assert 1 < h.attempts <= 3, h.attempts  # not one per price


# -- 11. hedging names its position ----------------------------------------------------------------------


class Capturing(MockVenue):
    def __init__(self):
        super().__init__()
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        super().submit(request)


class HedgeThenClose(Strategy):
    def __init__(self):
        super().__init__()
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n in (1, 2):
            self.submit_order(orders.Market(side="buy", units=float(self.n)))
        if self.n == 4:
            ps = ctx.positions("A")
            self.close_position(position_id=ps[0].position_id, symbol="A")


def _run_hedge(oms):
    v = Capturing()
    node = LiveNode([HedgeThenClose()], ["A"], NO_FEES, bar_stamp="close", execution=v, oms_type=oms)
    for i in range(1, 7):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    return node, v


def test_a_close_under_hedging_names_the_position_it_ends():
    node, v = _run_hedge("hedging")
    entries = [r for r in v.requests if not r.reduce_only]
    [close] = [r for r in v.requests if r.reduce_only]
    assert all(r.position_id is None for r in entries)
    assert close.position_id is not None
    left = node.ctx(0).positions("A")
    assert [p.size for p in left] == [2.0] and close.position_id not in [p.position_id for p in left]


def test_a_node_that_nets_never_names_a_position():
    class NetClose(HedgeThenClose):
        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Market(side="buy", units=3.0))
            if self.n == 3:
                self.close_position(position_id=ctx.positions("A")[0].position_id, symbol="A")

    v = Capturing()
    node = LiveNode([NetClose()], ["A"], NO_FEES, bar_stamp="close", execution=v)
    for i in range(1, 6):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1)
        v.fill_all()
        v.deliver(node, i * MIN + 5)
    assert all(r.position_id is None for r in v.requests)
    assert node.ctx(0).positions("A") == []


def test_a_hedging_node_on_an_adapter_whose_venue_nets_is_refused_when_it_is_built():
    venue = ScriptedVenue()
    ad = broker_adapter(venue)
    assert ad.supports_hedging is False
    with pytest.raises(InvalidParameter) as e:
        LiveNode([Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=ad, oms_type="hedging")
    assert e.value.code == "hedging_unsupported_by_venue"
    LiveNode([Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=ad)  # netting is fine


def test_a_venue_that_keeps_positions_apart_may_run_a_hedging_node():
    LiveNode([Buyer()], ["A"], NO_FEES, bar_stamp="close", execution=MockVenue(), oms_type="hedging")


def test_an_adapter_that_nets_refuses_an_order_that_names_a_position():
    _, v = _run_hedge("hedging")
    [close] = [r for r in v.requests if r.reduce_only]
    ad = broker_adapter(ScriptedVenue())
    with pytest.raises(Unsupported, match="nets positions"):
        ad.submit(close)
