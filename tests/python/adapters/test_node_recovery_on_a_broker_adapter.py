"""A node on a real ``BrokerAdapter`` recovers: resend, fills during downtime, a dying executor.

Every case runs a ``LiveNode`` on the reference ``BrokerAdapter`` over the scripted venue,
the adapter a mapping would build, not a double that re-sends whatever it is given.
"""

import time

import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.adapters import ManualExecutor, ThreadExecutor
from raptorbt.adapters.base import Transient
from raptorbt.adapters.conformance import ScriptedVenue, _reference_adapter
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import orders

from .test_execution_request import request

MIN = 60_000_000_000
CONFIG = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
HEADER = {"run_id": "r-1", "algo": "buyer", "version": 1}
CASH = 1_000_000.0


class Buyer(Strategy):
    """Buys 5 units on its first bar."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=5.0))


def push(node, i, price=100.0):
    node.push_bar("A", (i + 1) * MIN, price, price + 1, price - 1, price + 0.5, 1000.0)


def adapter_on(venue, positions=(), cash=CASH, **options):
    """The reference adapter, with the account reads a mapping would give it."""
    from raptorbt.adapters import VenuePosition

    adapter = _reference_adapter(venue, options)
    adapter.fetch_positions = lambda: [VenuePosition(s, q, 100.0) for s, q in positions]
    adapter.fetch_balances = lambda: {"cash": cash}
    return adapter


def started(adapter, journal=None):
    node = LiveNode([Buyer()], ["A"], CONFIG, bar_stamp="close", execution=adapter, journal=journal)
    push(node, 0)
    return node


def hear(venue, adapter, node, ref):
    """The venue accepts the order and the node hears it."""
    venue.subscribers.append(adapter.handle_payload)
    venue.accept(ref)
    venue.subscribers.clear()
    adapter.deliver_to(node)


# -- resend through a real adapter ------------------------------------------------------------


def test_a_resend_through_a_broker_adapter_places_an_order_the_venue_lost():
    venue = ScriptedVenue()
    adapter = adapter_on(venue)
    node = started(adapter, rj.Journal(HEADER, memory=True))
    ref = adapter.ids.encode("s0-0")
    hear(venue, adapter, node, ref)
    assert venue.place_calls[ref] == 1

    venue.orders.clear()  # the broker no longer holds it
    assert [d.kind for d in node.reconcile()] == ["missing_order"]

    node.resend("s0-0")
    assert venue.place_calls[ref] == 2 and len(venue.orders) == 1 and not venue.duplicates
    hear(venue, adapter, node, ref)
    assert node.reconcile() == []


def test_a_resend_of_an_order_the_venue_holds_places_nothing():
    venue = ScriptedVenue()
    adapter = adapter_on(venue)
    node = started(adapter, rj.Journal(HEADER, memory=True))
    ref = adapter.ids.encode("s0-0")
    hear(venue, adapter, node, ref)

    node.resend("s0-0")
    node.resend("s0-0")
    assert venue.place_calls[ref] == 1 and not venue.duplicates
    assert node.reconcile() == []


def test_two_resends_before_the_adapter_runs_are_one_request():
    venue = ScriptedVenue()
    adapter = adapter_on(venue, executor=ManualExecutor())
    node = started(adapter, rj.Journal(HEADER, memory=True))
    ref = adapter.ids.encode("s0-0")
    adapter.pump()
    venue.orders.clear()

    node.resend("s0-0")
    node.resend("s0-0")
    adapter.pump()
    assert venue.place_calls[ref] == 2


def test_a_resend_is_not_placed_when_the_book_cannot_be_read():
    venue = ScriptedVenue()
    adapter = adapter_on(venue)
    node = started(adapter, rj.Journal(HEADER, memory=True))
    ref = adapter.ids.encode("s0-0")
    adapter.drain_events()

    def down(tag):
        raise Transient("the book is down")

    adapter.find_by_tag = down
    node.resend("s0-0")
    # the order may be out there: nothing is placed on a guess, and the adapter says so
    assert venue.place_calls[ref] == 1
    assert [e.kind for e in adapter.drain_events()] == ["unconfirmed"]


def test_a_resend_is_not_placed_on_one_read_that_lags_the_venue():
    venue = ScriptedVenue()
    adapter = adapter_on(venue)
    node = started(adapter, rj.Journal(HEADER, memory=True))
    ref = adapter.ids.encode("s0-0")
    real = adapter.find_by_tag
    reads = []

    def lagging(tag):
        reads.append(tag)
        return [] if len(reads) == 1 else real(tag)  # the first read misses an order that is there

    adapter.find_by_tag = lagging
    node.resend("s0-0")
    assert len(reads) == 2 and venue.place_calls[ref] == 1 and not venue.duplicates


def test_a_fill_on_an_order_of_no_node_here_is_counted_and_named_at_reconcile():
    venue = ScriptedVenue()
    adapter = adapter_on(venue)
    node = started(adapter, rj.Journal(HEADER, memory=True))
    hear(venue, adapter, node, adapter.ids.encode("s0-0"))
    venue.preload("someone-elses", "A", "buy", 3.0, 100.0, 3.0)
    adapter.drain_events()
    node.reconcile()
    assert adapter.ignored_updates == 1
    assert [e.kind for e in adapter.drain_events()] == ["ignored_update"]


# -- a fill while the node was down, and the cash tolerance ----------------------------------


def downtime_story(*, cash):
    """A node accepted its order and died; the venue filled it 5 @ 100 meanwhile."""
    venue = ScriptedVenue()
    first = adapter_on(venue)
    journal = rj.Journal(HEADER, memory=True)
    node = started(first, journal)
    ref = first.ids.encode("s0-0")
    hear(venue, first, node, ref)
    records = journal.since(journal.seq_base)
    first.close()
    del node
    venue.fill(ref, 5.0, 100.0)  # nobody is listening
    second = adapter_on(venue, positions=[("A", 5.0)], cash=cash)
    return venue, second, records


def restore(second, records, **kw):
    seen = []
    node = LiveNode.restore(
        None, [Buyer()], ["A"], CONFIG, journal=rj.Journal(HEADER, memory=True), records=records,
        execution=second, bar_stamp="close", on_discrepancy=seen.append, **kw,
    )  # fmt: skip
    return node, seen


def test_a_fill_while_the_node_was_down_is_booked_and_the_reconcile_is_clean():
    _, second, records = downtime_story(cash=CASH - 500.0)
    node, seen = restore(second, records)
    assert seen == []
    assert node.state == "live"
    assert [(p.direction, p.size) for p in node.positions(0, "A")] == [(1, 5.0)]


def test_restore_takes_a_cash_tolerance():
    # the broker's balance is off by 0.40 from the node's cash
    _, second, records = downtime_story(cash=CASH - 500.0 - 0.4)
    node, seen = restore(second, records)
    assert [d.kind for d in seen] == ["cash_mismatch"] and node.state == "recovering"

    _, second, records = downtime_story(cash=CASH - 500.0 - 0.4)
    node, seen = restore(second, records, cash_tolerance=0.5)
    assert seen == [] and node.state == "live"


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "0.5", True])
def test_a_cash_tolerance_that_could_never_report_a_difference_is_refused(bad):
    _, second, records = downtime_story(cash=CASH - 500.0)
    with pytest.raises(InvalidParameter) as e:
        restore(second, records, cash_tolerance=bad)
    assert e.value.code == "invalid_parameter" and e.value.parameter == "cash_tolerance"


# -- an executor that meets an unexpected error --------------------------------------------------


def wait_for(condition, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_a_job_that_raises_what_is_not_an_adapter_error_is_shown_and_the_outbox_goes_on():
    venue = ScriptedVenue()
    real_place = venue.place
    calls = []

    def place(*args):
        calls.append(args)
        if len(calls) == 1:
            raise KeyError("a field the mapping expected")  # a bug in a mapping
        return real_place(*args)

    venue.place = place
    adapter = adapter_on(venue, executor=ThreadExecutor("t-fault"))

    class Two(Strategy):
        n = 0

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Market(side="buy", units=1.0), client_id="doomed")
                self.submit_order(orders.Market(side="buy", units=2.0), client_id="fine")

    try:
        node = LiveNode([Two()], ["A"], CONFIG, bar_stamp="close", execution=adapter)
        push(node, 0)
        assert wait_for(lambda: len(venue.orders) == 1), "the second order never reached the venue"
        health = node.health()["adapter"]
        assert health["faults"] == 1
        assert "KeyError" in health["recent_faults"][0]["error"]
        assert health["executor"]["alive"] is True
        assert health["outbox"] == 0 and health["paused"] is False
        # the order whose job failed is named, and neither rejected nor placed again
        assert health["recent_faults"][0]["client_order_id"] == "s0-0"
        assert not adapter.drain_reports()
    finally:
        adapter.close()


def test_a_failure_outside_any_job_does_not_end_the_worker():
    venue = ScriptedVenue()
    adapter = adapter_on(venue, executor=ThreadExecutor("t-net"))
    real_sweep = adapter.sweep
    failures = []

    def sweep():
        if not failures:
            failures.append(1)
            raise RuntimeError("housekeeping failed")
        return real_sweep()

    adapter.sweep = sweep
    try:
        adapter.submit(request())
        assert wait_for(lambda: adapter.health()["executor"]["errors"] == 1)
        assert adapter.health()["executor"]["alive"] is True
        assert "RuntimeError" in adapter.health()["executor"]["last_error"]
        adapter.submit(request(strategy=1))  # the worker is still there to run it
        assert wait_for(lambda: sum(venue.place_calls.values()) == 2)
    finally:
        adapter.close()
