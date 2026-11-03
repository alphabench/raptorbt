"""Gate fixes of the live node's groups and recovery, each from a failing run.

* a restore after a crash in the middle of a timestamp group reconciles, and the host completes
  the group it left open: the run then equals the one that was never interrupted;
* ``push_group`` validates every bar before it opens a group, so a bad bar delivers nothing and
  the retry delivers each good bar once;
* a restore refuses an adapter that neither adopts the node's working orders nor says it keeps
  its own records;
* pushes at one timestamp that are not grouped are counted and logged once.
"""

import logging

import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.adapters.conformance import ScriptedVenue
from raptorbt.errors import InvalidParameter, StateError

from test_journal import CONFIG, HEADER, MIN, MockVenue, SYMBOLS, Trader, build, run_step, script
from test_recovery_gate import Buyer, broker_adapter

CAPITAL = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)


def bar(price):
    return (price, price, price, price, 1.0)


# -- a crash in the middle of a group ---------------------------------------------------


def torn_journal():
    """A node whose process died after the first bar of a group, before the second: the journal
    holds ``group_begin`` and one bar, and no ``group_end``."""
    venue, store = ScriptedVenue(), []
    node = LiveNode(
        [Buyer()], SYMBOLS, CAPITAL, bar_stamp="close", execution=broker_adapter(venue),
        journal=rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())),
    )
    node.push_group(MIN, {"A": bar(100.0), "B": bar(100.0)})
    node.begin_group(2 * MIN)
    node.push_bar("A", 2 * MIN, *bar(101.0))
    return venue, store


def uninterrupted():
    venue = ScriptedVenue()
    node = LiveNode([Buyer()], SYMBOLS, CAPITAL, bar_stamp="close", execution=broker_adapter(venue),
                    journal=rj.Journal(HEADER, memory=True))
    node.push_group(MIN, {"A": bar(100.0), "B": bar(100.0)})
    node.push_group(2 * MIN, {"A": bar(101.0), "B": bar(101.0)})
    return node


def test_a_restore_after_a_crash_mid_group_reconciles_and_the_host_completes_the_group():
    venue, store = torn_journal()
    restored = LiveNode.restore(
        None, [Buyer()], SYMBOLS, CAPITAL, journal=rj.Journal(HEADER, memory=True),
        records=list(store), execution=broker_adapter(venue), bar_stamp="close",
    )
    # The reconcile ran against the fresh adapter and found nothing wrong.
    assert restored.state == "live"
    # The group is still open, and the node says which bars it holds, so the host sends the rest.
    assert restored.open_group == {"timestamp": 2 * MIN, "symbols": ["A"]}
    restored.push_bar("B", 2 * MIN, *bar(101.0))
    restored.flush()
    assert restored.open_group is None
    assert restored.state_digest() == uninterrupted().state_digest()


def test_a_node_with_no_open_group_reports_none():
    assert uninterrupted().open_group is None


# -- M1: push_group on a bad bar ----------------------------------------------------------------


class Seer(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx, bar):
        self.seen.append((ctx.symbol, ctx.timestamp // MIN))


def test_push_group_with_a_bad_bar_delivers_nothing_and_the_retry_delivers_each_bar_once():
    s = Seer()
    node = LiveNode([s], SYMBOLS, CAPITAL, bar_stamp="close")
    node.push_group(MIN, {"A": bar(100.0), "B": bar(50.0)})
    with pytest.raises(Exception) as e:
        node.push_group(2 * MIN, {"A": bar(101.0), "B": (float("nan"), 50.0, 50.0, 50.0, 1.0)})
    assert e.value.code == "non_finite_price"
    assert s.seen == [("A", 1), ("B", 1)], "the good bar of the failed group was not delivered"
    assert node.open_group is None, "no group was left open"
    node.push_group(2 * MIN, {"A": bar(101.0), "B": bar(51.0)})
    assert s.seen == [("A", 1), ("B", 1), ("A", 2), ("B", 2)]


def test_push_group_with_an_unknown_symbol_delivers_nothing():
    s = Seer()
    node = LiveNode([s], SYMBOLS, CAPITAL, bar_stamp="close")
    with pytest.raises(Exception) as e:
        node.push_group(MIN, {"A": bar(100.0), "Z": bar(50.0)})
    assert e.value.code == "unknown_symbol"
    assert s.seen == [] and node.open_group is None


def test_push_group_with_a_stale_bar_delivers_nothing():
    s = Seer()
    node = LiveNode([s], SYMBOLS, CAPITAL, bar_stamp="close")
    node.push_group(2 * MIN, {"A": bar(100.0), "B": bar(50.0)})
    seen = list(s.seen)
    with pytest.raises(Exception):
        node.push_group(MIN, {"A": bar(101.0), "B": bar(51.0)})
    assert s.seen == seen and node.open_group is None


# -- M3: an adapter that cannot adopt the node's orders ---------------------------------------------


def restore_with(adapter, store):
    return LiveNode.restore(
        None, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
        records=list(store), execution=adapter, bar_stamp="close",
    )


def journal_of_a_run(venue):
    store = []
    node = build(venue, rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())))
    for step in script(6):
        run_step(node, venue, step)
    return store


def test_restore_refuses_an_adapter_that_neither_adopts_orders_nor_keeps_its_own_records():
    class Undeclared(MockVenue):
        keeps_own_records = False

    store = journal_of_a_run(Undeclared())
    with pytest.raises(StateError) as e:
        restore_with(Undeclared(), store)
    assert e.value.code == "adapter_cannot_adopt"
    assert "restore_orders" in str(e.value) and "keeps_own_records" in str(e.value)


def test_restore_accepts_an_adapter_that_declares_it_keeps_its_own_records():
    venue = MockVenue()
    assert venue.keeps_own_records is True
    restored = restore_with(venue, journal_of_a_run(venue))
    assert restored.state == "live"


def test_restore_accepts_an_adapter_that_can_adopt_orders():
    venue = ScriptedVenue()
    store = []
    node = LiveNode([Buyer()], ["A"], CAPITAL, bar_stamp="close", execution=broker_adapter(venue),
                    journal=rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())))
    node.push_bar("A", MIN, *bar(100.0))
    again = LiveNode.restore(None, [Buyer()], ["A"], CAPITAL, journal=rj.Journal(HEADER, memory=True),
                             records=list(store), execution=broker_adapter(venue), bar_stamp="close")
    assert again.state == "live"


# -- M5: pushes at one timestamp that are not grouped -------------------------------------------------


def test_ungrouped_pushes_at_one_timestamp_are_counted_and_logged_once(caplog):
    node = LiveNode([Seer()], SYMBOLS, CAPITAL, bar_stamp="close")
    assert node.health()["ungrouped_same_timestamp_pushes"] == 0
    with caplog.at_level(logging.WARNING, logger="raptorbt.strategy.live_node"):
        node.push_bar("A", MIN, *bar(100.0))
        node.push_bar("B", MIN, *bar(50.0))
        node.push_bar("A", 2 * MIN, *bar(100.0))
        node.push_bar("B", 2 * MIN, *bar(50.0))
    mine = [r for r in caplog.records if "not grouped" in r.getMessage()]
    assert len(mine) == 1, "logged the first time only"
    assert "begin_group" in mine[0].getMessage() and "push_group" in mine[0].getMessage()
    assert node.health()["ungrouped_same_timestamp_pushes"] == 2


def test_grouped_pushes_are_not_counted():
    node = LiveNode([Seer()], SYMBOLS, CAPITAL, bar_stamp="close")
    node.push_group(MIN, {"A": bar(100.0), "B": bar(50.0)})
    node.push_group(2 * MIN, {"A": bar(100.0), "B": bar(50.0)})
    assert node.health()["ungrouped_same_timestamp_pushes"] == 0
