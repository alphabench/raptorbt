"""Nothing a strategy does goes unrecorded, from its first act; a cut the engine
makes to a working order is an event, a journal record and replays identically.

* what ``on_start`` does (an order, a halt, a position adopted at build) is in the
  journal after ``start`` and before any input, also through muted warm-up, which
  used to discard an order made at start without a trace;
* the one-cancels-other cut of a sibling on a partial fill is an ``order_resized``
  event heard in ``on_order_event`` by the batch runner and the live node alike,
  journaled, and reproduced by a replay.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.strategy import orders

from test_journal import CONFIG, HEADER, MIN, SYMBOLS, MockVenue, settle

NS = 1_000_000_000


def kinds(j):
    return [r.kind for r in j.since(0)]


# -- on_start ---------------------------------------------------------------------


class Boots(Strategy):
    """Orders and halts nothing but what ``on_start`` makes it."""

    def __init__(self, halt=False):
        super().__init__()
        self.halt = halt

    def on_start(self, ctx):
        self.submit_order(orders.Market(side="buy", units=3.0))
        if self.halt:
            ctx.halt("boot")

    def on_bar(self, ctx, bar):
        pass


def _warm(n=4):
    p = np.full(n, 100.0)
    return {"A": dict(timestamps=np.arange(1, n + 1) * MIN, open=p, high=p + 1, low=p - 1,
                      close=p, volume=np.full(n, 10.0))}


def test_a_halt_made_in_on_start_is_journaled_after_start_and_before_any_input():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    LiveNode([Boots(halt=True)], SYMBOLS, CONFIG, bar_stamp="close", execution=venue, journal=j)
    assert kinds(j) == ["start", "halt"]
    halt = j.since(1)[0]
    assert halt.source == "strategy" and halt.payload["reason"] == "boot"


def test_a_position_adopted_at_build_is_journaled_after_start():
    j = rj.Journal(HEADER, memory=True)
    LiveNode(
        [Boots()], SYMBOLS, CONFIG, bar_stamp="close", journal=j,
        initial_positions={0: {"A": {"quantity": 5.0, "avg_price": 100.0, "timestamp_ns": 7}}},
    )
    assert kinds(j)[:2] == ["start", "adoption"]
    p = j.since(1)[0].payload
    assert (p["quantity"], p["price"], p["ts_ns"]) == ("5", "100", 7)


@pytest.mark.parametrize("warmup", ["muted", "live"])
def test_an_order_made_in_on_start_is_journaled_and_sent_at_the_first_live_event(warmup):
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    kw = {"warmup": warmup, "warmup_bars": _warm()}
    if warmup == "live":
        # A real venue is never sent an order made on history: run it on paper.
        kw = {"warmup": None}
    node = LiveNode(
        [Boots()], SYMBOLS, CONFIG, bar_stamp="close", execution=venue, journal=j, **kw
    )
    assert venue.sent == []
    node.push_bar("A", 10 * MIN, 100.0, 101.0, 99.0, 100.0, 10.0, arrival_ns=10 * MIN)
    assert venue.sent == ["s0-0"]
    k = kinds(j)
    i = k.index("command")
    assert k[i - 1] == "bar" and k[i + 1 : i + 3] == ["adapter_request", "adapter_result"]
    assert j.since(i)[0].payload["commands"][0]["command"] == "submit"


def test_the_replay_of_a_node_that_ordered_in_on_start_reproduces_every_record():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode(
        [Boots(halt=False)], SYMBOLS, CONFIG, bar_stamp="close", execution=venue, journal=j,
        warmup="muted", warmup_bars=_warm(),
    )
    node.push_bar("A", 10 * MIN, 100.0, 101.0, 99.0, 100.0, 10.0, arrival_ns=10 * MIN)
    venue.fill_all()
    venue.deliver(node, 11 * MIN)
    want = node.state_digest()
    records = list(j.since(0))
    j2 = rj.Journal(HEADER, memory=True)
    back = LiveNode.restore(
        None, [Boots()], SYMBOLS, CONFIG, journal=j2, records=records, execution=venue,
        bar_stamp="close", warmup="muted", warmup_bars=_warm(),
    )
    settle(back, venue)
    assert back.state_digest() == want
    assert [r.canonical() for r in j2.since(0)][: len(records)] == [r.canonical() for r in records]
    assert venue.sent == ["s0-0"], "recovery sent nothing again"


# -- the one-cancels-other cut ----------------------------------------------------------


class Bracketer(Strategy):
    """Buys 100, then offers a target and a stop for all 100 as alternatives."""

    def __init__(self):
        super().__init__()
        self.n = 0
        self.stop = None
        self.heard = []

    def on_trade(self, ctx, tick):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=100.0))
        elif self.n == 3:
            target = self.submit_order(
                orders.Limit(side="sell", price=110.0, units=100.0, reduce_only=True)
            )
            self.stop = self.submit_order(
                orders.StopMarket(side="sell", trigger=90.0, units=100.0, reduce_only=True)
            )
            self.link_oco(target, self.stop)

    def on_order_event(self, ctx, event):
        if event.kind == "order_resized":
            self.heard.append((event.kind, event.client_order_id, event.size, event.previous_size))


#: (ltp, ltq): the entry fills whole, the target trades 40 of its 100, then the rest.
TICKS = [(100.0, 100.0), (100.0, 1.0), (100.0, 1.0), (110.0, 40.0), (110.0, 60.0)]


def _cfg():
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0, slippage=0.0, partial_fills=True)


def _batch():
    n = len(TICKS)
    ticks = {
        "A": {
            "timestamps": (np.arange(n) * NS).astype(np.int64),
            "ltp": np.array([t[0] for t in TICKS]),
            "ltq": np.array([t[1] for t in TICKS]),
            "bid": np.array([t[0] - 0.5 for t in TICKS]),
            "ask": np.array([t[0] + 0.5 for t in TICKS]),
        }
    }
    s = Bracketer()
    raptorbt.run_strategies([s], ticks=ticks, config=_cfg())
    return s


def _live(journal=None, symbol="A"):
    s = Bracketer()
    node = LiveNode([s], ["A"], _cfg(), journal=journal)
    for i, (ltp, ltq) in enumerate(TICKS):
        node.push_tick(symbol, i * NS, ltp, ltp - 0.5, ltp + 0.5, ltq=ltq)
    return node, s


def test_the_cut_of_an_oco_sibling_is_an_event_heard_alike_in_a_batch_and_on_a_node():
    b = _batch()
    assert b.heard == [("order_resized", b.stop, 60.0, 100.0)]
    node, s = _live()
    assert s.heard == [("order_resized", s.stop, 60.0, 100.0)]


def test_the_cut_is_a_journal_record_and_a_replay_reproduces_it():
    j = rj.Journal(HEADER, memory=True)
    node, s = _live(j)
    cuts = [r for r in j.since(0) if r.kind == "order_resized"]
    assert len(cuts) == 1
    cut = cuts[0]
    assert cut.source == "engine" and cut.client_order_id == s.stop
    assert (cut.payload["from"], cut.payload["to"]) == ("100", "60")
    # It follows the print that caused it and comes before the next input.
    ks = kinds(j)
    assert ks[ks.index("order_resized") - 1] == "tick"
    want = node.state_digest()
    records = list(j.since(0))
    j2 = rj.Journal(HEADER, memory=True)
    back = LiveNode.restore(None, [Bracketer()], ["A"], _cfg(), journal=j2, records=records)
    assert back.strategies[0].heard == s.heard
    assert back.state_digest() == want
    assert [r.canonical() for r in j2.since(0)] == [r.canonical() for r in records]
