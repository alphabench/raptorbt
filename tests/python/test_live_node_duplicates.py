"""A bar is delivered once: a retried push is refused with a code, never delivered twice, and a
node a call poisoned says so in ``health()``."""

import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.errors import StateError
from raptorbt.execution import ExecutionReport, PaperExecution
from raptorbt.strategy import orders as O

from _market_harness import ist

MIN = 60_000_000_000


class Seer(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx):
        self.seen.append((ctx.symbol, ctx.timestamp // MIN, ctx.bar.close))


def _node():
    s = Seer()
    return s, LiveNode([s], ["A", "B"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")


def test_a_retried_push_is_refused_not_delivered_twice():
    s, node = _node()
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    with pytest.raises(StateError) as e:
        node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    assert e.value.code == "duplicate_bar"
    node.push_bar("A", 2 * MIN, 101, 101, 101, 101, 1)
    assert s.seen == [("A", 1, 100.0), ("A", 2, 101.0)]


def test_a_second_bar_with_other_values_at_the_same_time_is_refused_as_conflicting():
    s, node = _node()
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    with pytest.raises(StateError) as e:
        node.push_bar("A", MIN, 101, 101, 101, 101, 1)
    assert e.value.code == "conflicting_bar"
    assert s.seen == [("A", 1, 100.0)]


def test_a_repeated_bar_inside_a_group_is_refused_too():
    s, node = _node()
    node.begin_group(MIN)
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    with pytest.raises(StateError) as e:
        node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    assert e.value.code == "duplicate_bar"
    node.push_bar("B", MIN, 50, 50, 50, 50, 1)
    node.flush()
    assert sorted(s.seen) == [("A", 1, 100.0), ("B", 1, 50.0)]


def test_a_retried_group_delivers_nothing_the_second_time():
    s, node = _node()
    node.push_group(MIN, {"A": (100, 100, 100, 100, 1), "B": (50, 50, 50, 50, 1)})
    with pytest.raises(StateError) as e:
        node.push_group(MIN, {"A": (100, 100, 100, 100, 1), "B": (50, 50, 50, 50, 1)})
    assert e.value.code == "duplicate_bar"
    assert len(s.seen) == 2


def test_a_push_refused_before_delivery_can_be_retried_once_corrected():
    s, node = _node()
    with pytest.raises(Exception):
        node.push_bar("A", MIN, float("nan"), 100, 100, 100, 1)
    node.push_bar("A", MIN, 100, 100, 100, 100, 1)
    assert s.seen == [("A", 1, 100.0)]


# ---- health() on a poisoned node ---------------------------------------------------------


class Buys(Strategy):
    n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 2:
            self.submit_order(O.Market(side="buy", units=10.0), client_id="b1")


@pytest.mark.parametrize("execution", [None, PaperExecution()], ids=["simulated", "paper"])
def test_health_says_a_node_a_call_poisoned_is_poisoned_and_why(execution):
    presets = rb.FeeSchedule.exchange_presets()
    spec = rb.InstrumentSpec.equity("X", venue="NSE", session=rb.SessionSchedule.nse_equity())
    node = LiveNode([Buys()], ["X"], BacktestConfig(initial_capital=1e6, fee_schedule=presets),
                    bar_stamp="close", instruments={"X": spec}, execution=execution)
    t0 = ist(2023, 3, 30, 9, 15)
    node.push_bar("X", t0 + MIN, 100, 100, 100, 100, 1000)
    assert node.health()["poisoned"] is False and node.state == "live"
    failed = None
    for i in range(2, 6):
        try:
            node.push_bar("X", t0 + i * MIN, 100, 100, 100, 100, 1000)
        except Exception as e:  # noqa: BLE001 - the first error is the one health must keep
            failed = failed or e
        h = node.health()
        if h["poisoned"]:
            break
    h = node.health()
    assert h["poisoned"] is True and h["state"] == "poisoned"
    assert h["poison_reason"][0] == failed.code and h["poison_reason"][1]


# ---- push_report on a fill the book cannot take ----------------------------------------------


class _Overclose(Strategy):
    n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(O.Market(side="buy", units=10.0), client_id="a")
        if self.n == 3:  # two closes of one position: the venue fills both
            self.submit_order(O.Market(side="sell", units=10.0, reduce_only=True), client_id="b")
            self.submit_order(O.Market(side="sell", units=10.0, reduce_only=True), client_id="c")


def test_a_fill_the_book_could_not_take_yet_is_reported_deferred_not_applied():
    from test_structure_hedge_first import Broker

    broker = Broker()
    node = LiveNode([_Overclose()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                    execution=broker)
    seen, done = [], 0
    for i in range(1, 5):
        node.push_bar("A", i * MIN, 100, 100, 100, 100, 1000)
        while done < len(broker.requests):
            request = broker.requests[done]
            done += 1
            node.push_report(request.client_order_id, ExecutionReport.accepted("V", i * MIN), arrival_ns=i * MIN)
            seen.append(node.push_report(
                request.client_order_id, ExecutionReport.fill(f"e{done}", request.quantity, 100.0, i * MIN),
                arrival_ns=i * MIN))
    assert seen == ["applied", "applied", "deferred"]
    ((_, qty, price, reason),) = node.health()["unbooked_fills"]
    assert (qty, price, reason) == (10.0, 100.0, "a reduce-only fill arrived with no position to reduce")


# a hook that raises when a group is flushed has still taken the group's bars
def test_a_group_whose_flush_raised_refuses_the_retry_of_its_bars():
    class Boom(Strategy):
        def on_bar(self, ctx):
            raise RuntimeError("boom")

    node = LiveNode([Boom()], ["A", "B"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    bars = {"A": (100.0, 101.0, 99.0, 100.5, 10.0), "B": (50.0, 51.0, 49.0, 50.5, 10.0)}
    with pytest.raises(RuntimeError, match="boom"):
        node.push_group(60 * 10**9, bars)
    with pytest.raises(StateError) as e:
        node.push_group(60 * 10**9, bars)
    assert e.value.code == "duplicate_bar"
