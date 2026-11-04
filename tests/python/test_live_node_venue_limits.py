"""The book's limits are applied before an order reaches a venue that does not apply them.

A venue that is not a simulation (here a plain adapter that only records what it is
sent) knows nothing of ``max_order_qty``, ``max_order_notional``, ``max_position_qty`` or
``max_gross_exposure``. Every case sends through such a venue and asks what it was sent:
the refusal happens in the node, before the adapter, and the strategy hears it as
``on_order_rejected``.

* a market order names no price, so it is priced at the last price seen;
* an order the venue is working, or has not yet answered, counts as if it had filled;
* a market order that cannot be priced at all is refused (``no_mark``), never passed.
"""

import sys
from pathlib import Path

from raptorbt import BacktestConfig
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from test_execution import MIN, Hooked, Venue, bar, node_with  # noqa: E402


class Listening(Hooked):
    """Submits what the script says, and keeps the code of every refusal."""

    def __init__(self, script=None):
        super().__init__(script)
        self.reasons = []

    def on_order_rejected(self, ctx, event):
        self.reasons.append((event.client_order_id, event.reject_reason))


def limited(**limits):
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0, **limits)


def sent(venue):
    return [r.tags["client_id"] for r in venue.requests]


def test_two_orders_that_fit_a_position_cap_alone_are_not_both_sent():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Market(side="buy", units=5.0), client_id="first")
            s.submit_order(orders.Market(side="buy", units=5.0), client_id="second")

    strategy = Listening(script)
    node = node_with(venue, [strategy], symbols=("A",), config=limited(max_position_qty=7.0))
    bar(node, "A", 0)
    bar(node, "A", 1)
    assert sent(venue) == ["first"]
    assert strategy.reasons == [("second", "max_position")]


def test_a_held_position_and_a_working_order_count_together_towards_the_cap():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Market(side="buy", units=4.0), client_id="held")
        if n == 2:
            s.submit_order(orders.Limit(side="buy", price=90.0, units=2.0), client_id="resting")
        if n == 3:
            s.submit_order(orders.Market(side="buy", units=2.0), client_id="over")
            s.submit_order(orders.Market(side="buy", units=1.0), client_id="exact")

    strategy = Listening(script)
    node = node_with(venue, [strategy], symbols=("A",), config=limited(max_position_qty=7.0))
    bar(node, "A", 0)
    request = venue.requests[0]
    node.push_report(request.client_order_id, ExecutionReport.accepted("V1", MIN), arrival_ns=MIN)
    node.push_report(
        request.client_order_id, ExecutionReport.fill("E1", 4.0, 100.0, MIN), arrival_ns=MIN
    )
    bar(node, "A", 1)
    bar(node, "A", 2)
    bar(node, "A", 3)
    # 4 held + 2 working = 6: two more would be 8 (refused), one more is exactly 7 (allowed)
    assert sent(venue) == ["held", "resting", "exact"]
    assert strategy.reasons == [("over", "max_position")]


def test_a_market_order_over_the_size_cap_is_refused_before_the_venue_sees_it():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Market(side="buy", units=10.0), client_id="big")
            s.submit_order(orders.Market(side="buy", units=5.0), client_id="fits")

    strategy = Listening(script)
    node = node_with(venue, [strategy], symbols=("A",), config=limited(max_order_qty=5.0))
    bar(node, "A", 0)
    bar(node, "A", 1)
    assert sent(venue) == ["fits"]
    assert strategy.reasons == [("big", "order_size")]


def test_a_market_order_is_priced_at_the_last_price_for_the_notional_cap():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            # the bar closes at 100.5: 10 units are 1005, 9 are 904.5
            s.submit_order(orders.Market(side="buy", units=10.0), client_id="over")
            s.submit_order(orders.Market(side="buy", units=9.0), client_id="under")

    strategy = Listening(script)
    node = node_with(venue, [strategy], symbols=("A",), config=limited(max_order_notional=1000.0))
    bar(node, "A", 0)
    bar(node, "A", 1)
    assert sent(venue) == ["under"]
    assert strategy.reasons == [("over", "order_size")]


def test_a_market_order_with_no_price_to_check_a_notional_cap_against_is_refused():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            # B has had no print: nothing prices this order
            s.submit_order(orders.Market(side="buy", units=1.0, instrument="B"), client_id="blind")

    strategy = Listening(script)
    node = node_with(venue, [strategy], config=limited(max_order_notional=1_000_000.0))
    bar(node, "A", 0)
    bar(node, "B", 1)  # the refusal is heard at the next event of the order's instrument
    assert sent(venue) == []
    assert strategy.reasons == [("blind", "no_mark")]


def test_a_market_order_with_no_price_is_sent_when_no_limit_needs_one():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Market(side="buy", units=1.0, instrument="B"), client_id="blind")

    strategy = Listening(script)
    node = node_with(venue, [strategy], config=limited(max_order_qty=5.0))
    bar(node, "A", 0)
    bar(node, "A", 1)
    assert sent(venue) == ["blind"]
    assert strategy.reasons == []


def test_orders_working_on_another_instrument_count_towards_gross_exposure():
    def run(cap):
        venue = Venue()

        def script(s, ctx, n):
            if n == 1:
                s.submit_order(orders.Market(side="buy", units=10.0), client_id="a")
            if n == 2:
                s.submit_order(orders.Market(side="buy", units=10.0), client_id="b")

        strategy = Listening(script)
        node = node_with(venue, [strategy], config=limited(max_gross_exposure=cap))
        bar(node, "A", 0)
        bar(node, "B", 1)
        bar(node, "A", 2)
        return sent(venue), strategy.reasons

    # each order is 10 x 100.5 = 1005; a is working when b is asked
    assert run(1500.0) == (["a"], [("b", "max_exposure")])
    assert run(2100.0) == (["a", "b"], [])
