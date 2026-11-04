"""Netting accumulates like a broker's net position.

Under ``oms_type="netting"`` an instrument has one position. An order on its side adds to it
(size adds, the entry price becomes the size-weighted average, each fill is charged on its own);
an order against it reduces it, closes it, or flips it (closes it and opens the remainder on its
own side). Hedging keeps a position per order. The same rules hold in a batch run and on a node
with a simulated venue.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders

MIN = 60 * 10**9


def bars(opens):
    """Bars whose open, high, low and close are all the same price, stamped at their close."""
    p = np.asarray(opens, dtype=np.float64)
    n = len(p)
    return {
        "timestamps": (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": p.copy(), "high": p.copy(), "low": p.copy(), "close": p.copy(),
        "volume": np.full(n, 1_000_000.0),
    }


class Script(Strategy):
    """Submits ``plan[idx]`` (a list of ``Order`` objects or callables of the strategy) on bar
    ``idx`` and notes the positions, after the bar's fills, every bar."""

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self.held = []
        self.rejected = []

    def on_bar(self, ctx, bar=None):
        self.held.append([(p.direction, p.size, p.entry_price) for p in ctx.positions()])
        for order in self.plan.get(ctx.idx, ()):
            self.submit_order(order(self) if callable(order) else order)

    def on_order_rejected(self, ctx, event):
        self.rejected.append(event.reject_reason)


def buy(units, **kw):
    return orders.Market(side="buy", units=units, **kw)


def sell(units, **kw):
    return orders.Market(side="sell", units=units, **kw)


def run(plan, opens, *, fees=0.0, oms="netting", **kw):
    s = Script(plan)
    config = BacktestConfig(fees=fees, initial_capital=1_000_000.0)
    r = run_strategies([(s, ["X"])], {"X": bars(opens)}, config, oms_type=oms, bar_stamp="close", **kw)[0]
    return s, r.result


def test_a_second_order_on_the_side_held_adds_at_the_average_price():
    # decided on bar 0 and 2, filled at the next bar's open: 100 and 110
    s, result = run({0: [buy(10)], 2: [buy(10)]}, [100, 100, 110, 110, 110])
    assert s.rejected == []
    assert s.held[2] == [(1, 10.0, 100.0)]
    assert s.held[4] == [(1, 20.0, 105.0)]
    orders_ = [o for o in result.orders() if o.status == "filled"]
    assert [o.filled_qty for o in orders_] == [10.0, 10.0]


def test_each_fill_is_charged_on_its_own_and_the_trade_carries_the_sum():
    fee = 0.001
    s, result = run({0: [buy(10)], 2: [buy(10)], 4: [sell(20)]}, [100, 100, 110, 110, 120, 120], fees=fee)
    (trade,) = result.trades()
    assert (trade.size, trade.entry_price, trade.exit_price) == (20.0, 105.0, 120.0)
    assert trade.entry_fees == pytest.approx(fee * (10 * 100 + 10 * 110))
    assert trade.exit_fees == pytest.approx(fee * 20 * 120)
    assert trade.pnl == pytest.approx(20 * 15 - trade.fees)


def test_a_smaller_opposite_order_reduces_at_the_average_entry():
    s, result = run({0: [buy(10)], 2: [buy(10)], 4: [sell(5)]}, [100, 100, 110, 110, 120, 120, 120])
    trade, rest = result.trades()
    # booked against the average entry (105), as a broker's net position is
    assert (trade.size, trade.entry_price, trade.exit_price) == (5.0, 105.0, 120.0)
    assert trade.pnl == pytest.approx(5 * 15)
    assert (rest.size, rest.entry_price, rest.exit_reason) == (15.0, 105.0, "EndOfData")
    assert s.held[6] == [(1, 15.0, 105.0)]


def test_a_larger_opposite_order_flips_the_position():
    s, result = run({0: [buy(10)], 2: [sell(25)]}, [100, 100, 110, 110, 120, 120])
    closed, still_open = sorted(result.trades(), key=lambda t: t.entry_idx)
    assert (closed.size, closed.direction, closed.entry_price, closed.exit_price) == (10.0, 1, 100.0, 110.0)
    assert closed.pnl == pytest.approx(100.0)
    # the remainder is a short of 15 opened at the same fill, still held (ended at the data's end)
    assert (still_open.size, still_open.direction, still_open.entry_price) == (15.0, -1, 110.0)
    assert s.held[3] == [(-1, 15.0, 110.0)]
    assert s.rejected == []
    (flip,) = [o for o in result.orders() if o.requested_qty == 25.0]
    assert (flip.status, flip.filled_qty) == ("filled", 25.0)


def test_a_flip_pays_the_close_and_the_open_each_its_fee():
    fee = 0.001
    s, result = run({0: [buy(10)], 2: [sell(25)]}, [100, 100, 110, 110, 110], fees=fee)
    closed = min(result.trades(), key=lambda t: t.entry_idx)
    assert closed.exit_fees == pytest.approx(fee * 10 * 110)
    opened = max(result.trades(), key=lambda t: t.entry_idx)
    assert opened.entry_fees == pytest.approx(fee * 15 * 110)


def test_a_flip_whose_rest_cannot_open_ends_filled_at_what_it_closed():
    config = BacktestConfig(fees=0.0, initial_capital=1_000_000.0, max_position_qty=12.0)
    s = Script({0: [buy(10)], 2: [sell(25)]})
    r = run_strategies([(s, ["X"])], {"X": bars([100, 100, 110, 110, 110])}, config, bar_stamp="close")[0].result
    (flip,) = [o for o in r.orders() if o.client_id == "O-1"]
    # never rejected after fills: done at the 10 it closed, its quantity cut to that
    assert (flip.status, flip.filled_qty, flip.requested_qty) == ("filled", 10.0, 10.0)
    assert [(t.size, t.direction) for t in r.trades()] == [(10.0, 1)]
    assert s.held[4] == []


def test_a_reduce_only_order_is_never_a_flip():
    s, result = run({0: [buy(10)], 2: [sell(25, reduce_only=True)]}, [100, 100, 110, 110, 110])
    (trade,) = result.trades()
    assert trade.size == 10.0
    assert s.held[3] == []


def test_a_capital_fraction_order_adds_too():
    s, result = run(
        {0: [buy(10)], 2: [orders.Market(side="buy", size_frac=0.1)]}, [100, 100, 100, 100, 100],
    )
    assert s.rejected == []
    (direction, size, price) = s.held[4][0]
    assert direction == 1 and size > 10.0 and price == pytest.approx(100.0)


def test_hedging_keeps_a_position_per_order():
    s, result = run({0: [buy(10)], 2: [buy(10)]}, [100, 100, 110, 110, 110], oms="hedging")
    assert s.held[4] == [(1, 10.0, 100.0), (1, 10.0, 110.0)]


def test_a_bracket_protects_the_net_position_whichever_slice_it_came_with():
    """Bracket legs close the full position: after an add they close the net position."""

    # the second order carries a bracket: stop 90 and target 130 on the full position
    class Bracketed(Script):
        def on_bar(self, ctx, bar=None):
            self.held.append([(p.direction, p.size, p.entry_price) for p in ctx.positions()])
            if ctx.idx == 0:
                self.submit_order(buy(10))
            if ctx.idx == 2:
                self.submit_bracket(buy(10), stop_trigger=90.0, target_price=130.0)

    b = Bracketed({})
    config = BacktestConfig(fees=0.0, initial_capital=1_000_000.0)
    opens = [100, 100, 110, 110, 135, 135]
    data = {"X": bars(opens)}
    data["X"]["high"] = np.array([100, 100, 110, 110, 140, 140], dtype=np.float64)
    r = run_strategies([(b, ["X"])], data, config, bar_stamp="close")[0].result
    (trade,) = r.trades()
    assert (trade.size, trade.entry_price, trade.exit_reason) == (20.0, 105.0, "Order")
    assert trade.exit_price == pytest.approx(130.0)


def test_an_added_order_that_names_a_stop_sets_the_position_stop():
    class Stops(Script):
        def on_bar(self, ctx, bar=None):
            snap = ctx.positions()
            self.held.append([(p.size, p.stop_price) for p in snap])
            if ctx.idx == 0:
                self.submit_order(buy(10, stop_price=95.0))
            if ctx.idx == 2:
                self.submit_order(buy(10, stop_price=105.0))

    s = Stops({})
    run_strategies([(s, ["X"])], {"X": bars([100, 100, 110, 110, 110])},
                   BacktestConfig(fees=0.0, initial_capital=1_000_000.0), bar_stamp="close")
    assert s.held[2] == [(10.0, 95.0)]
    assert s.held[4] == [(20.0, 105.0)]


def test_an_add_is_refused_when_the_strategy_is_halted():
    class Halting(Script):
        def on_bar(self, ctx, bar=None):
            super().on_bar(ctx, bar)
            if ctx.idx == 1:
                ctx.halt("stop trading", flatten=False)

    s = Halting({0: [buy(10)], 2: [buy(10)]})
    run_strategies([(s, ["X"])], {"X": bars([100, 100, 110, 110, 110])},
                   BacktestConfig(fees=0.0, initial_capital=1_000_000.0), bar_stamp="close")
    assert s.rejected == ["killed"]
    assert s.held[4] == [(1, 10.0, 100.0)]


def test_a_paper_node_books_the_same_adds_reductions_and_flips():
    plan = {0: [buy(10)], 2: [buy(10)], 4: [sell(5)], 6: [sell(40)]}
    opens = [100, 100, 110, 110, 120, 120, 125, 125, 125]
    batch, batch_result = run(plan, opens)
    config = BacktestConfig(fees=0.0, initial_capital=1_000_000.0)
    node_strategy = Script(plan)
    node = LiveNode([(node_strategy, ["X"])], ["X"], config, bar_stamp="close", execution=PaperExecution())
    data = bars(opens)
    for i in range(len(opens)):
        node.push_bar("X", int(data["timestamps"][i]), opens[i], opens[i], opens[i], opens[i], 1_000_000.0)
    (node_result,) = node.finish()
    key = lambda t: (t.entry_idx, t.exit_idx, t.direction, t.size, t.entry_price, t.exit_price, t.pnl)
    assert sorted(map(key, node_result.result.trades())) == sorted(map(key, batch_result.trades()))
    assert node_strategy.held == batch.held
