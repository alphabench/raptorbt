"""A halt that flattens: on ``PaperExecution`` its own closes are sent and filled, and the orders
that would add exposure are cancelled, as with simulated execution."""

import pytest

from raptorbt import BacktestConfig, LiveNode
from raptorbt.execution import PaperExecution
from raptorbt.strategy import Strategy, orders as O

MIN = 60_000_000_000


class Halts(Strategy):
    def __init__(self, flatten):
        super().__init__()
        self.flatten = flatten

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=10.0), client_id="entry")
            self.submit_order(O.Limit(side="buy", units=5.0, price=50.0), client_id="resting")
        if ctx.idx == 3:
            ctx.halt("stop", self.flatten)


def _run(execution, flatten, timing="same_bar_close"):
    s = Halts(flatten)
    node = LiveNode([s], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing), bar_stamp="close", execution=execution)
    for i in range(8):
        node.push_bar("A", (i + 1) * MIN, 100 + i, 100 + i, 100 + i, 100 + i, 1000)
    r = node.finish()[0]
    return {o.client_id: o.status for o in r.result.orders()}, [(t.entry_idx, t.exit_idx, t.entry_price, t.exit_price) for t in r.result.trades()]


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
def test_a_flattening_halt_books_the_same_trade_on_paper_and_simulated(timing):
    sim_orders, sim_trades = _run(None, True, timing)
    paper_orders, paper_trades = _run(PaperExecution(), True, timing)
    assert paper_orders["resting"] == sim_orders["resting"] == "canceled"
    assert paper_trades == sim_trades
    assert len(sim_trades) == 1 and sim_trades[0][1] < 7, "closed at the halt, not held to the end"
    if timing == "same_bar_close":
        # Corrected (was the previous bar's close, 102): the halt is raised while deciding
        # bar 3, and closes as an ordinary reduce-only market order decided there does.
        assert sim_trades == [(1, 3, 101.0, 103.0)], "the close of the bar the halt was raised on"
    else:
        assert sim_trades == [(2, 4, 102.0, 104.0)], "the open of the bar after the halt, for an entry that itself filled at an open"


def test_a_halt_that_does_not_flatten_leaves_the_position_open_on_paper():
    orders, trades = _run(PaperExecution(), False)
    assert orders["resting"] == "canceled" and not [c for c in orders if c.startswith("close")]
    assert trades[0][1] == 7, "held to the end of the run"
