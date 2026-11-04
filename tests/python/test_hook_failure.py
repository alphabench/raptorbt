"""A bar whose handler raised is still applied to the books.

The handler decided nothing, so what it queued before it raised is dropped. The market the bar
describes happened all the same: a protective level it crossed fills, the equity is marked, and
the strategies that did not fail are matched. The error then reaches the caller.
"""
import pytest
from _market_harness import ist

from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.strategy import orders as O

MIN = 60_000_000_000
T0 = ist(2026, 11, 2, 9, 15)
ROWS = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 90, 92), (92, 93, 91, 92.5), (92, 93, 91, 92)]


class _Boom(Strategy):
    """Long 10 with a stop at 95; on bar 2 it queues a half-close and raises."""

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0))
        if ctx.idx == 2:
            self.submit_order(O.Market(side="sell", units=5.0, reduce_only=True), client_id="half")
            raise RuntimeError("boom")


def _cfg():
    return BacktestConfig(initial_capital=1e6, fees=0.0)


def test_a_bar_that_raised_still_trades_its_stop_and_marks_equity():
    node = LiveNode([_Boom()], ["A"], _cfg(), bar_stamp="close")
    for i, row in enumerate(ROWS):
        stamp = T0 + (i + 1) * MIN
        if i == 2:
            with pytest.raises(RuntimeError, match="boom"):
                node.push_bar("A", stamp, *row, 1000.0)
        else:
            node.push_bar("A", stamp, *row, 1000.0)
    result = node.finish()[0].result
    (trade,) = result.trades()
    # the stop (95) was crossed on bar 2 and traded there, not on the next bar's open (92);
    # the half-close the handler queued before it raised was dropped: the whole 10 closed at the stop
    assert (trade.size, trade.exit_price, trade.exit_time) == (10.0, 95.0, T0 + 3 * MIN)
    assert len(result.equity_curve()) == len(ROWS)


class _Quiet(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 2 and ctx.symbol == "B":
            self.submit_order(O.Market(side="buy", units=1.0), client_id="b")


class _BoomA(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 2 and ctx.symbol == "A":
            raise RuntimeError("boom")


def test_a_strategy_that_raised_does_not_stop_the_others_of_its_group_being_matched():
    a, b = _BoomA(), _Quiet()
    node = LiveNode([a, b], ["A", "B"], _cfg(), bar_stamp="close")
    for i, row in enumerate(ROWS[:4]):
        stamp = T0 + (i + 1) * MIN
        bars = {"A": row + (1000.0,), "B": row + (1000.0,)}
        if i == 2:
            with pytest.raises(RuntimeError, match="boom"):
                node.push_group(stamp, bars)
        else:
            node.push_group(stamp, bars)
    results = node.finish()
    (fill,) = [o for o in results[1].result.orders() if o.client_id == "b"]
    assert (fill.status, fill.filled_qty) == ("filled", 1.0)
    assert len(results[0].result.equity_curve()) == 4
