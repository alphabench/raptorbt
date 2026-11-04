"""A target meets the market once, on the first bar that examines it.

A resting limit arrives at the open of the first bar that looks at it: if that bar opens through
the level it trades at the open, and from then on it trades at its level. The price the position
was entered at plays no part (it used to: a target through the entry price was marketable, one
set through the market mid-trade was not, one equal to the entry was). The order matcher on the
paper venue works this way for every limit, so a backtest and a paper run now agree.
"""
import numpy as np
import pytest
from test_venue_parity import MIN, T0, bars, run

from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders as O


def _cfg(timing):
    return lambda: BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing)


class _Mid(Strategy):
    """Buys on bar 1; sets a target mid-trade on bar 3 (only the single-instrument runner moves levels;
    its hook runs before the bar is stepped, so the market it meets is the previous close)."""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=10.0))
        if ctx.idx == 3:
            ctx.set_target_price(self.target)


def _mid(rows, target, timing):
    a = np.array(rows, float)
    n = len(rows)
    ts = np.array([T0 + (i + 1) * MIN for i in range(n)], dtype=np.int64)
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing)
    r = run_strategy_backtest(_Mid(target), ts, a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.full(n, 1e6),
                              config=cfg, bar_stamp="close")
    return r.trades()


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
def test_a_target_set_through_the_market_mid_trade_trades_at_the_open(timing):
    # long from 100; bar 2 closed at 110, so a target of 105 set on bar 3 is below the bid
    rows = [(100, 100.5, 99.5, 100)] * 2 + [(100, 111, 100, 110), (111, 112, 109, 111)] + [(111, 112, 109, 111)] * 3
    (t,) = _mid(rows, 105.0, timing)
    assert t.exit_price == pytest.approx(111.0)


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
def test_a_target_below_the_entry_is_met_like_any_other_on_its_first_bar(timing):
    # bar 2 closed at 90: a target of 95 set on bar 3 rests above the bid. Bar 3 opens at 98, through
    # it, and that first bar trades it at the open (a later bar would trade it at 95)
    rows = [(100, 100.5, 99.5, 100)] * 2 + [(95, 95, 89, 90), (98, 99, 97, 98)] + [(98, 99, 97, 98)] * 3
    (t,) = _mid(rows, 95.0, timing)
    assert t.exit_price == pytest.approx(98.0)


def test_a_target_that_the_first_bar_does_not_open_through_rests_at_its_level():
    # the first bar opens at 100, below the target of 105, and trades up through it: filled at 105
    rows = [(100, 100.5, 99.5, 100)] * 4 + [(100, 106, 100, 105)] + [(105, 106, 104, 105)] * 2
    (t,) = _mid(rows, 105.0, "same_bar_close")
    assert t.exit_price == pytest.approx(105.0)


@pytest.mark.parametrize("target", [100.0, 105.0])
def test_the_entry_price_makes_no_difference_a_gapped_first_bar_trades_at_its_open(target):
    # long at the 100 close with a target of 100 (equal to the entry) or 105 (above it); the next
    # bar opens at 107, through either: the same fill in a backtest, on a node and on paper
    rows = [(100, 100.5, 99.5, 100)] * 2 + [(107, 108, 106, 107)] * 4
    out = run({1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, target_price=target))},
              {"X": bars(rows, T0)}, _cfg("same_bar_close"))
    for path, res in out.items():
        (t,) = res.result.trades()
        assert (t.entry_price, t.exit_price) == (100.0, 107.0), (path, t.entry_price, t.exit_price)
