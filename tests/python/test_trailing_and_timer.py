"""A venue's trailing order tests before it ratchets, and a timer hook cannot read the bar
still forming. Every number is computed by hand from the bars written here. Bars are
1 minute, stamped at their close; fees are zero."""

import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders as O
from test_venue_parity import MIN, T0, bars, run


def nbo():
    return BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")


def closed(out):
    return {p: sorted((round(t.size, 6), round(t.exit_price, 6)) for t in r.result.trades()) for p, r in out.items()}


FLAT = (100, 101, 99, 100)


def trailing():
    c = nbo()
    c.set_trailing_stop(0.10)
    return c


def test_a_config_trailing_stop_on_paper_holds_where_the_batch_holds():
    """Engine only (the config sets no trailing stop). 10% trail, long at 100 on bar 2,
    stop 90.9 after bar 2. Bar 3 (195, 200, 175, 190): the batch checks the level the bar found
    (90.9), holds, then ratchets to 180; bar 4 (low 185) and bar 5 (low 185) hold. Before: paper's
    matcher ratcheted to 180 first and sold at 180 on bar 3."""
    rows = [FLAT, FLAT, FLAT, (195, 200, 175, 190), (190, 190, 185, 186), (186, 187, 185, 186)]
    out = run({1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0))}, {"X": bars(rows, T0)}, trailing, paths=("batch", "paper"))
    got = closed(out)
    assert got["paper"] == got["batch"], got


def test_a_timer_hook_in_the_single_symbol_runner_cannot_read_the_bar_still_forming():
    """Engine only (run_strategy_backtest). Bars of 15 minutes, stamped at their close. A timer due
    5 minutes before bar 2's close fires ahead of bar 2. Its hook must see bars 0 and 1 only.
    Before: ctx._advance(2) ran first, so ctx.close[-1] was bar 2's close."""
    q = 15 * MIN
    close = np.array([100.0, 101.0, 150.0, 151.0])
    ts = T0 + q * np.arange(1, 5, dtype=np.int64)

    class Timer(Strategy):
        seen = None

        def on_start(self, ctx):
            self.clock.set_time_alert("t", int(ts[2]) - 5 * MIN)

        def on_timer(self, ctx, event):
            type(self).seen = float(ctx.close[-1])

    run_strategy_backtest(Timer(), ts, close, close + 1, close - 1, close, np.full(4, 1e6),
                          config=BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    assert Timer.seen == 101.0
