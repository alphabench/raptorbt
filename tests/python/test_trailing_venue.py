"""A standalone trailing stop on a venue trails what a batch run's does.

A backtest's trailing stop stands a fraction of the highest price since entry below that price. A
stop sent to a venue as a fixed price distance (the entry price times the fraction) stood closer as
the price rose, so it fired where a backtest's held.
"""
from test_venue_parity import PATHS, T0, bars, run

from raptorbt import BacktestConfig
from raptorbt.strategy import orders as O


def _sizes(out):
    return {p: sorted((round(t.size, 6), round(t.exit_price, 6)) for t in r.result.trades()) for p, r in out.items()}


def test_a_standalone_trailing_stop_trails_a_percent_of_the_high_on_every_path():
    # long at 100 with a 10% trail; the market runs to 200 and sits at 185: the stop stands at 180
    rows = [(100, 101, 99, 100)] * 2 + [(195, 200, 190, 200), (190, 190, 185, 186), (186, 187, 185, 186)]

    def cfg():
        c = BacktestConfig(initial_capital=1e6, fees=0.0)
        c.set_trailing_stop(0.10)
        return c

    out = run({1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0))},
              {"X": bars(rows, T0)}, cfg, paths=PATHS)
    got = _sizes(out)
    assert got["node"] == got["batch"], got
    assert got["paper"] == got["batch"], got
