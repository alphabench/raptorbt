"""An entry queued in the hook that hears an exit fill at a bar's open takes its ATR
stop from the bar before, not from the bar that is opening.

Bar 5 has a huge range; the entry fills at bar 5's open, so a stop that used bar 5's ATR would sit
far below the one built on bars up to 4.
"""
import numpy as np

import raptorbt as rb
from raptorbt import BacktestConfig, Strategy

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
N = 9


def _data():
    close = np.full(N, 100.0)
    high, low = close + 1.0, close - 1.0
    high[5], low[5] = 300.0, 20.0  # the bar that opens when the entry fills
    return {"X": dict(timestamps=T0 + np.arange(1, N + 1, dtype=np.int64) * MIN,
                      open=close.copy(), high=high, low=low, close=close, volume=np.full(N, 1e3))}


class Reenter(Strategy):
    def __init__(self):
        super().__init__()
        self.stops, self.exits = [], 0

    def on_bar(self, ctx):
        if ctx.idx == 1 and ctx.position is None:
            self.enter(size_frac=0.1)
        if ctx.idx == 4 and ctx.position is not None:
            self.close_position()

    def on_position_event(self, ctx, e):
        if e.kind == "entered":
            self.stops.append(ctx.position.stop_price)
        elif e.kind == "exited" and self.exits == 0:
            self.exits += 1
            self.enter(size_frac=0.1)  # queued in the hook that hears the exit, at bar 5's open


def test_an_entry_queued_in_an_open_hook_takes_the_atr_of_the_bar_before():
    data = _data()
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")
    cfg.set_atr_stop(2.0, 5)
    s = Reenter()
    rb.run_strategies([s], data, cfg, bar_stamp="close")
    d = data["X"]
    atr = rb.atr(d["high"], d["low"], d["close"], 5)
    assert len(s.stops) == 2 and s.stops[1] is not None, s.stops
    entry = d["open"][5]
    assert abs(s.stops[1] - (entry - 2.0 * atr[4])) < 1e-6, (s.stops, atr[4], atr[5])
    assert atr[5] > atr[4] * 3, "bar 5 must be able to tell the two apart"
