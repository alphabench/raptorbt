"""A "1w" bar is a trading week, Monday to Sunday, not seven days from the epoch.

The epoch (1970-01-01) was a Thursday, so weekly bars used to run Thursday to Wednesday.
"""
import numpy as np

import raptorbt as rb
from raptorbt import BacktestConfig, Strategy

DAY = 86_400 * 10**9
IST = 19_800 * 10**9
#: 2026-10-28, a Wednesday, 00:00 UTC (20,754 days after the epoch)
WED = 20_754 * DAY


class Weeks(Strategy):
    def __init__(self):
        super().__init__()
        self.weeks = []

    def on_start(self, ctx):
        self.subscribe_bars(1, "w")

    def on_bar(self, ctx, event):
        if getattr(event, "timeframe", None):
            self.weeks.append((event.timestamp, event.close))


def test_weekly_bars_end_on_monday_midnight():
    n = 20  # Wed 28 Oct .. Mon 16 Nov, one bar a day, close = 100 + day number
    close = 100.0 + np.arange(n, dtype=np.float64)
    data = {"X": {
        "timestamps": WED + DAY * np.arange(n, dtype=np.int64),
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
        "volume": np.full(n, 1_000.0),
    }}
    s = Weeks()
    cfg = BacktestConfig(initial_capital=1e6)
    cfg.fees = 0.0
    rb.run_strategies([s], data, cfg, bar_stamp="open", bar_ns=DAY)
    # Wed 28 Oct to Sun 1 Nov, Mon 2 to Sun 8 Nov, Mon 9 to Sun 15 Nov: each completes when the
    # next Monday's bar arrives; the closes are the three Sundays'
    assert [c for _, c in s.weeks] == [104.0, 111.0, 118.0]
    # each bar is stamped at the Monday that completes it, 00:00 UTC (05:30 IST); the epoch was a
    # Thursday, so a Monday is day 4 of the seven-day cycle counted from it
    assert [t for t, _ in s.weeks] == [WED + 5 * DAY, WED + 12 * DAY, WED + 19 * DAY]
    assert all(((t + IST) // DAY) % 7 == 4 for t, _ in s.weeks)
