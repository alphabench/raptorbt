"""``auto_roll`` steps that cannot complete.

* A notional-sized roll into a dearer contract whose notional is below one lot of it: the close
  has filled and no open follows. It is a failed roll (``on_roll_failed``, raising
  ``roll_open_failed`` by default), not a ``KeyError`` from the roll's own bookkeeping.
* A roll due on the bar stamped at the session's close: the market takes no order at that instant,
  so the close is sent again at the next event instead of failing the roll with ``market_closed``.
"""

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, SessionSchedule, Strategy, run_strategies
from raptorbt.errors import StateError
from raptorbt.strategy import Future, Holding, Requirements, Underlying
from raptorbt.strategy import orders as O
from raptorbt.strategy.continuous import RollEvents

from _market_harness import MIN, fmt, ist


def _carry(roll_days, sizing, hook=None):
    class Carry(RollEvents, Strategy):
        requirements = Requirements(
            roles={
                "index": Underlying(asset="index"),
                "fut": Future(of="index", contract="continuous", roll_days=roll_days),
            },
            timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
        )
        auto_roll = True
        roll_sizing = sizing

        def __init__(self):
            super().__init__()
            self.n, self.events, self.failed = 0, [], []

        def on_bar(self, ctx, bar):
            if ctx.symbol == "IDX":
                self.n += 1
                if self.n == 1:
                    self.submit_order(O.Market(side="buy", units=1.0, instrument=ctx.role("fut")))

        def on_order_event(self, ctx, e):
            self.events.append((fmt(ctx.event_ns), e.kind, e.client_order_id, getattr(e, "reject_reason", None)))

    if hook:
        Carry.on_roll_failed = hook
    return Carry()


def _data(starts, per_day, price):
    ts = np.concatenate([s + (np.arange(per_day, dtype=np.int64) + 1) * MIN for s in starts])
    out = {}
    for sym, base in price.items():
        c = base + np.arange(len(ts)) * 0.01
        out[sym] = {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c,
                    "volume": np.full(len(ts), 100.0)}
    return out


def _run(strategy, data, specs):
    return run_strategies([strategy], data, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                          instruments=specs, bindings={"index": "IDX"})[0]


def test_a_zero_size_roll_fails_the_open_not_the_bookkeeping():
    t0 = ist(2026, 11, 2, 9, 15)
    exp_a, exp_b = t0 + 20 * MIN, t0 + 400 * MIN
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", exp_a, 1.0, underlying="IDX"),
        "FB": rb.InstrumentSpec.futures_contract("FB", exp_b, 1.0, underlying="IDX"),
    }
    # one unit of a 110 contract does not buy one unit of a 510 one
    data = _data([t0], 30, {"IDX": 100.0, "FA": 110.0, "FB": 510.0})
    with pytest.raises(StateError) as e:
        _run(_carry(5 / 1440, "notional"), data, specs)
    assert e.value.code == "roll_open_failed" and "zero_size" in str(e.value)

    seen = []
    s = _carry(5 / 1440, "notional", lambda self, ctx, roll, close_id, status, reason: seen.append((close_id, status)))
    _run(s, data, specs)
    assert seen == [("fut:roll0:o0", "zero_size")]


def test_a_roll_due_at_the_session_close_waits_for_the_next_session():
    day1, day2 = ist(2026, 11, 2, 9, 15), ist(2026, 11, 3, 9, 15)
    sess = SessionSchedule.nse_equity()
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", ist(2026, 11, 3, 15, 30), 1.0, underlying="IDX", venue="NSE", session=sess),
        "FB": rb.InstrumentSpec.futures_contract("FB", ist(2026, 12, 1, 15, 30), 1.0, underlying="IDX", venue="NSE", session=sess),
    }
    # the rule's time is 11-02 15:30, the stamp of day 1's last session bar: an action then is closed
    data = _data([day1, day2], 376, {"IDX": 100.0, "FA": 110.0, "FB": 120.0})
    s = _carry(1, "lots")
    _run(s, data, specs)
    roll = [(ts, kind, cid, why) for ts, kind, cid, why in s.events if cid.startswith("fut:roll0")]
    assert roll[0] == ("11-02 15:30", "order_rejected", "fut:roll0:c0", "market_closed")
    filled = [(ts, cid.partition(".r")[0]) for ts, kind, cid, _ in roll if kind == "order_filled"]
    assert filled == [("11-03 09:16", "fut:roll0:c0"), ("11-03 09:16", "fut:roll0:o0")]
