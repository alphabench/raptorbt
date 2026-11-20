"""``auto_roll`` when the old contract has no bar left to fill the close on.

The close of a roll fills at the next bar of the OLD contract. When that contract has none (its
data ends before its expiry, or ``roll_days=0`` puts the roll on the expiry instant itself) the
close can never fill: at the expiry the engine settles the position and the close ends ``canceled``
(or ``rejected`` for ``expired``). The position has left the old contract, so that is not a failed
roll: the open follows in the new contract for what was held, ``on_roll_settled`` says so, and the
close keeps its own status. A close refused before the expiry, with the position still held,
still fails the roll (``test_error_contract.py``: ``roll_close_failed``).
"""

import numpy as np

import raptorbt as rb
from raptorbt import BacktestConfig, SessionSchedule, Strategy, run_strategies
from raptorbt.strategy import Future, Holding, Requirements, Underlying
from raptorbt.strategy import orders as O
from raptorbt.strategy.continuous import RollEvents

from _market_harness import MIN, fmt, ist


def _carry(roll_days, sizing, units, settled):
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
            self.n, self.events = 0, []

        def on_bar(self, ctx, bar):
            if ctx.symbol == "IDX":
                self.n += 1
                if self.n == 1:
                    self.submit_order(O.Market(side="buy", quantity=units, instrument=ctx.role("fut")))

        def on_order_event(self, ctx, e):
            self.events.append((fmt(ctx.event_ns), e.kind, e.client_order_id, getattr(e, "reject_reason", None)))

        def on_roll_settled(self, ctx, roll, close_id, price):
            settled.append((roll.from_symbol, roll.to_symbol, close_id, price))

    return Carry()


def _bars(stamps, base, until=None):
    """One-minute bars on the close ``stamps``, none after ``until``."""
    ts = np.array([t for t in stamps if until is None or t <= until], dtype=np.int64)
    c = base + np.arange(len(ts)) * 0.01
    return {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(len(ts), 100.0)}


def _run(strategy, data, specs):
    return run_strategies([strategy], data, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                          instruments=specs, bindings={"index": "IDX"})[0]


def _orders(result):
    return {o.client_id: (o.symbol, o.status, o.filled_qty) for o in result.result.orders()}


def test_a_close_with_no_bar_left_in_the_old_contract_is_settled_and_the_roll_goes_on():
    t0 = ist(2026, 11, 2, 9, 15)
    stamps = [t0 + (i + 1) * MIN for i in range(300)]
    exp_a, exp_b = t0 + 200 * MIN, t0 + 2000 * MIN
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", exp_a, 15.0, underlying="IDX"),
        "FB": rb.InstrumentSpec.futures_contract("FB", exp_b, 15.0, underlying="IDX"),
    }
    # FA's data ends 100 minutes before its expiry, 40 before the roll's time: the close has no bar
    data = {"IDX": _bars(stamps, 100.0), "FA": _bars(stamps, 110.0, until=t0 + 100 * MIN), "FB": _bars(stamps, 120.0)}
    for sizing in ("lots", "notional"):
        settled = []
        s = _carry(60 / 1440, sizing, 30.0, settled)  # the roll is due 60 minutes before the expiry
        result = _run(s, data, specs)  # no ROLL_CLOSE_FAILED
        orders = _orders(result)
        assert orders["fut:roll0:c0"][:2] == ("FA", "canceled")
        assert orders["fut:roll0:o0"][:2] == ("FB", "filled")
        assert len(settled) == 1 and settled[0][:3] == ("FA", "FB", "fut:roll0:c0")
        assert settled[0][3] == data["FA"]["close"][-1]  # the contract's last price, which it settled at
        exits = [t for t in result.result.trades() if t.symbol == "FA"]
        assert len(exits) == 1 and exits[0].exit_reason == "settlement"


def test_a_roll_due_at_the_expiry_itself_is_settled_and_the_open_waits_for_the_market():
    day1, day2 = ist(2026, 11, 2, 9, 15), ist(2026, 11, 3, 9, 15)
    sess = SessionSchedule.nse_equity()
    stamps = [day1 + (i + 1) * MIN for i in range(376)] + [day2 + (i + 1) * MIN for i in range(376)]
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", ist(2026, 11, 2, 15, 30), 15.0, underlying="IDX", venue="NSE", session=sess),
        "FB": rb.InstrumentSpec.futures_contract("FB", ist(2026, 12, 1, 15, 30), 15.0, underlying="IDX", venue="NSE", session=sess),
    }
    data = {"IDX": _bars(stamps, 100.0), "FA": _bars(stamps, 110.0, until=ist(2026, 11, 2, 15, 30)), "FB": _bars(stamps, 120.0)}
    settled = []
    s = _carry(0, "lots", 30.0, settled)  # roll_days=0: the roll is decided on the expiry's last bar
    result = _run(s, data, specs)
    orders = _orders(result)
    assert orders["fut:roll0:c0"][1] in ("rejected", "canceled")
    assert orders["fut:roll0:o0"] == ("FB", "rejected", 0.0)  # the market is closed at the expiry instant ...
    assert orders["fut:roll0:o0.r1"] == ("FB", "filled", 30.0)  # ... so the next event sends it again, and it fills
    assert [ts for ts, kind, cid, _ in s.events if cid == "fut:roll0:o0.r1" and kind == "order_filled"][0].startswith("11-03")
    assert len(settled) == 1


def test_a_lot_that_changes_between_the_contracts_is_kept_in_lots_after_a_settlement():
    t0 = ist(2026, 11, 2, 9, 15)
    stamps = [t0 + (i + 1) * MIN for i in range(300)]
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", t0 + 200 * MIN, 15.0, underlying="IDX"),
        "FB": rb.InstrumentSpec.futures_contract("FB", t0 + 2000 * MIN, 30.0, underlying="IDX"),
    }
    data = {"IDX": _bars(stamps, 100.0), "FA": _bars(stamps, 110.0, until=t0 + 100 * MIN), "FB": _bars(stamps, 120.0)}
    settled = []
    result = _run(_carry(60 / 1440, "lots", 30.0, settled), data, specs)  # two lots of 15 ...
    opened = [o for o in result.result.orders() if o.client_id.startswith("fut:roll0:o")]
    assert [(o.symbol, o.filled_qty) for o in opened] == [("FB", 60.0)]  # ... are two lots of 30
