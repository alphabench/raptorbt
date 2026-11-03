"""One instrument runs on the portfolio book, so it follows the book's rules.

Four behaviours a single-instrument run shares with every other runner, pinned:
one equity sample per instant, a last equity point that is the booked balance,
a margin call that reports `halted`, and a physical settlement a one-instrument
run cannot deliver that is refused out loud.
"""

import numpy as np

import raptorbt as r
from raptorbt.strategy import Strategy, run_strategy_backtest

N = 40


def _bars(close):
    close = np.asarray(close, dtype=np.float64)
    return dict(
        timestamps=np.arange(len(close), dtype=np.int64) * 60 * 10**9 + 1_700_000_000_000_000_000,
        open=close.copy(), high=close + 0.5, low=close - 0.5, close=close,
        volume=np.full(len(close), 1_000.0),
    )


def _single(bars, entries, exits=None, config=None):
    n = len(bars["close"])
    return r.run_single_backtest(
        **bars,
        entries=np.asarray(entries, dtype=bool),
        exits=np.asarray(exits if exits is not None else np.zeros(n, bool), dtype=bool),
        direction=1, weight=1.0, symbol="X",
        config=config or r.BacktestConfig(initial_capital=100_000.0, fees=0.001),
    )


def test_bars_at_one_instant_are_one_observation():
    """Three bars stamped alike are one instant: one equity sample, not three.

    A sample per bar would add two zero returns at an instant that has one
    price, which inflates the period count the annualised metrics divide by.
    """
    bars = _bars(100.0 + np.arange(N, dtype=float))
    bars["timestamps"][10:13] = bars["timestamps"][10]
    entries = np.zeros(N, bool)
    entries[3] = True
    res = _single(bars, entries)
    assert len(res.equity_curve()) == N - 2
    assert len(res.returns()) == N - 2


def test_the_last_equity_point_is_the_booked_balance():
    """With a position open at the end, the curve ends on the cash after the
    closing fee, in booked minor units, not on the unbooked mark before it."""
    bars = _bars(100.0 + np.arange(N, dtype=float) * 0.37)
    entries = np.zeros(N, bool)
    entries[5] = True
    res = _single(bars, entries)
    last = float(res.equity_curve()[-1])
    assert last == round(last, 2), "booked to the cent"
    booked = 100_000.0 + sum(t.pnl for t in res.trades())
    assert abs(last - booked) < 0.011
    assert last == res.metrics.end_value


def test_a_margin_call_reports_halted():
    """A margin call latches a halt on the account; the result must say so."""

    class Long(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 2:
                self.enter()

    close = np.concatenate([np.full(10, 100.0), np.linspace(100.0, 80.0, 30)])
    res = run_strategy_backtest(
        Long(), **_bars(close), symbol="X",
        config=r.BacktestConfig(initial_capital=100_000.0, fees=0.0, liquidate_on_margin_call=True),
        account_type="margin", leverage=50.0, bar_stamp="close",
    )
    assert res.halted
    assert res.halted_at is not None


def test_a_physical_settlement_a_single_run_cannot_deliver_is_refused_in_the_events():
    """The underlying is not in a one-instrument run, so nothing can be
    delivered; the run says so rather than dropping the contract's end."""
    fut = r.InstrumentSpec.futures_contract(
        "FUT", expiration_ns=3, lot_size=1.0, multiplier=2.0, underlying="U"
    )
    fut.settlement_style = "physical"
    fut.settlement_price = 120.0
    session = r.KernelSession(
        symbol="FUT", direction=1,
        config=r.BacktestConfig(initial_capital=10_000.0, fees=0.0), instrument=fut,
    )
    refused = []
    for i in range(5):
        for e in session.step(i, i, 100.0, 101.0, 99.0, 100.0, 10.0, entry=(i == 0)):
            if e.kind == "lifecycle_refused":
                refused.append(e)
    assert len(refused) == 1
    assert refused[0].outcome == "settlement"
    assert refused[0].reject_reason == "physical_settlement_needs_a_portfolio"
