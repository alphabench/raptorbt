"""Edge cases where a windowed run could differ from the whole run, each built by hand.

A fill hook blind to its instant, an ``at("1w")`` over bars that start on a Friday, a series read
through code the strategy class does not name, a split and a dividend across windows, and an
option that expires inside or across windows. Each runs whole (``run_strategies``) and in windows
(``run_strategies_windowed``) and compares trades, curves and what the hooks saw; the expected
values are worked out by hand in the comments.
"""

from __future__ import annotations

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Indicator, Strategy, Window, run_strategies, run_strategies_windowed
from raptorbt.errors import RaptorBTError
from raptorbt.strategy import orders

MIN = 60 * 10**9
DAY_NS = 86_400 * 10**9
T0 = 1_767_589_800 * 10**9 + MIN  # 2026-01-05 10:01 IST, the close of the first 1-minute bar


def _bars(closes, low_at=None):
    """Close-stamped 1-minute bars: open = previous close (first: its own close),
    high = max(open, close) + 0.5, low = min(open, close) - 0.5, volume 1000;
    ``low_at`` = {k: low} overrides a low."""
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    lo = np.minimum(o, c) - 0.5
    for k, v in (low_at or {}).items():
        lo[k] = v
    return {
        "timestamps": T0 + np.arange(len(c), dtype=np.int64) * MIN,
        "open": o,
        "high": np.maximum(o, c) + 0.5,
        "low": lo,
        "close": c,
        "volume": np.full(len(c), 1000.0),
    }


def _windows(data, ends):
    """Windows over (previous end, end] of each end (event time = stamp: close-stamped)."""
    out, prev = [], None
    for end in ends:
        bars = {}
        for s, a in data.items():
            ts = np.asarray(a["timestamps"])
            lo = 0 if prev is None else int(np.searchsorted(ts, prev, side="right"))
            hi = int(np.searchsorted(ts, end, side="right"))
            bars[s] = {k: np.asarray(v)[lo:hi] for k, v in a.items()}
        out.append(Window(bars=bars, end_ns=int(end)))
        prev = end
    return out


def _every_bar(data):
    return sorted({int(t) for a in data.values() for t in a["timestamps"]})


def _key(result):
    r = result.result
    return (
        [(t.symbol, t.entry_idx, t.exit_idx, t.entry_price, t.exit_price, t.size, t.pnl, t.fees) for t in r.trades()],
        r.equity_curve().tolist(),
        len(r.orders()),
    )


# -- 1. what a fill hook reads of ctx.bar while it is blind to the instant --------------------------
#
# A: closes 100..107 at t0..t7; bar t5's low is 90.   B: closes 60..67, nothing special.
# The strategy trades A only: a market buy of 10 at t1's decision, a resting StopMarket sell 10
# @ 95 (reduce_only) placed at t2. Bar t5 (open 104, low 90) trades through 95: the stop fills at
# 95 in the rest phase, where the hook is blind to the t5 bars and must read A's t4 bar
# (close 104.0, stamp t4). Under next_bar_open the buy fills at t2's open (101) and its hook,
# blind to t2, must read A's t1 bar (close 101.0, stamp t1).
# The hook re-enters 5 units when ctx.bar.close > 80 on the stop fill, so a wrong bar changes
# the orders and trades, not only the record.


class Hooked(Strategy):
    def on_start(self, ctx):
        self.seen = []
        self.fills = 0

    def on_bar(self, ctx):
        if ctx.symbol != "A":
            return
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", units=10.0))
        elif ctx.idx == 2:
            self.submit_order(orders.StopMarket(side="sell", trigger=95.0, units=10.0, reduce_only=True))

    def on_fill(self, ctx, event):
        self.fills += 1
        b = ctx.bar
        self.seen.append((ctx.symbol, int(b.timestamp), float(b.close)))
        if self.fills == 2 and b.close > 80.0:
            self.submit_order(orders.Market(side="buy", units=5.0))


DATA1 = {"A": _bars([100 + k for k in range(8)], low_at={5: 90.0}), "B": _bars([60 + k for k in range(8)])}


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
@pytest.mark.parametrize("history", ["declared", "all"])
@pytest.mark.parametrize("ends", ["whole", "each-bar"])
def test_a_blind_fill_hook_reads_the_filling_instruments_previous_bar(timing, history, ends):
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0, fill_timing=timing)
    whole = Hooked()
    want = run_strategies([whole], DATA1, cfg, bar_stamp="close")[0]
    # hand: the stop fill's hook reads A's t4 bar
    stop_record = ("A", int(T0 + 4 * MIN), 104.0)
    assert stop_record in whole.seen, whole.seen
    if timing == "next_bar_open":
        assert whole.seen[0] == ("A", int(T0 + 1 * MIN), 101.0), whole.seen
    every = _every_bar(DATA1)
    cuts = [every[-1]] if ends == "whole" else every
    windowed = Hooked()
    got = run_strategies_windowed([windowed], _windows(DATA1, cuts), cfg, symbols=["A", "B"],
                                  bar_stamp="close", history=history)[0]
    assert windowed.seen == whole.seen
    assert _key(got) == _key(want)


# -- 2. at("1w") over daily bars whose first two straddle a weekend ---------------------------------
#
# Daily bars stamped at 10:00 UTC (15:30 IST close): Fri 2026-01-02, then Mon-Fri for three
# weeks. The stored bars' most common gap is 1 day, so the batch run accepts at("1w") (7d is a
# coarser multiple of 1d). A windowed run has no stored bars to read the width from, and
# the first two bars are 3 days apart: it takes the width the caller gives (bar_ns) and refuses
# the expression without it.


def _daily():
    first = 20_455  # 2026-01-02, a Friday, as days since the epoch
    days = [first] + [first + 3 + w * 7 + d for w in range(3) for d in range(5)]
    ts = np.array([d * DAY_NS + 10 * 3600 * 10**9 for d in days], dtype=np.int64)
    c = 100.0 + np.arange(len(ts), dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return {"W": {"timestamps": ts, "open": o, "high": np.maximum(o, c) + 1, "low": np.minimum(o, c) - 1,
                  "close": c, "volume": np.full(len(ts), 1e4)}}


class Weekly(Strategy):
    def on_start(self, ctx):
        self.h = ctx.indicators.expr('at("1w", close)', instrument="W")
        self.seen = []

    def on_bar(self, ctx):
        v = self.h.value
        self.seen.append(None if v is None else float(v))


def test_at_one_week_over_daily_bars_starting_on_a_friday():
    data = _daily()
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    whole = Weekly()
    run_strategies([whole], data, cfg, bar_stamp="close", bar_ns=DAY_NS)
    windowed = Weekly()
    run_strategies_windowed([windowed], _windows(data, _every_bar(data)), cfg, symbols=["W"],
                            bar_stamp="close", bar_ns=DAY_NS)
    assert windowed.seen == whole.seen
    with pytest.raises(RaptorBTError) as refused:
        run_strategies_windowed([Weekly()], _windows(data, _every_bar(data)), cfg, symbols=["W"], bar_stamp="close")
    assert refused.value.code == "expr_unsupported_here"


# -- 3. a series read through a helper the class source does not name -------------------------------
#
# The helper stands for a module the strategy imports. The batch run answers the read; a windowed run keeps
# every row of what the strategy declared nothing for, from the start, so it answers it too.


def _closes(c):
    return c.series()["close"]


class ViaHelper(Strategy):
    def on_start(self, ctx):
        self.sma = self.register_indicator(Indicator.sma(5), symbol=ctx.symbols[0])
        self.n = []

    def on_bar(self, ctx):
        if ctx.idx < 10:
            return
        self.n.append(len(_closes(ctx)))


def test_a_series_read_through_a_helper_runs_as_the_batch_run_does():
    data = {"A": _bars([100 + (k % 7) for k in range(40)])}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    whole = ViaHelper()
    run_strategies([whole], data, cfg, bar_stamp="close")
    assert whole.n[0] == 11 and whole.n[-1] == 40  # hand: rows 0..idx
    windowed = ViaHelper()
    run_strategies_windowed([windowed], _windows(data, _every_bar(data)[4::5]), cfg, symbols=["A"], bar_stamp="close")
    assert windowed.n == whole.n


# -- 4. corporate actions (raw) with the ex-date first in a window, the pay date in a later one -----
#
# A: 30 bars, closes 100..129; a 2:1 split effective at t10 and a dividend of 1.0 effective at t20
# paid at t25. Buy 10 units at t2, sell everything at t28. Windows end at t9 (the ex-date bar t10
# opens the next window), t19, t24, t29; and every bar.


class Holder(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 2:
            self.submit_order(orders.Market(side="buy", units=10.0))
        elif ctx.idx == 28:
            self.close_position()


@pytest.mark.parametrize("cut", ["edges", "each-bar"])
def test_split_and_dividend_across_windows_are_the_batch_run(cut):
    data = {"A": _bars([100.0 + k for k in range(30)])}
    t = lambda k: int(T0 + k * MIN)  # noqa: E731
    actions = [raptorbt.CorporateAction.split("A", t(10), 2.0),
               raptorbt.CorporateAction.dividend("A", t(20), 1.0, pay_ts=t(25))]
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    kw = dict(bar_stamp="close", corporate_actions=actions, adjustment_policy="raw")
    want = run_strategies([Holder()], data, cfg, **kw)[0]
    ends = [t(9), t(19), t(24), t(29)] if cut == "edges" else _every_bar(data)
    got = run_strategies_windowed([Holder()], _windows(data, ends), cfg, symbols=["A"], **kw)[0]
    assert want.result.trades()
    assert _key(got) == _key(want)


# -- 5. an option that expires inside a window, and one that expires on a window's edge ---------------
#
# One call option, lot 50, expiry at the close of bar 20 of 40, bought at bar 2 and held: the
# engine settles it at expiry, whichever window the expiry falls in.


class BuyAndHold(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 2:
            self.submit_order(orders.Market(side="buy", units=50.0))


@pytest.mark.parametrize("cut", ["expiry-inside", "expiry-on-edge", "each-bar"])
def test_an_option_expiring_inside_or_across_windows_settles_as_in_the_batch_run(cut):
    symbol, expiry = "OPT", int(T0 + 20 * MIN)
    spec = raptorbt.InstrumentSpec.option(symbol, 100.0, "call", expiry, 50.0, underlying="U",
                                          span_pct=0.1, exposure_pct=0.02)
    data = {symbol: _bars([5.0 + 0.1 * k for k in range(40)])}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    kw = dict(bar_stamp="close", instruments={symbol: spec})
    want = run_strategies([BuyAndHold()], data, cfg, **kw)[0]
    t = lambda k: int(T0 + k * MIN)  # noqa: E731
    ends = {"expiry-inside": [t(15), t(25), t(39)], "expiry-on-edge": [t(19), t(20), t(39)],
            "each-bar": _every_bar(data)}[cut]
    got = run_strategies_windowed([BuyAndHold()], _windows(data, ends), cfg, symbols=[symbol], **kw)[0]
    assert want.result.trades()
    assert _key(got) == _key(want)
