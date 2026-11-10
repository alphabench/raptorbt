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


def _bars(closes, low_at=None, stamps=None):
    """Close-stamped 1-minute bars: open = previous close (first: its own close),
    high = max(open, close) + 0.5, low = min(open, close) - 0.5, volume 1000;
    ``low_at`` = {k: low} overrides a low."""
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    lo = np.minimum(o, c) - 0.5
    for k, v in (low_at or {}).items():
        lo[k] = v
    return {
        "timestamps": T0 + (np.arange(len(c)) if stamps is None else np.asarray(stamps)).astype(np.int64) * MIN,
        "open": o,
        "high": np.maximum(o, c) + 0.5,
        "low": lo,
        "close": c,
        "volume": np.full(len(c), 1000.0),
    }


def _windows(data, ends, ticks=None):
    """Windows over (previous end, end] of each end (event time = stamp: close-stamped)."""
    out, prev = [], None
    for end in ends:
        def cut(arrays):
            ts = np.asarray(arrays["timestamps"])
            lo = 0 if prev is None else int(np.searchsorted(ts, prev, side="right"))
            hi = int(np.searchsorted(ts, end, side="right"))
            return {k: np.asarray(v)[lo:hi] for k, v in arrays.items()}

        out.append(Window(bars={s: cut(a) for s, a in (data or {}).items()},
                          ticks={s: cut(a) for s, a in ticks.items()} if ticks else None,
                          end_ns=int(end)))
        prev = end
    return out


def _every_bar(data):
    return sorted({int(t) for a in data.values() for t in a["timestamps"]})


_every = _every_bar


def _outcome(fn):
    """What a call gives, or the code it was refused with."""
    try:
        return ("ok", fn())
    except RaptorBTError as e:
        return ("refused", e.code)


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


# -- 6. a strategy that declared N rows and reads more through ctx.series: its own contract
#
# A: closes 100 at rows 0..14, 110 at rows 15..20, 120 at rows 21..29 (1-minute, close-stamped).
# The strategy declares keep_history(5) and, at row 20, buys 10 if close > mean(last 20 closes) + 3;
# it closes at row 25. same_bar_close.
# Batch at row 20: last 20 closes = rows 1..20 = 14 x 100 + 6 x 110 = 2060 -> mean 103.0;
#   110 > 106 -> buy 10 @ 110; exit at row 25 @ 120 -> one trade, pnl 10 x (120 - 110) = 100.0.
# Windowed at row 20: ctx.series shows exactly rows 16..20 = 5 x 110 -> mean 110.0; 110 > 113 is
#   false -> no trade, pnl 0. No error is raised.

DATA_OVER = {"A": _bars([100.0] * 15 + [110.0] * 6 + [120.0] * 9)}


class OverReader(Strategy):
    def on_start(self, ctx):
        ctx.keep_history(5, symbol="A")
        self.lens = []

    def on_bar(self, ctx):
        closes = ctx.series()["close"]
        self.lens.append(len(closes))
        if ctx.idx == 20 and closes[-1] > float(np.mean(closes[-20:])) + 3.0:
            self.submit_order(orders.Market(side="buy", units=10.0))
        elif ctx.idx == 25:
            self.close_position()


def test_reading_past_the_declared_rows_through_ctx_series_is_the_documented_contract():
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0, fill_timing="same_bar_close")
    whole = OverReader()
    want = run_strategies([whole], DATA_OVER, cfg, bar_stamp="close")[0]
    win = OverReader()
    got = run_strategies_windowed([win], _windows(DATA_OVER, sorted(set(_every(DATA_OVER)[4::5]) | {_every(DATA_OVER)[-1]})), cfg,
                                  symbols=["A"], bar_stamp="close")[0]
    wt = want.result.trades()
    assert len(wt) == 1 and wt[0].pnl == pytest.approx(100.0)  # hand
    assert win.lens[20] == 5 and whole.lens[20] == 21  # hand: exactly N vs rows 0..20
    # Documented: ctx.series shows exactly the declared rows and raises nothing, so code that
    # reads further back decides differently. ctx.data.history, which can say so, refuses.
    assert got.result.trades() == [] or _key(got) != _key(want)


class DataReader(Strategy):
    def on_start(self, ctx):
        ctx.keep_history(5, symbol="A")

    def on_bar(self, ctx):
        if ctx.idx == 8:
            ctx.data.history("A", count=20)


def test_ctx_data_history_past_the_declared_rows_is_refused():
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    run_strategies([DataReader()], DATA_OVER, cfg, bar_stamp="close")  # the batch run answers
    with pytest.raises(RaptorBTError) as refused:
        run_strategies_windowed([DataReader()], _windows(DATA_OVER, _every(DATA_OVER)), cfg, symbols=["A"],
                                bar_stamp="close")
    assert (refused.value.code, refused.value.parameter) == ("invalid_history", "count")


# -- 7. history="all" and a keep_history made after the first window ------------------------------
#
# The docstring: "all": every row of every symbol, for every strategy, whatever it declared.
# The strategy calls keep_history(2) at row 3 (in on_bar, not on_start) and records len(series).
# Expected per the docstring: len == idx + 1 at every row, as in the batch run.

class LateKeeper(Strategy):
    def on_start(self, ctx):
        self.lens = []

    def on_bar(self, ctx):
        if ctx.idx == 3:
            ctx.keep_history(2)
        self.lens.append(len(ctx.series()["close"]))


@pytest.mark.parametrize("history", ["all", "declared"])
def test_a_late_keep_history_under_history_all(history):
    data = {"A": _bars([100.0 + k for k in range(12)])}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    whole = LateKeeper()
    run_strategies([whole], data, cfg, bar_stamp="close")
    assert whole.lens == list(range(1, 13))  # hand
    win = LateKeeper()
    run_strategies_windowed([win], _windows(data, _every(data)), cfg, symbols=["A"], bar_stamp="close",
                            history=history)
    if history == "all":
        assert win.lens == whole.lens, win.lens  # the docstring's promise
    else:
        # "declared": record what happens (exact 2 from row 3 is the documented reading)
        assert win.lens[:3] == [1, 2, 3] and win.lens[3:] == [2] * 9, win.lens


# -- 8. at(...) on the primary bars a tick run aggregates (primary_bars) --------------------------
#
# X: a print every 10 s from 10:00:10 IST for 30 min, ltp 100 + 0.01 k. primary_bars=(1, "m").
# The batch run has no stored bars and infers the width; the windowed run requires a width and
# reads bar_ns only for bar symbols. Parity: both accept with the same values, or both refuse with
# the same code.

def _ticks():
    n = 180
    ts = (T0 - MIN + (np.arange(n, dtype=np.int64) + 1) * 10 * 10**9).astype(np.int64)
    ltp = 100.0 + 0.01 * np.arange(n)
    return {"X": {"timestamps": ts, "ltp": ltp, "bid": ltp - 0.05, "ask": ltp + 0.05}}


class AtFive(Strategy):
    def on_start(self, ctx):
        self.h = ctx.indicators.expr('at("5m", close)', instrument="X")
        self.seen = []

    def on_bar(self, ctx):
        v = self.h.value
        self.seen.append(None if v is None else float(v))


@pytest.mark.parametrize("bar_ns", [None, MIN])
def test_at_on_tick_primary_bars_is_the_batch_run(bar_ns):
    ticks = _ticks()
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)

    def batch():
        s = AtFive()
        run_strategies([s], None, cfg, ticks=ticks, primary_bars=(1, "m"))
        return s.seen

    def windowed():
        s = AtFive()
        ts = ticks["X"]["timestamps"]
        ends = [int(ts[59]), int(ts[119]), int(ts[-1])]
        kw = {} if bar_ns is None else {"bar_ns": bar_ns}
        run_strategies_windowed([s], _windows(None, ends, ticks=ticks), cfg, symbols=["X"],
                                primary_bars=(1, "m"), **kw)
        return s.seen

    b, w = _outcome(batch), _outcome(windowed)
    assert b == w and b[0] == "ok", (b, w)


# -- 9. the width at(...) runs on: bar_ns where it is given, in both drivers ----
#
# Sparse 1-minute close-stamped bars (an illiquid symbol): stamps at minutes 0, 1, 2, then every
# 2 minutes to 58. Modal gap = 2 min (28 gaps of 2 vs 2 of 1); the true width, given as bar_ns, is
# 1 min. Parity: same values, or the same refusal.

def test_at_width_on_sparse_bars_is_the_batch_runs():
    stamps = [0, 1, 2] + list(range(4, 60, 2))
    data = {"X": _bars([100.0 + 0.1 * k for k in range(len(stamps))], stamps=stamps)}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)

    def batch():
        s = AtFive()
        run_strategies([s], data, cfg, bar_stamp="close", bar_ns=MIN)
        return s.seen

    def windowed():
        s = AtFive()
        run_strategies_windowed([s], _windows(data, sorted(set(_every(data)[::7]) | {_every(data)[-1]})), cfg,
                                symbols=["X"], bar_stamp="close", bar_ns=MIN)
        return s.seen

    b, w = _outcome(batch), _outcome(windowed)
    # both use bar_ns (1 minute) where it is given, not the stored bars' 2-minute gap
    assert b == w and b[0] == "ok", (b, w)


# -- 10. an option that expires in a gap with no bar of its own, cut at the gap, and after the last bar
#
# OPT (call, strike 100, lot 50, underlying U): bars at minutes 0..20, closes 5.0 + 0.1 k.
# U: bars at minutes 0..39, closes 105.0. Expiry at T0 + 22 min (no bar of either at 22? U has one:
# so a second variant has U stop at 20 too). Buy 50 OPT at row 2; record OPT's position size at each
# U bar. Windows end at 20, 21, 22, 30, 39 and at every bar.

class OptHolder(Strategy):
    def on_start(self, ctx):
        self.sizes = []

    def on_bar(self, ctx):
        if ctx.symbol == "OPT" and ctx.idx == 2:
            self.submit_order(orders.Market(side="buy", units=50.0, instrument="OPT"))
        if ctx.symbol == "U":
            p = ctx.position_for("OPT")
            self.sizes.append((ctx.idx, None if p is None else float(p.size)))


@pytest.mark.parametrize("u_last", [39, 20])
@pytest.mark.parametrize("cut", ["gap-edges", "each-bar", "whole"])
def test_expiry_in_a_gap_and_after_the_last_bar(u_last, cut):
    expiry = int(T0 + 22 * MIN)
    spec = raptorbt.InstrumentSpec.option("OPT", 100.0, "call", expiry, 50.0, underlying="U",
                                          span_pct=0.1, exposure_pct=0.02)
    data = {"OPT": _bars([5.0 + 0.1 * k for k in range(21)]),
            "U": _bars([105.0] * (u_last + 1))}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    kw = dict(bar_stamp="close", instruments={"OPT": spec})
    whole = OptHolder()
    want = run_strategies([whole], data, cfg, **kw)[0]
    t = lambda k: int(T0 + k * MIN)  # noqa: E731
    last = _every(data)[-1]
    ends = {"gap-edges": sorted({t(20), t(21), t(22), t(30), last} & set(_every(data))) or [last],
            "each-bar": _every(data), "whole": [last]}[cut]
    if ends[-1] != last:
        ends.append(last)
    win = OptHolder()
    got = run_strategies_windowed([win], _windows(data, ends), cfg, symbols=["OPT", "U"], **kw)[0]
    assert win.sizes == whole.sizes
    assert _key(got) == _key(want)


# -- 11. the blind fill hook with the smallest declarations: keep_history(1) and history=1 -----------
#
# A: closes 100..107, row 5's low 90; B: 60..67. Buy 10 A at row 1; StopMarket sell 10 @ 95 placed at
# row 2. Row 5 (open 104, low 90) trades through 95. The stop's hook is blind to row 5 and reads A's
# row 4 (stamp t4, close 104.0); under next_bar_open the buy's hook (fill at row 2's open) reads A's
# row 1 (close 101.0). The hook records ctx.bar and the last row ctx.series("A") shows.

def _low_at(d, k, v):
    d["low"] = d["low"].copy()
    d["low"][k] = v
    return d


DATA_BLIND = {"A": _low_at(_bars([100.0 + k for k in range(8)]), 5, 90.0), "B": _bars([60.0 + k for k in range(8)])}


class BlindReader(Strategy):
    declare = False

    def on_start(self, ctx):
        self.seen = []
        if self.declare:
            for s in ctx.symbols:
                ctx.keep_history(1, symbol=s)

    def on_bar(self, ctx):
        if ctx.symbol != "A":
            return
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", units=10.0))
        elif ctx.idx == 2:
            self.submit_order(orders.StopMarket(side="sell", trigger=95.0, units=10.0, reduce_only=True))

    def on_fill(self, ctx, event):
        b = ctx.bar
        s = ctx.series("A")
        self.seen.append((ctx.symbol, int(b.timestamp), float(b.close),
                          int(s["timestamps"][-1]), float(s["close"][-1])))


class BlindReaderDeclared(BlindReader):
    declare = True


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
@pytest.mark.parametrize("mode", ["keep1", "history1"])
def test_blind_hook_with_one_row_declared(timing, mode):
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0, fill_timing=timing)
    cls = BlindReaderDeclared if mode == "keep1" else BlindReader
    whole = cls()
    want = run_strategies([whole], DATA_BLIND, cfg, bar_stamp="close")[0]
    assert ("A", int(T0 + 4 * MIN), 104.0, int(T0 + 4 * MIN), 104.0) in whole.seen, whole.seen  # hand
    win = cls()
    got = run_strategies_windowed([win], _windows(DATA_BLIND, _every(DATA_BLIND)), cfg, symbols=["A", "B"], bar_stamp="close",
                                  history=1 if mode == "history1" else "declared")[0]
    assert win.seen == whole.seen
    assert _key(got) == _key(want)


# -- 12. what a declaration is for ---------------------------------------------------------------------


class QuoteSubscriber(Strategy):
    """Subscribes to quotes with a history of 3: that is for ticks, not for the bar series."""

    def on_start(self, ctx):
        self.subscribe("A", data_type="quote", history=3)
        self.lens = []

    def on_bar(self, ctx):
        self.lens.append(len(ctx.series()["close"]))


def test_a_quote_subscription_history_does_not_cap_the_bar_series():
    data = {"A": _bars([100.0 + k for k in range(12)])}
    ticks = {"A": {"timestamps": data["A"]["timestamps"].copy(), "ltp": data["A"]["close"].copy(),
                   "bid": data["A"]["close"] - 0.05, "ask": data["A"]["close"] + 0.05}}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    whole = QuoteSubscriber()
    run_strategies([whole], data, cfg, ticks=ticks, bar_stamp="close")
    win = QuoteSubscriber()
    windows = _windows(data, _every(data), ticks=ticks)
    run_strategies_windowed([win], windows, cfg, symbols=["A"], bar_stamp="close")
    assert whole.lens == list(range(1, 13)) and win.lens == whole.lens


class LateDeclarer(Strategy):
    def on_start(self, ctx):
        self.refused = None

    def on_bar(self, ctx):
        if ctx.idx == 6:
            try:
                ctx.keep_history(5)
            except RaptorBTError as e:
                self.refused = e.code
        ctx.series()


def test_a_keep_history_that_asks_for_released_rows_after_the_first_window_is_refused():
    data = {"A": _bars([100.0 + k for k in range(12)])}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    ok = LateDeclarer()
    run_strategies_windowed([ok], _windows(data, _every(data)), cfg, symbols=["A"], bar_stamp="close")
    assert ok.refused is None, "undeclared symbols keep every row, so there is nothing to refuse"

    class Released(LateDeclarer):
        def on_start(self, ctx):
            super().on_start(ctx)
            ctx.keep_history(2, symbol="A")  # holds two rows: row 6 cannot ask for five

    gone = Released()
    run_strategies_windowed([gone], _windows(data, _every(data)), cfg, symbols=["A"], bar_stamp="close")
    assert gone.refused == "window_state"


class Idle(Strategy):
    pass


def test_each_batch_holds_its_own_history_and_the_report_shows_it():
    data = {"A": _bars([100.0 + k for k in range(30)])}
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    windows = _windows(data, _every(data)[::5] + [_every(data)[-1]])
    kw = dict(symbols=["A"], bar_stamp="close", memory_report=True)
    _, one = run_strategies_windowed([Idle(), Idle()], windows, cfg, **kw)
    _, two = run_strategies_windowed([Idle(), Idle()], windows, cfg, batch_size=1, **kw)
    assert two["batches"] == 2 and two["live_rings"] > one["live_rings"]
