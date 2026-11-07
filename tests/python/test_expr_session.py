"""The session functions: `session_bar()` and what is built on it.

They read the session the schedule cuts, so they need the same clock inputs as `at` and a
`SessionSchedule`. The value at a bar is made of that bar and earlier bars of its session; batch
evaluation, ``ctx.indicators.expr`` in a backtest and on a ``LiveNode`` give the same numbers.
The kernels are checked by hand in ``fns::session::tests`` (Rust); this file checks the binding.
"""

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, InstrumentSpec, LiveNode, SessionSchedule, Strategy, run_strategies
from raptorbt.analysis import Session
from raptorbt.errors import ExpressionError, RaptorBTError

MIN = 60 * 10**9
HOUR = 60 * MIN
DAY = 24 * HOUR
MON = 19_730  # Monday 2024-01-08
OHLCV = ("open", "high", "low", "close", "volume")
NSE = SessionSchedule.nse_equity()

TEXTS = {
    "bar": "session_bar()",
    "open": "session_open()",
    "range": "session_high() - session_low()",
    "volume": "session_volume()",
    "vwap": "session_vwap()",
    "ret": "session_return()",
    "hour": "session_first(session_return(), 4)",
    "gap": "session_gap()",
    "flag": "close > session_vwap() and session_bar() >= 2",
}


def week(days=5, per_day=25):
    """NSE weekdays of `per_day` fifteen-minute bars from 09:15 IST, each stamped at its close."""
    ts = []
    for d in range(days):
        ts.extend(
            (MON + d) * DAY + (225 + 15 * (np.arange(per_day) + 1)) * MIN
        )
    n = len(ts)
    k = np.arange(n)
    close = 100.0 + np.sin(k / 7.0) * 3 + (k // per_day)
    open_ = np.concatenate([[close[0]], close[:-1]]) + (k % per_day == 0) * 0.5
    return {
        "timestamps": np.array(ts, dtype=np.int64),
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
        "volume": 100.0 + (k % 7),
    }


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[~np.isnan(a)].view(np.int64), b[~np.isnan(b)].view(np.int64))


def hand_session_index(d):
    """The index of each bar within its date, by hand."""
    out, last, n = [], None, -1
    for t in d["timestamps"]:
        day = (int(t) - 1) // DAY
        n = 0 if day != last else n + 1
        last = day
        out.append(float(n))
    return np.array(out)


def test_the_index_the_open_and_the_running_extremes_by_hand():
    d = week(3)
    bar = rb.expr("session_bar()").eval(d, bar_stamp="close", session=NSE)
    same(bar, hand_session_index(d))
    o = rb.expr("session_open()").eval(d, bar_stamp="close", session=NSE)
    day_open = np.repeat(d["open"][::25], 25)
    same(o, day_open)
    hi = rb.expr("session_high()").eval(d, bar_stamp="close", session=NSE)
    want = np.concatenate([np.maximum.accumulate(d["high"][i : i + 25]) for i in range(0, 75, 25)])
    same(hi, want)
    vol = rb.expr("session_volume()").eval(d, bar_stamp="close", session=NSE)
    same(vol, np.concatenate([np.cumsum(d["volume"][i : i + 25]) for i in range(0, 75, 25)]))


def test_the_first_hour_return_is_known_from_the_fourth_bar_and_never_before():
    d = week(3)
    got = rb.expr("session_first(session_return(), 4)").eval(d, bar_stamp="close", session=NSE)
    for day in range(3):
        a = day * 25
        first = d["close"][a + 3] / d["open"][a] - 1.0
        assert np.isnan(got[a : a + 3]).all()
        assert (got[a + 3 : a + 25] == first).all()


@pytest.mark.parametrize("mode", ["batch", "node"])
def test_a_handle_in_a_backtest_and_on_a_node_equals_eval(mode):
    d = week()
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=NSE)}

    class Watch(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_start(self, ctx):
            self.h = {k: ctx.indicators.expr(v, instrument="ASSET") for k, v in TEXTS.items()}

        def on_bar(self, ctx):
            self.rows.append({k: h.value for k, h in self.h.items()})

    s = Watch()
    cfg = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
    if mode == "batch":
        run_strategies([s], {"ASSET": d}, cfg, bar_stamp="close", instruments=spec)
    else:
        node = LiveNode([s], ["ASSET"], cfg, bar_stamp="close", instruments=spec)
        for i in range(len(d["close"])):
            node.push_group(int(d["timestamps"][i]), {"ASSET": tuple(float(d[c][i]) for c in OHLCV)})
        node.finish()
    assert len(s.rows) == len(d["close"])
    for key, text in TEXTS.items():
        want = rb.expr(text).eval(d, bar_stamp="close", session=NSE)
        got = np.array([np.nan if r[key] is None else float(r[key]) for r in s.rows])
        same(got, want)
        assert (~np.isnan(want)).sum() > 50, key


def test_a_muhurat_session_and_a_holiday_follow_the_schedule():
    muhurat = 19_673  # Sunday 2023-11-12
    sched = NSE.with_session_override(20231112, [(18 * 60 + 15, 19 * 60 + 15)])
    ts = np.array([muhurat * DAY + 12 * HOUR + 45 * MIN + k * 15 * MIN for k in range(1, 6)], dtype=np.int64)
    c = 200.0 + np.arange(5)
    d = {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.ones(5)}
    same(rb.expr("session_bar()").eval(d, bar_stamp="close", session=sched), [0, 1, 2, 3, np.nan])
    same(rb.expr("session_bar()").eval(d, bar_stamp="close", session=NSE), [np.nan] * 5)
    # Tuesday 2024-01-09 a holiday: its bars are no session, and Wednesday's gap is over Monday's close.
    h = SessionSchedule.nse_equity().with_holidays([20240109])
    stamps = [(MON + k) * DAY + (225 + 15 * j) * MIN for k in range(3) for j in (1, 2)]
    c = np.array([100.5, 101.5, 90.5, 91.5, 110.5, 111.5])
    o = np.array([100.0, 101.0, 90.0, 91.0, 110.0, 111.0])
    d = {"timestamps": np.array(stamps, dtype=np.int64), "open": o, "high": c + 1, "low": c - 1, "close": c, "volume": np.ones(6)}
    same(rb.expr("session_bar()").eval(d, bar_stamp="close", session=h), [0, 1, np.nan, np.nan, 0, 1])
    gap = rb.expr("session_gap()").eval(d, bar_stamp="close", session=h)
    assert np.isnan(gap[:4]).all() and (gap[4:] == 110.0 / 101.5 - 1.0).all()


def test_a_plan_with_a_session_function_is_refused_what_it_lacks():
    d = week(2)
    e = rb.expr("session_open()")
    assert e.uses_at
    with pytest.raises(RaptorBTError) as err:
        e.eval({k: v for k, v in d.items() if k != "timestamps"}, bar_stamp="close", session=NSE)
    assert err.value.code == "missing_field" and err.value.parameter == "timestamps"
    with pytest.raises(RaptorBTError) as err:
        e.eval(d, session=NSE)
    assert err.value.code == "missing_bar_stamp"
    with pytest.raises(RaptorBTError) as err:
        e.eval(d, bar_stamp="close")
    assert err.value.code == "missing_field" and err.value.parameter == "session"

    class Plain(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr("session_open()", instrument="ASSET")

        def on_bar(self, ctx):
            pass

    with pytest.raises(RaptorBTError) as err:
        run_strategies([Plain()], {"ASSET": d}, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    assert err.value.code == "missing_field" and err.value.parameter == "session"


def test_the_family_is_not_defined_inside_at_and_at_may_sit_in_its_arguments():
    with pytest.raises(ExpressionError) as err:
        rb.expr('at("1d", session_open())')
    assert err.value.code == "expr_unsupported_here"
    d = week(4)
    daily = rb.expr('session_first(at("1d", close), 1)').eval(d, bar_stamp="close", session=NSE)
    assert not np.isnan(daily).all()


def test_bars_on_dates_a_schedule_does_not_cover_raise_session_not_covered():
    d = week(2)  # January 2024; the MCX preset covers from 2024-11-04
    mcx = SessionSchedule.mcx()
    with pytest.raises(RaptorBTError) as err:
        rb.expr("session_open()").eval(d, bar_stamp="close", session=mcx)
    assert err.value.code == "session_not_covered"
    s = Session()
    s.load(d, "close", name="w", session="mcx")
    with pytest.raises(RaptorBTError) as err:
        s.eval("session_return()", data="w")
    assert err.value.code == "session_not_covered"
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=mcx)}

    class Watch(Strategy):
        def on_start(self, ctx):
            self.h = ctx.indicators.expr("session_open()", instrument="ASSET")

        def on_bar(self, ctx):
            pass

    with pytest.raises(RaptorBTError) as err:
        run_strategies([Watch()], {"ASSET": d}, BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close", instruments=spec)
    assert err.value.code == "session_not_covered"


def test_research_takes_the_schedule_and_the_state_estimate_is_small():
    d = week(3)
    s = Session()
    s.load(d, "close", name="w", session="nse_equity")
    r = s.eval("session_first(session_return(), 4)", data="w")
    same(s.series(r["handle"]), rb.expr("session_first(session_return(), 4)").eval(d, bar_stamp="close", session=NSE))
    assert rb.expr("session_first(close, 100000)").state_bytes(400, 1, 1) < 1 << 20
