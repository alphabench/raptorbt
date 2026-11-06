"""``at("1d", ...)``: an expression evaluated on a coarser clock and held on the base one.

A daily value first appears on the first base bar of the *next* day, is held all that day, and is
the same in a batch evaluation, a stream, ``ctx.indicators.expr`` in ``run_strategies`` and on a
``LiveNode``. It is cut by the runtime's own aggregation, so the daily bar of
``ctx.indicators.sma(..., timeframe="1d")`` and ``at("1d", sma(close, n))`` are one sequence. No
number at bar i depends on a bar after i.
"""

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, Indicator, InstrumentSpec, LiveNode, SessionSchedule, Strategy, run_strategies
from raptorbt import journal as rj
from raptorbt.analysis import Session
from raptorbt.errors import RaptorBTError
from raptorbt.strategy import orders as O

HOUR = 3600 * 10**9
DAY = 24 * HOUR
IST = 19_800 * 10**9
OHLCV = ("open", "high", "low", "close", "volume")
FIRST_DAY = 20_000  # days since the epoch


# -- data ---------------------------------------------------------------------------


def hourly(days=12, *, stamp="close", tz=0, skip=(), seed=3):
    """Hourly bars, 24 a day, of the civil days (in the zone `tz`) FIRST_DAY.., minus `skip`.

    Close-stamped: the hour-24 bar of a day is stamped at the next midnight, exactly on the
    window boundary, and still belongs to the day it ends.
    """
    rng = np.random.default_rng(seed)
    ts = []
    for d in range(days):
        if d in skip:
            continue
        for k in range(24):
            ts.append((FIRST_DAY + d) * DAY - tz + k * HOUR + (HOUR if stamp == "close" else 0))
    n = len(ts)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return {
        "timestamps": np.array(ts, dtype=np.int64),
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
        "volume": np.round(rng.uniform(500.0, 1500.0, n)),
    }


def cut(data, lo, hi):
    return {k: v[lo:hi].copy() for k, v in data.items()}


def same_bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[~np.isnan(a)].view(np.int64), b[~np.isnan(b)].view(np.int64))


def hand_daily(ts, close, fn, *, tz=0, stamp="close"):
    """The held daily value by hand: `fn(list of completed daily closes)`, shown from the first
    bar of the next day. The window of a close-stamped bar is the day its last nanosecond is in."""
    bucket = ts - 1 if stamp == "close" else ts
    day = (bucket + tz) // DAY
    out = np.full(len(ts), np.nan)
    done, held, current, last = [], np.nan, None, None
    for i in range(len(ts)):
        if current is not None and day[i] != current:
            done.append(last)
            held = fn(done)
        current, last = day[i], close[i]
        out[i] = held
    return out


def sma_of(n):
    return lambda done: float(np.mean(done[-n:])) if len(done) >= n else np.nan


# -- the numbers ----------------------------------------------------------------------


def test_a_daily_sma_equals_the_hand_computation():
    d = hourly(12)
    got = rb.expr('at("1d", sma(close, 3))').eval(d, bar_stamp="close")
    want = hand_daily(d["timestamps"], d["close"], sma_of(3))
    assert np.isnan(got[:72]).all() and not np.isnan(got[72:]).any()
    np.testing.assert_allclose(got, want, rtol=1e-12, equal_nan=True)


def test_the_value_is_held_all_day_and_changes_on_the_first_bar_of_the_next():
    d = hourly(6)
    got = rb.expr('at("1d", close)').eval(d, bar_stamp="close")
    close = d["close"]
    for day in range(1, 6):
        window = got[day * 24 : (day + 1) * 24]
        # The whole day shows the previous day's last close, which is bar 24*day - 1.
        assert (window == close[day * 24 - 1]).all(), day
    assert np.isnan(got[:24]).all()
    # The last day's own close is never shown: its window was still open when the data ended.
    assert close[-1] not in set(got[~np.isnan(got)])


def test_inner_fields_are_the_aggregated_windows():
    d = hourly(4)
    got = rb.expr('at("1d", high - low)').eval(d, bar_stamp="close")
    for day in range(1, 4):
        w = slice((day - 1) * 24, day * 24)
        assert (got[day * 24 : (day + 1) * 24] == d["high"][w].max() - d["low"][w].min()).all()
    vol = rb.expr('at("1d", volume)').eval(d, bar_stamp="close")
    assert vol[24] == d["volume"][:24].sum()


def test_mixing_clocks_puts_the_base_bar_against_the_held_daily_value():
    d = hourly(10)
    got = rb.expr('close > at("1d", sma(close, 3))').eval(d, bar_stamp="close")
    held = rb.expr('at("1d", sma(close, 3))').eval(d, bar_stamp="close")
    assert np.isnan(got[:72]).all()
    ok = ~np.isnan(held)
    assert np.array_equal(got[ok], (d["close"][ok] > held[ok]).astype(float))


def test_a_python_built_at_is_the_same_program():
    a = rb.col.close.sma(3).at("1d")
    b = rb.expr('at("1d", sma(close, 3))')
    assert (a.canonical, a.hash) == (b.canonical, b.hash)
    assert b.uses_at and not rb.expr("sma(close, 3)").uses_at


# -- stamps, zones, gaps --------------------------------------------------------------


def test_open_and_close_stamped_bars_give_the_same_windows():
    close_stamped, open_stamped = hourly(8, stamp="close"), hourly(8, stamp="open")
    e = rb.expr('at("1d", ema(close, 2)) - close')
    same_bits(e.eval(close_stamped, bar_stamp="close"), e.eval(open_stamped, bar_stamp="open"))


def test_a_holiday_gap_is_not_a_window():
    d = hourly(8, skip=(3,))
    got = rb.expr('at("1d", close)').eval(d, bar_stamp="close")
    close = d["close"]
    # Day 2 is followed by day 4 in the data: day 4's first bar shows day 2's last close.
    first_after_gap = 3 * 24
    assert got[first_after_gap] == close[first_after_gap - 1]
    np.testing.assert_allclose(got, hand_daily(d["timestamps"], close, lambda done: done[-1]), equal_nan=True)


def test_ist_session_days_end_at_ist_midnight():
    ist = hourly(8, tz=IST)
    utc = hourly(8, tz=0)
    e = rb.expr('at("1d", close)')
    got_ist = e.eval(ist, bar_stamp="close", tz_offset_ns=IST)
    # The same shape in its own zone is the same series.
    same_bits(got_ist, e.eval(utc, bar_stamp="close"))
    # Read as UTC days the IST bars are cut 5.5 hours off, and the series moves.
    got_utc = e.eval(ist, bar_stamp="close", tz_offset_ns=0)
    assert not np.array_equal(np.isnan(got_ist), np.isnan(got_utc)) or not np.array_equal(
        got_ist[~np.isnan(got_ist)], got_utc[~np.isnan(got_utc)]
    )
    np.testing.assert_allclose(
        got_ist, hand_daily(ist["timestamps"], ist["close"], lambda done: done[-1], tz=IST), equal_nan=True
    )


def test_a_session_schedule_drops_closed_days():
    # Mon 2024-01-08 (day 19730) to the Monday after: one bar a day at the 15:30 IST close.
    days = np.arange(19_730, 19_738, dtype=np.int64)
    d = {
        "timestamps": days * DAY + 10 * HOUR,
        "close": np.arange(1.0, 9.0),
    }
    e = rb.expr('at("1d", close)')
    got = e.eval(d, bar_stamp="close", session=SessionSchedule.nse_equity())
    # Saturday and Sunday bars are dropped by the calendar: Friday's close first shows on Monday.
    np.testing.assert_array_equal(got, [np.nan, 1, 2, 3, 4, 4, 4, 5])
    # Without a calendar the weekend bars are days of their own.
    plain = e.eval(d, bar_stamp="close", tz_offset_ns=IST)
    np.testing.assert_array_equal(plain, [np.nan, 1, 2, 3, 4, 5, 6, 7])


def test_a_partial_last_day_is_never_part_of_the_output():
    d = hourly(5)
    e = rb.expr('at("1d", close)')
    full = e.eval(d, bar_stamp="close")
    partial = e.eval(cut(d, 0, 4 * 24 + 7), bar_stamp="close")
    same_bits(partial, full[: 4 * 24 + 7])


# -- no look-ahead --------------------------------------------------------------------


@pytest.mark.parametrize("src", [
    'close > at("1d", sma(close, 2))',
    'at("1d", ema(close, 3)) - at("2h", close)',
    'at("1d", high) > at("1d", open) or volume > 900',
])
def test_nothing_after_a_cut_reaches_the_prefix(src):
    d = hourly(10)
    e = rb.expr(src)
    full = e.eval(d, bar_stamp="close")
    rng = np.random.default_rng(11)
    cuts = sorted({1, 23, 24, 25, 47, 48, 49, 100, 239, *rng.integers(1, 239, 25).tolist()})
    for k in cuts:
        # The prefix on its own, with a mid-day cut among the cuts.
        same_bits(e.eval(cut(d, 0, k), bar_stamp="close"), full[:k])
        # And the same prefix with the future replaced by poison.
        for poison in (1e300, -1e300, np.nan, 0.0):
            bad = {key: v.copy() for key, v in d.items()}
            for col in OHLCV:
                bad[col][k:] = poison
            same_bits(e.eval(bad, bar_stamp="close")[:k], full[:k])


# -- batch, stream, runtime -----------------------------------------------------------


STREAMED = [
    'at("1d", close)',
    'at("1d", sma(close, 3)) - close',
    'close > at("1d", ema(close, 4)) and at("1h", high) > open',
    'at("2h", rsi(3)) < 50 or at("1d", close) > at("1d", open)',
]


def stream_all(e, d, **kw):
    s = e.stream(**kw)
    return np.array([
        s.update({c: float(d[c][i]) for c in OHLCV}, timestamp=int(d["timestamps"][i]))
        for i in range(len(d["close"]))
    ])


@pytest.mark.parametrize("src", STREAMED)
@pytest.mark.parametrize("tz", [0, IST])
@pytest.mark.parametrize("stamp", ["close", "open"])
def test_batch_equals_a_stream_bit_for_bit(src, tz, stamp):
    d = hourly(20, stamp=stamp, tz=tz)
    d["close"][::37] = np.nan  # absent bars
    e = rb.expr(src)
    batch = e.eval(d, bar_stamp=stamp, tz_offset_ns=tz)
    same_bits(batch, stream_all(e, d, bar_stamp=stamp, tz_offset_ns=tz))
    assert (~np.isnan(batch)).any()


class Watch(Strategy):
    """Registers `exprs` on ASSET and keeps what every handle shows on every bar."""

    def __init__(self, exprs):
        super().__init__()
        self.exprs = exprs
        self.rows = []
        self.ready = []

    def on_start(self, ctx):
        self.h = {k: ctx.indicators.expr(v, instrument="ASSET") for k, v in self.exprs.items()}

    def on_bar(self, ctx):
        self.rows.append({k: h.value for k, h in self.h.items()})
        self.ready.append({k: h.ready for k, h in self.h.items()})


def run(strategy, data, mode, *, specs=None, **cfg):
    config = BacktestConfig(initial_capital=1_000_000.0, fees=0.0, **cfg)
    if mode == "batch":
        return run_strategies([strategy], data, config, bar_stamp="close", instruments=specs)[0]
    node = LiveNode([strategy], list(data), config, bar_stamp="close", instruments=specs)
    syms = list(data)
    for i in range(len(data[syms[0]]["timestamps"])):
        node.push_group(
            int(data[syms[0]]["timestamps"][i]),
            {s: tuple(float(data[s][c][i]) for c in OHLCV) for s in syms},
        )
    return node.finish()[0]


EXPRS = {
    "daily": 'at("1d", sma(close, 3))',
    "mixed": 'close > at("1d", ema(close, 2))',
    "wick": 'at("1d", high - low) * 0.5 + at("2h", close)',
}
BOOLEAN = {"mixed"}


@pytest.mark.parametrize("mode", ["batch", "node"])
@pytest.mark.parametrize("tz", [0, IST])
def test_an_indicator_handle_equals_the_batch_evaluation_at_every_bar(mode, tz):
    d = hourly(14, tz=tz)
    s = Watch(EXPRS)
    run(s, {"ASSET": d}, mode, session_tz_offset_ns=tz)
    assert len(s.rows) == len(d["close"])
    for key, text in EXPRS.items():
        want = rb.expr(text).eval(d, bar_stamp="close", tz_offset_ns=tz)
        for i, row in enumerate(s.rows):
            value = row[key]
            if np.isnan(want[i]):
                assert value is None, (key, i, value)
            elif key in BOOLEAN:
                assert value is bool(want[i]), (key, i)
            else:
                assert value == want[i], (key, i, value, want[i])
        assert (~np.isnan(want)).sum() > 100


def test_the_runtime_hubs_daily_bar_and_at_are_one_aggregation():
    """`ctx.indicators.sma(..., timeframe="1d")` is fed the hub's daily bars; `at("1d", sma)` cuts
    its own. They are the same sequence: the hub's value is visible exactly when `at`'s is."""

    class Cross(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_start(self, ctx):
            self.hub = ctx.indicators.sma(instrument="ASSET", period=3, timeframe="1d")
            self.at = ctx.indicators.expr('at("1d", sma(close, 3))', instrument="ASSET")

        def on_bar(self, ctx):
            self.rows.append((self.hub.value, self.at.value))

    for mode in ("batch", "node"):
        s = Cross()
        run(s, {"ASSET": hourly(14)}, mode)
        assert any(h is not None for h, _ in s.rows)
        assert [h for h, _ in s.rows] == [a for _, a in s.rows], mode


def test_a_session_on_the_instrument_is_the_session_of_its_cells():
    nse = SessionSchedule.nse_equity()
    spec = InstrumentSpec.equity("ASSET", session=nse)
    days = np.arange(19_730, 19_760, dtype=np.int64)
    days = days[(days - 19_730) % 7 < 5]  # Monday to Friday
    ts = (days * DAY + 10 * HOUR).astype(np.int64)  # 15:30 IST, the close
    close = 100.0 + np.cumsum(np.random.default_rng(5).normal(0.0, 1.0, len(ts)))
    d = {"timestamps": ts, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": np.full(len(ts), 1000.0)}
    # One bar a day: a daily window is no coarser than the bars (refused), a week is.
    s = Watch({"d": 'at("1w", sma(close, 2))'})
    run(s, {"ASSET": d}, "batch", specs={"ASSET": spec})
    want = rb.expr('at("1w", sma(close, 2))').eval(d, bar_stamp="close", session=nse)
    got = np.array([np.nan if r["d"] is None else r["d"] for r in s.rows])
    same_bits(got, want)


# -- warm-up --------------------------------------------------------------------------


def test_warmup_is_reported_per_clock_and_unknown_without_a_bar_width():
    e = rb.expr('close > at("1d", sma(close, 200)) and sma(close, 3) > 0')
    assert e.warmup is None
    assert e.warmup_by_clock == {"base": 2, "1d": 199}
    assert e.warmup_bars() is None
    assert e.warmup_bars(300 * 10**9) == 200 * 288  # at most 288 5-minute bars a day
    assert rb.expr('at("1month", close)').warmup_bars(HOUR) is None
    plain = rb.expr("sma(close, 5)")
    assert plain.warmup == 4 and plain.warmup_by_clock == {"base": 4} and plain.warmup_bars() == 4


def test_a_handle_is_ready_only_when_every_clock_is_past_its_warmup():
    r = Watch({"x": 'at("1d", sma(close, 3))'})
    run(r, {"ASSET": hourly(8)}, "batch")
    ready = [row["x"] for row in r.ready]
    # The third daily bar is complete on the first bar of day 3: bar 72.
    assert not any(ready[:72]) and all(ready[72:])
    assert r.h["x"].warmup is None and r.h["x"].warmup_by_clock == {"base": 0, "1d": 2}


# -- identity -------------------------------------------------------------------------


def test_programs_without_at_keep_the_hashes_they_had():
    for src, want in {
        "close": "63a14bbe53bfdf48e2086a4f41586de9d16b13d54f8f5998d4acd3d5eae5cb6a",
        "sma(5)": "b2d173d37680ccf900eede9e5c7f1a23f94fa1c067e51e25a3efb35a86ad0ce3",
        "close > sma(20)": "2abfa0c16f9965d7f2dd3a0653be7f772b8f0da32bbae53281c5ff5d9ca84c3e",
        "rsi(14) < 30": "46200ab880ab4e63dd8a7f70d773326680ff6de4cd10b35193f6d377948c9d55",
    }.items():
        e = rb.expr(src)
        assert e.hash == want and e.bound_hash("close", IST) == want, src


def test_the_clock_is_part_of_the_identity_of_an_at_program():
    e = rb.expr('at("1d", sma(close, 3))')
    hashes = {
        e.bound_hash("close"),
        e.bound_hash("open"),
        e.bound_hash("close", IST),
        e.bound_hash("close", 0, SessionSchedule.nse_equity()),
        e.hash,
    }
    assert len(hashes) == 5
    assert e.bound_hash("close") == rb.expr('at("1d", sma(close, 3.0))').bound_hash("close")
    assert e.hash != rb.expr('at("1w", sma(close, 3))').hash != rb.expr("sma(close, 3)").hash


# -- refusals -------------------------------------------------------------------------


def code_of(fn, *a, **kw):
    with pytest.raises(RaptorBTError) as info:
        fn(*a, **kw)
    return info.value


def test_an_at_program_is_refused_what_it_lacks():
    e = rb.expr('at("1d", close)')
    d = hourly(3)
    no_ts = {k: v for k, v in d.items() if k != "timestamps"}
    assert code_of(e.eval, no_ts, bar_stamp="close").code == "missing_field"
    assert code_of(e.eval, d).code == "missing_bar_stamp"
    assert code_of(e.eval, d, bar_stamp="start").code == "invalid_bar_stamp"
    assert code_of(e.stream).code == "missing_bar_stamp"
    s = e.stream(bar_stamp="close")
    assert code_of(s.update, {"close": 1.0}).code == "missing_field"
    short = {**d, "timestamps": d["timestamps"][1:]}
    assert code_of(e.eval, short, bar_stamp="close").code == "length_mismatch"
    # Hand-fed bars carry no timestamp.
    ind = Indicator.expr(e._prog, 0)
    assert code_of(ind.update_bar, 1.0, 1.0, 1.0, 1.0, 1.0).code == "expr_unsupported_here"
    # A program without at ignores timestamps and stamps entirely.
    assert rb.expr("close + 1").eval(no_ts).shape == d["close"].shape


def test_a_bad_timeframe_is_refused_with_what_is_valid():
    for src in ['at("2x", close)', 'at("d", close)', 'at("0m", close)', 'at("100tick", close)', 'at(close, close)']:
        e = code_of(rb.expr, src)
        assert e.code == "expr_type", src
        assert "1d" in e.valid, src
    assert code_of(rb.expr, 'at("1d", at("1w", close))').code == "expr_unsupported_here"
    assert code_of(rb.expr, 'at("1d", 5)').code == "expr_type"
    assert code_of(rb.expr, 'at("1d", macd(close))').code == "expr_type"


def test_a_timeframe_must_be_a_coarser_exact_multiple_of_a_known_bar_width():
    d = hourly(3)
    for src in ['at("1h", close)', 'at("30m", close)', 'at("90m", close)']:
        e = code_of(rb.expr(src).eval, d, bar_stamp="close", bar_ns=HOUR)
        assert e.code == "expr_type" and e.valid == ["2h", "4h", "1d", "1w"], src
    assert code_of(rb.expr('at("1d", close)').eval, d, bar_stamp="close", bar_ns=7 * HOUR).code == "expr_type"
    rb.expr('at("1d", close)').eval(d, bar_stamp="close", bar_ns=HOUR)


# -- NSE and MCX sessions, by hand -----------------------------------------------------------------

MIN = 60 * 10**9
MON = 19_730  # Monday 2024-01-08, days since the epoch
MON_2025 = 20_094  # Monday 2025-01-06


def nse_bars(days, stamp="close", width=5, per_day=75):
    """`width`-minute bars of the NSE session 09:15-15:30 IST (03:45-10:00 UTC), `per_day` a day.
    The close of bar k of the d-th listed day is 1000 + 100 d + k + 1, so a value says which bar it
    came from."""
    ts, c = [], []
    for d, day in enumerate(days):
        for k in range(per_day):
            start = day * DAY + 3 * HOUR + 45 * MIN + k * width * MIN
            ts.append(start + (width * MIN if stamp == "close" else 0))
            c.append(1000.0 + 100.0 * d + k + 1)
    c = np.array(c)
    return {
        "timestamps": np.array(ts, dtype=np.int64),
        "open": c - 0.5,
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
        "volume": np.full(len(c), 10.0),
    }


def last_close(d):
    return 1000.0 + 100.0 * d + 75.0


def close_of(d, k):
    return 1000.0 + 100.0 * d + k + 1


def groups(data):
    stamps = sorted({int(t) for d in data.values() for t in d["timestamps"]})
    index = {s: {int(t): i for i, t in enumerate(d["timestamps"])} for s, d in data.items()}
    for t in stamps:
        yield t, {s: tuple(float(data[s][c][index[s][t]]) for c in OHLCV) for s in data if t in index[s]}


def push_all(node, data, lo=0, hi=None):
    for t, g in list(groups(data))[lo:hi]:
        node.push_group(t, g)


def run_with(strategy, data, mode, *, specs=None, stamp="close", bar_ns=None, **cfg):
    kw = {"bar_stamp": stamp}
    if bar_ns is not None:
        kw["bar_ns"] = bar_ns
    config = BacktestConfig(initial_capital=1_000_000.0, fees=0.0, **cfg)
    if mode == "batch":
        return run_strategies([strategy], data, config, instruments=specs, **kw)[0]
    node = LiveNode([strategy], list(data), config, instruments=specs, **kw)
    push_all(node, data)
    return node.finish()[0]


class TfWatch(Strategy):
    """Registers `exprs` on ASSET (on `timeframe`'s composite bars if given) and keeps the values."""

    def __init__(self, exprs, timeframe=None):
        super().__init__()
        self.exprs, self.timeframe, self.rows = exprs, timeframe, []

    def on_start(self, ctx):
        kw = {} if self.timeframe is None else {"timeframe": self.timeframe}
        self.h = {k: ctx.indicators.expr(v, instrument="ASSET", **kw) for k, v in self.exprs.items()}

    def on_bar(self, ctx):
        self.rows.append({k: h.value for k, h in self.h.items()})


def col(rows, key):
    return np.array([np.nan if r[key] is None else float(r[key]) for r in rows])


def test_the_value_at_the_open_hook_of_the_next_day_is_still_the_old_one():
    """Hourly bars of IST days, close-stamped. An order decided on the last bar of day 0 fills at
    the open of day 1's first bar. At that hook day 0 has not been revealed: at("1d", close) is
    unknown. On that bar's on_bar it is day 0's last close."""

    class Probe(Strategy):
        def on_start(self, ctx):
            self.h = ctx.indicators.expr('at("1d", close)', instrument="ASSET")
            self.at_fill, self.at_bar = "unset", {}

        def on_bar(self, ctx):
            self.at_bar[ctx.idx] = self.h.value
            if ctx.idx == 23:
                self.submit_order(O.Market(side="buy", units=1.0), client_id="p")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled" and e.client_order_id == "p":
                self.at_fill = self.h.value

    ts = (MON * DAY - IST) + np.arange(1, 24 * 3 + 1, dtype=np.int64) * HOUR
    c = 100.0 + np.arange(72, dtype=float)
    d = {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(72, 1.0)}
    for mode in ("batch", "node"):
        s = Probe()
        run_with(s, {"ASSET": d}, mode, fill_timing="next_bar_open", session_tz_offset_ns=IST)
        assert s.at_fill is None, (mode, s.at_fill)
        assert s.at_bar[23] is None and s.at_bar[24] == c[23], mode


@pytest.mark.parametrize("mode", ["batch", "node"])
def test_at_on_open_stamped_bars_in_a_runtime_equals_eval(mode):
    """Hourly bars stamped at their open, the 23:00 IST bar included: a batch run stores the stamp
    as given, a node moves it to the close, and the clock must read each as what it is."""
    ts = (MON * DAY - IST) + np.arange(0, 24 * 5, dtype=np.int64) * HOUR
    c = 100.0 + np.arange(24 * 5, dtype=float)
    d = {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(len(c), 1.0)}
    s = TfWatch({"d": 'at("1d", close)', "h": 'at("4h", close)'})
    run_with(s, {"ASSET": d}, mode, stamp="open", bar_ns=HOUR, session_tz_offset_ns=IST)
    for key, text in s.exprs.items():
        want = rb.expr(text).eval(d, bar_stamp="open", bar_ns=HOUR, tz_offset_ns=IST)
        same_bits(col(s.rows, key), want)
    # By hand: bars 0..23 are IST day 0. Bar 24 is the first of day 1 and shows bar 23's close.
    assert np.isnan(col(s.rows, "d")[23]) and col(s.rows, "d")[24] == c[23]


@pytest.mark.parametrize("stamp", ["close", "open"])
def test_at_1d_on_nse_days_by_hand_never_shows_the_same_day(stamp):
    """Three weekdays, session bound. Day d's every bar reads day d-1's last close, day 0 unknown.
    The last bar of a day (15:30 close-stamped, 15:25 open-stamped) does not read its own day."""
    d = nse_bars([MON, MON + 1, MON + 2], stamp)
    kw = {"bar_stamp": stamp, "session": SessionSchedule.nse_equity()}
    if stamp == "open":
        kw["bar_ns"] = 5 * MIN
    got = rb.expr('at("1d", close)').eval(d, **kw)
    assert np.isnan(got[:75]).all()
    same_bits(got[75:150], np.full(75, last_close(0)))
    same_bits(got[150:225], np.full(75, last_close(1)))


def test_at_1d_with_a_holiday_that_has_bars():
    """Tue 2024-01-09 is a holiday in the schedule and has bars in the data. They are dropped: no
    window, no emission. Wed reads Monday's last close (listed day 0), Thu reads Wed's (day 2)."""
    d = nse_bars([MON, MON + 1, MON + 2, MON + 3])
    sched = SessionSchedule.nse_equity().with_holidays([20240109])
    got = rb.expr('at("1d", close)').eval(d, bar_stamp="close", session=sched)
    assert np.isnan(got[:150]).all()
    same_bits(got[150:225], np.full(75, last_close(0)))
    same_bits(got[225:300], np.full(75, last_close(2)))
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=sched)}
    for mode in ("batch", "node"):
        s = TfWatch({"d": 'at("1d", close)'})
        run_with(s, {"ASSET": d}, mode, specs=spec, session_tz_offset_ns=IST)
        same_bits(col(s.rows, "d"), got)


def test_at_1h_with_a_session_by_hand():
    """Windows 09:15-10:15 .. 15:15-15:30: 12 bars each, the last 3. A window's close shows on the
    first bar of the next window: k = 12 reads bar 11's close, and the 15:15-15:30 window reads
    on the next day's first bar. Bar 74 (15:30) reads bar 71 (15:15), not itself."""
    d = nse_bars([MON, MON + 1])
    sched = SessionSchedule.nse_equity()
    got = rb.expr('at("1h", close)').eval(d, bar_stamp="close", session=sched)
    want = np.full(150, np.nan)
    for day in range(2):
        for k in range(75):
            w = min(k // 12, 6)
            if w >= 1:
                want[day * 75 + k] = close_of(day, w * 12 - 1)
            elif day > 0:
                want[day * 75 + k] = close_of(day - 1, 74)
    same_bits(got, want)
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=sched)}
    for mode in ("batch", "node"):
        s = TfWatch({"h": 'at("1h", close)'})
        run_with(s, {"ASSET": d}, mode, specs=spec, session_tz_offset_ns=IST)
        same_bits(col(s.rows, "h"), want)


def test_mcx_evening_is_one_day():
    """MCX 09:00-23:55 IST (23:30 in the US summer weeks), 30-minute bars from 09:30 to 23:30,
    29 a day, on Mon-Wed of January 2025. One session is one window whichever zone the run names:
    23:30 IST is 18:00 UTC, the same UTC date."""
    ts, c = [], []
    for dd in range(3):
        for k in range(29):
            ts.append((MON_2025 + dd) * DAY + 3 * HOUR + 30 * MIN + (k + 1) * 30 * MIN)
            c.append(1000.0 + 100.0 * dd + k)
    d = {
        "timestamps": np.array(ts, dtype=np.int64), "open": np.array(c), "high": np.array(c),
        "low": np.array(c), "close": np.array(c), "volume": np.ones(len(c)),
    }
    got = rb.expr('at("1d", close)').eval(d, bar_stamp="close", session=SessionSchedule.mcx())
    assert np.isnan(got[:29]).all()
    same_bits(got[29:58], np.full(29, 1028.0))
    same_bits(got[58:], np.full(29, 1128.0))
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=SessionSchedule.mcx())}
    for mode in ("batch", "node"):
        s = TfWatch({"d": 'at("1d", close)'})
        run_with(s, {"ASSET": d}, mode, specs=spec)
        same_bits(col(s.rows, "d"), got)


def test_bars_before_a_schedules_coverage_are_refused_not_all_unknown():
    """The MCX preset covers dates from 2024-11-04; a market hour before it would be a guess.
    `eval` used to return all unknown for such bars and say nothing; a run raises the same code."""
    d = nse_bars([MON, MON + 1])  # January 2024
    mcx = SessionSchedule.mcx()
    for call in (
        lambda: rb.expr('at("1d", close)').eval(d, bar_stamp="close", session=mcx),
        lambda: rb.expr('at("1d", close)').stream(bar_stamp="close", session=mcx).update(
            {"close": 1.0}, timestamp=int(d["timestamps"][0])
        ),
    ):
        with pytest.raises(RaptorBTError) as e:
            call()
        assert e.value.code == "session_not_covered" and e.value.valid, e.value
    s = Session()
    s.load(d, "close", name="w", session="mcx")
    with pytest.raises(RaptorBTError) as e:
        s.eval('at("1d", close)')
    assert e.value.code == "session_not_covered"
    # A program that does not read the schedule is not refused for it.
    assert not np.isnan(rb.expr("close + 1").eval(d, bar_stamp="close", session=mcx)).any()
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=mcx)}
    with pytest.raises(RaptorBTError) as e:
        run_with(TfWatch({"d": 'at("1d", close)'}), {"ASSET": d}, "batch", specs=spec)
    assert e.value.code == "session_not_covered"


def test_at_on_a_composite_stream_by_hand():
    """A 15m stream of the 5m bars, session bound. The first 15m bar of day d (09:15-09:30) is
    emitted by bar k = 3 of that day and carries stamp 09:30: only it closes day d-1's window, so
    the daily value turns at k = 3, not k = 0, and is the close of day d-1's last 15m bar."""
    d = nse_bars([MON, MON + 1, MON + 2])
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=SessionSchedule.nse_equity())}
    rows = {}
    for mode in ("batch", "node"):
        s = TfWatch({"d": 'at("1d", close)'}, timeframe="15m")
        run_with(s, {"ASSET": d}, mode, specs=spec, session_tz_offset_ns=IST)
        rows[mode] = col(s.rows, "d")
    same_bits(rows["batch"], rows["node"])
    got = rows["batch"]
    assert np.isnan(got[:78]).all(), got[70:80]
    same_bits(got[78:153], np.full(75, last_close(0)))
    same_bits(got[153:], np.full(72, last_close(1)))


def test_restart_with_at_mid_day_and_warm_up_equals_the_uninterrupted_run():
    d = nse_bars([MON + i for i in range(5)])
    head = {k: v[:100] for k, v in d.items()}  # warm-up ends mid-day, bar 25 of day 1
    tail = {k: v[100:] for k, v in d.items()}
    exprs = {"d": 'at("1d", sma(close, 2))', "h": 'at("1h", close)', "m": 'close > at("1d", close)'}
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=SessionSchedule.nse_equity())}
    kw = dict(bar_stamp="close", instruments=spec, warmup_bars={"ASSET": head})
    cfg = lambda: BacktestConfig(initial_capital=1_000_000.0, fees=0.0, session_tz_offset_ns=IST)  # noqa: E731
    n = len(tail["close"])
    for cut in (1, 40, 80, n - 1):  # 40 and 80: inside a day and just past a day's end
        journal = rj.Journal({"run": "at"}, memory=True)
        before = TfWatch(exprs)
        node = LiveNode([before], ["ASSET"], cfg(), journal=journal, **kw)
        push_all(node, {"ASSET": tail}, 0, cut)
        records = [r.canonical() for r in journal.since(0)]
        after = TfWatch(exprs)
        restored = LiveNode.restore(
            None, [after], ["ASSET"], cfg(), journal=rj.Journal({"run": "at"}, memory=True),
            records=records, **kw,
        )
        assert after.rows == before.rows, cut
        assert restored.state_digest() == node.state_digest(), cut
        push_all(node, {"ASSET": tail}, cut, n)
        push_all(restored, {"ASSET": tail}, cut, n)
        assert after.rows == before.rows, cut
    # And the same numbers as a backtest over all the bars.
    b = TfWatch(exprs)
    run_with(b, {"ASSET": d}, "batch", specs=spec, session_tz_offset_ns=IST)
    assert before.rows == b.rows


# -- the same text in research and in a strategy ---------------------------------------------------


def test_research_at_1h_is_the_strategys_at_1h_once_it_names_the_session():
    """Without a session the windows are clock hours of the zone (09:00); with the instrument's
    schedule they are anchored at 09:15, as a strategy's cell is. `Session` takes the schedule on
    `load`, `eval` and `sweep`, and the handle's hash and recipe say which one cut the numbers."""
    d = nse_bars([MON, MON + 1], "open")
    text = 'at("1h", close)'
    nse = SessionSchedule.nse_equity()
    strategy = rb.expr(text).eval(d, bar_stamp="open", bar_ns=5 * MIN, session=nse)
    clock_hours = rb.expr(text).eval(d, bar_stamp="open", bar_ns=5 * MIN, tz_offset_ns=IST)
    assert int(np.sum(~((np.isnan(clock_hours) & np.isnan(strategy)) | (clock_hours == strategy)))) > 100

    s = Session()
    s.load(d, "open", 5 * MIN, "plain")
    s.load(d, "open", 5 * MIN, "nse", session="nse_equity")
    plain = s.eval(text, data="plain", tz_offset_ns=IST)
    cut = s.eval(text, data="nse")
    explicit = s.eval(text, data="plain", session=nse)
    assert plain["handle"] != cut["handle"] and cut["handle"] == explicit["handle"]
    same_bits(s.series(cut["handle"]), strategy)
    same_bits(s.series(plain["handle"]), clock_hours)
    assert "session='nse_equity'" in s.describe_handle(cut["handle"])["recipe"]
    # An expression that never reads the calendar is the same handle with or without one.
    assert s.eval("close + 1", data="plain")["handle"] == s.eval("close + 1", data="nse")["handle"]
    # And the strategy that runs on the close-stamped bars sees the same numbers.
    closed = nse_bars([MON, MON + 1], "close")
    spec = {"ASSET": InstrumentSpec.equity("ASSET", session=nse)}
    w = TfWatch({"h": text})
    run_with(w, {"ASSET": closed}, "batch", specs=spec, session_tz_offset_ns=IST)
    same_bits(col(w.rows, "h"), strategy)


def test_a_session_sweep_cuts_every_binding_by_the_schedule():
    d = nse_bars([MON, MON + 1], "open")
    s = Session()
    s.load(d, "open", 5 * MIN, "w", session="nse_equity")
    table = s.sweep('at("1h", close) + $k', k=[0, 1])
    one = s.eval('at("1h", close) + 1', data="w")
    assert one["handle"] in {row[0] for row in table["rows"]}


def test_stats_on_series_of_other_bars_join_on_timestamps_or_refuse():
    rng = np.random.default_rng(0)
    t1 = np.arange(200, dtype=np.int64) * DAY
    x = 100 + np.cumsum(rng.normal(size=200))
    y = x + rng.normal(size=200)
    s = Session()
    s.load({"timestamps": t1, "close": x}, "close", None, "a")
    s.load({"timestamps": t1 + 50 * DAY, "close": y}, "close", None, "b")
    ha, hb = s.eval("close", data="a")["handle"], s.eval("close", data="b")["handle"]
    out = s.stats("cointegration", ha, hb)
    aligned = out["analysis"]["aligned"]
    assert aligned["common"] == 150 and aligned["first_timestamp"] == 50 * DAY
    assert aligned["dropped"] == {ha: 50, hb: 50}
    # No instant in common: refused, never a statistic of rows that are different days.
    s.load({"timestamps": t1 + 500 * DAY, "close": y}, "close", None, "c")
    hc = s.eval("close", data="c")["handle"]
    with pytest.raises(RaptorBTError) as e:
        s.stats("cointegration", ha, hc)
    assert e.value.code == "invalid_input"
    # A keyword naming a multi-output handle is typed, not a KeyError.
    macd = s.eval("macd(close)", data="a")["handle"]
    with pytest.raises(RaptorBTError) as e:
        s.stats("hac_tstat", ha, **{"target": macd})
    assert e.value.code == "unknown_handle"


# -- 75-minute bars, a primary stream as fine as its bars -------------------------------------------


def test_75_minute_bars_make_a_session_day_and_a_refusal_lists_what_is_valid():
    d = nse_bars([MON, MON + 1, MON + 2], width=75, per_day=5)
    nse = SessionSchedule.nse_equity()
    got = rb.expr('at("1d", close)').eval(d, bar_stamp="close", bar_ns=75 * MIN, session=nse)
    assert np.isnan(got[:5]).all()
    # Day 1 reads day 0's fifth bar (1000 + 4 + 1), day 2 reads day 1's (1105).
    same_bits(got[5:10], np.full(5, 1005.0))
    same_bits(got[10:], np.full(5, 1105.0))
    with pytest.raises(RaptorBTError) as e:
        rb.expr('at("1d", close)').eval(d, bar_stamp="close", bar_ns=75 * MIN)
    assert e.value.code == "expr_type" and e.value.valid == ["150m", "225m", "5h", "375m"]
    with pytest.raises(RaptorBTError) as e:
        rb.expr('at("1h", close)').eval(d, bar_stamp="close", bar_ns=75 * MIN, session=nse)
    assert e.value.valid == ["1d", "1w"]


def test_an_expression_that_would_hold_more_than_the_limit_is_refused_stating_bytes():
    e = rb.expr("sma(close, 5000000)")
    assert 40_000_000 <= e.state_bytes(400) < 512 << 20
    big = " + ".join(f"sma(close, {16_777_216 - k})" for k in range(8))
    with pytest.raises(RaptorBTError) as err:
        rb.expr(big).eval({"close": np.arange(400, dtype=float)})
    assert err.value.code == "expr_unsupported_here" and "bytes" in str(err.value)


def test_a_node_told_no_width_refuses_a_timeframe_as_fine_as_its_bars_on_the_second_bar():
    """No warm-up and no bar_ns: the node works the width out of its first two bars and raises
    the batch refusal (`expr_type`, with the valid timeframes) before `on_bar` reads a value."""
    d = hourly(3)

    class Seen(Strategy):
        def __init__(self, text):
            super().__init__()
            self.text, self.values = text, []

        def on_start(self, ctx):
            self.h = ctx.indicators.expr(self.text, instrument="ASSET")

        def on_bar(self, ctx):
            self.values.append(self.h.value)

    node_cfg = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
    bad = Seen('at("1h", close)')
    node = LiveNode([bad], ["ASSET"], node_cfg, bar_stamp="close")
    with pytest.raises(RaptorBTError) as e:
        for i in range(len(d["close"])):
            node.push_group(int(d["timestamps"][i]), {"ASSET": tuple(float(d[c][i]) for c in OHLCV)})
    assert e.value.code == "expr_type" and "2h" in e.value.valid and "1d" in e.value.valid
    assert all(v is None for v in bad.values) and len(bad.values) <= 1

    good = Seen('at("1d", close)')
    node = LiveNode([good], ["ASSET"], node_cfg, bar_stamp="close")
    for i in range(len(d["close"])):
        node.push_group(int(d["timestamps"][i]), {"ASSET": tuple(float(d[c][i]) for c in OHLCV)})
    want = rb.expr('at("1d", close)').eval(d, bar_stamp="close")
    got = np.array([np.nan if v is None else v for v in good.values])
    same_bits(got, want)
