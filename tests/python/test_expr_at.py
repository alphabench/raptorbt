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
from raptorbt.errors import RaptorBTError

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
    s = Watch({"d": 'at("1d", sma(close, 2))'})
    run(s, {"ASSET": d}, "batch", specs={"ASSET": spec})
    want = rb.expr('at("1d", sma(close, 2))').eval(d, bar_stamp="close", session=nse)
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
        "rsi(14) < 30": "379356a448b24963589402b063034317b7f2db927289be453afd3faaeead4328",
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
