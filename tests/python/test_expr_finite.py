"""Every expression value is a number or NaN, never an infinity (skills/expressions/SKILL.md),
and session functions need bars stamped inside the session."""

import math

import numpy as np

import raptorbt as rb

MIN = 60 * 10**9
HOUR = 60 * MIN
DAY = 24 * HOUR
MON = 19_730  # 2024-01-08, a Monday (NSE trading day)
IST = 19_800 * 10**9
NSE = rb.SessionSchedule.nse_equity()


def test_a_bare_field_turns_an_infinite_input_into_nan():
    close = np.array([1.0, np.inf, 3.0, -np.inf])
    got = rb.expr("close").eval({"close": close})
    assert got[0] == 1.0 and got[2] == 3.0
    assert np.isnan(got[1]) and np.isnan(got[3])
    # the same rule as after an operator
    plus0 = rb.expr("close + 0").eval({"close": close})
    assert np.array_equal(np.isnan(got), np.isnan(plus0))


def test_session_volume_overflow_is_nan_not_inf():
    """Two volumes of 1e308 in one NSE session: 1e308 + 1e308 overflows, so the second is NaN."""
    ts = np.array([MON * DAY + 3 * HOUR + 45 * MIN + (k + 1) * 15 * MIN for k in range(2)], dtype=np.int64)
    data = {
        "open": np.array([100.0, 100.0]),
        "high": np.array([101.0, 101.0]),
        "low": np.array([99.0, 99.0]),
        "close": np.array([100.0, 100.0]),
        "volume": np.array([1e308, 1e308]),
        "timestamps": ts,
    }
    sv = rb.expr("session_volume()").eval(data, bar_stamp="close", session=NSE)
    assert sv[0] == 1e308 and math.isnan(sv[1])


def test_session_functions_hold_no_infinite_input():
    ts = np.array([MON * DAY + 3 * HOUR + 45 * MIN + (k + 1) * 15 * MIN for k in range(3)], dtype=np.int64)
    one = np.array([100.0, np.inf, 100.0])
    data = {"open": one, "high": one, "low": one, "close": one, "volume": one, "timestamps": ts}
    for text in ("session_open()", "session_high()", "session_low()", "session_volume()",
                 "session_vwap()", "session_return()", "session_first(close, 1)"):
        out = rb.expr(text).eval(data, bar_stamp="close", session=NSE)
        assert not np.isinf(out).any(), (text, out.tolist())


def daily(stamps):
    return {
        "open": np.array([100.0, 106.0, 110.0]),
        "high": np.array([105.0, 108.0, 111.0]),
        "low": np.array([99.0, 105.0, 109.0]),
        "close": np.array([104.2, 107.5, 110.5]),
        "volume": np.ones(3),
        "timestamps": np.array(stamps, dtype=np.int64),
    }


def test_daily_bars_stamped_at_midnight_are_outside_the_session_and_give_nan():
    """Documented behaviour: a session function needs bars stamped inside the session. Daily bars
    stamped at 00:00 IST are outside it: NaN, with no refusal."""
    data = daily([(MON + d) * DAY - IST for d in range(3)])
    for stamp in ("open", "close"):
        for text in ("session_gap()", "session_bar()", "session_open()"):
            assert np.isnan(rb.expr(text).eval(data, bar_stamp=stamp, session=NSE)).all()


def test_daily_bars_stamped_at_the_session_close_work():
    """The same bars stamped 15:30 IST (10:00 UTC), close-stamped: inside the session. Hand:
    gap on day 2 = 106 / 104.2 - 1 = 0.017274472168905985, on day 3 = 110 / 107.5 - 1 =
    0.023255813953488413."""
    data = daily([(MON + d) * DAY + 10 * HOUR for d in range(3)])
    gap = rb.expr("session_gap()").eval(data, bar_stamp="close", session=NSE)
    assert math.isnan(gap[0])
    assert abs(gap[1] - (106.0 / 104.2 - 1.0)) < 1e-15
    assert abs(gap[2] - (110.0 / 107.5 - 1.0)) < 1e-15
