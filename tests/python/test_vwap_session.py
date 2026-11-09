"""`raptorbt.vwap_session`: the typical-price VWAP that restarts at each session start.

`vwap` stays cumulative over the whole array; this one is for several sessions in one array.
"""

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import SessionSchedule
from raptorbt.errors import RaptorBTError

DAY = 24 * 60 * 60 * 10**9
NSE = SessionSchedule.nse_equity()


def test_two_sessions_by_hand():
    # tp = (h + l + c) / 3: session 1 is 11, 12, 13 on volume 10, 30, 10 -> 11, 470/40, 600/50;
    # session 2 is 20, 21, 30 on volume 0, 5, 15 -> NaN, 105/5, 555/20.
    high = np.array([12.0, 13.0, 14.0, 21.0, 24.0, 33.0])
    low = np.array([10.0, 11.0, 12.0, 19.0, 18.0, 27.0])
    close = np.array([11.0, 12.0, 13.0, 20.0, 21.0, 30.0])
    volume = np.array([10.0, 30.0, 10.0, 0.0, 5.0, 15.0])
    starts = np.array([True, False, False, True, False, False])
    got = rb.vwap_session(high, low, close, volume, starts)
    assert got[[0, 1, 2, 4, 5]].tolist() == [11.0, 11.75, 12.0, 21.0, 27.75]
    assert np.isnan(got[3])


def test_one_start_at_zero_is_vwap():
    n = 40
    k = np.arange(n, dtype=float)
    close = 100.0 + np.sin(k / 3.0)
    high, low, volume = close + 0.4, close - 0.4, 10.0 + k % 5
    starts = np.zeros(n, dtype=bool)
    starts[0] = True
    a = rb.vwap(high, low, close, volume)
    b = rb.vwap_session(high, low, close, volume, starts)
    assert np.array_equal(a.view(np.int64), b.view(np.int64))


def test_length_mismatch_is_refused():
    x = np.ones(3)
    with pytest.raises(RaptorBTError):
        rb.vwap_session(x, x, x, x, np.array([True, False]))


def test_session_vwap_expression_equals_vwap_session_over_days():
    days, per_day = 4, 25
    mon = 19_730  # Monday 2024-01-08
    ts = np.concatenate(
        [
            (mon + d) * DAY + (225 + 15 * (np.arange(per_day) + 1)) * 60 * 10**9
            for d in range(days)
        ]
    ).astype(np.int64)
    k = np.arange(len(ts))
    close = 100.0 + np.sin(k / 7.0) * 3 + (k // per_day)
    data = {
        "timestamps": ts,
        "open": close,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": 100.0 + (k % 7),
    }
    expr = rb.expr("session_vwap()").eval(data, bar_stamp="close", session=NSE)
    day = (ts - 1) // DAY  # a bar stamped at its close belongs to the date its open is on
    starts = np.concatenate([[True], day[1:] != day[:-1]])
    got = rb.vwap_session(data["high"], data["low"], data["close"], data["volume"], starts)
    assert np.array_equal(np.isnan(expr), np.isnan(got))
    assert np.array_equal(expr.view(np.int64), got.view(np.int64))
