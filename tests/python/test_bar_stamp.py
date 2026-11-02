"""What the timestamp of an input bar means: open or close.

``BarAggregator`` and ``aggregate_bars`` take ``stamp="open"|"close"``. The
same bars, stamped either way, must build identical composites.
"""

import numpy as np
import pytest

import raptorbt

MIN = 60 * 10**9


def _bars(n=12):
    close = 100.0 + np.arange(n, dtype=np.float64)
    return close, close + 0.5, close - 0.5, close + 0.25, np.ones(n)


def _stream(stamp, step_ts):
    agg = raptorbt.BarAggregator(5, "m", stamp=stamp)
    o, h, l, c, v = _bars()
    out = []
    for i, ts in enumerate(step_ts):
        bar = agg.push_bar(int(ts), o[i], h[i], l[i], c[i], v[i])
        if bar is not None:
            out.append(bar)
    tail = agg.flush()
    if tail is not None:
        out.append(tail)
    return out


def test_close_stamped_bars_build_the_same_composites_as_open_stamped():
    opens = np.arange(12, dtype=np.int64) * MIN
    by_open = _stream("open", opens)
    by_close = _stream("close", opens + MIN)
    assert by_open == by_close
    assert [b[0] for b in by_open] == [5 * MIN, 10 * MIN, 15 * MIN]


def test_default_stamp_is_open_and_close_boundary_bar_lands_in_the_window_it_ends():
    boundary = np.arange(1, 13, dtype=np.int64) * MIN
    assert raptorbt.BarAggregator(5, "m").stamp == "open"
    # Read as open stamps, the bar stamped 5m starts the second window.
    assert _stream("open", boundary)[0][0] == 5 * MIN
    assert _stream("open", boundary)[0][5] == 4.0
    # Read as close stamps, it ends the first.
    assert _stream("close", boundary)[0][5] == 5.0


def test_aggregate_bars_stamp():
    o, h, l, c, v = _bars()
    ts = np.arange(12, dtype=np.int64) * MIN
    a = raptorbt.aggregate_bars(ts, o, h, l, c, v, 5, "m", stamp="open")
    b = raptorbt.aggregate_bars(ts + MIN, o, h, l, c, v, 5, "m", stamp="close")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("bad", ["", "mid", "Open", None])
def test_invalid_stamp_is_refused_with_the_valid_values(bad):
    with pytest.raises((ValueError, TypeError)) as exc:
        raptorbt.BarAggregator(5, "m", stamp=bad)
    if isinstance(exc.value, ValueError):
        assert "open" in str(exc.value) and "close" in str(exc.value)
    o, h, l, c, v = _bars()
    with pytest.raises(ValueError, match="close"):
        raptorbt.aggregate_bars(np.arange(12, dtype=np.int64), o, h, l, c, v, 5, "m", stamp="mid")


def test_count_units_ignore_stamp():
    ts = np.arange(1, 7, dtype=np.int64) * MIN
    o, h, l, c, v = _bars(6)
    a = raptorbt.aggregate_bars(ts, o, h, l, c, v, 3, "tick", stamp="open")
    b = raptorbt.aggregate_bars(ts, o, h, l, c, v, 3, "tick", stamp="close")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_trades_ignore_stamp():
    for stamp in ("open", "close"):
        agg = raptorbt.BarAggregator(5, "m", stamp=stamp)
        assert agg.push_trade(0, 100.0, 1.0) is None
        # A print exactly on the boundary opens the next window.
        bar = agg.push_trade(5 * MIN, 101.0, 1.0)
        assert bar is not None and bar[0] == 5 * MIN and bar[5] == 1.0
