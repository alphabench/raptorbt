"""Resampling: the standalone API on the engine's one time-window bar builder."""

import numpy as np
import pytest

import raptorbt as r
from raptorbt import sampling

S = 1_000_000_000
M = 60 * S
D0 = 1_767_225_600 * S


def ticks(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    ts = D0 + np.cumsum(rng.integers(1, 30 * S, n)).astype(np.int64)
    px = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    qty = rng.integers(1, 50, n).astype(np.float64)
    return ts, px, qty


def test_the_default_resampler_is_the_bar_aggregator():
    """Left-closed, no gaps: the same bars as the plain aggregator, bit for bit."""
    ts, px, qty = ticks()
    got = r.resample_trades(ts, px, qty, 5 * M)
    want = r.aggregate_bars(ts, px, px, px, px, qty, 5, "m")
    for name, w in zip(["timestamp", "open", "high", "low", "close", "volume"], want):
        assert np.array_equal(got[name], w), name
    assert got["complete"].all() and not got["carried"].any()
    assert got["n_source"].sum() == len(ts)


def test_a_boundary_print_follows_the_declared_edge():
    ts = np.array([D0 + 10 * S, D0 + M, D0 + M + S], dtype=np.int64)
    px = np.array([100.0, 101.0, 102.0])
    qty = np.ones(3)
    left = r.resample_trades(ts, px, qty, M, closed="left")
    right = r.resample_trades(ts, px, qty, M, closed="right")
    assert left["close"].tolist() == [100.0, 102.0]
    assert right["close"].tolist() == [101.0, 102.0]
    assert right["volume"].tolist() == [2.0, 1.0]


def test_streaming_and_batch_agree():
    ts, px, qty = ticks(seed=4)
    batch = r.resample_trades(ts, px, qty, 3 * M, closed="right", gaps="carry")
    s = r.Resampler(3 * M, closed="right", gaps="carry")
    bars = []
    for t, p, q in zip(ts, px, qty):
        bars += s.push_trade(int(t), float(p), float(q))
    bars += s.flush()
    assert [b.ts for b in bars] == batch["timestamp"].tolist()
    assert [b.close for b in bars] == batch["close"].tolist()
    assert [b.closed_by == "carry" for b in bars] == batch["carried"].tolist()


def test_bars_resample_with_an_anchor_and_a_zone():
    # 1-minute bars stamped at their close, into 30-minute bars from a 09:15 IST open
    open_ns = (9 * 3600 + 15 * 60) * S
    day = D0 - D0 % (86_400 * S)
    local0 = day + open_ns - r.IST_OFFSET_NS
    n = 90
    ts = local0 + np.arange(1, n + 1, dtype=np.int64) * M  # closes at 09:16 .. 10:45
    px = 100.0 + np.arange(n, dtype=np.float64)
    out = r.resample_bars(
        ts, px, px + 1, px - 1, px, np.ones(n), 30 * M, source_ns=M,
        anchor_ns=open_ns, tz_offset_ns=r.IST_OFFSET_NS, closed="right",
    )
    assert out["timestamp"].tolist() == [local0 + 30 * M, local0 + 60 * M, local0 + 90 * M]
    assert out["n_source"].tolist() == [30, 30, 30] and out["complete"].all()
    assert out["open"][1] == 130.0 and out["volume"].tolist() == [30.0] * 3


def test_a_short_bucket_is_flagged_or_dropped():
    ts = (D0 + np.array([1, 2, 3, 7], dtype=np.int64) * M)
    px = np.ones(4)
    kw = dict(source_ns=M, closed="right", flush=True)
    out = r.resample_bars(ts, px, px, px, px, px, 5 * M, **kw)
    assert out["complete"].tolist() == [False, False] and out["n_source"].tolist() == [3, 1]
    out = r.resample_bars(ts, px, px, px, px, px, 5 * M, min_source=3, **kw)
    assert out["n_source"].tolist() == [3]


def test_every_empty_interval_is_carried_even_after_a_clock_close():
    s = sampling.TimeBars("a", M, "b", close_on="clock", gaps="carry")
    got = s.push_trade(D0 + S, 100.0, 1.0)
    got += s.on_clock(D0 + M)  # the clock closes the bar
    assert [b.closed_by for b in got] == ["clock"]
    got = s.push_trade(D0 + 5 * M - S, 90.0, 1.0)
    # the reference carried only the intervals one clock step passed; now each empty minute is a bar
    assert [(b.ts - D0) // M for b in got] == [2, 3, 4]
    assert all(b.closed_by == "carry" and b.close == 100.0 and b.volume == 0.0 for b in got)


def test_late_and_bad_input_is_refused_and_the_resampler_is_unchanged():
    s = r.Resampler(M, closed="right")
    s.push_trade(D0 + 3 * M, 100.0, 1.0)
    with pytest.raises(ValueError) as e:
        s.push_trade(D0 + M, 99.0, 1.0)
    assert getattr(e.value, "code", None) == "decreasing_timestamps"
    with pytest.raises(ValueError):
        s.push_trade(D0 + 4 * M, float("nan"), 1.0)
    (bar,) = s.flush()
    assert (bar.n_source, bar.close) == (1, 100.0)

    b = r.Resampler(5 * M, source_ns=M, closed="right")
    with pytest.raises(ValueError):
        b.push_bar(D0 + M + S, 1, 1, 1, 1, 1)  # off the source grid
    with pytest.raises(ValueError):
        b.push_bar(D0 + M, 1, 1, 1, 1, 1, period=2 * M)  # another width
    with pytest.raises(ValueError):
        r.Resampler(5 * M, source_ns=2 * M)  # not a multiple: never an approximation
    with pytest.raises(ValueError):
        r.resample_trades(np.array([D0 + M, D0], dtype=np.int64), np.ones(2), np.ones(2), M)


def test_a_session_anchored_grid_restarts_at_each_open():
    s = sampling.TimeBars("a", 30 * M, "b", anchor="session")
    assert s.push_trade(D0 + 5 * M, 99.0, 1.0) == []  # before any open
    open_ns = D0 + 100 * M + 17 * S
    assert s.on_session(True, open_ns) == []
    s.push_trade(open_ns + M, 100.0, 1.0)
    bars = s.push_trade(open_ns + 31 * M, 101.0, 1.0)
    assert [b.ts for b in bars] == [open_ns + 30 * M]
    (last,) = s.on_session(False, open_ns + 45 * M)
    assert (last.ts, last.closed_by) == (open_ns + 45 * M, "session")


def test_the_gil_is_released_while_resampling():
    import threading
    import time

    ts, px, qty = ticks(n=1_000_000, seed=2)
    ticked = []
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ticked.append(time.perf_counter())
            time.sleep(0.0005)

    t = threading.Thread(target=spin)
    t.start()
    try:
        for _ in range(5):
            r.resample_trades(ts, px, qty, M, closed="right", gaps="carry")
    finally:
        stop.set()
        t.join()
    assert len(ticked) > 5
