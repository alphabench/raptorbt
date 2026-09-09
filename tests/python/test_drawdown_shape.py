"""Ulcer index and time under water, exercised through the wheel.

Max drawdown is one order statistic: it answers "how deep" and cannot tell a
-8% pit lasting five bars from a -8% pit lasting two hundred. The second is
the one holders abandon. Ulcer weights each shortfall by how long it
persisted; time under water measures persistence with depth removed.

Both are computed from the same streamed drawdown curve that max_drawdown_pct
folds, NOT by rebuilding one from the equity series -- the two seed their
running peak differently and disagree whenever a run's first equity sample
differs from its initial capital. These tests recompute both folds in NumPy
from the curve the wheel itself reports, so any drift between a metric and the
curve it claims to summarize fails at the boundary consumers actually use.
"""

import numpy as np

from raptorbt import Strategy, run_strategy_backtest

DAY_NS = 86_400_000_000_000


def _ohlcv(close):
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    openp = np.empty(n, dtype=np.float64)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    return {
        "timestamps": np.arange(n, dtype=np.int64) * DAY_NS,
        "open": openp,
        "high": np.maximum(openp, close) * 1.004,
        "low": np.minimum(openp, close) * 0.996,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }


class Churn(Strategy):
    """Alternate entry and exit every bar, so the curve actually moves."""

    def on_bar(self, ctx):
        if ctx.position is None:
            self.enter()
        else:
            self.close_position()


class Idle(Strategy):
    """Never trade: equity stays flat at initial capital."""

    def on_bar(self, ctx):
        return


def _run(seed=7, n=400):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=n)))
    return run_strategy_backtest(Churn, **_ohlcv(close))


def test_the_reported_ulcer_matches_the_reported_drawdown_curve():
    result = _run()
    dd = np.asarray(result.drawdown_curve(), dtype=np.float64)
    assert dd.size > 0

    expected = float(np.sqrt(np.mean(dd**2)))
    assert abs(result.metrics.ulcer_index - expected) < 1e-9


def test_time_under_water_matches_the_reported_drawdown_curve():
    result = _run()
    dd = np.asarray(result.drawdown_curve(), dtype=np.float64)

    expected = float(np.mean(dd > 0.0) * 100.0)
    assert abs(result.metrics.time_under_water_pct - expected) < 1e-9


def test_ulcer_divides_by_the_full_length_including_the_zeros():
    # Dividing by the count of underwater samples instead of the curve length
    # is the most likely reimplementation error, and it inflates the metric.
    result = _run()
    dd = np.asarray(result.drawdown_curve(), dtype=np.float64)
    underwater = int((dd > 0.0).sum())
    assert 0 < underwater < dd.size, "need a mixed curve for this to bite"

    wrong = float(np.sqrt(np.sum(dd**2) / underwater))
    assert abs(result.metrics.ulcer_index - wrong) > 1e-9


def test_ulcer_index_reaches_python_as_a_plain_float():
    # Neither metric can be non-finite, so neither goes through the finite()
    # scrubber that turns the ratio fields into Optional. A None here means
    # someone wrapped it.
    m = _run().metrics
    assert isinstance(m.ulcer_index, float)
    assert isinstance(m.time_under_water_pct, float)


def test_ulcer_never_exceeds_max_drawdown():
    m = _run().metrics
    assert m.ulcer_index <= m.max_drawdown_pct + 1e-9


def test_time_under_water_is_a_percentage():
    m = _run().metrics
    assert 0.0 <= m.time_under_water_pct <= 100.0


def test_a_flat_curve_is_never_under_water():
    # A strategy that never trades never moves its equity, so nothing is ever
    # below the high-water mark. This is the only genuine zero case: even
    # buy-and-hold on a strictly rising line dips for one bar, because the
    # entry fee is paid before the position appreciates -- which the metric
    # correctly notices.
    result = run_strategy_backtest(Idle, **_ohlcv(np.linspace(100.0, 160.0, 120)))
    m = result.metrics
    assert m.ulcer_index == 0.0
    assert m.time_under_water_pct == 0.0


def test_to_dict_carries_the_new_drawdown_shape_keys():
    d = _run().metrics.to_dict()
    assert "Ulcer Index" in d
    assert "Time Under Water [%]" in d
    assert len(d) == 28
