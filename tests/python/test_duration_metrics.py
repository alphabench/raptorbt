"""Duration metrics, exercised through the wheel.

A bar is not a unit of time. On daily data one bar is one day, so counting
bars and calling the answer "days" was right by accident. On a tick run one
bar is one tick: a trade lasting 45 seconds reported "329", and a six-day
backtest reported a drawdown of ~93,510 -- which a caller printed as 93,510
days, roughly 256 years.

So the engine reports the same figures twice: the bar counts, which still
mean bars, and `*_secs`, taken from the timestamps the run already carried.
These tests pin the property that actually matters -- the seconds figure
tracks the spacing of the data, and the bar count does not.

They also pin the exposure cap: time in the market cannot exceed the time
the backtest ran.
"""

import numpy as np

from raptorbt import Strategy, run_strategy_backtest

DAY_NS = 86_400_000_000_000
SEC_NS = 1_000_000_000


def _ohlcv(close, step_ns):
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    openp = np.empty(n, dtype=np.float64)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    return {
        "timestamps": np.arange(n, dtype=np.int64) * step_ns,
        "open": openp,
        "high": np.maximum(openp, close) * 1.004,
        "low": np.minimum(openp, close) * 0.996,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }


class HoldFourBars(Strategy):
    """Enter, hold exactly four bars, exit. Repeat."""

    def on_bar(self, ctx):
        if ctx.position is None:
            self.enter()
            self._entered = ctx.idx
        elif ctx.idx - getattr(self, "_entered", ctx.idx) >= 4:
            self.close_position()


def _close_series():
    rng = np.random.default_rng(7)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=200)))


def test_the_seconds_figure_tracks_the_spacing_and_the_bar_count_does_not():
    """The same trades, one second apart and one day apart."""
    close = _close_series()

    per_second = run_strategy_backtest(HoldFourBars, **_ohlcv(close, SEC_NS))
    per_day = run_strategy_backtest(HoldFourBars, **_ohlcv(close, DAY_NS))

    assert per_second.trades(), "no trades produced"

    # Identical bars, identical signals: the bar count cannot depend on how
    # far apart in time those bars sat.
    assert (
        abs(per_second.metrics.avg_holding_period - per_day.metrics.avg_holding_period)
        < 1e-9
    )

    secs = per_second.metrics.avg_holding_period_secs
    days = per_day.metrics.avg_holding_period_secs
    assert secs is not None and days is not None, "timestamps were supplied"

    # ...but the elapsed time must, and by exactly the ratio of the spacing.
    assert abs(days - secs * 86_400.0) < 1e-3, (
        f"day-spaced bars should report 86,400x the seconds of second-spaced "
        f"bars: {days} vs {secs}"
    )

    # The concrete claim: four one-second bars is four seconds, not four days.
    assert abs(secs - per_second.metrics.avg_holding_period) < 1e-6


def test_drawdown_duration_is_reported_in_seconds_too():
    close = _close_series()
    result = run_strategy_backtest(HoldFourBars, **_ohlcv(close, SEC_NS))

    bars = result.metrics.max_drawdown_duration
    secs = result.metrics.max_drawdown_duration_secs
    assert bars > 0, "the run drew down at some point"
    assert secs is not None, "timestamps were supplied"
    # One-second bars, so the stretch in seconds is the span between the first
    # and last underwater bar -- one interval shorter than the bar count.
    assert abs(secs - (bars - 1)) < 1e-6, f"{secs}s vs {bars} one-second bars"


def test_exposure_never_exceeds_the_time_available():
    close = _close_series()
    for step in (SEC_NS, DAY_NS):
        result = run_strategy_backtest(HoldFourBars, **_ohlcv(close, step))
        assert 0.0 <= result.metrics.exposure_pct <= 100.0, (
            f"exposure {result.metrics.exposure_pct} is outside 0-100%"
        )


def test_the_metrics_dict_carries_the_seconds_figure():
    close = _close_series()
    result = run_strategy_backtest(HoldFourBars, **_ohlcv(close, SEC_NS))
    as_dict = result.metrics.to_dict()

    assert "Max Drawdown Duration" in as_dict
    assert "Max Drawdown Duration [s]" in as_dict
    assert as_dict["Max Drawdown Duration [s]"] == pytest_approx(
        result.metrics.max_drawdown_duration_secs
    )


def pytest_approx(value):
    """Local shim so this file needs no pytest import for one comparison."""

    class _Approx:
        def __eq__(self, other):
            if value is None or other is None:
                return value is other
            return abs(float(other) - float(value)) < 1e-9

    return _Approx()
