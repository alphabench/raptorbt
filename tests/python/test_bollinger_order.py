"""Bollinger bands come in one order, ``(upper, middle, lower)``, with names, in every form."""

import numpy as np
import pytest

import raptorbt
from raptorbt import BollingerBands, Indicator, Strategy

CLOSE = 100.0 + np.cumsum(np.random.default_rng(3).normal(0, 1.0, 40))


def test_the_array_form_returns_named_bands():
    bands = raptorbt.bollinger_bands(CLOSE, 5, 2.0)
    assert isinstance(bands, BollingerBands)
    assert bands._fields == ("upper", "middle", "lower")
    upper, middle, lower = bands
    assert np.all(upper[4:] >= middle[4:]) and np.all(middle[4:] >= lower[4:])


def test_the_streaming_form_has_the_same_order_and_names():
    ind = Indicator.bollinger(5, 2.0)
    last = None
    for c in CLOSE:
        last = ind.update_bar(c, c, c, c)
    batch = raptorbt.bollinger_bands(CLOSE, 5, 2.0)
    assert isinstance(last, BollingerBands) and isinstance(ind.value, BollingerBands)
    for name in ("upper", "middle", "lower"):
        assert getattr(last, name) == pytest.approx(getattr(batch, name)[-1])
    upper, middle, lower = last
    assert upper > middle > lower


def test_the_registered_handle_has_the_same_order_and_names():
    seen = []

    class S(Strategy):
        def on_start(self, ctx):
            self.b = ctx.indicators.bollinger(instrument="ASSET", period=5, k=2.0)

        def on_bar(self, ctx):
            if self.b.ready:
                seen.append((self.b.value, self.b.values))

    n = len(CLOSE)
    raptorbt.run_strategy_backtest(
        S(), np.arange(n, dtype=np.int64), CLOSE, CLOSE + 1, CLOSE - 1, CLOSE, np.ones(n),
        bar_stamp="close",
    )
    batch = raptorbt.bollinger_bands(CLOSE, 5, 2.0)
    value, values = seen[-1]
    assert isinstance(value, BollingerBands)
    assert values._fields == ("upper", "middle", "lower")
    assert tuple(value) == tuple(values)
    assert value.upper == pytest.approx(batch.upper[-1])
    assert value.middle == pytest.approx(batch.middle[-1])
