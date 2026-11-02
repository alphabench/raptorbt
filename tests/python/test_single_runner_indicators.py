"""``ctx.indicators`` works in the single-symbol runner as it does in ``run_strategies``."""

import numpy as np
import pytest

import raptorbt
from raptorbt import Strategy


def _bars(n=20):
    c = np.linspace(100.0, 110.0, n)
    return dict(
        timestamps=np.arange(n, dtype=np.int64),
        open=c,
        high=c + 1.0,
        low=c - 1.0,
        close=c,
        volume=np.ones(n),
        bar_stamp="close",
    )


def test_registered_indicators_update_in_run_strategy_backtest():
    seen = []

    class S(Strategy):
        def on_start(self, ctx):
            self.sma = ctx.indicators.sma(instrument="ASSET", period=3)

        def on_bar(self, ctx):
            seen.append(self.sma.value)

    raptorbt.run_strategy_backtest(S(), **_bars())
    assert seen[:2] == [None, None]
    assert seen[2] == pytest.approx(np.mean(np.linspace(100.0, 110.0, 20)[:3]))


def test_indicator_for_the_named_symbol_follows_symbol_argument():
    seen = []

    class S(Strategy):
        def on_start(self, ctx):
            self.ema = ctx.indicators.ema(instrument="XYZ", period=2)

        def on_bar(self, ctx):
            seen.append(self.ema.ready)

    raptorbt.run_strategy_backtest(S(), symbol="XYZ", **_bars())
    assert seen[-1] is True
