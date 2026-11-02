"""A limit order that is marketable fills at the market, never at its own limit."""

import numpy as np
import pytest

from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders


def _bars(opens, highs, lows, closes):
    n = len(closes)
    return {
        "timestamps": np.arange(n, dtype=np.int64),
        "open": np.asarray(opens, dtype=np.float64),
        "high": np.asarray(highs, dtype=np.float64),
        "low": np.asarray(lows, dtype=np.float64),
        "close": np.asarray(closes, dtype=np.float64),
        "volume": np.full(n, 1_000.0),
    }


def _config(timing):
    return BacktestConfig(fees=0.0, fill_timing=timing)


def _run(order, data, timing="next_bar_open"):
    class S(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.submit_order(order)

    return run_strategy_backtest(S(), **data, config=_config(timing)).trades()


@pytest.mark.parametrize("timing", ["next_bar_open", "same_bar_close"])
def test_marketable_buy_limit_fills_at_the_open(timing):
    data = _bars(
        [100.0, 100.6, 100.8, 101.0],
        [101.0, 101.5, 101.5, 101.5],
        [99.5, 100.2, 100.2, 100.5],
        [100.5, 100.9, 101.0, 101.0],
    )
    trades = _run(orders.Limit(side="buy", price=200.0, units=10.0), data, timing)
    assert trades[0].entry_price == pytest.approx(100.6)


@pytest.mark.parametrize("timing", ["next_bar_open", "same_bar_close"])
def test_marketable_sell_limit_fills_at_the_open(timing):
    data = _bars(
        [102.0, 102.0, 102.0, 102.0],
        [103.0, 103.0, 103.0, 103.0],
        [101.0, 101.0, 101.0, 101.0],
        [102.0, 102.0, 102.0, 102.0],
    )
    trades = _run(orders.Limit(side="sell", price=1.0, units=10.0), data, timing)
    assert trades[0].entry_price == pytest.approx(102.0)


def test_resting_limit_touched_inside_the_bar_fills_at_the_limit():
    data = _bars(
        [100.0, 100.0, 99.0, 99.0],
        [101.0, 101.0, 100.0, 100.0],
        [99.5, 99.5, 96.5, 98.0],
        [100.0, 100.0, 98.0, 99.0],
    )
    trades = _run(orders.Limit(side="buy", price=97.0, units=10.0), data)
    assert trades[0].entry_price == pytest.approx(97.0)


def test_resting_limit_gapped_through_fills_at_the_open():
    data = _bars(
        [100.0, 100.0, 95.0, 95.0],
        [101.0, 101.0, 96.0, 96.0],
        [99.5, 99.5, 94.0, 94.0],
        [100.0, 100.0, 95.0, 95.0],
    )
    trades = _run(orders.Limit(side="buy", price=97.0, units=10.0), data)
    assert trades[0].entry_price == pytest.approx(95.0)
