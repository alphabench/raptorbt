"""The total_turnover metric, exercised through the wheel.

Turnover is the value of all buying and selling added together — a round
trip is TWO legs, because that is the per-leg price*|size| base the fee
models charge on, so fees/turnover is a meaningful cost rate. Exit legs
that never actually traded (EndOfData: the run ends while holding;
Settlement: an option left to expire) contribute nothing.

These tests recompute the fold in Python from the reported trade list and
hold the metric to it, so a drift between the metric and the trades it
claims to summarize fails loudly at the boundary consumers actually use.
"""

import numpy as np

from raptorbt import Strategy, run_strategy_backtest

DAY_NS = 86_400_000_000_000

UNTRADED_EXITS = {"EndOfData", "Settlement"}


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


def _expected_turnover(trades):
    total = 0.0
    for t in trades:
        size = abs(t.size)
        total += abs(t.entry_price) * size
        if t.exit_reason not in UNTRADED_EXITS:
            total += abs(t.exit_price) * size
    return total


class Churn(Strategy):
    """Alternate entry and exit every bar."""

    def on_bar(self, ctx):
        if ctx.position is None:
            self.enter()
        else:
            self.close_position()


class HoldToEnd(Strategy):
    """Enter once, never exit: the run ends holding (EndOfData)."""

    def on_bar(self, ctx):
        if ctx.idx == 1 and ctx.position is None:
            self.enter()


def test_turnover_matches_the_trade_list():
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=300)))
    result = run_strategy_backtest(Churn, **_ohlcv(close))

    trades = result.trades()
    assert trades, "churn produced no trades"
    assert result.metrics.total_turnover > 0.0
    assert abs(
        result.metrics.total_turnover - _expected_turnover(trades)
    ) < 1e-6


def test_end_of_data_exit_counts_entry_leg_only():
    close = np.linspace(100.0, 120.0, 50)
    result = run_strategy_backtest(HoldToEnd, **_ohlcv(close))

    trades = result.trades()
    assert len(trades) == 1
    assert trades[0].exit_reason == "EndOfData"
    # Entry leg only: the phantom exit moved no money and paid no fee.
    expected = abs(trades[0].entry_price) * abs(trades[0].size)
    assert abs(result.metrics.total_turnover - expected) < 1e-6


def test_turnover_appears_in_to_dict():
    close = np.linspace(100.0, 110.0, 40)
    result = run_strategy_backtest(Churn, **_ohlcv(close))
    d = result.metrics.to_dict()
    assert d["Total Turnover"] == result.metrics.total_turnover
