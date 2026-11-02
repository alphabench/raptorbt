"""``BacktestMetrics`` and ``full_report`` give one answer, and an undefined figure is absent."""

import numpy as np
import pytest

import raptorbt

N = 60
TS = (np.arange(N, dtype=np.int64) + 1) * 86_400_000_000_000


def _run(close, entries_at, exits_at, fees=0.0):
    c = np.asarray(close, dtype=np.float64)
    entries = np.zeros(N, bool)
    exits = np.zeros(N, bool)
    entries[list(entries_at)] = True
    exits[list(exits_at)] = True
    cfg = raptorbt.BacktestConfig(fees=fees)
    result = raptorbt.run_single_backtest(
        TS, c, c + 0.5, c - 0.5, c, np.ones(N), entries, exits, config=cfg, symbol="X"
    )
    return result, result.full_report(TS)["report"]


def test_a_run_that_never_traded_has_no_ratios():
    c = 100.0 + np.zeros(N)
    result, report = _run(c, [], [])
    m = result.metrics
    assert m.total_trades == 0
    assert m.profit_factor is None and m.sharpe_ratio is None and m.win_rate_pct is None
    assert report["profit_factor"] is None
    assert report["sharpe"] is None
    assert report["win_rate"] is None


def test_the_open_trade_at_the_end_is_not_a_closed_trade_in_either():
    # Rises into bar 10 (the closed trade wins), falls after bar 20 (the trade
    # still open at the end loses). One closed trade, and it won.
    close = np.concatenate(
        [np.linspace(100, 120, 11), np.linspace(120, 118, 10), np.linspace(118, 90, N - 21)]
    )
    result, report = _run(close, [2, 25], [10])
    m = result.metrics
    assert m.total_closed_trades == 1 and m.total_open_trades == 1
    assert m.win_rate_pct == pytest.approx(100.0)
    assert report["win_rate"] == pytest.approx(1.0)
    assert m.profit_factor is None  # no closed trade lost
    assert report["profit_factor"] is None
    assert report["trade_count"] == 1


def test_a_win_that_fees_turn_into_a_loss_is_a_loss_in_both():
    close = np.concatenate([np.linspace(100, 100.4, 11), np.full(N - 11, 100.4)])
    result, report = _run(close, [2], [10], fees=0.01)
    m = result.metrics
    assert m.losing_trades == 1 and m.winning_trades == 0
    assert m.win_rate_pct == pytest.approx(0.0)
    assert report["win_rate"] == pytest.approx(0.0)
    # No gross profit over a real loss is a profit factor of zero, in both.
    assert m.profit_factor == pytest.approx(0.0)
    assert report["profit_factor"] == pytest.approx(0.0)
