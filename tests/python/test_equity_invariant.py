"""Equity must equal initial capital plus PnL, for every direction and account."""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Strategy

CLOSES = np.array([100.0, 100.0, 95.0, 90.0, 104.0, 104.0])


def _bars():
    n = len(CLOSES)
    return {
        "timestamps": np.arange(n, dtype=np.int64),
        "open": CLOSES.copy(),
        "high": CLOSES + 0.5,
        "low": CLOSES - 0.5,
        "close": CLOSES.copy(),
        "volume": np.full(n, 1e3),
    }


class _RoundTrip(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.enter()
        if ctx.idx == 4 and ctx.position is not None:
            self.close_position()


def _run(direction, account_type, fees):
    cfg = BacktestConfig(
        initial_capital=100_000.0, fees=fees, slippage=0.0, upon_bar_close=True
    )
    return raptorbt.run_strategy_backtest(
        _RoundTrip(), **_bars(), direction=direction, config=cfg, account_type=account_type
    )


@pytest.mark.parametrize("account_type", ["cash", "margin"])
@pytest.mark.parametrize("direction", [1, -1])
def test_equity_is_marked_to_market_every_bar(direction, account_type):
    result = _run(direction, account_type, fees=0.0)
    (trade,) = result.trades()
    equity = result.equity_curve()
    entry = trade.entry_price
    for i in range(trade.entry_idx, trade.exit_idx):
        expected = 100_000.0 + (CLOSES[i] - entry) * direction * trade.size
        assert equity[i] == pytest.approx(expected), (i, equity)
    assert equity[-1] == pytest.approx(100_000.0 + trade.pnl)


@pytest.mark.parametrize("account_type", ["cash", "margin"])
@pytest.mark.parametrize("direction", [1, -1])
def test_final_equity_is_capital_plus_pnl_net_of_fees(direction, account_type):
    result = _run(direction, account_type, fees=0.001)
    trades = result.trades()
    assert trades and all(t.fees > 0 for t in trades)
    assert result.equity_curve()[-1] == pytest.approx(
        100_000.0 + sum(t.pnl for t in trades)
    )


def _pairs(exit_at):
    n = 6
    ts = np.arange(n, dtype=np.int64)
    p1 = np.array([100.0, 100.0, 104.0, 98.0, 107.0, 107.0])
    p2 = np.array([50.0, 50.0, 49.0, 52.0, 51.0, 51.0])
    v = np.full(n, 1e3)
    ent = np.zeros(n, bool)
    ent[1] = True
    ex = np.zeros(n, bool)
    if exit_at is not None:
        ex[exit_at] = True
    cfg = BacktestConfig(
        initial_capital=100_000.0, fees=0.001, slippage=0.0, upon_bar_close=True
    )
    return raptorbt.run_pairs_backtest(
        ts, p1, p1 + 1, p1 - 1, p1, v, ts, p2, p2 + 1, p2 - 1, p2, v, ent, ex,
        direction=1, config=cfg, hedge_ratio=1.0,
    )


@pytest.mark.parametrize("exit_at", [4, None])
def test_pairs_final_equity_is_capital_plus_pnl_net_of_fees(exit_at):
    result = _pairs(exit_at)
    trades = result.trades()
    assert trades and all(t.entry_fees > 0 for t in trades)
    assert all(t.fees == pytest.approx(t.entry_fees + t.exit_fees) for t in trades)
    assert result.equity_curve()[-1] == pytest.approx(
        100_000.0 + sum(t.pnl for t in trades)
    )
