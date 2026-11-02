"""Cash accumulation over many fills carries no drift at all.

Cash is booked in integer minor units, so the final balance is the initial
capital plus the sum of the trades' booked P&L, exactly: integer equality, not
a tolerance. This replays 100k round trips and also checks the float view
against ``math.fsum`` of the trade list.
"""

import math

import numpy as np

import raptorbt
from raptorbt import BacktestConfig

CAPITAL = 1_000_000.0


def test_final_equity_matches_an_exact_recomputation_from_the_trades():
    n = 200_000  # 100k round trips
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.00001, 0.0005, n)))
    entries = np.zeros(n, bool)
    exits = np.zeros(n, bool)
    entries[0::2] = True
    exits[1::2] = True
    cfg = BacktestConfig(
        initial_capital=CAPITAL, fees=1e-7, slippage=0.0, upon_bar_close=True
    )
    result = raptorbt.run_single_backtest(
        np.arange(n, dtype=np.int64) * 60_000_000_000,
        close, close + 0.1, close - 0.1, close, np.full(n, 1e3),
        entries, exits, direction=1, config=cfg,
    )
    trades = result.trades()
    assert len(trades) == n // 2
    exact = CAPITAL + math.fsum(t.pnl for t in trades)
    final = float(result.equity_curve()[-1])
    assert abs(final - exact) / exact < 1e-12
    # The books: initial capital plus the integer sum of booked P&L.
    booked = round(CAPITAL * 100) + sum(t.pnl_minor for t in trades)
    assert round(final * 100) == booked
