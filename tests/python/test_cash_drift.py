"""Cash accumulation drift over many fills stays far below any visible level.

The engine adds each fill's cash effect to a plain f64 balance. This replays
the trade list with ``math.fsum`` (exactly rounded) and compares the final
equity with it. Measured on this corpus: about 4e-14 relative at 100k trades
and 1e-14 at 500k, five orders under the 1e-9 bar that would justify
compensated summation. The 1e-12 bound below pins that with headroom; a
regression in the accumulation order or a lossy intermediate would break it.
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
