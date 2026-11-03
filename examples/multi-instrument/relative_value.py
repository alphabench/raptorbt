"""Two instruments read together in one decision: buy the cheap leg of a ratio.

Shows the equal-timestamp group rule. Bars of different instruments that share
a timestamp are one group: every bar of the group is applied before any
``on_bar`` runs, so when the strategy decides on instrument A it already sees
B's bar of that same instant, and the result does not depend on the order the
data mapping lists the symbols in. The example proves it by running the mapping
both ways and comparing the trades.

The strategy trades A (cash, long only) when the A/B ratio is more than one
standard deviation below its 40-bar mean, and closes when the ratio is back at
the mean.

Skills: raptorbt-strategy, raptorbt-market-data, raptorbt-portfolio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Strategy, run_strategies  # noqa: E402

WINDOW = 40


class RatioReversion(Strategy):
    """Long A when A is cheap against B; decided on A's bar, reading B's."""

    def on_bar(self, ctx):
        if ctx.symbol != "A":
            return
        a = ctx.series("A")["close"]
        b = ctx.series("B")["close"]  # includes B's bar stamped at this same instant
        if len(a) < WINDOW or len(a) != len(b):
            return
        ratio = a[-WINDOW:] / b[-WINDOW:]
        z = (ratio[-1] - ratio.mean()) / ratio.std()
        if ctx.is_flat and z < -1.0:
            self.enter_long(size_frac=0.5)
        elif not ctx.is_flat and z >= 0.0:
            self.close_position()


def run(data):
    (result,) = run_strategies(
        [(RatioReversion(), ["A", "B"])], data,
        BacktestConfig(initial_capital=100_000.0, fees=0.0003),
        bar_stamp="close",
    )
    return result


def main():
    # B drifts with A plus noise, so the ratio wanders around a level.
    a = _data.bars(days=2, seed=11, start=100.0, step=0.10)
    b_close = a["close"] * (1.0 + 0.02 * np.sin(np.arange(len(a["close"])) / 25.0))
    b = {"timestamps": a["timestamps"], **_data.ohlcv(b_close, seed=12, wick=0.05)}

    forward = run({"A": a, "B": b})
    reverse = run({"B": b, "A": a})  # the same data, listed the other way round

    trades = [(t.symbol, t.entry_idx, t.exit_idx, round(t.pnl, 2)) for t in forward.result.trades()]
    assert trades == [(t.symbol, t.entry_idx, t.exit_idx, round(t.pnl, 2)) for t in reverse.result.trades()]
    assert trades, "the strategy never traded"
    m = forward.result.metrics
    print(f"trades {m.total_trades}  return {m.total_return_pct:.2f}%  same result either symbol order: yes")
    for row in trades[:3]:
        print("  ", row)


if __name__ == "__main__":
    main()
