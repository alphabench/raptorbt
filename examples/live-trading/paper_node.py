"""A ``LiveNode`` on paper reproduces the batch backtest, and survives a restart.

Shows the three things a strategy goes through on its way to live trading:

1. **Batch.** ``run_strategies`` over the whole tape: the reference result.
2. **Paper.** The same strategy class on a ``LiveNode`` with ``PaperExecution()``,
   fed one timestamp at a time with ``push_group`` (every instrument's bar for that
   instant). Nothing about the strategy changes. The paper venue follows the node's
   own configuration, so the trades and the equity curve equal the batch run's.
3. **Checkpoint and restore.** Half way through, the node takes a ``checkpoint()``
   from its journal and is thrown away, as in a crash. ``LiveNode.restore`` rebuilds
   it by replaying the journal (every record it makes must equal the stored one),
   and the rest of the day runs on the rebuilt node to the same result.

Bars are close-stamped, so ``bar_stamp="close"`` is passed everywhere.

Skills: raptorbt-live-trading, raptorbt-execution, raptorbt-deployment,
raptorbt-strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Indicator, LiveNode, Strategy, run_strategies  # noqa: E402
from raptorbt import journal as rj  # noqa: E402
from raptorbt.execution import PaperExecution  # noqa: E402

SYMBOLS = ["AAA", "BBB"]
OHLCV = ("open", "high", "low", "close", "volume")
CONFIG = BacktestConfig(initial_capital=200_000.0, fees=0.0003)


class Trend(Strategy):
    """Long each symbol while its fast SMA is above its slow SMA."""

    def on_start(self, ctx):
        self.fast = {s: self.register_indicator(Indicator.sma(5), symbol=s) for s in SYMBOLS}
        self.slow = {s: self.register_indicator(Indicator.sma(20), symbol=s) for s in SYMBOLS}

    def on_bar(self, ctx):
        if not self.indicators_initialized():
            return
        up = self.fast[ctx.symbol].value > self.slow[ctx.symbol].value
        if ctx.is_flat and up:
            self.enter_long(size_frac=0.4)
        elif not ctx.is_flat and not up:
            self.close_position()


def tape():
    return {s: _data.bars(days=1, seed=40 + i, start=100.0 + 50 * i, step=0.25) for i, s in enumerate(SYMBOLS)}


def feed(node, data, rows):
    """Push each timestamp of ``rows`` as one group, the way a live feed delivers a bar of every symbol."""
    for i in rows:
        ts = int(data[SYMBOLS[0]]["timestamps"][i])
        node.push_group(ts, {s: tuple(float(data[s][c][i]) for c in OHLCV) for s in SYMBOLS})


def summary(result):
    trades = [(t.symbol, t.entry_idx, round(t.entry_price, 6), round(t.exit_price, 6), t.size, round(t.pnl, 6))
              for t in result.result.trades()]
    return trades, result.result.equity_curve()


def new_node(data, **kwargs):
    return LiveNode([Trend()], SYMBOLS, CONFIG, bar_stamp="close", **kwargs)


def main():
    data = tape()
    n = len(data[SYMBOLS[0]]["timestamps"])

    batch = run_strategies([Trend()], data, CONFIG, bar_stamp="close")[0]
    want_trades, want_equity = summary(batch)
    assert want_trades, "the strategy never traded"

    # Paper: the same strategy, one timestamp at a time.
    node = new_node(data, execution=PaperExecution())
    feed(node, data, range(n))
    paper_trades, paper_equity = summary(node.finish()[0])
    assert paper_trades == want_trades and np.array_equal(paper_equity, want_equity)
    print(f"batch and paper agree: {len(want_trades)} trades, final equity {want_equity[-1]:,.2f}")

    # Checkpoint and restore half way through.
    half = n // 2
    header = {"run": "example-1", "algo": "trend"}
    node = new_node(data, execution=PaperExecution(), journal=rj.Journal(header, memory=True))
    feed(node, data, range(half))
    blob = node.checkpoint()
    digest = node.state_digest()
    del node  # the process is gone; only the checkpoint survives

    restored = LiveNode.restore(
        blob, [Trend()], SYMBOLS, CONFIG, execution=PaperExecution(), bar_stamp="close"
    )
    assert restored.state == "live" and restored.state_digest() == digest
    feed(restored, data, range(half, n))
    restored_trades, restored_equity = summary(restored.finish()[0])
    assert restored_trades == want_trades and np.array_equal(restored_equity, want_equity)
    print(f"restored at bar {half} of {n} from a {len(blob)}-byte checkpoint: same trades, same equity")


if __name__ == "__main__":
    main()
