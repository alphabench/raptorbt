"""One expression, two places: explored in an analysis session, then run inside a strategy.

The same text, ``zscore(close, 20) > 1 and adx(14) > 20``, is evaluated over
the bars in a ``Session`` and registered on the strategy with ``ctx.indicators.expr``. The
strategy buys when it is true and closes when it is false. The values the strategy saw on every
bar are compared with the analysis series byte for byte: what was explored is what runs, with no
re-implementation in between. The backtest is then run with ``run_strategies`` and
``bar_stamp="close"``.

Skills: raptorbt-analysis, raptorbt-expressions, raptorbt-strategy, raptorbt-backtesting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import _data  # noqa: E402
from raptorbt import BacktestConfig, Strategy, run_strategies  # noqa: E402
from raptorbt.analysis import Session  # noqa: E402

ENTRY = "zscore(close, 20) > 1 and adx(14) > 20"
SYMBOL = "DEMO"


class Breakout(Strategy):
    def on_start(self, ctx):
        self.signal = ctx.indicators.expr(ENTRY, instrument=SYMBOL)
        self.seen = []

    def on_bar(self, ctx):
        value = self.signal.value  # True, False, or None while unknown
        self.seen.append(np.nan if value is None else float(value))
        if value is True and ctx.is_flat:
            self.enter(size_frac=0.5)
        elif value is False and not ctx.is_flat:
            self.close_position()


def canonical_bytes(a: np.ndarray) -> bytes:
    """The bytes of a series with one NaN pattern, so equal numbers compare equal."""
    return np.where(np.isnan(a), np.nan, a).astype("<f8").tobytes()


def main():
    bars = _data.bars(days=3, seed=21, start=100.0, step=0.12)

    # explore: the same text over the same bars
    session = Session()
    session.load(bars, bar_stamp="close", name=SYMBOL)
    summary = session.eval(ENTRY)
    print(f"analysis: {summary['handle']}  warm-up {summary['warmup']}  true {summary['true']} of {summary['known']}")

    # run: the strategy registers the identical text
    strategy = Breakout()
    config = BacktestConfig(initial_capital=100_000.0, fees=0.0003, fill_timing="next_bar_open")
    result = run_strategies([strategy], {SYMBOL: bars}, config, bar_stamp="close")[0]
    m = result.metrics
    print(f"strategy: trades {m.total_trades}  return {m.total_return_pct:.2f}%  max drawdown {m.max_drawdown_pct:.2f}%")

    # the core promise: the strategy saw exactly the analysis series
    seen = np.array(strategy.seen)
    assert canonical_bytes(seen) == canonical_bytes(session.series(summary["handle"]))
    assert m.total_trades > 0
    print("byte-identical on all", len(seen), "bars")


if __name__ == "__main__":
    main()
