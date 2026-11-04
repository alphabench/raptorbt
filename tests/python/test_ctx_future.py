"""Inside a bar hook, ``run_strategy_backtest``'s context cannot read the future.

``ctx.open/high/low/close/volume/timestamps`` held the whole run, so ``ctx.close[ctx.idx + 1]``
read tomorrow's close and "no lookahead" was a docstring. In a bar hook they now end at the
current bar (like ``ctx.data`` and the multi-symbol context); in ``on_start`` they are still the
whole run, for vectorised indicators.
"""
import numpy as np
import pytest

from raptorbt import BacktestConfig, Strategy, run_strategy_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
N = 12


def _bars():
    close = 100.0 + np.arange(N, dtype=np.float64)
    return dict(timestamps=T0 + np.arange(1, N + 1, dtype=np.int64) * MIN, open=close,
                high=close + 0.5, low=close - 0.5, close=close, volume=np.full(N, 1000.0))


class Peek(Strategy):
    def __init__(self):
        super().__init__()
        self.start_len = None
        self.lengths = []
        self.last = []
        self.peeked = []

    def on_start(self, ctx):
        self.start_len = len(ctx.close)
        self.precomputed = ctx.close * 2.0  # a vectorised indicator over the whole run

    def on_bar(self, ctx):
        self.lengths.append((len(ctx.close), len(ctx.timestamps), len(ctx.volume)))
        self.last.append((ctx.close[-1], ctx.close[ctx.idx], ctx.high[-1], ctx.open[-1], ctx.low[-1]))
        try:
            ctx.close[ctx.idx + 1]
        except IndexError:
            self.peeked.append("refused")
        else:
            self.peeked.append("read the future")
        assert self.precomputed[ctx.idx] == ctx.close[ctx.idx] * 2.0
        assert ctx.n_bars == N
        assert len(ctx.history(3)) == min(3, ctx.idx + 1)


def test_a_bar_hook_cannot_read_past_the_current_bar():
    s = Peek()
    run_strategy_backtest(s, **_bars(), symbol="X", bar_stamp="close", config=BacktestConfig(fees=0.0))
    assert s.start_len == N, "on_start still sees the whole run"
    assert s.lengths == [(i + 1, i + 1, i + 1) for i in range(N)]
    assert s.peeked == ["refused"] * N
    for i, (last_close, at_idx, last_high, last_open, last_low) in enumerate(s.last):
        assert last_close == at_idx == 100.0 + i, "ctx.close[-1] is the current bar"
        assert (last_high, last_open, last_low) == (100.5 + i, 100.0 + i, 99.5 + i)
