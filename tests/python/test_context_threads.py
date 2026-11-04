"""A run frees its contexts on the thread that made them, whatever its outcome.

The runtime's contexts are bound to their creating thread (PyO3 ``unsendable``). One a run leaves in
a reference cycle (context <-> strategy, or held by the frames of an exception that left the run)
is freed by whichever thread next runs the garbage collector, which fails with ``is being dropped on
another thread`` and fails a run that has nothing to do with it (a backtest worker runs jobs on
threads of one process)."""

import gc
import sys
import threading

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies

MIN = 60_000_000_000
T0 = 1_700_000_000_000_000_000


class Trades(Strategy):
    def on_bar(self, ctx):
        if ctx.is_flat:
            self.enter(size_frac=0.1)
        else:
            self.close_position()


class Raises(Strategy):
    def on_bar(self, ctx):
        raise ValueError("strategy failed")


class Stops(Strategy):
    def on_stop(self, ctx):
        raise ValueError("on_stop failed")


def _bars(n=60):
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n))
    return {
        "AAA": {
            "timestamps": np.arange(n, dtype=np.int64) * MIN + T0,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    }


def _run(strategies):
    try:
        run_strategies(
            [(s, ["AAA"]) for s in strategies],
            _bars(),
            BacktestConfig(initial_capital=100_000.0),
            bar_stamp="close",
        )
    except ValueError:
        pass  # the exception is the outcome under test; it leaves with its traceback


@pytest.mark.parametrize(
    "strategies",
    [(Trades, Trades), (Raises, Trades), (Trades, Stops)],
    ids=["ends", "a strategy raises", "on_stop raises"],
)
def test_a_run_leaves_no_context_for_another_threads_collector(strategies):
    freed = []
    previous = sys.unraisablehook
    sys.unraisablehook = lambda u: freed.append(str(u.exc_value))
    try:
        gc.collect()
        freed.clear()
        threads = [threading.Thread(target=_run, args=([c() for c in strategies],)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # nothing of the three runs is left for this thread to collect
        gc.collect()
    finally:
        sys.unraisablehook = previous
    assert freed == []


def _live(strategies):
    node = LiveNode([(s, ["AAA"]) for s in strategies], ["AAA"], BacktestConfig(initial_capital=100_000.0), bar_stamp="close")
    bars = _bars(20)["AAA"]
    for i in range(20):
        node.push_bar("AAA", int(bars["timestamps"][i]), *(float(bars[k][i]) for k in ("open", "high", "low", "close", "volume")))
    node.finish()


def test_a_live_node_leaves_no_context_for_another_threads_collector():
    freed = []
    previous = sys.unraisablehook
    sys.unraisablehook = lambda u: freed.append(str(u.exc_value))
    try:
        gc.collect()
        freed.clear()
        t = threading.Thread(target=_live, args=([Trades(), Trades()],))
        t.start()
        t.join()
        gc.collect()
    finally:
        sys.unraisablehook = previous
    assert freed == []
