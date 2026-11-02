"""research.sweep and row_from on real runs; determinism of the statistics."""

import numpy as np

import raptorbt as r
from raptorbt import research

DAY = 86_400_000_000_000
TS = np.array([1_767_225_600_000_000_000 + i * DAY for i in range(300)], dtype=np.int64)
CLOSE = 100.0 * np.cumprod(1.0 + np.random.default_rng(1).normal(0.0003, 0.01, 300))


def run(params, window):
    n = len(TS)
    entries = np.zeros(n, dtype=bool)
    entries[:: params["every"]] = True
    exits = np.zeros(n, dtype=bool)
    exits[params["hold"] :: params["every"]] = True
    cfg = r.BacktestConfig(initial_capital=1_000_000.0, periods_per_year=252.0)
    res = r.run_single_backtest(TS, CLOSE, CLOSE, CLOSE, CLOSE, np.ones(n), entries, exits, config=cfg)
    return research.row_from(res)


def test_sweep_rows_do_not_depend_on_the_worker_count():
    design = research.Design.grid({"every": [10, 20, 30], "hold": [3, 6]})
    wins = research.WalkForward.single([str(i) for i in range(10)])
    one = research.sweep(run, design, wins, workers=1)
    four = research.sweep(run, design, wins, workers=4, executor="thread")
    assert one.rows == four.rows
    assert one.trials == 6 and one.deflated["trials"] == 6
    assert 0.0 <= one.deflated["dsr"] <= 1.0
    row = one.rows[0]
    assert set(row) >= {"sharpe", "sortino", "total_return", "max_drawdown", "fills", "fees", "samples", "skew", "kurtosis"}
    assert row["samples"] == 300


def test_bootstrap_and_designs_are_a_function_of_the_seed():
    rs = list(np.random.default_rng(2).normal(0, 0.01, 200))
    a = research.stationary_bootstrap(rs, draws=300, seed=5)
    assert a == research.stationary_bootstrap(rs, draws=300, seed=5)
    assert a != research.stationary_bootstrap(rs, draws=300, seed=6)
    space = {"a": (0.0, 1.0), "b": [1, 2, 3]}
    assert research.Design.random(space, 20, seed=9).points == research.Design.random(space, 20, seed=9).points
