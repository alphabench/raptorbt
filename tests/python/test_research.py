"""research.sweep and row_from on real runs; determinism of the statistics."""

import numpy as np
import pytest

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


def test_row_kurtosis_is_raw_not_excess():
    """`return_kurtosis` is excess (a normal series is 0); the deflated Sharpe takes raw (a normal
    series is 3), so a row carries excess + 3."""
    from types import SimpleNamespace

    def stub(kurt):
        m = SimpleNamespace(
            sharpe_ratio=1.0, sortino_ratio=1.0, total_return_pct=1.0, max_drawdown_pct=1.0,
            total_fees_paid=0.0, return_skew=0.0, return_kurtosis=kurt,
        )
        return SimpleNamespace(metrics=m, orders=lambda: [], returns=lambda: [0.0, 0.0])

    assert research.row_from(stub(0.0))["kurtosis"] == 3.0
    assert research.row_from(stub(1.5))["kurtosis"] == 4.5
    assert research.row_from(stub(None))["kurtosis"] == 3.0
    res = r.run_single_backtest(
        TS, CLOSE, CLOSE, CLOSE, CLOSE, np.ones(len(TS)),
        np.arange(len(TS)) % 10 == 0, np.arange(len(TS)) % 10 == 5,
        config=r.BacktestConfig(initial_capital=1_000_000.0, periods_per_year=252.0),
    )
    assert res.metrics.return_kurtosis is not None
    assert research.row_from(res)["kurtosis"] == res.metrics.return_kurtosis + 3.0


def test_deflated_sharpe_of_a_normal_series_uses_raw_kurtosis():
    """Hand check, sharpes [0, 0], best 0.2 sqrt(252), 101 samples, skew 0, raw kurtosis 3:
    the engine gives dsr 0.9286296147097797, z 1.465660608864923. Passing the excess value 0 as
    if it were raw gave 0.9315852308839763."""
    kw = dict(samples=101, skew=0.0, periods_per_year=252.0, luck="samples")
    best = 0.2 * 252.0**0.5
    right = research.deflated_sharpe([0.0, 0.0], best, kurtosis=3.0, **kw)
    assert right["dsr"] == pytest.approx(0.9286296147097797, abs=1e-12)
    assert right["z"] == pytest.approx(1.465660608864923, abs=1e-12)
    wrong = research.deflated_sharpe([0.0, 0.0], best, kurtosis=0.0, **kw)
    assert wrong["dsr"] == pytest.approx(0.9315852308839763, abs=1e-12)
