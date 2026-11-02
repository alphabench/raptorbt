"""Sharpe and Sortino are None exactly when undefined, in the metrics and the report."""

import numpy as np
import pytest

import raptorbt

N = 60
TS = (np.arange(N, dtype=np.int64) + 1) * 86_400_000_000_000


def _run(close, entries_at=(), exits_at=(), fees=0.0, n=N):
    c = np.asarray(close, dtype=np.float64)
    entries = np.zeros(n, bool)
    exits = np.zeros(n, bool)
    entries[list(entries_at)] = True
    exits[list(exits_at)] = True
    cfg = raptorbt.BacktestConfig(fees=fees)
    r = raptorbt.run_single_backtest(
        TS[:n], c, c + 0.5, c - 0.5, c, np.ones(n), entries, exits, config=cfg, symbol="X"
    )
    return r, r.full_report(TS[:n])["report"]


def test_no_trades_flat_curve():
    r, rep = _run(np.full(N, 100.0))
    assert r.metrics.sharpe_ratio is None and r.metrics.sortino_ratio is None
    assert rep["sharpe"] is None and rep["sortino"] is None


def test_trades_on_a_flat_curve_are_undefined_too():
    # A round trip at an unchanged price with no fees leaves equity flat.
    r, rep = _run(np.full(N, 100.0), [2], [10])
    assert r.metrics.total_trades == 1
    assert r.metrics.sharpe_ratio is None and rep["sharpe"] is None
    assert r.metrics.sortino_ratio is None and rep["sortino"] is None


def test_one_return_is_undefined():
    r, rep = _run([100.0, 101.0], n=2)
    assert r.metrics.sharpe_ratio is None and rep["sharpe"] is None


def test_a_normal_run_is_unchanged():
    close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0.0, 1.0, N))
    r, rep = _run(close, [2, 30], [20, 50])
    assert r.metrics.sharpe_ratio is not None and rep["sharpe"] is not None
    assert r.metrics.sharpe_ratio == pytest.approx(rep["sharpe"], rel=5e-2)
    assert r.metrics.sortino_ratio is not None
