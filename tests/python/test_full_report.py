"""full_report on a real run's result."""

import math

import numpy as np
import pytest

import raptorbt as r

DAY = 86_400_000_000_000


def _run():
    n = 250
    ts = np.array([1_767_225_600_000_000_000 + i * DAY for i in range(n)], dtype=np.int64)
    close = 100.0 * np.cumprod(1.0 + np.random.default_rng(3).normal(0.0004, 0.01, n))
    entries = np.zeros(n, dtype=bool)
    entries[::25] = True
    exits = np.zeros(n, dtype=bool)
    exits[12::25] = True
    cfg = r.BacktestConfig(initial_capital=1_000_000.0, periods_per_year=252.0, fee_segment="NSE-EQ")
    res = r.run_single_backtest(ts, close, close, close, close, np.ones(n), entries, exits, symbol="AAA", config=cfg)
    return res, ts


def test_a_result_reports_its_money_trades_and_attribution():
    res, ts = _run()
    rep = res.full_report(ts.tolist(), periods_per_year=252.0)
    assert rep["report"]["trade_count"] == len(res.trades())
    assert math.isclose(rep["report"]["fees"], sum(t.fees for t in res.trades()), rel_tol=1e-9)
    assert rep["counters"]["samples"] == len(ts)
    assert "AAA" in rep["attribution"]["by_instrument"]
    assert len(rep["trades"]) == len(res.trades())
    # no rule counters: the report says so rather than inventing a split
    assert rep["attribution"]["by_rule"] is None
    assert rep["report_unavailable"]["attribution.by_rule"].startswith("NOT_RULE_BASED")


def test_a_timestamp_count_that_does_not_match_the_curve_is_refused():
    res, ts = _run()
    with pytest.raises(ValueError):
        res.full_report(ts[:-1].tolist())


def test_unavailable_figures_are_named_not_nan():
    rep = r.full_report([0, 1, 2], [100.0, 100.0, 100.0])
    assert rep["report"]["sharpe"] is None
    assert rep["report_unavailable"]["sharpe"].startswith("ZERO_VARIANCE")
    assert all(not (isinstance(v, float) and math.isnan(v)) for v in rep["report"].values())


def test_a_trade_row_marks_the_line_closed_at_the_end_of_the_data():
    # one closed trade (+40 net) and the line still open when the data ended (-12 net at the
    # last mark): the rows say which is which, and the trade statistics count the closed one
    trades = [
        {"symbol": "AAA", "entry_time": 0, "exit_time": 1, "pnl": 40.0, "fees": 2.0, "exit_reason": "Signal"},
        {"symbol": "AAA", "entry_time": 1, "exit_time": 2, "pnl": -12.0, "fees": 1.0, "exit_reason": "EndOfData"},
    ]
    rep = r.full_report([0, 1, 2], [1000.0, 1040.0, 1028.0], trades, [], attribution=True)
    assert [t["open_at_end"] for t in rep["trades"]] == [False, True]
    assert [t["net"] for t in rep["trades"]] == [40.0, -12.0]
    assert rep["report"]["trade_count"] == 1
