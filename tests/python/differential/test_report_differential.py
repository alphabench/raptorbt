"""The full report against the reference implementation, replayed from fixtures.

The fixture holds the numbers a reference implementation produced for these inputs. These tests replay them here and compare every key.

Floats agree to 1e-9 relative (the reference report summed with a correctly rounded sum; this one
takes its mean and standard deviation from ``StreamingMetrics``, a Welford pass). One figure
differs by definition, and the test states the difference instead of loosening a tolerance:

trade statistics
    The reference report classified a trade as a win or a loss on its P&L before fees and counted a
    position still open at the end of the data. This one counts closed trades only, on P&L
    after fees (the definition ``BacktestMetrics`` uses). ``expected_trade_figures`` computes
    those figures independently from the fixture's trades; they replace the reference values.

``sortino``
    This engine has one Sortino, ``StreamingMetrics``' (downside deviation over ``n - 1``
    returns, as ``BacktestMetrics.sortino_ratio``). The reference report divided by ``n``. The
    new value is the old one times ``sqrt((n - 1) / n)``.
"""

import json
import math
from pathlib import Path

import pytest

import raptorbt as r

HERE = Path(__file__).parent
CASES = json.loads((HERE / "report_fixture.json").read_text())
REL = 1e-9


def same(got, want, path, scale=None):
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        assert set(got) == set(want), (path, set(got) ^ set(want))
        for k, w in want.items():
            same(got[k], w, f"{path}.{k}", scale)
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want)):
            same(g, w, f"{path}[{i}]", scale)
    elif isinstance(want, bool) or want is None or isinstance(want, str):
        assert got == want, (path, got, want)
    elif isinstance(want, int) and not isinstance(want, bool):
        assert got == want, (path, got, want)
    else:
        assert math.isclose(got, want, rel_tol=REL, abs_tol=1e-9), (path, got, want)


TRADE_FIGURES = (
    "trade_count", "breakeven", "win_rate", "expectancy_gross", "expectancy_net",
    "avg_win", "avg_loss", "profit_factor", "payoff_ratio",
)


def expected_trade_figures(trades):
    """The trade statistics on closed trades and net P&L; ``None`` where undefined."""
    closed = [t for t in trades if t["exit_reason"].lower() != "endofdata"]
    net = [t["pnl"] for t in closed]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x < 0]
    n = len(closed)
    out = {"trade_count": n, "breakeven": n - len(wins) - len(losses)}
    out["win_rate"] = len(wins) / n if n else None
    out["expectancy_gross"] = sum(t["pnl"] + t["fees"] for t in closed) / n if n else None
    out["expectancy_net"] = sum(net) / n if n else None
    out["avg_win"] = sum(wins) / len(wins) if wins else None
    out["avg_loss"] = -sum(losses) / len(losses) if losses else None
    out["profit_factor"] = sum(wins) / -sum(losses) if losses else None
    out["payoff_ratio"] = (
        (sum(wins) / len(wins)) / (-sum(losses) / len(losses)) if wins and losses else None
    )
    return out


def replay(case):
    kw = dict(case["kwargs"])
    # The reference report assumed an NSE session in IST; the defaults are
    # generic now, so the case says so.
    kw.setdefault("session_minutes", 375.0)
    kw.setdefault("tz_offset_ns", r.IST_OFFSET_NS)
    if "period" in kw:
        kw["period"] = tuple(kw["period"])
    if "exposure" in kw:
        kw["exposure"] = [tuple(x) for x in kw["exposure"]]
    if "exposure_time" in kw:
        kw["exposure_time"] = tuple(kw["exposure_time"])
    if "benchmark" in kw:
        kw["benchmark"] = [tuple(x) for x in kw["benchmark"]]
    return r.full_report(
        case["ts"], case["equity"], case["trades"], case["orders"],
        attribution=case["attribution"], **kw,
    )


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_full_report_matches_the_old_engine(case):
    got, want = replay(case), case["result"]
    n = want["counters"]["returns"]
    expected = expected_trade_figures(case["trades"])
    for key in TRADE_FIGURES:
        figure = got["report"].pop(key, None)
        want["report"].pop(key, None)
        got["report_unavailable"].pop(key, None)
        want["report_unavailable"].pop(key, None)
        if expected[key] is None:
            assert figure is None, key
        else:
            assert math.isclose(figure, expected[key], rel_tol=REL, abs_tol=1e-9), (key, figure)
    got_sortino, want_sortino = got["report"].pop("sortino"), want["report"].pop("sortino")
    if want_sortino is None:
        assert got_sortino is None
    else:
        assert math.isclose(got_sortino, want_sortino * math.sqrt((n - 1) / n), rel_tol=REL)
    same(got, want, case["name"])


def test_the_report_and_backtest_metrics_read_the_same_ratios():
    """One definition: the report's Sharpe and drawdown come from the engine's own
    ``StreamingMetrics`` and ``DrawdownTracker``, the ones that fill ``BacktestMetrics``.

    The run's metrics average over its ``returns()``, which carry a leading zero for the first
    bar; the report averages over the curve's ``n - 1`` returns. The two agree to the weight of
    that one zero."""
    import numpy as np

    rng = np.random.default_rng(5)
    n = 400
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, n))
    ts = np.array([1_767_225_600_000_000_000 + i * 86_400_000_000_000 for i in range(n)], dtype=np.int64)
    entries = np.zeros(n, dtype=bool)
    entries[::20] = True
    exits = np.zeros(n, dtype=bool)
    exits[10::20] = True
    cfg = r.BacktestConfig(initial_capital=1_000_000.0, periods_per_year=252.0)
    res = r.run_single_backtest(ts, close, close, close, close, np.ones(n), entries, exits, config=cfg)
    rep = res.full_report(ts.tolist(), periods_per_year=252.0)["report"]
    m = res.metrics
    eq = res.equity_curve()
    rs = np.diff(eq) / eq[:-1]
    assert math.isclose(rep["sharpe"], rs.mean() / rs.std(ddof=1) * math.sqrt(252), rel_tol=1e-9)
    assert math.isclose(rep["sharpe"], m.sharpe_ratio, rel_tol=5e-3)
    assert math.isclose(rep["max_drawdown"], m.max_drawdown_pct / 100.0, rel_tol=1e-9)
    # the report's return is over the curve (first sample to last); the metrics' is over the
    # initial capital. They share the end of the curve.
    assert math.isclose(rep["end"], 1_000_000.0 * (1.0 + m.total_return_pct / 100.0), rel_tol=1e-9)
