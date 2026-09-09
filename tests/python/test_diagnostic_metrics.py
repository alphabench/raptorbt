"""Diagnostics a reader can act on, and the conventions they promise.

These metrics exist to answer questions a bare Sharpe cannot: is the edge real
or is it a short-vol illusion, do costs eat it, is the exit giving the move
back, where can a stop sit. That only works if the numbers mean exactly what
they say, so the conventions are pinned here rather than left to the reader:

* skew and excess kurtosis must equal ``scipy.stats`` with ``bias=False``,
  because a caller comparing against a remembered threshold of 3.0 rather than
  0.0 is off by exactly 3.0 and nothing in the number says so;
* ``None`` always means *not measured*, never a measured zero;
* ``mfe_capture_ratio`` is a ratio of sums over winners, gross of costs -- the
  naive form (mean of per-trade ratios, all trades) returns negative nonsense.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig

scipy_stats = pytest.importorskip(
    "scipy.stats", reason="conventions are pinned against scipy"
)


def _series(n=800, seed=7, drift=0.0004, vol=0.013):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    ts = np.arange(n, dtype=np.int64) * 86_400_000_000_000
    return ts, close


def _sma(a, w):
    cs = np.cumsum(np.insert(a, 0, 0.0))
    out = np.full(len(a), np.nan)
    out[w - 1 :] = (cs[w:] - cs[:-w]) / w
    return out


def _run(n=800, seed=7, drift=0.0004, fees=0.0005):
    ts, c = _series(n, seed, drift)
    f, s = _sma(c, 10), _sma(c, 40)
    up = (f > s) & ~np.isnan(f) & ~np.isnan(s)
    e = np.zeros(len(c), bool)
    x = np.zeros(len(c), bool)
    e[1:] = up[1:] & ~up[:-1]
    x[1:] = ~up[1:] & up[:-1]
    return raptorbt.run_single_backtest(
        ts,
        c * 0.999,
        c * 1.003,
        c * 0.997,
        c,
        np.ones(len(c)),
        e,
        x,
        direction=1,
        symbol="B",
        config=BacktestConfig(initial_capital=100_000.0, fees=fees),
    )


def _closed(result):
    return [t for t in result.trades() if t.exit_reason != "EndOfData"]


# ---------------------------------------------------------------------------
# Return shape: the conventions, against the library callers will compare to
# ---------------------------------------------------------------------------


def test_skew_matches_scipy_bias_corrected():
    res = _run()
    returns = np.asarray(res.returns())
    assert res.metrics.return_skew == pytest.approx(
        scipy_stats.skew(returns, bias=False), abs=1e-9
    )


def test_kurtosis_is_excess_not_raw():
    """Gaussian reads 0.0, not 3.0. The two differ by exactly 3.0."""
    res = _run()
    returns = np.asarray(res.returns())
    excess = scipy_stats.kurtosis(returns, fisher=True, bias=False)
    assert res.metrics.return_kurtosis == pytest.approx(excess, abs=1e-9)
    raw = scipy_stats.kurtosis(returns, fisher=False, bias=False)
    assert res.metrics.return_kurtosis != pytest.approx(raw, abs=1e-6)


def test_tail_ratio_uses_linear_interpolation_percentiles():
    res = _run()
    returns = np.asarray(res.returns())
    expected = abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))
    assert res.metrics.tail_ratio == pytest.approx(expected, abs=1e-9)


def test_return_shape_is_none_on_a_series_too_short_to_describe():
    """Three samples for skew, four for kurtosis -- the corrections need them."""
    ts = np.arange(3, dtype=np.int64) * 86_400_000_000_000
    c = np.array([100.0, 101.0, 102.0])
    e = np.zeros(3, bool)
    x = np.zeros(3, bool)
    res = raptorbt.run_single_backtest(
        ts, c, c, c, c, np.ones(3), e, x, direction=1, symbol="B"
    )
    assert res.metrics.return_kurtosis is None
    assert res.metrics.tail_ratio is None  # needs 20 samples


def test_consistency_counts_moving_bars_only():
    """A flat bar is not a losing bar; a mostly-flat run must not read as 0%."""
    res = _run()
    returns = np.asarray(res.returns())
    moving = returns[returns != 0.0]
    assert res.metrics.return_consistency_pct == pytest.approx(
        (moving > 0).sum() / len(moving) * 100.0, abs=1e-9
    )


# ---------------------------------------------------------------------------
# Cost pressure
# ---------------------------------------------------------------------------


def test_cost_ratio_divides_by_gross_profit_not_net():
    """`pnl` is already net of that trade's fees, so gross adds them back."""
    res = _run()
    gross_wins = sum(t.pnl + t.fees for t in _closed(res) if t.pnl > 0.0)
    assert res.metrics.cost_to_gross_profit_pct == pytest.approx(
        res.metrics.total_fees_paid / gross_wins * 100.0, abs=1e-9
    )


def test_breakeven_multiple_is_none_when_there_was_no_profit_to_protect():
    """A negative 'headroom' figure reads like headroom. Refuse it instead."""
    losing = _run(drift=0.0004)
    assert losing.metrics.total_return_pct < 0
    assert losing.metrics.breakeven_cost_multiple is None


def test_breakeven_multiple_says_how_far_costs_can_rise():
    res = _run(drift=0.003)
    m = res.metrics
    assert m.total_return_pct > 0
    assert m.breakeven_cost_multiple == pytest.approx(
        (m.end_value - m.start_value) / m.total_fees_paid, abs=1e-9
    )


# ---------------------------------------------------------------------------
# Drawdown texture
# ---------------------------------------------------------------------------


def test_avg_drawdown_is_conditional_on_being_underwater():
    res = _run()
    dd = np.asarray(res.drawdown_curve())
    underwater = dd[dd > 0.0]
    assert res.metrics.avg_drawdown_pct == pytest.approx(underwater.mean(), abs=1e-9)
    # Conditional, so it sits above the unconditional mean and never exceeds
    # the worst point.
    assert res.metrics.avg_drawdown_pct >= dd.mean()
    assert res.metrics.avg_drawdown_pct <= res.metrics.max_drawdown_pct


def test_avg_drawdown_is_none_when_the_curve_never_fell():
    n = 60
    ts = np.arange(n, dtype=np.int64) * 86_400_000_000_000
    c = np.linspace(100.0, 140.0, n)
    e = np.zeros(n, bool)
    x = np.zeros(n, bool)
    res = raptorbt.run_single_backtest(
        ts, c, c, c, c, np.ones(n), e, x, direction=1, symbol="B"
    )
    assert res.metrics.max_drawdown_pct == 0.0
    assert res.metrics.avg_drawdown_pct is None


# ---------------------------------------------------------------------------
# Excursions -- the part 0.13.2 measured per trade and never summarised
# ---------------------------------------------------------------------------


def test_coverage_is_full_on_the_single_path():
    res = _run()
    assert res.metrics.mae_mfe_coverage_pct == pytest.approx(100.0)


def test_avg_mae_is_never_positive_and_averages_measured_trades():
    res = _run()
    mae = [t.mae_pnl for t in _closed(res) if t.mae_pnl is not None]
    assert res.metrics.avg_mae_pnl == pytest.approx(float(np.mean(mae)), abs=1e-9)
    assert res.metrics.avg_mae_pnl <= 0.0


def test_capture_is_a_ratio_of_sums_over_winners_and_gross_of_costs():
    """The three restrictions, each of which changes the number."""
    res = _run()
    closed = _closed(res)
    winners = [t for t in closed if t.pnl > 0.0 and t.mfe_pnl is not None]
    expected = sum(t.pnl + t.fees for t in winners) / sum(t.mfe_pnl for t in winners)
    assert res.metrics.mfe_capture_ratio == pytest.approx(expected, abs=1e-9)
    assert 0.0 <= res.metrics.mfe_capture_ratio <= 1.0

    # The naive form -- mean of per-trade ratios over ALL closed trades --
    # returns a negative number, because a loser puts negative P&L over a
    # positive excursion. That is why winners-only is the definition.
    naive = float(
        np.mean(
            [
                (t.pnl + t.fees) / t.mfe_pnl
                for t in closed
                if t.mfe_pnl is not None and t.mfe_pnl > 0.0
            ]
        )
    )
    assert naive < 0.0
    assert res.metrics.mfe_capture_ratio > 0.0


def test_excursion_aggregates_are_none_without_closed_trades():
    n = 60
    ts = np.arange(n, dtype=np.int64) * 86_400_000_000_000
    c = np.linspace(100.0, 120.0, n)
    e = np.zeros(n, bool)
    x = np.zeros(n, bool)
    res = raptorbt.run_single_backtest(
        ts, c, c, c, c, np.ones(n), e, x, direction=1, symbol="B"
    )
    assert res.trades() == []
    assert res.metrics.mae_mfe_coverage_pct is None
    assert res.metrics.avg_mae_pnl is None
    assert res.metrics.mfe_capture_ratio is None


def test_spread_path_reports_no_excursions_and_carries_a_known_defect():
    """Two different things, and only the first is by design.

    MAE/MFE are per-position, so a synthesized spread leg genuinely has none.
    But this path also drops figures it demonstrably could compute -- see the
    inline notes below. Pinned so the fix has a baseline.
    """
    n = 400
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000
    rng = np.random.default_rng(3)
    und = 1000 * np.exp(np.cumsum(rng.normal(0, 0.0009, n)))
    prem = [
        np.abs(30 + 8 * np.sin(np.arange(n) / 97) + rng.normal(0, 0.6, n)),
        np.abs(18 + 5 * np.cos(np.arange(n) / 113) + rng.normal(0, 0.5, n)),
    ]
    e = np.zeros(n, bool)
    x = np.zeros(n, bool)
    e[np.arange(20, n, 150)] = True
    x[np.arange(90, n, 150)] = True
    item = raptorbt.BatchSpreadItem(
        "s", prem, [("CE", 1000.0, -1, 75), ("CE", 1050.0, 1, 75)], e, x, "custom"
    )
    (_, res), = raptorbt.batch_spread_backtest(ts, und, [item])

    assert res.trades(), "fixture must produce trades to be meaningful"
    assert all(t.mae_pnl is None for t in res.trades())

    # Excursions are genuinely absent here: the legs are synthesized rather
    # than closed from a tracked position, so there is nothing to average.
    assert res.metrics.avg_mae_pnl is None
    assert res.metrics.mfe_capture_ratio is None

    # The rest documents a KNOWN DEFECT rather than a property worth having.
    # This path builds its metrics through StreamingMetrics::finalize, which
    # receives only `returns` -- while its caller holds a real drawdown curve
    # and a real trade list and puts both in the result it returns. So the
    # result below contradicts itself, and these assertions pin the current
    # behaviour so a fix has a baseline, not because it is correct.
    dd = np.asarray(res.drawdown_curve())
    assert (dd > 0).sum() > 0, "curve really is underwater"
    assert res.metrics.time_under_water_pct == 0.0  # ...yet reports 0.0
    assert res.metrics.total_turnover == 0.0  # ...with trades in hand
    assert res.metrics.avg_drawdown_pct is None
    assert res.metrics.cost_to_gross_profit_pct is None
    assert res.metrics.mae_mfe_coverage_pct is None

    # What this path does see -- the return series -- is computed for real.
    assert res.metrics.return_skew is not None
    assert res.metrics.return_kurtosis is not None
