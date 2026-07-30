"""Behavioral tests for the raptorbt Python API.

These pin the correctness fixes in 0.5.0 against the wheel, not against Rust
internals. Each test names the defect it guards, because the value of these is
catching a silent regression years from now.
"""

import json
import math

import numpy as np
import pytest

import raptorbt

DAY_NS = 86_400_000_000_000
MIN_NS = 60_000_000_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ohlcv(close, timestamps, spread=0.004):
    """Build a coherent OHLCV set so bar geometry is always valid."""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    openp = np.empty(n, dtype=np.float64)
    openp[0] = close[0]
    openp[1:] = close[:-1]
    hi = np.maximum(openp, close) * (1.0 + spread)
    lo = np.minimum(openp, close) * (1.0 - spread)
    vol = 1_000_000.0 + (np.arange(n, dtype=np.float64) % 97) * 1000.0
    return {
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "open": openp,
        "high": hi,
        "low": lo,
        "close": close,
        "volume": vol,
    }


@pytest.fixture
def daily():
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, size=500)))
    return _ohlcv(close, np.arange(500, dtype=np.int64) * DAY_NS)


@pytest.fixture
def intraday():
    """30 sessions of 375 one-minute NSE bars."""
    rng = np.random.default_rng(17)
    bars, days = 375, 30
    n = bars * days
    close = 1500.0 * np.exp(np.cumsum(rng.normal(0.00002, 0.0009, size=n)))
    ts = np.empty(n, dtype=np.int64)
    k = 0
    for d in range(days):
        start = d * DAY_NS + (9 * 60 + 15) * MIN_NS
        for b in range(bars):
            ts[k] = start + b * MIN_NS
            k += 1
    return _ohlcv(close, ts, spread=0.0006)


def sma_crossover(ohlcv, fast=10, slow=30):
    close = ohlcv["close"]
    n = len(close)

    def sma(x, w):
        out = np.full(n, np.nan)
        if n >= w:
            c = np.cumsum(np.insert(x, 0, 0.0))
            out[w - 1 :] = (c[w:] - c[:-w]) / w
        return out

    f, s = sma(close, fast), sma(close, slow)
    valid = ~(np.isnan(f) | np.isnan(s))
    above = np.zeros(n, dtype=bool)
    above[valid] = f[valid] > s[valid]
    prev = np.roll(above, 1)
    prev[0] = False
    return (above & ~prev & valid), (~above & prev & valid)


def run(ohlcv, entries, exits, **cfg_kwargs):
    cfg = raptorbt.PyBacktestConfig(**cfg_kwargs)
    return raptorbt.run_single_backtest(
        ohlcv["timestamps"],
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        ohlcv["volume"],
        entries,
        exits,
        symbol="TEST",
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Slippage was silently ignored before 0.5.0
# ---------------------------------------------------------------------------


def test_slippage_changes_results(daily):
    """config.slippage never reached the engine before 0.5.0."""
    entries, exits = sma_crossover(daily)
    base = run(daily, entries, exits, initial_capital=100_000.0, fees=0.001, slippage=0.0)
    slipped = run(daily, entries, exits, initial_capital=100_000.0, fees=0.001, slippage=0.002)

    assert list(base.equity_curve()) != list(slipped.equity_curve())
    assert slipped.metrics.total_return_pct < base.metrics.total_return_pct, (
        "slippage is a cost; it must reduce returns"
    )


def test_apply_slippage_false_restores_legacy(daily):
    entries, exits = sma_crossover(daily)
    zero = run(daily, entries, exits, initial_capital=100_000.0, fees=0.001, slippage=0.0)
    off = run(
        daily,
        entries,
        exits,
        initial_capital=100_000.0,
        fees=0.001,
        slippage=0.002,
        apply_slippage=False,
    )
    assert list(zero.equity_curve()) == list(off.equity_curve())


# ---------------------------------------------------------------------------
# Annualization
# ---------------------------------------------------------------------------


def test_daily_annualization_is_unchanged(daily):
    """Daily data resolves to 365, so daily Sharpe must not drift."""
    entries, exits = sma_crossover(daily)
    inferred = run(daily, entries, exits, initial_capital=100_000.0, fees=0.001)
    explicit = run(
        daily, entries, exits, initial_capital=100_000.0, fees=0.001, periods_per_year=365.0
    )
    assert inferred.metrics.sharpe_ratio == pytest.approx(explicit.metrics.sharpe_ratio)


def test_intraday_sharpe_uses_session_count(intraday):
    """1-minute bars annualized as daily understate Sharpe by ~sqrt(94500/365)."""
    entries, exits = sma_crossover(intraday)
    correct = run(intraday, entries, exits, initial_capital=500_000.0, fees=0.0003)
    as_daily = run(
        intraday, entries, exits, initial_capital=500_000.0, fees=0.0003, periods_per_year=365.0
    )
    ratio = abs(correct.metrics.sharpe_ratio / as_daily.metrics.sharpe_ratio)
    assert ratio == pytest.approx(math.sqrt(94_500.0 / 365.0), rel=0.01)


def test_mcx_session_scales_annualization(intraday):
    """MCX trades 870 min/session vs NSE's 375; Sharpe scales with sqrt of that."""
    entries, exits = sma_crossover(intraday)
    nse = run(
        intraday,
        entries,
        exits,
        initial_capital=500_000.0,
        fees=0.0003,
        session_minutes=raptorbt.SESSION_NSE,
    )
    mcx = run(
        intraday,
        entries,
        exits,
        initial_capital=500_000.0,
        fees=0.0003,
        session_minutes=raptorbt.SESSION_MCX,
    )
    ratio = mcx.metrics.sharpe_ratio / nse.metrics.sharpe_ratio
    assert ratio == pytest.approx(math.sqrt(870.0 / 375.0), rel=0.001)


def test_intraday_calmar_is_not_derived_from_bar_count(intraday):
    """Bar-count years made an 11k-bar run look like ~31 years of compounding."""
    entries, exits = sma_crossover(intraday)
    r = run(intraday, entries, exits, initial_capital=500_000.0, fees=0.0003)
    calmar = r.metrics.calmar_ratio
    if calmar is not None and r.metrics.max_drawdown_pct > 0:
        # Over ~30 calendar days a plausible CAGR cannot be a rounding artifact
        # of dividing by 31 "years".
        assert abs(calmar) > 1e-6


# ---------------------------------------------------------------------------
# Undefined ratios
# ---------------------------------------------------------------------------


def test_undefined_ratios_are_none_and_json_safe():
    """inf serialized as a bare `Infinity` token, which is not valid JSON."""
    n = 200
    close = 100.0 * np.cumprod(np.full(n, 1.004))  # never falls
    ohlcv = _ohlcv(close, np.arange(n, dtype=np.int64) * DAY_NS, spread=0.001)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    entries[10] = entries[60] = True
    exits[40] = exits[90] = True

    r = run(ohlcv, entries, exits, initial_capital=100_000.0, fees=0.0)
    m = r.metrics
    assert m.profit_factor is None, "no losing trades means undefined, not infinite"
    assert m.payoff_ratio is None

    json.dumps(m.to_dict(), allow_nan=False)  # raises if any inf survives


# ---------------------------------------------------------------------------
# Itemized Indian costs
# ---------------------------------------------------------------------------


def test_fee_breakdown_reconciles_with_charged_fees(daily):
    """Itemized costs and the equity curve must be the same money."""
    entries, exits = sma_crossover(daily)
    r = run(
        daily,
        entries,
        exits,
        initial_capital=1_000_000.0,
        fees=0.001,
        fee_segment="NSE-INTRADAY",
    )
    trades = r.trades()
    assert trades, "fixture should produce trades"

    for t in trades:
        assert t.fee_breakdown is not None
        assert t.fee_breakdown["total"] == pytest.approx(t.fees, abs=1e-9)

    itemized = sum(t.fee_breakdown["total"] for t in trades)
    assert itemized == pytest.approx(r.metrics.total_fees_paid, abs=1e-6)


def test_gst_is_not_levied_on_taxes(daily):
    """GST applies to brokerage/exchange/SEBI, never to STT or stamp duty."""
    entries, exits = sma_crossover(daily)
    r = run(
        daily, entries, exits, initial_capital=1_000_000.0, fees=0.001, fee_segment="NSE-DELIVERY"
    )
    b = r.trades()[0].fee_breakdown
    expected = 0.18 * (b["brokerage"] + b["exchange_txn"] + b["sebi_fee"])
    assert b["gst"] == pytest.approx(expected, rel=1e-9)


def test_flat_fees_leave_breakdown_unset(daily):
    entries, exits = sma_crossover(daily)
    r = run(daily, entries, exits, initial_capital=1_000_000.0, fees=0.001)
    assert r.trades()[0].fee_breakdown is None


# ---------------------------------------------------------------------------
# Shared-capital portfolio runner
# ---------------------------------------------------------------------------


def _portfolio_inputs(n=300, seeds=(7, 23, 41)):
    out = []
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=n)))
        o = _ohlcv(close, np.arange(n, dtype=np.int64) * DAY_NS)
        e, x = sma_crossover(o)
        out.append(
            (
                o["timestamps"],
                o["open"],
                o["high"],
                o["low"],
                o["close"],
                o["volume"],
                e,
                x,
                1,
                1.0,
                f"SYM_{i}",
            )
        )
    return out


def test_portfolio_shares_one_capital_pool():
    """Summing independent per-symbol runs deploys N times the account."""
    instruments = _portfolio_inputs()
    capital = 300_000.0
    cfg = raptorbt.PyBacktestConfig(initial_capital=capital, fees=0.001)
    out = raptorbt.run_portfolio_backtest(instruments, config=cfg)

    peak = max(out.result.equity_curve())
    assert peak < capital * 2.0, (
        f"equity peaked at {peak:,.0f} on a {capital:,.0f} pool; capital is not shared"
    )


def test_max_positions_is_enforced():
    instruments = _portfolio_inputs()
    cfg = raptorbt.PyBacktestConfig(initial_capital=300_000.0, fees=0.001, max_positions=1)
    constrained = raptorbt.run_portfolio_backtest(instruments, config=cfg)

    unconstrained = raptorbt.run_portfolio_backtest(
        instruments, config=raptorbt.PyBacktestConfig(initial_capital=300_000.0, fees=0.001)
    )

    assert len(constrained.result.trades()) < len(unconstrained.result.trades())
    assert constrained.rejected_entries > 0
    assert sum(s.rejected_entries for s in constrained.per_instrument) == (
        constrained.rejected_entries
    )


def test_portfolio_rejects_mismatched_bar_counts():
    a, b = _portfolio_inputs(n=300, seeds=(7,))[0], _portfolio_inputs(n=200, seeds=(23,))[0]
    with pytest.raises(ValueError, match="same number of bars"):
        raptorbt.run_portfolio_backtest([a, b])


def test_portfolio_rejects_unknown_allocation():
    with pytest.raises(ValueError, match="allocation"):
        raptorbt.run_portfolio_backtest(_portfolio_inputs(), allocation="nonsense")


# ---------------------------------------------------------------------------
# Contract surface
# ---------------------------------------------------------------------------


def test_version_matches_package_metadata():
    """__version__ was hardcoded and drifted from the crate before 0.5.0."""
    from importlib.metadata import version

    assert raptorbt.__version__ == version("raptorbt")


def test_session_constants_exported():
    assert raptorbt.SESSION_NSE == 375.0
    assert raptorbt.SESSION_MCX == 870.0
    assert raptorbt.SESSION_CDS == 480.0
    assert raptorbt.SESSION_CONTINUOUS == 0.0
