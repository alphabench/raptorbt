"""Behavioral tests for the class-based strategy contract.

The load-bearing property is equivalence: a strategy class making the same
decisions as a precomputed signal array pair must produce bit-identical
trades, curves, and metrics, because both paths share one execution core.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import Strategy, run_strategy_backtest

DAY_NS = 86_400_000_000_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ohlcv(close, timestamps, spread=0.004):
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


def _sma(x, w):
    n = len(x)
    out = np.full(n, np.nan)
    if n >= w:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def sma_crossover_signals(ohlcv, fast=10, slow=30):
    close = ohlcv["close"]
    n = len(close)
    f, s = _sma(close, fast), _sma(close, slow)
    valid = ~(np.isnan(f) | np.isnan(s))
    above = np.zeros(n, dtype=bool)
    above[valid] = f[valid] > s[valid]
    prev = np.roll(above, 1)
    prev[0] = False
    return (above & ~prev & valid), (~above & prev & valid)


class SmaCross(Strategy):
    """Class twin of ``sma_crossover_signals``: same decisions, same bars."""

    def __init__(self, fast=10, slow=30):
        super().__init__()
        self.fast_period = fast
        self.slow_period = slow

    def on_start(self, ctx):
        self.fast = _sma(ctx.close, self.fast_period)
        self.slow = _sma(ctx.close, self.slow_period)

    def on_bar(self, ctx):
        i = ctx.idx
        f, s = self.fast, self.slow
        if np.isnan(f[i]) or np.isnan(s[i]):
            return
        above = f[i] > s[i]
        prev_above = False
        if i > 0 and not (np.isnan(f[i - 1]) or np.isnan(s[i - 1])):
            prev_above = f[i - 1] > s[i - 1]
        if above and not prev_above and ctx.position is None:
            self.enter()
        elif not above and prev_above and ctx.position is not None:
            self.close_position()


def _run_array(ohlcv, entries, exits, config=None):
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
        config=config,
    )


def _run_class(ohlcv, strategy, config=None):
    return run_strategy_backtest(
        strategy,
        ohlcv["timestamps"],
        ohlcv["open"],
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        ohlcv["volume"],
        symbol="TEST",
        config=config,
    )


def _assert_identical(res_a, res_b):
    ta, tb = res_a.trades(), res_b.trades()
    assert len(ta) == len(tb)
    for a, b in zip(ta, tb):
        assert a.entry_idx == b.entry_idx
        assert a.exit_idx == b.exit_idx
        assert a.entry_price == b.entry_price
        assert a.exit_price == b.exit_price
        assert a.pnl == b.pnl
        assert a.fees == b.fees
        assert a.exit_reason == b.exit_reason
    assert np.array_equal(res_a.equity_curve(), res_b.equity_curve())
    assert np.array_equal(res_a.drawdown_curve(), res_b.drawdown_curve())
    assert np.array_equal(res_a.returns(), res_b.returns())
    ma, mb = res_a.metrics, res_b.metrics
    for field in (
        "total_return_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "profit_factor",
        "total_trades",
        "total_fees_paid",
        "expectancy",
        "exposure_pct",
    ):
        assert getattr(ma, field) == getattr(mb, field), field


# ---------------------------------------------------------------------------
# Equivalence: the gate for the class-based path
# ---------------------------------------------------------------------------


def test_class_matches_array_no_stops(daily):
    entries, exits = sma_crossover_signals(daily)
    res_array = _run_array(daily, entries, exits)
    res_class = _run_class(daily, SmaCross())
    assert res_array.metrics.total_trades > 0
    _assert_identical(res_array, res_class)


def test_class_matches_array_fixed_stop(daily):
    entries, exits = sma_crossover_signals(daily)

    cfg_a = raptorbt.PyBacktestConfig()
    cfg_a.set_fixed_stop(0.03)
    cfg_b = raptorbt.PyBacktestConfig()
    cfg_b.set_fixed_stop(0.03)

    res_array = _run_array(daily, entries, exits, config=cfg_a)
    res_class = _run_class(daily, SmaCross(), config=cfg_b)
    _assert_identical(res_array, res_class)


def test_class_matches_array_atr_stop(daily):
    """Pins the runner's ATR precompute against the engine's."""
    entries, exits = sma_crossover_signals(daily)

    cfg_a = raptorbt.PyBacktestConfig()
    cfg_a.set_atr_stop(2.0, 14)
    cfg_a.set_atr_target(3.0, 14)
    cfg_b = raptorbt.PyBacktestConfig()
    cfg_b.set_atr_stop(2.0, 14)
    cfg_b.set_atr_target(3.0, 14)

    res_array = _run_array(daily, entries, exits, config=cfg_a)
    res_class = _run_class(daily, SmaCross(), config=cfg_b)
    _assert_identical(res_array, res_class)


def test_class_accepts_class_or_instance(daily):
    class Fixed(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 10:
                self.enter()
            elif ctx.idx == 50:
                self.close_position()

    res_from_class = _run_class(daily, Fixed)
    res_from_instance = _run_class(daily, Fixed())
    _assert_identical(res_from_class, res_from_instance)


# ---------------------------------------------------------------------------
# Hooks and event dispatch
# ---------------------------------------------------------------------------


def test_hook_firing_order_and_counts(daily):
    calls = []

    class Recorder(Strategy):
        def on_start(self, ctx):
            calls.append("start")

        def on_bar(self, ctx):
            if ctx.idx == 10:
                self.enter()
            elif ctx.idx == 50:
                self.close_position()

        def on_stop(self, ctx):
            calls.append("stop")

        def on_order_filled(self, ctx, event):
            calls.append(f"filled:{event.kind}@{event.idx}")

        def on_position_opened(self, ctx, event):
            calls.append(f"opened@{event.idx}")

        def on_position_closed(self, ctx, event):
            calls.append(f"closed@{event.idx}")
            assert event.trade is not None
            assert event.trade.entry_idx == 10
            assert event.trade.exit_idx == 50

    _run_class(daily, Recorder())
    assert calls == [
        "start",
        "filled:entered@10",
        "opened@10",
        "filled:exited@50",
        "closed@50",
        "stop",
    ]


def test_set_stop_price_mid_position_triggers_stop(daily):
    class TightenStop(Strategy):
        def __init__(self):
            super().__init__()
            self.stop_set_at = None

        def on_bar(self, ctx):
            if ctx.idx == 10:
                self.enter()
            elif ctx.idx == 15 and ctx.position is not None:
                # A stop just under the current close must be hit almost
                # immediately on noisy data.
                ctx.set_stop_price(ctx.bar.close * 0.998)
                self.stop_set_at = ctx.idx

    strat = TightenStop()
    res = _run_class(daily, strat)
    trades = res.trades()
    assert strat.stop_set_at == 15
    assert len(trades) == 1
    assert trades[0].exit_reason == "StopLoss"
    # set_stop_price in on_bar applies to the same bar's engine step.
    assert trades[0].exit_idx >= 15


def test_entry_stop_override(daily):
    class ExplicitStop(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 10:
                self.enter(stop_price=ctx.bar.close * 0.99)

    res = _run_class(daily, ExplicitStop())
    trades = res.trades()
    assert len(trades) == 1
    assert trades[0].exit_reason == "StopLoss"


def test_position_snapshot_fields(daily):
    seen = {}

    class Snapshot(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 10:
                self.enter()
            elif ctx.idx == 11:
                pos = ctx.position
                seen["entry_idx"] = pos.entry_idx
                seen["direction"] = pos.direction
                seen["size"] = pos.size

    _run_class(daily, Snapshot())
    assert seen["entry_idx"] == 10
    assert seen["direction"] == 1
    assert seen["size"] > 0


def test_size_frac_scales_position(daily):
    sizes = {}

    class Sized(Strategy):
        def __init__(self, frac, key):
            super().__init__()
            self.frac = frac
            self.key = key

        def on_bar(self, ctx):
            if ctx.idx == 10:
                self.enter(size_frac=self.frac)

        def on_position_opened(self, ctx, event):
            sizes[self.key] = event.size

    _run_class(daily, Sized(1.0, "full"))
    _run_class(daily, Sized(0.5, "half"))
    assert sizes["half"] == pytest.approx(sizes["full"] * 0.5)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_conflicting_intents_raise(daily):
    class Conflicted(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 10:
                self.enter()
            elif ctx.idx == 20:
                self.enter()
                self.close_position()

    with pytest.raises(ValueError, match="same bar"):
        _run_class(daily, Conflicted())


def test_double_finish_raises(daily):
    session = raptorbt.PyKernelSession(symbol="TEST")
    session.step(0, 0, 100.0, 101.0, 99.0, 100.0, 1000.0)
    session.finish()
    with pytest.raises(ValueError, match="finished"):
        session.finish()
    with pytest.raises(ValueError, match="finished"):
        session.step(1, 1, 100.0, 101.0, 99.0, 100.0, 1000.0)


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        raptorbt.PyKernelSession(symbol="TEST", direction=2)


def test_mismatched_lengths_raise(daily):
    with pytest.raises(ValueError, match="length"):
        run_strategy_backtest(
            SmaCross(),
            daily["timestamps"],
            daily["open"][:-1],
            daily["high"],
            daily["low"],
            daily["close"],
            daily["volume"],
        )


def test_short_direction_session(daily):
    entries, exits = sma_crossover_signals(daily)

    class ShortSma(SmaCross):
        pass

    res_array = raptorbt.run_single_backtest(
        daily["timestamps"],
        daily["open"],
        daily["high"],
        daily["low"],
        daily["close"],
        daily["volume"],
        entries,
        exits,
        direction=-1,
        symbol="TEST",
    )
    res_class = run_strategy_backtest(
        ShortSma(),
        daily["timestamps"],
        daily["open"],
        daily["high"],
        daily["low"],
        daily["close"],
        daily["volume"],
        direction=-1,
        symbol="TEST",
    )
    _assert_identical(res_array, res_class)


def test_zero_size_entry_fires_on_order_rejected(daily):
    rejections = []

    class TinySize(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 10 and ctx.position is None:
                # Far below one lot once lot_size is applied.
                self.enter(size_frac=0.000001)

        def on_order_rejected(self, ctx, event):
            rejections.append(event.reject_reason)

    inst = raptorbt.PyInstrumentConfig(lot_size=100.0)
    res = run_strategy_backtest(
        TinySize(),
        daily["timestamps"],
        daily["open"],
        daily["high"],
        daily["low"],
        daily["close"],
        daily["volume"],
        symbol="TEST",
        instrument_config=inst,
    )
    assert res.metrics.total_trades == 0
    assert rejections == ["ZeroSize"]
