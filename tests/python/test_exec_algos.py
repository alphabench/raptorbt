"""Execution algorithms and session-calendar DAY expiry (0.5.x)."""

import numpy as np
import pytest

import raptorbt
from raptorbt.strategy import orders


def _flat_bars(n, price=100.0, step=1):
    return {
        "timestamps": np.arange(0, n * step, step, dtype=np.int64),
        "open": np.full(n, price),
        "high": np.full(n, price),
        "low": np.full(n, price),
        "close": np.full(n, price),
        "volume": np.ones(n),
    }


def _config(**kwargs):
    config = raptorbt.PyBacktestConfig(**kwargs)
    config.fees = 0.0
    return config


class _TwapStrategy(raptorbt.Strategy):
    def __init__(self, config=None, **twap):
        super().__init__(config)
        self.twap = twap
        self.fills = []
        self.started = []
        self.completed = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.client_id = self.submit_order(orders.Twap(**self.twap))

    def on_order_filled(self, ctx, event):
        self.fills.append((event.idx, event.client_order_id, event.size))

    def on_algo_started(self, ctx, event):
        self.started.append(event.client_order_id)

    def on_algo_completed(self, ctx, event):
        self.completed.append(event.order_id)


class TestTwap:
    def test_slices_release_one_per_interval(self):
        strategy = _TwapStrategy(side="buy", units=30.0, slices=3, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        assert [f[0] for f in strategy.fills] == [0, 1, 2]
        assert [f[2] for f in strategy.fills] == [10.0, 10.0, 10.0]

    def test_slice_client_ids_derive_from_the_parent(self):
        strategy = _TwapStrategy(side="buy", units=30.0, slices=3, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        parent = strategy.started[0]
        assert [f[1] for f in strategy.fills] == [f"{parent}#{i}" for i in range(3)]

    def test_slices_sum_to_the_requested_size(self):
        # 100 into 3 does not divide evenly; nothing may be lost or invented.
        strategy = _TwapStrategy(side="buy", units=100.0, slices=3, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        assert sum(f[2] for f in strategy.fills) == pytest.approx(100.0)

    def test_lifecycle_events_fire(self):
        strategy = _TwapStrategy(side="buy", units=20.0, slices=2, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        assert len(strategy.started) == 1
        assert len(strategy.completed) == 1

    def test_a_single_slice_is_a_plain_order(self):
        strategy = _TwapStrategy(side="buy", units=25.0, slices=1, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(3), config=_config(), oms_type="hedging"
        )
        assert len(strategy.fills) == 1
        assert strategy.fills[0][2] == 25.0

    def test_slicing_is_timed_not_counted_in_bars(self):
        # Bars one nanosecond apart: an interval of 10ns spans several of
        # them, so slices must not release once per bar.
        strategy = _TwapStrategy(side="buy", units=30.0, slices=3, every=10)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(6), config=_config(), oms_type="hedging"
        )
        assert len(strategy.fills) == 1, "only the first slice is due"


class TestTwapValidation:
    def test_size_frac_cannot_be_sliced(self):
        with pytest.raises(ValueError, match="explicit units"):
            orders.Twap(side="buy", size_frac=0.5, slices=2, every=1)

    def test_slices_must_be_positive(self):
        with pytest.raises(ValueError, match="slices must be"):
            orders.Twap(side="buy", units=10.0, slices=0, every=1)

    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="every must be"):
            orders.Twap(side="buy", units=10.0, slices=2, every=0)

    def test_every_bars_converts_to_a_duration(self):
        twap = orders.Twap(
            side="buy", units=10.0, slices=2, every_bars=3, bar_ns=60_000_000_000
        )
        assert twap.interval_ns == 180_000_000_000

    def test_every_bars_needs_a_bar_duration(self):
        with pytest.raises(ValueError, match="bar_ns"):
            orders.Twap(side="buy", units=10.0, slices=2, every_bars=3)


class TestDayExpiryTradingDate:
    def _run(self, offset_ns):
        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.expired = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(side="buy", price=1.0, units=1.0, tif="day")
                    )

            def on_order_expired(self, ctx, event):
                self.expired.append(event.idx)

        day = 20_468 * 86_400_000_000_000
        # 22:30 UTC (04:00 IST next date), then 00:30 UTC (06:00 IST, same
        # IST trading date but a new UTC date).
        ts = np.array(
            [day + 81_000_000_000_000, day + 86_400_000_000_000 + 1_800_000_000_000],
            dtype=np.int64,
        )
        bars = {
            "timestamps": ts,
            "open": np.full(2, 100.0),
            "high": np.full(2, 100.0),
            "low": np.full(2, 100.0),
            "close": np.full(2, 100.0),
            "volume": np.ones(2),
        }
        strategy = S()
        raptorbt.run_strategy_backtest(
            strategy, **bars, config=_config(session_tz_offset_ns=offset_ns)
        )
        return strategy.expired

    def test_utc_default_expires_on_the_utc_rollover(self):
        assert self._run(0) == [1], "the UTC date rolled"

    def test_an_ist_session_keeps_the_order_alive(self):
        ist = (5 * 3600 + 30 * 60) * 1_000_000_000
        assert self._run(ist) == [], "the IST trading date did not roll"
