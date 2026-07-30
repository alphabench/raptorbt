"""Behavioral tests: streaming indicators, clock/timers, cache, portfolio
view (0.5.0)."""

import numpy as np
import pytest

import raptorbt
from raptorbt import Indicator, PyBacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

MIN_NS = 60_000_000_000


def _bars(closes, lows=None, highs=None, step_ns=MIN_NS):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(n, dtype=np.int64) * step_ns,
        "open": closes.copy(),
        "high": np.asarray(highs, dtype=np.float64) if highs is not None else closes + 1.0,
        "low": np.asarray(lows, dtype=np.float64) if lows is not None else closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _zero_fee_config():
    config = PyBacktestConfig()
    config.fees = 0.0
    return config


class TestStreamingIndicators:
    def test_matches_batch_functions(self):
        rng = np.random.default_rng(3)
        closes = 100.0 + np.cumsum(rng.normal(0, 1, 120))
        batch_sma = raptorbt.sma(closes, 14)
        batch_rsi = raptorbt.rsi(closes, 14)

        sma = Indicator.sma(14)
        rsi = Indicator.rsi(14)
        for i, c in enumerate(closes):
            s = sma.update_bar(c, c + 1, c - 1, c)
            r = rsi.update_bar(c, c + 1, c - 1, c)
            if s is not None:
                assert s == pytest.approx(batch_sma[i], abs=1e-9)
            else:
                assert np.isnan(batch_sma[i])
            if r is not None:
                assert r == pytest.approx(batch_rsi[i], abs=1e-9)

    def test_tuple_valued_indicators(self):
        boll = Indicator.bollinger(3, 2.0)
        assert boll.update_bar(0, 0, 0, 100.0) is None
        assert not boll.initialized
        boll.update_bar(0, 0, 0, 102.0)
        mid, upper, lower = boll.update_bar(0, 0, 0, 104.0)
        assert mid == pytest.approx(102.0)
        assert upper > mid > lower
        assert boll.initialized

        macd = Indicator.macd(2, 3, 2)
        for c in [100.0, 101.0, 102.0, 103.0, 104.0]:
            out = macd.update_bar(0, 0, 0, c)
        assert out is not None and len(out) == 3

    def test_registration_updates_before_on_bar(self):
        observed = []

        class S(Strategy):
            def on_start(self, ctx):
                self.fast = self.register_indicator(Indicator.sma(3))

            def on_bar(self, ctx):
                observed.append((ctx.idx, self.fast.value))

        data = _bars([100.0, 101.0, 102.0, 103.0])
        run_strategy_backtest(S, **data, config=_zero_fee_config())
        # Warm at idx 2 (three bars seen), value visible inside on_bar.
        assert observed[0] == (0, None)
        assert observed[1] == (1, None)
        assert observed[2][1] == pytest.approx(101.0)
        assert observed[3][1] == pytest.approx(102.0)

    def test_registration_on_composite_stream(self):
        values = []

        class S(Strategy):
            def on_start(self, ctx):
                h5 = self.subscribe_bars(5, "m")
                self.trend = self.register_indicator(Indicator.sma(2), stream_id=h5)

            def on_composite_bar(self, ctx, bar):
                values.append(self.trend.value)

        data = _bars(np.arange(100.0, 121.0))  # 21 one-minute bars
        run_strategy_backtest(S, **data, config=_zero_fee_config())
        # 4 completed 5m windows: SMA(2) of composite closes warms on the 2nd.
        assert values[0] is None
        assert values[1] is not None

    def test_indicators_initialized_helper(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.ready_at = None

            def on_start(self, ctx):
                self.register_indicator(Indicator.sma(3))
                self.register_indicator(Indicator.ema(5))

            def on_bar(self, ctx):
                if self.ready_at is None and self.indicators_initialized():
                    self.ready_at = ctx.idx

        data = _bars(np.arange(100.0, 110.0))
        strategy = S()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.ready_at == 4  # slowest (ema 5) warms at idx 4


class TestClock:
    def test_time_alert_fires_once_before_on_bar(self):
        events = []

        class S(Strategy):
            def on_start(self, ctx):
                self.set_ok = False

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.clock.set_time_alert("flatten", 3 * MIN_NS)
                events.append(("bar", ctx.idx))

            def on_time_event(self, ctx, event):
                events.append(("time", ctx.idx, event.name, event.ts_scheduled))

        data = _bars([100.0] * 6)
        run_strategy_backtest(S, **data, config=_zero_fee_config())
        time_events = [e for e in events if e[0] == "time"]
        assert time_events == [("time", 3, "flatten", 3 * MIN_NS)]
        # Fired before that bar's on_bar.
        assert events.index(time_events[0]) < events.index(("bar", 3))

    def test_recurring_timer_and_cancel(self):
        fires = []

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.clock.set_timer("beat", 2 * MIN_NS)
                if ctx.idx == 5:
                    self.clock.cancel_timer("beat")

            def on_time_event(self, ctx, event):
                fires.append(ctx.idx)

        data = _bars([100.0] * 10)
        run_strategy_backtest(S, **data, config=_zero_fee_config())
        # Set at ts=0 -> first due at 2min (idx 2), then 4 (idx 4); canceled at 5.
        assert fires == [2, 4]

    def test_timed_flatten_capability(self):
        """The capability test: an end-of-run alert flattens the position."""

        class S(Strategy):
            def on_start(self, ctx):
                self.clock.set_time_alert("eod", 4 * MIN_NS)

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()

            def on_time_event(self, ctx, event):
                if event.name == "eod" and ctx.position is not None:
                    self.close_position()

        data = _bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        result = run_strategy_backtest(S, **data, config=_zero_fee_config())
        trade = result.trades()[0]
        assert trade.exit_reason == "Signal"
        assert trade.exit_idx == 4


class TestCacheAndPortfolioView:
    def test_cache_tracks_order_lifecycle(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.checks = {}

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.oid = self.submit_order(
                        orders.Limit(side="buy", price=99.0, units=10.0)
                    )
                    self.kid = self.submit_order(
                        orders.Limit(side="buy", price=90.0, units=10.0)
                    )
                if ctx.idx == 1:
                    self.cancel_order(self.kid)
                if ctx.idx == 3:
                    self.checks["filled"] = self.cache.order(self.oid).status
                    self.checks["canceled"] = self.cache.order(self.kid).status
                    self.checks["open"] = [o.client_id for o in self.cache.orders_open()]

        data = _bars([100.0, 100.0, 98.5, 99.5])
        strategy = S()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.checks["filled"] == "filled"
        assert strategy.checks["canceled"] == "canceled"
        assert strategy.checks["open"] == []

    def test_cache_realized_pnl(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.observed = None

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()
                elif ctx.idx == 2:
                    self.close_position()
                elif ctx.idx == 3:
                    self.observed = self.cache.realized_pnl()

        data = _bars([100.0, 101.0, 102.0, 102.0])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.observed == pytest.approx(result.trades()[0].pnl)

    def test_net_position_view_under_hedging(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.views = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.Market(side="buy", units=10.0))
                    self.submit_order(orders.Market(side="sell", units=4.0))
                if ctx.idx == 1:
                    self.views.append(
                        (ctx.net_position, ctx.is_net_long, ctx.is_net_short, ctx.is_flat)
                    )

        data = _bars([100.0, 101.0, 102.0])
        strategy = S()
        run_strategy_backtest(
            strategy, **data, config=_zero_fee_config(), oms_type="hedging"
        )
        assert strategy.views == [(6.0, True, False, False)]

    def test_golden_gate_still_exact(self):
        import json
        import sys
        from pathlib import Path

        here = Path(__file__).parent
        sys.path.insert(0, str(here / "golden"))
        from generate import GoldenSma, result_digest, thaw_inputs

        fixtures = json.loads((here / "golden" / "fixtures.json").read_text())
        ts, o, h, l, c, v, _, _ = thaw_inputs(fixtures["inputs"]["shared"])
        result = raptorbt.run_strategy_backtest(GoldenSma, ts, o, h, l, c, v)
        assert result_digest(result) == fixtures["class/sma_cross"]
