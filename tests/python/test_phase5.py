"""Behavioral tests: bar aggregation and multi-timeframe strategies (0.5.0)."""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest

MIN_NS = 60_000_000_000


def _minute_bars(n, start_price=100.0, drift=0.1):
    """n one-minute bars with a gentle drift, ns timestamps."""
    ts = np.arange(n, dtype=np.int64) * MIN_NS
    close = start_price + drift * np.arange(n)
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(n, 100.0)
    return {
        "timestamps": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestAggregateBars:
    def test_five_minute_from_one_minute(self):
        data = _minute_bars(10)
        ts, o, h, l, c, v = raptorbt.aggregate_bars(
            data["timestamps"],
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            5,
            "m",
        )
        assert len(ts) == 2
        # Window-end stamps: bars 0-4 close the window ending at 5min.
        assert ts[0] == 5 * MIN_NS
        assert o[0] == data["open"][0]
        assert c[0] == data["close"][4]
        assert h[0] == data["high"][:5].max()
        assert l[0] == data["low"][:5].min()
        assert v[0] == pytest.approx(500.0)
        # Second window flushed at end of data.
        assert c[1] == data["close"][9]

    def test_tick_count_unit(self):
        data = _minute_bars(9)
        ts, o, h, l, c, v = raptorbt.aggregate_bars(
            data["timestamps"],
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            3,
            "tick",
        )
        assert len(ts) == 3
        assert v[0] == pytest.approx(300.0)

    def test_unknown_unit_raises(self):
        data = _minute_bars(4)
        with pytest.raises(ValueError, match="unknown aggregation unit"):
            raptorbt.aggregate_bars(
                data["timestamps"],
                data["open"],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
                1,
                "fortnight",
            )

    def test_length_mismatch_raises(self):
        data = _minute_bars(4)
        with pytest.raises(ValueError, match="length"):
            raptorbt.aggregate_bars(
                data["timestamps"],
                data["open"][:2],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
                5,
                "m",
            )


class TestBarsFromTicks:
    def test_volume_bars_from_ticks(self):
        ts = np.arange(10, dtype=np.int64) * 1_000_000_000
        ltp = np.full(10, 100.0)
        ltp[5] = 0.0  # missing print: skipped
        buys = np.full(10, 3.0)
        sells = np.full(10, 2.0)
        bts, o, h, l, c, v = raptorbt.bars_from_ticks(
            ts, ltp, buys, sells, 10, "volume"
        )
        # 9 valid trades of size 5: thresholds at 10 -> bars of 2 trades each.
        assert len(bts) >= 4
        assert v[0] == pytest.approx(10.0)

    def test_time_bars_from_ticks(self):
        ts = np.arange(6, dtype=np.int64) * 1_000_000_000  # 1s apart
        ltp = np.array([100.0, 101.0, 99.0, 100.5, 102.0, 101.5])
        deltas = np.ones(6)
        bts, o, h, l, c, v = raptorbt.bars_from_ticks(ts, ltp, deltas, deltas, 3, "s")
        assert len(bts) == 2
        assert h[0] == pytest.approx(101.0)
        assert l[0] == pytest.approx(99.0)


class TestStreamingAggregator:
    def test_streaming_matches_batch(self):
        data = _minute_bars(23)
        agg = raptorbt.BarAggregator(5, "m")
        streamed = []
        for i in range(23):
            done = agg.push_bar(
                int(data["timestamps"][i]),
                float(data["open"][i]),
                float(data["high"][i]),
                float(data["low"][i]),
                float(data["close"][i]),
                float(data["volume"][i]),
            )
            if done is not None:
                streamed.append(done)
        tail = agg.flush()
        if tail is not None:
            streamed.append(tail)

        ts, o, h, l, c, v = raptorbt.aggregate_bars(
            data["timestamps"],
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            5,
            "m",
        )
        assert len(streamed) == len(ts)
        for i, bar in enumerate(streamed):
            assert bar == (ts[i], o[i], h[i], l[i], c[i], v[i])


class TestMultiTimeframeStrategy:
    def test_composite_bars_dispatch_in_order(self):
        events = []

        class S(Strategy):
            def on_start(self, ctx):
                self.h5 = self.subscribe_bars(5, "m")

            def on_composite_bar(self, ctx, bar):
                events.append(("composite", ctx.idx, bar.timestamp, bar.stream_id))

            def on_bar(self, ctx):
                events.append(("bar", ctx.idx))

        data = _minute_bars(12)
        config = BacktestConfig()
        config.fees = 0.0
        run_strategy_backtest(S, **data, config=config)

        composites = [e for e in events if e[0] == "composite"]
        # Bars 0-4 fill window one; bar 5 (ts=5min... window keyed on ts)
        assert len(composites) == 2
        # Each composite dispatches before that bar index's own on_bar.
        for kind, idx, *_rest in composites:
            position = events.index(("composite", idx, *_rest))
            assert (
                events[position + 1] == ("bar", idx)
                or ("bar", idx) in events[position:]
            )

    def test_trend_filter_gates_entries(self):
        """The capability test: a 5-minute trend filter gating 1-minute entries."""

        class S(Strategy):
            def on_start(self, ctx):
                self.subscribe_bars(5, "m")
                self.trend_up = False

            def on_composite_bar(self, ctx, bar):
                self.trend_up = bar.close > bar.open

            def on_bar(self, ctx):
                if self.trend_up and ctx.position is None:
                    self.enter()

        # Falling first half, rising second half.
        n = 20
        ts = np.arange(n, dtype=np.int64) * MIN_NS
        close = np.concatenate([100 - 0.5 * np.arange(10), 95 + 0.8 * np.arange(10)])
        open_ = np.concatenate([[100.0], close[:-1]])
        data = {
            "timestamps": ts,
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": np.full(n, 100.0),
        }
        config = BacktestConfig()
        config.fees = 0.0
        result = run_strategy_backtest(S, **data, config=config)
        trades = result.trades()
        assert len(trades) == 1
        # The first up-trending composite closes at the 15-minute boundary
        # (windows 0-5,5-10 fall; 10-15 rises); entry follows it.
        assert trades[0].entry_idx >= 15

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


class TestRenkoBars:
    def test_a_burst_emits_every_brick(self):
        # A three-brick jump must not collapse into one bar.
        ts = np.array([0, 1], dtype=np.int64)
        px = np.array([100.0, 103.0])
        out = raptorbt.aggregate_bars(
            ts, px, px, px, px, np.ones(2), 1, "renko", brick_size=1.0
        )
        assert list(out[1]) == [100.0, 101.0, 102.0]
        assert list(out[4]) == [101.0, 102.0, 103.0]

    def test_bricks_have_no_wicks(self):
        ts = np.array([0, 1], dtype=np.int64)
        px = np.array([100.0, 102.0])
        _, o, h, l, c, _ = raptorbt.aggregate_bars(
            ts, px, px, px, px, np.ones(2), 1, "renko", brick_size=1.0
        )
        for i in range(len(o)):
            assert h[i] == max(o[i], c[i])
            assert l[i] == min(o[i], c[i])

    def test_time_and_volume_never_close_a_brick(self):
        # Hours pass with heavy volume but no price movement.
        n = 20
        ts = np.arange(n, dtype=np.int64) * 3_600_000_000_000
        px = np.full(n, 100.0)
        out = raptorbt.aggregate_bars(
            ts, px, px, px, px, np.full(n, 1e6), 1, "renko", brick_size=1.0
        )
        assert len(out[0]) == 0

    def test_streaming_drain_matches_batch(self):
        ts = np.array([0, 1], dtype=np.int64)
        px = np.array([100.0, 103.0])
        batch = raptorbt.aggregate_bars(
            ts, px, px, px, px, np.ones(2), 1, "renko", brick_size=1.0
        )

        agg = raptorbt.BarAggregator(1, "renko", brick_size=1.0)
        agg.push_trade(0, 100.0, 1.0)
        streamed = []
        first = agg.push_trade(1, 103.0, 1.0)
        while first is not None:
            streamed.append(first)
            first = agg.next_pending()

        assert [b[4] for b in streamed] == list(batch[4])

    def test_aggregator_honours_brick_size_when_it_differs_from_step(self):
        """BarAggregator must use the brick_size it was given, not the step.

        Regression test. ``BarAggregator.__init__`` accepted ``brick_size`` in
        its PyO3 signature but called a helper that hard-coded ``0.0``, and
        ``resolved_brick`` treats a non-positive brick as "fall back to step".
        So a user asking for 5-point bricks silently got ``step``-point bricks:
        a 10-point move produced 10 bars instead of 2, and every Renko backtest
        built through the streaming aggregator was wrong.

        Every pre-existing test used ``step=1, brick_size=1.0``, where the
        fallback returns the same number the caller asked for and the bug is
        invisible. This one separates them on purpose.
        """
        agg = raptorbt.BarAggregator(1, "renko", brick_size=5.0)
        closed = []
        for i, price in enumerate(range(100, 111)):
            bar = agg.push_bar(
                i, float(price), float(price), float(price), float(price), 1.0
            )
            while bar is not None:
                closed.append(bar)
                bar = agg.next_pending()

        # 100 -> 110 is two whole 5-point bricks.
        assert [b[4] for b in closed] == [105.0, 110.0]

    def test_aggregator_matches_batch_for_a_non_unit_brick(self):
        """Streaming and batch Renko agree when brick_size != step.

        The existing parity test pins ``step=1, brick_size=1.0``, where the two
        paths agreed even while the streaming one ignored its argument.
        """
        ts = np.arange(11, dtype=np.int64)
        px = np.arange(100.0, 111.0)
        batch = raptorbt.aggregate_bars(
            ts, px, px, px, px, np.ones(11), 1, "renko", brick_size=5.0
        )

        agg = raptorbt.BarAggregator(1, "renko", brick_size=5.0)
        streamed = []
        for i in range(11):
            bar = agg.push_bar(int(ts[i]), px[i], px[i], px[i], px[i], 1.0)
            while bar is not None:
                streamed.append(bar)
                bar = agg.next_pending()

        assert [b[4] for b in streamed] == list(batch[4])

    def test_brick_size_defaults_to_step(self):
        ts = np.array([0, 1], dtype=np.int64)
        px = np.array([100.0, 105.0])
        out = raptorbt.aggregate_bars(ts, px, px, px, px, np.ones(2), 5, "renko")
        # step=5 means 5.00-point bricks: exactly one.
        assert len(out[0]) == 1
        assert out[4][0] == 105.0


class TestSignedFlowBars:
    def _ticks(self, buys, sells, price=100.0):
        n = len(buys)
        return (
            np.arange(n, dtype=np.int64),
            np.full(n, price),
            np.asarray(buys, dtype=np.float64),
            np.asarray(sells, dtype=np.float64),
        )

    def test_balanced_flow_never_closes_an_imbalance_bar(self):
        # Alternating buys and sells cancel, however heavy. Only the
        # end-of-data flush emits, so the whole tape is one bar.
        ts, ltp, buy, sell = self._ticks([10.0, 0.0] * 10, [0.0, 10.0] * 10)
        out = raptorbt.bars_from_ticks(ts, ltp, buy, sell, 50, "volume_imbalance")
        assert len(out[0]) == 1
        assert out[0][0] == 19, "the flush of the trailing partial"

    def test_balanced_flow_does_close_a_runs_bar(self):
        # The same tape closes runs bars: one-sided accumulation still grows.
        ts, ltp, buy, sell = self._ticks([10.0, 0.0] * 10, [0.0, 10.0] * 10)
        out = raptorbt.bars_from_ticks(ts, ltp, buy, sell, 50, "volume_runs")
        assert len(out[0]) > 0

    def test_one_sided_flow_closes_on_the_threshold(self):
        ts, ltp, buy, sell = self._ticks([10.0] * 10, [0.0] * 10)
        out = raptorbt.bars_from_ticks(ts, ltp, buy, sell, 30, "volume_imbalance")
        # 100 units of one-way flow at a threshold of 30: three closed
        # bars of 30, plus the trailing 10 flushed at end of data.
        assert len(out[0]) == 4
        assert list(out[0]) == [2, 5, 8, 9]

    def test_tick_imbalance_counts_trades_not_size(self):
        ts, ltp, buy, sell = self._ticks([1.0] * 6, [0.0] * 6)
        out = raptorbt.bars_from_ticks(ts, ltp, buy, sell, 2, "tick_imbalance")
        assert len(out[0]) == 3

    def test_value_scales_by_price(self):
        # The same size at a higher price reaches the threshold sooner.
        ts, ltp, buy, sell = self._ticks([1.0] * 4, [0.0] * 4, price=100.0)
        cheap = raptorbt.bars_from_ticks(ts, ltp, buy, sell, 200, "value_imbalance")
        ts, ltp, buy, sell = self._ticks([1.0] * 4, [0.0] * 4, price=1000.0)
        rich = raptorbt.bars_from_ticks(ts, ltp, buy, sell, 200, "value_imbalance")
        assert len(rich[0]) > len(cheap[0])

    def test_unsigned_bars_fall_back_to_the_tick_rule(self):
        # aggregate_bars carries no flow data; direction comes from price.
        closes = np.array([100.0, 101.0, 102.0, 103.0])
        ts = np.arange(4, dtype=np.int64)
        out = raptorbt.aggregate_bars(
            ts, closes, closes, closes, closes, np.ones(4), 2, "tick_imbalance"
        )
        # Four consecutive up-ticks at a threshold of 2.
        assert len(out[0]) == 2
