"""Behavioral tests: bar aggregation and multi-timeframe strategies (0.5.0)."""

import numpy as np
import pytest

import raptorbt
from raptorbt import PyBacktestConfig, Strategy, run_strategy_backtest

MIN_NS = 60_000_000_000


def _minute_bars(n, start_price=100.0, drift=0.1):
    """n one-minute bars with a gentle drift, ns timestamps."""
    ts = (np.arange(n, dtype=np.int64) * MIN_NS)
    close = start_price + drift * np.arange(n)
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(n, 100.0)
    return {
        "timestamps": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }


class TestAggregateBars:
    def test_five_minute_from_one_minute(self):
        data = _minute_bars(10)
        ts, o, h, l, c, v = raptorbt.aggregate_bars(
            data["timestamps"], data["open"], data["high"], data["low"],
            data["close"], data["volume"], 5, "m",
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
            data["timestamps"], data["open"], data["high"], data["low"],
            data["close"], data["volume"], 3, "tick",
        )
        assert len(ts) == 3
        assert v[0] == pytest.approx(300.0)

    def test_unimplemented_unit_raises(self):
        data = _minute_bars(4)
        with pytest.raises(ValueError, match="not implemented"):
            raptorbt.aggregate_bars(
                data["timestamps"], data["open"], data["high"], data["low"],
                data["close"], data["volume"], 1, "renko",
            )

    def test_length_mismatch_raises(self):
        data = _minute_bars(4)
        with pytest.raises(ValueError, match="length"):
            raptorbt.aggregate_bars(
                data["timestamps"], data["open"][:2], data["high"], data["low"],
                data["close"], data["volume"], 5, "m",
            )


class TestBarsFromTicks:
    def test_volume_bars_from_ticks(self):
        ts = np.arange(10, dtype=np.int64) * 1_000_000_000
        ltp = np.full(10, 100.0)
        ltp[5] = 0.0  # missing print: skipped
        buys = np.full(10, 3.0)
        sells = np.full(10, 2.0)
        bts, o, h, l, c, v = raptorbt.bars_from_ticks(ts, ltp, buys, sells, 10, "volume")
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
                int(data["timestamps"][i]), float(data["open"][i]),
                float(data["high"][i]), float(data["low"][i]),
                float(data["close"][i]), float(data["volume"][i]),
            )
            if done is not None:
                streamed.append(done)
        tail = agg.flush()
        if tail is not None:
            streamed.append(tail)

        ts, o, h, l, c, v = raptorbt.aggregate_bars(
            data["timestamps"], data["open"], data["high"], data["low"],
            data["close"], data["volume"], 5, "m",
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
        config = PyBacktestConfig()
        config.fees = 0.0
        run_strategy_backtest(S, **data, config=config)

        composites = [e for e in events if e[0] == "composite"]
        # Bars 0-4 fill window one; bar 5 (ts=5min... window keyed on ts)
        assert len(composites) == 2
        # Each composite dispatches before that bar index's own on_bar.
        for kind, idx, *_rest in composites:
            position = events.index(("composite", idx, *_rest))
            assert events[position + 1] == ("bar", idx) or ("bar", idx) in events[position:]

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
            "timestamps": ts, "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close, "volume": np.full(n, 100.0),
        }
        config = PyBacktestConfig()
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
        from generate import GoldenSma, make_data, result_digest

        fixtures = json.loads((here / "golden" / "fixtures.json").read_text())
        ts, o, h, l, c, v = make_data()
        result = raptorbt.run_strategy_backtest(GoldenSma, ts, o, h, l, c, v)
        assert result_digest(result) == fixtures["class/sma_cross"]
