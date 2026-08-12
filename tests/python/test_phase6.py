"""Behavioral tests: calendar/session-aligned bars and the multi-instrument
portfolio session (0.5.0)."""

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    Strategy,
    run_portfolio_strategy,
)
from raptorbt.strategy import orders

DAY_NS = 86_400_000_000_000
HOUR_NS = 3_600_000_000_000


def _zero_fee_config():
    config = BacktestConfig()
    config.fees = 0.0
    return config


def _bars(closes, start_ts=0, step_ns=60_000_000_000):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": start_ts + np.arange(n, dtype=np.int64) * step_ns,
        "open": closes.copy(),
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


class TestCalendarAndSessionBars:
    def test_month_bars_split_on_civil_boundary(self):
        # Two bars in Jan 2024, one in Feb 2024. 2024-01-01 = epoch day 19723.
        ts = np.array(
            [19_723 * DAY_NS, (19_723 + 15) * DAY_NS, (19_723 + 31) * DAY_NS],
            dtype=np.int64,
        )
        closes = np.array([100.0, 101.0, 102.0])
        data = {
            "timestamps": ts,
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": np.full(3, 10.0),
        }
        bts, o, h, l, c, v = raptorbt.aggregate_bars(
            data["timestamps"],
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
            1,
            "month",
        )
        assert len(bts) == 2
        # January's bar is stamped at the start of February.
        assert bts[0] == (19_723 + 31) * DAY_NS
        assert v[0] == pytest.approx(20.0)

    def test_ist_aligned_day_bars(self):
        """23:30 IST belongs to the same trading date as 09:30 IST."""
        day0 = 19_724 * DAY_NS  # 2024-01-02 00:00 UTC
        ts = np.array(
            [day0 + 4 * HOUR_NS, day0 + 18 * HOUR_NS, day0 + 20 * HOUR_NS],
            dtype=np.int64,
        )
        closes = np.array([100.0, 101.0, 102.0])
        args = (ts, closes, closes + 1, closes - 1, closes, np.full(3, 10.0))

        # IST alignment: first two ticks share a trading date.
        bts, *_rest, v = raptorbt.aggregate_bars(*args, 1, "d", raptorbt.IST_OFFSET_NS)
        assert len(bts) == 2
        assert v[0] == pytest.approx(20.0)
        assert (bts[0] + raptorbt.IST_OFFSET_NS) % DAY_NS == 0

        # UTC alignment lumps all three into one day.
        bts_utc, *_r, v_utc = raptorbt.aggregate_bars(*args, 1, "d", 0)
        assert len(bts_utc) == 1
        assert v_utc[0] == pytest.approx(30.0)


class TestPortfolioSession:
    def test_shared_capital_and_per_instrument_positions(self):
        """The capability test: N symbols, one pool, per-symbol positions."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.snapshots = []

            def on_bar(self, ctx):
                # Enter each symbol with 40% of the pool on its first bar.
                if ctx.idx == 0:
                    self.enter(size_frac=0.4)
                if ctx.idx == 2:
                    self.snapshots.append(
                        (ctx.symbol, ctx.position is not None, ctx.equity, ctx.cash)
                    )

        data = {
            "AAA": _bars([100.0, 101.0, 102.0, 103.0]),
            "BBB": _bars([50.0, 50.5, 51.0, 51.5], start_ts=5_000_000_000),
        }
        strategy = S()
        result = run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        # Both instruments held positions simultaneously from one pool.
        assert len(strategy.snapshots) == 2
        assert all(has_position for _, has_position, _, _ in strategy.snapshots)
        # Cash: 100k -> 60k after AAA -> 36k after BBB (40% of remainder).
        _, _, _, cash = strategy.snapshots[-1]
        assert cash == pytest.approx(36_000.0, rel=1e-6)

        summary = {s.symbol: s for s in result.per_instrument}
        assert set(summary) == {"AAA", "BBB"}
        assert summary["AAA"].trades == 1  # force-closed at end
        assert summary["BBB"].trades == 1
        assert summary["AAA"].pnl > 0
        # Portfolio equity curve sampled per merged event: 8 bars total.
        assert len(result.result.equity_curve()) == 8

    def test_pool_exhaustion_rejects_later_entries(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.rejects = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()  # all-in on whichever symbol comes first

            def on_order_rejected(self, ctx, event):
                self.rejects.append((ctx.symbol, event.reject_reason))

        data = {
            "AAA": _bars([100.0, 101.0, 102.0]),
            "BBB": _bars([50.0, 51.0, 52.0], start_ts=5_000_000_000),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())
        assert strategy.rejects == [("BBB", "ZeroSize")]

    def test_typed_orders_route_by_symbol(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fills = []

            def on_bar(self, ctx):
                # From AAA's first bar, place a resting limit on BBB.
                if ctx.symbol == "AAA" and ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(side="buy", price=49.0, units=100.0),
                        symbol="BBB",
                    )

            def on_order_filled(self, ctx, event):
                self.fills.append((ctx.symbol, event.price))

        data = {
            "AAA": _bars([100.0, 101.0, 102.0, 103.0]),
            "BBB": _bars(
                [50.0, 50.0, 48.5, 49.5], start_ts=5_000_000_000
            ),  # dips through 49 on its bar 2
        }
        strategy = S()
        result = run_portfolio_strategy(strategy, data, config=_zero_fee_config())
        assert strategy.fills, "expected the BBB limit to fill"
        symbol, price = strategy.fills[0]
        assert symbol == "BBB"
        assert price == pytest.approx(49.0)
        summary = {s.symbol: s for s in result.per_instrument}
        assert summary["BBB"].trades == 1
        assert summary["AAA"].trades == 0

    def test_cross_symbol_close_routing(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.symbol == "AAA" and ctx.idx == 0:
                    self.enter()
                # From BBB's bar 2, flatten AAA by symbol routing.
                if ctx.symbol == "BBB" and ctx.idx == 2:
                    if ctx.position_for("AAA") is not None:
                        self.close_position(symbol="AAA")

        data = {
            "AAA": _bars([100.0, 101.0, 102.0, 103.0]),
            "BBB": _bars([50.0, 51.0, 52.0, 53.0], start_ts=5_000_000_000),
        }
        result = run_portfolio_strategy(S, data, config=_zero_fee_config())
        summary = {s.symbol: s for s in result.per_instrument}
        assert summary["AAA"].trades == 1
        trades = result.result.trades()
        aaa = [t for t in trades if t.symbol == "AAA"][0]
        # Closed by the routed request, not by end-of-data finalization.
        assert aaa.exit_reason == "Signal"

    def test_deterministic_across_runs(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=0.3)
                elif ctx.idx == 3 and ctx.position is not None:
                    self.close_position()

        data = {
            "AAA": _bars([100.0, 101.0, 99.0, 102.0, 103.0]),
            "BBB": _bars([50.0, 49.0, 51.0, 52.0, 50.0], start_ts=5_000_000_000),
        }
        r1 = run_portfolio_strategy(S, data, config=_zero_fee_config())
        r2 = run_portfolio_strategy(S, data, config=_zero_fee_config())
        assert np.array_equal(r1.result.equity_curve(), r2.result.equity_curve())

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
