"""Behavioral tests: full order-kind set, TIF completion, brackets (0.5.0)."""

import numpy as np
import pytest

from raptorbt import InstrumentSpec, PyBacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders


def _bars(closes, lows=None, highs=None, opens=None, start_ts=0, step=1):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(start_ts, start_ts + n * step, step, dtype=np.int64),
        "open": np.asarray(opens, dtype=np.float64) if opens is not None else closes.copy(),
        "high": np.asarray(highs, dtype=np.float64) if highs is not None else closes + 1.0,
        "low": np.asarray(lows, dtype=np.float64) if lows is not None else closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _zero_fee_config(**kwargs):
    config = PyBacktestConfig(**kwargs)
    config.fees = 0.0
    return config


class EventLog(Strategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.events = []

    def on_order_event(self, ctx, event):
        self.events.append((event.kind, event.idx, event.client_order_id))

    def kinds(self):
        return [k for k, _, _ in self.events]


class TestNewOrderKinds:
    def test_market_if_touched_buys_the_dip(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.MarketIfTouched(side="buy", trigger=98.0, units=10.0))

        data = _bars([100.0, 100.0, 98.5], lows=[99.5, 99.0, 97.5])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_filled"]
        assert result.trades()[0].entry_price == pytest.approx(98.0)

    def test_limit_if_touched_triggers_then_rests(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.LimitIfTouched(side="buy", trigger=98.0, price=97.5, units=10.0)
                    )

        data = _bars(
            [100.0, 98.5, 98.0],
            lows=[99.5, 97.8, 97.2],  # touch trigger on bar 1, limit fills bar 2
        )
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_triggered", "order_filled"]
        assert result.trades()[0].entry_price == pytest.approx(97.5)

    def test_market_to_limit_fills_at_next_open(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.MarketToLimit(side="buy", units=10.0))

        data = _bars([100.0, 101.0, 102.0], opens=[100.0, 100.7, 101.5])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert result.trades()[0].entry_price == pytest.approx(100.7)

    def test_at_open_and_at_close(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.Market(side="buy", units=10.0, tif="at_open"))
                if ctx.idx == 2 and ctx.position is not None:
                    self.close_position()

        data = _bars([100.0, 101.0, 102.0, 103.0], opens=[100.0, 100.4, 101.5, 102.5])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert result.trades()[0].entry_price == pytest.approx(100.4)

    def test_trailing_stop_market_bps(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()
                    # Protect with a 200bp trailing sell stop order.
                    self.submit_order(
                        orders.TrailingStopMarket(side="sell", offset=200.0, offset_kind="bps")
                    )

        # Rally to 110 then dip below 110*(1-2%)=107.8. Opens stay above the
        # trigger so the fill lands exactly on it (no gap-through).
        data = _bars(
            [100.0, 105.0, 110.0, 107.9],
            opens=[100.0, 104.5, 108.5, 108.2],
            highs=[100.5, 105.5, 110.0, 108.2],
            lows=[99.5, 104.0, 108.5, 106.5],
        )
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        trade = result.trades()[0]
        assert trade.exit_reason == "Order"
        assert trade.exit_price == pytest.approx(110.0 * 0.98)

    def test_trailing_ticks_requires_instrument(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.TrailingStopMarket(side="sell", offset=4.0, offset_kind="ticks")
                    )

        data = _bars([100.0, 101.0])
        with pytest.raises(ValueError, match="price_increment"):
            run_strategy_backtest(S, **data, config=_zero_fee_config())

        # With a tick size it converts to a price offset and works.
        spec = InstrumentSpec.equity("EQ", price_increment=0.5)
        run_strategy_backtest(S, **data, config=_zero_fee_config(), instrument=spec)


class TestFlags:
    def test_post_only_rejects_marketable_limit(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(side="buy", price=101.0, units=10.0, post_only=True)
                    )

        data = _bars([100.0, 100.5, 100.0], opens=[100.0, 100.2, 100.0])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_rejected"]
        assert strategy.events[-1][0] == "order_rejected"
        assert len(result.trades()) == 0

    def test_reduce_only_rejects_opening_fill(self):
        class S(EventLog):
            def __init__(self, config=None):
                super().__init__(config)
                self.rejects = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(side="buy", price=99.0, units=10.0, reduce_only=True)
                    )

            def on_order_rejected(self, ctx, event):
                self.rejects.append(event.reject_reason)

        data = _bars([100.0, 98.5, 99.0])
        strategy = S()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.rejects == ["reduce_only"]


class TestBrackets:
    def _bracket_strategy(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.ids = self.submit_bracket(
                        orders.Market(side="buy", size_frac=0.5),
                        stop_trigger=95.0,
                        target_price=106.0,
                    )

        return S

    def test_target_fill_cancels_stop(self):
        data = _bars(
            [100.0, 103.0, 107.0, 107.0],
            highs=[100.5, 103.5, 107.5, 107.5],
            lows=[99.5, 102.0, 105.0, 106.0],
        )
        strategy = self._bracket_strategy()()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        trade = result.trades()[0]
        assert trade.exit_price == pytest.approx(106.0)
        # Entry fill activates both legs; target fill cancels the stop.
        kinds = strategy.kinds()
        assert kinds.count("order_filled") == 2
        assert kinds.count("order_canceled") == 1
        assert kinds.count("order_accepted") == 3  # entry + both legs

    def test_stop_fill_cancels_target(self):
        data = _bars(
            [100.0, 98.0, 94.0, 95.0],
            highs=[100.5, 99.0, 96.0, 95.5],
            lows=[99.5, 97.5, 93.5, 94.5],
        )
        strategy = self._bracket_strategy()()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        trade = result.trades()[0]
        # Bar 2 opens at 94, gapping through the 95 trigger: fill at the open.
        assert trade.exit_price == pytest.approx(94.0)
        assert trade.exit_reason == "Order"
        assert strategy.kinds().count("order_canceled") == 1

    def test_legs_die_if_entry_never_fills(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    entry_id, self.stop_id, self.target_id = self.submit_bracket(
                        orders.Limit(side="buy", price=90.0, units=10.0),
                        stop_trigger=85.0,
                        target_price=95.0,
                    )
                    self.entry_id = entry_id
                if ctx.idx == 1:
                    self.cancel_order(self.entry_id)

        data = _bars([100.0, 100.0, 100.0, 100.0])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert len(result.trades()) == 0
        # Entry cancel is acknowledged, then both held legs die.
        assert strategy.kinds().count("order_canceled") == 3

    def test_golden_gate_still_exact(self):
        """Phase 4 machinery must not perturb the pinned baselines."""
        import json
        import sys
        from pathlib import Path

        here = Path(__file__).parent
        sys.path.insert(0, str(here / "golden"))
        from generate import GoldenSma, result_digest, thaw_inputs

        import raptorbt

        fixtures = json.loads((here / "golden" / "fixtures.json").read_text())
        ts, o, h, l, c, v, _, _ = thaw_inputs(fixtures["inputs"]["shared"])
        result = raptorbt.run_strategy_backtest(GoldenSma, ts, o, h, l, c, v)
        assert result_digest(result) == fixtures["class/sma_cross"]
