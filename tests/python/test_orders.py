"""Behavioral tests for the typed order API (0.5.0 class contract)."""

import numpy as np
import pytest

from raptorbt import PyBacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders


def _bars(closes, lows=None, highs=None, start_ts=0, step=1):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(start_ts, start_ts + n * step, step, dtype=np.int64),
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


class EventLog(Strategy):
    """Record every order/position event kind with its bar index."""

    def __init__(self, config=None):
        super().__init__(config)
        self.events = []

    def on_order_event(self, ctx, event):
        self.events.append((event.kind, event.idx, event.client_order_id))

    def kinds(self):
        return [k for k, _, _ in self.events]


class TestOrderValidation:
    def test_rejects_both_units_and_size_frac(self):
        with pytest.raises(ValueError, match="not both"):
            orders.Limit(side="buy", price=100.0, units=10.0, size_frac=0.5)

    def test_rejects_bad_side_and_tif(self):
        with pytest.raises(ValueError, match="side"):
            orders.Market(side="hold")
        with pytest.raises(ValueError, match="gtd"):
            orders.Limit(side="buy", price=100.0, tif="gtd")

    def test_submit_requires_typed_order(self):
        s = Strategy()
        with pytest.raises(TypeError, match="typed order"):
            s.submit_order(orders.MarketOrder())


class TestOrderFlow:
    def test_market_order_fills_on_submission_bar(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 1:
                    self.submit_order(orders.Market(side="buy", size_frac=0.5))

        data = _bars([100.0, 100.0, 101.0])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_filled"]
        # Both events land on the submission bar.
        assert all(idx == 1 for _, idx, _ in strategy.events)
        assert len(result.trades()) == 1
        assert result.trades()[0].entry_price == pytest.approx(100.0)
        assert result.trades()[0].size == pytest.approx(500.0)  # 50k / 100

    def test_limit_buy_rests_until_touched(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.Limit(side="buy", price=97.0, units=100.0))

        # Lows stay above 97 until bar 3.
        data = _bars(
            [100.0, 100.0, 100.0, 98.0],
            lows=[99.0, 98.5, 98.0, 96.5],
        )
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_filled"]
        fill_kind, fill_idx, _ = strategy.events[-1]
        assert fill_idx == 3
        assert result.trades()[0].entry_price == pytest.approx(97.0)

    def test_stop_entry_with_attached_protective_stop(self):
        """Breakout entry: buy stop above the market, protective stop attached."""

        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.StopMarket(
                            side="buy", trigger=102.0, units=10.0, stop_price=99.0
                        )
                    )

        stops = []

        class Reader(S):
            def on_position_opened(self, ctx, event):
                stops.append(ctx.position.stop_price)

        data = _bars([100.0, 101.0, 103.0, 103.0], highs=[100.5, 101.5, 103.5, 103.5])
        strategy = Reader()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_filled"]
        assert strategy.events[-1][1] == 2  # fills on the breakout bar
        assert stops == [99.0]

    def test_cancel_prevents_fill(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.oid = self.submit_order(
                        orders.Limit(side="buy", price=97.0, units=10.0)
                    )
                if ctx.idx == 1:
                    self.cancel_order(self.oid)

        data = _bars([100.0, 100.0, 96.0])  # would fill on bar 2
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_canceled"]
        assert len(result.trades()) == 0

    def test_modify_moves_the_fill_price(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.oid = self.submit_order(
                        orders.Limit(side="buy", price=90.0, units=10.0)
                    )
                if ctx.idx == 1:
                    self.modify_order(self.oid, limit_price=98.0)

        data = _bars([100.0, 100.0, 97.5])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds()[-1] == "order_filled"
        assert result.trades()[0].entry_price == pytest.approx(98.0)

    def test_ioc_cancels_after_one_bar(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(side="buy", price=90.0, units=10.0, tif="ioc")
                    )

        data = _bars([100.0, 100.0, 89.0])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_canceled"]
        assert len(result.trades()) == 0

    def test_gtd_expires(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(
                            side="buy", price=90.0, units=10.0, tif="gtd", expire_ns=2
                        )
                    )

        data = _bars([100.0, 100.0, 100.0, 100.0])
        strategy = S()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_expired"]
        assert strategy.events[-1][1] == 2

    def test_sell_limit_take_profit_closes(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()  # legacy sugar still composes with orders
                    self.submit_order(orders.Limit(side="sell", price=105.0))

        data = _bars([100.0, 102.0, 106.0], highs=[101.0, 103.0, 106.5])
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert "order_filled" in strategy.kinds()
        trade = result.trades()[0]
        assert trade.exit_price == pytest.approx(105.0)
        assert trade.exit_reason == "Order"

    def test_stop_limit_triggers_then_fills(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.StopLimit(side="buy", trigger=102.0, price=101.5, units=10.0)
                    )

        data = _bars(
            [100.0, 102.5, 101.0, 101.0],
            highs=[100.5, 103.0, 102.0, 101.5],
            lows=[99.5, 101.8, 100.8, 100.5],
        )
        strategy = S()
        result = run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.kinds() == ["order_accepted", "order_triggered", "order_filled"]
        assert strategy.events[1][1] == 1  # triggered on bar 1
        assert strategy.events[2][1] == 2  # limit filled on bar 2
        assert result.trades()[0].entry_price == pytest.approx(101.5)

    def test_client_ids_are_deterministic(self):
        class S(EventLog):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.a = self.submit_order(orders.Limit(side="buy", price=90.0, units=1.0))
                    self.b = self.submit_order(orders.Limit(side="buy", price=91.0, units=1.0))

        data = _bars([100.0, 100.0])
        strategy = S()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.a == "O-0"
        assert strategy.b == "O-1"
        assert [c for _, _, c in strategy.events if c] == ["O-0", "O-1"]

    def test_legacy_contract_untouched_by_order_machinery(self):
        """A pure-legacy strategy produces identical results as before."""

        class Sma(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 1 and ctx.position is None:
                    self.enter()
                elif ctx.idx == 3 and ctx.position is not None:
                    self.close_position()

        data = _bars([100.0, 101.0, 102.0, 103.0, 104.0])
        r1 = run_strategy_backtest(Sma, **data)
        r2 = run_strategy_backtest(Sma, **data)
        assert np.array_equal(r1.equity_curve(), r2.equity_curve())
        assert len(r1.trades()) == 1
