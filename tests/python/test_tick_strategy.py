"""Tick-driven class contract (0.5.x).

Orders match against prints, not bars. Quotes are observation only. Bars
built from ticks are a view that feeds ``on_bar`` and indicators — nothing
executes on them.
"""

import numpy as np
import pytest

from raptorbt import (
    Indicator,
    PyBacktestConfig,
    Strategy,
    run_portfolio_strategy,
    run_tick_strategy,
)
from raptorbt.strategy import orders


def _ticks(prices, bids=None, asks=None, start_ts=0, step=1):
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    out = {
        "timestamps": np.arange(start_ts, start_ts + n * step, step, dtype=np.int64),
        "ltp": prices,
    }
    if bids is not None:
        out["bid"] = np.asarray(bids, dtype=np.float64)
    if asks is not None:
        out["ask"] = np.asarray(asks, dtype=np.float64)
    return out


def _zero_fee_config(**kwargs):
    config = PyBacktestConfig(**kwargs)
    config.fees = 0.0
    return config


class TestTickDispatch:
    def test_trade_and_quote_hooks_fire_in_feed_order(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.seen = []

            def on_trade_tick(self, ctx, tick):
                self.seen.append(("trade", tick.timestamp, tick.price))

            def on_quote(self, ctx, quote):
                self.seen.append(("quote", quote.timestamp, quote.bid))

        data = {"AAA": _ticks([100.0, 101.0], bids=[99.0, 100.0], asks=[101.0, 102.0])}
        strategy = S()
        run_tick_strategy(strategy, data, config=_zero_fee_config())

        # A row's print precedes its quote: the book state followed the trade.
        assert strategy.seen == [
            ("trade", 0, 100.0),
            ("quote", 0, 99.0),
            ("trade", 1, 101.0),
            ("quote", 1, 100.0),
        ]

    def test_best_bid_in_on_trade_tick_is_the_pre_print_book(self):
        """Reading this row's quote inside on_trade_tick would be lookahead:
        it is the book the print itself moved."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.observed = []

            def on_trade_tick(self, ctx, tick):
                self.observed.append(ctx.best_bid)

        data = {"AAA": _ticks([100.0, 105.0, 110.0], bids=[99.0, 104.0, 109.0], asks=[101.0, 106.0, 111.0])}
        strategy = S()
        run_tick_strategy(strategy, data, config=_zero_fee_config())

        # No book before the first print; then always the previous row's bid.
        assert strategy.observed == [None, 99.0, 104.0]

    def test_rows_without_a_book_produce_no_quote(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.quotes = 0
                self.trades = 0

            def on_trade_tick(self, ctx, tick):
                self.trades += 1

            def on_quote(self, ctx, quote):
                self.quotes += 1

        # Only the middle row carries both sides of the book.
        data = {"AAA": _ticks([100.0, 101.0, 102.0], bids=[0.0, 100.0, 0.0], asks=[0.0, 102.0, 0.0])}
        strategy = S()
        run_tick_strategy(strategy, data, config=_zero_fee_config())

        assert strategy.trades == 3
        assert strategy.quotes == 1

    def test_on_bar_never_fires_without_primary_bars(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.bars = 0

            def on_bar(self, ctx):
                self.bars += 1

        data = {"AAA": _ticks([100.0, 101.0, 102.0])}
        strategy = S()
        run_tick_strategy(strategy, data, config=_zero_fee_config())
        assert strategy.bars == 0


class TestTickExecution:
    def test_market_entry_fills_at_the_print(self):
        class S(Strategy):
            def on_trade_tick(self, ctx, tick):
                if ctx.idx == 0:
                    self.enter(size_frac=0.5)

        data = {"AAA": _ticks([100.0, 110.0])}
        result = run_tick_strategy(S(), data, config=_zero_fee_config())

        trades = result.result.trades()
        assert len(trades) == 1
        assert trades[0].entry_price == pytest.approx(100.0)
        assert trades[0].exit_price == pytest.approx(110.0)  # force-closed

    def test_limit_from_a_quote_rests_and_fills_on_a_later_print(self):
        """Quotes do not fill. The next print at that price is the evidence."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.submitted = False

            def on_quote(self, ctx, quote):
                if not self.submitted:
                    self.submitted = True
                    self.submit_order(orders.Limit(side="buy", price=95.0, units=10.0))

        # The quote straddles 95 but nothing trades there until the last row.
        data = {
            "AAA": _ticks(
                [100.0, 100.0, 94.0],
                bids=[94.0, 94.0, 93.0],
                asks=[96.0, 96.0, 95.0],
            )
        }
        result = run_tick_strategy(S(), data, config=_zero_fee_config())

        trades = result.result.trades()
        assert len(trades) == 1
        assert trades[0].entry_price == pytest.approx(95.0)

    def test_quotes_do_not_lengthen_the_equity_curve(self):
        """Metrics must not shift with feed verbosity."""

        class S(Strategy):
            def on_trade_tick(self, ctx, tick):
                if ctx.idx == 0:
                    self.enter(size_frac=0.5)

        prices = [100.0, 101.0, 102.0]
        with_quotes = run_tick_strategy(
            S(),
            {"AAA": _ticks(prices, bids=[99.0] * 3, asks=[101.0] * 3)},
            config=_zero_fee_config(),
        )
        without = run_tick_strategy(
            S(), {"AAA": _ticks(prices)}, config=_zero_fee_config()
        )

        assert np.array_equal(
            np.asarray(with_quotes.result.equity_curve()),
            np.asarray(without.result.equity_curve()),
        )
        assert with_quotes.metrics.total_return_pct == pytest.approx(
            without.metrics.total_return_pct
        )

    def test_agrees_with_a_bar_run_when_each_bar_has_one_print(self):
        """Cross-validation against the golden-covered bar path.

        One print per bar, at the close, with no intra-bar range: the two
        runners must reach the same trades.
        """

        class TickS(Strategy):
            def on_trade_tick(self, ctx, tick):
                if ctx.idx == 0:
                    self.enter(size_frac=0.5)

        class BarS(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=0.5)

        prices = [100.0, 104.0, 108.0]
        tick_result = run_tick_strategy(
            TickS(), {"AAA": _ticks(prices)}, config=_zero_fee_config()
        )
        flat = np.asarray(prices, dtype=np.float64)
        bar_result = run_portfolio_strategy(
            BarS(),
            {
                "AAA": {
                    "timestamps": np.arange(len(prices), dtype=np.int64),
                    "open": flat,
                    "high": flat,
                    "low": flat,
                    "close": flat,
                    "volume": np.zeros(len(prices)),
                }
            },
            config=_zero_fee_config(),
        )

        tick_trades = tick_result.result.trades()
        bar_trades = bar_result.result.trades()
        assert len(tick_trades) == len(bar_trades) == 1
        assert tick_trades[0].entry_price == pytest.approx(bar_trades[0].entry_price)
        assert tick_trades[0].exit_price == pytest.approx(bar_trades[0].exit_price)
        assert tick_trades[0].pnl == pytest.approx(bar_trades[0].pnl)

    def test_max_positions_gates_tick_entries_across_symbols(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.rejects = []

            def on_trade_tick(self, ctx, tick):
                self.enter(size_frac=0.2)

            def on_order_rejected(self, ctx, event):
                self.rejects.append(event.reject_reason)

        data = {
            "AAA": _ticks([100.0, 101.0]),
            "BBB": _ticks([50.0, 51.0], start_ts=5_000_000_000),
        }
        strategy = S()
        result = run_tick_strategy(
            strategy, data, config=_zero_fee_config(max_positions=1)
        )

        assert "MaxPositions" in strategy.rejects
        assert result.rejected_entries > 0


class TestBarsFromTicks:
    def test_primary_bars_dispatch_on_bar(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.bars = []

            def on_bar(self, ctx):
                self.bars.append((ctx.bar.timestamp, ctx.bar.close))

        # 2-tick bars over six prints => three completed bars.
        data = {"AAA": _ticks([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])}
        strategy = S()
        run_tick_strategy(
            strategy, data, config=_zero_fee_config(), primary_bars=(2, "tick")
        )

        assert len(strategy.bars) == 3
        assert [close for _, close in strategy.bars] == [101.0, 103.0, 105.0]

    def test_indicators_are_fed_by_primary_bars(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.values = []

            def on_start(self, ctx):
                self.sma = self.register_indicator(Indicator.sma(2), symbol="AAA")

            def on_bar(self, ctx):
                if self.sma.value is not None:
                    self.values.append(self.sma.value)

        data = {"AAA": _ticks([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])}
        strategy = S()
        run_tick_strategy(
            strategy, data, config=_zero_fee_config(), primary_bars=(2, "tick")
        )

        # Bars close at 102, 106, 110; SMA(2) over those.
        assert strategy.values == [pytest.approx(104.0), pytest.approx(108.0)]

    def test_composite_subscriptions_work_on_ticks(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.composites = []

            def on_start(self, ctx):
                self.h = self.subscribe_bars(3, "tick")

            def on_composite_bar(self, ctx, bar):
                self.composites.append((bar.symbol, bar.close))

        data = {"AAA": _ticks([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])}
        strategy = S()
        run_tick_strategy(strategy, data, config=_zero_fee_config())

        assert strategy.composites == [("AAA", 102.0), ("AAA", 105.0)]
