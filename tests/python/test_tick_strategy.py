"""Tick-driven class contract (0.5.x).

Orders match against prints, not bars. Quotes are observation only. Bars
built from ticks are a view that feeds ``on_bar`` and indicators — nothing
executes on them.
"""

import numpy as np
import pytest

from raptorbt import (
    Indicator,
    BacktestConfig,
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
    config = BacktestConfig(**kwargs)
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

        data = {
            "AAA": _ticks(
                [100.0, 105.0, 110.0],
                bids=[99.0, 104.0, 109.0],
                asks=[101.0, 106.0, 111.0],
            )
        }
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
        data = {
            "AAA": _ticks(
                [100.0, 101.0, 102.0], bids=[0.0, 100.0, 0.0], asks=[0.0, 102.0, 0.0]
            )
        }
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

    def test_a_market_order_for_another_symbol_fills_on_that_symbols_next_print(self):
        """Ordinals are per instrument, and quotes consume them without
        stepping the kernel. An order placed for BBB from AAA's print
        carried AAA's ordinal and, whenever BBB's quotes had pushed its
        ordinals ahead, never met a print with exactly that ordinal: it sat
        working, unfilled and unacknowledged, to the end of the run."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fills = []

            def on_trade_tick(self, ctx, tick):
                if ctx.symbol == "AAA" and ctx.idx == 2:
                    self.submit_order(orders.Market(units=10, side="buy"), symbol="BBB")

            def on_order_filled(self, ctx, event):
                self.fills.append((ctx.symbol, event.price))

        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        for quoted in (False, True):
            bbb = (
                _ticks(prices, bids=[p - 1 for p in prices], asks=[p + 1 for p in prices])
                if quoted
                else _ticks(prices)
            )
            strategy = S()
            run_tick_strategy(
                strategy, {"AAA": _ticks(prices), "BBB": bbb}, config=_zero_fee_config()
            )
            # BBB's print at t=2 is dispatched after AAA's — the first trade
            # after the order was placed — so that is where it fills, with
            # or without quotes in BBB's tape.
            assert strategy.fills == [("BBB", 102.0)], (quoted, strategy.fills)

    def test_ltq_l1_sizes_and_oi_reach_the_hooks(self):
        """`ltq` is the print's size when present; the flow deltas stand in
        otherwise. `oi` rides the print and the L1 sizes ride the quote, as
        `nan` (never 0) when the feed carried none."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.prints, self.quotes = [], []

            def on_trade_tick(self, ctx, tick):
                self.prints.append((tick.size, tick.oi))

            def on_quote(self, ctx, quote):
                self.quotes.append((quote.bid_size, quote.ask_size))

        data = {
            "AAA": {
                **_ticks([100.0, 101.0], bids=[99.0, 100.0], asks=[101.0, 102.0]),
                "buy_qty_delta": np.array([5.0, 5.0]),
                "sell_qty_delta": np.array([2.0, 2.0]),
                "ltq": np.array([40.0, 0.0]),
                "oi": np.array([1500.0, 0.0]),
                "bid_qty": np.array([300.0, 0.0]),
            }
        }
        strategy = S()
        run_tick_strategy(strategy, data, config=_zero_fee_config())
        assert strategy.prints == [(40.0, 1500.0), (7.0, 0.0)]
        assert strategy.quotes[0][0] == 300.0
        assert all(np.isnan(v) for v in (strategy.quotes[0][1], *strategy.quotes[1]))

    def test_a_sized_quote_alone_lets_the_queue_model_hold_an_order(self):
        """No depth snapshot: the feed's best-bid size is what a resting
        limit joins behind. 300 displayed at 99.0; a 100 print does not
        reach us, the next 250 does."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.prints = 0
                self.fills = []

            def on_trade_tick(self, ctx, tick):
                self.prints += 1
                if self.prints == 1:
                    self.submit_order(orders.Limit(side="buy", price=99.0, units=10))

            def on_order_filled(self, ctx, event):
                # Which print the fill landed on: the 100 print (2nd) must
                # not reach us behind 300 displayed; the 250 print (3rd) does.
                self.fills.append(self.prints)

        data = {
            "AAA": {
                **_ticks([100.0, 99.0, 99.0], bids=[99.0, 99.0, 99.0], asks=[101.0] * 3),
                "buy_qty_delta": np.array([1.0, 100.0, 250.0]),
                "bid_qty": np.array([300.0, 300.0, 300.0]),
            }
        }
        config = _zero_fee_config()
        config.queue_fill_model = True
        strategy = S()
        run_tick_strategy(strategy, data, config=config)
        assert strategy.fills == [3], strategy.fills

    def test_partial_fills_span_prints_and_open_orders_shows_the_rest(self):
        """With partial_fills on, a 100-unit entry fills 40 then 60 across
        two prints and is ONE position; between them ctx.open_orders()
        shows the order working with filled_qty=40. The flag is detectable
        on BacktestConfig, so a stale wheel refuses a caller that needs it."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fills, self.working = [], []

            def on_trade_tick(self, ctx, tick):
                if ctx.idx == 0:
                    self.submit_order(orders.Market(units=100, side="buy"))
                self.working.append([(o.status, o.filled_qty, o.remaining) for o in ctx.open_orders()])

            def on_order_filled(self, ctx, event):
                self.fills.append(event.size)

        # The submission print itself sweeps the order (same-print market
        # semantics), so it carries the first 40; the next print the 60.
        data = {"AAA": {**_ticks([100.0, 110.0, 110.0, 110.0]), "ltq": np.array([40.0, 60.0, 5.0, 5.0])}}
        config = _zero_fee_config()
        assert hasattr(config, "partial_fills")
        config.partial_fills = True
        strategy = S()
        result = run_tick_strategy(strategy, data, config=config)
        assert strategy.fills == [40.0, 60.0]
        # Seen on the print after the first slice: 40 filled, 60 to go.
        assert strategy.working[1] == [("partially_filled", 40.0, 60.0)]
        assert strategy.working[2] == []
        trades = result.result.trades()
        assert len(trades) == 1  # closed at end of data as one position
        assert trades[0].size == pytest.approx(100.0)
        assert trades[0].entry_price == pytest.approx(106.0)

    def test_order_entry_latency_delays_the_first_eligible_print(self):
        """order_latency_ns=250ms: an order placed on the t=0 print cannot
        fill on the 100 ms or 200 ms prints and fills on the 300 ms one."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.prints = 0
                self.fill_on_print = []

            def on_trade_tick(self, ctx, tick):
                self.prints += 1
                if self.prints == 1:
                    self.submit_order(orders.Market(units=10, side="buy"))

            def on_order_filled(self, ctx, event):
                self.fill_on_print.append(self.prints)

        ticks = _ticks([100.0, 100.0, 100.0, 100.0], start_ts=0, step=100_000_000)
        config = _zero_fee_config()
        assert hasattr(config, "order_latency_ns")
        config.order_latency_ns = 250_000_000
        strategy = S()
        run_tick_strategy(strategy, {"AAA": ticks}, config=config)
        # Prints at 0, 100, 200, 300 ms: the fourth is the first past 250 ms.
        assert strategy.fill_on_print == [4], strategy.fill_on_print

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


def _depth(rows, start_ts=0, step=10):
    """rows: list of (bids, asks), each a list of (price, size)."""
    n = len(rows)
    width = max(max(len(b), len(a)) for b, a in rows)
    bp = np.zeros((n, width))
    bs = np.zeros((n, width))
    ap = np.zeros((n, width))
    asz = np.zeros((n, width))
    for i, (bids, asks) in enumerate(rows):
        for j, (price, size) in enumerate(bids):
            bp[i, j], bs[i, j] = price, size
        for j, (price, size) in enumerate(asks):
            ap[i, j], asz[i, j] = price, size
    return {
        "timestamps": np.arange(start_ts, start_ts + n * step, step, dtype=np.int64),
        "bid_prices": bp,
        "bid_sizes": bs,
        "ask_prices": ap,
        "ask_sizes": asz,
    }


class TestOrderBook:
    def test_on_order_book_fires_with_levels(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.books = []

            def on_order_book(self, ctx, book):
                self.books.append(book)

        depth = _depth([([(99.0, 300.0), (98.0, 200.0)], [(101.0, 400.0)])], start_ts=5)
        strategy = S()
        run_tick_strategy(
            strategy,
            {"AAA": _ticks([100.0], start_ts=10)},
            config=_zero_fee_config(),
            depth=depth_for("AAA", depth),
        )

        assert len(strategy.books) == 1
        book = strategy.books[0]
        assert book.best_bid == 99.0
        assert book.best_ask == 101.0
        assert book.spread == pytest.approx(2.0)
        assert book.symbol == "AAA"
        assert len(book.bids) == 2
        # 300 / (300 + 400)
        assert book.imbalance == pytest.approx(300.0 / 700.0)

    def test_ctx_book_persists_into_trade_handler(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.seen = []

            def on_trade_tick(self, ctx, tick):
                self.seen.append(ctx.book.best_bid if ctx.book else None)

        depth = _depth([([(99.0, 300.0)], [(101.0, 400.0)])], start_ts=5)
        strategy = S()
        run_tick_strategy(
            strategy,
            {"AAA": _ticks([100.0, 100.0], start_ts=10, step=10)},
            config=_zero_fee_config(),
            depth=depth_for("AAA", depth),
        )
        assert strategy.seen == [99.0, 99.0]

    def test_a_book_alone_produces_no_trades_or_equity_samples(self):
        class S(Strategy):
            def on_order_book(self, ctx, book):
                # Displayed size is intent; this rests until a print.
                self.submit_order(orders.Limit(side="buy", price=99.0, units=10.0))

        depth = _depth(
            [([(99.0, 300.0)], [(101.0, 400.0)]), ([(99.5, 300.0)], [(101.5, 400.0)])],
            start_ts=5,
        )
        result = run_tick_strategy(
            S(),
            {"AAA": _ticks([100.0], start_ts=100)},
            config=_zero_fee_config(),
            depth=depth_for("AAA", depth),
        )
        assert len(result.result.trades()) == 0
        # One print sampled equity; the two book updates did not.
        assert len(result.result.equity_curve()) == 1


class TestQueueFills:
    def _run(self, queue_model, print_sizes):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.sent = False

            def on_order_book(self, ctx, book):
                if not self.sent:
                    self.sent = True
                    self.submit_order(orders.Limit(side="buy", price=99.0, units=10.0))

        config = _zero_fee_config()
        config.queue_fill_model = queue_model
        prices = [99.0] * len(print_sizes)
        ticks = _ticks(prices, start_ts=100, step=10)
        ticks["buy_qty_delta"] = np.asarray(print_sizes, dtype=np.float64)
        depth = _depth([([(99.0, 300.0)], [(101.0, 400.0)])], start_ts=5)
        return run_tick_strategy(
            S(), {"AAA": ticks}, config=config, depth=depth_for("AAA", depth)
        )

    def test_order_waits_behind_displayed_size(self):
        # 300 displayed ahead; 50 + 50 does not reach us.
        result = self._run(queue_model=True, print_sizes=[50.0, 50.0])
        assert len(result.result.trades()) == 0

    def test_order_fills_once_the_queue_is_exhausted(self):
        # Cumulative 350 > 300 displayed ahead.
        result = self._run(queue_model=True, print_sizes=[150.0, 200.0])
        assert len(result.result.trades()) == 1

    def test_without_the_queue_model_the_first_print_fills(self):
        # The default path ignores size entirely.
        result = self._run(queue_model=False, print_sizes=[50.0, 50.0])
        assert len(result.result.trades()) == 1


def depth_for(symbol, arrays):
    return {symbol: arrays}
