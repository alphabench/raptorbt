"""Portfolio runs: routed modify_order, per-symbol indicators and composite
bars (0.5.x).

Single-instrument equivalents live in ``test_orders.py`` (modify),
``test_phase7.py`` (indicators) and ``test_phase5.py`` (composite bars).
These pin the portfolio versions, where every stream is per symbol.
"""

import numpy as np
import pytest

from raptorbt import Indicator, BacktestConfig, Strategy, run_portfolio_strategy
from raptorbt.strategy import orders


def _bars(closes, start_ts=0, step=1):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(start_ts, start_ts + n * step, step, dtype=np.int64),
        "open": closes.copy(),
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _zero_fee_config(**kwargs):
    config = BacktestConfig(**kwargs)
    config.fees = 0.0
    return config


class TestPortfolioModifyOrder:
    def test_modify_routes_to_the_owning_instrument(self):
        """A modify names only the client id; the runner routes it from the
        id map, and it must reach the symbol that owns the order."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.oid = None

            def on_bar(self, ctx):
                if ctx.symbol == "BBB" and ctx.idx == 0:
                    self.oid = self.submit_order(
                        orders.Limit(side="buy", price=40.0, units=10.0)
                    )
                if ctx.symbol == "BBB" and ctx.idx == 1:
                    self.modify_order(self.oid, limit_price=49.0)

        data = {
            # AAA never trades; it exists to prove routing does not leak.
            "AAA": _bars([100.0, 100.0, 100.0]),
            "BBB": _bars([50.0, 50.0, 48.5], start_ts=5_000_000_000),
        }
        result = run_portfolio_strategy(S(), data, config=_zero_fee_config())

        trades = result.result.trades()
        assert len(trades) == 1, "only the modified order should fill"
        assert trades[0].symbol == "BBB"
        assert trades[0].entry_price == pytest.approx(49.0)

    def test_modify_unknown_client_id_is_a_noop(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.modify_order("never-submitted", limit_price=1.0)

        data = {"AAA": _bars([100.0, 101.0])}
        # Mirrors cancel: an unknown id is skipped, not an error.
        run_portfolio_strategy(S(), data, config=_zero_fee_config())

    def test_modify_rejects_units_and_size_frac_together(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.oid = None

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.oid = self.submit_order(
                        orders.Limit(side="buy", price=40.0, units=10.0)
                    )
                if ctx.idx == 1:
                    self.modify_order(self.oid, units=5.0, size_frac=0.5)

        data = {"AAA": _bars([100.0, 100.0, 100.0])}
        with pytest.raises(ValueError, match="pass units or size_frac, not both"):
            run_portfolio_strategy(S(), data, config=_zero_fee_config())

    def test_modify_after_fill_returns_false(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.oid = None

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.oid = self.submit_order(
                        orders.Limit(side="buy", price=99.0, units=10.0)
                    )
                if ctx.idx == 2:
                    # Already filled on bar 1; modifying is a no-op, not a raise.
                    self.modify_order(self.oid, limit_price=95.0)

        data = {"AAA": _bars([100.0, 98.0, 98.0])}
        result = run_portfolio_strategy(S(), data, config=_zero_fee_config())
        assert len(result.result.trades()) == 1


class TestPerSymbolIndicators:
    def test_indicators_are_isolated_per_symbol(self):
        """The headline: each symbol's SMA reflects only its own closes.

        A single indicator fed interleaved bars from both symbols would
        average across them — the bug this routing exists to prevent.
        """

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.sma = {}
                self.seen = {}

            def on_start(self, ctx):
                self.sma = {
                    s: self.register_indicator(Indicator.sma(3), symbol=s)
                    for s in ctx.symbols
                }

            def on_bar(self, ctx):
                value = self.sma[ctx.symbol].value
                if value is not None:
                    self.seen.setdefault(ctx.symbol, []).append(value)

        data = {
            "AAA": _bars([100.0, 200.0, 300.0, 400.0]),
            "BBB": _bars([10.0, 20.0, 30.0, 40.0], start_ts=5_000_000_000),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        # SMA(3) of each symbol's own closes, nothing blended.
        assert strategy.seen["AAA"] == [pytest.approx(200.0), pytest.approx(300.0)]
        assert strategy.seen["BBB"] == [pytest.approx(20.0), pytest.approx(30.0)]

    def test_register_indicators_helper(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fast = {}
                self.values = {}

            def on_start(self, ctx):
                self.fast = self.register_indicators(
                    lambda: Indicator.sma(2), ctx.symbols
                )

            def on_bar(self, ctx):
                value = self.fast[ctx.symbol].value
                if value is not None:
                    self.values[ctx.symbol] = value

        data = {
            "AAA": _bars([100.0, 200.0]),
            "BBB": _bars([10.0, 20.0], start_ts=5_000_000_000),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        assert set(strategy.fast) == {"AAA", "BBB"}
        assert strategy.values["AAA"] == pytest.approx(150.0)
        assert strategy.values["BBB"] == pytest.approx(15.0)

    def test_unrouted_indicator_warns_and_sees_every_symbol(self):
        """Documented semantics: no symbol= means interleaved, with a warning."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.ind = None

            def on_start(self, ctx):
                self.ind = self.register_indicator(Indicator.sma(4))

        data = {
            "AAA": _bars([100.0, 100.0]),
            "BBB": _bars([200.0, 200.0], start_ts=5_000_000_000),
        }
        strategy = S()
        with pytest.warns(UserWarning, match="without symbol="):
            run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        # Four bars across both symbols: the mean blends them.
        assert strategy.ind.value == pytest.approx(150.0)

    def test_unknown_symbol_registration_raises(self):
        """A typo'd symbol would otherwise leave a silently dead indicator."""

        class S(Strategy):
            def on_start(self, ctx):
                self.ind = self.register_indicator(Indicator.sma(2), symbol="TYPO")

        data = {"AAA": _bars([100.0, 101.0])}
        with pytest.raises(ValueError, match="not in this run"):
            run_portfolio_strategy(S(), data, config=_zero_fee_config())

    def test_indicators_initialized_spans_symbols(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.ready_at = []

            def on_start(self, ctx):
                self.sma = {
                    s: self.register_indicator(Indicator.sma(2), symbol=s)
                    for s in ctx.symbols
                }

            def on_bar(self, ctx):
                self.ready_at.append(self.indicators_initialized())

        data = {
            "AAA": _bars([100.0, 101.0, 102.0]),
            "BBB": _bars([50.0, 51.0, 52.0], start_ts=5_000_000_000),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        # False until the slowest symbol's indicator has warmed.
        assert strategy.ready_at[0] is False
        assert strategy.ready_at[-1] is True

    def test_registrations_reset_between_runs(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.last = {}

            def on_start(self, ctx):
                self.sma = {
                    s: self.register_indicator(Indicator.sma(2), symbol=s)
                    for s in ctx.symbols
                }

            def on_bar(self, ctx):
                self.last[ctx.symbol] = self.sma[ctx.symbol].value

        data = {
            "AAA": _bars([100.0, 200.0]),
            "BBB": _bars([10.0, 20.0], start_ts=5_000_000_000),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())
        first = dict(strategy.last)
        # Re-running the same instance must not accumulate registrations.
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        assert len(strategy._indicators) == 2
        assert strategy.last == first


class TestPerSymbolCompositeBars:
    def test_composite_bars_are_built_per_symbol(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.bars = []

            def on_start(self, ctx):
                self.h = self.subscribe_bars(2, "s")

            def on_composite_bar(self, ctx, bar):
                self.bars.append((bar.symbol, ctx.symbol, bar.close))

        # 1-second bars; a 2s composite completes every two bars per symbol.
        data = {
            "AAA": _bars([100.0, 101.0, 102.0, 103.0], step=1_000_000_000),
            "BBB": _bars(
                [50.0, 51.0, 52.0, 53.0], start_ts=5_000_000_000, step=1_000_000_000
            ),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        assert strategy.bars, "composite bars must dispatch in portfolio runs"
        # bar.symbol and ctx.symbol agree, and closes come from that symbol.
        for bar_symbol, ctx_symbol, close in strategy.bars:
            assert bar_symbol == ctx_symbol
            if bar_symbol == "AAA":
                assert close >= 100.0
            else:
                assert close < 100.0

    def test_composite_dispatches_before_its_own_symbols_on_bar(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.events = []

            def on_start(self, ctx):
                self.h = self.subscribe_bars(2, "s")

            def on_composite_bar(self, ctx, bar):
                self.events.append(("composite", bar.symbol, bar.timestamp))

            def on_bar(self, ctx):
                self.events.append(("bar", ctx.symbol, ctx.bar.timestamp))

        data = {
            "AAA": _bars([100.0, 101.0, 102.0, 103.0], step=1_000_000_000),
            "BBB": _bars(
                [50.0, 51.0, 52.0, 53.0], start_ts=5_000_000_000, step=1_000_000_000
            ),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        composites = [row for row in strategy.events if row[0] == "composite"]
        assert composites, "composite bars must dispatch in portfolio runs"
        # Each composite is immediately followed by the on_bar of the same
        # symbol's bar that completed it, and closed no later than it.
        for i, (kind, symbol, ts) in enumerate(strategy.events):
            if kind != "composite":
                continue
            assert i + 1 < len(
                strategy.events
            ), "a composite must precede its own on_bar"
            next_kind, next_symbol, next_ts = strategy.events[i + 1]
            assert (next_kind, next_symbol) == ("bar", symbol)
            assert ts <= next_ts

    def test_indicator_on_a_composite_stream_per_symbol(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.values = {}

            def on_start(self, ctx):
                self.h = self.subscribe_bars(2, "s")
                self.trend = {
                    s: self.register_indicator(
                        Indicator.sma(2), stream_id=self.h, symbol=s
                    )
                    for s in ctx.symbols
                }

            def on_composite_bar(self, ctx, bar):
                value = self.trend[bar.symbol].value
                if value is not None:
                    self.values[bar.symbol] = value

        # Six bars per symbol so the 2s stream completes enough composites
        # for an SMA(2) over that stream to warm up.
        data = {
            "AAA": _bars(
                [100.0, 102.0, 104.0, 106.0, 108.0, 110.0], step=1_000_000_000
            ),
            "BBB": _bars(
                [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
                start_ts=5_000_000_000,
                step=1_000_000_000,
            ),
        }
        strategy = S()
        run_portfolio_strategy(strategy, data, config=_zero_fee_config())

        # Each symbol's composite-stream SMA stays in its own price range —
        # a shared indicator would blend them into the middle.
        assert strategy.values["AAA"] > 50.0
        assert strategy.values["BBB"] < 50.0

    def test_order_from_on_composite_bar_routes_to_that_symbol(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.sent = False

            def on_start(self, ctx):
                self.h = self.subscribe_bars(2, "s")

            def on_composite_bar(self, ctx, bar):
                # Unrouted submit: must target the completing symbol.
                if bar.symbol == "BBB" and not self.sent:
                    self.sent = True
                    self.submit_order(orders.Market(side="buy", units=5.0))

        data = {
            "AAA": _bars([100.0, 101.0, 102.0, 103.0], step=1_000_000_000),
            "BBB": _bars(
                [50.0, 51.0, 52.0, 53.0], start_ts=5_000_000_000, step=1_000_000_000
            ),
        }
        result = run_portfolio_strategy(S(), data, config=_zero_fee_config())

        trades = result.result.trades()
        assert len(trades) == 1
        assert trades[0].symbol == "BBB"
