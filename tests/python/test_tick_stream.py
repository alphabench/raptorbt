"""Live tick stream (0.5.x).

The stream drives the same dispatch loop as ``run_tick_strategy``, so a
push-per-row feed must land on the exact numbers a batch replay of the same
rows produces. Warmup bars merge ahead of the first push and execute.
"""

import numpy as np
import pytest

from raptorbt import (
    BacktestConfig,
    Strategy,
    TickStrategyStream,
    run_strategy_backtest,
    run_tick_strategy,
)


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


class EnterOnce(Strategy):
    """Enter on the first print, close on a later threshold."""

    def __init__(self, config=None):
        super().__init__(config)
        self.entered = False

    def on_trade_tick(self, ctx, tick):
        if not self.entered:
            self.entered = True
            self.enter(size_frac=0.5)
        elif tick.price >= 110.0 and ctx.position is not None:
            self.close_position()


class TestStreamMatchesBatch:
    def test_push_per_row_equals_batch_replay(self):
        prices = [100.0, 104.0, 111.0, 108.0]
        bids = [99.0, 0.0, 110.0, 0.0]
        asks = [101.0, 0.0, 112.0, 0.0]
        data = {"AAA": _ticks(prices, bids=bids, asks=asks)}

        batch = run_tick_strategy(EnterOnce(), data, config=_zero_fee_config())

        stream = TickStrategyStream(EnterOnce(), ["AAA"], config=_zero_fee_config())
        for ts, (ltp, bid, ask) in enumerate(zip(prices, bids, asks)):
            stream.push_tick("AAA", ts, ltp, bid, ask)
        streamed = stream.finish()

        np.testing.assert_array_equal(
            batch.result.equity_curve(), streamed.result.equity_curve()
        )
        assert len(batch.result.trades()) == len(streamed.result.trades()) == 1
        assert batch.per_instrument[0].pnl == streamed.per_instrument[0].pnl

    def test_hooks_fire_synchronously_inside_push(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.seen = []

            def on_trade_tick(self, ctx, tick):
                self.seen.append(("trade", tick.timestamp, tick.price))

            def on_quote(self, ctx, quote):
                self.seen.append(("quote", quote.timestamp, quote.bid))

        strategy = S()
        stream = TickStrategyStream(strategy, ["AAA"], config=_zero_fee_config())
        appended = stream.push_tick("AAA", 5, 100.0, 99.0, 101.0)
        assert appended == 2
        # Both hooks already fired: the print, then the book that followed it.
        assert strategy.seen == [("trade", 5, 100.0), ("quote", 5, 99.0)]

    def test_position_readable_between_pushes(self):
        stream = TickStrategyStream(EnterOnce(), ["AAA"], config=_zero_fee_config())
        stream.push_tick("AAA", 0, 100.0)
        assert stream.positions("AAA"), "entry filled on a later print"

        stream.push_tick("AAA", 1, 111.0)
        assert not stream.positions("AAA"), "threshold close should have filled"
        result = stream.finish()
        assert len(result.result.trades()) == 1
        assert result.per_instrument[0].pnl > 0.0


class TestWarmupBars:
    def test_warmup_primes_indicators_before_first_push(self):
        from raptorbt import Indicator

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.sma_at_first_tick = None

            def on_start(self, ctx):
                self.sma = self.register_indicator(Indicator.sma(3))

            def on_trade_tick(self, ctx, tick):
                if self.sma_at_first_tick is None:
                    self.sma_at_first_tick = self.sma.value

        closes = [100.0, 102.0, 104.0]
        warmup = {
            "AAA": {
                "timestamps": np.arange(3, dtype=np.int64),
                "open": np.array(closes),
                "high": np.array(closes),
                "low": np.array(closes),
                "close": np.array(closes),
                "volume": np.ones(3),
            }
        }
        strategy = S()
        stream = TickStrategyStream(
            strategy, ["AAA"], config=_zero_fee_config(), warmup_bars=warmup
        )
        stream.push_tick("AAA", 10, 105.0)
        assert strategy.sma_at_first_tick == pytest.approx(102.0)

    def test_warmup_bars_execute(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.position is None:
                    self.enter(size_frac=0.5)

        warmup = {
            "AAA": {
                "timestamps": np.arange(2, dtype=np.int64),
                "open": np.array([100.0, 101.0]),
                "high": np.array([100.0, 101.0]),
                "low": np.array([100.0, 101.0]),
                "close": np.array([100.0, 101.0]),
                "volume": np.ones(2),
            }
        }
        stream = TickStrategyStream(
            S(), ["AAA"], config=_zero_fee_config(), warmup_bars=warmup
        )
        assert stream.positions("AAA"), "a warmup bar carried the entry"


class SmaCross(Strategy):
    """Bar-style strategy: property reads only, as written for the bar path."""

    FAST, SLOW = 3, 5

    def __init__(self, config=None):
        super().__init__(config)
        self.closes = []

    def on_bar(self, ctx):
        self.closes.append(ctx.bar.close)
        if len(self.closes) < self.SLOW:
            return
        fast = sum(self.closes[-self.FAST :]) / self.FAST
        slow = sum(self.closes[-self.SLOW :]) / self.SLOW
        if fast > slow and ctx.position is None:
            self.enter(size_frac=0.5)
        elif fast < slow and ctx.position is not None:
            self.close_position()


class TestContextParity:
    """A bar-style strategy must behave identically on both context types.

    Regression for the 0.5.0 API wart where ``PortfolioContext.position``
    was a method: ``ctx.position is None`` saw a truthy bound method on the
    live stream, so a bar-style strategy silently never entered.
    """

    CLOSES = [
        100.0,
        101.0,
        99.0,
        98.0,
        97.0,
        99.0,
        103.0,
        106.0,
        108.0,
        107.0,
        104.0,
        100.0,
        97.0,
        96.0,
        98.0,
        102.0,
        105.0,
        107.0,
    ]

    def _bars(self):
        closes = np.asarray(self.CLOSES, dtype=np.float64)
        return {
            "timestamps": np.arange(len(closes), dtype=np.int64),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.ones(len(closes)),
        }

    def test_bar_style_strategy_identical_on_backtest_and_stream(self):
        bars = self._bars()
        batch = run_strategy_backtest(
            SmaCross(),
            bars["timestamps"],
            bars["open"],
            bars["high"],
            bars["low"],
            bars["close"],
            bars["volume"],
            symbol="AAA",
            config=_zero_fee_config(),
        )

        stream = TickStrategyStream(SmaCross(), ["AAA"], config=_zero_fee_config())
        for i in range(len(bars["timestamps"])):
            stream.push_bar(
                "AAA",
                int(bars["timestamps"][i]),
                bars["open"][i],
                bars["high"][i],
                bars["low"][i],
                bars["close"][i],
                bars["volume"][i],
            )
        streamed = stream.finish()

        batch_trades = batch.trades()
        stream_trades = streamed.result.trades()
        assert (
            len(batch_trades) == len(stream_trades) > 0
        ), "bar-style strategy must trade on the live stream too"
        for bt, st in zip(batch_trades, stream_trades):
            assert bt.entry_idx == st.entry_idx
            assert bt.entry_price == st.entry_price
            assert bt.exit_price == st.exit_price
            assert bt.pnl == st.pnl


class TestLifecycle:
    def test_finished_stream_refuses_pushes(self):
        stream = TickStrategyStream(EnterOnce(), ["AAA"], config=_zero_fee_config())
        stream.push_tick("AAA", 0, 100.0)
        stream.finish()
        with pytest.raises(RuntimeError, match="finished"):
            stream.push_tick("AAA", 1, 101.0)
        with pytest.raises(RuntimeError, match="finished"):
            stream.finish()

    def test_unknown_symbol_raises(self):
        stream = TickStrategyStream(EnterOnce(), ["AAA"], config=_zero_fee_config())
        with pytest.raises(KeyError):
            stream.push_tick("BBB", 0, 100.0)
