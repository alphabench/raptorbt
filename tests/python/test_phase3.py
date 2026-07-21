"""Behavioral tests: hedging ledger, margin accounts, stochastic fills,
adaptive bar-path model (0.5.0)."""

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


class TestHedging:
    def test_concurrent_long_and_short(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.seen = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.Market(side="buy", units=10.0))
                    self.submit_order(orders.Market(side="sell", units=5.0))
                if ctx.idx == 1:
                    self.seen = [(p.position_id, p.direction, p.size) for p in ctx.positions]
                    # Close only the short.
                    short = next(p for p in ctx.positions if p.direction == -1)
                    self.close_position(short.position_id)

        data = _bars([100.0, 101.0, 102.0, 103.0])
        strategy = S()
        result = run_strategy_backtest(
            strategy, **data, config=_zero_fee_config(), oms_type="hedging"
        )
        assert len(strategy.seen) == 2
        directions = sorted(d for _, d, _ in strategy.seen)
        assert directions == [-1, 1]
        # One closed trade (the short) + the long force-closed at end of data.
        trades = result.trades()
        assert len(trades) == 2
        assert {t.exit_reason for t in trades} == {"Signal", "EndOfData"}

    def test_netting_rejects_hedged_open(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.rejects = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.Market(side="buy", units=10.0))
                if ctx.idx == 1:
                    self.submit_order(orders.Market(side="buy", units=5.0))

            def on_order_rejected(self, ctx, event):
                self.rejects.append(event.reject_reason)

        data = _bars([100.0, 101.0, 102.0])
        strategy = S()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.rejects == ["position_open"]

    def test_per_position_stops_fire_independently(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Market(side="buy", units=10.0, stop_price=98.0)
                    )
                    self.submit_order(
                        orders.Market(side="sell", units=10.0, stop_price=104.0)
                    )

        # Bar 1 dips to 97.5: long's stop fires; short survives to end.
        data = _bars(
            [100.0, 99.0, 100.0, 101.0],
            lows=[99.5, 97.5, 99.5, 100.5],
            highs=[100.5, 100.0, 100.5, 101.5],
        )
        result = run_strategy_backtest(
            S, **data, config=_zero_fee_config(), oms_type="hedging"
        )
        trades = result.trades()
        assert len(trades) == 2
        stop_trades = [t for t in trades if t.exit_reason == "StopLoss"]
        assert len(stop_trades) == 1
        assert stop_trades[0].direction == 1


class TestMarginAccount:
    def test_leverage_scales_sizing(self):
        class EnterOnce(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()

        data = _bars([100.0, 101.0, 102.0])
        cash = run_strategy_backtest(EnterOnce, **data, config=_zero_fee_config())
        levered = run_strategy_backtest(
            EnterOnce, **data, config=_zero_fee_config(),
            account_type="margin", leverage=5.0,
        )
        # 5x leverage sizes ~5x the units of the fully-funded account.
        ratio = levered.trades()[0].size / cash.trades()[0].size
        assert ratio == pytest.approx(5.0, rel=1e-9)

    def test_margin_equity_marks_short_correctly(self):
        """A profitable short gains equity in margin mode (cash-mode quirk fixed)."""

        class ShortOnce(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.marks = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()
                else:
                    self.marks.append(ctx.equity)

        data = _bars([100.0, 90.0, 80.0])
        strategy = ShortOnce()
        run_strategy_backtest(
            strategy, **data, direction=-1, config=_zero_fee_config(),
            account_type="margin", leverage=1.0,
        )
        assert strategy.marks[-1] > strategy.marks[0] > 100_000.0 * 0.999

    def test_margin_call_halts_entries(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.margin_calls = 0
                self.rejects = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter()
                elif ctx.idx == 3 and ctx.position is not None:
                    self.close_position()
                elif ctx.idx == 4 and ctx.position is None:
                    self.enter()  # must reject: the margin call latched

            def on_margin_call(self, ctx, event):
                self.margin_calls += 1

            def on_order_rejected(self, ctx, event):
                self.rejects.append(event.reject_reason)

        # 10x leverage; a ~15% adverse move wipes >100% of margin equity.
        data = _bars([100.0, 95.0, 85.0, 85.0, 85.0])
        strategy = S()
        run_strategy_backtest(
            strategy, **data, config=_zero_fee_config(),
            account_type="margin", leverage=10.0,
        )
        assert strategy.margin_calls == 1
        assert "MarginCall" in strategy.rejects


class TestStochasticFill:
    def _limit_strategy(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fill_bars = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(orders.Limit(side="buy", price=99.5, units=10.0))

            def on_order_filled(self, ctx, event):
                self.fill_bars.append(event.idx)

        return S

    def test_prob_zero_never_fills(self):
        config = _zero_fee_config(fill_prob_limit=0.0)
        data = _bars([100.0] * 10, lows=[98.0] * 10)
        strategy = self._limit_strategy()()
        result = run_strategy_backtest(strategy, **data, config=config)
        assert strategy.fill_bars == []
        assert len(result.trades()) == 0

    def test_same_seed_same_fills(self):
        data = _bars([100.0] * 20, lows=[98.0] * 20)
        runs = []
        for _ in range(2):
            config = _zero_fee_config(fill_prob_limit=0.5, fill_seed=1234)
            strategy = self._limit_strategy()()
            run_strategy_backtest(strategy, **data, config=config)
            runs.append(tuple(strategy.fill_bars))
        assert runs[0] == runs[1]

        config = _zero_fee_config(fill_prob_limit=0.5, fill_seed=99)
        strategy = self._limit_strategy()()
        run_strategy_backtest(strategy, **data, config=config)
        # A different seed is allowed to fill on a different bar; the fill
        # itself must still eventually happen with 20 chances at p=0.5.
        assert strategy.fill_bars, "expected an eventual fill"

    def test_default_probs_are_deterministic_legacy(self):
        data = _bars([100.0] * 5, lows=[98.0] * 5)
        strategy = self._limit_strategy()()
        run_strategy_backtest(strategy, **data, config=_zero_fee_config())
        assert strategy.fill_bars == [1]  # first bar after submission


class TestBarPathModel:
    def _both_touched(self, adaptive: bool):
        """Long with stop 97 and target 103; one bar touches both."""

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(stop_price=97.0, target_price=103.0)

        config = _zero_fee_config(bar_path_adaptive=adaptive)
        # Bar 1: open 100, high 104, low 96 — both levels inside the range.
        # Down-candle (close 96.5 < open): adaptive assumes high visited first.
        data = _bars(
            [100.0, 96.5, 97.0],
            opens=[100.0, 100.0, 96.5],
            highs=[100.5, 104.0, 97.5],
            lows=[99.5, 96.0, 96.0],
        )
        result = run_strategy_backtest(S, **data, config=config)
        return result.trades()[0]

    def test_legacy_stop_first(self):
        trade = self._both_touched(adaptive=False)
        assert trade.exit_reason == "StopLoss"

    def test_adaptive_down_candle_hits_target_first(self):
        trade = self._both_touched(adaptive=True)
        assert trade.exit_reason == "TakeProfit"
        assert trade.exit_price == pytest.approx(103.0)
