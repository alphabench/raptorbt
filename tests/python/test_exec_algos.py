"""Execution algorithms and session-calendar DAY expiry (0.5.x)."""

import numpy as np
import pytest

import raptorbt
from raptorbt.strategy import orders


def _flat_bars(n, price=100.0, step=1):
    return {
        "timestamps": np.arange(0, n * step, step, dtype=np.int64),
        "open": np.full(n, price),
        "high": np.full(n, price),
        "low": np.full(n, price),
        "close": np.full(n, price),
        "volume": np.ones(n),
    }


def _config(**kwargs):
    config = raptorbt.PyBacktestConfig(**kwargs)
    config.fees = 0.0
    return config


class _TwapStrategy(raptorbt.Strategy):
    def __init__(self, config=None, **twap):
        super().__init__(config)
        self.twap = twap
        self.fills = []
        self.started = []
        self.completed = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.client_id = self.submit_order(orders.Twap(**self.twap))

    def on_order_filled(self, ctx, event):
        self.fills.append((event.idx, event.client_order_id, event.size))

    def on_algo_started(self, ctx, event):
        self.started.append(event.client_order_id)

    def on_algo_completed(self, ctx, event):
        self.completed.append(event.order_id)


class TestTwap:
    def test_slices_release_one_per_interval(self):
        strategy = _TwapStrategy(side="buy", units=30.0, slices=3, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        assert [f[0] for f in strategy.fills] == [0, 1, 2]
        assert [f[2] for f in strategy.fills] == [10.0, 10.0, 10.0]

    def test_slice_client_ids_derive_from_the_parent(self):
        strategy = _TwapStrategy(side="buy", units=30.0, slices=3, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        parent = strategy.started[0]
        assert [f[1] for f in strategy.fills] == [f"{parent}#{i}" for i in range(3)]

    def test_slices_sum_to_the_requested_size(self):
        # 100 into 3 does not divide evenly; nothing may be lost or invented.
        strategy = _TwapStrategy(side="buy", units=100.0, slices=3, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        assert sum(f[2] for f in strategy.fills) == pytest.approx(100.0)

    def test_lifecycle_events_fire(self):
        strategy = _TwapStrategy(side="buy", units=20.0, slices=2, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(5), config=_config(), oms_type="hedging"
        )
        assert len(strategy.started) == 1
        assert len(strategy.completed) == 1

    def test_a_single_slice_is_a_plain_order(self):
        strategy = _TwapStrategy(side="buy", units=25.0, slices=1, every=1)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(3), config=_config(), oms_type="hedging"
        )
        assert len(strategy.fills) == 1
        assert strategy.fills[0][2] == 25.0

    def test_slicing_is_timed_not_counted_in_bars(self):
        # Bars one nanosecond apart: an interval of 10ns spans several of
        # them, so slices must not release once per bar.
        strategy = _TwapStrategy(side="buy", units=30.0, slices=3, every=10)
        raptorbt.run_strategy_backtest(
            strategy, **_flat_bars(6), config=_config(), oms_type="hedging"
        )
        assert len(strategy.fills) == 1, "only the first slice is due"


class TestTwapValidation:
    def test_size_frac_cannot_be_sliced(self):
        with pytest.raises(ValueError, match="explicit units"):
            orders.Twap(side="buy", size_frac=0.5, slices=2, every=1)

    def test_slices_must_be_positive(self):
        with pytest.raises(ValueError, match="slices must be"):
            orders.Twap(side="buy", units=10.0, slices=0, every=1)

    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="every must be"):
            orders.Twap(side="buy", units=10.0, slices=2, every=0)

    def test_every_bars_converts_to_a_duration(self):
        twap = orders.Twap(
            side="buy", units=10.0, slices=2, every_bars=3, bar_ns=60_000_000_000
        )
        assert twap.interval_ns == 180_000_000_000

    def test_every_bars_needs_a_bar_duration(self):
        with pytest.raises(ValueError, match="bar_ns"):
            orders.Twap(side="buy", units=10.0, slices=2, every_bars=3)


class TestDayExpiryTradingDate:
    def _run(self, offset_ns):
        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.expired = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(side="buy", price=1.0, units=1.0, tif="day")
                    )

            def on_order_expired(self, ctx, event):
                self.expired.append(event.idx)

        day = 20_468 * 86_400_000_000_000
        # 22:30 UTC (04:00 IST next date), then 00:30 UTC (06:00 IST, same
        # IST trading date but a new UTC date).
        ts = np.array(
            [day + 81_000_000_000_000, day + 86_400_000_000_000 + 1_800_000_000_000],
            dtype=np.int64,
        )
        bars = {
            "timestamps": ts,
            "open": np.full(2, 100.0),
            "high": np.full(2, 100.0),
            "low": np.full(2, 100.0),
            "close": np.full(2, 100.0),
            "volume": np.ones(2),
        }
        strategy = S()
        raptorbt.run_strategy_backtest(
            strategy, **bars, config=_config(session_tz_offset_ns=offset_ns)
        )
        return strategy.expired

    def test_utc_default_expires_on_the_utc_rollover(self):
        assert self._run(0) == [1], "the UTC date rolled"

    def test_an_ist_session_keeps_the_order_alive(self):
        ist = (5 * 3600 + 30 * 60) * 1_000_000_000
        assert self._run(ist) == [], "the IST trading date did not roll"


class TestPerSymbolClock:
    def test_a_timer_fires_for_every_symbol(self):
        """A single clock would fire once for whichever symbol's event
        crossed the threshold first, silently skipping the rest."""

        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fired = []

            def on_start(self, ctx):
                self.clock.set_timer("beat", interval_ns=2, start_ns=2)

            def on_time_event(self, ctx, event):
                self.fired.append((ctx.symbol, event.ts_fired))

        bars = _flat_bars(6)
        strategy = S()
        raptorbt.run_portfolio_strategy(
            strategy, {"AAA": bars, "BBB": dict(bars)}, config=_config()
        )

        symbols = {sym for sym, _ in strategy.fired}
        assert symbols == {"AAA", "BBB"}, f"only fired for {symbols}"
        assert len(strategy.fired) == 4, strategy.fired

    def test_an_alert_fires_once_per_symbol(self):
        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fired = []

            def on_start(self, ctx):
                self.clock.set_time_alert("once", at_ns=3)

            def on_time_event(self, ctx, event):
                self.fired.append(ctx.symbol)

        bars = _flat_bars(6)
        strategy = S()
        raptorbt.run_portfolio_strategy(
            strategy, {"AAA": bars, "BBB": dict(bars)}, config=_config()
        )
        assert sorted(strategy.fired) == ["AAA", "BBB"]

    def test_a_timer_fires_for_every_symbol_in_a_tick_run(self):
        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fired = []

            def on_start(self, ctx):
                self.clock.set_timer("beat", interval_ns=2, start_ns=2)

            def on_time_event(self, ctx, event):
                self.fired.append(ctx.symbol)

        def ticks(n):
            return {
                "timestamps": np.arange(n, dtype=np.int64),
                "ltp": np.full(n, 100.0),
            }

        strategy = S()
        raptorbt.run_tick_strategy(
            strategy, {"AAA": ticks(6), "BBB": ticks(6)}, config=_config()
        )
        assert {sym for sym in strategy.fired} == {"AAA", "BBB"}
        assert len(strategy.fired) == 4

    def test_a_single_symbol_run_is_unchanged(self):
        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.fired = 0

            def on_start(self, ctx):
                self.clock.set_timer("beat", interval_ns=2, start_ns=2)

            def on_time_event(self, ctx, event):
                self.fired += 1

        strategy = S()
        raptorbt.run_strategy_backtest(strategy, **_flat_bars(6), config=_config())
        assert strategy.fired == 2


class TestLimitSlippage:
    def _fill_price(self, limit_slippage):
        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.submit_order(
                        orders.Limit(side="buy", price=99.0, units=10.0)
                    )

        config = _config()
        config.limit_slippage = limit_slippage
        bars = _flat_bars(3, price=98.0)
        result = raptorbt.run_strategy_backtest(S(), **bars, config=config)
        return result.trades()[0].entry_price

    def test_limit_fills_at_the_limit_by_default(self):
        assert self._fill_price(0.0) == pytest.approx(99.0)

    def test_slippage_moves_the_fill_against_the_buyer(self):
        # 1% adverse on a 99.0 limit.
        assert self._fill_price(0.01) == pytest.approx(99.99)


class TestOptionSettlement:
    """An option's bars carry the option's price, so intrinsic value has to
    come from an underlying the strategy supplies."""

    def _run(self, underlying):
        spec = raptorbt.InstrumentSpec.option(
            "CE", expiration_ns=3, strike=100.0, right="call", lot_size=1.0
        )

        class S(raptorbt.Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=0.5)
                if underlying is not None:
                    ctx.set_underlying_price(underlying)

        # The option decays to 0.5 while spot stays well above the strike.
        px = np.array([7.0, 6.0, 0.5, 0.5])
        bars = {
            "timestamps": np.arange(4, dtype=np.int64),
            "open": px, "high": px, "low": px, "close": px,
            "volume": np.ones(4),
        }
        result = raptorbt.run_strategy_backtest(
            S(), **bars, config=_config(), instrument=spec
        )
        return result.trades()[0]

    def test_without_an_underlying_it_settles_at_its_own_close(self):
        assert self._run(None).exit_price == pytest.approx(0.5)

    def test_with_an_underlying_it_settles_to_intrinsic(self):
        # Spot 112 against a 100 strike is worth 12, not the 0.5 the
        # contract last printed at.
        assert self._run(112.0).exit_price == pytest.approx(12.0)

    def test_an_out_of_the_money_option_settles_worthless(self):
        assert self._run(95.0).exit_price == pytest.approx(0.0)

    def test_the_underlying_routes_per_symbol_in_portfolio_runs(self):
        spec = raptorbt.InstrumentSpec.option(
            "CE", expiration_ns=3, strike=100.0, right="call", lot_size=1.0
        )

        class S(raptorbt.Strategy):
            def on_bar(self, ctx):
                if ctx.symbol == "CE":
                    if ctx.idx == 0:
                        self.enter(size_frac=0.5)
                    ctx.set_underlying_price(112.0)

        px = np.array([7.0, 6.0, 0.5, 0.5])
        opt = {
            "timestamps": np.arange(4, dtype=np.int64),
            "open": px, "high": px, "low": px, "close": px,
            "volume": np.ones(4),
        }
        result = raptorbt.run_portfolio_strategy(
            S(),
            {"CE": opt, "OTHER": _flat_bars(4)},
            config=_config(),
            instruments={"CE": spec},
        )
        settled = [t for t in result.result.trades() if t.symbol == "CE"]
        assert settled[0].exit_price == pytest.approx(12.0)


class TestSettlementFees:
    def _run(self, settlement_fee):
        spec = raptorbt.InstrumentSpec.futures_contract(
            "FUT", expiration_ns=3, lot_size=1.0
        )
        spec.settlement_fee = settlement_fee

        class S(raptorbt.Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=0.5)

        result = raptorbt.run_strategy_backtest(
            S(), **_flat_bars(5), config=_config(), instrument=spec
        )
        return result.trades()[0]

    def test_settlement_is_free_by_default(self):
        assert self._run(0.0).fees == pytest.approx(0.0)

    def test_a_settlement_fee_is_charged(self):
        assert self._run(0.01).fees > 0.0


class TestForcedLiquidation:
    def _run(self, liquidate):
        class S(raptorbt.Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.calls = 0

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=1.0)

            def on_margin_call(self, ctx, event):
                self.calls += 1

        config = _config()
        config.liquidate_on_margin_call = liquidate
        bars = {
            "timestamps": np.arange(4, dtype=np.int64),
            "open": np.array([100.0, 40.0, 40.0, 40.0]),
            "high": np.array([100.0, 40.0, 40.0, 40.0]),
            "low": np.array([100.0, 40.0, 40.0, 40.0]),
            "close": np.array([100.0, 40.0, 40.0, 40.0]),
            "volume": np.ones(4),
        }
        strategy = S()
        result = raptorbt.run_strategy_backtest(
            strategy, **bars, config=config, account_type="margin", leverage=50.0
        )
        return strategy, result

    def test_a_margin_call_only_halts_by_default(self):
        strategy, result = self._run(False)
        assert strategy.calls >= 1
        # The position survives to end-of-data.
        assert result.trades()[0].exit_reason == "EndOfData"

    def test_liquidation_closes_on_the_call(self):
        strategy, result = self._run(True)
        assert strategy.calls >= 1
        assert result.trades()[0].exit_reason == "Liquidation"
