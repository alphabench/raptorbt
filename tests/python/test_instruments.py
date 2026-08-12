"""Behavioral tests for instrument market definitions (InstrumentSpec)."""

import numpy as np
import pytest

import raptorbt
from raptorbt import InstrumentSpec, BacktestConfig, Strategy, run_strategy_backtest


def _bars(closes, start_ts=0, step=1):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    ts = np.arange(start_ts, start_ts + n * step, step, dtype=np.int64)
    return {
        "timestamps": ts,
        "open": closes.copy(),
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


class EnterOnce(Strategy):
    """Enter on the first bar and hold."""

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.enter()


class EnterEveryBar(Strategy):
    def on_bar(self, ctx):
        if ctx.position is None:
            self.enter()


def _zero_fee_config():
    config = BacktestConfig()
    config.fees = 0.0
    return config


class TestConstructors:
    def test_equity_defaults(self):
        spec = InstrumentSpec.equity("RELIANCE")
        assert spec.kind == "cash"
        assert spec.lot_size == 1.0
        assert spec.multiplier == 1.0
        # Market-neutral default: no tick grid unless the user declares one.
        assert spec.price_increment == 0.0
        assert spec.tradable

    def test_futures_fields(self):
        spec = InstrumentSpec.futures_contract(
            "NIFTY24AUGFUT", expiration_ns=1_000, lot_size=50.0, underlying="NIFTY"
        )
        assert spec.kind == "contract"
        assert spec.expiration_ns == 1_000
        assert spec.lot_size == 50.0
        assert spec.underlying == "NIFTY"

    def test_option_fields(self):
        spec = InstrumentSpec.option(
            "NIFTY24AUG25000CE",
            strike=25_000.0,
            right="CE",
            expiration_ns=1_000,
            lot_size=50.0,
        )
        assert spec.kind == "option"
        assert spec.strike == 25_000.0
        assert spec.right == "call"

    def test_option_rejects_bad_right(self):
        with pytest.raises(ValueError, match="right"):
            InstrumentSpec.option(
                "X", strike=100.0, right="straddle", expiration_ns=1, lot_size=1.0
            )

    def test_rejects_inverted_activation_expiry(self):
        with pytest.raises(ValueError, match="expiration_ns"):
            InstrumentSpec.futures_contract(
                "X", expiration_ns=5, lot_size=1.0, activation_ns=10
            )

    def test_index_not_tradable_in_session(self):
        spec = InstrumentSpec.index("NIFTY 50")
        assert not spec.tradable
        data = _bars([100.0, 101.0])
        with pytest.raises(ValueError, match="not tradable"):
            run_strategy_backtest(EnterOnce, **data, instrument=spec)


class TestKernelIntegration:
    def test_no_spec_is_bit_identical(self):
        """Attaching no spec must reproduce existing results exactly."""
        data = _bars([100.0, 101.0, 102.0, 103.0, 104.0])
        base = run_strategy_backtest(EnterOnce, **data)
        again = run_strategy_backtest(EnterOnce, **data, instrument=None)
        assert np.array_equal(base.equity_curve(), again.equity_curve())

    def test_neutral_spec_is_bit_identical(self):
        """A spec with neutral fields must not change results."""
        data = _bars([100.0, 101.0, 102.0, 103.0, 104.0])
        base = run_strategy_backtest(EnterOnce, **data)
        spec = InstrumentSpec.equity("ASSET", price_increment=0.0, lot_size=0.0)
        with_spec = run_strategy_backtest(EnterOnce, **data, instrument=spec)
        assert np.array_equal(base.equity_curve(), with_spec.equity_curve())

    def test_multiplier_scales_position_sizing(self):
        data = _bars([100.0, 102.0, 104.0, 106.0])
        spec = InstrumentSpec.futures_contract(
            "FUT",
            expiration_ns=10_000,  # far beyond the data
            lot_size=1.0,
            multiplier=50.0,
            price_increment=0.0,
        )
        result = run_strategy_backtest(
            EnterOnce, **data, config=_zero_fee_config(), instrument=spec
        )
        # 100k at price 100 with multiplier 50 -> 20 contracts;
        # +6 points -> pnl 6 * 20 * 50 = 6000 marked at the final close.
        assert result.equity_curve()[-1] == pytest.approx(106_000.0)

    def test_expiry_settles_open_position(self):
        data = _bars([100.0, 101.0, 102.0, 103.0])
        spec = InstrumentSpec.futures_contract(
            "FUT", expiration_ns=2, lot_size=1.0, price_increment=0.0
        )
        result = run_strategy_backtest(
            EnterOnce, **data, config=_zero_fee_config(), instrument=spec
        )
        assert len(result.trades()) == 1
        trade = result.trades()[0]
        assert trade.exit_reason == "Settlement"
        # Settled on the bar whose timestamp reached expiry (ts=2, close=102).
        assert trade.exit_price == pytest.approx(102.0)

    def test_entries_rejected_after_expiry(self):
        data = _bars([100.0, 101.0, 102.0, 103.0])
        spec = InstrumentSpec.futures_contract(
            "FUT", expiration_ns=2, lot_size=1.0, price_increment=0.0
        )
        rejects = []

        class Recorder(EnterEveryBar):
            def on_order_rejected(self, ctx, event):
                rejects.append(event.reject_reason)

        result = run_strategy_backtest(
            Recorder, **data, config=_zero_fee_config(), instrument=spec
        )
        assert len(result.trades()) == 1
        assert rejects and all(r == "Expired" or r == "expired" for r in rejects)

    def test_pre_activation_entry_rejected(self):
        data = _bars([100.0, 101.0, 102.0, 103.0])
        spec = InstrumentSpec.futures_contract(
            "FUT",
            expiration_ns=1_000,
            lot_size=1.0,
            activation_ns=2,
            price_increment=0.0,
        )
        result = run_strategy_backtest(
            EnterEveryBar, **data, config=_zero_fee_config(), instrument=spec
        )
        assert len(result.trades()) == 1
        # First fill can only happen once the contract activates at ts=2.
        assert result.trades()[0].entry_time >= 2

    def test_lot_size_floors_contracts(self):
        data = _bars([100.0, 101.0, 102.0])
        spec = InstrumentSpec.futures_contract(
            "FUT", expiration_ns=1_000, lot_size=300.0, price_increment=0.0
        )
        result = run_strategy_backtest(
            EnterOnce, **data, config=_zero_fee_config(), instrument=spec
        )
        # 100k / 100 = 1000 raw units -> floors to 900 (3 lots of 300).
        assert result.trades()[0].size == pytest.approx(900.0)

    def test_derived_stop_lands_on_tick_grid(self):
        data = _bars([100.0, 100.0, 100.0])
        config = _zero_fee_config()
        config.set_fixed_stop(0.033)  # raw stop 96.7 -> off a 0.25 grid

        spec = InstrumentSpec.equity("EQ", price_increment=0.25, lot_size=1.0)

        stops = []

        class StopReader(EnterOnce):
            def on_bar(self, ctx):
                super().on_bar(ctx)
                if ctx.position is not None:
                    stops.append(ctx.position.stop_price)

        run_strategy_backtest(StopReader, **data, config=config, instrument=spec)
        assert stops, "expected an open position"
        stop = stops[-1]
        assert stop == pytest.approx(round(stop / 0.25) * 0.25)
        # Conservative for a long: at or below the raw 96.7 level.
        assert stop <= 96.7 + 1e-9
