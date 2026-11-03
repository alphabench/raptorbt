"""Margin models: named rules over a line, and the one rule the engine applies."""

import numpy as np
import pytest

from raptorbt import BacktestConfig, InstrumentSpec, MarginModel, Strategy, run_portfolio_strategy
from raptorbt.errors import InvalidParameter
from raptorbt.margin import defined_risk

DAY = 86_400 * 10**9
E1 = 20_000 * DAY + 10 * 3_600 * 10**9


def _opt(strike, right="call", expiry=E1, **kw):
    return InstrumentSpec.option(
        f"N{strike}{right[0]}", strike=float(strike), right=right, expiration_ns=expiry,
        lot_size=75.0, underlying="NIFTY", **kw,
    )


def test_models_compute_what_they_say_and_describe_themselves():
    eq = InstrumentSpec.equity("X")
    fut = InstrumentSpec.futures_contract("XFUT", E1, 75.0)
    call = _opt(25000)
    assert MarginModel.no_margin().required(fut, 400.0, 1500.0) == 0.0
    assert MarginModel.cash_only().required(eq, 10.0, 100.0) == 1000.0
    assert MarginModel.cash_only().required(eq, -10.0, 100.0) == float("inf")
    assert MarginModel.leverage({"CONTRACT": 5}).required(fut, 400.0, 1500.0) == 120_000.0
    assert MarginModel.leverage({"FUTURE": 5}).required(fut, 400.0, 1500.0) == 120_000.0
    assert MarginModel.fraction_of_notional(0.1).required(fut, 400.0, 1500.0) == 60_000.0
    os15 = MarginModel.option_simple(15)
    assert os15.required(call, 75.0, 100.0) == 7500.0
    assert os15.required(call, -75.0, 100.0) == 25000.0 * 75.0 * 0.15 + 7500.0
    assert MarginModel.option_simple(15).describe() == (
        '{"name":"Margin.option_simple","params":{"seller_pct":15.0}}'
    )
    assert MarginModel.from_spec(5).params() == {"leverage": 5.0}


def test_a_class_without_a_rule_and_a_bad_class_name_are_errors_never_zero():
    fut = InstrumentSpec.futures_contract("XFUT", E1, 75.0)
    with pytest.raises(InvalidParameter) as e:
        MarginModel.leverage({"CASH": 5}).required(fut, 1.0, 1.0)
    assert e.value.code == "invalid_margin_model"
    with pytest.raises(InvalidParameter, match="unknown instrument class"):
        MarginModel.by_type({"SWAP": MarginModel.cash_only()})
    with pytest.raises(InvalidParameter):
        MarginModel.fraction_of_notional(-0.01)
    with pytest.raises(InvalidParameter):
        MarginModel.from_spec(0)
    with pytest.raises(InvalidParameter, match="no margin for"):
        MarginModel.table({}).required(fut, 1.0, 1.0)


def test_a_spread_locks_its_worst_loss_and_spread_credit_needs_the_time():
    short, long_ = _opt(25000), _opt(25100)
    legs = [(short, -75.0, 100.0), (long_, 75.0, 60.0)]
    assert defined_risk(legs) == 60.0 * 75.0
    assert MarginModel.defined_risk().required_legs(legs) == 4500.0
    sc = MarginModel.spread_credit(10, 4)
    assert sc.required_legs(legs, E1 - 3 * DAY) == 4500.0
    with pytest.raises(InvalidParameter, match="needs the time"):
        sc.required_legs(legs)


def test_by_product_prices_each_part():
    eq = InstrumentSpec.equity("X")
    m = MarginModel.by_product(
        MarginModel.by_type({"EQUITY": MarginModel.fraction_of_notional(0.2)}),
        MarginModel.by_type({"EQUITY": MarginModel.cash_only()}),
    )
    assert m.required_split(eq, 100.0, 100.0, 40.0) == 800.0 + 6000.0
    assert m.required(eq, 100.0, 100.0) == 10_000.0


def test_from_spec_is_the_deposit_and_the_rate_the_kernel_locks():
    """The model and the margin account read one rule: the size the account
    fills is the largest whose ``FromSpec`` figure fits the capital."""
    strike, lot = 57_500.0, 30.0
    spec = InstrumentSpec.option(
        "BN57500CE", strike=strike, right="call", expiration_ns=10_000, lot_size=lot,
        span_pct=0.0975, exposure_pct=0.02,
    )
    n = 4
    closes = np.array([360.0, 350.0, 340.0, 330.0])
    bars = {
        "timestamps": np.arange(n, dtype=np.int64),
        "open": closes.copy(), "high": closes + 1.0, "low": closes - 1.0,
        "close": closes, "volume": np.full(n, 1_000.0),
    }

    class SellOnce(Strategy):
        done = False

        def on_bar(self, ctx):
            if not self.done:
                self.done = True
                self.enter(size_frac=1.0)

    cfg = BacktestConfig(initial_capital=400_000.0)
    cfg.fees = 0.0
    result = run_portfolio_strategy(
        SellOnce(), {"BN57500CE": bars}, config=cfg, directions={"BN57500CE": -1},
        instruments={"BN57500CE": spec}, account_type="margin", leverage=1.0,
        bar_stamp="close",
    )
    trade = result.result.trades()[0]
    model = MarginModel.from_spec(1.0)
    # The account sized the entry to the deposit: this many lots fit the
    # capital, one lot more does not, by the model's own figure.
    assert model.required(spec, -trade.size, trade.entry_price) <= 400_000.0
    assert model.required(spec, -(trade.size + lot), trade.entry_price) > 400_000.0
    assert model.required(spec, -1.0, 100.0) == pytest.approx((0.0975 + 0.02) * strike)
