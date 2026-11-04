"""An order term that is NaN, infinite or out of range is refused, never traded.

A NaN compares false against every bound, so ``units <= 0`` let it through; a protective level of
0 exits at once; a typed order did not check its own quantity.
"""

import pytest

from raptorbt import BacktestConfig, Strategy, run_strategies, run_strategy_backtest
from raptorbt.errors import InvalidOrder
from raptorbt.strategy import orders as O

from _market_harness import bars, ist

NAN, INF = float("nan"), float("inf")
BAD = [NAN, INF, -INF, 0.0, -1.0]


@pytest.mark.parametrize("bad", BAD)
def test_a_typed_order_checks_its_own_quantity(bad):
    with pytest.raises(InvalidOrder) as e:
        O.Market(side="buy", quantity=bad)
    assert e.value.code == "invalid_order_value" and e.value.parameter == "quantity"
    with pytest.raises(InvalidOrder) as e:
        O.Market(side="buy", units=bad)
    assert e.value.code == "invalid_order_value"


@pytest.mark.parametrize("bad", [NAN, INF, 0.0, -0.5, 1.5])
def test_size_frac_must_be_in_zero_to_one(bad):
    with pytest.raises(InvalidOrder) as e:
        O.Market(side="buy", size_frac=bad)
    assert (e.value.code, e.value.parameter) == ("invalid_order_value", "size_frac")


@pytest.mark.parametrize("name", ["stop_price", "target_price"])
@pytest.mark.parametrize("bad", BAD)
def test_a_protective_level_must_be_finite_and_positive(name, bad):
    with pytest.raises(InvalidOrder) as e:
        O.Market(side="buy", quantity=1.0, **{name: bad})
    assert (e.value.code, e.value.parameter) == ("invalid_order_value", name)
    O.Market(side="buy", quantity=1.0, **{name: 0.01})  # a small positive level is fine


@pytest.mark.parametrize("bad", [NAN, INF, 0.0, -1.0])
def test_prices_triggers_and_offsets_must_be_finite_and_positive(bad):
    for make in (
        lambda: O.Limit(side="buy", quantity=1.0, price=bad),
        lambda: O.StopMarket(side="buy", quantity=1.0, trigger=bad),
        lambda: O.StopLimit(side="buy", quantity=1.0, trigger=bad, price=1.0),
        lambda: O.StopLimit(side="buy", quantity=1.0, trigger=1.0, price=bad),
        lambda: O.TrailingStopMarket(side="sell", quantity=1.0, offset=bad),
    ):
        with pytest.raises(InvalidOrder) as e:
            make()
        assert e.value.code == "invalid_order_value"
    for bad_offset in (NAN, INF, -1.0):
        with pytest.raises(InvalidOrder):
            O.TrailingStopLimit(side="sell", quantity=1.0, offset=1.0, limit_offset=bad_offset)


class Enters(Strategy):
    def __init__(self, **kw):
        super().__init__()
        self.kw = kw

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.enter(**self.kw)


@pytest.mark.parametrize("kw", [
    {"size_frac": NAN}, {"stop_price": 0.0}, {"stop_price": NAN}, {"target_price": -1.0},
    {"target_price": INF},
])
def test_enter_refuses_the_same_terms(kw):
    data = {"X": bars([(100.0,) * 4] * 4, ist(2026, 11, 2, 9, 15))}
    with pytest.raises(InvalidOrder) as e:
        run_strategies([Enters(**kw)], data, BacktestConfig(initial_capital=1e6), bar_stamp="close")
    assert e.value.code == "invalid_order_value"


class Sets(Strategy):
    def __init__(self, stop=None, target=None):
        super().__init__()
        self.stop, self.target = stop, target

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.enter(size_frac=0.5)
        if ctx.idx == 2:
            if self.stop is not None:
                ctx.set_stop_price(self.stop)
            if self.target is not None:
                ctx.set_target_price(self.target)


@pytest.mark.parametrize("kw", [{"stop": 0.0}, {"stop": NAN}, {"target": -5.0}, {"target": INF}])
def test_moving_a_level_refuses_a_bad_value_and_none_still_removes_it(kw):
    # ``set_stop_price`` / ``set_target_price`` live on the single-symbol context
    b = bars([(100.0,) * 4] * 5, ist(2026, 11, 2, 9, 15))
    with pytest.raises(InvalidOrder) as e:
        run_strategy_backtest(
            Sets(**kw), **b, symbol="X", bar_stamp="close", config=BacktestConfig(initial_capital=1e6)
        )
    assert e.value.code == "invalid_order_value"
