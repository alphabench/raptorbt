"""raptorbt.sizing: lots from capital, risk, volatility and margin; refusals are structured."""

import pytest

from raptorbt import sizing
from raptorbt.errors import InvalidParameter


def refused(code, fn, *a, **kw):
    with pytest.raises(InvalidParameter) as e:
        fn(*a, **kw)
    assert e.value.code == code
    return e.value


def test_capital_percent_and_units():
    assert sizing.lots_for_capital(1_000_000, 100.0, 50) == 200  # 5,000 per lot
    assert sizing.lots_for_capital(1_000_000, 100.0, 50, fraction=0.25) == 50
    assert sizing.lots_for_capital(1_000_000, 100.0, 50, multiplier=2.0) == 100
    assert sizing.lots_for_percent_of_equity(1_000_000, 10, 100.0, 50) == 20
    assert sizing.units(3, 50) == 150.0


def test_risk_sizing_rounds_down_and_never_up():
    # 1% of 1,000,000 = 10,000; 5 per unit x 50 = 250 per lot -> 40 lots
    assert sizing.lots_for_risk(1_000_000, 0.01, 100.0, 95.0, 50) == 40
    assert sizing.lots_for_risk(1_000_000, 0.01, 100.0, 105.0, 50) == 40  # a short stop above
    # 10,000 / (7 x 50) = 28.57
    assert sizing.lots_for_risk(1_000_000, 0.01, 100.0, 93.0, 50) == 28
    # an exact ratio that floats can land just under still gives the whole lots
    assert sizing.lots_for_risk(1_000_000, 0.0003, 100.0, 99.0, 1) == 300


def test_below_one_lot_is_an_error_with_the_fraction_attached():
    e = refused("below_one_lot", sizing.lots_for_risk, 10_000, 0.01, 100.0, 90.0, 50)
    assert e.value == pytest.approx(100 / 500)
    refused("below_one_lot", sizing.lots_for_capital, 4_000, 100.0, 50)
    refused("below_one_lot", sizing.lots_for_margin, 10_000, 12_000)


def test_volatility_target_and_leverage_cap():
    # notional = 1,000,000 x 0.10 / 0.20 = 500,000 ; per lot 5,000 -> 100
    assert sizing.lots_for_volatility(1_000_000, 0.10, 0.20, 100.0, 50) == 100
    # the cap holds notional to 0.25 x capital
    assert sizing.lots_for_volatility(1_000_000, 0.10, 0.20, 100.0, 50, max_leverage=0.25) == 50


def test_margin_limited_lots():
    assert sizing.lots_for_margin(1_000_000, 90_000) == 11
    assert sizing.lots_for_margin(1_000_000, 90_000, buffer=0.2) == 8


def test_margin_that_is_not_linear_in_lots():
    # a hedge benefit: the first lot costs 100, each further one 60
    req = lambda n: 100 + 60 * (n - 1)  # noqa: E731
    assert sizing.max_lots_by_margin(req, 1_000) == 16  # 100 + 60 x 15 = 1,000
    assert sizing.max_lots_by_margin(req, 999) == 15
    assert sizing.max_lots_by_margin(req, 100) == 1
    assert sizing.max_lots_by_margin(req, 10**9, cap=64) == 64
    refused("below_one_lot", sizing.max_lots_by_margin, req, 99)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), None, "x"])
def test_bad_inputs_are_refused_not_defaulted(bad):
    refused("invalid_sizing_input", sizing.lots_for_capital, bad, 100.0, 50)
    refused("invalid_sizing_input", sizing.lots_for_capital, 1_000, bad, 50)
    refused("invalid_sizing_input", sizing.lots_for_risk, 1_000_000, 0.01, 100.0, 95.0, bad)


def test_stop_at_entry_and_fractions_out_of_range():
    refused("invalid_sizing_input", sizing.lots_for_risk, 1_000_000, 0.01, 100.0, 100.0, 50)
    refused("invalid_sizing_input", sizing.lots_for_risk, 1_000_000, 1.5, 100.0, 95.0, 50)
    refused("invalid_sizing_input", sizing.lots_for_percent_of_equity, 1_000_000, 120, 100.0, 50)


def test_sizer_reads_the_context_at_call_time():
    class Ctx:
        equity = 1_000_000.0
        free_capital = 500_000.0

    ctx = Ctx()
    s = sizing.Sizer(ctx)
    assert s.risk(0.01, 100.0, 95.0, 50) == 40
    assert s.capital(100.0, 50) == 100  # free capital, not equity
    assert s.percent_of_equity(10, 100.0, 50) == 20
    assert s.margin(90_000) == 5
    ctx.equity = 2_000_000.0
    assert s.risk(0.01, 100.0, 95.0, 50) == 80
