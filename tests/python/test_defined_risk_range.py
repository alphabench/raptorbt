"""``defined_risk`` margin over the whole payoff range, and the live-node adjustment refusal.

``Margin.defined_risk`` was evaluated on the strike grid plus a far end at twice
the highest strike, so an uncovered short call looked like a finite loss
(1,867,500 for one 75-lot at 25,000 sold at 100) when it has no bound; it also
ignored future and cash legs and legs of different expiries.

The payoff of the legs at the option expiry is piecewise linear in the
underlying's price, so the exact worst loss is on the kinks (zero and each
strike) plus the slope above the highest strike.
"""

import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, InstrumentSpec, LiveNode, MarginModel, Strategy
from raptorbt.errors import InvalidParameter
from raptorbt.margin import defined_risk
from raptorbt.strategy import orders as O

from _market_harness import bars, ist, run_paths

DAY = 86_400 * 10**9
EXPIRY = 20_000 * DAY + 10 * 3_600 * 10**9
INF = float("inf")


def option(strike, right="call", expiry=EXPIRY):
    return InstrumentSpec.option(
        f"N{strike}{right[0]}", strike=float(strike), right=right, expiration_ns=expiry,
        lot_size=75.0, underlying="NIFTY",
    )


CALL, PUT = option(25000), option(25000, "put")
FUTURE = InstrumentSpec.futures_contract("XFUT", EXPIRY, 75.0)


def test_a_long_option_risks_its_premium_and_a_vertical_its_width_less_the_credit():
    assert defined_risk([(CALL, 75.0, 100.0)]) == 100.0 * 75.0
    spread = [(CALL, -75.0, 100.0), (option(25100), 75.0, 60.0)]
    assert defined_risk(spread) == 60.0 * 75.0  # width 100 - credit 40, times 75


def test_an_uncovered_short_call_has_no_bound():
    assert defined_risk([(CALL, -75.0, 100.0)]) == INF
    assert defined_risk([(CALL, -75.0, 100.0), (PUT, -75.0, 90.0)]) == INF  # a short straddle
    # more short calls than long still lose without bound above the highest strike
    assert defined_risk([(CALL, -150.0, 100.0), (option(25100), 75.0, 60.0)]) == INF


def test_an_uncovered_short_put_loses_at_most_the_strike_less_the_premium():
    assert defined_risk([(PUT, -75.0, 100.0)]) == (25000.0 - 100.0) * 75.0


def test_future_and_cash_legs_are_part_of_the_payoff():
    # covered call: the long future cancels the slope above the strike
    covered = [(CALL, -75.0, 100.0), (FUTURE, 75.0, 25000.0)]
    assert defined_risk(covered) == (25000.0 - 100.0) * 75.0  # the future falling to zero, less the premium
    # a short future makes the short call doubly unbounded
    assert defined_risk([(CALL, -75.0, 100.0), (FUTURE, -75.0, 25000.0)]) == INF
    # a protective put on a long future bounds the downside at strike-to-price
    protected = [(PUT, 75.0, 120.0), (FUTURE, 75.0, 25000.0)]
    assert defined_risk(protected) == 120.0 * 75.0


def test_legs_of_different_expiries_are_refused_not_priced_at_intrinsic_value():
    calendar = [(CALL, -75.0, 100.0), (option(25000, expiry=EXPIRY + 7 * DAY), 75.0, 300.0)]
    with pytest.raises(InvalidParameter, match="one expiry") as e:
        defined_risk(calendar)
    assert e.value.code == "invalid_margin_model"


def test_a_group_with_no_bound_refuses_every_line_of_it():
    model = MarginModel.defined_risk()
    straddle = [(CALL, -75.0, 100.0), (PUT, -75.0, 90.0)]
    assert model.required_legs(straddle) == INF
    assert model.required_legs([(CALL, -75.0, 100.0), (option(25100), 75.0, 60.0)]) == 4500.0


def test_a_margin_account_refuses_a_naked_short_call_under_defined_risk():
    """Capital 2,000,000 would have covered the old 1,867,500 and let the write through."""
    start = ist(2026, 11, 2, 9, 15)
    call = InstrumentSpec.option(
        "C", strike=24000.0, right="call", expiration_ns=ist(2026, 11, 24, 15, 30), lot_size=75.0,
        underlying="NIFTY", venue="NFO",
    )
    script = {1: lambda s, c: s.submit_order(O.Market(side="sell", quantity=75.0))}
    runs = run_paths(
        script, {"C": bars([(200,) * 4] * 5, start)},
        lambda: BacktestConfig(initial_capital=2_000_000.0, fees=0.0, margin_model=MarginModel.defined_risk()),
        instruments={"C": call}, account_type="margin", paths=("batch", "node"),
    )
    for path, (strategy, result) in runs.items():
        assert strategy.rejections == [("11-02 09:17", "insufficient_margin")], path
        assert result.result.trades() == []


# -- M18: a live node takes raw history only -----------------------------------------------------------


def _split():
    return rb.CorporateAction("X", "split", ist(2026, 11, 3, 9, 15), ratio=2.0)


def _node(policy):
    return LiveNode(
        [Strategy()], ["X"], BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close",
        corporate_actions=[_split()], adjustment_policy=policy,
    )


def test_a_live_node_refuses_adjusted_history_and_says_why():
    """Bars pushed before the ex-date, and what was built on them, cannot be restated."""
    with pytest.raises(InvalidParameter) as e:
        _node("adjusted")
    assert (e.value.code, e.value.parameter) == ("invalid_adjustment_policy", "adjustment_policy")
    assert list(e.value.valid) == ["raw"]
    assert "indicator state" in str(e.value) and "ex-date" in str(e.value)
    _node("raw")  # positions are rescaled at the ex-date instead


def test_a_batch_run_still_puts_history_on_the_adjusted_basis():
    data = {"X": bars([(100,) * 4] * 3 + [(50,) * 4] * 3, ist(2026, 11, 2, 9, 15))}
    action = rb.CorporateAction("X", "split", int(data["X"]["timestamps"][3]), ratio=2.0)
    rb.run_strategies(
        [Strategy()], data, BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close",
        corporate_actions=[action], adjustment_policy="adjusted",
    )
