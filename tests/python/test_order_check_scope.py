"""``order_check_scope="typed"``: a check that judges typed orders only leaves intents alone.

``enter()`` and ``close_position()`` carry no order to show a check, so by default (``"all"``) they
are refused while one is installed. A host whose check judges forms an intent cannot have (a kind,
a validity, ``post_only``) declares it with ``"typed"``: intents then fill as without a check, the
check still judges every typed order, and a live node takes the same option.
"""

import sys
from pathlib import Path

import pytest
from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt.errors import InvalidOrder, InvalidParameter
from raptorbt.strategy import orders

sys.path.insert(0, str(Path(__file__).parent))

from test_execution import make_bars  # noqa: E402


def config():
    return BacktestConfig(initial_capital=500_000.0, fees=0.0)


class Intents(Strategy):
    def on_bar(self, ctx, bar):
        if ctx.idx == 1:
            self.enter(size_frac=0.5)
        elif ctx.idx == 3:
            self.close_position()


class Typed(Strategy):
    def on_bar(self, ctx, bar):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", units=1))


def run(strategy, check, **kw):
    data = make_bars(["AAA"], n=6)
    return run_strategies([strategy], data, config(), bar_stamp="close", order_check=check, **kw)


def test_the_default_still_refuses_an_intent():
    with pytest.raises(InvalidOrder) as e:
        run(Intents(), lambda o, c: None)
    assert e.value.code == "order_check_intent"
    with pytest.raises(InvalidOrder) as e:
        run(Intents(), lambda o, c: None, order_check_scope="all")
    assert e.value.code == "order_check_intent"


def test_a_typed_scope_lets_enter_and_close_position_fill():
    seen = []

    def check(order, ctx):
        seen.append(order)

    got = run(Intents(), check, order_check_scope="typed")[0]
    assert len(got.result.trades()) == 1
    assert seen == []  # an intent is not shown to the check


def test_a_typed_scope_still_judges_typed_orders():
    got = run(Typed(), lambda o, c: "no_market", order_check_scope="typed")[0]
    orders_ = got.result.orders()
    assert [str(o.reject_reason) for o in orders_ if o.status == "rejected"] == ["no_market"]
    assert got.result.trades() == []


def test_a_live_node_takes_the_scope():
    node = LiveNode(
        [Intents()], ["AAA"], config(), bar_stamp="close",
        order_check=lambda o, c: None, order_check_scope="typed",
    )  # fmt: skip
    assert node is not None


def test_an_unknown_scope_is_a_structured_error():
    with pytest.raises(InvalidParameter) as e:
        run(Intents(), lambda o, c: None, order_check_scope="intents")
    assert e.value.code == "invalid_order_check_scope"
    assert e.value.parameter == "order_check_scope" and e.value.valid == ["all", "typed"]
