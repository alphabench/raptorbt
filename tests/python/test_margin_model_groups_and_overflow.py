"""A chosen margin model on account groups and on option structures, and the
structured error a money fault leaves Python as."""

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    AccountSpec,
    BacktestConfig,
    InstrumentSpec,
    LiveNode,
    MarginModel,
    Strategy,
    run_strategies,
)
from raptorbt import structures as st
from raptorbt.errors import InvalidParameter, MoneyOverflow, RaptorBTError

from _chain_fixture import universe
from test_structures import make


class _Idle(Strategy):
    def on_bar(self, ctx):
        pass


def test_a_model_on_a_cash_account_group_is_refused_not_ignored():
    config = BacktestConfig(initial_capital=1_000_000.0, margin_model=MarginModel.cash_only())
    data, inst = universe()
    with pytest.raises(InvalidParameter) as e:
        run_strategies(
            [{"strategy": _Idle(), "account": "a"}],
            data,
            config,
            instruments=inst,
            account_type="margin",
            leverage=1.0,
            accounts={"a": AccountSpec(initial_capital=500_000.0, account_type="cash")},
            bar_stamp="close",
        )
    assert e.value.code == "invalid_margin_model"
    # The same model on a margin account group is accepted.
    run_strategies(
        [{"strategy": _Idle(), "account": "a"}],
        data,
        config,
        instruments=inst,
        account_type="margin",
        leverage=1.0,
        accounts={"a": AccountSpec(initial_capital=500_000.0, account_type="margin")},
        bar_stamp="close",
    )


class _Watched(make(st.iron_condor(1, 2))):
    """The condor strategy, noting the free capital left after every bar."""

    def __init__(self):
        super().__init__()
        self.free = []

    def on_bar(self, ctx):
        super().on_bar(ctx)
        self.free.append(ctx.free_capital)


CAPITAL = 2_000_000.0


def _condor(model, grouped):
    s = _Watched()
    data, inst = universe()
    cfg = BacktestConfig(initial_capital=CAPITAL, fees=0.0, margin_model=model)
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    if grouped:
        results, _ = run_strategies(
            [{"strategy": s, "account": "a"}],
            data,
            cfg,
            accounts={"a": AccountSpec(initial_capital=CAPITAL, account_type="margin")},
            **kw,
        )
        res = results[0]
    else:
        res = run_strategies([s], data, cfg, **kw)[0]
    fills = {o.symbol: o.avg_fill_price for o in res.result.orders()}
    # Fees are zero and a margin entry debits only fees, so cash stays at
    # capital and what is not free is what is locked.
    return s, fills, CAPITAL - s.free[-3]


@pytest.mark.parametrize("grouped", [False, True], ids=["book", "account_group"])
@pytest.mark.parametrize(
    "model",
    [MarginModel.defined_risk(), MarginModel.spread_credit(15, 5), MarginModel.from_spec(1.0)],
    ids=["defined_risk", "spread_credit", "from_spec"],
)
def test_an_iron_condors_lock_is_what_required_legs_says_for_its_legs(model, grouped):
    s, fills, locked = _condor(model, grouped)
    structure = s.group.structure
    prices = {leg.instrument: fills[leg.instrument] for leg in structure.legs}
    assert locked == pytest.approx(structure.margin(model, prices), rel=1e-9)
    # Priced leg by leg (each sold leg naked) it would cost more.
    alone = sum(
        model.required(leg.spec, leg.sign * leg.quantity, prices[leg.instrument])
        for leg in structure.legs
    )
    assert locked < alone


def test_a_money_fault_is_a_structured_error_not_a_panic():
    n = 8
    close = np.linspace(100.0, 101.0, n)
    cfg = BacktestConfig(initial_capital=1e300)
    with pytest.raises(MoneyOverflow) as e:
        raptorbt.run_single_backtest(
            np.arange(n, dtype=np.int64) * 60_000_000_000,
            close, close + 0.1, close - 0.1, close, np.full(n, 1e3),
            np.array([True] + [False] * (n - 1)), np.zeros(n, bool),
            direction=1, config=cfg,
        )
    assert e.value.code == "money_overflow"
    assert isinstance(e.value, RaptorBTError) and isinstance(e.value, ArithmeticError)
