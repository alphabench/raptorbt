"""User mistakes surface as one structured raptorbt error: no panic notice, no raw
``PanicException``, no builtin ``KeyError``/``ValueError`` without a code."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies, run_strategy_backtest
from raptorbt.errors import InvalidParameter, MoneyOverflow, RaptorBTError, UnknownSymbol

N = 8
TS = np.arange(N, dtype=np.int64) * 60_000_000_000
CLOSE = np.linspace(100.0, 101.0, N)
ONES = np.ones(N)


class Idle(Strategy):
    def on_bar(self, ctx):
        pass


def _config(capital):
    cfg = BacktestConfig()
    cfg.initial_capital = capital
    return cfg


@pytest.mark.parametrize("capital", [1e300, float("inf"), float("nan")])
def test_run_strategy_backtest_refuses_an_unbookable_capital_with_money_overflow(capital):
    with pytest.raises(MoneyOverflow) as e:
        run_strategy_backtest(
            Idle, TS, CLOSE, CLOSE, CLOSE, CLOSE, ONES, config=_config(capital), bar_stamp="close"
        )
    assert e.value.code == "money_overflow"
    assert isinstance(e.value, RaptorBTError) and isinstance(e.value, ArithmeticError)


@pytest.mark.parametrize("entry", ["single", "portfolio", "node"])
def test_a_money_fault_prints_no_panic_notice(entry):
    code = textwrap.dedent(
        f"""
        import numpy as np, raptorbt as rb
        from raptorbt import BacktestConfig, LiveNode, Strategy
        class Idle(Strategy):
            def on_bar(self, ctx):
                pass
        cfg = BacktestConfig(); cfg.initial_capital = 1e300
        n = 4
        ts = np.arange(n, dtype=np.int64) * 60_000_000_000
        c = np.linspace(100.0, 101.0, n)
        bars = dict(timestamps=ts, open=c, high=c, low=c, close=c, volume=np.ones(n))
        try:
            if {entry!r} == "single":
                rb.run_strategy_backtest(Idle, ts, c, c, c, c, np.ones(n), config=cfg, bar_stamp="close")
            elif {entry!r} == "portfolio":
                rb.run_strategies([Idle()], {{"A": bars}}, config=cfg, bar_stamp="close")
            else:
                LiveNode([Idle()], ["A"], cfg, bar_stamp="close").push_bar("A", 1, 1.0, 1.0, 1.0, 1.0, 1.0)
        except rb.errors.MoneyOverflow as e:
            print("structured", e.code)
        """
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.stdout.strip() == "structured money_overflow", done.stderr
    assert "panicked" not in done.stderr and done.stderr.strip() == ""


@pytest.mark.parametrize("call", ["push_bar", "push_tick", "push_depth"])
def test_pushing_to_an_unknown_symbol_is_a_structured_unknown_symbol(call):
    node = LiveNode([Idle()], ["A"], BacktestConfig(), bar_stamp="close")
    args = {
        "push_bar": (1, 1.0, 1.0, 1.0, 1.0, 1.0),
        "push_tick": (1, 1.0),
        "push_depth": (1, [(1.0, 1.0)], [(2.0, 1.0)]),
    }[call]
    with pytest.raises(UnknownSymbol) as e:
        getattr(node, call)("ZZZ", *args)
    assert e.value.code == "unknown_symbol" and e.value.valid == ["A"]
    assert isinstance(e.value, KeyError)  # existing handlers keep working


@pytest.mark.parametrize(
    "seed, text",
    [
        ({"quantity": -5.0, "avg_price": 100.0}, "short on a cash account"),
        ({"quantity": 5.0, "avg_price": 100.0, "margin_minor": 100}, "margin accounts only"),
    ],
)
def test_a_seed_the_account_cannot_hold_is_a_structured_refusal(seed, text):
    with pytest.raises(InvalidParameter, match=text) as e:
        LiveNode(
            [Idle()], ["A"], BacktestConfig(), bar_stamp="close", initial_positions={0: {"A": seed}}
        )
    assert e.value.code == "adoption_refused"


def test_a_computation_that_refuses_its_input_raises_a_coded_error():
    with pytest.raises(InvalidParameter) as e:
        raptorbt.sma(np.arange(5.0), 0)
    assert e.value.code == "invalid_input" and isinstance(e.value, ValueError)
