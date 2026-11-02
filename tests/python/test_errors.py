"""Structured errors: classes, rendering, round-trip, and the code registry."""

import json
import pickle
import re
from pathlib import Path

import pytest

import raptorbt
from raptorbt import errors
from raptorbt.errors import (
    CODES,
    SKILLS,
    InvalidData,
    InvalidOrder,
    InvalidParameter,
    InvalidSubscription,
    RaptorBTError,
    StateError,
    UnknownSymbol,
    UnsupportedOperation,
)

CLASSES = [
    (InvalidParameter, ValueError),
    (InvalidData, ValueError),
    (InvalidOrder, ValueError),
    (InvalidSubscription, ValueError),
    (UnsupportedOperation, ValueError),
    (StateError, RuntimeError),
    (UnknownSymbol, KeyError),
    (UnknownSymbol, ValueError),
]


@pytest.mark.parametrize(("cls", "builtin"), CLASSES)
def test_class_is_catchable_as_itself_its_base_and_its_builtin(cls, builtin):
    with pytest.raises(cls):
        raise cls("boom", code="x")
    with pytest.raises(RaptorBTError):
        raise cls("boom", code="x")
    with pytest.raises(builtin):
        raise cls("boom", code="x")


def test_classes_are_exported_at_top_level():
    assert raptorbt.errors is errors
    for cls in {c for c, _ in CLASSES} | {RaptorBTError}:
        assert getattr(raptorbt, cls.__name__) is cls
        assert cls.__name__ in raptorbt.__all__


def test_str_leads_with_legacy_message_then_blocks():
    e = InvalidParameter(
        "bar_stamp must be one of 'open', 'close', got 'start'",
        code="invalid_bar_stamp",
        operation="run_strategies",
        parameter="bar_stamp",
        value="start",
    )
    text = str(e)
    assert text.splitlines()[0] == "bar_stamp must be one of 'open', 'close', got 'start'"
    assert "Operation:\n    run_strategies" in text
    assert "Requested:\n    bar_stamp = 'start'" in text
    assert "Supported:\n    'open', 'close', None" in text
    assert "Related skill:\n    raptorbt-timeframes" in text
    assert re.search(r"one of 'open'", text)  # legacy match= regexes still hit


def test_blocks_are_omitted_when_unknown():
    text = str(InvalidData("odd", code="no_such_code"))
    assert text == "odd"


def test_attributes_and_defaults_from_registry():
    e = InvalidOrder("bad", code="invalid_order_side", parameter="side", value="hold")
    assert (e.code, e.parameter, e.value) == ("invalid_order_side", "side", "hold")
    assert e.valid == ["buy", "sell"]
    assert e.skill == "orders"
    assert e.operation is None


def test_unknown_skill_is_refused():
    with pytest.raises(ValueError):
        InvalidData("x", code="c", skill="bogus")


def test_to_dict_round_trips_and_is_json():
    e = UnknownSymbol(
        "run_strategies: no data for 'ZZZ'",
        code="unknown_symbol",
        operation="run_strategies",
        parameter="symbols",
        value=["ZZZ"],
        valid=["AAA", "BBB"],
    )
    d = e.to_dict()
    assert json.loads(json.dumps(d)) == d
    back = errors.from_dict(json.loads(json.dumps(d)))
    assert type(back) is UnknownSymbol
    assert back.to_dict() == d
    assert str(back) == str(e)
    again = pickle.loads(pickle.dumps(e))
    assert again.to_dict() == d


def test_value_none_is_distinct_from_no_value():
    a = InvalidParameter("a", code="invalid_bar_stamp", parameter="bar_stamp", value=None)
    b = InvalidParameter("a", code="invalid_bar_stamp", parameter="bar_stamp")
    assert "bar_stamp = None" in str(a)
    assert "bar_stamp = None" not in str(b)
    assert errors.from_dict(a.to_dict()).has_value is True


def test_registry_is_well_formed():
    for code, entry in CODES.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", code), code
        assert entry.skill in SKILLS, code
        assert issubclass(entry.cls, RaptorBTError), code
        assert entry.meaning, code
        assert errors.make(code, "m").code == code


# ---------------------------------------------------------------------------
# Representative failures, from the public entry points
# ---------------------------------------------------------------------------

import numpy as np

from raptorbt import KernelSession, LiveNode, PortfolioSession, Strategy, run_strategies
from raptorbt import run_strategy_backtest
from raptorbt.strategy.orders import Market

MIN = 60 * 10**9


class Noop(Strategy):
    pass


def _bars(n=20):
    close = 100.0 + np.arange(n, dtype=np.float64)
    return {
        "timestamps": (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.ones(n),
    }


def _single(**override):
    bars = {**_bars(), **override}
    return run_strategy_backtest(
        Noop, bars["timestamps"], bars["open"], bars["high"], bars["low"], bars["close"],
        bars["volume"], bar_stamp="close",
    )


def _finished_session():
    s = KernelSession(symbol="X")
    s.step(0, 0, 100.0, 101.0, 99.0, 100.0, 1000.0)
    s.finish()
    return s


def _finished_node():
    node = LiveNode([Noop()], ["X"])
    node.finish()
    return node


ROWS = [
    # id, call, class, builtin, code, parameter, valid, skill
    (
        "bad-bar-stamp",
        lambda: run_strategies([Noop()], {"A": _bars()}, bar_stamp="start"),
        InvalidParameter, ValueError, "invalid_bar_stamp", "bar_stamp", ["open", "close", None], "timeframes",
    ),
    (
        "bad-account-type",
        lambda: KernelSession(symbol="X", account_type="spot"),
        InvalidParameter, ValueError, "invalid_account_type", "account_type", ["cash", "margin"], "execution",
    ),
    (
        "bad-account-type-portfolio",
        lambda: PortfolioSession(account_type="spot"),
        InvalidParameter, ValueError, "invalid_account_type", "account_type", ["cash", "margin"], "execution",
    ),
    (
        "bad-order-side",
        lambda: Market(side="hold", units=1),
        InvalidOrder, ValueError, "invalid_order_side", "side", ["buy", "sell"], "orders",
    ),
    (
        "bad-order-tif",
        lambda: Market(side="buy", units=1, tif="forever"),
        InvalidOrder, ValueError, "invalid_time_in_force", "tif",
        ["gtc", "day", "gtd", "ioc", "fok", "at_open", "at_close"], "orders",
    ),
    (
        "length-mismatch",
        lambda: _single(volume=np.ones(5)),
        InvalidData, ValueError, "length_mismatch", "volume", [20], "market-data",
    ),
    (
        "decreasing-timestamps",
        lambda: _single(timestamps=np.arange(20, 0, -1, dtype=np.int64) * MIN),
        InvalidData, ValueError, "decreasing_timestamps", "timestamps", None, "market-data",
    ),
    (
        "unknown-symbol-subset",
        lambda: run_strategies([(Noop(), ["A", "ZZZ"])], {"A": _bars(), "B": _bars()}, bar_stamp="close"),
        UnknownSymbol, ValueError, "unknown_symbol", "symbols", ["A", "B"], "market-data",
    ),
    (
        "finish-twice-session",
        lambda: _finished_session().finish(),
        errors.SessionFinished, ValueError, "session_finished", None, None, "backtesting",
    ),
    (
        "finish-twice-node",
        lambda: _finished_node().finish(),
        StateError, RuntimeError, "node_finished", None, None, "live-trading",
    ),
    (
        "empty-strategy-list",
        lambda: run_strategies([], {"A": _bars()}, bar_stamp="close"),
        InvalidParameter, ValueError, "empty_strategies", "strategies", None, "strategy",
    ),
    (
        "bad-batch-size",
        lambda: run_strategies([Noop()], {"A": _bars()}, bar_stamp="close", batch_size=0),
        InvalidParameter, ValueError, "invalid_batch_size", "batch_size", ["a positive integer", None], "backtesting",
    ),
    (
        "bad-direction-from-rust",
        lambda: KernelSession(symbol="X", direction=2),
        InvalidParameter, ValueError, "invalid_direction", "direction", [1, -1], "backtesting",
    ),
    (
        "bad-bar-unit-from-rust",
        lambda: raptorbt.BarAggregator(5, "fortnight"),
        InvalidSubscription, ValueError, "invalid_bar_spec", "unit", None, "timeframes",
    ),
    (
        "bad-squareoff-time-from-rust",
        lambda: raptorbt.BacktestConfig(squareoff_time="25:99"),
        InvalidParameter, ValueError, "invalid_squareoff_time", "squareoff_time", None, "execution",
    ),
]


@pytest.mark.parametrize(
    ("call", "cls", "builtin", "code", "parameter", "valid", "skill"),
    [pytest.param(*r[1:], id=r[0]) for r in ROWS],
)
def test_representative_failures_are_structured(call, cls, builtin, code, parameter, valid, skill):
    with pytest.raises(builtin) as info:
        call()
    e = info.value
    assert isinstance(e, cls) and isinstance(e, RaptorBTError)
    assert (e.code, e.parameter, e.skill) == (code, parameter, skill)
    if valid is not None:
        assert e.valid == valid
    assert str(e).startswith(e.message)
    assert f"raptorbt-{skill}" in str(e)
    assert errors.from_dict(json.loads(json.dumps(e.to_dict()))).to_dict() == e.to_dict()


def test_legacy_message_text_is_unchanged_in_the_first_line():
    with pytest.raises(ValueError, match="side must be one of") as info:
        Market(side="hold", units=1)
    assert str(info.value).splitlines()[0] == "side must be one of ('buy', 'sell'), got 'hold'"
    with pytest.raises(ValueError) as info:
        KernelSession(symbol="X", account_type="spot")
    assert str(info.value).splitlines()[0] == "account_type must be 'cash' or 'margin', got \"spot\""


def test_error_carries_the_value_it_rejected():
    with pytest.raises(InvalidParameter) as info:
        run_strategies([Noop()], {"A": _bars()}, bar_stamp="start")
    assert info.value.value == "start"
    assert info.value.operation == "run_strategies"
    assert "Requested:\n    bar_stamp = 'start'" in str(info.value)


# ---------------------------------------------------------------------------
# The registry is the single list of codes
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
PY_CODE = re.compile(r'code\s*=\s*"([a-z][a-z0-9_]*)"')
RS_CODE = re.compile(r'Fail::new\(\s*"([a-z][a-z0-9_]*)"')
RS_MAP = re.compile(r'=>\s*"([a-z][a-z0-9_]*)",')


def _raised_codes():
    found = set()
    for path in (ROOT / "python" / "raptorbt").rglob("*.py"):
        if path.name != "errors.py":
            found |= set(PY_CODE.findall(path.read_text()))
    src = ROOT / "src"
    if not src.is_dir():
        pytest.skip("Rust sources not available")
    for path in src.rglob("*.rs"):
        text = path.read_text()
        found |= set(RS_CODE.findall(text))
        if path.name == "error.rs" and path.parent.name == "core":
            found |= set(RS_MAP.findall(text))
    return found


def test_every_code_raised_in_the_source_is_registered():
    assert _raised_codes() - set(CODES) == set()


def test_every_registered_code_is_raised_somewhere():
    assert set(CODES) - _raised_codes() == set()


def test_docs_page_is_generated_from_the_registry():
    assert (ROOT / "docs" / "errors.md").read_text() == errors.markdown()
