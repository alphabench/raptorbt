"""Frozen behaviour of the event-driven strategy runners.

Replays every scenario in ``golden/runner_fixtures.json`` through
``run_portfolio_strategy``, ``run_tick_strategy`` and ``TickStrategyStream``
and requires the full observable record to match bit-for-bit: the hook log
(every hook with the account, positions, working orders, indicators and cache
a strategy could read at that instant), trades, orders, the equity / drawdown
/ returns curves, metrics, ``halted`` / ``halted_at``, ``per_instrument`` and
``rejected_entries``. Floats are compared as hex, so one ULP fails.

A failure prints the first differing hook-log entry field by field, then the
other differing sections.

The portfolio and tick scenarios also run through ``run_strategies`` (the
``runtime`` backend) and must match the same frozen record, so the multi-strategy
runtime is held to the legacy runners' behaviour, not to itself. The stream
scenarios run the same way through a one-strategy ``LiveNode`` fed the same
pushes.

Each scenario carries a marker naming the runner it covers, so one runner can
be checked alone::

    pytest tests/python/test_runner_fixtures.py -m portfolio_runner
    pytest tests/python/test_runner_fixtures.py -m tick_runner
    pytest tests/python/test_runner_fixtures.py -m tick_stream

Regenerating the fixtures
-------------------------
From the repo root, after building the extension::

    .venv/bin/python tests/python/golden/runner_fixtures.py

Regeneration is legitimate only for an INTENDED behaviour change of one of
those runners, with the reason recorded in the commit message. If this test
fails and you did not mean to change what a runner does, the fixtures are
right and the change is a regression: do not regenerate. Review the diff of
``runner_fixtures.json`` after regenerating; the unexpected hunks are the
bugs.

Recorded data is only what a strategy can see without lookahead. Values read
through ``ctx.series()`` and the other full-array context attributes are
deliberately absent.
"""

import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "golden"))

import runner_fixtures as rf  # noqa: E402

FIXTURES = rf.load()
DATASETS = FIXTURES["datasets"]
CODE_SCENARIOS = {sc["name"]: sc for sc in rf.scenario_table()}

MARKER = {"portfolio": "portfolio_runner", "tick": "tick_runner", "stream": "tick_stream"}
REGEN = "regenerate with `.venv/bin/python tests/python/golden/runner_fixtures.py` only for an intended behaviour change"

_HEX = re.compile(r"^-?(0x[0-9a-f.]+p[+-]\d+|nan|inf)$")
_EVENT_HOOKS = {
    "on_order_filled", "on_order_rejected", "on_order_accepted", "on_order_triggered",
    "on_order_canceled", "on_order_expired", "on_order_event", "on_algo_started",
    "on_algo_completed", "on_margin_call", "on_position_opened", "on_position_closed",
}
_PAYLOAD_NAMES = {
    "on_trade_tick": "timestamp price size symbol oi",
    "on_quote": "timestamp bid ask symbol bid_size ask_size",
    "on_composite_bar": "stream_id step unit timestamp open high low close volume symbol",
    "on_time_event": "name ts_scheduled ts_fired",
    "on_order_book": "timestamp bids asks symbol",
}


def _show(v):
    if isinstance(v, str) and _HEX.match(v):
        return f"{v} (= {float.fromhex(v)!r})"
    return repr(v)


def _diff(exp, act, path, out, limit=40):
    """Append 'path: expected X, got Y' lines. Lists report only the first
    differing element, which is the useful one for a time-ordered log."""
    if len(out) >= limit:
        return
    if isinstance(exp, dict) and isinstance(act, dict):
        for k in exp.keys() | act.keys():
            if k not in act:
                out.append(f"{path}.{k}: missing from actual (expected {_show(exp[k])[:120]})")
            elif k not in exp:
                out.append(f"{path}.{k}: not in fixture (actual {_show(act[k])[:120]})")
            else:
                _diff(exp[k], act[k], f"{path}.{k}", out, limit)
    elif isinstance(exp, list) and isinstance(act, list):
        if len(exp) != len(act):
            out.append(f"{path}: length expected {len(exp)}, got {len(act)}")
        for i, (a, b) in enumerate(zip(exp, act)):
            if a != b:
                _diff(a, b, f"{path}[{i}]", out, limit)
                break
    elif exp != act:
        out.append(f"{path}: expected {_show(exp)}, got {_show(act)}")


def _label(entry):
    """Name the positional payload so a diff reads as fields, not indexes."""
    e = dict(entry)
    payload = e.get("payload")
    if payload is None:
        return e
    if e["hook"] in _EVENT_HOOKS:
        e["payload"] = rf.label_event(payload)
    elif e["hook"] in _PAYLOAD_NAMES:
        e["payload"] = dict(zip(_PAYLOAD_NAMES[e["hook"]].split(), payload))
    return e


def _diff_hooks(exp, act, out):
    exp, act = rf.expand_hooks(exp), rf.expand_hooks(act)
    if len(exp) != len(act):
        out.append(f"hooks: {len(exp)} entries expected, {len(act)} produced")
    for i, (a, b) in enumerate(zip(exp, act)):
        if a != b:
            trail = " > ".join(h["hook"] for h in exp[max(0, i - 3) : i])
            out.append(f"hooks[{i}] is the first differing entry (after: {trail or 'start'})")
            a, b = _label(a), _label(b)
            _diff(a, b, f"hooks[{i}]", out)
            return


def describe_mismatch(expected, actual):
    out: list[str] = []
    for key in ("error", "warnings"):
        _diff(expected[key], actual[key], key, out)
    _diff_hooks(expected["hooks"], actual["hooks"], out)
    if expected["result"] is None or actual["result"] is None:
        if expected["result"] != actual["result"]:
            out.append("result: one side has no result (run refused vs completed)")
    else:
        for key in expected["result"]:
            _diff(expected["result"][key], actual["result"].get(key), f"result.{key}", out)
    _diff(expected["ops"], actual["ops"], "ops", out)
    return out


PARAMS = [
    pytest.param(name, id=f"{sc['runner']}-{name}", marks=getattr(pytest.mark, MARKER[sc["runner"]]))
    for name, sc in CODE_SCENARIOS.items()
]


def test_scenario_set_matches_fixture():
    assert list(CODE_SCENARIOS) == list(FIXTURES["scenarios"]), REGEN
    for name, sc in CODE_SCENARIOS.items():
        stored = FIXTURES["scenarios"][name]["descriptor"]
        assert json.loads(json.dumps(sc)) == stored, f"{name}: scenario definition drifted from the fixture; {REGEN}"


@pytest.mark.parametrize("name", PARAMS)
def test_scenario_matches_fixture(name):
    sc = json.loads(json.dumps(CODE_SCENARIOS[name]))
    expected = FIXTURES["scenarios"][name]["expected"]
    actual = json.loads(json.dumps(rf.run_scenario(sc, DATASETS)))
    if actual != expected:
        lines = describe_mismatch(expected, actual) or ["(differs, but no field-level difference found)"]
        pytest.fail(f"{name} diverged from the frozen behaviour:\n  " + "\n  ".join(lines[:40]), pytrace=False)


RUNTIME_PARAMS = [
    pytest.param(name, id=f"{sc['runner']}-{name}", marks=getattr(pytest.mark, MARKER[sc["runner"]]))
    for name, sc in CODE_SCENARIOS.items()
]


@pytest.mark.parametrize("name", RUNTIME_PARAMS)
def test_scenario_matches_fixture_on_the_runtime(name):
    sc = json.loads(json.dumps(CODE_SCENARIOS[name]))
    expected = FIXTURES["scenarios"][name]["expected"]
    actual = json.loads(json.dumps(rf.run_scenario(sc, DATASETS, backend="runtime")))
    if actual != expected:
        lines = describe_mismatch(expected, actual) or ["(differs, but no field-level difference found)"]
        on = "the live node" if sc["runner"] == "stream" else "the runtime"
        pytest.fail(f"{name} diverged from the frozen behaviour on {on}:\n  " + "\n  ".join(lines[:40]), pytrace=False)
