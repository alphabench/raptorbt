"""Research statistics against the reference implementation, replayed from fixtures.

The fixture holds the numbers a reference implementation produced for these inputs. These tests replay them here. Both engines use the same SplitMix64 stream and the
same correctly rounded sums, so draws, designs and windows are equal, and the floats agree to
1e-12 relative (the deflated Sharpe's normal quantile is a fixed bisection in both).
"""

import json
import math
from pathlib import Path

import pytest

from raptorbt import research

HERE = Path(__file__).parent
FIX = json.loads((HERE / "research_fixture.json").read_text())
REL = 1e-12


def close(a, b):
    return a == b or math.isclose(a, b, rel_tol=REL, abs_tol=1e-15)


def assert_same(got, want):
    assert set(got) == set(want)
    for k, w in want.items():
        g = got[k]
        if isinstance(w, dict):
            assert_same(g, w)
        elif isinstance(w, float):
            assert close(g, w), (k, g, w)
        else:
            assert g == w, (k, g, w)


@pytest.mark.parametrize("case", FIX["deflated_sharpe"])
def test_deflated_sharpe(case):
    got = research.deflated_sharpe(
        case["sharpes"], case["best"], samples=case["samples"], skew=case["skew"],
        kurtosis=case["kurtosis"], periods_per_year=case["periods_per_year"], luck=case["luck"],
    )
    assert_same(got, case["result"])


@pytest.mark.parametrize("case", FIX["stationary_bootstrap"])
def test_stationary_bootstrap(case):
    got = research.stationary_bootstrap(
        case["returns"], draws=case["draws"], block=case["block"], seed=case["seed"],
        periods_per_year=case["periods_per_year"],
    )
    assert_same(got, case["result"])


@pytest.mark.parametrize("case", FIX["without_best"])
def test_without_best(case):
    assert close(research.without_best(case["returns"], case["top"]), case["result"])


@pytest.mark.parametrize("case", FIX["designs"])
def test_designs(case):
    if case["kind"] == "grid":
        got = research.Design.grid(case["space"]).points
    else:
        space = {
            k: (tuple(v) if k in case["ranges"] else v) for k, v in case["space"].items()
        }
        got = research.Design.random(space, case["n"], seed=case["seed"]).points
    assert len(got) == len(case["points"])
    for g, w in zip(got, case["points"]):
        assert set(g) == set(w)
        for k, v in w.items():
            assert close(g[k], v) if isinstance(v, float) else g[k] == v


@pytest.mark.parametrize("case", FIX["windows"])
def test_walk_forward_windows(case):
    wf = research.WalkForward(
        list(range(case["days"])), train=case["train"], test=case["test"], step=case["step"]
    )
    got = [
        {"index": w.index, "train": list(w.train), "test": list(w.test)} for w in wf.windows()
    ]
    assert got == case["windows"]
