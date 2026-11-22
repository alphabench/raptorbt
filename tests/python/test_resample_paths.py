"""research.resample_paths: the Monte Carlo of a finished result's paths."""

import math

import numpy as np
import pytest

from raptorbt import research
from raptorbt.errors import RaptorBTError

PCTS = ("p05", "p50", "p95", "p99")
NOISY = list(np.random.default_rng(4).normal(0.0004, 0.012, 300))
TRADES = list(np.random.default_rng(5).normal(0.002, 0.02, 80))


def test_the_paths_are_a_function_of_the_seed():
    for kw in ({"method": "block"}, {"method": "trades"}, {"method": "trades", "replace": False}):
        values = NOISY if kw["method"] == "block" else TRADES
        a = research.resample_paths(values, draws=400, seed=7, **kw)
        assert a == research.resample_paths(values, draws=400, seed=7, **kw)
        assert a["max_drawdown"] != research.resample_paths(values, draws=400, seed=8, **kw)["max_drawdown"]
        assert a["n"] == len(values) and a["draws"] == 400 and a["seed"] == 7


def test_constant_returns_end_where_they_compound_and_never_fall():
    flat = [0.001] * 60
    p = research.resample_paths(flat, seed=1, draws=200, periods_per_year=252)
    end = 1.001**60 - 1
    assert all(p["ending_return"][k] == pytest.approx(end, abs=1e-12) for k in PCTS)
    assert p["ending_return"]["observed"] == pytest.approx(end, abs=1e-12)
    assert p["annual_return"]["p50"] == pytest.approx(1.001**252 - 1, abs=1e-9)
    assert all(p["max_drawdown"][k] == 0.0 for k in PCTS) and p["max_drawdown"]["observed"] == 0.0
    assert p["prob_loss"] == 0.0
    assert [r["probability"] for r in p["prob_drawdown"]] == [0.0, 0.0, 0.0]
    assert [r["level"] for r in p["prob_drawdown"]] == [0.1, 0.2, 0.3]
    h = p["drawdown_histogram"]
    assert len(h["edges"]) == 21 and h["edges"][0] == 0.0 and h["edges"][-1] == 1.0
    assert h["counts"][0] == 200 and sum(h["counts"]) == 200


def test_block_length_defaults_to_the_cube_root_of_the_length():
    assert research.resample_paths([0.001] * 60, seed=1, draws=10)["block"] == 4.0
    assert research.resample_paths(NOISY, seed=1, draws=10)["block"] == float(round(300 ** (1 / 3)))
    assert research.resample_paths(NOISY, seed=1, draws=10, block=12)["block"] == 12.0
    assert research.resample_paths(TRADES, method="trades", seed=1, draws=10)["block"] is None


def test_percentiles_are_ordered_and_probabilities_fall_with_the_level():
    for kw in ({"method": "block"}, {"method": "trades"}, {"method": "trades", "replace": False}):
        values = NOISY if kw["method"] == "block" else TRADES
        p = research.resample_paths(values, seed=3, drawdown_levels=[0.05, 0.1, 0.2, 0.4], **kw)
        for key in ("ending_return", "max_drawdown"):
            q = [p[key][k] for k in PCTS]
            assert q == sorted(q), (kw, key, q)
        assert 0.0 <= p["max_drawdown"]["p05"] and p["max_drawdown"]["p99"] <= 1.0
        probs = [r["probability"] for r in p["prob_drawdown"]]
        assert probs == sorted(probs, reverse=True)
        assert 0.0 <= p["prob_loss"] <= 1.0
        assert sum(p["drawdown_histogram"]["counts"]) == p["draws"]
        assert p["drawdown_histogram"]["edges"][-1] >= p["max_drawdown"]["p99"]


def test_a_shuffle_keeps_the_ending_return_and_says_so():
    p = research.resample_paths(TRADES, method="trades", replace=False, seed=2, draws=500)
    observed = math.prod(1 + t for t in TRADES) - 1
    assert p["ending_return_fixed"] is True and p["replace"] is False
    assert p["ending_return"]["observed"] == pytest.approx(observed, rel=1e-12)
    assert {p["ending_return"][k] for k in PCTS} == {p["ending_return"]["observed"]}
    assert p["prob_loss"] == (1.0 if observed < 0 else 0.0)
    assert p["max_drawdown"]["p05"] < p["max_drawdown"]["p95"]  # only the drawdown varies
    boot = research.resample_paths(TRADES, method="trades", seed=2, draws=500)
    assert boot["ending_return_fixed"] is False and boot["ending_return"]["p05"] < boot["ending_return"]["p95"]


def test_a_shuffle_of_one_gain_and_two_losses_is_enumerable():
    """The non-zero trades come in a uniform order of (+0.1, -0.1, -0.1). Two losses in a row
    (four orders of six) fall 1 - 0.9 ** 2 = 0.19 from the peak; a gain between them leaves the
    second loss 1 - 0.9 * 1.1 * 0.9 = 0.109 under the start."""
    trades = [0.0, 0.0, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, -0.1, 0.0]
    p = research.resample_paths(
        trades, method="trades", replace=False, seed=9, draws=6000, drawdown_levels=[0.1, 0.15]
    )
    for k in PCTS:
        assert min(abs(p["max_drawdown"][k] - 0.19), abs(p["max_drawdown"][k] - 0.109)) < 1e-12
    assert p["max_drawdown"]["observed"] == pytest.approx(0.19, abs=1e-12)
    at_10, at_15 = (r["probability"] for r in p["prob_drawdown"])
    assert at_10 == 1.0
    assert at_15 == pytest.approx(2 / 3, abs=0.03)
    assert p["ending_return"]["observed"] == pytest.approx(1.1 * 0.9 * 0.9 - 1, abs=1e-14)
    assert p["prob_loss"] == 1.0


def test_a_total_loss_ruins_the_path():
    p = research.resample_paths([-1.0] + [0.1] * 9, method="trades", replace=False, seed=1,
                                draws=50, drawdown_levels=[1.0])
    assert p["ending_return"]["observed"] == -1.0 and p["max_drawdown"]["observed"] == 1.0
    assert p["prob_drawdown"] == [{"level": 1.0, "probability": 1.0}]


@pytest.mark.parametrize(
    ("values", "kw", "code"),
    [
        ([0.01] * 9, {}, "insufficient_observations"),
        ([0.01] * 9, {"method": "trades"}, "insufficient_observations"),
        ([0.01] * 9 + [float("nan")], {}, "invalid_input"),
        ([0.01] * 9 + [float("inf")], {"method": "trades"}, "invalid_input"),
        ([0.01] * 9 + [-1.5], {}, "invalid_input"),
        ([[0.01] * 10], {}, "invalid_input"),
        (["x"] * 10, {}, "invalid_input"),
        ([0.01] * 10, {"draws": 0}, "invalid_input"),
        ([0.01] * 10, {"draws": 100_001}, "invalid_input"),
        ([0.01] * 10, {"drawdown_levels": [0.0]}, "invalid_input"),
        ([0.01] * 10, {"drawdown_levels": [1.5]}, "invalid_input"),
        ([0.01] * 10, {"drawdown_levels": [0.1] * 17}, "invalid_input"),
        ([0.01] * 10, {"periods_per_year": 0}, "invalid_input"),
        ([0.01] * 10, {"block": 0.5}, "invalid_input"),
        ([0.01] * 10, {"method": "trades", "block": 5}, "invalid_input"),
        ([0.01] * 10, {"replace": False}, "invalid_input"),
        ([0.01] * 10, {"method": "paths"}, "invalid_input"),
    ],
)
def test_bad_input_is_refused(values, kw, code):
    with pytest.raises(RaptorBTError) as err:
        research.resample_paths(values, seed=1, **kw)
    assert err.value.code == code
    assert err.value.operation == "raptorbt.research.resample_paths"
    assert err.value.skill == "optimization"


def test_the_minimum_is_ten_observations():
    with pytest.raises(RaptorBTError, match="9 observations, this statistic needs 10"):
        research.resample_paths([0.01] * 9, seed=1)
    assert research.resample_paths([0.01] * 10, seed=1, draws=5)["n"] == 10
