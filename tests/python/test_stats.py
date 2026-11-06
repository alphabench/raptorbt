"""Statistics that must say so, or refuse, when the data cannot carry them.

The oracle replays (`oracle/test_stats_replay.py`) pin the numbers to the libraries; these pin the
behaviour at the edges where a library prints a confident number that means nothing:

* a strategy set with no excess over its benchmark is not "significant at any level";
* Johansen on independent random walks: false positives by number of series and observations,
  and a flagged small sample;
* returns constant to a few ulps are refused, by GARCH and by the regime model alike;
* a GARCH fit at the integrated edge or at the bound of its shape says so;
* the regime model does not depend on the units of the returns.
"""

import numpy as np
import pytest

from raptorbt import stats
from raptorbt.errors import RaptorBTError


def refusal(fn, *args, **kw):
    with pytest.raises(RaptorBTError) as info:
        fn(*args, **kw)
    return info.value.code


# ------------------------------------------------------------------------------------- SPA


def test_spa_of_a_constant_excess_is_refused_not_p_zero():
    bench = np.random.default_rng(1).normal(0, 0.01, 300)
    assert refusal(stats.spa_test, {"a": np.zeros(300), "b": np.zeros(300)}) == "degenerate_series"
    assert (
        refusal(stats.spa_test, {"a": bench.copy(), "b": bench.copy()}, benchmark=bench)
        == "degenerate_series"
    )
    # one strategy that varies against the benchmark is enough to test
    ok = stats.spa_test({"a": bench.copy(), "b": 0.5 * bench}, benchmark=bench)
    assert 0.0 <= ok["p_value_spa_consistent"] <= 1.0


# ---------------------------------------------------------------------------------- Johansen


def walks(q, n, seed):
    rng = np.random.default_rng(seed)
    return {f"s{i}": 100.0 + np.cumsum(rng.normal(0, 1, n)) for i in range(q)}


def false_positives(q, n, count=40):
    raw = scaled = 0
    last = None
    for seed in range(count):
        last = stats.cointegration(walks(q, n, seed * 7 + q))["johansen"]
        raw += last["rank_at_5pct"] > 0
        scaled += last["rank"] > 0
    return raw / count, scaled / count, last


def test_johansen_on_independent_walks_scaled_rank_holds_where_the_raw_one_does_not():
    # Measured (400 samples each, driftless walks, one lagged difference, 5 % nominal), raw -> scaled:
    # q=6 n=60 43 % -> 2 %; q=12 n=60 100 % -> 0.3 %; q=12 n=120 80 % -> 1 %; q=12 n=500 15 % -> 3 %
    raw, scaled, last = false_positives(12, 60)
    assert raw > 0.9, raw  # the unscaled statistic is kept as it was, and is wrong here
    assert scaled <= 0.1, scaled
    assert last["small_sample"] is True
    assert last["scale_factor"] == pytest.approx((58 - 24) / 58)
    assert last["effective_observations"] == 58
    raw, scaled, last = false_positives(6, 60)
    assert raw > 0.2 and scaled <= 0.1, (raw, scaled)
    assert last["small_sample"] is True
    raw, scaled, last = false_positives(12, 500)
    assert scaled <= 0.15 and raw > scaled, (raw, scaled)
    assert last["small_sample"] is False


def test_johansen_scaled_statistics_are_the_raw_ones_times_the_factor():
    j = stats.cointegration(walks(4, 120, 3))["johansen"]
    f = (118 - 4 * 2) / 118
    assert j["scale_factor"] == pytest.approx(f)
    assert np.allclose(np.array(j["trace_statistics_scaled"]), f * np.array(j["trace_statistics"]), rtol=1e-14)
    assert j["small_sample"] is (f < 0.9)
    assert "warnings" in j and j["calibration"]


def test_johansen_still_finds_a_real_pair():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(0, 1, 250))
    u = np.zeros(250)
    for t in range(1, 250):
        u[t] = 0.5 * u[t - 1] + rng.normal()
    j = stats.cointegration({"y": 100.0 + x + u, "x": 100.0 + x})["johansen"]
    assert j["rank"] >= 1 and j["rank_at_5pct"] >= 1


# ----------------------------------------------------------------------- singular inputs


def test_returns_constant_to_a_few_ulps_are_refused_by_garch_and_regimes():
    r = 0.001 * (1.0 + 1e-14 * np.random.default_rng(2).normal(size=500))
    assert refusal(stats.garch, r) == "degenerate_series"
    assert refusal(stats.markov_regimes, r) == "degenerate_series"
    assert refusal(stats.garch, np.full(500, 0.001)) == "degenerate_series"
    assert refusal(stats.markov_regimes, np.zeros(500)) == "degenerate_series"
    # a real series of the same level is fitted
    ok = 0.001 + np.random.default_rng(2).normal(0, 0.01, 500)
    assert stats.garch(ok)["converged"]


def test_garch_at_the_integrated_edge_has_no_half_life_and_says_so():
    rng = np.random.default_rng(8)
    r = np.concatenate([rng.normal(0, 0.005, 750), rng.normal(0, 0.03, 750)])  # a step, no dynamics
    g = stats.garch(r)
    assert g["persistence"] > 1.0 - 1e-6
    assert g["integrated"] is True and g["stationary_variance"] is False
    assert g["half_life_periods"] is None and g["unconditional_vol_per_period"] is None
    assert any("1e-6" in w for w in g["warnings"])


def test_garch_flags_a_t_shape_at_its_bound_and_missing_standard_errors():
    rng = np.random.default_rng(7)
    r = np.where(rng.random(500) < 0.05, rng.choice([-0.01, 0.01], 500), 0.0)  # an illiquid name
    g = stats.garch(r)
    assert g["shape_at_bound"] is True and g["parameters"]["nu"] == pytest.approx(2.05, abs=1e-4)
    assert g["std_errors"] is None
    assert any("bound" in w for w in g["warnings"]) and any("standard errors" in w for w in g["warnings"])


def test_an_ordinary_garch_fit_raises_no_warning():
    rng = np.random.default_rng(6)
    s2, out = 1e-4, []
    for _ in range(2000):
        e = np.sqrt(s2) * rng.standard_t(6) * np.sqrt(4.0 / 6.0)
        out.append(e)
        s2 = 1e-6 + 0.08 * e * e + 0.9 * s2
    g = stats.garch(np.array(out))
    assert g["warnings"] == [] and not g["integrated"] and not g["shape_at_bound"]
    assert g["half_life_periods"] is not None


# ------------------------------------------------------------------------------- units


def two_regime_returns(n=1500, seed=5):
    rng = np.random.default_rng(seed)
    state, out = 0, []
    for _ in range(n):
        if rng.random() > 0.98:
            state = 1 - state
        out.append(rng.normal(0.0, 0.005 if state == 0 else 0.03))
    return np.array(out)


@pytest.mark.parametrize("scale", [1e-9, 1e-7, 1e-5, 1e-3, 1e3, 1e5])
def test_regimes_do_not_depend_on_the_units_of_the_returns(scale):
    # returns times s: volatilities times s, the same transitions, a log-likelihood moved by
    # -n ln(s) (it is of the series in percent). Measured before the rescale: 1e-9 gave
    # volatilities 2 % off and transitions 0.03 off, 1e-7 did not converge.
    r = two_regime_returns()
    base = stats.markov_regimes(r)
    got = stats.markov_regimes(r * scale)
    for a, b in zip(got["regime_stats"], base["regime_stats"]):
        assert a["vol_per_period"] / scale == pytest.approx(b["vol_per_period"], rel=1e-6)
        assert a["stay_probability"] == pytest.approx(b["stay_probability"], abs=1e-6)
    shift = -len(r) * np.log(scale)
    assert got["log_likelihood"] - base["log_likelihood"] == pytest.approx(shift, rel=1e-9)
    assert base["rescale"] == 1.0


@pytest.mark.parametrize("scale", [1e-3, 1e-1, 1e1, 1e3])
def test_garch_does_not_depend_on_the_units_of_the_returns(scale):
    rng = np.random.default_rng(6)
    s2, out = 1e-4, []
    for _ in range(2000):
        e = np.sqrt(s2) * rng.standard_t(6) * np.sqrt(4.0 / 6.0)
        out.append(e)
        s2 = 1e-6 + 0.08 * e * e + 0.9 * s2
    r = np.array(out)
    base, got = stats.garch(r), stats.garch(r * scale)
    assert got["persistence"] == pytest.approx(base["persistence"], abs=1e-3)
