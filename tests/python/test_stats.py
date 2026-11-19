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


# ------------------------------------------------------------------------ predictive_test


def planted_lead():
    """40 observations at lag 1, h 1 with `y = 0.01 x + 0.001 z`: x = (-1, -1, 1, 1) repeated,
    z = (1, -1, 1, -1), so sum x z = 0 and mean x = mean z = 0 per period. The slope is exactly
    0.01 and the intercept 0; the residual is 0.001 z, SSR = 40e-6, so the t is
    0.01 / sqrt(1e-6 / 38) = 10 sqrt(38). The rule sign(x) earns 0.01 + 0.001 x z, mean 0.01; its
    position changes are 1 (from flat) and 2 at each of 19 flips: mean 39 / 40 = 0.975, and the
    breakeven is 0.01 / 0.975 = 102.564 bps."""
    x = np.tile([-1.0, -1.0, 1.0, 1.0], 10)
    z = np.tile([1.0, -1.0, 1.0, -1.0], 10)
    r = 0.01 * x + 0.001 * z
    prices = 100.0 * np.concatenate([[1.0, 1.0], np.cumprod(1.0 + r)])
    return np.concatenate([x, [np.nan, np.nan]]), prices, r


def test_a_planted_lead_has_its_hand_computed_slope_t_and_breakeven():
    signal, prices, r = planted_lead()
    out = stats.predictive_test(signal, prices, horizons=[1], quantiles=2, cost_bps_per_side=10.0)
    h = out["horizons"]["1"]
    assert (h["n"], h["n_independent"]) == (40, 40)
    assert h["slope"] == pytest.approx(0.01, abs=1e-12) and h["intercept"] == pytest.approx(0.0, abs=1e-12)
    assert h["t_nonoverlap"] == pytest.approx(10.0 * np.sqrt(38.0), rel=1e-9)
    # x has two levels (ranks 10.5 and 30.5) and y four (ranks 5.5, 15.5, 25.5, 35.5, ten each, when
    # its ties hold exactly): cov 100, sd 10 and sqrt(125), so 2 / sqrt(5) = 0.894. The returns come
    # back from a product of prices, so the ties are broken by rounding at 1e-16 and the value
    # falls between that and the 10 / sqrt(133.25) = 0.866 of a total order
    assert h["hit_rate"] == 1.0
    assert 10.0 / np.sqrt(133.25) - 1e-9 <= h["rank_corr"] <= 2.0 / np.sqrt(5.0) + 1e-9
    assert h["quantile_spread"] == pytest.approx(0.02, abs=1e-12)
    assert h["rule_mean"] == pytest.approx(0.01, abs=1e-12)
    assert h["turnover"] == pytest.approx(0.975)
    assert h["breakeven_cost_bps_per_side"] == pytest.approx(0.01 / 0.975 * 1e4)
    assert h["net_mean"] == pytest.approx(0.01 - 0.001 * 0.975)
    assert h["t_net"] > 10 and out["flags"] == []
    assert out["warnings"] == []  # 40 independent observations: past the 30 of small_sample
    few = stats.predictive_test(signal[:22], prices[:22], horizons=[1])  # 20 observations
    assert any(w.startswith("small_sample") for w in few["warnings"])
    assert h["forward_return"][0] == pytest.approx(r[0]) and np.isnan(h["forward_return"][40])
    # a cost above the breakeven is flagged
    dear = stats.predictive_test(signal, prices, horizons=[1], cost_bps_per_side=150.0)
    assert any("does not pay its costs" in f for f in dear["flags"])
    assert "net_mean" not in stats.predictive_test(signal, prices, horizons=[1])["horizons"]["1"]


def test_a_session_observation_runs_from_the_known_signal_to_the_close():
    # twelve sessions of four hourly bars; the signal is known from the second bar, entered one
    # bar later (bar 2) and held to bar 3, a return of exactly 0.01 v for the session value v = +-1
    day = 86_400 * 10**9
    ts = np.concatenate([k * day + np.arange(4) * 3_600 * 10**9 for k in range(12)])
    v = np.where(np.arange(12) % 2 == 0, 1.0, -1.0)
    signal = np.concatenate([[np.nan, a, a, a] for a in v])
    prices = np.concatenate([[100.0, 100.0, 100.0, 100.0 * (1 + 0.01 * a)] for a in v])
    out = stats.predictive_test(signal, prices, to_session_close=True, timestamps=ts)
    h = out["horizons"]["session"]
    assert (h["n"], h["n_independent"]) == (12, 12)
    assert h["slope"] == pytest.approx(0.01, abs=1e-12)
    assert h["forward_return"][1] == pytest.approx(0.01) and h["forward_return"][5] == pytest.approx(-0.01)
    assert np.isnan(h["forward_return"][0]) and np.isnan(h["forward_return"][2])
    # a zone that moves midnight across the first bars splits the sessions elsewhere: IST puts the
    # 18:30 UTC boundary a day earlier, here every bar is before it, so the sessions are the same
    ist = stats.predictive_test(signal, prices, to_session_close=True, timestamps=ts, session=19_800 * 10**9)
    assert ist["horizons"]["session"]["n"] == 12
    assert refusal(stats.predictive_test, signal, prices, to_session_close=True) == "invalid_input"
    assert refusal(stats.predictive_test, signal, prices, to_session_close=True, timestamps=ts, horizons=[1]) == "invalid_input"


def test_a_session_pays_entry_and_exit_every_session():
    # six long sessions then six short, each returning 0.01 v from the entry at bar 2 to its close
    # (overnight excluded): every session is a round trip, turnover 2, rule_mean 0.01,
    # breakeven 0.01 / 2 * 1e4 = 50 bps per side, net at 10 bps = 0.01 - 0.001 * 2 = 0.008
    day = 86_400 * 10**9
    ts = np.concatenate([k * day + np.arange(4) * 3_600 * 10**9 for k in range(12)])
    v = np.array([1.0] * 6 + [-1.0] * 6)
    signal = np.concatenate([[np.nan, a, a, a] for a in v])
    prices = np.concatenate([[100.0, 100.0, 100.0, 100.0 * (1 + 0.01 * a)] for a in v])
    h = stats.predictive_test(
        signal, prices, to_session_close=True, timestamps=ts, cost_bps_per_side=10.0
    )["horizons"]["session"]
    assert h["rule_mean"] == pytest.approx(0.01, abs=1e-12)
    assert h["turnover"] == pytest.approx(2.0, abs=1e-12)
    assert h["breakeven_cost_bps_per_side"] == pytest.approx(50.0, abs=1e-9)
    assert h["net_mean"] == pytest.approx(0.008, abs=1e-12)


def test_no_skill_persistent_signal_rejects_near_five_percent_on_the_headline_t():
    # Measured on 1,500 samples (iid 1 % returns, 1,000 bars, h = 20, an AR(1) signal of
    # persistence 0.98): t_nonoverlap p < 0.05 on 5.6 %, sd 1.03; t_hac |t| > 1.96 on 14.5 %, sd
    # 1.34. Here 400 seeded samples; the bound is four binomial standard errors of 5 %.
    count, h = 400, 20
    reject = hac = 0
    for seed in range(count):
        rng = np.random.default_rng(5000 + seed)
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 1001))
        z = rng.normal(size=1001)
        x = np.empty(1001)
        x[0] = z[0]
        for i in range(1, 1001):
            x[i] = 0.98 * x[i - 1] + z[i]
        r = stats.predictive_test(x, px, horizons=[h])["horizons"][str(h)]
        reject += r["p_nonoverlap"] < 0.05
        hac += abs(r["t_hac"]) > 1.96
    se = np.sqrt(0.05 * 0.95 / count)
    assert abs(reject / count - 0.05) < 4 * se, reject / count
    assert hac / count > 0.05 + 4 * se, hac / count  # Newey-West still over-rejects: pinned


def test_bars_after_t_do_not_change_what_was_scored_before_it():
    rng = np.random.default_rng(11)
    n, lag, h, t_cut = 300, 1, 5, 200
    px = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    sig = rng.normal(size=n)
    later = px.copy()
    later[t_cut:] *= rng.uniform(0.2, 5.0, n - t_cut)  # poisoned from bar t_cut on
    a = stats.predictive_test(sig, px, horizons=[h], lag=lag)["horizons"][str(h)]["forward_return"]
    b = stats.predictive_test(sig, later, horizons=[h], lag=lag)["horizons"][str(h)]["forward_return"]
    last = t_cut - 1 - lag - h  # the last signal bar whose window ends before the poison
    assert np.array_equal(a[: last + 1], b[: last + 1])
    assert not np.array_equal(a[last + 1 : t_cut], b[last + 1 : t_cut], equal_nan=True)
    # and a truncated run agrees with the full one on every bar its windows cover
    cut = stats.predictive_test(sig[:t_cut], px[:t_cut], horizons=[h], lag=lag)["horizons"][str(h)]
    assert np.array_equal(cut["forward_return"][: last + 1], a[: last + 1])


def test_a_signal_that_cannot_be_tested_says_so():
    signal, prices, _ = planted_lead()
    flat = stats.predictive_test(np.ones(42), prices, horizons=[1])
    assert flat["horizons"]["1"]["signal_constant"] and np.isnan(flat["horizons"]["1"]["t_nonoverlap"])
    assert any(w.startswith("signal_constant") for w in flat["warnings"])
    sparse = np.full(42, np.nan)
    sparse[:12] = np.arange(12.0)
    assert any(w.startswith("mostly_nan") for w in stats.predictive_test(sparse, prices, horizons=[1])["warnings"])
    assert refusal(stats.predictive_test, np.full(42, np.nan), prices, horizons=[1]) == "insufficient_observations"
    assert refusal(stats.predictive_test, signal, prices[:-1], horizons=[1]) == "invalid_input"
    bad = prices.copy()
    bad[5] = -1.0
    assert refusal(stats.predictive_test, signal, bad, horizons=[1]) == "invalid_input"
    assert refusal(stats.predictive_test, signal, prices) == "invalid_input"  # no horizons
    # a signal that never goes below zero has an always-long rule: said
    assert any(w.startswith("nonnegative_signal") for w in stats.predictive_test(np.abs(signal), prices, horizons=[1])["warnings"])


def test_hac_tstat_warns_when_the_series_cannot_be_negative():
    rng = np.random.default_rng(3)
    squared = rng.normal(0.0, 0.01, 500) ** 2
    out = stats.hac_tstat(squared)
    assert out["hac_t"] > 10  # positive by construction
    assert any(w.startswith("nonnegative_series") for w in out["warnings"])
    assert stats.hac_tstat(rng.normal(0.001, 0.01, 500))["warnings"] == []


# ------------------------------------------------------------------ fields the answers carry


def test_garch_reports_the_variances_beside_the_volatilities():
    rng = np.random.default_rng(6)
    g = stats.garch(rng.normal(0, 0.01, 1000), horizon=5)
    vols, var = g["forecast_vol_per_period"], g["forecast_variance_per_period"]
    assert len(var) == len(vols) == 5
    assert var == pytest.approx([v * v for v in vols], rel=1e-12)
    assert g["forecast_cumulative_variance"] == pytest.approx(sum(var), rel=1e-14)
    assert g["forecast_cumulative_vol"] ** 2 == pytest.approx(g["forecast_cumulative_variance"], rel=1e-12)


def test_cointegration_carries_the_eigenvalues_and_max_eigenvalue_statistics():
    rng = np.random.default_rng(11)
    x = np.cumsum(rng.normal(size=400))
    y = 0.8 * x + rng.normal(size=400)
    jo = stats.cointegration({"x": x, "y": y})["johansen"]
    eig, lr2 = jo["eigenvalues"], jo["max_eigenvalue_statistics"]
    assert len(eig) == len(lr2) == len(jo["max_eigenvalue_critical_5pct"]) == 2
    assert eig[0] >= eig[1] > 0.0 and all(e < 1.0 for e in eig)
    t = jo["effective_observations"]
    assert lr2[0] == pytest.approx(-t * np.log(1.0 - eig[0]), rel=1e-12)
    assert lr2[1] == pytest.approx(-t * np.log(1.0 - eig[1]), rel=1e-12)


# ------------------------------------------------------- user parameters that used to panic


def test_audit_bars_refuses_zero_min_bars_and_a_gap_below_one_bar():
    x = np.array([100.0])
    args = (np.array([0.0]), x, x, x, x, np.array([1.0]))
    assert refusal(stats.audit_bars, *args, interval_seconds=60, min_bars=0) == "invalid_input"
    t = np.arange(10) * 60.0
    c = np.linspace(100.0, 101.0, 10)
    v = np.ones(10)
    for bad in (0.0, 0.5, float("nan"), float("inf")):
        code = refusal(stats.audit_bars, t, c, c, c, c, v, interval_seconds=60, intraday_gap_bars=bad)
        assert code == "invalid_input"


def test_predictive_test_refuses_a_lag_or_horizon_past_the_bars():
    signal = np.arange(50.0) - 25
    prices = np.linspace(100, 110, 50)
    for lag in (2**64 - 1, 50, 60):
        assert (
            refusal(stats.predictive_test, signal, prices, horizons=[1], lag=lag)
            == "insufficient_observations"
        )
    assert (
        refusal(stats.predictive_test, signal, prices, horizons=[2**64 - 1], lag=1)
        == "insufficient_observations"
    )


def test_factor_report_refuses_a_huge_lag():
    from raptorbt import expressions as ex

    rng = np.random.default_rng(3)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (60, 12)), axis=0)
    factor = rng.normal(size=(60, 12))
    for lag in (2**64 - 1, 2**63):
        assert refusal(ex.factor_report, factor, prices, horizons=[1], lag=lag) == "insufficient_observations"
