"""Replay the committed statistics fixtures through `raptorbt.stats`. Needs numpy only.

`fixtures/stats/stats_<family>.json.gz` hold, per case, the arguments the reference function was
called with (`input`), its answer at full precision (`reference_full`) or the error it raised
(`tests/oracle/stats_reference.py` says where each family's reference comes from), the numbers
read from the libraries (`lib`: statsmodels, arch) and independent answers (`known`). Floats are
the hex of their IEEE-754 bits. Every case is called through the public function and held to the
bar in `design-stats.md`:

    ADF and KPSS      lags exact; statistic and p-value 1e-8; critical values 1e-8
    Engle-Granger     1e-8; Johansen eigen-statistics 1e-8, the vector 1e-6
    HAC               1e-10 against the library (1e-6 where `known` has the exact standard error)
    Hurst, events     1e-10, scales exact
    PCA               1e-8, a component compared up to sign when its raw loading sum is near 0
    SPA               arch's own bootstrap indices replayed through the native at 1e-10; the
                      engine's seeded draws within four Monte Carlo standard errors of arch
    PBO               1e-12;  impact 1e-12;  surface 1e-12;  the bar audit exact
    predictive        statsmodels' OLS slope, intercept and t, the HAC t of the slope: 1e-9
    GARCH             log-likelihood no more than 1e-3 below arch's; where the two reach the same
                      optimum, the parameters 5e-3 and the forecast and volatility 1e-3
    regimes           log-likelihood no more than 1e-4 below statsmodels'; where the two reach the
                      same optimum, the parameters and the probabilities 1e-3

The cases the oracle itself answers inexactly and the panels both refuse are not re-derived here:
the lists are read out of the Rust tests that own them (`tests/stats_coint.rs`,
`tests/stats_stationarity.rs`), so there is one source.
A fixture family that replays no case fails.
"""

import gzip
import importlib.util
import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from raptorbt import _raptorbt, stats
from raptorbt.errors import RaptorBTError

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "stats"
TESTS = HERE.parents[1]  # tests/
ORACLE = TESTS / "oracle"


# ------------------------------------------------------------------------------ the fixtures


def decode(o):
    """Hex floats and ints to numpy; a `Columns` ({order, columns}) to an ordered dict."""
    if isinstance(o, dict):
        for kind, dtype in (("f8", ">f8"), ("i4", ">i4")):
            if kind in o:
                a = np.frombuffer(bytes.fromhex(o[kind]), dtype=dtype)
                a = a.astype(np.float64 if kind == "f8" else np.int64)
                return a.reshape(o["shape"]) if "shape" in o else float(a[0])
        if set(o) == {"order", "columns"}:
            return {k: decode(o["columns"][k]) for k in o["order"]}
        return {k: decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [decode(v) for v in o]
    return o


_cache = {}


def fixture(family):
    if family not in _cache:
        with gzip.open(FIXTURES / f"stats_{family}.json.gz", "rt") as f:
            _cache[family] = json.load(f)
    return _cache[family]


def _load_extract_helpers():
    """`hac_se_exact` and `pca_rank_aware_t`, which cut the Rust extracts: one source."""
    spec = importlib.util.spec_from_file_location("extract_stats", ORACLE / "extract_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- the lists the Rust tests own


def rust_initialiser(path, name):
    text = (TESTS / path).read_text()
    m = re.search(rf"const {name}:[^=]*=(.*?);\n", text, re.S)
    assert m, f"{path}: no const {name}"
    return m.group(1)


def rust_names(path, name):
    return set(re.findall(r'"([^"]+)"', rust_initialiser(path, name)))


def rust_pairs(path, name):
    """`[("case", number), ...]` as a dict; `f64::INFINITY` is `inf`."""
    out = {}
    for case, value in re.findall(r'\("([^"]+)",\s*([^)]+?)\)', rust_initialiser(path, name)):
        out[case] = math.inf if "INFINITY" in value else float(value)
    return out


def rust_triples(path, name):
    return {
        case: (float(exact), float(bound))
        for case, exact, bound in re.findall(
            r'\("([^"]+)",\s*([^,]+?),\s*([^)]+?)\)', rust_initialiser(path, name)
        )
    }


# -------------------------------------------------------------------------------- comparing


def close(got, want, tol, floor=1.0):
    """`want` is the fixture's number; `None` there means NaN or infinity."""
    if want is None:
        return got is None or not math.isfinite(float(got))
    if got is None:
        return False
    got, want = float(got), float(want)
    if math.isnan(want):
        return math.isnan(got)
    if got == want:
        return True
    return abs(got - want) <= tol * max(abs(want), floor)


def same(got, want, where, tol, floor=1.0, loose=None, skip=()):
    """`got` carries every key of `want`; floats within `tol` (`loose` names looser paths);
    strings, ints and booleans exact. Keys `got` has beyond `want` are additive."""
    loose = loose or {}
    if where in skip:
        return
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{where}: {type(got).__name__}, not a dict"
        for k, w in want.items():
            assert k in got, f"{where}.{k}: missing"
            same(got[k], w, f"{where}.{k}", tol, floor, loose, skip)
    elif isinstance(want, (list, np.ndarray)):
        got = got if isinstance(got, np.ndarray) else list(got)
        assert len(got) == len(want), f"{where}: length {len(got)}, want {len(want)}"
        for i, w in enumerate(want):
            same(got[i], w, f"{where}[{i}]", tol, floor, loose, skip)
    elif isinstance(want, (bool, np.bool_)):
        assert bool(got) is bool(want) and isinstance(got, (bool, np.bool_)), f"{where}: {got!r}"
    elif isinstance(want, (int, np.integer)):
        assert got == want, f"{where}: {got!r}, want {want!r}"
    elif want is None or isinstance(want, float):
        t = next((v for k, v in loose.items() if k in where), tol)
        assert close(got, want, t, floor), f"{where}: {got!r}, want {want!r} (tolerance {t:g})"
    else:
        assert got == want, f"{where}: {got!r}, want {want!r}"


def refusal_code(message):
    if "this statistic needs" in message or "too few to audit" in message:
        return "insufficient_observations"
    if "constant" in message or "no volume in these bars" in message or "Roll's" in message:
        return "degenerate_series"
    if "did not converge" in message:
        return "did_not_converge"
    return "invalid_input"


def check_refusal(call, error, *, exact_message, code=None):
    """The reference raised `error`; the engine must raise the typed error for it (the code the
    reference names, or for a library's own exception the code its message reads as)."""
    with pytest.raises(RaptorBTError) as info:
        call()
    err = info.value
    message = error["message"]
    want = code or error.get("code") or refusal_code(message)
    assert err.code == want, f"{err.code}: {err.message}"
    if exact_message:
        assert err.message == message
    assert isinstance(err, ValueError)


def run_family(family, replay, floor):
    """Replay `replay(case)` for each case of the family; it returns True when it replayed one."""
    cases = fixture(family)["cases"]
    assert cases, f"{family}: the fixture holds no case"
    replayed = 0
    for case in cases:
        try:
            if replay(case):
                replayed += 1
        except Exception as e:
            raise AssertionError(f"{family}/{case['name']}: {type(e).__name__}: {e}") from e
    assert replayed >= floor, f"{family}: only {replayed} cases replayed (at least {floor})"


def args_of(case):
    return decode(case["input"]["args"]), decode(case["input"]["kwargs"])


def reference_of(case):
    full = case.get("reference_full")
    if full is None:
        return None, None
    return (decode(full["ok"]), None) if "ok" in full else (None, full["error"])


# ----------------------------------------------------------------------------- stationarity

ORACLE_INEXACT = rust_pairs("stats_stationarity.rs", "ORACLE_INEXACT")


def replay_stationarity(case):
    (series, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if ok is None and error is None:
        return False  # a fixed-lag library case: the public function searches the lag
    name = case["name"]
    if error:
        check_refusal(lambda: stats.stationarity(series, **kw), error, exact_message=False)
        return True
    got = stats.stationarity(series, **kw)
    if name in ORACLE_INEXACT:
        # the oracle is limited by the conditioning of the regression, not the algorithm: held to
        # what is true of the answer, and to the oracle's statistic where it is good enough
        assert 0.0 <= got["adf"]["p_value"] <= 1.0 and math.isfinite(got["adf"]["statistic"])
        tol = ORACLE_INEXACT[name]
        if math.isfinite(tol):
            assert close(got["adf"]["statistic"], ok["adf"]["statistic"], tol), name
        same(got["kpss"], ok["kpss"], "kpss", 1e-8)
        return True
    same(got, ok, "stationarity", 1e-8)
    return True


# ---------------------------------------------------------------------------- cointegration

COINT = "stats_coint.rs"
MAY_REFUSE = rust_names(COINT, "MAY_REFUSE")
REFUSED = rust_names(COINT, "REFUSED")
EXACT_INTERCEPT = rust_triples(COINT, "EXACT_INTERCEPT")
LOOSE = rust_pairs(COINT, "LOOSE")


def replay_cointegration(case):
    name = case["name"]
    if name.startswith("half_life_"):
        return False  # the half-life of a bare spread: not a public function
    (cols, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error and error["type"] == "IndexError":
        # statsmodels' MacKinnon table stops at six series; the engine answers without a p-value
        got = stats.cointegration(cols, **kw)
        eg = got["engle_granger"]
        assert eg["p_value"] is None and len(cols) >= 7
        assert eg["cointegrated"] == (eg["statistic"] < eg["critical_values"]["5%"])
        want = decode(case["lib"])["engle_granger_parts"]["crit_2010"]
        for key, w in zip(("1%", "5%", "10%"), want):
            assert close(eg["critical_values"][key], w, 1e-12), key
        assert len(got["johansen"]["trace_statistics"]) == len(cols)
        return True
    if error and error["type"] == "LinAlgError":
        assert name in REFUSED, f"{name}: numpy refused, the engine is not held to refuse it"
        # numpy's LinAlgError has no wording to classify: the engine's typed refusal is this code
        check_refusal(
            lambda: stats.cointegration(cols, **kw),
            error,
            exact_message=False,
            code="degenerate_series",
        )
        return True
    if error:
        # a lone series is an argument error here; every other family counts it as too few
        code = "invalid_input" if re.search(r"\d+ series, this statistic needs", error["message"]) else None
        check_refusal(lambda: stats.cointegration(cols, **kw), error, exact_message=False, code=code)
        return True
    try:
        got = stats.cointegration(cols, **kw)
    except RaptorBTError as e:
        # the oracle answered from a moment matrix singular to working precision
        assert name in MAY_REFUSE | REFUSED and e.code == "degenerate_series", f"{name}: {e}"
        return True
    assert name not in REFUSED, f"{name}: the engine answered a panel that must be refused"
    tol = LOOSE.get(name, 1e-8)
    eg = got["engle_granger"]
    if name in EXACT_INTERCEPT:
        exact, bound = EXACT_INTERCEPT[name]
        assert abs(eg["intercept"] - exact) <= bound, f"intercept {eg['intercept']:e}"
    skip = {"cointegration.engle_granger.intercept"} if name in EXACT_INTERCEPT else set()
    # the spread's mean is rounding noise around 0: compared against the spread's own scale
    skip.add("cointegration.engle_granger.spread_mean")
    assert abs(eg["spread_mean"] - ok["engle_granger"]["spread_mean"]) <= 1e-8 * max(
        eg["spread_std"], 1.0
    )
    same(
        got,
        ok,
        "cointegration",
        tol,
        loose={
            "johansen.cointegrating_vector": max(tol, 1e-6),
            "engle_granger.half_life_periods": max(tol, 1e-7),
            "engle_granger.spread_last_z": max(tol, 1e-7),
            "johansen.trace_statistics": 1e-4 if name in LOOSE else tol,
        },
        skip=skip,
    )
    # statsmodels' own eigenvalues and max-eigenvalue statistics, as the answer now carries them
    sm = next(
        (
            v
            for v in decode(case["lib"])["johansen"]
            if v["det_order"] == 0 and v["k_ar_diff"] == 1 and "error" not in v
        ),
        None,
    )
    if sm is not None and name not in LOOSE:
        jo = got["johansen"]
        same(jo["eigenvalues"], list(sm["eig"]), f"{name}.eigenvalues", 1e-8, floor=0.0)
        same(jo["max_eigenvalue_statistics"], list(sm["lr2"]), f"{name}.lr2", 1e-8)
        assert jo["max_eigenvalue_critical_5pct"] == [r[1] for r in sm["cvm"]]
    if "known" in case:
        known = decode(case["known"])
        assert close(eg["intercept"], known["intercept"], 1e-9)
        assert close(next(iter(eg["hedge_ratios"].values())), known["hedge_ratio"], 1e-9)
    return True


# --------------------------------------------------------------------------------- the rest


def replay_hurst(case):
    (inc, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(lambda: stats.hurst(inc), error, exact_message=True)
        return True
    same(stats.hurst(inc), ok, "hurst", 1e-10, floor=0.0)
    return True


def replay_hac(case):
    (returns, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(lambda: stats.hac_tstat(returns, **kw), error, exact_message=True)
        return True
    got = stats.hac_tstat(returns, **kw)
    lib = decode(case["lib"])
    assert got["n"] == ok["n"] and got["lags"] == ok["lags"] == lib["lags_used"]
    # a mean of 1 with noise of 1e-9: the library's residuals carry the rounding error of its
    # mean, so the exact standard error is the judge there
    known = case.get("known")
    exact = _load_extract_helpers().hac_se_exact(case) if case["name"] == "large_mean_tiny_noise" else None
    tol = 1e-6 if exact is not None else 1e-10
    for key, want, t in (
        ("mean", lib["mean"], 1e-10),
        ("hac_std_error", lib["hac_bse"], tol),
        ("hac_t", lib["hac_t"], tol),
        ("hac_p_value", lib["hac_p_normal"], tol),
        ("naive_t", lib["naive_t"], tol),
        ("naive_p_value", lib["naive_p_t"], tol),
        ("per_period_sharpe", ok["per_period_sharpe"], tol),
        ("lag1_autocorrelation", ok["lag1_autocorrelation"], tol),
    ):
        assert close(got[key], want, t, floor=0.0), f"{key}: {got[key]!r}, want {want!r}"
    if exact is not None:
        assert close(got["hac_std_error"], decode(exact), 1e-12, floor=0.0)
    if known:
        known = decode(known)
        assert close(got["mean"], known["mean"], 1e-10, floor=0.0)
        assert close(got["hac_std_error"], known["hac_se"], 1e-10, floor=0.0)
        assert close(got["mean"] / got["naive_t"], known["naive_se"], 1e-10, floor=0.0)
    return True


def replay_garch(case):
    (returns, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error and "did not converge" in error["message"]:
        # the reference (arch) does not converge on returns of scale 1e-7; the engine rescales
        got = stats.garch(returns, **kw)
        assert got["rescale"] > 1.0 and got["converged"]
        return True
    if error:
        check_refusal(lambda: stats.garch(returns, **kw), error, exact_message=True)
        return True
    got = stats.garch(returns, **kw)
    # the shape of the answer and its labels are the reference's
    same(got, ok, "garch", math.inf, skip=_GARCH_NUMBERS)
    lib = decode(case["lib"]) if case.get("lib") else None
    if lib is None or lib["convergence_flag"] != 0:
        return True
    d_llf = got["log_likelihood"] - float(lib["loglikelihood"])
    assert d_llf >= -1e-3, f"log-likelihood {got['log_likelihood']} below arch's {lib['loglikelihood']}"
    if d_llf > 1e-3:
        return True  # another optimum at a higher likelihood: nothing else to compare
    p = got["parameters"]
    mine = [p["mu"] * 100.0, p["omega"] * 1e4, p["alpha"]]
    if p["gamma"] is not None:
        mine.append(p["gamma"])
    mine.append(p["beta"])
    if p["nu"] is not None:
        mine.append(p["nu"])
    names = got["parameter_names"]
    for i, (n, m, t) in enumerate(zip(names, mine, lib["params"])):
        if n == "nu":
            continue  # flat where the data are close to normal: held through the forecast
        err = abs(m - t) / max(abs(t), 1e-2) if n in ("mu", "omega") else abs(m - t)
        assert err <= 5e-3, f"{n}: {m} against {t}"
    fc = lib["forecast_variance_10"]
    for h, v in enumerate(got["forecast_vol_per_period"][: len(fc)]):
        assert abs(v * v * 1e4 - fc[h]) <= 1e-3 * abs(fc[h]), f"forecast variance at {h + 1}"
    tail = lib["conditional_volatility"][-len(got["recent_conditional_vol"]) :]
    for a, b in zip(got["recent_conditional_vol"], tail):
        assert abs(a * 100.0 - b) <= 1e-3 * abs(b), "conditional volatility"
    return True


def replay_predictive(case):
    (signal, prices), kw = args_of(case)
    got = stats.predictive_test(signal, prices, **kw)
    lib = decode(case["lib"])
    for h, want in lib.items():
        g = got["horizons"][h]
        assert g["n"] == want["n"] and g["n_independent"] == want["n_independent"], (h, g["n"])
        for ours, theirs in (
            ("slope", "slope"),
            ("intercept", "intercept"),
            ("slope_nonoverlap", "slope_nonoverlap"),
            ("t_nonoverlap", "t_nonoverlap"),
            ("p_nonoverlap", "p_nonoverlap"),
            ("t_hac", "t_hac"),
        ):
            assert close(g[ours], want[theirs], 1e-9, floor=1e-12), (
                f"{case['name']} h={h} {ours}: {g[ours]!r}, statsmodels {want[theirs]!r}"
            )
        if int(h) == 1:  # one observation per bar: the headline t is the OLS t
            assert close(g["t_nonoverlap"], want["t_ols"], 1e-9)
    return True


# the numbers of a GARCH fit are held to the library (above), the shape and labels to the reference
_GARCH_NUMBERS = {
    f"garch.{k}"
    for k in (
        "parameters",
        "stationary_variance",
        "persistence",
        "half_life_periods",
        "unconditional_vol_per_period",
        "log_likelihood",
        "aic",
        "last_vol_per_period",
        "forecast_vol_per_period",
        "forecast_cumulative_vol",
        "recent_conditional_vol",
        "last_vol_annualised",
        "forecast_vol_annualised",
    )
}


def replay_regimes(case):
    (returns, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(lambda: stats.markov_regimes(returns, **kw), error, exact_message=True)
        return True
    got = stats.markov_regimes(returns, **kw)
    lib = decode(case["lib"]) if case.get("lib") else None
    k = got["regimes"]
    assert k == ok["regimes"] and got["converged"] and got["n"] == ok["n"]
    assert got["smoothed_is_causal"] is False and got["filtered_is_causal"] is True
    stds = [s["vol_per_period"] for s in got["regime_stats"]]
    assert stds == sorted(stds), "regimes not ordered by variance"
    for row in got["recent_probabilities"] + got["recent_filtered_probabilities"]:
        assert abs(sum(row) - 1.0) < 1e-9
    assert np.allclose(got["recent_probabilities"][-1], got["recent_filtered_probabilities"][-1])
    if lib is None:
        return True
    d_llf = got["log_likelihood"] - float(lib["llf"])
    assert d_llf >= -1e-4, f"log-likelihood {got['log_likelihood']} below statsmodels' {lib['llf']}"
    if abs(d_llf) >= 1e-6:
        return True  # another optimum, at or above the reference's: nothing else to compare
    params = lib["params"]
    var_ref = params[k * (k - 1) + k :]
    order = sorted(range(k), key=lambda i: var_ref[i])
    mu_ref = [params[k * (k - 1) + i] for i in order]
    to_from = lib["regime_transition"]
    trans = [[to_from[t][f] for t in order] for f in order]
    scale = 100.0
    for i, s in enumerate(got["regime_stats"]):
        assert abs(s["mean_per_period"] * scale - mu_ref[i]) <= 1e-3 * max(abs(mu_ref[i]), 1.0)
        assert abs((s["vol_per_period"] * scale) ** 2 - var_ref[order[i]]) <= 1e-3 * max(
            var_ref[order[i]], 1.0
        )
    values = got["transition_matrix"]["values"]
    assert np.abs(np.array(values) - np.array(trans)).max() <= 1e-3
    rows = len(got["recent_probabilities"])
    smoothed = lib["smoothed_marginal_probabilities"][-rows:][:, order]
    filtered = lib["filtered_marginal_probabilities"][-rows:][:, order]
    assert np.abs(np.array(got["recent_probabilities"]) - smoothed).max() <= 1e-3
    assert np.abs(np.array(got["recent_filtered_probabilities"]) - filtered).max() <= 1e-3
    # the reference's answer, at full precision
    for g, w in zip(got["regime_stats"], ok["regime_stats"]):
        assert abs(g["share_of_periods"] - w["share_of_periods"]) <= 1e-3
    assert got["current_regime"] == ok["current_regime"]
    assert got["timeline_total"] == ok["timeline_total"]
    assert len(got["timeline"]) == len(ok["timeline"])
    for g, w in zip(got["timeline"], ok["timeline"]):
        assert (g["regime"], g["start_index"], g["end_index"]) == (
            w["regime"],
            w["start_index"],
            w["end_index"],
        )
    return True


def replay_pca(case):
    (cols, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(lambda: stats.pca_factors(cols, **kw), error, exact_message=False)
        return True
    got = stats.pca_factors(cols, **kw)
    lib = decode(case["lib"])
    tol = 1e-8
    assert got["n"] == ok["n"] and got["components"] == ok["components"]
    assert got["series"] == ok["series"]
    for g, w in zip(got["explained_variance_ratio"], ok["explained_variance_ratio"]):
        assert close(g, w, tol)
    assert close(got["cumulative_explained"], ok["cumulative_explained"], tol)
    signs = []
    for j in range(got["components"]):
        key = f"pc{j + 1}"
        mine = np.array([got["loadings"][key][n] for n in ok["series"]])
        want = np.array([ok["loadings"][key][n] for n in ok["series"]])
        # when the raw sum is near 0 the sign is a convention, not something the data decide:
        # such a component is compared up to sign
        sign = 1.0
        if abs(lib["row_sums_raw"][j]) <= 1e-6 and float(mine @ want) < 0:
            sign = -1.0
        for g, w in zip(mine, want):
            assert close(sign * g, w, tol), f"loading {key}"
        signs.append(sign)
    if all(s == 1.0 for s in signs):
        assert got["first_component_is_common_factor"] == ok["first_component_is_common_factor"]
    assert ("target" in got) == ("target" in ok)
    if "target" in ok:
        t, w = got["target"], ok["target"]
        exact = (
            decode(_load_extract_helpers().pca_rank_aware_t(case))
            if case["name"] == "two_identical_with_target"
            else None
        )
        assert close(t["r_squared"], w["r_squared"], tol)
        assert close(t["alpha_per_period"], w["alpha_per_period"], tol)
        if exact is None:
            assert close(t["alpha_t"], w["alpha_t"], tol)
        else:
            # a rank-deficient panel: ordinary least squares on the components with variance,
            # recomputed from the inputs, is the judge
            assert close(abs(t["alpha_t"]), exact["alpha_t_abs"][0], tol)
        for j in range(got["components"]):
            key = f"pc{j + 1}"
            e, lw = t["exposures_per_unit_score"][key], w["exposures_per_unit_score"][key]
            assert close(signs[j] * e["beta"], lw["beta"], tol), f"beta {key}"
            if exact is None:
                assert close(signs[j] * e["t"], lw["t"], tol), f"t {key}"
            else:
                assert close(abs(e["t"]), exact["t_abs"][j], tol), f"t {key}"
    if "known" in case:
        known = decode(case["known"])
        assert close(got["explained_variance_ratio"][0], known["first_ratio"], tol)
        assert close(got["explained_variance_ratio"][1], known["second_ratio"], tol)
        assert close(got["loadings"]["pc1"][ok["series"][0]], known["loading"], tol)
    return True


def spa_columns(cols, kw):
    names = list(cols)
    bench = kw.get("benchmark")
    diff = [cols[n] - (0.0 if bench is None else bench) for n in names]
    return names, diff


def within_monte_carlo(got, want, reps):
    """Two estimates of one probability: the difference has variance 2 p (1 - p) / reps; one
    repetition of slack covers p at 0 or 1."""
    return abs(got - want) <= 4.0 * math.sqrt(2.0 * want * (1.0 - want) / reps) + 1.0 / reps


def replay_spa(case):
    (cols, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(lambda: stats.spa_test(cols, **kw), error, exact_message=False)
        return True
    got = stats.spa_test(cols, **kw)
    names, diff = spa_columns(cols, kw)
    if case.get("lib"):
        lib = decode(case["lib"])
        # arch's own bootstrap indices, through the native replay
        rep = _raptorbt.stats_spa_replay(diff, int(lib["block"]), lib["indices"])
        for j, (m, v) in enumerate(zip(lib["loss_diff_mean"], lib["loss_diff_var"])):
            assert close(rep["loss_diff_mean"][j], m, 1e-10, floor=1e-3), f"mean[{j}]"
            assert close(rep["loss_diff_var"][j], v, 1e-10, floor=0.0), f"var[{j}]"
        assert [int(v) for v in rep["valid"]] == [int(v) for v in lib["valid_columns"]]
        sim = lib["simulated_vals"].max(axis=0)  # repetitions by the three centrings
        assert np.abs(np.array(rep["max_simulated"]) - sim).max() <= 1e-10 * max(
            1e-3, np.abs(sim).max()
        )
        for key, p in (("lower", "p_lower"), ("consistent", "p_consistent"), ("upper", "p_upper")):
            assert rep["pvalues"][key] == lib[p], key
        # the engine's own seeded draws
        assert got["bootstrap"]["mean_block"] == int(lib["block"])
        reps = kw.get("reps", 1000)
        for key, p in (
            ("p_value_spa_lower", "p_lower"),
            ("p_value_spa_consistent", "p_consistent"),
            ("p_value_white_reality_check", "p_upper"),
        ):
            assert within_monte_carlo(got[key], lib[p], reps), f"{key}: {got[key]}, arch {lib[p]}"
    same(
        got,
        ok,
        "spa",
        1e-10,
        floor=1e-3,
        skip={f"spa.{k}" for k in ("p_value_spa_lower", "p_value_spa_consistent")}
        | {"spa.p_value_white_reality_check"},
    )
    assert got["studentised"] is False
    return True


def replay_pbo(case):
    (cols, *_), kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(lambda: stats.pbo_cscv(cols, **kw), error, exact_message=False)
        return True
    got = stats.pbo_cscv(cols, **kw)
    same(got, ok, "pbo", 1e-12, floor=0.0)
    if "known" in case:
        known = decode(case["known"])
        assert close(got["pbo"], known["pbo"], 1e-12, floor=0.0)
        assert close(got["median_logit"], known["median_logit"], 1e-12, floor=0.0)
        if "prob_loss" in known:
            assert close(got["prob_out_of_sample_loss_of_best"], known["prob_loss"], 1e-12, 0.0)
    return True


def replay_events(case):
    (dates, asset, market, events), kw = args_of(case)
    ok, error = reference_of(case)
    day = date.fromisoformat

    def call():
        return stats.event_study([day(d) for d in dates], asset, market, [day(e) for e in events], **kw)

    if error:
        check_refusal(call, error, exact_message=True)
        return True
    got = call()
    # no residual noise: the t is undefined (NaN in the reference and the engine)
    noise_free = case["name"] == "known_noise_free"
    same(got, ok, "events", 1e-10, floor=1e-3)
    for row in got["events"]:
        assert row["degenerate_fit"] == noise_free and (not noise_free or math.isnan(row["t"]))
    dates_index = [day(d) for d in dates]
    for row, lib in zip(got["events"], [e for e in decode(case["lib"])["events"] if not e["skipped"]]):
        assert row["event_bar"] == dates_index[int(lib["idx"])].isoformat()
        assert close(row["alpha"], lib["alpha_ols"], 1e-10, floor=1e-3), "alpha against OLS"
        assert close(row["beta"], lib["beta_ols"], 1e-10, floor=0.0), "beta against OLS"
    assert ("caar_note" in got) == (got["caar_t"] is None)
    return True


def replay_impact(case):
    name = case["name"]
    if name.startswith("known_") or name.startswith("quantity_for_notional"):
        return False  # direct calls of the reference's helpers, and a tool-layer function
    (high, low, close_, volume), kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(
            lambda: stats.impact_estimate(high, low, close_, volume, **kw), error, exact_message=True
        )
        return True
    got = stats.impact_estimate(high, low, close_, volume, **kw)
    same(got, ok, "impact", 1e-12, floor=0.0)
    return True


def replay_surface(case):
    (names, cells), kw = args_of(case)
    ok, error = reference_of(case)
    cells = [(vals, metric) for vals, metric in cells]
    if error:
        check_refusal(
            lambda: stats.sensitivity_surface(names, cells), error, exact_message=False
        )
        return True
    got = stats.sensitivity_surface(names, cells)
    same(got, ok, "surface", 1e-12)
    if "known" in case:
        known = decode(case["known"])
        assert close(got["best"]["metric"], known["best_metric"], 1e-12)
        assert close(got["best"]["neighbour_mean"], known["neighbour_mean_of_best"], 1e-12)
        assert close(got["best"]["plateau_share"], known["plateau_share"], 1e-12)
        assert close(got["smoothness"], known["smoothness"], 1e-12)
        assert got["verdict"] == known["verdict"]
    return True


def replay_quality(case):
    args, kw = args_of(case)
    ok, error = reference_of(case)
    if error:
        check_refusal(lambda: stats.audit_bars(*args, **kw), error, exact_message=False)
        return True
    got = stats.audit_bars(*args, **kw)
    f = got["findings"]
    # exact, but for an `exp` of a difference of logs, which two libms may round one ulp apart
    same(got, ok, "audit", 0.0, loose={"return_pct": 1e-12, "close_ratio": 1e-12})
    assert got["clean"] == ok["clean"]
    if case["name"] == "all_flat":  # flat bars with no volume are not a clean series
        assert not got["clean"] and f["flat_zero_volume_bars"] == got["bars"]
    if "known" in case:
        known = decode(case["known"])
        for key in ("duplicate_timestamps", "flat_bars", "out_of_order", "zero_volume_bars"):
            assert f[key] == known[key], key
        assert f["impossible_ohlc"]["count"] == known["impossible_ohlc"]
    return True


# (family, replay, the fewest cases it must replay)
FAMILIES = {
    "stationarity": (replay_stationarity, 9),
    "coint": (replay_cointegration, 20),
    "hurst": (replay_hurst, 10),
    "garch": (replay_garch, 8),
    "hmm": (replay_regimes, 5),
    "hac": (replay_hac, 9),
    "pca": (replay_pca, 9),
    "spa": (replay_spa, 10),
    "events": (replay_events, 10),
    "pbo": (replay_pbo, 10),
    "impact": (replay_impact, 9),
    "surface": (replay_surface, 10),
    "quality": (replay_quality, 12),
    "predictive": (replay_predictive, 5),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_fixture_replays_through_raptorbt_stats(family):
    replay, floor = FAMILIES[family]
    run_family(family, replay, floor)


def test_every_fixture_file_has_a_replay():
    on_disk = {p.name.removeprefix("stats_").removesuffix(".json.gz") for p in FIXTURES.glob("*.gz")}
    assert on_disk == set(FAMILIES), sorted(on_disk ^ set(FAMILIES))


def test_the_lists_read_out_of_the_rust_tests_are_not_empty():
    # a regex that stopped matching must not turn every exception into "none"
    assert ORACLE_INEXACT and MAY_REFUSE and REFUSED and EXACT_INTERCEPT
    assert set(REFUSED) >= {"collinear_exact", "duplicate_regressors"}
    assert LOOSE == {}
    cases = {c["name"] for c in fixture("coint")["cases"]}
    assert (MAY_REFUSE | REFUSED | set(EXACT_INTERCEPT)) <= cases
    assert set(ORACLE_INEXACT) <= {c["name"] for c in fixture("stationarity")["cases"]}
