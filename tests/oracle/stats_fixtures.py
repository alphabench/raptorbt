"""Statistics oracle fixtures: the reference numbers for the statistics port.

Called by `generate.py` (`--families stats`); it is not run on its own. One fixture file per
family under tests/python/oracle/fixtures/stats/ (a subdirectory: the TA-Lib replay tests scan the
parent for *.json.gz and check TA-Lib versions in every header):

    stats_special.json          special functions, from scipy (plain JSON: Rust unit tests read it)
    stats_stationarity.json.gz  ADF, KPSS, the MacKinnon p-value and critical-value grids
    stats_coint.json.gz         Engle-Granger, Johansen, half-life
    stats_hurst.json.gz         rescaled range and DFA, Anis-Lloyd expectation
    stats_garch.json.gz         GARCH(1,1) and GJR by arch
    stats_hmm.json.gz           Markov-switching regression by statsmodels
    stats_hac.json.gz           Newey-West t-statistic
    stats_pca.json.gz           principal components and the factor regression
    stats_spa.json.gz           superior predictive ability, with arch's bootstrap indices
    stats_events.json.gz        event study
    stats_pbo.json.gz           probability of backtest overfitting
    stats_impact.json.gz        impact and spread estimates
    stats_surface.json.gz       parameter-sensitivity surface
    stats_quality.json.gz       bar-data audit
    stats_predictive.json.gz    one signal against the returns that follow it (statsmodels OLS)

Each case holds
    input           the arguments (`args`, `kwargs`) the reference function was called with,
    reference_full  what the reference function (`stats_reference.py`) returns, or the error it
                    raises: what the engine must return,
    reference       the same answer as plain JSON (decimal floats, NaN and infinity as null), to
                    read by eye,
    lib             numbers read from the underlying library (statsmodels, arch, scipy) at full
                    precision where the reference's answer does not carry them,
    known           an answer computed independently of every library (exact arithmetic or a closed
                    form), where one exists.

`stats_reference.py` says where each family's reference comes from: the public library's output
where one computes the statistic, otherwise an implementation written there from the published
definition. Floats in `input`, `reference_full`, `lib` and `known` are the hex of their IEEE-754
big-endian bits ({"f8": hex} for a scalar, plus "shape" for an array, "i4" for integers);
`reference` is plain JSON.
"""

import datetime as dt
import json
import math
import os
import sys
import warnings
from fractions import Fraction

import numpy as np

import stats_reference as R

FAMILY_FILES = {
    "stationarity": "stats_stationarity.json.gz",
    "coint": "stats_coint.json.gz",
    "hurst": "stats_hurst.json.gz",
    "garch": "stats_garch.json.gz",
    "hmm": "stats_hmm.json.gz",
    "hac": "stats_hac.json.gz",
    "pca": "stats_pca.json.gz",
    "spa": "stats_spa.json.gz",
    "events": "stats_events.json.gz",
    "pbo": "stats_pbo.json.gz",
    "impact": "stats_impact.json.gz",
    "surface": "stats_surface.json.gz",
    "quality": "stats_quality.json.gz",
    "predictive": "stats_predictive.json.gz",
}
SPECIAL_FILE = "stats_special.json"


# -------------------------------------------------------------------------------- encoding


def enc_input(g, o):
    """`enc`, except that a dict of arrays (a `Columns`) keeps its order, which the JSON
    writer would sort away and which matters (the first series of a cointegration is the
    dependent one): {"order": [names], "columns": {name: array}}."""
    if isinstance(o, dict) and o and all(isinstance(v, np.ndarray) for v in o.values()):
        return {"order": [str(k) for k in o], "columns": {str(k): enc(g, v) for k, v in o.items()}}
    if isinstance(o, (list, tuple)):
        return [enc_input(g, v) for v in o]
    if isinstance(o, dict):
        return {str(k): enc_input(g, v) for k, v in o.items()}
    return enc(g, o)


def enc(g, o):
    """Hex-encode floats, arrays and dates; leave ints, strings, bools and None."""
    if isinstance(o, np.ndarray):
        if o.dtype.kind == "f":
            return {"f8": g.encode_f8(o.reshape(-1)), "shape": list(o.shape)}
        if o.dtype.kind in "iub":
            return {"i4": g.encode_i4(o.reshape(-1).astype(np.int64)), "shape": list(o.shape)}
        raise TypeError(o.dtype)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return {"f8": g.encode_f8([float(o)])}
    if isinstance(o, (int, np.integer)):
        return int(o)
    if o is None or isinstance(o, str):
        return o
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): enc(g, v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        if o and all(isinstance(v, (float, np.floating)) and not isinstance(v, bool) for v in o):
            return {"f8": g.encode_f8(np.array(o, dtype=float)), "shape": [len(o)]}
        return [enc(g, v) for v in o]
    raise TypeError(type(o))


def plain(o):
    """JSON-able, floats as decimal (NaN and infinity as None)."""
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    return o


# ------------------------------------------------------------------------------ the context


class Ctx:
    def __init__(self, g):
        self.g = g

    # inputs come from the SplitMix64 of generate.py, not from numpy's generators
    def normals(self, seed, n):
        rng = self.g.SplitMix64(seed)
        return np.array([rng.normal() for _ in range(n)])

    def uniforms(self, seed, n):
        rng = self.g.SplitMix64(seed)
        return np.array([rng.uniform() for _ in range(n)])

    @staticmethod
    def run(fn, args, kwargs):
        """The reference's answer, {"ok": ...}, or the error it raised (a refusal carries its
        error code)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return {"ok": fn(*args, **kwargs)}
            except R.Refusal as e:
                return {"error": {"type": "Refusal", "code": e.code, "message": e.message}}
            except Exception as e:  # a library that cannot answer: the type is the evidence
                return {"error": {"type": type(e).__name__, "message": str(e)}}

    def case(self, name, kind, fn, args, kwargs=None, *, note=None, lib=None, known=None):
        kwargs = kwargs or {}
        answer = self.run(fn, args, kwargs)
        rec = {
            "name": name,
            "kind": kind,
            "input": {"args": enc_input(self.g, list(args)), "kwargs": enc_input(self.g, kwargs)},
            "reference": plain(answer),
            "reference_full": enc(self.g, answer),
        }
        if note:
            rec["note"] = note
        if lib is not None:
            rec["lib"] = enc(self.g, lib)
        if known is not None:
            rec["known"] = enc(self.g, known)
        return rec


def lib_call(fn):
    """A library call that may raise: its value, or {"error": type}."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn()
        except Exception as e:
            return {"error": {"type": type(e).__name__, "message": str(e)}}


# --------------------------------------------------------------------------- input series


def ar1(ctx, seed, n, phi, sigma=1.0, x0=0.0):
    z = ctx.normals(seed, n) * sigma
    x = np.empty(n)
    prev = x0
    for i in range(n):
        prev = phi * prev + z[i]
        x[i] = prev
    return x


def random_walk(ctx, seed, n, sigma=1.0, drift=0.0, x0=0.0):
    return x0 + np.cumsum(ctx.normals(seed, n) * sigma + drift)


def garch_sim(ctx, seed, n, omega, alpha, beta, gamma=0.0, burn=200):
    z = ctx.normals(seed, n + burn)
    h = omega / max(1.0 - alpha - beta - gamma / 2.0, 1e-3)
    r = np.empty(n + burn)
    for i in range(n + burn):
        r[i] = math.sqrt(h) * z[i]
        h = omega + alpha * r[i] ** 2 + gamma * (r[i] < 0) * r[i] ** 2 + beta * h
    return r[burn:]


def business_days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def header(ctx, family, extra=None):
    import arch
    import pandas
    import scipy
    import statsmodels

    h = {
        "generator": "tests/oracle/generate.py (stats_fixtures.py)",
        "kind": f"stats_{family}",
        "libraries": {
            "statsmodels": statsmodels.__version__,
            "arch": arch.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "python": sys.version.split()[0],
        },
        "encoding": "f8/i4 big-endian bits, hex, concatenated; {'f8': hex} scalar, 'shape' for arrays",
        "inputs": "SplitMix64 (generate.py), seeds in the case names' builders",
        "reference": "tests/oracle/stats_reference.py",
    }
    if extra:
        h.update(extra)
    return h


# --------------------------------------------------------------------------------- special


def build_special(ctx):
    from scipy import special as sp
    from scipy import stats as st

    g = ctx.g

    def hx(v):
        return g.encode_f8([float(v)])

    cases = {"ln_gamma": [], "inc_beta": [], "inc_gamma": [], "t": [], "chi2": [], "f": []}
    # ln G is accurate to ~1e-14 absolute, so a 1e-12 relative check needs |ln G| >= 0.01: the
    # neighbourhoods of the zeros at 1 and 2 are left out, the zeros themselves are exact
    for x in (
        1e-3,
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.5,
        2.5,
        3,
        3.7,
        5,
        7.5,
        10,
        14.9,
        15,
        20,
        50,
        100,
        171.5,
        1000.5,
        1e4,
        1e5,
        1e6,
        1e10,
    ):
        cases["ln_gamma"].append({"x": hx(x), "y": hx(sp.gammaln(x))})
    for a in (0.5, 1, 2, 5.5, 20, 100):
        for b in (0.5, 1, 3, 25, 150):
            for x in (1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
                p, q = sp.betainc(a, b, x), sp.betaincc(a, b, x)
                if min(p, q) < 1e-250 or max(p, q) == 0:
                    continue
                cases["inc_beta"].append(
                    {"a": hx(a), "b": hx(b), "x": hx(x), "p": hx(p), "q": hx(q)}
                )
    for a in (0.5, 1, 2.5, 10, 50, 200, 1000):
        for f in (0.01, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0):
            x = a * f
            p, q = sp.gammainc(a, x), sp.gammaincc(a, x)
            if min(p, q) < 1e-250:
                continue
            cases["inc_gamma"].append({"a": hx(a), "x": hx(x), "p": hx(p), "q": hx(q)})
    for df in (1, 2, 3, 5, 10, 30, 100, 1000):
        for x in (-10, -3, -1, -0.1, 0, 0.1, 1, 2.5, 5, 10, 40):
            c, s = st.t.cdf(x, df), st.t.sf(x, df)
            if min(c, s) < 1e-250:
                continue
            cases["t"].append({"x": hx(x), "df": hx(df), "cdf": hx(c), "sf": hx(s)})
    for df in (1, 2, 3, 10, 50, 200, 1000):
        for f in (0.01, 0.1, 0.5, 1, 2, 5):
            x = df * f
            c, s = st.chi2.cdf(x, df), st.chi2.sf(x, df)
            if min(c, s) < 1e-250:
                continue
            cases["chi2"].append({"x": hx(x), "df": hx(df), "cdf": hx(c), "sf": hx(s)})
    for d1, d2 in ((1, 1), (2, 10), (5, 5), (10, 30), (30, 100), (100, 1000)):
        for x in (0.01, 0.1, 0.5, 1, 2, 5, 20, 100):
            c, s = st.f.cdf(x, d1, d2), st.f.sf(x, d1, d2)
            if min(c, s) < 1e-250:
                continue
            cases["f"].append({"x": hx(x), "d1": hx(d1), "d2": hx(d2), "cdf": hx(c), "sf": hx(s)})
    h = header(
        ctx, "special", {"note": "relative 1e-12 is the bar; hex bits of the float64 reference"}
    )
    return {"header": h, "cases": cases}


# ---------------------------------------------------------------------------- stationarity


def build_stationarity(ctx):
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    from statsmodels.tsa.stattools import adfuller, kpss

    g = ctx.g

    def lib(x):
        out = {}
        a = lib_call(
            lambda: adfuller(x, regression="c", autolag="AIC", result_object=True, regresults=True)
        )
        if isinstance(a, dict) and "error" in a:
            out["adf"] = a
        else:
            startlag = 1 + 1  # the level and the constant precede the lagged differences
            ics = {int(k) - startlag: float(v.aic) for k, v in a.resstore.autolag_results.items()}
            out["adf"] = {
                "statistic": float(a.statistic),
                "p_value": float(a.pvalue),
                "lags": int(a.lags),
                "nobs": int(a.nobs),
                "critical_values": {k: float(v) for k, v in a.critical_values.items()},
                "icbest": float(a.icbest),
                "aic_by_lag": {str(k): v for k, v in sorted(ics.items())},
                "maxlag": int(a.resstore.maxlag),
            }
        k = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                k = kpss(x, regression="c", nlags="auto", result_object=True)
            except Exception as e:
                k = {"error": {"type": type(e).__name__, "message": str(e)}}
        if isinstance(k, dict):
            out["kpss"] = k
        else:
            from statsmodels.tools.sm_exceptions import InterpolationWarning

            out["kpss"] = {
                "statistic": float(k.statistic),
                "p_value": float(k.pvalue),
                "lags": int(k.lags),
                "critical_values": {a_: float(v) for a_, v in k.critical_values.items()},
                "clipped": any(issubclass(w.category, InterpolationWarning) for w in caught),
            }
        return out

    cases = []
    n = 500
    series = [
        ("stationary_ar1", "normal", ar1(ctx, 11, n, 0.5)),
        ("random_walk", "normal", random_walk(ctx, 12, n)),
        ("at_min_obs_stationary", "min_obs", ar1(ctx, 13, 50, 0.5)),
        ("at_min_obs_walk", "min_obs", random_walk(ctx, 14, 50)),
        ("below_min_obs", "min_obs", ar1(ctx, 15, 49, 0.5)),
        ("near_unit_root", "degenerate", ar1(ctx, 16, n, 0.999)),
        ("trend_stationary", "degenerate", 0.05 * np.arange(n) + ar1(ctx, 17, n, 0.3)),
        ("constant", "degenerate", np.full(60, 100.0)),
        ("near_constant", "degenerate", 100.0 + 1e-9 * ctx.normals(18, 120)),
        ("tiny_scale", "degenerate", 1e-8 * ar1(ctx, 19, n, 0.5)),
        ("large_level", "degenerate", 1e9 + ar1(ctx, 20, n, 0.5)),
        ("long_5000", "normal", ar1(ctx, 21, 5000, 0.8)),
        ("white_noise", "known", ctx.normals(22, n)),
        (
            "step_break",
            "degenerate",
            np.concatenate([ctx.normals(23, 250), 5.0 + ctx.normals(24, 250)]),
        ),
    ]
    for name, kind, x in series:
        note = None
        if name == "white_noise":
            note = "iid noise: the unit root must be rejected (p ~ 0) and stationarity kept"
        cases.append(
            ctx.case(
                name, kind, R.stationarity, [x], lib=lib(x) if len(x) >= 20 else None, note=note
            )
        )
    # fixed lag choices, to pin the regression without the lag search
    x = ar1(ctx, 25, 300, 0.7)
    for nl in ("legacy", 4):
        rec = lib_call(lambda nl=nl: kpss(x, regression="c", nlags=nl, result_object=True))
        cases.append(
            {
                "name": f"kpss_nlags_{nl}",
                "kind": "library",
                "input": {"args": enc(g, [x]), "kwargs": {"nlags": nl, "regression": "c"}},
                "lib": enc(
                    g,
                    {
                        "statistic": float(rec.statistic),
                        "p_value": float(rec.pvalue),
                        "lags": int(rec.lags),
                    },
                ),
            }
        )
    for reg in ("c", "ct", "ctt", "n"):
        for ml in (0, 3, 12):
            a = lib_call(
                lambda reg=reg, ml=ml: adfuller(
                    x, maxlag=ml, regression=reg, autolag=None, result_object=True
                )
            )
            cases.append(
                {
                    "name": f"adf_fixed_lag_{reg}_{ml}",
                    "kind": "library",
                    "input": {
                        "args": enc(g, [x]),
                        "kwargs": {"regression": reg, "maxlag": ml, "autolag": None},
                    },
                    "lib": enc(
                        g,
                        {
                            "statistic": float(a.statistic),
                            "p_value": float(a.pvalue),
                            "lags": int(a.lags),
                            "nobs": int(a.nobs),
                            "critical_values": {k: float(v) for k, v in a.critical_values.items()},
                        },
                    ),
                }
            )
    # MacKinnon grids: the p-value of a statistic and the 2010 critical values
    stats = np.array(
        [-12.0, -8.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    )
    grid = []
    for reg in ("n", "c", "ct", "ctt"):
        for N in range(1, 7):
            grid.append(
                {
                    "regression": reg,
                    "N": N,
                    "stat": stats,
                    "p": np.array([mackinnonp(s, regression=reg, N=N) for s in stats]),
                }
            )
    p_error = {}
    for N in (7, 12):
        r = lib_call(lambda N=N: mackinnonp(-3.0, regression="c", N=N))
        p_error[str(N)] = r["error"]["type"] if isinstance(r, dict) else "no error"
    crit = []
    nobs_list = [20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, float("inf")]
    for reg in ("c", "ct", "ctt"):
        for N in range(1, 13):
            crit.append(
                {
                    "regression": reg,
                    "N": N,
                    "nobs": np.array(nobs_list),
                    "crit": np.array(
                        [mackinnoncrit(N=N, regression=reg, nobs=v) for v in nobs_list]
                    ),
                }
            )
    crit.append(
        {
            "regression": "n",
            "N": 1,
            "nobs": np.array(nobs_list),
            "crit": np.array([mackinnoncrit(N=1, regression="n", nobs=v) for v in nobs_list]),
        }
    )
    crit_n2 = lib_call(lambda: mackinnoncrit(N=2, regression="n", nobs=100))
    # KPSS p-value interpolation grid
    kstat = np.array([0.0, 0.05, 0.119, 0.13, 0.2, 0.347, 0.4, 0.463, 0.5, 0.574, 0.65, 0.739, 1.0])
    kp = []
    for reg in ("c", "ct"):
        crit_k = lib_call(
            lambda reg=reg: kpss(ctx.normals(3, 60), regression=reg, nlags=2, result_object=True)
        ).critical_values
        keys = ["10%", "5%", "2.5%", "1%"]
        pv = np.interp(kstat, [crit_k[k] for k in keys], [0.10, 0.05, 0.025, 0.01])
        kp.append(
            {
                "regression": reg,
                "stat": kstat,
                "p": pv,
                "critical": np.array([crit_k[k] for k in keys]),
            }
        )
    tables = {
        "mackinnonp_1994": grid,
        "mackinnonp_index_error": p_error,
        "mackinnoncrit_2010": crit,
        "mackinnoncrit_n_N2": crit_n2["error"]["type"] if isinstance(crit_n2, dict) else "no error",
        "kpss_p_interpolation": kp,
    }
    return {"header": header(ctx, "stationarity"), "cases": cases, "tables": enc(g, tables)}


# -------------------------------------------------------------------------------------- coint


def build_coint(ctx):
    import statsmodels.api as sm
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen


    def lib(m, variants=((0, 1),)):
        y, x = m[:, 0], m[:, 1:]
        n = len(m)
        out = {}
        ols = sm.OLS(y, sm.add_constant(x)).fit()
        out["ols"] = {
            "params": np.asarray(ols.params),
            "rsquared": float(ols.rsquared),
            "resid": np.asarray(ols.resid),
        }
        k_vars = m.shape[1]
        eg = lib_call(lambda: tuple(coint(y, x, trend="c", autolag="aic")))
        if isinstance(eg, dict):
            out["engle_granger"] = eg
        else:
            out["engle_granger"] = {
                "statistic": float(eg[0]),
                "p_value": float(eg[1]),
                "critical_values": np.asarray(eg[2]),
            }
        # the pieces of that call, for the port to check one by one
        parts = {"crit_2010": np.asarray(mackinnoncrit(N=k_vars, regression="c", nobs=n - 1))}
        res = np.asarray(ols.resid)
        if ols.rsquared < 1 - 100 * np.sqrt(np.finfo(float).eps):
            a = adfuller(res, regression="n", autolag="aic", result_object=True)
            parts["adf_on_residual"] = {
                "statistic": float(a.statistic),
                "lags": int(a.lags),
                "nobs": int(a.nobs),
                "icbest": float(a.icbest),
            }
            p = lib_call(lambda: mackinnonp(a.statistic, regression="c", N=k_vars))
            parts["p_value"] = p if isinstance(p, float) else p
        else:
            parts["collinear_branch"] = True
        out["engle_granger_parts"] = parts
        out["johansen"] = []
        for det_order, k_ar_diff in variants:
            jo = lib_call(
                lambda d=det_order, k=k_ar_diff: coint_johansen(m, det_order=d, k_ar_diff=k)
            )
            if isinstance(jo, dict):
                out["johansen"].append({"det_order": det_order, "k_ar_diff": k_ar_diff, **jo})
                continue
            evec = np.asarray(jo.evec)
            out["johansen"].append(
                {
                    "det_order": det_order,
                    "k_ar_diff": k_ar_diff,
                    "eig": np.real(np.asarray(jo.eig)),
                    "eig_imag_max": float(np.abs(np.imag(np.asarray(jo.eig))).max()),
                    "evec": np.real(evec),
                    "evec_imag_max": float(np.abs(np.imag(evec)).max()),
                    "lr1": np.asarray(jo.lr1),
                    "lr2": np.asarray(jo.lr2),
                    "cvt": np.asarray(jo.cvt),
                    "cvm": np.asarray(jo.cvm),
                    "rkt": np.asarray(jo.rkt),
                    "r0t": np.asarray(jo.r0t),
                }
            )
        return out

    def cols(*arrays):
        return {f"s{i}": a for i, a in enumerate(arrays)}

    def matrix(c):
        return np.column_stack(list(c.values()))

    cases = []
    n = 300
    x1 = random_walk(ctx, 31, n, 1.0, 0.0, 100.0)
    x2 = random_walk(ctx, 32, n, 1.0, 0.0, 50.0)
    x3 = random_walk(ctx, 33, n, 1.0, 0.0, 70.0)
    noise = ar1(ctx, 34, n, 0.5, 0.5)

    def add(name, kind, c, variants=((0, 1),), **kw):
        m = matrix(c)
        cases.append(
            ctx.case(
                name,
                kind,
                R.cointegration,
                [c],
                lib=lib(m, variants) if 20 <= len(m) and m.shape[1] <= 12 else None,
                **kw,
            )
        )

    add(
        "cointegrated_pair",
        "normal",
        cols(0.5 * x1 + 3.0 + noise, x1),
        variants=((0, 1), (-1, 1), (1, 1), (0, 2)),
    )
    add("independent_walks", "normal", cols(x1, x2), variants=((0, 1), (-1, 1), (1, 1)))
    add(
        "three_series",
        "normal",
        cols(0.5 * x1 + 0.3 * x2 + noise, x1, x2),
        variants=((0, 1), (1, 1), (0, 3)),
    )
    add("three_series_two_relations", "normal", cols(x1 + noise, x1 + 0.5 * x3, x3))
    add("at_min_obs", "min_obs", cols((0.5 * x1 + noise)[:60], x1[:60]))
    add("below_min_obs", "min_obs", cols((0.5 * x1 + noise)[:59], x1[:59]))
    add("single_series", "min_obs", cols(x1))
    add(
        "collinear_exact",
        "degenerate",
        cols(2.0 * x1 + 3.0, x1),
        note="R2 = 1: Engle-Granger takes the collinear branch (-inf, p = 0)",
    )
    add(
        "collinear_near",
        "degenerate",
        cols(2.0 * x1 + 3.0 + 1e-9 * ctx.normals(35, n), x1),
        note="R2 above 1 - 100 sqrt(eps)",
    )
    add(
        "duplicate_regressors",
        "degenerate",
        cols(x1 + noise, x2, x2.copy()),
        note="a regressor repeated: OLS and Johansen see a singular system",
    )
    add("constant_series", "degenerate", cols(x1, np.full(n, 5.0)))
    add(
        "stationary_inputs",
        "degenerate",
        cols(ar1(ctx, 36, n, 0.3), ar1(ctx, 37, n, 0.3)),
        note="levels that are already stationary",
    )
    add(
        "unit_scale_extremes",
        "degenerate",
        cols(1e-6 * (0.5 * x1 + noise), 1e6 * x1),
        note="columns eleven orders of magnitude apart",
    )
    for k in (4, 6, 7, 12):
        c = cols(*[x1 + noise * i + 0.1 * random_walk(ctx, 40 + i, n) for i in range(k)])
        add(
            f"{k}_series",
            "degenerate" if k >= 7 else "normal",
            c,
            note="mackinnonp has no 1994 row for N >= 7: statsmodels raises IndexError"
            if k >= 7
            else None,
        )
    # panels of rank 2 with noise at decreasing scale: Johansen's matrices turn singular and
    # statsmodels lets numpy's LinAlgError out
    b1, b2 = random_walk(ctx, 31, 200), random_walk(ctx, 32, 200)
    for eps, k in ((1e-3, 5), (1e-5, 5), (1e-7, 3), (1e-9, 3), (1e-9, 4)):
        panel = cols(
            *[
                (1 + 0.3 * j) * b1 + (0.5 - 0.1 * j) * b2 + eps * ctx.normals(100 + j, 200)
                for j in range(k)
            ]
        )
        add(
            f"rank2_panel_{k}_series_noise_{eps:g}",
            "degenerate",
            panel,
            note="three or more series driven by two walks: the system has a null direction",
        )
    # more well-posed panels of the kinds above (the port refuses moment matrices whose condition
    # number is beyond what 1e-8 can be quoted at, which leaves fewer ill-posed fits to replay)
    w1, w2, w3, w4 = (random_walk(ctx, 200 + i, 150, 1.0, 0.0, 80.0) for i in range(4))
    wn = ar1(ctx, 204, 150, 0.5, 0.5)
    add(
        "cointegrated_pair_b",
        "normal",
        cols(0.8 * w1 - 2.0 + wn, w1),
        variants=((0, 1), (-1, 1), (1, 1), (0, 2)),
    )
    add(
        "three_series_b",
        "normal",
        cols(0.4 * w1 + 0.6 * w2 + wn, w1, w2),
        variants=((0, 1), (-1, 1), (1, 1), (0, 3)),
    )
    add("four_series_b", "normal", cols(w1 + wn, w2 + 0.3 * w3, w3, w4), variants=((0, 1), (1, 2)))
    # 13 series: refused before any library is called
    add("13_series", "min_obs", cols(*[random_walk(ctx, 60 + i, n) for i in range(13)]))
    # known answers
    # (a) half-life of an exact AR(1) decay: s_t = 0.9 s_{t-1}, so ln 2 / -ln 0.9
    spread = 0.9 ** np.arange(80, dtype=float)
    cases.append(
        ctx.case(
            "half_life_exact_decay",
            "known",
            R.half_life,
            [spread],
            known={"half_life": -math.log(2.0) / math.log(0.9)},
            note="the spread decays by exactly 0.9 per period",
        )
    )
    flat = np.linspace(0.0, 1.0, 50)
    cases.append(
        ctx.case(
            "half_life_trending_spread",
            "known",
            R.half_life,
            [flat],
            note="a non-reverting spread has no half-life",
        )
    )
    # (b) OLS with a known hedge ratio: noise projected off [1, x] so the fit is exact
    xx = x1[:200]
    e = ctx.normals(70, 200)
    design = np.column_stack([np.ones(200), xx])
    e = e - design @ np.linalg.lstsq(design, e, rcond=None)[0]
    yy = 1.0 + 2.0 * xx + 0.2 * e
    add(
        "known_hedge_ratio",
        "known",
        cols(yy, xx),
        known={"intercept": 1.0, "hedge_ratio": 2.0},
        note="the residual is orthogonal to [1, x] by construction, so OLS returns 1 and 2 to rounding",
    )
    return {"header": header(ctx, "coint"), "cases": cases}


# -------------------------------------------------------------------------------------- hurst


def build_hurst(ctx):
    cases = []
    n = 1024
    iid = ctx.normals(81, n) * 0.01
    z = ctx.normals(82, n + 1) * 0.01
    ma = z[1:] - 0.6 * z[:-1]
    persist = ar1(ctx, 83, n, 0.6, 0.01)
    series = [
        ("iid_normal", "normal", iid),
        ("persistent_ar1", "normal", persist),
        ("anti_persistent_ma", "normal", ma),
        ("drift_plus_noise", "normal", 0.002 + 0.01 * ctx.normals(84, n)),
        ("garch_clustering", "normal", garch_sim(ctx, 85, n, 1e-6, 0.1, 0.85)),
        ("at_min_obs", "min_obs", ctx.normals(86, 256) * 0.01),
        ("below_min_obs", "min_obs", ctx.normals(87, 255) * 0.01),
        ("constant", "degenerate", np.full(300, 0.001)),
        (
            "mostly_zero_runs",
            "degenerate",
            np.where(ctx.uniforms(88, 600) < 0.9, 0.0, 0.01 * ctx.normals(89, 600)),
        ),
        (
            "heavy_tails",
            "degenerate",
            ctx.normals(90, 800) / np.maximum(np.abs(ctx.normals(91, 800)), 0.02) * 0.001,
        ),
        ("n_300", "normal", ctx.normals(92, 300) * 0.01),
        ("n_341_peters_switch", "normal", ctx.normals(93, 1400) * 0.01),
        ("n_4000", "normal", ctx.normals(94, 4000) * 0.01),
        (
            "alternating_sign",
            "degenerate",
            0.01
            * np.where(np.arange(512) % 2 == 0, 1.0, -1.0)
            * (1.0 + 0.01 * ctx.normals(95, 512)),
        ),
    ]
    for name, kind, x in series:
        cases.append(
            ctx.case(
                name,
                kind,
                R.hurst,
                [x],
                note="reading is classified from the unrounded estimates here; users' rounded ones can differ at 0.45 and 0.55"
                if name == "iid_normal"
                else None,
            )
        )
    # E[R/S] at every window size the scales can take: the exact gamma form to 340, Peters' above
    ns = list(range(2, 70)) + [100, 150, 200, 250, 300, 339, 340, 341, 400, 500, 1000, 2000]
    expected = np.array([R.expected_rs(k) for k in ns])
    tables = {"expected_rs_n": np.array(ns), "expected_rs": expected}
    # the scale grid, including lengths whose logspace end point pow misses by an ulp
    scale_lens = list(range(256, 300)) + [341, 400, 512, 1000, 1024, 2048, 5000, 10000]
    scales = {str(k): np.array(R.hurst_scales(k)) for k in scale_lens}
    tables["scales"] = scales
    return {"header": header(ctx, "hurst"), "cases": cases, "tables": enc(ctx.g, tables)}


# -------------------------------------------------------------------------------------- garch


def build_garch(ctx):
    from arch import arch_model

    cases = []

    def lib(r, model, dist):
        am = arch_model(
            r * 100.0,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=1 if model == "gjr" else 0,
            q=1,
            dist="t" if dist == "t" else "normal",
        )
        res = lib_call(lambda: am.fit(disp="off", show_warning=False))
        if isinstance(res, dict):
            return res
        opt = res.optimization_result
        out = {
            "param_names": list(res.params.index),
            "params": np.asarray(res.params),
            "std_err": np.asarray(res.std_err),
            "loglikelihood": float(res.loglikelihood),
            "aic": float(res.aic),
            "convergence_flag": int(res.convergence_flag),
            "iterations": int(opt.nit),
            "function_evaluations": int(opt.nfev),
            "backcast": float(am._backcast),
            "conditional_volatility": np.asarray(res.conditional_volatility),
            "resid": np.asarray(res.resid),
            "param_cov_robust": np.asarray(res.param_cov),
            "cov_type": res.cov_type,
        }
        fc = res.forecast(horizon=10, reindex=False)
        out["forecast_variance_10"] = np.asarray(fc.variance.iloc[-1])
        return out

    def add(name, kind, r, note=None, **kw):
        cases.append(
            ctx.case(
                name,
                kind,
                R.garch,
                [r],
                kw,
                lib=lib(r, kw.get("model", "garch"), kw.get("dist", "t"))
                if len(r) >= 250
                else None,
                note=note,
            )
        )

    n = 1000
    base = garch_sim(ctx, 101, n, 1e-6, 0.08, 0.90)
    add("garch_t", "normal", base)
    add("garch_normal", "normal", base, model="garch", dist="normal")
    add("garch_horizon_1", "normal", base, horizon=1, dist="normal")
    add(
        "garch_horizon_60_annualised",
        "normal",
        base,
        horizon=60,
        periods_per_year=252,
        dist="normal",
    )
    add("gjr_t", "normal", garch_sim(ctx, 102, n, 1e-6, 0.03, 0.9, gamma=0.1), model="gjr")
    add(
        "gjr_normal",
        "normal",
        garch_sim(ctx, 103, n, 1e-6, 0.03, 0.9, gamma=0.1),
        model="gjr",
        dist="normal",
    )
    add("at_min_obs", "min_obs", garch_sim(ctx, 104, 250, 1e-6, 0.08, 0.90), dist="normal")
    add("below_min_obs", "min_obs", garch_sim(ctx, 105, 249, 1e-6, 0.08, 0.90), dist="normal")
    add(
        "near_igarch",
        "degenerate",
        garch_sim(ctx, 106, n, 1e-7, 0.09, 0.909),
        dist="normal",
        note="persistence 0.999: the stationarity boundary",
    )
    add(
        "igarch_exact",
        "degenerate",
        garch_sim(ctx, 107, n, 1e-7, 0.09, 0.91),
        dist="normal",
        note="persistence 1.0 in the generating process",
    )
    add("igarch_t", "degenerate", garch_sim(ctx, 108, n, 1e-7, 0.09, 0.91))
    add(
        "iid_no_arch",
        "degenerate",
        ctx.normals(109, n) * 0.01,
        dist="normal",
        note="no conditional heteroskedasticity: alpha tends to zero",
    )
    add("constant", "degenerate", np.full(300, 0.001))
    add(
        "tiny_returns",
        "degenerate",
        1e-7 * ctx.normals(110, 400),
        dist="normal",
        note="scale far from percent",
    )
    add("huge_returns", "degenerate", 0.5 * ctx.normals(111, 400), dist="normal")
    spiky = ctx.normals(112, 600) * 0.01
    spiky[300] = 0.4
    add("one_outlier", "degenerate", spiky, note="a single 40-sigma bar")
    add("bad_model", "min_obs", base, model="egarch")
    add("bad_horizon", "min_obs", base, horizon=61)
    return {"header": header(ctx, "garch"), "cases": cases}


# ---------------------------------------------------------------------------------------- hmm


def switching_sim(ctx, seed, n, mus, sigmas, stay):
    k = len(mus)
    u = ctx.uniforms(seed, n)
    z = ctx.normals(seed + 1, n)
    state = 0
    r = np.empty(n)
    path = np.empty(n, dtype=int)
    for i in range(n):
        if u[i] > stay[state]:
            state = (state + 1) % k
        path[i] = state
        r[i] = mus[state] + sigmas[state] * z[i]
    return r, path


def build_hmm(ctx):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    cases = []

    def lib(r, k):
        mod = MarkovRegression(r * 100.0, k_regimes=k, trend="c", switching_variance=True)
        res = lib_call(lambda: mod.fit(disp=False, maxiter=200))
        if isinstance(res, dict):
            return res
        return {
            "param_names": list(mod.param_names),
            "start_params": np.asarray(mod.start_params),
            "params": np.asarray(res.params),
            "llf": float(res.llf),
            "aic": float(res.aic),
            "converged": bool(res.mle_retvals.get("converged", False)),
            "regime_transition": np.asarray(res.regime_transition)[:, :, 0],
            "expected_durations": np.asarray(res.expected_durations),
            "smoothed_marginal_probabilities": np.asarray(res.smoothed_marginal_probabilities),
            "filtered_marginal_probabilities": np.asarray(res.filtered_marginal_probabilities),
            "scale": 100.0,
        }

    def add(name, kind, r, k=2, note=None, path=None):
        rec = ctx.case(
            name,
            kind,
            R.markov_regimes,
            [r],
            {"k": k},
            lib=lib(r, k) if len(r) >= 200 * (k - 1) else None,
            note=note,
        )
        if path is not None:
            rec["generating_path"] = enc(ctx.g, path)
        cases.append(rec)

    r, path = switching_sim(ctx, 201, 600, (0.0008, -0.002), (0.005, 0.02), (0.97, 0.95))
    add("two_regimes", "normal", r, path=path)
    r2, path2 = switching_sim(ctx, 203, 800, (0.001, -0.001), (0.004, 0.015), (0.98, 0.96))
    add("two_regimes_800", "normal", r2, path=path2)
    r3, path3 = switching_sim(
        ctx, 205, 900, (0.001, 0.0, -0.003), (0.003, 0.01, 0.03), (0.96, 0.95, 0.94)
    )
    add("three_regimes", "normal", r3, k=3, path=path3)
    add("at_min_obs", "min_obs", r[:200])
    add("below_min_obs", "min_obs", r[:199])
    add("k3_at_min_obs", "min_obs", r3[:400], k=3)
    add("k3_below_min_obs", "min_obs", r3[:399], k=3)
    add("bad_k", "min_obs", r, k=4)
    add(
        "single_regime_data",
        "degenerate",
        ctx.normals(207, 500) * 0.01,
        note="no switching in the data: the regimes are not identified",
    )
    add("constant", "degenerate", np.full(300, 0.001))
    add(
        "three_regimes_on_noise",
        "degenerate",
        ctx.normals(215, 400) * 0.01,
        k=3,
        note="more regimes than the data has",
    )
    add(
        "variance_only_switch",
        "degenerate",
        switching_sim(ctx, 209, 600, (0.0, 0.0), (0.005, 0.02), (0.95, 0.95))[0],
    )
    add(
        "mean_only_switch",
        "degenerate",
        switching_sim(ctx, 211, 600, (0.01, -0.01), (0.005, 0.005), (0.95, 0.95))[0],
    )
    add(
        "short_lived_regime",
        "degenerate",
        switching_sim(ctx, 213, 500, (0.0, -0.01), (0.005, 0.03), (0.995, 0.5))[0],
    )
    return {"header": header(ctx, "hmm"), "cases": cases}


# ---------------------------------------------------------------------------------------- hac


def build_hac(ctx):
    import statsmodels.api as sm
    from scipy import stats as st

    cases = []

    def lib(r, lags):
        n = len(r)
        used = lags if lags is not None else int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        ols = sm.OLS(r, np.ones(n)).fit()
        hac = sm.OLS(r, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": used})
        return {
            "lags_used": used,
            "mean": float(hac.params[0]),
            "hac_bse": float(hac.bse[0]),
            "hac_t": float(hac.tvalues[0]),
            "hac_p": float(hac.pvalues[0]),
            "hac_p_normal": float(2 * st.norm.sf(abs(hac.tvalues[0]))),
            "naive_bse": float(ols.bse[0]),
            "naive_t": float(ols.tvalues[0]),
            "naive_p": float(ols.pvalues[0]),
            "naive_p_t": float(2 * st.t.sf(abs(ols.tvalues[0]), n - 1)),
        }

    def add(name, kind, r, lags=None, note=None, known=None):
        ok = len(r) >= 30 and np.ptp(r) > 0 and (lags is None or 0 <= lags < len(r) // 2)
        cases.append(
            ctx.case(
                name,
                kind,
                R.hac_tstat,
                [r],
                {"lags": lags} if lags is not None else {},
                lib=lib(r, lags) if ok else None,
                note=note,
                known=known,
            )
        )

    add("iid", "normal", 0.001 + 0.01 * ctx.normals(301, 250))
    add("autocorrelated", "normal", 0.001 + ar1(ctx, 302, 500, 0.6, 0.01))
    add("negatively_autocorrelated", "normal", ar1(ctx, 303, 400, -0.5, 0.01))
    add("at_min_obs", "min_obs", 0.001 + 0.01 * ctx.normals(304, 30))
    add("below_min_obs", "min_obs", 0.01 * ctx.normals(305, 29))
    add("constant", "degenerate", np.full(60, 0.001))
    add("near_unit_root_returns", "degenerate", ar1(ctx, 306, 400, 0.99, 0.01))
    add("lags_max", "normal", 0.01 * ctx.normals(307, 100), lags=49)
    add("lags_too_many", "min_obs", 0.01 * ctx.normals(308, 100), lags=50)
    add("lags_negative", "min_obs", 0.01 * ctx.normals(309, 100), lags=-1)
    add("lags_zero", "normal", 0.001 + 0.01 * ctx.normals(310, 200), lags=0)
    add("default_lags_n_1000", "normal", 0.0005 + ar1(ctx, 311, 1000, 0.3, 0.01))
    add("large_mean_tiny_noise", "degenerate", 1.0 + 1e-9 * ctx.normals(312, 100))
    # known: with lags = 0 the HAC variance is sum(e^2) / n^2 and the naive one sum(e^2)/(n(n-1))
    r = 0.002 + 0.01 * ctx.normals(313, 120)
    n = len(r)
    m = math.fsum(r) / n
    s2 = math.fsum((x - m) ** 2 for x in r)
    add(
        "known_white_se",
        "known",
        r,
        lags=0,
        known={
            "mean": m,
            "hac_se": math.sqrt(s2) / n,
            "naive_se": math.sqrt(s2 / (n - 1)) / math.sqrt(n),
        },
        note="lags=0: statsmodels' HAC reduces to the White estimator without a degrees-of-freedom correction",
    )
    return {"header": header(ctx, "hac"), "cases": cases}


# ------------------------------------------------------------------------------------ predictive


def build_predictive(ctx):
    """`raptorbt.stats.predictive_test`: the reference is statsmodels' OLS of
    the forward return on the signal. For a horizon h and lag L the observation of bar t is
    (s[t], p[t + L + h] / p[t + L] - 1); `lib` holds, per horizon, the OLS line over all
    observations (slope, intercept, classical t on n - 2 degrees of freedom), the same line over
    every h-th observation (the headline t and its p), and statsmodels' HAC t of the slope over
    all of them with h - 1 lags (Bartlett, no small-sample correction)."""
    import statsmodels.api as sm
    from scipy import stats as st

    cases = []

    def lib(signal, prices, horizons, lag):
        out = {}
        n = len(signal)
        for h in horizons:
            t = np.array([i for i in range(n - lag - h) if np.isfinite(signal[i])])
            x = signal[t]
            y = prices[t + lag + h] / prices[t + lag] - 1.0
            full = sm.OLS(y, sm.add_constant(x)).fit()
            keep = t % h == 0
            sub = sm.OLS(y[keep], sm.add_constant(x[keep])).fit()
            hac = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": h - 1})
            out[str(h)] = {
                "n": int(len(t)),
                "n_independent": int(keep.sum()),
                "intercept": float(full.params[0]),
                "slope": float(full.params[1]),
                "slope_nonoverlap": float(sub.params[1]),
                "t_ols": float(full.tvalues[1]),
                "t_nonoverlap": float(sub.tvalues[1]),
                "p_nonoverlap": float(2 * st.t.sf(abs(sub.tvalues[1]), keep.sum() - 2)),
                "t_hac": float(hac.tvalues[1]),
            }
        return out

    def add(name, signal, prices, horizons, lag=1, note=None):
        cases.append(
            {
                "name": name,
                "kind": "normal",
                "input": {
                    "args": enc_input(ctx.g, [signal, prices]),
                    "kwargs": {"horizons": horizons, "lag": lag},
                },
                "lib": enc(ctx.g, lib(signal, prices, horizons, lag)),
                **({"note": note} if note else {}),
            }
        )

    def world(seed, n, edge, phi=0.9, vol=0.01):
        z = ctx.normals(seed, n)
        x = np.empty(n)
        prev = 0.0
        for i in range(n):
            prev = phi * prev + z[i]
            x[i] = prev
        e = ctx.normals(seed + 1, n + 1) * vol
        r = e.copy()
        r[2:] += edge * x[: n - 1] * vol  # bar t's signal leads the return two bars on its price: lag 1
        return x, 100.0 * np.cumprod(1.0 + r[:n])

    s, p = world(801, 400, 0.15)
    add("planted_h1", s, p, [1], note="a planted lead of 0.15 sd per unit of an AR(0.9) signal")
    s, p = world(803, 600, 0.10)
    add("planted_h1_h5_h20", s, p, [1, 5, 20])
    s, p = world(805, 500, 0.0, phi=0.97)
    add("no_skill_persistent", s, p, [1, 10], note="no edge: the t's are noise")
    s, p = world(807, 300, 0.2)
    s[:40] = np.nan  # a warm-up
    add("warm_up_nan", s, p, [3], lag=2)
    s, p = world(809, 300, 0.2)
    add("lag_zero", s, p, [2], lag=0)
    return {"header": header(ctx, "predictive"), "cases": cases}


# ---------------------------------------------------------------------------------------- pca


def build_pca(ctx):
    cases = []

    def lib(m, c):
        n, k = m.shape
        if np.ptp(m, axis=0).min() == 0:
            return None
        z = (m - m.mean(axis=0)) / m.std(axis=0, ddof=1)
        u, sv, vt_raw = np.linalg.svd(z, full_matrices=False)
        vt = vt_raw * np.where(vt_raw.sum(axis=1) < 0, -1.0, 1.0)[:, None]
        return {
            "singular_values": sv,
            "vt_raw": vt_raw,
            "vt_signed": vt,
            "row_sums_raw": vt_raw.sum(axis=1),
            "scores": z @ vt.T,
        }

    def factor_panel(seed, n, k, loadings=None, idio=0.5):
        f = ctx.normals(seed, n)
        out = {}
        for j in range(k):
            ld = 1.0 if loadings is None else loadings[j]
            out[f"s{j}"] = ld * f + idio * ctx.normals(seed + 1 + j, n)
        return out

    def add(name, kind, cols, comps=None, target=None, note=None, known=None):
        m = np.column_stack(list(cols.values())) if cols and len(cols) > 0 else None
        kw = {}
        if comps is not None:
            kw["components"] = comps
        if target is not None:
            kw["target"] = target
        ok = (
            m is not None
            and len({len(v) for v in cols.values()}) == 1
            and len(m) >= 60
            and len(cols) >= 2
        )
        cases.append(
            ctx.case(
                name,
                kind,
                R.pca_factors,
                [cols],
                kw,
                lib=lib(m, comps or 3) if ok else None,
                note=note,
                known=known,
            )
        )

    n = 250
    cols = factor_panel(401, n, 6)
    add("six_series_factor", "normal", cols)
    add(
        "six_series_target",
        "normal",
        cols,
        comps=3,
        target=cols["s0"] * 0.5 + 0.2 * ctx.normals(420, n),
    )
    add("five_components", "normal", cols, comps=5)
    add("components_exceed_series", "normal", factor_panel(430, 100, 3), comps=5)
    add(
        "forty_series",
        "normal",
        factor_panel(440, 300, 40, loadings=[1.0 + 0.02 * j for j in range(40)]),
    )
    add("at_min_obs", "min_obs", factor_panel(450, 60, 3))
    add("below_min_obs", "min_obs", factor_panel(455, 59, 3))
    add("one_series", "min_obs", {"a": ctx.normals(460, 100)})
    add("forty_one_series", "min_obs", factor_panel(461, 100, 41))
    x = ctx.normals(470, 200)
    add(
        "two_identical",
        "degenerate",
        {"a": x, "b": x.copy(), "c": ctx.normals(471, 200)},
        note="a perfectly collinear pair: one singular value is zero",
    )
    add(
        "two_identical_with_target",
        "degenerate",
        {"a": x, "b": x.copy(), "c": ctx.normals(472, 200)},
        comps=3,
        target=ctx.normals(474, 200),
        note="three components of a rank-2 panel: the third score is ~0 and the regression is near-singular",
    )
    add("constant_series", "degenerate", {"a": x, "b": np.full(200, 1.0)})
    add("target_wrong_length", "min_obs", cols, target=ctx.normals(480, 100))
    y = ctx.normals(481, 300)
    anti = {"a": y, "b": -y + 0.3 * ctx.normals(482, 300)}
    add(
        "anticorrelated_pair",
        "degenerate",
        anti,
        comps=2,
        note="the dominant component loads (-, +): its raw row sum is exactly 0, so the sign rule (row sum < 0) never fires and the sign is whatever LAPACK returned",
    )
    # known: two series with sample correlation r have eigenvalues 1 +- r, so the first component
    # explains (1 + |r|) / 2, with loadings 1/sqrt(2)
    a, b = ctx.normals(490, 400), ctx.normals(491, 400)
    b = 0.7 * a + 0.5 * b
    ma, mb = math.fsum(a) / len(a), math.fsum(b) / len(b)
    sab = math.fsum((p - ma) * (q - mb) for p, q in zip(a, b))
    saa = math.fsum((p - ma) ** 2 for p in a)
    sbb = math.fsum((q - mb) ** 2 for q in b)
    rho = sab / math.sqrt(saa * sbb)
    add(
        "known_two_series",
        "known",
        {"a": a, "b": b},
        comps=2,
        known={
            "correlation": rho,
            "first_ratio": (1 + abs(rho)) / 2,
            "second_ratio": (1 - abs(rho)) / 2,
            "loading": 1 / math.sqrt(2),
        },
        note="eigenvalues of a 2x2 correlation matrix are 1 +- rho",
    )
    return {"header": header(ctx, "pca"), "cases": cases}


# ---------------------------------------------------------------------------------------- spa


def build_spa(ctx):
    import arch.bootstrap.base as ab
    from arch.bootstrap import SPA

    cases = []

    recorded = []
    original = ab.IIDBootstrap.bootstrap

    def recording(self, reps, *a, **k):
        for item in original(self, reps, *a, **k):
            recorded.append(np.array(self.index, dtype=np.int64))
            yield item

    def run_recorded(r, bench, reps, block, seed):
        """The bootstrap indices arch drew for the reference's call, and arch's SPA internals."""
        recorded.clear()
        ab.IIDBootstrap.bootstrap = recording
        try:
            spa = SPA(-bench, -r, block_size=block, reps=reps, bootstrap="stationary", seed=seed)
            spa.compute()
        finally:
            ab.IIDBootstrap.bootstrap = original
        idx = np.array(recorded, dtype=np.int64)
        ld = spa._loss_diff
        return {
            "indices": idx,
            "reps": reps,
            "block": block,
            "loss_diff_mean": ld.mean(axis=0),
            "loss_diff_var": np.asarray(spa._loss_diff_var),
            "valid_columns": np.asarray(spa._valid_columns),
            "p_lower": float(spa.pvalues["lower"]),
            "p_consistent": float(spa.pvalues["consistent"]),
            "p_upper": float(spa.pvalues["upper"]),
            "simulated_vals": np.asarray(spa._simulated_vals),
            "studentize_flag": bool(spa.studentize),
        }

    def add(name, kind, cols, bench=None, reps=200, block=None, seed=0, note=None, replay=True):
        kw = {"reps": reps, "seed": seed}
        if bench is not None:
            kw["benchmark"] = bench
        if block is not None:
            kw["block_size"] = block
        m = np.column_stack(list(cols.values()))
        n = len(m)
        lib = None
        if replay and n >= 60 and 200 <= reps <= 5000:
            b = np.zeros(n) if bench is None else np.asarray(bench)
            blk = block if block is not None else max(2, int(round(n ** (1.0 / 3.0))))
            if 1 <= blk < n // 2:
                lib = run_recorded(m, b, reps, blk, seed)
                lib["simulated_vals"] = np.asarray(lib["simulated_vals"], dtype=float)
        cases.append(ctx.case(name, kind, R.spa_test, [cols], kw, lib=lib, note=note))

    n = 120
    mus = [0.0, 0.0002, -0.0001, 0.0005, 0.0, 0.0001, -0.0003, 0.0004]
    cols = {f"s{j}": mus[j] + 0.01 * ctx.normals(501 + j, n) for j in range(len(mus))}
    add("eight_strategies", "normal", cols, reps=500, seed=7)
    add("eight_strategies_seed0", "normal", cols, reps=200, seed=0)
    add(
        "with_benchmark",
        "normal",
        cols,
        bench=0.0001 + 0.01 * ctx.normals(520, n),
        reps=200,
        seed=3,
    )
    add("explicit_block", "normal", cols, reps=200, block=10, seed=5)
    add(
        "single_strategy",
        "normal",
        {"only": 0.0006 + 0.01 * ctx.normals(530, 100)},
        reps=200,
        seed=1,
    )
    add(
        "at_min_obs",
        "min_obs",
        {f"s{j}": 0.01 * ctx.normals(540 + j, 60) for j in range(3)},
        reps=200,
        seed=2,
    )
    add(
        "below_min_obs",
        "min_obs",
        {f"s{j}": 0.01 * ctx.normals(545 + j, 59) for j in range(3)},
        reps=200,
        seed=2,
    )
    add("reps_too_few", "min_obs", cols, reps=199, replay=False)
    add("reps_too_many", "min_obs", cols, reps=5001, replay=False)
    add("block_too_large", "min_obs", cols, reps=200, block=60, replay=False)
    wide = {f"s{j}": 0.01 * ctx.normals(600 + j, 80) for j in range(200)}
    add(
        "two_hundred_strategies",
        "normal",
        wide,
        reps=200,
        seed=4,
        note="the cap: the columns are pure noise",
    )
    add(
        "two_hundred_and_one",
        "min_obs",
        {**wide, "extra": 0.01 * ctx.normals(900, 80)},
        reps=200,
        replay=False,
    )
    flat = dict(cols)
    flat["flat"] = np.zeros(n)
    add(
        "zero_variance_strategy",
        "degenerate",
        flat,
        reps=200,
        seed=6,
        note="a strategy that never trades: zero loss differential variance",
    )
    add(
        "all_negative",
        "degenerate",
        {f"s{j}": -0.002 + 0.01 * ctx.normals(700 + j, n) for j in range(5)},
        reps=200,
        seed=8,
        note="every mean is below the benchmark",
    )
    add(
        "identical_strategies",
        "degenerate",
        {"a": cols["s3"], "b": cols["s3"].copy()},
        reps=200,
        seed=9,
        note="two copies of one strategy",
    )
    add(
        "strong_edge",
        "known",
        {"edge": 0.02 + 0.01 * ctx.normals(710, 100), "noise": 0.01 * ctx.normals(711, 100)},
        reps=200,
        seed=10,
        note="a 20-sigma-per-sqrt(n) mean: the p-values must be ~0",
    )
    add("benchmark_wrong_length", "min_obs", cols, bench=np.zeros(n - 1), reps=200, replay=False)
    return {"header": header(ctx, "spa"), "cases": cases}


# ------------------------------------------------------------------------------------- events


def build_events(ctx):
    from scipy import stats as st

    cases = []
    n = 700
    dates = business_days(dt.date(2022, 1, 3), n)
    mk = 0.0004 + 0.01 * ctx.normals(801, n)
    asset = 0.0001 + 1.2 * mk + 0.008 * ctx.normals(802, n)

    def lib(r, m, evs, estimation=120, gap=5, pre=1, post=1):
        out = []
        for ev in evs:
            idx = int(np.searchsorted(np.array(dates, dtype="datetime64[D]"), np.datetime64(ev)))
            lo, hi = idx - pre, idx + post
            est_hi = lo - gap
            est_lo = est_hi - estimation
            if idx >= n or est_lo < 0 or hi >= n:
                out.append({"event": ev.isoformat(), "skipped": True})
                continue
            x, y = m[est_lo:est_hi], r[est_lo:est_hi]
            if x.std() == 0:
                out.append({"event": ev.isoformat(), "skipped": True})
                continue
            xm = np.column_stack([np.ones(len(x)), x])
            beta_ols, *_ = np.linalg.lstsq(xm, y, rcond=None)
            out.append(
                {
                    "event": ev.isoformat(),
                    "skipped": False,
                    "idx": idx,
                    "alpha_ols": float(beta_ols[0]),
                    "beta_ols": float(beta_ols[1]),
                }
            )
        return {"events": out}

    def add(name, kind, evs, r=asset, m=mk, d=dates, note=None, known=None, **kw):
        ok = len(d) == len(r) == len(m)
        cases.append(
            ctx.case(
                name,
                kind,
                R.event_study,
                [d, r, m, evs],
                kw,
                lib=lib(r, m, evs, **kw) if ok else None,
                note=note,
                known=known,
            )
        )

    add("three_events", "normal", [dates[300], dates[400], dates[500]])
    add("one_event", "normal", [dates[350]], note="one event: the cross-sectional t is absent")
    add("two_events", "normal", [dates[300], dates[301]], note="overlapping estimation windows")
    add("wider_window", "normal", [dates[300], dates[450]], pre=5, post=10)
    add("zero_window", "normal", [dates[300], dates[450]], pre=0, post=0)
    add("custom_estimation", "normal", [dates[300], dates[450]], estimation=60, gap=0)
    saturday = dates[300] + dt.timedelta(days=(5 - dates[300].weekday()) % 7 or 7)
    add(
        "off_calendar_event",
        "normal",
        [saturday, dates[450]],
        note="a Saturday moves to the next trading day",
    )
    add(
        "early_and_late_skipped",
        "normal",
        [dates[10], dates[300], dates[699], dt.date(2030, 1, 1)],
        note="no estimation window; no event window; after the last bar",
    )
    add("all_skipped", "min_obs", [dates[5], dates[6]])
    add("too_many_events", "min_obs", [dates[300]] * 101)
    add("no_events", "min_obs", [])
    add("bad_estimation", "min_obs", [dates[300]], estimation=59)
    add("bad_window", "min_obs", [dates[300]], pre=21)
    add("length_mismatch", "min_obs", [dates[300]], r=asset[:-1])
    add("unsorted_dates", "min_obs", [dates[300]], d=[dates[1], dates[0]] + dates[2:])
    flatmk = mk.copy()
    flatmk[150:300] = 0.0
    add(
        "market_flat_in_estimation",
        "degenerate",
        [dates[290], dates[450]],
        m=flatmk,
        r=0.0001 + 1.2 * flatmk + 0.008 * ctx.normals(803, n),
        note="the first event's estimation window has a constant market",
    )
    # known: a noise-free market model, with +0.02 on the event bar
    r0 = 0.001 + 1.5 * mk
    r0 = r0.copy()
    r0[300] += 0.02
    add(
        "known_noise_free",
        "known",
        [dates[300]],
        r=r0,
        known={"alpha": 0.001, "beta": 1.5, "abnormal_return_on_event": 0.02, "car": 0.02},
        note="no residual: sigma is rounding noise (~1e-18, still > 0), so the per-event t is a huge number, not NaN; CAR = 0.02 to rounding",
    )
    # the cross-sectional p-value by scipy for the 3-event case
    c = np.array([0.01, -0.004, 0.02, 0.003])
    t_x = c.mean() / (c.std(ddof=1) / math.sqrt(len(c)))
    tables = {
        "caar_p_example": {"car": c, "t": float(t_x), "p": float(2 * st.t.sf(abs(t_x), len(c) - 1))}
    }
    return {"header": header(ctx, "events"), "cases": cases, "tables": enc(ctx.g, tables)}


# --------------------------------------------------------------------------------------- pbo


def build_pbo(ctx):
    cases = []

    def cols_of(m):
        return {f"s{j}": m[:, j] for j in range(m.shape[1])}

    def noise(seed, n, k, scale=0.01):
        return scale * ctx.normals(seed, n * k).reshape(k, n).T

    def add(name, kind, cols, splits=None, note=None, known=None):
        kw = {} if splits is None else {"splits": splits}
        cases.append(ctx.case(name, kind, R.pbo_cscv, [cols], kw, note=note, known=known))

    add("pure_noise_10", "normal", cols_of(noise(901, 640, 10)))
    m = noise(902, 640, 10)
    m[:, 3] += 0.002
    add("one_real_edge", "normal", cols_of(m), note="strategy 3 has a true mean of 0.2 sigma")
    add("splits_4", "normal", cols_of(noise(903, 80, 6)), splits=4)
    add("splits_8", "normal", cols_of(noise(904, 160, 6)), splits=8)
    add("splits_12", "normal", cols_of(noise(905, 240, 6)), splits=12)
    add("at_min_rows", "min_obs", cols_of(noise(906, 64, 4)), splits=16)
    add("below_min_rows", "min_obs", cols_of(noise(907, 63, 4)), splits=16)
    add(
        "rows_not_multiple_of_splits",
        "normal",
        cols_of(noise(908, 650, 5)),
        splits=16,
        note="the 10 oldest rows are dropped",
    )
    add("odd_splits", "min_obs", cols_of(noise(909, 100, 3)), splits=5)
    add("splits_too_few", "min_obs", cols_of(noise(910, 100, 3)), splits=2)
    add(
        "too_many_combinations",
        "min_obs",
        cols_of(noise(911, 400, 3)),
        splits=20,
        note="C(20, 10) = 184756 > 20000",
    )
    add("one_strategy", "min_obs", {"a": ctx.normals(912, 200)})
    wide = noise(913, 128, 200)
    add("two_hundred_strategies", "normal", cols_of(wide), splits=8)
    add("two_hundred_and_one", "min_obs", cols_of(noise(914, 128, 201)), splits=8)
    c = noise(915, 160, 1)[:, 0]
    add(
        "identical_columns",
        "degenerate",
        {"a": c, "b": c.copy(), "c": c.copy()},
        splits=8,
        note="ties everywhere: rank = (k+1)/2",
    )
    zero = noise(916, 160, 4)
    zero[:, 1] = 0.0
    add("zero_variance_column", "degenerate", cols_of(zero), splits=8)
    add(
        "constant_nonzero_column",
        "degenerate",
        {**cols_of(noise(917, 160, 3)), "flat": np.full(160, 0.001)},
        splits=8,
    )
    # known answers
    a = 0.01 + 0.001 * ctx.normals(918, 160)
    b = 0.002 + 0.001 * ctx.normals(919, 160)
    add(
        "known_dominant",
        "known",
        {"a": a, "b": b},
        splits=8,
        known={"pbo": 0.0, "median_logit": math.log(2.0), "prob_loss": 0.0},
        note="a beats b in every block: the in-sample best is the out-of-sample best, rank 2 of 2, w = 2/3",
    )
    add(
        "known_identical_two",
        "known",
        {"a": c, "b": c.copy()},
        splits=8,
        known={"pbo": 1.0, "median_logit": 0.0},
        note="two identical columns tie: rank 1.5, w = 1/2, logit 0, so pbo = 1",
    )
    return {"header": header(ctx, "pbo"), "cases": cases}


# ------------------------------------------------------------------------------------ impact


def build_impact(ctx):
    g = ctx.g
    cases = []

    def bars(seed, n, start=100.0):
        o, h, lo, c, v = g.scenario_random_walk(seed, n)
        return [np.array(x) for x in (h, lo, c, v)]

    def add(name, kind, h, lo, c, v, quantity, note=None, known=None, **kw):
        cases.append(
            ctx.case(
                name,
                kind,
                R.impact_estimate,
                [h, lo, c, v],
                {"quantity": quantity, **kw},
                note=note,
                known=known,
            )
        )

    h, lo, c, v = bars(1001, 60)
    adv = float(v.mean())
    add("corwin_schultz", "normal", h, lo, c, v, 0.01 * adv)
    add(
        "roll_on_random_walk",
        "normal",
        h,
        lo,
        c,
        v,
        0.01 * adv,
        spread_model="roll",
        note="a random walk: the sample covariance of changes is a small number of either sign",
    )
    # momentum in the changes (AR(1) with phi 0.8): positive serial covariance, Roll refuses
    mom = 100.0 + np.cumsum(ar1(ctx, 1007, 60, 0.8, 0.5))
    add(
        "roll_refused_momentum",
        "degenerate",
        mom + 0.3,
        mom - 0.3,
        mom,
        v,
        0.01 * adv,
        spread_model="roll",
    )
    # bid-ask bounce: close = 100 + 0.25 * s with s = +-1 iid, so changes are negatively autocorrelated
    s = np.where(ctx.uniforms(1002, 80) < 0.5, 1.0, -1.0)
    cb = 100.0 + 0.25 * s
    hb, lb = cb + 0.3, cb - 0.3
    vb = 1e5 + 1e3 * ctx.normals(1003, 80)
    add("roll_with_bounce", "normal", hb, lb, cb, vb, 5000.0, spread_model="roll")
    add("coefficient_two", "normal", h, lo, c, v, 0.02 * adv, coefficient=2.0)
    add("beyond_model_range", "normal", h, lo, c, v, 0.5 * adv, note="participation 0.5 > 0.25")
    h20, l20, c20, v20 = bars(1004, 20)
    add("at_min_days", "min_obs", h20, l20, c20, v20, 1000.0)
    h19, l19, c19, v19 = bars(1005, 19)
    add("below_min_days", "min_obs", h19, l19, c19, v19, 1000.0)
    add("zero_quantity", "min_obs", h, lo, c, v, 0.0)
    add("bad_spread_model", "min_obs", h, lo, c, v, 100.0, spread_model="vwap")
    add("length_mismatch", "min_obs", h[:-1], lo, c, v, 100.0)
    add(
        "zero_volume",
        "degenerate",
        h,
        lo,
        c,
        np.zeros(60),
        100.0,
        note="an index carries no volume",
    )
    add("high_below_low", "degenerate", lo, h, c, v, 100.0)
    add("negative_price", "degenerate", h, lo, c - 200.0, v, 100.0)
    add(
        "flat_bars",
        "degenerate",
        c.copy(),
        c.copy(),
        c,
        v,
        0.01 * adv,
        note="high = low: the log range is 0",
    )
    add(
        "constant_price",
        "degenerate",
        np.full(30, 100.5),
        np.full(30, 99.5),
        np.full(30, 100.0),
        np.full(30, 1e5),
        1000.0,
        note="zero daily sigma",
    )
    # a day whose two-day estimate is negative counts as zero
    hq, lq = np.full(30, 100.2), np.full(30, 99.8)
    hq[::2] = 101.0
    lq[1::2] = 99.0
    add(
        "negative_two_day_estimates",
        "degenerate",
        hq,
        lq,
        np.full(30, 100.0) + 0.05 * ctx.normals(1006, 30),
        np.full(30, 1e5),
        1000.0,
    )
    # known: flat bars give a spread of exactly 0
    cases.append(
        ctx.case(
            "known_corwin_schultz_flat",
            "known",
            R.corwin_schultz,
            [np.full(10, 100.0), np.full(10, 100.0)],
            known={"spread": 0.0},
        )
    )
    # known: alternating 100, 101 gives changes +-1: sample covariance of lagged changes in exact arithmetic
    alt = np.array([100.0, 101.0] * 10)
    d = [Fraction(int(x)) for x in np.diff(alt)]
    x, y = d[:-1], d[1:]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - 1)
    mean_price = Fraction(sum(Fraction(int(a)) for a in alt), len(alt))
    roll_exact = 2 * math.sqrt(-float(cov)) / float(mean_price)
    cases.append(
        ctx.case(
            "known_roll_alternating",
            "known",
            R.roll,
            [alt],
            known={"covariance": float(cov), "roll": roll_exact},
            note="covariance computed in exact rational arithmetic",
        )
    )
    # quantity_for_notional (kept in the tool layer): fixture for the arithmetic only
    cases.append(
        ctx.case(
            "quantity_for_notional",
            "known",
            R.quantity_for_notional,
            [250000.0, 125.0],
            known={"quantity": 2000.0},
        )
    )
    cases.append(
        ctx.case("quantity_for_notional_bad", "min_obs", R.quantity_for_notional, [0.0, 125.0])
    )
    return {"header": header(ctx, "impact"), "cases": cases}


# ----------------------------------------------------------------------------------- surface


def build_surface(ctx):
    cases = []

    def grid2(f, xs, ys):
        return [([x, y], f(x, y)) for x in xs for y in ys]

    def add(name, kind, names, cells, note=None, known=None):
        cases.append(
            ctx.case(name, kind, R.sensitivity_surface, [names, cells], note=note, known=known)
        )

    xs = [5, 10, 15, 20, 25, 30, 35]
    ys = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    add(
        "smooth_plateau",
        "normal",
        ["fast", "mult"],
        grid2(lambda x, y: 1.5 - 0.0004 * (x - 20) ** 2 - 0.05 * (y - 2.5) ** 2, xs, ys),
    )
    spike = grid2(lambda x, y: 0.2 + 0.01 * ctx.normals(1101 + int(x + 10 * y), 1)[0], xs, ys)
    spike = [(p, 2.5 if p == [20, 2.5] else v) for p, v in spike]
    add("spike", "normal", ["fast", "mult"], spike)
    add(
        "one_parameter",
        "normal",
        ["period"],
        [([p], 1.0 + math.sin(p / 7.0)) for p in range(5, 50, 4)],
    )
    add(
        "three_parameters",
        "normal",
        ["a", "b", "c"],
        [
            ([a, b, c], -((a - 2) ** 2) - (b - 3) ** 2 - 0.5 * (c - 1) ** 2)
            for a in (1, 2, 3, 4)
            for b in (2, 3, 4)
            for c in (0, 1, 2)
        ],
    )
    holes = [
        (p, v) for i, (p, v) in enumerate(grid2(lambda x, y: x / 10 + y, xs, ys)) if i % 5 != 0
    ]
    add("coverage_below_one", "normal", ["x", "y"], holes, note="every fifth cell was not run")
    add(
        "negative_metrics",
        "normal",
        ["x", "y"],
        grid2(lambda x, y: -3.0 - 0.01 * (x - 20) ** 2 - 0.2 * (y - 2.5) ** 2, xs, ys),
    )
    add(
        "uneven_axis_steps",
        "normal",
        ["x", "y"],
        grid2(lambda x, y: -abs(x - 40) / 50 - abs(y - 2) * 0.1, [5, 10, 40, 200], [1.0, 2.0, 5.0]),
        note="neighbours are the next distinct values, not the next integers",
    )
    add(
        "tied_best_cells",
        "degenerate",
        ["x", "y"],
        grid2(lambda x, y: 1.0 if (x, y) in {(10, 2.0), (30, 3.0)} else 0.0, xs, ys),
        note="two best cells: the first in input order wins",
    )
    add(
        "tied_best_cells_reversed",
        "degenerate",
        ["x", "y"],
        list(
            reversed(grid2(lambda x, y: 1.0 if (x, y) in {(10, 2.0), (30, 3.0)} else 0.0, xs, ys))
        ),
        note="the same cells in reverse order",
    )
    iso = [
        ([1, 1], 0.5),
        ([1, 2], 0.4),
        ([2, 1], 0.45),
        ([2, 2], 0.42),
        ([3, 3], 9.0),
        ([4, 4], 0.1),
        ([5, 5], 0.2),
        ([6, 6], 0.1),
        ([7, 7], 0.3),
    ]
    add(
        "isolated_best_cell",
        "degenerate",
        ["x", "y"],
        iso,
        note="the best cell has no neighbour: share 0 reads as a spike",
    )
    add("all_isolated", "degenerate", ["x", "y"], [([i, i], float(i)) for i in range(1, 11)])
    add("constant_metric", "degenerate", ["x", "y"], grid2(lambda x, y: 1.0, xs[:3], ys[:3]))
    add(
        "axis_with_one_value",
        "degenerate",
        ["x", "y"],
        grid2(lambda x, y: x * 1.0, list(range(1, 10)), [1.0]),
    )
    add(
        "duplicate_cell",
        "degenerate",
        ["x", "y"],
        grid2(lambda x, y: x + y, xs[:3], ys[:3]) + [([5, 1.0], 9.0)],
    )
    add(
        "non_finite_metric",
        "degenerate",
        ["x", "y"],
        grid2(lambda x, y: x + y, xs[:3], ys[:3])[:-1] + [([15, 2.0], float("nan"))],
    )
    add("too_few_cells", "min_obs", ["x", "y"], grid2(lambda x, y: x + y, xs[:2], ys[:2]))
    add("too_many_cells", "min_obs", ["x"], [([i], float(i % 7)) for i in range(2001)])
    add(
        "too_many_parameters",
        "min_obs",
        ["a", "b", "c", "d"],
        [([a, b, 1, 1], float(a + b)) for a in range(3) for b in range(4)],
    )
    add("repeated_name", "min_obs", ["x", "x"], grid2(lambda x, y: x + y, xs[:3], ys[:3]))
    add(
        "two_hundred_cells",
        "normal",
        ["a", "b"],
        grid2(
            lambda x, y: math.sin(x / 3.0) * math.cos(y / 2.0),
            list(range(1, 21)),
            list(range(1, 11)),
        ),
    )
    # known: metric = row-major 1..9 on a 3x3 grid. Neighbour differences: 6 horizontal of 1,
    # 6 vertical of 3, mean 2 over a range of 8 -> roughness 1/4, smoothness 3/4. The best (9) has
    # neighbours 6 and 8; the floor is 9 - 8/5 = 7.4, so one of two holds: share 1/2, "mixed".
    cells = [([i, j], float(3 * i + j + 1)) for i in range(3) for j in range(3)]
    add(
        "known_three_by_three",
        "known",
        ["i", "j"],
        cells,
        known={
            "smoothness": 0.75,
            "plateau_share": 0.5,
            "verdict": "mixed",
            "best_metric": 9.0,
            "neighbour_mean_of_best": 7.0,
        },
        note="derived by hand in the builder",
    )
    return {"header": header(ctx, "surface"), "cases": cases}


# ------------------------------------------------------------------------------------ quality

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _epoch(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=IST).timestamp()


def build_quality(ctx):
    g = ctx.g
    cases = []

    def daily(n, start=dt.date(2024, 1, 1), seed=1201):
        days = business_days(start, n)
        t = np.array([_epoch(d.year, d.month, d.day, 0, 0) for d in days])
        o, h, lo, c, v = g.scenario_random_walk(seed, n)
        return t, np.array(o), np.array(h), np.array(lo), np.array(c), np.array(v)

    def intraday(days=3, per_day=75, seed=1202, step=300):
        ts, base = [], business_days(dt.date(2024, 1, 2), days)
        for d in base:
            for k in range(per_day):
                ts.append(_epoch(d.year, d.month, d.day, 9, 15) + k * step)
        n = len(ts)
        o, h, lo, c, v = g.scenario_random_walk(seed, n)
        return np.array(ts), np.array(o), np.array(h), np.array(lo), np.array(c), np.array(v)

    def add(name, kind, bars, interval=86400, note=None, known=None):
        cases.append(
            ctx.case(
                name,
                kind,
                R.audit_bars,
                list(bars),
                {"interval_seconds": interval},
                note=note,
                known=known,
            )
        )

    t, o, h, lo, c, v = daily(80)
    add("clean_daily", "normal", (t, o, h, lo, c, v))
    ti, oi, hi, li, ci, vi = intraday()
    add("clean_intraday", "normal", (ti, oi, hi, li, ci, vi), interval=300)
    n = 5
    add("at_min_bars", "min_obs", tuple(a[:n] for a in (t, o, h, lo, c, v)))
    add("below_min_bars", "min_obs", tuple(a[:4] for a in (t, o, h, lo, c, v)))
    # planted defects with known counts
    t2, o2, h2, l2, c2, v2 = (a.copy() for a in (t, o, h, lo, c, v))
    t2[10] = t2[9]
    t2[20], t2[21] = t2[21], t2[20]
    h2[30] = l2[30] - 1.0
    v2[40] = 0.0
    v2[41] = 0.0
    o2[50] = h2[50] = l2[50] = c2[50] = c2[49]
    add(
        "planted_defects",
        "known",
        (t2, o2, h2, l2, c2, v2),
        known={
            "duplicate_timestamps": 1,
            "out_of_order": 1,
            "impossible_ohlc": 1,
            "zero_volume_bars": 2,
            "flat_bars": 1,
        },
        note="one duplicate time, one swapped pair (one backward step), one high below low, two zero volumes, one flat bar",
    )
    t3, o3, h3, l3, c3, v3 = (a.copy() for a in (t, o, h, lo, c, v))
    c3[55:] *= 8.0
    o3[55:], h3[55:], l3[55:] = o3[55:] * 8.0, h3[55:] * 8.0, l3[55:] * 8.0
    add(
        "return_outlier_and_split_up",
        "degenerate",
        (t3, o3, h3, l3, c3, v3),
        note="an 8x jump: a return outlier and not a listed split factor",
    )
    t4, o4, h4, l4, c4, v4 = (a.copy() for a in (t, o, h, lo, c, v))
    for arr in (o4, h4, l4, c4):
        arr[60:] /= 2.0
    add(
        "two_for_one_split",
        "degenerate",
        (t4, o4, h4, l4, c4, v4),
        note="close ratio 0.5: a suspected unadjusted split",
    )
    np.delete(t, range(30, 40))
    add(
        "daily_gap",
        "degenerate",
        tuple(np.delete(a, range(30, 40)) for a in (t, o, h, lo, c, v)),
        note="ten sessions missing",
    )
    keep = np.ones(len(ti), dtype=bool)
    keep[30:40] = False
    keep[110:112] = False
    add(
        "intraday_gaps",
        "degenerate",
        tuple(a[keep] for a in (ti, oi, hi, li, ci, vi)),
        interval=300,
        note="10 bars missing mid-session (flagged); 2 missing elsewhere is a hole of 3 bar-widths, not more than 3, so not flagged",
    )
    keep2 = np.ones(len(ti), dtype=bool)
    keep2[30:33] = False
    add(
        "intraday_gap_threshold",
        "degenerate",
        tuple(a[keep2] for a in (ti, oi, hi, li, ci, vi)),
        interval=300,
        note="three bars missing: a hole of 4 bar-widths, the smallest flagged (missing_bars 3)",
    )
    add(
        "all_flat",
        "degenerate",
        (
            t[:30],
            np.full(30, 100.0),
            np.full(30, 100.0),
            np.full(30, 100.0),
            np.full(30, 100.0),
            np.full(30, 0.0),
        ),
    )
    nanbars = tuple(a.copy() for a in (t, o, h, lo, c, v))
    nanbars[4][12] = float("nan")
    add("nan_close", "degenerate", nanbars)
    neg = tuple(a.copy() for a in (t, o, h, lo, c, v))
    neg[1][8] = -1.0
    add("negative_open", "degenerate", neg)
    many = tuple(a.copy() for a in (t, o, h, lo, c, v))
    for k in range(7):
        many[0][10 + 2 * k] = many[0][9 + 2 * k]
    add(
        "seven_duplicates",
        "degenerate",
        many,
        note="the listed examples cap at 5; the count does not",
    )
    # a quiet series with one extreme bar is flagged; a noisy one is not
    q = tuple(a.copy() for a in (t, o, h, lo, c, v))
    q[4][:] = 100.0 * np.exp(np.cumsum(0.001 * ctx.normals(1210, len(t))))
    q[1][:], q[2][:], q[3][:] = q[4] * 1.0005, q[4] * 1.002, q[4] * 0.998
    q[4][50] *= 1.2
    q[2][50] *= 1.2
    add("outlier_in_quiet_series", "degenerate", q)
    return {"header": header(ctx, "quality"), "cases": cases}


BUILDERS = {
    "stationarity": build_stationarity,
    "coint": build_coint,
    "hurst": build_hurst,
    "garch": build_garch,
    "hmm": build_hmm,
    "hac": build_hac,
    "pca": build_pca,
    "spa": build_spa,
    "events": build_events,
    "pbo": build_pbo,
    "impact": build_impact,
    "surface": build_surface,
    "quality": build_quality,
    "predictive": build_predictive,
}


def write_all(g, out_dir, only=None):
    """Write every statistics fixture (or the families in `only`); returns {file: bytes}."""
    ctx = Ctx(g)
    sizes = {}
    names = ["special"] + list(BUILDERS)
    for fam in names:
        if only and fam not in only:
            continue
        if fam == "special":
            obj = build_special(ctx)
            path = os.path.join(out_dir, SPECIAL_FILE)
            os.makedirs(out_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(obj, f, separators=(",", ":"), sort_keys=True)
            sizes[SPECIAL_FILE] = os.path.getsize(path)
        else:
            obj = BUILDERS[fam](ctx)
            fname = FAMILY_FILES[fam]
            sizes[fname] = g.write_gz(os.path.join(out_dir, fname), obj)
        print(
            f"{fam}: {sizes[SPECIAL_FILE if fam == 'special' else FAMILY_FILES[fam]]} bytes",
            flush=True,
        )
    return sizes
