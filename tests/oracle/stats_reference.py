"""Reference answers for the statistics fixtures, written from the published definitions.

`stats_fixtures.py` calls these functions to fill each fixture case's `reference_full` (what
the engine must return at full precision) and `reference` (the same answer in plain JSON). The
module imports numpy, scipy, statsmodels and arch (permissive licences, credited in `NOTICE`)
and nothing from raptorbt: a reference that imported the engine would judge nothing.

Where a public library computes the statistic, its output is the reference:

    stationarity   statsmodels `adfuller` (AIC lags) and `kpss` (level, automatic lags)
    cointegration  statsmodels `OLS`, `coint` and `coint_johansen`
    garch          arch `arch_model(...).fit()` on returns in percent
    markov_regimes statsmodels `MarkovRegression(trend="c", switching_variance=True)`
    hac_tstat      statsmodels `OLS(r, 1).fit(cov_type="HAC")`
    spa_test       arch `SPA` (stationary bootstrap, the given seed)

The rest have no library implementation, and are written here in plain numpy from the papers
cited at each function: the Hurst exponent (rescaled range with the Anis-Lloyd-Peters
expectation, and DFA), principal components and the factor regression, the probability of
backtest overfitting (CSCV), the market-model event study, the square-root impact law with the
Corwin-Schultz and Roll spreads, the parameter-sensitivity surface and the bar audit. The
derived quantities around a library fit (a half-life, a regime timeline, a forecast path) are
written here too, each from its definition.

What is not a statistic is a contract: the argument limits, the order in which arguments are
checked and the wording of a refusal are raptorbt's documented API (`skills/statistics/
SKILL.md`, `docs/errors.md`), and a refusal here carries that wording and its error `code`.
The free-text fields of an answer (`assumptions`, `reading_rule`, `formula`, ...) are not
references and are not produced.

Conventions. A refusal raises `Refusal(code, message)`. A library that cannot answer raises its
own exception (statsmodels' `IndexError` for a MacKinnon p-value beyond six series, numpy's
`LinAlgError` for a singular Johansen moment matrix): the fixture records that as the reference's
answer, because it is what the reference does. Floats are Python floats, NaN where a quantity is
undefined; `None` where the answer has no value.
"""

import bisect
import datetime as dt
import itertools
import math
import warnings

import numpy as np
from scipy import special as sp
from scipy import stats as st


class Refusal(Exception):
    """A statistic that cannot answer: `code` is raptorbt's error code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _too_few(what, found, needed):
    return Refusal(
        "insufficient_observations", f"{what}: {found} observations, this statistic needs {needed}"
    )


def _too_few_series(what, found, needed):
    return Refusal(
        "insufficient_observations", f"{what}: {found} series, this statistic needs {needed}"
    )


def _invalid(message):
    return Refusal("invalid_input", message)


def _degenerate(message):
    return Refusal("degenerate_series", message)


def _finite(x, what, needed):
    """A float array of at least `needed` finite values (length checked first)."""
    a = np.asarray(x, dtype=float)
    if len(a) < needed:
        raise _too_few(what, len(a), needed)
    if not np.all(np.isfinite(a)):
        raise _invalid(f"{what} holds a NaN or infinite value")
    return a


def _mean(x):
    return math.fsum(x) / len(x)


def _f(x):
    return None if x is None else float(x)


# ------------------------------------------------------------------------------ stationarity

STATIONARITY_MIN_OBS = 50


def stationarity(x, alpha=0.05):
    """Augmented Dickey-Fuller (Dickey and Fuller 1979, Said and Dickey 1984; constant, lag by
    AIC, MacKinnon 1994 p-value and 2010 critical values) and KPSS (Kwiatkowski, Phillips,
    Schmidt and Shin 1992; level, Hobijn-Franses-Ooms 2004 automatic lags), by statsmodels.
    `reading` crosses the two decisions at `alpha`: ADF rejects a unit root and KPSS does not
    reject stationarity is `stationary`; the reverse `unit_root`; both `contradictory`; neither
    `inconclusive`. `p_value_clipped` is statsmodels' InterpolationWarning: the KPSS statistic
    is outside its table and the p-value sits on a bound."""
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.stattools import adfuller, kpss

    x = _finite(x, "series", STATIONARITY_MIN_OBS)
    if np.all(x == x[0]):
        raise _degenerate("series is constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = adfuller(x, regression="c", autolag="AIC", result_object=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k = kpss(x, regression="c", nlags="auto", result_object=True)
    clipped = any(issubclass(w.category, InterpolationWarning) for w in caught)
    rejects_unit_root = bool(a.pvalue < alpha)
    rejects_stationarity = bool(k.pvalue < alpha)
    reading = {
        (True, False): "stationary",
        (False, True): "unit_root",
        (True, True): "contradictory",
        (False, False): "inconclusive",
    }[(rejects_unit_root, rejects_stationarity)]
    return {
        "n": len(x),
        "alpha": float(alpha),
        "adf": {
            "statistic": float(a.statistic),
            "p_value": float(a.pvalue),
            "lags": int(a.lags),
            "critical_values": {key: float(v) for key, v in a.critical_values.items()},
            "rejects_unit_root": rejects_unit_root,
        },
        "kpss": {
            "statistic": float(k.statistic),
            "p_value": float(k.pvalue),
            "lags": int(k.lags),
            "critical_values": {key: float(v) for key, v in k.critical_values.items()},
            "p_value_clipped": clipped,
            "rejects_stationarity": rejects_stationarity,
        },
        "reading": reading,
    }


# ----------------------------------------------------------------------------- cointegration

COINT_MIN_OBS = 60
COINT_MAX_SERIES = 12


def half_life(spread):
    """Periods for a deviation of the spread to halve: the AR(1) of the spread written as the
    change on the lagged level, `ds_t = a + b s_{t-1}`, so `phi = 1 + b` and the half-life is
    `ln 2 / -ln phi` (the Ornstein-Uhlenbeck reading of an AR(1)). None when the spread does not
    revert (`phi` outside (0, 1)) or has fewer than three values."""
    s = np.asarray(spread, dtype=float)
    if len(s) < 3:
        return None
    lag, diff = s[:-1], np.diff(s)
    x = np.column_stack([np.ones(len(lag)), lag])
    coef, _, rank, _ = np.linalg.lstsq(x, diff, rcond=None)
    if rank < 2:
        return None
    phi = 1.0 + coef[1]
    if not 0.0 < phi < 1.0:
        return None
    return -math.log(2.0) / math.log(phi)


def cointegration(cols, alpha=0.05):
    """Engle and Granger (1987): OLS of the first series on a constant and the others, the ADF
    test without constant on the residual (AIC lags), MacKinnon (1994) p-value for `N` series and
    (2010) critical values: statsmodels' `coint`. Johansen (1991): trace statistics of the VECM
    with a constant in the relation and one lagged difference, statsmodels' `coint_johansen`
    (`det_order` 0, `k_ar_diff` 1) with the MacKinnon-Haug-Michelis (1999) critical values; the
    rank is the number of hypotheses rejected in turn at 5%; the cointegrating vector is the
    first eigenvector over its first entry. Reinsel and Ahn (1992): the trace statistics scaled
    by `(T - q k) / T`, `T` the observations the VECM uses, `q` the series, `k = 2` the order of
    the VAR in levels; `small_sample` when that factor is under 0.9.

    The spread is the OLS residual: its mean, its sample standard deviation (n - 1), the z of its
    last value and its half-life (`half_life`)."""
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    names = list(cols)
    q = len(names)
    if q > COINT_MAX_SERIES:
        raise _invalid(f"{q} series, at most {COINT_MAX_SERIES}")
    if q < 2:
        raise _invalid(f"series: {q} series, this statistic needs 2")
    arrays = [_finite(cols[k], f"series[{k}]", COINT_MIN_OBS) for k in names]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise _invalid("series: the series differ in length")
    for k, a in zip(names, arrays):
        if np.all(a == a[0]):
            raise _degenerate(f"series[{k}] is constant")
    m = np.column_stack(arrays)
    y, x = m[:, 0], m[:, 1:]
    ols = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, crit = coint(y, x, trend="c", autolag="aic")  # IndexError from 7 series
    spread = np.asarray(ols.resid)
    s_mean = _mean(spread)
    s_std = math.sqrt(math.fsum((spread - s_mean) ** 2) / (n - 1))
    eg = {
        "dependent": names[0],
        "statistic": float(stat),
        "p_value": float(p),
        "critical_values": dict(zip(("1%", "5%", "10%"), (float(c) for c in crit))),
        "cointegrated": bool(p < alpha),
        "intercept": float(ols.params[0]),
        "hedge_ratios": {k: float(b) for k, b in zip(names[1:], ols.params[1:])},
        "spread_mean": s_mean,
        "spread_std": s_std,
        "spread_last_z": (float(spread[-1]) - s_mean) / s_std,
        "half_life_periods": half_life(spread),
    }
    jo = coint_johansen(m, det_order=0, k_ar_diff=1)  # LinAlgError on a singular moment matrix
    eig = np.asarray(jo.eig)
    if np.any(np.imag(eig) != 0) or np.any(~(np.real(eig) >= 0)) or np.any(np.real(eig) >= 1):
        # squared canonical correlations lie in [0, 1): an eigenvalue outside is what a moment
        # matrix singular to working precision gives, not an answer
        raise _degenerate(
            "Johansen's eigenvalues are outside [0, 1): its moment matrices are singular to "
            "working precision"
        )
    trace = [float(v) for v in np.asarray(jo.lr1)]
    trace_cv = [float(r[1]) for r in np.asarray(jo.cvt)]

    def rank_of(stats):
        r = 0
        while r < q and stats[r] > trace_cv[r]:
            r += 1
        return r

    t_obs = n - 1 - 1  # one difference, one lagged difference
    scale = max(t_obs - q * 2, 0) / t_obs
    scaled = [v * scale for v in trace]
    first = np.real(np.asarray(jo.evec)[:, 0])
    vector = first / first[0] if first[0] != 0 else first
    johansen = {
        "rank_at_5pct": rank_of(trace),
        "trace_statistics": trace,
        "trace_critical_5pct": trace_cv,
        "eigenvalues": [float(v) for v in np.real(np.asarray(jo.eig))],
        "max_eigenvalue_statistics": [float(v) for v in np.asarray(jo.lr2)],
        "max_eigenvalue_critical_5pct": [float(r[1]) for r in np.asarray(jo.cvm)],
        "rank": rank_of(scaled),
        "trace_statistics_scaled": scaled,
        "scale_factor": scale,
        "effective_observations": t_obs,
        "small_sample": scale < 0.9,
        "cointegrating_vector": {k: float(v) for k, v in zip(names, vector)},
    }
    return {
        "n": n,
        "alpha": float(alpha),
        "series": names,
        "engle_granger": eg,
        "johansen": johansen,
    }


# -------------------------------------------------------------------------------------- hurst

HURST_MIN_OBS = 256
HURST_MIN_WINDOW = 16
HURST_SCALES = 12
# A log-spaced scale within this relative distance below an integer is that integer: the end
# points 16 and n // 4 are integers that `pow` misses by an ulp either way depending on the libm.
HURST_SCALE_TIE = 1e-12
ANIS_LLOYD_EXACT_MAX = 340


def hurst_scales(n):
    """Block sizes for `n` increments: twelve log-spaced sizes from 16 to n/4 (at least four
    blocks at the largest), as numpy's `logspace` evaluates them, floored after scaling by
    1 + HURST_SCALE_TIE so that a size a rounding error below an integer is that integer on every
    platform, distinct."""
    v = np.logspace(math.log10(HURST_MIN_WINDOW), math.log10(n // 4), HURST_SCALES)
    s = np.floor(v * (1.0 + HURST_SCALE_TIE))
    return [int(v) for v in np.unique(s)]


def expected_rs(n):
    """E[R/S] of `n` iid normal observations: Anis and Lloyd (1976),
    `(n - 1/2)/n * G((n-1)/2) / (sqrt(pi) G(n/2)) * sum_{i=1}^{n-1} sqrt((n-i)/i)`, with Peters'
    (1994) replacement of the gamma ratio by `1 / sqrt(n pi / 2)` above n = 340, where the gamma
    functions overflow in the textbook form."""
    i = np.arange(1, n)
    tail = math.fsum(np.sqrt((n - i) / i))
    if n <= ANIS_LLOYD_EXACT_MAX:
        ratio = math.exp(sp.gammaln((n - 1) / 2.0) - sp.gammaln(n / 2.0)) / math.sqrt(math.pi)
    else:
        ratio = 1.0 / math.sqrt(n * math.pi / 2.0)
    return (n - 0.5) / n * ratio * tail


def _ols_slope(x, y):
    """Slope of y on x (with a constant) and its classical standard error."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    b = float(xc @ (y - y.mean())) / sxx
    resid = (y - y.mean()) - b * xc
    return b, math.sqrt(float(resid @ resid) / (len(x) - 2) / sxx)


def hurst(increments):
    """The Hurst exponent of increments (returns) two ways, over `hurst_scales(n)`:

    rescaled range (Hurst 1951; Mandelbrot and Wallis 1969): each scale cuts the series into
    non-overlapping blocks from the start; a block's R/S is the range of its cumulative
    deviations from its mean over its standard deviation (population); the slope of
    log mean R/S on log scale, less the slope of log E[R/S] (`expected_rs`) at the same scales,
    plus 0.5, is the Anis-Lloyd corrected exponent (Weron 2002). Blocks of zero variance are
    left out.

    detrended fluctuation analysis (Peng et al. 1994): the profile is the cumulative sum of
    the demeaned increments; each block of the profile is detrended by its own least-squares
    line; the fluctuation at a scale is the root of the mean squared residual over all blocks;
    the exponent is the slope of log fluctuation on log scale.

    `std_error` is the classical standard error of each log-log slope. `reading` is
    `persistent` above 0.55, `anti_persistent` below 0.45, else `random_walk`, on the mean of the
    two exponents."""
    a = _finite(increments, "increments", HURST_MIN_OBS)
    if np.ptp(a) == 0:
        raise _degenerate("increments are constant")
    n = len(a)
    scales = hurst_scales(n)

    used, mean_rs, exp_rs = [], [], []
    for s in scales:
        blocks = a[: n // s * s].reshape(-1, s)
        dev = blocks - blocks.mean(axis=1, keepdims=True)
        cum = np.cumsum(dev, axis=1)
        sd = blocks.std(axis=1)
        ok = sd > 0
        if ok.any():
            rs = (cum.max(axis=1) - cum.min(axis=1))[ok] / sd[ok]
            used.append(s)
            mean_rs.append(rs.mean())
            exp_rs.append(expected_rs(s))
    if len(used) < 4:
        raise _degenerate("too few usable scales for R/S")
    lx = np.log(used)
    raw, se = _ols_slope(lx, np.log(mean_rs))
    null, _ = _ols_slope(lx, np.log(exp_rs))
    rescaled = {"hurst": raw - null + 0.5, "raw_slope": raw, "std_error": se, "scales": used}

    profile = np.cumsum(a - a.mean())
    fluct = []
    for s in scales:
        blocks = profile[: n // s * s].reshape(-1, s)
        t = np.arange(s, dtype=float)
        tc = t - t.mean()
        bc = blocks - blocks.mean(axis=1, keepdims=True)
        slope = (bc @ tc) / (tc @ tc)
        resid = bc - slope[:, None] * tc[None, :]
        fluct.append(math.sqrt((resid**2).mean(axis=1).mean()))
    if any(not f > 0 for f in fluct):
        raise _degenerate("the detrended fluctuation is zero at some scale")
    h_dfa, se_dfa = _ols_slope(np.log(scales), np.log(fluct))
    dfa = {"hurst": h_dfa, "std_error": se_dfa, "scales": scales}

    mid = (rescaled["hurst"] + dfa["hurst"]) / 2.0
    reading = "persistent" if mid > 0.55 else "anti_persistent" if mid < 0.45 else "random_walk"
    return {"n": n, "rescaled_range": rescaled, "dfa": dfa, "reading": reading}


# -------------------------------------------------------------------------------------- garch

GARCH_MIN_OBS = 250
GARCH_MAX_HORIZON = 60
GARCH_RECENT = 60


def _numerically_constant(x):
    return np.ptp(x) <= 1e-12 * np.max(np.abs(x))


def garch(returns, model="garch", dist="t", horizon=10, periods_per_year=None):
    """GARCH(1,1) (Bollerslev 1986) or GJR-GARCH(1,1,1) (Glosten, Jagannathan and Runkle 1993)
    with a constant mean, normal or standardised Student-t innovations, by arch on the returns in
    percent. A fit arch does not report as converged is a refusal.

    From the fit, in the units of `returns`: the parameters; persistence
    `alpha + beta + gamma / 2` (the GJR term enters half the time under a symmetric
    distribution); while it is below 1 the half-life of a shock `ln(1/2) / ln(persistence)` and
    the unconditional volatility `sqrt(omega / (1 - persistence))`; the forecast volatility path
    (arch's analytic variance forecast), its cumulative volatility over the horizon (root of the
    summed variances); the last in-sample conditional volatility and the last 60; with
    `periods_per_year`, the annualised ones (times its square root)."""
    from arch import arch_model

    if model not in ("garch", "gjr") or dist not in ("normal", "t"):
        raise _invalid("model is garch or gjr, dist is normal or t")
    if not 1 <= horizon <= GARCH_MAX_HORIZON:
        raise _invalid(f"horizon must be 1 to {GARCH_MAX_HORIZON}")
    r = _finite(returns, "returns", GARCH_MIN_OBS)
    if _numerically_constant(r):
        raise _degenerate("returns are constant")
    gjr = model == "gjr"
    am = arch_model(
        r * 100.0, mean="Constant", vol="GARCH", p=1, o=1 if gjr else 0, q=1,
        dist="t" if dist == "t" else "normal",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    if res.convergence_flag != 0:
        raise Refusal(
            "did_not_converge", "the GARCH fit did not converge: use a longer or cleaner sample"
        )
    p = res.params
    alpha, beta = float(p["alpha[1]"]), float(p["beta[1]"])
    gamma = float(p["gamma[1]"]) if gjr else None
    persistence = alpha + beta + (gamma or 0.0) / 2.0
    omega = float(p["omega"]) / 1e4
    stationary = persistence < 1.0
    var_path = np.asarray(res.forecast(horizon=horizon, reindex=False).variance.iloc[-1]) / 1e4
    vol = np.asarray(res.conditional_volatility) / 100.0
    out = {
        "n": len(r),
        "model": "GJR-GARCH(1,1,1)" if gjr else "GARCH(1,1)",
        "distribution": dist,
        "converged": True,
        "parameters": {
            "mu": float(p["mu"]) / 100.0,
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "nu": float(p["nu"]) if dist == "t" else None,
        },
        "persistence": persistence,
        "stationary_variance": stationary,
        "half_life_periods": math.log(0.5) / math.log(persistence) if stationary else None,
        "unconditional_vol_per_period": math.sqrt(omega / (1.0 - persistence))
        if stationary
        else None,
        "log_likelihood": float(res.loglikelihood),
        "aic": float(res.aic),
        "last_vol_per_period": float(vol[-1]),
        "forecast_vol_per_period": [float(v) for v in np.sqrt(var_path)],
        "forecast_cumulative_vol": math.sqrt(math.fsum(var_path)),
        "recent_conditional_vol": [float(v) for v in vol[-GARCH_RECENT:]],
    }
    if periods_per_year is not None:
        a = math.sqrt(periods_per_year)
        out["periods_per_year"] = periods_per_year
        out["last_vol_annualised"] = float(vol[-1]) * a
        out["forecast_vol_annualised"] = [float(v) * a for v in np.sqrt(var_path)]
    return out


# ------------------------------------------------------------------------------------ regimes

REGIMES_OBS_PER_REGIME = 200
REGIMES_RECENT = 20
REGIMES_SEGMENTS = 40


def markov_regimes(returns, k=2):
    """Hamilton (1989) Markov-switching mean and variance, `k` regimes, by statsmodels'
    `MarkovRegression(trend="c", switching_variance=True)` on the returns in percent
    (`fit(disp=False, maxiter=200)`). Regimes are ordered by variance, calmest first.

    From the fit, in the units of `returns`: per regime its mean, volatility, the probability of
    staying, the expected duration `1 / (1 - stay)` and the share of bars whose most probable
    smoothed regime (Kim 1994 smoother) it is; the transition matrix `[from][to]`; the smoothed
    probabilities of the last 20 bars and of the last bar, its most probable regime; the
    timeline: the runs of equal most-probable regime over the whole series (the last 40, and
    how many there are)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    if k not in (2, 3):
        raise _invalid("k must be 2 or 3 regimes")
    r = _finite(returns, "returns", REGIMES_OBS_PER_REGIME * (k - 1))
    if _numerically_constant(r):
        raise _degenerate("returns are constant")
    mod = MarkovRegression(r * 100.0, k_regimes=k, trend="c", switching_variance=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=False, maxiter=200)
    params = np.asarray(res.params)
    mu = params[k * (k - 1) : k * (k - 1) + k]
    var = params[k * (k - 1) + k :]
    order = [int(i) for i in np.argsort(var, kind="stable")]
    to_from = np.asarray(res.regime_transition)[:, :, 0]
    trans = [[float(to_from[t, f]) for t in order] for f in order]
    smoothed = np.asarray(res.smoothed_marginal_probabilities)[:, order]
    label = smoothed.argmax(axis=1)
    n = len(r)
    runs, start = [], 0
    for i in range(1, n + 1):
        if i == n or label[i] != label[start]:
            runs.append(
                {"regime": int(label[start]), "start_index": start, "end_index": i - 1,
                 "periods": i - start}
            )
            start = i
    stats_ = []
    for i, o in enumerate(order):
        stay = trans[i][i]
        stats_.append(
            {
                "regime": i,
                "mean_per_period": float(mu[o]) / 100.0,
                "vol_per_period": math.sqrt(float(var[o])) / 100.0,
                "stay_probability": stay,
                "expected_duration_periods": 1.0 / (1.0 - stay) if stay < 1.0 else None,
                "share_of_periods": float(np.mean(label == i)),
            }
        )
    return {
        "n": n,
        "regimes": k,
        "converged": bool(res.mle_retvals.get("converged", False)),
        "log_likelihood": float(res.llf),
        "aic": float(res.aic),
        "regime_stats": stats_,
        "transition_matrix": {"values": trans},
        "current_regime": int(label[-1]),
        "current_probabilities": [float(v) for v in smoothed[-1]],
        "recent_probabilities": [[float(v) for v in row] for row in smoothed[-REGIMES_RECENT:]],
        "timeline_total": len(runs),
        "timeline": runs[-REGIMES_SEGMENTS:],
    }


# ---------------------------------------------------------------------------------------- hac

HAC_MIN_OBS = 30


def hac_tstat(returns, lags=None):
    """The mean against zero with the Newey and West (1987) standard error (Bartlett kernel, no
    small-sample correction), by statsmodels' `OLS(r, 1).fit(cov_type="HAC")`; the lag default is
    `floor(4 (n/100)^(2/9))` (Newey and West 1994). The HAC p-value is two-sided normal; the
    naive t is the classical OLS one with its two-sided t(n - 1) p-value. Also the per-period
    Sharpe ratio (mean over the sample standard deviation) and the lag-1 autocorrelation (the
    Pearson correlation of consecutive values)."""
    import statsmodels.api as sm

    r = _finite(returns, "returns", HAC_MIN_OBS)
    n = len(r)
    if np.ptp(r) == 0:
        raise _degenerate("returns are constant")
    used = lags if lags is not None else int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    if not 0 <= used < n // 2:
        raise _invalid(f"lags must be 0 to {n // 2 - 1} for {n} observations")
    ones = np.ones(n)
    hac = sm.OLS(r, ones).fit(cov_type="HAC", cov_kwds={"maxlags": used})
    naive = sm.OLS(r, ones).fit()
    m = _mean(r)
    e = r - m
    sd = math.sqrt(math.fsum(e * e) / (n - 1))
    a, b = r[:-1], r[1:]
    da, db = a - _mean(a), b - _mean(b)
    lag1 = math.fsum(da * db) / math.sqrt(math.fsum(da * da) * math.fsum(db * db))
    t_hac, t_naive = float(hac.tvalues[0]), float(naive.tvalues[0])
    return {
        "n": n,
        "mean": float(hac.params[0]),
        "hac_std_error": float(hac.bse[0]),
        "hac_t": t_hac,
        "hac_p_value": float(2.0 * st.norm.sf(abs(t_hac))),
        "naive_t": t_naive,
        "naive_p_value": float(2.0 * st.t.sf(abs(t_naive), n - 1)),
        "lags": used,
        "per_period_sharpe": m / sd,
        "lag1_autocorrelation": lag1,
    }


# ---------------------------------------------------------------------------------------- pca

PCA_MIN_OBS = 60
PCA_MAX_SERIES = 40
PCA_MAX_COMPONENTS = 5
PCA_SIGN_TOL = 1e-9


def _orient(v):
    """A component's sign: its loadings sum to a positive number; when the sum is within 1e-9
    of zero, the loading of largest magnitude (the first within 1e-9 of it) is positive."""
    s = float(v.sum())
    if abs(s) > PCA_SIGN_TOL:
        return -v if s < 0 else v
    big = np.abs(v).max()
    pivot = int(np.argmax(np.abs(v) >= big - PCA_SIGN_TOL))
    return -v if v[pivot] < 0 else v


def pca_factors(cols, components=3, target=None):
    """Principal components of the correlation matrix (Jolliffe 2002): each series centred and
    divided by its sample standard deviation, the SVD of that panel; a component's share of the
    variance is its squared singular value over their sum. The first component is a common
    factor when its loadings all have one sign. Components are capped at 5 and at the number of
    series, oriented by `_orient`.

    With `target`: OLS of the target on a constant and the scores (panel times loadings) of the
    components; R-squared, the intercept and every coefficient with its classical t. A component
    whose singular value is at rounding level (at most `max(n, k) eps` of the largest) carries no
    information: it gets coefficient 0 and t 0 and the regression is on the others, with the
    residual variance over `n - c - 1`, `c` the components asked for."""
    names = list(cols)
    k = len(names)
    if k > PCA_MAX_SERIES:
        raise _invalid(f"{k} series, at most {PCA_MAX_SERIES}")
    if k < 2:
        raise _too_few_series("returns", k, 2)
    arrays = [_finite(cols[name], f"returns[{name}]", PCA_MIN_OBS) for name in names]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise _invalid("returns: the series differ in length")
    if any(np.ptp(a) == 0 for a in arrays):
        raise _degenerate("a series is constant")
    c = min(components, PCA_MAX_COMPONENTS, k)
    m = np.column_stack(arrays)
    z = (m - m.mean(axis=0)) / m.std(axis=0, ddof=1)
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    ratio = sv**2 / np.sum(sv**2)
    vt = np.array([_orient(row) for row in vt])
    first = vt[0]
    out = {
        "n": n,
        "components": c,
        "series": names,
        "explained_variance_ratio": [float(v) for v in ratio[:c]],
        "cumulative_explained": float(np.sum(ratio[:c])),
        "loadings": {
            f"pc{j + 1}": {name: float(v) for name, v in zip(names, vt[j])} for j in range(c)
        },
        "first_component_is_common_factor": bool(np.all(first > 0) or np.all(first < 0)),
    }
    if target is not None:
        y = np.asarray(target, dtype=float)
        if len(y) < n:
            raise _too_few("target", len(y), n)
        if not np.all(np.isfinite(y)):
            raise _invalid("target holds a NaN or infinite value")
        if len(y) != n:
            raise _invalid("target and panel differ in length")
        if np.ptp(y) == 0:
            raise _degenerate("target is constant")
        null = [bool(sv[j] <= max(n, k) * np.finfo(float).eps * sv[0]) for j in range(c)]
        kept = [j for j in range(c) if not null[j]]
        x = np.column_stack([np.ones(n)] + [z @ vt[j] for j in kept])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        s2 = float(resid @ resid) / (n - c - 1)
        se = np.sqrt(s2 * np.diag(np.linalg.inv(x.T @ x)))
        yc = y - y.mean()
        exposures = {f"pc{j + 1}": {"beta": 0.0, "t": 0.0} for j in range(c)}
        for col, j in enumerate(kept, start=1):
            exposures[f"pc{j + 1}"] = {"beta": float(beta[col]), "t": float(beta[col] / se[col])}
        out["target"] = {
            "r_squared": 1.0 - float(resid @ resid) / float(yc @ yc),
            "alpha_per_period": float(beta[0]),
            "alpha_t": float(beta[0] / se[0]),
            "exposures_per_unit_score": exposures,
        }
    return out


# ---------------------------------------------------------------------------------------- spa

SPA_MIN_OBS = 60
SPA_MAX_STRATEGIES = 200
SPA_MIN_REPS = 200
SPA_MAX_REPS = 5000


def spa_test(cols, benchmark=None, reps=1000, block_size=None, seed=0):
    """Hansen's (2005) test of superior predictive ability and White's (2000) reality check, by
    arch's `SPA` with the stationary bootstrap (Politis and Romano 1994) of mean block
    `block_size` (default `max(2, round(n^(1/3)))`), `reps` resamples and `seed`; the losses are
    the negated returns of the strategies and of the benchmark (zero when none). The best
    strategy is the one of largest mean excess return (the first on a tie); its per-period
    Sharpe ratio of the excess is the mean over the sample standard deviation, None when the
    excess is constant."""
    from arch.bootstrap import SPA

    names = list(cols)
    if not names:
        raise _too_few_series("strategies", 0, 1)
    arrays = [_finite(cols[name], f"strategies[{name}]", SPA_MIN_OBS) for name in names]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise _invalid("strategies: the series differ in length")
    k = len(names)
    if k > SPA_MAX_STRATEGIES:
        raise _invalid(f"{k} strategies, at most {SPA_MAX_STRATEGIES}")
    if benchmark is None:
        bench = np.zeros(n)
    else:
        bench = _finite(benchmark, "benchmark", n)
        if len(bench) != n:
            raise _invalid("benchmark and strategies differ in length")
    if not SPA_MIN_REPS <= reps <= SPA_MAX_REPS:
        raise _invalid(f"reps must be {SPA_MIN_REPS} to {SPA_MAX_REPS}")
    block = block_size if block_size is not None else max(2, int(round(n ** (1.0 / 3.0))))
    if not 1 <= block < n // 2:
        raise _invalid(f"block_size must be 1 to {n // 2 - 1}")
    m = np.column_stack(arrays)
    spa = SPA(-bench, -m, block_size=block, reps=reps, bootstrap="stationary", seed=seed)
    spa.compute()
    excess = m - bench[:, None]
    means = [_mean(excess[:, j]) for j in range(k)]
    best = int(np.argmax(means))
    ex = excess[:, best]
    sd = math.sqrt(math.fsum((ex - means[best]) ** 2) / (n - 1))
    return {
        "n_periods": n,
        "n_strategies": k,
        "best_strategy": names[best],
        "best_mean_excess": means[best],
        "best_per_period_sharpe_excess": means[best] / sd if np.ptp(ex) > 0 else None,
        "p_value_spa_consistent": float(spa.pvalues["consistent"]),
        "p_value_spa_lower": float(spa.pvalues["lower"]),
        "p_value_white_reality_check": float(spa.pvalues["upper"]),
        "bootstrap": {"kind": "stationary", "mean_block": block, "reps": reps, "seed": seed},
    }


# ---------------------------------------------------------------------------------------- pbo

PBO_MIN_SPLITS = 4
PBO_MAX_STRATEGIES = 200
PBO_MAX_COMBINATIONS = 20_000


def _sharpe(x):
    """Per-period Sharpe ratio of each column: mean over sample standard deviation, 0 for a
    column without variation (its ratio is undefined; the convention ranks it at zero)."""
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.ptp(col) > 0:
            out[j] = col.mean() / col.std(ddof=1)
    return out


def pbo_cscv(cols, splits=16):
    """Probability of backtest overfitting by combinatorially symmetric cross-validation
    (Bailey, Borwein, Lopez de Prado and Zhu 2017). The rows (periods) are cut into `splits`
    equal blocks, the oldest rows beyond a whole number of blocks dropped. For every choice of
    half of the blocks as the in-sample set (in lexicographic order) the strategy of best
    in-sample Sharpe ratio (the first on a tie) is ranked by its out-of-sample Sharpe ratio among
    all strategies (1 the worst, ties sharing the mean rank); `w = rank / (k + 1)` and the logit
    `ln(w / (1 - w))`. PBO is the share of logits at or below 0. Also the median logit, the mean
    in- and out-of-sample Sharpe ratio of the chosen strategy and the share of combinations where
    its out-of-sample Sharpe ratio is negative. At least `4 splits` rows; `C(splits, splits/2)`
    at most 20000."""
    if splits < PBO_MIN_SPLITS or splits % 2:
        raise _invalid(f"splits must be an even number, at least {PBO_MIN_SPLITS}")
    names = list(cols)
    k = len(names)
    if k < 2:
        raise _too_few_series("strategies", k, 2)
    arrays = [_finite(cols[name], f"strategies[{name}]", splits * 4) for name in names]
    length = len(arrays[0])
    if any(len(a) != length for a in arrays):
        raise _invalid("strategies: the series differ in length")
    if k > PBO_MAX_STRATEGIES:
        raise _invalid(f"{k} strategies, at most {PBO_MAX_STRATEGIES}")
    n_comb = math.comb(splits, splits // 2)
    if n_comb > PBO_MAX_COMBINATIONS:
        raise _invalid(
            f"splits={splits} makes {n_comb} combinations, at most {PBO_MAX_COMBINATIONS}"
        )
    per = length // splits
    dropped = length - per * splits
    m = np.column_stack(arrays)[dropped:]
    blocks = [m[b * per : (b + 1) * per] for b in range(splits)]
    logits, is_best, oos_best = [], [], []
    for ins in itertools.combinations(range(splits), splits // 2):
        outs = [b for b in range(splits) if b not in ins]
        sr_is = _sharpe(np.vstack([blocks[b] for b in ins]))
        sr_oos = _sharpe(np.vstack([blocks[b] for b in outs]))
        best = int(np.argmax(sr_is))
        target = sr_oos[best]
        rank = 1.0 + np.sum(sr_oos < target) + 0.5 * (np.sum(sr_oos == target) - 1.0)
        w = rank / (k + 1.0)
        logits.append(math.log(w / (1.0 - w)))
        is_best.append(sr_is[best])
        oos_best.append(target)
    logits = np.array(logits)
    return {
        "n_periods": per * splits,
        "n_strategies": k,
        "splits": splits,
        "combinations": len(logits),
        "pbo": float(np.mean(logits <= 0.0)),
        "median_logit": float(np.median(logits)),
        "mean_in_sample_sharpe_of_best": math.fsum(is_best) / len(is_best),
        "mean_out_of_sample_sharpe_of_best": math.fsum(oos_best) / len(oos_best),
        "prob_out_of_sample_loss_of_best": float(np.mean(np.array(oos_best) < 0.0)),
    }


# ------------------------------------------------------------------------------------- events

EVENTS_MAX = 100
EVENTS_MIN_ESTIMATION = 60
EVENTS_MAX_WINDOW = 20
EVENTS_DEGENERATE_SIGMA = 1e-10


def event_study(dates, asset, market, events, estimation=120, gap=5, pre=1, post=1):
    """Event study by the market model (MacKinlay 1997; Campbell, Lo and MacKinlay 1997, ch. 4).
    An event dated off the calendar moves to the next bar. For each event, OLS of the asset on
    the market over the `estimation` bars ending `gap` bars before the event window
    `[-pre, +post]`; abnormal returns over the window, their sum CAR, the residual standard
    deviation (n - 2) and `t = CAR / (sigma sqrt(L))`, L the window length. A fit whose residual
    sigma is at most 1e-10 of the asset's own standard deviation over the estimation window has
    no residual noise: its t is undefined (NaN, `degenerate_fit`). An event without a full
    estimation and event window, after the last bar, or with a constant market over the
    estimation window is listed as skipped. Over the events: CAAR, the mean CAR, and with two or
    more events whose CARs differ the cross-sectional t, its two-sided t(m - 1) p-value and the
    share of positive CARs."""
    if not 1 <= len(events) <= EVENTS_MAX:
        raise _invalid(f"give 1 to {EVENTS_MAX} event dates")
    if not (0 <= pre <= EVENTS_MAX_WINDOW and 0 <= post <= EVENTS_MAX_WINDOW):
        raise _invalid(f"pre and post are 0 to {EVENTS_MAX_WINDOW} periods")
    if estimation < EVENTS_MIN_ESTIMATION or gap < 0:
        raise _invalid(f"estimation is at least {EVENTS_MIN_ESTIMATION} periods, gap not negative")
    n = len(dates)
    r = _finite(asset, "asset", 2)
    mk = _finite(market, "market", 2)
    if not len(r) == len(mk) == n:
        raise _invalid("dates, asset and market differ in length")
    if any(a >= b for a, b in zip(dates, dates[1:])):
        raise _invalid("dates are not strictly increasing")
    length = pre + post + 1
    rows, skipped, cars = [], [], []
    for ev in events:
        idx = bisect.bisect_left(dates, ev)
        if idx >= n:
            skipped.append({"event": ev.isoformat(), "reason": "after the last bar"})
            continue
        lo, hi = idx - pre, idx + post
        est_hi = lo - gap
        est_lo = est_hi - estimation
        if est_lo < 0 or hi >= n:
            reason = "no full estimation and event window"
            skipped.append({"event": ev.isoformat(), "reason": reason})
            continue
        x, y = mk[est_lo:est_hi], r[est_lo:est_hi]
        if np.ptp(x) == 0:
            skipped.append({"event": ev.isoformat(), "reason": "market constant in estimation"})
            continue
        design = np.column_stack([np.ones(len(x)), x])
        (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - alpha - beta * x
        sigma = math.sqrt(float(resid @ resid) / (len(x) - 2))
        degenerate = not sigma > EVENTS_DEGENERATE_SIGMA * float(np.std(y, ddof=1))
        ar = r[lo : hi + 1] - alpha - beta * mk[lo : hi + 1]
        car = math.fsum(ar)
        cars.append(car)
        rows.append(
            {
                "event": ev.isoformat(),
                "event_bar": dates[idx].isoformat(),
                "alpha": float(alpha),
                "beta": float(beta),
                "residual_sigma": sigma,
                "abnormal_returns": [float(v) for v in ar],
                "car": car,
                "t": math.nan if degenerate else car / (sigma * math.sqrt(length)),
                "degenerate_fit": degenerate,
            }
        )
    if not rows:
        raise _invalid(f"no event had a full window: {skipped!r}")
    m = len(cars)
    caar = math.fsum(cars) / m
    out = {
        "n_bars": n,
        "n_events": m,
        "event_window": {"from": -pre, "to": post, "periods": length},
        "estimation": {"periods": estimation, "gap": gap},
        "caar": caar,
        "events": rows,
        "skipped": skipped,
        "caar_t": None,
    }
    sd = float(np.std(cars, ddof=1)) if m >= 2 else 0.0
    if m >= 2 and sd > 0:
        t = caar / (sd / math.sqrt(m))
        out["caar_t"] = t
        out["caar_p_value"] = float(2.0 * st.t.sf(abs(t), m - 1))
        out["positive_share"] = float(np.mean(np.array(cars) > 0))
    return out


# ------------------------------------------------------------------------------------- impact

IMPACT_MIN_DAYS = 20
IMPACT_BEYOND_MODEL = 0.25


def corwin_schultz(high, low):
    """Corwin and Schultz (2012) high-low spread, as a fraction of price: for each pair of
    consecutive days `beta = sum of ln(H/L)^2` over the two days, `gamma = ln(H2/L2)^2` of the
    two-day high and low, `alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) -
    sqrt(gamma / (3 - 2 sqrt 2))`, `S = 2 (e^alpha - 1) / (1 + e^alpha)`, a negative two-day
    estimate set to 0 (the paper's choice); the mean over all pairs. No overnight adjustment."""
    h, lo = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    k = 3.0 - 2.0 * math.sqrt(2.0)
    hl = np.log(h / lo) ** 2
    beta = hl[:-1] + hl[1:]
    gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(lo[:-1], lo[1:])) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return math.fsum(np.maximum(s, 0.0)) / len(s)


def roll(close):
    """Roll (1984): `2 sqrt(-cov)` of consecutive price changes, over the mean price; None when
    the first-order sample autocovariance (n - 1) of the changes is not negative."""
    c = np.asarray(close, dtype=float)
    d = np.diff(c)
    a, b = d[:-1], d[1:]
    cov = math.fsum((a - _mean(a)) * (b - _mean(b))) / (len(a) - 1)
    return 2.0 * math.sqrt(-cov) / _mean(c) if cov < 0 else None


def quantity_for_notional(notional, last_close):
    """The quantity a notional buys at the last close."""
    if not (notional > 0 and last_close > 0):
        raise _invalid("a notional and a last close must both be positive")
    return notional / last_close


def impact_estimate(
    high, low, close, volume, quantity, spread_model="corwin_schultz", coefficient=1.0
):
    """One-way cost of trading `quantity` from daily bars: half the estimated spread plus the
    square-root impact `coefficient * sigma * sqrt(quantity / ADV)` (Almgren, Thum, Hauptmann
    and Li 2005; Toth et al. 2011), sigma the sample standard deviation of daily close-to-close
    log returns, ADV the mean volume. In basis points of the last close, per unit and in total;
    within the model's range while the participation `quantity / ADV` is at most 0.25."""
    c = _finite(close, "close", IMPACT_MIN_DAYS)
    h = _finite(high, "high", IMPACT_MIN_DAYS)
    lo = _finite(low, "low", IMPACT_MIN_DAYS)
    v = _finite(volume, "volume", IMPACT_MIN_DAYS)
    if not len(c) == len(h) == len(lo) == len(v):
        raise _invalid("high, low, close and volume differ in length")
    if np.any(c <= 0) or np.any(h <= 0) or np.any(lo <= 0) or np.any(h < lo):
        raise _invalid("prices must be positive with high not below low")
    if spread_model not in ("corwin_schultz", "roll"):
        raise _invalid("spread_model is corwin_schultz or roll")
    if not quantity > 0:
        raise _invalid("quantity must be positive")
    adv = _mean(v)
    if adv <= 0:
        raise _degenerate("no volume in these bars (an index carries none): use its futures")
    lr = np.diff(np.log(c))
    sigma = math.sqrt(math.fsum((lr - _mean(lr)) ** 2) / (len(lr) - 1))
    part = quantity / adv
    if spread_model == "roll":
        spread = roll(c)
        if spread is None:
            raise _degenerate(
                "Roll's estimate needs negative serial covariance of price changes; these bars "
                "have none: use corwin_schultz"
            )
    else:
        spread = corwin_schultz(h, lo)
    last = float(c[-1])
    half = spread / 2.0 * 1e4
    impact = coefficient * sigma * math.sqrt(part) * 1e4
    cost = half + impact
    return {
        "days": len(c),
        "inputs": {
            "quantity": float(quantity),
            "average_daily_volume": adv,
            "participation": part,
            "daily_sigma": sigma,
            "last_close": last,
            "spread_model": spread_model,
            "estimated_spread_bps": spread * 1e4,
            "impact_coefficient": float(coefficient),
        },
        "half_spread_bps": half,
        "impact_bps": impact,
        "one_way_cost_bps": cost,
        "one_way_cost_per_unit": last * cost / 1e4,
        "one_way_cost_total": quantity * last * cost / 1e4,
        "within_model_range": part <= IMPACT_BEYOND_MODEL,
    }


# ------------------------------------------------------------------------------------ surface

SURFACE_MAX_PARAMS = 3
SURFACE_MIN_CELLS = 9
SURFACE_MAX_CELLS = 2000
PLATEAU_FRACTION = 0.8
PLATEAU_SHARE = 0.7
SPIKE_SHARE = 0.3
SMOOTH_MIN = 0.8


def sensitivity_surface(names, cells):
    """How a metric (higher is better) moves across a grid of parameter values, by the
    definitions of raptorbt's statistics skill: each parameter's axis is its sorted distinct
    values; two cells are neighbours when they differ by one step along exactly one axis.

    smoothness: 1 minus the mean absolute metric difference over neighbouring pairs, divided by
    the metric's range. best: the first maximum in input order, its neighbours' mean and the
    plateau share, the share of its neighbours within a fifth of the range below it (0 when it
    has none). robust pick: the cell of largest `(metric + neighbour mean * d) / (1 + d)`, d its
    number of neighbours, among cells with a neighbour (the first on a tie). verdict: `plateau`
    when the share is at least 0.7 and smoothness at least 0.8, else `spike` when the share is at
    most 0.3, else `mixed`. cells: every cell in the lexicographic order of its axis positions,
    with its neighbours' mean and its lift over it; matrix (two parameters): the metric by the
    first axis' value (rows) and the second's (columns), None where not run."""
    k = len(names)
    if not 1 <= k <= SURFACE_MAX_PARAMS or len(set(names)) != k:
        raise _invalid(f"give 1 to {SURFACE_MAX_PARAMS} distinct parameter names")
    if not SURFACE_MIN_CELLS <= len(cells) <= SURFACE_MAX_CELLS:
        raise _invalid(
            f"{len(cells)} cells, a surface needs {SURFACE_MIN_CELLS} to {SURFACE_MAX_CELLS}"
        )
    for vals, metric in cells:
        if len(vals) != k or not math.isfinite(metric) or not all(math.isfinite(v) for v in vals):
            raise _invalid("a cell lacks a parameter or holds a non-finite number")
    axes = []
    for i in range(k):
        a = sorted({float(vals[i]) for vals, _ in cells})
        if len(a) < 2:
            raise _invalid(f"parameter '{names[i]}' has one value: nothing to vary")
        axes.append(a)
    grid, order = {}, []
    for vals, metric in cells:
        key = tuple(axes[i].index(float(vals[i])) for i in range(k))
        if key in grid:
            shown = ", ".join(f"'{n}': {v!r}" for n, v in zip(names, vals))
            raise _invalid(f"duplicate cell {{{shown}}}")
        grid[key] = float(metric)
        order.append(key)
    full = math.prod(len(a) for a in axes)
    lo, hi = min(grid.values()), max(grid.values())
    spread = hi - lo
    if spread == 0:
        raise _invalid("the metric is the same in every cell")

    def nbs(key):
        out = []
        for d in range(k):
            for step in (-1, 1):
                nb = key[:d] + (key[d] + step,) + key[d + 1 :]
                if nb in grid:
                    out.append(nb)
        return out

    diffs = [abs(grid[a] - grid[b]) for a in order for b in nbs(a) if a < b]
    if not diffs:
        raise _invalid("no two cells are neighbours: the grid has holes everywhere")
    smoothness = 1.0 - math.fsum(diffs) / len(diffs) / spread

    def nb_mean(key):
        v = [grid[b] for b in nbs(key)]
        return math.fsum(v) / len(v) if v else None

    best = order[0]
    for key in order:
        if grid[key] > grid[best]:
            best = key
    floor = grid[best] - (1.0 - PLATEAU_FRACTION) * spread
    best_nbs = nbs(best)
    share = sum(grid[b] >= floor for b in best_nbs) / len(best_nbs) if best_nbs else 0.0
    robust, robust_score = None, None
    for key in order:
        nm = nb_mean(key)
        if nm is None:
            continue
        d = len(nbs(key))
        score = (grid[key] + nm * d) / (1.0 + d)
        if robust is None or score > robust_score:
            robust, robust_score = key, score
    if share >= PLATEAU_SHARE and smoothness >= SMOOTH_MIN:
        verdict = "plateau"
    elif share <= SPIKE_SHARE:
        verdict = "spike"
    else:
        verdict = "mixed"

    def params(key):
        return {names[i]: axes[i][key[i]] for i in range(k)}

    rows = []
    for key in sorted(grid):
        nm = nb_mean(key)
        rows.append(
            {
                "params": params(key),
                "metric": grid[key],
                "neighbour_mean": nm,
                "neighbours": len(nbs(key)),
                "lift_over_neighbours": None if nm is None else grid[key] - nm,
            }
        )
    out = {
        "parameters": list(names),
        "axes": {names[i]: axes[i] for i in range(k)},
        "n_cells": len(grid),
        "grid_cells": full,
        "coverage": len(grid) / full,
        "metric_min": lo,
        "metric_max": hi,
        "smoothness": smoothness,
        "best": {
            "params": params(best),
            "metric": grid[best],
            "neighbour_mean": nb_mean(best),
            "neighbours": len(best_nbs),
            "plateau_share": share,
        },
        "robust_pick": {
            "params": params(robust),
            "metric": grid[robust],
            "neighbour_mean": nb_mean(robust),
        },
        "verdict": verdict,
        "cells": rows,
    }
    if k == 2:
        out["matrix"] = {
            "rows": names[0],
            "columns": names[1],
            "values": [
                [grid.get((i, j)) for j in range(len(axes[1]))] for i in range(len(axes[0]))
            ]
        }
    return out


# -------------------------------------------------------------------------------- bar audit

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
AUDIT_RULES = {
    "outlier_z": 8.0,
    "mad_scale": 1.4826,
    "listed": 5,
    "daily_gap_days": 4.0,
    "intraday_gap_bars": 3.0,
    "split_factors": (1.5, 2.0, 3.0, 4.0, 5.0, 10.0),
    "split_tolerance": 0.03,
    "min_bars": 5,
    "dead_bar_share_max": 0.5,
}


def _at(ts):
    return dt.datetime.fromtimestamp(math.floor(ts), IST).strftime("%Y-%m-%d %H:%M")


def audit_bars(t, o, h, low, c, v, interval_seconds=86400, rules=AUDIT_RULES):
    """Checks of OHLCV bars, by the rules raptorbt's statistics skill states (thresholds in
    `AUDIT_RULES`, times written in IST). Impossible bars: a non-finite or non-positive price,
    a high below the open or close, a low above them. Return outliers: close-to-close log returns
    over the bars that are not impossible, whose robust z (distance from the median over 1.4826
    MAD; Hampel 1974, Rousseeuw and Croux 1993) is at least 8; the five largest by |z| (the
    earlier first on a tie), as simple returns in percent. Suspected unadjusted splits (daily
    bars): a close ratio within 3% of a split factor or of its inverse. Gaps: a daily stretch of
    more than 4 days without a bar; an intraday hole inside one IST day wider than 3 bars (its
    missing bars `floor(dt / interval) - 1`); the first five. Duplicate timestamps and backward
    steps are counted from consecutive differences. Not clean when any of these is found, or when
    more than half of the bars are flat with no volume."""
    R = rules
    t, o, h, low, c, v = (np.asarray(a, dtype=float) for a in (t, o, h, low, c, v))
    n = len(t)
    if any(len(a) != n for a in (o, h, low, c, v)):
        raise _invalid("the columns of the bars differ in length")
    if n < R["min_bars"]:
        raise _too_few("bars to audit", n, R["min_bars"])
    if interval_seconds <= 0:
        raise _invalid("interval_seconds must be positive")
    if not np.all(np.isfinite(t)):
        raise _invalid("t holds a NaN or infinite value")
    daily = interval_seconds >= 86400
    dts = np.diff(t)
    listed = R["listed"]
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
        bad = (
            ~finite
            | (np.minimum.reduce([o, h, low, c]) <= 0)
            | (h < np.maximum(o, c))
            | (low > np.minimum(o, c))
        )
    flat = (o == h) & (h == low) & (low == c)
    ok_idx = np.flatnonzero(~bad)
    ret = np.diff(np.log(c[ok_idx])) if len(ok_idx) > 1 else np.array([])
    ret_at = ok_idx[1:]
    outliers = []
    if len(ret) >= R["min_bars"]:
        med = float(np.median(ret))
        mad = float(np.median(np.abs(ret - med))) * R["mad_scale"]
        if mad > 0:
            z = (ret - med) / mad
            by_size = sorted(range(len(z)), key=lambda i: (-abs(z[i]), i))
            for i in by_size[:listed]:
                if abs(z[i]) >= R["outlier_z"]:
                    outliers.append(
                        {"at": _at(t[ret_at[i]]), "return_pct": float(np.expm1(ret[i]) * 100.0)}
                    )
    splits = []
    if daily:
        for i, lr in enumerate(ret):
            ratio = math.exp(lr)
            for f in R["split_factors"]:
                if (
                    abs(ratio - f) / f < R["split_tolerance"]
                    or abs(ratio - 1.0 / f) * f < R["split_tolerance"]
                ):
                    splits.append({"at": _at(t[ret_at[i]]), "close_ratio": ratio})
                    break
    gaps = []
    if daily:
        for i in np.flatnonzero(dts > R["daily_gap_days"] * 86400)[:listed]:
            gaps.append({"from": _at(t[i]), "to": _at(t[i + 1]), "days": float(dts[i] / 86400)})
    else:
        day = np.floor((np.floor(t) + 19800) / 86400)
        hole = (day[:-1] == day[1:]) & (dts > R["intraday_gap_bars"] * interval_seconds)
        for i in np.flatnonzero(hole)[:listed]:
            gaps.append(
                {
                    "from": _at(t[i]),
                    "to": _at(t[i + 1]),
                    "missing_bars": int(math.floor(dts[i] / interval_seconds)) - 1,
                }
            )
    duplicates = int(np.sum(dts == 0))
    backward = int(np.sum(dts < 0))
    dead_share = float(np.sum(flat & (v == 0))) / n
    clean = not (
        duplicates or backward or bad.any() or outliers or splits or gaps
        or dead_share > R["dead_bar_share_max"]
    )
    return {
        "bars": n,
        "first": _at(t[0]),
        "last": _at(t[-1]),
        "findings": {
            "duplicate_timestamps": duplicates,
            "out_of_order": backward,
            "impossible_ohlc": {
                "count": int(bad.sum()),
                "first": [_at(t[i]) for i in np.flatnonzero(bad)[:listed]],
            },
            "zero_volume_bars": int(np.sum(v == 0)),
            "flat_bars": int(flat.sum()),
            "flat_zero_volume_bars": int(np.sum(flat & (v == 0))),
            "return_outliers": {"robust_z_over": R["outlier_z"], "worst": outliers},
            "possible_unadjusted_splits": splits[:listed],
            "gaps": gaps,
        },
        "clean": clean,
    }
