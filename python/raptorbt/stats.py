"""Statistics for research: tests and fits over arrays of numbers, in one place.

Every function takes arrays (or a name-to-array mapping) and returns a dict at full precision:
NaN and infinity are what the number is, nothing is rounded, and a caller that shows the result
rounds it. Each result carries the sample size and the ``assumptions`` of the test it ran. A
function that cannot answer from what it got raises a :class:`raptorbt.errors.RaptorBTError`
(also a ``ValueError``) and returns no number: ``insufficient_observations`` (too short),
``degenerate_series`` (constant, or exact combinations of one another) or ``did_not_converge``
(a fit that did not settle), and ``invalid_input`` for an argument out of range.

The module is pure: no I/O, no clock, no randomness the caller does not seed. It is importable
from a user algorithm.

**Causal or not.** A test or fit over a window describes that window; it is not a signal bar by
bar. Two results say more than that. ``markov_regimes`` returns *smoothed* probabilities
(``recent_probabilities``, ``smoothed_probabilities``), which are conditioned on the whole
sample: the label of an earlier bar uses returns after it, so it is not causal and never a
signal. The *filtered* probabilities (``recent_filtered_probabilities``,
``filtered_probabilities``) use only the returns up to each bar and are causal. ``garch``'s
``recent_conditional_vol`` is causal (each value uses the bars before it) but its parameters are
fitted on the whole sample.

>>> import numpy as np
>>> from raptorbt import stats
>>> r = stats.hac_tstat(np.array([0.01, -0.02, 0.015, 0.0, 0.01] * 12))
>>> r["n"], r["lags"]
(60, 3)
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from raptorbt import _raptorbt, errors

__all__ = [
    "audit_bars",
    "cointegration",
    "event_study",
    "garch",
    "hac_tstat",
    "hurst",
    "impact_estimate",
    "markov_regimes",
    "pbo_cscv",
    "pca_factors",
    "predictive_test",
    "sensitivity_surface",
    "spa_test",
    "stationarity",
]

Series = Sequence[float] | np.ndarray
Columns = Mapping[str, Series]

_SKILL = "statistics"

# Keyword defaults of the statistics functions, as explicit constants.
_ALPHA = 0.05  # significance level the tests are read at
_GARCH_MODEL = "garch"  # default volatility model
_GARCH_DIST = "t"  # default innovation distribution
_FORECAST_HORIZON = 10  # bars forecast ahead
_REGIMES = 2  # default number of hidden-Markov regimes
_PBO_SPLITS = 16  # default number of splits for the probability of backtest overfitting
_PCA_COMPONENTS = 3  # default number of principal components
_EVENT_ESTIMATION = 120  # bars in an event study's estimation window
_EVENT_GAP = 5  # bars between the estimation window and the event
_SPREAD_MODEL = "corwin_schultz"  # default bid-ask spread estimator
# Thresholds of `audit_bars`, its keyword defaults.
_OUTLIER_Z = 8.0
_MAD_SCALE = 1.4826
_LISTED = 5
_DAILY_GAP_DAYS = 4.0
_INTRADAY_GAP_BARS = 3.0
_SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0)
_SPLIT_TOLERANCE = 0.03
_MIN_BARS = 5
_DEAD_BAR_SHARE_MAX = 0.5  # a judgment: review it
_SPA_REPS = 1000  # bootstrap repetitions of the superior-predictive-ability test
# Hurst verdict cut-offs: the mean of the two estimates above / below which a series reads as
# persistent / anti-persistent.
_PERSISTENT_ABOVE = 0.55
_ANTI_PERSISTENT_BELOW = 0.45

# Measured: independent driftless random walks, 400 samples per cell, one lagged difference, a
# nominal 5%, rank above 0 (unscaled -> scaled). Its source is the statistics skill's table.
_JOHANSEN_CALIBRATION = (
    "independent driftless random walks, rank > 0 on (unscaled -> scaled): 2 series at 60 / 250 / "
    "500 observations 15% -> 13%, 14% -> 14%, 14% -> 14%; 6 series at 60 / 120 / 250 43% -> 2%, "
    "20% -> 5%, 11% -> 7%; 12 series at 60 / 120 / 250 / 500 100% -> 0.3%, 80% -> 1%, 32% -> 2%, "
    "15% -> 3% (5% nominal)"
)

_DAY_NS = 86_400 * 10**9

_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


# ------------------------------------------------------------------------------ input checks


def _refuse(operation: str, message: str, parameter: str | None = None) -> errors.RaptorBTError:
    return errors.make(
        "invalid_input", message, operation=operation, parameter=parameter, skill=_SKILL
    )


def _series(x: Any, name: str, operation: str) -> np.ndarray:
    """A 1-d float64 array; its length and finiteness are checked by the statistic."""
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        raise _refuse(operation, f"{name} must be numbers", name) from None
    if a.ndim != 1:
        raise _refuse(operation, f"{name} must be one series", name)
    return a


def _columns(cols: Any, name: str, operation: str) -> tuple[list[str], list[np.ndarray]]:
    if not isinstance(cols, Mapping):
        raise _refuse(operation, f"{name} must map each name to a series", name)
    names = [str(k) for k in cols]
    return names, [_series(v, f"{name}[{k}]", operation) for k, v in cols.items()]


def _int(value: Any, name: str, operation: str) -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise _refuse(operation, f"{name} must be a whole number", name) from None


def _day(d: Any, name: str, operation: str) -> int:
    try:
        return d.toordinal() - _EPOCH_ORDINAL
    except AttributeError:
        raise _refuse(operation, f"{name} must be dates", name) from None


# ----------------------------------------------------------------------------- stationarity

_MEANING = {
    "stationary": "ADF rejects a unit root and KPSS does not reject stationarity",
    "unit_root": "ADF cannot reject a unit root and KPSS rejects stationarity: difference it",
    "contradictory": "both reject: typically a trend or break, not a clean unit root",
    "inconclusive": "neither rejects: the sample is too short or noisy to tell",
}


def stationarity(x: Series, *, alpha: float = _ALPHA) -> dict[str, Any]:
    """Is a series stationary: the augmented Dickey-Fuller and the KPSS test, read together.

    They test opposite nulls (ADF: a unit root; KPSS: stationarity), so one alone is weak
    evidence. ``x`` is at least 50 observations, finite and not constant. The ADF has a constant
    and the lag that minimises the AIC; the KPSS is the level test with Hobijn-Franses-Ooms
    lags.

    ``reading`` is ``stationary``, ``unit_root``, ``contradictory`` or ``inconclusive`` from the
    two decisions at ``alpha``. ``kpss.p_value`` comes from a table and is clipped to
    [0.01, 0.10]; ``kpss.p_value_clipped`` says when it sits on a bound. Not causal: the result
    describes the whole sample given.
    """
    op = "raptorbt.stats.stationarity"
    out = _raptorbt.stats_stationarity(_series(x, "series", op), alpha)
    out["meaning"] = _MEANING[out["reading"]]
    out["assumptions"] = [
        "ADF: constant included, lags chosen by AIC, null is a unit root",
        "KPSS: level stationarity (constant), Newey-West lags, null is stationarity",
        "no structural break; a break biases both toward a unit root",
    ]
    return out


# ---------------------------------------------------------------------------- cointegration


def cointegration(cols: Columns, *, alpha: float = _ALPHA) -> dict[str, Any]:
    """Do series share a long-run equilibrium: Engle-Granger and Johansen.

    ``cols`` maps names to level series (prices, or log prices) of one length, at least 60
    observations, two to twelve series; the first is the dependent series of the hedge ratio, and
    the Engle-Granger answer depends on which one that is. Engle-Granger regresses the first on a
    constant and the others and tests the residual (the spread) for a unit root; Johansen reads
    the rank of the whole system (constant in the relation, one lagged difference, 5% trace).

    ``johansen["rank"]`` is the rank to read: the 5% trace test on statistics multiplied by
    Reinsel and Ahn's ``(T - q k) / T`` (``trace_statistics_scaled``, ``scale_factor``; ``T``
    the ``effective_observations``, ``q`` the number of series, ``k = 2``). The unscaled
    ``rank_at_5pct`` and ``trace_statistics`` are kept as the asymptotic test gives them, and
    over-reject when ``q`` is large against ``T``: independent random walks gave a rank above 0
    on 100% of samples at twelve series and 60 observations (scaled: 0.3%). ``small_sample`` is
    true when the correction removes over a tenth of the statistic (``T < 20 q``); then the test
    also has little power. ``johansen["calibration"]`` and ``["warnings"]`` say so in words, and
    the measured false-positive rates are in the statistics skill.

    From seven series the Engle-Granger ``p_value`` is ``None`` (MacKinnon's 1994 table stops at
    six); ``critical_values`` are the 2010 ones and ``cointegrated`` falls back to the statistic
    against the critical value when ``alpha`` is 0.01, 0.05 or 0.10, else ``None``. A statistic of
    ``-inf`` means the series are (almost) one line. Panels whose moment matrices are singular to
    working precision raise ``degenerate_series``. Hedge ratio and half-life are in-sample and
    not causal: they describe the sample given and drift out of it.
    """
    op = "raptorbt.stats.cointegration"
    names, arrays = _columns(cols, "series", op)
    out = _raptorbt.stats_cointegration(names, arrays, alpha)
    jo = out["johansen"]
    jo["calibration"] = _JOHANSEN_CALIBRATION
    jo["warnings"] = []
    if jo["small_sample"]:
        jo["warnings"].append(
            f"small sample: {jo['effective_observations']} observations for {len(arrays)} series "
            f"(Reinsel-Ahn factor {jo['scale_factor']:.2f}): the asymptotic critical values do "
            "not apply. Read `rank`, not `rank_at_5pct` (independent random walks showed rank "
            "above 0 on 43% at six series and 60 observations, on 100% at twelve); the scaled "
            "test has little power here, so rank 0 is not evidence of no relation either"
        )
    elif jo["rank"] != jo["rank_at_5pct"]:
        jo["warnings"].append(
            "the scaled rank differs from the unscaled one: the unscaled test over-rejects at "
            "this number of series and observations; read `rank`"
        )
    if jo["rank"] > 0:
        jo["warnings"].append(
            "a rank above 0 is a lead, not a finding: on independent driftless random walks the "
            "test reports one on about 13% of samples at two series however long the sample "
            "(5% nominal). Confirm with Engle-Granger and out of sample"
        )
    out["johansen"]["vector_note"] = (
        "first eigenvector, normalised to the first series; meaningful when the rank is at "
        "least 1"
    )
    out["assumptions"] = [
        "levels of the series, linear relation with a constant (no trend)",
        "Engle-Granger: OLS of the first series on the others, ADF on the residual with "
        "MacKinnon p-values, AIC lags; the answer depends on which series is dependent",
        "Johansen: constant in the cointegrating relation, one lagged difference, 5% trace test",
        "hedge ratio and half-life are in-sample; they drift out of sample",
    ]
    return out


# --------------------------------------------------------------------------------- memory


def hurst(increments: Series) -> dict[str, Any]:
    """Does a series remember: the Hurst exponent of its increments, two ways.

    ``increments`` are returns (log price changes), not prices; at least 256, not constant. The
    rescaled-range estimate (``rescaled_range``) has the expected R/S of an iid series removed
    (Anis-Lloyd with Peters' correction); the detrended fluctuation analysis (``dfa``) fits linear
    trends in blocks. Above 0.5 the increments persist (trend), below 0.5 they reverse.

    ``reading`` is ``persistent``, ``anti_persistent`` or ``random_walk`` from the mean of the two
    estimates at full precision. The ``std_error`` values are of the log-log fit.
    Not causal: a statement about the whole sample.
    """
    op = "raptorbt.stats.hurst"
    out = _raptorbt.stats_hurst(_series(increments, "increments", op))
    rs, dfa = out["rescaled_range"], out["dfa"]
    rs["correction"] = "Anis-Lloyd: the slope an iid series shows at the same scales is removed"
    dfa["detrending"] = "linear within each block"
    mid = (rs["hurst"] + dfa["hurst"]) / 2.0
    out["reading"] = "random_walk"
    if mid > _PERSISTENT_ABOVE:
        out["reading"] = "persistent"
    elif mid < _ANTI_PERSISTENT_BELOW:
        out["reading"] = "anti_persistent"
    out["reading_rule"] = (
        "mean of the two estimates: above 0.55 persistent, below 0.45 anti-persistent, else "
        "indistinguishable from a random walk (a heuristic band; the standard errors show how "
        "wide the estimate is)"
    )
    out["assumptions"] = [
        "applied to returns (increments), not prices",
        "a power law over the scales shown; the standard errors are of the log-log fit and "
        "understate the estimator's own variance",
        "heavy tails and volatility clustering bias both estimates",
    ]
    return out


# ------------------------------------------------------------------------------ volatility


def garch(
    returns: Series,
    *,
    model: str = _GARCH_MODEL,
    dist: str = _GARCH_DIST,
    horizon: int = _FORECAST_HORIZON,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Conditional volatility: GARCH(1,1) or GJR-GARCH(1,1,1) by maximum likelihood.

    ``returns`` are simple or log returns as fractions, at least 250, not constant. ``model`` is
    ``garch`` or ``gjr``; ``dist`` is ``normal`` or ``t``; ``horizon`` is 1 to 60 bars. The
    result carries the parameters, the persistence, the half-life of a shock, the forecast
    volatility path, the last 60 conditional volatilities and, with ``periods_per_year``, the
    annualised ones. Parameters are in the units of ``returns``; variances are per bar.

    Also returned: ``std_errors`` and ``covariance_robust`` (robust sandwich, ``None`` if
    the Hessian cannot be inverted; ``parameter_names`` is their order), and ``rescale`` (the
    power of ten the percent series was multiplied by to fit; returns of scale 1e-7 now fit
    instead of raising ``did_not_converge``). ``log_likelihood`` is of the series in percent.
    ``recent_conditional_vol`` is causal; the parameters are fitted on the whole sample.

    A series that varies by no more than 1e-12 of its level (returns constant to a few ulps) is
    refused with ``degenerate_series``: the fit would be to the rounding noise. Where the fit is
    not to be read at face value the result says so, in ``warnings`` and in flags:
    ``integrated`` is true when persistence is within 1e-6 of 1 or above (a volatility level
    shift or a break in the sample looks like this): ``stationary_variance`` is then false and
    ``half_life_periods`` and ``unconditional_vol_per_period`` are ``None``, not a huge number;
    ``shape_at_bound`` is true when the t shape sits at its bound (2.05 or 500), where it is
    the bound that was estimated; ``std_errors`` is ``None`` when the Hessian cannot be
    inverted.
    """
    op = "raptorbt.stats.garch"
    out = _raptorbt.stats_garch(
        _series(returns, "returns", op),
        model,
        dist,
        _int(horizon, "horizon", op),
        None if periods_per_year is None else _int(periods_per_year, "periods_per_year", op),
    )
    warnings: list[str] = []
    if out["integrated"]:
        warnings.append(
            f"persistence {out['persistence']:.7f} is within 1e-6 of 1 or above: shocks do not "
            "fade, so there is no half-life and no unconditional variance (both None, "
            "stationary_variance False). Often a level shift or break in volatility inside the "
            "sample, not a persistent GARCH process: look at the series, or fit a shorter window"
        )
    if out["shape_at_bound"]:
        warnings.append(
            f"the t shape nu = {out['parameters']['nu']:.4g} sits at its bound (2.05 or 500): "
            "the tail weight is not estimated from this sample; do not read it, or fit "
            "dist='normal'"
        )
    if out["std_errors"] is None:
        warnings.append(
            "no robust standard errors: the Hessian could not be inverted, typically because a "
            "parameter sits on its boundary; the parameters are not precisely estimated"
        )
    out["warnings"] = warnings
    out["assumptions"] = [
        f"{dist} innovations, constant mean, one lag each; fitted on returns in percent",
        "variances are per bar; the horizon is in bars",
        "forecast is the model's, from the fitted parameters of this sample",
    ]
    return out


# --------------------------------------------------------------------------------- regimes


def markov_regimes(
    returns: Series,
    *,
    k: int = _REGIMES,
    min_support_bars: float = 10.0,
    min_duration_periods: float = 2.0,
    series: bool = False,
) -> dict[str, Any]:
    """Hidden Markov regimes: a Markov-switching mean and variance fitted to returns.

    ``returns`` are fractions, at least ``200 * (k - 1)``, not constant; ``k`` is 2 or 3 regimes,
    ordered by variance (regime 0 is the calmest). The fit starts from three deterministic
    starts and keeps the best likelihood.

    **Smoothed probabilities are not causal.** ``recent_probabilities`` (the last 20 bars),
    ``current_regime``, ``regime_stats[*].share_of_periods`` and the ``timeline`` are smoothed:
    conditioned on the whole sample, so an earlier bar's label uses returns after it. They
    describe the past; never trade on them. ``recent_filtered_probabilities`` and, with
    ``series=True``, ``filtered_probabilities`` (every bar) use only the returns up to each bar
    and are causal. With ``series=True`` the full ``smoothed_probabilities`` are also returned,
    as ``n`` by ``k`` arrays.

    A regime fitted to a handful of bars is not a regime. Each ``regime_stats`` entry has
    ``support_bars`` (its smoothed probabilities summed) and ``low_support``, true when that is
    under ``min_support_bars`` or its expected duration is under ``min_duration_periods``;
    ``low_support_regimes`` lists them and ``warnings`` says so in words. ``log_likelihood`` is of
    the series in percent.

    The fit does not depend on the units of the returns: a series whose percent variance is
    outside 0.1 to 10000 is scaled by a power of ten (``rescale``, 1.0 when none) as ``garch``
    does, and everything is returned in the units given. A series that varies by no more than
    1e-12 of its level is refused with ``degenerate_series`` (the fit would be to the rounding
    noise); a fit that does not settle is ``did_not_converge``.
    """
    op = "raptorbt.stats.markov_regimes"
    out = _raptorbt.stats_regimes(
        _series(returns, "returns", op),
        _int(k, "k", op),
        float(min_support_bars),
        float(min_duration_periods),
        bool(series),
    )
    out["transition_matrix"] = {
        "rows_from_columns_to": "[from][to]",
        "values": out["transition_matrix"],
    }
    out["causality"] = {
        "recent_probabilities": "smoothed: conditioned on the whole sample, not causal",
        "recent_filtered_probabilities": "filtered: returns up to each bar only, causal",
        "current_probabilities": "the last bar, where smoothed and filtered agree",
        "timeline": "built from the smoothed labels, not causal",
    }
    out["warnings"] = [
        f"regime {s['regime']} rests on {s['support_bars']:.1f} bars of evidence"
        + (
            ""
            if s["expected_duration_periods"] is None
            else f" and lasts {s['expected_duration_periods']:.1f} bars on average"
        )
        + ": a spike on a few bars, not a regime to act on"
        for s in out["regime_stats"]
        if s["low_support"]
    ]
    out["assumptions"] = [
        "Gaussian returns with a regime-specific mean and variance; first-order Markov",
        "regimes are ordered by variance; probabilities are smoothed over the whole "
        "sample, so an earlier bar's label uses information a live reader would not have had",
        "the fit is a local optimum; re-run on another window to see if the regimes persist",
    ]
    return out


# ------------------------------------------------------------------------------- inference


def hac_tstat(returns: Series, *, lags: int | None = None) -> dict[str, Any]:
    """The mean of ``returns`` against zero with a Newey-West (HAC) standard error.

    Bartlett kernel, no small-sample correction; the HAC p-value uses the normal reference, the
    naive one the t distribution with ``n - 1`` degrees of freedom. At least 30 observations, not
    constant. ``lags`` defaults to ``floor(4 (n / 100) ** (2 / 9))`` and must be 0 to
    ``n // 2 - 1``. Not causal: a statement about the whole sample.

    ``warnings`` holds ``nonnegative_series`` when no value is below zero: the mean of squared
    returns, volumes or absolute moves is positive by construction, and a large t says so, not
    that the series carries an edge.
    """
    op = "raptorbt.stats.hac_tstat"
    out = _raptorbt.stats_hac(
        _series(returns, "returns", op), None if lags is None else _int(lags, "lags", op)
    )
    out["warnings"] = []
    if np.nanmin(np.asarray(returns, dtype=np.float64)) >= 0.0:
        out["warnings"].append(
            "nonnegative_series: the series cannot be negative (squared returns, volumes, "
            "absolute moves), so its mean is positive by construction and its t measures that "
            "sign, not an edge. Test a signed series, or test whether a signal predicts it "
            "(predictive_test)"
        )
    out["assumptions"] = [
        "tests the mean against zero, two-sided, normal reference for the t",
        "Newey-West: robust to autocorrelation and heteroskedasticity up to the lags shown",
        "a t near 2 on a strategy chosen from many tries is not evidence (see pbo_cscv, spa_test)",
    ]
    return out


def predictive_test(
    signal: Series,
    prices: Series,
    *,
    horizons: Sequence[int] | int | None = None,
    lag: int = 1,
    cost_bps_per_side: float | None = None,
    quantiles: int | None = None,
    timestamps: Series | None = None,
    session: Any = None,
    to_session_close: bool = False,
) -> dict[str, Any]:
    """Does one causal signal predict the return that follows it, on a single series.

    ``signal`` and ``prices`` hold one value per bar (NaN in ``signal`` is unknown, a price is
    positive). The observation of bar ``t`` is the signal there against the return of entering
    ``lag`` bars later (default 1: a signal computed at a close cannot trade that close) and
    holding ``h`` bars: ``prices[t + lag + h] / prices[t + lag] - 1``. This forward return exists
    only here and never feeds an expression.

    With ``to_session_close=True`` (and no ``horizons``) there is one observation per session in
    which the signal becomes known: the first bar with a finite signal, entered ``lag`` bars
    later, held to that session's last bar. "Does the first hour predict the rest of the day." It
    needs ``timestamps`` (integer nanoseconds, non-decreasing); a session is a calendar day in the
    zone of ``session`` (a :class:`raptorbt.SessionSchedule`, or the zone's offset from UTC in
    nanoseconds; UTC when ``None``). A session that crosses midnight (MCX evening) is two.

    Per horizon (``result["horizons"][str(h)]``, or ``"session"``): ``n`` observations and
    ``n_independent`` (those whose bar index is a multiple of ``h``, whose windows share no bar;
    all of them in session mode); ``slope`` and ``intercept`` of the least-squares line of return
    on signal over all ``n``; **``t_nonoverlap``** with ``p_nonoverlap``, the t of the slope of
    the same line over the independent observations (Student's t, ``n_independent - 2`` degrees
    of freedom, NaN under 10): the headline. ``t_hac`` is the Newey-West t of the slope over all
    ``n`` with ``h - 1`` lags and **over-rejects for ``h > 1`` when the signal is persistent**:
    on 1,500 no-skill samples (iid 1% returns, 1,000 bars, ``h = 20``, an AR(1) signal of
    persistence 0.98) its standard deviation was 1.34 and it passed 1.96 on 14.5% of them, where
    ``t_nonoverlap`` had 1.03 and a p-value under 0.05 on 5.6%; with an iid signal both were near
    5%. Read ``t_nonoverlap``.
    Also ``rank_corr`` (Spearman), ``hit_rate`` (the share of observations with signal and return
    of the same sign, among those where both are non-zero) and, with ``quantiles``,
    ``quantile_returns`` (mean return per signal bucket, highest last), ``quantile_spread`` and
    its Welch t over the independent observations.

    The rule "long above zero, short below, flat at zero", rebalanced every ``h`` bars, has
    ``rule_mean`` and ``rule_t_nonoverlap`` over the independent observations, its ``turnover``
    (mean absolute position change per rebalance: entering from flat is one, a flip two; with
    ``to_session_close`` every non-flat session is a round trip, entered and closed inside the
    session, so it is two) and
    ``breakeven_cost_bps_per_side = mean(rule return) / turnover``: the flat cost per side at
    which it earns nothing. A signal whose breakeven is below its costs is not a strategy. With
    ``cost_bps_per_side`` the result adds ``net_mean`` and ``t_net`` (the rule less ``c`` times the
    position change). That is a flat rate on the traded notional, not the broker's schedule (no
    minimums, no per-side taxes, no slippage model): it rules a signal out, a backtest prices the
    survivors. Centre a signal that cannot change sign: the rule is always long.

    ``warnings`` names what makes a number weak (``signal_constant``, ``mostly_nan``,
    ``small_sample``: under 30 independent observations, ``nonnegative_signal``); ``flags`` a
    likely unadjusted split and a breakeven below the cost; ``assumptions`` what the test took for
    granted. ``forward_return`` per horizon is the return each bar's observation was scored on
    (NaN where none): bars after ``T`` never change those before ``T - lag - h``. Refused:
    ``insufficient_observations`` (fewer than 10 known signal values, or no usable session),
    ``invalid_input`` (an infinite value, a non-positive price, lengths that differ).
    """
    op = "raptorbt.stats.predictive_test"
    sig, px = _series(signal, "signal", op), _series(prices, "prices", op)
    if sig.shape != px.shape:
        raise _refuse(op, f"signal and prices must have one value per bar, got {len(sig)} and {len(px)}")
    lg = _int(lag, "lag", op)
    if lg < 0:
        raise _refuse(op, "lag must be 0 or more", "lag")
    if quantiles is not None:
        quantiles = _int(quantiles, "quantiles", op)
        if quantiles < 2:
            raise _refuse(op, "quantiles must be at least 2", "quantiles")
    if cost_bps_per_side is not None:
        cost = float(cost_bps_per_side)
        if not np.isfinite(cost) or cost < 0:
            raise _refuse(op, "cost_bps_per_side must be a number of at least 0", "cost_bps_per_side")
        cost_bps_per_side = cost
    hs: list[int] = []
    labels: np.ndarray | None = None
    if to_session_close:
        if horizons is not None:
            raise _refuse(op, "to_session_close takes no horizons: the return runs to the close", "horizons")
        if timestamps is None:
            raise _refuse(op, "to_session_close needs timestamps, to know which bars share a session", "timestamps")
    else:
        if horizons is None:
            raise _refuse(op, "give horizons, bars ahead to score, e.g. [1, 5]", "horizons")
        hs = [_int(h, "horizons", op) for h in ([horizons] if isinstance(horizons, int) else horizons)]
        if not hs or min(hs) < 1:
            raise _refuse(op, "horizons must be a non-empty list of integers >= 1", "horizons")
        if session is not None or timestamps is not None:
            pass  # accepted for the caller's own bookkeeping; only session mode reads them
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.shape != sig.shape or ts.dtype.kind not in "iu" or (len(ts) > 1 and (np.diff(ts.astype(np.int64)) < 0).any()):
            raise _refuse(op, "timestamps must be one non-decreasing integer (nanoseconds) per bar", "timestamps")
        if to_session_close:
            zone = 0
            if session is not None:
                zone = getattr(session, "tz_offset_ns", session)
                if isinstance(zone, bool) or not isinstance(zone, (int, np.integer)):
                    raise _refuse(
                        op, "session is a raptorbt.SessionSchedule or the zone's offset from UTC in "
                        "nanoseconds", "session",
                    )
            labels = (ts.astype(np.int64) + int(zone)) // _DAY_NS
    out = _raptorbt.stats_predictive(
        sig, px, labels, hs, lg, quantiles, cost_bps_per_side, bool(to_session_close)
    )
    known = sig[np.isfinite(sig)]
    warnings: list[str] = []
    flags: list[str] = []
    if out["n_signal_known"] < 0.5 * out["n_bars"]:
        warnings.append(
            f"mostly_nan: the signal is known on {out['n_signal_known']} of {out['n_bars']} bars; "
            "the test rests on those alone"
        )
    if known.size and known.min() >= 0.0:
        warnings.append(
            "nonnegative_signal: the signal is never below zero, so the long/short rule is always "
            "long (rule_* is the return of holding, not of the signal); centre it first. The slope "
            "and rank correlation are unaffected"
        )
    for key, h in out["horizons"].items():
        if h["signal_constant"]:
            warnings.append(f"signal_constant: horizon {key}: the signal does not vary, so no slope or t exists")
        if h["small_sample"]:
            warnings.append(
                f"small_sample: horizon {key}: {h['n_independent']} independent observations "
                "(under 30): the t is a rough guide, and NaN under 10"
            )
        if cost_bps_per_side is not None and h["breakeven_cost_bps_per_side"] < cost_bps_per_side:
            flags.append(
                f"horizon {key}: breakeven {h['breakeven_cost_bps_per_side']:.3g} bps per side is "
                f"below the {cost_bps_per_side:g} bps charged: the rule does not pay its costs"
            )
    if out["big_move_count"]:
        flags.append(
            f"{out['big_move_count']} one-bar price moves beyond a ratio of 1.5: likely a split or "
            "bonus the prices are not adjusted for; it reads as a return and distorts every statistic"
        )
    out["warnings"], out["flags"] = warnings, flags
    out["assumptions"] = [
        "prices are adjusted for splits, bonuses and dividends; this is not checked",
        "the signal at a bar uses only data up to that bar (causal); this is not checked",
        "returns are simple, entered `lag` bars after the signal and held to the horizon or the "
        "session's last bar",
        "the headline t uses every h-th observation (windows that share no bar); t_hac still "
        "over-rejects for h > 1",
        "the rule is sign(signal), rebalanced every h bars; a flat per-side rate on the traded "
        "notional is not the broker's schedule",
    ]
    return out


def pbo_cscv(cols: Columns, *, splits: int = _PBO_SPLITS) -> dict[str, Any]:
    """The probability of backtest overfitting by combinatorially symmetric cross-validation.

    The share of ways to choose half of ``splits`` equal time blocks as the in-sample half for
    which the strategy best in-sample ranks at or below the median out of sample (Bailey,
    Borwein, Lopez de Prado and Zhu). ``cols`` maps strategy names to per-period returns of one
    length, two to 200 strategies, at least ``4 * splits`` periods; ``splits`` is even, at least
    4, with at most 20,000 combinations. The oldest rows beyond a whole number of blocks are
    dropped (``dropped_rows``). The answer does not depend on the number of threads. Not causal.
    """
    op = "raptorbt.stats.pbo_cscv"
    _, arrays = _columns(cols, "strategies", op)
    out = _raptorbt.stats_pbo(arrays, _int(splits, "splits", op))
    out["sharpe_unit"] = "per period, not annualised"
    out["reading"] = (
        "pbo near 0.5 means the in-sample winner is no better than chance out of sample; near "
        "0 means selection carried over"
    )
    out["assumptions"] = [
        "every strategy in the matrix was tried; leaving out the failed ones understates pbo",
        f"{splits} equal time blocks (oldest {out['dropped_rows']} rows dropped to fit); the "
        "rank is of the in-sample best among all strategies",
        "blocks are exchangeable: serial dependence across block edges is ignored",
    ]
    return out


def spa_test(
    cols: Columns,
    *,
    benchmark: Series | None = None,
    reps: int = _SPA_REPS,
    block_size: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Whether the best strategy beats the benchmark after allowing for the search across all.

    Hansen's superior predictive ability and White's reality check, by the stationary bootstrap.
    ``cols`` maps strategy names to per-period returns of one length (at least 60, up to 200
    strategies); ``benchmark`` is cash (zero) when ``None``. When every strategy's excess over
    the benchmark is constant (identical to it, or all zero) there is nothing to bootstrap and
    ``degenerate_series`` is raised, not p = 0. ``reps`` is 200 to 5000;
    ``block_size`` is the mean block, default ``max(2, round(n ** (1 / 3)))``. The p-values are
    Monte Carlo estimates: the same ``seed`` gives the same numbers on any machine and thread
    count.

    **The statistic is not studentised.** It is the largest mean excess return, as arch 8.0.0
    computes it (its ``studentize`` option has no effect in that release); the variance estimate
    only decides which strategies are recentred for the ``consistent`` p-value. Do not read it as
    the studentised test of Hansen (2005). Not causal.
    """
    op = "raptorbt.stats.spa_test"
    names, arrays = _columns(cols, "strategies", op)
    bench = None if benchmark is None else _series(benchmark, "benchmark", op)
    seed = _int(seed, "seed", op)
    if not 0 <= seed < 2**64:
        raise _refuse(op, "seed must be a whole number from 0 to 2**64 - 1", "seed")
    out = _raptorbt.stats_spa(
        names,
        arrays,
        bench,
        _int(reps, "reps", op),
        None if block_size is None else _int(block_size, "block_size", op),
        seed,
    )
    out["reading"] = (
        "null: no strategy beats the benchmark. A small SPA p-value (consistent) rejects it for "
        "the best of the set; the White p-value is the conservative upper bound"
    )
    out["assumptions"] = [
        "excess return of each strategy over the benchmark, mean per period as the criterion",
        "every strategy tried is in the set; omitted ones understate the p-value",
        "stationary bootstrap of the periods (serial dependence kept within blocks)",
    ]
    out["studentised"] = False
    out["statistic_note"] = (
        "the statistic is the largest mean excess return, not studentised: arch 8.0.0 ignores "
        "its studentize option, and this is arch's statistic"
    )
    return out


# ---------------------------------------------------------------------------------- factors


def pca_factors(
    cols: Columns,
    *,
    components: int = _PCA_COMPONENTS,
    target: Series | None = None,
) -> dict[str, Any]:
    """Principal components of a panel of return series, and what a target loads on them.

    Components of the correlation matrix (each series standardised). ``cols`` maps names to
    per-period returns of one length (at least 60), two to 40 series; ``components`` is capped at
    5 and at the number of series. A component is oriented to load up on net (its loadings sum to
    a positive number); when that sum is within 1e-9 of zero the loading of largest magnitude is
    made positive, so the sign never depends on the SVD routine. With ``target`` (same length)
    the result adds its regression on the components' scores: ``r_squared``, ``alpha_per_period``
    and the ``exposures_per_unit_score`` with OLS t-statistics (not HAC). A component with no
    variance (a collinear panel) is reported with beta 0 and t 0. Not causal.
    """
    op = "raptorbt.stats.pca_factors"
    names, arrays = _columns(cols, "returns", op)
    tgt = None if target is None else _series(target, "target", op)
    out = _raptorbt.stats_pca(names, arrays, _int(components, "components", op), tgt)
    out["assumptions"] = [
        "principal components of the correlation matrix (series standardised)",
        "aligned observations: the same periods in every series",
        "components are statistical, not named economic factors",
    ]
    if "target" in out:
        out["target"]["note"] = (
            "scores have unit-free scale (standardised inputs); t-statistics are OLS, not HAC"
        )
    return out


# ---------------------------------------------------------------------------------- events


def event_study(
    dates: Sequence[date],
    asset: Series,
    market: Series,
    events: Sequence[date],
    *,
    estimation: int = _EVENT_ESTIMATION,
    gap: int = _EVENT_GAP,
    pre: int = 1,
    post: int = 1,
) -> dict[str, Any]:
    """Event study by the market model: abnormal returns around dated events.

    ``dates``, ``asset`` and ``market`` are aligned per-period returns, oldest first, dates
    strictly increasing. For each of 1 to 100 ``events`` the market model (OLS) is fitted on the
    ``estimation`` periods (at least 60) ending ``gap`` periods before the event window
    ``[-pre, +post]`` (each 0 to 20); an event off the calendar moves to the next period on it.
    Events without a full estimation and event window are listed under ``skipped``. Each event has
    its cumulative abnormal return ``car`` and t-statistic; ``caar`` is their mean, with the
    cross-sectional ``caar_t``, ``caar_p_value`` and ``positive_share`` from two events with
    spread.

    A fit with no residual noise (the estimation returns lie on the market line to rounding) has
    an undefined t: it is NaN and ``degenerate_fit`` is true on that event (dividing by a sigma of 1e-18
    would give a t near 5e15). The market model is fitted on periods before the event: causal for the event
    date, not a signal.
    """
    op = "raptorbt.stats.event_study"
    out = _raptorbt.stats_events(
        [_day(d, "dates", op) for d in dates],
        _series(asset, "asset", op),
        _series(market, "market", op),
        [_day(d, "events", op) for d in events],
        _int(estimation, "estimation", op),
        _int(gap, "gap", op),
        _int(pre, "pre", op),
        _int(post, "post", op),
    )
    out["assumptions"] = [
        "market model r = alpha + beta * market, OLS on the estimation window",
        "per-event t = CAR / (residual sigma * sqrt(window length)): ignores the "
        "estimation error of alpha and beta",
        "the cross-sectional t treats events as independent; clustered events overstate it",
        "abnormal returns are in the units of the inputs (simple returns assumed additive)",
    ]
    if out["caar_t"] is None:
        out["caar_note"] = "the cross-sectional t needs at least two events with spread"
    return out


# ------------------------------------------------------------------------------------ cost


def impact_estimate(
    high: Series,
    low: Series,
    close: Series,
    volume: Series,
    *,
    quantity: float,
    spread_model: str = _SPREAD_MODEL,
    coefficient: float = 1.0,
) -> dict[str, Any]:
    """What it costs to trade a size: square-root impact plus an estimated bid-ask spread.

    Daily bars, oldest first, at least 20, positive prices with high not below low and some
    volume. ``quantity`` is in units (shares or lots as the volume counts them). One-way cost in
    basis points is half the spread plus ``coefficient`` times the daily sigma of close-to-close
    log returns times the square root of ``quantity`` over the average daily volume. The spread
    is Corwin-Schultz (from highs and lows) or Roll (from closes; raises ``invalid_input`` when
    the serial covariance of price changes is not negative). An estimate from daily bars, not a
    fill. ``within_model_range`` is false past 25% participation. Not causal: it uses the whole
    sample given.
    """
    op = "raptorbt.stats.impact_estimate"
    out = _raptorbt.stats_impact(
        _series(high, "high", op),
        _series(low, "low", op),
        _series(close, "close", op),
        _series(volume, "volume", op),
        float(quantity),
        spread_model,
        float(coefficient),
    )
    out["kind"] = "estimate"
    out["formula"] = (
        "one-way cost = spread / 2 + coefficient * sigma_daily * sqrt(quantity / ADV)"
    )
    out["assumptions"] = [
        "an estimate from daily bars: not a quote, not a fill, no queue or latency",
        "square-root law, coefficient of order one (a free input); daily sigma of close "
        "to close log returns; ADV is the mean volume of the days given",
        "Corwin-Schultz assumes no overnight gap in the high-low range and biases up in "
        "volatile days; Roll is noisy on short samples",
        "participation above 25% of a day's volume is outside the range the law is calibrated for",
    ]
    return out


# --------------------------------------------------------------------------------- surface

_PLATEAU_SHARE = 0.7
_PLATEAU_FRACTION = 0.8
_SPIKE_SHARE = 0.3
_SMOOTH_MIN = 0.8


def sensitivity_surface(
    names: Sequence[str], cells: Sequence[tuple[Series, float]]
) -> dict[str, Any]:
    """How a metric moves across a parameter grid: a plateau of similar cells, or a spike.

    ``cells`` are ``(parameter values in names order, metric)``, nine to 2,000 of them over one to
    three distinct parameters; a higher metric is better. Neighbours of a cell differ by one step
    along exactly one parameter (the steps are the sorted distinct values). Conventions the
    numbers keep: ``best`` is the first maximum in input order; ``robust_pick`` is the cell with
    the highest mean of itself and its neighbours and may be a different cell from the best and
    need not be adjacent to it; a best cell with no neighbour has ``plateau_share`` 0 and reads as
    a spike. ``verdict`` is a heuristic on the thresholds in ``verdict_rule``, not a test. The
    surface is in-sample; a plateau is necessary for robustness, not sufficient.
    """
    op = "raptorbt.stats.sensitivity_surface"
    try:
        grid = [([float(v) for v in vals], float(metric)) for vals, metric in cells]
    except (TypeError, ValueError):
        raise _refuse(op, "cells are (parameter values, metric) pairs of numbers", "cells") from None
    out = _raptorbt.stats_surface([str(n) for n in names], grid)
    out["verdict_rule"] = (
        f"plateau: at least {_PLATEAU_SHARE:.0%} of the best cell's neighbours within "
        f"{1 - _PLATEAU_FRACTION:.0%} of the grid's range below it and smoothness at least "
        f"{_SMOOTH_MIN}; spike: {_SPIKE_SHARE:.0%} or fewer (heuristic thresholds, not a test)"
    )
    out["robust_pick"]["rule"] = "highest mean of the cell and its neighbours"
    coverage = out["coverage"]
    out["assumptions"] = [
        "smoothness = 1 minus the mean absolute metric difference of neighbouring cells over "
        "the metric's range across the grid",
        "the grid is the whole search: cells not run are absent, not zero"
        + (f" (coverage {coverage:.0%})" if coverage < 1 else ""),
        "an in-sample surface: a plateau is necessary for robustness, not sufficient",
    ]
    return out


# ------------------------------------------------------------------------------ data quality


def audit_bars(
    t: Series,
    o: Series,
    h: Series,
    low: Series,
    c: Series,
    v: Series,
    *,
    interval_seconds: int,
    outlier_z: float = _OUTLIER_Z,
    mad_scale: float = _MAD_SCALE,
    listed: int = _LISTED,
    daily_gap_days: float = _DAILY_GAP_DAYS,
    intraday_gap_bars: float = _INTRADAY_GAP_BARS,
    split_factors: Sequence[float] = _SPLIT_FACTORS,
    split_tolerance: float = _SPLIT_TOLERANCE,
    min_bars: int = _MIN_BARS,
    dead_bar_share_max: float = _DEAD_BAR_SHARE_MAX,
) -> dict[str, Any]:
    """Is a stretch of bars fit to research on: pure checks over OHLCV arrays.

    ``t`` is epoch seconds, oldest first; ``interval_seconds`` is the bar width (a day or more is
    a daily series). Reports duplicate and out-of-order stamps, impossible OHLC, zero-volume and
    flat bars, return outliers (robust z of the log return, median over MAD, at least
    ``outlier_z``), gaps and daily close-to-close ratios near a whole split factor, each with its
    count and the first ``listed`` examples, in IST. ``clean`` is false when any of them is found
    or when more than ``dead_bar_share_max`` of the bars are flat with no volume (a series of nothing but such bars
    is not clean).

    It holds no exchange holiday calendar, so a gap is a stretch with no bars, not a missing
    session, and a flagged split is a ratio to check against the corporate actions, not a
    finding. Amounts (``return_pct``, ``close_ratio``, ``days``) are at full precision. Too few
    bars raises ``insufficient_observations``.
    """
    op = "raptorbt.stats.audit_bars"
    seconds = _int(interval_seconds, "interval_seconds", op)
    if seconds <= 0:
        raise _refuse(op, "interval_seconds must be positive", "interval_seconds")
    out = _raptorbt.stats_audit_bars(
        _series(t, "t", op),
        _series(o, "open", op),
        _series(h, "high", op),
        _series(low, "low", op),
        _series(c, "close", op),
        _series(v, "volume", op),
        seconds,
        float(outlier_z),
        float(mad_scale),
        _int(listed, "listed", op),
        float(daily_gap_days),
        float(intraday_gap_bars),
        [float(f) for f in split_factors],
        float(split_tolerance),
        _int(min_bars, "min_bars", op),
        float(dead_bar_share_max),
    )
    out["cannot_judge"] = [
        "missing sessions (no exchange holiday calendar here: a gap is a stretch with no bars)",
        "whether a flagged split is already adjusted (check the corporate actions)",
    ]
    return out
