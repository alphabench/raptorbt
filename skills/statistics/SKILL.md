---
name: raptorbt-statistics
description: >
    Statistical tests and fits over arrays for research: stationarity, cointegration, memory
    (Hurst), GARCH volatility, Markov regimes, HAC t-statistics, overfitting (PBO, SPA), factors,
    event studies, trading cost, parameter surfaces and a data audit. Which test answers which
    question, how many observations it needs, what it refuses and which results are not causal.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
status: stable
---

# RaptorBT Statistics

## 1. When to use this skill

Use it to put a number, with its sample size and its assumptions, on a research question about
returns, prices or backtest results: is this series stationary, do these prices share an
equilibrium, is this edge distinguishable from zero, was this winner picked out of many, what
will this size cost to trade, are these bars fit to study. Each function takes arrays and returns
a dict; nothing here reads a file, a clock or a network.

Pick the test by the question:

| Question | Function |
| --- | --- |
| Does this series revert to a level, or wander (a unit root)? | `stationarity` |
| Do two or more price series share a long-run relation I could trade as a spread? | `cointegration` |
| Do returns trend or reverse over longer horizons? | `hurst` |
| How volatile will the next bars be, and how fast does a shock fade? | `garch` |
| Which calm or turbulent regime was the market in, over a past sample? | `markov_regimes` |
| Is a strategy's mean return different from zero, given autocorrelation? | `hac_tstat` |
| How likely is it that the strategy I picked from many is overfitted? | `pbo_cscv` |
| After searching many strategies, does the best one beat the benchmark? | `spa_test` |
| What do several return series have in common, and what does a target load on it? | `pca_factors` |
| Did dated events move the price? | `event_study` |
| What will it cost to trade a given size? | `impact_estimate` |
| Does the best parameter sit on a plateau or on a spike? | `sensitivity_surface` |
| Are these bars fit to research on? | `audit_bars` |

Do not use it to size, to place orders or to decide in a running strategy: a test over a window
describes the window. The one exception is the filtered output of `markov_regimes`, which is
causal (section 7).

## 2. Domain overview

Every function lives in `raptorbt.stats`, which is not imported by `import raptorbt`: write
`from raptorbt import stats`. It is pure and may be imported from a user algorithm.

A result is a plain dict at full precision. NaN and infinity are what the number is, nothing is
rounded (round when you show it). It carries the sample size (`n`, or `n_periods`, `n_bars`,
`bars`) and `assumptions`, a list of plain sentences naming what the test took for granted. A
function that cannot answer raises a `RaptorBTError`, which is also a `ValueError`, and returns
no number:

- `insufficient_observations`: fewer observations than the statistic needs; the message gives
  the count it got and the count it needs.
- `degenerate_series`: the data cannot carry the statistic (a constant series, series that are
  exact combinations of one another, a fit with no residual).
- `did_not_converge`: a model fit (GARCH, regimes) did not settle.
- `invalid_input`: a NaN or infinite value, series of unequal length, an argument out of range.

The sample each function needs:

| Function | Input | Needs |
| --- | --- | --- |
| `stationarity` | one series | 50 observations |
| `cointegration` | 2 to 12 level series of one length | 60 observations |
| `hurst` | one series of returns | 256 |
| `garch` | one series of returns | 250 |
| `markov_regimes` | one series of returns | 200 for 2 regimes, 400 for 3 |
| `hac_tstat` | one series of returns | 30 |
| `pbo_cscv` | 2 to 200 return series | 4 times `splits` rows (64 at the default 16) |
| `spa_test` | 1 to 200 return series | 60 |
| `pca_factors` | 2 to 40 return series | 60 |
| `event_study` | dated asset and market returns, 1 to 100 events | an estimation window of 60 or more before each event |
| `impact_estimate` | daily high, low, close, volume | 20 days |
| `sensitivity_surface` | 9 to 2,000 cells, 1 to 3 parameters | 9 cells |
| `audit_bars` | OHLCV arrays and the bar width | 5 bars |

## 3. API reference

All keyword parameters below have the default shown; the first positional arguments are required.

| Name | What it returns |
| --- | --- |
| `raptorbt.stats.stationarity(x, *, alpha=0.05)` | ADF and KPSS side by side and a `reading` (`stationary`, `unit_root`, `contradictory`, `inconclusive`). |
| `raptorbt.stats.cointegration(cols, *, alpha=0.05)` | Engle-Granger (statistic, p-value, hedge ratios, spread z-score, half-life) and Johansen (`rank` to read, `rank_at_5pct` unscaled, trace statistics raw and scaled, `small_sample`, cointegrating vector). `cols` maps names to level series; the first is the dependent one. |
| `raptorbt.stats.hurst(increments)` | Hurst exponent by rescaled range and by DFA, and a `reading`. |
| `raptorbt.stats.garch(returns, *, model="garch", dist="t", horizon=10, periods_per_year=None)` | Parameters, persistence, half-life, forecast volatility path, recent conditional volatility, robust standard errors. |
| `raptorbt.stats.markov_regimes(returns, *, k=2, min_support_bars=10.0, min_duration_periods=2.0, series=False)` | Regime statistics, transition matrix, smoothed and filtered probabilities, timeline. |
| `raptorbt.stats.hac_tstat(returns, *, lags=None)` | Mean, Newey-West standard error, t and p, the naive t and p, per-period Sharpe. |
| `raptorbt.stats.pbo_cscv(cols, *, splits=16)` | Probability of backtest overfitting and the Sharpe ratios behind it. |
| `raptorbt.stats.spa_test(cols, *, benchmark=None, reps=1000, block_size=None, seed=0)` | Three p-values for the best strategy against the benchmark, with the bootstrap used. |
| `raptorbt.stats.pca_factors(cols, *, components=3, target=None)` | Explained variance, loadings by component, and a target's regression on the scores. |
| `raptorbt.stats.event_study(dates, asset, market, events, *, estimation=120, gap=5, pre=1, post=1)` | Per-event abnormal returns, CAR and t; CAAR with its t and p. |
| `raptorbt.stats.impact_estimate(high, low, close, volume, *, quantity, spread_model="corwin_schultz", coefficient=1.0)` | Half spread, impact and one-way cost in basis points and money. |
| `raptorbt.stats.sensitivity_surface(names, cells)` | Smoothness, best cell, robust pick, verdict, per-cell table, a matrix for two parameters. |
| `raptorbt.stats.audit_bars(t, o, h, low, c, v, *, interval_seconds)` | Counts and first examples of each defect, and whether the bars are `clean`. |

`garch(model=...)` is `garch` or `gjr`, `dist` is `normal` or `t`, `horizon` is 1 to 60 bars.
`spa_test` takes `reps` from 200 to 5000. `audit_bars` also takes its thresholds as keywords
(`outlier_z`, `mad_scale`, `listed`, `daily_gap_days`, `intraday_gap_bars`, `split_factors`,
`split_tolerance`, `min_bars`, `dead_bar_share_max`); the defaults are those in the signature.

## 4. Standard workflow

1. Name the question and take the function from the table in section 1.
2. Check the sample against the table in section 2. A short sample is refused, not guessed at.
3. Call the function and read `assumptions` before the number. If your data breaks one, the
   number does not answer your question.
4. Read the result's refusals as findings: a `degenerate_series` on a pair of prices says they
   are the same series, not that the test failed.
5. If you searched many variants to find the one you are testing, say so to the test: use
   `pbo_cscv` and `spa_test` on all of them, not `hac_tstat` on the winner.
6. Round only when you display.
7. Read `warnings` and the flags before the number (section 4.1): they name the cases in which
   a fit or a test prints a confident figure that means little.

### 4.1 Where a result is not to be read at face value

**Johansen rank.** Read `johansen["rank"]`, not `rank_at_5pct`. The asymptotic critical values
assume many observations against the number of series `q`. On independent random walks the
unscaled `rank_at_5pct` reports a relation far too often when `q` is large against the sample:
100% of 400 samples at twelve series and 60 observations. `rank` applies Reinsel and Ahn's
`(T - q k) / T` (here `k = 2`, `T` the `effective_observations`) to the trace statistics
(`trace_statistics_scaled`) and then reads the same 5% critical values. `small_sample` is true
when the factor is under 0.9 (`T < 20 q`); the critical values are then far from the finite-sample
distribution and the test has little power, so rank 0 there is not evidence of no relation.
Measured on independent driftless random walks, 400 samples per cell, one lagged difference,
share of samples with rank above 0 (5% nominal), unscaled `rank_at_5pct` then scaled `rank`:

| Series | 60 obs | 120 obs | 250 obs | 500 obs |
| --- | --- | --- | --- | --- |
| 2 | 15% then 13% | 14% then 11% | 14% then 14% | 14% then 14% |
| 4 | 22% then 6% | 10% then 6% | 11% then 8% | 10% then 9% |
| 6 | 43% then 2% | 20% then 5% | 11% then 7% | 8% then 5% |
| 12 | 100% then 0.3% | 80% then 1% | 32% then 2% | 15% then 3% |

Two things stay true after the correction. At two to four series the rate stays above 5% at any
length (about 13% at two): the critical values for a constant in the model are not exact for
driftless walks, so a rank of 1 is a lead to confirm with Engle-Granger and out of sample, not a
finding. And the correction costs power: with one real cointegrated pair among the walks, `rank`
above 0 on 69%, 100%, 100% of samples at two series (60, 120, 250 observations); 4%, 30%, 95% at
six; 0%, 2%, 27% at twelve. With many series, bring many observations.

**GARCH.** Returns constant to a few ulps (spread at most 1e-12 of the level) are refused as
`degenerate_series`. Read `warnings`: `integrated` is true when persistence is within 1e-6 of
1 or above, which is what a level shift in volatility looks like (a step from 0.5% to 3% daily
volatility fits to persistence 0.9999999999996): the half-life and unconditional volatility are
then `None` and `stationary_variance` false, not 1.9e12 bars and 578. `shape_at_bound` is true
when the t shape sits at its bound of 2.05 or 500; `std_errors` is `None` when the Hessian cannot
be inverted. An illiquid name that is 95% exactly zero returns gives both, with a confident
half-life nobody should read.

**Regimes.** The fit does not depend on the units of the returns (`rescale` is the power of ten
applied when the percent variance is outside 0.1 to 10000, as for GARCH). Returns constant to a
few ulps are refused as `degenerate_series`.

**SPA.** When every strategy's excess over the benchmark is constant (all zero against cash, or
identical to the benchmark) there is nothing to bootstrap: `degenerate_series`, not p = 0.

## 5. Examples

A stationary series and a random walk. The ADF has a unit root as its null, so the stationary
series rejects it and the walk does not.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(7)
noise = rng.normal(size=600)
ar = np.zeros(600)
for t in range(1, 600):
    ar[t] = 0.5 * ar[t - 1] + noise[t]
walk = np.cumsum(rng.normal(size=600))

a = stats.stationarity(ar)
w = stats.stationarity(walk)
assert a["reading"] == "stationary" and a["adf"]["rejects_unit_root"]
assert not w["adf"]["rejects_unit_root"]
assert w["n"] == 600 and w["alpha"] == 0.05
```

Cointegration. `y` follows `x` with a hedge ratio of 1.5; `z` is an unrelated walk. The first
series is the dependent one in the Engle-Granger regression.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(11)
x = np.cumsum(rng.normal(size=400)) + 100.0
y = 2.0 + 1.5 * x + rng.normal(size=400)
z = np.cumsum(rng.normal(size=400)) + 100.0

pair = stats.cointegration({"y": y, "x": x})
assert pair["engle_granger"]["cointegrated"]
assert abs(pair["engle_granger"]["hedge_ratios"]["x"] - 1.5) < 0.05
assert pair["johansen"]["rank_at_5pct"] >= 1

apart = stats.cointegration({"x": x, "z": z})
assert not apart["engle_granger"]["cointegrated"]
assert apart["johansen"]["rank_at_5pct"] == 0
```

Memory. Slow cycles in the returns read as persistence. The band is a heuristic, not a test: on
twelve seeds of 1,200 white-noise returns, two read under 0.45 (the reference
implementation, which this one replays to 1e-10).

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(1)
white = rng.normal(size=1200) * 0.01
slow = white + 0.005 * np.sin(2 * np.pi * np.arange(1200) / 600)

assert stats.hurst(white)["reading"] == "random_walk"
h = stats.hurst(slow)
assert h["reading"] == "persistent"
assert h["rescaled_range"]["hurst"] > 0.55 and h["dfa"]["hurst"] > 0.55
```

Volatility. Returns from a GARCH(1,1) with persistence 0.95; the fit recovers a persistence
near it and a half-life of shocks in bars. With `periods_per_year` the volatilities are also
annualised.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(5)
eps = rng.normal(size=1000)
r = np.zeros(1000)
var = 1e-4
for t in range(1000):
    r[t] = np.sqrt(var) * eps[t]
    var = 1e-6 + 0.1 * r[t] ** 2 + 0.85 * var

g = stats.garch(r, dist="normal", horizon=5, periods_per_year=252)
assert g["converged"] and 0.9 < g["persistence"] < 1.0
assert len(g["forecast_vol_per_period"]) == 5
assert abs(g["last_vol_annualised"] - g["last_vol_per_period"] * 252**0.5) < 1e-12
```

Regimes. Two volatility regimes, ordered by variance. The smoothed probabilities describe the
past and are not causal; the filtered ones use only the returns up to each bar. They agree on
the last bar only, because only there has the smoother seen nothing more than the filter.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(2)
state = np.zeros(800, dtype=int)
for t in range(1, 800):
    state[t] = state[t - 1] if rng.random() < 0.97 else 1 - state[t - 1]
ret = np.where(state == 0, 0.005, 0.02) * rng.normal(size=800)

m = stats.markov_regimes(ret, series=True)
calm, wild = m["regime_stats"]
assert calm["vol_per_period"] < wild["vol_per_period"]
assert m["regimes"] == 2 and m["converged"]
smooth, filt = m["smoothed_probabilities"], m["filtered_probabilities"]
assert smooth.shape == (800, 2)
assert np.allclose(smooth[-1], filt[-1])
assert not np.allclose(smooth, filt)
assert m["smoothed_is_causal"] is False and m["filtered_is_causal"] is True
assert m["low_support_regimes"] == []
```

HAC. With positively autocorrelated returns the naive standard error is too small; the
Newey-West one is larger.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(1)
e = rng.normal(size=500) * 0.01
r = np.zeros(500)
for t in range(1, 500):
    r[t] = 0.5 * r[t - 1] + e[t]
r += 0.0008

h = stats.hac_tstat(r)
naive_se = h["mean"] / h["naive_t"]
assert h["hac_std_error"] > naive_se > 0
assert abs(h["hac_t"]) < abs(h["naive_t"])
assert h["lags"] == 5
```

Overfitting and the best of many. Ten noise strategies and one with a real edge; `seed` fixes
the bootstrap, so the same call gives the same numbers.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(4)
noise = {f"s{i}": rng.normal(0.0, 0.01, size=640) for i in range(10)}
edge = dict(noise, edge=rng.normal(0.003, 0.01, size=640))

assert stats.pbo_cscv(edge, splits=8)["pbo"] < stats.pbo_cscv(noise, splits=8)["pbo"]
spa = stats.spa_test(edge, reps=500, seed=1)
assert spa["best_strategy"] == "edge" and spa["p_value_spa_consistent"] < 0.05
assert spa == stats.spa_test(edge, reps=500, seed=1)
```

Factors. Five series sharing a factor; a target that loads on it.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(6)
factor = rng.normal(0, 0.01, size=300)
panel = {f"a{i}": 0.8 * factor + rng.normal(0, 0.006, size=300) for i in range(5)}
target = 2.0 * factor + rng.normal(0, 0.004, size=300)

pc = stats.pca_factors(panel, components=2, target=target)
assert pc["first_component_is_common_factor"]
assert pc["explained_variance_ratio"][0] > 0.5
assert pc["target"]["r_squared"] > 0.5
```

Event study. A 5% jump planted on four event dates. An event too early for an estimation window
is listed under `skipped`, not dropped silently.

```python
from datetime import date, timedelta

import numpy as np
from raptorbt import stats

rng = np.random.default_rng(8)
days = [date(2022, 1, 3) + timedelta(days=i) for i in range(600)]
market = rng.normal(0.0004, 0.01, size=600)
asset = 0.0002 + 1.2 * market + rng.normal(0, 0.005, size=600)
for k in (200, 300, 400, 500):
    asset[k] += 0.05

es = stats.event_study(days, asset, market, [days[k] for k in (200, 300, 400, 500)])
assert es["n_events"] == 4 and es["caar"] > 0.03 and es["caar_p_value"] < 0.05

early = stats.event_study(days, asset, market, [days[10], days[300]])
assert early["n_events"] == 1 and early["skipped"][0]["event"] == "2022-01-13"
assert early["caar_t"] is None
```

Cost. The impact term grows with the square root of the size: four times the quantity, twice the
impact. Past a quarter of a day's volume the law is outside its range.

```python
import numpy as np
from raptorbt import stats

rng = np.random.default_rng(9)
close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=120)))
high, low, volume = close * 1.01, close * 0.99, np.full(120, 1e5)

a = stats.impact_estimate(high, low, close, volume, quantity=1000.0)
b = stats.impact_estimate(high, low, close, volume, quantity=4000.0)
assert abs(b["impact_bps"] / a["impact_bps"] - 2.0) < 1e-9
assert abs(a["one_way_cost_bps"] - (a["half_spread_bps"] + a["impact_bps"])) < 1e-9
assert a["within_model_range"]
assert not stats.impact_estimate(high, low, close, volume, quantity=40000.0)["within_model_range"]
```

Parameter surface. A smooth bump is a plateau; the same grid with one tall cell is a spike. The
robust pick includes the cell itself, so a tall enough spike is also the robust pick.

```python
from raptorbt import stats

fast = [5 + 2.5 * i for i in range(9)]
slow = [50 + 25 * j for j in range(9)]


def metric(f, s):
    return 1.5 - 0.5 * (((f - 15) / 10) ** 2 + ((s - 150) / 100) ** 2)


cells = [([f, s], metric(f, s)) for f in fast for s in slow]
assert stats.sensitivity_surface(["fast", "slow"], cells)["verdict"] == "plateau"

spiky = [([f, s], 2.5 if (f, s) == (15.0, 150) else metric(f, s) - 0.3) for f in fast for s in slow]
spike = stats.sensitivity_surface(["fast", "slow"], spiky)
assert spike["verdict"] == "spike" and spike["best"]["plateau_share"] == 0.0
assert spike["robust_pick"]["params"] == spike["best"]["params"]
```

Data audit. One bar whose high is below its low.

```python
import numpy as np
from raptorbt import stats

DAY = 86_400
t = 1_704_067_200 + DAY * np.arange(30, dtype=float)
rng = np.random.default_rng(12)
close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 30)))
o, h, low, v = close * 0.999, close * 1.004, close * 0.996, np.full(30, 1e5)

assert stats.audit_bars(t, o, h, low, close, v, interval_seconds=DAY)["clean"]
h[10] = low[10] * 0.9
bad = stats.audit_bars(t, o, h, low, close, v, interval_seconds=DAY)
assert not bad["clean"]
assert bad["findings"]["impossible_ohlc"] == {"count": 1, "first": ["2024-01-11 05:30"]}
```

A refusal is a `ValueError` carrying a code, so a loop over many series can skip the ones that
cannot be tested and say why.

```python
import numpy as np
from raptorbt import stats

try:
    stats.stationarity(np.ones(100))
except ValueError as e:
    assert e.code == "degenerate_series"
try:
    stats.stationarity(np.arange(20.0))
except ValueError as e:
    assert e.code == "insufficient_observations"
```

## 6. Composition

The statistics read arrays, so any source works: `raptorbt.run_strategies` results give returns
for `hac_tstat`, `pbo_cscv` and `spa_test`; bars give `audit_bars` and `impact_estimate`; a
parameter sweep gives the cells of `sensitivity_surface`. Run `pbo_cscv` and `spa_test` on every
variant you tried, not on the winner alone. `audit_bars` first: every other statistic assumes the
bars are right.

An indicator or a derived series can feed them without leaving expressions:
[Analysis](../analysis/SKILL.md) evaluates the formula, keeps it under a handle and runs
`Session.stats("stationarity", handle)` on it, cutting the warm-up for you.

`raptorbt.capabilities()["statistics"]["functions"]` lists the function names. Refusals are
registered in `raptorbt.errors.CODES`.

## 7. Constraints and correctness

**Causal or not.** A test over a window describes that window. `markov_regimes` makes the
difference explicit: `recent_probabilities` (and, with `series=True`, `smoothed_probabilities`),
`current_regime`, `share_of_periods` and the `timeline` are smoothed, conditioned on the whole
sample, so a past bar's label uses returns after it: not causal, never a signal.
`recent_filtered_probabilities` and `filtered_probabilities` use the returns up to each bar and
are causal. The result says so in `smoothed_is_causal`, `filtered_is_causal` and `causality`.
Cointegration hedge ratios, GARCH parameters, PCA loadings and the event-study market model are
fitted in sample and drift out of it.

**A low-support regime is not a regime.** Maximum likelihood can reward a regime fitted to a
handful of bars, and on data with no regimes the best fit can be a seven-bar one. Each regime
carries `support_bars` (its smoothed probabilities summed) and `low_support`, true under
`min_support_bars` or when its expected duration is under `min_duration_periods`. Say so when
you show one that is flagged; `warnings` states it in words.

**SPA is not studentised.** `spa_test` uses the largest mean excess return as the statistic, as
arch 8.0.0 does: arch's `studentize` option has no effect in that release, and replaying its
bootstrap indices without studentisation reproduces its p-values exactly. This is not the
studentised test of Hansen (2005); the `consistent` p-value recentres by a variance estimate but
the statistic is not divided by one. The p-values are Monte Carlo estimates and carry the noise
of `reps`.

**Other limits to state when you quote a result.**

- `cointegration`: the Engle-Granger p-value (MacKinnon 1994) exists up to six series; from
  seven it is `None`, and `cointegrated` falls back to the statistic against the 2010 critical
  value when `alpha` is 0.01, 0.05 or 0.10, else `None`. Johansen refuses a panel whose moment
  matrices are singular to working precision (condition number above about 4.5e7), such as exact
  copies or exact combinations: `degenerate_series`. The cointegrating vector is the first
  eigenvector normalised to the first series; it means something only when the rank is at least
  one.
- `hurst`: the band (above 0.55, below 0.45) is a heuristic; the standard errors are of the
  log-log fit and understate the estimator's own variance. The `reading` is classified from the
  unrounded estimates.
- `stationarity`: `kpss.p_value` is read from a table and clipped to [0.01, 0.10].
- `event_study`: the per-event t ignores the estimation error of alpha and beta, and the
  cross-sectional t treats events as independent. A fit with no residual noise has an undefined
  t: NaN, with `degenerate_fit` true.
- `sensitivity_surface`: `best` is the first maximum in input order; `robust_pick` need not be
  adjacent to it; an isolated best cell has `plateau_share` 0 and reads as a spike; the
  `verdict` is a heuristic on stated thresholds, not a test.
- `pca_factors`: a component's sign is fixed by its loadings summing to a positive number, or,
  when that sum is within 1e-9 of zero, by its largest loading being positive. A null component
  has beta 0 and t 0 in the target regression.
- `audit_bars`: no exchange holiday calendar, so a gap is a stretch with no bars; a flagged
  split is a ratio to check against corporate actions. A series of flat bars with no volume is
  not `clean`.
- `garch`: returns of a very small scale are rescaled to fit and reported in the units given
  (`rescale` says by how much); a fit that still does not settle is `did_not_converge`.

## 8. Common mistakes

- Testing a short sample and reading the refusal as a bug. `insufficient_observations` names the
  count it needs.

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import stats

stats.hac_tstat(np.arange(5.0))
```

- Passing prices where returns belong (`hurst`, `garch`, `hac_tstat`, `markov_regimes`), or
  returns where levels belong (`cointegration`). The tests run and the answer is about the
  wrong thing.
- Passing a NaN: nothing is imputed. `invalid_input` names the series.

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import stats

x = np.cumsum(np.ones(100))
x[3] = np.nan
stats.stationarity(x)
```

- Trading on `recent_probabilities`: it uses future returns. Use the filtered ones.
- Reading `hac_t` near 2 on a strategy chosen from many as evidence. It is not; see
  `pbo_cscv` and `spa_test`.
- Calling `spa_test` studentised.
- Treating a hedge ratio or a half-life as a constant of nature.

## 9. Validation

Every function is replayed against the library it replaces, on committed fixtures
(`tests/python/oracle/fixtures/stats/`, generated from statsmodels, arch and scipy and kept
beside the oracle harness): ADF and KPSS lags exactly and their statistics and p-values to
1e-8, Engle-Granger to 1e-8, Johansen eigenvalues to 1e-8 and vectors to 1e-6, the HAC
t-statistic, Hurst and the event study to 1e-10, the SPA p-values to 1e-10 when arch's own
bootstrap indices are replayed (the engine's own seeded draws agree with arch's within Monte
Carlo error), PCA to 1e-8, GARCH at a log-likelihood no more than 1e-3 below the reference's and
regimes no more than 1e-4 below (parameters and probabilities where the two reach the same
optimum). The replay fails if a family replays no case.

Before you rely on a number from your own data:

1. Read `assumptions` and check each against the data.
2. Run the test on data where you know the answer (white noise, a series you built to revert)
   and see it say so.
3. For a fit, check it on a second window: a regime or a volatility model that does not repeat
   is a description of one sample.
4. For anything you chose from a set, run `pbo_cscv` and `spa_test` on the whole set.

## 10. Related skills

- [Core](../core/SKILL.md): the engine concepts and the settings every run shares.
- [Analysis](../analysis/SKILL.md): run these tests on an expression's result by handle, and
  chain the result into the next formula.
- [Results](../results/SKILL.md): the returns and metrics of a backtest, which the inference
  functions here take as input.
- [Optimization](../optimization/SKILL.md): parameter sweeps, whose grids `sensitivity_surface`
  reads.
- [Troubleshooting](../troubleshooting/SKILL.md): the error codes, including those of this
  domain.
