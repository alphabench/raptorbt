---
name: raptorbt-optimization
description: >
    Parameter sweeps, batch runs and Monte Carlo validation.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
    - raptorbt-backtesting
    - raptorbt-results
status: stable
---

# RaptorBT Optimization

## 1. When to use this skill

Load it before you search over parameters, rank variants, quote the best of
many runs, or put a confidence statement on a result. Searching is cheap here,
which is the danger: the best of a hundred runs of pure noise looks like an edge.
This skill gives the tools that keep a search honest (walk-forward windows,
the deflated Sharpe ratio, bootstrap, a Monte Carlo of the portfolio) and the
rules for using them.

## 2. Domain overview

Three groups, each deterministic for a given input and seed:

- **Batch runs.** `batch_single_backtest` runs many signal pairs over one copy of
  the bars across all cores, and returns `(id, result)` pairs in input order,
  bit-identical to running each through `run_single_backtest`.
- **Search and validation** in `raptorbt.research`. `Design.grid` and
  `Design.random` list parameter points; `WalkForward` cuts a list of trading
  days into rolling train and test windows; `sweep` runs every (point, window)
  through your function and returns the rows, a streaming summary and the
  deflated Sharpe ratio of the best. `deflated_sharpe`, `stationary_bootstrap`
  and `without_best` answer "is this distinguishable from luck, and how fragile
  is it"; `resample_paths` answers "how deep could the drawdown have been" by
  resampling a finished result's returns or trades into many paths.
- **Portfolio Monte Carlo.** `simulate_portfolio_mc` simulates the combined
  return of several strategies forward from their sample statistics.

**Multiple testing.** Run N trials and the best Sharpe grows with N even if every
trial is noise. The deflated Sharpe ratio (DSR) is the probability that the best
trial's Sharpe exceeds what luck alone hands the best of N, given the sample
length, skew and kurtosis. Read a `dsr` under 0.95 as "not distinguishable from
the search", judge an **out-of-sample** Sharpe only, and report the number of
trials beside every headline figure.

**Event time.** A walk-forward window is a set of trading days: you train on the
earlier days and measure on the later ones. Nothing in `sweep` enforces that your
`run` function respects it; it only passes the window. Indicators computed on the
test slice must not use the train or the future slice beyond what the strategy
could have known (warm-up from earlier days is fine and is your decision).

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.BatchSingleItem(item_id, entries, exits, direction=1, weight=1.0, symbol=, config=None, position_sizes=None, instrument_config=None)` | One signal pair. |
| `raptorbt.batch_single_backtest(timestamps, open, high, low, close, volume, items, config=None)` | Runs all items in parallel; `[(item_id, BacktestResult), ...]` in input order. |
| `raptorbt.BatchSpreadItem`, `raptorbt.batch_spread_backtest(timestamps, underlying_close, items, config=None)` | The same for option spreads; see [backtesting](../backtesting/SKILL.md). |
| `raptorbt.research.Design.grid(space)`, `.random(space, n, seed=)` | Parameter points (`.points`). A random design draws each parameter from its list, or from a `(lo, hi)` range, by SplitMix64. |
| `raptorbt.research.WalkForward(days, train=, test=, step=None)` | Rolling windows over a list of day strings; `.windows()` gives `Window(index, train, test)`. `step` defaults to `test`, so test windows tile. `WalkForward.single(days)` is one window with no training days. |
| `raptorbt.research.sweep(run, design, windows, workers=1, executor="process", manifest=None, metric="sharpe", periods_per_year=252)` | Every (point, window) through `run(params, window)`; returns a `SweepResult` (`rows`, `summary`, `trials`, `deflated`, `.best(metric)`). |
| `raptorbt.research.row_from(result)` | A sweep row from a `BacktestResult`: `sharpe`, `sortino`, `total_return`, `max_drawdown` (fractions), `fills`, `fees`, `samples`, `skew`, `kurtosis`. |
| `raptorbt.research.deflated_sharpe(sharpes, best, *, samples, skew=0.0, kurtosis=3.0, periods_per_year=252.0, luck="trials")` | The deflated Sharpe: `dsr`, `sr0_annualised`, `best_annualised`, `z`, `trials`. |
| `raptorbt.research.stationary_bootstrap(returns, *, draws, block, seed, periods_per_year)` | The spread of the annualised mean return under block resampling: `annual_return` `p05`, `p50`, `p95` and `above_zero`. |
| `raptorbt.research.resample_paths(values, *, method="block", draws=2000, block=None, replace=True, seed, periods_per_year=None, drawdown_levels=(0.1, 0.2, 0.3))` | A Monte Carlo of a finished result: `draws` compounded paths from a return series (`method="block"`, the stationary bootstrap) or from per-trade returns (`method="trades"`, with replacement, or a shuffle with `replace=False`). `ending_return` and `max_drawdown` (`p05`, `p50`, `p95`, `p99`, `observed`), `annual_return`, `prob_loss`, `prob_drawdown` per level, `drawdown_histogram` (`edges`, `counts`), `n`, `draws`, `block`, `ending_return_fixed`. At least 10 observations. |
| `raptorbt.research.without_best(returns, top)` | The sum of a return series with its `top` best periods left out. |
| `raptorbt.research.OVERFITTING_NOTE` | The reading rule above, as text. |
| `raptorbt.simulate_portfolio_mc(returns, weights, correlation_matrix, initial_value, n_simulations=10000, horizon_days=252, seed=42)` | Correlated geometric Brownian motion; returns a dict. |

## 4. Standard workflow

1. Fix everything that is not a parameter: data, costs, fill timing, session.
2. Name the parameter space and its size. Prefer a small, meaningful grid; use
   `Design.random` when the space is large.
3. Choose windows with `WalkForward` and write `run(params, window)` so it fits on
   the window's train days and measures on its test days. Return `row_from(result)`.
4. Call `sweep`. Read `res.deflated`: if `dsr < 0.95`, the best is not
   distinguishable from the search.
5. Select on the train side, report the test side. Compare the chosen
   parameters' train and test Sharpe across windows; a large drop is overfitting.
6. Bootstrap the chosen strategy's returns, check `without_best`, and read the
   drawdown spread from `resample_paths`.
7. Report the number of trials, the windows and the costs with the result.

## 5. Examples

A batch run is the serial run, in order. Zip the results against the parameters
that produced them:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, BatchSingleItem, batch_single_backtest, run_single_backtest

DAY = 86_400 * 10**9
T0 = 1_767_225_600 * 10**9
N = 500
rng = np.random.default_rng(11)
close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, N)))
ts = T0 + np.arange(N, dtype=np.int64) * DAY
open_ = np.concatenate([[close[0]], close[:-1]])
high, low, volume = np.maximum(open_, close) * 1.002, np.minimum(open_, close) * 0.998, np.full(N, 1e6)
config = BacktestConfig(initial_capital=1e6, fees=0.0005, periods_per_year=252.0)

grid = [(f, s) for f in (5, 10, 20) for s in (50, 100)]
items = []
for f, s in grid:
    fast, slow = raptorbt.sma(close, f), raptorbt.sma(close, s)
    items.append(BatchSingleItem(f"sma{f}_{s}", fast > slow, fast < slow))

out = batch_single_backtest(ts, open_, high, low, close, volume, items, config)
assert [item_id for item_id, _ in out] == [item.item_id for item in items]   # input order
for (f, s), (_, batch) in zip(grid, out):
    fast, slow = raptorbt.sma(close, f), raptorbt.sma(close, s)
    one = run_single_backtest(ts, open_, high, low, close, volume, fast > slow, fast < slow,
                              config=config)
    assert np.array_equal(batch.equity_curve(), one.equity_curve())         # bit-identical
```

A walk-forward sweep over a random walk, which has no edge. The best of 120
trials looks good, the deflated Sharpe says it is not distinguishable from the
search, and what training selects does worse on the test windows. The same rows
come back whatever the worker count:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, research, run_single_backtest

DAY = 86_400 * 10**9
T0 = 1_767_225_600 * 10**9
N = 1000
rng = np.random.default_rng(11)
close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, N)))      # no edge by construction
ts = T0 + np.arange(N, dtype=np.int64) * DAY
open_ = np.concatenate([[close[0]], close[:-1]])
high, low, volume = np.maximum(open_, close) * 1.002, np.minimum(open_, close) * 0.998, np.full(N, 1e6)
days = [str(i) for i in range(N)]                                 # a day is its bar index

def backtest(first, last, fast, slow):
    sl = slice(first, last + 1)
    f, s = raptorbt.sma(close[sl], fast), raptorbt.sma(close[sl], slow)
    return run_single_backtest(
        ts[sl], open_[sl], high[sl], low[sl], close[sl], volume[sl], f > s, f < s,
        config=BacktestConfig(initial_capital=1e6, fees=0.0005, periods_per_year=252.0),
    )

def run(params, window):
    train = backtest(int(window.train[0]), int(window.train[-1]), params["fast"], params["slow"])
    test = backtest(int(window.test[0]), int(window.test[-1]), params["fast"], params["slow"])
    row = research.row_from(test)                  # the measurement is the test window
    row["train_sharpe"] = train.metrics.sharpe_ratio or 0.0     # None: the window never traded
    return row

design = research.Design.grid({"fast": [3, 5, 8, 13, 21], "slow": [30, 50, 80, 120]})
windows = research.WalkForward(days, train=250, test=125)
res = research.sweep(run, design, windows, periods_per_year=252)

assert res.trials == len(design) * len(windows.windows()) == 120
assert res.deflated["trials"] == 120 and res.deflated["dsr"] < 0.95        # not distinguishable
assert res.deflated["best_annualised"] < res.deflated["sr0_annualised"]    # luck hands more than this

# Choose on the train side in each window, then look at what the test side did.
chosen = []
for w in range(len(windows.windows())):
    rows = [r for r in res.rows if r["window"] == w]
    chosen.append(max(rows, key=lambda r: r["train_sharpe"]))
assert np.mean([r["train_sharpe"] for r in chosen]) > np.mean([r["sharpe"] for r in chosen])

threaded = research.sweep(run, design, windows, workers=3, executor="thread")
assert threaded.rows == res.rows                                           # worker count is irrelevant
```

The deflated Sharpe in isolation. The same best Sharpe means less as the number
of trials grows and more as the sample grows:

```python
import numpy as np
from raptorbt import research

def dsr(trials, samples):
    sharpes = list(np.linspace(0.0, 1.4, trials))        # the best of them is 1.4
    return research.deflated_sharpe(sharpes, 1.4, samples=samples, luck="samples")["dsr"]

assert dsr(2, 500) > dsr(10, 500) > dsr(100, 500) > dsr(1000, 500)   # more trials, less credit
assert dsr(50, 60) < dsr(50, 250) < dsr(50, 1000)                    # more data, more credit

one = research.deflated_sharpe([1.4], 1.4, samples=500)              # one trial: no multiple testing
assert one["dsr"] is None and "trials" in one["note"]
```

Fragility of one strategy's returns. The bootstrap resamples in random-length
blocks (so autocorrelation survives), the seed fixes the draws, and
`without_best` drops the best periods:

```python
import numpy as np
from raptorbt import research

rng = np.random.default_rng(11)
returns = list(rng.normal(0.0003, 0.01, 499))            # daily returns of some strategy

a = research.stationary_bootstrap(returns, draws=500, block=10, seed=3, periods_per_year=252)
b = research.stationary_bootstrap(returns, draws=500, block=10, seed=3, periods_per_year=252)
c = research.stationary_bootstrap(returns, draws=500, block=10, seed=4, periods_per_year=252)
assert a == b and a != c                                  # same seed, same draws
lo, mid, hi = (a["annual_return"][k] for k in ("p05", "p50", "p95"))
assert lo < mid < hi and 0.0 <= a["above_zero"] <= 1.0

total = sum(returns)
assert research.without_best(returns, 5) < total          # five best days removed
```

The paths a finished result could have taken. Resample the per-period returns in
blocks, or the trades with replacement, to see the spread of the ending return and
the maximum drawdown; shuffle the trades to see how much of the drawdown was the
order they came in. A shuffle cannot change where the path ends:

```python
import numpy as np
from raptorbt import research

rng = np.random.default_rng(11)
returns = rng.normal(0.0004, 0.01, 500)                  # daily returns of some strategy
trades = rng.normal(0.004, 0.03, 120)                     # its trades, P&L over capital used

paths = research.resample_paths(returns, seed=3, periods_per_year=252,
                                drawdown_levels=[0.1, 0.2, 0.3])
assert paths == research.resample_paths(returns, seed=3, periods_per_year=252,
                                        drawdown_levels=[0.1, 0.2, 0.3])   # same seed, same paths
dd = paths["max_drawdown"]
assert dd["p05"] <= dd["p50"] <= dd["p95"] <= dd["p99"]
assert paths["block"] == 8.0                              # cube root of 500, rounded
p10, p20, p30 = (row["probability"] for row in paths["prob_drawdown"])
assert p10 >= p20 >= p30
assert sum(paths["drawdown_histogram"]["counts"]) == paths["draws"] == 2000

shuffled = research.resample_paths(trades, method="trades", replace=False, seed=3)
assert shuffled["ending_return_fixed"]                    # every order ends in the same place
er = shuffled["ending_return"]
assert er["p05"] == er["p99"] == er["observed"]
assert shuffled["max_drawdown"]["p05"] < shuffled["max_drawdown"]["p95"]  # the drawdown is what varies

drawn = research.resample_paths(trades, method="trades", seed=3)          # with replacement
assert drawn["ending_return"]["p05"] < drawn["ending_return"]["p95"]
```

A portfolio Monte Carlo. It draws from a Gaussian model fitted to the sample, so
read it as a sizing aid and not as a forecast. Results are percent of the
starting value, and the seed makes them repeatable:

```python
import numpy as np
import raptorbt

rng = np.random.default_rng(3)
returns = np.ascontiguousarray(np.vstack([rng.normal(0.0005, 0.01, 500),
                                          rng.normal(0.0003, 0.012, 500)]))   # one row per strategy
weights = np.array([0.5, 0.5])
corr = np.corrcoef(returns)

mc = raptorbt.simulate_portfolio_mc(returns, weights, corr, 100_000.0,
                                    n_simulations=4000, horizon_days=252, seed=1)
final = mc["final_values"]
assert len(final) == 4000
assert np.isclose(mc["expected_return"], 100 * (final.mean() / 100_000.0 - 1))   # percent
assert np.isclose(mc["probability_of_loss"], (final < 100_000.0).mean())         # a fraction
assert mc["cvar_95"] >= mc["var_95"]                                             # tail average is worse
again = raptorbt.simulate_portfolio_mc(returns, weights, corr, 100_000.0,
                                       n_simulations=4000, horizon_days=252, seed=1)
assert np.array_equal(again["final_values"], final)
```

## 6. Composition

- Reading a run, and why ratios depend on the session length: [results](../results/SKILL.md).
- Which runner the sweep's `run` calls, and array versus class-based strategies:
  [backtesting](../backtesting/SKILL.md).
- Costs and fill timing that must stay fixed while you sweep: [execution](../execution/SKILL.md).
- Portfolio construction (the optimiser, not this search): [portfolio](../portfolio/SKILL.md).

## 7. Constraints and correctness

- **Search is a multiple-testing procedure.** The deflated Sharpe corrects the
  Sharpe of the best trial for the number of trials, the sample length, and the
  skew and kurtosis of its returns. `luck="trials"` takes the spread of luck from
  the trials themselves (right for a sweep over one family of parameters);
  `luck="samples"` takes it from the sampling variance `1 / (samples - 1)` (right
  when the trials are unlike each other and only their number counts). It needs at
  least two trials and three samples; otherwise `dsr` is `None` with a note.
- **Sharpe inputs are annualised.** `deflated_sharpe` takes annualised Sharpes
  and the `periods_per_year` they used; rows from `row_from` already are.
  Mix a daily and an annualised Sharpe and the answer is meaningless.
- **Count every trial,** including the ones you discarded by eye and the
  variants of the idea. `sweep` counts the rows it ran; a search you did by hand
  before is not in that number.
- **`sweep` does not fit or split for you.** It runs the function you give it on
  each window. The train/test discipline lives in `run`.
- **Determinism.** Every draw comes from SplitMix64 seeded by the seed you pass,
  so the same seed gives the same stream on every machine and thread count.
  `sweep` returns the same rows whatever `workers` is. `workers > 1` with
  `executor="process"` needs `run` to be an importable, module-level function.
- **Resumable sweeps.** `manifest=path` appends one JSON row per finished run and
  skips those on a second call with the same manifest. Delete it when the data,
  costs or function changed.
- **Monte Carlo assumes Gaussian returns with the sample mean and covariance,**
  no fat tails, no volatility clustering, no autocorrelation, and a drift you
  estimated with error. `var_95` and `cvar_95` are percent of the starting value
  (positive numbers are losses).
- **Pass `returns` to `simulate_portfolio_mc` as a C-contiguous array.** A
  transposed view panics inside the engine (a `PanicException`) instead of
  raising a clear error: apply `np.ascontiguousarray`.
- **The bootstrap resamples blocks of a stationary series.** It says nothing
  about a regime that is not in the sample.
- **`resample_paths` recombines what happened; it is not a forecast.** Every
  path is built from the observations, so a loss larger than the worst one in
  the sample never appears. Paths compound their returns from a starting value
  of 1, and the drawdown is measured on that compounded path from its running
  peak, the start included.
- **The block method keeps dependence only within a block.** Autocorrelation
  and volatility clustering survive inside a block of mean length `block` and
  are cut at its ends; a longer `block` keeps more of it and recombines less.
- **The trade methods assume the trades are independent and equally sized.**
  Each value is one trade's return on the capital it used, compounded as if each
  trade staked the whole account. Trades that overlapped in time, were sized
  differently or depend on one another (a losing streak in one regime) are
  treated as interchangeable.
- **A shuffle (`replace=False`) only reorders.** Every path holds each trade
  once, so the ending return is the observed one on every path:
  `ending_return_fixed` is `True`, its percentiles collapse to `observed`, and
  `prob_loss` is 0 or 1. Only the drawdown distribution is informative there.
- **Small samples.** It needs at least 10 observations (fewer raise
  `insufficient_observations`); a few dozen trades give a drawdown spread that
  is itself very uncertain, and percentiles in the tail (`p99`) rest on the few
  largest observations.

## 8. Common mistakes

An unknown `luck` model is refused:

<!-- skill: invalid -->
```python
from raptorbt import research

research.deflated_sharpe([1.0, 0.5, 0.2], 1.0, samples=500, luck="optimistic")  # InvalidParameter
```

The sample count has no default, because the answer depends on it:

<!-- skill: invalid -->
```python
from raptorbt import research

research.deflated_sharpe([1.0, 0.5, 0.2], 1.0)   # TypeError: missing the keyword argument 'samples'
```

- Reporting the best in-sample run. Report the out-of-sample figure and the trial
  count.
- Tuning on the test window after seeing it. Each look is another trial.
- Searching a large grid, taking the best, and quoting its Sharpe without
  `dsr`.
- Choosing parameters at the edge of the grid: the optimum is outside it, or the
  surface is a ramp.
- Preferring a sharp peak to a plateau. Neighbouring parameters should perform
  alike; if a one-step change flips the sign, the peak is noise.
- Changing costs or fill timing between variants and ranking them.
- Treating the Monte Carlo's percentiles as a forecast.
- Reading the ending return of a shuffle: it is the observed one on every path.
- Passing P&L in currency or points to `resample_paths`: it takes returns
  as fractions (0.01 is 1%).
- Using `workers > 1` with a lambda or nested function under `executor="process"`.

## 9. Validation

1. Run the sweep on shuffled or random-walk data (the example above). If the
   method finds an "edge" there, it will find one in your data too.
2. Check `res.trials` equals the number of points times windows you intended.
3. Read `res.deflated["dsr"]`. Below 0.95, the best is not evidence.
4. Compare train and test Sharpe of what you would have chosen in each window.
5. Look at the parameter neighbourhood of the chosen point, not only the point.
6. Bootstrap the chosen strategy and check `above_zero` and `p05`; read the
   `max_drawdown` `p95` of `resample_paths` against the drawdown you can bear.
7. Run `sweep` with `workers=1` and with `workers=3, executor="thread"`; the rows
   must be equal.
8. Re-run the chosen strategy with `fill_timing="next_bar_open"` and doubled
   costs.

## 10. Related skills

- [Results](../results/SKILL.md)
- [Backtesting](../backtesting/SKILL.md)
- [Execution](../execution/SKILL.md)
- [Portfolio](../portfolio/SKILL.md)
