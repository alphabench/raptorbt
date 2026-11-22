# Research statistics

`raptorbt.research` holds the numbers a parameter search needs to be honest about itself. Each is
a small, deterministic function; none runs a backtest.

| Function | What it answers |
|---|---|
| `deflated_sharpe(sharpes, best, *, samples, skew, kurtosis, periods_per_year, luck)` | Is the best Sharpe of N trials above what luck alone hands the best of N? Returns `dsr` (a probability), `sr0_annualised` and the inputs. |
| `stationary_bootstrap(returns, *, draws, block, seed, periods_per_year)` | The spread of the annualised mean return when the series is resampled in random-length blocks. |
| `resample_paths(values, *, method, draws, block, replace, seed, periods_per_year, drawdown_levels)` | The spread of a finished result's ending return and maximum drawdown over paths resampled from its returns (block bootstrap) or its trades (with replacement, or a shuffle). |
| `without_best(returns, top)` | The sum of a return series with its `top` best periods left out. |
| `Design.grid(space)`, `Design.random(space, n, seed=...)` | The points of a parameter space: the full product, or `n` seeded draws. |
| `WalkForward(days, train=, test=, step=)` | Rolling in-sample and out-of-sample windows over a list of days. |
| `sweep(run, design, windows, ...)` | Every (point, window) through your `run` function; the rows, a streaming summary and the deflated Sharpe of the best. |

## Determinism

Every draw comes from SplitMix64 seeded by the seed you pass (masked to 64 bits, so any Python
integer works). A draw is the top 53 bits over 2^53 and `below(n)` is a plain modulo, so the same
seed gives the same stream on every machine and any thread count: nothing here splits work by the
number of threads. The sums use the engine's correctly rounded sum (Python's `math.fsum`), and the
normal quantile is a fixed 200-step bisection on the platform-independent CDF in `pricing::norm`.

`sweep` returns the same rows whatever `workers` is: runs are put back in run order.

## Resampled paths

`resample_paths` is a Monte Carlo of a finished result, not of a model: each of `draws` paths is
built from the observations themselves and compounded from a starting value of 1, and its maximum
drawdown is measured on that compounded path from its running peak, the start included.

- `method="block"` takes per-period simple returns and resamples them with the same stationary
  bootstrap as `stationary_bootstrap` (mean block `block`, default the cube root of the length,
  rounded half to even). Dependence survives within a block and is cut between blocks.
- `method="trades"` takes per-trade returns (each trade's P&L over the capital it used). With
  `replace=True` the trades are drawn with replacement; with `replace=False` they are put in a
  uniformly random order (Fisher-Yates), so every path holds each trade once. The trades are
  assumed independent and equally sized.

A shuffle cannot change the ending return: every path ends at the observed one, so
`ending_return_fixed` is `True`, the ending-return percentiles all equal `observed` and `prob_loss`
is 0 or 1. Only the drawdown distribution is informative there: it says how much of the observed
drawdown was the order the trades came in.

The growth of a path is summed as `ln(1 + r)`, so a long path neither overflows nor rounds through
thousands of products; a return of -1 ruins the path (a drawdown of 1). Percentiles are nearest
rank, as in `stationary_bootstrap`. The drawdown histogram has 20 equal bins from 0 to the largest
drawn drawdown (to 1 when no path has one), so a caller can draw the distribution without the
paths.

It needs at least 10 observations: with ten, the default block (2) leaves about five blocks to
recombine and a shuffle has 10! orders, far more than the draws; at six there are 720 orders,
fewer than the default 2,000 draws. Ten is a floor for the arithmetic, not a sample that makes the
answer reliable: a drawdown spread from a few dozen trades is itself very uncertain, and `p99`
rests on the few largest observations. Fewer refuse with `insufficient_observations`; a non-finite
value, a return below -1, `draws` outside 1 to 100,000 and more than 16 `drawdown_levels` (each in
(0, 1]) refuse with `invalid_input`.

## Reading a deflated Sharpe

`luck="trials"` takes the spread of luck from the trials themselves, right for a sweep over one
family of parameters. `luck="samples"` takes it from the Sharpe's own sampling variance
`1 / (samples - 1)`, right when the trials are unlike each other and only their number counts.
Read a `dsr` under 0.95 as "not distinguishable from the search", read an out-of-sample Sharpe only,
and report the number of trials beside every headline figure.

## Agreement with the reference implementation

The functions are the reference implementation's. `tests/python/differential/test_research_differential.py`
replays a fixture written by the reference implementation: designs, windows and the
integer draws are equal; floats agree to 1e-12 relative.

`row_from(result)` is new in shape: the reference one read a session; this one reads a `BacktestResult`
(or `PortfolioResult`) and returns `sharpe`, `sortino`, `total_return`, `max_drawdown` (fractions),
`fills`, `fees`, `samples`, `skew` and `kurtosis`.
