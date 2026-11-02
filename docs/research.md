# Research statistics

`raptorbt.research` holds the numbers a parameter search needs to be honest about itself. Each is
a small, deterministic function; none runs a backtest.

| Function | What it answers |
|---|---|
| `deflated_sharpe(sharpes, best, *, samples, skew, kurtosis, periods_per_year, luck)` | Is the best Sharpe of N trials above what luck alone hands the best of N? Returns `dsr` (a probability), `sr0_annualised` and the inputs. |
| `stationary_bootstrap(returns, *, draws, block, seed, periods_per_year)` | The spread of the annualised mean return when the series is resampled in random-length blocks. |
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
