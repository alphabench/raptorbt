---
name: raptorbt-analysis
description: >
    Exploring a price series before building a strategy: load bars once, evaluate expressions and
    statistics, and get a small true summary with a short handle that the next expression reads as
    a series. In Python (a Session) or from a shell (python -m raptorbt). The loop an agent
    follows from a task to evidence, what each flag means, and how a tested expression moves into
    a strategy unchanged.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
    - raptorbt-expressions
    - raptorbt-statistics
status: stable
---

# RaptorBT Analysis

## 1. When to use this skill

Use it to find out what a series does before any strategy exists: how an indicator is
distributed, whether it is defined where you need it, whether a name even moves, whether a price
series is stationary, how fast it reverts. Each step is one call that returns a summary of at most
2 KB, so an agent can take many steps and keep the answers.

Do not use it inside a strategy. `raptorbt.analysis` reads and writes files (the command line
keeps its sessions on disk, `export` writes a file), so it cannot be imported inside a sandboxed
strategy environment that forbids file access. The text you explore here goes into a strategy unchanged: `ctx.indicators.expr` with
the same string gives the same numbers (section 6).

## 2. Domain overview

A **session** holds the bars you loaded and the results you made. A **handle** names one result:
`h_` and ten hex digits, a hash of the expression, the data and the engine version. The same work
always has the same handle, and asking again returns the stored answer instead of computing it.
A handle is a series in the next expression: `zscore(h_0a1b2c3d4e, 20)`.

A **summary** is the answer to `eval`. It states facts and never advice: `handle`, `kind`
(`number` or `bool`), `shape`, `warmup`, `nan` (leading and interior), `dist` (min, p05, p50, p95,
max, mean, std; for a boolean `known`, `true` and `true_rate` instead), `last` (5 values),
`flags`, `causal`, `reference`, `sha256`. Shown numbers are rounded to 6 digits; the stored
series is not. A result with several outputs (`macd(close)`) returns `{"handle", "outputs":
[summary, ...]}`, each output is the handle `<handle>_<output>`, and `truncated_outputs` lists
outputs left out if the 4 KB cap for a call is reached.

A **panel summary** (several instruments over many dates, `load_panel`) has the same shape with
the cross-section in it: `shape` is `[dates, names]`, `coverage` is the names with a known value
per date (min, median, max), and `last` is the last cross-section: `top` and `bottom` are lists
of `[name, value]` (for a boolean: `true` and the first `names` that are true). Flags count cells
and names rather than bars. Instruments are always named, so a rank or a z-score tells you which
ones.

The session is bounded. `max_bytes` (default 1 GiB) covers the loaded bars and every held series;
`max_handles` (default 256) the number of results. The least recently used handles are evicted
to make room. A handle that is gone raises `handle_expired` and the error carries the call that
recomputes it; a handle that never existed raises `unknown_handle`; a result larger than the
whole budget raises `session_memory_exceeded`.

The command line is the same session on disk, so a shell agent can chain handles across
invocations.

## 3. API reference

`raptorbt.analysis.Session(max_bytes=1 << 30, max_handles=256)`:

| Call | What it does |
| --- | --- |
| `load(bars, bar_stamp, bar_ns=None, name=None)` | Copy the bars in. `bars`: a dict (or a DataFrame, or anything answering `bars["close"]`; no pandas is imported) of float64 `open`, `high`, `low`, `close`, `volume` (any subset) and int64 `timestamps`. `bar_stamp` (`"open"` or `"close"`) is required: it says whether a timestamp starts or ends its bar. The input is never modified. Returns the data's facts. |
| `load_panel(bars, bar_stamp, bar_ns=None, instruments=None, name=None)` | A panel, in the shapes `Expr.eval_panel` takes: a dict of 2-D arrays (dates by names) with 1-D `timestamps` (and a bool `present`), or a mapping of instrument to its own bars (aligned on the union of timestamps, absent where there is no bar). `instruments` selects and orders, or names the columns (default S0, S1, ...); a name is text of at most 64 characters. `bar_stamp` is required. |
| `eval(expr, data=None, tz_offset_ns=0)` | Evaluate text (or an `Expr`) over the newest data (or `data=` a name) and return the summary. Over a panel, a cross-sectional expression (`rank`, `zscore_xs`, ...) is allowed and the summary is a panel summary. |
| `sweep(expr, n=[...], m=[...], score=None, data=None, tz_offset_ns=0)` | Evaluate text with `$name` constants under every combination (or `grid={...}`, or `points=[{...}]`; see `Expr.sweep`) in one pass, and keep each binding's result as a handle of its own: the handle `eval` of the bound text returns. Returns a table: `columns` (`handle`, the swept names, `score`), `rows`, `n_trials` (bindings tried), `n_distinct`, `kernel_steps` and a `note` on selection bias. `score` is a statistic (`true_rate`, `mean`, `std`, `min`, `max`, `last`, `known`, `leading_nan`; default `true_rate` for a boolean, `mean` for a number) or an expression, which may read the same `$name`s and is reduced to the mean of its known values. A table past the 4 KB call cap keeps the highest scores, ordered by score, and says `omitted`; every binding still has its handle. One series per binding: select an output of a multi-output call. Bars only, not a panel. |
| `factor(target, horizons=(1, 5, 20), quantiles=5, lag=1, min_names=10, data=None)` | `factor_report` on a panel handle (or an expression, evaluated first) against the loaded `close`. Returns, per horizon, `n_scored`, `ic_mean`, `t_hac` (the t-statistic with a Newey-West correction for the overlap of `h`-bar returns, `hac_lags` lags) beside `t_naive`, `hit_rate`, `long_short_mean` and `long_short_t_hac`; and `turnover` (top and bottom quantile), `autocorr`, `names` per date and `no_costs: true`: every spread is gross. Past 2 KB the last horizons move to `truncated_horizons`. The per-date arrays are behind the summary's `handle`. |
| `series(handle, tail=None, instruments=None)` | The stored series, a read-only float64 array at full precision (a dict of arrays for a multi-output handle or a factor report). `instruments` picks panel columns by name; an unknown one raises `unknown_symbol` with the nearest. |
| `describe_handle(handle)` | Expression, data, outputs, size, dependencies and the call that recreates it. |
| `drop(handle)` | Release it; later use raises `handle_expired`. |
| `stats(name, *handles, **kwargs)` | Run a `raptorbt.stats` function on handles: one handle for `stationarity`, `hurst`, `garch`, `markov_regimes`, `hac_tstat`; one or more for `cointegration`, `pbo_cscv`, `spa_test`, `pca_factors`; none for `audit_bars` and `impact_estimate` (the loaded bars). Warm-up NaNs are cut and the result's `analysis` says how many. |

The flags. Each is a count of bars, except `non_causal`; a flag that does not fire is absent.

| Flag | Definition |
| --- | --- |
| `flat_windows` | Bars with a known output whose trailing `warmup + 1` bars hold one value in every input the expression reads. Not computed for `at(...)` (no single warm-up). The window is the whole expression's, so a short indicator inside a long one is judged over the long window: evaluate it alone. |
| `gaps` | Runs of NaN that start after the first known value. |
| `outliers` | Known values more than 8 robust standard deviations (1.4826 times the MAD) from the median. A series with MAD 0 has none. Numbers only. |
| `nonpositive_input` | Bars on which a price read (open, high, low, close) is zero or negative. |
| `inf` | Infinities in the output and in the inputs read. |
| `short_history` | Fewer than 30 known values; the value is how many there are. |
| `non_causal` | The functions called whose registry row says a value uses later bars. |

`causal` is `true` unless `non_causal` is present. `reference` names, for up to 8 of the functions
used, what their numbers were checked against.

Looking things up: `raptorbt.expressions.functions(query)` searches names, TA-Lib names, families
and descriptions (all words must match; the best match first); `raptorbt.expressions.describe(name)`
returns the signature, a working `example`, `warmup`, `reference`, `relation` to TA-Lib and
`conventions`. `raptorbt.skills.for_task(text)` lists the skills a task needs.

Command line (`python -m raptorbt`), JSON on stdout:

| Command | Prints |
| --- | --- |
| `eval EXPR --data FILE --bar-stamp close` | The summary. `--data` is a `.csv` (header; integer or ISO-8601 timestamps with an offset), `.npz` or `.npy`; omit it to reuse the session's bars. |
| `sweep EXPR --n 5,10,20 --m 50,100` | The sweep table. Put `EXPR` first; each other `--name v1,v2` is a parameter (`--grid name=v1,v2` for a name that is also an option, `--points JSON` for listed bindings). `--score STAT_OR_EXPR`, `--data`, `--use` as for `eval`. |
| `functions WORDS...` | `{"count", "shown", "functions": [{name, family, signature, talib, doc}]}` (`--limit`, default 20). |
| `describe NAME` | The function in full. |
| `eval EXPR --panel FILE --bar-stamp close` | The panel summary. `--panel` is a long `.csv` (time, `instrument`, prices) or an `.npz` of 2-D arrays with `timestamps` and `instruments`; `--instruments A,B` selects. |
| `factor TARGET --horizons 1,5,20` | The factor summary (`--quantiles`, `--lag`, `--min-names`). `TARGET` is a panel handle or an expression. |
| `show HANDLE [--tail N \| --all]` | The handle's recipe and the last values, full precision. A panel result shows the names at both ends of its last cross-section; `--instruments A,B` picks others. |
| `export HANDLE --out FILE` | Writes `.csv`, `.npy` or `.npz` (a panel only `.npz`); prints the path and rows. |
| `stats NAME HANDLE... [--args JSON]` | The statistic, long lists cut to head and tail. |
| `skills for TEXT` | The skills a task needs, best first, then what they depend on. |
| `skills show NAME` | A skill, as markdown (the one command that is not JSON). |
| `skills export DIR [--task TEXT]` | Copies the skills to `DIR/raptorbt-<domain>/SKILL.md` for any agent. |

Every command takes `--session NAME` (default `default`) and `--cache-dir DIR`. A refusal prints
the error's fields (`type`, `code`, `message`, `operation`, `parameter`, `value`, `valid`, `skill`;
an expression error adds `position` and `snippet`) and exits 2.

## 4. Standard workflow

1. Find the skills: `python -m raptorbt skills for "<the task>"`, and read what it names.
2. Find the function: `functions "moving average"`, then `describe sma` for its warm-up, its
   reference and a working example.
3. Load the bars once, with the `bar_stamp` they have.
4. `eval` the expression. Read `warmup` (no value before it means anything), `nan`, `flags` and
   `causal` before the distribution.
5. Chain: write the handle into the next expression instead of repeating the work.
6. Read the numbers with `series` or `show` when the summary is not enough; chart them from the
   stored array, never from a printed list.
7. Test a claim with `stats` on a handle (a test over a window describes the window).
8. For several instruments: `load_panel`, evaluate a cross-sectional expression, chain it, then
   `factor` it. Read `no_costs`, `t_hac` against `t_naive` and `names` before the IC.
9. Move the expression into a strategy with `ctx.indicators.expr`, and run `run_strategies`.

## 5. Examples

Load, evaluate, read the flags and chain. The second expression reads the first by its handle,
and the repeat is the same handle without a recomputation.

```python
import numpy as np
from raptorbt.analysis import Session

n = 300
rng = np.random.default_rng(3)
close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
bars = {"timestamps": np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9, "close": close}

s = Session()
s.load(bars, bar_stamp="close")
rsi = s.eval("rsi(close, 14)")
assert rsi["warmup"] == 14 and rsi["nan"] == {"leading": 14, "interior": 0}
assert rsi["flags"] == {} and rsi["causal"] is True and rsi["shape"] == [n]

z = s.eval(f"zscore({rsi['handle']}, 20)")
assert z["warmup"] == 14 + 19
assert s.eval("rsi(close, 14)")["handle"] == rsi["handle"]
assert len(s.series(z["handle"], tail=3)) == 3
```

A name that has not traded. Its RSI is 100 (a decided behaviour, see the expression `rsi`), and
`flat_windows` says every one of those values sits on a window with no price movement.

```python
import numpy as np
from raptorbt.analysis import Session

n = 100
s = Session()
s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9, "close": np.full(n, 250.0)},
       bar_stamp="close")
r = s.eval("rsi(close, 14)")
assert r["dist"]["p50"] == 100.0
assert r["flags"] == {"flat_windows": n - 14}
```

A boolean result has no distribution, only how often it is true. Several outputs come back as one
summary each, with their own handles.

```python
import numpy as np
from raptorbt.analysis import Session

n = 200
rng = np.random.default_rng(4)
bars = {"timestamps": np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9,
        "close": 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))}
s = Session()
s.load(bars, bar_stamp="close")

above = s.eval("close > sma(close, 20)")
assert above["kind"] == "bool" and "dist" not in above
assert above["known"] == n - 19 and 0 < above["true_rate"] < 1

macd = s.eval("macd(close)")
assert [o["handle"].split("_", 2)[2] for o in macd["outputs"]] == ["macd", "macdsignal", "macdhist"]
line = macd["outputs"][0]["handle"]
assert s.eval(f"sma({line}, 5)")["warmup"] == macd["outputs"][0]["warmup"] + 4
```

A statistic on a handle. `stationarity` needs 50 values; the warm-up of the handle is cut for you.

```python
import numpy as np
from raptorbt.analysis import Session

n = 400
rng = np.random.default_rng(5)
x = np.zeros(n)
for i in range(1, n):
    x[i] = 0.8 * x[i - 1] + rng.normal()
s = Session()
s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9, "close": 100.0 + x},
       bar_stamp="close")
dev = s.eval("close - sma(close, 20)")["handle"]
r = s.stats("stationarity", dev)
assert r["reading"] == "stationary"
assert r["analysis"] == {"handles": [dev], "trimmed_leading": 19}
```

The session is bounded. A handle that was evicted names the call that recomputes it.

```python
import numpy as np
import raptorbt.errors as errors
from raptorbt.analysis import Session

n = 50
bars = {"timestamps": np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9, "close": np.linspace(1.0, 2.0, n)}
s = Session(max_handles=2)
name = s.load(bars, bar_stamp="close")["data"]
first = s.eval("sma(close, 3)")["handle"]
s.eval("sma(close, 4)")
s.eval("sma(close, 5)")
try:
    s.series(first)
except errors.RaptorBTError as err:
    assert err.code == "handle_expired"
    assert err.valid == [f'Session.eval("sma(close, 3)", data={name!r})']
```

An invented handle is refused, with the nearest real ones.

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt.analysis import Session

s = Session()
s.load({"timestamps": np.arange(1, 11, dtype=np.int64), "close": np.arange(10.0)}, bar_stamp="close")
s.eval("zscore(h_0000000000, 5)")
```

The same loop from a shell. A first call loads the file; later calls name only handles. Each
command prints JSON.

```python
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    rng = np.random.default_rng(6)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 120))
    lines = ["timestamp,close"] + [f"{(i + 1) * 60 * 10**9},{float(c)!r}" for i, c in enumerate(close)]
    (tmp / "bars.csv").write_text("\n".join(lines) + "\n")

    def cli(*args):
        done = subprocess.run([sys.executable, "-m", "raptorbt", *args, "--cache-dir", str(tmp / "cache")],
                              capture_output=True, text=True)
        return done.returncode, json.loads(done.stdout)

    code, first = cli("eval", "rsi(close, 14)", "--data", str(tmp / "bars.csv"), "--bar-stamp", "close")
    assert code == 0 and first["warmup"] == 14
    code, second = cli("eval", f"zscore({first['handle']}, 20)")
    assert code == 0 and second["warmup"] == 33
    code, bad = cli("eval", "smaa(close, 3)")
    assert code == 2 and bad["code"] == "expr_unknown_function" and "sma" in bad["valid"]
```

Several instruments: a panel. `load_panel` takes what `Expr.eval_panel` takes (2-D arrays, dates
by names, or a mapping of instrument to its own bars). A cross-sectional expression returns a
panel summary that names instruments: coverage per date (names with a value: min, median, max),
and the last cross-section as the names at both ends. Panel handles chain like any other, and
`factor` scores one against the returns that follow it.

```python
import numpy as np
from raptorbt.analysis import Session

n, m = 160, 30
rng = np.random.default_rng(7)
drift = rng.normal(0.0, 0.001, m)                      # each name's drift persists
close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.01, (n, m)), axis=0))
names = [f"N{j:02d}" for j in range(m)]

s = Session()
s.load_panel({"timestamps": np.arange(1, n + 1, dtype=np.int64) * 86_400 * 10**9, "close": close},
             bar_stamp="close", instruments=names)
rank = s.eval("rank(roc(close, 20))")
assert rank["shape"] == [n, m] and rank["coverage"] == {"min": m, "median": float(m), "max": m}
best = rank["last"]["top"][0]                          # [name, value] at the last date
assert best[0] == names[int(np.argmax(s.series(rank["handle"])[-1]))] and best[1] == 1.0

z = s.eval(f"zscore_xs({rank['handle']})")             # chain the panel handle
f = s.factor(z["handle"], horizons=(1, 5), quantiles=5)
assert f["no_costs"] is True and f["horizons"]["5"]["hac_lags"] == 4
assert set(f["horizons"]["5"]) >= {"ic_mean", "t_hac", "t_naive", "hit_rate", "long_short_mean", "long_short_t_hac"}
ic = s.series(f"{f['handle']}_ic_5")                   # the per-date arrays stay behind the handle
assert ic.shape == (n,)
assert s.series(z["handle"], tail=1, instruments=["N03", "N01"]).shape == (1, 2)
```

Try a family of settings in one call. The table gives one row per setting with its handle and
score, and `n_trials`: the best of them is the best of that many, not a setting chosen in
advance, so carry the count to `raptorbt.research.deflated_sharpe` with the Sharpe of every
trial. A row's handle is the one `eval` of the bound text returns.

```python
import numpy as np
from raptorbt.analysis import Session

n = 300
rng = np.random.default_rng(3)
close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
bars = {"timestamps": np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9, "close": close}

s = Session()
s.load(bars, bar_stamp="close")
t = s.sweep("sma(close, $n) > sma(close, $m)", n=[5, 10], m=[30, 60])
assert t["n_trials"] == 4 and t["columns"] == ["handle", "n", "m", "score"]
assert t["score"] == "true_rate" and len(t["rows"]) == 4
first = t["rows"][0]
assert first[1:3] == [5, 30]
assert s.eval("sma(close, 5) > sma(close, 30)")["handle"] == first[0]

by_distance = s.sweep("sma(close, $n)", n=[5, 20], score="abs(close - sma(close, $n))")
assert by_distance["score"] == "abs(close - sma(close, $n))" and by_distance["n_trials"] == 2
```

## 6. Composition

- [Expressions](../expressions/SKILL.md): the language, its functions and their warm-up. A
  handle is a numeric series there; a boolean handle is read as 1.0 or 0.0 and compared with
  `h > 0.5`.
- [Statistics](../statistics/SKILL.md): which test answers which question. `Session.stats` feeds
  it handles; the skill says what each result means and which parts are not causal.
- [Strategy](../strategy/SKILL.md) and [Backtesting](../backtesting/SKILL.md): register the
  expression you explored with `ctx.indicators.expr(text, instrument=...)` and run it with
  `run_strategies(..., bar_stamp="close")`. The strategy's value on every bar is the series
  `eval` returned, bit for bit (`examples/analysis/expression_to_strategy.py`).
  Handles do not exist inside a strategy: write the expression out in full there.
- [Troubleshooting](../troubleshooting/SKILL.md): `unknown_handle`, `handle_expired` and
  `session_memory_exceeded` are listed there.
- Examples: `examples/analysis/factor_research.py` (a panel: rank, chain, score against
  forward returns), `examples/analysis/stationarity_to_signal.py` (stationarity, half-life, a signal) and
  `examples/analysis/expression_to_strategy.py` (the same text in analysis and in a backtest).

## 7. Constraints and correctness

- **Handles are content-addressed.** The id hashes the canonical expression, the data (every
  column, the bar stamp and the bar width) and the engine version. A different spelling of the
  same expression is the same handle; the same text on other bars is another. A handle computed
  on other bars cannot be mixed into an expression over these (`invalid_input`).
- **A session is one process's memory** (and, from the command line, one directory). It is not
  shared between users, and it is bounded: never assume a handle is still there; the error tells
  you how to remake it.
- **Closed bars.** What you load is what the expression sees. Load closed bars only; an
  unfinished last bar is a value that will change.
- **`bar_stamp` is not optional.** Whether a timestamp starts or ends its bar decides which
  daily window a bar belongs to for `at(...)`; there is no default.
- **Warm-up.** Values before `warmup` are NaN. Reading a distribution without checking `warmup`,
  `nan` and `short_history` first is the usual mistake.
- **Flags are facts about this series, not verdicts.** `flat_windows` says the inputs did not
  move; whether an RSI of 100 there is wrong is for the reader.
- **The summary rounds, the store does not.** Compare with `sha256` of the stored series, or read
  it with `series`.
- **Statistics.** A test over a window describes the window. Smoothed regime probabilities are
  not causal; see the statistics skill before using one as a signal.
- **A factor report is gross.** No cost is charged (`no_costs: true`); a spread that looks small
  gross is smaller net. The forward return of a date starts `lag` bars later (default 1), so a
  signal computed at a close is not scored on that close. The daily ICs of an `h`-bar return
  overlap, so `t_naive` is too large by about the square root of `h`: read `t_hac`. An IC over
  fewer than 30 scored dates has no `t_hac`.
- **Panel handles** belong to the panel they were computed on, in its column order. A factor
  report is read with `series`, never inside an expression. `Session.stats` runs on one series:
  pick a name with `series(..., instruments=...)` and pass the array to `raptorbt.stats`.
- **The command line.** One command at a time per session (a lock). Sessions unused for 7 days
  are deleted, and the oldest go first past 1 GiB of cache. Session and dataset names are letters,
  digits, `_` and `-`; nothing else becomes a path.

## 8. Common mistakes

- Leaving out `bar_stamp`: `missing_bar_stamp`.
- Using a handle after the session dropped it: `handle_expired` (read `valid` for the recipe), or
  a made-up one: `unknown_handle`.
- Using the group handle of a multi-output result (`h_...` of `macd(close)`) as a series: name
  an output, `h_..._macdsignal`.
- Reading `dist` of a boolean: there is none, use `true_rate`.
- Judging an indicator by its distribution while `flags` shows `flat_windows`.
- Passing a DataFrame index as timestamps: put them in a `timestamps` column, int64 nanoseconds.
- Expecting `Session.stats` to accept NaN: warm-up is cut for you, but a NaN inside the series is
  refused by the test.
- Writing a handle in a strategy: it exists only in the session.
- Taking the best row of a `sweep` as the answer: it is the best of `n_trials`. Report the count and deflate with `raptorbt.research.deflated_sharpe`.
- A sweep over a panel dataset: `expr_unsupported_here`; sweep the arrays with `Expr.sweep`.
- Reading `t_naive` of a long horizon as evidence: the overlap inflates it; `t_hac` corrects it.
- Reading a factor's long-short spread as a return you could keep: it has no costs.
- An instrument name with a different spelling than the panel's: `unknown_symbol` lists the
  nearest names.

## 9. Validation

1. Evaluate the expression twice and compare the handles: equal means the same work.
2. Compare `sha256` of `Session.series` with the summary's.
3. Before a strategy uses the text, run it through `ctx.indicators.expr` and compare the values
   on every bar with the series (`examples/analysis/expression_to_strategy.py`).
4. Check `warmup` against the first known index of `series`.

## 10. Related skills

- [Core](../core/SKILL.md)
- [Expressions](../expressions/SKILL.md)
- [Statistics](../statistics/SKILL.md)
- [Strategy](../strategy/SKILL.md)
- [Backtesting](../backtesting/SKILL.md)
- [Troubleshooting](../troubleshooting/SKILL.md)
