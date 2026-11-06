---
name: raptorbt-expressions
description: >
    Writing a signal as one formula (close > sma(close, 200)), as text or from Python
    operators, and evaluating it over whole columns or one bar at a time with the same numbers.
    Covers unknown values (NaN and Kleene logic), warm-up, no-lookahead and the function list.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
    - raptorbt-indicators
status: stable
---

# RaptorBT Expressions

## 1. When to use this skill

Use it to write a condition or a derived series as one formula: `close > sma(close, 200)`,
`ema(close, 12) - ema(close, 26)`. Use it when the same formula must give the same numbers over
a stored history and bar by bar on a live feed, or when two spellings of a formula must be
recognised as one (they share a canonical text and a hash).

Do not use it for order logic, sizing or state that depends on fills: an expression is a pure
function of the bars it has seen. In a strategy, register one on one instrument with
`ctx.indicators.expr`; the existing indicator functions and `raptorbt.Indicator` are unchanged.

## 2. Domain overview

An expression is text over five columns (`open`, `high`, `low`, `close`, `volume`), numbers,
the operators `+ - * / **`, the comparisons `< <= > >= == !=`, `and`, `or`, `not`, and calls to
registered functions. There are no variables, assignments, loops or user functions. `$name`
stands for a constant given in `params`.

Every value is a number or NaN (unknown). A boolean is `1.0`, `0.0` or NaN. Division by zero,
`ln` of a non-positive number and overflow give NaN, never an infinity.

Compile once, evaluate many times: `raptorbt.expr(text)` checks the text against the function
registry and returns an `Expr` with a canonical text and a hash. `Expr.eval` runs it over
columns; `Expr.stream` runs it one bar at a time. The two give bit-identical numbers.

The same formula can be built from Python operators. `raptorbt.col.close` is the close column;
every registered function is a method (`col.close.sma(200)`) and a builder (`col.sma(200)`). The
methods are generated from the registry when the package is imported, so a function added to
the engine appears with no other change.

## 3. API reference

| Name | What it does |
| --- | --- |
| `raptorbt.expr(source, params=None)` | Compile text. Raises `ExpressionError` with `code`, `position`, `snippet` and `valid`. |
| `raptorbt.col` | The columns (`col.open`, `high`, `low`, `close`, `volume`) and a builder for every function. |
| `raptorbt.expressions.Expr` | A checked expression. |
| `raptorbt.expressions.ExprStream` | The bar-by-bar evaluator from `Expr.stream()`. |
| `Expr.eval_panel(data, instruments=None, present=None, ...)` | Evaluate over many names: 2-D arrays (dates by names), or a mapping of instrument to its own bars. Returns 2-D arrays. See "Panels". |
| `raptorbt.col.group(name)` / `raptorbt.expr(text, groups=[...])` | A group series (a numeric label per name) read by `group_rank`, `group_zscore`, `group_demean`. |
| `raptorbt.expressions.factor_report(factor, prices, horizons=(1, 5, 20), quantiles=5, lag=1, ...)` | Rank IC, quantile returns, long-short spread, turnover and coverage of a factor. See "Panels". |
| `raptorbt.expressions.functions(query=None)` | The functions, optionally filtered by a word in the name, family or description. |
| `raptorbt.expressions.describe(name)` | One function in full: inputs, parameters, outputs, warm-up, NaN handling, legacy pin. |
| `ctx.indicators.expr(expression, instrument=, params=None, output=None, timeframe=None)` | Register an expression (text or `Expr`) on one instrument's bars, or on a `timeframe`'s composite bars, in `on_init` or `on_start`. Returns a handle: `.ready`, `.value`, `.warmup`, `.settle(eps)`, `.canonical`. |

`Expr` members:

- `canonical`, `hash`: identity. Parentheses, the order of commutative operands, spaces and
  `5` against `5.0` do not change them; the arrangement of non-commutative operators does.
- `eval(data)`: `data` is a dict of 1-D float64 numpy arrays of one length keyed by field name.
  Returns an array, or a dict of arrays by output name when the expression has several outputs.
  Keys that are not fields are ignored; an array of another dtype is refused. Inputs are never
  modified.
- `stream()`: an `ExprStream`. `update(close=..., ...)` takes the fields the expression reads
  (a mapping or keywords) and returns the value for that bar; `ready`, `bars`, `reset()`.
- `warmup`: leading bars that are NaN on clean data. `settle(eps)`: bars until an `ema` has
  forgotten its seed to within `eps`.
- `outputs`, `fields`: the output names and the columns read.
- Operators `+ - * / **`, comparisons, `&`, `|`, `~` and unary `-`. A call with several outputs
  is selected by attribute (`x.fn().name`).

The registered functions, with signatures, warm-up and NaN handling, are in
[functions.md](functions.md), generated from the registry.
`raptorbt.capabilities()["functions"]` lists the same rows.

## 4. Standard workflow

1. Look up what exists: `raptorbt.expressions.functions("average")`, then
   `raptorbt.expressions.describe("sma")`.
2. Write the formula as text with `raptorbt.expr`, or build it with `raptorbt.col`.
3. Read `warmup`: no value before that index means anything.
4. Evaluate over history with `eval`, and feed live bars to `stream()`.
5. Check the two agree on a stretch of history before trusting the live path.

## 5. Examples

Two spellings of one formula, evaluated over a short close series. The first two bars have no
`sma(close, 3)` yet: the comparison is unknown there, so `and close > 1` is unknown on the
second bar, but false on the first, where `close > 1` is false and a false operand decides an
`and`.

```python
import numpy as np
import raptorbt as rb

a = rb.expr("sma(close, 3) < close and close > 1")
b = (rb.col.close.sma(3) < rb.col.close) & (rb.col.close > 1)
assert a.canonical == b.canonical and a.hash == b.hash

close = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 5.0])
signal = a.eval({"close": close})
assert signal[0] == 0.0 and np.isnan(signal[1])
assert list(signal[2:]) == [1.0, 1.0, 0.0, 0.0, 1.0]
assert a.warmup == 2
```

Streaming gives the same bits as the batch run, and a prefix of the history gives a prefix of
the result: a value never depends on a later bar.

```python
import numpy as np
import raptorbt as rb

a = rb.expr("sma(close, 3) < close and close > 1")
close = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 5.0])
full = a.eval({"close": close})

stream = a.stream()
live = np.array([stream.update(close=c) for c in close])
assert np.array_equal(live, full, equal_nan=True)
assert stream.ready and stream.bars == 7

for n in range(1, len(close) + 1):
    assert np.array_equal(a.eval({"close": close[:n]}), full[:n], equal_nan=True)
```

Warm-up and settling. An `ema` is seeded with the mean of its first `period` values; its seed
fades but never reaches zero, so `settle` says when it is below a tolerance.

```python
import raptorbt as rb

e = rb.expr("ema(close, 10)")
assert e.warmup == 9
assert e.settle(1e-6) == 78
```

A refusal says where and what would have worked.

```python
import raptorbt as rb

try:
    rb.expr("1 + smaa(close, 5)")
except rb.errors.ExpressionError as err:
    assert err.code == "expr_unknown_function"
    assert err.position == 4
    assert "sma" in err.valid
    print(err.snippet)
```

Python's `and`, `or`, `not` and a chained comparison would keep one operand and drop the other,
so an `Expr` refuses a truth test.

<!-- skill: invalid -->
```python
import raptorbt as rb

c = rb.col.close
if 1 < c < 3:
    pass
```

In a strategy. `ctx.indicators.expr` registers the expression on the shared indicator bank, so
two strategies with one canonical expression on one stream share one cell. The handle shows what
`Expr.eval` gives on the same bars, at the same bar, in a backtest and on a `LiveNode`. A boolean
expression reads as `True`, `False` or `None` (unknown), a numeric one as a float or `None`.
Like the built-in indicators, the value in `on_bar` includes the bar being handled, and a hook
that runs before the strategy has seen a bar (a fill at the next bar's open) sees the value
before it.

```python
import numpy as np
import raptorbt as rb
from raptorbt import Strategy, run_strategies

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 80
rng = np.random.default_rng(5)
close = 100.0 + np.cumsum(rng.normal(0, 1, n))
open_ = np.concatenate([[close[0]], close[:-1]])
bars = dict(timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
            open=open_, high=np.maximum(open_, close) + 0.3,
            low=np.minimum(open_, close) - 0.3, close=close,
            volume=np.round(rng.uniform(500, 1500, n)))

TEXT = "zscore(close, 10) < -1 and sma(volume, 5) > 800"

class Dip(Strategy):
    def on_start(self, ctx):
        self.sig = ctx.indicators.expr(TEXT, instrument="ASSET")
        self.seen = []

    def on_bar(self, ctx):
        self.seen.append((self.sig.ready, self.sig.value))

s = Dip()
run_strategies([s], {"ASSET": bars}, bar_stamp="close")

program = rb.expr(TEXT)
want = program.eval(bars)
assert s.sig.warmup == program.warmup
for i, (ready, got) in enumerate(s.seen):
    assert ready == (i >= program.warmup)
    assert got is None if np.isnan(want[i]) else got is bool(want[i])
```

A coarser clock with `at`. `at("1d", expression)` computes `expression` on daily bars cut from
the bars you give and holds the result on the base clock, so `close > at("1d", sma(close, 200))`
compares every bar with the daily average. The daily value first shows on the first bar of the
next day and stays all that day. The expression needs the bars' timestamps and to be told what
they mean:

```python
import numpy as np
import raptorbt as rb

DAY = 86_400 * 10**9
# Three days of four 6-hour bars, each stamped at its close; the fourth of a day is stamped at
# midnight and still belongs to the day it ends.
ts = 19_000 * DAY + (np.arange(12, dtype=np.int64) + 1) * 6 * 3600 * 10**9
close = np.arange(1.0, 13.0)

e = rb.expr('at("1d", close)')
held = e.eval({"timestamps": ts, "close": close}, bar_stamp="close")
assert np.isnan(held[:4]).all()                  # nothing is complete on the first day
assert list(held[4:]) == [4.0] * 4 + [8.0] * 4   # each day shows the previous day's close
assert e.warmup is None and e.warmup_by_clock == {"base": 0, "1d": 0}
```

Many names at once. A cross-sectional function (`rank`, `zscore_xs`, `demean_xs`, `winsorize_xs`,
`quantile_xs`, `count_xs`, and `group_rank`, `group_zscore`, `group_demean`) compares the names
of each date, so the expression has a value only over a panel and is evaluated with `eval_panel`:
arrays of shape `(n_dates, n_assets)`. Time-series functions run per name inside it.

```python
import numpy as np
import raptorbt as rb

rng = np.random.default_rng(7)
close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (60, 8)), axis=0)

r = rb.expr("rank(roc(close, 5))").eval_panel({"close": close})
assert r.shape == (60, 8)
assert np.isnan(r[:5]).all()                          # roc has no value yet
assert (r[5:].min(axis=1) == 0).all() and (r[5:].max(axis=1) == 1).all()   # [0, 1] each date

# Group neutral: remove each sector's mean on each date. A group series is read by name.
sector = np.array([0.0, 0.0, 1.0, 1.0])
e = rb.expr("group_demean(close, sector)", groups=["sector"])
out = e.eval_panel({"close": np.array([[1.0, 3.0, 10.0, 30.0]]), "sector": sector})
assert out.tolist() == [[-1.0, 1.0, -10.0, 10.0]]
```

A name that is not listed yet. Give each instrument its own bars with timestamps and the panel is
aligned on the union of the dates; a name is absent where it has no bar. An absent bar does not
step `sma`, so a name listed on date 3 counts its three bars from its own first one.

```python
import numpy as np
import raptorbt as rb

ts = (np.arange(8, dtype=np.int64) + 1) * 86_400 * 10**9
data = {
    "AAA": {"timestamps": ts, "close": np.arange(1.0, 9.0)},
    "BBB": {"timestamps": ts[3:], "close": np.arange(10.0, 15.0)},
}
out = rb.expr("sma(close, 3)").eval_panel(data)         # rows are the 8 dates
assert out.shape == (8, 2)
assert out[2, 0] == 2.0                                 # AAA: mean of 1, 2, 3
assert np.isnan(out[:5, 1]).all() and out[5, 1] == 11.0  # BBB: its third bar is date 5
```

Scoring a factor. The forward return of a signal date starts `lag` bars later (1 by default: a
signal computed at a close cannot trade that close). The signal below is the return of the bar
after next, which is what a horizon-1 return with `lag=1` measures, so its rank IC is 1.

```python
import numpy as np
import raptorbt as rb

rng = np.random.default_rng(1)
ret = rng.normal(0, 0.01, (120, 30))
close = 100.0 * np.cumprod(1.0 + ret, axis=0)
factor = np.full_like(close, np.nan)
factor[:-2] = ret[2:]

rep = rb.expressions.factor_report(factor, close, horizons=[1])
assert rep["no_costs"] is True and rep["lag"] == 1
h = rep["horizons"][1]
assert abs(h["ic_mean"] - 1.0) < 1e-12 and h["hit_rate"] == 1.0
assert h["long_short_mean"] > 0
```

## 6. Composition

- [Indicators](../indicators/SKILL.md): `sma` and `ema` here are pinned bitwise, on finite input,
  to the streaming cores behind `raptorbt.Indicator.sma` and `raptorbt.Indicator.ema`, so a
  strategy that reads the indicator and an expression that reads the formula see the same
  numbers. They differ when an input is NaN (below).
- Evaluate an expression over bars you hold, or stream it from your own loop.
- Register one on a strategy with `ctx.indicators.expr` (see Indicators). It reads one
  instrument's bars; the volume it reads is the bar's own, on the primary stream and on a
  `timeframe=` stream alike.
- [Core](../core/SKILL.md): errors are `raptorbt.errors.RaptorBTError`; an expression refusal
  is an `ExpressionError`, also a `ValueError`.

## 7. Constraints and correctness

**Unknown values.** Values are finite or NaN, and NaN means unknown.

- A comparison with an unknown side is unknown.
- Kleene logic: `false and unknown` is false and `true or unknown` is true, because the known
  side decides; otherwise an unknown operand makes the result unknown. `not unknown` is unknown,
  so `not (sma(close, 5) < 30)` is never true while the average is still warming up.
- A windowed function (`sma`) is NaN while a NaN is inside its window and recovers when it
  leaves. A recursive function (`ema`) outputs NaN for a NaN input and does not advance its
  state, so one bad bar does not poison the rest. Legacy indicators handle NaN their own way;
  each function's registry row (`describe`) says how, and `functions.md` repeats it.

**Warm-up.** Output before the warm-up index is NaN. The warm-up of a call is the largest warm-up
of its inputs plus its own lookback, so `sma(sma(close, 5), 5)` has 8 leading NaN bars.
`warmup` is exact on clean data. A stream reports `ready` once it has seen more bars than the
warm-up.

**No lookahead.** Nothing in the language refers to a later bar. Batch evaluation reads each
bar from the prefix up to it, streaming sees only bars already fed, and both run the same code
in the same order, which is why they are equal to the last bit. A result for the first `n`
bars does not change when more bars are appended.

**Another timeframe (`at`).** `at("1d", expression)` runs `expression` on windows cut from the
base bars by the same code the runtime uses for `timeframe="1d"` streams, and holds the value on
the base clock.

- *When a value shows.* A window's bar is complete when the first bar of a later window arrives,
  and only then does the value change: the first bar of the next day (the next trading day when
  a `session` is given: its closed days are dropped), in `eval`, `stream`, `run_strategies` and
  a live node alike. Nothing emits at a session close, so the value that closes a day is not
  visible on that day. The window in progress is never read, and the last, partial window of a
  run never appears. Before the first window completes, and until the expression's own warm-up
  has passed, the value is NaN.
- *Inner fields.* `close`, `high`, `low`, `open` and `volume` inside `at` are the window's:
  its last close, highest high, lowest low, first open and total volume. Inside `at` the
  expression is its own program: `close > at("1d", close)` compares the base `close` with the
  daily one, and nothing inside is shared with the outside, except an identical `at(...)`, which
  is one node.
- *Mixing.* The result of `at` is on the base clock, so it combines with base-clock values:
  `close > at("1d", sma(close, 200))`. One level only: `at` inside `at` is `expr_unsupported_here`.
  The timeframe is a string such as `"15m"`, `"1h"`, `"1d"`, `"1w"`; anything else is `expr_type`
  with the valid ones in `valid`. Give `bar_ns=` (the base bar width) to have a timeframe that is
  not a coarser exact multiple refused.
- *What it needs.* `eval` needs `data["timestamps"]` (int64 nanoseconds) and `bar_stamp="open"`
  or `"close"` (`missing_field`, `missing_bar_stamp`); `stream` needs `bar_stamp=` and each
  `update` a `timestamp=`. `tz_offset_ns` is the zone daily windows are cut in (default UTC, 0;
  IST is `19_800 * 10**9`) and `session=` a `raptorbt.SessionSchedule`, whose own zone replaces
  it and whose closed days are skipped. Inside `ctx.indicators.expr` the run's zone and bar
  stamp and the instrument's session are used. Hash, canonical text and cell sharing of a
  program without `at` are unchanged; `bound_hash(...)` folds the zone, stamp and session into
  the hash of one with `at`.
- *Warm-up.* It is reported per clock: `warmup_by_clock` is `{"base": n, "1d": m}`. `warmup` is
  `None` for a program with `at`, because the base bars a daily warm-up costs depend on how many
  bars a day holds; `warmup_bars(bar_ns)` gives the most it can be (`(m + 1)` windows of at
  most `ceil(window / bar_ns)` bars each, ignoring the session) and is `None` for a month or a
  year. `ready` is true once every clock is past its warm-up.
- *Cost.* A step is one window push and, once a window completes, one step of the inner stream,
  with no allocation of its own. With a session the window builder allocates per bar, as the
  runtime's does.

**Panels.** A panel is a 2-D array, dates by names, NaN where a value is unknown.

- *Shape.* Any cross-sectional call makes the expression a panel; time-series functions lift to
  every name; a constant lifts. A cross-sectional function needs a series or panel input (a
  constant has no names): `expr_type`. Evaluating a panel expression with `eval` or `stream`, or
  registering it with `ctx.indicators.expr` (sealing a panel on the bank is a later slice), is
  `expr_unsupported_here`. So is a cross-sectional function inside `at(...)`, which evaluates one
  instrument's bars. `at(...)` elsewhere in a panel expression works per name and reads
  `data["timestamps"]` (one int64 per date).
- *Which names count.* A name takes part in a date's comparison when it has a bar and its input
  is finite. The others are excluded, take no rank and give NaN. `min_names` (default 2, a
  parameter of each function): with fewer participants the whole date is NaN, and for the
  `group_*` forms that applies to each group. `rank` is `(average rank - 1) / (n - 1)` in
  [0, 1], ties sharing the mean rank; `zscore_xs` uses the population standard deviation;
  `winsorize_xs(x, pct)` clips the `floor(n * pct)` lowest and highest. These are the definitions
  of `portfolio::factor_panel` (`rank_panel`, `zscore_panel`, `winsorize_panel`), which share the
  code. `quantile_xs` and `count_xs` give the date's statistic to every present name.
- *Causality.* A cross-sectional value at a date uses only that date's values of series that
  are already causal; nothing normalises across dates. A result for the first `n` dates does not
  change when later dates change.
- *Absent bars.* An absent bar does not step a time-series function; a bar that is present with
  a NaN value does. In a panel the two look alike, so the input says which: with the mapping form
  every instrument's own timestamps do (a name is absent at the dates it has no bar), and with
  2-D arrays `present=` does. Without it a cell is absent when every field the expression reads
  is NaN there, so a NaN close counts as no bar. Either way a name listed late counts its warm-up
  from its own first bar, and is NaN at the dates it is absent.
- *Evaluation.* Time-series phases run per name in parallel, cross-section phases per date in
  parallel, alternating in the order the expression nests; the GIL is released, and the numbers
  are the same for any thread count. Each name equals a stream fed its own present bars, and each
  date equals the plain function of that date's row, bit for bit.
- *Factor evaluation.* `factor_report` is the only place a forward return exists; nothing in
  it can feed an expression. It reports `no_costs: True`: every spread is gross. The t-statistic
  of the IC and of the spread is Newey-West with `h - 1` lags (the daily ICs of an `h`-bar return
  overlap; the naive t is about `sqrt(h)` too large and is reported beside it as `t_naive`), and
  is NaN with fewer than 30 scored dates. It refuses (`invalid_input`) fewer than
  `max(min_names, quantiles)` names or fewer than `lag + max(horizons) + 30` dates.
- *Limits.* Group labels are read as numbers, equal labels are one group, a NaN label leaves the
  name out. Panels are held in memory as dates times names floats per intermediate; the group
  forms and `quantile_xs` sort each date. There are no cross-sectional functions across
  timeframes (one clock per panel).

**Periods and parameters.** A period is a constant integer between 1 and 16,777,216 (or a
`$name` bound in `params`); a series cannot be a period. Depth and size are limited (64 levels,
4,096 nodes).

**What `compat` means.** New functions follow TA-Lib's conventions. A function whose raptorbt
namesake differs from TA-Lib's lists a `compat` parameter among its parameters. It is part of
the program hash, so changing it changes the identity of the expression. A function with no
`compat` in `describe(name)["params"]` has one definition. Whether a function matches an older
raptorbt indicator, and to what tolerance, is its `legacy` row in the registry.

**Refusals.** A construct the language names but the engine does not run yet is refused with
`expr_unsupported_here`; the message says which. So is `ctx.indicators.expr` in
`run_strategy_backtest`, which has no shared runtime: use `run_strategies` or
`run_portfolio_strategy`.

**In a strategy.** An expression with several outputs needs one picked, with `output="macdsignal"` or
in the text (`macd(close).macdsignal`); otherwise `expr_type` lists the names. `.values` does not
exist on an expression handle. A recursive function (`ema`) keeps settling after `.warmup`:
`.ready` is true from the first bar the value can be known, and the value approaches the one a
longer history would give as `settle(eps)` bars pass. A manifest's `warmup=Bars(n)` is the
author's to set; use `n > handle.warmup` (or `rb.expr(text).warmup` at class definition) so the
node is primed past it. A node restored from its journal replays the same bars and gives the
same expression values as one that never stopped.

## 8. Common mistakes

- `if a < b:`, `a and b`, `not a`, `a < x < b` on expressions: each raises `expr_type`. Write
  `(a < b) & (b < c)` and `~a`.
- A series where a constant belongs, such as `sma(close, close)`: `expr_type` at the argument.
- A misspelt function or column: `expr_unknown_function` or `expr_type`, with `valid` holding
  the closest names.
- `sma(close, $n)` without `params={"n": 20}`: `expr_unbound_parameter`.
- An integer array passed to `eval`: refused with `invalid_array`; convert with
  `np.asarray(x, dtype=np.float64)`.
- Treating warm-up NaN as false: unknown is not false. Test with `np.isnan` or compare from the
  `warmup` index.
- Hashing an `Expr`: it defines `==` as an expression, so it is not hashable. Use `.hash`.
- `ctx.indicators.expr("macd(close)", ...)` with no `output=`: pick one output.
- Registering an expression in `run_strategy_backtest`: `expr_unsupported_here`.
- Expecting a daily value on the day it is made: `at("1d", close)` shows yesterday's close all
  day. To act on today's close, read the base bar.
- `rb.expr("rank(close)").eval(...)`: a cross-sectional expression needs `eval_panel`
  (`expr_unsupported_here`). Passing 1-D arrays to `eval_panel` is `invalid_array`.
- Reading a NaN close as a bar that exists: without `present=` a cell whose fields are all NaN is
  absent, and does not step a function.
- Quoting a factor's `t_naive` for a 5- or 20-day horizon: it ignores that the daily ICs overlap.
  Use `t_hac`.
- Evaluating `at(...)` without timestamps or `bar_stamp`: `missing_field` / `missing_bar_stamp`.
  Nothing is assumed; open- and close-stamped bars cut different windows near a boundary.
- Cutting IST days in UTC: pass `tz_offset_ns=19_800 * 10**9` (or a `session`), or a bar at
  01:00 IST lands on the previous date.
- A hand-fed `Indicator.expr(...)` whose expression uses `at`: `update_bar` has no timestamp,
  so it is `expr_unsupported_here`. Use `ctx.indicators.expr` or `Expr.stream(...).update(..., timestamp=)`.
- A hand-fed `Indicator.expr(...)` whose expression reads `volume`, given a bar with four
  prices: `missing_field`. Feed `update_bar(open, high, low, close, volume)`.

## 9. Validation

1. Compare the two spellings: `a.canonical == b.canonical`.
2. Evaluate on a stretch of history, stream the same bars and assert
   `np.array_equal(live, batch, equal_nan=True)`.
3. Check the first non-NaN index equals `warmup` on clean data.
4. Evaluate on a prefix and assert it equals the prefix of the full result.
5. `python scripts/gen_expr.py --check` fails when `functions.md` is stale.

## 10. Related skills

- [Core](../core/SKILL.md)
- [Indicators](../indicators/SKILL.md)
- [Troubleshooting](../troubleshooting/SKILL.md)
