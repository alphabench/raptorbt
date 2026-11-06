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
