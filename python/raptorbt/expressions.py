"""Expressions: signals written as one formula, evaluated by one engine.

An expression is text such as ``rsi(14) < 30 and close > sma(200)``, or the same thing built
from Python operators: ``(col.close.rsi(14) < 30) & (col.close > col.close.sma(200))``. Both
compile to one checked program with one canonical text and one hash; the program evaluates over
whole columns (:meth:`Expr.eval`) or one bar at a time (:meth:`Expr.stream`), and the two give
bit-identical numbers.

Values are finite or NaN. A boolean is ``1.0``, ``0.0`` or NaN (unknown); ``and``, ``or`` and
``not`` follow Kleene logic, so ``not (x < 30)`` is unknown, never true, while ``x`` is still
warming up.

The function list, the ``col`` methods and the reference page ``skills/expressions/functions.md``
are all generated from the engine's function registry: a function added there appears here
with no change to this module.

A formula can look at a coarser clock with ``at("1d", ...)``: the inner expression is computed on
daily bars cut from the base bars and held on the base clock, so ``close > at("1d", sma(close,
200))`` compares each bar with the daily average. A daily value first appears on the first base
bar of the next day. Such an expression needs the bars' timestamps and what they mean:
``expr.eval({"timestamps": ts, ...}, bar_stamp="close")``.

Over many names at once, ``rank(roc(close, 20))`` is a cross-sectional expression: it has a
value only over a panel (2-D arrays, dates by names), evaluated with :meth:`Expr.eval_panel`.
Time-series functions run per name; ``rank``, ``zscore_xs``, ``demean_xs``, ``winsorize_xs``,
``quantile_xs``, ``count_xs`` and the ``group_*`` forms compare the names of each date. A
name with no bar at a date is *absent*: it does not step a function, takes no rank and gets NaN.
:func:`factor_report` scores a factor against the returns that follow it.

>>> from raptorbt import col, expr
>>> a = (col.close.sma(3) < col.close) & (col.close > 1)
>>> b = expr("sma(close, 3) < close and close > 1")
>>> a.canonical == b.canonical and a.hash == b.hash
True
"""

from __future__ import annotations

import copy
import difflib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from raptorbt import _raptorbt, errors

__all__ = ["Expr", "ExprStream", "col", "describe", "expr", "factor_report", "functions"]

_FIELDS = ("open", "high", "low", "close", "volume")
_NAN = float("nan")


# ------------------------------------------------------------------ registry


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    rows = json.loads(_raptorbt.expr_registry_json())["functions"]
    return {r["name"]: r for r in rows}


def _signature(row: dict[str, Any]) -> str:
    """``name([input], ..., param=default, ...)`` for docs and messages."""
    parts = [f"[{i['name']}]" if i["default_field"] else i["name"] for i in row["inputs"]]
    for p in row["params"]:
        parts.append(p["name"] if p["default"] is None else f"{p['name']}={p['default']!r}")
    out = row["outputs"]
    tail = "" if len(out) == 1 else f" -> ({', '.join(out)})"
    return f"{row['name']}({', '.join(parts)}){tail}"


def functions(query: str | None = None) -> list[dict[str, Any]]:
    """The functions an expression can call, from the engine's registry.

    ``query`` is one or more words, all of which must appear (case ignored) in a function's
    name, family, TA-Lib name or description: ``"moving average"``, ``"doji"``, ``"rsi"``. With
    a query the best match is first (the exact name, then the TA-Lib name, then a name that
    starts with a word, then a word inside a name, then the family, then the description);
    without one the rows are sorted by name. Each row has ``name``, ``family``, ``signature``,
    ``outputs``, ``talib`` (the TA-Lib name, or ``None``) and ``doc``; :func:`describe` gives
    the rest.
    """
    words = [] if query is None else str(query).lower().replace(",", " ").split()
    ranked = []
    for row in _registry().values():
        score = _match(row, words)
        if words and score is None:
            continue
        ranked.append(
            (
                -(score or 0),
                row["name"],
                {
                    "name": row["name"],
                    "family": row["family"],
                    "signature": _signature(row),
                    "outputs": list(row["outputs"]),
                    "talib": row["talib_name"],
                    "doc": row["doc"],
                },
            )
        )
    # best match first, the shorter (more basic) name before a longer one; sorted by name alone
    # without a query
    ranked.sort(key=lambda t: (t[0], len(t[1]) if words else 0, t[1]))
    return [r for _, _, r in ranked]


def _match(row: dict[str, Any], words: list[str]) -> int | None:
    """How well ``row`` answers every word, or ``None`` when one word is nowhere in it."""
    name, family = row["name"].lower(), row["family"].lower()
    talib = (row["talib_name"] or "").lower()
    doc = str(row["doc"]).lower()
    total = 0
    for w in words:
        if w == name:
            total += 100
        elif w == talib:
            total += 60
        elif name.startswith(w):
            total += 40
        elif w in name:
            total += 30
        elif w == family:
            total += 20
        elif talib and w in talib:
            total += 10
        elif w in doc:
            total += 5
        else:
            return None
    return total


def describe(name: str) -> dict[str, Any]:
    """Everything the registry says about one function: inputs, parameters, outputs, how it
    treats NaN, the legacy indicator it is pinned to, its references, and

    - ``signature``, ``example`` (an expression that compiles and runs on bars, generated from
      the registry) and ``warmup`` (leading unknown bars of the example);
    - ``reference`` (what its numbers were checked against) and ``relation`` (to TA-Lib:
      ``exact``, ``defined_difference``, ``candle``, or ``no_library``);
    - ``conventions``: plain sentences on NaN handling, memory, causality and input domain.
    """
    row = _registry().get(name)
    if row is None:
        raise _refuse(
            "expr_unknown_function",
            f"unknown function {name!r}",
            operation="describe",
            parameter="name",
            value=name,
            valid=_close(name),
        )
    text = _example(row)
    return {
        **copy.deepcopy(row),
        "signature": _signature(row),
        "example": text,
        "warmup": _compile_example(text).warmup,
        "conventions": _conventions(row),
    }


def _compile_example(text: str) -> Expr:
    return Expr(text)


_NAN_POLICY = {
    "windowed": "a NaN in the window makes the value NaN; it recovers once the NaN leaves the window",
    "recursive": "a NaN input gives NaN and does not advance the state, so later bars are not poisoned",
    "elementwise": "a NaN input gives NaN for that bar only",
    "lagged": "reads earlier bars; a NaN there gives NaN",
    "delayed": "reads an earlier bar; a NaN there gives NaN",
}
_DOMAIN = {
    "positive": "inputs must be positive; a non-positive value gives NaN",
    "unit": "inputs must lie in [-1, 1]; a value outside gives NaN",
}


def _conventions(row: dict[str, Any]) -> list[str]:
    out = [
        _NAN_POLICY.get(row["nan_policy"], f"NaN policy: {row['nan_policy']}"),
        "values are finite or NaN; a division by zero, an overflow or a logarithm of a "
        "non-positive number gives NaN, never an infinity",
        "the first values are NaN until the warm-up has passed"
        if row["memory"] == "finite"
        else "the first values are NaN until the warm-up has passed; the seed keeps fading "
        "after that (see Expr.settle)",
        "causal: a value depends only on bars up to its own" if row["causal"] else "NOT causal: a value uses later bars",
    ]
    if row["domain"] in _DOMAIN:
        out.append(_DOMAIN[row["domain"]])
    if row["talib_name"]:
        out.append(f"TA-Lib name {row['talib_name']}; new functions follow TA-Lib's conventions")
    return out


#: Fields offered, in order, to inputs that have no default field of their own.
_SPARE_FIELDS = ("open", "volume", "high", "low")


def _example(row: dict[str, Any]) -> str:
    """A call of ``row`` that compiles and runs on ordinary bars.

    Every input is named, parameters without a default get a value from the registry's own
    range (14 for a period, else 5; 0.5 for a fraction), and a boolean input gets
    ``close > open``. A function whose inputs must lie in [-1, 1] reads ``rsi(close, 5) / 100``.
    """
    spare = iter(_SPARE_FIELDS)
    parts = []
    for i in row["inputs"]:
        if i["elem"] == "bool":
            parts.append("(close > open)")
        elif row["domain"] == "unit":
            parts.append("(rsi(close, 5) / 100)")
        else:
            parts.append(i["default_field"] or next(spare))
    for p in row["params"]:
        if p["default"] is not None:
            continue
        lo, hi = p["range"].get("min"), p["range"].get("max")
        if p["kind"] == "int":
            want = 14 if "period" in p["name"] else 5
            parts.append(str(min(max(want, lo), hi)))
        elif p["kind"] == "float":
            parts.append("0.5" if (lo is None or lo <= 0.5) and (hi is None or hi >= 0.5) else repr(float(lo)))
        else:
            parts.append(_quote(p["range"]["values"][0]))
    return f"{row['name']}({', '.join(parts)})"


def _close(name: str) -> list[str]:
    return difflib.get_close_matches(str(name), list(_registry()), n=5, cutoff=0.5)


def _refuse(code: str, message: str, *, operation: str, **detail: Any) -> errors.ExpressionError:
    detail.setdefault("parameter", "expression")
    if not detail.get("valid"):
        detail.pop("valid", None)
    return errors.make(code, message, operation=operation, **detail)  # type: ignore[return-value]


# ------------------------------------------------------------------ rendering


class _NotANumber(Exception):
    """An operand is not a number: caught where it is built, never seen by a caller."""


def _number(x: Any, operation: str) -> str:
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise _NotANumber
    if isinstance(x, numbers.Integral):
        return str(int(x))
    v = float(x)
    if not math.isfinite(v):
        raise _refuse(
            "expr_syntax",
            f"{x!r} is not a number an expression can hold; use a finite number",
            operation=operation,
            value=v,
        )
    return repr(v)


def _operand(x: Any, operation: str) -> tuple[str, dict[str, float]]:
    """Source text and `$name` bindings of one operand; ``_NotANumber`` when it is neither an
    expression nor a number."""
    if isinstance(x, Expr):
        return x._src, x._params
    text = _number(x, operation)
    return (f"({text})" if text.startswith("-") else text), {}


def _merge(a: dict[str, float], b: dict[str, float], operation: str) -> dict[str, float]:
    for k in a.keys() & b.keys():
        if a[k] != b[k]:
            raise _refuse(
                "expr_type",
                f"parameter ${k} is bound to {a[k]!r} on one side and {b[k]!r} on the other",
                operation=operation,
                parameter="params",
                valid=["one value per $name"],
            )
    return {**a, **b}


def _quote(s: str) -> str:
    return '"' + s + '"'


# ------------------------------------------------------------------ Expr


class Expr:
    """A checked expression: a formula over ``open``, ``high``, ``low``, ``close`` and ``volume``.

    Make one with :func:`expr` (from text) or from :data:`col` and Python operators. Every
    operand is parenthesised when it is rendered, so Python's precedence never changes the
    meaning: ``(a < b) & (c < d)`` is exactly ``a < b and c < d``.

    Use ``&``, ``|`` and ``~``; Python's ``and``, ``or``, ``not`` and chained comparisons would
    silently drop an operand, so an ``Expr`` refuses a truth test with ``expr_type``.

    Functions are methods: ``col.close.sma(20)`` is ``sma(close, 20)``. The methods are generated
    from the function registry. A call with several outputs is selected by attribute:
    ``x.macd().macdsignal``.
    """

    __slots__ = ("_src", "_params", "_prog", "_atom", "_outputs", "_groups")
    #: Make NumPy defer to the reflected operators instead of broadcasting over an Expr.
    __array_ufunc__ = None

    def __init__(
        self,
        source: str,
        params: Mapping[str, float] | None = None,
        *,
        groups: tuple[str, ...] = (),
        _atom: bool = False,
        series: Mapping[str, int] | None = None,
    ):
        if not isinstance(source, str):
            raise _refuse(
                "expr_type",
                f"an expression is built from text, got {type(source).__name__}",
                operation="expr",
                value=repr(source),
                valid=["a string such as 'close > sma(close, 20)'"],
            )
        bound = {} if params is None else dict(params)
        self._src = source
        self._params = bound
        self._atom = _atom
        self._groups = tuple(dict.fromkeys(groups))
        self._prog = _raptorbt.ExprProgram(
            source, bound or None, list(self._groups) or None, dict(series) if series else None
        )
        names = self._prog.outputs
        self._outputs = tuple(names) if len(names) > 1 else ()

    # -- identity

    @property
    def source(self) -> str:
        """The text this expression was made from (a built expression: its full rendering)."""
        return self._src

    @property
    def canonical(self) -> str:
        """The canonical text: equal for every spelling of the same expression."""
        return self._prog.canonical

    @property
    def hash(self) -> str:
        """Hex sha256 of the canonical text and the revisions of the functions it calls."""
        return self._prog.hash

    @property
    def warmup(self) -> int | None:
        """Leading bars that are unknown (NaN) on clean data; ``None`` when the expression uses
        ``at(...)``, because the base bars a coarser clock costs depend on how many bars a window
        holds. Then see :attr:`warmup_by_clock` and :meth:`warmup_bars`."""
        return self._prog.warmup

    @property
    def warmup_by_clock(self) -> dict[str, int]:
        """Warm-up per clock: ``"base"`` in base bars, and each ``at`` timeframe in bars of that
        timeframe (``{"base": 0, "1d": 199}`` for ``at("1d", sma(close, 200))``)."""
        return dict(self._prog.warmup_by_clock)

    def warmup_bars(self, bar_ns: int | None = None) -> int | None:
        """The most leading unknown base bars, given the base bar width in nanoseconds; ``None``
        when that cannot be known (no width, or an ``at`` over a month or a year). A window of
        width ``w`` holds at most ``ceil(w / bar_ns)`` bars, so ``at("1d", sma(close, 200))``
        on 5-minute bars is at most ``200 * 288`` bars, a bound that ignores the session. Without
        ``at`` it is :attr:`warmup`."""
        return self._prog.warmup_bars(bar_ns)

    @property
    def is_panel(self) -> bool:
        """True when the expression has a cross-sectional call (``rank``, ``zscore_xs``, ...):
        it has a value only over a panel, through :meth:`eval_panel`."""
        return self._prog.is_panel

    @property
    def groups(self) -> tuple[str, ...]:
        """The names of the group series the expression reads (``col.group("sector")``); give
        each to :meth:`eval_panel` as a label per name."""
        return self._groups

    @property
    def uses_at(self) -> bool:
        """True when the expression uses ``at(...)``: evaluating it needs timestamps."""
        return self._prog.uses_at

    def bound_hash(self, bar_stamp: str | None = None, tz_offset_ns: int = 0, session: Any = None) -> str:
        """The hash with the clock folded in when the expression uses ``at(...)``: the zone, the
        bar stamp and the session decide its numbers. Equal to :attr:`hash` without ``at``."""
        return self._prog.bound_hash(bar_stamp, tz_offset_ns, session)

    def at(self, timeframe: str) -> Expr:
        """This expression evaluated on ``timeframe`` bars and held on the base clock:
        ``col.close.sma(200).at("1d")`` is ``at("1d", sma(close, 200))``."""
        return Expr(f"at({_quote(str(timeframe))}, {self._src})", self._params, groups=self._groups)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Names of the output columns (``("value",)`` for a single series)."""
        return tuple(self._prog.outputs)

    @property
    def kinds(self) -> tuple[str, ...]:
        """What each output is, in the order of :attr:`outputs`: ``"bool"`` (1.0, 0.0 or NaN) or
        ``"number"``."""
        return tuple(self._prog.output_kinds)

    @property
    def fields(self) -> tuple[str, ...]:
        """The columns the expression reads, sorted."""
        return tuple(self._prog.fields)

    @property
    def handles(self) -> tuple[str, ...]:
        """The named series the expression reads (see ``series`` in :func:`expr`)."""
        return tuple(h for h in self._prog.handles if h not in self._groups)

    def settle(self, eps: float) -> int:
        """Bars until functions with infinite memory (``ema``) have forgotten their seed to
        within ``eps``, a number in (0, 1). Equals :attr:`warmup` when there are none."""
        return self._prog.settle(eps)

    def __repr__(self) -> str:
        return f"Expr({self._prog.canonical})"

    # -- evaluation

    def eval(
        self,
        data: Mapping[str, Any],
        *,
        bar_stamp: str | None = None,
        tz_offset_ns: int = 0,
        session: Any = None,
        bar_ns: int | None = None,
    ):
        """Evaluate over whole columns.

        ``data`` maps field names (``open``, ``high``, ``low``, ``close``, ``volume``) to 1-D
        float64 numpy arrays of one length; keys that are not fields are ignored, arrays of
        another dtype are refused. Returns an array for one output, or a dict of arrays keyed by
        output name. Unknown (warm-up, or a NaN input) is NaN. No input is modified.

        An expression that uses ``at(...)`` also needs ``data["timestamps"]`` (int64
        nanoseconds) and ``bar_stamp`` (``"open"`` or ``"close"``: whether a timestamp starts or
        ends its bar); it is refused without them, never given a default. ``tz_offset_ns`` is the
        zone daily windows are cut in (UTC, ``0``, unless given; ``session``, a
        ``raptorbt.SessionSchedule``, brings its own and skips non-trading days), and ``bar_ns``,
        the base bar width, lets a timeframe that is not a coarser exact multiple be refused.
        Without ``at`` all four are ignored.
        """
        if isinstance(data, Mapping) and not isinstance(data, dict):
            data = dict(data)
        cols = self._prog.evaluate(data, bar_stamp, tz_offset_ns, session, bar_ns)
        return cols[0] if len(cols) == 1 else dict(zip(self._prog.outputs, cols))

    def eval_panel(
        self,
        data: Mapping[str, Any],
        *,
        instruments: Sequence[str] | None = None,
        present: Any = None,
        bar_stamp: str | None = None,
        tz_offset_ns: int = 0,
        session: Any = None,
        bar_ns: int | None = None,
    ):
        """Evaluate over many names at once. Returns a 2-D float64 array ``(n_dates, n_assets)``
        for one output, or a dict of them.

        ``data`` is either

        * a dict of 2-D float64 arrays, one per field (``{"close": C, "volume": V}``), all of
          shape ``(n_dates, n_assets)``; or
        * a mapping of instrument to its columns (``{"AAA": {"timestamps": ts, "close": c}, ...}``,
          1-D arrays of that instrument's own bars). The bars are aligned on the sorted union of
          the instruments' timestamps (int64 nanoseconds, each strictly increasing), a name is
          absent at the dates it has no bar, and the rows of the result are those dates
          (``np.unique(np.concatenate(timestamps))``). ``instruments`` selects and orders the
          names, which otherwise follow the mapping.

        Absent bars. A time-series function never sees an absent bar: it does not step, so a name
        listed later counts its warm-up from its own first bar, and the result is NaN at the
        dates it has no bar. A bar that is present with a NaN value does step the function. In
        the 2-D form ``present`` (a bool array of the same shape) says which cells are bars;
        without it a cell is absent when every field the expression reads is NaN there, so a NaN
        close and a missing bar are the same thing unless you pass ``present``. A cross-sectional
        function leaves absent names out of the date's comparison and gives them NaN.

        A group series (``col.group("sector")``) is given in ``data`` under its name: a 2-D
        array, or 1-D with one numeric label per name (in the mapping form, a number beside the
        columns). An expression with ``at(...)`` reads ``data["timestamps"]`` (one int64
        timestamp per date) and needs ``bar_stamp`` as :meth:`eval` does. The GIL is released
        while it runs, and the numbers do not depend on the number of threads.
        """
        arrays, mask, _ = _panel_inputs(self, data, instruments, present, "Expr.eval_panel")
        cols = self._prog.eval_panel(arrays, mask, bar_stamp, tz_offset_ns, session, bar_ns)
        return cols[0] if len(cols) == 1 else dict(zip(self._prog.outputs, cols))

    def stream(
        self,
        *,
        bar_stamp: str | None = None,
        tz_offset_ns: int = 0,
        session: Any = None,
        bar_ns: int | None = None,
    ) -> ExprStream:
        """A bar-by-bar evaluator. Fed the same bars, it returns the numbers :meth:`eval` does,
        bit for bit. An expression that uses ``at(...)`` needs ``bar_stamp`` here (and the
        optional ``tz_offset_ns``, ``session``, ``bar_ns`` of :meth:`eval`) and takes each
        bar's ``timestamp=`` in :meth:`ExprStream.update`."""
        return ExprStream(self._prog, self._prog.stream(bar_stamp, tz_offset_ns, session, bar_ns))

    # -- truth is not defined

    def __bool__(self) -> bool:
        raise _refuse(
            "expr_type",
            "an expression has no truth value: Python's `and`, `or`, `not`, `if` and chained "
            "comparisons (`a < x < b`) would silently keep one operand. Use `&`, `|` and `~`, "
            "and write `(a < x) & (x < b)`",
            operation="Expr.__bool__",
            valid=["& | ~ between parenthesised expressions"],
        )

    # -- operators

    def _binary(self, op: str, other: Any, *, flip: bool = False) -> Expr:
        try:
            src, params = _operand(other, "Expr")
        except _NotANumber:
            return NotImplemented
        a, b = (src, self._src), (self._src, src)
        left, right = a if flip else b
        merged = _merge(self._params, params, "Expr")
        groups = self._groups + (other._groups if isinstance(other, Expr) else ())
        return Expr(f"({left} {op} {right})", merged, groups=groups)

    def __add__(self, o): return self._binary("+", o)
    def __radd__(self, o): return self._binary("+", o, flip=True)
    def __sub__(self, o): return self._binary("-", o)
    def __rsub__(self, o): return self._binary("-", o, flip=True)
    def __mul__(self, o): return self._binary("*", o)
    def __rmul__(self, o): return self._binary("*", o, flip=True)
    def __truediv__(self, o): return self._binary("/", o)
    def __rtruediv__(self, o): return self._binary("/", o, flip=True)
    def __pow__(self, o): return self._binary("**", o)
    def __rpow__(self, o): return self._binary("**", o, flip=True)
    def __lt__(self, o): return self._binary("<", o)
    def __le__(self, o): return self._binary("<=", o)
    def __gt__(self, o): return self._binary(">", o)
    def __ge__(self, o): return self._binary(">=", o)
    def __eq__(self, o): return self._binary("==", o)  # type: ignore[override]
    def __ne__(self, o): return self._binary("!=", o)  # type: ignore[override]

    # An Expr is not hashable: `==` builds an expression, so it cannot be a dict key.
    __hash__ = None  # type: ignore[assignment]

    def _logic(self, op: str, other: Any, *, flip: bool = False) -> Expr:
        if not isinstance(other, Expr):
            return NotImplemented
        left, right = (other._src, self._src) if flip else (self._src, other._src)
        return Expr(
            f"({left} {op} {right})",
            _merge(self._params, other._params, "Expr"),
            groups=self._groups + other._groups,
        )

    def __and__(self, o): return self._logic("and", o)
    def __rand__(self, o): return self._logic("and", o, flip=True)
    def __or__(self, o): return self._logic("or", o)
    def __ror__(self, o): return self._logic("or", o, flip=True)

    def __invert__(self) -> Expr:
        return Expr(f"(not {self._src})", self._params, groups=self._groups)

    def __neg__(self) -> Expr:
        return Expr(f"(-{self._src})", self._params, groups=self._groups)

    def __pos__(self) -> Expr:
        return self

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails: select an output of a multi-output call.
        outputs = object.__getattribute__(self, "_outputs") if name != "_outputs" else ()
        if not name.startswith("_") and name in outputs:
            base = object.__getattribute__(self, "_src")
            atom = object.__getattribute__(self, "_atom")
            params = object.__getattribute__(self, "_params")
            groups = object.__getattribute__(self, "_groups")
            return Expr(f"{base if atom else f'({base})'}.{name}", params, groups=groups)
        hint = f"; outputs: {', '.join(outputs)}" if outputs else ""
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}{hint}")


# ------------------------------------------------------------------ calls built from Python


def _call(row: dict[str, Any], lead: tuple[Any, ...], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Expr:
    """``name(inputs..., params...)`` from Python arguments: leading expressions are inputs, the
    rest are parameters in order, keywords name either. Omitted inputs read their default field
    and omitted parameters take their default; the full call is rendered, so the text means the
    same as the written form."""
    name, ins, prm = row["name"], row["inputs"], row["params"]
    op = name
    pos = [*lead, *args]
    k = 0
    while k < len(pos) and k < len(ins) and isinstance(pos[k], Expr):
        k += 1
    given: dict[str, Any] = {ins[i]["name"]: pos[i] for i in range(k)}
    rest = pos[k:]
    if len(rest) > len(prm):
        raise _refuse(
            "expr_type",
            f"wrong number of arguments; expected {_signature(row)}",
            operation=op,
            valid=[p["name"] for p in prm],
        )
    values: dict[str, Any] = {p["name"]: v for p, v in zip(prm, rest)}
    names = [*(i["name"] for i in ins), *(p["name"] for p in prm)]
    for key, v in kwargs.items():
        if key not in names:
            raise _refuse(
                "expr_type",
                f"{name}() has no argument {key!r}; expected {_signature(row)}",
                operation=op,
                parameter=key,
                valid=names,
            )
        target = given if any(i["name"] == key for i in ins) else values
        if key in target:
            raise _refuse(
                "expr_type", f"{name}() got {key!r} twice", operation=op, parameter=key, valid=names
            )
        target[key] = v

    parts: list[str] = []
    params: dict[str, float] = {}
    groups: tuple[str, ...] = ()
    for inp in ins:
        if inp["name"] in given:
            v = given[inp["name"]]
            if not isinstance(v, Expr):
                raise _refuse(
                    "expr_type",
                    f"input {inp['name']!r} of {name}() must be an expression, got {type(v).__name__}",
                    operation=op,
                    parameter=inp["name"],
                    valid=["an expression such as col.close"],
                )
            parts.append(v._src)
            params = _merge(params, v._params, op)
            groups += v._groups
        elif inp["default_field"]:
            parts.append(inp["default_field"])
        else:
            raise _refuse(
                "expr_type",
                f"{name}() needs input {inp['name']!r}; expected {_signature(row)}",
                operation=op,
                parameter=inp["name"],
                valid=["an expression such as col.close"],
            )
    for p in prm:
        if p["name"] in values:
            v = values[p["name"]]
        elif p["default"] is not None:
            v = p["default"]
        else:
            raise _refuse(
                "expr_type",
                f"{name}() needs parameter {p['name']!r}; expected {_signature(row)}",
                operation=op,
                parameter=p["name"],
                valid=[p["name"]],
            )
        if isinstance(v, str):
            parts.append(_quote(v))
            continue
        try:
            text = _number(v, op)
        except _NotANumber:
            raise _refuse(
                "expr_type",
                f"parameter {p['name']!r} of {name}() must be a constant, got {type(v).__name__}",
                operation=op,
                parameter=p["name"],
                valid=["a number, or a string for an enumerated parameter"],
            ) from None
        parts.append(text)
    return Expr(f"{name}({', '.join(parts)})", params, groups=groups, _atom=True)


def _method(row: dict[str, Any]):
    def method(self: Expr, *args: Any, **kwargs: Any) -> Expr:
        return _call(row, (self,), args, kwargs)

    method.__name__ = method.__qualname__ = row["name"]
    method.__doc__ = f"``{_signature(row)}`` with this expression as the first input.\n\n{row['doc']}"
    return method


def _builder(row: dict[str, Any]):
    def build(*args: Any, **kwargs: Any) -> Expr:
        return _call(row, (), args, kwargs)

    build.__name__ = build.__qualname__ = row["name"]
    build.__doc__ = f"``{_signature(row)}``\n\n{row['doc']}"
    return build


class _Col:
    """The columns of a bar series and every registered function as a builder.

    ``col.close`` is the close column; ``col.close.sma(20)`` and ``col.sma(20)`` both build
    ``sma(close, 20)``, and ``col.atr(14)`` reads high, low and close by default. The functions
    are generated from the registry when the module is imported.
    """

    __slots__ = ()

    @staticmethod
    def group(name: str) -> Expr:
        """A group series supplied per name to :meth:`Expr.eval_panel` under ``name`` (a numeric
        label such as a sector code): ``col.close.group_demean(col.group("sector"))``."""
        if not isinstance(name, str) or not name.isidentifier() or name in _FIELDS:
            raise _refuse(
                "expr_type",
                f"a group name is an identifier that is not a field, got {name!r}",
                operation="col.group",
                parameter="name",
                value=name,
                valid=["an identifier such as sector"],
            )
        return Expr(name, groups=(name,), _atom=True)

    def __repr__(self) -> str:
        return "col"


def _install() -> list[str]:
    """Generate the methods; returns the function names an Expr attribute already owns."""
    taken = set(dir(Expr)) | set(_FIELDS) | set(dir(_Col))
    shadowed = []
    for name, row in _registry().items():
        if name in taken:
            shadowed.append(name)
            continue
        setattr(_Col, name, staticmethod(_builder(row)))
        if row["inputs"]:
            setattr(Expr, name, _method(row))
    return shadowed


for _name in _FIELDS:
    setattr(_Col, _name, property(lambda self, _n=_name: Expr(_n, _atom=True)))
#: Functions whose names collide with an `Expr` or `col` attribute and so got no method. A test
#: keeps this empty: rename the function rather than shadow an attribute.
_SHADOWED = _install()

col = _Col()


def expr(
    source: str,
    params: Mapping[str, float] | None = None,
    series: Mapping[str, int] | None = None,
    *,
    groups: Sequence[str] = (),
) -> Expr:
    """Compile an expression from text.

    ``params`` binds the ``$name`` constants the text uses (``sma(close, $n)`` with
    ``params={"n": 20}``). ``series`` names numeric series the text may read besides the five
    columns, each with its leading unknown bars (``zscore(h_ab12, 20)`` with
    ``series={"h_ab12": 13}``); ``eval`` then takes each as a key of ``data``. Such a name is
    only available in text, not in operators built from :data:`col`. A refusal is an ``ExpressionError`` with ``code``, ``position`` (a
    character offset), ``snippet`` (the line with a caret) and ``valid`` (the closest names).

    ``groups`` names the group series the text reads (``group_demean(x, sector)`` with
    ``groups=["sector"]``), supplied per name to :meth:`Expr.eval_panel`.
    """
    return Expr(source, params, groups=tuple(groups), series=series)


# ------------------------------------------------------------------ panels


def _panel_inputs(
    e: Expr, data: Any, instruments: Sequence[str] | None, present: Any, operation: str,
    extra: Sequence[str] = (),
):
    """``(arrays, mask, axis)`` for the native panel evaluator: 2-D fields, group labels, the
    presence mask and (mapping form) the date axis. ``extra`` are fields to align besides the
    ones the expression reads."""
    if not isinstance(data, Mapping):
        raise _refuse(
            "invalid_input_type",
            f"data must be a dict of 2-D arrays or a mapping of instrument to columns, got "
            f"{type(data).__name__}",
            operation=operation,
            parameter="data",
        )
    keys = list(data)
    nested = bool(keys) and all(isinstance(data[k], Mapping) for k in keys)
    if not nested:
        arrays = dict(data)
        if instruments is not None:
            shape = next(
                (np.shape(arrays[f]) for f in (*_FIELDS,) if f in arrays and np.ndim(arrays[f]) == 2),
                None,
            )
            if shape is not None and len(instruments) != shape[1]:
                raise _refuse(
                    "length_mismatch",
                    f"{len(instruments)} instruments for panels with {shape[1]} columns",
                    operation=operation,
                    parameter="instruments",
                    valid=[f"{shape[1]} names"],
                )
        return arrays, present, None

    names = list(instruments) if instruments is not None else keys
    missing = [n for n in names if n not in data]
    if missing:
        raise _refuse(
            "invalid_input",
            f"no columns for instrument {missing[0]!r}",
            operation=operation,
            parameter="instruments",
            value=missing[0],
            valid=keys,
        )
    if present is not None:
        raise _refuse(
            "invalid_input",
            "`present` is derived from each instrument's timestamps in the mapping form",
            operation=operation,
            parameter="present",
        )
    need = sorted({*e.fields, *extra})
    stamps = []
    for n in names:
        ts = data[n].get("timestamps")
        if ts is None or not isinstance(ts, np.ndarray) or ts.dtype != np.int64 or ts.ndim != 1:
            raise _refuse(
                "missing_field",
                f"instrument {n!r} needs 'timestamps': a 1-D int64 array of nanoseconds, to align "
                "its bars with the other names",
                operation=operation,
                parameter="timestamps",
                valid=["timestamps=<int64 array>"],
            )
        if len(ts) > 1 and not (np.diff(ts) > 0).all():
            raise _refuse(
                "invalid_input",
                f"instrument {n!r}: timestamps must be strictly increasing",
                operation=operation,
                parameter="timestamps",
            )
        stamps.append(ts)
    axis = np.unique(np.concatenate(stamps)) if stamps else np.empty(0, np.int64)
    n_dates, n_assets = len(axis), len(names)
    arrays: dict[str, Any] = {f: np.full((n_dates, n_assets), np.nan) for f in need}
    mask = np.zeros((n_dates, n_assets), dtype=bool)
    for j, (n, ts) in enumerate(zip(names, stamps)):
        rows = np.searchsorted(axis, ts)
        mask[rows, j] = True
        for f in need:
            col = data[n].get(f)
            if col is None:
                raise _refuse(
                    "missing_field",
                    f"instrument {n!r} lacks the column {f!r} the expression reads",
                    operation=operation,
                    parameter=f,
                    valid=need,
                )
            col = np.asarray(col)
            if col.dtype != np.float64 or col.shape != ts.shape:
                raise _refuse(
                    "invalid_array",
                    f"instrument {n!r}: {f!r} must be a float64 array with one value per timestamp",
                    operation=operation,
                    parameter=f,
                )
            arrays[f][rows, j] = col
    for g in e.groups:
        try:
            arrays[g] = np.array([float(data[n][g]) for n in names], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            raise _refuse(
                "missing_field",
                f"every instrument needs a number {g!r} (its group label)",
                operation=operation,
                parameter=g,
            ) from None
    if e.uses_at:
        arrays["timestamps"] = axis
    return arrays, mask, axis


def _count(name: str, v: Any, lo: int, operation: str) -> int:
    if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < lo:
        raise _refuse(
            "invalid_parameter",
            f"{name} must be an integer >= {lo}, got {v!r}",
            operation=operation,
            parameter=name,
            value=v if isinstance(v, (int, float, str)) else repr(v),
            valid=[f"an integer >= {lo}"],
        )
    return int(v)


def factor_report(
    factor: Any,
    prices: Any,
    horizons: Sequence[int] = (1, 5, 20),
    quantiles: int = 5,
    *,
    lag: int = 1,
    min_names: int = 10,
    instruments: Sequence[str] | None = None,
    present: Any = None,
    bar_stamp: str | None = None,
    tz_offset_ns: int = 0,
    session: Any = None,
    bar_ns: int | None = None,
) -> dict[str, Any]:
    """Score a factor against the returns that follow it.

    ``factor`` is a 2-D float64 panel ``(n_dates, n_assets)`` or an :class:`Expr` evaluated with
    :meth:`Expr.eval_panel` (then ``prices`` may be a dict of panels, or a mapping of instrument
    to columns, and the expression reads from it; ``close`` is the price). ``prices`` is the
    price panel, or such a dict holding ``"close"``.

    The forward return of a signal date starts ``lag`` bars later (default 1: a signal computed
    at a close cannot trade that close) and runs ``h`` bars: ``price[d + lag + h] / price[d +
    lag] - 1``. Forward returns exist only here and never feed an expression. **No costs are
    charged**: the result says ``no_costs: True``, and every spread is gross.

    Returns a dict. Per horizon (``result["horizons"][h]``): ``ic`` (rank IC per date, NaN
    where not scored), ``ic_mean``, ``ic_std``, ``hit_rate`` (share of scored dates with IC > 0),
    ``t_naive`` and ``t_hac`` (Newey-West, Bartlett weights, ``h - 1`` lags: the daily ICs of an
    ``h``-bar return overlap, so the naive t is about ``sqrt(h)`` too large; NaN under 30
    scored dates), ``quantile_returns`` (mean forward return per factor bucket, bucket
    ``quantiles - 1`` the highest), ``quantile_returns_by_date``, ``long_short`` (top minus
    bottom per date) with ``long_short_mean`` and ``long_short_t_hac``. Beside them:
    ``coverage`` and ``names`` per date, ``autocorr`` (rank autocorrelation of the factor with
    its previous date) and ``turnover`` per quantile (``turnover_by_date`` per date).

    Refused with ``insufficient_observations``: fewer than ``max(min_names, quantiles)`` names or
    fewer than ``lag + max(horizons) + 30`` dates; ``invalid_input``: an infinite value;
    ``length_mismatch``: factor and prices of different shapes.
    """
    op = "factor_report"
    hs = [horizons] if isinstance(horizons, numbers.Integral) else list(horizons)
    if not hs:
        raise _refuse(
            "invalid_parameter", "horizons is empty", operation=op, parameter="horizons",
            valid=["a list of integers >= 1, e.g. [1, 5, 20]"],
        )
    hs = [_count("horizons", h, 1, op) for h in hs]
    q = _count("quantiles", quantiles, 2, op)
    lg = _count("lag", lag, 0, op)
    mn = _count("min_names", min_names, 2, op)

    meta: dict[str, Any] = {}
    if isinstance(factor, Expr):
        data = prices if isinstance(prices, Mapping) else {"close": prices}
        arrays, mask, _ = _panel_inputs(factor, data, instruments, present, op, extra=("close",))
        if "close" not in arrays:
            raise _refuse(
                "missing_field", "prices must hold 'close'", operation=op, parameter="close",
                valid=["close"],
            )
        if len(factor.outputs) != 1:
            raise _refuse(
                "expr_type",
                "a factor has one output; select one (macd(close).macdsignal)",
                operation=op,
                parameter="factor",
                valid=list(factor.outputs),
            )
        out = factor._prog.eval_panel(arrays, mask, bar_stamp, tz_offset_ns, session, bar_ns)
        panel, px = out[0], arrays["close"]
        meta = {"expression": factor.canonical, "hash": factor.hash}
    else:
        px = prices["close"] if isinstance(prices, Mapping) and "close" in prices else prices
        panel = factor
    for name, a in (("factor", panel), ("prices", px)):
        if not isinstance(a, np.ndarray) or a.ndim != 2 or a.dtype != np.float64:
            raise _refuse(
                "invalid_array",
                f"{name} must be a 2-D numpy float64 array (n_dates, n_assets)",
                operation=op,
                parameter=name,
            )
    result = _raptorbt.factor_report_native(panel, px, hs, q, lg, mn)
    return {**meta, **result}


# ------------------------------------------------------------------ streaming


class ExprStream:
    """Bar-by-bar evaluation of an expression; see :meth:`Expr.stream`."""

    __slots__ = ("_native", "_names", "_need", "_at")

    def __init__(self, program: Any, native: Any = None):
        self._native = program.stream() if native is None else native
        self._names = tuple(program.outputs)
        self._need = frozenset(program.fields)
        self._at = program.uses_at

    def update(self, bar: Mapping[str, float] | None = None, /, timestamp: int | None = None, **fields: float):
        """Feed one bar, as a mapping or keywords (``open``, ``high``, ``low``, ``close``,
        ``volume``). Returns the value for the bar, or a dict by output name for several
        outputs. A field the expression reads must be given; others may be left out. An
        expression that uses ``at(...)`` also needs ``timestamp=`` (int nanoseconds, read as the
        stream's ``bar_stamp`` says)."""
        if self._at and timestamp is None:
            raise _refuse(
                "missing_field",
                "the expression uses at(...) and reads each bar's timestamp; pass timestamp=",
                operation="ExprStream.update",
                parameter="timestamp",
                valid=["timestamp=<int nanoseconds>"],
            )
        given = {**(bar or {}), **fields}
        vals = []
        for f in _FIELDS:
            v = given.get(f)
            if v is None:
                if f in self._need:
                    raise _refuse(
                        "missing_field",
                        f"the expression reads {f!r} but the bar does not have it",
                        operation="ExprStream.update",
                        parameter=f,
                        valid=sorted(self._need),
                    )
                v = _NAN
            vals.append(v)
        out = self._native.update_at(int(timestamp), *vals) if self._at else self._native.update(*vals)
        return out[0] if len(out) == 1 else dict(zip(self._names, out))

    @property
    def ready(self) -> bool:
        """True once more bars have been fed than the expression's warm-up."""
        return self._native.ready

    @property
    def bars(self) -> int:
        """Bars fed so far."""
        return self._native.bars

    def reset(self) -> None:
        """Forget every bar."""
        self._native.reset()
