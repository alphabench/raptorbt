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
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from raptorbt import _raptorbt, errors

__all__ = ["Expr", "ExprStream", "col", "describe", "expr", "functions"]

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

    ``query`` keeps the functions whose name, family or description contains it (case
    ignored). Each row has ``name``, ``family``, ``signature``, ``outputs`` and ``doc``;
    :func:`describe` gives the rest.
    """
    needle = None if query is None else query.lower()
    rows = []
    for row in _registry().values():
        if needle is not None and not any(
            needle in str(row[k]).lower() for k in ("name", "family", "doc")
        ):
            continue
        rows.append(
            {
                "name": row["name"],
                "family": row["family"],
                "signature": _signature(row),
                "outputs": list(row["outputs"]),
                "doc": row["doc"],
            }
        )
    return rows


def describe(name: str) -> dict[str, Any]:
    """Everything the registry says about one function: inputs, parameters, outputs, warm-up at
    the defaults, how it treats NaN, the legacy indicator it is pinned to, and its references.
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
    return {**copy.deepcopy(row), "signature": _signature(row)}


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

    __slots__ = ("_src", "_params", "_prog", "_atom", "_outputs")
    #: Make NumPy defer to the reflected operators instead of broadcasting over an Expr.
    __array_ufunc__ = None

    def __init__(self, source: str, params: Mapping[str, float] | None = None, *, _atom: bool = False):
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
        self._prog = _raptorbt.ExprProgram(source, bound or None)
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
        return Expr(f"at({_quote(str(timeframe))}, {self._src})", self._params)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Names of the output columns (``("value",)`` for a single series)."""
        return tuple(self._prog.outputs)

    @property
    def fields(self) -> tuple[str, ...]:
        """The columns the expression reads, sorted."""
        return tuple(self._prog.fields)

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
        return Expr(f"({left} {op} {right})", merged)

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
        return Expr(f"({left} {op} {right})", _merge(self._params, other._params, "Expr"))

    def __and__(self, o): return self._logic("and", o)
    def __rand__(self, o): return self._logic("and", o, flip=True)
    def __or__(self, o): return self._logic("or", o)
    def __ror__(self, o): return self._logic("or", o, flip=True)

    def __invert__(self) -> Expr:
        return Expr(f"(not {self._src})", self._params)

    def __neg__(self) -> Expr:
        return Expr(f"(-{self._src})", self._params)

    def __pos__(self) -> Expr:
        return self

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails: select an output of a multi-output call.
        outputs = object.__getattribute__(self, "_outputs") if name != "_outputs" else ()
        if not name.startswith("_") and name in outputs:
            base = object.__getattribute__(self, "_src")
            atom = object.__getattribute__(self, "_atom")
            params = object.__getattribute__(self, "_params")
            return Expr(f"{base if atom else f'({base})'}.{name}", params)
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
    return Expr(f"{name}({', '.join(parts)})", params, _atom=True)


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


def expr(source: str, params: Mapping[str, float] | None = None) -> Expr:
    """Compile an expression from text.

    ``params`` binds the ``$name`` constants the text uses (``sma(close, $n)`` with
    ``params={"n": 20}``). A refusal is an ``ExpressionError`` with ``code``, ``position`` (a
    character offset), ``snippet`` (the line with a caret) and ``valid`` (the closest names).
    """
    return Expr(source, params)


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
