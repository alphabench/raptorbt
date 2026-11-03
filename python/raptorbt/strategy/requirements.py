"""What a strategy needs, declared as data.

A deployment system has to plan data, warm-up, capital and placement before it runs a strategy.
``Strategy.requirements`` is that declaration: pure data on the class, read without constructing
or running anything::

    class Spread(Strategy):
        requirements = Requirements(
            roles={
                "index": Underlying(asset="index"),
                "options": Chain(
                    of="index",
                    expiry=Expiry.weekly(),
                    strikes=StrikeWindow.around(4),
                ),
            },
            timeframes=["1m", "5m"],
            holding=Holding.intraday(flatten_before="15:15"),
            data=["bar"],
            warmup=Bars(50),
            max_structure=Legs(4, lots=2),
            params=SpreadConfig,
        )

A *role* is a name the strategy uses for an instrument it does not know in advance. A host binds
every :class:`Underlying` role to a symbol; the other kinds are derived from it
(:func:`resolve`). :func:`check_requirements` validates the declaration,
:func:`requirements_hash` fingerprints it, ``Requirements.to_dict`` / ``from_dict`` move it as JSON,
and :func:`params_schema` describes the strategy's ``StrategyConfig`` fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import types
import typing
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from raptorbt.errors import RaptorBTError
from raptorbt.strategy.api import TIMEFRAMES, Timeframe, resolve_timeframe
from raptorbt.strategy.config import StrategyConfig
from raptorbt.structures.legs import Expiry

__all__ = [
    "Bars",
    "ByDelta",
    "Chain",
    "Duration",
    "Future",
    "Holding",
    "Instrument",
    "Issue",
    "Legs",
    "ParamSchema",
    "ParamSpec",
    "Requirements",
    "RequirementsError",
    "RolePlan",
    "StrikeWindow",
    "SymbolPlan",
    "Underlying",
    "InstrumentPlan",
    "check_requirements",
    "param",
    "plan_instruments",
    "params_schema",
    "requirements_hash",
    "resolve",
]

#: The version of the dictionary ``Requirements.to_dict`` writes.
SCHEMA = 1
ASSETS = ("equity", "etf", "index", "currency", "commodity")
DATA_KINDS = ("bar", "trade", "quote", "book")
RIGHTS = ("call", "put")
EXPIRY_PICKS = ("nearest", "nth", "weekly", "monthly", "date", "exact")
CONTRACTS = ("front", "next", "continuous")
_CLOCK = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class Issue:
    """One thing wrong with a declaration. ``code`` is stable; ``path`` says where
    (``roles.options.strikes``); ``message`` is for people."""

    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class RequirementsError(ValueError):
    """A declaration that cannot be built. ``issues`` holds the structured reasons."""

    def __init__(self, issues: Sequence[Issue] | Issue):
        self.issues = tuple([issues] if isinstance(issues, Issue) else issues)
        super().__init__("; ".join(f"{i.path}: {i.message} [{i.code}]" for i in self.issues))


def _fail(code: str, path: str, message: str):
    raise RequirementsError(Issue(code, path, message))


def _positive_int(value: Any, path: str, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _fail("invalid_value", path, f"{what} is a positive integer, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Strikes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrikeWindow:
    """The strikes a chain may use, as listed-strike offsets from the one at the money:
    ``low`` to ``high`` inclusive (``StrikeWindow(-2, 6)``; ``StrikeWindow.around(4)`` is -4..4)."""

    low: int
    high: int

    def __post_init__(self) -> None:
        for name in ("low", "high"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                _fail("invalid_value", f"strikes.{name}", f"an integer offset, got {v!r}")
        if self.low > self.high:
            _fail("invalid_value", "strikes", f"low {self.low} is above high {self.high}")

    @staticmethod
    def around(n: int) -> StrikeWindow:
        """``n`` listed strikes each side of the money."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            _fail("invalid_value", "strikes", f"around(n): n is an integer >= 0, got {n!r}")
        return StrikeWindow(-n, n)

    def to_dict(self) -> dict[str, Any]:
        return {"by": "window", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class ByDelta:
    """The strikes whose absolute model delta lies in ``[low, high]`` (calls positive, puts
    mirrored). Resolving needs the underlying's price and a volatility."""

    low: float
    high: float

    def __post_init__(self) -> None:
        for name in ("low", "high"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < v < 1.0:
                _fail("invalid_value", f"strikes.{name}", f"an absolute delta in (0, 1), got {v!r}")
        if self.low > self.high:
            _fail("invalid_value", "strikes", f"low {self.low} is above high {self.high}")

    def to_dict(self) -> dict[str, Any]:
        return {"by": "delta", "low": float(self.low), "high": float(self.high)}


# ---------------------------------------------------------------------------
# Role kinds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Underlying:
    """An instrument the deployer binds to a symbol (a share, ETF, index, ...)."""

    asset: str = "equity"

    def __post_init__(self) -> None:
        if self.asset not in ASSETS:
            _fail("invalid_value", "asset", f"one of {', '.join(ASSETS)}, got {self.asset!r}")

    kind = "underlying"

    def refs(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "underlying", "asset": self.asset}


@dataclass(frozen=True)
class Instrument:
    """A fixed symbol the strategy always uses (a hedge, a benchmark)."""

    symbol: str
    asset: str = "equity"

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            _fail("invalid_value", "symbol", f"a non-empty symbol, got {self.symbol!r}")
        if self.asset not in ASSETS:
            _fail("invalid_value", "asset", f"one of {', '.join(ASSETS)}, got {self.asset!r}")

    kind = "instrument"

    def refs(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "instrument", "symbol": self.symbol, "asset": self.asset}


@dataclass(frozen=True)
class Future:
    """A futures contract of another role's underlying: ``front`` (the nearest unexpired),
    ``next``, or ``continuous`` (the front, rolled to the next ``roll_days`` days before its
    expiry; ``0`` rolls at the expiry itself)."""

    of: str
    contract: str = "front"
    roll_days: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.of, str) or not self.of:
            _fail("invalid_value", "of", f"the name of another role, got {self.of!r}")
        if self.contract not in CONTRACTS:
            _fail("invalid_value", "contract", f"one of {', '.join(CONTRACTS)}, got {self.contract!r}")
        if (
            isinstance(self.roll_days, bool)
            or not isinstance(self.roll_days, (int, float))
            or not (0.0 <= self.roll_days < 366.0)
        ):
            _fail("invalid_value", "roll_days", f"days in [0, 366), got {self.roll_days!r}")
        if self.roll_days and self.contract != "continuous":
            _fail("invalid_value", "roll_days", "only a continuous future rolls")

    kind = "future"

    def refs(self) -> tuple[str, ...]:
        return (self.of,)

    def to_dict(self) -> dict[str, Any]:
        out = {"kind": "future", "of": self.of, "contract": self.contract}
        if self.roll_days:
            out["roll_days"] = self.roll_days
        return out


@dataclass(frozen=True)
class Chain:
    """An option chain on another role: the expiries (one ``raptorbt.structures.Expiry``, or several) and the
    strikes (:class:`StrikeWindow` or :class:`ByDelta`) the strategy may hold, for ``rights``."""

    of: str
    expiry: Expiry | tuple[Expiry, ...]
    strikes: StrikeWindow | ByDelta
    rights: tuple[str, ...] = RIGHTS

    def __post_init__(self) -> None:
        if not isinstance(self.of, str) or not self.of:
            _fail("invalid_value", "of", f"the name of another role, got {self.of!r}")
        expiries = (self.expiry,) if isinstance(self.expiry, Expiry) else tuple(self.expiry)
        if not expiries or not all(isinstance(e, Expiry) for e in expiries):
            _fail("invalid_value", "expiry", "an Expiry or a non-empty sequence of Expiry")
        for e in expiries:
            if e.which not in EXPIRY_PICKS:
                _fail("invalid_value", "expiry.which", f"one of {', '.join(EXPIRY_PICKS)}, got {e.which!r}")
        if len(set(expiries)) != len(expiries):
            _fail("invalid_value", "expiry", "the same expiry is listed twice")
        object.__setattr__(self, "expiry", expiries)
        if not isinstance(self.strikes, (StrikeWindow, ByDelta)):
            _fail("invalid_value", "strikes", "a StrikeWindow or a ByDelta")
        rights = (self.rights,) if isinstance(self.rights, str) else tuple(self.rights)
        if not rights or any(r not in RIGHTS for r in rights) or len(set(rights)) != len(rights):
            _fail("invalid_value", "rights", f"a non-empty subset of {RIGHTS}, got {self.rights!r}")
        object.__setattr__(self, "rights", rights)

    kind = "chain"

    def refs(self) -> tuple[str, ...]:
        return (self.of,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "chain",
            "of": self.of,
            "expiry": [dataclasses.asdict(e) for e in self.expiry],
            "strikes": self.strikes.to_dict(),
            "rights": list(self.rights),
        }


Role = Underlying | Instrument | Future | Chain


def _role_from_dict(d: Mapping[str, Any]) -> Role:
    kind = d.get("kind")
    if kind == "underlying":
        return Underlying(d["asset"])
    if kind == "instrument":
        return Instrument(d["symbol"], d["asset"])
    if kind == "future":
        return Future(d["of"], d["contract"], d.get("roll_days", 0.0))
    if kind == "chain":
        s = d["strikes"]
        strikes = StrikeWindow(s["low"], s["high"]) if s["by"] == "window" else ByDelta(s["low"], s["high"])
        return Chain(d["of"], tuple(Expiry(**e) for e in d["expiry"]), strikes, tuple(d["rights"]))
    _fail("invalid_value", "kind", f"unknown role kind {kind!r}")


# ---------------------------------------------------------------------------
# Holding, warm-up, structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Holding:
    """How long positions live. ``intraday`` is flat by ``flatten_before`` (``"HH:MM"`` on the
    instrument's exchange clock); ``overnight`` may carry positions across sessions."""

    mode: str
    flatten_before: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("intraday", "overnight"):
            _fail("invalid_value", "holding.mode", f"intraday or overnight, got {self.mode!r}")
        if self.mode == "intraday":
            if not isinstance(self.flatten_before, str) or not _CLOCK.match(self.flatten_before):
                _fail("invalid_value", "holding.flatten_before", f"an HH:MM clock time, got {self.flatten_before!r}")
        elif self.flatten_before is not None:
            _fail("invalid_value", "holding.flatten_before", "an overnight strategy does not flatten")

    @staticmethod
    def intraday(flatten_before: str) -> Holding:
        return Holding("intraday", flatten_before)

    @staticmethod
    def overnight() -> Holding:
        return Holding("overnight")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "flatten_before": self.flatten_before}


@dataclass(frozen=True)
class Bars:
    """Warm-up as ``n`` bars of ``timeframe`` (default: every declared timeframe)."""

    n: int
    timeframe: str | None = None

    def __post_init__(self) -> None:
        _positive_int(self.n, "warmup.n", "n")
        if self.timeframe is not None and self.timeframe not in TIMEFRAMES:
            _fail("invalid_timeframe", "warmup.timeframe", f"one of {', '.join(TIMEFRAMES)}, got {self.timeframe!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"bars": self.n, "timeframe": self.timeframe}


@dataclass(frozen=True)
class Duration:
    """Warm-up as calendar time. ``Duration.days(5)``, ``.hours(3)``, ``.minutes(90)``."""

    seconds: int

    def __post_init__(self) -> None:
        _positive_int(self.seconds, "warmup.seconds", "seconds")

    @staticmethod
    def minutes(n: int) -> Duration:
        return Duration(n * 60)

    @staticmethod
    def hours(n: int) -> Duration:
        return Duration(n * 3600)

    @staticmethod
    def days(n: int) -> Duration:
        return Duration(n * 86400)

    def to_dict(self) -> dict[str, Any]:
        return {"seconds": self.seconds}


@dataclass(frozen=True)
class Legs:
    """The largest structure the strategy may hold at once: ``n`` legs of up to ``lots`` lots
    each. A host sizes margin and capital from it."""

    n: int
    lots: int = 1

    def __post_init__(self) -> None:
        _positive_int(self.n, "max_structure.n", "n")
        _positive_int(self.lots, "max_structure.lots", "lots")

    def to_dict(self) -> dict[str, Any]:
        return {"legs": self.n, "lots": self.lots}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def param(default: Any = dataclasses.MISSING, *, min: float | None = None, max: float | None = None,
          choices: Sequence[Any] | None = None, description: str | None = None) -> Any:
    """A ``StrategyConfig`` field with bounds or choices, read by :func:`params_schema`::

        @dataclass(frozen=True)
        class Config(StrategyConfig):
            fast: int = param(10, min=2, max=200)
            mode: str = param("long", choices=["long", "short"])
    """
    meta = {"min": min, "max": max, "choices": list(choices) if choices is not None else None, "description": description}
    meta = {k: v for k, v in meta.items() if v is not None}
    if default is dataclasses.MISSING:
        return field(metadata={"raptorbt": meta})
    return field(default=default, metadata={"raptorbt": meta})


_BASE_FIELDS = {f.name for f in dataclasses.fields(StrategyConfig)}
_SCALARS = {int: "int", float: "float", str: "str", bool: "bool"}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    default: Any = None
    required: bool = False
    min: float | None = None
    max: float | None = None
    choices: tuple[Any, ...] | None = None
    nullable: bool = False
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "type": self.type, "default": self.default, "required": self.required,
            "min": self.min, "max": self.max,
            "choices": list(self.choices) if self.choices is not None else None,
            "nullable": self.nullable, "description": self.description,
        }

    def check(self, value: Any) -> list[Issue]:
        path = f"params.{self.name}"
        if value is None:
            return [] if self.nullable else [Issue("invalid_param", path, "a value is required")]
        ok = (
            (self.type == "bool" and isinstance(value, bool))
            or (self.type == "int" and isinstance(value, int) and not isinstance(value, bool))
            or (self.type == "float" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (self.type == "str" and isinstance(value, str))
        )
        if not ok:
            return [Issue("invalid_param", path, f"expected {self.type}, got {type(value).__name__} {value!r}")]
        out = []
        if self.type == "float" and not math.isfinite(value):
            out.append(Issue("invalid_param", path, f"must be finite, got {value!r}"))
        elif self.min is not None and value < self.min:
            out.append(Issue("out_of_bounds", path, f"{value!r} is below the minimum {self.min!r}"))
        elif self.max is not None and value > self.max:
            out.append(Issue("out_of_bounds", path, f"{value!r} is above the maximum {self.max!r}"))
        if self.choices is not None and value not in self.choices:
            out.append(Issue("not_a_choice", path, f"{value!r} is not one of {list(self.choices)!r}"))
        return out


@dataclass(frozen=True)
class ParamSchema:
    """The typed parameters of a ``StrategyConfig`` subclass."""

    params: tuple[ParamSpec, ...] = ()

    def to_dict(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.params]

    @staticmethod
    def from_dict(items: Iterable[Mapping[str, Any]]) -> ParamSchema:
        specs = []
        for d in items:
            choices = d.get("choices")
            specs.append(ParamSpec(**{**d, "choices": tuple(choices) if choices is not None else None}))
        return ParamSchema(tuple(specs))

    def validate(self, values: Mapping[str, Any]) -> list[Issue]:
        """Issues for ``values`` (a name -> value mapping): unknown names, missing required
        ones, wrong types, bounds and choices. Absent optional names take their default."""
        known = {p.name: p for p in self.params}
        out = [Issue("unknown_param", f"params.{n}", "not a parameter of this strategy") for n in values if n not in known]
        for p in self.params:
            if p.name in values:
                out.extend(p.check(values[p.name]))
            elif p.required:
                out.append(Issue("missing_param", f"params.{p.name}", "required"))
        return out


def params_schema(config_cls: type[StrategyConfig]) -> ParamSchema:
    """The parameters of a ``StrategyConfig`` subclass, from its dataclass fields: the
    annotation gives the type (``int``, ``float``, ``str``, ``bool``, ``Literal[...]``, each
    optionally ``| None``), the default gives the default, and :func:`param` gives bounds,
    choices and a description. Raises :class:`RequirementsError` for a field it cannot
    describe (an unsupported type, a mutable or factory default, a bad bound)."""
    if not (isinstance(config_cls, type) and issubclass(config_cls, StrategyConfig) and dataclasses.is_dataclass(config_cls)):
        _fail("invalid_value", "params", f"a StrategyConfig dataclass subclass, got {config_cls!r}")
    hints = typing.get_type_hints(config_cls)
    issues: list[Issue] = []
    specs = []
    for f in dataclasses.fields(config_cls):
        if f.name in _BASE_FIELDS:
            continue
        path = f"params.{f.name}"
        tp, nullable = hints[f.name], False
        if typing.get_origin(tp) in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(tp) if a is not type(None)]
            nullable = len(args) != len(typing.get_args(tp))
            tp = args[0] if len(args) == 1 else tp
        choices = None
        if typing.get_origin(tp) is typing.Literal:
            choices = tuple(typing.get_args(tp))
            kinds = {type(c) for c in choices}
            tp = kinds.pop() if len(kinds) == 1 else None
        name = _SCALARS.get(tp)
        if name is None:
            issues.append(Issue("unsupported_type", path, f"{hints[f.name]!r}: use int, float, str, bool or Literal"))
            continue
        meta = dict(f.metadata.get("raptorbt", {}))
        if "choices" in meta:
            choices = tuple(meta["choices"])
        if f.default_factory is not dataclasses.MISSING:
            issues.append(Issue("unsupported_default", path, "a default_factory cannot be a parameter default"))
            continue
        has_default = f.default is not dataclasses.MISSING
        spec = ParamSpec(
            f.name, name, f.default if has_default else None, not has_default,
            meta.get("min"), meta.get("max"), choices, nullable, meta.get("description"),
        )
        if (spec.min is not None or spec.max is not None) and name not in ("int", "float"):
            issues.append(Issue("invalid_bounds", path, "min and max apply to int and float only"))
        elif spec.min is not None and spec.max is not None and spec.min > spec.max:
            issues.append(Issue("invalid_bounds", path, f"min {spec.min} is above max {spec.max}"))
        elif has_default and f.default is not None:
            bad = spec.check(f.default)
            if bad:
                issues.append(Issue("invalid_default", path, bad[0].message))
        specs.append(spec)
    if issues:
        raise RequirementsError(issues)
    return ParamSchema(tuple(specs))


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def _tf_key(tf: Timeframe) -> tuple:
    seconds = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if tf.unit in seconds:
        return (0, tf.step * seconds[tf.unit], tf.label)
    return (1, 0, tf.label)


def _tf_label(tf: Timeframe) -> str:
    return f"{tf.label}:{tf.brick_size:g}" if tf.unit == "renko" else tf.label


@dataclass(frozen=True)
class Requirements:
    """Everything a deployer must know about a strategy before it runs.

    ``roles`` maps a role name to a role kind; ``timeframes`` lists every bar series the
    strategy reads (strings of ``raptorbt.strategy.api.TIMEFRAMES``, or ``Timeframe``); ``data``
    is a subset of ``bar, trade, quote, book`` (the ``data_type`` of ``Strategy.subscribe``);
    ``warmup`` is :class:`Bars` or :class:`Duration`; ``max_structure`` is :class:`Legs`;
    ``params`` is a ``StrategyConfig`` subclass or a :class:`ParamSchema`.

    Construction validates each part on its own; :meth:`issues` checks them against each other.
    """

    roles: Mapping[str, Role] = field(default_factory=dict)
    timeframes: Sequence[str | Timeframe] = ()
    holding: Holding | None = None
    data: Sequence[str] = ("bar",)
    warmup: Bars | Duration | None = None
    max_structure: Legs | None = None
    params: Any = None

    def __post_init__(self) -> None:
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        roles = dict(self.roles)
        for name, role in roles.items():
            if not isinstance(name, str) or not name.isidentifier():
                _fail("invalid_role_name", f"roles.{name}", "a role name is a Python identifier")
            if not isinstance(role, (Underlying, Instrument, Future, Chain)):
                _fail("invalid_role", f"roles.{name}", f"an Underlying, Instrument, Future or Chain, got {role!r}")
        set_("roles", types.MappingProxyType(roles))
        tfs: dict[str, Timeframe] = {}
        for raw in self.timeframes:
            tf = raw if isinstance(raw, Timeframe) else resolve_timeframe(raw, "timeframes")
            if tf is None:
                _fail("invalid_timeframe", "timeframes", "None is not a timeframe")
            tfs[_tf_label(tf)] = tf
        set_("timeframes", tuple(sorted(tfs.values(), key=_tf_key)))
        data = (self.data,) if isinstance(self.data, str) else tuple(self.data)
        for d in data:
            if d not in DATA_KINDS:
                _fail("invalid_data", "data", f"one of {', '.join(DATA_KINDS)}, got {d!r}")
        set_("data", tuple(sorted(set(data), key=DATA_KINDS.index)))
        if self.holding is not None and not isinstance(self.holding, Holding):
            _fail("invalid_value", "holding", "Holding.intraday(...) or Holding.overnight()")
        if self.warmup is not None and not isinstance(self.warmup, (Bars, Duration)):
            _fail("invalid_value", "warmup", "Bars(n) or Duration")
        if self.max_structure is not None and not isinstance(self.max_structure, Legs):
            _fail("invalid_value", "max_structure", "Legs(n, lots=...)")
        if self.params is not None and not isinstance(self.params, ParamSchema):
            set_("params", params_schema(self.params))

    def issues(self) -> list[Issue]:
        """The declaration checked against itself: role references, role cycles, a chain on
        a chain, warm-up and timeframes."""
        out: list[Issue] = []
        for name, role in self.roles.items():
            for ref in role.refs():
                target = self.roles.get(ref)
                path = f"roles.{name}.of"
                if target is None:
                    out.append(Issue("unknown_role", path, f"refers to {ref!r}, which is not a role"))
                elif isinstance(role, Future) and not isinstance(target, Underlying):
                    out.append(Issue("invalid_base", path, f"a Future is of an Underlying, but {ref!r} is a {target.kind}"))
                elif isinstance(role, Chain) and not isinstance(target, (Underlying, Future)):
                    out.append(Issue("invalid_base", path, f"a Chain is of an Underlying or a Future, but {ref!r} is a {target.kind}"))
        if not self.roles:
            out.append(Issue("no_roles", "roles", "a strategy declares at least one role"))
        if self.holding is None:
            out.append(Issue("missing_holding", "holding", "declare Holding.intraday(...) or Holding.overnight()"))
        if not self.timeframes:
            out.append(Issue("no_timeframes", "timeframes", "declare the bar series the strategy reads"))
        if isinstance(self.warmup, Bars) and self.warmup.timeframe is not None:
            declared = {_tf_label(t) for t in self.timeframes}
            if self.warmup.timeframe not in declared:
                out.append(Issue("undeclared_timeframe", "warmup.timeframe", f"{self.warmup.timeframe!r} is not in timeframes"))
        if self.timeframes and "bar" not in self.data:
            out.append(Issue("timeframes_without_bars", "data", "timeframes are declared but data has no 'bar'"))
        return out

    @property
    def feed_interval(self) -> str | None:
        """The finest time-based timeframe: the bar interval a host must supply (the rest are
        aggregated from it). ``None`` when no time-based timeframe is declared."""
        for tf in self.timeframes:
            if _tf_key(tf)[0] == 0:
                return tf.label
        return None

    def to_dict(self) -> dict[str, Any]:
        """The manifest as plain JSON data (schema in ``docs/requirements.md``)."""
        w = self.warmup
        return {
            "schema": SCHEMA,
            "roles": {n: r.to_dict() for n, r in self.roles.items()},
            "timeframes": [_tf_label(t) for t in self.timeframes],
            "holding": self.holding.to_dict() if self.holding else None,
            "data": list(self.data),
            "warmup": w.to_dict() if w else None,
            "max_structure": self.max_structure.to_dict() if self.max_structure else None,
            "params": self.params.to_dict() if self.params else None,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Requirements:
        """The inverse of :meth:`to_dict`. ``params`` comes back as a :class:`ParamSchema`."""
        if d.get("schema") != SCHEMA:
            _fail("unsupported_schema", "schema", f"expected {SCHEMA}, got {d.get('schema')!r}")
        w = d.get("warmup")
        warmup = None if w is None else (Duration(w["seconds"]) if "seconds" in w else Bars(w["bars"], w["timeframe"]))
        h, m, p = d.get("holding"), d.get("max_structure"), d.get("params")
        tfs = []
        for label in d["timeframes"]:
            if ":" in label:
                step_unit, brick = label.split(":")
                tfs.append(Timeframe(int(step_unit.removesuffix("renko")), "renko", float(brick)))
            else:
                tfs.append(label if label in TIMEFRAMES else Timeframe(int(re.match(r"\d+", label).group()), label.lstrip("0123456789")))
        return Requirements(
            roles={n: _role_from_dict(r) for n, r in d["roles"].items()},
            timeframes=tfs,
            holding=Holding(h["mode"], h["flatten_before"]) if h else None,
            data=d["data"],
            warmup=warmup,
            max_structure=Legs(m["legs"], m["lots"]) if m else None,
            params=ParamSchema.from_dict(p) if p is not None else None,
        )


def _declared(strategy: Any) -> Requirements:
    r = strategy if isinstance(strategy, Requirements) else getattr(strategy, "requirements", None)
    if not isinstance(r, Requirements):
        _fail("no_requirements", "requirements", f"{strategy!r} declares no `requirements = Requirements(...)`")
    return r


def requirements_hash(strategy: Any) -> str:
    """The fingerprint of a declaration under this raptorbt version: the SHA-256 (hex) of the
    canonical JSON (sorted keys, no whitespace) of ``{"raptorbt": <version>, "requirements":
    <to_dict()>}``. Equal declarations hash equal whatever order roles, timeframes or data
    were written in. A platform stores it and refuses a run when the engine or the declaration
    differs. ``strategy`` is a Strategy class or a ``Requirements``."""
    import raptorbt

    body = {"raptorbt": raptorbt.__version__, "requirements": _declared(strategy).to_dict()}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolePlan:
    """What one role means at a time: ``symbols`` are the concrete contracts to subscribe to,
    ``expiries`` the expiries (ns) a chain resolved to, ``atm`` the money strike, ``detail``
    anything else (``roll_to`` for a continuous future)."""

    role: str
    kind: str
    symbols: tuple[str, ...] = ()
    expiries: tuple[int, ...] = ()
    atm: float | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolPlan:
    """The concrete symbols of every role at ``now_ns``. ``symbols`` is all of them, in role
    order without repeats. ``issues`` lists what could not be resolved; a plan with issues is
    partial."""

    now_ns: int
    roles: Mapping[str, RolePlan]
    symbols: tuple[str, ...]
    issues: tuple[Issue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "now_ns": self.now_ns,
            "roles": {n: {"kind": p.kind, "symbols": list(p.symbols), "expiries": list(p.expiries), "atm": p.atm, "detail": dict(p.detail)} for n, p in self.roles.items()},
            "symbols": list(self.symbols),
            "issues": [i.to_dict() for i in self.issues],
        }


def resolve(
    requirements: Requirements,
    *,
    now_ns: int,
    bindings: Mapping[str, str] | None = None,
    specs: Iterable[Any] = (),
    prices: Mapping[str, float] | None = None,
    vols: Mapping[str, float] | None = None,
    chain: Any = None,
) -> SymbolPlan:
    """Which concrete symbols each role means at ``now_ns``, so a host and a backtest resolve
    roles the same way. Never raises for data it lacks: that is an entry in ``issues`` and the
    role resolves to nothing.

    - ``bindings`` maps each :class:`Underlying` role to its symbol.
    - ``specs`` are the listed ``InstrumentSpec`` (futures and options); ``chain`` is an
      ``OptionChain`` already built from them (otherwise one is built from ``specs``).
    - ``prices`` maps an underlying symbol to its last price (the at-the-money reference);
      ``vols`` to the volatility :class:`ByDelta` prices deltas with.

    A :class:`Future` takes the unexpired futures contracts of its base symbol, ordered by
    expiry: ``front`` is the first, ``next`` the second, ``continuous`` the first with the
    second in ``detail["roll_to"]``. A :class:`Chain` on a Future is of that future's resolved
    symbol. :class:`StrikeWindow` offsets index the strikes listed for the expiry.
    """
    from raptorbt import OptionChain

    bindings = bindings or {}
    prices = prices or {}
    vols = vols or {}
    specs = list(specs)
    issues: list[Issue] = [Issue(i.code, i.path, i.message) for i in requirements.issues() if i.code in ("unknown_role", "invalid_base")]
    if issues:
        return SymbolPlan(now_ns, {}, (), tuple(issues))
    futures = sorted(
        (s for s in specs if s.kind == "contract" and s.expiration_ns is not None),
        key=lambda s: s.expiration_ns,
    )
    options = [s for s in specs if s.kind == "option"]
    if chain is None and options and any(isinstance(r, Chain) for r in requirements.roles.values()):
        chain = OptionChain(options)
    plans: dict[str, RolePlan] = {}

    def done(plan: RolePlan) -> None:
        plans[plan.role] = plan

    def need(code: str, name: str, message: str) -> None:
        issues.append(Issue(code, f"roles.{name}", message))

    def live(role_name: str) -> RolePlan:
        # Resolve in dependency order; roles are acyclic (a Future/Chain is of an Underlying/Future).
        if role_name in plans:
            return plans[role_name]
        role = requirements.roles[role_name]
        if isinstance(role, Underlying):
            sym = bindings.get(role_name)
            if sym is None:
                need("unbound_role", role_name, "no symbol bound to this role")
            done(RolePlan(role_name, role.kind, (sym,) if sym else ()))
        elif isinstance(role, Instrument):
            done(RolePlan(role_name, role.kind, (role.symbol,)))
        elif isinstance(role, Future):
            base = live(role.of).symbols
            cutoff = int(role.roll_days * 86_400 * 10**9) if role.contract == "continuous" else 0
            contracts = [
                s.symbol for s in futures
                if base and s.underlying == base[0] and s.expiration_ns - cutoff > now_ns
                and (s.activation_ns is None or s.activation_ns <= now_ns)
            ]
            idx = 1 if role.contract == "next" else 0
            if len(contracts) <= idx:
                if base:
                    need("no_contract", role_name, f"no {role.contract} futures contract of {base[0]!r} at {now_ns}")
                done(RolePlan(role_name, role.kind))
            else:
                detail = {"roll_to": contracts[1]} if role.contract == "continuous" and len(contracts) > 1 else {}
                done(RolePlan(role_name, role.kind, (contracts[idx],), detail=detail))
        else:
            _chain(role_name, role)
        return plans[role_name]

    def _chain(name: str, role: Chain) -> None:
        base = live(role.of).symbols
        if not base:
            done(RolePlan(name, role.kind))
            return
        u = base[0]
        if chain is None:
            need("no_options", name, f"no option contracts listed for {u!r}")
            done(RolePlan(name, role.kind))
            return
        picked = []
        for e in role.expiry:
            try:
                ns = e.resolve(chain, u, now_ns)
            except RaptorBTError:
                ns = None
            if ns is None:
                need("no_expiry", name, f"no {e.which} expiry (n={e.n}) listed for {u!r} at {now_ns}")
            else:
                picked.append(ns)
        price = prices.get(u)
        if price is None:
            need("needs_price", name, f"the price of {u!r} is needed to find the money strike")
            done(RolePlan(name, role.kind, expiries=tuple(picked)))
            return
        symbols: list[str] = []
        atm = None
        for ns in picked:
            ladder = chain.strikes(u, ns, now_ns)
            atm_k = chain.atm_strike(u, ns, now_ns, price)
            if atm is None:
                atm = atm_k
            if not ladder or atm_k is None:
                continue
            if isinstance(role.strikes, StrikeWindow):
                i = ladder.index(atm_k)
                chosen = ladder[max(0, i + role.strikes.low): max(0, i + role.strikes.high + 1)]
            else:
                vol = vols.get(u)
                if vol is None:
                    need("needs_vol", name, f"the volatility of {u!r} is needed to pick strikes by delta")
                    continue
                bounds = []
                for right in role.rights:
                    sign = 1.0 if right == "call" else -1.0
                    for target in (role.strikes.low, role.strikes.high):
                        s = chain.by_delta(u, ns, now_ns, right, sign * target, price, vol)
                        if s is not None:
                            bounds.append(s.strike)
                chosen = [k for k in ladder if bounds and min(bounds) <= k <= max(bounds)]
            for k in chosen:
                for right in role.rights:
                    c = chain.contract(u, ns, k, right, now_ns)
                    if c is not None and c.symbol not in symbols:
                        symbols.append(c.symbol)
        done(RolePlan(name, role.kind, tuple(symbols), tuple(picked), atm))

    for name in requirements.roles:
        live(name)
    ordered = {n: plans[n] for n in requirements.roles}
    all_symbols = tuple(dict.fromkeys(s for p in ordered.values() for s in p.symbols))
    return SymbolPlan(now_ns, ordered, all_symbols, tuple(issues))


# ---------------------------------------------------------------------------
# Moving the instrument set of a running node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentPlan:
    """What a host does to a running :class:`~raptorbt.LiveNode` so its symbols cover what the
    strategies' roles mean now.

    ``add`` maps a symbol to the strategy indices that should trade it; ``retire`` lists symbols
    that were added to the node, that no strategy's roles want any more and that nothing holds;
    ``holding`` lists those that no role wants but a position or a working order still holds, to
    be retired once it is gone. ``issues`` lists what could not be planned. Apply it with
    :meth:`~raptorbt.LiveNode.apply_plan` or one step at a time.
    """

    now_ns: int
    add: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    retire: tuple[str, ...] = ()
    holding: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.add or self.retire)

    def to_dict(self) -> dict[str, Any]:
        return {
            "now_ns": self.now_ns,
            "add": {s: list(k) for s, k in self.add.items()},
            "retire": list(self.retire),
            "holding": list(self.holding),
            "issues": [i.to_dict() for i in self.issues],
        }


def plan_instruments(
    node: Any,
    *,
    now_ns: int | None = None,
    prices: Mapping[str, float] | None = None,
    vols: Mapping[str, float] | None = None,
    margin_strikes: int = 0,
    roll_lead_days: float = 0.0,
) -> InstrumentPlan:
    """Which instruments to add to and retire from a running node at ``now_ns`` (default: the
    node's clock), so every strategy that declares ``requirements`` trades the symbols its roles
    resolve to. Pure: it reads the node (its strategies, symbols, specs, bindings and last
    prices) and changes nothing; the host applies the plan.

    Roles resolve with :func:`resolve` from ``node.instruments`` (the listed contracts the host
    gave) and the last prices the strategy reads; ``prices`` / ``vols`` override or add. A chain
    window that moves with the money, a continuous future that rolls, an expiry that passes: each
    shows up as symbols to add and symbols to retire.

    - ``margin_strikes``: subscribe, and keep subscribed, this many listed strikes beyond each
      side of a chain's :class:`StrikeWindow`. The window moves when the money moves, and the
      host plans after the move: with a margin the strikes the next event needs are already
      there, and a price that wanders back over the edge is not retired and revived at once.
      A retired symbol that a window needs again is revived by ``add_instrument`` (same id).
    - ``roll_lead_days``: add the next contract of a continuous :class:`Future` this many days
      before its rule rolls, so it has data when the roll comes.

    Only symbols added to the node are ever retired, never the ones it was built with.
    """
    from raptorbt.strategy.continuous import front

    specs = node.instruments
    bindings = node.bindings
    now = node.now_ns if now_ns is None else now_ns
    issues: list[Issue] = []
    add: dict[str, list[int]] = {}
    keep: set[str] = set()
    managed = False
    retired = set(node.retired_instruments)
    day = 86_400 * 10**9
    for k, strategy in enumerate(node.strategies):
        req = getattr(type(strategy), "requirements", None)
        if not isinstance(req, Requirements):
            continue
        managed = True
        ctx = node.ctx(k)
        held = set(ctx.symbols)
        seen = {**node.last_prices(k), **(prices or {})}

        def resolved(requirements: Requirements) -> SymbolPlan:
            return resolve(
                requirements, now_ns=now, bindings=bindings, specs=specs.values(), prices=seen, vols=vols
            )

        plan = resolved(req)
        issues.extend(Issue(i.code, f"strategies[{k}].{i.path}", i.message) for i in plan.issues)
        want = list(plan.symbols)
        for name, role in req.roles.items():
            rolled = plan.roles.get(name)
            if isinstance(role, Future) and role.contract == "continuous" and rolled is not None:
                nxt = rolled.detail.get("roll_to")
                base = plan.roles[role.of].symbols
                if nxt and base and roll_lead_days > 0:
                    futures = [s for s in specs.values() if s.kind == "contract" and s.underlying == base[0]]
                    _, roll_ns = front(futures, role.roll_days, now)
                    if roll_ns is not None and now >= roll_ns - int(roll_lead_days * day):
                        want.append(nxt)
        wide = list(dict.fromkeys(want))
        if margin_strikes > 0:
            widened = {
                name: dataclasses.replace(
                    role, strikes=StrikeWindow(role.strikes.low - margin_strikes, role.strikes.high + margin_strikes)
                )
                if isinstance(role, Chain) and isinstance(role.strikes, StrikeWindow)
                else role
                for name, role in req.roles.items()
            }
            wide = list(dict.fromkeys([*wide, *resolved(dataclasses.replace(req, roles=widened)).symbols]))
        keep.update(wide)
        for symbol in wide:
            if symbol in held:
                continue
            if symbol in retired and k not in node.holders_of(symbol):
                issues.append(Issue("never_held", f"strategies[{k}]", f"{symbol!r} was retired and strategy {k} never held it, so it cannot come back for that strategy"))
                continue
            if symbol not in specs and symbol not in bindings.values():
                issues.append(Issue("no_spec", f"strategies[{k}]", f"{symbol!r} has no InstrumentSpec in the node's instruments"))
                continue
            add.setdefault(symbol, []).append(k)
    retire: list[str] = []
    holding: list[str] = []
    if managed:
        for symbol in node.added_instruments:
            if symbol in retired or symbol in keep:
                continue
            busy = any(
                symbol in node.ctx(k).symbols
                and (node.ctx(k).positions(symbol) or node.ctx(k).open_orders(symbol))
                for k in range(len(node.strategies))
            )
            (holding if busy else retire).append(symbol)
    return InstrumentPlan(
        now,
        {s: tuple(k) for s, k in add.items()},
        tuple(retire),
        tuple(holding),
        tuple(issues),
    )


# ---------------------------------------------------------------------------
# Checking a strategy class
# ---------------------------------------------------------------------------

_T0 = 1_767_225_600 * 10**9  # 2026-01-01 00:00 UTC; the dry run's synthetic clock
_DAY = 86_400 * 10**9
_MINUTE = 60 * 10**9


def check_requirements(strategy_cls: type, *, dry_run: bool = True) -> list[Issue]:
    """Validate a Strategy class's declaration; an empty list means it is consistent.

    Always checks that ``requirements`` is a :class:`Requirements`, that it is consistent
    with itself (:meth:`Requirements.issues`) and that its parameters describe. With
    ``dry_run`` it also runs the strategy briefly on a synthetic tape built from the
    declaration (every role bound to a synthetic symbol, a synthetic option universe for each
    chain, default config) and reports any subscription outside it: an instrument no role
    declares (``undeclared_instrument``), a timeframe not in ``timeframes``
    (``undeclared_timeframe``) or a data type not in ``data`` (``undeclared_data``). A
    strategy that raises in the dry run is reported as ``dry_run_failed`` (it may need
    inputs the synthetic tape does not have); its subscriptions up to then are still checked.
    """
    try:
        req = _declared(strategy_cls)
    except RequirementsError as e:
        return list(e.issues)
    out = req.issues()
    if out or not dry_run:
        return out
    return _dry_run(strategy_cls, req)


def _dry_run(strategy_cls: type, req: Requirements) -> list[Issue]:
    import numpy as np

    from raptorbt import BacktestConfig, InstrumentSpec
    from raptorbt.strategy import api
    from raptorbt.strategy.multi_runner import run_strategies

    # Every Underlying role is bound to its upper-cased name; futures and options are listed
    # for those. Futures resolve first, since a chain can hang off a future.
    bindings = {name: name.upper() for name, r in req.roles.items() if isinstance(r, Underlying)}
    specs = [
        InstrumentSpec.futures_contract(f"{base}F{k}", _T0 + 30 * _DAY * (k + 1), 1.0, underlying=base)
        for base in bindings.values() for k in range(2)
    ]
    prices = {s: 100.0 for s in [*bindings.values(), *(sp.symbol for sp in specs)]}
    vols = {s: 0.2 for s in prices}
    first = resolve(req, now_ns=_T0, bindings=bindings, specs=specs, prices=prices, vols=vols)
    for name, role in req.roles.items():
        if not isinstance(role, Chain) or not first.roles[role.of].symbols:
            continue
        u = first.roles[role.of].symbols[0]
        for week in range(1, 9):
            for strike in (float(k) for k in range(60, 141, 5)):
                for right in RIGHTS:
                    specs.append(InstrumentSpec.option(
                        f"{u}W{week}{int(strike)}{right[0].upper()}", strike, right,
                        _T0 + week * 7 * _DAY, 50.0, underlying=u))
    plan = resolve(req, now_ns=_T0, bindings=bindings, specs=specs, prices=prices, vols=vols)
    if plan.issues:
        return [Issue("dry_run_unresolved", i.path, i.message) for i in plan.issues]

    n = 30
    close = 100.0 + np.sin(np.arange(n) / 4.0)
    data = {
        s: {"timestamps": _T0 + _MINUTE * np.arange(1, n + 1, dtype=np.int64), "open": close.copy(),
            "high": close + 0.05, "low": close - 0.05, "close": close.copy(), "volume": np.full(n, 1000.0)}
        for s in plan.symbols
    }
    listed = {sp.symbol: sp for sp in specs if sp.symbol in data}
    seen: list[tuple] = []
    real = api.subscribe

    def record(strategy, instrument, data_type, timeframe, history):
        seen.append((instrument, data_type, timeframe))
        return real(strategy, instrument, data_type, timeframe, history)

    strategy = strategy_cls()
    found: dict[tuple[str, str], Issue] = {}

    def add(code: str, path: str, message: str) -> None:
        found.setdefault((code, message), Issue(code, path, message))

    api.subscribe = record
    try:
        # The tape's stamps are bar close times. The synthetic universe is a probe: a
        # `register_indicator` without `symbol=` would warn about interleaved bars here, and
        # it warns again in the real run, so the probe does not repeat it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="register_indicator", category=UserWarning)
            run_strategies([(strategy, list(plan.symbols))], data, BacktestConfig(initial_capital=1_000_000.0, session_tz_offset_ns=0),
                           instruments=listed or None, account_type="margin" if listed else "cash",
                           bar_stamp="close")
    except Exception as exc:  # the strategy's own code: reported, never hidden
        add("dry_run_failed", "strategy", f"{type(exc).__name__}: {exc}")
    finally:
        api.subscribe = real
    declared = {_tf_label(t) for t in req.timeframes}
    for instrument, data_type, timeframe in seen:
        if instrument not in plan.symbols:
            add("undeclared_instrument", "roles", f"subscribes to {instrument!r}, which no role resolves to")
        if data_type not in req.data:
            add("undeclared_data", "data", f"subscribes to {data_type!r}, which data does not declare")
        if timeframe is not None and _tf_label(resolve_timeframe(timeframe)) not in declared:
            add("undeclared_timeframe", "timeframes", f"subscribes to {resolve_timeframe(timeframe).label!r}, which timeframes does not declare")
    for step, unit, brick in strategy._bar_subscriptions:
        label = _tf_label(Timeframe(step, unit, brick))
        if label not in declared:
            add("undeclared_timeframe", "timeframes", f"subscribes to {label!r}, which timeframes does not declare")
    return list(found.values())
