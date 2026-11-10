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
            max_structure=Legs(
                OptionLeg(side="sell", right="call", offset=2, expiry=Expiry.weekly()),
                OptionLeg(side="buy", right="call", offset=4, expiry=Expiry.weekly()),
            ),
            max_position={"options": MaxPosition(lots=2)},
            params=SpreadConfig,
        )

A *role* is a name the strategy uses for an instrument it does not know in advance. A host binds
every :class:`Underlying` role to a symbol and every :class:`Universe` role (a basket of N
instruments chosen at deploy time) to a list of symbols; the other kinds are derived from them
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
    "LegContracts",
    "LegPlan",
    "Legs",
    "MaxPosition",
    "OptionLeg",
    "ParamSchema",
    "ParamSpec",
    "Requirements",
    "RequirementsError",
    "RolePlan",
    "StrikeWindow",
    "SymbolPlan",
    "Underlying",
    "Universe",
    "InstrumentPlan",
    "check_requirements",
    "param",
    "plan_instruments",
    "params_schema",
    "requirements_hash",
    "resolve",
    "resolve_legs",
]

#: The version of the dictionary ``Requirements.to_dict`` writes. 2: a ``universe`` role, ``max_position``
#: and ``max_structure`` as the option legs' shape.
SCHEMA = 2
ASSETS = ("equity", "etf", "index", "currency", "commodity")
DATA_KINDS = ("bar", "trade", "quote", "book")
RIGHTS = ("call", "put")
SIDES = ("buy", "sell")
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
class Universe:
    """A basket: a role that holds ``min`` to ``max`` instruments of one asset class, chosen by
    the deployer (one strategy over a list, such as a top-N momentum book). ``data`` is what each
    member needs (a subset of the manifest's ``data``). A host binds it to a list of symbols
    (``bindings={"basket": ["A", "B"]}``); the strategy reads the list with ``ctx.role("basket")``."""

    asset: str = "equity"
    min: int = 1
    max: int = 1
    data: tuple[str, ...] = ("bar",)

    def __post_init__(self) -> None:
        if self.asset not in ASSETS:
            _fail("invalid_value", "asset", f"one of {', '.join(ASSETS)}, got {self.asset!r}")
        for name in ("min", "max"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                _fail("invalid_value", name, f"a positive integer count, got {v!r}")
        if self.min > self.max:
            _fail("invalid_value", "min", f"min {self.min} is above max {self.max}")
        data = (self.data,) if isinstance(self.data, str) else tuple(self.data)
        if not data or any(d not in DATA_KINDS for d in data):
            _fail("invalid_data", "data", f"a non-empty subset of {', '.join(DATA_KINDS)}, got {self.data!r}")
        object.__setattr__(self, "data", tuple(sorted(set(data), key=DATA_KINDS.index)))

    kind = "universe"

    def refs(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "universe", "asset": self.asset, "min": self.min, "max": self.max, "data": list(self.data)}


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


Role = Underlying | Instrument | Universe | Future | Chain


def _role_from_dict(d: Mapping[str, Any]) -> Role:
    kind = d.get("kind")
    if kind == "underlying":
        return Underlying(d["asset"])
    if kind == "instrument":
        return Instrument(d["symbol"], d["asset"])
    if kind == "universe":
        return Universe(d["asset"], d["min"], d["max"], tuple(d["data"]))
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


@dataclass(frozen=True, kw_only=True)
class OptionLeg:
    """One option leg of the structure the strategy holds, as data: ``side`` (``buy`` or
    ``sell``), ``right`` (``call`` or ``put``), ``lots``, the strike and the expiry.

    The strike is ``offset`` (listed strikes from the money strike, up positive; ``0`` is at the
    money) or ``delta`` (an absolute delta target in (0, 1); a call's delta is positive, a put's
    negative). A ``delta`` leg also gives ``window``: the most listed strikes from the money its
    strike may be, which bounds the contracts a host subscribes to. ``expiry`` picks the expiry
    (:class:`~raptorbt.structures.Expiry`). ``chain`` names the :class:`Chain` role the leg is on;
    it may be left out when the strategy declares exactly one chain."""

    side: str
    right: str
    expiry: Expiry
    lots: int = 1
    offset: int | None = None
    delta: float | None = None
    window: int | None = None
    chain: str | None = None

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            _fail("invalid_value", "leg.side", f"one of {', '.join(SIDES)}, got {self.side!r}")
        if self.right not in RIGHTS:
            _fail("invalid_value", "leg.right", f"one of {', '.join(RIGHTS)}, got {self.right!r}")
        if not isinstance(self.expiry, Expiry) or self.expiry.which not in EXPIRY_PICKS:
            _fail("invalid_value", "leg.expiry", f"an Expiry selector, got {self.expiry!r}")
        _positive_int(self.lots, "leg.lots", "lots")
        if (self.offset is None) == (self.delta is None):
            _fail("invalid_value", "leg.strike", "give exactly one of offset= (strikes from the money) or delta=")
        if self.offset is not None and (not isinstance(self.offset, int) or isinstance(self.offset, bool)):
            _fail("invalid_value", "leg.offset", f"an integer number of strikes, got {self.offset!r}")
        if self.delta is not None:
            d = self.delta
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not 0.0 < d < 1.0:
                _fail("invalid_value", "leg.delta", f"an absolute delta in (0, 1), got {d!r}")
            if self.window is None:
                _fail("missing_window", "leg.window", "a delta leg gives window=: the most strikes from the money it may sit")
        if self.window is not None:
            if self.delta is None:
                _fail("invalid_value", "leg.window", "window= belongs to a delta leg; an offset leg is exact")
            _positive_int(self.window, "leg.window", "window")
        if self.chain is not None and (not isinstance(self.chain, str) or not self.chain):
            _fail("invalid_value", "leg.chain", f"the name of a Chain role, got {self.chain!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side, "right": self.right, "lots": self.lots,
            "offset": self.offset, "delta": None if self.delta is None else float(self.delta),
            "window": self.window, "expiry": dataclasses.asdict(self.expiry), "chain": self.chain,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> OptionLeg:
        return OptionLeg(
            side=d["side"], right=d["right"], expiry=Expiry(**d["expiry"]), lots=d["lots"],
            offset=d["offset"], delta=d["delta"], window=d["window"], chain=d["chain"],
        )


class Legs:
    """The shape of the structure the strategy holds: one :class:`OptionLeg` per leg of the
    largest structure it may hold at once, written ``Legs(OptionLeg(...), OptionLeg(...))``.

    It is enough for a host to margin the structure (:meth:`structure` builds a
    :class:`~raptorbt.structures.Structure`, whose ``resolve(...).margin(model, prices)`` is the
    group margin of a ``MarginModel``) and to list the contracts to subscribe to
    (:func:`resolve_legs`).

    ``Legs(4, lots=2)``, the earlier count-only form, is refused (``legs_need_shape``): a count
    cannot say which side or strike, so it cannot be margined. Declare the legs, and the largest
    position with ``Requirements(max_position=...)``."""

    __slots__ = ("legs",)
    legs: tuple[OptionLeg, ...]

    def __init__(self, *legs: OptionLeg, **old: Any) -> None:
        if old or any(isinstance(leg, int) and not isinstance(leg, bool) for leg in legs):
            _fail(
                "legs_need_shape", "max_structure",
                "Legs(n, lots=) declared a count, which cannot be margined or subscribed; declare each leg: "
                "Legs(OptionLeg(side='sell', right='call', offset=0, expiry=Expiry.weekly()), ...), "
                "and the largest position with Requirements(max_position={role: MaxPosition(lots=...)})",
            )
        if not legs:
            _fail("invalid_value", "max_structure", "Legs needs at least one OptionLeg")
        for leg in legs:
            if not isinstance(leg, OptionLeg):
                _fail("invalid_value", "max_structure", f"Legs holds OptionLeg, got {leg!r}")
        object.__setattr__(self, "legs", tuple(legs))

    def __setattr__(self, name: str, value: Any) -> None:
        raise dataclasses.FrozenInstanceError(f"cannot assign to field {name!r}")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Legs) and self.legs == other.legs

    def __hash__(self) -> int:
        return hash(self.legs)

    def __repr__(self) -> str:
        return f"Legs({', '.join(repr(leg) for leg in self.legs)})"

    def __len__(self) -> int:
        return len(self.legs)

    def __iter__(self):
        return iter(self.legs)

    def to_dict(self) -> dict[str, Any]:
        return {"legs": [leg.to_dict() for leg in self.legs]}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Legs:
        return Legs(*(OptionLeg.from_dict(x) for x in d["legs"]))

    def structure(self, chain: str | None = None, *, vol: float | None = None, name: str = "manifest"):
        """The legs on one chain role as a :class:`~raptorbt.structures.Structure`, one lot of
        the structure per leg ``lots``. ``chain`` selects the legs of that role (default: all);
        a ``delta`` leg needs ``vol``, the flat volatility the delta is priced at. Resolve it
        against an option chain and margin it::

            resolved = legs.structure(vol=0.18).resolve(chain, "NIFTY", spot, now_ns)
            resolved.margin(MarginModel.defined_risk(), prices)
        """
        from raptorbt.structures import Delta, Leg, Steps, Structure

        picked = [leg for leg in self.legs if chain is None or leg.chain in (None, chain)]
        built = []
        for leg in picked:
            if leg.delta is not None:
                if vol is None:
                    _fail("needs_vol", "vol", "a delta leg is priced at a volatility: structure(vol=...)")
                strike: Any = Delta(leg.delta, vol)
            else:
                strike = Steps(leg.offset)
            built.append(Leg(leg.right, strike, leg.expiry, leg.side, leg.lots))
        return Structure(tuple(built), name=name)


@dataclass(frozen=True)
class MaxPosition:
    """The largest position a strategy takes in one role, so a host can estimate the capital
    without asking: ``lots`` (per instrument the role resolves to: per contract of a chain, per
    member of a universe), ``notional`` (currency; per such instrument) and/or ``fraction`` of
    the capital (``0 < fraction <= 1``; per such instrument). At least one is given; when more
    are, the position is bounded by each."""

    lots: int | None = None
    notional: float | None = None
    fraction: float | None = None

    def __post_init__(self) -> None:
        if self.lots is None and self.notional is None and self.fraction is None:
            _fail("invalid_value", "max_position", "give lots=, notional= or fraction=")
        if self.lots is not None:
            _positive_int(self.lots, "max_position.lots", "lots")
        if self.notional is not None and (
            isinstance(self.notional, bool) or not isinstance(self.notional, (int, float))
            or not math.isfinite(self.notional) or self.notional <= 0
        ):
            _fail("invalid_value", "max_position.notional", f"a positive finite amount, got {self.notional!r}")
        if self.fraction is not None and (
            isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float))
            or not 0.0 < self.fraction <= 1.0
        ):
            _fail("invalid_value", "max_position.fraction", f"a fraction of the capital in (0, 1], got {self.fraction!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lots": self.lots,
            "notional": None if self.notional is None else float(self.notional),
            "fraction": None if self.fraction is None else float(self.fraction),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> MaxPosition:
        return MaxPosition(d["lots"], d["notional"], d["fraction"])


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
    ``warmup`` is :class:`Bars` or :class:`Duration`; ``max_structure`` is :class:`Legs`, the
    option legs of the structure the strategy holds; ``max_position`` maps a role name to a
    :class:`MaxPosition`, the largest position the strategy takes in it; ``params`` is a
    ``StrategyConfig`` subclass or a :class:`ParamSchema`.

    Construction validates each part on its own; :meth:`issues` checks them against each other.
    """

    roles: Mapping[str, Role] = field(default_factory=dict)
    timeframes: Sequence[str | Timeframe] = ()
    holding: Holding | None = None
    data: Sequence[str] = ("bar",)
    warmup: Bars | Duration | None = None
    max_structure: Legs | None = None
    params: Any = None
    max_position: Mapping[str, MaxPosition] | None = None

    def __post_init__(self) -> None:
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731
        roles = dict(self.roles)
        for name, role in roles.items():
            if not isinstance(name, str) or not name.isidentifier():
                _fail("invalid_role_name", f"roles.{name}", "a role name is a Python identifier")
            if not isinstance(role, (Underlying, Instrument, Universe, Future, Chain)):
                _fail("invalid_role", f"roles.{name}", f"an Underlying, Instrument, Universe, Future or Chain, got {role!r}")
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
            _fail("invalid_value", "max_structure", "Legs(OptionLeg(...), ...)")
        if self.max_position is not None:
            hints = dict(self.max_position)
            for role_name, hint in hints.items():
                if not isinstance(role_name, str) or not isinstance(hint, MaxPosition):
                    _fail("invalid_value", "max_position", f"a mapping of role name to MaxPosition, got {role_name!r}: {hint!r}")
            set_("max_position", types.MappingProxyType(hints))
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
        for name, role in self.roles.items():
            if isinstance(role, Universe):
                for kind in role.data:
                    if kind not in self.data:
                        out.append(Issue("role_data_not_declared", f"roles.{name}.data", f"members need {kind!r}, which data does not declare"))
        for name in self.max_position or {}:
            if name not in self.roles:
                out.append(Issue("unknown_role", f"max_position.{name}", f"{name!r} is not a role"))
        if self.max_structure is not None:
            out.extend(self._leg_issues())
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

    def _leg_issues(self) -> list[Issue]:
        """The option legs checked against the chains they sit on."""
        out: list[Issue] = []
        chains = {n: r for n, r in self.roles.items() if isinstance(r, Chain)}
        for i, leg in enumerate(self.max_structure.legs):
            path = f"max_structure.legs[{i}]"
            if leg.chain is None:
                if len(chains) != 1:
                    out.append(Issue(
                        "leg_chain_ambiguous", f"{path}.chain",
                        "name the Chain role the leg is on" if chains else "the strategy declares no Chain role for the leg",
                    ))
                    continue
                name = next(iter(chains))
            elif leg.chain not in chains:
                out.append(Issue("unknown_role", f"{path}.chain", f"{leg.chain!r} is not a Chain role"))
                continue
            else:
                name = leg.chain
            chain = chains[name]
            if leg.right not in chain.rights:
                out.append(Issue("leg_outside_chain", f"{path}.right", f"roles.{name} lists {', '.join(chain.rights)}, not {leg.right}s"))
            if leg.expiry not in chain.expiry:
                out.append(Issue("leg_outside_chain", f"{path}.expiry", f"roles.{name} does not list this expiry"))
            reach = leg.offset if leg.offset is not None else leg.window
            strikes = chain.strikes
            if isinstance(strikes, StrikeWindow):
                lows = [leg.offset] if leg.offset is not None else [-leg.window, leg.window]
                if any(not strikes.low <= o <= strikes.high for o in lows):
                    out.append(Issue(
                        "leg_outside_chain", f"{path}.strike",
                        f"strikes {min(lows)}..{max(lows)} from the money are outside roles.{name}.strikes ({strikes.low}..{strikes.high})",
                    ))
            elif leg.offset is not None:
                out.append(Issue(
                    "leg_outside_chain", f"{path}.offset",
                    f"roles.{name} picks strikes by delta, so a leg at offset {reach} may not be listed: use StrikeWindow",
                ))
            elif not strikes.low <= leg.delta <= strikes.high:
                out.append(Issue("leg_outside_chain", f"{path}.delta", f"delta {leg.delta} is outside roles.{name}.strikes ({strikes.low}..{strikes.high})"))
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
            "max_position": {n: h.to_dict() for n, h in self.max_position.items()} if self.max_position else None,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Requirements:
        """The inverse of :meth:`to_dict`. ``params`` comes back as a :class:`ParamSchema`."""
        if d.get("schema") != SCHEMA:
            _fail("unsupported_schema", "schema", f"expected {SCHEMA}, got {d.get('schema')!r}; export the manifest again with this raptorbt")
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
            max_structure=Legs.from_dict(m) if m else None,
            params=ParamSchema.from_dict(p) if p is not None else None,
            max_position={n: MaxPosition.from_dict(h) for n, h in d["max_position"].items()} if d.get("max_position") else None,
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


def universe_binding_problem(role: Universe, members: Any) -> tuple[str, str] | None:
    """Why ``members`` cannot be the binding of a :class:`Universe` role, as ``(code, message)``
    (``invalid_binding`` for the wrong shape, ``universe_size`` for a count outside min..max), or
    ``None`` if it can. A missing binding (``None``) is not judged here."""
    if members is None:
        return None
    if isinstance(members, str) or not isinstance(members, Sequence):
        return "invalid_binding", f"a Universe is bound to a list of symbols, got {members!r}"
    if not all(isinstance(m, str) and m for m in members):
        return "invalid_binding", f"a Universe is bound to non-empty symbol strings, got {list(members)!r}"
    if len(set(members)) != len(members):
        return "invalid_binding", f"a symbol is listed twice in {list(members)!r}"
    if not role.min <= len(members) <= role.max:
        return "universe_size", f"{len(members)} symbols bound; the role takes {role.min} to {role.max}"
    return None


def resolve(
    requirements: Requirements,
    *,
    now_ns: int,
    bindings: Mapping[str, str | Sequence[str]] | None = None,
    specs: Iterable[Any] = (),
    prices: Mapping[str, float] | None = None,
    vols: Mapping[str, float] | None = None,
    chain: Any = None,
) -> SymbolPlan:
    """Which concrete symbols each role means at ``now_ns``, so a host and a backtest resolve
    roles the same way. Never raises for data it lacks: that is an entry in ``issues`` and the
    role resolves to nothing.

    - ``bindings`` maps each :class:`Underlying` role to its symbol and each :class:`Universe`
      role to its list of symbols (``unbound_role`` if missing, ``invalid_binding`` for the wrong
      shape, ``universe_size`` for a count outside the role's min..max).
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
            elif not isinstance(sym, str):
                need("invalid_binding", role_name, f"an Underlying is bound to one symbol, got {sym!r}")
                sym = None
            done(RolePlan(role_name, role.kind, (sym,) if sym else ()))
        elif isinstance(role, Universe):
            members = bindings.get(role_name)
            problem = universe_binding_problem(role, members)
            if members is None:
                need("unbound_role", role_name, "no symbols bound to this role")
            elif problem is not None:
                need(problem[0], role_name, problem[1])
            done(RolePlan(role_name, role.kind, tuple(members) if members is not None and problem is None else ()))
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


@dataclass(frozen=True)
class LegContracts:
    """The contracts one declared leg can mean at a time. An ``offset`` leg is one contract;
    a ``delta`` leg is any contract of its right within ``window`` strikes of the money (the run
    picks the one nearest the delta), so ``symbols`` are the candidates to subscribe to and to
    margin (the host takes the dearest). ``quantity`` is the leg's units, ``lots`` times the
    lot size of the first contract; ``sign`` is +1 for a buy and -1 for a sell, the sign margin
    models read."""

    index: int
    leg: OptionLeg
    symbols: tuple[str, ...]
    specs: tuple[Any, ...]
    quantity: float
    expiry_ns: int

    @property
    def sign(self) -> int:
        return 1 if self.leg.side == "buy" else -1


@dataclass(frozen=True)
class LegPlan:
    """:func:`resolve_legs`: the contracts of every declared leg at ``now_ns``. ``symbols`` is all
    of them, without repeats. ``issues`` lists what could not be resolved; a plan with issues is
    partial."""

    now_ns: int
    legs: tuple[LegContracts, ...]
    symbols: tuple[str, ...]
    issues: tuple[Issue, ...] = ()

    def lines(self, prices: Mapping[str, float]) -> list[tuple[Any, float, float]]:
        """``(spec, signed units, price)`` of the first candidate of each leg, the argument of
        ``MarginModel.required_legs``. A delta leg's other candidates are
        ``LegContracts.specs``; margin each when the dearest matters."""
        return [(c.specs[0], c.sign * c.quantity, float(prices[c.symbols[0]])) for c in self.legs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "now_ns": self.now_ns,
            "legs": [
                {"index": c.index, "symbols": list(c.symbols), "quantity": c.quantity,
                 "sign": c.sign, "expiry_ns": c.expiry_ns}
                for c in self.legs
            ],
            "symbols": list(self.symbols),
            "issues": [i.to_dict() for i in self.issues],
        }


def resolve_legs(
    requirements: Requirements,
    *,
    now_ns: int,
    bindings: Mapping[str, str | Sequence[str]] | None = None,
    specs: Iterable[Any] = (),
    prices: Mapping[str, float] | None = None,
    chain: Any = None,
) -> LegPlan:
    """The contracts the declared ``max_structure`` legs can mean at ``now_ns``, so a host can
    subscribe to them and margin the structure without running the strategy. Roles resolve as in
    :func:`resolve`; never raises for data it lacks (an ``Issue``, and that leg is left out).

    An ``offset`` leg is the contract that many listed strikes from the money strike (the strike
    nearest ``prices[underlying]``, the lower on a tie) at the leg's expiry. A ``delta`` leg is
    every contract of its right within ``window`` strikes of the money.
    """
    from raptorbt import OptionChain

    if requirements.max_structure is None:
        return LegPlan(now_ns, (), (), ())
    base = resolve(requirements, now_ns=now_ns, bindings=bindings, specs=specs, prices=prices, chain=chain)
    if not base.roles:
        return LegPlan(now_ns, (), (), base.issues)
    issues: list[Issue] = [i for i in requirements.issues() if i.path.startswith("max_structure")]
    if issues:
        return LegPlan(now_ns, (), (), tuple(issues))
    options = [s for s in specs if s.kind == "option"]
    if chain is None:
        chain = OptionChain(options) if options else None
    chains = [n for n, r in requirements.roles.items() if isinstance(r, Chain)]
    prices = prices or {}
    out: list[LegContracts] = []
    for i, leg in enumerate(requirements.max_structure.legs):
        path = f"max_structure.legs[{i}]"
        role = leg.chain or chains[0]
        u = (base.roles[requirements.roles[role].of].symbols or (None,))[0]
        if u is None or chain is None:
            issues.append(Issue("no_options", path, f"no option contracts listed for the underlying of roles.{role}"))
            continue
        try:
            ns = leg.expiry.resolve(chain, u, now_ns)
        except RaptorBTError:
            issues.append(Issue("no_expiry", path, f"no {leg.expiry.which} expiry (n={leg.expiry.n}) listed for {u!r} at {now_ns}"))
            continue
        price = prices.get(u)
        if price is None:
            issues.append(Issue("needs_price", path, f"the price of {u!r} is needed to find the money strike"))
            continue
        ladder = chain.strikes(u, ns, now_ns)
        money = chain.atm_strike(u, ns, now_ns, price)
        if not ladder or money is None:
            issues.append(Issue("no_options", path, f"no strikes listed for {u!r} at expiry {ns}"))
            continue
        at = ladder.index(money)
        if leg.offset is not None:
            picks = [at + leg.offset] if 0 <= at + leg.offset < len(ladder) else []
        else:
            picks = range(max(0, at - leg.window), min(len(ladder), at + leg.window + 1))
        found = [c for c in (chain.contract(u, ns, ladder[j], leg.right, now_ns) for j in picks) if c is not None]
        if not found:
            issues.append(Issue("leg_strike_unlisted", path, f"no {leg.right} listed {leg.offset if leg.offset is not None else leg.window} strikes from the money for {u!r} at expiry {ns}"))
            continue
        out.append(LegContracts(
            i, leg, tuple(c.symbol for c in found), tuple(found), leg.lots * found[0].lot_size, ns,
        ))
    symbols = tuple(dict.fromkeys(s for c in out for s in c.symbols))
    return LegPlan(now_ns, tuple(out), symbols, tuple(issues))


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
    bound = {s for v in bindings.values() for s in ((v,) if isinstance(v, str) else v)}
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
            if symbol not in specs and symbol not in bound:
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
    declaration (every role bound to a synthetic symbol, so ``ctx.role`` answers for each of them,
    a universe to a few synthetic ones, a synthetic option universe for each chain, default
    config) and reports any subscription outside it: an instrument no role
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

    # Every Underlying role is bound to its upper-cased name and every Universe role to a few
    # numbered ones; futures and options are listed for the Underlyings. Futures resolve first,
    # since a chain can hang off a future.
    bindings: dict[str, Any] = {}
    for name, r in req.roles.items():
        if isinstance(r, Underlying):
            bindings[name] = name.upper()
        elif isinstance(r, Universe):
            # three members, or as many as the role takes
            bindings[name] = tuple(f"{name.upper()}{k}" for k in range(1, min(max(r.min, 3), r.max) + 1))
    bases = [s for name, r in req.roles.items() if isinstance(r, Underlying) for s in (bindings[name],)]
    specs = [
        InstrumentSpec.futures_contract(f"{base}F{k}", _T0 + 30 * _DAY * (k + 1), 1.0, underlying=base)
        for base in bases for k in range(2)
    ]
    members = [s for v in bindings.values() for s in ((v,) if isinstance(v, str) else v)]
    prices = {s: 100.0 for s in [*members, *(sp.symbol for sp in specs)]}
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
    # Every listed contract goes in as an instrument, with or without bars: the contracts a
    # strategy reads with ``ctx.role`` or ``ctx.option_chain`` are those of the whole listing
    # (a weekly expiry is told from a monthly one by the expiries around it).
    listed = {sp.symbol: sp for sp in specs}
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
                           bar_stamp="close", bindings=bindings)
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
