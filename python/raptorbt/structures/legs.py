"""Legs, selectors and the resolution of a structure against an option chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

from raptorbt.errors import InvalidParameter, StateError

if TYPE_CHECKING:
    from raptorbt._raptorbt import InstrumentSpec, OptionChain

RIGHTS = ("call", "put")
SIDES = ("buy", "sell")


def _invalid(message: str, parameter: str, value: Any = None, valid=None) -> InvalidParameter:
    return InvalidParameter(
        message,
        code="invalid_structure",
        operation="structures",
        parameter=parameter,
        value=value,
        valid=valid,
    )


def _missing(message: str, parameter: str, value: Any = None) -> InvalidParameter:
    return InvalidParameter(
        message,
        code="structure_contract_missing",
        operation="Structure.resolve",
        parameter=parameter,
        value=value,
    )


# -- strike selectors -----------------------------------------------------------------------


@dataclass(frozen=True)
class Steps:
    """``n`` strikes from the money strike on the structure's anchor ladder, up positive.

    The money strike is the listed strike nearest the spot (the lower on a tie); the anchor
    ladder is the strikes of the structure's earliest expiry. A leg at a later expiry takes the
    same strike value there and is refused if that expiry does not list it.
    """

    n: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool):
            raise _invalid("Steps.n must be an integer", "n", self.n)


def Atm() -> Steps:  # noqa: N802 - reads as a selector name
    """The money strike: ``Steps(0)``."""
    return Steps(0)


@dataclass(frozen=True)
class Strike:
    """An explicit strike value; the leg's expiry must list it."""

    value: float


@dataclass(frozen=True)
class Delta:
    """The listed contract of the leg's right whose delta is nearest ``target`` in magnitude
    (0.25 picks a 25-delta call or put), priced at a flat ``vol`` at the event time, on the
    leg's own expiry. Ties go to the lower strike (``OptionChain.by_delta``)."""

    target: float
    vol: float
    rate: float = 0.0
    dividend_yield: float = 0.0
    model: str = "black_scholes"

    def __post_init__(self) -> None:
        if not (0.0 < abs(self.target) <= 1.0):
            raise _invalid("Delta.target is a magnitude in (0, 1]", "target", self.target)
        if not self.vol > 0.0:
            raise _invalid("Delta.vol must be positive", "vol", self.vol)


@dataclass(frozen=True)
class Premium:
    """The contract whose ``measure(spec)`` is nearest ``target`` (a quoted premium, say);
    contracts for which it returns ``None`` or NaN are skipped (``OptionChain.closest_by``)."""

    target: float
    measure: Callable[[Any], float | None]


StrikeSelector = Steps | Strike | Delta | Premium


# -- expiry selectors -------------------------------------------------------------------------


@dataclass(frozen=True)
class Expiry:
    """Which listed expiry, as ``OptionChain.pick_expiry`` reads it at the event time."""

    which: str = "nearest"
    n: int = 0
    min_days: int = 0
    date_ns: int | None = None
    utc_offset_ns: int = 0
    exact_ns: int | None = None

    @staticmethod
    def nearest(min_days: int = 0) -> "Expiry":
        return Expiry("nearest", min_days=min_days)

    @staticmethod
    def nth(n: int) -> "Expiry":
        return Expiry("nth", n=n)

    @staticmethod
    def monthly(n: int = 0, utc_offset_ns: int = 0) -> "Expiry":
        return Expiry("monthly", n=n, utc_offset_ns=utc_offset_ns)

    @staticmethod
    def weekly(n: int = 0, utc_offset_ns: int = 0) -> "Expiry":
        return Expiry("weekly", n=n, utc_offset_ns=utc_offset_ns)

    @staticmethod
    def on(date_ns: int, utc_offset_ns: int = 0) -> "Expiry":
        """The expiry on the calendar day of ``date_ns``."""
        return Expiry("date", date_ns=date_ns, utc_offset_ns=utc_offset_ns)

    @staticmethod
    def at(expiry_ns: int) -> "Expiry":
        """This exact expiry; refused if it is not listed at the event time."""
        return Expiry("exact", exact_ns=expiry_ns)

    def resolve(self, chain: "OptionChain", underlying: str, now_ns: int) -> int:
        if self.which == "exact":
            if self.exact_ns not in chain.expiries(underlying, now_ns):
                raise _missing(
                    f"{underlying}: expiry {self.exact_ns} is not listed at {now_ns}",
                    "expiry",
                    self.exact_ns,
                )
            return self.exact_ns
        if self.which == "nearest":
            got = chain.pick_expiry(underlying, now_ns, "nearest", min_days=self.min_days)
        elif self.which in ("nth", "monthly", "weekly"):
            got = chain.pick_expiry(
                underlying, now_ns, self.which, n=self.n, utc_offset_ns=self.utc_offset_ns
            )
        elif self.which == "date":
            got = chain.pick_expiry(
                underlying, now_ns, "date", date_ns=self.date_ns, utc_offset_ns=self.utc_offset_ns
            )
        else:
            raise _invalid(
                f"unknown expiry selector {self.which!r}",
                "which",
                self.which,
                ["nearest", "nth", "monthly", "weekly", "date", "exact"],
            )
        if got is None:
            raise _missing(f"{underlying}: no {self.which} expiry listed at {now_ns}", "expiry", self.which)
        return got


# -- legs ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    """One option leg: ``ratio`` lots of the contract ``strike`` and ``expiry`` pick, per lot of
    the structure."""

    right: str
    strike: StrikeSelector = field(default_factory=Atm)
    expiry: Expiry = field(default_factory=Expiry)
    side: str = "buy"
    ratio: int = 1

    def __post_init__(self) -> None:
        if self.right not in RIGHTS:
            raise _invalid(f"right must be 'call' or 'put', got {self.right!r}", "right", self.right, RIGHTS)
        if self.side not in SIDES:
            raise _invalid(f"side must be 'buy' or 'sell', got {self.side!r}", "side", self.side, SIDES)
        if not isinstance(self.ratio, int) or isinstance(self.ratio, bool) or self.ratio < 1:
            raise _invalid("ratio must be a whole number of at least 1", "ratio", self.ratio)
        if not isinstance(self.strike, (Steps, Strike, Delta, Premium)):
            raise _invalid("strike must be Steps, Strike, Delta or Premium", "strike", self.strike)
        if not isinstance(self.expiry, Expiry):
            raise _invalid("expiry must be an Expiry", "expiry", self.expiry)


@dataclass(frozen=True)
class UnderlyingLeg:
    """Shares (or the underlying instrument) held against the options: ``ratio`` option lots'
    worth of units per lot of the structure. The units must be a multiple of the underlying's
    lot size."""

    side: str = "buy"
    ratio: int = 1

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise _invalid(f"side must be 'buy' or 'sell', got {self.side!r}", "side", self.side, SIDES)
        if not isinstance(self.ratio, int) or isinstance(self.ratio, bool) or self.ratio < 1:
            raise _invalid("ratio must be a whole number of at least 1", "ratio", self.ratio)


@dataclass(frozen=True)
class ResolvedLeg:
    """A leg bound to a listed contract, with its size."""

    instrument: str  #: the symbol an order for this leg names
    spec: "InstrumentSpec"
    side: str
    lots: int  #: ``ratio`` x the structure's lots (option lots; for shares, option lots' worth)
    quantity: float  #: units: lots x the option's lot size
    right: str | None = None  #: ``None`` for the underlying leg
    strike: float | None = None
    expiry_ns: int | None = None
    ratio: int = 1

    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1

    @property
    def is_option(self) -> bool:
        return self.right is not None


@dataclass(frozen=True)
class ResolvedStructure:
    """A structure bound to contracts at one event time. Leg order is the template's."""

    name: str
    underlying: str
    now_ns: int
    lots: int
    legs: tuple[ResolvedLeg, ...]
    anchor_expiry_ns: int

    @property
    def option_legs(self) -> tuple[ResolvedLeg, ...]:
        return tuple(leg for leg in self.legs if leg.is_option)

    def net_premium(self, prices: Mapping[str, float]) -> float:
        """Debit (positive) or credit (negative) of one lot of the structure, per unit of the
        option: the signed sum of ``ratio x price`` over the option legs. Shares are left out."""
        total = 0.0
        for leg in self.option_legs:
            total += leg.sign * leg.ratio * _price(prices, leg)
        return total

    def margin(self, model: Any, prices: Mapping[str, float], now_ns: int | None = None) -> float:
        """Initial margin of the whole group under ``model``
        (``MarginModel.required_legs``): the option legs margined together, as a hedged book is."""
        lines = [(leg.spec, leg.sign * leg.quantity, _price(prices, leg)) for leg in self.option_legs]
        if now_ns is None:
            now_ns = self.now_ns
        required = model.required_legs(lines, now_ns)
        for leg in self.legs:
            if not leg.is_option:
                required += float(model.required(leg.spec, leg.sign * leg.quantity, _price(prices, leg)))
        return float(required)

    def submission_order(self) -> tuple[int, ...]:
        """Indices of the legs in the order they are sent: shares and bought options first, then
        sold options, so that a protective leg is on before the leg it protects."""
        idx = range(len(self.legs))
        return tuple(sorted(idx, key=lambda i: (0 if not self.legs[i].is_option else 1 if self.legs[i].side == "buy" else 2, i)))


def _price(prices: Mapping[str, float], leg: ResolvedLeg) -> float:
    try:
        p = float(prices[leg.instrument])
    except KeyError:
        raise StateError(
            f"no price for {leg.instrument}",
            code="structure_missing_mark",
            operation="ResolvedStructure",
            parameter="prices",
            value=leg.instrument,
        ) from None
    return p


def _pick_strike(
    sel: StrikeSelector,
    leg: Leg,
    chain: "OptionChain",
    underlying: str,
    now_ns: int,
    spot: float,
    expiry_ns: int,
    ladder: list[float],
    atm_index: int,
) -> "InstrumentSpec":
    if isinstance(sel, Delta):
        t = abs(sel.target) if leg.right == "call" else -abs(sel.target)
        spec = chain.by_delta(
            underlying, expiry_ns, now_ns, leg.right, t, spot, sel.vol,
            sel.rate, sel.dividend_yield, sel.model,
        )
        if spec is None:
            raise _missing(f"{underlying}: no {leg.right} priced for expiry {expiry_ns}", "strike", sel.target)
        return spec
    if isinstance(sel, Premium):
        spec = chain.closest_by(underlying, expiry_ns, now_ns, leg.right, sel.target, sel.measure)
        if spec is None:
            raise _missing(f"{underlying}: no {leg.right} with a measure for expiry {expiry_ns}", "strike", sel.target)
        return spec
    if isinstance(sel, Strike):
        strike = float(sel.value)
    else:
        j = atm_index + sel.n
        if not 0 <= j < len(ladder):
            raise _missing(f"{underlying}: the ladder has no strike {sel.n} steps from the money", "strike", sel.n)
        strike = ladder[j]
    spec = chain.contract(underlying, expiry_ns, strike, leg.right, now_ns)
    if spec is None:
        raise _missing(
            f"{underlying}: no {leg.right} at strike {strike} for expiry {expiry_ns}", "strike", strike
        )
    return spec


@dataclass(frozen=True)
class Structure:
    """A declarative multi-leg position: legs with strike and expiry selectors, no contracts yet.

    ``resolve`` binds it to the chain at an event time (``now_ns``); nothing listed after that
    time is visible. Use the factories (``straddle``, ``iron_condor``, ...) or build one from legs."""

    legs: tuple[Leg | UnderlyingLeg, ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        object.__setattr__(self, "legs", tuple(self.legs))
        if not any(isinstance(leg, Leg) for leg in self.legs):
            raise _invalid("a structure needs at least one option Leg", "legs", len(self.legs))
        if sum(isinstance(leg, UnderlyingLeg) for leg in self.legs) > 1:
            raise _invalid("a structure holds at most one UnderlyingLeg", "legs")
        for leg in self.legs:
            if not isinstance(leg, (Leg, UnderlyingLeg)):
                raise _invalid("legs are Leg or UnderlyingLeg", "legs", leg)

    def resolve(
        self,
        chain: "OptionChain",
        underlying: str,
        spot: float,
        now_ns: int,
        lots: int = 1,
        *,
        underlying_spec: "InstrumentSpec | None" = None,
    ) -> ResolvedStructure:
        """Bind to contracts and sizes. ``lots`` of the structure; a leg's lots are
        ``ratio x lots``. An ``UnderlyingLeg`` needs ``underlying_spec``."""
        if not isinstance(lots, int) or isinstance(lots, bool) or lots < 1:
            raise _invalid("lots must be a whole number of at least 1", "lots", lots)
        if not (spot > 0.0) or spot != spot or spot == float("inf"):
            raise _invalid("spot must be a positive finite price", "spot", spot)
        expiries = [
            leg.expiry.resolve(chain, underlying, now_ns) for leg in self.legs if isinstance(leg, Leg)
        ]
        anchor = min(expiries)
        ladder = list(chain.strikes(underlying, anchor, now_ns))
        atm = chain.atm_strike(underlying, anchor, now_ns, spot)
        if atm is None:
            raise _missing(f"{underlying}: no strikes for expiry {anchor}", "expiry", anchor)
        atm_index = ladder.index(atm)

        resolved: list[ResolvedLeg] = []
        lot_of_option: float | None = None
        it = iter(expiries)
        pending_underlying: list[int] = []
        for leg in self.legs:
            if isinstance(leg, UnderlyingLeg):
                pending_underlying.append(len(resolved))
                resolved.append(None)  # type: ignore[arg-type]
                continue
            expiry_ns = next(it)
            spec = _pick_strike(leg.strike, leg, chain, underlying, now_ns, spot, expiry_ns, ladder, atm_index)
            n = leg.ratio * lots
            resolved.append(
                ResolvedLeg(
                    instrument=spec.symbol,
                    spec=spec,
                    side=leg.side,
                    lots=n,
                    quantity=n * float(spec.lot_size),
                    right=leg.right,
                    strike=spec.strike,
                    expiry_ns=spec.expiration_ns,
                    ratio=leg.ratio,
                )
            )
            if lot_of_option is None:
                lot_of_option = float(spec.lot_size)
        for i in pending_underlying:
            leg = self.legs[i]
            assert isinstance(leg, UnderlyingLeg)
            if underlying_spec is None:
                raise _invalid("an UnderlyingLeg needs underlying_spec", "underlying_spec")
            first_option = next(r for r in resolved if r is not None)
            units = leg.ratio * lots * float(first_option.spec.lot_size)
            ulot = float(underlying_spec.lot_size)
            if units % ulot != 0:
                raise _invalid(
                    f"{units:g} units is not a multiple of {underlying_spec.symbol}'s lot {ulot:g}",
                    "underlying_spec",
                    units,
                )
            resolved[i] = ResolvedLeg(
                instrument=underlying_spec.symbol,
                spec=underlying_spec,
                side=leg.side,
                lots=leg.ratio * lots,
                quantity=units,
                ratio=leg.ratio,
            )
        return ResolvedStructure(
            name=self.name,
            underlying=underlying,
            now_ns=now_ns,
            lots=lots,
            legs=tuple(resolved),
            anchor_expiry_ns=anchor,
        )
