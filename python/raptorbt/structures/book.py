"""Submitting a structure as one group of ordinary orders, and managing it as a group.

Every leg is an ordinary order placed with ``Strategy.submit_order`` and read back from
``Strategy.cache``; nothing here touches the engine, so a group behaves the same in
``run_strategies``, the portfolio and tick runners and ``LiveNode``.

Leg failure
-----------
Legs are sent protective-first (shares and bought options, then sold options), each as its own
order with the client id ``"<group>:<leg index>"`` and ``tags={"structure": group}``. They fill
(or not) independently. A leg is *failed* when its order was rejected, cancelled or expired.
With ``on_leg_failure="unwind"`` (the default) the first failed leg starts an unwind: every leg
still working is cancelled and every leg that filled is closed with a ``reduce_only`` market
order, and the group ends ``closed`` with reason ``leg_failure``. With ``"keep"`` nothing is
undone: the group reads ``broken`` and the strategy decides. The unwind runs from
``update_structures`` (call it from ``on_bar``), or in the same event step when the strategy
uses :class:`StructureEvents`. A leg that fills after the unwind began is closed on the next
update, so a cancel that loses a race with a fill leaves no naked leg; a close that is itself
refused is retried (three attempts) and then reported as ``stuck``.

Fills are read from the cache as whole-order fills. With ``partial_fills`` on, a part-filled
leg reads as filled, and its close is sent for the planned quantity and is ``reduce_only``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from raptorbt.errors import InvalidParameter, StateError
from raptorbt.strategy.orders import Limit, Market
from raptorbt.structures.legs import ResolvedLeg, ResolvedStructure, Structure, _invalid

if TYPE_CHECKING:
    from raptorbt.strategy.base import Strategy

_DAY_NS = 86_400 * 10**9
_FAILED = ("rejected", "canceled", "expired")
_CLOSE_ATTEMPTS = 3


@dataclass(frozen=True)
class Exit:
    """When to close the whole group. Every field is optional; the first that fires wins.

    ``stop_loss`` / ``take_profit``: net P&L of the legs against their fill prices, in currency
    (positive numbers; costs excluded), checked once every leg has filled.
    ``premium_at_or_above`` / ``premium_at_or_below``: the net premium of one lot at the marks
    (see ``ResolvedStructure.net_premium``), checked once every leg has filled.
    ``close_at_ns``: close at or after this time, whatever has filled.
    ``days_before_expiry``: close this many days before the earliest option expiry.
    ``entry_timeout_ns``: unwind if the legs have not all filled this long after submission."""

    stop_loss: float | None = None
    take_profit: float | None = None
    premium_at_or_above: float | None = None
    premium_at_or_below: float | None = None
    close_at_ns: int | None = None
    days_before_expiry: float | None = None
    entry_timeout_ns: int | None = None

    def __post_init__(self) -> None:
        for name in ("stop_loss", "take_profit"):
            v = getattr(self, name)
            if v is not None and not (math.isfinite(v) and v > 0):
                raise _invalid(f"{name} is a positive amount of currency", name, v)
        for name in ("premium_at_or_above", "premium_at_or_below"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise _invalid(f"{name} must be finite", name, v)
        if self.days_before_expiry is not None and not self.days_before_expiry >= 0:
            raise _invalid("days_before_expiry must be non-negative", "days_before_expiry", self.days_before_expiry)
        if self.entry_timeout_ns is not None and not self.entry_timeout_ns > 0:
            raise _invalid("entry_timeout_ns must be positive", "entry_timeout_ns", self.entry_timeout_ns)

    @property
    def needs_marks(self) -> bool:
        return any(
            v is not None
            for v in (self.stop_loss, self.take_profit, self.premium_at_or_above, self.premium_at_or_below)
        )


@dataclass
class LegState:
    index: int
    client_id: str
    leg: ResolvedLeg
    cancel_sent: bool = False
    close_ids: list[str] = field(default_factory=list)


@dataclass
class StructureGroup:
    """The handle ``submit_structure`` returns. Read ``status`` / ``reason``; do not mutate."""

    id: str
    structure: ResolvedStructure
    legs: list[LegState]
    exit: Exit | None
    on_leg_failure: str
    submitted_ns: int
    phase: str = "entry"  #: entry | closing | closed
    reason: str | None = None  #: why it closed: leg_failure, entry_timeout, <exit rule>, manual

    def _entry(self, cache, ls: LegState) -> str:
        return _order_state(cache, ls.client_id)

    def leg_status(self, cache, index: int) -> str:
        """``submitted``, ``working``, ``filled``, ``rejected``, ``canceled`` or ``expired``."""
        return _order_state(cache, self.legs[index].client_id)

    def status(self, cache) -> str:
        """``submitted`` (no leg heard from), ``working``, ``filled`` (every leg), ``broken``
        (a leg failed, others did not; ``on_leg_failure="keep"``), ``closing``, ``closed`` or
        ``stuck`` (a close order was refused three times)."""
        if self.phase == "closed":
            return "closed"
        if self.phase == "closing":
            return "stuck" if self._stuck(cache) else "closing"
        states = [self._entry(cache, ls) for ls in self.legs]
        if any(s in _FAILED for s in states):
            return "broken"
        if all(s == "filled" for s in states):
            return "filled"
        if all(s == "submitted" for s in states):
            return "submitted"
        return "working"

    def fill_price(self, cache, index: int) -> float | None:
        rec = cache.order(self.legs[index].client_id)
        return None if rec is None or rec.status != "filled" else rec.last_price

    def pnl(self, cache, marks: Mapping[str, float]) -> float:
        """Net P&L of the filled legs at ``marks`` against their fill prices, before costs."""
        total = 0.0
        for ls in self.legs:
            fill = self.fill_price(cache, ls.index)
            if fill is None:
                continue
            total += ls.leg.sign * (_mark(marks, ls.leg) - fill) * ls.leg.quantity * float(ls.leg.spec.multiplier)
        return total

    def _stuck(self, cache) -> bool:
        for ls in self.legs:
            if len(ls.close_ids) >= _CLOSE_ATTEMPTS and _order_state(cache, ls.close_ids[-1]) in _FAILED:
                return True
        return False


def _order_state(cache, client_id: str) -> str:
    rec = cache.order(client_id)
    if rec is None:
        return "submitted"
    if rec.status in ("accepted", "triggered"):
        return "working"
    return rec.status


def _mark(marks: Mapping[str, float], leg: ResolvedLeg) -> float:
    try:
        return float(marks[leg.instrument])
    except KeyError:
        raise StateError(
            f"no mark for {leg.instrument}",
            code="structure_missing_mark",
            operation="update_structures",
            parameter="marks",
            value=leg.instrument,
        ) from None


class StructureBook:
    """The groups one strategy has placed in one run."""

    def __init__(self, strategy: "Strategy") -> None:
        self.strategy = strategy
        self.cache = strategy.cache
        self.groups: list[StructureGroup] = []
        self._seq = 0

    # -- submit ---------------------------------------------------------------------------

    def submit(
        self,
        structure: Structure | ResolvedStructure,
        *,
        now_ns: int,
        chain=None,
        underlying: str | None = None,
        spot: float | None = None,
        lots: int = 1,
        underlying_spec=None,
        order_type: str = "market",
        prices: Mapping[str, float] | None = None,
        net_limit: float | None = None,
        exit: Exit | None = None,
        on_leg_failure: str = "unwind",
        margin_model=None,
        available: float | None = None,
        tif: str = "gtc",
        group_id: str | None = None,
    ) -> StructureGroup:
        if order_type not in ("market", "limit"):
            raise _invalid("order_type is 'market' or 'limit'", "order_type", order_type, ["market", "limit"])
        if on_leg_failure not in ("unwind", "keep"):
            raise _invalid(
                "on_leg_failure is 'unwind' or 'keep'", "on_leg_failure", on_leg_failure, ["unwind", "keep"]
            )
        if isinstance(structure, ResolvedStructure):
            resolved = structure
        else:
            if chain is None or underlying is None or spot is None:
                raise _invalid("a Structure needs chain, underlying and spot to resolve", "chain")
            resolved = structure.resolve(chain, underlying, spot, now_ns, lots, underlying_spec=underlying_spec)

        symbols = [leg.instrument for leg in resolved.legs]
        if len(set(symbols)) != len(symbols):
            raise _invalid("two legs resolve to the same contract; merge them into one leg with a ratio", "legs", symbols)
        if (order_type == "limit" or net_limit is not None or margin_model is not None) and prices is None:
            raise _invalid("prices (a price for every leg's symbol) are needed for limit orders, net_limit or margin", "prices")
        if prices is not None:
            for leg in resolved.legs:
                p = prices.get(leg.instrument)
                if p is None or not (math.isfinite(p) and p > 0):
                    raise _invalid(f"prices needs a positive price for {leg.instrument}", "prices", leg.instrument)
        if net_limit is not None and resolved.net_premium(prices) > net_limit:
            raise InvalidParameter(
                f"net premium {resolved.net_premium(prices):g} is worse than net_limit {net_limit:g}",
                code="structure_limit_breached",
                operation="submit_structure",
                parameter="net_limit",
                value=net_limit,
            )
        if (margin_model is None) != (available is None):
            raise _invalid("margin_model and available go together", "available", available)
        if margin_model is not None:
            required = resolved.margin(margin_model, prices, now_ns)
            if required > available:
                raise InvalidParameter(
                    f"the group needs margin {required:g}, only {available:g} is available",
                    code="structure_margin_exceeded",
                    operation="submit_structure",
                    parameter="available",
                    value=available,
                )

        gid = group_id or self._next_id()
        if any(g.id == gid for g in self.groups):
            raise _invalid(f"group id {gid!r} is already in use", "group_id", gid)
        orders: list[tuple[int, Any]] = []
        for i in resolved.submission_order():
            leg = resolved.legs[i]
            common = dict(
                side=leg.side, quantity=leg.quantity, instrument=leg.instrument, tif=tif,
                tags={"structure": gid},
            )
            order = Limit(price=float(prices[leg.instrument]), **common) if order_type == "limit" else Market(**common)
            orders.append((i, order))
        states = [LegState(i, f"{gid}:{i}", resolved.legs[i]) for i in range(len(resolved.legs))]
        group = StructureGroup(gid, resolved, states, exit, on_leg_failure, now_ns)
        for i, order in orders:
            self.strategy.submit_order(order, client_id=states[i].client_id)
        self.groups.append(group)
        return group

    def _next_id(self) -> str:
        tag = getattr(self.strategy.config, "order_id_tag", None) or "S"
        gid = f"{tag}-st{self._seq}"
        self._seq += 1
        return gid

    # -- manage ---------------------------------------------------------------------------

    def close(self, group: StructureGroup, reason: str = "manual") -> None:
        """Start closing the group: cancel what works, close what filled."""
        if group.phase == "entry":
            group.phase = "closing"
            group.reason = reason
            self._pump(group)

    def update(
        self,
        now_ns: int,
        marks: Mapping[str, float] | None = None,
        *,
        exits: bool = True,
    ) -> list[StructureGroup]:
        """Advance every group: leg-failure policy, exits, and the pending cancels and closes.
        Returns the groups that began closing. ``exits=False`` runs only the lifecycle (leg
        failure, entry timeout and the pump), which needs no marks."""
        began: list[StructureGroup] = []
        for g in self.groups:
            if g.phase == "closed":
                continue
            if g.phase == "entry":
                reason = self._reason(g, now_ns, marks, exits)
                if reason is not None:
                    g.phase = "closing"
                    g.reason = reason
                    began.append(g)
            if g.phase == "closing":
                self._pump(g)
        return began

    def _reason(self, g: StructureGroup, now_ns: int, marks, exits: bool) -> str | None:
        cache = self.cache
        states = [g._entry(cache, ls) for ls in g.legs]
        if g.on_leg_failure == "unwind" and any(s in _FAILED for s in states):
            return "leg_failure"
        ex = g.exit
        if ex is None:
            return None
        all_filled = all(s == "filled" for s in states)
        if ex.entry_timeout_ns is not None and not all_filled and now_ns >= g.submitted_ns + ex.entry_timeout_ns:
            return "entry_timeout"
        if not exits:
            return None
        if ex.close_at_ns is not None and now_ns >= ex.close_at_ns:
            return "close_at"
        if ex.days_before_expiry is not None:
            earliest = min(leg.expiry_ns for leg in g.structure.option_legs)
            if now_ns >= earliest - ex.days_before_expiry * _DAY_NS:
                return "days_before_expiry"
        if not (ex.needs_marks and all_filled):
            return None
        if marks is None:
            raise StateError(
                "this group has P&L or premium exits and needs marks",
                code="structure_missing_mark",
                operation="update_structures",
                parameter="marks",
            )
        pnl = g.pnl(cache, marks)
        if ex.stop_loss is not None and pnl <= -ex.stop_loss:
            return "stop_loss"
        if ex.take_profit is not None and pnl >= ex.take_profit:
            return "take_profit"
        if ex.premium_at_or_above is not None or ex.premium_at_or_below is not None:
            prem = sum(
                leg.sign * leg.ratio * _mark(marks, leg) for leg in g.structure.option_legs
            )
            if ex.premium_at_or_above is not None and prem >= ex.premium_at_or_above:
                return "premium_at_or_above"
            if ex.premium_at_or_below is not None and prem <= ex.premium_at_or_below:
                return "premium_at_or_below"
        return None

    def _pump(self, g: StructureGroup) -> None:
        cache = self.cache
        done = True
        for ls in g.legs:
            state = g._entry(cache, ls)
            if state in ("submitted", "working"):
                done = False
                if not ls.cancel_sent:
                    self.strategy.cancel_order(ls.client_id)
                    ls.cancel_sent = True
                continue
            if state != "filled":
                continue
            if ls.close_ids:
                last = _order_state(cache, ls.close_ids[-1])
                if last == "filled":
                    continue
                done = False
                if last in _FAILED and len(ls.close_ids) < _CLOSE_ATTEMPTS:
                    self._send_close(g, ls)
                continue
            done = False
            self._send_close(g, ls)
        if done:
            g.phase = "closed"

    def _send_close(self, g: StructureGroup, ls: LegState) -> None:
        cid = f"{g.id}:x{ls.index}.{len(ls.close_ids)}"
        leg = ls.leg
        self.strategy.submit_order(
            Market(
                side="sell" if leg.side == "buy" else "buy",
                quantity=leg.quantity,
                instrument=leg.instrument,
                reduce_only=True,
                tags={"structure": g.id, "close": g.reason or "manual"},
            ),
            client_id=cid,
        )
        ls.close_ids.append(cid)


class StructureEvents:
    """Mix into a strategy (before ``Strategy``) to run the group lifecycle in the same event step
    as the order event that caused it, instead of on the next ``update_structures`` call::

        class Condor(StructureEvents, Strategy): ...

    It overrides ``on_order_event``; a strategy that defines its own must call
    ``self.update_structures(ctx, exits=False)`` there."""

    def on_order_event(self, ctx, event) -> None:
        self.update_structures(ctx, exits=False)
