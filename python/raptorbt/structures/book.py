"""Submitting a structure as a group of ordinary orders, hedge-first, and managing it as a group.

Every leg is an ordinary order placed with ``Strategy.submit_order`` and read back from
``Strategy.cache``; nothing here touches the engine, so a group behaves the same in
``run_strategies``, the portfolio and tick runners and ``LiveNode`` on the in-process engine, on
``PaperExecution`` and on a broker adapter.

Hedge-first
-----------
The bought legs (shares and bought options, the hedges) are sent first. A sold leg is sent only
once the hedges that cover it have filled (a sold call is covered by bought calls and bought
shares, a sold put by bought puts; a sold leg nothing covers is sent with the hedges). With
partial fills a sold leg is sent for the quantity its hedges have filled and so cover (the least
filled of them, in whole lots), and again as more of them fill, so a sold quantity never runs ahead
of its hedge cover. In a ratio spread (more sold than bought) the covered quantity is the hedge's
size and the uncovered rest goes out with the last of the hedge; in a back spread (more bought than
sold) the sold leg waits until the whole designed hedge has filled. The sequencing runs from the order events the strategy
hears (the strategy base class forwards them here), so it needs no call from the strategy and
no capability of the venue.

Leg failure
-----------
Each order has the client id ``"<group>:<leg index>"`` (a sold leg sent in several parts has
``"<group>:<leg index>.<k>"`` for the later ones) and ``tags={"structure": group}``. A leg is
*failed* when one of its orders was rejected, cancelled or expired. With
``on_leg_failure="unwind"`` (the default) the first failed leg starts a close: every order still
working is cancelled and every leg that filled is closed with a ``reduce_only`` market order (the
close is an order of its own, tagged with the reason, journaled like any other), and the group
ends ``closed`` with reason ``leg_failure``; ``failure`` says which order failed and why. A sold
leg that was refused for want of funds is a failed leg like any other, reported once, with the
engine's reason. With ``"keep"`` nothing is undone: the group reads ``broken`` and the strategy
decides. An order is working until its fills reach its size (the cache says ``filled`` at the first
partial fill), so the remainder of a part-filled leg is cancelled with the rest. What filled is
closed at once, the sold legs first and the bought legs (their cover) only when no sold leg is open
or could still fill. A fill that arrives after its leg's close, even after the group read
``closed``, is closed in turn. A close that is refused is tried again on a later step, never in the
step it was refused, three attempts in all, and then reported as ``stuck``.
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
_EPS = 1e-9


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
    client_id: str  #: the id of the leg's first order
    leg: ResolvedLeg
    orders: list[str] = field(default_factory=list)  #: every entry order sent for the leg
    sent: float = 0.0  #: units sent so far
    filled: float = 0.0  #: units filled so far, from the order events
    cost: float = 0.0  #: sum of fill price x fill units, for the average fill price
    cancelled: set[str] = field(default_factory=set)
    close_ids: list[str] = field(default_factory=list)
    #: Units asked of each order (entry and close) and units its fills have delivered so far. An
    #: order whose fills fall short of its size is still working, whatever the cache's status says.
    requested: dict[str, float] = field(default_factory=dict)
    got: dict[str, float] = field(default_factory=dict)
    #: The close orders seen refused, cancelled or expired, and the time the last was seen: a
    #: refused close is tried again on a later step, never in the step it was refused.
    close_failed: set[str] = field(default_factory=set)
    retry_after_ns: int | None = None

    @property
    def average_price(self) -> float | None:
        return self.cost / self.filled if self.filled > 0 else None

    @property
    def close_need(self) -> float:
        """Units filled that no live or finished close order covers yet."""
        covered = math.fsum(
            self.got.get(c, 0.0) if c in self.close_failed else self.requested.get(c, 0.0)
            for c in self.close_ids
        )
        return self.filled - covered


def _covers(sold: ResolvedLeg, hedge: ResolvedLeg) -> bool:
    """Whether the bought ``hedge`` covers the sold leg: a sold call is covered by bought calls
    and bought shares, a sold put by bought puts, sold shares by bought calls."""
    if hedge.side != "buy":
        return False
    if sold.right is None:
        return hedge.right == "call"
    if sold.right == "call":
        return hedge.right in ("call", None)
    return hedge.right == "put"


@dataclass
class StructureGroup:
    """The handle ``submit_structure`` returns. Read ``status`` / ``reason`` / ``failure``; do not
    mutate."""

    id: str
    structure: ResolvedStructure
    legs: list[LegState]
    exit: Exit | None
    on_leg_failure: str
    submitted_ns: int
    phase: str = "entry"  #: entry | closing | closed
    reason: str | None = None  #: why it closed: leg_failure, entry_timeout, <exit rule>, manual
    #: The order that failed and why, once a leg failed: ``(client id, status, engine reason)``.
    failure: tuple[str, str, str | None] | None = None
    order_type: str = "market"
    tif: str = "auto"
    prices: Mapping[str, float] | None = None
    #: Per sold leg index, the legs that cover it (empty: sent with the hedges).
    covers: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def leg_status(self, cache, index: int) -> str:
        """``waiting`` (a sold leg whose hedges have not filled), ``submitted``, ``working``,
        ``filled``, ``rejected``, ``canceled`` or ``expired``."""
        return _leg_state(cache, self.legs[index])

    def status(self, cache) -> str:
        """``submitted`` (no leg heard from), ``working``, ``filled`` (every leg), ``broken``
        (a leg failed, others did not; ``on_leg_failure="keep"``), ``closing``, ``closed`` or
        ``stuck`` (a close order was refused three times)."""
        if self.phase == "closed":
            return "closed"
        if self.phase == "closing":
            return "stuck" if self._stuck(cache) else "closing"
        states = [_leg_state(cache, ls) for ls in self.legs]
        if any(s in _FAILED for s in states):
            return "broken"
        if all(s == "filled" for s in states):
            return "filled"
        if all(s in ("submitted", "waiting") for s in states):
            return "submitted"
        return "working"

    def fill_price(self, cache, index: int) -> float | None:
        """The leg's average fill price (``None`` before any fill)."""
        return self.legs[index].average_price

    def pnl(self, cache, marks: Mapping[str, float]) -> float:
        """Net P&L of the filled legs at ``marks`` against their fill prices, before costs."""
        total = 0.0
        for ls in self.legs:
            fill = ls.average_price
            if fill is None:
                continue
            total += ls.leg.sign * (_mark(marks, ls.leg) - fill) * ls.filled * float(ls.leg.spec.multiplier)
        return total

    def _stuck(self, cache) -> bool:
        return any(len(ls.close_failed) >= _CLOSE_ATTEMPTS and ls.close_need > _EPS for ls in self.legs)


def _order_state(cache, ls: LegState, client_id: str) -> str:
    """``submitted`` (nothing heard), ``working`` (including a partly filled order: the cache
    says ``filled`` at the first fill, the order is done only when its fills reach its size),
    ``filled``, ``rejected``, ``canceled`` or ``expired``."""
    rec = cache.order(client_id)
    if rec is None:
        return "submitted"
    if rec.status in ("accepted", "triggered"):
        return "working"
    if rec.status == "filled" and ls.got.get(client_id, 0.0) < ls.requested.get(client_id, 0.0) * (1 - 1e-9):
        return "working"
    return rec.status


def _leg_state(cache, ls: LegState) -> str:
    states = [_order_state(cache, ls, c) for c in ls.orders]
    for s in states:
        if s in _FAILED:
            return s
    if ls.filled >= ls.leg.quantity * (1 - 1e-9):
        return "filled"
    if not states:
        return "waiting"
    if ls.filled == 0 and all(s == "submitted" for s in states):
        return "submitted"
    return "working"


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
        # The book reads order events: switch the hook on before any leg is queued.
        strategy._queue.enable_order_events()
        self.cache = strategy.cache
        self.groups: list[StructureGroup] = []
        self._seq = 0
        self._owner: dict[str, tuple[StructureGroup, LegState]] = {}

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
        tif: str = "auto",
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
        states = [LegState(i, f"{gid}:{i}", resolved.legs[i]) for i in range(len(resolved.legs))]
        covers = {
            i: tuple(j for j, h in enumerate(resolved.legs) if _covers(leg, h))
            for i, leg in enumerate(resolved.legs)
            if leg.side == "sell"
        }
        group = StructureGroup(
            gid, resolved, states, exit, on_leg_failure, now_ns,
            order_type=order_type, tif=tif, prices=prices, covers=covers,
        )
        self.groups.append(group)
        self._release(group, False)
        return group

    def _next_id(self) -> str:
        tag = getattr(self.strategy.config, "order_id_tag", None) or "S"
        gid = f"{tag}-st{self._seq}"
        self._seq += 1
        return gid

    def _release(self, g: StructureGroup, after_fill: bool) -> None:
        """Send what the hedges allow: the bought legs at once, each sold leg for the part of its
        planned quantity its covering hedges have filled (a back spread's sold leg: all of it, once
        the whole hedge has filled)."""
        resolved = g.structure
        for i in resolved.submission_order():
            ls = g.legs[i]
            leg = ls.leg
            if ls.sent >= leg.quantity * (1 - 1e-9):
                continue
            hedges = g.covers.get(i, ())
            if not hedges:
                allowed = leg.quantity
            else:
                frac = min(min(g.legs[h].filled / g.legs[h].leg.quantity, 1.0) for h in hedges)
                hedge_qty = min(g.legs[h].leg.quantity for h in hedges)
                if frac >= 1 - 1e-9:
                    allowed = leg.quantity
                elif hedge_qty > leg.quantity * (1 + 1e-9):
                    # a back spread (more bought than sold) waits for its whole designed hedge
                    allowed = 0.0
                else:
                    # the sold units the hedge filled so far covers; in a ratio spread (more sold
                    # than bought) the rest goes out with the last of the hedge
                    lot = float(leg.spec.lot_size) or 1.0
                    allowed = math.floor(min(leg.quantity, hedge_qty) * frac / lot + 1e-9) * lot
            qty = allowed - ls.sent
            if qty > 1e-9:
                self._send_entry(g, ls, qty, after_fill)

    def _send_entry(self, g: StructureGroup, ls: LegState, qty: float, after_fill: bool) -> None:
        cid = ls.client_id if not ls.orders else f"{ls.client_id}.{len(ls.orders)}"
        leg = ls.leg
        common = dict(
            side=leg.side, quantity=qty, instrument=leg.instrument, tif=g.tif,
            tags={"structure": g.id},
        )
        order = (
            Limit(price=float(g.prices[leg.instrument]), **common)
            if g.order_type == "limit"
            else Market(**common)
        )
        self.strategy.submit_order(order, client_id=cid)
        ls.orders.append(cid)
        ls.requested[cid] = qty
        ls.sent += qty
        self._owner[cid] = (g, ls)

    # -- manage ---------------------------------------------------------------------------

    def on_event(self, event, now_ns: int) -> None:
        """An order event: book a fill against its leg, then run the group's lifecycle in the same
        step (release the sold legs the hedges now cover, or start the close)."""
        cid = getattr(event, "client_order_id", None)
        owner = self._owner.get(cid) if cid else None
        if owner is None:
            return
        g, ls = owner
        if event.kind == "order_filled" and event.size:
            ls.got[cid] = ls.got.get(cid, 0.0) + float(event.size)
            if cid not in ls.close_ids:
                ls.filled += float(event.size)
                ls.cost += float(event.size) * float(event.price)
                if g.phase == "closed":
                    # a fill the close did not wait for (a cancel that lost the race): close it too
                    g.phase = "closing"
        elif event.kind == "order_resized" and event.size is not None and cid in ls.requested:
            ls.requested[cid] = float(event.size)
        self._advance(g, now_ns, None, False, True)

    def close(self, group: StructureGroup, reason: str = "manual") -> None:
        """Start closing the group: cancel what works, close what filled."""
        if group.phase == "entry":
            group.phase = "closing"
            group.reason = reason
            self._pump(group, False, None)

    def update(
        self,
        now_ns: int,
        marks: Mapping[str, float] | None = None,
        *,
        exits: bool = True,
    ) -> list[StructureGroup]:
        """Advance every group: leg-failure policy, exits, the sold legs the hedges cover and the
        pending cancels and closes. Returns the groups that began closing. ``exits=False`` runs
        only the lifecycle (leg failure, entry timeout and the pump), which needs no marks."""
        return [g for g in self.groups if self._advance(g, now_ns, marks, exits, False)]

    def _advance(self, g: StructureGroup, now_ns: int, marks, exits: bool, after_fill: bool) -> bool:
        began = False
        if g.phase == "entry":
            reason = self._reason(g, now_ns, marks, exits)
            if reason is not None:
                g.phase = "closing"
                g.reason = reason
                began = True
            else:
                self._release(g, after_fill)
        if g.phase == "closing":
            self._pump(g, after_fill, now_ns)
        return began

    def _reason(self, g: StructureGroup, now_ns: int, marks, exits: bool) -> str | None:
        cache = self.cache
        states = [_leg_state(cache, ls) for ls in g.legs]
        if g.on_leg_failure == "unwind" and any(s in _FAILED for s in states):
            if g.failure is None:
                for ls in g.legs:
                    for c in ls.orders:
                        if _order_state(cache, ls, c) in _FAILED:
                            rec = cache.order(c)
                            g.failure = (c, rec.status, rec.reject_reason)
                            break
                    if g.failure is not None:
                        break
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
            prem = math.fsum(
                leg.sign * leg.ratio * _mark(marks, leg) for leg in g.structure.option_legs
            )
            if ex.premium_at_or_above is not None and prem >= ex.premium_at_or_above:
                return "premium_at_or_above"
            if ex.premium_at_or_below is not None and prem <= ex.premium_at_or_below:
                return "premium_at_or_below"
        return None

    def _pump(self, g: StructureGroup, after_fill: bool, now_ns: int | None) -> None:
        """One step of the close. Every working entry order is cancelled first (a partly filled
        order's remainder included), then what filled is closed: the sold legs first, the bought
        legs (their cover) only once no sold leg is left open or could still fill. A fill that
        arrives after its leg was closed is closed in turn."""
        cache = self.cache
        for ls in g.legs:
            for c in ls.orders:
                if c not in ls.cancelled and _order_state(cache, ls, c) in ("submitted", "working"):
                    self.strategy.cancel_order(c)
                    ls.cancelled.add(c)
        done = True
        sold_open = False
        for ls in sorted(g.legs, key=lambda x: (x.leg.side != "sell", x.index)):
            if ls.leg.side == "buy" and sold_open:
                continue  # the cover stays on; the group is not done either
            pending = self._close_leg(g, ls, after_fill, now_ns)
            if pending:
                done = False
                sold_open = sold_open or ls.leg.side == "sell"
        if done:
            g.phase = "closed"

    def _live(self, ls: LegState) -> bool:
        return any(_order_state(self.cache, ls, c) in ("submitted", "working") for c in ls.orders)

    def _close_leg(self, g: StructureGroup, ls: LegState, after_fill: bool, now_ns: int | None) -> bool:
        """Send what the leg still needs closed; whether the leg is yet to be closed."""
        cache = self.cache
        for c in ls.close_ids:
            if c not in ls.close_failed and _order_state(cache, ls, c) in _FAILED:
                ls.close_failed.add(c)
                ls.retry_after_ns = now_ns
        need = ls.close_need
        if need > _EPS and len(ls.close_failed) < _CLOSE_ATTEMPTS and not (
            now_ns is not None and ls.retry_after_ns is not None and now_ns <= ls.retry_after_ns
        ):
            self._send_close(g, ls, need, after_fill)
        working = any(_order_state(cache, ls, c) in ("submitted", "working") for c in ls.close_ids)
        return self._live(ls) or working or ls.close_need > _EPS

    def _send_close(self, g: StructureGroup, ls: LegState, quantity: float, after_fill: bool) -> None:
        cid = f"{g.id}:x{ls.index}.{len(ls.close_ids)}"
        leg = ls.leg
        close = Market(
            side="sell" if leg.side == "buy" else "buy",
            quantity=quantity,
            instrument=leg.instrument,
            reduce_only=True,
            tags={"structure": g.id, "close": g.reason or "manual"},
        )
        self.strategy.submit_order(close, client_id=cid)
        ls.close_ids.append(cid)
        ls.requested[cid] = quantity
        self._owner[cid] = (g, ls)
