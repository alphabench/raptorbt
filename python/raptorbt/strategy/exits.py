"""Percent and absolute exits on one position or on a named group of positions.

A strategy states *when to get out* once, and the engine closes the positions
with real reduce-only market orders when the rule fires::

    rule = ExitRule(stop_loss_pct=30, take_profit_pct=50)          # of what the position cost
    self.set_exit(ctx, rule, symbol="NSE:XYZ")                      # one symbol's positions
    self.set_exit(ctx, rule, symbols=["CE", "PE"], name="strangle")  # a named group of legs
    self.set_exit(ctx, ExitRule(stop_loss=5_000.0), group=structure_group)  # a structure group

The rule is evaluated by the engine, locally, on every price the strategy sees
(mix :class:`ExitEvents` into the strategy, or call ``self.update_exits(ctx)``
from its own price hook). When the node's venue declares it can hold the rule
(an optional ``place_exit_rule(request)`` on the adapter), the rule is handed to
it instead and the engine stops watching; a venue that refuses it
(:class:`~raptorbt.execution.Unsupported` or
:class:`~raptorbt.execution.OrderRefused`) leaves the rule with the engine.

A rule is handed to the venue only for positions that exist: a rule set before its entry
has filled has no legs to give, so the engine watches it. The venue holds the legs it was
given, so the engine keeps comparing them with the positions: when a position's size changes
the rule is replaced at the venue (``replace_exit_rule``), or cancelled and watched by the
engine when the venue cannot replace it; when the positions are gone the held rule is
cancelled (``cancel_exit_rule``) unless the venue fired it. ``cancel_exit`` on a held rule
asks the venue too, and raises when the venue cannot take it back: a rule the venue still
holds is never reported cancelled. A venue that answers neither yes nor no leaves the rule
``unconfirmed``: its exit orders are registered so a fill is booked, the engine does not
watch it as well (that could close twice), and a reconciliation reports it.

What a rule measures
--------------------

For the positions a rule covers (every open position of the symbol, or of the
symbols of the group), at the last prices:

* ``pnl``: ``Σ direction × (mark − entry) × size × multiplier``, before costs;
* ``cost``: ``Σ entry × size × multiplier``, the value the positions were
  entered at (the premium paid or collected, for options);
* ``premium``: ``Σ direction × mark × size``, the net premium in price units
  (a position's own mark, for one position).

``stop_loss_pct`` / ``take_profit_pct`` are percents of ``cost`` (a loss of
``stop_loss_pct`` percent of cost fires the stop). ``stop_loss`` /
``take_profit`` are amounts of ``pnl``. ``premium_at_or_above`` /
``premium_at_or_below`` compare ``premium``. The first condition met fires, in
that order. Nothing fires until every symbol of the rule has a position and a
last price.

Firing
------

On firing the engine sends one reduce-only market order per open position of the
rule, with client id ``{rule id}:x{n}`` and tags ``{"exit_rule": id, "close":
reason}``, so the exit is an order like any other: it passes ``order_check``, is
journaled, shows in ``on_order_event`` and is sent to the venue. A rule fires
once, unless every close it sent was refused, cancelled or expired while positions remain: it then
arms again (``refused`` lists what happened), and tries again with new ids
``{rule id}:r{attempt}:x{n}``, so a rule is never left silently without its stop. Retries back
off (one second after the first refusal, doubling, at most half a minute apart) and are
bounded: after ``MAX_CLOSE_ATTEMPTS`` refused attempts the rule is ``failed`` and the
strategy's ``on_exit_failed(ctx, handle)`` is called; without one the book raises
``exit_close_failed``, because a position that cannot be closed is not a quiet condition. A
position's multiplier is the instrument spec's; an instrument with no spec counts with
multiplier 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from raptorbt.errors import InvalidOrder, InvalidParameter, StateError, UnknownSymbol
from raptorbt.execution import AdapterError, OrderRefused, Unsupported
from raptorbt.strategy.orders import Market

#: Refused attempts to close after which a rule is ``failed`` (and the strategy hears of it).
MAX_CLOSE_ATTEMPTS = 6
#: The wait before the next attempt after the first refusal (ns); it doubles, up to the cap.
RETRY_BACKOFF_NS = 1_000_000_000
RETRY_BACKOFF_CAP_NS = 30_000_000_000
#: How many close ids and refusals a handle remembers.
_KEPT = 64


class ExitRuleUnconfirmed(AdapterError):
    """The venue did not say whether it took, changed or dropped an exit rule (an error after
    the request left, or a request whose result was never written)."""


def _invalid(message: str, parameter: str, value) -> InvalidParameter:
    return InvalidParameter(
        message, code="invalid_exit_rule", operation="set_exit", parameter=parameter, value=value
    )


@dataclass(frozen=True)
class ExitRule:
    """When to close a position or a group. Every field is optional; the first that holds fires."""

    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    premium_at_or_above: float | None = None
    premium_at_or_below: float | None = None

    def __post_init__(self) -> None:
        given = [n for n in self.__dataclass_fields__ if getattr(self, n) is not None]
        if not given:
            raise _invalid("an exit rule needs at least one condition", "rule", None)
        for name in ("stop_loss_pct", "take_profit_pct", "stop_loss", "take_profit"):
            v = getattr(self, name)
            if v is not None and not (math.isfinite(v) and v > 0):
                raise _invalid(f"{name} must be a positive number", name, v)
        if self.stop_loss_pct is not None and self.stop_loss_pct > 100:
            raise _invalid("stop_loss_pct is at most 100 (percent of cost)", "stop_loss_pct", self.stop_loss_pct)
        for name in ("premium_at_or_above", "premium_at_or_below"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise _invalid(f"{name} must be finite", name, v)

    def to_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in self.__dataclass_fields__ if getattr(self, n) is not None}

    def reason(self, pnl: float, cost: float, premium: float) -> str | None:
        """The first condition that holds, or ``None``."""
        if self.stop_loss_pct is not None and cost > 0 and pnl <= -cost * self.stop_loss_pct / 100.0:
            return "stop_loss_pct"
        if self.take_profit_pct is not None and cost > 0 and pnl >= cost * self.take_profit_pct / 100.0:
            return "take_profit_pct"
        if self.stop_loss is not None and pnl <= -self.stop_loss:
            return "stop_loss"
        if self.take_profit is not None and pnl >= self.take_profit:
            return "take_profit"
        if self.premium_at_or_above is not None and premium >= self.premium_at_or_above:
            return "premium_at_or_above"
        if self.premium_at_or_below is not None and premium <= self.premium_at_or_below:
            return "premium_at_or_below"
        return None


@dataclass(frozen=True)
class ExitRuleRequest:
    """What a venue is asked to hold, when it declares ``place_exit_rule``.

    ``rule_id`` is the engine's id; ``kind`` is ``"position"`` or ``"group"``; ``legs`` lists
    ``(symbol, close_side, quantity)`` of the positions covered when it was placed (never
    empty: a rule with no position is watched by the engine); ``rule`` is
    :meth:`ExitRule.to_dict`; ``exit_client_ids`` are the ids the venue gives the orders it
    makes when the rule fires (``{rule_id}:x{n}``, one per leg; ``{rule_id}:g{generation}:x{n}``
    for a rule replaced ``generation`` times)."""

    rule_id: str
    kind: str
    legs: tuple[tuple[str, str, float], ...]
    rule: Mapping[str, float]
    exit_client_ids: tuple[str, ...]
    strategy_index: int | None = None
    generation: int = 0


@dataclass
class ExitHandle:
    """The handle ``set_exit`` returns. Read it; do not mutate it."""

    id: str
    name: str
    rule: ExitRule
    symbols: tuple[str, ...]
    #: ``armed`` (the engine watches), ``venue`` (the venue holds it), ``fired``,
    #: ``done`` (fired and flat), ``canceled`` or ``failed`` (its closes were refused
    #: ``MAX_CLOSE_ATTEMPTS`` times).
    state: str = "armed"
    #: Why it fired, once it has.
    reason: str | None = None
    close_ids: list[str] = field(default_factory=list)
    #: How many times the rule has fired (a close that was refused fires it again).
    attempts: int = 0
    #: ``(client_id, status, reason)`` of each close that ended without filling.
    refused: list[tuple[str, str, str | None]] = field(default_factory=list)
    #: A venue holds the rule but did not confirm it (see the module docs).
    unconfirmed: bool = False
    _sent: list[str] = field(default_factory=list, repr=False)
    _request: ExitRuleRequest | None = field(default=None, repr=False)
    _not_before: int = field(default=0, repr=False)
    _skip: int = field(default=0, repr=False)


class ExitBook:
    """The exit rules one strategy has set in one run."""

    def __init__(self, strategy) -> None:
        self.strategy = strategy
        self.cache = strategy.cache
        self.handles: list[ExitHandle] = []
        self._seq = 0
        self._venue_obj = None  # the node's exit-rule venue, remembered from the last context seen

    # -- set ----------------------------------------------------------------------------

    def set(self, ctx, rule: ExitRule, *, symbol=None, symbols=None, group=None, name=None) -> ExitHandle:
        if not isinstance(rule, ExitRule):
            raise _invalid(f"rule must be an ExitRule, got {type(rule).__name__}", "rule", type(rule).__name__)
        if sum(x is not None for x in (symbol, symbols, group)) != 1:
            raise _invalid("give exactly one of symbol, symbols and group", "symbol", None)
        if group is not None:
            syms = tuple(dict.fromkeys(ls.leg.instrument for ls in group.legs))
            name = name or group.id
        elif symbols is not None:
            syms = tuple(dict.fromkeys(symbols))
            if len(syms) < 1 or isinstance(symbols, str):
                raise _invalid("symbols is a non-empty list of symbols", "symbols", symbols)
            if name is None:
                raise _invalid("a named group needs name=", "name", None)
        else:
            syms = (symbol,)
            name = name or symbol
        missing = [s for s in syms if s not in ctx.symbols]
        if missing:
            raise _invalid(f"the strategy does not trade {missing}", "symbols", missing)
        rid = f"{self._tag()}-ex{self._seq}"
        self._seq += 1
        handle = ExitHandle(rid, name, rule, syms)
        kind = "position" if symbol is not None else "group"
        self.handles.append(handle)
        try:
            self._offer(ctx, handle, kind)
        except ExitRuleUnconfirmed as e:
            # The venue may hold the rule: the handle is kept (so it can be cancelled), the engine
            # does not watch it as well, and the strategy hears of it now.
            handle.state, handle.unconfirmed = "venue", True
            raise StateError(
                f"exit rule {handle.id}: the venue did not say whether it holds the rule ({e}); its "
                "exit orders are registered and a reconciliation reports it. Cancel it with "
                "cancel_exit(handle) or accept it after reconcile()",
                code="exit_rule_unconfirmed",
                operation="set_exit",
            ) from e
        except BaseException:
            self.handles.remove(handle)
            raise
        return handle

    def _venue(self, ctx):
        host = getattr(ctx, "_raptorbt_host", None)
        venue = getattr(host, "venue", None)
        self._venue_obj = venue
        return host, venue

    def _legs(self, ctx, symbols) -> tuple[tuple[str, str, float], ...]:
        legs = []
        for sym in symbols:
            for snap in ctx.positions(sym):
                legs.append((sym, "sell" if snap.direction > 0 else "buy", float(snap.size)))
        return tuple(legs)

    def _request_for(self, ctx, handle: ExitHandle, kind: str, legs, generation: int) -> ExitRuleRequest:
        host, _ = self._venue(ctx)
        index_of = getattr(host, "index_of", None)
        prefix = handle.id if generation == 0 else f"{handle.id}:g{generation}"
        return ExitRuleRequest(
            handle.id, kind, legs, handle.rule.to_dict(),
            tuple(f"{prefix}:x{i}" for i in range(len(legs))),
            strategy_index=index_of(self.strategy) if callable(index_of) else None,
            generation=generation,
        )  # fmt: skip

    def _offer(self, ctx, handle: ExitHandle, kind: str) -> None:
        """Hand the rule to the venue when it can hold it and there is a position to hold it for."""
        _, venue = self._venue(ctx)
        place = getattr(venue, "place_exit_rule", None)
        if place is None:
            return
        legs = self._legs(ctx, handle.symbols)
        if not legs:
            return  # nothing to protect yet: the engine watches until a position exists
        request = self._request_for(ctx, handle, kind, legs, 0)
        try:
            outcome = place(request)
        except (Unsupported, OrderRefused):
            return  # the engine keeps watching
        handle.state, handle._request = "venue", request
        handle.unconfirmed = outcome == "unconfirmed"

    def _tag(self) -> str:
        return getattr(self.strategy.config, "order_id_tag", None) or "X"

    def cancel(self, handle: ExitHandle) -> None:
        if handle.state == "venue":
            self._cancel_at_venue(handle, "canceled by the strategy")
            handle.state = "canceled"
        elif handle.state == "armed":
            handle.state = "canceled"

    # -- a rule the venue holds -----------------------------------------------------------

    def _cancel_at_venue(self, handle: ExitHandle, why: str) -> None:
        """Take a held rule back. A venue that cannot is an error, never a rule reported gone
        while the venue still holds it."""
        cancel = getattr(self._venue_obj, "cancel_exit_rule", None)
        request = handle._request
        if request is None:
            return
        if cancel is None:
            raise StateError(
                f"exit rule {handle.id} is held by the venue and this node cannot take it back ({why})",
                code="exit_rule_not_cancellable", operation="cancel_exit",
            )
        try:
            cancel(request)
        except (Unsupported, OrderRefused) as e:
            raise StateError(
                f"exit rule {handle.id} is held by the venue, which would not take it back ({why}): {e}. "
                "It stays held; do not assume its exit cannot fire",
                code="exit_rule_not_cancellable", operation="cancel_exit",
            ) from e
        except ExitRuleUnconfirmed as e:
            handle.unconfirmed = True
            raise StateError(
                f"exit rule {handle.id}: the venue did not say whether it dropped the rule ({e})",
                code="exit_rule_unconfirmed", operation="cancel_exit",
            ) from e
        handle._request, handle.unconfirmed = None, False

    def _sync_venue(self, ctx, h: ExitHandle) -> None:
        """Keep a held rule matching the positions it protects."""
        self._venue(ctx)
        request = h._request
        if request is None:
            return
        # Once the venue acts on the rule (an exit order it fired is accepted or filling) it is
        # the venue's: the engine neither replaces nor cancels it under way.
        records = [self.cache.order(cid) for cid in request.exit_client_ids]
        acting = any(r is not None and r.status != "submitted" for r in records)
        legs = self._legs(ctx, h.symbols)
        if not legs:
            if acting:
                h.state, h._request = "done", None  # the venue closed what it held
            else:
                self._cancel_at_venue(h, "the positions it protected are gone")
                h.state = "done"
            return
        if acting or legs == request.legs:
            return
        self._replace_at_venue(ctx, h, legs)

    def _replace_at_venue(self, ctx, h: ExitHandle, legs) -> None:
        _, venue = self._venue(ctx)
        old = h._request
        new = self._request_for(ctx, h, old.kind, legs, old.generation + 1)
        replace = getattr(venue, "replace_exit_rule", None)
        try:
            if replace is None:
                raise Unsupported("this venue does not change a held rule")
            replace(old, new)
        except (Unsupported, OrderRefused):
            # It cannot be changed in place: take it back and watch the rule here. If it cannot
            # be taken back either, the stale legs are the venue's and the strategy hears of it.
            self._cancel_at_venue(h, "its positions changed and the venue cannot replace it")
            h.state = "armed"
            return
        except ExitRuleUnconfirmed as e:
            h.unconfirmed = True
            raise StateError(
                f"exit rule {h.id}: the venue did not say whether it changed the rule ({e})",
                code="exit_rule_unconfirmed", operation="update_exits",
            ) from e
        h._request = new

    # -- watch --------------------------------------------------------------------------

    def update(self, ctx, marks: Mapping[str, float] | None = None) -> list[ExitHandle]:
        """Evaluate every armed rule at ``marks`` (default: the last price of each symbol) and
        send the closes of those that fire. Returns the handles that fired now."""
        fired: list[ExitHandle] = []
        now = getattr(ctx, "event_ns", None)
        for h in self.handles:
            if h.state == "venue":
                self._sync_venue(ctx, h)
                continue
            if h.state == "fired":
                if all(not ctx.positions(s) for s in h.symbols):
                    h.state = "done"
                elif self._closes_failed(h):
                    if h.attempts >= MAX_CLOSE_ATTEMPTS:
                        self._give_up(ctx, h)
                    else:
                        h.state, h.reason = "armed", None  # never leave the position without its stop
                        self._back_off(h, now)
                continue
            if h.state != "armed":
                continue
            if self._waiting(h, now):
                continue
            measured = self._measure(ctx, h, marks)
            if measured is None:
                continue
            reason = h.rule.reason(*measured)
            if reason is None:
                continue
            h.state, h.reason = "fired", reason
            self._close(ctx, h)
            fired.append(h)
        return fired

    @staticmethod
    def _back_off(h: ExitHandle, now) -> None:
        wait = min(RETRY_BACKOFF_NS * 2 ** (h.attempts - 1), RETRY_BACKOFF_CAP_NS)
        if now is not None:
            h._not_before = now + wait
        else:  # no clock on this event: wait by events instead
            h._skip = 2 ** (h.attempts - 1)

    @staticmethod
    def _waiting(h: ExitHandle, now) -> bool:
        if now is not None and now < h._not_before:
            return True
        if h._skip > 0:
            h._skip -= 1
            return True
        return False

    def _give_up(self, ctx, h: ExitHandle) -> None:
        h.state = "failed"
        hook = getattr(self.strategy, "on_exit_failed", None)
        if callable(hook):
            hook(ctx, h)
            return
        raise StateError(
            f"exit rule {h.id}: {h.attempts} attempts to close were refused or cancelled "
            f"({h.refused[-1][0]} {h.refused[-1][1]}: {h.refused[-1][2]}) and the position is still open",
            code="exit_close_failed",
            operation="update_exits",
        )

    def _measure(self, ctx, h: ExitHandle, marks):
        pnl = cost = premium = 0.0
        for sym in h.symbols:
            positions = ctx.positions(sym)
            if not positions:
                return None
            mark = self._mark(ctx, sym, marks)
            if mark is None:
                return None
            spec = self._spec(ctx, sym)
            mult = float(spec.multiplier) if spec is not None else 1.0
            for p in positions:
                pnl += p.direction * (mark - p.entry_price) * p.size * mult
                cost += abs(p.entry_price * p.size * mult)
                premium += p.direction * mark * p.size
        return pnl, cost, premium

    def _closes_failed(self, h: ExitHandle) -> bool:
        """Whether every close of the latest firing ended without filling (refused, cancelled
        or expired), so the rule must arm again. Records what happened in ``h.refused``."""
        records = [self.cache.order(cid) for cid in h._sent]
        if not h._sent or any(r is None or r.status not in ("rejected", "canceled", "expired") for r in records):
            return False
        for r in records:
            h.refused.append((r.client_id, r.status, r.reject_reason))
        del h.refused[:-_KEPT]
        h._sent = []
        return True

    @staticmethod
    def _spec(ctx, sym):
        try:
            return ctx.instrument(sym)
        except UnknownSymbol as e:
            if e.code != "no_instrument_spec":
                raise
            return None  # no spec: a plain instrument with multiplier 1

    @staticmethod
    def _mark(ctx, sym, marks):
        if marks is not None:
            return marks.get(sym)
        try:
            series = ctx.series(sym)
        except (KeyError, RuntimeError):
            return None
        for column in ("close", "ltp"):
            values = series.get(column)
            if values is not None and len(values):
                return float(values[-1])
        return None

    def _close(self, ctx, h: ExitHandle) -> None:
        n = 0
        h.attempts += 1
        h._sent = []
        prefix = h.id if h.attempts == 1 else f"{h.id}:r{h.attempts - 1}"
        for sym in h.symbols:
            for p in ctx.positions(sym):
                cid = f"{prefix}:x{n}"
                n += 1
                self.strategy.submit_order(
                    Market(
                        side="sell" if p.direction > 0 else "buy",
                        quantity=p.size,
                        instrument=sym,
                        reduce_only=True,
                        tags={"exit_rule": h.id, "close": h.reason or "exit"},
                    ),
                    client_id=cid,
                )
                h.close_ids.append(cid)
                h._sent.append(cid)
        del h.close_ids[:-_KEPT]
        if n == 0:
            raise InvalidOrder(
                f"exit rule {h.id} fired with no open position",
                code="invalid_exit_rule",
                operation="update_exits",
            )


class ExitEvents:
    """Mix into a strategy (before ``Strategy``) to have the engine evaluate its exit rules on
    every price it sees, after the strategy's own hook has run::

        class Wheel(ExitEvents, Strategy): ...

    It follows ``on_bar`` and ``on_trade`` (the strategy's own, if it defines them, first)."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for hook in ("on_bar", "on_trade"):
            own = cls.__dict__.get(hook)
            if own is not None and not getattr(own, "_raptorbt_exits", False):
                setattr(cls, hook, _then_update(own))

    def on_bar(self, ctx, bar):
        self.update_exits(ctx)

    on_bar._raptorbt_exits = True  # type: ignore[attr-defined]

    def on_trade(self, ctx, tick):
        self.update_exits(ctx)

    on_trade._raptorbt_exits = True  # type: ignore[attr-defined]


def _then_update(hook):
    import functools

    @functools.wraps(hook)
    def wrapped(self, ctx, *args):
        hook(self, ctx, *args)
        self.update_exits(ctx)

    wrapped._raptorbt_exits = True  # type: ignore[attr-defined]
    return wrapped
