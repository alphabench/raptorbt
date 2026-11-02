"""A conformance kit for execution adapters.

It drives an adapter through a :class:`~raptorbt.LiveNode` against a scripted
venue that duplicates, delays, reorders and drops what it says, and restarts
the adapter in the middle, then asserts the node ends where it ends when the
venue behaves: the same orders, fills, average prices and positions, and no
order sent twice.

Any :class:`~raptorbt.execution.ExecutionAdapter` can run it, including one
written outside raptorbt. You supply a *binding*: a small object that connects
your adapter to the scripted venue. The venue says things as neutral
:class:`VenueEvent` snapshots; the binding renders each one the way your venue
would (a broker payload, a FIX message) and feeds it to the adapter.

.. code-block:: python

    class MyBinding:
        adapter: ExecutionAdapter
        def __init__(self, venue, options): ...   # build the adapter over a fake transport to ``venue``
        def ref_of(self, client_order_id): ...    # the reference your adapter sends the venue for this order
        def on_event(self, event): ...            # render ``event`` and give it to the adapter
        def pump(self): ...                       # run any queued I/O (no-op for a synchronous adapter)
        def resync(self): ...                     # read ``venue.book()`` and give it to the adapter
        def restart(self): ...                    # replace ``adapter`` with a fresh one, keeping its saved state
        def drain(self): ...                      # [(client_order_id, ExecutionReport, arrival_ns)] produced since last call

    failures = run_conformance(lambda venue, options: MyBinding(venue, options))
    assert not failures, failures

The venue's side of the wire is ``ScriptedVenue.place/cancel/modify/book``,
which your fake transport calls; every call can be made to fail with
``fail_next``. See :class:`ReferenceBinding` for a complete example.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import ExecutionAdapter, ExecutionReport, OrderRequest
from raptorbt.strategy import orders

__all__ = [
    "Binding",
    "Failure",
    "PROFILES",
    "ReferenceBinding",
    "SCENARIOS",
    "Scenario",
    "ScriptedVenue",
    "VenueEvent",
    "assert_conforms",
    "run_conformance",
    "run_scenario",
]

MIN = 60_000_000_000


class VenueRefused(Exception):
    """Raised by ``ScriptedVenue`` calls the scenario made the venue refuse."""

    def __init__(self, op: str, reason: str) -> None:
        super().__init__(reason)
        self.op, self.reason = op, reason


class VenueTimeout(Exception):
    """The venue's answer never came. The request may or may not have been processed."""


class VenueUnavailable(Exception):
    """The venue was not reached; the request was certainly not processed."""


@dataclass(frozen=True)
class VenueEvent:
    """The venue's whole view of one order at one moment (cumulative, not a delta)."""

    seq: int
    ref: str
    venue_order_id: str
    status: str  # received, open, filled, cancelled, rejected, expired
    symbol: str
    side: str
    quantity: float
    limit_price: float | None
    trigger_price: float | None
    filled_qty: float
    avg_price: float | None
    reason: str | None = None


@dataclass
class _VOrder:
    ref: str
    vid: str
    symbol: str
    side: str
    qty: float
    limit: float | None
    trigger: float | None
    status: str = "received"
    cum: float = 0.0
    notional: float = 0.0
    reason: str | None = None
    pending: tuple[float | None, float | None, float | None] | None = None


class ScriptedVenue:
    """An exchange that does only what the scenario tells it to.

    Calls from the adapter's transport (``place``, ``cancel``, ``modify``) are
    answered at once like a broker's HTTP response; what happens to the order
    afterwards (``accept``, ``fill``, ``ack_cancel``...) is scripted. Each
    scripted step emits a :class:`VenueEvent` to the subscriber.
    """

    def __init__(self) -> None:
        self.orders: dict[str, _VOrder] = {}
        self.place_calls: dict[str, int] = defaultdict(int)  # ref -> times place() was called
        self.duplicates: list[str] = []  # refs placed again after the venue already had them: a second live order
        self.cancel_calls: dict[str, int] = defaultdict(int)
        self.subscribers: list[Callable[[VenueEvent], None]] = []
        self._seq = 0
        self._fail: dict[str, list[str]] = defaultdict(list)

    # -- failure injection -----------------------------------------------------------
    def fail_next(self, op: str, how: str, reason: str = "refused") -> None:
        """``op`` in place/cancel/modify. ``how``: ``refuse`` (answer no), ``unavailable``
        (not reached), ``timeout_before`` (lost on the way), ``timeout_after`` (processed, answer lost)."""
        self._fail[op].append(f"{how}:{reason}")

    def _maybe_fail(self, op: str, *, processed: Callable[[], None]) -> None:
        if not self._fail[op]:
            processed()
            return
        how, _, reason = self._fail[op].pop(0).partition(":")
        if how == "refuse":
            raise VenueRefused(op, reason)
        if how == "unavailable":
            raise VenueUnavailable(op)
        if how == "timeout_before":
            raise VenueTimeout(op)
        processed()
        if how == "timeout_after":
            raise VenueTimeout(op)

    # -- what the adapter's transport calls ------------------------------------------------
    def place(self, ref: str, symbol: str, side: str, qty: float, limit: float | None, trigger: float | None) -> str:
        self.place_calls[ref] += 1
        out: list[str] = []

        def do() -> None:
            if ref in self.orders:
                self.duplicates.append(ref)  # a real venue would now hold two orders
                out.append(self.orders[ref].vid)
                return
            vid = f"V{len(self.orders) + 1}"
            self.orders[ref] = _VOrder(ref, vid, symbol, side, qty, limit, trigger)
            out.append(vid)

        self._maybe_fail("place", processed=do)
        return out[0] if out else self.orders[ref].vid

    def cancel(self, vid: str) -> None:
        o = self._by_vid(vid)
        self.cancel_calls[o.ref] += 1
        self._maybe_fail("cancel", processed=lambda: None)

    def modify(self, vid: str, qty: float | None, limit: float | None, trigger: float | None) -> None:
        o = self._by_vid(vid)

        def do() -> None:
            o.pending = (qty, limit, trigger)

        self._maybe_fail("modify", processed=do)

    def book(self) -> list[VenueEvent]:
        return [self._snapshot(o) for o in self.orders.values()]

    def find(self, ref: str) -> VenueEvent | None:
        o = self.orders.get(ref)
        return self._snapshot(o) if o else None

    def _by_vid(self, vid: str) -> _VOrder:
        for o in self.orders.values():
            if o.vid == vid:
                return o
        raise VenueRefused("lookup", "order not found")

    # -- scenario steps ------------------------------------------------------------------------
    def _emit(self, o: _VOrder) -> None:
        ev = self._snapshot(o)
        for s in self.subscribers:
            s(ev)

    def _snapshot(self, o: _VOrder) -> VenueEvent:
        self._seq += 1
        return VenueEvent(
            self._seq, o.ref, o.vid, o.status, o.symbol, o.side, o.qty, o.limit, o.trigger, o.cum,
            (o.notional / o.cum) if o.cum else None, o.reason,
        )  # fmt: skip

    def accept(self, ref: str) -> None:
        o = self.orders[ref]
        o.status = "open"
        self._emit(o)

    def fill(self, ref: str, qty: float, price: float) -> None:
        o = self.orders[ref]
        o.cum += qty
        o.notional += qty * price
        o.status = "filled" if o.cum >= o.qty - 1e-9 else "open"
        self._emit(o)

    def reject(self, ref: str, reason: str) -> None:
        o = self.orders[ref]
        o.status, o.reason = "rejected", reason
        self._emit(o)

    def ack_cancel(self, ref: str) -> None:
        o = self.orders[ref]
        o.status = "cancelled"
        self._emit(o)

    def expire(self, ref: str) -> None:
        o = self.orders[ref]
        o.status = "expired"
        self._emit(o)

    def apply_modify(self, ref: str) -> None:
        o = self.orders[ref]
        if o.pending:
            q, lim, trg = o.pending
            o.qty = q if q is not None else o.qty
            o.limit = lim if lim is not None else o.limit
            o.trigger = trg if trg is not None else o.trigger
            o.pending = None
        self._emit(o)


# -- the binding contract -------------------------------------------------------------------------


class Binding(Protocol):
    adapter: ExecutionAdapter

    def ref_of(self, client_order_id: str) -> str: ...
    def on_event(self, event: VenueEvent) -> None: ...
    def pump(self) -> None: ...
    def resync(self) -> None: ...
    def restart(self) -> None: ...
    def drain(self) -> list[tuple[str, ExecutionReport, int]]: ...


BindingFactory = Callable[[ScriptedVenue, dict], Binding]


# -- scenarios -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A script. Steps are tuples; orders are named ``o0``, ``o1``... in the order the strategy sends them.

    ``("bar",)`` a market bar (the strategy then does its queued actions);
    ``("submit", side, qty, price)``, ``("cancel", name)``, ``("modify", name, qty, price)``;
    ``("venue", "fill", name, qty, price)`` and the other ScriptedVenue steps;
    ``("fail", op, how[, reason])`` arms ``fail_next``.
    """

    name: str
    steps: tuple[tuple, ...]
    options: dict = field(default_factory=dict)
    needs_resync: bool = False


B = ("bar",)
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("fill", (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("venue", "fill", "o0", 10, 99.0), B)),
    Scenario(
        "partial_fills",
        (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("venue", "fill", "o0", 4, 99.0), B,
         ("venue", "fill", "o0", 6, 98.5), B),
    ),  # fmt: skip
    Scenario("rejected", (("submit", "buy", 10, 99.0), B, ("venue", "reject", "o0", "RMS: margin shortfall"), B)),
    Scenario(
        "refused_at_place",
        (("fail", "place", "refuse", "bad price"), ("submit", "buy", 10, 99.0), B, B),
    ),
    Scenario(
        "cancel",
        (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("cancel", "o0"), B, ("venue", "ack_cancel", "o0"), B),
    ),
    Scenario(
        "cancel_loses_race_to_a_fill",
        (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("cancel", "o0"), B,
         ("venue", "fill", "o0", 3, 99.0), ("venue", "ack_cancel", "o0"), B),
    ),  # fmt: skip
    Scenario(
        "cancel_refused_because_filled",
        (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("fail", "cancel", "refuse", "already completed"),
         ("cancel", "o0"), B, ("venue", "fill", "o0", 10, 99.0), B),
    ),  # fmt: skip
    Scenario(
        "replace",
        (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("modify", "o0", 8, 98.0), B,
         ("venue", "apply_modify", "o0"), B, ("venue", "fill", "o0", 8, 98.0), B),
    ),  # fmt: skip
    Scenario(
        "replace_refused",
        (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("fail", "modify", "refuse", "too late"),
         ("modify", "o0", 8, 98.0), B, ("venue", "fill", "o0", 10, 99.0), B),
    ),  # fmt: skip
    Scenario("expired", (("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"), ("venue", "expire", "o0"), B)),
    Scenario(
        "two_orders_interleaved",
        (("submit", "buy", 10, 99.0), ("submit", "buy", 5, 105.0, "B"), B, ("venue", "accept", "o1"), ("venue", "accept", "o0"),
         ("venue", "fill", "o1", 2, 105.0), ("venue", "fill", "o0", 10, 99.0), ("venue", "fill", "o1", 3, 104.0), B),
    ),  # fmt: skip
    Scenario(
        "retry_after_rate_limit",
        (("fail", "place", "unavailable"), ("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"),
         ("venue", "fill", "o0", 10, 99.0), B),
    ),  # fmt: skip
    Scenario(
        "answer_lost_after_the_venue_took_the_order",
        (("fail", "place", "timeout_after"), ("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"),
         ("venue", "fill", "o0", 10, 99.0), B),
    ),  # fmt: skip
    Scenario(
        "request_lost_on_the_way",
        (("fail", "place", "timeout_before"), ("submit", "buy", 10, 99.0), B, ("venue", "accept", "o0"),
         ("venue", "fill", "o0", 10, 99.0), B),
        options={"resend_if_absent": True},
    ),  # fmt: skip
)

PROFILES = ("in_order", "duplicate", "delay", "reorder", "drop", "restart", "chaos")


def _bar(node, i: int) -> None:
    for sym in ("A", "B"):
        node.push_bar(sym, (i + 1) * MIN, 100.0, 101.0, 99.0, 100.5, 1000.0)


class _Scripted(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.todo: list[Callable[[Strategy], None]] = []

    def on_bar(self, ctx, bar):
        todo, self.todo = self.todo, []
        for fn in todo:
            fn(self)


class _Proxy:
    """The node holds this; ``restart`` swaps the adapter behind it."""

    def __init__(self, binding: Binding) -> None:
        self.binding = binding

    def submit(self, request: OrderRequest) -> None:
        self.binding.adapter.submit(request)

    def cancel(self, client_order_id: str) -> None:
        self.binding.adapter.cancel(client_order_id)

    def modify(self, client_order_id: str, *, quantity=None, limit_price=None, trigger_price=None) -> None:
        self.binding.adapter.modify(
            client_order_id, quantity=quantity, limit_price=limit_price, trigger_price=trigger_price
        )


class _Channel:
    """Between the venue and the binding: what the profile does to events."""

    def __init__(self, binding: Binding, profile: str, seed: int) -> None:
        self.binding, self.profile = binding, profile
        self.rng = random.Random(seed)
        self.held: list[VenueEvent] = []
        self.n = 0
        self.dropped = 0

    def send(self, ev: VenueEvent) -> None:
        self.n += 1
        p = self.profile
        if p in ("in_order", "restart"):
            self.binding.on_event(ev)
        elif p == "duplicate":
            self.binding.on_event(ev)
            self.binding.on_event(ev)
        elif p in ("delay", "reorder"):
            self.held.append(ev)
        elif p == "drop":
            if self.n % 2 == 0 and ev.status not in ("cancelled", "rejected", "expired"):
                self.dropped += 1  # keep terminal facts out of the drop: a lost last word is a resync's job, tested below
            else:
                self.binding.on_event(ev)
        elif p == "chaos":
            roll = self.rng.random()
            if roll < 0.2:
                self.dropped += 1
            else:
                self.held.append(ev)
                if roll > 0.7:
                    self.held.append(ev)

    def flush(self) -> None:
        batch, self.held = self.held, []
        if self.profile in ("reorder",):
            batch.reverse()
        elif self.profile == "chaos":
            self.rng.shuffle(batch)
        for ev in batch:
            self.binding.on_event(ev)


@dataclass
class Outcome:
    orders: list[tuple]
    positions: float
    errors: list[str]
    double_sent: list[str]


@dataclass(frozen=True)
class Failure:
    scenario: str
    profile: str
    what: str

    def __str__(self) -> str:
        return f"[{self.scenario} / {self.profile}] {self.what}"


def run_scenario(factory: BindingFactory, sc: Scenario, profile: str, *, seed: int = 7) -> Outcome:
    venue = ScriptedVenue()
    binding = factory(venue, dict(sc.options))
    chan = _Channel(binding, profile, seed)
    venue.subscribers.append(chan.send)
    strat = _Scripted()
    node = LiveNode(
        [strat], ["A", "B"], BacktestConfig(initial_capital=1_000_000.0, fees=0.0), bar_stamp="close", execution=_Proxy(binding)
    )
    errors: list[str] = []
    clock = [0]
    sent = [0]
    restart_at = max(1, len(sc.steps) // 2)

    def push_pending() -> None:
        binding.pump()
        for cid, report, _arrival in binding.drain():
            clock[0] += 1
            try:
                node.push_report(cid, report, arrival_ns=(bars[0] + 1) * MIN + clock[0])
            except Exception as exc:  # noqa: BLE001 - a rejected push is a finding, not a crash
                errors.append(f"{cid} {report.kind}: {exc}")

    bars = [0]
    for i, step in enumerate(sc.steps):
        op = step[0]
        if op == "bar":
            chan.flush()
            push_pending()
            acted = bool(strat.todo)
            _bar(node, bars[0])
            bars[0] += 1
            if acted:  # the engine sends what a strategy decided on the next event
                _bar(node, bars[0])
                bars[0] += 1
            push_pending()
        elif op == "submit":
            _, side, qty, price, *sym = step
            name = f"o{sent[0]}"
            sent[0] += 1
            strat.todo.append(
                lambda s, side=side, qty=qty, price=price, name=name, sym=(sym[0] if sym else "A"): s.submit_order(
                    orders.Limit(side=side, price=price, units=float(qty)), client_id=name, symbol=sym
                )
            )
        elif op == "cancel":
            strat.todo.append(lambda s, n=step[1]: s.cancel_order(n))
        elif op == "modify":
            _, name, qty, price = step
            strat.todo.append(lambda s, n=name, q=qty, p=price: s.modify_order(n, units=float(q), limit_price=p))
        elif op == "venue":
            chan.flush()
            push_pending()
            _, method, name, *args = step
            getattr(venue, method)(binding.ref_of(f"s0-{int(name[1:])}"), *args)
        elif op == "fail":
            venue.fail_next(step[1], step[2], *(step[3:4]))
        if profile == "restart" and i == restart_at:
            push_pending()
            binding.restart()
    chan.flush()
    push_pending()
    if profile in ("drop", "chaos", "restart") or sc.needs_resync:
        binding.resync()
        push_pending()
        chan.flush()
        push_pending()
    if profile in ("drop", "chaos"):
        # The last word of an order may itself have been lost: the venue's book, read again, restores it.
        binding.resync()
        push_pending()
    _bar(node, bars[0])
    push_pending()
    pos = sum(p.size for sym in ("A", "B") for p in node.positions(0, sym))
    result = node.finish()[0].result
    recs = sorted(
        (r.client_id, r.status, round(r.filled_qty, 9), round(r.avg_fill_price or 0.0, 4)) for r in result.orders()
    )
    return Outcome(recs, round(pos, 9), errors, list(venue.duplicates))


def run_conformance(
    factory: BindingFactory,
    *,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
    profiles: tuple[str, ...] = PROFILES,
) -> list[Failure]:
    """Run every scenario under every profile. Returns the failures (empty means conforming)."""
    out: list[Failure] = []
    for sc in scenarios:
        try:
            base = run_scenario(factory, sc, "in_order")
        except Exception as exc:  # noqa: BLE001
            out.append(Failure(sc.name, "in_order", f"baseline raised {type(exc).__name__}: {exc}"))
            continue
        if base.double_sent:
            out.append(Failure(sc.name, "in_order", f"sent twice: {base.double_sent}"))
        if base.errors:
            out.append(Failure(sc.name, "in_order", f"report refused by the node: {base.errors}"))
        for p in profiles:
            if p == "in_order":
                continue
            try:
                got = run_scenario(factory, sc, p)
            except Exception as exc:  # noqa: BLE001
                out.append(Failure(sc.name, p, f"raised {type(exc).__name__}: {exc}"))
                continue
            if got.double_sent:
                out.append(Failure(sc.name, p, f"sent twice: {got.double_sent}"))
            if got.errors:
                out.append(Failure(sc.name, p, f"report refused by the node: {got.errors}"))
            if got.orders != base.orders or got.positions != base.positions:
                out.append(Failure(sc.name, p, f"diverged: {got.orders} pos {got.positions} != {base.orders} pos {base.positions}"))
    return out


def assert_conforms(factory: BindingFactory, **kw: Any) -> None:
    failures = run_conformance(factory, **kw)
    if failures:
        raise AssertionError("adapter does not conform:\n" + "\n".join(str(f) for f in failures))


# -- a complete reference binding ---------------------------------------------------------------------


def _reference_adapter(venue: ScriptedVenue, options: dict):
    """A BrokerAdapter whose 'broker' is the ScriptedVenue itself. Shows what a mapping implements."""
    from .base import (
        Ambiguous,
        BrokerAdapter,
        Instrument,
        OrderStatusUpdate,
        Session,
        Transient,
        VenueRefusal,
    )

    class Reference(BrokerAdapter):
        name = "reference"
        order_types = frozenset({"limit", "market"})
        time_in_forces = frozenset({"gtc", "day"})

        def translate(self, request, instrument, product, tag):
            return {"ref": tag, "symbol": request.symbol, "side": request.side, "qty": request.quantity,
                    "limit": request.limit_price, "trigger": request.trigger_price}  # fmt: skip

        def _wire(self, fn):
            try:
                return fn()
            except VenueRefused as e:
                raise VenueRefusal(e.reason) from e
            except VenueUnavailable as e:
                raise Transient(str(e)) from e
            except VenueTimeout as e:
                raise Ambiguous(str(e)) from e

        def place(self, rec):
            w = rec.wire
            return self._wire(lambda: venue.place(w["ref"], w["symbol"], w["side"], w["qty"], w["limit"], w["trigger"]))

        def cancel_remote(self, rec):
            self._wire(lambda: venue.cancel(rec.venue_order_id))

        def modify_remote(self, rec, q, lim, trg):
            self._wire(lambda: venue.modify(rec.venue_order_id, q, lim, trg))

        def fetch_orders(self):
            return [self.row(e) for e in venue.book()]

        def find_by_tag(self, tag):
            e = venue.find(tag)
            return self.row(e) if e else None

        @staticmethod
        def row(e: VenueEvent) -> OrderStatusUpdate:
            return OrderStatusUpdate(
                e.venue_order_id, e.status, tag=e.ref, symbol=e.symbol, side=e.side, quantity=e.quantity,
                filled_qty=e.filled_qty, avg_price=e.avg_price, limit_price=e.limit_price,
                trigger_price=e.trigger_price, reason=e.reason,
            )  # fmt: skip

        def parse_order_update(self, payload):
            return self.row(payload)

    clock = iter(range(1, 10**9))
    return Reference(
        Session("reference-token"),
        lambda symbol: Instrument("X", symbol),
        sleep=lambda s: None,
        clock_ns=lambda: next(clock),
        **options,
    )


class ReferenceBinding:
    """Binds the base adapter to the scripted venue. Copy it to see what a binding is."""

    def __init__(self, venue: ScriptedVenue, options: dict) -> None:
        self.venue, self.options = venue, options
        self.adapter = _reference_adapter(venue, options)

    def ref_of(self, client_order_id: str) -> str:
        return self.adapter.ids.encode(client_order_id)

    def on_event(self, event: VenueEvent) -> None:
        self.adapter.handle_payload(event)

    def pump(self) -> None:
        self.adapter.pump()

    def resync(self) -> None:
        self.adapter.resync()
        self.adapter.pump()

    def restart(self) -> None:
        state = self.adapter.export_state()
        self.adapter.close()
        self.adapter = _reference_adapter(self.venue, self.options)
        self.adapter.import_state(state)

    def drain(self):
        return [(q.client_order_id, q.report, q.arrival_ns) for q in self.adapter.drain_reports()]
