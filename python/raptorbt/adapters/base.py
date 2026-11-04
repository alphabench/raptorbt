"""The common layer under every broker mapping.

A :class:`BrokerAdapter` is an :class:`~raptorbt.execution.ExecutionAdapter`
that does no I/O on the node's thread. Subclasses (one per broker) translate
and perform the calls; this class owns everything that is the same for every
broker: the outbox, idempotent ids, retries that never send an order twice,
report normalisation and the report queue.

Threading model
---------------

* The **node thread** calls ``submit``/``cancel``/``modify``. Each validates and
  translates the request (pure, local; ``Unsupported``/``OrderRefused`` are
  raised here), appends a job to the outbox and calls ``executor.kick()``.
* The **executor** drains the outbox in FIFO order (``pump``): rate limit, call
  the broker, record the answer. ``InlineExecutor`` does it before ``kick``
  returns (tests), ``ThreadExecutor`` on a worker thread, ``ManualExecutor``
  whenever the host calls ``pump``.
* The **feed thread** (a websocket, a postback handler) calls
  ``handle_payload``/``handle_update``/``handle_trade``.
* Everything the venue says becomes :class:`ExecutionReport` objects in the
  **report queue**. The host drains it on the node's thread, after the call
  that caused it has returned: ``adapter.deliver_to(node)`` or ``drain_reports``
  with ``node.push_report``. The adapter never touches the node.

Request outcomes
----------------

A request the broker certainly refuses becomes a ``rejected`` report (or
``cancel_rejected``/``replace_rejected``). A request that may or may not have
arrived (a timeout on a write) is never blindly repeated: the adapter looks the
order up by its tag, adopts it if found, and otherwise leaves the order
``Submitted`` with an ``unconfirmed`` event for the host to reconcile
(``resend_if_absent=True`` resends once when the lookup finds nothing). The
tag is the client order id, so a repeat can never create a second order the
adapter cannot see.

Fills
-----

Brokers report orders with cumulative filled quantity and average price. The
adapter turns each rise into one fill of the difference, at the price that
makes the running average right, with the deterministic execution id
``{venue_order_id}:{cumulative_qty}``. Stale, duplicate and out-of-order
updates add nothing. A mapping whose feed carries real trades
(``fill_source="trades"``) uses their ids through ``handle_trade`` instead.

Recovery hooks (used by the journal work)
-----------------------------------------

``orders()``, ``open_orders()``, ``positions()``, ``balances()`` read the
venue; ``fence(epoch)`` drops queued work of an older epoch;
``export_state()``/``import_state()`` carry the id maps and fill progress
across a restart. See ``docs/architecture/broker-adapters.md``.
"""

from __future__ import annotations

import json
import logging
import secrets
import string
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from raptorbt.execution import (
    AdapterError,
    ExecutionReport,
    Fenced,
    OrderRefused,
    OrderRequest,
    Unsupported,
)
from raptorbt.errors import InvalidParameter

from ._executor import Executor, InlineExecutor
from ._ids import IdCodec
from ._limits import Backoff, RateLimiter

log = logging.getLogger("raptorbt.adapters")

# Normalised order statuses a mapping reduces its broker's strings to.
RECEIVED, OPEN, FILLED, CANCELLED, REJECTED, EXPIRED = (
    "received",
    "open",
    "filled",
    "cancelled",
    "rejected",
    "expired",
)
_TERMINAL = {FILLED, CANCELLED, REJECTED, EXPIRED}

__all__ = [
    "AdapterEvent",
    "AuthExpired",
    "Ambiguous",
    "BrokerAdapter",
    "Fenced",
    "Instrument",
    "MalformedRow",
    "OrderStatusUpdate",
    "QueuedReport",
    "Secret",
    "Session",
    "TradeRow",
    "Transient",
    "VenueBalances",
    "VenueOrder",
    "VenuePosition",
    "VenueRefusal",
    "new_namespace",
]


# -- errors a mapping raises from its I/O methods ------------------------------


class VenueRefusal(AdapterError):
    """The broker answered and said no. ``code`` is its error code."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class Transient(AdapterError):
    """The request was certainly not processed (rate limited, refused, 503 before the OMS)."""


class Ambiguous(AdapterError):
    """A write that may have been processed (timeout, reset, 5xx from the gateway)."""


class MalformedRow(AdapterError):
    """A broker row or message lacks a field this layer cannot do without (a price, a quantity)."""


class AuthExpired(AdapterError):
    """The session token is no longer valid."""


# -- data ----------------------------------------------------------------------


class Secret:
    """A string that does not print. ``repr`` and ``str`` never show it."""

    __slots__ = ("_v",)

    def __init__(self, value: str) -> None:
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "Secret(****)"

    __str__ = __repr__

    def __reduce__(self):  # a pickle or deepcopy must not carry it into a log
        raise TypeError("Secret cannot be serialised")


class Session:
    """What the caller gives the adapter to authenticate: a token and, optionally,
    a way to get a new one.

    ``refresh`` is the caller's own login (it holds the credentials, TOTP seed
    and so on; the adapter never sees them) and returns a new token. When a
    call comes back unauthorised the adapter calls it once and retries; with no
    ``refresh`` the outbox pauses and a ``session_expired`` event is raised
    until the host calls ``adapter.set_token``. Note that a new login on some
    brokers revokes the previous token everywhere: refresh from one place only.
    """

    def __init__(
        self,
        token: str | Secret,
        *,
        refresh: Callable[[], str] | None = None,
        api_key: str | Secret | None = None,
    ) -> None:
        self._token = token if isinstance(token, Secret) else Secret(token)
        self._key = api_key if isinstance(api_key, Secret) or api_key is None else Secret(api_key)
        self.refresh = refresh

    def token(self) -> str:
        return self._token.reveal()

    def api_key(self) -> str | None:
        return self._key.reveal() if self._key else None

    def update(self, token: str | Secret) -> None:
        self._token = token if isinstance(token, Secret) else Secret(token)

    def __repr__(self) -> str:
        return "Session(token=****)"


@dataclass(frozen=True)
class Instrument:
    """How one engine symbol is named at the broker.

    ``exchange`` is the broker's exchange or segment name, ``symbol`` its
    trading symbol, ``key`` its native instrument id (Upstox ``instrument_key``,
    Dhan ``securityId``). ``lot_size``/``tick_size`` are checked locally when
    set; ``quantity_in_lots`` is for a broker that counts that instrument in
    lots on the wire (units on the engine side are always units).
    """

    exchange: str
    symbol: str
    key: str | None = None
    lot_size: float = 1.0
    tick_size: float | None = None
    quantity_in_lots: bool = False


@dataclass(frozen=True)
class OrderStatusUpdate:
    """One broker order row, normalised: a feed message or a row of the order book.

    ``filled_qty`` is cumulative and ``avg_price`` the average over it. A mapping builds its rows
    with the broker's own count (lots for an instrument declared ``quantity_in_lots``); the base
    converts every inbound row once, so rows seen through ``orders()``, ``open_orders()`` and
    ``handle_update`` are always in **units** (prices are per unit).
    """

    venue_order_id: str
    status: str
    client_order_id: str | None = None
    tag: str | None = None
    symbol: str | None = None
    side: str | None = None
    order_type: str | None = None
    product: str | None = None
    quantity: float | None = None
    filled_qty: float = 0.0
    avg_price: float | None = None
    limit_price: float | None = None
    trigger_price: float | None = None
    reason: str | None = None
    ts_ns: int = 0
    cancel_failed: bool = False  # the broker says the cancel it was asked for did not happen
    modify_failed: bool = False  # likewise for a modify


VenueOrder = OrderStatusUpdate


@dataclass(frozen=True)
class TradeRow:
    """One execution as the venue's trade book lists it, with the venue's own id."""

    exec_id: str
    venue_order_id: str
    qty: float
    price: float
    ts_ns: int = 0
    fee: float = 0.0


@dataclass(frozen=True)
class VenuePosition:
    """A net position at the venue. ``net_qty`` is in **units**, never lots (the base converts a
    broker's lot count), and signed: long positive, short negative, flat rows are not listed.
    ``avg_price`` is per unit and positive."""

    symbol: str
    net_qty: float
    avg_price: float
    product: str | None = None


@dataclass(frozen=True)
class VenueBalances:
    """Money at the venue, in ``currency`` (always named: the mapping knows its venue's).

    ``available`` can be used for new orders now; ``used`` is margin or funds already committed and
    ``total`` the account's whole balance, both ``None`` when the broker does not say. Amounts are
    positive money in the currency's main unit (rupees, not paise); a shortfall shows as a small or
    zero ``available``, never a negative ``used``."""

    available: float
    currency: str
    used: float | None = None
    total: float | None = None


@dataclass(frozen=True)
class QueuedReport:
    client_order_id: str
    report: ExecutionReport
    arrival_ns: int


@dataclass(frozen=True)
class AdapterEvent:
    """Something the host should know that is not an order report."""

    kind: str  # session_expired, unconfirmed, submit_failed, connection_lost, connection_restored, ...
    client_order_id: str | None = None
    detail: str = ""
    ts_ns: int = 0


@dataclass
class _Rec:
    client_order_id: str
    tag: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: float | None
    trigger_price: float | None
    tif: str
    product: str
    wire: dict[str, Any]
    fingerprint: tuple
    venue_order_id: str | None = None
    accepted: bool = False
    terminal: set[str] = field(default_factory=set)
    cancel_requested: bool = False
    pending_modify: tuple[float | None, float | None, float | None] | None = None
    cum_qty: float = 0.0
    cum_notional: float = 0.0
    trade_fed: bool = False
    seen_exec: set[str] = field(default_factory=set)
    unconfirmed: bool = False
    attempted: bool = False  # a place request for this order has been started: it may be out there
    terminal_at: float | None = None


@dataclass
class _Job:
    kind: str  # submit, recheck, cancel, modify, resync
    client_order_id: str | None = None
    epoch: int = 0
    change: tuple[float | None, float | None, float | None] | None = None


def new_namespace(length: int = 4) -> str:
    """A fresh per-run tag prefix from the operating system's random source.

    Client ids restart at ``s0-0`` every run, so tags from two runs of a node
    (or two nodes on one account) would collide without it. Generate one at
    each start; keep it in the saved state so a recovered run keeps its own.
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _eq(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-9 + 1e-9 * abs(a)


def _qty_text(q: float) -> str:
    """Exact text of a quantity: an integer count when it is one, else the shortest round-trip form."""
    return str(int(q)) if q == int(q) else repr(q)


def _clock_ns() -> int:
    return time.time_ns()


def _same(a: float | None, b: float | None) -> bool:
    return a is None or (b is not None and abs(a - b) <= 1e-9 + 1e-9 * abs(a))


class BrokerAdapter:
    """Base of the broker mappings. See the module docs; subclasses fill the hooks.

    ``lease`` (default ``None``): the host's proof that this process may still send for the
    current epoch, ``callable(epoch) -> bool`` or an object with ``holds(epoch)``. It is consulted
    before every outbound write (submit, cancel, modify and their retries). A ``False`` answer, or
    an exception, fails closed: the write is refused with :class:`Fenced`, queued writes are
    dropped (``fenced_drop`` events) and nothing more is sent until ``fence()`` is called with a
    higher epoch. Without a lease ``can_fence`` is ``False`` and ``fence()`` guards only this process.

    ``gtc`` (default ``"refuse"``): the venues here have no good-till-cancelled
    order, so ``gtc`` orders (the engine's default) are ``Unsupported`` unless
    ``gtc="day"`` sends them as day orders.

    Hooks a mapping implements: ``translate``, ``check_modify``, ``place``,
    ``cancel_remote``, ``modify_remote``, ``fetch_orders``, ``fetch_positions``,
    ``fetch_balances``, ``parse_order_update`` and optionally ``find_by_tag``.
    """

    #: Normalised order types, tifs and products the mapping supports.
    name = "broker"
    order_types: frozenset[str] = frozenset({"market", "limit"})
    time_in_forces: frozenset[str] = frozenset({"day"})
    products: frozenset[str] = frozenset({"intraday", "delivery"})
    fill_source = "order_updates"  # or "trades"
    #: Whether the venue keeps several positions per instrument apart and can close one by
    #: name (the request's ``position_id``). Brokers here net positions per instrument, so a
    #: mapping says ``False`` and a node with ``oms_type="hedging"`` is refused at construction.
    supports_hedging = False
    tag_max_len = 20
    tag_charset = "A-Za-z0-9"

    def __init__(
        self,
        session: Session,
        instruments: Mapping[str, Instrument] | Callable[[str], Instrument | None],
        *,
        executor: Executor | None = None,
        limiter: RateLimiter | None = None,
        backoff: Backoff | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock_ns: Callable[[], int] = _clock_ns,
        max_attempts: int = 4,
        resend_if_absent: bool = False,
        resend_delay: float = 2.0,
        default_product: str = "intraday",
        reduce_only: str = "refuse",
        gtc: str = "refuse",
        namespace: str,
        account_id: str | None = None,
        fill_source: str | None = None,
        verifier: Callable[[Any], bool] | None = None,
        trusted: bool = False,
        retention_s: float | None = 3600.0,
        monotonic: Callable[[], float] = time.monotonic,
        lease: Callable[[int], bool] | Any | None = None,
    ) -> None:
        if not namespace:
            raise InvalidParameter(
                "namespace is required: client ids restart at s0-0 every run, so pass new_namespace() (a fresh one per run)",
                code="adapter_namespace_missing", operation="BrokerAdapter", parameter="namespace",
            )
        if fill_source is not None:
            if fill_source not in ("order_updates", "trades"):
                raise InvalidParameter("fill_source is 'order_updates' or 'trades'",
                                       code="invalid_parameter", operation="BrokerAdapter",
                                       parameter="fill_source", value=fill_source,
                                       valid=["order_updates", "trades"])
            self.fill_source = fill_source
        if reduce_only not in ("refuse", "best_effort"):
            raise InvalidParameter("reduce_only is 'refuse' or 'best_effort'",
                                   code="invalid_parameter", operation="BrokerAdapter",
                                   parameter="reduce_only", value=reduce_only,
                                   valid=["refuse", "best_effort"])
        if gtc not in ("refuse", "day"):
            raise InvalidParameter("gtc is 'refuse' or 'day'", code="invalid_parameter",
                                   operation="BrokerAdapter", parameter="gtc", value=gtc,
                                   valid=["refuse", "day"])
        if default_product not in self.products:
            raise InvalidParameter(f"default_product must be one of {sorted(self.products)}",
                                   code="invalid_parameter", operation="BrokerAdapter",
                                   parameter="default_product", value=default_product,
                                   valid=sorted(self.products))
        self.session = session
        self._instruments = instruments
        #: Instruments a running node added, by symbol (see ``add_instrument``).
        self._added: dict[str, Instrument] = {}
        self._limiter = limiter or self.default_limiter(sleep)
        self._backoff = backoff or Backoff()
        self._sleep = sleep
        self._now = clock_ns
        self.max_attempts = max_attempts
        self.resend_if_absent = resend_if_absent
        self.resend_delay = resend_delay
        self._verifier = verifier
        self.trusted = trusted
        self.retention_s = retention_s
        self._mono = monotonic
        self._last_sweep = monotonic()
        self.default_product = default_product
        self.reduce_only = reduce_only
        self.gtc = gtc
        self.account_id = account_id
        self.ids = IdCodec(max_len=self.tag_max_len, charset=self.tag_charset, namespace=namespace)

        self._lock = threading.RLock()
        self._pump_lock = threading.Lock()
        self._orders: dict[str, _Rec] = {}
        self._by_venue: dict[str, str] = {}
        self._by_native: dict[str, str] = {}
        self._outbox: deque[_Job] = deque()
        self._reports: deque[QueuedReport] = deque()
        self._events: deque[AdapterEvent] = deque(maxlen=10_000)
        self._evicted: dict[str, None] = {}  # ids of finished orders forgotten after the retention
        self._paused = False
        self._faults: deque[dict[str, Any]] = deque(maxlen=20)
        self._fault_count = 0
        self._epoch = 0
        self._fenced = False
        self._lease = self._lease_check(lease)
        self._seen_unmapped: dict[tuple[str, str], None] = {}
        self._seen_ignored: dict[tuple[str, str, str], None] = {}
        self._native_index: tuple[int, dict[str, str]] = (-1, {})
        self.ignored_updates = 0
        self._executor = executor or InlineExecutor()
        self._executor.bind(self.pump)

    # -- hooks for the mapping ---------------------------------------------------

    def default_limiter(self, sleep: Callable[[float], None]) -> RateLimiter:
        return RateLimiter({}, sleep=sleep)

    def translate(self, request: OrderRequest, instrument: Instrument, product: str, tag: str) -> dict[str, Any]:
        """Pure: the broker's place-order parameters. Raise ``Unsupported``/``OrderRefused``."""
        raise NotImplementedError

    def check_modify(self, rec: _Rec, quantity: float | None, limit_price: float | None, trigger_price: float | None) -> None:
        """Pure: raise ``Unsupported`` if the broker cannot make this change."""

    def place(self, rec: _Rec) -> str:
        """Send ``rec.wire``. Return the venue order id. Raise VenueRefusal/Transient/Ambiguous/AuthExpired."""
        raise NotImplementedError

    def cancel_remote(self, rec: _Rec) -> None:
        raise NotImplementedError

    def modify_remote(self, rec: _Rec, quantity: float | None, limit_price: float | None, trigger_price: float | None) -> None:
        raise NotImplementedError

    def fetch_orders(self) -> list[OrderStatusUpdate]:
        """The day's order book. Rows carry ``tag``; the base decodes it."""
        raise NotImplementedError

    def fetch_trades(self) -> list[TradeRow]:
        raise Unsupported(f"{self.name}: the trade book is not mapped")

    def fetch_positions(self) -> list[VenuePosition]:
        raise Unsupported(f"{self.name}: positions are not mapped")

    def fetch_balances(self) -> VenueBalances:
        raise Unsupported(f"{self.name}: balances are not mapped")

    def place_exit_rule(self, request: Any) -> None:
        """Hold an exit rule (an :class:`~raptorbt.strategy.exits.ExitRuleRequest`) at the venue.

        Unsupported by default, so the engine evaluates every rule itself. A mapping whose
        broker can hold a percent or P&L exit overrides this, and reports the orders the rule
        makes under ``request.exit_client_ids``. A mapping that holds rules overrides
        :meth:`cancel_exit_rule` and :meth:`replace_exit_rule` too: the engine ends or changes
        a held rule through them, and refuses to leave one at a venue that cannot be told."""
        raise Unsupported(f"{self.name}: exit rules are not held at the venue")

    def cancel_exit_rule(self, request: Any) -> None:
        """Take back an exit rule :meth:`place_exit_rule` accepted (``request`` is the one that
        was placed). Return when the venue no longer holds it; raise ``Unsupported`` when it
        cannot be taken back, and the engine will not report the rule cancelled."""
        raise Unsupported(f"{self.name}: a held exit rule cannot be cancelled")

    def replace_exit_rule(self, old: Any, new: Any) -> None:
        """Change a held rule to cover ``new.legs`` (its size or positions changed): afterwards
        the venue holds ``new`` and not ``old``. Raise ``Unsupported`` when it cannot be changed
        in place; the engine then cancels it and watches the rule itself."""
        raise Unsupported(f"{self.name}: a held exit rule cannot be changed in place")

    def parse_order_update(self, payload: Any) -> OrderStatusUpdate | None:
        """A feed message to an update; ``None`` for messages that are not order updates."""
        raise NotImplementedError

    def find_by_tag(self, tag: str) -> list[OrderStatusUpdate]:
        """Every row of the day's book carrying ``tag`` (an earlier run's included)."""
        return [row for row in self.fetch_orders() if row.tag == tag]

    # -- the ExecutionAdapter surface (node thread) ---------------------------------

    def resolve_instrument(self, symbol: str, spec: Any) -> Instrument | None:
        """How a contract added to a running node is named at the broker, when ``instruments`` does
        not know it yet. A mapping that can find it (a lookup in the day's instrument master)
        overrides this; the default knows nothing."""
        return None

    def add_instrument(self, symbol: str, spec: Any = None) -> None:
        """A node added ``symbol`` (its :class:`~raptorbt.InstrumentSpec` is ``spec``): learn how
        the broker names it, so orders for it can be sent. Looks in ``instruments`` first, then
        asks :meth:`resolve_instrument`; raises :class:`Unsupported` when neither knows it, which
        makes the node refuse the add. Idempotent: a restored node replays it."""
        ins = self._added.get(symbol)
        if ins is None:
            try:
                ins = self.instrument(symbol)
            except Unsupported:
                ins = self.resolve_instrument(symbol, spec)
        if ins is None:
            raise Unsupported(f"{self.name}: no instrument is mapped for symbol {symbol!r}")
        self._added[symbol] = ins

    def retire_instrument(self, symbol: str) -> None:
        """A node retired ``symbol``: forget what :meth:`add_instrument` learned. Orders still
        working at the broker are not touched (a node retires a symbol only with none)."""
        self._added.pop(symbol, None)

    def instrument(self, symbol: str) -> Instrument:
        ins = self._added.get(symbol)
        if ins is None:
            ins = self._instruments.get(symbol) if isinstance(self._instruments, Mapping) else self._instruments(symbol)
        if ins is None:
            raise Unsupported(f"{self.name}: no instrument is mapped for symbol {symbol!r}")
        return ins

    def submit(self, request: OrderRequest) -> None:
        self._refuse_if_fenced()
        if getattr(request, "position_id", None) is not None and not self.supports_hedging:
            raise Unsupported(f"{self.name}: this venue nets positions; it cannot close a named position")
        if request.account_id and self.account_id and request.account_id != self.account_id:
            raise OrderRefused("order is for another account than this session")
        if request.post_only:
            raise Unsupported(f"{self.name}: post_only orders have no equivalent")
        if request.reduce_only and self.reduce_only == "refuse":
            raise Unsupported(f"{self.name}: no reduce-only flag; pass reduce_only='best_effort' to send it as a plain order")
        if request.order_type not in self.order_types:
            raise Unsupported(f"{self.name}: order type {request.order_type!r} is not supported")
        if request.time_in_force == "gtc" and self.gtc == "day":
            pass  # the caller chose to let a good-till-cancelled order lapse at the end of the day
        elif request.time_in_force not in self.time_in_forces:
            hint = "; pass gtc='day' to send it as a day order" if request.time_in_force == "gtc" else ""
            raise Unsupported(f"{self.name}: time in force {request.time_in_force!r} is not supported{hint}")
        product = request.tags.get("product", self.default_product)
        if product not in self.products:
            raise Unsupported(f"{self.name}: product {product!r} is not supported")
        ins = self.instrument(request.symbol)
        self._check_size(ins, request.quantity, request.limit_price, request.trigger_price)
        tag = self.ids.encode(request.client_order_id)
        wire = self.translate(request, ins, product, tag)
        fp = (
            request.symbol, request.side, request.quantity, request.order_type, request.limit_price,
            request.trigger_price, request.time_in_force, product,
        )  # fmt: skip
        with self._lock:
            if request.client_order_id in self._evicted:
                raise OrderRefused("duplicate_client_order_id: the id belongs to a finished order")
            have = self._orders.get(request.client_order_id)
            if have is not None:
                if have.fingerprint != fp:
                    raise OrderRefused("duplicate_client_order_id: the id was used for different terms")
                if have.terminal or any(
                    j.kind in ("submit", "recheck") and j.client_order_id == have.client_order_id
                    for j in self._outbox
                ):
                    return  # the same order again: already queued, or over
                # The same order again, asked for after it was sent: the node's resend of an
                # order its reconcile found missing. Never a second order blindly: the venue's
                # book is read by tag first, and the order is placed only if it is not there.
                have.wire = wire
                self._outbox.append(_Job("recheck", have.client_order_id, self._epoch))
                self._executor.kick()
                return
            self._orders[request.client_order_id] = _Rec(
                request.client_order_id, tag, request.symbol, request.side, request.quantity,
                request.order_type, request.limit_price, request.trigger_price,
                request.time_in_force, product, wire, fp,
            )  # fmt: skip
            self._by_native[self.native_key(ins)] = request.symbol
            self._outbox.append(_Job("submit", request.client_order_id, self._epoch))
        self._executor.kick()

    def cancel(self, client_order_id: str) -> None:
        self._refuse_if_fenced()
        with self._lock:
            rec = self._orders.get(client_order_id)
            if rec is None:
                raise OrderRefused(f"unknown order {client_order_id}")
            if rec.cancel_requested or rec.terminal:
                return
            rec.cancel_requested = True
            self._outbox.append(_Job("cancel", client_order_id, self._epoch))
        self._executor.kick()

    def modify(
        self,
        client_order_id: str,
        *,
        quantity: float | None = None,
        limit_price: float | None = None,
        trigger_price: float | None = None,
    ) -> None:
        self._refuse_if_fenced()
        with self._lock:
            rec = self._orders.get(client_order_id)
            if rec is None:
                raise OrderRefused(f"unknown order {client_order_id}")
            self.check_modify(rec, quantity, limit_price, trigger_price)
            self._check_size(self.instrument(rec.symbol), quantity or rec.quantity, limit_price, trigger_price)
            self._outbox.append(_Job("modify", client_order_id, self._epoch, (quantity, limit_price, trigger_price)))
        self._executor.kick()

    def _check_size(self, ins: Instrument, qty: float, limit: float | None, trigger: float | None) -> None:
        if qty <= 0 or qty != int(qty):
            raise OrderRefused(f"quantity {qty} is not a whole number of units")
        if ins.lot_size > 1 and qty % ins.lot_size:
            raise OrderRefused(f"quantity {qty} is not a multiple of the lot size {ins.lot_size}")
        if ins.tick_size:
            for p in (limit, trigger):
                if p is not None and abs(p / ins.tick_size - round(p / ins.tick_size)) > 1e-6:
                    raise OrderRefused(f"price {p} is not a multiple of the tick {ins.tick_size}")

    # -- the outbox (executor thread) ----------------------------------------------

    def pump(self) -> int:
        """Run queued jobs in order until the outbox is empty or paused. Returns jobs run."""
        done = 0
        while True:
            if not self._pump_lock.acquire(blocking=False):
                return done  # another thread is draining and will see new jobs
            try:
                while not self._paused:
                    with self._lock:
                        job = self._outbox[0] if self._outbox else None
                    if job is None:
                        break
                    try:
                        finished = self._run(job)
                    except Exception as exc:  # noqa: BLE001 - a bug in a mapping must not stop the outbox
                        self._job_failed(job, exc)
                        finished = True
                    if not finished:
                        break
                    with self._lock:
                        if self._outbox and self._outbox[0] is job:
                            self._outbox.popleft()
                    done += 1
            finally:
                self._pump_lock.release()
            self.sweep()
            with self._lock:
                if self._paused or not self._outbox:
                    return done

    def _run(self, job: _Job) -> bool:
        """True when the job is finished; False to hold it at the head (paused)."""
        if job.kind == "resync":
            self._resync_now()
            return True
        with self._lock:
            rec = self._orders.get(job.client_order_id or "")
            stale = job.epoch < self._epoch
        if rec is None or stale:
            if stale:
                self._event("fenced_drop", job.client_order_id, f"{job.kind} of epoch {job.epoch} < {self._epoch}")
                self._report_unsent(job, "fenced")
            return True
        try:
            return getattr(self, f"_do_{job.kind}")(rec, job)
        except Fenced:
            self._drop_fenced()
            return True

    def _call(self, fn: Callable[[], Any], klass: str, *, retry_ambiguous: bool) -> Any:
        """Rate limit, call, retry on ``Transient`` (and ``Ambiguous`` when idempotent).

        Raises ``Ambiguous`` or ``Transient`` when attempts run out, and
        ``AuthExpired`` after one failed refresh; ``VenueRefusal`` passes through.
        """
        refreshed = False
        for attempt in range(self.max_attempts):
            if klass != "read" and not self._holds_lease():
                raise Fenced(f"lease lost at epoch {self._epoch}")
            self._limiter.acquire(klass)
            try:
                return fn()
            except AuthExpired:
                if refreshed or not self._refresh():
                    raise
                refreshed = True
            except Transient:
                if attempt + 1 >= self.max_attempts:
                    raise
            except Ambiguous:
                if not retry_ambiguous or attempt + 1 >= self.max_attempts:
                    raise
            self._sleep(self._backoff.delay(attempt))
        raise Transient("out of attempts")  # pragma: no cover

    def _refresh(self) -> bool:
        if self.session.refresh is None:
            return False
        try:
            self.session.update(self.session.refresh())
        except Exception as exc:  # noqa: BLE001 - the caller's login failed; keep the reason out of logs
            self._event("session_refresh_failed", None, type(exc).__name__)
            return False
        self._event("session_refreshed")
        return True

    def _read(self, fn: Callable[[], Any]) -> Any:
        return self._call(fn, "read", retry_ambiguous=True)

    def _hold(self, job_kind: str, rec: _Rec) -> bool:
        self._paused = True
        self._event("session_expired", rec.client_order_id, f"{job_kind} held until set_token()")
        return False

    def _do_submit(self, rec: _Rec, job: _Job) -> bool:
        if rec.terminal or rec.venue_order_id:
            return True
        if rec.attempted:
            # A request for this order was already started (before a restart, or by an earlier try):
            # it may be at the venue. Look it up; never place it blindly again.
            self._settle_unconfirmed(rec, "resumed with a request possibly in flight")
            return True
        rec.attempted = True
        try:
            vid = self._call(lambda: self.place(rec), "orders", retry_ambiguous=False)
        except Fenced:
            rec.attempted = False  # place retries only after "certainly not processed": nothing is out there
            raise
        except VenueRefusal as e:
            self._terminal(rec, "rejected", ExecutionReport.rejected(self._reason(e), self._now()))
        except Transient as e:
            self._event("submit_failed", rec.client_order_id, str(e))
            self._terminal(rec, "rejected", ExecutionReport.rejected(f"venue unreachable: {e}", self._now()))
        except AuthExpired:
            rec.attempted = False  # the venue turned the request away; it goes again with the new token
            return self._hold("submit", rec)
        except Ambiguous as e:
            self._settle_unconfirmed(rec, str(e))
        else:
            self._bind(rec, vid)
        return True

    def _do_recheck(self, rec: _Rec, job: _Job) -> bool:
        """The node sent an order again that this adapter already knows (its reconcile found the
        order missing). Read the day's book by tag: an order that is there is adopted, whatever
        state it is in, and nothing is placed; an order that is not there is placed once more
        under the same tag, after the book has shown it absent twice. A book that cannot be read, or
        shows several fits, places nothing."""
        if rec.terminal:
            return True
        row, state = self._try_lookup(rec)
        if state == "absent":
            # The book can lag a request that was only just sent: absent must be seen twice,
            # ``resend_delay`` apart, before the order is placed again.
            self._sleep(self.resend_delay)
            row, state = self._try_lookup(rec)
        if state == "found" and row is not None:
            self.handle_update(row)
        elif state == "absent":
            rec.venue_order_id = None
            rec.attempted = False
            rec.unconfirmed = False
            self._event("resend", rec.client_order_id, "absent from the venue's book: placed again")
            return self._do_submit(rec, job)
        else:
            rec.unconfirmed = True
            self._event("unconfirmed", rec.client_order_id, f"resend not placed: the venue's book is {state}")
        return True

    def _job_failed(self, job: _Job, exc: Exception) -> None:
        """A job raised something that is not an adapter error: a bug in a mapping, or a broker
        answer it did not expect. It is recorded (``health()``), the strategy is told what it can
        be told, and the outbox goes on. A submit that failed this way may have reached the venue:
        the order is left unconfirmed for the node's reconcile, never rejected or placed again."""
        detail = f"{job.kind} raised {type(exc).__name__}: {exc}"
        log.error("%s %s %s", "job_failed", job.client_order_id or "-", detail, exc_info=exc)
        with self._lock:
            self._fault_count += 1
            self._faults.append({"job": job.kind, "client_order_id": job.client_order_id, "error": detail})
            rec = self._orders.get(job.client_order_id or "")
            if rec is not None and not rec.terminal:
                if job.kind in ("submit", "recheck"):
                    rec.unconfirmed = True
                elif job.kind == "cancel":
                    rec.cancel_requested = False
                    self._push(rec, ExecutionReport.cancel_rejected(f"adapter error: {detail}", self._now()))
                elif job.kind == "modify":
                    rec.pending_modify = None
                    self._push(rec, ExecutionReport.replace_rejected(f"adapter error: {detail}", self._now()))
        self._event("job_failed", job.client_order_id, detail)

    def health(self) -> dict[str, Any]:
        """What a supervisor needs to know the adapter is working: whether the outbox is paused
        (an expired token), how many jobs wait, how many raised an unexpected error and the last
        few, and the executor's own state (a thread that died is ``alive: False``)."""
        with self._lock:
            out = {
                "paused": self._paused,
                "outbox": len(self._outbox),
                "faults": self._fault_count,
                "recent_faults": list(self._faults),
            }
        executor_health = getattr(self._executor, "health", None)
        if callable(executor_health):
            out["executor"] = executor_health()
        return out

    def catch_up(self) -> list[QueuedReport]:
        """Read the venue's order book (and its trades, for ``fill_source="trades"``) now and
        return the reports of everything the node has not heard: a fill, a cancel or a rejection
        that happened while no node was listening. Blocking: call it from the node's thread
        before it compares itself with the venue (``LiveNode.reconcile`` does)."""
        # Only the node's own orders: a row for an order it finished before the restart, or for
        # one it never sent, is not an update this adapter ignores (``ignored_updates`` counts
        # those the feed brings); reconcile names the venue's unknown orders itself.
        for row in self._read_orders():
            with self._lock:
                ours = self._resolve(row) is not None
            if ours:
                self.handle_update(row)
            elif row.filled_qty > 0 and not (row.tag and self.ids.decode(row.tag)):
                # A fill on an order that is no node's of this account: it moves the venue's
                # position and no report can book it. Counted, and an event names it.
                self._ignored("catch_up", row.venue_order_id, row.status, "a filled order of no node here", row.tag)
        if self.fill_source == "trades":
            for t in self._read(self.fetch_trades):
                if t.venue_order_id in self._by_venue:
                    self.handle_trade(
                        exec_id=t.exec_id, qty=t.qty, price=t.price, venue_order_id=t.venue_order_id,
                        fee=t.fee, ts_ns=t.ts_ns, wire=True,
                    )
        return self.drain_reports()

    def _bind(self, rec: _Rec, vid: str) -> None:
        with self._lock:
            rec.venue_order_id = vid
            self._by_venue[vid] = rec.client_order_id

    def _try_lookup(self, rec: _Rec) -> tuple[OrderStatusUpdate | None, str]:
        """Find our order in the day's book by tag, among rows whose terms are this order's.

        Returns the row and ``found``; or ``absent``, ``ambiguous`` (several rows fit: an earlier
        run left an identical order) or ``failed`` (the book could not be read).
        """
        try:
            rows = [self._in_units(r) for r in self._read(lambda: self.find_by_tag(rec.tag))]
        except AdapterError:
            return None, "failed"
        ours = [r for r in rows if self._matches(rec, r)]
        if len(ours) > 1:
            return None, "ambiguous"
        return (ours[0], "found") if ours else (None, "absent")

    def _settle_unconfirmed(self, rec: _Rec, why: str) -> None:
        """A write may have landed. Look it up; adopt it, or leave it to reconciliation.

        With ``resend_if_absent`` an order still absent is sent once more, but only after
        ``resend_delay`` seconds and a second lookup: the book can lag a request that is
        still in flight.
        """
        row, state = self._try_lookup(rec)
        if state == "absent" and self.resend_if_absent:
            self._sleep(self.resend_delay)
            row, state = self._try_lookup(rec)
            if state == "absent":
                self._event("resend", rec.client_order_id, "absent from the book twice after an ambiguous failure")
                try:
                    vid = self._call(lambda: self.place(rec), "orders", retry_ambiguous=False)
                except VenueRefusal as e:
                    self._terminal(rec, "rejected", ExecutionReport.rejected(self._reason(e), self._now()))
                    return
                except Fenced:
                    raise
                except AdapterError as e:
                    why = f"resend failed: {e}"
                else:
                    self._bind(rec, vid)
                    return
        if row is not None:
            self.handle_update(row)
            return
        rec.unconfirmed = True
        self._event("unconfirmed", rec.client_order_id, f"{why} ({state})")

    def _ensure_venue_id(self, rec: _Rec) -> bool:
        if rec.venue_order_id:
            return True
        row, _ = self._try_lookup(rec)
        if row is None:
            return False
        self.handle_update(row)
        return bool(rec.venue_order_id)

    def _do_cancel(self, rec: _Rec, job: _Job) -> bool:
        if rec.terminal:
            return True
        if not self._ensure_venue_id(rec):
            rec.cancel_requested = False
            self._push(rec, ExecutionReport.cancel_rejected("order not found at the venue", self._now()))
            return True
        try:
            self._call(lambda: self.cancel_remote(rec), "cancel", retry_ambiguous=True)
        except VenueRefusal as e:
            rec.cancel_requested = False
            self._push(rec, ExecutionReport.cancel_rejected(self._reason(e), self._now()))
        except AuthExpired:
            return self._hold("cancel", rec)
        except Fenced:
            raise
        except AdapterError as e:
            self._settle_cancel(rec, str(e))
        return True

    def _settle_cancel(self, rec: _Rec, why: str) -> None:
        """The cancel could not be confirmed: read the book, ask once more.

        Only a definite answer (the venue refused, or certainly did not receive the request) is
        reported as ``cancel_rejected``. If the second try is ambiguous too, the cancel may have
        landed: the order stays ``cancel_requested`` (pending), a ``cancel_pending`` event is
        raised, and ``settle_pending()`` or the next resync decides against the venue's book.
        """
        self._event("cancel_unconfirmed", rec.client_order_id, why)
        try:
            self._resync_now()
        except AdapterError:
            pass
        if rec.terminal:
            return
        try:
            self._call(lambda: self.cancel_remote(rec), "cancel", retry_ambiguous=True)
            return  # the venue took it; the ack comes from the feed or the next resync
        except VenueRefusal as e:
            why = self._reason(e)
        except Transient as e:
            why = str(e)  # certainly not received
        except Fenced:
            raise
        except AdapterError as e:
            self._event("cancel_pending", rec.client_order_id, f"may have landed, left pending: {e}")
            return
        rec.cancel_requested = False
        self._push(rec, ExecutionReport.cancel_rejected(f"cancel unconfirmed: {why}", self._now()))

    def _do_modify(self, rec: _Rec, job: _Job) -> bool:
        if rec.terminal:
            return True
        if not self._ensure_venue_id(rec):
            self._push(rec, ExecutionReport.replace_rejected("order not found at the venue", self._now()))
            return True
        q, lim, trg = job.change or (None, None, None)
        with self._lock:
            rec.pending_modify = (q, lim, trg)
        try:
            self._call(lambda: self.modify_remote(rec, q, lim, trg), "modify", retry_ambiguous=True)
        except VenueRefusal as e:
            rec.pending_modify = None
            self._push(rec, ExecutionReport.replace_rejected(self._reason(e), self._now()))
        except AuthExpired:
            return self._hold("modify", rec)
        except Fenced:
            raise
        except AdapterError as e:
            self._settle_modify(rec, (q, lim, trg), str(e))
        return True

    def _settle_modify(self, rec: _Rec, change: tuple, why: str) -> None:
        """The modify could not be confirmed: read the book, ask once more.

        Only a definite answer (refused, or certainly not received) becomes ``replace_rejected``.
        If the second try is ambiguous too the change may have landed, so ``pending_modify`` stays
        (``rec.quantity`` is not touched, and ``_matches`` accepts the old and the new terms) with
        a ``modify_pending`` event, until ``settle_pending()`` or a resync shows the venue's terms.
        """
        self._event("modify_unconfirmed", rec.client_order_id, why)
        try:
            self._resync_now()
        except AdapterError:
            pass
        if rec.terminal or rec.pending_modify is None:
            return  # the change showed in the book (acked) or the order ended
        try:
            self._call(lambda: self.modify_remote(rec, *change), "modify", retry_ambiguous=True)
            return
        except VenueRefusal as e:
            why = self._reason(e)
        except Transient as e:
            why = str(e)  # certainly not received
        except Fenced:
            raise
        except AdapterError as e:
            self._event("modify_pending", rec.client_order_id, f"may have landed, left pending: {e}")
            return
        rec.pending_modify = None
        self._push(rec, ExecutionReport.replace_rejected(f"modify unconfirmed: {why}", self._now()))

    def settle_pending(self) -> None:
        """Answer cancels and modifies the venue took but never confirmed.

        Reads the order book (blocking: call it from a worker or a timer, not the node's thread).
        A modify whose terms the book still shows unchanged becomes ``replace_rejected``, a
        cancel of an order the book still shows working becomes ``cancel_rejected``, so the
        strategy is never left in PendingReplace or PendingCancel. Call it once the venue has had
        time to answer (seconds); brokers that report a failed request (Upstox "not modified")
        are answered from the feed without it.
        """
        rows = {r.venue_order_id: r for r in self._read_orders()}
        for r in rows.values():
            self.handle_update(r)
        with self._lock:
            queued = {j.client_order_id for j in self._outbox}
            for rec in list(self._orders.values()):
                row = rows.get(rec.venue_order_id or "")
                if row is None or rec.terminal or rec.client_order_id in queued:
                    continue
                if rec.pending_modify is not None:
                    rec.pending_modify = None
                    self._push(rec, ExecutionReport.replace_rejected("the venue still shows the old terms", self._now()))
                if rec.cancel_requested:
                    rec.cancel_requested = False
                    self._push(rec, ExecutionReport.cancel_rejected("the venue still shows the order working", self._now()))

    @staticmethod
    def _reason(e: VenueRefusal) -> str:
        return f"{e.code}: {e}" if e.code else str(e)

    def resume(self) -> None:
        """Continue after a pause (``set_token`` calls this)."""
        self._paused = False
        self._executor.kick()

    def set_token(self, token: str | Secret) -> None:
        self.session.update(token)
        self.resume()

    # -- feed side ----------------------------------------------------------------------

    def handle_payload(self, payload: Any, *, arrival_ns: int | None = None, authenticated: bool = False) -> None:
        """A raw feed message (postback, websocket frame already decoded).

        Nothing from a push channel becomes a report unless it is authenticated: the adapter was
        built with ``trusted=True`` (the host authenticated every channel itself, e.g. a websocket
        opened with the account's token), or with a ``verifier`` that accepts this payload (e.g.
        ``kite.postback_verifier``), or this call passes ``authenticated=True``. Otherwise it raises.
        A payload the verifier refuses, or that lacks a field the layer needs, is dropped with a
        ``payload_rejected`` / ``malformed_payload`` event.
        """
        if not (self.trusted or authenticated):
            if self._verifier is None:
                raise AdapterError(
                    "unauthenticated payload: pass verifier=..., trusted=True (the host authenticated the channel) "
                    "or authenticated=True for this call"
                )
            try:
                accepted = bool(self._verifier(payload))
            except Exception:  # noqa: BLE001 - a verifier that fails verifies nothing
                accepted = False
            if not accepted:
                self._event("payload_rejected", None, "failed verification")
                return
        try:
            update = self.parse_order_update(payload)
        except MalformedRow as e:
            self._event("malformed_payload", None, str(e))
            return
        if update is not None:
            self.handle_update(self._in_units(update), arrival_ns=arrival_ns)

    def handle_update(self, u: OrderStatusUpdate, *, arrival_ns: int | None = None) -> None:
        """Fold one order row into reports. Safe to repeat and to reorder."""
        with self._lock:
            rec = self._resolve(u)
            if rec is None:
                self._ignored("update", u.venue_order_id, u.status, "no order of this node has that id or tag", u.tag)
                return
            if rec.venue_order_id is None:
                if not self._matches(rec, u):
                    # the same tag with other terms: an earlier run's order, not ours
                    self._ignored("update", u.venue_order_id, u.status, "the tag is ours, the terms are not", u.tag, rec.client_order_id)
                    return
                rec.venue_order_id = u.venue_order_id
                self._by_venue[u.venue_order_id] = rec.client_order_id
            elif rec.venue_order_id != u.venue_order_id:
                if self._matches(rec, u):
                    self._event("venue_id_mismatch", rec.client_order_id, f"{u.venue_order_id} != {rec.venue_order_id}")
                else:
                    self._ignored("update", u.venue_order_id, u.status, "another venue order carries this order's tag", u.tag, rec.client_order_id)
                return
            arrival = arrival_ns if arrival_ns is not None else self._now()
            ts = u.ts_ns or arrival
            if u.status not in (RECEIVED, REJECTED) and not rec.accepted:
                rec.accepted = True
                self._push(rec, ExecutionReport.accepted(u.venue_order_id, ts), arrival)
            if rec.pending_modify is not None and u.status != REJECTED and self._terms_match(rec, u):
                q, lim, trg = rec.pending_modify
                rec.pending_modify = None
                rec.quantity = q if q is not None else rec.quantity
                rec.limit_price = lim if lim is not None else rec.limit_price
                rec.trigger_price = trg if trg is not None else rec.trigger_price
                self._push(rec, ExecutionReport.replace_ack(ts, new_qty=q, new_limit=lim, new_trigger=trg), arrival)
            if u.modify_failed and rec.pending_modify is not None:
                rec.pending_modify = None
                self._push(rec, ExecutionReport.replace_rejected(u.reason or "modify refused by the venue", ts), arrival)
            if u.cancel_failed and rec.cancel_requested and not rec.terminal:
                rec.cancel_requested = False
                self._push(rec, ExecutionReport.cancel_rejected(u.reason or "cancel refused by the venue", ts), arrival)
            if self.fill_source == "order_updates":
                self._fold_cumulative(rec, u, ts, arrival)
            if u.status == CANCELLED or (u.status == REJECTED and u.filled_qty > 0):
                self._terminal(rec, "cancel", ExecutionReport.cancel_ack(ts), arrival)
            elif u.status == REJECTED:
                self._terminal(rec, "rejected", ExecutionReport.rejected(u.reason or "rejected by the venue", ts), arrival)
            elif u.status == EXPIRED:
                self._terminal(rec, "expired", ExecutionReport.expired(ts), arrival)
            elif u.status == FILLED:
                self._mark_terminal(rec, "filled")

    def handle_trade(
        self,
        *,
        exec_id: str,
        qty: float,
        price: float,
        venue_order_id: str | None = None,
        client_order_id: str | None = None,
        fee: float = 0.0,
        ts_ns: int = 0,
        arrival_ns: int | None = None,
        wire: bool = False,
    ) -> None:
        """A real execution with the venue's own id (for ``fill_source="trades"``).

        ``qty`` is in units, unless ``wire=True``: then it is the broker's own count (lots for an
        instrument declared ``quantity_in_lots``) and is converted here, once."""
        with self._lock:
            cid = client_order_id or self._by_venue.get(venue_order_id or "")
            rec = self._orders.get(cid or "")
            if rec is None:
                self._ignored("trade", venue_order_id, None, f"execution {exec_id} belongs to no order of this node", None, client_order_id)
                return
            if exec_id in rec.seen_exec:
                return
            if wire:
                qty = self.units(self.instrument(rec.symbol), qty)
            rec.seen_exec.add(exec_id)
            rec.trade_fed = True
            arrival = arrival_ns if arrival_ns is not None else self._now()
            self._push(rec, ExecutionReport.fill(exec_id, qty, price, ts_ns or arrival, fee=fee), arrival)

    def _ignored(self, what: str, venue_order_id: str | None, status: str | None, why: str, tag: str | None = None, cid: str | None = None) -> None:
        """Count a row or trade that changed nothing, and say so once per venue order and status.

        Every venue order of the account that is not this node's lands here (another strategy, a
        manual order), so the event is not repeated for the same order and status; the counter
        still counts every one. An order of this node's that is ignored shows up here too, and
        that is the case to look for after a restart."""
        self.ignored_updates += 1
        key = (what, str(venue_order_id), str(status))
        if key in self._seen_ignored:
            return
        self._seen_ignored[key] = None
        while len(self._seen_ignored) > 10_000:
            self._seen_ignored.pop(next(iter(self._seen_ignored)))
        detail = json.dumps({"what": what, "venue_order_id": venue_order_id, "status": status, "tag": tag, "why": why})
        self._event("ignored_update", cid or (self.ids.decode(tag) if tag else None), detail)

    def _resolve(self, u: OrderStatusUpdate) -> _Rec | None:
        cid = self._by_venue.get(u.venue_order_id) or u.client_order_id or self.ids.decode(u.tag)
        return self._orders.get(cid) if cid else None

    def _matches(self, rec: _Rec, u: OrderStatusUpdate) -> bool:
        """Does a venue row carry this order's terms (symbol, side, quantity, type, price)?

        A tag alone never makes a row ours: ids restart every run, so an earlier run's order can
        carry the same one. A pending modify's target terms count as well.
        """
        if u.symbol is not None and u.symbol != rec.symbol:
            return False
        if u.side is not None and u.side != rec.side:
            return False
        if u.order_type is not None and u.order_type != rec.order_type:
            return False
        pq, pl, _ = rec.pending_modify or (None, None, None)
        if u.quantity is not None and not (_eq(rec.quantity, u.quantity) or _eq(pq, u.quantity)):
            return False
        if u.limit_price is not None and rec.limit_price is not None and not (_eq(rec.limit_price, u.limit_price) or _eq(pl, u.limit_price)):
            return False
        return True

    def _terms_match(self, rec: _Rec, u: OrderStatusUpdate) -> bool:
        q, lim, trg = rec.pending_modify or (None, None, None)
        return _same(q, u.quantity) and _same(lim, u.limit_price) and _same(trg, u.trigger_price)

    def _fold_cumulative(self, rec: _Rec, u: OrderStatusUpdate, ts: int, arrival: int) -> None:
        if u.filled_qty <= rec.cum_qty + 1e-12:
            return  # stale or repeated
        avg = u.avg_price
        if avg is None:
            self._event("fill_without_price", rec.client_order_id, u.venue_order_id)
            return
        delta = u.filled_qty - rec.cum_qty
        notional = u.filled_qty * avg
        px = round((notional - rec.cum_notional) / delta, 6)
        rec.cum_qty, rec.cum_notional = u.filled_qty, notional
        exec_id = f"{u.venue_order_id}:{_qty_text(u.filled_qty)}"
        rec.seen_exec.add(exec_id)
        self._push(rec, ExecutionReport.fill(exec_id, delta, px, ts), arrival)

    def _terminal(self, rec: _Rec, kind: str, report: ExecutionReport, arrival: int | None = None) -> None:
        with self._lock:
            if kind in rec.terminal:
                return
            self._mark_terminal(rec, kind)
            self._push(rec, report, arrival)

    def _mark_terminal(self, rec: _Rec, kind: str) -> None:
        rec.terminal.add(kind)
        if rec.terminal_at is None:
            rec.terminal_at = self._mono()

    def sweep(self) -> int:
        """Forget finished orders older than ``retention_s`` (None keeps them). Returns how many.

        What an order needs after it ends is its id (a repeated submit is refused), its venue id
        and fill ids (late and repeated reports add nothing). Reports for an order finish within
        seconds of its end, so an hour is far more than needed; the ids of forgotten orders are
        remembered (the last 100 000) so a repeated submit is still refused and a late message
        still ignored.
        """
        if self.retention_s is None:
            return 0
        now = self._mono()
        with self._lock:
            queued = {j.client_order_id for j in self._outbox}
            gone = [
                c for c, r in self._orders.items()
                if r.terminal_at is not None and now - r.terminal_at >= self.retention_s
                and c not in queued and r.pending_modify is None
            ]  # fmt: skip
            for c in gone:
                r = self._orders.pop(c)
                if r.venue_order_id:
                    self._by_venue.pop(r.venue_order_id, None)
                self._evicted[c] = None
            while len(self._evicted) > 100_000:
                self._evicted.pop(next(iter(self._evicted)))
        return len(gone)

    # -- queues ---------------------------------------------------------------------------

    def _push(self, rec: _Rec, report: ExecutionReport, arrival: int | None = None) -> None:
        self._reports.append(QueuedReport(rec.client_order_id, report, arrival if arrival is not None else self._now()))

    def _event(self, kind: str, cid: str | None = None, detail: str = "") -> None:
        self._events.append(AdapterEvent(kind, cid, detail, self._now()))
        log.info("%s %s %s", kind, cid or "-", detail)

    def drain_reports(self) -> list[QueuedReport]:
        """Reports in the order they were produced. Push each with ``node.push_report``."""
        out = []
        while True:
            try:
                out.append(self._reports.popleft())
            except IndexError:
                return out

    def drain_events(self) -> list[AdapterEvent]:
        out = []
        while True:
            try:
                out.append(self._events.popleft())
            except IndexError:
                return out

    def deliver_to(self, node: Any) -> list[str]:
        """Drain the queue into ``node.push_report`` (call on the node's thread, never from inside an adapter method)."""
        return [
            node.push_report(q.client_order_id, q.report, arrival_ns=q.arrival_ns) for q in self.drain_reports()
        ]

    # -- connection hooks -------------------------------------------------------------------

    def connection_lost(self, reason: str = "") -> None:
        self._event("connection_lost", None, reason)

    def connection_restored(self) -> None:
        """The feed is back: read the order book, since messages may have been missed."""
        self._event("connection_restored")
        self.resync()

    def resync(self) -> None:
        """Queue a read of the venue's order book (runs on the executor)."""
        with self._lock:
            self._outbox.append(_Job("resync"))
        self._executor.kick()

    def _resync_now(self) -> None:
        for row in self._read_orders():
            self.handle_update(row)
        if self.fill_source == "trades":
            self.sync_trades()

    def sync_trades(self) -> None:
        """Read the trade book and book every execution not yet seen (``fill_source="trades"``)."""
        for t in self._read(self.fetch_trades):
            self.handle_trade(exec_id=t.exec_id, qty=t.qty, price=t.price, venue_order_id=t.venue_order_id, fee=t.fee, ts_ns=t.ts_ns, wire=True)

    # -- reconcile reads and fencing (the journal work uses these) ---------------------------

    def orders(self) -> list[VenueOrder]:
        """The venue's order book for the day, normalised; ``client_order_id`` set where the tag is ours."""
        return [self._with_cid(r) for r in self._read_orders()]

    def open_orders(self) -> list[VenueOrder]:
        """``orders()`` that are not finished."""
        return [o for o in self.orders() if o.status not in _TERMINAL]

    def positions(self) -> list[VenuePosition]:
        """Net positions at the venue in **units** (short negative); lots are converted here."""
        out = []
        for p in self._read(self.fetch_positions):
            ins = self._instrument_or_none(p.symbol)
            if ins is None:
                # The broker's own key (a mapping that could not invert it): the node's symbols
                # are known to the adapter once a node is built on it (``know_symbols``).
                symbol = self.symbol_of(p.symbol)
                if symbol is not None:
                    p = replace(p, symbol=symbol)
                    ins = self._instrument_or_none(symbol)
            if ins is None:
                self._event("unmapped_instrument", None, json.dumps({"symbol": p.symbol, "net_qty": p.net_qty, "what": "position left as the broker counts it"}))
                out.append(p)
            else:
                out.append(replace(p, net_qty=self.units(ins, p.net_qty)))
        return out

    def balances(self) -> VenueBalances:
        return self._read(self.fetch_balances)

    def _read_orders(self) -> list[OrderStatusUpdate]:
        return [self._in_units(r) for r in self._read(self.fetch_orders)]

    def _instrument_or_none(self, symbol: str | None) -> Instrument | None:
        if symbol is None:
            return None
        try:
            return self.instrument(symbol)
        except Unsupported:
            return None

    def _in_units(self, u: OrderStatusUpdate) -> OrderStatusUpdate:
        """The one place an inbound order row goes from the broker's count to units."""
        ins = self._instrument_or_none(u.symbol)
        if ins is None or not ins.quantity_in_lots:
            return u
        return replace(
            u,
            quantity=None if u.quantity is None else self.units(ins, u.quantity),
            filled_qty=self.units(ins, u.filled_qty),
        )

    def _unmapped_status(self, order_id: Any, raw: Any, tag: Any = None) -> None:
        """A row whose status (or id) the mapping cannot read: say so, once per order and status."""
        key = (str(order_id), str(raw))
        if key in self._seen_unmapped:
            return
        self._seen_unmapped[key] = None
        while len(self._seen_unmapped) > 10_000:
            self._seen_unmapped.pop(next(iter(self._seen_unmapped)))
        detail = json.dumps({"order_id": str(order_id) if order_id else None, "status": None if raw is None else str(raw), "tag": tag})
        self._event("unmapped_status", self.ids.decode(tag) if tag else None, detail)

    def _with_cid(self, r: OrderStatusUpdate) -> OrderStatusUpdate:
        if r.client_order_id or not r.tag:
            return r
        cid = self.ids.decode(r.tag)
        return replace(r, client_order_id=cid) if cid else r

    def fence(self, epoch: int) -> None:
        """Make ``epoch`` current. Queued, unsent work of an older epoch is dropped.

        Monotonic: a lower epoch raises. **Without a ``lease`` this only guards this process**: it
        compares a number held here, so two processes that were each given an epoch both keep
        sending (``can_fence`` is ``False``). With a ``lease`` the host's check runs in this call
        (a node calls it before every request and from its heartbeat) and before every write, and a
        process that lost it is refused with :class:`~raptorbt.execution.Fenced`. Neither recalls a
        request already in flight. A higher epoch lifts a previous ``Fenced`` refusal (the host
        re-acquired the lease).
        """
        with self._lock:
            if epoch < self._epoch:
                raise AdapterError(f"epoch {epoch} is behind {self._epoch}")
            if epoch > self._epoch:
                self._fenced = False
            self._epoch = epoch
            # The node calls this before every request it sends and from its heartbeat: with a
            # lease the answer is the host's, so a node that lost it is fenced at once instead of
            # journaling a request nobody sends.
            if self._fenced or not self._holds_lease():
                raise Fenced(f"this adapter does not hold the lease for epoch {epoch}")

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def can_fence(self) -> bool:
        """True only when the host gave a ``lease``: fencing then holds across processes."""
        return self._lease is not None

    @property
    def namespace(self) -> str:
        """The tag prefix of this run (read-only). A saved state restores only under its own."""
        return self.ids.namespace

    @staticmethod
    def _lease_check(lease: Any) -> Callable[[int], bool] | None:
        if lease is None:
            return None
        holds = getattr(lease, "holds", None)
        if callable(holds):
            return holds
        if callable(lease):
            return lease
        raise InvalidParameter("lease is a callable(epoch) -> bool or an object with holds(epoch)",
                               code="invalid_parameter", operation="BrokerAdapter",
                               parameter="lease",
                               valid=["a callable(epoch) -> bool", "an object with holds(epoch)"])

    def _holds_lease(self) -> bool:
        """The host's lease check for the current epoch; fails closed (an error is a no)."""
        if self._lease is None:
            return True
        if self._fenced:
            return False
        try:
            ok = bool(self._lease(self._epoch))
        except Exception as exc:  # noqa: BLE001 - a check that fails proves nothing
            self._event("lease_check_failed", None, type(exc).__name__)
            ok = False
        if not ok:
            self._fenced = True
            self._event("fenced", None, f"lease not held at epoch {self._epoch}")
        return ok

    def _refuse_if_fenced(self) -> None:
        if self._fenced:
            raise Fenced(f"this adapter lost its lease at epoch {self._epoch}; fence() with a higher epoch after re-acquiring it")

    def _drop_fenced(self) -> None:
        """Lease lost: drop every queued write, and tell the node each one was not sent.

        A request the node handed over was journaled as taken. A dropped one gets a rejected
        report (reason ``fenced``) so the node never holds an order as working that no one sent;
        the new owner reconciles whatever was in flight before the lease was lost."""
        with self._lock:
            self._fenced = True
            keep: deque[_Job] = deque()
            for j in self._outbox:
                if j.kind == "resync":
                    keep.append(j)
                else:
                    self._event("fenced_drop", j.client_order_id, f"{j.kind} of epoch {j.epoch}: lease lost")
                    self._report_unsent(j, "fenced")
            self._outbox = keep

    def _report_unsent(self, job: _Job, why: str) -> None:
        """Answer a queued write that will never be sent, the way a refusal would."""
        rec = self._orders.get(job.client_order_id or "")
        if rec is None or rec.terminal:
            return
        if job.kind == "submit":
            if rec.attempted or rec.venue_order_id:
                return  # an earlier try may be at the venue: reconcile decides, not a rejection
            self._terminal(rec, "rejected", ExecutionReport.rejected(why, self._now()))
        elif job.kind == "cancel":
            rec.cancel_requested = False
            self._push(rec, ExecutionReport.cancel_rejected(why, self._now()))
        elif job.kind == "modify":
            rec.pending_modify = None
            self._push(rec, ExecutionReport.replace_rejected(why, self._now()))

    # -- state ---------------------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """JSON-able: id maps, fill progress, pending modifies and unsent jobs. No credentials."""
        with self._lock:
            return {
                "version": 1,
                "namespace": self.ids.namespace,
                "epoch": self._epoch,
                "registry": self.ids.export(),
                "orders": {
                    c: {
                        "tag": r.tag, "symbol": r.symbol, "side": r.side, "quantity": r.quantity,
                        "order_type": r.order_type, "limit_price": r.limit_price, "trigger_price": r.trigger_price,
                        "tif": r.tif, "product": r.product, "wire": r.wire, "venue_order_id": r.venue_order_id,
                        "accepted": r.accepted, "terminal": sorted(r.terminal), "attempted": r.attempted, "cancel_requested": r.cancel_requested,
                        "pending_modify": list(r.pending_modify) if r.pending_modify else None,
                        "cum_qty": r.cum_qty, "cum_notional": r.cum_notional, "seen_exec": sorted(r.seen_exec),
                    }  # fmt: skip
                    for c, r in self._orders.items()
                },
                "outbox": [
                    {"kind": j.kind, "client_order_id": j.client_order_id, "epoch": j.epoch, "change": j.change}
                    for j in self._outbox
                ],
            }

    def import_state(self, state: dict[str, Any]) -> None:
        """Load ``export_state()`` into a fresh adapter. Unsent jobs are queued again and run."""
        if state.get("version") != 1:
            raise AdapterError("unknown adapter state version")
        if state.get("namespace") != self.ids.namespace:
            raise AdapterError("the saved state belongs to another namespace: build the adapter with the namespace it was saved under")
        with self._lock:
            self._epoch = state["epoch"]
            self.ids.restore(state["registry"])
            for cid, d in state["orders"].items():
                fp = (d["symbol"], d["side"], d["quantity"], d["order_type"], d["limit_price"], d["trigger_price"], d["tif"], d["product"])
                rec = _Rec(
                    cid, d["tag"], d["symbol"], d["side"], d["quantity"], d["order_type"], d["limit_price"],
                    d["trigger_price"], d["tif"], d["product"], d["wire"], fp, d["venue_order_id"], d["accepted"],
                    set(d["terminal"]), d["cancel_requested"],
                    tuple(d["pending_modify"]) if d["pending_modify"] else None,
                    d["cum_qty"], d["cum_notional"], False, set(d["seen_exec"]),
                    attempted=d.get("attempted", d["venue_order_id"] is None),
                    terminal_at=self._mono() if d["terminal"] else None,
                )  # fmt: skip
                self._orders[cid] = rec
                if rec.venue_order_id:
                    self._by_venue[rec.venue_order_id] = cid
                try:
                    ins = self.instrument(rec.symbol)
                    self._by_native[self.native_key(ins)] = rec.symbol
                except Unsupported:
                    pass
            for j in state["outbox"]:
                ch = tuple(j["change"]) if j["change"] else None
                self._outbox.append(_Job(j["kind"], j["client_order_id"], j["epoch"], ch))
        self._executor.kick()

    def close(self) -> None:
        self._executor.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(orders={len(self._orders)}, session={self.session!r})"

    # -- helpers for mappings --------------------------------------------------------------------

    def wire_tif(self, request: OrderRequest) -> str:
        """``DAY`` or ``IOC`` for the orders this layer lets through."""
        return "IOC" if request.time_in_force == "ioc" else "DAY"

    def native_key(self, ins: Instrument) -> str:
        """The key broker rows are matched to instruments by."""
        return ins.key or f"{ins.exchange}:{ins.symbol}"

    def know_symbols(self, symbols: Iterable[str]) -> None:
        """Learn which broker key is which of ``symbols`` (a node calls this when it is built),
        so a position or row the broker reports under its own key is read as the engine's
        symbol even when no order has been sent yet. A symbol the mapping does not know is
        skipped here and refused when an order for it is sent."""
        for symbol in symbols:
            ins = self._instrument_or_none(symbol)
            if ins is not None:
                self._by_native[self.native_key(ins)] = symbol

    def restore_orders(self, working: Iterable[Mapping[str, Any]]) -> list[str]:
        """Teach a fresh adapter the orders a restarted node holds as working.

        A node rebuilt from its journal knows which orders are working at the venue; this
        adapter, built after the restart, does not, and without them it drops the fills of
        those orders (``ignored_update``) and refuses to cancel them. ``working`` has one
        mapping per order: ``request`` (the :class:`~raptorbt.execution.OrderRequest` with its
        current terms), ``status`` (the node's), ``filled_qty`` and ``avg_price`` (what the node
        has booked). The adapter keeps the node's fill progress, so the next row of the venue
        yields exactly the fills the node has not seen. It looks each order up in the day's
        book by its tag and terms and binds the venue's id; an order the book does not show is
        still registered (the reconciliation reports it missing). Returns the client ids
        adopted. Raises :class:`Unsupported` for an order whose symbol is not mapped: the
        adapter cannot account for it, so the restore is refused."""
        adopted: list[str] = []
        for row in working:
            req = row["request"]
            cid = req.client_order_id
            with self._lock:
                if cid in self._orders:
                    continue
            ins = self.instrument(req.symbol)
            product = req.tags.get("product", self.default_product)
            tag = self.ids.encode(cid)
            try:
                wire = self.translate(req, ins, product, tag)
            except AdapterError as exc:
                wire = {}
                self._event("restore_unsendable", cid, f"{type(exc).__name__}: {exc}")
            fp = (
                req.symbol, req.side, req.quantity, req.order_type, req.limit_price,
                req.trigger_price, req.time_in_force, product,
            )  # fmt: skip
            rec = _Rec(
                cid, tag, req.symbol, req.side, req.quantity, req.order_type, req.limit_price,
                req.trigger_price, req.time_in_force, product, wire, fp,
            )  # fmt: skip
            status = str(row.get("status", ""))
            filled = float(row.get("filled_qty") or 0.0)
            avg = row.get("avg_price")
            rec.attempted = True  # it was sent before the restart: never place it again blindly
            rec.accepted = status != "submitted"
            rec.cancel_requested = status == "pendingcancel"
            rec.cum_qty = filled
            rec.cum_notional = filled * float(avg) if filled and avg is not None else 0.0
            with self._lock:
                self._orders[cid] = rec
                self._by_native[self.native_key(ins)] = req.symbol
            found, state = self._try_lookup(rec)
            if found is not None:
                self._bind(rec, found.venue_order_id)
            else:
                self._event("restore_order_unbound", cid, f"the day's book {state} for tag {tag}")
            adopted.append(cid)
        return adopted

    def symbol_of(self, native_key: str | None) -> str | None:
        """Engine symbol for a broker-native key, as far as orders sent so far reveal it."""
        key = native_key or ""
        sym = self._by_native.get(key)
        if sym is not None or not isinstance(self._instruments, Mapping):
            return sym
        n, idx = self._native_index
        if n != len(self._instruments) + len(self._added):
            idx = {self.native_key(i): s for s, i in self._instruments.items()}
            idx.update({self.native_key(i): s for s, i in self._added.items()})
            self._native_index = (len(self._instruments) + len(self._added), idx)
        return idx.get(key)

    def units(self, ins: Instrument, wire_qty: float) -> float:
        return wire_qty * ins.lot_size if ins.quantity_in_lots else wire_qty

    def wire_qty(self, ins: Instrument, units: float) -> int:
        return int(units // ins.lot_size) if ins.quantity_in_lots else int(units)


def reports_of(rows: Iterable[QueuedReport]) -> list[tuple[str, str]]:
    """(client id, report kind) pairs, for tests and logs."""
    return [(q.client_order_id, q.report.kind) for q in rows]
