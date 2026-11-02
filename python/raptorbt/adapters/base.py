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

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from raptorbt.execution import (
    AdapterError,
    ExecutionReport,
    OrderRefused,
    OrderRequest,
    Unsupported,
)

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
    "Instrument",
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

    ``filled_qty`` is cumulative and ``avg_price`` the average over it (units).
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
    """A net position at the venue. ``net_qty`` is signed (short negative), in units."""

    symbol: str
    net_qty: float
    avg_price: float
    product: str | None = None


@dataclass(frozen=True)
class VenueBalances:
    """Money at the venue, in ``currency``. ``available`` can be used for new orders now."""

    available: float
    used: float | None = None
    total: float | None = None
    currency: str = "INR"


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


@dataclass
class _Job:
    kind: str  # submit, cancel, modify, resync
    client_order_id: str | None = None
    epoch: int = 0
    change: tuple[float | None, float | None, float | None] | None = None


def _clock_ns() -> int:
    return time.time_ns()


def _same(a: float | None, b: float | None) -> bool:
    return a is None or (b is not None and abs(a - b) <= 1e-9 + 1e-9 * abs(a))


class BrokerAdapter:
    """Base of the broker mappings. See the module docs; subclasses fill the hooks.

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
        default_product: str = "intraday",
        reduce_only: str = "refuse",
        gtc: str = "refuse",
        namespace: str = "",
        account_id: str | None = None,
        fill_source: str | None = None,
    ) -> None:
        if fill_source is not None:
            if fill_source not in ("order_updates", "trades"):
                raise ValueError("fill_source is 'order_updates' or 'trades'")
            self.fill_source = fill_source
        if reduce_only not in ("refuse", "best_effort"):
            raise ValueError("reduce_only is 'refuse' or 'best_effort'")
        if gtc not in ("refuse", "day"):
            raise ValueError("gtc is 'refuse' or 'day'")
        if default_product not in self.products:
            raise ValueError(f"default_product must be one of {sorted(self.products)}")
        self.session = session
        self._instruments = instruments
        self._limiter = limiter or self.default_limiter(sleep)
        self._backoff = backoff or Backoff()
        self._sleep = sleep
        self._now = clock_ns
        self.max_attempts = max_attempts
        self.resend_if_absent = resend_if_absent
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
        self._events: deque[AdapterEvent] = deque()
        self._paused = False
        self._epoch = 0
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

    def parse_order_update(self, payload: Any) -> OrderStatusUpdate | None:
        """A feed message to an update; ``None`` for messages that are not order updates."""
        raise NotImplementedError

    def find_by_tag(self, tag: str) -> OrderStatusUpdate | None:
        for row in self._read(self.fetch_orders):
            if row.tag == tag:
                return row
        return None

    # -- the ExecutionAdapter surface (node thread) ---------------------------------

    def instrument(self, symbol: str) -> Instrument:
        ins = self._instruments.get(symbol) if isinstance(self._instruments, Mapping) else self._instruments(symbol)
        if ins is None:
            raise Unsupported(f"{self.name}: no instrument is mapped for symbol {symbol!r}")
        return ins

    def submit(self, request: OrderRequest) -> None:
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
            have = self._orders.get(request.client_order_id)
            if have is not None:
                if have.fingerprint == fp:
                    return  # the same order again: already queued or sent
                raise OrderRefused("duplicate_client_order_id: the id was used for different terms")
            self._orders[request.client_order_id] = _Rec(
                request.client_order_id, tag, request.symbol, request.side, request.quantity,
                request.order_type, request.limit_price, request.trigger_price,
                request.time_in_force, product, wire, fp,
            )  # fmt: skip
            self._by_native[self.native_key(ins)] = request.symbol
            self._outbox.append(_Job("submit", request.client_order_id, self._epoch))
        self._executor.kick()

    def cancel(self, client_order_id: str) -> None:
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
                    if not self._run(job):
                        break
                    with self._lock:
                        if self._outbox and self._outbox[0] is job:
                            self._outbox.popleft()
                    done += 1
            finally:
                self._pump_lock.release()
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
            return True
        return getattr(self, f"_do_{job.kind}")(rec, job)

    def _call(self, fn: Callable[[], Any], klass: str, *, retry_ambiguous: bool) -> Any:
        """Rate limit, call, retry on ``Transient`` (and ``Ambiguous`` when idempotent).

        Raises ``Ambiguous`` or ``Transient`` when attempts run out, and
        ``AuthExpired`` after one failed refresh; ``VenueRefusal`` passes through.
        """
        refreshed = False
        for attempt in range(self.max_attempts):
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
        try:
            vid = self._call(lambda: self.place(rec), "orders", retry_ambiguous=False)
        except VenueRefusal as e:
            self._terminal(rec, "rejected", ExecutionReport.rejected(self._reason(e), self._now()))
        except Transient as e:
            self._event("submit_failed", rec.client_order_id, str(e))
            self._terminal(rec, "rejected", ExecutionReport.rejected(f"venue unreachable: {e}", self._now()))
        except AuthExpired:
            return self._hold("submit", rec)
        except Ambiguous as e:
            self._settle_unconfirmed(rec, str(e))
        else:
            with self._lock:
                rec.venue_order_id = vid
                self._by_venue[vid] = rec.client_order_id
        return True

    def _settle_unconfirmed(self, rec: _Rec, why: str) -> None:
        """A write may have landed. Look it up; adopt it, or leave it to reconciliation."""
        try:
            found = self._read(lambda: self.find_by_tag(rec.tag))
        except AuthExpired:
            found = None
        except AdapterError:
            found = None
            why += "; lookup failed"
        if found is not None:
            self.handle_update(found)
            return
        if found is None and self.resend_if_absent and "lookup failed" not in why:
            self._event("resend", rec.client_order_id, "absent at the venue after an ambiguous failure")
            try:
                vid = self._call(lambda: self.place(rec), "orders", retry_ambiguous=False)
            except VenueRefusal as e:
                self._terminal(rec, "rejected", ExecutionReport.rejected(self._reason(e), self._now()))
                return
            except AdapterError as e:
                why = f"resend failed: {e}"
            else:
                with self._lock:
                    rec.venue_order_id = vid
                    self._by_venue[vid] = rec.client_order_id
                return
        rec.unconfirmed = True
        self._event("unconfirmed", rec.client_order_id, why)

    def _ensure_venue_id(self, rec: _Rec) -> bool:
        if rec.venue_order_id:
            return True
        try:
            found = self._read(lambda: self.find_by_tag(rec.tag))
        except AdapterError:
            return False
        if found is None:
            return False
        self.handle_update(found)
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
        except AdapterError as e:
            self._event("cancel_unconfirmed", rec.client_order_id, str(e))
        return True

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
        except AdapterError as e:
            self._event("modify_unconfirmed", rec.client_order_id, str(e))
        return True

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

    def handle_payload(self, payload: Any, *, arrival_ns: int | None = None) -> None:
        """A raw feed message (postback, websocket frame already decoded)."""
        update = self.parse_order_update(payload)
        if update is not None:
            self.handle_update(update, arrival_ns=arrival_ns)

    def handle_update(self, u: OrderStatusUpdate, *, arrival_ns: int | None = None) -> None:
        """Fold one order row into reports. Safe to repeat and to reorder."""
        with self._lock:
            rec = self._resolve(u)
            if rec is None:
                self.ignored_updates += 1
                return
            if rec.venue_order_id is None:
                rec.venue_order_id = u.venue_order_id
                self._by_venue[u.venue_order_id] = rec.client_order_id
            elif rec.venue_order_id != u.venue_order_id:
                self._event("venue_id_mismatch", rec.client_order_id, f"{u.venue_order_id} != {rec.venue_order_id}")
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
            if self.fill_source == "order_updates":
                self._fold_cumulative(rec, u, ts, arrival)
            if u.status == CANCELLED or (u.status == REJECTED and u.filled_qty > 0):
                self._terminal(rec, "cancel", ExecutionReport.cancel_ack(ts), arrival)
            elif u.status == REJECTED:
                self._terminal(rec, "rejected", ExecutionReport.rejected(u.reason or "rejected by the venue", ts), arrival)
            elif u.status == EXPIRED:
                self._terminal(rec, "expired", ExecutionReport.expired(ts), arrival)
            elif u.status == FILLED:
                rec.terminal.add("filled")

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
    ) -> None:
        """A real execution with the venue's own id (for ``fill_source="trades"``)."""
        with self._lock:
            cid = client_order_id or self._by_venue.get(venue_order_id or "")
            rec = self._orders.get(cid or "")
            if rec is None:
                self.ignored_updates += 1
                return
            if exec_id in rec.seen_exec:
                return
            rec.seen_exec.add(exec_id)
            rec.trade_fed = True
            arrival = arrival_ns if arrival_ns is not None else self._now()
            self._push(rec, ExecutionReport.fill(exec_id, qty, price, ts_ns or arrival, fee=fee), arrival)

    def _resolve(self, u: OrderStatusUpdate) -> _Rec | None:
        cid = u.client_order_id or self.ids.decode(u.tag) or self._by_venue.get(u.venue_order_id)
        return self._orders.get(cid) if cid else None

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
        exec_id = f"{u.venue_order_id}:{u.filled_qty:g}"
        rec.seen_exec.add(exec_id)
        self._push(rec, ExecutionReport.fill(exec_id, delta, px, ts), arrival)

    def _terminal(self, rec: _Rec, kind: str, report: ExecutionReport, arrival: int | None = None) -> None:
        with self._lock:
            if kind in rec.terminal:
                return
            rec.terminal.add(kind)
            self._push(rec, report, arrival)

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
        for row in self._read(self.fetch_orders):
            self.handle_update(row)
        if self.fill_source == "trades":
            self.sync_trades()

    def sync_trades(self) -> None:
        """Read the trade book and book every execution not yet seen (``fill_source="trades"``)."""
        for t in self._read(self.fetch_trades):
            self.handle_trade(exec_id=t.exec_id, qty=t.qty, price=t.price, venue_order_id=t.venue_order_id, fee=t.fee, ts_ns=t.ts_ns)

    # -- reconcile reads and fencing (the journal work uses these) ---------------------------

    def orders(self) -> list[VenueOrder]:
        """The venue's order book for the day, normalised; ``client_order_id`` set where the tag is ours."""
        return [self._with_cid(r) for r in self._read(self.fetch_orders)]

    def open_orders(self) -> list[VenueOrder]:
        """``orders()`` that are not finished."""
        return [o for o in self.orders() if o.status not in _TERMINAL]

    def positions(self) -> list[VenuePosition]:
        return self._read(self.fetch_positions)

    def balances(self) -> VenueBalances:
        return self._read(self.fetch_balances)

    def _with_cid(self, r: OrderStatusUpdate) -> OrderStatusUpdate:
        if r.client_order_id or not r.tag:
            return r
        cid = self.ids.decode(r.tag)
        return replace(r, client_order_id=cid) if cid else r

    def fence(self, epoch: int) -> None:
        """Make ``epoch`` current. Queued, unsent work of an older epoch is dropped.

        Monotonic: a lower epoch raises. This stops *this process* sending
        stale work; it cannot recall a request already in flight, and does not
        stop another process (that is the lease's job).
        """
        with self._lock:
            if epoch < self._epoch:
                raise AdapterError(f"epoch {epoch} is behind {self._epoch}")
            self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    # -- state ---------------------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """JSON-able: id maps, fill progress, pending modifies and unsent jobs. No credentials."""
        with self._lock:
            return {
                "version": 1,
                "epoch": self._epoch,
                "registry": self.ids.export(),
                "orders": {
                    c: {
                        "tag": r.tag, "symbol": r.symbol, "side": r.side, "quantity": r.quantity,
                        "order_type": r.order_type, "limit_price": r.limit_price, "trigger_price": r.trigger_price,
                        "tif": r.tif, "product": r.product, "wire": r.wire, "venue_order_id": r.venue_order_id,
                        "accepted": r.accepted, "terminal": sorted(r.terminal), "cancel_requested": r.cancel_requested,
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

    def symbol_of(self, native_key: str | None) -> str | None:
        """Engine symbol for a broker-native key, as far as orders sent so far reveal it."""
        return self._by_native.get(native_key or "")

    def units(self, ins: Instrument, wire_qty: float) -> float:
        return wire_qty * ins.lot_size if ins.quantity_in_lots else wire_qty

    def wire_qty(self, ins: Instrument, units: float) -> int:
        return int(units // ins.lot_size) if ins.quantity_in_lots else int(units)


def reports_of(rows: Iterable[QueuedReport]) -> list[tuple[str, str]]:
    """(client id, report kind) pairs, for tests and logs."""
    return [(q.client_order_id, q.report.kind) for q in rows]
