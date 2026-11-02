"""Zerodha Kite Connect (v3) mapping.

Public documentation used (nothing else):

* Orders: https://kite.trade/docs/connect/v3/orders/
* Postbacks and order updates: https://kite.trade/docs/connect/v3/postbacks/
* Errors and limits: https://kite.trade/docs/connect/v3/exceptions/
* Positions and margins: https://kite.trade/docs/connect/v3/portfolio/ and /user/

Wire facts this module relies on: ``POST/PUT/DELETE /orders/{variety}[/{order_id}]``
form-encoded, ``Authorization: token api_key:access_token`` and
``X-Kite-Version: 3``; tags are alphanumeric, 20 characters; order updates carry
the *cumulative* ``filled_quantity`` and ``average_price`` and no execution id,
so fills are derived from the cumulative figures; ``TokenException`` (403) means
log in again; HTTP 429 is rate limiting; 10 orders a second, 400 a minute, 25
modifications per order.

Not supported (``Unsupported``, never silent): GTC/GTD/at-open/at-close/FOK
validity, ``post_only``, trailing, if-touched and market-to-limit orders, and
the CO, iceberg and auction varieties. ``reduce_only`` is refused unless the
adapter is built with ``reduce_only="best_effort"`` (Kite has no such flag).
Kite has no expiry status: a day order that lapses reads as cancelled.

Products: ``intraday`` MIS, ``delivery`` CNC, ``margin`` NRML, ``mtf`` MTF; set
per order with the tag ``product`` or for all with ``default_product``. A tag
``variety="amo"`` sends an after-market order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from raptorbt.execution import OrderRequest, Unsupported

from ._limits import RateLimiter, TokenBucket
from ._rest import ist_ns, num, price, rest
from .base import (
    CANCELLED,
    FILLED,
    OPEN,
    RECEIVED,
    REJECTED,
    Ambiguous,
    AuthExpired,
    BrokerAdapter,
    Instrument,
    OrderStatusUpdate,
    Session,
    TradeRow,
    Transient,
    VenueBalances,
    VenuePosition,
    VenueRefusal,
    _Rec,
)
from .http import HttpResponse, Transport, urllib_transport

BASE_URL = "https://api.kite.trade"

_STATUS = {
    "COMPLETE": FILLED,
    "REJECTED": REJECTED,
    "CANCELLED": CANCELLED,
    "CANCELLED AMO": CANCELLED,
    "OPEN": OPEN,
    "TRIGGER PENDING": OPEN,
    "MODIFIED": OPEN,
    "UPDATE": OPEN,
    "MODIFY PENDING": OPEN,
    "MODIFY VALIDATION PENDING": OPEN,
    "CANCEL PENDING": OPEN,
    "PUT ORDER REQ RECEIVED": RECEIVED,
    "VALIDATION PENDING": RECEIVED,
    "OPEN PENDING": RECEIVED,
    "AMO REQ RECEIVED": RECEIVED,
}
_PRODUCT = {"intraday": "MIS", "delivery": "CNC", "margin": "NRML", "mtf": "MTF"}
_PRODUCT_BACK = {v: k for k, v in _PRODUCT.items()}
_ORDER_TYPE = {"market": "MARKET", "limit": "LIMIT", "stop_limit": "SL", "stop_market": "SL-M"}
_ORDER_TYPE_BACK = {v: k for k, v in _ORDER_TYPE.items()}
# error_type values the docs list. These mean the OMS did not take the request.
_REFUSALS = {"InputException", "OrderException", "MarginException", "HoldingException", "UserException"}
# These can mean the OMS did, or might have, processed it.
_MAYBE = {"NetworkException", "DataException", "GeneralException"}


def verify_postback(payload: Mapping[str, Any], api_secret: str) -> bool:
    """Check a postback's ``checksum``: SHA-256 of ``order_id + order_timestamp + api_secret``.

    Call it on the raw body's decoded JSON before ``handle_payload``. The secret
    is used here and not kept.
    """
    raw = f"{payload.get('order_id', '')}{payload.get('order_timestamp', '')}{api_secret}"
    return hashlib.sha256(raw.encode()).hexdigest() == payload.get("checksum")


class KiteAdapter(BrokerAdapter):
    name = "kite"
    order_types = frozenset(_ORDER_TYPE_BACK.values())
    time_in_forces = frozenset({"day", "ioc"})
    products = frozenset(_PRODUCT)
    tag_max_len = 20
    tag_charset = "A-Za-z0-9"

    def __init__(
        self,
        session: Session,
        instruments: Mapping[str, Instrument] | Callable[[str], Instrument | None],
        *,
        transport: Transport = urllib_transport,
        base_url: str = BASE_URL,
        margin_segment: str = "equity",
        **kw: Any,
    ) -> None:
        if session.api_key() is None:
            raise ValueError("Kite needs the api_key on the Session")
        self._http, self._base, self._segment = transport, base_url.rstrip("/"), margin_segment
        super().__init__(session, instruments, **kw)

    def default_limiter(self, sleep: Callable[[float], None]) -> RateLimiter:
        per_s, per_min = TokenBucket(10, 10), TokenBucket(400 / 60, 400)
        return RateLimiter({"orders": [per_s, per_min], "modify": [per_s, per_min], "cancel": [per_s, per_min], "read": [TokenBucket(10, 10)]}, sleep=sleep)

    # -- wire ----------------------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"X-Kite-Version": "3", "Authorization": f"token {self.session.api_key()}:{self.session.token()}"}

    @staticmethod
    def _classify(resp: HttpResponse, *, write: bool) -> Any:
        body = resp.json()
        if resp.status == 200 and isinstance(body, dict) and body.get("status") == "success":
            return body.get("data")
        et = body.get("error_type") if isinstance(body, dict) else None
        msg = (body.get("message") if isinstance(body, dict) else None) or f"HTTP {resp.status}"
        if resp.status == 403 and et == "TokenException":
            raise AuthExpired(msg)
        if resp.status == 429:
            raise Transient("rate limited")
        if et in _REFUSALS:
            raise VenueRefusal(msg, et)
        if et in _MAYBE or resp.status >= 500:
            raise (Ambiguous if write else Transient)(f"{et or resp.status}: {msg}")
        if 400 <= resp.status < 500:
            raise VenueRefusal(msg, et or str(resp.status))
        raise (Ambiguous if write else Transient)(f"HTTP {resp.status}")

    def _do(self, method: str, path: str, *, write: bool, form: Mapping[str, Any] | None = None) -> Any:
        return rest(
            self._http, method, self._base + path, headers=self._headers(), form=form, write=write,
            classify=lambda r: self._classify(r, write=write),
        )  # fmt: skip

    # -- translation ------------------------------------------------------------------------------
    def translate(self, request: OrderRequest, ins: Instrument, product: str, tag: str) -> dict[str, Any]:
        variety = request.tags.get("variety", "regular")
        if variety not in ("regular", "amo"):
            raise Unsupported(f"kite: variety {variety!r} is not supported (regular, amo)")
        if request.trail_offset is not None:
            raise Unsupported("kite: trailing orders have no equivalent")
        wire: dict[str, Any] = {
            "tradingsymbol": ins.symbol,
            "exchange": ins.exchange,
            "transaction_type": request.side.upper(),
            "order_type": _ORDER_TYPE[request.order_type],
            "quantity": self.wire_qty(ins, request.quantity),
            "product": _PRODUCT[product],
            "validity": self.wire_tif(request),
            "tag": tag,
        }
        if request.limit_price is not None:
            wire["price"] = request.limit_price
        if request.trigger_price is not None:
            wire["trigger_price"] = request.trigger_price
        wire["_variety"] = variety  # not sent: part of the path
        return wire

    def check_modify(self, rec: _Rec, quantity, limit_price, trigger_price) -> None:
        if rec.order_type == "market" and (limit_price is not None or trigger_price is not None):
            raise Unsupported("kite: a market order has no price to change")
        if limit_price is not None and rec.order_type not in ("limit", "stop_limit"):
            raise Unsupported(f"kite: a {rec.order_type} order has no limit price")
        if trigger_price is not None and rec.order_type not in ("stop_limit", "stop_market"):
            raise Unsupported(f"kite: a {rec.order_type} order has no trigger price")

    # -- calls ---------------------------------------------------------------------------------------
    def place(self, rec: _Rec) -> str:
        form = {k: v for k, v in rec.wire.items() if not k.startswith("_")}
        data = self._do("POST", f"/orders/{rec.wire['_variety']}", write=True, form=form)
        if not isinstance(data, dict) or "order_id" not in data:
            raise Ambiguous("kite: place answered without an order id")
        return str(data["order_id"])

    def cancel_remote(self, rec: _Rec) -> None:
        self._do("DELETE", f"/orders/{rec.wire['_variety']}/{rec.venue_order_id}", write=True)

    def modify_remote(self, rec: _Rec, quantity, limit_price, trigger_price) -> None:
        ins = self.instrument(rec.symbol)
        form: dict[str, Any] = {}
        if quantity is not None:
            form["quantity"] = self.wire_qty(ins, quantity)
        if limit_price is not None:
            form["price"] = limit_price
        if trigger_price is not None:
            form["trigger_price"] = trigger_price
        self._do("PUT", f"/orders/{rec.wire['_variety']}/{rec.venue_order_id}", write=True, form=form)

    # -- reads ------------------------------------------------------------------------------------------
    def parse_order_update(self, payload: Any) -> OrderStatusUpdate | None:
        """A postback body, or a websocket text frame ``{"type": "order", "data": {...}}``."""
        if isinstance(payload, (str, bytes)):
            try:
                payload = json.loads(payload)
            except ValueError:
                return None
        if not isinstance(payload, dict):
            return None
        if "type" in payload:
            if payload.get("type") != "order":
                return None
            payload = payload.get("data") or {}
        return self._row(payload)

    def _row(self, d: Mapping[str, Any]) -> OrderStatusUpdate | None:
        status = _STATUS.get(str(d.get("status", "")).upper())
        if status is None or not d.get("order_id"):
            return None
        key = f"{d.get('exchange')}:{d.get('tradingsymbol')}"
        return OrderStatusUpdate(
            venue_order_id=str(d["order_id"]),
            status=status,
            tag=d.get("tag") or (d.get("tags") or [None])[0],
            symbol=self.symbol_of(key) or key,
            side=str(d.get("transaction_type", "")).lower() or None,
            order_type=_ORDER_TYPE_BACK.get(d.get("order_type")),
            product=_PRODUCT_BACK.get(d.get("product")),
            quantity=num(d.get("quantity")),
            filled_qty=num(d.get("filled_quantity"), 0.0) or 0.0,
            avg_price=price(d.get("average_price")),
            limit_price=price(d.get("price")),
            trigger_price=price(d.get("trigger_price")),
            reason=d.get("status_message") or d.get("status_message_raw"),
            ts_ns=ist_ns(d.get("exchange_update_timestamp")) or ist_ns(d.get("exchange_timestamp")) or ist_ns(d.get("order_timestamp")),
        )

    def fetch_orders(self) -> list[OrderStatusUpdate]:
        rows = self._do("GET", "/orders", write=False) or []
        return [r for r in (self._row(d) for d in rows) if r is not None]

    def fetch_trades(self) -> list[TradeRow]:
        rows = self._do("GET", "/trades", write=False) or []
        return [
            TradeRow(str(d["trade_id"]), str(d["order_id"]), num(d["quantity"], 0.0) or 0.0, num(d["average_price"], 0.0) or 0.0, ist_ns(d.get("fill_timestamp")))
            for d in rows
        ]  # fmt: skip

    def fetch_positions(self) -> list[VenuePosition]:
        data = self._do("GET", "/portfolio/positions", write=False) or {}
        out = []
        for d in data.get("net", []):
            qty = num(d.get("quantity"), 0.0) or 0.0
            if qty == 0:
                continue
            key = f"{d.get('exchange')}:{d.get('tradingsymbol')}"
            out.append(VenuePosition(self.symbol_of(key) or key, qty, num(d.get("average_price"), 0.0) or 0.0, _PRODUCT_BACK.get(d.get("product"))))
        return out

    def fetch_balances(self) -> VenueBalances:
        data = (self._do("GET", "/user/margins", write=False) or {}).get(self._segment) or {}
        avail = data.get("available") or {}
        used = (data.get("utilised") or {}).get("debits")
        a = num(avail.get("live_balance"))
        return VenueBalances(available=a if a is not None else (num(data.get("net"), 0.0) or 0.0), used=num(used), total=num(data.get("net")))
