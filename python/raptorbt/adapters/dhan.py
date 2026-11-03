"""DhanHQ v2 mapping.

Public documentation used (nothing else):

* Orders: https://dhanhq.co/docs/v2/orders/
* Live order update (websocket): https://dhanhq.co/docs/v2/order-update/
* Annexure (segments, products, statuses, ``DH-9xx`` errors): https://dhanhq.co/docs/v2/annexure/

Wire facts: ``POST /orders``, ``PUT /orders/{id}``, ``DELETE /orders/{id}``,
JSON bodies carrying ``dhanClientId``, header ``access-token``; the order's
tracking id is ``correlationId`` (30 characters, letters, digits, space, ``_``,
``-``); the live feed carries cumulative ``TradedQty`` and ``AvgTradedPrice``.
Order writes need the account's registered static IP. Error bodies are
``{"errorType", "errorCode": "DH-9xx", "errorMessage"}``: ``DH-901`` is an
invalid or expired token, ``DH-904`` rate limiting, ``DH-905``/``906``/``903``/
``902`` refusals, ``DH-908``/``909``/``910`` and any 5xx are treated as "may have
been processed". Order book field names ``filledQty`` and ``averageTradedPrice``
(and the trade book's) are not on the pages above; the mapping accepts the
spellings used in Dhan's published examples and is checked on synthetic data
only: verify against a live account. A token minted with a TOTP login should be
requested early in a 30-second step; the refresh is the caller's
(``Session(refresh=...)``).

Segments: the instrument's ``exchange`` is Dhan's ``exchangeSegment`` enum
(``NSE_EQ``, ``NSE_FNO``, ``BSE_EQ``, ``MCX_COMM``...) and ``key`` its
``securityId``.

Not supported: GTC/GTD/at-open/at-close/FOK validity, ``post_only``, trailing,
if-touched and market-to-limit orders, CO/BO/super and after-market orders,
slicing. ``reduce_only`` follows the base rule. Modification is limited by
Dhan (25 per order); a refusal comes back as a rejected replace.

Rate limits: defaults 25 a second, 250 a minute, 1000 an hour for order calls;
override with ``limiter=``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from raptorbt.execution import OrderRequest, Unsupported

from ._limits import RateLimiter, TokenBucket
from ._rest import ist_ns, need_num, num, price, rest
from .base import (
    CANCELLED,
    EXPIRED,
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

BASE_URL = "https://api.dhan.co/v2"

_STATUS = {
    "TRANSIT": RECEIVED,
    "PENDING": OPEN,
    "PART_TRADED": OPEN,
    "CONFIRM": OPEN,
    "TRADED": FILLED,
    "REJECTED": REJECTED,
    "CANCELLED": CANCELLED,
    "EXPIRED": EXPIRED,
}
_PRODUCT = {"intraday": "INTRADAY", "delivery": "CNC", "margin": "MARGIN", "mtf": "MTF"}
_PRODUCT_BACK = {v: k for k, v in _PRODUCT.items()} | {"C": "delivery", "I": "intraday", "M": "margin", "F": "mtf"}
_ORDER_TYPE = {"market": "MARKET", "limit": "LIMIT", "stop_limit": "STOP_LOSS", "stop_market": "STOP_LOSS_MARKET"}
_ORDER_TYPE_BACK = {v: k for k, v in _ORDER_TYPE.items()} | {"LMT": "limit", "MKT": "market", "SL": "stop_limit", "SLM": "stop_market"}
_SEGMENT = {  # websocket Exchange + Segment letters to the REST enum (letters beyond E are inferred from the enum names)
    ("NSE", "E"): "NSE_EQ", ("NSE", "D"): "NSE_FNO", ("NSE", "C"): "NSE_CURRENCY",
    ("BSE", "E"): "BSE_EQ", ("BSE", "D"): "BSE_FNO", ("BSE", "C"): "BSE_CURRENCY", ("MCX", "M"): "MCX_COMM",
}  # fmt: skip
_REFUSAL_CODES = {"DH-902", "DH-903", "DH-905", "DH-906"}
_MAYBE_CODES = {"DH-908", "DH-909", "DH-910"}


class DhanAdapter(BrokerAdapter):
    name = "dhan"
    order_types = frozenset({"market", "limit", "stop_limit", "stop_market"})
    time_in_forces = frozenset({"day", "ioc"})
    products = frozenset(_PRODUCT)
    tag_max_len = 30
    tag_charset = "A-Za-z0-9_-"
    currency = "INR"

    def __init__(
        self,
        session: Session,
        instruments: Mapping[str, Instrument] | Callable[[str], Instrument | None],
        *,
        client_id: str,
        transport: Transport = urllib_transport,
        base_url: str = BASE_URL,
        **kw: Any,
    ) -> None:
        self._client_id = client_id
        self._http, self._base = transport, base_url.rstrip("/")
        super().__init__(session, instruments, **kw)

    def default_limiter(self, sleep: Callable[[float], None]) -> RateLimiter:
        shared = [TokenBucket(25, 25), TokenBucket(250 / 60, 250), TokenBucket(1000 / 3600, 1000)]
        return RateLimiter({"orders": shared, "modify": shared, "cancel": shared, "read": [TokenBucket(20, 20)]}, sleep=sleep)

    def native_key(self, ins: Instrument) -> str:
        return f"{ins.exchange}:{ins.key}"

    # -- wire ------------------------------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"access-token": self.session.token(), "Accept": "application/json"}

    @staticmethod
    def _classify(resp: HttpResponse, *, write: bool) -> Any:
        body = resp.json()
        if 200 <= resp.status < 300 and body is not None and not (isinstance(body, dict) and "errorCode" in body):
            return body
        code = (body.get("errorCode") if isinstance(body, dict) else None) or ""
        msg = (body.get("errorMessage") if isinstance(body, dict) else None) or f"HTTP {resp.status}"
        if code == "DH-901" or resp.status == 401:
            raise AuthExpired(msg)
        if code == "DH-904" or resp.status == 429:
            raise Transient("rate limited")
        if code in _REFUSAL_CODES:
            raise VenueRefusal(msg, code)
        if code in _MAYBE_CODES or resp.status >= 500:
            raise (Ambiguous if write else Transient)(f"{code or resp.status}: {msg}")
        if 400 <= resp.status < 500:
            raise VenueRefusal(msg, code or str(resp.status))
        raise (Ambiguous if write else Transient)(f"HTTP {resp.status}")

    def _do(self, method: str, path: str, *, write: bool, body: Any = None) -> Any:
        return rest(
            self._http, method, self._base + path, headers=self._headers(), body=body, write=write,
            classify=lambda r: self._classify(r, write=write),
        )  # fmt: skip

    # -- translation ---------------------------------------------------------------------------------------
    def translate(self, request: OrderRequest, ins: Instrument, product: str, tag: str) -> dict[str, Any]:
        if request.trail_offset is not None:
            raise Unsupported("dhan: trailing orders have no equivalent")
        if request.tags.get("variety", "regular") != "regular":
            raise Unsupported("dhan: only regular orders are mapped (no after-market, CO, BO or super orders)")
        if not ins.key:
            raise Unsupported(f"dhan: instrument {request.symbol!r} has no securityId")
        wire: dict[str, Any] = {
            "dhanClientId": self._client_id,
            "correlationId": tag,
            "transactionType": request.side.upper(),
            "exchangeSegment": ins.exchange,
            "productType": _PRODUCT[product],
            "orderType": _ORDER_TYPE[request.order_type],
            "validity": self.wire_tif(request),
            "securityId": ins.key,
            "quantity": self.wire_qty(ins, request.quantity),
            "price": request.limit_price or 0,
        }
        if request.trigger_price is not None:
            wire["triggerPrice"] = request.trigger_price
        return wire

    def check_modify(self, rec: _Rec, quantity, limit_price, trigger_price) -> None:
        if rec.order_type == "market":
            raise Unsupported("dhan: a market order has nothing to change")
        if limit_price is not None and rec.order_type not in ("limit", "stop_limit"):
            raise Unsupported(f"dhan: a {rec.order_type} order has no limit price")
        if trigger_price is not None and rec.order_type not in ("stop_limit", "stop_market"):
            raise Unsupported(f"dhan: a {rec.order_type} order has no trigger price")

    # -- calls ----------------------------------------------------------------------------------------------
    def place(self, rec: _Rec) -> str:
        data = self._do("POST", "/orders", write=True, body=rec.wire)
        oid = data.get("orderId") if isinstance(data, dict) else None
        if not oid:
            raise Ambiguous("dhan: place answered without an order id")
        return str(oid)

    def cancel_remote(self, rec: _Rec) -> None:
        self._do("DELETE", f"/orders/{rec.venue_order_id}", write=True)

    def modify_remote(self, rec: _Rec, quantity, limit_price, trigger_price) -> None:
        ins = self.instrument(rec.symbol)
        body: dict[str, Any] = {
            "dhanClientId": self._client_id,
            "orderId": rec.venue_order_id,
            "orderType": rec.wire["orderType"],
            "validity": rec.wire["validity"],
            "quantity": self.wire_qty(ins, quantity if quantity is not None else rec.quantity),
        }
        lim = limit_price if limit_price is not None else rec.limit_price
        trg = trigger_price if trigger_price is not None else rec.trigger_price
        if lim is not None:
            body["price"] = lim
        if trg is not None:
            body["triggerPrice"] = trg
        self._do("PUT", f"/orders/{rec.venue_order_id}", write=True, body=body)

    # -- reads -----------------------------------------------------------------------------------------------
    def parse_order_update(self, payload: Any) -> OrderStatusUpdate | None:
        """A live-order-update frame ``{"Data": {...}, "Type": "order_alert"}``."""
        if isinstance(payload, (str, bytes)):
            try:
                payload = json.loads(payload)
            except ValueError:
                return None
        if not isinstance(payload, dict) or payload.get("Type") != "order_alert":
            return None
        d = payload.get("Data") or {}
        status = _STATUS.get(str(d.get("Status", "")).upper())
        oid = d.get("OrderNo") or d.get("orderNo")
        if status is None or not oid:
            self._unmapped_status(oid, d.get("Status"), d.get("CorrelationId"))
            return None
        seg = _SEGMENT.get((str(d.get("Exchange", "")).upper(), str(d.get("Segment", "")).upper()))
        key = f"{seg}:{d.get('SecurityId')}"
        side = {"B": "buy", "S": "sell"}.get(str(d.get("TxnType", "")).upper())
        return OrderStatusUpdate(
            venue_order_id=str(oid),
            status=status,
            tag=(d.get("CorrelationId") or None),
            symbol=self.symbol_of(key) or key,
            side=side,
            order_type=_ORDER_TYPE_BACK.get(d.get("OrderType")),
            product=_PRODUCT_BACK.get(d.get("Product")),
            quantity=num(d.get("Quantity")),
            filled_qty=need_num(d, "TradedQty", what="Dhan order update"),
            avg_price=price(d.get("AvgTradedPrice")),
            limit_price=price(d.get("Price")),
            trigger_price=price(d.get("TriggerPrice")),
            reason=d.get("ReasonDescription") or None,
            ts_ns=ist_ns(d.get("LastUpdatedTime")) or ist_ns(d.get("OrderDateTime")),
        )

    def _book_row(self, d: Mapping[str, Any]) -> OrderStatusUpdate | None:
        status = _STATUS.get(str(d.get("orderStatus", "")).upper())
        if status is None or not d.get("orderId"):
            self._unmapped_status(d.get("orderId"), d.get("orderStatus"), d.get("correlationId"))
            return None
        key = f"{d.get('exchangeSegment')}:{d.get('securityId')}"
        return OrderStatusUpdate(
            venue_order_id=str(d["orderId"]),
            status=status,
            tag=d.get("correlationId") or None,
            symbol=self.symbol_of(key) or key,
            side=str(d.get("transactionType", "")).lower() or None,
            order_type=_ORDER_TYPE_BACK.get(d.get("orderType")),
            product=_PRODUCT_BACK.get(d.get("productType")),
            quantity=num(d.get("quantity")),
            filled_qty=need_num(d, "filledQty", "tradedQuantity", what="Dhan order"),
            avg_price=price(d.get("averageTradedPrice")),
            limit_price=price(d.get("price")),
            trigger_price=price(d.get("triggerPrice")),
            reason=d.get("omsErrorDescription") or None,
            ts_ns=ist_ns(d.get("updateTime")) or ist_ns(d.get("createTime")),
        )

    def fetch_orders(self) -> list[OrderStatusUpdate]:
        rows = self._do("GET", "/orders", write=False) or []
        return [r for r in (self._book_row(d) for d in rows) if r is not None]

    def fetch_trades(self) -> list[TradeRow]:
        rows = self._do("GET", "/trades", write=False) or []
        return [
            TradeRow(str(d["exchangeTradeId"]), str(d["orderId"]), need_num(d, "tradedQuantity", what="Dhan trade"), need_num(d, "tradedPrice", what="Dhan trade"), ist_ns(d.get("updateTime")) or ist_ns(d.get("createTime")))
            for d in rows
        ]  # fmt: skip

    def fetch_positions(self) -> list[VenuePosition]:
        rows = self._do("GET", "/positions", write=False) or []
        out = []
        for d in rows:
            qty = need_num(d, "netQty", what="Dhan position")
            if qty == 0:
                continue
            key = f"{d.get('exchangeSegment')}:{d.get('securityId')}"
            avg = need_num(d, "buyAvg" if qty > 0 else "sellAvg", what="Dhan position")
            out.append(VenuePosition(self.symbol_of(key) or key, qty, avg, _PRODUCT_BACK.get(d.get("productType"))))
        return out

    def fetch_balances(self) -> VenueBalances:
        d = self._do("GET", "/fundlimit", write=False) or {}
        avail = need_num(d, "availabelBalance", "availableBalance", what="Dhan fund limit")  # Dhan spells it "availabel"
        return VenueBalances(available=avail, currency=self.currency, used=num(d.get("utilizedAmount")), total=num(d.get("sodLimit")))
