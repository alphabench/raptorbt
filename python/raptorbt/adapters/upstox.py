"""Upstox API (v2 reads, v3 order writes) mapping.

Public documentation used (nothing else):

* Place order v3: https://upstox.com/developer/api-documentation/v3/place-order/
* Modify order v3: https://upstox.com/developer/api-documentation/v3/modify-order/
* Cancel order v3: https://upstox.com/developer/api-documentation/v3/cancel-order/
* Order details and book: https://upstox.com/developer/api-documentation/get-order-details/
* Error codes are the ``UDAPI`` codes listed on those pages and on the login page.

Wire facts: order writes go to ``api-hft.upstox.com`` (``POST /v3/order/place``,
``PUT /v3/order/modify``, ``DELETE /v3/order/cancel?order_id=``) with a JSON
body and ``Authorization: Bearer``; reads use ``api.upstox.com/v2``. ``tag`` is
at most 40 characters (this mapping keeps to letters and digits, which the
page does not restrict further). Order rows carry cumulative ``filled_quantity``
and ``average_price``; the order status strings follow Upstox's status
appendix (not on the pages above; checked only against synthetic data, verify
on a live account). Order writes must leave from the app's registered static
IP (``UDAPI1154`` otherwise); the API is closed from 00:00 to 05:30 IST
(``UDAPI100074``, ``423 UDAPI100072`` for reads). A new token login revokes
the app's previous token everywhere, so refresh from one process only.

Market orders are sent with ``market_protection=-1`` (automatic); a market
order without protection is refused by Upstox. Quantity is in units, except
instruments flagged ``quantity_in_lots`` (commodities) which go as lots.

Not supported: GTC/GTD/at-open/at-close/FOK validity, ``post_only``, trailing,
if-touched and market-to-limit orders, after-market and slice orders (the
exchange freeze limit is the venue's to refuse). ``reduce_only`` follows the
base rule (refused unless ``reduce_only="best_effort"``). Upstox has no expiry
status: a lapsed day order reads as cancelled.

Rate limits: defaults 50 a second, 500 a minute, 2000 per 30 minutes for the
order endpoints (Upstox's published standard limits; confirm for your plan and
pass ``limiter=`` to change them).
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
    FILLED,
    OPEN,
    RECEIVED,
    REJECTED,
    Ambiguous,
    AuthExpired,
    MalformedRow,
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

WRITE_URL = "https://api-hft.upstox.com"
READ_URL = "https://api.upstox.com"

_STATUS = {
    "put order req received": RECEIVED,
    "validation pending": RECEIVED,
    "open pending": RECEIVED,
    "after market order req received": RECEIVED,
    "open": OPEN,
    "trigger pending": OPEN,
    "modify validation pending": OPEN,
    "modify pending": OPEN,
    "modified": OPEN,
    "not modified": OPEN,
    "cancel pending": OPEN,
    "not cancelled": OPEN,
    "complete": FILLED,
    "rejected": REJECTED,
    "cancelled": CANCELLED,
}
_PRODUCT = {"intraday": "I", "delivery": "D", "mtf": "MTF"}
_PRODUCT_BACK = {"I": "intraday", "D": "delivery", "MTF": "mtf"}
_ORDER_TYPE = {"market": "MARKET", "limit": "LIMIT", "stop_limit": "SL", "stop_market": "SL-M"}
_ORDER_TYPE_BACK = {v: k for k, v in _ORDER_TYPE.items()}
_AUTH = {"UDAPI100050", "UDAPI100067", "UDAPI100016"}


class UpstoxAdapter(BrokerAdapter):
    name = "upstox"
    order_types = frozenset(_ORDER_TYPE_BACK.values())
    time_in_forces = frozenset({"day", "ioc"})
    products = frozenset(_PRODUCT)
    tag_max_len = 40
    tag_charset = "A-Za-z0-9"
    currency = "INR"

    def __init__(
        self,
        session: Session,
        instruments: Mapping[str, Instrument] | Callable[[str], Instrument | None],
        *,
        transport: Transport = urllib_transport,
        write_url: str = WRITE_URL,
        read_url: str = READ_URL,
        **kw: Any,
    ) -> None:
        self._http, self._w, self._r = transport, write_url.rstrip("/"), read_url.rstrip("/")
        super().__init__(session, instruments, **kw)

    def default_limiter(self, sleep: Callable[[float], None]) -> RateLimiter:
        def windows() -> list[TokenBucket]:
            return [TokenBucket(50, 50), TokenBucket(500 / 60, 500), TokenBucket(2000 / 1800, 2000)]

        shared = windows()
        return RateLimiter({"orders": shared, "modify": shared, "cancel": shared, "read": windows()}, sleep=sleep)

    def native_key(self, ins: Instrument) -> str:
        return ins.key or f"{ins.exchange}:{ins.symbol}"

    # -- wire ---------------------------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.session.token()}", "Accept": "application/json"}

    @staticmethod
    def _classify(resp: HttpResponse, *, write: bool) -> Any:
        body = resp.json()
        if 200 <= resp.status < 300 and isinstance(body, dict) and body.get("status") == "success":
            return body.get("data")
        errs = (body.get("errors") if isinstance(body, dict) else None) or []
        code = (errs[0].get("errorCode") if errs and isinstance(errs[0], dict) else None) or ""
        msg = (errs[0].get("message") if errs and isinstance(errs[0], dict) else None) or f"HTTP {resp.status}"
        if resp.status == 401 or code in _AUTH:
            raise AuthExpired(f"{code} {msg}".strip())
        if resp.status == 429 or code == "UDAPI100072" or resp.status == 423:
            raise Transient(f"{code or resp.status}: {msg}")
        if resp.status >= 500:
            raise (Ambiguous if write else Transient)(f"HTTP {resp.status}: {msg}")
        if 400 <= resp.status < 500:
            raise VenueRefusal(msg, code or str(resp.status))
        raise (Ambiguous if write else Transient)(f"HTTP {resp.status}")

    def _do(self, method: str, base: str, path: str, *, write: bool, body: Any = None, params: Mapping[str, Any] | None = None) -> Any:
        return rest(
            self._http, method, base + path, headers=self._headers(), body=body, params=params, write=write,
            classify=lambda r: self._classify(r, write=write),
        )  # fmt: skip

    # -- translation ------------------------------------------------------------------------------------
    def translate(self, request: OrderRequest, ins: Instrument, product: str, tag: str) -> dict[str, Any]:
        if request.trail_offset is not None:
            raise Unsupported("upstox: trailing orders have no equivalent")
        if request.tags.get("variety", "regular") != "regular":
            raise Unsupported("upstox: only regular orders are mapped (it infers after-market itself)")
        otype = _ORDER_TYPE[request.order_type]
        wire: dict[str, Any] = {
            "quantity": self.wire_qty(ins, request.quantity),
            "product": _PRODUCT[product],
            "validity": self.wire_tif(request),
            "price": request.limit_price or 0,
            "tag": tag,
            "instrument_token": ins.key,
            "order_type": otype,
            "transaction_type": request.side.upper(),
            "disclosed_quantity": 0,
            "trigger_price": request.trigger_price or 0,
            "is_amo": False,
            "slice": False,
        }
        if not ins.key:
            raise Unsupported(f"upstox: instrument {request.symbol!r} has no instrument key")
        if otype in ("MARKET", "SL-M"):
            wire["market_protection"] = -1
        return wire

    def check_modify(self, rec: _Rec, quantity, limit_price, trigger_price) -> None:
        if rec.order_type == "market":
            raise Unsupported("upstox: a market order has nothing to change")
        if limit_price is not None and rec.order_type not in ("limit", "stop_limit"):
            raise Unsupported(f"upstox: a {rec.order_type} order has no limit price")
        if trigger_price is not None and rec.order_type not in ("stop_limit", "stop_market"):
            raise Unsupported(f"upstox: a {rec.order_type} order has no trigger price")

    # -- calls -------------------------------------------------------------------------------------------
    def place(self, rec: _Rec) -> str:
        data = self._do("POST", self._w, "/v3/order/place", write=True, body=rec.wire)
        ids = (data or {}).get("order_ids") or []
        if len(ids) != 1:
            raise Ambiguous(f"upstox: place answered with {len(ids)} order ids")
        return str(ids[0])

    def cancel_remote(self, rec: _Rec) -> None:
        self._do("DELETE", self._w, "/v3/order/cancel", write=True, params={"order_id": rec.venue_order_id})

    def modify_remote(self, rec: _Rec, quantity, limit_price, trigger_price) -> None:
        ins = self.instrument(rec.symbol)
        body: dict[str, Any] = {
            "order_id": rec.venue_order_id,
            "validity": rec.wire["validity"],
            "order_type": rec.wire["order_type"],
            "price": limit_price if limit_price is not None else (rec.limit_price or 0),
            "trigger_price": trigger_price if trigger_price is not None else (rec.trigger_price or 0),
        }
        if quantity is not None:
            body["quantity"] = self.wire_qty(ins, quantity)
        if "market_protection" in rec.wire:
            body["market_protection"] = rec.wire["market_protection"]
        self._do("PUT", self._w, "/v3/order/modify", write=True, body=body)

    # -- reads --------------------------------------------------------------------------------------------
    def parse_order_update(self, payload: Any) -> OrderStatusUpdate | None:
        """A portfolio-stream order message (decoded JSON or its text) in the order-row field names."""
        if isinstance(payload, (str, bytes)):
            try:
                payload = json.loads(payload)
            except ValueError:
                return None
        if not isinstance(payload, dict):
            return None
        if payload.get("update_type", "order") not in ("order", "order_update"):
            return None
        return self._row(payload.get("data", payload))

    def _row(self, d: Mapping[str, Any]) -> OrderStatusUpdate | None:
        status = _STATUS.get(str(d.get("status", "")).lower())
        if status is None or not d.get("order_id"):
            self._unmapped_status(d.get("order_id"), d.get("status"), d.get("tag"))
            return None
        key = d.get("instrument_token") or d.get("instrument_key") or ""
        sym = self.symbol_of(key)
        if sym is None:
            if self.ids.decode(d.get("tag")) is None:
                return None  # someone else's order, on an instrument we have not traded: its lot size is unknown
            raise MalformedRow(f"upstox: order {d.get('order_id')} carries our tag on an unmapped instrument {key!r}")
        raw = str(d.get("status", "")).lower()
        return OrderStatusUpdate(
            venue_order_id=str(d["order_id"]),
            status=status,
            tag=d.get("tag"),
            symbol=sym,
            side=str(d.get("transaction_type", "")).lower() or None,
            order_type=_ORDER_TYPE_BACK.get(d.get("order_type")),
            product=_PRODUCT_BACK.get(d.get("product")),
            quantity=num(d.get("quantity")),  # the broker's count: the base converts lots to units
            filled_qty=need_num(d, "filled_quantity", what="Upstox order"),
            avg_price=price(d.get("average_price")),
            limit_price=price(d.get("price")),
            trigger_price=price(d.get("trigger_price")),
            reason=d.get("status_message") or d.get("status_message_raw"),
            ts_ns=ist_ns(d.get("exchange_timestamp")) or ist_ns(d.get("order_timestamp")),
            modify_failed=raw == "not modified",
            cancel_failed=raw == "not cancelled",
        )

    def fetch_orders(self) -> list[OrderStatusUpdate]:
        rows = self._do("GET", self._r, "/v2/order/retrieve-all", write=False) or []
        return [r for r in (self._row(d) for d in rows) if r is not None]

    def fetch_trades(self) -> list[TradeRow]:
        rows = self._do("GET", self._r, "/v2/order/trades/get-trades-for-day", write=False) or []
        return [
            TradeRow(str(d["trade_id"]), str(d["order_id"]), need_num(d, "quantity", what="Upstox trade"), need_num(d, "average_price", what="Upstox trade"), ist_ns(d.get("exchange_timestamp")))
            for d in rows
        ]  # fmt: skip

    def fetch_positions(self) -> list[VenuePosition]:
        rows = self._do("GET", self._r, "/v2/portfolio/short-term-positions", write=False) or []
        out = []
        for d in rows:
            qty = need_num(d, "quantity", what="Upstox position")
            if qty == 0:
                continue
            key = d.get("instrument_token") or ""
            out.append(VenuePosition(self.symbol_of(key) or key or str(d.get("trading_symbol")), qty, need_num(d, "average_price", what="Upstox position"), _PRODUCT_BACK.get(d.get("product"))))
        return out

    def fetch_balances(self) -> VenueBalances:
        seg = (self._do("GET", self._r, "/v2/user/get-funds-and-margin", write=False) or {}).get("equity") or {}
        return VenueBalances(available=need_num(seg, "available_margin", what="Upstox equity margin"), currency=self.currency, used=num(seg.get("used_margin")))
