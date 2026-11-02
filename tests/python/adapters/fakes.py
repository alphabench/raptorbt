"""Synthetic broker servers over the scripted venue, for the conformance runs.

Each fake implements just the endpoints its mapping calls, answers in the
broker's documented shapes, and renders venue events the way the broker's feed
does. No network is involved.
"""

from __future__ import annotations

import json
import urllib.parse

from raptorbt.adapters import Instrument, Session
from raptorbt.adapters.conformance import VenueEvent, VenueRefused, ScriptedVenue, VenueTimeout, VenueUnavailable
from raptorbt.adapters.dhan import DhanAdapter
from raptorbt.adapters.http import HttpResponse, MaybeSent
from raptorbt.adapters.kite import KiteAdapter
from raptorbt.adapters.upstox import UpstoxAdapter

SYMS = ("A", "B")


def resp(status: int, body) -> HttpResponse:
    return HttpResponse(status, json.dumps(body).encode())


def _clock():
    n = [0]

    def tick() -> int:
        n[0] += 1
        return n[0]

    return tick


class Fake:
    """Common: run a venue call, turning the scenario's failures into HTTP outcomes."""

    def __init__(self, venue: ScriptedVenue) -> None:
        self.venue = venue
        self.calls: list[tuple[str, str]] = []

    def guard(self, fn, ok, refuse, unavailable):
        try:
            return ok(fn())
        except VenueRefused as e:
            return refuse(e.reason)
        except VenueUnavailable:
            return unavailable()
        except VenueTimeout:
            raise MaybeSent("timed out") from None


# -- Kite --------------------------------------------------------------------------------------------


class FakeKite(Fake):
    STATUS = {"received": "PUT ORDER REQ RECEIVED", "open": "OPEN", "filled": "COMPLETE", "cancelled": "CANCELLED",
              "rejected": "REJECTED", "expired": "CANCELLED"}  # fmt: skip

    def row(self, e: VenueEvent) -> dict:
        return {
            "order_id": e.venue_order_id, "status": self.STATUS[e.status], "status_message": e.reason, "tag": e.ref,
            "tradingsymbol": e.symbol, "exchange": "NSE", "transaction_type": e.side.upper(),
            "order_type": "LIMIT" if e.limit_price else "MARKET", "product": "MIS", "quantity": e.quantity,
            "filled_quantity": e.filled_qty, "average_price": e.avg_price or 0, "price": e.limit_price or 0,
            "trigger_price": e.trigger_price or 0,
        }  # fmt: skip

    def __call__(self, method, url, *, headers=None, params=None, form=None, body=None, timeout=10.0):
        path = urllib.parse.urlparse(url).path
        self.calls.append((method, path))
        err = lambda t: (lambda msg: resp(400, {"status": "error", "message": msg, "error_type": t}))  # noqa: E731
        busy = lambda: resp(429, {"status": "error", "message": "Too many requests", "error_type": "NetworkException"})  # noqa: E731
        ok = lambda data: resp(200, {"status": "success", "data": data})  # noqa: E731
        if method == "POST" and path == "/orders/regular":
            f = form
            return self.guard(
                lambda: {"order_id": self.venue.place(f["tag"], f["tradingsymbol"], f["transaction_type"].lower(), float(f["quantity"]), f.get("price"), f.get("trigger_price"))},
                ok, err("InputException"), busy,
            )  # fmt: skip
        if path.startswith("/orders/regular/"):
            vid = path.rsplit("/", 1)[1]
            if method == "DELETE":
                return self.guard(lambda: (self.venue.cancel(vid), {"order_id": vid})[1], ok, err("OrderException"), busy)
            if method == "PUT":
                f = form
                return self.guard(
                    lambda: (self.venue.modify(vid, float(f["quantity"]) if "quantity" in f else None, f.get("price"), f.get("trigger_price")), {"order_id": vid})[1],
                    ok, err("InputException"), busy,
                )  # fmt: skip
        if method == "GET" and path == "/orders":
            return ok([self.row(e) for e in self.venue.book()])
        raise AssertionError(f"unmapped {method} {path}")


# -- Upstox -------------------------------------------------------------------------------------------


class FakeUpstox(Fake):
    STATUS = {"received": "put order req received", "open": "open", "filled": "complete", "cancelled": "cancelled",
              "rejected": "rejected", "expired": "cancelled"}  # fmt: skip

    def row(self, e: VenueEvent) -> dict:
        return {
            "order_id": e.venue_order_id, "status": self.STATUS[e.status], "status_message": e.reason, "tag": e.ref,
            "instrument_token": f"NSE_EQ|{e.symbol}", "transaction_type": e.side.upper(),
            "order_type": "LIMIT" if e.limit_price else "MARKET", "product": "I", "quantity": e.quantity,
            "filled_quantity": e.filled_qty, "average_price": e.avg_price or 0, "price": e.limit_price or 0,
            "trigger_price": e.trigger_price or 0,
        }  # fmt: skip

    def __call__(self, method, url, *, headers=None, params=None, form=None, body=None, timeout=10.0):
        u = urllib.parse.urlparse(url)
        path = u.path
        self.calls.append((method, path))
        err = lambda msg: resp(400, {"status": "error", "errors": [{"errorCode": "UDAPI1052", "message": msg}]})  # noqa: E731
        busy = lambda: resp(429, {"status": "error", "errors": [{"errorCode": "UDAPI10005", "message": "rate"}]})  # noqa: E731
        ok = lambda data: resp(200, {"status": "success", "data": data})  # noqa: E731
        assert "api-hft" in u.netloc if path.startswith("/v3/") else "api-hft" not in u.netloc
        if method == "POST" and path == "/v3/order/place":
            b = body
            return self.guard(
                lambda: {"order_ids": [self.venue.place(b["tag"], b["instrument_token"].split("|")[1], b["transaction_type"].lower(), float(b["quantity"]), b["price"] or None, b["trigger_price"] or None)]},
                ok, err, busy,
            )  # fmt: skip
        if method == "DELETE" and path == "/v3/order/cancel":
            vid = params["order_id"]
            return self.guard(lambda: (self.venue.cancel(vid), {"order_id": vid})[1], ok, err, busy)
        if method == "PUT" and path == "/v3/order/modify":
            b = body
            return self.guard(
                lambda: (self.venue.modify(b["order_id"], float(b["quantity"]) if "quantity" in b else None, b["price"] or None, b["trigger_price"] or None), {"order_id": b["order_id"]})[1],
                ok, err, busy,
            )  # fmt: skip
        if method == "GET" and path == "/v2/order/retrieve-all":
            return ok([self.row(e) for e in self.venue.book()])
        raise AssertionError(f"unmapped {method} {path}")


# -- Dhan ------------------------------------------------------------------------------------------------


class FakeDhan(Fake):
    STATUS = {"received": "TRANSIT", "open": "PENDING", "filled": "TRADED", "cancelled": "CANCELLED",
              "rejected": "REJECTED", "expired": "EXPIRED"}  # fmt: skip
    IDS = {"A": "1001", "B": "1002"}

    def frame(self, e: VenueEvent) -> dict:
        return {"Type": "order_alert", "Data": {
            "Exchange": "NSE", "Segment": "E", "SecurityId": self.IDS[e.symbol], "OrderNo": e.venue_order_id,
            "CorrelationId": e.ref, "Status": self.STATUS[e.status], "TxnType": "B" if e.side == "buy" else "S",
            "OrderType": "LMT" if e.limit_price else "MKT", "Product": "I", "Quantity": e.quantity,
            "TradedQty": e.filled_qty, "AvgTradedPrice": e.avg_price or 0, "Price": e.limit_price or 0,
            "TriggerPrice": e.trigger_price or 0, "ReasonDescription": e.reason or "",
        }}  # fmt: skip

    def row(self, e: VenueEvent) -> dict:
        return {
            "orderId": e.venue_order_id, "correlationId": e.ref, "orderStatus": self.STATUS[e.status],
            "transactionType": e.side.upper(), "exchangeSegment": "NSE_EQ", "productType": "INTRADAY",
            "orderType": "LIMIT" if e.limit_price else "MARKET", "securityId": self.IDS[e.symbol],
            "quantity": e.quantity, "price": e.limit_price or 0, "triggerPrice": e.trigger_price or 0,
            "filledQty": e.filled_qty, "averageTradedPrice": e.avg_price or 0, "omsErrorDescription": e.reason or "",
        }  # fmt: skip

    def __call__(self, method, url, *, headers=None, params=None, form=None, body=None, timeout=10.0):
        path = urllib.parse.urlparse(url).path.removeprefix("/v2")
        self.calls.append((method, path))
        err = lambda msg: resp(400, {"errorType": "Order_Error", "errorCode": "DH-906", "errorMessage": msg})  # noqa: E731
        busy = lambda: resp(429, {"errorType": "Rate_Limit", "errorCode": "DH-904", "errorMessage": "Too many requests"})  # noqa: E731
        ok = lambda data: resp(200, data)  # noqa: E731
        rev = {v: k for k, v in self.IDS.items()}
        if method == "POST" and path == "/orders":
            b = body
            return self.guard(
                lambda: {"orderId": self.venue.place(b["correlationId"], rev[b["securityId"]], b["transactionType"].lower(), float(b["quantity"]), b["price"] or None, b.get("triggerPrice")), "orderStatus": "TRANSIT"},
                ok, err, busy,
            )  # fmt: skip
        if path.startswith("/orders/") and method == "DELETE":
            vid = path.rsplit("/", 1)[1]
            return self.guard(lambda: (self.venue.cancel(vid), {"orderId": vid, "orderStatus": "CANCELLED"})[1], ok, err, busy)
        if path.startswith("/orders/") and method == "PUT":
            vid, b = path.rsplit("/", 1)[1], body
            return self.guard(
                lambda: (self.venue.modify(vid, float(b["quantity"]), b.get("price"), b.get("triggerPrice")), {"orderId": vid, "orderStatus": "PENDING"})[1],
                ok, err, busy,
            )  # fmt: skip
        if method == "GET" and path == "/orders":
            return ok([self.row(e) for e in self.venue.book()])
        raise AssertionError(f"unmapped {method} {path}")


# -- bindings ----------------------------------------------------------------------------------------------


class BrokerBinding:
    """Conformance binding for a broker mapping over its synthetic server."""

    def __init__(self, venue, options, *, fake_cls, build, render):
        self.venue, self.options = venue, options
        self.fake = fake_cls(venue)
        self._build, self._render = build, render
        self.adapter = self._new()

    def _new(self):
        clock = _clock()
        return self._build(self.fake, clock, self.options)

    def ref_of(self, cid):
        return self.adapter.ids.encode(cid)

    def on_event(self, ev):
        self.adapter.handle_payload(self._render(self.fake, ev))

    def pump(self):
        self.adapter.pump()

    def resync(self):
        self.adapter.resync()
        self.adapter.pump()

    def restart(self):
        state = self.adapter.export_state()
        self.adapter.close()
        self.adapter = self._new()
        self.adapter.import_state(state)

    def drain(self):
        return [(q.client_order_id, q.report, q.arrival_ns) for q in self.adapter.drain_reports()]


def _common(clock, options):
    return dict(sleep=lambda s: None, clock_ns=clock, gtc="day", **options)


def kite_binding(venue, options):
    def build(fake, clock, opts):
        ins = {s: Instrument("NSE", s) for s in SYMS}
        return KiteAdapter(Session("tok", api_key="key"), ins, transport=fake, **_common(clock, opts))

    return BrokerBinding(venue, options, fake_cls=FakeKite, build=build, render=lambda f, ev: {"type": "order", "data": f.row(ev)})


def upstox_binding(venue, options):
    def build(fake, clock, opts):
        ins = {s: Instrument("NSE_EQ", s, key=f"NSE_EQ|{s}") for s in SYMS}
        return UpstoxAdapter(Session("tok"), ins, transport=fake, **_common(clock, opts))

    return BrokerBinding(venue, options, fake_cls=FakeUpstox, build=build, render=lambda f, ev: f.row(ev))


def dhan_binding(venue, options):
    def build(fake, clock, opts):
        ins = {s: Instrument("NSE_EQ", s, key=FakeDhan.IDS[s]) for s in SYMS}
        return DhanAdapter(Session("tok"), ins, client_id="1000000001", transport=fake, **_common(clock, opts))

    return BrokerBinding(venue, options, fake_cls=FakeDhan, build=build, render=lambda f, ev: f.frame(ev))
