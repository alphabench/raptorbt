"""Broker mappings on recorded-shape payloads: translation, errors, rows, reads. No network."""

import hashlib
import json

import pytest

from raptorbt.adapters import Instrument, Session
from raptorbt.adapters.dhan import DhanAdapter
from raptorbt.adapters.http import HttpResponse, MaybeSent, NotSent
from raptorbt.adapters.kite import KiteAdapter, verify_postback
from raptorbt.adapters.upstox import UpstoxAdapter
from raptorbt.execution import OrderRefused, Unsupported

from .test_execution_request import request


class Script:
    """A transport that answers from a list and records what it was sent."""

    def __init__(self, *answers):
        self.answers, self.sent = list(answers), []

    def __call__(self, method, url, *, headers=None, params=None, form=None, body=None, timeout=10.0):
        self.sent.append(dict(method=method, url=url, headers=dict(headers or {}), params=params, form=form, body=body))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        status, payload = a
        return HttpResponse(status, json.dumps(payload).encode())


def j(status, payload):
    return (status, payload)


def kite(*answers, **kw):
    t = Script(*answers)
    kw.setdefault("gtc", "day")
    a = KiteAdapter(Session("ACCESS", api_key="KEY"), {"A": Instrument("NFO", "NIFTY25NOVFUT", lot_size=25.0), "B": Instrument("NSE", "INFY")}, transport=t, sleep=lambda s: None, **kw)
    return a, t


def upstox(*answers, **kw):
    t = Script(*answers)
    kw.setdefault("gtc", "day")
    ins = {"A": Instrument("NSE_EQ", "INFY", key="NSE_EQ|INE009A01021"), "B": Instrument("MCX_FO", "CRUDE", key="MCX_FO|123", lot_size=100.0, quantity_in_lots=True)}
    return UpstoxAdapter(Session("ACCESS"), ins, transport=t, sleep=lambda s: None, **kw), t


def dhan(*answers, **kw):
    t = Script(*answers)
    kw.setdefault("gtc", "day")
    ins = {"A": Instrument("NSE_EQ", "INFY", key="1594"), "B": Instrument("NSE_FNO", "NIFTY", key="35003")}
    return DhanAdapter(Session("ACCESS"), ins, client_id="1000", transport=t, sleep=lambda s: None, **kw), t


# -- Kite --------------------------------------------------------------------------------------------


def test_kite_place_translates_and_authenticates():
    a, t = kite(j(200, {"status": "success", "data": {"order_id": "2211"}}))
    a.submit(request(symbol="B", quantity=10.0, price=99.0))
    (s,) = t.sent
    assert (s["method"], s["url"]) == ("POST", "https://api.kite.trade/orders/regular")
    assert s["headers"] == {"X-Kite-Version": "3", "Authorization": "token KEY:ACCESS"}
    assert s["form"] == {
        "tradingsymbol": "INFY", "exchange": "NSE", "transaction_type": "BUY", "order_type": "LIMIT",
        "quantity": 10, "product": "MIS", "validity": "DAY", "tag": "s0x0", "price": 99.0,
    }  # fmt: skip


@pytest.mark.parametrize(
    "kw,order_type,extra",
    [
        ({"order_type": "market"}, "MARKET", {}),
        ({"order_type": "stop_market", "trigger": 95.0}, "SL-M", {"trigger_price": 95.0}),
        ({"order_type": "stop_limit", "trigger": 95.0, "price": 94.0}, "SL", {"trigger_price": 95.0, "price": 94.0}),
    ],
)
def test_kite_order_types(kw, order_type, extra):
    a, t = kite(j(200, {"status": "success", "data": {"order_id": "1"}}))
    a.submit(request(symbol="B", **kw))
    assert t.sent[0]["form"]["order_type"] == order_type
    for k, v in extra.items():
        assert t.sent[0]["form"][k] == v


@pytest.mark.parametrize("kw", [{"order_type": "trailing_stop_market"}, {"order_type": "market_if_touched"}, {"tif": "fok"}, {"post_only": True}])
def test_kite_says_what_it_cannot_do(kw):
    a, t = kite()
    with pytest.raises(Unsupported):
        a.submit(request(symbol="B", **kw))
    assert t.sent == []


def test_gtc_is_refused_unless_the_caller_chooses_day():
    a, _ = kite(gtc="refuse")
    with pytest.raises(Unsupported, match="gtc='day'"):
        a.submit(request(symbol="B"))


@pytest.mark.parametrize(
    "answer,kind",
    [
        (j(400, {"status": "error", "message": "Insufficient funds", "error_type": "MarginException"}), "rejected"),
        (j(400, {"status": "error", "message": "bad", "error_type": "InputException"}), "rejected"),
    ],
)
def test_kite_refusals_become_rejections_with_the_broker_reason(answer, kind):
    a, _ = kite(answer)
    a.submit(request(symbol="B"))
    (q,) = a.drain_reports()
    assert q.report.kind == kind
    assert answer[1]["error_type"] in q.report.reason and answer[1]["message"] in q.report.reason


def test_kite_rate_limit_is_retried_and_a_gateway_timeout_is_looked_up_not_resent():
    a, t = kite(j(429, {"status": "error", "message": "slow", "error_type": "NetworkException"}), j(200, {"status": "success", "data": {"order_id": "9"}}))
    a.submit(request(symbol="B"))
    assert [s["method"] for s in t.sent] == ["POST", "POST"] and a._orders["s0-0"].venue_order_id == "9"
    row = {"order_id": "77", "status": "OPEN", "tag": "s0x0", "exchange": "NSE", "tradingsymbol": "INFY", "quantity": 10, "filled_quantity": 0, "price": 99.0}
    a, t = kite(j(504, {"status": "error", "message": "timeout", "error_type": "NetworkException"}), j(200, {"status": "success", "data": [row]}))
    a.submit(request(symbol="B"))
    assert [(s["method"], s["url"].rsplit("/", 1)[1]) for s in t.sent] == [("POST", "regular"), ("GET", "orders")]
    assert a._orders["s0-0"].venue_order_id == "77" and [q.report.kind for q in a.drain_reports()] == ["accepted"]
    a, t = kite(MaybeSent("reset"), j(200, {"status": "success", "data": []}))
    a.submit(request(symbol="B"))
    assert [e.kind for e in a.drain_events()] == ["unconfirmed"] and len(t.sent) == 2


def test_kite_token_exception_pauses_until_a_new_token():
    a, t = kite(j(403, {"status": "error", "message": "Incorrect api_key or access_token", "error_type": "TokenException"}), j(200, {"status": "success", "data": {"order_id": "5"}}))
    a.submit(request(symbol="B"))
    assert [e.kind for e in a.drain_events()] == ["session_expired"]
    a.set_token("NEW")
    assert t.sent[1]["headers"]["Authorization"] == "token KEY:NEW" and a._orders["s0-0"].venue_order_id == "5"


def test_kite_cumulative_updates_become_deterministic_fills():
    a, _ = kite(j(200, {"status": "success", "data": {"order_id": "9"}}))
    a.submit(request(symbol="B"))

    def post(status, filled, avg, **kw):
        return {"order_id": "9", "status": status, "tag": "s0x0", "exchange": "NSE", "tradingsymbol": "INFY", "quantity": 10,
                "filled_quantity": filled, "average_price": avg, "price": 99.0, "order_timestamp": "2026-11-02 09:15:01",
                "exchange_update_timestamp": "2026-11-02 09:15:02", **kw}  # fmt: skip

    for p in (post("OPEN", 0, 0), post("OPEN", 4, 99.0), post("OPEN", 4, 99.0), post("OPEN", 10, 98.7), post("OPEN", 2, 99.0), post("COMPLETE", 10, 98.7)):
        a.handle_payload({"type": "order", "data": p})
    got = [(q.report.kind, q.report.exec_id, q.report.qty, q.report.price) for q in a.drain_reports()]
    assert got == [("accepted", None, None, None), ("fill", "9:4", 4.0, 99.0), ("fill", "9:10", 6.0, 98.5)]
    assert a.drain_reports() == []


def test_kite_modify_cancel_and_replace_ack():
    a, t = kite(
        j(200, {"status": "success", "data": {"order_id": "9"}}),
        j(200, {"status": "success", "data": {"order_id": "9"}}),
        j(400, {"status": "error", "message": "Order is not open", "error_type": "OrderException"}),
    )
    a.submit(request(symbol="B"))
    a.modify("s0-0", quantity=8.0, limit_price=98.0)
    assert (t.sent[1]["method"], t.sent[1]["url"], t.sent[1]["form"]) == ("PUT", "https://api.kite.trade/orders/regular/9", {"quantity": 8, "price": 98.0})
    a.handle_payload({"order_id": "9", "status": "UPDATE", "tag": "s0x0", "exchange": "NSE", "tradingsymbol": "INFY", "quantity": 8, "price": 98.0, "filled_quantity": 0})
    assert [q.report.kind for q in a.drain_reports()] == ["accepted", "replace_ack"]
    a.cancel("s0-0")
    assert t.sent[2]["method"] == "DELETE" and [q.report.kind for q in a.drain_reports()] == ["cancel_rejected"]
    with pytest.raises(Unsupported):
        a.modify("s0-0", trigger_price=1.0)


def test_kite_reads_orders_positions_balances_and_trades():
    a, _ = kite(
        j(200, {"status": "success", "data": [{"order_id": "1", "status": "TRIGGER PENDING", "tag": "s0x3", "exchange": "NSE", "tradingsymbol": "INFY", "quantity": 5, "filled_quantity": 0, "trigger_price": 90}]}),
        j(200, {"status": "success", "data": {"net": [{"tradingsymbol": "INFY", "exchange": "NSE", "product": "MIS", "quantity": -5, "average_price": 1500.5}, {"tradingsymbol": "X", "exchange": "NSE", "quantity": 0}]}}),
        j(200, {"status": "success", "data": {"equity": {"net": 1000.0, "available": {"live_balance": 900.0}, "utilised": {"debits": 100.0}}}}),
        j(200, {"status": "success", "data": [{"trade_id": "T1", "order_id": "1", "quantity": 5, "average_price": 90.0, "fill_timestamp": "2026-11-02 09:20:00"}]}),
    )
    (o,) = a.open_orders()
    assert (o.client_order_id, o.status, o.trigger_price) == ("s0-3", "open", 90.0)
    (p,) = a.positions()
    assert (p.symbol, p.net_qty, p.avg_price, p.product) == ("NSE:INFY", -5.0, 1500.5, "intraday")
    b = a.balances()
    assert (b.available, b.used, b.total, b.currency) == (900.0, 100.0, 1000.0, "INR")
    (tr,) = a.fetch_trades()
    assert (tr.exec_id, tr.qty) == ("T1", 5.0)


def test_kite_postback_checksum():
    body = {"order_id": "1", "order_timestamp": "2026-11-02 09:15:01"}
    body["checksum"] = hashlib.sha256(b"12026-11-02 09:15:01SECRET").hexdigest()
    assert verify_postback(body, "SECRET") and not verify_postback(body, "OTHER")


def test_kite_needs_the_api_key_and_prints_no_secrets():
    with pytest.raises(ValueError):
        KiteAdapter(Session("tok"), {})
    a, _ = kite()
    assert "ACCESS" not in repr(a) and "KEY" not in repr(a)


# -- Upstox -------------------------------------------------------------------------------------------


def test_upstox_place_uses_the_hft_host_v3_body_and_protection():
    a, t = upstox(j(200, {"status": "success", "data": {"order_ids": ["1644490272000"]}, "metadata": {"latency": 40}}))
    a.submit(request(order_type="market"))
    (s,) = t.sent
    assert (s["method"], s["url"]) == ("POST", "https://api-hft.upstox.com/v3/order/place")
    assert s["headers"]["Authorization"] == "Bearer ACCESS"
    assert s["body"] == {
        "quantity": 10, "product": "I", "validity": "DAY", "price": 0, "tag": "s0x0", "instrument_token": "NSE_EQ|INE009A01021",
        "order_type": "MARKET", "transaction_type": "BUY", "disclosed_quantity": 0, "trigger_price": 0, "is_amo": False,
        "slice": False, "market_protection": -1,
    }  # fmt: skip
    assert a._orders["s0-0"].venue_order_id == "1644490272000"


def test_upstox_commodity_quantity_goes_as_lots_and_comes_back_as_units():
    a, t = upstox(j(200, {"status": "success", "data": {"order_ids": ["5"]}}))
    a.submit(request(symbol="B", quantity=300.0))
    assert t.sent[0]["body"]["quantity"] == 3
    a.handle_payload({"update_type": "order", "order_id": "5", "status": "complete", "tag": "s0x0", "instrument_token": "MCX_FO|123", "quantity": 3, "filled_quantity": 3, "average_price": 7000.0, "price": 7000.0})
    assert [(q.report.kind, q.report.qty) for q in a.drain_reports()] == [("accepted", None), ("fill", 300.0)]
    with pytest.raises(OrderRefused, match="lot size"):
        a.submit(request(symbol="B", quantity=250.0, strategy=1))


@pytest.mark.parametrize(
    "answer,expect",
    [
        (j(400, {"status": "error", "errors": [{"errorCode": "UDAPI1052", "message": "Quantity cannot be zero"}]}), "rejected"),
        (j(403, {"status": "error", "errors": [{"errorCode": "UDAPI1154", "message": "Request from non-whitelisted IP"}]}), "rejected"),
    ],
)
def test_upstox_refusals(answer, expect):
    a, _ = upstox(answer)
    a.submit(request())
    (q,) = a.drain_reports()
    assert q.report.kind == expect and answer[1]["errors"][0]["errorCode"] in q.report.reason


def test_upstox_invalid_token_pauses_and_cancel_modify_wire():
    a, t = upstox(
        j(401, {"status": "error", "errors": [{"errorCode": "UDAPI100050", "message": "Invalid token"}]}),
        j(200, {"status": "success", "data": {"order_ids": ["7"]}}),
        j(200, {"status": "success", "data": {"order_id": "7"}}),
        j(400, {"status": "error", "errors": [{"errorCode": "UDAPI100040", "message": "already complete"}]}),
    )
    a.submit(request(order_type="stop_limit", price=94.0, trigger=95.0))
    assert [e.kind for e in a.drain_events()] == ["session_expired"]
    a.set_token("N2")
    a.modify("s0-0", limit_price=93.0)
    assert (t.sent[2]["method"], t.sent[2]["url"], t.sent[2]["body"]) == (
        "PUT", "https://api-hft.upstox.com/v3/order/modify",
        {"order_id": "7", "validity": "DAY", "order_type": "SL", "price": 93.0, "trigger_price": 95.0},
    )  # fmt: skip
    a.cancel("s0-0")
    assert (t.sent[3]["method"], t.sent[3]["url"], t.sent[3]["params"]) == ("DELETE", "https://api-hft.upstox.com/v3/order/cancel", {"order_id": "7"})
    assert [q.report.kind for q in a.drain_reports()] == ["cancel_rejected"]


def test_upstox_status_strings():
    a, _ = upstox()
    base = {"order_id": "1", "tag": "s0x0", "instrument_token": "NSE_EQ|INE009A01021", "quantity": 1, "filled_quantity": 0}
    cases = {"put order req received": "received", "open pending": "received", "trigger pending": "open", "cancel pending": "open",
             "complete": "filled", "rejected": "rejected", "cancelled": "cancelled", "modified": "open"}  # fmt: skip
    for raw, norm in cases.items():
        assert a.parse_order_update({**base, "status": raw}).status == norm
    assert a.parse_order_update({**base, "status": "mystery"}) is None
    assert a.parse_order_update({"update_type": "position", "order_id": "1"}) is None


def test_upstox_reads():
    a, _ = upstox(
        j(200, {"status": "success", "data": [{"instrument_token": "NSE_EQ|INE009A01021", "trading_symbol": "INFY", "product": "D", "quantity": 4, "average_price": 1500.0}]}),
        j(200, {"status": "success", "data": {"equity": {"used_margin": 10.0, "available_margin": 90.0}}}),
        j(200, {"status": "success", "data": [{"trade_id": "t1", "order_id": "9", "quantity": 4, "average_price": 1500.0}]}),
    )
    (p,) = a.positions()
    assert (p.net_qty, p.avg_price, p.product) == (4.0, 1500.0, "delivery")
    assert a.balances().available == 90.0
    assert a.fetch_trades()[0].exec_id == "t1"


# -- Dhan ---------------------------------------------------------------------------------------------------


def test_dhan_place_body_and_headers():
    a, t = dhan(j(200, {"orderId": "112111182198", "orderStatus": "TRANSIT"}))
    a.submit(request(order_type="stop_limit", price=94.0, trigger=95.0))
    (s,) = t.sent
    assert (s["method"], s["url"]) == ("POST", "https://api.dhan.co/v2/orders")
    assert s["headers"]["access-token"] == "ACCESS" and "Authorization" not in s["headers"]
    assert s["body"] == {
        "dhanClientId": "1000", "correlationId": "s0-0", "transactionType": "BUY", "exchangeSegment": "NSE_EQ",
        "productType": "INTRADAY", "orderType": "STOP_LOSS", "validity": "DAY", "securityId": "1594", "quantity": 10,
        "price": 94.0, "triggerPrice": 95.0,
    }  # fmt: skip


@pytest.mark.parametrize(
    "status,code,kind",
    [(400, "DH-906", "rejected"), (400, "DH-905", "rejected"), (400, "DH-903", "rejected")],
)
def test_dhan_refusals(status, code, kind):
    a, _ = dhan(j(status, {"errorType": "Order_Error", "errorCode": code, "errorMessage": "Missing required fields"}))
    a.submit(request())
    (q,) = a.drain_reports()
    assert q.report.kind == kind and code in q.report.reason


def test_dhan_rate_limit_retries_internal_error_is_looked_up_and_token_error_pauses():
    a, t = dhan(j(429, {"errorCode": "DH-904", "errorMessage": "slow"}), j(200, {"orderId": "1"}))
    a.submit(request())
    assert len(t.sent) == 2 and a._orders["s0-0"].venue_order_id == "1"
    book = [{"orderId": "88", "correlationId": "s0-0", "orderStatus": "PENDING", "exchangeSegment": "NSE_EQ", "securityId": "1594", "quantity": 10, "price": 99.0, "filledQty": 0}]
    a, t = dhan(j(500, {"errorCode": "DH-908", "errorMessage": "internal"}), j(200, book))
    a.submit(request())
    assert a._orders["s0-0"].venue_order_id == "88" and len(t.sent) == 2
    a, _ = dhan(j(401, {"errorCode": "DH-901", "errorMessage": "Invalid"}))
    a.submit(request())
    assert [e.kind for e in a.drain_events()] == ["session_expired"]


def test_dhan_live_update_frames_are_cumulative_fills():
    a, _ = dhan(j(200, {"orderId": "55"}))
    a.submit(request())

    def frame(status, traded, avg):
        return json.dumps({"Type": "order_alert", "Data": {
            "Exchange": "NSE", "Segment": "E", "SecurityId": "1594", "OrderNo": "55", "CorrelationId": "s0-0", "Status": status,
            "TxnType": "B", "OrderType": "LMT", "Product": "I", "Quantity": 10, "TradedQty": traded, "AvgTradedPrice": avg,
            "Price": 99.0, "TriggerPrice": 0, "LastUpdatedTime": "2026-11-02 09:16:00"}})  # fmt: skip

    for f in (frame("PENDING", 0, 0), frame("PENDING", 3, 99.0), frame("TRADED", 10, 98.0), frame("TRADED", 10, 98.0)):
        a.handle_payload(f)
    got = [(q.report.kind, q.report.exec_id, round(q.report.qty or 0, 6), q.report.price) for q in a.drain_reports()]
    assert got == [("accepted", None, 0, None), ("fill", "55:3", 3.0, 99.0), ("fill", "55:10", 7.0, round((980 - 297) / 7, 6))]
    assert a.parse_order_update({"Type": "other"}) is None
    assert a.parse_order_update("not json") is None


def test_dhan_modify_cancel_and_reads():
    a, t = dhan(
        j(200, {"orderId": "55"}),
        j(200, {"orderId": "55", "orderStatus": "PENDING"}),
        j(200, [{"securityId": "1594", "exchangeSegment": "NSE_EQ", "productType": "CNC", "netQty": 3, "buyAvg": 1500.0, "sellAvg": 0}, {"securityId": "1", "exchangeSegment": "NSE_EQ", "netQty": 0}]),
        j(200, {"dhanClientId": "1000", "availabelBalance": 5000.5, "utilizedAmount": 10.0, "sodLimit": 6000.0}),
        j(200, [{"orderId": "55", "exchangeTradeId": "X1", "tradedQuantity": 3, "tradedPrice": 99.0}]),
    )
    a.submit(request())
    a.modify("s0-0", quantity=8.0, limit_price=98.0)
    assert (t.sent[1]["method"], t.sent[1]["url"], t.sent[1]["body"]) == (
        "PUT", "https://api.dhan.co/v2/orders/55",
        {"dhanClientId": "1000", "orderId": "55", "orderType": "LIMIT", "validity": "DAY", "quantity": 8, "price": 98.0},
    )  # fmt: skip
    (p,) = a.positions()
    assert (p.net_qty, p.avg_price, p.product) == (3.0, 1500.0, "delivery")
    assert a.balances().available == 5000.5
    assert a.fetch_trades()[0].exec_id == "X1"


def test_dhan_instruments_without_a_security_id_are_unsupported():
    t = Script()
    a = DhanAdapter(Session("T"), {"A": Instrument("NSE_EQ", "INFY")}, client_id="1", transport=t, gtc="day")
    with pytest.raises(Unsupported, match="securityId"):
        a.submit(request())


# -- transport edge ----------------------------------------------------------------------------------------------


def test_a_connection_that_never_opened_is_retried_and_exhaustion_rejects():
    a, t = kite(NotSent("refused"), NotSent("refused"), NotSent("refused"), NotSent("refused"))
    a.submit(request(symbol="B"))
    assert len(t.sent) == 4
    (q,) = a.drain_reports()
    assert q.report.kind == "rejected" and "unreachable" in q.report.reason
