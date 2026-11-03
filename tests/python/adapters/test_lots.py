"""One units() conversion in the base: every inbound row and every outbound quantity, per broker, lot size > 1."""

import pytest

from raptorbt.adapters import Instrument, OrderStatusUpdate, Session
from raptorbt.adapters.dhan import DhanAdapter
from raptorbt.adapters.kite import KiteAdapter
from raptorbt.adapters.upstox import UpstoxAdapter
from raptorbt.execution import OrderRefused

from .test_brokers import Script, j
from .test_execution_request import request

LOT = 25.0
NS = "N1"


def _ok(data):
    return j(200, {"status": "success", "data": data})


class Case:
    """How one broker writes the same facts, with the instrument counted in lots of 25."""

    def __init__(self, name):
        self.name = name
        self.tag = "N1s0-0" if name == "dhan" else "N1s0x0"  # Dhan's tags may carry a dash

    def build(self, *answers, **kw):
        t = Script(*answers)
        kw.update(namespace=NS, trusted=True, gtc="day", sleep=lambda s: None, transport=t)
        if self.name == "kite":
            ins = {"A": Instrument("NFO", "NIFTYFUT", lot_size=LOT, quantity_in_lots=True)}
            return KiteAdapter(Session("T", api_key="K"), ins, **kw), t
        if self.name == "upstox":
            ins = {"A": Instrument("MCX_FO", "CRUDE", key="MCX_FO|123", lot_size=LOT, quantity_in_lots=True)}
            return UpstoxAdapter(Session("T"), ins, **kw), t
        ins = {"A": Instrument("NSE_FNO", "NIFTY", key="35003", lot_size=LOT, quantity_in_lots=True)}
        return DhanAdapter(Session("T"), ins, client_id="1", **kw), t

    # replies
    def placed(self, vid="9"):
        return {"kite": _ok({"order_id": vid}), "upstox": _ok({"order_ids": [vid]}), "dhan": j(200, {"orderId": vid, "orderStatus": "TRANSIT"})}[self.name]

    def done(self):
        return {"kite": _ok({"order_id": "9"}), "upstox": _ok({"order_id": "9"}), "dhan": j(200, {"orderId": "9"})}[self.name]

    def sent_qty(self, call):
        if self.name == "kite":
            return call["form"]["quantity"]
        return call["body"]["quantity"]

    # rows, quantities as the broker counts them (lots)
    def order_row(self, qty, filled, status="complete", price=7000.0):
        if self.name == "kite":
            return {"order_id": "9", "status": "COMPLETE" if status == "complete" else "OPEN", "tag": self.tag, "exchange": "NFO", "tradingsymbol": "NIFTYFUT",
                    "transaction_type": "BUY", "order_type": "LIMIT", "quantity": qty, "filled_quantity": filled, "average_price": price, "price": 99.0}
        if self.name == "upstox":
            return {"order_id": "9", "status": status, "tag": self.tag, "instrument_token": "MCX_FO|123", "transaction_type": "BUY", "order_type": "LIMIT",
                    "quantity": qty, "filled_quantity": filled, "average_price": price, "price": 99.0}
        return {"orderId": "9", "orderStatus": "TRADED" if status == "complete" else "PENDING", "correlationId": self.tag, "exchangeSegment": "NSE_FNO", "securityId": "35003",
                "transactionType": "BUY", "orderType": "LIMIT", "quantity": qty, "filledQty": filled, "averageTradedPrice": price, "price": 99.0}

    def frame(self, row):
        """The same row as a push message."""
        if self.name == "kite":
            return {"type": "order", "data": row}
        if self.name == "upstox":
            return {"update_type": "order", "data": row}
        return {"Type": "order_alert", "Data": {
            "Exchange": "NSE", "Segment": "D", "SecurityId": "35003", "OrderNo": "9", "CorrelationId": self.tag,
            "Status": row["orderStatus"], "TxnType": "B", "OrderType": "LMT", "Quantity": row["quantity"], "TradedQty": row["filledQty"],
            "AvgTradedPrice": row["averageTradedPrice"], "Price": 99.0}}

    def book(self, rows):
        return _ok(rows) if self.name != "dhan" else j(200, rows)

    def trade(self, qty):
        if self.name == "kite":
            return {"trade_id": "T1", "order_id": "9", "quantity": qty, "average_price": 7000.0}
        if self.name == "upstox":
            return {"trade_id": "T1", "order_id": "9", "quantity": qty, "average_price": 7000.0}
        return {"exchangeTradeId": "T1", "orderId": "9", "tradedQuantity": qty, "tradedPrice": 7000.0}

    def positions(self, qty):
        if self.name == "kite":
            return _ok({"net": [{"exchange": "NFO", "tradingsymbol": "NIFTYFUT", "quantity": qty, "average_price": 7000.0, "product": "MIS"}]})
        if self.name == "upstox":
            return _ok([{"instrument_token": "MCX_FO|123", "quantity": qty, "average_price": 7000.0, "product": "I"}])
        return j(200, [{"exchangeSegment": "NSE_FNO", "securityId": "35003", "netQty": qty, "buyAvg": 7000.0, "sellAvg": 7000.0, "productType": "INTRADAY"}])


CASES = [Case(n) for n in ("kite", "upstox", "dhan")]
ids = [c.name for c in CASES]


@pytest.fixture(params=CASES, ids=ids)
def case(request):
    return request.param


def test_an_order_goes_out_in_lots_and_a_modify_too(case):
    a, t = case.build(case.placed(), case.done())
    a.submit(request(quantity=100.0))
    assert case.sent_qty(t.sent[0]) == 4
    a.modify("s0-0", quantity=75.0, limit_price=98.0)
    assert case.sent_qty(t.sent[1]) == 3
    with pytest.raises(OrderRefused, match="lot size"):
        a.submit(request(quantity=60.0, strategy=1))  # not a whole number of lots


def test_a_stream_frame_in_lots_is_a_fill_in_units(case):
    a, _ = case.build(case.placed())
    a.submit(request(quantity=100.0))
    row = case.order_row(4, 4)
    a.handle_payload(case.frame(row))
    assert [(q.report.kind, q.report.qty) for q in a.drain_reports()] == [("accepted", None), ("fill", 100.0)]


def test_a_partial_fill_in_lots_is_booked_in_units_not_one_per_lot(case):
    a, _ = case.build(case.placed())
    a.submit(request(quantity=100.0))
    a.handle_payload(case.frame(case.order_row(4, 1, status="open")))
    a.handle_payload(case.frame(case.order_row(4, 4)))
    assert [q.report.qty for q in a.drain_reports() if q.report.kind == "fill"] == [25.0, 75.0]


def test_a_modified_quantity_in_lots_is_acknowledged(case):
    a, _ = case.build(case.placed(), case.done())
    a.submit(request(quantity=100.0))
    a.handle_payload(case.frame(case.order_row(4, 0, status="open")))
    a.modify("s0-0", quantity=75.0)
    a.handle_payload(case.frame(case.order_row(3, 0, status="open")))  # the venue now shows 3 lots
    assert [q.report.kind for q in a.drain_reports()] == ["accepted", "replace_ack"]
    assert a._orders["s0-0"].quantity == 75.0


def test_the_order_book_read_is_in_units(case):
    a, _ = case.build(case.placed(), case.book([case.order_row(4, 2, status="open")]))
    a.submit(request(quantity=100.0))
    (o,) = a.orders()
    assert (o.quantity, o.filled_qty, o.client_order_id) == (100.0, 50.0, "s0-0")
    a2, _ = case.build(case.placed(), case.book([case.order_row(4, 2, status="open")]))
    a2.submit(request(quantity=100.0))
    a2.resync()
    a2.pump()
    assert [q.report.qty for q in a2.drain_reports() if q.report.kind == "fill"] == [50.0]


def test_the_trade_book_is_in_units(case):
    a, _ = case.build(case.placed(), case.book([case.trade(1)]), fill_source="trades")
    a.submit(request(quantity=100.0))
    a.handle_update(OrderStatusUpdate("9", "open", tag=a.ids.encode("s0-0"), symbol="A", quantity=100.0))
    a.sync_trades()
    assert [(q.report.kind, q.report.qty) for q in a.drain_reports() if q.report.kind == "fill"] == [("fill", 25.0)]


def test_a_trade_fed_by_the_host_can_name_the_brokers_count(case):
    a, _ = case.build(case.placed(), fill_source="trades")
    a.submit(request(quantity=100.0))
    a.handle_trade(exec_id="T1", qty=2, price=7000.0, venue_order_id="9", wire=True)
    a.handle_trade(exec_id="T2", qty=10.0, price=7000.0, venue_order_id="9")  # units unless said otherwise
    assert [q.report.qty for q in a.drain_reports()] == [50.0, 10.0]


def test_positions_are_in_units_long_and_short(case):
    a, _ = case.build(case.positions(3))
    (p,) = a.positions()
    assert (p.symbol, p.net_qty) == ("A", 75.0) and p.avg_price == 7000.0


def test_a_short_position_stays_negative_in_units(case):
    if case.name == "dhan":
        rows = j(200, [{"exchangeSegment": "NSE_FNO", "securityId": "35003", "netQty": -2, "buyAvg": 0.0, "sellAvg": 7000.0, "productType": "INTRADAY"}])
    else:
        rows = case.positions(-2)
    a, _ = case.build(rows)
    assert [p.net_qty for p in a.positions()] == [-50.0]


def test_a_position_on_an_instrument_the_map_does_not_know_is_flagged_not_converted():
    t = Script(j(200, {"status": "success", "data": {"net": [{"exchange": "NFO", "tradingsymbol": "OTHER", "quantity": 3, "average_price": 1.0, "product": "MIS"}]}}))
    a = KiteAdapter(Session("T", api_key="K"), {}, namespace=NS, transport=t, sleep=lambda s: None)
    (p,) = a.positions()
    assert p.net_qty == 3.0 and [e.kind for e in a.drain_events()] == ["unmapped_instrument"]
