"""Defects found at the M1 gate: each test failed on the code before the fix."""

import hashlib
import json

import pytest

from raptorbt.adapters import Instrument, ManualExecutor, OrderStatusUpdate, Session
from raptorbt.adapters import kite as kite_mod
from raptorbt.adapters.base import AuthExpired, BrokerAdapter, VenueBalances
from raptorbt.adapters.conformance import SCENARIOS, ReferenceBinding, ScriptedVenue, _reference_adapter, run_conformance
from raptorbt.adapters.dhan import DhanAdapter
from raptorbt.adapters.kite import KiteAdapter
from raptorbt.adapters.upstox import UpstoxAdapter
from raptorbt.execution import AdapterError, OrderRefused

from .test_brokers import Script, j
from .test_execution_request import request


def make(**kw):
    venue = ScriptedVenue()
    kw.setdefault("namespace", "N1")
    return venue, _reference_adapter(venue, kw)


# -- another run's order in the day's book is never this run's -------------------------------------


def test_a_row_of_an_earlier_run_under_the_same_tag_is_not_adopted_by_a_fast_feed():
    venue, a = make(executor=ManualExecutor())
    r = request(quantity=5.0, price=50.0, symbol="B")
    a.submit(r)
    tag = a.ids.encode("s0-0")
    venue.preload(tag, "A", "buy", 10.0, 99.0, 10.0)  # the old run's s0-0, complete
    a.handle_update(a.fetch_orders()[0])  # the feed delivers the old row before our place answers
    assert a.drain_reports() == []
    a.pump()
    venue.accept(tag)
    for row in a.fetch_orders():
        a.handle_update(row)
    assert [q.report.kind for q in a.drain_reports()] == ["accepted"]
    assert a._orders["s0-0"].venue_order_id == "V2"


def test_a_lookup_after_an_ambiguous_place_skips_rows_with_other_terms():
    venue, a = make()
    tag = a.ids.encode("s0-0")
    venue.preload(tag, "A", "buy", 10.0, 99.0, 10.0)
    venue.fail_next("place", "timeout_before")
    a.submit(request(quantity=5.0, price=50.0, symbol="B"))
    assert a._orders["s0-0"].venue_order_id is None and a.drain_reports() == []
    assert [e.kind for e in a.drain_events()] == ["unconfirmed"]


def test_a_lookup_finds_our_order_among_rows_with_the_same_tag():
    venue, a = make()
    tag = a.ids.encode("s0-0")
    venue.preload(tag, "A", "buy", 10.0, 99.0, 10.0)
    venue.fail_next("place", "timeout_after")
    a.submit(request(quantity=5.0, price=50.0, symbol="B"))
    assert a._orders["s0-0"].venue_order_id == "V2"


def test_a_real_adapter_needs_a_namespace():
    with pytest.raises(TypeError):
        KiteAdapter(Session("t", api_key="k"), {}, transport=Script())
    with pytest.raises(ValueError, match="namespace"):
        KiteAdapter(Session("t", api_key="k"), {}, namespace="", transport=Script())


def test_the_tag_carries_the_namespace_within_each_brokers_limits():
    for cls, kw, lim in ((KiteAdapter, {}, 20), (UpstoxAdapter, {}, 40), (DhanAdapter, {"client_id": "1"}, 30)):
        a = cls(Session("t", api_key="k"), {}, namespace="Ab3x", transport=Script(), **kw)
        tag = a.ids.encode("s12-123456")
        assert tag.startswith("Ab3x") and len(tag) <= lim
        assert a.ids.decode(tag) == "s12-123456"


def test_conformance_scenario_restart_the_same_day_with_old_orders_in_the_book():
    sc = next(s for s in SCENARIOS if s.name == "restart_same_day_with_old_orders_in_the_book")
    assert not run_conformance(ReferenceBinding, scenarios=(sc,))


# -- cumulative quantities are exact ------------------------------------------------------------------


def test_fills_of_large_orders_keep_distinct_ids():
    venue, a = make()
    a.submit(request(quantity=10_000_000.0, price=99.0, capital=1e12))
    row = lambda cum: OrderStatusUpdate("V1", "open", tag=a.ids.encode("s0-0"), symbol="A", side="buy", quantity=10_000_000.0, filled_qty=cum, avg_price=99.0, limit_price=99.0)  # noqa: E731
    venue.place_calls  # the order is placed; V1 exists
    for cum in (1_000_000.0, 1_000_001.0, 1_000_002.0, 9_999_999.0, 10_000_000.0):
        a.handle_update(row(cum))
    fills = [q.report for q in a.drain_reports() if q.report.kind == "fill"]
    assert [f.qty for f in fills] == [1_000_000.0, 1.0, 1.0, 8_999_997.0, 1.0]
    assert len({f.exec_id for f in fills}) == 5
    assert sum(f.qty for f in fills) == 10_000_000.0


def test_fractional_cumulative_quantities_keep_distinct_ids():
    _, a = make()
    a.submit(request(quantity=10.0))
    tag = a.ids.encode("s0-0")
    for cum in (1.0000001, 1.0000002):
        a.handle_update(OrderStatusUpdate("V1", "open", tag=tag, symbol="A", side="buy", quantity=10.0, filled_qty=cum, avg_price=99.0, limit_price=99.0))
    assert len({q.report.exec_id for q in a.drain_reports() if q.report.kind == "fill"}) == 2


# -- a restart never places an in-flight order again ---------------------------------------------------------


def test_import_of_a_state_taken_during_place_looks_the_order_up_first():
    venue, a = make()
    snap = {}
    real = a.place

    def place(rec):
        snap["s"] = json.loads(json.dumps(a.export_state()))  # the job is still at the head of the outbox
        return real(rec)

    a.place = place
    venue.fail_next("place", "timeout_after")  # the venue has it; the process dies before it hears
    a.submit(request())
    b = _reference_adapter(venue, {"namespace": "N1"})
    b.import_state(snap["s"])
    assert sum(venue.place_calls.values()) == 1 and venue.duplicates == []
    assert b._orders["s0-0"].venue_order_id == "V1"


def test_resend_waits_and_looks_again_before_sending():
    slept = []
    venue, a = make(resend_if_absent=True, resend_delay=2.5, sleep=slept.append)
    venue.fail_next("place", "timeout_before")
    a.submit(request())
    assert 2.5 in slept
    assert sum(venue.place_calls.values()) == 2 and venue.duplicates == []


def test_an_order_that_shows_up_during_the_wait_is_adopted_not_resent():
    venue, a = make(resend_if_absent=True, resend_delay=2.5)
    tag = a.ids.encode("s0-0")
    venue.fail_next("place", "timeout_before")  # the book is behind: the request is still in flight

    def late_arrival(seconds):
        if seconds == 2.5:
            venue.place(tag, "A", "buy", 10.0, 99.0, None)

    a._sleep = late_arrival
    a.submit(request())
    assert venue.duplicates == [] and a._orders["s0-0"].venue_order_id == "V1"
    assert sum(venue.place_calls.values()) == 2  # the late arrival of the first request, not a resend


# -- a cancel or modify that cannot be confirmed is answered ----------------------------------------------------


def _open(a, venue):
    a.submit(request())
    venue.accept(a.ids.encode("s0-0"))
    a.handle_update(a.fetch_orders()[0])
    a.drain_reports()


def test_a_cancel_the_venue_certainly_never_received_is_rejected_so_the_strategy_can_act():
    venue, a = make(max_attempts=2)
    _open(a, venue)
    for _ in range(4):
        venue.fail_next("cancel", "unavailable")
    a.cancel("s0-0")
    assert [q.report.kind for q in a.drain_reports()] == ["cancel_rejected"]
    assert not a._orders["s0-0"].cancel_requested
    assert venue.cancel_calls[a.ids.encode("s0-0")] == 4
    a.cancel("s0-0")  # the strategy may ask again
    assert venue.cancel_calls[a.ids.encode("s0-0")] == 5


def test_a_cancel_that_stays_ambiguous_is_not_reported_as_refused_it_may_have_landed():
    venue, a = make(max_attempts=2)
    _open(a, venue)
    for _ in range(4):
        venue.fail_next("cancel", "timeout_after")  # the venue took every one of them
    a.cancel("s0-0")
    assert a.drain_reports() == [] and a._orders["s0-0"].cancel_requested
    assert "cancel_pending" in [e.kind for e in a.drain_events()]
    venue.ack_cancel(a.ids.encode("s0-0"))
    a.settle_pending()  # the book shows what happened
    assert [q.report.kind for q in a.drain_reports()] == ["cancel_ack"]


def test_a_cancel_left_pending_is_answered_by_settle_pending_when_the_order_still_works():
    venue, a = make(max_attempts=2)
    _open(a, venue)
    for _ in range(4):
        venue.fail_next("cancel", "timeout_before")
    a.cancel("s0-0")
    assert a.drain_reports() == []
    a.settle_pending()
    assert [q.report.kind for q in a.drain_reports()] == ["cancel_rejected"]


def test_a_cancel_that_fails_once_more_than_its_retries_is_retried_after_a_resync():
    venue, a = make(max_attempts=2)
    _open(a, venue)
    for _ in range(2):
        venue.fail_next("cancel", "timeout_before")
    a.cancel("s0-0")
    assert a.drain_reports() == [] and venue.cancel_calls[a.ids.encode("s0-0")] == 3


def test_a_modify_the_venue_certainly_never_received_is_rejected_and_clears_the_pending_change():
    venue, a = make(max_attempts=2)
    _open(a, venue)
    for _ in range(4):
        venue.fail_next("modify", "unavailable")
    a.modify("s0-0", quantity=8.0, limit_price=98.0)
    assert [q.report.kind for q in a.drain_reports()] == ["replace_rejected"]
    assert a._orders["s0-0"].pending_modify is None


def test_a_modify_that_stays_ambiguous_stays_pending_and_is_resolved_against_the_venues_terms():
    venue, a = make(max_attempts=2)
    _open(a, venue)
    for _ in range(4):
        venue.fail_next("modify", "timeout_after")  # the venue took every one of them
    a.modify("s0-0", quantity=8.0, limit_price=98.0)
    rec = a._orders["s0-0"]
    assert a.drain_reports() == [] and rec.pending_modify == (8.0, 98.0, None) and rec.quantity == 10.0
    assert "modify_pending" in [e.kind for e in a.drain_events()]
    venue.apply_modify(a.ids.encode("s0-0"))
    venue.fill(a.ids.encode("s0-0"), 8.0, 98.0)  # a fill of the new size is not an overfill
    a.settle_pending()
    assert [q.report.kind for q in a.drain_reports()] == ["replace_ack", "fill"] and rec.quantity == 8.0


def test_a_modify_left_pending_that_the_venue_never_applied_is_rejected_by_settle_pending():
    venue, a = make(max_attempts=2)
    _open(a, venue)
    for _ in range(4):
        venue.fail_next("modify", "timeout_before")
    a.modify("s0-0", quantity=8.0, limit_price=98.0)
    assert a.drain_reports() == []
    a.settle_pending()
    assert [q.report.kind for q in a.drain_reports()] == ["replace_rejected"]


def test_an_async_modify_rejection_from_the_feed_clears_the_pending_change():
    venue, a = make()
    _open(a, venue)
    a.modify("s0-0", quantity=8.0, limit_price=98.0)
    row = a.fetch_orders()[0]
    a.handle_update(OrderStatusUpdate(**{**row.__dict__, "modify_failed": True, "reason": "price outside range"}))
    (q,) = a.drain_reports()
    assert (q.report.kind, q.report.reason) == ("replace_rejected", "price outside range")
    assert a._orders["s0-0"].pending_modify is None


def test_settle_pending_answers_a_modify_the_venue_never_applied():
    venue, a = make()
    _open(a, venue)
    a.modify("s0-0", quantity=8.0, limit_price=98.0)  # the call succeeded; the venue applies nothing
    assert a.drain_reports() == []
    a.settle_pending()
    assert [q.report.kind for q in a.drain_reports()] == ["replace_rejected"]


def test_settle_pending_leaves_a_modify_the_venue_did_apply():
    venue, a = make()
    _open(a, venue)
    a.modify("s0-0", quantity=8.0, limit_price=98.0)
    venue.apply_modify(a.ids.encode("s0-0"))
    a.settle_pending()
    assert [q.report.kind for q in a.drain_reports()] == ["replace_ack"]


def test_upstox_not_modified_and_not_cancelled_are_failures_of_the_request_not_states():
    t = Script(j(200, {"status": "success", "data": {"order_ids": ["1"]}}))
    a = UpstoxAdapter(Session("t"), {"A": Instrument("NSE_EQ", "INFY", key="K")}, namespace="N1", transport=t, trusted=True, gtc="day", sleep=lambda s: None)
    a.submit(request())
    base = {"order_id": "1", "tag": "N1s0x0", "instrument_token": "K", "quantity": 1, "filled_quantity": 0}
    assert a.parse_order_update({**base, "status": "not modified"}).modify_failed
    assert a.parse_order_update({**base, "status": "not cancelled"}).cancel_failed


# -- unauthenticated payloads never become reports -----------------------------------------------------------------


def test_a_payload_is_refused_unless_the_host_vouches_for_the_channel():
    venue, a = make(trusted=False)
    _open(a, venue)
    with pytest.raises(AdapterError, match="unauthenticated"):
        a.handle_payload(venue.find(a.ids.encode("s0-0"))[0])


def test_a_verifier_gates_payloads_and_a_failed_one_is_an_event_not_a_report():
    ok = [False]
    venue, a = make(trusted=False, verifier=lambda p: ok[0])
    _open(a, venue)
    venue.fill(a.ids.encode("s0-0"), 10.0, 99.0)
    a.handle_payload(venue.find(a.ids.encode("s0-0"))[0])
    assert a.drain_reports() == [] and [e.kind for e in a.drain_events()][-1] == "payload_rejected"
    ok[0] = True
    a.handle_payload(venue.find(a.ids.encode("s0-0"))[0])
    assert [q.report.kind for q in a.drain_reports()] == ["fill"]


def test_trusted_lets_a_host_authenticated_channel_through():
    venue, a = make(trusted=True)
    _open(a, venue)
    venue.fill(a.ids.encode("s0-0"), 10.0, 99.0)
    a.handle_payload(venue.find(a.ids.encode("s0-0"))[0])
    assert [q.report.kind for q in a.drain_reports()] == ["fill"]


def test_kite_checksum_is_compared_in_constant_time(monkeypatch):
    calls = []
    real = kite_mod.hmac.compare_digest
    monkeypatch.setattr(kite_mod.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    body = {"order_id": "1", "order_timestamp": "t", "checksum": hashlib.sha256(b"1tSECRET").hexdigest()}
    assert kite_mod.verify_postback(body, "SECRET") and calls


def test_kite_verifier_factory_accepts_good_postbacks_and_refuses_forged_ones():
    t = Script(j(200, {"status": "success", "data": {"order_id": "9"}}))
    a = KiteAdapter(Session("tok", api_key="k"), {"A": Instrument("NSE", "INFY")}, namespace="N1", transport=t,
                    verifier=kite_mod.postback_verifier("SECRET"), gtc="day", sleep=lambda s: None)  # fmt: skip
    a.submit(request(symbol="A"))
    row = {"order_id": "9", "status": "COMPLETE", "tag": a.ids.encode("s0-0"), "exchange": "NSE", "tradingsymbol": "INFY",
           "quantity": 10, "filled_quantity": 10, "average_price": 99.0, "price": 99.0, "order_timestamp": "2026-11-02 09:15:01"}  # fmt: skip
    good = {**row, "checksum": hashlib.sha256(("9" + row["order_timestamp"] + "SECRET").encode()).hexdigest()}
    a.handle_payload({**row, "checksum": "forged"})
    assert a.drain_reports() == []
    a.handle_payload(good)
    assert "fill" in [q.report.kind for q in a.drain_reports()]


# -- M4: a lot size is never guessed ----------------------------------------------------------------------------------------


def test_upstox_converts_lots_with_instruments_given_as_a_function():
    ins = Instrument("MCX_FO", "CRUDE", key="MCX_FO|123", lot_size=100.0, quantity_in_lots=True)
    t = Script(j(200, {"status": "success", "data": {"order_ids": ["5"]}}))
    a = UpstoxAdapter(Session("t"), lambda s: ins if s == "A" else None, namespace="N1", transport=t, gtc="day", trusted=True, sleep=lambda s: None)
    a.submit(request(quantity=300.0))
    a.handle_payload({"order_id": "5", "status": "complete", "tag": a.ids.encode("s0-0"), "instrument_token": "MCX_FO|123",
                      "quantity": 3, "filled_quantity": 3, "average_price": 7000.0, "price": 7000.0})  # fmt: skip
    fills = [q.report for q in a.drain_reports() if q.report.kind == "fill"]
    assert [f.qty for f in fills] == [300.0]


# -- M6: finished orders are forgotten after the retention -----------------------------------------------------------------------


def test_finished_orders_are_evicted_after_the_retention_and_stay_refused():
    now = [0.0]
    venue, a = make(retention_s=60.0, monotonic=lambda: now[0])
    _open(a, venue)
    tag = a.ids.encode("s0-0")
    venue.fill(tag, 10.0, 99.0)
    a.handle_update(a.fetch_orders()[0])
    a.drain_reports()
    now[0] = 30.0
    a.pump()
    assert "s0-0" in a._orders  # late reports still find it
    now[0] = 61.0
    a.sweep()
    assert a._orders == {} and a._by_venue == {}
    before = a.ignored_updates
    a.handle_update(a.fetch_orders()[0])  # a very late repeat
    assert a.ignored_updates == before + 1 and a.drain_reports() == []
    with pytest.raises(OrderRefused, match="duplicate_client_order_id"):
        a.submit(request())


def test_open_orders_are_never_evicted():
    now = [0.0]
    venue, a = make(retention_s=1.0, monotonic=lambda: now[0])
    _open(a, venue)
    now[0] = 10_000.0
    a.sweep()
    assert "s0-0" in a._orders


# -- M7: malformed rows are refused where they are parsed ---------------------------------------------------------------------------


def _k(*answers):
    return KiteAdapter(Session("t", api_key="k"), {}, namespace="N1", transport=Script(*answers), sleep=lambda s: None)


def test_kite_trade_rows_without_a_quantity_or_price_are_refused():
    with pytest.raises(AdapterError, match="quantity"):
        _k(j(200, {"status": "success", "data": [{"trade_id": "T1", "order_id": "1", "average_price": 9.0}]})).fetch_trades()
    with pytest.raises(AdapterError, match="average_price"):
        _k(j(200, {"status": "success", "data": [{"trade_id": "T1", "order_id": "1", "quantity": 1}]})).fetch_trades()


def test_kite_balance_does_not_fall_back_to_net():
    with pytest.raises(AdapterError, match="live_balance"):
        _k(j(200, {"status": "success", "data": {"equity": {"net": 1000.0, "available": {}}}})).balances()


def test_upstox_and_dhan_refuse_rows_with_missing_numbers():
    u = UpstoxAdapter(Session("t"), {}, namespace="N1", transport=Script(j(200, {"status": "success", "data": [{"trade_id": "t", "order_id": "1", "quantity": 1}]})), sleep=lambda s: None)
    with pytest.raises(AdapterError, match="average_price"):
        u.fetch_trades()
    u = UpstoxAdapter(Session("t"), {}, namespace="N1", transport=Script(j(200, {"status": "success", "data": {"equity": {}}})), sleep=lambda s: None)
    with pytest.raises(AdapterError, match="available_margin"):
        u.balances()
    d = DhanAdapter(Session("t"), {}, client_id="1", namespace="N1", transport=Script(j(200, {"dhanClientId": "1"})), sleep=lambda s: None)
    with pytest.raises(AdapterError, match="availabelBalance"):
        d.balances()
    d = DhanAdapter(Session("t"), {}, client_id="1", namespace="N1", transport=Script(j(200, [{"orderId": "1", "exchangeTradeId": "X", "tradedQuantity": 1}])), sleep=lambda s: None)
    with pytest.raises(AdapterError, match="tradedPrice"):
        d.fetch_trades()


def test_a_malformed_feed_message_is_dropped_with_an_event_not_booked():
    venue, a = make(trusted=True)
    k = KiteAdapter(Session("t", api_key="k"), {"A": Instrument("NSE", "INFY")}, namespace="N1", transport=Script(j(200, {"status": "success", "data": {"order_id": "9"}})), trusted=True, gtc="day", sleep=lambda s: None)
    k.submit(request())
    k.handle_payload({"order_id": "9", "status": "COMPLETE", "tag": k.ids.encode("s0-0"), "exchange": "NSE", "tradingsymbol": "INFY", "quantity": 10})
    assert k.drain_reports() == [] and [e.kind for e in k.drain_events()][-1] == "malformed_payload"


def test_balances_name_their_currency():
    with pytest.raises(TypeError):
        VenueBalances(available=1.0)
    t = Script(j(200, {"status": "success", "data": {"equity": {"net": 1.0, "available": {"live_balance": 1.0}}}}))
    assert KiteAdapter(Session("t", api_key="k"), {}, namespace="N1", transport=t, sleep=lambda s: None).balances().currency == "INR"
