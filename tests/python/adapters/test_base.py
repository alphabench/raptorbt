"""The common layer: ids, limits, outbox, threading, state, secrets."""

import threading
import time

import pytest

from raptorbt.adapters import (
    Backoff,
    IdCodec,
    Instrument,
    ManualExecutor,
    OrderStatusUpdate,
    RateLimiter,
    Secret,
    Session,
    ThreadExecutor,
    reconnect_loop,
)
from raptorbt.adapters.base import AuthExpired
from raptorbt.execution import AdapterError, ExecutionAdapter, OrderRefused, Unsupported
from raptorbt.adapters.conformance import ScriptedVenue, _reference_adapter

from .test_execution_request import request


def make(**kw):
    venue = ScriptedVenue()
    return venue, _reference_adapter(venue, kw)


# -- ids ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "charset,max_len,cid,tag",
    [("A-Za-z0-9", 20, "s0-12", "s0x12"), ("A-Za-z0-9_-", 30, "s3-4567", "s3-4567"), ("A-Za-z0-9", 20, "s12-100000", "s12x100000")],
)
def test_engine_ids_round_trip_without_state(charset, max_len, cid, tag):
    c = IdCodec(max_len=max_len, charset=charset)
    assert c.encode(cid) == tag
    assert IdCodec(max_len=max_len, charset=charset).decode(tag) == cid


def test_other_ids_are_hashed_to_a_legal_tag_and_remembered():
    c = IdCodec(max_len=20)
    tag = c.encode("my order #1")
    assert len(tag) <= 20 and tag.isalnum()
    assert c.decode(tag) == "my order #1"
    assert c.encode("s0x5") != c.encode("s0-5")  # a literal that looks like an encoded id does not collide
    fresh = IdCodec(max_len=20)
    fresh.restore(c.export())
    assert fresh.decode(tag) == "my order #1"


def test_a_namespace_keeps_two_nodes_apart():
    a, b = IdCodec(max_len=20, namespace="AA"), IdCodec(max_len=20, namespace="BB")
    assert a.decode(b.encode("s0-1")) is None and a.decode(a.encode("s0-1")) == "s0-1"
    assert len(a.encode("s0-1")) <= 20


# -- limits --------------------------------------------------------------------------------------


def test_the_token_bucket_makes_the_caller_wait_exactly_as_long_as_needed():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    lim = RateLimiter.per_window({"orders": [(2, 1), (3, 10)]}, clock=lambda: t[0], sleep=sleep)
    for _ in range(2):
        assert lim.acquire("orders") == 0  # burst of 2
    assert lim.acquire("orders") == pytest.approx(0.5)  # third waits for the 2/s refill
    lim.acquire("orders")  # fourth: the 3-per-10s window is now the limit
    assert sum(slept) >= 3.0
    assert lim.acquire("unlimited-class") == 0


def test_backoff_and_reconnect_loop():
    b = Backoff(base=1, factor=2, cap=5)
    assert [b.delay(i) for i in range(5)] == [1, 2, 4, 5, 5]
    seen, lost, slept = [], [], []

    def connect():
        seen.append(1)
        if len(seen) < 3:
            raise ConnectionError("down")

    reconnect_loop(connect, backoff=b, sleep=slept.append, should_stop=lambda: len(seen) >= 3, on_lost=lost.append)
    assert len(lost) == 2 and slept == [1, 2]


# -- the surface -------------------------------------------------------------------------------------


def test_it_is_an_execution_adapter():
    _, a = make()
    assert isinstance(a, ExecutionAdapter)


def test_submit_does_no_io_with_a_manual_executor_and_repeats_are_one_order():
    venue, a = make(executor=ManualExecutor())
    r = request()
    a.submit(r)
    a.submit(r)  # the same id and terms again: not a second order
    assert venue.place_calls == {}
    assert a.pump() == 1
    assert sum(venue.place_calls.values()) == 1
    with pytest.raises(OrderRefused, match="duplicate_client_order_id"):
        a.submit(request(price=50.0))


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"order_type": "trailing_stop_market"}, "order type"),
        ({"tif": "fok"}, "time in force"),
        ({"post_only": True}, "post_only"),
        ({"reduce_only": True}, "reduce"),
    ],
)
def test_what_a_broker_cannot_do_is_unsupported_not_silent(kw, msg):
    _, a = make()
    a._instruments = {"A": Instrument("X", "A")}
    with pytest.raises(Unsupported, match=msg):
        a.submit(request(**kw))


def test_reduce_only_is_sent_only_when_the_caller_accepts_the_difference():
    venue, a = make(reduce_only="best_effort")
    a.submit(request(reduce_only=True))
    assert sum(venue.place_calls.values()) == 1


def test_lot_and_tick_checks_are_local():
    _, a = make()
    a._instruments = {"A": Instrument("X", "A", lot_size=25, tick_size=0.05)}
    with pytest.raises(OrderRefused, match="lot size"):
        a.submit(request(quantity=30.0))
    with pytest.raises(OrderRefused, match="tick"):
        a.submit(request(quantity=50.0, price=99.03))
    a.submit(request(quantity=50.0, price=99.05))


def test_a_venue_refusal_becomes_a_rejected_report_and_never_raises():
    venue, a = make()
    venue.fail_next("place", "refuse", "no margin")
    a.submit(request())
    (q,) = a.drain_reports()
    assert (q.report.kind, q.report.reason) == ("rejected", "no margin")


def test_an_ambiguous_write_is_looked_up_and_never_resent():
    venue, a = make()
    venue.fail_next("place", "timeout_after")
    a.submit(request())
    assert sum(venue.place_calls.values()) == 1 and venue.duplicates == []
    venue.accept(a.ids.encode("s0-0"))
    a.handle_update(a.fetch_orders()[0])  # the feed says it is open
    assert [q.report.kind for q in a.drain_reports()] == ["accepted"]


def test_an_order_lost_in_flight_stays_unconfirmed_unless_resend_is_chosen():
    venue, a = make()
    venue.fail_next("place", "timeout_before")
    a.submit(request())
    assert [e.kind for e in a.drain_events()] == ["unconfirmed"]
    assert a.drain_reports() == []
    venue2, b = make(resend_if_absent=True)
    venue2.fail_next("place", "timeout_before")
    b.submit(request())
    assert sum(venue2.place_calls.values()) == 2 and venue2.duplicates == []


def test_exhausted_transient_failures_reject_with_a_reason_the_order_never_left():
    venue, a = make(max_attempts=2)
    venue.fail_next("place", "unavailable")
    venue.fail_next("place", "unavailable")
    a.submit(request())
    (q,) = a.drain_reports()
    assert q.report.kind == "rejected" and "unreachable" in q.report.reason


def test_cancel_and_modify_of_an_unknown_order_are_refused_and_cancel_is_asked_once():
    venue, a = make()
    with pytest.raises(OrderRefused):
        a.cancel("s9-9")
    with pytest.raises(OrderRefused):
        a.modify("s9-9", quantity=1.0)
    a.submit(request())
    a.handle_update(a.fetch_orders()[0])
    a.cancel("s0-0")
    a.cancel("s0-0")
    assert venue.cancel_calls[a.ids.encode("s0-0")] == 1


# -- sessions --------------------------------------------------------------------------------------------


def test_secrets_never_show_in_repr_or_state():
    s = Session("super-secret-token", api_key="key-123", refresh=lambda: "x")
    _, a = make()
    a.session = s
    for text in (repr(s), repr(a), repr(Secret("abc")), str(Secret("abc"))):
        assert "super-secret" not in text and "key-123" not in text and "abc" not in text
    a.submit(request())
    assert "super-secret" not in repr(a.export_state())
    with pytest.raises(TypeError):
        import pickle

        pickle.dumps(Secret("abc"))


def test_an_expired_token_is_refreshed_once_and_the_call_repeated():
    venue, a = make()
    calls = []
    a.session = Session("old", refresh=lambda: calls.append("login") or "new")
    real = a.place
    n = [0]

    def place(rec):
        n[0] += 1
        if n[0] == 1:
            raise AuthExpired("token")
        return real(rec)

    a.place = place
    a.submit(request())
    assert calls == ["login"] and a.session.token() == "new" and sum(venue.place_calls.values()) == 1


def test_without_a_refresh_the_outbox_pauses_and_resumes_on_a_new_token():
    venue, a = make()
    real = a.place
    ok = [False]

    def place(rec):
        if not ok[0]:
            raise AuthExpired("token")
        return real(rec)

    a.place = place
    a.submit(request())
    assert [e.kind for e in a.drain_events()] == ["session_expired"] and venue.place_calls == {}
    ok[0] = True
    a.set_token("fresh")
    assert sum(venue.place_calls.values()) == 1 and a.session.token() == "fresh"


# -- threading ---------------------------------------------------------------------------------------------


def test_the_thread_executor_keeps_the_node_thread_free_and_the_order_of_requests():
    gate = threading.Event()
    venue = ScriptedVenue()
    real_place = venue.place
    names = []

    def slow_place(*args):
        gate.wait(5)
        names.append(threading.current_thread().name)
        return real_place(*args)

    venue.place = slow_place
    a = _reference_adapter(venue, {"executor": ThreadExecutor("t-test")})
    t0 = time.perf_counter()
    a.submit(request())
    a.submit(request(strategy=1))
    assert time.perf_counter() - t0 < 0.5  # returned while the broker call is still blocked
    assert venue.place_calls == {}
    gate.set()
    for _ in range(100):
        if len(names) == 2:
            break
        time.sleep(0.02)
    a.close()
    assert names == ["t-test", "t-test"] and [o.ref for o in venue.orders.values()] == [a.ids.encode("s0-0"), a.ids.encode("s1-0")]


# -- state, fencing, reconcile reads --------------------------------------------------------------------------


def test_state_round_trips_and_unsent_jobs_run_after_import():
    venue, a = make(executor=ManualExecutor())
    a.submit(request())
    state = a.export_state()
    import json

    state = json.loads(json.dumps(state))
    b = _reference_adapter(venue, {})
    b.import_state(state)  # inline executor: the queued submit runs now
    assert sum(venue.place_calls.values()) == 1
    b.submit(request())  # still the same order after the restart
    assert sum(venue.place_calls.values()) == 1


def test_a_fence_drops_older_queued_work_and_never_goes_backwards():
    venue, a = make(executor=ManualExecutor())
    a.submit(request())
    a.fence(2)
    a.pump()
    assert venue.place_calls == {} and [e.kind for e in a.drain_events()] == ["fenced_drop"]
    with pytest.raises(AdapterError):
        a.fence(1)
    a.fence(2)


def test_reconcile_reads_carry_our_client_ids():
    venue, a = make()
    a.submit(request())
    venue.accept(a.ids.encode("s0-0"))
    (o,) = a.open_orders()
    assert (o.venue_order_id, o.status, o.client_order_id) == ("V1", "open", "s0-0")
    venue.fill(a.ids.encode("s0-0"), 10.0, 99.0)
    assert a.open_orders() == [] and a.orders()[0].status == "filled"
    with pytest.raises(Unsupported):
        a.positions()
    with pytest.raises(Unsupported):
        a.balances()


def test_trades_with_real_ids_are_the_fills_when_the_mapping_is_set_to_trades():
    venue, a = make(fill_source="trades")
    a.submit(request())
    ref = a.ids.encode("s0-0")
    venue.accept(ref)
    a.handle_update(a.fetch_orders()[0])
    venue.fill(ref, 4.0, 99.0)
    a.handle_update(a.fetch_orders()[0])  # cumulative figures add nothing in this mode
    a.handle_trade(exec_id="T1", qty=4.0, price=99.0, venue_order_id="V1")
    a.handle_trade(exec_id="T1", qty=4.0, price=99.0, venue_order_id="V1")  # repeated
    kinds = [(q.report.kind, q.report.exec_id) for q in a.drain_reports()]
    assert kinds == [("accepted", None), ("fill", "T1")]


def test_updates_for_orders_that_are_not_ours_are_ignored():
    _, a = make()
    a.handle_update(OrderStatusUpdate("X9", "open", tag="somebody-else"))
    assert a.ignored_updates == 1 and a.drain_reports() == []


def test_an_unmapped_symbol_is_unsupported():
    _, a = make()
    a._instruments = {}
    with pytest.raises(Unsupported, match="no instrument"):
        a.submit(request())
