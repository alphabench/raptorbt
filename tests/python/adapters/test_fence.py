"""The host's lease: fencing that holds across processes, and the surface F2's node reads."""

import pytest

from raptorbt.adapters import ManualExecutor, Session
from raptorbt.adapters.base import Fenced
from raptorbt.adapters.conformance import ScriptedVenue, _reference_adapter
from raptorbt.adapters.kite import KiteAdapter

from .test_brokers import Script, j
from .test_execution_request import request


def make(**kw):
    venue = ScriptedVenue()
    kw.setdefault("namespace", "N1")
    return venue, _reference_adapter(venue, kw)


class Lease:
    def __init__(self):
        self.ok, self.asked = True, []

    def holds(self, epoch):
        self.asked.append(epoch)
        return self.ok


def test_without_a_lease_fencing_only_guards_this_process_and_says_so():
    _, a = make()
    assert a.can_fence is False
    assert "only guards this process" in " ".join(a.fence.__doc__.split()).replace("**", "")


def test_a_lease_makes_the_adapter_able_to_fence():
    _, a = make(lease=Lease())
    assert a.can_fence is True
    _, b = make(lease=lambda epoch: True)
    assert b.can_fence is True


def test_the_lease_is_asked_before_every_write_with_the_current_epoch():
    lease = Lease()
    venue, a = make(lease=lease)
    a.fence(3)
    a.submit(request())
    assert lease.asked == [3] and sum(venue.place_calls.values()) == 1


def test_a_lost_lease_refuses_with_fenced_and_sends_nothing():
    lease = Lease()
    venue, a = make(lease=lease, executor=ManualExecutor())
    a.submit(request())
    a.submit(request(symbol="B", strategy=1))
    lease.ok = False  # another process took over before the executor ran
    a.pump()
    assert venue.place_calls == {}
    kinds = [e.kind for e in a.drain_events()]
    assert kinds.count("fenced_drop") == 2 and "fenced" in kinds
    with pytest.raises(Fenced):
        a.submit(request(strategy=2))
    with pytest.raises(Fenced):
        a.cancel("s0-0")
    with pytest.raises(Fenced):
        a.modify("s0-0", quantity=5.0)


def test_a_failing_lease_check_fails_closed():
    def broken(epoch):
        raise OSError("lease store down")

    venue, a = make(lease=broken, executor=ManualExecutor())
    a.submit(request())
    a.pump()
    assert venue.place_calls == {}
    assert "lease_check_failed" in [e.kind for e in a.drain_events()]


def test_a_higher_epoch_after_re_acquiring_the_lease_sends_again():
    lease = Lease()
    venue, a = make(lease=lease, executor=ManualExecutor())
    lease.ok = False
    a.submit(request())
    a.pump()
    lease.ok = True
    a.fence(2)
    a.submit(request(strategy=1))
    a.pump()
    assert sum(venue.place_calls.values()) == 1


def test_a_lease_lost_between_retries_stops_the_retries():
    calls = []

    def holds(epoch):
        calls.append(epoch)
        return len(calls) < 2  # held for the first attempt, lost before the retry

    venue, a = make(lease=holds, max_attempts=4)
    venue.fail_next("place", "unavailable")
    a.submit(request())
    assert sum(venue.place_calls.values()) == 1
    assert "fenced" in [e.kind for e in a.drain_events()]


def test_resync_reads_are_not_writes_and_do_not_need_the_lease():
    lease = Lease()
    venue, a = make(lease=lease)
    lease.ok = False
    a.resync()
    assert lease.asked == []


def test_the_namespace_is_a_read_only_property():
    _, a = make(namespace="Zq9")
    assert a.namespace == "Zq9"
    with pytest.raises(AttributeError):
        a.namespace = "other"
    assert a.export_state()["namespace"] == "Zq9"


def test_a_lease_that_is_neither_callable_nor_has_holds_is_refused():
    with pytest.raises(ValueError, match="lease"):
        make(lease=object())


def test_a_row_with_a_status_the_mapping_cannot_read_is_reported_once_not_dropped_silently():
    t = Script(j(200, {"status": "success", "data": [{"order_id": "5", "status": "WEIRD STATE", "tag": "N1s0x0", "exchange": "NSE", "tradingsymbol": "X", "filled_quantity": 0}]}) )
    a = KiteAdapter(Session("T", api_key="K"), {}, namespace="N1", transport=t, sleep=lambda s: None)
    assert a.orders() == []
    (e,) = a.drain_events()
    assert e.kind == "unmapped_status" and e.client_order_id == "s0-0" and '"WEIRD STATE"' in e.detail and '"5"' in e.detail
    a._unmapped_status("5", "WEIRD STATE", "N1s0x0")
    assert a.drain_events() == []  # once per order and status
