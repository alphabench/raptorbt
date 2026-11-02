"""The reference binding passes the kit; broken adapters do not."""

import pytest

from raptorbt.adapters.conformance import (
    SCENARIOS,
    ReferenceBinding,
    run_conformance,
    run_scenario,
)


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_reference_adapter_converges_under_every_fault(sc):
    failures = run_conformance(ReferenceBinding, scenarios=(sc,))
    assert not failures, "\n".join(map(str, failures))


def test_the_baseline_ends_where_the_script_says():
    sc = next(s for s in SCENARIOS if s.name == "cancel_loses_race_to_a_fill")
    out = run_scenario(ReferenceBinding, sc, "in_order")
    assert out.orders == [("o0", "canceled", 3.0, 99.0)]
    assert out.positions == 3.0
    assert out.errors == [] and out.double_sent == []


# -- the kit catches a broken adapter -------------------------------------------------------


def test_the_kit_catches_a_blind_retry(monkeypatch):
    from raptorbt.adapters.base import BrokerAdapter

    def blind(self, rec, why):  # sends again without asking the venue
        self.place(rec)

    monkeypatch.setattr(BrokerAdapter, "_settle_unconfirmed", blind)
    sc = next(s for s in SCENARIOS if s.name == "answer_lost_after_the_venue_took_the_order")
    failures = run_conformance(ReferenceBinding, scenarios=(sc,), profiles=())
    assert any("sent twice" in f.what for f in failures)


def test_the_kit_catches_cumulative_figures_booked_as_fills(monkeypatch):
    from raptorbt.adapters.base import BrokerAdapter
    from raptorbt.execution import ExecutionReport

    def wrong(self, rec, u, ts, arrival):
        if u.filled_qty > 0:
            self._push(rec, ExecutionReport.fill(f"{u.venue_order_id}:{u.status}:{u.filled_qty}", u.filled_qty, u.avg_price, ts), arrival)

    monkeypatch.setattr(BrokerAdapter, "_fold_cumulative", wrong)
    sc = next(s for s in SCENARIOS if s.name == "partial_fills")
    assert run_conformance(ReferenceBinding, scenarios=(sc,))


def test_the_kit_catches_an_adapter_that_loses_the_last_word(monkeypatch):
    from raptorbt.adapters.base import BrokerAdapter

    monkeypatch.setattr(BrokerAdapter, "_resync_now", lambda self: None)
    sc = next(s for s in SCENARIOS if s.name == "fill")
    assert run_conformance(ReferenceBinding, scenarios=(sc,), profiles=("drop",))
