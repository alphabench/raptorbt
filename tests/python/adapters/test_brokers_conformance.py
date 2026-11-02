"""Each broker mapping passes the conformance kit over its synthetic server."""

import pytest

from raptorbt.adapters.conformance import SCENARIOS, run_conformance

from .fakes import dhan_binding, kite_binding, upstox_binding

BINDINGS = {"kite": kite_binding, "upstox": upstox_binding, "dhan": dhan_binding}


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize("broker", BINDINGS)
def test_broker_converges_under_every_fault(broker, sc):
    failures = run_conformance(BINDINGS[broker], scenarios=(sc,))
    assert not failures, "\n".join(map(str, failures))
