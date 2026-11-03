"""Every decision the engine makes on its own is a journal record, on the venue path as
on the in-process path: an automatic halt (the day's loss limit, the drawdown limit), a
fill past a limit, an exit the engine sends to a venue as an order, a settlement on a node
with a venue, and a venue's refusal (the `report` record names the order and the reason).

Each case replays its journal into a fresh node, which must write the same records.
"""

import json

import numpy as np
import pytest

from raptorbt import BacktestConfig, InstrumentSpec, LiveNode, SessionSchedule, Strategy, journal as rj
from raptorbt.adapters.conformance import ReferenceBinding, ScriptedVenue, _Proxy
from raptorbt.execution import ExecutionReport, PaperExecution
from raptorbt.strategy import orders

from test_journal import HEADER, MIN, MockVenue
from test_journal_completeness import Holder, Tape, _bars, _drive_bars, _factory, _scenario, replays
from test_sessions import THU, ist


def of(records, kind, **match):
    return [r for r in records if r.kind == kind and all(r.payload.get(k) == v for k, v in match.items())]


# -- an automatic halt --------------------------------------------------------------------


def _fall(cfg):
    data = {"A": _bars([100.0, 100.0, 90.0, 90.0, 90.0], spread=0.0)}
    factory = _factory(lambda: [Holder("A", units=100.0)], ["A"], cfg, bar_stamp="close")
    return _scenario(factory, lambda tape, node: _drive_bars(node, data, ["A"]), ["halt"])


def test_the_day_loss_limit_halting_a_strategy_is_a_record_of_the_engine():
    wrote = _fall(BacktestConfig(initial_capital=100_000.0, fees=0.0, max_daily_loss=500.0))
    (halt,) = of(wrote, "halt")
    assert halt.source == "engine" and halt.payload["reason"] == "daily_loss"


def test_the_drawdown_kill_switch_is_a_record_of_the_engine():
    wrote = _fall(BacktestConfig(initial_capital=20_000.0, fees=0.0, max_drawdown_pct=5.0))
    (halt,) = of(wrote, "halt")
    assert halt.source == "engine" and halt.payload["reason"] == "drawdown"


def test_a_halt_by_hand_is_recorded_once_not_again_as_the_engines():
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([Holder("A")], ["A"], BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close", journal=j)
    node.halt(0, False, "operator")
    _drive_bars(node, {"A": _bars([100.0] * 3)}, ["A"])
    halts = of(j.since(0), "halt")
    assert [(h.source, h.payload["reason"]) for h in halts] == [("host", "operator")]


# -- the venue path ------------------------------------------------------------------------


def _venue_run(execution, strategy, symbols, cfg, drive, **kw):
    factory = _factory(strategy, symbols, cfg, bar_stamp="close", execution=execution, **kw)
    return _scenario(factory, drive, [])


def test_a_session_square_off_on_a_venue_node_is_a_decision_then_the_close_order():
    ts = [ist(THU, 9, 20), ist(THU, 9, 25), ist(THU, 15, 25), ist(THU, 15, 30)]
    data = {"A": {**_bars([100.0, 101.0, 102.0, 103.0]), "timestamps": np.array(ts, dtype=np.int64)}}
    venue = MockVenue()

    def drive(tape, node):
        for i in range(4):
            b = data["A"]
            node.push_bar("A", int(b["timestamps"][i]), float(b["open"][i]), float(b["high"][i]),
                          float(b["low"][i]), float(b["close"][i]), 1000.0, arrival_ns=int(b["timestamps"][i]))
            venue.fill_all()
            venue.deliver(node, int(b["timestamps"][i]) + 1)

    kw = dict(instruments={"A": InstrumentSpec.equity("A", session=SessionSchedule.nse_equity())})
    wrote = _venue_run(venue, lambda: [Holder("A", intent=True)], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), drive, **kw)
    kinds = [r.kind for r in wrote]
    (exit_,) = of(wrote, "engine_exit", reason="squareoff")
    assert exit_.payload["venue"] is True
    after = [r for r in wrote[kinds.index("engine_exit"):] if r.kind == "adapter_request"]
    assert after and after[0].payload["request"]["client_order_id"]


def test_a_stop_the_engine_watches_for_a_venue_is_a_decision_then_the_exit_order():
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0)
    cfg.set_fixed_stop(0.05)
    data = {"A": _bars([100.0, 100.0, 100.0, 100.0], spread=0.0)}
    data["A"]["low"][2] = 90.0
    venue = MockVenue()

    def drive(tape, node):
        b = data["A"]
        for i in range(4):
            node.push_bar("A", int(b["timestamps"][i]), float(b["open"][i]), float(b["high"][i]),
                          float(b["low"][i]), float(b["close"][i]), 1000.0, arrival_ns=int(b["timestamps"][i]))
            venue.fill_all()
            venue.deliver(node, int(b["timestamps"][i]) + 1)

    wrote = _venue_run(venue, lambda: [Holder("A", intent=True)], ["A"], cfg, drive, protective="local")
    (exit_,) = of(wrote, "engine_exit", reason="watch_stop")
    assert exit_.payload["venue"] is True
    kinds = [r.kind for r in wrote]
    assert "adapter_request" in kinds[kinds.index("engine_exit"):]


def test_a_fill_past_a_limit_is_a_record():
    class Twice(Strategy):
        def on_bar(self, ctx, bar):
            if not getattr(self, "done", False):
                self.done = True
                self.submit_order(orders.Market(side="buy", units=5.0), client_id="a")
                self.submit_order(orders.Market(side="buy", units=5.0), client_id="b")

    venue = MockVenue()

    def drive(tape, node):
        node.push_bar("A", MIN, 100.0, 101.0, 99.0, 100.0, 10.0, arrival_ns=MIN)
        venue.fill_all()
        tape.step("fills", lambda: venue.deliver(node, MIN + 2), "report", "rejection")

    wrote = _venue_run(venue, lambda: [Twice()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), drive)
    (breach,) = of(wrote, "rejection", event="fill_breach")
    assert breach.source == "engine" and breach.payload["reason"] == "position_open"


def test_a_settlement_on_a_node_with_a_venue_is_a_record():
    fut = InstrumentSpec.futures_contract("A", expiration_ns=4 * MIN, lot_size=1.0, underlying="U")
    data = {"A": _bars([100.0, 101.0, 102.0, 103.0, 104.0]), "U": _bars([50.0] * 5)}
    venue = MockVenue()

    def drive(tape, node):
        for i in range(5):
            _drive_bars(node, {s: {k: v[i : i + 1] for k, v in data[s].items()} for s in data}, ["A", "U"])
            venue.fill_all()
            venue.deliver(node, (i + 1) * MIN + 1)

    wrote = _venue_run(venue, lambda: [Holder("A")], ["A", "U"], BacktestConfig(initial_capital=1e6, fees=0.0),
                       drive, instruments={"A": fut})
    assert of(wrote, "lifecycle", what="settlement")


def test_a_refusal_the_book_makes_before_sending_is_a_record_on_a_paper_node():
    class Twice(Strategy):
        def __init__(self):
            super().__init__()
            self.n = 0

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n in (1, 3):
                self.submit_order(orders.Market(side="buy", units=5.0), client_id=f"b{self.n}")

    def drive(tape, node):
        for i in range(5):
            node.push_bar("A", (i + 1) * MIN, 100.0, 101.0, 99.0, 100.0, 10.0, arrival_ns=(i + 1) * MIN)

    wrote = _venue_run(PaperExecution(), lambda: [Twice()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), drive)
    # The simulated venue's refusal is on the record, with its reason and the order it is about.
    (refused,) = of(wrote, "simulated_report", report="rejected")
    assert refused.payload["reason"] == "position_open"
    sent = [r for r in of(wrote, "adapter_request") if r.client_order_id == refused.client_order_id]
    assert sent and sent[0].seq < refused.seq
    assert len(of(wrote, "simulated_report", report="fill")) == 1


# -- a venue's refusal ----------------------------------------------------------------------


class Planned(Strategy):
    def __init__(self):
        super().__init__()
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Limit(side="buy", price=95.0, units=5.0), client_id="L1")
        elif self.n == 3:
            self.submit_order(orders.Limit(side="buy", price=94.0, units=3.0), client_id="L2")


def test_a_broker_adapters_node_journals_a_venues_refusal_with_its_reason_and_the_order_terms():
    scripted = ScriptedVenue()
    binding = ReferenceBinding(scripted, {})
    scripted.subscribers.append(binding.on_event)
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([Planned()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close",
                    execution=_Proxy(binding), journal=j)
    t = [0]

    def bar():
        t[0] += 1
        node.push_bar("A", t[0] * MIN, 100.0, 101.0, 99.0, 100.0, 10.0, arrival_ns=t[0] * MIN)

    def pump():
        binding.pump()
        for cid, report, _ in binding.drain():
            t[0] += 1
            node.push_report(cid, report, arrival_ns=t[0] * MIN)

    bar()
    bar()  # the order goes out on the next event
    scripted.accept(binding.ref_of("s0-0"))
    pump()
    bar()
    bar()
    scripted.reject(binding.ref_of("s0-1"), "insufficient funds")
    pump()
    scripted.fill(binding.ref_of("s0-0"), 5.0, 95.0)
    pump()
    records = list(j.since(0))
    (refused,) = [r for r in of(records, "report", report="rejected")]
    assert refused.client_order_id == "s0-1" and refused.payload["reason"] == "insufficient funds"
    # The order it refuses is reconstructed from the request written before it was sent.
    req = [r for r in of(records, "adapter_request") if r.client_order_id == "s0-1"][0]
    terms = req.payload["request"]
    assert (terms["symbol"], terms["side"], terms["quantity"]) == ("A", "buy", "3")
    assert "94" in json.dumps(terms)
    assert of(records, "report", report="fill")

    def revive(jr, recs):
        return LiveNode.restore(None, [Planned()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0),
                                journal=jr, records=recs, bar_stamp="close", execution=_Proxy(binding), reconcile="skip")

    back, got, want = replays(j, revive)
    assert got == want and back.state_digest() == node.state_digest()
