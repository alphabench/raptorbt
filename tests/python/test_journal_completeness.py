"""Journal completeness: everything that changes an order, a position or the cash
of a live node leaves a record, and a replay of the records alone rebuilds the node.

A scripted run drives one node on a venue and several in-process nodes through
each change a node can make. After every step the test asks the journal what the
step wrote, in order, and a change of the node's state with no new record fails
the step. At the end each journal is replayed into a fresh node, which must reach
the same state and write the same records, byte for byte.

A step that is only an input (a bar, a report) is its own record; the engine's own
decisions without an input of their own (a refusal, an order cut, a position it
closed by a rule, a settlement, a corporate action, an adopted position) have
kinds of their own, asserted here.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    CorporateAction,
    InstrumentSpec,
    LiveNode,
    Strategy,
    journal as rj,
)
from raptorbt.execution import ExecutionReport, PaperExecution
from raptorbt.strategy import orders
from raptorbt.strategy.exits import ExitEvents, ExitRule

from test_journal import HEADER, MIN, MockVenue
from test_journal_start_and_resize import Bracketer, NS, TICKS, _cfg as oco_cfg


class Tape:
    """Runs steps against a node and checks what each wrote."""

    def __init__(self, node, journal):
        self.node, self.journal = node, journal

    def step(self, label, fn, *expect, raises=False):
        before, digest = self.journal.count, self.node.state_digest()
        if raises:
            with pytest.raises(Exception):
                fn()
        else:
            fn()
        wrote = [r.kind for r in self.journal.since(before)]
        it = iter(wrote)
        assert all(k in it for k in expect), f"{label}: wanted {list(expect)} in order, wrote {wrote}"
        if self.node.state_digest() != digest:
            assert wrote, f"{label}: the node changed and wrote nothing"
        return list(self.journal.since(before))


def replays(journal, factory):
    """Rebuild a node from the journal's records alone; it must write the same."""
    records = list(journal.since(0))
    j2 = rj.Journal(HEADER, memory=True)
    back = factory(j2, records)
    return back, [r.canonical() for r in j2.since(0)][: len(records)], [r.canonical() for r in records]


# -- a venue node -------------------------------------------------------------------------


def _twice(s):
    s.submit_order(orders.Limit(side="buy", price=80.0, units=1.0), client_id="D")
    s.submit_order(orders.Limit(side="buy", price=81.0, units=2.0), client_id="D")


#: What the strategy does on its Nth live bar. It lives in the strategy, not in the
#: test, so that a replay (which re-runs the strategy) makes the same commands.
PLAN = {
    1: lambda s: s.submit_order(orders.Limit(side="buy", price=90.0, units=5.0), client_id="L1"),
    2: _twice,
    3: lambda s: s.submit_order(orders.Market(side="buy", units=999.0), client_id="big"),
    4: lambda s: s.modify_order("L1", limit_price=91.0),
    5: lambda s: s.cancel_order("L1"),
    6: lambda s: s.submit_order(orders.Market(side="buy", units=5.0), client_id="M1"),
    7: lambda s: s.submit_order(orders.Limit(side="buy", price=90.0, units=5.0), client_id="R1"),
    8: lambda s: s.submit_order(orders.Limit(side="buy", price=90.0, units=6.0), client_id="P1"),
}


class Actor(Strategy):
    """Orders on every warm-up bar (muted, so discarded), then follows ``PLAN``."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def on_bar(self, ctx, bar):
        if ctx.is_warmup:
            self.submit_order(orders.Market(side="buy", units=1.0))
            return
        self.n += 1
        if self.n in PLAN:
            PLAN[self.n](self)

    def state(self):
        return {"n": self.n}

    def load_state(self, state):
        self.n = state["n"]


def _history(n=3):
    p = np.full(n, 100.0)
    return {"A": dict(timestamps=np.arange(1, n + 1) * MIN, open=p, high=p + 1, low=p - 1,
                      close=p, volume=np.full(n, 10.0))}


SEED = {0: {"A": {"quantity": 10.0, "avg_price": 100.0, "timestamp_ns": 1}}}


def _lot_check(order, ctx):
    return "lot_limit" if order.units and order.units > 100 else None


def _venue_node(venue, journal, records=None):
    kw = dict(
        bar_stamp="close", execution=venue, journal=journal, order_check=_lot_check,
        warmup="muted", warmup_bars=_history(),
    )
    if records is None:
        return LiveNode([Actor()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), **kw)
    # The scripted busts and corrections leave the mock venue's own books apart from the
    # node's, so the rebuild does not ask the venue; what is compared is the journal.
    return LiveNode.restore(
        None, [Actor()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), records=records,
        reconcile="skip", **kw
    )


def test_every_change_a_venue_node_makes_leaves_a_record():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = _venue_node(venue, j)
    s = node.strategies[0]
    tape = Tape(node, j)
    t = [10]

    def bar():
        t[0] += 1
        node.push_bar("A", t[0] * MIN, 100.0, 101.0, 99.0, 100.0, 10.0, arrival_ns=t[0] * MIN)

    # Build: warm-up is muted, so its bars are records (flagged) and nothing else, and
    # nothing reaches the venue although the strategy ordered on every one of them.
    built = [r.kind for r in j.since(0)]
    assert built[0] == "start" and set(built[1:]) == {"bar"}
    assert all(r.payload["warmup"] for r in j.since(1)) and venue.sent == []

    tape.step("submit", bar, "bar", "command", "adapter_request", "adapter_result")
    l1 = venue.sent[-1]
    tape.step("refusal", bar, "bar", "command", "refusal", "error", raises=True)
    assert [r.payload["code"] for r in j.since(0) if r.kind == "refusal"][-1] == "duplicate_client_order_id"
    tape.step("order_check refusal", bar, "bar", "command", "refusal")
    assert [r.payload["message"] for r in j.since(0) if r.kind == "refusal"][-1] == "order_check"
    venue.deliver(node, t[0] * MIN + 1)  # the venue accepts what it holds
    tape.step("resend", lambda: node.resend(l1), "annotation", "adapter_request")
    tape.step("modify", bar, "bar", "command", "adapter_request")
    tape.step("cancel", bar, "bar", "command", "adapter_request", "adapter_result")
    venue.deliver(node, t[0] * MIN + 2)

    def market():
        bar()
        venue.fill_all()
        venue.deliver(node, t[0] * MIN + 3)

    tape.step("fill", market, "bar", "command", "adapter_request", "report")
    m1 = venue.sent[-1]
    # The book refuses a second entry on the side the position holds.
    tape.step("book rejection", bar, "bar", "command", "rejection")
    rej = [r for r in j.since(0) if r.kind == "rejection"][-1]
    assert (rej.client_order_id, rej.payload["reason"]) == ("R1", "position_open")
    corrected = ExecutionReport.correction(f"X-{m1}", 5.0, 100.5, t[0] * MIN + 5)
    tape.step("correction", lambda: node.push_report(m1, corrected, arrival_ns=t[0] * MIN + 5), "report", "restatement")
    bust = ExecutionReport.bust(f"X-{m1}", t[0] * MIN + 6)
    tape.step("bust", lambda: node.push_report(m1, bust, arrival_ns=t[0] * MIN + 6), "report", "restatement")

    bar()  # P1
    cid = venue.sent[-1]
    venue.deliver(node, t[0] * MIN + 7)
    fill = ExecutionReport.fill("E1", 2.0, 90.0, t[0] * MIN + 8)
    tape.step("partial fill", lambda: node.push_report(cid, fill, arrival_ns=t[0] * MIN + 8), "report")
    tape.step("halt", lambda: node.halt(0, False, "operator"), "halt")
    tape.step("note", lambda: node.annotate(t[0] * MIN + 7, "shift", {"who": "ops"}), "annotation")

    def revive(jr, records):
        return _venue_node(venue, jr, records)

    back, got, want = replays(j, revive)
    assert got == want and back.state_digest() == node.state_digest()


def test_a_position_adopted_at_build_leaves_a_record_and_replays():
    factory = _factory(lambda: [Holder("A")], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0),
                       bar_stamp="close", initial_positions=SEED)
    wrote = _scenario(factory, lambda tape, node: _drive_bars(node, {"A": _bars([100.0] * 3)}, ["A"]), ["adoption"])
    assert [r.kind for r in wrote][:2] == ["start", "adoption"]


def test_a_fill_the_book_cannot_book_yet_leaves_a_record():
    class Both(Strategy):
        def on_bar(self, ctx, bar):
            if not getattr(self, "sent", False):
                self.sent = True
                self.submit_order(orders.Market(side="buy", units=5.0), client_id="in")
                self.submit_order(orders.Market(side="sell", units=5.0, reduce_only=True), client_id="out")

    venue, j = MockVenue(), rj.Journal(HEADER, memory=True)
    node = LiveNode([Both()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close", execution=venue, journal=j)
    tape = Tape(node, j)
    tape.step("both", lambda: node.push_bar("A", MIN, 100.0, 101.0, 99.0, 100.5, 1000.0), "bar", "command")
    _, out = sorted(venue.sent)
    tape.step("unbooked fill", lambda: node.push_report(out, ExecutionReport.fill("X1", 5.0, 101.0, MIN), arrival_ns=MIN),
              "report", "unbooked_fill")


# -- in-process nodes ---------------------------------------------------------------------


def _bars(closes, step=MIN, start=MIN, spread=1.0):
    c = np.asarray(closes, dtype=np.float64)
    return {"timestamps": start + np.arange(len(c), dtype=np.int64) * step, "open": c, "high": c + spread,
            "low": c - spread, "close": c, "volume": np.full(len(c), 1000.0)}


def _drive_bars(node, data, symbols):
    n = len(next(iter(data.values()))["close"])
    for i in range(n):
        for sym in symbols:
            b = data[sym]
            node.push_bar(sym, int(b["timestamps"][i]), *(float(b[c][i]) for c in ("open", "high", "low", "close", "volume")),
                          arrival_ns=int(b["timestamps"][i]))


class Holder(Strategy):
    """Enters ``sym`` on its first bar: with a market order, or with an entry
    intent, whose position the engine's own stops and square-off manage."""

    def __init__(self, sym, units=10.0, intent=False):
        super().__init__()
        self.sym, self.units, self.intent, self.done = sym, units, intent, False

    def on_bar(self, ctx, bar):
        if not self.done and ctx.symbol == self.sym:
            self.done = True
            if self.intent:
                self.enter()
            else:
                self.submit_order(orders.Market(side="buy", units=self.units))


def _factory(strategies, symbols, cfg, **kw):
    """Build a journaled node, or restore one from records, with the same arguments."""

    def factory(journal, records=None):
        if records is None:
            return LiveNode(strategies(), symbols, cfg, journal=journal, **kw)
        return LiveNode.restore(None, strategies(), symbols, cfg, journal=journal, records=records, **kw)

    return factory


def _scenario(factory, drive, expect):
    """Run ``drive`` on a fresh journaled node, check the kinds it wrote, replay it."""
    j = rj.Journal(HEADER, memory=True)
    node = factory(j)
    drive(Tape(node, j), node)
    wrote = [r.kind for r in j.since(0)]
    for kind in expect:
        assert kind in wrote, f"{kind} missing from {sorted(set(wrote))}"
    back, got, want = replays(j, factory)
    assert got == want and back.state_digest() == node.state_digest()
    return list(j.since(0))


def test_an_oco_cut_leaves_a_record():
    def drive(tape, node):
        for i, (ltp, ltq) in enumerate(TICKS):
            tape.step(f"tick {i}", lambda: node.push_tick("A", i * NS, ltp, ltp - 0.5, ltp + 0.5, ltq=ltq), "tick")

    wrote = _scenario(_factory(lambda: [Bracketer()], ["A"], oco_cfg()), drive, ["order_resized", "command"])
    assert [r.kind for r in wrote].count("order_resized") == 1


@pytest.mark.parametrize("case", ["stop_loss", "squareoff", "settlement", "corporate_action"])
def test_an_exit_the_engine_makes_on_its_own_leaves_a_record(case):
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0)
    symbols, kw = ["A"], {"bar_stamp": "close"}
    data = {"A": _bars([100.0, 100.0, 100.0, 100.0, 100.0])}
    expect = {"stop_loss": ("engine_exit", "stop_loss"), "squareoff": ("engine_exit", "squareoff"),
              "settlement": ("lifecycle", "settlement"), "corporate_action": ("lifecycle", "corporate_action")}[case]
    if case == "stop_loss":
        cfg.set_fixed_stop(0.05)
        data["A"]["low"][3] = 90.0
    elif case == "squareoff":
        from raptorbt import SessionSchedule
        from test_sessions import THU, ist

        # The entry fills on the second bar; the third is past the 15:20 square-off.
        ts = [ist(THU, 9, 20), ist(THU, 9, 25), ist(THU, 15, 25), ist(THU, 15, 30)]
        data = {"A": {**_bars([100.0, 101.0, 102.0, 103.0]), "timestamps": np.array(ts, dtype=np.int64)}}
        kw["instruments"] = {"A": InstrumentSpec.equity("A", session=SessionSchedule.nse_equity())}
    elif case == "settlement":
        fut = InstrumentSpec.futures_contract("A", expiration_ns=4 * MIN, lot_size=1.0, underlying="U")
        symbols, data = ["A", "U"], {"A": _bars([100.0, 101.0, 102.0, 103.0, 104.0]), "U": _bars([50.0] * 5)}
        kw["instruments"] = {"A": fut}
    else:
        kw.update(corporate_actions=[CorporateAction.split("A", 3 * MIN, 2.0)], adjustment_policy="raw")

    wrote = _scenario(_factory(lambda: [Holder("A", intent=case in ("stop_loss", "squareoff"))], symbols, cfg, **kw),
                      lambda tape, node: _drive_bars(node, data, symbols), [expect[0]])
    rec = next(r for r in wrote if r.kind == expect[0])
    assert expect[1] in (rec.payload.get("reason"), rec.payload.get("what")), rec.payload


def test_an_exit_rules_order_leaves_a_record():
    class Mixed(ExitEvents, Strategy):
        def __init__(self):
            super().__init__()
            self.n = 0

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Market(side="buy", units=5.0), client_id="in")
            if self.n == 2:
                self.set_exit(ctx, ExitRule(stop_loss_pct=1), symbol="A")

    def drive(tape, node):
        for i, px in enumerate([100, 100, 100, 90, 90]):
            node.push_bar("A", (i + 1) * MIN, px, px + 1, px - 1, px, 10.0, arrival_ns=(i + 1) * MIN)

    factory = _factory(lambda: [Mixed()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0),
                       bar_stamp="close", execution=PaperExecution())
    wrote = _scenario(factory, drive, ["command", "adapter_request"])
    sent = [c["client_id"] for r in wrote if r.kind == "command" for c in r.payload["commands"] if c["command"] == "submit"]
    assert any(":x" in cid for cid in sent), sent


def test_a_structures_legs_leave_a_record_each():
    import test_structures as ts

    cls = ts.make(ts.st.iron_condor(1, 2), exit=ts.st.Exit(close_at_ns=ts.T0 + 25 * ts.STEP))
    data, inst = ts.universe()
    symbols = list(data)

    def drive(tape, node):
        for i in range(ts.N_BARS):
            for sym in symbols:
                b = data[sym]
                node.push_bar(sym, int(b["timestamps"][i]), float(b["open"][i]), float(b["high"][i]),
                              float(b["low"][i]), float(b["close"][i]), float(b["volume"][i]),
                              arrival_ns=int(b["timestamps"][i]))

    factory = _factory(lambda: [cls()], symbols, BacktestConfig(initial_capital=2_000_000.0, fees=0.0),
                       instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    wrote = _scenario(factory, drive, ["command"])
    legs = [c for r in wrote if r.kind == "command" for c in r.payload["commands"] if c["command"] == "submit"]
    assert len(legs) >= 4, "an iron condor is four legs"
