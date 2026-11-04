"""The journal, checkpoint, recovery and fencing of a live node.

* the journal records every input, report, command (refusals included),
  adapter request (before the adapter is called) and adapter failure, and the
  same inputs give the same chain head;
* a node killed at **every** journal offset of a scripted run, rebuilt from
  what was written, ends in the same account state with no order sent twice;
* a venue that drops, duplicates and reorders leaves recovery consistent;
* an epoch fences a stale node; corrections and busts restate the book.
"""

import json
import random

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.errors import InvalidData, InvalidParameter, StateError
from raptorbt.execution import ExecutionReport, Fenced, OrderRefused, PaperExecution
from raptorbt.strategy import orders

MIN = 60_000_000_000


# -- a venue that lives outside the node -------------------------------------------


class MockVenue:
    """An exchange on the far side of a connection. It outlives the node.

    It fills market orders at the next price it is told about, answers
    ``open_orders`` / ``positions`` / ``balances`` for reconciliation, and
    counts what it was sent so a test can assert nothing was sent twice.
    """

    can_fence = True
    namespace = "T1"  # an adapter marks its node's orders; a node refuses one that does not
    keeps_own_records = True  # a test venue: it remembers the node's orders itself

    def __init__(self):
        self.orders = {}  # client_order_id -> dict
        self.sent = []  # every submit, in order
        self.cancels = []
        self.outbox = []  # (client_order_id, report) not yet delivered
        self.epoch = 0
        self.position = {}
        self.cash = 1_000_000.0
        self.fail_next = None  # an exception to raise instead of accepting
        self.fail_after_accept = None  # accept, then raise (the ack was lost)
        self.execs = {}

    # adapter protocol
    def submit(self, request):
        if self.fail_next is not None:
            err, self.fail_next = self.fail_next, None
            raise err
        cid = request.client_order_id
        self.sent.append(cid)
        if cid in self.orders:  # idempotent on the client id, as venues are
            return
        self.orders[cid] = {
            "client_order_id": cid,
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
            "reduce_only": request.reduce_only,
            "filled_qty": 0.0,
            "open": True,
            "ts": request.submitted_ns,
        }
        self.outbox.append((cid, ExecutionReport.accepted("V-" + cid, request.submitted_ns)))
        if self.fail_after_accept is not None:
            err, self.fail_after_accept = self.fail_after_accept, None
            raise err

    def cancel(self, client_order_id):
        self.cancels.append(client_order_id)
        o = self.orders.get(client_order_id)
        if o and o["open"]:
            o["open"] = False
            self.outbox.append((client_order_id, ExecutionReport.cancel_ack(o["ts"] + 1)))

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        pass

    def fence(self, epoch):
        if epoch < self.epoch:
            raise Fenced(f"epoch {self.epoch} owns the account")
        self.epoch = epoch

    # reconciliation
    def open_orders(self):
        return [dict(o) for o in self.orders.values() if o["open"]]

    def positions(self):
        return dict(self.position)

    def balances(self):
        return {"cash": self.cash}

    # the exchange's side
    def fill_all(self, price=None, ts=None):
        """Fill every open order. The price and time depend only on the order
        (as a venue's would on its own book), so a late fill is the same fill."""
        for cid, o in self.orders.items():
            if o["open"]:
                o["open"] = False
                q = o["quantity"]
                sign = 1 if o["side"] == "buy" else -1
                held = self.position.get(o["symbol"], 0.0)
                if o["reduce_only"] and held * sign >= 0:
                    # A venue does not fill a reduce-only order that cannot reduce.
                    self.outbox.append((cid, ExecutionReport.rejected("reduce_only", o["ts"] + 1000)))
                    continue
                o["filled_qty"] = q
                self.position[o["symbol"]] = self.position.get(o["symbol"], 0.0) + sign * q
                px = 100.0 + (o["ts"] // MIN) % 5
                self.cash -= sign * q * px
                self.outbox.append((cid, ExecutionReport.fill(f"X-{cid}", q, px, o["ts"] + 1000)))

    def deliver(self, node, arrival, shuffle=None, dup=False):
        """Hand the node every report it has not taken. A report leaves the
        outbox only once the node took it, so a node that dies keeps its mail."""
        batch = list(self.outbox)
        if shuffle is not None:
            # Reports of one order come in any order; one order's reports do not
            # overtake an earlier order's (a venue sequences its own book).
            by_order = {}
            for item in batch:
                by_order.setdefault(item[0], []).append(item)
            batch = []
            for group in by_order.values():
                shuffle.shuffle(group)
                batch.extend(group)
        for k, (cid, report) in enumerate(batch):
            node.push_report(cid, report, arrival_ns=arrival + k)
            if dup:
                node.push_report(cid, report, arrival_ns=arrival + k)
            self.outbox.remove((cid, report))


class Trader(Strategy):
    """Buys and sells market orders on a rhythm, and remembers how many."""

    def __init__(self):
        super().__init__()
        self.count = {}

    def on_bar(self, ctx, bar):
        n = self.count[ctx.symbol] = self.count.get(ctx.symbol, 0) + 1
        if n % 7 == 2 and not ctx.positions():
            # one 5-lot at most per symbol (a netting book would add to a held position)
            self.submit_order(orders.Market(side="buy", units=5.0))
        elif n % 7 == 5:
            self.submit_order(orders.Market(side="sell", units=5.0, reduce_only=True))

    def state(self):
        return {"count": self.count}

    def load_state(self, state):
        self.count = state["count"]


class StartTrader(Trader):
    """Trades the same rhythm and also orders once in ``on_start``."""

    def on_start(self, ctx):
        self.submit_order(orders.Market(side="buy", units=5.0), symbol="A")


CONFIG = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
SYMBOLS = ["A", "B"]
HEADER = {"run_id": "r-1", "algo": "trader", "version": 3}


def build(venue, journal=None, strategy=Trader, **kw):
    return LiveNode(
        [strategy()], SYMBOLS, CONFIG, bar_stamp="close", execution=venue, journal=journal,
        epoch=kw.pop("epoch", None), **kw,
    )


def script(n=40):
    """The feed of a run: bars on two symbols, and 'tick marks' where the venue
    acts on what it holds (fills open orders at the latest close)."""
    steps = []
    for i in range(n):
        for s in SYMBOLS:
            price = 100.0 + (i % 9) + (0.5 if s == "B" else 0.0)
            steps.append(("bar", s, (i + 1) * MIN, price, price + 1, price - 1, price + 0.5, 1000.0))
        if i % 3 == 1:
            steps.append(("venue", 100.5 + (i % 9), (i + 1) * MIN + 1000))
    return steps


def run_step(node, venue, step, arrival=None):
    if step[0] == "bar":
        node.push_bar(step[1], step[2], *step[3:8], arrival_ns=step[2] + 7)
    else:
        venue.fill_all()
        venue.deliver(node, step[2] + 500)


def snapshot(node, venue):
    return {"digest": node.state_digest(), "sent": sorted(venue.sent), "positions": dict(venue.position)}


# -- the chain -----------------------------------------------------------------------


def test_chain_step_is_pure_and_links_canonical_bytes():
    j = rj.Journal(HEADER, memory=True)
    head = rj.chain_start(HEADER)
    j.append("annotation", 5, {"kind": "x", "detail": {"price": 100.5}})
    j.append("annotation", 6, {"kind": "y", "detail": None})
    records = j.since(0)
    for r in records:
        head = rj.chain_step(head, r.canonical())
        assert head == r.chain
    assert head == j.head()[1]
    # Re-encoding what a record says gives the bytes the chain was taken over.
    r = records[0]
    again = rj.canonical(
        {
            "seq": r.seq, "kind": r.kind, "ts_event": r.ts_event, "ts_arrival": r.ts_arrival,
            "source": r.source, "client_order_id": r.client_order_id, "payload": r.payload,
        }
    )
    assert again == r.canonical()
    assert json.loads(r.canonical())["payload"]["detail"]["price"] == "100.5"  # no float


def test_the_same_inputs_give_the_same_chain_head_and_a_different_input_does_not():
    heads = []
    for tweak in (0.0, 0.0, 0.01):
        venue = MockVenue()
        j = rj.Journal(HEADER, memory=True)
        node = build(venue, j)
        for step in script(12):
            if step[0] == "bar" and step[2] == 5 * MIN and tweak:
                step = step[:3] + (step[3] + tweak,) + step[4:]
            run_step(node, venue, step)
        heads.append(j.head()[:2])
    assert heads[0] == heads[1]
    assert heads[0] != heads[2]


def test_a_continued_run_carries_the_sequence_and_the_chain():
    j1 = rj.Journal({"run": 1}, memory=True)
    node = build(MockVenue(), j1)
    for step in script(4):
        if step[0] == "bar":
            run_step(node, MockVenue(), step)
    count, chain, _ = j1.head()
    j2 = rj.Journal({"run": 2}, seq_base=count, chain=chain, memory=True)
    j2.append("annotation", 1, {"kind": "next", "detail": None})
    assert j2.since(count)[0].seq == count
    assert rj.chain_step(chain, j2.since(count)[0].canonical()) == j2.head()[1]


# -- what is recorded ------------------------------------------------------------------


def kinds(j):
    return [r.kind for r in j.since(0)]


def test_every_kind_of_input_and_decision_is_recorded_in_order():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j)
    node.annotate(7, "adopt", {"symbol": "A", "qty": 3})
    for step in script(4):
        run_step(node, venue, step)
    node.push_tick("A", 9 * MIN, 101.0, 100.9, 101.1)
    node.push_depth("A", 9 * MIN + 1, [(100.0, 5.0)], [(101.0, 5.0)])
    node.advance_time(10 * MIN)
    node.halt(0, flatten=False, reason="operator")
    got = kinds(j)
    for k in ("start", "annotation", "bar", "command", "adapter_request", "adapter_result",
              "report", "tick", "depth", "time", "halt"):
        assert k in got, (k, got)
    assert got[0] == "start"
    # The request precedes the venue ever hearing of the order.
    req = next(r for r in j.since(0) if r.kind == "adapter_request")
    assert req.client_order_id == "s0-0"
    assert req.payload["request"]["quantity"] == "5"
    halt = next(r for r in j.since(0) if r.kind == "halt")
    assert halt.source == "host"
    ann = next(r for r in j.since(0) if r.kind == "annotation")
    assert ann.ts_event == 7 and ann.payload["detail"] == {"symbol": "A", "qty": 3}
    # Arrival time is the one given, event time the bar's close.
    bar0 = next(r for r in j.since(0) if r.kind == "bar")
    assert bar0.ts_arrival == bar0.ts_event + 7


def test_the_request_is_in_the_journal_before_the_adapter_is_called():
    seen = []

    class Spy(MockVenue):
        def submit(self, request):
            seen.append([r.kind for r in j.since(0)][-1])
            super().submit(request)

    venue = Spy()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j)
    for step in script(4):
        run_step(node, venue, step)
    assert seen and set(seen) == {"adapter_request"}


def test_a_failing_adapter_is_recorded_and_the_order_is_not_resent():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j)
    venue.fail_next = ConnectionError("link down")
    with pytest.raises(ConnectionError):
        for step in script(6):
            run_step(node, venue, step)
    res = [r for r in j.since(0) if r.kind == "adapter_result"]
    assert res[-1].payload["status"] == "error" and "link down" in res[-1].payload["message"]
    assert venue.sent == []
    assert node.health()["unconfirmed"] == ["s0-0"]
    # The push that raised left a note, so a replay expects the same failure.
    assert kinds(j)[-1] == "error"
    assert j.since(0)[-1].payload["op"] == "bar"


def test_a_refused_order_and_a_refused_command_are_recorded():
    class Refuser(MockVenue):
        def submit(self, request):
            raise OrderRefused("closed")

    venue = Refuser()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j)
    for step in script(4):
        run_step(node, venue, step)
    res = [r.payload["status"] for r in j.since(0) if r.kind == "adapter_result"]
    assert res and set(res) == {"refused"}

    class Bad(Strategy):
        def on_bar(self, ctx, bar):
            self.submit_order(orders.Market(side="buy", units=1.0), client_id="dup")
            self.submit_order(orders.Limit(side="buy", price=1.0, units=2.0), client_id="dup")

    j2 = rj.Journal(HEADER, memory=True)
    n2 = LiveNode([Bad()], ["A"], CONFIG, bar_stamp="close", execution=MockVenue(), journal=j2)
    with pytest.raises(Exception):
        n2.push_bar("A", MIN, 100.0, 101.0, 99.0, 100.5, 10.0)
    assert "refusal" in kinds(j2)
    r = next(r for r in j2.since(0) if r.kind == "refusal")
    assert r.payload["code"] == "duplicate_client_order_id"


def test_a_report_for_an_unknown_order_is_journaled_and_a_push_refused_before_it_is_leaves_no_record():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j)
    run_step(node, venue, script(1)[1])
    n0 = j.count
    with pytest.raises(Exception):  # report for an order that was never sent
        node.push_report("s0-99", ExecutionReport.cancel_ack(1), arrival_ns=1)
    # Every inbound report is written first: the report, then the failure it caused.
    assert [r.kind for r in j.since(n0)] == ["report", "error"]
    n0 = j.count
    node.push_bar("A", 5 * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
    n0 = j.count
    with pytest.raises(Exception):  # time going backwards is refused before the journal too
        node.push_bar("A", MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert j.count == n0


def test_a_journal_off_node_records_nothing_and_still_runs():
    venue = MockVenue()
    node = build(venue, None)
    for step in script(8):
        run_step(node, venue, step)
    assert node.journal is None
    with pytest.raises(InvalidParameter):
        node.annotate(1, "x")
    with pytest.raises(StateError):
        node.checkpoint()


# -- file sink -------------------------------------------------------------------------


def test_a_file_journal_reads_back_and_a_torn_tail_is_cut(tmp_path):
    path = tmp_path / "run.journal"
    venue = MockVenue()
    j = rj.Journal(HEADER, file=str(path), fsync="orders", memory=True)
    node = build(venue, j)
    for step in script(10):
        run_step(node, venue, step)
    j.sync()
    header, base, chain_from, records, torn = rj.read(str(path))
    assert header == HEADER and base == 0 and torn == 0
    assert [r.canonical() for r in records] == [r.canonical() for r in j.since(0)]
    assert rj.verify(records, header) == j.head()[1]
    data = path.read_bytes()
    path.write_bytes(data + b"\x40\x00\x00")  # a frame header cut short
    assert rj.read(str(path))[4] == 3
    flipped = bytearray(data)
    flipped[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(flipped))
    with pytest.raises(Exception):
        rj.read(str(path))


def test_a_sink_callback_is_the_durable_store_and_its_failure_stops_the_node():
    stored = []
    venue = MockVenue()
    j = rj.Journal(HEADER, sink=lambda r: stored.append(r.canonical()))
    node = build(venue, j)
    for step in script(6):
        run_step(node, venue, step)
    assert len(stored) == j.count

    def refuse(_):
        raise OSError("disk full")

    j2 = rj.Journal(HEADER, sink=refuse)
    with pytest.raises(Exception) as e:
        build(MockVenue(), j2)
    assert getattr(e.value, "code", None) == "journal_failed"
    assert j2.broken


def test_a_bounded_memory_says_what_it_dropped():
    j = rj.Journal(HEADER, memory=5)
    node = build(MockVenue(), j)
    for step in script(6):
        if step[0] == "bar":
            node.push_bar(step[1], step[2], *step[3:8])
    assert len(j.since(j.count - 5)) == 5
    with pytest.raises(Exception) as e:
        j.since(0)
    assert getattr(e.value, "code", None) == "journal_evicted"


# -- crash anywhere, recover, end in the same place --------------------------------------


class Crash(Exception):
    pass


def baseline(strategy=Trader):
    venue = MockVenue()
    store = []
    j = rj.Journal(HEADER, sink=lambda r: store.append(r.canonical()))
    node = build(venue, j, strategy)
    steps = script()
    for step in steps:
        run_step(node, venue, step)
    run_step(node, venue, ("venue", 0, steps[-1][2] + 10 * MIN))
    return snapshot(node, venue), len(store), venue


def run_with_crash(k, hostile=False, strategy=Trader):
    """Run the script with a process that dies when it tries to write record
    ``k``; rebuild a node from what was written; finish the script."""
    venue = MockVenue()
    store = []

    def sink(rec):
        if len(store) >= k:
            raise Crash("killed")
        store.append(rec.canonical())

    steps = script() + [("venue", 0, 0)]
    node = None
    done = 0
    try:
        node = build(venue, rj.Journal(HEADER, sink=sink), strategy)
        for done, step in enumerate(steps):
            run_step(node, venue, step)
        return snapshot(node, venue), None
    except Exception as e:
        assert getattr(e, "code", None) == "journal_failed" or "killed" in str(e), repr(e)

    # The process is gone; only `store` (the durable journal) and the venue remain.
    if not store:
        node = build(venue, rj.Journal(HEADER, memory=True), strategy)
        done = 0
    else:
        j = rj.Journal(HEADER, memory=True)
        node = LiveNode.restore(
            None, [strategy()], SYMBOLS, CONFIG, journal=j, records=list(store),
            execution=venue, bar_stamp="close", reconcile=True,
        )
        settle(node, venue)
        assert node.state == "live"
        bars_done = sum(1 for c in store if json.loads(c)["kind"] == "bar")
        # The recovery finished the step of the last bar the journal holds.
        done = 0
        if bars_done:
            done = 1 + [i for i, st in enumerate(steps) if st[0] == "bar"][bars_done - 1]
    for step in steps[done:]:
        run_step(node, venue, step)
    return snapshot(node, venue), node


def settle(node, venue):
    """What a host does with a recovery's discrepancies: an order the venue does
    not hold whose request had no confirmed outcome is sent again, once, by id;
    anything the venue did while the node was down arrives as the reports the
    venue still holds; then ask again."""
    for _ in range(6):
        if node.state == "live":
            return
        found = node.reconcile()
        if not found:
            node.resume()
            return
        for d in found:
            assert d.kind in ("missing_order", "order_mismatch", "position_mismatch", "cash_mismatch"), d
            if d.kind == "missing_order" and "no confirmed outcome" in d.detail and d.client_order_id not in venue.orders:
                node.resend(d.client_order_id)
        venue.deliver(node, 10**15)
    raise AssertionError(f"still recovering: {node.reconcile()}")


def test_killed_at_every_journal_offset_the_node_recovers_to_the_same_account_without_a_duplicate_order():
    want, total, _ = baseline()
    assert total > 150
    # Every offset, including the one just past the last record (a clean stop).
    for k in range(total + 1):
        got, _ = run_with_crash(k)
        assert got["digest"] == want["digest"], f"offset {k}"
        assert got["positions"] == want["positions"], f"offset {k}"
        assert len(got["sent"]) == len(set(got["sent"])), f"offset {k}: an order was sent twice"
        assert got["sent"] == want["sent"], f"offset {k}"


def test_a_strategy_that_orders_in_on_start_is_recovered_from_every_offset_too():
    want, total, _ = baseline(StartTrader)
    assert total > 150
    for k in range(total + 1):
        got, _ = run_with_crash(k, strategy=StartTrader)
        assert got["digest"] == want["digest"], f"offset {k}"
        assert got["positions"] == want["positions"], f"offset {k}"
        assert got["sent"] == want["sent"], f"offset {k}"
        assert len(got["sent"]) == len(set(got["sent"])), f"offset {k}: an order was sent twice"
    assert want["sent"][0] == "s0-0" and want["positions"], "the order made at start was sent"


def test_a_hostile_venue_that_drops_duplicates_and_reorders_leaves_recovery_consistent():
    """Reports arrive twice and out of order; some never arrive (the outbox keeps
    them for the next delivery). The node dies at a sample of offsets and the
    recovered book agrees with the venue.

    The book's own pre-trade rules read the positions it knows: an entry is not
    sent while a late report still hides that the position is closed, so a run
    with late mail is not the run with prompt mail, and a recovered run (which
    sees the mail the venue held at the crash) is not the uninterrupted one."""
    steps = script() + [("venue", 0, 0)]

    def plan(seed):
        rng = random.Random(seed)
        return [(rng.random() < 0.4, rng.randrange(10**9), rng.random() < 0.5) for _ in steps]

    def hostile_step(node, venue, step, mail):
        if step[0] == "bar":
            node.push_bar(step[1], step[2], *step[3:8], arrival_ns=step[2] + 7)
        else:
            venue.fill_all()
            late, shuffle, dup = mail
            if late:
                return  # the mail is late: nothing arrives at this step
            venue.deliver(node, step[2] + 500, shuffle=random.Random(shuffle), dup=dup)
        # The strategy never holds more than one 5-lot per symbol: the venue must not either.
        for sym, qty in venue.position.items():
            assert 0.0 <= qty <= 5.0, f"{sym} position {qty} is outside the strategy's bound"

    def entries(venue):
        """Entries sent per (symbol, decision bar)."""
        out = {}
        for o in venue.orders.values():
            if o["side"] == "buy":
                key = (o["symbol"], o["ts"] // MIN)
                out[key] = out.get(key, 0) + 1
        return out

    for seed in range(6):
        mail = plan(seed)
        venue = MockVenue()
        store = []
        total = []

        def count(r):
            total.append(1)
            store.append(r.canonical())

        node = build(venue, rj.Journal(HEADER, sink=count))
        for step, m in zip(steps, mail):
            hostile_step(node, venue, step, m)
        n = len(store)  # the records of the steps, before the last delivery
        # Whatever the venue still holds arrives, once, at the end.
        venue.deliver(node, 10**15)
        want_entries = entries(venue)
        assert max(want_entries.values()) == 1, "one entry per decision bar"
        # Kill it at a spread of offsets of this hostile run and recover.
        for k in sorted({1, 2, n // 7, n // 3, n // 2, (2 * n) // 3, n - 3, n - 1}):
            venue2 = MockVenue()
            kept = []

            def sink(r, kept=kept, k=k):
                if len(kept) >= k:
                    raise Crash("killed")
                kept.append(r.canonical())

            node2 = build(venue2, rj.Journal(HEADER, sink=sink))
            try:
                for step, m in zip(steps, mail):
                    hostile_step(node2, venue2, step, m)
            except Exception as e:
                # The crash is the one injected, not some other failure.
                assert getattr(e, "code", None) == "journal_failed" or "killed" in str(e), repr(e)
            else:
                # The write that failed was an adapter result, which a node swallows (the
                # request is durable; recovery settles it as an unknown outcome): the
                # journal must nevertheless be the broken one.
                assert node2.journal.broken, f"seed {seed}: no crash at record {k} of {n}"
            j = rj.Journal(HEADER, memory=True)
            node2 = LiveNode.restore(
                None, [Trader()], SYMBOLS, CONFIG, journal=j, records=list(kept),
                execution=venue2, bar_stamp="close",
            )
            settle(node2, venue2)
            bars_done = sum(1 for c in kept if json.loads(c)["kind"] == "bar")
            done = 0
            if bars_done:
                done = 1 + [i for i, st in enumerate(steps) if st[0] == "bar"][bars_done - 1]
            for step, m in zip(steps[done:], mail[done:]):
                hostile_step(node2, venue2, step, m)
            venue2.deliver(node2, 10**15)
            # The recovered run's later decisions read the book as the late mail
            # left it, so its digest is not the uninterrupted run's; what must
            # hold is that nothing was sent twice, that no decision bar sent more
            # than the one entry the uninterrupted run sends, and that the book and the
            # venue agree on every position and on the cash.
            assert len(venue2.sent) == len(set(venue2.sent)), (seed, k)
            got = entries(venue2)
            assert max(got.values(), default=1) == 1, (seed, k, "two entries on one decision bar")
            assert node2.reconcile() == [], (seed, k)


# -- checkpoint ------------------------------------------------------------------------


def midway():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j)
    steps = script()
    for step in steps[:60]:
        run_step(node, venue, step)
    return node, venue, j, steps


def test_a_checkpoint_rebuilds_the_node_and_the_run_goes_on_identically():
    node, venue, j, steps = midway()
    blob = node.checkpoint()
    assert blob[:8] == b"RBTCKPT1"
    body = rj.decode_checkpoint(blob)
    assert body["v"] == 1 and body["count"] == j.count and body["chain"] == j.chain
    assert body["states"] == [json.dumps(node.strategies[0].state(), sort_keys=True)]
    for step in steps[60:]:
        run_step(node, venue, step)
    want = snapshot(node, venue)

    # The process died right after the checkpoint; the venue went on.
    venue2 = MockVenue()
    venue2.orders = json.loads(json.dumps(venue.orders))  # a copy, taken before the run went on
    venue3 = MockVenue()
    n2, v2, _, _ = midway()
    blob2 = n2.checkpoint()
    restored = LiveNode.restore(blob2, [Trader()], SYMBOLS, CONFIG, execution=v2, bar_stamp="close")
    assert restored.state == "live"
    assert restored.state_digest() == n2.state_digest()
    # The chain is the same one, continued by the recovery's own two records.
    assert [r.kind for r in restored.journal.since(n2.journal.count)] == ["reconcile", "resume"]
    assert restored.journal.since(n2.journal.count - 1)[0].chain == n2.journal.chain
    assert restored.strategies[0].count == n2.strategies[0].count
    for step in steps[60:]:
        run_step(restored, v2, step)
    assert snapshot(restored, v2) == want
    assert venue3 is not None and venue2 is not None


def test_a_damaged_or_wrong_checkpoint_is_refused():
    node, venue, j, steps = midway()
    blob = bytearray(node.checkpoint())
    with pytest.raises(InvalidData) as e:
        rj.decode_checkpoint(bytes(blob[:-5]))
    assert e.value.code == "invalid_checkpoint"
    flipped = bytearray(blob)
    flipped[40] ^= 0x01
    with pytest.raises(InvalidData):
        LiveNode.restore(bytes(flipped), [Trader()], SYMBOLS, CONFIG, execution=MockVenue(), bar_stamp="close")
    with pytest.raises(InvalidData):
        rj.decode_checkpoint(b"nonsense")
    # A checkpoint whose digest is not the replay's: someone edited the state.
    body = rj.decode_checkpoint(bytes(blob))
    body["digest"] = "0" * 64
    with pytest.raises(StateError) as e2:
        LiveNode.restore(rj.encode_checkpoint(body), [Trader()], SYMBOLS, CONFIG, execution=venue, bar_stamp="close")
    assert e2.value.code == "replay_diverged"


def test_a_checkpoint_without_its_log_needs_the_records():
    node, venue, j, steps = midway()
    light = node.checkpoint(include_log=False)
    assert len(light) < len(node.checkpoint()) // 4
    with pytest.raises(InvalidParameter):
        LiveNode.restore(light, [Trader()], SYMBOLS, CONFIG, execution=venue, bar_stamp="close")
    records = j.since(j.seq_base)
    restored = LiveNode.restore(
        light, [Trader()], SYMBOLS, CONFIG, records=records, execution=venue, bar_stamp="close"
    )
    assert restored.state_digest() == node.state_digest()


def test_strategy_state_crosses_runs_through_the_checkpoint_and_a_continued_journal():
    node, venue, j, steps = midway()
    body = rj.decode_checkpoint(node.checkpoint())
    states = [json.loads(s) for s in body["states"]]
    count, chain, _ = j.head()
    j2 = rj.Journal({"run_id": "r-2"}, seq_base=count, chain=chain, memory=True)
    n2 = build(MockVenue(), j2, strategy_states=states)
    assert n2.strategies[0].count == node.strategies[0].count
    assert j2.since(count)[0].seq == count
    start = j2.since(count)[0]
    assert start.kind == "start" and start.payload["strategy_states"] == body["states"]


def test_a_replay_that_goes_its_own_way_is_named():
    node, venue, j, steps = midway()
    records = j.since(j.seq_base)

    class Other(Trader):
        def on_bar(self, ctx, bar):
            super().on_bar(ctx, bar)
            self.count[ctx.symbol] += 0
            if self.count[ctx.symbol] == 3:
                self.submit_order(orders.Market(side="buy", units=1.0))

    with pytest.raises(StateError) as e:
        LiveNode.restore(
            None, [Other()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
            records=records, execution=MockVenue(), bar_stamp="close",
        )
    assert e.value.code == "replay_diverged"
    # A different feed (warmup) or symbols is a different node too.
    with pytest.raises(StateError):
        LiveNode.restore(
            None, [Trader()], ["A"], CONFIG, journal=rj.Journal(HEADER, memory=True),
            records=records, execution=MockVenue(), bar_stamp="close",
        )


def test_a_file_journal_is_the_whole_recovery(tmp_path):
    path = str(tmp_path / "run.journal")
    venue = MockVenue()
    node = build(venue, rj.Journal(HEADER, file=path, fsync="every"))
    steps = script()
    for step in steps[:70]:
        run_step(node, venue, step)
    want = node.state_digest()
    del node  # the process is gone
    # A partly written last record, as a kill mid-write leaves it.
    with open(path, "ab") as f:
        f.write(b"\x99\x00\x00\x00\x12\x34")
    j, records = rj.open_for_recovery(path, fsync="every", memory=True)
    restored = LiveNode.restore(
        None, [Trader()], SYMBOLS, CONFIG, journal=j, records=records, execution=venue,
        bar_stamp="close",
    )
    settle(restored, venue)
    assert restored.state_digest() == want
    for step in steps[70:]:
        run_step(restored, venue, step)
    restored.journal.sync()
    # What was appended after recovery went to the same file, on the same chain.
    header, base, chain_from, again, torn = rj.read(path)
    assert torn == 0 and rj.verify(again, header) == restored.journal.head()[1]


# -- the recovery gate ---------------------------------------------------------------------


def lost_order_story():
    """A node dies after journaling an order's request but before the venue got it."""
    venue = MockVenue()
    store = []

    def sink(r):
        if len(store) >= 6:  # start, 3 bars, command, adapter_request
            raise Crash("killed")
        store.append(r.canonical())

    node = build(venue, rj.Journal(HEADER, sink=sink))
    venue.fail_next = Crash("the process died as it called the venue")
    with pytest.raises(Crash):
        for step in script():
            run_step(node, venue, step)
    assert venue.sent == [] and store.__len__() == 6
    return venue, store


def test_recovery_never_resends_an_order_whose_request_has_no_result_it_asks_the_venue():
    venue, store = lost_order_story()
    seen = []
    node = LiveNode.restore(
        None, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True), records=list(store),
        execution=venue, bar_stamp="close", on_discrepancy=seen.append,
    )
    assert venue.sent == [], "recovery must not send"
    assert node.state == "recovering"
    assert [(d.kind, d.client_order_id) for d in seen] == [("missing_order", "s0-0")]
    assert node.health()["unconfirmed"] == ["s0-0"]
    # Market data is refused until the host decides; reports and resend are not.
    with pytest.raises(StateError) as e:
        node.push_bar("A", 99 * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert e.value.code == "recovering"
    with pytest.raises(StateError):
        node.resume()
    # Accepting the discrepancy resumes without sending anything.
    node.resume(accept="all")
    assert node.state == "live" and venue.sent == []
    kinds_ = [r.kind for r in node.journal.since(store.__len__())]
    assert "reconcile" in kinds_ and "discrepancy" in kinds_ and "resume" in kinds_


def test_a_discrepancy_is_delivered_to_the_strategy_and_resend_settles_it():
    venue, store = lost_order_story()
    got = []

    class Listener(Trader):
        def on_discrepancy(self, ctx, d):
            got.append(d)

    node = LiveNode.restore(
        None, [Listener()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True), records=list(store),
        execution=venue, bar_stamp="close",
    )
    assert [d.kind for d in got] == ["missing_order"]
    node.resend("s0-0")
    assert venue.sent == ["s0-0"]
    assert node.reconcile() == []
    node.resume()
    assert node.state == "live"
    # The id is the first request's: the venue de-duplicates a second try.
    node.resend("s0-0")
    assert venue.sent == ["s0-0", "s0-0"] and len(venue.orders) == 1


def test_unknown_orders_and_position_and_cash_mismatches_are_found():
    venue = MockVenue()
    node = build(venue, rj.Journal(HEADER, memory=True))
    for step in script()[:9]:
        run_step(node, venue, step)
    venue.orders["ghost"] = {"client_order_id": "ghost", "symbol": "A", "side": "buy", "quantity": 1.0, "filled_qty": 0.0, "open": True, "ts": 0}
    venue.position["A"] = 3.0
    venue.cash -= 12.5
    found = {d.kind: d for d in node.reconcile()}
    assert found["unknown_order"].client_order_id == "ghost"
    assert found["position_mismatch"].expected != found["position_mismatch"].actual
    assert found["cash_mismatch"].actual == pytest.approx(found["cash_mismatch"].expected - 12.5 - (venue.cash + 12.5 - found["cash_mismatch"].expected) * 0)


def test_an_adapter_that_cannot_reconcile_is_refused_unless_the_host_says_skip():
    class Plain:
        namespace = "P1"
        keeps_own_records = True  # a test venue: it remembers the node's orders itself

        def submit(self, r):
            pass

        def cancel(self, c):
            pass

        def modify(self, c, **k):
            pass

    plain = Plain()
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([Trader()], SYMBOLS, CONFIG, bar_stamp="close", execution=plain, journal=j)
    for step in script()[:6]:
        if step[0] == "bar":
            node.push_bar(step[1], step[2], *step[3:8])
    records = j.since(0)
    with pytest.raises(Exception) as e:
        LiveNode.restore(None, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
                         records=records, execution=plain, bar_stamp="close")
    assert e.value.code == "no_reconcile"
    ok = LiveNode.restore(None, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
                          records=records, execution=plain, bar_stamp="close", reconcile="skip")
    assert ok.state == "live"
    last = ok.journal.since(len(records))[-1]
    assert last.kind == "resume" and last.payload == {"skipped": True}


# -- fencing -----------------------------------------------------------------------------


def test_a_node_with_a_stale_epoch_sends_nothing_and_says_why():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j, epoch=1)
    node.heartbeat()
    assert venue.epoch == 1
    successor = build(venue, rj.Journal(HEADER, memory=True), epoch=2)  # takes the account
    successor.heartbeat()
    assert venue.epoch == 2 and not successor.fenced
    with pytest.raises(Exception) as e:
        node.heartbeat()
    assert e.value.code == "fenced"
    assert node.fenced and node.health()["fenced"]
    assert [r.kind for r in j.since(0)][-1] == "fence"
    # Everything it does after that is refused, and no order leaves it.
    with pytest.raises(StateError) as e2:
        node.push_bar("A", 5 * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert e2.value.code == "fenced"
    assert venue.sent == []


def test_the_epoch_is_checked_before_each_request_not_only_on_a_heartbeat():
    venue = MockVenue()
    node = build(venue, rj.Journal(HEADER, memory=True), epoch=1)
    for step in script()[:2]:
        run_step(node, venue, step)
    venue.epoch = 5  # someone else owns the account now
    with pytest.raises(Exception) as e:
        for step in script()[2:12]:
            run_step(node, venue, step)
    assert e.value.code == "fenced"
    assert venue.sent == []
    res = [r for r in node.journal.since(0) if r.kind == "adapter_request"]
    assert res == [], "a fenced node writes no request: nothing was about to be sent"


# -- corrections and busts ---------------------------------------------------------------


class Sink:
    """A venue that takes everything and says nothing; the test speaks for it."""

    namespace = "T1"
    keeps_own_records = True  # a test venue: it remembers the node's orders itself

    def __init__(self):
        self.sent = []

    def submit(self, r):
        self.sent.append(r.client_order_id)

    def cancel(self, c):
        pass

    def modify(self, c, **k):
        pass


class Once(Strategy):
    def __init__(self, side="buy", units=10.0, then=None):
        super().__init__()
        self.side, self.units, self.then, self.n = side, units, then, 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side=self.side, units=self.units))
        elif self.n == 3 and self.then:
            self.submit_order(orders.Market(side="sell", units=self.then, reduce_only=True))


def restate_node(fees=0.0003, account="cash", journal=None):
    cfg = BacktestConfig(initial_capital=100_000.0, fees=fees)
    kw = {} if account == "cash" else {"account_type": "margin", "leverage": 5.0}
    node = LiveNode([Once()], ["A"], cfg, bar_stamp="close", execution=Sink(), journal=journal, **kw)
    node.push_bar("A", MIN, 100.0, 101.0, 99.0, 100.5, 10.0)
    return node


def fills(node, *execs):
    for k, (eid, q, p) in enumerate(execs):
        r = ExecutionReport.fill(eid, q, p, 2 * MIN + k)
        node.push_report("s0-0", r, arrival_ns=2 * MIN + k)


def books(node):
    ctx = node.ctx(0)
    return {
        "cash": node._views[0].cash(),
        "equity": node._views[0].equity(),
        "pos": [(p.direction, p.size, round(p.entry_price, 9)) for p in ctx.positions("A")],
        "locked": node._rt.venue_orders(),
    }


def test_a_bust_takes_an_entry_back_exactly_in_cash_and_position():
    node = restate_node()
    start = books(node)["cash"]
    fills(node, ("E1", 10.0, 100.0))
    assert books(node)["cash"] < start and books(node)["pos"]
    node.push_report("s0-0", ExecutionReport.bust("E1", 3 * MIN), arrival_ns=3 * MIN)
    got = books(node)
    assert got["cash"] == start and got["pos"] == [] and got["equity"] == start
    assert node.finish()[0].result.trades() == []


@pytest.mark.parametrize("account", ["cash", "margin"])
def test_a_correction_leaves_the_book_as_if_the_fill_had_been_reported_as_corrected(account):
    corrected = restate_node(account=account)
    fills(corrected, ("E1", 6.0, 100.0), ("E2", 4.0, 101.0))
    corrected.push_report("s0-0", ExecutionReport.correction("E1", 5.0, 99.0, 4 * MIN), arrival_ns=4 * MIN)

    reference = restate_node(account=account)
    fills(reference, ("E1", 5.0, 99.0), ("E2", 4.0, 101.0))

    a, b = books(corrected), books(reference)
    assert a["cash"] == b["cash"], "cash must agree to the minor unit"
    assert a["equity"] == b["equity"]
    assert a["pos"] == b["pos"]
    # And the same for restating the later of the two.
    again = restate_node(account=account)
    fills(again, ("E1", 6.0, 100.0), ("E2", 4.0, 101.0))
    again.push_report("s0-0", ExecutionReport.correction("E2", 3.0, 102.0, 4 * MIN), arrival_ns=4 * MIN)
    ref2 = restate_node(account=account)
    fills(ref2, ("E1", 6.0, 100.0), ("E2", 3.0, 102.0))
    assert books(again)["cash"] == books(ref2)["cash"] and books(again)["pos"] == books(ref2)["pos"]


def test_a_bust_of_the_opening_execution_keeps_the_later_one():
    busted = restate_node()
    fills(busted, ("E1", 6.0, 100.0), ("E2", 4.0, 101.0))
    busted.push_report("s0-0", ExecutionReport.bust("E1", 4 * MIN), arrival_ns=4 * MIN)
    ref = restate_node()
    fills(ref, ("E2", 4.0, 101.0))
    assert books(busted)["cash"] == books(ref)["cash"]
    assert books(busted)["pos"] == books(ref)["pos"]


def test_a_correction_that_arrives_before_the_fill_it_restates_books_the_corrected_fill():
    early = restate_node()
    early.push_report("s0-0", ExecutionReport.correction("E1", 5.0, 99.0, 4 * MIN), arrival_ns=4 * MIN)
    fills(early, ("E1", 6.0, 100.0))
    ref = restate_node()
    fills(ref, ("E1", 5.0, 99.0))
    assert books(early)["cash"] == books(ref)["cash"] and books(early)["pos"] == books(ref)["pos"]
    busted_first = restate_node()
    start = books(busted_first)["cash"]
    busted_first.push_report("s0-0", ExecutionReport.bust("E1", 4 * MIN), arrival_ns=4 * MIN)
    fills(busted_first, ("E1", 6.0, 100.0))
    assert books(busted_first)["cash"] == start and books(busted_first)["pos"] == []


def test_the_same_correction_twice_changes_nothing():
    node = restate_node()
    fills(node, ("E1", 10.0, 100.0))
    c = ExecutionReport.correction("E1", 8.0, 100.5, 4 * MIN)
    node.push_report("s0-0", c, arrival_ns=4 * MIN)
    once = books(node)
    assert node.push_report("s0-0", c, arrival_ns=5 * MIN) == "duplicate"
    assert books(node) == once


def test_an_exit_cannot_be_taken_back_and_nothing_changes_when_it_is_refused():
    node = LiveNode([Once(then=10.0)], ["A"], BacktestConfig(initial_capital=100_000.0, fees=0.0003),
                    bar_stamp="close", execution=Sink(), journal=rj.Journal(HEADER, memory=True))
    node.push_bar("A", MIN, 100.0, 101.0, 99.0, 100.5, 10.0)
    node.push_report("s0-0", ExecutionReport.fill("E1", 10.0, 100.0, 2 * MIN), arrival_ns=2 * MIN)
    node.push_bar("A", 2 * MIN, 100.0, 101.0, 99.0, 100.5, 10.0)
    node.push_bar("A", 3 * MIN, 100.0, 101.0, 99.0, 100.5, 10.0)
    node.push_report("s0-1", ExecutionReport.fill("X1", 10.0, 102.0, 4 * MIN), arrival_ns=4 * MIN)
    before = node.state_digest()
    with pytest.raises(StateError) as e:
        node.push_report("s0-1", ExecutionReport.bust("X1", 5 * MIN), arrival_ns=5 * MIN)
    assert e.value.code == "correction_not_reversible"
    assert node.state_digest() == before
    # An entry that has since been reduced cannot be taken back either.
    with pytest.raises(StateError):
        node.push_report("s0-0", ExecutionReport.bust("E1", 5 * MIN), arrival_ns=5 * MIN)
    assert node.state_digest() == before
    kinds_ = [r.kind for r in node.journal.since(0)]
    assert kinds_.count("error") == 2


def test_a_restatement_is_journaled_and_replays_to_the_same_book():
    j = rj.Journal(HEADER, memory=True)
    node = restate_node(journal=j)
    fills(node, ("E1", 6.0, 100.0), ("E2", 4.0, 101.0))
    node.push_report("s0-0", ExecutionReport.correction("E1", 5.0, 99.0, 4 * MIN), arrival_ns=4 * MIN)
    node.push_report("s0-0", ExecutionReport.bust("E2", 5 * MIN), arrival_ns=5 * MIN)
    kinds_ = [r.kind for r in j.since(0)]
    assert kinds_.count("restatement") == 2
    r = [r for r in j.since(0) if r.kind == "restatement"][0]
    assert r.client_order_id == "s0-0" and r.payload["report"] == "correction" and r.payload["outcome"] == "applied"
    want = books(node)
    again = LiveNode.restore(
        None, [Once()], ["A"], BacktestConfig(initial_capital=100_000.0, fees=0.0003),
        journal=rj.Journal(HEADER, memory=True), records=j.since(0), execution=Sink(),
        bar_stamp="close", reconcile="skip",
    )
    assert books(again) == want


def test_ids_continue_after_a_restore_and_never_restart_at_zero():
    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = build(venue, j)
    for step in script()[:40]:
        run_step(node, venue, step)
    before = list(venue.sent)
    assert len(before) >= 3
    restored = LiveNode.restore(
        None, [Trader()], SYMBOLS, CONFIG, journal=rj.Journal(HEADER, memory=True),
        records=j.since(0), execution=venue, bar_stamp="close",
    )
    settle(restored, venue)
    for step in script()[40:80]:
        run_step(restored, venue, step)
    new = venue.sent[len(before):]
    assert new and new[0] == f"s0-{len(before)}", (before[-1], new[:2])
    assert len(venue.sent) == len(set(venue.sent))


def test_a_fill_the_book_could_not_book_yet_is_a_typed_record_and_booked_later():
    """A reduce-only fill reported before the fill that opens its position is
    kept, journaled as `unbooked_fill`, and booked when the entry arrives."""

    class Both(Strategy):
        def on_bar(self, ctx, bar):
            if not getattr(self, "sent", False):
                self.sent = True
                self.submit_order(orders.Market(side="buy", units=5.0), client_id="in")
                self.submit_order(orders.Market(side="sell", units=5.0, reduce_only=True), client_id="out")

    venue = MockVenue()
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([Both()], ["A"], CONFIG, bar_stamp="close", execution=venue, journal=j)
    node.push_bar("A", MIN, 100.0, 101.0, 99.0, 100.5, 1000.0)
    entry, exit_ = sorted(venue.sent)
    node.push_report(exit_, ExecutionReport.fill("X1", 5.0, 101.0, MIN), arrival_ns=MIN)
    (rec,) = [r for r in j.since(0) if r.kind == "unbooked_fill"]
    assert rec.client_order_id == exit_ and rec.payload["exec_id"] == "X1"
    node.push_report(entry, ExecutionReport.fill("E1", 5.0, 100.0, MIN), arrival_ns=MIN)
    assert node.ctx(0).positions("A") == []
