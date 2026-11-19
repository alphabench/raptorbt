"""A stop or target held overnight on a segment whose orders are day orders.

On NFO, BFO, CDS, BCD, BSE and MCX the engine sends its protective legs as day orders: the
segment holds no good-till-cancelled one. A day leg ends at the session's close, while a backtest
keeps the position's level across sessions. The engine sends a lapsed leg again at the first event
of the next session, for what the position then holds, and journals the lapse and the re-arm.
NSE cash is the control: its legs are good-till-cancelled and never lapse.

Calendar (IST): Mon 2026-11-02, Tue 11-03, Wed 11-04, Thu 11-05. The NFO session closes at 15:40
(`SessionSchedule.nse_fo`, from 2026-08-03), BSE and NSE cash at 15:30, and every one of them opens
at 09:15. Bars are close-stamped: the bar stamped 09:16 is the session's first.
"""

import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, LiveNode, journal as rj
from raptorbt.execution import ExecutionReport, OrderRefused, PaperExecution
from raptorbt.strategy import orders as O

from _market_harness import Script, bars, ist, run_paths

MON, TUE, WED, THU = (2026, 11, 2), (2026, 11, 3), (2026, 11, 4), (2026, 11, 5)
EXPIRY = ist(2026, 11, 26, 15, 30)
FLAT = (100, 100, 100, 100)


def at(day, hh, mm):
    return ist(*day, hh, mm)


def spec(venue, kind):
    if kind == "future":
        return rb.InstrumentSpec.futures_contract("F", EXPIRY, 1.0, venue=venue)
    return rb.InstrumentSpec.equity("F", venue=venue)


def cfg():
    return BacktestConfig(initial_capital=1e6, fees=0.0)


def exits(out):
    """``{path: [(entry bar, exit bar, exit price)]}``. The exit reason is left out: a venue's
    close is an order (`Order`) where the batch names the rule (`StopLoss`)."""
    return {
        p: [(t.entry_idx, t.exit_idx, round(t.exit_price, 6)) for t in r.result.trades()]
        for p, (_, r) in out.items()
    }


def run(rows, stamps, script, venue="NFO", kind="future", config=cfg):
    return run_paths(
        script, {"F": bars(rows, stamps)}, config,
        instruments={"F": spec(venue, kind)}, account_type="margin",
    )


def bracket(s, c):
    s.enter(size_frac=0.1, stop_price=90.0, target_price=110.0)


# -- paper and live agree with the backtest -----------------------------------------------------

# Bars 0-2 Monday, 3-5 Tuesday. Bar 3 is quiet: a day leg sent Monday has lapsed by then.
STAMPS = [at(MON, 10, 1), at(MON, 10, 2), at(MON, 10, 3), at(TUE, 9, 16), at(TUE, 9, 17), at(TUE, 9, 18)]


@pytest.mark.parametrize("venue, kind", [("NFO", "future"), ("BSE", "cash"), ("NSE", "cash")])
def test_an_overnight_stop_trips_on_paper_where_the_backtest_trips_it(venue, kind):
    # entered on bar 1 at 100, stop 90, target 110. Bars 0-3 trade at 100 only. Bar 4 opens at 95
    # (above the stop) and trades down to 85: the stop is reached inside the bar and fills at its
    # level, 90. NSE cash is the control (its legs are gtc and never lapse).
    rows = [FLAT] * 4 + [(95, 96, 85, 88), (88, 88, 88, 88)]
    got = exits(run(rows, STAMPS, {1: bracket}, venue, kind))
    assert [x[1:] for x in got["batch"]] == [(4, 90.0)], got
    assert got["node"] == got["batch"], got
    assert got["paper"] == got["batch"], got


def test_an_overnight_target_fills_on_paper_where_the_backtest_fills_it():
    # bar 4 opens at 105 (below the target) and trades up to 112: the target fills at its level,
    # 110; its low, 104, is above the stop at 90
    rows = [FLAT] * 4 + [(105, 112, 104, 111), (111, 111, 111, 111)]
    got = exits(run(rows, STAMPS, {1: bracket}))
    assert [x[1:] for x in got["batch"]] == [(4, 110.0)], got
    assert got["paper"] == got["batch"], got


def test_a_stop_gapped_through_at_the_next_open_fills_at_that_open():
    # bar 3, the first of Tuesday, opens at 85, below the stop at 90: the stop fills at the open,
    # 85, on bar 3. The leg sent again on that bar meets it, as the backtest's level does.
    rows = [FLAT] * 3 + [(85, 86, 84, 85), (85, 85, 85, 85)]
    got = exits(run(rows, STAMPS[:5], {1: bracket}))
    assert [x[1:] for x in got["batch"]] == [(3, 85.0)], got
    assert got["paper"] == got["batch"], got


def test_a_trailing_stop_held_overnight_trails_from_where_it_had_reached():
    # 10% trail. Entered on bar 1 at 100; bar 3 reaches 120, so the stop stands at
    # 120 * (1 - 0.10) = 108. Bar 4 (Tuesday's first) trades at 115, above it. Bar 5 opens at 115
    # and trades down to 100: the stop fills at 108. A trail started again from Tuesday's prices
    # (115 * 0.9 = 103.5) would fill lower.
    def trailing():
        c = cfg()
        c.set_trailing_stop(0.10)
        return c

    stamps = [at(MON, 10, 1), at(MON, 10, 2), at(MON, 10, 3), at(MON, 10, 4),
              at(TUE, 9, 16), at(TUE, 9, 17), at(TUE, 9, 18)]  # fmt: skip
    rows = [FLAT] * 3 + [(100, 120, 100, 120), (115, 115, 115, 115), (115, 116, 100, 105),
                         (105, 105, 105, 105)]  # fmt: skip
    got = exits(run(rows, stamps, {1: lambda s, c: s.enter(size_frac=0.1)}, config=trailing))
    assert [x[1:] for x in got["batch"]] == [(5, 108.0)], got
    assert got["paper"] == got["batch"], got


def test_a_stop_held_over_three_sessions_is_sent_again_each_morning():
    # entered on bar 1 (Monday); Tuesday and Wednesday are quiet; Thursday's second bar (bar 6)
    # opens at 95 and trades down to 85: the stop fills at 90 on bar 6. The leg lapses at the
    # close of Monday, Tuesday and Wednesday, and is sent again at the first bar of Tuesday,
    # Wednesday and Thursday: three times.
    stamps = [at(MON, 10, 1), at(MON, 10, 2), at(MON, 10, 3), at(TUE, 9, 16), at(WED, 9, 16),
              at(THU, 9, 16), at(THU, 9, 17), at(THU, 9, 18)]  # fmt: skip
    rows = [FLAT] * 6 + [(95, 96, 85, 88), (88, 88, 88, 88)]
    got = exits(run(rows, stamps, {1: bracket}))
    assert [x[1:] for x in got["batch"]] == [(6, 90.0)], got
    assert got["paper"] == got["batch"], got

    records = paper_journal(rows, stamps)
    stops = [r for r in engine_exits(records, "rearmed") if r.payload["leg"] == "stop"]
    assert [r.payload["event_ts"] for r in stops] == [at(TUE, 9, 16), at(WED, 9, 16), at(THU, 9, 16)]


# -- the journal --------------------------------------------------------------------------------


def engine_exits(records, reason):
    return [r for r in records if r.kind == "engine_exit" and r.payload.get("reason") == reason]


def paper_journal(rows, stamps, script=None):
    journal = rj.Journal({"run": "overnight"}, memory=True)
    node = LiveNode(
        [Script(script or {1: bracket})], ["F"], cfg(), bar_stamp="close", execution=PaperExecution(),
        instruments={"F": spec("NFO", "future")}, account_type="margin", journal=journal,
    )
    a = bars(rows, stamps)
    for i, ts in enumerate(a["timestamps"]):
        node.push_bar("F", int(ts), *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume")))
    node.finish()
    return list(journal.since(0))


def test_each_lapse_and_each_re_arm_is_journaled():
    rows = [FLAT] * 4 + [(95, 96, 85, 88), (88, 88, 88, 88)]
    records = paper_journal(rows, STAMPS)
    lapsed = engine_exits(records, "lapsed_day_order")
    rearmed = engine_exits(records, "rearmed")
    assert sorted(r.payload["leg"] for r in lapsed) == ["stop", "target"], records
    assert sorted(r.payload["leg"] for r in rearmed) == ["stop", "target"], records
    for r in lapsed:
        p = r.payload
        assert r.source == "engine" and p["venue"] is True and p["status"] == "expired"
        assert float(p["level"]) == {"stop": 90.0, "target": 110.0}[p["leg"]]
        # sent Monday, a day order lapses at Monday's NFO close, 15:40; the paper venue says so on
        # Tuesday's first bar, which is when the leg is sent again
        assert p["rearm_after_ns"] == at(MON, 15, 40)
        assert p["event_ts"] == at(TUE, 9, 16)
        assert p["client_id"].startswith("prot-") and r.client_order_id == p["order"]
    by_leg = {r.payload["leg"]: r.payload for r in lapsed}
    for r in rearmed:
        p, old = r.payload, by_leg[r.payload["leg"]]
        assert p["event_ts"] == at(TUE, 9, 16) and float(p["level"]) == float(old["level"])
        assert (p["replaces"], p["replaces_client_id"]) == (old["order"], old["client_id"])
        assert p["order"] != old["order"] and p["client_id"].startswith("prot-")
        # the position did not change between the lapse and the re-arm
        assert float(p["qty"]) == float(old["qty"]) > 0
        # the new leg's request is on the record before the re-arm that names it
        requests = [i for i, x in enumerate(records)
                    if x.kind == "adapter_request" and x.client_order_id == p["order"]]  # fmt: skip
        assert requests and requests[0] < records.index(r)


# -- a broker-like venue: reports pushed by hand --------------------------------------------------


class Venue:
    """A venue that records what it is sent; the test pushes its reports."""

    namespace = "OVN"
    keeps_own_records = True

    def __init__(self):
        self.requests, self.cancels = [], []

    def submit(self, r):
        self.requests.append(r)

    def cancel(self, cid):
        self.cancels.append(cid)

    def modify(self, cid, *, quantity=None, limit_price=None, trigger_price=None):
        pass


def both(qty):
    """The stop at 90 and the target at 110 for `qty`, as day orders that only reduce."""
    return [("stop_market", 90.0, qty, "day", True), ("limit", 110.0, qty, "day", True)]


def legs(requests):
    return [
        (r.order_type, r.trigger_price if r.order_type == "stop_market" else r.limit_price, r.quantity,
         r.time_in_force, r.reduce_only)
        for r in requests
        if r.reduce_only and r.order_type in ("stop_market", "limit")
    ]  # fmt: skip


def build(script, venue=None, journal=None):
    venue = venue or Venue()
    node = LiveNode(
        [Script(script)], ["F"], cfg(), bar_stamp="close", execution=venue,
        instruments={"F": spec("NFO", "future")}, account_type="margin", journal=journal,
    )
    return venue, node


def bar(node, ts, row=FLAT):
    node.push_bar("F", ts, *map(float, row), 1e6)


def report(node, cid, rep, ts):
    node.push_report(cid, rep, arrival_ns=ts)


ENTRY = {0: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=90.0, target_price=110.0))}
# Monday 15:45: after the 15:40 close, before Tuesday's 09:15 open
EVENING = at(MON, 15, 45)


def long_ten(script=None, venue=None, journal=None):
    """Long 10 at 100 on Monday with a stop at 90 and a target at 110, both accepted; Monday's
    bars 10:01-10:03 pushed. Returns the venue, the node and the two legs' requests."""
    venue, node = build({**ENTRY, **(script or {})}, venue, journal)
    ts = at(MON, 10, 1)
    bar(node, ts)
    entry = venue.requests[0]
    report(node, entry.client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), ts)
    stop, target = venue.requests[1:3]
    assert legs([stop, target]) == both(10.0)
    for leg in (stop, target):
        report(node, leg.client_order_id, ExecutionReport.accepted("V" + leg.client_order_id, ts), ts)
    bar(node, at(MON, 10, 2))
    bar(node, at(MON, 10, 3))
    return venue, node, stop, target


def lapse(node, stop, target, how=ExecutionReport.expired):
    for leg in (stop, target):
        report(node, leg.client_order_id, how(EVENING), EVENING)


def test_a_lapsed_leg_is_sent_again_at_the_next_sessions_first_bar_and_not_before():
    venue, node, stop, target = long_ten()
    lapse(node, stop, target)
    sent = len(venue.requests)
    assert sent == 3, "the entry and its two legs: nothing is sent when the legs lapse"
    bar(node, at(TUE, 9, 16))
    assert legs(venue.requests[sent:]) == both(10.0)
    bar(node, at(TUE, 9, 17))
    assert len(venue.requests) == sent + 2, "sent once, not on every bar"


def test_a_lapse_a_venue_reports_as_a_cancel_after_the_close_is_a_lapse():
    # a venue with no expiry status (Kite) reports a lapsed day order as cancelled
    venue, node, stop, target = long_ten()
    lapse(node, stop, target, how=ExecutionReport.cancel_ack)
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 16))
    assert legs(venue.requests[sent:]) == both(10.0)


def test_a_leg_the_strategy_cancelled_is_not_sent_again():
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten({2: lambda s, c: s.cancel_all_orders()}, journal=journal)
    assert set(venue.cancels) == {stop.client_order_id, target.client_order_id}
    # the venue answers the cancel after the close, when a lapse would also have ended them
    lapse(node, stop, target, how=ExecutionReport.cancel_ack)
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 16))
    bar(node, at(TUE, 9, 17))
    assert venue.requests[sent:] == []
    assert engine_exits(journal.since(0), "lapsed_day_order") == []


def test_a_position_closed_before_the_next_session_gets_no_leg():
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten({2: lambda s, c: s.close_position()}, journal=journal)
    (close,) = [r for r in venue.requests if r.reduce_only and r.order_type == "market"]
    lapse(node, stop, target)
    report(node, close.client_order_id, ExecutionReport.fill("C1", 10.0, 100.0, EVENING + 1), EVENING + 1)
    assert node.positions(0, "F") == []
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 16))
    assert venue.requests[sent:] == []
    skipped = engine_exits(journal.since(0), "rearm_skipped")
    assert sorted((r.payload["leg"], r.payload["why"]) for r in skipped) == [
        ("stop", "position_closed"), ("target", "position_closed")
    ]


def test_a_leg_waits_while_a_close_is_working_and_is_dropped_once_it_fills():
    venue, node, stop, target = long_ten({3: lambda s, c: s.close_position()})
    lapse(node, stop, target)
    sent = len(venue.requests)
    ts = at(TUE, 9, 16)
    bar(node, ts)  # the strategy closes on this bar: the close is working, no leg is sent
    (close,) = [r for r in venue.requests[sent:] if r.reduce_only and r.order_type == "market"]
    assert legs(venue.requests[sent:]) == []
    report(node, close.client_order_id, ExecutionReport.fill("C1", 10.0, 100.0, ts), ts)
    bar(node, at(TUE, 9, 17))
    assert legs(venue.requests[sent:]) == []


@pytest.mark.parametrize("side, units, held", [("buy", 5.0, 15.0), ("sell", 4.0, 6.0)])
def test_a_leg_sent_again_covers_the_position_as_it_stands_after_an_add_or_a_part_close(side, units, held):
    # long 10; an add of 5 (10 + 5 = 15) or a sale of 4 (10 - 4 = 6) fills after the legs lapsed
    venue, node, stop, target = long_ten({2: lambda s, c: s.submit_order(O.Market(side=side, units=units))})
    (change,) = [r for r in venue.requests[3:] if r.order_type == "market"]
    lapse(node, stop, target)
    report(node, change.client_order_id, ExecutionReport.fill("A1", units, 100.0, EVENING + 1), EVENING + 1)
    assert [p.size for p in node.positions(0, "F")] == [held]
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 16))
    assert legs(venue.requests[sent:]) == both(held)


def test_a_node_restored_overnight_sends_the_lapsed_legs_at_the_next_session():
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten(journal=journal)
    lapse(node, stop, target)
    records = [r.canonical() for r in journal.since(0)]
    # the node is lost overnight and rebuilt from its journal
    again = Venue()
    restored = LiveNode.restore(
        None, [Script(ENTRY)], ["F"], cfg(), journal=rj.Journal({"run": "overnight"}, memory=True),
        records=records, execution=again, reconcile="skip", bar_stamp="close",
        instruments={"F": spec("NFO", "future")}, account_type="margin",
    )
    assert restored.state_digest() == node.state_digest()
    assert again.requests == [], "a replay sends nothing"
    bar(restored, at(TUE, 9, 16))
    assert legs(again.requests) == both(10.0)


# -- what happens to a leg sent again: the first bar, refusals, an unreachable venue -------------


def test_the_strategy_sees_the_overnight_stop_fill_before_deciding_on_the_first_bar():
    # Tuesday's first bar (bar 3) opens at 95 and trades down to 85: the stop at 90 is reached
    # inside it. The batch closes the position while the bar forms, before `on_bar`, so the
    # strategy sees itself flat on bar 3; on paper the leg sent again meets the bar first too.
    seen = {}

    def look(s, c):
        seen.setdefault(id(s), len(c.positions(c.symbol)))

    rows = [FLAT] * 3 + [(95, 96, 85, 88), (88, 88, 88, 88)]
    out = run(rows, STAMPS[:5], {1: bracket, 3: look})
    flat_at_3 = {p: seen[id(s)] for p, (s, _) in out.items()}
    assert flat_at_3 == {"batch": 0, "node": 0, "paper": 0}, flat_at_3
    got = exits(out)
    assert [x[1:] for x in got["batch"]] == [(3, 90.0)], got
    assert got["paper"] == got["batch"], got


def test_a_target_gapped_through_at_the_next_open_fills_at_that_open():
    # entered at 100, target 110; bar 3 opens at 115 (high 116, low 114): a sell limit at 110
    # marketable at the open fills at the open, 115
    rows = [FLAT] * 3 + [(115, 116, 114, 115), (115, 115, 115, 115)]
    got = exits(run(rows, STAMPS[:5], {1: bracket}))
    assert [x[1:] for x in got["batch"]] == [(3, 115.0)], got
    assert got["paper"] == got["batch"], got


def rearm_records(journal, reason=None):
    return [
        r for r in journal.since(0)
        if r.kind == "engine_exit" and str(r.payload.get("reason", "")).startswith("rearm")
        and (reason is None or r.payload["reason"] == reason)
    ]  # fmt: skip


def market_closes(requests):
    return [(r.side, r.quantity) for r in requests if r.reduce_only and r.order_type == "market"]


def test_a_first_bar_through_the_stop_on_a_broker_venue_closes_at_market_on_that_bar():
    # long 10, stop 90: Tuesday's first bar trades 85-86-84-85, through the stop (84 <= 90). A
    # stop sent after it would rest above the market (a broker refuses it); the engine closes the
    # 10 at market on this bar instead and sends no leg.
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten(journal=journal)
    lapse(node, stop, target)
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 16), (85, 86, 84, 85))
    assert legs(venue.requests[sent:]) == []
    assert market_closes(venue.requests[sent:]) == [("sell", 10.0)]
    (crossed,) = rearm_records(journal, "rearm_level_crossed")
    assert crossed.payload["leg"] == "stop" and float(crossed.payload["bar"]["low"]) == 84.0
    (skipped,) = rearm_records(journal, "rearm_skipped")
    assert (skipped.payload["leg"], skipped.payload["why"]) == ("target", "level_crossed_closing")


class Refusing(Venue):
    """Refuses every request once `refuse` is set (a synchronous OrderRefused)."""

    refuse = False

    def submit(self, r):
        if self.refuse:
            raise OrderRefused("trigger price is above the last traded price")
        super().submit(r)


def test_a_leg_refused_when_sent_again_is_journaled_and_watched():
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten(venue=Refusing(), journal=journal)
    lapse(node, stop, target)
    venue.refuse = True
    bar(node, at(TUE, 9, 16))
    recs = rearm_records(journal)
    assert sorted((r.payload["leg"], r.payload["reason"]) for r in recs) == [
        ("stop", "rearm_rejected"), ("target", "rearm_rejected")
    ], [r.payload for r in recs]
    for r in recs:
        p = r.payload
        assert p["order"] and p["client_id"].startswith("prot-") and p["watching"] is True
        assert "trigger price is above the last traded price" in p["why"]
    assert node._rt.pending_rearms() == []
    # the stop is the engine's now: the next bar trades down to 85, through 90, and the 10 are
    # closed at market
    venue.refuse = False
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 17), (90, 90, 85, 85))
    assert market_closes(venue.requests[sent:]) == [("sell", 10.0)]


def test_a_leg_rejected_after_the_venue_took_it_falls_back_to_a_watched_level():
    # the stop sent again on a quiet first bar is accepted, then rejected; the next bar trades
    # down to 85, through the stop at 90: the engine closes the 10 at market
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten(journal=journal)
    lapse(node, stop, target)
    sent = len(venue.requests)
    ts = at(TUE, 9, 16)
    bar(node, ts)
    (new_stop,) = [r for r in venue.requests[sent:] if r.order_type == "stop_market"]
    report(node, new_stop.client_order_id, ExecutionReport.accepted("VS2", ts), ts)
    report(node, new_stop.client_order_id, ExecutionReport.rejected("trigger above last price", ts + 1), ts + 1)
    (rejected,) = rearm_records(journal, "rearm_rejected")
    assert rejected.payload["order"] == new_stop.client_order_id and rejected.payload["watching"] is True
    assert rejected.payload["why"] == "trigger above last price"
    after = len(venue.requests)
    bar(node, at(TUE, 9, 17), (85, 85, 85, 85))
    assert market_closes(venue.requests[after:]) == [("sell", 10.0)]


class Flaky(Venue):
    """Takes orders until `fail` is set, then raises as a dropped connection would."""

    fail = False

    def submit(self, r):
        if self.fail:
            raise ConnectionError("broker unreachable at the open")
        super().submit(r)


def test_a_leg_the_venue_could_not_be_reached_for_is_sent_on_a_later_bar():
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten(venue=Flaky(), journal=journal)
    lapse(node, stop, target)
    venue.fail = True
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 16))  # the node does not raise: the legs wait
    bar(node, at(TUE, 9, 17))
    assert {r[3] for r in node._rt.pending_rearms()} == {"stop", "target"}
    deferred = rearm_records(journal, "rearm_deferred")
    # once per leg, not once per bar of the outage
    assert sorted(r.payload["leg"] for r in deferred) == ["stop", "target"]
    assert all(r.payload["event_ts"] == at(TUE, 9, 16) for r in deferred)
    venue.fail = False
    bar(node, at(TUE, 9, 18))
    assert legs(venue.requests[sent:]) == both(10.0)
    assert node._rt.pending_rearms() == []
    rearmed = {r.payload["leg"]: r.payload["order"] for r in rearm_records(journal, "rearmed")}
    # each is the order that could not be sent, sent again under its own id
    assert rearmed == {r.payload["leg"]: r.payload["order"] for r in deferred}


def test_a_leg_still_unsent_when_its_session_closes_is_given_up_and_journaled():
    journal = rj.Journal({"run": "overnight"}, memory=True)
    venue, node, stop, target = long_ten(venue=Flaky(), journal=journal)
    lapse(node, stop, target)
    venue.fail = True
    sent = len(venue.requests)
    bar(node, at(TUE, 9, 16))
    # deferred on Tuesday; Tuesday's NFO session closes at 15:40, so Wednesday's first bar gives up
    bar(node, at(WED, 9, 16))
    abandoned = rearm_records(journal, "rearm_abandoned")
    assert sorted(r.payload["leg"] for r in abandoned) == ["stop", "target"]
    assert node._rt.pending_rearms() == []
    assert venue.requests[sent:] == []
