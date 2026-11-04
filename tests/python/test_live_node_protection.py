"""Protective orders on a real venue: closing past them, growing with an add, and being resized.

A venue whose resting stop or target counts against the position (``resting_exits_hold_the_line``)
refuses an order that closes the position while one rests. The engine takes the protection off
first, in the same breath, and watches the levels until the close is done.
"""

from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.execution import ExecutionReport, Unsupported
from raptorbt.strategy import orders as O

MIN = 60_000_000_000
T0 = 1_700_000_000_000_000_000
FLAT = (100, 101, 99, 100)


def config(trail=None):
    c = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")
    if trail is not None:
        c.set_trailing_stop(trail)
    return c


class Venue:
    namespace = "PROT"
    keeps_own_records = True

    def __init__(self):
        self.calls, self.requests, self.cancels, self.modifies = [], [], [], []

    def submit(self, r):
        self.requests.append(r)
        self.calls.append(("submit", r.client_order_id))

    def cancel(self, cid):
        self.cancels.append(cid)
        self.calls.append(("cancel", cid))

    def modify(self, cid, *, quantity=None, limit_price=None, trigger_price=None):
        self.modifies.append((cid, quantity))


class Holding(Venue):
    """A venue whose resting protective order holds the line."""

    resting_exits_hold_the_line = True


class Script(Strategy):
    def __init__(self, script):
        super().__init__()
        self.script = script

    def on_bar(self, ctx):
        act = self.script.get(ctx.idx)
        if act:
            act(self, ctx)


def node_on(venue, script, cfg=None):
    return LiveNode([Script(script)], ["A"], cfg or config(), bar_stamp="close", execution=venue, protective="venue")


def bar(node, i, row=FLAT):
    node.push_bar("A", T0 + (i + 1) * MIN, *map(float, row), 1e6)
    return T0 + (i + 1) * MIN


def bracketed_long(venue):
    node = node_on(venue, {1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0, target_price=110.0))})
    bar(node, 0)
    ts = bar(node, 1)
    (entry,) = venue.requests
    node.push_report(entry.client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    stop, target = venue.requests[1:]
    node.push_report(target.client_order_id, ExecutionReport.accepted("VT", ts), arrival_ns=ts)
    node.push_report(stop.client_order_id, ExecutionReport.accepted("VS", ts), arrival_ns=ts)
    return node, stop, target, ts


def closes(venue):
    return [r for r in venue.requests if r.reduce_only and r.order_type == "market"]


# -- a close with protection resting ---------------------------------------------------------


def test_a_close_cancels_the_resting_protection_ahead_of_itself_where_it_holds_the_line():
    venue = Holding()
    node, stop, target, _ = bracketed_long(venue)
    before = len(venue.calls)
    node.halt(0, flatten=True)
    bar(node, 2)
    sent = venue.calls[before:]
    (close,) = closes(venue)
    at = sent.index(("submit", close.client_order_id))
    assert {("cancel", stop.client_order_id), ("cancel", target.client_order_id)} <= set(sent[:at]), sent


def test_a_close_leaves_the_protection_alone_where_it_does_not_hold_the_line():
    venue = Venue()
    node, *_ = bracketed_long(venue)
    node.halt(0, flatten=True)
    bar(node, 2)
    assert venue.cancels == [] and len(closes(venue)) == 1


def test_a_close_the_venue_refuses_leaves_the_levels_watched_by_the_engine():
    venue = Holding()
    node, stop, target, ts = bracketed_long(venue)
    node.halt(0, flatten=True)
    ts = bar(node, 2)
    (close,) = closes(venue)
    node.push_report(close.client_order_id, ExecutionReport.rejected("the close never placed", ts), arrival_ns=ts)
    bar(node, 3, (100, 100, 94, 95))  # touches the stop at 95
    bar(node, 4)
    assert len(closes(venue)) == 2, [r.client_order_id for r in venue.requests]


# -- an add is protected by legs of its own -----------------------------------------------------


def test_an_add_gets_a_stop_of_its_own_where_a_leg_is_the_size_of_its_entry():
    venue = Holding()
    node = node_on(
        venue,
        {
            1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
            2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
        },
    )
    bar(node, 0)
    ts = bar(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    ts = bar(node, 2)
    add = venue.requests[2]
    assert not add.reduce_only
    node.push_report(add.client_order_id, ExecutionReport.fill("E2", 10.0, 101.0, ts), arrival_ns=ts)
    stops = [r for r in venue.requests if r.reduce_only and r.order_type == "stop_market"]
    assert [(r.quantity, r.trigger_price) for r in stops] == [(10.0, 95.0), (10.0, 95.0)]
    assert venue.modifies == [], "the first leg is not grown to the whole position"


def _stop_then_add(venue, accept_before_add):
    node = node_on(
        venue,
        {
            1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
            2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
        },
    )
    bar(node, 0)
    ts = bar(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    stop = venue.requests[1]
    if accept_before_add:
        node.push_report(stop.client_order_id, ExecutionReport.accepted("VS", ts), arrival_ns=ts)
    ts = bar(node, 2)
    add = venue.requests[2]
    node.push_report(add.client_order_id, ExecutionReport.fill("E2", 10.0, 101.0, ts), arrival_ns=ts)
    return node, stop, ts


def test_an_add_grows_the_leg_where_the_venue_does_not_hold_the_line():
    venue = Venue()
    _stop_then_add(venue, accept_before_add=True)
    assert [q for _, q in venue.modifies] == [20.0]


def test_a_leg_the_venue_has_not_accepted_is_grown_once_it_is():
    """A change to an order still Submitted is not a legal transition: it waits for the venue."""
    venue = Venue()
    node, stop, ts = _stop_then_add(venue, accept_before_add=False)
    assert venue.modifies == []
    node.push_report(stop.client_order_id, ExecutionReport.accepted("VS", ts), arrival_ns=ts)
    assert [q for _, q in venue.modifies] == [20.0]


def test_a_watcher_of_the_bare_part_closes_that_part_and_not_the_whole_position():
    """Long 10 with a stop the venue holds, 10 more added whose stop the venue cannot hold (the
    engine watches it). The watcher covers the 10 the venue's leg does not: when it fires it sells
    10, not 20, and the sold quantity ends at the position, 20, with no flip."""

    class OneStop(Holding):
        def submit(self, r):
            if r.reduce_only and r.order_type == "stop_market" and any(
                x.reduce_only and x.order_type == "stop_market" for x in self.requests
            ):
                raise Unsupported("one stop at a time")
            super().submit(r)

    venue = OneStop()
    node = node_on(
        venue,
        {
            1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
            2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
        },
    )
    bar(node, 0)
    ts = bar(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    stop = venue.requests[1]
    node.push_report(stop.client_order_id, ExecutionReport.accepted("VS", ts), arrival_ns=ts)
    ts = bar(node, 2)
    node.push_report(venue.requests[2].client_order_id, ExecutionReport.fill("E2", 10.0, 101.0, ts), arrival_ns=ts)
    # the venue's leg is booked again after the add: the watcher's size is made again there
    node.push_report(stop.client_order_id, ExecutionReport.accepted("VS", ts), arrival_ns=ts)
    bar(node, 3, (100, 100, 94, 95))  # the level of both
    bar(node, 4)
    ts = bar(node, 5)
    sells = closes(venue)
    assert [r.quantity for r in sells] == [10.0, 10.0], "the watcher sold its own part, then the rest"
    for n, r in enumerate(sells):
        node.push_report(r.client_order_id, ExecutionReport.fill(f"X{n}", r.quantity, 94.0, ts), arrival_ns=ts)
    assert node.positions(0, "A") == [] and node.health()["unbooked_fills"] == []


def test_a_stop_sent_again_at_a_new_level_still_cancels_with_its_target():
    """The stop was not accepted when an add moved it, so it is cancelled and sent again at the
    new level. It stays one-cancels-other with the target: a target that fills takes it off."""
    venue = Holding()
    node = node_on(
        venue,
        {
            1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0, target_price=110.0)),
            2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=102.0)),
        },
    )
    bar(node, 0)
    ts = bar(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    old_stop, target = venue.requests[1:3]
    ts = bar(node, 2)
    add = venue.requests[3]  # the add is sent by bar 2's hook, after the two legs
    node.push_report(add.client_order_id, ExecutionReport.fill("E2", 10.0, 101.0, ts), arrival_ns=ts)
    assert venue.cancels == [old_stop.client_order_id]
    (new_stop,) = [r for r in venue.requests if r.order_type == "stop_market" and r.trigger_price == 102.0][:1]
    node.push_report(target.client_order_id, ExecutionReport.accepted("VT", ts), arrival_ns=ts)
    node.push_report(target.client_order_id, ExecutionReport.fill("T1", 10.0, 110.0, ts), arrival_ns=ts)
    assert new_stop.client_order_id in venue.cancels


# -- a leg that has partly filled is resized by what it has left ------------------------------------


def test_a_partly_filled_target_is_not_resized_below_what_it_still_has_to_sell():
    venue = Venue()
    node, stop, target, ts = bracketed_long(venue)
    node.push_report(target.client_order_id, ExecutionReport.fill("T1", 4.0, 110.0, ts), arrival_ns=ts)
    # position 6: the target has 6 to go (10 total, 4 done) and stays; the stop of 10 goes to 6
    assert [(c, q) for c, q in venue.modifies if c == target.client_order_id] == []
    assert [(c, q) for c, q in venue.modifies if c == stop.client_order_id] == [(stop.client_order_id, 6.0)]


def test_a_partly_filled_target_that_must_shrink_is_sent_what_it_has_left_plus_what_it_filled():
    venue = Venue()
    node, stop, target, ts = bracketed_long(venue)
    node.push_report(target.client_order_id, ExecutionReport.fill("T1", 4.0, 110.0, ts), arrival_ns=ts)
    for cid, q in venue.modifies:
        node.push_report(cid, ExecutionReport.replace_ack(ts, q, None, None), arrival_ns=ts)
    node.push_report(target.client_order_id, ExecutionReport.fill("T2", 6.0, 110.0, ts), arrival_ns=ts)
    assert node.positions(0, "A") == []
    assert node.health()["unbooked_fills"] == []


# -- a refused flatten close is sent again --------------------------------------------------------


def test_a_refused_flatten_close_can_be_sent_again():
    venue = Venue()
    node = node_on(venue, {1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0))})
    bar(node, 0)
    ts = bar(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    node.halt(0, flatten=True)
    ts = bar(node, 2)
    (first,) = closes(venue)
    node.push_report(first.client_order_id, ExecutionReport.rejected("broker refused", ts), arrival_ns=ts)
    node.halt(0, flatten=True)
    bar(node, 3)
    assert len(closes(venue)) == 2


def test_a_flatten_close_still_working_is_not_sent_twice():
    venue = Venue()
    node = node_on(venue, {1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0))})
    bar(node, 0)
    ts = bar(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    node.halt(0, flatten=True)
    bar(node, 2)
    node.halt(0, flatten=True)
    bar(node, 3)
    assert len(closes(venue)) == 1


# -- M2: a trailing stop the venue refuses is still a trailing stop ---------------------------------


def test_a_trailing_stop_a_venue_cannot_hold_still_trails_when_the_engine_watches_it():
    class NoTrail(Venue):
        def submit(self, r):
            if r.order_type == "trailing_stop_market":
                raise Unsupported("no trailing orders")
            super().submit(r)

    venue = NoTrail()
    node = node_on(venue, {1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0))}, config(trail=0.10))
    bar(node, 0)
    ts = bar(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    bar(node, 2)
    bar(node, 3, (195, 200, 190, 200))
    bar(node, 4, (190, 190, 175, 176))  # the level trails to 180: this bar reaches it
    assert closes(venue), [r.order_type for r in venue.requests]
