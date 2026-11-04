"""Hand-computed pins for resting orders and levels. Every number is computed by hand
from the bars written here. T0 bars are 1 minute, stamped at their close; fees are zero."""

import pytest
from test_venue_parity import MIN, PATHS, T0, bars, run

from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategy_backtest
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders as O

FLAT = (100.0, 101.0, 99.0, 100.0)


def nbo():
    return BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")


def sbc():
    return BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="same_bar_close")


class Mover(Strategy):
    def __init__(self, level):
        super().__init__()
        self.level, self.n = level, 0

    def on_bar(self, ctx, bar=None):
        if self.n == 0:
            self.submit_order(O.Limit(side="buy", units=10.0, price=85.0), client_id="m")
        if self.n == 2:
            self.modify_order("m", limit_price=self.level)
        self.n += 1


def fill_of(result, cid):
    return [(o.avg_fill_price, o.last_fill_idx) for o in result.result.orders() if o.client_id == cid and o.filled_qty > 0]


@pytest.mark.parametrize("path", PATHS)
def test_a_limit_moved_through_the_close_of_the_bar_that_decided_it_arrives_at_the_next_open(path):
    """Close of bar 1 = 100, close of bar 2 = 90. On bar 2 a buy limit moves 85 -> 95: through bar 2's
    close, what the strategy decided on and what a venue crosses at. Bar 3 opens at 91: the fill is
    91, at the open (event-semantics "The rest of a bar"). 10 x 91 = 910. Before the fix: modify_order reads
    last_close, which step_rest leaves at bar 1's 100, so 95 is 'not through', the order keeps its
    old arrival and fills at its limit: 95, 10 x 95 = 950, Rs 40 worse than a venue."""
    rows = [FLAT, FLAT, (100.0, 100.0, 89.0, 90.0), (91.0, 92.0, 90.0, 91.0), (91.0, 92.0, 90.0, 91.0)]
    out = run({}, {"X": bars(rows, T0)}, nbo, paths=(path,), strategy=lambda: Mover(95.0))
    assert fill_of(out[path], "m") == [(91.0, 3)]


@pytest.mark.parametrize("path", PATHS)
def test_a_limit_moved_to_a_level_the_decision_bar_closed_above_keeps_resting(path):
    """Close of bar 1 = 90, close of bar 2 = 100. On bar 2 a buy limit moves 85 -> 95: below bar 2's
    close, not through. It rests; bar 3 (94, 96, 93, 95) opens past it mid-session and a resting
    limit fills at its limit, 95 (fill.rs get_limit_fill_price: 'never better'). Before the fix:
    the stale 90 reads as through, the order 'arrives' and fills at the open, 94: Rs 10 better on 10."""
    rows = [FLAT, (100.0, 100.0, 89.0, 90.0), (90.0, 101.0, 89.0, 100.0), (94.0, 96.0, 93.0, 95.0), (95.0, 96.0, 94.0, 95.0)]
    out = run({}, {"X": bars(rows, T0)}, nbo, paths=(path,), strategy=lambda: Mover(95.0))
    assert fill_of(out[path], "m") == [(95.0, 3)]


GAP = [FLAT, FLAT, FLAT, (90.0, 92.0, 85.0, 86.0), (87.0, 88.0, 86.0, 87.0), (87.0, 88.0, 86.0, 87.0)]


class OwnStop(Strategy):
    """Long 10 at bar 2's open with the order's own stop at 95. The hook of the close buys 5 back."""

    def __init__(self):
        super().__init__()
        self.heard = None

    def on_bar(self, ctx, bar=None):
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0), client_id="in")

    def on_position_event(self, ctx, e):
        if e.kind == "exited" and self.heard is None:
            self.heard = ctx.bar.close
            self.submit_order(O.Market(side="buy", units=5.0), client_id="ans")


@pytest.mark.parametrize("path", PATHS)
def test_the_answer_to_an_order_s_own_stop_the_bar_gapped_through_fills_at_that_open(path):
    """Long 10 at 100, stop_price 95. Bar 3 gaps to (90, 92, 85, 86): out at 90, -100. The hook buys 5:
    90, as a resting stop's answer does (test_rest_before_decide) and as a venue would. Predicted
    failure: `heard` looks for OrderFilled only and a position's own stop is an Exited event, so batch
    and node answer at bar 4's open, 87, and paper (a prot- order, OrderFilled) at 90."""
    out = run({}, {"X": bars(GAP, T0)}, nbo, paths=(path,), strategy=OwnStop)
    assert fill_of(out[path], "ans") == [(90.0, 3)]


@pytest.mark.parametrize("path", PATHS)
def test_the_hook_of_an_order_s_own_stop_inside_a_bar_does_not_read_that_bar(path):
    """Bar 2 closes at 100; bar 3 (100, 100, 94, 96) touches the stop at 95. The hook runs before the
    bar has closed (event-semantics "When hooks hear a match, and what they know"): ctx.bar is bar 2, close 100. Predicted
    failure: as above no fill is seen, the hook is not blind and reads bar 3's close, 96."""
    rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0), (96.0, 97.0, 95.0, 96.0)]
    made = []

    def make():
        made.append(OwnStop())
        return made[-1]

    run({}, {"X": bars(rows, T0)}, nbo, paths=(path,), strategy=make)
    assert made[0].heard == 100.0


class Answered(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = {}

    def on_bar(self, ctx, bar=None):
        self.seen[ctx.idx] = sum(p.size for p in ctx.positions())
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=10.0), client_id="in")
        if ctx.idx == 2:
            self.submit_order(O.StopMarket(side="sell", units=10.0, trigger=95.0), client_id="stop")

    def on_order_event(self, ctx, e):
        if e.kind == "order_filled" and e.client_order_id == "stop":
            self.submit_order(O.Market(side="buy", units=5.0), client_id="ans")


@pytest.mark.parametrize("path", PATHS)
def test_on_bar_sees_the_answer_that_filled_at_the_open_of_its_bar(path):
    """A resting stop gaps at bar 3: sold 10 at 90, the hook buys 5, filled at that open, 90. Live both
    are reported before bar 3 closes, so on_bar(3) reads long 5 and a strategy that re-enters when flat
    stays out. Before the fix: fill_follow_up finds last_step still at bar 2 (step_rest does not move
    it) and leaves the answer to bar 3's step, after on_bar(3): it reads 0 and re-enters on a phantom
    flat, with the answer priced at an open it has not yet booked."""
    made = []

    def make():
        made.append(Answered())
        return made[-1]

    out = run({}, {"X": bars(GAP, T0)}, nbo, paths=(path,), strategy=make)
    assert fill_of(out[path], "ans") == [(90.0, 3)]
    assert made[0].seen[3] == 5.0, made[0].seen


class Venue:
    namespace = "QCB"
    keeps_own_records = True
    resting_exits_hold_the_line = True

    def __init__(self):
        self.requests, self.cancels, self.modifies = [], [], []

    def submit(self, r):
        self.requests.append(r)

    def cancel(self, cid):
        self.cancels.append(cid)

    def modify(self, cid, *, quantity=None, limit_price=None, trigger_price=None):
        self.modifies.append((cid, quantity))


class Script(Strategy):
    def __init__(self, script):
        super().__init__()
        self.script = script

    def on_bar(self, ctx):
        act = self.script.get(ctx.idx)
        if act:
            act(self, ctx)


def push(node, i, row=FLAT):
    node.push_bar("A", T0 + (i + 1) * MIN, *map(float, row), 1e6)
    return T0 + (i + 1) * MIN


def test_a_plain_sale_after_an_add_leaves_no_more_resting_than_the_position():
    """Long 10 with a stop at 95, add 10: legs of 10 and 10 (extend_protection). A plain (not reduce-only)
    sell of 5 fills: position 15. Each leg is judged alone against the whole 15, 10 < 15 reads as
    'grows' and neither shrinks: 20 of stops on 15. A gap through 95 then sells 20: flat in a backtest
    (15 x (95 - 100) = -75), short 5 live. Expected: the legs' remainders sum to at most 15."""
    venue = Venue()
    node = LiveNode(
        [Script({
            1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
            2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
            3: lambda s, c: s.submit_order(O.Market(side="sell", units=5.0)),
        })],
        ["A"], nbo(), bar_stamp="close", execution=venue, protective="venue",
    )  # fmt: skip
    push(node, 0)
    ts = push(node, 1)
    node.push_report(venue.requests[0].client_order_id, ExecutionReport.fill("E1", 10.0, 100.0, ts), arrival_ns=ts)
    node.push_report(venue.requests[1].client_order_id, ExecutionReport.accepted("S1", ts), arrival_ns=ts)
    ts = push(node, 2)
    node.push_report(venue.requests[2].client_order_id, ExecutionReport.fill("E2", 10.0, 100.0, ts), arrival_ns=ts)
    node.push_report(venue.requests[3].client_order_id, ExecutionReport.accepted("S2", ts), arrival_ns=ts)
    ts = push(node, 3)
    sale = venue.requests[4]
    assert not sale.reduce_only
    node.push_report(sale.client_order_id, ExecutionReport.fill("X1", 5.0, 100.0, ts), arrival_ns=ts)
    push(node, 4)
    assert [p.size for p in node.positions(0, "A")] == [15.0]
    sizes = {r.client_order_id: r.quantity for r in venue.requests if r.order_type == "stop_market" and r.client_order_id not in venue.cancels}
    sizes.update({cid: q for cid, q in venue.modifies if q is not None and cid in sizes})
    assert sum(sizes.values()) <= 15.0, sizes


@pytest.mark.parametrize("cfg", [nbo, sbc])
def test_a_fill_hook_in_the_single_symbol_runner_sees_the_bar_but_cannot_trade_inside_it(cfg):
    """run_strategy_backtest's rest hook reads bar 3 whole (its comment says so). What it can reach: a buy
    limit at 94.5 placed from the hook of the stop at 95 on bar 3 (100, 100, 94, 96) is inside the bar's
    range. It must not fill on bar 3 and bars 4 and 5 stay above it: one trade, no fill of `peek`."""
    rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0), (96.0, 97.0, 95.0, 96.0)]

    class Peek(Strategy):
        peeked = None

        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0))

        def on_position_event(self, ctx, e):
            if e.kind == "exited":
                type(self).peeked = float(ctx.close[-1])
                self.submit_order(O.Limit(side="buy", units=10.0, price=94.5), client_id="peek")

    r = run_strategy_backtest(Peek(), **bars(rows, T0), symbol="X", config=cfg(), bar_stamp="close")
    assert Peek.peeked == 96.0
    assert [o.filled_qty for o in r.orders() if o.client_id == "peek"] == [0.0]
