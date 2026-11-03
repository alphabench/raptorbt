"""A structure group's close: a partly filled leg's remainder is cancelled and what filled is closed,
a late fill is closed too, a refused close is tried again on a later step, the sold legs go first,
and a subclass that calls ``super().on_order_event`` hears each fill once."""

import pytest

from raptorbt import BacktestConfig, InstrumentSpec, LiveNode, MarginModel, OptionChain, Strategy, run_strategies
from raptorbt import structures as st
from raptorbt.execution import ExecutionReport, PaperExecution

from _chain_fixture import N_BARS, U, specs, universe
from test_structure_hedge_first import (
    Broker, Opens, bear_call_spread, config, iron_condor, make, push_bar, run_mode,
)

CHAIN = OptionChain(specs())
OHLCV = ("open", "high", "low", "close", "volume")


def _node(strategy, broker, capital=1e6):
    data, inst = universe()
    node = LiveNode([strategy], list(data), config(capital), execution=broker, instruments=inst,
                    account_type="margin", leverage=1.0, bar_stamp="close")
    return node, data


def _net(result):
    net = {}
    for o in result.result.orders():
        if o.status in ("filled", "partially_filled") or o.filled_qty:
            net[o.symbol] = net.get(o.symbol, 0.0) + (o.filled_qty if o.side == "buy" else -o.filled_qty)
    return {k: v for k, v in net.items() if abs(v) > 1e-9}


def _opening(request):
    return not request.reduce_only


# ---- a partly filled leg ---------------------------------------------------------------


def test_a_sold_leg_filled_in_part_has_its_remainder_cancelled_before_it_is_closed_in_a_broker_run():
    s = make(iron_condor(), lots=2)
    broker = Broker()
    node, data = _node(s, broker)
    handled, part, acked = set(), None, False
    for i in range(N_BARS):
        push_bar(node, data, i)
        ts = int(data[U]["timestamps"][i])
        for req in list(broker.requests):
            cid = req.client_order_id
            if cid in handled:
                continue
            handled.add(cid)
            px = float(data[req.symbol]["close"][i])
            node.push_report(cid, ExecutionReport.accepted(f"V{cid}", ts), arrival_ns=ts)
            if req.reduce_only or req.side == "buy":
                node.push_report(cid, ExecutionReport.fill(f"f{cid}", req.quantity, px, ts), arrival_ns=ts)
            elif req.symbol.endswith("CE"):
                node.push_report(cid, ExecutionReport.fill(f"f{cid}a", req.quantity / 2, px, ts), arrival_ns=ts)
                part = cid
            else:
                node.push_report(cid, ExecutionReport.rejected("RMS: margin", ts), arrival_ns=ts)
        if part in broker.cancelled and not acked:
            # the broker confirms the cancel a bar after it was sent
            assert s.structure_status(s.group) == "closing", "not closed while a remainder is working"
            node.push_report(part, ExecutionReport.cancel_ack(ts), arrival_ns=ts)
            acked = True
    assert part in broker.cancelled, "the working remainder of the partly filled sold call was cancelled"
    closes = [r for r in broker.requests if r.reduce_only]
    sold_close = next(r for r in closes if r.side == "buy")
    assert sold_close.quantity == 50.0, "the 50 that filled, nothing else"
    assert broker.requests.index(sold_close) < min(broker.requests.index(r) for r in closes if r.side == "sell"), \
        "the sold leg is closed before the hedges that cover it"
    assert s.structure_status(s.group) == "closed"
    assert _net(node.finish()[0]) == {}, "nothing is left open"


def test_the_rest_of_a_hedge_that_fills_after_the_group_closed_is_closed_too():
    s = make(bear_call_spread(), lots=2)
    broker = Broker()
    node, data = _node(s, broker)
    handled, hedge = set(), None
    late = []
    for i in range(N_BARS):
        push_bar(node, data, i)
        ts = int(data[U]["timestamps"][i])
        for req in list(broker.requests):
            cid = req.client_order_id
            if cid in handled:
                continue
            handled.add(cid)
            px = float(data[req.symbol]["close"][i])
            node.push_report(cid, ExecutionReport.accepted(f"V{cid}", ts), arrival_ns=ts)
            if req.reduce_only:
                node.push_report(cid, ExecutionReport.fill(f"f{cid}", req.quantity, px, ts), arrival_ns=ts)
            elif req.side == "buy":
                node.push_report(cid, ExecutionReport.fill(f"f{cid}a", req.quantity / 2, px, ts), arrival_ns=ts)
                hedge = (cid, req.symbol, req.quantity / 2)
            else:
                node.push_report(cid, ExecutionReport.rejected("RMS: margin", ts), arrival_ns=ts)
        if hedge is not None and not late and i >= 8:
            cid, sym, q = hedge
            assert cid in broker.cancelled, "the hedge's working remainder was cancelled"
            # the broker fills the rest before it hears of the cancel
            node.push_report(cid, ExecutionReport.fill(f"f{cid}b", q, float(data[sym]["close"][i]), ts), arrival_ns=ts)
            late.append(i)
    closes = [r for r in broker.requests if r.reduce_only and r.side == "sell"]
    assert [r.quantity for r in closes] == [50.0, 50.0], "the late 50 got a close of its own"
    assert s.structure_status(s.group) == "closed"
    assert _net(node.finish()[0]) == {}


class _Ev:
    def __init__(self, kind, cid, size=None, price=None, reason=None):
        self.kind, self.client_order_id, self.size, self.price = kind, cid, size, price
        self.reject_reason = reason


class _Fake(Strategy):
    def __init__(self):
        super().__init__()
        self.sent, self.cancels = [], []

    def submit_order(self, order, client_id=None):
        self.sent.append((client_id, order.side, order.quantity, bool(getattr(order, "reduce_only", False))))

    def cancel_order(self, client_id):
        self.cancels.append(client_id)


def _fake_book():
    """A book driven by hand-written order events, which is all it reads."""
    from raptorbt.structures.book import StructureBook

    s = _Fake()
    book = StructureBook(s)
    g = book.submit(bear_call_spread().resolve(CHAIN, U, 100.0, 1, 2), now_ns=1)

    def hear(ev, ns):
        s.cache._observe(ev)
        book.on_event(ev, ns)

    s.structure_status = lambda group: group.status(s.cache)
    return s, book, g, hear, _Ev


def test_a_working_remainder_is_cancelled_and_the_filled_part_closed_from_the_events_alone():
    # bar data never gives a partial fill, so the events are written out
    s, book, g, hear, Ev = _fake_book()
    hedge, = [ls for ls in g.legs if ls.leg.side == "buy"]
    sold, = [ls for ls in g.legs if ls.leg.side == "sell"]

    hear(Ev("order_accepted", hedge.client_id), 1)
    hear(Ev("order_filled", hedge.client_id, 100.0, 5.0), 1)
    hear(Ev("order_accepted", sold.client_id), 1)
    hear(Ev("order_filled", sold.client_id, 50.0, 7.0), 1)  # half of the sold call
    book.close(g, "manual")
    assert sold.client_id in s.cancels, "the sold call's remainder is cancelled"
    assert (f"{g.id}:x{sold.index}.0", "buy", 50.0, True) in s.sent
    assert g.phase == "closing", "not closed while the remainder may still fill"
    hear(Ev("order_filled", sold.client_id, 50.0, 7.0), 2)  # the rest fills before the cancel lands
    assert (f"{g.id}:x{sold.index}.1", "buy", 50.0, True) in s.sent, "the late fill is closed too"
    for cid, ns in ((f"{g.id}:x{sold.index}.0", 2), (f"{g.id}:x{sold.index}.1", 2)):
        hear(Ev("order_accepted", cid), ns)
        hear(Ev("order_filled", cid, 50.0, 7.0), ns)
    assert (f"{g.id}:x{hedge.index}.0", "sell", 100.0, True) in s.sent, "the hedge goes once the sold leg is closed"
    hear(Ev("order_accepted", f"{g.id}:x{hedge.index}.0"), 2)
    hear(Ev("order_filled", f"{g.id}:x{hedge.index}.0", 100.0, 5.0), 2)
    assert g.phase == "closed"
    hear(Ev("order_filled", sold.client_id, 10.0, 7.0), 3)  # a fill after the group read closed
    assert g.phase == "closing" and (f"{g.id}:x{sold.index}.2", "buy", 10.0, True) in s.sent


# ----------------------------------------------------------------------------------------


class Plain(Opens):
    structure = bear_call_spread()
    kwargs: dict = {}


class WithSuper(Plain):
    def on_order_event(self, ctx, event):
        super().on_order_event(ctx, event)


class WithoutSuper(Plain):
    def on_order_event(self, ctx, event):
        pass


@pytest.mark.parametrize("cls", [Plain, WithSuper, WithoutSuper])
@pytest.mark.parametrize("mode", ["batch", "paper"])
def test_a_fill_is_booked_once_whether_or_not_the_subclass_calls_super(cls, mode):
    s, r, _ = run_mode(mode, cls(), 1e6)
    assert [ls.filled for ls in s.group.legs] == [ls.leg.quantity for ls in s.group.legs]


class ClosesAt8(Plain):
    def on_bar(self, ctx):
        super().on_bar(ctx)
        if ctx.symbol == U and ctx.idx == 8 and self.group is not None:
            self.close_structure(self.group)


class ClosesAt8Super(ClosesAt8):
    def on_order_event(self, ctx, event):
        super().on_order_event(ctx, event)


@pytest.mark.parametrize("mode", ["batch", "paper"])
def test_closing_a_structure_is_the_same_with_and_without_super(mode):
    a = run_mode(mode, ClosesAt8(), 1e6)
    b = run_mode(mode, ClosesAt8Super(), 1e6)
    for s, r, _ in (a, b):
        assert s.structure_status(s.group) == "closed"
        assert not [o for o in r.result.orders() if o.status == "rejected"]
    assert sorted((o.client_id, o.status) for o in a[1].result.orders()) == sorted(
        (o.client_id, o.status) for o in b[1].result.orders())


# ---- close sequencing ----------------------------------------------------------------------


class Closes(Strategy):
    def __init__(self, structure, spot, spec=None):
        super().__init__()
        self.structure, self.spot, self.spec, self.group = structure, spot, spec, None
        self.steps = []

    def on_bar(self, ctx):
        if ctx.symbol != U:
            return
        if self.group is None and ctx.idx == 3:
            kw = dict(underlying_spec=self.spec) if self.spec else {}
            self.group = self.submit_structure(ctx, self.structure, chain=CHAIN, underlying=U, spot=self.spot, **kw)
        if ctx.idx == 8 and self.group is not None:
            self.close_structure(self.group)
        if self.group is not None:
            self.update_structures(ctx, exits=False)

    def on_order_event(self, ctx, event):
        if event.client_order_id and ":x" in event.client_order_id:
            self.steps.append((ctx.timestamp, event.client_order_id, event.kind))


def _run(mode, strategy, capital=1000.0, margin=True):
    data, inst = universe()
    if strategy.spec is not None:
        inst = dict(inst, **{U: strategy.spec})
    cfg = config(capital, margin)
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    if mode == "batch":
        return run_strategies([strategy], data, cfg, **kw)[0]
    node = LiveNode([strategy], list(data), cfg, execution=PaperExecution(), **kw)
    for i in range(N_BARS):
        push_bar(node, data, i)
    return node.finish()[0]


CASES = {
    "call_spread_95_100": (lambda: st.bear_call_spread(short_steps=0, long_steps=1), 95.0, None),
    "call_spread_100_105": (lambda: st.bear_call_spread(short_steps=0, long_steps=1), 100.0, None),
    "calendar": (lambda: st.calendar("call", 0), 100.0, None),
    "covered_call": (lambda: st.covered_call(1), 100.0, InstrumentSpec.equity(U, lot_size=1.0)),
}
#: Structures the defined-risk model cannot margin (a calendar, shares) run on the cash rule.
FUNDED = {"calendar", "covered_call"}


@pytest.mark.parametrize("mode", ["batch", "paper"])
@pytest.mark.parametrize("case", list(CASES))
def test_a_structure_closes_flat_whatever_the_strikes_with_the_sold_legs_first(case, mode):
    make_structure, spot, spec = CASES[case]
    s = Closes(make_structure(), spot, spec)
    r = _run(mode, s, 100_000.0 if case in FUNDED else 1000.0, margin=case not in FUNDED)
    g = s.group
    assert s.structure_status(g) == "closed", (s.structure_status(g), s.steps)
    orders = r.result.orders()
    assert not [o for o in orders if o.status == "rejected"], [(o.client_id, o.reject_reason) for o in orders]
    assert _net(r) == {}
    filled = [cid for _, cid, kind in s.steps if kind == "order_filled"]
    sold = [f"{g.id}:x{ls.index}.0" for ls in g.legs if ls.leg.side == "sell"]
    assert filled[: len(sold)] == sold, "the sold legs were closed first"


def test_a_refused_close_is_tried_again_on_a_later_step_never_in_the_step_it_was_refused():
    s, book, g, hear, Ev = _fake_book()
    sold, = [ls for ls in g.legs if ls.leg.side == "sell"]
    hedge, = [ls for ls in g.legs if ls.leg.side == "buy"]
    for ls in sorted(g.legs, key=lambda x: x.leg.side != "buy"):
        hear(Ev("order_accepted", ls.client_id), 1)
        hear(Ev("order_filled", ls.client_id, ls.leg.quantity, 5.0), 1)
    book.close(g, "manual")
    sent = lambda: [c for c in s.sent if c[0].startswith(f"{g.id}:x{sold.index}")]  # noqa: E731
    assert len(sent()) == 1
    for attempt in range(1, 3):
        cid = f"{g.id}:x{sold.index}.{attempt - 1}"
        hear(Ev("order_rejected", cid, reason="insufficient_margin"), 10 * attempt)
        assert len(sent()) == attempt, "no second try in the step of the refusal"
        book.update(10 * attempt, exits=False)
        assert len(sent()) == attempt, "none in the same step by the next call either"
        book.update(10 * attempt + 1, exits=False)
        assert len(sent()) == attempt + 1, "the next step tries again"
    hear(Ev("order_rejected", f"{g.id}:x{sold.index}.2", reason="insufficient_margin"), 40)
    book.update(41, exits=False)
    assert len(sent()) == 3 and s.structure_status(g) == "stuck"
    assert not [c for c in s.sent if c[0].startswith(f"{g.id}:x{hedge.index}")], "the cover stays on"


# ---- M4: the order-event hook is switched on by the first structure group ----------------


def _calls_of_base_hook(monkeypatch, strategy):
    calls = []
    original = Strategy.on_order_event

    def counting(self, ctx, event):
        calls.append(event.kind)
        original(self, ctx, event)

    monkeypatch.setattr(Strategy, "on_order_event", counting)
    from raptorbt.strategy import api

    api._install_base(Strategy)
    data, inst = universe()
    run_strategies([strategy], data, config(1e6), instruments=inst, account_type="margin", leverage=1.0,
                   bar_stamp="close")
    return calls


class _Orders(Strategy):
    def on_bar(self, ctx):
        if ctx.symbol == U and ctx.idx == 3:
            from raptorbt.strategy.orders import Market

            self.submit_order(Market(side="buy", quantity=1.0, instrument=U), client_id="plain")


def test_a_strategy_with_no_structures_is_not_called_on_order_events(monkeypatch):
    assert _calls_of_base_hook(monkeypatch, _Orders()) == []


def test_a_strategy_that_places_structures_is_called_on_order_events(monkeypatch):
    assert _calls_of_base_hook(monkeypatch, make(bear_call_spread())) != []


def _helper(strategy, ctx):
    return strategy.submit_structure(ctx, bear_call_spread(), chain=CHAIN, underlying=U, spot=ctx.bar.close)


class _ViaHelper(Strategy):
    def __init__(self):
        super().__init__()
        self.group = None

    def on_bar(self, ctx):
        if ctx.symbol == U and self.group is None and ctx.idx == 3:
            self.group = _helper(self, ctx)


def test_a_structure_placed_by_a_helper_hears_its_fills_with_no_declaration(monkeypatch):
    s = _ViaHelper()
    assert _calls_of_base_hook(monkeypatch, s) != []
    assert [ls.filled for ls in s.group.legs] == [ls.leg.quantity for ls in s.group.legs]
