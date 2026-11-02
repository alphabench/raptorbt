"""Execution adapters: one strategy, three ways to run it.

* the same ``Strategy`` runs unchanged through ``run_strategies``, a
  ``LiveNode`` that matches in-process, and a ``LiveNode`` on
  ``PaperExecution``, and the three agree;
* on the frozen stream fixtures paper equals the in-process simulator fill
  for fill, in every figure the result reports;
* a venue that duplicates, delays and reorders its reports leaves the same
  orders, positions and cash as one that does not;
* an order is never sent twice, and its id depends on nothing but the
  strategy and the count of orders it has sent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt.errors import (
    InvalidData,
    InvalidOrder,
    InvalidParameter,
    StateError,
    UnsupportedOperation,
)
from raptorbt.execution import (
    ExecutionAdapter,
    ExecutionReport,
    OrderRefused,
    OrderRequest,
    PaperExecution,
    SimulatedExecution,
    Unsupported,
)
from raptorbt.strategy import orders

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "golden"))

import runner_fixtures as rf  # noqa: E402

MIN = 60_000_000_000
T0 = 1_700_000_000_000_000_000
NS = rf.NS
FIXTURES = rf.load()
DATASETS = FIXTURES["datasets"]
SCENARIOS = {sc["name"]: sc for sc in rf.scenario_table()}
S0, S1, S2 = rf.S0, rf.S1, rf.S2
ALL = [S0, S1, S2]


# -- data ----------------------------------------------------------------------


def make_bars(symbols, n=300, seed=3):
    rng = np.random.default_rng(seed)
    out = {}
    for k, s in enumerate(symbols):
        ts = (np.arange(n, dtype=np.int64) + 1) * MIN + T0
        close = np.maximum(100 + 10 * k + np.cumsum(rng.normal(0, 0.8, n)), 1.0)
        open_ = np.concatenate([[close[0]], close[:-1]])
        out[s] = {
            "timestamps": ts,
            "open": open_,
            "high": np.maximum(open_, close) + rng.random(n),
            "low": np.maximum(np.minimum(open_, close) - rng.random(n), 0.5),
            "close": close,
            "volume": rng.random(n) * 1000 + 100,
        }
    return out


def push_bars(node, data, symbols):
    """Every bar of every symbol, in time order, ties by symbol order."""
    rows = sorted(
        (int(data[s]["timestamps"][i]), k, i)
        for k, s in enumerate(symbols)
        for i in range(len(data[s]["timestamps"]))
    )
    for ts, k, i in rows:
        a = data[symbols[k]]
        node.push_bar(
            symbols[k], ts, a["open"][i], a["high"][i], a["low"][i], a["close"][i], a["volume"][i]
        )


def push_ops(node, ops):
    for op in ops:
        if op[0] == "tick":
            node.push_tick(op[1], op[2], *op[3:])
        elif op[0] == "depth":
            node.push_depth(op[1], op[2], op[3], op[4])
        else:
            node.push_bar(op[1], op[2], *op[3:])


# -- one strategy class, three ways to run it ------------------------------------


class Trader(Strategy):
    """Typed orders sized in units, on a rhythm per symbol."""

    def __init__(self):
        super().__init__()
        self.n = {}

    def on_bar(self, ctx, bar):
        n = self.n.get(ctx.symbol, 0)
        self.n[ctx.symbol] = n + 1
        c = bar.close
        r = n % 20
        if r == 2:
            self.submit_order(orders.Market(side="buy", units=10.0))
        elif r == 5:
            self.submit_order(orders.Limit(side="sell", price=c * 1.002, units=10.0))
        elif r == 9:
            self.submit_order(orders.StopMarket(side="sell", trigger=c * 0.99, units=10.0, reduce_only=True))
        elif r == 15:
            self.cancel_all_orders()
        elif r == 17:
            self.submit_order(orders.Market(side="sell", units=10.0, reduce_only=True))


def trades_of(result):
    return [
        (t.symbol, t.entry_price, t.exit_price, t.size, t.fees, t.entry_time, t.exit_time)
        for t in result.result.trades()
    ]


def test_one_strategy_runs_in_a_batch_and_in_simulated_and_paper_live_nodes():
    symbols = ["AAA", "BBB", "CCC"]
    data = make_bars(symbols)

    def config():
        return BacktestConfig(initial_capital=500_000.0, fees=0.0005, slippage=0.0003)

    batch = run_strategies([Trader()], data, config(), bar_stamp="close")[0]

    def live(execution):
        node = LiveNode([Trader()], symbols, config(), bar_stamp="close", execution=execution)
        push_bars(node, data, symbols)
        return node.finish()[0]

    simulated = live(SimulatedExecution())
    default = live(None)
    paper = live(PaperExecution())
    assert len(trades_of(batch)) > 30, "the strategy never traded"
    for other in (simulated, default, paper):
        assert trades_of(other) == trades_of(batch)
        assert list(other.result.equity_curve()) == list(batch.result.equity_curve())


# -- paper equals the simulator on the frozen streams ----------------------------

U = 25.0


class Program(Strategy):
    """A table of (symbol index, print ordinal, action) of unit-sized orders."""

    def __init__(self, program):
        super().__init__()
        self.program = program
        self.count = {}
        self.ids = []

    def on_trade(self, ctx, tick):
        n = self.count.get(ctx.symbol, 0)
        self.count[ctx.symbol] = n + 1
        si = ctx.symbols.index(ctx.symbol)
        for s, ordinal, action in self.program:
            if s == si and ordinal == n:
                ACTIONS[action](self, ctx, tick.price, tick.timestamp)

    def put(self, order):
        self.ids.append(self.submit_order(order))


ACTIONS = {
    "market_buy": lambda s, c, r, t: s.put(orders.Market(side="buy", units=U)),
    "market_sell": lambda s, c, r, t: s.put(orders.Market(side="sell", units=U)),
    "limit_buy": lambda s, c, r, t: s.put(orders.Limit(side="buy", price=r * 0.9996, units=U)),
    "limit_far": lambda s, c, r, t: s.put(orders.Limit(side="buy", price=r * 0.97, units=U)),
    "stop_buy": lambda s, c, r, t: s.put(orders.StopMarket(side="buy", trigger=r * 1.0004, units=U)),
    "stop_limit_buy": lambda s, c, r, t: s.put(
        orders.StopLimit(side="buy", trigger=r * 1.0003, price=r * 1.0007, units=U)
    ),
    "mit_buy": lambda s, c, r, t: s.put(orders.MarketIfTouched(side="buy", trigger=r * 0.9997, units=U)),
    "lit_buy": lambda s, c, r, t: s.put(
        orders.LimitIfTouched(side="buy", trigger=r * 0.9997, price=r * 0.9995, units=U)
    ),
    "mtl_buy": lambda s, c, r, t: s.put(orders.MarketToLimit(side="buy", units=U)),
    "ioc_limit": lambda s, c, r, t: s.put(orders.Limit(side="buy", price=r * 0.9999, units=U, tif="ioc")),
    "gtd_limit": lambda s, c, r, t: s.put(
        orders.Limit(side="buy", price=r * 0.995, units=U, tif="gtd", expire_ns=t + 90 * NS)
    ),
    "day_limit": lambda s, c, r, t: s.put(orders.Limit(side="buy", price=r * 0.996, units=U, tif="day")),
    "limit_sell_reduce": lambda s, c, r, t: s.put(
        orders.Limit(side="sell", price=r * 1.0005, units=U, reduce_only=True)
    ),
    "trail": lambda s, c, r, t: s.put(
        orders.TrailingStopMarket(side="sell", offset=r * 0.0006, units=U, reduce_only=True)
    ),
    "bracket": lambda s, c, r, t: (
        s.submit_bracket(orders.Market(side="buy", units=U), stop_trigger=r * 0.9992, target_price=r * 1.0008)
        if c.is_flat
        else None
    ),
    "oco_exit": lambda s, c, r, t: (
        None
        if c.is_flat
        else s.link_oco(
            s.submit_order(orders.Limit(side="sell", price=r * 1.0006, units=U, reduce_only=True)),
            s.submit_order(orders.StopMarket(side="sell", trigger=r * 0.9994, units=U, reduce_only=True)),
        )
    ),
    "cancel_last": lambda s, c, r, t: s.cancel_order(s.ids.pop()) if s.ids else None,
    "cancel_all": lambda s, c, r, t: s.cancel_all_orders(),
    "modify_last": lambda s, c, r, t: (
        s.modify_order(s.ids[-1], limit_price=r * 0.9998) if s.ids else None
    ),
}

#: Every kind of order, on every symbol, entered and exited with orders.
PROGRAM = [
    (0, 1, "limit_buy"), (0, 3, "modify_last"), (0, 8, "market_sell"), (0, 10, "stop_buy"),
    (0, 14, "stop_limit_buy"), (0, 18, "mit_buy"), (0, 22, "lit_buy"), (0, 27, "cancel_all"),
    (0, 30, "market_buy"), (0, 40, "limit_sell_reduce"),
    (1, 2, "limit_far"), (1, 4, "cancel_last"), (1, 6, "ioc_limit"), (1, 9, "gtd_limit"),
    (1, 12, "day_limit"), (1, 15, "market_buy"), (1, 30, "market_sell"), (1, 40, "mtl_buy"),
    (1, 50, "limit_sell_reduce"),
    (2, 3, "market_buy"), (2, 6, "limit_sell_reduce"), (2, 20, "bracket"), (2, 35, "market_buy"),
    (2, 36, "oco_exit"), (2, 45, "trail"),
]


def run_stream(execution, config_spec, program=PROGRAM, config_kwargs=None, **node_kwargs):
    config = rf.build_config(config_spec)
    strategy = Program(program)
    node = LiveNode([strategy], ALL, config, execution=execution, **node_kwargs)
    push_ops(node, rf.thaw_ops(DATASETS["stream_ops"]))
    return node.finish()[0]


enc_zero = rf.enc(0.0)


def digest(result):
    """The result's digest, less what only the in-process path can know: an
    order sent to a venue has its size resolved to units, and the venue, not
    the book, holds its one-cancels-other link."""
    d = rf.digest_result(result)
    orders_ = []
    for o in d["orders"]:
        requested = o["requested_qty"]
        o = {k: v for k, v in o.items() if k not in ("requested_qty", "oco_group")}
        # An exit larger than the position is over once the position is
        # closed; a venue says the surplus was cancelled.
        if o["status"] == "filled" and requested is not None:
            if float.fromhex(o["filled_qty"]) < float.fromhex(requested):
                o["status"] = "canceled"
        # A venue does not reject an order it has partly filled; it stops
        # working the rest.
        if o["status"] == "rejected" and o["filled_qty"] != enc_zero:
            o["status"], o["reject_reason"] = "canceled", None
            # ...so it is not a rejected entry either.
            d["rejected_entries"] -= 1
            for p in d["per_instrument"]:
                if p["symbol"] == o["symbol"]:
                    p["rejected_entries"] -= 1
        orders_.append(o)
    d["orders"] = orders_
    return d


STREAM_CONFIGS = [
    "st_push_basic",
    "st_queue_depth_latency",
    "st_margin_hedging",
    "st_drawdown_halt",
]


@pytest.mark.parametrize("name", STREAM_CONFIGS)
def test_paper_equals_the_simulator_fill_for_fill_on_the_frozen_streams(name):
    spec = SCENARIOS[name]["config"]
    simulated = run_stream(SimulatedExecution(), spec)
    paper = run_stream(PaperExecution(), spec)
    a, b = digest(simulated), digest(paper)
    # Every figure a result reports: curves, trades (prices, sizes, times,
    # fees), metrics, rejections, per-instrument totals, and each order.
    assert a == b
    assert len(a["trades"]) >= 2, "the program never traded"
    assert sum(o["fill_slices"] or 0 for o in a["orders"]) >= 4


def test_paper_equals_the_simulator_with_fees_slippage_and_latency():
    spec = {
        "kwargs": {
            "initial_capital": 1_000_000.0,
            "fees": 0.0007,
            "slippage": 0.0004,
            "order_latency_ns": 15 * NS,
            "partial_fills": True,
        },
        "calls": [],
    }
    simulated = run_stream(None, spec)
    paper = run_stream(PaperExecution(), spec)
    a, b = digest(simulated), digest(paper)
    assert a == b
    fees = [t.fees for t in simulated.result.trades()]
    assert fees and all(fee > 0 for fee in fees)


def test_closing_a_position_goes_to_the_venue_as_an_order_for_its_size():
    program = [(1, 2, "market_buy"), (1, 9, "close")]
    ACTIONS["close"] = lambda s, c, r, t: s.close_position(symbol=c.symbol)
    spec = SCENARIOS["st_push_basic"]["config"]
    try:
        simulated = run_stream(None, spec, program)
        paper = run_stream(PaperExecution(), spec, program)
    finally:
        del ACTIONS["close"]
    sim, pap = simulated.result.trades(), paper.result.trades()
    assert len(sim) == len(pap) == 1
    # The same close, at the same price and time. The simulator's is a
    # position-level exit; the venue's is a reduce-only market order.
    for field in ("symbol", "entry_price", "exit_price", "size", "fees", "entry_time", "exit_time"):
        assert getattr(sim[0], field) == getattr(pap[0], field), field
    assert [o.client_id for o in paper.result.orders()][-1].startswith("close-")


def test_paper_follows_positions_the_node_was_seeded_with():
    seeds = {0: {S1: {"quantity": 40.0, "avg_price": 100.0, "timestamp_ns": 5}}}
    program = [(1, 3, "market_sell"), (1, 12, "market_buy")]
    spec = SCENARIOS["st_push_basic"]["config"]
    a = run_stream(None, spec, program, initial_positions=seeds)
    b = run_stream(PaperExecution(), spec, program, initial_positions=seeds)
    assert digest(a) == digest(b)


# -- a venue of your own ---------------------------------------------------------


class Venue:
    """Records what it is sent. It answers nothing: the test does that."""

    def __init__(self, refuse=None, fail=None):
        self.requests, self.cancels, self.modifies, self.links = [], [], [], []
        self.refuse, self.fail = refuse or {}, fail

    def submit(self, request):
        if self.fail:
            raise self.fail
        if request.client_order_id in self.refuse:
            raise OrderRefused(self.refuse[request.client_order_id])
        self.requests.append(request)

    def cancel(self, client_order_id):
        self.cancels.append(client_order_id)

    def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
        self.modifies.append((client_order_id, quantity, limit_price, trigger_price))


class Hooked(Strategy):
    """Submits what it is told to, and records every order event."""

    def __init__(self, script=None):
        super().__init__()
        self.script = script or (lambda s, ctx, n: None)
        self.n = 0
        self.seen = []
        self.times = []

    def on_bar(self, ctx, bar):
        self.n += 1
        self.times.append((ctx.event_ns, ctx.arrival_ns, ctx.processing_ns))
        self.script(self, ctx, self.n)

    def on_order_event(self, ctx, event):
        self.seen.append((event.kind, event.client_order_id, event.price, event.size, ctx.symbol))
        self.times.append((ctx.event_ns, ctx.arrival_ns, ctx.processing_ns))


def node_with(venue, strategies=None, symbols=("A", "B"), **kw):
    strategies = strategies or [Hooked()]
    config = kw.pop("config", None) or BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
    return LiveNode(
        strategies, list(symbols), config, bar_stamp="close", execution=venue, **kw
    )


def bar(node, symbol, i, price=100.0):
    node.push_bar(symbol, (i + 1) * MIN, price, price + 1, price - 1, price + 0.5, 1000.0)


def test_the_protocol_is_what_a_plain_object_with_three_methods_satisfies():
    assert isinstance(Venue(), ExecutionAdapter)
    assert not isinstance(object(), ExecutionAdapter)
    with pytest.raises(InvalidParameter) as e:
        LiveNode([Hooked()], ["A"], execution=object(), bar_stamp="close")
    assert e.value.code == "invalid_execution"


def test_a_request_carries_the_order_the_venue_needs():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Limit(side="buy", price=99.0, units=7.0, tif="gtd", expire_ns=10**18, post_only=True))
        if n == 2:
            s.submit_order(orders.StopLimit(side="sell", trigger=95.0, price=94.0, units=3.0, reduce_only=True), client_id="mine")

    node = node_with(venue, [Hooked(script)], venue_account="ACC-1")
    bar(node, "A", 0)
    bar(node, "A", 1)
    first, second = venue.requests
    assert isinstance(first, OrderRequest)
    assert (first.client_order_id, first.symbol, first.side, first.quantity) == ("s0-0", "A", "buy", 7.0)
    assert (first.order_type, first.limit_price, first.trigger_price) == ("limit", 99.0, None)
    assert (first.time_in_force, first.expire_ns, first.post_only, first.reduce_only) == ("gtd", 10**18, True, False)
    assert (first.strategy_id, first.account_id, first.submitted_ns) == (0, "ACC-1", MIN)
    assert first.tags == {"client_id": "O-0"}
    assert (second.client_order_id, second.order_type) == ("s0-1", "stop_limit")
    assert (second.limit_price, second.trigger_price, second.reduce_only) == (94.0, 95.0, True)
    assert second.tags == {"client_id": "mine"}
    assert first == first and first != second
    with pytest.raises(AttributeError):
        first.quantity = 1.0  # frozen


def test_client_order_ids_depend_only_on_the_strategy_and_its_count():
    def script(s, ctx, n):
        s.submit_order(orders.Limit(side="buy", price=50.0 + n, units=1.0), symbol=ctx.symbol)

    def run():
        venue = Venue()
        node = node_with(venue, [Hooked(script), Hooked(script)])
        for i in range(3):
            bar(node, "A", i)
            bar(node, "B", i)
        return [(r.client_order_id, r.strategy_id, r.symbol) for r in venue.requests]

    first = run()
    assert first == run(), "a replay sends the same ids"
    assert sorted({i for i, *_ in first}) == [f"s{k}-{n}" for k in (0, 1) for n in range(6)]
    # Strategy 0's seq counts its own orders only, in the order it sent them.
    assert [i for i, k, _ in first if k == 0] == [f"s0-{n}" for n in range(6)]


def test_a_duplicated_command_never_sends_a_second_order():
    venue = Venue()

    def script(s, ctx, n):
        if n in (1, 2):  # the same command, in this event and the next
            s.submit_order(orders.Limit(side="buy", price=90.0, units=5.0), client_id="dup")
            s.submit_order(orders.Limit(side="buy", price=90.0, units=5.0), client_id="dup")
        if n == 3:
            s.cancel_order("dup")
            s.cancel_order("dup")
        if n == 4:
            s.cancel_order("dup")

    node = node_with(venue, [Hooked(script)], symbols=("A",))
    for i in range(4):
        bar(node, "A", i)
    assert [r.client_order_id for r in venue.requests] == ["s0-0"]
    assert venue.cancels == ["s0-0"], "the cancel is asked once"


def test_reusing_an_order_id_for_a_different_order_is_refused():
    venue = Venue()

    def script(s, ctx, n):
        s.submit_order(orders.Limit(side="buy", price=90.0 + n, units=5.0), client_id="same")

    node = node_with(venue, [Hooked(script)], symbols=("A",))
    bar(node, "A", 0)
    with pytest.raises(InvalidOrder) as e:
        bar(node, "A", 1)
    assert e.value.code == "duplicate_client_order_id"
    assert len(venue.requests) == 1


def test_the_venue_decides_what_happens_to_an_order_only_by_reports():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Limit(side="buy", price=99.0, units=10.0))

    strategy = Hooked(script)
    node = node_with(venue, [strategy], symbols=("A",))
    bar(node, "A", 0)
    (request,) = venue.requests
    cid = request.client_order_id
    # The order is working at the venue and the book has no position, however
    # the market moves.
    bar(node, "A", 1, price=50.0)
    assert node.positions(0, "A") == []
    assert strategy.seen == []

    assert node.push_report(cid, ExecutionReport.accepted("V1", MIN), arrival_ns=MIN + 5) == "applied"
    assert [k for k, *_ in strategy.seen] == ["order_accepted"]
    assert node.push_report(cid, ExecutionReport.fill("E1", 4.0, 99.0, 2 * MIN, fee=1.5), arrival_ns=2 * MIN) == "applied"
    (pos,) = node.positions(0, "A")
    assert (pos.size, pos.entry_price) == (4.0, 99.0)
    assert node.push_report(cid, ExecutionReport.fill("E1", 4.0, 99.0, 2 * MIN, fee=1.5), arrival_ns=2 * MIN) == "duplicate"
    assert node.positions(0, "A")[0].size == 4.0
    assert node.push_report(cid, ExecutionReport.fill("E2", 6.0, 98.5, 3 * MIN), arrival_ns=3 * MIN) == "applied"
    assert sum(p.size for p in node.positions(0, "A")) == 10.0
    kinds = [k for k, *_ in strategy.seen]
    assert kinds == ["order_accepted", "order_filled", "entered", "order_filled", "entered"] or kinds.count("order_filled") == 2
    result = node.finish()[0]
    (record,) = result.result.orders()
    assert (record.status, record.filled_qty) == ("filled", 10.0)
    assert record.avg_fill_price == pytest.approx((4 * 99.0 + 6 * 98.5) / 10)


def test_a_report_can_reach_the_book_only_through_the_node():
    venue = Venue()
    node = node_with(venue, [Hooked(lambda s, c, n: s.submit_order(orders.Market(side="buy", units=1.0)) if n == 1 else None)], symbols=("A",))
    bar(node, "A", 0)
    cid = venue.requests[0].client_order_id
    with pytest.raises(InvalidParameter) as e:
        node.push_report("s9-9", ExecutionReport.accepted("V", 1), arrival_ns=1)
    assert e.value.code == "unknown_client_order_id"
    with pytest.raises(InvalidParameter):
        node.push_report(cid, "accepted", arrival_ns=1)
    with pytest.raises(UnsupportedOperation) as e:
        node.push_report(cid, ExecutionReport.correction("E", 1.0, 1.0, 1), arrival_ns=1)
    assert e.value.code == "unsupported_in_venue_run"
    with pytest.raises(InvalidData) as e:  # fills beyond the order
        node.push_report(cid, ExecutionReport.fill("E", 5.0, 1.0, 1), arrival_ns=1)
    assert e.value.code == "invalid_execution_report"
    plain = LiveNode([Hooked()], ["A"], bar_stamp="close")
    with pytest.raises(InvalidParameter) as e:
        plain.push_report("s0-0", ExecutionReport.accepted("V", 1), arrival_ns=1)
    assert e.value.code == "invalid_execution"


def test_a_venue_that_refuses_an_order_rejects_it_and_one_that_fails_does_not():
    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Limit(side="buy", price=99.0, units=1.0), client_id="a")

    refusing = Hooked(script)
    node = node_with(Venue(refuse={"s0-0": "no margin"}), [refusing], symbols=("A",))
    bar(node, "A", 0)
    bar(node, "A", 1)  # the rejection is heard at the next event
    assert any(k == "order_rejected" for k, *_ in refusing.seen)
    (record,) = node.finish()[0].result.orders()
    assert (record.status, record.reject_reason) == ("rejected", "no margin")

    # Any other exception means "may or may not have been sent": it reaches the
    # caller unchanged and the order waits for the venue to say.
    failing = Venue(fail=ConnectionError("link down"))
    node = node_with(failing, [Hooked(script)], symbols=("A",))
    with pytest.raises(ConnectionError, match="link down"):
        bar(node, "A", 0)


def test_changing_and_cancelling_go_to_the_venue_and_wait_for_its_answer():
    venue = Venue()
    ids = []

    def script(s, ctx, n):
        if n == 1:
            ids.append(s.submit_order(orders.Limit(side="buy", price=99.0, units=10.0)))
        if n == 3:
            s.modify_order(ids[0], limit_price=98.0, units=8.0)
        if n == 5:
            s.cancel_order(ids[0])

    strategy = Hooked(script)
    node = node_with(venue, [strategy], symbols=("A",))
    bar(node, "A", 0)
    cid = venue.requests[0].client_order_id
    node.push_report(cid, ExecutionReport.accepted("V1", MIN), arrival_ns=MIN)
    bar(node, "A", 1)
    bar(node, "A", 2)
    assert venue.modifies == [(cid, 8.0, 98.0, None)]
    (working,) = node.ctx(0).open_orders("A")
    assert working.status == "pending_replace"
    node.push_report(cid, ExecutionReport.replace_ack(4 * MIN, new_qty=8.0, new_limit=98.0), arrival_ns=4 * MIN)
    (working,) = node.ctx(0).open_orders("A")
    assert (working.status, working.units) == ("accepted", 8.0)
    bar(node, "A", 3)
    bar(node, "A", 4)
    assert venue.cancels == [cid]
    assert node.ctx(0).open_orders("A")[0].status == "pending_cancel"
    # A fill that beat the cancel is real.
    node.push_report(cid, ExecutionReport.fill("E1", 3.0, 98.0, 5 * MIN), arrival_ns=5 * MIN)
    assert node.push_report(cid, ExecutionReport.cancel_ack(5 * MIN), arrival_ns=5 * MIN) == "applied"
    assert node.ctx(0).open_orders("A") == []
    (record,) = node.finish()[0].result.orders()
    assert (record.status, record.filled_qty) == ("canceled", 3.0)


def test_a_venue_without_one_cancels_other_has_the_link_kept_by_the_engine():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            a = s.submit_order(orders.Limit(side="sell", price=110.0, units=1.0))
            b = s.submit_order(orders.Limit(side="sell", price=111.0, units=1.0))
            s.link_oco(a, b)

    node = node_with(venue, [Hooked(script)], symbols=("A",))
    bar(node, "A", 0)
    first, second = (r.client_order_id for r in venue.requests)
    node.push_report(first, ExecutionReport.fill("E1", 1.0, 110.0, MIN), arrival_ns=MIN)
    assert venue.cancels == [second]


@pytest.mark.parametrize("venue", [Venue(), PaperExecution()])
def test_what_a_venue_cannot_follow_is_refused_up_front(venue):
    from raptorbt import AccountSpec

    with pytest.raises(InvalidParameter) as e:
        LiveNode(
            [(Hooked(), ["A"], {"account": "x"})],
            ["A"],
            BacktestConfig(initial_capital=1e6),
            accounts={"x": AccountSpec(initial_capital=1e6)},
            bar_stamp="close",
            execution=venue,
        )
    assert e.value.code == "invalid_execution"
    config = BacktestConfig(initial_capital=1e6, liquidate_on_margin_call=True)
    with pytest.raises(InvalidParameter):
        node_with(venue, config=config)
    with pytest.raises(InvalidParameter):
        node_with(venue, protective="sideways")


# -- duplicated, late and reordered reports --------------------------------------


class Buyer(Strategy):
    """Sends a batch of buys, spread over two symbols, at fixed bars."""

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        for step, symbol, kind, units, price in self.plan:
            if step == self.n and symbol == ctx.symbol:
                order = (
                    orders.Market(side="buy", units=units)
                    if kind == "market"
                    else orders.Limit(side="buy", price=price, units=units)
                )
                self.submit_order(order)


def venue_story(request, rng):
    """What a venue would say about one order, in a valid order."""
    ts = request.submitted_ns
    story = [ExecutionReport.accepted("V-" + request.client_order_id, ts)]
    qty, price = request.quantity, request.limit_price or 100.0
    ends = rng.choice(["filled", "cancelled", "open"])
    cut = {"filled": qty, "cancelled": qty * rng.uniform(0.2, 0.8), "open": qty * rng.uniform(0.2, 0.8)}[ends]
    cuts = sorted(rng.uniform(0, cut, size=int(rng.integers(0, 3))).tolist()) + [cut]
    done = 0.0
    for k, c in enumerate(cuts):
        q = round(c - done, 6)
        if q <= 0:
            continue
        done += q
        story.append(
            ExecutionReport.fill(
                f"{request.client_order_id}.{k}", q, round(price + rng.uniform(-0.5, 0.5), 4), ts + (k + 1) * 1000,
                fee=round(q * 0.01, 6),
            )
        )
    if ends == "cancelled":
        story.append(ExecutionReport.cancel_ack(ts + 10_000))
    return story


def state_of(node):
    """Orders, positions and cash, in a form two runs can be compared in."""
    result = node.finish()[0]
    got = {}
    for o in result.result.orders():
        got[o.client_id] = (o.status, round(o.filled_qty, 9), None if o.avg_fill_price is None else round(o.avg_fill_price, 9))
    held = {}
    for t in result.result.trades():
        held[t.symbol] = held.get(t.symbol, 0.0) + t.size
    return got, held, result.result.equity_curve()[-1]


@pytest.mark.parametrize("seed", range(40))
def test_duplicated_late_and_reordered_reports_leave_the_same_book(seed):
    rng = np.random.default_rng(seed)
    plan = [(1, "A", "market", 10.0, 100.0)] + [
        (int(rng.integers(1, 8)), str(rng.choice(["A", "B"])), str(rng.choice(["market", "limit"])), float(rng.integers(5, 60)), 100.0 + float(rng.integers(-3, 3)))
        for _ in range(int(rng.integers(3, 9)))
    ]

    def build():
        venue = Venue()
        node = LiveNode(
            [Buyer(plan)], ["A", "B"], BacktestConfig(initial_capital=10_000_000.0, fees=0.0003),
            bar_stamp="close", execution=venue, oms_type="hedging",
        )
        for i in range(10):
            bar(node, "A", i)
            bar(node, "B", i)
        return node, venue

    ref, venue = build()
    chaos, chaos_venue = build()
    assert [r.client_order_id for r in venue.requests] == [r.client_order_id for r in chaos_venue.requests]
    stories = {r.client_order_id: venue_story(r, rng) for r in venue.requests}

    # In order: each order's reports one after the other.
    for cid, story in stories.items():
        for k, report in enumerate(story):
            ref.push_report(cid, report, arrival_ns=100 * MIN + k)

    # A hostile venue: every report may come twice or thrice, any report may
    # come after reports that were sent later, and orders interleave. Only
    # the venue's own causality is kept: an order's end after its fills is
    # not required, because the engine must not rely on it.
    pending = [(cid, r) for cid, story in stories.items() for r in story]
    delivery = []
    for item in pending:
        delivery.extend([item] * int(rng.choice([1, 1, 2, 3])))
    order = rng.permutation(len(delivery))
    for k, j in enumerate(order):
        cid, report = delivery[j]
        chaos.push_report(cid, report, arrival_ns=100 * MIN + k)

    got_ref, held_ref, equity_ref = state_of(ref)
    got_chaos, held_chaos, equity_chaos = state_of(chaos)
    assert got_chaos == got_ref
    assert held_chaos.keys() == held_ref.keys()
    for symbol in held_ref:
        assert held_chaos[symbol] == pytest.approx(held_ref[symbol], rel=1e-9)
    # Money books in minor units: the one rounding step may fall either side.
    assert equity_chaos == pytest.approx(equity_ref, rel=1e-9, abs=0.011)
    assert stories and all(len(v) >= 2 for v in stories.values())


# -- the three times of an event -------------------------------------------------


def test_every_event_has_an_exchange_arrival_and_processing_time():
    class Reader(Strategy):
        def __init__(self):
            super().__init__()
            self.seen = []

        def on_bar(self, ctx, bar):
            self.seen.append((ctx.event_ns, ctx.arrival_ns, ctx.processing_ns))

    # A batch run: all three are the event's own time.
    data = make_bars(["A"], n=5)
    batch = Reader()
    run_strategies([batch], data, BacktestConfig(initial_capital=1e6), bar_stamp="close")
    assert batch.seen == [(int(t), int(t), int(t)) for t in data["A"]["timestamps"]]

    live = Reader()
    node = LiveNode([live], ["A"], BacktestConfig(initial_capital=1e6), bar_stamp="close", processing_latency_ns=700)
    bar(node, "A", 0)
    node.push_bar("A", 2 * MIN, 100.0, 101.0, 99.0, 100.5, 1000.0, arrival_ns=2 * MIN + 40)
    assert live.seen == [(MIN, MIN, MIN + 700), (2 * MIN, 2 * MIN + 40, 2 * MIN + 740)]


def test_a_report_carries_the_venues_time_and_its_own_arrival():
    venue = Venue()
    strategy = Hooked(lambda s, c, n: s.submit_order(orders.Market(side="buy", units=2.0)) if n == 1 else None)
    node = node_with(venue, [strategy], symbols=("A",), processing_latency_ns=25)
    bar(node, "A", 0)
    strategy.times.clear()
    cid = venue.requests[0].client_order_id
    node.push_report(cid, ExecutionReport.accepted("V", 111), arrival_ns=500)
    assert strategy.times == [(111, 500, 525)]
    # The time of a report is not the node's market time: it does not move it.
    node.push_bar("A", 2 * MIN, 100.0, 101.0, 99.0, 100.5, 1000.0)


def test_pushing_to_a_node_from_inside_one_of_its_adapters_is_refused():
    class Reentrant(Venue):
        node = None

        def submit(self, request):
            super().submit(request)
            self.node.push_report(request.client_order_id, ExecutionReport.accepted("V", 1), arrival_ns=1)

    venue = Reentrant()
    node = node_with(venue, [Hooked(lambda s, c, n: s.submit_order(orders.Market(side="buy", units=1.0)) if n == 1 else None)], symbols=("A",))
    venue.node = node
    with pytest.raises(StateError):
        bar(node, "A", 0)


# -- the book's limits ----------------------------------------------------------


def test_a_venue_that_does_not_know_the_books_limits_is_not_sent_what_they_forbid():
    venue = Venue()

    def script(s, ctx, n):
        if n == 1:
            s.submit_order(orders.Market(side="buy", units=5.0), client_id="first")
        if n == 3:
            s.submit_order(orders.Market(side="buy", units=5.0), client_id="second")  # a second position
        if n == 5:
            s.submit_order(orders.Market(side="sell", units=5.0, reduce_only=True), client_id="exit")

    strategy = Hooked(script)
    config = BacktestConfig(initial_capital=1_000_000.0, fees=0.0, max_positions=1)
    node = node_with(venue, [strategy], symbols=("A",), config=config, oms_type="hedging")
    bar(node, "A", 0)
    cid = venue.requests[0].client_order_id
    node.push_report(cid, ExecutionReport.fill("E1", 5.0, 100.0, MIN), arrival_ns=MIN)
    bar(node, "A", 1)
    bar(node, "A", 2)  # max_positions is reached: refused here, never sent
    assert [r.tags["client_id"] for r in venue.requests] == ["first"]
    bar(node, "A", 3)
    bar(node, "A", 4)  # an exit is always allowed
    assert [r.tags["client_id"] for r in venue.requests] == ["first", "exit"]
    rejected = [o for o in node.finish()[0].result.orders() if o.status == "rejected"]
    assert [(o.client_id, o.reject_reason) for o in rejected] == [("second", "max_positions")]


def test_a_real_venue_is_never_sent_an_order_made_on_warmup_history():
    warmup = {"A": {"timestamps": np.arange(1, 6, dtype=np.int64) * MIN, "open": np.full(5, 100.0), "high": np.full(5, 101.0), "low": np.full(5, 99.0), "close": np.full(5, 100.5), "volume": np.full(5, 1000.0)}}

    def script(s, ctx, n):
        s.submit_order(orders.Limit(side="buy", price=90.0, units=1.0))

    venue = Venue()
    with pytest.raises(UnsupportedOperation, match="warmup"):
        LiveNode([Hooked(script)], ["A"], BacktestConfig(initial_capital=1e6), bar_stamp="close", warmup_bars=warmup, execution=venue)
    assert venue.requests == []
    # A simulated venue has no one to hurt: the history trades there.
    node = LiveNode([Hooked(script)], ["A"], BacktestConfig(initial_capital=1e6), bar_stamp="close", warmup_bars=warmup, execution=PaperExecution())
    assert len(node.finish()[0].result.orders()) == 5


# -- everything a backtest accepts runs the same on a venue ---------------------------


def strip_internal(d):
    """The digest less what only the in-process path can know. An order sent to
    a venue has its size resolved to units, the venue holds its
    one-cancels-other link, and the engine's own orders (intents, closes,
    protection) have ids of their own; a trade closed by an order has the exit
    reason `Order`."""
    z = rf.enc(0.0)
    orders_ = []
    for o in d["orders"]:
        if o["client_id"].split("-")[0] in ("close", "enter", "prot"):
            continue
        requested = o["requested_qty"]
        o = {k: v for k, v in o.items() if k not in ("requested_qty", "oco_group", "id")}
        if o["status"] == "rejected" and o["filled_qty"] != z:
            o["status"], o["reject_reason"] = "canceled", None
        if o["status"] == "filled" and requested is not None:
            if float.fromhex(o["filled_qty"]) < float.fromhex(requested):
                o["status"] = "canceled"
        orders_.append(o)
    d["orders"] = orders_
    d["trades"] = [{k: v for k, v in t.items() if k != "exit_reason"} for t in d["trades"]]
    d.pop("rejected_entries", None)
    d["per_instrument"] = [{k: v for k, v in p.items() if k != "rejected_entries"} for p in d["per_instrument"]]
    return d


def stream_digest(name, execution):
    sc = SCENARIOS[name]
    rec = rf.Recorder(script=rf._script_for(sc, DATASETS))
    rec.tick_surface = True
    real = rf.LiveNode
    rf.LiveNode = lambda *a, **k: real(*a, execution=execution, **k)
    try:
        result, _ = rf._run_stream(sc, DATASETS, rec, rf.build_config(sc["config"]), "runtime")
    finally:
        rf.LiveNode = real
    return strip_internal(rf.digest_result(result))


#: Scenarios that differ, and why. Each is a strict xfail so the list stays true.
DIFFERS = {
    "st_refused_rows_timers": "an entry intent held for the next print fills after a typed market "
    "order made earlier in the same step; in-process it fills first",
    "st_margin_hedging": "under hedging every order opens, so no order can close a position",
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=DIFFERS[n])) if n in DIFFERS else n
        for n, sc in SCENARIOS.items()
        if sc["runner"] == "stream"
    ],
)
def test_the_stream_corpus_equals_in_process_on_a_paper_node(name):
    """enter(), close_position(), capital-fraction sizes, attached stops and
    targets, brackets, adopted positions, warmup and halts: the strategies of
    the frozen stream scenarios give the same result when their orders go to a
    simulated venue."""
    assert stream_digest(name, PaperExecution()) == stream_digest(name, None)


def portfolio_result(name, execution):
    sc = SCENARIOS[name]
    rec = rf.Recorder(script=rf._script_for(sc, DATASETS))
    config = rf.build_config(sc["config"])
    kw = rf._run_kwargs(sc)
    bars = rf._bars_for(sc, DATASETS)
    if execution == "batch":
        return run_strategies(
            [rec], config=config, data=bars, instrument_configs=rf._instrument_configs(sc), **kw
        )[0]
    symbols = list(bars)
    node = LiveNode(
        [rec], symbols, config, instrument_configs=rf._instrument_configs(sc), execution=execution, **kw
    )
    stamp = kw.get("bar_stamp", "open")
    shift = kw.get("bar_ns", 0) if stamp == "open" else 0
    if isinstance(shift, dict):
        shift = shift[symbols[0]]
    rows = sorted(
        (int(bars[s]["timestamps"][i]) + shift, k, i)
        for k, s in enumerate(symbols)
        for i in range(len(bars[s]["timestamps"]))
    )
    for _, k, i in rows:
        a = bars[symbols[k]]
        node.push_bar(symbols[k], int(a["timestamps"][i]), *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume")))
    return node.finish()[0]


PORTFOLIO_OK = [
    "pf_composites_indicators", "pf_fill_same_bar_open_lookahead", "pf_margin_netting_shorts",
    "pf_stops_targets_legacy", "pf_typed_exits", "pf_timers", "pf_twap", "pf_max_positions_rejections",
    "pf_drawdown_halt", "pf_margin_call_halt", "pf_order_latency", "pf_bar_stamp_close",
    "pf_bar_stamp_dict_tz", "pf_legacy_stamp_warning",
]
PORTFOLIO_DIFFERS = {
    "pf_fill_next_bar_open": "an entry intent is sized from the capital on the signal bar and fills at the next open; in-process it is sized there",
    "pf_margin_hedging": "under hedging every order opens, so no order can close a position",
    "pf_cash_hedging": "under hedging every order opens, so no order can close a position",
}


@pytest.mark.parametrize(
    "name",
    PORTFOLIO_OK
    + [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=r)) for n, r in PORTFOLIO_DIFFERS.items()],
)
def test_the_portfolio_corpus_equals_run_strategies_on_a_paper_node(name):
    """The strategies of the frozen portfolio scenarios (config stops and
    targets, brackets, OCO, trailing, Twap, timers, risk limits, halts,
    margin) give the result `run_strategies` gives when run on a paper node."""
    batch = strip_internal(rf.digest_result(portfolio_result(name, "batch")))
    paper = strip_internal(rf.digest_result(portfolio_result(name, PaperExecution())))
    assert paper == batch


def test_atr_stops_are_refused_on_a_live_node_with_or_without_a_venue():
    # A live node has no bar series to compute ATR from.
    for execution in (None, PaperExecution()):
        with pytest.raises(ValueError, match="ATR"):
            portfolio_result("pf_atr_stops", execution)


@pytest.mark.parametrize(
    "structure_name", ["iron_condor_time_exit", "straddle_stop", "condor_leg_failure"]
)
def test_option_structures_give_run_strategies_results_on_a_paper_node(structure_name):
    import test_structures as ts

    structure, attrs, config = {
        "iron_condor_time_exit": (ts.st.iron_condor(1, 2), {"exit": ts.st.Exit(close_at_ns=ts.T0 + 25 * ts.STEP)}, {}),
        "straddle_stop": (ts.SHORT_CALL, {"exit": ts.st.Exit(stop_loss=150.0)}, {}),
        "condor_leg_failure": (ts.st.iron_condor(1, 2), {}, {"max_order_notional": ts.notional_cut()}),
    }[structure_name]
    cls = ts.make(structure, **attrs)
    batch_strategy, paper_strategy = cls(), cls()
    data, inst = ts.universe(drift=0.4 if "stop_loss" in str(attrs) else 0.0)
    cfg = BacktestConfig(initial_capital=2_000_000.0, fees=0.0, **config)
    batch = run_strategies([batch_strategy], data, cfg, instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")[0]
    symbols = list(data)
    node = LiveNode([paper_strategy], symbols, cfg, instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close", execution=PaperExecution())
    for i in range(ts.N_BARS):
        for sym in symbols:
            b = data[sym]
            node.push_bar(sym, int(b["timestamps"][i]), float(b["open"][i]), float(b["high"][i]), float(b["low"][i]), float(b["close"][i]), float(b["volume"][i]))
    d1, d2 = ts.digest(batch), ts.digest(node.finish()[0])
    assert d1["orders"], "the structure placed nothing"
    assert d1 == d2
    assert batch_strategy.group.reason == paper_strategy.group.reason


# -- protection, the square-off, schedules and halts as orders ---------------------


def trending_bars(n=60, start=100.0, step=-0.5):
    close = start + step * np.arange(n)
    return {
        "timestamps": (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": close + 0.1, "high": close + 0.6, "low": close - 0.6, "close": close,
        "volume": np.full(n, 1000.0),
    }


class Enters(Strategy):
    def __init__(self, **kw):
        super().__init__()
        self.kw = kw
        self.n = 0

    def on_bar(self, ctx, bar):
        self.n += 1
        if self.n == 3:
            self.enter(size_frac=0.5, **self.kw)


def run_enters(execution, protective="venue", config=None, **kw):
    node = LiveNode(
        [Enters(**kw)], ["A"], config or BacktestConfig(initial_capital=1e6), bar_stamp="close",
        execution=execution, protective=protective,
    )
    bars = trending_bars()
    for i in range(len(bars["close"])):
        node.push_bar("A", int(bars["timestamps"][i]), bars["open"][i], bars["high"][i], bars["low"][i], bars["close"][i], 1000.0)
    return node, node.finish()[0]


def test_a_stop_on_an_entry_is_a_protective_order_the_venue_holds():
    venue = Venue()
    node, _ = None, None
    node = LiveNode([Enters(stop_price=95.0, target_price=120.0)], ["A"], BacktestConfig(initial_capital=1e6), bar_stamp="close", execution=venue)
    assert node.protective == "venue"
    bars = trending_bars()
    for i in range(4):
        node.push_bar("A", int(bars["timestamps"][i]), bars["open"][i], bars["high"][i], bars["low"][i], bars["close"][i], 1000.0)
    (entry,) = venue.requests
    assert entry.tags["client_id"].startswith("enter-") and entry.side == "buy"
    node.push_report(entry.client_order_id, ExecutionReport.fill("E1", entry.quantity, 99.0, 4 * MIN), arrival_ns=4 * MIN)
    stop, target = venue.requests[1:]
    assert (stop.order_type, stop.trigger_price, stop.side, stop.reduce_only) == ("stop_market", 95.0, "sell", True)
    assert (target.order_type, target.limit_price, target.reduce_only) == ("limit", 120.0, True)
    assert stop.tags["client_id"].startswith("prot-") and stop.quantity == entry.quantity
    # This venue has no link: the engine keeps it, and cancels the other leg.
    node.push_report(target.client_order_id, ExecutionReport.fill("E2", target.quantity, 120.0, 5 * MIN), arrival_ns=5 * MIN)
    assert venue.cancels == [stop.client_order_id]


def test_a_stop_the_venue_cannot_hold_is_watched_by_the_engine():
    class NoStops(Venue):
        def submit(self, request):
            if request.order_type == "stop_market":
                raise Unsupported("no stop orders")
            super().submit(request)

    venue = NoStops()
    node = LiveNode([Enters(stop_price=98.0)], ["A"], BacktestConfig(initial_capital=1e6), bar_stamp="close", execution=venue)
    bars = trending_bars()
    for i in range(3):
        node.push_bar("A", int(bars["timestamps"][i]), bars["open"][i], bars["high"][i], bars["low"][i], bars["close"][i], 1000.0)
    (entry,) = venue.requests
    node.push_report(entry.client_order_id, ExecutionReport.fill("E1", entry.quantity, 99.0, 3 * MIN), arrival_ns=3 * MIN)
    assert [r.order_type for r in venue.requests] == ["market"]
    for i in range(3, 8):  # falls through 98
        node.push_bar("A", int(bars["timestamps"][i]), bars["open"][i], bars["high"][i], bars["low"][i], bars["close"][i], 1000.0)
    close = venue.requests[1]
    assert (close.order_type, close.side, close.reduce_only, close.quantity) == ("market", "sell", True, entry.quantity)


def test_protective_local_watches_levels_and_sends_a_market_order():
    spec = {"kwargs": {"initial_capital": 1e6}, "calls": [["set_fixed_stop", [0.02]], ["set_fixed_target", [0.2]]]}
    config = rf.build_config(spec)
    node, result = run_enters(PaperExecution(), protective="local", config=config)
    assert node is not None
    sim_node, sim = run_enters(None, config=rf.build_config(spec))
    (trade,) = result.result.trades()
    (want,) = sim.result.trades()
    # Same position, closed on the same bar; the venue-side market order fills at
    # that bar's price, not at the stop level (the documented parity limit).
    assert (trade.entry_price, trade.size, trade.exit_idx) == (want.entry_price, want.size, want.exit_idx)
    assert trade.exit_price != want.exit_price
    protected, protected_sim = run_enters(PaperExecution(), config=rf.build_config(spec))[1], sim
    (t,) = protected.result.trades()
    assert (t.exit_idx, t.exit_price) == (want.exit_idx, want.exit_price), "venue-held protection fills at the stop"


def test_the_session_square_off_sends_the_closing_orders():
    import test_sessions as tss
    from raptorbt import InstrumentSpec, SessionSchedule

    data = tss.bars(tss.regular_day(tss.THU), tss.regular_day(tss.FRI))
    per_day = len(tss.bars(tss.regular_day(tss.THU))["timestamps"])

    class Enters4(Strategy):
        def on_bar(self, ctx, bar):
            if ctx.idx in (4, per_day + 4):
                self.submit_order(orders.Market(side="buy", units=10.0))

    spec = InstrumentSpec.equity("ASSET", session=SessionSchedule.nse_equity())
    runs = []
    for execution in (None, PaperExecution()):
        node = LiveNode(
            [Enters4()], ["ASSET"], tss._zero_fee_config(), bar_stamp="close",
            instruments={"ASSET": spec}, execution=execution,
        )
        for i in range(len(data["timestamps"])):
            node.push_bar("ASSET", int(data["timestamps"][i]), *(float(data[c][i]) for c in ("open", "high", "low", "close", "volume")))
        runs.append(node.finish()[0].result.trades())
    sim, paper = runs
    assert len(sim) == len(paper) == 2
    assert [t.exit_reason for t in sim] == ["Squareoff", "Squareoff"]
    for a, b in zip(sim, paper):  # closed at the same bar, the same price, by an order
        assert (a.entry_idx, a.exit_idx, a.entry_price, a.exit_price, a.size) == (b.entry_idx, b.exit_idx, b.entry_price, b.exit_price, b.size)


def test_a_halt_cancels_what_adds_exposure_refuses_entries_and_can_flatten():
    venue = Venue()
    strategy = Hooked(lambda s, c, n: (s.submit_order(orders.Limit(side="buy", price=90.0, units=5.0)) if n == 1 else None))
    node = node_with(venue, [strategy], symbols=("A",))
    bar(node, "A", 0)
    cid = venue.requests[0].client_order_id
    node.push_report(cid, ExecutionReport.accepted("V", MIN), arrival_ns=MIN)
    node.push_report(cid, ExecutionReport.fill("E", 2.0, 90.0, MIN), arrival_ns=MIN)
    node.halt(0, flatten=True)
    assert venue.cancels == [cid], "the rest of the order is cancelled"
    closing = venue.requests[-1]
    assert (closing.side, closing.reduce_only, closing.quantity) == ("sell", True, 2.0)
    assert node.is_halted(0)


def test_twap_sends_its_slices_as_market_orders_on_its_schedule():
    venue = Venue()
    strategy = Hooked(lambda s, c, n: s.submit_order(orders.Twap(side="buy", units=9.0, slices=3, every=2 * MIN)) if n == 1 else None)
    node = node_with(venue, [strategy], symbols=("A",))
    for i in range(8):
        bar(node, "A", i)
    sliced = [(r.order_type, r.side, r.quantity) for r in venue.requests]
    assert sliced == [("market", "buy", 3.0)] * 3
    assert [r.submitted_ns for r in venue.requests] == [MIN, 3 * MIN, 5 * MIN]


# -- pushed prices must be finite; corporate actions on a live node -------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_pushed_price_that_is_not_finite_is_refused_and_changes_nothing(bad):
    for execution in (None, PaperExecution(), Venue()):
        strategy = Hooked()
        node = node_with(execution, [strategy], symbols=("A",))
        bar(node, "A", 0)
        with pytest.raises(InvalidData) as e:
            node.push_bar("A", 2 * MIN, 100.0, bad, 99.0, 100.0, 10.0)
        assert e.value.code == "non_finite_price" and e.value.parameter == "high"
        with pytest.raises(InvalidData) as e:
            node.push_tick("A", 3 * MIN, bad, 99.0, 101.0)
        assert e.value.parameter == "ltp"
        with pytest.raises(InvalidData) as e:
            node.push_tick("A", 3 * MIN, 100.0, 99.0, bad)
        assert e.value.parameter == "ask"
        with pytest.raises(InvalidData) as e:
            node.push_depth("A", 3 * MIN, [(99.0, 5.0), (bad, 5.0)], [(101.0, 5.0)])
        assert e.value.parameter == "bids[1]"
        assert strategy.n == 1, "nothing was delivered"
        bar(node, "A", 1)  # the node still takes good data
        assert strategy.n == 2


def test_warmup_bars_are_checked_for_finite_prices_too():
    warm = {"timestamps": np.array([MIN, 2 * MIN], dtype=np.int64), "open": np.array([1.0, 1.0]), "high": np.array([1.0, float("nan")]), "low": np.array([1.0, 1.0]), "close": np.array([1.0, 1.0]), "volume": np.array([1.0, 1.0])}
    with pytest.raises(InvalidData) as e:
        LiveNode([Hooked()], ["A"], bar_stamp="close", warmup_bars={"A": warm})
    assert e.value.code == "non_finite_price"


def test_a_live_node_takes_corporate_actions_as_a_batch_run_does():
    import test_lifecycle as tl
    from raptorbt import CorporateAction

    closes = [100.0, 100.0, 50.0, 50.0, 55.0]
    data = {"A": tl.bars(closes)}
    actions = [CorporateAction.split("A", 2, 2.0)]
    batch_strategy = tl.holder("A", units=10.0)
    batch = tl.run(batch_strategy, data, corporate_actions=actions, adjustment_policy="raw")
    for execution in (None, PaperExecution()):
        live_strategy = tl.holder("A", units=10.0)
        node = LiveNode([live_strategy], ["A"], tl.config(), bar_stamp="close", corporate_actions=actions, adjustment_policy="raw", execution=execution)
        a = data["A"]
        for i in range(len(closes)):
            node.push_bar("A", int(a["timestamps"][i]), *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume")))
        live = node.finish()[0]
        assert [(t.size, t.entry_price, t.pnl) for t in live.result.trades()] == [(t.size, t.entry_price, t.pnl) for t in batch.result.trades()]
        assert list(live.result.equity_curve()) == list(batch.result.equity_curve())
        assert [e.outcome for e in live_strategy.lifecycle if e.kind == "corporate_action"] == ["split"]


def test_a_live_node_refuses_what_a_batch_run_refuses_about_corporate_actions():
    from raptorbt import CorporateAction

    split = [CorporateAction.split("A", 2, 2.0)]
    with pytest.raises(InvalidParameter) as e:
        LiveNode([Hooked()], ["A"], bar_stamp="close", corporate_actions=split)
    assert e.value.code == "invalid_adjustment_policy"
    with pytest.raises(InvalidParameter) as e:  # pushed events cannot be put on a later basis
        LiveNode([Hooked()], ["A"], bar_stamp="close", corporate_actions=split, adjustment_policy="adjusted")
    assert e.value.code == "invalid_adjustment_policy"
    with pytest.raises(Exception) as e:
        LiveNode([Hooked()], ["A"], bar_stamp="close", corporate_actions=[CorporateAction.split("B", 2, 2.0)], adjustment_policy="raw")
    assert e.value.code == "unknown_symbol"
