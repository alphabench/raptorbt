"""Protective levels, naked short margin, push_report outcomes."""
from _chain_fixture import EXP1, N_BARS, U, symbol, universe
from _market_harness import bars, ist
from test_structure_hedge_first import Broker, config, push_bar

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj, run_strategies
from raptorbt.execution import ExecutionReport, PaperExecution
from raptorbt.strategy import orders as O

MIN = 60_000_000_000


class _Long(Strategy):
    KW: dict = {}

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=10.0, **self.KW), client_id="b")


def _run(mode, kw):
    _Long.KW = kw
    data = {"A": bars([(100.0, 101.0, 99.0, 100.0)] * 8, ist(2026, 11, 2, 9, 15))}
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0)
    if mode == "batch":
        r = run_strategies([_Long()], data, cfg, bar_stamp="close")[0]
    else:
        node = LiveNode([_Long()], ["A"], cfg, bar_stamp="close", execution=PaperExecution())
        for i in range(8):
            node.push_bar("A", int(data["A"]["timestamps"][i]), 100.0, 101.0, 99.0, 100.0, 1000)
        r = node.finish()[0]
    return [(t.exit_price, t.pnl) for t in r.result.trades()]


def test_a_target_the_bar_never_traded_at_fills_at_the_open_in_batch_and_paper():
    for mode in ("batch", "paper"):
        # a long's target below every bar was already through: it fills where the bar opened
        assert _run(mode, dict(target_price=60.0)) == [(100.0, 0.0)], mode
        # a stop above every bar was already through too
        assert _run(mode, dict(stop_price=150.0)) == [(100.0, 0.0)], mode


class _HedgeThenSold(Strategy):
    def on_bar(self, ctx):
        if ctx.symbol != U:
            return
        hedge, sold = symbol(105, "call", EXP1), symbol(100, "call", EXP1)
        if ctx.idx == 3:
            self.submit_order(O.Market(side="buy", units=50.0, instrument=hedge), client_id="hedge")
        if ctx.idx == 4:
            self.submit_order(O.Market(side="sell", units=50.0, instrument=sold), client_id="sold")
        if ctx.idx == 6:
            self.submit_order(O.Market(side="sell", units=50.0, instrument=hedge, reduce_only=True), client_id="off")


def test_a_sold_option_left_without_a_hedge_stops_the_node_and_says_why():
    data, inst = universe()
    broker = Broker()
    node = LiveNode([_HedgeThenSold()], list(data), config(300.0), execution=broker, instruments=inst,
                    account_type="margin", leverage=1.0, bar_stamp="close")
    done, seen = set(), []
    for i in range(8):
        push_bar(node, data, i)
        ts = int(data[U]["timestamps"][i])
        for r in list(broker.requests):
            if r.client_order_id in done:
                continue
            done.add(r.client_order_id)
            node.push_report(r.client_order_id, ExecutionReport.accepted("V" + r.client_order_id, ts), arrival_ns=ts)
            node.push_report(r.client_order_id, ExecutionReport.fill("F" + r.client_order_id, r.quantity,
                                                                    float(data[r.symbol]["close"][i]), ts),
                             arrival_ns=ts)
        seen.append(node.health()["margin_faults"])
    assert seen[5] == [] and seen[6], seen
    assert "no hedge left" in seen[6][0][1]
    # the account is stopped: no entry is taken from here on
    assert node.health()["margin_faults"] == seen[6]


def test_push_report_tells_a_deferred_fill_from_a_refused_one():
    class Venue:
        namespace = "N1"

        def __init__(self):
            self.requests = []

        def submit(self, request):
            self.requests.append(request)

        def cancel(self, client_order_id):
            pass

        def modify(self, client_order_id, *, quantity=None, limit_price=None, trigger_price=None):
            pass

    class Trader(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.submit_order(O.Market(side="buy", quantity=5), client_id="in")
                self.submit_order(O.Market(side="sell", quantity=5, reduce_only=True), client_id="out")

    venue = Venue()
    node = LiveNode([Trader()], ["AAA"], BacktestConfig(fees=0.0), bar_stamp="close", execution=venue,
                    journal=rj.Journal({"run": "demo"}, memory=True))
    node.push_bar("AAA", MIN, 100.0, 101.0, 99.0, 100.0, 1000.0)
    entry, exit_ = [r.client_order_id for r in venue.requests]
    # the exit filled before the entry it reduces: kept, and books itself when the entry lands
    assert node.push_report(exit_, ExecutionReport.fill("X1", 5.0, 101.0, MIN), arrival_ns=MIN) == "deferred"
    assert node.health()["unbooked_fills"]
    assert node.push_report(entry, ExecutionReport.fill("E1", 5.0, 100.0, MIN), arrival_ns=MIN) == "applied"
    assert node.health()["unbooked_fills"] == []
