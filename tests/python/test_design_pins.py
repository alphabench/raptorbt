"""Behaviours that are deliberate designs, pinned so a change to them is a decision.

* after a sold option is left without a hedge the strategy is stopped for good: it cannot buy
  the hedge back, and ``margin_faults`` keeps saying why (a halt cannot be undone).
* the hook of a fill at a bar's open reads another instrument's indicator as of the previous
  bar, not the value the group of that bar brings.
"""
from _chain_fixture import EXP1, U, symbol, universe
from _market_harness import bars, fmt, ist
from raptorbt import BacktestConfig, Indicator, LiveNode, Strategy, run_strategies
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders as O
from test_hedge_money import _HedgeThenSold
from test_structure_hedge_first import Broker, config, push_bar


def test_a_strategy_stopped_for_a_bare_sold_leg_cannot_buy_the_hedge_back():
    class Rebuy(_HedgeThenSold):
        def on_bar(self, ctx):
            super().on_bar(ctx)
            if ctx.symbol == U and ctx.idx == 8:
                self.submit_order(O.Market(side="buy", units=50.0, instrument=symbol(105, "call", EXP1)),
                                  client_id="rebuy")

        def on_order_event(self, ctx, e):
            if e.kind == "order_rejected":
                self.rejected = (e.client_order_id, e.reject_reason)

    data, inst = universe()
    broker, strategy, done = Broker(), Rebuy(), set()
    node = LiveNode([strategy], list(data), config(300.0), execution=broker, instruments=inst,
                    account_type="margin", leverage=1.0, bar_stamp="close")
    for i in range(10):
        push_bar(node, data, i)
        ts = int(data[U]["timestamps"][i])
        for r in list(broker.requests):
            if r.client_order_id not in done:
                done.add(r.client_order_id)
                node.push_report(r.client_order_id, ExecutionReport.accepted("V" + r.client_order_id, ts), arrival_ns=ts)
                node.push_report(r.client_order_id, ExecutionReport.fill(
                    "F" + r.client_order_id, r.quantity, float(data[r.symbol]["close"][i]), ts), arrival_ns=ts)
    assert strategy.rejected == ("rebuy", "margin_call")
    assert "no hedge left" in node.health()["margin_faults"][0][1]


class _OpenHook(Strategy):
    def __init__(self):
        super().__init__()
        self.heard = []

    def on_start(self, ctx):
        self.b = self.register_indicator(Indicator.sma(1), symbol="B")

    def on_bar(self, ctx):
        if ctx.symbol == "A" and ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=1.0), client_id="p")

    def on_order_event(self, ctx, e):
        if e.kind == "order_filled":
            self.heard.append((fmt(ctx.event_ns), self.b.value))


def test_the_hook_of_an_open_fill_reads_another_instruments_indicator_before_the_bar():
    t0 = ist(2026, 11, 2, 9, 15)
    a = [(100,) * 4] * 6
    b = [(10 * k, 10 * k + 1, 10 * k - 1, 10 * k) for k in range(1, 7)]
    data = {"A": bars(a, t0), "B": bars(b, t0)}
    cfg = lambda: BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")  # noqa: E731
    s = _OpenHook()
    run_strategies([s], data, cfg(), bar_stamp="close")
    # decided on bar 1, filled at the open of bar 2: B's close of bar 1 (20) is known, bar 2's (30) is not
    assert s.heard == [("11-02 09:18", 20.0)]
    s = _OpenHook()
    node = LiveNode([s], ["A", "B"], cfg(), bar_stamp="close")
    for i in range(6):
        node.push_group(int(data["A"]["timestamps"][i]), {
            x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume")) for x in data})
    node.finish()
    assert s.heard == [("11-02 09:18", 20.0)]
