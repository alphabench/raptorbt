"""``submit_structure`` places a multi-leg structure hedge-first, the same way in every mode.

In plain words: the bought legs of a structure (its hedges) go out first. A sold leg goes out only
once the hedges covering it have filled, and only for the quantity they cover. If a sold leg is
refused, the hedges that filled are closed with orders of their own and the structure says why.
One engine path does this in a backtest, on a node with simulated execution, on paper and on a
broker, so a strategy that backtests with trades places the same trades when it is deployed.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, MarginModel, OptionChain, Strategy, run_strategies
from raptorbt import structures as st
from raptorbt.execution import ExecutionReport, PaperExecution

from _chain_fixture import N_BARS, U, specs, universe

CHAIN = OptionChain(specs())
OHLCV = ("open", "high", "low", "close", "volume")
MODES = ("batch", "node", "paper", "broker")


def bear_call_spread():
    return st.bear_call_spread(short_steps=0, long_steps=1)


def iron_condor():
    return st.iron_condor(1, 2)


class Opens(Strategy):
    """Submits ``structure`` on the underlying's third bar; the call is the one a user makes."""

    structure = None
    kwargs: dict = {}

    def __init__(self):
        super().__init__()
        self.group = None

    def on_bar(self, ctx):
        if ctx.symbol == U and self.group is None and ctx.idx == 3:
            self.group = self.submit_structure(
                ctx, self.structure, chain=CHAIN, underlying=U, spot=ctx.bar.close, **self.kwargs
            )


def make(structure, **kwargs):
    return type("Opens", (Opens,), {"structure": structure, "kwargs": kwargs})()


def config(capital, margin=True):
    return BacktestConfig(
        initial_capital=capital, fees=0.0, margin_model=MarginModel.defined_risk() if margin else None
    )


class Broker:
    """A broker connection: it records what it is sent and says nothing until told to."""

    namespace = "BROKER"
    keeps_own_records = True

    def __init__(self):
        self.requests = []
        self.cancelled = []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        self.cancelled.append(client_order_id)

    def modify(self, client_order_id, **changes):
        pass


def push_bar(node, data, i):
    node.push_group(
        int(data[U]["timestamps"][i]), {s: tuple(float(data[s][c][i]) for c in OHLCV) for s in data}
    )


def run_mode(mode, strategy, capital, margin=True):
    """Run ``strategy`` on the option universe in ``mode``; returns ``(strategy, result, broker)``."""
    data, inst = universe()
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    if mode == "batch":
        return strategy, run_strategies([strategy], data, config(capital, margin), **kw)[0], None
    broker = Broker() if mode == "broker" else None
    execution = {"node": None, "paper": PaperExecution(), "broker": broker}[mode]
    node = LiveNode([strategy], list(data), config(capital, margin), execution=execution, **kw)
    done = 0
    for i in range(N_BARS):
        push_bar(node, data, i)
        if broker is None:
            continue
        ts = int(data[U]["timestamps"][i])
        # the broker answers every request at the bar's close, and the answers may bring more requests
        while done < len(broker.requests):
            request = broker.requests[done]
            done += 1
            node.push_report(request.client_order_id, ExecutionReport.accepted(f"V{done}", ts), arrival_ns=ts)
            price = float(data[request.symbol]["close"][i])
            node.push_report(
                request.client_order_id, ExecutionReport.fill(f"x{done}", request.quantity, price, ts),
                arrival_ns=ts,
            )
    return strategy, node.finish()[0], broker


def digest(result):
    """What a run did, by symbol: the trades and the final state of each order."""
    trades = sorted(
        (t.symbol, t.entry_idx, round(t.entry_price, 9), t.size, round(t.pnl, 6)) for t in result.result.trades()
    )
    orders = sorted((o.symbol, o.side, o.status, o.reject_reason) for o in result.result.orders())
    return trades, orders


@pytest.mark.parametrize("structure", [bear_call_spread, iron_condor], ids=["bear_call_spread", "iron_condor"])
def test_a_structure_gives_identical_trades_in_batch_a_node_paper_and_a_broker(structure):
    runs = {mode: run_mode(mode, make(structure()), 1_000.0) for mode in MODES}
    want = digest(runs["batch"][1])
    assert len(want[0]) == len(runs["batch"][0].group.legs) > 1, "the structure placed no trades"
    for mode, (strategy, result, _) in runs.items():
        assert digest(result) == want, mode
        assert strategy.structure_status(strategy.group) == "filled", mode
    entries = {t[1] for t in want[0]}
    assert len(entries) == 1, "every leg filled on the bar the structure was submitted on"


@pytest.mark.parametrize("structure", [bear_call_spread, iron_condor], ids=["bear_call_spread", "iron_condor"])
def test_the_hedges_go_out_first_and_a_sold_leg_waits_for_its_hedge(structure):
    strategy, _, broker = run_mode("broker", make(structure()), 1_000.0)
    sides = [r.side for r in broker.requests]
    legs = strategy.group.structure.legs
    # the broker answers a request at once, so a sold leg follows its own hedge; none precedes it
    assert sides[0] == "buy" and "buy" in sides[: sides.index("sell")]
    for covering, sold in zip(*(
        [r for r in broker.requests if r.side == side] for side in ("buy", "sell")
    )):
        assert broker.requests.index(covering) < broker.requests.index(sold)
    assert sorted(sides) == sorted(leg.side for leg in legs)


def test_a_sold_leg_the_broker_refuses_closes_the_hedge_with_an_order_that_says_why():
    strategy = make(bear_call_spread())
    data, inst = universe()
    broker = Broker()
    node = LiveNode([strategy], list(data), config(1_000.0), instruments=inst, account_type="margin",
                    leverage=1.0, bar_stamp="close", execution=broker)
    done = 0
    for i in range(N_BARS):
        push_bar(node, data, i)
        ts = int(data[U]["timestamps"][i])
        while done < len(broker.requests):
            request = broker.requests[done]
            done += 1
            node.push_report(request.client_order_id, ExecutionReport.accepted(f"V{done}", ts), arrival_ns=ts)
            if request.side == "sell" and not request.reduce_only:
                node.push_report(request.client_order_id, ExecutionReport.rejected("RMS: margin", ts), arrival_ns=ts)
            else:
                price = float(data[request.symbol]["close"][i])
                node.push_report(request.client_order_id, ExecutionReport.fill(f"x{done}", request.quantity, price, ts),
                                 arrival_ns=ts)
    group = strategy.group
    assert strategy.structure_status(group) == "closed" and group.reason == "leg_failure"
    sold = next(ls for ls in group.legs if ls.leg.side == "sell")
    assert group.failure == (sold.client_id, "rejected", "RMS: margin")
    closes = [r for r in broker.requests if r.reduce_only]
    assert len(closes) == 1, "one close, for the one hedge that filled"
    assert closes[0].side == "sell" and closes[0].tags["close"] == "leg_failure"
    assert closes[0].tags["structure"] == group.id
    hedge = next(ls for ls in group.legs if ls.leg.side == "buy")
    assert closes[0].symbol == hedge.leg.instrument and closes[0].quantity == hedge.leg.quantity
    trades = node.finish()[0].result.trades()
    assert all(t.symbol == hedge.leg.instrument for t in trades), "only the hedge was ever held"


@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_a_sold_leg_refused_for_want_of_funds_closes_the_hedge_in_every_mode(mode):
    # 300 buys the hedge (294) and cannot carry the sold call on the built-in deposit
    strategy, result, _ = run_mode(mode, make(bear_call_spread()), 300.0, margin=False)
    group = strategy.group
    assert strategy.structure_status(group) == "closed" and group.reason == "leg_failure"
    assert group.failure is not None and group.failure[1] == "rejected"
    orders = result.result.orders()
    assert [o.reject_reason for o in orders if o.status == "rejected"] == [group.failure[2]]
    closes = [o for o in orders if o.client_id.startswith(f"{group.id}:x")]
    assert len(closes) == 1 and closes[0].status == "filled"


def test_partial_hedge_fills_cap_the_sold_quantity():
    strategy = make(bear_call_spread(), lots=2)
    data, inst = universe()
    broker = Broker()
    node = LiveNode([strategy], list(data), config(5_000.0), instruments=inst, account_type="margin",
                    leverage=1.0, bar_stamp="close", execution=broker)
    served = 0
    sold_quantities = []
    for i in range(N_BARS):
        push_bar(node, data, i)
        if strategy.group is None:
            continue
        ts = int(data[U]["timestamps"][i])
        hedge = broker.requests[0]
        assert hedge.side == "buy" and hedge.quantity == 100.0
        assert [r.side for r in broker.requests] == ["buy"], "no sold leg before a hedge filled"
        price = float(data[hedge.symbol]["close"][i])
        node.push_report(hedge.client_order_id, ExecutionReport.accepted("V1", ts), arrival_ns=ts)
        node.push_report(hedge.client_order_id, ExecutionReport.fill("x1", 50.0, price, ts), arrival_ns=ts)
        sold_quantities.append([r.quantity for r in broker.requests if r.side == "sell"])
        node.push_report(hedge.client_order_id, ExecutionReport.fill("x2", 50.0, price, ts), arrival_ns=ts)
        sold_quantities.append([r.quantity for r in broker.requests if r.side == "sell"])
        break
    assert sold_quantities == [[50.0], [50.0, 50.0]], "a lot sold per lot of hedge, never ahead of it"


def test_the_iron_condor_at_capital_300_is_closed_by_its_unfunded_hedge_without_reduce_only_noise():
    for mode in ("batch", "node", "paper"):
        strategy, result, _ = run_mode(mode, make(iron_condor()), 300.0)
        group = strategy.group
        assert strategy.structure_status(group) == "closed", mode
        assert group.reason == "leg_failure" and group.failure[1:] == ("rejected", "insufficient_capital"), mode
        orders = result.result.orders()
        assert [o.reject_reason for o in orders if o.status == "rejected"] == ["insufficient_capital"], mode
        assert not any(o.reject_reason == "reduce_only" for o in orders), mode
        # nothing is left open: every leg that filled was closed by the engine's own order
        net = {}
        for o in orders:
            if o.status == "filled":
                net[o.symbol] = net.get(o.symbol, 0.0) + (o.filled_qty if o.side == "buy" else -o.filled_qty)
        assert all(v == 0.0 for v in net.values()), (mode, net)


def test_the_order_of_the_legs_in_the_request_does_not_change_the_result():
    ordered = iron_condor()
    reversed_legs = st.Structure(legs=list(reversed(ordered.legs)), name="reversed")
    for capital in (1_000.0, 300.0):
        want = digest(run_mode("batch", make(ordered), capital)[1])
        for mode in ("batch", "node", "paper"):
            got = digest(run_mode(mode, make(reversed_legs), capital)[1])
            assert got == want, (capital, mode)


@pytest.mark.parametrize("mode", ["batch", "node", "paper"])
def test_a_funding_refusal_on_the_order_path_is_counted_as_an_unfunded_entry(mode):
    # the put hedge is bought (164 of 300), the call hedge (193) is refused for capital
    _, result, _ = run_mode(mode, make(iron_condor()), 300.0)
    refused = [o for o in result.result.orders() if o.reject_reason == "insufficient_capital"]
    assert len(refused) == 1
    assert result.result.unfunded_entries == 1


def test_taking_a_hedge_off_a_sold_option_the_model_cannot_margin_alone_is_refused_not_a_crash():
    from raptorbt.strategy import orders as O

    from _chain_fixture import EXP1, symbol

    hedge, sold = symbol(105, "call", EXP1), symbol(100, "call", EXP1)

    class Closes(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol != U:
                return
            if ctx.idx == 3:
                self.submit_order(O.Market(side="buy", units=50.0, instrument=hedge), client_id="hedge")
            if ctx.idx == 4:
                self.submit_order(O.Market(side="sell", units=50.0, instrument=sold), client_id="sold")
            if ctx.idx == 6:
                self.submit_order(O.Market(side="sell", units=50.0, instrument=hedge, reduce_only=True),
                                  client_id="take_off")

    data, inst = universe()
    result = run_strategies([Closes()], data, config(300.0), instruments=inst, account_type="margin",
                            leverage=1.0, bar_stamp="close")[0]
    rows = {o.client_id: (o.status, o.reject_reason) for o in result.result.orders()}
    assert rows["hedge"][0] == rows["sold"][0] == "filled"
    assert rows["take_off"] == ("rejected", "insufficient_margin"), "the hedge stays on"
    assert result.result.unfunded_entries == 1
