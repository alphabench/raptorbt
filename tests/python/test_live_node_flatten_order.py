"""A flattening halt on a venue closes the sold legs before the bought legs that cover them.

On a broker a bought wing closed while its sold leg is still open leaves the sold leg without
cover: the broker can refuse the close for margin, or the short stands naked for the seconds it
takes to follow. The halt goes through the same path as a stop on a hedge: the sold legs are
sent first, and the wing once the broker has said they are closed.
"""

import numpy as np
from _chain_fixture import EXP1, STEP, T0, U, option_spec, symbol

from raptorbt import BacktestConfig, LiveNode, MarginModel, Strategy
from raptorbt.execution import ExecutionReport
from raptorbt.strategy import orders as O

N = 12
LOT = 50.0

WING_PUT, WING_CALL = symbol(90, "put", EXP1), symbol(110, "call", EXP1)
BODY_PUT, BODY_CALL = symbol(95, "put", EXP1), symbol(105, "call", EXP1)


def _series(price):
    flat = np.full(N, float(price))
    return {
        "timestamps": T0 + STEP * np.arange(1, N + 1, dtype=np.int64),
        "open": flat, "high": flat + 0.05, "low": flat - 0.05, "close": flat.copy(),
        "volume": np.full(N, 1000.0),
    }  # fmt: skip


# the wings sit before the bodies in the node's instrument order, so a flatten that closes in
# that order sends the wings first
DATA = {U: _series(100.0), WING_PUT: _series(2.0), WING_CALL: _series(2.0), BODY_PUT: _series(5.0), BODY_CALL: _series(5.0)}
INSTRUMENTS = {
    s: option_spec(k, r, EXP1)
    for s, k, r in ((WING_PUT, 90, "put"), (WING_CALL, 110, "call"), (BODY_PUT, 95, "put"), (BODY_CALL, 105, "call"))
}  # fmt: skip


class Condor(Strategy):
    def on_bar(self, ctx):
        if ctx.symbol != U:
            return
        if ctx.idx == 3:
            for wing in (WING_PUT, WING_CALL):
                self.submit_order(O.Market(side="buy", units=LOT, instrument=wing), client_id=f"wing-{wing}")
        if ctx.idx == 4:
            for body in (BODY_PUT, BODY_CALL):
                self.submit_order(O.Market(side="sell", units=LOT, instrument=body), client_id=f"body-{body}")


class Broker:
    """A broker that records what it is sent. It answers when the test tells it to."""

    namespace = "BROKER"
    keeps_own_records = True

    def __init__(self):
        self.requests = []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, **changes):
        pass


def push(node, i):
    ts = int(DATA[U]["timestamps"][i])
    node.push_group(
        ts, {s: tuple(float(DATA[s][c][i]) for c in ("open", "high", "low", "close", "volume")) for s in DATA}
    )
    return ts


def answer(node, requests, i, ts, fill=True):
    for r in requests:
        node.push_report(r.client_order_id, ExecutionReport.accepted("V" + r.client_order_id, ts), arrival_ns=ts)
        if fill:
            price = float(DATA[r.symbol]["close"][i])
            node.push_report(
                r.client_order_id, ExecutionReport.fill("F" + r.client_order_id, r.quantity, price, ts), arrival_ns=ts
            )


def legs(requests):
    return sorted((r.symbol, str(r.side).lower()) for r in requests)


def condor_on_a_broker(**config_kw):
    broker = Broker()
    config = BacktestConfig(initial_capital=1e6, fees=0.0, margin_model=MarginModel.defined_risk(), **config_kw)
    node = LiveNode(
        [Condor()], list(DATA), config, execution=broker, instruments=INSTRUMENTS,
        account_type="margin", leverage=1.0, bar_stamp="close",
    )  # fmt: skip
    seen = 0
    for i in range(7):
        ts = push(node, i)
        answer(node, broker.requests[seen:], i, ts)
        seen = len(broker.requests)
    held = {s for s, rows in node.positions(0).items() if rows}
    assert held == {WING_PUT, WING_CALL, BODY_PUT, BODY_CALL}, held
    return node, broker


def test_a_flattening_halt_sends_the_sold_call_first_and_its_wing_once_it_is_closed():
    node, broker = condor_on_a_broker()
    before = len(broker.requests)
    node.halt(0, flatten=True)
    ts = push(node, 7)
    first = broker.requests[before:]
    # the call wing covers the sold call (a short call has no bound): the wing waits
    assert (BODY_CALL, "buy") in legs(first), legs(first)
    assert (WING_CALL, "sell") not in legs(first), legs(first)
    assert all(r.reduce_only for r in first)
    answer(node, first, 7, ts)
    second = broker.requests[before + len(first) :]
    assert (WING_CALL, "sell") in legs(second), legs(second)
    answer(node, second, 7, ts)
    assert not any(rows for rows in node.positions(0).values())
    # every position was closed once, and none twice
    closes = legs(broker.requests[before:])
    assert closes == sorted([(WING_PUT, "sell"), (WING_CALL, "sell"), (BODY_PUT, "buy"), (BODY_CALL, "buy")]), closes


def test_a_session_square_off_closes_each_leg_once_sold_call_first_and_flips_no_line():
    node, broker = condor_on_a_broker(squareoff_time="00:08", session_tz_offset_ns=0)
    before = len(broker.requests)
    ts = push(node, 7)  # 00:08: the square-off
    first = broker.requests[before:]
    assert (BODY_CALL, "buy") in legs(first), legs(first)
    assert (WING_CALL, "sell") not in legs(first), legs(first)
    answer(node, first, 7, ts)
    second = broker.requests[before + len(first) :]
    assert (WING_CALL, "sell") in legs(second), legs(second)
    answer(node, second, 7, ts)
    assert not any(rows for rows in node.positions(0).values())
    # a second close of a sold leg would turn it long: each line is closed exactly once
    closes = legs(broker.requests[before:])
    assert closes == sorted([(WING_PUT, "sell"), (WING_CALL, "sell"), (BODY_PUT, "buy"), (BODY_CALL, "buy")]), closes
