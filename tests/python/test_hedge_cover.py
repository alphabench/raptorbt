"""The stop of a hedge leg is the engine's, in every mode: sold legs close first, then the hedge.

One event, one outcome: the same trades in a batch run, on a node, on the paper venue, whichever
fill timing and whichever strike is the sold one. On a broker the stop of a hedge is an ordinary
resting order only while no sold leg depends on it; once one does, the order is withdrawn and the
engine sends reduce-only market closes, the sold leg first and the hedge once the broker has said
the sold leg is closed (a level that uncovers nothing is not the engine's).
"""
import numpy as np
import pytest
from _chain_fixture import EXP1, STEP, T0, U, option_spec, symbol

from raptorbt import BacktestConfig, LiveNode, MarginModel, Strategy, run_strategies
from raptorbt.execution import ExecutionReport, PaperExecution
from raptorbt.strategy import orders as O

N = 14


def _series(o, h, l, c):
    return {"timestamps": T0 + STEP * np.arange(1, N + 1, dtype=np.int64), "open": np.array(o, float),
            "high": np.array(h, float), "low": np.array(l, float), "close": np.array(c, float),
            "volume": np.full(N, 1000.0)}


def _flat(v):
    return [v] * N


def _build(sold_k, hedge_k, hedge_open):
    S, H = symbol(sold_k, "call", EXP1), symbol(hedge_k, "call", EXP1)
    ho, hh, hl, hc = _flat(5.0), _flat(5.0), _flat(5.0), _flat(5.0)
    so, sh_, sl, sc = _flat(5.0), _flat(5.0), _flat(5.0), _flat(5.0)
    # bar 10: the sold leg runs up; the hedge either gaps through its stop 4.0 at the open
    # (open 3.0) or trades down to it inside the bar (open 5.0, low 3.5, close 3.8)
    if hedge_open:
        ho[10], hh[10], hl[10], hc[10] = 3.0, 3.2, 2.8, 3.0
    else:
        ho[10], hh[10], hl[10], hc[10] = 5.0, 5.0, 3.5, 3.8
    so[10], sh_[10], sl[10], sc[10] = 6.0, 9.5, 5.9, 9.0
    flat = _series(_flat(100.0), _flat(100.0), _flat(100.0), _flat(100.0))
    data = {U: flat, H: _series(ho, hh, hl, hc), S: _series(so, sh_, sl, sc)}
    inst = {S: option_spec(sold_k, "call", EXP1), H: option_spec(hedge_k, "call", EXP1)}
    return data, inst, S, H


class _Spread(Strategy):
    def __init__(self, S, H, stop=4.0):
        super().__init__()
        self.S, self.H, self.stop, self.heard = S, H, stop, []

    def on_bar(self, ctx):
        if ctx.symbol != U:
            return
        if ctx.idx == 3:
            self.submit_order(O.Market(side="buy", units=50.0, instrument=self.H, stop_price=self.stop),
                              client_id="hedge")
        if ctx.idx == 4:
            self.submit_order(O.Market(side="sell", units=50.0, instrument=self.S), client_id="sold")

    def on_order_event(self, ctx, e):
        if e.kind == "order_filled":
            self.heard.append(e.client_order_id)


def _cfg(timing):
    return BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing, margin_model=MarginModel.defined_risk())


def _trades(res):
    return sorted((t.symbol, t.entry_price, t.exit_price, int((t.exit_time - T0) // STEP), str(t.exit_reason))
                  for t in res.trades())


def _run(mode, timing, sold_k, hedge_k, hedge_open):
    data, inst, S, H = _build(sold_k, hedge_k, hedge_open)
    s = _Spread(S, H)
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    if mode == "batch":
        r = run_strategies([s], data, _cfg(timing), **kw)[0]
    else:
        ex = {"node": {}, "paper": {"execution": PaperExecution()}}[mode]
        node = LiveNode([s], list(data), _cfg(timing), **ex, **kw)
        syms = list(data)
        for i in range(N):
            node.push_group(int(data[U]["timestamps"][i]),
                            {x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume"))
                             for x in syms})
        r = node.finish()[0]
    return _trades(r.result), s.heard


@pytest.mark.parametrize("hedge_open", [True, False])
@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
@pytest.mark.parametrize("strikes", [(100, 105), (95, 100)])
def test_a_hedge_stop_closes_the_sold_leg_first_and_alike_in_every_mode(strikes, timing, hedge_open):
    got = {m: _run(m, timing, *strikes, hedge_open) for m in ("batch", "node", "paper")}
    assert got["batch"][0] == got["node"][0] == got["paper"][0], got
    trades = got["batch"][0]
    assert len(trades) == 2 and all(t[4] == "Order" for t in trades)
    # the two strike pairs sort their symbols the other way round: the price of the cover is the
    # same event's price whichever instrument the book steps first (see the strikes parametrisation)
    sold = [t for t in trades if t[0].endswith(f"{strikes[0]}CE")][0]
    hedge = [t for t in trades if t[0].endswith(f"{strikes[1]}CE")][0]
    # the sold leg is bought back at the bar of the event (close, or the next open); no later than the hedge
    assert sold[3] <= hedge[3]
    if timing == "same_bar_close":
        assert (sold[2], hedge[2]) == (9.0, 3.0 if hedge_open else 3.8)


class _Broker:
    namespace = "BROKER"
    keeps_own_records = True

    def __init__(self):
        self.requests, self.cancelled = [], []

    def submit(self, request):
        self.requests.append(request)

    def cancel(self, client_order_id):
        self.cancelled.append(client_order_id)

    def modify(self, client_order_id, **changes):
        pass


@pytest.mark.parametrize("strikes", [(100, 105), (95, 100)])
def test_a_broker_holds_no_stop_of_a_hedge_leg_once_a_sold_leg_depends_on_it_and_is_sent_the_sold_close_first(
    strikes,
):
    data, inst, S, H = _build(*strikes, True)
    s = _Spread(S, H)
    broker = _Broker()
    node = LiveNode([s], list(data), _cfg("same_bar_close"), execution=broker, instruments=inst,
                    account_type="margin", leverage=1.0, bar_stamp="close")
    done, acked, order_of_close = set(), set(), []
    for i in range(N):
        ts = int(data[U]["timestamps"][i])
        node.push_group(ts, {x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume"))
                             for x in data})
        progressed = True
        while progressed:
            progressed = False
            for r in list(broker.requests):
                if r.client_order_id in done:
                    continue
                done.add(r.client_order_id)
                progressed = True
                resting = "stop" in str(getattr(r, "order_type", getattr(r, "kind", ""))).lower()
                if r.symbol in (S, H) and getattr(r, "reduce_only", False) and not resting:
                    order_of_close.append((r.symbol, str(r.side)))
                node.push_report(r.client_order_id, ExecutionReport.accepted("V" + r.client_order_id, ts), arrival_ns=ts)
                if resting:
                    continue  # a resting stop does not trade: the market has not reached it
                node.push_report(r.client_order_id, ExecutionReport.fill(
                    "F" + r.client_order_id, r.quantity, float(data[r.symbol]["close"][i]), ts), arrival_ns=ts)
            for cid in broker.cancelled:
                if cid not in acked:
                    acked.add(cid)
                    progressed = True
                    node.push_report(cid, ExecutionReport.cancel_ack(ts), arrival_ns=ts)
    stops = {r.client_order_id for r in broker.requests
             if "stop" in str(getattr(r, "order_type", getattr(r, "kind", ""))).lower()}
    # the stop sent while the hedge stood alone was withdrawn when the sold leg opened
    assert stops <= set(broker.cancelled), (stops, broker.cancelled)
    assert [sym for sym, _ in order_of_close] == [S, H], order_of_close
    assert [side.lower() for _, side in order_of_close] == ["buy", "sell"], order_of_close
