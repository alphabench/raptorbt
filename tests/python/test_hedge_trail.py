"""A trailing stop on a hedge leg trails the same in a backtest and on a node.

A bought call is the hedge of a sold call (a margin group), so its stop is the engine's to carry
out: it closes the sold leg first and then the hedge. With a trailing stop of 10% the level is 10%
below the highest price since entry, ratcheting up after each bar. A node used to hold the level
the hedge was bought with (4.5) and never raise it.
"""
import numpy as np
import pytest
from _chain_fixture import EXP1, STEP, T0, U, option_spec, symbol

from raptorbt import BacktestConfig, LiveNode, MarginModel, Strategy, run_strategies
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders as O

N = 14
OHLCV = ("open", "high", "low", "close", "volume")


def _series(o, h, l, c):
    return {"timestamps": T0 + STEP * np.arange(1, N + 1, dtype=np.int64), "open": np.array(o, float),
            "high": np.array(h, float), "low": np.array(l, float), "close": np.array(c, float),
            "volume": np.full(N, 1000.0)}


def _flat(v):
    return [v] * N


def _build():
    S, H = symbol(105.0, "call", EXP1), symbol(100.0, "call", EXP1)
    ho, hh, hl, hc = _flat(5.0), _flat(5.0), _flat(5.0), _flat(5.0)
    # bar 8: the hedge runs up to 8.0, so a 10% trail stands at 7.2 from the end of the bar;
    # bar 10: it trades down to 7.0 inside the bar (open 7.6), through 7.2 but far above 4.5
    ho[8], hh[8], hl[8], hc[8] = 5.0, 8.0, 5.0, 7.9
    for i in (9,):
        ho[i], hh[i], hl[i], hc[i] = 7.9, 7.9, 7.8, 7.8
    ho[10], hh[10], hl[10], hc[10] = 7.6, 7.6, 7.0, 7.1
    for i in (11, 12, 13):
        ho[i], hh[i], hl[i], hc[i] = 7.1, 7.1, 7.1, 7.1
    flat = _series(_flat(100.0), _flat(100.0), _flat(100.0), _flat(100.0))
    sold = _series(_flat(5.0), _flat(5.0), _flat(5.0), _flat(5.0))
    data = {U: flat, H: _series(ho, hh, hl, hc), S: sold}
    inst = {S: option_spec(105.0, "call", EXP1), H: option_spec(100.0, "call", EXP1)}
    return data, inst, S, H


class _Spread(Strategy):
    def __init__(self, S, H):
        super().__init__()
        self.S, self.H = S, H

    def on_bar(self, ctx):
        if ctx.symbol != U:
            return
        if ctx.idx == 3:
            self.submit_order(O.Market(side="buy", units=50.0, instrument=self.H), client_id="hedge")
        if ctx.idx == 4:
            self.submit_order(O.Market(side="sell", units=50.0, instrument=self.S), client_id="sold")


def _cfg():
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, margin_model=MarginModel.defined_risk())
    cfg.set_trailing_stop(0.10)
    return cfg


def _trades(res):
    return sorted((t.symbol, t.entry_price, t.exit_price, int((t.exit_time - T0) // STEP), str(t.exit_reason))
                  for t in res.trades())


def _run(mode):
    data, inst, S, H = _build()
    s = _Spread(S, H)
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    if mode == "batch":
        r = run_strategies([s], data, _cfg(), **kw)[0]
    else:
        ex = {"node": {}, "paper": {"execution": PaperExecution()}}[mode]
        node = LiveNode([s], list(data), _cfg(), **ex, **kw)
        for i in range(N):
            node.push_group(int(data[U]["timestamps"][i]),
                            {x: tuple(float(data[x][c][i]) for c in OHLCV) for x in data})
        r = node.finish()[0]
    return _trades(r.result), S, H


def test_a_trailing_stop_on_a_hedge_leg_ratchets_and_closes_the_group_the_same_in_every_mode():
    want, S, H = _run("batch")
    hedge = [t for t in want if t[0] == H]
    assert hedge and hedge[0][3] == 10, "the batch run stops the hedge out on bar 10 at its trailed level"
    assert [t for t in want if t[0] == S][0][3] == 10, "and closes the sold leg with it"
    for mode in ("node", "paper"):
        assert _run(mode)[0] == want, mode
