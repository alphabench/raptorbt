"""A stop or target is the engine's only when closing the position would leave a sold leg
without its cover; otherwise it is an ordinary resting order, whichever mode and whichever timing.

On an account whose margin model groups option lines (`defined_risk`) the stop of a bought option
used to be held by the engine whatever the rest of the group held: a lone long call's target
became a market order at the next price, and the stop of a leg that covers one sold option closed
every sold option of the group, then the hedge, though other bought legs still covered them.

Now a level is held only when closing leaves a sold leg the model cannot bound, and then only the
sold legs left bare are closed first. The same trades in a batch run, on a node and on the paper
venue.
"""

import pytest
from _chain_fixture import EXP1, N_BARS, U, symbol, universe
from test_structure_hedge_first import push_bar

from raptorbt import BacktestConfig, LiveNode, MarginModel, Strategy, run_strategies
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders as O

MODES = ["batch", "node", "paper"]
TIMINGS = ["same_bar_close", "next_bar_open"]


def _run(mode, strategy, data, inst, timing):
    cfg = BacktestConfig(
        initial_capital=1e6, fees=0.0, fill_timing=timing, margin_model=MarginModel.defined_risk()
    )
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    if mode == "batch":
        return run_strategies([strategy], data, cfg, **kw)[0]
    node = LiveNode(
        [strategy], list(data), cfg, execution=PaperExecution() if mode == "paper" else None, **kw
    )
    for i in range(N_BARS):
        push_bar(node, data, i)
    return node.finish()[0]


def _trades(r):
    return sorted(
        (t.symbol, t.entry_idx, t.exit_idx, round(t.entry_price, 3), round(t.exit_price, 3))
        for t in r.result.trades()
    )


CALL = symbol(105, "call", EXP1)


class _Lone(Strategy):
    """Buys a call on bar 3, with the level under test; no option is sold anywhere."""

    def __init__(self, **level):
        super().__init__()
        self.level = level

    def on_bar(self, ctx):
        if ctx.symbol == U and ctx.idx == 3:
            self.submit_order(
                O.Market(side="buy", units=50.0, instrument=CALL, **self.level), client_id="lone"
            )


def _lone(mode, timing, level, bar5):
    data, inst = universe()
    d = data[CALL]
    for k, v in zip(("open", "high", "low", "close"), bar5):
        d[k][5] = v
    return _run(mode, _Lone(**level), data, inst, timing)


@pytest.mark.parametrize("timing", TIMINGS)
def test_the_target_of_a_lone_long_call_fills_at_its_level_in_every_mode(timing):
    # Bar 5 runs up through the target 6.2 (high 6.5) and closes lower, at 5.75.
    got = {m: _trades(_lone(m, timing, {"target_price": 6.2}, (5.8, 6.5, 5.7, 5.75))) for m in MODES}
    assert got["batch"] == got["node"] == got["paper"], got
    ((_, _, exit_idx, _, exit_price),) = got["batch"]
    assert (exit_idx, exit_price) == (5, 6.2), got


@pytest.mark.parametrize("timing", TIMINGS)
def test_the_stop_of_a_lone_long_call_fills_at_its_level_in_every_mode(timing):
    # Bar 5 opens at 5.8 and trades down through the stop 5.75 (low 5.7).
    got = {m: _trades(_lone(m, timing, {"stop_price": 5.75}, (5.8, 5.85, 5.7, 5.78))) for m in MODES}
    assert got["batch"] == got["node"] == got["paper"], got
    ((_, _, exit_idx, _, exit_price),) = got["batch"]
    assert (exit_idx, exit_price) == (5, 5.75), got


LONG_CALL, SHORT_CALL = symbol(110, "call", EXP1), symbol(105, "call", EXP1)
SHORT_PUT, LONG_PUT = symbol(95, "put", EXP1), symbol(90, "put", EXP1)


class _Condor(Strategy):
    """An iron condor, the bought legs first. One bought leg carries the stop."""

    def __init__(self, leg, stop):
        super().__init__()
        self.leg, self.stop = leg, stop

    def on_bar(self, ctx):
        if ctx.symbol != U:
            return
        if ctx.idx == 3:
            for leg in (LONG_CALL, LONG_PUT):
                level = {"stop_price": self.stop} if leg == self.leg else {}
                self.submit_order(O.Market(side="buy", units=50.0, instrument=leg, **level), client_id=leg)
        if ctx.idx == 4:
            self.submit_order(O.Market(side="sell", units=50.0, instrument=SHORT_CALL), client_id="short_call")
            self.submit_order(O.Market(side="sell", units=50.0, instrument=SHORT_PUT), client_id="short_put")


def _condor(mode, timing, leg):
    data, inst = universe()
    # The stop sits just above the leg's low on bar 7: it trips there and nowhere before.
    stop = float(round(data[leg]["low"][7] + 0.0005, 4))
    return _run(mode, _Condor(leg, stop), data, inst, timing)


def _exits(mode, timing, leg):
    r = _condor(mode, timing, leg)
    return _trades(r), {sym[-5:]: exit_idx for sym, _, exit_idx, _, _ in _trades(r)}


@pytest.mark.parametrize("timing", TIMINGS)
def test_the_stop_of_a_put_wing_that_leaves_no_sold_leg_bare_is_an_ordinary_stop(timing):
    got = {m: _exits(m, timing, LONG_PUT) for m in MODES}
    assert got["batch"] == got["node"] == got["paper"], got
    exits = got["batch"][1]
    # A naked sold put has a bound under the model (its strike): only the stopped leg closes,
    # at its level, and the sold 105 call stays covered by the long 110 call.
    assert exits["A90PE"] <= 8 and exits["A95PE"] == exits["105CE"] == exits["110CE"] == N_BARS - 1, exits
    stopped = [t for t in got["batch"][0] if t[0] == LONG_PUT][0]
    assert stopped[4] == pytest.approx(universe()[0][LONG_PUT]["low"][7] + 0.0005, abs=1e-3)


@pytest.mark.parametrize("timing", TIMINGS)
def test_the_stop_of_the_call_wing_closes_only_the_sold_call_it_leaves_bare(timing):
    got = {m: _exits(m, timing, LONG_CALL) for m in MODES}
    assert got["batch"] == got["node"] == got["paper"], got
    exits = got["batch"][1]
    # The sold 105 call has no bound without the long 110 call: it closes first, then the stopped
    # leg. The put wing is not touched.
    assert exits["105CE"] <= exits["110CE"] <= 8, exits
    assert exits["A90PE"] == exits["A95PE"] == N_BARS - 1, exits
