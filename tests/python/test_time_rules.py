"""Time. A decision at t meets no bar stamped at or before t; a halt closes like an ordinary
reduce-only market order; only a fill the engine is delivering can be answered at its instant; no
hook reaches a bar of the group that has not been matched; time rules in the fill path use the
fill's instant; a retried bar is a duplicate; a group holds its own timestamp."""

import numpy as np
import pytest
from _chain_fixture import STEP, U, specs, universe
from _market_harness import fmt, ist

from raptorbt import (
    BacktestConfig,
    LiveNode,
    MarginModel,
    OptionChain,
    Strategy,
    run_strategies,
    structures as st,
)
from raptorbt.errors import StateError
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders as O

MIN = 60_000_000_000
T0 = ist(2026, 11, 2, 9, 15)
TIMINGS = ("same_bar_close", "next_bar_open")
MODES = ("batch", "node", "paper")


def _bars(rows):
    a = np.array(rows, dtype=float)
    return {
        "timestamps": np.array([T0 + (i + 1) * MIN for i in range(len(rows))], dtype=np.int64),
        "open": a[:, 0],
        "high": a[:, 1],
        "low": a[:, 2],
        "close": a[:, 3],
        "volume": np.full(len(rows), 1e6),
    }


def _run(strategy, data, cfg, mode, **kw):
    if mode == "batch":
        return run_strategies([strategy], data, cfg, bar_stamp="close", **kw)[0]
    syms = list(data)
    node = LiveNode(
        [strategy], syms, cfg, bar_stamp="close",
        execution=PaperExecution() if mode == "paper" else None, **kw,
    )
    n = len(data[syms[0]]["timestamps"])
    for i in range(n):
        node.push_group(
            int(data[syms[0]]["timestamps"][i]),
            {x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume")) for x in syms},
        )
    return node.finish()[0]


# 1. a resting order decided at t meets no bar of its target stamped at or before t
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("names", [("AX", "BY"), ("BX", "AY")])
@pytest.mark.parametrize("own", [False, True])
def test_a_resting_limit_decided_at_t_never_meets_a_bar_that_closed_at_or_before_t(mode, timing, names, own):
    xs, ys = names
    data = {
        xs: _bars([(100, 100, 100, 100)] * 6),
        ys: _bars([(100, 100, 100, 100)] * 3 + [(100, 110, 94, 108)] + [(108, 108, 108, 108)] * 2),
    }

    class S(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol == (ys if own else xs) and ctx.idx == 3:
                self.submit_order(O.Limit(side="buy", units=10.0, price=95.0, instrument=ys), client_id="ylim")

    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing)
    r = _run(S(), data, cfg, mode)
    # bar 3 traded down to 95 but closed before the decision; the price never comes back
    assert [(t.symbol, t.entry_price) for t in r.result.trades()] == []


# 2. a flattening halt closes as an ordinary reduce-only market order would
_HALT_ROWS = [
    (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
    (100, 100, 89, 90), (85, 86, 84, 85), (85, 85, 85, 85), (85, 85, 85, 85),
]


class _Guard(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=10.0))
        if ctx.idx > 1 and ctx.bar.close < 95:
            ctx.halt("loss_guard", True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "timing,exit_at,exit_price",
    [("same_bar_close", "09:19", 90.0), ("next_bar_open", "09:20", 85.0)],
)
def test_a_flattening_halt_exits_at_the_close_of_its_bar_or_the_next_open(mode, timing, exit_at, exit_price):
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing)
    r = _run(_Guard(), {"A": _bars(_HALT_ROWS)}, cfg, mode)
    trades = r.result.trades()
    assert len(trades) == 1
    assert fmt(trades[0].exit_time).endswith(exit_at)
    assert trades[0].exit_price == exit_price


# 3. only an order queued while the engine delivers a fill is its answer
_ANSWER_ROWS = [(100, 100, 100, 100), (100, 100, 100, 100), (100, 121, 99, 120), (130, 131, 129, 130)] + [
    (130, 131, 129, 130)
] * 2


@pytest.mark.parametrize("mode", MODES)
def test_a_strategy_cannot_mark_its_own_order_as_the_answer_to_a_fill(mode):
    class OnBar(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 2:
                order = O.Market(side="buy", units=10.0)
                object.__setattr__(order, "_follow_up", True)
                self.submit_order(order, client_id="x")

    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")
    r = _run(OnBar(), {"A": _bars(_ANSWER_ROWS)}, cfg, mode)
    # decided on bar 2's close (120): it fills at bar 3's open, never at bar 2's open
    assert [(t.entry_idx, t.entry_price) for t in r.result.trades()] == [(3, 130.0)]


def test_the_public_helper_that_marked_an_order_is_gone():
    import raptorbt.strategy.orders.types as types

    assert not hasattr(types, "answering_fill")


@pytest.mark.parametrize("mode", MODES)
def test_a_fill_hook_at_the_open_does_not_see_the_close_of_that_bar(mode):
    seen = []

    class OnFill(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 1:
                self.submit_order(O.Market(side="buy", units=1.0), client_id="seed")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled" and e.client_order_id == "seed":
                seen.append((float(ctx.bar.close), float(ctx.data.latest("A").close)))
                self.submit_order(O.Market(side="buy", units=10.0), client_id="x")

    rows = [(100, 100, 100, 100), (100, 100, 100, 100), (100, 121, 99, 120), (130, 131, 129, 130)] + [
        (130, 131, 129, 130)
    ] * 2
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")
    r = _run(OnFill(), {"A": _bars(rows)}, cfg, mode)
    # the seed fills at bar 2's open, when bar 2's close (120) is not known: the hook reads bar 1
    assert seen == [(100.0, 100.0)]
    assert r.result.trades()


# 4. no hook's command reaches a bar of the group that has not been matched
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("names", [("AX", "BY"), ("BX", "AY")])
def test_a_fill_hook_cannot_cancel_an_order_on_a_bar_of_the_group_not_yet_matched(mode, names):
    xs, ys = names
    data = {
        xs: _bars([(100, 100, 100, 100)] * 6),
        ys: _bars([(100, 100, 100, 100)] * 3 + [(100, 100, 89, 90)] + [(90, 90, 90, 90)] * 2),
    }
    seen = []

    class S(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol == xs and ctx.idx == 2:
                self.submit_order(O.Market(side="buy", units=1.0, instrument=xs), client_id="x")
                self.submit_order(O.Limit(side="buy", units=10.0, price=95.0, instrument=ys), client_id="ylim")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled" and e.client_order_id == "x":
                last = ctx.data.latest(ys)
                seen.append(float(last.close))
                if last.close < 95.0:
                    self.cancel_order("ylim")

    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")
    r = _run(S(), data, cfg, mode)
    # x fills at bar 3's open; bar 3 of Y (closing 90) is not known to the hook, and Y's order is matched first
    assert seen == [100.0]
    assert {o.client_id: o.status for o in r.result.orders()} == {"x": "filled", "ylim": "filled"}


# 5. entry_timeout is judged at the fill's instant
@pytest.mark.parametrize("paper", [False, True])
def test_entry_timeout_is_judged_at_the_open_a_fill_happened_at(paper):
    chain = OptionChain(specs())

    class OC(Strategy):
        def __init__(self):
            super().__init__()
            self.group = None

        def on_bar(self, ctx):
            if ctx.symbol != U:
                return
            if self.group is None and ctx.idx == 3:
                self.group = self.submit_structure(
                    ctx, st.bear_call_spread(short_steps=0, long_steps=1), chain=chain,
                    underlying=U, spot=100.0, exit=st.Exit(entry_timeout_ns=STEP),
                )

    data, inst = universe()
    s = OC()
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open", margin_model=MarginModel.defined_risk())
    kw = dict(instruments=inst, account_type="margin", leverage=1.0)
    if not paper:
        run_strategies([s], data, cfg, bar_stamp="close", **kw)
    else:
        syms = list(data)
        node = LiveNode([s], syms, cfg, bar_stamp="close", execution=PaperExecution(), **kw)
        for i in range(len(data[U]["timestamps"])):
            node.push_group(
                int(data[U]["timestamps"][i]),
                {x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume")) for x in syms},
            )
        node.finish()
    # both legs fill at the open of the bar after the decision: no time has passed since it
    assert s.group.reason is None
    assert s.structure_status(s.group) == "filled"


# 6. a bar the runtime took is delivered, even when a hook raised
def test_a_retry_of_a_bar_whose_handler_raised_is_refused_as_a_duplicate():
    class Boom(Strategy):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def on_bar(self, ctx):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient")

    b = Boom()
    node = LiveNode([b], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    node.push_bar("A", MIN, 1.0, 1.0, 1.0, 1.0, 1)
    with pytest.raises(RuntimeError):
        node.push_bar("A", 2 * MIN, 2.0, 2.0, 2.0, 2.0, 1)
    with pytest.raises(StateError) as e:
        node.push_bar("A", 2 * MIN, 2.0, 2.0, 2.0, 2.0, 1)
    assert e.value.code == "duplicate_bar"
    assert b.calls == 2


# 7. a group holds the timestamp it was declared at
def test_a_bar_stamped_otherwise_is_refused_inside_a_declared_group():
    class C(Strategy):
        def __init__(self):
            super().__init__()
            self.seen = []

        def on_bar(self, ctx):
            self.seen.append((ctx.symbol, ctx.idx))

    s = C()
    node = LiveNode([s], ["A", "B"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    node.push_group(MIN, {"A": (100.0,) * 4 + (1000,), "B": (50.0,) * 4 + (1000,)})
    node.begin_group(2 * MIN)
    node.push_bar("A", 2 * MIN, 101.0, 101.0, 101.0, 101.0, 1000)
    with pytest.raises(StateError) as e:
        node.push_bar("B", 3 * MIN, 51.0, 51.0, 51.0, 51.0, 1000)
    assert e.value.code == "group_timestamp_mismatch"
    node.push_bar("B", 2 * MIN, 51.0, 51.0, 51.0, 51.0, 1000)
    node.flush()
    assert sorted(s.seen[2:]) == [("A", 1), ("B", 1)]
