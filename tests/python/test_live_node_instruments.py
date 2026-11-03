"""Instruments added to, and retired from, a running LiveNode.

* an instrument added mid-run trades exactly as if it had been there from the start, in
  process and on a ``PaperExecution`` venue;
* an instrument with a position or a working order cannot be retired, and nothing changes;
* a node killed at every journal offset of a run that adds and retires instruments recovers to
  the same account;
* rings are sized by the declared history and memory comes back when an instrument is retired.
"""

import json

import numpy as np
import pytest

from raptorbt import BacktestConfig, Indicator, InstrumentSpec, LiveNode, Strategy, journal as rj
from raptorbt.errors import InvalidParameter, StateError, UnknownSymbol
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders

MIN = 60_000_000_000
CONFIG = BacktestConfig(initial_capital=1_000_000.0, fees=0.0)
HEADER = {"run": "instruments"}


class Mover(Strategy):
    """Buys the symbol of each bar on its 2nd bar and sells it on its 5th; sees every composite."""

    def __init__(self):
        super().__init__()
        self.count = {}
        self.composites = []

    def on_start(self, ctx):
        self.subscribe_bars(2, "m")

    def on_bar(self, ctx, bar):
        if type(bar).__name__ == "CompositeBar":
            self.composites.append((ctx.symbol, bar.timestamp, bar.close))
            return
        n = self.count[ctx.symbol] = self.count.get(ctx.symbol, 0) + 1
        if n == 2:
            self.submit_order(orders.Market(side="buy", units=3.0, instrument=ctx.symbol))
        elif n == 5:
            self.submit_order(
                orders.Market(side="sell", units=3.0, instrument=ctx.symbol, reduce_only=True)
            )

    def state(self):
        return {"count": self.count, "composites": [list(c) for c in self.composites]}

    def load_state(self, state):
        self.count = state["count"]
        self.composites = [tuple(c) for c in state["composites"]]


def feed(symbols, first, last):
    """Bars `first`..`last` (minutes) of every symbol, in time order."""
    out = []
    for i in range(first, last):
        for k, s in enumerate(symbols):
            p = 100.0 + 10 * k + (i % 7)
            out.append((s, (i + 1) * MIN, p, p + 1, p - 1, p + 0.5, 100.0))
    return out


def run(node, bars):
    for s, ts, o, h, low, c, v in bars:
        node.push_bar(s, ts, o, h, low, c, v)


def spec(symbol):
    return InstrumentSpec.equity(symbol)


def build(execution=None, symbols=("A",), **kw):
    mover = Mover()
    node = LiveNode(
        [mover], list(symbols), CONFIG, bar_stamp="close", history=64, execution=execution,
        instruments={s: spec(s) for s in symbols}, **kw,
    )
    return node, mover


def account(node, mover):
    ctx = node.ctx(0)
    return {
        "equity": round(node.equity(0), 6),
        "positions": {s: [(p.direction, float(p.size)) for p in ctx.positions(s)] for s in node.symbols},
        "composites": list(mover.composites),
    }


@pytest.mark.parametrize("venue", [None, "paper"])
def test_an_instrument_added_mid_run_trades_as_if_it_had_been_there(venue):
    mk = lambda: PaperExecution() if venue else None  # noqa: E731
    # The twin has B from the start, and no bars of B before the cut.
    twin, tm = build(mk(), ("A", "B"))
    run(twin, [b for b in feed(["A", "B"], 0, 20) if b[0] == "A" or b[1] > 8 * MIN])
    node, nm = build(mk())
    run(node, feed(["A"], 0, 8))
    node.add_instrument("B", spec("B"), history=16)
    assert node.symbols == ["A", "B"] and node.ctx(0).symbols == ["A", "B"]
    assert node.ctx(0).instrument("B").symbol == "B"
    run(node, [b for b in feed(["A", "B"], 8, 20)])
    # Bars of B before the cut do not exist in the twin: feed the twin the same.
    assert account(node, nm) == account(twin, tm)
    assert any(c[0] == "B" for c in nm.composites), "the subscription followed the new symbol"
    r, rt = node.finish(), twin.finish()
    assert [t.pnl for t in r[0].result.trades()] == [t.pnl for t in rt[0].result.trades()]
    assert len(r[0].result.trades()) >= 2


def test_a_pushed_bar_of_an_unknown_or_retired_symbol_is_refused_and_a_symbol_is_added_once():
    node, _ = build()
    with pytest.raises(KeyError):
        node.push_bar("B", MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
    node.add_instrument("B", spec("B"))
    with pytest.raises(InvalidParameter) as e:
        node.add_instrument("B", spec("B"))
    assert e.value.code == "duplicate_symbols"
    with pytest.raises(InvalidParameter) as e:
        node.add_instrument("C", spec("Z"))
    assert e.value.code == "invalid_instrument_spec"
    node.retire_instrument("B")
    with pytest.raises(KeyError):
        node.push_bar("B", MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(UnknownSymbol):  # already retired: it is not an active symbol
        node.retire_instrument("B")
    assert node.symbols == ["A"] and node.ctx(0).symbols == ["A"]
    node.add_instrument("B", spec("B"))  # comes back with its id
    assert node.symbols == ["A", "B"] and node.retired_instruments == ()


@pytest.mark.parametrize("venue", [None, "paper"])
def test_an_instrument_in_use_cannot_be_retired_and_nothing_changes(venue):
    node, mover = build(PaperExecution() if venue else None)
    node.add_instrument("B", spec("B"))
    run(node, feed(["A", "B"], 0, 3))  # both hold 3 units after their 2nd bar
    before = account(node, mover)
    with pytest.raises(StateError) as e:
        node.retire_instrument("B")
    assert e.value.code == "instrument_in_use"
    assert "1 open positions" in str(e.value)
    assert account(node, mover) == before and node.symbols == ["A", "B"]
    run(node, feed(["A", "B"], 3, 6))  # sold on the 5th bar
    assert not node.ctx(0).positions("B")
    node.retire_instrument("B")
    assert node.symbols == ["A"]
    run(node, feed(["A"], 6, 9))  # the rest of the node runs on
    results = node.finish()
    assert any(t.size == 3.0 for t in results[0].result.trades()) and len(results[0].result.trades()) >= 2


def test_a_working_order_also_keeps_an_instrument_in_use():
    class Rester(Strategy):
        def on_bar(self, ctx, bar):
            if ctx.symbol == "B" and not ctx.open_orders("B"):
                self.submit_order(
                    orders.Limit(side="buy", price=1.0, units=1.0, instrument="B"), client_id="rest"
                )

    node = LiveNode([Rester()], ["A"], CONFIG, bar_stamp="close", history=16)
    node.add_instrument("B", spec("B"))
    run(node, feed(["A", "B"], 0, 2))
    with pytest.raises(StateError) as e:
        node.retire_instrument("B")
    assert e.value.code == "instrument_in_use" and "1 working orders" in str(e.value)


def test_symbol_less_indicators_and_subscriptions_follow_a_new_instrument():
    class Indic(Strategy):
        def on_start(self, ctx):
            self.subscribe_bars(3, "m")
            self.fast = Indicator.sma(3)
            self.register_indicator(self.fast)  # no symbol: every symbol's bars feed it
            self.seen = []

        def on_bar(self, ctx, bar):
            if type(bar).__name__ == "CompositeBar":
                self.seen.append(ctx.symbol)

    s = Indic()
    # the symbol-less registration is the case under test, so its warning is expected
    with pytest.warns(UserWarning, match="register_indicator"):
        node = LiveNode([s], ["A"], CONFIG, bar_stamp="close", history=16)
    run(node, feed(["A"], 0, 6))
    before = s.fast.update_count
    node.add_instrument("B", spec("B"))
    run(node, feed(["A", "B"], 6, 12))
    assert s.fast.update_count == before + 12, "the indicator is fed the new symbol's bars too"
    assert s.seen.count("B") >= 1 and s.seen.count("A") >= 3, "the 3m subscription follows B too"


def test_an_indicator_with_a_symbol_stays_with_it_and_strategies_are_checked():
    class Pinned(Strategy):
        def on_start(self, ctx):
            self.sma = Indicator.sma(3)
            self.register_indicator(self.sma, symbol="A")

    p = Pinned()
    node = LiveNode([p, Mover()], ["A"], CONFIG, bar_stamp="close", history=16)
    node.add_instrument("B", spec("B"), strategies=[0])
    assert node.ctx(0).symbols == ["A", "B"] and node.ctx(1).symbols == ["A"]
    run(node, feed(["A", "B"], 0, 5))
    assert p.sma.update_count == 5, "only A's bars feed the indicator registered for A"
    with pytest.raises(InvalidParameter) as e:
        node.add_instrument("C", spec("C"), strategies=[5])
    assert e.value.code == "invalid_strategies"


def test_every_strategy_with_its_own_symbols_needs_strategies_named():
    node = LiveNode([(Mover(), ["A"])], ["A", "B"], CONFIG, bar_stamp="close", history=16)
    with pytest.raises(InvalidParameter) as e:
        node.add_instrument("C", spec("C"))
    assert e.value.code == "invalid_strategies"
    node.add_instrument("C", spec("C"), strategies=[0])
    assert node.ctx(0).symbols == ["A", "C"]


# -- memory ---------------------------------------------------------------------------


class Passive(Strategy):
    def on_bar(self, ctx, bar):
        pass


#: What a retired instrument still costs (its id, its slot and the books' kernels) is a
#: bounded few hundred bytes of the rings' accounting; the rings themselves are gone.
SLOT = 600


def test_rings_are_sized_by_the_declared_history_and_come_back_on_retire():
    node = LiveNode([Passive()], ["A"], CONFIG, bar_stamp="close", history=64)
    run(node, feed(["A"], 0, 200))
    base = node.memory_report()["live_rings"]
    node.add_instrument("B", spec("B"), history=8)
    node.add_instrument("C", spec("C"), history=1000)
    for i in range(300):
        for s in "BC":
            node.push_bar(s, (201 + i) * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
    ring = {s: len(node.ctx(0).series(s)["close"]) for s in "BC"}
    assert ring == {"B": 8, "C": 300}
    grown = node.memory_report()["live_rings"]
    assert grown > base + 300 * 40
    node.retire_instrument("C")
    assert node.memory_report()["live_rings"] < grown - 300 * 40
    node.retire_instrument("B")
    assert node.memory_report()["live_rings"] <= base + 2 * SLOT


def test_a_day_of_moving_windows_holds_memory_to_the_instruments_in_the_window():
    node = LiveNode([Passive()], ["A"], CONFIG, bar_stamp="close", history=64)
    run(node, feed(["A"], 0, 10))
    window = []
    held = []
    for step in range(60):  # a strike window that moves one strike every minute
        new = f"K{step}"
        node.add_instrument(new, spec(new), history=32)
        window.append(new)
        if len(window) > 3:
            node.retire_instrument(window.pop(0))
        for s in window:
            node.push_bar(s, (11 + step) * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
        held.append(node.memory_report()["live_rings"])
    assert node.symbols == ["A", *window]
    # Three live strikes hold their rings; each retired one leaves only its slot behind.
    growth = (held[-1] - held[19]) / (len(held) - 20)
    assert growth <= SLOT, growth


# -- the journal ------------------------------------------------------------------------


def test_the_add_and_the_retire_are_journaled_with_the_spec_and_stamped_at_the_clock():
    j = rj.Journal(HEADER, memory=True)
    node, _ = build(journal=j)
    run(node, feed(["A"], 0, 3))
    node.add_instrument("B", spec("B"), history=16)
    node.retire_instrument("B")
    recs = [r for r in j.since(0) if r.kind.startswith("instrument_")]
    assert [r.kind for r in recs] == ["instrument_add", "instrument_retire"]
    add = recs[0]
    assert add.ts_event == 3 * MIN and add.source == "host"
    assert add.payload["symbol"] == "B" and add.payload["strategies"] == [0]
    assert add.payload["history"] == 16
    assert InstrumentSpec.from_json(add.payload["spec"]).symbol == "B"


STEPS = (
    [("bar", b) for b in feed(["A"], 0, 6)]
    + [("add", "B", 16)]
    + [("bar", b) for b in feed(["A", "B"], 6, 14)]
    + [("add", "C", None)]
    + [("bar", b) for b in feed(["A", "B", "C"], 14, 22)]
    + [("retire", "B")]
    + [("bar", b) for b in feed(["A", "C"], 22, 26)]
    + [("add", "B", 8)]  # B comes back with its id and fresh rings
    + [("bar", b) for b in feed(["A", "B", "C"], 26, 32)]
)


def apply(node, step):
    if step[0] == "bar":
        s, ts, o, h, low, c, v = step[1]
        node.push_bar(s, ts, o, h, low, c, v)
    elif step[0] == "add":
        node.add_instrument(step[1], spec(step[1]), history=step[2])
    else:
        node.retire_instrument(step[1])


class Killed(Exception):
    pass


def play(kill_at=None, venue=True):
    store = []

    def sink(rec):
        if kill_at is not None and len(store) >= kill_at:
            raise Killed
        store.append(rec.canonical())

    mover = Mover()
    try:
        node = LiveNode(
            [mover], ["A"], CONFIG, bar_stamp="close", history=64,
            execution=PaperExecution() if venue else None, journal=rj.Journal(HEADER, sink=sink),
        )
        for step in STEPS:
            apply(node, step)
    except Exception as e:
        assert getattr(e, "code", None) == "journal_failed" or "Killed" in repr(e), repr(e)
        return store, None, None
    return store, node, mover


def recover(store, venue=True):
    mover = Mover()
    node = LiveNode.restore(
        None, [mover], ["A"], CONFIG, journal=rj.Journal(HEADER, memory=True), records=list(store),
        bar_stamp="close", history=64, execution=PaperExecution() if venue else None,
    )
    kinds = [json.loads(c)["kind"] for c in store]
    done = sum(k in ("bar", "instrument_add", "instrument_retire") for k in kinds)
    return node, mover, done


@pytest.mark.parametrize("venue", [True, False])
def test_killed_at_every_journal_offset_after_an_add_the_node_recovers_to_the_same_account(venue):
    full_store, full, fm = play(venue=venue)
    want = (full.state_digest(), account(full, fm), full.symbols)
    assert any(json.loads(c)["kind"] == "instrument_add" for c in full_store)
    for k in range(len(full_store)):
        store, node, _ = play(kill_at=k, venue=venue)
        assert node is None
        if not store:
            continue
        node, mover, done = recover(store, venue)
        for step in STEPS[done:]:
            apply(node, step)
        assert (node.state_digest(), account(node, mover), node.symbols) == want, f"offset {k}"


# -- a chain window that moves, a future that rolls ---------------------------------------


def _chain_universe():
    import _chain_fixture as cf

    universe = {cf.U: InstrumentSpec.equity(cf.U)}
    universe.update({s.symbol: s for s in cf.specs([cf.EXP1])})
    return cf, universe


def _chain_strategy():
    from raptorbt.strategy.requirements import Chain, Holding, Requirements, StrikeWindow, Underlying
    from raptorbt.structures.legs import Expiry

    class Windowed(Strategy):
        requirements = Requirements(
            roles={
                "index": Underlying(asset="index"),
                "opts": Chain(of="index", expiry=Expiry.nearest(), strikes=StrikeWindow.around(1)),
            },
            timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
        )

        def __init__(self):
            super().__init__()
            self.rows = []

        def on_bar(self, ctx, bar):
            if ctx.symbol == "IDX":
                self.rows.append((ctx.role("opts"), tuple(ctx.symbols)))

    return Windowed


@pytest.mark.parametrize("execution", [None, "paper"])
def test_a_chain_window_follows_the_money_across_a_day_without_rebuilding_the_node(execution):
    from raptorbt.strategy.requirements import plan_instruments

    cf, universe = _chain_universe()
    strategy = _chain_strategy()()
    node = LiveNode(
        [strategy], [cf.U], CONFIG, bar_stamp="close", history=32,
        execution=PaperExecution() if execution else None,
        instruments=universe, bindings={"index": cf.U},
    )
    held = []
    spot = [92.0 + 0.5 * i for i in range(33)]  # 92 -> 108 across the "day"
    for i, px in enumerate(spot):
        ts = cf.T0 + (i + 1) * cf.STEP
        node.push_bar(cf.U, ts, px, px + 0.05, px - 0.05, px, 1000.0)
        for sym in node.symbols[1:]:
            node.push_bar(sym, ts, 5.0, 5.1, 4.9, 5.0, 10.0)
        plan = plan_instruments(node, margin_strikes=1)
        node.apply_plan(plan)
        held.append(len(node.symbols))
    # The role named at every event after the first plan was already subscribed (a strike of margin).
    for role, symbols in strategy.rows[1:]:
        assert set(role) <= set(symbols), (role, symbols)
    first, last = strategy.rows[1][0], strategy.rows[-1][0]
    assert first != last, "the window moved"
    assert all("90" in s or "95" in s or "100" in s for s in first[:2])
    assert node.retired_instruments, "strikes left the window and were retired"
    assert max(held) <= 1 + 2 * 5, "a window of 3 strikes and a margin of 1 on each side"
    # Nothing in the plan applied twice, and a settled node plans nothing.
    assert plan_instruments(node, margin_strikes=1).empty
    node.finish()


def test_the_plan_is_pure_and_names_what_holds_a_strike_that_left_the_window():
    from raptorbt.strategy.requirements import plan_instruments

    cf, universe = _chain_universe()
    base = _chain_strategy()

    class Holder(base):
        order = None

        def on_bar(self, ctx, bar):
            super().on_bar(ctx, bar)
            if ctx.symbol == "IDX" and self.order:
                side, sym = self.order
                self.order = None
                self.submit_order(orders.Market(side=side, units=cf.LOT, instrument=sym))

    strategy = Holder()
    node = LiveNode(
        [strategy], [cf.U], CONFIG, bar_stamp="close", history=32,
        instruments=universe, bindings={"index": cf.U},
    )
    def tick(i, spot):
        ts = cf.T0 + i * cf.STEP
        node.push_bar(cf.U, ts, spot, spot + 0.05, spot - 0.05, spot, 1.0)
        for sym in node.symbols[1:]:
            node.push_bar(sym, ts, 5.0, 5.1, 4.9, 5.0, 10.0)

    tick(1, 100.0)
    plan = plan_instruments(node)
    assert set(plan.add) == {
        s for s in universe if s.startswith("IDXA") and s[4:].rstrip("CPE") in ("95", "100", "105")
    }
    assert node.symbols == [cf.U], "planning changed nothing"
    assert plan_instruments(node).to_dict() == plan.to_dict(), "and is repeatable"
    node.apply_plan(plan)
    held = "IDXA100CE"
    assert held in node.symbols
    strategy.order = ("buy", held)
    tick(2, 100.0)
    tick(3, 100.0)
    assert node.ctx(0).positions(held), "the option position is open"
    # The money moves 10 points: the window leaves the strike the strategy holds.
    tick(4, 110.0)
    far = plan_instruments(node)
    assert held in far.holding and held not in far.retire
    assert far.retire and set(far.add)
    node.apply_plan(far)
    assert held in node.symbols, "a held strike stays subscribed"
    # Closed, it is retired by the next plan.
    strategy.order = ("sell", held)
    tick(5, 110.0)
    tick(6, 110.0)
    assert not node.ctx(0).positions(held)
    done = plan_instruments(node)
    assert held in done.retire and not done.holding
    node.apply_plan(done)
    assert held not in node.symbols and held in node.retired_instruments


# -- continuous futures ----------------------------------------------------------------------


def _carry(auto_roll=True):
    import test_live_node_host as h
    return h


def test_a_continuous_future_adds_the_next_contract_at_roll_time_on_a_node_that_did_not_have_it():
    h = _carry()
    specs = h._fut_specs()
    n = 35
    data = {}
    for k, s in enumerate(("IDX", "FA", "FB")):
        ts = h.T0C + (np.arange(n, dtype=np.int64) + 1) * MIN
        c = np.full(n, 100.0 + 5 * k)
        data[s] = {"timestamps": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 100.0)}

    def play(symbols, execution=None, journal=None):
        s = h.Carry()
        node = LiveNode(
            [s], symbols, CONFIG, bar_stamp="close", execution=execution, journal=journal,
            instruments=specs, bindings={"index": "IDX"},
        )
        for i in range(n):
            for sym in ("IDX", "FA", "FB"):
                if sym in node.symbols:
                    c = {k: float(v[i]) for k, v in data[sym].items()}
                    node.push_bar(sym, int(data[sym]["timestamps"][i]), c["open"], c["high"], c["low"], c["close"], c["volume"])
        return s, node

    ref, _ = play(["IDX", "FA", "FB"])  # pre-registered, as a batch run needs
    for execution in (None, PaperExecution()):
        s, node = play(["IDX", "FA"], execution)  # FB is only a listed contract
        assert node.added_instruments == ("FB",) and "FB" in node.symbols
        assert [(r.role, r.from_symbol, r.to_symbol) for r in s.rolls] == [("fut", "FA", "FB")]
        assert ("order_filled", "fut:roll0:c0") in s.seen and ("order_filled", "fut:roll0:o0") in s.seen
        ctx = node.ctx(0)
        assert not ctx.positions("FA") and ctx.positions("FB")
    # The position moved one event later than with FB pre-registered (the node adds at its next input).
    assert ref.rolls == s.rolls


def test_the_plan_adds_the_next_contract_before_the_roll():
    from raptorbt.strategy.requirements import plan_instruments

    h = _carry()
    specs = h._fut_specs()
    s = h.Carry()
    node = LiveNode([s], ["IDX", "FA"], CONFIG, bar_stamp="close", instruments=specs, bindings={"index": "IDX"})
    price = lambda i: (100.0, 100.5, 99.5, 100.0, 100.0)  # noqa: E731
    seen = {}
    for i in range(35):
        ts = h.T0C + (i + 1) * MIN
        for sym in node.symbols:
            node.push_bar(sym, ts, 100.0, 100.5, 99.5, 100.0, 10.0)
        plan = plan_instruments(node, roll_lead_days=3 / 1440)  # three minutes before the roll
        seen[i] = plan
        node.apply_plan(plan)
    assert "FB" in seen[11].add and "FB" in seen[11].to_dict()["add"], "added three minutes ahead"
    assert "FB" not in seen[10].add
    # The roll found FB already there and moved the position with no request of its own.
    assert node.added_instruments == ("FB",) and not node.ctx(0).positions("FA")
    assert node.ctx(0).positions("FB")
    assert [(r.from_symbol, r.to_symbol) for r in s.rolls] == [("FA", "FB")]
    # FA came with the node, so the planner never retires it.
    assert all("FA" not in p.retire for p in seen.values())


def test_a_node_killed_at_every_offset_around_a_roll_that_added_a_contract_recovers_to_the_same_account():
    h = _carry()
    specs = h._fut_specs()
    n = 24
    steps = [
        (sym, h.T0C + (i + 1) * MIN)
        for i in range(n)
        for sym in ("IDX", "FA", "FB")
        if sym != "FB" or i >= 15  # FB exists only once the roll has asked for it
    ]
    kw = dict(bar_stamp="close", instruments=specs, bindings={"index": "IDX"})

    def push(node, step):
        sym, ts = step
        node.push_bar(sym, ts, 100.0, 101.0, 99.0, 100.0, 10.0)

    def fresh(sink=None, replay=None):
        s = h.Carry()
        j = rj.Journal(HEADER, sink=sink) if replay is None else rj.Journal(HEADER, memory=True)
        if replay is None:
            return s, LiveNode([s], ["IDX", "FA"], CONFIG, journal=j, execution=PaperExecution(), **kw)
        return s, LiveNode.restore(
            None, [s], ["IDX", "FA"], CONFIG, journal=j, records=list(replay),
            execution=PaperExecution(), **kw,
        )

    store = []
    s, node = fresh(sink=lambda r: store.append(r.canonical()))
    for step in steps:
        push(node, step)
    want = (node.state_digest(), node.symbols, [tuple(r) for r in s.rolls])
    assert node.symbols == ["IDX", "FA", "FB"]
    assert any(json.loads(c)["kind"] == "instrument_add" for c in store)

    for k in range(len(store)):
        killed = []

        def sink(rec, k=k, killed=killed):
            if len(killed) >= k:
                raise Killed
            killed.append(rec.canonical())

        s = h.Carry()
        try:
            crashed = LiveNode(
                [s], ["IDX", "FA"], CONFIG, journal=rj.Journal(HEADER, sink=sink),
                execution=PaperExecution(), **kw,
            )
            for step in steps:
                push(crashed, step)
        except Exception as e:
            assert getattr(e, "code", None) == "journal_failed" or "Killed" in repr(e), repr(e)
        if not killed:
            continue
        s, node = fresh(replay=killed)
        # A bar is a step; the add is the node's own, at the next input.
        done = sum(json.loads(c)["kind"] == "bar" for c in killed)
        for step in steps[done:]:
            push(node, step)
        assert (node.state_digest(), node.symbols, [tuple(r) for r in s.rolls]) == want, f"offset {k}"


# -- a retired symbol comes back ---------------------------------------------------------------


@pytest.mark.parametrize("venue", [None, "paper"])
def test_a_retired_symbol_comes_back_with_its_id_fresh_rings_and_flat_books(venue):
    mk = lambda: PaperExecution() if venue else None  # noqa: E731
    node, nm = build(mk())
    node.add_instrument("B", spec("B"), history=16)
    run(node, feed(["A", "B"], 0, 6))  # bought on the 2nd bar, sold on the 5th
    assert not node.ctx(0).positions("B")
    index = node._index_of["B"]
    node.retire_instrument("B")
    run(node, feed(["A"], 6, 9))
    node.add_instrument("B", history=4)
    assert node._index_of["B"] == index, "the same id"
    assert node.symbols == ["A", "B"] and node.retired_instruments == ()
    run(node, feed(["A", "B"], 9, 14))
    assert len(node.ctx(0).series("B")["close"]) == 4, "fresh rings of the new size"
    # B's own bar count carries on (the strategy's counter is its own), so it neither
    # re-entered nor kept a position, and the node's account is consistent throughout.
    assert not node.ctx(0).positions("B")
    results = node.finish()
    assert len(results[0].result.trades()) == 2
    # The spec it came back with is the one it had.
    n2, _ = build()
    n2.add_instrument("C", spec("C"))
    n2.retire_instrument("C")
    with pytest.raises(InvalidParameter) as e:
        n2.add_instrument("C", InstrumentSpec.option("C", 100.0, "call", 10**18, 50.0))
    assert e.value.code == "invalid_instrument_spec"


def test_a_chain_window_that_wanders_back_over_a_strike_revives_it_and_keeps_going():
    from raptorbt.strategy.requirements import plan_instruments

    cf, universe = _chain_universe()
    strategy = _chain_strategy()()
    node = LiveNode(
        [strategy], [cf.U], CONFIG, bar_stamp="close", history=32,
        instruments=universe, bindings={"index": cf.U},
    )
    path = [100.0, 100.0, 107.0, 107.0, 100.0, 100.0, 107.0, 100.0]  # up, back, up, back
    revived = []
    for i, px in enumerate(path):
        ts = cf.T0 + (i + 1) * cf.STEP
        node.push_bar(cf.U, ts, px, px + 0.05, px - 0.05, px, 1000.0)
        for sym in node.symbols[1:]:
            node.push_bar(sym, ts, 5.0, 5.1, 4.9, 5.0, 10.0)
        before = set(node.retired_instruments)
        plan = plan_instruments(node)  # no margin: the window is retired and revived as it moves
        assert not plan.issues, plan.issues
        revived += [s for s in plan.add if s in before]
        node.apply_plan(plan)
    assert revived, "a strike left the window and came back"
    assert set(node.retired_instruments) | set(node.symbols) >= set(revived)
    for sym in revived:
        assert sym in node.symbols or sym in node.retired_instruments
    node.finish()


# -- the venue hears about the instrument set ------------------------------------------------------


class Venue:
    """The smallest venue of one's own, with the optional instrument capability."""

    namespace = "e3"

    def __init__(self, refuse=()):
        self.calls, self.refuse = [], set(refuse)
        self.sent = []

    def submit(self, request):
        self.sent.append(request.symbol)

    def cancel(self, client_order_id):
        pass

    def modify(self, client_order_id, **kw):
        pass

    def add_instrument(self, symbol, spec):
        if symbol in self.refuse:
            from raptorbt.execution import Unsupported

            raise Unsupported(f"{symbol} is not tradable here")
        self.calls.append(("add", symbol, getattr(spec, "symbol", None)))

    def retire_instrument(self, symbol):
        self.calls.append(("retire", symbol))


def test_an_adapter_is_told_before_any_order_can_reach_it_and_may_refuse():
    venue = Venue(refuse={"BAD"})
    node = LiveNode([Passive()], ["A"], CONFIG, bar_stamp="close", history=8, execution=venue)
    node.add_instrument("B", spec("B"))
    with pytest.raises(Exception) as e:
        node.add_instrument("BAD", spec("BAD"))
    assert "not tradable here" in str(e.value)
    assert node.symbols == ["A", "B"], "a refusal leaves the node as it was"
    node.retire_instrument("B")
    node.add_instrument("B", spec("B"))  # a revive tells it again
    assert venue.calls == [("add", "B", "B"), ("retire", "B"), ("add", "B", "B")]


def test_a_restored_node_tells_its_new_adapter_about_every_instrument_it_added():
    store = []
    venue = Venue()
    node = LiveNode(
        [Passive()], ["A"], CONFIG, bar_stamp="close", history=8, execution=venue,
        journal=rj.Journal(HEADER, sink=lambda r: store.append(r.canonical())),
    )
    node.add_instrument("B", spec("B"))
    node.add_instrument("C", spec("C"))
    fresh = Venue()
    LiveNode.restore(
        None, [Passive()], ["A"], CONFIG, journal=rj.Journal(HEADER, memory=True), records=list(store),
        bar_stamp="close", history=8, execution=fresh, reconcile="skip",
    )
    assert fresh.calls == [("add", "B", "B"), ("add", "C", "C")]


def test_the_broker_adapter_learns_a_contract_from_its_resolver_and_refuses_one_it_cannot_name():
    from raptorbt.adapters import Instrument
    from raptorbt.adapters.conformance import ScriptedVenue, _reference_adapter

    adapter = _reference_adapter(ScriptedVenue(), {})
    adapter._instruments = {"A": Instrument("NSE", "A")}
    with pytest.raises(Exception) as e:
        adapter.add_instrument("B", spec("B"))
    assert "no instrument is mapped" in str(e.value)
    adapter.resolve_instrument = lambda symbol, sp: Instrument("NFO", symbol, key="123")
    adapter.add_instrument("B", spec("B"))
    assert adapter.instrument("B").key == "123"
    assert adapter.symbol_of("123") == "B"
    adapter.retire_instrument("B")
    with pytest.raises(Exception):
        adapter.instrument("B")


# -- strategies hear about it ---------------------------------------------------------------------


def test_strategies_hear_when_an_instrument_is_added_revived_and_retired():
    class Listener(Strategy):
        def __init__(self):
            super().__init__()
            self.heard = []

        def on_instrument_added(self, ctx, symbol):
            self.heard.append(("added", symbol, tuple(ctx.symbols)))

        def on_instrument_retired(self, ctx, symbol):
            self.heard.append(("retired", symbol, tuple(ctx.symbols)))

    a, b = Listener(), Listener()
    node = LiveNode([a, (b, ["A"])], ["A"], CONFIG, bar_stamp="close", history=8)
    node.add_instrument("B", spec("B"))  # only the strategy given no symbols of its own
    node.retire_instrument("B")
    node.add_instrument("B", spec("B"))
    assert a.heard == [
        ("added", "B", ("A", "B")), ("retired", "B", ("A",)), ("added", "B", ("A", "B")),
    ]
    assert b.heard == []


def test_the_hooks_are_listed_in_the_capabilities():
    import raptorbt

    names = {h["name"] for h in raptorbt.capabilities()["hooks"]}
    assert {"on_instrument_added", "on_instrument_retired"} <= names


# -- the cost of moving instruments ----------------------------------------------------------------


def test_a_session_of_hundreds_of_adds_and_retires_leaves_the_event_loop_as_it_found_it():
    import time

    clock = iter(range(1000, 10**9))

    def cost(node, bars=300):
        t = time.process_time()
        for _ in range(bars):
            node.push_bar("A", next(clock) * MIN, 1.0, 1.0, 1.0, 1.0, 1.0)
        return time.process_time() - t

    fresh = LiveNode([Passive()], ["A"], CONFIG, bar_stamp="close", history=8)
    worn = LiveNode([Passive()], ["A"], CONFIG, bar_stamp="close", history=8)
    for i in range(500):
        sym = f"K{i}"
        worn.add_instrument(sym, spec(sym), history=4)
        worn.retire_instrument(sym)
    # Retired kernels are skipped by the per-event loops: an event costs what a fresh node's does
    # (loose bound; the bench measures it).
    best_fresh = min(cost(fresh) for _ in range(3))
    best_worn = min(cost(worn) for _ in range(3))
    assert best_worn < best_fresh * 2 + 0.01, (best_fresh, best_worn)
