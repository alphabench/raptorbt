"""The equal-timestamp two-phase rule.

Events that share a timestamp and a kind are one group: every event is
applied before any is decided, and every decision is taken before any is
matched (docs/architecture/event-semantics.md). A multi-instrument result
therefore does not depend on the order the data mapping lists its symbols in.

* permuting the mapping gives the same trades, orders, fees and equity, for a
  strategy that reads another instrument's bar and for one that sends orders
  across instruments, under both fill timings;
* a ``LiveNode`` that delivers the same groups reproduces the batch run;
* the journal of a grouped run replays into a node that writes the same
  records, and a journal with no group records replays push by push.
"""

import itertools

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders

SYMBOLS = ["A", "B", "C"]
T0 = 1_700_000_000 * 10**9
MIN = 60 * 10**9
N = 40


def _bars(seed, n=N):
    rng = np.random.default_rng(seed)
    ts = T0 + (np.arange(n, dtype=np.int64) + 1) * MIN
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = np.r_[close[0], close[:-1]]
    return {
        "timestamps": ts,
        "open": open_,
        "high": np.maximum(open_, close) + 0.5,
        "low": np.minimum(open_, close) - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
    }


DATA = {s: _bars(i + 1) for i, s in enumerate(SYMBOLS)}


class Reader(Strategy):
    """Enters on A when another instrument's visible close is above A's."""

    def on_bar(self, ctx):
        if ctx.symbol != "A":
            return
        other = ctx.series("B")["close"]
        mine = ctx.series("A")["close"]
        if not len(other):
            return
        if float(other[-1]) > float(mine[-1]) and ctx.position is None:
            self.submit_order(orders.Market(side="buy", units=5.0))
        elif float(other[-1]) < float(mine[-1]) and ctx.position is not None:
            self.close_position()


class Sizer(Strategy):
    """Holds A, and sizes B by the equity it sees, which marks A at the
    group's price or the one before it."""

    def on_bar(self, ctx):
        if ctx.symbol == "A" and ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", units=40.0))
        if ctx.symbol == "B" and ctx.idx % 5 == 2 and ctx.position is None:
            self.submit_order(orders.Market(side="buy", units=float(int(ctx.equity * 100) % 7 + 1)))
        if ctx.symbol == "B" and ctx.idx % 5 == 4 and ctx.position is not None:
            self.close_position()


class Cross(Strategy):
    """Decides on A and sends the order to B (and to C)."""

    def on_bar(self, ctx):
        if ctx.symbol != "A":
            return
        if ctx.idx % 7 == 3:
            self.submit_order(orders.Market(side="buy", units=3.0), symbol="B")
            self.submit_order(orders.Market(side="buy", units=2.0), symbol="C")
        if ctx.idx % 7 == 6:
            self.close_position(symbol="B")
            self.close_position(symbol="C")


STRATEGIES = [Reader, Sizer, Cross]
TIMINGS = ["same_bar_close", "next_bar_open"]


def _config(timing):
    return BacktestConfig(fill_timing=timing, fees=0.001, slippage=0.0, initial_capital=100_000.0)


def _digest(result):
    """What a run did, keyed by symbol and bar, not by order id."""
    r = result.result
    trades = sorted(
        (t.symbol, t.entry_idx, t.exit_idx, round(t.entry_price, 6), round(t.exit_price, 6),
         round(t.size, 6), round(t.pnl, 6), round(t.fees, 6))
        for t in r.trades()
    )
    placed = sorted(
        (o.symbol, o.submitted_idx, o.status, o.avg_fill_price and round(o.avg_fill_price, 6),
         o.last_fill_idx)
        for o in r.orders()
    )
    curve = [round(float(x), 6) for x in r.equity_curve()]
    return trades, placed, curve, round(r.metrics.total_return_pct, 6)


def _batch(strategy, order, timing):
    data = {s: DATA[s] for s in order}
    return _digest(raptorbt.run_strategies([strategy()], data, _config(timing), bar_stamp="close")[0])


@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.__name__)
def test_the_mapping_order_does_not_change_the_result(strategy, timing):
    runs = [_batch(strategy, order, timing) for order in itertools.permutations(SYMBOLS)]
    assert runs[0][0], "the scenario must trade"
    assert all(run == runs[0] for run in runs)


@pytest.mark.parametrize("timing", TIMINGS)
def test_a_cross_instrument_order_meets_the_targets_bar_of_the_decision(timing):
    trades = _batch(Cross, SYMBOLS, timing)[0]
    first_b = next(t for t in trades if t[0] == "B")
    # Decided on A's bar 3: the close of B's bar 3, or the open of its bar 4.
    if timing == "same_bar_close":
        assert first_b[1] == 3
        assert first_b[3] == round(float(DATA["B"]["close"][3]), 6)
    else:
        assert first_b[1] == 4
        assert first_b[3] == round(float(DATA["B"]["open"][4]), 6)


# -- a live node that delivers the same groups --------------------------------------


def _feed(node, order, grouped=True):
    for i in range(N):
        ts = int(DATA["A"]["timestamps"][i])
        bars = {
            s: tuple(float(DATA[s][k][i]) for k in ("open", "high", "low", "close", "volume"))
            for s in order
        }
        if grouped:
            node.push_group(ts, bars)
        else:
            for s, b in bars.items():
                node.push_bar(s, ts, *b)


@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.__name__)
def test_a_node_delivering_the_groups_reproduces_the_batch_run(strategy, timing):
    want = _batch(strategy, SYMBOLS, timing)
    for order in (SYMBOLS, SYMBOLS[::-1]):
        node = LiveNode([strategy()], SYMBOLS, _config(timing), bar_stamp="close", history=N + 1)
        _feed(node, order)
        assert _digest(node.finish()[0]) == want


@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("target", ["B", "ZB"])
def test_a_paper_node_delivering_the_groups_meets_a_cross_instrument_order_on_the_batch_bar(
    timing, target
):
    """Whether the target is stepped before or after the deciding instrument (its name sorts
    ahead of the decider's, "M", or behind it), the order meets the target's bar of the decision, as in batch."""

    class Decides(Strategy):
        def __init__(self):
            super().__init__()
            self.fills = []

        def on_bar(self, ctx):
            if ctx.symbol == "M" and ctx.idx == 5:
                self.submit_order(orders.Market(side="buy", units=3.0), symbol=names[1])

        def on_order_event(self, ctx, event):
            if event.kind == "order_filled":
                self.fills.append((event.price, ctx.event_ns))

    names = ["M", target]
    data = {"M": DATA["A"], target: DATA["B"]}
    batch = Decides()
    raptorbt.run_strategies([batch], data, _config(timing), bar_stamp="close")
    assert batch.fills, "the scenario must trade"
    for order in (names, names[::-1]):
        paper_strategy = Decides()
        node = LiveNode([paper_strategy], names, _config(timing), bar_stamp="close",
                        history=N + 1, execution=PaperExecution())
        for i in range(N):
            ts = int(DATA["A"]["timestamps"][i])
            node.push_group(ts, {s: tuple(float(data[s][k][i]) for k in
                                          ("open", "high", "low", "close", "volume")) for s in order})
        node.finish()
        assert paper_strategy.fills == batch.fills, (timing, order)


def test_begin_group_and_flush_are_push_group():
    want = _batch(Reader, SYMBOLS, "same_bar_close")
    node = LiveNode([Reader()], SYMBOLS, _config("same_bar_close"), bar_stamp="close", history=N + 1)
    node.begin_group()
    _feed(node, SYMBOLS[::-1], grouped=False)
    node.flush()
    assert _digest(node.finish()[0]) == want


def test_finish_delivers_an_open_group():
    want = _batch(Reader, SYMBOLS, "same_bar_close")
    node = LiveNode([Reader()], SYMBOLS, _config("same_bar_close"), bar_stamp="close", history=N + 1)
    node.begin_group()
    _feed(node, SYMBOLS, grouped=False)
    assert _digest(node.finish()[0]) == want


def test_a_group_cannot_be_opened_twice_nor_checkpointed_open():
    node = LiveNode([Reader()], SYMBOLS, _config("same_bar_close"), bar_stamp="close",
                    journal=rj.Journal({"run_id": "g"}, memory=True))
    node.begin_group()
    with pytest.raises(raptorbt.errors.StateError) as e:
        node.begin_group()
    assert e.value.code == "group_open"
    with pytest.raises(raptorbt.errors.StateError) as e:
        node.checkpoint()
    assert e.value.code == "group_open"
    node.flush()
    node.checkpoint()


# -- the journal -------------------------------------------------------------------


def _journaled(grouped):
    j = rj.Journal({"run_id": "g"}, memory=True)
    node = LiveNode([Cross()], SYMBOLS, _config("same_bar_close"), bar_stamp="close",
                    history=N + 1, journal=j)
    _feed(node, SYMBOLS[::-1], grouped=grouped)
    return node, j


def test_a_grouped_run_writes_its_boundaries():
    node, j = _journaled(grouped=True)
    kinds = [r.kind for r in j.since(0)]
    assert kinds.count("group_begin") == N
    assert kinds.count("group_end") == N
    first = kinds.index("group_begin")
    assert kinds[first : first + 5] == ["group_begin", "bar", "bar", "bar", "group_end"]
    node.finish()


@pytest.mark.parametrize("grouped", [True, False])
def test_a_journal_replays_into_a_node_that_writes_the_same_records(grouped):
    node, j = _journaled(grouped)
    want_digest = node.state_digest()
    want = [r.canonical() for r in j.since(0)]
    j2 = rj.Journal({"run_id": "g"}, memory=True)
    restored = LiveNode.restore(
        None, [Cross()], SYMBOLS, _config("same_bar_close"), journal=j2, records=list(want),
        bar_stamp="close", history=N + 1,
    )
    assert [r.canonical() for r in j2.since(0)] == want
    assert restored.state_digest() == want_digest


def test_grouping_changes_what_a_decision_sees():
    """The same pushes, one at a time, are the old order-dependent run."""
    grouped = LiveNode([Reader()], SYMBOLS, _config("same_bar_close"), bar_stamp="close", history=N + 1)
    _feed(grouped, SYMBOLS)
    alone = LiveNode([Reader()], SYMBOLS, _config("same_bar_close"), bar_stamp="close", history=N + 1)
    _feed(alone, SYMBOLS, grouped=False)
    assert _digest(grouped.finish()[0]) == _batch(Reader, SYMBOLS, "same_bar_close")
    assert _digest(alone.finish()[0]) != _batch(Reader, SYMBOLS, "same_bar_close")


# -- a strategy's view covers its own subscriptions ---------------------------------


class Watch(Strategy):
    """Records what it sees at each decision: equity, and its symbols' last closes."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx):
        if ctx.idx == 1 and ctx.symbol == ctx.symbols[0]:
            self.submit_order(orders.Market(side="buy", units=10.0))
        self.seen.append((ctx.symbol, ctx.idx, round(ctx.equity, 6), round(ctx.cash, 6)))


@pytest.mark.parametrize("subscribed", [["A"], ["B"], ["A", "B"], ["A", "C"]])
def test_a_strategy_in_a_shared_run_sees_what_it_sees_alone(subscribed):
    cfg = _config("same_bar_close")
    shared = Watch()
    raptorbt.run_strategies([(shared, subscribed), Watch()], DATA, cfg, bar_stamp="close")
    alone = Watch()
    raptorbt.run_strategies([alone], {s: DATA[s] for s in subscribed}, cfg, bar_stamp="close")
    assert shared.seen == alone.seen


# -- orders of one event named by another of the group ------------------------------


class Names(Strategy):
    """A creates an order and C cancels it; C creates one and A cancels it."""

    def on_bar(self, ctx):
        if ctx.idx != 3:
            return
        far = float(ctx.series(ctx.symbol)["close"][-1]) * 0.5
        mine = {"A": "from-a", "C": "from-c"}.get(ctx.symbol)
        if mine:
            self.submit_order(orders.Limit(side="buy", price=far, units=1.0), client_id=mine)
        theirs = {"A": "from-c", "C": "from-a"}.get(ctx.symbol)
        if theirs:
            self.cancel_order(theirs)


@pytest.mark.parametrize("order", list(itertools.permutations(SYMBOLS)))
def test_a_cancel_finds_an_order_another_event_of_the_group_created(order):
    s = Names()
    res = raptorbt.run_strategies([s], {k: DATA[k] for k in order}, _config("same_bar_close"),
                                  bar_stamp="close")[0]
    status = {o.client_id: o.status for o in res.result.orders()}
    assert status == {"from-a": "canceled", "from-c": "canceled"}


# -- an open group refuses what needs the node between pushes -----------------------


def test_reconcile_resume_and_health_refuse_an_open_group():
    node = LiveNode([Reader()], SYMBOLS, _config("same_bar_close"), bar_stamp="close")
    node.begin_group()
    for call in (node.reconcile, node.resume, node.health):
        with pytest.raises(raptorbt.errors.StateError) as e:
            call()
        assert e.value.code == "group_open"
    node.flush()
    node.health()


# -- a contract expires on its own bar of the group ---------------------------------


class Holds(Strategy):
    def on_bar(self, ctx):
        if ctx.symbol == "AAA" and ctx.idx == 2:
            self.submit_order(orders.Market(side="buy", units=1.0), symbol="FUT")


@pytest.mark.parametrize("order", [("AAA", "FUT"), ("FUT", "AAA")])
def test_a_contract_settles_on_its_own_bar_whichever_event_notices_the_expiry(order):
    expiry = T0 + 10 * MIN
    bars = {k: _bars(0 if k == "AAA" else 10, n=20) for k in order}
    specs = {"FUT": raptorbt.InstrumentSpec.futures_contract("FUT", expiry, 1.0)}
    r = raptorbt.run_strategies([Holds()], bars, BacktestConfig(initial_capital=1e6, fees=0.0),
                                bar_stamp="close", instruments=specs)[0]
    [t] = r.result.trades()
    # the bar stamped at the expiry is FUT's own bar of the group that holds AAA's
    assert (t.entry_idx, t.exit_idx, t.exit_reason) == (2, 9, "Settlement")
    assert t.exit_price == pytest.approx(float(bars["FUT"]["close"][9]))
