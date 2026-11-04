"""LiveNode: many strategies on one pushed feed.

The frozen single-strategy behaviour is pinned in ``test_runner_fixtures.py``
(the ``st_*`` scenarios, run on ``TickStrategyStream`` and on a one-strategy
``LiveNode``). These tests cover what only a node has: several strategies on
one push, equality with the batch runtime, time without data, per-strategy
seeds, the end of the run, and storage that stays bounded.
"""

import resource
import subprocess
import sys
import os
import json
from pathlib import Path

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, LiveNode, Strategy, TickStrategyStream, run_strategies
from raptorbt.strategy import orders

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "golden"))

import runner_fixtures as rf  # noqa: E402

FIXTURES = rf.load()
DATASETS = FIXTURES["datasets"]
SCENARIOS = {sc["name"]: sc for sc in rf.scenario_table()}
S0, S1, S2 = rf.S0, rf.S1, rf.S2
ALL = [S0, S1, S2]
NS = rf.NS


def _config():
    return rf.build_config(SCENARIOS["st_push_basic"]["config"])


def _recorder(script_spec_or_scenario):
    """A fresh Recorder; a scenario name takes that scenario's script."""
    sc = SCENARIOS[script_spec_or_scenario]
    rec = rf.Recorder(script=rf._script_for(sc, DATASETS))
    rec.tick_surface = True
    return rec


# -- one push feeds many strategies, each as its own stream would be ----------

# (scenario whose script to use, symbols the strategy trades)
MIX = [
    ("st_push_basic", ALL),
    ("st_refused_rows_timers", ALL),
    ("st_queue_depth_latency", ALL),
    ("st_push_basic", [S1, S2]),
    ("st_refused_rows_timers", [S0]),
]


def _push(target, op):
    if op[0] == "tick":
        return target.push_tick(op[1], op[2], *op[3:])
    if op[0] == "depth":
        return target.push_depth(op[1], op[2], op[3], op[4])
    return target.push_bar(op[1], op[2], *op[3:])


def _held(positions):
    return {s: [rf.enc_position(p) for p in ps] for s, ps in positions.items() if ps}


def test_each_strategy_equals_its_own_stream():
    ops = rf.thaw_ops(DATASETS["stream_ops"])[:56]
    node_recs = [_recorder(name) for name, _ in MIX]
    node = LiveNode(
        [(rec, syms) for rec, (_, syms) in zip(node_recs, MIX)], ALL, _config()
    )
    streams = [
        TickStrategyStream(_recorder(name), syms, config=_config()) for name, syms in MIX
    ]

    for op in ops:
        returned = _push(node, op)
        for k, stream in enumerate(streams):
            if op[1] not in MIX[k][1]:
                continue
            assert _push(stream, op) == (returned if op[0] == "tick" else None)
            for symbol in MIX[k][1]:
                got = [rf.enc_position(p) for p in node.positions(k, symbol)]
                want = [rf.enc_position(p) for p in stream.positions(symbol)]
                assert got == want, (k, op[:3], symbol)
            assert rf.enc(node.equity(k)) == rf.enc(stream.equity)
            assert node.is_halted(k) == stream.is_halted

    results = node.finish()
    stream_results = [stream.finish() for stream in streams]
    assert len(results) == len(MIX)
    for k, (stream, result) in enumerate(zip(streams, results)):
        assert node_recs[k].hooks == stream.strategy.hooks, f"strategy {k}: hook log differs"
        assert len(node_recs[k].hooks) > 20, "the strategy never ran"
        assert rf.digest_result(result) == rf.digest_result(stream_results[k]), f"strategy {k}"


def test_warmup_bars_and_pushed_bars_equal_the_streams():
    sc = SCENARIOS["st_warmup_executes"]
    kw = rf._run_kwargs(sc)
    warm = {}
    for s in ALL:
        a = rf._thaw_arrays(DATASETS["warmup"][s])
        warm[s] = {"timestamps": a["ts_close"], **{k: a[k] for k in ("open", "high", "low", "close", "volume")}}
    bar_data = {s: rf._thaw_arrays(DATASETS["bars"][s]) for s in ALL}
    ops = rf._stream_ops(sc, DATASETS, bar_data, "close")
    subsets = [ALL, [S0, S2]]

    def make_rec():
        rec = rf.Recorder(script=rf._script_for(sc, DATASETS))
        rec.tick_surface = True
        return rec

    node_recs = [make_rec() for _ in subsets]
    node = LiveNode(
        list(zip(node_recs, subsets)), ALL, _config(), warmup_bars=warm, **kw
    )
    streams = [
        TickStrategyStream(make_rec(), syms, config=_config(), warmup_bars={s: warm[s] for s in syms}, **kw)
        for syms in subsets
    ]
    for op in ops:
        _push(node, op)
        for stream, syms in zip(streams, subsets):
            if op[1] in syms:
                _push(stream, op)
    results = node.finish()
    stream_results = [stream.finish() for stream in streams]
    for k, (stream, result) in enumerate(zip(streams, results)):
        assert node_recs[k].hooks == stream.strategy.hooks, f"strategy {k}: hook log differs"
        assert rf.digest_result(result) == rf.digest_result(stream_results[k])


def test_a_push_reaches_every_subscriber_in_registration_order():
    seen = []

    class Tag(Strategy):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def on_trade(self, ctx, tick):
            seen.append((self.tag, ctx.symbol))

    node = LiveNode(
        [(Tag("a"), ["X", "Y"]), (Tag("b"), ["Y"]), (Tag("c"), ["X", "Y"]), (Tag("d"), ["X"])],
        ["X", "Y"],
    )
    node.push_tick("Y", 1, 10.0)
    node.push_tick("X", 2, 10.0)
    assert seen == [("a", "Y"), ("b", "Y"), ("c", "Y"), ("a", "X"), ("c", "X"), ("d", "X")]


def test_strategies_have_isolated_accounts():
    class AllIn(Strategy):
        def __init__(self, frac):
            super().__init__()
            self.frac = frac

        def on_trade(self, ctx, tick):
            if ctx.is_flat:
                self.enter(size_frac=self.frac)

    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    node = LiveNode([AllIn(0.9), AllIn(0.1)], ["X"], cfg)
    for i, px in enumerate([10.0, 10.0, 11.0, 12.0]):
        node.push_tick("X", i + 1, px)
    big, small = node.equity(0), node.equity(1)
    assert big > 100_000.0 and small > 100_000.0
    assert (big - 100_000.0) > 5 * (small - 100_000.0)
    assert node.positions(0)["X"][0].size != node.positions(1)["X"][0].size
    assert node.positions(0, "X")[0].size == pytest.approx(9 * node.positions(1, "X")[0].size, rel=0.2)


# -- push == batch -----------------------------------------------------------


def _feed(n=240, symbols=("A", "B", "C")):
    """Distinct timestamps across symbols, so the tape order is the push order.

    Returns the per-symbol tick arrays, depth arrays and the pushes in time
    order (trade, then its quote, then depth at a later instant).
    """
    rng = np.random.default_rng(7)
    ticks, depth, pushes = {}, {}, []
    for si, symbol in enumerate(symbols):
        px = 100.0 + 10 * si + np.cumsum(rng.normal(0, 0.05, n))
        ts = 1_700_000_000 * NS + (np.arange(n) * len(symbols) + si) * 1_000_000
        qty = rng.integers(1, 50, n).astype(float)
        ticks[symbol] = {
            "timestamps": ts,
            "ltp": px,
            "bid": px - 0.05,
            "ask": px + 0.05,
            "buy_qty_delta": qty,
            "sell_qty_delta": np.zeros(n),
            "ltq": qty,
            "bid_qty": qty + 5,
            "ask_qty": qty + 7,
            "oi": np.zeros(n),
        }
        dk = np.arange(3, n, 9)
        dts = ts[dk] + 400_000
        depth[symbol] = {
            "timestamps": dts,
            "bid_prices": np.stack([px[dk] - 0.05 - 0.05 * i for i in range(3)], axis=1),
            "bid_sizes": np.stack([qty[dk] + 10 * i for i in range(3)], axis=1),
            "ask_prices": np.stack([px[dk] + 0.05 + 0.05 * i for i in range(3)], axis=1),
            "ask_sizes": np.stack([qty[dk] + 10 * i for i in range(3)], axis=1),
        }
        for k in range(n):
            pushes.append((int(ts[k]), 0, si, k, ("tick", symbol, int(ts[k]), *[float(ticks[symbol][c][k]) for c in rf.TICK_KEYS[1:]])))
        for j in range(len(dk)):
            d = depth[symbol]
            bids = [(float(d["bid_prices"][j][i]), float(d["bid_sizes"][j][i])) for i in range(3)]
            asks = [(float(d["ask_prices"][j][i]), float(d["ask_sizes"][j][i])) for i in range(3)]
            pushes.append((int(dts[j]), 1, si, j, ("depth", symbol, int(dts[j]), bids, asks)))
    pushes.sort(key=lambda p: p[:4])
    return ticks, depth, [p[4] for p in pushes]


def test_push_equals_batch_hook_by_hook():
    ticks, depth, pushes = _feed()
    sym_sets = [["A", "B", "C"], ["B", "C"], ["A"]]
    scripts = [
        {"name": "legacy", "args": {"drive": "on_trade_tick", "every": 11, "enter_at": 3, "exit_at": 8,
                                    "frac": 0.2, "extra": [["on_quote", 7, 2], ["on_order_book", 3, 1]],
                                    "after_close": "both"}},
        {"name": "hedge", "args": {"drive": "on_trade_tick", "every": 8}},
        {"name": "legacy", "args": {"drive": "on_trade_tick", "every": 5, "enter_at": 1, "exit_at": 3, "frac": 0.1,
                                    "subs": [[1, "s"]], "inds": [["sma", 4]]}},
    ]
    cfg = lambda: BacktestConfig(initial_capital=1_000_000.0, fees=0.0)  # noqa: E731

    def recs():
        out = []
        for spec in scripts:
            rec = rf.Recorder(script=rf.make_script(spec))
            rec.tick_surface = True
            out.append(rec)
        return out

    live = recs()
    node = LiveNode(list(zip(live, sym_sets)), ["A", "B", "C"], cfg(), bar_stamp="close")
    for op in pushes:
        _push(node, op)
    live_results = node.finish()

    batch = recs()
    batch_results = run_strategies(
        list(zip(batch, sym_sets)), ticks=ticks, depth=depth, config=cfg(), bar_stamp="close"
    )
    for k in range(len(scripts)):
        assert len(live[k].hooks) > 100, f"strategy {k} barely ran"
        assert live[k].hooks == batch[k].hooks, f"strategy {k}: hook log differs"
        assert rf.digest_result(live_results[k]) == rf.digest_result(batch_results[k])


# -- time without data ---------------------------------------------------------


class Timed(Strategy):
    def __init__(self, interval=10 * NS, buy_on=None):
        super().__init__()
        self.interval, self.buy_on = interval, buy_on
        self.fired = []
        self.ticks = 0

    def on_start(self, ctx):
        self.clock.set_timer("t", self.interval, start_ns=self.interval)

    def on_timer(self, ctx, event):
        self.fired.append((ctx.symbol, event.ts_fired))
        if self.buy_on == event.ts_fired:
            self.submit_order(orders.Market(side="buy", size_frac=0.5))

    def on_trade(self, ctx, tick):
        self.ticks += 1


def test_advance_time_fires_due_timers_for_every_strategy_without_data():
    a, b = Timed(), Timed(interval=25 * NS)
    node = LiveNode([(a, ["X"]), (b, ["X", "Y"])], ["X", "Y"])
    node.advance_time(5 * NS)
    assert (a.fired, b.fired) == ([], [])
    node.advance_time(12 * NS)
    assert a.fired == [("X", 12 * NS)]
    assert b.fired == []
    node.advance_time(30 * NS)
    # One clock per symbol: b trades two symbols, so its timer fires on each.
    assert a.fired == [("X", 12 * NS), ("X", 30 * NS)]
    assert b.fired == [("X", 30 * NS), ("Y", 30 * NS)]
    assert (a.ticks, b.ticks) == (0, 0), "no data event was involved"
    assert node.equity(0) == node.equity(1) == BacktestConfig().initial_capital


def test_an_order_from_advance_time_matches_from_the_next_print():
    s = Timed(interval=10 * NS, buy_on=10 * NS)
    node = LiveNode([s], ["X"], BacktestConfig(initial_capital=100_000.0, fees=0.0))
    node.push_tick("X", 1 * NS, 50.0)
    node.advance_time(10 * NS)
    assert node.positions(0, "X") == [], "an order cannot fill on a moment with no print"
    node.push_tick("X", 11 * NS, 50.0)
    (position,) = node.positions(0, "X")
    assert position.size == pytest.approx(1000.0)


def test_advance_time_equals_a_refused_row_on_a_stream():
    a, b = Timed(), Timed()
    node = LiveNode([a], ["X"])
    stream = TickStrategyStream(b, ["X"])
    for target in (node, stream):
        target.push_tick("X", 1 * NS, 10.0)
    node.advance_time(15 * NS)
    stream.push_tick("X", 15 * NS, 0.0)
    assert a.fired == b.fired == [("X", 15 * NS)]


# -- per-strategy seeds ----------------------------------------------------------


class Reads(Strategy):
    """Records what on_start could see."""

    def on_start(self, ctx):
        self.at_start = {s: [(p.size, p.entry_price) for p in ctx.positions(s)] for s in ctx.symbols}


def test_initial_positions_are_per_strategy_by_index_or_by_object():
    a, b, c = Reads(), Reads(), Reads()
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    node = LiveNode(
        [a, (b, ["X"]), c],
        ["X", "Y"],
        cfg,
        initial_positions={
            0: {"X": {"quantity": 10.0, "avg_price": 100.0}},
            b: {"X": {"quantity": 30.0, "avg_price": 90.0, "timestamp_ns": 5}},
        },
    )
    assert a.at_start == {"X": [(10.0, 100.0)], "Y": []}
    assert b.at_start == {"X": [(30.0, 90.0)]}
    assert c.at_start == {"X": [], "Y": []}
    assert [p.size for p in node.positions(0, "X")] == [10.0]
    assert [p.size for p in node.positions(1, "X")] == [30.0]
    assert node.positions(2) == {"X": [], "Y": []}
    # Cash left after the cost basis, per account; no fees, no trade record.
    node.push_tick("X", 1, 100.0)
    assert node.equity(0) == pytest.approx(100_000.0)
    assert node.equity(1) == pytest.approx(100_000.0 - 30 * 90 + 30 * 100)
    assert node.equity(2) == pytest.approx(100_000.0)
    # Adoption books no trade; finish closes each seeded position, once.
    assert [len(r.result.trades()) for r in node.finish()] == [1, 1, 0]


@pytest.mark.parametrize(
    "seeds, message",
    [
        ({5: {"X": {"quantity": 1.0, "avg_price": 1.0}}}, "strategy index 5"),
        ({"nope": {"X": {"quantity": 1.0, "avg_price": 1.0}}}, "neither a strategy index"),
        ({0: {"Z": {"quantity": 1.0, "avg_price": 1.0}}}, "unknown symbol 'Z'"),
        ({1: {"Y": {"quantity": 1.0, "avg_price": 1.0}}}, "unknown symbol 'Y'"),
        ({0: {"X": {"quantity": 0.0, "avg_price": 1.0}}}, "non-zero quantity"),
        ({0: {"X": {"quantity": 1.0, "avg_price": -1.0}}}, "non-zero quantity"),
    ],
)
def test_bad_seeds_are_refused(seeds, message):
    with pytest.raises(ValueError, match=message):
        LiveNode([Reads(), (Reads(), ["X"])], ["X", "Y"], initial_positions=seeds)


def test_the_same_strategy_cannot_be_seeded_twice():
    s = Reads()
    with pytest.raises(ValueError, match="twice"):
        LiveNode([s], ["X"], initial_positions={0: {}, s: {}})


def test_a_short_or_leveraged_seed_is_refused_not_skipped():
    with pytest.raises(ValueError):
        LiveNode(
            [Reads()], ["X"], account_type="margin", leverage=5.0,
            initial_positions={0: {"X": {"quantity": 1.0, "avg_price": 10.0}}},
        )


# -- the end of the run ------------------------------------------------------------


def test_finish_returns_results_in_order_and_a_second_call_raises():
    node = LiveNode([Reads(), Reads()], ["X"])
    node.push_tick("X", 1, 10.0)
    results = node.finish()
    assert len(results) == 2
    with pytest.raises(RuntimeError, match="finished"):
        node.finish()
    for call in (
        lambda: node.push_tick("X", 2, 10.0),
        lambda: node.push_bar("X", 2, 1.0, 1.0, 1.0, 1.0, 1.0),
        lambda: node.push_depth("X", 2, [(1.0, 1.0)], [(2.0, 1.0)]),
        lambda: node.advance_time(3),
        lambda: node.equity(0),
        lambda: node.is_halted(0),
        lambda: node.positions(0),
    ):
        with pytest.raises(RuntimeError, match="finished"):
            call()


def test_on_stop_runs_once_per_strategy_before_the_result():
    stops = []

    class Stop(Strategy):
        def on_stop(self, ctx):
            stops.append(id(self))

    a, b = Stop(), Stop()
    LiveNode([a, b], ["X"]).finish()
    assert stops == [id(a), id(b)]


# -- timestamps never go backwards ------------------------------------------------


def test_a_decreasing_timestamp_raises_and_changes_nothing():
    class Count(Strategy):
        n = 0

        def on_trade(self, ctx, tick):
            Count.n += 1

    node = LiveNode([Count()], ["X", "Y"], bar_stamp="close")
    assert node.push_tick("X", 100, 10.0) == 1
    node.push_tick("X", 100, 10.0)  # equal is allowed
    node.push_tick("Y", 50, 10.0)  # another symbol keeps its own time
    seen = Count.n
    equity = node.equity(0)
    with pytest.raises(ValueError, match=r"X live feed: timestamps must not decrease.* 99 after 100"):
        node.push_tick("X", 99, 10.0)
    with pytest.raises(ValueError, match="timestamps must not decrease"):
        node.push_tick("X", 99, 0.0)  # a row with no print is checked too
    with pytest.raises(ValueError, match="timestamps must not decrease"):
        node.push_bar("X", 99, 1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="timestamps must not decrease"):
        node.push_depth("X", 99, [(9.0, 1.0)], [(11.0, 1.0)])
    assert (Count.n, node.equity(0)) == (seen, equity)
    assert node.push_tick("X", 101, 10.0) == 1, "the node is still usable"
    assert Count.n == seen + 1


def test_bar_timestamps_are_checked_at_the_bar_close():
    node = LiveNode([Reads()], ["X"], bar_stamp="open", bar_ns=60 * NS)
    node.push_bar("X", 0, 1.0, 1.0, 1.0, 1.0, 1.0)  # closes at 60 s
    node.push_tick("X", 60 * NS, 1.0)
    with pytest.raises(ValueError, match="timestamps must not decrease"):
        node.push_tick("X", 59 * NS, 1.0)


# -- storage ---------------------------------------------------------------------------


class SeriesProbe(Strategy):
    def __init__(self, indicator=None):
        super().__init__()
        self.indicator = indicator
        self.lengths = []
        self.last = None

    def on_start(self, ctx):
        if self.indicator:
            self.register_indicator(self.indicator(), symbol=ctx.symbols[0])
        self.lengths_at_start = len(ctx.series()["close"])

    def on_bar(self, ctx):
        s = ctx.series()
        self.lengths.append(len(s["close"]))
        self.last = s

    def on_trade(self, ctx, tick):
        s = ctx.series()
        self.lengths.append(len(s["ltp"]))
        self.last = s


def test_series_comes_from_the_ring_and_stops_at_the_watermark():
    probe = SeriesProbe()
    node = LiveNode([probe], ["X"], history=5, bar_stamp="close")
    assert probe.lengths_at_start == 0
    for i in range(12):
        node.push_bar("X", (i + 1) * NS, 1.0, 2.0, 0.5, float(i), 10.0)
    # Never longer than the history, never past the bar being handled.
    assert probe.lengths == [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5]
    assert probe.last["close"].tolist() == [7.0, 8.0, 9.0, 10.0, 11.0]
    assert probe.last["timestamps"].tolist() == [(i + 1) * NS for i in range(7, 12)]
    assert not probe.last["close"].flags.writeable


def test_tick_series_for_a_symbol_without_bars():
    probe = SeriesProbe()
    node = LiveNode([probe], ["X"], history=4)
    for i in range(7):
        node.push_tick("X", i + 1, 10.0 + i, 9.9 + i, 10.1 + i, ltq=2.0)
    assert probe.lengths == [1, 2, 3, 4, 4, 4, 4]
    assert probe.last["ltp"].tolist() == [13.0, 14.0, 15.0, 16.0]
    assert set(probe.last) == set(rf.TICK_KEYS)
    assert probe.last["bid"].tolist() == pytest.approx([12.9, 13.9, 14.9, 15.9])


def test_history_grows_to_what_the_indicators_read():
    probe = SeriesProbe(indicator=lambda: raptorbt.Indicator.sma(20))
    node = LiveNode([probe], ["X"], history=3, bar_stamp="close")
    for i in range(30):
        node.push_bar("X", (i + 1) * NS, 1.0, 2.0, 0.5, float(i), 10.0)
    assert max(probe.lengths) == 20


class Keeper(Strategy):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.ticks = 0

    def on_start(self, ctx):
        ctx.keep_history(self.rows)

    def on_trade(self, ctx, tick):
        self.ticks += 1


def test_backpressure_refuses_the_push_that_would_drop_a_kept_row():
    keeper = Keeper(rows=9)
    node = LiveNode([keeper], ["X"], history=8, bar_stamp="close")
    for i in range(8):
        node.push_tick("X", i + 1, 10.0)
    equity = node.equity(0)
    with pytest.raises(RuntimeError, match=r"X: the tick history is full \(8 rows\).*Nothing was dropped"):
        node.push_tick("X", 9, 10.0)
    # Nothing happened: no hook, no state change, no timestamp consumed.
    assert keeper.ticks == 8
    assert node.equity(0) == equity
    with pytest.raises(RuntimeError, match="history is full"):
        node.push_tick("X", 9, 10.0)
    # A bar ring is separate, and also holds what the strategy keeps.
    for i in range(8):
        node.push_bar("X", 100 + i, 1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(RuntimeError, match="bar history is full"):
        node.push_bar("X", 200, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_keeping_what_fits_never_refuses():
    keeper = Keeper(rows=8)
    node = LiveNode([keeper], ["X"], history=8)
    for i in range(500):
        node.push_tick("X", i + 1, 10.0)
    assert keeper.ticks == 500


def test_keeping_history_in_a_batch_run_changes_nothing():
    keeper = Keeper(rows=10**9)
    ticks = {"X": {"timestamps": np.arange(1, 6, dtype=np.int64), "ltp": np.full(5, 10.0)}}
    run_strategies([keeper], ticks=ticks)
    assert keeper.ticks == 5


def test_a_hook_cannot_push_into_the_node_it_runs_in():
    holder = {}

    class Reentrant(Strategy):
        def on_trade(self, ctx, tick):
            holder["node"].push_tick("X", tick.timestamp + 1, 10.0)

    node = LiveNode([Reentrant()], ["X"])
    holder["node"] = node
    with pytest.raises(RuntimeError, match="busy"):
        node.push_tick("X", 1, 10.0)


def _rss_kb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(out.stdout.strip())


def test_memory_stays_flat_over_a_million_ticks():
    symbols = [f"S{i}" for i in range(5)]
    node = LiveNode(
        [Reads() for _ in range(3)], symbols, history=1_000, curve_samples=512,
    )
    # Warm every ring, then measure.
    for i in range(20_000):
        node.push_tick(symbols[i % 5], i + 1, 100.0 + (i % 17) * 0.01, 99.9, 100.1, ltq=1.0)
    heap_warm = node.memory_report()
    rss_warm = _rss_kb()
    n = 1_000_000
    base = 20_001
    for i in range(n):
        node.push_tick(symbols[i % 5], base + i, 100.0 + (i % 17) * 0.01, 99.9, 100.1, ltq=1.0)
        if i % 250_000 == 249_999:
            assert node.memory_report()["total"] == heap_warm["total"], f"heap grew by tick {i}"
    heap_end = node.memory_report()
    rss_end = _rss_kb()
    assert heap_end["live_rings"] == heap_warm["live_rings"] > 0
    assert heap_end["total"] == heap_warm["total"], (heap_warm, heap_end)
    # The process as a whole: a million ticks add no meaningful memory.
    assert rss_end - rss_warm < 15_000, f"RSS grew {(rss_end - rss_warm) / 1000:.1f} MB"
    assert len(node.finish()) == 3


def test_full_curves_are_the_only_thing_that_grows_by_default():
    node = LiveNode([Reads()], ["X"])
    for i in range(20_000):
        node.push_tick("X", i + 1, 100.0 + (i % 17) * 0.01)
    early = node.memory_report()
    for i in range(20_000, 80_000):
        node.push_tick("X", i + 1, 100.0 + (i % 17) * 0.01)
    late = node.memory_report()
    assert late["live_rings"] == early["live_rings"]
    assert late["shared_store"] == early["shared_store"]
    assert late["curves"] > early["curves"] >= 20_000 * 32
    result = node.finish()[0]
    assert len(result.result.equity_curve()) == 80_000


def test_curve_samples_bounds_the_curves_and_leaves_trades_complete():
    class Churn(Strategy):
        def on_trade(self, ctx, tick):
            if ctx.idx % 50 == 1 and ctx.is_flat:
                self.enter(size_frac=0.1)
            elif ctx.idx % 50 == 26 and not ctx.is_flat:
                self.close_position()

    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    full = LiveNode([Churn()], ["X"], cfg)
    cut = LiveNode([Churn()], ["X"], cfg, curve_samples=200)
    for i in range(5_000):
        for node in (full, cut):
            node.push_tick("X", i + 1, 100.0 + (i % 23) * 0.1)
    (rf_,), (rc,) = full.finish(), cut.finish()
    assert len(rf_.result.equity_curve()) == 5_000
    kept = len(rc.result.equity_curve())
    assert 200 <= kept < 400
    assert np.array_equal(rc.result.equity_curve(), rf_.result.equity_curve()[-kept:])
    assert len(rc.result.trades()) == len(rf_.result.trades()) > 10


def test_bad_constructor_input():
    with pytest.raises(ValueError, match="at least one"):
        LiveNode([Reads()], [])
    with pytest.raises(ValueError, match="unique"):
        LiveNode([Reads()], ["X", "X"])
    with pytest.raises(ValueError, match="history"):
        LiveNode([Reads()], ["X"], history=0)
    with pytest.raises(ValueError, match="curve_samples"):
        LiveNode([Reads()], ["X"], curve_samples=0)
    with pytest.raises(ValueError, match="strategies is empty"):
        LiveNode([], ["X"])
    with pytest.raises(ValueError, match="no data for 'Z'"):
        LiveNode([(Reads(), ["Z"])], ["X"])
    with pytest.raises(ValueError, match="unknown symbol"):
        LiveNode([Reads()], ["X"], warmup_bars={"Z": {}}, bar_stamp="close")
    s = Reads()
    with pytest.raises(ValueError, match="twice"):
        LiveNode([s, s], ["X"])
    with pytest.raises(KeyError):
        LiveNode([Reads()], ["X"]).push_tick("Z", 1, 1.0)


# E-1: the node's context is cleared when the node finishes; asking for it then is a refusal
def test_ctx_after_finish_is_a_refusal_not_a_cleared_context():
    node = LiveNode([Strategy()], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close")
    node.push_bar("A", 60 * 10**9, 100.0, 101.0, 99.0, 100.0, 10.0)
    assert node.ctx(0) is not None
    node.finish()
    with pytest.raises(raptorbt.errors.StateError) as e:
        node.ctx(0)
    assert e.value.code == "node_finished"


# E-1: a node whose start fails after its strategies were built releases them
def test_a_node_whose_start_failed_releases_its_strategies():
    class Boom(Strategy):
        def on_bar(self, ctx):
            raise RuntimeError("boom in warm-up")

    s = Boom()
    one = 60 * 10**9
    warm = {"A": {"timestamps": np.array([one, 2 * one], dtype=np.int64), "open": np.array([1.0, 1.0]),
                  "high": np.array([1.0, 1.0]), "low": np.array([1.0, 1.0]), "close": np.array([1.0, 1.0]),
                  "volume": np.array([1.0, 1.0])}}
    with pytest.raises(RuntimeError, match="boom in warm-up"):
        LiveNode([s], ["A"], BacktestConfig(initial_capital=1e6, fees=0.0), bar_stamp="close", warmup_bars=warm)
    assert "_api" not in s.__dict__
