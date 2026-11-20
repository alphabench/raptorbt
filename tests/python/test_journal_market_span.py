"""Market-data records a host keeps in its own store and rebuilds at restore.

A host that already stores the market data it feeds a :class:`~raptorbt.LiveNode`
need not keep the journal's ``tick`` and ``time`` records as well: it keeps the
other records, keeps per market input the input's values and the sequence number
it was journaled under, and at restore rebuilds the records with the public
:func:`raptorbt.journal.canonical` and folds the chain with
:func:`raptorbt.journal.chain_step`. These tests establish what that takes and
what the engine does today on the edges a host meets when it does it.

What a ``tick`` and a ``time`` record's canonical form holds (pinned by
``test_the_fields_of_a_tick_and_a_time_record``):

================  ===========================================  =========================
field             ``tick`` (``push_tick``)                     ``time`` (``advance_time``)
================  ===========================================  =========================
seq               int: ``journal.count`` just before the push  the same
kind              ``"tick"``                                   ``"time"``
ts_event          int ns: the push's ``timestamp``             int ns: the timestamp
ts_arrival        int ns: ``arrival_ns``, else ``ts_event``    always ``ts_event``
source            ``"feed"``                                   ``"feed"``
client_order_id   ``null``                                     ``null``
payload           ``symbol`` (text) and the nine numbers       ``{"timestamp": int}``
                  ``ltp bid ask buy_qty_delta sell_qty_delta``
                  ``ltq bid_qty ask_qty oi``, each a float
                  written as text; an argument not given is
                  ``"0"``
================  ===========================================  =========================

A float is the text of Rust's shortest round-trip decimal without an exponent
(``100.0`` is ``"100"``, ``1e-07`` is ``"0.0000001"``): hand
:func:`~raptorbt.journal.canonical` the float, never text you formatted, and
never an int (``100`` is written as the JSON integer ``100``, not ``"100"``).
"""

import json
import random

import numpy as np
import pytest

from raptorbt import BacktestConfig, Indicator, LiveNode, Strategy, journal as rj
from raptorbt.errors import InvalidData, InvalidParameter, StateError
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders
from raptorbt.strategy._recovery import state_summary

NS = 1_000_000_000
MIN = 60 * NS
HEADER = {"run": "market-span"}
SYMBOLS = ["A", "B"]
MARKET = ("tick", "time")
TICK_FIELDS = ("ltp", "bid", "ask", "buy_qty_delta", "sell_qty_delta", "ltq", "bid_qty", "ask_qty", "oi")


def _cfg():
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0)


# -- 1. caller-rebuilt market records restore exactly ---------------------------------


class TickTrader(Strategy):
    """Rests a limit buy under the print on a symbol's 2nd, 8th, ... trade and sells at
    market on its 5th, 11th, ...: orders and fills land on different pushes."""

    def __init__(self):
        super().__init__()
        self.count = {}

    def on_trade(self, ctx, tick):
        n = self.count[ctx.symbol] = self.count.get(ctx.symbol, 0) + 1
        if n % 6 == 2 and not ctx.positions() and not ctx.open_orders():
            self.submit_order(orders.Limit(side="buy", price=round(tick.price - 0.4, 2), units=3.0))
        elif n % 6 == 5 and ctx.positions():
            self.submit_order(orders.Market(side="sell", units=3.0, reduce_only=True))


def _stream(seed=11, n=120):
    """Ticks on two symbols with a time step after every ninth; arrival trails the event."""
    rng = random.Random(seed)
    px = {"A": 100.0, "B": 50.0}
    t = 0
    out = []
    for i in range(n):
        t += rng.randrange(1, 4) * NS // 2
        s = rng.choice(SYMBOLS)
        px[s] = round(px[s] + rng.choice([-0.3, -0.1, 0.1, 0.3]), 2)
        ltq = float(rng.randrange(1, 9))
        out.append(("tick", s, t, px[s], round(px[s] - 0.05, 2), round(px[s] + 0.05, 2), ltq, t + rng.randrange(0, 5000)))
        if i % 9 == 8:
            t += NS // 3
            out.append(("time", t))
    return out


def _feed(node, x):
    if x[0] == "tick":
        _, s, t, ltp, bid, ask, ltq, arrival = x
        node.push_tick(s, t, ltp, bid, ask, ltq=ltq, arrival_ns=arrival)
    else:
        node.advance_time(x[1])


def _rebuild(x, seq):
    """The canonical bytes of a market record, from the input's values and its seq alone."""
    if x[0] == "tick":
        _, s, t, ltp, bid, ask, ltq, arrival = x
        payload = dict.fromkeys(TICK_FIELDS, 0.0)
        payload.update(symbol=s, ltp=float(ltp), bid=float(bid), ask=float(ask), ltq=float(ltq))
        return rj.canonical({
            "seq": seq, "kind": "tick", "ts_event": t, "ts_arrival": arrival,
            "source": "feed", "client_order_id": None, "payload": payload,
        })
    return rj.canonical({
        "seq": seq, "kind": "time", "ts_event": x[1], "ts_arrival": x[1],
        "source": "feed", "client_order_id": None, "payload": {"timestamp": x[1]},
    })


def _node(journal):
    return LiveNode([TickTrader()], SYMBOLS, _cfg(), execution=PaperExecution(), journal=journal)


def _restore(records, checkpoint=None):
    journal = rj.Journal(HEADER, memory=True)
    node = LiveNode.restore(
        checkpoint, [TickTrader()], SYMBOLS, _cfg(), execution=PaperExecution(),
        journal=journal, records=records,
    )
    return node, journal


@pytest.fixture(scope="module")
def run():
    """The uninterrupted run, the host's two stores, and a checkpoint after every input."""
    kept = {}  # seq -> canonical bytes, every record that is not market data

    def sink(record):
        if record.kind not in MARKET:
            kept[record.seq] = record.canonical()

    journal = rj.Journal(HEADER, memory=True, sink=sink)
    node = _node(journal)
    inputs = _stream()
    market = []  # (seq, input): what the host keeps of its market data
    bounds, checkpoints = [], []
    for x in inputs:
        market.append((journal.count, x))
        _feed(node, x)
        bounds.append(journal.count)
        checkpoints.append(node.checkpoint(include_log=False))
    original = journal.since(0)
    rebuilt = dict(kept)
    for seq, x in market:
        rebuilt[seq] = _rebuild(x, seq)
    return {
        "node": node, "journal": journal, "inputs": inputs, "market": market, "kept": kept,
        "bounds": bounds, "checkpoints": checkpoints, "original": original,
        "records": [rebuilt[q] for q in sorted(rebuilt)], "seqs": sorted(rebuilt),
    }


def test_the_host_keeps_no_market_record_and_rebuilds_each_byte_for_byte(run):
    original, records = run["original"], run["records"]
    assert not any(json.loads(b)["kind"] in MARKET for b in run["kept"].values())
    assert {r.kind for r in original} >= {"tick", "time", "command", "adapter_request", "simulated_report"}
    # Every seq is accounted for: the host's market store and the kept records tile the journal.
    assert run["seqs"] == list(range(len(original)))
    for seq, x in run["market"]:
        assert original[seq].kind == x[0], "the input record's seq is journal.count before the push"
    head = rj.chain_start(HEADER)
    for record, rebuilt in zip(original, records):
        head = rj.chain_step(head, rebuilt)
        assert rebuilt == record.canonical(), record.seq
        assert head == record.chain, record.seq
    assert head == run["journal"].head()[1]


def _cut(run, name):
    """The input after which to cut, by what that input made the node write."""
    original, bounds = run["original"], run["bounds"]
    starts = [seq for seq, _ in run["market"]]
    made = [[original[q] for q in range(a, b)] for a, b in zip(starts, bounds)]

    def fills(rs):
        return [r for r in rs if r.kind == "simulated_report" and r.payload["report"] == "fill"]

    for i, rs in enumerate(made):
        kinds = [r.kind for r in rs]
        if name == "order" and "adapter_request" in kinds and not fills(rs):
            return i
        if name == "fill" and kinds[0] == "tick" and "command" not in kinds and fills(rs):
            return i
        if name == "after_time" and kinds == ["time"]:
            return i
        if name == "before_time" and i + 1 < len(made) and [r.kind for r in made[i + 1]] == ["time"]:
            return i
    if name == "mid":
        return len(made) // 2
    raise AssertionError(f"the stream has no {name!r} cut")


@pytest.mark.parametrize("name", ["mid", "order", "fill", "before_time", "after_time"])
def test_a_node_restored_from_rebuilt_market_records_equals_the_uninterrupted_run(run, name):
    i = _cut(run, name)
    count = run["bounds"][i]
    node, journal = _restore(run["records"][:count], run["checkpoints"][i])
    summary = state_summary(node)["strategies"][0]
    if name == "order":
        assert summary["orders"], "the cut holds a resting order"
    if name == "fill":
        assert any(summary["positions"].values()), "the cut holds the position the fill opened"
    for x in run["inputs"][i + 1:]:
        _feed(node, x)
    assert node.state_digest() == run["node"].state_digest()
    assert node.equity(0) == run["node"].equity(0)
    # What the restored node wrote after the cut is what the uninterrupted node wrote.
    assert [r.canonical() for r in journal.since(count)] == [r.canonical() for r in run["original"][count:]]
    assert journal.head()[:2] == run["journal"].head()[:2]


def test_rebuilt_records_restore_at_every_input_boundary(run):
    for i, count in enumerate(run["bounds"]):
        node, _ = _restore(run["records"][:count], run["checkpoints"][i])
        original, _ = _restore([r.canonical() for r in run["original"][:count]], run["checkpoints"][i])
        assert node.state_digest() == original.state_digest(), i


def _altered(run, i, change):
    records = list(run["records"][: run["bounds"][i]])
    seq = run["market"][0][0]
    d = json.loads(records[seq])
    change(d)
    records[seq] = rj.canonical(d)
    return records


def test_a_rebuilt_record_that_differs_is_refused_only_against_a_checkpoint(run):
    """The replay pushes each input record as it is given and compares what it then writes:
    an input that differs where nothing downstream reads it (here, the first tick's arrival,
    before any order) is reproduced from itself. Only the checkpoint's chain head finds it."""
    i = _cut(run, "after_time")

    def later(d):
        d["ts_arrival"] += 1

    records = _altered(run, i, later)
    count = run["bounds"][i]
    # Without a checkpoint it is accepted, and the restored journal carries another chain.
    _, journal = _restore(records)
    assert journal.count == count
    assert journal.chain != run["original"][count - 1].chain
    with pytest.raises(StateError) as e:
        _restore(records, run["checkpoints"][i])
    assert e.value.code == "replay_diverged"
    assert "chain head" in str(e.value)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda d: d.update(source="host"), id="source"),
        pytest.param(lambda d: d["payload"].update(oi=0), id="int-zero"),
    ],
)
def test_a_rebuilt_record_in_the_wrong_form_is_refused_at_that_record(run, change):
    """The replay writes the record from the values it parsed, in the engine's form: a record
    whose form differs (another source, a zero as the JSON integer ``0`` instead of ``"0"``)
    diverges where it stands."""
    i = _cut(run, "after_time")
    with pytest.raises(StateError) as e:
        _restore(_altered(run, i, change))
    assert e.value.code == "replay_diverged"
    assert f"record {run['market'][0][0]}:" in str(e.value)


# -- 2. what a record needs -----------------------------------------------------------


def test_the_fields_of_a_tick_and_a_time_record():
    journal = rj.Journal(HEADER, memory=True)
    node = LiveNode([Strategy()], SYMBOLS, _cfg(), journal=journal)
    node.push_tick("A", 5 * NS, 100.25, 100.2, 100.3, 5.0, 2.0, 3.0, 40.0, 50.0, 1234.0, arrival_ns=5 * NS + 777)
    node.push_tick("B", 6 * NS, 100.0)
    node.advance_time(7 * NS)
    full, bare, time_step = journal.since(1)
    assert json.loads(full.canonical()) == {
        "seq": 1, "kind": "tick", "ts_event": 5 * NS, "ts_arrival": 5 * NS + 777,
        "source": "feed", "client_order_id": None,
        "payload": {
            "symbol": "A", "ltp": "100.25", "bid": "100.2", "ask": "100.3", "buy_qty_delta": "5",
            "sell_qty_delta": "2", "ltq": "3", "bid_qty": "40", "ask_qty": "50", "oi": "1234",
        },
    }
    assert json.loads(bare.canonical()) == {
        "seq": 2, "kind": "tick", "ts_event": 6 * NS, "ts_arrival": 6 * NS,
        "source": "feed", "client_order_id": None,
        "payload": {"symbol": "B", "ltp": "100", **dict.fromkeys(TICK_FIELDS[1:], "0")},
    }
    assert json.loads(time_step.canonical()) == {
        "seq": 3, "kind": "time", "ts_event": 7 * NS, "ts_arrival": 7 * NS,
        "source": "feed", "client_order_id": None, "payload": {"timestamp": 7 * NS},
    }
    # The canonical form is sorted, separator-free JSON of exactly these values.
    for r in (full, bare, time_step):
        as_read = {
            "seq": r.seq, "kind": r.kind, "ts_event": r.ts_event, "ts_arrival": r.ts_arrival,
            "source": r.source, "client_order_id": r.client_order_id, "payload": r.payload,
        }
        assert r.canonical() == json.dumps(as_read, sort_keys=True, separators=(",", ":")).encode()
    # Rebuilt from the pushed values, with floats given as floats.
    assert full.canonical() == rj.canonical({
        "seq": 1, "kind": "tick", "ts_event": 5 * NS, "ts_arrival": 5 * NS + 777, "source": "feed",
        "client_order_id": None,
        "payload": dict(zip(TICK_FIELDS, (100.25, 100.2, 100.3, 5.0, 2.0, 3.0, 40.0, 50.0, 1234.0)), symbol="A"),
    })
    # A price given as an int is written as a JSON integer, not the engine's text.
    as_int = rj.canonical({
        "seq": 2, "kind": "tick", "ts_event": 6 * NS, "ts_arrival": 6 * NS, "source": "feed",
        "client_order_id": None, "payload": {**dict.fromkeys(TICK_FIELDS, 0.0), "symbol": "B", "ltp": 100},
    })
    assert as_int != bare.canonical() and b'"ltp":100,' in as_int


@pytest.mark.parametrize(
    "value, text",
    [
        (100.0, "100"), (100.25, "100.25"), (0.1 + 0.2, "0.30000000000000004"), (-0.0, "-0"),
        (1e-7, "0.0000001"), (1e21, "1000000000000000000000"), (np.float64(99.95), "99.95"),
    ],
)
def test_a_float_is_written_as_shortest_round_trip_text_without_an_exponent(value, text):
    assert rj.canonical(value) == json.dumps(text).encode()


# -- 3. a tick older than the last time step ------------------------------------------


class Watcher(Strategy):
    def __init__(self):
        super().__init__()
        self.heard = []

    def on_start(self, ctx):
        self.clock.set_timer("t", 10 * NS, start_ns=10 * NS)

    def on_timer(self, ctx, event):
        self.heard.append(("timer", ctx.symbol, event.ts_fired // NS))

    def on_trade(self, ctx, tick):
        self.heard.append(("trade", ctx.symbol, tick.timestamp // NS, ctx.event_ns // NS, tick.price))

    def on_bar(self, ctx, bar):
        self.heard.append(
            ("bar", ctx.symbol, bar.timestamp // NS, ctx.event_ns // NS, bar.open, bar.high, bar.low, bar.close, bar.volume)
        )


def _late_tick_run(node):
    node.push_tick("A", 10 * NS, 100.0, ltq=1.0)
    node.push_tick("B", 30 * NS, 50.0, ltq=1.0)
    node.advance_time(70 * NS)
    clock_after_time = node.health()["clock_ns"]
    appended = node.push_tick("A", 40 * NS, 120.0, ltq=1.0)  # older than the time step, newer than A's last
    clock_after_late = node.health()["clock_ns"]
    node.push_tick("A", 75 * NS, 101.0, ltq=1.0)
    return appended, clock_after_time, clock_after_late


def test_a_tick_older_than_the_last_time_step_is_accepted_at_its_own_time():
    """``advance_time`` moves the strategies' clocks, not any symbol's feed watermark: a tick
    stamped before it but not before its own symbol's last tick is accepted unclamped, its hook
    sees its own timestamp, it joins the bar of its own window, no timer fires again, and the
    node's reported clock steps back to it. The journal holds it after the ``time`` record and
    a replay reproduces it."""
    strategy, journal = Watcher(), rj.Journal(HEADER, memory=True)
    node = LiveNode([strategy], SYMBOLS, _cfg(), journal=journal, primary_bars=(1, "m"))
    assert _late_tick_run(node) == (1, 70 * NS, 40 * NS)
    assert strategy.heard == [
        ("timer", "A", 10), ("trade", "A", 10, 10, 100.0),
        ("timer", "B", 30), ("trade", "B", 30, 30, 50.0),
        ("timer", "A", 70), ("timer", "B", 70),
        ("trade", "A", 40, 40, 120.0),
        # The 1-minute bar of A closes on A's first tick past 60 s and holds the late print.
        ("bar", "A", 60, 75, 100.0, 120.0, 100.0, 120.0, 2.0),
        ("trade", "A", 75, 75, 101.0),
    ]
    assert [(r.kind, r.ts_event // NS) for r in journal.since(1)] == [
        ("tick", 10), ("tick", 30), ("time", 70), ("tick", 40), ("tick", 75),
    ]
    replayed = Watcher()
    restored = LiveNode.restore(
        None, [replayed], SYMBOLS, _cfg(), journal=rj.Journal(HEADER, memory=True),
        records=[r.canonical() for r in journal.since(0)], primary_bars=(1, "m"),
    )
    assert replayed.heard == strategy.heard
    assert restored.state_digest() == node.state_digest()
    # Older than its own symbol's last tick is refused, before it is journaled.
    count = journal.count
    with pytest.raises(InvalidParameter) as e:
        node.push_tick("A", 74 * NS, 101.0)
    assert e.value.code == "invalid_parameter"
    assert journal.count == count


# -- 4. advance_time and bars built from ticks ----------------------------------------


class BarListener(Strategy):
    def __init__(self):
        super().__init__()
        self.heard = []

    def on_start(self, ctx):
        self.subscribe_bars(3, "m")

    def on_bar(self, ctx, bar):
        self.heard.append(
            (type(bar).__name__, bar.timestamp // NS, ctx.event_ns // NS, bar.open, bar.high, bar.low, bar.close, bar.volume)
        )


def test_advance_time_past_a_boundary_closes_no_bar_built_from_ticks():
    """A bar built from ticks, 1-minute or coarser, closes on its symbol's first tick of a later
    window, never on ``advance_time``; windows with no tick give no bar. The journal holds only
    the ``tick`` and ``time`` inputs: the bars are derived again on replay."""
    strategy, journal = BarListener(), rj.Journal(HEADER, memory=True)
    node = LiveNode([strategy], ["A"], _cfg(), journal=journal, primary_bars=(1, "m"))
    node.push_tick("A", 10 * NS, 100.0, ltq=1.0)
    node.push_tick("A", 50 * NS, 101.0, ltq=2.0)
    node.advance_time(65 * NS)
    assert strategy.heard == [], "past the 1-minute boundary"
    node.advance_time(200 * NS)
    assert strategy.heard == [], "past the 3-minute boundary too"
    node.push_tick("A", 201 * NS, 105.0, ltq=1.0)
    assert strategy.heard == [
        ("Bar", 60, 201, 100.0, 101.0, 100.0, 101.0, 3.0),
        ("CompositeBar", 180, 201, 100.0, 101.0, 100.0, 101.0, 3.0),
    ]
    assert [r.kind for r in journal.since(1)] == ["tick", "tick", "time", "time", "tick"]
    replayed = BarListener()
    LiveNode.restore(
        None, [replayed], ["A"], _cfg(), journal=rj.Journal(HEADER, memory=True),
        records=[r.canonical() for r in journal.since(0)], primary_bars=(1, "m"),
    )
    assert replayed.heard == strategy.heard


# -- 5. warm-up bars, then ticks for the same symbol ----------------------------------


CLOSES = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
HISTORY = {
    "A": dict(
        timestamps=np.arange(1, 6) * MIN, open=CLOSES, high=CLOSES + 1, low=CLOSES - 1, close=CLOSES,
        volume=np.full(5, 10.0),
    )
}


class Primed(Strategy):
    def __init__(self):
        super().__init__()
        self.heard = []

    def on_start(self, ctx):
        self.sma = self.register_indicator(Indicator.sma(3), symbol="A")

    def on_bar(self, ctx, bar):
        series = ctx.series()
        self.heard.append(
            (bar.timestamp // NS, bar.open, bar.close, bar.volume, ctx.is_warmup, self.sma.value,
             len(series["close"]), int(series["timestamps"][-1]) // NS)
        )


def _prime(how, node):
    if how == "warm_up":
        node.warm_up(HISTORY)
    for i, c in enumerate(CLOSES):
        if how == "push_bar":
            node.push_bar("A", (i + 1) * MIN, c, c + 1, c - 1, c, 10.0)
        elif how == "push_group":
            node.push_group((i + 1) * MIN, {"A": (c, c + 1, c - 1, c, 10.0)})


def _ticks(node):
    node.push_tick("A", 5 * MIN + 10 * NS, 110.0, ltq=1.0)
    node.push_tick("A", 5 * MIN + 50 * NS, 111.0, ltq=1.0)
    node.push_tick("A", 6 * MIN + 10 * NS, 112.0, ltq=1.0)
    node.push_tick("A", 7 * MIN + 10 * NS, 113.0, ltq=1.0)


def _primed_kwargs(how):
    kw = dict(primary_bars=(1, "m"), bar_stamp="close")
    if how in ("live", "muted"):
        kw.update(warmup_bars=HISTORY, warmup=how)
    return kw


HOWS = ["muted", "live", "warm_up", "push_bar", "push_group"]


@pytest.mark.parametrize("how", HOWS)
def test_ticks_after_pushed_history_build_the_next_bar_and_the_indicator_goes_on(how):
    """History given as closed bars (constructor warm-up muted or live, ``warm_up``,
    ``push_bar``, ``push_group``), then ticks: nothing is refused, the first bar the engine
    builds is the window after the last pushed bar, and the indicator runs on across the
    join. ``ctx.series()`` keeps reading the pushed bars only: the bars built from ticks do not
    join it."""
    strategy = Primed()
    node = LiveNode([strategy], ["A"], _cfg(), journal=rj.Journal(HEADER, memory=True), **_primed_kwargs(how))
    _prime(how, node)
    _ticks(node)
    warm = how in ("muted", "live", "warm_up")
    assert strategy.heard == [
        (60, 100.0, 100.0, 10.0, warm, None, 1, 60),
        (120, 101.0, 101.0, 10.0, warm, None, 2, 120),
        (180, 102.0, 102.0, 10.0, warm, 101.0, 3, 180),
        (240, 103.0, 103.0, 10.0, warm, 102.0, 4, 240),
        (300, 104.0, 104.0, 10.0, warm, 103.0, 5, 300),
        (360, 110.0, 111.0, 2.0, False, (103.0 + 104.0 + 111.0) / 3, 5, 300),
        (420, 112.0, 112.0, 1.0, False, (104.0 + 111.0 + 112.0) / 3, 5, 300),
    ]


def test_a_tick_at_the_last_pushed_bars_stamp_opens_the_next_bar_and_an_earlier_one_is_refused():
    strategy = Primed()
    node = LiveNode([strategy], ["A"], _cfg(), journal=rj.Journal(HEADER, memory=True), **_primed_kwargs("push_bar"))
    _prime("push_bar", node)
    with pytest.raises(InvalidParameter) as e:
        node.push_tick("A", 5 * MIN - 10 * NS, 200.0, ltq=1.0)
    assert e.value.code == "invalid_parameter"
    assert not node.health()["poisoned"]
    node.push_tick("A", 5 * MIN, 200.0, ltq=1.0)
    node.push_tick("A", 6 * MIN + 5 * NS, 201.0, ltq=1.0)
    assert strategy.heard[-1][:4] == (360, 200.0, 200.0, 1.0)


@pytest.mark.parametrize("how", [h for h in HOWS if h != "warm_up"])
def test_a_node_primed_with_bars_then_fed_ticks_restores(how):
    journal = rj.Journal(HEADER, memory=True)
    strategy = Primed()
    node = LiveNode([strategy], ["A"], _cfg(), journal=journal, **_primed_kwargs(how))
    _prime(how, node)
    _ticks(node)
    replayed = Primed()
    restored = LiveNode.restore(
        None, [replayed], ["A"], _cfg(), journal=rj.Journal(HEADER, memory=True),
        records=[r.canonical() for r in journal.since(0)], **_primed_kwargs(how),
    )
    assert restored.state_digest() == node.state_digest()
    assert replayed.heard == strategy.heard


def test_a_node_primed_with_warm_up_after_construction_restores():
    journal = rj.Journal(HEADER, memory=True)
    strategy = Primed()
    node = LiveNode([strategy], ["A"], _cfg(), journal=journal, **_primed_kwargs("warm_up"))
    _prime("warm_up", node)
    _ticks(node)
    replayed = Primed()
    restored = LiveNode.restore(
        None, [replayed], ["A"], _cfg(), journal=rj.Journal(HEADER, memory=True),
        records=[r.canonical() for r in journal.since(0)], **_primed_kwargs("warm_up"),
    )
    assert restored.state_digest() == node.state_digest()
    assert replayed.heard == strategy.heard


class PrimedPair(Strategy):
    def __init__(self):
        super().__init__()
        self.heard = []

    def on_start(self, ctx):
        self.sma = {s: self.register_indicator(Indicator.sma(3), symbol=s) for s in SYMBOLS}

    def on_bar(self, ctx, bar):
        self.heard.append(
            (bar.timestamp // NS, bar.close, ctx.is_warmup, self.sma["A"].value, self.sma["B"].value)
        )


def _history(first, n, base):
    c = base + np.arange(n, dtype=float)
    return dict(
        timestamps=(np.arange(n) + first) * MIN, open=c, high=c + 1, low=c - 1, close=c,
        volume=np.full(n, 10.0),
    )


def _warm(**bars):
    return lambda node: node.warm_up(bars)


def _bar(symbol, minute, c):
    return lambda node: node.push_bar(symbol, minute * MIN, c, c + 1, c - 1, c, 10.0)


WARM_UP_STEPS = {
    "twice": [_warm(A=_history(1, 3, 100.0)), _warm(A=_history(4, 2, 103.0)), _bar("A", 6, 105.0)],
    "after_live_bars": [
        _bar("A", 1, 99.0), _bar("A", 2, 99.5), _warm(A=_history(3, 3, 100.0)), _bar("A", 6, 105.0),
        _warm(A=_history(7, 2, 106.0)), _bar("A", 9, 108.0),
    ],
    "two_symbols": [
        _warm(A=_history(1, 5, 100.0), B=_history(1, 5, 200.0)),
        lambda node: node.push_group(6 * MIN, {"A": (105.0, 106.0, 104.0, 105.0, 10.0),
                                               "B": (205.0, 206.0, 204.0, 205.0, 10.0)}),
    ],
    "second_call_older_than_the_first": [
        _warm(A=_history(1, 5, 100.0)), _warm(B=_history(1, 5, 200.0)), _bar("B", 6, 205.0),
    ],
}


@pytest.mark.parametrize("steps", WARM_UP_STEPS.values(), ids=WARM_UP_STEPS)
def test_warm_up_calls_restore_at_every_step(steps):
    """``warm_up`` called twice, after live bars, for two symbols, and a second call older than
    the first: a restore from the records up to the end of each step (right after a warm-up, and
    later) reproduces every record, the state digest and what the strategy heard."""
    journal = rj.Journal(HEADER, memory=True)
    strategy = PrimedPair()
    node = LiveNode([strategy], SYMBOLS, _cfg(), journal=journal, **_primed_kwargs("warm_up"))
    cuts = []
    for step in steps:
        step(node)
        cuts.append((journal.count, node.state_digest(), list(strategy.heard)))
    records = [r.canonical() for r in journal.since(0)]
    for count, digest, heard in cuts:
        replayed = PrimedPair()
        restored = LiveNode.restore(
            None, [replayed], SYMBOLS, _cfg(), journal=rj.Journal(HEADER, memory=True),
            records=records[:count], **_primed_kwargs("warm_up"),
        )
        assert restored.state_digest() == digest
        assert replayed.heard == heard


def test_a_checkpoint_between_two_warm_up_calls_restores():
    journal = rj.Journal(HEADER, memory=True)
    node = LiveNode([PrimedPair()], SYMBOLS, _cfg(), journal=journal, **_primed_kwargs("warm_up"))
    node.warm_up({"A": _history(1, 3, 100.0)})
    checkpoint = node.checkpoint()
    node.warm_up({"A": _history(4, 2, 103.0)})
    node.push_bar("A", 6 * MIN, 105.0, 106.0, 104.0, 105.0, 10.0)
    restored = LiveNode.restore(
        checkpoint, [PrimedPair()], SYMBOLS, _cfg(), journal=rj.Journal(HEADER, memory=True),
        records=[r.canonical() for r in journal.since(0)], **_primed_kwargs("warm_up"),
    )
    assert restored.state_digest() == node.state_digest()


@pytest.mark.xfail(strict=True, reason=(
    "known limit: a warm_up() whose first bar shares its close time with the previous "
    "warm_up()'s last bar, of a later symbol, with no record between, journals the same bytes "
    "as one call; the replay applies them as one group and a hook sees the other symbol's bar"
))
def test_a_second_warm_up_at_the_close_time_the_first_ended_on_restores():
    journal = rj.Journal(HEADER, memory=True)
    strategy = PrimedPair()
    node = LiveNode([strategy], SYMBOLS, _cfg(), journal=journal, **_primed_kwargs("warm_up"))
    node.warm_up({"A": _history(1, 3, 100.0), "B": _history(1, 2, 200.0)})
    node.warm_up({"B": _history(3, 3, 202.0)})
    replayed = PrimedPair()
    restored = LiveNode.restore(
        None, [replayed], SYMBOLS, _cfg(), journal=rj.Journal(HEADER, memory=True),
        records=[r.canonical() for r in journal.since(0)], **_primed_kwargs("warm_up"),
    )
    assert restored.state_digest() == node.state_digest()
    assert replayed.heard == strategy.heard


# -- 6. an absent book and a NaN book -------------------------------------------------


class BookWatcher(Strategy):
    def __init__(self):
        super().__init__()
        self.heard = []

    def on_trade(self, ctx, tick):
        self.heard.append(("trade", tick.timestamp // NS, ctx.best_bid, ctx.best_ask))

    def on_quote(self, ctx, quote):
        self.heard.append(("quote", quote.timestamp // NS, quote.bid, quote.ask, ctx.best_bid, ctx.best_ask))


def test_an_absent_book_is_written_as_zero_and_leaves_the_last_quote_standing():
    """bid and ask not given are 0: the row is a trade only (no quote event), journaled with
    ``"0"``, and ``best_bid``/``best_ask`` keep the last two-sided quote (``None`` before any).
    A one-sided book makes no quote either; a row with no print and a two-sided book is a
    quote only; a row with neither makes no event and is still journaled."""
    strategy, journal = BookWatcher(), rj.Journal(HEADER, memory=True)
    node = LiveNode([strategy], ["A"], _cfg(), journal=journal)
    assert [
        node.push_tick("A", 1 * NS, 100.0),
        node.push_tick("A", 2 * NS, 100.0, 99.5, 100.5),
        node.push_tick("A", 3 * NS, 101.0),
        node.push_tick("A", 4 * NS, 101.0, 100.0),
        node.push_tick("A", 5 * NS, 0.0),
        node.push_tick("A", 6 * NS, 0.0, 99.0, 99.5),
    ] == [1, 2, 1, 1, 0, 1]
    assert strategy.heard == [
        ("trade", 1, None, None),
        ("trade", 2, None, None),  # the print comes before the quote of its own row
        ("quote", 2, 99.5, 100.5, 99.5, 100.5),
        ("trade", 3, 99.5, 100.5),
        ("trade", 4, 99.5, 100.5),
        ("quote", 6, 99.0, 99.5, 99.0, 99.5),
    ]
    books = [(r.payload["bid"], r.payload["ask"]) for r in journal.since(1)]
    assert books == [("0", "0"), ("99.5", "100.5"), ("0", "0"), ("100", "0"), ("0", "0"), ("99", "99.5")]


@pytest.mark.parametrize(
    "bid, ask, parameter",
    [(float("nan"), float("nan"), "bid"), (float("nan"), 100.5, "bid"), (99.5, float("nan"), "ask"), (99.5, float("inf"), "ask")],
)
def test_a_nan_book_is_refused_before_it_is_journaled(bid, ask, parameter):
    strategy, journal = BookWatcher(), rj.Journal(HEADER, memory=True)
    node = LiveNode([strategy], ["A"], _cfg(), journal=journal)
    count = journal.count
    with pytest.raises(InvalidData) as e:
        node.push_tick("A", 1 * NS, 100.0, bid, ask)
    assert (e.value.code, e.value.parameter) == ("non_finite_price", parameter)
    assert journal.count == count and strategy.heard == []
    assert not node.health()["poisoned"]
    assert node.push_tick("A", 1 * NS, 100.0) == 1, "the node goes on"
