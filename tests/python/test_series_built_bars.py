"""``ctx.series()`` of a symbol given closed bars and then prints that ``primary_bars`` builds into
bars: the built bars join its bars, in a live node, a batch run and a windowed run alike.

Every bar ``on_bar`` hears is in the series it reads, the bar being delivered included, in the
order it was heard. A built bar is stamped as a pushed bar of the same close would be (its open
under ``bar_stamp="open"``, its close under ``"close"``). Bars built before the symbol's first
pushed bar do not join: until then its series is its ticks, as for a symbol fed only ticks. A
live node shows at most ``history`` rows; a windowed run the rows a strategy declared.
"""

import random

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, journal as rj
from raptorbt.errors import StateError
from raptorbt.strategy.multi_runner import run_strategies

from test_exit_marks import Probe, Ruled, _Nothing, _same
from test_exit_marks import ex as exits
from test_windowed import windowed_like

NS = 1_000_000_000
MIN = 60 * NS
HEADER = {"run": "series-built-bars"}
CUTS = [("whole", 0), ("random", 1), ("random", 4), ("random", 9)]
BUILT_ROW = 56  # bytes of one built bar: the pushed bars before it (u64) and the bar (6 x 8)


def _cfg():
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0)


def _stamp_kw(stamp):
    return dict(bar_stamp=stamp, bar_ns=MIN)


def _bars(n=5, stamp="close"):
    """``n`` one-minute bars, the first closing at one minute, stamped as ``stamp`` says."""
    c = 100.0 + np.arange(n, dtype=float)
    first = 1 if stamp == "close" else 0
    return dict(
        timestamps=(np.arange(n, dtype=np.int64) + first) * MIN, open=c, high=c + 1, low=c - 1, close=c,
        volume=np.full(n, 10.0),
    )


def _ticks(seed=7, n=60, start=5 * MIN):
    """Prints after the bars: a walk, several a minute, with minutes that have none."""
    rng = random.Random(seed)
    t, px = start, 104.0
    ts, ltp, ltq = [], [], []
    for _ in range(n):
        t += rng.randrange(5, 40) * NS
        if rng.random() < 0.1:
            t += 2 * MIN
        px = round(px + rng.choice([-0.3, -0.1, 0.1, 0.3]), 2)
        ts.append(t)
        ltp.append(px)
        ltq.append(float(rng.randrange(1, 9)))
    return dict(timestamps=np.array(ts, dtype=np.int64), ltp=np.array(ltp), ltq=np.array(ltq))


class Reader(Strategy):
    """Records, at every ``on_bar``, the bar and the whole series: keys, dtypes, flags, values."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx, bar):
        s = ctx.series()
        self.seen.append((
            (bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume),
            {k: (str(v.dtype), bool(v.flags.writeable), v.tolist()) for k, v in s.items()},
        ))

    def series(self):
        return [s for _, s in self.seen]


def _live(bars, ticks, stamp, **kw):
    s = Reader()
    node = LiveNode([s], ["A"], _cfg(), primary_bars=(1, "m"), **_stamp_kw(stamp), **kw)
    _feed(node, bars, ticks)
    return s, node


def _feed(node, bars, ticks):
    for i in range(len(bars["timestamps"])):
        node.push_bar("A", int(bars["timestamps"][i]), *(float(bars[k][i]) for k in ("open", "high", "low", "close", "volume")))
    for t, p, q in zip(ticks["timestamps"].tolist(), ticks["ltp"].tolist(), ticks["ltq"].tolist()):
        node.push_tick("A", t, p, ltq=q)


def _batch(bars, ticks, stamp, **kw):
    s = Reader()
    run_strategies([s], {"A": bars}, _cfg(), ticks={"A": ticks}, primary_bars=(1, "m"), **_stamp_kw(stamp), **kw)
    return s


def _windowed(bars, ticks, stamp, cut, **kw):
    s = Reader()
    windowed_like([s], {"A": bars}, _cfg(), ticks={"A": ticks}, primary_bars=(1, "m"), cut=cut,
                  **_stamp_kw(stamp), **kw)
    return s


# -- 1. live = batch = windowed --------------------------------------------------------------


@pytest.mark.parametrize("stamp", ["close", "open"])
def test_the_series_is_the_same_live_in_a_batch_run_and_in_windows(stamp):
    bars, ticks = _bars(stamp=stamp), _ticks()
    live, _ = _live(bars, ticks, stamp)
    batch = _batch(bars, ticks, stamp)
    assert live.seen == batch.seen
    for cut in CUTS:
        assert _windowed(bars, ticks, stamp, cut).seen == batch.seen, cut
    # Every bar heard is in the series it reads: the last row, and one more row each time.
    n = len(bars["timestamps"])
    assert len(live.seen) > n + 5, "the prints built bars"
    for k, (bar, series) in enumerate(live.seen):
        assert len(series["close"][2]) == k + 1
        assert series["close"][2][-1] == bar[4] and series["volume"][2][-1] == bar[5]
        assert series["timestamps"][0] == "int64" and series["close"][0] == "float64"
        assert not series["close"][1], "read-only"
    assert list(live.seen[-1][1]) == ["timestamps", "open", "high", "low", "close", "volume"]


@pytest.mark.parametrize("stamp", ["close", "open"])
def test_a_built_bar_is_stamped_as_a_pushed_bar_and_its_on_bar_stamp_is_its_close(stamp):
    """The series stamps a built bar as the caller stamps pushed bars. ``on_bar`` gives a bar
    built from prints its close whatever ``bar_stamp`` says, while a pushed bar keeps the
    caller's stamp: under ``bar_stamp="open"`` the two differ by the bar's width. That predates
    the series and is pinned here, not changed."""
    bars, ticks = _bars(stamp=stamp), _ticks()
    live, _ = _live(bars, ticks, stamp)
    n = len(bars["timestamps"])
    for k, (bar, series) in enumerate(live.seen):
        last = series["timestamps"][2][-1]
        if k < n or stamp == "close":
            assert bar[0] == last
        else:
            assert bar[0] == last + MIN


@pytest.mark.parametrize("stamp", ["close", "open"])
def test_a_symbol_fed_prints_reads_what_the_same_bars_pushed_closed_read(stamp):
    bars, ticks = _bars(stamp=stamp), _ticks()
    fed, _ = _live(bars, ticks, stamp)
    n = len(bars["timestamps"])
    built = [bar for bar, _ in fed.seen[n:]]
    shift = MIN if stamp == "open" else 0
    pushed = Reader()
    node = LiveNode([pushed], ["A"], _cfg(), **_stamp_kw(stamp))
    _feed(node, bars, dict(timestamps=np.array([], dtype=np.int64), ltp=np.array([]), ltq=np.array([])))
    for ts, o, h, low, c, v in built:
        node.push_bar("A", ts - shift, o, h, low, c, v)
    assert pushed.series() == fed.series()


def test_bars_built_before_the_first_pushed_bar_do_not_join_and_ticks_alone_stay_ticks():
    early = _ticks(seed=3, n=20, start=0)
    first = int(early["timestamps"][-1]) // MIN * MIN + 2 * MIN
    later = _ticks(seed=4, n=20, start=first)
    live = Reader()
    node = LiveNode([live], ["A"], _cfg(), primary_bars=(1, "m"), bar_stamp="close")
    for t, p, q in zip(early["timestamps"].tolist(), early["ltp"].tolist(), early["ltq"].tolist()):
        node.push_tick("A", t, p, ltq=q)
    # Fed only prints, the series is the prints, as it always was.
    pushed_at = len(live.seen)
    assert pushed_at > 3
    assert all("ltp" in series and "close" not in series for series in live.series())
    node.push_bar("A", first, 1.0, 2.0, 0.5, 1.5, 3.0)
    for t, p, q in zip(later["timestamps"].tolist(), later["ltp"].tolist(), later["ltq"].tolist()):
        node.push_tick("A", t, p, ltq=q)
    bar = dict(timestamps=np.array([first], dtype=np.int64), open=np.array([1.0]), high=np.array([2.0]),
               low=np.array([0.5]), close=np.array([1.5]), volume=np.array([3.0]))
    batch = _batch(bar, {k: np.concatenate([early[k], later[k]]) for k in early}, "close")
    # From the first pushed bar on, the built bars join it, the same in a batch run.
    assert live.seen[pushed_at][0][0] == first
    assert live.seen[pushed_at:] == batch.seen[pushed_at:]
    assert len(live.seen) > pushed_at + 3
    assert all(series["timestamps"][2][0] == first for series in live.series()[pushed_at:])
    assert [len(series["close"][2]) for series in live.series()[pushed_at:]] == list(
        range(1, len(live.seen) - pushed_at + 1)
    )


# -- 2. history ------------------------------------------------------------------------------


@pytest.mark.parametrize("stamp", ["close", "open"])
def test_a_small_history_shows_the_last_rows_of_the_batch_series(stamp):
    bars, ticks = _bars(stamp=stamp), _ticks()
    batch = _batch(bars, ticks, stamp)
    for history in (1, 3, 7):
        live, _ = _live(bars, ticks, stamp, history=history)
        assert [b for b, _ in live.seen] == [b for b, _ in batch.seen]
        want = [{k: (d, w, v[-history:]) for k, (d, w, v) in series.items()} for series in batch.series()]
        assert live.series() == want, history


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("history", [2, 5])
def test_declared_rows_in_windows_are_the_last_rows_of_the_batch_series(cut, history):
    bars, ticks = _bars(), _ticks()
    batch = _batch(bars, ticks, "close")
    got = _windowed(bars, ticks, "close", cut, history=history)
    want = [{k: (d, w, v[-history:]) for k, (d, w, v) in series.items()} for series in batch.series()]
    assert got.series() == want


class Keeper(Reader):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def on_start(self, ctx):
        ctx.keep_history(self.rows, symbol="A")


def test_a_strategy_keeping_more_than_the_history_has_the_print_refused_before_anything_changes():
    """A strategy keeps 5 rows of a node that holds 4. The built bars keep no row back
    themselves (their ring never refuses, after the print was taken in): every built bar's print
    is a row of the tick history, which holds the same keep at the same size, so the print is
    refused there first, before anything changes, and the node is not poisoned. A pushed bar is
    refused the same way."""
    s = Keeper(5)
    node = LiveNode([s], ["A"], _cfg(), primary_bars=(1, "m"), bar_stamp="close", history=4)
    for m in range(1, 5):
        node.push_bar("A", m * MIN, 1.0, 2.0, 0.5, 1.5, 3.0)
    for m in range(4, 8):
        node.push_tick("A", m * MIN + 10 * NS, 100.0 + m, ltq=1.0)
    assert [b[0] for b, _ in s.seen] == [m * MIN for m in range(1, 8)]
    heard, digest = list(s.seen), node.state_digest()
    shown = node.ctx(0).series("A")["timestamps"].tolist()
    assert shown == [4 * MIN, 5 * MIN, 6 * MIN, 7 * MIN]
    with pytest.raises(StateError) as e:
        node.push_tick("A", 8 * MIN + 10 * NS, 108.0, ltq=1.0)
    assert e.value.code == "history_exhausted"
    assert "tick history is full (4 rows)" in str(e.value)
    assert not node.health()["poisoned"]
    assert s.seen == heard and node.state_digest() == digest
    assert node.ctx(0).series("A")["timestamps"].tolist() == shown
    with pytest.raises(StateError) as e:
        node.push_bar("A", 9 * MIN, 1.0, 2.0, 0.5, 1.5, 3.0)
    assert e.value.code == "history_exhausted"
    assert not node.health()["poisoned"]


def test_keeping_what_fits_never_refuses_a_print():
    s = Keeper(4)
    node = LiveNode([s], ["A"], _cfg(), primary_bars=(1, "m"), bar_stamp="close", history=4)
    for m in range(1, 5):
        node.push_bar("A", m * MIN, 1.0, 2.0, 0.5, 1.5, 3.0)
    for m in range(4, 30):
        node.push_tick("A", m * MIN + 10 * NS, 100.0 + m, ltq=1.0)
    assert node.ctx(0).series("A")["timestamps"].tolist() == [m * MIN for m in range(26, 30)]


# -- 3. restore ------------------------------------------------------------------------------


def _inputs(stamp):
    bars, ticks = _bars(stamp=stamp), _ticks(n=40)
    out = [("bar", int(bars["timestamps"][i]), *(float(bars[k][i]) for k in ("open", "high", "low", "close", "volume")))
           for i in range(len(bars["timestamps"]))]
    for i, (t, p, q) in enumerate(zip(ticks["timestamps"].tolist(), ticks["ltp"].tolist(), ticks["ltq"].tolist())):
        out.append(("tick", t, p, q))
        if i % 7 == 6:
            out.append(("time", t + NS))
    return out


def _push(node, x):
    if x[0] == "bar":
        node.push_bar("A", *x[1:])
    elif x[0] == "tick":
        node.push_tick("A", x[1], x[2], ltq=x[3])
    else:
        node.advance_time(x[1])


@pytest.mark.parametrize("stamp", ["close", "open"])
def test_a_restore_at_every_input_boundary_reads_the_same_series(stamp):
    kw = dict(primary_bars=(1, "m"), history=4, **_stamp_kw(stamp))
    journal = rj.Journal(HEADER, memory=True)
    whole = Reader()
    node = LiveNode([whole], ["A"], _cfg(), journal=journal, **kw)
    inputs, bounds = _inputs(stamp), []
    for x in inputs:
        _push(node, x)
        bounds.append(journal.count)
    records = [r.canonical() for r in journal.since(0)]
    for i, count in enumerate(bounds):
        again = Reader()
        restored = LiveNode.restore(
            None, [again], ["A"], _cfg(), journal=rj.Journal(HEADER, memory=True), records=records[:count], **kw
        )
        for x in inputs[i + 1:]:
            _push(restored, x)
        assert again.seen == whole.seen, i
        assert restored.state_digest() == node.state_digest(), i


# -- 4. exit rules ---------------------------------------------------------------------------


def _probe_events():
    bars, ticks = _bars(), _ticks(n=80)
    out = [("bar", int(bars["timestamps"][i]), float(bars["close"][i])) for i in range(len(bars["timestamps"]))]
    out += [("tick", t, p) for t, p in zip(ticks["timestamps"].tolist(), ticks["ltp"].tolist())]
    return out


@pytest.mark.parametrize("history", [None, 3])
def test_the_fast_mark_is_the_series_mark_live(history):
    s = Probe(live=True)
    node = LiveNode([s], ["A"], _cfg(), primary_bars=(1, "m"), bar_stamp="close", history=history)
    for kind, t, px in _probe_events():
        if kind == "bar":
            node.push_bar("A", t, px, px + 1, px - 1, px, 10.0)
        else:
            node.push_tick("A", t, px, px - 0.05, px + 0.05, ltq=1.0)
    node.finish()
    assert not s.wrong, s.wrong[:5]
    assert s.known > 0


@pytest.mark.parametrize("cut", CUTS)
def test_the_fast_mark_is_the_series_mark_in_windows(cut):
    bars, ticks = _bars(), _ticks(n=80)
    for keep in (None, [("A", 3, "bars")]):
        s = Probe(live=True, keep=keep)
        windowed_like([s], {"A": bars}, _cfg(), ticks={"A": ticks}, primary_bars=(1, "m"), bar_stamp="close", cut=cut)
        assert not s.wrong, (keep, s.wrong[:5])
        assert s.known > 0


def test_a_batch_run_with_built_bars_reads_the_series_itself():
    s = Probe(live=False)
    run_strategies([s], {"A": _bars()}, _cfg(), ticks={"A": _ticks(n=80)}, primary_bars=(1, "m"), bar_stamp="close")
    assert not s.wrong and s.known == 0 and s.asked > 0


def _ruled(seed, *, fast, history=None):
    s = Ruled(seed)
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([s], ["A", "B"], _cfg(), primary_bars=(1, "m"), bar_stamp="close", journal=j, history=history)
    rng = random.Random(seed)
    px = {"A": 100.0, "B": 40.0}
    with pytest.MonkeyPatch.context() as mp:
        if not fast:
            mp.setattr(exits, "RuntimeContext", _Nothing)
        for m in range(1, 6):
            node.push_group(m * MIN, {sym: (px[sym], px[sym] + 1, px[sym] - 1, px[sym], 10.0) for sym in ("A", "B")})
        t = 5 * MIN
        for i in range(400):
            t += rng.randrange(2, 9) * NS
            sym = rng.choice(["A", "B"])
            px[sym] *= 1.0 + rng.uniform(-0.004, 0.004)
            node.push_tick(sym, t, px[sym], px[sym] - 0.01, px[sym] + 0.01, ltq=1.0)
    return s, node, j


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("history", [None, 3])
def test_exit_rules_decide_the_same_with_the_fast_mark(seed, history):
    slow = _ruled(seed, fast=False, history=history)
    fast = _ruled(seed, fast=True, history=history)
    _same(fast, slow)
    assert slow[0].handles


# -- 5. memory -------------------------------------------------------------------------------


def test_a_live_node_holds_its_built_bars_in_a_ring_made_once():
    node = LiveNode([Reader()], ["A"], _cfg(), primary_bars=(1, "m"), bar_stamp="close", history=8)
    for m in range(1, 6):
        node.push_bar("A", m * MIN, 1.0, 2.0, 0.5, 1.5, 3.0)
    node.push_tick("A", 5 * MIN + NS, 100.0, ltq=1.0)
    before = node.memory_report()
    node.push_tick("A", 6 * MIN + NS, 101.0, ltq=1.0)  # the first built bar
    made = node.memory_report()
    grown = made["live_rings"] - before["live_rings"]
    # The rows and the ring's own fields (its rows' box and their window), nothing per bar.
    assert 8 * BUILT_ROW <= grown <= 8 * BUILT_ROW + 128, grown
    for m in range(7, 60):
        node.push_tick("A", m * MIN + NS, 100.0 + m, ltq=1.0)
    after = node.memory_report()
    assert after["live_rings"] == made["live_rings"] and after["built_bars"] == 0


def test_a_symbol_fed_only_ticks_makes_no_built_bar_ring():
    node = LiveNode([Reader()], ["A"], _cfg(), primary_bars=(1, "m"), bar_stamp="close", history=8)
    node.push_tick("A", NS, 100.0, ltq=1.0)
    before = node.memory_report()["live_rings"]
    for m in range(1, 60):
        node.push_tick("A", m * MIN + NS, 100.0 + m, ltq=1.0)
    assert node.memory_report()["live_rings"] == before


def _category_total(report):
    return sum(report[k] for k in ("shared_store", "aggregators", "indicators", "index", "atr",
                                   "engine_per_strategy", "portfolio_per_strategy", "curves",
                                   "live_rings", "built_bars"))


def test_a_batch_run_counts_its_built_bars_exactly():
    bars, ticks = _bars(), _ticks()
    s = Reader()
    _, report = run_strategies([s], {"A": bars}, _cfg(), ticks={"A": ticks}, primary_bars=(1, "m"),
                               bar_stamp="close", memory_report=True)
    built = len(s.seen) - len(bars["timestamps"])
    assert built > 5
    capacity = max(4, 1 << (built - 1).bit_length())
    assert report["built_bars"] == capacity * BUILT_ROW
    assert report["live_rings"] == 0
    assert report["total"] == _category_total(report)
    _, plain = run_strategies([Reader()], {"A": bars}, _cfg(), bar_stamp="close", memory_report=True)
    assert plain["built_bars"] == 0 and plain["total"] == _category_total(plain)


def test_a_windowed_run_holds_its_built_bars_in_its_rings():
    bars, ticks = _bars(), _ticks()
    _, report = windowed_like([Reader()], {"A": bars}, _cfg(), ticks={"A": ticks}, primary_bars=(1, "m"),
                              bar_stamp="close", cut=("random", 4), memory_report=True)
    _, without = windowed_like([Reader()], {"A": bars}, _cfg(), ticks={"A": ticks}, cut=("random", 4),
                               bar_stamp="close", memory_report=True)
    assert report["built_bars"] == without["built_bars"] == 0
    assert report["live_rings"] > without["live_rings"]
