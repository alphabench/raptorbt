"""Exit rules mark a symbol at the last ``close``, else the last traded ``ltp`` (``> 0``: a
quote-only row is no trade), of ``ctx.series``.

A runtime context answers that price without building the series (``ctx._last_price``), which a
live node would otherwise copy out of its ring on every new row. These tests hold the answer to
the series, bit for bit, wherever it is given, in live nodes (bars, ticks, both, a ring that
wraps) and windowed runs (declared history); and hold runs of exit rules with it to the same
runs without it: the same orders, hooks, handles, state and journal bytes, also across a restore
taken while positions and rules are open.
"""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt import journal as rj
from raptorbt.strategy import exits as ex
from raptorbt.strategy import orders
from raptorbt.strategy.exits import ExitEvents, ExitRule
from raptorbt.strategy.tick_runner import run_tick_strategy

from test_journal import HEADER
from test_windowed import windowed_like

NS = 1_000_000_000
T0 = 1_767_589_800 * NS
CFG = BacktestConfig(initial_capital=1e7, fees=0.0)


def _series_mark(ctx, sym):
    """How exit rules read a mark from the series."""
    try:
        series = ctx.series(sym)
    except (KeyError, RuntimeError):
        return None, False
    values = series.get("close")
    if values is not None and len(values):
        return float(values[-1]), True
    traded = [float(v) for v in series.get("ltp", ()) if v > 0.0]
    return (traded[-1] if traded else None), True


def _bits(x):
    return None if x is None else struct.pack("<d", x)


class Probe(Strategy):
    """Asks for the fast mark, then reads the series, for every symbol at every event."""

    def __init__(self, live, keep=None):
        super().__init__()
        self.live, self.keep = live, keep
        self.known = self.asked = 0
        self.wrong: list = []

    def on_start(self, ctx):
        for sym, rows, kind in self.keep or ():
            ctx.keep_history(rows, symbol=sym, kind=kind)
        self._check(ctx, "start")

    def on_bar(self, ctx, bar=None):
        self._check(ctx, "bar")

    def on_trade(self, ctx, tick=None):
        self._check(ctx, "trade")

    def on_quote(self, ctx, quote=None):
        self._check(ctx, "quote")

    def _check(self, ctx, where):
        for sym in ctx.symbols:
            self.asked += 1
            known, price = ctx._last_price(sym)
            want, built = _series_mark(ctx, sym)
            if not self.live and known:
                self.wrong.append((where, sym, "answered in a batch run"))
            if known:
                self.known += 1
                if _bits(price) != _bits(want):
                    self.wrong.append((where, sym, ctx.event_ns, price, want))
            if built and self.live and ctx._last_price(sym)[0]:
                # The series of these rows is cached now: it is read as it is, never bypassed.
                self.wrong.append((where, sym, ctx.event_ns, "answered over a cached series"))


def _walk(rng, n, start):
    out, px = [], start
    for _ in range(n):
        px *= 1.0 + rng.uniform(-0.004, 0.004)
        out.append(px)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("feed", ["bars", "ticks", "both", "small_ring"])
def test_the_fast_mark_is_the_series_mark_in_a_live_node(feed, seed):
    rng = random.Random(seed)
    symbols = ["A", "B"]
    s = Probe(live=True)
    node = LiveNode([s], symbols, CFG, bar_stamp="close", history=4 if feed == "small_ring" else None)
    walks = {sym: _walk(rng, 60, 100.0 * (k + 1)) for k, sym in enumerate(symbols)}
    for i in range(60):
        for k, sym in enumerate(symbols):
            ts = T0 + (i * 60 + k + 1) * NS
            px = walks[sym][i]
            tick = feed in ("ticks", "small_ring") or (feed == "both" and (sym == "B" or i % 3))
            bar = feed in ("bars", "small_ring") or (feed == "both" and sym == "A" and i % 5 == 4)
            if tick:
                node.push_tick(sym, ts, px, px - 0.05, px + 0.05)
            if bar:
                node.push_bar(sym, ts + NS // 2, px, px * 1.001, px * 0.999, px * (1 + rng.uniform(-1e-3, 1e-3)), 10.0)
    node.finish()
    assert not s.wrong, s.wrong[:5]
    assert s.known > 0


@pytest.mark.parametrize("cut", [("whole", 0), ("random", 1), ("random", 4)])
def test_the_fast_mark_is_the_series_mark_in_a_windowed_run_with_declared_history(cut):
    rng = np.random.default_rng(7)
    n = 400
    ts = T0 + np.cumsum(rng.integers(1, 4, n)) * NS
    ltp_a = 100 + np.cumsum(rng.normal(0, 0.1, n))
    ltp_b = 50 + np.cumsum(rng.normal(0, 0.05, n))
    ticks = {
        "A": {"timestamps": ts, "ltp": ltp_a, "bid": ltp_a - 0.05, "ask": ltp_a + 0.05},
        "B": {"timestamps": ts + NS // 2, "ltp": ltp_b},
    }
    m = n // 20
    bts = T0 + np.arange(1, m + 1, dtype=np.int64) * 60 * NS
    close = 100 + np.cumsum(rng.normal(0, 0.3, m))
    data = {"A": {"timestamps": bts, "open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
                  "volume": np.full(m, 10.0)}}
    for keep in (None, [("A", 3, "bars"), ("B", 2, "ticks")]):
        s = Probe(live=True, keep=keep)
        windowed_like([s], data, CFG, ticks=ticks, cut=cut, bar_stamp="close")
        assert not s.wrong, (keep, s.wrong[:5])
        assert s.known > 0


def test_a_batch_run_reads_the_series_itself():
    rng = random.Random(3)
    closes = np.array(_walk(rng, 50, 100.0))
    ts = T0 + np.arange(1, 51, dtype=np.int64) * 60 * NS
    data = {"A": {"timestamps": ts, "open": closes, "high": closes + 0.1, "low": closes - 0.1, "close": closes,
                  "volume": np.full(50, 10.0)}}
    s = Probe(live=False)
    run_strategies([s], data, CFG, bar_stamp="close")
    assert not s.wrong and s.known == 0 and s.asked > 0


# -- runs of exit rules, with the fast mark and without it ----------------------------------------


def _rule(rng, premium):
    kinds = rng.sample(["stop_loss_pct", "take_profit_pct", "stop_loss", "take_profit", "above", "below"],
                       rng.randint(1, 3))
    kw = {}
    for k in kinds:
        if k == "stop_loss_pct":
            kw[k] = rng.uniform(0.2, 2.5)
        elif k == "take_profit_pct":
            kw[k] = rng.uniform(0.2, 2.5)
        elif k in ("stop_loss", "take_profit"):
            kw[k] = rng.uniform(2.0, 60.0)
        elif k == "above":
            kw["premium_at_or_above"] = premium * rng.uniform(1.002, 1.03)
        else:
            kw["premium_at_or_below"] = premium * rng.uniform(0.97, 0.998)
    return ExitRule(**kw)


class Ruled(ExitEvents, Strategy):
    """Buys both symbols, sets seeded rules on one symbol or both, changes them while the position
    is open, and buys again once flat. Everything it decides comes from its seed and the event
    count, so a replay decides the same."""

    def __init__(self, seed):
        super().__init__()
        self.rng = random.Random(seed)
        self.n = 0
        self.handles: list = []
        self.trail: list = []

    def on_bar(self, ctx, bar):
        self._act(ctx, "bar")

    def on_trade(self, ctx, tick):
        self._act(ctx, "trade")

    def on_order_event(self, ctx, event):
        self.trail.append(("order", event.kind, event.client_order_id))

    def _act(self, ctx, where):
        self.n += 1
        self.trail.append((where, ctx.symbol, ctx.event_ns, tuple(h.state for h in self.handles)))
        rng = self.rng
        if self.n % 40 == 1:
            for sym in ctx.symbols:
                if not ctx.positions(sym):
                    self.submit_order(orders.Market(side="buy", units=float(rng.randint(1, 20)), instrument=sym))
        elif self.n % 40 in (6, 23) and all(ctx.positions(s) for s in ctx.symbols):
            if self.n % 40 == 23 and self.handles and rng.random() < 0.6:
                # A strategy moves its stop: the old rule goes, a new one comes.
                for h in self.handles:
                    if h.state == "armed":
                        self.cancel_exit(h)
            for _ in range(rng.randint(1, 4)):
                group = rng.random() < 0.4
                syms = list(ctx.symbols) if group else [rng.choice(ctx.symbols)]
                premium = sum(p.direction * float(ctx.series(s)["close" if "close" in ctx.series(s) else "ltp"][-1])
                              * p.size for s in syms for p in ctx.positions(s))
                rule = _rule(rng, premium)
                if group:
                    self.handles.append(self.set_exit(ctx, rule, symbols=syms, name=f"g{self.n}"))
                else:
                    self.handles.append(self.set_exit(ctx, rule, symbol=syms[0]))
        elif self.n % 40 == 31 and rng.random() < 0.5:
            sym = rng.choice(ctx.symbols)
            if ctx.positions(sym):
                self.submit_order(orders.Market(side="buy", units=float(rng.randint(1, 5)), instrument=sym))

    def summary(self):
        return [(h.id, h.name, h.state, h.reason, tuple(h.close_ids), h.attempts, tuple(h.refused))
                for h in self.handles]


def _events(seed, feed, n=400):
    rng = random.Random(seed * 101)
    walks = {"A": _walk(rng, n, 100.0), "B": _walk(rng, n, 37.5)}
    out = []
    for i in range(n):
        for k, sym in enumerate(("A", "B")):
            ts = T0 + (i * 10 + k + 1) * NS
            px = walks[sym][i]
            if feed == "ticks":
                out.append(("tick", sym, ts, px))
            else:
                out.append(("bar", sym, ts, px))
    return out


def _push(node, event):
    kind, sym, ts, px = event
    if kind == "tick":
        node.push_tick(sym, ts, px, px - 0.01, px + 0.01)
    else:
        node.push_bar(sym, ts, px, px * 1.001, px * 0.999, px, 100.0)


class _Nothing:
    """Matches no context, so exit rules read the series themselves."""


def _run(seed, feed, *, fast, events, history=None):
    s = Ruled(seed)
    j = rj.Journal(HEADER, memory=True)
    node = LiveNode([s], ["A", "B"], CFG, bar_stamp="close", journal=j, history=history)
    with pytest.MonkeyPatch.context() as mp:
        if not fast:
            mp.setattr(ex, "RuntimeContext", _Nothing)
        for e in events:
            _push(node, e)
    return s, node, j


def _same(a, b, *, hooks=True):
    (sa, na, ja), (sb, nb, jb) = a, b
    if hooks:
        assert sa.trail == sb.trail
    assert sa.summary() == sb.summary()
    assert na.state_digest() == nb.state_digest()
    assert [r.canonical() for r in ja.since(0)] == [r.canonical() for r in jb.since(0)]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("feed", ["ticks", "bars"])
def test_exit_rules_decide_the_same_with_the_fast_mark(feed, seed):
    events = _events(seed, feed)
    slow = _run(seed, feed, fast=False, events=events)
    fast = _run(seed, feed, fast=True, events=events)
    _same(fast, slow)
    assert slow[0].handles


def test_the_seeded_rules_fire_and_stay_armed_both():
    # The comparison above means something only if rules fire and rules wait.
    fired = armed = 0
    for seed in range(8):
        for feed in ("ticks", "bars"):
            s, _, _ = _run(seed, feed, fast=True, events=_events(seed, feed))
            fired += sum(h.attempts > 0 for h in s.handles)
            armed += sum(h.state == "armed" for h in s.handles)
    assert fired > 0 and armed > 0


@pytest.mark.parametrize("seed", range(4))
def test_exit_rules_decide_the_same_in_a_ring_that_wraps(seed):
    events = _events(seed, "ticks", n=200)
    _same(_run(seed, "ticks", fast=True, events=events, history=3),
          _run(seed, "ticks", fast=False, events=events, history=3))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("feed", ["ticks", "bars"])
def test_a_restore_with_rules_open_replays_and_continues_the_same(feed, seed):
    events = _events(seed, feed)
    whole = _run(seed, feed, fast=False, events=events)
    # Cut at the first event after which a rule is armed over open positions.
    s0 = whole[0]
    first = next(i for i, e in enumerate(s0.trail) if e[0] != "order" and "armed" in e[3])
    first = sum(1 for e in s0.trail[:first] if e[0] != "order")
    for cut in range(first, len(events)):
        head = _run(seed, feed, fast=False, events=events[:cut])
        if any(h.state == "armed" for h in head[0].handles):
            break
    else:
        pytest.fail("no rule is ever armed")
    records = list(head[2].since(0))
    s = Ruled(seed)
    j = rj.Journal(HEADER, memory=True)
    # The replay must write every record the journal holds (else ``replay_diverged``).
    node = LiveNode.restore(None, [s], ["A", "B"], CFG, journal=j, records=records, bar_stamp="close")
    assert [r.canonical() for r in j.since(0)] == [r.canonical() for r in records]
    for e in events[cut:]:
        _push(node, e)
    # The hook log is not compared: what a restore replays to the strategy is the restore's own
    # contract, and the slow-path run ``whole`` is compared hook for hook above.
    _same((s, node, j), whole, hooks=False)


# -- a quote-only row is no price -----------------------------------------------------------------
#
# A live or batch tick row with ``ltp`` 0 and a two-sided quote is a quote and no trade. Before the
# fix the rules marked it at 0.0 (the fast mark and the series mark alike), so a long's stop fired on
# a price that never traded.


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("history", [None, 3])
def test_the_fast_mark_is_the_series_mark_over_quote_only_rows(history, seed):
    rng = random.Random(seed)
    s = Probe(live=True)
    node = LiveNode([s], ["A", "B"], CFG, bar_stamp="close", history=history)
    walks = {"A": _walk(rng, 80, 100.0), "B": _walk(rng, 80, 40.0)}
    for i in range(80):
        for k, sym in enumerate(("A", "B")):
            ts = T0 + (i * 60 + k + 1) * NS
            px = walks[sym][i]
            # B quotes only for its first rows; then both quote without a print, now and then in runs.
            quote_only = (sym == "B" and i < 6) or rng.random() < 0.35
            node.push_tick(sym, ts, 0.0 if quote_only else px, px - 0.05, px + 0.05)
    node.finish()
    assert not s.wrong, s.wrong[:5]
    assert s.known > 0


@pytest.mark.parametrize("cut", [("whole", 0), ("random", 4)])
def test_the_fast_mark_is_the_series_mark_over_quote_only_rows_in_a_windowed_run(cut):
    rng = np.random.default_rng(11)
    n = 300
    ts = T0 + np.cumsum(rng.integers(1, 4, n)) * NS
    px = 100 + np.cumsum(rng.normal(0, 0.1, n))
    ltp = np.where(rng.random(n) < 0.35, 0.0, px)
    ltp[:5] = 0.0
    ticks = {"A": {"timestamps": ts, "ltp": ltp, "bid": px - 0.05, "ask": px + 0.05}}
    for keep in (None, [("A", 2, "ticks")]):
        s = Probe(live=True, keep=keep)
        windowed_like([s], None, CFG, ticks=ticks, cut=cut)
        assert not s.wrong, (keep, s.wrong[:5])
        assert s.known > 0


class StopOnQuotes(Strategy):
    """Buys 10 of AAA, sets a 5% stop once filled, and evaluates its rules on every trade and every
    quote. ``trail`` records each event's kind, time, the mark the rule reads and the rule state."""

    def __init__(self):
        super().__init__()
        self.handle = None
        self.trail: list = []

    def on_trade(self, ctx, tick=None):
        if ctx.symbol == "AAA":
            if self.handle is None and not ctx.positions("AAA") and not self.trail:
                self.submit_order(orders.Market(side="buy", units=10.0, instrument="AAA"))
            elif self.handle is None and ctx.positions("AAA"):
                self.handle = self.set_exit(ctx, ExitRule(stop_loss_pct=5), symbol="AAA")
        self._step(ctx, "trade")

    def on_quote(self, ctx, quote=None):
        self._step(ctx, "quote")

    def _step(self, ctx, kind):
        self.update_exits(ctx)
        mark = ex.ExitBook._mark(ctx, "AAA", None)
        self.trail.append((kind, ctx.symbol, ctx.event_ns, mark, self.handle.state if self.handle else None))

    def fired_at(self):
        return next(ts for kind, _, ts, _, state in self.trail if state not in (None, "armed"))


# The tape: three prints at 100, a quote-only row, a print at 96 (above the stop), another quote-only
# row, then a print at 94 (below it). A quote-only row has ``ltp`` 0.
TAPE = [(100.0, 99.95, 100.05), (100.0, 99.95, 100.05), (100.0, 99.95, 100.05), (0.0, 99.9, 100.1),
        (96.0, 95.95, 96.05), (0.0, 50.0, 96.1), (94.0, 93.95, 94.05), (94.0, 93.95, 94.05)]
TAPE_TS = [T0 + (i + 1) * NS for i in range(len(TAPE))]


def _tape_live(fast):
    s = StopOnQuotes()
    node = LiveNode([s], ["AAA"], CFG, bar_stamp="close")
    with pytest.MonkeyPatch.context() as mp:
        if not fast:
            mp.setattr(ex, "RuntimeContext", _Nothing)
        for ts, (ltp, bid, ask) in zip(TAPE_TS, TAPE):
            node.push_tick("AAA", ts, ltp, bid, ask)
    return s


def _tape_ticks():
    cols = np.array(TAPE)
    return {"AAA": {"timestamps": np.array(TAPE_TS, dtype=np.int64), "ltp": cols[:, 0], "bid": cols[:, 1],
                    "ask": cols[:, 2]}}


def _assert_stopped_at_the_trade(s):
    assert s.handle is not None
    # Every quote-only row reads the last trade, never 0.0, and fires nothing.
    quotes = [(ts, mark, state) for kind, _, ts, mark, state in s.trail
              if kind == "quote" and ts in (TAPE_TS[3], TAPE_TS[5])]
    assert [m for _, m, _ in quotes] == [100.0, 96.0], quotes
    assert all(state == "armed" for _, _, state in quotes), quotes
    # The stop fires at the print below it, and for the reason it was set.
    assert s.fired_at() == TAPE_TS[6]
    assert s.handle.reason == "stop_loss_pct"


def test_a_quote_only_tick_does_not_fire_a_stop_on_a_live_node():
    fast, slow = _tape_live(fast=True), _tape_live(fast=False)
    _assert_stopped_at_the_trade(fast)
    _assert_stopped_at_the_trade(slow)
    assert fast.trail == slow.trail


@pytest.mark.parametrize("run", ["run_tick_strategy", "run_strategies", "windowed"])
def test_a_quote_only_tick_does_not_fire_a_stop_in_a_batch_or_windowed_run(run):
    s = StopOnQuotes()
    if run == "run_tick_strategy":
        run_tick_strategy(s, _tape_ticks(), CFG)
    elif run == "run_strategies":
        run_strategies([s], {}, CFG, ticks=_tape_ticks())
    else:
        windowed_like([s], None, CFG, ticks=_tape_ticks(), cut=("random", 3))
    _assert_stopped_at_the_trade(s)


@pytest.mark.parametrize("fast", [True, False])
def test_a_symbol_with_only_quotes_so_far_has_no_mark(fast):
    seen = []

    class Watch(Strategy):
        def on_quote(self, ctx, quote=None):
            # The fast answer is asked only on the fast path: reading the series caches it.
            asked = ctx._last_price("BBB") if fast else None
            seen.append((ctx.symbol, asked, ex.ExitBook._mark(ctx, "BBB", None)))

    node = LiveNode([Watch()], ["AAA", "BBB"], CFG, bar_stamp="close")
    with pytest.MonkeyPatch.context() as mp:
        if not fast:
            mp.setattr(ex, "RuntimeContext", _Nothing)
        node.push_tick("BBB", T0 + NS, 0.0, 9.9, 10.1)
        node.push_tick("AAA", T0 + 2 * NS, 100.0, 99.9, 100.1)
        node.push_tick("BBB", T0 + 3 * NS, 0.0, 9.8, 10.2)
        none = (True, None) if fast else None
        assert seen == [("BBB", none, None), ("AAA", none, None), ("BBB", none, None)]
        node.push_tick("BBB", T0 + 4 * NS, 10.05, 10.0, 10.1)
    assert seen[-1] == ("BBB", (True, 10.05) if fast else None, 10.05)


@pytest.mark.parametrize("fast", [True, False])
def test_a_rule_whose_shown_rows_are_all_quotes_waits_for_the_next_trade(fast):
    # With two rows shown, two quote-only rows after the last print leave no trade in view: the
    # rule has no mark and does not fire (before the fix it read 0.0 and stopped the long out).
    s = StopOnQuotes()
    node = LiveNode([s], ["AAA"], CFG, bar_stamp="close", history=2)
    rows = [(100.0, 99.95, 100.05)] * 3 + [(0.0, 50.0, 100.1)] * 3 + [(94.0, 93.95, 94.05)] * 2
    stamps = [T0 + (i + 1) * NS for i in range(len(rows))]
    with pytest.MonkeyPatch.context() as mp:
        if not fast:
            mp.setattr(ex, "RuntimeContext", _Nothing)
        for ts, (ltp, bid, ask) in zip(stamps, rows):
            node.push_tick("AAA", ts, ltp, bid, ask)
    marks = [mark for kind, _, ts, mark, state in s.trail if kind == "quote" and ts in stamps[4:6]]
    assert marks == [None, None]
    assert s.fired_at() == stamps[6]


class Trailing(Strategy):
    """Buys 10 of AAA, then rests a trailing stop 2.0 under the high; records fills and equity."""

    def __init__(self):
        super().__init__()
        self.n = 0
        self.fills: list = []
        self.equity: list = []

    def on_trade(self, ctx, tick=None):
        self.n += 1
        if self.n == 1:
            self.submit_order(orders.Market(side="buy", units=10.0, instrument="AAA"))
        elif self.n == 3:
            self.submit_order(orders.TrailingStopMarket(side="sell", offset=2.0, units=10.0, instrument="AAA"))

    def on_quote(self, ctx, quote=None):
        self.equity.append((ctx.event_ns, ctx.equity))

    def on_order_event(self, ctx, event):
        if event.kind == "order_filled":
            self.fills.append((ctx.event_ns, event.price))


@pytest.mark.parametrize("mode", ["live", "batch"])
def test_a_quote_only_tick_moves_no_trailing_stop_and_marks_no_equity(mode):
    # The stop trails to 99.0 under the 101 high. A quote-only row with its bid at 49.95 neither
    # triggers it nor marks equity; the print at 98 does.
    ltp = np.array([100.0, 100.0, 100.0, 101.0, 0.0, 101.0, 98.0])
    ref = np.where(ltp > 0, ltp, 50.0)
    stamps = T0 + np.arange(1, len(ltp) + 1, dtype=np.int64) * NS
    s = Trailing()
    if mode == "live":
        node = LiveNode([s], ["AAA"], CFG, bar_stamp="close")
        for ts, p, r in zip(stamps, ltp, ref):
            node.push_tick("AAA", int(ts), float(p), float(r) - 0.05, float(r) + 0.05)
    else:
        run_tick_strategy(s, {"AAA": {"timestamps": stamps, "ltp": ltp, "bid": ref - 0.05, "ask": ref + 0.05}}, CFG)
    assert [px for _, px in s.fills][-1] == 98.0
    assert s.fills[-1][0] == int(stamps[6])
    equity = dict(s.equity)
    assert equity[int(stamps[4])] == equity[int(stamps[3])]
