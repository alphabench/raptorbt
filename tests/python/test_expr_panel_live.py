"""A cross-sectional expression on the shared indicator bank (``ctx.indicators.expr`` with
``rank(roc(close, 20))``), in a backtest and on a live node.

The expression compares the strategy's instruments, so it steps once per timestamp group, after
every bar of the group is in and before any strategy reveals a bar. A member's handle then follows
the group rule of an instrument-set indicator: the value before the group until the strategy
reveals a bar of the universe, the value after the whole group once it has. The numbers are those
of ``Expr.eval_panel`` over the same bars, bit for bit.

* batch (``run_strategies``) and node (``LiveNode.push_group``) agree with ``eval_panel`` at every
  date, absent members and thin cross-sections included;
* the order instruments arrive in changes nothing;
* a poisoned future does not move the past;
* a restored node continues as an uninterrupted one;
* the universe is the strategy's instruments at build and is frozen;
* runners without the shared runtime refuse the expression, typed;
* a strategy with no cross-sectional expression is unmoved (the seal is not run).
"""

import time

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, Indicator, InstrumentSpec, LiveNode, Strategy, run_strategies, run_strategy_backtest
from raptorbt import journal as rj
from raptorbt.errors import RaptorBTError
from raptorbt.strategy import orders as O

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
OHLCV = ("open", "high", "low", "close", "volume")
TEXT = "rank(roc(close, 5))"
SYMS = [f"S{i}" for i in range(6)]


def one(n, seed):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return {
        "timestamps": T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
        "volume": np.round(rng.uniform(500.0, 1500.0, n)),
    }


def universe(n=40, syms=SYMS, absent=()):
    """Bars of every symbol on one grid, minus the (symbol, index) pairs in `absent`."""
    data = {}
    for k, s in enumerate(syms):
        b = one(n, 10 + k)
        keep = np.array([(s, i) not in absent for i in range(n)])
        data[s] = {c: v[keep] for c, v in b.items()}
    return data


def config(**kw):
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0, **kw)


def groups(data):
    """(timestamp, {symbol: bar}) per timestamp, in time order."""
    stamps = sorted({int(t) for d in data.values() for t in d["timestamps"]})
    index = {s: {int(t): i for i, t in enumerate(d["timestamps"])} for s, d in data.items()}
    for t in stamps:
        yield t, {
            s: tuple(float(data[s][c][index[s][t]]) for c in OHLCV)
            for s in data
            if t in index[s]
        }


def node_for(strategies, data, order=None, **kw):
    return LiveNode(strategies, list(order or data), config(), bar_stamp="close", **kw)


def feed(node, data, lo=0, hi=None, shuffle=None):
    gs = list(groups(data))[lo:hi]
    for t, g in gs:
        items = list(g.items())
        if shuffle is not None:
            shuffle.shuffle(items)
        node.push_group(t, dict(items))


def run(strategy, data, mode="batch", **kw):
    if mode == "batch":
        return run_strategies([strategy], data, config(**kw), bar_stamp="close")[0]
    node = LiveNode([strategy], list(data), config(**kw), bar_stamp="close")
    feed(node, data)
    return node.finish()[0]


class Rank(Strategy):
    """Registers TEXT for every instrument and keeps what every handle shows in every hook."""

    def __init__(self, text=TEXT):
        super().__init__()
        self.text = text
        self.rows = []  # (timestamp, symbol, {member: (ready, value)})

    def on_start(self, ctx):
        self.h = ctx.indicators.expr_all(self.text)

    def on_bar(self, ctx):
        self.rows.append(
            (ctx.event_ns, ctx.symbol, {s: (h.ready, h.value) for s, h in self.h.items()})
        )


def expected(data, text=TEXT, syms=None):
    """eval_panel over the same bars: the stamps of the rows and the (dates, names) panel."""
    syms = list(syms or data)
    stamps = np.unique(np.concatenate([data[s]["timestamps"] for s in syms]))
    return stamps, rb.expr(text).eval_panel(data, instruments=syms)


def check_against_panel(rows, data, text=TEXT, syms=None):
    syms = list(syms or data)
    stamps, want = expected(data, text, syms)
    at = {int(t): i for i, t in enumerate(stamps)}
    program = rb.expr(text)
    seen = set()
    for t, symbol, values in rows:
        d = at[int(t)]
        seen.add(d)
        for j, s in enumerate(syms):
            ready, value = values[s]
            w = want[d, j]
            if np.isnan(w):
                assert value is None, (d, s, symbol, value)
            else:
                assert ready and value == w, (d, s, symbol, value, w)
    assert len(seen) > 20, "too few dates were compared"
    return stamps, want


@pytest.mark.parametrize("mode", ["batch", "node"])
def test_rank_on_the_bank_equals_eval_panel_at_every_date(mode):
    data = universe()
    s = Rank()
    run(s, data, mode)
    assert len(s.rows) == sum(len(d["close"]) for d in data.values())
    stamps, want = check_against_panel(s.rows, data)
    # Warm-up: roc(close, 5) is unknown for the first 5 bars, so is its rank.
    assert np.isnan(want[:5]).all() and not np.isnan(want[5:]).any()
    # Every instrument of a date reads the same row (post-group), whichever hook it is in.
    by_date = {}
    for t, _, values in s.rows:
        by_date.setdefault(t, []).append(values)
    for t, vs in by_date.items():
        assert all(v == vs[0] for v in vs), t


@pytest.mark.parametrize("mode", ["batch", "node"])
def test_a_member_with_no_bar_at_a_date_reads_unknown_and_thins_the_cross_section(mode):
    absent = {("S2", 20), ("S2", 21), ("S4", 25)}
    data = universe(absent=absent)
    s = Rank()
    run(s, data, mode)
    stamps, want = check_against_panel(s.rows, data)
    j2 = SYMS.index("S2")
    assert np.isnan(want[20, j2]) and np.isnan(want[21, j2])  # absent at the seal: unknown
    # The other five were ranked among themselves: 0 .. 1 over five names.
    assert np.nanmin(want[20]) == 0.0 and np.nanmax(want[20]) == 1.0
    # A date where fewer than min_names (2) members have a bar is unknown for all of them.
    thin = universe(absent={(s_, 30) for s_ in SYMS[1:]})
    t = Rank()
    run(t, thin, mode)
    _, w = check_against_panel(t.rows, thin)
    assert np.isnan(w[30]).all()


@pytest.mark.parametrize("mode", ["batch", "node"])
def test_before_the_reveal_a_hook_sees_the_previous_date_and_after_it_the_whole_group(mode):
    class Probe(Strategy):
        def __init__(self):
            super().__init__()
            self.seen, self.at_hook = {}, None

        def on_start(self, ctx):
            self.h = ctx.indicators.expr_all(TEXT)

        def now(self):
            return {s: h.value for s, h in self.h.items()}

        def on_bar(self, ctx):
            if ctx.symbol != "S0":
                return
            self.seen[ctx.event_ns] = self.now()
            if len(self.seen) == 12:
                self.submit_order(O.Market(side="buy", units=1.0), client_id="probe", symbol="S0")

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled" and e.client_order_id == "probe":
                self.at_hook = (max(self.seen), self.now())

    data = universe()
    s = Probe()
    run(s, data, mode, fill_timing="next_bar_open")
    assert s.at_hook is not None, "the order never filled"
    last, shown = s.at_hook
    # The fill is heard at the next bar's open, before the strategy reveals that bar: the whole
    # universe shows the date before, not a mixture.
    assert shown == s.seen[last]
    stamps = sorted(s.seen)
    assert s.seen[stamps[-1]] != s.seen[stamps[-2]], "the check proves nothing if nothing moved"
    _, want = expected(data)
    row = want[int(np.searchsorted(np.unique(data["S0"]["timestamps"]), last))]
    assert [shown[k] for k in SYMS] == [None if np.isnan(w) else w for w in row]


def test_the_order_instruments_arrive_in_changes_nothing():
    data = universe(absent={("S3", 22)})
    a = Rank()
    node = node_for([a], data)
    feed(node, data)
    node.finish()
    b = Rank()
    node = node_for([b], data)
    feed(node, data, shuffle=np.random.default_rng(5))
    node.finish()
    values = lambda s: sorted(((t, sym, tuple(sorted(v.items()))) for t, sym, v in s.rows), key=repr)
    assert values(a) == values(b)
    # And the same as the run that names its symbols in another order.
    c = Rank()
    run_strategies([c], dict(reversed(list(data.items()))), config(), bar_stamp="close")
    assert values(a) == values(c)


def test_a_poisoned_future_does_not_move_the_past():
    data = universe(n=50)
    full = Rank()
    run(full, data, "node")
    for cut in (12, 30):
        poisoned = {
            s: {
                c: np.concatenate(
                    [v[: cut + 1], np.full(len(v) - cut - 1, 1e9 if c != "timestamps" else 0)]
                ).astype(v.dtype)
                if c != "timestamps"
                else v
                for c, v in d.items()
            }
            for s, d in data.items()
        }
        p = Rank()
        run(p, poisoned, "node")
        head = lambda s: sorted(
            ((t, sym, tuple(sorted(v.items()))) for t, sym, v in s.rows if t <= int(T0 + (cut + 1) * MIN)),
            key=repr,
        )
        assert head(full) == head(p), cut


def test_a_restored_node_gives_the_values_of_an_uninterrupted_one():
    data = universe(n=40, absent={("S1", 18)})
    n = len(list(groups(data)))
    for cut in (1, 15, 19, n - 1):
        journal = rj.Journal({"run": "panel"}, memory=True)
        before = Rank()
        node = node_for([before], data, journal=journal)
        feed(node, data, 0, cut)
        records = [r.canonical() for r in journal.since(0)]
        after = Rank()
        restored = LiveNode.restore(
            None, [after], list(data), config(), journal=rj.Journal({"run": "panel"}, memory=True),
            records=records, bar_stamp="close",
        )
        assert after.rows == before.rows, cut
        assert restored.state_digest() == node.state_digest(), cut
        feed(node, data, cut, n)
        feed(restored, data, cut, n)
        assert after.rows == before.rows, cut
    check_against_panel(after.rows, data)


def test_the_universe_is_frozen_when_the_run_is_built():
    class Joiner(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_start(self, ctx):
            self.h = ctx.indicators.expr("rank(close)", instrument="S0")

        def on_bar(self, ctx):
            self.rows.append((ctx.event_ns, ctx.symbol, self.h.value))

    data = universe(n=20, syms=["S0", "S1", "S2"])
    s = Joiner()
    node = LiveNode([(s, ["S0", "S1"])], ["S0", "S1"], config(), bar_stamp="close")
    groups_ = list(groups(data))
    for t, g in groups_[:8]:
        node.push_group(t, {k: v for k, v in g.items() if k != "S2"})
    node.add_instrument("S2", InstrumentSpec.equity("S2"), strategies=[0])
    for t, g in groups_[8:]:
        node.push_group(t, g)
    node.finish()
    two = {s_: data[s_] for s_ in ("S0", "S1")}
    stamps, want = expected(two, "rank(close)")
    at = {int(t): i for i, t in enumerate(stamps)}
    for t, sym, value in s.rows:
        if sym == "S0":
            w = want[at[t], 0]
            assert (value is None and np.isnan(w)) or value == w, (t, value, w)


def test_a_strategy_without_a_cross_sectional_expression_is_unmoved():
    class Cross(Strategy):
        def __init__(self):
            super().__init__()
            self.seen = []

        def on_start(self, ctx):
            self.f = ctx.indicators.sma(instrument="S0", period=3)

        def on_bar(self, ctx):
            self.seen.append((ctx.symbol, ctx.event_ns, self.f.value, ctx.equity))

    data = universe()
    a, b = Cross(), Cross()
    run(a, data)
    run_strategies([b, Rank()], data, config(), bar_stamp="close")
    assert a.seen == b.seen


def test_runners_without_the_shared_runtime_and_unlinkable_forms_are_refused_typed():
    class S(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr(TEXT, instrument="ASSET")

        def on_bar(self, ctx):
            pass

    b = one(30, 1)
    with pytest.raises(RaptorBTError) as e:
        run_strategy_backtest(
            S(), b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"],
            bar_stamp="close",
        )
    assert e.value.code == "expr_unsupported_here"

    class OnATimeframe(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr(TEXT, instrument="S0", timeframe="5m")

    with pytest.raises(RaptorBTError) as e:
        run(OnATimeframe(), universe())
    assert e.value.code == "expr_unsupported_here" and e.value.parameter == "timeframe"

    class Twice(Strategy):
        def on_start(self, ctx):
            ind = Indicator.expr(rb.expr(TEXT)._prog)
            self.register_indicator(ind, symbol="S0")
            self.register_indicator(ind, symbol="S1")

    with pytest.raises(RaptorBTError) as e:
        run(Twice(), universe())
    assert e.value.code == "expr_unsupported_here"

    class WithAt(Strategy):
        def on_start(self, ctx):
            ctx.indicators.expr('rank(at("1d", close))', instrument="S0")

    with pytest.raises(RaptorBTError):
        run(WithAt(), universe())

    hand = Indicator.expr(rb.expr(TEXT)._prog)
    with pytest.raises(RaptorBTError) as e:
        hand.update_bar(1.0, 1.0, 1.0, 1.0, 10.0)
    assert e.value.code == "expr_unsupported_here"


def test_two_strategies_with_one_expression_share_the_cells():
    def report(strategies):
        data = universe(n=5)
        node = node_for(strategies, data)
        feed(node, data, 0, 1)
        out = node.memory_report()
        node.finish()
        return out

    one_ = report([Rank()])
    two = report([Rank(), Rank(" rank( roc(close,5) ) ")])
    assert one_["indicator_cells"] == len(SYMS)
    assert (two["indicator_cells"], two["indicator_registrations"]) == (len(SYMS), 2 * len(SYMS))
    assert two["indicators"] == one_["indicators"]


FRAME_BUDGET_S = 0.056


def test_the_seal_of_500_instruments_is_inside_the_frame_budget():
    """Measured: the added cost of a rank over 500 instruments per timestamp group, against the
    same node without the expression (``ctx.indicators.sma`` on one). The claim is that the seal
    adds under ``FRAME_BUDGET_S`` (56 ms) at p99 over 500 instruments, a budget this test sets
    for itself and measures against.

        pytest tests/python/test_expr_panel_live.py -k 500 -s
    """
    m, n = 500, 60
    syms = [f"I{i:03d}" for i in range(m)]
    rng = np.random.default_rng(1)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, (n, m)), axis=0)

    class Watch(Strategy):
        def __init__(self, panel):
            super().__init__()
            self.panel = panel

        def on_start(self, ctx):
            if self.panel:
                self.h = ctx.indicators.expr(TEXT, instrument="I000")
            else:
                self.h = ctx.indicators.sma(instrument="I000", period=5)

        def on_bar(self, ctx):
            pass

    def timings(panel):
        node = LiveNode([Watch(panel)], syms, config(), bar_stamp="close")
        out = []
        for d in range(n):
            ts = T0 + (d + 1) * MIN
            g = {s: (closes[d, j], closes[d, j] + 0.3, closes[d, j] - 0.3, closes[d, j], 100.0) for j, s in enumerate(syms)}
            t = time.perf_counter()
            node.push_group(ts, g)
            out.append(time.perf_counter() - t)
        node.finish()
        return np.array(out[10:])

    base, panel = timings(False), timings(True)
    added = np.percentile(panel, 99) - np.percentile(base, 50)
    print(
        f"500 instruments: push_group p50 {np.percentile(base, 50) * 1e3:.2f} ms without, "
        f"{np.percentile(panel, 50) * 1e3:.2f} ms with; p99 with {np.percentile(panel, 99) * 1e3:.2f} ms; "
        f"added at p99 {added * 1e3:.2f} ms"
    )
    assert added < FRAME_BUDGET_S, f"the seal added {added * 1e3:.1f} ms at p99 over 500 instruments"


# -- late listings, a thin open, warm-up and restart, and the trades a rank makes -----------------

NESTED = "ema(rank(roc(close, 3)), 4) - zscore_xs(sma(close, 5))"
LATE_ABSENT = (
    {("S5", i) for i in range(15)}  # listed at date 15
    | {("S4", i) for i in range(31, 60)}  # gone after date 30
    | {("S2", 20), ("S2", 21)}  # halted for two dates
)


@pytest.mark.parametrize("mode", ["batch", "node"])
@pytest.mark.parametrize("text", [TEXT, NESTED, "zscore(rank(close), 6)"])
def test_late_listing_delisting_and_halt_equal_eval_panel(mode, text):
    data = universe(n=60, absent=LATE_ABSENT)
    s = Rank(text)
    run(s, data, mode)
    check_against_panel(s.rows, data, text)


def test_a_late_listings_warm_up_counts_from_its_own_first_bar_by_hand():
    """rank(roc(close, 5)): S5 has its first bar at date 15, so its roc is known at date 20.
    Dates 15-19 rank the other names only and S5 reads unknown; at date 20 S2 is halted, so five
    names are ranked and the rank lies on the grid i / 4. S4 is gone after date 30."""
    data = universe(n=60, absent=LATE_ABSENT)
    s = Rank(TEXT)
    run(s, data)
    by_date = {}
    for t, _, v in s.rows:
        by_date[int((t - T0) // MIN) - 1] = v
    for d in range(15, 20):
        assert by_date[d]["S5"][1] is None, d
    r = by_date[20]["S5"][1]
    assert r is not None and abs(r * 4 - round(r * 4)) < 1e-12, r
    assert by_date[32]["S4"][1] is None


def test_a_thin_open_by_hand():
    """Date 0: only S0 (10) and S1 (20) print, so rank is 0 and 1 and the others are unknown.
    Date 1: all six print 10, 20, .., 60 and rank is i / 5."""
    t1, t2 = T0 + MIN, T0 + 2 * MIN
    data = {s: {"timestamps": [], **{c: [] for c in OHLCV}} for s in SYMS}
    for s, v in (("S0", 10.0), ("S1", 20.0)):
        data[s]["timestamps"].append(t1)
        for c in OHLCV:
            data[s][c].append(1.0 if c == "volume" else v)
    for j, s in enumerate(SYMS):
        data[s]["timestamps"].append(t2)
        for c in OHLCV:
            data[s][c].append(1.0 if c == "volume" else 10.0 * (j + 1))
    data = {
        s: {k: np.array(v, dtype=np.int64 if k == "timestamps" else np.float64) for k, v in d.items()}
        for s, d in data.items()
    }
    for mode in ("batch", "node"):
        s = Rank("rank(close)")
        run(s, data, mode)
        first = next(v for t, _, v in s.rows if t == t1)
        assert {k: x[1] for k, x in first.items()} == {
            "S0": 0.0, "S1": 1.0, "S2": None, "S3": None, "S4": None, "S5": None,
        }, mode
        last = next(v for t, _, v in s.rows if t == t2)
        assert [last[x][1] for x in SYMS] == [j / 5 for j in range(6)], mode


def test_warm_up_bars_then_live_and_a_restart_equal_a_backtest_of_all_the_bars():
    data = universe(n=60, absent=LATE_ABSENT)
    cut_ts = T0 + 25 * MIN
    head = {s: {k: v[d["timestamps"] <= cut_ts] for k, v in d.items()} for s, d in data.items()}
    tail = {s: {k: v[d["timestamps"] > cut_ts] for k, v in d.items()} for s, d in data.items()}
    whole = Rank(NESTED)
    run(whole, data)
    journal = rj.Journal({"run": "panel"}, memory=True)
    live = Rank(NESTED)
    node = node_for([live], data, journal=journal, warmup_bars=head)
    gs = list(groups(tail))
    for t, g in gs[:10]:
        node.push_group(t, g)
    records = [r.canonical() for r in journal.since(0)]
    after = Rank(NESTED)
    restored = LiveNode.restore(
        None, [after], list(data), config(), journal=rj.Journal({"run": "panel"}, memory=True),
        records=records, bar_stamp="close", warmup_bars=head,
    )
    assert after.rows == live.rows
    for t, g in gs[10:]:
        node.push_group(t, g)
        restored.push_group(t, g)
    assert after.rows == live.rows
    assert live.rows == whole.rows


class RankTrader(Strategy):
    """Buys one unit of a name whose rank is at least 0.8 and sells when it falls to 0.2."""

    def on_start(self, ctx):
        self.h = ctx.indicators.expr_all(TEXT)
        self.held, self.n = {}, 0

    def on_bar(self, ctx):
        s = ctx.symbol
        r = self.h[s].value
        if r is None:
            return
        self.n += 1
        if not self.held.get(s) and r >= 0.8:
            self.held[s] = True
            self.submit_order(O.Market(side="buy", units=1.0), client_id=f"b{self.n}", symbol=s)
        elif self.held.get(s) and r <= 0.2:
            self.held[s] = False
            self.submit_order(O.Market(side="sell", units=1.0), client_id=f"s{self.n}", symbol=s)


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
def test_the_trades_a_rank_makes_are_the_same_in_a_backtest_and_on_a_node(timing):
    data = universe(n=200, absent={("S2", 80), ("S2", 81), ("S3", 120)})

    def trades(res):
        return sorted(
            (
                (t.symbol, str(t.direction), t.entry_time, t.entry_price, t.exit_time, t.exit_price, t.size, t.pnl)
                for t in res.result.trades()
            ),
            key=repr,
        )

    a = run(RankTrader(), data, "batch", fill_timing=timing)
    b = run(RankTrader(), data, "node", fill_timing=timing)
    assert len(trades(a)) > 10
    assert trades(a) == trades(b)
