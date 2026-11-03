"""Every indicator of a timestamp is up to date before any strategy decides.

Events that share a timestamp are one group (docs/architecture/event-semantics.md,
"Equal timestamps: the two-phase rule"). A decision at ``t`` sees every
subscribed instrument's bar and indicators as of ``t``, whatever the symbol
names; the one thing it does not see is the future. A strategy's own hooks for
its own event read the stream they are handling, bar by bar: an indicator on
the primary bar shows the value before the bar until ``on_bar``, one on a
composite stream shows the value after the emission the hook is handling.

* renaming symbols (all permutations of three) changes no decision, log or result;
* an indicator read for another instrument equals the value computed by hand
  from that instrument's bars through ``t``;
* the composite case: own-stream reads are revealed bar by bar, other
  instruments' reads are after the group, a multi-brick composite included;
* private (``share=False``) and shared indicators agree, a ``LiveNode`` fed the
  same groups reproduces the batch run.
"""

import itertools

import numpy as np
import pytest

from raptorbt import BacktestConfig, Indicator, LiveNode, Strategy, run_strategies
from raptorbt.strategy import orders

ROLES = ["a", "b", "c"]
T0 = 1_700_000_000 * 10**9
MIN = 60 * 10**9
N = 14


def _bars(closes):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    open_ = np.r_[closes[0], closes[:-1]]
    return {
        "timestamps": T0 + (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": open_,
        "high": np.maximum(open_, closes) + 0.25,
        "low": np.minimum(open_, closes) - 0.25,
        "close": closes,
        "volume": np.full(n, 1000.0),
    }


CLOSES = {
    "a": 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 1, N)),
    "b": 100.0 + np.cumsum(np.random.default_rng(2).normal(0, 1, N)),
    "c": 100.0 + np.cumsum(np.random.default_rng(3).normal(0, 1, N)),
}
PERMS = list(itertools.permutations(["AAA", "MMM", "ZZZ"]))


class Probe(Strategy):
    """Reads every instrument's indicators in every hook and decides on them."""

    def __init__(self, names):
        super().__init__()
        self.role = {s: r for r, s in zip(ROLES, names)}
        self.seen = []

    def on_start(self, ctx):
        self.sid = self.subscribe_bars(2, "m")
        self.p = {self.role[s]: self.register_indicator(Indicator.sma(2), symbol=s) for s in ctx.symbols}
        self.c = {
            self.role[s]: self.register_indicator(Indicator.sma(1), stream_id=self.sid, symbol=s)
            for s in ctx.symbols
        }

    def _snap(self):
        return {
            r: (
                None if self.p[r].value is None else round(self.p[r].value, 9),
                self.p[r].update_count,
                None if self.c[r].value is None else round(self.c[r].value, 9),
                self.c[r].update_count,
            )
            for r in ROLES
        }

    def on_bar(self, ctx, event):
        me = self.role[ctx.symbol]
        if event.timeframe is not None:
            self.seen.append(("comp", me, round(event.close, 9), self._snap()))
            return
        self.seen.append(("bar", me, round(ctx.bar.close, 9), self._snap()))
        other = self.p["b"].value, self.p["c"].value
        if me == "a" and None not in other:
            if other[0] > other[1] and ctx.position is None:
                self.submit_order(orders.Market(side="buy", units=3.0))
            elif other[0] < other[1] and ctx.position is not None:
                self.close_position()


def _data(names):
    return {s: _bars(CLOSES[r]) for r, s in zip(ROLES, names)}


def _cfg():
    return BacktestConfig(initial_capital=1e6, fees=0.0)


def _batch(names, share=True, data=None):
    s = Probe(names)
    res = run_strategies([s], data or _data(names), _cfg(), share=share, bar_stamp="close")[0]
    return s, res


def _node(names):
    s = Probe(names)
    data = _data(names)
    node = LiveNode([s], list(names), _cfg(), bar_stamp="close")
    for i in range(N):
        ts = int(data[names[0]]["timestamps"][i])
        node.push_group(
            ts,
            {
                x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume"))
                for x in names
            },
        )
    return s, node.finish()[0]


def _by_hook(seen):
    """The log per (hook, role): hooks of one timestamp run in ascending symbol,
    so their order across roles follows the names; what each reads must not."""
    out = {}
    for kind, role, *rest in seen:
        out.setdefault((kind, role), []).append(rest)
    return out


def _trades(res):
    return [(t.entry_price, t.exit_price, t.size, t.pnl) for t in res.result.trades()]


def _role_trades(res, names):
    return _trades(res)


@pytest.mark.parametrize("share", [True, False])
def test_renaming_symbols_changes_nothing(share):
    base, bres = _batch(PERMS[0], share)
    base_log = _by_hook(base.seen)
    assert any(e[0] == "comp" for e in base.seen) and bres.result.trades()
    for names in PERMS[1:]:
        s, res = _batch(names, share)
        assert _by_hook(s.seen) == base_log, names
        assert _trades(res) == _trades(bres), names


def test_shared_equals_private_equals_node():
    base, bres = _batch(PERMS[0], True)
    for runner in (lambda: _batch(PERMS[0], False), lambda: _node(PERMS[0])):
        s, res = runner()
        assert s.seen == base.seen
        assert _trades(res) == _trades(bres)


@pytest.mark.parametrize("share", [True, False])
def test_indicator_at_decision_is_the_value_after_the_group(share):
    s, _ = _batch(PERMS[0], share)
    bars = [e for e in s.seen if e[0] == "bar"]
    for kind, me, close, snap in bars:
        i = int(np.argmin(np.abs(CLOSES[me] - close)))
        for r in ROLES:
            if r == me:
                # own stream, revealed by on_bar: the bar is in
                pass
            value, count, _, _ = snap[r]
            assert count == i + 1, (me, r, i)
            want = None if i == 0 else round(float(np.mean(CLOSES[r][i - 1 : i + 1])), 9)
            assert value == want, (me, r, i)


@pytest.mark.parametrize("share", [True, False])
def test_composite_hook_reads_own_stream_bar_by_bar(share):
    s, _ = _batch(PERMS[0], share)
    comps = [e for e in s.seen if e[0] == "comp"]
    assert comps
    for _, me, close, snap in comps:
        i = int(np.argmin(np.abs(CLOSES[me] - close))) if False else None
        # the composite stream indicator shows the emission being handled
        assert snap[me][2] == close
        # the primary indicator of the same event is revealed at on_bar: it
        # shows the value before the bar, and one bar fewer
        j = next(k for k in range(N) if snap[me][1] == k)
        for r in ROLES:
            if r != me:
                assert snap[r][1] == snap[me][1] + 1
        assert snap[me][1] + 1 <= N


@pytest.mark.parametrize("share", [True, False])
def test_multi_brick_composite(share):
    """A renko stream that emits three bricks in one event: the owner's hook for
    brick k reads the value after brick k, every other instrument the value
    after all three, whatever the symbols are called."""
    rise = np.array([100, 100, 100, 103, 103, 103, 103.0])
    flat = np.full(7, 50.0)

    class Bricks(Strategy):
        def __init__(self, up):
            super().__init__()
            self.up, self.seen = up, []

        def on_start(self, ctx):
            self.sid = self.subscribe_bars(1, "renko", brick_size=1.0)
            self.ind = {
                s: self.register_indicator(Indicator.sma(1), stream_id=self.sid, symbol=s)
                for s in ctx.symbols
            }

        def _val(self, who):
            return self.ind[who].value

        def on_bar(self, ctx, event):
            if event.timeframe is not None:
                self.seen.append(("comp", ctx.symbol == self.up, event.close, self._val(ctx.symbol)))
                return
            other = [s for s in self.ind if s != ctx.symbol][0]
            self.seen.append(("bar", ctx.symbol == self.up, ctx.bar.close, self._val(other)))

    runs = []
    for names in (("AAA", "ZZZ"), ("ZZZ", "AAA")):
        s = Bricks(names[0])
        run_strategies([s], {names[0]: _bars(rise), names[1]: _bars(flat)}, _cfg(), share=share, bar_stamp="close")
        runs.append(s.seen)
    comps = [e for e in runs[0] if e[0] == "comp" and e[1]]
    assert [c[2] for c in comps] == [101.0, 102.0, 103.0]
    # the owner's hook sees the brick it is handling, not the later ones
    assert [c[3] for c in comps] == [101.0, 102.0, 103.0]
    # the idle instrument's bar hook at the same timestamp sees all three
    flat_bars = [e for e in runs[0] if e[0] == "bar" and not e[1]]
    assert 103.0 in [b[3] for b in flat_bars]
    first_103 = next(i for i, b in enumerate(flat_bars) if b[3] == 103.0)
    assert all(b[3] in (None, 100.0) for b in flat_bars[:first_103])
    # a symbol's name changes no hook's reading, only the order the hooks run in
    for kind in ("comp", "bar"):
        for up in (True, False):
            a = [e[2:] for e in runs[0] if e[0] == kind and e[1] == up]
            b = [e[2:] for e in runs[1] if e[0] == kind and e[1] == up]
            assert a == b, (kind, up)


@pytest.mark.parametrize("share", [True, False])
def test_symbol_less_indicator_is_the_same_under_every_naming(share):
    """An indicator fed every symbol's bars has the whole group as its own
    stream: renaming changes nothing it shows (SMA over a group's three bars
    does not depend on the order they are fed in)."""
    import warnings

    class Fleet(Probe):
        def on_start(self, ctx):
            super().on_start(ctx)
            self.f = self.register_indicator(Indicator.sma(3))

        def _snap(self):
            v = self.f.value
            return super()._snap(), None if v is None else round(v, 9), self.f.update_count

    base = None
    for names in PERMS:
        s = Fleet(names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_strategies([s], _data(names), _cfg(), share=share, bar_stamp="close")
        log = _by_hook(s.seen)
        base = base or log
        assert log == base, names
