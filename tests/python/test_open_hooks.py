"""Under next_bar_open the hook that hears a fill at a bar's open knows only what was known at
that open, and what it answers fills at that open.

A venue reports what was decided on the previous bar at the open of the next one, before that bar
has traded. So for each timestamp the books match the open first, the hooks hear those fills while
the strategy and its indicators know only the bars up to the previous one, and only then is the
bar applied to the strategy and decided on. A resting order the bar gapped through fills at the
open and is one of those fills: heard before the bar, answered at that open. A fill the rest of the
bar brings is heard before the strategy decides on the bar, knowing only the bars before it,
and what it is answered with goes to the next bar's open.

Bars are 1-minute, stamped at close, from 09:16 IST on 2026-11-02. The same answer in a batch
run, on a node and on the paper venue.
"""

import pytest
from _market_harness import bars, fmt, ist

from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt.execution import PaperExecution
from raptorbt.strategy import orders as O

T0 = ist(2026, 11, 2, 9, 15)
MIN = 60 * 10**9
NBO = dict(initial_capital=1e5, fees=0.0, fill_timing="next_bar_open")
MODES = ["batch", "node", "paper"]
OHLCV = ("open", "high", "low", "close", "volume")


def _run(strategy, data, mode, **config):
    cfg = BacktestConfig(**{**NBO, **config})
    if mode == "batch":
        return run_strategies([strategy], data, cfg, bar_stamp="close")[0]
    syms = list(data)
    node = LiveNode(
        [strategy], syms, cfg, bar_stamp="close", execution=PaperExecution() if mode == "paper" else None
    )
    for i in range(len(data[syms[0]]["timestamps"])):
        node.push_group(
            int(data[syms[0]]["timestamps"][i]), {s: tuple(float(data[s][c][i]) for c in OHLCV) for s in syms}
        )
    return node.finish()[0]


# X falls on bar 1 and crashes on bar 2; Y rises on bar 1 and jumps on bar 2.
X = [(100,) * 4, (100, 100, 99, 99), (100, 101, 89, 90), (90, 91, 89, 90), (90, 91, 89, 90)]
Y = [(100,) * 4, (100, 101, 100, 101), (100, 111, 99, 110), (110, 111, 109, 110), (110, 111, 109, 110)]
T2 = fmt(T0 + 3 * MIN)  # the stamp of bar 2: 09:18
T3 = fmt(T0 + 4 * MIN)


class Answer(Strategy):
    """Buys 1 X on bar 1: it fills at the open of bar 2, 100. Every bar it keeps what it saw. The
    hook of that fill answers from what it knows: sell 1 X if X fell, buy 10 Y if Y rose."""

    def __init__(self):
        super().__init__()
        self.known, self.at_hook, self.calls, self.fills = {}, None, [], []

    def on_bar(self, ctx):
        self.calls.append(("bar", ctx.symbol, ctx.idx))
        self.known[ctx.symbol] = (ctx.idx, ctx.bar.close - ctx.bar.open)
        if ctx.symbol == "X" and ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=1.0, instrument="X"), client_id="probe")

    def on_order_event(self, ctx, e):
        if e.kind != "order_filled":
            return
        self.calls.append(("fill", e.client_order_id))
        self.fills.append((e.client_order_id, fmt(ctx.event_ns), e.price))
        if e.client_order_id != "probe":
            return
        self.at_hook = dict(self.known)
        if self.known["X"][1] < 0:
            self.submit_order(O.Market(side="sell", units=1.0, instrument="X"), client_id="ansX")
        if self.known["Y"][1] > 0:
            self.submit_order(O.Market(side="buy", units=10.0, instrument="Y"), client_id="ansY")


@pytest.mark.parametrize("mode", MODES)
def test_the_hook_of_a_fill_at_the_open_runs_before_the_strategy_sees_the_bar(mode):
    s = Answer()
    _run(s, {"X": bars(X, T0), "Y": bars(Y, T0)}, mode)
    # Bar 2's decision comes after the fill of its open is heard, and the hook knew bar 1 only.
    assert s.calls.index(("fill", "probe")) < s.calls.index(("bar", "X", 2)), s.calls
    assert s.calls.index(("fill", "probe")) < s.calls.index(("bar", "Y", 2)), s.calls
    assert s.at_hook == {"X": (1, -1.0), "Y": (1, 1.0)}, s.at_hook


@pytest.mark.parametrize("mode", MODES)
def test_an_answer_made_from_what_was_known_at_the_open_fills_at_that_open(mode):
    s = Answer()
    _run(s, {"X": bars(X, T0), "Y": bars(Y, T0)}, mode)
    got = sorted(f for f in s.fills if f[0] != "probe")
    # Both answers were decided from bar 1 and meet the open of bar 2: X at 100, Y at 100.
    assert got == [("ansX", T2, 100.0), ("ansY", T2, 100.0)], (got, s.calls)


class Peek(Strategy):
    """Buys 10 on bar 1 (fills at the open of bar 2, 100). Bar 2 crashes to 80. A hook that knew
    the crash would sell at once."""

    def __init__(self):
        super().__init__()
        self.last, self.at_hook = None, None

    def on_bar(self, ctx):
        self.last = (ctx.idx, ctx.bar.close)
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=10.0), client_id="seed")

    def on_order_event(self, ctx, e):
        if e.kind == "order_filled" and e.client_order_id == "seed":
            self.at_hook = self.last
            if self.last[1] < 90:
                self.submit_order(O.Market(side="sell", units=10.0, reduce_only=True), client_id="escape")


CRASH = [(100,) * 4, (100,) * 4, (100, 101, 79, 80), (80, 81, 79, 80), (80, 81, 79, 80), (80, 81, 79, 80)]


@pytest.mark.parametrize("mode", MODES)
def test_a_hook_cannot_escape_a_crash_of_the_bar_whose_open_it_hears(mode):
    s = Peek()
    r = _run(s, {"A": bars(CRASH, T0)}, mode)
    assert s.at_hook == (1, 100.0), s.at_hook
    assert [o.client_id for o in r.result.orders()] == ["seed"]
    # Held through the crash: bought at 100, closed with the data at 80.
    assert [(t.entry_price, t.exit_price) for t in r.result.trades()] == [(100.0, 80.0)]


GAP = [(100,) * 4, (98, 99, 97, 98.5), (90, 91, 89, 90), (90, 91, 89, 90)]
T1 = fmt(T0 + 2 * MIN)


class Gap(Strategy):
    """A buy limit at 99 rests from bar 0; bar 1 opens at 98, through it, and fills it there. That
    fill is at bar 1's open: a venue reports it before the bar has traded, so its hook runs before
    `on_bar(1)` (which sees the close at 98.5 only afterwards) and what it answers fills at that
    same open, 98 (it used to be heard after `on_bar(1)` and answered at bar 2's open, 90)."""

    def __init__(self):
        super().__init__()
        self.fills = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(O.Limit(side="buy", units=1.0, price=99.0), client_id="rest")

    def on_order_event(self, ctx, e):
        if e.kind != "order_filled":
            return
        self.fills.append((e.client_order_id, fmt(ctx.event_ns), e.price))
        if e.client_order_id == "rest":
            self.submit_order(O.Market(side="sell", units=1.0, reduce_only=True), client_id="ans")


@pytest.mark.parametrize("mode", MODES)
def test_the_answer_to_a_resting_order_the_bar_gapped_through_fills_at_that_open(mode):
    s = Gap()
    _run(s, {"X": bars(GAP, T0)}, mode)
    assert s.fills == [("rest", T1, 98.0), ("ans", T1, 98.0)], s.fills


class Inside(Strategy):
    """The same limit, but bar 1 opens at 100 and trades down through 99: the fill is inside the
    bar, at 99, and what it answers goes to bar 2's open."""

    def __init__(self):
        super().__init__()
        self.fills = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(O.Limit(side="buy", units=1.0, price=99.0), client_id="rest")

    def on_order_event(self, ctx, e):
        if e.kind != "order_filled":
            return
        self.fills.append((e.client_order_id, fmt(ctx.event_ns), e.price))
        if e.client_order_id == "rest":
            self.submit_order(O.Market(side="sell", units=1.0, reduce_only=True), client_id="ans")


INSIDE = [(100,) * 4, (100, 101, 98, 98.5), (90, 91, 89, 90), (90, 91, 89, 90)]


@pytest.mark.parametrize("mode", MODES)
def test_the_answer_to_a_fill_inside_a_bar_fills_at_the_next_open(mode):
    s = Inside()
    _run(s, {"X": bars(INSIDE, T0)}, mode)
    assert s.fills == [("rest", T1, 99.0), ("ans", T2, 90.0)], s.fills


class Close(Strategy):
    def __init__(self):
        super().__init__()
        self.fills = []

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=1.0), client_id="probe")

    def on_order_event(self, ctx, e):
        if e.kind != "order_filled":
            return
        self.fills.append((e.client_order_id, fmt(ctx.event_ns), e.price))
        if e.client_order_id == "probe":
            self.submit_order(O.Market(side="sell", units=1.0, reduce_only=True), client_id="ans")


@pytest.mark.parametrize("mode", MODES)
def test_same_bar_close_is_unchanged_an_answer_fills_at_the_close_of_the_bar(mode):
    s = Close()
    _run(s, {"X": bars(X, T0)}, mode, fill_timing="same_bar_close")
    # Decided on bar 1, both fill at bar 1's own close (99), as before.
    assert [f[0] for f in s.fills] == ["probe", "ans"] and {f[2] for f in s.fills} == {99.0}, s.fills
