"""A resting order that trades inside bar t is matched before ``on_bar(t)`` decides, on every path.

A venue fills a stop or a limit while the bar forms, so the strategy that decides on the bar's close
sees the position the bar left, not the one it opened with. The batch run used to match the resting
orders after the decision. Orders placed at the decision never meet that bar's range. Every number
is hand-computed from the bars written here; fees are zero.
"""

import pytest
from test_venue_parity import MIN, PATHS, T0, bars, run

from raptorbt import BacktestConfig, Strategy
from raptorbt.strategy import orders as O

FLAT = (100.0, 101.0, 99.0, 100.0)


def nbo():
    return BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="next_bar_open")


def sbc():
    return BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing="same_bar_close")


TIMINGS = {"next_bar_open": nbo, "same_bar_close": sbc}


class Probe(Strategy):
    """Runs ``script[n]`` on bar n and records the position size ``on_bar`` sees on every bar."""

    def __init__(self, script):
        super().__init__()
        self.script, self.n, self.seen = script, 0, []

    def on_bar(self, ctx, bar=None):
        self.seen.append(sum(p.size for p in ctx.positions()))
        act = self.script.get(self.n)
        self.n += 1
        if act:
            act(self, ctx)


def play(script, rows, cfg):
    """``{path: (strategies, trades as (size, entry, exit))}``."""
    made = []

    def make():
        s = Probe(script)
        made.append(s)
        return s

    out = run(script, {"X": bars(rows, T0)}, cfg, strategy=make)
    return {
        p: (made[i].seen, sorted((round(t.size, 6), round(t.entry_price, 6), round(t.exit_price, 6)) for t in r.result.trades()))
        for i, (p, r) in enumerate(out.items())
    }


@pytest.mark.parametrize("timing", TIMINGS)
def test_on_bar_sees_flat_after_the_stop_inside_the_bar(timing):
    """Long 10 with a stop at 95. Bar 3 (100, 100, 94, 96) trades through the stop: it sells at 95,
    -50, and on_bar(3) sees flat, so its own sell does not open a short."""
    rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0), (96.0, 106.0, 96.0, 105.0)]

    def sell_if_held(s, c):
        if c.positions():
            s.submit_order(O.Market(side="sell", units=10.0))

    script = {
        0: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
        3: sell_if_held,
    }
    got = play(script, rows, TIMINGS[timing])
    entry = 100.0
    for path in PATHS:
        seen, trades = got[path]
        assert seen[3] == 0.0, (path, seen)
        assert trades == [(10.0, entry, 95.0)], (path, trades)


@pytest.mark.parametrize("timing", TIMINGS)
def test_on_bar_sees_the_long_a_resting_limit_opened_inside_the_bar(timing):
    """A buy limit at 99.5 rests from bar 0. Bar 1 (100, 101, 99, 100) trades down to 99 and fills
    it at 99.5, so on_bar(1) sees the 10 it bought."""
    script = {0: lambda s, c: s.submit_order(O.Limit(side="buy", units=10.0, price=99.5))}
    got = play(script, [FLAT, FLAT, FLAT, FLAT], TIMINGS[timing])
    for path in PATHS:
        seen, _ = got[path]
        assert seen[:3] == [0.0, 10.0, 10.0], (path, seen)


@pytest.mark.parametrize("timing", TIMINGS)
def test_an_order_placed_at_the_decision_never_meets_that_bars_range(timing):
    """On bar 3 (100, 100, 94, 96) the strategy places a buy limit at 94.5: inside the bar's range
    but below its close, so it rests. It does not trade on the bar that decided it, and the later
    bars stay above it: no position, no trade."""
    rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0), (96.0, 97.0, 95.0, 96.0)]
    script = {3: lambda s, c: s.submit_order(O.Limit(side="buy", units=10.0, price=94.5))}
    got = play(script, rows, TIMINGS[timing])
    for path in PATHS:
        seen, trades = got[path]
        assert seen == [0.0] * 6 and trades == [], (path, seen, trades)


@pytest.mark.parametrize("timing", TIMINGS)
def test_a_close_decided_with_the_stop_touched_trades_the_stop_once(timing):
    """close_position on the bar that also touches the stop: the stop trades first (bar 3 low 94),
    the close finds nothing. One trade at 95."""
    rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0)]
    script = {
        0: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)),
        3: lambda s, c: s.close_position(),
    }
    got = play(script, rows, TIMINGS[timing])
    for path in PATHS:
        assert got[path][1] == [(10.0, 100.0, 95.0)], (path, got[path])


@pytest.mark.parametrize("path", ["batch", "node", "paper"])
def test_a_stop_the_bar_gapped_through_is_answered_at_the_open_it_filled_at(path):
    """Long 10 at 100, a resting sell stop at 95. Bar 3 gaps to (90, 92, 85, 86): the stop sells at
    90, at the open, which a venue reports before the bar trades. The hook buys 5 back and those 5
    fill at that same open, 90 (they used to be heard after on_bar(3) and filled at bar 4's open,
    87; live they go at about 90)."""
    from raptorbt import LiveNode, run_strategies
    from raptorbt.execution import PaperExecution

    rows = [FLAT, FLAT, FLAT, (90.0, 92.0, 85.0, 86.0), (87.0, 88.0, 86.0, 87.0), (87.0, 88.0, 86.0, 87.0)]

    class Answer(Strategy):
        def __init__(self):
            super().__init__()
            self.prices, self.at_hook = {}, None

        def on_bar(self, ctx):
            self.last = ctx.idx
            if ctx.idx == 1:
                self.submit_order(O.Market(side="buy", units=10.0), client_id="in")
            if ctx.idx == 2:
                self.submit_order(O.StopMarket(side="sell", units=10.0, trigger=95.0), client_id="stop")

        def on_order_event(self, ctx, e):
            if e.kind != "order_filled":
                return
            self.prices[e.client_order_id] = e.price
            if e.client_order_id == "stop":
                self.at_hook = self.last
                self.submit_order(O.Market(side="buy", units=5.0), client_id="ans")

    s = Answer()
    data = {"X": bars(rows, T0)}
    if path == "batch":
        run_strategies([s], data, nbo(), bar_stamp="close")
    else:
        node = LiveNode([s], ["X"], nbo(), bar_stamp="close", execution=PaperExecution() if path == "paper" else None)
        for i, row in enumerate(rows):
            node.push_bar("X", T0 + (i + 1) * MIN, *map(float, row), 1e6)
        node.finish()
    assert s.prices == {"in": 100.0, "stop": 90.0, "ans": 90.0}, s.prices
    # Heard before on_bar(3): the last bar the strategy had decided on is bar 2.
    assert s.at_hook == 2, s.at_hook


@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
@pytest.mark.parametrize("timing", TIMINGS)
def test_a_decision_on_one_instrument_sees_the_stop_another_traded_at_the_same_instant(timing, order):
    """Long 10 B with a stop at 95; bar 3 of B trades through it. On the same instant `on_bar` of A
    reads B's position: flat, whichever symbol is listed first, and the stop's fill is the same."""
    from raptorbt import run_strategies

    quiet = [FLAT] * 5
    b_rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0)]
    seen = {}

    class Both(Strategy):
        def on_bar(self, ctx):
            if ctx.symbol == "B" and ctx.idx == 0:
                self.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0))
            if ctx.symbol == "A":
                seen[ctx.idx] = sum(p.size for p in ctx.positions("B"))

    rows = {"A": quiet, "B": b_rows}
    data = {s: bars(rows[s], T0) for s in order}
    out = run_strategies([Both()], data, TIMINGS[timing](), bar_stamp="close")[0]
    assert seen[3] == 0.0 and seen[2] == 10.0, seen
    assert [(t.symbol, t.exit_price) for t in out.result.trades()] == [("B", 95.0)]


@pytest.mark.parametrize("timing", TIMINGS)
def test_cash_a_stop_frees_is_in_the_pool_when_the_other_member_of_the_account_decides(timing):
    """Two strategies share an account of 100,000. The first buys 900 AAA at 100 with a stop at 95
    (90,000 out of the pool). Bar 3 of AAA trades through the stop at 95: 85,500 comes back, so
    the pool holds 95,500. The second, on BBB, reads the account's cash in `on_bar` of the same
    instant and sees the 95,500, not the 10,000 a decision ahead of the stop would."""
    from raptorbt import AccountSpec, run_strategies

    class First(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.submit_order(O.Market(side="buy", units=900.0, stop_price=95.0))

    class Second(Strategy):
        def __init__(self):
            super().__init__()
            self.cash = {}

        def on_bar(self, ctx):
            self.cash[ctx.idx] = ctx.cash

    rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0)]
    data = {"AAA": bars(rows, T0), "BBB": bars([FLAT] * 5, T0)}
    second = Second()
    cfg = TIMINGS[timing]()
    run_strategies(
        [(First(), ["AAA"], {"account": "x"}), (second, ["BBB"], {"account": "x"})],
        data,
        cfg,
        accounts={"x": AccountSpec(initial_capital=100_000.0)},
        bar_stamp="close",
    )
    assert second.cash[2] == pytest.approx(10_000.0) and second.cash[3] == pytest.approx(95_500.0), second.cash


@pytest.mark.parametrize("timing", TIMINGS)
def test_the_single_instrument_runner_agrees(timing):
    """`run_strategy_backtest` meets the stop before `on_bar` too: flat in bar 3, one trade at 95."""
    from raptorbt import run_strategy_backtest

    rows = [FLAT, FLAT, FLAT, (100.0, 100.0, 94.0, 96.0), (96.0, 97.0, 95.0, 96.0), (96.0, 106.0, 96.0, 105.0)]

    def sell_if_held(s, c):
        if c.positions():
            s.submit_order(O.Market(side="sell", units=10.0))

    s = Probe({0: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0, stop_price=95.0)), 3: sell_if_held})
    result = run_strategy_backtest(s, **bars(rows, T0), symbol="X", config=TIMINGS[timing](), bar_stamp="close")
    assert s.seen[3] == 0.0, s.seen
    assert [(t.size, t.exit_price) for t in result.trades()] == [(10.0, 95.0)]


@pytest.mark.parametrize("timing", TIMINGS)
def test_a_limit_moved_through_the_market_trades_at_the_next_open(timing):
    """A buy limit at 90 rests (bars near 100). On bar 2 it is moved to 110, through the market. It
    is not examined against bar 2 (decided on its close); it meets bar 3 as an arrival and fills at
    that bar's open, 100, not at 110."""
    rows = [FLAT, FLAT, FLAT, FLAT, FLAT]
    state = {}

    def place(s, c):
        state["id"] = s.submit_order(O.Limit(side="buy", units=1.0, price=90.0), client_id="m")

    script = {0: place, 2: lambda s, c: s.modify_order("m", limit_price=110.0)}
    made = []

    def make():
        p = Probe(script)
        made.append(p)
        return p

    out = run(script, {"X": bars(rows, T0)}, TIMINGS[timing], strategy=make)
    for path in PATHS:
        fills = [(o.client_id, o.avg_fill_price, o.last_fill_idx) for o in out[path].result.orders() if o.client_id == "m"]
        assert fills == [("m", 100.0, 3)], (path, fills)
