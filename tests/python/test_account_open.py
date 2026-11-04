"""A strategy on a shared account hears the fills at a bar's open before it sees the bar,
as it does on an account of its own, so its result cannot differ by membership alone.

Under ``next_bar_open`` what was decided on a bar fills at the next open and the hook that hears it
answers at that open. Books on a shared account used to be matched with the bar, after ``on_bar``,
so the same answer filled a bar later (90 where an isolated book filled it at 100).
"""
from _market_harness import bars, fmt, ist
from raptorbt import AccountSpec, BacktestConfig, Strategy, run_strategies
from raptorbt.strategy import orders as O

T0 = ist(2026, 11, 2, 9, 15)
TAPE = [(100,) * 4, (100,) * 4, (100, 101, 99, 100), (90, 91, 89, 90), (90, 91, 89, 90)]


class _Answer(Strategy):
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


def _cfg():
    return BacktestConfig(initial_capital=1e5, fees=0.0, fill_timing="next_bar_open")


def test_an_answer_to_an_open_fill_fills_at_that_open_on_a_shared_account_too():
    alone = _Answer()
    run_strategies([alone], {"X": bars(TAPE, T0)}, _cfg(), bar_stamp="close")
    shared = _Answer()
    run_strategies([(shared, ["X"], {"account": "a"})], {"X": bars(TAPE, T0)}, _cfg(),
                   accounts={"a": AccountSpec(initial_capital=1e5)}, bar_stamp="close")
    assert alone.fills == [("probe", "11-02 09:18", 100.0), ("ans", "11-02 09:18", 100.0)]
    assert shared.fills == alone.fills


class _Buy(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(O.Market(side="buy", units=1.0), client_id="in")


def test_when_a_shared_limit_binds_the_first_strategy_in_the_group_gets_the_position():
    a, b = _Buy(), _Buy()
    (ra, rb), _ = run_strategies(
        [(a, ["X"], {"account": "a"}), (b, ["X"], {"account": "a"})],
        {"X": bars(TAPE, T0)}, _cfg(),
        accounts={"a": AccountSpec(initial_capital=1e5, max_positions=1)}, bar_stamp="close",
    )
    status = lambda r: [o.status for o in r.result.orders()]  # noqa: E731
    assert status(ra) == ["filled"]
    assert status(rb) == ["rejected"]
