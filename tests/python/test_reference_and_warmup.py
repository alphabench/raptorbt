"""An index as a data-only role, `trade_from` warm-up, and the scrip group on a spec.

* An index is accepted among a run's instruments, with a fee schedule, as data: its bars feed
  the strategy, an order on it is refused (`not_tradable`), and no fee row is asked for it.
* `run_strategies(trade_from=)` runs the bars before it in full but refuses every order decided
  before it, `warmup`, and `ctx.is_warmup` is true at those events, as on a live node's warm-up.
* `InstrumentSpec` factories take `scrip_group`, so BSE equity runs under a fee schedule.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, FeeSchedule, InstrumentSpec, KernelSession, LiveNode, Strategy, run_strategies
from raptorbt.errors import InvalidParameter, RaptorBTError
from raptorbt.strategy import orders

MIN = 60 * 10**9
IST = 19_800 * 10**9
OPEN = (1_788_287_400 + 10 * 3600) * 10**9  # 2026-09-02 10:00 IST


def bars(base, n=30):
    c = base + np.arange(n) * 0.1
    return {
        "timestamps": OPEN + (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 1000.0),
    }


def config(**kw):
    return BacktestConfig(fee_schedule=FeeSchedule.exchange_presets(), session_tz_offset_ns=IST, **kw)


class Script(Strategy):
    """Does ``plan[n]`` on its nth bar of the first symbol and records the refusals it hears."""

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self.n = 0
        self.rejected = []
        self.index_closes = []

    def on_bar(self, ctx, bar=None):
        if ctx.symbol != "A":
            if ctx.symbol == "IDX":
                self.index_closes.append(float(ctx.series("IDX")["close"][-1]))
            return
        self.n += 1
        if self.n in self.plan:
            self.plan[self.n](self, ctx)

    def on_order_rejected(self, ctx, event):
        self.rejected.append((event.client_order_id if hasattr(event, "client_order_id") else None, event.reject_reason))


# -- an index is data --------------------------------------------------------------------------


def specs():
    return {"A": InstrumentSpec.equity("A", venue="NSE"), "IDX": InstrumentSpec.index("IDX", venue="NSE")}


def run(strategy, symbols=("A", "IDX"), **kw):
    return run_strategies(
        [(strategy, list(symbols))], {"A": bars(100), "IDX": bars(1000)}, config(), instruments=specs(),
        bar_stamp="close", **kw,
    )[0]


def test_an_index_feeds_data_under_a_fee_schedule():
    s = Script({3: lambda s, ctx: s.submit_order(orders.Market(side="buy", units=1.0))})
    result = run(s)
    assert s.rejected == [] and len(s.index_closes) == 30 and s.index_closes[0] == pytest.approx(1000.0)
    assert [o.status for o in result.result.orders()] == ["filled"]


def test_a_typed_order_on_an_index_is_refused_not_tradable():
    plan = {3: lambda s, ctx: s.submit_order(orders.Market(side="buy", units=1.0), client_id="x", symbol="IDX")}
    s = Script(plan)
    result = run(s)
    assert {reason for _, reason in s.rejected} == {"not_tradable"}
    (order,) = result.result.orders()
    assert (order.symbol, order.status, order.reject_reason) == ("IDX", "rejected", "not_tradable")
    assert result.result.metrics.total_trades == 0


def test_an_enter_intent_on_an_index_is_a_refused_entry():
    class Enters(Strategy):
        def __init__(self):
            super().__init__()
            self.rejected = []

        def on_bar(self, ctx, bar=None):
            if ctx.idx == 3:
                self.enter_long(size_frac=0.1)

        def on_order_rejected(self, ctx, event):
            self.rejected.append(event.reject_reason)

    s = Enters()
    result = run(s, symbols=("IDX",))
    assert set(s.rejected) == {"not_tradable"}
    assert result.rejected_entries == 1 and result.result.metrics.total_trades == 0


def test_an_index_is_refused_to_a_single_instrument_run():
    with pytest.raises(RaptorBTError) as e:
        KernelSession(symbol="IDX", instrument=InstrumentSpec.index("IDX"))
    assert e.value.code == "untradable_instrument"


def test_an_index_can_be_added_to_a_node_as_data():
    s = Script({})
    node = LiveNode([(s, ["A"])], ["A"], config(), bar_stamp="close", instruments=specs())
    node.add_instrument("IDX", InstrumentSpec.index("IDX", venue="NSE"), strategies=[0])
    node.push_bar("IDX", OPEN + MIN, 1000.0, 1001.0, 999.0, 1000.0, 10.0)
    assert tuple(node.symbols) == ("A", "IDX")


# -- trade_from --------------------------------------------------------------------------------


def buy(s, ctx):
    s.submit_order(orders.Market(side="buy", units=1.0), client_id=f"o{s.n}")


def test_orders_before_trade_from_are_refused_warmup_and_later_ones_are_not():
    cut = OPEN + 6 * MIN
    s = Script({3: buy, 8: buy})
    result = run(s, symbols=("A",), trade_from=cut)
    assert {reason for _, reason in s.rejected} == {"warmup"}
    by_id = {o.client_id: (o.status, o.reject_reason) for o in result.result.orders()}
    assert by_id["o3"] == ("rejected", "warmup") and by_id["o8"][0] == "filled"


def test_the_bars_before_trade_from_still_feed_the_strategy():
    cut = OPEN + 25 * MIN
    s = Script({})
    run(s, trade_from=cut)
    assert len(s.index_closes) == 30 and s.n == 30


def test_an_enter_intent_before_trade_from_is_a_refused_entry():
    cut = OPEN + 6 * MIN
    s = Script({4: lambda s, ctx: s.enter_long(size_frac=0.1), 9: lambda s, ctx: s.enter_long(size_frac=0.1)})
    result = run(s, symbols=("A",), trade_from=cut)
    assert {reason for _, reason in s.rejected} == {"warmup"}
    assert result.rejected_entries == 1
    (trade,) = result.result.trades()
    assert trade.entry_idx >= 8  # decided on the 9th bar (index 8), after the cut


def test_a_close_before_trade_from_is_a_no_op_not_a_failure():
    s = Script({2: lambda s, ctx: s.close_position()})
    run(s, symbols=("A",), trade_from=OPEN + 6 * MIN)
    assert s.rejected == []


def test_trade_from_at_the_first_decision_changes_nothing():
    plan = {3: buy}
    a = run(Script(plan), symbols=("A",))
    b = run(Script(plan), symbols=("A",), trade_from=OPEN)
    c = run(Script(plan), symbols=("A",), trade_from=None)
    key = lambda r: [(t.entry_idx, t.exit_idx, t.size, t.pnl) for t in r.result.trades()]
    assert key(a) == key(b) == key(c) and a.result.metrics.total_trades == b.result.metrics.total_trades


class Guarded(Strategy):
    """Buys one unit on every bar it is not warming up, and records what each bar said."""

    def __init__(self):
        super().__init__()
        self.seen = []
        self.rejected = []

    def on_bar(self, ctx, bar=None):
        self.seen.append((ctx.event_ns, ctx.is_warmup))
        if not ctx.is_warmup and len(self.seen) % 5 == 0:
            self.submit_order(orders.Market(side="buy", units=1.0), client_id=f"g{len(self.seen)}")

    def on_order_rejected(self, ctx, event):
        self.rejected.append(event.reject_reason)


def test_is_warmup_is_true_at_the_events_before_trade_from_and_only_there():
    """As on a live node's warm-up: the bars at 10:01..10:05 are before a cut at 10:06."""
    cut = OPEN + 6 * MIN
    s = Guarded()
    run(s, symbols=("A",), trade_from=cut)
    assert len(s.seen) == 30
    assert [w for _, w in s.seen] == [True] * 5 + [False] * 25
    assert all(w == (ts < cut) for ts, w in s.seen)


def test_a_strategy_that_checks_is_warmup_sends_no_order_refused_warmup():
    """The 5th bar (10:05) is before the cut: without the flag it would buy and be refused."""
    s = Guarded()
    result = run(s, symbols=("A",), trade_from=OPEN + 6 * MIN)
    assert s.rejected == []
    by_id = {o.client_id: (o.status, o.reject_reason) for o in result.result.orders()}
    assert sorted(by_id) == ["g10", "g15", "g20", "g25", "g30"]
    assert all(by_id[f"g{n}"][0] == "filled" for n in (10, 15, 20, 25)), by_id
    assert all(status != "rejected" for status, _ in by_id.values()), by_id


def test_without_trade_from_no_event_is_warmup():
    s = Guarded()
    run(s, symbols=("A",))
    assert [w for _, w in s.seen] == [False] * 30


@pytest.mark.parametrize("bad", ["09:30", 1.5, True, [1]])
def test_trade_from_is_integer_nanoseconds(bad):
    with pytest.raises(InvalidParameter) as e:
        run(Script({}), trade_from=bad)
    assert (e.value.code, e.value.parameter) == ("invalid_trade_from", "trade_from")


# -- the scrip group ---------------------------------------------------------------------------


def test_scrip_group_is_a_constructor_argument_of_every_factory():
    assert InstrumentSpec.equity("B1", venue="BSE", scrip_group="A").scrip_group == "A"
    assert InstrumentSpec.futures_contract("F", 10**18, 1.0, scrip_group="B").scrip_group == "B"
    assert InstrumentSpec.perpetual("P", scrip_group="A").scrip_group == "A"
    assert InstrumentSpec.option("O", 100.0, "call", 10**18, 50.0, scrip_group="A").scrip_group == "A"
    assert InstrumentSpec.currency_pair("C", scrip_group="A").scrip_group == "A"
    assert InstrumentSpec.index("I", scrip_group="A").scrip_group == "A"
    assert InstrumentSpec.equity("E").scrip_group is None
    spec = InstrumentSpec.equity("Q", scrip_group="A")
    assert InstrumentSpec.from_json(spec.to_json()).scrip_group == "A"
    with pytest.raises(RaptorBTError) as e:
        InstrumentSpec.equity("Q", scrip_group=" ")
    assert e.value.code == "invalid_instrument_metadata"


def test_bse_equity_runs_under_a_fee_schedule_with_its_group_and_not_without():
    def go(spec):
        s = Script({3: lambda s, ctx: s.submit_order(orders.Market(side="buy", units=1.0))})
        return run_strategies([(s, ["A"])], {"A": bars(100)}, config(), instruments={"A": spec}, bar_stamp="close")[0]

    assert go(InstrumentSpec.equity("A", venue="BSE", scrip_group="A")).result.metrics.total_trades == 1
    with pytest.raises(RaptorBTError) as e:
        go(InstrumentSpec.equity("A", venue="BSE"))
    assert e.value.code == "fee_schedule_gap" and "scrip_group" in str(e.value)
