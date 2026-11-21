"""A ``Future`` role is the contract that is front at the event's time: read in ``on_bar`` it follows
the roll, read before the first event (``on_start``) it is refused ``role_before_event`` rather than
answered for time 0, which named whichever contract expires first (an expired one in a live node)."""

import numpy as np
import pytest

from raptorbt import BacktestConfig, InstrumentSpec, Strategy, run_strategies
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import Chain, Future, Holding, Requirements, StrikeWindow, Underlying, orders
from raptorbt.structures import Expiry

MIN = 60 * 10**9
DAY = 1_440 * MIN
N = 40
EXPIRY_A, EXPIRY_B = 20 * MIN, 10 * DAY  # FA expires mid-run; FB long after


def bars(base):
    c = base + np.arange(N) * 0.1
    return {
        "timestamps": (np.arange(N, dtype=np.int64) + 1) * MIN,
        "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(N, 1000.0),
    }


def world(activation_ns=None):
    kw = {} if activation_ns is None else {"activation_ns": activation_ns}
    data = {"IDX": bars(100.0), "FA": bars(101.0), "FB": bars(102.0)}
    specs = {
        "FA": InstrumentSpec.futures_contract("FA", EXPIRY_A, 1.0, underlying="IDX", **kw),
        "FB": InstrumentSpec.futures_contract("FB", EXPIRY_B, 1.0, underlying="IDX", **kw),
    }
    return data, specs


def manifest(**roles):
    return Requirements(
        roles={"index": Underlying(asset="index"), **roles}, timeframes=["1m"],
        holding=Holding.overnight(), data=["bar"],
    )


class Reads(Strategy):
    requirements = manifest(fut=Future(of="index", contract="continuous", roll_days=5 / 1440))

    def __init__(self, where):
        super().__init__()
        self.where, self.fronts = where, []

    def on_start(self, ctx):
        if self.where == "start":
            ctx.role("fut")

    def on_bar(self, ctx, bar=None):
        if ctx.symbol == "IDX":
            self.fronts.append((ctx.event_ns // MIN, ctx.role("fut")))


def run(strategy, activation_ns=None):
    data, specs = world(activation_ns)
    return run_strategies([strategy], data, BacktestConfig(initial_capital=1e6, fees=0.0),
                          instruments=specs, bindings={"index": "IDX"}, bar_stamp="close")


@pytest.mark.parametrize("activation_ns", [None, 0])
def test_read_in_on_start_is_refused_whether_or_not_the_contracts_carry_a_listing_time(activation_ns):
    with pytest.raises(InvalidParameter) as info:
        run(Reads("start"), activation_ns)
    assert info.value.code == "role_before_event"
    assert "on_bar" in str(info.value) and "on_start" in str(info.value)


@pytest.mark.parametrize("activation_ns", [None, 0])
def test_read_in_on_bar_follows_the_roll(activation_ns):
    s = Reads("bar")
    run(s, activation_ns)
    by_minute = dict(s.fronts)
    assert by_minute[1] == "FA"
    assert by_minute[14] == "FA"          # the roll is 5 minutes before FA's expiry: FA is front to minute 14
    assert by_minute[16] == "FB"
    assert {f for _, f in s.fronts} == {"FA", "FB"}


def test_a_chain_on_a_future_is_refused_before_the_first_event_too():
    class ChainAtStart(Strategy):
        requirements = manifest(
            fut=Future(of="index"), opts=Chain("fut", Expiry.weekly(), StrikeWindow.around(1)))

        def on_start(self, ctx):
            ctx.role("opts")

    with pytest.raises(InvalidParameter) as info:
        run(ChainAtStart())
    assert info.value.code == "role_before_event"


def test_an_underlying_and_an_instrument_role_are_still_readable_in_on_start():
    class Fixed(Strategy):
        requirements = manifest()

        def on_start(self, ctx):
            self.index = ctx.role("index")

    s = Fixed()
    run(s)
    assert s.index == "IDX"


def test_an_open_position_is_moved_to_the_next_contract_at_the_roll():
    from raptorbt.strategy import RollEvents

    class Carry(RollEvents, Strategy):
        requirements = Reads.requirements
        auto_roll = True

        def __init__(self):
            super().__init__()
            self.n, self.rolls = 0, []

        def on_bar(self, ctx, bar=None):
            if ctx.symbol != "IDX":
                return
            self.n += 1
            if self.n == 1:
                self.submit_order(orders.Market(side="buy", quantity=3), symbol=ctx.role("fut"))

        def on_roll(self, ctx, roll):
            self.rolls.append((roll.from_symbol, roll.to_symbol))

    s = Carry()
    run(s)
    assert s.rolls == [("FA", "FB")]
