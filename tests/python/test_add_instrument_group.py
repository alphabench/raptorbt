"""A group pushed after `add_instrument` that includes the added symbol is delivered."""

import pytest

from raptorbt import BacktestConfig, InstrumentSpec, LiveNode, Strategy

MIN = 60 * 10**9


class Seen(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx, bar=None):
        self.seen.append((ctx.symbol, ctx.event_ns))


def bars(**prices):
    return {s: (p, p + 1, p - 1, p, 10.0) for s, p in prices.items()}


@pytest.mark.parametrize("timing", ["next_bar_open", "same_bar_close"])
def test_a_group_with_an_added_symbol_is_delivered_and_the_node_stays_live(timing):
    s = Seen()
    node = LiveNode([(s, ["B", "D"])], ["B", "D"], BacktestConfig(fill_timing=timing), bar_stamp="close")
    node.push_group(MIN, bars(B=100.0, D=50.0))
    # "A" sorts before the names already held, "C" between them, "Z" after
    for k, name in enumerate(("C", "A", "Z")):
        node.add_instrument(name, InstrumentSpec.equity(name), strategies=[0])
        t = (k + 2) * MIN
        node.push_group(t, bars(B=100.0, D=50.0, **{name: 10.0}))
        assert {sym for sym, ns in s.seen if ns == t} == {name, "B", "D"}
    assert node.state == "live"
    # one name in order of the symbols, as a group is decided
    last = [sym for sym, ns in s.seen if ns == 4 * MIN]
    assert last == sorted(last)
