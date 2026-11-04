"""A node restored from its journal at any point equals the node that wrote it.

the exit of a hedge leg in progress (the sold legs closed first, the hedge after) lives in the
book, not in an order list, so the digest a checkpoint is checked against must cover it.
the equity a restored node shows is the one of the bars it replayed, at every record cut.
"""

import pytest

from raptorbt import LiveNode, journal as rj
from raptorbt.strategy._recovery import state_summary
from test_hedge_cover import N, U, _Spread, _build, _cfg


def _feed(node, data, i):
    node.push_group(
        int(data[U]["timestamps"][i]),
        {x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume")) for x in data},
    )


@pytest.mark.parametrize("hedge_open", [True, False])
@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
def test_a_node_restored_mid_hedge_exit_equals_the_node_that_wrote_the_journal(timing, hedge_open):
    data, inst, S, H = _build(100, 105, hedge_open)
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    journal = rj.Journal({"run": "cover"}, memory=True)
    node = LiveNode([_Spread(S, H)], list(data), _cfg(timing), journal=journal, **kw)
    pending = False
    for i in range(N):
        _feed(node, data, i)
        pending |= bool(state_summary(node).get("covers"))
        records = [r.canonical() for r in journal.since(0)]
        restored = LiveNode.restore(
            None, [_Spread(S, H)], list(data), _cfg(timing), journal=rj.Journal({"run": "cover"}, memory=True),
            records=records, **kw,
        )
        assert restored.state_digest() == node.state_digest(), (timing, i)
        assert restored.equity(0) == node.equity(0), (timing, i)
    if (timing, hedge_open) == ("next_bar_open", False):
        assert pending, "the case that holds a hedge exit between two bars must have been exercised"
