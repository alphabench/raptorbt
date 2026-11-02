"""Resampling against the reference implementation, replayed from fixtures.

The fixture holds the bars a reference sampler returned, call by call, for the same streams of
trades, bars, clock ticks and flushes. These tests replay
them through ``raptorbt.sampling`` and require the same bars, call by call: same stamp, same
prices and volume (bit for bit), same source count, same ``complete`` and same ``closed_by``.

The differences that are intended are not in these streams; each has its own test in
``test_resample.py`` and an entry in ``docs/resampling.md``.
"""

import json
from pathlib import Path

import pytest

from raptorbt import sampling

HERE = Path(__file__).parent
SCENARIOS = json.loads((HERE / "resample_fixture.json").read_text())


def build(decl):
    if decl["class"] == "TimeBars":
        return sampling.TimeBars(
            "K/tick", decl["period"], "K/x", anchor=decl["anchor"],
            close_on=decl["close_on"], gaps=decl["gaps"],
        )
    return sampling.Resample(
        "K/1m", decl["source"], decl["period"], "K/x", anchor=decl["anchor"],
        close_on=decl["close_on"], min_source=decl["min_source"],
    )


def as_dict(b):
    return {
        "ts": b.ts, "open": b.open, "high": b.high, "low": b.low, "close": b.close,
        "volume": b.volume, "period": b.period, "closed_by": b.closed_by,
        "n_source": b.n_source, "complete": b.complete,
    }


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_same_bars_as_the_old_engine(sc):
    s = build(sc["decl"])
    for i, (op, want) in enumerate(zip(sc["ops"], sc["result"])):
        kind = op[0]
        if kind == "trade":
            got = s.push_trade(op[1], op[2], op[3])
        elif kind == "bar":
            got = s.push_bar(op[1], op[2], op[3], op[4], op[5], op[6], op[7])
        elif kind == "clock":
            got = s.on_clock(op[1])
        elif kind == "session":
            got = s.on_session(op[1], op[2])
        else:
            got = s.flush()
        assert [as_dict(b) for b in got] == want, (i, op)
    assert any(r for r in sc["result"]), "the scenario produced no bars"
