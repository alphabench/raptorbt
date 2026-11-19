"""A ``Universe`` role bound to ``PointInTime``: the union is the binding, ``ctx.role`` answers the
members at the event's time, and the manifest is the one a plain list is bound to."""

import pickle

import numpy as np
import pytest

from raptorbt import BacktestConfig, Strategy, run_strategies
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import Holding, PointInTime, Requirements, Universe, check_requirements, requirements_hash, resolve

MIN = 60 * 10**9


def bars(base, n=10):
    c = base + np.arange(n) * 0.1
    return {
        "timestamps": (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 1000.0),
    }


class Reads(Strategy):
    requirements = Requirements(
        roles={"basket": Universe(asset="equity", min=1, max=4)},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
    )

    def __init__(self):
        super().__init__()
        self.seen = []

    def on_bar(self, ctx, bar=None):
        if ctx.symbol == "A":
            self.seen.append((ctx.event_ns // MIN, ctx.role("basket")))


def test_at_answers_the_members_of_the_last_step_started():
    p = PointInTime(["A", "B", "C"], [(10, ["A", "B"]), (20, []), (30, ["C"])])
    assert tuple(p) == ("A", "B", "C")
    assert p.at(9) == ()
    assert p.at(10) == p.at(19) == ("A", "B")
    assert p.at(20) == ()
    assert p.at(10**18) == ("C",)
    assert p.at(None) == ("A", "B", "C")


@pytest.mark.parametrize(
    "symbols, steps",
    [
        ([], [(0, [])]),
        (["A"], []),
        (["A"], [(0, ["B"])]),
        (["A", "B"], [(5, ["A"]), (5, ["B"])]),
        (["A", "B"], [(5, ["A"]), (4, ["B"])]),
        (["A", "B"], [(0, ["A", "A"])]),
        (["A"], [(0.5, ["A"])]),
    ],
)
def test_a_bad_binding_is_refused(symbols, steps):
    with pytest.raises(InvalidParameter) as e:
        PointInTime(symbols, steps)
    assert e.value.code == "invalid_binding"


def test_it_is_a_plain_tuple_of_the_union_for_the_manifest_and_survives_a_pickle():
    p = PointInTime(["A", "B"], [(0, ["A"]), (5, ["B"])])
    plan = resolve(Reads.requirements, now_ns=0, bindings={"basket": p})
    assert plan.roles["basket"].symbols == ("A", "B") and plan.issues == ()
    back = pickle.loads(pickle.dumps(p))
    assert tuple(back) == ("A", "B") and back.steps == p.steps
    assert check_requirements(Reads) == []


def test_the_manifest_is_the_same_for_a_list_and_a_point_in_time_binding():
    assert Reads.requirements.to_dict()["roles"]["basket"] == {
        "kind": "universe", "asset": "equity", "min": 1, "max": 4, "data": ["bar"],
    }
    assert requirements_hash(Reads) == requirements_hash(Reads.requirements)


def test_ctx_role_follows_the_steps_in_a_run():
    s = Reads()
    p = PointInTime(["A", "B", "C"], [(0, ["A", "B"]), (4 * MIN, ["B", "C"]), (8 * MIN, ["C"])])
    run_strategies(
        [(s, ["A", "B", "C"])], {"A": bars(100), "B": bars(50), "C": bars(20)}, BacktestConfig(),
        bindings={"basket": p}, bar_stamp="close",
    )
    by_minute = dict(s.seen)
    assert by_minute[1] == by_minute[3] == ("A", "B")  # the bar closing at minute 4 is stamped 4
    assert by_minute[4] == by_minute[7] == ("B", "C")
    assert by_minute[8] == by_minute[10] == ("C",)


def test_a_list_binding_is_unchanged():
    s = Reads()
    run_strategies(
        [(s, ["A", "B"])], {"A": bars(100), "B": bars(50)}, BacktestConfig(),
        bindings={"basket": ["B", "A"]}, bar_stamp="close",
    )
    assert {seen for _, seen in s.seen} == {("B", "A")}


def test_a_step_may_not_name_a_symbol_the_run_lacks():
    with pytest.raises(InvalidParameter):
        PointInTime(["A", "B"], [(0, ["A", "Z"])])
