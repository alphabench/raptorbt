"""The manifest's universe role, position hint and option-leg shape, and `check_requirements`
binding every role.

A universe is a basket of N instruments chosen at deploy time. `max_position` is the strategy's
largest position per role. `Legs` declares each leg of the structure (side, right, strike offset
or delta with a window, expiry), enough to list contracts and margin the structure.
"""

import json

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, InstrumentSpec, LiveNode, MarginModel, Strategy, run_strategies
from raptorbt.errors import InvalidParameter
from raptorbt.structures import Expiry
from raptorbt.strategy import (
    ByDelta,
    Chain,
    Future,
    Holding,
    Instrument,
    Legs,
    MaxPosition,
    OptionLeg,
    Requirements,
    RequirementsError,
    StrikeWindow,
    Underlying,
    Universe,
    check_requirements,
    requirements_hash,
    resolve,
    resolve_legs,
)

MIN = 60 * 10**9
DAY = 86_400 * 10**9
T0 = 1_767_225_600 * 10**9  # 2026-01-01 00:00 UTC


def bars(base, n=20):
    c = base + np.arange(n) * 0.1
    return {
        "timestamps": (np.arange(n, dtype=np.int64) + 1) * MIN,
        "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 1000.0),
    }


def basket(**kw):
    return Requirements(
        roles={"basket": Universe(asset="equity", min=2, max=4)},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar"], **kw,
    )


class Momentum(Strategy):
    """Reads its basket each bar."""

    requirements = basket(max_position={"basket": MaxPosition(fraction=0.25)})

    def __init__(self):
        super().__init__()
        self.seen = None

    def on_bar(self, ctx, bar=None):
        self.seen = ctx.role("basket")


# -- the universe role --------------------------------------------------------------------


def test_a_universe_declares_its_asset_count_and_data_and_round_trips():
    r = Requirements(
        roles={"b": Universe(asset="etf", min=3, max=10, data=("quote", "bar"))},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar", "quote"],
    )
    assert r.roles["b"].data == ("bar", "quote")
    d = r.to_dict()
    assert d["roles"]["b"] == {"kind": "universe", "asset": "etf", "min": 3, "max": 10, "data": ["bar", "quote"]}
    back = Requirements.from_dict(json.loads(json.dumps(d)))
    assert back.to_dict() == d and requirements_hash(back) == requirements_hash(r)
    assert r.issues() == []


@pytest.mark.parametrize(
    "build",
    [
        lambda: Universe(asset="bond"),
        lambda: Universe(min=0, max=3),
        lambda: Universe(min=5, max=3),
        lambda: Universe(min=True, max=3),
        lambda: Universe(data=()),
        lambda: Universe(data=("ticks",)),
    ],
)
def test_a_universe_refuses_a_bad_declaration(build):
    with pytest.raises(RequirementsError):
        build()


def test_member_data_must_be_declared_by_the_manifest():
    r = Requirements(
        roles={"b": Universe(min=1, max=2, data=("bar", "quote"))},
        timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
    )
    assert [(i.code, i.path) for i in r.issues()] == [("role_data_not_declared", "roles.b.data")]


def test_nothing_hangs_off_a_universe():
    r = Requirements(
        roles={"b": Universe(min=1, max=2), "f": Future(of="b"), "c": Chain("b", Expiry.weekly(), StrikeWindow.around(1))},
        timeframes=["1m"], holding=Holding.overnight(),
    )
    assert {(i.code, i.path) for i in r.issues()} == {("invalid_base", "roles.f.of"), ("invalid_base", "roles.c.of")}


def test_resolve_names_the_members_and_reports_a_bad_binding():
    r = basket()
    plan = resolve(r, now_ns=T0, bindings={"basket": ["A", "B"]})
    assert plan.roles["basket"].symbols == ("A", "B") and plan.symbols == ("A", "B") and plan.issues == ()
    for binding, code in ((None, "unbound_role"), (["A"], "universe_size"), ("A", "invalid_binding"),
                          (["A", "A"], "invalid_binding"), (["A", "B", "C", "D", "E"], "universe_size")):
        got = resolve(r, now_ns=T0, bindings={} if binding is None else {"basket": binding})
        assert [i.code for i in got.issues] == [code], binding
        assert got.roles["basket"].symbols == ()


def test_ctx_role_returns_the_bound_symbols_in_a_batch_run():
    s = Momentum()
    run_strategies(
        [(s, ["A", "B", "C"])], {"A": bars(100), "B": bars(50), "C": bars(20)}, BacktestConfig(),
        bindings={"basket": ["C", "A"]}, bar_stamp="close",
    )
    assert s.seen == ("C", "A")


def test_ctx_role_returns_the_bound_symbols_on_a_node():
    s = Momentum()
    node = LiveNode([(s, ["A", "B", "C"])], ["A", "B", "C"], BacktestConfig(), bar_stamp="close",
                    bindings={"basket": ["B", "C", "A"]})
    assert node.bindings == {"basket": ("B", "C", "A")}
    for sym in "ABC":
        node.push_bar(sym, MIN, 100.0, 101.0, 99.0, 100.0, 10.0)
    assert s.seen == ("B", "C", "A")


@pytest.mark.parametrize(
    "binding, code",
    [(["A"], "universe_size"), ("A", "invalid_binding"), (["A", "A"], "invalid_binding")],
)
def test_a_bad_universe_binding_is_refused_at_the_call(binding, code):
    with pytest.raises(InvalidParameter) as e:
        run_strategies([(Momentum(), ["A", "B"])], {"A": bars(100), "B": bars(50)}, BacktestConfig(),
                       bindings={"basket": binding}, bar_stamp="close")
    assert (e.value.code, e.value.parameter) == (code, "bindings")


def test_a_symbol_outside_the_run_is_refused_for_a_universe_member():
    with pytest.raises(raptorbt.errors.UnknownSymbol) as e:
        run_strategies([(Momentum(), ["A", "B"])], {"A": bars(100), "B": bars(50)}, BacktestConfig(),
                       bindings={"basket": ["A", "Z"]}, bar_stamp="close")
    assert e.value.code == "unknown_symbol" and e.value.value == "Z"


def test_an_underlying_refuses_a_list():
    class One(Strategy):
        requirements = Requirements(roles={"u": Underlying()}, timeframes=["1m"], holding=Holding.overnight())

    with pytest.raises(InvalidParameter) as e:
        run_strategies([(One(), ["A", "B"])], {"A": bars(100), "B": bars(50)}, BacktestConfig(),
                       bindings={"u": ["A", "B"]}, bar_stamp="close")
    assert e.value.code == "invalid_binding"


def test_ctx_instrument_of_a_universe_role_says_to_ask_for_a_member():
    class Asks(Momentum):
        def on_bar(self, ctx, bar=None):
            ctx.instrument("basket")

    with pytest.raises(InvalidParameter) as e:
        run_strategies([(Asks(), ["A", "B"])], {"A": bars(100), "B": bars(50)}, BacktestConfig(),
                       instruments={"A": InstrumentSpec.equity("A"), "B": InstrumentSpec.equity("B")},
                       bindings={"basket": ["A", "B"]}, bar_stamp="close")
    assert e.value.code == "universe_instrument" and e.value.valid == ["A", "B"]


# -- check_requirements binds every role ----------------------------------------------------


def test_the_dry_run_binds_the_universe_so_ctx_role_works():
    assert check_requirements(Momentum) == []


def test_the_dry_run_binds_every_kind_of_role():
    class Everything(Strategy):
        requirements = Requirements(
            roles={
                "index": Underlying(asset="index"),
                "fut": Future(of="index"),
                "opts": Chain("index", Expiry.weekly(), StrikeWindow.around(2)),
                "hedge": Instrument("HEDGE"),
                "basket": Universe(min=2, max=3),
            },
            timeframes=["1m"], holding=Holding.overnight(),
        )

        def on_init(self, ctx):
            self.subscribe("INDEX")

        def on_bar(self, ctx, bar=None):
            got = {name: ctx.role(name) for name in ("index", "fut", "hedge", "basket", "opts")}
            assert got["index"] == "INDEX" and len(got["basket"]) in (2, 3) and got["opts"]

    issues = check_requirements(Everything)
    assert [i for i in issues if i.code == "dry_run_failed"] == [], issues


def test_the_dry_run_still_reports_a_role_that_cannot_resolve():
    class Bad(Momentum):
        requirements = Requirements(
            roles={"basket": Universe(min=1, max=1), "ghost": Future(of="basket")},
            timeframes=["1m"], holding=Holding.overnight(),
        )

    assert {i.code for i in check_requirements(Bad)} == {"invalid_base"}


# -- the position hint ----------------------------------------------------------------------


def test_max_position_is_validated_serialised_and_hashed():
    r = basket(max_position={"basket": MaxPosition(lots=2, notional=500_000.0, fraction=0.2)})
    assert r.issues() == []
    d = r.to_dict()
    assert d["max_position"] == {"basket": {"lots": 2, "notional": 500_000.0, "fraction": 0.2}}
    back = Requirements.from_dict(json.loads(json.dumps(d)))
    assert back.max_position["basket"] == MaxPosition(2, 500_000.0, 0.2)
    assert requirements_hash(back) == requirements_hash(r) != requirements_hash(basket())
    assert basket().to_dict()["max_position"] is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: MaxPosition(),
        lambda: MaxPosition(lots=0),
        lambda: MaxPosition(lots=1.5),
        lambda: MaxPosition(notional=0),
        lambda: MaxPosition(notional=float("inf")),
        lambda: MaxPosition(fraction=0),
        lambda: MaxPosition(fraction=1.5),
        lambda: basket(max_position={"basket": 3}),
    ],
)
def test_max_position_refuses_bad_values(build):
    with pytest.raises(RequirementsError):
        build()


def test_max_position_must_name_a_role():
    r = basket(max_position={"nope": MaxPosition(lots=1)})
    assert [(i.code, i.path) for i in r.issues()] == [("unknown_role", "max_position.nope")]


# -- the option legs ------------------------------------------------------------------------


def chain_requirements(legs, strikes=StrikeWindow.around(4), **kw):
    return Requirements(
        roles={"index": Underlying(asset="index"), "opts": Chain("index", Expiry.weekly(), strikes)},
        timeframes=["1m"], holding=Holding.intraday("15:15"), max_structure=legs, **kw,
    )


def listing(underlying="IDX", spot=100.0):
    """Two weekly expiries of strikes 90..110 every 2.5, calls and puts."""
    specs = []
    for week in (1, 2):
        for k in range(-4, 5):
            strike = spot + 2.5 * k
            for right in ("call", "put"):
                specs.append(InstrumentSpec.option(
                    f"{underlying}W{week}{strike:g}{right[0].upper()}", strike, right,
                    T0 + week * 7 * DAY, 50.0, underlying=underlying))
    return specs


def test_the_old_count_form_is_refused_with_a_way_forward():
    with pytest.raises(RequirementsError) as e:
        Legs(4, lots=2)
    (issue,) = e.value.issues
    assert issue.code == "legs_need_shape" and "OptionLeg" in issue.message and "max_position" in issue.message
    with pytest.raises(RequirementsError):
        Legs(3)


def test_a_leg_states_side_right_strike_and_expiry_and_round_trips():
    wk = Expiry.weekly()
    legs = Legs(
        OptionLeg(side="sell", right="call", offset=0, expiry=wk),
        OptionLeg(side="buy", right="put", delta=0.25, window=3, expiry=wk, lots=2),
    )
    r = chain_requirements(legs)
    assert r.issues() == []
    d = r.to_dict()
    assert d["max_structure"]["legs"][1] == {
        "side": "buy", "right": "put", "lots": 2, "offset": None, "delta": 0.25, "window": 3,
        "expiry": {"which": "weekly", "n": 0, "min_days": 0, "date_ns": None, "utc_offset_ns": 0, "exact_ns": None},
        "chain": None,
    }
    back = Requirements.from_dict(json.loads(json.dumps(d)))
    assert back.max_structure == legs and requirements_hash(back) == requirements_hash(r)


@pytest.mark.parametrize(
    "kw, field",
    [
        (dict(side="hold", right="call", offset=0), "leg.side"),
        (dict(side="buy", right="straddle", offset=0), "leg.right"),
        (dict(side="buy", right="call"), "leg.strike"),
        (dict(side="buy", right="call", offset=0, delta=0.3, window=2), "leg.strike"),
        (dict(side="buy", right="call", delta=0.3), "leg.window"),
        (dict(side="buy", right="call", delta=1.2, window=2), "leg.delta"),
        (dict(side="buy", right="call", offset=0, window=2), "leg.window"),
        (dict(side="buy", right="call", offset=1.5), "leg.offset"),
        (dict(side="buy", right="call", offset=0, lots=0), "leg.lots"),
    ],
)
def test_a_leg_refuses_bad_values(kw, field):
    with pytest.raises(RequirementsError) as e:
        OptionLeg(expiry=Expiry.weekly(), **kw)
    assert e.value.issues[0].path == field


def test_legs_are_checked_against_the_chain_they_sit_on():
    wk = Expiry.weekly()
    ok = lambda **kw: chain_requirements(Legs(OptionLeg(side="buy", right="call", expiry=wk, **kw)))
    assert ok(offset=4).issues() == [] and ok(delta=0.2, window=4).issues() == []
    far = ok(offset=5).issues()
    assert [(i.code, i.path) for i in far] == [("leg_outside_chain", "max_structure.legs[0].strike")]
    assert [i.code for i in ok(delta=0.2, window=5).issues()] == ["leg_outside_chain"]
    put_chain = chain_requirements(Legs(OptionLeg(side="buy", right="call", offset=0, expiry=wk)))
    only_puts = Requirements(**{**put_chain.__dict__, "roles": {
        "index": Underlying(), "opts": Chain("index", wk, StrikeWindow.around(4), ("put",))}})
    assert [i.path for i in only_puts.issues()] == ["max_structure.legs[0].right"]
    other_expiry = chain_requirements(Legs(OptionLeg(side="buy", right="call", offset=0, expiry=Expiry.monthly())))
    assert [i.path for i in other_expiry.issues()] == ["max_structure.legs[0].expiry"]
    by_delta = chain_requirements(Legs(OptionLeg(side="buy", right="call", offset=0, expiry=wk)), strikes=ByDelta(0.1, 0.5))
    assert [i.code for i in by_delta.issues()] == ["leg_outside_chain"]
    in_band = chain_requirements(Legs(OptionLeg(side="buy", right="call", delta=0.3, window=2, expiry=wk)), strikes=ByDelta(0.1, 0.5))
    assert in_band.issues() == []


def test_which_chain_a_leg_is_on_must_be_clear():
    wk = Expiry.weekly()
    two = Requirements(
        roles={"i": Underlying(), "a": Chain("i", wk, StrikeWindow.around(2)), "b": Chain("i", wk, StrikeWindow.around(2))},
        timeframes=["1m"], holding=Holding.overnight(),
        max_structure=Legs(OptionLeg(side="buy", right="call", offset=0, expiry=wk)),
    )
    assert [i.code for i in two.issues()] == ["leg_chain_ambiguous"]
    named = Requirements(**{**two.__dict__, "max_structure": Legs(OptionLeg(side="buy", right="call", offset=0, expiry=wk, chain="b"))})
    assert named.issues() == []
    unknown = Requirements(**{**two.__dict__, "max_structure": Legs(OptionLeg(side="buy", right="call", offset=0, expiry=wk, chain="z"))})
    assert [i.code for i in unknown.issues()] == ["unknown_role"]
    nochain = Requirements(roles={"i": Underlying()}, timeframes=["1m"], holding=Holding.overnight(),
                           max_structure=Legs(OptionLeg(side="buy", right="call", offset=0, expiry=wk)))
    assert [i.code for i in nochain.issues()] == ["leg_chain_ambiguous"]


def condor_legs():
    wk = Expiry.weekly()
    return Legs(
        OptionLeg(side="sell", right="call", offset=1, expiry=wk),
        OptionLeg(side="buy", right="call", offset=3, expiry=wk),
        OptionLeg(side="sell", right="put", delta=0.3, window=2, expiry=wk),
    )


def test_resolve_legs_lists_the_contracts_to_subscribe_and_the_units():
    r = chain_requirements(condor_legs())
    plan = resolve_legs(r, now_ns=T0, bindings={"index": "IDX"}, specs=listing(), prices={"IDX": 100.0})
    assert plan.issues == ()
    sell_call, buy_call, delta_put = plan.legs
    # the money strike is 100; one listed step is 2.5
    assert (sell_call.specs[0].strike, buy_call.specs[0].strike) == (102.5, 107.5)
    assert (sell_call.sign, buy_call.sign, delta_put.sign) == (-1, 1, -1)
    assert sell_call.quantity == 50.0
    # a delta leg is every put within 2 strikes of the money: the run picks the one at the delta
    assert sorted(s.strike for s in delta_put.specs) == [95.0, 97.5, 100.0, 102.5, 105.0]
    assert set(plan.symbols) == {s for c in plan.legs for s in c.symbols}
    assert all(s.expiration_ns == T0 + 7 * DAY for c in plan.legs for s in c.specs)
    json.dumps(plan.to_dict())


def test_the_contracts_a_plan_lists_are_inside_what_the_chain_role_resolves_to():
    r = chain_requirements(condor_legs())
    specs = listing()
    legs = resolve_legs(r, now_ns=T0, bindings={"index": "IDX"}, specs=specs, prices={"IDX": 100.0})
    role = resolve(r, now_ns=T0, bindings={"index": "IDX"}, specs=specs, prices={"IDX": 100.0})
    assert set(legs.symbols) <= set(role.symbols)


def test_a_leg_plan_margins_with_a_margin_model():
    r = chain_requirements(Legs(
        OptionLeg(side="sell", right="call", offset=1, expiry=Expiry.weekly()),
        OptionLeg(side="buy", right="call", offset=3, expiry=Expiry.weekly()),
    ))
    plan = resolve_legs(r, now_ns=T0, bindings={"index": "IDX"}, specs=listing(), prices={"IDX": 100.0})
    prices = {c.symbols[0]: p for c, p in zip(plan.legs, (3.0, 1.0))}
    lines = plan.lines(prices)
    assert [(q, p) for _, q, p in lines] == [(-50.0, 3.0), (50.0, 1.0)]
    assert MarginModel.defined_risk().required_legs(lines, T0) > 0.0


def test_the_structure_of_the_legs_margins_through_the_structure_book():
    legs = chain_requirements(condor_legs()).max_structure
    with pytest.raises(RequirementsError) as e:
        legs.structure()
    assert e.value.issues[0].code == "needs_vol"
    chain = raptorbt.OptionChain(listing())
    resolved = legs.structure(vol=0.2).resolve(chain, "IDX", 100.0, T0)
    assert [(leg.side, leg.right, leg.lots) for leg in resolved.legs] == [("sell", "call", 1), ("buy", "call", 1), ("sell", "put", 1)]
    prices = {leg.instrument: 2.0 for leg in resolved.legs}
    assert resolved.margin(MarginModel.option_simple(20.0), prices, T0) > 0.0


def test_resolve_legs_reports_what_it_cannot_resolve():
    r = chain_requirements(condor_legs())
    no_price = resolve_legs(r, now_ns=T0, bindings={"index": "IDX"}, specs=listing())
    assert {i.code for i in no_price.issues} == {"needs_price"} and no_price.legs == ()
    no_options = resolve_legs(r, now_ns=T0, bindings={"index": "IDX"}, specs=[], prices={"IDX": 100.0})
    assert {i.code for i in no_options.issues} == {"no_options"}
    off_ladder = chain_requirements(Legs(OptionLeg(side="buy", right="call", offset=4, expiry=Expiry.weekly())))
    edge = resolve_legs(off_ladder, now_ns=T0, bindings={"index": "IDX"}, specs=listing(), prices={"IDX": 107.5})
    assert [i.code for i in edge.issues] == ["leg_strike_unlisted"]
    assert resolve_legs(basket(), now_ns=T0).legs == ()
