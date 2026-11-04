"""The strategy manifest: declaration, parameters, checks, hash, JSON and resolution."""

import json
from dataclasses import dataclass
from typing import Literal

import pytest

import raptorbt
from raptorbt import InstrumentSpec
from raptorbt.structures import Expiry
from raptorbt.strategy import (
    Bars,
    ByDelta,
    Chain,
    Duration,
    Future,
    Holding,
    Instrument,
    Legs,
    MaxPosition,
    OptionLeg,
    Requirements,
    RequirementsError,
    Strategy,
    StrategyConfig,
    StrikeWindow,
    Underlying,
    check_requirements,
    param,
    params_schema,
    requirements_hash,
    resolve,
)

DAY = 86_400 * 10**9
T0 = 1_767_225_600 * 10**9  # 2026-01-01 00:00 UTC


@dataclass(frozen=True)
class Config(StrategyConfig):
    fast: int = param(10, min=2, max=200, description="fast period")
    mode: str = param("long", choices=["long", "short"])
    scale: float = 1.5
    on: bool = True
    side: Literal["a", "b"] = "a"
    cap: int | None = None
    must: int = 3


def condor() -> Legs:
    wk = Expiry.weekly()
    return Legs(
        OptionLeg(side="sell", right="call", offset=1, expiry=wk),
        OptionLeg(side="buy", right="call", offset=2, expiry=wk),
        OptionLeg(side="sell", right="put", offset=-1, expiry=wk),
        OptionLeg(side="buy", right="put", offset=-2, expiry=wk, lots=2),
    )


def full() -> Requirements:
    return Requirements(
        roles={
            "index": Underlying(asset="index"),
            "fut": Future(of="index", contract="continuous"),
            "opts": Chain(of="index", expiry=Expiry.weekly(), strikes=StrikeWindow.around(2)),
            "hedge": Instrument("HEDGE"),
        },
        timeframes=["5m", "1m"],
        holding=Holding.intraday(flatten_before="15:15"),
        data=["bar"],
        warmup=Bars(50),
        max_structure=condor(),
        max_position={"opts": MaxPosition(lots=2), "hedge": MaxPosition(fraction=0.5)},
        params=Config,
    )


class Good(Strategy):
    requirements = Requirements(
        roles={"index": Underlying(asset="index")},
        timeframes=["1m", "5m"],
        holding=Holding.overnight(),
    )

    def on_init(self, ctx):
        self.subscribe("INDEX")
        self.subscribe("INDEX", timeframe="5m")


# -- declaration ------------------------------------------------------------


def test_declaration_is_data_on_the_class():
    assert Strategy.requirements is None
    assert Good.requirements.feed_interval == "1m"
    assert [t.label for t in Good.requirements.timeframes] == ["1m", "5m"]
    assert full().timeframes == full().timeframes  # sorted: 1m before 5m
    assert [t.label for t in full().timeframes] == ["1m", "5m"]


@pytest.mark.parametrize(
    "build, code",
    [
        (lambda: Underlying("bond"), "invalid_value"),
        (lambda: Future("x", "last"), "invalid_value"),
        (lambda: Holding("intraday"), "invalid_value"),
        (lambda: Holding.intraday("25:00"), "invalid_value"),
        (lambda: StrikeWindow(3, 1), "invalid_value"),
        (lambda: ByDelta(0.0, 0.5), "invalid_value"),
        (lambda: Legs(4, lots=2), "legs_need_shape"),
        (lambda: Legs(), "invalid_value"),
        (lambda: Bars(5, "7m"), "invalid_timeframe"),
        (lambda: Requirements(timeframes=["7m"]), "invalid"),
        (lambda: Requirements(data=["ticks"]), "invalid_data"),
        (lambda: Requirements(roles={"a b": Underlying()}), "invalid_role_name"),
    ],
)
def test_parts_refuse_bad_values(build, code):
    with pytest.raises((RequirementsError, raptorbt.errors.RaptorBTError)) as e:
        build()
    if isinstance(e.value, RequirementsError):
        assert e.value.issues[0].code == code


def test_cross_role_issues():
    r = Requirements(
        roles={
            "a": Underlying(),
            "f": Future(of="missing"),
            "c": Chain(of="a", expiry=Expiry.nth(0), strikes=StrikeWindow.around(1)),
            "cc": Chain(of="c", expiry=Expiry.nth(0), strikes=StrikeWindow.around(1)),
        },
        timeframes=["1m"],
        holding=Holding.overnight(),
        warmup=Bars(5, "5m"),
    )
    codes = {(i.code, i.path) for i in r.issues()}
    assert ("unknown_role", "roles.f.of") in codes
    assert ("invalid_base", "roles.cc.of") in codes
    assert ("undeclared_timeframe", "warmup.timeframe") in codes
    assert [i.code for i in Requirements().issues()] == ["no_roles", "missing_holding", "no_timeframes"]


# -- parameters ---------------------------------------------------------------


def test_params_schema():
    schema = params_schema(Config)
    by = {p.name: p for p in schema.params}
    assert [p.name for p in schema.params] == ["fast", "mode", "scale", "on", "side", "cap", "must"]
    assert (by["fast"].type, by["fast"].min, by["fast"].max, by["fast"].default) == ("int", 2, 200, 10)
    assert by["mode"].choices == ("long", "short")
    assert by["side"].choices == ("a", "b") and by["side"].type == "str"
    assert by["cap"].nullable and by["cap"].default is None
    assert by["must"].required is False
    assert by["fast"].description == "fast period"


def test_params_validation_is_structured():
    schema = params_schema(Config)
    assert schema.validate({"fast": 5, "mode": "short"}) == []
    got = {(i.code, i.path) for i in schema.validate({"fast": 1, "mode": "x", "scale": "1", "nope": 1, "on": 1})}
    assert got == {
        ("out_of_bounds", "params.fast"),
        ("not_a_choice", "params.mode"),
        ("invalid_param", "params.scale"),
        ("unknown_param", "params.nope"),
        ("invalid_param", "params.on"),
    }
    assert schema.validate({"fast": float("nan")})[0].code == "invalid_param"


def test_params_refuse_what_they_cannot_describe():
    @dataclass(frozen=True)
    class Bad(StrategyConfig):
        xs: list = None
        n: int = param(5, min=10)

    with pytest.raises(RequirementsError) as e:
        params_schema(Bad)
    assert {i.code for i in e.value.issues} == {"unsupported_type", "invalid_default"}


def test_required_param_without_default():
    @dataclass(frozen=True, kw_only=True)
    class Req(StrategyConfig):
        n: int = param(min=1)

    schema = params_schema(Req)
    assert schema.params[0].required
    assert schema.validate({})[0].code == "missing_param"


# -- hash and JSON ---------------------------------------------------------------


def test_hash_is_stable_and_order_free():
    a = full()
    b = Requirements(
        roles=dict(reversed(list(full().roles.items()))),
        timeframes=["1m", "5m", "1m"],
        holding=Holding.intraday("15:15"),
        data=("bar",),
        warmup=Bars(50),
        max_structure=condor(),
        max_position={"hedge": MaxPosition(fraction=0.5), "opts": MaxPosition(lots=2)},
        params=Config,
    )
    assert requirements_hash(a) == requirements_hash(b) == requirements_hash(a)
    assert len(requirements_hash(a)) == 64


def test_hash_moves_with_declaration_and_version(monkeypatch):
    base = requirements_hash(full())
    other = Requirements(**{**full().__dict__, "warmup": Bars(51), "params": full().params})
    assert requirements_hash(other) != base
    monkeypatch.setattr(raptorbt, "__version__", "9.9.9")
    assert requirements_hash(full()) != base


def test_hash_of_a_class_and_of_nothing():
    assert requirements_hash(Good) == requirements_hash(Good.requirements)
    with pytest.raises(RequirementsError) as e:
        requirements_hash(Strategy)
    assert e.value.issues[0].code == "no_requirements"


def test_json_round_trip():
    r = full()
    d = r.to_dict()
    assert json.loads(json.dumps(d)) == d
    back = Requirements.from_dict(json.loads(json.dumps(d)))
    assert back.to_dict() == d
    assert requirements_hash(back) == requirements_hash(r)
    assert d["schema"] == 2 and d["roles"]["opts"]["strikes"] == {"by": "window", "low": -2, "high": 2}
    with pytest.raises(RequirementsError):
        Requirements.from_dict({**d, "schema": 1})


def test_json_round_trip_variants():
    r = Requirements(
        roles={
            "u": Underlying(),
            "c": Chain("u", (Expiry.monthly(1), Expiry.nearest(3)), ByDelta(0.1, 0.4), ("put",)),
        },
        timeframes=["1d", Timeframe_renko()],
        holding=Holding.overnight(),
        data=["bar", "quote"],
        warmup=Duration.days(5),
    )
    assert Requirements.from_dict(r.to_dict()).to_dict() == r.to_dict()


def Timeframe_renko():
    from raptorbt.strategy import Timeframe

    return Timeframe.renko(2.5)


# -- check_requirements -------------------------------------------------------------


def test_check_passes_a_consistent_strategy():
    assert check_requirements(Good) == []


def test_check_reports_undeclared_subscriptions():
    class Wide(Strategy):
        requirements = Good.requirements

        def on_init(self, ctx):
            self.subscribe("INDEX", timeframe="15m")
            self.subscribe("INDEX", data_type="quote")
            self.subscribe_bars(30, "m")

    got = {i.code for i in check_requirements(Wide)}
    assert got == {"undeclared_timeframe", "undeclared_data"}


def test_check_reports_unknown_instrument_as_dry_run_failure():
    class Stray(Strategy):
        requirements = Good.requirements

        def on_init(self, ctx):
            self.subscribe("OTHER")

    assert {i.code for i in check_requirements(Stray)} == {"dry_run_failed", "undeclared_instrument"}


def test_check_without_requirements_and_static_only():
    assert check_requirements(Strategy)[0].code == "no_requirements"

    class Broken(Strategy):
        requirements = Requirements(roles={"f": Future("zz")}, timeframes=["1m"], holding=Holding.overnight())

    assert [i.code for i in check_requirements(Broken)] == ["unknown_role"]


def test_check_dry_runs_chains_and_futures():
    class Straddle(Strategy):
        requirements = full()

        def on_init(self, ctx):
            self.subscribe("INDEX")

    assert check_requirements(Straddle) == []


# -- resolution --------------------------------------------------------------------


def listed():
    out = []
    for n, f in enumerate(("IDXF1", "IDXF2", "IDXF0")):
        out.append(InstrumentSpec.futures_contract(f, T0 + (n + 1) * 30 * DAY if f != "IDXF0" else T0 - DAY, 25.0, underlying="IDX"))
    # weekly expiries on 8, 15, 22, 29 Jan; 5, 26 Feb (Jan 29 and Feb 26 are each month's last)
    for e in (7, 14, 21, 28, 35, 56):
        for k in (90.0, 95.0, 100.0, 105.0, 110.0):
            for r in ("call", "put"):
                out.append(InstrumentSpec.option(f"IDX{e}{int(k)}{r[0].upper()}", k, r, T0 + e * DAY, 25.0, underlying="IDX"))
    return out


def test_resolve_roles():
    r = Requirements(
        roles={
            "index": Underlying(asset="index"),
            "front": Future("index"),
            "nxt": Future("index", "next"),
            "cont": Future("index", "continuous"),
            "w": Chain("index", Expiry.weekly(1), StrikeWindow(-1, 2), ("call",)),
            "m": Chain("index", Expiry.monthly(), StrikeWindow.around(0), ("put",)),
            "hedge": Instrument("HEDGE"),
        },
        timeframes=["1m"],
        holding=Holding.overnight(),
    )
    plan = resolve(r, now_ns=T0 + DAY // 2, bindings={"index": "IDX"}, specs=listed(), prices={"IDX": 101.0})
    assert plan.issues == ()
    p = plan.roles
    assert p["index"].symbols == ("IDX",)
    assert p["front"].symbols == ("IDXF1",) and p["nxt"].symbols == ("IDXF2",)  # IDXF0 expired
    assert p["cont"].symbols == ("IDXF1",) and p["cont"].detail["roll_to"] == "IDXF2"
    assert p["w"].expiries == (T0 + 14 * DAY,) and p["w"].atm == 100.0
    assert p["w"].symbols == ("IDX1495C", "IDX14100C", "IDX14105C", "IDX14110C")
    assert p["m"].expiries == (T0 + 28 * DAY,) and p["m"].symbols == ("IDX28100P",)
    assert plan.symbols[0] == "IDX" and plan.symbols[-1] == "HEDGE" and len(set(plan.symbols)) == len(plan.symbols)
    assert json.loads(json.dumps(plan.to_dict()))["symbols"] == list(plan.symbols)


def test_resolve_follows_time():
    r = Requirements(
        roles={"index": Underlying(), "f": Future("index")},
        timeframes=["1m"],
        holding=Holding.overnight(),
    )
    at = lambda t: resolve(r, now_ns=t, bindings={"index": "IDX"}, specs=listed()).roles["f"].symbols  # noqa: E731
    assert at(T0) == ("IDXF1",)
    assert at(T0 + 31 * DAY) == ("IDXF2",)
    assert at(T0 + 61 * DAY) == ()


def test_resolve_window_clips_at_the_ends():
    r = Requirements(
        roles={"u": Underlying(), "c": Chain("u", Expiry.nth(0), StrikeWindow.around(9))},
        timeframes=["1m"],
        holding=Holding.overnight(),
    )
    plan = resolve(r, now_ns=T0, bindings={"u": "IDX"}, specs=listed(), prices={"u": 0, "IDX": 100.0})
    assert len(plan.roles["c"].symbols) == 10  # five strikes, both rights


def test_resolve_by_delta_band():
    r = Requirements(
        roles={"u": Underlying(), "c": Chain("u", Expiry.nth(0), ByDelta(0.2, 0.6), ("call",))},
        timeframes=["1m"],
        holding=Holding.overnight(),
    )
    specs = listed()
    plan = resolve(r, now_ns=T0, bindings={"u": "IDX"}, specs=specs, prices={"IDX": 100.0}, vols={"IDX": 0.2})
    assert plan.issues == ()
    # 7 days at 20% vol: delta 0.6 is the 100 strike, delta 0.2 the 105
    assert plan.roles["c"].symbols == ("IDX7100C", "IDX7105C")
    assert not resolve(r, now_ns=T0, bindings={"u": "IDX"}, specs=specs, prices={"IDX": 100.0}).roles["c"].symbols


def test_resolve_reports_what_it_lacks():
    r = Requirements(
        roles={"u": Underlying(), "f": Future("u"), "c": Chain("u", Expiry.weekly(9), StrikeWindow.around(1))},
        timeframes=["1m"],
        holding=Holding.overnight(),
    )
    plan = resolve(r, now_ns=T0, specs=[])
    assert {i.code for i in plan.issues} == {"unbound_role"}
    plan = resolve(r, now_ns=T0, bindings={"u": "IDX"}, specs=listed())
    assert {i.code for i in plan.issues} == {"no_expiry", "needs_price"}
    plan = resolve(r, now_ns=T0, bindings={"u": "NONE"}, specs=listed(), prices={"NONE": 1.0})
    assert {i.code for i in plan.issues} >= {"no_contract", "no_expiry"}
    broken = resolve(Requirements(roles={"f": Future("zz")}), now_ns=T0)
    assert [i.code for i in broken.issues] == ["unknown_role"]


def test_resolve_matches_the_dry_run_universe():
    # The same function resolves a host's roles and the check's synthetic ones.
    assert check_requirements(type("S", (Strategy,), {"requirements": full()})) == []
