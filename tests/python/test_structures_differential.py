"""The structures and sizing helpers against a reference implementation.

``golden/structures_reference.json`` holds what a reference implementation picked for each structure on a
fixed chain. The new code must pick the same
contracts, sides and sizes, or the difference is listed below and asserted.

Intended differences
--------------------
* A rule that comes to less than one lot raises ``below_one_lot``; the reference ``qty_from_risk``
  returned 0.
* A contract the chain does not list raises ``structure_contract_missing`` (an ``InvalidParameter``),
  where the reference raised ``KeyError``.
* A stop equal to the entry raises ``invalid_sizing_input`` in both.
"""

import json
from pathlib import Path

import pytest

from raptorbt import InstrumentSpec, OptionChain, sizing
from raptorbt import structures as st
from raptorbt.errors import InvalidParameter

REF = json.loads((Path(__file__).parent / "golden" / "structures_reference.json").read_text())
T0, NEAR, FAR, LOT = REF["t0_ns"], REF["near_ns"], REF["far_ns"], REF["lot"]
DAY = 86_400 * 10**9
UND = "IDX"

BUILD = {
    "straddle": lambda a: st.straddle(),
    "strangle": lambda a: st.strangle(a["steps"]),
    "risk_reversal": lambda a: st.risk_reversal(a["steps"]),
    "iron_fly": lambda a: st.iron_butterfly(a["wing_steps"]),
    "butterfly": lambda a: st.butterfly(_r(a), a["wing_steps"]),
    "iron_condor": lambda a: st.iron_condor(a["inner_steps"], a["outer_steps"]),
    "condor": lambda a: st.condor(_r(a), a["inner_steps"], a["outer_steps"]),
    "vertical": lambda a: st.vertical(_r(a), a["long_steps"], a["short_steps"]),
    "calendar": lambda a: st.calendar(
        _r(a), a["steps"], near=st.Expiry.at(NEAR), far=st.Expiry.at(FAR)
    ),
    "covered_call": lambda a: st.covered_call(a["call_steps"]),
    "collar": lambda a: st.collar(a["put_steps"], a["call_steps"]),
}


def _r(a):
    return "call" if a["right"] == "C" else "put"


def _spec(und, strike, right, expiry):
    return InstrumentSpec.option(
        f"{und}{'N' if expiry == NEAR else 'F'}{int(strike)}{right[0]}",
        float(strike), right, expiry, float(LOT), underlying=und,
    )


def _chain(und):
    specs = []
    for e, ks in ((NEAR, REF["strikes"]), (FAR, REF["far_strikes"])):
        for k in ks:
            for r in ("call", "put"):
                specs.append(_spec(und, k, r, e))
    return OptionChain(specs)


def _pin_expiry(structure):
    """The reference picks expiry NEAR for every leg (calendars aside): pin them."""
    legs = []
    for leg in structure.legs:
        if isinstance(leg, st.Leg) and leg.expiry == st.Expiry():
            leg = st.Leg(leg.right, leg.strike, st.Expiry.at(NEAR), leg.side, leg.ratio)
        legs.append(leg)
    return st.Structure(tuple(legs), structure.name)


def _rows(resolved, equity_lot=None):
    rows = []
    for leg in resolved.legs:
        if leg.is_option:
            rows.append([leg.strike, leg.right, (leg.expiry_ns - T0) // DAY, leg.side, leg.lots])
        else:
            rows.append(["shares", None, None, leg.side, int(leg.quantity // equity_lot)])
    return rows


def _case_id(c):
    return f"{c['name']}-{c['args']}-{c['spot']}-{c['lots']}-{c.get('equity_lot', '')}"


@pytest.mark.parametrize("case", REF["structures"], ids=_case_id)
def test_option_structures_pick_what_the_reference_picked(case):
    chain = _chain(UND)
    structure = _pin_expiry(BUILD[case["name"]](case["args"]))
    if "error" in case:
        with pytest.raises(InvalidParameter) as e:
            structure.resolve(chain, UND, case["spot"], T0, case["lots"])
        assert e.value.code == "structure_contract_missing"
        return
    got = structure.resolve(chain, UND, case["spot"], T0, case["lots"])
    assert _rows(got) == case["legs"]
    for leg in got.option_legs:
        assert leg.quantity == leg.lots * LOT


@pytest.mark.parametrize("case", REF["equity_structures"], ids=_case_id)
def test_equity_structures_pick_what_the_reference_picked(case):
    und = "EQ"
    chain = _chain(und)
    eq = InstrumentSpec.equity(und, lot_size=float(case["equity_lot"]))
    structure = _pin_expiry(BUILD[case["name"]](case["args"]))
    if "error" in case:
        with pytest.raises(InvalidParameter) as e:
            structure.resolve(chain, und, case["spot"], T0, case["lots"], underlying_spec=eq)
        assert e.value.code == "structure_contract_missing"
        return
    got = structure.resolve(chain, und, case["spot"], T0, case["lots"], underlying_spec=eq)
    assert _rows(got, case["equity_lot"]) == case["legs"]


@pytest.mark.parametrize("case", REF["risk_sizing"], ids=lambda c: str(c))
def test_risk_sizing_matches_qty_from_risk(case):
    # The reference helper takes a money budget; the new one a fraction of capital. Same arithmetic.
    capital = 1_000_000.0
    fraction = case["risk"] / capital
    want_lots = case["units"] // case["lot"]
    if want_lots == 0:
        with pytest.raises(InvalidParameter) as e:
            sizing.lots_for_risk(capital, fraction, case["entry"], case["stop"], case["lot"])
        assert e.value.code == "below_one_lot"
        return
    got = sizing.lots_for_risk(capital, fraction, case["entry"], case["stop"], case["lot"])
    assert got == want_lots
