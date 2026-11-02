"""OptionChain: expiry and strike selection with no lookahead."""

import pytest

import raptorbt
from raptorbt import OptionChain, pricing as pr

DAY = 86_400 * 10**9
T0 = 1_767_225_600 * 10**9  # 2026-01-01 00:00 UTC
U = "IDX"


def opt(strike, right, expiry, activation=None, underlying=U):
    return raptorbt.InstrumentSpec.option(
        f"{underlying}-{expiry // DAY}-{strike:g}-{right[0]}",
        strike,
        right,
        expiry,
        50.0,
        underlying=underlying,
        activation_ns=activation,
    )


def weekly_chain(listed_after=None):
    specs = [raptorbt.InstrumentSpec.index(U)]
    for w in range(1, 6):
        e = T0 + 7 * w * DAY
        for k in range(90, 111, 5):
            for right in ("call", "put"):
                specs.append(opt(float(k), right, e))
    return OptionChain(specs)


def test_lists_are_ascending_and_ignore_non_options():
    c = weekly_chain()
    assert len(c) == 5 * 5 * 2
    assert c.underlyings() == [U]
    e = [T0 + 7 * w * DAY for w in range(1, 6)]
    assert c.expiries(U, T0) == e
    assert c.expiries(U, e[0]) == e[1:], "an expiry is gone at its own timestamp"
    assert c.strikes(U, e[0], T0) == [90.0, 95.0, 100.0, 105.0, 110.0]
    assert c.expiries("OTHER", T0) == [] and c.strikes(U, 1, T0) == []


def test_contracts_are_invisible_until_listed():
    e = T0 + 7 * DAY
    listed = T0 + 3 * DAY
    c = OptionChain([opt(100.0, "call", e), opt(105.0, "call", e, activation=listed)])
    assert c.strikes(U, e, listed - 1) == [100.0]
    assert c.strikes(U, e, listed) == [100.0, 105.0]
    assert c.contract(U, e, 105.0, "call", listed - 1) is None
    assert c.contract(U, e, 105.0, "call", listed).symbol.endswith("105-c")
    # ATM follows what is listed, not what will be.
    assert c.atm_strike(U, e, listed - 1, 106.0) == 100.0
    assert c.atm_strike(U, e, listed, 106.0) == 105.0
    # A whole expiry appears only once something in it is listed.
    later = T0 + 30 * DAY
    c = OptionChain([opt(100.0, "put", later, activation=T0 + 10 * DAY)])
    assert c.expiries(U, T0) == []
    assert c.expiries(U, T0 + 10 * DAY) == [later]
    assert c.pick_expiry(U, T0) is None


def test_pick_expiry_variants():
    c = weekly_chain()
    e = [T0 + 7 * w * DAY for w in range(1, 6)]
    now = T0 + 2 * DAY
    assert c.pick_expiry(U, now) == e[0]
    assert c.pick_expiry(U, now, min_days=6) == e[1]
    assert c.pick_expiry(U, now, "nth", n=2) == e[2]
    assert c.pick_expiry(U, now, "nth", n=9) is None
    assert c.pick_expiry(U, now, "date", date_ns=e[3] + 5 * 3600 * 10**9) == e[3]
    assert c.pick_expiry(U, now, "date", date_ns=e[3] + DAY) is None
    # January 29 closes January; February 5 is February's only (last) expiry here.
    assert c.pick_expiry(U, now, "monthly", n=0) == e[3]
    assert c.pick_expiry(U, now, "monthly", n=1) == e[4]
    assert [c.pick_expiry(U, now, "weekly", n=i) for i in range(4)] == [e[0], e[1], e[2], None]
    with pytest.raises(ValueError):
        c.pick_expiry(U, now, "fortnightly")
    with pytest.raises(ValueError):
        c.pick_expiry(U, now, "date")


def test_monthly_classification_does_not_move_with_time():
    c = weekly_chain()
    e4 = T0 + 28 * DAY
    assert c.pick_expiry(U, T0, "monthly") == e4
    assert c.pick_expiry(U, T0 + 21 * DAY + 3600 * 10**9, "monthly") == e4


def test_strike_selection_by_atm_offset_and_forward():
    c = weekly_chain()
    e = T0 + 7 * DAY
    assert c.atm_strike(U, e, T0, 101.0) == 100.0
    assert c.atm_strike(U, e, T0, 102.5) == 100.0, "ties go to the lower strike"
    assert c.atm_strike(U, e, T0, 102.6) == 105.0
    assert c.atm_strike(U, e, T0, 1e6) == 110.0
    assert c.strike_at_offset(U, e, T0, 101.0, 2) == 110.0
    assert c.strike_at_offset(U, e, T0, 101.0, -2) == 90.0
    assert c.strike_at_offset(U, e, T0, 101.0, 3) is None
    # The forward of 100 a year out at 8% carry is ~108: ATM by forward is 110.
    far = T0 + 28 * DAY
    assert c.atm_strike(U, far, T0, 100.0) == 100.0
    fwd = pr.forward_price(100.0, pr.year_fraction(T0, far), 0.8)
    assert c.atm_strike_forward(U, far, T0, 100.0, rate=0.8) == c.atm_strike(U, far, T0, fwd)
    assert c.atm_strike_forward(U, far, T0, 100.0, rate=0.8) > 100.0


def test_by_delta_and_by_premium():
    c = weekly_chain()
    e = T0 + 7 * DAY
    call = c.by_delta(U, e, T0, "call", 0.25, 100.0, 0.5)
    put = c.by_delta(U, e, T0, "put", -0.25, 100.0, 0.5)
    assert call.strike > 100.0 > put.strike
    # The pick really is the nearest delta among the listed calls.
    tau = pr.year_fraction(T0, e)
    deltas = {
        s.strike: pr.greeks("call", 100.0, s.strike, tau, 0.5).delta
        for s in c.contracts(U, e, T0, "call")
    }
    assert abs(deltas[call.strike] - 0.25) == min(abs(d - 0.25) for d in deltas.values())
    # Black-76 on a forward.
    assert c.by_delta(U, e, T0, "call", 0.5, 100.0, 0.5, model="black76").strike == 100.0

    quotes = {90.0: 10.5, 95.0: 6.0, 100.0: 3.2, 105.0: 1.0, 110.0: 0.2}
    got = c.closest_by(U, e, T0, "call", 3.0, lambda s: quotes.get(s.strike))
    assert got.strike == 100.0
    # Unquoted contracts are skipped (None), not treated as zero.
    got = c.closest_by(
        U, e, T0, "call", 0.0, lambda s: quotes.get(s.strike) if s.strike < 105 else None
    )
    assert got.strike == 100.0
    assert c.closest_by(U, e, T0, "call", 1.0, lambda s: None) is None
    with pytest.raises(RuntimeError, match="boom"):
        c.closest_by(U, e, T0, "call", 1.0, lambda s: (_ for _ in ()).throw(RuntimeError("boom")))


def test_contracts_come_back_in_strike_order():
    c = weekly_chain()
    e = T0 + 7 * DAY
    both = c.contracts(U, e, T0)
    assert [s.strike for s in both] == sorted(s.strike for s in both)
    assert len(both) == 10
    assert {s.right for s in c.contracts(U, e, T0, "put")} == {"put"}
    assert c.contract(U, e, 100.0, "put", T0).right == "put"
    assert c.contract(U, e, 101.0, "put", T0) is None


def test_bad_chains_are_refused():
    e = T0 + DAY
    a = opt(100.0, "call", e)
    b = raptorbt.InstrumentSpec.option("OTHER", 100.0, "call", e, 50.0, underlying=U)
    with pytest.raises(ValueError, match="two contracts"):
        OptionChain([a, b])
    with pytest.raises(ValueError, match="underlying"):
        OptionChain([raptorbt.InstrumentSpec.option("X", 100.0, "call", e, 50.0)])
    assert len(OptionChain([])) == 0
    with pytest.raises(ValueError):
        weekly_chain().strikes(U, e, T0, "straddle")
