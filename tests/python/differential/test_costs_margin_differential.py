"""Costs and margin against the reference implementation, replayed from fixtures.

The fixture holds the numbers a reference implementation produced for these inputs. These tests replay them through this engine.
Charges are to the paisa, so they must agree to the paisa; margin figures are
plain arithmetic and must agree to a few ulps. The differences that are
intended (rates, accepted documents, zero-value fills) are listed in
``docs/quantitative-correctness/numerics.md``; none shows up in these grids.
"""

import json
from pathlib import Path

import pytest

import raptorbt as r

HERE = Path(__file__).parent


def _load(name):
    return json.loads((HERE / name).read_text())


COSTS = _load("costs_fixture.json")
MARGIN = _load("margin_fixture.json")

PAISA = 1e-9  # both sides are quantised to the paisa: equal means equal


def _rate_row(card_row):
    return (card_row["exchange"], card_row["segment"], card_row["underlying"])


# --- costs ----------------------------------------------------------------


@pytest.fixture(scope="module")
def card_schedule():
    return r.RateCard.from_dict(COSTS["card"]).fee_schedule()


def test_contract_note_charges_match_to_the_paisa(card_schedule):
    rows = COSTS["card"]["rates"]
    cols = COSTS["case_columns"]
    bad = []
    for case in COSTS["cases"]:
        c = dict(zip(cols, case))
        venue, segment, underlying = _rate_row(rows[c["row"]])
        got = card_schedule.charge(
            venue, segment, c["side"], float(c["price"]), float(c["quantity"]),
            multiplier=float(c["multiplier"]), underlying=underlying,
            order_turnover=float(c["order_before"]),
        )
        for k in ("brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst", "total"):
            if abs(got[k] - c[k]) > PAISA:
                bad.append((case, k, got[k], c[k]))
        if abs(got["exact_total"] - c["exact_total"]) > PAISA:
            bad.append((case, "exact_total", got["exact_total"], c["exact_total"]))
    assert len(COSTS["cases"]) > 2000
    assert not bad, f"{len(bad)} of {len(COSTS['cases'])} differ, first: {bad[:3]}"


def test_paise_each_presets_match_to_the_paisa():
    schedule = r.RateCard.from_dict(
        {
            "name": "illustrative-2026-09",
            "effective_from": "2026-09-01T00:00:00+05:30",
            "rates": COSTS["illustrative_rows"],
        }
    ).fee_schedule()
    rows = COSTS["illustrative_rows"]
    cols = COSTS["illustrative_columns"]
    bad = []
    for case in COSTS["illustrative_cases"]:
        c = dict(zip(cols, case))
        venue, segment, underlying = _rate_row(rows[c["row"]])
        got = schedule.charge(
            venue, segment, c["side"], float(c["price"]), float(c["quantity"]),
            multiplier=float(c["multiplier"]), underlying=underlying,
            # the reference presets charge the depository fee on every delivery sell fill
            day_fee_owed=(c["side"] == "SELL"),
        )
        for k in ("brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst", "dp_charge", "total"):
            if abs(got[k] - c[k]) > PAISA:
                bad.append((case, k, got[k], c[k]))
    assert len(COSTS["illustrative_cases"]) > 1000
    assert not bad, f"{len(bad)} of {len(COSTS['illustrative_cases'])} differ, first: {bad[:3]}"


# --- margin ---------------------------------------------------------------

DAY = 86_400 * 10**9


def _spec(d):
    if d["kind"] == "equity":
        return r.InstrumentSpec.equity(d["symbol"])
    if d["kind"] == "future":
        return r.InstrumentSpec.futures_contract(
            d["symbol"], d["expiry"], d["lot"], multiplier=d["mult"]
        )
    return r.InstrumentSpec.option(
        d["symbol"], d["strike"], d["right"], d["expiry"], d["lot"],
        multiplier=d["mult"], underlying="NSE:NIFTY50",
    )


def _model(d):
    kind = d["kind"]
    M = r.MarginModel
    if kind == "no_margin":
        return M.no_margin()
    if kind == "cash_only":
        return M.cash_only()
    if kind == "leverage":
        return M.leverage(d["factors"])
    if kind == "percent_of_notional":
        return M.percent_of_notional(d["pct"])
    if kind == "option_simple":
        return M.option_simple(d["seller_pct"])
    if kind == "table":
        return M.table(d["per_unit"], _model(d["default"]))
    if kind == "by_type":
        return M.by_type({k: _model(v) for k, v in d["models"].items()})
    if kind == "defined_risk":
        return M.defined_risk()
    if kind == "spread_credit":
        return M.spread_credit(d["seller_pct"], d["spread_pct"])
    raise AssertionError(kind)


SPECS = {k: _spec(v) for k, v in MARGIN["specs"].items()}
MODELS = {k: _model(v) for k, v in MARGIN["models"].items()}


def _close(got, want):
    if want == "inf":
        return got == float("inf")
    return abs(got - want) <= 1e-9 * max(1.0, abs(want))


def _outcome(call):
    try:
        return call()
    except ValueError:
        return "refused"


def test_single_line_margin_matches():
    cols = MARGIN["line_columns"]
    bad = []
    for case in MARGIN["line_cases"]:
        c = dict(zip(cols, case))
        got = _outcome(lambda: MODELS[c["model"]].required(SPECS[c["spec"]], c["quantity"], c["price"]))
        want = c["required"]
        if want == "refused" or got == "refused":
            ok = want == got
        else:
            ok = _close(got, want)
        if not ok:
            bad.append((case, got))
    assert len(MARGIN["line_cases"]) > 4000
    assert not bad, f"{len(bad)} differ, first: {bad[:3]}"


def test_by_product_split_matches():
    model = r.MarginModel.by_product(
        r.MarginModel.percent_of_notional(20), r.MarginModel.cash_only()
    )
    cols = MARGIN["split_columns"]
    for case in MARGIN["split_cases"]:
        c = dict(zip(cols, case))
        got = _outcome(
            lambda: model.required_split(SPECS[c["spec"]], c["quantity"], c["price"], c["intraday_quantity"])
        )
        if c["required"] == "refused":
            assert got == "refused", case
        else:
            assert _close(got, c["required"]), (case, got)


def test_multi_leg_margin_matches():
    cols = MARGIN["leg_columns"]
    structures = MARGIN["structures"]
    bad = []
    for case in MARGIN["leg_cases"]:
        c = dict(zip(cols, case))
        legs = [(SPECS[k], q, p) for k, q, p in structures[c["structure"]]]
        model = MODELS[c["model"]]
        got = _outcome(
            lambda: model.required_legs(legs, c["now"]) if c["now"] is not None else model.required_legs(legs)
        )
        want = c["required"]
        ok = (want == got) if "refused" in (want, got) else _close(got, want)
        if not ok:
            bad.append((case, got))
    assert len(MARGIN["leg_cases"]) > 100
    assert any(c[3] == "refused" for c in MARGIN["leg_cases"])
    assert not bad, f"{len(bad)} differ, first: {bad[:3]}"
