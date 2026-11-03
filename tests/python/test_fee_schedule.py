"""Fees as data: rate cards, fee schedules, and running on one.

Pins that the built-in presets are the numbers the engine always charged, that
a loaded schedule changes a run exactly as its rates say, that the itemised
breakdown is the money taken (one computation), and that a schedule never
silently charges nothing.
"""

import copy
import json

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, FeeSchedule, RateCard
from raptorbt.errors import InvalidParameter

DAY_NS = 86_400_000_000_000
# 2026-09-02 00:00 IST: after the test card (2026-09-01) and every preset row
T0 = 1_788_287_400 * 10**9

CARD = {
    "name": "test-card-2026-09",
    "effective_from": "2026-09-01T00:00:00+05:30",
    "source": "https://example.org/charges (read 2026-09-01)",
    "rates": [
        {
            "exchange": "NSE",
            "segment": "INTRADAY",
            "underlying": None,
            "brokerage": {"percent": "0.03", "cap": "20"},
            "transaction_tax": {"kind": "STT", "percent": "0.025", "sides": ["SELL"]},
            "exchange_turnover_percent": "0.00307",
            "sebi_percent": "0.0001",
            "stamp_duty_percent": "0.003",
            "gst_percent": "18",
        },
        {
            "exchange": "NSE",
            "segment": "DELIVERY",
            "underlying": None,
            "brokerage": {"per_order": "0"},
            "transaction_tax": {"kind": "STT", "percent": "0.1", "sides": ["BUY", "SELL"]},
            "exchange_turnover_percent": "0.00307",
            "sebi_percent": "0.0001",
            "stamp_duty_percent": "0.015",
            "gst_percent": "18",
            "dp_charge": {"per_scrip_day": "13"},
        },
    ],
}


@pytest.fixture
def daily():
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, size=500)))
    n = len(close)
    openp = np.concatenate([[close[0]], close[:-1]])
    return {
        "timestamps": T0 + np.arange(n, dtype=np.int64) * DAY_NS,
        "open": openp,
        "high": np.maximum(openp, close) * 1.004,
        "low": np.minimum(openp, close) * 0.996,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }


def _signals(close, fast=10, slow=30):
    n = len(close)

    def sma(w):
        out = np.full(n, np.nan)
        c = np.cumsum(np.insert(close, 0, 0.0))
        out[w - 1 :] = (c[w:] - c[:-w]) / w
        return out

    f, s = sma(fast), sma(slow)
    valid = ~(np.isnan(f) | np.isnan(s))
    above = np.zeros(n, dtype=bool)
    above[valid] = f[valid] > s[valid]
    prev = np.roll(above, 1)
    prev[0] = False
    return above & ~prev & valid, ~above & prev & valid


def _run(d, **cfg):
    entries, exits = _signals(d["close"])
    return raptorbt.run_single_backtest(
        d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"],
        entries, exits, symbol="TEST", config=BacktestConfig(initial_capital=1_000_000.0, **cfg),
    )


def test_presets_are_the_statutory_rates_in_force_now():
    s = FeeSchedule.exchange_presets()
    assert {(r["venue"], r["segment"]) for r in s.rows()} == {
        (v, g) for v in ("NSE", "BSE") for g in ("INTRADAY", "DELIVERY")
    } | {(v, g) for v in ("NFO", "BFO", "MCX", "CDS", "BCD") for g in ("FUTURES", "OPTIONS")}
    for name, (venue, segment) in {
        "equity_delivery": ("NSE", "DELIVERY"),
        "equity_intraday": ("NSE", "INTRADAY"),
        "futures_nfo": ("NFO", "FUTURES"),
        "options_nfo": ("NFO", "OPTIONS"),
        "futures_mcx": ("MCX", "FUTURES"),
        "options_mcx": ("MCX", "OPTIONS"),
        "futures_cds": ("CDS", "FUTURES"),
        "options_cds": ("CDS", "OPTIONS"),
    }.items():
        old = raptorbt.indian_cost_schedule(name)
        row = s.row(venue, segment)["rates"]
        assert row["exchange_rate"] == old["exchange_txn_rate"], name
        assert row["regulatory_rate"] == old["sebi_turnover_rate"], name
        assert row["stamp_buy_rate"] == old["stamp_duty_rate"], name
        assert row["gst_rate"] == old["gst_rate"], name
        assert max(row["tax_buy_rate"], row["tax_sell_rate"]) == old["stt_rate"], name
    # BSE and BFO have the exchange charges of their own, not the NSE family's
    assert s.row("BSE", "INTRADAY", underlying="group:A")["rates"]["exchange_rate"] == 3.75e-5
    assert s.row("BFO", "OPTIONS", underlying="SENSEX")["rates"]["exchange_rate"] == 3.25e-4
    assert s.row("BFO", "OPTIONS")["rates"]["exchange_rate"] == 5e-5  # stock options, 500 per crore
    assert s.row("BFO", "FUTURES")["rates"]["exchange_rate"] == 0.0
    # and none of them charges brokerage or a depository fee: those are a broker's
    for row in s.rows():
        r_ = row["rates"]
        assert (r_["brokerage_flat"], r_["brokerage_rate"], r_["day_fee"]) == (0.0, 0.0, None)


def test_an_explicit_preset_schedule_leaves_a_run_bit_identical(daily):
    plain = _run(daily, fees=0.001, fee_segment="NSE-INTRADAY")
    via = _run(
        daily, fees=0.001, fee_segment="NSE-INTRADAY", fee_schedule=FeeSchedule.exchange_presets()
    )
    assert list(plain.equity_curve()) == list(via.equity_curve())
    assert [t.fee_breakdown for t in plain.trades()] == [t.fee_breakdown for t in via.trades()]


def test_a_loaded_schedule_prices_the_run_and_the_breakdown_is_the_money_taken(daily):
    doc = copy.deepcopy(CARD)
    doc["rates"][0]["brokerage"] = {"per_order": "55"}
    schedule = RateCard.from_dict(doc).fee_schedule()
    base = _run(daily, fee_segment="NSE-INTRADAY")
    custom = _run(daily, fee_segment="NSE-INTRADAY", fee_schedule=schedule)
    assert custom.metrics.total_fees_paid > base.metrics.total_fees_paid
    for t in custom.trades():
        b = t.fee_breakdown
        assert b["brokerage"] == 110.0  # 55 each side
        assert b["total"] == pytest.approx(t.fees, abs=1e-9)
    itemised = sum(t.fee_breakdown["total"] for t in custom.trades())
    assert itemised == pytest.approx(custom.metrics.total_fees_paid, abs=1e-6)


def test_a_contract_note_row_reports_the_total_it_charges(daily):
    schedule = RateCard.from_dict(CARD).fee_schedule()
    r = _run(daily, fee_segment="NSE-INTRADAY", fee_schedule=schedule)
    for t in r.trades():
        b = t.fee_breakdown
        # every shown part is to the paisa, and the total is what was charged
        assert b["total"] == pytest.approx(t.fees, abs=1e-9)
        assert round(b["brokerage"], 2) == pytest.approx(b["brokerage"], abs=1e-9)


def test_a_schedule_without_the_row_is_refused_not_free():
    schedule = RateCard.from_dict(CARD).fee_schedule()
    with pytest.raises(ValueError, match="no rate for MCX FUTURES") as e:
        BacktestConfig(fee_segment="MCX-FUT", fee_schedule=schedule)
    assert e.value.code == "fee_schedule_gap"
    # a schedule alone prices each instrument on its own row (tests/python/test_statutory_fees.py)
    assert BacktestConfig(fee_schedule=schedule).fee_segment is None
    cfg = BacktestConfig(fee_segment="NSE-INTRADAY", fee_schedule=schedule)
    with pytest.raises(ValueError):
        cfg.fee_segment = "MCX-FUT"
    assert cfg.fee_segment == "NSE-INTRADAY"
    cfg.fee_segment = "NSE-DELIVERY"
    assert cfg.fee_segment == "NSE-DELIVERY"


def test_charge_returns_the_itemised_parts():
    s = RateCard.from_dict(CARD).fee_schedule()
    sell = s.charge("NSE", "INTRADAY", "SELL", price=1500.55, quantity=10)
    assert sell["total"] == sell["exact_total"] == 9.62
    assert sell["stt"] == 3.75 and sell["stamp_duty"] == 0.0
    buy = s.charge("NSE", "INTRADAY", "buy", price=1500.55, quantity=10)
    assert buy["stt"] == 0.0 and buy["stamp_duty"] == 0.0  # 0.003% of 15,005.50 rounds to 0 rupees


def test_the_per_day_fee_is_charged_only_when_the_caller_says_it_is_owed():
    s = RateCard.from_dict(CARD).fee_schedule()
    plain = s.charge("NSE", "DELIVERY", "SELL", price=100.0, quantity=3)
    owed = s.charge("NSE", "DELIVERY", "SELL", price=100.0, quantity=3, day_fee_owed=True)
    assert plain["dp_charge"] == 0.0
    assert owed["dp_charge"] == 13.0
    assert owed["gst"] == pytest.approx(plain["gst"] + 2.34)
    buy = s.charge("NSE", "DELIVERY", "BUY", price=100.0, quantity=3, day_fee_owed=True)
    assert buy["dp_charge"] == 0.0


def test_a_brokerage_cap_holds_over_the_fills_of_one_order():
    doc = copy.deepcopy(CARD)
    s = RateCard.from_dict(doc).fee_schedule()
    first = s.charge("NSE", "INTRADAY", "BUY", price=100.0, quantity=500)  # 0.03% of 50,000
    second = s.charge("NSE", "INTRADAY", "BUY", price=100.0, quantity=500, order_turnover=50_000.0)
    assert first["brokerage"] == 15.0
    assert second["brokerage"] == 5.0  # the order's brokerage reaches the 20 cap


def test_a_card_round_trips_through_json_and_dict():
    card = RateCard.from_dict(CARD)
    again = RateCard.from_json(card.to_json())
    assert again.to_dict() == card.to_dict()
    # the presets take effect on several dates, which one card cannot carry
    presets = FeeSchedule.exchange_presets()
    with pytest.raises(InvalidParameter, match="different dates"):
        presets.to_json("2026-09-01T00:00:00+05:30")
    on_day = presets.in_force_at(RateCard.from_dict(CARD).effective_ns)
    back = FeeSchedule.from_json(on_day.to_json("2026-09-01T00:00:00+05:30"))
    for a, b in zip(on_day.rows(), back.rows()):
        assert a["rates"] == b["rates"]


def test_rows_pick_the_underlying_and_the_date():
    doc = copy.deepcopy(CARD)
    gold = copy.deepcopy(doc["rates"][0])
    gold["underlying"] = "GOLD"
    gold["exchange_turnover_percent"] = "0.5"
    doc["rates"].append(gold)
    s = RateCard.from_dict(doc).fee_schedule()
    assert s.row("NSE", "INTRADAY", underlying="X")["rates"]["exchange_rate"] == 3.07e-5
    assert s.row("NSE", "INTRADAY", underlying="GOLD")["rates"]["exchange_rate"] == 0.005
    card = RateCard.from_dict(doc)
    assert card.backdated_fills([card.effective_ns - 1, card.effective_ns]) == 1


def test_merged_cards_keep_every_row_in_order():
    a = RateCard.from_dict(CARD)
    other = copy.deepcopy(CARD)
    other["name"] = "other"
    other["rates"] = [dict(other["rates"][0], exchange="BSE")]
    merged = FeeSchedule.from_cards([a, RateCard.from_dict(other)])
    assert merged.name == "test-card-2026-09+other"
    assert [r["venue"] for r in merged.rows()] == ["NSE", "NSE", "BSE"]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda d: d.pop("name"), "no name"),
        (lambda d: d.pop("effective_from"), "effective_from is required"),
        (lambda d: d.update(effective_from="2026-09-01"), "RFC 3339"),
        (lambda d: d.update(rates=[]), "non-empty list"),
        (lambda d: d["rates"][0].update(segment="SWAP"), "unknown segment"),
        (lambda d: d["rates"][0]["transaction_tax"].update(kind="VAT"), "STT, CTT or NONE"),
        (lambda d: d["rates"][0].update(gst_percent="-1"), "non-negative"),
        (lambda d: d["rates"][0].update(gst_percent="1e2"), "not a decimal string"),
        (lambda d: d["rates"][0].update(rounding="up"), "rounding"),
        (lambda d: d["rates"][1].update(segment="INTRADAY"), "only a DELIVERY"),
    ],
)
def test_an_invalid_card_is_refused_with_a_structured_error(edit, message):
    doc = copy.deepcopy(CARD)
    edit(doc)
    with pytest.raises(InvalidParameter, match=message) as e:
        RateCard.from_dict(doc)
    assert e.value.code == "invalid_fee_schedule"
    with pytest.raises(InvalidParameter):
        RateCard.from_json("not json")


def test_a_custom_gst_base_and_rounding_are_declared_per_row():
    doc = copy.deepcopy(CARD)
    doc["rates"][0].update(rounding="paise_each", gst_base=["brokerage", "transaction_tax"])
    s = RateCard.from_dict(doc).fee_schedule()
    c = s.charge("NSE", "INTRADAY", "SELL", price=1500.55, quantity=10)
    # GST on brokerage and the transaction tax only (0.18 * 8.25 = 1.485, a
    # tie, which goes half up), each part to the paisa
    assert (c["brokerage"], c["stt"], c["gst"]) == (4.5, 3.75, 1.49)
    assert c["total"] == pytest.approx(
        c["brokerage"] + c["stt"] + c["exchange_txn"] + c["sebi_fee"] + c["stamp_duty"] + c["gst"]
    )
    json.dumps(c)
