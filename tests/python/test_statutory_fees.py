"""Fees by instrument, from the dated statutory schedule, with brokerage from a card.

Pins the fee rules with hand-computed numbers:

* a future is charged the future row and an option the option row, not one
  ``fee_segment`` for the run (``fe1``); ``fee_segment`` stays an explicit
  override;
* the built-in presets are the dated statutory schedule only, stored in rupees per
  crore with every component separate, equal to ``fixtures/statutory_fees.json`` on
  every date (``fees``); rows not matched to an official circular are ``unverified``
  and a result lists them;
  with the STT change of 2026-04-01 built in (``fe2``), no brokerage and no
  depository fee;
* the STT on an option exercised at expiry is charged by the kernel from the
  settlement row;
* brokerage comes from a rate card, per order.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, FeeSchedule, InstrumentSpec, RateCard
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import orders as O

from _market_harness import IST, MIN, bars, ist, run_paths

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "statutory_fees.json").read_text())
PRESETS = FeeSchedule.exchange_presets()
GST = 0.18


# -- fe1: one row per instrument ---------------------------------------------------------------

T0 = ist(2026, 11, 2, 9, 15)
EXPIRY = ist(2026, 11, 24, 15, 30)
OPTION = InstrumentSpec.option(
    "NIFTY24000CE", strike=24000.0, right="call", expiration_ns=EXPIRY, lot_size=75.0,
    underlying="NIFTY", price_increment=0.05, venue="NFO",
)
FUTURE = InstrumentSpec.futures_contract(
    "NIFTYFUT", EXPIRY, lot_size=75.0, price_increment=0.1, underlying="NIFTY", venue="NFO",
)
INSTRUMENTS = {"NIFTY24000CE": OPTION, "NIFTYFUT": FUTURE}
OPTION_ROWS = [(200,) * 4, (200,) * 4, (210,) * 4, (220,) * 4, (220,) * 4]
FUTURE_ROWS = [(24000,) * 4, (24000,) * 4, (24100,) * 4, (24200,) * 4, (24200,) * 4]
DATA = {"NIFTY24000CE": bars(OPTION_ROWS, T0), "NIFTYFUT": bars(FUTURE_ROWS, T0)}
ROUND_TRIPS = {
    ("NIFTY24000CE", 1): lambda s, c: s.submit_order(O.Market(side="buy", quantity=75.0)),
    ("NIFTY24000CE", 3): lambda s, c: s.submit_order(O.Market(side="sell", quantity=75.0)),
    ("NIFTYFUT", 1): lambda s, c: s.submit_order(O.Market(side="buy", quantity=75.0)),
    ("NIFTYFUT", 3): lambda s, c: s.submit_order(O.Market(side="sell", quantity=75.0)),
}


def _fees(**fee_args):
    runs = run_paths(
        ROUND_TRIPS, DATA, lambda: BacktestConfig(initial_capital=5_000_000.0, **fee_args),
        instruments=INSTRUMENTS, account_type="margin",
    )
    return {path: res for path, (_, res) in runs.items()}


@pytest.mark.parametrize("fee_args", [{"fee_schedule": PRESETS}, {"fee_segment": "auto"}])
def test_an_option_and_a_future_are_each_priced_on_their_own_row(fee_args):
    """Hand: option 15,000 in / 16,500 out; future 1,800,000 in / 1,815,000 out.

    Per crore, from 2026-03-01: options exchange 3,552 + IPFT 0.01, futures 182.99 + 0.01,
    SEBI 10, stamp 300 / 200, GST 18% of exchange + IPFT + SEBI.
    Option entry: exchange 5.328, IPFT 0.000015, SEBI 0.015, stamp 0.45, GST 0.96 -> 6.76.
    Exit: STT 0.15% = 24.75, exchange 5.8608, SEBI 0.0165, GST 1.06 -> 31.68. Total 38.44.
    Future entry: exchange 32.94, SEBI 1.8, stamp 36, GST 6.25 -> 76.99. Exit: STT 0.05% =
    907.5, exchange 33.21, SEBI 1.815, GST 6.31 -> 948.83. Total 1025.82 (the option row
    would charge 4295.95).
    """
    for path, res in _fees(**fee_args).items():
        trades = {t.symbol: t for t in res.result.trades()}
        option, future = trades["NIFTY24000CE"], trades["NIFTYFUT"]
        assert (option.fees, round(option.pnl, 2)) == (38.44, 1461.56), path
        assert (future.fees, round(future.pnl, 2)) == (1025.82, 13974.18), path
        assert future.fee_breakdown["stt"] == pytest.approx(0.0005 * 24200.0 * 75.0)
        assert future.fee_breakdown["stamp_duty"] == pytest.approx(0.00002 * 24000.0 * 75.0)
        assert future.fee_breakdown["exchange_txn"] == pytest.approx(182.99 / 1e7 * 3_615_000.0)
        assert future.fee_breakdown["ipft"] == pytest.approx(0.01 / 1e7 * 3_615_000.0)
        assert future.fee_breakdown["brokerage"] == 0.0
        assert option.fee_breakdown["stt"] == pytest.approx(0.0015 * 220.0 * 75.0)
        assert res.result.equity_curve()[-1] == pytest.approx(5_015_435.74, abs=1e-6), path


def test_fee_segment_is_an_explicit_override_for_every_instrument():
    """The option row on a future: 3,615,000 x 0.00183% would have been the right exchange charge."""
    trades = {t.symbol: t for t in _fees(fee_segment="NFO-OPT")["batch"].result.trades()}
    assert trades["NIFTYFUT"].fees == 4295.95  # STT 0.15% of 1,815,000 = 2722.5, not 907.5
    assert trades["NIFTYFUT"].fee_breakdown["stt"] == pytest.approx(0.0015 * 24200.0 * 75.0)
    assert trades["NIFTY24000CE"].fees == 38.44


def _refusal(**kw):
    with pytest.raises(InvalidParameter) as e:
        rb.run_strategies(
            [rb.Strategy()], {"X": bars([(1.0,) * 4] * 3, T0)},
            BacktestConfig(fee_schedule=PRESETS), bar_stamp="close", **kw,
        )
    return e.value


def test_an_instrument_the_schedule_cannot_place_is_refused_not_charged_nothing():
    no_spec = _refusal()
    assert (no_spec.code, no_spec.parameter, no_spec.value) == ("fee_schedule_gap", "fee_schedule", "X")
    assert no_spec.symbol == "X", "the instrument is its own attribute; value keeps the symbol for older readers"
    assert no_spec.skill == "execution"
    assert "NFO" in no_spec.valid and "MCX" in no_spec.valid  # the venues the schedule prices

    no_venue = _refusal(instruments={"X": InstrumentSpec.equity("X")})
    assert no_venue.code == "fee_schedule_gap" and "has no venue" in str(no_venue)

    stranger = InstrumentSpec.equity("X", venue="LSE")
    unknown_venue = _refusal(instruments={"X": stranger})
    assert unknown_venue.code == "fee_schedule_gap" and "LSE" in str(unknown_venue)


# -- fe2: the dated statutory change -------------------------------------------------------------


def test_the_stt_change_of_2026_04_01_is_built_in():
    """Options sold on 31 March pay 0.1% STT; the same fill on 1 April pays 0.15%."""
    expected = [
        (ist(2026, 3, 31, 15, 29), 15.0),
        (ist(2026, 4, 1, 0, 0), 22.5),
        (ist(2026, 4, 1, 9, 16), 22.5),
    ]
    for at, stt in expected:
        assert PRESETS.charge("NFO", "OPTIONS", "sell", price=200.0, quantity=75, at=at)["stt"] == stt
    # futures went from 0.02% to 0.05% on the same day: 24,000 x 75 = 1,800,000
    sell = lambda at: PRESETS.charge("NFO", "FUTURES", "sell", price=24000.0, quantity=75, at=at)
    assert sell(ist(2026, 3, 31, 15, 29))["stt"] == pytest.approx(360.0)
    assert sell(ist(2026, 4, 1, 9, 16))["stt"] == pytest.approx(900.0)


def test_a_run_across_the_change_prices_each_fill_on_the_row_of_its_day():
    option = InstrumentSpec.option(
        "O", strike=24000.0, right="call", expiration_ns=ist(2026, 4, 30, 15, 30), lot_size=75.0,
        underlying="NIFTY", price_increment=0.05, venue="NFO",
    )
    stamps = [ist(2026, 3, 31, 15, m) for m in (27, 28, 29)] + [ist(2026, 4, 1, 9, m) for m in (16, 17, 18)]
    data = {"O": bars([(200,) * 4] * 6, stamps)}
    script = {
        0: lambda s, c: s.submit_order(O.Market(side="buy", quantity=75.0)),
        1: lambda s, c: s.submit_order(O.Market(side="sell", quantity=75.0)),
        3: lambda s, c: s.submit_order(O.Market(side="buy", quantity=75.0)),
        4: lambda s, c: s.submit_order(O.Market(side="sell", quantity=75.0)),
    }
    for tz in (None, IST, 0):
        runs = run_paths(
            script, data,
            lambda: BacktestConfig(initial_capital=1e6, fee_schedule=PRESETS, session_tz_offset_ns=tz),
            instruments={"O": option}, paths=("batch",), account_type="margin",
        )
        stts = [t.fee_breakdown["stt"] for t in runs["batch"][1].result.trades()]
        assert stts == pytest.approx([15.0, 22.5]), tz  # 0.10% then 0.15% of the 15,000 premium sold


# -- fees: the presets are the published schedule on every date -------------------------------------------

CRORE = 1e7


def _expected(row, side, price, quantity):
    """The parts of one fill from the fixture's per-crore figures (GST is on exchange + IPFT + SEBI)."""
    turnover = price * quantity
    c = row["per_crore"]
    part = lambda cr: cr / CRORE * turnover
    exchange, ipft, sebi = part(c["exchange"]), part(c["ipft"]), part(c["sebi"])
    return {
        "brokerage": 0.0,
        "stt": part(c["tax_sell" if side == "sell" else "tax_buy"]),
        "exchange_txn": exchange,
        "ipft": ipft,
        "sebi_fee": sebi,
        "stamp_duty": part(c["stamp_buy"]) if side == "buy" else 0.0,
        "gst": GST * (exchange + ipft + sebi),
        "dp_charge": 0.0,
    }


def _day(iso):
    y, m, d = (int(x) for x in iso.split("-"))
    return ist(y, m, d, 0, 0)


def _timeline():
    """``(key, [row, ...])`` for the trade rows of the fixture, oldest first."""
    groups = {}
    for row in FIXTURE["rows"]:
        if row["event"] == "TRADE":
            groups.setdefault((row["venue"], row["segment"], row["underlying"]), []).append(row)
    return [(k, sorted(v, key=lambda r: r["effective_from"])) for k, v in groups.items()]


@pytest.mark.parametrize(
    "key, rows",
    [pytest.param(k, rows, id="-".join(str(p) for p in k if p)) for k, rows in _timeline()],
)
def test_every_trade_row_equals_the_published_schedule_on_each_of_its_dates(key, rows):
    venue, segment, underlying = key
    for i, row in enumerate(rows):
        moments = [_day(row["effective_from"]) + 3_600 * 10**9]
        if i + 1 < len(rows):
            moments.append(_day(rows[i + 1]["effective_from"]) - 1)
        for at in moments:
            for side in ("buy", "sell"):
                for price, quantity in ((200.0, 75), (81_234.5, 20)):
                    got = PRESETS.charge(
                        venue, segment, side, price=price, quantity=quantity,
                        underlying=underlying, at=at,
                    )
                    want = _expected(row, side, price, quantity)
                    for part, value in want.items():
                        assert got[part] == pytest.approx(value, rel=1e-12, abs=1e-12), (
                            key, row["effective_from"], side, part,
                        )
        stored = PRESETS.row(venue, segment, underlying, at=_day(row["effective_from"]))
        assert stored["status"] == row["status"] and stored["base"] == row["base"]
        for name, cr in row["per_crore"].items():
            if name != "tax_kind":
                assert stored["per_crore"][name] == pytest.approx(cr, abs=1e-6), (key, name)
        if i + 1 < len(rows):  # the next row takes over at its own midnight, not before
            nxt = _day(rows[i + 1]["effective_from"])
            before = PRESETS.row(venue, segment, underlying, at=nxt - 1)["rates"]
            after = PRESETS.row(venue, segment, underlying, at=nxt)["rates"]
            assert before != after
            assert row["effective_until"] == rows[i + 1]["effective_from"]


def test_nse_exchange_charge_and_ipft_split_on_2026_03_01_keeps_each_component_separate():
    """NSE/FA/73061: cash 297 + 10, futures 173 + 10, options 3,503 + 50 per crore, then 306.99 + 0.01, 182.99 + 0.01, 3,552 + 0.01."""
    before, after = ist(2026, 2, 27, 12, 0), ist(2026, 3, 2, 12, 0)
    expect = {
        ("NSE", "DELIVERY"): ((297, 10), (306.99, 0.01)),
        ("NSE", "INTRADAY"): ((297, 10), (306.99, 0.01)),
        ("NFO", "FUTURES"): ((173, 10), (182.99, 0.01)),
        ("NFO", "OPTIONS"): ((3503, 50), (3552, 0.01)),
    }
    for (venue, segment), (old, new) in expect.items():
        for at, (exchange, ipft) in ((before, old), (after, new)):
            c = PRESETS.row(venue, segment, at=at)["per_crore"]
            assert (c["exchange"], c["ipft"]) == (pytest.approx(exchange), pytest.approx(ipft))
    # GST includes the IPFT: 18% of exchange + IPFT + SEBI on 1 crore of NFO futures bought
    bill = PRESETS.charge("NFO", "FUTURES", "buy", price=100_000.0, quantity=100, at=before)
    assert bill["exchange_txn"] == pytest.approx(173.0) and bill["ipft"] == pytest.approx(10.0)
    assert bill["gst"] == pytest.approx(0.18 * (173.0 + 10.0 + 10.0))
    assert bill["sebi_fee"] == pytest.approx(10.0) and bill["stamp_duty"] == pytest.approx(200.0)


def test_the_exercise_rows_charge_the_buyer_on_intrinsic_value_only():
    settlement = [r for r in FIXTURE["rows"] if r["event"] == "SETTLEMENT"]
    assert {(r["venue"], r["segment"]) for r in settlement} == {
        ("NFO", "OPTIONS"), ("BFO", "OPTIONS"), ("MCX", "OPTIONS"),
    }
    for row in settlement:
        at = _day(row["effective_from"]) + 3_600 * 10**9
        stored = PRESETS.row(row["venue"], row["segment"], at=at, settlement=True)
        assert stored["base"] == "intrinsic_value" == row["base"]
        assert stored["per_crore"]["tax_sell"] == pytest.approx(row["per_crore"]["tax_sell"])
        c = stored["per_crore"]
        assert (c["tax_buy"], c["exchange"], c["ipft"], c["sebi"], c["stamp_buy"]) == (0, 0, 0, 0, 0)
        assert stored["rates"]["gst_rate"] == 0.0
    row = lambda v, at: PRESETS.row(v, "OPTIONS", at=at, settlement=True)["per_crore"]["tax_sell"]
    # equity options 0.125% until 2026-03-31, 0.15% after; commodity options exercised 0.0001%
    assert (row("NFO", ist(2026, 3, 31, 12, 0)), row("NFO", ist(2026, 4, 1, 12, 0))) == (12_500, 15_000)
    assert row("MCX", ist(2026, 6, 1, 12, 0)) == pytest.approx(10.0)


def test_the_fixture_cites_a_source_and_a_status_for_every_row():
    assert len(FIXTURE["rows"]) == len(PRESETS) > 30
    for row in FIXTURE["rows"]:
        assert row["status"] in ("confirmed", "unverified") and row["source"], row
        assert row["base"] in ("turnover", "premium", "intrinsic_value", "delivery_value")
    unverified = {(r["venue"], r["segment"]) for r in FIXTURE["rows"] if r["status"] == "unverified"}
    # what the sources could not confirm: BSE cash by group, every BSE derivatives row, NSE
    # currency, the MCX exchange charges, and the NSE slab-era rows before 2024-10-01
    assert unverified == {
        ("BSE", "DELIVERY"), ("BSE", "INTRADAY"), ("BFO", "FUTURES"), ("BFO", "OPTIONS"),
        ("CDS", "FUTURES"), ("CDS", "OPTIONS"), ("MCX", "FUTURES"), ("MCX", "OPTIONS"),
        ("NSE", "DELIVERY"), ("NSE", "INTRADAY"), ("NFO", "FUTURES"), ("NFO", "OPTIONS"),
    }
    slab = [r for r in FIXTURE["rows"] if r["effective_from"] == "2023-04-01" and r["event"] == "TRADE"]
    assert slab and all(r["status"] == "unverified" and "slab-era" in r["source"] for r in slab)
    confirmed = {(r["venue"], r["segment"]) for r in FIXTURE["rows"] if r["status"] == "confirmed"}
    assert {("NSE", "DELIVERY"), ("NFO", "FUTURES"), ("NFO", "OPTIONS"), ("BCD", "FUTURES")} <= confirmed
    assert PRESETS.source.startswith("statutory and exchange charges only")


def test_the_presets_name_the_dates_they_start_on_and_change_on():
    dates = sorted({r["effective_from"] for r in FIXTURE["rows"]})
    assert dates == ["2023-04-01", "2024-10-01", "2026-03-01", "2026-04-01"]
    # nothing is encoded before 2023-04-01: a fill earlier is refused, not back-dated
    assert min(r["effective_from"] for r in PRESETS.rows()) == _day("2023-04-01")
    # in force in mid-2025: every 2024-10-01 row, and the exercise tax of NFO options (2023-04-01)
    assert PRESETS.in_force_at(_day("2025-06-01")).rows() == [
        r for r in PRESETS.rows()
        if r["effective_from"] == _day("2024-10-01")
        or (r["settlement"] and r["venue"] == "NFO" and r["effective_from"] == _day("2023-04-01"))
    ]


# -- unverified rows are named by every result ---------------------------------------------------------------------


def _one_future(venue, symbol, underlying, scrip_group=None, kind="future"):
    start = ist(2026, 11, 2, 10, 0)
    if kind == "future":
        spec = InstrumentSpec.futures_contract(
            symbol, ist(2027, 1, 1, 0, 0), 1.0, underlying=underlying, venue=venue
        )
    else:
        spec = InstrumentSpec.equity(symbol, venue=venue)
        spec.scrip_group = scrip_group
    script = {
        1: lambda s, c: s.submit_order(O.Market(side="buy", quantity=10.0)),
        3: lambda s, c: s.submit_order(O.Market(side="sell", quantity=10.0)),
    }
    runs = run_paths(
        script, {symbol: bars([(1000,) * 4] * 5, start)},
        lambda: BacktestConfig(initial_capital=1e7, fee_schedule=PRESETS, session_tz_offset_ns=IST),
        instruments={symbol: spec}, account_type="margin", paths=("batch", "node"),
    )
    return {path: res for path, (_, res) in runs.items()}


def test_a_result_lists_the_unverified_fee_rows_it_used_and_confirmed_rows_are_not_listed():
    confirmed = _one_future("NFO", "NIFTYFUT", "NIFTY")
    assert all(list(r.result.unverified_fee_rows) == [] for r in confirmed.values())
    mcx = _one_future("MCX", "GOLD", "GOLD")
    for path, res in mcx.items():
        assert list(res.result.unverified_fee_rows) == ["MCX:FUTURES"], path
        report = res.full_report([int(t) for t in range(len(res.result.equity_curve()))])
        assert report["unverified_fee_rows"] == ["MCX:FUTURES"]
    # the agricultural row is its own row, and says so
    agri = _one_future("MCX", "KAPASX", "KAPAS")
    assert list(agri["batch"].result.unverified_fee_rows) == ["MCX:FUTURES:class:AGRI"]


AGRI_NAMES = ["COTTON", "COTWASOIL", "CARDAMOM", "RBDPMOLEIN", "KAPAS", "CPO", "MENTHAOIL"]


@pytest.mark.parametrize("name", AGRI_NAMES)
def test_every_mcx_agri_name_of_the_session_table_pays_the_agri_fee_row(name):
    """Hand: 10 x Rs 1,000 = Rs 10,000 each side. SEBI Re 1 per crore = 0.001 a side, so 0.002 over
    the round trip, and no commodities transaction tax (agricultural contracts are excluded).
    Exchange 210 per crore = 0.21 a side, stamp duty 200 per crore on the buy = 0.20."""
    for path, res in _one_future("MCX", name + "26DECFUT", name).items():
        (trade,) = res.result.trades()
        fb = trade.fee_breakdown
        assert fb["stt"] == 0.0, (name, path)
        assert fb["sebi_fee"] == pytest.approx(0.002), (name, path)
        assert fb["exchange_txn"] == pytest.approx(0.42), (name, path)
        assert fb["stamp_duty"] == pytest.approx(0.20), (name, path)
        assert list(res.result.unverified_fee_rows) == ["MCX:FUTURES:class:AGRI"], (name, path)


@pytest.mark.parametrize("name", ["GOLD", "CRUDEOIL", "COPPER", "NATURALGAS"])
def test_every_mcx_non_agri_name_pays_ctt_and_sebi_ten(name):
    """Hand: Rs 10,000 each side. CTT 1,000 per crore on the sell = 1.00; SEBI 10 per crore a side = 0.01."""
    for path, res in _one_future("MCX", name + "26DECFUT", name).items():
        (trade,) = res.result.trades()
        assert trade.fee_breakdown["stt"] == pytest.approx(1.0), (name, path)
        assert trade.fee_breakdown["sebi_fee"] == pytest.approx(0.02), (name, path)
        assert list(res.result.unverified_fee_rows) == ["MCX:FUTURES"], (name, path)


def test_an_mcx_name_in_no_class_is_refused_for_fees_even_with_a_session_given():
    spec = InstrumentSpec.futures_contract(
        "NOSUCH26DECFUT", ist(2027, 1, 1, 0, 0), 1.0, underlying="NOSUCH", venue="MCX",
        session=rb.SessionSchedule.mcx(),
    )
    with pytest.raises(InvalidParameter) as e:
        rb.run_strategies(
            [rb.Strategy()], {"NOSUCH26DECFUT": bars([(1000.0,) * 4] * 3, ist(2026, 11, 2, 10, 0))},
            BacktestConfig(fee_schedule=PRESETS), bar_stamp="close",
            instruments={"NOSUCH26DECFUT": spec},
        )
    assert (e.value.code, e.value.parameter) == ("fee_schedule_gap", "fee_schedule")
    assert "NOSUCH" in str(e.value) and "agricultural" in str(e.value)


def test_an_unverified_row_carries_its_figure_and_never_a_defaulted_zero():
    mcx = PRESETS.row("MCX", "FUTURES")
    assert mcx["status"] == "unverified" and mcx["per_crore"]["exchange"] == 210.0  # Rs 2.10 per lakh
    assert PRESETS.row("MCX", "OPTIONS")["per_crore"]["exchange"] == 4180.0
    assert PRESETS.row("CDS", "OPTIONS")["per_crore"]["exchange"] == 3110.0
    assert PRESETS.row("CDS", "FUTURES")["per_crore"]["exchange"] == 35.0
    assert PRESETS.row("BCD", "FUTURES")["status"] == "confirmed"
    assert PRESETS.row("BCD", "FUTURES")["per_crore"]["exchange"] == 45.0
    assert PRESETS.row("BCD", "OPTIONS")["per_crore"]["exchange"] == 100.0
    # BFO: Sensex and Bankex options 3,250; Sensex 50 and stock options 500; futures nil, all unverified
    assert [PRESETS.row("BFO", "OPTIONS", underlying=u)["per_crore"]["exchange"] for u in ("SENSEX", "BANKEX", "SENSEX50", "ANYSTOCK")] == [3250, 3250, 500, 500]


def test_bse_cash_needs_the_scrip_group_and_names_the_groups_it_prices():
    with pytest.raises(InvalidParameter) as e:
        _one_future("BSE", "RELI", None, scrip_group=None, kind="cash")
    assert (e.value.code, e.value.parameter, e.value.value) == ("fee_schedule_gap", "fee_schedule", "RELI")
    assert "scrip group" in str(e.value) and "A, B" in str(e.value)
    with pytest.raises(InvalidParameter, match="not known for scrip group"):
        _one_future("BSE", "RELI", None, scrip_group="X", kind="cash")
    runs = _one_future("BSE", "RELI", None, scrip_group="A", kind="cash")
    # the same-day round trip is repriced on the intraday row; the delivery row priced the entry first
    assert list(runs["batch"].result.unverified_fee_rows) == [
        "BSE:DELIVERY:group:A", "BSE:INTRADAY:group:A",
    ]
    # the figure is the group A/B one, 375 per crore, and no group is assumed
    assert PRESETS.row("BSE", "DELIVERY", underlying="group:A")["per_crore"]["exchange"] == 375.0


# -- exercise: STT on an option exercised at expiry --------------------------------------------------------


def _held_to_expiry(day, side="buy"):
    start = ist(*day, 9, 15)
    option = InstrumentSpec.option(
        "C", strike=24000.0, right="call", expiration_ns=start + 4 * MIN, lot_size=75.0,
        underlying="NIFTY", venue="NFO",
    )
    option.settlement_price = 200.0  # in the money by 200 at expiry
    script = {("C", 1): lambda s, c: s.submit_order(O.Market(side=side, quantity=75.0))}
    runs = run_paths(
        script, {"C": bars([(150,) * 4] * 6, start)},
        lambda: BacktestConfig(initial_capital=1e6, fee_schedule=PRESETS, session_tz_offset_ns=IST),
        instruments={"C": option}, paths=("batch",), account_type="margin",
    )
    (trade,) = runs["batch"][1].result.trades()
    return trade


@pytest.mark.parametrize("day, rate", [((2026, 4, 6), 0.0015), ((2026, 3, 30), 0.00125)])
def test_the_kernel_charges_stt_on_the_exercised_option_at_expiry(day, rate):
    """Long a 24,000 call bought at 150, 200 in the money: STT is rate x 200 x 75 on the exercise."""
    trade = _held_to_expiry(day)
    assert trade.exit_reason == "Settlement"
    assert trade.fee_breakdown["stt"] == pytest.approx(rate * 200.0 * 75.0)  # 22.5 or 18.75
    # entry: exchange 0.03553% of 11,250 + SEBI + stamp + GST = 5.07, nothing else at expiry
    assert trade.entry_fees == pytest.approx(5.07)
    assert trade.exit_fees == pytest.approx(rate * 200.0 * 75.0)
    assert round(trade.pnl, 2) == round((200.0 - 150.0) * 75.0 - trade.fees, 2)


def test_the_seller_of_an_exercised_option_pays_no_stt_at_expiry():
    trade = _held_to_expiry((2026, 4, 6), side="sell")
    # the sale that opened it paid 0.15% of 11,250; being assigned costs nothing more
    assert trade.exit_reason == "Settlement" and trade.exit_fees == 0.0
    assert trade.fee_breakdown["stt"] == pytest.approx(0.0015 * 150.0 * 75.0)


# -- brokerage: from a rate card, per order -----------------------------------------------------------------

BROKER_CARD = {
    "name": "example-card-2026-11",
    "effective_from": "2026-10-31T00:00:00+05:30",
    "source": "an example card for tests",
    "rates": [
        {
            "exchange": "NFO", "segment": "OPTIONS", "underlying": None,
            "brokerage": {"per_order": "20"},
            "transaction_tax": {"kind": "STT", "percent": "0.15", "sides": ["SELL"]},
            "exchange_turnover_percent": "0.03553", "sebi_percent": "0.0001",
            "stamp_duty_percent": "0.003", "gst_percent": "18",
            "rounding": "none", "brokerage_per": "order",
        },
        {
            "exchange": "NFO", "segment": "FUTURES", "underlying": None,
            "brokerage": {"percent": "0.03", "cap": "20"},
            "transaction_tax": {"kind": "STT", "percent": "0.05", "sides": ["SELL"]},
            "exchange_turnover_percent": "0.00183", "sebi_percent": "0.0001",
            "stamp_duty_percent": "0.002", "gst_percent": "18",
            "rounding": "none", "brokerage_per": "order",
        },
    ],
}


def test_brokerage_comes_from_the_card_per_order_and_the_presets_have_none():
    card = RateCard.from_dict(BROKER_CARD).fee_schedule()
    with_card = {t.symbol: t for t in _fees(fee_schedule=card)["batch"].result.trades()}
    bare = {t.symbol: t for t in _fees(fee_schedule=PRESETS)["batch"].result.trades()}
    for symbol in ("NIFTY24000CE", "NIFTYFUT"):
        # two orders at 20 (the futures' 0.03% is over the 20 cap), plus 18% GST on them
        assert with_card[symbol].fee_breakdown["brokerage"] == 40.0
        # 40 of brokerage and the 18% GST on it, on top of the preset's statutory charges
        # (the card's own statutory parts differ from the presets' by a cent or two)
        assert with_card[symbol].fees == pytest.approx(bare[symbol].fees + 47.2, abs=0.02)
        assert bare[symbol].fee_breakdown["brokerage"] == 0.0
    assert with_card["NIFTY24000CE"].fees == 85.65 and with_card["NIFTYFUT"].fees == 1073.02


# -- physical delivery is priced from the dated cash rows, or refused ----------------------------------------------


def _delivered_future(venue, scrip_group=None, fee_args=None):
    start = ist(2026, 11, 2, 10, 0)
    spec = InstrumentSpec.futures_contract(
        "STKFUT", start + 4 * MIN, 1.0, underlying="STK", venue=venue
    )
    spec.settlement_style = "physical"
    spec.settlement_price = 100.0
    spec.scrip_group = scrip_group
    under = InstrumentSpec.equity("STK", venue="NSE" if venue == "NFO" else "BSE")
    under.scrip_group = scrip_group
    script = {("STKFUT", 1): lambda s, c: s.submit_order(O.Market(side="buy", quantity=10.0))}
    data = {"STKFUT": bars([(100,) * 4] * 6, start), "STK": bars([(100,) * 4] * 6, start)}
    return run_paths(
        script, data,
        lambda: BacktestConfig(initial_capital=1e7, session_tz_offset_ns=IST, **(fee_args or {"fee_schedule": PRESETS})),
        instruments={"STKFUT": spec, "STK": under}, account_type="margin", paths=("batch",),
    )["batch"][1]


def test_a_delivered_nse_stock_future_is_charged_the_nse_cash_delivery_rows():
    res = _delivered_future("NFO")
    stock = [t for t in res.result.trades() if t.symbol == "STK"]
    assert stock, "the delivered stock is booked on the underlying"
    value = 10 * 100.0
    # delivery STT 0.1% and stamp duty 0.015% on the buyer (1,500 per crore), not the future's 0.05%/0.002%
    assert stock[0].fee_breakdown["stt"] == pytest.approx(value * 0.001)
    assert stock[0].fee_breakdown["stamp_duty"] == pytest.approx(value * 1500 / 1e7)


def test_a_delivered_bse_stock_future_without_a_scrip_group_is_refused_not_charged_as_a_future():
    with pytest.raises(InvalidParameter) as e:
        _delivered_future("BFO")
    err = e.value
    assert (err.code, err.parameter, err.value) == ("fee_schedule_gap", "fee_schedule", "STKFUT")
    assert "scrip group" in str(err) and list(err.valid) == ["A", "B"]
    with pytest.raises(InvalidParameter, match="scrip group"):
        _delivered_future("BFO", scrip_group="Z")
    res = _delivered_future("BFO", scrip_group="A")
    (stock,) = [t for t in res.result.trades() if t.symbol == "STK"]
    assert stock.fee_breakdown["stt"] == pytest.approx(1.0)  # delivery STT 0.1% of 1,000, not the future's
    assert list(res.result.unverified_fee_rows) == ["BFO:FUTURES", "BSE:DELIVERY:group:A"]
