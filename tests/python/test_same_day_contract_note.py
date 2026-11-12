"""A same-day CNC round trip costs, to the paisa, what the contract note charges for it.

The card is Zerodha's NSE cash rows (November 2026), on contract-note rounding: stamp duty in whole
rupees, each order's total the rounded sum of its unrounded parts. Figures are worked by hand.
"""

import pytest
from _market_harness import Script, bars, ist

import raptorbt as rb
from raptorbt import BacktestConfig, run_strategies
from raptorbt.strategy import Holding, Requirements
from raptorbt.strategy import orders as O

_COMMON = {"exchange": "NSE", "underlying": None, "exchange_turnover_percent": "0.00307", "sebi_percent": "0.0001", "gst_percent": "18"}
CARD = {
    "name": "zerodha-nse-cash-2026-11",
    "effective_from": "2026-10-31T00:00:00+05:30",
    "source": "https://zerodha.com/charges (read 2026-11-01)",
    "rates": [
        {
            **_COMMON,
            "segment": "DELIVERY",
            "brokerage": {"per_order": "0"},
            "transaction_tax": {"kind": "STT", "percent": "0.1", "sides": ["BUY", "SELL"]},
            "stamp_duty_percent": "0.015",
            "dp_charge": {"per_scrip_day": "13"},
        },
        {
            **_COMMON,
            "segment": "INTRADAY",
            "brokerage": {"percent": "0.03", "cap": "20"},
            "transaction_tax": {"kind": "STT", "percent": "0.025", "sides": ["SELL"]},
            "stamp_duty_percent": "0.003",
        },
    ],
}
SCHEDULE = rb.RateCard.from_dict(CARD).fee_schedule()


class Overnight(Script):
    requirements = Requirements(holding=Holding.overnight())


def _run(rows, stamps, script):
    spec = rb.InstrumentSpec.equity("X", venue="NSE", session=rb.SessionSchedule.nse_equity())
    return run_strategies(
        [Overnight(script)], {"X": bars(rows, stamps)},
        BacktestConfig(initial_capital=1e6, fee_schedule=SCHEDULE),
        bar_stamp="close", instruments={"X": spec},
    )[0].result.trades()


def _paise(x):
    return round(x, 2)


@pytest.mark.parametrize(
    "buy,sell,qty,note_buy,note_sell",
    [
        # buy 2 @284.65: exchange 0.0174775, SEBI 0.0005693, stamp 0.017079 (0), GST 0.0032484 = 0.02;
        # sell 2 @285.10: STT 0.14255, exchange 0.0175051, SEBI 0.0005702, GST 0.0032536 = 0.16
        ("284.65", "285.10", 2, 0.02, 0.16),
        # buy 2 @1500: 0.112218 = 0.11; sell 2 @1510: 0.86798612 = 0.87
        ("1500", "1510", 2, 0.11, 0.87),
    ],
)
def test_a_same_day_round_trip_costs_what_the_contract_note_charges(buy, sell, qty, note_buy, note_sell):
    t0 = ist(2026, 11, 16, 9, 15)
    b, s = float(buy), float(sell)
    rows = [(b,) * 4, (b,) * 4, (b,) * 4, (s,) * 4, (s,) * 4]
    script = {
        1: lambda st, c: st.submit_order(O.Market(side="buy", units=float(qty))),
        3: lambda st, c: st.submit_order(O.Market(side="sell", units=float(qty))),
    }
    (t,) = _run(rows, t0, script)
    assert (_paise(t.entry_fees), _paise(t.exit_fees)) == (note_buy, note_sell), t.fee_breakdown


def test_a_partly_squared_entry_is_charged_as_one_order_rounded_once():
    # buy 2 @284.65, sell 1 @285.10 the same day and the other 1 the next day. The buy is one order,
    # 1 intraday and 1 delivery: STT 0.28465, exchange 0.0174775, SEBI 0.0005693, stamp 0.051237 (0),
    # GST 0.0032484 = 0.3059452, 0.31 (each part rounded alone: 0.01 + 0.30, or 0.29 when the
    # delivery part keeps half of the 0.59 booked for both). The same-day sell of 1 is all intraday:
    # STT 0.071275, exchange 0.00875257, SEBI 0.0002851, GST 0.00162678 = 0.08193945, 0.08.
    ts = [ist(2026, 11, 16, 9, 16), ist(2026, 11, 16, 9, 17), ist(2026, 11, 17, 9, 16), ist(2026, 11, 17, 9, 17)]
    rows = [(284.65,) * 4, (285.10,) * 4, (286.0,) * 4, (286.0,) * 4]
    script = {
        0: lambda st, c: st.submit_order(O.Market(side="buy", units=2.0)),
        1: lambda st, c: st.submit_order(O.Market(side="sell", units=1.0)),
        2: lambda st, c: st.submit_order(O.Market(side="sell", units=1.0)),
    }
    first, second = _run(rows, ts, script)
    assert _paise(first.exit_fees) == 0.08, first.fee_breakdown
    assert _paise(first.entry_fees + second.entry_fees) == 0.31, (first.fee_breakdown, second.fee_breakdown)


def test_an_entry_squared_in_two_slices_the_same_day_is_charged_as_the_intraday_order():
    # buy 2 @284.65 and sell 1 + 1 the same day: the buy is all intraday, 0.02, as one sale of 2
    ts = [ist(2026, 11, 16, 9, 16), ist(2026, 11, 16, 9, 17), ist(2026, 11, 16, 9, 18), ist(2026, 11, 16, 9, 19)]
    rows = [(284.65,) * 4, (285.10,) * 4, (285.10,) * 4, (285.10,) * 4]
    script = {
        0: lambda st, c: st.submit_order(O.Market(side="buy", units=2.0)),
        1: lambda st, c: st.submit_order(O.Market(side="sell", units=1.0)),
        2: lambda st, c: st.submit_order(O.Market(side="sell", units=1.0)),
    }
    first, second = _run(rows, ts, script)
    assert _paise(first.entry_fees + second.entry_fees) == 0.02, (first.fee_breakdown, second.fee_breakdown)
