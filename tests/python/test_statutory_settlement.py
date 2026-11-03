"""The statutory events of an expiry stay separate.

Exercise STT is charged on the intrinsic value, on the buyer's side, and never on an
option that expires worthless. Physical delivery of stock is an event of its own, charged
at the cash market's delivery rates, not stacked on the derivative's own charges. A
cash-settled contract settles at the underlying's print at the expiry instant (the index's
official closing value is the bar the caller supplies for that instant; the engine takes
no VWAP of it).
"""

import pytest

from raptorbt import FeeSchedule, InstrumentSpec, RateCard

from test_lifecycle import bars as _bars
from test_lifecycle import card, config, holder, holder_cls, run, trades

# These runs sit a few nanoseconds after 2026-02-02, in force of the schedule's rows (a fill before a
# schedule's first row is refused): the exercised-option STT then is 0.125% (it is 0.15% from
# 2026-04-01; see test_statutory_fees.py).
T0 = 1_770_000_000 * 10**9
EXERCISE_STT = 0.00125


def bars(closes):
    return _bars(closes, start=T0)


def priced(segment):
    cfg = config()
    cfg.fee_segment = segment
    cfg.fee_schedule = FeeSchedule.exchange_presets()
    return cfg


def call(strike=100.0):
    opt = InstrumentSpec.option(
        "OPT", strike=strike, right="call", expiration_ns=T0 + 4, lot_size=1.0, underlying="U"
    )
    opt.settlement_price = "underlying"
    return opt


def settle(side, underlying_closes):
    s = holder("OPT", units=10.0, side=side)
    r = run(
        s,
        {"OPT": bars([7, 8]), "U": bars(underlying_closes)},
        instruments={"OPT": call()},
        config=priced("NFO-OPT"),
    )
    (t,) = [t for t in trades(r) if t.symbol == "OPT"]
    return t


def test_an_exercised_call_pays_stt_on_the_intrinsic_value_only():
    """U prints 112 at the expiry: intrinsic 12, STT 0.125% x 12 x 10 = 0.15, no brokerage, no
    exchange charge (the premium paid on entry is a trade, not the exercise)."""
    t = settle("buy", [105, 108, 110, 111, 112, 113])
    assert t.exit_reason == "Settlement" and t.exit_price == 12.0
    assert t.exit_fees == pytest.approx(EXERCISE_STT * 12.0 * 10.0, abs=1e-9)


def test_a_call_that_expires_worthless_owes_no_exercise_stt():
    t = settle("buy", [90, 90, 90, 90, 90, 90])
    assert t.exit_price == 0.0
    assert t.exit_fees == 0.0


def test_the_writer_of_an_exercised_call_owes_no_exercise_stt():
    """The tax falls on the buyer: closing a short at the same settlement costs nothing."""
    t = settle("sell", [105, 108, 110, 111, 112, 113])
    assert t.exit_price == 12.0
    assert t.exit_fees == 0.0


def physical_future():
    fut = InstrumentSpec.futures_contract(
        "FUT", expiration_ns=T0 + 3, lot_size=1.0, multiplier=1.0, underlying="U"
    )
    fut.settlement_style = "physical"
    fut.settlement_price = 120.0
    return fut


def _rate(exchange, segment, tax_percent, sides, stamp="0", event=None):
    row = {
        "exchange": exchange,
        "segment": segment,
        "underlying": None,
        "brokerage": {"per_order": "0"},
        "transaction_tax": {"kind": "STT", "percent": tax_percent, "sides": sides},
        "exchange_turnover_percent": "0",
        "sebi_percent": "0",
        "stamp_duty_percent": stamp,
        "gst_percent": "0",
    }
    if event:
        row["event"] = event
    return row


def _schedule():
    """A derivatives row with a settlement row, and the cash market's delivery row (STT on both
    sides, stamp on the buyer)."""
    return RateCard.from_dict(
        {
            "name": "c",
            "effective_from": "2026-01-01T00:00:00+00:00",
            "source": "test",
            "rates": [
                _rate("NFO", "FUTURES", "0", ["SELL"]),
                _rate("NFO", "FUTURES", "0.1", ["SELL"], event="SETTLEMENT"),
                _rate("NSE", "DELIVERY", "0.1", ["BUY", "SELL"], stamp="0.015"),
            ],
        }
    ).fee_schedule()


def test_a_physically_delivered_stock_future_is_charged_as_a_delivery_of_the_stock():
    """600 units delivered at 120 (72,000): the future's own settlement carries no charge (the
    schedule's settlement row is for cash settlement), and the stock leg carries the cash
    market's delivery charges: 0.1% STT = 72 and 0.015% stamp on the buyer = 10.80, which a
    contract note rounds to 11 rupees."""
    cfg = config()
    cfg.fee_segment = "NFO-FUT"
    cfg.fee_schedule = _schedule()
    r = run(
        holder("FUT", units=600.0),
        {"FUT": bars([100, 101]), "U": bars([100, 100, 100, 100, 100])},
        instruments={"FUT": physical_future(), "U": InstrumentSpec.equity("U")},
        config=cfg,
    )
    (fut,) = [t for t in trades(r) if t.symbol == "FUT"]
    assert fut.exit_reason == "Settlement" and fut.exit_fees == 0.0, "the derivative's STT is not stacked"
    (stock,) = [t for t in trades(r) if t.symbol == "U"]
    assert stock.entry_price == 120.0 and stock.size == 600.0
    assert stock.entry_fees == pytest.approx(72.0 + 11.0, abs=1e-9)


def test_a_cash_settled_future_still_pays_its_settlement_row():
    """The physical rule does not remove the row from a cash-settled contract."""
    sched = FeeSchedule.from_cards([card("2026-01-01T00:00:00+00:00", "0", settle="0.1")])
    cfg = config()
    cfg.fee_segment = "NFO-FUT"
    cfg.fee_schedule = sched
    fut = InstrumentSpec.futures_contract("FUT", expiration_ns=T0 + 2, lot_size=1.0, multiplier=1.0)
    r = run(holder("FUT", units=6.0), {"FUT": bars([100, 101, 102, 103])}, instruments={"FUT": fut}, config=cfg)
    (t,) = trades(r)
    assert t.exit_fees == pytest.approx(0.001 * 102.0 * 6.0, abs=0.01)
