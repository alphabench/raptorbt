"""Fees priced by the date of the fill, the same-day equity round trip, and daily annualisation.

Each test states what a user's backtest should do; the figures are the presets' own rows in rupees
per crore (1e-7 of the traded value), worked by hand.
"""

import datetime as dt

import numpy as np
import pytest
from _market_harness import Script, bars, ist

import raptorbt as rb
from raptorbt import BacktestConfig, run_strategies
from raptorbt.strategy import Holding, Requirements
from raptorbt.strategy import orders as O

PRE = rb.FeeSchedule.exchange_presets()
CR = 1e-7


class Overnight(Script):
    requirements = Requirements(holding=Holding.overnight())


def _nse(symbol="X"):
    return rb.InstrumentSpec.equity(symbol, venue="NSE", session=rb.SessionSchedule.nse_equity())


def _round_trip(data, script, cls=Overnight, spec=None, **cfg):
    return run_strategies(
        [cls(script)], data, BacktestConfig(initial_capital=1e6, fee_schedule=PRE, **cfg),
        bar_stamp="close", instruments={"X": spec or _nse()},
    )[0]


BUY_SELL = {
    1: lambda s, c: s.submit_order(O.Market(side="buy", units=100.0)),
    3: lambda s, c: s.submit_order(O.Market(side="sell", units=100.0)),
}


# a same-day equity round trip in a positional run is priced on the intraday row
def test_same_day_equity_round_trip_in_a_positional_run_pays_non_delivery_stt():
    t0 = ist(2026, 11, 2, 9, 15)
    data = {"X": bars([(1000,) * 4, (1000,) * 4, (1005,) * 4, (1010,) * 4, (1010,) * 4], t0)}
    r = _round_trip(data, BUY_SELL)
    (t,) = r.result.trades()
    # bought 1,00,000 and sold 1,01,000 the same day: STT 0.025% of the sale = 25.25, stamp 0.003% = 3.00
    assert t.fee_breakdown["stt"] == pytest.approx(25.25)
    assert t.fee_breakdown["stamp_duty"] == pytest.approx(3.0)
    assert t.fees == pytest.approx(35.7686, abs=0.01)  # booked to the paisa
    assert t.fee_breakdown["total"] == pytest.approx(t.fees)
    assert t.entry_fees + t.exit_fees == pytest.approx(t.fees)
    # the account paid exactly that: nothing of the delivery entry charge stays debited
    equity = r.result.equity_curve()
    gross = 100.0 * (1010.0 - 1000.0)
    assert float(equity[-1]) == pytest.approx(1e6 + gross - t.fees, abs=0.01)


def test_the_same_day_correction_is_the_intraday_run_exactly():
    t0 = ist(2026, 11, 2, 9, 15)
    data = {"X": bars([(1000,) * 4, (1000,) * 4, (1005,) * 4, (1010,) * 4, (1010,) * 4], t0)}
    positional = _round_trip(data, BUY_SELL)
    intraday = _round_trip(data, BUY_SELL, cls=Script)  # no requirement: squared off the same day
    a, b = positional.result.trades()[0], intraday.result.trades()[0]
    assert a.fee_breakdown == b.fee_breakdown
    assert (a.fees, a.pnl) == (b.fees, b.pnl)
    assert float(positional.result.equity_curve()[-1]) == float(intraday.result.equity_curve()[-1])


def test_a_position_carried_overnight_stays_on_the_delivery_rows():
    ts = [ist(2026, 11, 2, 9, 16), ist(2026, 11, 2, 9, 17), ist(2026, 11, 3, 9, 16), ist(2026, 11, 3, 9, 17)]
    data = {"X": bars([(1000,) * 4] * 4, ts)}
    script = {0: BUY_SELL[1], 2: BUY_SELL[3]}
    (t,) = _round_trip(data, script).result.trades()
    # delivery: 0.1% on both sides, stamp 0.015% on the buy
    assert t.fee_breakdown["stt"] == pytest.approx(2e5 * 1e4 * CR)
    assert t.fee_breakdown["stamp_duty"] == pytest.approx(1e5 * 1500 * CR)


def test_a_partial_close_the_same_day_is_intraday_and_the_carried_slice_is_delivery():
    ts = [ist(2026, 11, 2, 9, 16), ist(2026, 11, 2, 9, 17), ist(2026, 11, 3, 9, 16), ist(2026, 11, 3, 9, 17)]
    data = {"X": bars([(1000,) * 4] * 4, ts)}
    script = {
        0: lambda s, c: s.submit_order(O.Market(side="buy", units=100.0)),
        1: lambda s, c: s.submit_order(O.Market(side="sell", units=40.0)),
        2: lambda s, c: s.submit_order(O.Market(side="sell", units=60.0)),
    }
    first, second = _round_trip(data, script).result.trades()
    # 40 sold the day it was bought: sale 40,000 at 0.025%, stamp on the 40,000 bought at 0.003%
    assert first.fee_breakdown["stt"] == pytest.approx(40_000 * 2500 * CR)
    assert first.fee_breakdown["stamp_duty"] == pytest.approx(40_000 * 300 * CR)
    # 60 carried: STT 0.1% on the buy and the sale, stamp 0.015% on the buy
    assert second.entry_fees > 0.0
    assert second.fees == pytest.approx(second.entry_fees + second.exit_fees)


# a fill before a row family's first row is refused, naming the earliest date
def _option(expiry=(2023, 6, 29)):
    return rb.InstrumentSpec.option(
        "OPT", strike=18000.0, right="call", expiration_ns=ist(*expiry, 15, 30), lot_size=50.0,
        underlying="NIFTY", price_increment=0.05, venue="NFO",
    )


def _option_run(t0, spec, schedule=PRE):
    data = {"OPT": bars([(200,) * 4, (200,) * 4, (210,) * 4, (220,) * 4], t0)}
    sc = {1: lambda s, c: s.submit_order(O.Market(side="buy", units=50.0)),
          3: lambda s, c: s.submit_order(O.Market(side="sell", units=50.0))}
    return run_strategies([Script(sc)], data, BacktestConfig(initial_capital=1e6, fee_schedule=schedule),
                          bar_stamp="close", instruments={"OPT": spec})[0]


def test_an_option_fill_before_the_first_preset_row_is_refused_naming_the_earliest_date():
    spec = _option((2023, 3, 30))
    with pytest.raises(rb.errors.InvalidParameter) as e:
        _option_run(ist(2023, 3, 1, 9, 15), spec)
    assert e.value.code == "fee_schedule_gap"
    assert "2023-03-0" in str(e.value) and "2023-04-01" in str(e.value)


def test_the_first_preset_day_is_priced_and_its_rows_are_listed_as_unverified():
    # 2023-06: after the first row (2023-04-01), before the true-to-label charges (2024-10-01)
    r = _option_run(ist(2023, 6, 1, 9, 15), _option())
    (t,) = r.result.trades()
    buy, sell = 50 * 200.0, 50 * 220.0
    # STT on the premium sold at 0.0625%; the slab-era exchange charge 4950 per crore; no IPFT
    assert t.fee_breakdown["stt"] == pytest.approx(sell * 6250 * CR)
    assert t.fee_breakdown["exchange_txn"] == pytest.approx((buy + sell) * 4950 * CR)
    assert t.fee_breakdown["ipft"] == 0.0
    assert any("NFO:OPTIONS" in row for row in r.result.unverified_fee_rows)


@pytest.mark.parametrize(
    ("segment", "day", "stt_sell", "exchange"),
    [
        ("futures", (2023, 6, 1), 1250, 188), ("futures", (2024, 8, 1), 1250, 188),
        ("futures", (2024, 11, 1), 2000, 173), ("futures", (2026, 5, 4), 5000, 182.99),
    ],
)
def test_future_stt_follows_its_dated_rates(segment, day, stt_sell, exchange):
    expiry = ist(day[0] + 1, 1, 1, 15, 30)
    spec = rb.InstrumentSpec.futures_contract("F", expiry, 1.0, venue="NFO", underlying="NIFTY")
    data = {"F": bars([(1000,) * 4] * 4, ist(*day, 9, 15))}
    sc = {1: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0)),
          3: lambda s, c: s.submit_order(O.Market(side="sell", units=10.0))}
    r = run_strategies([Script(sc)], data, BacktestConfig(initial_capital=1e6, fee_schedule=PRE),
                       bar_stamp="close", instruments={"F": spec})[0]
    (t,) = r.result.trades()
    assert t.fee_breakdown["stt"] == pytest.approx(10 * 1000.0 * stt_sell * CR)
    assert t.fee_breakdown["exchange_txn"] == pytest.approx(2 * 10 * 1000.0 * exchange * CR)


def test_every_preset_row_family_names_its_earliest_date_and_the_unverified_history_is_flagged():
    rows = [r for r in PRE.rows() if not r["settlement"]]
    firsts = {}
    for row in rows:
        key = (row["venue"], row["segment"], row["underlying"])
        firsts[key] = min(firsts.get(key, row["effective_from"]), row["effective_from"])
    nse = ist(2023, 4, 1, 0, 0)
    assert firsts[("NSE", "DELIVERY", None)] == nse and firsts[("NFO", "OPTIONS", None)] == nse
    assert min(firsts.values()) == nse  # nothing is encoded before 2023-04-01
    slab = [r for r in rows if r["effective_from"] < ist(2024, 10, 1, 0, 0)]
    assert slab and all(r["status"] == "unverified" and "slab-era" in r["source"] for r in slab)


def test_a_rate_card_is_refused_for_a_fill_before_its_date():
    from test_fee_schedule import CARD

    card = rb.RateCard.from_dict(CARD)  # takes effect 2026-09-01
    data = {"X": bars([(1000,) * 4] * 5, ist(2026, 8, 31, 9, 15))}
    spec = rb.InstrumentSpec.equity("X", venue="NSE", session=rb.SessionSchedule.nse_equity())
    with pytest.raises(rb.errors.InvalidParameter) as e:
        run_strategies([Script(BUY_SELL)], data,
                       BacktestConfig(initial_capital=1e6, fee_schedule=card.fee_schedule()),
                       bar_stamp="close", instruments={"X": spec})
    assert e.value.code == "fee_schedule_gap" and "2026-09-01" in str(e.value)
    # the day the card takes effect it is priced
    data = {"X": bars([(1000,) * 4] * 5, ist(2026, 9, 1, 9, 15))}
    run_strategies([Script(BUY_SELL)], data,
                   BacktestConfig(initial_capital=1e6, fee_schedule=card.fee_schedule()),
                   bar_stamp="close", instruments={"X": spec})


def test_a_run_whose_fills_all_fall_after_the_first_row_is_not_refused_for_earlier_bars():
    # warm-up bars before the first row are fine: only a priced fill needs a rate
    ts = [ist(2023, 3, 30, 15, 29), ist(2023, 3, 31, 15, 29), ist(2023, 4, 3, 9, 16), ist(2023, 4, 3, 9, 17),
          ist(2023, 4, 3, 9, 18)]
    data = {"X": bars([(1000,) * 4] * 5, ts)}
    r = _round_trip(data, {2: BUY_SELL[1], 3: BUY_SELL[3]})
    assert len(r.result.trades()) == 1


# daily bars of a session market count trading days
def _weekday_closes(n=60):
    days, d = [], dt.date(2026, 6, 1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return [ist(x.year, x.month, x.day, 15, 30) for x in days]


def _implied_periods(ts, spec, **cfg):
    px = np.round(100 * np.cumprod(1 + np.random.default_rng(7).normal(0.001, 0.01, len(ts))), 2)
    data = {"X": bars([(p,) * 4 for p in px], ts)}
    r = run_strategies([Script({0: lambda s, c: s.submit_order(O.Market(side="buy", units=50.0))})], data,
                       BacktestConfig(initial_capital=1e4, fees=0.0, **cfg), bar_stamp="close",
                       instruments={"X": spec})[0]
    ret = np.asarray(r.result.returns(), float)
    ret = ret[np.isfinite(ret)]
    return (r.result.metrics.sharpe_ratio / (ret.mean() / ret.std(ddof=1))) ** 2


def test_daily_bars_of_a_session_market_annualise_on_trading_days():
    ts = _weekday_closes()
    assert _implied_periods(ts, rb.InstrumentSpec.equity("X")) == pytest.approx(252.0, rel=0.01)
    assert _implied_periods(ts, rb.InstrumentSpec.equity("X"), session_minutes=375.0) == pytest.approx(252.0, rel=0.01)


def test_daily_bars_count_calendar_days_only_when_the_market_trades_every_day():
    weekday = _weekday_closes()
    # an explicit session_minutes=0 names a market that trades round the clock
    assert _implied_periods(weekday, rb.InstrumentSpec.equity("X"), session_minutes=0.0) == pytest.approx(365.0, rel=0.01)
    # a seven-day series (weekend bars present) is a seven-day market
    every_day = [ist(2026, 6, 1, 15, 30) + i * 86_400 * 10**9 for i in range(60)]
    assert _implied_periods(every_day, rb.InstrumentSpec.equity("X")) == pytest.approx(365.0, rel=0.01)
