"""Market structure: dated sessions, session-anchored bars, and instrument-driven defaults.

Pins the session rules with hand-computed times:

* the sessions of NSE, BSE, NFO, BFO, MCX, CDS and BCD, with the equity-derivatives
  close moving to 15:40 on 2026-08-03, per-date overrides (Muhurat, half days)
  and a dated calendar document (``c1``'s anchoring, ``s2``'s zone);
* intraday composite bars start at the session open and end at its close (``c1``);
* ``squareoff_time`` is read in the instrument's session zone, and refused with none (``s2``);
* an order defaults to DAY validity on an instrument with a session (``gap``, ``ml2``), and one the
  engine builds (``enter()``, ``close_position()``, their legs) is gtc only where the segment holds it;
* a strategy is squared off only if its requirements say it holds intraday, or the run says so.
"""

import json
from pathlib import Path

import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, LiveNode, SessionSchedule, Strategy, run_strategies
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import Holding, Requirements, Underlying
from raptorbt.strategy import orders as O

from _market_harness import IST, MIN, Script, bars, fmt, ist, run_paths

FIXTURES = Path(__file__).parent / "fixtures"
Q4 = (FIXTURES / "calendar-2026-q4.json").read_text()
Q4_DST = (FIXTURES / "calendar-2026-q4-dst.json").read_text()


def minutes(hh, mm):
    return hh * 60 + mm


# -- sessions ---------------------------------------------------------------------------------------


def test_the_presets_carry_no_holidays_and_no_square_off():
    for make in (SessionSchedule.nse_equity, SessionSchedule.nse_fo, SessionSchedule.bse_equity,
                 SessionSchedule.bse_fo, SessionSchedule.mcx, SessionSchedule.cds, SessionSchedule.bcd):
        schedule = make()
        assert schedule.squareoff_minute is None and schedule.holidays == [], make.__name__
        assert schedule.tz_offset_ns == IST
    # holidays and square-off are the caller's: 26 January trades until they say otherwise
    assert SessionSchedule.nse_equity().is_open(ist(2026, 1, 26, 10, 0))
    assert SessionSchedule.nse_equity(squareoff_minute=minutes(15, 20)).squareoff_minute == 920


def test_the_equity_derivatives_close_moves_to_1540_on_2026_08_03():
    fo = SessionSchedule.nse_fo()
    assert fo.windows_on(20260731) == [(minutes(9, 15), minutes(15, 30))]
    assert fo.windows_on(20260803) == [(minutes(9, 15), minutes(15, 40))]
    assert fo.is_open(ist(2026, 7, 31, 15, 29)) and not fo.is_open(ist(2026, 7, 31, 15, 30))
    assert fo.is_open(ist(2026, 8, 3, 15, 39)) and not fo.is_open(ist(2026, 8, 3, 15, 40))
    assert fo.session_close_ns(20260803) == ist(2026, 8, 3, 15, 40)
    assert SessionSchedule.bse_fo() == fo
    # cash equities keep 15:30 whatever the date
    cash = SessionSchedule.nse_equity()
    assert not cash.is_open(ist(2026, 8, 3, 15, 31)) and SessionSchedule.bse_equity() == cash


def test_mcx_follows_the_exchanges_dated_evening_closes_and_refuses_earlier_dates():
    mcx = SessionSchedule.mcx()
    day = (540, 1020)
    winter, summer = (1020, minutes(23, 55)), (1020, minutes(23, 30))
    # 2024-11-04 winter, 2025-03-10 summer, 2025-11-03 winter, 2026-03-09 summer, 2026-11-02 winter
    for date, evening in (
        (20241105, winter), (20250611, summer), (20251201, winter), (20260615, summer), (20261102, winter),
    ):
        assert mcx.windows_on(date) == [day, evening], date
    assert mcx.windows_on(20251206) == []  # Saturday
    assert mcx.session_close_ns(20251201) == ist(2025, 12, 1, 23, 55)
    # before the first dated period nothing is guessed
    assert mcx.coverage == [(20241104, 22001231)]
    assert mcx.windows_on(20241101) == [] and mcx.session_date(ist(2024, 11, 1, 12, 0)) is None
    # the internationally referenceable agricultural contracts close at 21:00 all year
    agri = SessionSchedule.mcx_select_agri()
    assert agri.windows_on(20251201) == [day, (1020, minutes(21, 0))]
    assert agri.windows_on(20260615) == [day, (1020, minutes(21, 0))]
    # the other agricultural contracts close at 17:00 all year
    assert SessionSchedule.mcx_other_agri().windows_on(20260615) == [day]
    assert SessionSchedule.mcx_other_agri().windows_on(20251201) == [day]


def test_currency_cross_pairs_trade_until_1930():
    cross = SessionSchedule.cds_cross()
    assert cross.windows_on(20261201) == [(540, minutes(19, 30))]
    assert SessionSchedule.cds().trading_minutes == 480 and cross.trading_minutes == 630


def test_currency_derivatives_trade_nine_to_five():
    for schedule in (SessionSchedule.cds(), SessionSchedule.bcd()):
        assert schedule.windows_on(20261201) == [(540, 1020)]
        assert schedule.is_open(ist(2026, 12, 1, 9, 0)) and not schedule.is_open(ist(2026, 12, 1, 17, 0))


def test_a_date_override_makes_a_special_session_a_half_day_or_a_closure():
    muhurat = (minutes(18, 15), minutes(19, 15))
    s = (
        SessionSchedule.nse_equity(holidays=[20261108])
        .with_session_override(20261108, [muhurat])  # a Sunday, also listed as a holiday
        .with_session_override(20260115, [(minutes(9, 15), minutes(12, 0))])  # a half day
        .with_session_override(20260126, [])  # a closure
    )
    assert s.is_open(ist(2026, 11, 8, 18, 30)) and not s.is_open(ist(2026, 11, 8, 19, 15))
    assert s.session_close_ns(20261108) == ist(2026, 11, 8, 19, 15)
    assert s.is_open(ist(2026, 1, 15, 11, 59)) and not s.is_open(ist(2026, 1, 15, 12, 0))
    assert not s.is_open(ist(2026, 1, 26, 10, 0))
    assert s.overrides == {20260115: [(555, 720)], 20260126: [], 20261108: [muhurat]}
    assert s.is_open(ist(2026, 1, 16, 10, 0)), "the next day is the weekday it always was"


def test_a_dated_window_change_applies_between_its_dates_only():
    wide = {day: [(minutes(9, 0), minutes(16, 0))] for day in ("mon", "tue", "wed", "thu", "fri")}
    s = SessionSchedule.nse_equity().with_window_change(20260105, wide, to_date=20260109)
    assert s.windows_on(20260102) == [(555, 930)]
    assert s.windows_on(20260105) == [(540, 960)] and s.windows_on(20260109) == [(540, 960)]
    assert s.windows_on(20260112) == [(555, 930)]


@pytest.mark.parametrize(
    "call",
    [
        lambda: SessionSchedule.nse_equity().with_session_override(20260115, [(900, 540)]),
        lambda: SessionSchedule.nse_equity().with_session_override(20260231, []),
        lambda: SessionSchedule.nse_equity().with_window_change(20260109, {"mon": [(540, 900)]}, to_date=20260105),
        lambda: SessionSchedule.nse_equity().with_window_change(20260105, {"mon": [(540, 900), (800, 960)]}),
    ],
)
def test_a_bad_override_or_window_change_is_refused_by_name(call):
    with pytest.raises(InvalidParameter) as e:
        call()
    assert e.value.code == "invalid_session_schedule" and e.value.skill == "instruments"


# -- the calendar document ----------------------------------------------------------------------------


def test_a_calendar_document_gives_sessions_holidays_and_evening_overrides():
    nse = SessionSchedule.from_calendars("NSE", [Q4])
    assert nse.is_open(ist(2026, 10, 15, 10, 0))
    assert not nse.is_open(ist(2026, 10, 2, 10, 0)), "a holiday the document lists"
    assert not nse.is_open(ist(2026, 11, 8, 18, 30)), "Muhurat is not in the document"
    muhurat = nse.with_session_override(20261108, [(minutes(18, 15), minutes(19, 15))])
    assert muhurat.is_open(ist(2026, 11, 8, 18, 30)), "the caller adds it when the exchange publishes it"
    nfo = SessionSchedule.from_calendars("NFO", [Q4])
    assert nfo.is_open(ist(2026, 10, 15, 15, 39)) and not nfo.is_open(ist(2026, 10, 15, 15, 40))
    mcx = SessionSchedule.from_calendars("MCX", [Q4])
    assert not mcx.is_open(ist(2026, 10, 20, 12, 0)), "Dussehra: no day session"
    assert mcx.is_open(ist(2026, 10, 20, 17, 0)) and mcx.is_open(ist(2026, 10, 20, 23, 54))


def test_a_date_no_document_covers_is_closed_not_guessed():
    nse = SessionSchedule.from_calendars("NSE", [Q4])
    for when in (ist(2026, 9, 15, 10, 0), ist(2027, 1, 5, 10, 0)):
        assert not nse.is_open(when) and nse.session_date(when) is None


def test_the_latest_document_answers_for_the_dates_it_covers():
    mcx = SessionSchedule.from_calendars("MCX", [Q4, Q4_DST])
    assert mcx.is_open(ist(2026, 10, 15, 23, 29)) and not mcx.is_open(ist(2026, 10, 15, 23, 30))
    assert mcx.is_open(ist(2026, 11, 2, 23, 54)), "the DST document ends on 2026-11-01"
    plain = SessionSchedule.from_calendars("MCX", [Q4])
    assert plain.is_open(ist(2026, 10, 15, 23, 54))


@pytest.mark.parametrize(
    "documents, message",
    [
        (["{"], "not JSON"),
        ([json.dumps({"name": "x"})], "utc_offset"),
        ([Q4.replace('"NSE"', '"XSE"')], 'no entry for venue "NSE"'),
        ([Q4.replace('"15:30"', '"25:30"')], "not an HH:MM time"),
        ([Q4.replace('"2026-12-31"', '"2026-02-30"')], "valid_to"),
        ([Q4, Q4_DST.replace("+05:30", "+00:00")], "utc_offset differs"),
        ([], "at least one document"),
    ],
)
def test_a_bad_calendar_document_is_refused_with_its_reason(documents, message):
    with pytest.raises(InvalidParameter, match=message) as e:
        SessionSchedule.from_calendars("NSE", documents)
    assert (e.value.code, e.value.parameter, e.value.skill) == ("invalid_calendar", "documents", "instruments")


# -- composite bars anchored at the session open ---------------------------------------------------


def _two_days():
    stamps, rows = [], []
    for day, (start, n) in {2: (ist(2026, 11, 2, 9, 15), 375), 3: (ist(2026, 11, 3, 9, 15), 30)}.items():
        for i in range(n):
            stamps.append(start + (i + 1) * MIN)
            price = float(len(rows))
            rows.append((price, price + 0.5, price - 0.5, price))
    return {"X": bars(rows, stamps)}


class Composites(Strategy):
    def __init__(self):
        super().__init__()
        self.got = {}

    def on_start(self, ctx):
        self.subscribe_bars(5, "m")
        self.subscribe_bars(30, "m")
        self.subscribe_bars(1, "h")

    def on_bar(self, ctx, event):
        if event.timeframe is not None:  # a completed composite bar
            self.got.setdefault(event.timeframe, []).append((fmt(event.timestamp), event.open, event.close))


def _composites(**instrument):
    strategy = Composites()
    run_strategies(
        [strategy], _two_days(), BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close",
        instruments={"X": rb.InstrumentSpec.equity("X", **instrument)} if instrument else None,
    )
    return strategy.got


def test_composites_of_a_session_instrument_start_at_the_open_and_end_at_the_close():
    """30-minute NSE candles are 09:15-09:45, ...; the last 15:15-15:30 is cut at the close."""
    got = _composites(session=SessionSchedule.nse_equity())
    first_day = lambda name: [g for g in got[name] if g[0].startswith("11-02")]
    thirty = first_day("30m")
    assert [g[0] for g in thirty[:3]] == ["11-02 09:45", "11-02 10:15", "11-02 10:45"]
    assert thirty[0][1:] == (0.0, 29.0)  # the first thirty one-minute bars
    assert len(thirty) == 13 and thirty[-1] == ("11-02 15:30", 360.0, 374.0)
    hourly = first_day("1h")
    assert [g[0] for g in hourly[:2]] == ["11-02 10:15", "11-02 11:15"]
    assert len(hourly) == 7 and hourly[-1] == ("11-02 15:30", 360.0, 374.0), "15:15-15:30, not 15:15-16:15"
    assert first_day("5m")[0] == ("11-02 09:20", 0.0, 4.0)  # five-minute bars sat on the grid already


def test_without_a_session_composites_stay_on_the_clock_grid():
    got = _composites()
    assert [g[0] for g in got["30m"][:2]] == ["11-02 09:30", "11-02 10:00"]


# -- s2: squareoff_time in the instrument's zone ------------------------------------------------------------


def _squareoff_tape():
    stamps = [ist(2026, 11, 2, 15, m) for m in range(17, 31)] + [ist(2026, 11, 3, 9, m) for m in (16, 17, 18)]
    return {"X": bars([(100 + i, 101 + i, 99 + i, 100 + i) for i in range(len(stamps))], stamps)}


BUYS = {
    0: lambda s, c: s.submit_order(O.Market(side="buy", quantity=10.0)),
    5: lambda s, c: s.submit_order(O.Market(side="buy", quantity=5.0)),
    14: lambda s, c: s.submit_order(O.Market(side="buy", quantity=5.0)),
}


def _closed(strategy_cls, config, with_session=True):
    strategy = strategy_cls(BUYS)
    spec = rb.InstrumentSpec.equity("X", session=SessionSchedule.nse_equity() if with_session else None)
    (result,) = run_strategies(
        [strategy], _squareoff_tape(), config, bar_stamp="close", instruments={"X": spec},
    )
    rows = [(t.size, fmt(t.entry_time), fmt(t.exit_time), str(t.exit_reason)) for t in result.result.trades()]
    return rows, strategy.rejections


def test_squareoff_time_without_a_zone_is_refused():
    with pytest.raises(InvalidParameter) as e:
        _closed(Script, BacktestConfig(initial_capital=1e5, fees=0.0, squareoff_time="15:20"), with_session=False)
    assert (e.value.code, e.value.parameter, e.value.skill) == ("squareoff_needs_timezone", "squareoff_time", "execution")
    assert "session_tz_offset_ns" in str(e.value) and e.value.value == "X"


def test_squareoff_time_is_read_in_the_zone_it_is_given():
    """15:20 IST, not 15:20 UTC (which is 20:50 IST and never reached in this session)."""
    config = BacktestConfig(initial_capital=1e5, fees=0.0, squareoff_time="15:20", session_tz_offset_ns=IST)
    rows, rejected = _closed(Script, config, with_session=False)
    assert rows == [
        (10.0, "11-02 15:17", "11-02 15:20", "Squareoff"),
        (5.0, "11-03 09:16", "11-03 09:18", "EndOfData"),
    ]
    assert rejected == [("11-02 15:22", "squareoff")]


def test_squareoff_time_uses_the_instruments_session_zone_when_the_config_names_none():
    config = BacktestConfig(initial_capital=1e5, fees=0.0, squareoff_time="15:20")
    rows, _ = _closed(Script, config)
    assert rows[0] == (10.0, "11-02 15:17", "11-02 15:20", "Squareoff")


def _two_sessions(config):
    """A bullion-like and an agricultural MCX contract (17:00 close), each bought at its first
    bar on 2026-11-02 (MCX closes 23:55 that day); their exits."""
    agri = [ist(2026, 11, 2, 16, m) for m in range(30, 60)]
    crude = agri + [ist(2026, 11, 2, 17, 0)] + [ist(2026, 11, 2, 23, m) for m in range(0, 11)]
    data = {
        "AGRI": bars([(100, 101, 99, 100)] * len(agri), agri),
        "CRUDE": bars([(100, 101, 99, 100)] * len(crude), crude),
    }
    instruments = {
        "AGRI": rb.InstrumentSpec.equity("AGRI", session=SessionSchedule.mcx_other_agri()),
        "CRUDE": rb.InstrumentSpec.equity("CRUDE", session=SessionSchedule.mcx()),
    }
    buy = lambda s, c: s.submit_order(O.Market(side="buy", quantity=1.0))  # noqa: E731
    strategy = Script({("AGRI", 0): buy, ("CRUDE", 0): buy})
    (result,) = run_strategies([strategy], data, config, bar_stamp="close", instruments=instruments)
    return sorted((t.symbol, fmt(t.exit_time), str(t.exit_reason)) for t in result.result.trades())


def test_an_instrument_named_in_squareoff_times_is_squared_off_at_its_own_time():
    run = BacktestConfig(initial_capital=1e5, fees=0.0, squareoff_time="23:05")
    assert _two_sessions(run.with_squareoff_times({"AGRI": "16:35"})) == [
        ("AGRI", "11-02 16:35", "Squareoff"),
        ("CRUDE", "11-02 23:05", "Squareoff"),
    ]
    # without it the run's one time never comes inside the agricultural session
    agri, crude = _two_sessions(run)
    assert agri[0] == "AGRI" and agri[2] != "Squareoff"
    assert crude == ("CRUDE", "11-02 23:05", "Squareoff")
    assert run.with_squareoff_times({"AGRI": "16:35"}).squareoff_times == {"AGRI": 995}
    assert run.with_squareoff_times({"AGRI": "16:35"}).with_squareoff_times({}).squareoff_times == {}
    with pytest.raises(InvalidParameter):
        run.with_squareoff_times({"AGRI": "4:35pm"})


# -- M16: square-off only when the strategy says it holds intraday ----------------------------------------


class Intraday(Script):
    requirements = Requirements(
        roles={"x": Underlying(asset="equity")}, holding=Holding.intraday(flatten_before="15:20")
    )


class Overnight(Script):
    requirements = Requirements(roles={"x": Underlying(asset="equity")}, holding=Holding.overnight())


def test_a_positional_strategy_is_not_squared_off_by_the_session_preset():
    """Bought at 15:17, added to at 15:22 and the next morning (netting adds to the one position),
    held overnight, closed only when the data ends."""
    rows, rejected = _closed(Script, BacktestConfig(initial_capital=1e5, fees=0.0))
    assert rows == [(20.0, "11-02 15:17", "11-03 09:18", "EndOfData")]
    assert rejected == []
    rows, _ = _closed(Overnight, BacktestConfig(initial_capital=1e5, fees=0.0))
    assert rows == [(20.0, "11-02 15:17", "11-03 09:18", "EndOfData")]


def test_a_strategy_whose_requirements_say_intraday_is_flat_by_its_flatten_time():
    rows, rejected = _closed(Intraday, BacktestConfig(initial_capital=1e5, fees=0.0))
    assert rows == [
        (10.0, "11-02 15:17", "11-02 15:20", "Squareoff"),
        (5.0, "11-03 09:16", "11-03 09:18", "EndOfData"),
    ]
    assert rejected == [("11-02 15:22", "squareoff")]


def test_an_explicit_squareoff_time_beats_the_one_the_requirements_declare():
    rows, _ = _closed(Intraday, BacktestConfig(initial_capital=1e5, fees=0.0, squareoff_time="15:25"))
    assert rows[0] == (15.0, "11-02 15:17", "11-02 15:25", "Squareoff")  # the 15:22 buy adds before 15:25


def test_strategies_with_different_holdings_cannot_share_one_squareoff():
    spec = {"X": rb.InstrumentSpec.equity("X", session=SessionSchedule.nse_equity())}
    with pytest.raises(InvalidParameter) as e:
        run_strategies(
            [Intraday(BUYS), Overnight(BUYS)], _squareoff_tape(),
            BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close", instruments=spec,
        )
    assert (e.value.code, e.value.parameter, e.value.skill) == ("mixed_holding", "requirements.holding", "execution")
    assert e.value.value == ["intraday 15:20", "overnight"]


# -- gap / ml2: DAY validity by default on an instrument with a session ---------------------------------------


def _limit_run(session, **order):
    stamps = [ist(2026, 11, 2, 15, 28), ist(2026, 11, 2, 15, 29), ist(2026, 11, 3, 9, 16), ist(2026, 11, 3, 9, 17)]
    rows = [(100,) * 4, (100,) * 4, (98, 99, 97.5, 98.5), (98.5,) * 4]
    place = {0: lambda s, c: s.submit_order(O.Limit(side="buy", price=99.0, quantity=10.0, **order))}
    instruments = {"X": rb.InstrumentSpec.equity("X", session=SessionSchedule.nse_equity())} if session else None
    runs = run_paths(
        place, {"X": bars(rows, stamps)}, lambda: BacktestConfig(initial_capital=1e5, fees=0.0),
        instruments=instruments, account_type="margin",
    )
    return {
        path: ([(t.size, t.entry_price, fmt(t.entry_time)) for t in res.result.trades()], s.events)
        for path, (s, res) in runs.items()
    }


@pytest.mark.parametrize("order", [{}, {"tif": "day"}])
def test_a_limit_order_on_a_session_instrument_lapses_at_the_close(order):
    """Placed at 15:28 to buy at 99; the next session opens at 98 but the order is gone."""
    for path, (trades, events) in _limit_run(session=True, **order).items():
        assert trades == [] and events == ["order_accepted", "order_expired"], path


def test_gtc_is_still_available_explicitly_and_fills_at_the_next_open():
    for path, (trades, events) in _limit_run(session=True, tif="gtc").items():
        assert trades == [(10.0, 98.0, "11-03 09:16")], path  # the open, better than the limit
        assert events == ["order_accepted", "order_filled"], path


def test_an_instrument_without_a_session_keeps_good_till_cancelled_as_its_default():
    for path, (trades, events) in _limit_run(session=False).items():
        assert trades == [(10.0, 98.0, "11-03 09:16")], path
        assert events == ["order_accepted", "order_filled"], path


def test_the_order_default_is_auto_and_auto_is_among_the_documented_alternatives():
    assert O.Limit(side="buy", price=1.0, quantity=1.0).tif == "auto"
    with pytest.raises(rb.errors.InvalidOrder) as e:
        O.Limit(side="buy", price=1.0, quantity=1.0, tif="forever")
    assert e.value.code == "invalid_time_in_force" and "auto" in e.value.valid and "gtc" in e.value.valid


def test_an_order_placed_in_the_last_minute_expires_at_the_close_not_by_the_squareoff():
    """ml2: placed on the 15:29 bar it rests to 15:30 and lapses; placed on 15:30 it is refused."""
    stamps = [ist(2026, 11, 2, 15, 29), ist(2026, 11, 2, 15, 30), ist(2026, 11, 3, 9, 16), ist(2026, 11, 3, 9, 17)]
    rows = [(100,) * 4, (100,) * 4, (98, 99, 97.5, 98.5), (98.5,) * 4]
    for placed_on, events, rejections in (
        (0, ["order_accepted", "order_expired"], []),
        (1, ["order_rejected"], [("11-02 15:30", "market_closed")]),
    ):
        script = {placed_on: lambda s, c: s.submit_order(O.Limit(side="buy", price=99.0, quantity=10.0))}
        runs = run_paths(
            script, {"X": bars(rows, stamps)}, lambda: BacktestConfig(initial_capital=1e5, fees=0.0),
            instruments={"X": rb.InstrumentSpec.equity("X", session=SessionSchedule.nse_equity())},
            paths=("batch",),
        )
        strategy, result = runs["batch"]
        assert result.result.trades() == []
        assert strategy.events == events and strategy.rejections == rejections


# -- a date the calendar does not cover is not a holiday ------------------------------------------------------


def _mcx_run(year, month, day):
    stamp = ist(year, month, day, 10, 0)
    data = {"M": bars([(100,) * 4] * 3, stamp)}
    spec = {"M": rb.InstrumentSpec.futures_contract("M", ist(2027, 1, 1, 0, 0), 1.0, session=SessionSchedule.mcx())}
    return run_strategies(
        [Script({})], data, BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close", instruments=spec
    )


def test_bars_on_a_date_the_session_schedule_does_not_cover_raise_session_not_covered():
    """MCX before 2024-11-04 has no dated close here; a holiday would be `market_closed`."""
    with pytest.raises(rb.errors.InvalidData) as e:
        _mcx_run(2024, 6, 10)
    err = e.value
    assert (err.code, err.parameter, err.value, err.skill) == ("session_not_covered", "session", 20240610, "instruments")
    assert list(err.valid) == ["20241104..22001231"] and "20240610" in str(err)
    _mcx_run(2026, 6, 15)  # inside the covered dates it runs


def test_the_mcx_preset_follows_the_us_clock_rule_past_2027():
    """The close has no dated end: it moves with the US clocks every year, and a bar in 2028 runs."""
    mcx = SessionSchedule.mcx()
    # 2028: the second Sunday of March is the 12th, the first of November the 5th
    assert mcx.windows_on(20280310)[-1] == (1020, minutes(23, 55))
    assert mcx.windows_on(20280313)[-1] == (1020, minutes(23, 30))
    assert mcx.windows_on(20281103)[-1] == (1020, minutes(23, 30))
    assert mcx.windows_on(20281106)[-1] == (1020, minutes(23, 55))
    spec = {"M": rb.InstrumentSpec.futures_contract("M", ist(2028, 6, 1, 0, 0), 1.0, session=mcx)}
    run_strategies([Script({})], {"M": bars([(100,) * 4] * 3, ist(2028, 3, 15, 10, 0))},
                   BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close", instruments=spec)


def test_a_live_node_refuses_a_pushed_bar_outside_the_covered_dates():
    spec = {"M": rb.InstrumentSpec.futures_contract("M", ist(2027, 1, 1, 0, 0), 1.0, session=SessionSchedule.mcx())}
    node = LiveNode([Script({})], ["M"], BacktestConfig(initial_capital=1e5, fees=0.0), bar_stamp="close", instruments=spec)
    node.push_bar("M", ist(2026, 6, 15, 10, 0), 100.0, 100.0, 100.0, 100.0, 1.0)
    with pytest.raises(rb.errors.InvalidData) as e:
        node.push_bar("M", ist(2024, 11, 1, 10, 0), 100.0, 100.0, 100.0, 100.0, 1.0)
    assert (e.value.code, e.value.value) == ("session_not_covered", 20241101)


# -- order validity is a capability of the exchange segment ------------------------------------------------------------


def _validity_run(venue, kind="future", **order):
    start = ist(2026, 11, 2, 10, 0)
    if kind == "future":
        # "F" is a name MCX has no class for: its session is given, not implied
        spec = rb.InstrumentSpec.futures_contract(
            "F", ist(2026, 11, 26, 15, 30), 1.0, venue=venue,
            session=rb.SessionSchedule.mcx() if venue == "MCX" else None,
        )
    else:
        spec = rb.InstrumentSpec.equity("F", venue=venue)
    place = {1: lambda s, c: s.submit_order(O.Limit(side="buy", price=99.0, quantity=1.0, **order))}
    return run_paths(
        place, {"F": bars([(100,) * 4] * 4, start)}, lambda: BacktestConfig(initial_capital=1e6, fees=0.0),
        instruments={"F": spec}, account_type="margin", paths=("batch", "node"),
    )


@pytest.mark.parametrize("venue", ["NFO", "BFO", "CDS", "BCD", "MCX", "BSE"])
def test_gtc_is_refused_where_the_segment_holds_only_day_orders(venue):
    with pytest.raises(rb.errors.InvalidOrder) as e:
        _validity_run(venue, tif="gtc")
    err = e.value
    assert (err.code, err.parameter, err.value, err.skill) == ("order_validity_unsupported", "tif", "gtc", "orders")
    assert list(err.valid) == ["day", "ioc", "fok"] and "host" in str(err)


@pytest.mark.parametrize("venue, kind", [("NSE", "cash"), ("NCO", "future")])
def test_gtc_is_native_on_nse_cash_and_nse_commodity(venue, kind):
    for path, (strategy, _) in _validity_run(venue, kind=kind, tif="gtc").items():
        assert strategy.events[:1] == ["order_accepted"], path


def test_a_gtd_on_nse_commodity_cannot_outlive_the_contract():
    with pytest.raises(rb.errors.InvalidOrder, match="past the maturity"):
        _validity_run("NCO", tif="gtd", expire_ns=ist(2026, 12, 31, 15, 30))
    _validity_run("NCO", tif="gtd", expire_ns=ist(2026, 11, 20, 15, 30))


def test_the_default_on_a_day_only_segment_is_day_even_without_a_session():
    for path, (strategy, _) in _validity_run("NFO").items():
        assert strategy.events[:1] == ["order_accepted"], path


# -- the orders the engine builds take a validity the segment holds -------------------------------------------------


def _intent_run(venue, kind):
    """``enter()`` with a stop and a target on bar 1, ``close_position()`` on bar 3, flat at 100:
    a backtest builds no order for either, and a venue run built them gtc, which a day-only
    segment refuses (``order_validity_unsupported`` on the entry)."""
    start, expiry = ist(2026, 11, 2, 10, 0), ist(2026, 11, 26, 15, 30)
    if kind == "option":
        spec = rb.InstrumentSpec.option("F", 25000.0, "call", expiry, 75.0, venue=venue)
    elif kind == "future":
        spec = rb.InstrumentSpec.futures_contract(
            "F", expiry, 1.0, venue=venue, session=rb.SessionSchedule.mcx() if venue == "MCX" else None
        )
    else:
        spec = rb.InstrumentSpec.equity("F", venue=venue)
    script = {
        1: lambda s, c: s.enter(size_frac=0.1, stop_price=90.0, target_price=110.0),
        3: lambda s, c: s.close_position(),
    }
    return run_paths(
        script, {"F": bars([(100,) * 4] * 6, start)}, lambda: BacktestConfig(initial_capital=1e6, fees=0.0),
        instruments={"F": spec}, account_type="margin",
    )


@pytest.mark.parametrize(
    "venue, kind, tif",
    [("NFO", "option", "day"), ("NFO", "future", "day"), ("MCX", "future", "day"), ("NSE", "cash", "gtc")],
)
def test_enter_and_close_position_send_the_validity_the_segment_holds_and_trade_as_a_backtest(venue, kind, tif):
    out = _intent_run(venue, kind)
    trades = {p: [(t.entry_idx, t.exit_idx, t.size) for t in r.result.trades()] for p, (_, r) in out.items()}
    assert len(trades["batch"]) == 1 and trades["batch"][0][2] > 0, trades
    assert trades["node"] == trades["batch"] and trades["paper"] == trades["batch"], trades
    orders = out["paper"][1].result.orders()
    # the entry, its stop and target legs and the close: day where the segment holds no gtc (NFO,
    # MCX), gtc where it does (NSE cash, as before)
    assert {o.client_id.split("-")[0] for o in orders} >= {"enter", "prot", "close"}, orders
    assert {o.tif for o in orders} == {tif}, [(o.client_id, o.tif) for o in orders]
