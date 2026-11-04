"""Trading sessions: SessionSchedule, instrument venue/currency, enforcement.

An instrument with no schedule behaves exactly as it always did; with one,
the shared kernel refuses orders while the market is closed, skips matching
outside it, squares positions off, expires DAY orders at the close, and the
daily composite bars follow the trading dates.

Times are IST wall clock, written as ``(yyyymmdd, hour, minute)`` and turned
into UTC nanoseconds by :func:`ist`. 15 Jan 2026 is a Thursday.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    InstrumentSpec,
    SessionSchedule,
    Strategy,
    run_portfolio_strategy,
    run_strategies,
    run_strategy_backtest,
)
from raptorbt.strategy import orders

NS_MIN = 60 * 10**9
NS_DAY = 86_400 * 10**9
IST_NS = (5 * 3600 + 30 * 60) * 10**9

THU, FRI, SAT, MON, TUE = 20260115, 20260116, 20260117, 20260119, 20260120


def _days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def ist(date, hour=0, minute=0):
    """UTC nanoseconds of an IST wall-clock time on a yyyymmdd date."""
    day = _days_from_civil(date // 10000, date // 100 % 100, date % 100)
    return day * NS_DAY + (hour * 60 + minute) * NS_MIN - IST_NS


def _zero_fee_config():
    config = BacktestConfig()
    config.fees = 0.0
    return config


def bars(*spans, step=5, price=100.0, low_gap=1.0):
    """Close-stamped bars for ``(date, (h, m), (h, m))`` spans, ``step`` minutes.

    The first bar of a span closes ``step`` minutes after its start.
    """
    stamps = []
    for date, (h0, m0), (h1, m1) in spans:
        t = ist(date, h0, m0) + step * NS_MIN
        end = ist(date, h1, m1)
        while t <= end:
            stamps.append(t)
            t += step * NS_MIN
    n = len(stamps)
    close = np.full(n, price)
    return {
        "timestamps": np.array(stamps, dtype=np.int64),
        "open": close.copy(),
        "high": close + 1.0,
        "low": close - low_gap,
        "close": close,
        "volume": np.full(n, 1000.0),
    }


def regular_day(date, **kw):
    """09:20..15:30 close-stamped, i.e. exactly the NSE session."""
    return (date, (9, 15), (15, 30))


def spec(session=None, symbol="ASSET", **kw):
    return InstrumentSpec.equity(symbol, session=session, **kw)


def run(strategy, data, instrument=None, **kw):
    return run_strategy_backtest(
        strategy,
        **data,
        instrument=instrument,
        config=kw.pop("config", _zero_fee_config()),
        bar_stamp="close",
        **kw,
    )


class Log(Strategy):
    """Records order events as (kind, bar index, reject reason or None)."""

    def __init__(self):
        super().__init__(None)
        self.events = []

    def on_order_event(self, ctx, event):
        if event.kind != "order_rejected":  # recorded by on_order_rejected
            self.events.append((event.kind, event.idx, event.reject_reason))

    def on_order_rejected(self, ctx, event):
        self.events.append((event.kind, event.idx, event.reject_reason))

    def reasons(self):
        return [r for k, _, r in self.events if k in ("order_rejected", "entry_rejected")]

    def kinds(self):
        return [k for k, _, _ in self.events]


# ---------------------------------------------------------------------------
# SessionSchedule
# ---------------------------------------------------------------------------
def nse_with_squareoff(**kw):
    """The NSE equity preset with the 15:20 square-off most tests here are about (it has none of its own)."""
    return SessionSchedule.nse_equity(**{"squareoff_minute": 15 * 60 + 20, **kw})


class TestScheduleMaths:
    def test_nse_window_is_half_open_and_weekday_only(self):
        s = SessionSchedule.nse_equity()
        assert not s.is_open(ist(THU, 9, 14))
        assert s.is_open(ist(THU, 9, 15))
        assert s.is_open(ist(THU, 15, 29))
        assert not s.is_open(ist(THU, 15, 30))
        assert not s.is_open(ist(SAT, 11, 0))
        assert s.is_open(ist(MON, 11, 0))

    def test_presets_match_documented_times(self):
        for s in (
            SessionSchedule.nse_equity(),
            SessionSchedule.nse_fo(),
            SessionSchedule.bse_equity(),
        ):
            assert s.tz_offset_ns == IST_NS
            assert s.windows == {d: [(9 * 60 + 15, 15 * 60 + 30)] for d in
                                 ("mon", "tue", "wed", "thu", "fri")}
            assert s.squareoff_minute is None, "square-off is the run's choice, not the preset's"
            assert s.holidays == []

    def test_preset_arguments(self):
        s = SessionSchedule.nse_equity(holidays=[20260126, 20260101], squareoff_minute=None)
        assert s.holidays == [20260101, 20260126]
        assert s.squareoff_minute is None
        assert SessionSchedule.nse_fo(squareoff_minute=15 * 60 + 25).squareoff_minute == 925

    def test_a_lunch_break_is_two_windows(self):
        day = [(9 * 60, 12 * 60), (13 * 60, 15 * 60)]
        s = SessionSchedule({d: day for d in ("mon", "tue", "wed", "thu", "fri")},
                            tz_offset_ns=IST_NS)
        assert s.is_open(ist(THU, 11, 59))
        assert not s.is_open(ist(THU, 12, 0))
        assert not s.is_open(ist(THU, 12, 59))
        assert s.is_open(ist(THU, 13, 0))
        assert s.next_open_ns(ist(THU, 10, 0)) == ist(THU, 13, 0)
        assert s.session_close_ns(THU) == ist(THU, 15, 0)

    def test_weekday_windows_differ_by_day(self):
        s = SessionSchedule(
            {"mon": [(540, 900)], 4: [(540, 720)], "SAT": [(600, 660)]}, tz_offset_ns=IST_NS
        )
        assert s.windows == {"mon": [(540, 900)], "fri": [(540, 720)], "sat": [(600, 660)]}
        assert s.is_open(ist(SAT, 10, 30)) and not s.is_open(ist(SAT, 11, 0))
        assert not s.is_open(ist(20260118, 10, 30)), "Sunday has no window"

    def test_holidays_are_caller_supplied_and_close_the_day(self):
        plain = SessionSchedule.nse_equity()
        assert plain.holidays == []
        assert plain.is_open(ist(20260126, 10, 0)), "no calendar ships with the library"
        s = plain.with_holidays([20260126])
        assert not s.is_open(ist(20260126, 10, 0))
        assert s.session_date(ist(20260126, 10, 0)) is None
        assert s.session_close_ns(20260126) is None
        assert s.next_open_ns(ist(20260123, 15, 30)) == ist(20260127, 9, 15)
        # with_holidays sets the list; it does not accumulate.
        assert s.with_holidays([]).is_open(ist(20260126, 10, 0))
        assert plain.holidays == [], "the original is unchanged"

    def test_timezone_decides_the_local_day_and_time(self):
        utc_open = _days_from_civil(2026, 1, 15) * NS_DAY + (3 * 60 + 45) * NS_MIN  # 09:15 IST
        assert SessionSchedule.nse_equity().is_open(utc_open)
        utc = SessionSchedule({"thu": [(9 * 60 + 15, 15 * 60 + 30)]})
        assert not utc.is_open(utc_open)
        new_york = SessionSchedule({"thu": [(570, 960)]}, tz_offset_ns=-5 * 3600 * 10**9)
        at_1430_utc = _days_from_civil(2026, 1, 15) * NS_DAY + (14 * 60 + 30) * NS_MIN
        assert new_york.is_open(at_1430_utc)
        assert not new_york.is_open(at_1430_utc - NS_MIN)

    def test_session_date_ignores_time_of_day(self):
        s = SessionSchedule.nse_equity()
        assert s.session_date(ist(THU, 3, 0)) == THU
        assert s.session_date(ist(THU, 23, 59)) == THU
        assert s.session_date(ist(SAT, 12, 0)) is None

    def test_next_open_is_strictly_after(self):
        s = SessionSchedule.nse_equity()
        assert s.next_open_ns(ist(THU, 8, 0)) == ist(THU, 9, 15)
        assert s.next_open_ns(ist(THU, 9, 15)) == ist(FRI, 9, 15)
        assert s.next_open_ns(ist(FRI, 16, 0)) == ist(MON, 9, 15)

    def test_next_open_is_none_when_every_day_is_a_holiday(self):
        first = _days_from_civil(2026, 1, 1)
        # Not worth a calendar library: cover the whole look-ahead year.
        dates = []
        for offset in range(-5, 500):
            z = first + offset + 719468
            era = z // 146097
            doe = z - era * 146097
            yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
            doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
            mp = (5 * doy + 2) // 153
            d = doy - (153 * mp + 2) // 5 + 1
            m = mp + 3 if mp < 10 else mp - 9
            dates.append((yoe + era * 400 + (m <= 2)) * 10000 + m * 100 + d)
        s = SessionSchedule.nse_equity(holidays=dates)
        assert s.next_open_ns(ist(THU, 10, 0)) is None

    def test_square_off_minute_can_be_changed_or_removed(self):
        s = nse_with_squareoff()
        assert s.squareoff_minute == 920
        assert s.with_squareoff_minute(900).squareoff_minute == 900
        assert s.with_squareoff_minute(None).squareoff_minute is None

    def test_equality_and_repr(self):
        assert SessionSchedule.nse_equity() == SessionSchedule.bse_equity()
        assert SessionSchedule.nse_fo() == SessionSchedule.bse_fo()
        assert SessionSchedule.nse_equity() != SessionSchedule.nse_fo(), "the derivatives close later from 2026-08-03"
        assert SessionSchedule.nse_equity() != SessionSchedule.nse_equity().with_holidays([20260126])
        assert "SessionSchedule" in repr(SessionSchedule.nse_equity())


class TestScheduleValidation:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(windows={"mon": [(900, 540)]}), "before close"),
            (dict(windows={"mon": [(540, 540)]}), "before close"),
            (dict(windows={"mon": [(0, 1500)]}), "1440"),
            (dict(windows={"mon": [(540, 700), (600, 900)]}), "overlap"),
            (dict(windows={}), "at least one"),
            (dict(windows={"mon": []}), "at least one"),
            (dict(windows={"funday": [(1, 2)]}), "not a weekday"),
            (dict(windows={7: [(1, 2)]}), "not a weekday"),
            (dict(windows={"mon": [(540, 900)]}, tz_offset_ns=25 * 3600 * 10**9), "tz_offset_ns"),
            (dict(windows={"mon": [(540, 900)]}, holidays=[20260230]), "holiday"),
            (dict(windows={"mon": [(540, 900)]}, holidays=[2026115]), "holiday"),
            (dict(windows={"mon": [(540, 900)]}, squareoff_minute=1000), "squareoff_minute"),
            (dict(windows={"mon": "nine to five"}), "list of"),
        ],
    )
    def test_bad_input_has_a_clear_error(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SessionSchedule(**kwargs)

    def test_with_methods_validate_too(self):
        s = nse_with_squareoff()
        with pytest.raises(ValueError, match="holiday"):
            s.with_holidays([20261340])
        with pytest.raises(ValueError, match="squareoff_minute"):
            s.with_squareoff_minute(60)

    def test_session_close_ns_rejects_a_non_date(self):
        with pytest.raises(ValueError, match="yyyymmdd"):
            nse_with_squareoff().session_close_ns(20260231)


# ---------------------------------------------------------------------------
# InstrumentSpec metadata
# ---------------------------------------------------------------------------
FACTORIES = {
    "equity": lambda **kw: InstrumentSpec.equity("A", **kw),
    "futures_contract": lambda **kw: InstrumentSpec.futures_contract("A", 10**18, 1.0, **kw),
    "perpetual": lambda **kw: InstrumentSpec.perpetual("A", **kw),
    "option": lambda **kw: InstrumentSpec.option("A", 100.0, "call", 10**18, 1.0, **kw),
    "currency_pair": lambda **kw: InstrumentSpec.currency_pair("A", **kw),
    "index": lambda **kw: InstrumentSpec.index("A", **kw),
}


class TestInstrumentMetadata:
    @pytest.mark.parametrize("name", FACTORIES)
    def test_every_factory_takes_venue_currency_and_session(self, name):
        schedule = nse_with_squareoff()
        made = FACTORIES[name](venue="NSE", currency="INR", session=schedule)
        assert (made.venue, made.currency) == ("NSE", "INR")
        assert made.session == schedule

    @pytest.mark.parametrize("name", FACTORIES)
    def test_metadata_defaults_to_none(self, name):
        made = FACTORIES[name]()
        assert made.venue is None and made.currency is None and made.session is None

    def test_properties_are_settable(self):
        made = InstrumentSpec.equity("A")
        made.venue = "BSE"
        made.currency = "INR"
        made.session = SessionSchedule.bse_equity()
        assert (made.venue, made.currency) == ("BSE", "INR")
        assert made.session == SessionSchedule.bse_equity()
        made.session = None
        made.venue = None
        assert made.session is None and made.venue is None

    def test_empty_strings_are_refused(self):
        with pytest.raises(ValueError, match="venue"):
            InstrumentSpec.equity("A", venue="")
        with pytest.raises(ValueError, match="currency"):
            InstrumentSpec.equity("A", currency="  ")

    def test_currency_is_metadata_only(self):
        data = bars(regular_day(THU), regular_day(FRI))

        class Trade(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 3:
                    self.enter()
                if ctx.idx == 40:
                    self.close_position()

        plain = run(Trade(), data, spec())
        tagged = run(Trade(), data, spec(venue="LSE", currency="USD"))
        _assert_same_result(plain, tagged)


def _assert_same_result(a, b):
    ta, tb = a.trades(), b.trades()
    assert len(ta) == len(tb)
    for x, y in zip(ta, tb):
        assert (x.entry_idx, x.exit_idx, x.entry_price, x.exit_price, x.size, x.pnl, x.fees,
                x.exit_reason) == (
            y.entry_idx, y.exit_idx, y.entry_price, y.exit_price, y.size, y.pnl, y.fees,
            y.exit_reason,
        )
    np.testing.assert_array_equal(np.asarray(a.equity_curve()), np.asarray(b.equity_curve()))


# ---------------------------------------------------------------------------
# No schedule: nothing changes
# ---------------------------------------------------------------------------
class TestNoScheduleIsUnchanged:
    def _busy_strategy(self):
        class Busy(Log):
            def on_bar(self, ctx):
                i = ctx.idx
                if i % 17 == 3:
                    self.enter()
                if i % 17 == 9:
                    self.close_position()
                if i % 11 == 5:
                    self.submit_order(orders.Limit(side="buy", price=99.0, size_frac=0.1,
                                                   tif="day"))

        return Busy()

    def test_spec_without_schedule_equals_no_spec_bit_for_bit(self):
        # Off-hours bars, a weekend and a long gap are all just data.
        data = bars((THU, (3, 0), (23, 0)), (SAT, (3, 0), (20, 0)), (MON, (9, 15), (15, 30)),
                    step=7)
        a = self._busy_strategy()
        b = self._busy_strategy()
        no_spec = run(a, data, None)
        # a spec that names no venue (so implies no session) is pure data; a venue the presets
        # cover would bring its session (pinned below), and a day order on one they do not is refused
        no_schedule = run(b, data, spec(currency="INR"))
        _assert_same_result(no_spec, no_schedule)
        assert a.events == b.events
        assert len(no_spec.trades()) > 3

    def test_portfolio_runner_without_schedule_is_unchanged(self):
        data = {"A": bars((THU, (3, 0), (23, 0)), step=9)}
        a = run_portfolio_strategy(
            self._busy_strategy(), data, config=_zero_fee_config(), bar_stamp="close"
        ).result
        b = run_portfolio_strategy(
            self._busy_strategy(), data, config=_zero_fee_config(), bar_stamp="close",
            instruments={"A": spec(symbol="A", currency="INR")},
        ).result
        assert np.array_equal(np.asarray(a.equity_curve()), np.asarray(b.equity_curve()))
        assert len(a.trades()) == len(b.trades())

    def test_a_known_venue_implies_its_session_so_off_hours_bars_are_not_just_data(self):
        data = bars((THU, (3, 0), (23, 0)), (SAT, (3, 0), (20, 0)), (MON, (9, 15), (15, 30)),
                    step=7)
        implied = self._busy_strategy()
        none = self._busy_strategy()
        with_venue = run(implied, data, spec(venue="NSE", currency="INR"))
        without = run(none, data, None)
        assert len(with_venue.trades()) < len(without.trades()), "the closed hours took no entries"
        assert spec(venue="NSE").session is not None and spec(venue="LSE").session is None


# ---------------------------------------------------------------------------
# Out-of-hours rejection
# ---------------------------------------------------------------------------
ORDER_KINDS = {
    "market": lambda: orders.Market(side="buy", units=1.0),
    "limit": lambda: orders.Limit(side="buy", price=95.0, units=1.0),
    "stop_market": lambda: orders.StopMarket(side="buy", trigger=105.0, units=1.0),
    "stop_limit": lambda: orders.StopLimit(side="buy", trigger=105.0, price=106.0, units=1.0),
    "market_if_touched": lambda: orders.MarketIfTouched(side="buy", trigger=95.0, units=1.0),
    "limit_if_touched": lambda: orders.LimitIfTouched(side="buy", trigger=95.0, price=94.0,
                                                       units=1.0),
    "market_to_limit": lambda: orders.MarketToLimit(side="buy", units=1.0),
    "trailing_stop_market": lambda: orders.TrailingStopMarket(side="buy", offset=1.0, units=1.0),
    "trailing_stop_limit": lambda: orders.TrailingStopLimit(side="buy", offset=1.0,
                                                             limit_offset=0.5, units=1.0),
}


class TestClosedMarket:
    def _submit_at(self, idx, make):
        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == idx:
                    self.submit_order(make())

        return S()

    @pytest.mark.parametrize("kind", ORDER_KINDS)
    def test_every_order_kind_is_rejected_outside_the_session(self, kind):
        # 08:00-09:10 bars precede the open; the first is bar 0.
        data = bars((THU, (7, 55), (9, 10)), regular_day(THU))
        s = self._submit_at(2, ORDER_KINDS[kind])
        result = run(s, data, spec(nse_with_squareoff()))
        assert s.reasons() == ["market_closed"], kind
        assert len(result.trades()) == 0

    @pytest.mark.parametrize("kind", ORDER_KINDS)
    def test_the_same_order_is_accepted_inside_the_session(self, kind):
        data = bars((THU, (7, 55), (9, 10)), regular_day(THU))
        n_before = 16  # bars before the open
        s = self._submit_at(n_before + 3, ORDER_KINDS[kind])
        run(s, data, spec(nse_with_squareoff()))
        assert s.reasons() == [], kind
        assert "order_accepted" in s.kinds() or "order_filled" in s.kinds()

    def test_rejection_is_on_the_order_rejected_hook_with_its_reason(self):
        seen = []

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 1:
                    self.submit_order(orders.Market(side="buy", units=1.0))

            def on_order_rejected(self, ctx, event):
                seen.append((event.kind, event.reject_reason))

        data = bars((THU, (7, 55), (9, 10)))
        run(S(), data, spec(nse_with_squareoff()))
        assert seen == [("order_rejected", "market_closed")]

    def test_entry_intents_are_rejected_when_closed(self):
        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == 2:
                    self.enter()

        data = bars((THU, (7, 55), (9, 10)))
        s = S()
        result = run(s, data, spec(nse_with_squareoff()))
        assert s.reasons() == ["market_closed"]
        assert len(result.trades()) == 0

    def test_entry_intents_trade_inside_the_session(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 5:
                    self.enter()

        result = run(S(), bars(regular_day(THU)), spec(nse_with_squareoff()))
        assert len(result.trades()) == 1

    def test_holidays_and_weekends_are_closed(self):
        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx in (2, 30, 60):
                    self.submit_order(orders.Market(side="buy", units=1.0))

        schedule = nse_with_squareoff(holidays=[FRI], squareoff_minute=None)
        data = bars((THU, (15, 0), (15, 25)), regular_day(FRI), regular_day(SAT), regular_day(MON))
        # idx 2 is Thursday 15:15 (open); 30 is Friday (holiday); 60 is Saturday.
        s = S()
        run(s, data, spec(schedule))
        assert s.reasons() == ["market_closed", "market_closed"]

    def test_the_session_close_instant_is_closed(self):
        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == 3:
                    self.submit_order(orders.Market(side="buy", units=1.0))

        # Bar 3 is the one closing at exactly 15:30.
        data = bars((THU, (15, 10), (15, 30)))
        s = S()
        run(s, data, spec(nse_with_squareoff().with_squareoff_minute(None)))
        assert data["timestamps"][3] == ist(THU, 15, 30)
        assert s.reasons() == ["market_closed"]


# ---------------------------------------------------------------------------
# Matching outside the session
# ---------------------------------------------------------------------------
class TestResting:
    def test_a_resting_order_does_not_match_data_outside_the_session(self):
        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == 2:
                    # GTC on purpose: a DAY order would lapse at Thursday's close
                    self.submit_order(orders.Limit(side="buy", price=95.0, units=1.0, tif="gtc"))

        # Thursday's session, then a post-close bar that trades through the
        # limit, then Friday's session where the limit is touched again.
        data = bars((THU, (15, 0), (15, 20)))
        post = bars((THU, (16, 0), (16, 5)), low_gap=10.0)
        fri = bars((FRI, (9, 15), (9, 30)), low_gap=10.0)
        merged = {k: np.concatenate([data[k], post[k], fri[k]]) for k in data}
        s = S()
        result = run(s, merged, spec(nse_with_squareoff(squareoff_minute=None)))
        filled_at = [idx for kind, idx, _ in s.events if kind == "order_filled"]
        first_friday_bar = len(data["timestamps"]) + len(post["timestamps"])
        assert filled_at == [first_friday_bar]
        assert len(result.trades()) == 1

    def test_without_a_schedule_the_same_order_fills_after_hours(self):
        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == 2:
                    self.submit_order(orders.Limit(side="buy", price=95.0, units=1.0))

        data = bars((THU, (15, 0), (15, 20)))
        post = bars((THU, (16, 0), (16, 5)), low_gap=10.0)
        merged = {k: np.concatenate([data[k], post[k]]) for k in data}
        s = S()
        run(s, merged, spec())
        assert [i for k, i, _ in s.events if k == "order_filled"] == [len(data["timestamps"])]


# ---------------------------------------------------------------------------
# Square-off
# ---------------------------------------------------------------------------
class TestSquareoff:
    def _two_days(self):
        return bars(regular_day(THU), regular_day(FRI))

    @pytest.mark.parametrize("side, direction", [("buy", 1), ("sell", -1)])
    def test_open_positions_close_once_per_session(self, side, direction):
        per_day = len(bars(regular_day(THU))["timestamps"])

        class S(Strategy):
            def on_bar(self, ctx):
                # Enter early on each day; never exit.
                if ctx.idx in (4, per_day + 4):
                    self.submit_order(orders.Market(side=side, units=10.0))

        result = run(S(), self._two_days(), spec(nse_with_squareoff()),
                     direction=direction)
        trades = result.trades()
        assert [t.exit_reason for t in trades] == ["Squareoff", "Squareoff"]
        close_ts = result_timestamps(self._two_days())
        # First event at or after 15:20, each day.
        assert [close_ts[t.exit_idx] for t in trades] == [ist(THU, 15, 20), ist(FRI, 15, 20)]

    def test_nothing_is_closed_without_a_square_off_minute(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 4:
                    self.enter()

        schedule = nse_with_squareoff(squareoff_minute=None)
        result = run(S(), self._two_days(), spec(schedule))
        assert [t.exit_reason for t in result.trades()] == ["EndOfData"]

    def test_no_schedule_means_no_square_off(self):
        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 4:
                    self.enter()

        result = run(S(), self._two_days(), spec())
        assert [t.exit_reason for t in result.trades()] == ["EndOfData"]

    def test_an_entry_on_the_square_off_bar_is_refused(self):
        data = bars(regular_day(THU))
        sq_idx = int(np.searchsorted(data["timestamps"], ist(THU, 15, 20)))

        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == sq_idx:
                    self.enter()

        s = S()
        result = run(s, data, spec(nse_with_squareoff()))
        assert s.reasons() == ["squareoff"]
        assert len(result.trades()) == 0

    def test_a_holiday_has_no_square_off_and_a_later_day_is_unaffected(self):
        per_day = len(bars(regular_day(THU))["timestamps"])

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == per_day + 4:
                    self.submit_order(orders.Market(side="buy", units=10.0))

        schedule = nse_with_squareoff(holidays=[FRI])
        # Friday is a holiday, so the entry is refused and nothing trades.
        result = run(S(), self._two_days(), spec(schedule))
        assert len(result.trades()) == 0

    def test_a_squared_off_position_cannot_reopen_and_carry_overnight(self):
        data = self._two_days()
        per_day = len(bars(regular_day(THU))["timestamps"])
        sq_idx = int(np.searchsorted(data["timestamps"], ist(THU, 15, 20)))

        class Eager(Log):
            def on_bar(self, ctx):
                # Try to hold a position at every bar until the Friday morning.
                if ctx.idx < per_day + 4 and not ctx.positions():
                    self.enter()

        s = Eager()
        result = run(s, data, spec(nse_with_squareoff()))
        trades = result.trades()
        # Thursday's trade is squared off; nothing re-opens on Thursday after
        # that (every later bar is refused); Friday trades again from 09:20.
        assert trades[0].exit_reason == "Squareoff"
        assert [t.entry_idx for t in trades if t.entry_idx > sq_idx][0] >= per_day
        # The square-off happens before the decision, so `on_bar` of the square-off bar already
        # sees a flat book and tries to re-enter: refused on that bar too (per_day - sq_idx bars,
        # sq_idx included). Nothing reopens; the money rule is unchanged.
        assert s.reasons().count("squareoff") == per_day - sq_idx
        assert all(t.exit_reason != "EndOfData" or t.entry_idx >= per_day for t in trades)

    def test_square_off_cancels_working_entry_orders_but_not_reduce_only_ones(self):
        data = bars(regular_day(THU))
        sq_idx = int(np.searchsorted(data["timestamps"], ist(THU, 15, 20)))

        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == 2:
                    self.submit_order(orders.Limit(side="buy", price=50.0, units=1.0),
                                      client_id="entry")
                    self.submit_order(orders.Limit(side="sell", price=500.0, units=1.0,
                                                   reduce_only=True), client_id="exit")

            def on_order_canceled(self, ctx, event):
                self.events.append(("canceled", ctx.idx, event.client_order_id))

        s = S()
        run(s, data, spec(nse_with_squareoff()))
        assert [(i, c) for k, i, c in s.events if k == "canceled"] == [(sq_idx, "entry")]

    def test_the_block_is_the_same_in_every_runner(self):
        data = self._two_days()
        per_day = len(bars(regular_day(THU))["timestamps"])

        with pytest.warns(DeprecationWarning, match="deprecated hooks"):
            class Eager(Log):
                holding = False

                def on_position_opened(self, ctx, event):
                    self.holding = True

                def on_position_closed(self, ctx, event):
                    self.holding = False

                def on_bar(self, ctx):
                    if ctx.idx < per_day + 4 and not self.holding:
                        self.enter()

        schedule = nse_with_squareoff()
        a = Eager()
        r1 = run(a, data, spec(schedule))
        b = Eager()
        r2 = run_portfolio_strategy(b, {"ASSET": data}, config=_zero_fee_config(),
                                    bar_stamp="close", instruments={"ASSET": spec(schedule)})
        c = Eager()
        (r3,) = run_strategies([(c, ["ASSET"])], {"ASSET": data}, config=_zero_fee_config(),
                               bar_stamp="close", instruments={"ASSET": spec(schedule)})

        def key(r):
            return [(t.entry_idx, t.exit_idx, t.exit_reason) for t in r.trades()]

        assert key(r1) == key(r2.result) == key(r3.result)
        assert a.reasons() == b.reasons() == c.reasons()


def result_timestamps(data):
    return data["timestamps"]


# ---------------------------------------------------------------------------
# DAY orders
# ---------------------------------------------------------------------------
class TestDayOrders:
    def _run(self, schedule, data):
        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == 2:
                    self.submit_order(orders.Limit(side="buy", price=50.0, units=1.0, tif="day"))

        s = S()
        run(s, data, spec(schedule))
        return s, data

    def test_a_day_order_expires_at_the_session_close(self):
        # A bar after the close, same IST date: the order lapses there.
        data = bars((THU, (15, 0), (15, 30)))
        post = bars((THU, (15, 40), (15, 45)))
        merged = {k: np.concatenate([data[k], post[k]]) for k in data}
        s, merged = self._run(nse_with_squareoff(squareoff_minute=None), merged)
        expired = [i for k, i, _ in s.events if k == "order_expired"]
        assert expired == [len(data["timestamps"])]

    def test_the_last_bar_of_the_session_is_still_inside_it(self):
        data = bars((THU, (15, 0), (15, 30)))
        s, _ = self._run(nse_with_squareoff(squareoff_minute=None), data)
        assert "order_expired" not in s.kinds()

    def test_without_a_schedule_it_expires_at_the_local_date_rollover(self):
        data = bars((THU, (15, 0), (15, 30)))
        post = bars((THU, (15, 40), (15, 45)))
        merged = {k: np.concatenate([data[k], post[k]]) for k in data}
        s, _ = self._run(None, merged)
        assert "order_expired" not in s.kinds()


# ---------------------------------------------------------------------------
# Daily composite bars
# ---------------------------------------------------------------------------
with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class DailyBars(Strategy):
        def __init__(self):
            super().__init__(None)
            self.daily = []

        def on_start(self, ctx):
            self.subscribe_bars(1, "d")

        def on_composite_bar(self, ctx, bar):
            self.daily.append(bar)


class TestDailyBars:
    def _data(self):
        # Thu, Fri (a holiday that still has prints), Sat and Mon, then Tue so
        # Monday's bar completes.
        return bars(regular_day(THU), regular_day(FRI), regular_day(SAT), regular_day(MON),
                    regular_day(TUE), step=30)

    def _run(self, schedule):
        s = DailyBars()
        run(s, self._data(), spec(schedule))
        return s.daily

    def test_holidays_and_weekends_make_no_bar(self):
        daily = self._run(nse_with_squareoff(holidays=[FRI]))
        # Bars close at the following local midnight; only Thursday and Monday trade.
        closes = [b.timestamp for b in daily]
        assert closes == [ist(FRI, 0, 0), ist(TUE, 0, 0)]

    def test_without_a_schedule_every_date_with_data_makes_a_bar(self):
        daily = self._run(None)
        assert len(daily) == 4  # Thu, Fri, Sat, Mon (Tue is still forming)

    def test_the_schedule_timezone_sets_the_day_cut(self):
        # Even with a UTC tz_offset_ns in the config, bars cut at IST midnight.
        s = DailyBars()
        config = _zero_fee_config()
        run(s, self._data(), spec(nse_with_squareoff()), config=config)
        assert all((b.timestamp + IST_NS) % NS_DAY == 0 for b in s.daily)


# ---------------------------------------------------------------------------
# One answer in every runner
# ---------------------------------------------------------------------------
class Scripted(Log):
    """Open-hours buy, a closed-hours buy, a resting day order, never exits."""

    def on_bar(self, ctx):
        n = ctx.idx
        if n == 2:
            self.submit_order(orders.Market(side="buy", units=5.0))
        if n == 3:
            self.submit_order(orders.Limit(side="buy", price=50.0, units=1.0, tif="day"))


class TestRunners:
    def _data(self):
        pre = bars((THU, (7, 55), (9, 10)))
        main = bars(regular_day(THU), regular_day(FRI))
        return {k: np.concatenate([pre[k], main[k]]) for k in pre}

    def _schedule(self):
        return nse_with_squareoff()

    def test_closed_hours_entry_is_rejected_the_same_way_in_every_runner(self):
        # Bar 2 and 3 fall before the open in this data, so both orders are
        # refused everywhere; later trading is untouched.
        data = self._data()
        single = Scripted()
        r1 = run(single, data, spec(self._schedule()))

        pf = Scripted()
        r2 = run_portfolio_strategy(
            pf, {"ASSET": data}, config=_zero_fee_config(), bar_stamp="close",
            instruments={"ASSET": spec(self._schedule())},
        )

        many = Scripted()
        (r3,) = run_strategies(
            [(many, ["ASSET"])], {"ASSET": data}, config=_zero_fee_config(), bar_stamp="close",
            instruments={"ASSET": spec(self._schedule())},
        )

        assert single.events == pf.events == many.events
        assert single.reasons() == ["market_closed", "market_closed"]
        assert len(r1.trades()) == len(r2.result.trades()) == len(r3.result.trades()) == 0

    def test_square_off_agrees_across_runners(self):
        data = bars(regular_day(THU), regular_day(FRI))

        class Long(Log):
            def on_bar(self, ctx):
                if ctx.idx in (4, 80):
                    self.submit_order(orders.Market(side="buy", units=10.0))

        schedule = self._schedule()
        r1 = run(Long(), data, spec(schedule))
        r2 = run_portfolio_strategy(
            Long(), {"ASSET": data}, config=_zero_fee_config(), bar_stamp="close",
            instruments={"ASSET": spec(schedule)},
        )
        (r3,) = run_strategies(
            [(Long(), ["ASSET"])], {"ASSET": data}, config=_zero_fee_config(), bar_stamp="close",
            instruments={"ASSET": spec(schedule)},
        )

        def key(result):
            return [(t.entry_idx, t.exit_idx, t.entry_price, t.exit_price, t.size, t.exit_reason)
                    for t in result.trades()]

        assert key(r1) == key(r2.result) == key(r3.result)
        assert [t.exit_reason for t in r1.trades()] == ["Squareoff", "Squareoff"]

    def test_daily_bars_agree_across_runners(self):
        data = bars(regular_day(THU), regular_day(FRI), regular_day(SAT), regular_day(MON),
                    regular_day(TUE), step=30)
        schedule = nse_with_squareoff(holidays=[FRI])

        a = DailyBars()
        run(a, data, spec(schedule))
        b = DailyBars()
        run_portfolio_strategy(
            b, {"ASSET": data}, config=_zero_fee_config(), bar_stamp="close",
            instruments={"ASSET": spec(schedule)},
        )
        c = DailyBars()
        run_strategies(
            [(c, ["ASSET"])], {"ASSET": data}, config=_zero_fee_config(), bar_stamp="close",
            instruments={"ASSET": spec(schedule)},
        )

        def key(s):
            return [(x.timestamp, x.open, x.high, x.low, x.close, x.volume) for x in s.daily]

        assert key(a) == key(b) == key(c)
        assert len(a.daily) == 2

    def test_a_schedule_per_symbol_only_binds_that_symbol(self):
        data = bars((THU, (7, 55), (9, 10)))

        class Both(Log):
            def on_bar(self, ctx):
                if ctx.idx == 2:
                    self.submit_order(orders.Market(side="buy", units=1.0))

        s = Both()
        result = run_portfolio_strategy(
            s, {"A": data, "B": data}, config=_zero_fee_config(), bar_stamp="close",
            instruments={"A": spec(self._schedule(), symbol="A"), "B": spec(symbol="B")},
        )
        # The order placed while stepping A is refused; the one on B trades.
        assert s.reasons() == ["market_closed"]
        assert len(result.result.trades()) == 1


# ---------------------------------------------------------------------------
# Ticks and the live node
# ---------------------------------------------------------------------------
class TickBuyer(Log):
    """Buys on every print until it holds something."""

    def on_trade(self, ctx, tick):
        if not self.kinds().count("order_filled"):
            self.submit_order(orders.Market(side="buy", units=1.0))


class TestLiveNode:
    def _node(self, schedule):
        strategy = TickBuyer()
        node = raptorbt.LiveNode(
            [(strategy, ["ASSET"])],
            ["ASSET"],
            _zero_fee_config(),
            instruments={"ASSET": spec(schedule)},
            bar_stamp="close",
        )
        return node, strategy

    def test_prints_outside_the_session_are_refused_and_inside_it_trade(self):
        node, strategy = self._node(nse_with_squareoff())
        node.push_tick("ASSET", ist(THU, 8, 0), 100.0, ltq=1.0)
        node.push_tick("ASSET", ist(THU, 8, 1), 100.0, ltq=1.0)
        assert strategy.reasons() == ["market_closed"] * 2
        assert "order_filled" not in strategy.kinds()
        node.push_tick("ASSET", ist(THU, 10, 0), 100.0, ltq=1.0)
        node.push_tick("ASSET", ist(THU, 10, 1), 101.0, ltq=1.0)
        assert "order_filled" in strategy.kinds()
        (result,) = node.finish()
        assert len(result.result.trades()) == 1

    def test_square_off_closes_the_position_on_a_print_at_the_cutoff(self):
        node, strategy = self._node(nse_with_squareoff())
        node.push_tick("ASSET", ist(THU, 10, 0), 100.0, ltq=1.0)
        node.push_tick("ASSET", ist(THU, 10, 1), 100.0, ltq=1.0)
        node.push_tick("ASSET", ist(THU, 15, 20), 102.0, ltq=1.0)
        (result,) = node.finish()
        trades = result.result.trades()
        assert [t.exit_reason for t in trades] == ["Squareoff"]
        assert trades[0].exit_price == pytest.approx(102.0)

    def test_without_a_schedule_the_same_prints_trade_at_any_hour(self):
        node, strategy = self._node(None)
        node.push_tick("ASSET", ist(THU, 8, 0), 100.0, ltq=1.0)
        node.push_tick("ASSET", ist(THU, 8, 1), 100.0, ltq=1.0)
        assert strategy.reasons() == []
        assert "order_filled" in strategy.kinds()


# ---------------------------------------------------------------------------
# Next-bar-open fills
# ---------------------------------------------------------------------------
class TestNextBarOpen:
    def _config(self):
        return BacktestConfig(fees=0.0, fill_timing="next_bar_open")

    def test_an_entry_decided_at_the_close_is_refused_not_parked_for_the_open(self):
        data = bars((THU, (15, 10), (15, 30)), regular_day(FRI))

        class S(Log):
            def on_bar(self, ctx):
                if ctx.idx == 3:  # the bar closing at 15:30
                    self.enter()

        s = S()
        schedule = nse_with_squareoff(squareoff_minute=None)
        result = run(s, data, spec(schedule), config=self._config())
        assert s.reasons() == ["market_closed"]
        assert len(result.trades()) == 0

    def test_an_entry_decided_mid_session_fills_on_the_next_bar(self):
        data = bars(regular_day(THU))

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 3:
                    self.enter()

        schedule = nse_with_squareoff(squareoff_minute=None)
        result = run(S(), data, spec(schedule), config=self._config())
        assert [t.entry_idx for t in result.trades()] == [4]


# ---------------------------------------------------------------------------
# Tick runners
# ---------------------------------------------------------------------------
class TestTickRunners:
    STAMPS = [ist(THU, 8, 0), ist(THU, 8, 1), ist(THU, 10, 0), ist(THU, 10, 1), ist(THU, 15, 20)]
    PRICES = [100.0, 100.0, 100.0, 101.0, 102.0]

    def _ticks(self):
        return {
            "ASSET": {
                "timestamps": np.array(self.STAMPS, dtype=np.int64),
                "ltp": np.array(self.PRICES),
            }
        }

    def _summary(self, strategy, result):
        return (
            strategy.reasons(),
            [(t.exit_reason, t.exit_price) for t in result.result.trades()],
        )

    def test_run_tick_strategy_and_the_stream_agree_with_the_live_node(self):
        schedule = nse_with_squareoff()
        instruments = {"ASSET": spec(schedule)}

        batch = TickBuyer()
        r_batch = raptorbt.run_tick_strategy(
            batch, self._ticks(), config=_zero_fee_config(), instruments=instruments,
        )

        stream = TickBuyer()
        node = raptorbt.TickStrategyStream(
            stream, ["ASSET"], config=_zero_fee_config(), instruments=instruments,
        )
        for ts, price in zip(self.STAMPS, self.PRICES):
            node.push_tick("ASSET", ts, price)
        r_stream = node.finish()

        assert self._summary(batch, r_batch) == self._summary(stream, r_stream)
        reasons, trades = self._summary(batch, r_batch)
        assert reasons == ["market_closed", "market_closed"]
        assert trades == [("Squareoff", 102.0)]

    def test_daily_bars_follow_the_schedule_in_the_tick_runner(self):
        stamps = [ist(d, h, m) for d in (THU, FRI, SAT, MON, TUE) for h, m in ((10, 0), (11, 0))]
        ticks = {"ASSET": {"timestamps": np.array(stamps, dtype=np.int64),
                           "ltp": np.full(len(stamps), 100.0)}}
        seen = {}
        for name, schedule in (("with", nse_with_squareoff(holidays=[FRI])),
                               ("without", None)):
            s = DailyBars()
            raptorbt.run_tick_strategy(
                s, ticks, config=_zero_fee_config(), instruments={"ASSET": spec(schedule)},
            )
            seen[name] = [b.timestamp for b in s.daily]
        assert seen["with"] == [ist(FRI, 0, 0), ist(TUE, 0, 0)]
        assert len(seen["without"]) == 4
