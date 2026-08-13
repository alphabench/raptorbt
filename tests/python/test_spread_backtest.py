"""Behavior tests for the spread backtest surface, against the built wheel.

``run_spread_backtest`` shipped a sign error through 0.6.3: it returned a
``pnl`` that was the negative of the truth, so a multi-leg options structure
that made money was reported as a loss. Worse, the ``max_loss`` and
``target_profit`` thresholds compare against the same figure, so a stop closed
positions that had GAINED and a target booked wins on positions that had LOST.

The defect survived because nothing tested it. The Rust unit tests asserted
only that a trade was recorded, and this Python suite -- the one CI runs
against the real wheel -- did not call ``run_spread_backtest`` at all. These
tests close that gap at the boundary the defect actually crossed.

Fees are left at the default, so P&L assertions carry a tolerance; the sign
and the magnitude-to-the-rupee are what matter, not the last few paise.
"""

import numpy as np
import pytest

import raptorbt as r

LOT = 75
STRIKE = 24_800.0
#: Entry is bar 1 and exit is bar 4, so the premium moves thirty points across
#: a 75 lot. Every correct answer below is 2250, plus or minus costs.
EXPECTED = 30.0 * LOT

#: Premium falls: entered at 90, closed at 60.
FALLING = [100.0, 90.0, 80.0, 70.0, 60.0, 60.0]
#: The mirror: entered at 70, closed at 100.
RISING = [60.0, 70.0, 80.0, 90.0, 100.0, 100.0]


def _run(premiums, quantity, *, max_loss=None, target_profit=None, exit_on_bar_4=True):
    """One CE leg, lot 75, entered on bar 1 and exited on bar 4.

    Dropping the exit signal leaves a threshold as the only way out, which is
    how the trigger tests make ``exit_reason`` the assertion.
    """
    n = len(premiums)
    timestamps = (
        np.arange(n, dtype=np.int64) * 300_000_000_000 + 1_786_000_000_000_000_000
    )
    entries = np.zeros(n, dtype=bool)
    entries[1] = True
    exits = np.zeros(n, dtype=bool)
    if exit_on_bar_4:
        exits[4] = True

    kwargs = {}
    if max_loss is not None:
        kwargs["max_loss"] = max_loss
    if target_profit is not None:
        kwargs["target_profit"] = target_profit

    return r.run_spread_backtest(
        timestamps=timestamps,
        underlying_close=np.full(n, 24_550.0),
        legs_premiums=[np.array(premiums, dtype=np.float64)],
        leg_configs=[("CE", STRIKE, quantity, LOT)],
        entries=entries,
        exits=exits,
        config=r.BacktestConfig(initial_capital=500_000.0),
        spread_type="custom",
        **kwargs,
    )


@pytest.mark.parametrize(
    "premiums,quantity,label",
    [
        (FALLING, -1, "short leg that gained"),
        (RISING, 1, "long leg that gained"),
    ],
)
def test_a_winning_spread_reports_a_profit(premiums, quantity, label):
    """A structure that made money must not be reported as a loss.

    If this regresses, a user backtests a credit spread that worked, sees a
    negative return and a negative Sharpe, and discards a profitable strategy.
    """
    trade = _run(premiums, quantity).trades()[0]

    assert trade.pnl > 0, f"{label}: gained {EXPECTED} but reported {trade.pnl}"
    assert trade.pnl == pytest.approx(EXPECTED, abs=20.0)


@pytest.mark.parametrize(
    "premiums,quantity,label",
    [
        (RISING, -1, "short leg that lost"),
        (FALLING, 1, "long leg that lost"),
    ],
)
def test_a_losing_spread_reports_a_loss(premiums, quantity, label):
    """The mirror, and the more dangerous half.

    A losing structure reported as profitable is the one that gets deployed.
    """
    trade = _run(premiums, quantity).trades()[0]

    assert trade.pnl < 0, f"{label}: lost {EXPECTED} but reported {trade.pnl}"
    assert trade.pnl == pytest.approx(-EXPECTED, abs=20.0)


def test_pnl_agrees_with_the_price_difference():
    """``pnl`` and ``exit_price - entry_price`` must not disagree in sign.

    They are computed independently, and through 0.6.3 they disagreed: the
    trade record carried the right answer and the wrong one side by side.
    That is what made the bug reconcilable by hand, and it is the cheapest
    invariant to check.
    """
    trade = _run(FALLING, -1).trades()[0]

    assert trade.exit_price - trade.entry_price == pytest.approx(EXPECTED)
    assert trade.pnl > 0
    # And the leg is sized off its stated lot: -90 * 75 is the entry credit.
    assert trade.entry_price == pytest.approx(-90.0 * LOT)


def test_max_loss_does_not_close_a_winner():
    """A stop must not fire on a position that is up.

    Through 0.6.3 it did: the threshold read the negated P&L as a 2250 loss
    and closed a leg that had gained 2250. With no exit signal and no trigger
    reached, the position runs to the end of the data.

    Asserted as ``EndOfData`` rather than an empty trade book: before 0.7.2 the
    terminal open position was settled into cash without being recorded, so
    "the stop did not fire" and "the position was never reported" looked
    identical here. They are different things, and only one of them is correct.
    """
    trades = _run(FALLING, -1, max_loss=1000.0, exit_on_bar_4=False).trades()

    assert len(trades) == 1
    assert trades[0].exit_reason == "EndOfData"
    assert trades[0].pnl > 0


def test_max_loss_still_closes_a_real_loser():
    """The stop must keep working for the case it exists for."""
    trades = _run(RISING, -1, max_loss=1000.0, exit_on_bar_4=False).trades()

    assert len(trades) == 1
    assert trades[0].pnl < 0


def test_target_profit_does_not_close_a_loser():
    """A target must not book a win on a position that is down.

    ``EndOfData`` for the same reason as the max-loss case above.
    """
    trades = _run(RISING, -1, target_profit=1000.0, exit_on_bar_4=False).trades()

    assert len(trades) == 1
    assert trades[0].exit_reason == "EndOfData"
    assert trades[0].pnl < 0


def test_target_profit_still_closes_a_real_winner():
    """The target must keep working for the case it exists for."""
    trades = _run(FALLING, -1, target_profit=1000.0, exit_on_bar_4=False).trades()

    assert len(trades) == 1
    assert trades[0].pnl > 0


# ---------------------------------------------------------------------------
# Session squareoff
#
# In plain words: an intraday backtest must close its positions before the
# market shuts, because a real broker would have. Through 0.7.1 the engine had
# no way to be told that, so a position opened one morning stayed open for
# days and the overnight price jumps were booked as profit or loss the trader
# could never have experienced.
#
# The backend passed `session_aware=True` for months. It reached a `hasattr`
# guard for a `set_session_config` method that existed only in the type stub,
# never in the engine, so the call was silently dropped. These tests exist so
# that cannot recur: the setting is real, it is reachable from Python, and bad
# input refuses instead of quietly doing nothing.
# ---------------------------------------------------------------------------

#: IST is UTC+5:30, in nanoseconds.
IST_NS = (5 * 3600 + 30 * 60) * 1_000_000_000
#: NSE squares off intraday positions five minutes before its 15:30 close.
NSE_SQUAREOFF = "15:25"


def _two_session_run(squareoff_time):
    """Two sessions, entered each morning, with a gap between them.

    The premium drifts down five points within each session (a gain for the
    short leg) and then jumps twenty points overnight (a large loss). The two
    are deliberately far apart in size so a squareoff that fails cannot be
    mistaken for one that works.
    """
    day_ns = 86_400 * 1_000_000_000

    def ist(day, hour, minute):
        """UTC nanoseconds for an IST wall-clock time."""
        return day * day_ns + (hour * 3600 + minute * 60) * 1_000_000_000 - IST_NS

    timestamps = np.array(
        [ist(0, 9, 15), ist(0, 15, 29), ist(1, 9, 15), ist(1, 15, 29)],
        dtype=np.int64,
    )
    premiums = np.array([100.0, 95.0, 115.0, 110.0], dtype=np.float64)

    config_kwargs = {"initial_capital": 500_000.0, "fees": 0.0, "slippage": 0.0}
    if squareoff_time is not None:
        config_kwargs["squareoff_time"] = squareoff_time
        # The squareoff time is a LOCAL time, so the offset defining "local"
        # must be set with it. At the 0 (UTC) default, 15:29 IST reads as
        # 09:59 and the squareoff never fires.
        config_kwargs["session_tz_offset_ns"] = IST_NS

    return r.run_spread_backtest(
        timestamps=timestamps,
        underlying_close=np.full(4, 24_550.0),
        legs_premiums=[premiums],
        leg_configs=[("CE", STRIKE, -1, LOT)],
        entries=np.array([True, False, True, False]),
        exits=np.zeros(4, dtype=bool),
        config=r.BacktestConfig(**config_kwargs),
        spread_type="custom",
    )


def test_without_squareoff_a_position_rides_through_the_overnight_gap():
    """The defect, pinned as behaviour so a fix cannot silently regress it."""
    trades = _two_session_run(None).trades()

    # One position, opened on the first morning and never closed until the
    # data ran out -- straight through the night.
    assert len(trades) == 1
    assert trades[0].exit_reason == "EndOfData"
    # Sold at 100, marked at 110: the overnight jump is booked as a loss.
    assert trades[0].pnl == pytest.approx(-10.0 * LOT)


def test_squareoff_closes_each_session_before_the_gap():
    """With squareoff on, each day is its own trade and the gap never lands."""
    trades = _two_session_run(NSE_SQUAREOFF).trades()

    assert len(trades) == 2
    assert [t.exit_reason for t in trades] == ["Squareoff", "Squareoff"]
    # Each captured only its own session's five-point decay.
    assert trades[0].pnl == pytest.approx(5.0 * LOT)
    assert trades[1].pnl == pytest.approx(5.0 * LOT)


def test_squareoff_changes_the_reported_result():
    """The headline: this is not a cosmetic difference.

    A short option that looks like a loser without squareoff is a winner with
    it, because the loss came entirely from hours the trader was not exposed.
    """
    without = _two_session_run(None).metrics
    with_ = _two_session_run(NSE_SQUAREOFF).metrics

    assert without.total_return_pct < 0 < with_.total_return_pct


@pytest.mark.parametrize("bad", ["1525", "25:00", "15:99", "abc", "", "15:", ":25"])
def test_an_unreadable_squareoff_time_refuses(bad):
    """Refusing loudly is the point.

    A squareoff that silently does nothing is precisely the defect being
    fixed. Never add a fallback that guesses.
    """
    with pytest.raises(ValueError, match="invalid squareoff_time"):
        r.BacktestConfig(squareoff_time=bad)


def test_squareoff_time_parses_to_minutes_from_midnight():
    """And ``None`` leaves it off, which stays the default."""
    assert r.BacktestConfig(squareoff_time=NSE_SQUAREOFF).squareoff_time_minutes == 925
    assert r.BacktestConfig(squareoff_time=None).squareoff_time_minutes is None
    assert r.BacktestConfig().squareoff_time_minutes is None


def test_batch_spread_backtest_honours_squareoff():
    """The batch path is what production calls, so it gets its own pin.

    ``batch_spread_backtest`` builds each item's config by cloning the shared
    base config. That is why squareoff reaches it -- but "why" is an argument,
    and this is the measurement. The pipeline that ran option spreads with no
    squareoff for months went through this function.
    """
    day_ns = 86_400 * 1_000_000_000

    def ist(day, hour, minute):
        return day * day_ns + (hour * 3600 + minute * 60) * 1_000_000_000 - IST_NS

    timestamps = np.array(
        [ist(0, 9, 15), ist(0, 15, 29), ist(1, 9, 15), ist(1, 15, 29)],
        dtype=np.int64,
    )
    item = r.BatchSpreadItem(
        strategy_id="s1",
        legs_premiums=[np.array([100.0, 95.0, 115.0, 110.0], dtype=np.float64)],
        leg_configs=[("CE", STRIKE, -1, LOT)],
        entries=np.array([True, False, True, False]),
        exits=np.zeros(4, dtype=bool),
        spread_type="custom",
    )

    def run(**extra):
        results = r.batch_spread_backtest(
            timestamps,
            np.full(4, 24_550.0),
            [item],
            config=r.BacktestConfig(
                initial_capital=500_000.0, fees=0.0, slippage=0.0, **extra
            ),
        )
        return dict(results)["s1"]

    without = run()
    with_ = run(squareoff_time=NSE_SQUAREOFF, session_tz_offset_ns=IST_NS)

    assert [t.exit_reason for t in without.trades()] == ["EndOfData"]
    assert [t.exit_reason for t in with_.trades()] == ["Squareoff", "Squareoff"]
    # The overnight gap turns a winning pair of day trades into a loser.
    assert without.metrics.total_return_pct < 0 < with_.metrics.total_return_pct
