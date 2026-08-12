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
    reached, the position runs to the end of the data and records no trade.
    """
    result = _run(FALLING, -1, max_loss=1000.0, exit_on_bar_4=False)

    assert len(result.trades()) == 0


def test_max_loss_still_closes_a_real_loser():
    """The stop must keep working for the case it exists for."""
    trades = _run(RISING, -1, max_loss=1000.0, exit_on_bar_4=False).trades()

    assert len(trades) == 1
    assert trades[0].pnl < 0


def test_target_profit_does_not_close_a_loser():
    """A target must not book a win on a position that is down."""
    result = _run(RISING, -1, target_profit=1000.0, exit_on_bar_4=False)

    assert len(result.trades()) == 0


def test_target_profit_still_closes_a_real_winner():
    """The target must keep working for the case it exists for."""
    trades = _run(FALLING, -1, target_profit=1000.0, exit_on_bar_4=False).trades()

    assert len(trades) == 1
    assert trades[0].pnl > 0
