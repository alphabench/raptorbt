"""MAE and MFE, exercised through the wheel.

In plain words: how far a trade went against you before it worked, and how
far in your favour before it failed. Without these two numbers a losing
strategy cannot be told apart from a winning one that was stopped too early
-- both report the same realised loss, and only the excursion says which it
was.

They are MEASURED, bar by bar, while the position is open -- not inferred
afterwards from an OHLC window. That distinction is the reason they live in
the engine: reconstructing them later cannot know whether the high or the
low came first within a bar, so a reconstructed "the stop was hit before the
target" is a coin flip. What the engine reports is bounded by the bar's own
resolution -- a 5m run knows the worst 5m extreme, not the worst tick inside
it -- and no finer claim is made anywhere.

`None` means the path never tracked extremes (a synthesised spread or basket
leg), never a zero excursion.
"""

import numpy as np

from raptorbt import Strategy, run_strategy_backtest

SEC_NS = 1_000_000_000


class HoldTenBars(Strategy):
    """Enter on the first bar, hold ten bars, exit. One trade."""

    def on_bar(self, ctx):
        if ctx.position is None and ctx.idx == 0:
            self.enter()
        elif ctx.position is not None and ctx.idx >= 10:
            self.close_position()


def _bars(open_, high, low, close):
    n = len(close)
    return {
        "timestamps": np.arange(n, dtype=np.int64) * SEC_NS,
        "open": np.asarray(open_, dtype=np.float64),
        "high": np.asarray(high, dtype=np.float64),
        "low": np.asarray(low, dtype=np.float64),
        "close": np.asarray(close, dtype=np.float64),
        "volume": np.full(n, 1_000_000.0),
    }


def _dip_then_rally(n=12):
    """A path that dips to 90 and peaks at 130 before closing at 120."""
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    close[3], low[3] = 90.0, 90.0      # the dip
    close[7], high[7] = 130.0, 130.0   # the peak
    close[8:] = 120.0
    high[8:] = 120.0
    low[8:] = 120.0
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    return _bars(close, high, low, close)


def test_a_long_reports_the_dip_as_mae_and_the_peak_as_mfe():
    result = run_strategy_backtest(HoldTenBars, **_dip_then_rally())
    trades = result.trades()
    assert trades, "no trades produced"
    trade = trades[0]

    assert trade.mae_pnl is not None, "a tracked position must measure its excursions"
    assert trade.mfe_pnl is not None

    # Sign is the whole point: adverse is never positive, favourable never
    # negative. Getting these backwards would report a winning trade as a
    # losing one.
    assert trade.mae_pnl <= 0.0, f"MAE must not be positive: {trade.mae_pnl}"
    assert trade.mfe_pnl >= 0.0, f"MFE must not be negative: {trade.mfe_pnl}"

    # The prices are exact: the run saw a 90 low and a 130 high while open.
    assert abs(trade.mae_price - 90.0) < 1e-9, trade.mae_price
    assert abs(trade.mfe_price - 130.0) < 1e-9, trade.mfe_price

    # And the money follows the size the engine actually took.
    assert abs(trade.mae_pnl - (90.0 - trade.entry_price) * trade.size) < 1e-6
    assert abs(trade.mfe_pnl - (130.0 - trade.entry_price) * trade.size) < 1e-6


def test_the_excursions_bracket_the_realised_pnl():
    """The property that holds for every trade on every path.

    Gross P&L is realised somewhere between the worst and best the trade ever
    showed. A violation means the extremes were not tracked over the whole
    holding period -- the failure this pins.
    """
    result = run_strategy_backtest(HoldTenBars, **_dip_then_rally())
    for trade in result.trades():
        if trade.mae_pnl is None:
            continue
        gross = trade.pnl + trade.fees
        assert trade.mae_pnl - 1e-6 <= gross <= trade.mfe_pnl + 1e-6, (
            f"realised {gross} outside excursion band "
            f"[{trade.mae_pnl}, {trade.mfe_pnl}]"
        )


def test_a_flat_trade_measures_zero_rather_than_reporting_nothing():
    """A position that never moved has a zero excursion, which is a
    measurement. Only an untracked path reports None."""
    n = 12
    flat = np.full(n, 100.0)
    result = run_strategy_backtest(HoldTenBars, **_bars(flat, flat, flat, flat))
    trades = result.trades()
    assert trades, "no trades produced"
    trade = trades[0]

    assert trade.mae_pnl == 0.0
    assert trade.mfe_pnl == 0.0
