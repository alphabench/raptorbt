"""``BacktestConfig.with_close_open_at_end``: a position still open at the last bar is traded
out as a charged exit, so a run that is one window of a longer history pays for its last
position.

By default the run marks it to market with no exit costs (``EndOfData``, the position is still
open). With the option the exit is a real one (``WindowEnd``): exit fees are paid, the trade
is counted, and the run's return is the sum of its trades' net P&L.
"""

import numpy as np

from raptorbt import BacktestConfig, Strategy, run_strategy_backtest

DAY_NS = 86_400_000_000_000
CAPITAL = 100_000.0
FEES = 0.001


class HoldToEnd(Strategy):
    """Enter once, never exit: the run ends holding."""

    def on_bar(self, ctx):
        if ctx.idx == 1 and ctx.position is None:
            self.enter()


def _bars(n=30):
    close = np.linspace(100.0, 130.0, n)
    openp = np.concatenate([[close[0]], close[:-1]])
    return {
        "timestamps": np.arange(n, dtype=np.int64) * DAY_NS,
        "open": openp,
        "high": np.maximum(openp, close) * 1.002,
        "low": np.minimum(openp, close) * 0.998,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }


def _run(config):
    return run_strategy_backtest(HoldToEnd, **_bars(), config=config, bar_stamp="close")


def test_the_default_marks_the_open_position_with_no_exit_cost():
    result = _run(BacktestConfig(initial_capital=CAPITAL, fees=FEES))
    (trade,) = result.trades()
    assert trade.exit_reason == "EndOfData"
    assert trade.exit_fees == 0.0


def test_the_option_trades_the_position_out_at_the_last_close_and_charges_the_exit():
    base = BacktestConfig(initial_capital=CAPITAL, fees=FEES)
    marked = _run(base)
    charged = _run(base.with_close_open_at_end())
    (trade,) = charged.trades()
    (open_trade,) = marked.trades()

    assert trade.exit_reason == "WindowEnd"
    assert trade.exit_price == _bars()["close"][-1]
    assert trade.exit_fees > 0.0
    assert abs(trade.exit_fees - trade.exit_price * trade.size * FEES) < 0.01
    # the same position, the same gross move: the charged run is worse by exactly the exit fee
    assert abs((open_trade.pnl - trade.pnl) - trade.exit_fees) < 0.01
    assert charged.metrics.total_return_pct < marked.metrics.total_return_pct
    # the run's return is its trades' net P&L over the capital
    assert abs(charged.metrics.total_return_pct - trade.pnl / CAPITAL * 100.0) < 0.01
    # the charged exit is a closed trade; the marked position is still open
    assert charged.metrics.total_closed_trades == 1
    assert charged.metrics.total_open_trades == 0
    assert marked.metrics.total_open_trades == 1


def test_the_option_can_be_switched_off_again():
    cfg = BacktestConfig(initial_capital=CAPITAL, fees=FEES).with_close_open_at_end()
    assert cfg.close_open_at_end is True
    assert cfg.with_close_open_at_end(False).close_open_at_end is False


def test_slippage_moves_the_charged_exit_against_a_long():
    config = BacktestConfig(initial_capital=CAPITAL, fees=FEES, slippage=0.001)
    (trade,) = _run(config.with_close_open_at_end()).trades()
    assert trade.exit_reason == "WindowEnd"
    assert trade.exit_price < _bars()["close"][-1]
