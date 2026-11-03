"""1-minute entries confirmed by the 15-minute trend.

Shows composite bars. ``subscribe_bars(15, "m")`` asks the engine to build
15-minute bars from the 1-minute stream; each completed one reaches
``on_bar(self, ctx, event)`` with ``event.timeframe == "15m"``, and the primary
1-minute bar then arrives with ``event.timeframe is None``. A composite bar is
delivered one primary bar after its window ends, so the strategy never acts on
a bar that is still forming (no lookahead).

The strategy keeps the last 15-minute closes as the trend, and buys when the
1-minute close crosses above its 10-bar SMA while the 15-minute trend is up
(the last 15-minute close is above the average of the last four). It closes when
the 1-minute close falls back under the SMA. For per-instrument control in a
many-symbol run use ``subscribe(symbol, timeframe="15m")`` instead.

Skills: raptorbt-subscriptions, raptorbt-timeframes, raptorbt-indicators,
raptorbt-strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Indicator, Strategy, run_strategy_backtest  # noqa: E402

TREND_BARS = 4  # how many 15-minute closes make the trend


class TrendConfirmed(Strategy):
    def on_start(self, ctx):
        self.sma = self.register_indicator(Indicator.sma(10))
        self.bars15 = self.subscribe_bars(15, "m")
        self.closes15 = []
        self.prev_below = False
        self.composites = 0

    def trend_is_up(self):
        recent = self.closes15[-TREND_BARS:]
        return len(recent) == TREND_BARS and recent[-1] > sum(recent) / TREND_BARS

    def on_bar(self, ctx, event):
        if event.timeframe is not None:        # a completed 15-minute bar
            self.closes15.append(event.close)
            self.composites += 1
            return
        if not self.indicators_initialized():  # a 1-minute bar
            return
        below = ctx.bar.close < self.sma.value
        crossed_up = self.prev_below and not below
        self.prev_below = below
        if ctx.is_flat and crossed_up and self.trend_is_up():
            self.enter(size_frac=0.5)
        elif not ctx.is_flat and below:
            self.close_position()


def main():
    data = _data.bars(days=3, seed=21, start=250.0, step=0.15, drift_period=240)
    strategy = TrendConfirmed()
    result = run_strategy_backtest(
        strategy, **data, symbol="DEMO",
        config=BacktestConfig(initial_capital=100_000.0, fees=0.0003),
        bar_stamp="close",
    )
    m = result.metrics
    print(f"15m bars delivered {strategy.composites}  trades {m.total_trades}  "
          f"return {m.total_return_pct:.2f}%  win rate {m.win_rate_pct:.0f}%")
    assert strategy.composites == len(data["timestamps"]) // 15 - 1  # the last window is still open when the data ends
    assert m.total_trades > 0


if __name__ == "__main__":
    main()
