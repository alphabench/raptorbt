"""An SMA cross on 1-minute bars, with fees, a stop and a target.

Shows the shape of every class-based strategy: a frozen ``StrategyConfig`` of
bounded parameters, streaming indicators registered in ``on_start``, one entry
or close intent per bar in ``on_bar``, and the result read back as metrics and
trades. The fast SMA crossing above the slow one buys; crossing below closes.
A 1 percent stop and a 2 percent target are attached to the config, and the
fee is 0.03 percent of traded value on each side.

Skills: raptorbt-core, raptorbt-strategy, raptorbt-indicators, raptorbt-orders,
raptorbt-results.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Indicator, Strategy, StrategyConfig, run_strategy_backtest  # noqa: E402
from raptorbt.strategy import param  # noqa: E402


@dataclass(frozen=True)
class CrossConfig(StrategyConfig):
    fast: int = param(5, min=2, max=50, description="fast SMA period, in bars")
    slow: int = param(20, min=5, max=200, description="slow SMA period, in bars")
    size: float = param(0.5, min=0.05, max=1.0, description="fraction of capital per entry")


class SmaCross(Strategy):
    def __init__(self, config=None):
        super().__init__(config or CrossConfig())

    def on_start(self, ctx):
        self.fast = self.register_indicator(Indicator.sma(self.config.fast))
        self.slow = self.register_indicator(Indicator.sma(self.config.slow))

    def on_bar(self, ctx):
        if not self.indicators_initialized():
            return  # still warming up
        if ctx.is_flat and self.fast.value > self.slow.value:
            self.enter(size_frac=self.config.size)
        elif not ctx.is_flat and self.fast.value < self.slow.value:
            self.close_position()


def main():
    data = _data.bars(days=2, seed=7, start=100.0)

    config = BacktestConfig(initial_capital=100_000.0, fees=0.0003, fill_timing="next_bar_open")
    config.set_fixed_stop(0.01)    # exit if price falls 1 percent below the entry
    config.set_fixed_target(0.02)  # or rises 2 percent above it

    result = run_strategy_backtest(
        SmaCross(), **data, symbol="DEMO", config=config, bar_stamp="close"
    )

    m = result.metrics
    print(f"trades {m.total_trades}  win rate {m.win_rate_pct:.0f}%  return {m.total_return_pct:.2f}%  "
          f"fees paid {m.total_fees_paid:.2f}  max drawdown {m.max_drawdown_pct:.2f}%")
    for t in result.trades()[:3]:
        print(f"  bar {t.entry_idx} -> {t.exit_idx}  {t.entry_price:.2f} -> {t.exit_price:.2f}  "
              f"pnl {t.pnl:+.2f}  exit {t.exit_reason}")
    assert m.total_trades > 0


if __name__ == "__main__":
    main()
