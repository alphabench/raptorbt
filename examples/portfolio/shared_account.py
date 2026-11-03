"""Several strategies on one shared account, under account limits and a kill switch.

Shows ``run_strategies`` with ``accounts=``. Three strategies trade three symbols
out of one 300,000 account: they share its cash and its limits, and within one
bar they step in list order, so the first listed gets scarce cash first. The
account allows two positions at once, stops new entries for the day after a 0.5
percent daily loss, and halts for good at a 1 percent drawdown from its peak
(``AccountSpec``). One strategy also keeps its own kill switch: ``ctx.halt``
stops that strategy alone and flattens it.

Each strategy gets its own result (with an equity curve attributed to it); the
account gets one too, with the shared curve and whether it halted.

Skills: raptorbt-portfolio, raptorbt-risk, raptorbt-strategy.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import AccountSpec, BacktestConfig, Indicator, Strategy, run_strategies  # noqa: E402

SYMBOLS = ("AAA", "BBB", "CCC")
COOLDOWN = 30  # bars to wait after a refusal: retrying into the same limit every bar changes nothing


class Refusals(Strategy):
    """Counts what the account refused, and holds the strategy back for a while after each refusal."""

    def __init__(self):
        super().__init__()
        self.refused = Counter()
        self.retry_at = 0

    def on_order_rejected(self, ctx, event):
        self.refused[event.reject_reason] += 1
        self.retry_at = ctx.idx + COOLDOWN

    def may_enter(self, ctx):
        return ctx.is_flat and ctx.idx >= self.retry_at


class SmaTrend(Refusals):
    """Long its symbol while the fast SMA is above the slow one."""

    def __init__(self, symbol, fast=8, slow=30):
        super().__init__()
        self.symbol, self.fast_n, self.slow_n = symbol, fast, slow

    def on_start(self, ctx):
        self.fast = self.register_indicator(Indicator.sma(self.fast_n), symbol=self.symbol)
        self.slow = self.register_indicator(Indicator.sma(self.slow_n), symbol=self.symbol)

    def on_bar(self, ctx):
        if ctx.symbol != self.symbol or not self.indicators_initialized():
            return
        if self.may_enter(ctx) and self.fast.value > self.slow.value:
            self.enter_long(size_frac=0.3)
        elif not ctx.is_flat and self.fast.value < self.slow.value:
            self.close_position()


class Breakout(Refusals):
    """Long a 40-bar high, out under the 15-bar low; stops itself if it loses 2 percent of its capital."""

    def __init__(self, symbol, start_equity):
        super().__init__()
        self.symbol, self.start_equity = symbol, start_equity

    def on_bar(self, ctx):
        if ctx.symbol != self.symbol:
            return
        if ctx.equity < self.start_equity * 0.98:
            ctx.halt("strategy loss limit", flatten=True)  # latches: no further entries from this strategy
            return
        close = ctx.series(self.symbol)["close"]
        if len(close) <= 40:
            return
        if self.may_enter(ctx) and close[-1] > close[-41:-1].max():
            self.enter_long(size_frac=0.3)
        elif not ctx.is_flat and close[-1] < close[-16:-1].min():
            self.close_position()


def main():
    data = {s: _data.bars(days=3, seed=30 + i, start=100.0 + 40 * i, step=0.35 + 0.1 * i)
            for i, s in enumerate(SYMBOLS)}
    strategies = [
        (SmaTrend("AAA"), ["AAA"], {"account": "desk"}),
        (SmaTrend("BBB", fast=5, slow=20), ["BBB"], {"account": "desk"}),
        (Breakout("CCC", start_equity=100_000.0), ["CCC"], {"account": "desk"}),
    ]
    desk = AccountSpec(
        initial_capital=300_000.0, max_positions=2,
        max_daily_loss_pct=0.5, max_drawdown_pct=1.0,
    )
    results, accounts = run_strategies(
        strategies, data,
        BacktestConfig(fees=0.0003, session_tz_offset_ns=_data.IST_OFFSET_NS),
        accounts={"desk": desk}, bar_stamp="close",
    )

    for label, (strategy, *_), r in zip(("SmaTrend AAA", "SmaTrend BBB", "Breakout CCC"), strategies, results):
        print(f"{label:<13} trades {r.result.metrics.total_trades:>2}  halted {r.halted}  "
              f"refused {dict(strategy.refused)}")
    account = accounts["desk"]
    peak = account.equity_curve().max()
    print(f"account       balance {account.balance:,.0f}  peak {peak:,.0f}  halted {account.halted}")
    assert account.initial_capital == 300_000.0
    assert sum(r.result.metrics.total_trades for r in results) > 0


if __name__ == "__main__":
    main()
