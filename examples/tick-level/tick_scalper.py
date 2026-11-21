"""A tick scalper: buys short bursts of upward momentum from its trade hook, under a fixed stop
and a trailing stop, with a bounded number of entries.

Shows a strategy that trades from ``on_trade``:

- **The manifest.** ``requirements`` declares the prints and quotes it handles and the
  1-minute bars it reads (built from the prints by ``primary_bars=(1, "m")``), so a host can
  check it before a run; ``check_requirements`` finds no issue.
- **The signal.** On every print it compares the price with the one ``LOOKBACK`` prints back.
  A rise of ``MOMENTUM_TICKS`` ticks, a spread of at most ``MAX_SPREAD_TICKS`` ticks (from
  ``on_quote``) and a price above the 5-bar average of the 1-minute bars open a long.
- **The stops.** Each entry carries a fixed stop ``STOP_TICKS`` ticks under the print
  (``stop_price=``); the trailing stop comes from the configuration
  (``set_trailing_stop``), a fraction under the best print since entry. The engine checks
  both on every print (a quote moves neither); whichever is reached first closes the
  position. Both are stops to the trade record (``exit_reason == "StopLoss"``), so ``main``
  tells them apart by the exit price: at or under the entry's fixed level, it was the fixed
  stop. A market entry fills at the next print, so the fixed level sits ``STOP_TICKS`` under
  the signal's print, not under the fill.
- **A bounded order count.** No more than ``MAX_ENTRIES`` entries, and none while one is
  waiting to fill.

The run is ``run_tick_strategy`` over deterministic synthetic ticks. The same class runs
unchanged on a live node fed one print at a time, and ``main`` proves it trades the same:

    node = LiveNode([TickScalper()], ["DEMO"], config, primary_bars=(1, "m"))
    node.push_tick("DEMO", ts, ltp, bid, ask, ltq=size, arrival_ns=received)  # per print
    results = node.finish()

On a node a 1-minute bar built from prints closes on the first print of a later minute (never
on ``advance_time``), a print with no book (``bid`` and ``ask`` 0) or a one-sided book is a
trade with no quote, and a row's trade is heard before that row's quote, as in the batch run.

Skills: raptorbt-market-data, raptorbt-orders, raptorbt-risk, raptorbt-subscriptions,
raptorbt-live-trading.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, LiveNode, Strategy, run_tick_strategy  # noqa: E402
from raptorbt.strategy import Holding, Requirements, Underlying, check_requirements, orders  # noqa: E402

TICK = 0.05
QUANTITY = 100
LOOKBACK = 10  # prints
MOMENTUM_TICKS = 3
MAX_SPREAD_TICKS = 2
STOP_TICKS = 3
TRAILING = 0.004  # the trailing stop, a fraction of the best price since entry
MAX_ENTRIES = 25


class TickScalper(Strategy):
    requirements = Requirements(
        roles={"asset": Underlying(asset="equity")},
        timeframes=["1m"],  # the bars built from the prints, read by the trend filter
        holding=Holding.intraday(flatten_before="15:15"),
        data=["bar", "trade", "quote"],
    )

    def on_start(self, ctx):
        for symbol in ctx.symbols:  # the symbol the deployer bound, never a name written here
            self.subscribe(symbol, data_type="trade")
            self.subscribe(symbol, data_type="quote")
        self.trend = {s: ctx.indicators.sma(instrument=s, period=5) for s in ctx.symbols}
        self.prices = deque(maxlen=LOOKBACK + 1)
        self.spread = None
        self.entries = 0
        self.stops = []  # the fixed level of each entry, in order
        self.waiting = False

    def on_quote(self, ctx, quote):
        self.spread = quote.spread

    def on_trade(self, ctx, tick):
        self.prices.append(tick.price)
        if self.waiting or not ctx.is_flat or self.entries >= MAX_ENTRIES:
            return
        trend = self.trend[ctx.symbol]
        rising = len(self.prices) == self.prices.maxlen and tick.price - self.prices[0] >= MOMENTUM_TICKS * TICK - 1e-9
        tight = self.spread is not None and self.spread <= MAX_SPREAD_TICKS * TICK + 1e-9
        if rising and tight and trend.ready and tick.price > trend.value:
            stop = round(tick.price - STOP_TICKS * TICK, 2)
            self.submit_order(orders.Market(side="buy", quantity=QUANTITY, stop_price=stop))
            self.stops.append(stop)
            self.entries += 1
            self.waiting = True

    def on_fill(self, ctx, event):
        self.waiting = False

    def on_order_rejected(self, ctx, event):
        self.waiting = False


def config():
    # The manifest's intraday holding squares off at 15:15 local time; the synthetic tape's
    # clock is the one the shared data helper uses.
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0, session_tz_offset_ns=_data.IST_OFFSET_NS)
    cfg.set_trailing_stop(TRAILING)
    return cfg


def tape():
    ticks = _data.ticks(n=6_000, seed=21, start=100.0, tick=TICK)
    ticks["ltq"] = np.full(len(ticks["timestamps"]), 10.0)
    return {"DEMO": ticks}


def summary(result):
    """Each round trip, rounded so two runs compare exactly."""
    return [(t.entry_idx, round(t.entry_price, 6), round(t.exit_price, 6), t.exit_reason, round(t.pnl, 6))
            for t in result.result.trades()]


def main():
    assert check_requirements(TickScalper) == []

    data = tape()
    strategy = TickScalper()
    result = run_tick_strategy(strategy, data, config=config(), primary_bars=(1, "m"))
    trades = summary(result)
    assert trades, "the scalper never traded"
    assert strategy.entries <= MAX_ENTRIES
    fixed = sum(t[2] <= stop + 1e-9 for t, stop in zip(trades, strategy.stops))  # one position at a time
    assert 0 < fixed < len(trades), "both stops should close some positions"
    print(f"entries {strategy.entries} (cap {MAX_ENTRIES})  round trips {len(trades)}  "
          f"net pnl {sum(t[4] for t in trades):+.2f}  fixed stop {fixed}  trailing stop {len(trades) - fixed}")
    for _, entry, exit_, reason, pnl in trades[:3]:
        print(f"  bought {entry:.2f}  sold {exit_:.2f}  pnl {pnl:+.2f}  exit {reason}")

    # The same class on a live node, one print at a time: the same trades.
    t = data["DEMO"]
    node = LiveNode([TickScalper()], ["DEMO"], config(), primary_bars=(1, "m"))
    for i in range(len(t["timestamps"])):
        node.push_tick("DEMO", int(t["timestamps"][i]), float(t["ltp"][i]), float(t["bid"][i]),
                       float(t["ask"][i]), ltq=float(t["ltq"][i]))
    (live,) = node.finish()
    assert summary(live) == trades
    print(f"a live node fed the same {len(t['timestamps'])} prints one at a time made the same {len(trades)} trades")


if __name__ == "__main__":
    main()
