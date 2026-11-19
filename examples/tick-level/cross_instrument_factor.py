"""A flow factor kept across several instruments on every trade, read with a coarser bar.

Shows a tick strategy that reads more than the instrument it trades. ``on_trade`` fires for
every print of every subscribed instrument; the strategy keeps one running number, the signed
volume of the last ``FLOW_PRINTS`` prints of each instrument (an uptick adds the print's size, a
downtick subtracts it), and the factor is the sum over the two leaders ``LEAD_A`` and
``LEAD_B``. It trades only ``TRADED``. ``primary_bars=(1, "m")`` makes one-minute bars from the
prints, and ``subscribe_bars(5, "m")`` asks the engine for five-minute bars on top of them: the
factor is read on every trade, the trend on each completed five-minute bar of the traded
instrument (``event.timeframe == "5m"``, ``ctx.symbol`` the instrument it closed on).

The strategy buys ``TRADED`` when the factor is above ``ENTRY_FLOW`` and the last five-minute
close is above the one before; it sells when the factor turns negative. ``submit_order`` takes
the instrument in the order, and ``on_fill`` says when the position changed. The same strategy
runs unchanged on a live node fed the same prints with ``push_tick`` (and no bars of its own:
the engine builds them), which is what a deployed tick strategy receives.

Skills: raptorbt-market-data, raptorbt-subscriptions, raptorbt-timeframes, raptorbt-orders.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Strategy, run_tick_strategy  # noqa: E402
from raptorbt.strategy import orders  # noqa: E402

TRADED, LEAD_A, LEAD_B = "TRADED", "LEAD_A", "LEAD_B"
SYMBOLS = [TRADED, LEAD_A, LEAD_B]
TICK = 0.05
QUANTITY = 100
FLOW_PRINTS = 40  # prints of each instrument the running signed volume holds
ENTRY_FLOW = 150.0  # units of signed volume the two leaders must sum to before a buy


class FlowFactor(Strategy):
    def on_init(self, ctx):
        for symbol in SYMBOLS:
            self.subscribe(symbol, data_type="trade")

    def on_start(self, ctx):
        self.five_minute = self.subscribe_bars(5, "m")
        self.last_price = {}
        self.flow = {s: deque(maxlen=FLOW_PRINTS) for s in SYMBOLS}  # signed sizes, newest last
        self.closes = deque(maxlen=2)  # the traded instrument's last two five-minute closes
        self.long = False
        self.trades_seen = {s: 0 for s in SYMBOLS}
        self.five_minute_bars = 0
        self.decisions = []

    def factor(self):
        """The leaders' signed volume over their last prints: one number across instruments."""
        return sum(sum(self.flow[s]) for s in (LEAD_A, LEAD_B))

    def on_trade(self, ctx, tick):
        symbol = tick.symbol
        self.trades_seen[symbol] += 1
        before = self.last_price.get(symbol)
        self.last_price[symbol] = tick.price
        if before is not None and tick.price != before:
            self.flow[symbol].append(tick.size if tick.price > before else -tick.size)
        if symbol != TRADED:
            return
        if not self.long and self.trend_is_up() and self.factor() > ENTRY_FLOW:
            self.buy_id = self.submit_order(
                orders.Market(side="buy", quantity=QUANTITY, instrument=TRADED)
            )
            self.decisions.append(("buy", tick.timestamp, round(self.factor(), 1)))
        elif self.long and self.factor() < 0.0:
            self.sell_id = self.submit_order(
                orders.Market(side="sell", quantity=QUANTITY, instrument=TRADED)
            )
            self.decisions.append(("sell", tick.timestamp, round(self.factor(), 1)))

    def on_bar(self, ctx, event):
        if event.timeframe == "5m" and ctx.symbol == TRADED:  # a completed five-minute bar
            self.closes.append(event.close)
            self.five_minute_bars += 1

    def trend_is_up(self):
        return len(self.closes) == 2 and self.closes[-1] > self.closes[0]

    def on_fill(self, ctx, event):
        if event.client_order_id == getattr(self, "buy_id", None):
            self.long = True
        elif event.client_order_id == getattr(self, "sell_id", None):
            self.long = False


def make_ticks():
    """Three instruments on the same one-second grid; the leaders' flow runs ahead of the
    traded one's price. Every row carries ``ltq``, the size of its print."""
    rng = np.random.default_rng(8)
    n = 6_000
    ticks = {}
    for k, symbol in enumerate(SYMBOLS):
        t = _data.ticks(n=n, seed=20 + k, start=100.0 + 10 * k, tick=TICK)
        t["ltq"] = rng.integers(1, 40, n).astype(np.float64)
        ticks[symbol] = t
    return ticks


def main():
    strategy = FlowFactor()
    result = run_tick_strategy(
        strategy,
        make_ticks(),
        config=BacktestConfig(initial_capital=1_000_000.0, fees=0.0),
        primary_bars=(1, "m"),
    )
    trades = result.result.trades()
    print(f"prints {strategy.trades_seen}  five-minute bars {strategy.five_minute_bars}  "
          f"round trips {len(trades)}  net pnl {sum(t.pnl for t in trades):+.2f}")
    for row in strategy.decisions[:4]:
        print(f"  {row[0]:4s} at {row[1]}  factor {row[2]:+.1f}")
    assert all(n == 6_000 for n in strategy.trades_seen.values()), "a print was not delivered"
    assert strategy.five_minute_bars > 0, "no five-minute bar closed"
    assert strategy.decisions, "the factor never moved the strategy"


if __name__ == "__main__":
    main()
