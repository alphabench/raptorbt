"""A tick strategy that rests limit orders at the bid and takes a few ticks of profit.

Shows the tick runner: ``run_tick_strategy`` feeds ``on_trade`` and ``on_quote``
events one at a time, orders match against the prints and quotes (not against bars),
and a resting limit order fills only when the market trades through it. The
strategy quotes a buy at the best bid, gives up on it if it has not filled after
20 quotes, and once filled rests a sell three ticks higher. ``self.subscribe``
narrows the feed to the symbol and data types it uses.

``submit_order`` returns the order's client id, which ``on_fill`` and
``cancel_order`` use. A fill is reported to ``on_fill`` with ``event.kind ==
"order_filled"`` for a typed order.

Skills: raptorbt-market-data, raptorbt-orders, raptorbt-subscriptions,
raptorbt-execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Strategy, run_tick_strategy  # noqa: E402
from raptorbt.strategy import orders  # noqa: E402

TICK = 0.05
QUANTITY = 100
PROFIT_TICKS = 3
PATIENCE = 20  # quotes to wait for the buy before cancelling it


class PassiveBid(Strategy):
    def on_init(self, ctx):
        self.subscribe("DEMO", data_type="trade")
        self.subscribe("DEMO", data_type="quote")

    def on_start(self, ctx):
        self.buy_id = self.sell_id = None
        self.waited = 0
        self.fills = []

    def on_quote(self, ctx, quote):
        if self.buy_id is None and self.sell_id is None and ctx.is_flat:
            self.buy_id = self.submit_order(orders.Limit(side="buy", price=quote.bid, quantity=QUANTITY))
            self.waited = 0
        elif self.buy_id is not None:
            self.waited += 1
            if self.waited >= PATIENCE:
                self.cancel_order(self.buy_id)  # the market left without us
                self.buy_id = None

    def on_order_canceled(self, ctx, event):
        self.buy_id = None if event.client_order_id == self.buy_id else self.buy_id

    def on_fill(self, ctx, event):
        self.fills.append((event.kind, event.price))
        if event.client_order_id == self.buy_id:
            self.buy_id = None
            self.sell_id = self.submit_order(
                orders.Limit(side="sell", price=round(event.price + PROFIT_TICKS * TICK, 2), quantity=QUANTITY)
            )
        elif event.client_order_id == self.sell_id:
            self.sell_id = None


def main():
    ticks = {"DEMO": _data.ticks(n=3_000, seed=4, start=100.0, tick=TICK)}
    strategy = PassiveBid()
    result = run_tick_strategy(strategy, ticks, config=BacktestConfig(initial_capital=100_000.0, fees=0.0))

    trades = result.result.trades()
    print(f"fills {len(strategy.fills)}  round trips {len(trades)}  "
          f"net pnl {sum(t.pnl for t in trades):+.2f}")
    for t in trades[:3]:
        print(f"  bought {t.entry_price:.2f}  sold {t.exit_price:.2f}  pnl {t.pnl:+.2f}  exit {t.exit_reason}")
    assert strategy.fills, "no order filled"


if __name__ == "__main__":
    main()
