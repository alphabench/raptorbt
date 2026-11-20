"""NIFTY options momentum: 1-minute entries, 15-minute trend confirmation, ATR stops, portfolio risk limits.

Shows the pieces an options strategy is built from, on a synthetic chain:

* the index and every contract run together in one ``run_strategies`` call, and
  the strategy decides on the index bar and trades a contract picked from an
  ``OptionChain`` at the event time ``ctx.bar_close_ts``;
* a 1-minute SMA cross gives the entry, a 15-minute EMA pair
  (``ctx.indicators.ema(..., timeframe="15m")``) says whether the trend agrees,
  and a 1-minute ATR sets the stop and the target on the index;
* size comes from ``raptorbt.sizing`` (a fraction of capital, rounded down to
  whole lots), and the account carries limits: one position at a time, a daily
  loss that flattens, and a drawdown halt. ``on_risk_event`` reports a breach.

A bullish cross with the 15-minute trend up buys the at-the-money call; a bearish
cross with the trend down buys the at-the-money put. Everything is flat by 15:15 IST.

Skills: raptorbt-instruments, raptorbt-subscriptions, raptorbt-indicators,
raptorbt-orders, raptorbt-risk, raptorbt-portfolio.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, OptionChain, Strategy, run_strategies, sizing  # noqa: E402
from raptorbt.errors import InvalidParameter  # noqa: E402
from raptorbt.strategy import orders  # noqa: E402

FLATTEN_MINUTE = 15 * 60 + 15  # 15:15 IST, as minutes after midnight
ATR_STOP, ATR_TARGET = 2.0, 3.0


class OptionsMomentum(Strategy):
    def __init__(self, chain):
        super().__init__()
        self.chain = chain

    def on_init(self, ctx):
        self.subscribe("NIFTY", timeframe="15m", history=20)

    def on_start(self, ctx):
        self.sma = ctx.indicators.sma(instrument="NIFTY", period=10)
        self.atr = ctx.indicators.atr(instrument="NIFTY", period=14)
        self.fast15 = ctx.indicators.ema(instrument="NIFTY", period=3, timeframe="15m")
        self.slow15 = ctx.indicators.ema(instrument="NIFTY", period=8, timeframe="15m")
        self.prev_side = 0  # which side of the SMA the last index close was on
        self.trade = None   # (contract symbol, +1 call / -1 put, stop, target) while one is open
        self.risk_events = []

    def on_risk_event(self, ctx, event):
        self.risk_events.append((event.kind, event.reject_reason))

    def on_order_rejected(self, ctx, event):
        self.trade = None  # a limit refused the entry: nothing is held, so nothing to stop

    def on_bar(self, ctx, event):
        if ctx.symbol != "NIFTY" or event.timeframe is not None:
            return
        ready = self.sma.ready and self.atr.ready and self.fast15.ready and self.slow15.ready
        if not ready:
            return
        spot = ctx.bar.close
        minute = (ctx.bar_close_ts + _data.IST_OFFSET_NS) // _data.MIN_NS % 1440

        if self.trade is not None:
            symbol, side, stop, target = self.trade
            hit = spot <= stop if side > 0 else spot >= stop
            won = spot >= target if side > 0 else spot <= target
            if hit or won or minute >= FLATTEN_MINUTE:
                self.close_position(symbol=symbol)
                self.trade = None
            return

        side_now = 1 if spot > self.sma.value else -1
        crossed, self.prev_side = side_now != self.prev_side and self.prev_side != 0, side_now
        trend = 1 if self.fast15.value > self.slow15.value else -1
        if crossed and side_now == trend and minute < FLATTEN_MINUTE - 30:
            self.enter_option(ctx, spot, side_now)

    def enter_option(self, ctx, spot, side):
        now = ctx.bar_close_ts
        expiry = self.chain.pick_expiry("NIFTY", now, "nearest")
        strike = self.chain.atm_strike("NIFTY", expiry, now, spot)
        contract = self.chain.contract("NIFTY", expiry, strike, "call" if side > 0 else "put", now)
        premium = float(ctx.series(contract.symbol)["close"][-1])  # the contract's bar at this instant
        try:
            lots = sizing.lots_for_capital(ctx.free_capital, premium, contract.lot_size, fraction=0.2)
        except InvalidParameter as e:
            if e.code == "below_one_lot":
                return  # not enough capital for one lot: do not trade
            raise
        self.submit_order(orders.Market(side="buy", quantity=sizing.units(lots, contract.lot_size)),
                          symbol=contract.symbol)
        atr = self.atr.value
        self.trade = (contract.symbol, side, spot - side * ATR_STOP * atr, spot + side * ATR_TARGET * atr)


def main():
    data, instruments, specs = _data.nifty_universe(days=2, seed=5)
    strategy = OptionsMomentum(OptionChain(specs))
    config = BacktestConfig(
        initial_capital=500_000.0, fees=0.0003, fill_timing="next_bar_open",
        session_tz_offset_ns=_data.IST_OFFSET_NS,     # a "day" is the IST day
        max_positions=1,                               # portfolio limits
        max_daily_loss_pct=3.0, flatten_on_daily_loss=True,
        max_drawdown_pct=10.0,
    )
    (result,) = run_strategies([strategy], data, config, instruments=instruments, bar_stamp="close")

    m = result.result.metrics
    print(f"trades {m.total_trades}  return {m.total_return_pct:.2f}%  max drawdown {m.max_drawdown_pct:.2f}%  "
          f"halted {result.halted}  risk events {len(strategy.risk_events)}")
    for t in result.result.trades()[:4]:
        print(f"  {t.symbol}  {t.entry_price:.2f} -> {t.exit_price:.2f}  pnl {t.pnl:+.2f}  exit {t.exit_reason}")
    assert m.total_trades > 0


if __name__ == "__main__":
    main()
