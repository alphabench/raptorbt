"""NIFTY option scalp: long at-the-money options on a 1-minute index cross, gated by the chain's implied volatility.

Shows a signal that reads two things at once, the index's bars and the state of the chain:

* a 1-minute EMA pair on the index gives the direction: fast above slow buys the at-the-money
  call, fast below slow buys the at-the-money put. Only options are bought, never sold;
* the chain gives a filter: the implied volatility of the at-the-money call, solved from its
  last close with ``raptorbt.pricing.implied_vol_status`` (a price with no volatility is skipped,
  never guessed). An entry is skipped while that volatility is more than ``IV_RICH`` times its
  mean so far this session: the premium is rich and a quick exit has the least room;
* the exit is quick: the option is sold at a premium target or stop, after ``MAX_HOLD_BARS``
  minutes, or from 15:15 IST, whichever is first. Size comes from ``raptorbt.sizing``
  (a fraction of free capital, rounded down to whole lots).

The index carries no open interest or volume in a bar run, so the chain state here is the
premium and the volatility solved from it; a strategy with recorded open interest would read
it the same way, from the contract's own series.

Skills: raptorbt-instruments, raptorbt-indicators, raptorbt-orders, raptorbt-risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, OptionChain, Strategy, pricing, run_strategies, sizing  # noqa: E402
from raptorbt.errors import InvalidParameter  # noqa: E402
from raptorbt.strategy import orders  # noqa: E402

ENTRY_MINUTE = 9 * 60 + 30     # no entry before 09:30 IST, as minutes after midnight
FLATTEN_MINUTE = 15 * 60 + 15  # everything is sold from 15:15 IST
IV_RICH = 1.10                 # skip an entry while the money call's volatility is this multiple of its session mean
TARGET, STOP = 1.08, 0.96      # sell at this multiple of the entry premium (target) or this (stop)
MAX_HOLD_BARS = 6              # a scalp is sold after this many index bars
FRACTION = 0.1                 # the share of free capital one entry may spend on premium


class ChainScalp(Strategy):
    def __init__(self, chain):
        super().__init__()
        self.chain = chain

    def on_start(self, ctx):
        self.fast = ctx.indicators.ema(instrument="NIFTY", period=5)
        self.slow = ctx.indicators.ema(instrument="NIFTY", period=20)
        self.prev_side = 0   # which side of the slow EMA the fast one was on at the last bar
        self.trade = None    # (contract symbol, entry premium, bars held) while one is open
        self.day = None      # the IST day the volatility mean belongs to
        self.iv_sum, self.iv_n = 0.0, 0

    def on_order_rejected(self, ctx, event):
        self.trade = None  # nothing is held, so nothing to sell

    def on_bar(self, ctx, event):
        if ctx.symbol != "NIFTY" or event.timeframe is not None:
            return
        local = ctx.bar_close_ts + _data.IST_OFFSET_NS
        minute, day = local // _data.MIN_NS % 1440, local // _data.DAY_NS
        if day != self.day:  # a new session: its own volatility mean, nothing carried over
            self.day, self.iv_sum, self.iv_n = day, 0.0, 0
        spot = ctx.bar.close
        iv = self.money_iv(ctx, spot)
        if iv is not None:
            self.iv_sum, self.iv_n = self.iv_sum + iv, self.iv_n + 1

        if self.trade is not None:
            symbol, entry, held = self.trade
            premium = float(ctx.series(symbol)["close"][-1])
            held += 1
            if premium >= entry * TARGET or premium <= entry * STOP or held >= MAX_HOLD_BARS or minute >= FLATTEN_MINUTE:
                self.close_position(symbol=symbol)
                self.trade = None
            else:
                self.trade = (symbol, entry, held)
            return

        if not (self.fast.ready and self.slow.ready):
            return
        side_now = 1 if self.fast.value > self.slow.value else -1
        crossed, self.prev_side = side_now != self.prev_side and self.prev_side != 0, side_now
        rich = iv is None or (self.iv_n > 1 and iv > IV_RICH * self.iv_sum / self.iv_n)
        if crossed and not rich and ENTRY_MINUTE <= minute < FLATTEN_MINUTE - 15:
            self.enter_option(ctx, spot, side_now)

    def money_iv(self, ctx, spot):
        """The implied volatility of the at-the-money call on the nearest expiry, or None when its
        price has none."""
        now = ctx.bar_close_ts
        expiry = self.chain.pick_expiry("NIFTY", now, "nearest")
        strike = self.chain.atm_strike("NIFTY", expiry, now, spot)
        call = self.chain.contract("NIFTY", expiry, strike, "call", now)
        premium = float(ctx.series(call.symbol)["close"][-1])
        tau = pricing.year_fraction(now, expiry)
        iv, status = pricing.implied_vol_status(premium, "call", spot, strike, tau, rate=_data.NIFTY_RATE)
        return float(iv) if status == pricing.IvStatus.OK else None

    def enter_option(self, ctx, spot, side):
        now = ctx.bar_close_ts
        expiry = self.chain.pick_expiry("NIFTY", now, "nearest")
        strike = self.chain.atm_strike("NIFTY", expiry, now, spot)
        contract = self.chain.contract("NIFTY", expiry, strike, "call" if side > 0 else "put", now)
        premium = float(ctx.series(contract.symbol)["close"][-1])  # the contract's bar at this instant
        try:
            lots = sizing.lots_for_capital(ctx.free_capital, premium, contract.lot_size, fraction=FRACTION)
        except InvalidParameter as e:
            if e.code == "below_one_lot":
                return  # not enough capital for one lot: do not trade
            raise
        self.submit_order(orders.Market(side="buy", quantity=sizing.units(lots, contract.lot_size)),
                          symbol=contract.symbol)
        self.trade = (contract.symbol, premium, 0)


def main():
    data, instruments, specs = _data.nifty_universe(days=3, seed=13)
    strategy = ChainScalp(OptionChain(specs))
    config = BacktestConfig(
        initial_capital=500_000.0, fees=0.0003, fill_timing="next_bar_open",
        session_tz_offset_ns=_data.IST_OFFSET_NS,  # a "day" is the IST day
        max_positions=1,
    )
    (result,) = run_strategies([strategy], data, config, instruments=instruments, bar_stamp="close")

    m = result.result.metrics
    print(f"trades {m.total_trades}  return {m.total_return_pct:.2f}%  max drawdown {m.max_drawdown_pct:.2f}%")
    for t in result.result.trades()[:4]:
        print(f"  {t.symbol}  {t.entry_price:.2f} -> {t.exit_price:.2f}  pnl {t.pnl:+.2f}  exit {t.exit_reason}")
    assert m.total_trades > 0


if __name__ == "__main__":
    main()
