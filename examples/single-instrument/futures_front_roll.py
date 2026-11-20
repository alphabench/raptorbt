"""Index futures: hold the front contract through its expiry, with the position carried across the roll.

Shows how a ``Future`` role works, on two synthetic monthly NIFTY contracts:

* the manifest names an ``Underlying`` (the index, bound to a symbol by whoever runs the
  strategy) and a ``Future(of="index", contract="continuous", roll_days=1)``: the contract that is
  front now, moved to the next one a day before its expiry;
* ``ctx.role("fut")`` answers which contract that is **at the event's time**, so it is read in
  ``on_bar``, never in ``on_start`` (before the first event the context has no time and the read is
  refused ``role_before_event``);
* ``RollEvents`` with ``auto_roll = True`` moves an open position into the new front with real
  orders at the roll, and ``on_roll`` reports each switch.

The strategy buys one lot of the front future on the first bar its 10-bar index SMA is ready and
holds it: the order names the contract the role names at that bar, and the roll carries the position
into the next contract a day before the first one expires.

Skills: raptorbt-strategy, raptorbt-instruments, raptorbt-live-trading, raptorbt-orders.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, InstrumentSpec, Strategy, run_strategies  # noqa: E402
from raptorbt.strategy import Future, Holding, Requirements, RollEvents, Underlying, orders  # noqa: E402

LOT = 75
EXPIRIES = {"NIFTY26JANFUT": dt.date(2026, 1, 8), "NIFTY26FEBFUT": dt.date(2026, 1, 15)}


class FrontFuture(RollEvents, Strategy):
    requirements = Requirements(
        roles={
            "index": Underlying(asset="index"),
            "fut": Future(of="index", contract="continuous", roll_days=1.0),  # rolls a day before expiry
        },
        timeframes=["1m"],
        holding=Holding.overnight(),
    )
    auto_roll = True  # an open position follows the front contract at each roll
    roll_sizing = "lots"  # ... with the same number of lots (the default keeps the notional, which rounds one lot down)

    def __init__(self):
        super().__init__()
        self.rolls = []
        self.entered = False

    def on_start(self, ctx):
        self.sma = ctx.indicators.sma(instrument="NIFTY", period=10)  # no ctx.role("fut") here: no time yet

    def on_roll(self, ctx, roll):
        self.rolls.append((roll.from_symbol, roll.to_symbol))

    def on_bar(self, ctx, event):
        if ctx.symbol != "NIFTY" or not self.sma.ready:
            return
        front = ctx.role("fut")  # the contract that is front at this bar
        if not self.entered:
            self.submit_order(orders.Market(side="buy", quantity=LOT), symbol=front)
            self.entered = True


def main():
    index = _data.bars(days=6, seed=11, start=24_000.0, step=3.0, wick=1.0, drift_period=600)
    ts, close = index["timestamps"], index["close"]
    data, specs = {"NIFTY": index}, {}
    for k, (symbol, expiry) in enumerate(EXPIRIES.items()):
        data[symbol] = {"timestamps": ts, **_data.ohlcv(close + 12.0 * (k + 1), 11 + k, wick=1.0)}  # a basis over the index
        specs[symbol] = InstrumentSpec.futures_contract(
            symbol, _data.ist_ns(expiry, 15, 30), float(LOT), underlying="NIFTY", margin_init=0.12)
    strategy = FrontFuture()
    config = BacktestConfig(initial_capital=2_500_000.0, fees=0.0003, fill_timing="next_bar_open",
                            session_tz_offset_ns=_data.IST_OFFSET_NS)
    (result,) = run_strategies([strategy], data, config, instruments=specs,
                               bindings={"index": "NIFTY"}, bar_stamp="close")

    m = result.result.metrics
    print(f"trades {m.total_trades}  return {m.total_return_pct:.2f}%  rolls {strategy.rolls}")
    for t in result.result.trades()[:4]:
        print(f"  {t.symbol}  {t.entry_price:.2f} -> {t.exit_price:.2f}  pnl {t.pnl:+.2f}  exit {t.exit_reason}")
    assert strategy.rolls == [("NIFTY26JANFUT", "NIFTY26FEBFUT")]


if __name__ == "__main__":
    main()
