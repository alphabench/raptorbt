"""A position whose stop-loss and take-profit move with the market, through exit rules.

Shows the exit-rule API on ticks (``ExitRule``, ``set_exit``, ``cancel_exit`` and the
``ExitEvents`` mix-in):

- **The entry.** A long of ``QUANTITY`` units on the first print after every flat spell of
  ``WAIT`` prints, at most ``MAX_ENTRIES`` times.
- **The rule.** Once the position exists, ``set_exit`` arms a rule on its symbol with a stop
  ``STOP`` under the entry price and a target ``TARGET`` above it. For a long of ``q`` units a
  rule's ``premium`` is the last price times ``q``, so ``premium_at_or_below=level * q`` is a
  stop at ``level`` and ``premium_at_or_above=level * q`` a target at ``level``.
- **Moving it.** Each time the print reaches ``STEP`` above the level the stop was last moved
  from, the strategy moves both levels up by ``STEP``: it cancels the armed rule and sets a new
  one. A rule has no modify; cancel-and-set is how it changes, and the cancelled handle stays
  in the strategy's exit book (``state == "canceled"``). Moving on a step, not on every print,
  keeps that book short.
- **Firing.** ``ExitEvents`` evaluates the armed rules after every ``on_trade``; a rule that
  fires sends one reduce-only market order per open position (client id ``{rule}:x{n}``), and
  the handle ends ``done`` once the position is flat.

The run is ``run_tick_strategy`` over deterministic synthetic ticks. The same class runs
unchanged on a live node, ``LiveNode([MovingExit()], ["DEMO"], config)`` with one
``node.push_tick(...)`` per print, and ``main`` checks that it moves and fires the same rules
there. On a node an armed rule reads the symbol's last price from the engine instead of copying
the symbol's series out of its ring on every print.

Skills: raptorbt-orders, raptorbt-live-trading, raptorbt-market-data, raptorbt-strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, LiveNode, Strategy, run_tick_strategy  # noqa: E402
from raptorbt.strategy import Holding, Requirements, Underlying, check_requirements, orders  # noqa: E402
from raptorbt.strategy.exits import ExitEvents, ExitRule  # noqa: E402

TICK = 0.05
QUANTITY = 100
STOP = 0.30  # under the entry price
TARGET = 0.60  # over the entry price
STEP = 0.20  # how far the print must climb before both levels move up by the same amount
WAIT = 50  # flat prints before the next entry
MAX_ENTRIES = 6


class MovingExit(ExitEvents, Strategy):
    requirements = Requirements(
        roles={"asset": Underlying(asset="equity")},
        timeframes=["1m"],
        holding=Holding.overnight(),
        data=["bar", "trade"],
    )

    def on_start(self, ctx):
        for symbol in ctx.symbols:
            self.subscribe(symbol, data_type="trade")
        self.entries = 0
        self.flat_prints = WAIT
        self.handle = None
        self.anchor = self.stop = self.target = None
        self.moves = []  # (entry number, print, old stop, new stop, new target) while the position is open
        self.handles = []  # every handle set, in order

    def on_trade(self, ctx, tick):
        held = ctx.positions(ctx.symbol)
        if not held:
            self.handle = None
            self.flat_prints += 1
            if self.flat_prints >= WAIT and self.entries < MAX_ENTRIES:
                self.submit_order(orders.Market(side="buy", quantity=QUANTITY))
                self.entries += 1
                self.flat_prints = 0
            return
        if self.handle is None:  # the entry has filled: arm the first rule at its price
            entry = held[0].entry_price
            self.anchor = entry
            self._arm(ctx, entry - STOP, entry + TARGET)
        elif self.handle.state == "armed" and tick.price >= self.anchor + STEP - 1e-9:
            # The market moved our way: move the stop and the target up by one step.
            self.anchor += STEP
            old = self.stop
            self.cancel_exit(self.handle)
            self._arm(ctx, self.stop + STEP, self.target + STEP)
            self.moves.append((self.entries, tick.price, old, self.stop, self.target))
        # ExitEvents evaluates the armed rule after this hook returns.

    def _arm(self, ctx, stop, target):
        self.stop, self.target = round(stop, 2), round(target, 2)
        rule = ExitRule(premium_at_or_below=self.stop * QUANTITY, premium_at_or_above=self.target * QUANTITY)
        self.handle = self.set_exit(ctx, rule, symbol=ctx.symbol)
        self.handles.append(self.handle)


def main():
    assert check_requirements(MovingExit) == []

    ticks = {"DEMO": _data.ticks(n=4_000, seed=8, start=100.0, tick=TICK)}
    config = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    strategy = MovingExit()
    result = run_tick_strategy(strategy, ticks, config=config)

    trades = result.result.trades()
    fired = [h for h in strategy.handles if h.state == "done"]
    canceled = [h for h in strategy.handles if h.state == "canceled"]
    assert strategy.moves, "no rule was moved while its position was open"
    assert fired and all(h.reason in ("premium_at_or_below", "premium_at_or_above") for h in fired)
    assert len(trades) == len(fired)  # every position was closed by its rule
    print(f"entries {strategy.entries}  rules set {len(strategy.handles)}  moved {len(strategy.moves)} times  "
          f"fired {len(fired)}  cancelled by a move {len(canceled)}")
    for n, price, old, stop, target in strategy.moves[:3]:
        print(f"  entry {n}: print {price:.2f} moved the stop {old:.2f} -> {stop:.2f}, target -> {target:.2f}")
    for t, h in zip(trades, fired):
        print(f"  bought {t.entry_price:.2f}  sold {t.exit_price:.2f}  pnl {t.pnl:+.2f}  rule {h.reason}")

    # The same class on a live node, one print at a time: the same moves, rules and trades.
    t = ticks["DEMO"]
    live = MovingExit()
    node = LiveNode([live], ["DEMO"], config)
    for i in range(len(t["timestamps"])):
        node.push_tick("DEMO", int(t["timestamps"][i]), float(t["ltp"][i]), float(t["bid"][i]), float(t["ask"][i]))
    (live_result,) = node.finish()
    assert live.moves == strategy.moves
    assert [(h.state, h.reason, h.close_ids) for h in live.handles] == [(h.state, h.reason, h.close_ids) for h in strategy.handles]
    assert [(x.entry_price, x.exit_price, x.pnl) for x in live_result.result.trades()] == [
        (x.entry_price, x.exit_price, x.pnl) for x in trades
    ]
    print(f"a live node fed the same {len(t['timestamps'])} prints moved and fired the same rules")


if __name__ == "__main__":
    main()
