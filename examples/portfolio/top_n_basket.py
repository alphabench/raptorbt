"""A basket strategy: one strategy over N instruments the deployer picks, holding the top few by momentum.

Shows the manifest of a basket. The strategy declares a ``Universe`` role (an asset class, how
many instruments it takes, and the data each needs) and the largest position it takes in each
member (``max_position``); a host can read both before it binds anything. The deployer then
binds the role to a list of symbols (``bindings=``), and the strategy reads that list with
``ctx.role("basket")``. ``check_requirements`` binds the role to synthetic symbols and runs the
strategy, so a strategy that calls ``ctx.role`` is checked too.

The run also uses ``trade_from``: the first session only warms the momentum up, every order
decided in it is refused (``warmup``), and trading starts on the second.

Each rebalance it holds the two names with the best 60-bar momentum, equal weight, and sells
the rest. Netting: a name already held is left alone.

Skills: raptorbt-portfolio, raptorbt-strategy, raptorbt-deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, Strategy, run_strategies  # noqa: E402
from raptorbt.strategy import (  # noqa: E402
    Holding,
    MaxPosition,
    Requirements,
    Universe,
    check_requirements,
    requirements_hash,
)
from raptorbt.strategy.orders import Market  # noqa: E402

LOOKBACK = 60
REBALANCE_EVERY = 30
HOLD = 2


class TopMomentum(Strategy):
    """Long the ``HOLD`` basket members with the best ``LOOKBACK``-bar momentum."""

    requirements = Requirements(
        roles={"basket": Universe(asset="equity", min=3, max=10, data=("bar",))},
        timeframes=["1m"],
        holding=Holding.overnight(),
        data=["bar"],
        # a host can size the capital from this without asking: no member is more than a third
        max_position={"basket": MaxPosition(fraction=1.0 / 3.0)},
    )

    def __init__(self):
        super().__init__()
        self.bars = 0
        self.held = set()
        self.refused = 0

    def on_bar(self, ctx):
        members = ctx.role("basket")  # the symbols the deployer bound, in the order they gave them
        if ctx.symbol != members[0]:
            return  # decide once per step, on the first member
        self.bars += 1
        if self.bars <= LOOKBACK or self.bars % REBALANCE_EVERY:
            return
        closes = {s: ctx.series(s)["close"] for s in members}
        momentum = {s: closes[s][-1] / closes[s][-LOOKBACK] - 1.0 for s in members}
        best = set(sorted(members, key=momentum.get, reverse=True)[:HOLD])
        for symbol in members:
            held = bool(ctx.positions(symbol))
            if symbol in best and not held:
                self.submit_order(Market(side="buy", size_frac=1.0 / HOLD), symbol=symbol)
            elif symbol not in best and held:
                self.close_position(symbol=symbol)

    def on_order_rejected(self, ctx, event):
        self.refused += 1


def main():
    assert check_requirements(TopMomentum) == []  # the manifest, and the strategy's own use of it
    print("manifest", requirements_hash(TopMomentum)[:12], TopMomentum.requirements.roles["basket"])

    chosen = ["AAA", "BBB", "CCC", "DDD"]  # what the deployer picked for this run
    data = {s: _data.bars(days=2, seed=40 + i, start=100.0 + 20 * i, step=0.2) for i, s in enumerate(chosen)}
    strategy = TopMomentum()
    day_two = int(data["AAA"]["timestamps"][_data.MINUTES_PER_DAY])  # trading starts on the second session
    (result,) = run_strategies(
        [(strategy, chosen)], data,
        BacktestConfig(initial_capital=500_000.0, fees=0.0003, session_tz_offset_ns=_data.IST_OFFSET_NS),
        bar_stamp="close",
        bindings={"basket": chosen},
        trade_from=day_two,
    )
    m = result.result.metrics
    print(f"trades {m.total_trades}  return {m.total_return_pct:.2f}%  refused during warm-up {strategy.refused}")
    assert strategy.refused > 0, "the first session should have tried to trade"
    assert all(t.entry_time >= day_two for t in result.result.trades())
    assert m.total_trades > 0


if __name__ == "__main__":
    main()
