"""A hedge-first bear call spread through ``submit_structure``.

Shows a multi-leg structure placed as one group. ``st.bear_call_spread`` describes
the legs (sell the at-the-money call, buy the call one strike higher); the engine
resolves them against the chain at the event time, sends the bought hedge first,
and releases the sold leg only once the hedge has filled and covers it. The same
call places the same orders in a backtest, on paper and on a broker.

``MarginModel.defined_risk()`` prices the pair for the most it can lose at expiry
(the width of the spread less the credit), not for the naked short call, so a small
account can hold it. Without it the sold leg would be margined on its own.

Skills: raptorbt-instruments, raptorbt-risk, raptorbt-portfolio.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, MarginModel, OptionChain, Strategy, run_strategies  # noqa: E402
from raptorbt import structures as st  # noqa: E402


class SellSpread(Strategy):
    """Opens one bear call spread on the index's tenth bar and holds it to the end."""

    def __init__(self, chain):
        super().__init__()
        self.chain = chain
        self.group = None

    def on_bar(self, ctx):
        if ctx.symbol == "NIFTY" and self.group is None and ctx.idx == 10:
            self.group = self.submit_structure(
                ctx, st.bear_call_spread(short_steps=0, long_steps=1),
                chain=self.chain, underlying="NIFTY", spot=ctx.bar.close, lots=1,
            )


def main():
    data, instruments, specs = _data.nifty_universe(days=1, seed=9)
    strategy = SellSpread(OptionChain(specs))
    config = BacktestConfig(
        initial_capital=200_000.0, fees=0.0, margin_model=MarginModel.defined_risk(),
    )
    (result,) = run_strategies(
        [strategy], data, config, instruments=instruments,
        account_type="margin", leverage=1.0, bar_stamp="close",
    )

    legs = [state.leg for state in strategy.group.legs]  # the resolved legs
    print("status", strategy.structure_status(strategy.group))
    for leg in legs:
        print(f"  {leg.side:<4} {leg.quantity:>5.0f} x {leg.instrument}")
    print(f"trades {result.result.metrics.total_trades}  net pnl {sum(t.pnl for t in result.result.trades()):+.2f}")
    assert strategy.structure_status(strategy.group) == "filled"
    assert [leg.side for leg in legs].count("buy") == 1


if __name__ == "__main__":
    main()
