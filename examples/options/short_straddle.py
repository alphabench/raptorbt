"""NIFTY intraday short straddle: sell the at-the-money call and put as one group, flat before the close.

Shows a short straddle placed through ``submit_structure`` on a synthetic chain:

* ``st.straddle("sell")`` describes the two legs (the money call and put, both sold);
  the engine resolves them against the chain at the event time ``ctx.bar_close_ts`` and
  sends them as one group of ordinary orders. A sold leg nothing covers goes out with the
  group, so there is no hedge: the margin is the whole risk of the position;
* size comes from ``raptorbt.sizing``: ``max_lots_by_margin`` finds the most lots whose
  margin, under the account's own ``MarginModel``, fits half of the free capital, from the
  premiums of the two resolved legs at that instant (the lot size is the contract's own);
* one straddle a session is opened after the first half hour, and ``close_structure``
  closes the group at 15:15 IST, before the exchange closes. Nothing else exits it.

The margin model is the example's own (``SELLER_MARGIN_PCT`` of the strike notional plus
the premium, ``MarginModel.option_simple``); a broker's span margin differs, so set yours.

Skills: raptorbt-instruments, raptorbt-orders, raptorbt-risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _data  # noqa: E402
from raptorbt import BacktestConfig, MarginModel, OptionChain, Strategy, run_strategies, sizing  # noqa: E402
from raptorbt import structures as st  # noqa: E402
from raptorbt.errors import InvalidParameter  # noqa: E402

ENTRY_MINUTE = 9 * 60 + 45     # no entry before 09:45 IST, as minutes after midnight
FLATTEN_MINUTE = 15 * 60 + 15  # the group is closed from 15:15 IST
SELLER_MARGIN_PCT = 15.0       # the writer's margin as a percent of the strike notional (an example's choice)
MARGIN_SHARE = 0.5             # the share of the free capital the straddle may margin


class ShortStraddle(Strategy):
    """Sells one at-the-money straddle on the nearest expiry each session and closes it at 15:15."""

    def __init__(self, chain, margin_model):
        super().__init__()
        self.chain = chain
        self.margin_model = margin_model
        self.structure = st.straddle("sell")
        self.group = None
        self.marks = {}      # symbol -> last close, what the group's margin is priced on
        self.last_day = None  # the IST day a straddle was last opened

    def on_bar(self, ctx):
        self.marks[ctx.symbol] = ctx.bar.close
        if ctx.symbol != "NIFTY":
            return
        self.update_structures(ctx, self.marks)
        local = ctx.bar_close_ts + _data.IST_OFFSET_NS
        minute, day = local // _data.MIN_NS % 1440, local // _data.DAY_NS

        if self.group is not None:
            status = self.structure_status(self.group)
            if status == "closed":
                self.group = None  # a new session may open another
            elif minute >= FLATTEN_MINUTE and status != "closing":
                self.close_structure(self.group, "flat before the close")
            return
        if day == self.last_day or not ENTRY_MINUTE <= minute < FLATTEN_MINUTE:
            return
        lots = self.lots_that_fit(ctx, ctx.bar.close)
        if lots is None:
            return
        self.group = self.submit_structure(
            ctx, self.structure, chain=self.chain, underlying="NIFTY", spot=ctx.bar.close, lots=lots,
        )
        self.last_day = day

    def lots_that_fit(self, ctx, spot):
        """The most lots of the straddle whose margin fits the share of free capital, or None."""
        now = ctx.bar_close_ts
        one = self.structure.resolve(self.chain, "NIFTY", spot, now, 1)
        prices = {leg.instrument: float(ctx.series(leg.instrument)["close"][-1]) for leg in one.legs}

        def margin(lots):
            return self.structure.resolve(self.chain, "NIFTY", spot, now, lots).margin(self.margin_model, prices)

        try:
            return sizing.max_lots_by_margin(margin, ctx.free_capital * MARGIN_SHARE)
        except InvalidParameter as e:
            if e.code == "below_one_lot":
                return None  # not enough margin for one lot: do not trade
            raise


def main():
    data, instruments, specs = _data.nifty_universe(days=3, seed=11)
    margin_model = MarginModel.option_simple(SELLER_MARGIN_PCT)
    strategy = ShortStraddle(OptionChain(specs), margin_model)
    config = BacktestConfig(
        initial_capital=2_000_000.0, fees=0.0003, fill_timing="next_bar_open",
        session_tz_offset_ns=_data.IST_OFFSET_NS,  # a "day" is the IST day
        margin_model=margin_model,
    )
    (result,) = run_strategies(
        [strategy], data, config, instruments=instruments,
        account_type="margin", leverage=1.0, bar_stamp="close",
    )

    m = result.result.metrics
    print(f"trades {m.total_trades}  return {m.total_return_pct:.2f}%  max drawdown {m.max_drawdown_pct:.2f}%")
    for t in result.result.trades()[:4]:
        print(f"  {t.symbol}  {t.entry_price:.2f} -> {t.exit_price:.2f}  pnl {t.pnl:+.2f}  exit {t.exit_reason}")
    assert m.total_trades > 0


if __name__ == "__main__":
    main()
