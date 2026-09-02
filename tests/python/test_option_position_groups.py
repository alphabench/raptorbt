"""Sold options that hedge each other are margined as one group (0.13.0).

In plain words: a sold call and a sold put on the same underlying and
expiry cannot both lose at once, and an exchange charges the pair much less
than two separate deposits. Since 0.13.0 the portfolio session re-prices
sold legs that share an underlying and expiry as one group once they are
open together, so a hedged book keeps more capital free and is not
margin-called at a level only two naked deposits would have breached.
"""

import numpy as np

from raptorbt import BacktestConfig, InstrumentSpec, Strategy, run_portfolio_strategy

CE, PE = "BANKNIFTY57000CE", "BANKNIFTY57000PE"
STRIKE, LOT = 57_000.0, 30.0
SPAN, EXPOSURE = 0.0975, 0.02
NAKED_PER_LOT = (SPAN + EXPOSURE) * STRIKE * LOT  # 2,00,925


def _bars(closes, start_ts=0, step=10):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(start_ts, start_ts + n * step, step, dtype=np.int64),
        "open": closes.copy(),
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _spec(symbol, right):
    return InstrumentSpec.option(
        symbol,
        strike=STRIKE,
        right=right,
        expiration_ns=10_000,
        lot_size=LOT,
        underlying="BANKNIFTY",
        span_pct=SPAN,
        exposure_pct=EXPOSURE,
    )


# An unsized entry takes every lot the pool can carry and starves the leg
# behind it, so each leg asks for the fraction that lands on exactly one lot:
# the call first (half the pool), then the put against what is left.
FRACTIONS = {CE: 0.5, PE: 0.97}


class _SellBothOnce(Strategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.done = set()
        self.free_after_pair = None
        self.margin_calls = 0

    def on_bar(self, ctx):
        if ctx.symbol not in self.done:
            self.done.add(ctx.symbol)
            self.enter(size_frac=FRACTIONS[ctx.symbol])
            return
        if len(self.done) == 2 and self.free_after_pair is None:
            self.free_after_pair = ctx.free_capital

    def on_margin_call(self, ctx, event):
        self.margin_calls += 1


def _run(capital, ce_premiums, pe_premiums):
    strategy = _SellBothOnce()
    result = run_portfolio_strategy(
        strategy,
        {CE: _bars(ce_premiums, start_ts=0), PE: _bars(pe_premiums, start_ts=5)},
        config=BacktestConfig(initial_capital=capital, fees=0.0),
        directions={CE: -1, PE: -1},
        instruments={CE: _spec(CE, "call"), PE: _spec(PE, "put")},
        account_type="margin",
        leverage=1.0,
    )
    return result, strategy


def test_a_sold_straddle_frees_the_group_benefit_once_both_legs_are_on():
    result, strategy = _run(
        420_000.0, [1_006.15, 1_006.15, 1_006.15], [551.05, 551.05, 551.05]
    )
    assert len(result.result.trades()) == 2
    # Group: span once + exposure on both − premium collected.
    group = SPAN * STRIKE * LOT + EXPOSURE * STRIKE * LOT * 2 - (1_006.15 + 551.05) * LOT
    assert strategy.free_after_pair is not None
    assert abs(strategy.free_after_pair - (420_000.0 - group)) < 1.0, strategy.free_after_pair
    # Not what two naked deposits would have left.
    assert strategy.free_after_pair > 420_000.0 - 2 * NAKED_PER_LOT + 100_000.0


def test_a_hedged_pair_survives_a_move_two_naked_deposits_would_not():
    # The call runs against the seller by 3,993.85 × 30 = 1,19,815: equity
    # 2,90,185 is under two naked deposits but over the group figure.
    result, strategy = _run(
        410_000.0, [1_006.15, 5_000.0, 5_000.0], [551.05, 551.05, 551.05]
    )
    assert len(result.result.trades()) == 2
    assert strategy.margin_calls == 0
    assert not result.halted


def test_the_second_sold_leg_must_still_be_carriable_on_its_own():
    """Sizing a new sold leg uses its naked deposit; the group benefit
    arrives after it is on. ₹4 lakh carries one deposit (2,00,925), and the
    1,99,075 left cannot carry a second."""
    result, _ = _run(400_000.0, [1_006.15, 1_006.15], [551.05, 551.05])
    assert len(result.result.trades()) == 1
