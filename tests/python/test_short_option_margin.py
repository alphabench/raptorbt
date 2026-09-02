"""Sold options reserve an exchange-style deposit, not just their premium (0.12.0).

In plain words: selling an option collects a small premium but can lose
without limit, so a real account must set aside a deposit scaled to the
underlying's value. Before 0.12.0 a margin-account backtest charged a sold
option only its premium, so a small book could sell many more lots than any
broker would allow. ``InstrumentSpec.option(span_pct=..., exposure_pct=...)``
now models that deposit; both default to zero, so existing runs are
unchanged unless a caller opts in.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, InstrumentSpec, Strategy, run_portfolio_strategy

SYM = "BANKNIFTY57500CE"
STRIKE = 57_500.0
LOT = 30.0
SPAN, EXPOSURE = 0.0975, 0.02
DEPOSIT_PER_CONTRACT = (SPAN + EXPOSURE) * STRIKE  # 6,756.25


def _bars(closes, start_ts=0, step=1):
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


def _config(capital):
    config = BacktestConfig(initial_capital=capital)
    config.fees = 0.0
    return config


def _spec(span_pct=SPAN, exposure_pct=EXPOSURE):
    return InstrumentSpec.option(
        SYM,
        strike=STRIKE,
        right="call",
        expiration_ns=10_000,
        lot_size=LOT,
        span_pct=span_pct,
        exposure_pct=exposure_pct,
    )


class _SellOnce(Strategy):
    """Sell the whole free balance's worth on the first bar, once."""

    def __init__(self, config=None):
        super().__init__(config)
        self.rejects = []
        self.done = False

    def on_bar(self, ctx):
        if ctx.symbol != SYM or self.done:
            return
        self.done = True
        self.enter(size_frac=1.0)

    def on_order_rejected(self, ctx, event):
        self.rejects.append(event.reject_reason)


def _run(capital, spec, direction=-1):
    strategy = _SellOnce()
    result = run_portfolio_strategy(
        strategy,
        {SYM: _bars([360.0, 350.0, 340.0, 330.0])},
        config=_config(capital),
        directions={SYM: direction},
        instruments={SYM: spec},
        account_type="margin",
        leverage=1.0,
    )
    return result, strategy


class TestSoldOptionDeposit:
    def test_the_deposit_sizes_the_entry_one_lot_at_four_lakh(self):
        result, _ = _run(400_000.0, _spec())
        trades = result.result.trades()
        assert len(trades) == 1
        # 400,000 / 6,756.25 = 59.2 contracts → exactly one lot of 30, where
        # premium alone (400,000 / 360 = 1,111) would have sold 37 lots.
        assert trades[0].size == pytest.approx(LOT)

    def test_one_lakh_cannot_carry_the_deposit_and_says_margin(self):
        result, strategy = _run(100_000.0, _spec())
        assert result.result.trades() == []
        assert strategy.rejects, "the refusal must be surfaced, not silently skipped"
        assert all(r.lower() == "insufficientmargin" for r in strategy.rejects), strategy.rejects

    def test_the_zero_default_still_funds_a_sold_option_at_its_premium(self):
        """Unchanged behaviour: without the model ₹1 lakh sells nine lots."""
        result, strategy = _run(100_000.0, _spec(span_pct=0.0, exposure_pct=0.0))
        trades = result.result.trades()
        assert len(trades) == 1
        assert trades[0].size == pytest.approx(9 * LOT)
        assert strategy.rejects == []

    def test_a_bought_option_is_funded_at_its_premium_regardless(self):
        result, _ = _run(400_000.0, _spec(), direction=1)
        trades = result.result.trades()
        assert len(trades) == 1
        # 400,000 / 360 = 1,111 contracts → 37 lots; the deposit model is for
        # the seller, a buyer can only lose the premium.
        assert trades[0].size == pytest.approx(37 * LOT)


class TestSpecSurface:
    def test_the_percentages_are_readable_and_default_to_zero(self):
        assert _spec().span_pct == SPAN
        assert _spec().exposure_pct == EXPOSURE
        plain = InstrumentSpec.option(SYM, strike=STRIKE, right="call", expiration_ns=1, lot_size=LOT)
        assert plain.span_pct == 0.0 and plain.exposure_pct == 0.0

    def test_negative_percentages_are_refused(self):
        with pytest.raises(ValueError):
            _spec(span_pct=-0.1)
        with pytest.raises(ValueError):
            _spec(exposure_pct=-0.1)
