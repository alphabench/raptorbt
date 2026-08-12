"""Shared margin accounts in portfolio runs (0.5.0).

The single-instrument margin path is covered by
``test_phase3.py::TestMarginAccount``; these pin the portfolio case, where
one account funds N instruments: leverage is shared, equity marking is
direction-aware, and a margin call halts every instrument at once.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, Strategy, run_portfolio_strategy


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


def _zero_fee_config(**kwargs):
    config = BacktestConfig(**kwargs)
    config.fees = 0.0
    return config


class _EnterOnce(Strategy):
    """Enter every symbol once, on its first bar, at a fixed fraction."""

    def __init__(self, config=None, size_frac=0.25):
        super().__init__(config)
        self.size_frac = size_frac
        self.sizes = {}

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.enter(size_frac=self.size_frac)

    def on_position_opened(self, ctx, event):
        self.sizes[ctx.symbol] = event.size


class TestPortfolioMargin:
    def test_margin_scales_sizing(self):
        data = {
            "AAA": _bars([100.0, 101.0, 102.0]),
            "BBB": _bars([50.0, 51.0, 52.0], start_ts=5_000_000_000),
        }

        cash_strategy = _EnterOnce()
        run_portfolio_strategy(cash_strategy, data, config=_zero_fee_config())

        margin_strategy = _EnterOnce()
        run_portfolio_strategy(
            margin_strategy,
            data,
            config=_zero_fee_config(),
            account_type="margin",
            leverage=5.0,
        )

        assert set(cash_strategy.sizes) == {"AAA", "BBB"}
        # 5x leverage buys 5x the units for the first instrument's slice.
        assert margin_strategy.sizes["AAA"] == pytest.approx(
            5.0 * cash_strategy.sizes["AAA"], rel=1e-9
        )

    def test_shared_account_funds_both_symbols(self):
        """Locks reserve capital without debiting it, so the balance stays
        whole and the second symbol still sizes off portfolio free capital."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.seen = []

            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=0.25)
                if ctx.idx == 2:
                    self.seen.append((ctx.symbol, ctx.position is not None))

        data = {
            "AAA": _bars([100.0, 101.0, 102.0]),
            "BBB": _bars([50.0, 51.0, 52.0], start_ts=5_000_000_000),
        }
        strategy = S()
        result = run_portfolio_strategy(
            strategy,
            data,
            config=_zero_fee_config(),
            account_type="margin",
            leverage=5.0,
        )

        assert len(strategy.seen) == 2
        assert all(has_position for _, has_position in strategy.seen)
        summary = {s.symbol: s for s in result.per_instrument}
        assert summary["AAA"].trades == 1
        assert summary["BBB"].trades == 1

    def test_margin_marks_short_correctly(self):
        """A winning short must raise portfolio equity.

        Cash-mode marking adds the short's *position value*, which falls as
        the price falls — reporting the gain as a loss.
        """

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=0.25)

        # Both symbols fall; BBB is short, so it profits.
        data = {
            "AAA": _bars([100.0, 100.0, 100.0]),
            "BBB": _bars([50.0, 45.0, 40.0], start_ts=5_000_000_000),
        }
        result = run_portfolio_strategy(
            S(),
            data,
            config=_zero_fee_config(),
            directions={"AAA": 1, "BBB": -1},
            account_type="margin",
            leverage=2.0,
        )

        summary = {s.symbol: s for s in result.per_instrument}
        assert summary["BBB"].pnl > 0, "the short profited from the decline"
        curve = result.result.equity_curve()
        assert curve[-1] > curve[0], "portfolio equity must reflect the gain"

    def test_margin_call_halts_all_symbols(self):
        """One shared account: a call on one symbol blocks entries on the
        symbol that never traded."""

        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.margin_calls = 0
                self.rejects = []

            def on_bar(self, ctx):
                # AAA takes a position immediately; BBB keeps trying later,
                # after the collapse has already tripped the account.
                if ctx.symbol == "AAA" and ctx.idx == 0:
                    self.enter(size_frac=1.0)
                if ctx.symbol == "BBB" and ctx.idx >= 2:
                    self.enter(size_frac=0.1)

            def on_margin_call(self, ctx, event):
                self.margin_calls += 1

            def on_order_rejected(self, ctx, event):
                self.rejects.append((ctx.symbol, event.reject_reason))

        data = {
            "AAA": _bars([100.0, 40.0, 30.0, 30.0]),
            "BBB": _bars([50.0, 50.0, 50.0, 50.0], start_ts=5_000_000_000),
        }
        strategy = S()
        result = run_portfolio_strategy(
            strategy,
            data,
            config=_zero_fee_config(),
            account_type="margin",
            leverage=50.0,
        )

        assert strategy.margin_calls >= 1
        assert result.halted is True
        assert result.halted_at is not None
        # The untouched symbol was halted by the shared account.
        assert any(
            symbol == "BBB" and reason == "MarginCall"
            for symbol, reason in strategy.rejects
        )

    def test_reports_rejected_entries(self):
        """Regression: the portfolio total was hardcoded to 0."""

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.symbol == "AAA" and ctx.idx == 0:
                    self.enter(size_frac=1.0)
                if ctx.symbol == "BBB" and ctx.idx >= 2:
                    self.enter(size_frac=0.1)

        data = {
            "AAA": _bars([100.0, 40.0, 30.0, 30.0]),
            "BBB": _bars([50.0, 50.0, 50.0, 50.0], start_ts=5_000_000_000),
        }
        result = run_portfolio_strategy(
            S(),
            data,
            config=_zero_fee_config(),
            account_type="margin",
            leverage=50.0,
        )

        assert result.rejected_entries > 0
        assert result.rejected_entries == sum(
            s.rejected_entries for s in result.per_instrument
        )

    def test_cash_portfolio_unchanged(self):
        """Default (cash) runs must be bit-identical to the single-pool
        implementation the shared account replaced."""

        class S(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 0:
                    self.enter(size_frac=0.4)

        data = {
            "AAA": _bars([100.0, 101.0, 102.0, 103.0]),
            "BBB": _bars([50.0, 50.5, 51.0, 51.5], start_ts=5_000_000_000),
        }
        explicit = run_portfolio_strategy(
            S(), data, config=_zero_fee_config(), account_type="cash"
        )
        default = run_portfolio_strategy(S(), data, config=_zero_fee_config())

        assert np.array_equal(
            np.asarray(default.result.equity_curve()),
            np.asarray(explicit.result.equity_curve()),
        )
        # Pinned from the pre-shared-account implementation: 40% of 100k,
        # then 40% of the 60k remainder.
        assert default.result.equity_curve()[0] == pytest.approx(100_000.0, rel=1e-9)
        assert default.halted is False
        assert default.halted_at is None

    def test_rejects_invalid_account_type(self):
        data = {"AAA": _bars([100.0, 101.0])}
        with pytest.raises(ValueError, match="account_type must be"):
            run_portfolio_strategy(
                _EnterOnce(), data, config=_zero_fee_config(), account_type="futures"
            )
        with pytest.raises(ValueError, match="leverage must be > 0"):
            run_portfolio_strategy(
                _EnterOnce(),
                data,
                config=_zero_fee_config(),
                account_type="margin",
                leverage=0.0,
            )
