"""Portfolio-level risk gates in session runs (0.5.0).

``max_positions`` and ``max_drawdown_pct`` apply across all instruments in
``run_portfolio_strategy``, matching the array runner's long-standing
behavior. Previously each instrument carried its own independent copy of the
gate, so a limit of N allowed N positions *per symbol*.
"""

import numpy as np

from raptorbt import PyBacktestConfig, Strategy, run_portfolio_strategy


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


def _config(**kwargs):
    config = PyBacktestConfig(**kwargs)
    config.fees = 0.0
    return config


class _EnterEverything(Strategy):
    """Signal an entry on every bar of every symbol."""

    def __init__(self, config=None, size_frac=0.2):
        super().__init__(config)
        self.size_frac = size_frac
        self.rejects = []
        self.max_concurrent = 0

    def on_bar(self, ctx):
        self.enter(size_frac=self.size_frac)
        open_now = sum(
            1 for symbol in ("AAA", "BBB", "CCC") if ctx.position_for(symbol) is not None
        )
        self.max_concurrent = max(self.max_concurrent, open_now)

    def on_order_rejected(self, ctx, event):
        self.rejects.append((ctx.symbol, event.reject_reason))


def _three_symbols():
    return {
        "AAA": _bars([100.0, 101.0, 102.0, 103.0]),
        "BBB": _bars([50.0, 50.5, 51.0, 51.5], start_ts=5_000_000_000),
        "CCC": _bars([25.0, 25.5, 26.0, 26.5], start_ts=9_000_000_000),
    }


class TestPortfolioMaxPositions:
    def test_max_positions_is_portfolio_wide(self):
        strategy = _EnterEverything()
        result = run_portfolio_strategy(
            strategy, _three_symbols(), config=_config(max_positions=2)
        )

        assert strategy.max_concurrent <= 2, (
            "max_positions=2 must cap concurrent positions across all symbols"
        )
        assert result.rejected_entries > 0
        assert result.rejected_entries == sum(
            s.rejected_entries for s in result.per_instrument
        )

    def test_unconstrained_allows_every_symbol(self):
        strategy = _EnterEverything()
        result = run_portfolio_strategy(strategy, _three_symbols(), config=_config())

        assert strategy.max_concurrent == 3, "all three symbols should hold at once"
        assert result.rejected_entries == 0

    def test_on_order_rejected_reports_max_positions(self):
        strategy = _EnterEverything()
        run_portfolio_strategy(
            strategy, _three_symbols(), config=_config(max_positions=1)
        )

        # Signal-path rejections surface the Debug form of the reason;
        # resting-order rejections surface the snake_case form.
        assert any(
            reason in ("MaxPositions", "max_positions")
            for _, reason in strategy.rejects
        ), f"expected a max_positions rejection, got {strategy.rejects}"

    def test_tighter_limit_opens_fewer_positions(self):
        loose = _EnterEverything()
        run_portfolio_strategy(loose, _three_symbols(), config=_config(max_positions=3))
        tight = _EnterEverything()
        run_portfolio_strategy(tight, _three_symbols(), config=_config(max_positions=1))

        assert tight.max_concurrent < loose.max_concurrent


class TestPortfolioDrawdownHalt:
    def test_drawdown_kill_switch_halts_portfolio(self):
        class S(Strategy):
            def __init__(self, config=None):
                super().__init__(config)
                self.rejects = []

            def on_bar(self, ctx):
                # AAA goes all-in immediately; BBB keeps trying afterward.
                if ctx.symbol == "AAA" and ctx.idx == 0:
                    self.enter(size_frac=1.0)
                if ctx.symbol == "BBB":
                    self.enter(size_frac=0.1)

            def on_order_rejected(self, ctx, event):
                self.rejects.append((ctx.symbol, event.reject_reason))

        data = {
            "AAA": _bars([100.0, 50.0, 40.0, 40.0]),
            "BBB": _bars([50.0, 50.0, 50.0, 50.0], start_ts=5_000_000_000),
        }
        strategy = S()
        result = run_portfolio_strategy(
            strategy, data, config=_config(max_drawdown_pct=15.0)
        )

        assert result.halted is True
        assert result.halted_at is not None
        # The untouched symbol is halted by the portfolio-level gate, and the
        # reason is the drawdown, not a margin call.
        assert any(
            symbol == "BBB" and reason == "DrawdownHalt"
            for symbol, reason in strategy.rejects
        ), f"expected a drawdown rejection on BBB, got {strategy.rejects}"
        assert not any(
            reason == "MarginCall" for _, reason in strategy.rejects
        ), "a cash-account drawdown halt must not report a margin call"

    def test_no_halt_on_a_clean_run(self):
        strategy = _EnterEverything()
        result = run_portfolio_strategy(
            strategy, _three_symbols(), config=_config(max_drawdown_pct=50.0)
        )

        assert result.halted is False
        assert result.halted_at is None


def test_session_and_array_paths_agree_on_max_positions():
    """The two portfolio APIs must enforce the same limit the same way."""

    strategy = _EnterEverything()
    constrained = run_portfolio_strategy(
        strategy, _three_symbols(), config=_config(max_positions=1)
    )
    unconstrained_strategy = _EnterEverything()
    unconstrained = run_portfolio_strategy(
        unconstrained_strategy, _three_symbols(), config=_config()
    )

    assert constrained.rejected_entries > 0
    assert unconstrained.rejected_entries == 0
    assert strategy.max_concurrent < unconstrained_strategy.max_concurrent
