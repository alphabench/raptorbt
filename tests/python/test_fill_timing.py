"""Execution-timing causality, exercised through the wheel.

In plain words: a decision made from a bar's data may trade no earlier than
that bar's own close. ``fill_timing="next_bar_open"`` executes it at the
next bar's open. Through 0.10, ``upon_bar_close=False`` filled the decision
at the SAME bar's open — a price from before the decision's information
existed — and that behavior is now only reachable by explicitly asking for
``"same_bar_open_lookahead"``.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig


def _rally_fixture(n=12):
    """One bar rallies 100 -> 200 intrabar; every other price is 150.

    A causal trader learns "bar 5 closed at 200" only when it closes; the
    cheapest price available from then on is 150 and the run ends at 150,
    so no causal strategy acting on this signal can make money.
    """
    ts = np.arange(n, dtype=np.int64) * 1_000_000_000
    o = np.full(n, 150.0)
    h = np.full(n, 150.0)
    l = np.full(n, 150.0)
    c = np.full(n, 150.0)
    o[5], l[5], h[5], c[5] = 100.0, 100.0, 200.0, 200.0
    v = np.full(n, 1000.0)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    entries[5] = True  # emitted because close[5] == 200
    exits[10] = True
    return ts, o, h, l, c, v, entries, exits


def _run(config):
    ts, o, h, l, c, v, entries, exits = _rally_fixture()
    return raptorbt.run_single_backtest(
        ts, o, h, l, c, v, entries, exits, direction=1, config=config
    )


def _no_cost_config(**kwargs):
    return BacktestConfig(fees=0.0, slippage=0.0, **kwargs)


class TestNextBarOpen:
    def test_fills_the_bar_after_the_signal(self):
        result = _run(_no_cost_config(fill_timing="next_bar_open"))
        trades = result.trades()
        assert len(trades) == 1
        # Decision at bar 5 (close 200) fills at bar 6's open (150).
        assert trades[0].entry_idx == 6
        assert trades[0].entry_price == 150.0
        # The causally-impossible ~+50% is gone.
        assert abs(result.metrics.total_return_pct) < 1e-9

    def test_deprecated_bool_false_means_next_bar_open(self):
        result = _run(_no_cost_config(upon_bar_close=False))
        assert result.trades()[0].entry_price == 150.0
        assert result.trades()[0].entry_idx == 6

    def test_explicit_fill_timing_wins_over_the_bool(self):
        cfg = _no_cost_config(upon_bar_close=True, fill_timing="next_bar_open")
        result = _run(cfg)
        assert result.trades()[0].entry_idx == 6


class TestLegacyAndDefaults:
    def test_lookahead_mode_reproduces_pre_0_11_results_by_name_only(self):
        result = _run(_no_cost_config(fill_timing="same_bar_open_lookahead"))
        trade = result.trades()[0]
        assert trade.entry_idx == 5
        assert trade.entry_price == 100.0  # the pre-0.11 look-ahead fill
        assert result.metrics.total_return_pct > 45.0

    def test_default_same_bar_close_is_unchanged(self):
        result = _run(_no_cost_config())
        trade = result.trades()[0]
        assert trade.entry_idx == 5
        assert trade.entry_price == 200.0  # the decision bar's close


class TestConfigSurface:
    def test_invalid_fill_timing_is_refused_loudly(self):
        with pytest.raises(ValueError, match="fill_timing"):
            BacktestConfig(fill_timing="next_open")

    def test_fill_timing_reads_back(self):
        assert BacktestConfig().fill_timing is None
        assert (
            BacktestConfig(fill_timing="next_bar_open").fill_timing
            == "next_bar_open"
        )


class TestPremiumOpenSeries:
    """Optional open premiums for the premium-only runners.

    Under fill_timing="next_bar_open" a signal fill prices at the fill
    bar's OPEN premium when the caller supplies the series; without it the
    bar's settled value remains the fill. Nothing is synthesized.
    """

    def _spot(self, n=10):
        ts = np.arange(n, dtype=np.int64) * 1_000_000_000
        o = np.full(n, 100.0)
        h = np.full(n, 101.0)
        l = np.full(n, 99.0)
        c = np.full(n, 100.0)
        v = np.full(n, 1000.0)
        return ts, o, h, l, c, v

    def test_options_fill_at_the_open_premium(self):
        ts, o, h, l, c, v = self._spot()
        premiums = 10.0 + np.arange(10, dtype=np.float64)
        opens = 100.0 + np.arange(10, dtype=np.float64)
        entries = np.zeros(10, dtype=bool)
        exits = np.zeros(10, dtype=bool)
        entries[3] = True
        exits[6] = True

        result = raptorbt.run_options_backtest(
            ts, o, h, l, c, v, premiums, entries, exits,
            direction=1,
            config=_no_cost_config(fill_timing="next_bar_open"),
            option_open_prices=opens,
        )
        trade = result.trades()[0]
        assert trade.entry_idx == 4
        assert trade.entry_price == 104.0  # bar 4's OPEN premium
        assert trade.exit_price == 107.0

    def test_options_open_series_length_is_validated(self):
        ts, o, h, l, c, v = self._spot()
        premiums = np.full(10, 10.0)
        with pytest.raises(ValueError, match="option_open_prices"):
            raptorbt.run_options_backtest(
                ts, o, h, l, c, v, premiums,
                np.zeros(10, dtype=bool), np.zeros(10, dtype=bool),
                option_open_prices=np.full(9, 10.0),
            )

    def test_spread_fills_legs_at_their_open_premiums(self):
        ts, _, _, _, c, _ = self._spot()
        legs = [10.0 + np.arange(10, dtype=np.float64),
                20.0 + np.arange(10, dtype=np.float64)]
        opens = [100.0 + np.arange(10, dtype=np.float64),
                 200.0 + np.arange(10, dtype=np.float64)]
        entries = np.zeros(10, dtype=bool)
        exits = np.zeros(10, dtype=bool)
        entries[3] = True
        exits[6] = True

        result = raptorbt.run_spread_backtest(
            ts, c, legs,
            [("CE", 100.0, 1, 1), ("PE", 100.0, 1, 1)],
            entries, exits,
            config=_no_cost_config(fill_timing="next_bar_open"),
            spread_type="straddle",
            legs_open_premiums=opens,
        )
        trade = result.trades()[0]
        assert trade.entry_idx == 4
        assert trade.exit_idx == 7
        # Long both legs, open-premium fills: +3 per leg. Settled-value
        # fills would report +6 per leg.
        assert abs(trade.pnl - 6.0) < 1e-9

    def test_spread_open_series_shape_is_validated(self):
        ts, _, _, _, c, _ = self._spot()
        legs = [np.full(10, 10.0), np.full(10, 20.0)]
        with pytest.raises(ValueError, match="legs_open_premiums"):
            raptorbt.run_spread_backtest(
                ts, c, legs,
                [("CE", 100.0, 1, 1), ("PE", 100.0, 1, 1)],
                np.zeros(10, dtype=bool), np.zeros(10, dtype=bool),
                legs_open_premiums=[np.full(10, 10.0)],  # one leg missing
            )
