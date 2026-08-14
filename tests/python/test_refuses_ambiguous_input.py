"""Inputs the engine cannot interpret must refuse, not guess.

Plain words: if you hand raptorbt something it does not understand -- an option
type it cannot parse, a direction that is neither long nor short, a correlation
matrix that is not mathematically valid -- it now raises an error instead of
quietly picking something and returning numbers that look fine.

Every case below silently produced a plausible wrong answer through 0.6.4. They
share one shape, and it is the most dangerous shape a financial library has: not
a crash, not an obviously silly figure, but a smooth well-formed result computed
from something other than what the caller asked for. Nothing downstream -- no
metric, no equity curve, no risk check -- could tell the difference.
"""

import numpy as np
import pytest

import raptorbt as r

N = 300


@pytest.fixture
def series():
    close = 1000 + np.cumsum(np.random.default_rng(2).normal(0, 1, N))
    ts = np.arange(N, dtype=np.int64) * 60_000_000_000
    entries = np.zeros(N, bool)
    exits = np.zeros(N, bool)
    entries[10] = True
    exits[200] = True
    return ts, close, np.full(N, 1e5), entries, exits


class TestDirection:
    """`direction` must be 1 or -1.

    A book encoded 0/1 instead of -1/1 -- a natural "flat or long" convention --
    used to backtest entirely long, flipping the sign of the P&L on every short
    while producing a perfectly well-formed equity curve.
    """

    @pytest.mark.parametrize("direction", [1, -1])
    def test_valid_directions_are_accepted(self, series, direction):
        ts, c, v, e, x = series
        r.run_single_backtest(timestamps=ts, open=c, high=c, low=c, close=c, volume=v,
                              entries=e, exits=x, direction=direction)

    @pytest.mark.parametrize("direction", [0, 2, -2, 100])
    def test_anything_else_is_refused(self, series, direction):
        ts, c, v, e, x = series
        with pytest.raises(ValueError, match="direction must be 1"):
            r.run_single_backtest(timestamps=ts, open=c, high=c, low=c, close=c, volume=v,
                                  entries=e, exits=x, direction=direction)


class TestOptionType:
    """An unparseable option type must not become a Call.

    ``OptionType::from_code`` documents this in so many words -- "defaulting an
    unrecognised code to Call would price a put as a call" -- and both PyO3 call
    sites did exactly that. An iron condor whose two put legs failed to parse
    became a four-leg call structure with the wrong payoff and no error.
    """

    def _prem(self):
        return [np.abs(30 + np.arange(N) * 0.01), np.abs(20 + np.arange(N) * 0.005)]

    @pytest.mark.parametrize("code", ["CE", "ce", "CALL", "C", "PE", "pe", "PUT", "P"])
    def test_known_codes_are_accepted(self, series, code):
        ts, c, _, e, x = series
        r.run_spread_backtest(timestamps=ts, underlying_close=c, legs_premiums=self._prem(),
                              leg_configs=[(code, 1000.0, -1, 75), (code, 950.0, 1, 75)],
                              entries=e, exits=x)

    @pytest.mark.parametrize("code", ["XX", "pe ", "Put Option", "", "P/E"])
    def test_unknown_codes_are_refused(self, series, code):
        ts, c, _, e, x = series
        with pytest.raises(ValueError, match="unknown option type"):
            r.run_spread_backtest(timestamps=ts, underlying_close=c, legs_premiums=self._prem(),
                                  leg_configs=[(code, 1000.0, -1, 75), (code, 950.0, 1, 75)],
                                  entries=e, exits=x)

    def test_batch_refuses_too(self, series):
        """batch_spread_backtest multiplies the mistake across a whole sweep."""
        ts, c, _, e, x = series
        item = r.BatchSpreadItem("s0", self._prem(),
                                 [("XX", 1000.0, -1, 75), ("XX", 950.0, 1, 75)],
                                 e, x, "custom", None, None)
        with pytest.raises(ValueError, match="unknown option type"):
            r.batch_spread_backtest(ts, c, [item])


class TestOptionsRunnerEnums:
    """`run_options_backtest` parsed its three enums with a catch-all arm.

    ``option_type="PUT"`` selected a long CALL -- roughly the mirror image of the
    intended payoff -- while the identical string is accepted by
    ``run_spread_backtest``. The same call meant two different things depending
    on which function you entered through.
    """

    def _args(self, series):
        ts, c, v, e, x = series
        return dict(timestamps=ts, open=c, high=c, low=c, close=c, volume=v,
                    option_prices=np.abs(30 + np.arange(N) * 0.01), entries=e, exits=x)

    @pytest.mark.parametrize("value", ["put", "PUT", "Put", "pe", "PE", "call", "CE"])
    def test_option_type_is_case_insensitive(self, series, value):
        r.run_options_backtest(**self._args(series), option_type=value)

    @pytest.mark.parametrize("value", ["puts", "long_put", "xx", ""])
    def test_unknown_option_type_is_refused(self, series, value):
        with pytest.raises(ValueError, match="unknown option_type"):
            r.run_options_backtest(**self._args(series), option_type=value)

    @pytest.mark.parametrize("value", ["otm_1", "OTM", "atm2", "nearest"])
    def test_unknown_strike_selection_is_refused(self, series, value):
        with pytest.raises(ValueError, match="unknown strike_selection"):
            r.run_options_backtest(**self._args(series), strike_selection=value)

    @pytest.mark.parametrize("value", ["pct", "percentage", "lots"])
    def test_unknown_size_type_is_refused(self, series, value):
        with pytest.raises(ValueError, match="unknown size_type"):
            r.run_options_backtest(**self._args(series), size_type=value)

    def test_documented_defaults_still_work(self, series):
        """The defaults must match the new explicit arms, or every call breaks."""
        r.run_options_backtest(**self._args(series))


class TestMonteCarloCorrelation:
    """A correlation matrix that is not positive definite must be refused.

    The Cholesky routine used to patch a negative pivot with ``sqrt(|diag|)`` and
    a zero pivot with ``0.0``, then return Ok. The simulation ran against a
    correlation structure the caller never supplied. Measured on the indefinite
    matrix below (smallest eigenvalue -0.8), the repaired decomposition returned
    ``var_95 = 0`` and ``probability_of_loss = 0`` -- a risk model reporting no
    risk whatsoever, from input it should have rejected.
    """

    def _returns(self, n=3):
        rng = np.random.default_rng(3)
        return [rng.normal(0.0004, 0.011, 400) for _ in range(n)]

    def test_indefinite_matrix_is_refused(self):
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        assert np.linalg.eigvalsh(bad).min() < 0, "fixture must be indefinite"
        with pytest.raises(ValueError, match="not positive definite"):
            r.simulate_portfolio_mc(self._returns(), np.full(3, 1 / 3), bad, 1e6, 500, 252, 42)

    def test_singular_matrix_is_refused(self):
        """Two perfectly collinear assets have no valid decomposition."""
        dup = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="not positive definite|singular"):
            r.simulate_portfolio_mc(self._returns(), np.full(3, 1 / 3), dup, 1e6, 500, 252, 42)

    def test_valid_matrix_still_simulates(self):
        """A well-formed matrix must produce a real distribution, not an error.

        Uses a negative-drift book so the loss tail is populated. Note that
        `var_95 = 0.0` is a legitimate result for a book that gains on every
        path -- GBM's `exp(drift + diffusion)` is positively skewed, so even
        zero-mean returns compound upward over 252 steps. Asserting a non-zero
        VaR only proves anything against inputs that can actually lose.
        """
        rng = np.random.default_rng(3)
        losing = [rng.normal(-0.002, 0.011, 400) for _ in range(3)]
        out = r.simulate_portfolio_mc(losing, np.full(3, 1 / 3), np.eye(3), 1e6, 2000, 252, 42)

        assert len(out["final_values"]) == 2000
        assert out["var_95"] > 0.0, "a losing book must show loss at the 95% level"
        assert out["cvar_95"] >= out["var_95"], "CVaR is the mean beyond VaR"
        assert out["probability_of_loss"] > 0.0


class TestTickTruncation:
    """`max_trades` must not silently truncate by default.

    On a 1,000,000-tick input the old default of 50 reported metrics covering
    0.81% of the tape: total return -0.12% against a true -14.13%, and a max
    drawdown of 0.124% against a true 14.13%.
    """

    def _tape(self, n):
        rng = np.random.default_rng(11)
        ltp = np.exp(np.clip(np.cumsum(rng.normal(0, 0.00005, n)), -0.5, 0.5) + np.log(1000.0))
        half = ltp * 0.00025
        ret = np.zeros(n)
        ret[60:] = (ltp[60:] / ltp[:-60] - 1.0) * 100
        return dict(
            timestamps=np.arange(n, dtype=np.int64) * 1_000_000,
            ltp=ltp, bid=ltp - half, ask=ltp + half,
            buy_qty_delta=np.abs(rng.normal(500, 150, n)),
            sell_qty_delta=np.abs(rng.normal(500, 150, n)),
            oi=np.abs(rng.normal(1e6, 1e4, n)),
            entries=ret > 0.02, exits=ret < -0.02, symbol="T")

    def test_default_traverses_the_whole_tape(self):
        n = 200_000
        res = r.run_tick_backtest(**self._tape(n), entry_cooldown_ticks=10)
        trades = res.trades()
        assert len(trades) > 50, "default must not stop at the old 50-trade cap"
        assert trades[-1].exit_idx >= n - 1000, (
            f"run ended at tick {trades[-1].exit_idx} of {n}; it truncated")

    def test_explicit_cap_still_truncates(self):
        """The knob must keep working, or the default change is a removal."""
        res = r.run_tick_backtest(**self._tape(200_000), entry_cooldown_ticks=10, max_trades=50)
        assert len(res.trades()) == 50


class TestTickPositionSize:
    """A tick backtest must charge and earn on the position actually traded.

    Plain words: through 0.7.3 this path priced everything as if you had traded
    exactly one unit, whatever your real lot size. It charged one unit's costs,
    earned one unit's profit, and then measured that profit against your whole
    account. A 75-lot option was off by roughly 75x on both sides.

    These pin the Python seam: the arguments exist, they reach the engine, and
    the itemized regulatory schedule is reachable from this path at all -- it
    never was before, because `fee_segment` was not on the signature.
    """

    def _flat_tape(self, n=6, price=100.0):
        entries = np.zeros(n, dtype=bool)
        entries[1] = True
        exits = np.zeros(n, dtype=bool)
        exits[4] = True
        flat = np.full(n, price)
        return dict(
            timestamps=np.arange(n, dtype=np.int64) * 1_000_000_000,
            ltp=flat, bid=flat.copy(), ask=flat.copy(),
            buy_qty_delta=np.zeros(n), sell_qty_delta=np.zeros(n), oi=np.zeros(n),
            entries=entries, exits=exits, symbol="T",
            stop_loss_pct=50.0, take_profit_pct=50.0,
            max_hold_seconds=0, entry_cooldown_ticks=0)

    def test_costs_scale_with_lot_size(self):
        one = r.run_tick_backtest(**self._flat_tape(), fees=0.001, lot_size=1, quantity=1)
        lot = r.run_tick_backtest(**self._flat_tape(), fees=0.001, lot_size=75, quantity=1)

        assert one.trades()[0].fees == pytest.approx(0.20)
        assert lot.trades()[0].fees == pytest.approx(15.00), (
            "a 75-lot round trip on a 100 premium costs 75x one unit")

    def test_defaults_reproduce_the_pre_074_numbers(self):
        """Upgrading must not change anyone's results silently."""
        res = r.run_tick_backtest(**self._flat_tape(), fees=0.001)
        trade = res.trades()[0]
        assert trade.size == 1.0
        assert trade.fees == pytest.approx(0.20), "price * rate, both sides, one unit"

    def test_fee_segment_reaches_the_tick_path(self):
        """Brokerage is flat per order, so no percentage rate can express it."""
        flat = r.run_tick_backtest(**self._flat_tape(), fees=0.001, lot_size=75, quantity=1)
        itemized = r.run_tick_backtest(
            **self._flat_tape(), fees=0.001, lot_size=75, quantity=1, fee_segment="NFO-OPT")

        assert itemized.trades()[0].fees > flat.trades()[0].fees, (
            "the real schedule adds 2 x Rs 20 brokerage plus GST")
        entry = itemized.trades()[0].entry_fees
        exit_ = itemized.trades()[0].exit_fees
        assert entry != exit_, "stamp duty falls on the buy, transaction tax on the sell"

    def test_a_short_is_refused_not_silently_run_long(self):
        with pytest.raises(ValueError, match="long-only"):
            r.run_tick_backtest(**self._flat_tape(), quantity=-1)


def test_leg_expiry_timestamps_must_match_the_legs():
    """One expiry per leg, matched by position -- or refuse.

    A short list would leave the trailing legs immortal and a long one
    would settle on a date belonging to no leg. Both are silent, and a
    silently mis-settled spread reports a clean-looking P&L for a trade
    nobody made.
    """
    n = 6
    timestamps = (
        np.arange(n, dtype=np.int64) * 300_000_000_000 + 1_786_000_000_000_000_000
    )
    entries = np.zeros(n, dtype=bool)
    entries[1] = True

    with pytest.raises(ValueError, match="leg_expiry_timestamps"):
        r.run_spread_backtest(
            timestamps=timestamps,
            underlying_close=np.full(n, 24_550.0),
            legs_premiums=[np.full(n, 50.0), np.full(n, 80.0)],
            leg_configs=[("CE", 24_800.0, -1, 75), ("CE", 24_800.0, 1, 75)],
            entries=entries,
            exits=np.zeros(n, dtype=bool),
            config=r.BacktestConfig(initial_capital=500_000.0),
            # Two legs, one expiry.
            leg_expiry_timestamps=[int(timestamps[4])],
        )
