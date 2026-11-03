"""Defects found while writing the skills: one test per defect, each failing for the stated reason.

A halt that flattens books one trade at the last mark; a dry run needs no ``bar_stamp``;
an unknown ``combine_mode`` and an unknown order id are refused; ``squareoff_time`` is
honoured by every runner that has a session; a bracket under hedging is refused; and
``ctx.positions()`` and ``MarginModel.fraction_of_notional`` take one form each.
"""

import warnings

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    MarginModel,
    Strategy,
    run_multi_backtest,
    run_portfolio_strategy,
    run_single_backtest,
    run_strategies,
    run_strategy_backtest,
)
from raptorbt.errors import InvalidOrder, InvalidParameter
from raptorbt.strategy import orders

HOUR = 3600 * 10**9
MIN = 60 * 10**9


def _bars(closes, step=HOUR):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(n, dtype=np.int64) * step,
        "open": closes.copy(),
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _trades(res):
    return getattr(res, "result", res).trades()


# -- 1. a halt that flattens ------------------------------------------------------------


class HaltsAfterBuying(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", units=10.0))
        if ctx.idx == 3:
            ctx.halt(flatten=True)


@pytest.mark.parametrize("oms", ["netting", "hedging"])
@pytest.mark.parametrize("runner", ["single", "shared"])
def test_a_flattening_halt_books_one_trade_at_the_close_of_its_bar(runner, oms):
    data = _bars([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    config = BacktestConfig(fees=0.0)
    if runner == "single":
        res = run_strategy_backtest(
            HaltsAfterBuying(), **data, symbol="S", config=config, oms_type=oms, bar_stamp="close"
        )
    else:
        res = run_strategies(
            [HaltsAfterBuying()], {"S": data}, config, oms_type=oms, bar_stamp="close"
        )[0]
    # Corrected (was bar 2's close, 100.0, a bar early): halted while deciding bar 3, it
    # closes as an ordinary reduce-only market order would, at bar 3's close.
    assert [(t.entry_idx, t.exit_idx, t.exit_price) for t in _trades(res)] == [(1, 3, 90.0)]


# -- 2. the requirements dry run ----------------------------------------------------------


def test_the_dry_run_passes_under_warnings_as_errors():
    from raptorbt.strategy import Holding, Requirements, Underlying, check_requirements
    from raptorbt import Indicator

    class Indicated(Strategy):
        requirements = Requirements(
            roles={"index": Underlying(asset="index")},
            timeframes=["1m"],
            holding=Holding.overnight(),
        )

        def on_init(self, ctx):
            self.subscribe("INDEX")
            # No symbol=: a portfolio run warns about this, and the probe must not.
            self.register_indicator(Indicator.sma(3))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_requirements(Indicated) == []


# -- 3. combine_mode ----------------------------------------------------------------------


def test_run_multi_backtest_refuses_an_unknown_combine_mode():
    d = _bars(np.linspace(100.0, 110.0, 20))
    args = (d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"], [])
    with pytest.raises(InvalidParameter) as e:
        run_multi_backtest(*args, combine_mode="bogus")
    assert e.value.code == "invalid_combine_mode"
    assert "any" in e.value.valid
    for mode in ("any", "all", "majority", "independent", "weighted"):
        run_multi_backtest(*args, combine_mode=mode)


# -- 4. squareoff_time ----------------------------------------------------------------------


def _session_day(minutes_after_open=40):
    """One UTC day of minute bars, 15:01 to 15:40, stamped at their close."""
    ts = (15 * 60 + np.arange(1, minutes_after_open + 1, dtype=np.int64)) * MIN
    c = np.full(minutes_after_open, 100.0)
    return {
        "timestamps": ts,
        "open": c.copy(),
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
        "volume": np.full(minutes_after_open, 1_000.0),
    }


class BuysAt(Strategy):
    def __init__(self, *at):
        super().__init__()
        self.at = set(at)
        self.rejected = []

    def on_bar(self, ctx):
        if ctx.idx in self.at:
            self.submit_order(orders.Market(side="buy", units=1.0))

    def on_order_rejected(self, ctx, event):
        self.rejected.append((event.idx, event.reject_reason))


def _exits(res):
    return [(t.exit_idx, t.exit_reason) for t in _trades(res)]


@pytest.mark.parametrize("runner", ["single", "shared", "portfolio", "array"])
def test_squareoff_time_closes_the_position_in_every_runner(runner):
    d = _session_day()
    # The day is UTC clock time: name the zone, as a squareoff_time needs one.
    config = BacktestConfig(fees=0.0, squareoff_time="15:25", session_tz_offset_ns=0)
    if runner == "single":
        res = run_strategy_backtest(BuysAt(2), **d, symbol="S", config=config, bar_stamp="close")
    elif runner == "shared":
        res = run_strategies([BuysAt(2)], {"S": d}, config, bar_stamp="close")[0]
    elif runner == "portfolio":
        res = run_portfolio_strategy(BuysAt(2), {"S": d}, config=config, bar_stamp="close")
    else:
        entries = np.zeros(40, dtype=bool)
        entries[2] = True
        res = run_single_backtest(
            d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"],
            entries, np.zeros(40, dtype=bool), 1, 1.0, "S", config,
        )
    # The bar stamped 15:25 is index 24: the first at or after the cutoff.
    assert _exits(res) == [(24, "Squareoff")]


def test_squareoff_time_refuses_a_second_entry_the_same_day():
    s = BuysAt(2, 30)
    run_strategy_backtest(
        s, **_session_day(), symbol="S",
        config=BacktestConfig(fees=0.0, squareoff_time="15:25", session_tz_offset_ns=0),
        bar_stamp="close",
    )
    assert s.rejected == [(30, "squareoff")]


def test_without_squareoff_time_the_position_runs_to_the_end():
    res = run_strategy_backtest(
        BuysAt(2), **_session_day(), symbol="S", config=BacktestConfig(fees=0.0), bar_stamp="close"
    )
    assert _exits(res) == [(39, "EndOfData")]


# -- 5. a bracket under hedging ---------------------------------------------------------------


class Bracketed(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_bracket(
                orders.Market(side="buy", units=1.0), stop_trigger=90.0, target_price=105.0
            )


def test_a_bracket_under_hedging_is_refused_at_submit():
    data = _bars([100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0])
    with pytest.raises(InvalidOrder) as e:
        run_strategy_backtest(
            Bracketed(), **data, symbol="S", config=BacktestConfig(fees=0.0),
            oms_type="hedging", bar_stamp="close",
        )
    assert e.value.code == "bracket_under_hedging"
    res = run_strategy_backtest(
        Bracketed(), **data, symbol="S", config=BacktestConfig(fees=0.0),
        oms_type="netting", bar_stamp="close",
    )
    assert _exits(res) == [(5, "Order")]


# -- 6. cancel and modify of an id -----------------------------------------------------------


class Cancels(Strategy):
    def __init__(self, ghost):
        super().__init__()
        self.ghost = ghost

    def on_bar(self, ctx):
        if ctx.idx == 2:
            self.submit_order(orders.Market(side="buy", units=1.0), client_id="m")
        if ctx.idx == 4:
            # "m" filled on bar 2: asking again is the documented no-op.
            self.cancel_order("m")
            self.modify_order("m", limit_price=1.0)
        if ctx.idx == 5 and self.ghost:
            getattr(self, self.ghost)("nope")


@pytest.mark.parametrize("call", ["cancel_order", "modify_order"])
def test_an_id_that_was_never_submitted_is_refused(call):
    data = _bars([100.0] * 8)
    with pytest.raises(InvalidOrder) as e:
        run_strategy_backtest(
            Cancels(call), **data, symbol="S", config=BacktestConfig(fees=0.0), bar_stamp="close"
        )
    assert e.value.code == "unknown_order"
    assert e.value.value == "nope" and e.value.valid == ["m"]


def test_a_finished_order_is_a_no_op():
    run_strategy_backtest(
        Cancels(None), **_bars([100.0] * 8), symbol="S", config=BacktestConfig(fees=0.0),
        bar_stamp="close",
    )


# -- 7. ctx.positions ---------------------------------------------------------------------------


class ReadsPositions(Strategy):
    def __init__(self):
        super().__init__()
        self.called = None
        self.as_list = None

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", units=1.0))
        if ctx.idx == 3:
            self.called = [p.size for p in ctx.positions()]
            with pytest.warns(DeprecationWarning, match="ctx.positions"):
                self.as_list = [p.size for p in ctx.positions]


def test_positions_is_called_in_a_single_run_and_the_list_form_warns():
    s = ReadsPositions()
    run_strategy_backtest(
        s, **_bars([100.0] * 6), symbol="S", config=BacktestConfig(fees=0.0), bar_stamp="close"
    )
    assert s.called == [1.0] and s.as_list == [1.0]


def test_positions_is_called_the_same_way_in_a_portfolio_run():
    seen = []

    class Portfolio(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 1:
                self.submit_order(orders.Market(side="buy", units=1.0))
            if ctx.idx == 3:
                seen.append([p.size for p in ctx.positions()])
                seen.append([p.size for p in ctx.positions("S")])

    run_strategies([Portfolio()], {"S": _bars([100.0] * 6)}, BacktestConfig(fees=0.0), bar_stamp="close")
    assert seen == [[1.0], [1.0]]


# -- 8. MarginModel fractions ---------------------------------------------------------------------


def test_fraction_of_notional_takes_a_fraction_and_percent_warns():
    spec = raptorbt.InstrumentSpec.futures_contract("FUT", 10**18, 1.0)
    with pytest.warns(DeprecationWarning, match="fraction_of_notional"):
        old = MarginModel.percent_of_notional(10)
    new = MarginModel.fraction_of_notional(0.1)
    assert new.required(spec, 400.0, 1500.0) == old.required(spec, 400.0, 1500.0) == 60_000.0
    for bad in (10, 1.5, -0.1, float("nan")):
        with pytest.raises(InvalidParameter):
            MarginModel.fraction_of_notional(bad)
    assert MarginModel.fraction_of_notional(1.0).required(spec, 1.0, 100.0) == 100.0


# -- 9. run_tick_strategy takes corporate actions ----------------------------------------------------


def test_run_tick_strategy_applies_a_split_to_the_held_position():
    NS = 10**9
    ticks = {
        "A": {
            "timestamps": (np.arange(5) * NS).astype(np.int64),
            "ltp": np.array([100.0, 100.0, 50.0, 50.0, 55.0]),
        }
    }

    class Holds(Strategy):
        def on_trade(self, ctx, trade):
            if trade.timestamp == 0:
                self.submit_order(orders.Market(side="buy", units=10.0))

    result = raptorbt.run_tick_strategy(
        Holds(), ticks, BacktestConfig(fees=0.0),
        corporate_actions=[raptorbt.CorporateAction.split("A", 2 * NS, 2.0)],
        adjustment_policy="raw",
    )
    (t,) = result.result.trades()
    assert t.size == 20.0 and t.entry_price == 50.0
    with pytest.raises(InvalidParameter) as e:
        raptorbt.run_tick_strategy(
            Holds(), ticks, BacktestConfig(fees=0.0),
            corporate_actions=[raptorbt.CorporateAction.split("A", 2 * NS, 2.0)],
        )
    assert e.value.code == "invalid_adjustment_policy"


# -- 9. a pair or basket is held whole or not at all ----------------------------------------------


def _leg(close):
    d = _bars(close)
    return d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"]


def test_a_pair_whose_second_leg_a_limit_refused_is_not_held_on_one_leg():
    closes = np.full(8, 100.0)
    entries = np.zeros(8, dtype=bool)
    entries[1] = True
    exits = np.zeros(8, dtype=bool)
    res = raptorbt.run_pairs_backtest(
        *_leg(closes), *_leg(closes), entries, exits, 1, "PAIR",
        BacktestConfig(fees=0.0, max_positions=1),
    )
    # Leg one entered, leg two was refused by max_positions: leg one is closed on the next bar,
    # not carried to the end of the data as an unhedged position.
    assert [(t.entry_idx, t.exit_idx, t.exit_reason) for t in res.trades()] == [(1, 2, "Signal")]
