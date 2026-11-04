"""Optional risk limits and kill switches.

Every limit is off by default; with none set a run is bit-identical to one on
the defaults. Each limit is checked at its exact boundary (equal passes, one
step above is refused), then across the runners that honour it.

Bars here are stamped one per hour on a UTC calendar, so ``DAY`` bars apart is
a day boundary. Fees and slippage are zero, so every figure is exact.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    LiveNode,
    Strategy,
    TickStrategyStream,
    run_portfolio_strategy,
    run_strategies,
    run_strategy_backtest,
)
from raptorbt.strategy import orders

HOUR = 3600 * 10**9
DAY = 24 * HOUR
RISK_REASONS = {"order_size", "max_position", "max_exposure", "daily_loss", "killed"}


def _bars(closes, start=0, step=HOUR):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(start, start + n * step, step, dtype=np.int64),
        "open": closes.copy(),
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _config(**limits):
    return BacktestConfig(fees=0.0, **limits)


class Script(Strategy):
    """Plays a script of actions by bar index and records what comes back.

    ``script`` maps a bar index to a list of actions: ``("buy", units)``,
    ``("sell", units)``, ``("enter", size_frac)``, ``("limit_buy", price,
    units)``, ``("halt", flatten)``. An action may end with a symbol.
    """

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.rejected = []  # (idx, reason) from on_order_rejected
        self.risk = []  # (kind, reason) from on_risk_event
        self.canceled = 0
        self.filled = []

    def on_bar(self, ctx):
        for action in self.script.get(ctx.idx, []):
            kind, *args = action
            symbol = args.pop() if args and isinstance(args[-1], str) else None
            if symbol is not None and ctx.symbol is not None and symbol != ctx.symbol:
                # An order for another instrument, placed from this one's bar, fills on that
                # instrument's next bar; the script means each action once, on its own bar.
                continue
            if kind == "buy":
                self.submit_order(orders.Market(side="buy", units=args[0]), symbol=symbol)
            elif kind == "sell":
                self.submit_order(orders.Market(side="sell", units=args[0]), symbol=symbol)
            elif kind == "enter":
                self.enter(size_frac=args[0])
            elif kind == "limit_buy":
                self.submit_order(
                    orders.Limit(side="buy", price=args[0], units=args[1]), symbol=symbol
                )
            elif kind == "halt":
                ctx.halt(flatten=args[0])
            else:
                raise AssertionError(kind)

    def on_order_rejected(self, ctx, event):
        self.rejected.append((event.idx, event.reject_reason))

    def on_risk_event(self, ctx, event):
        self.risk.append((event.kind, event.reject_reason))

    def on_order_canceled(self, ctx, event):
        self.canceled += 1

    def on_fill(self, ctx, event):
        self.filled.append((event.idx, event.size))


# -- runners: the same strategy through every path that honours the limits ----


def _result(res):
    return getattr(res, "result", res)


def _via_backtest(strategy, data, config, **kw):
    return run_strategy_backtest(strategy, **data, symbol="S", config=config, **kw, bar_stamp="close")


def _via_portfolio(strategy, data, config, **kw):
    return run_portfolio_strategy(strategy, {"S": data}, config=config, **kw, bar_stamp="close")


def _via_shared(strategy, data, config, **kw):
    return run_strategies([strategy], {"S": data}, config, **kw, bar_stamp="close")[0]


def _via_stream(strategy, data, config, **kw):
    stream = TickStrategyStream(strategy, ["S"], config, bar_stamp="close", **kw)
    for i in range(len(data["close"])):
        stream.push_bar("S", *[data[k][i] for k in _COLS])
    return stream.finish()


def _via_node(strategy, data, config, **kw):
    node = LiveNode([strategy], ["S"], config, bar_stamp="close", **kw)
    for i in range(len(data["close"])):
        node.push_bar("S", *[data[k][i] for k in _COLS])
    return node.finish()[0]


_COLS = ["timestamps", "open", "high", "low", "close", "volume"]
RUNNERS = {
    "backtest": _via_backtest,
    "portfolio": _via_portfolio,
    "shared": _via_shared,
    "stream": _via_stream,
    "node": _via_node,
}


@pytest.fixture(params=sorted(RUNNERS))
def run(request):
    return RUNNERS[request.param]


def _sizes(result):
    return [t.size for t in _result(result).trades()]


# -- order size ------------------------------------------------------------------


class TestOrderSize:
    def test_max_order_qty_equal_passes_above_is_refused(self, run):
        data = _bars([100.0] * 4)
        ok = Script({1: [("buy", 10.0)]})
        res = run(ok, data, _config(max_order_qty=10.0))
        assert ok.rejected == [] and ok.filled == [(1, 10.0)]
        assert _sizes(res) == [10.0]

        over = Script({1: [("buy", 10.5)]})
        res = run(over, data, _config(max_order_qty=10.0))
        assert over.rejected == [(1, "order_size")]
        assert over.filled == [] and _sizes(res) == []

    def test_max_order_notional_equal_passes_above_is_refused(self, run):
        data = _bars([100.0] * 4)
        ok = Script({1: [("buy", 10.0)]})  # 10 x 100 = 1000
        run(ok, data, _config(max_order_notional=1000.0))
        assert ok.rejected == [] and ok.filled == [(1, 10.0)]

        over = Script({1: [("buy", 10.0)]})
        run(over, data, _config(max_order_notional=999.99))
        assert over.rejected == [(1, "order_size")]

    def test_notional_uses_the_contract_multiplier(self):
        spec = raptorbt.InstrumentSpec.perpetual("S", lot_size=1.0, multiplier=50.0)
        data = _bars([100.0] * 4)
        # 2 contracts x 100 x 50 = 10_000
        ok = Script({1: [("buy", 2.0)]})
        run_strategy_backtest(
            ok, **data, symbol="S", config=_config(max_order_notional=10_000.0), instrument=spec,
            bar_stamp="close",
        )
        assert ok.rejected == []
        over = Script({1: [("buy", 2.0)]})
        run_strategy_backtest(
            over, **data, symbol="S", config=_config(max_order_notional=9_999.0), instrument=spec,
            bar_stamp="close",
        )
        assert over.rejected == [(1, "order_size")]

    def test_a_capital_fraction_entry_is_judged_on_its_resolved_size(self, run):
        # 50% of 100_000 at 100 = 500 units.
        data = _bars([100.0] * 4)
        ok = Script({1: [("enter", 0.5)]})
        run(ok, data, _config(max_order_qty=500.0))
        assert ok.rejected == []
        over = Script({1: [("enter", 0.5)]})
        run(over, data, _config(max_order_qty=499.0))
        assert over.rejected == [(1, "order_size")]

    def test_a_refused_order_leaves_no_position_and_no_trade(self, run):
        data = _bars([100.0, 100.0, 105.0, 105.0])
        s = Script({1: [("buy", 50.0)]})
        res = run(s, data, _config(max_order_qty=10.0))
        assert _result(res).trades() == []
        assert np.allclose(_result(res).equity_curve(), 100_000.0)


# -- position size ---------------------------------------------------------------


class TestPositionSize:
    def test_max_position_qty_equal_passes_above_is_refused(self):
        # Hedging lets orders stack on one instrument.
        data = _bars([100.0] * 6)
        s = Script({1: [("buy", 6.0)], 2: [("buy", 4.0)], 3: [("buy", 1.0)]})
        res = run_strategy_backtest(
            s, **data, symbol="S", config=_config(max_position_qty=10.0), oms_type="hedging",
            bar_stamp="close",
        )
        assert s.filled == [(1, 6.0), (2, 4.0)]
        assert s.rejected == [(3, "max_position")]
        assert sum(_sizes(res)) == 10.0

    def test_a_single_order_above_the_cap_is_refused_whole(self):
        data = _bars([100.0] * 4)
        s = Script({1: [("buy", 10.5)]})
        run_strategy_backtest(s, **data, symbol="S", config=_config(max_position_qty=10.0), bar_stamp="close")
        assert s.rejected == [(1, "max_position")] and s.filled == []

    def test_short_positions_are_capped_in_magnitude(self):
        data = _bars([100.0] * 6)
        s = Script({1: [("sell", 10.0)], 2: [("sell", 1.0)]})
        run_strategy_backtest(
            s,
            **data,
            symbol="S",
            direction=-1,
            config=_config(max_position_qty=10.0),
            oms_type="hedging",
            account_type="margin",
            leverage=2.0,
            bar_stamp="close",
        )
        assert s.filled == [(1, 10.0)]
        assert s.rejected == [(2, "max_position")]

    def test_an_opposite_fill_is_not_blocked_by_a_cap_it_reduces(self):
        data = _bars([100.0] * 6)
        s = Script({1: [("buy", 10.0)], 2: [("sell", 4.0)], 3: [("buy", 4.0)]})
        run_strategy_backtest(
            s, **data, symbol="S", config=_config(max_position_qty=10.0), oms_type="hedging",
            bar_stamp="close",
        )
        # Hedging opens a short of 4, so the net is 6; the buy brings it back
        # to exactly 10, which passes.
        assert s.rejected == []
        assert s.filled == [(1, 10.0), (2, 4.0), (3, 4.0)]


# -- gross exposure --------------------------------------------------------------


def _two(closes_a, closes_b=None):
    return {"A": _bars(closes_a), "B": _bars(closes_b if closes_b is not None else closes_a)}


class TestGrossExposure:
    def test_across_instruments_equal_passes_above_is_refused(self):
        data = _two([100.0] * 4)
        # A: 10 x 100 = 1000 held; B adds 1000 -> 2000.
        script = {1: [("buy", 10.0, "A")], 2: [("buy", 10.0, "B")]}
        ok = Script(script)
        run_portfolio_strategy(ok, data, config=_config(max_gross_exposure=2000.0), bar_stamp="close")
        assert ok.rejected == [] and len(ok.filled) == 2

        over = Script(script)
        run_portfolio_strategy(over, data, config=_config(max_gross_exposure=1999.99), bar_stamp="close")
        assert over.rejected == [(2, "max_exposure")]
        assert len(over.filled) == 1

    def test_existing_positions_are_marked_at_the_latest_price(self):
        # A doubles to 200 by bar 2, so its 10 units are worth 2000 there; B's
        # 10 units at 100 bring the total to 3000.
        data = _two([100.0, 100.0, 200.0, 200.0], [100.0] * 4)
        script = {1: [("buy", 10.0, "A")], 2: [("buy", 10.0, "B")]}
        s = Script(script)
        run_portfolio_strategy(s, data, config=_config(max_gross_exposure=2999.0), bar_stamp="close")
        assert s.rejected == [(2, "max_exposure")]
        s = Script(script)
        run_portfolio_strategy(s, data, config=_config(max_gross_exposure=3000.0), bar_stamp="close")
        assert s.rejected == []

    def test_within_one_instrument(self):
        data = _bars([100.0] * 5)
        s = Script({1: [("buy", 5.0)], 2: [("buy", 5.0)], 3: [("buy", 1.0)]})
        run_strategy_backtest(
            s, **data, symbol="S", config=_config(max_gross_exposure=1000.0), oms_type="hedging",
            bar_stamp="close",
        )
        assert s.filled == [(1, 5.0), (2, 5.0)]
        assert s.rejected == [(3, "max_exposure")]

    def test_runtime_batch_applies_it_per_strategy(self):
        data = _two([100.0] * 4)
        script = {1: [("buy", 10.0, "A")], 2: [("buy", 10.0, "B")]}
        a, b = Script(script), Script(script)
        run_strategies([a, b], data, _config(max_gross_exposure=1500.0), bar_stamp="close")
        assert a.rejected == b.rejected == [(2, "max_exposure")]


# -- daily loss ------------------------------------------------------------------

# Bars 0-3 are day 0, bars 4-7 day 1, 8-11 day 2 (one bar per 6 hours).
SIX_HOURS = 6 * HOUR


def _days(closes):
    return _bars(closes, step=SIX_HOURS)


class TestDailyLoss:
    # Hold 100 units from 100 down to 90: equity 100_000 -> 99_000.
    CLOSES = [100.0, 100.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]

    def _script(self):
        return {1: [("buy", 100.0)], 3: [("buy", 1.0)], 5: [("buy", 1.0)]}

    def _hedged(self, **limits):
        s = Script(self._script())
        run_strategy_backtest(
            s,
            **_days(self.CLOSES),
            symbol="S",
            config=_config(**limits),
            oms_type="hedging",
            bar_stamp="close",
        )
        return s

    def test_loss_equal_to_the_limit_blocks_entries_for_the_day(self):
        s = self._hedged(max_daily_loss=1000.0)
        # Bar 2 closes the day's loss at exactly 1000; bar 3 (same day) is
        # refused, bar 5 (next day) opens.
        assert s.rejected == [(3, "daily_loss")]
        assert [i for i, _ in s.filled] == [1, 5]

    def test_loss_just_under_the_limit_does_not_block(self):
        s = self._hedged(max_daily_loss=1000.01)
        assert s.rejected == []
        assert [i for i, _ in s.filled] == [1, 3, 5]

    def test_percent_of_the_days_starting_equity(self):
        assert self._hedged(max_daily_loss_pct=1.0).rejected == [(3, "daily_loss")]
        assert self._hedged(max_daily_loss_pct=1.01).rejected == []

    def test_the_tighter_of_amount_and_percent_applies(self):
        s = self._hedged(max_daily_loss=5000.0, max_daily_loss_pct=1.0)
        assert s.rejected == [(3, "daily_loss")]

    def test_unrealised_loss_counts(self):
        # The position is never closed, so the whole loss is unrealised.
        s = self._hedged(max_daily_loss=1000.0)
        assert s.rejected == [(3, "daily_loss")]

    def test_the_limit_resets_at_the_next_day(self):
        s = self._hedged(max_daily_loss=1000.0)
        assert (5, "daily_loss") not in s.rejected
        assert 5 in [i for i, _ in s.filled]

    def test_each_day_is_measured_from_its_own_start(self):
        # Day 0 loses 1000 and is blocked. Day 1 starts at 99_000 and loses
        # another 500 (not 1000), so it is not blocked.
        closes = [100.0, 100.0, 90.0, 90.0, 90.0, 85.0, 85.0, 85.0]
        s = Script({1: [("buy", 100.0)], 3: [("buy", 1.0)], 6: [("buy", 1.0)]})
        run_strategy_backtest(
            s, **_days(closes), symbol="S", config=_config(max_daily_loss=1000.0), oms_type="hedging",
            bar_stamp="close",
        )
        assert s.rejected == [(3, "daily_loss")]
        assert 6 in [i for i, _ in s.filled]

    def test_the_day_follows_the_session_tz_offset(self):
        # With a +6h offset, bar 3 (t=18h) is already the next local day, so
        # the block from the loss on bar 2 (t=12h, day 0) has lifted.
        s = Script(self._script())
        run_strategy_backtest(
            s,
            **_days(self.CLOSES),
            symbol="S",
            config=_config(max_daily_loss=1000.0, session_tz_offset_ns=6 * HOUR),
            oms_type="hedging",
            bar_stamp="close",
        )
        assert s.rejected == []

    def test_day_of_hook_is_the_local_calendar_date(self):
        # 23:00 UTC at +2h is already tomorrow locally.
        cfg = _config(max_daily_loss=1.0, session_tz_offset_ns=2 * HOUR)
        data = _bars([100.0, 100.0, 99.0, 99.0], start=21 * HOUR, step=HOUR)
        s = Script({0: [("buy", 1.0)], 3: [("buy", 1.0)]})
        run_strategy_backtest(s, **data, symbol="S", config=cfg, oms_type="hedging", bar_stamp="close")
        # Bars at 21h, 22h | 23h (local midnight) , 24h. The 1.0 loss lands at
        # 23h (a new local day), then bar 3 (24h, same local day) is refused.
        assert s.rejected == [(3, "daily_loss")]

    def test_flatten_closes_the_position_when_the_limit_is_reached(self):
        closes = [100.0, 100.0, 90.0, 80.0, 70.0, 70.0]
        data = _days(closes)
        keep = run_strategy_backtest(
            Script({1: [("buy", 100.0)]}),
            **data,
            symbol="S",
            config=_config(max_daily_loss=1000.0),
            bar_stamp="close",
        )
        flat = run_strategy_backtest(
            Script({1: [("buy", 100.0)]}),
            **data,
            symbol="S",
            config=_config(max_daily_loss=1000.0, flatten_on_daily_loss=True),
            bar_stamp="close",
        )
        # Not flattened: rides to the end of data and loses 3000.
        assert keep.trades()[0].exit_reason == "EndOfData"
        assert keep.equity_curve()[-1] == pytest.approx(97_000.0)
        # Flattened at bar 2's close, where the loss reached 1000.
        trade = flat.trades()[0]
        assert trade.exit_reason == "Liquidation"
        assert trade.exit_price == 90.0
        assert trade.pnl == pytest.approx(-1000.0)
        assert flat.equity_curve()[-1] == pytest.approx(99_000.0)
        assert flat.equity_curve()[2] == pytest.approx(99_000.0)

    def test_flatten_leaves_the_next_day_free_to_trade(self):
        closes = [100.0, 100.0, 90.0, 90.0, 90.0, 90.0]
        s = Script({1: [("buy", 100.0)], 5: [("buy", 10.0)]})
        res = run_strategy_backtest(
            s,
            **_days(closes),
            symbol="S",
            config=_config(max_daily_loss=1000.0, flatten_on_daily_loss=True),
            bar_stamp="close",
        )
        assert s.rejected == []
        assert len(res.trades()) == 2

    def test_halts_are_not_latched_across_days(self):
        res = run_strategy_backtest(
            Script(self._script()),
            **_days(self.CLOSES),
            symbol="S",
            config=_config(max_daily_loss=1000.0),
            oms_type="hedging",
            bar_stamp="close",
        )
        assert res.halted is False

    @pytest.mark.parametrize("name", ["portfolio", "shared", "stream", "node"])
    def test_every_runner_blocks_and_resets_alike(self, name):
        s = Script(self._script())
        RUNNERS[name](s, _days(self.CLOSES), _config(max_daily_loss=1000.0), oms_type="hedging")
        assert s.rejected == [(3, "daily_loss")]
        assert [i for i, _ in s.filled] == [1, 5]


# -- kill switches ---------------------------------------------------------------


class TestKillSwitch:
    DATA = _bars([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

    def test_ctx_halt_refuses_entries_and_reports_halted(self):
        s = Script({1: [("buy", 10.0)], 2: [("halt", False)], 4: [("buy", 5.0)]})
        res = run_strategy_backtest(
            s, **self.DATA, symbol="S", config=_config(), oms_type="hedging",
            bar_stamp="close",
        )
        assert s.rejected == [(4, "killed")]
        assert res.halted is True and res.halted_at is not None
        # Without flatten the position is left alone until the run ends.
        assert res.trades()[0].exit_reason == "EndOfData"

    def test_ctx_halt_with_flatten_closes_positions_at_the_close_of_the_bar_it_was_raised_on(self):
        s = Script({1: [("buy", 10.0)], 3: [("halt", True)]})
        res = run_strategy_backtest(s, **self.DATA, symbol="S", config=_config(), bar_stamp="close")
        trade = res.trades()[0]
        assert trade.exit_reason == "Liquidation"
        # Corrected (was bar 2's close, 100.0, a bar early): halted while deciding bar 3, it
        # closes as an ordinary reduce-only market order would, at bar 3's close.
        assert trade.exit_price == 90.0
        assert res.halted is True

    def test_halt_cancels_working_orders(self):
        s = Script({0: [("limit_buy", 50.0, 10.0)], 2: [("halt", False)]})
        res = run_strategy_backtest(s, **self.DATA, symbol="S", config=_config(), bar_stamp="close")
        assert s.canceled == 1
        assert res.orders()[0].status == "canceled"

    def test_halt_reason_reaches_on_risk_event(self):
        class Halts(Script):
            def on_bar(self, ctx):
                if ctx.idx == 2:
                    ctx.halt("loss of nerve")
                super().on_bar(ctx)

        s = Halts({4: [("buy", 1.0)]})
        run_strategy_backtest(s, **self.DATA, symbol="S", config=_config(), bar_stamp="close")
        assert ("risk_halted", "loss of nerve") in s.risk
        assert ("risk_rejected", "killed") in s.risk

    def test_no_halt_without_a_request(self):
        res = run_strategy_backtest(Script({1: [("buy", 1.0)]}), **self.DATA, config=_config(), bar_stamp="close")
        assert res.halted is False and res.halted_at is None

    @pytest.mark.parametrize("name", ["portfolio", "shared", "stream", "node"])
    def test_ctx_halt_in_every_portfolio_runner(self, name):
        s = Script({1: [("buy", 10.0)], 2: [("halt", True)], 4: [("buy", 5.0)]})
        res = RUNNERS[name](s, self.DATA, _config(), oms_type="hedging")
        assert s.rejected == [(4, "killed")]
        assert res.halted is True and res.halted_at is not None
        assert _result(res).trades()[0].exit_reason == "Liquidation"

    def test_halt_closes_every_instrument_in_a_portfolio(self):
        data = _two([100.0] * 6)
        s = Script({1: [("buy", 5.0, "A"), ("buy", 5.0, "B")], 3: [("halt", True)]})
        res = run_portfolio_strategy(s, data, config=_config(), bar_stamp="close")
        reasons = sorted(t.exit_reason for t in res.result.trades())
        assert reasons == ["Liquidation", "Liquidation"]
        assert res.halted is True


class TestLiveNodeHalt:
    def _node(self, strategies, **kw):
        return LiveNode(strategies, ["S"], _config(), bar_stamp="close", oms_type="hedging", **kw)

    def _push(self, node, closes, first=0):
        data = _bars(closes, start=first * HOUR)
        for i in range(len(closes)):
            node.push_bar("S", *[data[k][i] for k in _COLS])

    def test_halting_one_strategy_leaves_the_other_trading(self):
        a, b = Script({1: [("buy", 5.0)], 4: [("buy", 1.0)]}), Script({1: [("buy", 5.0)], 4: [("buy", 1.0)]})
        node = self._node([a, b])
        self._push(node, [100.0] * 3)
        assert node.is_halted(0) is False
        node.halt(0)
        assert node.is_halted(0) is True and node.is_halted(1) is False
        self._push(node, [100.0] * 3, first=3)
        results = node.finish()
        assert a.rejected == [(4, "killed")] and b.rejected == []
        assert results[0].halted is True and results[0].halted_at is not None
        assert results[1].halted is False

    def test_halt_with_flatten_closes_at_the_next_event(self):
        s = Script({1: [("buy", 5.0)]})
        node = self._node([s])
        self._push(node, [100.0, 100.0, 95.0])
        node.halt(0, flatten=True)
        self._push(node, [90.0], first=3)
        result = node.finish()[0]
        trade = result.result.trades()[0]
        # Corrected (was 95.0, the last mark): a halt asked for between events acts as an
        # order sent then, so it closes at the next event's close.
        assert trade.exit_reason == "Liquidation" and trade.exit_price == 90.0

    def test_halt_cancels_working_orders_and_tells_the_strategy(self):
        s = Script({0: [("limit_buy", 50.0, 5.0)]})
        node = self._node([s])
        self._push(node, [100.0, 100.0])
        node.halt(0, reason="operator")
        self._push(node, [100.0], first=2)
        assert s.canceled == 1
        assert ("risk_halted", "operator") in s.risk

    def test_halt_works_on_a_tick_stream_strategy_through_its_ctx(self):
        s = Script({1: [("halt", False)], 3: [("buy", 1.0)]})
        stream = TickStrategyStream(s, ["S"], _config(), bar_stamp="close")
        data = _bars([100.0] * 5)
        for i in range(5):
            stream.push_bar("S", *[data[k][i] for k in _COLS])
        assert stream.is_halted is True
        assert s.rejected == [(3, "killed")]


# -- risk events -----------------------------------------------------------------


class TestRiskEvents:
    def test_every_limit_breach_reaches_on_risk_event_and_on_order_rejected(self, run):
        data = _bars([100.0] * 4)
        s = Script({1: [("buy", 50.0)]})
        run(s, data, _config(max_order_qty=10.0))
        assert s.risk == [("risk_rejected", "order_size")]
        assert s.rejected == [(1, "order_size")]

    def test_daily_loss_emits_a_halt_event_then_rejections(self):
        s = Script({1: [("buy", 100.0)], 3: [("buy", 1.0)]})
        run_strategy_backtest(
            s,
            **_days(TestDailyLoss.CLOSES),
            symbol="S",
            config=_config(max_daily_loss=1000.0),
            oms_type="hedging",
            bar_stamp="close",
        )
        assert s.risk == [("risk_halted", "daily_loss"), ("risk_rejected", "daily_loss")]

    def test_events_carry_the_order_ids(self):
        events = []

        class Spy(Script):
            def on_risk_event(self, ctx, event):
                events.append(event)

        s = Spy({1: [("buy", 50.0)]})
        run_strategy_backtest(s, **_bars([100.0] * 3), symbol="S", config=_config(max_order_qty=1.0), bar_stamp="close")
        (event,) = events
        assert event.kind == "risk_rejected"
        assert event.order_id is not None and event.client_order_id
        assert event.idx == 1

    def test_a_signal_entry_refusal_has_no_order_id(self):
        events = []

        class Spy(Script):
            def on_risk_event(self, ctx, event):
                events.append(event)

        s = Spy({1: [("enter", 0.5)]})
        run_strategy_backtest(s, **_bars([100.0] * 3), symbol="S", config=_config(max_order_qty=1.0), bar_stamp="close")
        assert [e.order_id for e in events] == [None]

    def test_the_hook_is_a_no_op_by_default(self):
        class Plain(Strategy):
            def on_bar(self, ctx):
                if ctx.idx == 1:
                    self.submit_order(orders.Market(side="buy", units=50.0))

        run_strategy_backtest(Plain(), **_bars([100.0] * 3), symbol="S", config=_config(max_order_qty=1.0), bar_stamp="close")

    def test_existing_reasons_do_not_raise_risk_events(self):
        s = Script({1: [("buy", 10.0)]})  # more than the 500 of capital buys: insufficient_capital
        config = BacktestConfig(fees=0.0, initial_capital=500.0, max_order_qty=10.0)
        run_strategy_backtest(s, **_bars([100.0] * 4), symbol="S", config=config, bar_stamp="close")
        assert s.rejected == [(1, "insufficient_capital")]
        assert s.risk == []


# -- off by default --------------------------------------------------------------


class TestOffByDefault:
    LIMITS = dict(
        max_order_qty=None,
        max_order_notional=None,
        max_position_qty=None,
        max_gross_exposure=None,
        max_daily_loss=None,
        max_daily_loss_pct=None,
        flatten_on_daily_loss=False,
    )

    def test_defaults_are_off(self):
        cfg = BacktestConfig()
        for name, value in self.LIMITS.items():
            assert getattr(cfg, name) == value

    @staticmethod
    def _trace(res):
        r = _result(res)
        return (
            r.equity_curve().tobytes(),
            r.returns().tobytes(),
            [(t.entry_idx, t.exit_idx, t.size, t.pnl, t.exit_reason) for t in r.trades()],
            [(o.status, o.filled_qty) for o in r.orders()],
        )

    def _script(self):
        return {
            1: [("buy", 30.0)],
            2: [("sell", 30.0)],
            3: [("enter", 0.4)],
            5: [("buy", 20.0)],
            6: [("sell", 20.0)],
        }

    def test_bit_identical_with_explicit_none_and_with_zero(self, run):
        rng = np.random.default_rng(7)
        data = _days(100 + np.cumsum(rng.normal(0, 2, 12)))
        base = self._trace(run(Script(self._script()), data, BacktestConfig(fees=0.001)))
        off = BacktestConfig(fees=0.001, **self.LIMITS)
        assert self._trace(run(Script(self._script()), data, off)) == base
        # A non-positive limit means "no limit", never "reject everything".
        zeros = BacktestConfig(
            fees=0.001,
            max_order_qty=0.0,
            max_order_notional=0.0,
            max_position_qty=0.0,
            max_gross_exposure=0.0,
            max_daily_loss=0.0,
            max_daily_loss_pct=0.0,
        )
        assert self._trace(run(Script(self._script()), data, zeros)) == base

    def test_limits_that_never_bind_change_nothing(self):
        rng = np.random.default_rng(11)
        data = _days(100 + np.cumsum(rng.normal(0, 2, 12)))
        base = self._trace(
            run_strategy_backtest(Script(self._script()), **data, config=BacktestConfig(), bar_stamp="close")
        )
        loose = BacktestConfig(
            max_order_qty=1e9,
            max_order_notional=1e12,
            max_position_qty=1e9,
            max_gross_exposure=1e12,
            max_daily_loss=1e12,
            max_daily_loss_pct=100.0,
        )
        got = self._trace(run_strategy_backtest(Script(self._script()), **data, config=loose, bar_stamp="close"))
        assert got == base

    def test_no_events_without_limits(self):
        s = Script({1: [("buy", 5.0)], 2: [("buy", 5.0)]})
        run_strategy_backtest(s, **_bars([100.0] * 4), config=BacktestConfig(), bar_stamp="close")
        assert s.risk == []


# -- array runners ---------------------------------------------------------------


class TestArrayRunners:
    @staticmethod
    def _signals(n=12):
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        entries[1] = True
        exits[4] = True
        entries[6] = True
        exits[9] = True
        return entries, exits

    def _single(self, closes, config):
        d = _days(closes)
        entries, exits = self._signals(len(closes))
        return raptorbt.run_single_backtest(
            d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"],
            entries, exits, config=config,
        )

    def test_single_backtest_honours_order_size(self):
        closes = [100.0] * 12
        base = self._single(closes, BacktestConfig(fees=0.0))
        assert len(base.trades()) == 2
        # All-in sizing buys 1000 units; a 999-unit cap refuses both entries.
        capped = self._single(closes, BacktestConfig(fees=0.0, max_order_qty=999.0))
        assert capped.trades() == []
        exact = self._single(closes, BacktestConfig(fees=0.0, max_order_qty=1000.0))
        assert len(exact.trades()) == 2

    def test_single_backtest_honours_daily_loss(self):
        # Loss on day 0; the second signal is day 1 (bar 6), so it trades.
        closes = [100, 100, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
        res = self._single(closes, BacktestConfig(fees=0.0, max_daily_loss=500.0, flatten_on_daily_loss=True))
        assert [t.exit_reason for t in res.trades()][0] == "Liquidation"
        assert len(res.trades()) == 2

    def test_single_backtest_reports_the_drawdown_halt(self):
        closes = [100, 100, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
        res = self._single(closes, BacktestConfig(fees=0.0, max_drawdown_pct=10.0))
        assert res.halted is True and res.halted_at == 2

    def _portfolio(self, config, closes=None):
        n = 12
        entries, exits = self._signals(n)
        d = _days(closes if closes is not None else [100.0] * n)
        return raptorbt.run_portfolio_backtest(
            [
                (d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"],
                 entries, exits, 1, 1.0, sym)
                for sym in ("A", "B")
            ],
            config=config,
        )

    def test_portfolio_backtest_honours_order_size(self):
        # Equal weight gives each of the two 50_000 / 100 = 500 units.
        assert len(self._portfolio(BacktestConfig(fees=0.0)).result.trades()) > 0
        exact = self._portfolio(BacktestConfig(fees=0.0, max_order_qty=500.0))
        assert len(exact.result.trades()) > 0 and exact.rejected_entries == 0
        capped = self._portfolio(BacktestConfig(fees=0.0, max_order_qty=499.0))
        assert capped.result.trades() == []
        assert capped.rejected_entries > 0

    def test_portfolio_backtest_honours_gross_exposure(self):
        # Each entry is 50_000 of notional; a 50_000 cap admits one at a time.
        out = self._portfolio(BacktestConfig(fees=0.0, max_gross_exposure=50_000.0))
        assert out.rejected_entries > 0
        by_symbol = {s.symbol: s.trades for s in out.per_instrument}
        assert by_symbol["A"] > 0 and by_symbol["B"] == 0
        both = self._portfolio(BacktestConfig(fees=0.0, max_gross_exposure=100_000.0))
        assert both.rejected_entries == 0

    def test_portfolio_backtest_honours_daily_loss(self):
        closes = [100, 100, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
        free = self._portfolio(BacktestConfig(fees=0.0), closes)
        cut = self._portfolio(
            BacktestConfig(fees=0.0, max_daily_loss=500.0, flatten_on_daily_loss=True), closes
        )
        assert cut.result.trades()[0].exit_reason == "Liquidation"
        assert {t.exit_idx for t in cut.result.trades()} >= {2}
        assert free.result.trades()[0].exit_reason != "Liquidation"
        assert cut.result.equity_curve()[3] == pytest.approx(cut.result.equity_curve()[2])


# -- kernel properties -----------------------------------------------------------


class _Chaos(Strategy):
    """Random market and limit orders on one or two symbols; records state."""

    def __init__(self, rng, symbols):
        super().__init__()
        self.rng = rng
        self.symbols = symbols
        self.last = {}
        self.violations = []
        self.fills = []

    def on_bar(self, ctx):
        sym = getattr(ctx, "symbol", None) or self.symbols[0]
        self.last[sym] = ctx.bar.close
        for _ in range(int(self.rng.integers(0, 3))):
            side = "buy" if self.rng.random() < 0.6 else "sell"
            units = float(self.rng.integers(1, 40))
            if self.rng.random() < 0.3:
                price = ctx.bar.close * float(self.rng.uniform(0.97, 1.03))
                order = orders.Limit(side=side, price=price, units=units)
            else:
                order = orders.Market(side=side, units=units)
            self.submit_order(order, symbol=sym)

    def on_fill(self, ctx, event):
        if event.size is None:
            return
        self.fills.append(event.size)
        self.checks(ctx, event)

    def checks(self, ctx, event):
        raise NotImplementedError


class _SizeChecks(_Chaos):
    def __init__(self, rng, symbols, limits):
        super().__init__(rng, symbols)
        self.limits = limits

    def checks(self, ctx, event):
        lim = self.limits
        tol = 1e-9
        if lim.get("max_order_qty") is not None and event.size > lim["max_order_qty"] + tol:
            self.violations.append(("order_qty", event.size))
        if event.kind != "order_filled":
            return
        sym = getattr(ctx, "symbol", None) or self.symbols[0]
        if lim.get("max_position_qty") is not None:
            net = ctx.net_position_for(sym) if hasattr(ctx, "net_position_for") else ctx.net_position
            if abs(net) > lim["max_position_qty"] + tol:
                self.violations.append(("position", net))
        if lim.get("max_gross_exposure") is not None:
            gross = 0.0
            for s in self.symbols:
                pos = ctx.positions(s)
                gross += sum(p.size for p in pos) * self.last.get(s, 0.0)
            if gross > lim["max_gross_exposure"] * (1 + 1e-9):
                self.violations.append(("exposure", gross))


@pytest.mark.parametrize("seed", range(40))
def test_random_limits_are_never_violated_by_a_fill(seed):
    rng = np.random.default_rng(900 + seed)
    n = int(rng.integers(30, 70))
    symbols = ["A", "B"] if seed % 2 else ["S"]
    limits = {}
    for name, lo, hi in (
        ("max_order_qty", 5, 40),
        ("max_position_qty", 20, 120),
        ("max_gross_exposure", 3_000, 20_000),
    ):
        if rng.random() < 0.7:
            limits[name] = float(rng.integers(lo, hi))
    data = {
        s: _bars(100.0 + np.cumsum(rng.normal(0, 1.5, n)).clip(-60, None))
        for s in symbols
    }
    cfg = BacktestConfig(fees=0.0, initial_capital=1_000_000.0, **limits)
    strat = _SizeChecks(np.random.default_rng(seed), symbols, limits)
    if len(symbols) == 1:
        run_strategy_backtest(
            strat, **data["S"], symbol="S", config=cfg, oms_type="hedging",
            bar_stamp="close",
        )
    else:
        run_portfolio_strategy(strat, data, config=cfg, oms_type="hedging", bar_stamp="close")
    assert strat.fills, "the script should fill something"
    assert strat.violations == []


# -- the daily-loss day follows the instrument's session schedule ----------------

import datetime as _dt  # noqa: E402

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri"]


def _utc_ns(y, mo, d, h=0, mi=0):
    stamp = _dt.datetime(y, mo, d, h, mi, tzinfo=_dt.timezone.utc)
    return int(stamp.timestamp()) * 10**9


def _schedule(tz_offset_ns=0, holidays=None):
    windows = {day: [(0, 1440)] for day in _WEEKDAYS}
    return raptorbt.SessionSchedule(windows, tz_offset_ns=tz_offset_ns, holidays=holidays)


def _stamped(closes, stamps):
    d = _bars(closes)
    d["timestamps"] = np.asarray(stamps, dtype=np.int64)
    return d


def _risk_run(closes, stamps, script, schedule, **cfg):
    spec = raptorbt.InstrumentSpec.equity("S", session=schedule) if schedule else None
    s = Script(script)
    run_strategy_backtest(
        s,
        **_stamped(closes, stamps),
        symbol="S",
        config=_config(max_daily_loss=1000.0, **cfg),
        oms_type="hedging",
        instrument=spec,
        bar_stamp="close",
    )
    return s


class TestSessionDay:
    IST = 5 * HOUR + HOUR // 2

    def test_no_schedule_is_the_local_calendar_date(self):
        # Mon 20:00 UTC and 21:00 UTC: same UTC date, so no reset.
        stamps = [_utc_ns(2026, 1, 19, h) for h in (6, 12, 20, 21)]
        s = _risk_run([100.0, 90.0, 90.0, 90.0], stamps, {0: [("buy", 100.0)], 3: [("buy", 1.0)]}, None)
        assert s.rejected == [(3, "daily_loss")]

    def test_a_schedule_in_another_timezone_moves_the_day_boundary(self):
        # 20:00 UTC on Monday is 01:30 on Tuesday in IST: a new session day,
        # so the block from Monday's loss has lifted, though the config's own
        # offset (0) would still call it Monday.
        stamps = [_utc_ns(2026, 1, 19, h) for h in (6, 12, 20, 21)]
        script = {0: [("buy", 100.0)], 3: [("buy", 1.0)]}
        s = _risk_run([100.0, 90.0, 90.0, 90.0], stamps, script, _schedule(self.IST))
        assert s.rejected == []
        assert [i for i, _ in s.filled] == [0, 3]

    def test_the_schedule_day_ignores_the_config_offset(self):
        # Same bars as above with a config offset that would move the
        # boundary the other way: the schedule's day is what counts.
        stamps = [_utc_ns(2026, 1, 19, h) for h in (6, 12, 20, 21)]
        script = {0: [("buy", 100.0)], 3: [("buy", 1.0)]}
        s = _risk_run(
            [100.0, 90.0, 90.0, 90.0], stamps, script, _schedule(self.IST), session_tz_offset_ns=-12 * HOUR
        )
        assert s.rejected == []

    def test_a_weekend_has_no_session_date_so_the_day_does_not_roll(self):
        # Friday's loss is 600, then Saturday adds 600 more. Without a
        # schedule Saturday is a fresh day (600 on the day: no breach); with
        # one it is still Friday's session (1200 on the day: breach).
        stamps = [_utc_ns(2026, 1, 16, 6), _utc_ns(2026, 1, 16, 12), _utc_ns(2026, 1, 17, 12), _utc_ns(2026, 1, 19, 12)]
        closes = [100.0, 94.0, 88.0, 88.0]
        script = {0: [("buy", 100.0)]}
        free = _risk_run(closes, stamps, script, None)
        assert free.risk == []
        held = _risk_run(closes, stamps, script, _schedule())
        assert ("risk_halted", "daily_loss") in held.risk

    def test_a_holiday_has_no_session_date_so_the_day_does_not_roll(self):
        # Thu 15th trades, Fri 16th is a holiday, Mon 19th trades.
        stamps = [_utc_ns(2026, 1, 15, 6), _utc_ns(2026, 1, 15, 12), _utc_ns(2026, 1, 16, 12), _utc_ns(2026, 1, 19, 12)]
        closes = [100.0, 94.0, 88.0, 88.0]
        script = {0: [("buy", 100.0)]}
        held = _risk_run(closes, stamps, script, _schedule(holidays=[20260116]))
        assert ("risk_halted", "daily_loss") in held.risk
        no_holiday = _risk_run(closes, stamps, script, _schedule())
        assert no_holiday.risk == []

    def test_the_block_lifts_on_the_next_session_day(self):
        stamps = [_utc_ns(2026, 1, 16, 6), _utc_ns(2026, 1, 16, 12), _utc_ns(2026, 1, 16, 18), _utc_ns(2026, 1, 19, 12)]
        script = {0: [("buy", 100.0)], 2: [("buy", 1.0)], 3: [("buy", 1.0)]}
        s = _risk_run([100.0, 90.0, 90.0, 90.0], stamps, script, _schedule())
        assert s.rejected == [(2, "daily_loss")]
        assert [i for i, _ in s.filled] == [0, 3]

    def test_a_book_takes_its_day_from_the_event_being_applied(self):
        # A has an IST schedule, B has none (UTC date). After the loss on
        # Monday, A's 01:30 IST Tuesday event rolls the whole book to the
        # new day; B's event at the same instant would not have.
        spec_a = raptorbt.InstrumentSpec.equity("A", session=_schedule(self.IST))
        spec_b = raptorbt.InstrumentSpec.equity("B")
        stamps = [_utc_ns(2026, 1, 19, h) for h in (6, 12, 20, 21)]
        data = {"A": _stamped([100.0, 90.0, 90.0, 90.0], stamps), "B": _stamped([100.0] * 4, stamps)}
        s = Script({0: [("buy", 100.0, "A")], 3: [("buy", 1.0, "A")]})
        run_portfolio_strategy(
            s, data, config=_config(max_daily_loss=1000.0), oms_type="hedging",
            instruments={"A": spec_a, "B": spec_b},
            bar_stamp="close",
        )
        assert s.rejected == []
