"""Known defects, pinned.

Every test asserts the CORRECT behaviour with hand-computable inputs. A
defect that still reproduces is marked ``xfail(strict=True)``, so a fix turns
the test red (XPASS) and the mark is removed in the same change. Unmarked
tests are controls that pin the behaviour which is already correct.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Strategy
from raptorbt.strategy import orders

H_NS = 3_600_000_000_000


def _bars(closes, lows=None, highs=None, start_ts=0, step=1):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(start_ts, start_ts + n * step, step, dtype=np.int64),
        "open": closes.copy(),
        "high": (
            np.asarray(highs, dtype=np.float64) if highs is not None else closes + 1.0
        ),
        "low": np.asarray(lows, dtype=np.float64) if lows is not None else closes - 1.0,
        "close": closes,
        "volume": np.full(n, 1_000.0),
    }


def _cfg(**kwargs):
    kwargs.setdefault("initial_capital", 100_000.0)
    kwargs.setdefault("fees", 0.0)
    kwargs.setdefault("slippage", 0.0)
    return BacktestConfig(**kwargs)


# --------------------------------------------------------------------------
# cash-mode shorts are valued like longs
# --------------------------------------------------------------------------
def test_cash_mode_short_equity_gains_when_price_falls():
    # Short 1000 units at 100 (100k / 100). Price 100 -> 95 -> 90.
    # Trade pnl = (100 - 90) * 1000 = +10_000 (the engine reports this).
    # Correct equity: bar2 = 100k + 5k = 105k, final = 100k + 10k = 110k.
    # Observed: equity_curve == [100k, 100k, 95k, 90k, 90k] (identical to a long).
    closes = np.array([100.0, 100.0, 95.0, 90.0, 90.0])
    n = len(closes)
    ts = np.arange(n, dtype=np.int64)
    entries = np.array([1, 0, 0, 0, 0], dtype=bool)
    exits = np.array([0, 0, 0, 1, 0], dtype=bool)
    result = raptorbt.run_single_backtest(
        ts,
        closes,
        closes + 0.5,
        closes - 0.5,
        closes,
        np.full(n, 1e3),
        entries,
        exits,
        direction=-1,
        config=_cfg(upon_bar_close=True),
    )
    trade_pnl = sum(t.pnl for t in result.trades())
    assert trade_pnl == pytest.approx(10_000.0)  # premise: the trade record is right
    equity = result.equity_curve()
    assert equity[2] == pytest.approx(105_000.0)
    assert equity[-1] == pytest.approx(100_000.0 + trade_pnl)


# --------------------------------------------------------------------------
# run_portfolio_backtest drops NextBarOpen entries
# --------------------------------------------------------------------------
def test_portfolio_backtest_fills_next_bar_open_entry():
    # Entry signal on bar 1 only, exit signal on bar 5. Opens: 100,100,101,...
    # NextBarOpen => entry fills at bar 2 open = 101, exit at bar 6 open = 105.
    # run_single_backtest (control, same inputs) gives exactly (101, 105).
    n = 8
    op = np.array([100, 100, 101, 102, 103, 104, 105, 106.0])
    cl = op + 0.5
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000
    ent = np.zeros(n, bool)
    ent[1] = True
    ex = np.zeros(n, bool)
    ex[5] = True
    cfg = _cfg(fill_timing="next_bar_open")
    vol = np.full(n, 1e3)
    inst = [(ts, op, op + 1, op - 1, cl, vol, ent, ex, 1, 1.0, "A")]
    result = raptorbt.run_portfolio_backtest(inst, config=cfg)
    trades = result.result.trades()
    assert len(trades) == 1, f"entry dropped; trades={len(trades)}"
    assert trades[0].entry_price == pytest.approx(101.0)
    assert trades[0].exit_price == pytest.approx(105.0)


def test_control_single_backtest_fills_next_bar_open_entry():
    n = 8
    op = np.array([100, 100, 101, 102, 103, 104, 105, 106.0])
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000
    ent = np.zeros(n, bool)
    ent[1] = True
    ex = np.zeros(n, bool)
    ex[5] = True
    r = raptorbt.run_single_backtest(
        ts, op, op + 1, op - 1, op + 0.5, np.full(n, 1e3), ent, ex,
        config=_cfg(fill_timing="next_bar_open"),
    )
    t = r.trades()
    assert len(t) == 1 and t[0].entry_price == pytest.approx(101.0)


# --------------------------------------------------------------------------
# targeted close_position() fills same-bar under NextBarOpen
# --------------------------------------------------------------------------
def _nbo_data():
    op = np.array([100, 110, 120, 130, 140, 150, 160, 170.0])
    n = len(op)
    return {
        "timestamps": np.arange(n, dtype=np.int64) * 60_000_000_000,
        "open": op,
        "high": op + 6,
        "low": op - 1,
        "close": op + 5,
        "volume": np.full(n, 1e3),
    }


def _closer(mode):
    class S(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.enter()
            if ctx.idx == 3 and ctx.position is not None:
                if mode == "legacy":
                    self.close_position()
                elif mode == "position_id":
                    self.close_position(position_id=ctx.position.position_id)
                else:
                    self.close_position(symbol="A")

    return S()


def _nbo_exit(runner, mode):
    cfg = _cfg(fill_timing="next_bar_open")
    if runner == "single":
        r = raptorbt.run_strategy_backtest(_closer(mode), **_nbo_data(), config=cfg)
        return r.trades()
    r = raptorbt.run_portfolio_strategy(_closer(mode), {"A": _nbo_data()}, config=cfg)
    return r.result.trades()


def test_control_untargeted_close_defers_to_next_open():
    # Decision on bar 3 => fill at bar 4 open = 140. Entry (bar 0) fills bar 1 open = 110.
    trades = _nbo_exit("single", "legacy")
    assert trades[0].entry_price == pytest.approx(110.0)
    assert trades[0].exit_price == pytest.approx(140.0)


@pytest.mark.parametrize("runner", ["single", "portfolio"])
@pytest.mark.parametrize("mode", ["position_id", "symbol"])
def test_targeted_close_defers_to_next_open(runner, mode):
    # Same decision (bar 3) as the control: causal fill is bar 4 open = 140.
    # Observed: 130 (bar 3's open, a price from before the decision existed).
    trades = _nbo_exit(runner, mode)
    assert len(trades) == 1
    assert trades[0].exit_price == pytest.approx(140.0)


# --------------------------------------------------------------------------
# Twap in the portfolio and tick runners
# --------------------------------------------------------------------------
class _TwapProbe(Strategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.fills = []
        self.started = []

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(orders.Twap(side="buy", units=30.0, slices=3, every=1))

    def on_order_filled(self, ctx, event):
        self.fills.append((event.idx, event.size))

    def on_algo_started(self, ctx, event):
        self.started.append(event.client_order_id)


def test_control_single_runner_slices_twap():
    n = 5
    p = np.full(n, 100.0)
    d = {"timestamps": np.arange(n, dtype=np.int64), "open": p, "high": p,
         "low": p, "close": p, "volume": np.ones(n)}
    s = _TwapProbe()
    raptorbt.run_strategy_backtest(s, **d, config=_cfg(), oms_type="hedging")
    assert s.fills == [(0, 10.0), (1, 10.0), (2, 10.0)]


def test_portfolio_runner_slices_twap(portfolio_runner):
    # 30 units / 3 slices every 1ns on 1ns bars => fills at idx 0,1,2 of 10 units.
    n = 5
    p = np.full(n, 100.0)
    d = {"timestamps": np.arange(n, dtype=np.int64), "open": p, "high": p,
         "low": p, "close": p, "volume": np.ones(n)}
    s = _TwapProbe()
    portfolio_runner(s, {"A": d}, config=_cfg(), oms_type="hedging")
    assert s.fills == [(0, 10.0), (1, 10.0), (2, 10.0)]
    assert len(s.started) == 1


def test_tick_runner_slices_twap():
    class S(Strategy):
        def __init__(self, config=None):
            super().__init__(config)
            self.fills = []
            self.started = 0
            self.completed = 0

        def on_trade_tick(self, ctx, tick):
            if not self.started:
                self.submit_order(orders.Twap(side="buy", units=30.0, slices=3, every=1))

        def on_order_filled(self, ctx, event):
            self.fills.append(event.size)

        def on_algo_started(self, ctx, event):
            self.started += 1

        def on_algo_completed(self, ctx, event):
            self.completed += 1

    ticks = {"A": {"timestamps": np.arange(6, dtype=np.int64), "ltp": np.full(6, 100.0)}}
    s = S()
    raptorbt.run_tick_strategy(s, ticks, config=_cfg(), oms_type="hedging")
    assert s.fills == [10.0, 10.0, 10.0]
    assert (s.started, s.completed) == (1, 1)


# --------------------------------------------------------------------------
# hook-queued work is drained at the next event (any symbol)
# --------------------------------------------------------------------------
def _two_symbol_bars():
    def mk(ts, px):
        px = np.array(px, float)
        return {"timestamps": np.array(ts, dtype=np.int64), "open": px,
                "high": px + 1, "low": px - 1, "close": px,
                "volume": np.full(len(px), 1e3)}

    # Merged schedule: A@0, B@1, A@2, B@3, A@4, B@5.
    return {"A": mk([0, 2, 4], [100, 100, 100]), "B": mk([1, 3, 5], [50, 50, 50])}


def test_submit_order_from_hook_routes_to_hook_symbol():
    # A's first order fills at A@0; on_position_opened(A) submits a second
    # buy with no symbol. Correct: it belongs to A. Observed: order_accepted /
    # order_filled arrive with ctx.symbol == "B" and B ends with a trade.
    class S(Strategy):
        def __init__(self, config=None):
            super().__init__(config)
            self.evs = []
            self.fired = False

        def on_bar(self, ctx):
            if ctx.symbol == "A" and ctx.idx == 0:
                self.submit_order(orders.Market(side="buy", units=1.0))

        def on_position_opened(self, ctx, event):
            if ctx.symbol == "A" and not self.fired:
                self.fired = True
                self.submit_order(orders.Market(side="buy", units=3.0))

        def on_order_event(self, ctx, event):
            self.evs.append((ctx.symbol, event.kind))

    s = S()
    r = raptorbt.run_portfolio_strategy(
        s, _two_symbol_bars(), config=_cfg(), oms_type="hedging"
    )
    assert {sym for sym, _ in s.evs} == {"A"}, s.evs
    assert {i.symbol: i.trades for i in r.per_instrument}["B"] == 0


def test_enter_from_hook_does_not_open_other_symbol():
    class S(Strategy):
        def __init__(self, config=None):
            super().__init__(config)
            self.fired = False

        def on_bar(self, ctx):
            if ctx.symbol == "A" and ctx.idx == 0:
                self.enter(size_frac=0.5)

        def on_position_opened(self, ctx, event):
            if ctx.symbol == "A" and not self.fired:
                self.fired = True
                self.enter(size_frac=0.25)  # A already holds a position

    r = raptorbt.run_portfolio_strategy(
        S(), _two_symbol_bars(), config=_cfg(), oms_type="hedging"
    )
    # Observed: B opens a position from A's hook intent.
    assert {i.symbol: i.trades for i in r.per_instrument}["B"] == 0


def test_tick_enter_from_on_quote_does_not_open_other_symbol():
    # A@0 is a quote-only row (ltp=0). enter() from on_quote(A) cannot fill on
    # a quote, but it must not become an entry on B's print at ts=1.
    def tk(ts, ltp, bid, ask):
        return {"timestamps": np.array(ts, dtype=np.int64), "ltp": np.array(ltp, float),
                "bid": np.array(bid, float), "ask": np.array(ask, float)}

    data = {
        "A": tk([0, 2], [0, 100], [99, 99], [101, 101]),
        "B": tk([1, 3], [50, 50], [0, 0], [0, 0]),
    }

    class S(Strategy):
        def __init__(self, config=None):
            super().__init__(config)
            self.fired = False

        def on_quote(self, ctx, quote):
            if ctx.symbol == "A" and not self.fired:
                self.fired = True
                self.enter(size_frac=0.5)

    r = raptorbt.run_tick_strategy(S(), data, config=_cfg())
    assert {i.symbol: i.trades for i in r.per_instrument}["B"] == 0


def test_submit_order_from_on_quote_is_applied_at_the_quote():
    # submit_order from on_quote/on_order_book is not affected: it does
    # apply_commands runs right after on_quote, on A.
    def tk(ts, ltp, bid, ask):
        return {"timestamps": np.array(ts, dtype=np.int64), "ltp": np.array(ltp, float),
                "bid": np.array(bid, float), "ask": np.array(ask, float)}

    data = {
        "A": tk([0, 2], [0, 100], [99, 99], [101, 101]),
        "B": tk([1, 3], [50, 50], [0, 0], [0, 0]),
    }

    class S(Strategy):
        def __init__(self, config=None):
            super().__init__(config)
            self.fired = False

        def on_quote(self, ctx, quote):
            if ctx.symbol == "A" and not self.fired:
                self.fired = True
                self.submit_order(orders.Market(side="buy", units=1.0))

    r = raptorbt.run_tick_strategy(S(), data, config=_cfg())
    assert {i.symbol: i.trades for i in r.per_instrument} == {"A": 1, "B": 0}


# --------------------------------------------------------------------------
# ATR stops disabled in portfolio runs
# --------------------------------------------------------------------------
def _atr_data():
    # Constant true range 2 (high 101, low 99) => ATR(3) = 2; stop = 100 - 2*2 = 96.
    # Bar 6 low = 90 breaches it.
    n = 10
    lo = np.full(n, 99.0)
    lo[6] = 90.0
    px = np.full(n, 100.0)
    return {"timestamps": np.arange(n, dtype=np.int64), "open": px, "high": np.full(n, 101.0),
            "low": lo, "close": px.copy(), "volume": np.full(n, 1e3)}


def _atr_cfg():
    c = _cfg()
    c.set_atr_stop(2.0, 3)
    return c


class _EnterAt5(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 5:
            self.enter()


def test_control_single_runner_honours_atr_stop():
    r = raptorbt.run_strategy_backtest(_EnterAt5(), **_atr_data(), config=_atr_cfg())
    assert r.trades()[0].exit_price == pytest.approx(96.0)


def test_portfolio_runner_honours_atr_stop():
    # Expected exit at the 96.0 stop (as in the single-runner control).
    # Observed: 100.0, the end-of-data close; the stop never existed.
    r = raptorbt.run_portfolio_strategy(_EnterAt5(), {"A": _atr_data()}, config=_atr_cfg())
    assert r.result.trades()[0].exit_price == pytest.approx(96.0)


# --------------------------------------------------------------------------
# slippage applied on top of limit / target prices
# --------------------------------------------------------------------------
def test_limit_buy_never_fills_above_its_limit():
    # Buy limit 97, low reaches 96.5 on bar 3 => fill at 97.0 (README: with
    # limit_slippage=0 "fills exactly at the limit"). slippage=1% is a market
    # cost. Observed: 97 * 1.01 = 97.97, worse than the limit.
    class S(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.submit_order(orders.Limit(side="buy", price=97.0, units=10.0))

    d = _bars([100.0, 100.0, 100.0, 98.0], lows=[99.0, 99.0, 99.0, 96.5])
    r = raptorbt.run_strategy_backtest(S(), **d, config=_cfg(slippage=0.01))
    assert r.trades()[0].entry_price <= 97.0 + 1e-9


def test_target_exit_never_fills_below_target():
    # Market entry slips to 101 (legit). Target +5% => 106.05, bar 2 high 107
    # reaches it => exit at 106.05. Observed 106.05 * 0.99 = 104.9895.
    cfg = _cfg(slippage=0.01)
    cfg.set_fixed_target(0.05)

    class S(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                self.enter()

    d = _bars([100.0] * 4, highs=[101.0, 101.0, 107.0, 101.0])
    r = raptorbt.run_strategy_backtest(S(), **d, config=cfg)
    assert r.trades()[0].entry_price == pytest.approx(101.0)
    assert r.trades()[0].exit_price >= 106.05 - 1e-9


# --------------------------------------------------------------------------
# pairs exit credits the short leg as if long
# --------------------------------------------------------------------------
def test_pairs_hedged_flat_prices_leave_equity_unchanged():
    # Both legs 100 -> 110 -> 110. Leg1 long, leg2 short, 500 units each
    # (50% of 100k per leg, hedge 1.0): pnl = +5000 - 5000 = 0, no fees.
    # Equity must stay 100_000 through the exit at bar 2.
    # Observed: [100k, 100k, 110k, 110k] with both trades reporting pnl 0.
    n = 4
    ts = np.arange(n, dtype=np.int64)
    p = np.array([100.0, 110.0, 110.0, 110.0])
    v = np.full(n, 1e3)
    ent = np.array([1, 0, 0, 0], bool)
    ex = np.array([0, 0, 1, 0], bool)
    r = raptorbt.run_pairs_backtest(
        ts, p, p + 1, p - 1, p, v, ts, p, p + 1, p - 1, p, v, ent, ex,
        direction=1, config=_cfg(upon_bar_close=True), hedge_ratio=1.0,
    )
    assert sum(t.pnl for t in r.trades()) == pytest.approx(0.0)  # premise
    eq = r.equity_curve()
    assert eq[1] == pytest.approx(100_000.0)  # marked correctly while open
    assert eq[2] == pytest.approx(100_000.0)
    assert eq[-1] == pytest.approx(100_000.0)


# --------------------------------------------------------------------------
# portfolio equity sampled once per event, not per timestamp
# --------------------------------------------------------------------------
def _identical_instruments(k):
    n = 12
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    data = {
        f"S{i}": {
            "timestamps": np.arange(n, dtype=np.int64) * 86_400_000_000_000,
            "open": c.copy(), "high": c * 1.001, "low": c * 0.999,
            "close": c.copy(), "volume": np.full(n, 1e3),
        }
        for i in range(k)
    }

    class S(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                # Sequential fractions of free cash: 1/k, 1/(k-1), ..., 1 =>
                # each symbol gets exactly 1/k of the capital.
                self.enter(size_frac=1.0 / (k - int(ctx.symbol[1:])))

    return raptorbt.run_portfolio_strategy(S(), data, config=_cfg())


def test_equity_samples_once_per_timestamp():
    # Same economics (final equity identical: 103_616.25 for k=1 and k=4).
    # Expected: 12 samples and equal Sharpe. Observed: 12 vs 48 samples,
    # Sharpe 9.237 vs 9.489.
    one = _identical_instruments(1)
    four = _identical_instruments(4)
    assert four.result.equity_curve()[-1] == pytest.approx(
        one.result.equity_curve()[-1]
    )  # premise
    assert len(one.result.equity_curve()) == 12
    assert len(four.result.equity_curve()) == 12
    assert four.metrics.sharpe_ratio == pytest.approx(one.metrics.sharpe_ratio)


# --------------------------------------------------------------------------
# tick backtest fills the stop at the stop level through a gap
# --------------------------------------------------------------------------
def test_tick_stop_fills_at_gapped_price_not_stop_level():
    # Entry at ask 100, 5% stop => 95. Next tick prints 80 (bid 80 too): no
    # trade ever occurred at 95, so the fill cannot be better than 80.
    # Observed exit_price == 95.0 (pnl -5 instead of -20).
    n = 4
    ts = np.arange(n, dtype=np.int64) * 1_000_000_000
    ltp = np.array([100.0, 100.0, 80.0, 80.0])
    z = np.zeros(n)
    ent = np.array([1, 0, 0, 0], bool)
    ex = np.zeros(n, bool)
    r = raptorbt.run_tick_backtest(
        ts, ltp, ltp.copy(), ltp.copy(), z, z, z, ent, ex,
        fees=0.0, slippage=0.0, stop_loss_pct=5.0, take_profit_pct=50.0,
    )
    t = r.trades()[0]
    assert t.exit_reason == "StopLoss"
    assert t.exit_price <= 80.0 + 1e-9


# --------------------------------------------------------------------------
# unsorted timestamps are accepted silently (release build)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "entry",
    ["portfolio_strategy", "tick_strategy", "strategy_backtest", "single_backtest"],
)
def test_unsorted_timestamps_are_refused(entry):
    n = 4
    p = np.full(n, 100.0)
    ts = np.array([0, 2, 1, 3], dtype=np.int64)
    bars = {"timestamps": ts, "open": p, "high": p + 1, "low": p - 1,
            "close": p, "volume": np.full(n, 1e3)}
    with pytest.raises(ValueError):
        if entry == "portfolio_strategy":
            raptorbt.run_portfolio_strategy(Strategy(), {"A": bars}, config=_cfg())
        elif entry == "tick_strategy":
            raptorbt.run_tick_strategy(
                Strategy(), {"A": {"timestamps": ts, "ltp": p}}, config=_cfg()
            )
        elif entry == "strategy_backtest":
            raptorbt.run_strategy_backtest(
                Strategy(), ts, bars["open"], bars["high"], bars["low"],
                bars["close"], bars["volume"], config=_cfg(),
            )
        else:
            flags = np.zeros(n, dtype=bool)
            raptorbt.run_single_backtest(
                ts, bars["open"], bars["high"], bars["low"], bars["close"],
                bars["volume"], flags, flags, config=_cfg(),
            )


def test_unsorted_error_names_stream_index_and_pair():
    p = np.full(4, 100.0)
    ts = np.array([0, 2, 1, 3], dtype=np.int64)
    bars = {"timestamps": ts, "open": p, "high": p + 1, "low": p - 1,
            "close": p, "volume": np.full(4, 1e3)}
    with pytest.raises(ValueError, match=r"A bars.*index 2 is 1 after 2"):
        raptorbt.run_portfolio_strategy(Strategy(), {"A": bars}, config=_cfg())


def test_equal_timestamps_are_accepted():
    p = np.full(4, 100.0)
    ts = np.array([0, 1, 1, 2], dtype=np.int64)
    bars = {"timestamps": ts, "open": p, "high": p + 1, "low": p - 1,
            "close": p, "volume": np.full(4, 1e3)}
    raptorbt.run_portfolio_strategy(Strategy(), {"A": bars}, config=_cfg())


def test_live_push_refuses_a_decreasing_timestamp():
    stream = raptorbt.TickStrategyStream(Strategy(), ["A"], config=_cfg())
    stream.push_tick("A", 10, 100.0)
    stream.push_tick("A", 10, 100.5)  # equal is fine
    with pytest.raises(ValueError, match=r"A live feed.*9 after 10"):
        stream.push_tick("A", 9, 100.0)


# --------------------------------------------------------------------------
# daily composite bars ignore config.session_tz_offset_ns
# (the config field is currently documented only for DAY-order expiry)
# --------------------------------------------------------------------------
def test_daily_composite_bar_aligns_to_session_tz():
    # Hourly bars from UTC 00:00. IST day 1 = UTC 00:00..18:30, i.e. bars 0..18
    # (19 bars). BarAggregator(1, "d", IST_OFFSET_NS) emits it at idx 19 with
    # volume 19 (checked directly below). The runner emits at idx 24, vol 24.
    n = 53
    p = np.full(n, 100.0)
    d = {"timestamps": np.arange(n, dtype=np.int64) * H_NS, "open": p, "high": p,
         "low": p, "close": p, "volume": np.ones(n)}

    agg = raptorbt.BarAggregator(1, "d", raptorbt.IST_OFFSET_NS)
    direct = [
        (i, r[5])
        for i in range(30)
        if (r := agg.push_bar(i * H_NS, 100, 100, 100, 100, 1.0)) is not None
    ]
    assert direct == [(19, 19.0)]  # premise: the primitive honours the offset

    class S(Strategy):
        def __init__(self, config=None):
            super().__init__(config)
            self.got = []

        def on_start(self, ctx):
            self.subscribe_bars(1, "d")

        def on_composite_bar(self, ctx, bar):
            self.got.append((ctx.idx, bar.volume))

    s = S()
    raptorbt.run_strategy_backtest(
        s, **d, config=_cfg(session_tz_offset_ns=raptorbt.IST_OFFSET_NS)
    )
    assert s.got[0] == (19, 19.0), s.got


# --------------------------------------------------------------------------
# a single NaN poisons SMA forever (NaN handling is undocumented)
# --------------------------------------------------------------------------
def test_sma_recovers_after_nan_leaves_the_window():
    # x = 1,2,3,NaN,5,6,7,8; period 3. Windows free of NaN: idx 6 = mean(5,6,7)
    # = 6, idx 7 = mean(6,7,8) = 7. Observed: NaN from idx 3 onward.
    x = np.array([1, 2, 3, np.nan, 5, 6, 7, 8.0])
    out = raptorbt.sma(x, 3)
    assert out[6] == pytest.approx(6.0)
    assert out[7] == pytest.approx(7.0)


# --------------------------------------------------------------------------
# Monte Carlo output depends on the Rayon thread count
# --------------------------------------------------------------------------
_MC_SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    import raptorbt
    rng = np.random.default_rng(0)
    ret = rng.normal(0.001, 0.01, (2, 50))
    out = raptorbt.simulate_portfolio_mc(
        ret, np.array([0.5, 0.5]), np.array([[1, 0.3], [0.3, 1.0]]),
        100000.0, n_simulations=200, horizon_days=10, seed=7,
    )
    print(json.dumps({k: v for k, v in out.items() if isinstance(v, float)}, sort_keys=True))
    """
)


def _mc(threads):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    out = subprocess.run(
        [sys.executable, "-c", _MC_SCRIPT], env=env, check=True,
        capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def test_control_same_thread_count_is_reproducible():
    assert _mc(4) == _mc(4)


def test_monte_carlo_is_independent_of_thread_count():
    # Same seed, same inputs. Observed var_95 2.069 (1 thread) vs 2.605 (4).
    assert _mc(1) == _mc(4)


# --------------------------------------------------------------------------
# session runs return an empty order log
# --------------------------------------------------------------------------
def test_portfolio_run_reports_its_orders():
    # One typed Market buy on A at bar 0; it fills (1 trade). CHANGELOG 0.13.3:
    # orders() has "one Order record per order the run placed".
    n = 10
    c = np.linspace(100.0, 110.0, n)
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000
    data = {"open": c, "high": c + 1, "low": c - 1, "close": c,
            "volume": np.full(n, 1e3), "timestamps": ts}

    class OneOrder(Strategy):
        done = False

        def on_bar(self, ctx):
            if ctx.symbol == "A" and not self.done:
                self.submit_order(orders.Market(side="buy", units=10))
                self.done = True

    res = raptorbt.run_portfolio_strategy(OneOrder(), {"A": data, "B": data}).result
    assert len(res.trades()) == 1  # premise: the order was placed and filled
    assert len(res.orders()) == 1


def test_tick_run_refuses_atr_stop_instead_of_ignoring_it():
    class S(Strategy):
        pass

    ticks = {"A": {"timestamps": np.arange(3, dtype=np.int64), "ltp": np.full(3, 100.0)}}
    with pytest.raises(ValueError, match=r"ATR-based stop.*period 3.*'A'.*no bars"):
        raptorbt.run_tick_strategy(S(), ticks, config=_atr_cfg())
    with pytest.raises(ValueError, match=r"ATR-based stop.*no bars"):
        raptorbt.TickStrategyStream(S(), ["A"], config=_atr_cfg())


def test_tick_primary_daily_bar_aligns_to_session_tz():
    # Hourly prints from UTC 00:00: the first IST day closes on the print at
    # UTC 19:00 (idx 19), not at the UTC midnight print (idx 24).
    n = 53
    ticks = {"A": {"timestamps": np.arange(n, dtype=np.int64) * H_NS,
                   "ltp": np.full(n, 100.0)}}

    class S(Strategy):
        def __init__(self, config=None):
            super().__init__(config)
            self.at = []

        def on_bar(self, ctx):
            self.at.append(ctx.idx)

        def on_trade_tick(self, ctx, tick):
            self.last = ctx.idx

    s = S()
    raptorbt.run_tick_strategy(
        s, ticks, primary_bars=(1, "d"),
        config=_cfg(session_tz_offset_ns=raptorbt.IST_OFFSET_NS),
    )
    assert s.at[0] == 19, s.at


def test_tick_run_reports_its_orders():
    # One typed Market buy from a trade print; it fills at the next print.
    n = 6
    ltp = np.linspace(100.0, 105.0, n)
    ts = np.arange(n, dtype=np.int64) * 1_000_000_000
    data = {"A": {"timestamps": ts, "ltp": ltp}}

    class OneOrder(Strategy):
        done = False

        def on_trade_tick(self, ctx, tick):
            if not self.done:
                self.submit_order(orders.Market(side="buy", units=10))
                self.done = True

    res = raptorbt.run_tick_strategy(OneOrder(), data, config=_cfg()).result
    assert len(res.trades()) == 1  # premise: the order was placed and filled
    log = res.orders()
    assert len(log) == 1
    assert log[0].symbol == "A"
    assert log[0].status == "filled"
    assert log[0].filled_qty == pytest.approx(10.0)
