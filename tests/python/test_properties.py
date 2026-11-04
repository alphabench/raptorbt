"""Randomised invariants through the Python API.

The kernel-level properties live in ``tests/properties.rs`` (proptest). These
are the ones only the Python surface can drive: the array runners, the
strategy layer, the batch and Monte Carlo entry points, and bar stamps.

Every case draws from a numpy generator with a fixed seed, so a failure names
its seed and reruns identically. No new dependencies.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, BatchSingleItem, InstrumentSpec, Strategy
from raptorbt.strategy import orders

import synthetic  # noqa: E402
from test_batch_single import assert_identical  # noqa: E402

INITIAL = 100_000.0
NS_MIN = 60 * 10**9
FAR_FUTURE_NS = 4 * 10**18
# Relative tolerance on money: far above float accumulation over a few dozen
# fills (numerics.md), far below any accounting mistake.
REL = 1e-9
# Cash is booked in whole minor units. While a position is open its cost was
# debited rounded to the paisa (half a paisa at most); the model below marks
# the position at the exact price. A short's cost enters its marked value
# twice (the collateral and the gain), and the booked close at the end of the
# data rounds the exit notional too, so up to three half-paisas per position.
# Closed trades carry no such gap, because their P&L is the booked P&L.
BOOK_GRAIN = 0.015


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
# 2026-09-02 00:00 UTC: a whole UTC day, and inside the dated fee rows (a fill before a schedule's
# first row is refused, and some runs here price on a fee segment)
T0 = 1_788_307_200 * 10**9


def _bars(rng, n, start=100.0):
    """Positive random-walk OHLCV with gaps and wicks."""
    ret = rng.normal(0.0, 0.012, n)
    close = start * np.exp(np.cumsum(ret))
    gap = rng.normal(0.0, 0.004, n)
    open_ = np.concatenate([[start], close[:-1]]) * (1.0 + gap)
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    return {
        "timestamps": T0 + np.arange(n, dtype=np.int64) * NS_MIN,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1_000.0),
    }


def _signals(rng, n, p=0.12):
    return rng.random(n) < p, rng.random(n) < p


def _config(rng, **extra):
    kwargs = {
        "initial_capital": INITIAL,
        "fees": float(rng.choice([0.0, 0.0005, 0.003])),
        "slippage": float(rng.choice([0.0, 0.001])),
        "fill_timing": str(rng.choice(["same_bar_close", "next_bar_open"])),
    }
    if rng.random() < 0.25:
        kwargs["fee_segment"] = str(rng.choice(["NSE", "NSE-DELIVERY", "NFO-FUT", "MCX"]))
    kwargs.update(extra)
    cfg = BacktestConfig(**kwargs)
    stop, target = rng.random() < 0.3, rng.random() < 0.3
    if stop:
        cfg.set_fixed_stop(float(rng.uniform(0.01, 0.05)))
    if target:
        cfg.set_fixed_target(float(rng.uniform(0.01, 0.05)))
    return cfg


# ---------------------------------------------------------------------------
# 1. accounting identity
# ---------------------------------------------------------------------------
def _model_equity(trades, closes, n, mult=1.0):
    """Equity implied by trade records alone.

    Initial capital, plus the PnL of every closed trade from its exit bar on,
    plus each open trade's unrealised gain less the entry costs already paid.
    A trade closed only by end-of-data is marked open through the last bar,
    which is when the curve is sampled.
    """
    eq = np.full(n, INITIAL)
    for t in trades:
        end_of_data = t.exit_reason == "EndOfData"
        stop = n if end_of_data else t.exit_idx
        c = closes[t.symbol][t.entry_idx : stop]
        eq[t.entry_idx : stop] += t.direction * (c - t.entry_price) * t.size * mult - t.entry_fees
        if not end_of_data:
            eq[t.exit_idx :] += t.pnl
    return eq


def _assert_equity_matches(result, closes, n, label, mult=1.0):
    trades = result.trades()
    got = result.equity_curve()
    want = _model_equity(trades, closes, n, mult)
    open_at = np.zeros(n)
    for t in trades:
        open_at[t.entry_idx : n if t.exit_reason == "EndOfData" else t.exit_idx] += 1
    atol = REL * INITIAL + BOOK_GRAIN * max(1.0, open_at.max())
    np.testing.assert_allclose(got, want, rtol=0, atol=atol, err_msg=label)
    total = sum(t.pnl for t in trades)
    # The last sample marks a position still open at the end of the data at its
    # exact price; the booked close that follows rounds its notional.
    assert got[-1] == pytest.approx(INITIAL + total, rel=REL, abs=atol), label
    for t in trades:
        assert t.fees == pytest.approx(t.entry_fees + t.exit_fees, rel=REL, abs=1e-12), label
    return trades


@pytest.mark.parametrize("seed", range(100))
def test_single_backtest_equity_is_the_marked_book_on_every_bar(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(20, 120))
    b = _bars(rng, n)
    entries, exits = _signals(rng, n)
    direction = int(rng.choice([1, -1]))
    cfg = _config(rng)
    result = raptorbt.run_single_backtest(
        **b, entries=entries, exits=exits, direction=direction, symbol="S", config=cfg
    )
    _assert_equity_matches(result, {"S": b["close"]}, n, f"seed={seed}")


@pytest.mark.parametrize("seed", range(60))
def test_portfolio_backtest_equity_is_the_marked_book_on_every_bar(seed):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(20, 100))
    syms = [f"P{k}" for k in range(int(rng.integers(2, 5)))]
    data = {s: _bars(rng, n, start=float(rng.uniform(50, 200))) for s in syms}
    legs = []
    for s in syms:
        e, x = _signals(rng, n)
        b = data[s]
        legs.append(
            (b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"],
             e, x, int(rng.choice([1, -1])), 1.0, s)
        )
    cfg = _config(rng)
    pr = raptorbt.run_portfolio_backtest(
        legs, config=cfg, allocation=str(rng.choice(["equal_weight", "full"]))
    )
    closes = {s: data[s]["close"] for s in syms}
    _assert_equity_matches(pr.result, closes, n, f"seed={seed}")


@pytest.mark.parametrize("seed", range(12))
def test_pairs_backtest_final_equity_is_capital_plus_pnl(seed):
    """Per-bar marking of a pair depends on the hedge-ratio convention, so the
    pair is held to the end-state identity only."""
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(20, 100))
    b1, b2 = _bars(rng, n), _bars(rng, n, start=50.0)
    entries, exits = _signals(rng, n)
    result = raptorbt.run_pairs_backtest(
        b1["timestamps"], b1["open"], b1["high"], b1["low"], b1["close"], b1["volume"],
        b2["timestamps"], b2["open"], b2["high"], b2["low"], b2["close"], b2["volume"],
        entries, exits,
        direction=int(rng.choice([1, -1])),
        config=_config(rng),
        hedge_ratio=float(rng.uniform(0.5, 2.0)),
    )
    trades = result.trades()
    assert result.equity_curve()[-1] == pytest.approx(
        INITIAL + sum(t.pnl for t in trades), rel=REL
    ), f"seed={seed}"
    for t in trades:
        assert t.fees == pytest.approx(t.entry_fees + t.exit_fees, rel=REL, abs=1e-12)


def test_portfolio_reports_a_trade_stopped_out_on_its_fill_bar():
    n = 5
    price = np.full(n, 100.0)
    low = price.copy()
    low[2] = 97.0
    entries = np.zeros(n, bool)
    entries[1] = True
    cfg = BacktestConfig(initial_capital=INITIAL, fees=0.0, slippage=0.0,
                         fill_timing="next_bar_open")
    cfg.set_fixed_stop(0.01)
    leg = (np.arange(n, dtype=np.int64), price, price, low, price, np.full(n, 1e3),
           entries, np.zeros(n, bool), 1, 1.0, "A")
    pr = raptorbt.run_portfolio_backtest([leg], config=cfg)
    assert pr.result.equity_curve()[-1] == pytest.approx(INITIAL - 1000.0)
    assert [t.pnl for t in pr.result.trades()] == [pytest.approx(-1000.0)]


class _Scripted(Strategy):
    """Enters and closes on a fixed per-bar schedule."""

    def __init__(self, enter_at, close_at, size_frac):
        super().__init__()
        self.enter_at, self.close_at, self.size_frac = enter_at, close_at, size_frac

    def on_bar(self, ctx):
        if ctx.idx in self.close_at and ctx.position is not None:
            self.close_position()
        elif ctx.idx in self.enter_at and ctx.position is None:
            self.enter(size_frac=self.size_frac)


@pytest.mark.parametrize("seed", range(80))
def test_strategy_equity_is_the_marked_book_in_cash_and_margin(seed):
    """Margin accounts, tick grids and contract multipliers, which the array
    runners cannot configure."""
    rng = np.random.default_rng(4000 + seed)
    n = int(rng.integers(20, 100))
    b = _bars(rng, n)
    mult = float(rng.choice([1.0, 1.0, 5.0, 50.0]))
    incr = float(rng.choice([0.0, 0.01, 0.05, 0.5]))
    margin = rng.random() < 0.5
    if mult != 1.0 or margin:
        spec = InstrumentSpec.futures_contract(
            "S", FAR_FUTURE_NS, 1.0, multiplier=mult, price_increment=incr
        )
    else:
        spec = InstrumentSpec.equity("S", price_increment=incr)
    cfg = _config(rng)
    strat = _Scripted(
        set(np.flatnonzero(rng.random(n) < 0.15).tolist()),
        set(np.flatnonzero(rng.random(n) < 0.15).tolist()),
        float(rng.uniform(0.2, 1.0)),
    )
    result = raptorbt.run_strategy_backtest(
        strat, **b, direction=int(rng.choice([1, -1])), symbol="S", config=cfg,
        instrument=spec,
        account_type="margin" if margin else "cash",
        leverage=float(rng.uniform(1.5, 8.0)),
        bar_stamp="close",
    )
    trades = _assert_equity_matches(result, {"S": b["close"]}, n, f"seed={seed}", mult)
    if incr > 0.0:
        for t in trades:
            if t.exit_reason not in ("EndOfData", "StopLoss", "TakeProfit"):
                for price in (t.entry_price, t.exit_price):
                    ticks = price / incr
                    assert abs(ticks - round(ticks)) <= 1e-9 * max(abs(round(ticks)), 1.0), (
                        seed, price, incr)


# ---------------------------------------------------------------------------
# 4. fill timing for array signals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_next_bar_open_signals_never_trade_on_their_decision_bar(seed):
    rng = np.random.default_rng(5000 + seed)
    n = int(rng.integers(20, 120))
    b = _bars(rng, n)
    entries, exits = _signals(rng, n)
    cfg = BacktestConfig(
        initial_capital=INITIAL, fees=0.0005, slippage=0.0, fill_timing="next_bar_open"
    )
    result = raptorbt.run_single_backtest(
        **b, entries=entries, exits=exits, direction=int(rng.choice([1, -1])), config=cfg
    )
    trades = result.trades()
    for t in trades:
        assert t.entry_idx >= 1 and entries[t.entry_idx - 1], (seed, t.entry_idx)
        assert t.entry_time == b["timestamps"][t.entry_idx]
        assert t.entry_price == b["open"][t.entry_idx]
        if t.exit_reason == "Signal":
            assert exits[t.exit_idx - 1], (seed, t.exit_idx)
            assert t.exit_price == b["open"][t.exit_idx]


# ---------------------------------------------------------------------------
# 3-6. random order command sequences through the strategy layer
# ---------------------------------------------------------------------------
TIFS = ("gtc", "gtc", "gtc", "day", "ioc", "fok", "gtd")
KINDS = ("market", "market", "limit", "limit", "stop_market", "stop_limit",
         "market_if_touched", "limit_if_touched", "market_to_limit", "trailing_stop_market")
STATUSES = {"submitted", "accepted", "triggered", "partially_filled", "filled",
            "canceled", "expired", "rejected"}
TERMINAL = {"filled", "canceled", "expired", "rejected"}


class _OrderScript(Strategy):
    """Plays a random command script: submit, cancel, modify."""

    def __init__(self, script, incr):
        super().__init__()
        self.script, self.incr = script, incr
        self.ids = []

    def _px(self, close, off):
        p = close * (1.0 + off)
        p = round(p / self.incr) * self.incr if self.incr else round(p, 2)
        return max(p, self.incr or 0.01)

    def _order(self, close, spec):
        kind, side, off, off2, qty, tif, reduce_only, ts = spec
        common = dict(side=side, tif=tif, reduce_only=reduce_only)
        if tif == "gtd":
            common["expire_ns"] = ts + 5 * NS_MIN
        if qty is not None:
            common[qty[0]] = qty[1]
        p, p2 = self._px(close, off), self._px(close, off2)
        if kind == "market":
            return orders.Market(**common)
        common.pop("expire_ns", None) if tif != "gtd" else None
        if tif in ("at_open", "at_close"):
            common["tif"] = "gtc"
        if kind == "limit":
            return orders.Limit(price=p, **common)
        if kind == "stop_market":
            return orders.StopMarket(trigger=p, **common)
        if kind == "stop_limit":
            return orders.StopLimit(trigger=p, price=p2, **common)
        if kind == "market_if_touched":
            return orders.MarketIfTouched(trigger=p, **common)
        if kind == "limit_if_touched":
            return orders.LimitIfTouched(trigger=p, price=p2, **common)
        if kind == "market_to_limit":
            return orders.MarketToLimit(**common)
        return orders.TrailingStopMarket(offset=abs(off) * close + 0.01, **common)

    def on_bar(self, ctx):
        for cmd in self.script.get(ctx.idx, []):
            if cmd[0] == "submit":
                self.ids.append(self.submit_order(self._order(ctx.bar.close, cmd[1])))
            elif cmd[0] == "cancel" and self.ids:
                self.cancel_order(self.ids[cmd[1] % len(self.ids)])
            elif cmd[0] == "modify" and self.ids:
                self.modify_order(
                    self.ids[cmd[1] % len(self.ids)],
                    limit_price=self._px(ctx.bar.close, cmd[2]),
                )


def _random_script(rng, bars):
    n = len(bars["close"])
    script = {}
    for i in range(n):
        cmds = []
        for _ in range(int(rng.choice([0, 0, 1, 1, 2]))):
            r = rng.random()
            if r < 0.75:
                kind = str(rng.choice(KINDS))
                tif = str(rng.choice(TIFS))
                if kind == "market" and rng.random() < 0.15:
                    tif = str(rng.choice(["at_open", "at_close"]))
                qty = [("units", float(rng.integers(1, 200))),
                       ("size_frac", float(rng.uniform(0.1, 1.0))), None][int(rng.integers(0, 3))]
                cmds.append(("submit", (
                    kind, str(rng.choice(["buy", "sell"])), float(rng.uniform(-0.04, 0.04)),
                    float(rng.uniform(-0.04, 0.04)), qty, tif, bool(rng.random() < 0.25),
                    int(bars["timestamps"][i]),
                )))
            elif r < 0.9:
                cmds.append(("cancel", int(rng.integers(0, 64))))
            else:
                cmds.append(("modify", int(rng.integers(0, 64)), float(rng.uniform(-0.04, 0.04))))
        script[i] = cmds
    return script


def _run_script(seed):
    rng = np.random.default_rng(6000 + seed)
    n = int(rng.integers(20, 80))
    b = _bars(rng, n)
    incr = float(rng.choice([0.0, 0.01, 0.05, 0.5]))
    timing = str(rng.choice(["same_bar_close", "next_bar_open"]))
    cfg = BacktestConfig(
        initial_capital=INITIAL, fees=float(rng.choice([0.0, 0.001])),
        slippage=float(rng.choice([0.0, 0.001])), fill_timing=timing,
    )
    spec = InstrumentSpec.equity("S", price_increment=incr)
    result = raptorbt.run_strategy_backtest(
        _OrderScript(_random_script(rng, b), incr), **b, direction=1, symbol="S",
        config=cfg, instrument=spec, bar_stamp="close",
    )
    return result, incr, timing, b


@pytest.mark.parametrize("seed", range(120))
def test_random_order_sequences_stay_legal_and_consistent(seed):
    result, incr, timing, b = _run_script(seed)
    book = result.orders()
    assert book, "the script placed no orders"
    for o in book:
        label = (seed, o.id, o.kind, o.status)
        assert o.status in STATUSES, label
        # Lifecycle versus fills.
        if o.requested_qty is not None:
            assert o.filled_qty <= o.requested_qty + 1e-9, label
            if o.filled_qty + 1e-9 >= o.requested_qty:
                assert o.status == "filled", label
        assert (o.status == "filled") == (o.filled_qty > 0.0), label
        if o.status not in TERMINAL:
            assert o.filled_qty == 0.0, label
        if o.filled_qty == 0.0:
            continue
        # Timing: a fill is never on the bar that decided the order if the
        # order rests, nor under next_bar_open for any order.
        resting = o.kind != "market" or o.tif in ("at_open", "at_close")
        # A plain limit marketable at the decision bar's close trades there under
        # same_bar_close, like a market order placed on that bar.
        close = b["close"][o.submitted_idx]
        marketable = (
            o.kind == "limit"
            and timing == "same_bar_close"
            and o.tif not in ("at_open", "at_close")
            and o.limit_price is not None
            and (o.limit_price > close if o.side == "buy" else o.limit_price < close)
        )
        if marketable and o.last_fill_idx == o.submitted_idx:
            pass
        elif resting or timing == "next_bar_open":
            assert o.last_fill_idx > o.submitted_idx, label
        else:
            assert o.last_fill_idx == o.submitted_idx, label
        # Grid and limits (single-slice fills: partial fills are off).
        if incr > 0.0:
            ticks = o.avg_fill_price / incr
            assert abs(ticks - round(ticks)) <= 1e-9 * max(abs(round(ticks)), 1.0), label
        if o.limit_price is not None:
            slack = 1e-9 * o.limit_price
            if o.side == "buy":
                assert o.avg_fill_price <= o.limit_price + slack, label
            else:
                assert o.avg_fill_price >= o.limit_price - slack, label


# ---------------------------------------------------------------------------
# Order edge cases reachable from the strategy layer (the kernel-level
# reproductions are in tests/properties.rs)
# ---------------------------------------------------------------------------
def _flat(n, price=100.0):
    return {
        "timestamps": T0 + np.arange(n, dtype=np.int64) * NS_MIN,
        "open": np.full(n, price), "high": np.full(n, price),
        "low": np.full(n, price), "close": np.full(n, price),
        "volume": np.full(n, 1_000.0),
    }


class _Script(Strategy):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def on_bar(self, ctx):
        for fn in self.plan.get(ctx.idx, []):
            fn(self)


def test_bracket_with_both_legs_inside_one_bar_does_not_panic():
    b = _flat(3)
    b["high"][2], b["low"][2] = 102.0, 98.0
    plan = {0: [lambda s: s.submit_bracket(
        orders.Market(side="buy", units=1.0), stop_trigger=99.5, target_price=100.5)]}
    result = raptorbt.run_strategy_backtest(
        _Script(plan), **b, config=BacktestConfig(fees=0.0), bar_stamp="close"
    )
    assert all(o.status in TERMINAL or o.filled_qty == 0 for o in result.orders())


def test_limit_if_touched_fill_does_not_cross_its_limit_under_slippage():
    plan = {
        0: [lambda s: s.submit_order(orders.Market(side="sell", units=1.0))],
        1: [lambda s: s.submit_order(
            orders.LimitIfTouched(side="buy", units=1.0, trigger=100.0, price=100.0))],
    }
    result = raptorbt.run_strategy_backtest(
        _Script(plan), **_flat(4), config=BacktestConfig(fees=0.0, slippage=0.002),
        bar_stamp="close",
    )
    (lit,) = [o for o in result.orders() if o.kind == "limit_if_touched"]
    assert lit.filled_qty == 1.0 and lit.avg_fill_price <= 100.0 + 1e-9


def test_reduce_only_sell_while_short_is_refused():
    plan = {
        0: [lambda s: s.submit_order(orders.Market(side="sell", units=5.0))],
        1: [lambda s: s.submit_order(orders.Market(side="sell", units=5.0, reduce_only=True))],
    }
    result = raptorbt.run_strategy_backtest(
        _Script(plan), **_flat(3), config=BacktestConfig(fees=0.0), bar_stamp="close"
    )
    assert result.orders()[1].status == "rejected"


# ---------------------------------------------------------------------------
# 7. determinism
# ---------------------------------------------------------------------------
def _fingerprint(result):
    trades = [
        (t.entry_idx, t.exit_idx, t.entry_price, t.exit_price, t.size, t.pnl, t.fees,
         t.exit_reason, t.direction)
        for t in result.trades()
    ]
    book = [(o.id, o.status, o.filled_qty, o.avg_fill_price, o.last_fill_idx)
            for o in result.orders()]
    m = result.metrics
    metrics = tuple(getattr(m, k) for k in dir(m)
                    if not k.startswith("_") and not callable(getattr(m, k)))
    return (result.equity_curve().tobytes(), result.returns().tobytes(), trades, book, metrics)


@pytest.mark.parametrize("seed", range(10))
def test_repeated_runs_are_bit_identical(seed):
    rng = np.random.default_rng(7000 + seed)
    n = int(rng.integers(30, 120))
    b = _bars(rng, n)
    entries, exits = _signals(rng, n)
    cfg = _config(rng, fill_prob_limit=0.5, fill_prob_slippage=0.3,
                  fill_seed=int(rng.integers(0, 2**31)))
    direction = int(rng.choice([1, -1]))

    def single():
        return _fingerprint(raptorbt.run_single_backtest(
            **b, entries=entries, exits=exits, direction=direction, config=cfg))

    def scripted():
        r = np.random.default_rng(9000 + seed)
        script = _random_script(r, b)
        return _fingerprint(raptorbt.run_strategy_backtest(
            _OrderScript(script, 0.05), **b, config=cfg,
            instrument=InstrumentSpec.equity("S", price_increment=0.05),
            bar_stamp="close"))

    for run in (single, scripted):
        first = run()
        assert run() == first and run() == first, f"{run.__name__} seed={seed}"


@pytest.mark.parametrize("seed", range(3))
def test_batch_matches_serial_for_random_items(seed):
    rng = np.random.default_rng(8000 + seed)
    n = 400
    b = _bars(rng, n)
    base = BacktestConfig(initial_capital=INITIAL, fees=0.0005)
    items, serial = [], []
    for k in range(24):
        e, x = _signals(rng, n, p=float(rng.uniform(0.02, 0.2)))
        direction = int(rng.choice([1, -1]))
        cfg = _config(rng)
        items.append(BatchSingleItem(f"i{k}", e, x, direction, 1.0, "B", cfg))
        serial.append(raptorbt.run_single_backtest(
            **b, entries=e, exits=x, direction=direction, symbol="B", config=cfg))
    batch = raptorbt.batch_single_backtest(
        b["timestamps"], b["open"], b["high"], b["low"], b["close"], b["volume"], items, base)
    assert [i for i, _ in batch] == [f"i{k}" for k in range(24)]
    for (item_id, got), want in zip(batch, serial):
        assert_identical(want, got, f"seed={seed} {item_id}")
        assert _fingerprint(got) == _fingerprint(want), f"seed={seed} {item_id}"


_MC_CHILD = textwrap.dedent(
    """
    import hashlib, json, sys
    import numpy as np
    import raptorbt

    seed = int(sys.argv[1])
    rng = np.random.default_rng(seed)
    k = 4
    returns = [rng.normal(0.0004, 0.012, 250) for _ in range(k)]
    a = rng.normal(size=(k, k))
    corr = a @ a.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    w = rng.random(k)
    w /= w.sum()
    out = raptorbt.simulate_portfolio_mc(
        returns, w, [corr[i] for i in range(k)], 1e6,
        n_simulations=4000, horizon_days=60, seed=seed,
    )
    scalars = [out[x] for x in ("expected_return", "probability_of_loss", "var_95", "cvar_95")]
    # repr of a float round-trips exactly, so equal digests mean equal bits.
    blob = repr((out["percentile_paths"], np.asarray(out["final_values"]).tolist(), scalars))
    h = hashlib.sha256(blob.encode())
    print(json.dumps({"digest": h.hexdigest()}))
    """
)


def _mc(seed, threads):
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    out = subprocess.run(
        [sys.executable, "-c", _MC_CHILD, str(seed)],
        env=env, capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(out.stdout)


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_portfolio_monte_carlo_is_independent_of_thread_count(seed):
    one, four = _mc(seed, 1), _mc(seed, 4)
    assert one == four
    assert _mc(seed, 4) == four


# ---------------------------------------------------------------------------
# 8. stamp equivalence, randomised
# ---------------------------------------------------------------------------
with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class _Rand(Strategy):
        """Random-parameter strategy touching fills, resting orders, DAY expiry,
        composite bars and the clock; records only close-time quantities."""

        def __init__(self, p):
            super().__init__()
            self.p = p
            self.trace = []
            self.raw = []

        def on_start(self, ctx):
            syms = list(getattr(ctx, "symbols", [None]))
            self.fast = self.register_indicators(lambda: raptorbt.Indicator.sma(self.p["fast"]), syms)
            self.slow = self.register_indicators(lambda: raptorbt.Indicator.sma(self.p["slow"]), syms)
            self.comp_stream = self.subscribe_bars(self.p["comp"], "m")
            self.last = {}

        @staticmethod
        def _sym(ctx):
            return getattr(ctx, "symbol", None)

        def on_composite_bar(self, ctx, bar):
            self.last[bar.symbol] = bar.close
            self.trace.append(("comp", self._sym(ctx), tuple(bar)))

        def on_fill(self, ctx, e):
            self.trace.append(("fill", self._sym(ctx), ctx.bar_close_ts, e.kind, e.price, e.size))

        def on_order_expired(self, ctx, e):
            self.trace.append(("expired", self._sym(ctx), ctx.bar_close_ts))

        def on_bar(self, ctx):
            sym, now, pos = self._sym(ctx), ctx.bar_close_ts, ctx.position
            self.raw.append((sym, ctx.timestamp, now))
            fast, slow = self.fast[sym].value, self.slow[sym].value
            self.trace.append(("bar", sym, now, tuple(ctx.bar)[1:], fast, slow, ctx.equity,
                               None if pos is None else (pos.size, pos.entry_price)))
            minute = ((now + synthetic.IST_OFFSET_NS) // NS_MIN) % 1440
            late = minute >= 15 * 60 + 25
            if pos is not None and late:
                self.close_position()
                return
            if ctx.idx % self.p["limit_every"] == 3 and pos is None and not late:
                self.submit_order(orders.Limit(
                    side="buy", units=2.0, price=round(ctx.bar.close * self.p["limit_off"], 2),
                    tif=self.p["tif"]))
            if fast is None or slow is None or late:
                return
            if pos is None and fast > slow:
                self.enter(size_frac=self.p["frac"])
            elif pos is not None and fast < slow:
                self.close_position()


def _outcome(result):
    def pub(obj):
        return {k: repr(getattr(obj, k)) for k in dir(obj)
                if not k.startswith("_") and not callable(getattr(obj, k))}
    return {
        "trades": [pub(t) for t in result.trades()],
        "orders": [pub(o) for o in result.orders()],
        "equity": result.equity_curve().tolist(),
        "returns": result.returns().tolist(),
        "metrics": pub(result.metrics),
    }


def _stamped(bars, stamp):
    ts = bars["ts_open"] if stamp == "open" else bars["ts_close"]
    out = {"timestamps": np.ascontiguousarray(ts)}
    for key in ("open", "high", "low", "close", "volume"):
        out[key] = np.ascontiguousarray(bars[key])
    return out


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("runner", ["single", "portfolio"])
def test_open_and_close_stamped_runs_agree_for_random_widths_and_data(seed, runner, portfolio_runner):
    rng = np.random.default_rng(10_000 + seed)
    tf = str(rng.choice(["1m", "5m", "15m"]))
    width_min = {"1m": 1, "5m": 5, "15m": 15}[tf]
    ds = synthetic.generate(
        n_instruments=3, days=int(rng.integers(1, 3)), seed=int(rng.integers(1, 10**6)),
        timeframes=(tf,),
    )
    bars = ds.bars[tf]
    params = {
        "fast": int(rng.integers(2, 6)),
        "slow": int(rng.integers(7, 20)),
        "comp": width_min * int(rng.choice([3, 5])),
        "limit_every": int(rng.integers(7, 40)),
        "limit_off": float(rng.uniform(0.99, 0.9995)),
        "tif": str(rng.choice(["day", "gtc"])),
        "frac": float(rng.uniform(0.05, 0.3)),
    }
    cfg = BacktestConfig(
        initial_capital=10_000_000.0, fees=0.0002,
        fill_timing=str(rng.choice(["same_bar_close", "next_bar_open"])),
        session_tz_offset_ns=synthetic.IST_OFFSET_NS,
    )
    width_ns = width_min * NS_MIN

    def run(stamp):
        strat = _Rand(params)
        kw = ({"bar_stamp": "open", "bar_ns": width_ns} if stamp == "open"
              else {"bar_stamp": "close"})
        if runner == "single":
            sym = next(iter(bars))
            res = raptorbt.run_strategy_backtest(
                strat, **_stamped(bars[sym], stamp), config=cfg, symbol=sym, **kw)
        else:
            data = {s: _stamped(b, stamp) for s, b in bars.items()}
            if stamp == "open":
                kw["bar_ns"] = {s: width_ns for s in data}
            res = portfolio_runner(strat, data, config=cfg, **kw).result
        return strat, res

    s_open, r_open = run("open")
    s_close, r_close = run("close")
    label = f"seed={seed} tf={tf} params={params}"
    assert s_open.trace == s_close.trace, label
    assert _outcome(r_open) == _outcome(r_close), label
    kinds = {t[0] for t in s_close.trace}
    assert {"bar", "comp"} <= kinds, label
    assert r_close.trades(), label
    for (sym_o, raw_o, vis_o), (sym_c, raw_c, vis_c) in zip(s_open.raw, s_close.raw):
        assert sym_o == sym_c and vis_o == vis_c, label
        assert raw_o == vis_o - width_ns and raw_c == vis_c, label
