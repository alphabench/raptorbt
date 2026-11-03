"""Focused checks of ``run_strategies`` against ``run_portfolio_strategy``."""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Indicator, Strategy, run_portfolio_strategy, run_strategies
from raptorbt.errors import InvalidParameter
from raptorbt.strategy.orders import Limit

MIN = 60_000_000_000
T0 = 1_700_000_000_000_000_000
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def make_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    data = {}
    for k, s in enumerate(SYMBOLS):
        # A different phase per symbol so some bars tie on timestamp, some do not.
        ts = (np.arange(n, dtype=np.int64) + (k % 2)) * MIN + T0
        close = 100 + 10 * k + np.cumsum(rng.normal(0, 0.8, n))
        close = np.maximum(close, 1.0)
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) + rng.random(n)
        low = np.maximum(np.minimum(open_, close) - rng.random(n), 0.5)
        data[s] = {
            "timestamps": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.random(n) * 1000 + 100,
        }
    return data


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class Recorder(Strategy):
        """Exercises indicators, composites, timers, typed orders and intents."""

        def __init__(self, fast=5, slow=12, step=4, **kw):
            super().__init__(**kw)
            self.fast_n, self.slow_n, self.step = fast, slow, step
            self.rec = []

        def on_start(self, ctx):
            self.fast = {
                s: self.register_indicator(Indicator.sma(self.fast_n), symbol=s) for s in ctx.symbols
            }
            self.rsi = {
                s: self.register_indicator(Indicator.rsi(self.slow_n), symbol=s) for s in ctx.symbols
            }
            self.sid = self.subscribe_bars(self.step, "m")
            self.comp = {
                s: self.register_indicator(Indicator.ema(3), stream_id=self.sid, symbol=s)
                for s in ctx.symbols
            }
            self.clock.set_timer("t", 30 * MIN, start_ns=T0)
            self.rec.append(("start", tuple(ctx.symbols), ctx.equity, ctx.cash))

        def seen(self, ctx):
            s = ctx.symbol
            return (
                self.fast[s].value,
                self.rsi[s].initialized,
                self.comp[s].value,
                ctx.equity,
                ctx.cash,
                ctx.free_capital,
                ctx.net_position,
                ctx.is_flat,
            )

        def on_timer(self, ctx, event):
            self.rec.append(("time", ctx.symbol, ctx.idx, ctx.timestamp, event, self.seen(ctx)))

        def on_composite_bar(self, ctx, bar):
            self.rec.append(("comp", ctx.symbol, ctx.idx, bar, self.seen(ctx)))

        def on_bar(self, ctx):
            self.rec.append(("bar", ctx.symbol, ctx.idx, ctx.bar, ctx.bar_close_ts, self.seen(ctx)))
            v = self.fast[ctx.symbol].value
            if v is None:
                return
            i = ctx.idx
            if ctx.is_flat and ctx.bar.close > v and i % 3 == 0:
                self.enter(size_frac=0.3)
            elif not ctx.is_flat and ctx.bar.close < v:
                self.close_position()
            elif i % 17 == 0:
                self.submit_order(Limit(side="buy", price=ctx.bar.close * 0.99, units=1.0))
            elif i % 19 == 0:
                self.cancel_all_orders()

        def on_order_event(self, ctx, event):
            self.rec.append(("order", ctx.symbol, event.kind, event.price, event.size))

        def on_position_closed(self, ctx, event):
            self.rec.append(("closed", ctx.symbol, event.trade.pnl))

        def on_stop(self, ctx):
            self.rec.append(("stop", ctx.equity))


def fields(obj):
    return {
        a: getattr(obj, a)
        for a in dir(obj)
        if not a.startswith("_") and not callable(getattr(obj, a))
    }


def fingerprint(res):
    r = res.result
    return {
        "trades": [fields(t) for t in r.trades()],
        "orders": [fields(o) for o in r.orders()],
        "equity": r.equity_curve().tobytes(),
        "drawdown": r.drawdown_curve().tobytes(),
        "returns": r.returns().tobytes(),
        "metrics": fields(res.metrics),
        "halted": res.halted,
        "halted_at": res.halted_at,
        "rejected": res.rejected_entries,
        "per_instrument": [fields(p) for p in res.per_instrument],
    }


def same(a, b):
    """Equality that treats NaN as equal to NaN."""
    assert repr(a) == repr(b)


KW = dict(bar_stamp="close")


def legacy(strategy, data, syms, config=None, **kw):
    sub = {k: data[k] for k in data if k in syms}
    return run_portfolio_strategy(strategy, sub, config, **{**KW, **kw})


def test_one_strategy_equals_the_legacy_runner():
    data = make_data()
    cfg = BacktestConfig(fees=0.001)
    legacy_s, shared_s = Recorder(), Recorder()
    want = legacy(legacy_s, data, SYMBOLS, cfg)
    (got,) = run_strategies([shared_s], data, cfg, **KW)
    same(fingerprint(got), fingerprint(want))
    assert shared_s.rec == legacy_s.rec
    assert len(want.result.trades()) > 3
    assert any(e[0] == "comp" for e in shared_s.rec)
    assert any(e[0] == "time" for e in shared_s.rec)


def test_indicators_are_linked_when_shared_and_private_otherwise():
    data = make_data(n=60)
    a = Recorder()
    run_strategies([a], data, **KW)
    assert all(i.linked for i in a.fast.values())
    assert all(i.linked for i in a.comp.values())
    b = Recorder()
    run_strategies([b], data, share=False, **KW)
    assert not any(i.linked for i in b.fast.values())
    assert all(i.update_count > 0 for i in b.fast.values())


def test_a_linked_indicator_refuses_manual_updates():
    class Once(Strategy):
        def on_start(self, ctx):
            self.ind = self.register_indicator(Indicator.sma(3), symbol=ctx.symbols[0])

    s = Once()
    run_strategies([s], make_data(n=10), **KW)
    assert s.ind.linked
    with pytest.raises(RuntimeError, match="linked"):
        s.ind.update_bar(1, 1, 1, 1)
    with pytest.raises(RuntimeError, match="linked"):
        s.ind.reset()


def test_touched_or_doubly_registered_indicators_stay_private():
    class Mixed(Strategy):
        def on_start(self, ctx):
            s = ctx.symbols[0]
            self.fed = Indicator.sma(3)
            self.fed.update_bar(1, 1, 1, 1)
            self.register_indicator(self.fed, symbol=s)
            self.twice = Indicator.sma(3)
            self.register_indicator(self.twice, symbol=s)
            self.register_indicator(self.twice, symbol=s)
            self.plain = self.register_indicator(Indicator.sma(3), symbol=s)

    s = Mixed()
    run_strategies([s], make_data(n=10), **KW)
    assert not s.fed.linked and not s.twice.linked and s.plain.linked


def test_twenty_strategies_each_equal_their_independent_run():
    data = make_data(3)
    cfg = BacktestConfig(fees=0.001, max_drawdown_pct=12.0)
    rng = np.random.default_rng(5)
    specs = []
    for i in range(20):
        k = int(rng.integers(1, 6))
        syms = sorted(rng.choice(SYMBOLS, size=k, replace=False).tolist())
        specs.append((syms, dict(fast=int(rng.integers(3, 9)), slow=int(rng.integers(8, 20)),
                                 step=int(rng.integers(2, 6)))))
    many = [(Recorder(**kw), syms) for syms, kw in specs]
    out = run_strategies(many, data, cfg, **KW)
    for (strat, syms), (_, kw), res in zip(many, specs, out):
        ref = Recorder(**kw)
        want = legacy(ref, data, syms, cfg)
        same(fingerprint(res), fingerprint(want))
        assert strat.rec == ref.rec
    assert len({repr(fingerprint(r)["trades"]) for r in out}) > 5


def test_permuting_the_strategies_changes_nothing():
    data = make_data(4)
    specs = [(["AAA", "BBB"], 5), (SYMBOLS, 6), (["CCC"], 4), (["BBB", "DDD", "EEE"], 7)]
    forward = run_strategies([(Recorder(fast=f), s) for s, f in specs], data, **KW)
    order = [2, 0, 3, 1]
    shuffled = run_strategies([(Recorder(fast=specs[i][1]), specs[i][0]) for i in order], data, **KW)
    for pos, i in enumerate(order):
        same(fingerprint(shuffled[pos]), fingerprint(forward[i]))


def test_share_false_equals_share_true():
    data = make_data(6)
    mk = lambda: [(Recorder(fast=4), ["AAA", "CCC"]), (Recorder(fast=6), SYMBOLS)]
    a = run_strategies(mk(), data, **KW)
    b = run_strategies(mk(), data, share=False, **KW)
    for x, y in zip(a, b):
        same(fingerprint(x), fingerprint(y))


def test_keep_curves_false_keeps_trades_orders_and_metrics():
    data = make_data(7)
    full, lean = (
        run_strategies([Recorder()], data, keep_curves=k, **KW)[0] for k in (True, False)
    )
    assert [fields(t) for t in lean.result.trades()] == [fields(t) for t in full.result.trades()]
    assert repr(fields(lean.metrics)) == repr(fields(full.metrics))
    assert lean.result.equity_curve().tolist() == [full.result.equity_curve()[-1]]
    assert len(lean.result.returns()) == 0 and len(lean.result.drawdown_curve()) == 0


def test_series_is_truncated_read_only_and_never_shows_the_future():
    data = make_data(8, n=50)
    seen = []

    class Peek(Strategy):
        def on_start(self, ctx):
            seen.append(("start", len(ctx.series("AAA")["close"])))

        def on_bar(self, ctx):
            s = ctx.series()
            n = len(s["close"])
            seen.append((ctx.symbol, ctx.idx, n))
            assert s["close"][-1] == ctx.bar.close
            with pytest.raises(ValueError):
                s["close"][0] = 0.0

    run_strategies([Peek()], data, **KW)
    assert seen[0] == ("start", 0)
    assert all(n == idx + 1 for _, idx, n in seen[1:])


def test_memory_report_has_the_categories_and_counts_sharing():
    data = make_data(9, n=100)
    results, report = run_strategies(
        [(Recorder(), SYMBOLS) for _ in range(6)], data, memory_report=True, keep_curves=False, **KW
    )
    assert len(results) == 6
    for key in ("shared_store", "aggregators", "indicators", "index", "engine_per_strategy",
                "portfolio_per_strategy", "curves"):
        assert report[key] > 0, key
    assert report["strategies"] == 6
    # Six identical strategies share one cell per distinct indicator.
    assert report["indicator_cells"] == 5 * 3
    assert report["indicator_registrations"] == 6 * 5 * 3
    assert report["aggregator_count"] == 5
    # The columns are held once however many strategies read them.
    _, two = run_strategies(
        [(Recorder(), SYMBOLS) for _ in range(2)], data, memory_report=True, **KW
    )
    assert report["shared_store"] == two["shared_store"] >= 5 * 100 * 48
    assert report["engine_bytes_per_strategy"] > 0


def test_unrouted_indicator_warns_and_matches():
    class Unrouted(Strategy):
        def on_start(self, ctx):
            self.ind = self.register_indicator(Indicator.sma(3))
            self.vals = []

        def on_bar(self, ctx):
            self.vals.append((ctx.symbol, self.ind.value))

    data = make_data(10, n=30)
    a, b = Unrouted(), Unrouted()
    with pytest.warns(UserWarning, match="without symbol="):
        run_strategies([a], data, **KW)
    with pytest.warns(UserWarning, match="without symbol="):
        legacy(b, data, SYMBOLS)
    assert a.vals == b.vals and a.ind.linked


def test_bad_inputs():
    data = make_data(n=5)
    with pytest.raises(ValueError, match="no data for 'ZZZ'"):
        run_strategies([(Recorder(), ["ZZZ"])], data, **KW)
    s = Recorder()
    with pytest.raises(ValueError, match="twice"):
        run_strategies([s, s], data, **KW)
    with pytest.raises(ValueError, match="at least one symbol"):
        run_strategies([Recorder()], {}, **KW)
    with pytest.raises(ValueError, match="strategies is empty"):
        run_strategies([], data, **KW)
    with pytest.raises(InvalidParameter, match="bar_stamp is required"):
        run_strategies([Recorder()], data)


def test_exports():
    assert raptorbt.run_strategies is run_strategies
    from raptorbt.strategy import run_strategies as again

    assert again is run_strategies


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class Chaos(Strategy):
        """Decisions are a hash of what the strategy observes, so two runs that
        present the same observations take the same actions."""

        UNITS = [(3, "m"), (5, "m"), (2, "tick"), (400, "volume"), (1, "renko"), (1, "h")]

        def __init__(self, seed):
            super().__init__()
            self.seed = seed
            self.rec = []

        def _h(self, *parts):
            return hash((self.seed, *parts)) & 0xFFFFFF

        def on_start(self, ctx):
            r = np.random.default_rng(self.seed)
            self.streams = []
            for _ in range(int(r.integers(0, 3))):
                step, unit = self.UNITS[int(r.integers(len(self.UNITS)))]
                brick = 1.5 if unit == "renko" else 0.0
                self.streams.append(self.subscribe_bars(step, unit, brick_size=brick))
            make = [Indicator.sma, Indicator.ema, Indicator.rsi, Indicator.atr, Indicator.stddev]
            self.inds = []
            for s in ctx.symbols:
                for _ in range(int(r.integers(0, 3))):
                    ind = make[int(r.integers(len(make)))](int(r.integers(2, 9)))
                    stream = None
                    if self.streams and r.random() < 0.5:
                        stream = self.streams[int(r.integers(len(self.streams)))]
                    self.inds.append(self.register_indicator(ind, stream_id=stream, symbol=s))
            if r.random() < 0.5:
                self.clock.set_timer("tick", int(r.integers(2, 9)) * MIN, start_ns=T0)

        def obs(self, ctx):
            return (ctx.symbol, ctx.idx, tuple(i.value for i in self.inds), ctx.equity, ctx.free_capital)

        def on_timer(self, ctx, event):
            self.rec.append(("time", self.obs(ctx), event))

        def on_composite_bar(self, ctx, bar):
            self.rec.append(("comp", self.obs(ctx), bar))
            if self._h(ctx.symbol, bar.timestamp) % 11 == 0 and ctx.is_flat:
                self.enter_short(size_frac=0.2)

        def on_bar(self, ctx):
            h = self._h(ctx.symbol, ctx.idx, ctx.bar.close)
            self.rec.append(("bar", self.obs(ctx), ctx.bar))
            c = ctx.bar.close
            if h % 13 == 0 and ctx.is_flat:
                self.enter(size_frac=0.25)
            elif h % 13 == 1 and not ctx.is_flat:
                self.close_position()
            elif h % 13 == 2:
                self.submit_order(Limit(side="buy", price=c * 0.995, units=1.0))
            elif h % 13 == 3:
                self.cancel_all_orders()
            elif h % 13 == 4 and ctx.is_flat:
                self.enter_long(size_frac=0.1, stop_price=c * 0.97, target_price=c * 1.04)
            if h % 29 == 0 and self.streams:
                self.rec.append(("orders", [o.client_id for o in ctx.open_orders()]))

        def on_order_event(self, ctx, event):
            self.rec.append(("ev", ctx.symbol, event.kind, event.price, event.size, event.reject_reason))

        def on_margin_call(self, ctx, event):
            self.rec.append(("margin", ctx.symbol, event.price, event.size))


@pytest.mark.parametrize("case", range(24))
def test_random_configurations_match_the_legacy_runner(case):
    r = np.random.default_rng(1000 + case)
    data = make_data(case, n=160)
    cfg = BacktestConfig(fees=0.0005, max_drawdown_pct=float(r.choice([8.0, 25.0])),
                         max_positions=int(r.choice([1, 2, 5])))
    if r.random() < 0.5:
        cfg.set_atr_stop(2.0, int(r.integers(5, 15)))
        cfg.set_atr_target(3.0, int(r.integers(5, 15)))
    kw = dict(oms_type=str(r.choice(["netting", "hedging"])),
              account_type=str(r.choice(["cash", "margin"])),
              leverage=float(r.choice([1.0, 4.0])),
              directions={s: int(r.choice([1, -1])) for s in SYMBOLS},
              bar_stamp="open", bar_ns=MIN)
    specs = []
    for i in range(int(r.integers(3, 8))):
        k = int(r.integers(1, 6))
        specs.append((sorted(r.choice(SYMBOLS, size=k, replace=False).tolist()), 50 * case + i))
    shared = [(Chaos(seed), syms) for syms, seed in specs]
    got = run_strategies(shared, data, cfg, **kw)
    traded = 0
    for (strat, syms), (_, seed), res in zip(shared, specs, got):
        ref = Chaos(seed)
        want = run_portfolio_strategy(ref, {k: data[k] for k in data if k in syms}, cfg, **kw)
        same(fingerprint(res), fingerprint(want))
        assert repr(strat.rec) == repr(ref.rec)
        traded += len(want.result.trades())
    assert traded > 0
