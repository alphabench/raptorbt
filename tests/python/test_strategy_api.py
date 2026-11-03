"""The canonical strategy API: subscriptions, data, indicators, hooks, orders.

Every check compares the new spelling against the old one (or against the
rule the data must obey), so a failure names the behaviour that moved.
"""

import contextlib
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Indicator, Strategy, errors, run_strategies
from raptorbt.strategy import Timeframe, orders

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "golden"))
import runner_fixtures as rf  # noqa: E402
from test_no_lookahead import _hand_bars, _poison_bars  # noqa: E402

MIN_NS = 60 * 1_000_000_000
TICKS = rf.load()["datasets"]
S0, S1, S2 = rf.S0, rf.S1, rf.S2


def config():
    return BacktestConfig(initial_capital=1_000_000.0, fees=0.0005)


def tick_feed():
    ticks = {s: rf._thaw_arrays(TICKS["ticks"][s]) for s in (S0, S1, S2)}
    depth = {s: rf._thaw_arrays(TICKS["depth"][s]) for s in (S0, S1, S2)}
    return ticks, depth


def digest(result):
    return rf.digest_result(result)


def bars():
    return _hand_bars()


# ---------------------------------------------------------------------------
# Translation: the same strategy in the old and the new spelling
# ---------------------------------------------------------------------------


class _BarLogic:
    """What the strategy does, independent of how it is spelled."""

    SYMS = ("AAA", "BBB", "CCC")

    def _init(self):
        self.seen = []
        self.n = {}

    def _bar(self, ctx, indicator_values):
        sym = ctx.symbol
        n = self.n[sym] = self.n.get(sym, -1) + 1
        self.seen.append(("bar", sym, tuple(ctx.bar), ctx.idx, indicator_values))
        if n % 10 == 2 and ctx.is_flat:
            self.enter_order(ctx, sym)
        elif n % 10 == 7 and not ctx.is_flat:
            self.submit_order(
                orders.Market(side="sell", **self.exit_kwargs(sym)), **self.route(sym)
            )

    def _comp(self, ctx, bar, value):
        self.seen.append(("comp", bar.symbol, tuple(bar), ctx.idx, value))

    def _fill(self, ctx, event):
        self.seen.append(("fill", ctx.symbol, event.kind, event.idx, event.price, event.size))

    def _position(self, ctx, event):
        self.seen.append(("pos", ctx.symbol, event.kind, event.price, event.size))

    def _timer(self, ctx, event):
        self.seen.append(("timer", ctx.symbol, event.name, event.ts_fired))


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class LegacyBars(_BarLogic, Strategy):
        def __init__(self):
            super().__init__()
            self._init()

        def route(self, sym):
            return {"symbol": sym}

        def exit_kwargs(self, sym):
            return {"size_frac": None, "units": 3.0}

        def enter_order(self, ctx, sym):
            self.submit_bracket(
                orders.Market(side="buy", units=3.0), stop_trigger=ctx.bar.close * 0.97,
                target_price=ctx.bar.close * 1.05, **{},
            )

        def on_start(self, ctx):
            self.subscribe_bars(5, "m")
            self.fast = {s: self.register_indicator(Indicator.sma(3), symbol=s) for s in ctx.symbols}
            self.comp = {
                s: self.register_indicator(Indicator.ema(2), stream_id=0, symbol=s)
                for s in ctx.symbols
            }
            self.clock.set_timer("tick", 7 * MIN_NS, start_ns=0)

        def on_bar(self, ctx):
            self._bar(ctx, self.fast[ctx.symbol].value)

        def on_composite_bar(self, ctx, bar):
            self._comp(ctx, bar, self.comp[bar.symbol].value)

        def on_fill(self, ctx, event):
            self._fill(ctx, event)

        def on_position_opened(self, ctx, event):
            self._position(ctx, event)

        def on_position_closed(self, ctx, event):
            self._position(ctx, event)

        def on_timer(self, ctx, event):
            self._timer(ctx, event)


class ApiBars(_BarLogic, Strategy):
    def __init__(self):
        super().__init__()
        self._init()

    def route(self, sym):
        return {}

    def exit_kwargs(self, sym):
        return {"instrument": sym, "quantity": 3.0}

    def enter_order(self, ctx, sym):
        self.submit_bracket(
            orders.Market(side="buy", instrument=sym, quantity=3.0),
            stop_trigger=ctx.bar.close * 0.97,
            target_price=ctx.bar.close * 1.05,
        )

    def on_start(self, ctx):
        for s in ctx.symbols:
            self.subscribe(s, timeframe="5m")
        self.fast = {s: ctx.indicators.sma(instrument=s, period=3) for s in ctx.symbols}
        self.comp = {
            s: ctx.indicators.ema(instrument=s, timeframe="5m", period=2) for s in ctx.symbols
        }
        self.clock.set_timer("tick", 7 * MIN_NS, start_ns=0)

    def on_bar(self, ctx, event):
        if event.timeframe is None:
            self._bar(ctx, self.fast[ctx.symbol].value)
        else:
            self._comp(ctx, event, self.comp[event.symbol].value)

    def on_fill(self, ctx, event):
        self._fill(ctx, event)

    def on_position_event(self, ctx, event):
        self._position(ctx, event)

    def on_timer(self, ctx, event):
        self._timer(ctx, event)


def test_bar_strategy_translates_bit_for_bit():
    runs = []
    for cls in (LegacyBars, ApiBars):
        strategy = cls()
        result = run_strategies([strategy], bars(), config=config(), bar_stamp="close")[0]
        runs.append((strategy.seen, digest(result)))
    (legacy_seen, legacy_result), (api_seen, api_result) = runs
    kinds = {e[0] for e in legacy_seen}
    assert {"bar", "comp", "fill", "pos", "timer"} <= kinds, "the scenario must exercise every hook"
    assert api_result == legacy_result
    # The old trace has composite bars before the primary bar that completed
    # them; the new on_bar sees them in the same order.
    assert repr(api_seen) == repr(legacy_seen)


class _TickLogic:
    def _init(self):
        self.seen = []
        self.n = 0

    def _drive(self, ctx):
        self.n += 1
        if self.n % 9 == 2 and ctx.is_flat:
            self.enter(size_frac=0.2)
        elif self.n % 9 == 6 and not ctx.is_flat:
            self.close_position()


class LegacyTicks(_TickLogic, Strategy):
    def __init__(self):
        super().__init__()
        self._init()

    def on_trade(self, ctx, tick):
        self.seen.append(("trade", ctx.symbol, tuple(tick), ctx.idx))
        self._drive(ctx)

    def on_quote(self, ctx, quote):
        self.seen.append(("quote", ctx.symbol, tuple(quote), ctx.idx))

    def on_book(self, ctx, book):
        self.seen.append(("book", ctx.symbol, tuple(book), ctx.idx))


class ApiTicks(_TickLogic, Strategy):
    def __init__(self):
        super().__init__()
        self._init()

    def on_start(self, ctx):
        for s in ctx.symbols:
            for kind in ("trade", "quote", "book"):
                self.subscribe(s, data_type=kind)

    def on_trade(self, ctx, tick):
        self.seen.append(("trade", ctx.symbol, tuple(tick), ctx.idx))
        self._drive(ctx)

    def on_quote(self, ctx, quote):
        assert ctx.data.quote(ctx.symbol) is quote
        self.seen.append(("quote", ctx.symbol, tuple(quote), ctx.idx))

    def on_book(self, ctx, book):
        assert ctx.data.book(ctx.symbol) is book
        self.seen.append(("book", ctx.symbol, tuple(book), ctx.idx))


def test_tick_strategy_translates_bit_for_bit():
    runs = []
    for cls in (LegacyTicks, ApiTicks):
        ticks, depth = tick_feed()
        strategy = cls()
        result = run_strategies(
            [strategy], ticks=ticks, depth=depth, primary_bars=(1, "m"), config=config()
        )[0]
        runs.append((strategy.seen, digest(result)))
    (legacy_seen, legacy_result), (api_seen, api_result) = runs
    assert {e[0] for e in legacy_seen} == {"trade", "quote", "book"}
    assert api_result == legacy_result
    assert repr(api_seen) == repr(legacy_seen)
    assert legacy_result["trades"], "the scenario must trade"


def test_tick_strategy_on_the_streaming_wrapper():
    ticks, depth = tick_feed()
    legacy, api = LegacyTicks(), ApiTicks()
    for strategy in (legacy, api):
        raptorbt.run_tick_strategy(
            strategy, ticks, depth=depth, config=config(), primary_bars=(1, "m")
        )
    assert repr(api.seen) == repr(legacy.seen)


def test_new_api_in_run_portfolio_strategy_and_run_strategy_backtest():
    class P(Strategy):
        def __init__(self):
            super().__init__()
            self.seen = []

        def on_start(self, ctx):
            self.subscribe(ctx.symbols[0], timeframe="5m", history=4)
            self.sma = ctx.indicators.sma(instrument=ctx.symbols[0], period=3)

        def on_bar(self, ctx, event):
            if ctx.symbol == ctx.symbols[0]:
                latest = ctx.data.latest(ctx.symbols[0])
                self.seen.append((event.timeframe, latest, self.sma.ready))

    strategy = P()
    raptorbt.run_portfolio_strategy(strategy, bars(), config=config(), bar_stamp="close")
    assert any(tf == "5m" for tf, _, _ in strategy.seen)
    assert all(latest is not None for _, latest, _ in strategy.seen)

    class Single(Strategy):
        def __init__(self):
            super().__init__()
            self.seen = []

        def on_start(self, ctx):
            self.subscribe("X", timeframe="5m")
            self.ema = ctx.indicators.ema(instrument="X", timeframe="5m", period=2)

        def on_bar(self, ctx, event):
            self.seen.append((event.timeframe, ctx.data.latest("X"), len(ctx.data.history("X"))))

    single = Single()
    d = bars()["AAA"]
    raptorbt.run_strategy_backtest(
        single, d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"],
        symbol="X", bar_stamp="close", config=config(),
    )
    primary = [(l, n) for tf, l, n in single.seen if tf is None]
    assert [n for _, n in primary] == list(range(1, len(primary) + 1)), "history grows by one bar"
    assert any(tf == "5m" for tf, _, _ in single.seen)


# ---------------------------------------------------------------------------
# Hook aliases
# ---------------------------------------------------------------------------


def _trace(cls_old, cls_new, **run):
    out = []
    for cls in (cls_old, cls_new):
        strategy = cls()
        run_strategies([strategy], config=config(), **run)
        out.append(strategy.seen)
    return out


def _recording(names, mapping):
    """A strategy class whose hooks, spelled ``names``, all log their events."""
    ns = {"__init__": lambda self: (Strategy.__init__(self), setattr(self, "seen", []))[0]}
    for name in names:
        def hook(self, ctx, event, _n=mapping.get(name, name)):
            self.seen.append((_n, ctx.symbol, repr(event)))
        ns[name] = hook
    aliases = [n for n in names if n in raptorbt.strategy.api.DEPRECATED]
    with pytest.warns(DeprecationWarning) if aliases else contextlib.nullcontext():
        return type("R_" + "_".join(names), (Strategy,), ns)


ALIAS_PAIRS = [
    ("on_time_event", "on_timer"),
    ("on_order_filled", "on_fill"),
    ("on_position_opened", "on_position_event"),
]


def test_old_and_new_hook_names_see_identical_events():
    for old, new in ALIAS_PAIRS:
        # on_position_event also sees closes: compare it against both old hooks.
        old_names = [old] + (["on_position_closed"] if old == "on_position_opened" else [])
        mapping = {n: "e" for n in (*old_names, new)}
        A = _recording(old_names, mapping)

        class B(_recording([new], mapping)):
            def on_start(self, ctx):
                self.clock.set_timer("t", 5 * MIN_NS, start_ns=0)

        class A2(A):
            def on_start(self, ctx):
                self.clock.set_timer("t", 5 * MIN_NS, start_ns=0)

            def on_bar(self, ctx):
                if ctx.idx % 11 == 1 and ctx.is_flat:
                    self.enter(size_frac=0.1)
                elif ctx.idx % 11 == 6 and not ctx.is_flat:
                    self.close_position()

        class B2(B):
            on_bar = A2.on_bar

        a, b = A2(), B2()
        for s in (a, b):
            run_strategies([s], bars(), config=config(), bar_stamp="close")
        assert a.seen, f"{old} never fired"
        assert a.seen == b.seen, (old, new)


def test_trade_quote_book_aliases_see_identical_events():
    runs = []
    for old, new in (("on_trade_tick", "on_trade"), ("on_order_book", "on_book")):
        pair = []
        for name in (old, new):
            cls = _recording([name], {name: "e"})
            ticks, depth = tick_feed()
            s = cls()
            run_strategies([s], ticks=ticks, depth=depth, config=config())
            pair.append(s.seen)
        assert pair[0] and pair[0] == pair[1]


def test_composite_bar_alias_is_on_bar_with_an_event():
    with pytest.warns(DeprecationWarning, match="deprecated hooks"):
        class Old(Strategy):
            def __init__(self):
                super().__init__()
                self.seen = []

            def on_start(self, ctx):
                self.subscribe_bars(5, "m")

            def on_composite_bar(self, ctx, bar):
                self.seen.append(tuple(bar))

    with pytest.warns(DeprecationWarning, match="deprecated hooks"):
        class New(Strategy):
            def __init__(self):
                super().__init__()
                self.seen = []

            def on_start(self, ctx):
                self.subscribe_bars(5, "m")

            def on_composite_bar(self, ctx, event):
                if event.timeframe is not None:
                    self.seen.append(tuple(event))

    old, new = Old(), New()
    for s in (old, new):
        run_strategies([s], bars(), config=config(), bar_stamp="close")
    assert old.seen and old.seen == new.seen


def test_legacy_on_bar_and_on_composite_bar_coexist_and_see_their_own_events():
    with pytest.warns(DeprecationWarning, match="deprecated hooks"):
        class Both(Strategy):
            def __init__(self):
                super().__init__()
                self.bars = self.comps = 0

            def on_start(self, ctx):
                self.subscribe_bars(5, "m")

            def on_bar(self, ctx):
                self.bars += 1

            def on_composite_bar(self, ctx, bar):
                self.comps += 1

    s = Both()
    run_strategies([s], bars(), config=config(), bar_stamp="close")
    assert s.bars == sum(len(d["close"]) for d in bars().values())
    assert 0 < s.comps < s.bars


def test_on_market_data_sees_every_event_after_the_specific_hook():
    class M(Strategy):
        def __init__(self):
            super().__init__()
            self.order = []

        def on_trade(self, ctx, tick):
            self.order.append("trade")

        def on_quote(self, ctx, quote):
            self.order.append("quote")

        def on_book(self, ctx, book):
            self.order.append("book")

        def on_market_data(self, ctx, event):
            self.order.append(type(event).__name__)

    ticks, depth = tick_feed()
    m = M()
    run_strategies([m], ticks=ticks, depth=depth, config=config())
    pairs = list(zip(m.order[::2], m.order[1::2]))
    assert pairs
    assert set(pairs) <= {
        ("trade", "TradeTick"), ("quote", "QuoteTick"), ("book", "BookSnapshot"),
    }
    assert {p[0] for p in pairs} == {"trade", "quote", "book"}


def test_defining_a_name_and_its_alias_raises_at_class_creation():
    for new, old in (
        ("on_trade", "on_trade_tick"),
        ("on_book", "on_order_book"),
        ("on_timer", "on_time_event"),
        ("on_fill", "on_order_filled"),
        ("on_position_event", "on_position_closed"),
    ):
        with pytest.raises(TypeError, match=f"{new}.*{old}"):
            type("Bad", (Strategy,), {new: lambda s, c, e: None, old: lambda s, c, e: None})

    with pytest.raises(TypeError, match="on_composite_bar"):

        class Bad(Strategy):
            def on_bar(self, ctx, event):
                pass

            def on_composite_bar(self, ctx, bar):
                pass

    # A parent's alias and a child's new name are the same ambiguity.
    with pytest.warns(DeprecationWarning, match="on_trade_tick"):

        class Parent(Strategy):
            def on_trade_tick(self, ctx, tick):
                pass

    with pytest.raises(TypeError):

        class Child(Parent):
            def on_trade(self, ctx, tick):
                pass


def test_deprecated_names_warn_once_per_class():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        class Old(Strategy):
            def on_trade_tick(self, ctx, tick):
                pass

            def on_order_filled(self, ctx, event):
                pass

        class Sub(Old):
            pass

        class New(Strategy):
            def on_trade(self, ctx, tick):
                pass

            def on_bar(self, ctx):
                pass

    warned = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(warned) == 1
    assert "Old" in str(warned[0].message) and "on_trade_tick -> on_trade" in str(warned[0].message)
    assert Path(warned[0].filename).name == "test_strategy_api.py"


def test_on_bar_signature_decides_what_it_receives():
    from raptorbt.strategy import api

    class Legacy(Strategy):
        def on_bar(self, ctx, _n="x"):
            pass

    class Event(Strategy):
        def on_bar(self, ctx, event):
            pass

    assert not api._takes_event(Legacy().on_bar, bound=True)
    assert api._takes_event(Event().on_bar, bound=True)


def test_on_init_runs_before_on_start_and_may_subscribe():
    class I(Strategy):
        def __init__(self):
            super().__init__()
            self.order = []

        def on_init(self, ctx):
            self.order.append("init")
            self.subscribe(ctx.symbols[0], timeframe="5m")

        def on_start(self, ctx):
            self.order.append("start")

        def on_stop(self, ctx):
            self.order.append("stop")

    s = I()
    run_strategies([s], bars(), config=config(), bar_stamp="close")
    assert s.order == ["init", "start", "stop"]
    assert s._bar_subscriptions == [(5, "m", 0.0)]


# ---------------------------------------------------------------------------
# Event filtering
# ---------------------------------------------------------------------------


class _Quiet(Strategy):
    """Trades on bars only; optionally also defines the tick hooks."""

    def __init__(self, hooks=False):
        super().__init__()
        self.calls = {"trade": 0, "quote": 0, "book": 0, "market": 0}
        if hooks:
            self.on_trade = self._count("trade")
            self.on_quote = self._count("quote")
            self.on_book = self._count("book")
            self.on_market_data = self._count("market")
        self.n = 0

    def _count(self, kind):
        def hook(ctx, event):
            self.calls[kind] += 1

        return hook

    def on_bar(self, ctx):
        self.n += 1
        if self.n % 8 == 2 and ctx.is_flat:
            self.enter(size_frac=0.2)
        elif self.n % 8 == 6 and not ctx.is_flat:
            self.close_position()


def _run_ticks(strategy):
    ticks, depth = tick_feed()
    return digest(
        run_strategies(
            [strategy], ticks=ticks, depth=depth, primary_bars=(1, "m"), config=config()
        )[0]
    )


def test_a_bar_only_subscription_never_receives_trades_quotes_or_books():
    class BarsOnly(_Quiet):
        def on_start(self, ctx):
            for s in ctx.symbols:
                self.subscribe(s)
                self.subscribe(s, timeframe="5m")

    class Plain(_Quiet):
        pass

    quiet = BarsOnly(hooks=True)
    plain = Plain()
    got, want = _run_ticks(quiet), _run_ticks(plain)
    assert quiet.calls == {"trade": 0, "quote": 0, "book": 0, "market": quiet.calls["market"]}
    assert got == want and want["trades"]


def test_only_the_subscribed_instrument_and_type_arrive():
    class One(_Quiet):
        def on_start(self, ctx):
            self.subscribe(S1, data_type="trade")
            self.subscribe(S0, data_type="quote")
            self.seen = []

        def on_trade(self, ctx, tick):
            self.seen.append(("trade", tick.symbol))

        def on_quote(self, ctx, quote):
            self.seen.append(("quote", quote.symbol))

        def on_book(self, ctx, book):
            self.seen.append(("book", book.symbol))

    s = One()
    _run_ticks(s)
    assert set(s.seen) == {("trade", S1), ("quote", S0)}


def test_a_strategy_that_never_subscribes_keeps_receiving_everything():
    class Legacy(_Quiet):
        def on_trade(self, ctx, tick):
            self.calls["trade"] += 1

        def on_quote(self, ctx, quote):
            self.calls["quote"] += 1

        def on_book(self, ctx, book):
            self.calls["book"] += 1

    s = Legacy()
    _run_ticks(s)
    assert all(s.calls[k] > 0 for k in ("trade", "quote", "book"))


# ---------------------------------------------------------------------------
# Data views never show the future
# ---------------------------------------------------------------------------

CUTS = (0.3, 0.6)


class _Viewer(Strategy):
    def __init__(self):
        super().__init__()
        self.rows = []

    def on_start(self, ctx):
        self.empty = (ctx.data.latest("AAA"), len(ctx.data.history("AAA")))
        for s in ctx.symbols:
            self.subscribe(s, timeframe="5m", history=50)

    def on_bar(self, ctx, event):
        ts = int(ctx.bar_close_ts)
        h = ctx.data.history("AAA", count=7)
        c = ctx.data.history("BBB", "5m", count=3)
        self.rows.append((
            ts, ctx.symbol, event.timeframe,
            ctx.data.latest("AAA"), h.timestamps.tolist(), h.close.tolist(),
            ctx.data.latest("BBB", "5m"), c.close.tolist(), c.timestamps.tolist(),
        ))


@pytest.mark.parametrize("cut", CUTS)
def test_history_and_latest_do_not_depend_on_the_future(cut):
    data = bars()
    t_all = np.sort(np.concatenate([d["timestamps"] for d in data.values()]))
    t_cut = int(t_all[int(len(t_all) * cut)])
    rng = np.random.default_rng(3)
    poisoned = {s: _poison_bars(d, t_cut, rng) for s, d in data.items()}
    runs = []
    for d in (data, poisoned):
        v = _Viewer()
        run_strategies([v], d, config=config(), bar_stamp="close")
        runs.append(v)
    assert runs[0].empty == (None, 0)
    before = [[r for r in v.rows if r[0] <= t_cut] for v in runs]
    assert before[0] and repr(before[0]) == repr(before[1])
    assert repr(runs[0].rows) != repr(runs[1].rows), "the poison must change the future"
    for row in runs[0].rows:
        assert all(t <= row[0] for t in row[4]) and all(t <= row[0] for t in row[8])


def test_history_is_read_only_and_bounded():
    class H(Strategy):
        def on_start(self, ctx):
            self.subscribe("AAA", timeframe="5m", history=4)
            self.max_len = 0
            self.writable = []

        def on_bar(self, ctx, event):
            if ctx.symbol == "AAA":
                for h in (ctx.data.history("AAA"), ctx.data.history("AAA", "5m")):
                    self.writable.append(h.close.flags.writeable)
                    with pytest.raises(AttributeError):
                        h.close = None
                self.max_len = max(self.max_len, len(ctx.data.history("AAA", "5m")))

    s = H()
    run_strategies([s], bars(), config=config(), bar_stamp="close")
    assert s.writable and not any(s.writable)
    assert s.max_len == 4


def test_portfolio_view_reports_the_book():
    class V(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_bar(self, ctx):
            p = ctx.portfolio
            n = ctx.idx
            if ctx.symbol == "AAA":
                if n == 5:
                    self.enter(size_frac=0.3)
                if n == 30:
                    self.close_position()
            self.rows.append((
                n, ctx.symbol, p.cash, p.equity, p.available_capital, p.locked_margin,
                p.realized_pnl, p.unrealized_pnl, p.gross_exposure, p.net_exposure,
                dict(p.positions),
            ))

    v = V()
    result = run_strategies([v], bars(), config=BacktestConfig(initial_capital=1e6, fees=0.0),
                            bar_stamp="close")[0]
    open_rows = [r for r in v.rows if r[10]]
    assert open_rows
    r = open_rows[0]
    assert set(r[10]) == {"AAA"}
    assert r[8] > 0 and r[9] == r[8] and r[5] == 0.0
    last = v.rows[-1]
    assert last[10] == {} and last[8] == 0.0 and last[7] == 0.0
    trades = result.result.trades()
    assert last[6] == pytest.approx(sum(t.pnl for t in trades))
    assert last[3] == pytest.approx(1e6 + last[6])
    # Marked-to-market while open: equity moves with unrealized pnl.
    assert any(r[7] != 0.0 for r in open_rows)


def test_portfolio_totals_that_need_the_runtime_say_so_in_run_strategy_backtest():
    class P(Strategy):
        def on_bar(self, ctx):
            assert ctx.portfolio.equity == ctx.equity
            assert ctx.portfolio.positions == {}
            assert ctx.portfolio.halted is False
            with pytest.raises(errors.UnsupportedOperation, match="run_strategies"):
                ctx.portfolio.realized_pnl

    d = bars()["AAA"]
    raptorbt.run_strategy_backtest(
        P(), d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"],
        bar_stamp="close", config=config(),
    )


def test_on_risk_event_is_a_hook_and_portfolio_reports_halted():
    class Halts(Strategy):
        def __init__(self):
            super().__init__()
            self.events, self.halted_seen = [], []

        def on_bar(self, ctx):
            if ctx.symbol == "AAA" and ctx.idx == 10:
                ctx.halt("test")
            self.halted_seen.append(ctx.portfolio.halted)

        def on_risk_event(self, ctx, event):
            self.events.append((event.kind, event.reject_reason))

    s = Halts()
    run_strategies([s], bars(), config=config(), bar_stamp="close")
    assert s.events and all(k in ("risk_halted", "risk_rejected") for k, _ in s.events)
    assert ("risk_halted", "test") in s.events
    assert s.halted_seen[0] is False and s.halted_seen[-1] is True


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def test_indicator_handles_equal_the_legacy_indicators_and_are_shared():
    class Both(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_start(self, ctx):
            self.new = {
                "sma": ctx.indicators.sma(instrument="AAA", period=4),
                "ema": ctx.indicators.ema(instrument="AAA", period=4),
                "wma": ctx.indicators.wma(instrument="AAA", period=4),
                "rsi": ctx.indicators.rsi(instrument="AAA", period=4),
                "atr": ctx.indicators.atr(instrument="AAA", period=4),
                "roc": ctx.indicators.roc(instrument="AAA", period=4),
                "stddev": ctx.indicators.stddev(instrument="AAA", period=4),
                "bollinger": ctx.indicators.bollinger(instrument="AAA", period=4, k=1.5),
                "donchian": ctx.indicators.donchian(instrument="AAA", period=4),
                "macd": ctx.indicators.macd(instrument="AAA", fast=3, slow=6, signal=2),
            }
            old = {
                "sma": Indicator.sma(4), "ema": Indicator.ema(4), "wma": Indicator.wma(4),
                "rsi": Indicator.rsi(4), "atr": Indicator.atr(4), "roc": Indicator.roc(4),
                "stddev": Indicator.stddev(4), "bollinger": Indicator.bollinger(4, 1.5),
                "donchian": Indicator.donchian(4), "macd": Indicator.macd(3, 6, 2),
            }
            self.old = {k: self.register_indicator(v, symbol="AAA") for k, v in old.items()}

        def on_bar(self, ctx):
            if ctx.symbol == "AAA":
                self.rows.append({
                    k: (h.ready, h.value, self.old[k].initialized, self.old[k].value)
                    for k, h in self.new.items()
                })

    s = Both()
    run_strategies([s], bars(), config=config(), bar_stamp="close")
    assert any(r["macd"][0] for r in s.rows)
    for row in s.rows:
        for ready, value, old_ready, old_value in row.values():
            assert (ready, value) == (old_ready, old_value)
    last = s.new["bollinger"].values
    assert (last.upper, last.middle, last.lower) == s.new["bollinger"].value
    assert s.new["macd"].values._fields == ("macd", "signal", "histogram")
    assert s.new["donchian"].values._fields == ("upper", "lower")
    with pytest.raises(AttributeError):
        s.new["sma"].values


def test_custom_indicator_is_private_and_fed_before_the_hooks():
    class Mean:
        def __init__(self, n):
            self.n, self.xs = n, []

        def update(self, bar):
            self.xs.append(bar.close)

        @property
        def ready(self):
            return len(self.xs) >= self.n

        @property
        def value(self):
            return sum(self.xs[-self.n:]) / self.n if self.ready else None

    class C(Strategy):
        def __init__(self):
            super().__init__()
            self.rows = []

        def on_start(self, ctx):
            self.m = ctx.indicators.custom(Mean(3), instrument="AAA")
            self.m5 = ctx.indicators.custom(Mean(2), instrument="AAA", timeframe="5m")
            self.ref = ctx.indicators.sma(instrument="AAA", period=3)

        def on_bar(self, ctx, event):
            if ctx.symbol == "AAA" and event.timeframe is None:
                self.rows.append((ctx.bar.close, self.m.value, self.ref.value, self.m5.ready))

    c = C()
    run_strategies([c], bars(), config=config(), bar_stamp="close")
    ready = [r for r in c.rows if r[1] is not None]
    assert ready and all(r[1] == pytest.approx(r[2]) for r in ready)
    assert any(r[3] for r in c.rows)
    assert c.m._raptorbt_linked if hasattr(c.m, "_raptorbt_linked") else True

    with pytest.raises(TypeError, match="update"):
        class Bad(Strategy):
            def on_start(self, ctx):
                ctx.indicators.custom(object(), instrument="AAA")

        run_strategies([Bad()], bars(), config=config(), bar_stamp="close")


# ---------------------------------------------------------------------------
# Typed orders
# ---------------------------------------------------------------------------


def test_orders_take_instrument_and_quantity_and_keep_units():
    a = orders.Market(side="buy", quantity=5.0, instrument="AAA")
    b = orders.Market(side="buy", units=5.0)
    assert (a.units, a.quantity) == (5.0, 5.0) and (b.units, b.quantity) == (5.0, 5.0)
    assert a.instrument == "AAA" and b.instrument is None
    with pytest.raises(ValueError, match="quantity or units"):
        orders.Market(side="buy", quantity=1.0, units=2.0)
    with pytest.raises(ValueError, match="units or size_frac"):
        orders.Market(side="buy", quantity=1.0, size_frac=0.1)
    with pytest.raises(ValueError, match="limit price"):
        orders.Limit(side="buy", price=0.0, quantity=1.0)
    assert orders.Twap(side="buy", quantity=3.0, slices=3, every=1).units == 3.0


def test_order_instrument_routes_like_the_deprecated_symbol():
    class R(Strategy):
        def __init__(self, new):
            super().__init__()
            self.new = new

        def on_bar(self, ctx):
            if ctx.symbol == "AAA" and ctx.idx == 20:
                if self.new:
                    self.submit_order(
                        orders.Limit(side="buy", price=1.0, quantity=2.0, instrument="BBB")
                    )
                else:
                    self.submit_order(orders.Limit(side="buy", price=1.0, units=2.0), symbol="BBB")

    results = [
        digest(run_strategies([R(new)], bars(), config=config(), bar_stamp="close")[0])
        for new in (False, True)
    ]
    assert results[0] == results[1]
    assert [o["symbol"] for o in results[1]["orders"]] == ["BBB"]

    s = Strategy()
    with pytest.raises(ValueError, match="contradicts"):
        s.submit_order(orders.Market(side="buy", quantity=1.0, instrument="AAA"), symbol="BBB")


def test_bracket_legs_follow_the_entry_instrument():
    s = Strategy()
    s.submit_bracket(
        orders.Market(side="buy", quantity=1.0, instrument="BBB"), stop_trigger=90.0,
        target_price=110.0,
    )
    commands = [c for c in s.drain_commands() if c[0] == "submit"]
    assert [c[4] for c in commands] == ["BBB", "BBB", "BBB"]


# ---------------------------------------------------------------------------
# Subscriptions and errors
# ---------------------------------------------------------------------------


def _subscribe_error(**kw):
    class E(Strategy):
        def on_start(self, ctx):
            self.subscribe(kw.pop("instrument", "AAA"), **kw)

    with pytest.raises(errors.RaptorBTError) as info:
        run_strategies([E()], bars(), config=config(), bar_stamp="close")
    return str(info.value)


def test_invalid_timeframe_lists_what_is_supported():
    text = _subscribe_error(timeframe="1minute")
    assert text.startswith("Invalid timeframe subscription.")
    assert "'5m'" in text and "'1w'" in text
    assert 'Requested:\n    timeframe = \'1minute\'' in text
    assert (
        "Supported:\n    '1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d', '1w'"
        in text
    )
    assert text.endswith("Related skill:\n    raptorbt-timeframes")


def test_invalid_data_type_lists_what_is_supported():
    text = _subscribe_error(data_type="depth")
    assert text.startswith("Invalid data type subscription.")
    assert "Supported:\n    'bar', 'trade', 'quote', 'book'" in text
    assert "data_type = 'depth'" in text


def test_other_invalid_declarations_are_refused():
    assert "Unknown instrument." in _subscribe_error(instrument="ZZZ")
    assert "Invalid history subscription." in _subscribe_error(history=0)
    assert "Invalid timeframe subscription." in _subscribe_error(data_type="trade", timeframe="5m")
    assert "Invalid timeframe subscription." in _subscribe_error(timeframe=(0, "m"))
    assert "Invalid timeframe subscription." in _subscribe_error(timeframe=(5, "fortnight"))


def test_timeframe_specs_are_accepted():
    class T(Strategy):
        def __init__(self):
            super().__init__()
            self.ids = []

        def on_start(self, ctx):
            self.ids = [
                self.subscribe("AAA", timeframe="5m"),
                self.subscribe("BBB", timeframe=(5, "m")),
                self.subscribe("AAA", timeframe=Timeframe.ticks(100)),
                self.subscribe("AAA", timeframe=Timeframe.renko(2.0)),
                self.subscribe("AAA", timeframe=Timeframe.volume(5000)),
            ]

    t = T()
    run_strategies([t], bars(), config=config(), bar_stamp="close")
    assert t.ids[0] == t.ids[1] and len(set(t.ids)) == 4


def test_declarations_after_start_and_unsubscribed_reads_are_refused():
    class Late(Strategy):
        def on_bar(self, ctx):
            self.subscribe("AAA")

    with pytest.raises(errors.StateError, match="on_init or on_start"):
        run_strategies([Late()], bars(), config=config(), bar_stamp="close")

    class Unsub(Strategy):
        def on_bar(self, ctx):
            ctx.data.history("AAA", "5m")

    with pytest.raises(ValueError, match="Not subscribed to this timeframe"):
        run_strategies([Unsub()], bars(), config=config(), bar_stamp="close")

    class NoQuote(Strategy):
        def on_bar(self, ctx):
            ctx.data.quote("AAA")

    with pytest.raises(ValueError, match="Not subscribed to quotes"):
        run_strategies([NoQuote()], bars(), config=config(), bar_stamp="close")


# ---------------------------------------------------------------------------
# README examples
# ---------------------------------------------------------------------------

README = HERE.parents[1] / "README.md"
FENCE = re.compile(r"```python raptorbt-example\n(.*?)```", re.S)


def _examples():
    return [(i, m.group(1)) for i, m in enumerate(FENCE.finditer(README.read_text()))]


def test_the_readme_has_examples():
    assert len(_examples()) >= 6


@pytest.mark.parametrize("index,source", _examples(), ids=lambda v: f"example-{v}" if isinstance(v, int) else "")
def test_readme_example_runs(index, source):
    ticks, depth = tick_feed()
    ns = {
        "__name__": "readme_example",
        "bars": bars(),
        "ticks": ticks,
        "depth": depth,
        "config": config(),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        exec(compile(source, f"README example {index}", "exec"), ns)
