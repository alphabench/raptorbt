"""No-lookahead: a run's past must not depend on its future.

For each strategy runner, run a deterministic strategy twice: once on the data
as given, once with every input value strictly after a cut time ``T`` replaced
by large random garbage (timestamps and ordering kept valid). Everything the
strategy saw or did at or before ``T`` -- prices, indicator values, composite
bars, positions, the decisions it made, the order and fill events it was told
about, the trades that completed -- must be identical in both runs. The
second run must also differ after ``T``, so the check cannot pass vacuously.

A second, separate check covers a leak that poisoning cannot see: under
``next_bar_open`` a decision must never fill inside the event that made it.
"""

import sys

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Indicator, Strategy
from raptorbt.strategy import orders

import synthetic  # noqa: E402

MIN_NS = 60 * 1_000_000_000
CUTS = (0.3, 0.55, 0.8)
TIMINGS = ("same_bar_close", "next_bar_open")

# Entries logged by hooks that fire before the data hook of the same tick
# event; in tick runs they take that event's time.
_PRE_DATA = {"comp", "bar"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
_OHLCV = ("timestamps", "open", "high", "low", "close", "volume")


def _hand_bars():
    """Three symbols, 1m bars, staggered starts so the merged schedule interleaves."""
    out = {}
    for k, sym in enumerate(("AAA", "BBB", "CCC")):
        n = 90
        t = np.arange(n)
        close = 100.0 + 5.0 * np.sin(t / (6.0 + k)) + 0.05 * t * (k + 1)
        out[sym] = {
            "timestamps": (t * MIN_NS + k * (MIN_NS // 4)).astype(np.int64),
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    return out


_SYNTH_CACHE: dict = {}


def _synth(**kwargs):
    key = tuple(sorted(kwargs.items()))
    if key not in _SYNTH_CACHE:
        _SYNTH_CACHE[key] = synthetic.generate(
            n_instruments=6, days=2, seed=11, timeframes=("1m",), **kwargs
        )
    return _SYNTH_CACHE[key]


def _synth_bars():
    ds = _synth()
    return {
        sym: {k: np.ascontiguousarray(bars[k]) for k in _OHLCV}
        for sym, bars in ds.bars["1m"].items()
    }


def _hand_ticks():
    rng = np.random.default_rng(5)
    out = {}
    for k, sym in enumerate(("AAA", "BBB")):
        n = 900
        ts = (np.arange(n) * 7_000_000_000 + k * 1_000_000_000).astype(np.int64)
        px = 100.0 + np.cumsum(rng.normal(0.0, 0.2, n))
        out[sym] = {
            "timestamps": ts,
            "ltp": px,
            "bid": px - 0.05,
            "ask": px + 0.05,
            "ltq": np.full(n, 10.0),
        }
    return out


def _synth_ticks():
    ds = _synth(avg_ticks_per_sec=0.05)
    return {sym: dict(t) for sym, t in ds.ticks.items()}


BAR_DATA = {"hand": _hand_bars, "synthetic": _synth_bars}
TICK_DATA = {"hand": _hand_ticks, "synthetic": _synth_ticks}


def _poison_bars(d, cut, rng):
    out = {k: np.array(v, copy=True) for k, v in d.items()}
    m = out["timestamps"] > cut
    k = int(m.sum())
    o = rng.uniform(1e4, 1e6, k)
    c = rng.uniform(1e4, 1e6, k)
    out["open"][m] = o
    out["close"][m] = c
    out["high"][m] = np.maximum(o, c) * (1 + rng.uniform(0, 0.5, k))
    out["low"][m] = np.minimum(o, c) * (1 - rng.uniform(0, 0.5, k))
    out["volume"][m] = rng.uniform(1e6, 1e9, k)
    return out


def _poison_ticks(d, cut, rng):
    out = {k: np.array(v, copy=True) for k, v in d.items()}
    m = out["timestamps"] > cut
    k = int(m.sum())
    for key in out:
        if key == "timestamps":
            continue
        out[key][m] = rng.uniform(1.0, 1e6, k)
    if "bid" in out and "ask" in out:
        out["ask"][m] = out["bid"][m] * (1 + rng.uniform(0.001, 0.5, k))
    return out


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------
with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class _Recorder(Strategy):
        """Logs everything it sees and does; subclasses decide."""

        ticks = False

        def __init__(self, mode="intent", nbo=False):
            super().__init__()
            self.mode = mode
            self.nbo = nbo
            self.trace: list[tuple] = []
            self.violations: list[str] = []
            # (kind, symbol) -> event time of an enter/exit decision not yet filled.
            self._decided: dict = {}

        @staticmethod
        def _key(ctx):
            return getattr(ctx, "symbol", None)

        def _rec(self, ctx, kind, *payload, ets=None):
            if ets is None and not self.ticks:
                # A fill at a bar's open is heard before the strategy sees that bar: its event
                # is the next bar's, however far ahead of the last bar seen the engine's instant is.
                ahead = 0
                ns, seen = getattr(ctx, "event_ns", None), getattr(ctx, "bar_close_ts", None)
                if ns is not None and seen is not None:
                    ahead = int(ns) - int(seen)
                ets = int(ctx.timestamp) + ahead
            self.trace.append((kind, ets, payload))

        def _decide(self, ctx, kind, *detail):
            """Record a decision; False when an earlier one is still unfilled."""
            key = (kind, self._key(ctx))
            if kind in ("enter", "exit"):
                if key in self._decided:
                    return False
                self._decided[key] = None if self.ticks else self._instant(ctx)
            self._rec(ctx, "decide", kind, self._key(ctx), *detail)
            return True

        def on_fill(self, ctx, e):
            self._rec(ctx, "fill", self._key(ctx), e.kind, e.idx, e.price, e.size)

        def on_order_rejected(self, ctx, e):
            self._rec(ctx, "reject", self._key(ctx), e.kind, e.idx, e.reject_reason)
            self._decided.pop(("enter", self._key(ctx)), None)
            self._decided.pop(("exit", self._key(ctx)), None)

        @staticmethod
        def _instant(ctx):
            # The instant of the event being delivered. A fill at a bar's open is heard before
            # the strategy sees that bar, so ``ctx.timestamp`` (the last bar it saw) is the
            # previous bar there; ``event_ns`` is the instant of the fill itself.
            ns = getattr(ctx, "event_ns", None)
            return int(ctx.timestamp) if ns is None else int(ns)

        def _check_same_event(self, ctx, decision, what, reason=""):
            decided_at = self._decided.pop((decision, self._key(ctx)), None)
            # a protective stop or target fires inside the bar that decides nothing: not a decision
            if any(w in reason.lower() for w in ("stop", "profit", "target")):
                return
            if self.nbo and not self.ticks and decided_at == self._instant(ctx):
                self.violations.append(
                    f"{what} of {self._key(ctx)} filled in the event that decided it "
                    f"(ts {int(ctx.timestamp)})"
                )

        def on_position_opened(self, ctx, e):
            self._rec(ctx, "open", self._key(ctx), e.price, e.size)
            self._check_same_event(ctx, "enter", "entry")

        def on_position_closed(self, ctx, e):
            self._rec(ctx, "close", self._key(ctx), e.price, e.size)
            self._check_same_event(
                ctx, "exit", "exit", str(getattr(getattr(e, "trade", None), "exit_reason", ""))
            )

        # -- actions shared by bar and tick strategies -----------------------------
        def _enter(self, ctx, size_hint):
            if not self._decide(ctx, "enter", self.mode):
                return
            if self.mode == "orders":
                self.submit_order(orders.Market(side="buy", units=10.0))
            else:
                self.enter(size_frac=size_hint)

        def _exit(self, ctx, pos):
            if not self._decide(ctx, "exit", self.mode, pos.size):
                return
            if self.mode == "orders":
                self.submit_order(orders.Market(side="sell", units=pos.size, reduce_only=True))
            elif self.mode == "targeted":
                self.close_position(symbol=self._key(ctx) or "ASSET")
            else:
                self.close_position()


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class _BarProbe(_Recorder):
        """Dual SMA on the primary stream, gated by a 15m composite SMA."""

        def on_start(self, ctx):
            syms = list(getattr(ctx, "symbols", [None]))
            # A single-instrument run keys everything by ``None``.
            self.fast = self.register_indicators(lambda: Indicator.sma(5), syms)
            self.slow = self.register_indicators(lambda: Indicator.sma(12), syms)
            self.sid15 = self.subscribe_bars(15, "m")
            self.sid1h = self.subscribe_bars(1, "h")
            self.comp = self.register_indicators(
                lambda: Indicator.sma(3), syms, stream_id=self.sid15
            )
            self.last15: dict = {}

        @staticmethod
        def _history(ctx, n=5):
            if hasattr(ctx, "history"):
                return ctx.history(n).tolist()
            close = ctx.series()["close"]
            return close[max(0, ctx.idx + 1 - n) : ctx.idx + 1].tolist()

        def on_composite_bar(self, ctx, bar):
            sym = bar.symbol
            if bar.stream_id == self.sid15:
                self.last15[sym] = bar.close
            self._rec(ctx, "comp", sym, bar[:-1])

        def on_bar(self, ctx):
            sym = self._key(ctx)
            fast, slow, comp = self.fast[sym].value, self.slow[sym].value, self.comp[sym].value
            pos = ctx.position
            self._rec(
                ctx, "bar", sym, tuple(ctx.bar), fast, slow, comp, self.last15.get(sym),
                None if pos is None else (pos.size, pos.entry_price),
                ctx.equity, ctx.cash, self._history(ctx),
            )
            if self.mode == "orders" and ctx.idx % 40 == 7 and pos is None:
                px = round(ctx.bar.close * 0.997, 2)
                self._decide(ctx, "limit", px)
                self.submit_order(orders.Limit(side="buy", units=3.0, price=px))
            if fast is None or slow is None:
                return
            gate = comp is None or self.last15.get(sym, 0.0) >= comp
            if pos is None and fast > slow and gate:
                self._enter(ctx, 0.2)
            elif pos is not None and fast < slow:
                self._exit(ctx, pos)


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class _TickProbe(_Recorder):
        """Rolling-mean cross on prints, with quote, primary-bar and composite inputs."""

        ticks = True

        def on_start(self, ctx):
            syms = list(ctx.symbols)
            self.prices = {s: [] for s in syms}
            self.bar_fast = self.register_indicators(lambda: Indicator.sma(3), syms)
            self.sid = self.subscribe_bars(15, "m")
            self.comp = self.register_indicators(
                lambda: Indicator.sma(2), syms, stream_id=self.sid
            )
            self.last15: dict = {}

        def on_composite_bar(self, ctx, bar):
            self.last15[bar.symbol] = bar.close
            self._rec(ctx, "comp", bar.symbol, bar[:-1])

        def on_bar(self, ctx):
            sym = self._key(ctx)
            self._rec(ctx, "bar", sym, tuple(ctx.bar), self.bar_fast[sym].value)

        def on_quote(self, ctx, quote):
            self._rec(ctx, "quote", self._key(ctx), tuple(quote), ets=quote.timestamp)

        def on_trade(self, ctx, tick):
            sym = self._key(ctx)
            px = self.prices[sym]
            px.append(tick.price)
            pos = ctx.position
            short = sum(px[-10:]) / len(px[-10:])
            long_ = sum(px[-30:]) / len(px[-30:])
            self._rec(
                ctx, "tick", sym, tuple(tick), short, long_, self.bar_fast[sym].value,
                self.comp[sym].value, self.last15.get(sym), ctx.best_bid, ctx.best_ask,
                None if pos is None else (pos.size, pos.entry_price), ctx.equity,
                ets=tick.timestamp,
            )
            if len(px) < 30:
                return
            if pos is None and short > long_:
                self._enter(ctx, 0.2)
            elif pos is not None and short < long_:
                self._exit(ctx, pos)


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class _OpenHookProbe(_BarProbe):
        """Reads every symbol's indicators, own stream and composite, in the hooks that hear a
        fill, and queues an entry from the hook that hears an exit (an ATR stop is configured:
        the stop it gets is logged when the entry fills). Under ``next_bar_open`` those hooks run
        at a bar's open, before the strategy has seen the bar."""

        def _open_read(self, ctx, what):
            # every symbol's indicators, and what the hook sees of bars: all must be pre-open
            self._rec(
                ctx, what, self._key(ctx),
                tuple((s, self.fast[s].value, self.slow[s].value, self.comp[s].value)
                      for s in sorted(self.fast)),
                tuple(ctx.bar), self._history(ctx),
            )

        def on_fill(self, ctx, e):
            super().on_fill(ctx, e)
            self._open_read(ctx, "openread")

        def on_position_opened(self, ctx, e):
            super().on_position_opened(ctx, e)
            pos = ctx.position
            self._rec(ctx, "stop", self._key(ctx), None if pos is None else pos.stop_price)

        def on_position_closed(self, ctx, e):
            super().on_position_closed(ctx, e)
            key = ("enter", self._key(ctx))
            if self.mode == "intent" and ctx.idx % 3 == 0 and key not in self._decided:
                # An answer to a fill at the open fills at that same open (the engine's rule), so
                # it is no same-event violation; its ATR stop is what the poisoning checks.
                self._decided[key] = None
                self._rec(ctx, "decide", "enter", self._key(ctx), "hook")
                self.enter(size_frac=0.1)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _cfg(timing):
    return BacktestConfig(
        initial_capital=10_000_000.0, fees=0.0002, slippage=0.0, fill_timing=timing
    )


def _stamp_kwargs(stamp):
    """Keyword arguments declaring how the bars are stamped."""
    return {"bar_stamp": "open", "bar_ns": MIN_NS} if stamp == "open" else {"bar_stamp": "close"}


def _run(runner, strategy, data, timing, stamp="close", portfolio_runner=None, tick_runner=None):
    cfg = _cfg(timing)
    if runner == "tick":
        return tick_runner(strategy, data, config=cfg, primary_bars=(5, "m")).result.trades()
    if runner == "single":
        sym = next(iter(data))
        result = raptorbt.run_strategy_backtest(
            strategy, **data[sym], config=cfg, **_stamp_kwargs(stamp)
        )
    else:
        result = portfolio_runner(strategy, data, config=cfg, **_stamp_kwargs(stamp)).result
    return result.trades()


def _stamped(log):
    """Give every entry its event time (tick runs log pre-data hooks first)."""
    out, pending, cur = [], [], None
    for kind, ets, payload in log:
        if ets is not None:
            out.extend((k, ets, p) for k, _, p in pending)
            pending, cur = [], ets
            out.append((kind, ets, payload))
        elif kind in _PRE_DATA:
            pending.append((kind, None, payload))
        else:
            out.append((kind, cur, payload))
    assert not pending
    return out


def _visible(strategy, trades, cut, trade_cut=None):
    # Trades carry engine (bar close) time; the strategy's log carries the
    # caller's own stamps. They differ by one bar width for open stamps.
    trade_cut = cut if trade_cut is None else trade_cut
    log = _stamped(strategy.trace)
    seen = [e for e in log if e[1] <= cut]
    later = [e for e in log if e[1] > cut]
    # Trades list in close order, which is itself a function of the future.
    done = sorted(
        (t.symbol, t.entry_time, t.entry_price, t.size, t.direction,
         (t.exit_time, t.exit_price, t.pnl, t.exit_reason) if t.exit_time <= trade_cut else None)
        for t in trades
        if t.entry_time <= trade_cut
    )
    return seen, later, done


def _cut_time(data, frac):
    ts = np.unique(np.concatenate([d["timestamps"] for d in data.values()]))
    return int(ts[int(len(ts) * frac)])


def _check(
    runner, probe_cls, data, timing, mode, frac, poison, stamp="close", portfolio_runner=None,
    tick_runner=None,
):
    cut = _cut_time(data, frac)
    rng = np.random.default_rng(1234)
    garbage = {s: poison(d, cut, rng) for s, d in data.items()}

    clean = probe_cls(mode=mode, nbo=timing == "next_bar_open")
    dirty = probe_cls(mode=mode, nbo=timing == "next_bar_open")
    clean_trades = _run(runner, clean, data, timing, stamp, portfolio_runner, tick_runner)
    dirty_trades = _run(runner, dirty, garbage, timing, stamp, portfolio_runner, tick_runner)

    assert not clean.violations, clean.violations

    trade_cut = cut + (MIN_NS if stamp == "open" else 0)
    seen_c, later_c, done_c = _visible(clean, clean_trades, cut, trade_cut)
    seen_d, later_d, done_d = _visible(dirty, dirty_trades, cut, trade_cut)

    kinds = {e[0] for e in seen_c}
    assert {"comp", "decide", "fill"} <= kinds, f"vacuous prefix: {sorted(kinds)}"
    assert done_c, "no trade entered before the cut"
    assert repr(later_c) != repr(later_d), "garbage did not change the future"

    assert repr(seen_c) == repr(seen_d)
    assert repr(done_c) == repr(done_d)


def _bar_params():
    return [(mode, timing) for mode in ("intent", "orders", "targeted") for timing in TIMINGS]


@pytest.mark.parametrize("frac", CUTS)
@pytest.mark.parametrize("mode,timing", _bar_params())
@pytest.mark.parametrize("dataset", sorted(BAR_DATA))
@pytest.mark.parametrize("runner", ["single", "portfolio"])
@pytest.mark.parametrize("stamp", ["open", "close"])
def test_bar_runners_ignore_the_future_with_declared_stamps(
    stamp, runner, dataset, mode, timing, frac, portfolio_runner
):
    _check(
        runner, _BarProbe, BAR_DATA[dataset](), timing, mode, frac, _poison_bars, stamp,
        portfolio_runner,
    )


@pytest.mark.parametrize("frac", CUTS)
@pytest.mark.parametrize("mode", ["intent", "orders"])
@pytest.mark.parametrize("dataset", sorted(TICK_DATA))
def test_tick_runner_ignores_the_future(dataset, mode, frac, tick_runner):
    _check(
        "tick", _TickProbe, TICK_DATA[dataset](), "same_bar_close", mode, frac, _poison_ticks,
        tick_runner=tick_runner,
    )


class _Peeking(_BarProbe):
    """Reads the next bar's close from the full arrays the context exposes."""

    def on_bar(self, ctx):
        close = ctx.series()["close"]
        nxt = close[ctx.idx + 1] if ctx.idx + 1 < len(close) else None
        self._rec(ctx, "peek", self._key(ctx), nxt)
        super().on_bar(ctx)


def test_the_harness_catches_a_strategy_that_peeks(portfolio_runner):
    # Both runners hand out only what is visible at the current event, so
    # there is nothing to peek at: every peek reads no future.
    probe = _Peeking(mode="intent", nbo=False)
    _run("portfolio", probe, _hand_bars(), "same_bar_close", portfolio_runner=portfolio_runner)
    peeks = [p for kind, _, p in probe.trace if kind == "peek"]
    assert peeks, "the probe never ran"
    assert all(p[1] is None for p in peeks), "series() exposed a future bar"


_PLAIN_CFG = _cfg


def _atr_cfg(timing):
    cfg = _PLAIN_CFG(timing)
    cfg.set_atr_stop(2.0, 5)
    return cfg


@pytest.mark.parametrize("frac", CUTS)
@pytest.mark.parametrize("mode", ["intent", "orders"])
@pytest.mark.parametrize("dataset", sorted(BAR_DATA))
@pytest.mark.parametrize("stamp", ["open", "close"])
def test_a_fill_hook_at_the_open_sees_no_indicator_or_atr_of_the_bar_opening(
    stamp, dataset, mode, frac, portfolio_runner, monkeypatch
):
    """F-16: under next_bar_open the hook that hears a fill at a bar's open reads every symbol's
    indicators (a shared one fed by another symbol's bar, a composite one) as they stood before
    the bar, and an entry it queues takes its ATR stop from the bar before."""
    monkeypatch.setattr(sys.modules[__name__], "_cfg", _atr_cfg)
    _check(
        "portfolio", _OpenHookProbe, BAR_DATA[dataset](), "next_bar_open", mode, frac,
        _poison_bars, stamp, portfolio_runner,
    )


# ---------------------------------------------------------------------------
# expressions on the shared bank (ctx.indicators.expr)
# ---------------------------------------------------------------------------
def _add_expressions(probe, ctx):
    """A numeric, a boolean and a composite-bar expression on every symbol; all read volume or
    prices the poisoning replaces after the cut."""
    syms = list(ctx.symbols)
    probe.num = {
        s: ctx.indicators.expr("ema(close, 4) - sma(close, 9) + sma(volume, 3) * 0.001", instrument=s)
        for s in syms
    }
    probe.flag = {
        s: ctx.indicators.expr("close > sma(close, 6) and volume > sma(volume, 6)", instrument=s)
        for s in syms
    }
    probe.slow = {s: ctx.indicators.expr("sma(close, 2)", instrument=s, timeframe="15m") for s in syms}


# These define no deprecated hook of their own, so they are outside the warns blocks.
class _ExprProbe(_BarProbe):
    """_BarProbe that also logs what its expressions show on every bar."""

    def on_start(self, ctx):
        super().on_start(ctx)
        _add_expressions(self, ctx)

    def on_bar(self, ctx):
        sym = self._key(ctx)
        self._rec(
            ctx, "expr", sym, self.num[sym].value, self.flag[sym].value, self.slow[sym].value,
            tuple((s, self.num[s].value, self.flag[s].value) for s in sorted(self.num)),
        )
        super().on_bar(ctx)


class _ExprOpenProbe(_OpenHookProbe):
    """_OpenHookProbe whose fill hooks also read every symbol's expressions."""

    def on_start(self, ctx):
        super().on_start(ctx)
        _add_expressions(self, ctx)

    def _open_read(self, ctx, what):
        super()._open_read(ctx, what)
        self._rec(
            ctx, what + "-expr", self._key(ctx),
            tuple((s, self.num[s].value, self.flag[s].value, self.slow[s].value)
                  for s in sorted(self.num)),
        )


@pytest.mark.parametrize("frac", CUTS)
@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("dataset", sorted(BAR_DATA))
@pytest.mark.parametrize("stamp", ["open", "close"])
def test_expressions_on_the_bank_ignore_the_future(stamp, dataset, timing, frac, portfolio_runner):
    _check(
        "portfolio", _ExprProbe, BAR_DATA[dataset](), timing, "intent", frac, _poison_bars, stamp,
        portfolio_runner,
    )


@pytest.mark.parametrize("frac", CUTS)
@pytest.mark.parametrize("dataset", sorted(BAR_DATA))
@pytest.mark.parametrize("stamp", ["open", "close"])
def test_a_fill_hook_at_the_open_sees_no_expression_of_the_bar_opening(
    stamp, dataset, frac, portfolio_runner, monkeypatch
):
    monkeypatch.setattr(sys.modules[__name__], "_cfg", _atr_cfg)
    _check(
        "portfolio", _ExprOpenProbe, BAR_DATA[dataset](), "next_bar_open", "intent", frac,
        _poison_bars, stamp, portfolio_runner,
    )
