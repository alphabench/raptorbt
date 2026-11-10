"""``run_strategies_windowed`` gives the results of ``run_strategies``, whatever the cuts.

The same strategies over the same tape, fed whole and fed in windows (one session, 7, 30, the
whole run, and seeded random cuts), must agree on every trade, order, curve, metric and the
full report, to the bit. The examples and the batch-versus-live structure cases of
``test_structures`` are replayed that way; the other tests pin what only a windowed run has:
the refusals, the history it holds, the memory it saves and the curve it hands out.
"""

from __future__ import annotations

import contextlib
import io
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import raptorbt
from raptorbt import (
    BacktestConfig,
    Indicator,
    Strategy,
    Window,
    run_strategies,
    run_strategies_windowed,
)
from raptorbt import structures as st
from raptorbt.errors import RaptorBTError
from raptorbt.strategy import multi_runner
from raptorbt.strategy._stamps import BarStamps

from test_structures import make, notional_cut, universe

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SCRIPTS = sorted(p for p in EXAMPLES.glob("*/*.py") if not p.name.startswith("_"))
DAY_NS = 86_400 * 10**9
IST_NS = 19_800 * 10**9
MIN = 60 * 10**9

# How a tape is cut: (kind, size or seed). A session is an IST calendar day.
CUTS = [("sessions", 1), ("sessions", 7), ("sessions", 30), ("whole", 0), ("random", 1), ("random", 2)]
CUT_IDS = [f"{kind}-{n}" if kind != "whole" else "whole" for kind, n in CUTS]


# -- cutting a tape into windows ---------------------------------------------------------------


def _event_times(data, ticks, depth, offsets):
    """Every event time of the tape, with each symbol's own, ascending."""
    own = {}
    for sym, arrays in (data or {}).items():
        own[("bars", sym)] = np.asarray(arrays["timestamps"], dtype=np.int64) + offsets.get(sym, 0)
    for sym, arrays in (ticks or {}).items():
        own[("ticks", sym)] = np.asarray(arrays["timestamps"], dtype=np.int64)
    for sym, arrays in (depth or {}).items():
        own[("depth", sym)] = np.asarray(arrays["timestamps"], dtype=np.int64)
    every = np.unique(np.concatenate(list(own.values()))) if own else np.empty(0, dtype=np.int64)
    return own, every


def window_ends(every, cut, *, by="sessions"):
    """The last event time of each window: ``every`` (the distinct event times) cut in
    sessions of the IST calendar, or in runs of that many timestamps (``by="times"``), or at
    random timestamps, or not at all."""
    kind, n = cut
    if len(every) == 0 or kind == "whole":
        return [int(every[-1])] if len(every) else []
    if kind == "random":
        rng = random.Random(n)
        picks = sorted(rng.sample(range(len(every)), k=min(len(every), rng.randint(2, 12))))
        ends = {int(every[i]) for i in picks} | {int(every[-1])}
        return sorted(ends)
    if by == "times":
        return [int(every[min(i + n, len(every)) - 1]) for i in range(0, len(every), n)]
    sessions = (every + IST_NS) // DAY_NS
    firsts = sorted(set(sessions.tolist()))
    chunks = [firsts[i : i + n] for i in range(0, len(firsts), n)]
    return [int(every[np.searchsorted(sessions, chunk[-1], side="right") - 1]) for chunk in chunks]


def make_windows(data, ticks, depth, offsets, ends):
    """``Window`` objects over ``(previous end, end]`` for each of ``ends``: every symbol is in
    every window (an empty slice carries nothing), which is how the first window declares it."""
    own, _ = _event_times(data, ticks, depth, offsets)
    out, previous = [], None
    for end in ends:
        def piece(kind, sym, arrays):
            times = own[(kind, sym)]
            lo = 0 if previous is None else int(np.searchsorted(times, previous, side="right"))
            hi = int(np.searchsorted(times, end, side="right"))
            return {key: np.asarray(values)[lo:hi] for key, values in arrays.items()}

        out.append(
            Window(
                bars={s: piece("bars", s, a) for s, a in (data or {}).items()},
                ticks={s: piece("ticks", s, a) for s, a in (ticks or {}).items()},
                depth={s: piece("depth", s, a) for s, a in (depth or {}).items()},
                end_ns=end,
            )
        )
        previous = end
    return out


def windowed_like(strategies, data=None, config=None, *, ticks=None, depth=None, cut=("whole", 0),
                  by="sessions", **kw):
    """``run_strategies(strategies, data, config, ticks=, depth=, **kw)`` as a windowed run."""
    kw.pop("_via", None)
    data, ticks, depth = data or {}, ticks or {}, depth or {}
    stamps = BarStamps.resolve(kw.get("bar_stamp"), kw.get("bar_ns"), list(data), "test", require=bool(data))
    offsets = {s: stamps.offset(s) for s in data}
    own, every = _event_times(data, ticks, depth, offsets)
    windows = make_windows(data, ticks, depth, offsets, window_ends(every, cut, by=by))
    symbols = list({**dict.fromkeys(data), **dict.fromkeys(ticks)})
    return run_strategies_windowed(strategies, windows, config, symbols=symbols, **kw)


# -- comparing results ---------------------------------------------------------------------------


def _plain(obj):
    return {
        name: getattr(obj, name)
        for name in sorted(dir(obj))
        if not name.startswith("_") and not callable(getattr(obj, name))
    }


def digest(portfolio):
    """Everything a result shows, as text: a float is its ``repr``, so a bit that differs shows."""
    r = portfolio.result
    equity = r.equity_curve()
    stamps = np.arange(len(equity), dtype=np.int64) * MIN
    try:
        report = portfolio.full_report(stamps)
    except Exception as error:  # the same inputs must fail the same way
        report = ("error", type(error).__name__, str(error))
    return repr(
        {
            "trades": [_plain(t) for t in r.trades()],
            "orders": [_plain(o) for o in r.orders()],
            "equity": equity.tolist(),
            "drawdown": r.drawdown_curve().tolist(),
            "returns": r.returns().tolist(),
            "metrics": _plain(r.metrics),
            "result": (r.halted, r.halted_at, r.rejected_entries, r.unfunded_entries),
            "portfolio": (portfolio.halted, portfolio.halted_at, portfolio.rejected_entries),
            "per_instrument": [_plain(i) for i in portfolio.per_instrument],
            "report": report,
        }
    )


def digests(out):
    """The digests of what a ``run_strategies`` call returned (its results, then the accounts)."""
    results = out[0] if isinstance(out, tuple) else out
    return [digest(p) for p in results]


# -- the examples ------------------------------------------------------------------------------


def _interpose(monkeypatch, replacement, name="run_strategies"):
    """Every ``name`` (``run_strategies``, ``run_strategy_backtest``) a script can reach is
    ``replacement`` for the length of the test."""
    original = getattr(raptorbt, name)
    for module in list(sys.modules.values()):
        if module is sys.modules[__name__]:
            continue
        try:
            found = getattr(module, name, None) is original
        except Exception:  # a lazy module that cannot answer
            continue
        if found:
            monkeypatch.setattr(module, name, replacement)
    return original


def _play(script):
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script), run_name="__main__")


#: What a windowed run refuses by design (see its docstring): the call runs whole instead.
REFUSED = {"invalid_config", "invalid_adjustment_policy"}


def _fresh(strategy):
    """Another strategy like ``strategy``, not run yet (an instance holds native state and cannot
    be copied): its class, built with its config when it takes one."""
    cls = strategy if isinstance(strategy, type) else type(strategy)
    config = None if isinstance(strategy, type) else getattr(strategy, "config", None)
    try:
        return cls(config) if config is not None else cls()
    except TypeError:
        return cls()


def _single_instrument_check(cut, counted):
    """``run_strategy_backtest`` (the one-instrument runner) is not ``run_strategies``, but the
    strategy it is given is one: the same strategy and data go through both ``run_strategies``
    and the windowed driver, and what the script receives is still the runner's own result."""
    original = raptorbt.run_strategy_backtest

    def check(strategy, timestamps, open, high, low, close, volume, direction=1, symbol="ASSET",
              config=None, instrument_config=None, instrument=None, oms_type="netting",
              account_type="cash", leverage=1.0, bar_stamp=None, bar_ns=None,
              corporate_actions=None, adjustment_policy=None):
        spare = [_fresh(strategy) for _ in range(2)]
        data = {symbol: {"timestamps": timestamps, "open": open, "high": high, "low": low, "close": close,
                         "volume": volume}}
        kw = dict(
            directions={symbol: direction}, oms_type=oms_type, account_type=account_type, leverage=leverage,
            bar_stamp=bar_stamp, bar_ns=bar_ns, corporate_actions=corporate_actions,
            adjustment_policy=adjustment_policy,
        )
        if instrument is not None:
            kw["instruments"] = {symbol: instrument}
        if instrument_config is not None:
            kw["instrument_configs"] = {symbol: instrument_config}
        want = run_strategies([spare[0]], data, config, **kw)
        try:
            got = windowed_like([spare[1]], data, config, cut=cut, **kw)
        except RaptorBTError as refused:
            if refused.code not in REFUSED:
                raise
        else:
            assert digests(got) == digests(want)
            counted.append(1)
        return original(strategy, timestamps, open, high, low, close, volume, direction, symbol, config,
                        instrument_config, instrument, oms_type, account_type, leverage, bar_stamp, bar_ns,
                        corporate_actions, adjustment_policy)

    return check


@pytest.fixture(scope="module")
def recorded():
    """Each example's ``run_strategies`` calls, run whole, once: a list of digest lists."""
    cache = {}

    def get(script, monkeypatch):
        if script not in cache:
            original = raptorbt.run_strategies
            seen = []

            def spy(*args, **kw):
                out = original(*args, **kw)
                seen.append(digests(out))
                return out

            _interpose(monkeypatch, spy)
            _play(script)
            monkeypatch.undo()
            cache[script] = seen
        return cache[script]

    return get


@pytest.mark.parametrize("cut", CUTS, ids=CUT_IDS)
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_example_gives_the_same_results_in_windows(script, cut, recorded, monkeypatch):
    whole = recorded(script, monkeypatch)
    original = raptorbt.run_strategies
    seen = []

    def windowed(strategies, data=None, config=None, **kw):
        try:
            out = windowed_like(strategies, data, config, cut=cut, **kw)
        except RaptorBTError as refused:
            if refused.code not in REFUSED:
                raise
            seen.append(None)
            return original(strategies, data, config, **kw)
        seen.append(digests(out))
        return out

    singles = []
    _interpose(monkeypatch, windowed)
    _interpose(monkeypatch, _single_instrument_check(cut, singles), "run_strategy_backtest")
    _play(script)
    assert len(seen) == len(whole), "the example made another number of calls"
    for k, (got, want) in enumerate(zip(seen, whole)):
        if got is not None:
            assert got == want, f"call {k} of {script.name}, cut {cut}"
    if not whole and not singles:
        pytest.skip("the example runs no strategy through run_strategies or run_strategy_backtest")


# -- batch run and live node cases of test_structures, windowed -----------------------------------


STRUCTURE_CASES = [
    (st.iron_condor(1, 2), {"exit": "close_at"}, {}),
    (st.Structure([st.Leg("call", side="sell")], "short_call"), {"exit": "stop"}, {}),
    (st.iron_condor(1, 2), {}, {"max_order_notional": "cut"}),
]


@pytest.mark.parametrize("cut", [("times", 1), ("times", 7), ("times", 30), ("whole", 0), ("random", 3), ("random", 4)],
                         ids=["bar-1", "bar-7", "bar-30", "whole", "random-3", "random-4"])
@pytest.mark.parametrize("structure,attrs,config", STRUCTURE_CASES,
                         ids=["iron_condor_time_exit", "straddle_stop", "condor_leg_failure"])
def test_structure_runs_are_the_same_windowed(structure, attrs, config, cut):
    from _chain_fixture import STEP, T0

    attrs = dict(attrs)
    if attrs.get("exit") == "close_at":
        attrs["exit"] = st.Exit(close_at_ns=T0 + 25 * STEP)
    elif attrs.get("exit") == "stop":
        attrs["exit"] = st.Exit(stop_loss=150.0)
    if config.get("max_order_notional") == "cut":
        config = {"max_order_notional": notional_cut()}
    cls = make(structure, **attrs)
    whole_strategy, windowed_strategy = cls(), cls()
    data, inst = universe(drift=0.4 if "stop_loss" in str(attrs) else 0.0)
    cfg = BacktestConfig(initial_capital=2_000_000.0, fees=0.0, **config)
    kw = dict(instruments=inst, account_type="margin", leverage=1.0, bar_stamp="close")
    want = run_strategies([whole_strategy], data, cfg, **kw)[0]
    got = windowed_like([windowed_strategy], data, cfg, cut=cut, by="times", **kw)[0]
    assert want.result.orders(), "the structure placed nothing"
    assert digest(got) == digest(want)
    assert windowed_strategy.group.reason == whole_strategy.group.reason
    assert windowed_strategy.structure_status(windowed_strategy.group) == whole_strategy.structure_status(
        whole_strategy.group
    )


# -- strategies that read what a window must keep ------------------------------------------------


def _bars(days, seed, start=100.0, minutes=75):
    """``days`` sessions of ``minutes`` 1-minute bars each, close-stamped, from 09:16 IST."""
    rng = np.random.default_rng(seed)
    first = 1_767_589_800 * 10**9  # 2026-01-05 04:30 UTC: 10:00 IST
    ts = np.concatenate([first + d * DAY_NS + np.arange(1, minutes + 1, dtype=np.int64) * MIN for d in range(days)])
    close = start + np.cumsum(rng.normal(0.0, 0.3, len(ts)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return {
        "timestamps": ts,
        "open": open_,
        "high": np.maximum(open_, close) + 0.1,
        "low": np.minimum(open_, close) - 0.1,
        "close": close,
        "volume": np.full(len(ts), 1000.0),
    }


DATA = {"A": _bars(6, 7), "B": _bars(6, 11, start=60.0)}
CONFIG = BacktestConfig(initial_capital=100_000.0, fees=0.0003)
KW = dict(bar_stamp="close")


class Cross(Strategy):
    """Two SMAs per symbol; no series read."""

    def on_start(self, ctx):
        self.fast = {s: self.register_indicator(Indicator.sma(5), symbol=s) for s in ctx.symbols}
        self.slow = {s: self.register_indicator(Indicator.sma(20), symbol=s) for s in ctx.symbols}

    def on_bar(self, ctx):
        f, s = self.fast[ctx.symbol], self.slow[ctx.symbol]
        if not (f.initialized and s.initialized):
            return
        if ctx.is_flat and f.value > s.value:
            self.enter(size_frac=0.3)
        elif not ctx.is_flat and f.value < s.value:
            self.close_position()


class Reader(Cross):
    """The same, and it reads ``ctx.series`` without saying how much."""

    def on_start(self, ctx):
        super().on_start(ctx)
        self.seen = []

    def on_bar(self, ctx):
        closes = ctx.series()["close"]
        self.seen.append((ctx.symbol, ctx.idx, len(closes), float(closes[0]), float(closes[-1])))
        super().on_bar(ctx)


class Keeper(Reader):
    """Reads at most the last 30 rows, and says so."""

    def on_start(self, ctx):
        super().on_start(ctx)
        for symbol in ctx.symbols:
            ctx.keep_history(30, symbol=symbol)


def _same(strategy_cls, cut, **kw):
    kwargs = {**KW, **kw}
    want = run_strategies([strategy_cls()], DATA, CONFIG, **kwargs)[0]
    got = windowed_like([strategy_cls()], DATA, CONFIG, cut=cut, **kwargs)[0]
    assert digest(got) == digest(want)
    return want


@pytest.mark.parametrize("cut", CUTS, ids=CUT_IDS)
@pytest.mark.parametrize("cls", [Cross, Reader], ids=["no-series", "series-undeclared"])
def test_indicators_and_undeclared_series_are_the_batch_runs(cls, cut):
    assert _same(cls, cut).result.trades(), "the strategy traded"


def test_an_undeclared_series_reader_sees_the_arrays_of_the_batch_run():
    whole, windowed = Reader(), Reader()
    run_strategies([whole], DATA, CONFIG, **KW)
    windowed_like([windowed], DATA, CONFIG, cut=("sessions", 1), **KW)
    assert windowed.seen == whole.seen


def test_a_keep_history_holds_that_many_rows_and_the_trades_are_unchanged():
    keeper = Keeper()
    got = windowed_like([keeper], DATA, CONFIG, cut=("sessions", 1), **KW)[0]
    assert digest(got) == digest(run_strategies([Keeper()], DATA, CONFIG, **KW)[0])
    # the ring holds the declared rows or what the indicators read, whichever is more (20 here
    # is the slow SMA), never every row
    lengths = {n for _, _, n, _, _ in keeper.seen}
    assert max(lengths) < len(DATA["A"]["close"]) and max(lengths) >= 30


@pytest.mark.parametrize("history", ["all", 50, 5000])
def test_history_all_and_a_row_count_keep_the_trades(history):
    want = run_strategies([Cross()], DATA, CONFIG, **KW)[0]
    got = windowed_like([Cross()], DATA, CONFIG, cut=("sessions", 2), history=history, **KW)[0]
    assert digest(got) == digest(want)


def test_a_series_read_after_rows_were_released_is_refused_not_shortened():
    class Late(Cross):
        def on_bar(self, ctx):
            # not named `series` in the source: the first read is what finds it
            if ctx.idx >= 100:
                getattr(ctx, "ser" + "ies")()
            super().on_bar(ctx)

    with pytest.raises(RaptorBTError) as e:
        windowed_like([Late()], DATA, CONFIG, cut=("sessions", 1), **KW)
    assert e.value.code == "history_exhausted"


def test_keep_history_in_on_start_sizes_the_rings_and_declares_the_bound():
    class Kept(Reader):
        def on_start(self, ctx):
            super().on_start(ctx)
            for symbol in ctx.symbols:
                ctx.keep_history(40, symbol=symbol)

    strategy = Kept()
    got = windowed_like([strategy], DATA, CONFIG, cut=("sessions", 1), **KW)[0]
    assert digest(got) == digest(run_strategies([Kept()], DATA, CONFIG, **KW)[0])
    assert max(n for _, _, n, _, _ in strategy.seen) < len(DATA["A"]["close"])


# -- what is held ---------------------------------------------------------------------------------


def test_a_windowed_run_holds_fewer_rows_than_the_whole_run():
    _, whole = run_strategies([Cross()], DATA, CONFIG, memory_report=True, **KW)
    _, small = windowed_like([Cross()], DATA, CONFIG, cut=("sessions", 1), memory_report=True, **KW)
    _, more = windowed_like(
        [Cross()], {s: _bars(12, 7 if s == "A" else 11, start=100.0 if s == "A" else 60.0) for s in DATA},
        CONFIG, cut=("sessions", 1), memory_report=True, **KW)
    assert small["shared_store"] < whole["shared_store"] / 10
    assert more["shared_store"] == small["shared_store"], "twice the tape, the same history"


# -- the curve a window hands out -----------------------------------------------------------------


def test_drain_curve_hands_out_the_curve_a_window_at_a_time(monkeypatch):
    pieces = []
    real = multi_runner._feed_window

    def spy(call, window, arrays, k, offsets, prev_end, index, groups):
        out = real(call, window, arrays, k, offsets, prev_end, index, groups)
        for _, group in groups:
            for _, sid, _ in group.setups:
                pieces.append(group.rt.drain_curve(sid))
        return out

    monkeypatch.setattr(multi_runner, "_feed_window", spy)
    got = windowed_like([Cross()], DATA, CONFIG, cut=("sessions", 1), **KW)[0]
    curve = got.result.equity_curve()
    drained = np.concatenate([e for _, e in pieces])
    stamps = np.concatenate([t for t, _ in pieces])
    # the newest sample is held back until `finish` settles it
    assert drained.tolist() == curve[:-1].tolist()
    assert (np.diff(stamps) > 0).all()
    assert len(pieces) == 6


# -- expressions on a coarser clock ----------------------------------------------------------------


@pytest.mark.parametrize("stamp", ["close", "open"])
@pytest.mark.parametrize("cut", [("sessions", 1), ("sessions", 3), ("random", 7)])
def test_an_expression_on_a_daily_clock_is_the_batch_run(stamp, cut):
    hour = 60 * MIN
    rng = np.random.default_rng(3)
    ts = np.array([(20_000 + d) * DAY_NS + k * hour + (hour if stamp == "close" else 0)
                   for d in range(12) for k in range(24)], dtype=np.int64)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, len(ts)))
    opens = np.concatenate([[close[0]], close[:-1]])
    data = {"ASSET": {"timestamps": ts, "open": opens, "high": np.maximum(opens, close) + 0.3,
                      "low": np.minimum(opens, close) - 0.3, "close": close, "volume": np.full(len(ts), 900.0)}}

    class Daily(Strategy):
        def on_start(self, ctx):
            self.h = ctx.indicators.expr('at("1d", sma(close, 3))', instrument="ASSET")
            self.seen = []

        def on_bar(self, ctx):
            v = self.h.value
            self.seen.append(float(v) if v is not None else None)
            if ctx.is_flat and v is not None and ctx.bar.close > v:
                self.enter(size_frac=0.2)
            elif not ctx.is_flat and v is not None and ctx.bar.close < v:
                self.close_position()

    kw = dict(bar_stamp=stamp, **({"bar_ns": hour} if stamp == "open" else {}))
    whole, windowed = Daily(), Daily()
    want = run_strategies([whole], data, CONFIG, **kw)
    got = windowed_like([windowed], data, CONFIG, cut=cut, **kw)
    assert any(v is not None for v in whole.seen)
    assert windowed.seen == whole.seen
    assert digests(got) == digests(want)


# -- the other drivers' options -------------------------------------------------------------------


@pytest.mark.parametrize("cut", [("sessions", 1), ("random", 6)])
def test_bars_stamped_at_their_open_are_the_batch_run(cut):
    opened = {s: {**d, "timestamps": d["timestamps"] - MIN} for s, d in DATA.items()}
    kw = dict(bar_stamp="open", bar_ns=MIN)
    want = run_strategies([Reader()], opened, CONFIG, **kw)
    got = windowed_like([Reader()], opened, CONFIG, cut=cut, **kw)
    assert want[0].result.trades()
    assert digests(got) == digests(want)


@pytest.mark.parametrize("cut", [("sessions", 1), ("random", 5)])
def test_accounts_batches_and_trade_from_are_the_batch_runs(cut):
    kw = dict(
        accounts={"book": raptorbt.AccountSpec(initial_capital=200_000.0)},
        trade_from=int(DATA["A"]["timestamps"][100]),
        batch_size=1,
        **KW,
    )
    entries = lambda: [({"strategy": Cross(), "account": "book"}), ({"strategy": Reader(), "account": "book"}), Cross()]  # noqa: E731
    want = run_strategies(entries(), DATA, CONFIG, **kw)
    got = windowed_like(entries(), DATA, CONFIG, cut=cut, **kw)
    assert digests(got) == digests(want)
    for name in want[1]:
        assert got[1][name].balance == want[1][name].balance
        assert got[1][name].equity_curve().tolist() == want[1][name].equity_curve().tolist()
        assert got[1][name].strategies == want[1][name].strategies


def test_ticks_depth_and_primary_bars_are_the_batch_run():
    rng = np.random.default_rng(5)
    n = 1500
    ts = 1_767_589_800 * 10**9 + np.cumsum(rng.integers(0, 3, n)) * 10**9 + 10**9
    ltp = np.round(100 + np.cumsum(rng.normal(0, 0.1, n)), 2)
    ltp[rng.random(n) < 0.1] = 0.0  # a quote with no print, and a row with neither
    quote = np.where(rng.random(n) < 0.9, 0.05, 0.0)
    ticks = {"X": {"timestamps": ts, "ltp": ltp, "bid": 100.0 - quote, "ask": 100.0 + quote}}
    m = 200
    dts = np.sort(rng.choice(ts, m))
    depth = {"X": {"timestamps": dts, "bid_prices": np.tile([99.9, 99.8], (m, 1)), "bid_sizes": np.ones((m, 2)),
                   "ask_prices": np.tile([100.1, 100.2], (m, 1)), "ask_sizes": np.ones((m, 2))}}

    class Ticker(Strategy):
        def on_start(self, ctx):
            self.n = 0
            self.sma = self.register_indicator(Indicator.sma(3))

        def on_trade(self, ctx, tick):
            self.n += 1
            if ctx.is_flat and self.n % 40 == 0:
                self.enter(size_frac=0.2)
            elif not ctx.is_flat and self.n % 40 == 20:
                self.close_position()

        def on_book(self, ctx, book):
            self.top = ctx.series()["ltp"][-1:].tolist()

    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0003)
    want = run_strategies([Ticker()], None, cfg, ticks=ticks, depth=depth, primary_bars=(1, "m"))
    assert want[0].result.trades()
    for cut in [("whole", 0), ("random", 1), ("random", 9)]:
        got = windowed_like([Ticker()], None, cfg, ticks=ticks, depth=depth, primary_bars=(1, "m"), cut=cut)
        assert digests(got) == digests(want), cut


# -- what a windowed run refuses ------------------------------------------------------------------


def _windows(cut=("sessions", 2)):
    own, every = _event_times(DATA, None, None, {"A": 0, "B": 0})
    return make_windows(DATA, None, None, {}, window_ends(every, cut))


def _refused(call, code, parameter):
    with pytest.raises(RaptorBTError) as e:
        call()
    assert (e.value.code, e.value.parameter) == (code, parameter)
    assert e.value.operation == "run_strategies_windowed"
    return e.value


def test_a_window_that_overlaps_or_cuts_a_timestamp_is_refused():
    w = _windows()
    run = lambda ws: run_strategies_windowed([Cross()], ws, CONFIG, symbols=["A", "B"], **KW)  # noqa: E731
    _refused(lambda: run([w[0], w[0]]), "window_overlap", "windows")
    _refused(lambda: run([w[1], w[0]]), "window_overlap", "windows")
    # a window cut in the middle of an instant: its last timestamp is also the next one's first
    a = DATA["A"]
    cut = {s: {k: np.asarray(v)[:50] for k, v in d.items()} for s, d in DATA.items()}
    rest = {s: {k: np.asarray(v)[49:100] for k, v in d.items()} for s, d in DATA.items()}
    _refused(lambda: run([Window(bars=cut), Window(bars=rest)]), "window_overlap", "windows")
    # a row after the window's own end_ns
    late = Window(bars=cut, end_ns=int(a["timestamps"][10]))
    _refused(lambda: run([late]), "window_overlap", "end_ns")


def test_the_engine_refuses_an_overlapping_window_whatever_the_python_said():
    """The Rust cursor holds the line: ``push_window`` of a window at or before the last event."""
    w = _windows(("whole", 0))[0]
    original = multi_runner._feed_window
    calls = []

    def twice(call, window, arrays, k, offsets, prev_end, index, groups):
        out = original(call, window, arrays, k, offsets, prev_end, index, groups)
        # the same rows again, past the checks made in Python
        _, group = groups[0]
        bars = {index[s]: a for s, a in arrays[0].items()}
        try:
            group.rt.push_window(bars, {}, {})
        except RaptorBTError as error:
            calls.append(error.code)
        return out

    multi_runner._feed_window = twice
    try:
        run_strategies_windowed([Cross()], [w], CONFIG, symbols=["A", "B"], **KW)
    finally:
        multi_runner._feed_window = original
    assert calls == ["window_overlap"]


def test_a_window_must_match_what_the_first_one_declared():
    w = _windows()
    run = lambda ws, symbols=("A", "B"): run_strategies_windowed([Cross()], ws, CONFIG, symbols=list(symbols), **KW)  # noqa: E731
    only_a = Window(bars={"A": {k: np.asarray(v)[:5] for k, v in DATA["A"].items()}})
    _refused(lambda: run([only_a]), "window_schema", "windows")
    after = int(DATA["A"]["timestamps"][-1]) + MIN
    ticks = Window(ticks={"A": {"timestamps": np.array([after], dtype=np.int64), "ltp": np.array([1.0])}})
    _refused(lambda: run([w[0], ticks]), "window_schema", "ticks")


def test_what_is_not_a_window_is_refused():
    _refused(lambda: run_strategies_windowed([Cross()], [{"A": {}}], CONFIG, symbols=["A"], **KW), "invalid_window", "windows")
    _refused(lambda: run_strategies_windowed([Cross()], [], CONFIG, symbols=["A"], **KW), "invalid_window", "windows")
    bad = Window(bars={"A": {}}, end_ns="soon")
    _refused(lambda: run_strategies_windowed([Cross()], [bad], CONFIG, symbols=["A"], **KW), "invalid_window", "end_ns")


@pytest.mark.parametrize("history", [0, -3, "most", True])
def test_a_bad_history_is_refused(history):
    _refused(lambda: run_strategies_windowed([Cross()], _windows(), CONFIG, symbols=["A", "B"], history=history, **KW),
             "invalid_history", "history")


def test_atr_stops_and_restating_adjustments_are_refused():
    atr = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    atr.set_atr_stop(2.0, 14)
    e = _refused(lambda: run_strategies_windowed([Cross()], _windows(), atr, symbols=["A", "B"], **KW),
                 "invalid_config", "atr_period")
    assert "windowed" in e.message
    split = raptorbt.CorporateAction.split("A", int(DATA["A"]["timestamps"][60]), 2.0)
    _refused(lambda: run_strategies_windowed([Cross()], _windows(), CONFIG, symbols=["A", "B"],
                                             corporate_actions=[split], adjustment_policy="adjusted", **KW),
             "invalid_adjustment_policy", "adjustment_policy")


def test_the_on_window_callback_hears_each_window():
    heard = []
    ws = _windows(("sessions", 1))
    run_strategies_windowed([Cross()], ws, CONFIG, symbols=["A", "B"], on_window=heard.append, **KW)
    assert [d.index for d in heard] == list(range(len(ws)))
    assert [d.end_ns for d in heard] == [w.end_ns for w in ws]
    assert heard[-1].total_events == sum(d.events for d in heard) == 2 * len(DATA["A"]["close"])
