"""bar_stamp / bar_ns on the strategy runners.

The engine's one internal clock is bar close time. A run given open-stamped
bars (``bar_stamp="open", bar_ns=width``) and a run given the same bars
close-stamped (``bar_stamp="close"``) must therefore be indistinguishable:
same decisions, orders, fills, composite bars, equity and metrics, all at the
same visible times. Only the user-facing raw stamp (``ctx.timestamp``)
differs, and it must stay the user's own stamp.
"""

import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Indicator, Strategy
from raptorbt.errors import InvalidParameter
from raptorbt.strategy import TickStrategyStream, orders

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benches" / "runtime"))
import synthetic  # noqa: E402

MIN_NS = 60 * 10**9


@pytest.fixture(scope="module")
def ds():
    return synthetic.generate(n_instruments=4, days=2, timeframes=("1m",))


def _series(bars, stamp):
    ts = bars["ts_open"] if stamp == "open" else bars["ts_close"]
    out = {"timestamps": np.ascontiguousarray(ts)}
    for key in ("open", "high", "low", "close", "volume"):
        out[key] = np.ascontiguousarray(bars[key])
    return out


def _stamp_kwargs(stamp):
    return {"bar_stamp": "open", "bar_ns": 60e9} if stamp == "open" else {"bar_stamp": "close"}


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class _Det(Strategy):
        """Deterministic strategy touching every time-sensitive surface.

        Records only close-time quantities, so open- and close-stamped runs can be
        compared entry by entry.
        """

        def __init__(self):
            super().__init__()
            self.trace = []
            self.raw = []

        def on_start(self, ctx):
            syms = list(getattr(ctx, "symbols", [None]))
            self.fast = self.register_indicators(lambda: Indicator.sma(5), syms)
            self.slow = self.register_indicators(lambda: Indicator.sma(12), syms)
            self.s5 = self.subscribe_bars(5, "m")
            self.s15 = self.subscribe_bars(15, "m")
            self.s1d = self.subscribe_bars(1, "d")
            self.comp = self.register_indicators(
                lambda: Indicator.sma(3), syms, stream_id=self.s15
            )
            self.last15 = {}
            self.clock.set_timer("half-hour", 30 * 60 * 10**9)

        @staticmethod
        def _sym(ctx):
            return getattr(ctx, "symbol", None)

        def on_timer(self, ctx, event):
            self.trace.append(("time", self._sym(ctx), tuple(event)))

        def on_composite_bar(self, ctx, bar):
            if bar.stream_id == self.s15:
                self.last15[bar.symbol] = bar.close
            self.trace.append(("comp", self._sym(ctx), tuple(bar)))

        def on_fill(self, ctx, e):
            self.trace.append(("fill", self._sym(ctx), ctx.bar_close_ts, e.kind, e.price, e.size))

        def on_order_expired(self, ctx, e):
            self.trace.append(("expired", self._sym(ctx), ctx.bar_close_ts))

        def on_bar(self, ctx):
            sym = self._sym(ctx)
            now = ctx.bar_close_ts
            self.raw.append((sym, ctx.timestamp, now))
            fast, slow, comp = self.fast[sym].value, self.slow[sym].value, self.comp[sym].value
            pos = ctx.position
            bar = tuple(ctx.bar)[1:]
            self.trace.append(
                ("bar", sym, now, bar, fast, slow, comp, ctx.equity,
                 None if pos is None else (pos.size, pos.entry_price))
            )
            # Local (IST) minute of the bar's close: flatten from 15:25, the way a
            # broker's square-off reads the clock.
            minute = ((now + synthetic.IST_OFFSET_NS) // (60 * 10**9)) % 1440
            late = minute >= 15 * 60 + 25
            if pos is not None and late:
                self.close_position()
                return
            if ctx.idx % 97 == 50 and pos is None:
                # Far below the market: never fills, expires with the day.
                self.submit_order(
                    orders.Limit(side="buy", units=1.0, price=round(ctx.bar.close * 0.5, 2),
                                 tif="day")
                )
            if ctx.idx % 41 == 7 and pos is None and not late:
                self.submit_order(
                    orders.Limit(side="buy", units=2.0, price=round(ctx.bar.close * 0.998, 2),
                                 tif="day")
                )
            if fast is None or slow is None or late:
                return
            gate = comp is None or self.last15.get(sym, 0.0) >= comp
            if pos is None and fast > slow and gate:
                self.enter(size_frac=0.1)
            elif pos is not None and fast < slow:
                self.close_position()


def _public(obj):
    return {k: repr(getattr(obj, k)) for k in dir(obj) if not k.startswith("_")
            and not callable(getattr(obj, k))}


def _outcome(result):
    return {
        "trades": [_public(t) for t in result.trades()],
        "orders": [_public(o) for o in result.orders()],
        "equity": result.equity_curve().tolist(),
        "returns": result.returns().tolist(),
        "metrics": _public(result.metrics),
    }


def _config(timing):
    return BacktestConfig(
        initial_capital=10_000_000.0,
        fees=0.0002,
        fill_timing=timing,
        session_tz_offset_ns=synthetic.IST_OFFSET_NS,
    )


def _run_single(ds, stamp, timing):
    sym = next(iter(ds.bars["1m"]))
    strat = _Det()
    res = raptorbt.run_strategy_backtest(
        strat, **_series(ds.bars["1m"][sym], stamp), config=_config(timing),
        symbol=sym, **_stamp_kwargs(stamp),
    )
    return strat, res


def _run_portfolio(ds, stamp, timing, portfolio_runner):
    data = {s: _series(b, stamp) for s, b in ds.bars["1m"].items()}
    strat = _Det()
    kwargs = _stamp_kwargs(stamp)
    if stamp == "open":
        kwargs["bar_ns"] = {s: 60e9 for s in data}
    res = portfolio_runner(strat, data, config=_config(timing), **kwargs)
    return strat, res.result


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
@pytest.mark.parametrize("runner", [_run_single, _run_portfolio])
def test_open_and_close_stamped_runs_are_indistinguishable(ds, runner, timing, portfolio_runner):
    if runner is _run_portfolio:
        runner = functools.partial(runner, portfolio_runner=portfolio_runner)
    s_open, r_open = runner(ds, "open", timing)
    s_close, r_close = runner(ds, "close", timing)

    # Not vacuous: composites of every width, fills, an expired DAY order.
    kinds = {t[0] for t in s_close.trace}
    assert {"comp", "fill", "expired", "time", "bar"} <= kinds
    assert {t[2][1:3] for t in s_close.trace if t[0] == "comp"} >= {(5, "m"), (15, "m"), (1, "d")}
    assert r_close.trades()

    assert s_open.trace == s_close.trace
    assert _outcome(r_open) == _outcome(r_close)

    # The visible clock is the same; the user's own stamp is untouched.
    for (sym_o, raw_o, vis_o), (sym_c, raw_c, vis_c) in zip(s_open.raw, s_close.raw):
        assert sym_o == sym_c and vis_o == vis_c
        assert raw_o == vis_o - MIN_NS
        assert raw_c == vis_c


def test_daily_composite_holds_exactly_the_days_bars(ds):
    """The last bar of a session closes at 15:30 and belongs to that day."""
    strat, _ = _run_single(ds, "open", "same_bar_close")
    sym = next(iter(ds.bars["1m"]))
    bars = ds.bars["1m"][sym]
    ist = synthetic.IST_OFFSET_NS
    day_end = ((int(bars["ts_close"][0]) + ist) // synthetic.DAY_NS + 1) * synthetic.DAY_NS - ist
    daily = [t[2] for t in strat.trace if t[0] == "comp" and t[2][2] == "d"]
    assert daily and daily[0][3] == day_end
    in_day = bars["ts_close"] <= day_end
    assert daily[0][8] == pytest.approx(float(bars["volume"][in_day].sum()))
    assert daily[0][4] == bars["open"][0]


def test_declaring_the_wrong_convention_is_visible(ds):
    """Open-stamped bars declared as close shift every fill by one bar."""
    sym = next(iter(ds.bars["1m"]))
    b = _series(ds.bars["1m"][sym], "open")
    wrong, right = _Det(), _Det()
    raptorbt.run_strategy_backtest(
        wrong, **b, config=_config("same_bar_close"), bar_stamp="close"
    )
    raptorbt.run_strategy_backtest(
        right, **b, config=_config("same_bar_close"), bar_stamp="open", bar_ns=60e9
    )
    assert wrong.trace != right.trace


def test_missing_stamp_is_refused_and_names_the_parameter(ds, portfolio_runner):
    sym = next(iter(ds.bars["1m"]))
    b = _series(ds.bars["1m"][sym], "close")
    with pytest.raises(InvalidParameter, match="bar_stamp is required") as err:
        raptorbt.run_strategy_backtest(_Det(), **b, config=_config("same_bar_close"))
    assert (err.value.code, err.value.parameter, err.value.valid) == (
        "missing_bar_stamp", "bar_stamp", ["open", "close"],
    )
    assert err.value.operation == "run_strategy_backtest"
    with pytest.raises(InvalidParameter, match="bar_stamp is required") as err:
        portfolio_runner(_Det(), {sym: b}, config=_config("same_bar_close"))
    assert err.value.code == "missing_bar_stamp"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raptorbt.run_strategy_backtest(
            _Det(), **b, config=_config("same_bar_close"), bar_stamp="close"
        )


BAD = [
    ({"bar_stamp": "mid"}, "'open', 'close'"),
    ({"bar_stamp": "open"}, "bar_ns"),
    ({"bar_stamp": "open", "bar_ns": 0}, "positive"),
    ({"bar_stamp": "open", "bar_ns": -60e9}, "positive"),
    ({"bar_stamp": "open", "bar_ns": float("nan")}, "positive"),
    ({"bar_stamp": "open", "bar_ns": 1.5}, "whole number"),
]


@pytest.mark.parametrize("kwargs,match", BAD)
def test_invalid_stamp_arguments_raise(ds, kwargs, match, portfolio_runner):
    sym = next(iter(ds.bars["1m"]))
    b = _series(ds.bars["1m"][sym], "open")
    with pytest.raises(ValueError, match=match):
        raptorbt.run_strategy_backtest(_Det(), **b, **kwargs)
    with pytest.raises(ValueError, match=match):
        portfolio_runner(_Det(), {sym: b}, **kwargs)


def test_bar_ns_dict_must_cover_every_symbol(ds, portfolio_runner):
    data = {s: _series(b, "open") for s, b in ds.bars["1m"].items()}
    syms = list(data)
    with pytest.raises(ValueError, match=repr(syms[-1])):
        portfolio_runner(
            _Det(), data, bar_stamp="open", bar_ns={s: 60e9 for s in syms[:-1]}
        )


def test_per_symbol_widths(ds, portfolio_runner):
    """Two symbols on different bar widths still share one close-time clock."""
    sym = next(iter(ds.bars["1m"]))
    b1 = _series(ds.bars["1m"][sym], "open")
    # Re-stamp the same bars as 2-minute bars: ``stamp + 2m`` is a close time.
    ts2 = b1["timestamps"]
    strat = _Det()
    portfolio_runner(
        strat, {"A": b1, "B": {**b1, "timestamps": ts2}}, config=_config("same_bar_close"),
        bar_stamp="open", bar_ns={"A": 60e9, "B": 120e9},
    )
    vis = {(s, v - r) for s, r, v in strat.raw}
    assert vis == {("A", MIN_NS), ("B", 2 * MIN_NS)}


def test_tick_stream_bar_stamp(ds):
    sym = next(iter(ds.bars["1m"]))
    b = ds.bars["1m"][sym]
    n = 40

    def warm(stamp):
        return {sym: {k: v[:n] for k, v in _series(b, stamp).items()}}

    def run(stamp):
        strat = _Det()
        stream = TickStrategyStream(
            strat, [sym], config=_config("same_bar_close"),
            warmup_bars=warm(stamp), **_stamp_kwargs(stamp),
        )
        nxt = _series(b, stamp)
        stream.push_bar(
            sym, int(nxt["timestamps"][n]), *(float(nxt[k][n]) for k in
            ("open", "high", "low", "close", "volume")),
        )
        return strat

    a, c = run("open"), run("close")
    assert a.trace == c.trace
    assert [(r, v) for _, r, v in a.raw] != [(r, v) for _, r, v in c.raw]
    assert all(v - r == MIN_NS for _, r, v in a.raw)

    with pytest.raises(InvalidParameter, match="TickStrategyStream.*bar_stamp is required"):
        TickStrategyStream(_Det(), [sym], warmup_bars=warm("close"))
    pushed = TickStrategyStream(_Det(), [sym])  # no bars supplied, nothing to declare yet
    with pytest.raises(InvalidParameter, match="bar_stamp is required"):
        pushed.push_bar(sym, 60 * 10**9, 1.0, 2.0, 0.5, 1.5, 10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TickStrategyStream(_Det(), [sym])  # no bars supplied, nothing to declare
    with pytest.raises(ValueError, match="bar_stamp"):
        TickStrategyStream(_Det(), [sym], bar_stamp="mid")
