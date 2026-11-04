"""Golden fixtures for the event-driven strategy runners.

Pins, bit-for-bit, what ``run_portfolio_strategy``, ``run_tick_strategy`` and
``TickStrategyStream`` do today: every hook a strategy sees (with the account,
positions, working orders, indicators and cache it could read at that moment),
the trades, orders, curves, metrics and halt state they produce.

Regenerate from the repo root, after building the extension::

    .venv/bin/python tests/python/golden/runner_fixtures.py

Regeneration is legitimate only for an *intended* behaviour change of one of
those runners, and the commit must record the reason. A diff you did not
intend is a regression, not a fixture to refresh.

Inputs are frozen next to the outputs (float hex), for the reason given in
``generate.py``: NumPy kernels are not correctly rounded, so rebuilding the
data at test time would measure NumPy as much as the engine. The test replays
the frozen arrays; only this generator touches ``tests/python/synthetic.py``.

What is deliberately NOT recorded: ``ctx.series()`` and the other full-array
context attributes (``ctx.timestamps``, ``ctx.close``, ...) were the full
arrays on the old per-event loops and are truncated to the visibility
watermark on the runtime. The fixtures record only what a strategy can see
without lookahead. The scripted strategies never read them either.
"""

from __future__ import annotations

import functools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import raptorbt
from raptorbt import (
    BacktestConfig,
    Indicator,
    InstrumentConfig,
    LiveNode,
    Strategy,
    TickStrategyStream,
    run_portfolio_strategy,
    run_strategies,
    run_tick_strategy,
)
from raptorbt.strategy import orders

HERE = Path(__file__).parent
FIXTURE_PATH = HERE / "runner_fixtures.json"
SYNTHETIC_DIR = HERE.parent

NS = 1_000_000_000
MIN_NS = 60 * NS

# ---------------------------------------------------------------------------
# Exact encoding
# ---------------------------------------------------------------------------


def _hex(x: float) -> str:
    """``float.hex`` without trailing mantissa zeros: shorter, still exact
    (``float.fromhex`` reads it back to the same bits, NaN and -0.0 included)."""
    h = float.hex(x)
    if "p" not in h:  # nan, inf, -inf
        return h
    mant, exp = h.split("p")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"{mant}p{exp}"


def enc(x):
    """JSON-safe, bit-exact image of a value: floats become hex strings."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _hex(float(x))
    if isinstance(x, np.ndarray):
        return [enc(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    raise TypeError(f"cannot encode {type(x).__name__}: {x!r}")


def dec_f(arr):
    return np.array([float.fromhex(v) for v in arr], dtype=np.float64)


def _fields(obj, names):
    return {n: enc(getattr(obj, n)) for n in names}


TRADE_FIELDS = (
    "id symbol entry_idx exit_idx entry_price exit_price size direction pnl "
    "return_pct entry_time exit_time fees entry_fees exit_fees fee_breakdown "
    "exit_reason mae_price mfe_price mae_pnl mfe_pnl"
).split()
ORDER_FIELDS = (
    "id client_id symbol side kind tif status submitted_idx submitted_ts "
    "requested_qty filled_qty avg_fill_price last_fill_idx fill_slices "
    "limit_price trigger_price reject_reason parent_id oco_group"
).split()
EVENT_FIELDS = (
    "kind idx price size direction reject_reason order_id client_order_id"
).split()
POSITION_FIELDS = (
    "position_id entry_idx entry_price size direction stop_price target_price"
).split()


def enc_trade(t):
    return _fields(t, TRADE_FIELDS)


def enc_event(ev):
    """An engine event as a value list in ``EVENT_FIELDS`` order, then the
    trade (``TRADE_FIELDS`` order) or None. ``label_event`` names the fields."""
    trade = None if ev.trade is None else [enc(getattr(ev.trade, n)) for n in TRADE_FIELDS]
    return [enc(getattr(ev, n)) for n in EVENT_FIELDS] + [trade]


def label_event(values):
    """Inverse of :func:`enc_event`'s shape, for readable diffs."""
    out = dict(zip(EVENT_FIELDS, values))
    trade = values[len(EVENT_FIELDS)]
    out["trade"] = None if trade is None else dict(zip(TRADE_FIELDS, trade))
    return out


def enc_position(p):
    return [enc(getattr(p, n)) for n in POSITION_FIELDS]


# ---------------------------------------------------------------------------
# The recording strategy
# ---------------------------------------------------------------------------

_PAYLOAD_HOOKS = (
    "on_time_event on_trade_tick on_quote on_order_book on_composite_bar "
    "on_order_filled on_order_rejected on_order_accepted on_order_triggered "
    "on_order_canceled on_order_expired on_order_event on_algo_started "
    "on_algo_completed on_margin_call on_position_opened on_position_closed"
).split()
_BARE_HOOKS = "on_start on_bar on_stop".split()
_MISSING = object()
# Keys a hook entry carries only when they changed since the previous entry.
STATE_KEYS = (
    "bar bar_close_ts book best_bid best_ask last_price equity cash free_capital "
    "positions open_orders indicators indicators_ready cache clock"
).split()


def expand_hooks(hooks):
    """Delta-encoded hook log -> one full record per hook."""
    state: dict = {}
    orders: dict = {}
    out = []
    for h in hooks:
        state = {**state, **{k: h[k] for k in STATE_KEYS if k in h}}
        for cid, v in h.get("cache_orders", ()):
            orders[cid] = v
        rest = {k: v for k, v in h.items() if k not in STATE_KEYS and k != "cache_orders"}
        out.append({**rest, **state, "cache_orders": dict(orders)})
    return out


def _enc_payload(hook, payload):
    if payload is None:
        return None
    if hook in ("on_trade_tick", "on_quote", "on_composite_bar", "on_time_event"):
        return enc(tuple(payload))
    if hook == "on_order_book":
        return enc((payload.timestamp, payload.bids, payload.asks, payload.symbol))
    return enc_event(payload)


class Recorder(Strategy):
    """Logs every hook with the state visible at that instant, then lets a
    script act. Behaviour lives in ``script`` (a plain function), so one class
    serves every scenario."""

    def __init__(self, script=None):
        super().__init__()
        self.script = script
        self.hooks: list[dict] = []
        self.count: dict = {}
        self.last_ids: list[str] = []
        self.queued: set = set()
        self._state: dict = {}
        self._records: dict = {}
        # Whether the context has the tick surface (book, best bid/ask, ...)
        # to record: tick and stream runs do, portfolio runs do not.
        self.tick_surface = False

    # -- helpers for scripts ------------------------------------------------

    def tick(self, ctx, hook):
        """0-based ordinal of this call of ``hook`` for ``ctx.symbol``."""
        key = (hook, ctx.symbol)
        n = self.count.get(key, 0)
        self.count[key] = n + 1
        return n

    def submit(self, order, **kw):
        cid = self.submit_order(order, **kw)
        self.last_ids.append(cid)
        return cid

    # -- recording ----------------------------------------------------------

    def _record(self, hook, ctx, payload):
        e = {"hook": hook, "symbol": ctx.symbol, "idx": ctx.idx}
        p = _enc_payload(hook, payload)
        if p is not None:
            e["payload"] = p
        st = {}
        bar = ctx.bar
        st["bar"] = enc(tuple(bar)) if bar is not None else None
        st["bar_close_ts"] = ctx.bar_close_ts
        if self.tick_surface:
            b = ctx.book
            st["book"] = (
                enc((b.timestamp, len(b.bids), len(b.asks), b.best_bid, b.best_ask)) if b is not None else None
            )
            for name in ("best_bid", "best_ask", "last_price"):
                st[name] = enc(getattr(ctx, name))
        st["equity"] = enc(ctx.equity)
        st["cash"] = enc(ctx.cash)
        st["free_capital"] = enc(ctx.free_capital)
        positions = {}
        working = {}
        for s in ctx.symbols:
            ps = ctx.positions(s)
            if ps:
                positions[s] = [enc_position(p) for p in ps]
            oo = ctx.open_orders(s)
            if oo:
                working[s] = [enc(tuple(o)[:-1]) for o in oo]
        st["positions"] = positions
        st["open_orders"] = working
        if self._indicators:
            st["indicators"] = [
                [enc(ind.value), bool(ind.initialized), int(ind.update_count)]
                for ind, _, _ in self._indicators
            ]
            st["indicators_ready"] = self.indicators_initialized()
        cache = self.cache
        st["cache"] = [
            len(cache.orders()),
            len(cache.orders_open()),
            len(cache.closed_trades()),
            enc(cache.realized_pnl()),
        ]
        # Order records, as the ones that changed since the previous entry.
        records = {
            o.client_id: enc((o.status, o.symbol, o.last_price, o.last_size, o.reject_reason))
            for o in cache.orders()
        }
        changed = [[cid, v] for cid, v in records.items() if self._records.get(cid) != v]
        if changed:
            e["cache_orders"] = changed
        self._records = records
        st["clock"] = [self.clock.timestamp_ns(), self.clock.timer_names()]
        # Delta-encode the visible state: a key is stored only when it differs
        # from the previous entry. Equality of the delta sequences is equality
        # of the full sequences, and the log stays small enough to check in.
        for k, v in st.items():
            if self._state.get(k, _MISSING) != v:
                e[k] = v
        self._state = st
        self.hooks.append(e)

    def _fire(self, hook, ctx, payload):
        self._record(hook, ctx, payload)
        if self.script is not None:
            self.script(self, hook, ctx, payload)


def _install_hooks():
    for name in _BARE_HOOKS:
        def bare(self, ctx, _n=name):
            self._fire(_n, ctx, None)
        setattr(Recorder, name, bare)
    for name in _PAYLOAD_HOOKS:
        def with_payload(self, ctx, payload, _n=name):
            self._fire(_n, ctx, payload)
        setattr(Recorder, name, with_payload)


_install_hooks()


# ---------------------------------------------------------------------------
# Scripts: deterministic behaviours, selected by name from the scenario
# ---------------------------------------------------------------------------


def _price_of(ctx):
    """Reference price for pricing orders: the last print, else the bar close."""
    lp = getattr(ctx, "last_price", None)
    if lp is not None:
        return lp
    return ctx._bar.close


def _indicator(kind, period):
    return getattr(Indicator, kind)(period)


def s_legacy(
    rec, hook, ctx, payload, *, drive, every=9, enter_at=2, exit_at=6, frac=0.3,
    sided=None, stops=False, extra=(), subs=(), inds=(), comp_inds=(),
    close_other=None, unrouted=(), after_close=None,
):
    """Legacy intents (enter/close_position) on an ordinal rhythm."""
    sided = sided or {}
    if hook == "on_start":
        for step, unit in subs:
            rec.subscribe_bars(step, unit)
        for kind, period in inds:
            for s in ctx.symbols:
                rec.register_indicator(_indicator(kind, period), symbol=s)
        for kind, period in unrouted:
            rec.register_indicator(_indicator(kind, period))
        for sid, kind, period in comp_inds:
            for s in ctx.symbols:
                rec.register_indicator(_indicator(kind, period), stream_id=sid, symbol=s)
        return

    if hook == "on_position_closed" and after_close:
        # Work queued from a post-step event hook: a command is routed to the
        # event's instrument at once; an intent is held for the next step.
        if after_close in ("order", "both"):
            rec.submit(orders.Limit(side="buy", price=_price_of(ctx) * 0.999, size_frac=0.05))
        if after_close in ("intent", "both") and ctx.symbol not in rec.queued:
            rec.queued.add(ctx.symbol)
            rec.enter(size_frac=frac)
        return

    def enter():
        side = sided.get(ctx.symbol)
        ref = _price_of(ctx)
        if stops and side != "short":
            rec.enter(size_frac=frac, stop_price=ref * 0.995, target_price=ref * 1.006)
        elif stops:
            rec.enter_short(size_frac=frac, stop_price=ref * 1.005, target_price=ref * 0.994)
        elif side == "long":
            rec.enter_long(size_frac=frac)
        elif side == "short":
            rec.enter_short(size_frac=frac)
        else:
            rec.enter(size_frac=frac)

    for ehook, eevery, eat in extra:
        if hook == ehook:
            n = rec.tick(ctx, "extra:" + ehook)
            if n % eevery == eat and ctx.is_flat and ctx.symbol not in rec.queued:
                rec.queued.add(ctx.symbol)
                enter()
    if hook != drive:
        return
    n = rec.tick(ctx, "drive")
    r = n % every
    held = ctx.symbol in rec.queued  # an intent from another hook rides this step
    rec.queued.discard(ctx.symbol)
    if held:
        return
    if r == enter_at and ctx.is_flat:
        enter()
    elif r == exit_at and not ctx.is_flat:
        rec.close_position()
    elif close_other and r == close_other[0]:
        other = close_other[1]
        if ctx.position_for(other) is not None:
            rec.close_position(symbol=other)


def s_hedge(rec, hook, ctx, payload, *, drive, every=6, frac=0.12):
    """Several positions per symbol: enter, hedge short, close by position id."""
    if hook != drive:
        return
    n = rec.tick(ctx, "drive")
    r = n % every
    ref = _price_of(ctx)
    if r in (1, 3):
        rec.enter(size_frac=frac, stop_price=ref * 0.994, target_price=ref * 1.008)
    elif r == 4:
        rec.enter_short(size_frac=frac / 2)
    elif r == 5:
        ps = ctx.positions()
        if ps:
            rec.close_position(position_id=ps[0].position_id)


def _act_limit_buy(rec, ctx, ref, ts):
    rec.submit(orders.Limit(side="buy", price=ref * 0.9996, size_frac=0.15))


def _act_limit_far(rec, ctx, ref, ts):
    rec.submit(orders.Limit(side="buy", price=ref * 0.97, size_frac=0.1))


def _act_stop_buy(rec, ctx, ref, ts):
    rec.submit(orders.StopMarket(side="buy", trigger=ref * 1.0004, size_frac=0.15))


def _act_stop_limit_buy(rec, ctx, ref, ts):
    rec.submit(orders.StopLimit(side="buy", trigger=ref * 1.0003, price=ref * 1.0007, size_frac=0.15))


def _act_mit_buy(rec, ctx, ref, ts):
    rec.submit(orders.MarketIfTouched(side="buy", trigger=ref * 0.9997, size_frac=0.1))


def _act_lit_buy(rec, ctx, ref, ts):
    rec.submit(orders.LimitIfTouched(side="buy", trigger=ref * 0.9997, price=ref * 0.9995, size_frac=0.1))


def _act_mtl_buy(rec, ctx, ref, ts):
    rec.submit(orders.MarketToLimit(side="buy", size_frac=0.1))


def _act_market_buy(rec, ctx, ref, ts):
    rec.submit(orders.Market(side="buy", size_frac=0.2))


def _act_market_attach(rec, ctx, ref, ts):
    rec.submit(orders.Market(side="buy", size_frac=0.2, stop_price=ref * 0.996, target_price=ref * 1.004))


def _act_market_sell(rec, ctx, ref, ts):
    rec.submit(orders.Market(side="sell", size_frac=0.2))


def _act_ioc_limit(rec, ctx, ref, ts):
    rec.submit(orders.Limit(side="buy", price=ref * 0.9999, size_frac=0.1, tif="ioc"))


def _act_gtd_limit(rec, ctx, ref, ts):
    rec.submit(orders.Limit(side="buy", price=ref * 0.995, size_frac=0.1, tif="gtd", expire_ns=ts + 90 * NS))


def _act_day_limit(rec, ctx, ref, ts):
    rec.submit(orders.Limit(side="buy", price=ref * 0.996, size_frac=0.1, tif="day"))


def _act_post_only(rec, ctx, ref, ts):
    rec.submit(orders.Limit(side="buy", price=ref * 1.01, size_frac=0.1, post_only=True))


def _act_limit_sell_reduce(rec, ctx, ref, ts):
    if not ctx.is_flat:
        rec.submit(orders.Limit(side="sell", price=ref * 1.0005, reduce_only=True))


def _act_trail(rec, ctx, ref, ts):
    if not ctx.is_flat:
        rec.submit(orders.TrailingStopMarket(side="sell", offset=ref * 0.0006, reduce_only=True))


def _act_trail_bps(rec, ctx, ref, ts):
    if not ctx.is_flat:
        rec.submit(orders.TrailingStopMarket(side="sell", offset=8.0, offset_kind="bps", reduce_only=True))


def _act_trail_limit(rec, ctx, ref, ts):
    if not ctx.is_flat:
        rec.submit(
            orders.TrailingStopLimit(
                side="sell", offset=ref * 0.0006, limit_offset=ref * 0.0002, reduce_only=True
            )
        )


def _act_bracket(rec, ctx, ref, ts):
    if ctx.is_flat:
        rec.submit_bracket(
            orders.Market(side="buy", size_frac=0.25), stop_trigger=ref * 0.9992, target_price=ref * 1.0008
        )


def _act_bracket_limit(rec, ctx, ref, ts):
    if ctx.is_flat:
        rec.submit_bracket(
            orders.Market(side="buy", size_frac=0.25),
            stop_trigger=ref * 0.9992,
            target_price=ref * 1.0008,
            stop_limit_price=ref * 0.9990,
        )


def _act_oco_exit(rec, ctx, ref, ts):
    if not ctx.is_flat:
        a = rec.submit(orders.Limit(side="sell", price=ref * 1.0006, reduce_only=True))
        b = rec.submit(orders.StopMarket(side="sell", trigger=ref * 0.9994, reduce_only=True))
        rec.link_oco(a, b)


def _act_cancel_last(rec, ctx, ref, ts):
    if rec.last_ids:
        rec.cancel_order(rec.last_ids.pop())


def _act_cancel_all(rec, ctx, ref, ts):
    rec.cancel_all_orders()


def _act_modify_last(rec, ctx, ref, ts):
    if rec.last_ids:
        rec.modify_order(rec.last_ids[-1], limit_price=ref * 0.9998)


def _act_close_all(rec, ctx, ref, ts):
    rec.close_position(symbol=ctx.symbol)


ACTIONS = {
    name[len("_act_"):]: fn for name, fn in globals().items() if name.startswith("_act_")
}


def s_typed(rec, hook, ctx, payload, *, drive, program):
    """Typed orders on a (symbol index, ordinal, action) program."""
    if hook != drive:
        return
    n = rec.tick(ctx, "drive")
    sym_idx = ctx.symbols.index(ctx.symbol)
    ts = ctx.bar_close_ts if ctx.bar_close_ts is not None else payload.timestamp
    for s, ordinal, action in program:
        if s == sym_idx and ordinal == n:
            ACTIONS[action](rec, ctx, _price_of(ctx), ts)


def s_twap(rec, hook, ctx, payload, *, drive, at=2):
    if hook != drive:
        return
    if rec.tick(ctx, "drive") == at:
        rec.submit(orders.Twap(side="buy", units=3.0, slices=3, every=MIN_NS))


def s_timers(rec, hook, ctx, payload, *, drive, alert_ns, timer_ns, timer_start_ns, stop_ns, frac=0.2):
    """Alerts and timers set in on_start (cloned per symbol) and in handlers.

    A due alert queues an intent (held for the symbol's next step) and a typed
    order (a command, routed to the instrument at once)."""
    if hook == "on_start":
        rec.clock.set_time_alert("alert_a", alert_ns)
        rec.clock.set_timer("tick_1", timer_ns, start_ns=timer_start_ns)
        rec.clock.set_timer("tick_2", timer_ns * 2, start_ns=alert_ns, stop_ns=stop_ns)
        return
    if hook == "on_time_event":
        if payload.name == "alert_a":
            rec.clock.set_time_alert("alert_b", payload.ts_fired + timer_ns)
            if ctx.is_flat and ctx.symbol not in rec.queued:
                rec.queued.add(ctx.symbol)
                rec.enter(size_frac=frac)
            rec.submit(orders.Market(side="buy", size_frac=0.05))
        elif payload.name == "alert_b":
            rec.submit(orders.Market(side="sell", size_frac=0.1))
        elif payload.name == "tick_1" and rec.tick(ctx, "tick_1") == 2:
            rec.clock.cancel_timer("tick_2")
        return
    if hook == drive:
        rec.queued.discard(ctx.symbol)
        if rec.tick(ctx, "drive") % 11 == 7 and not ctx.is_flat:
            rec.close_position()


def s_allin(rec, hook, ctx, payload, *, drive, frac=1.0, short=False):
    """Fully committed entries after every flat, to drive a halt."""
    if hook != drive:
        return
    n = rec.tick(ctx, "drive")
    if n >= 1 and ctx.is_flat:
        if short:
            rec.enter_short(size_frac=frac)
        else:
            rec.enter(size_frac=frac)


SCRIPTS = {
    "legacy": s_legacy,
    "hedge": s_hedge,
    "typed": s_typed,
    "twap": s_twap,
    "timers": s_timers,
    "allin": s_allin,
}


def make_script(spec):
    if spec is None:
        return None
    fn = SCRIPTS[spec["name"]]
    args = {k: (tuple(map(tuple, v)) if k in ("subs", "inds", "comp_inds", "extra", "unrouted") else v)
            for k, v in spec.get("args", {}).items()}
    if "program" in args:
        args["program"] = tuple(map(tuple, args["program"]))
    if "close_other" in args and args["close_other"] is not None:
        args["close_other"] = tuple(args["close_other"])
    return functools.partial(_call_script, fn, args)


def _call_script(fn, args, rec, hook, ctx, payload):
    fn(rec, hook, ctx, payload, **args)


# ---------------------------------------------------------------------------
# Data: built once from the synthetic generator, then frozen
# ---------------------------------------------------------------------------

BAR_KEYS = ("ts_open", "ts_close", "open", "high", "low", "close", "volume")
TICK_KEYS = (
    "timestamps ltp bid ask buy_qty_delta sell_qty_delta ltq bid_qty ask_qty oi"
).split()
DEPTH_KEYS = ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes")


def _freeze_arrays(d):
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        if v.dtype == np.int64:
            out[k] = [int(x) for x in v.tolist()]
        else:
            out[k] = enc(v)
    return out


def _thaw_arrays(d):
    out = {}
    for k, v in d.items():
        if k in ("timestamps", "ts_open", "ts_close"):
            out[k] = np.array(v, dtype=np.int64)
        elif k in DEPTH_KEYS:
            out[k] = np.array([[float.fromhex(x) for x in row] for row in v], dtype=np.float64)
        else:
            out[k] = dec_f(v)
    return out


def _crash_bars(n=40, step_ns=MIN_NS, start=1_700_000_000 * NS, fall=2.2):
    """Hand-built bars (no NumPy transcendentals): a rally, then a long fall."""
    out = {k: [] for k in ("ts_open", "ts_close", "open", "high", "low", "close", "volume")}
    price = 100.0
    for i in range(n):
        if i < 10:
            nxt = price + 0.8
        else:
            nxt = price - fall - 0.05 * (i - 10)
        out["ts_open"].append(start + i * step_ns)
        out["ts_close"].append(start + (i + 1) * step_ns)
        out["open"].append(price)
        out["high"].append(max(price, nxt) + 0.4)
        out["low"].append(min(price, nxt) - 0.4)
        out["close"].append(nxt)
        out["volume"].append(1000.0 + 10 * i)
        price = nxt
    return {k: np.array(v, dtype=np.int64 if k.startswith("ts") else np.float64) for k, v in out.items()}


def build_datasets():
    """The synthetic-derived inputs, as frozen JSON-safe dicts."""
    sys.path.insert(0, str(SYNTHETIC_DIR))
    import synthetic  # noqa: E402

    ds = synthetic.generate(n_instruments=3, days=2, avg_ticks_per_sec=0.1, timeframes=("1m",))
    syms = [i.symbol for i in ds.instruments]
    tick_size = {i.symbol: i.tick_size for i in ds.instruments}

    bars, bars_day2 = {}, {}
    ticks_day1, ticks_day2 = {}, {}
    for s in syms:
        b = ds.bars["1m"][s]
        to = np.asarray(b["ts_open"])
        split = int(np.searchsorted(to, to[0] + 24 * 3600 * NS // 2))
        pick = {k: np.asarray(b[k]) for k in BAR_KEYS}
        head = {k: v[split : split + 45] for k, v in pick.items()}
        tail = {k: v[split - 30 : split] for k, v in pick.items()}
        bars[s] = {k: np.concatenate([tail[k], head[k]]) for k in BAR_KEYS}
        bars_day2[s] = tail  # warmup history for the stream (day 1 tail)

        t = ds.ticks[s]
        ts = np.asarray(t["timestamps"])
        d1 = int(np.searchsorted(ts, ts[0] + 24 * 3600 * NS // 2))
        end1 = int(np.searchsorted(ts, ts[0] + 6 * MIN_NS))
        ticks_day1[s] = {k: np.asarray(t[k])[: min(end1, 100)].copy() for k in TICK_KEYS}
        ticks_day2[s] = {
            k: np.asarray(t[k])[d1 : d1 + 60].copy() for k in TICK_KEYS
        }

    # Shape the feed: quote-only rows (no print), print-only rows (no book),
    # and equal timestamps across symbols.
    for feed in (ticks_day1, ticks_day2):
        for s in syms:
            a = feed[s]
            a["ltp"][7] = 0.0  # a row with a book but no print
            for j in (12, 13):  # prints with no book
                a["bid"][j] = 0.0
                a["ask"][j] = 0.0
        t1, t2 = feed[syms[1]]["timestamps"], feed[syms[2]]["timestamps"]
        for k in (10, 20, 30):
            for j in range(len(t1)):
                if t2[k - 1] < t1[j] < t2[k + 1]:
                    t2[k] = t1[j]
                    break

    depth = {}
    for s in syms:
        a = ticks_day1[s]
        ts, n = a["timestamps"], len(a["timestamps"])
        rows_ts, bp, bs, ap, asz = [], [], [], [], []
        for k in range(0, n - 1, 3):
            if a["bid"][k] <= 0 or a["ask"][k] <= 0:
                continue
            t = int(ts[k]) if k % 6 == 0 else int(ts[k] + (ts[k + 1] - ts[k]) // 2)
            if rows_ts and t <= rows_ts[-1]:
                continue
            ts_ = tick_size[s]
            rows_ts.append(t)
            bp.append([a["bid"][k] - i * ts_ for i in range(3)])
            bs.append([a["bid_qty"][k] * (1.0 + 0.5 * i) for i in range(3)])
            ap.append([a["ask"][k] + i * ts_ for i in range(3)])
            asz.append([a["ask_qty"][k] * (1.0 + 0.5 * i) for i in range(3)])
        depth[s] = {
            "timestamps": np.array(rows_ts, dtype=np.int64),
            "bid_prices": np.array(bp), "bid_sizes": np.array(bs),
            "ask_prices": np.array(ap), "ask_sizes": np.array(asz),
        }

    # Stream feed: day-2 ticks, refused rows, bars and depth in time order.
    ops = _build_stream_ops(syms, ticks_day2, bars, tick_size)

    return {
        "symbols": syms,
        "bars": {s: _freeze_arrays(v) for s, v in bars.items()},
        "warmup": {s: _freeze_arrays(v) for s, v in bars_day2.items()},
        "ticks": {s: _freeze_arrays(v) for s, v in ticks_day1.items()},
        "depth": {s: _freeze_arrays(v) for s, v in depth.items()},
        "crash_bars": {"CRASH": _freeze_arrays(_crash_bars()), "CRASH2": _freeze_arrays(_crash_bars(fall=1.7))},
        "stream_ops": ops,
    }


def _build_stream_ops(syms, ticks, bars, tick_size):
    """A time-ordered live feed: prints, quote-only and priceless rows, bars
    (pushed at their close), and depth snapshots."""
    events = []  # (ts, order, op)
    for si, s in enumerate(syms):
        a = ticks[s]
        for k in range(len(a["timestamps"])):
            row = [a[key][k] for key in TICK_KEYS[1:]]
            events.append((int(a["timestamps"][k]), 1, si, k, ["tick", s, int(a["timestamps"][k])] + [enc(v) for v in row]))
        # Refused rows: nothing printed and no two-sided book (an index that
        # has not ticked) -- they only prove the clock reached their instant.
        for k in range(5, len(a["timestamps"]) - 1, 9):
            t = int(a["timestamps"][k]) + 1
            events.append((t, 2, si, k, ["tick", s, t] + [enc(0.0)] * 9))
        for k in range(2, len(a["timestamps"]) - 1, 11):
            t = int(a["timestamps"][k]) + 2
            if a["bid"][k] > 0 and a["ask"][k] > 0:
                ts_ = tick_size[s]
                bids = [[float(a["bid"][k] - i * ts_), float(a["bid_qty"][k] + 10 * i)] for i in range(3)]
                asks = [[float(a["ask"][k] + i * ts_), float(a["ask_qty"][k] + 10 * i)] for i in range(3)]
                events.append((t, 3, si, k, ["depth", s, t, enc(bids), enc(asks)]))
    events.sort(key=lambda e: e[:4])
    return [e[4] for e in events]


def thaw_ops(ops):
    out = []
    for op in ops:
        if op[0] == "tick":
            out.append(("tick", op[1], op[2], *[float.fromhex(v) for v in op[3:]]))
        elif op[0] == "depth":
            bids = [(float.fromhex(p), float.fromhex(q)) for p, q in op[3]]
            asks = [(float.fromhex(p), float.fromhex(q)) for p, q in op[4]]
            out.append(("depth", op[1], op[2], bids, asks))
        else:
            out.append(("bar", op[1], op[2], *[float.fromhex(v) for v in op[3:]]))
    return out


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

S0, S1, S2 = "SYN000", "SYN001", "SYN002"


def _zero_fee(**kw):
    return {"kwargs": {"initial_capital": 1_000_000.0, "fees": 0.0, **kw}}


def _default_fee(**kw):
    return {"kwargs": {"initial_capital": 1_000_000.0, **kw}}


def _bars_run(**kw):
    return {"bar_stamp": "open", "bar_ns": MIN_NS, **kw}


def scenario_table():
    """Every scenario as a JSON-able descriptor. Code is looked up by name;
    the descriptor is stored in the fixture so drift is detected."""
    P, T, ST = "portfolio", "tick", "stream"
    sc = []

    def add(name, runner, covers, **kw):
        sc.append({"name": name, "runner": runner, "covers": covers, **kw})

    bars_all = {"symbols": [S0, S1, S2], "n": 56}
    bars_40 = {"symbols": [S0, S1, S2], "n": 32}
    bars_2 = {"symbols": [S1, S2], "n": 32}
    comp = [[5, "m"], [15, "m"], [1, "d"]]

    # ---- portfolio runner: bars --------------------------------------------
    add("pf_composites_indicators", P,
        ["composite-before-on_bar", "ctx.idx", "indicators+readiness", "equity-once-per-ts", "bar_stamp open"],
        data=bars_all, config=_default_fee(), run=_bars_run(),
        script={"name": "legacy", "args": {
            "drive": "on_bar", "subs": comp, "inds": [["sma", 3], ["ema", 5]],
            "comp_inds": [[0, "sma", 2], [1, "ema", 2]], "every": 9, "enter_at": 2, "exit_at": 6, "frac": 0.3}})
    add("pf_fill_next_bar_open", P, ["fill_timing next_bar_open", "legacy intents", "cash netting", "intent queued in a post-step hook"],
        data=bars_40, config=_zero_fee(fill_timing="next_bar_open"), run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 7, "enter_at": 1, "exit_at": 4, "frac": 0.25,
                                           "after_close": "intent"}})
    add("pf_fill_same_bar_close", P, ["fill_timing same_bar_close", "legacy intents", "fees", "settle(): command and intent queued in a post-step hook"],
        data=bars_40, config=_default_fee(fill_timing="same_bar_close", slippage=0.0005), run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 7, "enter_at": 1, "exit_at": 4, "frac": 0.25,
                                           "after_close": "both"}})
    add("pf_fill_same_bar_open_lookahead", P, ["fill_timing same_bar_open_lookahead"],
        data=bars_2, config=_zero_fee(fill_timing="same_bar_open_lookahead"), run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 8, "enter_at": 2, "exit_at": 5}})
    add("pf_margin_netting_shorts", P, ["margin account", "netting", "short entries", "directions", "cross-symbol close"],
        data=bars_40, config=_default_fee(), run=_bars_run(account_type="margin", leverage=4.0, oms_type="netting",
                                                            directions={S1: -1}),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 8, "enter_at": 1, "exit_at": 5, "frac": 0.4,
                                           "sided": {S0: "long", S1: "short", S2: "short"},
                                           "close_other": [3, S0]}})
    add("pf_margin_hedging", P, ["margin account", "hedging", "close by position id", "attached stop/target"],
        data=bars_40, config=_zero_fee(), run=_bars_run(account_type="margin", leverage=3.0, oms_type="hedging"),
        script={"name": "hedge", "args": {"drive": "on_bar"}})
    add("pf_cash_hedging", P, ["cash account", "hedging", "entry rejection"],
        data=bars_2, config=_zero_fee(), run=_bars_run(account_type="cash", oms_type="hedging"),
        script={"name": "hedge", "args": {"drive": "on_bar", "frac": 0.3}})
    add("pf_stops_targets_legacy", P, ["legacy intents", "stop_price/target_price overrides", "config stops"],
        data=bars_40, config={"kwargs": {"initial_capital": 1_000_000.0, "fees": 0.0},
                              "calls": [["set_fixed_stop", [0.004]], ["set_fixed_target", [0.006]]]},
        run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 9, "enter_at": 2, "exit_at": 8, "stops": True}})
    add("pf_atr_stops", P, ["ATR stop", "ATR risk-reward target"],
        data=bars_all, config={"kwargs": {"initial_capital": 1_000_000.0, "fees": 0.0},
                               "calls": [["set_atr_stop", [1.5, 5]], ["set_risk_reward_target", [2.0]]]},
        run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 10, "enter_at": 1, "exit_at": 9}})
    add("pf_instrument_atr_stops", P, ["per-symbol ATR stop"],
        data=bars_2, config=_zero_fee(),
        instrument_configs={S1: {"calls": [["set_atr_stop", [1.0, 4]]]}},
        run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 10, "enter_at": 1, "exit_at": 9}})
    add("pf_typed_resting", P, ["limit", "stop", "stop-limit", "MIT", "LIT", "MTL", "tif", "cancel/modify", "order events"],
        data=bars_40, config=_zero_fee(), run=_bars_run(),
        script={"name": "typed", "args": {"drive": "on_bar", "program": [
            [0, 1, "limit_buy"], [0, 3, "modify_last"], [0, 5, "market_sell"], [0, 7, "stop_buy"], [0, 10, "stop_limit_buy"],
            [0, 14, "mit_buy"], [0, 17, "lit_buy"], [0, 20, "cancel_all"], [0, 22, "mtl_buy"],
            [1, 2, "limit_far"], [1, 4, "cancel_last"], [1, 6, "ioc_limit"], [1, 9, "gtd_limit"], [1, 12, "day_limit"],
            [1, 15, "market_attach"], [1, 25, "close_all"], [1, 27, "post_only"],
            [2, 3, "market_buy"], [2, 6, "limit_sell_reduce"], [2, 12, "market_sell"], [2, 14, "stop_buy"],
            [2, 18, "limit_buy"], [2, 24, "market_buy"], [2, 30, "close_all"]]}})
    add("pf_typed_exits", P, ["bracket", "bracket stop-limit", "OCO", "trailing market", "trailing bps", "trailing limit",
                              "reduce_only", "parent/child"],
        data=bars_all, config=_zero_fee(), run=_bars_run(),
        script={"name": "typed", "args": {"drive": "on_bar", "program": [
            [0, 2, "bracket"], [0, 20, "bracket_limit"], [0, 45, "market_buy"], [0, 47, "trail"],
            [1, 3, "market_buy"], [1, 4, "oco_exit"], [1, 25, "market_buy"], [1, 26, "trail_bps"],
            [1, 50, "market_buy"], [1, 51, "trail_limit"],
            [2, 5, "market_buy"], [2, 6, "limit_sell_reduce"], [2, 30, "bracket"]]}})
    add("pf_timers", P, ["alerts and timers cloned per symbol", "time events before on_bar", "intent from on_time_event"],
        data=bars_40, config=_zero_fee(), run=_bars_run(),
        script={"name": "timers", "args": {"drive": "on_bar", "alert_ns": 0, "timer_ns": 7 * MIN_NS, "timer_start_ns": 0, "stop_ns": 0}},
        timers_from_data=True)
    add("pf_twap", P, ["TWAP slices", "on_algo_started", "on_algo_completed"],
        data=bars_2, config=_zero_fee(), run=_bars_run(oms_type="hedging"),
        script={"name": "twap", "args": {"drive": "on_bar", "at": 2}})
    add("pf_max_positions_rejections", P, ["max_positions", "rejected_entries", "on_order_rejected"],
        data=bars_all, config=_zero_fee(max_positions=1), run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 6, "enter_at": 1, "exit_at": 4, "frac": 0.2}})
    add("pf_drawdown_halt", P, ["max_drawdown_pct", "halted", "halted_at"],
        data={"symbols": ["CRASH", "CRASH2"], "n": 40, "source": "crash_bars"},
        config=_zero_fee(max_drawdown_pct=15.0), run={"bar_stamp": "close"},
        script={"name": "allin", "args": {"drive": "on_bar", "frac": 0.45}})
    add("pf_margin_call_halt", P, ["margin call", "on_margin_call", "halted", "halted_at"],
        data={"symbols": ["CRASH", "CRASH2"], "n": 40, "source": "crash_bars"},
        config=_zero_fee(), run={"bar_stamp": "close", "account_type": "margin", "leverage": 8.0},
        script={"name": "allin", "args": {"drive": "on_bar", "frac": 0.9}})
    add("pf_margin_call_liquidate", P, ["margin call", "liquidate_on_margin_call"],
        data={"symbols": ["CRASH", "CRASH2"], "n": 40, "source": "crash_bars"},
        config=_zero_fee(liquidate_on_margin_call=True),
        run={"bar_stamp": "close", "account_type": "margin", "leverage": 8.0},
        script={"name": "allin", "args": {"drive": "on_bar", "frac": 0.9}})
    add("pf_bar_stamp_close", P, ["bar_stamp close", "composite stamps"],
        data={**bars_all, "n": 46, "stamp": "close"}, config=_zero_fee(), run={"bar_stamp": "close"},
        script={"name": "legacy", "args": {"drive": "on_bar", "subs": [[5, "m"]], "inds": [["sma", 3]],
                                           "comp_inds": [[0, "sma", 2]], "every": 9, "enter_at": 2, "exit_at": 6}})
    add("pf_bar_stamp_dict_tz", P, ["bar_ns per symbol", "session_tz_offset_ns composites"],
        data={**bars_all, "n": 46}, config=_zero_fee(session_tz_offset_ns=19_800 * NS),
        run={"bar_stamp": "open", "bar_ns": {S0: MIN_NS, S1: MIN_NS, S2: MIN_NS}},
        script={"name": "legacy", "args": {"drive": "on_bar", "subs": [[1, "d"], [15, "m"]], "every": 9, "enter_at": 2, "exit_at": 6}})
    add("pf_unrouted_indicator_warning", P, ["unrouted indicator warning"],
        data=bars_2, config=_zero_fee(), run=_bars_run(),
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 9, "enter_at": 2, "exit_at": 6,
                                           "unrouted": [["sma", 3]]}})

    # ---- tick runner -------------------------------------------------------
    t2 = {"symbols": [S1, S2], "n": 34}
    t3 = {"symbols": [S0, S1, S2], "n": 30}
    t_long = {"symbols": [S1, S2], "n": 70}
    depth = [S0, S1, S2]
    add("tk_dispatch_order", T,
        ["trade/quote/book/bar dispatch order", "trade before quote in a row", "ctx.idx event ordinal",
         "quote/book intents held per symbol", "best_bid pre-print", "quote-only and print-only rows",
         "equal timestamps across symbols"],
        data=t3, depth=depth, config=_zero_fee(), run={},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 10, "enter_at": 3, "exit_at": 8,
                                           "frac": 0.2, "extra": [["on_quote", 6, 2], ["on_order_book", 3, 1]]}})
    add("tk_primary_bars", T,
        ["primary on_bar before composites", "primary-bar intents ride the next print", "primary indicators",
         "composite bars from prints"],
        data=t_long, config=_default_fee(), run={"primary_bars": [1, "m"]},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 1000, "enter_at": 999, "exit_at": 998,
                                           "subs": [[2, "m"], [3, "m"], [20, "tick"]], "inds": [["sma", 2]],
                                           "comp_inds": [[0, "sma", 2]], "extra": [["on_bar", 2, 1]]}})
    add("tk_primary_bars_exit", T, ["primary-bar close intent", "primary bars with depth"],
        data=t_long, depth=[S1, S2], config=_zero_fee(), run={"primary_bars": [1, "m"], "account_type": "margin", "leverage": 2.0},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 25, "enter_at": 5, "exit_at": 40,
                                           "frac": 0.3, "extra": [["on_bar", 3, 0]]}})
    add("tk_margin_hedging", T, ["margin account", "hedging", "attached stop/target on ticks"],
        data=t2, config=_zero_fee(), run={"account_type": "margin", "leverage": 3.0, "oms_type": "hedging"},
        script={"name": "hedge", "args": {"drive": "on_trade_tick", "every": 8}})
    add("tk_cash_netting_shorts", T, ["cash account", "netting", "short entries rejected", "directions",
                                      "settle(): command and intent queued in a post-step hook"],
        data=t2, config=_default_fee(), run={"directions": {S2: -1}},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 12, "enter_at": 2, "exit_at": 9,
                                           "sided": {S1: "long", S2: "short"}, "frac": 0.2, "after_close": "both"}})
    add("tk_margin_netting_shorts", T, ["margin account", "netting", "short entries", "order queued in a post-step hook"],
        data=t2, config=_zero_fee(), run={"account_type": "margin", "leverage": 4.0},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 12, "enter_at": 2, "exit_at": 9,
                                           "sided": {S1: "short", S2: "long"}, "frac": 0.3, "close_other": [5, S1],
                                           "after_close": "order"}})
    add("tk_typed_resting", T, ["limit", "stop", "stop-limit", "MIT", "LIT", "MTL", "tif", "cancel/modify"],
        data=t2, config=_zero_fee(), run={},
        script={"name": "typed", "args": {"drive": "on_trade_tick", "program": [
            [0, 1, "limit_buy"], [0, 3, "modify_last"], [0, 8, "market_sell"], [0, 10, "stop_buy"], [0, 14, "stop_limit_buy"],
            [0, 18, "mit_buy"], [0, 22, "lit_buy"], [0, 27, "cancel_all"], [0, 30, "mtl_buy"],
            [1, 2, "limit_far"], [1, 4, "cancel_last"], [1, 6, "ioc_limit"], [1, 9, "gtd_limit"], [1, 12, "day_limit"],
            [1, 15, "market_attach"], [1, 33, "close_all"], [1, 36, "post_only"]]}})
    add("tk_typed_exits", T, ["bracket", "OCO", "trailing market", "trailing bps", "trailing limit", "reduce_only"],
        data=t_long, config=_zero_fee(), run={},
        script={"name": "typed", "args": {"drive": "on_trade_tick", "program": [
            [0, 2, "bracket"], [0, 40, "bracket_limit"], [0, 70, "market_buy"], [0, 72, "trail"],
            [1, 3, "market_buy"], [1, 4, "oco_exit"], [1, 30, "market_buy"], [1, 31, "trail_bps"],
            [1, 55, "market_buy"], [1, 56, "trail_limit"]]}})
    add("tk_order_latency", T, ["order latency on ticks"],
        data=t2, config=_zero_fee(order_latency_ns=20 * NS), run={},
        script={"name": "typed", "args": {"drive": "on_trade_tick", "program": [
            [0, 2, "market_buy"], [0, 15, "market_sell"], [1, 3, "limit_buy"], [1, 25, "market_sell"]]}})
    add("tk_partial_fills", T, ["partial fills", "working order filled_qty", "fill slices"],
        data=t2, config=_zero_fee(partial_fills=True), run={},
        script={"name": "typed", "args": {"drive": "on_trade_tick", "program": [
            [0, 2, "market_buy"], [0, 20, "market_sell"], [1, 3, "limit_buy"], [1, 25, "market_sell"]]}})
    add("tk_queue_model_depth", T, ["queue model", "depth snapshots", "L1 sizes", "resting limits"],
        data=t3, depth=depth, config=_zero_fee(queue_fill_model=True, partial_fills=True), run={},
        script={"name": "typed", "args": {"drive": "on_trade_tick", "program": [
            [0, 2, "limit_buy"], [0, 12, "limit_buy"], [0, 30, "market_sell"],
            [1, 3, "limit_buy"], [1, 15, "limit_buy"], [1, 38, "close_all"],
            [2, 4, "limit_buy"], [2, 20, "limit_sell_reduce"], [2, 30, "market_sell"]]}})
    add("tk_queue_model_off", T, ["queue model off baseline", "depth snapshots ignored"],
        data=t3, depth=depth, config=_zero_fee(queue_fill_model=False), run={},
        script={"name": "typed", "args": {"drive": "on_trade_tick", "program": [
            [0, 2, "limit_buy"], [1, 3, "limit_buy"], [2, 4, "limit_buy"], [1, 30, "close_all"]]}})
    add("tk_twap", T, ["TWAP slices", "on_algo_started", "on_algo_completed"],
        data=t2, config=_zero_fee(), run={"oms_type": "hedging"},
        script={"name": "twap", "args": {"drive": "on_trade_tick", "at": 3}})
    add("tk_atr_refusal", T, ["ATR stop refused up front"],
        data=t2, config={"kwargs": {"initial_capital": 1_000_000.0}, "calls": [["set_atr_stop", [2.0, 5]]]}, run={},
        script={"name": "legacy", "args": {"drive": "on_trade_tick"}}, expect_error=True)
    add("tk_timers", T, ["clock per symbol", "time events before data", "intent from on_time_event"],
        data=t3, config=_zero_fee(), run={},
        script={"name": "timers", "args": {"drive": "on_trade_tick", "alert_ns": 0, "timer_ns": 40 * NS, "timer_start_ns": 0, "stop_ns": 0}},
        timers_from_data=True)
    add("tk_drawdown_halt", T, ["max_drawdown_pct", "halted", "halted_at"],
        data={**t_long, "n": 45}, config=_zero_fee(max_drawdown_pct=0.2), run={},
        script={"name": "allin", "args": {"drive": "on_trade_tick", "frac": 0.9}})
    add("tk_margin_call_halt", T, ["margin call", "on_margin_call", "halted_at"],
        data={**t_long, "n": 45}, config=_zero_fee(), run={"account_type": "margin", "leverage": 400.0},
        script={"name": "allin", "args": {"drive": "on_trade_tick", "frac": 1.0}})
    add("tk_max_positions", T, ["max_positions", "rejected_entries"],
        data=t3, config=_zero_fee(max_positions=1), run={},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 9, "enter_at": 1, "exit_at": 7, "frac": 0.1}})

    # ---- live stream ------------------------------------------------------
    sdata = {"symbols": [S0, S1, S2]}
    add("st_push_basic", ST,
        ["push_tick/push_depth/push_bar dispatch", "refused rows advance the clock", "intents held per symbol",
         "hooks fire inside the push", "ctx.idx event ordinal", "settle(): command and intent queued in a post-step hook"],
        data=sdata, config=_zero_fee(), run={},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 10, "enter_at": 3, "exit_at": 8,
                                           "frac": 0.2, "extra": [["on_quote", 6, 2], ["on_order_book", 3, 1]],
                                           "after_close": "both"}})
    add("st_refused_rows_timers", ST,
        ["_mark_time on refused rows", "timers fire on refused rows", "commands from time events routed at once",
         "submission_idx stamping across instruments"],
        data=sdata, config=_zero_fee(), run={},
        script={"name": "timers", "args": {"drive": "on_trade_tick", "alert_ns": 0, "timer_ns": 25 * NS, "timer_start_ns": 0, "stop_ns": 0}},
        timers_from_data=True, ops_from=0)
    add("st_warmup_executes", ST, ["warmup replay executes", "primary_bars on stream", "push_bar executes", "bar_stamp close"],
        ops_n=50, data=sdata, config=_zero_fee(), warmup=True, push_bars=True,
        run={"primary_bars": [1, "m"], "bar_stamp": "close"},
        script={"name": "legacy", "args": {"drive": "on_bar", "every": 7, "enter_at": 2, "exit_at": 5, "frac": 0.2,
                                           "subs": [[5, "m"]], "inds": [["sma", 3]]}})
    add("st_warmup_open_stamp", ST, ["warmup bars with bar_stamp open", "bar_ns offsets", "typed orders on warmup bars"],
        data=sdata, config=_zero_fee(), warmup=True, push_bars=True,
        run={"bar_stamp": "open", "bar_ns": MIN_NS},
        script={"name": "typed", "args": {"drive": "on_bar", "program": [
            [0, 3, "market_buy"], [0, 9, "market_sell"], [1, 4, "limit_buy"], [1, 12, "close_all"], [2, 6, "bracket"]]}})
    add("st_adopted_positions", ST, ["adopt_position before on_start", "no trade record for adoption", "cash account"],
        data=sdata, config=_zero_fee(), run={},
        initial_positions={S1: {"quantity": 500.0, "avg_price_from": "SYN001"}, S2: {"quantity": 100.0, "avg_price_from": "SYN002"}},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 14, "enter_at": 1, "exit_at": 3, "frac": 0.1}})
    add("st_adopted_margin", ST, ["adopt_position on fully funded margin"],
        data=sdata, config=_zero_fee(), run={"account_type": "margin", "leverage": 1.0},
        initial_positions={S1: {"quantity": 300.0, "avg_price_from": "SYN001", "timestamp_ns": 5}},
        script={"name": "legacy", "args": {"drive": "on_trade_tick", "every": 14, "enter_at": 6, "exit_at": 3, "frac": 0.1}})
    add("st_queue_depth_latency", ST, ["queue model on stream", "depth pushes", "order latency", "partial fills"],
        data=sdata, config=_zero_fee(queue_fill_model=True, partial_fills=True, order_latency_ns=10 * NS), run={},
        script={"name": "typed", "args": {"drive": "on_trade_tick", "program": [
            [0, 2, "limit_buy"], [0, 20, "market_sell"], [1, 3, "market_buy"], [1, 30, "close_all"],
            [2, 4, "limit_buy"], [2, 25, "close_all"]]}})
    add("st_drawdown_halt", ST, ["is_halted", "halted_at"],
        data=sdata, config=_zero_fee(max_drawdown_pct=0.15), run={},
        script={"name": "allin", "args": {"drive": "on_trade_tick", "frac": 0.9}})
    add("st_margin_hedging", ST, ["margin hedging on stream"],
        data=sdata, config=_zero_fee(), run={"account_type": "margin", "leverage": 3.0, "oms_type": "hedging"},
        script={"name": "hedge", "args": {"drive": "on_trade_tick", "every": 8}})
    return sc


# ---------------------------------------------------------------------------
# Running a scenario
# ---------------------------------------------------------------------------


def build_config(spec):
    cfg = BacktestConfig(**spec.get("kwargs", {}))
    for method, args in spec.get("calls", []):
        getattr(cfg, method)(*args)
    return cfg


def _bars_for(sc, datasets):
    d = sc["data"]
    src = datasets[d.get("source", "bars")]
    out = {}
    for s in d["symbols"]:
        a = _thaw_arrays(src[s])
        n = d["n"]
        ts = a["ts_close"] if d.get("stamp") == "close" else a["ts_open"]
        out[s] = {
            "timestamps": ts[:n],
            **{k: a[k][:n] for k in ("open", "high", "low", "close", "volume")},
        }
    return out


def _ticks_for(sc, datasets):
    d = sc["data"]
    out = {}
    for s in d["symbols"]:
        a = _thaw_arrays(datasets["ticks"][s])
        out[s] = {k: v[: d["n"]].copy() for k, v in a.items()}
    return out


def _depth_for(sc, datasets):
    return {s: _thaw_arrays(datasets["depth"][s]) for s in sc.get("depth", [])} or None


def _refused_ts(datasets):
    """Timestamps of the stream's priceless rows (no print, no book)."""
    return [op[2] for op in datasets["stream_ops"] if op[0] == "tick" and all(v == "0x0p+0" for v in op[3:])]


def _script_for(sc, datasets):
    spec = json.loads(json.dumps(sc["script"]))
    if sc.get("timers_from_data"):
        # Absolute times come from the feed: a fixed lead after its first
        # timestamp, or (stream) the instants of its refused rows.
        a = spec["args"]
        if sc["runner"] == "portfolio":
            first = int(_bars_for(sc, datasets)[sc["data"]["symbols"][0]]["timestamps"][0])
            a["alert_ns"], a["stop_ns"] = first + 5 * MIN_NS, first + 30 * MIN_NS
            a["timer_start_ns"] = a["alert_ns"] - a["timer_ns"]
        elif sc["runner"] == "tick":
            first = int(_ticks_for(sc, datasets)[sc["data"]["symbols"][0]]["timestamps"][0])
            a["alert_ns"], a["stop_ns"] = first + 30 * NS, first + 120 * NS
            a["timer_start_ns"] = a["alert_ns"] - a["timer_ns"]
        else:
            refused = _refused_ts(datasets)
            a["alert_ns"], a["timer_start_ns"] = refused[1], refused[3]
            a["stop_ns"] = refused[-1]
    return make_script(spec)


def _run_kwargs(sc):
    kw = dict(sc["run"])
    if "directions" in kw:
        kw["directions"] = dict(kw["directions"])
    if "primary_bars" in kw:
        kw["primary_bars"] = tuple(kw["primary_bars"])
    return kw


def _instrument_configs(sc):
    out = {}
    for sym, spec in sc.get("instrument_configs", {}).items():
        ic = InstrumentConfig(**spec.get("kwargs", {}))
        for method, args in spec.get("calls", []):
            getattr(ic, method)(*args)
        out[sym] = ic
    return out or None


def digest_result(pr):
    if pr is None:
        return None
    r = pr.result
    return {
        "equity_curve": enc(r.equity_curve()),
        "drawdown_curve": enc(r.drawdown_curve()),
        "returns": enc(r.returns()),
        "trades": [enc_trade(t) for t in r.trades()],
        "orders": [_fields(o, ORDER_FIELDS) for o in r.orders()],
        "metrics": enc(pr.metrics.to_dict()),
        "result_metrics_equal": enc(r.metrics.to_dict()) == enc(pr.metrics.to_dict()),
        "halted": pr.halted,
        "halted_at": pr.halted_at,
        "rejected_entries": pr.rejected_entries,
        "per_instrument": [_fields(p, ["symbol", "trades", "pnl", "rejected_entries"]) for p in pr.per_instrument],
    }


def run_scenario(sc, datasets, backend="legacy"):
    """Run one scenario from frozen inputs; return its observable record.

    ``backend`` is ``"legacy"`` (the runner the scenario is named for) or
    ``"runtime"``: ``run_strategies`` with the scenario's one strategy for the
    portfolio and tick scenarios, a one-strategy ``LiveNode`` fed the same
    pushes for the stream scenarios.
    """
    rec = Recorder(script=_script_for(sc, datasets))
    rec.tick_surface = sc["runner"] != "portfolio"
    config = build_config(sc["config"])
    ops_log = None
    result = None
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if backend == "runtime" and sc["runner"] != "stream":
                result = _run_on_runtime(sc, datasets, rec, config)
            elif sc["runner"] == "portfolio":
                result = run_portfolio_strategy(
                    rec, _bars_for(sc, datasets), config=config,
                    instrument_configs=_instrument_configs(sc), **_run_kwargs(sc),
                )
            elif sc["runner"] == "tick":
                result = run_tick_strategy(
                    rec, _ticks_for(sc, datasets), config=config, depth=_depth_for(sc, datasets),
                    instrument_configs=_instrument_configs(sc), **_run_kwargs(sc),
                )
            else:
                result, ops_log = _run_stream(sc, datasets, rec, config, backend)
        except Exception as ex:  # the fixture pins refusals too
            error = {"type": type(ex).__name__, "message": str(ex)}
    if sc.get("expect_error") and error is None:
        raise AssertionError(f"{sc['name']}: expected the run to be refused")
    if not sc.get("expect_error") and error is not None:
        raise AssertionError(f"{sc['name']}: unexpected {error}")
    return {
        "hooks": rec.hooks,
        "result": digest_result(result),
        "error": error,
        "warnings": [[w.category.__name__, _as_legacy_call(str(w.message), sc, backend)] for w in caught],
        "ops": ops_log,
    }


def _as_legacy_call(message, sc, backend):
    """A warning names the entry point that raised it; on the runtime backend
    that is ``run_strategies``, the one deliberate difference from the legacy
    record. Everything after the name must match."""
    if backend != "runtime":
        return message
    if sc["runner"] == "stream":
        return message.replace("LiveNode", "TickStrategyStream", 1) if message.startswith("LiveNode") else message
    legacy = "run_portfolio_strategy" if sc["runner"] == "portfolio" else "run_tick_strategy"
    return message.replace("run_strategies:", f"{legacy}:", 1) if message.startswith("run_strategies:") else message


def _run_on_runtime(sc, datasets, rec, config):
    if sc["runner"] == "portfolio":
        data = dict(data=_bars_for(sc, datasets))
    elif sc["runner"] == "tick":
        data = dict(ticks=_ticks_for(sc, datasets), depth=_depth_for(sc, datasets))
    else:
        raise ValueError(f"{sc['name']}: a stream scenario runs on a LiveNode, see _run_stream")
    return run_strategies(
        [rec], config=config, instrument_configs=_instrument_configs(sc), **data, **_run_kwargs(sc)
    )[0]


class _OneStrategyNode:
    """A one-strategy ``LiveNode`` with the surface of ``TickStrategyStream``."""

    def __init__(self, strategy, symbols, **kwargs):
        self._node = LiveNode(
            [strategy], symbols, kwargs.pop("config"),
            warmup_bars=kwargs.pop("warmup_bars"), initial_positions=_keyed(kwargs.pop("initial_positions")),
            **kwargs,
        )

    def push_tick(self, *args):
        return self._node.push_tick(*args)

    def push_depth(self, *args):
        return self._node.push_depth(*args)

    def push_bar(self, *args):
        return self._node.push_bar(*args)

    def positions(self, symbol):
        return self._node.positions(0, symbol)

    @property
    def equity(self):
        return self._node.equity(0)

    @property
    def is_halted(self):
        return self._node.is_halted(0)

    def finish(self):
        (result,) = self._node.finish()
        return result


def _keyed(seeds):
    return {0: seeds} if seeds else None


def _run_stream(sc, datasets, rec, config, backend="legacy"):
    syms = sc["data"]["symbols"]
    kw = _run_kwargs(sc)
    warm = None
    if sc.get("warmup"):
        warm = {}
        for s in syms:
            a = _thaw_arrays(datasets["warmup"][s])
            stamp = kw.get("bar_stamp", "open")
            ts = a["ts_close"] if stamp == "close" else a["ts_open"]
            warm[s] = {"timestamps": ts, **{k: a[k] for k in ("open", "high", "low", "close", "volume")}}
    init = None
    if sc.get("initial_positions"):
        first_px = {}
        for op in thaw_ops(datasets["stream_ops"]):
            if op[0] == "tick" and op[1] not in first_px and op[3] > 0:
                first_px[op[1]] = op[3]
        init = {}
        for s, p in sc["initial_positions"].items():
            init[s] = {"quantity": p["quantity"], "avg_price": first_px[p["avg_price_from"]]}
            if "timestamp_ns" in p:
                init[s]["timestamp_ns"] = p["timestamp_ns"]
    make = _OneStrategyNode if backend == "runtime" else TickStrategyStream
    stream = make(rec, syms, config=config, warmup_bars=warm, initial_positions=init, **kw)
    ops_log = []
    last_held: dict = {}
    bar_data = {s: _thaw_arrays(datasets["bars"][s]) for s in syms} if sc.get("push_bars") else {}
    stamp = kw.get("bar_stamp", "open")
    for i, op in enumerate(_stream_ops(sc, datasets, bar_data, stamp)):
        before = len(rec.hooks)
        if op[0] == "tick":
            ret = stream.push_tick(op[1], op[2], *op[3:])
        elif op[0] == "depth":
            stream.push_depth(op[1], op[2], op[3], op[4])
            ret = None
        else:
            stream.push_bar(op[1], op[2], *op[3:])
            ret = None
        held = {s: [enc_position(p) for p in stream.positions(s)] for s in syms if stream.positions(s)}
        ops_log.append([
            i, op[0], op[1], ret, len(rec.hooks) - before, enc(stream.equity), stream.is_halted,
            held if held != last_held else None,  # positions, when they changed
        ])
        last_held = held
    return stream.finish(), ops_log


def _stream_ops(sc, datasets, bar_data, stamp):
    """The pushed feed. With ``push_bars``, the day-2 bars are pushed at their
    close time, in order with the prints (their stamps follow ``bar_stamp``)."""
    ops = thaw_ops(datasets["stream_ops"])[: sc.get("ops_n", 56)]
    if not bar_data:
        return ops
    merged = [(op[2], 0, k, op) for k, op in enumerate(ops)]
    horizon = ops[-1][2]
    for s, a in bar_data.items():
        for j in range(30, len(a["ts_open"])):  # day-2 head
            close = int(a["ts_close"][j])
            if close > horizon:
                break
            ts = close if stamp == "close" else int(a["ts_open"][j])
            merged.append((close, 1, j, ("bar", s, ts, *[float(a[k][j]) for k in ("open", "high", "low", "close", "volume")])))
    merged.sort(key=lambda m: m[:3])
    return [m[3] for m in merged]


# ---------------------------------------------------------------------------
# Generate / load
# ---------------------------------------------------------------------------


def generate_all():
    datasets = build_datasets()
    scenarios = {}
    for sc in scenario_table():
        scenarios[sc["name"]] = {"descriptor": sc, "expected": run_scenario(sc, datasets)}
    return {"version": 1, "datasets": datasets, "scenarios": scenarios}


def dump(fixtures, path=FIXTURE_PATH):
    """One top-level entry per line keeps git diffs reviewable."""
    parts = []
    for name, entry in fixtures["scenarios"].items():
        parts.append(f"{json.dumps(name)}:{json.dumps(entry, separators=(',', ':'))}")
    body = (
        '{"version":%d,\n"datasets":%s,\n"scenarios":{\n%s\n}}\n'
        % (fixtures["version"], json.dumps(fixtures["datasets"], separators=(",", ":")), ",\n".join(parts))
    )
    Path(path).write_text(body)


def load(path=FIXTURE_PATH):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    fx = generate_all()
    dump(fx)
    size = FIXTURE_PATH.stat().st_size
    print(f"wrote {FIXTURE_PATH} ({size / 1e6:.2f} MB, {len(fx['scenarios'])} scenarios)")
