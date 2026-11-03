"""run_strategies: many strategies over one shared copy of the data.

The contract under test: for any strategy ``s`` and symbol subset ``syms``,

    run_strategies([(s, syms)], data, **kw)[0]
        == run_portfolio_strategy(s, {k: data[k] for k in data if k in syms}, **kw)

exactly, and it stays so when other strategies run in the same call. Exact
means trades field by field, ``orders()``, the equity/drawdown/returns curves,
metrics, ``halted``/``halted_at``, ``per_instrument``, and every hook the
strategy was called on with the values it observed there.

The one deliberate difference: ``ctx.series(sym)`` in the new runner returns
arrays cut to what is visible at the current event, not the full arrays. The
recording strategy never reads it, and one test pins the truncation.

Every test here skips until ``raptorbt.run_strategies`` exists.
"""

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Indicator, InstrumentConfig, Strategy
from raptorbt.strategy import orders

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benches" / "runtime"))
import synthetic  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(raptorbt, "run_strategies"), reason="run_strategies not available"
)

# Size knobs. The full differential run is ~200 cases; lower N_CASES for a
# quick local pass.
N_CASES = 200
MAX_INSTRUMENTS = 5
MAX_DAYS = 1
MAX_BARS = 120  # per symbol in the differential cases ...
FULL_DAY_EVERY = 20  # ... except every Nth, which keeps the whole session
MAX_STRATEGIES = 5
DATASET_SEEDS = 10  # distinct synthetic datasets reused across cases

MIN_NS = 60 * 10**9
ENTRY_KINDS = ("intent", "market", "limit", "stop", "bracket", "mixed")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _dataset(n_instruments, days, tf, seed):
    return synthetic.generate(
        n_instruments=n_instruments, days=days, seed=seed, timeframes=(tf,)
    ).bars[tf]


def _series(bars, stamp):
    ts = bars["ts_open"] if stamp == "open" else bars["ts_close"]
    out = {"timestamps": np.ascontiguousarray(ts)}
    for key in ("open", "high", "low", "close", "volume"):
        out[key] = np.ascontiguousarray(bars[key])
    return out


def _subset(data, syms):
    return {k: data[k] for k in data if k in syms}


def _stamp_kwargs(stamp, width_ns, symbols, per_symbol):
    if stamp == "close":
        return {"bar_stamp": "close"}
    if per_symbol:
        return {"bar_stamp": "open", "bar_ns": {s: width_ns for s in symbols}}
    return {"bar_stamp": "open", "bar_ns": width_ns}


def _kw_for(kw, syms):
    """Keyword arguments as a single-strategy legacy run on ``syms`` sees them."""
    kw = dict(kw)
    if isinstance(kw.get("bar_ns"), dict):
        kw["bar_ns"] = {s: w for s, w in kw["bar_ns"].items() if s in syms}
    return kw


# ---------------------------------------------------------------------------
# recording strategy
# ---------------------------------------------------------------------------
def _dump(o, depth=0):
    """A hashable, exactly comparable picture of whatever a hook was handed."""
    if o is None or isinstance(o, (bool, int, str, bytes)):
        return o
    if isinstance(o, float):
        return repr(o)  # nan-safe, and -0.0 != 0.0
    if isinstance(o, (list, tuple)):
        return tuple(_dump(x, depth) for x in o)
    if isinstance(o, dict):
        return tuple(sorted(((str(k), _dump(v, depth)) for k, v in o.items())))
    if depth > 3:
        return repr(o)
    out = []
    for name in dir(o):
        if name.startswith("_"):
            continue
        val = getattr(o, name)
        if not callable(val):
            out.append((name, _dump(val, depth + 1)))
    return tuple(out)


with pytest.warns(DeprecationWarning, match="deprecated hooks"):
    class Rec(Strategy):
        """Logs every hook with what it saw; decides deterministically from it."""

        def __init__(self, p):
            super().__init__()
            self.p = p
            self.rec = []
            self.last15 = {}

        # -- observation ------------------------------------------------------
        def _ind(self):
            return tuple(
                (name, sym, _dump(ind.value), ind.initialized)
                for name, group in self.inds.items()
                for sym, ind in group.items()
            )

        def _cache(self):
            c = self.cache
            recs = c.orders()
            return (
                len(recs),
                tuple(r.client_id for r in c.orders_open()),
                _dump(recs[-1]) if recs else None,
                len(c.closed_trades()),
                repr(c.realized_pnl()),
            )

        def _ctx(self, ctx):
            sym = ctx.symbol
            return (
                sym, ctx.idx, ctx.timestamp, ctx.bar_close_ts, _dump(tuple(ctx.bar)),
                repr(ctx.equity), repr(ctx.cash), repr(ctx.free_capital),
                _dump(ctx.position), _dump(ctx.positions()), _dump(ctx.position_for(sym)),
                repr(ctx.net_position), ctx.is_flat,
                _dump(ctx.open_orders()),
            )

        def _log(self, hook, ctx, *extra):
            self.rec.append((hook, self._ctx(ctx), self._ind(), self._cache(), _dump(extra)))

        # -- hooks ------------------------------------------------------------
        def on_start(self, ctx):
            p = self.p
            syms = list(ctx.symbols)
            self.rec.append(
                ("start", tuple(syms), repr(ctx.equity), repr(ctx.cash), repr(ctx.free_capital))
            )
            reg = self.register_indicators
            self.inds = {
                "sma_f": reg(lambda: Indicator.sma(p["fast"]), syms),
                "sma_s": reg(lambda: Indicator.sma(p["slow"]), syms),
                "ema": reg(lambda: Indicator.ema(p["ema"]), syms),
                "rsi": reg(lambda: Indicator.rsi(p["rsi"]), syms),
                "atr": reg(lambda: Indicator.atr(p["atr"]), syms),
            }
            self.s5 = self.s15 = self.s1d = None
            if p["sub5"]:
                self.s5 = self.subscribe_bars(5, "m")
                self.inds["ema5"] = reg(lambda: Indicator.ema(4), syms, stream_id=self.s5)
            if p["sub15"]:
                self.s15 = self.subscribe_bars(15, "m")
                self.inds["sma15"] = reg(lambda: Indicator.sma(3), syms, stream_id=self.s15)
            if p["sub1d"]:
                self.s1d = self.subscribe_bars(1, "d")
            # Anchored at the data: a timer starting at the epoch would spin
            # through every missed interval on its first firing.
            self.clock.set_timer(
                "timer", p["timer_min"] * MIN_NS, start_ns=p["t0"] + p["timer_min"] * MIN_NS
            )
            self.clock.set_time_alert("alert", p["t0"] + p["alert_min"] * MIN_NS)

        def on_stop(self, ctx):
            self._log("stop", ctx)

        def on_timer(self, ctx, event):
            self._log("time", ctx, tuple(event))

        def on_composite_bar(self, ctx, bar):
            if bar.stream_id == self.s15:
                self.last15[bar.symbol] = bar.close
            self._log("composite", ctx, bar)

        def on_fill(self, ctx, e):
            self._log("filled", ctx, e)

        def on_order_rejected(self, ctx, e):
            self._log("rejected", ctx, e)

        def on_order_accepted(self, ctx, e):
            self._log("accepted", ctx, e)

        def on_order_triggered(self, ctx, e):
            self._log("triggered", ctx, e)

        def on_order_canceled(self, ctx, e):
            self._log("canceled", ctx, e)

        def on_order_expired(self, ctx, e):
            self._log("expired", ctx, e)

        def on_order_event(self, ctx, e):
            self._log("order_event", ctx, e)

        def on_margin_call(self, ctx, e):
            self._log("margin_call", ctx, e)

        def on_position_opened(self, ctx, e):
            self._log("opened", ctx, e)

        def on_position_closed(self, ctx, e):
            self._log("closed", ctx, e)

        def on_bar(self, ctx):
            self._log("bar", ctx)
            p, sym = self.p, ctx.symbol
            fast, slow = self.inds["sma_f"][sym].value, self.inds["sma_s"][sym].value
            rsi = self.inds["rsi"][sym].value
            atr = self.inds["atr"][sym].value
            side = "buy" if p["dirs"].get(sym, 1) > 0 else "sell"
            exit_side = "sell" if side == "buy" else "buy"
            px = ctx.bar.close
            positions = ctx.positions()
            working = ctx.open_orders()

            # Cancel the oldest resting order now and then, by cache state.
            open_ids = [r.client_id for r in self.cache.orders_open()]
            if p["cancel_every"] and ctx.idx % p["cancel_every"] == 0 and working and open_ids:
                self.cancel_order(open_ids[0])

            # Exits.
            exit_now = (
                fast is not None and slow is not None
                and (fast < slow if side == "buy" else fast > slow)
            ) or (rsi is not None and rsi > p["rsi_hi"] and side == "buy")
            if positions and exit_now:
                if p["hedging"]:
                    self.close_position(position_id=positions[0].position_id)
                elif p["close_kind"] == "intent":
                    self.close_position()
                else:
                    self.submit_order(orders.Market(side=exit_side, reduce_only=True))
                return

            # Entries.
            room = (len(positions) < 3) if p["hedging"] else (not positions)
            throttled = len(self.cache.orders_open()) >= 2 or len(working) >= 2
            gate = (
                sym not in self.last15 or not p["sub15"]
                or self.inds["sma15"][sym].value is None
                or self.last15[sym] >= self.inds["sma15"][sym].value
            )
            trend = fast is None or slow is None or (fast > slow if side == "buy" else fast < slow)
            due = ctx.idx % p["every"] == p["offset"]
            if not (room and due and trend and gate and not throttled):
                return
            kind = p["entry"]
            if kind == "mixed":
                kind = ENTRY_KINDS[(ctx.idx // p["every"]) % 5]
            frac = p["frac"]
            off = p["off"] + (0.0 if atr is None else 0.1 * atr / px)
            if kind == "intent":
                self.enter(size_frac=frac)
            elif kind == "market":
                self.submit_order(orders.Market(side=side, size_frac=frac, tif="gtc"))
            elif kind == "limit":
                price = round(px * (1 - off if side == "buy" else 1 + off), 2)
                self.submit_order(
                    orders.Limit(side=side, size_frac=frac, price=price, tif=p["tif"])
                )
            elif kind == "stop":
                trig = round(px * (1 + off if side == "buy" else 1 - off), 2)
                self.submit_order(
                    orders.StopMarket(side=side, size_frac=frac, trigger=trig, tif=p["tif"])
                )
            else:  # bracket
                up, dn = round(px * (1 + 1.5 * off), 2), round(px * (1 - off), 2)
                stop, tgt = (dn, up) if side == "buy" else (up, dn)
                if p["hedging"]:
                    # Bracket legs are netting-only; attach levels to the position.
                    self.submit_order(
                        orders.Market(
                            side=side, size_frac=frac, stop_price=stop, target_price=tgt
                        )
                    )
                else:
                    self.submit_bracket(orders.Market(side=side, size_frac=frac), stop, tgt)


def _start(data):
    return min(int(d["timestamps"][0]) for d in data.values())


def _params(rng, symbols, hedging, t0):
    return {
        "t0": t0,
        "fast": int(rng.integers(2, 6)),
        "slow": int(rng.integers(7, 20)),
        "ema": int(rng.integers(3, 15)),
        "rsi": int(rng.integers(3, 15)),
        "atr": int(rng.integers(3, 15)),
        "rsi_hi": float(rng.uniform(55, 80)),
        "entry": str(rng.choice(ENTRY_KINDS)),
        "close_kind": str(rng.choice(["intent", "typed"])),
        "every": int(rng.integers(5, 40)),
        "offset": int(rng.integers(0, 5)),
        "frac": float(rng.uniform(0.02, 0.15)),
        "off": float(rng.uniform(0.0005, 0.004)),
        "tif": str(rng.choice(["day", "gtc"])),
        "sub5": bool(rng.integers(0, 2)),
        "sub15": bool(rng.integers(0, 2)),
        "sub1d": bool(rng.integers(0, 2)),
        "timer_min": int(rng.integers(20, 90)),
        "alert_min": int(rng.integers(1, 200)),
        "cancel_every": int(rng.choice([0, 0, 13, 29])),
        "hedging": hedging,
        "dirs": {},
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _same_array(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    assert np.array_equal(a, b, equal_nan=True), f"{what}: values differ"
    assert a.tobytes() == b.tobytes(), f"{what}: bytes differ (e.g. -0.0 vs 0.0)"


def _same_list(a, b, what):
    assert len(a) == len(b), f"{what}: {len(a)} != {len(b)} entries"
    for i, (x, y) in enumerate(zip(a, b)):
        assert x == y, f"{what}[{i}]:\n  {x}\n!=\n  {y}"


def assert_results_equal(a, b, log_a=None, log_b=None, label=""):
    """Two ``PortfolioResult`` are identical in everything the contract names.

    Covers: trades field by field, orders(), equity / drawdown / returns
    curves (value and byte identical), metrics (on the result and on the
    portfolio result), halted, halted_at, rejected_entries, per_instrument,
    and, when the recording logs are given, every hook call with the values
    observed there.
    """
    ra, rb = a.result, b.result
    _same_list([_dump(t) for t in ra.trades()], [_dump(t) for t in rb.trades()],
               f"{label} trades")
    _same_list([_dump(o) for o in ra.orders()], [_dump(o) for o in rb.orders()],
               f"{label} orders")
    for name in ("equity_curve", "drawdown_curve", "returns"):
        _same_array(getattr(ra, name)(), getattr(rb, name)(), f"{label} {name}")
    assert _dump(ra.metrics) == _dump(rb.metrics), f"{label} result.metrics"
    assert _dump(a.metrics) == _dump(b.metrics), f"{label} metrics"
    assert a.halted == b.halted, f"{label} halted {a.halted} != {b.halted}"
    assert a.halted_at == b.halted_at, f"{label} halted_at {a.halted_at} != {b.halted_at}"
    assert a.rejected_entries == b.rejected_entries, f"{label} rejected_entries"
    _same_list([_dump(s) for s in a.per_instrument], [_dump(s) for s in b.per_instrument],
               f"{label} per_instrument")
    if log_a is not None or log_b is not None:
        _same_list(log_a, log_b, f"{label} hook log")


def _digest(results, strategies):
    h = hashlib.sha256()
    for res, strat in zip(results, strategies):
        r = res.result
        for arr in (r.equity_curve(), r.drawdown_curve(), r.returns()):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr([_dump(t) for t in r.trades()]).encode())
        h.update(repr([_dump(o) for o in r.orders()]).encode())
        h.update(repr((_dump(r.metrics), res.halted, res.halted_at,
                       [_dump(s) for s in res.per_instrument])).encode())
        h.update(repr(strat.rec).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _case(i):
    """One seeded scenario: data, shared keyword arguments and strategy specs."""
    rng = np.random.default_rng(50_000 + i)
    n_inst = int(rng.integers(2, MAX_INSTRUMENTS + 1))
    days = int(rng.integers(1, MAX_DAYS + 1))
    tf = str(rng.choice(["1m", "5m"]))
    width_ns = (1 if tf == "1m" else 5) * MIN_NS
    bars = _dataset(n_inst, days, tf, int(rng.integers(1, DATASET_SEEDS + 1)))
    stamp = str(rng.choice(["open", "close"]))
    cut = None if i % FULL_DAY_EVERY == 0 else MAX_BARS
    data = {s: {k: v[:cut] for k, v in _series(b, stamp).items()} for s, b in bars.items()}
    symbols = list(data)

    account = str(rng.choice(["cash", "margin"]))
    hedging = bool(rng.integers(0, 2))
    cfg = BacktestConfig(
        initial_capital=10_000_000.0,
        fees=0.0002,
        fill_timing=str(rng.choice(["same_bar_close", "next_bar_open"])),
        session_tz_offset_ns=synthetic.IST_OFFSET_NS,
        max_positions=int(rng.choice([2, 3])) if rng.random() < 0.25 else None,
    )
    if rng.random() < 0.4:
        cfg.set_atr_stop(float(rng.uniform(1.5, 4.0)), int(rng.integers(5, 15)))
    if rng.random() < 0.3:
        cfg.set_atr_target(float(rng.uniform(2.0, 5.0)), int(rng.integers(5, 15)))
    if rng.random() < 0.2:
        cfg.set_fixed_stop(float(rng.uniform(0.005, 0.03)))
    kw = {
        "config": cfg,
        "oms_type": "hedging" if hedging else "netting",
        "account_type": account,
        "leverage": float(rng.integers(1, 6)) if account == "margin" else 1.0,
        **_stamp_kwargs(stamp, width_ns, symbols, bool(rng.integers(0, 2))),
    }
    dirs = {s: (-1 if account == "margin" and rng.random() < 0.3 else 1) for s in symbols}
    if rng.random() < 0.7:
        kw["directions"] = dirs
    else:
        dirs = {}
    if rng.random() < 0.3:
        kw["instrument_configs"] = {
            s: InstrumentConfig(lot_size=float(rng.choice([1.0, 5.0, 25.0])))
            for s in symbols if rng.random() < 0.7
        }

    specs = []
    for _ in range(int(rng.integers(1, MAX_STRATEGIES + 1))):
        p = _params(rng, symbols, hedging, _start(data))
        p["dirs"] = dirs
        k = int(rng.integers(1, len(symbols) + 1))
        chosen = [symbols[j] for j in sorted(rng.choice(len(symbols), k, replace=False))]
        specs.append((p, None if rng.random() < 0.25 else chosen))
    return data, kw, specs


def _entries(specs):
    """Fresh strategy instances and the ``strategies`` argument for a call."""
    strats = [Rec(p) for p, _ in specs]
    items = [s if syms is None else (s, syms) for s, (_, syms) in zip(strats, specs)]
    return strats, items


def _legacy(spec, data, kw):
    p, syms = spec
    strat = Rec(p)
    subset = _subset(data, set(data) if syms is None else syms)
    res = raptorbt.run_portfolio_strategy(strat, subset, **_kw_for(kw, set(subset)))
    return strat, res


@pytest.mark.parametrize("case", range(N_CASES))
def test_run_strategies_matches_independent_legacy_runs(case):
    data, kw, specs = _case(case)
    strats, items = _entries(specs)
    results = raptorbt.run_strategies(items, data, **kw)
    assert len(results) == len(specs)
    for j, (spec, strat, res) in enumerate(zip(specs, strats, results)):
        ref_strat, ref = _legacy(spec, data, kw)
        assert ref_strat.rec, "the legacy run recorded nothing"
        assert_results_equal(
            res, ref, strat.rec, ref_strat.rec, label=f"case {case} strategy {j}"
        )


def test_the_differential_cases_are_not_vacuous():
    """Across the cases, every hook family fires and trades happen."""
    hooks, trades, kinds = set(), 0, set()
    for i in range(min(N_CASES, 20)):
        data, kw, specs = _case(i)
        for spec in specs[:1]:
            strat, res = _legacy(spec, data, kw)
            hooks |= {r[0] for r in strat.rec}
            trades += len(res.result.trades())
            kinds |= {o.kind for o in res.result.orders()}
    assert {"bar", "composite", "time", "filled", "opened", "closed"} <= hooks, hooks
    assert trades > 0
    assert {"market", "limit"} <= kinds, kinds


# ---------------------------------------------------------------------------
# fixed scenarios
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _shared_world():
    bars = _dataset(3, 1, "1m", 3)
    data = {s: _series(b, "close") for s, b in bars.items()}
    cfg = BacktestConfig(
        initial_capital=10_000_000.0, fees=0.0002, session_tz_offset_ns=synthetic.IST_OFFSET_NS
    )
    cfg.set_atr_stop(2.5, 10)
    kw = {"config": cfg, "bar_stamp": "close"}
    return data, kw


def _twenty():
    """20 strategies: identical indicators and subscriptions, overlapping symbols."""
    data, kw = _shared_world()
    symbols = list(data)
    rng = np.random.default_rng(777)
    specs = []
    for j in range(20):
        p = {
            "t0": _start(data), "fast": 5, "slow": 12, "ema": 9, "rsi": 14, "atr": 14, "rsi_hi": 70.0,
            "sub5": True, "sub15": True, "sub1d": True, "timer_min": 30, "alert_min": 45,
            "hedging": False, "dirs": {}, "close_kind": "intent",
            # Only the decisions differ between strategies.
            "entry": ENTRY_KINDS[j % len(ENTRY_KINDS)],
            "every": 7 + j, "offset": j % 5, "frac": 0.02 + 0.005 * j,
            "off": 0.001 + 0.0002 * (j % 4), "tif": "day" if j % 2 else "gtc",
            "cancel_every": 0 if j % 3 else 17,
        }
        k = int(rng.integers(2, len(symbols) + 1))
        chosen = [symbols[x] for x in sorted(rng.choice(len(symbols), k, replace=False))]
        specs.append((p, chosen))
    return data, kw, specs


def test_twenty_strategies_with_shared_indicators_match_independent_runs():
    data, kw, specs = _twenty()
    strats, items = _entries(specs)
    results = raptorbt.run_strategies(items, data, **kw)
    assert len(results) == 20
    for j, (spec, strat, res) in enumerate(zip(specs, strats, results)):
        ref_strat, ref = _legacy(spec, data, kw)
        assert_results_equal(res, ref, strat.rec, ref_strat.rec, label=f"strategy {j}")
    assert sum(len(r.result.trades()) for r in results) > 0


def test_result_order_is_independent_of_input_order():
    data, kw, specs = _twenty()
    specs = specs[:4]
    strats, items = _entries(specs)
    base = raptorbt.run_strategies(items, data, **kw)
    for seed in range(2):
        perm = list(np.random.default_rng(seed).permutation(len(specs)))
        p_strats, p_items = _entries([specs[k] for k in perm])
        shuffled = raptorbt.run_strategies(p_items, data, **kw)
        for pos, k in enumerate(perm):
            assert_results_equal(
                shuffled[pos], base[k], p_strats[pos].rec, strats[k].rec,
                label=f"perm {seed} strategy {k}",
            )


def test_sharing_does_not_change_results():
    data, kw, specs = _twenty()
    specs = specs[:6]
    a_strats, a_items = _entries(specs)
    b_strats, b_items = _entries(specs)
    shared = raptorbt.run_strategies(a_items, data, share=True, **kw)
    private = raptorbt.run_strategies(b_items, data, share=False, **kw)
    for j in range(len(specs)):
        assert_results_equal(
            shared[j], private[j], a_strats[j].rec, b_strats[j].rec, label=f"strategy {j}"
        )


@pytest.mark.parametrize("share", [True, False])
def test_two_identical_calls_are_byte_identical(share):
    data, kw, specs = _twenty()
    specs = specs[:4]
    digests = []
    for _ in range(2):
        strats, items = _entries(specs)
        results = raptorbt.run_strategies(items, data, share=share, **kw)
        digests.append(_digest(results, strats))
    assert digests[0] == digests[1]


def test_a_class_runs_like_an_instance():
    class Plain(Rec):
        def __init__(self):
            super().__init__(dict(_twenty()[2][0][0]))

    data, kw = _shared_world()
    res = raptorbt.run_strategies([Plain], data, **kw)[0]
    ref = raptorbt.run_portfolio_strategy(Plain(), data, **kw)
    assert_results_equal(res, ref)


def test_a_bare_strategy_gets_every_symbol():
    data, kw, specs = _twenty()
    spec = (specs[0][0], None)
    strats, items = _entries([spec])
    res = raptorbt.run_strategies(items, data, **kw)[0]
    ref_strat, ref = _legacy(spec, data, kw)
    assert_results_equal(res, ref, strats[0].rec, ref_strat.rec)
    assert {s.symbol for s in res.per_instrument} == set(data)


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------
def _crash_bars(n=40):
    close = np.concatenate([np.full(10, 100.0), np.linspace(100.0, 55.0, n - 10)])
    ts = np.arange(n, dtype=np.int64) * MIN_NS + MIN_NS
    return {
        "timestamps": ts, "open": close.copy(), "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1_000.0),
    }


class _All(Strategy):
    """Enters every symbol once with a given fraction of capital."""

    def __init__(self, frac):
        super().__init__()
        self.frac = frac
        self.done = set()
        self.margin_calls = 0
        self.seen = []

    def on_bar(self, ctx):
        self.seen.append((ctx.symbol, ctx.idx, repr(ctx.equity), repr(ctx.free_capital)))
        if ctx.symbol not in self.done:
            self.done.add(ctx.symbol)
            self.enter(size_frac=self.frac)

    def on_margin_call(self, ctx, event):
        self.margin_calls += 1


def test_a_margin_call_in_one_strategy_does_not_touch_another():
    data = {"AAA": _crash_bars(), "BBB": _crash_bars()}
    kw = {
        "config": BacktestConfig(initial_capital=100_000.0, fees=0.0),
        "account_type": "margin", "leverage": 5.0, "bar_stamp": "close",
    }
    wild, calm = _All(1.0), _All(0.05)
    r_wild, r_calm = raptorbt.run_strategies([wild, calm], data, **kw)

    assert r_wild.halted and wild.margin_calls > 0, "the scenario must margin-call"
    assert not r_calm.halted and calm.margin_calls == 0

    solo = _All(0.05)
    ref = raptorbt.run_portfolio_strategy(solo, data, **kw)
    assert_results_equal(r_calm, ref)
    assert calm.seen == solo.seen

    solo_wild = _All(1.0)
    ref_wild = raptorbt.run_portfolio_strategy(solo_wild, data, **kw)
    assert_results_equal(r_wild, ref_wild)
    assert (wild.seen, wild.margin_calls) == (solo_wild.seen, solo_wild.margin_calls)


def test_a_drawdown_halt_in_one_strategy_does_not_touch_another():
    data = {"AAA": _crash_bars(), "BBB": _crash_bars()}
    kw = {
        "config": BacktestConfig(initial_capital=100_000.0, fees=0.0, max_drawdown_pct=10.0),
        "bar_stamp": "close",
    }
    heavy, light = _All(0.9), _All(0.01)
    r_heavy, r_light = raptorbt.run_strategies([heavy, light], data, **kw)
    assert r_heavy.halted and r_heavy.halted_at is not None
    assert not r_light.halted and r_light.halted_at is None
    assert_results_equal(r_light, raptorbt.run_portfolio_strategy(_All(0.01), data, **kw))
    assert_results_equal(r_heavy, raptorbt.run_portfolio_strategy(_All(0.9), data, **kw))


def test_strategies_do_not_share_an_account():
    """Both enter the same names at once; neither's capital is touched by the other."""
    data = {"AAA": _crash_bars(), "BBB": _crash_bars()}
    kw = {"config": BacktestConfig(initial_capital=100_000.0, fees=0.0), "bar_stamp": "close"}
    a, b = _All(0.9), _All(0.9)
    ra, rb = raptorbt.run_strategies([a, b], data, **kw)
    ref = raptorbt.run_portfolio_strategy(_All(0.9), data, **kw)
    assert_results_equal(ra, ref)
    assert_results_equal(rb, ref)


# ---------------------------------------------------------------------------
# series truncation (the deliberate difference)
# ---------------------------------------------------------------------------
class _Watcher(Strategy):
    def __init__(self):
        super().__init__()
        self.lens = []
        self.bad = []

    def on_bar(self, ctx):
        own = ctx.series(ctx.symbol)
        self.lens.append(len(own["close"]))
        for sym in ctx.symbols:
            ts = ctx.series(sym)["timestamps"]
            if len(ts) and ts[-1] > ctx.bar_close_ts:
                self.bad.append((ctx.symbol, sym, int(ts[-1]), ctx.bar_close_ts))


def test_series_is_truncated_to_what_is_visible():
    bars = _dataset(3, 1, "1m", 4)
    data = {s: _series(b, "close") for s, b in bars.items()}
    w = _Watcher()
    raptorbt.run_strategies([w], data, bar_stamp="close")
    assert not w.bad, w.bad[:3]
    # Each symbol's own array grows 1, 2, 3, ... one bar per event.
    n = len(next(iter(data.values()))["close"])
    assert max(w.lens) == n and min(w.lens) == 1
    assert w.lens.count(1) == len(data)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
def test_an_unknown_symbol_raises_and_names_it():
    data, kw = _shared_world()
    with pytest.raises(ValueError, match="NOPE"):
        raptorbt.run_strategies([(Rec(_twenty()[2][0][0]), [next(iter(data)), "NOPE"])],
                                data, **kw)


def test_an_unknown_symbol_in_a_later_strategy_still_raises():
    data, kw, specs = _twenty()
    first = Rec(specs[0][0])
    with pytest.raises(ValueError, match="MISSING"):
        raptorbt.run_strategies(
            [(first, specs[0][1]), (Rec(specs[1][0]), ["MISSING"])], data, **kw
        )


def test_an_empty_strategy_list_raises():
    data, kw = _shared_world()
    with pytest.raises(ValueError):
        raptorbt.run_strategies([], data, **kw)
