"""Produce every published raptorbt performance figure.

Run after `uv run maturin develop --release`. Prints the table used in the docs
and writes a JSON summary next to this file. See README.md in this directory for
why the 0.7.0 numbers are not comparable to the 0.6.4 ones.
"""

import hashlib
import json
import pathlib
import platform
import resource
import sys
import time

import numpy as np

import raptorbt as r

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _data import bars, sma_signals  # noqa: E402

CAPITAL = 100_000.0
FEES = 0.0002


def _cfg():
    return r.BacktestConfig(initial_capital=CAPITAL, fees=FEES)


def _time(fn, reps):
    """Fastest of `reps` runs -- engine time, not scheduler noise."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[0], samples[len(samples) // 2], out


def bench_bars():
    rows = {}
    cfg = _cfg()
    for n, reps in ((1_000, 5000), (5_000, 3000), (10_000, 2000), (50_000, 1000),
                    (93_750, 500), (1_875_000, 30), (25_000_000, 3)):
        ts, o, h, l, c, v = bars(n)
        e, x = sma_signals(c)
        best, p50, res = _time(
            lambda: r.run_single_backtest(
                timestamps=ts, open=o, high=h, low=l, close=c, volume=v,
                entries=e, exits=x, direction=1, weight=1.0, symbol="B", config=cfg),
            reps)
        rows[n] = {"best_ms": best * 1e3, "p50_ms": p50 * 1e3,
                   "m_bars_per_s": n / best / 1e6, "trades": res.metrics.total_trades}
    return rows


def _ticks(n, seed=11):
    rng = np.random.default_rng(seed)
    logp = np.clip(np.cumsum(rng.normal(0, 0.00005, n)), -0.5, 0.5) + np.log(1000.0)
    ltp = np.exp(logp)
    half = ltp * 0.00025
    bq = np.abs(rng.normal(500, 150, n))
    sq = np.abs(rng.normal(500, 150, n))
    oi = np.abs(rng.normal(1e6, 1e4, n))
    ts = np.arange(n, dtype=np.int64) * 1_000_000
    ret = np.zeros(n)
    ret[60:] = (ltp[60:] / ltp[:-60] - 1.0) * 100
    return ts, ltp, ltp - half, ltp + half, bq, sq, oi, ret > 0.02, ret < -0.02


def bench_ticks():
    rows = {}
    for n, reps in ((10_000, 300), (100_000, 100), (1_000_000, 20), (10_000_000, 3)):
        ts, ltp, bid, ask, bq, sq, oi, e, x = _ticks(n)
        # max_trades defaults to 50 and BREAKS THE LOOP once reached, so the
        # default would time a partial traversal and inflate ticks/sec.
        best, _, res = _time(
            lambda: r.run_tick_backtest(
                timestamps=ts, ltp=ltp, bid=bid, ask=ask, buy_qty_delta=bq,
                sell_qty_delta=sq, oi=oi, entries=e, exits=x, symbol="T",
                max_trades=10_000_000, entry_cooldown_ticks=10),
            reps)
        trades = res.trades()
        last_exit = trades[-1].exit_idx if trades else -1
        rows[n] = {"best_ms": best * 1e3, "m_ticks_per_s": n / best / 1e6,
                   "trades": len(trades), "last_exit_idx": last_exit,
                   "reached_end": bool(last_exit >= n - 100)}
    return rows


def _spread_items(k, n_bars, seed=9):
    g = np.random.default_rng(seed)
    items = []
    for i in range(k):
        prem = [np.abs(30 + 8 * np.sin(np.arange(n_bars) / 97 + i) + g.normal(0, 0.6, n_bars)),
                np.abs(18 + 5 * np.cos(np.arange(n_bars) / 113 + i) + g.normal(0, 0.5, n_bars))]
        e = np.zeros(n_bars, bool)
        x = np.zeros(n_bars, bool)
        e[np.arange(20 + i % 7, n_bars, 200)] = True
        x[np.arange(120 + i % 7, n_bars, 200)] = True
        items.append(r.BatchSpreadItem(
            f"spread_{i}", prem, [("CE", 1000.0, -1, 75), ("CE", 1050.0, 1, 75)],
            e, x, "custom", None, None))
    return items


def bench_spreads():
    n_bars = 2000
    rng = np.random.default_rng(5)
    ts = np.arange(n_bars, dtype=np.int64) * 60_000_000_000
    und = 1000 * np.exp(np.clip(np.cumsum(rng.normal(0, 0.0008, n_bars)), -0.4, 0.4))

    items = _spread_items(500, n_bars)
    t0 = time.perf_counter()
    par = r.batch_spread_backtest(ts, und, items)
    t_par = time.perf_counter() - t0

    t0 = time.perf_counter()
    ser = [r.batch_spread_backtest(ts, und, [it])[0] for it in _spread_items(500, n_bars)]
    t_ser = time.perf_counter() - t0

    pv = [round(a[1].metrics.total_return_pct, 12) for a in par]
    sv = [round(a[1].metrics.total_return_pct, 12) for a in ser]
    return {"per_sec": 500 / t_par, "speedup": t_ser / t_par,
            "serial_ms": t_ser * 1e3, "parallel_ms": t_par * 1e3,
            "matches_serial": pv == sv, "unique_returns": len(set(pv))}


def bench_determinism():
    ts, o, h, l, c, v = bars(50_000)
    e, x = sma_signals(c)
    cfg = _cfg()

    def digest():
        res = r.run_single_backtest(
            timestamps=ts, open=o, high=h, low=l, close=c, volume=v,
            entries=e, exits=x, direction=1, weight=1.0, symbol="B", config=cfg)
        payload = json.dumps({
            "eq": [round(float(z), 10) for z in np.asarray(res.equity_curve())],
            "tr": [[t.entry_idx, t.exit_idx, round(t.pnl, 10)] for t in res.trades()],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    hashes = {digest() for _ in range(20)}
    return {"runs": 20, "unique_hashes": len(hashes), "sha256": sorted(hashes)[0]}


def bench_sweep():
    n = 93_750
    ts, o, h, l, c, v = bars(n)
    cfg = _cfg()

    def sma(a, w):
        cs = np.cumsum(np.insert(a, 0, 0.0))
        out = np.full(len(a), np.nan)
        out[w - 1:] = (cs[w:] - cs[:-w]) / w
        return out

    combos = [(f, s) for f in range(5, 55, 5) for s in range(20, 220, 10) if f < s]
    t0 = time.perf_counter()
    engine = 0.0
    for f, s in combos:
        ff, ss = sma(c, f), sma(c, s)
        up = (ff > ss) & ~np.isnan(ff) & ~np.isnan(ss)
        e = np.zeros(n, bool)
        x = np.zeros(n, bool)
        e[1:] = up[1:] & ~up[:-1]
        x[1:] = ~up[1:] & up[:-1]
        a = time.perf_counter()
        r.run_single_backtest(timestamps=ts, open=o, high=h, low=l, close=c, volume=v,
                              entries=e, exits=x, direction=1, weight=1.0,
                              symbol="B", config=cfg)
        engine += time.perf_counter() - a
    return {"combos": len(combos), "bars": n, "bar_evals": len(combos) * n,
            "wall_s": time.perf_counter() - t0, "engine_s": engine,
            # ru_maxrss is a whole-process high-water mark, so this is only
            # meaningful when the sweep runs alone: `python run_all.py sweep`.
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576}


def main():
    so = pathlib.Path(r.__file__).parent / f"_raptorbt.cpython-{sys.version_info.major}{sys.version_info.minor}-darwin.so"
    ts, o, h, l, c, v = bars(1_000)
    e, x = sma_signals(c)
    m = r.run_single_backtest(timestamps=ts, open=o, high=h, low=l, close=c, volume=v,
                              entries=e, exits=x, direction=1, weight=1.0,
                              symbol="B", config=_cfg()).metrics
    metrics = [a for a in dir(m) if not a.startswith("_") and not callable(getattr(m, a))]

    out = {
        "version": r.__version__,
        "env": {"cpu": platform.processor() or platform.machine(),
                "python": platform.python_version(), "machine": platform.machine()},
        "bars": bench_bars(),
        "ticks": bench_ticks(),
        "spreads": bench_spreads(),
        "determinism": bench_determinism(),
        "sweep": bench_sweep(),
        "metrics": {"attributes": len(metrics), "to_dict_keys": len(m.to_dict())},
        "engine_mb": so.stat().st_size / 1048576 if so.exists() else None,
    }
    dest = pathlib.Path(__file__).parent / "results.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


SECTIONS = {
    "bars": bench_bars,
    "ticks": bench_ticks,
    "spreads": bench_spreads,
    "determinism": bench_determinism,
    "sweep": bench_sweep,
}


if __name__ == "__main__":
    # Run one section alone when its figure is sensitive to what ran before it:
    # `spreads` competes for the same cores Rayon wants, and `sweep`'s peak-RSS
    # reading is a whole-process high-water mark. Both are measured standalone
    # for the published tables.
    if len(sys.argv) > 1 and sys.argv[1] in SECTIONS:
        print(json.dumps({sys.argv[1]: SECTIONS[sys.argv[1]]()}, indent=2))
    else:
        main()
