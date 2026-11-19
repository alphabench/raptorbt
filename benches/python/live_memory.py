"""Memory of one ``LiveNode`` as the feed grows.

Every configuration runs in a fresh interpreter, twice:

* the RSS pass: RSS is read once raptorbt and numpy are imported (the baseline), then the
  node is built, fed, and RSS is read again; reported are the current RSS over the
  baseline and the process's peak RSS (``getrusage``) over it, with
  ``node.memory_report()`` (the Rust heap by category, as the engine counts it);
* the tracemalloc pass: the same run under ``tracemalloc`` (Python allocations only; it
  slows the run and inflates RSS, so it never shares a pass with the RSS reading). "now" is
  what Python holds after the last push; "peak" also counts one chunk of the generated
  tape (under 1 MB).

The tape is generated in chunks while it is pushed, so the data itself stays a few MB
and does not show up as node memory. One strategy (the momentum strategy of
``live_latency.py``, trading every symbol) runs on every node, with 1-minute bars
aggregated from the ticks.

Matrix:

* ``symbols``: 1 / 10 / 50 / 200 symbols x ``history`` default / 512, 20,000 ticks per
  symbol (more than the default history, so every ring has wrapped);
* ``depth``: the same symbol counts with a 5-level depth snapshot pushed after every 10th
  tick;
* ``session``: 100k / 1M / 5M ticks on one symbol, ``curve_samples`` None / 1,000, to
  show what grows per million ticks.

    python live_memory.py
    python live_memory.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _live  # noqa: E402
from raptorbt import LiveNode  # noqa: E402

CHUNK = 8_192


def feed(n: int, symbols: list[str], seed: int = 3):
    """Yield ``(symbol_index, ts, ltp)`` chunks of a round-robin tape over one session,
    generated as it is consumed. All symbols follow one mean-reverting walk, scaled."""
    k = len(symbols)
    step = _live.SESSION_NS // n
    start = _live.day_ns(0) + _live.SESSION_OPEN_NS
    per_symbol = -(-n // k)
    vol = 0.02 / np.sqrt(per_symbol)
    keep = 1.0 - min(0.5, (vol / 0.01) ** 2 / 2)
    rng = np.random.default_rng(seed)
    scale = 1.0 + 0.01 * np.arange(k)
    acc = 0.0
    for lo in range(0, n, CHUNK):
        m = min(CHUNK, n - lo)
        shocks = rng.normal(0.0, vol, m)
        x = np.empty(m)
        for i in range(m):
            acc = acc * keep + shocks[i]
            x[i] = acc
        idx = np.arange(lo, lo + m)
        sym = idx % k
        ltp = np.round(100.0 * scale[sym] * np.exp(x) / _live.TICK) * _live.TICK
        ts = start + idx.astype(np.int64) * step
        yield sym.tolist(), ts.tolist(), ltp.tolist()


def child(cfg: dict) -> dict:
    """One configuration, in this (fresh) process."""
    gc.collect()
    base_rss = _live.rss_bytes()
    base_peak = _live.peak_rss_bytes()
    if cfg["tracemalloc"]:
        tracemalloc.start(1)

    symbols = [f"S{i}" for i in range(cfg["symbols"])]
    strategy = _live.MomentumTick()
    node = LiveNode(
        [strategy], symbols, _live.config(), primary_bars=(1, "m"),
        history=cfg["history"], curve_samples=cfg["curve_samples"],
    )
    built_rss = None if cfg["tracemalloc"] else _live.rss_bytes()
    n = cfg["ticks"]
    depth_every = cfg["depth_every"]
    push, push_depth = node.push_tick, node.push_depth
    count = 0
    for sym, ts, ltp in feed(n, symbols):
        for j, t, p in zip(sym, ts, ltp):
            s = symbols[j]
            push(s, t, p, p - _live.TICK, p + _live.TICK)
            count += 1
            if depth_every and count % depth_every == 0:
                bids = [(p - _live.TICK * (1 + lvl), 100.0 * (1 + lvl)) for lvl in range(5)]
                asks = [(p + _live.TICK * (1 + lvl), 100.0 * (1 + lvl)) for lvl in range(5)]
                push_depth(s, t, bids, asks)

    report = node.memory_report()
    row = {"memory_report": report, "prints_seen": strategy.prints, "fills": strategy.fills}
    if cfg["tracemalloc"]:
        current, peak = tracemalloc.get_traced_memory()
        top = tracemalloc.take_snapshot().statistics("filename")[:5]
        tracemalloc.stop()
        row.update({
            "traced_current_bytes": current,
            "traced_peak_bytes": peak,
            "traced_top": [{"file": str(s.traceback[0].filename), "bytes": s.size, "blocks": s.count} for s in top],
        })
    else:
        gc.collect()
        rss = _live.rss_bytes()
        row.update({
            "baseline_rss_bytes": base_rss,
            "built_rss_delta_bytes": built_rss - base_rss,
            "rss_delta_bytes": rss - base_rss,
            "peak_rss_delta_bytes": _live.peak_rss_bytes() - base_peak,
        })
    if strategy.prints != n:
        row["error"] = f"check failed: the strategy saw {strategy.prints} prints of {n}"
    return row


def matrix(quick: bool) -> list[dict]:
    if quick:
        per, counts, sessions = 1_500, [1, 10], [5_000, 20_000]
    else:
        per, counts, sessions = 20_000, [1, 10, 50, 200], [100_000, 1_000_000, 5_000_000]
    out = []
    for k in counts:
        for history in (None, 512):
            out.append({"group": "symbols", "symbols": k, "ticks": per * k, "history": history,
                        "curve_samples": None, "depth_every": 0})
    for k in counts:
        out.append({"group": "depth", "symbols": k, "ticks": per * k, "history": None,
                    "curve_samples": None, "depth_every": 10})
    for n in sessions:
        for cs in (None, 1_000):
            out.append({"group": "session", "symbols": 1, "ticks": n, "history": None,
                        "curve_samples": cs, "depth_every": 0})
    return out


def name_of(cfg: dict) -> str:
    h = "default" if cfg["history"] is None else cfg["history"]
    cs = "none" if cfg["curve_samples"] is None else cfg["curve_samples"]
    return (f"{cfg['group']}/sym={cfg['symbols']}/ticks={cfg['ticks']}/history={h}"
            f"/curve_samples={cs}/depth={'on' if cfg['depth_every'] else 'off'}")


def run(cfg: dict) -> dict:
    rss = _live.run_child(__file__, [json.dumps({**cfg, "tracemalloc": False})])
    traced = _live.run_child(__file__, [json.dumps({**cfg, "tracemalloc": True})])
    mb = 1 / 1048576
    row = {
        **cfg,
        "rss_delta_mb": rss["rss_delta_bytes"] * mb,
        "peak_rss_delta_mb": rss["peak_rss_delta_bytes"] * mb,
        "built_rss_delta_mb": rss["built_rss_delta_bytes"] * mb,
        "rss_per_symbol_kb": rss["rss_delta_bytes"] / cfg["symbols"] / 1024,
        "engine_total_mb": rss["memory_report"].get("total", 0) * mb,
        "engine_rings_mb": rss["memory_report"].get("live_rings", 0) * mb,
        "engine_curves_mb": rss["memory_report"].get("curves", 0) * mb,
        "traced_peak_mb": traced["traced_peak_bytes"] * mb,
        "traced_current_mb": traced["traced_current_bytes"] * mb,
        "fills": rss["fills"],
        "rss_pass": rss,
        "tracemalloc_pass": traced,
    }
    errors = [p["error"] for p in (rss, traced) if "error" in p]
    if errors:
        row["error"] = "; ".join(errors)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="a tiny run to check the script works")
    p.add_argument("--json", default=None)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child is not None:
        print(json.dumps(child(json.loads(args.child))))
        return

    environment = _live.print_env("LiveNode memory")
    print(f"RSS via {'psutil' if _live.psutil is not None else 'ps -o rss='}; peak via getrusage; "
          f"one fresh interpreter per pass\n")
    results: list[dict] = []
    for cfg in matrix(args.quick):
        _live.run_case(name_of(cfg), lambda: run(cfg), results)
        r = results[-1]
        if "rss_delta_mb" in r:
            print(f"   RSS +{r['rss_delta_mb']:.1f} MB (peak +{r['peak_rss_delta_mb']:.1f})  "
                  f"engine {r['engine_total_mb']:.1f} MB  Python {r['traced_current_mb']:.1f} MB", flush=True)

    print()
    _live.table(results, [
        ("case", "case"), ("rss_delta_mb", "RSS+ MB"), ("peak_rss_delta_mb", "peak+ MB"),
        ("rss_per_symbol_kb", "RSS/sym KB"), ("engine_total_mb", "engine MB"),
        ("engine_rings_mb", "rings MB"), ("engine_curves_mb", "curves MB"),
        ("traced_current_mb", "py now MB"), ("traced_peak_mb", "py peak MB"), ("fills", "fills"),
    ])
    _live.write_json(args.json or _live.default_json(__file__), {
        "script": "live_memory",
        "env": environment,
        "params": {"quick": args.quick, "capital": _live.CAPITAL, "quantity": _live.QUANTITY},
        "cases": results,
    })


if __name__ == "__main__":
    main()
