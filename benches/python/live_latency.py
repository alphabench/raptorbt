"""Per-tick latency of ``LiveNode.push_tick``.

One number is one call: ``perf_counter_ns()`` immediately before ``node.push_tick(...)``
and immediately after it returns. Inside that region: the Python wrapper's checks, the
Rust core applying the print and the quote, the 1-minute aggregation, every strategy hook
the tick triggers (Python), the orders they submit being matched in-process, stops and
trailing stops being checked, and the journal record being written when there is one.
Outside it: converting the inputs, storing the sample, the loop itself.

Matrix (each a named case ``<hook>/<journal>/<strategies>``):

* hook ``bars``: the strategy defines ``on_bar`` only (no tick hook), ``noop``: an
  ``on_trade`` that only counts, ``momentum``: an ``on_trade`` that reads a 1-minute SMA,
  buys a fixed 10 units on a momentum condition with a fixed stop, and holds a trailing
  stop from the config;
* journal ``none``, ``file`` (a file in a temporary directory, fsync policy ``--fsync``,
  default ``orders``) and ``memory`` (the latest ``--journal-memory`` records kept);
* 1, 10 and 50 strategies on one node, one symbol.

    python live_latency.py                 # 1,000,000 ticks per case
    python live_latency.py --quick         # a few thousand ticks, to check it runs
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _live  # noqa: E402
from raptorbt import LiveNode  # noqa: E402
from raptorbt.journal import Journal  # noqa: E402

HOOKS = {"bars": _live.BarsOnly, "noop": _live.NoopTick, "momentum": _live.MomentumTick}


def make_journal(kind: str, args, tmpdir: str, name: str):
    if kind == "none":
        return None, None
    header = {"bench": "live_latency", "case": name}
    if kind == "file":
        path = pathlib.Path(tmpdir) / (name.replace("/", "_") + ".journal")
        return Journal(header, file=str(path), fsync=args.fsync), path
    if kind == "memory":
        return Journal(header, memory=args.journal_memory), None
    raise ValueError(f"unknown journal kind {kind!r}")


def run(hook: str, journal_kind: str, k: int, tape: dict, warm: int, args, tmpdir: str) -> dict:
    name = f"{hook}/{journal_kind}/{k}"
    ts, ltp, bid, ask = tape["ts"], tape["ltp"], tape["bid"], tape["ask"]
    n = len(ts)
    strategies = [HOOKS[hook]() for _ in range(k)]
    journal, path = make_journal(journal_kind, args, tmpdir, name)
    node = LiveNode(strategies, ["S0"], _live.config(), primary_bars=(1, "m"), journal=journal,
                    curve_samples=args.curve_samples)

    samples = np.empty(n, dtype=np.int64)
    push = node.push_tick
    clock = time.perf_counter_ns
    events = 0
    gc.collect()
    wall0 = clock()
    for i in range(n):
        a, b, c, d = ts[i], ltp[i], bid[i], ask[i]
        t0 = clock()
        e = push("S0", a, b, c, d)
        samples[i] = clock() - t0
        events += e
    wall = clock() - wall0

    memory = node.memory_report()
    now_ns = node.now_ns
    results = node.finish()
    records = journal.count if journal is not None else 0

    totals = {"orders": 0, "filled_orders": 0, "trades": 0}
    reasons: dict[str, int] = {}
    for r in results:
        s = _live.trade_summary(r)
        for key in totals:
            totals[key] += s[key]
        for key, v in s["exit_reasons"].items():
            reasons[key] = reasons.get(key, 0) + v

    checks = []
    if events != 2 * n:  # every row is a print and a two-sided quote
        checks.append(f"{events} events from {n} ticks, expected {2 * n}")
    if now_ns != ts[-1]:
        checks.append(f"node clock {now_ns} != last tick {ts[-1]}")
    for j, s in enumerate(strategies):
        if hook == "bars":
            if s.bars < 300:  # a session of 390 minutes closes 389 or 390 bars
                checks.append(f"strategy {j} saw {s.bars} bars")
        elif s.prints != n:
            checks.append(f"strategy {j} saw {s.prints} prints of {n}")
    if hook == "momentum":
        per = [_live.trade_summary(r)["trades"] for r in results]
        if min(per) == 0:
            checks.append(f"a momentum strategy made no trade (trades per strategy {per})")

    stats = _live.percentiles(samples[warm:])
    row = {
        "hook": hook,
        "journal": journal_kind,
        "strategies": k,
        "ticks": n,
        "warmup_ticks": warm,
        **stats,
        "p50_us": stats["p50_ns"] / 1e3,
        "p90_us": stats["p90_ns"] / 1e3,
        "p99_us": stats["p99_ns"] / 1e3,
        "p999_us": stats["p999_ns"] / 1e3,
        "max_us": stats["max_ns"] / 1e3,
        "ticks_per_s": n / (wall / 1e9),
        "wall_s": wall / 1e9,
        "events": events,
        **totals,
        "exit_reasons": reasons,
        "journal_records": records,
        "journal_file_bytes": path.stat().st_size if path is not None and path.exists() else None,
        "memory_report": memory,
    }
    if hook == "momentum":
        row["position_open_fraction"] = sum(s.open_prints for s in strategies) / (k * n)
    if checks:
        row["error"] = "check failed: " + "; ".join(checks)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticks", type=int, default=1_000_000)
    p.add_argument("--warmup", type=int, default=None, help="ticks excluded from the stats (default 5%%)")
    p.add_argument("--strategies", default="1,10,50")
    p.add_argument("--hooks", default="bars,noop,momentum")
    p.add_argument("--journals", default="none,file,memory")
    p.add_argument("--fsync", default="orders", help="file journal fsync policy: every, orders, never or an int")
    p.add_argument("--journal-memory", type=int, default=100_000, help="records the memory journal keeps")
    p.add_argument("--curve-samples", type=int, default=None,
                   help="bound each strategy's equity curve (default: unbounded, the node's default; "
                        "about 32 bytes per tick per strategy, so 50 strategies x 1M ticks hold about 1.6 GB)")
    p.add_argument("--quick", action="store_true", help="a tiny run to check the script works")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if args.fsync.isdigit():
        args.fsync = int(args.fsync)
    if args.quick:
        args.ticks, args.strategies = 4_000, "1,3"
    warm = args.warmup if args.warmup is not None else args.ticks // 20
    counts = [int(x) for x in args.strategies.split(",")]
    hooks = args.hooks.split(",")
    journals = args.journals.split(",")

    environment = _live.print_env("LiveNode.push_tick latency")
    print(f"capital {_live.CAPITAL:,.0f} per strategy, orders of {_live.QUANTITY} units, no fees")
    print(f"{args.ticks:,} ticks per case over one 390-minute session, first {warm:,} excluded")
    print(f"gc enabled: {gc.isenabled()}  thresholds {gc.get_threshold()}  curve_samples {args.curve_samples}")

    overhead = _live.timer_overhead(min(args.ticks, 1_000_000))
    print(f"timer overhead: p50 {overhead['p50_ns']:.0f} ns  p99 {overhead['p99_ns']:.0f} ns  "
          f"max {overhead['max_ns']:.0f} ns (subtract from per-call figures)\n")

    raw = _live.ticks(args.ticks, seed=11)
    tape = {
        "ts": raw["timestamps"].tolist(),
        "ltp": raw["ltp"].tolist(),
        "bid": raw["bid"].tolist(),
        "ask": raw["ask"].tolist(),
    }

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="rbt-latency-") as tmpdir:
        for hook in hooks:
            for journal in journals:
                for k in counts:
                    _live.run_case(
                        f"{hook}/{journal}/{k}",
                        lambda: run(hook, journal, k, tape, warm, args, tmpdir),
                        results,
                    )
                    r = results[-1]
                    if "p50_ns" in r:
                        print(f"   p50 {r['p50_us']:.2f} us  p99 {r['p99_us']:.2f} us  "
                              f"{r['ticks_per_s']:,.0f} ticks/s  trades {r['trades']}", flush=True)

    print()
    _live.table(results, [
        ("case", "case"), ("p50_us", "p50 us"), ("p90_us", "p90 us"), ("p99_us", "p99 us"),
        ("p999_us", "p99.9 us"), ("max_us", "max us"), ("ticks_per_s", "ticks/s"),
        ("orders", "orders"), ("filled_orders", "filled"), ("trades", "trades"),
        ("journal_records", "journal recs"),
    ])
    _live.write_json(args.json or _live.default_json(__file__), {
        "script": "live_latency",
        "env": environment,
        "params": {"ticks": args.ticks, "warmup": warm, "capital": _live.CAPITAL,
                   "quantity": _live.QUANTITY, "fsync": args.fsync,
                   "journal_memory": args.journal_memory, "curve_samples": args.curve_samples,
                   "quick": args.quick},
        "timer_overhead": overhead,
        "cases": results,
    })


if __name__ == "__main__":
    main()
