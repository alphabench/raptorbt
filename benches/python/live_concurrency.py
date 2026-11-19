"""Many independent ``LiveNode`` objects: in one process, in many processes, in threads.

Every node runs one momentum strategy (the one in ``live_latency.py``) on its own symbol
``S0``, fed ticks it aggregates into 1-minute bars. One number per push is the time of
``node.push_tick(...)``, call to return, as in ``live_latency.py``.

Modes:

* ``inproc``: N nodes (1, 10, 50, 100, 200) in ONE process, fed round-robin (tick i to
  every node, then tick i+1), each in a fresh interpreter so RSS is clean. Reports total
  pushes/s, per-push p50 / p99 / p99.9 and RSS per node (RSS over the baseline, divided
  by N). Every node reads the same tape, so the tape is held once.
* ``procs``: N processes (1, 2, 4, 8 and ``os.cpu_count()``), each with one node,
  started with ``multiprocessing`` spawn and released together by a barrier once their
  nodes and tapes are built. Reports aggregate ticks/s (all ticks over the span from the
  first start to the last finish), each process's p99 and the sum of the children's RSS
  (pages shared between them, such as the extension's code, are counted once per child).
* ``threads``: N threads (1, 2, 4, 8), each building its own node inside the thread,
  released together by a barrier, to show what the interpreter lock allows. A node that
  cannot be used this way is reported as "not supported".

    python live_concurrency.py
    python live_concurrency.py --quick
    python live_concurrency.py --modes procs
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pathlib
import queue as queue_mod
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _live  # noqa: E402
from raptorbt import LiveNode  # noqa: E402


def tape_lists(n: int, seed: int = 11):
    t = _live.ticks(n, seed=seed)
    return t["timestamps"].tolist(), t["ltp"].tolist(), t["bid"].tolist(), t["ask"].tolist()


def new_node():
    strategy = _live.MomentumTick()
    return strategy, LiveNode([strategy], ["S0"], _live.config(), primary_bars=(1, "m"))


def check(strategy, node, n: int) -> list[str]:
    problems = []
    if strategy.prints != n:
        problems.append(f"a strategy saw {strategy.prints} prints of {n}")
    if strategy.fills == 0:
        problems.append("a strategy made no fill")
    return problems


# -- one process, many nodes -------------------------------------------------------


def inproc_child(n_nodes: int, ticks: int) -> dict:
    ts, ltp, bid, ask = tape_lists(ticks)
    gc.collect()
    base = _live.rss_bytes()
    pairs = [new_node() for _ in range(n_nodes)]
    gc.collect()
    built = _live.rss_bytes()
    pushes = [node.push_tick for _, node in pairs]
    samples = np.empty(ticks * n_nodes, dtype=np.int64)
    clock = time.perf_counter_ns
    k = 0
    wall0 = clock()
    for i in range(ticks):
        a, b, c, d = ts[i], ltp[i], bid[i], ask[i]
        for push in pushes:
            t0 = clock()
            push("S0", a, b, c, d)
            samples[k] = clock() - t0
            k += 1
    wall = clock() - wall0
    gc.collect()
    after = _live.rss_bytes()
    problems = [p for s, node in pairs for p in check(s, node, ticks)]
    warm = (ticks // 20) * n_nodes
    row = {
        "nodes": n_nodes,
        "ticks_per_node": ticks,
        "pushes": ticks * n_nodes,
        "pushes_per_s": ticks * n_nodes / (wall / 1e9),
        "wall_s": wall / 1e9,
        **_live.percentiles(samples[warm:]),
        "rss_built_per_node_kb": (built - base) / n_nodes / 1024,
        "rss_per_node_kb": (after - base) / n_nodes / 1024,
        "rss_delta_mb": (after - base) / 1048576,
        "memory_report_node0": pairs[0][1].memory_report(),
        "fills_node0": pairs[0][0].fills,
    }
    if problems:
        row["error"] = "check failed: " + "; ".join(sorted(set(problems)))
    return row


def run_inproc(n_nodes: int, ticks: int) -> dict:
    return _live.run_child(__file__, [json.dumps({"mode": "inproc", "nodes": n_nodes, "ticks": ticks})])


# -- many processes, one node each ---------------------------------------------------


def proc_worker(ticks: int, seed: int, barrier, queue) -> None:
    try:
        ts, ltp, bid, ask = tape_lists(ticks, seed)
        strategy, node = new_node()
        push = node.push_tick
        samples = np.empty(ticks, dtype=np.int64)
        clock = time.perf_counter_ns
        gc.collect()
        barrier.wait()
        start = time.time_ns()
        p0 = clock()
        for i in range(ticks):
            a, b, c, d = ts[i], ltp[i], bid[i], ask[i]
            t0 = clock()
            push("S0", a, b, c, d)
            samples[i] = clock() - t0
        elapsed = clock() - p0
        end = time.time_ns()
        stats = _live.percentiles(samples[ticks // 20:])
        problems = check(strategy, node, ticks)
        queue.put({"pid": os.getpid(), "start_ns": start, "end_ns": end, "elapsed_ns": elapsed,
                   "ticks": ticks, "p50_ns": stats["p50_ns"], "p99_ns": stats["p99_ns"],
                   "p999_ns": stats["p999_ns"], "rss_bytes": _live.rss_bytes(),
                   "peak_rss_bytes": _live.peak_rss_bytes(), "problems": problems})
    except BaseException as e:
        try:
            barrier.abort()
        except Exception:
            pass
        queue.put({"pid": os.getpid(), "error": f"{type(e).__name__}: {e}"})


def run_procs(n: int, ticks: int) -> dict:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n, timeout=600)  # a child that dies before it breaks the barrier for the rest
    queue = ctx.Queue()
    procs = [ctx.Process(target=proc_worker, args=(ticks, 100 + j, barrier, queue)) for j in range(n)]
    for p in procs:
        p.start()
    out = []
    while len(out) < n:
        try:
            out.append(queue.get(timeout=1))
        except queue_mod.Empty:
            if not any(p.is_alive() for p in procs) and queue.empty():
                break
    for p in procs:
        p.join()
    if len(out) < n:
        codes = [p.exitcode for p in procs]
        return {"processes": n, "error": f"{n - len(out)} children died without a result (exit codes {codes})"}
    errors = [o["error"] for o in out if "error" in o]
    if errors:
        return {"processes": n, "error": "; ".join(sorted(set(errors)))}
    span = max(o["end_ns"] for o in out) - min(o["start_ns"] for o in out)
    total = sum(o["ticks"] for o in out)
    p99 = [o["p99_ns"] for o in out]
    row = {
        "processes": n,
        "ticks_per_process": ticks,
        "ticks_per_s": total / (span / 1e9),
        "ticks_per_s_per_process": float(np.median([o["ticks"] / (o["elapsed_ns"] / 1e9) for o in out])),
        "span_s": span / 1e9,
        "p50_ns_median": float(np.median([o["p50_ns"] for o in out])),
        "p99_ns_median": float(np.median(p99)),
        "p99_ns_worst": float(max(p99)),
        "p999_ns_worst": float(max(o["p999_ns"] for o in out)),
        "rss_total_mb": sum(o["rss_bytes"] for o in out) / 1048576,
        "rss_per_process_mb": float(np.median([o["rss_bytes"] for o in out])) / 1048576,
        "children": out,
    }
    problems = sorted({p for o in out for p in o["problems"]})
    if problems:
        row["error"] = "check failed: " + "; ".join(problems)
    return row


# -- threads, one node each ----------------------------------------------------------


def run_threads(n: int, ticks: int) -> dict:
    ts, ltp, bid, ask = tape_lists(ticks)
    barrier = threading.Barrier(n)
    out: list[dict] = [{} for _ in range(n)]

    def worker(j: int) -> None:
        try:
            strategy, node = new_node()  # built inside the thread that uses it
            push = node.push_tick
            samples = np.empty(ticks, dtype=np.int64)
            clock = time.perf_counter_ns
            barrier.wait()
            start = clock()
            for i in range(ticks):
                a, b, c, d = ts[i], ltp[i], bid[i], ask[i]
                t0 = clock()
                push("S0", a, b, c, d)
                samples[i] = clock() - t0
            end = clock()
            out[j] = {"start": start, "end": end, "p99_ns": _live.percentiles(samples[ticks // 20:])["p99_ns"],
                      "problems": check(strategy, node, ticks)}
        except BaseException as e:
            try:
                barrier.abort()
            except Exception:
                pass
            out[j] = {"error": f"{type(e).__name__}: {e}"}

    threads = [threading.Thread(target=worker, args=(j,)) for j in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errors = sorted({o["error"] for o in out if "error" in o})
    if errors:
        return {"threads": n, "supported": False, "error": "not supported: " + "; ".join(errors)}
    span = max(o["end"] for o in out) - min(o["start"] for o in out)
    row = {
        "threads": n,
        "supported": True,
        "ticks_per_thread": ticks,
        "ticks_per_s": n * ticks / (span / 1e9),
        "span_s": span / 1e9,
        "p99_ns_worst": float(max(o["p99_ns"] for o in out)),
        "gil_disabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)() is False),
    }
    problems = sorted({p for o in out for p in o["problems"]})
    if problems:
        row["error"] = "check failed: " + "; ".join(problems)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", default="inproc,procs,threads")
    p.add_argument("--ticks-per-node", type=int, default=20_000, help="inproc: ticks each node gets")
    p.add_argument("--ticks-per-process", type=int, default=200_000)
    p.add_argument("--ticks-per-thread", type=int, default=100_000)
    p.add_argument("--quick", action="store_true", help="a tiny run to check the script works")
    p.add_argument("--json", default=None)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child is not None:
        cfg = json.loads(args.child)
        print(json.dumps(inproc_child(cfg["nodes"], cfg["ticks"])))
        return

    cpus = os.cpu_count() or 1
    if args.quick:
        node_counts, proc_counts, thread_counts = [1, 5], [1, 2], [1, 2]
        args.ticks_per_node = args.ticks_per_process = args.ticks_per_thread = 2_000
    else:
        node_counts = [1, 10, 50, 100, 200]
        proc_counts = sorted({1, 2, 4, 8, cpus})
        thread_counts = [1, 2, 4, 8]
    modes = args.modes.split(",")

    environment = _live.print_env("LiveNode concurrency")
    print(f"capital {_live.CAPITAL:,.0f} per node, orders of {_live.QUANTITY} units, no fees\n")
    out: dict[str, list] = {"inproc": [], "procs": [], "threads": []}

    if "inproc" in modes:
        for n in node_counts:
            _live.run_case(f"inproc/nodes={n}", lambda: run_inproc(n, args.ticks_per_node), out["inproc"])
        print()
        _live.table(out["inproc"], [
            ("case", "case"), ("pushes_per_s", "pushes/s"), ("p50_ns", "p50 ns"), ("p99_ns", "p99 ns"),
            ("p999_ns", "p99.9 ns"), ("rss_per_node_kb", "RSS/node KB"), ("rss_delta_mb", "RSS+ MB"),
        ])
        print()
    if "procs" in modes:
        for n in proc_counts:
            _live.run_case(f"procs/n={n}", lambda: run_procs(n, args.ticks_per_process), out["procs"])
        print()
        _live.table(out["procs"], [
            ("case", "case"), ("ticks_per_s", "ticks/s"), ("ticks_per_s_per_process", "ticks/s/proc"),
            ("p99_ns_median", "p99 ns med"), ("p99_ns_worst", "p99 ns worst"),
            ("rss_total_mb", "RSS sum MB"), ("rss_per_process_mb", "RSS/proc MB"),
        ])
        print()
    if "threads" in modes:
        for n in thread_counts:
            _live.run_case(f"threads/n={n}", lambda: run_threads(n, args.ticks_per_thread), out["threads"])
        print()
        _live.table(out["threads"], [
            ("case", "case"), ("ticks_per_s", "ticks/s"), ("p99_ns_worst", "p99 ns worst"), ("supported", "supported"),
        ])

    _live.write_json(args.json or _live.default_json(__file__), {
        "script": "live_concurrency",
        "env": environment,
        "params": {"ticks_per_node": args.ticks_per_node, "ticks_per_process": args.ticks_per_process,
                   "ticks_per_thread": args.ticks_per_thread, "capital": _live.CAPITAL,
                   "quantity": _live.QUANTITY, "quick": args.quick},
        **out,
    })


if __name__ == "__main__":
    main()
