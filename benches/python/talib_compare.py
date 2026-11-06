"""TA-Lib against raptorbt on the speed bar, one process per library.

TA-Lib and a raptorbt build rarely share an environment (the extension is built in place), so
each library is timed in its own process and the results meet in a JSON file:

    # 1. TA-Lib (single-threaded by nature; also a thread-pool run for the instruments case)
    uv run --no-project --python 3.12 --with ta-lib==0.6.4 --with numpy \\
        python benches/python/talib_compare.py talib --out /tmp/talib.json

    # 2. raptorbt, single-threaded then parallel (rayon reads the variable at import)
    uv run maturin develop --release
    python benches/python/talib_compare.py raptorbt --threads 1 --out /tmp/rb1.json
    python benches/python/talib_compare.py raptorbt --threads 0 --out /tmp/rbN.json   # 0 = all cores

    # 3. merge, print the verdicts, write docs/quantitative-correctness/speed.md
    python benches/python/talib_compare.py report /tmp/talib.json /tmp/rb1.json /tmp/rbN.json \\
        --json /tmp/speed.json --md docs/quantitative-correctness/speed.md

Speed bar (single-threaded, single series): within 1.1x of TA-Lib for 90% of functions and
within 1.5x for all. A function's ratio is its worst case across sizes and periods.
Time is the median of several runs after one warm-up; ratio = raptorbt / TA-Lib, so below 1.0 is faster.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORACLE = ROOT / "tests" / "oracle" / "talib_functions.json"
SIZES = (1_000_000, 10_000_000)
PERIODS = (14, 200)
INSTRUMENTS = 500
SHARED_TERMS = 20
BAR_ONE = 1.1
BAR_ALL = 1.5
BAR_SHARE = 0.9

FIELD_OF = {  # TA-Lib input name -> bar column
    "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
    "real": "close", "real0": "high", "real1": "low", "periods": "close",
}


def runs_for(n: int) -> int:
    return 7 if n <= 1_000_000 else 3


def bench(fn, runs: int) -> float:
    """Median seconds of `runs` calls after one warm-up call."""
    fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bars(n: int):
    import numpy as np

    sys.path.insert(0, str(HERE))
    from _data import bars as make  # noqa: PLC0415

    _, o, h, l, c, v = make(n)
    return {k: np.ascontiguousarray(a, dtype=np.float64) for k, a in
            zip(("open", "high", "low", "close", "volume"), (o, h, l, c, v))}


def machine(threads=None) -> dict:
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "threads": threads,
    }


def oracle() -> dict:
    return json.loads(ORACLE.read_text())["functions"]


def cases_for(name: str, spec: dict):
    """(period or None, kwargs) per TA-Lib call: periods 14 and 200 where a period applies."""
    params = spec["parameters"]
    period_keys = [k for k in params if "period" in k]
    if not period_keys:
        return [(None, {})]
    out = []
    for p in PERIODS:
        kw = {k: p for k in period_keys[:1]}
        # a second period parameter (slow/signal) keeps its default unless it would be shorter
        out.append((p, kw))
    return out


# ------------------------------------------------------------------ TA-Lib side


def run_talib(args) -> None:
    import numpy as np
    import talib
    from concurrent.futures import ThreadPoolExecutor

    result = {"library": "talib", "version": talib.__version__, "machine": machine(1),
              "series": [], "cases": []}
    funcs = oracle()
    for n in SIZES:
        data = bars(n)
        for name, spec in funcs.items():
            fn = getattr(talib, name, None)
            if fn is None:
                continue
            ins = [data[FIELD_OF.get(i, "close")] for i in spec["inputs"]]
            for period, kw in cases_for(name, spec):
                try:
                    t = bench(lambda: fn(*ins, **kw), runs_for(n))
                except Exception as e:  # noqa: BLE001
                    print(f"talib {name}: {e}", file=sys.stderr)
                    continue
                result["series"].append({"function": name, "n": n, "period": period, "seconds": t})
                print(f"talib {name:<14} n={n:>9,} p={period}  {t*1e3:9.2f} ms", flush=True)

        close, high, low = data["close"], data["high"], data["low"]
        # rolling_max at p=200, the long-window case
        t = bench(lambda: talib.MAX(close, 200), runs_for(n))
        result["cases"].append({"case": "rolling_max_200", "n": n, "seconds": t})

        # the shared-node formula, written as a TA-Lib user would: every term recomputes SMA and EMA
        def shared():
            acc = 0.0
            for k in range(1, SHARED_TERMS + 1):
                s, e = talib.SMA(close, 20), talib.EMA(close, 20)
                acc = acc + (s - e * k) / s
            return acc

        result["cases"].append({"case": "shared_nodes", "n": n, "seconds": bench(shared, runs_for(n))})

        per = n // INSTRUMENTS
        cols = [np.ascontiguousarray(close[i * per:(i + 1) * per]) for i in range(INSTRUMENTS)]
        result["cases"].append({"case": "instruments_500", "mode": "single", "n": n,
                                "seconds": bench(lambda: [talib.SMA(c, 20) for c in cols], runs_for(n))})
        workers = os.cpu_count() or 1
        with ThreadPoolExecutor(workers) as pool:  # TA-Lib releases the GIL
            result["cases"].append({"case": "instruments_500", "mode": "parallel", "n": n,
                                    "workers": workers,
                                    "seconds": bench(lambda: list(pool.map(lambda c: talib.SMA(c, 20), cols)),
                                                     runs_for(n))})
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


# ------------------------------------------------------------------ raptorbt side


def registry(rb):
    import raptorbt._raptorbt as ext  # noqa: PLC0415

    rows = json.loads(ext.expr_registry_json())["functions"]
    by_talib = {r["talib_name"]: r for r in rows if r.get("talib_name")}
    return rows, by_talib


def expression_for(row: dict, name: str, spec: dict, period, kw: dict):
    """The expression text for a TA-Lib call, or (None, reason)."""
    t_inputs = spec["inputs"]
    args = []
    if len(row["inputs"]) == len(t_inputs):
        args += [FIELD_OF.get(i, "close") for i in t_inputs]
    elif not all(i["default_field"] for i in row["inputs"]):
        return None, "input count differs"
    t_params = list(spec["parameters"].items())
    for i, p in enumerate(row["params"]):
        if i < len(t_params):
            key, default = t_params[i]
            val = kw.get(key, default)
            if "matype" in key:
                if val != 0:
                    return None, "matype"
                val = None
        else:
            val = None
        if val is None:
            if p["default"] is None:
                return None, f"no value for {p['name']}"
            break  # the rest take their defaults
        args.append(str(int(val)) if float(val).is_integer() and p["kind"] == "int" else repr(float(val)))
    return f"{row['name']}({', '.join(args)})", None


def run_raptorbt(args) -> None:
    if args.threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    else:
        os.environ.pop("RAYON_NUM_THREADS", None)
    import numpy as np
    import raptorbt as rb
    from concurrent.futures import ThreadPoolExecutor

    threads = args.threads or os.cpu_count()
    result = {"library": "raptorbt", "machine": machine(threads), "series": [], "cases": [],
              "skipped": []}
    funcs = oracle()
    _, by_talib = registry(rb)
    for n in SIZES:
        data = bars(n)
        for name, spec in funcs.items():
            row = by_talib.get(name)
            if row is None:
                continue
            for period, kw in cases_for(name, spec):
                src, why = expression_for(row, name, spec, period, kw)
                if src is None:
                    result["skipped"].append({"function": name, "reason": why})
                    continue
                try:
                    e = rb.expr(src)
                    t = bench(lambda: e.eval(data), runs_for(n))
                except Exception as ex:  # noqa: BLE001
                    result["skipped"].append({"function": name, "reason": f"{type(ex).__name__}: {ex}"})
                    continue
                result["series"].append({"function": name, "n": n, "period": period, "seconds": t,
                                         "expression": src})
                print(f"rb    {name:<14} n={n:>9,} p={period}  {t*1e3:9.2f} ms  {src}", flush=True)

        close = data["close"]
        result["cases"].append({"case": "rolling_max_200", "n": n,
                                "seconds": bench(lambda: rb.rolling_max(close, 200), runs_for(n))})
        terms = " + ".join(f"(sma(close, 20) - ema(close, 20) * {k}) / sma(close, 20)"
                           for k in range(1, SHARED_TERMS + 1))
        try:
            e = rb.expr(terms)
            result["cases"].append({"case": "shared_nodes", "n": n,
                                    "seconds": bench(lambda: e.eval(data), runs_for(n))})
        except Exception as ex:  # noqa: BLE001
            print(f"note: shared_nodes skipped: {ex}")

        per = n // INSTRUMENTS
        cols = [{"close": np.ascontiguousarray(close[i * per:(i + 1) * per])} for i in range(INSTRUMENTS)]
        e = rb.expr("sma(close, 20)")
        result["cases"].append({"case": "instruments_500", "mode": "single", "n": n,
                                "seconds": bench(lambda: [e.eval(c) for c in cols], runs_for(n))})
        workers = os.cpu_count() or 1
        with ThreadPoolExecutor(workers) as pool:  # Expr.eval releases the GIL
            result["cases"].append({"case": "instruments_500", "mode": "parallel", "n": n,
                                    "workers": workers,
                                    "seconds": bench(lambda: list(pool.map(e.eval, cols)), runs_for(n))})

    if hasattr(rb.expr("close"), "sweep"):
        print("note: sweep exists but this harness has no sweep case wired yet")
        result["skipped"].append({"function": "sweep", "reason": "not wired"})
    else:
        print("note: sweep case skipped: Expr.sweep is not available in this build")
        result["skipped"].append({"function": "sweep", "reason": "Expr.sweep not available"})
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


# ------------------------------------------------------------------ report


def load(paths):
    talib, rbs = None, []
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        if d["library"] == "talib":
            talib = d
        else:
            rbs.append(d)
    if talib is None or not rbs:
        sys.exit("report needs one talib JSON and at least one raptorbt JSON")
    return talib, rbs


def verdicts(talib: dict, rb: dict):
    base = {(s["function"], s["n"], s["period"]): s["seconds"] for s in talib["series"]}
    worst: dict[str, tuple[float, str]] = {}
    for s in rb["series"]:
        k = (s["function"], s["n"], s["period"])
        if k not in base:
            continue
        ratio = s["seconds"] / base[k]
        label = f"n={s['n']:,} p={s['period']}"
        if s["function"] not in worst or ratio > worst[s["function"]][0]:
            worst[s["function"]] = (ratio, label)
    return worst


def verdict_word(r: float) -> str:
    return "PASS 1.1x" if r <= BAR_ONE else "PASS 1.5x" if r <= BAR_ALL else "FAIL"


def run_report(args) -> None:
    talib, rbs = load(args.files)
    single = next((r for r in rbs if r["machine"]["threads"] == 1), rbs[0])
    out = {"talib": talib["machine"] | {"version": talib.get("version")},
           "raptorbt": [r["machine"] for r in rbs], "verdicts": {}, "cases": []}
    worst = verdicts(talib, single)
    lines = ["# Speed against TA-Lib", "",
             f"TA-Lib {talib.get('version')} on {talib['machine']['platform']}, "
             f"{talib['machine']['cpu_count']} cores. Median of several runs after a warm-up. "
             "Ratio is raptorbt / TA-Lib; below 1.0 is faster. "
             f"Single-threaded raptorbt run: {single['machine']['threads']} thread(s).", "",
             "## Single series, single thread", "",
             "| Function | Worst ratio | Where | Verdict |", "|---|---|---|---|"]
    for f, (r, where) in sorted(worst.items()):
        out["verdicts"][f] = {"ratio": r, "where": where, "verdict": verdict_word(r)}
        lines.append(f"| {f} | {r:.2f} | {where} | {verdict_word(r)} |")
    total = len(worst)
    within_one = sum(1 for r, _ in worst.values() if r <= BAR_ONE)
    all_ok = all(r <= BAR_ALL for r, _ in worst.values())
    share = within_one / total if total else 0.0
    bar = total > 0 and share >= BAR_SHARE and all_ok
    summary = (f"{within_one}/{total} functions within {BAR_ONE}x ({share:.0%}, need {BAR_SHARE:.0%}); "
               f"all within {BAR_ALL}x: {all_ok}. Speed bar: {'PASS' if bar else 'FAIL'}")
    out["summary"] = {"functions": total, "within_1_1": within_one, "all_within_1_5": all_ok, "pass": bar}
    skipped = sorted({s["function"] for s in single.get("skipped", [])})
    lines += ["", summary, "", f"Not registered or not comparable here: {len(skipped)} (see JSON)."]

    lines += ["", "## Cases", "", "| Case | Bars | Mode | TA-Lib s | raptorbt s | Ratio |", "|---|---|---|---|---|---|"]
    for rb in rbs:
        for c in rb["cases"]:
            ref = next((t for t in talib["cases"]
                        if t["case"] == c["case"] and t["n"] == c["n"] and t.get("mode") == c.get("mode")), None)
            if ref is None:
                continue
            ratio = c["seconds"] / ref["seconds"]
            mode = c.get("mode") or ("single" if rb["machine"]["threads"] == 1 else "parallel")
            mode = f"{mode} ({rb['machine']['threads']} rayon threads)"
            lines.append(f"| {c['case']} | {c['n']:,} | {mode} | {ref['seconds']:.4f} | "
                         f"{c['seconds']:.4f} | {ratio:.2f} |")
            out["cases"].append(c | {"talib_seconds": ref["seconds"], "ratio": ratio, "mode": mode})
    lines += ["", "The shared-nodes case is one raptorbt program summing 20 terms that share `sma` and `ema`, "
              "against TA-Lib recomputing both per term. The instruments case is `sma(close, 20)` over 500 "
              "instruments; the parallel rows use a thread pool on both sides.", ""]
    print(summary)
    for f, v in sorted(out["verdicts"].items()):
        print(f"{f:<16} {v['ratio']:6.2f}  {v['verdict']:<10} {v['where']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1))
    if args.md:
        pathlib.Path(args.md).write_text("\n".join(lines))
        print(f"wrote {args.md}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ("talib", "raptorbt"):
        p = sub.add_parser(m)
        p.add_argument("--out", required=True)
        if m == "raptorbt":
            p.add_argument("--threads", type=int, default=1, help="rayon threads; 0 = all cores")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--json")
    p.add_argument("--md")
    args = ap.parse_args()
    {"talib": run_talib, "raptorbt": run_raptorbt, "report": run_report}[args.mode](args)


if __name__ == "__main__":
    main()
