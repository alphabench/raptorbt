"""``run_strategies`` with a real ``Strategy`` subclass.

``run_all.py`` times the signal-array runners, where the whole strategy is two boolean
arrays. This times the event-driven path a strategy written against the Strategy API
takes: one ``run_strategies`` call, from the call to its return, including building the
runtime, every hook (Python), order matching, stops, and the results.

* bars: ``SmaCrossBars`` (long above a 20-bar SMA, fixed stop, trailing stop) over
  1,000 / 100,000 / 1,000,000 close-stamped 1-minute bars (390 per session, consecutive
  weekdays);
* ticks: the momentum strategy of ``live_latency.py`` over 100,000 / 1,000,000 ticks in
  one session, aggregated into 1-minute bars (``primary_bars=(1, "m")``).

Each size is run several times (fewer for the larger ones); best and median are reported,
as in ``run_all.py``. The run must reach the end of the input (the strategy counts every
bar or print it is handed) and trade.

    python strategy_backtest.py
    python strategy_backtest.py --quick
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _live  # noqa: E402
from raptorbt import run_strategies  # noqa: E402


def timed(make_strategy, run, reps: int):
    times, last = [], None
    for _ in range(reps):
        strategy = make_strategy()
        gc.collect()
        t0 = time.perf_counter_ns()
        result = run(strategy)
        times.append(time.perf_counter_ns() - t0)
        last = (strategy, result)
    times.sort()
    return times[0], times[len(times) // 2], last


def bench_bars(n: int, reps: int) -> dict:
    data = {"S0": _live.bars(n, seed=7)}
    best, med, (strategy, results) = timed(
        _live.SmaCrossBars,
        lambda s: run_strategies([s], data, _live.config(), bar_stamp="close"),
        reps,
    )
    summary = _live.trade_summary(results[0])
    row = {"input": "bars", "n": n, "reps": reps, "best_ms": best / 1e6, "median_ms": med / 1e6,
           "ns_per_event": best / n, "m_per_s": n / (best / 1e9) / 1e6, "seen": strategy.bars, **summary}
    checks = []
    if strategy.bars != n:
        checks.append(f"the strategy saw {strategy.bars} bars of {n}")
    if summary["trades"] == 0:
        checks.append("no trade")
    if checks:
        row["error"] = "check failed: " + "; ".join(checks)
    return row


def bench_ticks(n: int, reps: int) -> dict:
    t = _live.ticks(n, seed=11)
    ticks = {"S0": {"timestamps": t["timestamps"], "ltp": t["ltp"], "bid": t["bid"], "ask": t["ask"]}}
    best, med, (strategy, results) = timed(
        _live.MomentumTick,
        lambda s: run_strategies([s], None, _live.config(), ticks=ticks, primary_bars=(1, "m")),
        reps,
    )
    summary = _live.trade_summary(results[0])
    row = {"input": "ticks", "n": n, "reps": reps, "best_ms": best / 1e6, "median_ms": med / 1e6,
           "ns_per_event": best / n, "m_per_s": n / (best / 1e9) / 1e6, "seen": strategy.prints, **summary}
    checks = []
    if strategy.prints != n:
        checks.append(f"the strategy saw {strategy.prints} prints of {n}")
    if summary["trades"] == 0:
        checks.append("no trade")
    if checks:
        row["error"] = "check failed: " + "; ".join(checks)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="a tiny run to check the script works")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if args.quick:
        bar_sizes, tick_sizes = [(1_000, 3), (5_000, 1)], [(5_000, 1)]
    else:
        bar_sizes = [(1_000, 20), (100_000, 3), (1_000_000, 1)]
        tick_sizes = [(100_000, 3), (1_000_000, 1)]

    environment = _live.print_env("run_strategies with a Strategy subclass")
    print(f"capital {_live.CAPITAL:,.0f}, orders of {_live.QUANTITY} units, no fees\n")
    results: list[dict] = []
    for n, reps in bar_sizes:
        _live.run_case(f"bars/{n}", lambda: bench_bars(n, reps), results)
    for n, reps in tick_sizes:
        _live.run_case(f"ticks/{n}", lambda: bench_ticks(n, reps), results)

    print()
    _live.table(results, [
        ("case", "case"), ("best_ms", "best ms"), ("median_ms", "median ms"), ("ns_per_event", "ns/event"),
        ("m_per_s", "M/s"), ("seen", "seen"), ("orders", "orders"), ("trades", "trades"),
    ])
    _live.write_json(args.json or _live.default_json(__file__), {
        "script": "strategy_backtest",
        "env": environment,
        "params": {"capital": _live.CAPITAL, "quantity": _live.QUANTITY, "quick": args.quick},
        "cases": results,
    })


if __name__ == "__main__":
    main()
