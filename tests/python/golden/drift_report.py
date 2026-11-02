"""Measure how far the current engine's results sit from the committed golden baselines.

    .venv/bin/python tests/python/golden/drift_report.py [--fixtures PATH]

Replays the golden corpus (as ``test_golden.py`` does) and, for each metric,
prints the maximum absolute and relative difference from the baseline in
``fixtures.json`` across the whole set. Floats are stored as hex, so this
compares exact doubles. Used to state the size of a results change when the
accounting arithmetic moves (see CHANGELOG, "Results change").
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from generate import (  # noqa: E402
    MULTILEG_KINDS,
    MULTILEG_TIMINGS,
    GoldenSma,
    config_variants,
    result_digest,
    run_multileg,
    thaw_inputs,
)

import raptorbt  # noqa: E402
from raptorbt import BacktestConfig  # noqa: E402


def unhex(x):
    return float.fromhex(x)


class Drift:
    def __init__(self):
        self.abs = defaultdict(float)
        self.rel = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, metric, new, old):
        d = abs(new - old)
        self.abs[metric] = max(self.abs[metric], d)
        denom = max(abs(old), 1e-300)
        self.rel[metric] = max(self.rel[metric], d / denom if old != 0 else (0.0 if d == 0 else float("inf")))
        self.n[metric] += 1

    def report(self):
        print(f"{'metric':<26}{'n':>8}{'max abs':>16}{'max rel':>16}")
        for m in sorted(self.abs):
            print(f"{m:<26}{self.n[m]:>8}{self.abs[m]:>16.6g}{self.rel[m]:>16.3g}")


def compare_digest(drift, new, old, notes, name):
    for i, (a, b) in enumerate(zip(new["equity_curve"], old["equity_curve"])):
        drift.add("equity_curve", unhex(a), unhex(b))
    if len(new["trades"]) != len(old["trades"]):
        notes.append(f"{name}: trade count {len(old['trades'])} -> {len(new['trades'])}")
    for tn, to in zip(new["trades"], old["trades"]):
        for k in ("entry_price", "exit_price", "size", "pnl", "fees"):
            drift.add(f"trade.{k}", unhex(tn[k]), unhex(to[k]))
        if tn["exit_idx"] != to["exit_idx"] or tn["exit_reason"] != to["exit_reason"]:
            notes.append(f"{name}: a trade exits differently")
    for k in ("sharpe", "sortino", "omega", "total_return_pct", "max_drawdown_pct"):
        drift.add(k, unhex(new[k]), unhex(old[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=str(HERE / "fixtures.json"))
    args = ap.parse_args()
    fixtures = json.loads(Path(args.fixtures).read_text())
    inputs = fixtures["inputs"]
    drift, notes = Drift(), []

    for name, config, ic, direction in config_variants():
        ts, o, h, l, c, v, entries, exits = thaw_inputs(inputs["shared"])
        result = raptorbt.run_single_backtest(
            ts, o, h, l, c, v, entries, exits,
            direction=direction, config=config, instrument_config=ic,
        )
        compare_digest(drift, result_digest(result), fixtures[f"single/{name}"], notes, name)

    ts, o, h, l, c, v, _, _ = thaw_inputs(inputs["shared"])
    result = raptorbt.run_strategy_backtest(GoldenSma, ts, o, h, l, c, v)
    compare_digest(drift, result_digest(result), fixtures["class/sma_cross"], notes, "class")

    instruments = []
    for seed in (11, 12, 13):
        symbol = f"SYM{seed}"
        pts, po, ph, pl, pc, pv, pe, px = thaw_inputs(inputs[symbol])
        instruments.append((pts, po, ph, pl, pc, pv, pe, px, 1, 1.0, symbol))
    portfolio = raptorbt.run_portfolio_backtest(
        instruments, config=BacktestConfig(), allocation="equal_weight"
    )
    old = fixtures["portfolio/shared_pool"]
    for a, b in zip(portfolio.result.equity_curve(), old["equity_curve"]):
        drift.add("portfolio.equity_curve", float(a), unhex(b))
    drift.add("portfolio.total_return_pct", portfolio.metrics.total_return_pct, unhex(old["total_return_pct"]))
    for s in portfolio.per_instrument:
        o_ = old["per_instrument"][s.symbol]
        drift.add("portfolio.pnl", s.pnl, unhex(o_["pnl"]))
        if s.trades != o_["trades"]:
            notes.append(f"portfolio {s.symbol}: trades {o_['trades']} -> {s.trades}")

    for kind in MULTILEG_KINDS:
        for timing in MULTILEG_TIMINGS:
            result = run_multileg(inputs, kind, timing)
            compare_digest(drift, result_digest(result), fixtures[f"{kind}/{timing}"], notes, f"{kind}/{timing}")

    drift.report()
    for n in sorted(set(notes)):
        print("NOTE:", n)


if __name__ == "__main__":
    main()
