"""Research tooling: deterministic parameter sweeps and walk-forward windows.

The numerics (SplitMix64 designs, walk-forward windows, the deflated Sharpe ratio, the
stationary bootstrap, ``without_best``) are the Rust ones in ``raptorbt._raptorbt``; this
module is the orchestration around them.

A ``Design`` is the set of parameter points: a full ``grid``, or a ``random`` design of
``n`` points drawn by SplitMix64 from a declared seed. A ``WalkForward`` cuts a list of
trading days into in-sample and out-of-sample windows (rolling, by counts of days).
``sweep`` runs every (point, window) through a user ``run`` function and returns a
``SweepResult``: the per-run rows, a streaming summary (Welford per metric, constant
memory), and a deflated Sharpe ratio that discounts the best Sharpe for the number of
trials.

Determinism: the run list is sorted by (point index, window index); each run is
independent; with ``workers > 1`` the runs go through a ``multiprocessing`` pool (spawn),
or with ``executor="thread"`` through threads of this process, and the rows are put back in
run order, so the result is the same whatever the worker count.

Threads suit runs of ``raptorbt.Engine.run_columns``: a ruleset whose conditions and series
are not Python callables runs its batch without the interpreter lock (Python actions take it
only when their rule fires), so the engines of different points run at once on one copy of
the data, with no process start and nothing pickled. ``run`` may then be any callable; it
must not share mutable state between points. The manifest (a JSONL file, one row per finished run, keyed ``"point/window"``)
records each finished run; a sweep resumed with the same manifest skips them.

The ``run`` function is the user's: it takes ``(params, window)`` and returns a JSON-able
row with at least ``sharpe`` and ``samples`` (and ``skew`` / ``kurtosis`` when known). It
must be importable by name (a module-level function) for process workers.
"""

from __future__ import annotations

import json
import math
import multiprocessing
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raptorbt._raptorbt import (
    deflated_sharpe,
    design_grid,
    design_random,
    stationary_bootstrap,
    walk_forward_windows,
    without_best,
)

__all__ = [
    "OVERFITTING_NOTE",
    "Design",
    "SweepResult",
    "WalkForward",
    "Welford",
    "Window",
    "deflated_sharpe",
    "row_from",
    "stationary_bootstrap",
    "sweep",
    "without_best",
]

Params = dict[str, Any]


# --- designs --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Design:
    """Points in parameter space. ``grid`` is the full product in key order; ``random``
    draws ``n`` points, each parameter from its declared list or ``(lo, hi)`` range, by
    SplitMix64 (the same draws for the same seed)."""

    points: tuple[Params, ...]
    name: str

    @classmethod
    def grid(cls, space: Mapping[str, Sequence[Any]]) -> Design:
        out = design_grid(dict(space))
        return cls(tuple(out), f"grid({len(out)})")

    @classmethod
    def random(
        cls, space: Mapping[str, Sequence[Any] | tuple[float, float]], n: int, *, seed: int
    ) -> Design:
        out = design_random(dict(space), n, seed=seed)
        return cls(tuple(out), f"random({n}, seed={seed})")

    def __len__(self) -> int:
        return len(self.points)


# --- windows --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    train: tuple[str, ...]  # trading days, in order
    test: tuple[str, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train": [self.train[0], self.train[-1]] if self.train else [],
            "test": [self.test[0], self.test[-1]] if self.test else [],
        }


class WalkForward:
    """Rolling in-sample / out-of-sample windows over trading days: ``train`` days then
    ``test`` days, stepped by ``step`` (default the test length, so the test windows
    tile)."""

    def __init__(self, days: Sequence[str], *, train: int, test: int, step: int | None = None):
        self.days = tuple(days)
        self.train, self.test = train, test
        self.step = test if step is None else step
        self._single = False
        walk_forward_windows([], train=train, test=test, step=step)  # validates the counts

    def windows(self) -> list[Window]:
        if self._single:
            return [Window(0, (), self.days)]
        rows = walk_forward_windows(
            list(self.days), train=self.train, test=self.test, step=self.step
        )
        return [Window(r["index"], tuple(r["train"]), tuple(r["test"])) for r in rows]

    @classmethod
    def single(cls, days: Sequence[str]) -> WalkForward:
        """One window: no training days, every day out of sample (a plain sweep)."""
        wf = cls.__new__(cls)
        wf.days = tuple(days)
        wf.train, wf.test, wf.step = 0, len(wf.days), len(wf.days)
        wf._single = True
        return wf


# --- results --------------------------------------------------------------------------------------


class Welford:
    __slots__ = ("m2", "max", "mean", "min", "n")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"n": self.n, "mean": self.mean, "std": self.std(), "min": self.min, "max": self.max}


@dataclass
class SweepResult:
    rows: list[dict[str, Any]]  # one per (point, window), in run order
    summary: dict[str, dict[str, float]]  # per numeric metric over the rows
    trials: int
    deflated: dict[str, Any]
    design: str
    windows: int
    workers: int

    def best(self, metric: str = "sharpe") -> dict[str, Any] | None:
        rows = [r for r in self.rows if _is_number(r.get(metric))]
        return max(rows, key=lambda r: (r[metric], -r["run"])) if rows else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "windows": self.windows,
            "workers": self.workers,
            "trials": self.trials,
            "summary": self.summary,
            "deflated": self.deflated,
            "rows": self.rows,
        }


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


# --- the sweep ------------------------------------------------------------------------------------

RunFn = Callable[[Params, Window], dict[str, Any]]


def _key(point: int, window: int) -> str:
    return f"{point}/{window}"


def _load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["key"]] = row
    return out


def _run_one(args: tuple[RunFn, int, Params, Window]) -> dict[str, Any]:
    fn, i, params, window = args
    row = dict(fn(params, window))
    row.update({"point": i, "params": params, "window": window.index, "key": _key(i, window.index)})
    return row


def sweep(
    run: RunFn,
    design: Design,
    windows: Sequence[Window] | WalkForward,
    *,
    workers: int = 1,
    executor: str = "process",
    manifest: str | Path | None = None,
    metric: str = "sharpe",
    periods_per_year: int = 252,
) -> SweepResult:
    """Every (point, window) through ``run``; rows in run order whatever ``workers`` is.
    ``executor``: with ``workers > 1``, ``"process"`` (a spawned pool) or ``"thread"``
    (threads of this process; see the module notes)."""
    wins = windows.windows() if isinstance(windows, WalkForward) else list(windows)
    if not wins:
        raise ValueError("no windows")
    if workers < 1:
        raise ValueError("workers is at least 1")
    if executor not in ("process", "thread"):
        raise ValueError("executor is 'process' or 'thread'")
    path = None if manifest is None else Path(manifest)
    done = _load_manifest(path)
    jobs: list[tuple[RunFn, int, Params, Window]] = []
    rows: dict[str, dict[str, Any]] = dict(done)
    for i, p in enumerate(design.points):
        for w in wins:
            if _key(i, w.index) not in done:
                jobs.append((run, i, p, w))
    if jobs:
        if workers == 1:
            fresh: Iterator[dict[str, Any]] = (_run_one(j) for j in jobs)
        elif executor == "thread":
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(workers) as threads:
                fresh = iter(list(threads.map(_run_one, jobs)))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(workers) as pool:
                fresh = iter(pool.map(_run_one, jobs))
        with path.open("a", encoding="utf-8") if path is not None else _Null() as out:
            for row in fresh:
                rows[row["key"]] = row
                out.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    ordered = [rows[_key(i, w.index)] for i in range(len(design.points)) for w in wins]
    for n, r in enumerate(ordered):
        r["run"] = n
    summary: dict[str, Welford] = {}
    for r in ordered:
        for k, v in r.items():
            if _is_number(v) and k not in ("point", "window", "run"):
                summary.setdefault(k, Welford()).add(float(v))
    scored = [r for r in ordered if isinstance(r.get(metric), int | float)]
    sharpes = [float(r[metric]) for r in scored]
    best_row = max(scored, key=lambda r: (r[metric], -r["run"]), default=None)
    deflated: dict[str, Any] = {"trials": len(sharpes), "dsr": None}
    if best_row is not None:
        deflated = deflated_sharpe(
            sharpes,
            float(best_row[metric]),
            samples=int(best_row.get("samples", 0)),
            skew=float(best_row.get("skew", 0.0)),
            kurtosis=float(best_row.get("kurtosis", 3.0)),
            periods_per_year=periods_per_year,
        )
    return SweepResult(
        rows=ordered,
        summary={k: v.as_dict() for k, v in sorted(summary.items())},
        trials=len(ordered),
        deflated=deflated,
        design=design.name,
        windows=len(wins),
        workers=workers,
    )


class _Null:
    def __enter__(self) -> _Null:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def write(self, _s: str) -> None:
        return None


def row_from(result: Any) -> dict[str, Any]:
    """A sweep row from a finished run (a ``BacktestResult``, or a ``PortfolioResult``, whose
    combined result is read): the run's own figures, the return moments a deflated Sharpe
    needs, and its sample count. Ratios are fractions, as the engine reports them.

    A run that never traded has no Sharpe (``metrics.sharpe_ratio`` is ``None``). The row
    holds ``0.0`` for it: a sweep ranks and deflates numbers, and a trial that earned
    nothing ranks as zero."""
    result = getattr(result, "result", result)
    m = result.metrics
    orders = result.orders()
    return {
        "sharpe": m.sharpe_ratio if m.sharpe_ratio is not None else 0.0,
        "sortino": m.sortino_ratio,
        "total_return": m.total_return_pct / 100.0,
        "max_drawdown": m.max_drawdown_pct / 100.0,
        "fills": sum(o.fill_slices for o in orders),
        "fees": m.total_fees_paid,
        "samples": len(result.returns()),
        "skew": m.return_skew if m.return_skew is not None else 0.0,
        "kurtosis": m.return_kurtosis if m.return_kurtosis is not None else 3.0,
    }


OVERFITTING_NOTE = """\
A sweep is a multiple-testing search: the best of N trials has a Sharpe that grows with N even
when every trial is noise. The summary reports the deflated Sharpe ratio (Bailey & Lopez de
Prado): sr0 is the Sharpe luck alone hands the best of N trials given their spread, and dsr is
the probability the best trial's Sharpe exceeds it, from its sample count, skew and kurtosis.
Read a dsr under 0.95 as "not distinguishable from the search"; read an out-of-sample Sharpe
only, never the in-sample one; and report the number of trials with every headline figure.
"""
