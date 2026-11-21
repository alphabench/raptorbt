"""Shared helpers for the live-path benchmarks (live_*.py, strategy_backtest.py).

Synthetic data only, seeded, no network and no clock. Timestamps are nanoseconds since
the epoch inside a 09:00-15:30 UTC session on consecutive weekdays from Monday
2026-01-05; no session schedule or square-off is configured, so nothing in the engine
reacts to the time of day.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

import raptorbt as rb
from raptorbt import BacktestConfig, Strategy
from raptorbt.strategy import orders

HERE = pathlib.Path(__file__).resolve().parent

#: Enough that no strategy here can run out of cash: orders are a fixed 10 units of
#: an instrument priced around 100, no fees.
CAPITAL = 1_000_000_000.0
QUANTITY = 10
TICK = 0.05
MIN_NS = 60 * 10**9
SESSION_OPEN_NS = 9 * 3600 * 10**9  # 09:00 UTC
SESSION_NS = 390 * MIN_NS  # 09:00-15:30
FIRST_DAY = dt.date(2026, 1, 5)  # a Monday


# -- environment and output ---------------------------------------------------


def env() -> dict:
    return {
        "raptorbt": rb.__version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def print_env(title: str) -> dict:
    e = env()
    print(title)
    print(
        f"raptorbt {e['raptorbt']}  Python {e['python']} ({e['implementation']})  numpy {e['numpy']}\n"
        f"{e['platform']}  machine {e['machine']}  cpu {e['processor']}  cpus {e['cpu_count']}"
    )
    print()
    return e


def default_json(script: str) -> pathlib.Path:
    return HERE / f"{pathlib.Path(script).stem}_results.json"


def write_json(path, payload) -> None:
    path = pathlib.Path(path)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {path}")


def run_case(name: str, fn, results: list) -> dict | None:
    """Run one case; an exception becomes an error entry and the next case still runs."""
    print(f"-- {name}", flush=True)
    try:
        row = fn()
    except Exception as e:  # one failing case must not hide the others
        row = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        print(f"   ERROR {row['error']}", flush=True)
        results.append({"case": name, **row})
        return None
    results.append({"case": name, **row})
    return row


def table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    """Print ``rows`` as a plain table; ``columns`` is ``[(key, header)]``."""

    def cell(row, key):
        if "error" in row and key not in row:
            return "ERROR" if key == columns[1][0] else ""
        v = row.get(key, "")
        if isinstance(v, float):
            return f"{v:,.1f}" if abs(v) >= 100 else f"{v:,.3f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)

    cells = [[cell(r, k) for k, _ in columns] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, (_, h) in enumerate(columns)]
    print("  ".join(h.rjust(w) for (_, h), w in zip(columns, widths)))
    for c in cells:
        print("  ".join(v.rjust(w) for v, w in zip(c, widths)))
    for r in rows:
        if "error" in r:
            print(f"   {r['case']}: {r['error']}")


# -- timing --------------------------------------------------------------------


def percentiles(samples_ns: np.ndarray) -> dict:
    """p50 / p90 / p99 / p99.9 / max / mean of per-call samples, in ns."""
    if len(samples_ns) == 0:
        return {"n": 0}
    q = np.percentile(samples_ns, [50, 90, 99, 99.9])
    return {
        "n": int(len(samples_ns)),
        "p50_ns": float(q[0]),
        "p90_ns": float(q[1]),
        "p99_ns": float(q[2]),
        "p999_ns": float(q[3]),
        "max_ns": float(samples_ns.max()),
        "mean_ns": float(samples_ns.mean()),
    }


def timer_overhead(n: int = 1_000_000) -> dict:
    """The cost of the measurement itself: two perf_counter_ns calls and a store per
    iteration, with nothing between them. Subtract its p50 from a per-call p50."""
    clock = time.perf_counter_ns
    samples = np.empty(n, dtype=np.int64)
    for i in range(n):
        t0 = clock()
        samples[i] = clock() - t0
    return percentiles(samples)


# -- memory ----------------------------------------------------------------------

try:
    import psutil  # optional; never installed by these scripts
except ImportError:
    psutil = None


def rss_bytes(pid: int | None = None) -> int:
    """Current resident set size of ``pid`` (default: this process)."""
    pid = os.getpid() if pid is None else pid
    if psutil is not None:
        return psutil.Process(pid).memory_info().rss
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip()) * 1024


def peak_rss_bytes() -> int:
    """This process's peak resident set size (ru_maxrss: bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def run_child(script: str, args: list[str], timeout: float | None = None) -> dict:
    """Run ``script --child ...`` in a fresh interpreter and return the JSON it prints last."""
    proc = subprocess.run(
        [sys.executable, script, "--child", *args], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"child exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# -- data ------------------------------------------------------------------------


def day_ns(k: int) -> int:
    """Midnight UTC of the ``k``-th weekday from FIRST_DAY, in ns."""
    d = FIRST_DAY
    seen = -1
    while True:
        if d.weekday() < 5:
            seen += 1
            if seen == k:
                return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()) * 10**9
        d += dt.timedelta(days=1)


def session_ticks_ts(n: int, days: int = 1, offset_ns: int = 0) -> np.ndarray:
    """``n`` strictly increasing tick stamps spread evenly over ``days`` sessions."""
    per_day = -(-n // days)
    step = SESSION_NS // max(per_day, 1)
    if step < 1:
        raise ValueError(f"{per_day} ticks do not fit in one session at 1 ns apart")
    out = []
    for k in range(days):
        m = min(per_day, n - k * per_day)
        if m <= 0:
            break
        out.append(day_ns(k) + SESSION_OPEN_NS + offset_ns + np.arange(m, dtype=np.int64) * step)
    return np.concatenate(out)


def bar_closes_ts(n: int, minutes: int = 1) -> np.ndarray:
    """Close stamps of ``n`` bars of ``minutes`` width, 390 // minutes per session, on
    consecutive weekdays (as many sessions as ``n`` needs)."""
    per_day = 390 // minutes
    days = -(-n // per_day)
    width = minutes * MIN_NS
    one = np.arange(1, per_day + 1, dtype=np.int64) * width
    ts = np.concatenate([day_ns(k) + SESSION_OPEN_NS + one for k in range(days)])
    return ts[:n]


def prices(n: int, seed: int, start: float = 100.0, steps_per_session: int | None = None,
           session_vol: float = 0.02, level_sd: float = 0.01) -> np.ndarray:
    """A mean-reverting log walk on a 0.05 grid.

    Each step moves ``session_vol / sqrt(steps_per_session)`` (so a session moves about as
    much whatever the step count), and an AR(1) pull holds the level within about
    ``level_sd`` of ``start``: the price oscillates instead of drifting away, so a momentum
    rule keeps finding entries for millions of steps, at a rate set by the session rather
    than by how many ticks it is cut into.
    """
    steps_per_session = n if steps_per_session is None else steps_per_session
    vol = session_vol / np.sqrt(max(steps_per_session, 1))
    keep = 1.0 - min(0.5, (vol / level_sd) ** 2 / 2)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, vol, n)
    x = np.empty(n)
    acc = 0.0
    for i in range(n):  # an AR(1); a loop is clearer than a filter and runs once per case
        acc = acc * keep + shocks[i]
        x[i] = acc
    return np.round(start * np.exp(x) / TICK) * TICK


def ticks(n: int, seed: int = 11, days: int = 1, offset_ns: int = 0, start: float = 100.0) -> dict:
    """A tick tape: ``timestamps``, ``ltp``, ``bid``, ``ask`` (one tick either side)."""
    ltp = prices(n, seed, start, steps_per_session=-(-n // days))
    return {
        "timestamps": session_ticks_ts(n, days, offset_ns),
        "ltp": ltp,
        "bid": ltp - TICK,
        "ask": ltp + TICK,
    }


def bars(n: int, seed: int = 7, start: float = 100.0) -> dict:
    """``n`` close-stamped 1-minute bars around a mean-reverting walk."""
    close = prices(n, seed, start, steps_per_session=390)
    rng = np.random.default_rng(seed + 1)
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0.0, 0.0006, n)) * close
    return {
        "timestamps": bar_closes_ts(n),
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": np.full(n, 1_000.0),
    }


def config(trailing: float = 0.002) -> BacktestConfig:
    """Ample capital, no fees, a trailing stop on every position (``trailing`` is a fraction)."""
    cfg = BacktestConfig(initial_capital=CAPITAL, fees=0.0)
    cfg.set_trailing_stop(trailing)
    return cfg


# -- strategies ------------------------------------------------------------------


class BarsOnly(Strategy):
    """No tick hook: only the 1-minute bars the node aggregates reach it."""

    def on_start(self, ctx):
        self.bars = 0

    def on_bar(self, ctx):
        self.bars += 1


class NoopTick(Strategy):
    """A tick hook that does nothing but count."""

    def on_start(self, ctx):
        self.prints = 0

    def on_trade(self, ctx, tick):
        self.prints += 1


class MomentumTick(Strategy):
    """Reads a 1-minute SMA on every print and buys a fixed quantity when the print is
    0.05 percent above it, with a fixed stop 0.4 percent under the entry. The trailing
    stop comes from the config (``config()``), so a position is open much of the time."""

    def on_start(self, ctx):
        self.sma = {s: ctx.indicators.sma(instrument=s, period=5) for s in ctx.symbols}
        self.prints = 0
        self.open_prints = 0
        self.fills = 0

    def on_fill(self, ctx, event):
        self.fills += 1

    def on_trade(self, ctx, tick):
        self.prints += 1
        if not ctx.is_flat:
            self.open_prints += 1
            return
        sma = self.sma[ctx.symbol]
        if sma.ready and tick.price > sma.value * 1.0005:
            self.submit_order(
                orders.Market(side="buy", quantity=QUANTITY, stop_price=round(tick.price * 0.996, 2))
            )


class SmaCrossBars(Strategy):
    """Bar strategy for the batch runner: long while close is above a 20-bar SMA, with a
    fixed stop 1 percent under the entry and the config's trailing stop."""

    def on_start(self, ctx):
        self.sma = {s: ctx.indicators.sma(instrument=s, period=20) for s in ctx.symbols}
        self.bars = 0

    def on_bar(self, ctx):
        self.bars += 1
        sma = self.sma[ctx.symbol]
        if not sma.ready:
            return
        close = ctx.bar.close
        if ctx.is_flat and close > sma.value:
            self.submit_order(orders.Market(side="buy", quantity=QUANTITY, stop_price=round(close * 0.99, 2)))
        elif not ctx.is_flat and close < sma.value:
            self.close_position()


def trade_summary(result) -> dict:
    """Orders, fills and round trips of one PortfolioResult."""
    rs = result.result
    trades = rs.trades()
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    od = rs.orders()
    return {
        "orders": len(od),
        "filled_orders": sum(1 for o in od if o.filled_qty > 0),
        "trades": len(trades),
        "exit_reasons": reasons,
    }
