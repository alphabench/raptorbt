"""Cost of many instruments and many timeframes on one ``LiveNode``.

One node, one strategy trading K symbols (1, 5, 20, 50), with timeframe sets:

* ``1m``: the primary 1-minute bars only;
* ``1m+5m``: plus a 5-minute composite stream per symbol (``subscribe(timeframe=)``)
  with an SMA on it;
* ``1m+5m+15m+1h+1d``: composite streams of each, an SMA on each;
* ``1m+at``: no composite subscription; one ``ctx.indicators.expr`` per symbol that reads
  ``at("5m", ...)``, ``at("15m", ...)``, ``at("1h", ...)`` and ``at("1d", ...)``.

Fed two ways:

* ``ticks``: ticks round-robin over the symbols (every tick its own timestamp), which the
  node aggregates into 1-minute bars (``primary_bars=(1, "m")``); the strategy decides on
  every print. One number is one ``push_tick`` call.
* ``bars``: closed 1-minute bars, one per symbol per timestamp, delivered as one group with
  ``push_group``; the strategy decides on every bar. One number is one ``push_group`` call
  (K bars).

Every push is put in one of three distributions: a push on which a coarser bar closed
(a composite bar reached the strategy, or, for ``1m+at``, the primary bar that opens a new
5-minute window and so moves an ``at`` value), a push on which only a 1-minute bar closed,
and the rest. The strategy reads every indicator it has on every decision, and trades a
fixed 10 units with a fixed stop and the config's trailing stop.

Plus one ``chain`` case: an underlying and 50 option contracts (25 strikes, calls and
puts, one expiry) given as ``instruments=``, every contract receiving ticks; on every
100th print of the underlying the strategy asks ``ctx.option_chain`` for the at-the-money
call and buys or closes it.

    python live_multi.py
    python live_multi.py --quick
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _live  # noqa: E402
from raptorbt import InstrumentSpec, LiveNode, Strategy, pricing  # noqa: E402
from raptorbt.strategy import orders  # noqa: E402

TF_SETS = {
    "1m": [],
    "1m+5m": ["5m"],
    "1m+5m+15m+1h+1d": ["5m", "15m", "1h", "1d"],
    "1m+at": "at",
}
AT_EXPR = ('close > at("5m", sma(close, 5)) and close > at("15m", sma(close, 5)) '
           'and close > at("1h", sma(close, 3)) and close > at("1d", close)')
FIVE_MIN = 5 * _live.MIN_NS


class MultiTF(Strategy):
    def __init__(self, timeframes, on_ticks: bool):
        super().__init__()
        self.timeframes = [] if timeframes == "at" else list(timeframes)
        self.use_at = timeframes == "at"
        self.on_ticks = on_ticks

    def on_init(self, ctx):
        for s in ctx.symbols:
            if self.on_ticks:
                self.subscribe(s, data_type="trade")
            self.subscribe(s, data_type="bar")
            for tf in self.timeframes:
                self.subscribe(s, timeframe=tf, history=32)

    def on_start(self, ctx):
        self.sma = {s: ctx.indicators.sma(instrument=s, period=5) for s in ctx.symbols}
        self.coarse = {s: [ctx.indicators.sma(instrument=s, period=5, timeframe=tf) for tf in self.timeframes]
                       for s in ctx.symbols}
        self.at = {s: ctx.indicators.expr(AT_EXPR, instrument=s) for s in ctx.symbols} if self.use_at else {}
        self.prints = self.primary_bars = self.coarse_bars = self.fills = 0
        self.flag_primary = self.flag_coarse = False

    def on_fill(self, ctx, event):
        self.fills += 1

    def on_bar(self, ctx, event):
        if event.timeframe is not None:
            self.coarse_bars += 1
            self.flag_coarse = True
            return
        self.primary_bars += 1
        self.flag_primary = True
        if self.use_at:
            close_ts = ctx.bar_close_ts if ctx.bar_close_ts is not None else event.timestamp
            if (close_ts - _live.MIN_NS) % FIVE_MIN == 0:
                self.flag_coarse = True
        if not self.on_ticks:
            self.decide(ctx, ctx.symbol, ctx.bar.close)

    def on_trade(self, ctx, tick):
        self.prints += 1
        self.decide(ctx, ctx.symbol, tick.price)

    def decide(self, ctx, symbol, price):
        sma = self.sma[symbol]
        ready = [h.value for h in self.coarse[symbol] if h.ready]
        above = sum(1 for v in ready if price > v)
        if self.use_at:
            h = self.at[symbol]
            above += 1 if h.ready and h.value else 0
        if not ctx.is_flat:
            if not self.on_ticks and sma.ready and price < sma.value:
                self.close_position(symbol=symbol)
            return
        if sma.ready and price > sma.value * 1.0005 and above >= len(ready) // 2:
            self.submit_order(orders.Market(side="buy", quantity=_live.QUANTITY, instrument=symbol,
                                            stop_price=round(price * 0.996, 2)))


def classify(samples, coarse, primary) -> dict:
    other = ~coarse & ~primary
    return {
        "all": _live.percentiles(samples),
        "coarse_close": _live.percentiles(samples[coarse]),
        "primary_close": _live.percentiles(samples[primary & ~coarse]),
        "no_close": _live.percentiles(samples[other]),
    }


def run_ticks(k: int, tf_name: str, total: int, days: int) -> dict:
    symbols = [f"S{i}" for i in range(k)]
    tape = _live.ticks(total, seed=21, days=days)
    ts, ltp = tape["timestamps"].tolist(), tape["ltp"].tolist()
    scale = [1.0 + 0.01 * j for j in range(k)]
    strategy = MultiTF(TF_SETS[tf_name], on_ticks=True)
    node = LiveNode([strategy], symbols, _live.config(), primary_bars=(1, "m"), history=4_096)
    n = len(ts)
    samples = np.empty(n, dtype=np.int64)
    coarse = np.zeros(n, dtype=bool)
    primary = np.zeros(n, dtype=bool)
    push = node.push_tick
    clock = time.perf_counter_ns
    tick = _live.TICK
    gc.collect()
    wall0 = clock()
    for i in range(n):
        j = i % k
        p = round(ltp[i] * scale[j] / tick) * tick
        s, t, b, a = symbols[j], ts[i], p - tick, p + tick
        strategy.flag_primary = strategy.flag_coarse = False
        t0 = clock()
        push(s, t, p, b, a)
        samples[i] = clock() - t0
        coarse[i] = strategy.flag_coarse
        primary[i] = strategy.flag_primary
    wall = clock() - wall0
    memory = node.memory_report()
    summary = _live.trade_summary(node.finish()[0])
    warm = n // 20
    row = {
        "input": "ticks", "symbols": k, "timeframes": tf_name, "pushes": n, "days": days,
        "pushes_per_s": n / (wall / 1e9),
        **classify(samples[warm:], coarse[warm:], primary[warm:]),
        "coarse_pushes": int(coarse.sum()), "primary_pushes": int(primary.sum()),
        "primary_bars": strategy.primary_bars, "coarse_bars": strategy.coarse_bars,
        **summary, "memory_report": memory,
    }
    checks = []
    if strategy.prints != n:
        checks.append(f"the strategy saw {strategy.prints} prints of {n}")
    if strategy.primary_bars == 0:
        checks.append("no 1-minute bar closed")
    if TF_SETS[tf_name] not in ([], "at") and strategy.coarse_bars == 0:
        checks.append("no coarser bar closed")
    if summary["trades"] == 0:
        checks.append("no trade")
    if checks:
        row["error"] = "check failed: " + "; ".join(checks)
    return row


def run_bars(k: int, tf_name: str, days: int) -> dict:
    symbols = [f"S{i}" for i in range(k)]
    n = days * 390
    data = [_live.bars(n, seed=40 + j, start=100.0 + j) for j in range(k)]
    ts = data[0]["timestamps"].tolist()
    cols = [list(zip(*(d[c].tolist() for c in ("open", "high", "low", "close", "volume")))) for d in data]
    strategy = MultiTF(TF_SETS[tf_name], on_ticks=False)
    node = LiveNode([strategy], symbols, _live.config(), bar_stamp="close", history=4_096)
    samples = np.empty(n, dtype=np.int64)
    coarse = np.zeros(n, dtype=bool)
    primary = np.zeros(n, dtype=bool)
    push_group = node.push_group
    clock = time.perf_counter_ns
    gc.collect()
    wall0 = clock()
    for i in range(n):
        group = {symbols[j]: cols[j][i] for j in range(k)}
        strategy.flag_primary = strategy.flag_coarse = False
        t0 = clock()
        push_group(ts[i], group)
        samples[i] = clock() - t0
        coarse[i] = strategy.flag_coarse
        primary[i] = strategy.flag_primary
    wall = clock() - wall0
    memory = node.memory_report()
    summary = _live.trade_summary(node.finish()[0])
    warm = n // 20
    row = {
        "input": "bars", "symbols": k, "timeframes": tf_name, "pushes": n, "bars": n * k, "days": days,
        "pushes_per_s": n / (wall / 1e9),
        **classify(samples[warm:], coarse[warm:], primary[warm:]),
        "coarse_pushes": int(coarse.sum()), "primary_pushes": int(primary.sum()),
        "primary_bars": strategy.primary_bars, "coarse_bars": strategy.coarse_bars,
        **summary, "memory_report": memory,
    }
    checks = []
    if strategy.primary_bars != n * k:
        checks.append(f"the strategy saw {strategy.primary_bars} bars of {n * k}")
    if TF_SETS[tf_name] not in ([], "at") and strategy.coarse_bars == 0:
        checks.append("no coarser bar closed")
    if summary["trades"] == 0:
        checks.append("no trade")
    if checks:
        row["error"] = "check failed: " + "; ".join(checks)
    return row


class ChainTrader(Strategy):
    """Every 100th print of the underlying: look up the at-the-money call and buy it when
    flat, or close what is held."""

    def on_start(self, ctx):
        self.prints = self.underlying_prints = self.lookups = self.fills = 0
        self.held = None

    def on_fill(self, ctx, event):
        self.fills += 1

    def on_trade(self, ctx, tick):
        self.prints += 1
        if ctx.symbol != "U":
            return
        self.underlying_prints += 1
        if self.underlying_prints % 100:
            return
        now = ctx.event_ns
        chain = ctx.option_chain("U")
        expiry = chain.pick_expiry("U", now, "nearest")
        strike = chain.atm_strike("U", expiry, now, tick.price)
        contract = chain.contract("U", expiry, strike, "call", now)
        self.lookups += 1
        if self.held is None:
            self.submit_order(orders.Market(side="buy", quantity=1, instrument=contract.symbol))
            self.held = contract.symbol
        else:
            self.close_position(symbol=self.held)
            self.held = None


def run_chain(total: int) -> dict:
    tape = _live.ticks(total, seed=31)
    ts = tape["timestamps"]
    rounds = total // 51
    spot = tape["ltp"][::51][:rounds]
    stamps = ts[: rounds * 51].reshape(rounds, 51)
    expiry = _live.day_ns(7) + 15 * 3600 * 10**9  # after the data
    strikes = 100.0 + 0.5 * np.arange(-12, 13)
    specs, series = {}, []
    tau = np.maximum(expiry - stamps[:, 0], 0) / (365 * 86_400 * 10**9)
    for right in ("call", "put"):
        for k in strikes:
            sym = f"U-{'C' if right == 'call' else 'P'}-{k:.1f}"
            specs[sym] = InstrumentSpec.option(sym, float(k), right, int(expiry), 1.0, underlying="U")
            prem = np.maximum(np.asarray(pricing.price(right, spot, float(k), tau, 0.2)), _live.TICK)
            series.append((sym, (np.round(prem / _live.TICK) * _live.TICK).tolist()))
    series = series[:50]
    symbols = ["U"] + [s for s, _ in series]
    strategy = ChainTrader()
    node = LiveNode([strategy], symbols, _live.config(), primary_bars=(1, "m"), instruments=specs)
    order = [("U", spot.tolist())] + series
    flat = stamps.tolist()
    samples = np.empty(rounds * 51, dtype=np.int64)
    push = node.push_tick
    clock = time.perf_counter_ns
    tick = _live.TICK
    gc.collect()
    i = 0
    wall0 = clock()
    for r in range(rounds):
        row_ts = flat[r]
        for c, (sym, px) in enumerate(order):
            p = px[r]
            t0 = clock()
            push(sym, row_ts[c], p, max(p - tick, tick), p + tick)
            samples[i] = clock() - t0
            i += 1
    wall = clock() - wall0
    memory = node.memory_report()
    summary = _live.trade_summary(node.finish()[0])
    row = {
        "input": "ticks", "symbols": len(symbols), "timeframes": "chain", "pushes": i,
        "pushes_per_s": i / (wall / 1e9), "all": _live.percentiles(samples[i // 20:]),
        "chain_lookups": strategy.lookups, **summary, "memory_report": memory,
    }
    checks = []
    if strategy.prints != i:
        checks.append(f"the strategy saw {strategy.prints} prints of {i}")
    if summary["trades"] == 0:
        checks.append("no trade")
    if checks:
        row["error"] = "check failed: " + "; ".join(checks)
    return row


def flat_row(r: dict) -> dict:
    out = dict(r)
    for key in ("all", "coarse_close", "primary_close"):
        if isinstance(r.get(key), dict) and r[key].get("n"):
            out[f"{key}_p50_us"] = r[key]["p50_ns"] / 1e3
            out[f"{key}_p99_us"] = r[key]["p99_ns"] / 1e3
            out[f"{key}_n"] = r[key]["n"]
    out["engine_mb"] = r.get("memory_report", {}).get("total", 0) / 1048576
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticks", type=int, default=1_000_000, help="ticks per tick case, over all symbols")
    p.add_argument("--tick-days", type=int, default=5, help="sessions the ticks span (1d bars need more than one)")
    p.add_argument("--bar-days", type=int, default=20, help="sessions of 1-minute bars per bar case")
    p.add_argument("--symbols", default="1,5,20,50")
    p.add_argument("--inputs", default="ticks,bars,chain")
    p.add_argument("--quick", action="store_true", help="a tiny run to check the script works")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if args.quick:
        args.ticks, args.tick_days, args.bar_days, args.symbols = 6_000, 2, 2, "1,5"
    counts = [int(x) for x in args.symbols.split(",")]
    inputs = args.inputs.split(",")

    environment = _live.print_env("LiveNode: many instruments x many timeframes")
    print(f"capital {_live.CAPITAL:,.0f}, orders of {_live.QUANTITY} units, no fees; "
          f"ticks {args.ticks:,} over {args.tick_days} sessions; bars {args.bar_days} sessions\n")
    results: list[dict] = []
    for k in counts:
        for tf in TF_SETS:
            if "ticks" in inputs:
                _live.run_case(f"ticks/sym={k}/{tf}", lambda: run_ticks(k, tf, args.ticks, args.tick_days), results)
            if "bars" in inputs:
                _live.run_case(f"bars/sym={k}/{tf}", lambda: run_bars(k, tf, args.bar_days), results)
    if "chain" in inputs:
        _live.run_case("chain/50 contracts", lambda: run_chain(args.ticks), results)

    rows = [flat_row(r) for r in results]
    print()
    _live.table(rows, [
        ("case", "case"), ("pushes_per_s", "pushes/s"), ("all_p50_us", "p50 us"), ("all_p99_us", "p99 us"),
        ("primary_close_p50_us", "1m p50 us"), ("primary_close_p99_us", "1m p99 us"),
        ("coarse_close_p50_us", "coarse p50 us"), ("coarse_close_p99_us", "coarse p99 us"),
        ("coarse_close_n", "coarse n"), ("trades", "trades"), ("engine_mb", "engine MB"),
    ])
    _live.write_json(args.json or _live.default_json(__file__), {
        "script": "live_multi",
        "env": environment,
        "params": {"ticks": args.ticks, "tick_days": args.tick_days, "bar_days": args.bar_days,
                   "capital": _live.CAPITAL, "quantity": _live.QUANTITY, "quick": args.quick},
        "cases": results,
    })


if __name__ == "__main__":
    main()
