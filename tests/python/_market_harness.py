"""Helpers for the market-structure tests: fees by instrument, sessions, defaults.

A scripted strategy that submits what a dict says on the n-th bar of a symbol,
run three ways (the batch runner, a live node, a live node on paper
execution), and IST wall-clock helpers for building a tape.
"""

import datetime as dt

import numpy as np

import raptorbt as rb
from raptorbt import BacktestConfig, LiveNode, Strategy, run_strategies
from raptorbt.execution import PaperExecution

MIN = 60_000_000_000
IST = 19_800 * 10**9


def ist(y, m, d, hh, mm):
    """UTC nanoseconds of an IST wall-clock time."""
    zone = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return int(dt.datetime(y, m, d, hh, mm, tzinfo=zone).timestamp()) * 10**9


def fmt(ns):
    """``MM-DD HH:MM`` in IST."""
    zone = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return dt.datetime.fromtimestamp(ns / 1e9, zone).strftime("%m-%d %H:%M")


def bars(rows, stamps):
    """OHLC rows on close stamps: a list of ns, or ``t0`` for one-minute bars after it."""
    a = np.array(rows, dtype=float)
    if isinstance(stamps, list):
        ts = np.array(stamps, dtype=np.int64)
    else:
        ts = np.array([stamps + (i + 1) * MIN for i in range(len(rows))], dtype=np.int64)
    return {
        "timestamps": ts,
        "open": a[:, 0],
        "high": a[:, 1],
        "low": a[:, 2],
        "close": a[:, 3],
        "volume": np.full(len(rows), 1e6),
    }


class Script(Strategy):
    """Runs ``script[(symbol, n)]`` or ``script[n]`` on the n-th bar of a symbol."""

    def __init__(self, script):
        super().__init__()
        self.script, self.n = script, {}
        self.events = []
        self.rejections = []

    def on_bar(self, ctx):
        n = self.n.get(ctx.symbol, 0)
        self.n[ctx.symbol] = n + 1
        act = self.script.get((ctx.symbol, n)) or self.script.get(n)
        if act:
            act(self, ctx)

    def on_order_event(self, ctx, e):
        self.events.append(e.kind)

    def on_order_rejected(self, ctx, e):
        self.rejections.append((fmt(ctx.bar_close_ts), e.reject_reason))


def run_paths(script, data, config, paths=("batch", "node", "paper"), **kw):
    """``{path: (strategy, result)}`` for the batch runner, a live node, and a node on paper."""
    out = {}
    symbols = list(data)
    for path in paths:
        strategy = Script(script)
        if path == "batch":
            res = run_strategies([strategy], data, config(), bar_stamp="close", **kw)[0]
        else:
            execution = PaperExecution() if path == "paper" else None
            node = LiveNode(
                [strategy], symbols, config(), bar_stamp="close", execution=execution, **kw
            )
            tape = sorted(
                (int(data[s]["timestamps"][i]), k, i)
                for k, s in enumerate(symbols)
                for i in range(len(data[s]["timestamps"]))
            )
            for ts, k, i in tape:
                a = data[symbols[k]]
                node.push_bar(
                    symbols[k], ts, *(float(a[c][i]) for c in ("open", "high", "low", "close", "volume"))
                )
            res = node.finish()[0]
        out[path] = (strategy, res)
    return out


def trade_rows(res):
    """``(symbol, entry, exit, size, fees, pnl)`` per trade of a result."""
    return [
        (t.symbol, t.entry_price, t.exit_price, t.size, t.fees, round(t.pnl, 2))
        for t in res.result.trades()
    ]
