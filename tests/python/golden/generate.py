"""Regenerate golden backtest fixtures.

Run from the repo root AFTER building the extension::

    .venv/bin/python raptorbt/tests/python/golden/generate.py

Fixtures pin bit-exact results (float hex) for a corpus of runs across the
array and class paths. ``test_golden.py`` replays the corpus and asserts
equality, gating any refactor of the execution core. Regenerating fixtures
is a deliberate act: it declares that numeric results are allowed to change
and requires a version bump + changelog entry per the compatibility rules.

**Inputs are frozen alongside the outputs.** ``make_data`` builds its series
through ``np.exp``/``np.cumsum``, whose vectorized kernels are not correctly
rounded -- they dispatch on CPU features and have changed between NumPy
releases. Regenerating the inputs at test time therefore made the gate
compare *NumPy composed with the engine*, so the same commit passed on one
runner and failed by 1-2 ULP on another. The arrays are written into
``fixtures.json`` as float hex and replayed verbatim, which is what confines
the gate to the Rust core it exists to protect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import raptorbt
from raptorbt import PyBacktestConfig, PyInstrumentConfig

HERE = Path(__file__).parent


def make_data(n=400, seed=7):
    """Deterministic synthetic OHLCV with trends, chop, and gaps."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.012, n)
    steps[::37] += 0.03  # occasional gaps
    steps[::53] -= 0.035
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[100.0], close[:-1] * (1 + rng.normal(0, 0.002, n - 1))])
    spread = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(10_000, 1_000_000, n).astype(np.float64)
    # Ns timestamps, one bar per minute.
    ts = (1_700_000_000_000_000_000 + np.arange(n) * 60_000_000_000).astype(np.int64)
    return ts, open_, high, low, close, volume


def freeze_inputs(ts, o, h, l, c, v, entries, exits):
    """Serialize one instrument's arrays exactly, for replay without NumPy."""
    return {
        "ts": [int(x) for x in ts],
        "open": [float.hex(float(x)) for x in o],
        "high": [float.hex(float(x)) for x in h],
        "low": [float.hex(float(x)) for x in l],
        "close": [float.hex(float(x)) for x in c],
        "volume": [float.hex(float(x)) for x in v],
        "entries": [bool(x) for x in entries],
        "exits": [bool(x) for x in exits],
    }


def thaw_inputs(frozen):
    """Rebuild the arrays written by :func:`freeze_inputs`, bit-for-bit."""
    return (
        np.array(frozen["ts"], dtype=np.int64),
        np.array([float.fromhex(x) for x in frozen["open"]], dtype=np.float64),
        np.array([float.fromhex(x) for x in frozen["high"]], dtype=np.float64),
        np.array([float.fromhex(x) for x in frozen["low"]], dtype=np.float64),
        np.array([float.fromhex(x) for x in frozen["close"]], dtype=np.float64),
        np.array([float.fromhex(x) for x in frozen["volume"]], dtype=np.float64),
        np.array(frozen["entries"], dtype=bool),
        np.array(frozen["exits"], dtype=bool),
    )


def make_signals(close, fast=10, slow=30):
    fast_ma = raptorbt.sma(close, fast)
    slow_ma = raptorbt.sma(close, slow)
    with np.errstate(invalid="ignore"):
        above = fast_ma > slow_ma
        below = fast_ma < slow_ma
    entries = above & ~np.roll(above, 1)
    exits = below & ~np.roll(below, 1)
    entries[0] = exits[0] = False
    return entries.astype(bool), exits.astype(bool)


def result_digest(result):
    """Exact-float digest of a backtest result."""
    return {
        "equity_curve": [float.hex(float(x)) for x in result.equity_curve()],
        "trades": [
            {
                "entry_idx": t.entry_idx,
                "exit_idx": t.exit_idx,
                "entry_price": float.hex(t.entry_price),
                "exit_price": float.hex(t.exit_price),
                "size": float.hex(t.size),
                "pnl": float.hex(t.pnl),
                "fees": float.hex(t.fees),
                "exit_reason": t.exit_reason,
            }
            for t in result.trades()
        ],
        "sharpe": float.hex(result.metrics.sharpe_ratio),
        "total_return_pct": float.hex(result.metrics.total_return_pct),
        "max_drawdown_pct": float.hex(result.metrics.max_drawdown_pct),
    }


def config_variants():
    """(name, config, instrument_config, direction) corpus for the single path."""
    variants = []

    variants.append(("default_long", PyBacktestConfig(), None, 1))
    variants.append(("default_short", PyBacktestConfig(), None, -1))

    c = PyBacktestConfig()
    c.set_fixed_stop(0.03)
    c.set_fixed_target(0.06)
    variants.append(("fixed_stop_target", c, None, 1))

    c = PyBacktestConfig()
    c.set_trailing_stop(0.04)
    variants.append(("trailing_stop", c, None, 1))

    c = PyBacktestConfig()
    c.set_atr_stop(2.0, 14)
    c.set_risk_reward_target(2.0)
    variants.append(("atr_stop_rr_target", c, None, 1))

    c = PyBacktestConfig()
    c.fee_segment = "NFO-FUT"
    variants.append(("indian_fees_nfo", c, None, 1))

    c = PyBacktestConfig()
    c.slippage = 0.001
    variants.append(("slippage_pct", c, None, 1))

    c = PyBacktestConfig()
    c.max_positions = 1
    c.max_drawdown_pct = 15.0
    variants.append(("risk_gated", c, None, 1))

    ic = PyInstrumentConfig(lot_size=50.0, alloted_capital=60_000.0)
    variants.append(("lots_and_cap", PyBacktestConfig(), ic, 1))

    return variants


class GoldenSma(raptorbt.Strategy):
    """Class-path twin of the array SMA cross."""

    def on_start(self, ctx):
        self.fast = raptorbt.sma(ctx.close, 10)
        self.slow = raptorbt.sma(ctx.close, 30)

    def on_bar(self, ctx):
        i = ctx.idx
        if i == 0 or np.isnan(self.slow[i]):
            return
        above = self.fast[i] > self.slow[i]
        was_above = self.fast[i - 1] > self.slow[i - 1]
        if above and not was_above and ctx.position is None:
            self.enter()
        elif not above and was_above and ctx.position is not None:
            self.close_position()


def generate():
    ts, o, h, l, c, v = make_data()
    entries, exits = make_signals(c)
    fixtures = {"inputs": {"shared": freeze_inputs(ts, o, h, l, c, v, entries, exits)}}

    for name, config, ic, direction in config_variants():
        result = raptorbt.run_single_backtest(
            ts, o, h, l, c, v, entries, exits,
            direction=direction, config=config, instrument_config=ic,
        )
        fixtures[f"single/{name}"] = result_digest(result)

    fixtures["class/sma_cross"] = result_digest(
        raptorbt.run_strategy_backtest(GoldenSma, ts, o, h, l, c, v)
    )

    # Portfolio: three instruments sharing one capital pool.
    instruments = []
    for seed in (11, 12, 13):
        pts, po, ph, pl, pc, pv = make_data(300, seed=seed)
        pe, px = make_signals(pc)
        fixtures["inputs"][f"SYM{seed}"] = freeze_inputs(
            pts, po, ph, pl, pc, pv, pe, px
        )
        instruments.append((pts, po, ph, pl, pc, pv, pe, px, 1, 1.0, f"SYM{seed}"))
    portfolio = raptorbt.run_portfolio_backtest(
        instruments, config=PyBacktestConfig(), allocation="equal_weight"
    )
    fixtures["portfolio/shared_pool"] = {
        "equity_curve": [float.hex(float(x)) for x in portfolio.result.equity_curve()],
        "total_return_pct": float.hex(portfolio.metrics.total_return_pct),
        "per_instrument": {
            s.symbol: {"trades": s.trades, "pnl": float.hex(s.pnl)}
            for s in portfolio.per_instrument
        },
    }

    out = HERE / "fixtures.json"
    out.write_text(json.dumps(fixtures, indent=1, sort_keys=True))
    print(f"wrote {out} ({len(fixtures)} fixtures)")


if __name__ == "__main__":
    generate()
