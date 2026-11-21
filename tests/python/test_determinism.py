"""One input, one result: twenty runs of the same backtest hash to one digest.

The construction is the published harness's determinism section
(``benches/python/run_all.py``): a seeded bounded random walk, SMA crossover
signals, ``run_single_backtest``. The digest covers every float of the curves
bit for bit, every trade and every metric. It compares runs on one machine
with each other, so it pins no value and holds on any platform.
"""

import hashlib
import json

import numpy as np

import raptorbt as r

RUNS = 20


def _bars(n, seed=7):
    rng = np.random.default_rng(seed)
    logp = np.clip(np.cumsum(rng.normal(0, 0.001, n)), -1.0, 1.0) + np.log(1000.0)
    close = np.exp(logp)
    spread = close * 0.0008
    o = close + rng.normal(0, spread / 2)
    h = np.maximum(o, close) + np.abs(rng.normal(0, spread))
    lo = np.minimum(o, close) - np.abs(rng.normal(0, spread))
    v = np.abs(rng.normal(1e5, 2e4, n))
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000
    return ts, o, h, lo, close, v


def _sma_signals(close, fast=10, slow=30):
    def sma(a, w):
        c = np.cumsum(np.insert(a, 0, 0.0))
        out = np.full(len(a), np.nan)
        out[w - 1:] = (c[w:] - c[:-w]) / w
        return out

    f, s = sma(close, fast), sma(close, slow)
    up = (f > s) & ~np.isnan(f) & ~np.isnan(s)
    entries = np.zeros(len(close), bool)
    exits = np.zeros(len(close), bool)
    entries[1:] = up[1:] & ~up[:-1]
    exits[1:] = ~up[1:] & up[:-1]
    return entries, exits


def test_twenty_runs_hash_to_one_digest():
    ts, o, h, lo, c, v = _bars(50_000)
    e, x = _sma_signals(c)
    config = r.BacktestConfig(initial_capital=100_000.0, fees=0.0002)

    def digest():
        res = r.run_single_backtest(
            timestamps=ts, open=o, high=h, low=lo, close=c, volume=v,
            entries=e, exits=x, direction=1, weight=1.0, symbol="B", config=config)
        trades = [
            [t.entry_idx, t.exit_idx, t.entry_price.hex(), t.exit_price.hex(), t.size.hex(),
             t.fees.hex(), t.pnl.hex()]
            for t in res.trades()
        ]
        payload = json.dumps(
            {"trades": trades, "metrics": res.metrics.to_dict()}, sort_keys=True, default=repr
        ).encode()
        sha = hashlib.sha256(payload)
        for curve in (res.equity_curve(), res.drawdown_curve(), res.returns()):
            sha.update(np.ascontiguousarray(curve, dtype=np.float64).tobytes())
        return sha.hexdigest(), len(trades)

    runs = [digest() for _ in range(RUNS)]
    assert runs[0][1] > 100, "the run trades enough for the digest to mean something"
    digests = {d for d, _ in runs}
    assert len(digests) == 1, f"{RUNS} runs gave {len(digests)} digests"
