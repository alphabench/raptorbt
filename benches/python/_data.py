import numpy as np

def bars(n, seed=7):
    """Bounded random walk -- clipped in log space so 25M bars stay finite."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.001, n)
    logp = np.cumsum(steps)
    logp = np.clip(logp, -1.0, 1.0) + np.log(1000.0)
    close = np.exp(logp)
    spread = close * 0.0008
    o = close + rng.normal(0, spread / 2)
    h = np.maximum(o, close) + np.abs(rng.normal(0, spread))
    l = np.minimum(o, close) - np.abs(rng.normal(0, spread))
    v = np.abs(rng.normal(1e5, 2e4, n))
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000
    return ts, o, h, l, close, v

def sma_signals(close, fast=10, slow=30):
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
