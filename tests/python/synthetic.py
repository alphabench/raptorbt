"""Deterministic synthetic multi-instrument market data (ticks + bars).

Purpose: baseline runtime benchmarks and, later, a correctness oracle for the
engine's own bar aggregation. Pure numpy; never imports raptorbt.

ENGINE FORMAT MAPPING
---------------------
Ticks (``ds.ticks[symbol]``) use exactly the keys ``run_tick_strategy``
consumes (see ``run_strategies(ticks=...)``), all float64 except timestamps:

    timestamps      int64 ns UTC since epoch, non-decreasing (ties allowed:
                    ns resolution after flooring, two arrivals may coincide)
    ltp             trade price (on the tick grid)
    bid, ask        prevailing best quote at the print, on the grid, bid < ask
    bid_qty/ask_qty displayed L1 sizes (positive lot multiples)
    ltq             last traded quantity (positive lot multiple) -> tick.size
    buy_qty_delta   ltq if the print lifted the ask, ltq/2 if at mid, else 0
    sell_qty_delta  ltq if the print hit the bid,  ltq/2 if at mid, else 0
    oi              open interest, positive random walk in lot units

Every row is a trade print carrying the quote that prevailed at it, so the
engine sees one print plus one quote per row.

Bars (``ds.bars[tf][symbol]``) are directly usable by
``run_portfolio_strategy`` (``timestamps/open/high/low/close/volume``, float64
OHLCV, int64 ts) plus ``ts_open`` and ``ts_close``. ``timestamps`` == ``ts_open``
(an open-stamped bar); a harness wanting close-stamped bars should pass
``ts_close`` as ``timestamps`` itself.

BAR CONVENTION (reference oracle) -- READ THIS
----------------------------------------------
Windows are HALF-OPEN ``[ts_open, ts_close)``: a tick at exactly ts_close
belongs to the NEXT window. Intraday windows (1m, 5m, 15m, 1h) are aligned to
the session open (09:15 IST), not to the clock hour: 1h windows are 09:15,
10:15, ... 14:15, and the last one 15:15-15:30 is TRUNCATED (ts_close = session
close, 15 minutes long). 1d is one window per session [open, close). Empty
windows are skipped (no forward fill). ts_open/ts_close are the window
BOUNDARIES, not the first/last tick times. OHLC come from trade prices
(``ltp``), volume = sum of ``ltq``.

The engine's own conventions conflict with each other: ``BarAggregator``
assumes open-stamped bars, while the feed merge in src/data/feed.rs orders
bars after ticks at equal ts (i.e. assumes close stamps). This module takes no
side; both stamps are provided.

DATA MODEL
----------
Sessions: IST 09:15-15:30 (UTC 03:45-10:00), consecutive weekdays from
2026-01-05; no ticks outside [open, close). Price: per-instrument annualised
vol in [15%, 80%], log random walk in session time, occasional jumps, and an
overnight gap return between sessions. The quote reference ``m`` is the mid
snapped to the tick grid; bid = m - s//2, ask = bid + s with spread s >= 1
tick (wider for illiquid names); the trade prints at bid, ask or m. All prices
are integer multiples of tick_size (stored as ``int * tick``).
Liquidity is lognormal across instruments, normalised so the mean rate is
``avg_ticks_per_sec``; arrivals are a non-homogeneous Poisson process with a
U-shaped intraday intensity (exact inverse-CDF sampling, vectorised).
Each instrument draws from its own ``SeedSequence(seed).spawn(n)`` child, so
output is byte-identical for equal arguments (same numpy build).

Memory: ticks are 10 float/int64 columns = 80 bytes/row. The L tier (~56M
rows) is ~4.5 GB resident when fully materialised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GENERATOR_VERSION = "1"

TIMEFRAMES = ("1m", "5m", "15m", "1h", "1d")
TIERS = {
    "S": dict(n_instruments=250, days=1, avg_ticks_per_sec=0.2),
    "M": dict(n_instruments=250, days=5, avg_ticks_per_sec=0.2),
    "L": dict(n_instruments=500, days=5, avg_ticks_per_sec=1.0),
}
DEFAULT_SEED = 20261101

NS = 1_000_000_000
DAY_NS = 86_400 * NS
IST_OFFSET_NS = 19_800 * NS
SESSION_OPEN_LOCAL_NS = (9 * 3600 + 15 * 60) * NS  # 09:15 local time-of-day
SESSION_LEN_NS = (6 * 3600 + 15 * 60) * NS  # 09:15 -> 15:30
_TF_NS = {"1m": 60 * NS, "5m": 300 * NS, "15m": 900 * NS, "1h": 3600 * NS, "1d": SESSION_LEN_NS}
_FIRST_DAY = np.datetime64("2026-01-05")

TICK_FIELDS = (
    "timestamps", "ltp", "bid", "ask", "bid_qty", "ask_qty",
    "ltq", "buy_qty_delta", "sell_qty_delta", "oi",
)
BAR_FIELDS = ("timestamps", "ts_open", "ts_close", "open", "high", "low", "close", "volume")

_TICK_SIZES = (0.05, 0.1, 0.5, 1.0, 5.0)
_LOT_SIZES = (1, 15, 25, 50, 75, 250, 500)


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    lot_size: int
    multiplier: float
    base_price: float


@dataclass
class Dataset:
    instruments: list[Instrument]
    ticks: dict[str, dict[str, np.ndarray]]
    bars: dict[str, dict[str, dict[str, np.ndarray]]]
    manifest: dict


# --------------------------------------------------------------------------
# Intensity profile (U-shape), shared by all instruments
# --------------------------------------------------------------------------
_GRID = np.linspace(0.0, 1.0, 2049)
_F = 1.0 + 3.0 * (np.exp(-_GRID / 0.07) + np.exp(-(1.0 - _GRID) / 0.07))
_CDF = np.concatenate([[0.0], np.cumsum((_F[1:] + _F[:-1]) / 2.0)])
_CDF /= _CDF[-1]


def _session_opens(days: int) -> np.ndarray:
    """UTC ns of each session open: consecutive weekdays from 2026-01-05."""
    out, d = [], _FIRST_DAY
    while len(out) < days:
        if (d.astype("datetime64[D]").astype(int) + 3) % 7 < 5:  # 1970-01-01 was Thu
            out.append(int(d.astype(int)) * DAY_NS + SESSION_OPEN_LOCAL_NS - IST_OFFSET_NS)
        d = d + np.timedelta64(1, "D")
    return np.array(out, dtype=np.int64)


def _make_instruments(n: int, rng: np.random.Generator):
    base = np.exp(rng.uniform(np.log(20.0), np.log(50000.0), n))
    weights = rng.lognormal(0.0, 1.0, n)
    weights /= weights.mean()
    vols = rng.uniform(0.15, 0.80, n)
    lots = rng.choice(_LOT_SIZES, n)
    fut = rng.random(n) < 0.2
    tick_pick = rng.random(n)
    ticks = np.array(_TICK_SIZES)
    instruments = []
    for i in range(n):
        ratio = ticks / base[i]
        ok = ticks[(ratio >= 1e-4) & (ratio <= 1e-3)]
        if len(ok) == 0:  # nearest in log space to 3e-4
            ok = ticks[[np.argmin(np.abs(np.log(ratio / 3e-4)))]]
        tick = float(ok[int(tick_pick[i] * len(ok))])
        instruments.append(Instrument(
            symbol=f"SYN{i:03d}", tick_size=tick, lot_size=int(lots[i]),
            multiplier=float(lots[i]) if fut[i] else 1.0,
            base_price=float(max(round(base[i] / tick), 10) * tick)))
    return instruments, weights, vols


def _gen_instrument(inst, rate, vol, ss, opens) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(ss)
    sig_s = vol / np.sqrt(252.0)  # per-session sigma
    tick, lot = inst.tick_size, inst.lot_size
    cur_log = np.log(inst.base_price)
    liq_spread = float(np.clip(0.3 * rate ** -0.7 if rate > 0 else 6.0, 0.02, 6.0))
    ts_p, m_p, s_p = [], [], []
    for d, open_ns in enumerate(opens):
        n = int(rng.poisson(rate * SESSION_LEN_NS / NS))
        u = np.interp(np.sort(rng.random(n)), _CDF, _GRID)
        ts = np.minimum(open_ns + (u * SESSION_LEN_NS).astype(np.int64),
                        open_ns + SESSION_LEN_NS - 1)
        ret = sig_s * np.sqrt(np.diff(u, prepend=0.0)) * rng.standard_normal(n)
        k = int(rng.poisson(1.0))
        if n:
            np.add.at(ret, rng.integers(0, n, k), rng.normal(0.0, 3.0 * sig_s, k))
        if d > 0:
            cur_log += rng.normal(0.0, 0.7 * sig_s)  # overnight gap
        logp = cur_log + np.cumsum(ret)
        if n:
            cur_log = float(logp[-1])
        m = np.maximum(np.rint(np.exp(logp) / tick), 10).astype(np.int64)
        ts_p.append(ts)
        m_p.append(m)
        s_p.append(1 + rng.poisson(liq_spread, n))
    ts = np.concatenate(ts_p)
    m = np.concatenate(m_p)
    s = np.concatenate(s_p).astype(np.int64)
    n = len(ts)
    bid_i = m - s // 2
    ask_i = bid_i + s
    r = rng.random(n)
    ltp_i = np.where(r < 0.4, ask_i, np.where(r < 0.8, bid_i, m))
    ltq = lot * (1 + np.floor(rng.lognormal(0.0, 1.0, n))).astype(np.float64)
    at_ask, at_bid = r < 0.4, (r >= 0.4) & (r < 0.8)
    oi0 = lot * int(rng.integers(1_000, 100_000))
    oi = np.maximum(oi0 + np.cumsum(lot * rng.integers(-2, 3, n)), lot).astype(np.float64)
    return {
        "timestamps": ts,
        "ltp": ltp_i * tick,
        "bid": bid_i * tick,
        "ask": ask_i * tick,
        "bid_qty": lot * (1 + np.floor(rng.lognormal(2.0, 1.0, n))),
        "ask_qty": lot * (1 + np.floor(rng.lognormal(2.0, 1.0, n))),
        "ltq": ltq,
        "buy_qty_delta": np.where(at_ask, ltq, np.where(at_bid, 0.0, ltq / 2)),
        "sell_qty_delta": np.where(at_bid, ltq, np.where(at_ask, 0.0, ltq / 2)),
        "oi": oi,
    }


# --------------------------------------------------------------------------
# Reference bar aggregation (oracle)
# --------------------------------------------------------------------------
def aggregate_bars(ticks: dict[str, np.ndarray], timeframe: str,
                   session_tz_offset_ns: int, session_open_ns: int) -> dict[str, np.ndarray]:
    """Aggregate one instrument's ticks into half-open session-aligned bars.

    ``session_tz_offset_ns``: local - UTC (IST = 19_800e9).
    ``session_open_ns``: session open as LOCAL time-of-day, ns since local
    midnight (09:15 = 33_300e9). The session length is fixed at 6h15m.
    Ticks outside [open, open+6h15m) raise ValueError. See module docstring
    for the full convention.
    """
    if timeframe not in _TF_NS:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    step = _TF_NS[timeframe]
    ts = np.asarray(ticks["timestamps"], dtype=np.int64)
    empty = np.empty(0, dtype=np.float64)
    if len(ts) == 0:
        e = np.empty(0, dtype=np.int64)
        return {"timestamps": e, "ts_open": e, "ts_close": e, "open": empty, "high": empty,
                "low": empty, "close": empty, "volume": empty}
    local = ts + session_tz_offset_ns
    day = local // DAY_NS
    tod = local - day * DAY_NS - session_open_ns
    if tod.min() < 0 or tod.max() >= SESSION_LEN_NS:
        raise ValueError("tick outside session window")
    w = tod // step
    key = day * 4096 + w
    starts = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1])
    ends = np.concatenate([starts[1:], [len(ts)]])
    d, wk = day[starts], w[starts]
    ts_open = d * DAY_NS + session_open_ns + wk * step - session_tz_offset_ns
    sess_close = d * DAY_NS + session_open_ns + SESSION_LEN_NS - session_tz_offset_ns
    ts_close = np.minimum(ts_open + step, sess_close)
    ltp = ticks["ltp"]
    return {
        "timestamps": ts_open.copy(), "ts_open": ts_open, "ts_close": ts_close,
        "open": ltp[starts].astype(np.float64), "high": np.maximum.reduceat(ltp, starts),
        "low": np.minimum.reduceat(ltp, starts), "close": ltp[ends - 1].astype(np.float64),
        "volume": np.add.reduceat(ticks["ltq"], starts),
    }


# --------------------------------------------------------------------------
# Manifest / hashing
# --------------------------------------------------------------------------
def _sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _flat(ds_ticks, ds_bars) -> dict[str, np.ndarray]:
    out = {}
    for sym, f in ds_ticks.items():
        for k, a in f.items():
            out[f"ticks/{sym}/{k}"] = a
    for tf, per in ds_bars.items():
        for sym, f in per.items():
            for k, a in f.items():
                out[f"bars/{tf}/{sym}/{k}"] = a
    return out


def _build_manifest(seed, params, instruments, ticks, bars, arrays_sha) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "params": params,
        "instruments": [asdict(i) for i in instruments],
        "arrays": arrays_sha,
        "row_counts": {
            "ticks": {s: len(f["timestamps"]) for s, f in ticks.items()},
            "ticks_total": int(sum(len(f["timestamps"]) for f in ticks.values())),
            "bars": {tf: {s: len(f["ts_open"]) for s, f in per.items()} for tf, per in bars.items()},
        },
    }


def verify(ds: Dataset) -> None:
    """Raise ValueError if any array differs from its manifest hash."""
    flat = _flat(ds.ticks, ds.bars)
    want = ds.manifest["arrays"]
    if set(flat) != set(want):
        diff = sorted(set(flat) ^ set(want))[:5]
        raise ValueError(f"manifest/array key mismatch, e.g. {diff}")
    for k, a in flat.items():
        if _sha(a) != want[k]:
            raise ValueError(f"array {k!r} does not match its manifest sha256 (tampered or corrupt)")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def generate(n_instruments=250, days=1, avg_ticks_per_sec=0.2, seed=DEFAULT_SEED,
             timeframes=TIMEFRAMES) -> Dataset:
    # Global attributes come from their own stream so they don't shift with n.
    g = np.random.default_rng(np.random.SeedSequence([seed, 0x5EED]))
    instruments, weights, vols = _make_instruments(n_instruments, g)
    children = np.random.SeedSequence(seed).spawn(n_instruments)
    opens = _session_opens(days)
    ticks, bars = {}, {tf: {} for tf in timeframes}
    for inst, w, vol, ss in zip(instruments, weights, vols, children):
        t = _gen_instrument(inst, avg_ticks_per_sec * w, float(vol), ss, opens)
        ticks[inst.symbol] = t
        for tf in timeframes:
            bars[tf][inst.symbol] = aggregate_bars(t, tf, IST_OFFSET_NS, SESSION_OPEN_LOCAL_NS)
    flat = _flat(ticks, bars)
    params = dict(n_instruments=n_instruments, days=days, avg_ticks_per_sec=avg_ticks_per_sec,
                  timeframes=list(timeframes))
    manifest = _build_manifest(seed, params, instruments, ticks, bars,
                               {k: _sha(a) for k, a in flat.items()})
    return Dataset(instruments, ticks, bars, manifest)


def save(ds: Dataset, path) -> None:
    """Write ``<path>/data.npz`` and ``<path>/manifest.json``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / "data.npz", **_flat(ds.ticks, ds.bars))
    (p / "manifest.json").write_text(json.dumps(ds.manifest, indent=1, sort_keys=True))


def load(path) -> Dataset:
    """Load and verify (every array's sha256) a dataset written by ``save``."""
    p = Path(path)
    manifest = json.loads((p / "manifest.json").read_text())
    ticks: dict = {}
    bars: dict = {tf: {} for tf in manifest["params"]["timeframes"]}
    with np.load(p / "data.npz") as z:
        for key in z.files:
            parts = key.split("/")
            if parts[0] == "ticks":
                ticks.setdefault(parts[1], {})[parts[2]] = z[key]
            else:
                bars.setdefault(parts[1], {}).setdefault(parts[2], {})[parts[3]] = z[key]
    ds = Dataset([Instrument(**i) for i in manifest["instruments"]], ticks, bars, manifest)
    verify(ds)
    return ds


def load_tier(name: str, cache_dir: str | None = None) -> Dataset:
    """Generate-or-load a tier. A cache that fails verification raises."""
    if name not in TIERS:
        raise ValueError(f"unknown tier {name!r}; choose from {sorted(TIERS)}")
    root = Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache"
    d = root / f"{name}-seed{DEFAULT_SEED}-v{GENERATOR_VERSION}"
    if (d / "manifest.json").exists():
        ds = load(d)
        want = dict(TIERS[name], timeframes=list(TIMEFRAMES))
        if ds.manifest["params"] != want or ds.manifest["seed"] != DEFAULT_SEED:
            raise ValueError(f"cache at {d} does not match tier {name} parameters; delete it")
        return ds
    ds = generate(**TIERS[name])
    save(ds, d)
    return ds


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", choices=sorted(TIERS), default="S")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    t0 = time.perf_counter()
    ds = generate(**TIERS[a.tier], seed=a.seed)
    dt = time.perf_counter() - t0
    flat = _flat(ds.ticks, ds.bars)
    tick_b = sum(x.nbytes for k, x in flat.items() if k.startswith("ticks/"))
    bar_b = sum(x.nbytes for k, x in flat.items() if k.startswith("bars/"))
    rc = ds.manifest["row_counts"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
    print(f"tier {a.tier}: {len(ds.instruments)} instruments, generated in {dt:.2f}s")
    print(f"ticks: {rc['ticks_total']:,} rows, {tick_b/1e6:.1f} MB")
    for tf, per in rc["bars"].items():
        print(f"bars {tf}: {sum(per.values()):,} rows")
    print(f"bars total {bar_b/1e6:.1f} MB; peak RSS {rss_mb:.0f} MB")


if __name__ == "__main__":
    _main()
