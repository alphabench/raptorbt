"""Deterministic synthetic market data shared by the examples.

Nothing here touches the network or the clock: every function takes a seed and
returns the same arrays on every run. Bars are stamped at their **close** in
nanoseconds since the epoch (``bar_stamp="close"``), one per minute, 09:16 to
15:30 IST on consecutive weekdays from Monday 2026-01-05.

An example imports it with::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import _data
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from raptorbt import InstrumentSpec, pricing

MIN_NS = 60 * 10**9
DAY_NS = 86_400 * 10**9
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
IST_OFFSET_NS = 19_800 * 10**9
MINUTES_PER_DAY = 375  # 09:15 to 15:30
FIRST_DAY = dt.date(2026, 1, 5)  # a Monday


def ist_ns(date: dt.date, hour: int = 0, minute: int = 0) -> int:
    """Nanoseconds since the epoch of a wall-clock time in IST."""
    return int(dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=IST).timestamp()) * 10**9


def session_closes(days: int = 1) -> np.ndarray:
    """Close stamps of the 1-minute bars of ``days`` sessions: 09:16, 09:17, ... 15:30 IST."""
    out, day = [], FIRST_DAY
    while len(out) < days:
        if day.weekday() < 5:
            out.append(ist_ns(day, 9, 15) + np.arange(1, MINUTES_PER_DAY + 1, dtype=np.int64) * MIN_NS)
        day += dt.timedelta(days=1)
    return np.concatenate(out)


def walk(n: int, seed: int, start: float, step: float, drift_period: int = 90) -> np.ndarray:
    """A random walk whose drift swings up and down, so trend followers find trades.

    ``step`` is the standard deviation of one move; the drift is a quarter of ``step`` at its
    strongest and flips sign every half ``drift_period`` bars.
    """
    rng = np.random.default_rng(seed)
    drift = 0.25 * step * np.sin(2 * np.pi * np.arange(n) / drift_period)
    return start + np.cumsum(drift + rng.normal(0.0, step, n))


def ohlcv(close: np.ndarray, seed: int, wick: float, volume: float = 1_000.0) -> dict[str, np.ndarray]:
    """OHLCV arrays around a close series: each open is the previous close."""
    rng = np.random.default_rng(seed + 1_000)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.0, wick, len(close))
    low = np.minimum(open_, close) - rng.uniform(0.0, wick, len(close))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": np.full(len(close), volume)}


def bars(days: int = 1, seed: int = 1, start: float = 100.0, step: float = 0.08, wick: float = 0.05,
         drift_period: int = 90) -> dict[str, np.ndarray]:
    """1-minute bars for ``days`` sessions, close-stamped, ready for ``run_strategies``."""
    ts = session_closes(days)
    close = walk(len(ts), seed, start, step, drift_period)
    return {"timestamps": ts, **ohlcv(close, seed, wick)}


def ticks(n: int = 600, seed: int = 3, start: float = 100.0, tick: float = 0.05) -> dict[str, np.ndarray]:
    """A one-second quote stream on a ``tick`` grid: ``timestamps``, ``ltp``, ``bid``, ``ask``."""
    rng = np.random.default_rng(seed)
    ltp = np.round((start + np.cumsum(rng.normal(0.0, tick * 2, n))) / tick) * tick
    ts = ist_ns(FIRST_DAY, 9, 15) + np.arange(1, n + 1, dtype=np.int64) * 10**9
    return {"timestamps": ts, "ltp": ltp, "bid": ltp - tick, "ask": ltp + tick}


NIFTY_LOT = 75.0
NIFTY_VOL = 0.14
NIFTY_RATE = 0.065


def option_symbol(strike: float, right: str, expiry: dt.date) -> str:
    return f"NIFTY{expiry:%d%b%y}".upper() + f"{int(strike)}{'CE' if right == 'call' else 'PE'}"


def nifty_universe(days: int = 2, seed: int = 5, spot: float = 24_000.0, step: float = 3.0,
                   half_width: int = 6, expiries: tuple[dt.date, ...] = (dt.date(2026, 1, 8), dt.date(2026, 1, 15))):
    """A synthetic NIFTY index and its weekly option chain.

    Returns ``(data, instruments, specs)``: bars for the index (``"NIFTY"``) and for every
    listed contract, the ``InstrumentSpec`` of each symbol, and the option specs alone (for
    an ``OptionChain``). Strikes run ``half_width`` steps of 50 either side of ``spot``; each
    premium is the Black-Scholes price at a flat 14 percent volatility, so option bars move
    with the index and decay toward expiry. Contracts expire 15:30 IST on their date.
    """
    index = bars(days=days, seed=seed, start=spot, step=step, wick=1.0, drift_period=180)
    ts, close = index["timestamps"], index["close"]
    data = {"NIFTY": index}
    instruments = {}  # the index is a plain series: a run refuses an index spec as "not tradable"
    specs = []
    centre = round(spot / 50.0) * 50.0
    for expiry in expiries:
        expiry_ns = ist_ns(expiry, 15, 30)
        tau = np.maximum(expiry_ns - ts, 0) / (365 * DAY_NS)
        for strike in centre + 50.0 * np.arange(-half_width, half_width + 1):
            for right in ("call", "put"):
                symbol = option_symbol(strike, right, expiry)
                premium = np.maximum(
                    pricing.price(right, close, strike, tau, NIFTY_VOL, rate=NIFTY_RATE), 0.05
                )
                data[symbol] = {"timestamps": ts, **ohlcv(premium, seed, wick=0.2)}
                spec = InstrumentSpec.option(
                    symbol, float(strike), right, expiry_ns, NIFTY_LOT, underlying="NIFTY",
                    span_pct=0.1, exposure_pct=0.02,
                )
                instruments[symbol] = spec
                specs.append(spec)
    return data, instruments, specs
