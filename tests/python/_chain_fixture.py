"""A synthetic option universe for the structure tests: one index, strikes 90..110 by 5, two
expiries, bars for every contract on one time grid."""

import numpy as np

from raptorbt import InstrumentSpec

DAY = 86_400 * 10**9
T0 = 1_767_225_600 * 10**9  # 2026-01-01 00:00 UTC
STEP = 60 * 10**9
LOT = 50.0
U = "IDX"
STRIKES = [90.0, 95.0, 100.0, 105.0, 110.0]
N_BARS = 40
EXP1 = T0 + (N_BARS + 10) * STEP
EXP2 = T0 + (N_BARS + 20) * STEP


def symbol(strike, right, expiry):
    return f"{U}{'A' if expiry == EXP1 else 'B'}{int(strike)}{right[0].upper()}E"


def option_spec(strike, right, expiry):
    return InstrumentSpec.option(
        symbol(strike, right, expiry), strike, right, expiry, LOT,
        underlying=U, span_pct=0.1, exposure_pct=0.02,
    )


def specs(expiries=(EXP1, EXP2)):
    out = []
    for e in expiries:
        for k in STRIKES:
            for r in ("call", "put"):
                out.append(option_spec(k, r, e))
    return out


def spot_path(n=N_BARS, drift=0.0):
    return 100.0 + drift * np.arange(n) + np.sin(np.arange(n) / 4.0)


def bars(closes):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": T0 + STEP * np.arange(1, n + 1, dtype=np.int64),
        "open": closes.copy(),
        "high": closes + 0.05,
        "low": closes - 0.05,
        "close": closes,
        "volume": np.full(n, 1000.0),
    }


def premium(spot, strike, right, expiry, t):
    """Intrinsic plus a time value that decays toward expiry."""
    intrinsic = np.maximum(spot - strike, 0.0) if right == "call" else np.maximum(strike - spot, 0.0)
    left = max(expiry - t, 0) / (EXP2 - T0)
    return intrinsic + 8.0 * left * np.exp(-((spot - strike) / 12.0) ** 2) + 0.5


def universe(drift=0.0, expiries=(EXP1, EXP2)):
    spot = spot_path(drift=drift)
    ts = T0 + STEP * np.arange(1, N_BARS + 1, dtype=np.int64)
    data = {U: bars(spot)}
    for e in expiries:
        for k in STRIKES:
            for r in ("call", "put"):
                data[symbol(k, r, e)] = bars([premium(s, k, r, e, t) for s, t in zip(spot, ts)])
    inst = {sp.symbol: sp for sp in specs(expiries)}
    return data, inst
