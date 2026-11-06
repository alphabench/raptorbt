"""Measure today's raptorbt array and streaming indicators against the TA-Lib fixtures.

Prints, per function and form: the first valid index on both sides, the largest absolute
difference after both are valid, and what each does on a flat input. It changes nothing and
asserts nothing; the numbers feed the `defined_difference` pins and the `compat` decision.

    python tests/oracle/measure_legacy.py            (needs raptorbt built, numpy)
    uv run --with ta-lib==0.6.4 ... measure_legacy.py   also prints TA-Lib on flat input

Not reachable from Python today, so reported as such: array WMA/ROC/STDDEV and OBV, MFI, AD
(the Rust functions exist; only the streaming Indicator has WMA, ROC and STDDEV).
"""

import gzip
import json
import os

import numpy as np

import raptorbt as rb

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "oracle", "fixtures")
)
FLAT_N = 200


def load(name):
    with gzip.open(os.path.join(ROOT, name), "rt") as f:
        return json.load(f)


def dec(entry):
    if isinstance(entry, str):
        return np.frombuffer(bytes.fromhex(entry), dtype=">f8").astype(float)
    dt = ">f8" if entry["dtype"] == "f8" else ">i4"
    return np.frombuffer(bytes.fromhex(entry["data"]), dtype=dt).astype(float)


def first_valid(a):
    i = np.flatnonzero(~np.isnan(a))
    return int(i[0]) if len(i) else -1


def stream_run(factory, o, h, l, c):
    ind = factory()
    rows = []
    for i in range(len(c)):
        v = ind.update_bar(float(o[i]), float(h[i]), float(l[i]), float(c[i]))
        rows.append(v)
    width = 1
    for v in rows:
        if v is not None and not np.isscalar(v):
            width = len(v)
            break
    out = np.full((width, len(c)), np.nan)
    for i, v in enumerate(rows):
        if v is None:
            continue
        if width == 1:
            out[0, i] = v
        else:
            out[:, i] = list(v)
    return out


def arr(x):
    return (
        [np.asarray(t, dtype=float) for t in x]
        if isinstance(x, tuple)
        else [np.asarray(x, dtype=float)]
    )


# name -> (talib outputs, array fn or None, stream factory or None, param mapper)
# params: TA-Lib parameter dict -> kwargs for the helper lambdas
SPECS = {
    "SMA": (
        ["real"],
        lambda d, p: rb.sma(d["close"], p["timeperiod"]),
        lambda p: rb.Indicator.sma(p["timeperiod"]),
    ),
    "EMA": (
        ["real"],
        lambda d, p: rb.ema(d["close"], p["timeperiod"]),
        lambda p: rb.Indicator.ema(p["timeperiod"]),
    ),
    "RSI": (
        ["real"],
        lambda d, p: rb.rsi(d["close"], p["timeperiod"]),
        lambda p: rb.Indicator.rsi(p["timeperiod"]),
    ),
    "MACD": (
        ["macd", "macdsignal", "macdhist"],
        lambda d, p: rb.macd(d["close"], p["fastperiod"], p["slowperiod"], p["signalperiod"]),
        lambda p: rb.Indicator.macd(p["fastperiod"], p["slowperiod"], p["signalperiod"]),
    ),
    "ATR": (
        ["real"],
        lambda d, p: rb.atr(d["high"], d["low"], d["close"], p["timeperiod"]),
        lambda p: rb.Indicator.atr(p["timeperiod"]),
    ),
    "BBANDS": (
        ["upperband", "middleband", "lowerband"],
        lambda d, p: tuple(rb.bollinger_bands(d["close"], p["timeperiod"], p["nbdevup"])),
        lambda p: rb.Indicator.bollinger(p["timeperiod"], p["nbdevup"]),
    ),
    "STOCH": (
        ["slowk", "slowd"],
        lambda d, p: rb.stochastic(
            d["high"], d["low"], d["close"], p["fastk_period"], p["slowd_period"]
        ),
        None,
    ),
    "ADX": (["real"], lambda d, p: rb.adx(d["high"], d["low"], d["close"], p["timeperiod"]), None),
    "OBV": (["real"], None, None),
    "MFI": (["real"], None, None),
    "AD": (["real"], None, None),
    "WMA": (["real"], None, lambda p: rb.Indicator.wma(p["timeperiod"])),
    "ROC": (["real"], None, lambda p: rb.Indicator.roc(p["timeperiod"])),
    "STDDEV": (["real"], None, lambda p: rb.Indicator.stddev(p["timeperiod"])),
}
GROUPS = {
    "SMA": "overlap_studies",
    "EMA": "overlap_studies",
    "BBANDS": "overlap_studies",
    "WMA": "overlap_studies",
    "RSI": "momentum_indicators",
    "MACD": "momentum_indicators",
    "STOCH": "momentum_indicators",
    "ADX": "momentum_indicators",
    "MFI": "momentum_indicators",
    "ROC": "momentum_indicators",
    "ATR": "volatility_indicators",
    "OBV": "volume_indicators",
    "AD": "volume_indicators",
    "STDDEV": "statistic_functions",
}


def flat_summary(label, out):
    cols = out if isinstance(out, list) else [out]
    parts = []
    for c in cols:
        c = np.asarray(c, dtype=float)
        tail = c[-1]
        parts.append(f"first_valid={first_valid(c)} last={tail!r} inf={bool(np.isinf(c).any())}")
    return f"    flat {label}: " + " | ".join(parts)


def main():
    inputs = load("inputs.json.gz")["series"]
    try:
        import talib
    except ImportError:
        talib = None
    flat = {k: np.full(FLAT_N, 100.0) for k in ("open", "high", "low", "close")}
    flat["volume"] = np.full(FLAT_N, 1000.0)

    for name, (outs, arr_fn, stream_f) in SPECS.items():
        print(f"== {name}")
        if arr_fn is None:
            print("  array form: not reachable from Python")
        if stream_f is None:
            print("  streaming form: no Indicator for it")
        if arr_fn is None and stream_f is None:
            continue
        group = load(f"talib_{GROUPS[name]}.json.gz")["functions"][name]
        for case in group["cases"]:
            p = case["params"]
            if name == "BBANDS" and (p["nbdevup"] != p["nbdevdn"] or p["matype"] != 0):
                continue
            d = {k: dec(v) for k, v in inputs[case["scenario"]].items()}
            ref = [dec(case["out"][o]) for o in outs]
            ref = [np.where(np.arange(len(r)) < case["lookback"], np.nan, r) for r in ref]
            for form, fn in (("array", arr_fn), ("stream", stream_f)):
                if fn is None:
                    continue
                if form == "array":
                    ours = arr(fn(d, p))
                else:
                    ours = list(
                        stream_run(lambda: fn(p), d["open"], d["high"], d["low"], d["close"])
                    )
                for oname, o, r in zip(outs, ours, ref):
                    both = ~np.isnan(o) & ~np.isnan(r)
                    diff = float(np.max(np.abs(o[both] - r[both]))) if both.any() else float("nan")
                    print(
                        f"  {form:6} {case['scenario']:20} {case['label']:12} {oname:10} "
                        f"first_valid ours={first_valid(o):4d} talib={first_valid(r):4d}  max|diff|={diff:.3e}"
                    )
        # Flat input, default parameters.
        dp = group["cases"][0]["params"]
        if arr_fn is not None:
            try:
                print(flat_summary("array ", arr(arr_fn(flat, dp))))
            except Exception as e:
                print(f"    flat array : raised {type(e).__name__}: {e}")
        if stream_f is not None:
            try:
                print(
                    flat_summary(
                        "stream",
                        list(
                            stream_run(
                                lambda: stream_f(dp),
                                flat["open"],
                                flat["high"],
                                flat["low"],
                                flat["close"],
                            )
                        ),
                    )
                )
            except Exception as e:
                print(f"    flat stream: raised {type(e).__name__}: {e}")
        if talib is not None:
            f = getattr(talib, name)
            args = {
                "SMA": ["close"],
                "EMA": ["close"],
                "RSI": ["close"],
                "MACD": ["close"],
                "BBANDS": ["close"],
                "WMA": ["close"],
                "ROC": ["close"],
                "STDDEV": ["close"],
                "ATR": ["high", "low", "close"],
                "STOCH": ["high", "low", "close"],
                "ADX": ["high", "low", "close"],
                "OBV": ["close", "volume"],
                "MFI": ["high", "low", "close", "volume"],
                "AD": ["high", "low", "close", "volume"],
            }[name]
            res = f(*[flat[a] for a in args], **dp)
            print(flat_summary("TA-Lib", list(res) if isinstance(res, tuple) else [res]))
        else:
            print("    flat TA-Lib: (talib not importable here)")


if __name__ == "__main__":
    main()
