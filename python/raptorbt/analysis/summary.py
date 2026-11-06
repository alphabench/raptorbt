"""The summary of one result series: small, true and the same shape every time.

A summary states facts about the numbers and never advice. Its size is capped (``OUTPUT_BYTES``
per output, ``CALL_BYTES`` per call) by dropping optional detail in a fixed order, never by
rounding the facts away. The stored series keeps full precision; only what is shown is rounded
(to ``SIGNIFICANT`` digits).

Each flag below has one definition, in one function, with a test.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

OUTPUT_BYTES = 2048
CALL_BYTES = 4096
SIGNIFICANT = 6
LAST = 5
#: Fewer finite values than this and a series is flagged ``short_history``.
SHORT_HISTORY_BARS = 30
#: A value is an outlier at this many robust standard deviations from the median (the audit
#: in ``raptorbt.stats.audit_bars`` uses the same cut and scale).
OUTLIER_Z = 8.0
MAD_SCALE = 1.4826
#: Entries of ``reference`` and of the ``non_causal`` list shown before "more" counts the rest.
LISTED = 8


def show(x: float) -> float | int | None:
    """A number as the summary prints it: ``SIGNIFICANT`` digits; NaN and infinity as ``None``
    (the ``inf`` flag counts infinities)."""
    v = float(x)
    if not math.isfinite(v):
        return None
    return float(f"{v:.{SIGNIFICANT - 1}e}")


def sha256_of(a: np.ndarray) -> str:
    """Hex sha256 of the series (or panel, with its shape) as little-endian float64 with one NaN
    pattern."""
    x = np.where(np.isnan(a), np.float64("nan"), a).astype("<f8", copy=False)
    h = hashlib.sha256()
    if x.ndim > 1:
        h.update(repr(x.shape).encode())
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ flags: one definition each


def count_inf(*series: np.ndarray) -> int:
    """``inf``: the number of values equal to +inf or -inf, in the output and the inputs read."""
    return int(sum(np.isinf(s).sum() for s in series))


def count_nonpositive(prices: Sequence[np.ndarray]) -> int:
    """``nonpositive_input``: the bars on which a price read (open, high, low or close) is zero
    or negative. NaN is not; volume is never a price."""
    if not prices:
        return 0
    bad = np.zeros(prices[0].shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for p in prices:
            bad |= p <= 0
    return int(bad.sum())


def count_gaps(out: np.ndarray) -> int:
    """``gaps``: the runs of NaN that start after the first finite value."""
    finite = np.isfinite(out)
    if not finite.any():
        return 0
    rest = ~finite[int(np.argmax(finite)) :]
    starts = rest[1:] & ~rest[:-1]
    return int(starts.sum()) + (1 if rest[0] else 0)


def count_outliers(out: np.ndarray) -> int:
    """``outliers``: finite values more than ``OUTLIER_Z`` robust standard deviations from the
    median (``1.4826 * MAD``). A series whose MAD is zero has no scale and reports none."""
    v = out[np.isfinite(out)]
    if v.size < 3:
        return 0
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return 0
    return int((np.abs(v - med) > OUTLIER_Z * MAD_SCALE * mad).sum())


def count_flat_windows(inputs: Sequence[np.ndarray], window: int, out: np.ndarray) -> int:
    """``flat_windows``: the bars with a known output whose trailing ``window`` bars hold one
    value in every input read (all fields and handles). On a name that has not traded this is
    what an RSI of 100 sits on."""
    n = len(out)
    if window < 2 or not inputs or n < window:
        return 0
    moved = np.zeros(n - 1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        for x in inputs:
            moved |= (np.diff(x) != 0) | np.isnan(x[1:]) | np.isnan(x[:-1])
    cum = np.concatenate([[0], np.cumsum(moved)])
    # bar i (i >= window - 1) spans the window - 1 steps ending at step i - 1
    i = np.arange(window - 1, n)
    flat = (cum[i] - cum[i - window + 1]) == 0
    return int((flat & np.isfinite(out[i])).sum())


def is_short(finite_count: int) -> bool:
    """``short_history``: fewer than ``SHORT_HISTORY_BARS`` known values."""
    return finite_count < SHORT_HISTORY_BARS


def non_causal_names(functions: Sequence[str], causal: dict[str, bool]) -> list[str]:
    """``non_causal``: the functions called whose registry row says they use later bars."""
    return sorted({f for f in functions if causal.get(f) is False})


# ------------------------------------------------------------------ the summary


def leading_nan(a: np.ndarray) -> int:
    finite = np.isfinite(a)
    return int(np.argmax(finite)) if finite.any() else len(a)


def summarise(
    handle: str,
    series: np.ndarray,
    kind: str,
    *,
    warmup: int | None,
    inputs: Sequence[np.ndarray],
    prices: Sequence[np.ndarray],
    causal: bool,
    non_causal: Sequence[str],
    reference: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The summary of one output. ``inputs`` are the arrays the expression read (fields and
    handles), ``prices`` the price fields among them."""
    n = len(series)
    finite = np.isfinite(series)
    known = int(finite.sum())
    lead = leading_nan(series)
    nan_total = int(np.isnan(series).sum())
    out: dict[str, Any] = {
        "handle": handle,
        "kind": kind,
        "shape": [n],
        "warmup": warmup,
        "nan": {"leading": lead, "interior": nan_total - lead if known else 0},
    }
    if extra:
        out.update(extra)
    if kind == "bool":
        true = int((series[finite] == 1.0).sum())
        out["known"] = known
        out["true"] = true
        out["true_rate"] = show(true / known) if known else None
    else:
        v = series[finite]
        if v.size:
            p05, p50, p95 = np.percentile(v, [5, 50, 95])
            top = float(np.abs(v).max())
            scale = top if top > 1e100 else 1.0  # a square of 1e200 overflows: measure in units of the largest
            out["dist"] = {
                "min": show(v.min()), "p05": show(p05), "p50": show(p50), "p95": show(p95),
                "max": show(v.max()), "mean": show(scale * (v / scale).mean()),
                "std": show(scale * (v / scale).std()),
            }  # fmt: skip
        else:
            out["dist"] = None
    out["last"] = [show(x) for x in series[-LAST:]]
    flags: dict[str, Any] = {}
    window = None if warmup is None else warmup + 1
    if kind == "number":
        if (c := count_outliers(series)):
            flags["outliers"] = c
    if window is not None and (c := count_flat_windows(inputs, window, series)):
        flags["flat_windows"] = c
    if (c := count_gaps(series)):
        flags["gaps"] = c
    if (c := count_nonpositive(prices)):
        flags["nonpositive_input"] = c
    if (c := count_inf(series, *inputs)):
        flags["inf"] = c
    if is_short(known):
        flags["short_history"] = known
    if non_causal:
        flags["non_causal"] = list(non_causal[:LISTED])
    out["flags"] = flags
    out["causal"] = causal
    out["reference"] = dict(list(reference.items())[:LISTED])
    if len(reference) > LISTED:
        out["reference_more"] = len(reference) - LISTED
    if len(non_causal) > LISTED:
        out["non_causal_more"] = len(non_causal) - LISTED
    out["sha256"] = sha256_of(series)
    return fit(out, OUTPUT_BYTES)


#: Names listed from each end of the last cross-section.
EDGE = 5


def leading_dates(a: np.ndarray) -> int:
    """The dates before the first one on which any name has a known value."""
    any_known = np.isfinite(a).any(axis=1)
    return int(np.argmax(any_known)) if any_known.any() else len(a)


def _dist(v: np.ndarray) -> dict[str, Any] | None:
    if not v.size:
        return None
    p05, p50, p95 = np.percentile(v, [5, 50, 95])
    top = float(np.abs(v).max())
    scale = top if top > 1e100 else 1.0  # a square of 1e200 overflows: measure in units of the largest
    return {
        "min": show(v.min()), "p05": show(p05), "p50": show(p50), "p95": show(p95),
        "max": show(v.max()), "mean": show(scale * (v / scale).mean()),
        "std": show(scale * (v / scale).std()),
    }  # fmt: skip


def summarise_panel(
    handle: str,
    panel: np.ndarray,
    kind: str,
    names: Sequence[str],
    timestamps: np.ndarray,
    *,
    warmup: int | None,
    inputs: Sequence[np.ndarray],
    prices: Sequence[np.ndarray],
    causal: bool,
    non_causal: Sequence[str],
    reference: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The summary of one panel output (dates by names). ``coverage`` is the number of names
    with a known value on a date, over the dates that have any; ``last`` is the cross-section of
    the last such date, with the names at both ends (for a boolean, how many are true and the
    first few)."""
    n, m = panel.shape
    finite = np.isfinite(panel)
    cov = finite.sum(axis=1)
    live = cov[cov > 0]
    known = int(finite.sum())
    out: dict[str, Any] = {
        "handle": handle,
        "kind": kind,
        "shape": [n, m],
        "warmup": warmup,
        "nan": {"leading": leading_dates(panel), "cells": int(np.isnan(panel).sum())},
    }
    if extra:
        out.update(extra)
    out["coverage"] = (
        {"min": int(live.min()), "median": show(np.median(live)), "max": int(live.max())} if live.size else None
    )
    if kind == "bool":
        true = int((panel[finite] == 1.0).sum())
        out["known"] = known
        out["true"] = true
        out["true_rate"] = show(true / known) if known else None
    else:
        out["dist"] = _dist(panel[finite])
    if live.size:
        li = int(np.flatnonzero(cov > 0)[-1])
        row = panel[li]
        idx = np.flatnonzero(np.isfinite(row))
        last: dict[str, Any] = {"index": li, "timestamp": int(timestamps[li]), "known": int(idx.size)}
        if kind == "bool":
            on = [names[j] for j in idx if row[j] == 1.0]
            last["true"] = len(on)
            last["names"] = on[:EDGE]
        else:
            vals = row[idx]
            order = np.argsort(vals, kind="stable")
            last["bottom"] = [[names[idx[k]], show(vals[k])] for k in order[:EDGE]]
            desc = np.argsort(-vals, kind="stable")
            last["top"] = [[names[idx[k]], show(vals[k])] for k in desc[:EDGE]]
        out["last"] = last
    else:
        out["last"] = None
    flags: dict[str, Any] = {}
    if kind == "number" and (c := count_outliers(panel.ravel())):
        flags["outliers"] = c
    if warmup is not None and warmup >= 1:
        flat = sum(count_flat_windows([x[:, j] for x in inputs], warmup + 1, panel[:, j]) for j in range(m))
        if flat:
            flags["flat_windows"] = flat
    if (c := sum(count_gaps(panel[:, j]) for j in range(m))):
        flags["gaps"] = c
    if (c := count_nonpositive(prices)):
        flags["nonpositive_input"] = c
    if (c := count_inf(panel, *inputs)):
        flags["inf"] = c
    if is_short(int((cov > 0).sum())):
        flags["short_history"] = int((cov > 0).sum())
    if non_causal:
        flags["non_causal"] = list(non_causal[:LISTED])
    out["flags"] = flags
    out["causal"] = causal
    out["reference"] = dict(list(reference.items())[:LISTED])
    if len(reference) > LISTED:
        out["reference_more"] = len(reference) - LISTED
    if len(non_causal) > LISTED:
        out["non_causal_more"] = len(non_causal) - LISTED
    out["sha256"] = sha256_of(panel)
    return fit(out, OUTPUT_BYTES)


def size(obj: Any) -> int:
    """Bytes of the compact JSON of ``obj``."""
    return len(json.dumps(obj, separators=(",", ":"), allow_nan=False).encode())


def fit(summary: dict[str, Any], limit: int) -> dict[str, Any]:
    """Drop optional detail, least informative first, until the summary fits ``limit`` bytes."""
    for step in (
        lambda s: s.pop("reference_more", None),
        lambda s: s.__setitem__("reference", dict(list(s["reference"].items())[:2])),
        lambda s: s.pop("reference"),
        lambda s: s.pop("non_causal_more", None),
        lambda s: s["flags"].__setitem__("non_causal", s["flags"].get("non_causal", [])[:1]) if "non_causal" in s["flags"] else None,
        lambda s: _shrink_last(s),
        lambda s: _shrink_last(s),
        lambda s: _shrink_last(s),
    ):
        if size(summary) <= limit:
            break
        step(summary)
    return summary


def _shrink_last(s: dict[str, Any]) -> None:
    """Fewer last values; for a panel, fewer names at each end, then none."""
    last = s["last"]
    if isinstance(last, dict):
        for k in ("top", "bottom", "names"):
            if k in last:
                last[k] = last[k][: max(1, len(last[k]) // 2)] if len(last[k]) > 1 else []
    elif isinstance(last, list):
        s["last"] = last[-2:]
