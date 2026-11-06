"""Bars in, checked: the arrays a session evaluates over, and a digest that names them.

Nothing here imports pandas. A DataFrame works because it answers ``frame["close"]``; so does a
dict, a structured array or an ``npz`` file.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from raptorbt import errors

FIELDS = ("open", "high", "low", "close", "volume")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_DIGEST_TAG = b"raptorbt-data-v1"


@dataclass
class Dataset:
    """Float64 columns and int64 timestamps, with what the timestamps mean."""

    name: str
    arrays: dict[str, np.ndarray]
    bar_stamp: str
    bar_ns: int | None
    sha256: str
    ignored: tuple[str, ...] = ()
    path: Any = field(default=None, repr=False)  # set by `disk` when the arrays live in a file
    #: "bars" (1-D columns of one instrument) or "panel" (2-D columns, dates by names).
    kind: str = "bars"
    names: tuple[str, ...] = ()
    #: The trading calendar `at(...)` windows are cut by by default (a preset name or
    #: ``custom:<digest>``); none: clock hours and days of the zone.
    session: str | None = None

    @property
    def rows(self) -> int:
        """Bars, or dates of a panel."""
        return len(self.arrays["timestamps"])

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(f for f in FIELDS if f in self.arrays)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())

    def facts(self) -> dict[str, Any]:
        ts = self.arrays["timestamps"]
        panel = {"kind": "panel", "instruments": len(self.names)} if self.kind == "panel" else {}
        session = {"session": self.session} if self.session is not None else {}
        return {
            "data": self.name,
            **panel,
            "rows": self.rows,
            "fields": list(self.fields),
            "ignored": list(self.ignored),
            "bar_stamp": self.bar_stamp,
            "bar_ns": self.bar_ns,
            "first_timestamp": int(ts[0]),
            "last_timestamp": int(ts[-1]),
            "sha256": self.sha256,
            **session,
        }


def _refuse(code: str, message: str, parameter: str, **kw: Any) -> errors.RaptorBTError:
    return errors.make(code, message, operation="Session.load", parameter=parameter, **kw)


def digest(
    arrays: Mapping[str, np.ndarray], bar_stamp: str, bar_ns: int | None, names: Sequence[str] = ()
) -> str:
    """Hex sha256 of the columns, their order-independent names, the stamp, the bar width and
    (for a panel) the instrument names in column order."""
    h = hashlib.sha256(_DIGEST_TAG)
    h.update(f"|{bar_stamp}|{bar_ns}|".encode())
    if names:
        h.update(("panel:" + "\x1f".join(names) + "|").encode())
    for key in sorted(arrays):
        a = arrays[key]
        h.update(f"{key}:{a.dtype.str}:{len(a)}{f'x{a.shape[1]}' if a.ndim > 1 else ''};".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _column(bars: Any, key: str) -> Any:
    try:
        return bars[key]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _floats(x: Any, key: str) -> np.ndarray:
    try:
        a = np.array(x, dtype="<f8")  # a copy: the session never aliases the caller's array
    except (TypeError, ValueError):
        raise _refuse(
            "invalid_array", f"'{key}' must be numbers, convert with np.asarray(x, dtype=np.float64)", key
        ) from None
    if a.ndim != 1:
        raise _refuse("invalid_array", f"'{key}' must be a 1-D array, got {a.ndim}-D", key)
    return a


def _stamps(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind == "M":
        a = a.astype("datetime64[ns]").astype("<i8")
    elif a.dtype.kind in "iu":
        a = a.astype("<i8")
    else:
        raise _refuse(
            "invalid_array",
            f"'timestamps' must be int64 nanoseconds (or datetime64), got dtype {a.dtype}",
            "timestamps",
        )
    if a.ndim != 1:
        raise _refuse("invalid_array", f"'timestamps' must be a 1-D array, got {a.ndim}-D", "timestamps")
    return np.array(a, dtype="<i8")


def _check_stamp(bar_stamp: str | None, bar_ns: int | None) -> int | None:
    """The bar stamp is required and is open or close; the width is a positive whole number."""
    if bar_stamp is None:
        raise _refuse(
            "missing_bar_stamp",
            "bars were given without bar_stamp: say whether a timestamp starts ('open') or ends "
            "('close') its bar; as-of alignment depends on it",
            "bar_stamp",
        )
    if bar_stamp not in ("open", "close"):
        raise _refuse("invalid_bar_stamp", f"bar_stamp must be 'open' or 'close', got {bar_stamp!r}", "bar_stamp", value=bar_stamp)
    if bar_ns is not None and (isinstance(bar_ns, bool) or not isinstance(bar_ns, (int, np.integer)) or bar_ns <= 0):
        raise _refuse("invalid_bar_ns", f"bar_ns must be a positive whole number of nanoseconds, got {bar_ns!r}", "bar_ns", value=bar_ns)
    return None if bar_ns is None else int(bar_ns)


def _label(name: str | None, sha: str) -> str:
    label = name if name is not None else f"d_{sha[:8]}"
    if not isinstance(label, str) or not NAME.fullmatch(label):
        raise _refuse(
            "invalid_parameter",
            f"a data name is letters, digits, '_' and '-' (at most 64), got {label!r}",
            "name",
            value=label,
            valid=["letters, digits, '_' and '-', at most 64, starting with a letter or digit"],
        )
    return label


def build(bars: Any, bar_stamp: str | None, bar_ns: int | None, name: str | None) -> Dataset:
    """Check ``bars`` and copy it into a :class:`Dataset`; ``bars`` is never modified."""
    bar_ns = _check_stamp(bar_stamp, bar_ns)
    raw_ts = _column(bars, "timestamps")
    if raw_ts is None:
        raise _refuse(
            "missing_field", "bars have no 'timestamps' column (int64 nanoseconds)", "timestamps",
            valid=["timestamps: int64 nanoseconds, one per bar"],
        )
    arrays: dict[str, np.ndarray] = {"timestamps": _stamps(raw_ts)}
    for f in FIELDS:
        col = _column(bars, f)
        if col is not None:
            arrays[f] = _floats(col, f)
    if len(arrays) == 1:
        raise _refuse("missing_field", f"bars have none of the columns {', '.join(FIELDS)}", "close", valid=list(FIELDS))
    n = len(arrays["timestamps"])
    if n == 0:
        raise _refuse("invalid_array", "bars are empty", "timestamps")
    for k, a in arrays.items():
        if len(a) != n:
            raise _refuse("length_mismatch", f"'{k}' has {len(a)} values, 'timestamps' has {n}", k)
    if n > 1 and (np.diff(arrays["timestamps"]) < 0).any():
        raise _refuse("decreasing_timestamps", "timestamps must not decrease", "timestamps")
    for a in arrays.values():
        a.setflags(write=False)
    ignored = ()
    keys = getattr(bars, "keys", None)
    if callable(keys):
        try:
            ignored = tuple(sorted(str(k) for k in keys() if str(k) not in arrays))
        except TypeError:
            ignored = ()
    sha = digest(arrays, bar_stamp, bar_ns)
    label = _label(name, sha)
    return Dataset(label, arrays, bar_stamp, bar_ns, sha, ignored)


# ------------------------------------------------------------------ panels

MAX_NAME = 64


def _check_name(label: str) -> str:
    if not isinstance(label, str) or not 1 <= len(label) <= MAX_NAME:
        raise _refuse(
            "invalid_parameter", f"an instrument name is text of 1 to {MAX_NAME} characters, got {label!r}",
            "instruments", value=label if isinstance(label, (str, int)) else repr(label),
            valid=[f"text of 1 to {MAX_NAME} characters"],
        )
    return label


def build_panel(
    bars: Any,
    bar_stamp: str | None,
    bar_ns: int | None,
    instruments: Sequence[str] | None,
    name: str | None,
) -> Dataset:
    """Check a panel and copy it into a :class:`Dataset` of ``kind="panel"``.

    ``bars`` is what ``Expr.eval_panel`` takes: a dict of 2-D float64 arrays (dates by names)
    plus 1-D ``timestamps`` (and optionally a bool ``present``), or a mapping of instrument to
    its own columns (each with its ``timestamps``), aligned here on the union of timestamps
    with a name absent where it has no bar. ``instruments`` selects and orders (or, for 2-D
    arrays, names) the columns; 2-D arrays without names are called S0, S1, ...
    """
    bar_ns = _check_stamp(bar_stamp, bar_ns)
    if not isinstance(bars, Mapping) or not len(bars):
        raise _refuse("invalid_array", "a panel is a dict of 2-D arrays or a mapping of instrument to columns", "bars")
    nested = all(isinstance(v, Mapping) for v in bars.values())
    if nested:
        names = list(instruments) if instruments is not None else [str(k) for k in bars]
        missing = [n for n in names if n not in bars]
        if missing:
            raise errors.UnknownSymbol(
                f"no columns for instrument {missing[0]!r}", code="unknown_symbol", operation="Session.load_panel",
                parameter="instruments", value=missing[0], valid=[str(k) for k in bars][:20],
            )
        stamps = []
        for n in names:
            raw = _column(bars[n], "timestamps")
            ts = _stamps(raw if raw is not None else _missing_ts(n))
            if len(ts) > 1 and not (np.diff(ts) > 0).all():
                raise _refuse("decreasing_timestamps", f"instrument {n!r}: timestamps must be strictly increasing", "timestamps")
            stamps.append(ts)
        axis = np.unique(np.concatenate(stamps))
        fields = [f for f in FIELDS if all(_column(bars[n], f) is not None for n in names)]
        if not fields:
            raise _refuse("missing_field", f"the instruments share none of the columns {', '.join(FIELDS)}", "close", valid=list(FIELDS))
        arrays: dict[str, np.ndarray] = {f: np.full((len(axis), len(names)), np.nan, dtype="<f8") for f in fields}
        present = np.zeros((len(axis), len(names)), dtype=bool)
        for j, (n, ts) in enumerate(zip(names, stamps)):
            rows = np.searchsorted(axis, ts)
            present[rows, j] = True
            for f in fields:
                col = _floats(bars[n][f], f)
                if col.shape != ts.shape:
                    raise _refuse("length_mismatch", f"instrument {n!r}: '{f}' has {len(col)} values, 'timestamps' has {len(ts)}", f)
                arrays[f][rows, j] = col
        arrays = {"timestamps": axis, **arrays, "present": present}
    else:
        raw_ts = _column(bars, "timestamps")
        if raw_ts is None:
            _missing_ts(None)
        ts = _stamps(raw_ts)
        arrays = {"timestamps": ts}
        for f in FIELDS:
            col = _column(bars, f)
            if col is not None:
                a = np.array(col, dtype="<f8") if not isinstance(col, str) else None
                if a is None or a.ndim != 2:
                    raise _refuse("invalid_array", f"'{f}' must be a 2-D float64 array (dates by names)", f)
                arrays[f] = a
        if len(arrays) == 1:
            raise _refuse("missing_field", f"the panel has none of the columns {', '.join(FIELDS)}", "close", valid=list(FIELDS))
        shape = next(a.shape for k, a in arrays.items() if k != "timestamps")
        for k, a in arrays.items():
            if k != "timestamps" and a.shape != shape:
                raise _refuse("length_mismatch", f"'{k}' has shape {a.shape}, expected {shape} as the other columns", k)
        if shape[0] != len(ts):
            raise _refuse("length_mismatch", f"'timestamps' has {len(ts)} dates, the columns have {shape[0]}", "timestamps")
        if len(ts) > 1 and (np.diff(ts) < 0).any():
            raise _refuse("decreasing_timestamps", "timestamps must not decrease", "timestamps")
        pres = _column(bars, "present")
        if pres is not None:
            pres = np.array(pres, dtype=bool)
            if pres.shape != shape:
                raise _refuse("length_mismatch", f"'present' has shape {pres.shape}, expected {shape}", "present")
            arrays["present"] = pres
        names = list(instruments) if instruments is not None else [f"S{j}" for j in range(shape[1])]
        if len(names) != shape[1]:
            raise _refuse("length_mismatch", f"{len(names)} instruments for {shape[1]} columns", "instruments")
    if len(arrays["timestamps"]) == 0 or not names:
        raise _refuse("invalid_array", "a panel needs at least one date and one instrument", "bars")
    names = [_check_name(str(n)) for n in names]
    if len(set(names)) != len(names):
        raise _refuse("invalid_parameter", "instrument names must be distinct", "instruments")
    for a in arrays.values():
        a.setflags(write=False)
    sha = digest(arrays, bar_stamp, bar_ns, names)
    return Dataset(_label(name, sha), arrays, bar_stamp, bar_ns, sha, (), None, "panel", tuple(names))


def _missing_ts(n: str | None) -> Any:
    who = "" if n is None else f"instrument {n!r} needs "
    raise _refuse(
        "missing_field", f"{who}'timestamps': int64 nanoseconds, one per bar (per date for 2-D arrays)", "timestamps",
        valid=["timestamps: int64 nanoseconds"],
    )
