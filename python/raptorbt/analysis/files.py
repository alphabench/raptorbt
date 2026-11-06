"""Bars from a file, with the standard library and numpy only: ``.csv``, ``.npz`` and ``.npy``."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from raptorbt import errors
from raptorbt.analysis.data import FIELDS

_TIME_COLUMNS = ("timestamps", "timestamp", "time", "ts", "datetime", "date")
_UNIT_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
SUFFIXES = (".csv", ".npz", ".npy")


def _refuse(message: str, parameter: str, **kw: Any) -> errors.RaptorBTError:
    return errors.make("invalid_input", message, operation="read_bars", parameter=parameter, **kw)


def read_bars(
    path: str | Path,
    *,
    time_unit: str = "ns",
    field: str = "close",
    timestamps: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Columns from ``path``.

    - ``.csv``: a header row; columns named ``open``, ``high``, ``low``, ``close``, ``volume`` and
      one of ``timestamps`` / ``timestamp`` / ``time`` / ``ts`` / ``datetime`` / ``date``. A
      timestamp is an integer in ``time_unit`` since the epoch, or ISO-8601 text *with* a UTC
      offset (a time with no offset is refused, never guessed). An empty cell is NaN.
    - ``.npz``: arrays named as the columns (``timestamps`` int64 nanoseconds).
    - ``.npy``: a structured array with those names, or a 1-D float array that is the column
      ``field``, with ``timestamps`` the path of an int64 ``.npy`` of nanoseconds.
    """
    p = Path(path)
    if time_unit not in _UNIT_NS:
        raise _refuse(f"time_unit must be one of {', '.join(_UNIT_NS)}, got {time_unit!r}", "time_unit", value=time_unit, valid=list(_UNIT_NS))
    if not p.is_file():
        raise _refuse(f"no such file: {p}", "data", value=str(p), valid=["a path to a .csv, .npz or .npy file"])
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _csv(p, _UNIT_NS[time_unit])
    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if suffix == ".npy":
        a = np.load(p, allow_pickle=False)
        if a.dtype.names:
            return {k: a[k] for k in a.dtype.names}
        if timestamps is None:
            raise _refuse(
                f"{p.name} is a plain array: give timestamps= (an int64 .npy of nanoseconds), and "
                f"field= for the column it holds (default close)", "timestamps",
                valid=["--timestamps FILE.npy"],
            )
        return {field: a, "timestamps": np.load(timestamps, allow_pickle=False)}
    raise _refuse(f"cannot read {p.name}: expected {', '.join(SUFFIXES)}", "data", value=str(p), valid=list(SUFFIXES))


def _stamp(cell: str, unit: int, row: int) -> int:
    text = cell.strip()
    try:
        return int(text) * unit
    except ValueError:
        pass
    try:
        t = dt.datetime.fromisoformat(text)
    except ValueError:
        raise _refuse(f"row {row}: {cell!r} is not an integer timestamp or ISO-8601 text", "timestamps", value=cell) from None
    if t.tzinfo is None:
        raise _refuse(
            f"row {row}: {cell!r} has no UTC offset; write it as 2026-01-05T09:16:00+05:30 or use "
            f"integer epoch timestamps", "timestamps", value=cell,
        )
    delta = t - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _csv(p: Path, unit: int) -> dict[str, np.ndarray]:
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = csv.reader(fh)
        header = next(rows, None)
        if not header:
            raise _refuse(f"{p.name} is empty", "data", value=str(p))
        names = [h.strip().lower() for h in header]
        tcol = next((names.index(c) for c in _TIME_COLUMNS if c in names), None)
        if tcol is None:
            raise _refuse(f"{p.name} has no time column ({', '.join(_TIME_COLUMNS)})", "timestamps", valid=list(_TIME_COLUMNS))
        cols = {f: names.index(f) for f in FIELDS if f in names}
        ts: list[int] = []
        out: dict[str, list[float]] = {f: [] for f in cols}
        for i, row in enumerate(rows, start=2):
            if not row:
                continue
            if len(row) != len(names):
                raise _refuse(f"row {i} of {p.name} has {len(row)} cells, the header has {len(names)}", "data", value=str(p))
            ts.append(_stamp(row[tcol], unit, i))
            for f, c in cols.items():
                cell = row[c].strip()
                try:
                    out[f].append(float(cell) if cell else float("nan"))
                except ValueError:
                    raise _refuse(f"row {i}, column {f}: {cell!r} is not a number", f, value=cell) from None
    return {"timestamps": np.array(ts, dtype=np.int64), **{f: np.array(v, dtype=np.float64) for f, v in out.items()}}


_NAME_COLUMNS = ("instrument", "symbol", "ticker", "name")


def read_panel(path: str | Path, *, time_unit: str = "ns") -> tuple[dict[str, Any], list[str] | None]:
    """A panel from ``path`` and the instrument names it carries (or ``None``).

    - ``.csv``: one row per instrument and bar, with a time column as for :func:`read_bars`, an
      ``instrument`` (or ``symbol`` / ``ticker`` / ``name``) column and the price columns.
      Returns a mapping of instrument, in order of first appearance, to its own columns.
    - ``.npz``: 2-D arrays named as the columns (dates by names), 1-D ``timestamps`` (int64
      nanoseconds) and optionally ``instruments`` (text) and ``present`` (bool).
    """
    p = Path(path)
    if time_unit not in _UNIT_NS:
        raise _refuse(f"time_unit must be one of {', '.join(_UNIT_NS)}, got {time_unit!r}", "time_unit", value=time_unit, valid=list(_UNIT_NS))
    if not p.is_file():
        raise _refuse(f"no such file: {p}", "panel", value=str(p), valid=["a path to a .csv or .npz file"])
    suffix = p.suffix.lower()
    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        names = arrays.pop("instruments", None)
        return arrays, None if names is None else [str(n) for n in names]
    if suffix != ".csv":
        raise _refuse(f"cannot read {p.name} as a panel: expected .csv or .npz", "panel", value=str(p), valid=[".csv", ".npz"])
    unit = _UNIT_NS[time_unit]
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = csv.reader(fh)
        header = next(rows, None)
        if not header:
            raise _refuse(f"{p.name} is empty", "panel", value=str(p))
        names = [h.strip().lower() for h in header]
        tcol = next((names.index(c) for c in _TIME_COLUMNS if c in names), None)
        icol = next((names.index(c) for c in _NAME_COLUMNS if c in names), None)
        if tcol is None or icol is None:
            raise _refuse(
                f"{p.name} needs a time column ({', '.join(_TIME_COLUMNS)}) and an instrument column "
                f"({', '.join(_NAME_COLUMNS)})", "panel", valid=[*_TIME_COLUMNS, *_NAME_COLUMNS],
            )
        cols = {f: names.index(f) for f in FIELDS if f in names}
        by: dict[str, dict[str, list[Any]]] = {}
        for i, row in enumerate(rows, start=2):
            if not row:
                continue
            if len(row) != len(names):
                raise _refuse(f"row {i} of {p.name} has {len(row)} cells, the header has {len(names)}", "panel", value=str(p))
            cell = by.setdefault(row[icol].strip(), {"timestamps": [], **{f: [] for f in cols}})
            cell["timestamps"].append(_stamp(row[tcol], unit, i))
            for f, c in cols.items():
                v = row[c].strip()
                try:
                    cell[f].append(float(v) if v else float("nan"))
                except ValueError:
                    raise _refuse(f"row {i}, column {f}: {v!r} is not a number", f, value=v) from None
    out = {
        n: {"timestamps": np.array(c["timestamps"], dtype=np.int64), **{f: np.array(c[f], dtype=np.float64) for f in cols}}
        for n, c in by.items()
    }
    return out, None
