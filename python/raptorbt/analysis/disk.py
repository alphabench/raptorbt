"""A session on disk, so a shell agent can chain handles across invocations.

Layout of one session (``<cache>/sessions/<name>/``)::

    meta.json                      recipes, summaries, LRU order; written last, atomically
    data/<dataset>/<column>.npy    the loaded bars
    h/<handle>/<output>.npy        one result series (full precision)

Series are memory-mapped when read, so a command that reads one handle touches one file. Every
path is built from a name that was checked against a pattern (a session or dataset name:
letters, digits, ``_``, ``-``; a handle: ``h_`` and hex): a name with a separator or a dot
cannot leave the cache.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from raptorbt import errors
from raptorbt.analysis import data as _data
from raptorbt.analysis.session import _HANDLE_FULL, Session, _Handle

try:  # POSIX; without it a session has no lock and two commands at once may race
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

MAX_AGE_DAYS = 7
MAX_CACHE_BYTES = 1 << 30
_VERSION = 2
_OUTPUT = re.compile(r"[a-z0-9_]{1,64}")


def cache_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The directory holding ``sessions/``: ``explicit``, else ``$XDG_CACHE_HOME/raptorbt``,
    else ``~/.cache/raptorbt``."""
    if explicit is not None:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "raptorbt"


def session_dir(root: Path, name: str) -> Path:
    """``<root>/sessions/<name>``; refuses a name that is not one plain path component."""
    if not isinstance(name, str) or not _data.NAME.fullmatch(name):
        raise errors.InvalidParameter(
            f"a session name is letters, digits, '_' and '-' (at most 64), got {name!r}",
            code="invalid_parameter", operation="session", parameter="session", value=name,
            valid=["letters, digits, '_' and '-', at most 64, starting with a letter or digit"],
        )
    base = (root / "sessions").resolve()
    path = (base / name).resolve()
    if path.parent != base:
        raise errors.InvalidParameter(
            f"session {name!r} resolves outside the cache", code="invalid_parameter",
            operation="session", parameter="session", value=name, valid=["a plain name"],
        )
    return path


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold the session's lock: one command at a time per session."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / ".lock", "a+") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _write_npy(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(a), allow_pickle=False)
    os.replace(tmp, path)


def _read_npy(path: Path, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    try:
        a = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as e:
        raise errors.InvalidData(
            f"the session file {path.name} cannot be read ({e}); drop the session and load again",
            code="invalid_array", operation="session", parameter=path.name, valid=["a session written by raptorbt"],
        ) from None
    if a.shape != shape or a.dtype.str != dtype:
        raise errors.InvalidData(
            f"the session file {path.name} holds {a.dtype}{a.shape}, expected {dtype}{shape}",
            code="invalid_array", operation="session", parameter=path.name, valid=["a session written by raptorbt"],
        )
    return a


def save(session: Session, path: Path) -> None:
    """Write what changed, drop what is gone, then replace ``meta.json``."""
    for ds in session._data.values():
        if ds.path != path:
            for col, a in ds.arrays.items():
                _write_npy(path / "data" / ds.name / f"{col}.npy", a)
            ds.path = path
    for h in session._handles.values():
        if h.path != path:
            for out, a in h.get().items():
                if not _OUTPUT.fullmatch(out):
                    raise errors.InvalidData(
                        f"output name {out!r} is not a plain name", code="invalid_array",
                        operation="session", parameter="output", valid=["a-z, 0-9, _"],
                    )
                _write_npy(path / "h" / h.id / f"{out}.npy", a)
            h.path = path
    for sub, keep in (("data", set(session._data)), ("h", set(session._handles))):
        base = path / sub
        if base.is_dir():
            for child in base.iterdir():
                if child.name not in keep:
                    shutil.rmtree(child, ignore_errors=True)
    meta = {
        "version": _VERSION,
        "max_bytes": session.max_bytes,
        "max_handles": session.max_handles,
        "current": session._current,
        "data": {
            n: {
                "bar_stamp": d.bar_stamp, "bar_ns": d.bar_ns, "sha256": d.sha256, "rows": d.rows,
                "fields": list(d.fields), "ignored": list(d.ignored), "kind": d.kind,
                "names": list(d.names), "present": "present" in d.arrays, "session": d.session,
            }
            for n, d in session._data.items()
        },
        "handles": [  # least recently used first
            {
                "id": h.id, "expression": h.expression, "data": h.data, "data_sha": h.data_sha,
                "tz_offset_ns": h.tz_offset_ns, "session": h.session, "outputs": list(h.outputs),
                "kinds": list(h.kinds),
                "deps": list(h.deps), "functions": list(h.functions), "causal": h.causal,
                "result": h.result, "nbytes": h.nbytes, "leading": h.leading,
                "shapes": {o: list(sh) for o, sh in h.shapes.items()},
            }
            for h in session._handles.values()
        ],
        "gone": [[k, *v] for k, v in session._gone.items()],
    }
    tmp = path / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path / "meta.json")


def open_session(path: Path, max_bytes: int | None = None, max_handles: int | None = None) -> Session:
    """The session stored at ``path``, or a new empty one when there is none."""
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        return Session(**{k: v for k, v in (("max_bytes", max_bytes), ("max_handles", max_handles)) if v is not None})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("version") != _VERSION:
        raise errors.InvalidData(
            f"the session at {path.name} has format {meta.get('version')!r}, this engine reads {_VERSION}",
            code="invalid_array", operation="session", parameter="session", valid=["drop the session and load again"],
        )
    s = Session(max_bytes or meta["max_bytes"], max_handles or meta["max_handles"])
    for name, m in meta["data"].items():
        if not _data.NAME.fullmatch(name):
            raise errors.InvalidData(f"bad dataset name {name!r} in the session", code="invalid_array",
                                     operation="session", parameter="data", valid=["drop the session"])
        kind, names = m.get("kind", "bars"), tuple(m.get("names", ()))
        shape = (m["rows"], len(names)) if kind == "panel" else (m["rows"],)
        arrays = {"timestamps": _read_npy(path / "data" / name / "timestamps.npy", (m["rows"],), "<i8")}
        for f in m["fields"]:
            arrays[f] = _read_npy(path / "data" / name / f"{f}.npy", shape, "<f8")
        if m.get("present"):
            arrays["present"] = _read_npy(path / "data" / name / "present.npy", shape, "|b1")
        ds = _data.Dataset(
            name, arrays, m["bar_stamp"], m["bar_ns"], m["sha256"], tuple(m["ignored"]), path, kind, names,
            session=m.get("session"),
        )
        s._data[name] = ds
        s._used += ds.nbytes
    s._current = meta["current"]
    for m in meta["handles"]:
        hid = m["id"]
        if not _HANDLE_FULL.fullmatch(hid) or len(hid) != 12:
            raise errors.InvalidData(f"bad handle {hid!r} in the session", code="invalid_array",
                                     operation="session", parameter="handle", valid=["drop the session"])
        h = _Handle(
            id=hid, expression=m["expression"], data=m["data"], data_sha=m["data_sha"],
            tz_offset_ns=m["tz_offset_ns"], outputs=tuple(m["outputs"]), kinds=tuple(m["kinds"]),
            deps=tuple(m["deps"]), functions=tuple(m["functions"]), causal=m["causal"],
            result=m["result"], nbytes=m["nbytes"], leading=m["leading"],
            shapes={o: tuple(sh) for o, sh in m["shapes"].items()}, path=path,
            session=m.get("session"),
        )
        h.loader = _loader(path, h)
        s._handles[hid] = h
        s._used += h.nbytes
    s._gone = OrderedDict((k, (recipe, reason)) for k, recipe, reason in meta["gone"])
    s._evict("session")
    return s


def _loader(path: Path, h: _Handle):
    def load() -> dict[str, np.ndarray]:
        return {o: _read_npy(path / "h" / h.id / f"{o}.npy", h.shapes[o], "<f8") for o in h.outputs}

    return load


def touch(path: Path) -> None:
    meta = path / "meta.json"
    if meta.exists():
        os.utime(meta)


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _stamp(path: Path) -> float:
    meta = path / "meta.json"
    return (meta if meta.exists() else path).stat().st_mtime


def purge(root: Path, keep: Path | None = None, *, now: float | None = None,
          max_age_days: float = MAX_AGE_DAYS, max_bytes: int = MAX_CACHE_BYTES) -> list[str]:
    """Delete sessions unused for ``max_age_days``, then the least recently used until the cache
    is under ``max_bytes``; never ``keep``. Returns the names removed."""
    base = root / "sessions"
    if not base.is_dir():
        return []
    now = time.time() if now is None else now
    removed = []
    sessions = [p for p in base.iterdir() if p.is_dir() and (keep is None or p.resolve() != keep.resolve())]
    for p in list(sessions):
        if now - _stamp(p) > max_age_days * 86400:
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p.name)
            sessions.remove(p)
    sizes = {p: _size(p) for p in sessions}
    total = sum(sizes.values()) + (_size(keep) if keep is not None and keep.exists() else 0)
    for p in sorted(sessions, key=_stamp):
        if total <= max_bytes:
            break
        total -= sizes[p]
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p.name)
    return removed
