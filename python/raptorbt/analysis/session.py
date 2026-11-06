"""A session: load bars once, evaluate expressions, keep each result under a short handle.

>>> import numpy as np
>>> from raptorbt.analysis import Session
>>> s = Session()
>>> n = 60
>>> close = 100.0 + np.cumsum(np.ones(n))
>>> _ = s.load({"timestamps": np.arange(1, n + 1) * 60 * 10**9, "close": close}, bar_stamp="close")
>>> r = s.eval("sma(close, 5)")
>>> r["shape"], r["warmup"], r["last"][-1]
([60], 4, 158.0)
>>> z = s.eval(f"zscore({r['handle']}, 20)")      # a handle is a series in the next expression
>>> z["warmup"]
23

A handle is ``h_`` and ten hex digits: a hash of the expression, the data and the engine version,
so the same work always has the same handle and a repeat is served from the store. A handle
that was evicted, dropped or never made raises ``handle_expired`` or ``unknown_handle``; the
first carries the call that recomputes it.

The panel path (several instruments at once, ``Expr.eval_panel`` and ``factor_report``) is meant
to enter through :meth:`Session._evaluate`, the one place a program meets data, and through a
second kind of :class:`~raptorbt.analysis.data.Dataset`; nothing here assumes one instrument
beyond that.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import raptorbt
from raptorbt import errors, expressions, stats
from raptorbt.analysis import data as _data
from raptorbt.analysis import summary as _sum

__all__ = ["Session"]

#: ``h_`` + ten hex digits, then ``_<output>`` for one output of a multi-output result.
HANDLE = re.compile(r"\bh_[0-9a-f]{10}(?:_[a-z0-9_]+)?\b")
_HANDLE_FULL = re.compile(HANDLE.pattern.replace(r"\b", ""))
_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")
_TOMBSTONES = 4096
_FIELD_PRICES = ("open", "high", "low", "close")

_STATS_ONE = ("stationarity", "hurst", "garch", "markov_regimes", "hac_tstat")
_STATS_MANY = ("cointegration", "pbo_cscv", "spa_test", "pca_factors")
_STATS_BARS = ("audit_bars", "impact_estimate")
_STATS_ALL = (*_STATS_ONE, *_STATS_MANY, *_STATS_BARS)


@dataclass
class _Handle:
    """One result held by the session: the recipe, the summaries it printed, and the arrays."""

    id: str
    expression: str
    data: str
    data_sha: str
    tz_offset_ns: int
    outputs: tuple[str, ...]
    kinds: tuple[str, ...]
    deps: tuple[str, ...]
    functions: tuple[str, ...]
    causal: bool
    result: dict[str, Any]
    nbytes: int
    leading: dict[str, int]
    shapes: dict[str, tuple[int, ...]]
    arrays: dict[str, np.ndarray] | None = None
    loader: Callable[[], dict[str, np.ndarray]] | None = field(default=None, repr=False)
    path: Any = field(default=None, repr=False)

    def series_name(self, output: str) -> str:
        return self.id if len(self.outputs) == 1 else f"{self.id}_{output}"

    def get(self) -> dict[str, np.ndarray]:
        if self.arrays is None:
            assert self.loader is not None
            self.arrays = self.loader()
        return self.arrays


def _plain(x: Any) -> Any:
    """JSON-ready: numpy scalars and arrays to Python, NaN and infinity to None."""
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Session:
    """Bars, expressions and the results kept between calls, within a memory budget.

    ``max_bytes`` bounds the loaded bars plus every held series; ``max_handles`` bounds the
    number of results. When a new result does not fit, the least recently used handles are
    evicted (their recipes are kept, so an error can say how to recompute one). Series stay
    numpy arrays, read-only.
    """

    def __init__(self, max_bytes: int = 1 << 30, max_handles: int = 256):
        for name, v in (("max_bytes", max_bytes), ("max_handles", max_handles)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise errors.InvalidParameter(
                    f"{name} must be a positive whole number, got {v!r}",
                    code="invalid_parameter", operation="Session", parameter=name, value=v,
                    valid=["a positive whole number"],
                )
        self.max_bytes = max_bytes
        self.max_handles = max_handles
        self._data: dict[str, _data.Dataset] = {}
        self._current: str | None = None
        self._handles: OrderedDict[str, _Handle] = OrderedDict()
        self._gone: OrderedDict[str, tuple[str, str]] = OrderedDict()  # id -> (recipe, reason)
        self._used = 0

    # ------------------------------------------------------------------ data

    def load(
        self,
        bars: Any,
        bar_stamp: str | None = None,
        bar_ns: int | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Hold a copy of ``bars`` and return its facts.

        ``bars`` maps ``timestamps`` (int64 nanoseconds or datetime64) and any of ``open``,
        ``high``, ``low``, ``close``, ``volume`` to 1-D arrays; a DataFrame or any object that
        answers ``bars["close"]`` works, and nothing is imported to read it. ``bar_stamp``
        (``"open"`` or ``"close"``: whether a timestamp starts or ends its bar) is required.
        The input is never modified. The newest dataset is the default for :meth:`eval`.
        """
        return self._adopt(_data.build(bars, bar_stamp, bar_ns, name), "Session.load")

    def _adopt(self, ds: _data.Dataset, op: str) -> dict[str, Any]:
        old = self._data.get(ds.name)
        if old is not None and old.sha256 == ds.sha256:
            self._current = old.name  # the same bars again: nothing to store
            return old.facts()
        extra = ds.nbytes - (old.nbytes if old else 0)
        self._room(ds.nbytes, op, keep=(), replacing=old.nbytes if old else 0)
        self._data[ds.name] = ds
        self._current = ds.name
        self._used += extra
        self._evict(op)
        return ds.facts()

    def load_panel(
        self,
        bars: Any,
        bar_stamp: str | None = None,
        bar_ns: int | None = None,
        instruments: Any = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Hold a copy of a panel (many instruments over many dates) and return its facts.

        ``bars`` is what :meth:`raptorbt.expressions.Expr.eval_panel` takes: a dict of 2-D
        float64 arrays (dates by names) with 1-D int64 ``timestamps`` (and optionally a bool
        ``present``), or a mapping of instrument to its own columns, each with its
        ``timestamps``; the second form is aligned on the union of the timestamps, a name absent
        where it has no bar. ``instruments`` selects and orders the names (for 2-D arrays it
        names the columns, else S0, S1, ...); a name is text of at most 64 characters.
        ``bar_stamp`` is required. The input is never modified. :meth:`eval` then evaluates
        cross-sectional expressions over it and :meth:`factor` scores a result.
        """
        ds = _data.build_panel(bars, bar_stamp, bar_ns, instruments, name)
        return self._adopt(ds, "Session.load_panel")

    def dataset(self, name: str | None = None) -> _data.Dataset:
        """The named dataset, or the newest one."""
        key = name if name is not None else self._current
        if key is None or key not in self._data:
            what = "no bars are loaded: call Session.load first" if name is None else f"no data named {name!r}"
            raise errors.InvalidParameter(
                what, code="missing_field", operation="Session.eval", parameter="data", value=name,
                valid=sorted(self._data) or ["Session.load(bars, bar_stamp=...)"],
            )
        return self._data[key]

    # ------------------------------------------------------------------ eval

    def eval(self, expr: Any, *, data: str | None = None, tz_offset_ns: int = 0) -> dict[str, Any]:
        """Evaluate ``expr`` (text, or an ``Expr``) over the loaded bars and keep the result.

        Returns the summary (see :mod:`raptorbt.analysis.summary`): ``handle``, ``kind``,
        ``shape``, ``warmup``, ``nan``, ``dist`` (or ``true_rate`` for a boolean), ``last``,
        ``flags``, ``causal``, ``reference``, ``sha256``. An expression with several outputs
        returns ``{"handle", "outputs": [summary, ...]}`` and each output is the handle
        ``<handle>_<output>``. Name an earlier result in the text to use it as a series.
        ``data`` picks a loaded dataset (default: the newest); ``tz_offset_ns`` is the zone
        daily windows of ``at(...)`` are cut in.
        """
        op = "Session.eval"
        text = expr.source if isinstance(expr, expressions.Expr) else expr
        if not isinstance(text, str):
            raise errors.InvalidType(
                f"an expression is text, got {type(expr).__name__}", code="expr_type", operation=op,
                parameter="expr", value=repr(expr), valid=["a string such as 'rsi(close, 14)'"],
            )
        ds = self.dataset(data)
        used = self._resolve_text(text, ds, op)
        env = {name: h.leading[out] for name, (h, out) in used.items()}
        program = expressions.Expr(text, series=env)
        bound = program.bound_hash(ds.bar_stamp, tz_offset_ns)
        key = hashlib.sha256(f"{bound}\n{ds.sha256}\n{raptorbt.__version__}".encode()).hexdigest()
        hid = "h_" + key[:10]
        for h, _ in used.values():
            self._handles.move_to_end(h.id)
        held = self._handles.get(hid)
        if held is not None:
            self._handles.move_to_end(hid)
            return copy.deepcopy(held.result)
        arrays = self._evaluate(program, ds, {n: h.get()[o] for n, (h, o) in used.items()}, tz_offset_ns)
        nbytes = sum(a.nbytes for a in arrays.values())
        self._room(nbytes, op, keep=[h.id for h, _ in used.values()], what=f"the result of {_short(text)}")
        entry = self._summarise(hid, text, program, ds, tz_offset_ns, arrays, used, nbytes)
        self._gone.pop(hid, None)
        self._handles[hid] = entry
        self._used += nbytes
        self._evict(op, keep=hid)
        return copy.deepcopy(entry.result)

    def _evaluate(
        self,
        program: expressions.Expr,
        ds: _data.Dataset,
        handles: dict[str, np.ndarray],
        tz_offset_ns: int,
    ) -> dict[str, np.ndarray]:
        """Run ``program`` over ``ds`` and the handle series it reads: output name -> array.

        The one place a program meets data. A panel dataset would be run by ``eval_panel`` here.
        """
        if ds.kind == "panel":
            feed = {k: v for k, v in ds.arrays.items() if k != "present"}
            feed.update(handles)
            got = program.eval_panel(
                feed, present=ds.arrays.get("present"), bar_stamp=ds.bar_stamp,
                tz_offset_ns=tz_offset_ns, bar_ns=ds.bar_ns,
            )
        else:
            feed = {**ds.arrays, **handles}
            got = program.eval(feed, bar_stamp=ds.bar_stamp, tz_offset_ns=tz_offset_ns, bar_ns=ds.bar_ns)
        if not isinstance(got, dict):
            got = {program.outputs[0]: got}
        out = {}
        for name, a in got.items():
            a = np.ascontiguousarray(a, dtype="<f8")
            a.setflags(write=False)
            out[name] = a
        return out

    def _summarise(
        self,
        hid: str,
        text: str,
        program: expressions.Expr,
        ds: _data.Dataset,
        tz_offset_ns: int,
        arrays: dict[str, np.ndarray],
        used: dict[str, tuple[_Handle, str]],
        nbytes: int,
    ) -> _Handle:
        registry = expressions._registry()
        own = [f for f in dict.fromkeys(_CALL.findall(program.canonical)) if f in registry]
        inherited = [f for h, _ in used.values() for f in h.functions]
        functions = tuple(dict.fromkeys([*own, *inherited]))
        causal_by_name = {f: registry[f]["causal"] for f in functions}
        non_causal = _sum.non_causal_names(functions, causal_by_name)
        reference = {f: str(registry[f]["reference"]) for f in functions}
        reads = [ds.arrays[f] for f in program.fields]
        prices = [ds.arrays[f] for f in program.fields if f in _FIELD_PRICES]
        reads += [h.get()[o] for h, o in used.values()]
        names = tuple(program.outputs)
        kinds = program.kinds
        multi = len(names) > 1
        summaries = []
        leading = {}
        for out, kind in zip(names, kinds):
            name = f"{hid}_{out}" if multi else hid
            if ds.kind == "panel":
                s = _sum.summarise_panel(
                    name, arrays[out], kind, ds.names, ds.arrays["timestamps"], warmup=program.warmup,
                    inputs=reads, prices=prices, causal=not non_causal, non_causal=non_causal,
                    reference=reference, extra={"data": ds.name},
                )
            else:
                s = _sum.summarise(
                    name, arrays[out], kind, warmup=program.warmup, inputs=reads, prices=prices,
                    causal=not non_causal, non_causal=non_causal, reference=reference,
                    extra={"data": ds.name},
                )
            if program.warmup is None:
                s["warmup_by_clock"] = program.warmup_by_clock
            summaries.append(s)
            leading[out] = s["nan"]["leading"]
        if multi:
            result: dict[str, Any] = {"handle": hid, "data": ds.name, "outputs": []}
            for s in summaries:
                if _sum.size({**result, "outputs": [*result["outputs"], s]}) > _sum.CALL_BYTES:
                    result.setdefault("truncated_outputs", []).append(s["handle"])
                else:
                    result["outputs"].append(s)
        else:
            result = summaries[0]
        return _Handle(
            id=hid, expression=text, data=ds.name, data_sha=ds.sha256, tz_offset_ns=tz_offset_ns,
            outputs=names, kinds=kinds, deps=tuple(dict.fromkeys(h.id for h, _ in used.values())),
            functions=functions, causal=not non_causal, result=result, nbytes=nbytes,
            leading=leading, shapes={o: a.shape for o, a in arrays.items()}, arrays=arrays,
        )

    def _resolve_text(self, text: str, ds: _data.Dataset, op: str) -> dict[str, tuple[_Handle, str]]:
        """Handle names in ``text`` -> (entry, output name); a missing one raises."""
        used: dict[str, tuple[_Handle, str]] = {}
        for name in dict.fromkeys(HANDLE.findall(text)):
            h, out = self._lookup(name, op)
            if h.data_sha != ds.sha256:
                raise errors.InvalidData(
                    f"{name} was computed on data {h.data!r} (sha256 {h.data_sha[:12]}), not on "
                    f"{ds.name!r} (sha256 {ds.sha256[:12]}); a series of other bars cannot be mixed in",
                    code="invalid_input", operation=op, parameter=name, value=name,
                    valid=[self._recipe(h.expression, h.data, h.tz_offset_ns)],
                )
            if "report" in h.kinds:
                raise self._no_output(name, h, op, "is a factor report, not a series: read it with Session.series")
            if out == "":
                raise self._no_output(name, h, op, "is a result with several outputs: name one")
            used[name] = (h, out)
        return used

    def _lookup(self, name: Any, op: str) -> tuple[_Handle, str]:
        """The entry and the output a handle name stands for, or the typed error."""
        if not isinstance(name, str) or not _HANDLE_FULL.fullmatch(name):
            raise errors.InvalidParameter(
                f"{name!r} is not a handle: handles look like h_0123456789",
                code="unknown_handle", operation=op, parameter="handle", value=name,
                valid=list(self._handles)[-5:] or ["a handle returned by Session.eval"],
            )
        base = name[:12]
        out = name[13:] or None
        h = self._handles.get(base)
        if h is None:
            if base in self._gone:
                recipe, reason = self._gone[base]
                raise errors.StateError(
                    f"handle {base} was {reason}; recompute it with {recipe}",
                    code="handle_expired", operation=op, parameter="handle", value=name,
                    valid=[recipe],
                )
            near = difflib.get_close_matches(base, list(self._handles), n=5, cutoff=0.5)
            raise errors.InvalidParameter(
                f"no handle {name!r} in this session",
                code="unknown_handle", operation=op, parameter="handle", value=name,
                valid=near or list(self._handles)[-5:] or ["a handle returned by Session.eval"],
            )
        if len(h.outputs) == 1:
            if out is not None:
                raise self._no_output(name, h, op)
            return h, h.outputs[0]
        if out is None:
            return h, ""  # the group; a caller that needs one series must name an output
        if out not in h.outputs:
            raise self._no_output(name, h, op)
        return h, out

    def _no_output(self, name: str, h: _Handle, op: str, why: str | None = None) -> errors.RaptorBTError:
        return errors.InvalidParameter(
            f"{name!r} {why or f'names no output of {h.id}'}",
            code="unknown_handle", operation=op, parameter="handle", value=name,
            valid=[h.series_name(o) for o in h.outputs],
        )

    # ------------------------------------------------------------------ the store

    def _recipe(self, expression: str, data: str, tz_offset_ns: int = 0) -> str:
        tz = f", tz_offset_ns={tz_offset_ns}" if tz_offset_ns else ""
        return f"Session.eval({json.dumps(expression)}, data={data!r}{tz})"

    def _forget(self, h: _Handle, reason: str) -> None:
        self._gone[h.id] = (self._recipe(h.expression, h.data, h.tz_offset_ns), reason)
        while len(self._gone) > _TOMBSTONES:
            self._gone.popitem(last=False)
        self._used -= h.nbytes

    def _evict(self, op: str, keep: str | None = None) -> None:
        """Evict least recently used handles until both limits hold; never ``keep``."""
        while len(self._handles) > self.max_handles or self._used > self.max_bytes:
            victim = next((k for k in self._handles if k != keep), None)
            if victim is None:
                return
            h = self._handles.pop(victim)
            self._forget(h, f"evicted to keep the session inside max_handles={self.max_handles} "
                            f"and max_bytes={self.max_bytes}")

    def _room(self, extra: int, op: str, keep: Any, what: str = "the data", replacing: int = 0) -> None:
        """Refuse an addition of ``extra`` bytes that no eviction could fit."""
        held_data = sum(d.nbytes for d in self._data.values()) - replacing
        pinned = sum(self._handles[k].nbytes for k in keep if k in self._handles)
        if held_data + pinned + extra > self.max_bytes:
            raise errors.InvalidParameter(
                f"{what} needs {extra} bytes; with the {held_data} bytes of loaded bars"
                f"{f' and the {pinned} bytes of the handles it reads' if pinned else ''} that is "
                f"over max_bytes={self.max_bytes}",
                code="session_memory_exceeded", operation=op, parameter="max_bytes",
                value=self.max_bytes,
                valid=["fewer bars", "fewer outputs", f"Session(max_bytes>={held_data + pinned + extra})"],
            )

    # ------------------------------------------------------------------ handles

    def series(self, handle: str, tail: int | None = None, instruments: Any = None) -> Any:
        """The stored series of ``handle`` as a read-only float64 array, full precision.

        ``tail=n`` gives the last ``n`` values (dates, for a panel). The handle of a
        multi-output result gives a dict of arrays by output name. For a panel result
        ``instruments`` (a name or a list) picks columns, in that order; an unknown name raises
        ``unknown_symbol`` listing the nearest. Using a handle makes it the most recently used.
        """
        op = "Session.series"
        if tail is not None and (isinstance(tail, bool) or not isinstance(tail, int) or tail < 1):
            raise errors.InvalidParameter(
                f"tail must be a positive whole number, got {tail!r}", code="invalid_parameter",
                operation=op, parameter="tail", value=tail, valid=["a positive whole number"],
            )
        h, out = self._lookup(handle, op)
        self._handles.move_to_end(h.id)
        arrays = h.get()
        cols = None if instruments is None else self._columns(h, instruments, op)

        def pick(a: np.ndarray) -> np.ndarray:
            a = a if tail is None else a[-tail:]
            return a if cols is None else a[:, cols]

        if out == "":
            return {o: pick(a) for o, a in arrays.items()}
        return pick(arrays[out])

    def _columns(self, h: _Handle, instruments: Any, op: str) -> list[int]:
        """Column indices of the named instruments in the panel ``h`` was computed on."""
        ds = self._data.get(h.data)
        if ds is None or ds.sha256 != h.data_sha or ds.kind != "panel":
            raise errors.InvalidParameter(
                f"{h.id} is not a result over a loaded panel, so it has no instruments to pick",
                code="invalid_parameter", operation=op, parameter="instruments",
                valid=["no instruments= for a series"],
            )
        wanted = [instruments] if isinstance(instruments, str) else list(instruments)
        index = {n: j for j, n in enumerate(ds.names)}
        for n in wanted:
            if n not in index:
                raise errors.UnknownSymbol(
                    f"no instrument {n!r} in {ds.name!r}", code="unknown_symbol", operation=op,
                    parameter="instruments", value=n,
                    valid=difflib.get_close_matches(str(n), ds.names, n=5, cutoff=0.4) or list(ds.names[:5]),
                )
        return [index[n] for n in wanted]

    def drop(self, handle: str) -> None:
        """Release a handle; later use raises ``handle_expired`` with its recipe."""
        h, _ = self._lookup(handle, "Session.drop")
        del self._handles[h.id]
        self._forget(h, "dropped")

    def describe_handle(self, handle: str) -> dict[str, Any]:
        """What a handle is: its expression, data, outputs, size and the call that remakes it."""
        h, _ = self._lookup(handle, "Session.describe_handle")
        self._handles.move_to_end(h.id)
        return {
            "handle": h.id,
            "expression": h.expression,
            "data": h.data,
            "outputs": [h.series_name(o) for o in h.outputs],
            "kinds": list(h.kinds),
            "shape": list(next(iter(h.shapes.values()))),
            "bytes": h.nbytes,
            "depends_on": list(h.deps),
            "causal": h.causal,
            "recipe": self._recipe(h.expression, h.data, h.tz_offset_ns),
            "highlights": self._highlights(h),
        }

    @staticmethod
    def _highlights(h: _Handle) -> list[str]:
        """For a panel result, the names at the ends of its last cross-section."""
        res = h.result["outputs"][0] if "outputs" in h.result else h.result
        last = res.get("last") if isinstance(res, dict) else None
        if not isinstance(last, dict):
            return []
        pairs = [*last.get("top", []), *last.get("bottom", [])]
        return list(dict.fromkeys([p[0] for p in pairs] + list(last.get("names", []))))

    def handles(self) -> list[str]:
        """The held handles, least recently used first."""
        return list(self._handles)

    # ------------------------------------------------------------------ factors

    def factor(
        self,
        target: Any,
        *,
        horizons: Any = (1, 5, 20),
        quantiles: int = 5,
        lag: int = 1,
        min_names: int = 10,
        data: str | None = None,
    ) -> dict[str, Any]:
        """Score a panel factor against the returns that follow it (``factor_report``).

        ``target`` is the handle of a panel result (one output) or an expression, evaluated
        first. The forward return of a date starts ``lag`` bars later and runs ``h`` bars over the
        loaded ``close``; nothing here charges costs, and the summary says ``no_costs: true``.

        The summary holds, per horizon, ``n_scored`` dates, ``ic_mean`` (rank IC), ``t_hac`` (its
        t-statistic with a Newey-West correction for the overlap of ``h``-bar returns, ``hac_lags``
        lags) beside ``t_naive``, ``hit_rate``, ``long_short_mean`` (top minus bottom quantile,
        gross) and ``long_short_t_hac``; and ``turnover`` of the top and bottom quantile, the
        rank ``autocorr`` of the factor, and ``names`` held per date (min, median, max). The
        per-date arrays (IC, long-short, coverage, turnover, quantile returns) stay behind the
        summary's ``handle``: ``series(handle)`` returns them by name.
        """
        op = "Session.factor"
        ds = self.dataset(data)
        if ds.kind != "panel":
            raise errors.InvalidParameter(
                f"{ds.name!r} holds bars of one instrument; a factor needs a panel (Session.load_panel)",
                code="invalid_parameter", operation=op, parameter="data", value=ds.name,
                valid=["a dataset made with Session.load_panel"],
            )
        if "close" not in ds.arrays:
            raise errors.InvalidData(
                f"panel {ds.name!r} has no 'close' to take forward returns from", code="missing_field",
                operation=op, parameter="close", valid=["a panel with a close column"],
            )
        if isinstance(target, str) and _HANDLE_FULL.fullmatch(target):
            name = target
        else:
            r = self.eval(target, data=ds.name)
            if "outputs" in r:
                raise errors.InvalidParameter(
                    "a factor has one output: name one of " + ", ".join(o["handle"] for o in r["outputs"]),
                    code="invalid_parameter", operation=op, parameter="target",
                    valid=[o["handle"] for o in r["outputs"]],
                )
            name = r["handle"]
        h, out = self._lookup(name, op)
        if "report" not in h.kinds and out == "":
            raise self._no_output(name, h, op, "is a result with several outputs: name one")
        if "report" in h.kinds or h.data_sha != ds.sha256 or h.get()[out].ndim != 2:
            raise errors.InvalidParameter(
                f"{name} is not a panel result over {ds.name!r}", code="invalid_parameter", operation=op,
                parameter="target", value=name, valid=["a handle of an expression evaluated over this panel"],
            )
        hs = [horizons] if isinstance(horizons, int) and not isinstance(horizons, bool) else horizons
        key = hashlib.sha256(
            f"factor\n{name}\n{list(hs) if not isinstance(hs, int) else hs}\n{quantiles}\n{lag}\n{min_names}"
            f"\n{ds.sha256}\n{raptorbt.__version__}".encode()
        ).hexdigest()
        rid = "h_" + key[:10]
        self._handles.move_to_end(h.id)
        held = self._handles.get(rid)
        if held is not None:
            self._handles.move_to_end(rid)
            return copy.deepcopy(held.result)
        report = expressions.factor_report(
            h.get()[out], ds.arrays["close"], horizons=hs, quantiles=quantiles, lag=lag, min_names=min_names
        )
        arrays: dict[str, np.ndarray] = {}
        for hz, row in report["horizons"].items():
            arrays[f"ic_{hz}"] = row["ic"]
            arrays[f"long_short_{hz}"] = row["long_short"]
            arrays[f"quantile_returns_by_date_{hz}"] = row["quantile_returns_by_date"]
        for k in ("coverage", "names", "autocorr", "turnover_by_date"):
            arrays[k] = report[k]
        arrays = {k: np.ascontiguousarray(a, dtype="<f8") for k, a in arrays.items()}
        for a in arrays.values():
            a.setflags(write=False)
        nbytes = sum(a.nbytes for a in arrays.values())
        self._room(nbytes, op, keep=[h.id], what="the factor report")
        result = _factor_summary(rid, name, ds, report)
        entry = _Handle(
            id=rid, expression=f"factor({name})", data=ds.name, data_sha=ds.sha256, tz_offset_ns=0,
            outputs=tuple(arrays), kinds=("report",) * len(arrays), deps=(h.id,), functions=h.functions,
            causal=h.causal, result=result, nbytes=nbytes, leading={o: 0 for o in arrays},
            shapes={o: a.shape for o, a in arrays.items()}, arrays=arrays,
        )
        self._gone.pop(rid, None)
        self._handles[rid] = entry
        self._used += nbytes
        self._evict(op, keep=rid)
        return copy.deepcopy(result)

    # ------------------------------------------------------------------ statistics

    def stats(self, name: str, *handles: str, data: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Run a :mod:`raptorbt.stats` function on handles.

        - One series (``stationarity``, ``hurst``, ``garch``, ``markov_regimes``,
          ``hac_tstat``): one handle.
        - Several series (``cointegration``, ``pbo_cscv``, ``spa_test``, ``pca_factors``): one
          or more handles, keyed by handle name; a keyword that is a handle name
          (``target=``, ``benchmark=``) is read as its series.
        - Bars (``audit_bars``, ``impact_estimate``): no handles; the loaded bars are used
          (``interval_seconds`` defaults from ``bar_ns``).

        The leading unknown values (warm-up) of the handles are cut, in step, because a test
        refuses NaN; the result's ``analysis`` entry says how many. Other keywords go to the
        function unchanged.
        """
        op = "Session.stats"
        if name not in _STATS_ALL:
            raise errors.InvalidParameter(
                f"Session.stats runs {', '.join(_STATS_ALL)}; got {name!r}", code="invalid_parameter",
                operation=op, parameter="name", value=name, valid=list(_STATS_ALL),
            )
        fn = getattr(stats, name)
        if name in _STATS_BARS:
            if handles:
                raise self._arity(op, name, "takes no handles: it reads the loaded bars")
            return self._stats_bars(name, fn, self.dataset(data), kwargs)
        if not handles or (name in _STATS_ONE and len(handles) != 1):
            raise self._arity(op, name, "takes one handle" if name in _STATS_ONE else "takes one or more handles")
        series = {}
        for h in handles:
            entry, out = self._lookup(h, op)
            if out == "":
                raise self._no_output(h, entry, op)
            self._handles.move_to_end(entry.id)
            series[h] = entry.get()[out]
            if series[h].ndim != 1:
                raise errors.InvalidParameter(
                    f"{h} is a panel (or a report), and {name} runs on one series: pick a name with "
                    f"Session.series(..., instruments=...) and run raptorbt.stats on it",
                    code="invalid_parameter", operation=op, parameter="handles", value=h,
                    valid=["a handle of a series over bars"],
                )
        lead = max(_sum.leading_nan(a) for a in series.values())
        series = {k: a[lead:] for k, a in series.items()}
        for key, v in list(kwargs.items()):
            if isinstance(v, str) and _HANDLE_FULL.fullmatch(v):
                entry, out = self._lookup(v, op)
                kwargs[key] = entry.get()[out][lead:]
        first = next(iter(series.values()))
        result = fn(first, **kwargs) if name in _STATS_ONE else fn(series, **kwargs)
        return {**result, "analysis": {"handles": list(handles), "trimmed_leading": lead}}

    def _arity(self, op: str, name: str, why: str) -> errors.RaptorBTError:
        return errors.InvalidParameter(
            f"{name} {why}", code="invalid_parameter", operation=op, parameter="handles",
            valid=[why],
        )

    def _stats_bars(self, name: str, fn: Callable[..., dict[str, Any]], ds: _data.Dataset, kw: dict[str, Any]) -> dict[str, Any]:
        if ds.kind != "bars":
            raise errors.InvalidParameter(
                f"{name} reads one instrument's bars; {ds.name!r} is a panel", code="invalid_parameter",
                operation="Session.stats", parameter="data", value=ds.name, valid=["a dataset made with Session.load"],
            )
        a = ds.arrays
        need = ("open", "high", "low", "close", "volume") if name == "audit_bars" else ("high", "low", "close", "volume")
        missing = [f for f in need if f not in a]
        if missing:
            raise errors.InvalidData(
                f"{name} needs {', '.join(need)}; the data has no {', '.join(missing)}",
                code="missing_field", operation="Session.stats", parameter=missing[0], valid=list(need),
            )
        if name == "impact_estimate":
            return fn(a["high"], a["low"], a["close"], a["volume"], **kw)
        if "interval_seconds" not in kw:
            if ds.bar_ns is None:
                raise errors.InvalidParameter(
                    "audit_bars needs interval_seconds, and the data has no bar_ns to take it from",
                    code="invalid_parameter", operation="Session.stats", parameter="interval_seconds",
                    valid=["interval_seconds=<bar width in seconds>"],
                )
            kw["interval_seconds"] = max(1, ds.bar_ns // 10**9)
        return fn(a["timestamps"] / 1e9, a["open"], a["high"], a["low"], a["close"], a["volume"], **kw)


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"



def _factor_summary(rid: str, factor: str, ds: _data.Dataset, report: dict[str, Any]) -> dict[str, Any]:
    """The report as numbers: per horizon the IC, its overlap-corrected t, the hit rate and the
    long-short spread; no per-date array (``per_date`` names the ones behind the handle). Cut to
    ``OUTPUT_BYTES`` by dropping the last horizons (named in ``truncated_horizons``)."""
    show = _sum.show
    names = np.asarray(report["names"])
    live = names[names > 0]
    turnover = report["turnover"]
    out: dict[str, Any] = {
        "handle": rid,
        "kind": "report",
        "factor": factor,
        "data": ds.name,
        "no_costs": True,
        "lag": report["lag"],
        "quantiles": report["quantiles"],
        "min_names": report["min_names"],
        "dates": report["n_dates"],
        "names": {"min": int(live.min()), "median": show(np.median(live)), "max": int(live.max())} if live.size else None,
        "autocorr": show(report["autocorr_mean"]),
        "turnover": {"top": show(turnover[-1]), "bottom": show(turnover[0])},
        "horizons": {},
        "per_date": ["ic_<h>", "long_short_<h>", "quantile_returns_by_date_<h>", "coverage", "names", "autocorr", "turnover_by_date"],
    }
    for hz, row in report["horizons"].items():
        out["horizons"][str(hz)] = {
            "n_scored": row["n_scored"],
            "ic_mean": show(row["ic_mean"]),
            "t_hac": show(row["t_hac"]),
            "hac_lags": row["hac_lags"],
            "t_naive": show(row["t_naive"]),
            "hit_rate": show(row["hit_rate"]),
            "long_short_mean": show(row["long_short_mean"]),
            "long_short_t_hac": show(row["long_short_t_hac"]),
        }
    cut = []
    while _sum.size(out) > _sum.OUTPUT_BYTES and len(out["horizons"]) > 1:
        k = next(reversed(out["horizons"]))
        del out["horizons"][k]
        cut.append(k)
        out["truncated_horizons"] = cut
    return out
