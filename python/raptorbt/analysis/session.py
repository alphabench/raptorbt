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
import math
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

#: ``SessionSchedule`` presets a ``session=`` can name (and a stored session can restore).
_PRESETS = (
    "nse_equity", "nse_fo", "bse_equity", "bse_fo", "mcx", "mcx_select_agri", "mcx_other_agri",
    "cds", "cds_cross", "bcd",
)

_STATS_ONE = ("stationarity", "hurst", "garch", "markov_regimes", "hac_tstat", "predictive_test")
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
    #: The session schedule the numbers were cut by (a preset name, or ``custom:<digest>``), when
    #: the expression uses ``at(...)``; none otherwise.
    session: str | None = None

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
        self._schedules: dict[str, Any] = {}  # label -> SessionSchedule, for custom ones
        self._used = 0

    # ------------------------------------------------------------------ data

    def load(
        self,
        bars: Any,
        bar_stamp: str | None = None,
        bar_ns: int | None = None,
        name: str | None = None,
        session: Any = None,
    ) -> dict[str, Any]:
        """Hold a copy of ``bars`` and return its facts.

        ``bars`` maps ``timestamps`` (int64 nanoseconds or datetime64) and any of ``open``,
        ``high``, ``low``, ``close``, ``volume`` to 1-D arrays; a DataFrame or any object that
        answers ``bars["close"]`` works, and nothing is imported to read it. ``bar_stamp``
        (``"open"`` or ``"close"``: whether a timestamp starts or ends its bar) is required.
        The input is never modified. The newest dataset is the default for :meth:`eval`.

        ``session`` is the instrument's trading calendar: a ``raptorbt.SessionSchedule``, or the
        name of a preset (``"nse_equity"``, ``"mcx"``, ...). It is the default of every
        :meth:`eval` and :meth:`sweep` over these bars, and decides what an ``at(...)`` window
        is: without it windows are clock hours and days of the zone, with it they are cut from
        the session's open and a day is the session. Pass the one the instrument's strategy runs
        with, or an ``at("1h", ...)`` here is not the strategy's.
        """
        ds = _data.build(bars, bar_stamp, bar_ns, name)
        _, ds.session = self._schedule_of(session, "Session.load")
        return self._adopt(ds, "Session.load")

    def _schedule_of(self, session: Any, op: str) -> tuple[Any, str | None]:
        """A ``session=`` argument as (schedule, label): a preset name, or a schedule object."""
        if session is None:
            return None, None
        if isinstance(session, str):
            if session not in _PRESETS:
                raise errors.InvalidParameter(
                    f"session {session!r} is not a schedule preset", code="invalid_parameter",
                    operation=op, parameter="session", value=session, valid=list(_PRESETS),
                )
            self._schedules.setdefault(session, getattr(raptorbt.SessionSchedule, session)())
            return self._schedules[session], session
        if not isinstance(session, raptorbt.SessionSchedule):
            raise errors.InvalidType(
                f"session is a SessionSchedule or a preset name, got {type(session).__name__}",
                code="invalid_argument_type", operation=op, parameter="session",
                value=type(session).__name__, valid=["a raptorbt.SessionSchedule", *_PRESETS],
            )
        digest = hashlib.sha256(json.dumps(
            [session.tz_offset_ns, session.squareoff_minute, session.holidays, session.windows,
             sorted(session.overrides.items()), session.coverage], sort_keys=True, default=str,
        ).encode()).hexdigest()[:12]
        label = f"custom:{digest}"
        self._schedules[label] = session
        return session, label

    def _schedule(self, ds: _data.Dataset, session: Any, program: expressions.Expr, op: str) -> tuple[Any, str | None]:
        """The schedule ``program`` is cut by over ``ds``: ``session``, else the dataset's own.
        None when the program does not use ``at(...)``: it would not read it."""
        if session is not None:
            schedule, label = self._schedule_of(session, op)
        elif ds.session is None:
            return None, None
        else:
            label = ds.session
            schedule = self._schedules.get(label)
            if schedule is None and label in _PRESETS:
                schedule, _ = self._schedule_of(label, op)
        if schedule is None:
            raise errors.InvalidParameter(
                f"{ds.name!r} was loaded with the schedule {label}, which a stored session cannot "
                "restore: pass session= again", code="invalid_parameter", operation=op,
                parameter="session", value=label, valid=["a raptorbt.SessionSchedule", *_PRESETS],
            )
        return (schedule, label) if _expressions_at(program) else (None, None)

    def _adopt(self, ds: _data.Dataset, op: str) -> dict[str, Any]:
        old = self._data.get(ds.name)
        if old is not None and old.sha256 == ds.sha256 and old.session == ds.session:
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

    def eval(
        self, expr: Any, *, data: str | None = None, tz_offset_ns: int = 0, session: Any = None
    ) -> dict[str, Any]:
        """Evaluate ``expr`` (text, or an ``Expr``) over the loaded bars and keep the result.

        Returns the summary (see :mod:`raptorbt.analysis.summary`): ``handle``, ``kind``,
        ``shape``, ``warmup``, ``nan``, ``dist`` (or ``true_rate`` for a boolean), ``last``,
        ``flags``, ``causal``, ``reference``, ``sha256``. An expression with several outputs
        returns ``{"handle", "outputs": [summary, ...]}`` and each output is the handle
        ``<handle>_<output>``. Name an earlier result in the text to use it as a series.
        ``data`` picks a loaded dataset (default: the newest); ``tz_offset_ns`` is the zone
        daily windows of ``at(...)`` are cut in; ``session`` (a ``raptorbt.SessionSchedule`` or a
        preset name, default: the one the dataset was loaded with) is the calendar they are cut
        by, and an ``at(...)`` over bars outside the dates it covers raises
        ``session_not_covered``. The schedule is part of the handle's hash.
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
        schedule, label = self._schedule(ds, session, program, op)
        bound = program.bound_hash(ds.bar_stamp, tz_offset_ns, schedule)
        key = hashlib.sha256(f"{bound}\n{ds.sha256}\n{raptorbt.__version__}".encode()).hexdigest()
        hid = "h_" + key[:10]
        for h, _ in used.values():
            self._handles.move_to_end(h.id)
        held = self._handles.get(hid)
        if held is not None:
            self._handles.move_to_end(hid)
            return copy.deepcopy(held.result)
        arrays = self._evaluate(program, ds, {n: h.get()[o] for n, (h, o) in used.items()}, tz_offset_ns, schedule)
        nbytes = sum(a.nbytes for a in arrays.values())
        self._room(nbytes, op, keep=[h.id for h, _ in used.values()], what=f"the result of {_short(text)}")
        entry = self._summarise(hid, text, program, ds, tz_offset_ns, arrays, used, nbytes, label)
        self._gone.pop(hid, None)
        self._handles[hid] = entry
        self._used += nbytes
        self._evict(op, keep=hid)
        return copy.deepcopy(entry.result)

    def sweep(
        self,
        expr: Any,
        *,
        data: str | None = None,
        tz_offset_ns: int = 0,
        session: Any = None,
        score: str | None = None,
        grid: Mapping[str, Any] | None = None,
        points: Any = None,
        **axes: Any,
    ) -> dict[str, Any]:
        """Evaluate ``expr`` under many values of its ``$name`` constants and keep each result.

        ``sma(close, $n) > sma(close, $m)`` with ``n=[5, 10, 20], m=[50, 100]`` (or ``grid=``,
        or ``points=[{...}]``, see :meth:`raptorbt.expressions.Expr.sweep`) is six bindings run
        in one pass; each binding's result is a handle of its own, the one :meth:`eval` of the
        bound expression returns. Returns a compact table: ``columns`` (``handle``, the swept
        parameters, ``score``), one row per binding, and ``n_trials``, the number of bindings
        tried. ``score`` is a statistic of each series (``true_rate``, ``mean``, ``std``,
        ``min``, ``max``, ``last``, ``known``, ``leading_nan``; the default is ``true_rate`` for
        a boolean and ``mean`` for a number) or an expression, which may read the same
        ``$name``s and is reduced to the mean of its known values. A table too long for the
        size cap keeps the highest scores, in score order, and says how many it left out; every
        binding still has its handle. The table states the selection bias: the best of
        ``n_trials`` scores is not the score of one chosen in advance. ``tz_offset_ns`` and
        ``session`` are as for :meth:`eval`.
        """
        op = "Session.sweep"
        text = expr.source if isinstance(expr, expressions.Expr) else expr
        if not isinstance(text, str):
            raise errors.InvalidType(
                f"an expression is text, got {type(expr).__name__}", code="expr_type", operation=op,
                parameter="expr", value=repr(expr), valid=["a string such as 'sma(close, $n) > close'"],
            )
        ds = self.dataset(data)
        if ds.kind == "panel":
            raise errors.make(
                "expr_unsupported_here",
                "a sweep runs over the bars of one instrument; sweep a panel with "
                "Expr.sweep(arrays, ...) and eval_panel's arguments",
                operation=op, parameter="data", value=ds.name,
                valid=["a dataset loaded with Session.load"],
            )
        used = self._resolve_text(text, ds, op)
        env = {name: h.leading[out] for name, (h, out) in used.items()}
        template = expressions.Expr(text, series=env)
        schedule, label = self._schedule(ds, session, template, op)
        feed = {**ds.arrays, **{n: h.get()[o] for n, (h, o) in used.items()}}
        swept = _sweep_names(grid, axes, points)
        found = template.sweep(
            feed, grid=grid, points=points, bar_stamp=ds.bar_stamp, tz_offset_ns=tz_offset_ns,
            session=schedule, bar_ns=ds.bar_ns, **axes,
        )
        if len(found.outputs) != 1 or found.outputs[0] != "value":
            raise errors.make(
                "expr_type",
                "a sweep summarises one series per binding; select an output of the expression",
                operation=op, parameter="expr", value=text,
                valid=[f"({text}).{o}" for o in found.outputs],
            )
        keep = [h.id for h, _ in used.values()]
        bound = {}
        for point in found:
            key = tuple(sorted(point.params.items()))
            if key not in bound:
                bound[key] = expressions.Expr(text, point.params, series=env)
        ids = {
            key: "h_" + hashlib.sha256(
                f"{e.bound_hash(ds.bar_stamp, tz_offset_ns, schedule)}\n{ds.sha256}\n{raptorbt.__version__}".encode()
            ).hexdigest()[:10]
            for key, e in bound.items()
        }
        distinct = len(ids)
        if distinct > self.max_handles:
            raise errors.InvalidParameter(
                f"the sweep makes {distinct} distinct results; the session holds at most "
                f"max_handles={self.max_handles}",
                code="session_memory_exceeded", operation=op, parameter="max_handles",
                value=self.max_handles,
                valid=["fewer combinations", f"Session(max_handles>={distinct})"],
            )
        new = {ids[k]: next(p for p in found if tuple(sorted(p.params.items())) == k).value.nbytes
               for k in ids if ids[k] not in self._handles}
        self._room(sum(new.values()), op, keep=keep, what=f"the {len(new)} results of the sweep")
        for point in found:
            key = tuple(sorted(point.params.items()))
            hid = ids[key]
            if hid in self._handles:
                self._handles.move_to_end(hid)
                continue
            a = np.ascontiguousarray(point.value, dtype="<f8")
            a.setflags(write=False)
            e = bound[key]
            entry = self._summarise(
                hid, e.canonical, e, ds, tz_offset_ns, {e.outputs[0]: a}, used, a.nbytes, label
            )
            self._gone.pop(hid, None)
            self._handles[hid] = entry
            self._used += a.nbytes
            self._evict(op, keep=hid)
        for h, _ in used.values():
            if h.id in self._handles:
                self._handles.move_to_end(h.id)

        kind = bound[next(iter(bound))].kinds[0]
        score_label, scores = self._scores(score, kind, found, tz_offset_ns, schedule, ds, op)
        rows = [
            [ids[tuple(sorted(p.params.items()))], *[_whole(p.params[n]) for n in swept], _sum.show(sc) if sc is not None else None]
            for p, sc in zip(found, scores)
        ]
        result: dict[str, Any] = {
            "expression": text, "data": ds.name, "n_trials": found.n_trials, "n_distinct": distinct,
            "swept": swept, "score": score_label, "columns": ["handle", *swept, "score"], "rows": rows,
            "order": "input",
            "kernel_steps": {"joint": found.n_kernel_steps, "independent": found.independent_kernel_steps},
            "note": f"best of n_trials={found.n_trials}: a selection among them is biased upward; "
                    "pass the Sharpe of every trial to raptorbt.research.deflated_sharpe",
        }
        if _sum.size(result) > _sum.CALL_BYTES:
            ranked = sorted(range(len(rows)), key=lambda i: (rows[i][-1] is None, -(rows[i][-1] or 0.0)))
            keep_n = len(rows)
            while keep_n > 1 and _sum.size({**result, "rows": [rows[i] for i in ranked[:keep_n]], "order": "score, highest first", "omitted": len(rows) - keep_n}) > _sum.CALL_BYTES:
                keep_n -= 1
            result["rows"] = [rows[i] for i in ranked[:keep_n]]
            result["order"] = "score, highest first"
            result["omitted"] = len(rows) - keep_n
        return result

    def _scores(self, score: str | None, kind: str, found: Any, tz_offset_ns: int, schedule: Any, ds: _data.Dataset, op: str) -> tuple[str, list[float | None]]:
        """The label and one score per binding of ``found``: a statistic or an expression."""
        label = score if score is not None else ("true_rate" if kind == "bool" else "mean")
        if label in _SCORES:
            return label, [_SCORES[label](p.value) for p in found]
        if not isinstance(label, str):
            raise errors.InvalidParameter(
                f"score is a statistic name or an expression, got {type(label).__name__}",
                code="invalid_parameter", operation=op, parameter="score", value=repr(label),
                valid=sorted(_SCORES),
            )
        probe = expressions.Expr(label)
        names = list(probe.parameters)
        if not names:
            out = probe.eval(ds.arrays, bar_stamp=ds.bar_stamp, tz_offset_ns=tz_offset_ns, session=schedule, bar_ns=ds.bar_ns)
            return label, [_known_mean(out)] * len(found)
        pts = [{n: p.params[n] for n in names if n in p.params} for p in found]
        scored = probe.sweep(
            ds.arrays, points=pts, bar_stamp=ds.bar_stamp, tz_offset_ns=tz_offset_ns,
            session=schedule, bar_ns=ds.bar_ns,
        )
        return label, [_known_mean(p.value) for p in scored]

    def _evaluate(
        self,
        program: expressions.Expr,
        ds: _data.Dataset,
        handles: dict[str, np.ndarray],
        tz_offset_ns: int,
        schedule: Any = None,
    ) -> dict[str, np.ndarray]:
        """Run ``program`` over ``ds`` and the handle series it reads: output name -> array.

        The one place a program meets data. A panel dataset would be run by ``eval_panel`` here.
        """
        if ds.kind == "panel":
            feed = {k: v for k, v in ds.arrays.items() if k != "present"}
            feed.update(handles)
            got = program.eval_panel(
                feed, present=ds.arrays.get("present"), bar_stamp=ds.bar_stamp,
                tz_offset_ns=tz_offset_ns, session=schedule, bar_ns=ds.bar_ns,
            )
        else:
            feed = {**ds.arrays, **handles}
            got = program.eval(
                feed, bar_stamp=ds.bar_stamp, tz_offset_ns=tz_offset_ns, session=schedule,
                bar_ns=ds.bar_ns,
            )
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
        session: str | None = None,
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
            session=session,
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
                    valid=[self._recipe(h.expression, h.data, h.tz_offset_ns, h.session)],
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

    def _recipe(self, expression: str, data: str, tz_offset_ns: int = 0, session: str | None = None) -> str:
        tz = f", tz_offset_ns={tz_offset_ns}" if tz_offset_ns else ""
        if session is None:
            cut = ""
        elif session in _PRESETS:
            cut = f", session={session!r}"
        else:
            cut = f", session=<the SessionSchedule {session}>"
        return f"Session.eval({json.dumps(expression)}, data={data!r}{tz}{cut})"

    def _forget(self, h: _Handle, reason: str) -> None:
        self._gone[h.id] = (self._recipe(h.expression, h.data, h.tz_offset_ns, h.session), reason)
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
            "recipe": self._recipe(h.expression, h.data, h.tz_offset_ns, h.session),
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
        cost_bps_per_side: float | None = None,
        data: str | None = None,
    ) -> dict[str, Any]:
        """Score a panel factor against the returns that follow it (``factor_report``).

        ``target`` is the handle of a panel result (one output) or an expression, evaluated
        first. The forward return of a date starts ``lag`` bars later and runs ``h`` bars over the
        loaded ``close``; nothing here charges costs unless ``cost_bps_per_side`` is given, and the
        summary says ``no_costs: true``. With it each horizon also holds ``long_short_net_mean``
        and ``long_short_net_t_nonoverlap`` (the spread less ``2 c (T_top + T_bottom)``, both legs,
        entry and exit: see :func:`raptorbt.expressions.factor_report`); every horizon holds
        ``breakeven_cost_bps_per_side``, the flat cost per side at which the spread is gone. A
        signal whose breakeven is below its costs is not a strategy.

        The summary holds, per horizon, ``n_scored`` dates, ``ic_mean`` (rank IC), ``t_nonoverlap``
        (the headline t: the IC of every ``h``-th date, with ``p_nonoverlap`` and
        ``n_independent``) beside ``t_hac`` (Newey-West, ``hac_lags`` lags; known to over-reject
        for ``h > 1``) and ``t_naive``, ``hit_rate``, ``long_short_mean`` (top minus bottom
        quantile, gross), ``long_short_t_nonoverlap``, ``long_short_t_hac`` and
        ``names_left_share`` (names that left within the horizon, dropped not scored). Top level:
        ``flags`` (unadjusted-split and leaver warnings) and ``assumptions``; and ``turnover`` of the top and bottom quantile, the
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
            f"\n{cost_bps_per_side}\n{ds.sha256}\n{raptorbt.__version__}".encode()
        ).hexdigest()
        rid = "h_" + key[:10]
        self._handles.move_to_end(h.id)
        held = self._handles.get(rid)
        if held is not None:
            self._handles.move_to_end(rid)
            return copy.deepcopy(held.result)
        report = expressions.factor_report(
            h.get()[out], ds.arrays["close"], horizons=hs, quantiles=quantiles, lag=lag, min_names=min_names,
            cost_bps_per_side=cost_bps_per_side,
        )
        arrays: dict[str, np.ndarray] = {}
        for hz, row in report["horizons"].items():
            arrays[f"ic_{hz}"] = row["ic"]
            arrays[f"long_short_{hz}"] = row["long_short"]
            if "long_short_net" in row:
                arrays[f"long_short_net_{hz}"] = row["long_short_net"]
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
          ``hac_tstat``): one handle. ``predictive_test`` takes the signal's handle; its prices
          are the ``close`` of the bars the signal was made on (or ``prices=`` another handle),
          its timestamps and, with ``to_session_close``, its session calendar those bars' too.
        - Several series (``cointegration``, ``pbo_cscv``, ``spa_test``, ``pca_factors``): one
          or more handles, keyed by handle name; a keyword that is a handle name
          (``target=``, ``benchmark=``) is read as its series.
        - Bars (``audit_bars``, ``impact_estimate``): no handles; the loaded bars are used
          (``interval_seconds`` defaults from ``bar_ns``).

        The leading unknown values (warm-up) of the handles are cut, in step, because a test
        refuses NaN; the result's ``analysis`` entry says how many. Series made on different bars
        are joined on their timestamps, never by position: only the instants they all have are
        used, and ``analysis["aligned"]`` says how many, from when to when, and how many each
        series lost; series with no common instant (or bars no longer loaded) are refused.
        Other keywords go to the function unchanged.
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
        entries = {}
        for h in handles:
            entry, out = self._lookup(h, op)
            if out == "":
                raise self._no_output(h, entry, op)
            self._handles.move_to_end(entry.id)
            entries[h] = (entry, out)
            if entry.get()[out].ndim != 1:
                raise errors.InvalidParameter(
                    f"{h} is a panel (or a report), and {name} runs on one series: pick a name with "
                    f"Session.series(..., instruments=...) and run raptorbt.stats on it",
                    code="invalid_parameter", operation=op, parameter="handles", value=h,
                    valid=["a handle of a series over bars"],
                )
        for key, v in kwargs.items():
            if isinstance(v, str) and _HANDLE_FULL.fullmatch(v):
                entry, out = self._lookup(v, op)
                if out == "":
                    raise self._no_output(v, entry, op)
                self._handles.move_to_end(entry.id)
                entries[v] = (entry, out)
        columns, alignment = self._aligned({h: e.get()[out] for h, (e, out) in entries.items()},
                                           {h: e for h, (e, _) in entries.items()}, op)
        lead = max(_sum.leading_nan(columns[h]) for h in handles)
        series = {h: columns[h][lead:] for h in handles}
        for key, v in list(kwargs.items()):
            if isinstance(v, str) and v in columns:
                kwargs[key] = columns[v][lead:]
        first = next(iter(series.values()))
        if name == "predictive_test":
            self._predictive_inputs(entries[handles[0]][0], lead, kwargs, op)
        result = fn(first, **kwargs) if name in _STATS_ONE else fn(series, **kwargs)
        analysis = {"handles": list(handles), "trimmed_leading": lead}
        if alignment is not None:
            analysis["aligned"] = alignment
        return {**result, "analysis": analysis}

    def _predictive_inputs(self, entry: _Handle, lead: int, kw: dict[str, Any], op: str) -> None:
        """What ``predictive_test`` reads beside the signal, from the bars the signal was made on:
        the close (unless ``prices=`` names a handle), the timestamps and the session calendar of
        a ``to_session_close`` test."""
        ds = self._data.get(entry.data)
        if ds is None or ds.sha256 != entry.data_sha or ds.kind != "bars":
            raise errors.InvalidParameter(
                f"the signal was made on bars ({entry.data!r}) that are no longer loaded, so its "
                "prices cannot be read", code="invalid_input", operation=op, parameter="handles",
                valid=[f"Session.load the bars of {entry.data!r} again and recompute the signal"],
            )
        if "prices" not in kw:
            if "close" not in ds.arrays:
                raise errors.InvalidData(
                    f"{ds.name!r} has no 'close' to take forward returns from", code="missing_field",
                    operation=op, parameter="prices", valid=["bars with a close column", "prices=<handle>"],
                )
            kw["prices"] = ds.arrays["close"][lead:]
        if kw.get("to_session_close"):
            kw.setdefault("timestamps", ds.arrays["timestamps"][lead:])
            session = kw.get("session", ds.session)
            if isinstance(session, str):
                session, _ = self._schedule_of(session, op)
            elif session is None and ds.session is not None:
                session, _ = self._schedule_of(ds.session, op)
            kw["session"] = session

    def _aligned(
        self, arrays: dict[str, np.ndarray], entries: dict[str, _Handle], op: str
    ) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
        """The series on the timestamps they all have, in step.

        Series made on the same bars are already in step. Series of different bars (two
        instruments, two date ranges) are joined on their timestamps, never by position: a
        statistic of rows that are not the same instant is a number about nothing. Their common
        timestamps are returned (``aligned``: how many, from when to when); with none, or bars
        no longer held to read the timestamps from, the call is refused.
        """
        if len({e.data_sha for e in entries.values()}) <= 1:
            return arrays, None
        stamps: dict[str, np.ndarray] = {}
        for h, e in entries.items():
            ds = self._data.get(e.data)
            if ds is None or ds.sha256 != e.data_sha or ds.kind != "bars":
                raise errors.InvalidParameter(
                    f"{h} was computed on bars ({e.data!r}) that are no longer loaded, so its "
                    "timestamps cannot be matched with the other series",
                    code="invalid_input", operation=op, parameter=h, value=h,
                    valid=[f"Session.load the bars of {e.data!r} again and recompute {h}"],
                )
            stamps[h] = ds.arrays["timestamps"]
        common = None
        for ts in stamps.values():
            common = ts if common is None else np.intersect1d(common, ts, assume_unique=True)
        if common is None or common.size < 2:
            raise errors.InvalidData(
                f"the series share {int(common.size)} timestamp(s): a statistic needs at least two",
                code="invalid_input", operation=op, parameter="handles",
                valid=["handles of series over bars with common timestamps"],
            )
        out = {h: a[np.isin(stamps[h], common, assume_unique=True)] for h, a in arrays.items()}
        info = {
            "common": int(common.size), "first_timestamp": int(common[0]),
            "last_timestamp": int(common[-1]),
            "dropped": {h: int(len(stamps[h]) - common.size) for h in arrays},
        }
        return out, info

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


def _expressions_at(program: expressions.Expr) -> bool:
    """Whether the program calls ``at(...)``, read from its text: a sweep template may still have
    an unbound ``$name``, which compiling would refuse."""
    return bool(expressions._AT_CALL.search(program.source))


def _whole(x: float) -> float | int:
    """A parameter as shown: ``5`` for ``5.0``."""
    return int(x) if float(x).is_integer() else x


def _known(a: Any) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def _known_mean(a: Any) -> float | None:
    v = _known(a)
    return float(v.mean()) if v.size else None


def _last_known(a: Any) -> float | None:
    v = _known(a)
    return float(v[-1]) if v.size else None


#: The statistics ``Session.sweep`` can score a series by; each is None where nothing is known.
_SCORES: dict[str, Callable[[Any], float | None]] = {
    "true_rate": _known_mean,
    "mean": _known_mean,
    "std": lambda a: float(_known(a).std()) if _known(a).size else None,
    "min": lambda a: float(_known(a).min()) if _known(a).size else None,
    "max": lambda a: float(_known(a).max()) if _known(a).size else None,
    "last": _last_known,
    "known": lambda a: float(_known(a).size),
    "leading_nan": lambda a: float(_sum.leading_nan(np.asarray(a, dtype=np.float64))),
}


def _sweep_names(grid: Mapping[str, Any] | None, axes: Mapping[str, Any], points: Any) -> list[str]:
    """The parameters a sweep varies, in the order the table shows them."""
    if points is not None:
        return sorted({str(k) for p in points for k in p})
    return [str(k) for k in (*(grid or {}), *axes)]


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"



def _flags_by_weight(flags: list[str]) -> list[str]:
    """The first flag that a signal does not pay its costs, then the rest in order: the cap keeps
    three, and one cost flag per horizon must not crowd out the split and leaver warnings."""
    cost = [f for f in flags if "pay its costs" in f][:1]
    return cost + [f for f in flags if f not in cost]


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
        "no_costs": report["no_costs"],
        "cost_bps_per_side": report["cost_bps_per_side"],
        "lag": report["lag"],
        "quantiles": report["quantiles"],
        "min_names": report["min_names"],
        "dates": report["n_dates"],
        "names": {"min": int(live.min()), "median": show(np.median(live)), "max": int(live.max())} if live.size else None,
        "autocorr": show(report["autocorr_mean"]),
        "turnover": {"top": show(turnover[-1]), "bottom": show(turnover[0])},
        "headline": "t_nonoverlap",
        "t_hac_note": "t_hac is known to over-reject for h > 1; read t_nonoverlap",
        "flags": [
            f if len(f) <= 160 else f[:157] + "..."
            for f in _flags_by_weight(report["flags"])[:3]
        ],
        "assumptions": report["assumptions"][:2],
        "horizons": {},
        "per_date": ["ic_<h>", "long_short_<h>", "long_short_net_<h> (with a cost)", "quantile_returns_by_date_<h>", "coverage", "names", "autocorr", "turnover_by_date"],
    }
    for hz, row in report["horizons"].items():
        out["horizons"][str(hz)] = {
            "n_scored": row["n_scored"],
            "ic_mean": show(row["ic_mean"]),
            "t_nonoverlap": show(row["t_nonoverlap"]),
            "p_nonoverlap": show(row["p_nonoverlap"]),
            "n_independent": row["n_independent"],
            "names_left_share": show(row["names_left_share"]),
            "t_hac": show(row["t_hac"]),
            "hac_lags": row["hac_lags"],
            "t_naive": show(row["t_naive"]),
            "hit_rate": show(row["hit_rate"]),
            "long_short_mean": show(row["long_short_mean"]),
            "long_short_t_nonoverlap": show(row["long_short_t_nonoverlap"]),
            "long_short_t_hac": show(row["long_short_t_hac"]),
            "breakeven_cost_bps_per_side": (
                "unbounded" if row["breakeven_cost_bps_per_side"] == math.inf
                else show(row["breakeven_cost_bps_per_side"])
            ),
        }
        if "long_short_net" in row:
            out["horizons"][str(hz)]["long_short_net_mean"] = show(row["long_short_net_mean"])
            out["horizons"][str(hz)]["long_short_net_t_nonoverlap"] = show(row["long_short_net_t_nonoverlap"])
    cut = []
    while _sum.size(out) > _sum.OUTPUT_BYTES and len(out["horizons"]) > 1:
        k = next(reversed(out["horizons"]))
        del out["horizons"][k]
        cut.append(k)
        out["truncated_horizons"] = cut
    return out
