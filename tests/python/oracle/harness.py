"""The only module in the oracle tests that knows the expression API.

Everything else (replay, conformance) asks this module to evaluate a registered function on
columns, step it bar by bar, or report its warm-up.

API used (python/raptorbt/expressions.py):
    raptorbt.expressions.functions() / describe(name)   registry rows; the full row (inputs with
        `default_field`, params with `default`/`range`, `talib_name`, `reference`, `relation`,
        `decay`, `nan_policy`, `domain`, `invariance`, `causal`) comes from describe()
    raptorbt.expr(text).eval(data) -> array, or dict of arrays for several outputs
    raptorbt.expr(text).stream().update(bar) -> float, or dict for several outputs
    raptorbt.expr(text).warmup
Expressions read only the five fields open/high/low/close/volume, so a fixture column that is
not a field ("unit", "periods") is bound to an unused field of the same position.
"""

import numpy as np

FIELDS = ("open", "high", "low", "close", "volume")
_PREFERRED = ("close", "high", "low", "open", "volume")


def registry():
    from raptorbt import expressions

    # The session family reads the bars' clock and a schedule, not a column: its hand tests are
    # fns::session::tests and tests/python/test_expr_session.py.
    return [expressions.describe(r["name"]) for r in expressions.functions() if r["family"] != "session"]


def talib_names(fn):
    t = fn.get("talib_name")
    if not t:
        return []
    return [t] if isinstance(t, str) else list(t)


def output_names(fn):
    return list(fn.get("outputs") or [])


def input_names(fn):
    """The field each input reads when the harness has no fixture to say: its default field,
    else the next unused one."""
    used, cols = set(), []
    for i in fn.get("inputs") or []:
        f = i.get("default_field")
        if f is None or f in used:
            f = next(x for x in _PREFERRED if x not in used)
        used.add(f)
        cols.append(f)
    return cols


def _pick(p):
    if p["default"] is not None:
        return p["default"]
    r = p["range"]
    if p["kind"] == "int":
        return min(max(14, r["min"]), r["max"])
    if p["kind"] == "float":
        return min(max(2.0, r["min"]), r["max"])
    return r["values"][0]


def param_defaults(fn):
    """A usable value for every parameter: its default, else 14 / 2.0 / the first enum value
    clamped into range."""
    return {p["name"]: _pick(p) for p in fn.get("params") or []}


def _bind(columns, data):
    """Fixture column names -> fields. Returns (field names, {field: array})."""
    fields, out, used = [], {}, set()
    for c in columns:
        f = c
        if f not in FIELDS or (f in used):
            f = next(x for x in _PREFERRED if x not in used and x not in columns)
        used.add(f)
        fields.append(f)
        out[f] = data[c]
    return fields, out


def call_text(fn, params=None, fields=None):
    params = {**param_defaults(fn), **(params or {})}
    args = list(fields if fields is not None else input_names(fn))
    for k, i in enumerate(fn.get("inputs") or []):
        if i.get("elem") == "bool":
            # A boolean input is fed a comparison of two of the function's own columns.
            other = args[(k + 1) % len(args)] if len(args) > 1 else "0"
            args[k] = f"({args[k]} > {other})"
    for p in fn.get("params") or []:
        v = params[p["name"]]
        args.append(f'"{v}"' if p["kind"] == "enum" else repr(v))
    return f"{fn['name']}({', '.join(args)})"


def _compile(text):
    import raptorbt as rb

    return rb.expr(text)


def _as_dict(res, fn):
    if isinstance(res, dict):
        return {k: np.asarray(v, dtype=np.float64) for k, v in res.items()}
    return {output_names(fn)[0]: np.asarray(res, dtype=np.float64)}


# A cross-sectional function has a value only over a panel. The harness folds a series into a
# panel of ROW names per date (bar i is name i % ROW of date i // ROW; the last date may be
# short, its missing names absent), so every bar is a cell, the cross-section is made of real
# data, and a uniform scale or shift of the series is a uniform scale or shift of the panel.
# A group input is replaced by the labels i % ROW // 2 (names in pairs).
ROW = 6


def is_cross(fn):
    return fn.get("family") == "cross"


def row_start(i):
    """First bar of the date (row) that bar `i` belongs to."""
    return (i // ROW) * ROW


def _fold(x):
    n = len(x)
    rows = -(-n // ROW)
    flat = np.full(rows * ROW, np.nan)
    flat[:n] = x
    return flat.reshape(rows, ROW)


def _panel_fields(fn, fields, fdata):
    panels = {f: _fold(v) for f, v in fdata.items()}
    if any(i.get("name") == "group" for i in fn.get("inputs") or []):
        g = fields[1]
        labels = np.tile((np.arange(ROW) // 2).astype(np.float64), (panels[g].shape[0], 1))
        panels[g] = labels
    return panels


def _eval_panel_series(fn, text, fields, fdata, rows=None):
    n = len(next(iter(fdata.values())))
    panels = _panel_fields(fn, fields, fdata)
    if rows is not None:
        panels = {f: p[rows : rows + 1] for f, p in panels.items()}
    res = _compile(text).eval_panel(panels)
    return np.asarray(res, dtype=np.float64).reshape(-1)[: n if rows is None else ROW]


def evaluate(fn, data, params=None, columns=None):
    """Batch evaluation. `data` maps column name to a float64 array; `columns` names the
    columns the inputs read, in order (default: the fields). Returns {output: array}."""
    cols = list(columns) if columns is not None else input_names(fn)
    fields, fdata = _bind(cols, data)
    if is_cross(fn):
        text = call_text(fn, params, fields)
        return {output_names(fn)[0]: _eval_panel_series(fn, text, fields, fdata)}
    return _as_dict(_compile(call_text(fn, params, fields)).eval(fdata), fn)


def evaluate_stream(fn, data, params=None, columns=None):
    cols = list(columns) if columns is not None else input_names(fn)
    fields, fdata = _bind(cols, data)
    n = len(next(iter(fdata.values())))
    outs = output_names(fn)
    rec = {o: np.full(n, np.nan) for o in outs}
    if is_cross(fn):
        # "Streaming" a cross-section is evaluating one date on its own: the result of the
        # whole panel must be the date-by-date result, bit for bit.
        text = call_text(fn, params, fields)
        for r in range(-(-n // ROW)):
            row = _eval_panel_series(fn, text, fields, fdata, rows=r)
            lo = r * ROW
            rec[outs[0]][lo : min(lo + ROW, n)] = row[: min(ROW, n - lo)]
        return rec
    st = _compile(call_text(fn, params, fields)).stream()
    for i in range(n):
        res = st.update({f: float(fdata[f][i]) for f in fdata})
        if isinstance(res, dict):
            for o in outs:
                rec[o][i] = res[o]
        else:
            rec[outs[0]][i] = res
    return rec


def warmup(fn, params=None, columns=None):
    fields = _bind(list(columns), {c: None for c in columns})[0] if columns is not None else None
    return int(_compile(call_text(fn, params, fields)).warmup)


def canon_bits(a):
    """float64 array -> uint64 bits with every NaN made one NaN, for bit-identity checks."""
    a = np.asarray(a, dtype=np.float64).copy()
    a[np.isnan(a)] = np.nan
    return a.view(np.uint64)
