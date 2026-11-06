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

    return [expressions.describe(r["name"]) for r in expressions.functions()]


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


def evaluate(fn, data, params=None, columns=None):
    """Batch evaluation. `data` maps column name to a float64 array; `columns` names the
    columns the inputs read, in order (default: the fields). Returns {output: array}."""
    cols = list(columns) if columns is not None else input_names(fn)
    if columns is not None and any(p["name"] == "compat" for p in fn.get("params") or []):
        # Only replay names `columns`: a TA-Lib fixture is matched by the talib path of a name
        # that has a compat parameter. An explicit `compat` in `params` wins.
        params = {"compat": "talib", **(params or {})}
    fields, fdata = _bind(cols, data)
    return _as_dict(_compile(call_text(fn, params, fields)).eval(fdata), fn)


def evaluate_stream(fn, data, params=None, columns=None):
    cols = list(columns) if columns is not None else input_names(fn)
    fields, fdata = _bind(cols, data)
    n = len(next(iter(fdata.values())))
    outs = output_names(fn)
    rec = {o: np.full(n, np.nan) for o in outs}
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
