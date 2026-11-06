"""Replay committed TA-Lib fixtures through the expression engine. Needs numpy only.

Relations (the registry's `relation` for the function):
    exact               relative error <= 1e-10 where both are defined, identical NaN mask
    converges           NaN masks agree from the later first-valid on; difference <= 1e-8
                        after K bars, K derived from the decay; bars before K are pinned
    defined_difference  the two sides are defined differently (first-valid index, seed,
                        convention); the per-case difference is pinned in pinned.json and
                        any change fails
    candle              integers exactly equal

`python tests/python/oracle/replay.py --pin` rewrites pinned.json from the current engine
(review the diff: a pin is a statement that the difference is understood).
"""

import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
PINNED = os.path.join(HERE, "pinned.json")
FUNCTIONS_JSON = os.path.normpath(os.path.join(HERE, "..", "..", "oracle", "talib_functions.json"))

EXACT_REL = 1e-10
CONVERGE_ABS = 1e-8
CONVERGE_FLOOR = 1e-9  # K is where a seed difference of order 1 (relative) decays below this
SCALE_FLOOR = 1e-12  # relative error is taken against max(|ref|, SCALE_FLOOR * max|series|)
RELATIONS = ("exact", "converges", "defined_difference", "candle")

_cache = {}


def _load(path):
    if path not in _cache:
        with gzip.open(path, "rt") as f:
            _cache[path] = json.load(f)
    return _cache[path]


def decode(entry):
    if isinstance(entry, str):
        return np.frombuffer(bytes.fromhex(entry), dtype=">f8").astype(np.float64)
    dt = ">f8" if entry["dtype"] == "f8" else ">i4"
    a = np.frombuffer(bytes.fromhex(entry["data"]), dtype=dt)
    return a.astype(np.float64) if entry["dtype"] == "f8" else a.astype(np.int64)


def functions_table():
    with open(FUNCTIONS_JSON) as f:
        return json.load(f)["functions"]


def group_slug(group):
    return group.lower().replace(" ", "_")


def inputs():
    return _load(os.path.join(FIXTURES, "inputs.json.gz"))["series"]


def group_fixture(talib_name, changed=False):
    table = functions_table()
    if changed:
        path = os.path.join(FIXTURES, "talib_candles_changed.json.gz")
    else:
        path = os.path.join(FIXTURES, f"talib_{group_slug(table[talib_name]['group'])}.json.gz")
    if not os.path.exists(path):
        return None
    return _load(path)


def function_fixture(talib_name):
    g = group_fixture(talib_name)
    return None if g is None else g["functions"].get(talib_name)


def columns(scenario, names):
    s = inputs()[scenario]
    return {n: decode(s[n]) for n in set(names)}


def reference_array(out, lookback, relation):
    """TA-Lib's output as float, NaN before its lookback. Integer outputs are padded with 0
    by the Python wrapper; only the candle relation keeps them as integers."""
    a = decode(out)
    if out["dtype"] == "i4" and relation != "candle":
        a = a.astype(np.float64)
        a[:lookback] = np.nan
    return a


def first_valid(a):
    idx = np.flatnonzero(~np.isnan(a))
    return int(idx[0]) if len(idx) else len(a)


def hexf(x):
    return float(x).hex()


# ---------------------------------------------------------------------------- decay


def _period_of(params):
    ps = [v for k, v in params.items() if "period" in k.lower() and isinstance(v, (int, float))]
    return max(ps) if ps else None


def decay_factor(fn, params):
    d = fn.get("decay")
    if isinstance(d, (int, float)):
        return float(d)
    p = _period_of(params)
    if d == "ema" and p:
        return 1.0 - 2.0 / (p + 1.0)
    if d == "wilder" and p:
        return 1.0 - 1.0 / p
    raise AssertionError(
        f"{fn['name']}: relation 'converges' needs a numeric `decay` or 'ema'/'wilder' with a period parameter, got {d!r}"
    )


def convergence_bars(fn, params):
    rho = decay_factor(fn, params)
    if not 0.0 < rho < 1.0:
        raise AssertionError(f"{fn['name']}: decay {rho} outside (0, 1)")
    return int(math.ceil(math.log(CONVERGE_FLOOR) / math.log(rho)))


# --------------------------------------------------------------------- comparisons


def _rel_err(ours, ref, both):
    scale = max(float(np.max(np.abs(ref[both]))) if both.any() else 0.0, 0.0) * SCALE_FLOOR
    den = np.maximum(np.abs(ref[both]), scale if scale > 0 else 1e-300)
    return np.abs(ours[both] - ref[both]) / den


def check_exact(ours, ref):
    problems = []
    if len(ours) != len(ref):
        return [f"length {len(ours)} != {len(ref)}"]
    nm = np.isnan(ours) != np.isnan(ref)
    if nm.any():
        i = int(np.flatnonzero(nm)[0])
        problems.append(
            f"NaN mask differs at {int(nm.sum())} bars, first {i} (ours first valid {first_valid(ours)}, TA-Lib {first_valid(ref)})"
        )
    both = ~np.isnan(ours) & ~np.isnan(ref)
    if both.any():
        err = _rel_err(ours, ref, both)
        if err.max() > EXACT_REL:
            j = int(np.flatnonzero(both)[int(err.argmax())])
            problems.append(
                f"relative error {err.max():.3e} > {EXACT_REL:g} at bar {j} (ours {ours[j]!r}, TA-Lib {ref[j]!r})"
            )
    return problems


def check_converges(ours, ref, k):
    """Returns (problems, record). Record = the pinned early-bar difference."""
    problems = []
    if len(ours) != len(ref):
        return [f"length {len(ours)} != {len(ref)}"], None
    f = max(first_valid(ours), first_valid(ref))
    late = f + k
    nm = np.isnan(ours[f:]) != np.isnan(ref[f:])
    if nm.any():
        problems.append(
            f"NaN mask differs after both are valid ({int(nm.sum())} bars, first {f + int(np.flatnonzero(nm)[0])})"
        )
    both = ~np.isnan(ours) & ~np.isnan(ref)
    idx = np.arange(len(ours))
    lm = both & (idx >= late)
    if lm.any():
        d = np.abs(ours[lm] - ref[lm]) / np.maximum(1.0, np.abs(ref[lm]))
        if d.max() > CONVERGE_ABS:
            problems.append(
                f"difference {d.max():.3e} > {CONVERGE_ABS:g} after K={k} bars (from bar {late})"
            )
    em = both & (idx < late)
    early = float(np.max(np.abs(ours[em] - ref[em]))) if em.any() else 0.0
    rec = {
        "first_valid_ours": first_valid(ours),
        "first_valid_ref": first_valid(ref),
        "k": k,
        "early_max_abs": hexf(early),
    }
    return problems, rec


def difference_record(ours, ref):
    both = ~np.isnan(ours) & ~np.isnan(ref)
    d = np.abs(ours[both] - ref[both])
    return {
        "first_valid_ours": first_valid(ours),
        "first_valid_ref": first_valid(ref),
        "n_compared": int(both.sum()),
        "max_abs": hexf(d.max()) if len(d) else hexf(0.0),
    }


def check_candle(ours, ref, lookback):
    if len(ours) != len(ref):
        return [f"length {len(ours)} != {len(ref)}"]
    o = ours.copy()
    head = np.isnan(o)
    head[lookback:] = False
    o[head] = 0.0
    bad = np.isnan(o)
    if bad.any():
        return [f"NaN after warm-up at bar {int(np.flatnonzero(bad)[0])} (lookback {lookback})"]
    if not np.array_equal(o, np.round(o)):
        return ["non-integer candle output"]
    ne = o.astype(np.int64) != ref.astype(np.int64)
    if ne.any():
        i = int(np.flatnonzero(ne)[0])
        return [f"{int(ne.sum())} bars differ, first {i}: ours {o[i]:.0f}, TA-Lib {ref[i]}"]
    return []


# ----------------------------------------------------------------------- pin store


def load_pins():
    if not os.path.exists(PINNED):
        return {}
    with open(PINNED) as f:
        return json.load(f)


def pin_key(talib_name, case, oname):
    return f"{talib_name}|{case['scenario']}|{case['label']}|{oname}"


def save_pins(pins):
    """Replace the pin file in one step: write a sibling, then rename it over."""
    tmp = PINNED + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pins, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, PINNED)


# A function whose first valid bar may differ from TA-Lib's outside a compat `native` path, by
# TA-Lib name, each with the reason. Empty: a different first valid bar is a wiring error until
# someone declares otherwise here.
FIRST_VALID_MAY_DIFFER = {}


def refuse_blessed(pins):
    """Pins that would record a wiring error: first valid differs from the reference for a
    function not declared in FIRST_VALID_MAY_DIFFER and not a compat `native` record."""
    bad = []
    for k, rec in sorted(pins.items()):
        name = k.split("|")[0]
        if k.endswith("|native") or name in FIRST_VALID_MAY_DIFFER:
            continue
        if rec["first_valid_ours"] != rec["first_valid_ref"]:
            bad.append(
                f"{k}: first valid {rec['first_valid_ours']} vs TA-Lib {rec['first_valid_ref']}"
            )
    return bad


def summarise(pins):
    by = {}
    for k, rec in pins.items():
        name = k.split("|")[0] + ("|native" if k.endswith("|native") else "")
        m = float.fromhex(rec.get("max_abs", rec.get("early_max_abs", "0x0.0p+0")))
        cur = by.setdefault(name, [0.0, False])
        cur[0] = max(cur[0], m)
        cur[1] |= rec["first_valid_ours"] != rec["first_valid_ref"]
    for name in sorted(by):
        m, d = by[name]
        print(f"  {name:24s} max_abs {m:.6g}  first_valid {'DIFFERS' if d else 'same'}")


# ------------------------------------------------------------------------- replay


def has_compat(fn):
    return any(p["name"] == "compat" for p in fn.get("params") or [])


def replay_case(fn, talib_name, entry, case, pins, collect=None):
    """Run one fixture case through the engine and judge it by fn['relation'].
    Returns a list of problems. With `collect` (a dict), pinned records are stored there
    instead of being compared.

    A function with a `compat` parameter is judged twice: its "talib" path must match TA-Lib
    by the Exact rule whatever its declared relation, and its "native" path is pinned as a
    defined difference from TA-Lib (key suffix `|native`)."""
    if not has_compat(fn):
        return _replay(fn, talib_name, entry, case, pins, collect, fn.get("relation"), None, "")
    problems = [
        f"compat=talib: {p}"
        for p in _replay(fn, talib_name, entry, case, pins, collect, "exact", "talib", "")
    ]
    problems += [
        f"compat=native: {p}"
        for p in _replay(
            fn, talib_name, entry, case, pins, collect, "defined_difference", "native", "|native"
        )
    ]
    return problems


def _replay(fn, talib_name, entry, case, pins, collect, relation, compat, suffix):
    from harness import evaluate, output_names

    if relation not in RELATIONS:
        return [f"{fn['name']}: registry relation {relation!r} is not one of {RELATIONS}"]
    cols = columns(case["scenario"], entry["inputs"])
    params = {k: v for k, v in case["params"].items()}
    reg_params = _registry_params(fn, talib_name, params)
    if compat is not None:
        reg_params["compat"] = compat
    ours_all = evaluate(fn, cols, params=reg_params, columns=entry["inputs"])
    problems = []
    outs = output_names(fn)
    for oname, out in case["out"].items():
        # Registry names single outputs freely; match by position when names differ.
        key = (
            oname if oname in ours_all else outs[list(case["out"]).index(oname)] if outs else oname
        )
        if key not in ours_all:
            problems.append(f"{oname}: engine has no such output (has {sorted(ours_all)})")
            continue
        ours = ours_all[key]
        ref = reference_array(out, case["lookback"], relation)
        pk = pin_key(talib_name, case, oname) + suffix
        if relation == "exact":
            problems += [f"{oname}: {p}" for p in check_exact(ours, ref)]
        elif relation == "candle":
            problems += [f"{oname}: {p}" for p in check_candle(ours, ref, case["lookback"])]
        elif relation == "converges":
            k = convergence_bars(fn, params)
            p, rec = check_converges(ours, ref, k)
            problems += [f"{oname}: {x}" for x in p]
            problems += _judge_pin(pk, rec, pins, collect)
        else:
            problems += _judge_pin(pk, difference_record(ours, ref), pins, collect)
    return problems


def _judge_pin(pk, rec, pins, collect):
    if rec is None:
        return []
    if collect is not None:
        collect[pk] = rec
        return []
    if pk not in pins:
        return [f"no pinned record for {pk} (run replay.py --pin and review)"]
    if pins[pk] != rec:
        return [f"pinned difference changed for {pk}: pinned {pins[pk]}, now {rec}"]
    return []


# TA-Lib parameter names the registry spells differently; any other name must match exactly.
TALIB_ALIASES = {
    "timeperiod": ("period",),
    "fastperiod": ("fast",),
    "slowperiod": ("slow",),
    "signalperiod": ("signal",),
}


# TA-Lib parameters a function accepts and ignores, so the registry has no counterpart. Each
# claim is measured, and test_talib_replay.py asserts it on the fixtures.
TALIB_IGNORED = {
    # VAR with nbdev 1.0 and 2.5 gives identical output (fixture cases `default` and `nbdev_2_5`).
    "VAR": {"nbdev"},
}


def _registry_params(fn, talib_name, params):
    """TA-Lib parameter name -> registry parameter name, bound by name (never by position).
    A registry param may carry `talib_param`; otherwise the name is the same or one of
    TALIB_ALIASES. A fixture parameter with no registered counterpart is an error: the
    engine would otherwise silently run on its defaults."""
    reg = {p["name"] for p in (fn.get("params") or []) if isinstance(p, dict)}
    by_talib = {
        p["talib_param"]: p["name"]
        for p in (fn.get("params") or [])
        if isinstance(p, dict) and p.get("talib_param")
    }
    out = {}
    for k, v in params.items():
        if k in TALIB_IGNORED.get(talib_name, ()):
            continue
        name = by_talib.get(k) or next((c for c in (k, *TALIB_ALIASES.get(k, ())) if c in reg), None)
        if name is None:
            raise AssertionError(
                f"{fn['name']} ({talib_name}): fixture parameter {k!r} has no registered "
                f"counterpart (registered: {sorted(reg)}; add an alias to TALIB_ALIASES only "
                "if it is the same parameter)"
            )
        if name in out:
            raise AssertionError(f"{fn['name']}: two fixture parameters bind to {name!r}")
        out[name] = v
    return out


def registered_with_reference():
    from harness import registry, talib_names

    for fn in registry():
        for t in talib_names(fn):
            yield fn, t


def main(argv):
    if "--pin" not in argv:
        print(__doc__)
        return 2
    sys.path.insert(0, HERE)
    pins, hard = {}, []
    for fn, t in registered_with_reference():
        if fn.get("relation") not in ("converges", "defined_difference") and not has_compat(fn):
            continue
        entry = function_fixture(t)
        for case in entry["cases"]:
            problems = replay_case(fn, t, entry, case, {}, collect=pins)
            hard += [f"{t} {case['scenario']} {case['label']}: {p}" for p in problems]
    hard += refuse_blessed(pins)
    summarise(pins)
    if hard:
        print("refusing to pin; pinned.json is unchanged:")
        print("\n".join(hard))
        return 1
    save_pins(pins)
    print(f"pinned {len(pins)} records -> {PINNED}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
