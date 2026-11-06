"""Adversarial and conformance tests, generic over the function registry. A new registered
function is covered by being registered: nothing here names a function.

Registry fields read (all declared by the function's row; a missing one fails the test that
needs it, with the field's name, rather than being skipped):
    causal         bool; False is allowed only for names in noncausal.json
    nan_policy     "elementwise" | "lagged" | "delayed" | "windowed" | "recursive" | "cross_section"
                   cross_section: a cross-sectional function, run over a series folded into a
                   panel of ROW names per date (see harness): a NaN spoils its own bar and
                   changes the other bars of its date only
                   lagged: the lag is the warm-up; a NaN at bar i spoils bars i and i + warm-up only
                   delayed: the output is the input `warm-up` bars back; a NaN at bar i appears at
                   bar i + warm-up only
    invariance     list of: scale_equivariant, scale_invariant, shift_equivariant, shift_invariant
    domain         "positive" for functions that take a log or power, "unit" for those defined
                   on [-1, 1] (optional)
"""

import json
import os

import numpy as np
import pytest

import replay
from harness import (
    canon_bits,
    evaluate,
    evaluate_stream,
    input_names,
    ROW,
    param_defaults,
    registry,
    row_start,
    warmup,
)

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "noncausal.json")) as _f:
    NONCAUSAL = set(json.load(_f))
LOG_FUNCTIONS = {"LN", "LOG10"}
N = 400
CUTS = 8

REG = registry()
IDS = [f["name"] for f in REG]


def _clean():
    s = replay.inputs()["random_walk"]
    return {k: replay.decode(v)[:N] for k, v in s.items()}


def _data(fn, base=None):
    d = dict(base or _clean())
    cols = {c: d.get(c, d["close"]).copy() for c in input_names(fn)}
    if fn.get("domain") == "unit":
        # Prices are far outside [-1, 1]; a unit-domain function is fed a series inside it.
        top = max(float(np.max(np.abs(v))) for v in cols.values())
        cols = {c: v / (1.1 * top) for c, v in cols.items()}
    return cols


def _period(fn, w):
    ints = [
        v for v in param_defaults(fn).values() if isinstance(v, int) and not isinstance(v, bool)
    ]
    return max(ints) if ints else max(w, 1)


def _run(fn, data):
    return evaluate(fn, data)


def _all(fn, data):
    return _run(fn, data).values()


def _finite_or_nan(arrs):
    for a in arrs:
        assert not np.isinf(a).any(), "infinity leaked into a series"


@pytest.mark.parametrize("fn", REG, ids=IDS)
def test_first_valid_equals_warmup(fn):
    data = _data(fn)
    w = warmup(fn)
    for name, out in _run(fn, data).items():
        assert replay.first_valid(out) == w, (
            f"{fn['name']}.{name}: first valid {replay.first_valid(out)}, warm-up {w}"
        )


@pytest.mark.parametrize("fn", REG, ids=IDS)
def test_lengths(fn):
    full = _data(fn)
    w = warmup(fn)
    p = _period(fn, w)
    for n in sorted({0, 1, max(p - 1, 0), p + 1, w, w + 1}):
        d = {k: v[:n].copy() for k, v in full.items()}
        try:
            outs = _run(fn, d)
        except Exception as e:  # a clean refusal of an empty series is not a panic
            if n == 0 and type(e).__name__ != "PanicException":
                continue
            raise
        for name, out in outs.items():
            assert len(out) == n, f"{name}: length {len(out)} for input length {n}"
            if n <= w:
                assert np.isnan(out).all(), f"{name}: values before warm-up at length {n}"


@pytest.mark.parametrize("fn", REG, ids=IDS)
def test_nan_head_middle_tail_and_all(fn):
    base = _data(fn)
    w = warmup(fn)
    p = _period(fn, w)
    ref = _run(fn, base)
    policy = fn.get("nan_policy")
    assert policy in (
        "elementwise", "lagged", "delayed", "windowed", "recursive", "cross_section"
    ), (
        f"{fn['name']}: registry has no usable nan_policy ({policy!r})"
    )
    for where, idx in (("head", 3), ("middle", N // 2), ("tail", N - 2)):
        d = {k: v.copy() for k, v in base.items()}
        for v in d.values():
            v[idx] = np.nan
        outs = _run(fn, d)
        for name, out in outs.items():
            _finite_or_nan([out])
            assert len(out) == N
            if idx > w and policy != "delayed":
                assert np.isnan(out[idx]), f"{name}: NaN input at {where} ({idx}) did not give NaN"
            if policy == "cross_section" and idx > w:
                # Only the NaN bar and the other names of its own date may change.
                keep = np.ones(N, bool)
                lo = row_start(idx)
                keep[lo : lo + ROW] = False
                assert np.array_equal(canon_bits(out[keep]), canon_bits(ref[name][keep])), (
                    f"{name}: a NaN reached a date other than its own"
                )
            if policy == "elementwise" and idx > w:
                keep = np.ones(N, bool)
                keep[idx] = False
                assert np.array_equal(canon_bits(out[keep]), canon_bits(ref[name][keep])), (
                    f"{name}: elementwise NaN leaked to other bars"
                )
            if policy == "delayed":
                # Bar i keeps its value (it reads an earlier bar); bar i + warm-up is the one
                # that shows the NaN.
                spoiled = [i for i in (idx + w,) if i < N]
                keep = np.ones(N, bool)
                keep[spoiled] = False
                assert np.isnan(out[spoiled]).all(), (
                    f"{name}: NaN at {where} ({idx}) did not appear at bars {spoiled}"
                )
                assert np.array_equal(canon_bits(out[keep]), canon_bits(ref[name][keep])), (
                    f"{name}: delayed NaN reached a bar other than {spoiled}"
                )
            if policy == "lagged":
                # Exactly the NaN bar and the bar `warm-up` later, nothing else: not an
                # exemption but a stricter rule than elementwise.
                spoiled = [i for i in (idx, idx + w) if i < N]
                keep = np.ones(N, bool)
                keep[spoiled] = False
                assert np.isnan(out[spoiled]).all(), (
                    f"{name}: NaN at {where} ({idx}) did not reach bars {spoiled}"
                )
                assert np.array_equal(canon_bits(out[keep]), canon_bits(ref[name][keep])), (
                    f"{name}: lagged NaN reached a bar other than {spoiled}"
                )
            if policy == "windowed" and where != "tail":
                assert not np.isnan(out[min(idx + p + w + 5, N - 1)]) or idx + p + w + 5 >= N, (
                    f"{name}: windowed output did not recover after the window"
                )
            if policy == "recursive" and where == "middle":
                # NaN without advancing state == the bar not being there.
                drop = {k: np.delete(v, idx) for k, v in base.items() if k in d}
                gone = _run(fn, drop)[name]
                assert np.array_equal(canon_bits(out[idx + 1 :]), canon_bits(gone[idx:])), (
                    f"{name}: recursive kernel advanced on NaN"
                )
    allnan = {k: np.full(N, np.nan) for k in base}
    for name, out in _run(fn, allnan).items():
        assert np.isnan(out).all(), f"{name}: all-NaN input gave values"


@pytest.mark.parametrize("fn", REG, ids=IDS)
def test_flat_inf_extremes_zero_volume(fn):
    base = _data(fn)
    cases = {
        "flat": {k: np.full(N, 100.0) for k in base},
        "plus_inf_middle": _with(base, N // 2, np.inf),
        "minus_inf_middle": _with(base, N // 2, -np.inf),
        "huge": {k: v * 1e300 / 100.0 for k, v in base.items()},
        "tiny": {k: v * 1e-300 / 100.0 for k, v in base.items()},
    }
    if "volume" in base:
        cases["zero_volume"] = {**base, "volume": np.zeros(N)}
    positive_only = fn.get("domain") == "positive" or fn.get("talib_name") in LOG_FUNCTIONS
    if positive_only:
        cases["zero"] = {k: np.zeros(N) for k in base}
        cases["negative"] = {k: -np.abs(v) for k, v in base.items()}
    if fn.get("domain") == "unit":
        cases["outside"] = {k: np.full(N, 100.0) for k in base}
    for label, d in cases.items():
        outs = _run(fn, d)  # must return, not panic
        for name, out in outs.items():
            assert len(out) == N, f"{label}: {name} length"
            if (
                label in ("flat", "zero_volume", "plus_inf_middle", "minus_inf_middle")
                or positive_only
            ):
                _finite_or_nan([out])
            if label in ("zero", "negative"):
                assert np.isnan(out[warmup(fn) :]).all(), (
                    f"{name}: non-positive input to a log/power function gave values ({label})"
                )
            if label == "outside":
                assert np.isnan(out[warmup(fn) :]).all(), (
                    f"{name}: input outside [-1, 1] to a unit-domain function gave values"
                )


def _with(base, idx, value):
    d = {k: v.copy() for k, v in base.items()}
    for v in d.values():
        v[idx] = value
    return d


@pytest.mark.parametrize("fn", REG, ids=IDS)
def test_batch_equals_streaming_bitwise(fn):
    data = _data(fn)
    batch = _run(fn, data)
    stream = evaluate_stream(fn, data)
    for name, out in batch.items():
        assert np.array_equal(canon_bits(out), canon_bits(stream[name])), (
            f"{name}: batch and streaming differ"
        )


@pytest.mark.parametrize("fn", REG, ids=IDS)
def test_prefix_identity_under_poisoned_future(fn):
    causal = fn.get("causal")
    assert isinstance(causal, bool), f"{fn['name']}: registry `causal` must be declared as a bool"
    if not causal:
        assert fn["name"] in NONCAUSAL, (
            f"{fn['name']} is non-causal but not listed in noncausal.json"
        )
        return
    assert fn["name"] not in NONCAUSAL, (
        f"{fn['name']} is listed in noncausal.json but declared causal"
    )
    base = _data(fn)
    ref = _run(fn, base)
    rng = np.random.default_rng(20261106)
    for cut in rng.integers(1, N - 1, size=CUTS):
        for poison in ("nan", "wild"):
            d = {k: v.copy() for k, v in base.items()}
            for v in d.values():
                v[cut + 1 :] = np.nan if poison == "nan" else rng.uniform(-1e6, 1e6, N - cut - 1)
            outs = _run(fn, d)
            for name, out in outs.items():
                # A cross-sectional date is whole or not at all: dates before the cut's date.
                end = row_start(cut) if fn.get("family") == "cross" else cut + 1
                assert np.array_equal(canon_bits(out[:end]), canon_bits(ref[name][:end])), (
                    f"{name}: prefix changed with {poison} future after bar {cut}"
                )


def test_noncausal_list_is_explicit():
    names = {f["name"] for f in REG}
    assert NONCAUSAL <= names, (
        f"noncausal.json names unknown functions: {sorted(NONCAUSAL - names)}"
    )


@pytest.mark.parametrize("fn", REG, ids=IDS)
def test_declared_invariance(fn):
    tokens = fn.get("invariance") or []
    base = _data(fn)
    ref = _run(fn, base)
    price_like = {k: v for k, v in base.items() if k != "volume"}
    for tok in tokens:
        if tok.startswith("scale"):
            d = {**base, **{k: v * 4.0 for k, v in price_like.items()}}
            want = (lambda r: r * 4.0) if tok == "scale_equivariant" else (lambda r: r)
        elif tok.startswith("shift"):
            d = {**base, **{k: v + 1000.0 for k, v in price_like.items()}}
            want = (lambda r: r + 1000.0) if tok == "shift_equivariant" else (lambda r: r)
        else:
            pytest.fail(f"{fn['name']}: unknown invariance token {tok!r}")
        outs = _run(fn, d)
        for name, out in outs.items():
            exp = want(ref[name])
            assert np.array_equal(np.isnan(out), np.isnan(exp)), (
                f"{name}: {tok} changed the NaN mask"
            )
            ok = ~np.isnan(exp)
            scale = max(1.0, float(np.max(np.abs(exp[ok]))) if ok.any() else 1.0)
            assert np.max(np.abs(out[ok] - exp[ok]), initial=0.0) <= 1e-9 * scale, (
                f"{name}: {tok} violated"
            )
