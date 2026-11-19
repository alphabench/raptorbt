"""Cut the replay cases of the statistics out of the committed fixtures for the Rust tests.

The Rust crate has no gzip dependency, so `tests/stats_<family>.rs` replays these plain-JSON
extracts instead of `tests/python/oracle/fixtures/stats/stats_<family>.json.gz`. Floats stay
the hex of their IEEE-754 big-endian bits ({"f8": hex} and "shape"), so nothing is rounded.
Per case: the arguments (`input`), what the engine must return at full precision (`expect`,
from the fixture's `reference_full`) or the refusal the reference raised (`error`, its message),
and the library numbers and exact answers the fixture holds (`lib`, `known`).

Each file stays under 500 KB; a family that does not fit is split into `..._extract.json`,
`..._extract_2.json`, ... Cases the tests build themselves (a 201st strategy, 41 series, 2001
cells) are left out.

    python3 tests/oracle/extract_stats.py
"""

import glob
import gzip
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "python", "oracle", "fixtures", "stats")
LIMIT = 450_000

# cases a test builds itself, or whose size is in the input only to reach a refusal
LEFT_OUT = {
    "pbo": {"two_hundred_strategies", "two_hundred_and_one"},
    "spa": {"two_hundred_strategies", "two_hundred_and_one"},
    "pca": {"forty_one_series"},
    "surface": {"too_many_cells"},
    "events": {"too_many_events"},
}
LIB_KEEP = {
    "pca": {"singular_values", "row_sums_raw"},
    "spa": None,  # handled by `spa_lib`
}


def load(family):
    with gzip.open(os.path.join(FIX, f"stats_{family}.json.gz"), "rt") as f:
        return json.load(f)


def f8s(entry):
    return np.frombuffer(bytes.fromhex(entry["f8"]), dtype=">f8")


def i4s(entry):
    return np.frombuffer(bytes.fromhex(entry["i4"]), dtype=">i4").astype(np.int64)


def spa_lib(lib):
    """arch's numbers: the bootstrap indices (u8 or u16 hex), the loss differential's mean and
    variance, the recentring flags, the p-values, and the largest simulated value per
    repetition for the lower, consistent and upper centring."""
    idx = i4s(lib["indices"])
    width = 1 if idx.max() < 256 else 2
    sim = f8s(lib["simulated_vals"]).reshape(lib["simulated_vals"]["shape"])
    mx = sim.max(axis=0)  # reps x 3
    return {
        "block": lib["block"],
        "reps": lib["reps"],
        "indices": {
            "width": width,
            "shape": lib["indices"]["shape"],
            "hex": idx.astype(f">u{width}").tobytes().hex(),
        },
        "loss_diff_mean": lib["loss_diff_mean"],
        "loss_diff_var": lib["loss_diff_var"],
        "valid_columns": [int(v) for v in i4s(lib["valid_columns"])],
        "p_lower": lib["p_lower"],
        "p_consistent": lib["p_consistent"],
        "p_upper": lib["p_upper"],
        "max_simulated": {"f8": mx.astype(">f8").tobytes().hex(), "shape": list(mx.shape)},
    }


def hac_se_exact(case):
    """The Newey-West standard error of the fixture's returns in exact rational arithmetic. With a
    mean of 1 and noise of 1e-9 the library's own residuals carry the rounding error of its mean
    (the fixture is 1e-8 off), so this, not the library, is what the engine is held to."""
    from fractions import Fraction as F

    r = f8s(case["input"]["args"][0])
    n, lags = len(r), case["lib"]["lags_used"]
    x = [F(float(v)) for v in r]
    m = sum(x) / n
    e = [v - m for v in x]
    s = sum(v * v for v in e)
    for j in range(1, lags + 1):
        s += 2 * (1 - F(j, lags + 1)) * sum(e[t] * e[t - j] for t in range(j, n))
    se = math.sqrt(s / (n * n))
    return {"f8": np.array([se], dtype=">f8").tobytes().hex()}


def pca_rank_aware_t(case):
    """The target regression's t-statistics for a rank-deficient panel, by ordinary least squares
    on the components that carry variance (the null component gets 0), recomputed here from the
    inputs alone: inverting the nearly singular X'X instead pollutes the t of every component
    (0.138 instead of 0.929 for the first in two_identical_with_target)."""
    cols = case["input"]["args"][0]
    m = np.column_stack([f8s(cols["columns"][k]) for k in cols["order"]])
    n, k = m.shape
    c = min(case["input"]["kwargs"].get("components", 3), 5, k)
    z = (m - m.mean(axis=0)) / m.std(axis=0, ddof=1)
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    y = f8s(case["input"]["kwargs"]["target"])
    keep = [j for j in range(c) if sv[j] > 1e-8 * sv[0]]
    x = np.column_stack([np.ones(n)] + [z @ vt[j] for j in keep])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    res = y - x @ beta
    se = np.sqrt(res @ res / (n - c - 1) * np.diag(np.linalg.inv(x.T @ x)))
    t = [0.0] * c
    for i, j in enumerate(keep):
        t[j] = abs(float(beta[i + 1] / se[i + 1]))
    enc = lambda a: {"f8": np.array(a, dtype=">f8").tobytes().hex(), "shape": [len(a)]}
    return {"alpha_t_abs": enc([abs(float(beta[0] / se[0]))]), "t_abs": enc(t)}


def slim(family, case):
    out = {"name": case["name"], "kind": case["kind"], "input": case["input"]}
    if case.get("note"):
        out["note"] = case["note"]
    full = case["reference_full"]
    if "ok" in full:
        out["expect"] = full["ok"]
    else:
        out["error"] = full["error"]["message"]
    if case.get("known"):
        out["known"] = case["known"]
    if family == "pca" and case["name"] == "two_identical_with_target":
        out["known"] = pca_rank_aware_t(case)
    if family == "hac" and case["name"] == "large_mean_tiny_noise":
        out["known"] = {"hac_se_exact": hac_se_exact(case)}
    lib = case.get("lib")
    if lib:
        if family == "spa":
            out["lib"] = spa_lib(lib)
        elif LIB_KEEP.get(family):
            out["lib"] = {k: v for k, v in lib.items() if k in LIB_KEEP[family]}
        else:
            out["lib"] = lib
    return out


def scale_sweep():
    """unique(floor(logspace(log10(16), log10(n // 4), 12) * (1 + 1e-12))), as numpy computes
    it, for many n: `stats_reference.hurst_scales` (not imported here, it needs scipy)."""
    out = {}
    for n in list(range(256, 1500)) + list(range(1500, 60001, 97)):
        top = n // 4
        v = np.logspace(math.log10(16), math.log10(top), 12)
        s = np.unique(np.floor(v * (1.0 + 1e-12)))
        out[str(n)] = [int(x) for x in s]
    return out


def dedupe_events(cases):
    """The event cases share their dates and series: store each distinct array once."""
    pool, ids = {}, {}

    def ref(v):
        key = json.dumps(v, sort_keys=True)
        if key not in ids:
            ids[key] = f"s{len(ids)}"
            pool[ids[key]] = v
        return ids[key]

    for c in cases:
        args = c["input"]["args"]
        c["input"] = {
            "dates": ref(args[0]),
            "asset": ref(args[1]),
            "market": ref(args[2]),
            "events": args[3],
            "kwargs": c["input"]["kwargs"],
        }
    return pool


def write(family, cases, extra=None):
    for stale in glob.glob(os.path.join(HERE, f"stats_{family}_extract*.json")):
        os.remove(stale)
    chunks, cur, size = [], [], len(json.dumps(extra, separators=(",", ":"))) if extra else 0
    for c in cases:
        n = len(json.dumps(c, separators=(",", ":")))
        if cur and size + n > LIMIT:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += n
    chunks.append(cur)
    for i, chunk in enumerate(chunks):
        name = f"stats_{family}_extract.json" if i == 0 else f"stats_{family}_extract_{i + 1}.json"
        doc = {"source": f"stats_{family}.json.gz", "cases": chunk}
        if i == 0 and extra:
            doc.update(extra)
        text = json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n"
        with open(os.path.join(HERE, name), "w") as f:
            f.write(text)
        print(f"{name}: {len(chunk)} cases, {len(text) // 1000} KB")


def main():
    for family in ("hac", "pbo", "spa", "pca", "hurst", "events", "impact", "surface", "quality"):
        d = load(family)
        cases = [
            slim(family, c)
            for c in d["cases"]
            if c["name"] not in LEFT_OUT.get(family, ())
        ]
        extra = None
        if family == "hurst":
            extra = {"tables": d["tables"], "scale_sweep": scale_sweep()}
        if family == "events":
            extra = {"series": dedupe_events(cases), "tables": d["tables"]}
        write(family, cases, extra)


if __name__ == "__main__":
    main()
