"""Cut the stationarity and cointegration replays for the Rust tests out of the committed
fixtures.

The Rust crate has no gzip dependency, so `tests/stats_stationarity.rs` and
`tests/stats_coint.rs` replay these plain-JSON extracts instead of
`stats_stationarity.json.gz` and `stats_coint.json.gz`. Floats stay in the fixtures' own
encoding (`{"f8": hex of big-endian IEEE bits, "shape": [...]}`), so nothing is rounded. What the
tests do not read is left out: the plain-JSON copy of the reference's answer and the matrices
(`resid`, `r0t`, `rkt`) that dominate the size.

    python3 tests/oracle/extract_stats_stationarity_coint.py
"""

import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "python", "oracle", "fixtures", "stats")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rt") as f:
        return json.load(f)


def dump(name, obj):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    size = os.path.getsize(path)
    assert size < 500_000, (name, size)
    print(name, size)


def without(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def stationarity():
    src = load("stats_stationarity.json.gz")
    cases = []
    for c in src["cases"]:
        rec = {"name": c["name"], "kind": c["kind"], "input": c["input"]}
        if "reference_full" in c:
            rec["reference"] = c["reference_full"]
        for key in ("lib", "note"):
            if key in c:
                rec[key] = c[key]
        cases.append(rec)
    dump(
        "stats_stationarity_extract.json",
        {"source": "stats_stationarity.json.gz", "cases": cases, "tables": src["tables"]},
    )


def coint():
    src = load("stats_coint.json.gz")
    cases = []
    for c in src["cases"]:
        rec = {"name": c["name"], "kind": c["kind"], "input": c["input"]}
        rec["reference"] = c["reference_full"]
        if c.get("lib"):
            lib = dict(c["lib"])
            lib["ols"] = without(lib["ols"], "resid")
            lib["johansen"] = [without(j, "r0t", "rkt") for j in lib["johansen"]]
            rec["lib"] = lib
        for key in ("known", "note"):
            if key in c:
                rec[key] = c[key]
        cases.append(rec)
    dump("stats_coint_extract.json", {"source": "stats_coint.json.gz", "cases": cases})


if __name__ == "__main__":
    stationarity()
    coint()
