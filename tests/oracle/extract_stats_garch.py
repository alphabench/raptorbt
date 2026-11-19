"""Cut the GARCH replay for the Rust test out of the committed fixture.

The Rust crate has no gzip dependency, so `tests/stats_garch.rs` replays this plain-JSON extract
of `tests/python/oracle/fixtures/stats/stats_garch.json.gz`: each case's returns (the fixture's
own encoding, the hex of each f64's big-endian bits, nothing rounded; a series shared by several
cases is stored once), the arguments, the error the reference raised if it raised one, and
arch's numbers for the fit on returns in percent.

    python3 tests/oracle/extract_stats_garch.py
"""

import gzip
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "python", "oracle", "fixtures", "stats", "stats_garch.json.gz")
TAIL = 60  # bars of conditional volatility kept
LIB_KEYS = (
    "param_names",
    "params",
    "std_err",
    "loglikelihood",
    "aic",
    "convergence_flag",
    "forecast_variance_10",
    "param_cov_robust",
)


def tail_f8(entry, n):
    """The last n values of an {"f8": hex, "shape": [len]} array, same encoding."""
    hexs = entry["f8"]
    return {"f8": hexs[len(hexs) - 16 * n :], "shape": [n]}


def main():
    with gzip.open(FIXTURE, "rt") as f:
        src = json.load(f)
    series, cases = {}, []
    for case in src["cases"]:
        arr = case["input"]["args"][0]
        sid = "s" + hashlib.sha256(arr["f8"].encode()).hexdigest()[:12]
        series[sid] = arr
        full = case["reference_full"]
        out = {
            "name": case["name"],
            "kind": case["kind"],
            "series": sid,
            "kwargs": case["input"]["kwargs"],
            "reference_error": full.get("error"),
        }
        lib = case.get("lib")
        if lib is not None and "error" not in lib:
            out["lib"] = {k: lib[k] for k in LIB_KEYS}
            out["lib"]["conditional_volatility_tail"] = tail_f8(lib["conditional_volatility"], TAIL)
        cases.append(out)
    doc = {
        "source": "stats_garch.json.gz",
        "libraries": src["header"]["libraries"],
        "series": series,
        "cases": cases,
    }
    path = os.path.join(HERE, "stats_garch_extract.json")
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    print(path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
