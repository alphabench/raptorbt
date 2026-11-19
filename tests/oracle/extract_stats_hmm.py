"""Cut the Markov-regime replay for the Rust test out of the committed fixture.

The Rust crate has no gzip dependency, so `tests/stats_hmm.rs` replays this plain-JSON extract of
`tests/python/oracle/fixtures/stats/stats_hmm.json.gz`: each case's returns (the fixture's own
encoding, the hex of each f64's big-endian bits, nothing rounded; a series shared by several cases
is stored once), the arguments, the reference's answer at full precision (or the error it
raised), and statsmodels' numbers for the fit on returns in percent. The probabilities are cut to
their last TAIL bars.

    python3 tests/oracle/extract_stats_hmm.py
"""

import gzip
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "python", "oracle", "fixtures", "stats", "stats_hmm.json.gz")
TAIL = 120
LIB_KEYS = ("param_names", "params", "llf", "converged", "regime_transition")


def tail_rows(entry, rows):
    """The last `rows` rows of an {"f8": hex, "shape": [n, k]} matrix, same encoding."""
    n, k = entry["shape"]
    hexs = entry["f8"]
    return {"f8": hexs[16 * k * (n - rows) :], "shape": [rows, k]}


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
        if "ok" in full:
            out["reference"] = full["ok"]
        lib = case.get("lib")
        if lib is not None and "error" not in lib:
            out["lib"] = {k: lib[k] for k in LIB_KEYS}
            for key in ("smoothed_marginal_probabilities", "filtered_marginal_probabilities"):
                out["lib"][key + "_tail"] = tail_rows(lib[key], TAIL)
        cases.append(out)
    doc = {
        "source": "stats_hmm.json.gz",
        "libraries": src["header"]["libraries"],
        "tail_rows": TAIL,
        "series": series,
        "cases": cases,
    }
    path = os.path.join(HERE, "stats_hmm_extract.json")
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    print(path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
