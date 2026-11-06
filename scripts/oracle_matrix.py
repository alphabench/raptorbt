#!/usr/bin/env python3
"""TA-Lib coverage matrix: every TA-Lib function against the expression registry.

    python scripts/oracle_matrix.py --report   list what is missing, write the matrix, exit 0
    python scripts/oracle_matrix.py --check    the same, exit 1 on any gap

Gaps: a TA-Lib function with no registered counterpart (`talib_name`); a registered function
with no `reference` (`fixture` or `hand:<test>`); a relation that is missing, unknown, or has
no pinned record (`converges` and `defined_difference` pin one per fixture case and output).
The registry comes from raptorbt.expressions.functions(); where the engine is not built or
the module does not exist yet, --report treats the registry as empty and says so.
"""

import argparse
import gzip
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FUNCTIONS_JSON = os.path.join(ROOT, "tests", "oracle", "talib_functions.json")
FIXTURES = os.path.join(ROOT, "tests", "python", "oracle", "fixtures")
PINNED = os.path.join(ROOT, "tests", "python", "oracle", "pinned.json")
MATRIX = os.path.join(ROOT, "docs", "quantitative-correctness", "matrix.md")
RELATIONS = ("exact", "converges", "defined_difference", "candle")
PINNED_RELATIONS = ("converges", "defined_difference")


def load_registry():
    sys.path.insert(0, os.path.join(ROOT, "python"))
    try:
        from raptorbt.expressions import describe, functions
    except Exception as e:  # not built, or the registry slice has not landed
        return None, f"{type(e).__name__}: {e}"
    return [describe(r["name"]) for r in functions()], None


def talib_names(fn):
    t = fn.get("talib_name")
    return [] if not t else [t] if isinstance(t, str) else list(t)


def slug(group):
    return group.lower().replace(" ", "_")


_fx = {}


def fixture_cases(group, name):
    path = os.path.join(FIXTURES, f"talib_{slug(group)}.json.gz")
    if not os.path.exists(path):
        return None
    if path not in _fx:
        with gzip.open(path, "rt") as f:
            _fx[path] = json.load(f)["functions"]
    return _fx[path].get(name, {}).get("cases")


def pin_keys(group, name):
    cases = fixture_cases(group, name) or []
    return [f"{name}|{c['scenario']}|{c['label']}|{o}" for c in cases for o in c["out"]]


def analyse():
    with open(FUNCTIONS_JSON) as f:
        table = json.load(f)["functions"]
    reg, why = load_registry()
    pins = {}
    if os.path.exists(PINNED):
        with open(PINNED) as f:
            pins = json.load(f)
    by_talib = {}
    for fn in reg or []:
        for t in talib_names(fn):
            by_talib.setdefault(t, []).append(fn)

    rows, problems = [], []
    for name, info in table.items():
        fns = by_talib.get(name, [])
        if not fns:
            rows.append((name, info["group"], "", "", "", "no counterpart"))
            problems.append(f"{name}: no registered function names it as talib_name")
            continue
        for fn in fns:
            ref, rel = fn.get("reference"), fn.get("relation")
            status = []
            if not (
                ref == "fixture"
                or (isinstance(ref, str) and ref.startswith("hand:") and len(ref) > 5)
            ):
                status.append("no reference")
                problems.append(
                    f"{fn['name']} ({name}): reference {ref!r} is not 'fixture' or 'hand:<test>'"
                )
            elif ref == "fixture" and not fixture_cases(info["group"], name):
                status.append("no fixture")
                problems.append(
                    f"{fn['name']} ({name}): reference 'fixture' but no fixture cases exist"
                )
            if rel not in RELATIONS:
                status.append("no relation")
                problems.append(
                    f"{fn['name']} ({name}): relation {rel!r} is not one of {RELATIONS}"
                )
            elif rel in PINNED_RELATIONS:
                missing = [k for k in pin_keys(info["group"], name) if k not in pins]
                if missing:
                    status.append(f"unpinned x{len(missing)}")
                    problems.append(
                        f"{fn['name']} ({name}): relation {rel} has no pinned record for {len(missing)} case(s), e.g. {missing[0]}"
                    )
            rows.append(
                (name, info["group"], fn["name"], rel or "", ref or "", ", ".join(status) or "ok")
            )

    extra = [fn["name"] for fn in reg or [] if not talib_names(fn)]
    for fn in reg or []:
        if not talib_names(fn):
            ref = fn.get("reference")
            if not (ref == "fixture" or (isinstance(ref, str) and ref.startswith("hand:"))):
                problems.append(
                    f"{fn['name']}: reference {ref!r} is not 'fixture' or 'hand:<test>'"
                )
    return table, reg, why, rows, extra, problems


def write_matrix(table, reg, why, rows, extra, problems):
    ok = sum(1 for r in rows if r[5] == "ok")
    lines = [
        "# TA-Lib coverage matrix",
        "",
        "Generated by `scripts/oracle_matrix.py`; do not edit.",
        "",
        f"TA-Lib functions: {len(table)}. Counterparts with reference, relation and pins in order: {ok}. Open gaps: {len(problems)}.",
    ]
    if reg is None:
        lines += ["", f"The registry could not be read ({why}); every row below is a gap."]
    lines += [
        "",
        "| TA-Lib | Group | Registered as | Relation | Reference | Status |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    if extra:
        lines += [
            "",
            "Registered functions with no TA-Lib counterpart (checked on their own reference):",
            "",
        ]
        lines += [f"- {n}" for n in extra]
    os.makedirs(os.path.dirname(MATRIX), exist_ok=True)
    with open(MATRIX, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="exit 1 on any gap")
    g.add_argument("--report", action="store_true", help="list gaps, exit 0")
    args = ap.parse_args()
    table, reg, why, rows, extra, problems = analyse()
    write_matrix(table, reg, why, rows, extra, problems)
    if reg is None:
        print(f"registry unavailable: {why}")
    for p in problems:
        print(("MISSING " if args.report else "FAIL ") + p)
    print(
        f"{len(table)} TA-Lib functions, {len(problems)} gaps; matrix -> {os.path.relpath(MATRIX, ROOT)}"
    )
    return 1 if (args.check and (problems or reg is None)) else 0


if __name__ == "__main__":
    sys.exit(main())
