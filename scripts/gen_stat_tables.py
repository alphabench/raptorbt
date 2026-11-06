"""Generate src/stats/tables.rs: the MacKinnon (1994, 2010) and Johansen tables, as statsmodels
stores them.

Oracle environment only (it imports statsmodels; nothing at test or run time does):

    uv run --no-project --python 3.12 --with statsmodels==0.15.0 --with numpy==2.5.3 \
        --with scipy==1.18.1 --with pandas==3.1.0 python scripts/gen_stat_tables.py [--check]

`--check` regenerates in memory and fails if src/stats/tables.rs differs (no write).

Where statsmodels keeps them:
- statsmodels/tsa/adfvalues.py: the MacKinnon (1994) p-value approximations for the tau
  statistic, per regression ("n", "c", "ct", "ctt") and N = 1..6 series (`_tau_maxs`, `_tau_mins`,
  `_tau_stars`, `_tau_smallps`, `_tau_largeps`: the small-p and large-p tables are stored already
  multiplied by `small_scaling` / `large_scaling`, and are copied after that multiplication), and
  the MacKinnon (2010) response surfaces for critical values (`tau_2010s`: regression ->
  array [N - 1][1%, 5%, 10%][beta_inf, beta_1, beta_2, beta_3]), N = 1..12; the "n" (no constant)
  surface has N = 1 only.
- statsmodels/tsa/coint_tables.py: the Johansen critical values (`ejcp0..2`, maximum eigenvalue,
  and `tjcp0..2`, trace), rows n = 1..12, columns 90%, 95%, 99%, for det_order -1, 0, 1.
- statsmodels/tsa/stattools: the KPSS critical values are literals inside `kpss`; they are read
  here by calling `kpss` and taking the critical values it returns.

The Rust file carries `constants()`, every constant flattened in a fixed order; the test
tests/stats_tables.rs hashes it (sha256 of "name\n" then the 16 hex digits of each value's IEEE
bits and "\n") and compares with the digest this script prints. A changed table is then a visible
change in a diff, never a silent one.
"""

import hashlib
import math
import os
import shutil
import struct
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import statsmodels
from statsmodels.tsa import adfvalues, coint_tables
from statsmodels.tsa.stattools import kpss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "src", "stats", "tables.rs")

EXPECTED_STATSMODELS = "0.15.0"
REGRESSIONS = ("n", "c", "ct", "ctt")  # index order of every [4]-array below
DET_ORDERS = (-1, 0, 1)  # index order of the Johansen tables


def rs_float(v):
    v = float(v)
    if math.isinf(v):
        return "f64::INFINITY" if v > 0 else "f64::NEG_INFINITY"
    if math.isnan(v):
        raise SystemExit("a NaN in a table")
    return repr(v)


def rs_array(a):
    if isinstance(a, (list, tuple, np.ndarray)):
        return "[" + ", ".join(rs_array(x) for x in a) + "]"
    return rs_float(a)


def shape_of(a):
    return list(np.asarray(a, dtype=float).shape)


def rs_type(shape):
    t = "f64"
    for d in reversed(shape):
        t = f"[{t}; {d}]"
    return t


def collect():
    if statsmodels.__version__ != EXPECTED_STATSMODELS:
        raise SystemExit(f"statsmodels {statsmodels.__version__}, expected {EXPECTED_STATSMODELS}")
    out = []  # (name, doc, array)

    def add(name, doc, arr):
        out.append((name, doc, np.asarray(arr, dtype=float)))

    order = lambda d: [np.asarray(d[r], dtype=float) for r in REGRESSIONS]  # noqa: E731
    add(
        "TAU_MAX",
        "Above this tau statistic the p-value is 1. [regression][N - 1]",
        order(adfvalues._tau_maxs),
    )
    add(
        "TAU_MIN",
        "Below this tau statistic the p-value is 0. [regression][N - 1]",
        order(adfvalues._tau_mins),
    )
    add(
        "TAU_STAR",
        "At or below this the small-p polynomial applies, above it the large-p one. [regression][N - 1]",
        order(adfvalues._tau_stars),
    )
    add(
        "TAU_SMALLP",
        "p = normal_cdf(c0 + c1 tau + c2 tau^2) for tau <= TAU_STAR (coefficients rising; "
        "MacKinnon 1994 Table 5, after statsmodels' scaling). [regression][N - 1][coefficient]",
        order(adfvalues._tau_smallps),
    )
    add(
        "TAU_LARGEP",
        "p = normal_cdf(c0 + c1 tau + c2 tau^2 + c3 tau^3) for tau > TAU_STAR (coefficients rising; "
        "MacKinnon 1994 Table 6, after statsmodels' scaling). [regression][N - 1][coefficient]",
        order(adfvalues._tau_largeps),
    )
    for reg, name in (
        ("n", "TAU_2010_N"),
        ("c", "TAU_2010_C"),
        ("ct", "TAU_2010_CT"),
        ("ctt", "TAU_2010_CTT"),
    ):
        add(
            name,
            f"MacKinnon (2010) critical-value surface, regression {reg!r}: "
            "[N - 1][1%, 5%, 10%][beta_inf, beta_1, beta_2, beta_3]; the value at sample size T is "
            "beta_inf + beta_1/T + beta_2/T^2 + beta_3/T^3.",
            adfvalues.tau_2010s[reg],
        )
    # KPSS critical values come out of the function, for both regressions
    rng = np.random.default_rng(0)
    probe = rng.normal(size=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crit = {
            r: kpss(probe, regression=r, nlags=2, result_object=True).critical_values
            for r in ("c", "ct")
        }
    labels = ["10%", "5%", "2.5%", "1%"]
    add(
        "KPSS_PVALS",
        "p-values at the KPSS critical values (10%, 5%, 2.5%, 1%).",
        [0.10, 0.05, 0.025, 0.01],
    )
    add(
        "KPSS_CRIT_C",
        "KPSS critical values, level stationarity ('c'), at 10%, 5%, 2.5%, 1%.",
        [crit["c"][k] for k in labels],
    )
    add(
        "KPSS_CRIT_CT",
        "KPSS critical values, trend stationarity ('ct'), at 10%, 5%, 2.5%, 1%.",
        [crit["ct"][k] for k in labels],
    )
    add(
        "JOHANSEN_MAX_EIG",
        "Johansen maximum-eigenvalue critical values. [det_order -1, 0, 1][n - 1][90%, 95%, 99%]",
        [coint_tables.ejcp0, coint_tables.ejcp1, coint_tables.ejcp2],
    )
    add(
        "JOHANSEN_TRACE",
        "Johansen trace critical values. [det_order -1, 0, 1][n - 1][90%, 95%, 99%]",
        [coint_tables.tjcp0, coint_tables.tjcp1, coint_tables.tjcp2],
    )
    return out


def verify(tables):
    by = {n: a for n, _, a in tables}
    # the known defect depends on these two extents
    n94 = by["TAU_SMALLP"].shape[1]
    n10 = {r: by[f"TAU_2010_{r.upper()}"].shape[0] for r in REGRESSIONS}
    assert n94 == 6, f"1994 p-value tables should stop at N=6, found {n94}"
    assert by["TAU_LARGEP"].shape[1] == 6 and by["TAU_MAX"].shape[1] == 6
    assert n10["c"] == n10["ct"] == n10["ctt"] == 12, n10
    assert n10["n"] == 1, n10
    # the defect itself: mackinnonp works to N=6 and raises IndexError from N=7
    adfvalues.mackinnonp(-3.0, regression="c", N=6)
    for n in (7, 12):
        try:
            adfvalues.mackinnonp(-3.0, regression="c", N=n)
        except IndexError:
            pass
        else:
            raise SystemExit(f"mackinnonp(N={n}) no longer raises IndexError: re-read the design")
    # the 2010 critical values answer for N=7..12
    for n in range(7, 13):
        adfvalues.mackinnoncrit(N=n, regression="c", nobs=100)
    for name, a in by.items():
        if not np.isfinite(a[np.isfinite(a)]).all():
            raise SystemExit(name)
    return n94, n10


def digest(tables):
    h = hashlib.sha256()
    for name, _, a in tables:
        h.update(f"{name}\n".encode())
        for v in a.reshape(-1):
            h.update(f"{struct.unpack('>Q', struct.pack('>d', float(v)))[0]:016x}\n".encode())
    return h.hexdigest()


def render(tables):
    lines = [
        "//! The statistics' critical-value and p-value tables.",
        "//!",
        "//! Generated by `scripts/gen_stat_tables.py` from statsmodels "
        f"{EXPECTED_STATSMODELS} (BSD-3-Clause,",
        "//! credited in NOTICE); do not edit. The numbers are",
        "//! MacKinnon (1994, 2010) and MacKinnon, Haug and Michelis (1999) for Johansen, and the KPSS",
        "//! (1992) critical values. The tests pin `constants()` by digest, so a changed value shows up",
        "//! in review.",
        "//!",
        "//! Regression order in every `[4]` axis is `n`, `c`, `ct`, `ctt` (`REGRESSIONS`). The 1994",
        '//! p-value tables stop at N = 6 series; the 2010 critical values reach N = 12 (the "n"',
        "//! surface, N = 1 only). Johansen's tables are indexed by `det_order` -1, 0, 1.",
        "",
        "/// Regression names, in the order of the first axis of the `TAU_*` tables.",
        'pub const REGRESSIONS: [&str; 4] = ["n", "c", "ct", "ctt"];',
        "/// The largest N the 1994 p-value tables answer for.",
        "pub const MACKINNON_1994_MAX_N: usize = 6;",
        '/// The largest N the 2010 critical values answer for ("c", "ct", "ctt").',
        "pub const MACKINNON_2010_MAX_N: usize = 12;",
        "/// The largest n the Johansen tables answer for.",
        "pub const JOHANSEN_MAX_N: usize = 12;",
        "",
    ]
    for name, doc, a in tables:
        lines.extend(textwrap.wrap(doc, 96, initial_indent="/// ", subsequent_indent="/// "))
        lines.append(f"pub const {name}: {rs_type(list(a.shape))} = {rs_array(a)};")
        lines.append("")
    lines.append(
        "/// Every constant above, flattened in declaration order: what the digest test hashes."
    )
    lines.append("pub fn constants() -> Vec<(&'static str, Vec<f64>)> {")
    lines.append("    vec![")
    for name, _, a in tables:
        if a.ndim == 1:
            lines.append(f'        ("{name}", {name}.to_vec()),')
        else:
            flat = ".flatten()" * (a.ndim - 1)
            lines.append(f'        ("{name}", {name}.iter(){flat}.copied().collect()),')
    lines.append("    ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def rustfmt(text):
    exe = shutil.which("rustfmt")
    if exe is None:
        sys.stderr.write("rustfmt not found: the file is not formatted\n")
        return text
    proc = subprocess.run(
        [
            exe,
            "--edition",
            "2021",
            "--config-path",
            os.path.join(ROOT, "rustfmt.toml"),
            "--emit",
            "stdout",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    # `--emit stdout` prefixes the file name on some versions
    if out.startswith("<stdin>"):
        out = out.split("\n", 2)[2]
    return out


def reparse(rust_text, tables):
    """Read the numbers back out of the Rust text and require them to be bit-equal."""
    import re

    for name, _, a in tables:
        m = re.search(rf"pub const {name}: [^=]+= (.*?);\n", rust_text, re.S)
        body = m.group(1).replace("f64::INFINITY", "inf").replace("f64::NEG_INFINITY", "-inf")
        nums = [float(x) for x in re.findall(r"-?(?:inf|\d[\d.e+\-]*)", body)]
        flat = [float(v) for v in a.reshape(-1)]
        if len(nums) != len(flat) or any(
            struct.pack(">d", x) != struct.pack(">d", y) for x, y in zip(nums, flat)
        ):
            raise SystemExit(f"{name} does not read back bit-equal")


def main():
    tables = collect()
    n94, n10 = verify(tables)
    text = rustfmt(render(tables))
    reparse(text, tables)
    d = digest(tables)
    if "--check" in sys.argv:
        with open(OUT) as f:
            if f.read() != text:
                raise SystemExit("src/stats/tables.rs is stale: rerun scripts/gen_stat_tables.py")
        print("tables.rs is current")
    else:
        with open(OUT, "w") as f:
            f.write(text)
        print(f"wrote {OUT}")
    print(
        f"1994 p-value tables: N = 1..{n94}; 2010 critical values: N = 1..{n10['c']} (n: {n10['n']})"
    )
    print(f"digest {d}")


if __name__ == "__main__":
    main()
