"""Generate the TA-Lib oracle fixtures.

Run (it needs TA-Lib and numpy, and nothing from raptorbt: the fixtures must not
depend on the engine they judge):

    uv run --no-project --python 3.12 --with ta-lib==0.6.4 --with numpy \
        python tests/oracle/generate.py --families talib [--only GROUP_SLUG ...]

and the statistics fixtures (statsmodels, arch and scipy, and the reference implementations in
stats_reference.py; see stats_fixtures.py):

    uv run --no-project --python 3.12 --with numpy==2.5.3 --with pandas==3.1.0 \
        --with scipy==1.18.1 --with statsmodels==0.15.0 --with arch==8.0.0 \
        python tests/oracle/generate.py --families stats

or `uv run --only-group oracle python tests/oracle/generate.py` for both (`--families all`,
the default). `--out DIR` writes somewhere other than tests/python/oracle/fixtures (the
statistics files go in its `stats/` subdirectory, which the TA-Lib replay tests do not scan).

Output, under tests/python/oracle/fixtures/:
    inputs.json.gz                    every input series, by scenario
    talib_<group-slug>.json.gz        one file per TA-Lib group
    talib_candles_changed.json.gz     the candlestick set with three changed settings
and tests/oracle/talib_functions.json (the function table).

Floats are stored as the hex of their IEEE-754 big-endian bits (16 hex digits a value,
whole arrays concatenated; NaN canonical 7ff8000000000000), so no decimal rounding can
touch them. Inputs come from a SplitMix64 written below, not from numpy's generator.

Families are independent: each TA-Lib group is one function in GROUP_GENERATORS, and a
later stage adds statsmodels/arch/scipy families the same way (a function that writes
its own fixture file with the same header).
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

try:  # the statistics fixtures need none of TA-Lib
    import talib
    import talib._ta_lib as _tl
    from talib import abstract
except ImportError:
    talib = _tl = abstract = None

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.normpath(os.path.join(HERE, "..", "python", "oracle", "fixtures"))
FUNCTIONS_JSON = os.path.join(HERE, "talib_functions.json")

GENERATOR_VERSION = 1
LENGTH = 500  # bars per scenario: past 200 so the long-period case has data to settle on
CANDLE_BARS = 4000
MIN_FIRING = 5  # firing bars per sign, per pattern
PERIOD_TARGETS = (14, 200)
FALLBACK_TARGET = 60  # for functions whose 200-period lookback outruns LENGTH

# TA-Lib 0.6.4 defaults (ta_global.c). The Python package has no getter, so these are
# written down here and verified in check_default_settings(): setting them explicitly
# must leave every pattern's output unchanged.
RANGE = {"RealBody": 0, "HighLow": 1, "Shadows": 2}
DEFAULT_CANDLE_SETTINGS = OrderedDict(
    [
        ("BodyLong", ("RealBody", 10, 1.0)),
        ("BodyVeryLong", ("RealBody", 10, 3.0)),
        ("BodyShort", ("RealBody", 10, 1.0)),
        ("BodyDoji", ("HighLow", 10, 0.1)),
        ("ShadowLong", ("RealBody", 0, 1.0)),
        ("ShadowVeryLong", ("RealBody", 0, 2.0)),
        ("ShadowShort", ("Shadows", 10, 1.0)),
        ("ShadowVeryShort", ("HighLow", 10, 0.1)),
        ("Near", ("HighLow", 5, 0.2)),
        ("Far", ("HighLow", 5, 0.6)),
        ("Equal", ("HighLow", 5, 0.05)),
    ]
)
CHANGED_CANDLE_SETTINGS = OrderedDict(
    [
        ("BodyLong", ("RealBody", 10, 1.5)),
        ("BodyDoji", ("HighLow", 5, 0.2)),
        ("Near", ("HighLow", 5, 0.4)),
    ]
)

# Functions whose domain is not the price level: they get the bounded "unit" series.
UNIT_INPUT = {"ACOS", "ASIN", "ATANH"}


# --------------------------------------------------------------------------- random


class SplitMix64:
    MASK = (1 << 64) - 1

    def __init__(self, seed):
        self.s = seed & self.MASK

    def next_u64(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & self.MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def normal(self):
        u1 = max(self.uniform(), 1e-300)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * self.uniform())

    def below(self, n):
        return self.next_u64() % n


def _ohlc_from_closes(rng, closes, opens, wick):
    highs, lows = [], []
    for o, c in zip(opens, closes):
        top, bot = max(o, c), min(o, c)
        highs.append(top * (1.0 + abs(wick * rng.normal())))
        lows.append(bot * (1.0 - abs(wick * rng.normal())))
    return highs, lows


def _volume(rng, n):
    return [float(int(100000.0 * math.exp(0.5 * rng.normal()))) + 1.0 for _ in range(n)]


def scenario_random_walk(seed, n):
    rng = SplitMix64(seed)
    closes, opens, prev = [], [], 100.0
    for _ in range(n):
        o = prev * (1.0 + 0.001 * rng.normal())
        c = prev * math.exp(0.01 * rng.normal())
        opens.append(o)
        closes.append(c)
        prev = c
    h, l = _ohlc_from_closes(rng, closes, opens, 0.004)
    return opens, h, l, closes, _volume(rng, n)


def scenario_trending(seed, n):
    rng = SplitMix64(seed)
    closes, opens, prev = [], [], 50.0
    for _ in range(n):
        o = prev * (1.0 + 0.001 * rng.normal())
        c = prev * math.exp(0.003 + 0.008 * rng.normal())
        opens.append(o)
        closes.append(c)
        prev = c
    h, l = _ohlc_from_closes(rng, closes, opens, 0.003)
    return opens, h, l, closes, _volume(rng, n)


def scenario_mean_reverting_gaps(seed, n):
    rng = SplitMix64(seed)
    closes, opens, prev = [], [], 100.0
    for _ in range(n):
        gap = 0.0
        if rng.uniform() < 0.07:
            gap = 0.03 * rng.normal()
        o = prev * (1.0 + gap)
        c = o + 0.15 * (100.0 - o) + 0.8 * rng.normal()
        opens.append(o)
        closes.append(c)
        prev = c
    h, l = _ohlc_from_closes(rng, closes, opens, 0.003)
    return opens, h, l, closes, _volume(rng, n)


def scenario_tight_range(seed, n):
    # Prices on a 0.01 tick inside a 0.2 band: ties, equal highs/lows and zero ranges.
    rng = SplitMix64(seed)
    closes, opens, highs, lows = [], [], [], []
    for _ in range(n):
        c = round(100.0 + 0.1 * math.tanh(rng.normal()), 2)
        o = round(100.0 + 0.1 * math.tanh(rng.normal()), 2)
        h = round(max(o, c) + abs(0.02 * rng.normal()), 2)
        l = round(min(o, c) - abs(0.02 * rng.normal()), 2)
        opens.append(o)
        closes.append(c)
        highs.append(h)
        lows.append(l)
    return opens, highs, lows, closes, _volume(rng, n)


SCENARIOS = OrderedDict(
    [
        ("random_walk", (scenario_random_walk, 0xA11CE001)),
        ("trending", (scenario_trending, 0xA11CE002)),
        ("mean_reverting_gaps", (scenario_mean_reverting_gaps, 0xA11CE003)),
        ("tight_range", (scenario_tight_range, 0xA11CE004)),
    ]
)


# ------------------------------------------------------------------- constructed candles


def _bar_kind(rng):
    """Weights lean to extremes: dojis, marubozu, hammers, long bodies, big gaps."""
    return rng.below(10)


def _build_bars(rng, n, start):
    """Candles built from kinds so that patterns that need a long body, a doji, a long
    shadow or a gap meet them often. Scale ~1.0 on a price near `start`."""
    o_l, h_l, l_l, c_l = [], [], [], []
    prev = start
    drift = 0.0
    for i in range(n):
        if i % 12 == 0:
            drift = rng.normal() * 0.4
        kind = _bar_kind(rng)
        gap = 0.0
        r = rng.uniform()
        if r < 0.2:
            gap = (1.0 + 2.0 * rng.uniform()) * (1 if rng.uniform() < 0.5 else -1)
        elif r < 0.35:
            gap = 0.0  # opens at the prior close
        else:
            gap = 0.15 * rng.normal()
        o = prev + gap + drift * 0.2
        up = 1.0 if rng.uniform() < 0.5 else -1.0
        if kind == 0:  # doji
            body = 0.02 * rng.uniform() * up
            us, ls = 0.3 + rng.uniform(), 0.3 + rng.uniform()
        elif kind == 1:  # dragonfly / hammer
            body = 0.2 * rng.uniform() * up
            us, ls = 0.01 * rng.uniform(), 1.5 + 2.0 * rng.uniform()
        elif kind == 2:  # gravestone / shooting star
            body = 0.2 * rng.uniform() * up
            us, ls = 1.5 + 2.0 * rng.uniform(), 0.01 * rng.uniform()
        elif kind == 3:  # marubozu
            body = (1.5 + 2.0 * rng.uniform()) * up
            us, ls = 0.0, 0.0
        elif kind == 4:  # long body, short shadows
            body = (1.0 + 2.0 * rng.uniform()) * up
            us, ls = 0.2 * rng.uniform(), 0.2 * rng.uniform()
        elif kind == 5:  # short body, long shadows (spinning top)
            body = 0.3 * rng.uniform() * up
            us, ls = 0.6 + rng.uniform(), 0.6 + rng.uniform()
        elif kind == 6:  # very long body
            body = (3.0 + 3.0 * rng.uniform()) * up
            us, ls = 0.1 * rng.uniform(), 0.1 * rng.uniform()
        elif kind == 7:  # small body
            body = 0.4 * (0.5 + rng.uniform()) * up
            us, ls = 0.1 * rng.uniform(), 0.1 * rng.uniform()
        else:
            body = (0.2 + 1.8 * rng.uniform()) * up
            us, ls = rng.uniform(), rng.uniform()
        c = o + body + drift * 0.2
        h = max(o, c) + us
        l = min(o, c) - ls
        o_l.append(o)
        h_l.append(h)
        l_l.append(l)
        c_l.append(c)
        prev = c
        if prev < 20.0 or prev > 400.0:
            prev = start
    return o_l, h_l, l_l, c_l


def _three_stars_in_the_south(rng):
    """Hand template for a pattern random bars almost never form: three black candles, the
    first long with a long lower shadow, the second smaller inside its range, the third a
    short black marubozu inside the second. Jittered; the search keeps what fires."""
    o, h, l, c = [], [], [], []
    for i in range(12):  # small bodies, so "long" means long
        base = 100.0 + 0.1 * (i % 2)
        o.append(base)
        c.append(base + 0.3 * (1 if i % 2 else -1))
        h.append(max(o[-1], c[-1]) + 0.1)
        l.append(min(o[-1], c[-1]) - 0.1)
    b1 = 2.0 + 2.0 * rng.uniform()
    o1, c1 = 100.0, 100.0 - b1
    o.append(o1)
    c.append(c1)
    h.append(o1 + 0.2 * rng.uniform())
    l.append(c1 - b1 * (1.05 + rng.uniform()))
    l1 = l[-1]
    b2 = b1 * (0.3 + 0.5 * rng.uniform())
    o2 = c1 + (o1 - c1) * (0.2 + 0.6 * rng.uniform())
    c2 = o2 - b2
    o.append(o2)
    c.append(c2)
    h.append(o2 + 0.05 * rng.uniform())
    # Opens above the first close, ranges below it, never under the first low.
    l.append(min(c2, c1 - (c1 - l1) * (0.1 + 0.8 * rng.uniform())))
    b3 = 0.05 + 0.3 * rng.uniform()
    o3 = o2 - 0.1 * rng.uniform()
    c3 = max(o3 - b3, l[-1] + 0.01)
    o.append(o3)
    c.append(c3)
    h.append(o3)
    l.append(c3)
    return o, h, l, c


def _mat_hold(rng):
    """Rising three methods' cousin: long white, a gapped small black, two small reaction
    bars that hold above the first body's middle, then a long white above all their highs."""
    o, h, l, c = [], [], [], []
    for i in range(12):
        base = 100.0 + 0.1 * (i % 2)
        o.append(base)
        c.append(base + 0.3 * (1 if i % 2 else -1))
        h.append(max(o[-1], c[-1]) + 0.1)
        l.append(min(o[-1], c[-1]) - 0.1)

    def add(op, cl, up, dn):
        o.append(op)
        c.append(cl)
        h.append(max(op, cl) + up)
        l.append(min(op, cl) - dn)

    b1 = 3.0 + 2.0 * rng.uniform()
    add(100.0, 100.0 + b1, 0.1 * rng.uniform(), 0.1 * rng.uniform())
    top = 100.0 + b1
    o2 = top + 0.3 + 0.8 * rng.uniform()
    c2 = o2 - 0.1 - 0.4 * rng.uniform()
    add(o2, c2, 0.2 * rng.uniform(), 0.2 * rng.uniform())
    o3 = c2 + 0.1 * rng.uniform()
    c3 = o3 - 0.1 - 0.3 * rng.uniform()
    add(o3, c3, 0.1 * rng.uniform(), 0.1 * rng.uniform())
    o4 = c3 + 0.1 * rng.uniform()
    c4 = max(o4 - 0.1 - 0.3 * rng.uniform(), top - 0.4 * b1)
    add(o4, c4, 0.1 * rng.uniform(), 0.1 * rng.uniform())
    o5 = c4 + 0.1 * rng.uniform()
    add(o5, max(h[-3:]) + 0.2 + 1.5 * rng.uniform(), 0.1 * rng.uniform(), 0.1 * rng.uniform())
    return o, h, l, c


TEMPLATES = {"CDL3STARSINSOUTH": _three_stars_in_the_south, "CDLMATHOLD": _mat_hold}


def _mirror(ohlc):
    o, h, l, c = ohlc
    return (
        [200.0 - x for x in o],
        [200.0 - x for x in l],
        [200.0 - x for x in h],
        [200.0 - x for x in c],
    )


def _cdl(name, o, h, l, c):
    return getattr(talib, name)(np.asarray(o), np.asarray(h), np.asarray(l), np.asarray(c))


def _signs(out):
    return (int((out > 0).sum()), int((out < 0).sum()))


def build_candles(pattern_names):
    """One shared constructed series. A base mixture first; then, for each pattern still
    short of MIN_FIRING in a sign, seeded segments are searched (also mirrored) and
    appended until the full series fires enough, or the budget runs out, in which case
    that sign is recorded as never observed."""
    rng = SplitMix64(0xC4D1E5)
    o, h, l, c = (list(x) for x in _build_bars(rng, CANDLE_BARS, 100.0))
    absent = {}  # name -> list of signs never seen
    for name in pattern_names:
        for sign in (1, -1):
            budget = 30000
            pos, neg = _signs(_cdl(name, o, h, l, c))
            while budget > 0:
                have = pos if sign > 0 else neg
                if have >= MIN_FIRING:
                    break
                budget -= 1
                seg = (
                    TEMPLATES[name](rng)
                    if name in TEMPLATES and rng.uniform() < 0.5
                    else _build_bars(rng, 12, 100.0)
                )
                if rng.uniform() < 0.5:
                    seg = _mirror(seg)
                # Trial on the tail context plus the segment; cheap.
                ctx = 40
                to = o[-ctx:] + list(seg[0])
                th = h[-ctx:] + list(seg[1])
                tl = l[-ctx:] + list(seg[2])
                tc = c[-ctx:] + list(seg[3])
                tail = _cdl(name, to, th, tl, tc)[ctx:]
                if (tail > 0).any() if sign > 0 else (tail < 0).any():
                    o += list(seg[0])
                    h += list(seg[1])
                    l += list(seg[2])
                    c += list(seg[3])
                    pos, neg = _signs(_cdl(name, o, h, l, c))
            else:
                absent.setdefault(name, []).append(sign)
    # Final accounting on the whole series (the appended segments change later context).
    counts = {}
    for name in pattern_names:
        pos, neg = _signs(_cdl(name, o, h, l, c))
        counts[name] = {"pos": pos, "neg": neg}
    return (o, h, l, c), counts, absent


# ------------------------------------------------------------------------------ encoding


def encode_f8(arr):
    a = np.asarray(arr, dtype=np.float64).copy()
    a[np.isnan(a)] = np.nan  # canonical quiet NaN
    return a.astype(">f8").tobytes().hex()


def encode_i4(arr):
    return np.asarray(arr, dtype=">i4").tobytes().hex()


def encode_out(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return "i4", encode_i4(arr)
    return "f8", encode_f8(arr)


def write_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
    with gzip.GzipFile(path, "wb", mtime=0) as f:  # mtime 0: byte-identical reruns
        f.write(data)
    return os.path.getsize(path)


def digest_inputs(series):
    h = hashlib.sha256()
    for scen in sorted(series):
        for col in sorted(series[scen]):
            h.update(f"{scen}/{col}:{series[scen][col]}".encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------- inputs


def make_inputs(candles):
    series = OrderedDict()
    for name, (fn, seed) in SCENARIOS.items():
        o, h, l, c, v = fn(seed, LENGTH)
        rng = SplitMix64(seed ^ 0x5EED)
        periods = [float(2 + rng.below(29)) for _ in range(LENGTH)]  # MAVP: 2..30
        unit = [math.tanh(math.log(x / 100.0) * 3.0) * 0.99 for x in c]
        series[name] = {
            "open": encode_f8(o),
            "high": encode_f8(h),
            "low": encode_f8(l),
            "close": encode_f8(c),
            "volume": encode_f8(v),
            "periods": encode_f8(periods),
            "unit": encode_f8(unit),
        }
    o, h, l, c = candles
    n = len(o)
    rng = SplitMix64(0xC4D1E5 ^ 0x5EED)
    series["candles"] = {
        "open": encode_f8(o),
        "high": encode_f8(h),
        "low": encode_f8(l),
        "close": encode_f8(c),
        "volume": encode_f8(_volume(rng, n)),
    }
    return series


def decode_f8(s):
    return np.frombuffer(bytes.fromhex(s), dtype=">f8").astype(np.float64)


def library_versions():
    return {
        "ta-lib-python": talib.__version__,
        "ta-lib-c": talib.__ta_version__.decode()
        if isinstance(talib.__ta_version__, bytes)
        else str(talib.__ta_version__),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "compatibility": talib.get_compatibility(),
        "unstable_period_ema": talib.get_unstable_period("EMA"),
    }


def header(kind, series_digest, extra=None):
    h = {
        "generator": "tests/oracle/generate.py",
        "generator_version": GENERATOR_VERSION,
        "kind": kind,
        "libraries": library_versions(),
        "input_digest": series_digest,
        "scenarios": list(SCENARIOS) + ["candles"],
        "length": LENGTH,
        "encoding": "f8/i4 big-endian bits, hex, concatenated",
    }
    if extra:
        h.update(extra)
    return h


# ----------------------------------------------------------------------- function table


def function_table():
    out = OrderedDict()
    for group, names in talib.get_function_groups().items():
        for n in names:
            info = abstract.Function(n).info
            inputs = []
            for k, v in info["input_names"].items():
                inputs.extend(v if isinstance(v, list) else [v])
            out[n] = {
                "group": group,
                "display_name": info["display_name"],
                "inputs": inputs,
                "parameters": dict(info["parameters"]),
                "outputs": list(info["output_names"]),
            }
    return {
        "source": "TA-Lib " + library_versions()["ta-lib-c"],
        "generated_by": "tests/oracle/generate.py",
        "functions": out,
    }


def group_slug(group):
    return group.lower().replace(" ", "_")


# --------------------------------------------------------------------- case enumeration


def period_params(params):
    return [k for k in params if "period" in k.lower()]


# Functions whose `matype` is swept at their default periods, every kind 0-8, in all scenarios.
MATYPE_SWEEP = {"MA", "BBANDS", "MAVP"}
MATYPES = (0, 1, 2, 3, 4, 5, 6, 7, 8)

# Momentum composites with one or more moving-average types: each type parameter is swept over
# every kind on its own (the others stay at their default), in one scenario only to keep the
# fixture small. Label: `matype_<k>` for a lone `matype`, else `<parameter>_<k>`.
MATYPE_K = {
    "APO": ("matype",),
    "PPO": ("matype",),
    "MACDEXT": ("fastmatype", "slowmatype", "signalmatype"),
    "STOCH": ("slowk_matype", "slowd_matype"),
    "STOCHF": ("fastd_matype",),
    "STOCHRSI": ("fastd_matype",),
}
MATYPE_K_SCENARIO = "random_walk"


# `nbdev` scales STDDEV's output; for VAR TA-Lib ignores it (the replay test asserts the two
# fixtures are equal). Both get a case with nbdev other than 1.
NBDEV_SWEEP = {"STDDEV", "VAR"}
NBDEV_VALUE = 2.5


def parameter_sets(name):
    """Defaults first, then every period parameter scaled so the largest becomes 14 and
    then 200 (minimum 2), keeping the relative order of fast/slow pairs."""
    info = abstract.Function(name).info
    defaults = dict(info["parameters"])
    sets = [("default", dict(defaults))]
    if name in NBDEV_SWEEP:
        sets.append(("nbdev_2_5", dict(defaults, nbdev=NBDEV_VALUE)))
    if name in MATYPE_SWEEP:
        for m in MATYPES:
            p = dict(defaults, matype=m)
            if all(p != s[1] for s in sets):
                sets.append((f"matype_{m}", p))
    for key in MATYPE_K.get(name, ()):
        for m in MATYPES:
            p = dict(defaults, **{key: m})
            if all(p != s[1] for s in sets):
                sets.append((f"matype_{m}" if key == "matype" else f"{key}_{m}", p))
    pk = period_params(defaults)
    if not pk:
        return sets
    biggest = max(defaults[k] for k in pk)
    for target in PERIOD_TARGETS:
        p = dict(defaults)
        for k in pk:
            p[k] = max(2, int(round(defaults[k] * target / biggest)))
        if p != defaults and all(p != s[1] for s in sets):
            sets.append((f"periods_{target}", p))
    return sets


def input_columns(name):
    info = abstract.Function(name).info
    cols = []
    for v in info["input_names"].values():
        cols.extend(v if isinstance(v, list) else [v])
    if name in UNIT_INPUT:
        cols = ["unit"]
    return cols


def run_function(name, cols, columns, params):
    """One TA-Lib call through the abstract API. Returns (outputs dict, lookback)."""
    f = abstract.Function(name)
    f.parameters = params
    # The abstract API names its inputs by price series: {"close": ...}, {"high": ...}.
    named = {}
    for v in f.info["input_names"].values():
        for x in v if isinstance(v, list) else [v]:
            named[x] = columns[x]
    if name in UNIT_INPUT:
        named = {"close": columns["unit"]}
    f.input_arrays = named
    res = f.run()
    names = f.info["output_names"]
    if len(names) == 1:
        res = [res]
    return OrderedDict(zip(names, res)), f.lookback


def _scaled(defaults, target):
    pk = period_params(defaults)
    biggest = max(defaults[k] for k in pk)
    p = dict(defaults)
    for k in pk:
        p[k] = max(2, int(round(defaults[k] * target / biggest)))
    return p


def generate_function(name, series_dec):
    cols = input_columns(name)
    sets = parameter_sets(name)
    defaults = dict(abstract.Function(name).info["parameters"])
    entry = {
        "inputs": cols,
        "param_order": list(defaults),
        "cases": [],
        "skipped": [],
    }
    # A lookback past the series leaves nothing to compare: record it, and fall back to a
    # shorter period so the function still has a long-period case.
    if period_params(defaults) and any(label == "periods_200" for label, _ in sets):
        if abstract.Function(name).lookback >= 0:
            f = abstract.Function(name)
            f.parameters = _scaled(defaults, 200)
            if f.lookback >= LENGTH:
                sets.append(("periods_%d" % FALLBACK_TARGET, _scaled(defaults, FALLBACK_TARGET)))
    for scen in SCENARIOS:
        columns = series_dec[scen]
        for label, params in sets:
            if name in MATYPE_K and "matype_" in label and scen != MATYPE_K_SCENARIO:
                continue
            try:
                outs, lookback = run_function(name, cols, columns, params)
            except Exception as e:  # a parameter set TA-Lib rejects
                entry["skipped"].append({"scenario": scen, "params": label, "error": str(e)})
                continue
            if lookback >= LENGTH:
                entry["skipped"].append(
                    {
                        "scenario": scen,
                        "params": label,
                        "error": f"lookback {lookback} >= length {LENGTH}",
                    }
                )
                continue
            case = {
                "scenario": scen,
                "label": label,
                "params": params,
                "lookback": int(lookback),
                "out": {},
            }
            for oname, arr in outs.items():
                dt, enc = encode_out(arr)
                case["out"][oname] = {"dtype": dt, "data": enc}
            entry["cases"].append(case)
    return entry


# --------------------------------------------------------------- per-group generators


def _plain_group(group):
    def build(ctx):
        funcs = OrderedDict()
        for name in talib.get_function_groups()[group]:
            funcs[name] = generate_function(name, ctx["series_dec"])
        return funcs, {}

    return build


def _pattern_group(ctx):
    group = "Pattern Recognition"
    names = talib.get_function_groups()[group]
    o, h, l, c = (ctx["series_dec"]["candles"][k] for k in ("open", "high", "low", "close"))
    funcs = OrderedDict()
    for name in names:
        out = getattr(talib, name)(o, h, l, c)
        dt, enc = encode_out(out)
        funcs[name] = {
            "inputs": ["open", "high", "low", "close"],
            "param_order": [],
            "cases": [
                {
                    "scenario": "candles",
                    "label": "default",
                    "params": {},
                    "lookback": int(abstract.Function(name).lookback),
                    "out": {"integer": {"dtype": dt, "data": enc}},
                }
            ],
            "skipped": [],
        }
        pos, neg = _signs(out)
        funcs[name]["firing"] = {"pos": pos, "neg": neg}
    extra = {
        "candle_settings": {k: list(v) for k, v in DEFAULT_CANDLE_SETTINGS.items()},
        "candle_settings_note": "TA-Lib defaults; verified by setting them explicitly (outputs unchanged)",
        "min_firing_per_sign": MIN_FIRING,
        "never_fires_sign": ctx["candle_absent"],
    }
    return funcs, extra


GROUP_GENERATORS = OrderedDict()
if talib is not None:  # the statistics fixtures run without TA-Lib
    for _g in talib.get_function_groups():
        GROUP_GENERATORS[group_slug(_g)] = (_g, _plain_group(_g))
    GROUP_GENERATORS["pattern_recognition"] = ("Pattern Recognition", _pattern_group)


# ------------------------------------------------------------------------ candle settings


def set_settings(table):
    for key, (rng_name, avg, factor) in table.items():
        _tl._ta_set_candle_settings(
            getattr(_tl.CandleSettingType, key), RANGE[rng_name], avg, factor
        )


def restore_settings():
    _tl._ta_restore_candle_default_settings(_tl.CandleSettingType.AllCandleSettings)


def check_default_settings(series_dec):
    cs = series_dec["candles"]
    args = [cs[k] for k in ("open", "high", "low", "close")]
    names = talib.get_function_groups()["Pattern Recognition"]
    base = {n: getattr(talib, n)(*args) for n in names}
    set_settings(DEFAULT_CANDLE_SETTINGS)
    try:
        for n in names:
            if not np.array_equal(base[n], getattr(talib, n)(*args)):
                raise SystemExit(f"default candle settings table is wrong: {n} changed")
    finally:
        restore_settings()


def generate_changed_candles(series_dec, series_digest):
    names = talib.get_function_groups()["Pattern Recognition"]
    cs = series_dec["candles"]
    args = [cs[k] for k in ("open", "high", "low", "close")]
    base = {n: getattr(talib, n)(*args) for n in names}
    funcs = OrderedDict()
    differing = {}
    set_settings(CHANGED_CANDLE_SETTINGS)
    try:
        for n in names:
            out = getattr(talib, n)(*args)
            dt, enc = encode_out(out)
            funcs[n] = {
                "inputs": ["open", "high", "low", "close"],
                "param_order": [],
                "cases": [
                    {
                        "scenario": "candles",
                        "label": "changed_settings",
                        "params": {},
                        "lookback": int(abstract.Function(n).lookback),
                        "out": {"integer": {"dtype": dt, "data": enc}},
                    }
                ],
                "skipped": [],
            }
            differing[n] = int((out != base[n]).sum())
    finally:
        restore_settings()
    extra = {
        "candle_settings": {
            k: list(v) for k, v in {**DEFAULT_CANDLE_SETTINGS, **CHANGED_CANDLE_SETTINGS}.items()
        },
        "changed": {k: list(v) for k, v in CHANGED_CANDLE_SETTINGS.items()},
        "bars_differing_from_default": differing,
    }
    return funcs, extra


# --------------------------------------------------------------------------------- main


def main(argv=None):
    global FIXTURES
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--families",
        choices=("talib", "stats", "all"),
        default="all",
        help="which fixtures to write",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        help="TA-Lib group slugs, or with --families stats the statistics families to write "
        "(for talib: inputs and the function table are always written)",
    )
    ap.add_argument("--out", help="fixture directory (default tests/python/oracle/fixtures)")
    args = ap.parse_args(argv)
    if args.out:
        FIXTURES = os.path.abspath(args.out)
    if args.families in ("talib", "all"):
        if talib is None:
            sys.exit("TA-Lib is not installed: --families stats, or add --with ta-lib==0.6.4")
        main_talib(args)
    if args.families in ("stats", "all"):
        import stats_fixtures

        stats_only = args.only if args.families == "stats" else None
        sizes = stats_fixtures.write_all(
            sys.modules[__name__], os.path.join(FIXTURES, "stats"), stats_only
        )
        print("statistics fixtures total bytes:", sum(sizes.values()))


def main_talib(args):
    # The group list changes the fixtures; make that visible.
    pattern_names = talib.get_function_groups()["Pattern Recognition"]
    candles, counts, absent = build_candles(pattern_names)
    series = make_inputs(candles)
    series_dec = {s: {k: decode_f8(v) for k, v in cols.items()} for s, cols in series.items()}
    digest = digest_inputs(series)

    check_default_settings(series_dec)

    sizes = {}
    sizes["inputs.json.gz"] = write_gz(
        os.path.join(FIXTURES, "inputs.json.gz"),
        {"header": header("inputs", digest), "series": series},
    )
    with open(FUNCTIONS_JSON, "w") as f:
        json.dump(function_table(), f, indent=1, sort_keys=False)
        f.write("\n")

    ctx = {
        "series_dec": series_dec,
        "candle_absent": {k: ["pos" if s > 0 else "neg" for s in v] for k, v in absent.items()},
    }
    total_functions = 0
    for slug, (group, build) in GROUP_GENERATORS.items():
        if args.only and slug not in args.only:
            continue
        funcs, extra = build(ctx)
        total_functions += len(funcs)
        fname = f"talib_{slug}.json.gz"
        sizes[fname] = write_gz(
            os.path.join(FIXTURES, fname),
            {"header": header("talib", digest, {"group": group, **extra}), "functions": funcs},
        )
        print(f"{fname}: {len(funcs)} functions, {sizes[fname]} bytes")

    if not args.only or "pattern_recognition" in args.only:
        funcs, extra = generate_changed_candles(series_dec, digest)
        fname = "talib_candles_changed.json.gz"
        sizes[fname] = write_gz(
            os.path.join(FIXTURES, fname),
            {
                "header": header(
                    "talib_candles_changed", digest, {"group": "Pattern Recognition", **extra}
                ),
                "functions": funcs,
            },
        )
        print(f"{fname}: {len(funcs)} functions, {sizes[fname]} bytes")

    print("functions:", total_functions, "total bytes:", sum(sizes.values()))
    one_sided = {n: v for n, v in absent.items()}
    print(
        "patterns with a sign never observed:",
        {n: ["+" if s > 0 else "-" for s in v] for n, v in one_sided.items()},
    )
    short = {
        n: c
        for n, c in counts.items()
        if (c["pos"] < MIN_FIRING and n not in absent)
        or (c["neg"] < MIN_FIRING and n not in absent)
    }
    print("patterns short of MIN_FIRING without being one-signed:", short)


if __name__ == "__main__":
    main()
