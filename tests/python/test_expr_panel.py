"""Cross-sectional expressions over a panel, and factor evaluation.

A panel is ``(n_dates, n_assets)``. Time-series functions run per name and never see an absent
bar; cross-sectional functions compare the names of one date. The tests state the rules, compute
them by hand, and pin the engine to independent plain-Python implementations, bit for bit:

* ties, range, absent names, ``min_names``, groups;
* a name listed mid-sample counts its warm-up from its own first bar;
* a date with every name absent, and a panel of one name;
* nothing at a date depends on a later date (prefix identity under a poisoned future);
* the result is the same for any number of threads;
* the factor report: a known-answer factor, a shuffled one, the lag, the Newey-West t-statistic.
"""

import math
import os
import subprocess
import sys

import numpy as np
import pytest

import raptorbt as rb
from raptorbt.errors import RaptorBTError

NAN = float("nan")
DAY = 86_400 * 10**9


def canon(a):
    a = np.array(a, dtype=np.float64)
    a[np.isnan(a)] = np.nan
    return a.view(np.uint64)


def same_bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.array_equal(canon(a), canon(b)), f"\n{a}\n!=\n{b}"


def code_of(fn, *a, **kw):
    with pytest.raises(RaptorBTError) as info:
        fn(*a, **kw)
    return info.value.code


def panel(n=70, m=7, seed=11):
    """Prices with two late listings (names 2 and 5) and a halt (name 1), and the presence mask."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, (n, m)), axis=0)
    present = np.ones((n, m), dtype=bool)
    present[:12, 2] = False
    present[:30, 5] = False
    present[40:43, 1] = False
    return np.where(present, close, np.nan), present


# -- independent, plain-Python definitions (no use of the engine) ----------------------------


def naive_rank(row, min_names=2):
    idx = [i for i, v in enumerate(row) if math.isfinite(v)]
    out = [NAN] * len(row)
    if len(idx) < min_names:
        return out
    order = sorted(idx, key=lambda i: row[i])
    m, i = len(order), 0
    while i < m:
        j = i
        while j + 1 < m and row[order[j + 1]] == row[order[i]]:
            j += 1
        for c in order[i : j + 1]:
            out[c] = ((i + j) / 2.0) / (m - 1)
        i = j + 1
    return out


def _seq_sum(xs):
    s = 0.0
    for x in xs:
        s += x
    return s


def naive_zscore(row, min_names=2):
    vals = [v for v in row if math.isfinite(v)]
    out = [NAN] * len(row)
    if len(vals) < min_names:
        return out
    mean = _seq_sum(vals) / len(vals)
    var = _seq_sum((v - mean) * (v - mean) for v in vals) / len(vals)
    if var <= 0.0:
        return out
    sd = math.sqrt(var)
    return [(v - mean) / sd if math.isfinite(v) else NAN for v in row]


def naive_demean(row, min_names=2):
    vals = [v for v in row if math.isfinite(v)]
    if len(vals) < min_names or not vals:
        return [NAN] * len(row)
    mean = _seq_sum(vals) / len(vals)
    return [v - mean if math.isfinite(v) else NAN for v in row]


def naive_winsorize(row, pct, min_names=2):
    vals = sorted(v for v in row if math.isfinite(v))
    if len(vals) < min_names:
        return [NAN] * len(row)
    k = int(math.floor(len(vals) * pct))
    lo, hi = vals[k], vals[len(vals) - 1 - k]
    return [min(max(v, lo), hi) if math.isfinite(v) else NAN for v in row]


def naive_quantile(row, present, q, min_names=2):
    vals = sorted(v for v in row if math.isfinite(v))
    if len(vals) < min_names:
        value = NAN
    else:
        pos = q * (len(vals) - 1)
        lo = int(math.floor(pos))
        frac = pos - lo
        value = vals[min(lo, len(vals) - 1)] if frac == 0.0 or lo + 1 >= len(vals) else (
            vals[lo] + (vals[lo + 1] - vals[lo]) * frac
        )
    return [value if p else NAN for p in present]


def column_values(text, close, present):
    """Each name's own series, streamed on its present bars only, put back at its dates."""
    out = np.full(close.shape, np.nan)
    for a in range(close.shape[1]):
        s = rb.expr(text).stream()
        for d in range(close.shape[0]):
            if present[d, a]:
                out[d, a] = s.update(close=close[d, a])
    return out


# -- the rules, by hand ------------------------------------------------------------------------


def test_rank_ties_range_and_absent_names_by_hand():
    # Date 0: 10, 30, 20, 30 -> ranks 0, 1, 2.5, 2.5 over (n - 1) = 3.
    # Date 1: one NaN and one name without a bar: only 1 and 3 take part -> 0 and 1.
    close = np.array([[10.0, 30.0, 20.0, 30.0], [1.0, 99.0, NAN, 3.0]])
    present = np.array([[True] * 4, [True, False, True, True]])
    got = rb.expr("rank(close)").eval_panel({"close": close}, present=present)
    same_bits(got, [[0.0, 2.5 / 3.0, 1.0 / 3.0, 2.5 / 3.0], [0.0, NAN, NAN, 1.0]])
    assert not np.isinf(got).any()


def test_min_names_below_it_the_date_is_nan_and_it_is_a_parameter_with_a_default():
    close = np.array([[1.0, 2.0, NAN], [1.0, 2.0, 3.0]])
    same_bits(rb.expr("rank(close)").eval_panel({"close": close}), [[0.0, 1.0, NAN], [0.0, 0.5, 1.0]])
    same_bits(rb.expr("rank(close, 3)").eval_panel({"close": close}), [[NAN] * 3, [0.0, 0.5, 1.0]])
    assert rb.expressions.describe("rank")["params"][0]["default"] == 2
    assert code_of(rb.expr, "rank(close, 1)") == "expr_type"  # one name has nothing to compare


def test_zscore_demean_winsorize_quantile_count_by_hand():
    one = {"close": np.array([[1.0, 2.0, 6.0]])}
    same_bits(rb.expr("demean_xs(close)").eval_panel(one), [[-2.0, -1.0, 3.0]])
    s = math.sqrt(2.0 / 3.0)  # 1, 2, 3: mean 2, population variance 2/3
    same_bits(
        rb.expr("zscore_xs(close)").eval_panel({"close": np.array([[1.0, 2.0, 3.0]])}),
        [[-1.0 / s, 0.0, 1.0 / s]],
    )
    same_bits(rb.expr("zscore_xs(close)").eval_panel({"close": np.array([[4.0, 4.0]])}), [[NAN, NAN]])
    ten = {"close": np.arange(1.0, 11.0)[None, :]}  # n = 10, pct 0.1: k = 1
    same_bits(
        rb.expr("winsorize_xs(close, 0.1)").eval_panel(ten),
        [[2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0]],
    )
    five = {"close": np.array([[5.0, 1.0, 4.0, 2.0, 3.0]])}
    same_bits(rb.expr("quantile_xs(close, 0.25)").eval_panel(five), [[2.0] * 5])  # position 1
    assert abs(rb.expr("quantile_xs(close, 0.3)").eval_panel(five)[0, 0] - 2.2) < 1e-15
    # count: a NaN is not counted, but a present name reads the count; an absent one reads NaN.
    c = {"close": np.array([[1.0, NAN, 3.0, 4.0]])}
    mask = np.array([[True, True, True, False]])
    same_bits(rb.expr("count_xs(close)").eval_panel(c, present=mask), [[2.0, 2.0, 2.0, NAN]])
    breadth = rb.expr("count_xs(close) >= 3").eval_panel({"close": np.array([[1.0, 2.0, 3.0]])})
    same_bits(breadth, [[1.0, 1.0, 1.0]])


def test_groups_by_hand_and_min_names_applies_to_each_group():
    sector = np.array([1.0, 2.0, 1.0, 2.0, 2.0])
    x = np.array([[1.0, 10.0, 3.0, 20.0, 60.0]])  # groups {1, 3} and {10, 20, 60}
    e = rb.expr("group_demean(close, sector)", groups=["sector"])
    same_bits(e.eval_panel({"close": x, "sector": sector}), [[-1.0, -20.0, 1.0, -10.0, 30.0]])
    # A group of one has nothing to compare; a NaN label leaves the name out.
    r = rb.expr("group_rank(close, sector)", groups=["sector"]).eval_panel(
        {"close": np.array([[1.0, 2.0, 3.0, 4.0]]), "sector": np.array([1.0, 1.0, 2.0, NAN])}
    )
    same_bits(r, [[0.0, 1.0, NAN, NAN]])
    # The label may also be a full panel, and the built form is the same program.
    built = rb.col.close.group_demean(rb.col.group("sector"))
    assert built.canonical == e.canonical and built.groups == ("sector",)
    full = np.tile(sector, (1, 1))
    same_bits(built.eval_panel({"close": x, "sector": full}), e.eval_panel({"close": x, "sector": sector}))
    assert code_of(built.eval_panel, {"close": x}) == "missing_field"


@pytest.mark.parametrize("seed", range(3))
def test_pinned_to_the_portfolio_panel_functions_where_the_semantics_match(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 6, (40, 9)).astype(np.float64)  # many ties
    x[rng.random(x.shape) < 0.2] = np.nan
    for text, want in (
        ("rank(close)", rb.rank_panel(x, 2)),
        ("rank(close, 4)", rb.rank_panel(x, 4)),
        ("zscore_xs(close)", rb.zscore_panel(x, 2)),
        ("zscore_xs(close, 5)", rb.zscore_panel(x, 5)),
    ):
        same_bits(rb.expr(text).eval_panel({"close": x}), want)
    # winsorize_panel leaves a date with one name alone; winsorize_xs makes it NaN (min_names).
    got = rb.expr("winsorize_xs(close, 0.2)").eval_panel({"close": x})
    want = rb.winsorize_panel(x, 0.2)
    ok = np.isfinite(x).sum(axis=1) >= 2
    same_bits(got[ok], want[ok])


@pytest.mark.parametrize(
    "text, naive",
    [
        ("rank(close)", lambda r, p: naive_rank(r)),
        ("zscore_xs(close)", lambda r, p: naive_zscore(r)),
        ("demean_xs(close)", lambda r, p: naive_demean(r)),
        ("winsorize_xs(close, 0.15)", lambda r, p: naive_winsorize(r, 0.15)),
        ("quantile_xs(close, 0.8)", lambda r, p: naive_quantile(r, p, 0.8)),
    ],
)
def test_each_date_equals_a_plain_implementation_bit_for_bit(text, naive):
    close, present = panel()
    got = rb.expr(text).eval_panel({"close": close}, present=present)
    want = np.array([naive(list(close[d]), list(present[d])) for d in range(len(close))])
    same_bits(got, want)


@pytest.mark.parametrize(
    "text, series, naive",
    [
        ("rank(roc(close, 3))", "roc(close, 3)", naive_rank),
        ("zscore_xs(sma(close, 4))", "sma(close, 4)", naive_zscore),
        ("demean_xs(ema(close, 5))", "ema(close, 5)", naive_demean),
    ],
)
def test_each_name_is_its_own_stream_and_each_date_a_plain_comparison(text, series, naive):
    close, present = panel()
    got = rb.expr(text).eval_panel({"close": close}, present=present)
    cols = column_values(series, close, present)
    want = np.array([naive(list(cols[d])) for d in range(len(cols))])
    want[~present] = np.nan
    same_bits(got, want)
    # Without a mask the NaN cells say the same thing.
    same_bits(rb.expr(text).eval_panel({"close": close}), got)


def test_time_series_after_a_cross_section_and_back():
    close, present = panel()
    got = rb.expr("ema(rank(roc(close, 3)), 4) - zscore_xs(sma(close, 5))").eval_panel(
        {"close": close}, present=present
    )
    ranks = rb.expr("rank(roc(close, 3))").eval_panel({"close": close}, present=present)
    z = rb.expr("zscore_xs(sma(close, 5))").eval_panel({"close": close}, present=present)
    # ema of the rank panel, name by name on its present bars.
    ema = np.full(close.shape, np.nan)
    for a in range(close.shape[1]):
        s = rb.expr("ema(close, 4)").stream()
        for d in range(close.shape[0]):
            if present[d, a]:
                ema[d, a] = s.update(close=ranks[d, a])
    same_bits(got, ema - z)


# -- absent bars and listings ---------------------------------------------------------------------


@pytest.mark.parametrize("text", ["sma(close, 3)", "ema(close, 5)", "rsi(close, 4)", "roc(close, 2)"])
def test_a_late_listing_counts_its_warmup_from_its_own_first_bar(text):
    ts = (np.arange(30, dtype=np.int64) + 1) * DAY
    rng = np.random.default_rng(2)
    a_close = 50.0 + np.cumsum(rng.normal(0, 1, 30))
    b_close = 20.0 + np.cumsum(rng.normal(0, 1, 12))
    data = {"AAA": {"timestamps": ts, "close": a_close}, "BBB": {"timestamps": ts[18:], "close": b_close}}
    e = rb.expr(text)
    got = e.eval_panel(data)
    assert got.shape == (30, 2)
    same_bits(got[:, 0], e.eval({"close": a_close}))
    own = e.eval({"close": b_close})
    same_bits(got[18:, 1], own)  # the same numbers as if BBB were alone
    assert np.isnan(got[:18, 1]).all()
    assert e.warmup > 0 and np.isnan(got[18 : 18 + e.warmup, 1]).all()  # warm-up from ITS first bar


def test_an_absent_bar_does_not_step_but_a_present_nan_does():
    close = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    absent = np.array([[True], [True], [False], [True], [True], [True]])
    got = rb.expr("sma(close, 3)").eval_panel({"close": close}, present=absent)
    # Bars 1, 2, 4, 5, 6 only: the third present bar is the one with 4.
    same_bits(got[:, 0], [NAN, NAN, NAN, (1 + 2 + 4) / 3, (2 + 4 + 5) / 3, (4 + 5 + 6) / 3])
    # The same cell present but NaN steps sma: it is NaN while the NaN is in the window.
    nan_close = close.copy()
    nan_close[2, 0] = NAN
    stepped = rb.expr("sma(close, 3)").eval_panel({"close": nan_close}, present=np.ones((6, 1), bool))
    same_bits(stepped[:, 0], [NAN, NAN, NAN, NAN, NAN, (4 + 5 + 6) / 3])
    # With no mask a NaN close is no bar: it does not step.
    default = rb.expr("sma(close, 3)").eval_panel({"close": nan_close})
    same_bits(default, got)


def test_every_name_absent_on_a_date():
    close, present = panel()
    present = present.copy()
    present[50, :] = False
    close = np.where(present, close, np.nan)
    e = rb.expr("rank(roc(close, 3))")
    got = e.eval_panel({"close": close}, present=present)
    assert np.isnan(got[50]).all()
    assert not np.isnan(got[51:]).all(axis=1).any()
    # The row is not a bar: the series continue as if it had never existed.
    kept = np.delete(np.arange(len(close)), 50)
    same_bits(
        rb.expr("roc(close, 3)").eval_panel({"close": close}, present=present)[kept],
        rb.expr("roc(close, 3)").eval_panel({"close": close[kept]}, present=present[kept]),
    )
    for text in ("count_xs(close)", "quantile_xs(close)", "zscore_xs(close)"):
        assert np.isnan(rb.expr(text).eval_panel({"close": close}, present=present)[50]).all()


def test_one_name_only():
    close = (100.0 + np.arange(10.0))[:, None]
    assert np.isnan(rb.expr("rank(close)").eval_panel({"close": close})).all()
    assert np.isnan(rb.expr("zscore_xs(close)").eval_panel({"close": close})).all()
    assert np.isnan(rb.expr("demean_xs(close)").eval_panel({"close": close})).all()
    same_bits(rb.expr("count_xs(close)").eval_panel({"close": close}), np.ones((10, 1)))
    # Time-series functions are unaffected by how many names there are.
    same_bits(rb.expr("sma(close, 3)").eval_panel({"close": close})[:, 0], rb.expr("sma(close, 3)").eval({"close": close[:, 0]}))
    assert rb.expr("rank(close, 2)").eval_panel({"close": close[:0]}).shape == (0, 1)
    assert rb.expr("rank(close)").eval_panel({"close": np.empty((5, 0))}).shape == (5, 0)


def test_at_inside_a_panel_works_per_name_and_a_cross_section_inside_at_is_refused():
    hour = 3600 * 10**9
    ts = (np.arange(24 * 6, dtype=np.int64) + 1) * hour
    rng = np.random.default_rng(4)
    a = 100.0 + np.cumsum(rng.normal(0, 1, len(ts)))
    b = 50.0 + np.cumsum(rng.normal(0, 1, len(ts)))
    keep = np.ones(len(ts), bool)
    keep[[30, 31, 70]] = False
    data = {"AAA": {"timestamps": ts, "close": a}, "BBB": {"timestamps": ts[keep], "close": b[keep]}}
    e = rb.expr('close > at("1d", close)')
    got = e.eval_panel(data, bar_stamp="close")
    same_bits(got[:, 0], e.eval({"timestamps": ts, "close": a}, bar_stamp="close"))
    own = e.eval({"timestamps": ts[keep], "close": b[keep]}, bar_stamp="close")
    same_bits(got[keep, 1], own)
    # Mixed with a cross-section outside at: the daily value feeds the rank.
    r = rb.expr('rank(at("1d", close))').eval_panel(data, bar_stamp="close")
    assert r.shape == (len(ts), 2)
    assert code_of(rb.expr, 'at("1d", rank(close))') == "expr_unsupported_here"
    assert code_of(e.eval_panel, data) in ("missing_bar_stamp", "missing_field")


# -- types and refusals -----------------------------------------------------------------------------


def test_the_shape_is_a_panel_and_a_non_panel_input_is_refused_typed():
    e = rb.expr("rank(roc(close, 20))")
    assert e.is_panel and not rb.expr("roc(close, 20)").is_panel
    assert rb.expr("roc(rank(close), 3)").is_panel and rb.expr("1 + rank(close)").is_panel
    assert code_of(rb.expr, "rank(1 + 2)") == "expr_type"
    assert code_of(rb.expr, "zscore_xs(2 * 3)") == "expr_type"
    assert code_of(rb.expr, "rank(close > open)") == "expr_type"
    assert code_of(rb.expr, "group_rank(close, 3)") == "expr_type"
    assert code_of(e.eval, {"close": np.arange(30.0)}) == "expr_unsupported_here"
    assert code_of(e.stream) == "expr_unsupported_here"
    assert code_of(rb.expr("1 + 2").eval_panel, {"close": np.ones((3, 3))}) == "expr_unsupported_here"
    # On the bank it runs on a universe (test_expr_panel_live); fed one bar by hand it is refused.
    hand = rb.Indicator.expr(e._prog, 0)
    assert code_of(hand.update_bar, 1.0, 1.0, 1.0, 1.0, 10.0) == "expr_unsupported_here"


def test_eval_panel_checks_its_input():
    e = rb.expr("rank(close)")
    ok = np.ones((4, 3))
    assert code_of(e.eval_panel, {"close": ok[0]}) == "invalid_array"
    assert code_of(e.eval_panel, {"close": ok.astype(np.float32)}) == "invalid_array"
    assert code_of(e.eval_panel, {"close": ok, "volume": np.ones((3, 4))}) == "length_mismatch"
    assert code_of(e.eval_panel, {"open": ok}) == "missing_field"
    assert code_of(e.eval_panel, {}) == "missing_field"
    assert code_of(e.eval_panel, {"close": ok}, present=np.ones((3, 4), bool)) == "length_mismatch"
    assert code_of(e.eval_panel, {"close": ok}, instruments=["a", "b"]) == "length_mismatch"
    assert code_of(e.eval_panel, {"AAA": {"close": np.ones(3)}}) == "missing_field"
    assert code_of(e.eval_panel, 5) == "invalid_input_type"
    before = ok.copy()
    e.eval_panel({"close": ok})
    assert np.array_equal(ok, before)


def test_a_fortran_ordered_panel_gives_the_same_numbers():
    close, present = panel()
    e = rb.expr("rank(roc(close, 3))")
    same_bits(e.eval_panel({"close": np.asfortranarray(close)}), e.eval_panel({"close": close}))


def test_cross_family_rows_say_what_they_are():
    rows = [rb.expressions.describe(r["name"]) for r in rb.expressions.functions() if r["family"] == "cross"]
    assert {r["name"] for r in rows} >= {
        "rank", "zscore_xs", "demean_xs", "winsorize_xs", "quantile_xs", "count_xs",
        "group_rank", "group_zscore", "group_demean",
    }  # fmt: skip
    for r in rows:
        assert r["relation"] == "no_library" and r["reference"].startswith("hand:"), r["name"]
        assert r["nan_policy"] == "cross_section" and r["causal"] is True, r["name"]
        assert r["warmup_at_defaults"] == 0 and "panel" in r["doc"].lower(), r["name"]
        mn = [p for p in r["params"] if p["name"] == "min_names"]
        assert all(p["default"] == 2 for p in mn), r["name"]


# -- causality, determinism ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "rank(roc(close, 5))",
        "zscore_xs(ema(close, 5))",
        "ema(rank(close), 3) - demean_xs(sma(close, 4))",
        "close > quantile_xs(close, 0.7)",
        "winsorize_xs(roc(close, 2), 0.1) * count_xs(close)",
    ],
)
@pytest.mark.parametrize("poison", ["nan", "wild"])
def test_nothing_after_a_date_reaches_it(text, poison):
    close, present = panel()
    e = rb.expr(text)
    ref = e.eval_panel({"close": close}, present=present)
    rng = np.random.default_rng(5)
    for cut in (8, 25, 41, 60):
        c = close.copy()
        tail = c[cut + 1 :]
        c[cut + 1 :] = np.where(
            np.isnan(tail), np.nan, np.nan if poison == "nan" else rng.uniform(-1e6, 1e6, tail.shape)
        )
        got = e.eval_panel({"close": c}, present=present)
        same_bits(got[: cut + 1], ref[: cut + 1])
    # No operation normalises across dates: a prefix of the panel is a prefix of the result.
    for n in (15, 33, 70):
        same_bits(e.eval_panel({"close": close[:n]}, present=present[:n]), ref[:n])


def test_the_result_does_not_depend_on_the_thread_count():
    code = (
        "import numpy as np, hashlib, raptorbt as rb\n"
        "rng = np.random.default_rng(3)\n"
        "c = 100 * np.cumprod(1 + rng.normal(0, .01, (200, 40)), axis=0)\n"
        "c[:20, ::3] = np.nan\n"
        "r = rb.expr('ema(rank(roc(close, 3)), 4) - zscore_xs(sma(close, 5))').eval_panel({'close': c})\n"
        "r = np.where(np.isnan(r), np.nan, r)\n"
        "print(hashlib.sha256(r.tobytes()).hexdigest())\n"
    )
    digests = set()
    for threads in ("1", "2", "7"):
        env = {**os.environ, "RAYON_NUM_THREADS": threads}
        out = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        digests.add(out.stdout.strip())
    assert len(digests) == 1


# -- mapping form -----------------------------------------------------------------------------------------


def test_the_mapping_form_aligns_on_the_union_of_dates_and_selects_instruments():
    ts = (np.arange(6, dtype=np.int64) + 1) * DAY
    data = {
        "AAA": {"timestamps": ts, "close": np.arange(1.0, 7.0)},
        "BBB": {"timestamps": ts[2:], "close": np.arange(10.0, 14.0)},
        "CCC": {"timestamps": ts[::2], "close": np.array([5.0, 6.0, 7.0])},
    }
    both = rb.expr("close").eval_panel(data)
    same_bits(both[:, 0], np.arange(1.0, 7.0))
    same_bits(both[:, 1], [NAN, NAN, 10.0, 11.0, 12.0, 13.0])
    same_bits(both[:, 2], [5.0, NAN, 6.0, NAN, 7.0, NAN])
    picked = rb.expr("close").eval_panel(data, instruments=["CCC", "AAA"])
    same_bits(picked, both[:, [2, 0]])
    bad = {"AAA": {"timestamps": ts[[0, 0, 1]], "close": np.ones(3)}, "BBB": {"timestamps": ts, "close": np.ones(6)}}
    assert code_of(rb.expr("close").eval_panel, bad) == "invalid_input"
    nostamp = {"AAA": {"close": np.ones(3)}, "BBB": {"close": np.ones(3)}}
    assert code_of(rb.expr("close").eval_panel, nostamp) == "missing_field"


# -- factor evaluation -------------------------------------------------------------------------------------


def ic_data(n=120, m=30, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 0.01, (n, m))
    close = 100.0 * np.cumprod(1.0 + ret, axis=0)
    return ret, close


def test_a_signal_equal_to_the_next_bars_return_has_ic_one():
    ret, close = ic_data()
    factor = np.full_like(close, np.nan)
    factor[:-2] = ret[2:]  # the return of the bar after next: what lag=1, h=1 measures
    rep = rb.expressions.factor_report(factor, close, horizons=[1])
    h = rep["horizons"][1]
    assert rep["no_costs"] is True and rep["lag"] == 1
    assert abs(h["ic_mean"] - 1.0) < 1e-12 and h["hit_rate"] == 1.0
    assert h["long_short_mean"] > 0 and np.all(np.diff(h["quantile_returns"]) > 0)
    # Entering the same close the signal is computed on (lag=0) sees a different return.
    lag0 = rb.expressions.factor_report(factor, close, horizons=[1], lag=0)["horizons"][1]
    se = 1.0 / math.sqrt((close.shape[1] - 1) * lag0["n_scored"])
    assert abs(lag0["ic_mean"]) < 4 * se


def test_a_shuffled_signal_has_ic_near_zero_within_its_standard_error():
    ret, close = ic_data(n=160, m=40, seed=2)
    factor = np.full_like(close, np.nan)
    factor[:-2] = ret[2:]
    rng = np.random.default_rng(9)
    shuffled = np.array([rng.permutation(row) for row in factor])
    h = rb.expressions.factor_report(shuffled, close, horizons=[1])["horizons"][1]
    # Each date's rank IC of unrelated names has standard deviation 1 / sqrt(m - 1); the mean of
    # n_scored independent dates has that over sqrt(n_scored). Four standard errors.
    se = 1.0 / math.sqrt((close.shape[1] - 1) * h["n_scored"])
    assert abs(h["ic_mean"]) < 4 * se
    assert abs(h["ic_std"] - 1.0 / math.sqrt(close.shape[1] - 1)) < 0.2 * h["ic_std"]


def test_the_report_is_pinned_to_rank_ic():
    ret, close = ic_data()
    factor = np.where(np.random.default_rng(3).random(close.shape) < 0.1, np.nan, -ret)
    for lag in (0, 1, 3):
        rep = rb.expressions.factor_report(factor, close, horizons=[1, 5], lag=lag)
        for hz in (1, 5):
            direct = rb.rank_ic(factor[: len(close) - lag], close[lag:], hz, 10)
            ic = rep["horizons"][hz]["ic"]
            same_bits(ic[: len(close) - lag], direct.daily_ic())
            assert np.isnan(ic[len(close) - lag :]).all()
            assert rep["horizons"][hz]["ic_mean"] == direct.mean_ic
            assert rep["horizons"][hz]["t_naive"] == direct.t_stat


def newey_west_t(x, lags):
    """Newey-West t of the mean (Bartlett weights, no small-sample correction), by the formula."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    e = x - x.mean()
    s = float(e @ e)
    for j in range(1, lags + 1):
        s += 2.0 * (1.0 - j / (lags + 1.0)) * float(e[j:] @ e[:-j])
    return x.mean() / math.sqrt(s / n**2)


def test_the_t_statistic_under_overlap_is_newey_west_with_h_minus_one_lags():
    ret, close = ic_data(n=200, m=30, seed=6)
    factor = np.cumsum(-ret, axis=0)  # a slow signal: a long window of past returns, reversed
    rep = rb.expressions.factor_report(factor, close, horizons=[1, 5, 20])
    for hz in (1, 5, 20):
        h = rep["horizons"][hz]
        ic = h["ic"][np.isfinite(h["ic"])]
        assert h["hac_lags"] == hz - 1 and len(ic) == h["n_scored"]
        assert abs(h["t_hac"] - newey_west_t(ic, hz - 1)) < 1e-9 * max(1.0, abs(h["t_hac"]))
        ls = h["long_short"][np.isfinite(h["long_short"])]
        assert abs(h["long_short_t_hac"] - newey_west_t(ls, hz - 1)) < 1e-9 * max(1.0, abs(h["long_short_t_hac"]))
    h1 = rep["horizons"][1]
    n = h1["n_scored"]
    assert abs(h1["t_hac"] - h1["t_naive"] * math.sqrt(n / (n - 1))) < 1e-9 * abs(h1["t_naive"])


def test_the_t_statistic_matches_statsmodels_when_it_is_installed():
    sm = pytest.importorskip("statsmodels.api")
    ret, close = ic_data(n=200, m=30, seed=6)
    rep = rb.expressions.factor_report(np.cumsum(-ret, axis=0), close, horizons=[5])
    ic = rep["horizons"][5]["ic"]
    ic = ic[np.isfinite(ic)]
    fit = sm.OLS(ic, np.ones_like(ic)).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False})
    assert abs(float(fit.tvalues[0]) - rep["horizons"][5]["t_hac"]) < 1e-8


def test_quantiles_coverage_turnover_and_autocorrelation_by_hand():
    n, m = 60, 20
    ret, close = ic_data(n=n, m=m, seed=8)
    static = np.tile(np.arange(float(m)), (n, 1))
    rep = rb.expressions.factor_report(static, close, horizons=[1], quantiles=4)
    assert rep["quantiles"] == 4 and rep["turnover"].shape == (4,)
    assert np.all(rep["turnover"] == 0.0)  # a static factor never changes bucket
    assert abs(rep["autocorr_mean"] - 1.0) < 1e-12
    assert np.all(rep["coverage"] == 1.0) and np.all(rep["names"] == m)
    h = rep["horizons"][1]
    # Buckets of 5 names: ascending names, so bucket q holds names 5q..5q+4. Date 0 hand-check.
    fwd = close[2] / close[1] - 1.0  # lag 1, horizon 1
    want = [fwd[5 * q : 5 * q + 5].mean() for q in range(4)]
    assert np.allclose(h["quantile_returns_by_date"][0], want, rtol=0, atol=1e-15)
    assert abs(h["long_short"][0] - (want[3] - want[0])) < 1e-15
    # A factor that reverses every date turns the whole top bucket over.
    flip = static * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None]
    rev = rb.expressions.factor_report(flip, close, horizons=[1], quantiles=4)
    assert np.allclose(rev["turnover"][np.isfinite(rev["turnover"])], 1.0)
    assert abs(rev["autocorr_mean"] + 1.0) < 1e-12
    # Names without a factor value lower the coverage.
    gap = static.copy()
    gap[:, :5] = np.nan
    assert np.allclose(rb.expressions.factor_report(gap, close, horizons=[1])["coverage"], 0.75)


def test_an_expression_is_a_factor_and_the_result_says_which():
    ret, close = ic_data(n=140, m=24, seed=4)
    volume = np.exp(np.random.default_rng(1).normal(10, 1, close.shape))
    e = rb.expr("rank(roc(close, 5))")
    rep = rb.expressions.factor_report(e, {"close": close, "volume": volume}, horizons=[1, 5])
    direct = rb.expressions.factor_report(e.eval_panel({"close": close}), close, horizons=[1, 5])
    assert rep["expression"] == e.canonical and rep["hash"] == e.hash
    for hz in (1, 5):
        same_bits(rep["horizons"][hz]["ic"], direct["horizons"][hz]["ic"])
    same_bits(rep["coverage"], direct["coverage"])
    # Prices alone also do; so does the mapping form.
    assert rb.expressions.factor_report(e, close, horizons=[1])["horizons"][1]["n_scored"] > 0
    assert code_of(rb.expressions.factor_report, rb.expr("macd(close)"), close) == "expr_type"


def test_factor_report_refuses_what_it_cannot_support():
    ret, close = ic_data(n=60, m=30)
    f = -ret
    report = rb.expressions.factor_report
    assert code_of(report, f[:25], close[:25]) == "insufficient_observations"  # too few dates
    assert code_of(report, f[:, :6], close[:, :6]) == "insufficient_observations"  # fewer names than min_names
    assert code_of(report, f[:, :4], close[:, :4], min_names=2, quantiles=5) == "insufficient_observations"
    assert code_of(report, f[:50], close) == "length_mismatch"  # shapes differ
    inf = f.copy()
    inf[3, 3] = np.inf
    assert code_of(report, inf, close) == "invalid_input"
    assert code_of(report, f, close, horizons=[]) == "invalid_parameter"
    assert code_of(report, f, close, horizons=[0]) == "invalid_parameter"
    assert code_of(report, f, close, quantiles=1) == "invalid_parameter"
    assert code_of(report, f, close, lag=-1) == "invalid_parameter"
    assert code_of(report, f.astype(np.float32), close) == "invalid_array"
    assert code_of(report, f[0], close[0]) == "invalid_array"


def test_forward_returns_do_not_exist_outside_the_report():
    # The factor is computed from its inputs; only the report reads later prices.
    ret, close = ic_data(n=100, m=24, seed=12)
    e = rb.expr("rank(roc(close, 5))")
    later = close.copy()
    # From row 90 on, even names soar and odd names collapse: every forward return that reaches
    # row 90 is ranked by parity, which reorders the ranks any window ending there sees.
    later[90:, ::2] *= 100.0
    later[90:, 1::2] *= 0.01
    a = e.eval_panel({"close": close})
    same_bits(e.eval_panel({"close": later})[:90], a[:90])  # the factor never reads ahead
    h = 5  # lag 1: date d earns close[d + 6] / close[d + 1]
    ic = lambda c: rb.expressions.factor_report(e, c, horizons=[h])["horizons"][h]["ic"]
    before, after = ic(close), ic(later)
    last_unmoved = 90 - 1 - h - 1  # its exit row, d + 6, is 89
    same_bits(before[: last_unmoved + 1], after[: last_unmoved + 1])
    # Dates whose window reaches row 90 and whose own factor is unchanged do move.
    assert (before[last_unmoved + 1 : 90] != after[last_unmoved + 1 : 90]).any()
