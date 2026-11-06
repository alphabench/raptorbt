"""``raptorbt.analysis.Session``: the summary contract, handles, the store and its limits.

The contract an agent leans on: one call returns a summary of at most 2 KB (4 KB for a call),
the same work gives the same handle, a handle is a series in the next expression, a handle that
is gone says how to remake it, and nothing here needs pandas or touches the input.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import errors, expressions
from raptorbt.analysis import Session
from raptorbt.analysis import data as adata
from raptorbt.analysis import summary as asum

MIN = 60 * 10**9


def bars(n=300, seed=3, start=100.0, volume=True):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    out = {
        "timestamps": np.arange(1, n + 1, dtype=np.int64) * MIN,
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
    }
    if volume:
        out["volume"] = np.round(rng.uniform(500.0, 1500.0, n))
    return out


def session(n=300, **kw):
    s = Session(**kw)
    s.load(bars(n), bar_stamp="close")
    return s


def size(x):
    return asum.size(x)


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


# ----------------------------------------------------------------------------------- load


def test_load_needs_a_bar_stamp_and_names_both_choices():
    with pytest.raises(errors.RaptorBTError) as e:
        Session().load(bars())
    assert e.value.code == "missing_bar_stamp" and e.value.valid == ["open", "close"]
    with pytest.raises(errors.RaptorBTError) as e:
        Session().load(bars(), bar_stamp="start")
    assert e.value.code == "invalid_bar_stamp"


def test_load_returns_facts_and_never_modifies_or_aliases_the_input():
    data = bars(50)
    before = {k: v.copy() for k, v in data.items()}
    s = Session()
    facts = s.load(data, bar_stamp="close", bar_ns=MIN, name="demo")
    assert facts["data"] == "demo" and facts["rows"] == 50 and facts["bar_ns"] == MIN
    assert facts["fields"] == ["open", "high", "low", "close", "volume"]
    assert facts["first_timestamp"] == MIN and facts["last_timestamp"] == 50 * MIN
    s.eval("sma(close, 5)")
    for k in data:
        assert np.array_equal(data[k], before[k]) and data[k].flags.writeable
    data["close"][0] = -1.0  # the session holds a copy
    assert s.dataset("demo").arrays["close"][0] == before["close"][0]
    assert not s.dataset("demo").arrays["close"].flags.writeable


def test_a_frame_like_object_loads_and_pandas_is_never_imported():
    class Frame:  # what a DataFrame answers: frame[name] and frame.keys()
        def __init__(self, cols):
            self._c = cols

        def __getitem__(self, k):
            return self._c[k]

        def keys(self):
            return self._c.keys()

    d = bars(40)
    d["vwap"] = d["close"].copy()
    f = adata.build(Frame(d), "close", None, None)
    assert f.ignored == ("vwap",) and f.rows == 40
    code = (
        "import sys, numpy as np\n"
        "from raptorbt.analysis import Session\n"
        "n = 60\n"
        "s = Session(); s.load({'timestamps': np.arange(1, n + 1), 'close': np.arange(1.0, n + 1)}, bar_stamp='close')\n"
        "h = s.eval('rsi(close, 14)')['handle']; s.eval(f'zscore({h}, 20)')\n"
        "assert 'pandas' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_load_refuses_bad_bars_with_a_typed_error():
    d = bars(10)
    for bad, code in (
        ({k: v for k, v in d.items() if k != "timestamps"}, "missing_field"),
        ({"timestamps": d["timestamps"]}, "missing_field"),
        ({**d, "close": d["close"][:5]}, "length_mismatch"),
        ({**d, "timestamps": d["timestamps"][::-1].copy()}, "decreasing_timestamps"),
        ({**d, "timestamps": d["timestamps"].astype(float)}, "invalid_array"),
        ({**d, "close": ["a"] * 10}, "invalid_array"),
    ):
        with pytest.raises(errors.RaptorBTError) as e:
            Session().load(bad, bar_stamp="close")
        assert e.value.code == code, (code, e.value)
    with pytest.raises(errors.RaptorBTError) as e:
        Session().load(d, bar_stamp="close", name="../x")
    assert e.value.code == "invalid_parameter"


def test_datetime64_timestamps_are_nanoseconds():
    d = bars(10)
    d["timestamps"] = d["timestamps"].astype("datetime64[ns]")
    assert Session().load(d, bar_stamp="close")["first_timestamp"] == MIN


def test_the_same_bars_again_are_not_stored_twice():
    s = Session()
    a = s.load(bars(), bar_stamp="close", name="x")
    used = s._used
    assert s.load(bars(), bar_stamp="close", name="x") == a and s._used == used


# ------------------------------------------------------------------------------- the summary


def test_the_summary_of_a_known_series_is_exact():
    n = 60
    s = Session()
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64) * MIN, "close": np.arange(1.0, n + 1)}, bar_stamp="close")
    r = s.eval("sma(close, 5)")  # 3, 4, ..., 58 after four unknown bars
    assert r["shape"] == [n] and r["warmup"] == 4 and r["kind"] == "number"
    assert r["nan"] == {"leading": 4, "interior": 0}
    assert r["dist"]["min"] == 3.0 and r["dist"]["max"] == 58.0
    assert r["dist"]["p50"] == 30.5 and r["dist"]["mean"] == 30.5
    assert r["last"] == [54.0, 55.0, 56.0, 57.0, 58.0]
    assert r["causal"] is True and r["reference"] == {"sma": "fixture"}
    assert r["sha256"] == asum.sha256_of(s.series(r["handle"]))
    assert set(r) >= {"handle", "shape", "warmup", "nan", "dist", "last", "flags", "causal", "reference", "sha256"}


def test_shown_numbers_are_rounded_and_the_stored_series_is_not():
    s = session()
    r = s.eval("zscore(close, 20)")
    stored = s.series(r["handle"])
    assert not stored.flags.writeable and stored.dtype == np.float64
    last = float(stored[-1])
    assert r["last"][-1] == float(f"{last:.5e}") and r["last"][-1] != last
    assert same(stored, expressions.expr("zscore(close, 20)").eval(bars()))


def test_a_boolean_result_reports_how_often_it_is_true():
    n = 60
    s = Session()
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64) * MIN, "close": np.arange(1.0, n + 1)}, bar_stamp="close")
    r = s.eval("close > sma(close, 5)")  # rising: true from the first known bar
    assert r["kind"] == "bool" and "dist" not in r
    assert (r["known"], r["true"], r["true_rate"]) == (n - 4, n - 4, 1.0)
    assert r["last"] == [1.0] * 5
    q = s.eval("close > sma(close, 5) and close < 30")
    assert q["true"] == 25 and q["known"] == n - 4 and q["true_rate"] == round(25 / 56, 6)


def test_unknown_values_are_none_in_last_and_counted_not_hidden():
    n = 40
    close = np.arange(1.0, n + 1)
    close[30] = np.nan
    s = Session()
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64), "close": close}, bar_stamp="close")
    r = s.eval("sma(close, 3)")  # unknown at 30, 31, 32
    assert r["nan"] == {"leading": 2, "interior": 3}
    assert r["flags"]["gaps"] == 1
    t = s.eval("close")
    assert t["last"] == [36.0, 37.0, 38.0, 39.0, 40.0] and t["nan"] == {"leading": 0, "interior": 1}
    short = Session()
    short.load({"timestamps": np.arange(1, 4, dtype=np.int64), "close": np.array([1.0, 2.0, np.nan])}, bar_stamp="close")
    assert short.eval("close")["last"] == [1.0, 2.0, None]


def test_a_multi_output_result_has_a_summary_per_output_and_each_is_a_handle():
    s = session()
    r = s.eval("macd(close)")
    assert [o["handle"] for o in r["outputs"]] == [f"{r['handle']}_{n}" for n in ("macd", "macdsignal", "macdhist")]
    assert "truncated_outputs" not in r
    want = expressions.expr("macd(close)").eval(bars())
    for o, name in zip(r["outputs"], want):
        assert same(s.series(o["handle"]), want[name])
    both = s.series(r["handle"])
    assert set(both) == {"macd", "macdsignal", "macdhist"}
    z = s.eval(f"zscore({r['outputs'][1]['handle']}, 10)")
    assert z["warmup"] == r["outputs"][1]["warmup"] + 9
    with pytest.raises(errors.RaptorBTError) as e:  # the group is not a series
        s.eval(f"sma({r['handle']}, 3)")
    assert e.value.code == "unknown_handle" and r["outputs"][0]["handle"] in e.value.valid


# ---------------------------------------------------------------------------------- size caps


def test_every_summary_fits_its_cap_on_worst_cases():
    s = session(400)
    worst = [
        "macd(close)",
        "stoch(high, low, close)",
        "bbands(close, 20)",
        "close * 1e300",                      # the widest numbers print
        "close * 1e-300",
        "(close - 100) * 1e150",
        "ht_phasor(close)",
        "if_else(close > open, close * 1.2345678901234567e100, low)",
        "close > open",
    ]
    calls = []
    for text in worst:
        r = s.eval(text)
        calls.append(r)
        outs = r["outputs"] if "outputs" in r else [r]
        assert size(r) <= asum.CALL_BYTES, text
        assert all(size(o) <= asum.OUTPUT_BYTES for o in outs), text
    assert len(calls) == len(worst)


def test_an_expression_over_every_function_still_fits_and_says_what_it_left_out():
    parts = []
    for f in expressions.functions():
        text = expressions.describe(f["name"])["example"]
        e = expressions.expr(text)
        if e.kinds == ("number",) and len(e.outputs) == 1 and not e.is_panel:
            parts.append(f"({text})")
    assert len(parts) > 100

    def balanced(xs):  # the language limits depth to 64
        return xs[0] if len(xs) == 1 else f"({balanced(xs[: len(xs) // 2])} + {balanced(xs[len(xs) // 2 :])})"

    s = session(400)
    r = s.eval(balanced(parts))
    assert size(r) <= asum.OUTPUT_BYTES
    assert len(r["reference"]) == asum.LISTED and r["reference_more"] > 100


def test_a_summary_with_a_huge_reference_list_is_cut_to_its_cap():
    fns = {f"fn{i}": "hand:fns::rolling::tests::some_very_long_test_name_" + "x" * 80 for i in range(400)}
    series = np.arange(100.0)
    out = asum.summarise(
        "h_0123456789", series, "number", warmup=3, inputs=[series], prices=[series], causal=False,
        non_causal=[f"function_with_a_long_name_{i}" for i in range(400)], reference=fns,
    )
    assert size(out) <= asum.OUTPUT_BYTES
    assert out["causal"] is False and out["flags"]["non_causal"]


def test_outputs_past_the_call_cap_are_named_not_silently_dropped(monkeypatch):
    s = session()
    one = size(s.eval("macd(close)")["outputs"][0])
    monkeypatch.setattr(asum, "CALL_BYTES", 2 * one + 120)
    r = s.eval("bbands(close, 20)")  # a different expression: computed under the smaller cap
    assert len(r["outputs"]) == 2 and len(r["truncated_outputs"]) == 1
    assert r["truncated_outputs"][0] == f"{r['handle']}_lowerband"
    assert size(r) <= 2 * one + 120
    assert s.series(r["truncated_outputs"][0]).shape == (300,)  # the handle still exists


# --------------------------------------------------------------------------------- the flags


def test_flat_windows_fires_on_the_rsi_of_a_name_that_does_not_trade():
    n = 120
    s = Session()
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64) * MIN, "close": np.full(n, 250.0)}, bar_stamp="close")
    r = s.eval("rsi(close, 14)")
    assert r["dist"]["p50"] == 100.0  # the decided behaviour: no losses, so 100
    assert r["flags"] == {"flat_windows": n - 14}
    moving = session().eval("rsi(close, 14)")
    assert "flat_windows" not in moving["flags"]


def test_flat_windows_counts_the_bars_whose_whole_window_is_flat():
    x = np.concatenate([np.full(30, 7.0), 7.0 + np.arange(1.0, 21.0)])
    out = np.concatenate([np.full(14, np.nan), np.linspace(1, 2, 36)])
    # window 15: bars 14..29 have a flat trailing 15 bars and a known output
    assert asum.count_flat_windows([x], 15, out) == 16
    assert asum.count_flat_windows([x, x[::-1].copy()], 15, out) == 0  # every input must be flat
    assert asum.count_flat_windows([x], 1, out) == 0 and asum.count_flat_windows([], 15, out) == 0
    nan_in = x.copy()
    nan_in[20] = np.nan  # a NaN in the window is not flat
    assert asum.count_flat_windows([nan_in], 15, out) < 16


def test_gaps_count_runs_of_unknown_after_the_first_known_value():
    nan = np.nan
    assert asum.count_gaps(np.array([nan, nan, 1, 2, nan, nan, 3, nan])) == 2
    assert asum.count_gaps(np.array([nan, nan, 1, 2, 3.0])) == 0
    assert asum.count_gaps(np.array([nan, nan])) == 0


def test_outliers_are_values_far_from_the_median_in_robust_units():
    x = np.arange(1.0, 101.0)
    assert asum.count_outliers(x) == 0
    y = x.copy()
    y[50] = 10_000.0  # median 50.5, MAD 25: the cut is 8 * 1.4826 * 25 = 296.5
    assert asum.count_outliers(y) == 1
    assert asum.count_outliers(np.zeros(50)) == 0  # MAD 0: no scale, none reported
    r = session().eval("close * 1.0")
    assert "outliers" not in r["flags"]


def test_nonpositive_input_counts_bars_with_a_price_at_or_below_zero():
    a = np.array([1.0, 0.0, 3.0, -1.0, np.nan])
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert asum.count_nonpositive([a, b]) == 2
    assert asum.count_nonpositive([]) == 0
    n = 40
    close = np.arange(1.0, n + 1)
    close[10] = -2.0
    s = Session()
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64), "close": close, "volume": np.zeros(n)}, bar_stamp="close")
    assert s.eval("ln(close)")["flags"]["nonpositive_input"] == 1
    assert "nonpositive_input" not in s.eval("sma(volume, 3)")["flags"]  # volume is not a price


def test_inf_counts_infinities_in_the_inputs_and_the_output():
    assert asum.count_inf(np.array([1.0, np.inf, -np.inf]), np.array([np.inf])) == 3
    n = 40
    close = np.arange(1.0, n + 1)
    close[5] = np.inf
    s = Session()
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64), "close": close}, bar_stamp="close")
    r = s.eval("sma(close, 3)")
    assert r["flags"]["inf"] == 1
    assert all(x is None or np.isfinite(x) for x in r["last"])


def test_short_history_is_fewer_than_thirty_known_values():
    assert asum.is_short(29) and not asum.is_short(30)
    s = Session()
    s.load({"timestamps": np.arange(1, 21, dtype=np.int64), "close": np.arange(1.0, 21)}, bar_stamp="close")
    assert s.eval("sma(close, 3)")["flags"]["short_history"] == 18
    assert "short_history" not in session().eval("sma(close, 3)")["flags"]


def test_non_causal_names_the_functions_that_use_later_bars(monkeypatch):
    assert asum.non_causal_names(["a", "b", "c"], {"a": False, "b": True, "c": False}) == ["a", "c"]
    registry = expressions._registry()
    row = dict(registry["sma"], causal=False)
    monkeypatch.setitem(registry, "sma", row)
    r = session().eval("sma(close, 3) + 1")
    assert r["causal"] is False and r["flags"]["non_causal"] == ["sma"]
    chained = session()
    h = chained.eval("sma(close, 3)")["handle"]
    assert chained.eval(f"{h} + 1")["flags"]["non_causal"] == ["sma"]  # inherited through the handle
    assert "non_causal" not in session().eval("ema(close, 3)")["flags"]


# --------------------------------------------------------------------- handles, determinism


def test_the_same_work_has_the_same_handle_in_any_session_and_spelling():
    a, b = session(), session()
    b.eval("ema(close, 9)")  # other work first
    h = a.eval("rsi(close, 14)")["handle"]
    assert h.startswith("h_") and len(h) == 12
    assert b.eval("rsi(close, 14)")["handle"] == h
    assert b.eval("rsi( close , 14.0 )")["handle"] == h
    assert b.eval(expressions.expr("rsi(close, 14)"))["handle"] == h
    assert b.eval("rsi(close, 15)")["handle"] != h


def test_the_handle_depends_on_the_data_and_on_the_engine_version(monkeypatch):
    base = session().eval("rsi(close, 14)")["handle"]
    other = Session()
    d = bars()
    d["close"][-1] += 1e-9
    other.load(d, bar_stamp="close")
    assert other.eval("rsi(close, 14)")["handle"] != base
    stamped = Session()
    stamped.load(bars(), bar_stamp="open")
    assert stamped.eval("rsi(close, 14)")["handle"] != base
    monkeypatch.setattr(rb, "__version__", "9.9.9")
    assert session().eval("rsi(close, 14)")["handle"] != base


def test_a_repeat_is_served_from_the_store_not_recomputed(monkeypatch):
    s = session()
    first = s.eval("rsi(close, 14)")
    calls = []
    real = Session._evaluate

    def counting(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(Session, "_evaluate", counting)
    again = s.eval("rsi(close, 14)")
    assert again == first and calls == []
    s.eval("rsi(close, 13)")
    assert calls == [1]
    again["last"][0] = -1  # a copy: the stored summary cannot be edited through a result
    assert s.eval("rsi(close, 14)") == first


def test_a_handle_is_a_series_in_the_next_expression_bit_for_bit():
    s = session()
    h = s.eval("sma(close, 5)")["handle"]
    chained = s.eval(f"zscore({h}, 20)")
    nested = s.eval("zscore(sma(close, 5), 20)")
    assert same(s.series(chained["handle"]), s.series(nested["handle"]))
    assert chained["warmup"] == nested["warmup"] == 23
    assert chained["handle"] != nested["handle"]  # different text, so different work
    mixed = s.eval(f"({h} - close) / atr(14)")
    assert mixed["warmup"] == expressions.expr("atr(14)").warmup == 13
    two = s.eval(f"{h} - {s.eval('ema(close, 5)')['handle']}")
    assert two["warmup"] == 4 and two["handle"] in s.handles()
    assert s.describe_handle(chained["handle"])["depends_on"] == [h]


def test_a_handle_from_other_bars_cannot_be_mixed_in():
    a = session()
    h = a.eval("sma(close, 5)")["handle"]
    other = bars(300, seed=9)
    a.load(other, bar_stamp="close", name="other")
    with pytest.raises(errors.RaptorBTError) as e:
        a.eval(f"zscore({h}, 20)", data="other")
    assert e.value.code == "invalid_input" and e.value.parameter == h


def test_series_tail_and_bad_handles():
    s = session()
    h = s.eval("sma(close, 5)")["handle"]
    assert s.series(h, tail=3).shape == (3,) and same(s.series(h, tail=3), s.series(h)[-3:])
    for bad in (0, -1, 1.5, True):
        with pytest.raises(errors.RaptorBTError) as e:
            s.series(h, tail=bad)
        assert e.value.code == "invalid_parameter"
    for name in ("../../etc/passwd", "h_zz", "", "h_0123456789/../x", None, 5):
        with pytest.raises(errors.RaptorBTError) as e:
            s.series(name)
        assert e.value.code == "unknown_handle"


def test_describe_handle_gives_the_recipe():
    s = session()
    h = s.eval("rsi(close, 14)")["handle"]
    d = s.describe_handle(h)
    name = s.dataset().name
    assert d["recipe"] == f'Session.eval("rsi(close, 14)", data={name!r})'
    assert d["shape"] == [300] and d["bytes"] == 300 * 8 and d["outputs"] == [h]


# --------------------------------------------------------------------------- store and limits


def test_least_recently_used_handles_are_evicted_and_use_counts():
    s = session(max_handles=3)
    a = s.eval("sma(close, 3)")["handle"]
    b = s.eval("sma(close, 4)")["handle"]
    c = s.eval("sma(close, 5)")["handle"]
    s.series(a)  # a is now the newest; b is the oldest
    d = s.eval("sma(close, 6)")["handle"]
    assert s.handles() == [c, a, d]
    with pytest.raises(errors.RaptorBTError) as e:
        s.series(b)
    assert e.value.code == "handle_expired"


def test_an_expired_handle_carries_the_call_that_recomputes_it():
    s = session(max_handles=1)
    first = s.eval("sma(close, 3)")
    s.eval("sma(close, 4)")
    name = s.dataset().name
    with pytest.raises(errors.RaptorBTError) as e:
        s.eval(f"zscore({first['handle']}, 10)")
    err = e.value
    assert err.code == "handle_expired" and isinstance(err, RuntimeError)
    assert err.valid == [f'Session.eval("sma(close, 3)", data={name!r})']
    assert err.valid[0] in str(err) and err.parameter == "handle" and err.value == first["handle"]
    assert errors.from_dict(json.loads(json.dumps(err.to_dict()))).to_dict() == err.to_dict()
    again = s.eval("sma(close, 3)")  # following the recipe gives the same handle and the same numbers
    assert again["handle"] == first["handle"] and again["sha256"] == first["sha256"]
    assert s.series(first["handle"]).shape == (300,)


def test_a_dropped_handle_is_expired_not_unknown_and_an_invented_one_is_unknown():
    s = session()
    h = s.eval("sma(close, 3)")["handle"]
    s.drop(h)
    with pytest.raises(errors.RaptorBTError) as e:
        s.series(h)
    assert e.value.code == "handle_expired" and "dropped" in e.value.message
    with pytest.raises(errors.RaptorBTError) as e:
        s.series("h_00000000aa")
    assert e.value.code == "unknown_handle" and isinstance(e.value, ValueError)
    keep = s.eval("sma(close, 4)")["handle"]
    with pytest.raises(errors.RaptorBTError) as e:
        s.series(keep[:-1] + ("0" if keep[-1] != "0" else "1"))
    assert e.value.code == "unknown_handle" and keep in e.value.valid


def test_max_bytes_evicts_the_oldest_and_never_the_data():
    data_bytes = 300 * 8 * 6  # six columns
    s = Session(max_bytes=data_bytes + 2 * 300 * 8)
    s.load(bars(), bar_stamp="close")
    a = s.eval("sma(close, 3)")["handle"]
    b = s.eval("sma(close, 4)")["handle"]
    c = s.eval("sma(close, 5)")["handle"]
    assert s.handles() == [b, c] and s._used == data_bytes + 2 * 300 * 8
    assert s.dataset().rows == 300
    with pytest.raises(errors.RaptorBTError) as e:
        s.series(a)
    assert e.value.code == "handle_expired" and "max_bytes" in e.value.message


def test_a_result_larger_than_the_budget_is_refused_and_nothing_is_evicted():
    s = Session(max_bytes=300 * 8 * 6 + 300 * 8 + 10)
    s.load(bars(), bar_stamp="close")
    keep = s.eval("sma(close, 3)")["handle"]
    with pytest.raises(errors.RaptorBTError) as e:
        s.eval("macd(close)")  # three outputs
    err = e.value
    assert err.code == "session_memory_exceeded" and err.parameter == "max_bytes"
    assert err.valid and keep in s.handles()
    with pytest.raises(errors.RaptorBTError) as e:
        Session(max_bytes=100).load(bars(), bar_stamp="close")
    assert e.value.code == "session_memory_exceeded"


def test_a_handle_the_expression_reads_is_not_evicted_by_its_own_result():
    s = session(max_handles=2)
    a = s.eval("sma(close, 3)")["handle"]
    b = s.eval("sma(close, 4)")["handle"]
    r = s.eval(f"{a} + 1")  # a is newest when the result lands; b is the one to go
    assert s.handles() == [a, r["handle"]] and b not in s.handles()


def test_session_limits_are_checked():
    for kw in ({"max_bytes": 0}, {"max_handles": 0}, {"max_bytes": 1.5}, {"max_handles": True}):
        with pytest.raises(errors.RaptorBTError):
            Session(**kw)
    s = Session()
    assert (s.max_bytes, s.max_handles) == (1 << 30, 256)


def test_eval_before_load_and_bad_text_are_typed():
    with pytest.raises(errors.RaptorBTError) as e:
        Session().eval("close")
    assert e.value.code == "missing_field" and e.value.parameter == "data"
    s = session()
    with pytest.raises(errors.ExpressionError) as e:
        s.eval("smaa(close, 3)")
    assert e.value.code == "expr_unknown_function" and "sma" in e.value.valid
    with pytest.raises(errors.RaptorBTError) as e:
        s.eval(5)
    assert e.value.code == "expr_type"
    with pytest.raises(errors.RaptorBTError):
        s.eval("close", data="nope")


def test_an_at_expression_needs_and_uses_the_loaded_stamp():
    day = 86_400 * 10**9
    n = 60
    ts = 19_000 * day + np.arange(1, n + 1, dtype=np.int64) * (day // 4)
    s = Session()
    s.load({"timestamps": ts, "close": np.arange(1.0, n + 1)}, bar_stamp="close", bar_ns=day // 4)
    r = s.eval('at("1d", close)')
    assert r["warmup"] is None and r["warmup_by_clock"] == {"base": 0, "1d": 0}
    direct = expressions.expr('at("1d", close)').eval(
        {"timestamps": ts, "close": np.arange(1.0, n + 1)}, bar_stamp="close", bar_ns=day // 4
    )
    assert same(s.series(r["handle"]), direct)
    assert "flat_windows" not in r["flags"]


# --------------------------------------------------------------------------------- statistics


def test_stats_run_on_handles_and_cut_the_warm_up():
    n = 400
    rng = np.random.default_rng(5)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    s = Session()
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64) * MIN, "close": 100.0 + x}, bar_stamp="close")
    h = s.eval("close - sma(close, 20)")["handle"]
    r = s.stats("stationarity", h)
    assert r["reading"] == "stationary" and r["analysis"] == {"handles": [h], "trimmed_leading": 19}
    direct = rb.stats.stationarity(s.series(h)[19:])
    assert r["adf"] == direct["adf"]
    a, b = s.eval("close")["handle"], s.eval("sma(close, 3)")["handle"]
    c = s.stats("cointegration", a, b)
    assert c["analysis"]["trimmed_leading"] == 2 and "analysis" in c


def test_stats_refuse_what_they_cannot_run_with_typed_errors():
    s = session()
    h = s.eval("sma(close, 3)")["handle"]
    for call in (
        lambda: s.stats("event_study", h),
        lambda: s.stats("stationarity"),
        lambda: s.stats("stationarity", h, h),
        lambda: s.stats("audit_bars", h),
        lambda: s.stats("stationarity", "h_00000000aa"),
    ):
        with pytest.raises(errors.RaptorBTError):
            call()


def test_audit_bars_reads_the_loaded_bars():
    s = Session()
    s.load(bars(120), bar_stamp="close", bar_ns=MIN)
    r = s.stats("audit_bars")
    assert "clean" in r
    s2 = Session()
    s2.load(bars(120), bar_stamp="close")
    with pytest.raises(errors.RaptorBTError) as e:
        s2.stats("audit_bars")
    assert e.value.parameter == "interval_seconds"


# --------------------------------------------------------------------- functions and describe


def test_functions_search_names_talib_names_families_and_words():
    names = lambda q: [f["name"] for f in expressions.functions(q)]  # noqa: E731
    assert names("rsi")[0] == "rsi" and names("RSI")[0] == "rsi"
    assert "cdldoji" in names("doji")
    assert {"ma", "sma", "ema"} <= set(names("moving average")[:4])
    assert set(names("relative strength")) >= {"rsi"}
    assert "rsi" in names("momentum") and names("zzzz nothing") == []
    assert names(None) == sorted(names(None))
    row = expressions.functions("sma")[0]
    assert set(row) == {"name", "family", "signature", "outputs", "talib", "doc"} and row["talib"] == "SMA"


def test_describe_adds_an_example_a_warmup_a_reference_and_conventions():
    d = expressions.describe("rsi")
    assert d["example"] == "rsi(close, 14)" and d["warmup"] == 14
    assert d["reference"] == "fixture" and d["relation"] == "defined_difference"
    assert any("NaN" in c for c in d["conventions"]) and any("causal" in c for c in d["conventions"])
    assert d["signature"].startswith("rsi(")


def test_every_function_has_an_example_that_compiles_and_runs():
    d = bars(400)
    failures = []
    for f in expressions.functions():
        text = expressions.describe(f["name"])["example"]
        try:
            e = expressions.expr(text)
            out = e.eval_panel({k: np.column_stack([v] * 3) for k, v in d.items() if k != "timestamps"}) if e.is_panel else e.eval(d)
            first = next(iter(out.values())) if isinstance(out, dict) else out
            assert len(first) == 400
        except Exception as e:  # noqa: BLE001 - collect them all
            failures.append((f["name"], text, str(e)[:80]))
    assert failures == []


# ----------------------------------------------------------------- the root stays file-free


def test_importing_raptorbt_does_not_import_analysis_or_the_command_line():
    code = (
        "import sys\n"
        "import raptorbt\n"
        "import raptorbt.expressions, raptorbt.stats, raptorbt.strategy, raptorbt.skills, raptorbt.errors\n"
        "banned = [m for m in sys.modules if m == 'raptorbt.analysis' or m.startswith('raptorbt.analysis.') or m == 'raptorbt.__main__']\n"
        "assert banned == [], banned\n"
        "assert not hasattr(raptorbt, 'analysis') and 'analysis' not in raptorbt.__all__\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_strategy_run_does_not_import_analysis_either():
    code = (
        "import sys, numpy as np\n"
        "import raptorbt as rb\n"
        "n = 30\n"
        "c = 100 + np.arange(n, dtype=float)\n"
        "d = {'timestamps': np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9, 'open': c, 'high': c + 1, 'low': c - 1, 'close': c, 'volume': np.ones(n)}\n"
        "class S(rb.Strategy):\n"
        "    def on_start(self, ctx):\n"
        "        self.h = ctx.indicators.expr('sma(close, 3)', instrument='A')\n"
        "rb.run_strategies([S()], {'A': d}, bar_stamp='close')\n"
        "assert not [m for m in sys.modules if m.startswith('raptorbt.analysis') or m == 'raptorbt.__main__']\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# ------------------------------------------------------------------------------------ panels

DAY = 86_400 * 10**9


def panel_data(n=120, m=30, seed=1, drift=True):
    """Close prices of ``m`` names whose daily drifts persist, so a momentum factor has an IC."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.001, m) if drift else np.zeros(m)
    close = 100.0 * np.exp(np.cumsum(mu + rng.normal(0.0, 0.01, (n, m)), axis=0))
    return {"timestamps": np.arange(1, n + 1, dtype=np.int64) * DAY, "close": close}


def names_of(m, width=3):
    return [f"N{j:0{width}d}" for j in range(m)]


def psession(n=120, m=30, **kw):
    s = Session(**kw)
    s.load_panel(panel_data(n, m), bar_stamp="close", instruments=names_of(m))
    return s


def test_load_panel_needs_a_bar_stamp_checks_shapes_and_never_modifies_the_input():
    d = panel_data(40, 5)
    before = {k: v.copy() for k, v in d.items()}
    with pytest.raises(errors.RaptorBTError) as e:
        Session().load_panel(d)
    assert e.value.code == "missing_bar_stamp"
    s = Session()
    facts = s.load_panel(d, bar_stamp="close", instruments=names_of(5), name="px")
    assert facts["kind"] == "panel" and facts["instruments"] == 5 and facts["rows"] == 40 and facts["fields"] == ["close"]
    assert all(np.array_equal(d[k], before[k]) for k in d)
    for bad, code in (
        ({**d, "close": d["close"][:, :0]}, "invalid_array"),
        ({**d, "close": d["close"][:10]}, "length_mismatch"),
        ({**d, "close": d["close"][0]}, "invalid_array"),
        ({k: v for k, v in d.items() if k != "timestamps"}, "missing_field"),
    ):
        with pytest.raises(errors.RaptorBTError) as e:
            Session().load_panel(bad, bar_stamp="close")
        assert e.value.code == code, (code, e.value)
    for names, why in ((["a", "a", "b", "c", "d"], "distinct"), (["x" * 65] + names_of(4), "64"), (names_of(4), "4 instruments")):
        with pytest.raises(errors.RaptorBTError):
            Session().load_panel(d, bar_stamp="close", instruments=names)


def test_a_mapping_of_instruments_is_aligned_with_absent_bars():
    ts = np.arange(1, 61, dtype=np.int64) * DAY
    rng = np.random.default_rng(3)

    def one(skip=0):
        c = 100.0 + np.cumsum(rng.normal(0, 1, 60 - skip))
        return {"timestamps": ts[skip:], "close": c}

    s = Session()
    facts = s.load_panel({"AAA": one(), "BBB": one(), "CCC": one(20)}, bar_stamp="close")
    assert facts["rows"] == 60 and facts["instruments"] == 3
    r = s.eval("sma(close, 5)")
    assert r["shape"] == [60, 3] and r["coverage"]["min"] == 2 and r["coverage"]["max"] == 3
    only = Session()
    only.load_panel({"AAA": one(), "BBB": one(), "CCC": one(20)}, bar_stamp="close", instruments=["CCC", "AAA"])
    assert only.eval("close")["last"]["known"] == 2
    with pytest.raises(errors.RaptorBTError) as e:
        Session().load_panel({"AAA": one()}, bar_stamp="close", instruments=["AAA", "ZZZ"])
    assert e.value.code == "unknown_symbol" and e.value.valid == ["AAA"]


def test_a_panel_summary_names_the_instruments_at_both_ends():
    s = psession()
    r = s.eval("rank(roc(close, 20))")
    assert r["shape"] == [120, 30] and r["warmup"] == 20 and r["kind"] == "number"
    assert r["nan"] == {"leading": 20, "cells": 20 * 30}
    assert r["coverage"] == {"min": 30, "median": 30.0, "max": 30}
    want = expressions.expr("rank(roc(close, 20))").eval_panel({"close": panel_data()["close"]})
    last = want[-1]
    order = np.argsort(last, kind="stable")
    names = names_of(30)
    assert [n for n, _ in r["last"]["bottom"]] == [names[j] for j in order[:5]]
    assert [n for n, _ in r["last"]["top"]] == [names[j] for j in np.argsort(-last, kind="stable")[:5]]
    assert r["last"]["index"] == 119 and r["last"]["known"] == 30 and r["last"]["timestamp"] == 120 * DAY
    assert r["dist"]["min"] == 0.0 and r["dist"]["max"] == 1.0 and r["sha256"] == asum.sha256_of(s.series(r["handle"]))
    assert same(s.series(r["handle"]), want)


def test_panel_handles_chain_and_are_deterministic():
    s = psession()
    a = s.eval("rank(roc(close, 20))")["handle"]
    z = s.eval(f"zscore_xs({a})")
    nested = s.eval("zscore_xs(rank(roc(close, 20)))")
    assert z["warmup"] == nested["warmup"] == 20
    assert same(s.series(z["handle"]), s.series(nested["handle"]))
    assert same(s.series(z["handle"], instruments=["N003", "N001"]), s.series(z["handle"])[:, [3, 1]])
    assert psession().eval("rank(roc(close, 20))")["handle"] == a  # the same work, the same handle anywhere
    other = Session()
    other.load_panel(panel_data(120, 30), bar_stamp="close", instruments=list(reversed(names_of(30))))
    assert other.eval("rank(roc(close, 20))")["handle"] != a  # other column order, other data
    assert s.describe_handle(z["handle"])["depends_on"] == [a]


def test_a_bool_panel_counts_and_names_what_is_true():
    s = psession()
    r = s.eval("rank(roc(close, 20)) > 0.8")
    assert r["kind"] == "bool" and "dist" not in r and r["known"] == 100 * 30
    last = s.series(r["handle"])[-1]
    assert r["last"]["true"] == int(last.sum()) and len(r["last"]["names"]) == min(5, int(last.sum()))
    assert r["last"]["names"] == [n for n, v in zip(names_of(30), last) if v == 1.0][:5]


def test_a_series_expression_on_bars_and_a_panel_expression_do_not_mix():
    s = psession()
    with pytest.raises(errors.RaptorBTError) as e:
        s.series("h_0000000000", instruments=["N000"])
    assert e.value.code == "unknown_handle"
    b = Session()
    b.load(bars(), bar_stamp="close")
    with pytest.raises(errors.RaptorBTError) as e:
        b.eval("rank(close)")
    assert e.value.code == "expr_unsupported_here"
    with pytest.raises(errors.RaptorBTError) as e:
        b.factor("rank(close)")
    assert e.value.code == "invalid_parameter"
    h = s.eval("sma(close, 5)")["handle"]
    with pytest.raises(errors.RaptorBTError) as e:
        s.stats("stationarity", h)  # a panel is not one series
    assert e.value.code == "invalid_parameter"
    with pytest.raises(errors.RaptorBTError) as e:
        s.series(h, instruments=["N000", "NOPE"])
    assert e.value.code == "unknown_symbol" and e.value.parameter == "instruments" and e.value.value == "NOPE"
    assert e.value.valid and isinstance(e.value, KeyError)
    with pytest.raises(errors.RaptorBTError) as e:
        b.series(b.eval("sma(close, 3)")["handle"], instruments=["x"])
    assert e.value.code == "invalid_parameter"


def test_an_unknown_instrument_is_typed_and_offers_the_nearest():
    s = psession()
    h = s.eval("zscore_xs(close)")["handle"]
    with pytest.raises(errors.RaptorBTError) as e:
        s.series(h, instruments="N0003")
    assert e.value.code == "unknown_symbol" and "N003" in e.value.valid
    assert s.series(h, instruments="N003").shape == (120, 1)


def test_the_factor_summary_is_factor_report_in_numbers():
    s = psession(200, 40)
    h = s.eval("zscore_xs(roc(close, 20))")["handle"]
    f = s.factor(h, horizons=(1, 5, 10), quantiles=5)
    rep = expressions.factor_report(s.series(h), panel_data(200, 40)["close"], horizons=(1, 5, 10), quantiles=5)
    r = asum.show
    assert f["no_costs"] is True and f["kind"] == "report" and f["factor"] == h and f["lag"] == 1
    assert set(f["horizons"]) == {"1", "5", "10"}
    for hz in (1, 5, 10):
        row, got = rep["horizons"][hz], f["horizons"][str(hz)]
        assert got == {
            "n_scored": row["n_scored"], "ic_mean": r(row["ic_mean"]),
            "t_nonoverlap": r(row["t_nonoverlap"]), "p_nonoverlap": r(row["p_nonoverlap"]),
            "n_independent": row["n_independent"], "names_left_share": r(row["names_left_share"]),
            "t_hac": r(row["t_hac"]), "hac_lags": row["hac_lags"], "t_naive": r(row["t_naive"]),
            "hit_rate": r(row["hit_rate"]), "long_short_mean": r(row["long_short_mean"]),
            "long_short_t_nonoverlap": r(row["long_short_t_nonoverlap"]),
            "long_short_t_hac": r(row["long_short_t_hac"]),
        }
        assert got["hac_lags"] == hz - 1  # the overlap correction that was used
        assert same(s.series(f"{f['handle']}_ic_{hz}"), row["ic"])
        assert same(s.series(f["handle"])[f"long_short_{hz}"], row["long_short"])
        assert same(s.series(f["handle"])[f"quantile_returns_by_date_{hz}"], row["quantile_returns_by_date"])
    assert f["headline"] == "t_nonoverlap" and "over-reject" in f["t_hac_note"]
    assert f["flags"] == [x if len(x) <= 160 else x[:157] + "..." for x in rep["flags"][:3]]
    assert f["assumptions"] == rep["assumptions"][:2]
    assert f["turnover"] == {"top": r(rep["turnover"][-1]), "bottom": r(rep["turnover"][0])}
    assert f["names"] == {"min": 40, "median": 40.0, "max": 40} and f["dates"] == 200
    assert same(s.series(f"{f['handle']}_coverage"), rep["coverage"])
    assert size(f) <= asum.OUTPUT_BYTES
    assert f["horizons"]["5"]["ic_mean"] > 0  # the planted drift persists, so momentum scores


def test_the_factor_summary_with_many_horizons_and_full_flags_fits_the_cap():
    # Measured: 8 horizons, a split and two leavers (three flags): 1,898 bytes with horizons 1 and
    # 2 kept and the rest named in truncated_horizons; one horizon with the same flags is 1,378.
    n, m = 300, 40
    data = panel_data(n, m)
    close = data["close"]
    close[150:, 3] /= 10.0
    close[200:, 5] = np.nan
    close[220:, 6] = 0.0
    s = Session()
    s.load_panel(data, bar_stamp="close", instruments=names_of(m))
    h = s.eval("zscore_xs(roc(close, 20))")["handle"]
    f = s.factor(h, horizons=(1, 2, 3, 5, 10, 20, 40, 60), quantiles=5)
    assert size(f) <= asum.OUTPUT_BYTES
    assert len(f["flags"]) == 3 and f["assumptions"]
    assert f["horizons"] and f["truncated_horizons"]
    assert set(f["horizons"]) | set(f["truncated_horizons"]) == {"1", "2", "3", "5", "10", "20", "40", "60"}


def test_a_factor_can_be_an_expression_and_a_repeat_is_not_recomputed(monkeypatch):
    s = psession(200, 40)
    f = s.factor("zscore_xs(roc(close, 20))", horizons=5)
    assert list(f["horizons"]) == ["5"]
    calls = []
    real = expressions.factor_report
    monkeypatch.setattr(expressions, "factor_report", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert s.factor(f["factor"], horizons=5) == f and calls == []
    s.factor(f["factor"], horizons=6)
    assert calls == [1]
    assert s.factor("zscore_xs(roc(close, 20))", horizons=5)["handle"] == f["handle"]


def test_factor_refuses_what_it_cannot_score_with_typed_errors():
    s = psession(200, 40)
    h = s.eval("zscore_xs(roc(close, 20))")["handle"]
    f = s.factor(h, horizons=5)
    for call, code in (
        (lambda: s.factor(f["handle"]), "invalid_parameter"),            # a report is not a factor
        (lambda: s.factor(f"{f['handle']}_ic_5"), "invalid_parameter"),
        (lambda: s.factor("macd(close)"), "invalid_parameter"),          # several outputs
        (lambda: s.factor("h_00000000aa"), "unknown_handle"),
        (lambda: s.factor(h, horizons=500), "insufficient_observations"),
        (lambda: s.factor(h, quantiles=1), "invalid_parameter"),
    ):
        with pytest.raises(errors.RaptorBTError) as e:
            call()
        assert e.value.code == code, (code, e.value.code)
    with pytest.raises(errors.RaptorBTError) as e:
        s.eval(f"sma({f['handle']}_coverage, 3)")  # a report is not a series
    assert e.value.code == "unknown_handle"
    nc = Session()
    nc.load_panel({"timestamps": panel_data(80, 12)["timestamps"], "open": panel_data(80, 12)["close"]}, bar_stamp="close")
    with pytest.raises(errors.RaptorBTError) as e:
        nc.factor("zscore_xs(open)")
    assert e.value.code == "missing_field"


def test_panel_summaries_fit_their_caps_on_two_thousand_long_names():
    n, m = 90, 2000
    names = [f"NSE:{'X' * 59}{j:01d}"[:64] if j < 10 else f"NSE:{'Y' * 50}{j:010d}" for j in range(m)]
    assert len(set(names)) == m and max(map(len, names)) <= 64
    s = Session()
    s.load_panel(panel_data(n, m), bar_stamp="close", instruments=names)
    for text in (
        "rank(roc(close, 20))", "zscore_xs(close) * 1e300", "rank(roc(close, 20)) > 0.5", "macd(close)",
        "winsorize_xs(roc(close, 5))", "close * 1e-300",
    ):
        r = s.eval(text)
        outs = r["outputs"] if "outputs" in r else [r]
        assert size(r) <= asum.CALL_BYTES, text
        assert all(size(o) <= asum.OUTPUT_BYTES for o in outs), text
    wide = s.factor("zscore_xs(roc(close, 5))", horizons=list(range(1, 25)), min_names=100)
    assert size(wide) <= asum.OUTPUT_BYTES and wide["truncated_horizons"]
    assert len(wide["horizons"]) + len(wide["truncated_horizons"]) == 24
    assert s.series(wide["handle"])["coverage"].shape == (n,)
    narrow = s.factor("zscore_xs(roc(close, 5))", horizons=(1, 5), min_names=100)
    assert size(narrow) <= asum.OUTPUT_BYTES and "truncated_horizons" not in narrow


def test_a_panel_expression_with_at_reads_the_loaded_timestamps():
    d = panel_data(60, 8)
    d["timestamps"] = 19_000 * DAY + np.arange(1, 61, dtype=np.int64) * (DAY // 4)
    s = Session()
    s.load_panel(d, bar_stamp="close", bar_ns=DAY // 4, instruments=names_of(8))
    r = s.eval('rank(at("1d", close))')
    assert r["warmup"] is None and r["shape"] == [60, 8]
    direct = expressions.expr('rank(at("1d", close))').eval_panel(d, bar_stamp="close", bar_ns=DAY // 4)
    assert same(s.series(r["handle"]), direct)


def test_the_panel_flags_count_cells_and_names():
    d = panel_data(80, 10)
    d["close"][:, 4] = 50.0  # one name that never trades
    s = Session()
    s.load_panel(d, bar_stamp="close", instruments=names_of(10))
    r = s.eval("rsi(close, 14)")
    assert r["flags"]["flat_windows"] == 80 - 14
    assert "flat_windows" not in s.eval("rank(close)")["flags"]
    d2 = panel_data(80, 10)
    d2["close"][10, 2] = -1.0
    s2 = Session()
    s2.load_panel(d2, bar_stamp="close", instruments=names_of(10))
    assert s2.eval("zscore_xs(close)")["flags"]["nonpositive_input"] == 1
    late = Session()
    late.load_panel(
        {"A": {"timestamps": np.arange(1, 41, dtype=np.int64), "close": np.arange(1.0, 41)},
         "B": {"timestamps": np.arange(11, 41, dtype=np.int64), "close": np.arange(1.0, 31)}},
        bar_stamp="close",
    )
    assert late.eval("close")["coverage"] == {"min": 1, "median": 2.0, "max": 2}


def test_the_capabilities_list_names_things_that_exist():
    caps = rb.capabilities()["analysis"]
    assert caps and all(isinstance(n, str) for n in caps)
    import importlib

    for name in caps:
        if name.startswith("python -m raptorbt "):
            continue
        module, attr = name.rsplit(".", 1)
        try:
            obj = importlib.import_module(module)
        except ModuleNotFoundError:  # a method: raptorbt.analysis.Session.eval
            module, cls = module.rsplit(".", 1)
            obj = getattr(importlib.import_module(module), cls)
        assert hasattr(obj, attr), name
    assert {"python -m raptorbt eval", "python -m raptorbt factor", "raptorbt.analysis.Session.factor"} <= set(caps)
