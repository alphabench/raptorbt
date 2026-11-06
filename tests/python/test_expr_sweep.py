"""``Expr.sweep``: one expression under many bindings of its ``$name`` constants.

The engine's own proof is Rust (``src/quant/eval/sweep.rs``: a proptest of bit-identity against
evaluating each binding alone). These tests prove what Python adds around it: every binding
equals ``expr(text, params=binding).eval`` bit for bit (NaN data, ``at(...)``, panels), shared
nodes run once, the order of parameters and of bindings changes no number, the size bound
refuses with the size, the thread count changes nothing, and the agent surfaces
(``Session.sweep``, ``python -m raptorbt sweep``) report ``n_trials``.
"""

import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import raptorbt as rb
from raptorbt.analysis import Session
from raptorbt.errors import ExpressionError, RaptorBTError

MIN = 60 * 10**9
HOUR = 3600 * 10**9
DAY = 24 * HOUR
CROSS = "sma(close, $n) > sma(close, $m)"


def bits(a):
    """The bit pattern of a float64 array with every NaN made one NaN."""
    a = np.array(a, dtype=np.float64)
    a[np.isnan(a)] = np.nan
    return a.view(np.uint64)


def prices(n=400, seed=11, holes=()):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    for i in holes:
        close[i] = np.nan
    return {
        "open": close + rng.normal(0.0, 0.3, n),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": rng.uniform(1_000.0, 5_000.0, n),
    }


def alone(text, params, data, **kw):
    return rb.expr(text, params=params).eval(data, **kw)


def raises(code, fn, parameter=None):
    with pytest.raises(RaptorBTError) as info:
        fn()
    assert info.value.code == code, info.value
    if parameter is not None:
        assert info.value.parameter == parameter
    return info.value


# ---------------------------------------------------------------- bit-identity


@pytest.mark.parametrize(
    "text",
    [
        CROSS,
        "ema(close, $n) - sma(close, $m) / sma(close, $n)",
        "(rsi(close, $n) < 30) and (close > ema(close, $m))",
        "sma(volume, $n) * close ** 2 - ($m + 1)",
    ],
)
def test_every_binding_equals_its_own_evaluation_with_nan_data(text):
    data = prices(holes=(3, 40, 41, 250))
    swept = rb.expr(text).sweep(data, n=[2, 5, 14], m=[3, 21, 50])
    assert len(swept) == swept.n_trials == 9
    for point in swept:
        assert bits(point.value).tolist() == bits(alone(text, point.params, data)).tolist()


def test_results_come_back_in_input_order_with_their_params():
    data = prices()
    swept = rb.expr(CROSS).sweep(data, n=[5, 10], m=[50, 100])
    # the last parameter varies fastest
    assert swept.params == [
        {"n": 5, "m": 50},
        {"n": 5, "m": 100},
        {"n": 10, "m": 50},
        {"n": 10, "m": 100},
    ]
    pts = [{"n": 10, "m": 100}, {"n": 5, "m": 50}]
    listed = rb.expr(CROSS).sweep(data, points=pts)
    assert listed.params == pts
    for p in listed:
        assert bits(p.value).tolist() == bits(alone(CROSS, p.params, data)).tolist()


def test_a_multi_output_expression_gives_a_dict_per_binding():
    data = prices()
    swept = rb.expr("bbands(close, $n)").sweep(data, n=[10, 20])
    assert swept.outputs == tuple(rb.expr("bbands(close, 10)").outputs)
    for p in swept:
        want = alone("bbands(close, $n)", p.params, data)
        assert set(p.value) == set(want)
        for k in want:
            assert bits(p.value[k]).tolist() == bits(want[k]).tolist()


def test_a_base_param_is_kept_and_the_sweep_overrides_it():
    data = prices()
    e = rb.expr(CROSS, params={"m": 50})
    swept = e.sweep(data, n=[5, 10])
    assert swept.params == [{"n": 5, "m": 50}, {"n": 10, "m": 50}]
    assert bits(swept[1].value).tolist() == bits(alone(CROSS, {"n": 10, "m": 50}, data)).tolist()
    over = e.sweep(data, n=[5], m=[100])
    assert over.params == [{"n": 5, "m": 100}]


def at_bars(days=6, seed=3):
    rng = np.random.default_rng(seed)
    ts = np.array([(20_000 + d) * DAY + (k + 1) * HOUR for d in range(days) for k in range(24)], dtype=np.int64)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, len(ts)))
    return {"timestamps": ts, "close": close, "open": close, "high": close + 1, "low": close - 1,
            "volume": np.ones(len(ts))}


def test_an_expression_with_at_sweeps_like_each_binding_alone():
    # a `$n` inside the at(...) body is bound like any other: the body shares the environment
    data = at_bars()
    text = 'close > at("1d", sma(close, $n))'
    swept = rb.expr(text).sweep(data, n=[1, 2, 3], bar_stamp="close", bar_ns=HOUR)
    assert swept.n_trials == 3
    for p in swept:
        want = alone(text, p.params, data, bar_stamp="close", bar_ns=HOUR)
        assert bits(p.value).tolist() == bits(want).tolist()
    assert not np.array_equal(bits(swept[0].value), bits(swept[2].value))


def test_a_panel_expression_sweeps_one_binding_at_a_time():
    rng = np.random.default_rng(4)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, (60, 7)), axis=0)
    close[10:14, 2] = np.nan
    text = "rank(sma(close, $n))"
    swept = rb.expr(text).sweep({"close": close}, n=[2, 5, 9])
    assert swept.n_trials == 3 and swept.n_kernel_steps is None
    for p in swept:
        want = rb.expr(text, params=p.params).eval_panel({"close": close})
        assert bits(p.value).tolist() == bits(want).tolist()


# ---------------------------------------------------------------- shared nodes


def test_a_shared_node_is_computed_once():
    data = prices()
    swept = rb.expr(CROSS).sweep(data, n=[5, 10, 20], m=[50, 100])
    # sma(close, 5), 10, 20, 50 and 100: five kernels for six bindings that would run twelve
    assert swept.n_kernel_steps == 5
    assert swept.independent_kernel_steps == 12


def test_a_node_two_parameters_spell_alike_is_one_node():
    data = prices()
    swept = rb.expr(CROSS).sweep(data, n=[5, 10], m=[5, 10])
    # (5, 5) and (10, 10) are one node alone (`sma > sma`), the other two bindings two each
    assert swept.n_kernel_steps == 2 and swept.independent_kernel_steps == 6


def test_a_node_no_parameter_touches_is_computed_once():
    data = prices()
    text = "ema(close, 30) + sma(close, $n)"
    swept = rb.expr(text).sweep(data, n=[3, 4, 5])
    assert swept.n_kernel_steps == 4  # one ema and three smas, not six kernels
    for p in swept:
        assert bits(p.value).tolist() == bits(alone(text, p.params, data)).tolist()


# ---------------------------------------------------------------- order and repeats


def test_the_order_of_parameters_changes_no_number():
    data = prices()
    a = rb.expr(CROSS).sweep(data, n=[5, 10, 20], m=[50, 100])
    b = rb.expr(CROSS).sweep(data, m=[50, 100], n=[5, 10, 20])
    c = rb.expr(CROSS).sweep(data, grid={"m": [50, 100], "n": [5, 10, 20]})
    by_params = {tuple(sorted(p.params.items())): bits(p.value).tolist() for p in a}
    for other in (b, c):
        assert len(other) == len(a)
        for p in other:
            assert bits(p.value).tolist() == by_params[tuple(sorted(p.params.items()))]


def test_the_order_of_listed_points_only_reorders_the_results():
    data = prices()
    pts = [{"n": 5, "m": 50}, {"n": 10, "m": 100}, {"n": 20, "m": 50}]
    fwd = rb.expr(CROSS).sweep(data, points=pts)
    rev = rb.expr(CROSS).sweep(data, points=pts[::-1])
    for f, r in zip(fwd, rev[::-1]):
        assert f.params == r.params
        assert bits(f.value).tolist() == bits(r.value).tolist()


def test_duplicate_bindings_are_each_returned_and_counted():
    data = prices()
    swept = rb.expr(CROSS).sweep(data, points=[{"n": 5, "m": 50}, {"n": 5, "m": 50}, {"n": 6, "m": 50}])
    assert swept.n_trials == 3
    assert bits(swept[0].value).tolist() == bits(swept[1].value).tolist()
    assert swept.n_kernel_steps == 3  # sma 5, 6 and 50
    dup_axis = rb.expr(CROSS).sweep(data, n=[5, 5], m=[50])
    assert dup_axis.n_trials == 2
    assert bits(dup_axis[0].value).tolist() == bits(dup_axis[1].value).tolist()
    assert dup_axis[0].value is not dup_axis[1].value  # separate arrays: editing one leaves the other


def test_a_sweep_does_not_modify_its_input():
    data = prices(holes=(7,))
    before = {k: v.copy() for k, v in data.items()}
    rb.expr(CROSS).sweep(data, n=[5], m=[50])
    for k in data:
        assert bits(data[k]).tolist() == bits(before[k]).tolist()


# ---------------------------------------------------------------- refusals


def test_a_sweep_past_the_bound_is_refused_with_its_size():
    data = prices()
    e = rb.expr(CROSS)
    err = raises("sweep_too_large", lambda: e.sweep(data, n=[2, 3, 4], m=[5, 6], max_combinations=5), "max_combinations")
    assert err.value == 6 and "6 combinations" in err.message and "at most 5" in err.message
    err = raises("sweep_too_large", lambda: e.sweep(data, n=[2, 3], m=[5, 6], max_bytes=10_000), "max_bytes")
    assert isinstance(err.value, int) and err.value > 10_000
    assert str(err.value) in err.message and "GiB" in err.message
    # a grid whose product does not fit a machine word is refused, not wrapped
    huge = {f"p{i}": list(range(1, 200)) for i in range(12)}
    text = " + ".join(f"sma(close, $p{i})" for i in range(12))
    raises("sweep_too_large", lambda: rb.expr(text).sweep(data, grid=huge), "max_combinations")


def test_the_default_bound_is_the_documented_one():
    assert rb._raptorbt.expr_sweep_limits() == (10_000, 4 << 30)


def test_a_parameter_left_unbound_is_refused():
    data = prices()
    e = rb.expr(CROSS)
    err = raises("expr_unbound_parameter", lambda: e.sweep(data, n=[5, 10]))
    assert "$m" in err.message and isinstance(err, ExpressionError)
    raises("expr_unbound_parameter", lambda: e.eval(data))
    raises("expr_unbound_parameter", lambda: e.stream())
    raises("expr_unbound_parameter", lambda: e.canonical)
    assert "unbound" in repr(e) and e.parameters == ("m", "n")


def test_a_parameter_the_text_never_reads_is_refused():
    data = prices()
    err = raises("invalid_parameter", lambda: rb.expr(CROSS).sweep(data, n=[5], m=[50], k=[1]), "k")
    assert "k" in err.message and set(err.valid) == {"m", "n"}
    raises("invalid_parameter", lambda: rb.expr(CROSS).sweep(data, points=[{"n": 5, "m": 50, "k": 1}]), "k")


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({}, "grid"),
        ({"n": 5, "m": [50]}, "n"),
        ({"n": [], "m": [50]}, "n"),
        ({"n": ["a"], "m": [50]}, "n"),
        ({"n": [float("nan")], "m": [50]}, "n"),
        ({"n": [True], "m": [50]}, "n"),
        ({"points": []}, "points"),
        ({"points": [{"n": 5, "m": 50}], "n": [5]}, "points"),
    ],
)
def test_a_malformed_space_is_refused(kwargs, parameter):
    raises("invalid_parameter", lambda: rb.expr(CROSS).sweep(prices(), **kwargs), parameter)


def test_a_value_out_of_range_is_the_compile_refusal():
    err = raises("expr_type", lambda: rb.expr(CROSS).sweep(prices(), n=[0], m=[50]))
    assert err.operation == "expr"


def test_a_missing_column_is_the_evaluators_refusal():
    raises("missing_field", lambda: rb.expr("sma(volume, $n)").sweep({"close": prices()["close"]}, n=[3]))


# ---------------------------------------------------------------- threads

_THREADS = """
import hashlib, sys
import numpy as np
import raptorbt as rb
rng = np.random.default_rng(1)
close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 5000))
close[100] = np.nan
swept = rb.expr("(sma(close, $n) > ema(close, $m)) and (rsi(close, $n) < 60)").sweep(
    {"close": close}, n=[3, 5, 8, 13, 21], m=[30, 40, 50, 60]
)
h = hashlib.sha256()
for p in swept:
    a = np.array(p.value)
    a[np.isnan(a)] = np.nan
    h.update(a.tobytes())
print(h.hexdigest(), swept.n_kernel_steps)
"""


def test_the_numbers_do_not_depend_on_the_thread_count():
    outs = set()
    for threads in ("1", "2", "8"):
        env = {**os.environ, "RAYON_NUM_THREADS": threads, "RAPTORBT_NO_VERSION_CHECK": "1"}
        done = subprocess.run([sys.executable, "-c", _THREADS], capture_output=True, text=True, env=env)
        assert done.returncode == 0, done.stderr
        outs.add(done.stdout.strip())
    assert len(outs) == 1, outs


# ---------------------------------------------------------------- the agent surfaces


def session(n=300, seed=3, **kw):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    s = Session(**kw)
    s.load({"timestamps": np.arange(1, n + 1, dtype=np.int64) * MIN, "close": close}, bar_stamp="close")
    return s


def test_session_sweep_gives_a_table_and_a_handle_per_binding():
    s = session()
    t = s.sweep(CROSS, n=[5, 10], m=[30, 60])
    assert t["n_trials"] == t["n_distinct"] == 4
    assert t["columns"] == ["handle", "n", "m", "score"] and t["swept"] == ["n", "m"]
    assert t["score"] == "true_rate" and t["order"] == "input"
    assert [r[1:3] for r in t["rows"]] == [[5, 30], [5, 60], [10, 30], [10, 60]]
    assert "deflated_sharpe" in t["note"] and t["kernel_steps"] == {"joint": 4, "independent": 8}
    for row in t["rows"]:
        series = s.series(row[0])
        assert row[3] == pytest.approx(float(np.mean(series[np.isfinite(series)])), rel=1e-5)
    assert s.eval("sma(close, 5) > sma(close, 30)")["handle"] == t["rows"][0][0]
    assert s.sweep(CROSS, n=[5, 10], m=[30, 60]) == t  # the same work again is the same table


def test_session_sweep_scores_by_a_statistic_or_an_expression():
    s = session()
    by_std = s.sweep("sma(close, $n)", n=[5, 20], score="std")
    assert by_std["score"] == "std" and by_std["rows"][0][2] != by_std["rows"][1][2]
    by_expr = s.sweep("sma(close, $n)", n=[5, 20], score="abs(close - sma(close, $n))")
    close = s.dataset().arrays["close"]
    want = np.nanmean(np.abs(close - s.series(by_expr["rows"][0][0])))
    assert by_expr["rows"][0][2] == pytest.approx(want, rel=1e-5)
    raises("expr_unknown_function", lambda: s.sweep("sma(close, $n)", n=[5], score="nonesuch(close)"))


def test_session_sweep_keeps_the_highest_scores_when_the_table_is_too_long():
    s = session(n=400)
    t = s.sweep("sma(close, $n) > sma(close, 100)", n=list(range(2, 140)))
    assert t["n_trials"] == 138 and t["omitted"] > 0 and t["order"] == "score, highest first"
    scores = [r[-1] for r in t["rows"] if r[-1] is not None]
    assert scores == sorted(scores, reverse=True)
    assert len(t["rows"]) + t["omitted"] == 138
    assert len(json.dumps(t, separators=(",", ":"))) <= 4096


def test_session_sweep_refusals():
    s = session()
    raises("expr_unbound_parameter", lambda: s.sweep(CROSS, n=[5]))
    raises("expr_type", lambda: s.sweep("macd(close, $n)", n=[12]))
    err = raises("session_memory_exceeded", lambda: session(max_handles=2).sweep(CROSS, n=[2, 3], m=[5, 6]), "max_handles")
    assert "4 distinct" in err.message


def cli(tmp_path, *args):
    env = {**os.environ, "RAPTORBT_NO_VERSION_CHECK": "1"}
    cmd = [sys.executable, "-m", "raptorbt", *args, "--cache-dir", str(tmp_path / "cache")]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tmp_path)


def test_the_command_line_sweeps_a_grid(tmp_path):
    n = 200
    rng = np.random.default_rng(2)
    npz = tmp_path / "bars.npz"
    np.savez(npz, timestamps=np.arange(1, n + 1, dtype=np.int64) * MIN, close=100.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
    done = cli(tmp_path, "sweep", CROSS, "--n", "5,10,20", "--m", "50,100", "--data", str(npz), "--bar-stamp", "close")
    assert done.returncode == 0, done.stdout + done.stderr
    out = json.loads(done.stdout)
    assert out["n_trials"] == 6 and out["columns"] == ["handle", "n", "m", "score"]
    again = cli(tmp_path, "sweep", CROSS, "--grid", "n=5,10,20", "--grid", "m=50,100")
    assert json.loads(again.stdout) == out  # the session kept the bars; same work, same table
    bad = cli(tmp_path, "sweep", CROSS, "--n", "5,10")
    assert bad.returncode == 2 and json.loads(bad.stdout)["code"] == "expr_unbound_parameter"
    stray = cli(tmp_path, "eval", "close", "--bogus", "5")
    assert stray.returncode == 2 and json.loads(stray.stdout)["code"] == "invalid_parameter"


def test_every_combination_is_tried_in_a_python_loop_too():
    """The reference the engine is held to: a plain loop over the product."""
    data = prices()
    swept = rb.expr(CROSS).sweep(data, n=[4, 8], m=[16, 32])
    for (n, m), p in zip(itertools.product([4, 8], [16, 32]), swept):
        assert p.params == {"n": n, "m": m}
        assert bits(p.value).tolist() == bits(alone(CROSS, {"n": n, "m": m}, data)).tolist()
