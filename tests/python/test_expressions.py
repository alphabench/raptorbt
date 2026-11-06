"""The Python surface of the expression language: raptorbt.expr, raptorbt.col, Expr.

The engine's own numbers are proved in Rust (kernels, evaluator, canonical form); these tests
prove what Python adds: the builder renders the same program as the parser, evaluation and
streaming agree through the binding, refusals are structured, and the surface follows the
function registry with no Python edit.
"""

import json

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import _raptorbt, errors
from raptorbt.errors import ExpressionError, RaptorBTError
from raptorbt.expressions import Expr

c = rb.col.close
NAN = float("nan")


def bits(a):
    """The bit pattern of a float64 array with every NaN made one NaN."""
    a = np.array(a, dtype=np.float64)
    a[np.isnan(a)] = np.nan
    return a.view(np.uint64)


def series(n=400, seed=11):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    spread = rng.uniform(0.1, 2.0, n)
    return {
        "open": close + rng.normal(0.0, 0.3, n),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.uniform(1_000.0, 5_000.0, n),
    }


def stream_all(e, data):
    """Feed every bar of `data` to a stream of `e`; one row per bar, as `eval` lays them out."""
    s = e.stream()
    n = len(next(iter(data.values())))
    rows = [s.update({k: v[i] for k, v in data.items()}) for i in range(n)]
    if isinstance(rows[0], dict):
        return {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return np.array(rows)


# ---------------------------------------------------------------- builder and parser


@pytest.mark.parametrize(
    ("built", "text"),
    [
        ((c.ema(14) < 30) & (c > c.sma(200)), "ema(close, 14) < 30 and close > sma(close, 200)"),
        ((c.sma(5) > c) | (c.sma(5) < 1), "sma(close, 5) > close or sma(close, 5) < 1"),
        (~(c.sma(5) > c), "not sma(close, 5) > close"),
        (c.sma(5) - c.sma(20), "sma(close, 5) - sma(close, 20)"),
        (c.ema(3).sma(4), "sma(ema(close, 3), 4)"),
        (2 * c + 1, "2 * close + 1"),
        (1 - c, "1 - close"),
        (c / rb.col.open, "close / open"),
        (c**2, "close ** 2"),
        (2**c, "2 ** close"),
        (-c, "-close"),
        (c - -5, "close - (-5)"),
        (rb.col.sma(7), "sma(close, 7)"),
        (c.sma(period=7), "sma(close, 7)"),
        (rb.col.high.sma(7), "sma(high, 7)"),
        (c.sma(5.0), "sma(close, 5)"),
        ((c >= 3) & (c != 4) & (c == 5) & (c <= 6), "close >= 3 and close != 4 and close == 5 and close <= 6"),
    ],
)
def test_builder_and_parser_give_the_same_canonical_text_and_hash(built, text):
    parsed = rb.expr(text)
    assert isinstance(built, Expr)
    assert built.canonical == parsed.canonical
    assert built.hash == parsed.hash


def test_spelling_does_not_change_the_program():
    assert rb.expr("close+open").hash == rb.expr("open + close").hash
    assert rb.expr("sma(5)").canonical == "sma(close, 5)"
    assert rb.expr("sma(close, 5)").hash != rb.expr("sma(close, 6)").hash


def test_parameters_are_bound_by_name():
    bound = rb.expr("sma(close, $n)", params={"n": 5})
    assert bound.canonical == "sma(close, 5)"
    assert bound.hash == rb.expr("sma(close, 5)").hash
    assert (bound > c).canonical == (rb.expr("sma(close, 5)") > c).canonical


def test_reflected_operators_keep_their_order():
    assert (1 - c).canonical == rb.expr("1 - close").canonical
    assert (1 - c).canonical != (c - 1).canonical
    assert (2**c).canonical != (c**2).canonical


def test_numpy_scalars_are_numbers_to_an_expression():
    assert (np.float64(2.5) * c).canonical == rb.expr("2.5 * close").canonical
    assert (c + np.int64(3)).canonical == rb.expr("close + 3").canonical


# ---------------------------------------------------------------- truth values


def test_a_truth_test_of_an_expression_is_refused():
    e = c > 1
    for use in (lambda: bool(e), lambda: e and e, lambda: not e, lambda: 1 < c < 3, lambda: (e if e else e)):
        with pytest.raises(ExpressionError) as info:
            use()
        assert info.value.code == "expr_type"
        assert info.value.operation == "Expr.__bool__"
        assert "&" in str(info.value)


def test_an_expression_is_not_hashable():
    with pytest.raises(TypeError):
        hash(c)


# ---------------------------------------------------------------- evaluation


def test_eval_is_arithmetic_on_the_columns():
    close = np.array([1.0, 2.0, 3.0])
    out = rb.expr("close * 2 + 1").eval({"close": close})
    assert out.tolist() == [3.0, 5.0, 7.0]  # 1*2+1, 2*2+1, 3*2+1


def test_division_by_zero_is_unknown_not_infinite():
    out = (rb.expr("close / (close - close)")).eval({"close": np.array([1.0, 2.0])})
    assert np.isnan(out).all()


@pytest.mark.parametrize(
    "text",
    [
        "close",
        "sma(close, 5)",
        "ema(close, 12) - ema(close, 26)",
        "sma(high, 3) < ema(low, 7)",
        "ema(close, 5) > sma(close, 20) and volume > sma(volume, 10)",
        "not (sma(close, 5) > close) or sma(open, 3) > close",
        "sma(ema(close, 4), 6) / sma(close, 9) ** 2",
    ],
)
@pytest.mark.parametrize("dirty", [False, True])
def test_eval_equals_stream_bit_for_bit(text, dirty):
    data = series()
    if dirty:  # unknown values in the middle of the run must not split the two modes
        for name, i in (("close", 50), ("high", 120), ("volume", 200)):
            data[name][i] = NAN
        data["low"][300] = np.inf
    e = rb.expr(text)
    batch = e.eval(data)
    live = stream_all(e, data)
    assert np.array_equal(bits(batch), bits(live))
    assert e.stream().ready is False


def test_a_stream_is_ready_after_its_warmup():
    e = rb.expr("sma(close, 5)")
    s = e.stream()
    for i in range(e.warmup):
        s.update(close=100.0 + i)
        assert not s.ready
    s.update(close=1.0)  # the first known value
    assert s.ready and s.bars == e.warmup + 1
    s.reset()
    assert s.bars == 0 and not s.ready


def test_the_first_known_value_is_at_the_warmup_index():
    close = series()["close"]
    for text in ("sma(close, 5)", "ema(close, 10)", "sma(sma(close, 5), 5) + 1", "sma(close, 5) - sma(close, 20)"):
        e = rb.expr(text)
        out = e.eval({"close": close})
        assert int(np.argmax(~np.isnan(out))) == e.warmup, text


def test_settle_is_the_warmup_plus_the_seed_fading():
    assert rb.expr("sma(close, 5)").settle(1e-6) == 4
    assert rb.expr("ema(close, 10)").settle(1e-6) == 78  # 9 + ceil(ln 1e-6 / ln(9/11))
    for bad in (0.0, 1.0, -1e-3, float("nan")):
        with pytest.raises(RaptorBTError) as info:
            rb.expr("ema(close, 10)").settle(bad)
        assert info.value.code == "invalid_parameter" and info.value.parameter == "eps"


def test_sma_matches_the_legacy_streaming_indicator():
    close = series()["close"]
    for period in (1, 3, 20):
        ind = rb.Indicator.sma(period)
        legacy = []
        for x in close:
            v = ind.update_bar(x, x, x, x)
            legacy.append(NAN if v is None else float(v))
        mine = rb.expr(f"sma(close, {period})").eval({"close": close})
        assert np.array_equal(bits(mine), bits(legacy)), period


def test_a_longer_history_never_changes_an_earlier_value():
    data = series(120)
    e = rb.expr("ema(close, 5) > sma(close, 20) and volume > sma(volume, 10)")
    full = e.eval(data)
    for n in (1, 7, 30, 119):
        part = e.eval({k: v[:n] for k, v in data.items()})
        assert np.array_equal(bits(part), bits(full[:n])), n


# ---------------------------------------------------------------- unknown values


def test_kleene_unknown_shows_as_nan():
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    below = c.sma(3) < c  # unknown on the first two bars, then true
    assert np.isnan(below.eval({"close": close})[:2]).all()
    assert below.eval({"close": close})[2:].tolist() == [1.0, 1.0, 1.0]
    # not unknown is unknown; never true while warming up
    out = (~below).eval({"close": close})
    assert np.isnan(out[:2]).all() and out[2:].tolist() == [0.0, 0.0, 0.0]
    # a known operand decides: unknown and false is false, unknown or true is true
    assert (below & (c > 100)).eval({"close": close}).tolist() == [0.0] * 5
    assert (below | (c > 0)).eval({"close": close}).tolist() == [1.0] * 5
    # unknown and true / unknown or false stay unknown on the first two bars
    assert np.isnan((below & (c > 0)).eval({"close": close})[:2]).all()
    assert np.isnan((below | (c > 100)).eval({"close": close})[:2]).all()


def test_a_nan_input_is_unknown_in_a_window_and_then_recovers():
    close = np.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0, 7.0])
    out = rb.expr("sma(close, 2)").eval({"close": close})
    # windows: [1], [1,2], [2,nan], [nan,4], [4,5], [5,6], [6,7]
    expected = [NAN, 1.5, NAN, NAN, 4.5, 5.5, 6.5]
    assert np.array_equal(bits(out), bits(expected))


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("text", "params", "code", "position", "valid"),
    [
        ("close +", None, "expr_syntax", 7, None),
        ("1 < close < 3", None, "expr_syntax", 10, None),
        ("1 + smaa(close, 5)", None, "expr_unknown_function", 4, ["sma"]),
        ("clse > 1", None, "expr_type", 0, ["close"]),
        ("sma(close, close)", None, "expr_type", 11, None),
        ("close and open", None, "expr_type", None, None),
        ("sma(close, $m)", {"n": 5}, "expr_unbound_parameter", 11, ["n"]),
        ("sma(close, $n)", None, "expr_unbound_parameter", 11, None),
        ('at("1d", close)', None, "expr_unsupported_here", 0, None),
    ],
)
def test_refusals_carry_code_position_snippet_and_valid(text, params, code, position, valid):
    with pytest.raises(ExpressionError) as info:
        rb.expr(text, params=params)
    e = info.value
    assert e.code == code and e.operation == "expr" and e.parameter == "expression"
    assert e.value == text
    assert e.valid, "an error lists alternatives or states the rule"
    if valid is not None:
        assert e.valid == valid or set(valid) <= set(e.valid)
    if position is not None:
        assert e.position == position
        line = text.splitlines()[0]
        assert e.snippet == f"{line}\n{' ' * position}^"
    assert isinstance(e, ValueError)
    back = errors.from_dict(json.loads(json.dumps(e.to_dict())))
    assert (back.code, back.position, back.snippet) == (e.code, e.position, e.snippet)
    assert str(e).splitlines()[0] == e.message


def test_non_ascii_text_is_refused_at_its_character():
    with pytest.raises(ExpressionError) as info:
        rb.expr("close é 1")
    e = info.value
    assert (e.code, e.position) == ("expr_syntax", 6)
    assert e.snippet == "close é 1\n      ^"


def test_building_from_python_refuses_what_the_parser_refuses():
    for bad in (
        lambda: c.sma(0),
        lambda: c.sma(2.5),
        lambda: c.sma(),
        lambda: c.sma(1, 2),
        lambda: c.sma(c),
        lambda: c.sma(period=3, bogus=1),
        lambda: c.sma(3, period=3),
        lambda: c + "x",
    ):
        with pytest.raises((RaptorBTError, TypeError)) as info:
            bad()
        if isinstance(info.value, RaptorBTError):
            assert info.value.code.startswith("expr_")


def test_an_unknown_function_is_listed_with_close_names():
    with pytest.raises(ExpressionError) as info:
        rb.expressions.describe("smaa")
    assert info.value.code == "expr_unknown_function" and "sma" in info.value.valid


def test_a_multi_output_name_is_not_an_attribute_of_a_single_output():
    with pytest.raises(AttributeError):
        c.sma(3).signal


# ---------------------------------------------------------------- inputs


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.float32),
        [1.0, 2.0, 3.0],
        np.ones((3, 2)),
        1.0,
    ],
)
def test_a_non_float64_array_is_refused(bad):
    with pytest.raises(RaptorBTError) as info:
        rb.expr("close + 1").eval({"close": bad})
    assert info.value.code == "invalid_array" and info.value.parameter == "close"
    assert info.value.operation


def test_data_must_be_a_dict():
    with pytest.raises(RaptorBTError) as info:
        rb.expr("close").eval(np.ones(3))
    assert info.value.code == "invalid_input_type"


def test_missing_and_mismatched_columns_are_refused_by_name():
    with pytest.raises(RaptorBTError) as info:
        rb.expr("high - low").eval({"close": np.ones(3)})
    assert info.value.code == "missing_field" and info.value.parameter in {"high", "low"}
    with pytest.raises(RaptorBTError) as info:
        rb.expr("high - low").eval({"high": np.ones(3), "low": np.ones(2)})
    assert info.value.code == "length_mismatch"
    with pytest.raises(RaptorBTError) as info:
        rb.expr("close").eval({})
    assert info.value.code == "missing_field"
    with pytest.raises(RaptorBTError) as info:
        rb.expr("high - low").stream().update(low=1.0)
    assert info.value.code == "missing_field" and info.value.parameter == "high"


def test_keys_that_are_not_fields_are_ignored_and_a_strided_view_is_read_correctly():
    close = series()["close"]
    e = rb.expr("sma(close, 5)")
    plain = e.eval({"close": close})
    extra = e.eval({"close": close, "timestamps": np.arange(len(close), dtype=np.int64)})
    assert np.array_equal(bits(plain), bits(extra))
    wide = np.empty(2 * len(close))
    wide[::2] = close
    assert np.array_equal(bits(e.eval({"close": wide[::2]})), bits(plain))


def test_no_input_is_modified():
    data = series(60)
    frozen = {k: v.copy() for k, v in data.items()}
    for v in data.values():
        v.flags.writeable = False  # a read-only array is accepted and left alone
    for text in ("close", "sma(close, 5) < close", "high - low", "volume * 2"):
        out = rb.expr(text).eval(data)
        assert not any(np.shares_memory(out, v) for v in data.values())
    for k in data:
        assert np.array_equal(bits(data[k]), bits(frozen[k]))


def test_an_empty_or_short_series_is_all_unknown_not_an_error():
    e = rb.expr("sma(close, 5)")
    assert e.eval({"close": np.array([])}).shape == (0,)
    assert np.isnan(e.eval({"close": np.array([1.0, 2.0])})).all()


# ---------------------------------------------------------------- the registry drives the surface


def registry():
    return json.loads(_raptorbt.expr_registry_json())["functions"]


def sample_args(row):
    """One valid constant per parameter, chosen from the registry's own range."""
    out = []
    for p in row["params"]:
        lo, hi = p["range"].get("min"), p["range"].get("max")
        if p["kind"] == "int":
            out.append(min(max(3, lo), hi))
        elif p["kind"] == "float":
            out.append(0.5 if (lo is None or lo <= 0.5) and (hi is None or hi >= 0.5) else lo)
        else:
            out.append(p["range"]["values"][0])
    return out


def literal(v):
    return f'"{v}"' if isinstance(v, str) else repr(v)


def test_no_function_name_shadows_an_attribute():
    assert rb.expressions._SHADOWED == []


def test_every_registered_function_is_a_col_builder_and_an_expr_method():
    names = {r["name"] for r in registry()}
    assert names and names <= set(dir(rb.col))
    assert {r["name"] for r in registry() if r["inputs"]} <= set(dir(Expr))
    assert {f["name"] for f in rb.expressions.functions()} == names
    assert {f["name"] for f in rb.capabilities()["functions"]} == names


def test_every_method_builds_what_the_text_form_parses():
    for row in registry():
        name, ins = row["name"], row["inputs"]
        if not ins:
            continue
        args = sample_args(row)
        # the method supplies the first input; name the rest as the language would default them
        rest = [i["default_field"] or "close" for i in ins[1:]]
        text = f"{name}({', '.join(['close', *rest, *map(literal, args)])})"
        built = getattr(c, name)(*[getattr(rb.col, f) for f in rest], *args)
        assert built.canonical == rb.expr(text).canonical, name
        assert built.hash == rb.expr(text).hash, name
        if all(i["default_field"] for i in ins):
            defaults = [i["default_field"] for i in ins]
            via_col = getattr(rb.col, name)(*args)
            assert via_col.canonical == rb.expr(f"{name}({', '.join([*defaults, *map(literal, args)])})").canonical


def test_functions_filters_by_word_and_describe_returns_the_whole_row():
    rows = rb.expressions.functions()
    assert [r["name"] for r in rows] == sorted(r["name"] for r in rows)
    one = rb.expressions.functions("SMA")
    assert one and all("sma" in (r["name"] + r["family"] + r["doc"]).lower() for r in one)
    assert rb.expressions.functions("no such word in any function") == []
    full = rb.expressions.describe("sma")
    assert full["name"] == "sma" and full["outputs"] == ["value"] and full["signature"].startswith("sma(")
    full["params"].clear()  # a copy: the registry is not changed by the caller
    assert rb.expressions.describe("sma")["params"]


def test_each_function_row_has_what_the_docs_are_generated_from():
    keys = {
        "name", "family", "inputs", "params", "outputs", "out_kind", "warmup_at_defaults",
        "memory", "legacy", "talib_name", "doc", "refs", "rev", "invariance", "causal",
    }  # fmt: skip
    for row in registry():
        assert keys <= set(row), row["name"]
        assert row["doc"], row["name"]
        assert 1 <= len(row["outputs"]) <= 3


# ---------------------------------------------------------------- the root surface


def test_the_root_exports_expr_and_col_and_the_rest_lives_in_the_module():
    assert rb.expr is rb.expressions.expr and rb.col is rb.expressions.col
    assert {"expr", "col"} <= set(rb.__all__)
    assert not {"Expr", "ExprStream", "functions", "describe"} & set(rb.__all__)
    assert set(rb.expressions.__all__) == {"Expr", "ExprStream", "col", "describe", "expr", "functions"}
