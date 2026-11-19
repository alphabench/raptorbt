"""Every registered function that names a TA-Lib function is replayed against the committed
fixture, by the relation the registry declares. Needs numpy only (no TA-Lib)."""

import os

import numpy as np
import pytest

import replay
from harness import registry, talib_names

FIXTURE_BYTES_CAP = 15 * 1024 * 1024


def _rows():
    rows = []
    for fn in registry():
        for t in talib_names(fn):
            entry = replay.function_fixture(t)
            if entry is None:
                rows.append(pytest.param(fn, t, None, None, id=f"{fn['name']}-nofixture"))
                continue
            for case in entry["cases"]:
                rows.append(
                    pytest.param(
                        fn, t, entry, case, id=f"{fn['name']}-{case['scenario']}-{case['label']}"
                    )
                )
    return rows


def test_fixture_size_cap():
    total = sum(
        os.path.getsize(os.path.join(replay.FIXTURES, f))
        for f in os.listdir(replay.FIXTURES)
        if f.endswith(".json.gz")
    )
    assert total <= FIXTURE_BYTES_CAP, f"fixtures are {total} bytes, cap {FIXTURE_BYTES_CAP}"


def test_headers_record_versions_and_digest():
    for f in sorted(os.listdir(replay.FIXTURES)):
        if not f.endswith(".json.gz"):
            continue
        h = replay._load(os.path.join(replay.FIXTURES, f))["header"]
        assert h["libraries"]["ta-lib-python"] == "0.6.4", f
        assert len(h["input_digest"]) == 64, f


def test_input_digest_matches_inputs():
    import hashlib

    inp = replay._load(os.path.join(replay.FIXTURES, "inputs.json.gz"))
    h = hashlib.sha256()
    for scen in sorted(inp["series"]):
        for col in sorted(inp["series"][scen]):
            h.update(f"{scen}/{col}:{inp['series'][scen][col]}".encode())
    assert h.hexdigest() == inp["header"]["input_digest"]


def test_candle_firing_per_sign():
    """Every pattern fires on at least 5 bars per sign, except the signs the generator
    recorded as never observed (declared in the header)."""
    g = replay.group_fixture("CDLDOJI")
    never = g["header"]["never_fires_sign"]
    for name, f in g["functions"].items():
        for sign, key in (("pos", "pos"), ("neg", "neg")):
            if sign in never.get(name, []):
                assert f["firing"][key] == 0, f"{name} was recorded as never firing {sign}"
            else:
                assert f["firing"][key] >= g["header"]["min_firing_per_sign"], (
                    f"{name} fires {f['firing'][key]} {sign}"
                )


@pytest.mark.parametrize("fn,talib_name,entry,case", _rows())
def test_replay(fn, talib_name, entry, case):
    assert entry is not None, (
        f"{fn['name']} names TA-Lib {talib_name} but there is no fixture for it"
    )
    problems = replay.replay_case(fn, talib_name, entry, case, replay.load_pins())
    assert not problems, "\n".join(problems)


COMPARED_RELATIONS = ("exact", "converges", "candle")


def test_every_compared_function_replays_at_least_one_case():
    """A suite that compares nothing passes; fail when a function that should be compared has
    no fixture cases, or none that were run."""
    empty = []
    for fn in registry():
        if fn.get("relation") not in COMPARED_RELATIONS and not replay.has_compat(fn):
            continue
        for t in talib_names(fn):
            entry = replay.function_fixture(t)
            if entry is None or not entry["cases"]:
                empty.append(f"{fn['name']} ({t}): no fixture cases")
    assert not empty, "\n".join(empty)


def _fixture_series(name, output):
    entry = replay.function_fixture(name)
    case = entry["cases"][0]
    ref = replay.reference_array(case["out"][output], case["lookback"], "exact")
    return case, ref


def test_exact_comparison_detects_a_small_change():
    case, ref = _fixture_series("SMA", "real")
    assert replay.check_exact(ref.copy(), ref) == []
    ours = ref.copy()
    i = case["lookback"] + 10
    ours[i] += 1e-6
    assert replay.check_exact(ours, ref), "a 1e-6 change on one bar went unnoticed"
    ours = ref.copy()
    ours[i] = float("nan")
    assert replay.check_exact(ours, ref), "a NaN where TA-Lib has a value went unnoticed"


def test_exact_comparison_floors_noise_near_zero_but_not_more():
    # A histogram value near zero is a difference of two outputs of order 1: its last digits
    # are libm noise (macOS and Linux disagree by 9.4e-15 here), so the relative error is taken
    # against NOISE_FLOOR of the case's scale there; 1e-12 off is still a failure.
    ref = np.array([1.0, -2.7, -4.643472073628807e-06, 0.5])
    ours = ref.copy()
    ours[2] = -4.643472083065703e-06
    assert replay.check_exact(ours, ref) == []
    ours[2] = ref[2] + 1e-12
    assert replay.check_exact(ours, ref), "a 1e-12 change near zero went unnoticed"


def test_a_pin_holds_through_libm_noise_and_fails_on_a_real_change():
    # Pins of a defined difference are compared to PIN_REL of the pinned magnitude, floored at
    # NOISE_FLOOR of the case's scale, because exact hex equality failed on Linux's libm
    # (HT_SINE lead sine moved 8.0e-14 -> 7.3e-14 on a series of scale 1). Counts and bars stay
    # exact.
    noise = {"first_valid_ours": 63, "first_valid_ref": 63, "n_compared": 437,
             "max_abs": "0x1.6a20000000000p-44"}
    assert replay.same_pin(noise, {**noise, "max_abs": "0x1.4720000000000p-44"}, 1.0)
    assert not replay.same_pin(noise, {**noise, "max_abs": (3e-13).hex()}, 1.0)
    assert not replay.same_pin(noise, {**noise, "n_compared": 436}, 1.0)
    assert not replay.same_pin(noise, {**noise, "first_valid_ours": 64}, 1.0)
    real = {**noise, "max_abs": (0.4864682203212638).hex()}
    assert replay.same_pin(real, dict(real), 100.0)
    assert not replay.same_pin(real, {**real, "max_abs": (0.4864692203212638).hex()}, 100.0)


def test_converges_comparison_detects_a_small_change_after_k():
    case, ref = _fixture_series("EMA", "real")
    k = 20
    assert replay.check_converges(ref.copy(), ref, k)[0] == []
    ours = ref.copy()
    ours[case["lookback"] + k + 10] += 1e-6
    assert replay.check_converges(ours, ref, k)[0], "a 1e-6 change after K went unnoticed"


def test_candle_comparison_detects_a_changed_integer():
    entry = replay.function_fixture("CDLDOJI")
    case = entry["cases"][0]
    ref = replay.reference_array(case["out"]["integer"], case["lookback"], "candle").astype(float)
    assert replay.check_candle(ref.copy(), ref, case["lookback"]) == []
    ours = ref.copy()
    i = int(ours.nonzero()[0][0])
    ours[i] = 0.0
    assert replay.check_candle(ours, ref, case["lookback"]), "a missed firing went unnoticed"


def test_ignored_talib_parameters_really_are_ignored():
    """TALIB_IGNORED claims TA-Lib gives the same output whatever the parameter; prove it on
    the fixtures, comparing each case with the default case of its scenario."""
    for talib_name, ignored in replay.TALIB_IGNORED.items():
        entry = replay.function_fixture(talib_name)
        by = {(c["scenario"], c["label"]): c for c in entry["cases"]}
        tested = 0
        for (scen, label), c in by.items():
            if label == "default":
                continue
            base = by[(scen, "default")]
            if any(c["params"][k] != base["params"][k] for k in ignored):
                assert c["out"] == base["out"], (
                    f"{talib_name}: {sorted(ignored)} changed the output in {scen}/{label}"
                )
                tested += 1
        assert tested, f"{talib_name}: no fixture case varies {sorted(ignored)}"


def test_stddev_nbdev_scales_the_output_in_the_fixtures():
    entry = replay.function_fixture("STDDEV")
    by = {(c["scenario"], c["label"]): c for c in entry["cases"]}
    scen = "random_walk"
    one = replay.decode(by[(scen, "default")]["out"]["real"])
    k = replay.decode(by[(scen, "nbdev_2_5")]["out"]["real"])
    ok = ~np.isnan(one)
    assert np.allclose(k[ok], one[ok] * 2.5, rtol=1e-12)
