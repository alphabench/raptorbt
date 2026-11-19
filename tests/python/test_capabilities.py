"""``raptorbt.capabilities()`` lists what the engine has, and fails when it does not."""

import inspect
import json
import re
from pathlib import Path

import raptorbt
from raptorbt import _raptorbt
from raptorbt.strategy import Strategy

# The Rust sources are read from the checkout this test file lives in, not from
# next to the installed package: CI installs a wheel, whose package directory
# has no `src/` beside it.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").is_file())
SRC = REPO / "src"


def test_the_manifest_is_json_and_versioned():
    caps = raptorbt.capabilities()
    assert json.loads(json.dumps(caps)) == caps
    assert caps["schema"] == 1
    for key in (
        "order_kinds", "time_in_force", "hooks", "indicators", "instrument_kinds",
        "settlement", "margin_models", "fee_rounding", "data_types", "adapters", "error_codes",
    ):
        assert caps[key], key


def test_every_order_kind_and_tif_the_bindings_parse_is_listed():
    text = (SRC / "python" / "strategy_bindings.rs").read_text()
    kinds = set(re.findall(r'"(\w+)" => OrderKind::', text))
    tifs = set(re.findall(r'"(\w+)" => TimeInForce::', text))
    caps = raptorbt.capabilities()
    assert kinds and tifs
    assert kinds <= {k["name"] for k in caps["order_kinds"]}
    assert tifs <= set(caps["time_in_force"])


def test_every_hook_a_strategy_has_is_listed():
    caps = raptorbt.capabilities()
    listed = set()
    for h in caps["hooks"]:
        listed.add(h["name"])
        listed.update(h["aliases"])
    own = {n for n, f in inspect.getmembers(Strategy, inspect.isfunction) if n.startswith("on_")}
    assert own <= listed, own - listed


def test_every_indicator_the_extension_registers_is_listed_with_its_parameters():
    caps = raptorbt.capabilities()
    arrays = {i["name"]: i for i in caps["indicators"]["arrays"]}
    text = (SRC / "lib.rs").read_text()
    block = text[text.index("Register indicator functions") :]
    block = block[: block.index("_INDICATOR_FUNCTIONS")]
    registered = re.findall(r"indicator!\(python::bindings::(\w+)\)", block)
    assert registered and set(registered) == set(arrays)
    assert [p["name"] for p in arrays["sma"]["parameters"]] == ["data", "period"]
    streaming = {i["name"] for i in caps["indicators"]["streaming"]}
    constructors = {
        n for n, m in inspect.getmembers(_raptorbt.Indicator)
        if not n.startswith("_") and callable(m) and n not in {"reset", "update_bar"}
    }
    assert constructors == streaming


def test_every_error_code_is_listed():
    assert set(raptorbt.errors.CODES) == set(raptorbt.capabilities()["error_codes"])


def test_the_built_in_adapters_are_listed():
    names = {a["name"] for a in raptorbt.capabilities()["adapters"]}
    assert {"PaperExecution", "SimulatedExecution"} <= names


def test_every_statistic_is_listed():
    import raptorbt.stats

    assert raptorbt.capabilities()["statistics"]["functions"] == sorted(raptorbt.stats.__all__)
