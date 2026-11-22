"""``raptorbt.bundle``: strategy code packed as one hashed file and imported from memory.

The same files always make the same bytes; the bytes are the bundle only if they hash to its
sha and hold nothing but ``.py`` under ``defs/`` and ``lib/``; every ``defs/<family>/<id>.py``
loads as one definition or the whole bundle is refused and nothing of it stays imported; two
bundles holding the same module names live side by side; a checkpoint at rest names ``bundle.``;
the contract hash follows the API line, the manifest schema and the manifest parser's bytes.
"""

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest
import raptorbt
from raptorbt import bundle
from raptorbt.errors import StateError
from raptorbt.strategy import requirements as manifest

REQUIREMENTS = (
    "requirements = Requirements(roles={'a': Underlying()}, timeframes=['1m'],"
    " holding=Holding.overnight()__PARAMS__)"
)
DEFINITION = '''
from dataclasses import dataclass
from raptorbt import Strategy, StrategyConfig
from raptorbt.strategy import Holding, Requirements, Underlying, param
from bundle.lib.sizing import units

NAME = "bundle.lib.sizing"


@dataclass(frozen=True)
class Config(StrategyConfig):
    lots: int = param(1, min=1, max=5, description="lots per entry")


class Breakout(Strategy):
    """A breakout on one underlying.

    More about it."""

    version = 2
    __REQUIREMENTS__
'''.replace("__REQUIREMENTS__", REQUIREMENTS.replace("__PARAMS__", ", params=Config"))


def files(lot: int = 50, extra: dict[str, bytes] | None = None) -> dict[str, bytes]:
    return {
        "defs/__init__.py": b"",
        "defs/trend/__init__.py": b"",
        "lib/__init__.py": b"",
        "defs/trend/breakout.py": DEFINITION.encode(),
        "lib/sizing.py": f"def units(n):\n    return n * {lot}\n".encode(),
        **(extra or {}),
    }


def modules_of(sha: str) -> list[str]:
    return [n for n in sys.modules if n.split(".")[0] == bundle.package_of(sha)]


def strategy_files(attrs: str) -> dict[str, bytes]:
    body = (
        "from raptorbt import Strategy\n"
        "from raptorbt.strategy import Holding, Requirements, Underlying\n"
        f"class S(Strategy):\n    {attrs}\n"
    )
    return {"defs/trend/__init__.py": b"", "defs/trend/s.py": body.encode()}


def test_the_same_files_make_the_same_bytes() -> None:
    sha, content = bundle.pack(files())
    assert bundle.pack(dict(reversed(files().items()))) == (sha, content)
    assert sha == hashlib.sha256(content).hexdigest()
    assert bundle.pack(files(lot=51))[0] != sha
    assert bundle.files(sha, content) == files()


def test_a_bundle_loads_as_definitions_of_the_files_it_was_packed_from() -> None:
    sha, content = bundle.pack(files())
    (d,) = bundle.load(sha, content)
    assert (d.id, d.family, d.name, d.version) == ("breakout", "trend", "Breakout", 2)
    assert d.summary == "A breakout on one underlying."
    assert d.module == f"rb_{sha[:16]}.defs.trend.breakout" == f"{bundle.package_of(sha)}.defs.trend.breakout"
    assert d.config is not None and d.cls.requirements.params is not None
    # the source is the definition's file as stored, then every lib file, in path order
    source = "".join(files()[p].decode() for p in ("defs/trend/breakout.py", "lib/__init__.py", "lib/sizing.py"))
    assert d.source == source
    assert d.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert d.fingerprint == hashlib.sha256((source + d.requirements_hash).encode()).hexdigest()
    assert d.describe()["fingerprint"] == d.fingerprint
    assert sys.modules[f"{bundle.package_of(sha)}.lib.sizing"].units(2) == 100
    # the string "bundle.lib.sizing" in the file names the package that was made
    assert sys.modules[d.module].NAME == f"{bundle.package_of(sha)}.lib.sizing"
    assert bundle.load(sha, content) is bundle.load(sha, content)  # answered as it was


def test_two_bundles_hold_the_same_modules_side_by_side() -> None:
    (sa, ca), (sb, cb) = bundle.pack(files(lot=50)), bundle.pack(files(lot=70))
    bundle.load(sa, ca)
    bundle.load(sb, cb)
    one = sys.modules[f"{bundle.package_of(sa)}.lib.sizing"]
    two = sys.modules[f"{bundle.package_of(sb)}.lib.sizing"]
    assert one is not two and (one.units(1), two.units(1)) == (50, 70)


def test_a_prefix_names_the_package() -> None:
    sha, content = bundle.pack(files(lot=71))
    (d,) = bundle.load(sha, content, prefix="host_")
    assert d.module == f"host_{sha[:16]}.defs.trend.breakout"
    assert bundle.package_of(sha, "host_") == f"host_{sha[:16]}"
    assert modules_of(sha) == []  # nothing under the default prefix


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        ({"defs/trend/__init__.py": b"", "defs/trend/b.py": b"raise RuntimeError('boom')\n"}, "RuntimeError: boom"),
        ({"defs/trend/__init__.py": b"", "defs/trend/b.py": b"X = 1\n"}, "defines 0 raptorbt.Strategy classes"),
        ({"defs/trend/b.py": b"from bundle import lib\n"}, "by full name"),
        ({"defs/trend/deep/x.py": b"X = 1\n"}, "defs/<family>/<id>.py"),
        (strategy_files("pass"), "declares no requirements"),
        (strategy_files("version = 1"), "declares no requirements"),
        (strategy_files(REQUIREMENTS.replace("__PARAMS__", "")), "declares no version"),
        (strategy_files("version = 0\n    " + REQUIREMENTS.replace("__PARAMS__", "")), "declares no version"),
        (
            strategy_files(
                "version = 1\n    "
                "requirements = Requirements(roles={'a': Underlying()}, timeframes=['1m'])"
            ),
            "missing_holding",
        ),
        (
            strategy_files("version = 1\n    " + REQUIREMENTS.replace("__PARAMS__", ""))
            | {"defs/trend/t.py": strategy_files("pass")["defs/trend/s.py"]},
            "declares no requirements",
        ),
    ],
)
def test_a_bundle_with_a_definition_that_does_not_load_is_refused_whole(
    bad: dict[str, bytes], why: str
) -> None:
    sha, content = bundle.pack(bad)
    hooks = len(sys.meta_path)
    with pytest.raises(bundle.BundleError, match=why):
        bundle.load(sha, content)
    assert modules_of(sha) == [] and len(sys.meta_path) == hooks  # nothing of it stays imported
    with pytest.raises(bundle.BundleError, match=why):  # and a refusal is not cached
        bundle.load(sha, content)


def test_the_config_class_is_the_manifests_schema() -> None:
    both = DEFINITION.replace(", params=Config", "")
    sha, content = bundle.pack(files(extra={"defs/trend/breakout.py": both.encode()}))
    with pytest.raises(bundle.BundleError, match="requirements.params is the schema"):
        bundle.load(sha, content)


def test_bytes_that_are_not_the_bundle_are_refused() -> None:
    content = bundle.pack(files())[1]
    with pytest.raises(bundle.BundleError, match="not bundle"):
        bundle.files("0" * 64, content)
    junk = b"not a tar"
    with pytest.raises(bundle.BundleError, match="not a tar"):
        bundle.files(hashlib.sha256(junk).hexdigest(), junk)
    for path in ("setup.py", "lib/data.json", "defs/../../x.py"):
        s, c = bundle.pack({path: b"X = 1\n"})
        with pytest.raises(bundle.BundleError):
            bundle.files(s, c)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for _ in range(2):  # a path twice
            info = tarfile.TarInfo("lib/a.py")
            tar.addfile(info, io.BytesIO(b""))
    twice = buf.getvalue()
    with pytest.raises(bundle.BundleError, match="not a path of a bundle"):
        bundle.files(hashlib.sha256(twice).hexdigest(), twice)


def test_refuse_is_asked_of_every_import_and_names_the_line() -> None:
    sha, content = bundle.pack(
        {"defs/trend/__init__.py": b"", "defs/trend/x.py": b"import os\nfrom socket import socket\n"}
    )
    seen: list[str] = []

    def refuse(name: str) -> str | None:
        seen.append(name)
        return "the network is not allowed" if name == "socket" else None

    with pytest.raises(bundle.BundleError, match="line 2: the network is not allowed"):
        bundle.load(sha, content, refuse=refuse)
    assert seen == ["os", "socket"]


def test_a_checkpoint_at_rest_never_names_a_bundle() -> None:
    sha, other = "ab" * 32, "cd" * 32
    p = bundle.package_of(sha)
    doc = {
        "cls": f"{p}.lib._trader:BarTrader",
        "nested": [{"k": f"{p}.defs.trend.crossover_1m:Thing", f"{p}.lib._carry": 1}],
        "engine": "raptorbt.strategy:Strategy",
        "text": f"mentions {p}.lib.x in passing",
        "n": 3,
    }
    stored = bundle.canonical(doc, sha)
    assert stored == {
        "cls": "bundle.lib._trader:BarTrader",
        "nested": [{"k": "bundle.defs.trend.crossover_1m:Thing", "bundle.lib._carry": 1}],
        "engine": "raptorbt.strategy:Strategy",
        "text": f"mentions {p}.lib.x in passing",
        "n": 3,
    }
    assert bundle.local(stored, sha) == doc
    assert bundle.local(stored, other)["cls"] == f"{bundle.package_of(other)}.lib._trader:BarTrader"
    assert bundle.canonical(doc, "") == doc  # no bundle: nothing to turn
    assert bundle.canonical(doc, sha, "host_") == doc  # another prefix: not this bundle's names


def test_the_contract_hash_is_the_api_line_the_schema_and_the_parser() -> None:
    h = hashlib.sha256()
    h.update(raptorbt.__version__.split(".")[0].encode() + b".x\0")
    h.update(str(manifest.SCHEMA).encode() + b"\0")
    h.update(Path(manifest.__file__).read_bytes())
    assert bundle.contract_hash() == h.hexdigest()
    assert bundle.engine_version() == raptorbt.__version__


def test_a_release_in_the_same_api_line_keeps_the_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    major = raptorbt.__version__.split(".")[0]
    bundle.contract_hash.cache_clear()
    base = bundle.contract_hash()
    try:
        for version, same in ((f"{major}.99.0", True), (f"{int(major) + 1}.0.0", False)):
            monkeypatch.setattr(raptorbt, "__version__", version)
            bundle.contract_hash.cache_clear()
            assert (bundle.contract_hash() == base) is same, version
        monkeypatch.setattr(raptorbt, "__version__", "unknown")
        bundle.contract_hash.cache_clear()
        with pytest.raises(StateError, match="no installed version") as e:
            bundle.contract_hash()
        assert e.value.code == "engine_version_unknown"
    finally:
        monkeypatch.undo()
        bundle.contract_hash.cache_clear()
