"""Strategy code packed as a bundle: a tar of ``defs/**.py`` and ``lib/**.py``, named by its sha256.

:func:`pack` makes the bytes (the same files always make the same bytes), :func:`load` imports
them from memory as a package of their own, ``<prefix><first 16 of the sha>``, so two bundles
holding the same module names live in one process without touching each other, and each
``defs/<family>/<id>.py`` comes back as a :class:`Definition`: its one :class:`~raptorbt.Strategy`
subclass, its at most one :class:`~raptorbt.StrategyConfig`, and the ``requirements`` manifest the
class declares. A host that admits strategies written by other people (a worker, a runner on
someone's own machine) names a bundle by its sha, checks the bytes against it, and loads it
here; nothing is written to disk and nothing is imported by path.

The files import raptorbt as ``raptorbt`` and their own modules as ``bundle.lib.x`` and
``bundle.defs.f.y``: in the code that runs, that prefix (in imports and in string constants) names
the bundle's package instead. The text a module answers for its source is the file as stored, so
a definition's ``source_sha256`` and ``fingerprint`` are what they are in the directory it was
packed from. What a strategy writes into a checkpoint names its bundle's package: :func:`canonical`
and :func:`local` turn those names to ``bundle.`` and back, so a checkpoint at rest never names a
bundle's package.

:func:`contract_hash` is the hash of what reads a bundle's declaration in this process (the API
line and the manifest's schema and parser). A host records it with each bundle it admits and
compares it on load: a bundle admitted under another reader is stale, a release inside the same
API line is not.

A bundle is untrusted code that runs with the host's privileges: this module checks its shape,
not its behaviour. A host narrows what a bundle may import with the ``refuse`` hook of
:func:`load` and by its own review of :func:`files` before it loads.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any

import raptorbt
from raptorbt.errors import StateError
from raptorbt.strategy import (
    Requirements,
    Strategy,
    StrategyConfig,
    params_schema,
    requirements_hash,
)
from raptorbt.strategy import requirements as _manifest

__all__ = [
    "LIBRARY",
    "PREFIX",
    "BundleError",
    "Definition",
    "canonical",
    "contract_hash",
    "engine_version",
    "files",
    "load",
    "local",
    "pack",
    "package_of",
]

LIBRARY = "bundle"
"""The name a bundle's files give their own package (``bundle.lib.x``, ``bundle.defs.f.y``)."""
PREFIX = "rb_"
"""The default start of the package a bundle is imported under (``rb_<first 16 of the sha>``)."""
_INSIDE = (f"{LIBRARY}.defs", f"{LIBRARY}.lib")
_DEFINITION_PATH_PARTS = 3  # defs/<family>/<id>: where a definition's module sits in a bundle


class BundleError(Exception):
    """The bundle is not what its name says, or a definition in it does not load. ``rule``: a
    stable code when the refusal has one."""

    def __init__(self, message: str, rule: str | None = None) -> None:
        super().__init__(message)
        self.rule = rule


def engine_version() -> str:
    """The raptorbt this process runs (its installed distribution's version)."""
    v: str = raptorbt.__version__
    if v == "unknown":
        raise StateError(
            "raptorbt has no installed version: install it as a package",
            code="engine_version_unknown",
            operation="raptorbt.bundle.engine_version",
        )
    return v


@functools.cache
def contract_hash() -> str:
    """The hash of what reads a bundle's declaration in this process: the raptorbt API line
    (``1.x``, not the release), the manifest's schema number and the manifest module's own source
    (its parser and validator). A release inside the same API line, or a host change that does not
    touch how a manifest is read, does not move it; a new major does. The value is cached for the
    process (``contract_hash.cache_clear()`` rereads it)."""
    h = hashlib.sha256()
    for part in (engine_version().split(".")[0] + ".x", str(_manifest.SCHEMA)):
        h.update(part.encode() + b"\0")
    h.update(Path(_manifest.__file__).read_bytes())
    return h.hexdigest()


def package_of(sha: str, prefix: str = PREFIX) -> str:
    """The package a bundle's modules are imported under."""
    return f"{prefix}{sha[:16]}"


def _renamed(name: str, package: str) -> str:
    """``name`` with the ``bundle.defs``/``bundle.lib`` prefix turned to the bundle package's."""
    for inside in _INSIDE:
        if name == inside or name.startswith(inside + "."):
            return package + name[len(LIBRARY) :]
    return name


class _Rename(ast.NodeTransformer):
    def __init__(self, package: str, refuse: Callable[[str], str | None] | None) -> None:
        self.package = package
        self.refuse = refuse

    def _check(self, name: str, lineno: int) -> None:
        message = self.refuse(name) if self.refuse else None
        if message:
            raise BundleError(f"line {lineno}: {message}")
        if name == LIBRARY:
            raise BundleError(f"line {lineno}: import {LIBRARY}.defs or .lib by full name")

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            self._check(alias.name, node.lineno)
            alias.name = _renamed(alias.name, self.package)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.level == 0 and node.module is not None:
            self._check(node.module, node.lineno)
            node.module = _renamed(node.module, self.package)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        # "bundle.lib.x" (a logic module) and the bare prefix "bundle.lib." (what a checkpoint
        # may rebuild from).
        if isinstance(node.value, str):
            node.value = _renamed(node.value, self.package)
        return node


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.InspectLoader):
    """One bundle's modules, from memory."""

    def __init__(
        self,
        sha: str,
        files: dict[str, bytes],
        prefix: str,
        refuse: Callable[[str], str | None] | None,
    ) -> None:
        self.sha, self.package, self.refuse = sha, package_of(sha, prefix), refuse
        self.sources: dict[str, bytes] = {}  # module name -> its file's bytes
        self.packages: set[str] = {self.package}
        for path, data in files.items():
            parts = path[: -len(".py")].split("/")
            for i in range(1, len(parts)):
                self.packages.add(".".join((self.package, *parts[:i])))
            if parts[-1] == "__init__":
                self.sources[".".join((self.package, *parts[:-1]))] = data
            else:
                self.sources[".".join((self.package, *parts))] = data

    def _path(self, name: str) -> str:
        rel = name[len(self.package) + 1 :].replace(".", "/")
        tail = "/__init__.py" if name in self.packages else ".py"
        return f"<bundle {self.sha}>/{rel}{tail}" if rel else f"<bundle {self.sha}>/__init__.py"

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in self.packages and fullname not in self.sources:
            return None
        return importlib.util.spec_from_loader(
            fullname, self, origin=self._path(fullname), is_package=fullname in self.packages
        )

    def is_package(self, fullname: str) -> bool:
        return fullname in self.packages

    def get_source(self, fullname: str) -> str:
        """The file as stored: what ``inspect.getsource`` and so the fingerprint read."""
        return importlib.util.decode_source(self.sources.get(fullname, b""))

    def get_code(self, fullname: str) -> CodeType:
        tree = ast.parse(self.sources.get(fullname, b""), self._path(fullname))
        tree = ast.fix_missing_locations(_Rename(self.package, self.refuse).visit(tree))
        code: CodeType = compile(tree, self._path(fullname), "exec", dont_inherit=True)
        return code

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = self._path(module.__name__)
        exec(self.get_code(module.__name__), module.__dict__)  # noqa: S102 - how a loader runs a module


def pack(files: dict[str, bytes]) -> tuple[str, bytes]:
    """``files`` (path -> bytes) as a bundle: a tar in path order with fixed metadata, so the same
    files always make the same bytes; its sha256 and bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(files):
            info = tarfile.TarInfo(path)
            info.size, info.mode, info.mtime = len(files[path]), 0o644, 0
            tar.addfile(info, io.BytesIO(files[path]))
    content = buf.getvalue()
    return hashlib.sha256(content).hexdigest(), content


def files(sha: str, content: bytes) -> dict[str, bytes]:
    """The files of bundle ``sha`` (path -> bytes), nothing imported. The bytes must hash to
    ``sha`` and hold only ``.py`` files under ``defs/`` and ``lib/``, each path once and none
    leaving the bundle; anything else is a :class:`BundleError`."""
    if hashlib.sha256(content).hexdigest() != sha:
        raise BundleError(f"the bytes are not bundle {sha}")
    found: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as tar:
            for member in tar:
                top = member.name.split("/", 1)[0]
                if (
                    not member.isfile()
                    or top not in ("defs", "lib")
                    or not member.name.endswith(".py")
                ):
                    raise BundleError(
                        f"{member.name}: a bundle holds .py files under defs/ and lib/"
                    )
                if ".." in member.name.split("/") or member.name in found:
                    raise BundleError(f"{member.name}: not a path of a bundle")
                f = tar.extractfile(member)
                found[member.name] = f.read() if f is not None else b""
    except tarfile.TarError as exc:
        raise BundleError(f"bundle {sha} is not a tar: {exc}") from exc
    return found


@dataclass(frozen=True, slots=True)
class Definition:
    """One ``defs/<family>/<id>.py`` as loaded: its ``raptorbt.Strategy`` class, its config class
    (None when the strategy has none) and the manifest it declares."""

    id: str
    family: str
    module: str  # the module it was imported as (under ``package_of(sha)``)
    cls: type[Strategy]
    config: type[StrategyConfig] | None
    requirements: Requirements
    requirements_hash: str
    name: str  # the class's ``name``, else its class name
    version: int  # the class's ``version``: the author raises it to submit a change
    source: str  # the definition file's text, then every lib file's, in path order
    summary: str = ""  # the first paragraph of the class's docstring, else the module's

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()

    @property
    def fingerprint(self) -> str:
        """sha256(``source`` + ``requirements_hash``): what a run records. The hash carries the
        engine's version and the manifest, so a new engine or declaration moves it."""
        return hashlib.sha256((self.source + self.requirements_hash).encode()).hexdigest()

    def describe(self) -> dict[str, Any]:
        """The definition and its manifest as plain data, for a host to list."""
        return {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "family": self.family,
            "version": self.version,
            "fingerprint": self.fingerprint,
            "requirements": self.requirements.to_dict(),
            "requirements_hash": self.requirements_hash,
        }


_loaded: dict[tuple[str, str], tuple[Definition, ...]] = {}  # (prefix, sha) -> definitions


def _summary(doc: str | None) -> str:
    """The first paragraph of a docstring, its lines joined."""
    return " ".join(((doc or "").strip().split("\n\n")[0]).split())


def _definition_of(
    module: ModuleType, family: str, id_: str, bundle_files: dict[str, bytes]
) -> Definition:
    """The module's definition: the one ``Strategy`` subclass it defines, its at most one
    ``StrategyConfig`` subclass and the ``requirements`` the class declares (consistent with
    itself, its ``params`` the config's schema)."""
    own = {
        k: v
        for k, v in vars(module).items()
        if isinstance(v, type) and v.__module__ == module.__name__
    }
    strategies = [v for v in own.values() if issubclass(v, Strategy)]
    configs = [v for v in own.values() if issubclass(v, StrategyConfig)]
    if len(strategies) != 1:
        raise BundleError(
            f"{module.__name__} defines {len(strategies)} raptorbt.Strategy classes: it defines"
            " exactly one"
        )
    if len(configs) > 1:
        raise BundleError(
            f"{module.__name__} defines {len(configs)} StrategyConfig classes: at most one"
        )
    cls, config = strategies[0], (configs[0] if configs else None)
    req = getattr(cls, "requirements", None)
    if not isinstance(req, Requirements):
        raise BundleError(
            f"{cls.__name__} declares no requirements: set requirements = Requirements(...)"
            " on the class"
        )
    issues = req.issues()
    if issues:
        raise BundleError(
            f"{cls.__name__}'s requirements: " + "; ".join(f"{i.code}: {i.message}" for i in issues)
        )
    if (config is None) != (req.params is None) or (
        config is not None and req.params != params_schema(config)
    ):
        raise BundleError(
            f"{cls.__name__}: requirements.params is the schema of its one StrategyConfig"
            " (params=<ConfigClass>), and a strategy without a config declares none"
        )
    version = getattr(cls, "version", None)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise BundleError(
            f"{cls.__name__} declares no version: set version = <a whole number from 1> on the"
            " class, and raise it to submit a change"
        )
    name = getattr(cls, "name", None) or cls.__name__
    if not isinstance(name, str):
        raise BundleError(f"{cls.__name__}.name is text")
    libs = [path for path in sorted(bundle_files) if path.startswith("lib/")]
    source = "".join(bundle_files[p].decode() for p in (f"defs/{family}/{id_}.py", *libs))
    return Definition(
        id=id_,
        family=family,
        module=module.__name__,
        cls=cls,
        config=config,
        requirements=req,
        requirements_hash=requirements_hash(cls),
        name=name,
        version=version,
        source=source,
        summary=_summary(cls.__doc__ or module.__doc__),
    )


def load(
    sha: str,
    content: bytes,
    *,
    prefix: str = PREFIX,
    refuse: Callable[[str], str | None] | None = None,
) -> tuple[Definition, ...]:
    """The definitions of bundle ``sha``, by id, imported under ``package_of(sha, prefix)``.
    Every ``defs/<family>/<id>.py`` must load as one definition or the whole bundle is refused
    and nothing of it stays imported. A bundle loaded before (same ``prefix``) is answered as it
    was, whatever ``refuse`` says the second time.

    ``refuse``: called with the module name of every import in the bundle's files, returns the
    reason to refuse that import (the load fails with it and the line) or None to allow it."""
    if (prefix, sha) in _loaded:
        return _loaded[prefix, sha]
    bundle_files = files(sha, content)
    finder = _Finder(sha, bundle_files, prefix, refuse)
    package = finder.package
    sys.meta_path.append(finder)
    found: dict[str, Definition] = {}
    try:
        for name in sorted(finder.sources):
            parts = name.split(".")[1:]
            if name in finder.packages or parts[0] != "defs":
                continue
            if len(parts) != _DEFINITION_PATH_PARTS:
                raise BundleError(f"{name}: a definition is defs/<family>/<id>.py")
            d = _definition_of(importlib.import_module(name), parts[1], parts[2], bundle_files)
            if d.id in found:
                raise BundleError(f"duplicate strategy id {d.id!r}")
            found[d.id] = d
    except BaseException as exc:
        sys.meta_path.remove(finder)
        for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
            del sys.modules[name]
        if isinstance(exc, BundleError):
            raise
        if isinstance(exc, Exception):
            raise BundleError(
                f"bundle {sha}: {type(exc).__name__}: {exc}", getattr(exc, "rule", None)
            ) from exc
        raise
    _loaded[prefix, sha] = tuple(found[k] for k in sorted(found))
    return _loaded[prefix, sha]


def _swap(doc: Any, old: str, new: str) -> Any:
    if isinstance(doc, str):
        for inside in ("defs", "lib"):
            head = f"{old}.{inside}"
            if doc == head or doc.startswith(head + "."):
                return new + doc[len(old) :]
        return doc
    if isinstance(doc, dict):
        return {_swap(k, old, new): _swap(v, old, new) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_swap(v, old, new) for v in doc]
    return doc


def canonical(doc: Any, sha: str, prefix: str = PREFIX) -> Any:
    """``doc`` (JSON) with bundle ``sha``'s module names turned to ``bundle.``: how a checkpoint
    is stored. ``sha`` "": no bundle, nothing to turn."""
    return _swap(doc, package_of(sha, prefix), LIBRARY) if sha else doc


def local(doc: Any, sha: str, prefix: str = PREFIX) -> Any:
    """``doc`` (JSON) with ``bundle.defs``/``bundle.lib`` module names turned to bundle ``sha``'s:
    how a stored checkpoint is handed to a strategy of that bundle."""
    return _swap(doc, LIBRARY, package_of(sha, prefix)) if sha else doc
