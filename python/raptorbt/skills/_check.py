"""The skill validator behind ``python -m raptorbt.skills check``.

A skill is documentation that has to stay true, so every claim a machine can
test is tested here: the manifest, the files, the frontmatter, the links, the
symbols named, the deprecated names, and every Python example (run in a
subprocess with the network blocked). Draft skills are held only to the
structure checks; their examples are not run. ``--strict`` adds coverage and
forbids drafts.
"""

from __future__ import annotations

import ast
import concurrent.futures
import functools
import importlib
import json
import os
import pkgutil
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

SECTIONS = (
    "When to use this skill",
    "Domain overview",
    "API reference",
    "Standard workflow",
    "Examples",
    "Composition",
    "Constraints and correctness",
    "Common mistakes",
    "Validation",
    "Related skills",
)
STATUSES = ("draft", "stable")
FRONTMATTER_KEYS = ("name", "description", "version", "api_version", "depends_on", "status")
MANIFEST_KEYS = (
    "name", "domain", "title", "path", "description", "load_when",
    "depends_on", "api_version", "status", "modules",
)  # fmt: skip
#: A python block that shows invalid usage says so on the line above its fence.
INVALID_MARKER = "<!-- skill: invalid -->"
REGISTRY_START = "<!-- registry:start -->"
REGISTRY_END = "<!-- registry:end -->"

#: Deprecated names that are not hook aliases (those come from
#: ``raptorbt.strategy.api.DEPRECATED``). ``PortfolioContext`` is the alias
#: kept by ``raptorbt.__getattr__``; ``upon_bar_close`` is the flag
#: ``fill_timing`` replaced. Keep in step with ``tests/python/test_deprecated_names.py``.
DEPRECATED_EXTRA = ("PortfolioContext", "upon_bar_close")

_FENCE = re.compile(r"^(```|~~~)\s*([A-Za-z0-9_+-]*)\s*$")
_LINK = re.compile(r"(?<!\!)\[[^\]\n]*\]\(([^)\s]+)\)")
_TICKED_SYMBOL = re.compile(r"`(raptorbt(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")

# Runs one example: block the network, then execute the file as __main__.
_RUNNER = r"""
import socket, sys
def _blocked(*a, **k):
    raise OSError("network access is blocked while running skill examples")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.getaddrinfo = _blocked
socket.create_connection = _blocked
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    code = compile(f.read(), path, "exec")
exec(code, {"__name__": "__main__", "__file__": path})
"""


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")


@dataclass
class Block:
    line: int  # 1-based line of the first code line
    lang: str
    code: str
    invalid: bool


# --------------------------------------------------------------------- parsing


def parse_frontmatter(text: str) -> tuple[dict[str, Any], int]:
    """Parse the ``---`` block. Returns (fields, line number after it).

    Supports exactly what the template uses: ``key: scalar``, ``key: []``,
    ``key: >`` with indented continuation lines, and ``key:`` followed by
    indented ``- item`` lines. Anything else is an error, so the format stays
    readable without a YAML dependency.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("file must start with a '---' frontmatter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("frontmatter is not closed by '---'") from None
    out: dict[str, Any] = {}
    i = 1
    while i < end:
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue
        m = re.fullmatch(r"([a-z_]+):\s*(.*)", raw)
        if not m or raw[0] == " ":
            raise ValueError(f"line {i + 1}: unsupported frontmatter line {raw!r}")
        key, rest = m.group(1), m.group(2).strip()
        i += 1
        if key in out:
            raise ValueError(f"line {i}: duplicate key {key!r}")
        if rest in (">", "|"):
            parts = []
            while i < end and (lines[i].startswith(" ") or not lines[i].strip()):
                parts.append(lines[i].strip())
                i += 1
            out[key] = " ".join(p for p in parts if p)
        elif rest == "":
            items = []
            while i < end and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:].strip())
                i += 1
            out[key] = items
        elif rest == "[]":
            out[key] = []
        else:
            out[key] = rest
    return out, end + 2


def code_blocks(text: str) -> list[Block]:
    """Fenced code blocks, with whether the line above carries the invalid marker."""
    lines = text.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, lang = m.group(1), m.group(2).lower()
        prev = next((lines[j].strip() for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        invalid = prev == INVALID_MARKER
        start = i + 2
        j = i + 1
        while j < len(lines) and not lines[j].startswith(fence):
            j += 1
        blocks.append(Block(start, lang, "\n".join(lines[i + 1 : j]), invalid))
        i = j + 1
    return blocks


def prose(text: str) -> list[tuple[int, str]]:
    """(line number, line) for every line outside fenced blocks."""
    out, fence = [], None
    for n, line in enumerate(text.split("\n"), 1):
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)
        elif fence is not None and line.startswith(fence):
            fence = None
        elif fence is None:
            out.append((n, line))
    return out


# -------------------------------------------------------------------- registry


def render_registry(skills: list[dict[str, Any]]) -> str:
    """The generated middle of INDEX.md."""
    out = []
    for s in skills:
        deps = ", ".join(s["depends_on"]) or "none"
        out += [
            f"## {s['title']}",
            "",
            s["description"],
            "",
            f"Name: `{s['name']}`  ",
            f"Path: [{s['path']}]({s['path']})  ",
            f"Status: {s['status']}  ",
            f"API version: {s['api_version']}  ",
            f"Depends on: {deps}",
            "",
            "Load when:",
            "",
            *[f"- {w}" for w in s["load_when"]],
            "",
        ]
    return "\n".join(out)


def _skills_dir() -> Path:
    from raptorbt import skills

    return Path(str(skills._root()))


def print_or_write_index(write: bool, out: IO[str]) -> int:
    root = _skills_dir()
    skills = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["skills"]
    block = render_registry(skills)
    if not write:
        out.write(block + "\n")
        return 0
    path = root / "INDEX.md"
    text = path.read_text(encoding="utf-8")
    a, b = text.index(REGISTRY_START) + len(REGISTRY_START), text.index(REGISTRY_END)
    path.write_text(text[:a] + "\n\n" + block + "\n" + text[b:], encoding="utf-8")
    out.write(f"wrote {path}\n")
    return 0


# ---------------------------------------------------------------------- checks


def resolve(dotted: str) -> Any:
    """Import ``raptorbt.a.b`` as far as modules go, then getattr the rest."""
    parts = dotted.split(".")
    obj: Any = importlib.import_module(parts[0])
    for k in range(1, len(parts)):
        try:
            obj = importlib.import_module(".".join(parts[: k + 1]))
        except ImportError:
            obj = getattr(obj, parts[k])
    return obj


def _deprecated_names() -> set[str]:
    names = set(DEPRECATED_EXTRA)
    try:
        from raptorbt.strategy.api import DEPRECATED

        names |= set(DEPRECATED)
    except ImportError:  # pragma: no cover - the table moved; the extra list still applies
        pass
    return names


def _symbols_in_python(code: str) -> set[str]:
    """Dotted ``raptorbt...`` names a code block imports or attribute-accesses."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    aliases = {"raptorbt": "raptorbt"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "raptorbt" or a.name.startswith("raptorbt."):
                    aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else "raptorbt"
                    found.add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "raptorbt" or node.module.startswith("raptorbt.")
        ):
            found.add(node.module)
            for a in node.names:
                if a.name != "*":
                    found.add(f"{node.module}.{a.name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain, cur = [], node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id in aliases:
                found.add(".".join([aliases[cur.id], *reversed(chain)]))
    return found


def _check_symbol(rep: Report, where: str, dotted: str) -> None:
    if any(p.startswith("_") for p in dotted.split(".")[1:]):
        rep.err(where, f"{dotted}: skills must not name private internals")
        return
    try:
        resolve(dotted)
    except Exception as e:  # noqa: BLE001 - any failure to resolve is the finding
        rep.err(where, f"{dotted} does not exist in the installed package ({type(e).__name__}: {e})")


@functools.lru_cache(maxsize=None)
def run_block(code: str, timeout: float) -> tuple[int | None, str]:
    """Execute one example. Returns (exit code or None on timeout, stderr tail)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["RAPTORBT_NO_VERSION_CHECK"] = "1"
    for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env[proxy] = "http://127.0.0.1:9"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "example.py"
        path.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, "-W", "error::DeprecationWarning", "-c", _RUNNER, str(path)],
                capture_output=True, text=True, timeout=timeout, env=env, cwd=tmp,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return None, ""
    return p.returncode, p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""


def prerun_examples(root: Path, skills: list[dict[str, Any]], timeout: float) -> None:
    """Run every example of the non-draft skills at once, so the checks below find them cached.

    Each example is a fresh interpreter and shares nothing with another, so the order they run in
    cannot change a result; check_skill_text still reports the findings in source order.
    """
    codes: set[str] = set()
    for s in skills:
        f = root / s["path"]
        if s["status"] != "draft" and f.is_file():
            codes |= {b.code for b in code_blocks(f.read_text(encoding="utf-8")) if b.lang in ("python", "py")}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        list(pool.map(lambda code: run_block(code, timeout), codes))


def check_skill_text(rep: Report, where: str, text: str, body_from: int, draft: bool, timeout: float) -> None:
    """Checks on a skill body: sections, then (non-draft) examples, symbols, deprecations."""
    headings = [(n, ln) for n, ln in prose(text) if ln.startswith("## ")]
    found = [ln[3:].strip() for _, ln in headings]
    want = [f"{i}. {t}" for i, t in enumerate(SECTIONS, 1)]
    if found != want:
        rep.err(where, f"the ten sections must be, in order: {'; '.join(want)} (found: {'; '.join(found) or 'none'})")
    if draft:
        return

    deprecated = _deprecated_names()
    blocks = [b for b in code_blocks(text)]
    for b in blocks:
        at = f"{where}:{b.line}"
        if b.lang not in ("python", "py"):
            if b.invalid:
                rep.err(at, f"{INVALID_MARKER} marks a block that is not python")
            continue
        if b.invalid:
            code, tail = run_block(b.code, timeout)
            if code is None:
                rep.err(at, f"example timed out after {timeout:g}s")
            elif code == 0:
                rep.err(at, "example is marked invalid but ran without error")
            elif "SyntaxError" in tail:
                rep.err(at, f"invalid-usage example must fail by running, not by a syntax error ({tail})")
            continue
        for dotted in sorted(_symbols_in_python(b.code)):
            _check_symbol(rep, at, dotted)
        for name in sorted(deprecated):
            if re.search(rf"\b{re.escape(name)}\b", b.code):
                rep.err(at, f"example uses the deprecated name {name!r}")
        code, tail = run_block(b.code, timeout)
        if code is None:
            rep.err(at, f"example timed out after {timeout:g}s")
        elif code != 0:
            rep.err(at, f"example failed: {tail}")

    for n, line in prose(text):
        if n < body_from:
            continue
        for m in _TICKED_SYMBOL.finditer(line):
            _check_symbol(rep, f"{where}:{n}", m.group(1).rstrip("."))
        if "deprecated" not in line.lower():
            for name in sorted(deprecated):
                if re.search(rf"`[^`]*\b{re.escape(name)}\b[^`]*`", line):
                    rep.err(f"{where}:{n}", f"names the deprecated {name!r} without saying it is deprecated")


def check_links(rep: Report, root: Path, file: Path) -> None:
    for n, line in prose(file.read_text(encoding="utf-8")):
        for m in _LINK.finditer(line):
            target = m.group(1)
            if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path = (file.parent / target.split("#")[0]).resolve()
            where = f"{file.relative_to(root)}:{n}"
            if not path.is_relative_to(root.resolve()):
                rep.err(where, f"link {target!r} leaves the skills directory (it would break in the wheel)")
            elif not path.exists():
                rep.err(where, f"link {target!r} does not resolve")


def find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(n: str) -> list[str] | None:
        state[n] = 1
        stack.append(n)
        for d in graph.get(n, []):
            if state.get(d) == 1:
                return stack[stack.index(d) :] + [d]
            if d in graph and state.get(d) is None:
                if (c := visit(d)) is not None:
                    return c
        stack.pop()
        state[n] = 2
        return None

    for n in graph:
        if state.get(n) is None and (c := visit(n)) is not None:
            return c
    return None


def public_surface() -> dict[str, list[str]]:
    """Public module -> its public names. The root's names come from ``__all__``."""
    import raptorbt

    surface = {"raptorbt": [n for n in raptorbt.__all__ if not isinstance(getattr(raptorbt, n, None), type(raptorbt))]}
    for mi in pkgutil.walk_packages(raptorbt.__path__, "raptorbt."):
        parts = mi.name.split(".")
        if any(p.startswith("_") for p in parts):
            continue
        if len(parts) > 2 and not mi.ispkg:
            continue  # implementation modules; reached through their package's names
        mod = importlib.import_module(mi.name)
        surface[mi.name] = list(getattr(mod, "__all__", []))
    for n in raptorbt.__all__:  # submodules re-exported in __all__ (raptorbt.errors)
        if isinstance(getattr(raptorbt, n, None), type(raptorbt)):
            surface.setdefault(f"raptorbt.{n}", list(getattr(getattr(raptorbt, n), "__all__", [])))
    return surface


def check_coverage(rep: Report, skills: list[dict[str, Any]]) -> None:
    entries = {m for s in skills for m in s["modules"]}
    covered_objs = []
    for e in entries:
        try:
            covered_objs.append(resolve(e))
        except Exception:  # reported elsewhere
            pass
    for mod, names in sorted(public_surface().items()):
        if mod != "raptorbt" and mod in entries:
            continue
        if mod != "raptorbt":
            rep.err("coverage", f"public module {mod} is not covered by any skill's modules")
        for n in names:
            dotted = f"{mod}.{n}"
            if dotted in entries:
                continue
            obj = getattr(importlib.import_module(mod), n, None)
            if callable(obj) and any(obj is c for c in covered_objs):
                continue  # covered under another public path (a re-export)
            rep.err("coverage", f"public symbol {dotted} is not covered by any skill's modules")


def validate(root: Path, strict: bool = False, timeout: float = 60.0) -> Report:
    import raptorbt
    from raptorbt import errors

    rep = Report()
    mpath = root / "manifest.json"
    if not mpath.is_file():
        rep.err(str(mpath), "manifest.json is missing")
        return rep
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        skills = manifest["skills"]
    except (ValueError, KeyError) as e:
        rep.err("manifest.json", f"not a manifest: {e}")
        return rep

    # -- manifest schema
    names: dict[str, dict[str, Any]] = {}
    for s in skills:
        label = f"manifest.json[{s.get('name', '?')}]"
        if set(s) != set(MANIFEST_KEYS):
            rep.err(label, f"keys must be exactly {sorted(MANIFEST_KEYS)}, got {sorted(s)}")
            continue
        if s["name"] in names:
            rep.err(label, "duplicate name")
        names[s["name"]] = s
        if s["name"] != f"raptorbt-{s['domain']}":
            rep.err(label, f"name must be raptorbt-{s['domain']}")
        if s["path"] != f"{s['domain']}/SKILL.md":
            rep.err(label, f"path must be {s['domain']}/SKILL.md")
        if s["status"] not in STATUSES:
            rep.err(label, f"status must be one of {STATUSES}")
        if not re.fullmatch(r"\d+\.x", s["api_version"]):
            rep.err(label, "api_version must look like '1.x'")
        for k in ("load_when", "depends_on", "modules"):
            if not isinstance(s[k], list) or not all(isinstance(x, str) for x in s[k]):
                rep.err(label, f"{k} must be a list of strings")
        if not s["load_when"]:
            rep.err(label, "load_when must name at least one situation")
        m = re.match(r"(\d+)", getattr(raptorbt, "__version__", "") or "")
        if m and re.fullmatch(r"\d+\.x", s["api_version"]) and s["api_version"].split(".")[0] != m.group(1):
            rep.err(label, f"api_version {s['api_version']} does not match installed raptorbt {raptorbt.__version__}")

    prerun_examples(root, list(names.values()), timeout)

    # -- files <-> manifest
    on_disk = {p.parent.name for p in root.glob("*/SKILL.md") if not p.parent.name.startswith("_")}
    for d in sorted(on_disk - {s["domain"] for s in names.values()}):
        rep.err(f"{d}/SKILL.md", "exists but is not in manifest.json")
    for s in names.values():
        f = root / s["path"]
        if not f.is_file():
            rep.err(s["path"], "listed in manifest.json but missing")
            continue
        text = f.read_text(encoding="utf-8")
        try:
            fm, body_from = parse_frontmatter(text)
        except ValueError as e:
            rep.err(s["path"], str(e))
            continue
        if set(fm) != set(FRONTMATTER_KEYS):
            rep.err(s["path"], f"frontmatter keys must be exactly {list(FRONTMATTER_KEYS)}, got {sorted(fm)}")
            continue
        for k in ("name", "description", "api_version", "depends_on", "status"):
            if fm[k] != s[k]:
                rep.err(s["path"], f"frontmatter {k} {fm[k]!r} differs from manifest {s[k]!r}")
        if not re.fullmatch(r"\d+(\.\d+)*", fm["version"]):
            rep.err(s["path"], f"frontmatter version {fm['version']!r} is not a number like 1.0")
        check_skill_text(rep, s["path"], text, body_from, s["status"] == "draft", timeout)
        check_links(rep, root, f)

    # -- dependencies
    graph = {n: s["depends_on"] for n, s in names.items()}
    for n, deps in graph.items():
        for d in deps:
            if d == n:
                rep.err(f"manifest.json[{n}]", "depends on itself")
            elif d not in names:
                rep.err(f"manifest.json[{n}]", f"depends_on {d!r} is not a skill")
    if (cycle := find_cycle({n: [d for d in g if d != n] for n, g in graph.items()})) is not None:
        rep.err("manifest.json", f"dependency cycle: {' -> '.join(cycle)}")

    # -- modules resolve
    for s in names.values():
        for mod in s["modules"]:
            try:
                resolve(mod)
            except Exception as e:  # noqa: BLE001
                rep.err(f"manifest.json[{s['name']}]", f"modules entry {mod!r} does not resolve ({type(e).__name__}: {e})")

    # -- INDEX
    index = root / "INDEX.md"
    if not index.is_file():
        rep.err("INDEX.md", "missing")
    else:
        itext = index.read_text(encoding="utf-8")
        if REGISTRY_START not in itext or REGISTRY_END not in itext:
            rep.err("INDEX.md", f"needs the {REGISTRY_START} ... {REGISTRY_END} markers around the registry")
        else:
            a, b = itext.index(REGISTRY_START) + len(REGISTRY_START), itext.index(REGISTRY_END)
            if itext[a:b].strip() != render_registry(skills).strip():
                rep.err("INDEX.md", "registry is out of date with manifest.json; run python -m raptorbt.skills index --write")
        check_links(rep, root, index)

    # -- structured errors name real skills
    domains = {s["domain"] for s in names.values()}
    for dom in errors.SKILLS:
        if dom not in domains:
            rep.err("errors.py", f"errors.SKILLS names {dom!r}, which is not a skill in the manifest")
    for code, entry in errors.CODES.items():
        if entry.skill not in domains:
            rep.err("errors.py", f"error code {code!r} names skill {entry.skill!r}, which is not in the manifest")

    # -- strict
    drafts = [s["name"] for s in names.values() if s["status"] == "draft"]
    if strict:
        for n in drafts:
            rep.err(f"manifest.json[{n}]", "is still a draft")
        check_coverage(rep, skills)
    elif drafts:
        rep.notes.append(f"{len(drafts)} of {len(names)} skills are drafts (their examples are not run; --strict forbids them)")
    return rep


def run(root: Path | None, strict: bool, timeout: float, out: IO[str]) -> int:
    root = root or _skills_dir()
    rep = validate(Path(root), strict=strict, timeout=timeout)
    for e in rep.errors:
        out.write(f"error: {e}\n")
    for n in rep.notes:
        out.write(f"note: {n}\n")
    out.write(f"skills check{' --strict' if strict else ''}: {len(rep.errors)} error(s)\n")
    return 1 if rep.errors else 0
