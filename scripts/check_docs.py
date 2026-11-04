#!/usr/bin/env python
"""Check that the front-door documents tell the truth about the installed package.

    python scripts/check_docs.py                  # README.md
    python scripts/check_docs.py docs/foo.md      # other files, held to the same rules
    python scripts/check_docs.py --api            # also fail when docs/api is stale

For each file: every ``python`` block is executed in a subprocess with the network
blocked and ``DeprecationWarning`` as an error (the runner the skills use); a block
marked ``<!-- skill: invalid -->`` on the line above must fail by running, not by a
syntax error; a block marked ``<!-- doc: skip -->`` is illustration (for example a
snippet of a removed API) and is not run. Backticked ``raptorbt.x.y`` names must resolve (files
that talk about removed names pass ``--no-symbols`` through ``NO_SYMBOLS`` below),
relative links must exist, and a backticked path under a repository directory must
exist. It reuses the skill validator's runner, so docs and skills share one rule.

A block fenced ``python raptorbt-example`` needs fixtures (``bars``, ``ticks``, ``depth``,
``config``) that ``tests/python/test_strategy_api.py`` supplies, and that test is its one
checker: this script blanks those blocks (keeping line numbers) and never runs them.
"""

from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

from raptorbt.skills import _check

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ("README.md",)
#: Files that name removed APIs on purpose: their backticked names are not resolved.
NO_SYMBOLS = ()
SKIP = "<!-- doc: skip -->"
_PATH = re.compile(r"`((?:tests|docs|benches|scripts|skills|python|src|\.github)/[A-Za-z0-9_./*\-]+)`")


def _skipped(text: str, line: int) -> bool:
    """Whether the nearest non-blank line above the fence at ``line`` is the skip marker."""
    lines = text.split("\n")
    for j in range(line - 3, -1, -1):  # ``line`` is the first code line, so the fence is line - 2 (0-based)
        if lines[j].strip():
            return lines[j].strip() == SKIP
    return False


_EXAMPLE = re.compile(r"^```python raptorbt-example\n.*?^```[ \t]*$", re.S | re.M)


def _blank_examples(text: str) -> str:
    """Replace each ``raptorbt-example`` block by empty lines, so line numbers stay."""
    return _EXAMPLE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _check_links(rep: _check.Report, path: Path, text: str) -> None:
    for n, line in _check.prose(text):
        for m in _check._LINK.finditer(line):  # noqa: SLF001
            target = m.group(1)
            if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            if not (path.parent / target.split("#")[0]).resolve().exists():
                rep.err(f"{path.relative_to(ROOT).as_posix()}:{n}", f"link {target!r} does not resolve")


def check_file(rep: _check.Report, path: Path, timeout: float) -> int:
    rel = path.relative_to(ROOT).as_posix()
    text = _blank_examples(path.read_text(encoding="utf-8"))
    deprecated = _check._deprecated_names()  # noqa: SLF001
    ran = 0
    for b in _check.code_blocks(text):
        at = f"{rel}:{b.line}"
        if b.lang not in ("python", "py"):
            continue
        if _skipped(text, b.line):
            continue
        if b.invalid:
            code, tail = _check.run_block(b.code, timeout)
            if code is None:
                rep.err(at, f"example timed out after {timeout:g}s")
            elif code == 0:
                rep.err(at, "example is marked invalid but ran without error")
            elif "SyntaxError" in tail:
                rep.err(at, f"invalid-usage example must fail by running, not by a syntax error ({tail})")
            ran += 1
            continue
        if rel not in NO_SYMBOLS:
            for dotted in sorted(_check._symbols_in_python(b.code)):  # noqa: SLF001
                _check._check_symbol(rep, at, dotted)  # noqa: SLF001
            for name in sorted(deprecated):
                if re.search(rf"\b{re.escape(name)}\b", b.code):
                    rep.err(at, f"example uses the deprecated name {name!r}")
        code, tail = _check.run_block(b.code, timeout)
        ran += 1
        if code is None:
            rep.err(at, f"example timed out after {timeout:g}s")
        elif code != 0:
            rep.err(at, f"example failed: {tail}")

    for n, line in _check.prose(text):
        if rel not in NO_SYMBOLS:
            for m in _check._TICKED_SYMBOL.finditer(line):  # noqa: SLF001
                _check._check_symbol(rep, f"{rel}:{n}", m.group(1).rstrip("."))  # noqa: SLF001
        for m in _PATH.finditer(line):
            ref = m.group(1)
            if not (glob.glob(str(ROOT / ref)) or (ROOT / ref).exists()):
                rep.err(f"{rel}:{n}", f"names {ref}, which does not exist")
    _check_links(rep, path, text)
    return ran


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="markdown files, relative to the repository (default: the front-door set)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds allowed per code example")
    ap.add_argument("--api", action="store_true", help="also run gen_api_reference.py --check")
    args = ap.parse_args(argv)

    rep = _check.Report()
    total = 0
    for name in args.files or DEFAULT:
        path = ROOT / name
        if not path.is_file():
            rep.err(name, "file does not exist")
            continue
        total += check_file(rep, path, args.timeout)
    for e in rep.errors:
        print(e, file=sys.stderr)
    status = 1 if rep.errors else 0
    if args.api:
        status |= subprocess.call([sys.executable, str(ROOT / "scripts" / "gen_api_reference.py"), "--check"])
    print(f"check_docs: {len(rep.errors)} error(s), {total} example(s) run")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
