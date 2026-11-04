"""F-15: a deliberate error in the package is a structured ``RaptorBTError``.

A bare ``raise ValueError/TypeError/KeyError/RuntimeError`` in a public module has no ``code``,
``parameter`` or ``valid``, so an agent cannot act on it without parsing prose. This lists the
ones that remain; each exception below is a protocol or a bootstrap, with its reason.
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "python" / "raptorbt"
BARE = {"ValueError", "TypeError", "KeyError", "RuntimeError"}

#: (file relative to the package, enclosing function) -> why it stays bare.
ALLOWED = {
    ("errors.py", "__init__"): "RaptorBTError validating its own skill argument; it cannot raise itself",
    ("adapters/base.py", "__reduce__"): "pickle and deepcopy refuse a Secret by the protocol's TypeError",
    ("skills/_check.py", "parse_frontmatter"): "the skills validator reports these as findings, not to a caller",
}


def _bare_raises():
    found = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Raise) or node.exc is None:
                    continue
                call = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(call, ast.Name) and call.id in BARE:
                    found.append((path.relative_to(PKG).as_posix(), fn.name, node.lineno))
    return found


def test_no_public_module_raises_a_bare_builtin_error():
    left = [f for f in _bare_raises() if (f[0], f[1]) not in ALLOWED]
    assert left == [], "raise a RaptorBTError (raptorbt.errors) with a registered code instead: " + repr(left)


def test_every_allowed_exception_still_exists():
    seen = {(f[0], f[1]) for f in _bare_raises()}
    assert set(ALLOWED) <= seen, sorted(set(ALLOWED) - seen)
