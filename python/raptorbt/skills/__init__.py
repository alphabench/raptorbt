"""Skills: domain guides an AI agent loads to learn only the API it needs.

Each public API domain has one ``SKILL.md``. ``manifest()`` is the machine
registry, ``index()`` lists the skills, ``for_task(text)`` picks the ones a
task needs, and ``load(name)`` returns a skill's text. The files ship in the
wheel, so this works from an installed package without the repository.

>>> import raptorbt.skills as skills
>>> "raptorbt-core" in [s.name for s in skills.index()]
True
>>> skills.for_task("set a stop-loss and target")[0]
'raptorbt-orders'

``python -m raptorbt.skills check`` verifies every skill against the package.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from raptorbt.errors import UnknownSymbol

__all__ = ["SkillInfo", "for_task", "index", "load", "manifest"]

_PREFIX = "raptorbt-"


@dataclass(frozen=True)
class SkillInfo:
    """One manifest entry. ``name`` is ``raptorbt-<domain>``."""

    name: str
    domain: str
    title: str
    path: str
    description: str
    load_when: tuple[str, ...]
    depends_on: tuple[str, ...]
    api_version: str
    status: str
    modules: tuple[str, ...]


def _root():
    return resources.files("raptorbt.skills").joinpath("_data")


def manifest() -> dict[str, Any]:
    """The parsed ``manifest.json``."""
    return json.loads(_root().joinpath("manifest.json").read_text(encoding="utf-8"))


def index() -> list[SkillInfo]:
    """Every skill in the manifest, in registry order."""
    out = []
    for s in manifest()["skills"]:
        out.append(
            SkillInfo(
                name=s["name"],
                domain=s["domain"],
                title=s["title"],
                path=s["path"],
                description=s["description"],
                load_when=tuple(s["load_when"]),
                depends_on=tuple(s["depends_on"]),
                api_version=s["api_version"],
                status=s["status"],
                modules=tuple(s["modules"]),
            )
        )
    return out


def load(name: str) -> str:
    """The text of one skill. ``name`` is ``"raptorbt-core"`` or ``"core"``."""
    by_name = {s.name: s for s in index()}
    full = name if name.startswith(_PREFIX) else _PREFIX + name
    if full not in by_name:
        raise UnknownSymbol(
            f"no skill {name!r}; available: {', '.join(sorted(by_name))}",
            code="unknown_skill", operation="skills.load", parameter="name", value=name,
            valid=sorted(by_name),
        )
    return _root().joinpath(by_name[full].path).read_text(encoding="utf-8")


_STOP = frozenset(
    "a an and are as at be build by can for from how i in into is it its make of on or "
    "that the this to use using want with write my me your".split()
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Crude plural/gerund stemming is enough to match "orders" to "order".
    return {re.sub(r"(ing|es|s)$", "", w) if len(w) > 4 else w.rstrip("s") for w in words if w not in _STOP}


def for_task(text: str) -> list[str]:
    """Skill names whose domain, description or ``load_when`` match the task.

    Plain keyword matching, best match first (ties keep registry order). A
    match on the domain name counts three times. It does not add
    dependencies: read ``SkillInfo.depends_on`` for those. Returns ``[]`` when
    nothing matches.
    """
    query = _tokens(text)
    scored = []
    for pos, s in enumerate(index()):
        body = _tokens(" ".join((s.title, s.description, *s.load_when)))
        domain = _tokens(s.domain.replace("-", " "))
        score = len(query & body) + 3 * len(query & domain)
        if score:
            scored.append((-score, pos, s.name))
    return [name for _, _, name in sorted(scored)]
