"""Skills: domain guides an AI agent loads to learn only the API it needs.

Each public API domain has one ``SKILL.md``. ``manifest()`` is the machine
registry, ``index()`` lists the skills, ``for_task(text)`` picks the ones a
task needs (and what they depend on), and ``load(name)`` returns a skill's text. The files ship in the
wheel, so this works from an installed package without the repository.

>>> import raptorbt.skills as skills
>>> "raptorbt-core" in [s.name for s in skills.index()]
True
>>> skills.for_task("set a stop-loss and target")[0]
'raptorbt-orders'

``python -m raptorbt.skills check`` verifies every skill against the package.
"""

from __future__ import annotations

import functools
import json
import math
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
    "that the this to use using want with write my me your do does get need should what which "
    "when then there their them these those we you find show give tell list".split()
)

#: Weights of a word found in a skill's text, its domain name, and its function and module names.
_W_TEXT, _W_DOMAIN, _W_NAME = 1.0, 3.0, 0.75


def _stem(w: str) -> str:
    """Crude stemming so ``orders`` meets ``order`` and ``analyses`` meets ``analysis``."""
    if len(w) <= 3:
        return w
    if w.endswith(("ysis", "yses")):
        return w[:-2]  # analysis, analyses -> analys
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"  # strategies -> strategy
    if w.endswith(("sses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("ation") and len(w) > 7:
        return w[:-3]  # cointegration -> cointegrat, estimation -> estimat
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "is", "us")):
        return w[:-1]
    if w.endswith("e") and len(w) > 3:
        return w[:-1]  # compute / computing, explore / exploring
    return w


def _tokens(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def _name_tokens(name: str) -> set[str]:
    """Words of an identifier: ``markov_regimes`` -> markov, regime; ``cdldoji`` also doji."""
    out = _tokens(name.replace("_", " "))
    if name.startswith("cdl") and len(name) > 3:
        out.add(_stem(name[3:]))
    return out


@functools.lru_cache(maxsize=1)
def _index() -> tuple[list[SkillInfo], list[tuple[set[str], set[str], set[str]]], dict[str, float]]:
    """Per skill: (words of its text, words of its domain, words of its function and module
    names), and the idf of every word across skills (rare words count for more)."""
    from raptorbt import expressions, stats

    infos = index()
    named: dict[str, set[str]] = {s.name: set() for s in infos}
    for s in infos:
        for mod in s.modules:
            named[s.name] |= _name_tokens(mod.rsplit(".", 1)[-1])
    named["raptorbt-expressions"] |= {
        t for row in expressions._registry().values()
        for t in (*_name_tokens(row["name"]), *_tokens(row["talib_name"] or ""))
    }
    named["raptorbt-statistics"] |= {t for f in stats.__all__ for t in _name_tokens(f)}
    rows = []
    for s in infos:
        body = _tokens(" ".join((s.title, s.description, *s.load_when)))
        domain = _tokens(s.domain.replace("-", " "))
        rows.append((body, domain, named[s.name]))
    df: dict[str, int] = {}
    for body, domain, names in rows:
        for t in body | domain | names:
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(1 + len(infos) / n) for t, n in df.items()}
    return infos, rows, idf


def for_task(text: str) -> list[str]:
    """Skill names that fit the task, best first, then what they depend on.

    Words are stemmed and weighted by how rare they are across skills (a word only one skill
    uses counts for more than one every skill uses). A match on the domain name counts three
    times, and a match on a function or module name (``doji``, ``garch``, ``sizing``) counts
    three quarters. The skills that match come first, in score order (ties keep registry order); the
    skills they depend on, transitively and not already listed, follow in dependency order, so
    reading the list from the top gives the answer and reading it to the end gives everything
    needed. Returns ``[]`` when nothing matches.
    """
    query = _tokens(text)
    infos, rows, idf = _index()
    scored = []
    for pos, (s, (body, domain, names)) in enumerate(zip(infos, rows)):
        score = (
            _W_TEXT * sum(idf[t] for t in query & body)
            + _W_DOMAIN * sum(idf[t] for t in query & domain)
            + _W_NAME * sum(idf[t] for t in query & names)
        )
        if score:
            scored.append((-score, pos, s.name))
    matched = [name for _, _, name in sorted(scored)]
    by_name = {s.name: s for s in infos}
    out = list(matched)

    def need(name: str) -> None:
        for dep in by_name[name].depends_on:
            need(dep)
            if dep not in out:
                out.append(dep)

    for name in matched:
        need(name)
    return out
