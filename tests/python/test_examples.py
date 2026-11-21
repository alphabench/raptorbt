"""Every script under ``examples/`` runs, offline, to completion and without a warning.

The examples are what a user (or an AI agent) copies from, so a script that
stops running, or begins to warn because an API it uses was deprecated, is a defect.
Each is run in its own interpreter with every warning turned into an error. A script
must also say what it shows and which skills it uses, print a result, and be reachable
from the skills (``skills/INDEX.md``).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
FOLDERS = (
    "analysis",
    "single-instrument",
    "multi-instrument",
    "multi-timeframe",
    "options",
    "portfolio",
    "tick-level",
    "live-trading",
)
SCRIPTS = sorted(p for p in EXAMPLES.glob("*/*.py") if not p.name.startswith("_"))


def _id(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def test_every_folder_of_the_examples_tree_holds_a_script():
    assert sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir() and not p.name.startswith("_")) == sorted(FOLDERS)
    for folder in FOLDERS:
        assert 1 <= len(list((EXAMPLES / folder).glob("*.py"))) <= 3, folder


@pytest.mark.parametrize("script", SCRIPTS, ids=_id)
def test_the_example_runs_clean(script):
    done = subprocess.run(
        [sys.executable, "-W", "error", str(script)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, f"{_id(script)} exited {done.returncode}:\n{done.stderr[-2000:]}"
    assert "Warning" not in done.stderr, done.stderr[-2000:]
    assert done.stdout.strip(), f"{_id(script)} printed no result"


@pytest.mark.parametrize("script", SCRIPTS, ids=_id)
def test_the_example_says_what_it_shows_and_which_skills_it_uses(script):
    doc = ast.get_docstring(ast.parse(script.read_text()))
    assert doc and "Skills: raptorbt-" in doc, f"{_id(script)} needs a header docstring ending in its skills"
    skills = {w.strip(" ,.") for w in doc.split("Skills:")[1].split() if w.startswith("raptorbt-")}
    known = {f"raptorbt-{p.parent.name}" for p in (ROOT / "skills").glob("*/SKILL.md")}
    assert skills and skills <= known, (_id(script), skills - known)


def _ticks_only(script: Path) -> bool:
    """A tick-level script, or one elsewhere whose only run is ``run_tick_strategy``."""
    text = script.read_text()
    bar_runs = ("run_strategies(", "run_strategy_backtest(", "run_strategies_windowed(", "LiveNode(")
    return script.parent.name == "tick-level" or (
        "run_tick_strategy(" in text and not any(r in text for r in bar_runs)
    )


# a tick-only run takes no bars, so it has no stamp to declare
@pytest.mark.parametrize("script", [s for s in SCRIPTS if not _ticks_only(s)], ids=_id)
def test_the_example_passes_the_bar_stamp_it_means(script):
    assert "bar_stamp=" in script.read_text(), f"{_id(script)} must pass bar_stamp explicitly"


def test_the_skills_and_their_index_point_at_every_example():
    index = (ROOT / "skills" / "INDEX.md").read_text()
    for script in SCRIPTS:
        rel = f"examples/{_id(script)}"
        assert rel in index, f"skills/INDEX.md does not map a task to {rel}"
        linked = [
            p for p in (ROOT / "skills").glob("*/SKILL.md") if rel in p.read_text()
        ]
        assert linked, f"no SKILL.md links {rel}"
