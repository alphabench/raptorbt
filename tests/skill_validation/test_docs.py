"""The front-door documents and the generated API reference stay true to the installed package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

pytestmark = pytest.mark.skipif(not SCRIPTS.is_dir(), reason="scripts/ is not in this tree (an sdist)")


def _run(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, cwd=ROOT)


@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="signatures print slightly differently across interpreters")
def test_api_reference_is_current():
    """``docs/api`` is what ``scripts/gen_api_reference.py`` writes from the installed package."""
    p = _run("gen_api_reference.py", "--check")
    assert p.returncode == 0, p.stderr or p.stdout


def test_front_door_examples_run_and_names_exist():
    """README, the migration guide and the release criteria: examples run, names, paths and links resolve."""
    p = _run("check_docs.py")
    assert p.returncode == 0, p.stderr
