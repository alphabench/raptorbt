"""The skill system: the accessor, and the validator on real and broken skills."""

from __future__ import annotations

import doctest
import json
import shutil
import subprocess
import sys

import pytest

import raptorbt
import raptorbt.skills as skills
from raptorbt import errors
from raptorbt.skills import _check

REAL = _check._skills_dir().resolve()


@pytest.fixture
def tree(tmp_path):
    """A writable copy of the skills directory."""
    dst = tmp_path / "skills"
    shutil.copytree(REAL, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def edit_manifest(tree, fn):
    p = tree / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    fn(m["skills"])
    p.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    # Keep INDEX.md in step so a test breaks only the thing it means to.
    idx = tree / "INDEX.md"
    text = idx.read_text(encoding="utf-8")
    a, b = text.index(_check.REGISTRY_START) + len(_check.REGISTRY_START), text.index(_check.REGISTRY_END)
    idx.write_text(text[:a] + "\n\n" + _check.render_registry(m["skills"]) + "\n" + text[b:], encoding="utf-8")


def skill(m, domain):
    return next(s for s in m if s["domain"] == domain)


def bare_draft(tree, domain):
    """Reduce ``domain`` to an empty draft in the copy, whatever it is in the real tree."""
    path = tree / domain / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    end = text.index("\n---\n", 4) + 5
    body = "\n# Skeleton\n\n" + "".join(f"## {i}. {t}\n\n" for i, t in enumerate(_check.SECTIONS, 1))
    path.write_text(text[:end].replace("status: stable", "status: draft") + body, encoding="utf-8")
    edit_manifest(tree, lambda m: skill(m, domain).update(status="draft"))


def stable_with(tree, block):
    """Make the 'results' skill stable, with one extra example appended to section 5.

    The real 'results' skill is written, so the copy is first reduced to a bare draft.
    """
    bare_draft(tree, "results")
    edit_manifest(tree, lambda m: skill(m, "results").update(status="stable"))
    path = tree / "results" / "SKILL.md"
    edit(path, "status: draft", "status: stable")
    edit(path, "## 5. Examples\n", f"## 5. Examples\n\n{block}\n")
    return path


def errs(tree, **kw):
    return _check.validate(tree, **kw).errors


def has(errors_, *words):
    return any(all(w in e for w in words) for e in errors_)


# ------------------------------------------------------------------ the real skills


def test_real_skills_pass():
    assert _check.validate(REAL).errors == []


def test_real_skills_strict_only_fail_on_drafts():
    rep = _check.validate(REAL, strict=True)
    assert [e for e in rep.errors if "still a draft" not in e] == []


def test_strict_passes_once_nothing_is_draft():
    if any(s.status == "draft" for s in skills.index()):
        pytest.skip("drafts remain; strict is the M4 gate")
    assert _check.validate(REAL, strict=True).errors == []


def test_every_error_skill_is_in_the_manifest():
    domains = {s.domain for s in skills.index()}
    assert set(errors.SKILLS) <= domains
    assert {c.skill for c in errors.CODES.values()} <= domains


def test_core_examples_are_run():
    blocks = _check.code_blocks(skills.load("core"))
    assert sum(not b.invalid for b in blocks if b.lang == "python") >= 3
    assert any(b.invalid for b in blocks)


# ----------------------------------------------------------------------- accessor


def test_index_and_manifest_agree():
    m = skills.manifest()
    assert [s.name for s in skills.index()] == [s["name"] for s in m["skills"]]
    assert len(m["skills"]) == 19


def test_load_accepts_both_spellings():
    assert skills.load("core") == skills.load("raptorbt-core")
    assert skills.load("core").startswith("---\nname: raptorbt-core")


def test_load_unknown_skill():
    with pytest.raises(KeyError, match="no skill"):
        skills.load("nope")


@pytest.mark.parametrize(
    "task, first",
    [
        ("set a stop-loss and target", "raptorbt-orders"),
        ("compute ATR and RSI", "raptorbt-indicators"),
        ("my call raised an error", "raptorbt-troubleshooting"),
        ("combine 5m and 15m bars", "raptorbt-timeframes"),
    ],
)
def test_for_task(task, first):
    assert skills.for_task(task)[0] == first


def test_for_task_no_match():
    assert skills.for_task("xyzzy plugh") == []


def test_module_doctests():
    assert doctest.testmod(skills).failed == 0


def test_cli_check_and_for_task():
    ok = subprocess.run([sys.executable, "-m", "raptorbt.skills", "check"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    out = subprocess.run(
        [sys.executable, "-m", "raptorbt.skills", "for-task", "set a stop-loss and target"], capture_output=True, text=True
    )
    assert out.stdout.split()[0] == "raptorbt-orders"
    strict = subprocess.run([sys.executable, "-m", "raptorbt.skills", "check", "--strict"], capture_output=True, text=True)
    assert strict.returncode == (1 if any(s.status == "draft" for s in skills.index()) else 0)


# ------------------------------------------------------------- structure checks


def test_broken_link(tree):
    edit(tree / "core" / "SKILL.md", "(../strategy/SKILL.md)", "(../strategy/GONE.md)")
    assert has(errs(tree), "core/SKILL.md", "does not resolve")


def test_link_leaving_the_directory(tree):
    edit(tree / "core" / "SKILL.md", "(../strategy/SKILL.md)", "(../../docs/errors.md)")
    assert has(errs(tree), "leaves the skills directory")


def test_missing_section(tree):
    edit(tree / "core" / "SKILL.md", "## 8. Common mistakes", "## 8. Pitfalls")
    assert has(errs(tree), "the ten sections")


def test_frontmatter_differs_from_manifest(tree):
    edit(tree / "core" / "SKILL.md", "api_version: 1.x", "api_version: 3.x")
    assert has(errs(tree), "frontmatter api_version")


def test_frontmatter_unknown_key(tree):
    edit(tree / "core" / "SKILL.md", "status: stable", "status: stable\nowner: me")
    assert has(errs(tree), "frontmatter keys must be exactly")


def test_frontmatter_unclosed(tree):
    edit(tree / "execution" / "SKILL.md", "status: stable\n---", "status: stable")
    assert has(errs(tree), "execution/SKILL.md", "not closed")


def test_manifest_api_version_must_match_package(tree):
    edit_manifest(tree, lambda m: skill(m, "core").update(api_version="9.x"))
    assert has(errs(tree), "does not match installed")


def test_manifest_file_missing(tree):
    (tree / "risk" / "SKILL.md").unlink()
    assert has(errs(tree), "risk/SKILL.md", "missing")


def test_skill_file_not_in_manifest(tree):
    (tree / "extra").mkdir()
    shutil.copy(tree / "risk" / "SKILL.md", tree / "extra" / "SKILL.md")
    assert has(errs(tree), "extra/SKILL.md", "not in manifest.json")


def test_dependency_unknown(tree):
    edit_manifest(tree, lambda m: skill(m, "risk")["depends_on"].append("raptorbt-ghost"))
    assert has(errs(tree), "'raptorbt-ghost' is not a skill")


def test_dependency_cycle(tree):
    edit_manifest(tree, lambda m: skill(m, "core")["depends_on"].append("raptorbt-risk"))
    edit(tree / "core" / "SKILL.md", "depends_on: []", "depends_on:\n    - raptorbt-risk")
    assert has(errs(tree), "dependency cycle")


def test_modules_entry_must_resolve(tree):
    edit_manifest(tree, lambda m: skill(m, "risk")["modules"].append("raptorbt.no_such_thing"))
    assert has(errs(tree), "raptorbt.no_such_thing", "does not resolve")


def test_index_out_of_date(tree):
    edit(tree / "INDEX.md", "Where a strategy's orders go:", "Something else entirely:")
    assert has(errs(tree), "INDEX.md", "out of date")


def test_error_names_a_skill_not_in_the_manifest(tree):
    def drop(m):
        m[:] = [s for s in m if s["domain"] != "troubleshooting"]
        for s in m:
            s["depends_on"] = [d for d in s["depends_on"] if d != "raptorbt-troubleshooting"]

    edit_manifest(tree, drop)
    shutil.rmtree(tree / "troubleshooting")
    assert has(errs(tree), "errors.py", "'troubleshooting'")


# ------------------------------------------------------------- example checks


def test_failing_example(tree):
    stable_with(tree, "```python\nraise RuntimeError('boom')\n```\n")
    assert has(errs(tree), "results/SKILL.md", "example failed", "boom")


def test_example_needing_the_network_fails(tree):
    stable_with(tree, "```python\nimport socket\nsocket.create_connection(('127.0.0.1', 9))\n```\n")
    assert has(errs(tree), "example failed", "network access is blocked")


def test_example_timeout(tree):
    stable_with(tree, "```python\nwhile True:\n    pass\n```\n")
    assert has(_check.validate(tree, timeout=1.0).errors, "timed out")


def test_deprecation_warning_fails_an_example(tree):
    stable_with(tree, "```python\nimport warnings\nwarnings.warn('old', DeprecationWarning)\n```\n")
    assert has(errs(tree), "example failed", "DeprecationWarning")


def test_invalid_example_must_fail(tree):
    stable_with(tree, "<!-- skill: invalid -->\n```python\nx = 1\n```\n")
    assert has(errs(tree), "marked invalid but ran without error")


def test_invalid_example_may_not_be_a_syntax_error(tree):
    stable_with(tree, "<!-- skill: invalid -->\n```python\nx = (\n```\n")
    assert has(errs(tree), "not by a syntax error")


def test_invalid_example_that_fails_is_fine(tree):
    stable_with(tree, "<!-- skill: invalid -->\n```python\nraise ValueError('shown on purpose')\n```\n")
    assert errs(tree) == []


def test_unmarked_failure_is_not_excused_by_prose(tree):
    stable_with(tree, "This example is invalid.\n\n```python\nraise ValueError('x')\n```\n")
    assert has(errs(tree), "example failed")


def test_unknown_symbol_in_code(tree):
    stable_with(tree, "```python\nimport raptorbt\nraptorbt.no_such_function\n```\n")
    assert has(errs(tree), "raptorbt.no_such_function", "does not exist")


def test_unknown_symbol_in_prose(tree):
    stable_with(tree, "Call `raptorbt.run_everything()` to do it all.\n")
    assert has(errs(tree), "raptorbt.run_everything", "does not exist")


def test_from_import_of_missing_name(tree):
    stable_with(tree, "```python\nfrom raptorbt import Missing\n```\n")
    assert has(errs(tree), "raptorbt.Missing", "does not exist")


def test_private_internals_are_refused(tree):
    stable_with(tree, "Use `raptorbt._raptorbt.KernelSession` directly.\n")
    assert has(errs(tree), "private internals")


@pytest.mark.filterwarnings("ignore:.*PortfolioContext:DeprecationWarning")
def test_deprecated_name_in_code(tree):
    stable_with(tree, "```python\nimport raptorbt\nctx_type = raptorbt.PortfolioContext\n```\n")
    assert has(errs(tree), "deprecated name 'PortfolioContext'")


def test_deprecated_hook_alias_in_code(tree):
    alias = next(iter(__import__("raptorbt.strategy.api", fromlist=["x"]).DEPRECATED))
    stable_with(tree, f"```python\nclass S:\n    def {alias}(self): ...\n```\n")
    assert has(errs(tree), f"deprecated name {alias!r}")


def test_deprecated_name_in_prose_needs_the_word_deprecated(tree):
    stable_with(tree, "Pass `upon_bar_close` to choose the timing.\n")
    assert has(errs(tree), "upon_bar_close", "without saying it is deprecated")
    edit(tree / "results" / "SKILL.md", "Pass `upon_bar_close` to", "Do not pass the deprecated `upon_bar_close` to")
    assert not has(errs(tree), "upon_bar_close")


# ----------------------------------------------------------------------- strict


def test_strict_flags_a_symbol_no_skill_covers(tree):
    edit_manifest(tree, lambda m: skill(m, "indicators")["modules"].remove("raptorbt.sma"))
    assert not has(errs(tree), "raptorbt.sma ")
    assert has(errs(tree, strict=True), "raptorbt.sma", "not covered")


def test_strict_flags_an_uncovered_module(tree):
    edit_manifest(tree, lambda m: skill(m, "execution")["modules"].remove("raptorbt.costs"))
    assert has(errs(tree, strict=True), "raptorbt.costs", "not covered")


def test_strict_accepts_a_module_entry_for_its_names(tree):
    # raptorbt.pricing covers every name in its __all__ without listing them.
    assert not has(errs(tree, strict=True), "raptorbt.pricing")


def test_strict_flags_drafts(tree):
    assert has(errs(tree, strict=True) if any(s.status == "draft" for s in skills.index()) else ["still a draft"], "still a draft")


def test_public_surface_includes_every_all_name():
    surface = _check.public_surface()
    listed = set(surface["raptorbt"]) | {n.split(".")[1] for n in surface if n.count(".") == 1}
    assert set(raptorbt.__all__) <= listed
