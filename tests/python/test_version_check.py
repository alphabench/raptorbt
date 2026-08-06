"""
Pins the update notice: it must be useful, and it must never hurt.

The notice tells a user their raptorbt is behind the newest release. It runs on
the import path of a production trading service, so the tests that matter most
here are the negative ones -- that it stays silent when it is unsure, that it
cannot raise, and that it opens no socket when told not to.

If these fail, the risk is not a missing log line. It is an engine that is
slower to start, noisier than it should be, or that refuses to import at all
because pypi.org is having a bad day.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from raptorbt import version_check
from raptorbt.version_check import check_for_update, is_outdated, parse_version


# --- the comparison itself -------------------------------------------------


@pytest.mark.parametrize(
    "installed,latest,expected",
    [
        ("0.6.2", "0.6.3", True),
        ("0.6.3", "0.6.3", False),
        ("0.6.4", "0.6.3", False),  # ahead of PyPI: a local dev build, not old
        ("0.5.9", "0.6.0", True),
        ("1.0.0", "0.9.9", False),  # 10 > 9 only numerically, not as strings
        ("0.6", "0.6.0", False),  # padded, so equal -- not "older"
        ("0.6", "0.6.1", True),
    ],
)
def test_ordering_is_numeric_not_lexicographic(installed, latest, expected):
    """0.9.9 must not look newer than 1.0.0 because '9' sorts above '1'."""
    assert is_outdated(installed, latest) is expected


@pytest.mark.parametrize(
    "raw",
    ["unknown", "", "0.6.3.dev1", "0.6.3+local", "1.0.0rc1", "v0.6.3", "abc"],
)
def test_unparseable_versions_stay_silent(raw):
    """
    A version this code cannot read with certainty produces no message.

    `__version__` is "unknown" in a source checkout without install metadata,
    and that must not be reported as an outdated release.
    """
    assert parse_version(raw) is None
    assert is_outdated(raw, "9.9.9") is False
    assert is_outdated("0.0.1", raw) is False


# --- the properties that protect the import path ---------------------------


@pytest.mark.parametrize(
    "env_name,env_value",
    [
        ("RAPTORBT_NO_VERSION_CHECK", "1"),
        ("RAPTORBT_NO_VERSION_CHECK", "true"),
        ("RAPTORBT_NO_VERSION_CHECK", "YES"),
        ("CI", "true"),
        ("GITHUB_ACTIONS", "true"),
        ("GITLAB_CI", "true"),
        ("JENKINS_URL", "http://jenkins.local"),
        ("BUILDKITE", "true"),
    ],
)
def test_a_skipping_env_does_no_work_at_all(monkeypatch, env_name, env_value):
    """
    Opting out, or running in CI, must skip the check entirely.

    Both the cache read and the network call are stubbed to record and raise.
    Asserting only that _fetch_latest went uncalled would pass even with all
    skipping removed, because a warm cache short-circuits the fetch on its own
    -- the assertion would hold for a reason unrelated to the guard.
    """
    touched = []

    def forbidden(*_args, **_kwargs):
        touched.append(1)
        raise AssertionError("the check did work in a skipping environment")

    monkeypatch.setattr(version_check, "_read_cache", forbidden)
    monkeypatch.setattr(version_check, "_fetch_latest", forbidden)
    monkeypatch.setattr(version_check, "_run", forbidden)
    _clear_env(monkeypatch)
    monkeypatch.setenv(env_name, env_value)

    check_for_update("0.0.1", blocking=True)

    assert touched == []


def test_an_unset_or_falsey_opt_out_still_checks(monkeypatch):
    """
    The opt-out must be opt-in. RAPTORBT_NO_VERSION_CHECK=0 is not opting out.

    Without this, widening the truthy set to "any value at all" would silently
    disable the feature for anyone who set the variable to 0 to mean "on".
    """
    ran = []
    monkeypatch.setattr(version_check, "_run", lambda installed: ran.append(installed))
    _clear_env(monkeypatch)
    monkeypatch.setenv("RAPTORBT_NO_VERSION_CHECK", "0")

    check_for_update("0.0.1", blocking=True)

    assert ran == ["0.0.1"]


def test_a_failing_network_call_is_silent(monkeypatch, caplog):
    """
    pypi.org being unreachable must produce nothing at all -- no log, no raise.

    This is the common case on a locked-down production host, and it must be
    indistinguishable from the check never having run.
    """
    monkeypatch.setattr(version_check, "_read_cache", lambda: None)
    monkeypatch.setattr(version_check, "_fetch_latest", lambda: None)
    _clear_env(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="raptorbt"):
        check_for_update("0.0.1", blocking=True)

    assert caplog.records == []


def test_an_exploding_fetch_cannot_escape(monkeypatch, caplog):
    """
    If the fetch raises rather than returning None, the import still succeeds.

    A raise here would propagate out of `import raptorbt` and stop a trading
    service from starting over a version nag.

    This asserts on _run directly, not through check_for_update. Both carry a
    catch-all, and going through the outer one would let this pass even with
    the inner guard deleted -- proving only that *some* layer holds, which is
    not what the docstring above claims.
    """

    def boom():
        raise RuntimeError("pypi is on fire")

    monkeypatch.setattr(version_check, "_read_cache", lambda: None)
    monkeypatch.setattr(version_check, "_fetch_latest", boom)
    _clear_env(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="raptorbt"):
        version_check._run("0.0.1")  # must not raise

    assert caplog.records == []


def _run_in_subprocess(body: str, env_extra=None):
    """
    Run `body` in a fresh interpreter with NO logging configured, and return
    its (stdout, stderr).

    A subprocess is the only honest way to test silence. Inside pytest, caplog
    attaches a handler to the root logger, which suppresses logging.lastResort
    -- so a logger.warning() that WOULD print to stderr for a real user prints
    nothing under test. Monkeypatching has the same problem at the thread
    boundary: patching version_check._run leaves _thread_body resolving the
    patched name, so the guard under test is never actually exercised.
    """
    env = dict(os.environ)
    for name in version_check._CI_ENV_VARS:
        env.pop(name, None)
    env.pop("RAPTORBT_NO_VERSION_CHECK", None)
    env["XDG_CACHE_HOME"] = tempfile.mkdtemp()
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    return completed.stdout, completed.stderr


def test_a_raise_on_the_thread_never_reaches_stderr():
    """
    An exception escaping onto the worker thread must not print a traceback.

    This is the loudest failure available to this module, and the one no
    try/except at the call site can catch: Python's default threading.excepthook
    writes the full traceback to stderr itself. Only a guard inside the thread
    body stops it.

    _run's own catch-all is REPLACED here rather than worked around. That is
    the scenario the thread guard exists for -- a future refactor that lets an
    exception out of _run -- and it is the only way to reach the guard, since
    an intact _run absorbs everything before the boundary. Substituting the
    module-level name is sound because _thread_body resolves _run at call time,
    which is exactly the coupling under test.
    """
    stdout, stderr = _run_in_subprocess(
        """
import raptorbt.version_check as vc

# Simulate a future refactor that lets an exception out of _run. The thread
# body is the only thing standing between this and a traceback on stderr.
def leaky_run(installed):
    raise RuntimeError("boom on thread")

vc._run = leaky_run
vc._has_started = False
vc.check_for_update("0.0.1")
for t in __import__("threading").enumerate():
    if t.name == "raptorbt-version-check":
        t.join(timeout=10)
print("done")
"""
    )
    assert "done" in stdout
    assert "Traceback" not in stderr, stderr
    assert "boom on thread" not in stderr, stderr
    assert stderr == "", stderr


def test_the_thread_runs_through_the_guarded_body():
    """
    The thread's target must be _thread_body, not _run directly.

    Pointing the thread at _run removes the last line of defence while every
    behavioural test still passes, because _run is well-behaved today. This
    asserts the wiring itself, since the property it protects is unobservable
    until the day something breaks.
    """
    captured = {}
    real_thread = threading.Thread

    def capture(*args, **kwargs):
        captured["target"] = kwargs.get("target")
        return real_thread(*args, **kwargs)

    monkeypatch_target = version_check.threading.Thread
    version_check.threading.Thread = capture
    try:
        version_check._has_started = False
        os.environ.pop("RAPTORBT_NO_VERSION_CHECK", None)
        saved = {n: os.environ.pop(n, None) for n in version_check._CI_ENV_VARS}
        try:
            version_check.check_for_update("0.0.1")
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value
    finally:
        version_check.threading.Thread = monkeypatch_target

    assert captured["target"] is version_check._thread_body, (
        "the worker must run inside _thread_body's catch-all, not _run directly"
    )
    for thread in threading.enumerate():
        if thread.name == "raptorbt-version-check":
            thread.join(timeout=5)


def test_a_raising_fetch_is_contained_before_the_thread_boundary():
    """
    _run absorbs a fetch that raises, so the thread guard is never needed.

    Defence in depth only counts if each layer is checked on its own. This
    pins the inner layer against a fetch that raises rather than returning
    None -- the realistic form of a urllib change or a proxy returning junk.
    """
    stdout, stderr = _run_in_subprocess(
        """
import raptorbt.version_check as vc
vc._read_cache = lambda: None

def exploding_fetch():
    raise RuntimeError("urllib exploded")

vc._fetch_latest = exploding_fetch
vc._run("0.0.1")   # must return normally, printing nothing
print("done")
"""
    )
    assert "done" in stdout
    assert stderr == "", stderr


def test_nothing_reaches_stderr_with_no_logging_configured():
    """
    The default case for a library: no logging configured, PyPI unreachable.

    With no handler installed, logging.lastResort prints WARNING and above to
    stderr. The notice is INFO precisely so it stays under that bar. If any
    path here logged at WARNING or ERROR -- including the success notice being
    raised to WARNING -- every user with default logging would see it on stderr.

    Must run in a subprocess: caplog would install a handler and mask exactly
    the behaviour being tested.
    """
    stdout, stderr = _run_in_subprocess(
        """
import socket
socket.socket = lambda *a, **k: (_ for _ in ()).throw(OSError("no network"))
import raptorbt.version_check as vc
vc._run("0.0.1")
print("done")
"""
    )
    assert "done" in stdout
    assert stderr == "", stderr


def test_the_success_notice_is_quiet_under_default_logging():
    """
    Even the notice itself must not hit stderr for a user who configured nothing.

    It is genuinely useful information, but it is not a warning -- a library
    telling you to upgrade has not detected a problem with your program. INFO
    keeps it under logging.lastResort's WARNING bar, so it appears for anyone
    who asked for INFO logs and stays invisible to everyone else.
    """
    stdout, stderr = _run_in_subprocess(
        """
import raptorbt.version_check as vc
vc._read_cache = lambda: "99.99.99"
vc._run("0.0.1")
print("done")
"""
    )
    assert "done" in stdout
    assert stderr == "", stderr

    # ...and the same call IS visible once INFO is asked for.
    stdout, stderr = _run_in_subprocess(
        """
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
import raptorbt.version_check as vc
vc._read_cache = lambda: "99.99.99"
vc._run("0.0.1")
print("done")
"""
    )
    assert "done" in stdout
    assert "99.99.99" in stderr and "pip install -U raptorbt" in stderr


def test_import_is_silent_when_the_network_is_dead():
    """
    `import raptorbt` with no network and no logging must print absolutely
    nothing, and must still succeed. This is the shape of a locked-down
    production host.
    """
    stdout, stderr = _run_in_subprocess(
        """
import socket
socket.socket = lambda *a, **k: (_ for _ in ()).throw(OSError("no network"))
import time
import raptorbt
time.sleep(1.0)
print("version:", raptorbt.__version__)
"""
    )
    assert "version:" in stdout
    assert stderr == "", stderr


def test_the_outer_guard_holds_independently(monkeypatch):
    """
    check_for_update swallows a failure even if _run itself starts raising.

    Pinned separately from the test above so that deleting either catch-all is
    caught. Together they are defence in depth; tested only through the outer
    one, they would be defence in depth with one layer silently rotted out.
    """

    def boom(_installed):
        raise RuntimeError("run is broken")

    monkeypatch.setattr(version_check, "_run", boom)
    _clear_env(monkeypatch)

    check_for_update("0.0.1", blocking=True)  # must not raise


def test_the_notice_is_emitted_when_behind(monkeypatch, caplog):
    """The whole point: an older install says so, once, at INFO."""
    monkeypatch.setattr(version_check, "_read_cache", lambda: "9.9.9")
    _clear_env(monkeypatch)

    with caplog.at_level(logging.INFO, logger="raptorbt"):
        check_for_update("0.0.1", blocking=True)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "0.0.1" in message and "9.9.9" in message
    assert "pip install -U raptorbt" in message
    assert "RAPTORBT_NO_VERSION_CHECK" in message


def test_a_current_install_says_nothing(monkeypatch, caplog):
    """Being up to date is the quiet case -- no reassurance line."""
    monkeypatch.setattr(version_check, "_read_cache", lambda: "0.6.3")
    _clear_env(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="raptorbt"):
        check_for_update("0.6.3", blocking=True)

    assert caplog.records == []


# --- the cache -------------------------------------------------------------


def test_a_fresh_cache_prevents_a_second_request(monkeypatch, tmp_path):
    """
    A fleet of services restarting must not become a burst of PyPI requests.

    The cache is on disk precisely so the interval survives process restarts.
    """
    _use_temp_cache(monkeypatch, tmp_path)
    version_check._write_cache("9.9.9")

    called = []
    monkeypatch.setattr(version_check, "_fetch_latest", lambda: called.append(1))
    _clear_env(monkeypatch)

    check_for_update("0.0.1", blocking=True)

    assert called == []
    assert version_check._read_cache() == "9.9.9"


def test_a_stale_cache_is_refetched(monkeypatch, tmp_path):
    """Past the TTL the cached answer is ignored, so advice cannot go old."""
    _use_temp_cache(monkeypatch, tmp_path)
    path = version_check._cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    expired = time.time() - version_check._CACHE_TTL_SECONDS - 1
    path.write_text(json.dumps({"latest": "9.9.9", "checked_at": expired}))

    assert version_check._read_cache() is None


def test_a_corrupt_cache_is_ignored_not_fatal(monkeypatch, tmp_path):
    """Half-written JSON on disk must degrade to a refetch, never to a crash."""
    _use_temp_cache(monkeypatch, tmp_path)
    path = version_check._cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all")

    assert version_check._read_cache() is None


def test_an_unwritable_cache_dir_is_survivable(monkeypatch, tmp_path):
    """A read-only filesystem costs the cache, not the import."""
    _use_temp_cache(monkeypatch, tmp_path / "nested")
    monkeypatch.setattr(
        Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(PermissionError("ro"))
    )

    version_check._write_cache("9.9.9")  # must not raise


# --- the import path itself ------------------------------------------------


def test_the_check_runs_on_a_daemon_thread(monkeypatch):
    """
    The worker must be a daemon thread, and the call must not join it.

    Two separate hazards. A non-daemon thread keeps the interpreter alive at
    shutdown until its socket times out. A join would put the full timeout on
    every process start of every service.

    The thread is captured at construction rather than by enumerating live
    threads -- a finished thread has already left the enumeration, so scanning
    for it asserts nothing at all.
    """
    captured = {}
    real_thread = threading.Thread

    def capture(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        captured["thread"] = thread
        captured["daemon_at_construction"] = kwargs.get("daemon")
        return thread

    monkeypatch.setattr(version_check, "_has_started", False)
    monkeypatch.setattr(version_check.threading, "Thread", capture)
    monkeypatch.setattr(version_check, "_read_cache", lambda: "0.0.1")
    _clear_env(monkeypatch)

    check_for_update("0.0.1")

    assert captured["daemon_at_construction"] is True
    assert captured["thread"].daemon is True
    captured["thread"].join(timeout=5)


def test_a_hanging_fetch_does_not_delay_the_caller(monkeypatch):
    """
    check_for_update returns promptly even while the network call is stuck.

    This is the property that actually protects service startup: if someone
    makes the check synchronous, or joins the thread, this test blocks for the
    full hang and fails on elapsed time.
    """
    release = threading.Event()

    def hang():
        release.wait(timeout=30)
        return None

    monkeypatch.setattr(version_check, "_has_started", False)
    monkeypatch.setattr(version_check, "_read_cache", lambda: None)
    monkeypatch.setattr(version_check, "_fetch_latest", hang)
    _clear_env(monkeypatch)

    started = time.monotonic()
    check_for_update("0.0.1")
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 1.0, f"import path blocked for {elapsed:.2f}s on a hung fetch"


def test_import_exposes_a_version():
    """The wiring in __init__ resolves a version and imports cleanly."""
    import raptorbt

    assert raptorbt.__version__


def _clear_env(monkeypatch):
    monkeypatch.delenv("RAPTORBT_NO_VERSION_CHECK", raising=False)
    for name in version_check._CI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _use_temp_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
