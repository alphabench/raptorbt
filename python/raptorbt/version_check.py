"""
Tells you, once a day, when the installed raptorbt is behind the newest release.

Plain words: this module asks pypi.org what the latest published version of
raptorbt is, compares it to the one you have, and writes a single log line if
yours is older. That is all it does. It never changes how the engine behaves,
never raises, and never delays your program.

Four properties are load-bearing, because this runs on the import path of a
production trading service:

1. **It cannot block.** The network call happens on a daemon thread. `import
   raptorbt` returns at once whether or not pypi.org answers, and a hung
   socket cannot keep the process alive at shutdown.
2. **It cannot fail.** Every path is wrapped. No DNS, a proxy, a firewall, a
   read-only cache directory, or a malformed response all end the same way --
   silence.
3. **It cannot spam.** The answer is cached on disk for a day, so a fleet of
   services restarting does not mean a burst of requests, and a busy process
   checks once per interpreter regardless.
4. **It can be turned off.** Set RAPTORBT_NO_VERSION_CHECK=1 and no socket is
   ever opened. Continuous-integration environments are skipped automatically.

The comparison is deliberately conservative: anything it cannot parse with
certainty is treated as "not behind", so a version scheme this code does not
understand produces no message rather than a wrong one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

__all__ = ["check_for_update", "is_outdated", "parse_version"]

logger = logging.getLogger("raptorbt")

_PYPI_URL = "https://pypi.org/pypi/raptorbt/json"

# A version check is worth a moment, never a stall. Two seconds is long enough
# for a healthy round trip and short enough that a black-holed connection is
# abandoned well before anyone notices.
_TIMEOUT_SECONDS = 2.0

# One check per day per machine. The upper bound on how stale the advice can be
# is a day, which is far tighter than a release cadence.
_CACHE_TTL_SECONDS = 24 * 60 * 60

_OPT_OUT_ENV = "RAPTORBT_NO_VERSION_CHECK"

# Set by every mainstream CI provider. A pinned wheel in CI is a deliberate
# choice, so telling the log it is old is noise, not news.
_CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "BUILDKITE")

# Guards against a second check if raptorbt is imported from several threads
# before the first has finished.
_started = threading.Lock()
_has_started = False


def parse_version(raw: str) -> Optional[Tuple[int, ...]]:
    """
    Turn "0.6.3" into (0, 6, 3), or return None if that is not safe to do.

    Only plain dotted release numbers are understood. A pre-release, a local
    version, or anything else returns None, and a None on either side of the
    comparison means no message is emitted. Being silent about a version we do
    not understand is always better than being wrong about it.
    """
    if not raw:
        return None
    if not re.fullmatch(r"\d+(\.\d+)*", raw.strip()):
        return None
    try:
        return tuple(int(part) for part in raw.strip().split("."))
    except ValueError:  # pragma: no cover - the regex already excludes this
        return None


def is_outdated(installed: str, latest: str) -> bool:
    """
    True only when `installed` is certainly an older release than `latest`.

    Shorter tuples are padded, so 0.6 and 0.6.0 compare equal rather than the
    shorter one looking older. Anything unparseable is False.
    """
    current = parse_version(installed)
    newest = parse_version(latest)
    if current is None or newest is None:
        return False
    width = max(len(current), len(newest))
    current += (0,) * (width - len(current))
    newest += (0,) * (width - len(newest))
    return current < newest


def _cache_path() -> Path:
    """Where the last answer from pypi.org is remembered between runs."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "raptorbt" / "latest_version.json"


def _read_cache() -> Optional[str]:
    """The cached version string, or None if absent, stale, or unreadable."""
    try:
        raw = json.loads(_cache_path().read_text())
        if time.time() - float(raw["checked_at"]) > _CACHE_TTL_SECONDS:
            return None
        version = raw["latest"]
        return version if isinstance(version, str) else None
    except Exception:
        return None


def _write_cache(latest: str) -> None:
    """Remember an answer. A read-only or full disk is not an error here."""
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"latest": latest, "checked_at": time.time()}))
    except Exception:
        pass


def _fetch_latest() -> Optional[str]:
    """Ask pypi.org for the newest published version. None on any failure."""
    try:
        from urllib.request import Request, urlopen

        request = Request(_PYPI_URL, headers={"Accept": "application/json"})
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latest = payload["info"]["version"]
        return latest if isinstance(latest, str) else None
    except Exception:
        return None


def _should_skip() -> bool:
    """True when the user, or the environment, has said not to check."""
    if os.environ.get(_OPT_OUT_ENV, "").strip().lower() in ("1", "true", "yes"):
        return True
    return any(os.environ.get(name) for name in _CI_ENV_VARS)


def _thread_body(installed: str) -> None:
    """
    What the worker thread actually runs. The last line of defence against noise.

    _run already swallows everything, so in practice this never catches. It
    exists because the failure it guards is the loudest one available: anything
    escaping onto a thread is printed as a full traceback to stderr by Python's
    default threading.excepthook, which no try/except at the call site can
    intercept. One refactor inside _run that lets an exception through would put
    a traceback in a production trading log. The cost of the guard is a function
    call; the cost of not having it is the one outcome this module promises
    cannot happen.
    """
    try:
        _run(installed)
    except BaseException:
        pass


def _run(installed: str) -> None:
    """The body of the check. Runs on the worker thread; never raises."""
    try:
        latest = _read_cache()
        if latest is None:
            latest = _fetch_latest()
            if latest is None:
                return
            _write_cache(latest)
        if is_outdated(installed, latest):
            logger.info(
                "raptorbt %s is behind the latest release %s. "
                "Install the latest version: pip install -U raptorbt "
                "(set %s=1 to silence this).",
                installed,
                latest,
                _OPT_OUT_ENV,
            )
    except Exception:
        pass


def check_for_update(installed: str, blocking: bool = False) -> None:
    """
    Start the check for `installed`. Returns immediately unless blocking=True.

    Called once from raptorbt/__init__.py at import. `blocking` exists so tests
    can assert on the outcome without waiting on a thread; nothing in the
    library ever passes it.
    """
    global _has_started
    try:
        if _should_skip():
            return
        if blocking:
            _run(installed)
            return
        with _started:
            if _has_started:
                return
            _has_started = True
        # A daemon thread cannot hold up interpreter shutdown, so a slow or
        # hung request can never delay the process exiting.
        threading.Thread(
            target=_thread_body,
            args=(installed,),
            name="raptorbt-version-check",
            daemon=True,
        ).start()
    except Exception:
        pass
