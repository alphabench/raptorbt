"""
Optional, off-by-default crash reports for raptorbt itself.

Plain words: if you turn this on and point it at a collector, a bug in the
engine (an internal panic, not an error you caused) sends one small anonymous
report saying which raptorbt function crashed and where in the engine's source.
Nothing is sent unless you enable it, and nothing is sent until you give it an
endpoint and key: enabled without them, it logs one line and sends nothing.

Enable with ``RAPTORBT_CRASH_REPORTING=1`` or ``raptorbt.enable_crash_reporting()``.
Point it at your own collector with its address and project key, as arguments or as
``RAPTORBT_CRASH_ENDPOINT`` and ``RAPTORBT_CRASH_KEY``.

What a report holds (and nothing else): ``project`` ("raptorbt"), ``version``,
``error_type`` ("PanicException"), ``operation`` (the raptorbt function that
crashed), ``location`` (engine source ``file:line``), ``os``, ``arch``, the Python
major.minor version, a random id that lives only as long as the process, and a
one-line ``$exception_list`` entry built from the operation and location. Never
the panic message, argument values, data, symbols, prices, your code or paths,
environment variables or the host name.

Only an engine panic is reported. A ``RaptorBTError`` is a refusal you can fix,
and an exception raised by your own strategy callback is your code; neither is
ever sent, nor is a panic raised by another extension. A panic that is a structured money or fee-schedule error is not sent.

Load-bearing properties, as for the version check: the send happens on a daemon
thread with a two second timeout, nothing here ever raises into your program, at
most a few reports leave a process, and the same crash is reported once.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import threading
import uuid
from typing import Any, Dict, Optional, Tuple

__all__ = ["enable_crash_reporting", "report_panic"]

logger = logging.getLogger("raptorbt")

_ENABLE_ENV = "RAPTORBT_CRASH_REPORTING"
_ENDPOINT_ENV = "RAPTORBT_CRASH_ENDPOINT"
_KEY_ENV = "RAPTORBT_CRASH_KEY"

_TIMEOUT_SECONDS = 2.0

# A crash loop must not become a request loop.
_MAX_REPORTS = 5

_lock = threading.Lock()
_forced = False
_endpoint_arg: Optional[str] = None
_key_arg: Optional[str] = None
_seen: set = set()
_sent = 0
_warned_unconfigured = False
_distinct_id: Optional[str] = None

# Set while a report is being built or sent on this thread, so nothing the reporter
# calls can report a crash of its own.
_local = threading.local()

_LOCATION = re.compile(r"[\w.\-/]+:\d+")


def enable_crash_reporting(endpoint: Optional[str] = None, key: Optional[str] = None) -> bool:
    """
    Turn crash reporting on for this process; return whether it can send.

    ``endpoint`` and ``key`` default to ``RAPTORBT_CRASH_ENDPOINT`` and
    ``RAPTORBT_CRASH_KEY``. If either is empty, nothing is sent and the function returns
    False.
    """
    global _forced, _endpoint_arg, _key_arg
    with _lock:
        _forced = True
        _native_location()  # drop a location left by a panic from before reporting was on
        if endpoint is not None:
            _endpoint_arg = endpoint
        if key is not None:
            _key_arg = key
    return _config() is not None


def _enabled() -> bool:
    return _forced or os.environ.get(_ENABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def _config() -> Optional[Tuple[str, str]]:
    """The (endpoint, key) to send to, or None (logging once at DEBUG when enabled without them)."""
    global _warned_unconfigured
    endpoint = _endpoint_arg or os.environ.get(_ENDPOINT_ENV, "").strip()
    key = _key_arg or os.environ.get(_KEY_ENV, "").strip()
    if endpoint.lower().startswith(("http://", "https://")) and key:
        return endpoint, key
    if not _warned_unconfigured:
        _warned_unconfigured = True
        logger.debug(
            "raptorbt crash reporting is enabled but has no collector endpoint configured: set %s and %s "
            "(or pass endpoint and key to enable_crash_reporting). Nothing is sent.",
            _ENDPOINT_ENV,
            _KEY_ENV,
        )
    return None


def _native_location() -> Optional[str]:
    try:
        from raptorbt import _raptorbt

        return _raptorbt._last_panic_location()
    except Exception:
        return None


def _clean_location(raw: Optional[str]) -> Optional[str]:
    """Reduce the engine's ``file:line`` to a package-relative path; None if it is not one."""
    if not raw or not _LOCATION.fullmatch(raw):
        return None
    path, _, line = raw.rpartition(":")
    marker = "/registry/src/"
    if marker in path:
        # a dependency: drop the registry directory, keep "crate-version/src/file.rs"
        path = path.split(marker, 1)[1].split("/", 1)[-1]
    elif not path.startswith("src/"):
        i = path.rfind("/src/")
        path = path[i + 1 :] if i >= 0 else path.rsplit("/", 1)[-1]
    return f"{path}:{line}"


def _payload(key: str, operation: str, location: Optional[str]) -> Dict[str, Any]:
    global _distinct_id
    import raptorbt

    if _distinct_id is None:
        _distinct_id = str(uuid.uuid4())
    summary = f"panic in {operation}" + (f" at {location}" if location else "")
    properties: Dict[str, Any] = {
        "project": "raptorbt",
        "version": str(getattr(raptorbt, "__version__", "unknown")),
        "error_type": "PanicException",
        "operation": operation,
        "location": location,
        "os": platform.system(),
        "arch": platform.machine(),
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "$exception_list": [
            {"type": "PanicException", "value": summary, "mechanism": {"handled": False, "synthetic": True}}
        ],
        "$process_person_profile": False,
        "$geoip_disable": True,
    }
    return {
        "api_key": key,
        "batch": [{"event": "$exception", "distinct_id": _distinct_id, "properties": properties}],
    }


def _send(endpoint: str, body: bytes) -> None:
    """POST the report. Runs on the worker thread; never raises. Plain urllib, deliberately
    outside raptorbt's own adapters and guards."""
    _local.busy = True
    try:
        from urllib.request import Request, urlopen

        url = endpoint.rstrip("/")
        if not url.endswith(("/batch", "/e")):
            url += "/batch"
        # A named User-Agent: some collectors' browser checks refuse urllib's default one.
        headers = {"Content-Type": "application/json", "User-Agent": "raptorbt-crash-report"}
        request = Request(url, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            response.read()
    except BaseException:
        pass
    finally:
        _local.busy = False


def report_panic(operation: str, exc: Optional[BaseException] = None) -> None:
    """
    Report an engine panic raised by ``operation``. Called by the guard; a no-op unless enabled.

    Never raises and never blocks: the request goes out on a daemon thread.
    """
    global _sent
    try:
        if getattr(_local, "busy", False):
            return
        _local.busy = True
        try:
            # Take the engine's panic location first, whatever happens next, so it is consumed
            # once. A real raptorbt panic always wrote one; without it the exception is not ours
            # (another extension's PanicException, say) and nothing is reported.
            location = _clean_location(_native_location())
            if location is None or not _enabled():
                return
            if exc is not None:
                if getattr(exc, "_raptorbt_reported", False):
                    return  # already reported by an inner guarded call
                try:
                    exc._raptorbt_reported = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            config = _config()
            if config is None:
                return
            endpoint, key = config
            identity = (operation, location)
            with _lock:
                if identity in _seen or _sent >= _MAX_REPORTS:
                    return
                _seen.add(identity)
                _sent += 1
            body = json.dumps(_payload(key, operation, location)).encode("utf-8")
            threading.Thread(
                target=_send, args=(endpoint, body), name="raptorbt-crash-report", daemon=True
            ).start()
        finally:
            _local.busy = False
    except Exception:
        pass
