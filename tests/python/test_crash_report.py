"""
Pins the opt-in crash reporter: silent unless enabled, and when enabled it sends an engine
panic's function and source line and nothing else.

A real engine panic is not reproducible on demand, so the tests raise an exception named
``PanicException`` (the name the guard keys on) from a function wrapped by ``_guard._wrap`` and
fake the native panic location. The endpoint is a local http.server.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import raptorbt
from raptorbt import _guard, crash_report, errors

THREAD = "raptorbt-crash-report"
ALLOWED_PROPERTIES = {
    "project",
    "version",
    "error_type",
    "operation",
    "location",
    "os",
    "arch",
    "python",
    "$exception_list",
    "$process_person_profile",
    "$geoip_disable",
}
SECRET = "secret-symbol-NIFTY24500CE-at-123.45"


class PanicException(BaseException):
    """Same name as pyo3's, which is what the guard checks."""


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    for name in ("RAPTORBT_CRASH_REPORTING", "RAPTORBT_CRASH_ENDPOINT", "RAPTORBT_CRASH_KEY"):
        monkeypatch.delenv(name, raising=False)
    for attr, value in (
        ("_forced", False),
        ("_endpoint_arg", None),
        ("_key_arg", None),
        ("_sent", 0),
        ("_warned_unconfigured", False),
        ("_distinct_id", None),
    ):
        monkeypatch.setattr(crash_report, attr, value)
    monkeypatch.setattr(crash_report, "_seen", set())
    monkeypatch.setattr(crash_report, "_native_location", lambda: "src/execution/fees.rs:42")
    yield
    _join_senders()


def _join_senders():
    for t in threading.enumerate():
        if t.name == THREAD:
            t.join(timeout=5)


@pytest.fixture
def collector():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append((self.path, json.loads(self.rfile.read(length))))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.url = f"http://127.0.0.1:{server.server_port}"
    server.received = received
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _wait(collector, count, timeout=5.0):
    end = time.time() + timeout
    while len(collector.received) < count and time.time() < end:
        time.sleep(0.01)
    time.sleep(0.1)  # let a wrongly sent extra report arrive
    return collector.received


def _guarded(exc, operation="run_single_backtest"):
    def fn(*args, **kwargs):
        raise exc

    return _guard._wrap(fn, operation)


def _enable(collector):
    assert crash_report.enable_crash_reporting(endpoint=collector.url, key="phc_test") is True


def test_off_by_default_no_thread_no_socket(collector, monkeypatch):
    # a collector is configured, yet being off still opens nothing
    monkeypatch.setenv("RAPTORBT_CRASH_ENDPOINT", collector.url)
    monkeypatch.setenv("RAPTORBT_CRASH_KEY", "phc_env")
    def forbidden(*a, **k):
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(PanicException):
        _guarded(PanicException(SECRET))()
    assert not [t for t in threading.enumerate() if t.name == THREAD]
    assert {t.ident for t in threading.enumerate()} <= before
    assert collector.received == []


def test_enabled_without_collector_does_nothing(monkeypatch, caplog):
    monkeypatch.setenv("RAPTORBT_CRASH_REPORTING", "1")
    with caplog.at_level("DEBUG", logger="raptorbt"):
        for _ in range(2):
            with pytest.raises(PanicException):
                _guarded(PanicException("x"))()
    assert not [t for t in threading.enumerate() if t.name == THREAD]
    assert sum("no collector" in r.getMessage() for r in caplog.records) == 1
    assert crash_report.enable_crash_reporting() is False


def test_argument_overrides_environment(collector, monkeypatch):
    monkeypatch.setenv("RAPTORBT_CRASH_ENDPOINT", collector.url)
    monkeypatch.setenv("RAPTORBT_CRASH_KEY", "phc_env")
    assert crash_report._config() == (collector.url, "phc_env")
    assert crash_report.enable_crash_reporting(endpoint=collector.url, key="phc_arg") is True
    assert crash_report._config() == (collector.url, "phc_arg")
    with pytest.raises(PanicException):
        _guarded(PanicException("m"))()
    (_, body), = _wait(collector, 1)
    assert body["api_key"] == "phc_arg"


def test_panic_sends_exactly_the_allowed_fields_and_no_message(collector):
    _enable(collector)
    with pytest.raises(PanicException):
        _guarded(PanicException(SECRET))("positional", token=SECRET)
    (path, body), = _wait(collector, 1)
    assert path == "/batch"
    assert body["api_key"] == "phc_test"
    (event,) = body["batch"]
    assert event["event"] == "$exception"
    assert len(event["distinct_id"]) == 36
    props = event["properties"]
    assert set(props) == ALLOWED_PROPERTIES
    assert props["project"] == "raptorbt"
    assert props["version"] == raptorbt.__version__
    assert props["error_type"] == "PanicException"
    assert props["operation"] == "run_single_backtest"
    assert props["location"] == "src/execution/fees.rs:42"
    assert props["$process_person_profile"] is False
    assert props["$geoip_disable"] is True
    assert "secret" not in json.dumps(body) and "NIFTY" not in json.dumps(body)


def test_enabled_by_environment(collector, monkeypatch):
    monkeypatch.setenv("RAPTORBT_CRASH_REPORTING", "1")
    monkeypatch.setenv("RAPTORBT_CRASH_ENDPOINT", collector.url)
    monkeypatch.setenv("RAPTORBT_CRASH_KEY", "phc_env")
    with pytest.raises(PanicException):
        _guarded(PanicException("m"))()
    (_, body), = _wait(collector, 1)
    assert body["api_key"] == "phc_env"


def test_refusals_and_user_exceptions_are_never_reported(collector):
    _enable(collector)
    for exc in (
        errors.InvalidParameter("bad", code="invalid_input"),
        ValueError(SECRET),  # a user strategy callback's own exception
        RuntimeError(SECRET),
        KeyboardInterrupt(),
    ):
        with pytest.raises(type(exc)):
            _guarded(exc)()
    assert _wait(collector, 1, timeout=0.5) == []


def test_structured_money_and_fee_faults_are_not_reported(collector):
    _enable(collector)
    with pytest.raises(errors.MoneyOverflow):
        _guarded(PanicException("money_overflow: amount 1e30"))()
    with pytest.raises(errors.InvalidParameter):
        _guarded(PanicException("fee_schedule_gap: no rate for 2020-01-01"))()
    assert _wait(collector, 1, timeout=0.5) == []


def test_endpoint_down_never_raises_or_blocks():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens here now
    assert crash_report.enable_crash_reporting(endpoint=f"http://127.0.0.1:{port}", key="k") is True
    start = time.time()
    with pytest.raises(PanicException):
        _guarded(PanicException("m"))()
    assert time.time() - start < 1.0
    _join_senders()


def test_endpoint_that_never_answers_does_not_block():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        port = listener.getsockname()[1]
        crash_report.enable_crash_reporting(endpoint=f"http://127.0.0.1:{port}", key="k")
        start = time.time()
        with pytest.raises(PanicException):
            _guarded(PanicException("m"))()
        assert time.time() - start < 1.0
        _join_senders()  # the sender gives up at its own two second timeout
    finally:
        listener.close()


def test_same_crash_is_reported_once(collector):
    _enable(collector)
    for _ in range(3):
        with pytest.raises(PanicException):
            _guarded(PanicException("m"))()
    assert len(_wait(collector, 1)) == 1


def test_cap_per_process(collector, monkeypatch):
    _enable(collector)
    n = 0

    def fake_location():
        return f"src/x.rs:{n}"

    monkeypatch.setattr(crash_report, "_native_location", fake_location)
    for n in range(crash_report._MAX_REPORTS + 4):
        with pytest.raises(PanicException):
            _guarded(PanicException("m"))()
    assert len(_wait(collector, crash_report._MAX_REPORTS)) == crash_report._MAX_REPORTS


def test_reentrancy_does_not_report_recursively(collector, monkeypatch):
    _enable(collector)
    guarded = _guarded(PanicException("inner"), "inner_op")

    def location_that_panics_again():
        # a guarded call made while the reporter runs on this thread must not report
        with pytest.raises(PanicException):
            guarded()
        return "src/x.rs:1"

    monkeypatch.setattr(crash_report, "_native_location", location_that_panics_again)
    with pytest.raises(PanicException):
        _guarded(PanicException("outer"), "outer_op")()
    received = _wait(collector, 1)
    assert [b["batch"][0]["properties"]["operation"] for _, b in received] == ["outer_op"]


def test_nested_guarded_calls_report_one_panic_once(collector):
    _enable(collector)
    exc = PanicException("m")
    inner = _guarded(exc, "inner_op")

    def outer_fn():
        inner()

    with pytest.raises(PanicException):
        _guard._wrap(outer_fn, "outer_op")()
    received = _wait(collector, 1)
    assert [b["batch"][0]["properties"]["operation"] for _, b in received] == ["inner_op"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("src/execution/fees.rs:42", "src/execution/fees.rs:42"),
        ("/Users/someone/proj/raptorbt/src/lib.rs:9", "src/lib.rs:9"),
        ("/home/u/.cargo/registry/src/index.crates.io-abc/rayon-1.10.0/src/iter.rs:7", "rayon-1.10.0/src/iter.rs:7"),
        ("weird.rs:3", "weird.rs:3"),
        ("has space/file.rs:3", None),
        ("no-line.rs", None),
        (None, None),
    ],
)
def test_location_is_package_relative(raw, expected):
    assert crash_report._clean_location(raw) == expected


def test_report_panic_never_raises(monkeypatch):
    monkeypatch.setenv("RAPTORBT_CRASH_REPORTING", "1")
    monkeypatch.setenv("RAPTORBT_CRASH_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("RAPTORBT_CRASH_KEY", "k")

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(crash_report, "_payload", boom)
    crash_report.report_panic("op")


def test_no_native_location_means_not_ours_and_nothing_is_sent(collector, monkeypatch):
    _enable(collector)
    monkeypatch.setattr(crash_report, "_native_location", lambda: None)
    with pytest.raises(PanicException):
        _guarded(PanicException("a foreign extension's panic"))()
    assert _wait(collector, 1) == []


@pytest.mark.parametrize(
    "suffix, expected",
    [("", "/batch"), ("/", "/batch"), ("/batch", "/batch"), ("/batch/", "/batch"), ("/e", "/e")],
)
def test_endpoint_path_is_not_doubled(collector, suffix, expected):
    assert crash_report.enable_crash_reporting(endpoint=collector.url + suffix, key="k") is True
    with pytest.raises(PanicException):
        _guarded(PanicException("m"))()
    (path, _), = _wait(collector, 1)
    assert path == expected
