"""Rate limiting and backoff, with injectable clocks so tests run instantly."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping


class TokenBucket:
    """``rate`` tokens a second up to ``burst``. Take one per request."""

    def __init__(self, rate: float, burst: float | None = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.burst = float(burst if burst is not None else max(1.0, rate))
        self._clock = clock
        self._tokens = self.burst
        self._at = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._at) * self.rate)
        self._at = now

    def wait_needed(self, n: float = 1.0) -> float:
        self._refill()
        return 0.0 if self._tokens >= n else (n - self._tokens) / self.rate

    def take(self, n: float = 1.0) -> None:
        self._refill()
        self._tokens -= n


class RateLimiter:
    """Endpoint classes, each with one or more windows (per second, per minute...).

    A request waits until every window of its class has a token. The wait is in
    the caller's thread, which for an adapter is the executor, never the node.
    """

    def __init__(
        self,
        classes: Mapping[str, Iterable[TokenBucket]],
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._classes = {k: list(v) for k, v in classes.items()}
        self._sleep = sleep
        self._lock = threading.Lock()

    @classmethod
    def per_window(
        cls,
        spec: Mapping[str, Iterable[tuple[int, float]]],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> RateLimiter:
        """``{"orders": [(10, 1), (400, 60)]}``: 10 a second and 400 a minute."""
        return cls(
            {k: [TokenBucket(n / w, n, clock=clock) for n, w in v] for k, v in spec.items()},
            sleep=sleep,
        )

    def acquire(self, klass: str) -> float:
        """Block until a request of ``klass`` may go. Returns the seconds waited."""
        buckets = self._classes.get(klass)
        if not buckets:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                wait = max(b.wait_needed() for b in buckets)
                if wait <= 0:
                    for b in buckets:
                        b.take()
                    return waited
            self._sleep(wait)
            waited += wait


class Backoff:
    """Exponential delays: ``base * factor**attempt`` capped at ``cap``.

    No jitter by default so a run is reproducible; pass ``jitter`` (a function
    of the delay) to spread a fleet of reconnecting clients.
    """

    def __init__(
        self,
        base: float = 0.2,
        factor: float = 2.0,
        cap: float = 30.0,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        self.base, self.factor, self.cap, self.jitter = base, factor, cap, jitter

    def delay(self, attempt: int) -> float:
        d = min(self.cap, self.base * self.factor**attempt)
        return self.jitter(d) if self.jitter else d


def reconnect_loop(
    connect: Callable[[], None],
    *,
    backoff: Backoff | None = None,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
    on_connected: Callable[[], None] | None = None,
    on_lost: Callable[[BaseException], None] | None = None,
    stable_after: float = 10.0,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Run ``connect`` (which blocks while the connection lives) until stopped.

    After every loss, wait the backoff and connect again; the backoff starts over
    once a connection has lasted ``stable_after`` seconds. Wire ``on_lost`` to
    ``adapter.connection_lost`` and ``on_connected`` to
    ``adapter.connection_restored``, so the adapter reads the venue's order book
    after any gap.
    """
    backoff = backoff or Backoff()
    attempt = 0
    started = clock()
    while not should_stop():
        try:
            if on_connected:
                on_connected()
            started = clock()
            connect()
            if clock() - started >= stable_after:
                attempt = 0
        except Exception as exc:  # noqa: BLE001 - any loss is retried
            if clock() - started >= stable_after:
                attempt = 0
            if on_lost:
                on_lost(exc)
        if should_stop():
            return
        sleep(backoff.delay(attempt))
        attempt += 1
