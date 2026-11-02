"""Who runs the broker calls.

``submit``, ``cancel`` and ``modify`` only append to the adapter's outbox and
call ``executor.kick()``. The executor decides where the outbox is drained.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol


class Executor(Protocol):
    def bind(self, pump: Callable[[], int]) -> None:
        """Called once with the adapter's ``pump``."""

    def kick(self) -> None:
        """There is work in the outbox. Must not block on I/O."""

    def close(self) -> None: ...


class InlineExecutor:
    """Drain on the caller's thread, before ``kick`` returns. For tests and scripts."""

    def __init__(self) -> None:
        self._pump: Callable[[], int] = lambda: 0

    def bind(self, pump: Callable[[], int]) -> None:
        self._pump = pump

    def kick(self) -> None:
        self._pump()

    def close(self) -> None:
        pass


class ManualExecutor:
    """Never drains by itself: the host calls ``adapter.pump()`` when it chooses."""

    def bind(self, pump: Callable[[], int]) -> None:
        pass

    def kick(self) -> None:
        pass

    def close(self) -> None:
        pass


class ThreadExecutor:
    """One daemon worker thread drains the outbox in order.

    One thread keeps the order of a venue's requests (a cancel never overtakes
    its submit). Rate-limit waits and retry sleeps happen on it, never on the
    node's thread.
    """

    def __init__(self, name: str = "raptorbt-adapter") -> None:
        self._name = name
        self._wake = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._pump: Callable[[], int] = lambda: 0

    def bind(self, pump: Callable[[], int]) -> None:
        self._pump = pump
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            self._pump()

    def kick(self) -> None:
        self._wake.set()

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
