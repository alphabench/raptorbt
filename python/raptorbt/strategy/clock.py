"""Bar-driven simulation clock: one-shot alerts and recurring timers.

In a multi-symbol run each symbol gets its own clock, so a timer set in
``on_start`` fires once per symbol rather than once for whichever symbol's
event happened to cross the threshold first. ``ctx.clock`` is the clock of
``ctx.symbol``; alerts and timers set from a handler therefore belong to
the symbol being handled.


Time events fire when a bar's timestamp reaches the scheduled time — at bar
granularity, which is why events carry both ``ts_scheduled`` and
``ts_fired`` (the bar timestamp that triggered them). Events dispatch to
``on_time_event`` *before* the bar's own data handlers: scheduled time
precedes the bar that revealed it had passed.
"""

from __future__ import annotations

from typing import NamedTuple


class TimeEvent(NamedTuple):
    """A fired alert or timer tick."""

    name: str
    ts_scheduled: int
    ts_fired: int


class Clock:
    """Alert/timer registry advanced by the runner once per bar."""

    def __init__(self) -> None:
        self._now: int = 0
        self._alerts: dict[str, int] = {}
        # name -> [next_fire, interval, stop_ns]
        self._timers: dict[str, list] = {}

    def timestamp_ns(self) -> int:
        """Timestamp of the most recent bar."""
        return self._now

    def set_time_alert(self, name: str, at_ns: int) -> None:
        """Fire once at the first bar whose timestamp reaches ``at_ns``."""
        self._alerts[name] = at_ns

    def set_timer(
        self,
        name: str,
        interval_ns: int,
        start_ns: int | None = None,
        stop_ns: int | None = None,
    ) -> None:
        """Fire repeatedly every ``interval_ns``, starting at ``start_ns``
        (default: one interval after the current bar). At most one firing
        per bar per timer; missed schedule points collapse into the next
        bar's single firing.
        """
        if interval_ns <= 0:
            raise ValueError("interval_ns must be > 0")
        first = start_ns if start_ns is not None else self._now + interval_ns
        self._timers[name] = [first, interval_ns, stop_ns]

    def cancel_timer(self, name: str) -> None:
        """Remove an alert or timer by name; unknown names are a no-op."""
        self._alerts.pop(name, None)
        self._timers.pop(name, None)

    def timer_names(self) -> list[str]:
        return [*self._alerts, *self._timers]

    def clone_schedule(self) -> "Clock":
        """A new clock carrying the same alerts and timers.

        Used to give each symbol its own copy of whatever was scheduled in
        ``on_start``, before any of them has advanced.
        """
        copy = Clock()
        copy._alerts = dict(self._alerts)
        copy._timers = {name: list(state) for name, state in self._timers.items()}
        return copy

    def _advance(self, ts: int) -> list[TimeEvent]:
        """Advance to a bar timestamp; return due events in scheduled order."""
        self._now = ts
        due: list[TimeEvent] = []

        for name, at_ns in list(self._alerts.items()):
            if ts >= at_ns:
                due.append(TimeEvent(name, at_ns, ts))
                del self._alerts[name]

        for name, state in list(self._timers.items()):
            next_fire, interval, stop_ns = state
            if stop_ns is not None and next_fire > stop_ns:
                del self._timers[name]
                continue
            if ts >= next_fire:
                due.append(TimeEvent(name, next_fire, ts))
                # Skip past every missed point so a data gap fires once.
                while state[0] <= ts:
                    state[0] += interval
                if stop_ns is not None and state[0] > stop_ns:
                    del self._timers[name]

        due.sort(key=lambda e: (e.ts_scheduled, e.name))
        return due
