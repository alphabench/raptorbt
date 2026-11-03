"""Input checks shared by the strategy runners."""

from __future__ import annotations

import datetime
import math

import numpy as np

from raptorbt.errors import InvalidData


def ensure_prices_finite(label: str, operation: str, arrays: dict) -> None:
    """Refuse a NaN or infinite open, high, low or close.

    A non-finite price compares false against everything, so the run would
    complete and leave a NaN in the equity curve. The error names the column
    and the first offending index.
    """
    for column in ("open", "high", "low", "close"):
        bad = np.flatnonzero(~np.isfinite(arrays[column]))
        if bad.size:
            index = int(bad[0])
            raise InvalidData(
                f"{label}: {column} is not finite ({float(arrays[column][index])}) "
                f"at index {index}",
                code="non_finite_price",
                operation=operation,
                parameter=column,
                value=index,
            )


def ensure_pushed_finite(label: str, operation: str, **prices: float) -> None:
    """Refuse a NaN or infinite price pushed to a live node, naming it.

    A non-finite price would be ignored by some comparisons and poison
    others, so the push is refused and changes nothing.
    """
    for name, value in prices.items():
        if not math.isfinite(value):
            raise InvalidData(
                f"{label}: {name} is not finite ({value}) in a pushed event",
                code="non_finite_price",
                operation=operation,
                parameter=name,
                value=repr(value),
            )


_DAY_NS = 86_400 * 10**9
_EPOCH_ORDINAL = 719_163  # datetime.date.fromordinal of 1970-01-01


def _yyyymmdd(day_number: int) -> int:
    d = datetime.date.fromordinal(_EPOCH_ORDINAL + day_number)
    return d.year * 10_000 + d.month * 100 + d.day


class SessionCoverage:
    """Refuse a bar on a date the instrument's session schedule does not cover.

    A schedule loaded from calendar documents, and the MCX preset, answer only for the
    dates they were given. Beyond them "the market is closed" would be a guess (a holiday and
    "no calendar" look alike), so a bar there raises ``session_not_covered`` naming the date
    and the dates covered.
    """

    def __init__(self, symbol: str, schedule, operation: str):
        self.symbol, self.operation = symbol, operation
        self.ranges = list(schedule.coverage)
        self.tz = schedule.tz_offset_ns
        self._seen: set[int] = set()

    @classmethod
    def of(cls, symbol: str, spec, operation: str):
        schedule = getattr(spec, "session", None) if spec is not None else None
        if schedule is None or not schedule.coverage:
            return None
        return cls(symbol, schedule, operation)

    def check(self, timestamp: int) -> None:
        day = (int(timestamp) + self.tz) // _DAY_NS
        if day in self._seen:
            return
        date = _yyyymmdd(day)
        if not any(a <= date <= b for a, b in self.ranges):
            raise InvalidData(
                f"{self.symbol}: {date} is outside the dates its session schedule covers "
                f"({', '.join(f'{a}..{b}' for a, b in self.ranges)}); a market hour for it "
                "would be a guess: load a calendar document for that period "
                "(SessionSchedule.from_calendars)",
                code="session_not_covered",
                operation=self.operation,
                parameter="session",
                value=date,
                valid=[f"{a}..{b}" for a, b in self.ranges],
            )
        self._seen.add(day)

    def check_all(self, timestamps) -> None:
        for day in np.unique((np.asarray(timestamps, dtype=np.int64) + self.tz) // _DAY_NS):
            self.check(int(day) * _DAY_NS - self.tz)


def ensure_sessions_cover(arrays: dict, instruments, operation: str) -> None:
    """Check every symbol's bar dates against its instrument's schedule coverage."""
    for symbol, bars in arrays.items():
        coverage = SessionCoverage.of(symbol, (instruments or {}).get(symbol), operation)
        if coverage is not None:
            coverage.check_all(bars["timestamps"])
