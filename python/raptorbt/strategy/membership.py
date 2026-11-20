"""A basket whose members change over the run: :class:`PointInTime`.

A :class:`~raptorbt.strategy.requirements.Universe` role is bound to a list of symbols. When the
list a run should have traded on one day is not the list of another (an index's constituents, a
segment's eligible stocks), bind the role to a :class:`PointInTime` instead: the union of every
symbol that was ever a member, with the members from each change on. The run feeds bars for the
union; ``ctx.role(name)`` answers the members **at the context's event time**, so a strategy that
ranks and trades only what it is handed never sees a name before it joined or after it left.

The manifest does not change: a strategy declares the same ``Universe`` role either way. The
binding is checked as a plain list of the union (the role's ``min`` and ``max`` bound the union),
and a point-in-time binding is a tuple of that union, so every host that takes a list takes it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from raptorbt.errors import InvalidParameter

__all__ = ["PointInTime"]


def _bad(message: str, parameter: str, value: object) -> InvalidParameter:
    return InvalidParameter(
        message, code="invalid_binding", operation="PointInTime", parameter=parameter, value=value
    )


class PointInTime(tuple):
    """The binding of a ``Universe`` role whose members change: a tuple of the union of the
    members, carrying ``steps``, ``(start_ns, members)`` pairs in ascending order.

    ``steps[i]`` holds from ``start_ns`` (ns since the epoch, an event time) until the next
    step's start; the last holds on. :meth:`at` answers the members at an instant: before the
    first step none, and with no instant (a context before its first event) the union.
    Every member of a step must be in ``symbols``, and a step lists a symbol once.

    >>> p = PointInTime(["A", "B", "C"], [(0, ["A", "B"]), (100, ["B", "C"])])
    >>> p.at(99), p.at(100), tuple(p)
    (('A', 'B'), ('B', 'C'), ('A', 'B', 'C'))
    """

    steps: tuple[tuple[int, tuple[str, ...]], ...]

    def __new__(
        cls, symbols: Iterable[str], steps: Iterable[tuple[int, Sequence[str]]]
    ) -> PointInTime:
        union = tuple(symbols)
        if isinstance(symbols, str) or not union:
            raise _bad("a point-in-time binding needs a list of symbols", "symbols", symbols)
        held = set(union)
        kept: list[tuple[int, tuple[str, ...]]] = []
        for step in steps:
            start, members = step
            if isinstance(start, bool) or not isinstance(start, int):
                raise _bad("a step starts at an integer instant (ns)", "steps", start)
            if kept and start <= kept[-1][0]:
                raise _bad("steps must start in ascending order", "steps", start)
            members = tuple(members)
            if len(set(members)) != len(members):
                raise _bad("a step lists a symbol once", "steps", list(members))
            outside = [m for m in members if m not in held]
            if outside:
                raise _bad(
                    "a step's members must be among the bound symbols", "steps", outside[:5]
                )
            kept.append((start, members))
        if not kept:
            raise _bad("a point-in-time binding needs at least one step", "steps", [])
        self = super().__new__(cls, union)
        self.steps = tuple(kept)
        return self

    def at(self, now_ns: int | None) -> tuple[str, ...]:
        """The members at ``now_ns``; the whole union when it is ``None``."""
        if now_ns is None:
            return tuple(self)
        lo, hi = 0, len(self.steps)
        while lo < hi:  # the last step starting at or before now_ns
            mid = (lo + hi) // 2
            if self.steps[mid][0] <= now_ns:
                lo = mid + 1
            else:
                hi = mid
        return self.steps[lo - 1][1] if lo else ()

    def __reduce__(self):
        return (PointInTime, (tuple(self), self.steps))
