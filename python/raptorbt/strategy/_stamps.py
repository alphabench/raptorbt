"""What the timestamp of an input bar means, resolved once per run.

The engine's single internal convention is **visible time = bar close time**:
event ordering, ``DAY`` order expiry, session square-off, the strategy clock
and ``ctx.datetime`` all read it. A caller's bars may be stamped either way,
so each bar-taking strategy entry point calls :meth:`BarStamps.resolve` and
shifts the stamps it hands to the engine:

- ``bar_stamp="close"``: the stamp is already the close time.
- ``bar_stamp="open"``: close time is ``stamp + bar_ns``.

``bar_stamp`` is required wherever bars are given: leaving it out raises
``missing_bar_stamp``. The one run that takes no bars (a tick-only run or node)
has nothing to stamp and resolves with ``bar_stamp=None``; it fails the moment a
bar is pushed.
"""

from __future__ import annotations

import math

import numpy as np

from raptorbt.errors import InvalidParameter

BAR_STAMPS = ("open", "close")


def _missing(call: str) -> InvalidParameter:
    return InvalidParameter(
        f"{call}: bar_stamp is required: say whether the bar timestamps are bar opens "
        "('open', with bar_ns, the bar width in nanoseconds) or bar closes ('close')",
        code="missing_bar_stamp",
        operation=call,
        parameter="bar_stamp",
        value=None,
    )


def _bad_width(message: str, where: str, value) -> InvalidParameter:
    return InvalidParameter(
        message,
        code="invalid_bar_ns",
        operation=where.split(":")[0],
        parameter=where.split(": ", 1)[-1],
        value=value,
    )


def _width(value, where: str) -> int:
    if isinstance(value, bool):
        raise _bad_width(
            f"{where} must be a positive number of nanoseconds, got {value!r}", where, value
        )
    try:
        ns = float(value)
    except (TypeError, ValueError):
        raise _bad_width(
            f"{where} must be a positive number of nanoseconds, got {value!r}", where, value
        ) from None
    if not math.isfinite(ns) or ns <= 0 or ns != int(ns):
        raise _bad_width(
            f"{where} must be a positive whole number of nanoseconds, got {value!r}", where, value
        )
    return int(ns)


class BarStamps:
    """The resolved stamp contract of one run."""

    def __init__(self, bar_stamp: str | None, offsets: dict, call: str, widths: dict | None = None):
        self.bar_stamp = bar_stamp
        self._offsets = offsets
        self._call = call
        self._widths = widths or {}

    @classmethod
    def resolve(
        cls,
        bar_stamp,
        bar_ns,
        symbols,
        call: str,
        require: bool = True,
    ) -> "BarStamps":
        """Validate ``bar_stamp`` / ``bar_ns`` for ``symbols``.

        ``bar_stamp`` is required when ``require`` (the call takes bars); a call that
        may take none passes ``require=False`` and :meth:`require_given` fires at the
        first bar. ``bar_ns`` moves the stamps when ``bar_stamp == "open"``; with
        ``"close"`` it is the width of the bars, used where the engine needs it (an
        expression on a coarser clock) in place of the most common gap of the bars.
        """
        if bar_stamp is None and require:
            raise _missing(call)
        if bar_stamp is not None and bar_stamp not in BAR_STAMPS:
            raise InvalidParameter(
                f"{call}: bar_stamp must be one of {', '.join(map(repr, BAR_STAMPS))}, "
                f"got {bar_stamp!r}",
                code="invalid_bar_stamp",
                operation=call,
                parameter="bar_stamp",
                value=bar_stamp,
            )
        offsets = {symbol: 0 for symbol in symbols}
        if bar_stamp == "open":
            if bar_ns is None:
                raise InvalidParameter(
                    f"{call}: bar_stamp='open' needs bar_ns, the bar width in "
                    "nanoseconds (a number, or a dict symbol -> number)",
                    code="missing_bar_ns",
                    operation=call,
                    parameter="bar_ns",
                    value=None,
                )
            if isinstance(bar_ns, dict):
                missing = [s for s in symbols if s not in bar_ns]
                if missing:
                    raise InvalidParameter(
                        f"{call}: bar_ns has no entry for "
                        + ", ".join(repr(s) for s in missing),
                        code="missing_bar_ns",
                        operation=call,
                        parameter="bar_ns",
                        value=missing,
                        valid=list(symbols),
                    )
                for symbol in symbols:
                    offsets[symbol] = _width(bar_ns[symbol], f"{call}: bar_ns[{symbol!r}]")
            else:
                width = _width(bar_ns, f"{call}: bar_ns")
                offsets = {symbol: width for symbol in symbols}
        widths: dict = {}
        if bar_stamp == "close" and bar_ns is not None:
            for symbol in symbols:
                given = bar_ns.get(symbol) if isinstance(bar_ns, dict) else bar_ns
                if given is not None:
                    widths[symbol] = _width(given, f"{call}: bar_ns")
        return cls(bar_stamp, offsets, call, widths)

    def width(self, symbol: str | None) -> int:
        """The width of a close-stamped symbol's bars the caller gave (``bar_ns``), else ``0``."""
        return self._widths.get(symbol, 0)

    def require_given(self, call: str | None = None) -> None:
        """Raise ``missing_bar_stamp`` if a bar arrives and no ``bar_stamp`` was given."""
        if self.bar_stamp is None:
            raise _missing(call or self._call)

    @property
    def aggregator_stamp(self) -> str:
        """The ``stamp`` composite aggregators are built with.

        Input is normalised to close time whenever ``bar_stamp`` is given. ``None``
        only occurs on a run that took no bars, and keeps the aggregator's default.
        """
        return "open" if self.bar_stamp is None else "close"

    def offset(self, symbol: str | None) -> int:
        """Nanoseconds from a symbol's stamp to its bar's close (0 if none)."""
        return self._offsets.get(symbol, 0)

    def visible(self, symbol: str | None, timestamps: np.ndarray) -> np.ndarray:
        """Bar close times for a symbol's stamps."""
        offset = self.offset(symbol)
        return timestamps + offset if offset else timestamps

    @property
    def offsets(self) -> dict:
        return dict(self._offsets)

    def offset_for_new(self, symbol: str, bar_ns, call: str) -> int:
        """The offset a symbol added to a running node would get: the node's single bar width
        under ``bar_stamp="open"`` unless ``bar_ns`` gives this symbol's own, else ``0``."""
        if self.bar_stamp != "open":
            return 0
        if bar_ns is not None:
            return _width(bar_ns, f"{call}: bar_ns")
        widths = set(self._offsets.values())
        if len(widths) != 1:
            raise InvalidParameter(
                f"{call}: bar_stamp='open' with different bar widths per symbol needs bar_ns "
                f"for {symbol!r}, the bar width in nanoseconds",
                code="missing_bar_ns",
                operation=call,
                parameter="bar_ns",
                value=None,
            )
        return widths.pop()

    def add(self, symbol: str, offset: int) -> None:
        self._offsets[symbol] = offset
