"""What the timestamp of an input bar means, resolved once per run.

The engine's single internal convention is **visible time = bar close time**:
event ordering, ``DAY`` order expiry, session square-off, the strategy clock
and ``ctx.datetime`` all read it. A caller's bars may be stamped either way,
so each bar-taking strategy entry point calls :meth:`BarStamps.resolve` and
shifts the stamps it hands to the engine:

- ``bar_stamp="close"``: the stamp is already the close time.
- ``bar_stamp="open"``: close time is ``stamp + bar_ns``.
- ``bar_stamp=None``: the legacy path. Nothing is shifted, composite bars are
  built as if input stamps were open stamps and the feed merge as if they were
  close stamps. A ``DeprecationWarning`` says so.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from raptorbt.errors import InvalidParameter

BAR_STAMPS = ("open", "close")


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

    def __init__(self, bar_stamp: str | None, offsets: dict, call: str):
        self.bar_stamp = bar_stamp
        self._offsets = offsets
        self._call = call
        self._warned = False

    @classmethod
    def resolve(
        cls,
        bar_stamp,
        bar_ns,
        symbols,
        call: str,
        warn: bool = True,
        stacklevel: int = 4,
    ) -> "BarStamps":
        """Validate ``bar_stamp`` / ``bar_ns`` for ``symbols`` and warn if unset.

        ``bar_ns`` is read only when ``bar_stamp == "open"``.
        """
        if bar_stamp is not None and bar_stamp not in BAR_STAMPS:
            raise InvalidParameter(
                f"{call}: bar_stamp must be one of {', '.join(map(repr, BAR_STAMPS))} "
                f"or None, got {bar_stamp!r}",
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
        stamps = cls(bar_stamp, offsets, call)
        if warn:
            stamps.warn_missing(stacklevel=stacklevel)
        return stamps

    def warn_missing(self, call: str | None = None, stacklevel: int = 3) -> None:
        """Emit the deprecation warning once, if no ``bar_stamp`` was given."""
        if self.bar_stamp is not None or self._warned:
            return
        self._warned = True
        warnings.warn(
            f"{call or self._call}: bar_stamp not given; assuming today's behaviour "
            "(composite bars built from open-stamped input, event ordering "
            "assuming close-stamped input). bar_stamp will be required in 1.0; "
            "pass bar_stamp='open' with bar_ns=<bar width in ns>, or 'close'.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )

    @property
    def aggregator_stamp(self) -> str:
        """The ``stamp`` composite aggregators are built with.

        Input is normalised to close time whenever ``bar_stamp`` is given.
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
