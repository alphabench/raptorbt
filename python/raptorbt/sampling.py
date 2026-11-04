"""Bars from trades and coarser bars from finer ones, as ``TimeBars`` and ``Resample``.

Both are the engine's one time-window bar builder (the one behind ``BarAggregator``),
declared as a ``raptorbt.Resampler``; this module only names the two common declarations,
right-closed as a live feed reads them (a print exactly on a boundary closes the bar it
ends, a bar stamped at its close lands in the bucket that close ends).

``TimeBars(source, period, stream, ...)``
    trades into bars of ``period`` ns, on a grid at ``anchor`` (ns from the epoch, in the
    zone ``tz_offset_ns``), or restarted at each session open with ``anchor="session"``
    (feed ``on_session(True, ts)``).
``Resample(source, source_period, period, stream, ...)``
    bars of ``source_period`` ns, stamped at their close, into bars of ``period`` ns. The
    period must be a multiple of the source's and the anchor must sit on the source's grid.

``push_trade`` / ``push_bar`` / ``on_clock`` / ``on_session`` / ``flush`` return the
``SampledBar`` objects they closed (``ts`` is the close instant). See ``raptorbt.Resampler``
for every option, and ``raptorbt.resample_bars`` / ``resample_trades`` for whole arrays.
"""

from __future__ import annotations

from raptorbt._raptorbt import ResampledBar as SampledBar
from raptorbt._raptorbt import Resampler, resample_bars, resample_trades
from raptorbt.errors import InvalidParameter

__all__ = [
    "Resample",
    "Resampler",
    "SampledBar",
    "TimeBars",
    "resample_bars",
    "resample_trades",
]


def TimeBars(
    source: str,
    period: int,
    stream: str,
    *,
    anchor: int | str | None = None,
    tz_offset_ns: int = 0,
    close_on: str = "clock",
    gaps: str = "skip",
) -> Resampler:
    """Trades of ``source`` into bars of ``period`` ns named ``stream``."""
    if source == stream:
        raise InvalidParameter("a derived stream needs its own id", code="invalid_parameter",
                               operation="Resample", parameter="stream", value=stream,
                               valid=["an id other than the source's"])
    session = isinstance(anchor, str)
    if session and anchor != "session":
        raise InvalidParameter("anchor must be a number of ns or 'session'", code="invalid_parameter",
                               operation="Resample", parameter="anchor", value=anchor,
                               valid=["a number of ns", "session"])
    return Resampler(
        period,
        anchor_ns=0 if session or anchor is None else anchor,
        tz_offset_ns=tz_offset_ns,
        closed="right",
        gaps=gaps,
        close_on=close_on,
        session_anchored=session,
        stream=stream,
    )


def Resample(
    source: str,
    source_period: int,
    period: int,
    stream: str,
    *,
    anchor: int = 0,
    tz_offset_ns: int = 0,
    close_on: str = "clock",
    min_source: int = 0,
    gaps: str = "skip",
) -> Resampler:
    """Bars of ``source_period`` ns (stamped at their close) into bars of ``period`` ns."""
    if source == stream:
        raise InvalidParameter("a derived stream needs its own id", code="invalid_parameter",
                               operation="Resample", parameter="stream", value=stream,
                               valid=["an id other than the source's"])
    return Resampler(
        period,
        source_ns=source_period,
        anchor_ns=anchor,
        tz_offset_ns=tz_offset_ns,
        closed="right",
        gaps=gaps,
        close_on=close_on,
        min_source=min_source,
        stream=stream,
    )
