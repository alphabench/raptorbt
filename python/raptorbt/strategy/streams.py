"""Composite-bar aggregation and indicator fan-out, shared by both runners.

A strategy declares timeframes with ``subscribe_bars`` and indicators with
``register_indicator``, both from ``on_start``. This module owns what those
declarations become at run time: one :class:`BarAggregator` per subscription
(per symbol, in portfolio runs) and an index from stream to the indicators
listening on it.

Ordering, in both runners: a completed composite bar dispatches *before* the
primary ``on_bar`` of the bar that completed it, because the composite closed
strictly earlier. In portfolio runs that guarantee is per symbol — a
symbol's composite bars are built only from its own bars, and cross-symbol
order follows the merged event schedule.
"""

from __future__ import annotations

from raptorbt._raptorbt import BarAggregator
from raptorbt.errors import InvalidParameter
from raptorbt.strategy.context import CompositeBar


def sessions_of(instruments: dict | None) -> dict:
    """``{symbol: SessionSchedule}`` for the instruments that carry one."""
    return {
        symbol: spec.session
        for symbol, spec in (instruments or {}).items()
        if spec is not None and spec.session is not None
    }


def enumerate_subscriptions(subscriptions):
    """Yield ``(stream_id, (step, unit, brick_size))`` for each subscription.

    Subscriptions are stored as 2- or 3-tuples depending on whether a brick
    size was given.
    """
    for stream_id, sub in enumerate(subscriptions):
        step, unit = sub[0], sub[1]
        brick = sub[2] if len(sub) > 2 else 0.0
        yield stream_id, (step, unit, brick)


class StreamState:
    """Per-symbol aggregators and indicator routing for one run.

    ``symbols`` is the list of symbols in a portfolio run, or ``None`` for a
    single-instrument run (where registrations carry no symbol and every
    indicator sees the one stream). ``tz_offset_ns`` is the session's UTC
    offset: daily and longer composite bars cut at its midnight. ``stamp`` is
    what the timestamps of pushed *bars* mean (see ``BarAggregator``); trades
    are instants and ignore it. ``sessions`` maps a symbol (``None`` in a
    single-instrument run) to its ``SessionSchedule``: that symbol's daily and
    longer bars then follow the schedule's trading dates and skip holidays.
    """

    def __init__(
        self,
        strategy,
        symbols: list[str] | None = None,
        tz_offset_ns: int = 0,
        stamp: str = "open",
        sessions: dict | None = None,
    ):
        subscriptions = list(strategy._bar_subscriptions)
        keys: list[str | None] = list(symbols) if symbols else [None]

        # (symbol, stream_id) -> aggregator. One per symbol per subscription,
        # so a symbol's composite bars are built from its bars alone.
        self._aggregators: dict[
            str | None, list[tuple[int, int, str, BarAggregator]]
        ] = {
            key: [
                (stream_id, step, unit, BarAggregator(
                        step, unit, tz_offset_ns=tz_offset_ns, brick_size=brick,
                        stamp=stamp, session=(sessions or {}).get(key),
                    ))
                for stream_id, (step, unit, brick) in enumerate_subscriptions(
                    subscriptions
                )
            ]
            for key in keys
        }

        # Indicator routing, indexed once rather than scanned per bar.
        self._primary: dict[str | None, list] = {key: [] for key in keys}
        self._composite: dict[tuple[str | None, int], list] = {}
        for indicator, stream_id, symbol in strategy._indicators:
            # An unrouted registration listens on every symbol; see
            # `Strategy.register_indicator` for why that is rarely wanted.
            if (
                symbols is not None
                and symbol is not None
                and symbol not in self._primary
            ):
                known = ", ".join(str(k) for k in keys)
                raise InvalidParameter(
                    f"register_indicator(symbol={symbol!r}) names a symbol that is "
                    f"not in this run; known symbols: {known}",
                    code="unknown_symbol", operation="register_indicator", parameter="symbol",
                    value=symbol, valid=[str(k) for k in keys],
                )
            # A run without declared symbols has one stream, keyed None: the
            # symbol an indicator names (already checked by `ctx.indicators`
            # against the run's instrument) is that stream.
            targets = [symbol] if symbol is not None and symbols is not None else keys
            for key in targets:
                if stream_id is None:
                    self._primary[key].append(indicator)
                else:
                    self._composite.setdefault((key, stream_id), []).append(indicator)

    def _dispatch(
        self, strategy, ctx, aggregator, completed, stream_id, step, unit, symbol
    ) -> None:
        """Dispatch a completed bar and everything queued behind it.

        Renko completes several bricks from one record; ``push`` returns
        only the first, so the rest must be drained or they are lost.
        """
        while completed is not None:
            bar = CompositeBar(stream_id, step, unit, *completed, symbol=symbol)
            for indicator in self._composite.get((symbol, stream_id), ()):
                indicator.update_bar(bar.open, bar.high, bar.low, bar.close)
            strategy.on_composite_bar(ctx, bar)
            completed = aggregator.next_pending()

    def push(self, strategy, ctx, ts, o, h, l, c, v, symbol: str | None = None) -> None:
        """Feed one primary bar: aggregate, dispatch, update indicators.

        Call after the clock advance and before ``on_bar``, so composite
        bars and indicator values are current when handlers see the bar.
        """
        for stream_id, step, unit, aggregator in self._aggregators[symbol]:
            completed = aggregator.push_bar(ts, o, h, l, c, v)
            self._dispatch(
                strategy, ctx, aggregator, completed, stream_id, step, unit, symbol
            )

        # Primary-registered indicators update before on_bar sees the bar.
        for indicator in self._primary[symbol]:
            indicator.update_bar(o, h, l, c)
