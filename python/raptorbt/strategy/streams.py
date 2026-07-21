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
from raptorbt.strategy.context import CompositeBar


class StreamState:
    """Per-symbol aggregators and indicator routing for one run.

    ``symbols`` is the list of symbols in a portfolio run, or ``None`` for a
    single-instrument run (where registrations carry no symbol and every
    indicator sees the one stream).
    """

    def __init__(self, strategy, symbols: list[str] | None = None):
        subscriptions = list(strategy._bar_subscriptions)
        keys: list[str | None] = list(symbols) if symbols else [None]

        # (symbol, stream_id) -> aggregator. One per symbol per subscription,
        # so a symbol's composite bars are built from its bars alone.
        self._aggregators: dict[str | None, list[tuple[int, int, str, BarAggregator]]] = {
            key: [
                (stream_id, step, unit, BarAggregator(step, unit))
                for stream_id, (step, unit) in enumerate(subscriptions)
            ]
            for key in keys
        }

        # Indicator routing, indexed once rather than scanned per bar.
        self._primary: dict[str | None, list] = {key: [] for key in keys}
        self._composite: dict[tuple[str | None, int], list] = {}
        self._unrouted = False
        for indicator, stream_id, symbol in strategy._indicators:
            if symbol is None:
                self._unrouted = True
            # An unrouted registration listens on every symbol; see
            # `Strategy.register_indicator` for why that is rarely wanted.
            if symbols is not None and symbol is not None and symbol not in self._primary:
                known = ", ".join(str(k) for k in keys)
                raise ValueError(
                    f"register_indicator(symbol={symbol!r}) names a symbol that is "
                    f"not in this run; known symbols: {known}"
                )
            targets = [symbol] if symbol is not None else keys
            for key in targets:
                if stream_id is None:
                    self._primary[key].append(indicator)
                else:
                    self._composite.setdefault((key, stream_id), []).append(indicator)

    @property
    def has_unrouted_indicators(self) -> bool:
        """Whether any indicator was registered without a symbol."""
        return self._unrouted

    def push(self, strategy, ctx, ts, o, h, l, c, v, symbol: str | None = None) -> None:
        """Feed one primary bar: aggregate, dispatch, update indicators.

        Call after the clock advance and before ``on_bar``, so composite
        bars and indicator values are current when handlers see the bar.
        """
        for stream_id, step, unit, aggregator in self._aggregators[symbol]:
            completed = aggregator.push_bar(ts, o, h, l, c, v)
            if completed is not None:
                bar = CompositeBar(stream_id, step, unit, *completed, symbol=symbol)
                for indicator in self._composite.get((symbol, stream_id), ()):
                    indicator.update_bar(bar.open, bar.high, bar.low, bar.close)
                strategy.on_composite_bar(ctx, bar)

        # Primary-registered indicators update before on_bar sees the bar.
        for indicator in self._primary[symbol]:
            indicator.update_bar(o, h, l, c)
