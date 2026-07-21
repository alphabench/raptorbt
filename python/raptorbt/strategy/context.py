"""Per-run context handed to strategy hooks."""

from __future__ import annotations

import datetime as _dt
from typing import NamedTuple

import numpy as np


class Bar(NamedTuple):
    """The bar currently being processed."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class StrategyContext:
    """Read/act surface a strategy sees inside its hooks.

    Data access contract: the full OHLCV arrays are exposed for indicator
    precomputation in ``on_start``, but decision logic inside ``on_bar`` must
    only read values at ``idx`` or earlier. Indexing past ``idx`` reads the
    future and invalidates the backtest.
    """

    def __init__(
        self,
        session,
        timestamps: np.ndarray,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> None:
        self._session = session
        self.timestamps = timestamps
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        #: Index of the bar currently being processed.
        self.idx: int = 0

    # -- current bar -------------------------------------------------------

    @property
    def bar(self) -> Bar:
        """The bar currently being processed."""
        i = self.idx
        return Bar(
            timestamp=int(self.timestamps[i]),
            open=float(self.open[i]),
            high=float(self.high[i]),
            low=float(self.low[i]),
            close=float(self.close[i]),
            volume=float(self.volume[i]),
        )

    @property
    def timestamp(self) -> int:
        """Epoch timestamp of the current bar (engine time units)."""
        return int(self.timestamps[self.idx])

    @property
    def datetime(self) -> _dt.datetime:
        """Current bar timestamp as a naive UTC datetime.

        Assumes nanosecond epoch timestamps, the convention used by the
        array-based runners.
        """
        return _dt.datetime.utcfromtimestamp(self.timestamps[self.idx] / 1e9)

    def history(self, n: int) -> np.ndarray:
        """Trailing window of up to ``n`` closes ending at the current bar."""
        start = max(0, self.idx + 1 - n)
        return self.close[start : self.idx + 1]

    @property
    def n_bars(self) -> int:
        """Total number of bars in the run."""
        return len(self.close)

    # -- portfolio state ---------------------------------------------------

    @property
    def position(self):
        """Open position snapshot, or ``None`` when flat."""
        return self._session.position()

    @property
    def positions(self):
        """All open positions, in opening order (hedging holds several)."""
        return self._session.positions()

    @property
    def free_capital(self) -> float:
        """Cash not locked as margin (margin accounts); all cash otherwise."""
        return self._session.free_capital()

    @property
    def net_position(self) -> float:
        """Signed unit total across open positions (hedging nets out)."""
        return sum(p.size * p.direction for p in self._session.positions())

    @property
    def is_net_long(self) -> bool:
        return self.net_position > 0.0

    @property
    def is_net_short(self) -> bool:
        return self.net_position < 0.0

    @property
    def is_flat(self) -> bool:
        return not self._session.positions()

    @property
    def equity(self) -> float:
        """Mark-to-market equity after the most recent completed bar."""
        return self._session.equity()

    @property
    def cash(self) -> float:
        """Uninvested cash."""
        return self._session.cash()

    # -- position management -----------------------------------------------

    def set_stop_price(self, price: float | None) -> None:
        """Overwrite the open position's stop; ``None`` removes it.

        No-op when flat. Called from ``on_bar``, the new stop is already
        checked against the current bar's range.
        """
        self._session.set_stop_price(price)

    def set_target_price(self, price: float | None) -> None:
        """Overwrite the open position's target; ``None`` removes it.

        No-op when flat. Called from ``on_bar``, the new target is already
        checked against the current bar's range.
        """
        self._session.set_target_price(price)


class CompositeBar(NamedTuple):
    """A completed higher-timeframe bar built from the primary stream.

    ``timestamp`` is the window-end for time aggregations — the bar contains
    only data strictly before it, never the bar that completed it.
    ``stream_id`` is the handle returned by ``Strategy.subscribe_bars``.
    ``symbol`` names the instrument the bar was built from in portfolio
    runs, and is ``None`` in single-instrument ones.
    """

    stream_id: int
    step: int
    unit: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str | None = None


class TradeTick(NamedTuple):
    """One trade print.

    ``symbol`` names the instrument in portfolio/tick runs.
    """

    timestamp: int
    price: float
    size: float
    symbol: str | None = None


class QuoteTick(NamedTuple):
    """One top-of-book quote.

    Quotes are observation only: they do not fill orders, move trailing
    stops, or mark equity. The next trade print does all three.
    """

    timestamp: int
    bid: float
    ask: float
    symbol: str | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid
