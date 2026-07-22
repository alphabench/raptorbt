"""Live tick stream for class-based strategies.

The batch runners replay a finished dataset; this drives the same session,
kernels and strategy hooks from events pushed one at a time — an open-ended
feed whose end is not known in advance. Push a tick, and every hook it
triggers fires before ``push_tick`` returns, so the caller reads positions
and queued state synchronously.

Semantics are identical to :func:`raptorbt.run_tick_strategy` — both drive
the same dispatch loop — with one addition: pushed or warmup *bars* execute
(they match orders and mark equity), exactly as in the bar runner. Bars
aggregated from prints via ``primary_bars`` remain a view only.

Typical shape::

    stream = TickStrategyStream(
        MyStrategy(), ["NSE:TCS"], warmup_bars={"NSE:TCS": bars},
        primary_bars=(1, "m"),
    )
    for tick in feed:                      # days-long, open-ended
        stream.push_tick("NSE:TCS", tick.ts_ns, tick.ltp, tick.bid, tick.ask)
    result = stream.finish()               # once the session truly ends
"""

from __future__ import annotations

import numpy as np

from raptorbt._raptorbt import (
    PyBacktestConfig,
    PyInstrumentConfig,
    PyPortfolioResult,
    PyPortfolioSession,
)
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.portfolio_runner import apply_commands_on
from raptorbt.strategy.tick_runner import (
    TickContext,
    drive_tick_events,
    setup_tick_strategy,
)


class TickStrategyStream:
    """One strategy over N instruments, fed events as they arrive.

    ``warmup_bars`` maps symbol -> dict of OHLCV arrays replayed through the
    strategy during construction, so indicators are primed before the first
    live push. Warmup bars execute; hand the stream a strategy that stays
    passive on history if that is not wanted.

    Everything else — account type, leverage, risk limits, OMS type —
    behaves as in :func:`raptorbt.run_tick_strategy`.
    """

    def __init__(
        self,
        strategy: Strategy | type[Strategy],
        symbols: list[str],
        config: PyBacktestConfig | None = None,
        primary_bars: tuple[int, str] | None = None,
        warmup_bars: dict[str, dict] | None = None,
        directions: dict[str, int] | None = None,
        instruments: dict | None = None,
        instrument_configs: dict[str, PyInstrumentConfig] | None = None,
        oms_type: str = "netting",
        account_type: str = "cash",
        leverage: float = 1.0,
    ):
        if isinstance(strategy, type):
            strategy = strategy()
        if not isinstance(strategy, Strategy):
            raise ValueError(
                f"strategy must be a Strategy instance or subclass, got {type(strategy).__name__}"
            )
        if not symbols:
            raise ValueError("symbols must name at least one instrument")

        self._strategy = strategy
        self._symbols = list(symbols)
        self._index_of = {s: i for i, s in enumerate(self._symbols)}
        self._finished = False

        session = PyPortfolioSession(
            config=config, account_type=account_type, leverage=leverage
        )
        for symbol in self._symbols:
            session.add_instrument(
                symbol,
                direction=(directions or {}).get(symbol, 1),
                instrument_config=(instrument_configs or {}).get(symbol),
                instrument=(instruments or {}).get(symbol),
                oms_type=oms_type,
            )
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for symbol, bars in (warmup_bars or {}).items():
            i = self._index_of[symbol]
            a = {
                "timestamps": np.ascontiguousarray(bars["timestamps"], dtype=np.int64),
            }
            for key in ("open", "high", "low", "close", "volume"):
                a[key] = np.ascontiguousarray(bars[key], dtype=np.float64)
            session.set_bars(
                i, a["timestamps"], a["open"], a["high"], a["low"], a["close"], a["volume"]
            )
            arrays[symbol] = a
        session.seal()

        self._session = session
        self.ctx = TickContext(session, self._symbols, arrays)
        self._clocks, self._streams, self._primary = setup_tick_strategy(
            strategy, self.ctx, self._symbols, primary_bars
        )
        self._id_map: dict[str, tuple[int, int]] = {}
        self._apply_commands = apply_commands_on(
            strategy, session, self.ctx, self._symbols, self._id_map
        )
        # Prime indicators and strategy state from the warmup history now,
        # so the stream is warm before the first live event.
        self._drain()

    # -- feeding -------------------------------------------------------------

    def push_tick(
        self,
        symbol: str,
        timestamp: int,
        ltp: float,
        bid: float = 0.0,
        ask: float = 0.0,
        buy_qty_delta: float = 0.0,
        sell_qty_delta: float = 0.0,
    ) -> int:
        """Feed one tick row; fires every hook it triggers before returning.

        ``ltp > 0`` prints a trade; a two-sided book yields a quote after
        it. Returns how many events were appended (0–2).
        """
        self._check_open()
        appended = self._session.push_tick(
            self._index_of[symbol], timestamp, ltp, bid, ask, buy_qty_delta, sell_qty_delta
        )
        if appended:
            self._drain()
        return appended

    def push_bar(
        self,
        symbol: str,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Feed one closed bar. It executes: orders match, equity marks."""
        self._check_open()
        self._session.push_bar(
            self._index_of[symbol], timestamp, open, high, low, close, volume
        )
        self._drain()

    def push_depth(
        self,
        symbol: str,
        timestamp: int,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Feed one depth snapshot: ``(price, size)`` lists, best first."""
        self._check_open()
        self._session.push_depth(self._index_of[symbol], timestamp, bids, asks)
        self._drain()

    # -- state ---------------------------------------------------------------

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def equity(self) -> float:
        return self._session.equity()

    @property
    def is_halted(self) -> bool:
        """Whether a margin call or drawdown kill-switch has latched."""
        return self._session.is_halted()

    def positions(self, symbol: str):
        return self._session.positions(self._index_of[symbol])

    def finish(self) -> PyPortfolioResult:
        """Close out and compute metrics; the stream is unusable after."""
        self._check_open()
        self._finished = True
        self._strategy.on_stop(self.ctx)
        return self._session.finish()

    # -- internals -----------------------------------------------------------

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError("stream is finished; create a new one")

    def _drain(self) -> None:
        drive_tick_events(
            self._strategy,
            self.ctx,
            self._session,
            self._symbols,
            self._clocks,
            self._streams,
            self._primary,
            self._apply_commands,
        )
