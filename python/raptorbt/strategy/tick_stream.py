"""Live tick stream for class-based strategies.

The batch runners replay a finished dataset; this drives the same session,
kernels and strategy hooks from events pushed one at a time — an open-ended
feed whose end is not known in advance. Push a tick, and every hook it
triggers fires before ``push_tick`` returns, so the caller reads positions
and queued state synchronously.

Semantics are identical to :func:`raptorbt.run_tick_strategy` — both run on
the same Rust runtime; a stream is a one-strategy :class:`~raptorbt.LiveNode`
— with two additions. Pushed or warmup *bars* execute
(they match orders and mark equity), exactly as in the bar runner; bars
aggregated from prints via ``primary_bars`` remain a view only. And a print
the session refuses (``ltp == 0`` and no two-sided book — an index that has
not ticked, a quote-only row) still advances that symbol's clock: alerts
and timers due by its timestamp fire inside that push, instead of waiting
for the next accepted print. The batch runner never sees such rows, so it
cannot fire on them; feed it rows that carry a price.

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

from typing import Literal

from raptorbt._raptorbt import BacktestConfig, InstrumentConfig, PortfolioResult
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.live_node import LiveNode


class TickStrategyStream:
    """One strategy over N instruments, fed events as they arrive.

    ``warmup_bars`` maps symbol -> dict of OHLCV arrays replayed through the
    strategy during construction, so indicators are primed before the first
    live push. Warmup bars execute; hand the stream a strategy that stays
    passive on history if that is not wanted.

    ``initial_positions`` maps symbol -> ``{"quantity", "avg_price",
    "timestamp_ns"?}`` and ADOPTS a position the account already holds
    (broker-truth seeding): the strategy starts knowing it is long that many
    units at that average cost, with no order, no fill, no fees and no trade
    record. Adoption happens before warmup replay and before the first push,
    so the position appears in every snapshot from the start — a caller
    diffing ``positions()`` before/after pushes will never see it as a fresh
    entry. Cash or fully funded margin accounts (``leverage=1.0``), long-only;
    refused (never skipped) otherwise. Fully funded margin is supported
    because a strategy holding a short must run under a margin account for
    that short to transact at all, and such a book still needs seeding.

    ``bar_stamp`` / ``bar_ns`` say what the timestamps of warmup and pushed
    *bars* mean, as in :func:`raptorbt.run_portfolio_strategy`: ``"close"``,
    or ``"open"`` with ``bar_ns`` (the bar width in ns, or a ``{symbol: ns}``
    dict covering every symbol). Ticks are instants and ignore both. Leaving
    ``bar_stamp`` unset while supplying bars keeps the pre-1.0 behaviour and
    emits a ``DeprecationWarning``.

    Everything else — account type, leverage, risk limits, OMS type —
    behaves as in :func:`raptorbt.run_tick_strategy`.

    The stream wraps a one-strategy :class:`~raptorbt.LiveNode`, so
    ``ctx.series()`` returns read-only views of the rows still held (the last
    10,000 per symbol and kind, see ``LiveNode(history=...)``), cut at what
    has arrived. It no longer exposes the full arrays.
    """

    def __init__(
        self,
        strategy: Strategy | type[Strategy],
        symbols: list[str],
        config: BacktestConfig | None = None,
        primary_bars: tuple[int, str] | None = None,
        warmup_bars: dict[str, dict] | None = None,
        directions: dict[str, int] | None = None,
        instruments: dict | None = None,
        instrument_configs: dict[str, InstrumentConfig] | None = None,
        oms_type: str = "netting",
        account_type: str = "cash",
        leverage: float = 1.0,
        initial_positions: dict[str, dict] | None = None,
        bar_stamp: Literal["open", "close"] | None = None,
        bar_ns: int | dict[str, int] | None = None,
    ):
        self._node = LiveNode(
            [strategy],
            symbols,
            config,
            primary_bars=primary_bars,
            warmup_bars=warmup_bars,
            directions=directions,
            instruments=instruments,
            instrument_configs=instrument_configs,
            oms_type=oms_type,
            account_type=account_type,
            leverage=leverage,
            initial_positions={0: initial_positions} if initial_positions else None,
            bar_stamp=bar_stamp,
            bar_ns=bar_ns,
            _via="TickStrategyStream",
        )

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
        ltq: float = 0.0,
        bid_qty: float = 0.0,
        ask_qty: float = 0.0,
        oi: float = 0.0,
    ) -> int:
        """Feed one tick row; fires every hook it triggers before returning.

        ``ltp > 0`` prints a trade; a two-sided book yields a quote after
        it. ``ltq`` is the print's own size when the feed has it (else the
        flow deltas stand in); ``bid_qty``/``ask_qty`` are the displayed L1
        sizes; ``oi`` the open interest. All four default to "unknown".
        Returns how many events were appended (0–2).

        A row that yields no event still carries a timestamp, and alerts or
        timers due by it fire before this returns. Orders they submit are
        routed to their instrument at once and match from its next print;
        intents they queue are held for this symbol's next print.
        """
        return self._node.push_tick(
            symbol, timestamp, ltp, bid, ask, buy_qty_delta, sell_qty_delta, ltq, bid_qty, ask_qty, oi
        )

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
        """Feed one closed bar. It executes: orders match, equity marks.

        ``timestamp`` is stamped as the stream's ``bar_stamp`` declares.
        """
        self._node.push_bar(symbol, timestamp, open, high, low, close, volume)

    def push_depth(
        self,
        symbol: str,
        timestamp: int,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Feed one depth snapshot: ``(price, size)`` lists, best first."""
        self._node.push_depth(symbol, timestamp, bids, asks)

    # -- state ---------------------------------------------------------------

    @property
    def ctx(self):
        """The context handed to the strategy's hooks."""
        return self._node.ctx(0)

    @property
    def strategy(self) -> Strategy:
        return self._node.strategies[0]

    @property
    def equity(self) -> float:
        return self._node.equity(0)

    @property
    def is_halted(self) -> bool:
        """Whether a margin call or drawdown kill-switch has latched."""
        return self._node.is_halted(0)

    def positions(self, symbol: str):
        return self._node.positions(0, symbol)

    def finish(self) -> PortfolioResult:
        """Close out and compute metrics; the stream is unusable after."""
        (result,) = self._node.finish()
        return result
