"""Live tick stream for class-based strategies.

The batch runners replay a finished dataset; this drives the same session,
kernels and strategy hooks from events pushed one at a time — an open-ended
feed whose end is not known in advance. Push a tick, and every hook it
triggers fires before ``push_tick`` returns, so the caller reads positions
and queued state synchronously.

Semantics are identical to :func:`raptorbt.run_tick_strategy` — both drive
the same dispatch loop — with two additions. Pushed or warmup *bars* execute
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

import numpy as np

from raptorbt._raptorbt import (
    BacktestConfig,
    InstrumentConfig,
    PortfolioResult,
    PortfolioSession,
)
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.portfolio_runner import apply_commands_on
from raptorbt.strategy.tick_runner import (
    TickContext,
    advance_clock,
    drive_tick_events,
    setup_tick_strategy,
)


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

    Everything else — account type, leverage, risk limits, OMS type —
    behaves as in :func:`raptorbt.run_tick_strategy`.
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

        session = PortfolioSession(
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
                i,
                a["timestamps"],
                a["open"],
                a["high"],
                a["low"],
                a["close"],
                a["volume"],
            )
            arrays[symbol] = a
        session.seal()

        for symbol, seed in (initial_positions or {}).items():
            if symbol not in self._index_of:
                raise ValueError(f"initial_positions names unknown symbol {symbol!r}")
            quantity = float(seed.get("quantity") or 0)
            avg_price = float(seed.get("avg_price") or 0)
            if quantity <= 0 or avg_price <= 0:
                raise ValueError(
                    f"initial_positions[{symbol!r}] needs positive quantity "
                    f"and avg_price (got {quantity} @ {avg_price})"
                )
            session.adopt_position(
                self._index_of[symbol],
                int(seed.get("timestamp_ns") or 0),
                avg_price,
                quantity,
            )

        self._session = session
        self.ctx = TickContext(session, self._symbols, arrays)
        self._clocks, self._streams, self._primary = setup_tick_strategy(
            strategy, self.ctx, self._symbols, primary_bars
        )
        self._id_map: dict[str, tuple[int, int]] = {}
        self._apply_commands = apply_commands_on(
            strategy, session, self.ctx, self._symbols, self._id_map
        )
        # Intents (enter / close_position) a timer queued for a symbol
        # between its prints, waiting for that symbol's next print.
        self._held_intents: dict[str, list] = {}
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
        self._check_open()
        appended = self._session.push_tick(
            self._index_of[symbol],
            timestamp,
            ltp,
            bid,
            ask,
            buy_qty_delta,
            sell_qty_delta,
            ltq=ltq,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            oi=oi,
        )
        if appended:
            self._drain()
        else:
            self._mark_time(symbol, timestamp)
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

    def finish(self) -> PortfolioResult:
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
            before_trade=self._release_held_intents,
        )

    def _mark_time(self, symbol: str, timestamp: int) -> None:
        """A refused print still proves the market reached its instant.

        Fire what is due on this symbol's clock now. Commands (submit_order,
        cancel, close by id) are routed at once; with no event in flight the
        session stamps them on the target's last print, so they match from
        its next one. Intents have no symbol of their own and ride a print,
        so they are held for this symbol's next trade event rather than
        left on the queue for whichever symbol prints first.
        """
        advance_clock(self._strategy, self.ctx, self._clocks, symbol, timestamp)
        self._apply_commands(self._index_of[symbol], 0, timestamp)
        queued = self._strategy.drain_orders()
        if queued:
            self._held_intents.setdefault(symbol, []).extend(queued)

    def _release_held_intents(self, symbol: str) -> None:
        held = self._held_intents.pop(symbol, None)
        if held:
            self._strategy._pending_orders[:0] = held
