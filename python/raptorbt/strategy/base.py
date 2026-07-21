"""Strategy base class."""

from __future__ import annotations

import logging

from raptorbt.strategy.cache import Cache
from raptorbt.strategy.clock import Clock
from raptorbt.strategy.config import StrategyConfig
from raptorbt.strategy.context import StrategyContext
from raptorbt.strategy.orders import ClosePosition, MarketOrder
from raptorbt.strategy.orders.types import _OrderBase


class Strategy:
    """Base class for event-driven strategies.

    Subclass and override the hooks you need. All hooks default to no-ops.
    Order intents are emitted from inside ``on_bar`` via :meth:`enter` /
    :meth:`close_position`; the engine applies them on the same bar using its
    configured fill model, then feeds resulting events back through the
    ``on_order_*`` / ``on_position_*`` hooks.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config if config is not None else StrategyConfig()
        self._pending_orders: list[MarketOrder | ClosePosition] = []
        # Typed-order commands, drained by the runner each bar:
        # ("submit", client_id, order, parent) / ("cancel", client_id) /
        # ("cancel_all",) / ("modify", client_id, kwargs) / ("close", pid) /
        # ("link_oco", ids).
        self._pending_commands: list[tuple] = []
        self._order_seq = 0
        # (step, unit) bar subscriptions declared in on_start.
        self._bar_subscriptions: list[tuple[int, str]] = []
        #: Bar-driven clock: ``set_time_alert`` / ``set_timer``; fresh per run.
        self.clock = Clock()
        #: Event-sourced order/trade cache; fresh per run.
        self.cache = Cache()
        # (indicator, stream_id) registrations; stream_id None = primary bars.
        self._indicators: list[tuple[object, int | None]] = []

    # -- lifecycle hooks ----------------------------------------------------

    def on_start(self, ctx: StrategyContext) -> None:
        """Called once before the first bar.

        The full OHLCV arrays on ``ctx`` are available here for indicator
        precomputation.
        """

    def on_bar(self, ctx: StrategyContext) -> None:
        """Called once per bar, in ascending time order."""

    def on_stop(self, ctx: StrategyContext) -> None:
        """Called once after the last bar, before finalization."""

    def on_time_event(self, ctx: StrategyContext, event) -> None:
        """Called for each due clock alert/timer, *before* the bar's data
        handlers — the scheduled time precedes the bar that revealed it."""

    def on_trade_tick(self, ctx: StrategyContext, tick) -> None:
        """A trade printed. Only fires in tick runs.

        ``ctx.best_bid`` / ``ctx.best_ask`` hold the last book observed
        *before* this print — the quote from the same feed row arrives in
        the following :meth:`on_quote`, so acting on it here would be
        reading a book this very print already moved.
        """

    def on_quote(self, ctx: StrategyContext, quote) -> None:
        """The top of book changed. Only fires in tick runs.

        Quotes do not fill orders. An order submitted here rests and matches
        against the next print, which is the first evidence of a trade at
        that price.
        """

    def on_order_book(self, ctx: StrategyContext, book) -> None:
        """The order book changed. Only fires in tick runs with depth data.

        Like quotes, books do not fill orders — displayed size is intent,
        not a trade. An order submitted here rests and matches on the next
        print.
        """

    def on_composite_bar(self, ctx: StrategyContext, bar) -> None:
        """Called when a subscribed higher-timeframe bar completes.

        ``bar`` is a :class:`~raptorbt.strategy.context.CompositeBar`;
        dispatched *before* ``on_bar`` of the primary bar that completed it,
        since the composite closed strictly earlier.
        """

    # -- order/position event hooks -----------------------------------------

    def on_order_filled(self, ctx: StrategyContext, event) -> None:
        """Called when an entry or exit fill occurs."""

    def on_order_rejected(self, ctx: StrategyContext, event) -> None:
        """Called when an entry intent or order is refused."""

    def on_order_accepted(self, ctx: StrategyContext, event) -> None:
        """Called when a submitted order starts working."""

    def on_order_triggered(self, ctx: StrategyContext, event) -> None:
        """Called when a stop-limit's trigger fires."""

    def on_order_canceled(self, ctx: StrategyContext, event) -> None:
        """Called when an order is canceled (explicitly or IOC/FOK)."""

    def on_order_expired(self, ctx: StrategyContext, event) -> None:
        """Called when an order's time-in-force lapses."""

    def on_order_event(self, ctx: StrategyContext, event) -> None:
        """Called for every order/account event, after its granular hook."""

    def on_algo_started(self, ctx: StrategyContext, event) -> None:
        """An execution schedule was registered. Slices arrive afterwards
        as ordinary order events, with client ids of ``"<parent>#<n>"``."""

    def on_algo_completed(self, ctx: StrategyContext, event) -> None:
        """A schedule released its last slice, or was cancelled.

        "Completed" means fully *released*, not necessarily fully filled.
        """

    def on_margin_call(self, ctx: StrategyContext, event) -> None:
        """Called when equity breaches the maintenance requirement (margin
        accounts). ``event.price`` carries the equity, ``event.size`` the
        requirement; new entries halt for the rest of the run."""

    def on_position_opened(self, ctx: StrategyContext, event) -> None:
        """Called after a position opens."""

    def on_position_closed(self, ctx: StrategyContext, event) -> None:
        """Called after a position closes; ``event.trade`` is the round trip."""

    # -- order API -----------------------------------------------------------

    def enter(
        self,
        size_frac: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
    ) -> None:
        """Queue a market entry in the session's direction for this bar."""
        self._pending_orders.append(
            MarketOrder(size_frac=size_frac, stop_price=stop_price, target_price=target_price)
        )

    # ``buy`` reads naturally in long-only strategies; it is the same intent.
    buy = enter

    def close_position(self, position_id: int | None = None, symbol: str | None = None) -> None:
        """Queue a close of the open position for this bar.

        With ``position_id`` (from ``ctx.positions``), closes that specific
        position — required under the hedging policy, where several are open
        at once. Without it, the legacy whole-position close intent.
        ``symbol`` routes in multi-instrument runs (defaults to the current
        bar's symbol).
        """
        if position_id is None and symbol is None:
            self._pending_orders.append(ClosePosition())
        elif position_id is None:
            self._pending_commands.append(("close_all_for", symbol))
        else:
            self._pending_commands.append(("close", position_id, symbol))

    def submit_order(
        self,
        order: _OrderBase,
        client_id: str | None = None,
        parent: str | None = None,
        symbol: str | None = None,
    ) -> str:
        """Queue a typed order (:mod:`raptorbt.strategy.orders`).

        Returns the client order id — auto-generated as
        ``"{order_id_tag}-{seq}"`` when not supplied — which identifies the
        order on every subsequent event and in :meth:`cancel_order` /
        :meth:`modify_order`. ``parent`` (another order's client id) holds
        this order until the parent fills; it dies if the parent does.
        ``symbol`` routes the order in multi-instrument runs; single-
        instrument runs ignore it, and the portfolio runner defaults it to
        the symbol whose bar is being processed.
        """
        if not isinstance(order, _OrderBase):
            raise TypeError(
                "submit_order takes a typed order (orders.Market/Limit/StopMarket/"
                f"StopLimit/...), got {type(order).__name__}"
            )
        if client_id is None:
            tag = getattr(self.config, "order_id_tag", None) or "O"
            client_id = f"{tag}-{self._order_seq}"
        self._order_seq += 1
        self._pending_commands.append(("submit", client_id, order, parent, symbol))
        return client_id

    def register_indicator(
        self, indicator, stream_id: int | None = None, symbol: str | None = None
    ):
        """Auto-update a streaming indicator (``raptorbt.Indicator``) from
        bar data, *before* handlers see the bar.

        ``stream_id=None`` feeds it primary-stream bars; a handle from
        :meth:`subscribe_bars` feeds it that composite stream instead.
        Returns the indicator for chained assignment. Call from
        ``on_start``.

        ``symbol`` routes the indicator in portfolio runs and is ignored in
        single-instrument ones. **Pass it.** Without it the indicator is fed
        every symbol's bars interleaved, which is almost never meaningful —
        one indicator cannot track N series. The usual shape is one
        indicator per symbol::

            self.fast = {
                s: self.register_indicator(Indicator.sma(10), symbol=s)
                for s in ctx.symbols
            }

        or, equivalently, :meth:`register_indicators`.
        """
        self._indicators.append((indicator, stream_id, symbol))
        return indicator

    def register_indicators(self, factory, symbols, stream_id: int | None = None) -> dict:
        """Register one indicator per symbol, built by ``factory()``.

        Returns ``symbol -> indicator``::

            self.fast = self.register_indicators(
                lambda: Indicator.sma(10), ctx.symbols
            )
        """
        return {
            symbol: self.register_indicator(factory(), stream_id=stream_id, symbol=symbol)
            for symbol in symbols
        }

    def indicators_initialized(self) -> bool:
        """Whether every registered indicator has completed warmup.

        In a portfolio run that means every symbol's indicator is warm.
        """
        return all(ind.initialized for ind, _, _ in self._indicators)

    def subscribe_bars(self, step: int, unit: str, *, brick_size: float = 0.0) -> int:
        """Subscribe to bars aggregated from the primary stream.

        Call from ``on_start``. Completed bars arrive via
        :meth:`on_composite_bar`. Returns the stream handle carried on each
        bar's ``stream_id``. Units: time (``"ms"``/``"s"``/``"m"``/``"h"``/
        ``"d"``/``"w"``), ``"tick"``, ``"volume"``, ``"value"``.

        In a portfolio run one subscription yields one aggregated stream
        *per symbol*, each built only from that symbol's bars. The symbol
        that completed a bar arrives as ``bar.symbol`` (and ``ctx.symbol``).
        """
        if step < 1:
            raise ValueError("step must be >= 1")
        self._bar_subscriptions.append((step, unit, brick_size))
        return len(self._bar_subscriptions) - 1

    def link_oco(self, *client_ids: str) -> None:
        """Queue a one-cancels-other link between submitted orders."""
        if len(client_ids) < 2:
            raise ValueError("link_oco needs at least two client ids")
        self._pending_commands.append(("link_oco", list(client_ids)))

    def submit_bracket(
        self,
        entry: _OrderBase,
        stop_trigger: float,
        target_price: float,
        stop_limit_price: float | None = None,
    ) -> tuple[str, str, str]:
        """Queue an entry with protective legs: one-triggers-other children
        (stop + target activate when the entry fills) linked
        one-cancels-other with each other.

        The legs close the full position on the opposite side. Netting
        policy only — under hedging every order opens, so use per-position
        ``stop_price``/``target_price`` attachments instead.

        Returns ``(entry_id, stop_id, target_id)``.
        """
        from raptorbt.strategy.orders.types import Limit, StopLimit, StopMarket

        exit_side = "sell" if entry.side == "buy" else "buy"
        entry_id = self.submit_order(entry)
        stop_order = (
            StopMarket(side=exit_side, trigger=stop_trigger)
            if stop_limit_price is None
            else StopLimit(side=exit_side, trigger=stop_trigger, price=stop_limit_price)
        )
        stop_id = self.submit_order(stop_order, parent=entry_id)
        target_id = self.submit_order(
            Limit(side=exit_side, price=target_price), parent=entry_id
        )
        self.link_oco(stop_id, target_id)
        return entry_id, stop_id, target_id

    def cancel_order(self, client_id: str) -> None:
        """Queue cancellation of a working order by client id."""
        self._pending_commands.append(("cancel", client_id))

    def cancel_all_orders(self) -> None:
        """Queue cancellation of every working order."""
        self._pending_commands.append(("cancel_all",))

    def modify_order(
        self,
        client_id: str,
        units: float | None = None,
        size_frac: float | None = None,
        limit_price: float | None = None,
        trigger_price: float | None = None,
    ) -> None:
        """Queue a price/quantity replacement of a working order."""
        self._pending_commands.append(
            (
                "modify",
                client_id,
                {
                    "units": units,
                    "size_frac": size_frac,
                    "limit_price": limit_price,
                    "trigger_price": trigger_price,
                },
            )
        )

    def drain_orders(self) -> list[MarketOrder | ClosePosition]:
        """Return and clear pending intents. Called by the runner."""
        pending = self._pending_orders
        self._pending_orders = []
        return pending

    def drain_commands(self) -> list[tuple]:
        """Return and clear pending typed-order commands. Called by the runner."""
        pending = self._pending_commands
        self._pending_commands = []
        return pending

    # -- logging -------------------------------------------------------------

    @property
    def log(self) -> logging.Logger:
        """Logger namespaced to the concrete strategy class."""
        return logging.getLogger(f"raptorbt.strategy.{type(self).__name__}")
