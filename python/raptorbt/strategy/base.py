"""Strategy base class."""

from __future__ import annotations

import logging

from raptorbt.strategy.config import StrategyConfig
from raptorbt.strategy.context import StrategyContext
from raptorbt.strategy.orders import ClosePosition, MarketOrder


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

    # -- order/position event hooks -----------------------------------------

    def on_order_filled(self, ctx: StrategyContext, event) -> None:
        """Called when an entry or exit fill occurs."""

    def on_order_rejected(self, ctx: StrategyContext, event) -> None:
        """Called when an entry intent is refused (e.g. by risk constraints)."""

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

    def close_position(self) -> None:
        """Queue a close of the open position for this bar."""
        self._pending_orders.append(ClosePosition())

    def drain_orders(self) -> list[MarketOrder | ClosePosition]:
        """Return and clear pending intents. Called by the runner."""
        pending = self._pending_orders
        self._pending_orders = []
        return pending

    # -- logging -------------------------------------------------------------

    @property
    def log(self) -> logging.Logger:
        """Logger namespaced to the concrete strategy class."""
        return logging.getLogger(f"raptorbt.strategy.{type(self).__name__}")
