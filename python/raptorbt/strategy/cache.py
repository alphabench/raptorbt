"""Event-sourced order/trade cache.

A Python-side mirror rebuilt purely from the engine's event stream — no
per-query FFI. The engine remains the source of truth; the cache exists so
strategies can ask "what happened to my order?" without bookkeeping of
their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderRecord:
    """Latest known state of one client order."""

    client_id: str
    status: str  # accepted | triggered | filled | canceled | expired | rejected
    symbol: str | None = None
    last_price: float | None = None
    last_size: float | None = None
    reject_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status in ("accepted", "triggered")


_STATUS_BY_KIND = {
    "order_accepted": "accepted",
    "order_triggered": "triggered",
    "order_filled": "filled",
    "order_canceled": "canceled",
    "order_expired": "expired",
    "order_rejected": "rejected",
}


@dataclass
class Cache:
    """Order states and closed trades, folded from events."""

    _orders: dict[str, OrderRecord] = field(default_factory=dict)
    _closed_trades: list = field(default_factory=list)

    def _observe(self, event, symbol: str | None = None) -> None:
        status = _STATUS_BY_KIND.get(event.kind)
        if status is not None and event.client_order_id:
            record = self._orders.setdefault(
                event.client_order_id,
                OrderRecord(client_id=event.client_order_id, status=status),
            )
            record.status = status
            if symbol is not None:
                record.symbol = symbol
            if event.price is not None:
                record.last_price = event.price
            if event.size is not None:
                record.last_size = event.size
            if event.reject_reason is not None:
                record.reject_reason = event.reject_reason
        elif event.kind == "exited" and event.trade is not None:
            self._closed_trades.append(event.trade)

    # -- queries -------------------------------------------------------------

    def order(self, client_id: str) -> OrderRecord | None:
        return self._orders.get(client_id)

    def orders(self) -> list[OrderRecord]:
        return list(self._orders.values())

    def orders_open(self) -> list[OrderRecord]:
        return [o for o in self._orders.values() if o.is_open]

    def orders_closed(self) -> list[OrderRecord]:
        return [o for o in self._orders.values() if not o.is_open]

    def is_order_open(self, client_id: str) -> bool:
        record = self._orders.get(client_id)
        return record is not None and record.is_open

    def closed_trades(self) -> list:
        """Round-trip trades observed so far, in close order."""
        return list(self._closed_trades)

    def realized_pnl(self, symbol: str | None = None) -> float:
        """Sum of closed-trade PnL, optionally for one symbol."""
        return sum(
            t.pnl for t in self._closed_trades if symbol is None or t.symbol == symbol
        )
