"""Order intents and typed orders emitted by strategies.

The legacy intents (:class:`MarketOrder`, :class:`ClosePosition`) keep their
import path from the original ``orders`` module. The typed orders
(:class:`Market`, :class:`Limit`, :class:`StopMarket`, :class:`StopLimit`)
are the 0.5.0 order API, submitted via ``Strategy.submit_order``.
"""

from raptorbt.strategy.orders.intents import ClosePosition, MarketOrder
from raptorbt.strategy.orders.types import (
    Limit,
    LimitIfTouched,
    Market,
    MarketIfTouched,
    MarketToLimit,
    StopLimit,
    StopMarket,
    TrailingStopLimit,
    TrailingStopMarket,
    Twap,
)

__all__ = [
    "ClosePosition",
    "Limit",
    "LimitIfTouched",
    "Market",
    "MarketIfTouched",
    "MarketOrder",
    "MarketToLimit",
    "StopLimit",
    "Twap",
    "StopMarket",
    "TrailingStopLimit",
    "TrailingStopMarket",
]
