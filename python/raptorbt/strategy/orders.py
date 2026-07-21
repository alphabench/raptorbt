"""Order intents emitted by strategies.

Intents describe what the strategy wants; the engine decides fills, sizing
against available capital, lot rounding, and fees.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketOrder:
    """Open a position at the next fill price in the session's direction.

    Attributes:
        size_frac: Fraction of available capital to deploy (0, 1]. ``None``
            deploys all available capital.
        stop_price: Explicit stop price for the new position, overriding any
            configured stop model.
        target_price: Explicit target price for the new position, overriding
            any configured target model.
    """

    size_frac: float | None = None
    stop_price: float | None = None
    target_price: float | None = None


@dataclass(frozen=True)
class ClosePosition:
    """Close the open position at the next fill price."""
