"""Strategy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable parameter set for a strategy.

    Subclass to declare typed parameters::

        @dataclass(frozen=True)
        class SmaCrossConfig(StrategyConfig):
            fast_period: int = 10
            slow_period: int = 30

    ``params`` is a free-form escape hatch for callers that parameterize
    strategies generically (e.g. parameter sweeps) without declaring a
    config subclass.
    """

    params: dict[str, Any] = field(default_factory=dict)
