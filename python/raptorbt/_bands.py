"""The Bollinger bands result type, shared by the array and streaming forms."""

from __future__ import annotations

from typing import NamedTuple


class BollingerBands(NamedTuple):
    """Bollinger bands in one order, ``(upper, middle, lower)``, by name or position.

    ``bollinger_bands(data, ...)`` returns three arrays; ``Indicator.bollinger(...).value``
    and ``ctx.indicators.bollinger(...).value`` return three floats (``None`` until warm).
    Read the fields by name when in doubt.
    """

    upper: object
    middle: object
    lower: object
