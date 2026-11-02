"""Input checks shared by the strategy runners."""

from __future__ import annotations

import math

import numpy as np

from raptorbt.errors import InvalidData


def ensure_prices_finite(label: str, operation: str, arrays: dict) -> None:
    """Refuse a NaN or infinite open, high, low or close.

    A non-finite price compares false against everything, so the run would
    complete and leave a NaN in the equity curve. The error names the column
    and the first offending index.
    """
    for column in ("open", "high", "low", "close"):
        bad = np.flatnonzero(~np.isfinite(arrays[column]))
        if bad.size:
            index = int(bad[0])
            raise InvalidData(
                f"{label}: {column} is not finite ({float(arrays[column][index])}) "
                f"at index {index}",
                code="non_finite_price",
                operation=operation,
                parameter=column,
                value=index,
            )


def ensure_pushed_finite(label: str, operation: str, **prices: float) -> None:
    """Refuse a NaN or infinite price pushed to a live node, naming it.

    A non-finite price would be ignored by some comparisons and poison
    others, so the push is refused and changes nothing.
    """
    for name, value in prices.items():
        if not math.isfinite(value):
            raise InvalidData(
                f"{label}: {name} is not finite ({value}) in a pushed event",
                code="non_finite_price",
                operation=operation,
                parameter=name,
                value=repr(value),
            )
