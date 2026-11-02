"""Input checks shared by the strategy runners."""

from __future__ import annotations

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
