"""Position sizing: how many lots, from capital, risk, volatility or margin.

Every function returns a whole number of **lots** (multiply by the lot size for units). They are
pure: no engine state, no rounding up. A rule that comes to less than one lot raises
:class:`~raptorbt.errors.InvalidParameter` with code ``below_one_lot`` (``.value`` is the
fractional lots), and an input that is not a positive finite number raises
``invalid_sizing_input``. Zero is never returned silently.

``Sizer(ctx)`` binds the same rules to a strategy context's equity and free capital.

Lot size and ``multiplier`` convert a price to money per lot: ``price x lot_size x multiplier``.
"""

from __future__ import annotations

import math
from typing import Callable

from raptorbt.errors import InvalidParameter

__all__ = [
    "Sizer",
    "lots_for_capital",
    "lots_for_margin",
    "lots_for_percent_of_equity",
    "lots_for_risk",
    "lots_for_volatility",
    "max_lots_by_margin",
    "units",
]

_MAX_LOTS = 10_000_000


def _bad(name: str, value) -> InvalidParameter:
    return InvalidParameter(
        f"{name} must be a positive finite number, got {value!r}",
        code="invalid_sizing_input",
        operation="sizing",
        parameter=name,
        value=value,
    )


def _pos(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise _bad(name, value) from None
    if not (math.isfinite(v) and v > 0.0):
        raise _bad(name, value)
    return v


def _frac(name: str, value, hi: float = 1.0) -> float:
    v = _pos(name, value)
    if v > hi:
        raise InvalidParameter(
            f"{name} must be in (0, {hi:g}], got {value!r}",
            code="invalid_sizing_input",
            operation="sizing",
            parameter=name,
            value=value,
        )
    return v


def _floor_lots(raw: float, operation: str) -> int:
    # A tiny tolerance keeps 3 lots that arrive as 2.9999999999999996 from reading as 2.
    lots = math.floor(raw + 1e-9 * max(1.0, abs(raw)))
    if lots < 1:
        raise InvalidParameter(
            f"{operation} comes to {raw:.6g} lots, below one lot",
            code="below_one_lot",
            operation=operation,
            parameter="lots",
            value=raw,
        )
    return min(lots, _MAX_LOTS)


def units(lots: int, lot_size: float) -> float:
    """``lots`` as a quantity in units."""
    return lots * _pos("lot_size", lot_size)


def lots_for_capital(
    capital: float, price: float, lot_size: float, *, fraction: float = 1.0, multiplier: float = 1.0
) -> int:
    """The most lots whose cost (``price x lot_size x multiplier`` each) fits ``fraction`` of ``capital``."""
    cost = _pos("price", price) * _pos("lot_size", lot_size) * _pos("multiplier", multiplier)
    return _floor_lots(_pos("capital", capital) * _frac("fraction", fraction) / cost, "lots_for_capital")


def lots_for_percent_of_equity(
    equity: float, percent: float, price: float, lot_size: float, *, multiplier: float = 1.0
) -> int:
    """Lots worth ``percent`` (0 to 100) of ``equity`` at ``price``."""
    return lots_for_capital(
        equity, price, lot_size, fraction=_frac("percent", percent, 100.0) / 100.0, multiplier=multiplier
    )


def lots_for_risk(
    capital: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    lot_size: float,
    *,
    multiplier: float = 1.0,
) -> int:
    """The most lots that lose at most ``risk_fraction`` of ``capital`` if price runs from
    ``entry`` to ``stop``. A stop at the entry is refused (infinite size)."""
    per_unit = abs(_pos("entry", entry) - _pos("stop", stop))
    if per_unit == 0.0:
        raise InvalidParameter(
            "stop equals entry: the risk per unit is zero",
            code="invalid_sizing_input",
            operation="lots_for_risk",
            parameter="stop",
            value=stop,
        )
    risk = _pos("capital", capital) * _frac("risk_fraction", risk_fraction)
    per_lot = per_unit * _pos("lot_size", lot_size) * _pos("multiplier", multiplier)
    return _floor_lots(risk / per_lot, "lots_for_risk")


def lots_for_volatility(
    capital: float,
    target_vol: float,
    asset_vol: float,
    price: float,
    lot_size: float,
    *,
    max_leverage: float | None = None,
    multiplier: float = 1.0,
) -> int:
    """Lots whose notional is ``capital x target_vol / asset_vol`` (both annualised decimals),
    capped at ``capital x max_leverage`` of notional when given."""
    notional = _pos("capital", capital) * _pos("target_vol", target_vol) / _pos("asset_vol", asset_vol)
    if max_leverage is not None:
        notional = min(notional, capital * _pos("max_leverage", max_leverage))
    per_lot = _pos("price", price) * _pos("lot_size", lot_size) * _pos("multiplier", multiplier)
    return _floor_lots(notional / per_lot, "lots_for_volatility")


def lots_for_margin(available: float, margin_per_lot: float, *, buffer: float = 0.0) -> int:
    """The most lots whose margin fits ``available`` less a ``buffer`` fraction held back."""
    free = _pos("available", available) * (1.0 - _frac("buffer", buffer) if buffer else 1.0)
    return _floor_lots(free / _pos("margin_per_lot", margin_per_lot), "lots_for_margin")


def max_lots_by_margin(
    required: Callable[[int], float], available: float, *, cap: int = 100_000
) -> int:
    """The largest ``n`` (up to ``cap``) with ``required(n) <= available``, for margin that is
    not linear in lots (a hedged group is cheaper than its legs). ``required`` must not fall as
    ``n`` grows. Raises ``below_one_lot`` when even one lot does not fit::

        max_lots_by_margin(lambda n: structure.resolve(chain, U, spot, now, n).margin(model, px), free)
    """
    avail = _pos("available", available)
    one = _pos("required(1)", required(1))
    if one > avail:
        raise InvalidParameter(
            f"one lot needs margin {one:g}, only {avail:g} is available",
            code="below_one_lot",
            operation="max_lots_by_margin",
            parameter="available",
            value=avail / one,
        )
    hi = 1
    while hi * 2 <= cap and required(hi * 2) <= avail:
        hi *= 2
    lo, hi = hi, min(hi * 2, cap + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if required(mid) <= avail:
            lo = mid
        else:
            hi = mid
    return lo


class Sizer:
    """The rules above over a strategy context: ``equity`` is ``ctx.equity``, ``available`` is
    ``ctx.free_capital``. Read at call time, so one ``Sizer`` serves a whole run."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    def capital(self, price: float, lot_size: float, *, fraction: float = 1.0, multiplier: float = 1.0) -> int:
        return lots_for_capital(self._ctx.free_capital, price, lot_size, fraction=fraction, multiplier=multiplier)

    def percent_of_equity(self, percent: float, price: float, lot_size: float, *, multiplier: float = 1.0) -> int:
        return lots_for_percent_of_equity(self._ctx.equity, percent, price, lot_size, multiplier=multiplier)

    def risk(self, risk_fraction: float, entry: float, stop: float, lot_size: float, *, multiplier: float = 1.0) -> int:
        return lots_for_risk(self._ctx.equity, risk_fraction, entry, stop, lot_size, multiplier=multiplier)

    def volatility(
        self, target_vol: float, asset_vol: float, price: float, lot_size: float,
        *, max_leverage: float | None = None, multiplier: float = 1.0,
    ) -> int:
        return lots_for_volatility(
            self._ctx.equity, target_vol, asset_vol, price, lot_size,
            max_leverage=max_leverage, multiplier=multiplier,
        )

    def margin(self, margin_per_lot: float, *, buffer: float = 0.0) -> int:
        return lots_for_margin(self._ctx.free_capital, margin_per_lot, buffer=buffer)

    def max_lots_by_margin(self, required: Callable[[int], float], *, cap: int = 100_000) -> int:
        return max_lots_by_margin(required, self._ctx.free_capital, cap=cap)
