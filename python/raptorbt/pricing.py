"""European option pricing, Greeks and implied volatility.

Two models, both lognormal:

``"black_scholes"``
    ``underlying`` is a spot price. The forward is ``S * exp((rate - dividend_yield) * tau)``
    (Black-Scholes-Merton with a continuous dividend or carry yield).
``"black76"``
    ``underlying`` is already a forward (a futures price); no carry is applied, the rate only
    discounts the premium, and ``dividend_yield`` must be 0.

Every function takes scalars or NumPy-broadcastable arrays. With scalars it returns scalars and
raises ``ValueError`` for inputs no model can price. With arrays it returns arrays and marks any
invalid element ``NaN`` (``NaN`` means absent, never zero); the work runs in Rust, off the GIL,
on all cores for large inputs.

Units: ``tau`` is years (see :func:`year_fraction`); ``vol``, ``rate`` and ``dividend_yield`` are
annual decimals (0.2 = 20%); ``vega`` is per 1.00 of volatility, ``rho`` per 1.00 of rate,
``theta`` per year (negative when the option decays), ``delta`` per unit of ``underlying``.
Divide ``vega`` and ``rho`` by 100 for "per volatility point" and "per 1%", and ``theta`` by 365
for "per calendar day".

Implied volatility never guesses. :func:`implied_vol` raises :class:`ImpliedVolError` (scalar) or
returns ``NaN`` (arrays) when no volatility in ``[lo, hi]`` reproduces the price;
:func:`implied_vol_status` says why, with an :class:`IvStatus`.

See ``docs/pricing.md`` for the numerical assumptions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from raptorbt.errors import InvalidParameter

from raptorbt._raptorbt import (
    Greeks,
    OptionChain,
    pricing_forward as _forward,
    pricing_greeks as _greeks,
    pricing_greeks_array as _greeks_array,
    pricing_implied_vol as _iv,
    pricing_implied_vol_array as _iv_array,
    pricing_price as _price,
    pricing_price_array as _price_array,
    pricing_status_name as _status_name,
    pricing_year_fraction as _year_fraction,
)

_F64 = npt.NDArray[np.float64]

MODELS = ("black_scholes", "black76")


class IvStatus(enum.IntEnum):
    """Why an implied volatility exists or does not."""

    OK = 0
    EXPIRED = 1  #: ``tau == 0``
    INVALID_INPUT = 2  #: an input is invalid or not finite
    BELOW_INTRINSIC = 3  #: price is negative or under discounted intrinsic value
    ABOVE_UPPER_BOUND = 4  #: price at or above the discounted forward (call) or strike (put)
    BELOW_BRACKET = 5  #: needs a volatility at or below ``lo`` (zero time value)
    ABOVE_BRACKET = 6  #: needs a volatility at or above ``hi``
    NO_CONVERGENCE = 7  #: ran out of iterations inside the bracket

    @property
    def text(self) -> str:
        return _status_name(int(self))


class ImpliedVolError(ValueError):
    """No implied volatility; ``status`` is the :class:`IvStatus` and says why."""

    def __init__(self, status: IvStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class GreeksArray:
    """Price and Greeks of many options, one array per field."""

    price: _F64
    delta: _F64
    gamma: _F64
    vega: _F64
    theta: _F64
    rho: _F64


def _is_call(right: Any) -> Any:
    """``True``/``False`` for one right, a boolean array for many."""
    r = np.char.lower(np.asarray(right, dtype=str))
    call = (r == "call") | (r == "c")
    put = (r == "put") | (r == "p")
    if not bool(np.all(call | put)):
        raise InvalidParameter(f"right is 'call' or 'put' (or 'c'/'p'), got {right!r}",
                               code="invalid_option_right", parameter="right", value=repr(right))
    return bool(call) if call.ndim == 0 else call


def _check_model(model: str) -> str:
    if model not in MODELS:
        raise InvalidParameter(f"model is one of {', '.join(MODELS)}, got {model!r}",
                               code="invalid_parameter", parameter="model", value=model,
                               valid=list(MODELS))
    return model


def _scalar(*xs: Any) -> bool:
    return all(np.ndim(x) == 0 for x in xs)


def _flat(shape: tuple[int, ...], is_call: Any, *xs: Any) -> tuple[Any, ...]:
    """Broadcast to one shape and flatten to C-contiguous 1-D arrays."""
    c = np.ascontiguousarray(np.broadcast_to(is_call, shape).ravel(), dtype=np.bool_)
    out = [
        np.ascontiguousarray(np.broadcast_to(np.asarray(x, np.float64), shape).ravel()) for x in xs
    ]
    return (c, *out)


def _shape(*xs: Any) -> tuple[int, ...]:
    return np.broadcast_shapes(*[np.shape(x) for x in xs])


def price(
    right: Any,
    underlying: Any,
    strike: Any,
    tau: Any,
    vol: Any,
    rate: Any = 0.0,
    dividend_yield: Any = 0.0,
    *,
    model: str = "black_scholes",
) -> Any:
    """Option premium. ``right`` is ``"call"`` or ``"put"`` (or an array of them)."""
    _check_model(model)
    call = _is_call(right)
    args = (underlying, strike, tau, vol, rate, dividend_yield)
    if _scalar(right, *args):
        return _price(model, call, *(float(a) for a in args))
    shape = _shape(right, *args)
    c, *flat = _flat(shape, call, *args)
    return _price_array(model, c, *flat).reshape(shape)


def greeks(
    right: Any,
    underlying: Any,
    strike: Any,
    tau: Any,
    vol: Any,
    rate: Any = 0.0,
    dividend_yield: Any = 0.0,
    *,
    model: str = "black_scholes",
) -> Greeks | GreeksArray:
    """Price, delta, gamma, vega, theta and rho. A :class:`Greeks` for scalars, a
    :class:`GreeksArray` for arrays."""
    _check_model(model)
    call = _is_call(right)
    args = (underlying, strike, tau, vol, rate, dividend_yield)
    if _scalar(right, *args):
        return _greeks(model, call, *(float(a) for a in args))
    shape = _shape(right, *args)
    c, *flat = _flat(shape, call, *args)
    cols = _greeks_array(model, c, *flat)
    return GreeksArray(*(a.reshape(shape) for a in cols))


def implied_vol_status(
    price: Any,
    right: Any,
    underlying: Any,
    strike: Any,
    tau: Any,
    rate: Any = 0.0,
    dividend_yield: Any = 0.0,
    *,
    model: str = "black_scholes",
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[Any, Any]:
    """``(iv, status)``: never raises for a price with no volatility. ``iv`` is ``NaN`` unless
    ``status`` is :attr:`IvStatus.OK`. A scalar call gives ``(float, IvStatus)``, an array call
    ``(float64 array, uint8 array of IvStatus codes)``."""
    _check_model(model)
    call = _is_call(right)
    args = (underlying, strike, tau, rate, dividend_yield)
    if _scalar(price, right, *args):
        iv, code = _iv(model, float(price), call, *(float(a) for a in args), lo, hi, tol, max_iter)
        return iv, IvStatus(code)
    shape = _shape(price, right, *args)
    c, p, *flat = _flat(shape, call, price, *args)
    iv, code = _iv_array(model, p, c, *flat, lo, hi, tol, max_iter)
    return iv.reshape(shape), code.reshape(shape)


def implied_vol(
    price: Any,
    right: Any,
    underlying: Any,
    strike: Any,
    tau: Any,
    rate: Any = 0.0,
    dividend_yield: Any = 0.0,
    *,
    model: str = "black_scholes",
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Any:
    """The volatility at which the model prices ``price``.

    Scalars: returns a float or raises :class:`ImpliedVolError` (its ``status`` says why).
    Arrays: returns an array with ``NaN`` wherever there is none; use
    :func:`implied_vol_status` to see why. Search is limited to ``[lo, hi]``
    (default 0.01% to 500%); a price outside what that range reaches is reported, not clamped.
    """
    iv, status = implied_vol_status(
        price, right, underlying, strike, tau, rate, dividend_yield,
        model=model, lo=lo, hi=hi, tol=tol, max_iter=max_iter,
    )  # fmt: skip
    if np.ndim(iv) == 0 and status != IvStatus.OK:
        raise ImpliedVolError(status, f"no implied volatility: {status.text}")
    return iv


def forward_price(spot: Any, tau: Any, rate: Any = 0.0, dividend_yield: Any = 0.0) -> Any:
    """``spot * exp((rate - dividend_yield) * tau)``."""
    if _scalar(spot, tau, rate, dividend_yield):
        return _forward(float(spot), float(tau), float(rate), float(dividend_yield))
    return np.asarray(spot, np.float64) * np.exp(
        (np.asarray(rate, np.float64) - np.asarray(dividend_yield, np.float64))
        * np.asarray(tau, np.float64)
    )


def year_fraction(now_ns: int, expiry_ns: int) -> float:
    """Years from ``now_ns`` to ``expiry_ns`` (nanosecond timestamps) on a 365-day year: the
    ``tau`` to pass to the pricing functions."""
    return _year_fraction(int(now_ns), int(expiry_ns))


__all__ = [
    "MODELS",
    "Greeks",
    "GreeksArray",
    "ImpliedVolError",
    "IvStatus",
    "OptionChain",
    "forward_price",
    "greeks",
    "implied_vol",
    "implied_vol_status",
    "price",
    "year_fraction",
]
