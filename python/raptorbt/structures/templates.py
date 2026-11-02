"""The named structures, as factories returning a declarative :class:`Structure`.

Steps count strikes of the earliest expiry's ladder from the money strike, up positive; for a
put "steps out" means down, as a call's means up. ``side="sell"`` on a long structure (and
``"buy"`` on a short one) flips every leg.
"""

from __future__ import annotations

from raptorbt.structures.legs import (
    Delta,
    Expiry,
    Leg,
    Steps,
    Structure,
    UnderlyingLeg,
    _invalid,
)

_FLIP = {"buy": "sell", "sell": "buy"}


def _out(right: str) -> int:
    return 1 if right == "call" else -1


def _check_side(side: str) -> None:
    if side not in ("buy", "sell"):
        raise _invalid(f"side must be 'buy' or 'sell', got {side!r}", "side", side, ["buy", "sell"])


def _pos(n: int, name: str, minimum: int = 1) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < minimum:
        raise _invalid(f"{name} must be a whole number of at least {minimum}", name, n)


def _leg(right, steps, side, expiry, ratio=1) -> Leg:
    return Leg(right, Steps(steps), expiry or Expiry.nearest(), side, ratio)


def straddle(side: str = "buy", *, expiry: Expiry | None = None) -> Structure:
    """The money call and put, both ``side``."""
    _check_side(side)
    return Structure(
        (_leg("call", 0, side, expiry), _leg("put", 0, side, expiry)), "straddle"
    )


def strangle(steps: int = 1, side: str = "buy", *, expiry: Expiry | None = None) -> Structure:
    """The call ``steps`` up and the put ``steps`` down, both ``side``."""
    _check_side(side)
    _pos(steps, "steps")
    return Structure(
        (_leg("call", steps, side, expiry), _leg("put", -steps, side, expiry)), "strangle"
    )


def strangle_by_delta(
    delta: float, vol: float, side: str = "buy", *, rate: float = 0.0,
    dividend_yield: float = 0.0, expiry: Expiry | None = None,
) -> Structure:
    """A strangle whose call and put are the contracts nearest ``delta`` in magnitude."""
    _check_side(side)
    sel = Delta(delta, vol, rate, dividend_yield)
    ex = expiry or Expiry.nearest()
    return Structure((Leg("call", sel, ex, side), Leg("put", sel, ex, side)), "strangle")


def vertical(
    right: str, long_steps: int, short_steps: int, *, expiry: Expiry | None = None
) -> Structure:
    """Buy the contract ``long_steps`` out and sell the one ``short_steps`` out.
    ``long_steps < short_steps`` is a debit spread."""
    if long_steps == short_steps:
        raise _invalid("a vertical needs two different strikes", "short_steps", short_steps)
    s = _out(right) if right in ("call", "put") else 0
    return Structure(
        (_leg(right, s * long_steps, "buy", expiry), _leg(right, s * short_steps, "sell", expiry)),
        "vertical",
    )


def bull_call_spread(long_steps: int = 0, short_steps: int = 1, **kw) -> Structure:
    """Debit: buy the call ``long_steps`` up, sell the call ``short_steps`` up."""
    _require_order(long_steps, short_steps)
    return _renamed(vertical("call", long_steps, short_steps, **kw), "bull_call_spread")


def bear_put_spread(long_steps: int = 0, short_steps: int = 1, **kw) -> Structure:
    """Debit: buy the put ``long_steps`` down, sell the put ``short_steps`` down."""
    _require_order(long_steps, short_steps)
    return _renamed(vertical("put", long_steps, short_steps, **kw), "bear_put_spread")


def bear_call_spread(short_steps: int = 0, long_steps: int = 1, **kw) -> Structure:
    """Credit: sell the call ``short_steps`` up, buy the call ``long_steps`` up."""
    _require_order(short_steps, long_steps)
    return _renamed(vertical("call", long_steps, short_steps, **kw), "bear_call_spread")


def bull_put_spread(short_steps: int = 0, long_steps: int = 1, **kw) -> Structure:
    """Credit: sell the put ``short_steps`` down, buy the put ``long_steps`` down."""
    _require_order(short_steps, long_steps)
    return _renamed(vertical("put", long_steps, short_steps, **kw), "bull_put_spread")


def _require_order(near: int, far: int) -> None:
    if not far > near:
        raise _invalid("the second strike must be further out than the first", "steps", (near, far))


def _renamed(s: Structure, name: str) -> Structure:
    return Structure(s.legs, name)


def butterfly(
    right: str, wing_steps: int, side: str = "buy", *, expiry: Expiry | None = None
) -> Structure:
    """``side`` one wing, opposite two at the money, ``side`` the other wing, ``wing_steps`` apart."""
    _check_side(side)
    _pos(wing_steps, "wing_steps")
    o = _FLIP[side]
    return Structure(
        (
            _leg(right, -wing_steps, side, expiry),
            _leg(right, 0, o, expiry, 2),
            _leg(right, wing_steps, side, expiry),
        ),
        "butterfly",
    )


def condor(
    right: str, inner_steps: int, outer_steps: int, side: str = "buy", *, expiry: Expiry | None = None
) -> Structure:
    """In one right: ``side`` the outer strikes, the opposite the inner ones, symmetric about the money."""
    _check_side(side)
    _check_inner_outer(inner_steps, outer_steps)
    o = _FLIP[side]
    return Structure(
        (
            _leg(right, -outer_steps, side, expiry),
            _leg(right, -inner_steps, o, expiry),
            _leg(right, inner_steps, o, expiry),
            _leg(right, outer_steps, side, expiry),
        ),
        "condor",
    )


def _check_inner_outer(inner: int, outer: int) -> None:
    if not (isinstance(inner, int) and isinstance(outer, int) and 1 <= inner < outer):
        raise _invalid("need 1 <= inner_steps < outer_steps", "inner_steps", (inner, outer))


def iron_condor(
    inner_steps: int, outer_steps: int, side: str = "sell", *, expiry: Expiry | None = None
) -> Structure:
    """``side="sell"`` (the usual): sell the put ``inner_steps`` down and the call ``inner_steps``
    up, buy those ``outer_steps`` out. ``"buy"`` is the reverse."""
    _check_side(side)
    _check_inner_outer(inner_steps, outer_steps)
    o = _FLIP[side]
    return Structure(
        (
            _leg("put", -inner_steps, side, expiry),
            _leg("call", inner_steps, side, expiry),
            _leg("call", outer_steps, o, expiry),
            _leg("put", -outer_steps, o, expiry),
        ),
        "iron_condor",
    )


def iron_butterfly(wing_steps: int, side: str = "sell", *, expiry: Expiry | None = None) -> Structure:
    """``side="sell"``: sell the money call and put, buy a call and a put ``wing_steps`` out."""
    _check_side(side)
    _pos(wing_steps, "wing_steps")
    o = _FLIP[side]
    return Structure(
        (
            _leg("call", 0, side, expiry),
            _leg("put", 0, side, expiry),
            _leg("call", wing_steps, o, expiry),
            _leg("put", -wing_steps, o, expiry),
        ),
        "iron_butterfly",
    )


def calendar(
    right: str, steps: int = 0, *, near: Expiry | None = None, far: Expiry | None = None
) -> Structure:
    """Sell the contract ``steps`` out at the near expiry, buy the same strike at the far one
    (default: the first and second listed expiries). Refused when the far expiry does not list
    the strike."""
    _pos(steps, "steps", 0)
    at = _out(right) * steps
    return Structure(
        (
            _leg(right, at, "buy", far or Expiry.nth(1)),
            _leg(right, at, "sell", near or Expiry.nth(0)),
        ),
        "calendar",
    )


def ratio_spread(
    right: str,
    long_steps: int = 0,
    short_steps: int = 1,
    long_ratio: int = 1,
    short_ratio: int = 2,
    *,
    expiry: Expiry | None = None,
) -> Structure:
    """Buy ``long_ratio`` lots ``long_steps`` out, sell ``short_ratio`` lots ``short_steps`` out.
    The default is a front (1x2) spread; ``long_ratio > short_ratio`` is a back spread."""
    if long_steps == short_steps:
        raise _invalid("a ratio spread needs two different strikes", "short_steps", short_steps)
    if long_ratio == short_ratio:
        raise _invalid("a ratio spread needs unequal ratios (use a vertical)", "short_ratio", short_ratio)
    s = _out(right)
    return Structure(
        (
            _leg(right, s * long_steps, "buy", expiry, long_ratio),
            _leg(right, s * short_steps, "sell", expiry, short_ratio),
        ),
        "ratio_spread",
    )


def risk_reversal(steps: int = 1, side: str = "buy", *, expiry: Expiry | None = None) -> Structure:
    """``side="buy"``: buy the call ``steps`` up, sell the put ``steps`` down."""
    _check_side(side)
    _pos(steps, "steps")
    o = _FLIP[side]
    return Structure(
        (_leg("call", steps, side, expiry), _leg("put", -steps, o, expiry)), "risk_reversal"
    )


def covered_call(call_steps: int = 0, *, expiry: Expiry | None = None) -> Structure:
    """Long the underlying, short the call ``call_steps`` up. Needs ``underlying_spec`` on resolve."""
    _pos(call_steps, "call_steps", 0)
    return Structure((UnderlyingLeg("buy"), _leg("call", call_steps, "sell", expiry)), "covered_call")


def protective_put(put_steps: int = 0, *, expiry: Expiry | None = None) -> Structure:
    """Long the underlying, long the put ``put_steps`` down."""
    _pos(put_steps, "put_steps", 0)
    return Structure((UnderlyingLeg("buy"), _leg("put", -put_steps, "buy", expiry)), "protective_put")


def collar(put_steps: int = 1, call_steps: int = 1, *, expiry: Expiry | None = None) -> Structure:
    """Long the underlying, long the put ``put_steps`` down, short the call ``call_steps`` up."""
    _pos(put_steps, "put_steps")
    _pos(call_steps, "call_steps")
    return Structure(
        (
            UnderlyingLeg("buy"),
            _leg("put", -put_steps, "buy", expiry),
            _leg("call", call_steps, "sell", expiry),
        ),
        "collar",
    )
