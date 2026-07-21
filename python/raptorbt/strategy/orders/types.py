"""Typed orders for the class-based order API.

Unlike the legacy intents (:class:`~raptorbt.strategy.orders.MarketOrder` /
:class:`~raptorbt.strategy.orders.ClosePosition`), these carry an explicit
side and time-in-force, may rest across bars (limit/stop kinds), and report
their lifecycle through the granular ``on_order_*`` hooks.

Sizing: pass exactly one of ``units`` (contract count) or ``size_frac``
(fraction of available capital, resolved at fill time). Omit both on a
closing-side order to close the full position. ``stop_price`` /
``target_price`` attach protective levels to the position an opening order
creates.

Semantics: market orders fill on the bar they are submitted, at the engine's
fill-price model — the same contract as ``enter()``. Resting orders begin
matching on the *next* bar; an order cannot rest into a bar that had already
closed when it was placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_SIDES = ("buy", "sell")
_VALID_TIFS = ("gtc", "day", "gtd", "ioc", "fok", "at_open", "at_close")
_VALID_OFFSET_KINDS = ("price", "bps", "ticks")


@dataclass(frozen=True, kw_only=True)
class _OrderBase:
    side: str
    units: float | None = None
    size_frac: float | None = None
    tif: str = "gtc"
    expire_ns: int | None = None
    stop_price: float | None = None
    target_price: float | None = None
    reduce_only: bool = False
    tags: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side must be one of {_VALID_SIDES}, got {self.side!r}")
        if self.tif not in _VALID_TIFS:
            raise ValueError(f"tif must be one of {_VALID_TIFS}, got {self.tif!r}")
        if self.tif == "gtd" and self.expire_ns is None:
            raise ValueError("tif='gtd' requires expire_ns")
        if self.tif in ("at_open", "at_close") and self.kind != "market":
            raise ValueError("at_open/at_close apply to market orders")
        if self.units is not None and self.size_frac is not None:
            raise ValueError("pass units or size_frac, not both")

    @property
    def kind(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Market(_OrderBase):
    """Fill at the engine's fill-price model on the submission bar."""

    @property
    def kind(self) -> str:
        return "market"


@dataclass(frozen=True, kw_only=True)
class Limit(_OrderBase):
    """Rest until the market trades at or through ``price``.

    ``post_only=True`` rejects the order instead of filling if it is already
    marketable at the open of the first bar it rests into.
    """

    price: float = 0.0
    post_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.price <= 0.0:
            raise ValueError("limit price must be > 0")

    @property
    def kind(self) -> str:
        return "limit"


@dataclass(frozen=True, kw_only=True)
class StopMarket(_OrderBase):
    """Become marketable once the market trades at or through ``trigger``."""

    trigger: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trigger <= 0.0:
            raise ValueError("stop trigger must be > 0")

    @property
    def kind(self) -> str:
        return "stop_market"


@dataclass(frozen=True, kw_only=True)
class StopLimit(_OrderBase):
    """Once ``trigger`` fires, rest as a limit at ``price`` from the next bar."""

    trigger: float = 0.0
    price: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trigger <= 0.0 or self.price <= 0.0:
            raise ValueError("stop_limit needs trigger > 0 and price > 0")

    @property
    def kind(self) -> str:
        return "stop_limit"


@dataclass(frozen=True, kw_only=True)
class MarketIfTouched(_OrderBase):
    """Fill once the market trades *favorably* to ``trigger`` (a buy fires
    when price falls to it) — the stop's mirror."""

    trigger: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trigger <= 0.0:
            raise ValueError("trigger must be > 0")

    @property
    def kind(self) -> str:
        return "market_if_touched"


@dataclass(frozen=True, kw_only=True)
class LimitIfTouched(_OrderBase):
    """Favorable touch at ``trigger``, then rest as a limit at ``price``."""

    trigger: float = 0.0
    price: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.trigger <= 0.0 or self.price <= 0.0:
            raise ValueError("limit_if_touched needs trigger > 0 and price > 0")

    @property
    def kind(self) -> str:
        return "limit_if_touched"


@dataclass(frozen=True, kw_only=True)
class MarketToLimit(_OrderBase):
    """Fill at the next bar's open (identical to an at-the-open market until
    partial fills exist)."""

    @property
    def kind(self) -> str:
        return "market_to_limit"


@dataclass(frozen=True, kw_only=True)
class TrailingStopMarket(_OrderBase):
    """Stop whose trigger trails the running favorable extreme by
    ``offset`` — ``offset_kind`` is ``"price"``, ``"bps"``, or ``"ticks"``
    (ticks need an instrument with a ``price_increment``)."""

    offset: float = 0.0
    offset_kind: str = "price"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.offset <= 0.0:
            raise ValueError("offset must be > 0")
        if self.offset_kind not in _VALID_OFFSET_KINDS:
            raise ValueError(f"offset_kind must be one of {_VALID_OFFSET_KINDS}")

    @property
    def kind(self) -> str:
        return "trailing_stop_market"


@dataclass(frozen=True, kw_only=True)
class TrailingStopLimit(_OrderBase):
    """Trailing stop that rests as a limit ``limit_offset`` through the
    trigger once it fires (bounded slippage)."""

    offset: float = 0.0
    offset_kind: str = "price"
    limit_offset: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.offset <= 0.0:
            raise ValueError("offset must be > 0")
        if self.offset_kind not in _VALID_OFFSET_KINDS:
            raise ValueError(f"offset_kind must be one of {_VALID_OFFSET_KINDS}")
        if self.limit_offset < 0.0:
            raise ValueError("limit_offset must be >= 0")

    @property
    def kind(self) -> str:
        return "trailing_stop_limit"
