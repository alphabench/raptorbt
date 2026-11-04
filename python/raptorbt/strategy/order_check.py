"""The host's pre-submit order check.

An embedding platform has its own rules about which orders may go out (lot
sizes, price bands, term restrictions). ``order_check=`` on
:func:`~raptorbt.run_strategies` and :class:`~raptorbt.LiveNode` installs one
callable that every typed order a strategy submits passes through before the
engine or a venue sees it, so the same rule refuses the same orders in a
backtest and live::

    def check(order: OrderProposal, ctx) -> None | str | OrderRefusal:
        if order.symbol.endswith("-OPT") and order.side == "sell" and not order.reduce_only:
            return "naked_short_not_allowed"

``None`` accepts. A ``str`` (or an :class:`OrderRefusal`) refuses: the order
is recorded as rejected with that code, never works, never reaches the venue,
and the strategy hears it in ``on_order_rejected`` / ``on_order_event`` with
``event.reject_reason`` equal to the code. A refusal counts as a rejection in the
run's risk counters. The check runs before the engine's own checks, so its
code is the one the strategy hears.

The check must be a pure function of the proposal and the context: it must
not submit or cancel orders. It runs once per order, on the event that
produced the order. An exception raised by the check propagates and stops the
run; it is never read as an acceptance. A code is 1 to 64 bytes and the
distinct codes of a process are bounded (1024), so codes must be a small
stable set, not carry ids.

Only typed orders (``submit_order``) are checked. The legacy ``enter()`` and
``close_position()`` intents carry no order to show the check, so with an
``order_check`` installed they are refused (``order_check_intent``).

``order_check_scope="typed"`` declares that the check judges typed orders only: intents are then
neither refused nor shown to it, and they pass as they would without a check. Use it when the
check judges forms an intent cannot have (an order kind, a validity, ``post_only``); the intents
are then outside what the check can say. Leave the default ``"all"`` when the check must see
every order the strategy sends: an intent is refused rather than let past it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from raptorbt.errors import InvalidParameter


@dataclass(frozen=True, kw_only=True)
class OrderProposal:
    """An order about to be submitted, as the check sees it."""

    symbol: str
    #: The submitting strategy's index (its position in the run or node).
    strategy_index: int
    client_id: str
    side: str
    kind: str
    #: ``None`` for an order sized as a fraction of capital or of a position.
    units: float | None
    size_frac: float | None
    limit_price: float | None
    trigger_price: float | None
    tif: str
    reduce_only: bool
    post_only: bool
    #: The order's own tags (a copy).
    tags: dict = field(default_factory=dict)
    #: Event time of the decision (ns).
    ts_ns: int


@dataclass(frozen=True)
class OrderRefusal:
    """A refusal with the host's code."""

    code: str


OrderCheck = Callable[[OrderProposal, object], "None | str | OrderRefusal"]


SCOPES = ("all", "typed")


def install(rt, order_check, operation: str, scope: str = "all") -> None:
    """Install ``order_check`` on a runtime, or do nothing for ``None``. ``scope`` is
    ``"all"`` (intents are refused) or ``"typed"`` (the check judges typed orders only)."""
    if scope not in SCOPES:
        raise InvalidParameter(
            f"order_check_scope must be 'all' or 'typed', got {scope!r}",
            code="invalid_order_check_scope",
            operation=operation,
            parameter="order_check_scope",
            value=scope,
            valid=list(SCOPES),
        )
    if order_check is None:
        return
    if not callable(order_check):
        raise InvalidParameter(
            f"order_check must be a callable (order, ctx) -> None | str | OrderRefusal, "
            f"got {type(order_check).__name__}",
            code="invalid_order_check",
            operation=operation,
            parameter="order_check",
            value=type(order_check).__name__,
        )
    rt.set_order_check(order_check, OrderProposal, scope == "typed")
