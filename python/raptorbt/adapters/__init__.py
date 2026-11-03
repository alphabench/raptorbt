"""Broker adapters: the common layer, a conformance kit and thin mappings.

``BrokerAdapter`` is the base every broker mapping builds on
(``raptorbt.adapters.kite``, ``.upstox``, ``.dhan``). ``raptorbt.adapters.conformance``
tests any execution adapter, including your own. Guide:
``docs/architecture/broker-adapters.md``.
"""

from ._executor import Executor, InlineExecutor, ManualExecutor, ThreadExecutor
from ._ids import IdCodec
from ._limits import Backoff, RateLimiter, TokenBucket, reconnect_loop
from .base import (
    Ambiguous,
    AdapterEvent,
    AuthExpired,
    BrokerAdapter,
    Instrument,
    MalformedRow,
    OrderStatusUpdate,
    QueuedReport,
    Secret,
    Session,
    TradeRow,
    Transient,
    VenueBalances,
    VenueOrder,
    VenuePosition,
    VenueRefusal,
    new_namespace,
)

__all__ = [
    "AdapterEvent",
    "Ambiguous",
    "AuthExpired",
    "Backoff",
    "BrokerAdapter",
    "Executor",
    "IdCodec",
    "InlineExecutor",
    "Instrument",
    "MalformedRow",
    "ManualExecutor",
    "OrderStatusUpdate",
    "QueuedReport",
    "RateLimiter",
    "Secret",
    "Session",
    "ThreadExecutor",
    "TokenBucket",
    "TradeRow",
    "Transient",
    "VenueBalances",
    "VenueOrder",
    "VenuePosition",
    "VenueRefusal",
    "new_namespace",
    "reconnect_loop",
]
