"""Where a strategy's orders go.

A strategy only queues orders. What happens to them is the *execution* of the
node that runs it:

* **In-process** (the default, and :class:`SimulatedExecution`): the engine's
  own matcher fills orders against the data it is given. This is what a
  backtest does, and what a :class:`~raptorbt.LiveNode` does unless told
  otherwise.
* **Paper** (:class:`PaperExecution`): a simulated exchange. Orders are sent to
  it as requests, and it answers with the same :class:`ExecutionReport` objects
  a broker would, as the market data pushed to the node moves. It matches,
  prices and charges with the same kernel as a backtest, so a strategy gets
  the fills it would get in a backtest of the same stream, by the route a real
  venue's fills take.
* **A venue of your own:** any object with ``submit``, ``cancel`` and
  ``modify`` (the :class:`ExecutionAdapter` protocol). The node sends
  requests; the venue's answers come back only through
  :meth:`LiveNode.push_report <raptorbt.LiveNode.push_report>`.

The same :class:`~raptorbt.Strategy` class runs under all three, unchanged.

Requests are fire and forget
----------------------------

``submit``, ``cancel`` and ``modify`` return when the request has been
handed over, never when the venue has acted on it. Returning normally says
"sent". What became of the request arrives later as reports, which may repeat,
come late or come out of order: the node applies them so that none of that
changes the outcome (see ``docs/architecture/order-lifecycle.md``).

Raise :class:`OrderRefused` from a method when the venue will certainly not
take the request (the order is then rejected). Raise :class:`Unsupported` when
the venue has no equivalent of it. Any other exception means "this may or may
not have reached the venue": it propagates to the caller unchanged, the order
stays as it was, and the venue's reports (or a reconciliation) settle it.

An adapter must not call back into the node from inside these methods.
Collect what the venue says and push it after the call that caused it
returns.

Identity
--------

Every order sent has a ``client_order_id`` of the form ``s{strategy}-{n}``: the
strategy's index in the node and a count of the orders that strategy has sent.
It depends on nothing else, so a replay of the same decisions sends the same
ids, and an id is never sent twice.

Everything a backtest accepts runs on a venue
---------------------------------------------

A strategy that passes its backtest runs on a venue: each form is turned into
real orders. A size given as ``size_frac`` or not at all is resolved to units
when the order is sent, from the book on that event. ``enter()`` becomes a
market order and ``close_position()`` a reduce-only market order. Stops and
targets (on an order, an entry, or the position's configuration) become
protective orders linked one-cancels-other; ``LiveNode(protective="venue")``
has the venue hold them, ``"local"`` has the engine watch the levels and send
a market order. The session square-off, ``Twap`` slices and ``halt`` send
their orders too. The rules and the exact limits of parity are in
``docs/architecture/execution-adapters.md``.

Refused up front: shared account groups, ``liquidate_on_margin_call`` and
(as on any live node) ATR-based stops. Corrections and busts of an execution
that already moved a position are refused. An order made on warmup history is
never sent to a real venue.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from raptorbt._raptorbt import ExecutionReport, OrderRequest

__all__ = [
    "AdapterError",
    "ExecutionAdapter",
    "ExecutionReport",
    "OrderRefused",
    "OrderRequest",
    "PaperExecution",
    "SimulatedExecution",
    "Unsupported",
]


class AdapterError(Exception):
    """Base class of the errors an adapter raises on purpose."""


class OrderRefused(AdapterError):
    """The venue will certainly not take the request.

    Raised from ``submit``, the order is rejected with the message as the
    reason. Raised from ``cancel`` or ``modify``, the request is rejected and
    the order keeps its state.
    """


class Unsupported(AdapterError):
    """The venue has no equivalent of the request (an order type, a link)."""


@runtime_checkable
class ExecutionAdapter(Protocol):
    """A venue the engine can send orders to."""

    def submit(self, request: OrderRequest) -> None:
        """Send a new order. See the module docs for what returning means."""

    def cancel(self, client_order_id: str) -> None:
        """Ask for a working order to be cancelled."""

    def modify(
        self,
        client_order_id: str,
        *,
        quantity: float | None = None,
        limit_price: float | None = None,
        trigger_price: float | None = None,
    ) -> None:
        """Ask for a working order to change. ``quantity`` is the new total."""


class SimulatedExecution:
    """Match orders in-process, as a backtest does. The default."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "SimulatedExecution()"


class PaperExecution:
    """A simulated exchange that answers with execution reports.

    Takes no arguments: it follows the node's own configuration (fees,
    slippage, fill timing, latency, instruments, account), which is what makes
    its fills those of a backtest on the same stream.

    A strategy run on it must move positions with orders only: position-level
    stops and targets and the session square-off are refused when the node is
    built.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "PaperExecution()"
