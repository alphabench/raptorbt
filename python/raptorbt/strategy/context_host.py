"""What a strategy can ask its context about the instruments of a run or node.

``ctx.instrument(symbol_or_role=None)``, ``ctx.option_chain(underlying=None)`` and
``ctx.role(name)`` are the same in :func:`~raptorbt.run_strategies` and
:class:`~raptorbt.LiveNode`; both build one :class:`ContextHost` over the
``instruments=`` and ``bindings=`` they were given and attach it to every
strategy's context.

* ``ctx.instrument(key=None)`` returns the :class:`~raptorbt.InstrumentSpec` of a
  symbol, of a role bound to a symbol, or, with no key, of the symbol whose event
  is being handled. A symbol with no spec raises ``no_instrument_spec``.
* ``ctx.option_chain(underlying=None)`` returns an :class:`~raptorbt.OptionChain`
  of the option contracts among the run's instruments that are visible at the
  context's time: not yet listed (``activation_ns`` after it) or already expired
  contracts are left out, so a strategy cannot choose a contract the market had not
  listed. Before the first event the context has no time and the chain holds every
  option contract. Pass ``ctx.event_ns`` as ``now_ns`` to its queries.
* ``ctx.role(name)`` returns the symbol a :class:`~raptorbt.strategy.requirements.Requirements`
  role means now: the bound symbol of an ``Underlying``, the fixed one of an
  ``Instrument``, the contract of a ``Future`` and, for a ``Chain``, a tuple of
  contract symbols; for a ``Universe``, the tuple of symbols the role is bound to, in the
  order they were bound (for a :class:`~raptorbt.strategy.membership.PointInTime` binding, the
  members at the context's event time). Resolved at the context's time with
  :func:`~raptorbt.strategy.requirements.resolve` from the instruments and last
  prices the strategy can see. A role that cannot be resolved raises
  ``unresolved_role`` naming why. A ``Future`` (and a ``Chain`` on one) is the contract that is
  front *now*, so it is read in a hook that has an event (``on_bar``); before the first event
  the context has no time and the read is refused ``role_before_event``, not answered for some
  other instant.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Mapping, Sequence

from raptorbt import OptionChain
from raptorbt.errors import InvalidParameter, UnknownSymbol
from raptorbt.strategy.exits import _price_of
from raptorbt.strategy.membership import PointInTime
from raptorbt.strategy.requirements import (
    Chain,
    Future,
    Requirements,
    Universe,
    resolve,
    universe_binding_problem,
)


class ContextHost:
    """The instruments and role bindings behind ``ctx.instrument`` and friends.

    ``specs`` is shared, not copied: a node that adds or removes an instrument
    calls :meth:`changed` and every strategy sees the new set at its next call.
    """

    def __init__(
        self, specs: Mapping[str, object] | None, bindings: Mapping[str, str | Sequence[str]] | None
    ):
        self.specs: dict[str, object] = dict(specs or {})
        #: A role to its symbol; a ``Universe`` role to the tuple of its symbols (a ``PointInTime``
        #: keeps its steps).
        self.bindings: dict[str, str | tuple[str, ...]] = dict(bindings or {})
        self._version = 0
        #: The execution adapter of a node with a venue of its own, if any; a strategy's exit
        #: rules are offered to it (``place_exit_rule``).
        self.venue = None
        #: ``index_of(strategy)``: the strategy's index in a node that has a venue.
        self.index_of = None
        self._cache: tuple[list, list[int], list[int]] | None = None
        self._chains: dict[tuple, OptionChain] = {}
        #: The :class:`~raptorbt.LiveNode` this host serves, as a weak reference, if any; and the
        #: instruments strategies asked it for (see :meth:`request_add`), added at its next input.
        self.node = None
        self.pending: list[tuple[str, int]] = []

    # -- changes ---------------------------------------------------------------

    def changed(self) -> None:
        """The instrument set changed: drop what was derived from it."""
        self._version += 1
        self._cache = None
        self._chains.clear()

    def request_add(self, symbol: str, strategy) -> bool:
        """A strategy asks its node to add ``symbol``, which must be a listed contract the host
        gave a spec for. The node adds it (or brings it back, if it was retired) at its next input,
        for this strategy. Returns whether the request stands; ``False`` in a batch run (which
        keeps pre-registration) or for an unknown spec."""
        node = self.node() if self.node is not None else None
        if node is None or symbol not in self.specs:
            return False
        k = next((i for i, s in enumerate(node._strategies) if s is strategy), None)
        if k is None:
            return False
        if (symbol, k) not in self.pending:
            self.pending.append((symbol, k))
        node._requested()
        return True

    # -- binding to a strategy's context -----------------------------------------

    def attach(self, ctx, strategy) -> None:
        # ``ctx.instrument`` / ``option_chain`` / ``role`` are methods of the runtime's context
        # that call back into this host; the host and the strategy are all a context holds.
        ctx._raptorbt_host = self
        ctx._raptorbt_strategy = strategy

    # -- the three questions -----------------------------------------------------

    def instrument(self, ctx, key=None):
        if key is None:
            key = ctx.symbol
        symbol = self.bindings.get(key, key) if key not in self.specs else key
        if not isinstance(symbol, str):
            raise InvalidParameter(
                f"ctx.instrument({key!r}): the role is a Universe of {len(symbol)} symbols; "
                f"read them with ctx.role({key!r}) and pass one",
                code="universe_instrument",
                operation="ctx.instrument",
                parameter="symbol_or_role",
                value=key,
                valid=list(symbol),
            )
        spec = self.specs.get(symbol)
        if spec is None:
            raise UnknownSymbol(
                f"ctx.instrument: no instrument spec for {key!r}; give it in instruments=",
                code="no_instrument_spec",
                operation="ctx.instrument",
                parameter="symbol_or_role",
                value=key,
                valid=sorted(self.specs),
            )
        return spec

    def option_chain(self, ctx, underlying=None) -> OptionChain:
        now = ctx.event_ns
        options, activations, expiries = self._options()
        # The visible set changes only as `now` crosses a listing or an expiry.
        cut = (
            (len(options), 0)
            if now is None
            else (bisect_right(activations, now), bisect_right(expiries, now))
        )
        key = (self._version, underlying, cut)
        chain = self._chains.get(key)
        if chain is None:
            visible = [
                s
                for s in options
                if (underlying is None or s.underlying == underlying)
                and (
                    now is None
                    or (
                        (s.activation_ns is None or s.activation_ns <= now)
                        and (s.expiration_ns is None or s.expiration_ns > now)
                    )
                )
            ]
            if len(self._chains) > 64:
                self._chains.clear()
            chain = self._chains[key] = OptionChain(visible)
        return chain

    def futures_of(self, underlying: str) -> list:
        """The futures contracts listed on ``underlying``, in no particular order."""
        return [
            s for s in self.specs.values()
            if s.kind == "contract" and s.expiration_ns is not None and s.underlying == underlying
        ]

    def role(self, ctx, name: str):
        strategy = ctx._raptorbt_strategy
        requirements = getattr(type(strategy), "requirements", None)
        if not isinstance(requirements, Requirements):
            raise InvalidParameter(
                f"ctx.role({name!r}): {type(strategy).__name__} declares no requirements",
                code="no_requirements",
                operation="ctx.role",
                parameter="name",
                value=name,
            )
        if name not in requirements.roles:
            raise UnknownSymbol(
                f"ctx.role: {type(strategy).__name__} declares no role {name!r}",
                code="unknown_role",
                operation="ctx.role",
                parameter="name",
                value=name,
                valid=list(requirements.roles),
            )
        now = ctx.event_ns
        if now is None and _follows_front(requirements.roles, name):
            raise InvalidParameter(
                f"ctx.role({name!r}) names the futures contract that is front at the event's time, "
                "and before the first event (on_init, on_start) the context has no time: read "
                "ctx.role in a hook that has an event (on_bar), not in on_start",
                code="role_before_event",
                operation="ctx.role",
                parameter="name",
                value=name,
            )
        has_chain = any(isinstance(r, Chain) for r in requirements.roles.values())
        plan = resolve(
            requirements,
            now_ns=0 if now is None else now,
            bindings=self.bindings,
            specs=list(self.specs.values()),
            prices=self._prices(ctx),
            chain=self.option_chain(ctx) if has_chain else None,
        )
        role = plan.roles.get(name)
        problems = [i for i in plan.issues if i.path == f"roles.{name}"] if role is not None else list(plan.issues)
        if role is None or problems or not role.symbols:
            why = "; ".join(f"{i.code}: {i.message}" for i in (problems or plan.issues)) or "no symbols"
            raise UnknownSymbol(
                f"ctx.role({name!r}) cannot be resolved ({why})",
                code="unresolved_role",
                operation="ctx.role",
                parameter="name",
                value=name,
            )
        declared = requirements.roles[name]
        bound = self.bindings.get(name)
        if isinstance(declared, Universe) and isinstance(bound, PointInTime):
            return bound.at(now)  # the members now: a name is not offered before it joined
        return tuple(role.symbols) if isinstance(declared, (Chain, Universe)) else role.symbols[0]

    # -- internals -----------------------------------------------------------------

    def _options(self):
        """The option specs by symbol, and the sorted listing and expiry times."""
        if self._cache is None:
            options = sorted(
                (s for s in self.specs.values() if s.kind == "option"), key=lambda s: s.symbol
            )
            never = -(2**63)
            activations = sorted(
                never if s.activation_ns is None else s.activation_ns for s in options
            )
            expiries = sorted(s.expiration_ns for s in options if s.expiration_ns is not None)
            self._cache = (options, activations, expiries)
        return self._cache

    def _prices(self, ctx) -> dict[str, float]:
        """The last price of each symbol, as exit rules read it (the last close, else the last
        traded ``ltp``). A symbol with none (no data, or only quote-only rows in view) is left
        out, so a role that needs its price says ``needs_price`` instead of resolving at 0."""
        out: dict[str, float] = {}
        for symbol in ctx.symbols:
            price = _price_of(ctx, symbol)
            if price is not None:
                out[symbol] = price
        return out


def _follows_front(roles: Mapping[str, object], name: str) -> bool:
    """Whether role ``name`` is a ``Future`` or a ``Chain`` on one: the roles whose symbol depends on
    the time of the read."""
    role = roles.get(name)
    return isinstance(role, Future) or (isinstance(role, Chain) and _follows_front(roles, role.of))


def build(instruments, bindings, entries, universe, call: str) -> ContextHost:
    """The host of a run or node, with ``bindings`` checked.

    Every key must be a role of at least one strategy; a ``Universe`` role is bound to a list of
    symbols (``universe_size`` if the count is outside the role's min..max) and any other role to
    one symbol (``invalid_binding`` for the wrong shape); every bound symbol must be a symbol of
    the run (``universe``).
    """
    bindings = dict(bindings or {})
    declared: dict[str, list] = {}
    for strategy, _ in entries:
        requirements = getattr(type(strategy), "requirements", None)
        if isinstance(requirements, Requirements):
            for name, role in requirements.roles.items():
                declared.setdefault(name, []).append(role)
    for role, bound in bindings.items():
        if role not in declared:
            raise InvalidParameter(
                f"{call}: bindings names role {role!r}, which no strategy declares",
                code="unknown_role_binding",
                operation=call,
                parameter="bindings",
                value=role,
                valid=sorted(declared),
            )
        symbols = [bound]
        for kind in declared[role]:
            if isinstance(kind, Universe):
                problem = universe_binding_problem(kind, bound)
                if problem is not None:
                    shown = list(bound) if not isinstance(bound, str) and isinstance(bound, Sequence) else bound
                    message = f"{call}: role {role!r}: {problem[1]}"
                    if problem[0] == "universe_size":
                        raise InvalidParameter(
                            message, code="universe_size", operation=call, parameter="bindings", value=shown
                        )
                    raise InvalidParameter(
                        message, code="invalid_binding", operation=call, parameter="bindings", value=shown
                    )
                symbols = list(bound)
                bindings[role] = bound if isinstance(bound, PointInTime) else tuple(bound)
            elif not isinstance(bound, str):
                raise InvalidParameter(
                    f"{call}: role {role!r} takes one symbol, got {bound!r}",
                    code="invalid_binding",
                    operation=call,
                    parameter="bindings",
                    value=bound if isinstance(bound, (str, int, float)) else repr(bound),
                )
        for symbol in symbols:
            if symbol not in universe:
                raise UnknownSymbol(
                    f"{call}: role {role!r} is bound to {symbol!r}, which is not a symbol here",
                    code="unknown_symbol",
                    operation=call,
                    parameter="bindings",
                    value=symbol,
                    valid=list(universe),
                )
    return ContextHost(instruments, bindings)
