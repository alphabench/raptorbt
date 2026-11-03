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
  contract symbols, resolved at the context's time with
  :func:`~raptorbt.strategy.requirements.resolve` from the instruments and last
  prices the strategy can see. A role that cannot be resolved raises
  ``unresolved_role`` naming why.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Mapping

from raptorbt import OptionChain
from raptorbt.errors import InvalidParameter, UnknownSymbol
from raptorbt.strategy.requirements import Chain, Requirements, resolve


class ContextHost:
    """The instruments and role bindings behind ``ctx.instrument`` and friends.

    ``specs`` is shared, not copied: a node that adds or removes an instrument
    calls :meth:`changed` and every strategy sees the new set at its next call.
    """

    def __init__(self, specs: Mapping[str, object] | None, bindings: Mapping[str, str] | None):
        self.specs: dict[str, object] = dict(specs or {})
        self.bindings: dict[str, str] = dict(bindings or {})
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
        return tuple(role.symbols) if isinstance(requirements.roles[name], Chain) else role.symbols[0]

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
        out: dict[str, float] = {}
        for symbol in ctx.symbols:
            try:
                series = ctx.series(symbol)
            except (KeyError, RuntimeError):
                continue
            column = series.get("close") if len(series.get("close", ())) else series.get("ltp")
            if column is not None and len(column):
                out[symbol] = float(column[-1])
        return out


def build(instruments, bindings, entries, universe, call: str) -> ContextHost:
    """The host of a run or node, with ``bindings`` checked.

    Every key must be a role of at least one strategy, and every bound symbol a
    symbol of the run (``universe``).
    """
    bindings = dict(bindings or {})
    declared: set[str] = set()
    for strategy, _ in entries:
        requirements = getattr(type(strategy), "requirements", None)
        if isinstance(requirements, Requirements):
            declared.update(requirements.roles)
    for role, symbol in bindings.items():
        if role not in declared:
            raise InvalidParameter(
                f"{call}: bindings names role {role!r}, which no strategy declares",
                code="unknown_role_binding",
                operation=call,
                parameter="bindings",
                value=role,
                valid=sorted(declared),
            )
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
