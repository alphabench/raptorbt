"""Continuous futures: a role that rolls from one contract to the next at a declared rule.

A strategy declares the rule in its requirements::

    Future(of="index", contract="continuous", roll_days=3)   # roll 3 days before expiry

``ctx.role(name)`` then names the contract that is front now, and changes to the next one when
the rule says so. Two things build on it:

* :class:`RollEvents`, a strategy mixin that tells the strategy when a role rolls
  (``on_roll(ctx, RollEvent)``) and, if ``auto_roll = True``, moves the strategy's positions
  with real orders: a reduce-only market order closing each position in the old contract and,
  **once that close has filled**, a market order opening the quantity it closed, in the same
  direction, in the new one; ids ``{role}:roll{n}:c{k}`` / ``:o{k}``, tags
  ``{"roll": role, "from": old, "to": new}``. A venue may refuse the close (a reduce-only order
  with nothing to reduce, a halted strategy): the open is then never sent, so the exposure is
  never doubled, and ``on_roll_failed`` is called (it raises unless you override it). Without
  ``auto_roll`` the strategy rolls its own positions, in ``on_roll``. If the open itself is
  rejected, cancelled or expired, ``on_roll_failed`` is called for it too (the position is then
  closed in the old contract and open in neither). The open keeps the **notional exposure**
  (``roll_sizing = "notional"``, the default): ``N_new = floor(N_old * L_old * P_old / (L_new *
  P_new))`` lots, where ``N`` is lots, ``L`` the lot size and ``P`` the price (times the
  contract multiplier), the old price being the close's fill and the new one the new contract's
  last close. The open's tags carry the lots, the notionals and the residual exposure the rounding
  leaves, and ``on_roll_sized(ctx, roll, sizing)`` hears the same. ``roll_sizing = "lots"`` keeps the
  lot count instead (the quantity opened is the lots closed times the new contract's lot). A roll
  that sizes to no lot, or whose new contract has no price yet, is a failed roll
  (``on_roll_failed``), never a silent fallback.
  The roll's progress (the
  contract each role names, the roll counter and the opens still waiting for their close) is part
  of the strategy's ``state()``, so a restored node resumes it.
* :func:`continuous_series`, which builds one price history from the contracts' bars under an
  explicit adjustment policy, for indicators and research.

Both contracts must be among the symbols of a batch run: this module switches which one a
role names, it does not subscribe a contract that was not given. A :class:`~raptorbt.LiveNode`
can add the next contract itself: when an ``auto_roll`` rolls to a contract the node does not
trade but whose ``InstrumentSpec`` it was given, the node adds it at its next input (journaled
like :meth:`~raptorbt.LiveNode.add_instrument`) and the strategy's positions move on the first
event after it is there. A host that wants the contract's data before the roll adds it earlier
with :func:`~raptorbt.strategy.requirements.plan_instruments` (``roll_lead_days``).
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Sequence

import numpy as np

from raptorbt.errors import InvalidData, InvalidParameter, InvalidSubscription, StateError, UnknownSymbol
from raptorbt.strategy.orders import Market

_DAY = 86_400 * 10**9


class RollEvent(NamedTuple):
    """A continuous role moved from one contract to the next."""

    role: str
    from_symbol: str
    to_symbol: str
    ts_ns: int


def front(futures: Sequence, roll_days: float, now_ns: int):
    """``(symbol, roll_ns)``: the front contract of ``futures`` (specs of one underlying) at
    ``now_ns`` under a roll ``roll_days`` before expiry, and the time it stops being front
    (``None`` for the last contract). ``(None, None)`` before any is listed or after all rolled."""
    cutoff = int(roll_days * _DAY)
    ordered = sorted(
        (s for s in futures if s.expiration_ns is not None), key=lambda s: s.expiration_ns
    )
    for s in ordered:
        if s.expiration_ns - cutoff > now_ns and (s.activation_ns is None or s.activation_ns <= now_ns):
            return s.symbol, s.expiration_ns - cutoff
    return None, None


class RollEvents:
    """Mix into a strategy (before ``Strategy``) whose requirements hold a continuous
    :class:`~raptorbt.strategy.requirements.Future`::

        class Carry(RollEvents, Strategy):
            auto_roll = True
            def on_roll(self, ctx, roll): ...

    The roll is noticed after the strategy's own ``on_bar`` / ``on_trade`` of the first event at
    or after the rule's time. A role's first event only records its contract: no roll."""

    #: Move the strategy's positions to the new contract with orders when a role rolls.
    auto_roll = False
    roll_sizing = "notional"

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for hook in ("on_bar", "on_trade"):
            own = cls.__dict__.get(hook)
            if own is not None and not getattr(own, "_raptorbt_rolls", False):
                setattr(cls, hook, _then_roll(own))
        own = cls.__dict__.get("on_order_event")
        if own is not None and not getattr(own, "_raptorbt_rolls", False):
            setattr(cls, "on_order_event", _then_orders(own))
        own_state, own_load = cls.__dict__.get("state"), cls.__dict__.get("load_state")
        if (own_state or own_load) and not any(getattr(f, "_raptorbt_rolls", False) for f in (own_state, own_load) if f):
            def inherited(name, default):
                for c in cls.__mro__[1:]:
                    if c is not RollEvents and name in c.__dict__:
                        return c.__dict__[name]
                return default

            saved, loaded = _with_roll_state(
                own_state or inherited("state", lambda self: None),
                own_load or inherited("load_state", lambda self, state: None),
            )
            setattr(cls, "state", saved)
            setattr(cls, "load_state", loaded)

    def on_roll(self, ctx, roll: RollEvent) -> None:
        """A continuous role moved to another contract."""

    def on_roll_sized(self, ctx, roll: RollEvent, sizing: dict) -> None:
        """The open of a roll was sized. ``sizing`` has ``policy``, ``lots_old``, ``lots_new``,
        ``notional_old``, ``notional_new`` and ``residual`` (the notional the rounding leaves
        out of the new contract: old minus new, so positive is under-exposed)."""

    def on_roll_failed(self, ctx, roll: RollEvent, close_id: str, status: str, reason) -> None:
        """A step of the roll ended without filling (``status`` is ``rejected``, ``canceled`` or
        ``expired``). ``close_id`` names the order: a close of a position in the old contract
        (the open in the new contract was not sent, and the position is still in the old one),
        or, when its id ends in ``:o{k}``, the open in the new contract (the position is closed
        and not open in the new contract). Raises by default."""
        if close_id.rpartition(":")[2].startswith("o"):  # ids are {role}:roll{n}:c{k} and :o{k}[.part]
            raise StateError(
                f"auto_roll: opening {close_id!r} in {roll.to_symbol!r} ended {status} ({reason}); "
                f"the position was closed in {roll.from_symbol!r} and is open in neither contract",
                code="roll_open_failed",
                operation="auto_roll",
            )
        raise StateError(
            f"auto_roll: closing {close_id!r} in {roll.from_symbol!r} ended {status} ({reason}); "
            f"the open in {roll.to_symbol!r} was not sent and the position is still in the old contract",
            code="roll_close_failed",
            operation="auto_roll",
        )

    def on_bar(self, ctx, bar):
        self._check_rolls(ctx)

    on_bar._raptorbt_rolls = True  # type: ignore[attr-defined]

    def on_trade(self, ctx, tick):
        self._check_rolls(ctx)

    on_trade._raptorbt_rolls = True  # type: ignore[attr-defined]

    def on_order_event(self, ctx, event):
        self._roll_orders(ctx, event)

    on_order_event._raptorbt_rolls = True  # type: ignore[attr-defined]

    def state(self):
        return self._roll_snapshot() or None

    def load_state(self, state):
        self._roll_load(state)

    # -- internals ------------------------------------------------------------------

    def _roll_state(self):
        state = self.__dict__.get("_rolls")
        if state is None or state[0] is not self.cache:
            seed = self.__dict__.pop("_rolls_loaded", None) or {}
            seen = {k: (v[0], int(v[1])) for k, v in seed.get("seen", {}).items()}
            pending = {k: dict(v) for k, v in seed.get("pending", {}).items()}
            # (cache, role -> (contract, roll time), roll counter, pending rolls, rolls waiting for a contract)
            state = self.__dict__["_rolls"] = (self.cache, seen, [int(seed.get("seq", 0))], pending, [])
        return state

    def _roll_snapshot(self) -> dict:
        """The roll's progress as plain data (see ``state()``)."""
        _, seen, seq, pending, _ = self._roll_state()
        if not seen and not seq[0] and not pending:
            return {}
        return {
            "seen": {k: [v[0], v[1]] for k, v in sorted(seen.items())},
            "seq": seq[0],
            "pending": {k: dict(v) for k, v in sorted(pending.items())},
        }

    def _roll_load(self, snapshot) -> None:
        self.__dict__["_rolls_loaded"] = dict(snapshot or {})
        self.__dict__.pop("_rolls", None)

    def _check_rolls(self, ctx) -> None:
        from raptorbt.strategy.requirements import Future, Requirements

        requirements = getattr(type(self), "requirements", None)
        now = ctx.event_ns
        if not isinstance(requirements, Requirements) or now is None:
            return
        _, seen, seq, _, deferred = self._roll_state()
        host = ctx._raptorbt_host
        if deferred:
            for event in [e for e in deferred if e.to_symbol in ctx.symbols]:
                deferred.remove(event)
                self._move_positions(ctx, event, seq, deferred)
        for name, role in requirements.roles.items():
            if not (isinstance(role, Future) and role.contract == "continuous"):
                continue
            known = seen.get(name)
            if known is not None and now < known[1]:
                continue  # before the rule's time: nothing to resolve
            underlying = ctx.role(role.of)
            symbol, roll_ns = front(host.futures_of(underlying), role.roll_days, now)
            if symbol is None:
                continue
            seen[name] = (symbol, roll_ns if roll_ns is not None else 2**63 - 1)
            if known is None or known[0] == symbol:
                continue
            event = RollEvent(name, known[0], symbol, now)
            self.on_roll(ctx, event)
            if self.auto_roll:
                self._move_positions(ctx, event, seq, deferred)

    def _move_positions(self, ctx, event: RollEvent, seq: list[int], deferred: list) -> None:
        if event.to_symbol not in ctx.symbols:
            if ctx._raptorbt_host.request_add(event.to_symbol, self):
                deferred.append(event)  # the node adds it; the positions move once it is there
                return
            raise InvalidSubscription(
                f"{event.role} rolled to {event.to_symbol!r}, which is not a symbol of this run "
                "(a LiveNode adds it when it was given the contract's InstrumentSpec)",
                code="roll_target_not_subscribed",
                operation="auto_roll",
                parameter="symbols",
                value=event.to_symbol,
                valid=list(ctx.symbols),
            )
        _, _, _, pending, _ = self._roll_state()
        n = seq[0]
        seq[0] += 1
        tags = {"roll": event.role, "from": event.from_symbol, "to": event.to_symbol}
        for k, p in enumerate(ctx.positions(event.from_symbol)):
            closing, opening = ("sell", "buy") if p.direction > 0 else ("buy", "sell")
            close_id = f"{event.role}:roll{n}:c{k}"
            # The open waits for the close to fill: a venue that refuses the close
            # (reduce-only with nothing to reduce, a halted strategy) must not be
            # left holding both contracts.
            pending[close_id] = {
                "open_id": f"{event.role}:roll{n}:o{k}",
                "side": opening,
                "role": event.role,
                "from": event.from_symbol,
                "to": event.to_symbol,
                "ts": event.ts_ns,
                "left": float(p.size),
                "from_lot": self._lot(ctx, event.from_symbol),
                "to_lot": self._lot(ctx, event.to_symbol),
            }
            self.submit_order(
                Market(side=closing, quantity=p.size, instrument=event.from_symbol, reduce_only=True, tags=tags),
                client_id=close_id,
            )

    def _notional_sizing(self, ctx, job, event, qty, lots_old, from_lot, to_lot):
        """``N_new = floor(N_old * L_old * P_old / (L_new * P_new))`` and what it leaves; ``None``
        when it is no lot. The old price is the close's fill, the new one the new contract's last
        close (a contract with no bar yet cannot be priced: the roll fails loudly)."""
        p_new = self._price_known_at_fill(ctx, job["to"])
        if p_new is None:
            raise StateError(
                f"auto_roll: {job['to']!r} has no price yet, so the roll cannot keep the exposure; "
                "feed it a bar before the roll, or set roll_sizing = 'lots'",
                code="roll_price_missing",
                operation="auto_roll",
                parameter="to",
                value=job["to"],
            )
        old_mult, new_mult = self._multiplier(ctx, job["from"]), self._multiplier(ctx, job["to"])
        p_old = float(event.price)
        notional_old = qty * p_old * old_mult
        unit_new = to_lot * p_new * new_mult
        lots_new = float(math.floor(notional_old / unit_new + 1e-9))
        if lots_new < 1:
            return None
        notional_new = lots_new * unit_new
        return {
            "price_new": p_new,
            "policy": "notional",
            "lots_old": lots_old,
            "lots_new": lots_new,
            "notional_old": notional_old,
            "notional_new": notional_new,
            "residual": notional_old - notional_new,
        }

    @staticmethod
    def _price_known_at_fill(ctx, symbol: str) -> float | None:
        """The price of ``symbol`` a roll's open can be sized on when the close fills: the close of
        the bar the fill came in under ``same_bar_close``, the last close before that bar under
        ``next_bar_open`` (the open fills at the bar's open, before its close exists)."""
        if not ctx.fills_at_next_open:
            last = ctx.data.latest(symbol)
            return None if last is None else float(last.close)
        recent = ctx.data.history(symbol, count=2)
        stamp = ctx.timestamp
        for i in range(len(recent.timestamps) - 1, -1, -1):
            if recent.timestamps[i] < stamp:
                return float(recent.close[i])
        return None

    @staticmethod
    def _multiplier(ctx, symbol: str) -> float:
        try:
            return float(ctx.instrument(symbol).multiplier or 1.0)
        except UnknownSymbol as e:
            if e.code != "no_instrument_spec":
                raise
            return 1.0

    @staticmethod
    def _lot(ctx, symbol: str) -> float:
        """The instrument spec's lot size; an instrument with no spec trades in units."""
        try:
            return float(ctx.instrument(symbol).lot_size)
        except UnknownSymbol as e:
            if e.code != "no_instrument_spec":
                raise
            return 1.0

    def _roll_orders(self, ctx, event) -> None:
        """A close of a roll filled (send its open) or failed (say so); an open of a roll that
        is rejected, cancelled or expired is a failed roll too."""
        _, _, _, pending, _ = self._roll_state()
        cid = getattr(event, "client_order_id", None)
        job = pending.get(cid)
        if job is None:
            return
        kind = event.kind
        roll = RollEvent(job["role"], job["from"], job["to"], job["ts"])
        if job.get("open"):
            # The open is the last step: the job stays until the venue has taken it. A sized open
            # stays until it has filled, to report its exposure at the price it filled at.
            sizing = job.get("sizing")
            if sizing is not None:
                if kind == "order_filled":
                    job["filled"] = job.get("filled", 0.0) + float(event.size)
                    job["value"] = job.get("value", 0.0) + float(event.size) * float(event.price)
                    if job["filled"] + 1e-9 >= job["qty"]:
                        del pending[cid]
                        notional_new = job["value"] * job["mult"]
                        self.on_roll_sized(
                            ctx,
                            roll,
                            dict(
                                sizing,
                                price_new=job["value"] / job["filled"],
                                notional_new=notional_new,
                                residual=sizing["notional_old"] - notional_new,
                            ),
                        )
                elif kind in ("order_rejected", "order_canceled", "order_expired"):
                    del pending[cid]
                    self.on_roll_failed(ctx, roll, cid, kind.removeprefix("order_"), getattr(event, "reject_reason", None))
            elif kind in ("order_accepted", "order_filled"):
                del pending[cid]
            elif kind in ("order_rejected", "order_canceled", "order_expired"):
                del pending[cid]
                self.on_roll_failed(ctx, roll, cid, kind.removeprefix("order_"), getattr(event, "reject_reason", None))
            return
        if kind == "order_filled":
            qty = min(float(event.size), job["left"])
            job["left"] -= qty
            part = job.get("part", 0)
            job["part"] = part + 1
            if job["left"] <= 1e-9 * max(1.0, qty):
                del pending[cid]
            from_lot, to_lot = job.get("from_lot", 1.0), job.get("to_lot", 1.0)
            # the open answers the close's fill, so it meets the bar the close filled in
            tags = {"roll": job["role"], "from": job["from"], "to": job["to"]}
            open_qty = qty
            policy = self.roll_sizing
            if policy not in ("notional", "lots"):
                raise InvalidParameter(
                    f"roll_sizing is 'notional' or 'lots', got {policy!r}",
                    code="invalid_roll_sizing",
                    operation="auto_roll",
                    parameter="roll_sizing",
                    value=policy,
                    valid=["notional", "lots"],
                )
            lots_old = qty / from_lot
            if policy == "lots":
                # The same number of lots in the new contract's lot.
                if from_lot != to_lot:
                    open_qty = lots_old * to_lot
                    tags["lots"] = f"{lots_old:g}"
                    tags["lot_size"] = f"{from_lot:g}->{to_lot:g}"
            else:
                sizing = self._notional_sizing(ctx, job, event, qty, lots_old, from_lot, to_lot)
                if sizing is None:
                    del pending[cid]
                    self.on_roll_failed(ctx, roll, cid, "zero_size", "the new contract's notional is below one lot of it")
                    return
                open_qty = sizing["lots_new"] * to_lot
                tags.update(
                    roll_sizing="notional",
                    lots=f"{lots_old:g}->{sizing['lots_new']:g}",
                    notional_old=f"{sizing['notional_old']:.2f}",
                    price_new=f"{sizing['price_new']:.2f}",
                    notional_new=f"{sizing['notional_new']:.2f}",
                    residual=f"{sizing['residual']:.2f}",
                )
            # One open per fill of the close, for the quantity that fill closed.
            open_id = job["open_id"] if part == 0 else f"{job['open_id']}.{part}"
            pending[open_id] = {
                "open": True, "role": job["role"], "from": job["from"], "to": job["to"], "ts": job["ts"],
            }  # fmt: skip
            if policy == "notional":
                # Reported on ``on_roll_sized`` once the open has filled, at the price it filled at.
                pending[open_id].update(sizing=sizing, qty=open_qty, mult=self._multiplier(ctx, job["to"]))
            self.submit_order(
                Market(side=job["side"], quantity=open_qty, instrument=job["to"], tags=tags),
                client_id=open_id,
            )
        elif kind in ("order_rejected", "order_canceled", "order_expired"):
            del pending[cid]
            self.on_roll_failed(ctx, roll, cid, kind.removeprefix("order_"), getattr(event, "reject_reason", None))


def _then_roll(hook):
    import functools

    @functools.wraps(hook)
    def wrapped(self, ctx, *args):
        hook(self, ctx, *args)
        self._check_rolls(ctx)

    wrapped._raptorbt_rolls = True  # type: ignore[attr-defined]
    return wrapped


def _then_orders(hook):
    import functools

    @functools.wraps(hook)
    def wrapped(self, ctx, *args):
        hook(self, ctx, *args)
        self._roll_orders(ctx, *args)

    wrapped._raptorbt_rolls = True  # type: ignore[attr-defined]
    return wrapped


def _with_roll_state(state, load_state):
    """Wrap a strategy's own ``state`` / ``load_state`` so the roll's progress rides along
    under the key ``"__rolls__"``."""
    import functools

    @functools.wraps(state)
    def saved(self):
        own = state(self)
        rolls = self._roll_snapshot()
        if not rolls:
            return own
        if own is None:
            own = {}
        if not isinstance(own, dict):
            raise InvalidData(
                "a RollEvents strategy's state() must return a dict (or None) so the roll's "
                f"progress can be saved with it, got {type(own).__name__}",
                code="invalid_checkpoint",
                operation="state",
                parameter="state",
            )
        return {**own, "__rolls__": rolls}

    @functools.wraps(load_state)
    def loaded(self, snapshot):
        if isinstance(snapshot, dict) and "__rolls__" in snapshot:
            snapshot = dict(snapshot)
            self._roll_load(snapshot.pop("__rolls__"))
            if not snapshot:
                snapshot = None
        load_state(self, snapshot)

    saved._raptorbt_rolls = loaded._raptorbt_rolls = True  # type: ignore[attr-defined]
    return saved, loaded


class ContinuousSeries(NamedTuple):
    """One history built from several contracts. ``bars`` holds ``timestamps`` and OHLCV arrays
    (volume is never adjusted); ``contract`` names the contract each bar came from; ``rolls`` lists
    ``(ts_ns, from_symbol, to_symbol, gap)`` where ``gap`` is the new contract's close less the old
    one's at the roll (the factor, as a ratio, for ``adjust="ratio"``)."""

    bars: dict
    contract: list
    rolls: list


def continuous_series(
    contracts: Mapping[str, Mapping],
    specs: Mapping[str, object],
    *,
    roll_days: float,
    adjust: str,
) -> ContinuousSeries:
    """Join the contracts' bars into one series, rolling ``roll_days`` before each expiry.

    ``contracts`` maps a contract symbol to its bars (``timestamps``, ``open``, ``high``, ``low``,
    ``close``, ``volume``); ``specs`` has each contract's spec (its expiry orders the contracts).
    Contract ``i`` supplies the bars from the previous contract's roll time up to its own.

    ``adjust`` is required, because it changes every earlier price: ``"raw"`` keeps prices as
    traded (the series jumps at a roll), ``"back"`` shifts every earlier bar by the gap at each
    roll so the series is continuous and ends on the latest contract's prices, ``"ratio"``
    multiplies them by the ratio of the two closes instead. The gap is read at the first bar of
    the new contract against the old contract's last close at or before that time.
    """
    if adjust not in ("raw", "back", "ratio"):
        raise InvalidParameter(
            f"adjust must be 'raw', 'back' or 'ratio', got {adjust!r}",
            code="invalid_adjust",
            operation="continuous_series",
            parameter="adjust",
            value=adjust,
            valid=["raw", "back", "ratio"],
        )
    order = sorted(contracts, key=lambda s: specs[s].expiration_ns)
    cutoff = int(roll_days * _DAY)
    edges = [specs[s].expiration_ns - cutoff for s in order]
    cols = ("open", "high", "low", "close")
    parts, names, rolls = [], [], []
    lower = -(2**63)
    for i, symbol in enumerate(order):
        b = contracts[symbol]
        ts = np.asarray(b["timestamps"], dtype=np.int64)
        keep = (ts >= lower) & (ts < edges[i])
        if not keep.any():
            lower = max(lower, edges[i])
            continue
        parts.append({k: np.asarray(b[k], dtype=np.float64)[keep] for k in cols + ("volume",)} | {"timestamps": ts[keep]})
        names.extend([symbol] * int(keep.sum()))
        lower = edges[i]
    if not parts:
        raise InvalidData(
            "no contract has bars in its own window",
            code="no_continuous_bars",
            operation="continuous_series",
            parameter="contracts",
        )
    used = list(dict.fromkeys(names))
    by_contract = dict(zip(used, parts))
    for prev, nxt in zip(used, used[1:]):
        first = by_contract[nxt]
        t = int(first["timestamps"][0])
        old = contracts[prev]
        old_ts = np.asarray(old["timestamps"], dtype=np.int64)
        j = int(np.searchsorted(old_ts, t, side="right")) - 1
        if j < 0:
            raise InvalidData(
                f"{prev} has no bar at or before the roll to {nxt}",
                code="no_continuous_bars",
                operation="continuous_series",
                parameter="contracts",
                value=prev,
            )
        old_close = float(np.asarray(old["close"], dtype=np.float64)[j])
        new_close = float(first["close"][0])
        gap = new_close / old_close if adjust == "ratio" else new_close - old_close
        rolls.append((t, prev, nxt, gap))
    if adjust != "raw":
        # Walk back from the latest contract, carrying the cumulative shift.
        shift = 0.0 if adjust == "back" else 1.0
        for (t, prev, nxt, gap) in reversed(rolls):
            shift = shift + gap if adjust == "back" else shift * gap
            part = by_contract[prev]
            for k in cols:
                part[k] = part[k] + shift if adjust == "back" else part[k] * shift
    bars = {k: np.concatenate([p[k] for p in parts]) for k in ("timestamps", "volume") + cols}
    return ContinuousSeries(bars, names, rolls)
