"""The one path from a strategy's queued work to the engine.

Every runner (single-instrument, portfolio, tick, live stream) drains the
strategy's intents, applies its queued commands and routes engine events to
hooks through this module. What differs between a ``KernelSession`` and a
``PortfolioSession`` is confined to the two small target adapters below;
the logic is not duplicated per runner.
"""

from __future__ import annotations

from raptorbt._raptorbt import atr as _atr, resolve_atr_period
from raptorbt.errors import InvalidOrder


def atr_series(config, instrument_config, high, low, close):
    """The ATR series the array engine would use for ATR-based stops/targets.

    ``None`` when the config sets no ATR-based level, or when the data cannot
    produce one (period resolution happens in the engine crate). Unusable ATR
    degrades to "no stop" (an ATR of 0.0 arms nothing) rather than failing
    the run, mirroring the array engine.
    """
    period = resolve_atr_period(config, instrument_config)
    if not period:
        return None
    try:
        return _atr(high, low, close, period)
    except ValueError:
        return None


def session_tz_offset_ns(config) -> int:
    """The session's UTC offset in ns, 0 without a config."""
    return config.session_tz_offset_ns if config is not None else 0


def atr_at(series, idx: int) -> float:
    """The ATR for bar ``idx``, or 0.0 (arms nothing) without a series."""
    return float(series[idx]) if series is not None else 0.0


def reject_atr_stops(config, instrument_configs, symbols) -> None:
    """Refuse an ATR-based stop/target in a run that has no bars.

    Tick sessions have no bar series to compute ATR from; leaving the stop
    unarmed would silently disable it.
    """
    for symbol in symbols:
        period = resolve_atr_period(config, (instrument_configs or {}).get(symbol))
        if period:
            raise ValueError(
                f"ATR-based stop/target configured (ATR period {period}, "
                f"symbol {symbol!r}), but tick runs have no bars to compute "
                "ATR from, so the stop would never arm. Use a fixed stop or "
                "target, or run on bars with run_portfolio_strategy."
            )


def dispatch_events(strategy, ctx, events) -> None:
    """Route engine events into strategy hooks.

    The ``entered``/``exited`` events double as fill notifications for the
    legacy intent path; an order-driven fill already fired
    ``on_order_filled`` for its own ``order_filled`` event, so the
    immediately following position event must not fire it again.
    """
    symbol = getattr(ctx, "symbol", None)
    order_fill_preceded = False
    for event in events:
        strategy.cache._observe(event, symbol)
        if event.kind == "entered":
            if not order_fill_preceded:
                strategy.on_order_filled(ctx, event)
            strategy.on_position_opened(ctx, event)
            order_fill_preceded = False
        elif event.kind == "exited":
            if not order_fill_preceded:
                strategy.on_order_filled(ctx, event)
            strategy.on_position_closed(ctx, event)
            order_fill_preceded = False
        elif event.kind == "entry_rejected":
            strategy.on_order_rejected(ctx, event)
        elif event.kind == "algo_started":
            strategy.on_algo_started(ctx, event)
            strategy.on_order_event(ctx, event)
        elif event.kind == "algo_completed":
            strategy.on_algo_completed(ctx, event)
            strategy.on_order_event(ctx, event)
        elif event.kind in ("risk_rejected", "risk_halted"):
            strategy.on_risk_event(ctx, event)
        elif event.kind == "margin_call":
            strategy.on_margin_call(ctx, event)
            strategy.on_order_event(ctx, event)
        elif event.kind.startswith("order_"):
            handler = {
                "order_accepted": strategy.on_order_accepted,
                "order_triggered": strategy.on_order_triggered,
                "order_filled": strategy.on_order_filled,
                "order_canceled": strategy.on_order_canceled,
                "order_expired": strategy.on_order_expired,
                "order_rejected": strategy.on_order_rejected,
            }.get(event.kind)
            if handler is not None:
                handler(ctx, event)
            strategy.on_order_event(ctx, event)
            order_fill_preceded = event.kind == "order_filled"


def fold_intents(strategy, idx: int, symbol: str | None = None, atr: float = 0.0) -> dict:
    """Fold a strategy's queued intents into per-step keyword arguments.

    ``idx`` and ``symbol`` name the event for error messages. The result
    feeds ``KernelSession.step`` and ``PortfolioSession.apply_current`` alike.
    """
    queued = strategy.drain_orders()
    if not queued:
        return {
            "entry": False,
            "exit": False,
            "atr": atr,
            "size_mult": None,
            "stop_price": None,
            "target_price": None,
        }
    where = f"bar {idx}" if symbol is None else f"{symbol} event {idx}"
    entry = False
    exit_ = False
    size_mult = None
    stop_override = None
    target_override = None
    for intent in queued:
        if intent[0] == "enter":
            if entry:
                raise InvalidOrder(
                    f"duplicate entry intents queued on {where}",
                    code="invalid_order_intent",
                    parameter="intents",
                    value="entry",
                )
            entry = True
            _, size_mult, stop_override, target_override = intent
        else:
            if exit_:
                raise InvalidOrder(
                    f"duplicate close intents queued on {where}",
                    code="invalid_order_intent",
                    parameter="intents",
                    value="close",
                )
            exit_ = True
    return {
        "entry": entry,
        "exit": exit_,
        "atr": atr,
        "size_mult": size_mult,
        "stop_price": stop_override,
        "target_price": target_override,
    }


class KernelTarget:
    """Adapter over a single-instrument ``KernelSession``.

    There is one instrument, so every instrument argument is ignored and
    symbol routing degenerates to that instrument.
    """

    n_instruments = 1

    def __init__(self, session):
        self.session = session

    def instrument_of(self, symbol, current: int) -> int:
        return 0

    def submit_order(self, instrument: int, **kwargs) -> int:
        return self.session.submit_order(**kwargs)

    def submit_twap(self, instrument: int, idx, ts, client_id, order) -> int:
        return self.session.submit_twap(
            order["units"],
            order["side"],
            order["slices"],
            order["interval_ns"],
            idx,
            ts,
            client_id,
            order["reduce_only"],
        )

    def cancel_twap(self, instrument: int, algo_id: int, idx: int) -> None:
        self.session.cancel_twap(algo_id, idx)

    def cancel_order(self, instrument: int, idx: int, order_id: int) -> None:
        self.session.cancel_order(idx, order_id)

    def cancel_all_orders(self, instrument: int, idx: int) -> None:
        self.session.cancel_all_orders(idx)

    def link_oco(self, instrument: int, order_ids: list[int]) -> None:
        self.session.link_oco(order_ids)

    def request_close(self, instrument: int, position_id: int) -> None:
        self.session.request_close(position_id)

    def positions(self, instrument: int):
        return self.session.positions()

    def modify_order(self, instrument: int, order_id: int, **changes) -> None:
        self.session.modify_order(order_id, **changes)


class PortfolioTarget:
    """Adapter over a ``PortfolioSession``: one kernel per instrument."""

    def __init__(self, session, ctx):
        self.session = session
        self._ctx = ctx
        self.n_instruments = len(ctx.symbols)

    def instrument_of(self, symbol, current: int) -> int:
        return self._ctx._instrument_index(symbol) if symbol else current

    def submit_order(self, instrument: int, **kwargs) -> int:
        return self.session.submit_order(instrument, **kwargs)

    def submit_twap(self, instrument: int, idx, ts, client_id, order) -> int:
        return self.session.submit_twap(
            instrument,
            order["units"],
            order["side"],
            order["slices"],
            order["interval_ns"],
            idx,
            ts,
            client_id,
            order["reduce_only"],
        )

    def cancel_twap(self, instrument: int, algo_id: int, idx: int) -> None:
        self.session.cancel_twap(instrument, algo_id, idx)

    def cancel_order(self, instrument: int, idx: int, order_id: int) -> None:
        self.session.cancel_order(instrument, idx, order_id)

    def cancel_all_orders(self, instrument: int, idx: int) -> None:
        self.session.cancel_all_orders(instrument, idx)

    def link_oco(self, instrument: int, order_ids: list[int]) -> None:
        self.session.link_oco(instrument, order_ids)

    def request_close(self, instrument: int, position_id: int) -> None:
        self.session.request_close(instrument, position_id)

    def positions(self, instrument: int):
        return self.session.positions(instrument)

    def modify_order(self, instrument: int, order_id: int, **changes) -> None:
        self.session.modify_order(instrument, order_id, **changes)


def order_kwargs(order: dict, idx: int, ts: int, client_id: str, parent_engine_id) -> dict:
    """``submit_order`` keyword arguments for a queued order."""
    return dict(
        order,
        submitted_idx=idx,
        submitted_ts=ts,
        client_id=client_id,
        parent_id=parent_engine_id,
    )


class CommandPath:
    """A run's route from queued strategy work to the engine.

    Calling it with ``(current_instrument, local_idx, ts)`` drains the
    strategy's queued commands and routes each to its instrument through
    ``target``; client order ids map to ``(instrument, engine id)`` for the
    life of the run. :meth:`fold` and :meth:`settle` handle intents (the
    ``enter`` / ``close_position`` requests), which only a step can carry.
    """

    def __init__(self, strategy, target):
        self._strategy = strategy
        self._target = target
        self._id_map: dict[str, tuple[int, int]] = {}
        # Schedules live apart from orders: an algo id is not an order id,
        # so a modify, parent or OCO link naming one must not resolve to it.
        self._algo_map: dict[str, tuple[int, int]] = {}
        # Intents queued where no step could carry them, by symbol.
        self._held: dict[str, list] = {}

    def fold(self, idx: int, symbol: str | None = None, atr: float = 0.0) -> dict:
        """Step arguments for the event being stepped.

        Intents held for ``symbol`` by :meth:`settle` ride this step ahead
        of anything queued since.
        """
        if self._held:
            held = self._held.pop(symbol, None)
            if held:
                self._strategy._queue.restore_intents(held)
        return fold_intents(self._strategy, idx, symbol, atr)

    def settle(self, instrument: int, symbol: str, local_idx: int, ts: int) -> None:
        """Close out an event whose hooks may have queued more work.

        Commands go to the event's own instrument now. Intents cannot step
        a second time, so they are held for the symbol's next step rather
        than left on the queue for whichever symbol is stepped next.
        """
        strategy = self._strategy
        if strategy._queue.has_commands():
            self(instrument, local_idx, ts)
        if strategy._queue.has_intents():
            self._held.setdefault(symbol, []).extend(strategy.drain_orders())

    def __call__(self, current_instrument: int, local_idx: int, ts: int) -> None:
        strategy = self._strategy
        if not strategy._queue.has_commands():
            return
        target = self._target
        id_map = self._id_map
        for command in strategy.drain_commands():
            kind = command[0]
            if kind == "submit":
                _, client_id, order, parent, symbol = command
                instrument = target.instrument_of(symbol, current_instrument)
                if order.get("twap"):
                    # A schedule, not an order: it releases its own slices.
                    self._algo_map[client_id] = (
                        instrument,
                        target.submit_twap(instrument, local_idx, ts, client_id, order),
                    )
                    continue
                parent_engine_id = None
                if parent:
                    mapped = id_map.get(parent)
                    if mapped is None:
                        raise InvalidOrder(
                            f"unknown parent order {parent!r}",
                            code="invalid_order_link",
                            parameter="parent",
                            value=parent,
                            valid=list(id_map),
                        )
                    if mapped[0] != instrument:
                        raise InvalidOrder(
                            "parent order belongs to a different symbol",
                            code="invalid_order_link",
                            parameter="parent",
                            value=parent,
                        )
                    parent_engine_id = mapped[1]
                engine_id = target.submit_order(
                    instrument,
                    **order_kwargs(order, local_idx, ts, client_id, parent_engine_id),
                )
                id_map[client_id] = (instrument, engine_id)
            elif kind == "cancel":
                mapped = id_map.get(command[1])
                if mapped is not None:
                    target.cancel_order(mapped[0], local_idx, mapped[1])
                elif command[1] in self._algo_map:
                    # A finished schedule is gone from the engine; cancelling
                    # it does nothing, as for a filled order.
                    instrument, algo_id = self._algo_map[command[1]]
                    target.cancel_twap(instrument, algo_id, local_idx)
            elif kind == "cancel_all":
                for instrument in range(target.n_instruments):
                    target.cancel_all_orders(instrument, local_idx)
            elif kind == "link_oco":
                mapped_ids = [id_map[c] for c in command[1] if c in id_map]
                if len(mapped_ids) >= 2:
                    if len({m[0] for m in mapped_ids}) != 1:
                        raise InvalidOrder(
                            "one-cancels-other links cannot span symbols",
                            code="invalid_order_link",
                            parameter="client_ids",
                            value=list(command[1]),
                        )
                    target.link_oco(mapped_ids[0][0], [m[1] for m in mapped_ids])
            elif kind == "close":
                _, position_id, symbol = command
                target.request_close(
                    target.instrument_of(symbol, current_instrument), position_id
                )
            elif kind == "close_all_for":
                instrument = target.instrument_of(command[1], current_instrument)
                for snapshot in target.positions(instrument):
                    target.request_close(instrument, snapshot.position_id)
            elif kind == "modify":
                # The id map carries the owning instrument, so a modify
                # routes without the caller naming a symbol.
                mapped = id_map.get(command[1])
                if mapped is not None:
                    target.modify_order(mapped[0], mapped[1], **command[2])
