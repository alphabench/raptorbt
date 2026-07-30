"""Driver loop for class-based strategies."""

from __future__ import annotations

import numpy as np

from raptorbt._raptorbt import (
    InstrumentSpec,
    PyBacktestConfig,
    PyBacktestResult,
    PyInstrumentConfig,
    PyKernelSession,
    atr as _atr,
    resolve_atr_period,
)
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.context import StrategyContext
from raptorbt.strategy.orders import ClosePosition, MarketOrder, Twap
from raptorbt.strategy.streams import StreamState


def dispatch_events(strategy: Strategy, ctx, events) -> None:
    """Route engine events into strategy hooks.

    Shared by the single-instrument and portfolio runners. The `entered`/
    `exited` events double as fill notifications for the legacy intent
    path; an order-driven fill already fired ``on_order_filled`` for its
    own ``order_filled`` event, so the immediately following position event
    must not fire it again.
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


def run_strategy_backtest(
    strategy: Strategy | type[Strategy],
    timestamps,
    open,
    high,
    low,
    close,
    volume,
    direction: int = 1,
    symbol: str = "ASSET",
    config: PyBacktestConfig | None = None,
    instrument_config: PyInstrumentConfig | None = None,
    instrument: InstrumentSpec | None = None,
    oms_type: str = "netting",
    account_type: str = "cash",
    leverage: float = 1.0,
) -> PyBacktestResult:
    """Run a class-based strategy over OHLCV arrays.

    ``instrument`` optionally attaches a market definition
    (:class:`InstrumentSpec`): tick-size quantization of derived stops and
    targets, contract-multiplier notional scaling, and expiry settlement.
    When both ``instrument_config.lot_size`` and the spec's lot size are set,
    the explicit config wins.

    Accepts a :class:`Strategy` instance or class (instantiated with no
    arguments). Returns the same ``PyBacktestResult`` as
    ``run_single_backtest``, so downstream result handling is identical for
    both paths.

    Per bar: ``on_bar`` runs first, queued intents are applied through the
    engine (exits before entries, stop > target > signal), and resulting
    events are dispatched to the ``on_order_*`` / ``on_position_*`` hooks
    before the next bar.

    Raises:
        ValueError: on inconsistent array lengths, on conflicting same-bar
            intents (enter and close while in position), or on duplicate
            same-bar intents.
    """
    if isinstance(strategy, type):
        strategy = strategy()
    if not isinstance(strategy, Strategy):
        raise ValueError(
            f"strategy must be a Strategy instance or subclass, got {type(strategy).__name__}"
        )

    timestamps = np.ascontiguousarray(timestamps, dtype=np.int64)
    open_ = np.ascontiguousarray(open, dtype=np.float64)
    high = np.ascontiguousarray(high, dtype=np.float64)
    low = np.ascontiguousarray(low, dtype=np.float64)
    close = np.ascontiguousarray(close, dtype=np.float64)
    volume = np.ascontiguousarray(volume, dtype=np.float64)

    n = len(timestamps)
    for name, arr in (
        ("open", open_),
        ("high", high),
        ("low", low),
        ("close", close),
        ("volume", volume),
    ):
        if len(arr) != n:
            raise ValueError(f"{name} has length {len(arr)}, expected {n} (same as timestamps)")
    if n == 0:
        raise ValueError("cannot backtest zero bars")

    session = PyKernelSession(
        symbol=symbol,
        direction=direction,
        config=config,
        instrument_config=instrument_config,
        instrument=instrument,
        oms_type=oms_type,
        account_type=account_type,
        leverage=leverage,
    )

    # Same ATR series the array-based engine would compute for ATR-based
    # stop/target configs; period resolution happens in the engine crate.
    atr_period = resolve_atr_period(config, instrument_config)
    atr_values = None
    if atr_period:
        try:
            atr_values = _atr(high, low, close, atr_period)
        except ValueError:
            # Mirror the array engine: unusable ATR degrades to "no stop"
            # (an ATR of 0.0 sets no stop/target) rather than failing the run.
            atr_values = None

    ctx = StrategyContext(session, timestamps, open_, high, low, close, volume)

    strategy.drain_orders()  # discard intents queued before the run
    strategy.drain_commands()
    strategy._bar_subscriptions = []
    strategy._indicators = []
    from raptorbt.strategy.cache import Cache
    from raptorbt.strategy.clock import Clock

    strategy.clock = Clock()
    strategy.cache = Cache()
    strategy.on_start(ctx)

    # Multi-timeframe: one streaming aggregator per subscription declared in
    # on_start. Completed composite bars dispatch before the primary on_bar
    # of the bar that completed them (they closed strictly earlier).
    streams = StreamState(strategy)

    # client order id -> engine order id, for cancel/modify routing.
    id_map: dict[str, int] = {}

    def apply_commands(i: int) -> None:
        for command in strategy.drain_commands():
            if command[0] == "submit":
                _, client_id, order, parent, _symbol = command
                if isinstance(order, Twap):
                    # A schedule, not an order: it releases its own slices.
                    session.submit_twap(
                        order.units,
                        order.side,
                        order.slices,
                        order.interval_ns,
                        i,
                        int(timestamps[i]),
                        client_id,
                        order.reduce_only,
                    )
                    continue
                parent_engine_id = id_map.get(parent) if parent else None
                if parent and parent_engine_id is None:
                    raise ValueError(f"unknown parent order {parent!r}")
                engine_id = session.submit_order(
                    side=order.side,
                    kind=order.kind,
                    submitted_idx=i,
                    submitted_ts=int(timestamps[i]),
                    client_id=client_id,
                    units=order.units,
                    size_frac=order.size_frac,
                    limit_price=getattr(order, "price", None),
                    trigger_price=getattr(order, "trigger", None),
                    tif=order.tif,
                    expire_ns=order.expire_ns,
                    stop_price=order.stop_price,
                    target_price=order.target_price,
                    offset=getattr(order, "offset", None),
                    offset_kind=getattr(order, "offset_kind", "price"),
                    limit_offset=getattr(order, "limit_offset", 0.0),
                    post_only=getattr(order, "post_only", False),
                    reduce_only=order.reduce_only,
                    parent_id=parent_engine_id,
                )
                id_map[client_id] = engine_id
            elif command[0] == "link_oco":
                engine_ids = [id_map[c] for c in command[1] if c in id_map]
                if len(engine_ids) >= 2:
                    session.link_oco(engine_ids)
            elif command[0] == "cancel":
                engine_id = id_map.get(command[1])
                if engine_id is not None:
                    session.cancel_order(i, engine_id)
            elif command[0] == "cancel_all":
                session.cancel_all_orders(i)
            elif command[0] == "close":
                session.request_close(command[1])
            elif command[0] == "close_all_for":
                # Single-instrument run: symbol routing degenerates to the
                # legacy whole-position close.
                for snapshot in session.positions():
                    session.request_close(snapshot.position_id)
            elif command[0] == "modify":
                engine_id = id_map.get(command[1])
                if engine_id is not None:
                    session.modify_order(engine_id, **command[2])

    for i in range(n):
        ctx.idx = i

        # Clock first: scheduled times precede the bar revealing them.
        for time_event in strategy.clock._advance(int(timestamps[i])):
            strategy.on_time_event(ctx, time_event)

        streams.push(
            strategy,
            ctx,
            int(timestamps[i]),
            float(open_[i]),
            float(high[i]),
            float(low[i]),
            float(close[i]),
            float(volume[i]),
        )

        strategy.on_bar(ctx)
        apply_commands(i)

        entry = False
        exit_ = False
        size_mult: float | None = None
        stop_override: float | None = None
        target_override: float | None = None

        for intent in strategy.drain_orders():
            if isinstance(intent, MarketOrder):
                if entry:
                    raise ValueError(f"duplicate entry intents queued on bar {i}")
                entry = True
                size_mult = intent.size_frac
                stop_override = intent.stop_price
                target_override = intent.target_price
            elif isinstance(intent, ClosePosition):
                if exit_:
                    raise ValueError(f"duplicate close intents queued on bar {i}")
                exit_ = True
            else:
                raise ValueError(f"unknown order intent on bar {i}: {intent!r}")

        # An enter+close pair while in position would exit and immediately
        # re-enter on the same bar; refuse rather than guess the intent.
        if entry and exit_ and session.is_in_position():
            raise ValueError(
                f"bar {i}: enter() and close_position() queued on the same bar "
                "while in position; emit one intent per bar"
            )

        events = session.step(
            i,
            int(timestamps[i]),
            float(open_[i]),
            float(high[i]),
            float(low[i]),
            float(close[i]),
            float(volume[i]),
            entry=entry,
            exit=exit_,
            atr=float(atr_values[i]) if atr_values is not None else 0.0,
            size_mult=size_mult,
            stop_price=stop_override,
            target_price=target_override,
        )

        dispatch_events(strategy, ctx, events)

    strategy.on_stop(ctx)

    return session.finish()
