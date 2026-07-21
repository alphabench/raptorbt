"""Driver loop for class-based strategies."""

from __future__ import annotations

import numpy as np

from raptorbt._raptorbt import (
    PyBacktestConfig,
    PyBacktestResult,
    PyInstrumentConfig,
    PyKernelSession,
    atr as _atr,
    resolve_atr_period,
)
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.context import StrategyContext
from raptorbt.strategy.orders import ClosePosition, MarketOrder


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
) -> PyBacktestResult:
    """Run a class-based strategy over OHLCV arrays.

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
    strategy.on_start(ctx)

    for i in range(n):
        ctx.idx = i
        strategy.on_bar(ctx)

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

        for event in events:
            if event.kind == "entered":
                strategy.on_order_filled(ctx, event)
                strategy.on_position_opened(ctx, event)
            elif event.kind == "exited":
                strategy.on_order_filled(ctx, event)
                strategy.on_position_closed(ctx, event)
            elif event.kind == "entry_rejected":
                strategy.on_order_rejected(ctx, event)

    strategy.on_stop(ctx)

    return session.finish()
