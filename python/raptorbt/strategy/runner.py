"""Driver loop for class-based strategies."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from raptorbt.errors import InvalidData, InvalidOrder, InvalidParameter
from raptorbt._raptorbt import (
    InstrumentSpec,
    BacktestConfig,
    BacktestResult,
    InstrumentConfig,
    KernelDriver,
    KernelSession,
)
from raptorbt.strategy import api
from raptorbt.strategy._commands import (
    KernelTarget,
    CommandPath,
    atr_at,
    atr_series,
    session_tz_offset_ns,
    config_for_holding,
    dispatch_events,
)
from raptorbt.strategy._stamps import BarStamps
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.context import StrategyContext
from raptorbt.strategy._validate import SessionCoverage, ensure_prices_finite
from raptorbt.strategy.streams import StreamState


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
    config: BacktestConfig | None = None,
    instrument_config: InstrumentConfig | None = None,
    instrument: InstrumentSpec | None = None,
    oms_type: str = "netting",
    account_type: str = "cash",
    leverage: float = 1.0,
    bar_stamp: Literal["open", "close"] | None = None,
    bar_ns: int | dict[str, int] | None = None,
    corporate_actions=None,
    adjustment_policy: Literal["raw", "adjusted"] | None = None,
) -> BacktestResult:
    """Run a class-based strategy over OHLCV arrays.

    ``instrument`` optionally attaches a market definition
    (:class:`InstrumentSpec`): tick-size quantization of derived stops and
    targets, contract-multiplier notional scaling, and expiry settlement.
    When both ``instrument_config.lot_size`` and the spec's lot size are set,
    the explicit config wins.

    Accepts a :class:`Strategy` instance or class (instantiated with no
    arguments). Returns the same ``BacktestResult`` as
    ``run_single_backtest``, so downstream result handling is identical for
    both paths.

    Per bar: ``on_bar`` runs first, queued intents are applied through the
    engine (exits before entries, stop > target > signal), and resulting
    events are dispatched to the ``on_order_*`` / ``on_position_*`` hooks
    before the next bar.

    ``bar_stamp`` says what ``timestamps`` mean: ``"close"`` (a bar is stamped
    when it ends) or ``"open"`` (when it starts; ``bar_ns`` is then required,
    the bar width in nanoseconds, or a ``{symbol: ns}`` dict). The engine
    runs on bar close time: fills, ``DAY`` expiry, session square-off, the
    strategy clock and ``ctx.datetime`` use ``ctx.bar_close_ts``, while
    ``ctx.timestamp`` and ``ctx.bar.timestamp`` stay the stamps you passed.
    Composite bars (``subscribe_bars``) are stamped at their window end, a
    close time. ``bar_stamp`` is required: leaving it out raises
    ``InvalidParameter`` (``missing_bar_stamp``).

    ``corporate_actions`` (:class:`~raptorbt.CorporateAction` for ``symbol``)
    are the splits, bonuses and cash dividends of the run; they need
    ``adjustment_policy``, which says what the bars are: ``"raw"`` (as
    traded; positions are rescaled at the ex-date) or ``"adjusted"`` (history
    rescaled here onto the basis after the last split). The run is a
    one-instrument book, so they apply exactly as in ``run_strategies``.

    Raises:
        ValueError: on a missing or invalid ``bar_stamp``, on ``bar_stamp="open"``
            without a positive ``bar_ns``, on inconsistent array lengths, on decreasing
            timestamps (equal timestamps are allowed), on conflicting same-bar
            intents (enter and close while in position), or on duplicate
            same-bar intents.
    """
    if isinstance(strategy, type):
        strategy = strategy()
    if not isinstance(strategy, Strategy):
        raise InvalidParameter(
            f"strategy must be a Strategy instance or subclass, got {type(strategy).__name__}",
            code="invalid_strategy",
            operation="run_strategy_backtest",
            parameter="strategy",
            value=type(strategy).__name__,
            valid=["Strategy instance", "Strategy subclass"],
        )

    stamps = BarStamps.resolve(bar_stamp, bar_ns, [symbol], "run_strategy_backtest")

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
            raise InvalidData(
                f"{name} has length {len(arr)}, expected {n} (same as timestamps)",
                code="length_mismatch",
                operation="run_strategy_backtest",
                parameter=name,
                value=len(arr),
                valid=[n],
            )
    if n == 0:
        raise InvalidData(
            "cannot backtest zero bars",
            code="empty_data",
            operation="run_strategy_backtest",
            parameter="timestamps",
            value=0,
        )
    decreasing = np.flatnonzero(timestamps[1:] < timestamps[:-1])
    if decreasing.size:
        i = int(decreasing[0]) + 1
        raise InvalidData(
            f"{symbol} bars: timestamps must not decrease, but index {i} is "
            f"{int(timestamps[i])} after {int(timestamps[i - 1])}",
            code="decreasing_timestamps",
            operation="run_strategy_backtest",
            parameter="timestamps",
            value=int(timestamps[i]),
        )

    ensure_prices_finite(
        f"{symbol} bars",
        "run_strategy_backtest",
        {"open": open_, "high": high, "low": low, "close": close},
    )

    coverage = SessionCoverage.of(symbol, instrument, "run_strategy_backtest")
    if coverage is not None:
        coverage.check_all(timestamps)

    from raptorbt.strategy.multi_runner import _adjusted, _lifecycle

    corporate_actions, adjustment_policy = _lifecycle(
        corporate_actions, adjustment_policy, {symbol}, set(), "run_strategy_backtest"
    )
    if adjustment_policy == "adjusted":
        adjusted = _adjusted(
            symbol,
            corporate_actions,
            {"timestamps": timestamps, "open": open_, "high": high, "low": low,
             "close": close, "volume": volume},
            "run_strategy_backtest",
        )
        open_, high, low, close, volume = (
            adjusted["open"], adjusted["high"], adjusted["low"], adjusted["close"],
            adjusted["volume"],
        )

    # Everything the engine does in time runs on bar close time.
    visible = stamps.visible(symbol, timestamps)

    config = config_for_holding(config, [strategy], "run_strategy_backtest")

    session = KernelSession(
        symbol=symbol,
        direction=direction,
        config=config,
        instrument_config=instrument_config,
        instrument=instrument,
        oms_type=oms_type,
        account_type=account_type,
        leverage=leverage,
    )

    if adjustment_policy is not None:
        session.set_corporate_actions(corporate_actions, adjustment_policy)

    # Same ATR series the array-based engine would compute for ATR-based
    # stop/target configs.
    atr_values = atr_series(config, instrument_config, high, low, close)

    ctx = StrategyContext(
        session, timestamps, open_, high, low, close, volume, close_timestamps=visible
    )

    strategy.drain_orders()  # discard intents queued before the run
    strategy.drain_commands()
    strategy._bar_subscriptions = []
    strategy._indicators = []
    from raptorbt.strategy.cache import Cache
    from raptorbt.strategy.clock import Clock

    strategy.clock = Clock()
    strategy.cache = Cache()
    api.begin(strategy, ctx, [symbol], symbol)
    strategy.on_start(ctx)
    mask = api.finish(strategy)

    # Multi-timeframe: one streaming aggregator per subscription declared in
    # on_start. Completed composite bars dispatch before the primary on_bar
    # of the bar that completed them (they closed strictly earlier).
    streams = StreamState(
        strategy,
        tz_offset_ns=session_tz_offset_ns(config),
        stamp=stamps.aggregator_stamp,
        sessions={None: instrument.session} if instrument is not None else None,
    )

    # The Python command path stays reachable while the native one settles.
    if os.environ.get("RAPTORBT_PY_COMMANDS") == "1":
        commands = CommandPath(strategy, KernelTarget(session))
        driver = None
    else:
        commands = None
        driver = KernelDriver(session, strategy, ctx, mask, atr_values)

    for i in range(n):
        ctx._advance(i)

        # Clock first: scheduled times precede the bar revealing them.
        for time_event in strategy.clock._advance(int(visible[i])):
            strategy.on_time_event(ctx, time_event)

        streams.push(
            strategy,
            ctx,
            int(visible[i]),
            float(open_[i]),
            float(high[i]),
            float(low[i]),
            float(close[i]),
            float(volume[i]),
        )

        strategy.on_bar(ctx)
        if driver is not None:
            driver.step(
                i,
                int(visible[i]),
                float(open_[i]),
                float(high[i]),
                float(low[i]),
                float(close[i]),
                float(volume[i]),
            )
            continue
        commands(0, i, int(visible[i]))

        intents = commands.fold(i, atr=atr_at(atr_values, i))

        # An enter+close pair while in position would exit and immediately
        # re-enter on the same bar; refuse rather than guess the intent.
        if intents["entry"] and intents["exit"] and session.is_in_position():
            raise InvalidOrder(
                f"bar {i}: enter() and close_position() queued on the same bar "
                "while in position; emit one intent per bar",
                code="invalid_order_intent",
                operation="run_strategy_backtest",
                parameter="intents",
                value=i,
            )

        events = session.step(
            i,
            int(visible[i]),
            float(open_[i]),
            float(high[i]),
            float(low[i]),
            float(close[i]),
            float(volume[i]),
            **intents,
        )

        dispatch_events(strategy, ctx, events)

    strategy.on_stop(ctx)

    return session.finish()
