"""The canonical strategy API: hooks, subscriptions, data, indicators, portfolio.

Everything a strategy declares or reads through the new names lives here, so
that the runners stay ignorant of it. A runner does two things:

* ``begin(strategy, ctx, ...)`` right after it creates the strategy's context.
  It attaches ``ctx.data`` / ``ctx.indicators`` / ``ctx.portfolio`` and calls
  ``on_init``.
* ``finish(strategy)`` once ``on_start`` has returned. It wires the hooks the
  strategy defined to the events the runtime delivers and returns the hook
  mask the runtime uses to skip events nobody handles.

The runtime itself still calls the historical hook names (``on_trade_tick``,
``on_composite_bar``, ...). ``HOOKS`` maps every canonical name onto them, and
``finish`` installs a small dispatcher on the strategy instance where more
than one name (or a subscription filter) is involved, so each event reaches
each defined hook exactly once. A legacy strategy that defines only legacy
names and never calls ``subscribe`` gets no dispatcher at all: the runtime
calls its methods directly, as before.
"""

from __future__ import annotations

import inspect
import warnings
from collections import deque
from typing import Any, NamedTuple

import numpy as np

from raptorbt import errors
from raptorbt._raptorbt import BarAggregator, Indicator
from raptorbt.strategy.context import Bar

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _invalid(title, parameter, value, supported=None, *, code):
    """Raise the structured error every invalid declaration uses; ``code``
    is its entry in ``raptorbt.errors.CODES``."""
    raise errors.CODES[code].cls(
        title, code=code, operation="strategy", parameter=parameter, value=value, valid=supported
    )


# ---------------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------------

#: Strings accepted wherever a timeframe is: ``(step, unit)`` of the aggregator.
TIMEFRAMES: dict[str, tuple[int, str]] = {
    "1m": (1, "m"),
    "3m": (3, "m"),
    "5m": (5, "m"),
    "10m": (10, "m"),
    "15m": (15, "m"),
    "30m": (30, "m"),
    "1h": (1, "h"),
    "2h": (2, "h"),
    "4h": (4, "h"),
    "1d": (1, "d"),
    "1w": (1, "w"),
}


class Timeframe(NamedTuple):
    """A bar spec for the timeframes a string cannot name.

    ``Timeframe(100, "tick")`` is also what the plain tuple ``(100, "tick")``
    means. Units: ``"ms"``, ``"s"``, ``"m"``, ``"h"``, ``"d"``, ``"w"``,
    ``"tick"``, ``"volume"``, ``"value"``, ``"renko"``.
    """

    step: int
    unit: str
    brick_size: float = 0.0

    @staticmethod
    def ticks(n: int) -> Timeframe:
        """A bar every ``n`` trades."""
        return Timeframe(n, "tick")

    @staticmethod
    def volume(n: int) -> Timeframe:
        """A bar every ``n`` units traded."""
        return Timeframe(n, "volume")

    @staticmethod
    def value(n: int) -> Timeframe:
        """A bar every ``n`` of traded value (price * volume)."""
        return Timeframe(n, "value")

    @staticmethod
    def renko(brick_size: float) -> Timeframe:
        """Renko bricks ``brick_size`` price units tall."""
        return Timeframe(1, "renko", float(brick_size))

    @property
    def label(self) -> str:
        return f"{self.step}{self.unit}"


def resolve_timeframe(value, parameter="timeframe") -> Timeframe | None:
    """``None`` for the primary bars, else the validated :class:`Timeframe`."""
    if value is None:
        return None
    spec = None
    if isinstance(value, str):
        pair = TIMEFRAMES.get(value)
        if pair is not None:
            spec = Timeframe(*pair)
    elif isinstance(value, (tuple, list)) and 2 <= len(value) <= 3:
        step, unit = value[0], value[1]
        if isinstance(step, int) and not isinstance(step, bool) and isinstance(unit, str):
            try:
                brick = float(value[2]) if len(value) == 3 else 0.0
                BarAggregator(step, unit, brick_size=brick)
            except (TypeError, ValueError):
                pass
            else:
                spec = Timeframe(step, unit, brick)
    if spec is None:
        _invalid(
            "Invalid timeframe subscription.",
            parameter,
            value,
            list(TIMEFRAMES),
        code="invalid_timeframe")
    return spec


# ---------------------------------------------------------------------------
# The hook table
# ---------------------------------------------------------------------------


class Hook(NamedTuple):
    #: Deprecated names that receive the same events.
    aliases: tuple[str, ...] = ()
    #: The names the runtime calls that carry this hook's events.
    channels: tuple[str, ...] = ()


#: Every hook a strategy may define, canonical name first. The runtime (and
#: ``dispatch_events``) call the *channel* names; a canonical hook, its
#: aliases and ``on_market_data`` are all served from them.
#:
#: * ``on_bar`` is a canonical hook on two channels. Defined as ``on_bar(ctx)``
#:   it is the legacy primary-bar hook and sees primary bars only; defined as
#:   ``on_bar(ctx, event)`` it sees primary and composite bars, and
#:   ``on_composite_bar`` becomes its alias.
#: * ``on_market_data`` sees every bar, composite bar, trade, quote and book,
#:   after the specific hook.
#: * ``on_risk_event`` has no alias; the runners raise it for ``risk_rejected``
#:   and ``risk_halted`` engine events.
#:
#: This table, ``CHANNEL_BITS`` and ``DEPRECATED`` are the single source of
#: truth for hook names: a runner or native command path that dispatches a
#: new event should add its channel here rather than call a name directly.
HOOKS: dict[str, Hook] = {
    "on_init": Hook(),
    "on_start": Hook(),
    "on_timer": Hook(("on_time_event",), ("on_time_event",)),
    "on_trade": Hook(("on_trade_tick",), ("on_trade_tick",)),
    "on_quote": Hook((), ("on_quote",)),
    "on_book": Hook(("on_order_book",), ("on_order_book",)),
    "on_bar": Hook(("on_composite_bar",), ("on_bar", "on_composite_bar")),
    "on_fill": Hook(("on_order_filled",), ("on_order_filled",)),
    "on_position_event": Hook(
        ("on_position_opened", "on_position_closed"),
        ("on_position_opened", "on_position_closed"),
    ),
    "on_order_event": Hook(),
    "on_risk_event": Hook((), ("on_risk_event",)),
    "on_stop": Hook(),
    # Last: it follows the specific hook of the same event.
    "on_market_data": Hook(
        (), ("on_bar", "on_composite_bar", "on_trade_tick", "on_quote", "on_order_book")
    ),
}

#: Bit of each channel in the mask handed to the runtime. The event
#: channels' bits say whether anything receives them.
CHANNEL_BITS: dict[str, int] = {
    "on_time_event": 1,
    "on_composite_bar": 2,
    "on_bar": 4,
    "on_trade_tick": 8,
    "on_quote": 16,
    "on_order_book": 32,
    "on_order_filled": 1 << 6,
    "on_risk_event": 1 << 7,
    "on_position_opened": 1 << 8,
    "on_position_closed": 1 << 9,
}

#: Bits of the engine-event hooks that have no channel: they are plain
#: methods, called by name, so the mask only records whether they are
#: overridden. The native dispatch skips the ones that are not.
PLAIN_EVENT_BITS: dict[str, int] = {
    "on_order_rejected": 1 << 10,
    "on_order_accepted": 1 << 11,
    "on_order_triggered": 1 << 12,
    "on_order_canceled": 1 << 13,
    "on_order_expired": 1 << 14,
    "on_order_event": 1 << 15,
    "on_algo_started": 1 << 16,
    "on_algo_completed": 1 << 17,
    "on_margin_call": 1 << 18,
    "on_contract_event": 1 << 19,
    "on_corporate_action": 1 << 20,
}

#: Channels that carry a subscribable data type.
_KINDS = {"on_trade_tick": "trade", "on_quote": "quote", "on_order_book": "book"}

#: Deprecated name -> canonical name.
DEPRECATED: dict[str, str] = {
    alias: name for name, hook in HOOKS.items() for alias in hook.aliases
}

_BASE: type | None = None
_DEFAULT_HISTORY = 1000


def _install_base(cls: type) -> None:
    global _BASE
    _BASE = cls


def _takes_event(fn, bound: bool) -> bool:
    """Whether an ``on_bar`` takes ``(ctx, event)`` rather than ``(ctx)``.

    Counted by *required* positional parameters: two or more besides ``self``
    means the hook wants the event. A defaulted second parameter
    (``on_bar(self, ctx, event=None)``) or ``*args`` keeps the legacy single
    argument, so nothing that ran before changes meaning.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    return len(positional) - (0 if bound else 1) >= 2


def check_class(cls: type) -> None:
    """Class-creation checks: ambiguous hook pairs raise, deprecated names warn."""
    base = _BASE

    def defined(name):
        return getattr(cls, name, None) is not getattr(base, name)

    for name, hook in HOOKS.items():
        if not defined(name):
            continue
        for alias in hook.aliases:
            if not defined(alias):
                continue
            if name == "on_bar" and not _takes_event(cls.on_bar, bound=False):
                continue  # on_bar(ctx) is the primary-bar hook, not the alias's twin
            raise TypeError(
                f"{cls.__qualname__} defines both {name} and its deprecated alias "
                f"{alias}; they receive the same events, so one of them would run "
                f"twice. Keep {name}."
            )
    own = [alias for alias in DEPRECATED if alias in cls.__dict__]
    if own:
        names = ", ".join(f"{a} -> {DEPRECATED[a]}" for a in own)
        warnings.warn(
            f"{cls.__qualname__} defines deprecated hooks ({names}); the old names "
            "keep working and receive the same events",
            DeprecationWarning,
            stacklevel=3,
        )


def _overridden(strategy, name: str) -> bool:
    method = getattr(strategy, name, None)
    return getattr(method, "__func__", method) is not getattr(_BASE, name)


def _unbind(strategy) -> None:
    for channel in CHANNEL_BITS:
        if getattr(strategy.__dict__.get(channel), "_raptorbt_dispatcher", False):
            del strategy.__dict__[channel]


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

_DATA_TYPES = ("bar", "trade", "quote", "book")


class _Stream:
    """One composite stream of the API: a timeframe the strategy asked for."""

    def __init__(self, stream_id: int, tf: Timeframe):
        self.id = stream_id
        self.tf = tf
        #: Instruments whose bars reach the strategy's hooks.
        self.deliver: set[str] = set()
        #: Instrument -> bounded buffer of closed bars.
        self.buffers: dict[str, deque] = {}
        #: Instrument -> private custom indicators fed by this stream.
        self.customs: dict[str, list] = {}


class _Run:
    def __init__(self, strategy, ctx, symbols, default):
        self.strategy = strategy
        self.ctx = ctx
        self.symbols = tuple(symbols) if symbols else None
        self.default = default
        self.started = False
        #: Once true the strategy gets only the trades, quotes and books it
        #: subscribed to. Legacy strategies never set it.
        self.explicit = False
        self.types: dict[str, set[str]] = {"trade": set(), "quote": set(), "book": set()}
        self.streams: dict[Timeframe, _Stream] = {}
        self.by_id: dict[int, _Stream] = {}
        self.customs_primary: dict[str, list] = {}
        self.quotes: dict[str, Any] = {}
        self.books: dict[str, Any] = {}

    # -- the instrument set of a running node -------------------------------------

    def add_symbol(self, symbol: str) -> None:
        if self.symbols is not None and symbol not in self.symbols:
            self.symbols = (*self.symbols, symbol)

    def drop_symbol(self, symbol: str) -> None:
        if self.symbols is not None:
            self.symbols = tuple(s for s in self.symbols if s != symbol)

    # -- declaration -----------------------------------------------------

    def open(self, what: str) -> None:
        if self.started:
            raise errors.StateError(
                f"{what} must be called from on_init or on_start; the run has already started",
                code="declaration_outside_run",
                operation="strategy",
                parameter=what,
            )

    def instrument(self, value, parameter="instrument") -> str:
        known = self.symbols if self.symbols is not None else (
            (self.default,) if self.default is not None else None
        )
        if not isinstance(value, str) or (known is not None and value not in known):
            _invalid(
                "Unknown instrument.",
                parameter,
                value,
                known if known is not None else ["a symbol name"],
            code="unknown_symbol")
        return value

    def stream(self, tf: Timeframe) -> _Stream:
        info = self.streams.get(tf)
        if info is None:
            sid = self.strategy.subscribe_bars(tf.step, tf.unit, brick_size=tf.brick_size)
            info = self.streams[tf] = self.by_id[sid] = _Stream(sid, tf)
        return info

    def buffered(self, info: _Stream, symbol: str, size: int) -> None:
        old = info.buffers.get(symbol)
        if old is None or (old.maxlen or 0) < size:
            info.buffers[symbol] = deque(old or (), maxlen=size)

    def keep(self, symbol: str, rows: int | None) -> None:
        keep = getattr(self.ctx, "keep_history", None)
        if rows is not None and keep is not None:
            keep(rows, symbol)

    def symbol_of(self, ctx) -> str | None:
        symbol = getattr(ctx, "symbol", None)
        return self.default if symbol is None else symbol

    # -- event feeds (run before the hooks of the same event) --------------

    def feed_composite(self, ctx, bar) -> bool:
        info = self.by_id.get(bar.stream_id)
        if info is None:
            return True  # a subscribe_bars stream: every symbol, as before
        symbol = bar.symbol if bar.symbol is not None else self.default
        buffer = info.buffers.get(symbol)
        if buffer is not None:
            buffer.append(bar)
        for custom in info.customs.get(symbol, ()):
            custom.update(bar)
        return symbol in info.deliver

    def feed_primary(self, ctx, _event) -> bool:
        customs = self.customs_primary.get(self.symbol_of(ctx))
        if customs:
            bar = ctx.bar
            for custom in customs:
                custom.update(bar)
        return True

    def feed_trade(self, ctx, tick) -> bool:
        return tick.symbol in self.types["trade"]

    def feed_quote(self, ctx, quote) -> bool:
        if quote.symbol in self.types["quote"]:
            self.quotes[quote.symbol] = quote
            return True
        return False

    def feed_book(self, ctx, book) -> bool:
        if book.symbol in self.types["book"]:
            self.books[book.symbol] = book
            return True
        return False


def _run_of(strategy) -> _Run:
    run = getattr(strategy, "_api", None)
    if run is None:
        raise errors.StateError(
            "subscribe() and ctx.indicators work during a run: call them from on_init or on_start",
            code="declaration_outside_run",
            operation="strategy",
        )
    return run


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def begin(strategy, ctx, symbols=None, default=None, host=None) -> None:
    """Start a run's API state, attach the views to ``ctx`` and call ``on_init``.

    ``symbols`` are the run's instruments (``None``: unchecked); ``default`` is
    the symbol a single-instrument run reports for ``ctx``. ``host`` is the
    run's :class:`~raptorbt.strategy.context_host.ContextHost`, behind
    ``ctx.instrument`` / ``option_chain`` / ``role``.
    """
    _unbind(strategy)
    run = _Run(strategy, ctx, symbols, default)
    strategy._api = run
    ctx.data = DataView(run)
    ctx.indicators = IndicatorsView(run)
    ctx.portfolio = PortfolioView(run)
    from raptorbt.strategy.context_host import ContextHost

    (host or ContextHost(None, None)).attach(ctx, strategy)
    strategy.on_init(ctx)


def finish(strategy) -> int:
    """Wire the strategy's hooks to the runtime's channels; return the mask.

    Call after ``on_start``. Subscriptions are frozen from here.
    """
    run = getattr(strategy, "_api", None)
    if run is None:
        run = strategy._api = _Run(strategy, None, None, None)
    run.started = True
    _unbind(strategy)
    event_style = _overridden(strategy, "on_bar") and _takes_event(strategy.on_bar, bound=True)
    mask = 0
    for channel, bit in CHANNEL_BITS.items():
        names = _receivers(strategy, channel, event_style)
        feed = _feed(run, channel)
        kind = _KINDS.get(channel)
        if run.explicit and kind is not None and not run.types[kind]:
            continue  # never subscribed: the runtime is not asked for it
        needed = bool(names) or _feed_needed(run, channel)
        if bit and needed:
            mask |= bit
        plain = names == [channel] and feed is None and not (channel == "on_bar" and event_style)
        if not needed or plain:
            continue
        calls = []
        for name in names:
            method = getattr(strategy, name)
            if channel == "on_bar" and name == "on_bar" and not event_style:
                method = _drop_event(method)
            calls.append(method)
        strategy.__dict__[channel] = _dispatcher(channel, calls, feed)
    for name, bit in PLAIN_EVENT_BITS.items():
        if _overridden(strategy, name):
            mask |= bit
    return mask


def _drop_event(method):
    return lambda ctx, _event: method(ctx)


def _receivers(strategy, channel: str, event_style: bool) -> list[str]:
    names = [channel] if _overridden(strategy, channel) else []
    for name, hook in HOOKS.items():
        if name == channel or channel not in hook.channels or not _overridden(strategy, name):
            continue
        if name == "on_bar" and channel == "on_composite_bar" and not event_style:
            continue
        names.append(name)
    return names


def _feed(run: _Run, channel: str):
    if channel == "on_composite_bar":
        return run.feed_composite if run.streams else None
    if channel == "on_bar":
        return run.feed_primary if run.customs_primary else None
    if run.explicit:
        return {
            "on_trade_tick": run.feed_trade,
            "on_quote": run.feed_quote,
            "on_order_book": run.feed_book,
        }.get(channel)
    return None


def _feed_needed(run: _Run, channel: str) -> bool:
    """Whether the runtime must deliver ``channel`` for the feed alone."""
    if channel == "on_composite_bar":
        return any(i.buffers or i.customs for i in run.streams.values())
    if channel == "on_bar":
        return bool(run.customs_primary)
    # ctx.data.quote / book read what the feed recorded.
    kind = _KINDS.get(channel)
    return kind in ("quote", "book") and bool(run.types[kind])


def _dispatcher(channel: str, calls: list, feed):
    if channel == "on_bar":

        def dispatch(ctx):
            if feed is not None:
                feed(ctx, None)
            if calls:
                bar = ctx.bar
                for call in calls:
                    call(ctx, bar)

    else:

        def dispatch(ctx, event):
            if feed is not None and not feed(ctx, event):
                return
            for call in calls:
                call(ctx, event)

    dispatch._raptorbt_dispatcher = True
    return dispatch


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------


def subscribe(strategy, instrument, data_type, timeframe, history):
    """Backs :meth:`Strategy.subscribe`; see there."""
    run = _run_of(strategy)
    run.open("subscribe()")
    if data_type not in _DATA_TYPES:
        _invalid("Invalid data type subscription.", "data_type", data_type, _DATA_TYPES, code="invalid_data_type")
    symbol = run.instrument(instrument)
    tf = resolve_timeframe(timeframe)
    if tf is not None and data_type != "bar":
        _invalid(
            "Invalid timeframe subscription.",
            "timeframe",
            timeframe,
            [None],
        code="invalid_timeframe")
    if history is not None and (
        not isinstance(history, int) or isinstance(history, bool) or history < 1
    ):
        _invalid("Invalid history subscription.", "history", history, ["a positive integer"], code="invalid_history")
    run.explicit = True
    if data_type != "bar":
        run.types[data_type].add(symbol)
        run.keep(symbol, history)
        return None
    if tf is None:
        run.keep(symbol, history)
        return None
    info = run.stream(tf)
    info.deliver.add(symbol)
    run.buffered(info, symbol, history or _DEFAULT_HISTORY)
    return info.id


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

_SERIES = ("timestamps", "open", "high", "low", "close", "volume")


class BarHistory:
    """Read-only arrays of the last closed bars, oldest first.

    ``len(h)`` is the number of bars; ``timestamps`` are the caller's own
    stamps for primary bars and close times for composite bars.
    """

    __slots__ = _SERIES

    def __init__(self, **arrays):
        for name in _SERIES:
            array = arrays[name]
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def __setattr__(self, name, value):
        raise AttributeError("BarHistory is read-only")

    def __len__(self) -> int:
        return len(self.timestamps)

    def __repr__(self) -> str:
        return f"BarHistory({len(self)} bars)"


def _empty() -> BarHistory:
    return BarHistory(
        timestamps=np.empty(0, dtype=np.int64),
        **{k: np.empty(0, dtype=np.float64) for k in _SERIES[1:]},
    )


class DataView:
    """``ctx.data``: what the strategy may know, up to the current event."""

    def __init__(self, run: _Run):
        self._run = run

    def _series(self, symbol: str):
        """Read-only primary arrays up to the latest closed bar, or ``None``."""
        ctx = self._run.ctx
        series = getattr(ctx, "series", None)
        if series is not None:
            data = series(symbol)
            return data if "open" in data else None
        if not self._run.started:
            return None
        n = ctx.idx + 1
        out = {}
        for name in _SERIES:
            view = getattr(ctx, "timestamps" if name == "timestamps" else name)[:n]
            view = view.view()
            view.setflags(write=False)
            out[name] = view
        return out

    def _stream(self, symbol: str, timeframe) -> _Stream:
        tf = resolve_timeframe(timeframe)
        info = self._run.streams.get(tf)
        if info is None or symbol not in info.buffers:
            subscribed = [
                f'{s} {i.tf.label}' for i in self._run.streams.values() for s in i.buffers
            ]
            _invalid(
                "Not subscribed to this timeframe.",
                "timeframe",
                timeframe,
                subscribed or ["call subscribe(instrument, timeframe=...) in on_start"],
            code="not_subscribed_timeframe")
        return info

    def latest(self, instrument: str, timeframe=None):
        """The last closed bar of ``instrument``, or ``None`` before the first.

        Primary bars come back as :class:`~raptorbt.strategy.Bar`, composite
        ones as :class:`~raptorbt.strategy.CompositeBar`. A composite
        ``timeframe`` must have been subscribed to.
        """
        symbol = self._run.instrument(instrument)
        if timeframe is not None:
            buffer = self._stream(symbol, timeframe).buffers[symbol]
            return buffer[-1] if buffer else None
        data = self._series(symbol)
        if data is None or not len(data["timestamps"]):
            return None
        return Bar(*(data[k][-1].item() for k in _SERIES))

    def history(self, instrument: str, timeframe=None, count: int | None = None) -> BarHistory:
        """The last ``count`` closed bars (all that are kept when ``None``).

        Read-only arrays, oldest first. Composite timeframes keep the
        ``history`` the subscription declared (1000 by default).
        """
        symbol = self._run.instrument(instrument)
        if count is not None and (
            not isinstance(count, int) or isinstance(count, bool) or count < 1
        ):
            _invalid("Invalid history request.", "count", count, ["a positive integer"], code="invalid_history")
        if timeframe is not None:
            bars = list(self._stream(symbol, timeframe).buffers[symbol])
            if count is not None:
                bars = bars[-count:]
            if not bars:
                return _empty()
            return BarHistory(
                timestamps=np.array([b.timestamp for b in bars], dtype=np.int64),
                **{k: np.array([getattr(b, k) for b in bars], dtype=np.float64) for k in _SERIES[1:]},
            )
        data = self._series(symbol)
        if data is None:
            return _empty()
        cut = slice(None) if count is None else slice(-count, None)
        return BarHistory(**{k: data[k][cut].view() for k in _SERIES})

    def _recorded(self, kind: str, store: dict, instrument: str):
        symbol = self._run.instrument(instrument)
        if symbol not in self._run.types[kind]:
            _invalid(
                f"Not subscribed to {kind}s.",
                "instrument",
                instrument,
                [f'subscribe({instrument!r}, data_type="{kind}") in on_start'],
            code="not_subscribed_data")
        return store.get(symbol)

    def quote(self, instrument: str):
        """The last quote of ``instrument``, or ``None``. Needs a
        ``data_type="quote"`` subscription."""
        return self._recorded("quote", self._run.quotes, instrument)

    def book(self, instrument: str):
        """The last order book of ``instrument``, or ``None``. Needs a
        ``data_type="book"`` subscription."""
        return self._recorded("book", self._run.books, instrument)

    def position(self, instrument: str):
        """The earliest open position of ``instrument``, or ``None`` when flat."""
        symbol = self._run.instrument(instrument)
        ctx = self._run.ctx
        if hasattr(ctx, "position_for"):
            return ctx.position_for(symbol)
        return ctx.position


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


class IndicatorHandle:
    """A shared streaming indicator: ``.ready`` and ``.value``.

    ``.value`` is ``None`` until ``.ready``. Multi-output indicators put a
    tuple there and expose the same numbers by name on ``.values``.
    """

    def __init__(self, indicator, names: tuple[str, ...] = ()):
        self._indicator = indicator
        self._names = names
        if names:
            self._tuple = NamedTuple("Values", [(n, float) for n in names])

    @property
    def ready(self) -> bool:
        return self._indicator.initialized

    @property
    def value(self):
        return self._indicator.value

    @property
    def values(self):
        if not self._names:
            raise AttributeError("a single-output indicator has .value, not .values")
        value = self._indicator.value
        return None if value is None else self._tuple(*value)

    def __repr__(self) -> str:
        return f"IndicatorHandle({self._indicator.kind}, ready={self.ready})"


class IndicatorsView:
    """``ctx.indicators``: streaming indicators the runtime updates and shares.

    Every factory takes ``instrument=`` (required), ``timeframe=`` (``None``:
    the primary bars) and ``field=`` (only ``"close"``; ATR and Donchian read
    the whole bar). Create them in ``on_init`` or ``on_start``.
    """

    def __init__(self, run: _Run):
        self._run = run

    def _make(self, indicator, instrument, timeframe, field, names=()):
        run = self._run
        run.open("ctx.indicators")
        symbol = run.instrument(instrument)
        if field != "close":
            _invalid("Invalid indicator input.", "field", field, ["close"], code="invalid_indicator_input")
        tf = resolve_timeframe(timeframe)
        stream_id = None if tf is None else run.stream(tf).id
        run.strategy.register_indicator(indicator, stream_id=stream_id, symbol=symbol)
        return IndicatorHandle(indicator, names)

    def sma(self, *, instrument, period, timeframe=None, field="close"):
        return self._make(Indicator.sma(period), instrument, timeframe, field)

    def ema(self, *, instrument, period, timeframe=None, field="close"):
        return self._make(Indicator.ema(period), instrument, timeframe, field)

    def wma(self, *, instrument, period, timeframe=None, field="close"):
        return self._make(Indicator.wma(period), instrument, timeframe, field)

    def rsi(self, *, instrument, period, timeframe=None, field="close"):
        return self._make(Indicator.rsi(period), instrument, timeframe, field)

    def roc(self, *, instrument, period, timeframe=None, field="close"):
        return self._make(Indicator.roc(period), instrument, timeframe, field)

    def stddev(self, *, instrument, period, timeframe=None, field="close"):
        return self._make(Indicator.stddev(period), instrument, timeframe, field)

    def atr(self, *, instrument, period, timeframe=None, field="close"):
        return self._make(Indicator.atr(period), instrument, timeframe, field)

    def donchian(self, *, instrument, period, timeframe=None, field="close"):
        """Channel over ``period`` bars; ``.values`` is ``(upper, lower)``."""
        return self._make(
            Indicator.donchian(period), instrument, timeframe, field, ("upper", "lower")
        )

    def bollinger(self, *, instrument, period, k=2.0, timeframe=None, field="close"):
        """Bands ``k`` deviations out; ``.values`` is ``(upper, middle, lower)``."""
        return self._make(
            Indicator.bollinger(period, k), instrument, timeframe, field,
            ("upper", "middle", "lower"),
        )

    def macd(self, *, instrument, fast=12, slow=26, signal=9, timeframe=None, field="close"):
        """``.values`` is ``(macd, signal, histogram)``."""
        return self._make(
            Indicator.macd(fast, slow, signal), instrument, timeframe, field,
            ("macd", "signal", "histogram"),
        )

    def custom(self, obj, *, instrument, timeframe=None):
        """Feed your own indicator the closed bars of ``instrument``.

        ``obj`` needs ``update(bar)``, called once per closed bar (a
        :class:`~raptorbt.strategy.Bar` for the primary bars, a
        :class:`~raptorbt.strategy.CompositeBar` for a timeframe) *before* your
        hooks see that bar, and the attributes ``ready`` (``bool``) and
        ``value``. It is private to this strategy and never shared or linked.
        Returns ``obj``.
        """
        run = self._run
        run.open("ctx.indicators.custom")
        missing = [
            n for n in ("update", "ready", "value")
            if not hasattr(obj, n) or (n == "update" and not callable(obj.update))
        ]
        if missing:
            raise TypeError(
                "a custom indicator needs update(bar), ready and value; "
                f"{type(obj).__name__} lacks {', '.join(missing)}"
            )
        symbol = run.instrument(instrument)
        tf = resolve_timeframe(timeframe)
        if tf is None:
            run.customs_primary.setdefault(symbol, []).append(obj)
        else:
            run.stream(tf).customs.setdefault(symbol, []).append(obj)
        return obj


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


class PortfolioView:
    """``ctx.portfolio``: this strategy's own book (not the account).

    ``available_capital`` is the cash that can fund a new position; in a cash
    account that is all of ``cash``. Exposures are notional (price * size *
    contract multiplier) at each instrument's last mark: ``gross`` unsigned,
    ``net`` long minus short. The totals need the shared runtime
    (``run_strategies`` and the runners built on it).
    """

    def __init__(self, run: _Run):
        self._run = run

    def _totals(self) -> tuple:
        totals = getattr(self._run.ctx, "_portfolio_totals", None)
        if totals is None:
            raise errors.UnsupportedOperation(
                "this portfolio total is not available in run_strategy_backtest; "
                "use run_strategies or run_portfolio_strategy",
                code="portfolio_total_unavailable",
                operation="run_strategy_backtest",
            )
        return totals()

    @property
    def positions(self) -> dict:
        """Open positions by instrument: ``{symbol: (PositionSnapshot, ...)}``."""
        ctx = self._run.ctx
        if hasattr(ctx, "symbols"):
            out = {s: tuple(ctx.positions(s)) for s in ctx.symbols}
        else:
            out = {self._run.default: tuple(ctx.positions())}
        return {s: p for s, p in out.items() if p}

    @property
    def cash(self) -> float:
        return self._run.ctx.cash

    @property
    def equity(self) -> float:
        return self._run.ctx.equity

    @property
    def available_capital(self) -> float:
        return self._run.ctx.free_capital

    @property
    def halted(self) -> bool:
        """Whether a risk limit or ``ctx.halt`` stopped this strategy trading."""
        ctx = self._run.ctx
        return ctx.is_halted() if hasattr(ctx, "is_halted") else ctx._session.is_halted()

    @property
    def locked_margin(self) -> float:
        return self._totals()[0]

    @property
    def realized_pnl(self) -> float:
        return self._totals()[1]

    @property
    def unrealized_pnl(self) -> float:
        return self._totals()[2]

    @property
    def gross_exposure(self) -> float:
        return self._totals()[3]

    @property
    def net_exposure(self) -> float:
        return self._totals()[4]
