"""Structured errors for RaptorBT.

Every error the library raises on purpose says what failed, which parameter
was wrong, what would have been accepted, and which documentation domain
covers it. The same facts are on the exception as attributes, so a program
(or an AI agent) can act on them without parsing prose.

Each class also subclasses the builtin exception it replaces, so existing
``except ValueError`` / ``except RuntimeError`` handlers keep working, and the
first line of ``str(error)`` is the short message older versions raised.

>>> try:
...     raise InvalidParameter(
...         "bar_stamp must be 'open' or 'close'",
...         code="invalid_bar_stamp", parameter="bar_stamp", value="start",
...     )
... except ValueError as e:
...     e.code, e.valid, e.skill
('invalid_bar_stamp', ['open', 'close', None], 'timeframes')

The registry :data:`CODES` is the single list of codes. ``docs/errors.md`` is
generated from it, and a test checks that every code raised in the source is
registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Documentation domains an error can point at (``raptorbt-<domain>``).
SKILLS = (
    "core",
    "strategy",
    "instruments",
    "market-data",
    "subscriptions",
    "timeframes",
    "indicators",
    "orders",
    "execution",
    "portfolio",
    "risk",
    "backtesting",
    "optimization",
    "live-trading",
    "results",
    "deployment",
    "troubleshooting",
)

_UNSET: Any = object()


class RaptorBTError(Exception):
    """Base class of every structured RaptorBT error.

    Attributes:
        code: Stable snake_case identifier, listed in :data:`CODES`.
        operation: The call or step that failed (e.g. ``"run_strategies"``).
        parameter: The argument or state that was invalid.
        value: What was supplied.
        valid: Accepted alternatives, as a list, or ``None`` if not a finite set.
        skill: Documentation domain, one of :data:`SKILLS`.
        message: The short one-line message (``str(error)`` starts with it).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        operation: str | None = None,
        parameter: str | None = None,
        value: Any = _UNSET,
        valid: Any = None,
        skill: str | None = None,
    ):
        super().__init__(message)
        entry = CODES.get(code)
        self.message = message
        self.code = code
        self.operation = operation
        self.parameter = parameter
        self.has_value = value is not _UNSET
        self.value = None if value is _UNSET else value
        if valid is None and entry is not None:
            valid = entry.valid
        self.valid = None if valid is None else list(valid)
        if skill is None and entry is not None:
            skill = entry.skill
        if skill is not None and skill not in SKILLS:
            raise ValueError(f"skill must be one of {SKILLS}, got {skill!r}")
        self.skill = skill

    def __str__(self) -> str:
        blocks = [self.message]
        if self.operation:
            blocks.append(f"Operation:\n    {self.operation}")
        if self.parameter is not None or self.has_value:
            name = self.parameter if self.parameter is not None else "value"
            line = f"{name} = {self.value!r}" if self.has_value else name
            blocks.append(f"Requested:\n    {line}")
        if self.valid:
            blocks.append("Supported:\n    " + ", ".join(repr(v) for v in self.valid))
        if self.skill:
            blocks.append(f"Related skill:\n    raptorbt-{self.skill}")
        return "\n\n".join(blocks)

    def __reduce__(self):
        return (_rebuild, (type(self), self.to_dict()))

    def to_dict(self) -> dict:
        """The error as plain JSON-ready data; see :func:`from_dict`."""
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "operation": self.operation,
            "parameter": self.parameter,
            "value": _plain(self.value) if self.has_value else None,
            "has_value": self.has_value,
            "valid": None if self.valid is None else [_plain(v) for v in self.valid],
            "skill": self.skill,
        }


class InvalidParameter(RaptorBTError, ValueError):
    """An argument has a value the call does not accept."""


class InvalidData(RaptorBTError, ValueError):
    """Input data is malformed: lengths, order, NaN, missing fields."""


class InvalidOrder(RaptorBTError, ValueError):
    """An order is malformed or its options contradict each other."""


class InvalidSubscription(RaptorBTError, ValueError):
    """A bar or tick subscription cannot be built."""


class UnsupportedOperation(RaptorBTError, ValueError):
    """The request is well formed but this engine mode cannot do it."""


class StateError(RaptorBTError, RuntimeError):
    """The object is in the wrong state for the call (e.g. already finished)."""


class SessionFinished(StateError, ValueError):
    """A finished session was used again.

    Sessions raised ``ValueError`` here before structured errors existed, so
    this class is both a ``StateError`` and a ``ValueError``.
    """


class UnknownSymbol(RaptorBTError, KeyError, ValueError):
    """A symbol was named that the run does not know."""


def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return repr(v)


def _rebuild(cls, data: dict):
    return from_dict(data, cls)


def from_dict(data: dict, cls: type | None = None) -> RaptorBTError:
    """Rebuild an error from :meth:`RaptorBTError.to_dict` output."""
    cls = cls or globals().get(data["type"]) or error_class(data["code"])
    return cls(
        data["message"],
        code=data["code"],
        operation=data["operation"],
        parameter=data["parameter"],
        value=data["value"] if data["has_value"] else _UNSET,
        valid=data["valid"],
        skill=data["skill"],
    )


@dataclass(frozen=True)
class Code:
    """One registered error code."""

    cls: type
    skill: str
    meaning: str
    valid: tuple | None = None


def _c(cls, skill, meaning, valid=None) -> Code:
    return Code(cls, skill, meaning, None if valid is None else tuple(valid))


#: Every error code, with its class, domain, meaning and default alternatives.
CODES: dict[str, Code] = {
    # Timeframes and bar stamps
    "invalid_bar_stamp": _c(
        InvalidParameter, "timeframes", "bar_stamp is not 'open', 'close' or None.",
        ["open", "close", None],
    ),
    "missing_bar_ns": _c(
        InvalidParameter, "timeframes", "bar_stamp='open' was given without the bar width bar_ns."
    ),
    "invalid_bar_ns": _c(
        InvalidParameter, "timeframes", "bar_ns is not a positive whole number of nanoseconds."
    ),
    "invalid_primary_bars": _c(
        InvalidSubscription, "timeframes", "primary_bars is not a valid (step, unit) pair.",
    ),
    "invalid_bar_spec": _c(
        InvalidSubscription, "timeframes", "A bar step or unit is not valid.",
    ),
    "invalid_bar_step": _c(
        InvalidSubscription, "timeframes", "A bar subscription step is below 1.",
    ),
    "primary_bars_without_ticks": _c(
        InvalidSubscription, "subscriptions", "primary_bars was given but there are no ticks to aggregate."
    ),
    # Strategies and runs
    "invalid_strategy": _c(
        InvalidParameter, "strategy", "A strategy entry is not a Strategy instance or subclass."
    ),
    "duplicate_strategy": _c(
        InvalidParameter, "strategy", "The same strategy instance was passed twice in one call."
    ),
    "empty_strategies": _c(
        InvalidParameter, "strategy", "The strategy list is empty."
    ),
    "invalid_strategy_entry": _c(
        InvalidParameter, "strategy", "A strategy entry is a tuple that is not (strategy, symbols)."
    ),
    "invalid_accounts": _c(
        InvalidParameter, "portfolio", "accounts is not a dict of account name -> AccountSpec."
    ),
    "invalid_account_spec": _c(
        InvalidParameter, "portfolio", "An AccountSpec field has a value the account does not accept."
    ),
    "unknown_account": _c(
        InvalidParameter, "portfolio", "A strategy or call names an account that is not in accounts."
    ),
    "account_too_large": _c(
        InvalidParameter, "portfolio", "More strategies share one account than a group may hold."
    ),
    "invalid_batch_size": _c(
        InvalidParameter, "backtesting", "batch_size is not a positive integer or None."
    ),
    "invalid_history": _c(
        InvalidParameter, "live-trading", "history is not a positive number of rows."
    ),
    "invalid_curve_samples": _c(
        InvalidParameter, "live-trading", "curve_samples is not a positive integer or None."
    ),
    "invalid_initial_positions": _c(
        InvalidParameter, "live-trading", "initial_positions names a strategy or position that does not fit the node."
    ),
    "no_symbols": _c(
        InvalidParameter, "market-data", "No symbols were given, or the data/ticks dict is empty."
    ),
    "duplicate_symbols": _c(
        InvalidParameter, "market-data", "A symbol appears more than once."
    ),
    "unknown_symbol": _c(
        UnknownSymbol, "market-data", "A symbol was named that has no data in this run."
    ),
    # Strategy API: subscriptions, data views, indicators
    "invalid_timeframe": _c(
        InvalidSubscription, "timeframes", "A timeframe is not one of the supported strings or a valid bar spec.",
        ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"],
    ),
    "invalid_data_type": _c(
        InvalidSubscription, "subscriptions", "subscribe() was given a data_type it does not know.",
        ["bar", "trade", "quote", "book"],
    ),
    "invalid_history": _c(
        InvalidParameter, "subscriptions", "A history or count is not a positive integer."
    ),
    "not_subscribed_timeframe": _c(
        InvalidSubscription, "subscriptions", "ctx.data was asked for a timeframe the strategy did not subscribe to."
    ),
    "not_subscribed_data": _c(
        InvalidSubscription, "subscriptions", "ctx.data was asked for quotes or books the strategy did not subscribe to."
    ),
    "invalid_indicator_input": _c(
        InvalidParameter, "indicators", "An indicator input field other than the close was requested.",
        ["close"],
    ),
    "declaration_outside_run": _c(
        StateError, "subscriptions", "subscribe() or ctx.indicators was used outside on_init / on_start."
    ),
    "portfolio_total_unavailable": _c(
        UnsupportedOperation, "portfolio", "A ctx.portfolio total needs the shared runtime (run_strategies)."
    ),
    # Input data
    "invalid_input_type": _c(
        InvalidData, "market-data", "A data, ticks or depth argument is not a dict of arrays."
    ),
    "missing_field": _c(
        InvalidData, "market-data", "A required field is missing from a symbol's arrays."
    ),
    "invalid_array": _c(
        InvalidData, "market-data", "A field is not a numeric array of the required dimension."
    ),
    "length_mismatch": _c(
        InvalidData, "market-data", "Arrays that must share one length do not."
    ),
    "decreasing_timestamps": _c(
        InvalidData, "market-data", "Timestamps go backwards; bars must be in time order."
    ),
    "empty_data": _c(
        InvalidData, "market-data", "There is no data to run on (zero bars or rows)."
    ),
    "depth_without_ticks": _c(
        InvalidData, "market-data", "depth was given for a symbol that has no ticks."
    ),
    "insufficient_data": _c(
        InvalidData, "market-data", "Fewer elements than the calculation needs."
    ),
    "invalid_data": _c(
        InvalidData, "market-data", "Other malformed input data (division by zero, index out of range, conversion)."
    ),
    # Orders
    "invalid_order_side": _c(
        InvalidOrder, "orders", "Order side is not 'buy' or 'sell'.", ["buy", "sell"]
    ),
    "invalid_time_in_force": _c(
        InvalidOrder, "orders", "Time in force is not recognised.",
        ["gtc", "day", "gtd", "ioc", "fok", "at_open", "at_close"],
    ),
    "invalid_order_kind": _c(
        InvalidOrder, "orders", "Order kind is not recognised.",
        [
            "market", "limit", "stop_market", "stop_limit", "market_if_touched",
            "limit_if_touched", "market_to_limit", "trailing_stop_market", "trailing_stop_limit",
        ],
    ),
    "invalid_offset_kind": _c(
        InvalidOrder, "orders", "Trailing offset kind is not recognised.", ["price", "bps", "ticks"]
    ),
    "invalid_order_value": _c(
        InvalidOrder, "orders", "A price, size, offset or count is out of range."
    ),
    "missing_order_field": _c(
        InvalidOrder, "orders", "A field this order kind or time in force needs was not given."
    ),
    "conflicting_order_args": _c(
        InvalidOrder, "orders", "Two options that exclude each other were both given (e.g. units and size_frac)."
    ),
    "incompatible_order_options": _c(
        InvalidOrder, "orders", "A time-in-force or flag does not apply to this order kind."
    ),
    "invalid_order_link": _c(
        InvalidOrder, "orders", "A parent or one-cancels-other link cannot be made."
    ),
    "invalid_order_intent": _c(
        InvalidOrder, "orders", "Intents queued on one bar are duplicated or unknown."
    ),
    "invalid_order_reference": _c(
        InvalidOrder, "orders", "An order or position id does not exist."
    ),
    # Account and execution
    "invalid_account_type": _c(
        InvalidParameter, "execution", "account_type is not recognised.", ["cash", "margin"]
    ),
    "invalid_leverage": _c(
        InvalidParameter, "execution", "Margin accounts need leverage > 0."
    ),
    "invalid_oms_type": _c(
        InvalidParameter, "execution", "oms_type is not recognised.", ["netting", "hedging"]
    ),
    "invalid_direction": _c(
        InvalidParameter, "backtesting", "direction is not 1 (long) or -1 (short).", [1, -1]
    ),
    "invalid_fill_timing": _c(
        InvalidParameter, "execution", "fill_timing is not recognised.",
        ["same_bar_close", "next_bar_open", "same_bar_open_lookahead"],
    ),
    "invalid_allocation": _c(
        InvalidParameter, "portfolio", "allocation is not recognised.", ["equal_weight", "full"]
    ),
    "invalid_option_type": _c(
        InvalidParameter, "backtesting", "option_type is not recognised.",
        ["call", "ce", "c", "put", "pe", "p"],
    ),
    "invalid_strike_selection": _c(
        InvalidParameter, "backtesting", "strike_selection is not recognised.",
        ["atm", "otm1", "otm2", "itm1", "itm2"],
    ),
    "invalid_size_type": _c(
        InvalidParameter, "backtesting", "size_type is not recognised.",
        ["percent", "contracts", "notional", "risk"],
    ),
    "unsupported_short_ticks": _c(
        UnsupportedOperation, "backtesting", "Tick backtests are long-only; quantity must be >= 0."
    ),
    "invalid_squareoff_time": _c(
        InvalidParameter, "execution", "squareoff_time is not a 24-hour 'HH:MM' string or None."
    ),
    "untradable_instrument": _c(
        UnsupportedOperation, "instruments", "The instrument kind cannot be traded."
    ),
    "invalid_session_schedule": _c(
        InvalidParameter, "instruments",
        "A SessionSchedule is invalid: a bad window, weekday, holiday date, offset or square-off minute.",
    ),
    "invalid_fee_schedule": _c(
        InvalidParameter, "execution",
        "A fee schedule or rate card document is invalid, or fee_schedule is set without a fee_segment.",
    ),
    "fee_schedule_gap": _c(
        InvalidParameter, "execution",
        "A fee schedule has no row for the fill's venue and segment; the run is refused rather than charged nothing.",
    ),
    "invalid_margin_model": _c(
        InvalidParameter, "risk",
        "A margin model or the line it prices is invalid: an unknown name, a missing rule for an instrument type, or legs that do not fit.",
    ),
    "invalid_session_date": _c(
        InvalidParameter, "instruments", "A session date is not a real yyyymmdd date."
    ),
    "invalid_instrument_metadata": _c(
        InvalidParameter, "instruments", "An instrument's venue or currency is an empty string."
    ),
    "invalid_parameter": _c(
        InvalidParameter, "core", "A parameter value is invalid (engine-side check)."
    ),
    "invalid_config": _c(
        InvalidParameter, "core", "A configuration value is invalid (engine-side check)."
    ),
    # Lifecycle
    "session_finished": _c(
        SessionFinished, "backtesting", "The session is finished; create a new one."
    ),
    "node_finished": _c(
        StateError, "live-trading", "The live node is finished; create a new one."
    ),
    "runtime_busy": _c(
        StateError, "backtesting", "The runtime is in use; a call re-entered it from a hook."
    ),
    "runtime_not_ready": _c(
        StateError, "backtesting", "A call was made before the runtime was finalized (or after it was built)."
    ),
    "run_finished": _c(
        StateError, "backtesting", "The strategy's run is finished."
    ),
}


def error_class(code: str) -> type:
    """The exception class registered for ``code``."""
    return CODES[code].cls


def make(code: str, message: str, **detail) -> RaptorBTError:
    """Build the error registered for ``code``. Used by the native extension."""
    return CODES[code].cls(message, code=code, **detail)


def markdown() -> str:
    """The body of ``docs/errors.md``, generated from :data:`CODES`."""
    lines = [
        "# Errors",
        "",
        "<!-- Generated from raptorbt.errors.CODES: regenerate with `python -c 'import raptorbt.errors as e; print(e.markdown(), end="")' > docs/errors.md`. -->",
        "",
        "Every error RaptorBT raises on purpose is a `raptorbt.errors.RaptorBTError`.",
        "It carries `code`, `operation`, `parameter`, `value`, `valid` (the accepted",
        "alternatives, or `None`), `skill` (the documentation domain) and `to_dict()`.",
        "The first line of `str(error)` is the short message; the lines after it list",
        "what was requested, what is supported and the related skill.",
        "",
        "Each class also subclasses the builtin it replaced, so `except ValueError`",
        "(or `RuntimeError`, `KeyError`) keeps working.",
        "",
        "| Class | Also a |",
        "| --- | --- |",
    ]
    for cls in (
        InvalidParameter, InvalidData, InvalidOrder, InvalidSubscription,
        UnsupportedOperation, StateError, SessionFinished, UnknownSymbol,
    ):
        skip = (object, Exception, BaseException)
        builtins = ", ".join(
            f"`{b.__name__}`" for b in cls.__mro__[1:] if b.__module__ == "builtins" and b not in skip
        )
        lines.append(f"| `{cls.__name__}` | {builtins} |")
    lines += ["", "## Codes", "", "| Code | Class | Meaning | Alternatives | Related skill |", "| --- | --- | --- | --- | --- |"]
    for code in sorted(CODES):
        c = CODES[code]
        valid = ", ".join(f"`{v!r}`" for v in c.valid) if c.valid else "depends on the call"
        lines.append(f"| `{code}` | `{c.cls.__name__}` | {c.meaning} | {valid} | `raptorbt-{c.skill}` |")
    return "\n".join(lines) + "\n"


__all__ = [
    "CODES",
    "SKILLS",
    "Code",
    "InvalidData",
    "InvalidOrder",
    "InvalidParameter",
    "InvalidSubscription",
    "RaptorBTError",
    "SessionFinished",
    "StateError",
    "UnknownSymbol",
    "UnsupportedOperation",
    "error_class",
    "from_dict",
    "make",
    "markdown",
]
