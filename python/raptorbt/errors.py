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


class MoneyOverflow(RaptorBTError, ArithmeticError):
    """A booked amount or balance cannot be represented.

    Cash, fees and P&L are integer minor units (``BacktestConfig.money_decimals``);
    an amount that is not finite, or a sum past the range of the integer, stops
    the run with this error rather than wrapping or going quietly wrong."""


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
    "invalid_input": _c(
        InvalidParameter, "core", "An argument or array was refused by the computation it was passed to."
    ),
    "adoption_refused": _c(
        InvalidParameter, "live-trading",
        "A position cannot be adopted as given: a short on a cash account, a posted margin on a cash account or a cost basis on a margin account.",
    ),
    "history_exhausted": _c(
        StateError, "live-trading",
        "A live node's history cannot hold a row: the ring is full of rows a strategy asked to keep, or the row was overwritten.",
    ),
    "not_live_node": _c(
        UnsupportedOperation, "live-trading", "A live-node call was made on a runtime that is not a live node."
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
    "misaligned_legs": _c(
        InvalidData, "market-data",
        "Legs traded bar by bar (a basket, a pair) do not share their timestamps.",
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
    "non_finite_price": _c(
        InvalidData, "market-data",
        "A bar's open, high, low or close is NaN or infinite; the message names the column and first index.",
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
        ["auto", "gtc", "day", "gtd", "ioc", "fok", "at_open", "at_close"],
    ),
    "order_validity_unsupported": _c(
        InvalidOrder, "orders",
        "A gtc or gtd order on an instrument whose exchange segment holds only day orders (NSE equity derivatives, currency, BSE, MCX), or a gtd dated past the contract's maturity where the segment holds them, or a day order on an instrument whose venue no session preset covers and that has no session.",
        ["day", "ioc", "fok"],
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
    "invalid_roll_sizing": _c(
        InvalidParameter, "instruments", "roll_sizing is not 'notional' or 'lots'.", ["notional", "lots"]
    ),
    "roll_price_missing": _c(
        StateError, "instruments",
        "An auto-roll sized by notional could not price the new contract: it has no bar yet.",
    ),
    "off_lot_quantity": _c(
        InvalidOrder, "orders",
        "An order's quantity is not a whole number of the instrument's lots (or size steps). It is "
        "refused in every mode so the simulation and a venue see the same terms; `valid` lists the "
        "nearest valid quantities.",
    ),
    "off_tick_price": _c(
        InvalidOrder, "orders",
        "A limit, trigger, stop or target price is not a whole number of the instrument's ticks. It "
        "is refused in every mode so the simulation and a venue see the same terms; `valid` lists "
        "the nearest valid prices.",
    ),
    "unknown_order": _c(
        InvalidOrder, "orders",
        "A cancel or modify named a client id this run never submitted. A submitted order that "
        "has finished is a documented no-op, not this error.",
    ),
    "bracket_under_hedging": _c(
        InvalidOrder, "orders",
        "A reduce-only child order (a bracket's stop or target) was submitted under hedging, "
        "where every order opens its own position and a reduce-only order is always refused.",
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
    "invalid_combine_mode": _c(
        InvalidParameter, "backtesting", "combine_mode is not recognised.",
        ["any", "all", "majority", "independent", "weighted"],
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
    "invalid_corporate_action": _c(
        InvalidParameter, "execution",
        "A corporate action is invalid: an unknown kind, a ratio that is not positive, a negative dividend, or a pay date before the ex-date.",
    ),
    "invalid_adjustment_policy": _c(
        InvalidParameter, "execution",
        "Corporate actions were given without an adjustment_policy, or the policy is not raw or adjusted.",
    ),
    "unsupported_corporate_actions": _c(
        UnsupportedOperation, "execution",
        "corporate_actions were given to a runner that cannot apply them; they are refused, never ignored.",
        ["run_single_backtest", "run_strategy_backtest", "run_strategies"],
    ),
    "invalid_settlement": _c(
        InvalidParameter, "instruments",
        "An instrument's settlement style or settlement price is invalid.",
    ),
    "invalid_fee_schedule": _c(
        InvalidParameter, "execution",
        "A fee schedule or rate card document is invalid, or fee_segment names a row the schedule does not have.",
    ),
    "fee_schedule_gap": _c(
        InvalidParameter, "execution",
        "A fee schedule has no row for an instrument's or fill's venue and segment, or the instrument has no InstrumentSpec venue to choose a row from, or a fill or settlement falls on a date before the schedule's first row (the error names the earliest date); the run is refused rather than charged nothing or charged a rate that was not in force.",
    ),
    "squareoff_needs_timezone": _c(
        InvalidParameter, "execution",
        "config.squareoff_time is a local time of day but the instrument has no session schedule and the config names no session_tz_offset_ns.",
    ),
    "mixed_holding": _c(
        InvalidParameter, "execution",
        "Strategies in one run declare different holding requirements (intraday with different flatten times, or intraday beside overnight), so no single square-off time can be derived; set config.squareoff_time or run them apart.",
    ),
    "session_not_covered": _c(
        InvalidData, "instruments",
        "A bar falls on a date outside those a session schedule covers (the MCX preset, or a schedule loaded from calendar documents), so whether the market traded is unknown, not closed; the date and the covered ranges are named.",
    ),
    "invalid_calendar": _c(
        InvalidParameter, "instruments",
        "A calendar document given to SessionSchedule.from_calendars is not valid JSON in the documented format, names no entry for the venue, or has overlapping or out-of-range windows or dates.",
    ),
    "invalid_margin_model": _c(
        InvalidParameter, "risk",
        "A margin model or the line it prices is invalid: an unknown name, a missing rule for an instrument type, legs that do not fit, or a config.margin_model given to a cash account.",
    ),
    "money_overflow": _c(
        MoneyOverflow, "execution",
        "A booked amount is not finite or a balance or sum is past the integer range of minor units; the run stops instead of wrapping.",
    ),
    "invalid_money_decimals": _c(
        InvalidParameter, "execution",
        "config.money_decimals is outside 0..=9: the decimals cash, fees and realised P&L are booked in.",
    ),
    "invalid_session_date": _c(
        InvalidParameter, "instruments", "A session date is not a real yyyymmdd date."
    ),
    "invalid_instrument_metadata": _c(
        InvalidParameter, "instruments", "An instrument's venue or currency is an empty string."
    ),
    "mcx_session_unknown": _c(
        InvalidParameter, "instruments",
        "An MCX instrument with no session names no known contract class (select agricultural COTTON, KAPAS, COTWASOIL at 09:00-21:00; other agricultural at 09:00-17:00; bullion, base metals, energy), so no session is implied: pass session=.",
    ),
    "invalid_option_right": _c(
        InvalidParameter, "instruments", "An option's right is not call or put.", ["call", "put"]
    ),
    "invalid_expiry_pick": _c(
        InvalidParameter, "instruments", "OptionChain.pick_expiry was given an unknown rule, or 'date' without date_ns.",
        ["nearest", "nth", "weekly", "monthly", "date"],
    ),
    "invalid_parameter": _c(
        InvalidParameter, "core", "A parameter value is invalid (engine-side check)."
    ),
    "invalid_config": _c(
        InvalidParameter, "core", "A configuration value is invalid (engine-side check)."
    ),
    # Option structures and sizing
    "invalid_structure": _c(
        InvalidParameter, "instruments", "A structure, leg, selector, exit or submission argument is invalid."
    ),
    "structure_contract_missing": _c(
        InvalidParameter, "instruments", "The chain lists no contract (or expiry) a structure's leg selects at the event time."
    ),
    "structure_limit_breached": _c(
        InvalidParameter, "orders", "The legs' prices make the net premium worse than net_limit."
    ),
    "structure_margin_exceeded": _c(
        InvalidParameter, "risk", "The margin model needs more for the group than is available."
    ),
    "structure_missing_mark": _c(
        StateError, "instruments", "A price or mark a structure needs was not supplied."
    ),
    "below_one_lot": _c(
        InvalidParameter, "risk", "The sizing rule yields less than one lot; nothing is rounded up or to zero."
    ),
    "invalid_sizing_input": _c(
        InvalidParameter, "risk", "A sizing input is not a positive finite number (or a fraction out of range)."
    ),
    "invalid_order_tags": _c(
        InvalidOrder, "orders",
        "An order's tags are not a dict of at most 16 str pairs (keys 1-64 bytes, values up to 256), or use the reserved key 'client_id'.",
    ),
    "invalid_order_check": _c(
        InvalidParameter, "live-trading", "order_check is not callable."
    ),
    "invalid_order_refusal": _c(
        InvalidParameter, "live-trading",
        "order_check returned something other than None, a code, or an object with a str code; or the code is empty, over 64 bytes, or the 1025th distinct one.",
    ),
    "order_check_intent": _c(
        InvalidOrder, "live-trading",
        "enter() or close_position() was used with an order_check installed; they carry no order for it to see.",
    ),
    "unknown_role_binding": _c(
        InvalidParameter, "live-trading", "bindings names a role that no strategy declares."
    ),
    "no_instrument_spec": _c(
        UnknownSymbol, "instruments", "ctx.instrument was asked for a symbol with no InstrumentSpec."
    ),
    "unresolved_role": _c(
        UnknownSymbol, "instruments", "ctx.role could not resolve a requirements role at this time."
    ),
    "invalid_adjust": _c(
        InvalidParameter, "market-data", "continuous_series needs adjust to be 'raw', 'back' or 'ratio'."
    ),
    "no_continuous_bars": _c(
        InvalidData, "market-data", "A continuous series has no bars in a contract's window, or an older contract has none before a roll."
    ),
    "roll_target_not_subscribed": _c(
        InvalidSubscription, "live-trading", "A continuous role rolled to a contract that is not a symbol of the run."
    ),
    "invalid_exit_rule": _c(
        InvalidParameter, "live-trading",
        "An exit rule is empty, has a non-positive or non-finite threshold, names no or several targets, or fired with no position.",
    ),
    "unknown_role": _c(
        UnknownSymbol, "instruments", "ctx.role was asked for a role the strategy's requirements do not declare."
    ),
    "no_requirements": _c(
        InvalidParameter, "instruments", "ctx.role was used by a strategy that declares no requirements."
    ),
    "invalid_warmup": _c(
        InvalidParameter, "live-trading", "warmup is not 'live', 'muted' or None."
    ),
    # Lifecycle
    "session_finished": _c(
        SessionFinished, "backtesting", "The session is finished; create a new one."
    ),
    "node_finished": _c(
        StateError, "live-trading", "The live node is finished; create a new one."
    ),
    # Execution adapters and venues
    "invalid_execution": _c(
        InvalidParameter, "live-trading",
        "execution is not an execution adapter, or the run cannot send its orders to one.",
    ),
    "unsupported_on_bars": _c(
        UnsupportedOperation, "core",
        "A setting that only the tick path implements (order_latency_ns, partial_fills) was given "
        "to a run that has bars; on bars it would change nothing.",
    ),
    "unsupported_in_venue_run": _c(
        UnsupportedOperation, "live-trading",
        "The request has no equivalent when orders go to an execution adapter.",
    ),
    "venue_requires_units": _c(
        InvalidOrder, "live-trading",
        "An order sent to an execution adapter gave its size as a capital fraction or the whole position.",
    ),
    "duplicate_client_order_id": _c(
        InvalidOrder, "live-trading", "An order id was reused for a different order."
    ),
    "unknown_client_order_id": _c(
        InvalidParameter, "live-trading", "A report names an order that was never sent."
    ),
    "invalid_execution_report": _c(
        InvalidData, "live-trading",
        "A venue report contradicts the order it is about, or cannot be applied to it.",
    ),
    "venue_fill_not_booked": _c(
        StateError, "live-trading",
        "The venue filled an order but the book could not take the fill: reconcile the position.",
    ),
    "venue_unavailable": _c(
        StateError, "live-trading",
        "A request may not have reached the venue; the order stays as it was until the venue reports.",
    ),
    "journal_failed": _c(
        StateError, "live-trading",
        "The journal could not take a record, so the node stopped before acting on it.",
    ),
    "journal_evicted": _c(
        StateError, "live-trading",
        "A journal read asked for records a bounded memory has dropped; read them from the durable sink.",
    ),
    "replay_diverged": _c(
        StateError, "live-trading",
        "A replay produced a record that is not the one the journal holds: the node is not the one that wrote it, or a strategy is not deterministic.",
    ),
    "invalid_checkpoint": _c(
        InvalidData, "live-trading",
        "The checkpoint or journal records cannot be used to rebuild the node.",
    ),
    "instrument_in_use": _c(
        StateError, "live-trading",
        "An instrument cannot be retired while a strategy holds a position or a working order on it.",
    ),
    "instrument_retired": _c(
        StateError, "live-trading",
        "The instrument was already retired from the node.",
    ),
    "group_open": _c(
        StateError, "live-trading",
        "A timestamp group is open: flush() it before opening another, reconciling, resuming, reading health, taking a checkpoint or a state digest.",
    ),
    "invalid_strategies": _c(
        InvalidParameter, "live-trading",
        "strategies must name existing strategies of the node, each once.",
    ),
    "invalid_instrument_spec": _c(
        InvalidParameter, "live-trading",
        "The spec given for an instrument names another symbol.",
    ),
    "recovering": _c(
        StateError, "live-trading",
        "The node was rebuilt from its journal and has not resumed: reconcile with the venue, then resume.",
    ),
    "fenced": _c(
        StateError, "live-trading",
        "A newer execution epoch owns the account; this node sends nothing more.",
    ),
    "fence_unsupported": _c(
        InvalidParameter, "live-trading",
        "epoch= was given with an adapter whose can_fence is not true, so the epoch would protect nothing.",
    ),
    "stale_epoch": _c(
        StateError, "live-trading",
        "restore() was given an epoch that is not newer than the one the journal already holds.",
    ),
    "epoch_required": _c(
        StateError, "live-trading",
        "restore() of a journal that holds an epoch was given none: the node that wrote it would not be fenced.",
    ),
    "adapter_cannot_adopt": _c(
        StateError, "live-trading",
        "restore() was given an adapter that cannot account for an order the node holds as working.",
    ),
    "hedging_unsupported_by_venue": _c(
        InvalidParameter, "live-trading",
        "oms_type='hedging' was given with an adapter whose venue nets positions.",
    ),
    "exit_rule_unconfirmed": _c(
        StateError, "live-trading",
        "The venue did not say whether it holds, changed or dropped an exit rule.",
    ),
    "exit_rule_not_cancellable": _c(
        StateError, "live-trading",
        "An exit rule the venue holds cannot be taken back; it stays held.",
    ),
    "exit_close_failed": _c(
        StateError, "live-trading",
        "The closes of an exit rule were refused or cancelled the allowed number of times and the position is still open.",
    ),
    "adapter_namespace_missing": _c(
        InvalidParameter, "live-trading",
        "An execution adapter must expose a non-empty namespace that marks the orders of one node.",
    ),
    "adapter_namespace_mismatch": _c(
        StateError, "live-trading",
        "restore() was given an adapter whose namespace is not the one the journal was written for.",
    ),
    "roll_close_failed": _c(
        StateError, "live-trading",
        "The close of an auto-roll ended without filling, so the open in the new contract was not sent.",
    ),
    "roll_open_failed": _c(
        StateError, "live-trading",
        "The open of an auto-roll ended without filling: the position is closed in the old contract and open in neither.",
    ),
    "no_reconcile": _c(
        UnsupportedOperation, "live-trading",
        "The adapter cannot say what the venue holds (no open_orders, positions or balances).",
    ),
    "correction_not_reversible": _c(
        StateError, "live-trading",
        "A correction or bust cannot be applied to the book as it stands: reconcile the position.",
    ),
    "paper_unsolicited_position_change": _c(
        StateError, "live-trading",
        "The simulated venue changed a position with no order behind it.",
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
    "group_timestamp_mismatch": _c(
        StateError, "live-trading",
        "A bar was pushed into an open group declared at one timestamp with another timestamp; it is not part of the group and is refused.",
    ),
    "duplicate_bar": _c(
        StateError, "live-trading",
        "A bar was pushed again for a symbol and timestamp that already has an identical bar; the retry is refused, not delivered twice.",
    ),
    "conflicting_bar": _c(
        StateError, "live-trading",
        "A bar was pushed for a symbol and timestamp that already has a bar with other values; a delivered bar is not revised.",
    ),
    "node_poisoned": _c(
        StateError, "live-trading",
        "A call stopped part way through a booking; the node's books cannot be trusted. Rebuild it from the venue's positions.",
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
