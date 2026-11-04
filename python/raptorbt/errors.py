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
('invalid_bar_stamp', ['open', 'close'], 'timeframes')

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
        valid: Accepted alternatives as a list: the values, or, where the set is
            open (a number range, a user's name), the rule that decides.
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
        if not valid and entry is not None:
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
    "missing_bar_stamp": _c(
        InvalidParameter, "timeframes", "Bars were given without bar_stamp, which is required.",
        ["open", "close"],
    ),
    "invalid_bar_stamp": _c(
        InvalidParameter, "timeframes", "bar_stamp is not 'open' or 'close'.",
        ["open", "close"],
    ),
    "missing_bar_ns": _c(
        InvalidParameter, "timeframes", "bar_stamp='open' was given without the bar width bar_ns.",
        ["bar_ns=<bar width in nanoseconds>, or a dict symbol -> nanoseconds", "bar_stamp='close' (needs no width)"],
    ),
    "invalid_bar_ns": _c(
        InvalidParameter, "timeframes", "bar_ns is not a positive whole number of nanoseconds.",
        ["a positive whole number of nanoseconds, e.g. 60_000_000_000 for one minute"],
    ),
    "invalid_primary_bars": _c(
        InvalidSubscription, "timeframes", "primary_bars is not a valid (step, unit) pair.",
        ["(step, unit) with step >= 1 and unit a bar unit, e.g. (5, 'm')"],
    ),
    "invalid_bar_spec": _c(
        InvalidSubscription, "timeframes", "A bar step or unit is not valid.",
        ["step >= 1", "unit: ms, s, m, h, d, w, month, y, tick, volume, value, renko, tick_imbalance, tick_runs, volume_imbalance, volume_runs, value_imbalance, value_runs"],
    ),
    "invalid_bar_step": _c(
        InvalidSubscription, "timeframes", "A bar subscription step is below 1.",
        ["a whole number >= 1"],
    ),
    "primary_bars_without_ticks": _c(
        InvalidSubscription, "subscriptions", "primary_bars was given but there are no ticks to aggregate.",
        ["ticks={symbol: {'timestamps': ..., 'ltp': ...}}", "primary_bars=None to run on the bars given"],
    ),
    # Strategies and runs
    "invalid_strategy": _c(
        InvalidParameter, "strategy", "A strategy entry is not a Strategy instance or subclass.",
        ["Strategy instance", "Strategy subclass", "(strategy, symbols) pair"],
    ),
    "duplicate_strategy": _c(
        InvalidParameter, "strategy", "The same strategy instance was passed twice in one call.",
        ["one instance per strategy: pass the class, or a new instance for each entry"],
    ),
    "empty_strategies": _c(
        InvalidParameter, "strategy", "The strategy list is empty.",
        ["at least one Strategy, (strategy, symbols) pair or entry dict"],
    ),
    "invalid_strategy_entry": _c(
        InvalidParameter, "strategy", "A strategy entry is a tuple that is not (strategy, symbols).",
        ["strategy", "(strategy, symbols)", "(strategy, symbols, {'account': name})", "{'strategy': s, 'symbols': [...], 'account': name}"],
    ),
    "invalid_accounts": _c(
        InvalidParameter, "portfolio", "accounts is not a dict of account name -> AccountSpec.",
        ["dict[str, AccountSpec]"],
    ),
    "invalid_account_spec": _c(
        InvalidParameter, "portfolio", "An AccountSpec field has a value the account does not accept.",
        ["initial_capital > 0", "account_type: cash, margin", "leverage > 0", "max_drawdown_pct and max_daily_loss_pct in (0, 100]", "max_daily_loss, max_gross_exposure > 0", "max_positions: a whole number > 0"],
    ),
    "unknown_account": _c(
        InvalidParameter, "portfolio", "A strategy or call names an account that is not in accounts.",
        ["a name from accounts={name: AccountSpec(...)}"],
    ),
    "account_too_large": _c(
        InvalidParameter, "portfolio", "More strategies share one account than a group may hold.",
        ["at most 500 strategies per account"],
    ),
    "invalid_batch_size": _c(
        InvalidParameter, "backtesting", "batch_size is not a positive integer or None.",
        ["a positive integer", None],
    ),
    "invalid_curve_samples": _c(
        InvalidParameter, "live-trading", "curve_samples is not a positive integer or None.",
        ["a positive integer", None],
    ),
    "invalid_input": _c(
        InvalidParameter, "core", "An argument or array was refused by the computation it was passed to.",
        ["the argument shape and range the message names"],
    ),
    "adoption_refused": _c(
        InvalidParameter, "live-trading",
        "A position cannot be adopted as given: a short on a cash account, a posted margin on a cash account or a cost basis on a margin account.",
        ["a long position on a cash account", "a short position or a posted margin on a margin account", "a cost basis (cost_minor) on a cash account"],
    ),
    "history_exhausted": _c(
        StateError, "live-trading",
        "A live node's history cannot hold a row: the ring is full of rows a strategy asked to keep, or the row was overwritten.",
        ["raise history= on the node", "ask the strategy to keep fewer rows (subscribe(..., history=))"],
    ),
    "invalid_initial_positions": _c(
        InvalidParameter, "live-trading", "initial_positions names a strategy or position that does not fit the node.",
        ["{strategy_index: {symbol: {'quantity': ..., 'avg_price': ..., 'margin_minor': ..., 'cost_minor': ...}}}", "a strategy index and a symbol of the node"],
    ),
    "no_symbols": _c(
        InvalidParameter, "market-data", "No symbols were given, or the data/ticks dict is empty.",
        ["at least one symbol"],
    ),
    "duplicate_symbols": _c(
        InvalidParameter, "market-data", "A symbol appears more than once.",
        ["each symbol once"],
    ),
    "unknown_symbol": _c(
        UnknownSymbol, "market-data", "A symbol was named that has no data in this run.",
        ["a symbol the run has data for"],
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
        InvalidParameter, "subscriptions", "A history or count is not a positive integer.",
        ["a positive integer"],
    ),
    "not_subscribed_timeframe": _c(
        InvalidSubscription, "subscriptions", "ctx.data was asked for a timeframe the strategy did not subscribe to.",
        ["subscribe(instrument, timeframe=...) in on_init or on_start"],
    ),
    "not_subscribed_data": _c(
        InvalidSubscription, "subscriptions", "ctx.data was asked for quotes or books the strategy did not subscribe to.",
        ["subscribe(instrument, data_type='quote' or 'book') in on_init or on_start"],
    ),
    "invalid_indicator_input": _c(
        InvalidParameter, "indicators", "An indicator input field other than the close was requested.",
        ["close"],
    ),
    "declaration_outside_run": _c(
        StateError, "subscriptions", "subscribe() or ctx.indicators was used outside on_init / on_start.",
        ["call subscribe() and ctx.indicators in on_init or on_start"],
    ),
    "portfolio_total_unavailable": _c(
        UnsupportedOperation, "portfolio", "A ctx.portfolio total needs the shared runtime (run_strategies).",
        ["run_strategies", "run_portfolio_strategy"],
    ),
    # Input data
    "invalid_input_type": _c(
        InvalidData, "market-data", "A data, ticks or depth argument is not a dict of arrays.",
        ["dict"],
    ),
    "missing_field": _c(
        InvalidData, "market-data", "A required field is missing from a symbol's arrays.",
        ["bars: timestamps, open, high, low, close, volume", "ticks: timestamps, ltp", "depth: timestamps, bid_prices, bid_sizes, ask_prices, ask_sizes"],
    ),
    "invalid_array": _c(
        InvalidData, "market-data", "A field is not a numeric array of the required dimension.",
        ["a numeric 1-D numpy array (depth levels: 2-D)"],
    ),
    "length_mismatch": _c(
        InvalidData, "market-data", "Arrays that must share one length do not.",
        ["every array the length of timestamps"],
    ),
    "misaligned_legs": _c(
        InvalidData, "market-data",
        "Legs traded bar by bar (a basket, a pair) do not share their timestamps.",
        ["the same timestamps on every leg"],
    ),
    "decreasing_timestamps": _c(
        InvalidData, "market-data", "Timestamps go backwards; bars must be in time order.",
        ["timestamps in non-decreasing order"],
    ),
    "empty_data": _c(
        InvalidData, "market-data", "There is no data to run on (zero bars or rows).",
        ["at least one bar, tick or instrument"],
    ),
    "depth_without_ticks": _c(
        InvalidData, "market-data", "depth was given for a symbol that has no ticks.",
        ["ticks for every symbol that has depth"],
    ),
    "non_finite_price": _c(
        InvalidData, "market-data",
        "A bar's open, high, low or close is NaN or infinite; the message names the column and first index.",
        ["finite open, high, low and close"],
    ),
    "invalid_data": _c(
        InvalidData, "market-data", "Other malformed input data (division by zero, index out of range, conversion).",
        ["the arrays and fields the message names"],
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
        InvalidOrder, "orders", "A price, size, offset, protective level or count is out of range, NaN or infinite.",
        ["price, trigger, quantity, offset, stop_price, target_price, slices and every > 0 and finite", "size_frac in (0, 1]", "limit_offset >= 0"],
    ),
    "missing_order_field": _c(
        InvalidOrder, "orders", "A field this order kind or time in force needs was not given.",
        ["limit_price for limit kinds", "trigger_price for stop and touched kinds", "offset for trailing kinds", "expire_ns for tif='gtd'", "units for a Twap"],
    ),
    "conflicting_order_args": _c(
        InvalidOrder, "orders", "Two options that exclude each other were both given (e.g. units and size_frac).",
        ["units (or quantity) or size_frac, not both", "every or every_bars, not both"],
    ),
    "incompatible_order_options": _c(
        InvalidOrder, "orders", "A time-in-force or flag does not apply to this order kind.",
        ["tif at_open or at_close with a market order", "post_only with a limit order", "offset_kind='ticks' with an instrument that has a price_increment"],
    ),
    "invalid_order_link": _c(
        InvalidOrder, "orders", "A parent or one-cancels-other link cannot be made.",
        ["link_oco(a, b, ...) with at least two ids of orders of one symbol", "parent: the client id of an order already submitted for the same symbol"],
    ),
    "invalid_roll_sizing": _c(
        InvalidParameter, "instruments", "roll_sizing is not 'notional' or 'lots'.", ["notional", "lots"]
    ),
    "roll_price_missing": _c(
        StateError, "instruments",
        "An auto-roll sized by notional could not price the new contract: it has no bar yet.",
        ["feed the new contract a bar before the roll", "roll_sizing='lots'"],
    ),
    "off_lot_quantity": _c(
        InvalidOrder, "orders",
        "An order's quantity is not a whole number of the instrument's lots (or size steps). It is "
        "refused in every mode so the simulation and a venue see the same terms; `valid` lists the "
        "nearest valid quantities.",
        ["a whole number of lots (the error lists the nearest)"],
    ),
    "off_tick_price": _c(
        InvalidOrder, "orders",
        "A limit, trigger, stop or target price is not a whole number of the instrument's ticks. It "
        "is refused in every mode so the simulation and a venue see the same terms; `valid` lists "
        "the nearest valid prices.",
        ["a whole number of ticks (the error lists the nearest)"],
    ),
    "unknown_order": _c(
        InvalidOrder, "orders",
        "A cancel or modify named a client id this run never submitted. A submitted order that "
        "has finished is a documented no-op, not this error.",
        ["a client id this run submitted (the error lists them)"],
    ),
    "bracket_under_hedging": _c(
        InvalidOrder, "orders",
        "A reduce-only child order (a bracket's stop or target) was submitted under hedging, "
        "where every order opens its own position and a reduce-only order is always refused.",
        ["oms_type='netting'", "stop_price and target_price on the entry order"],
    ),
    "invalid_order_intent": _c(
        InvalidOrder, "orders", "Intents queued on one bar are duplicated or unknown.",
        ["one enter() and one close_position() per bar", "not both while in position"],
    ),
    "invalid_order_reference": _c(
        InvalidOrder, "orders", "An order or position id does not exist.",
        ["a position id from ctx.positions()"],
    ),
    # Account and execution
    "invalid_account_type": _c(
        InvalidParameter, "execution", "account_type is not recognised.", ["cash", "margin"]
    ),
    "invalid_leverage": _c(
        InvalidParameter, "execution", "Margin accounts need leverage > 0.",
        ["a number > 0"],
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
        UnsupportedOperation, "backtesting", "Tick backtests are long-only; quantity must be >= 0.",
        ["quantity >= 0", "run_strategies with ticks for shorts"],
    ),
    "invalid_squareoff_time": _c(
        InvalidParameter, "execution", "squareoff_time is not a 24-hour 'HH:MM' string or None.",
        ["'HH:MM' (24-hour, local), e.g. '15:25'", None],
    ),
    "untradable_instrument": _c(
        UnsupportedOperation, "instruments", "The instrument kind cannot be traded.",
        ["a tradable kind: equity, contract, option, currency pair"],
    ),
    "invalid_session_schedule": _c(
        InvalidParameter, "instruments",
        "A SessionSchedule is invalid: a bad window, weekday, holiday date, offset or square-off minute.",
        ["windows: {'mon'..'sun' or 0..6: [(open_minute, close_minute), ...]} with 0 <= open < close <= 1440", "holidays: yyyymmdd ints", "squareoff_minute inside a window"],
    ),
    "invalid_corporate_action": _c(
        InvalidParameter, "execution",
        "A corporate action is invalid: an unknown kind, a ratio that is not positive, a negative dividend, or a pay date before the ex-date.",
        ["kind: split, bonus, dividend", "ratio > 0 for a split or bonus", "amount >= 0 for a dividend", "pay_ts at or after ts"],
    ),
    "invalid_adjustment_policy": _c(
        InvalidParameter, "execution",
        "Corporate actions were given without an adjustment_policy, or the policy is not raw or adjusted.",
        ["raw", "adjusted"],
    ),
    "unsupported_corporate_actions": _c(
        UnsupportedOperation, "execution",
        "corporate_actions were given to a runner that cannot apply them; they are refused, never ignored.",
        ["run_single_backtest", "run_strategy_backtest", "run_strategies"],
    ),
    "invalid_settlement": _c(
        InvalidParameter, "instruments",
        "An instrument's settlement style or settlement price is invalid.",
        ["settlement_style: cash, physical", "settlement_price: auto, last_price, underlying, or a number >= 0"],
    ),
    "invalid_fee_schedule": _c(
        InvalidParameter, "execution",
        "A fee schedule or rate card document is invalid, or fee_segment names a row the schedule does not have.",
        ["a fee schedule or rate card document in the documented format", "FeeSchedule.exchange_presets()"],
    ),
    "fee_schedule_gap": _c(
        InvalidParameter, "execution",
        "A fee schedule has no row for an instrument's or fill's venue and segment, or the instrument has no InstrumentSpec venue to choose a row from, or a fill or settlement falls on a date before the schedule's first row (the error names the earliest date); the run is refused rather than charged nothing or charged a rate that was not in force.",
        ["give the InstrumentSpec a venue the schedule prices", "set config.fee_segment", "add the missing row to the schedule"],
    ),
    "squareoff_needs_timezone": _c(
        InvalidParameter, "execution",
        "config.squareoff_time is a local time of day but the instrument has no session schedule and the config names no session_tz_offset_ns.",
        ["config.session_tz_offset_ns (IST_OFFSET_NS, or 0 for UTC)", "an InstrumentSpec with a SessionSchedule"],
    ),
    "mixed_holding": _c(
        InvalidParameter, "execution",
        "Strategies in one run declare different holding requirements (intraday with different flatten times, or intraday beside overnight), so no single square-off time can be derived; set config.squareoff_time or run them apart.",
        ["config.squareoff_time", "run the strategies apart"],
    ),
    "session_not_covered": _c(
        InvalidData, "instruments",
        "A bar falls on a date outside those a session schedule covers (the MCX preset, or a schedule loaded from calendar documents), so whether the market traded is unknown, not closed; the date and the covered ranges are named.",
        ["load a calendar document for the period (SessionSchedule.from_calendars)", "bars inside the dates the schedule covers"],
    ),
    "invalid_calendar": _c(
        InvalidParameter, "instruments",
        "A calendar document given to SessionSchedule.from_calendars is not valid JSON in the documented format, names no entry for the venue, or has overlapping or out-of-range windows or dates.",
        ["a JSON document with utc_offset and an entry for the venue", "windows and dates inside range", "one zone for every document"],
    ),
    "invalid_margin_model": _c(
        InvalidParameter, "risk",
        "A margin model or the line it prices is invalid: an unknown name, a missing rule for an instrument type, legs that do not fit, or a config.margin_model given to a cash account.",
        ["a model name from MarginModel (no_margin, cash_only, leverage, fraction_of_notional, option_simple, table, by_type, by_product, defined_risk, spread_credit, from_spec)", "a rule for every instrument type used", "no margin_model on a cash account"],
    ),
    "money_overflow": _c(
        MoneyOverflow, "execution",
        "A booked amount is not finite or a balance or sum is past the integer range of minor units; the run stops instead of wrapping.",
        ["amounts that are finite and inside the range of minor units", "a smaller config.money_decimals"],
    ),
    "invalid_money_decimals": _c(
        InvalidParameter, "execution",
        "config.money_decimals is outside 0..=9: the decimals cash, fees and realised P&L are booked in.",
        ["a whole number from 0 to 9"],
    ),
    "invalid_session_date": _c(
        InvalidParameter, "instruments", "A session date is not a real yyyymmdd date.",
        ["a real calendar date as yyyymmdd"],
    ),
    "invalid_instrument_metadata": _c(
        InvalidParameter, "instruments", "An instrument's venue or currency is an empty string.",
        ["a non-empty string", None],
    ),
    "mcx_session_unknown": _c(
        InvalidParameter, "instruments",
        "An MCX instrument with no session names no known contract class (select agricultural COTTON, KAPAS, COTWASOIL at 09:00-21:00; other agricultural at 09:00-17:00; bullion, base metals, energy), so no session is implied: pass session=.",
        ["pass session= (a SessionSchedule)", "name an underlying of a known contract class (COTTON, KAPAS, COTWASOIL: 09:00-21:00; other agricultural: 09:00-17:00; bullion, base metals, energy)"],
    ),
    "invalid_option_right": _c(
        InvalidParameter, "instruments", "An option's right is not call or put.", ["call", "put"]
    ),
    "invalid_expiry_pick": _c(
        InvalidParameter, "instruments", "OptionChain.pick_expiry was given an unknown rule, or 'date' without date_ns.",
        ["nearest", "nth", "weekly", "monthly", "date"],
    ),
    "invalid_parameter": _c(
        InvalidParameter, "core", "A parameter value is invalid (engine-side check).",
        ["the argument the message names"],
    ),
    "invalid_config": _c(
        InvalidParameter, "core", "A configuration value is invalid (engine-side check).",
        ["a configuration value inside the range the message names"],
    ),
    # Option structures and sizing
    "invalid_structure": _c(
        InvalidParameter, "instruments", "A structure, leg, selector, exit or submission argument is invalid.",
        ["the leg, selector, exit or submission argument the message names"],
    ),
    "structure_contract_missing": _c(
        InvalidParameter, "instruments", "The chain lists no contract (or expiry) a structure's leg selects at the event time.",
        ["a strike, expiry or right the chain lists at the event time"],
    ),
    "structure_limit_breached": _c(
        InvalidParameter, "orders", "The legs' prices make the net premium worse than net_limit.",
        ["a net_limit the legs' prices can meet", "order_type='market' without net_limit"],
    ),
    "structure_margin_exceeded": _c(
        InvalidParameter, "risk", "The margin model needs more for the group than is available.",
        ["fewer lots", "more available margin", "a structure that needs less margin"],
    ),
    "structure_missing_mark": _c(
        StateError, "instruments", "A price or mark a structure needs was not supplied.",
        ["marks: a price for every leg's symbol"],
    ),
    "below_one_lot": _c(
        InvalidParameter, "risk", "The sizing rule yields less than one lot; nothing is rounded up or to zero.",
        ["more capital, equity or available margin", "a smaller lot_size or price"],
    ),
    "invalid_sizing_input": _c(
        InvalidParameter, "risk", "A sizing input is not a positive finite number (or a fraction out of range).",
        ["a positive finite number (fractions in (0, 1], percent in (0, 100])"],
    ),
    "invalid_order_tags": _c(
        InvalidOrder, "orders",
        "An order's tags are not a dict of at most 16 str pairs (keys 1-64 bytes, values up to 256), or use the reserved key 'client_id'.",
        ["tags: dict of at most 16 str -> str pairs, keys 1-64 bytes, values up to 256, no 'client_id'"],
    ),
    "invalid_order_check": _c(
        InvalidParameter, "live-trading", "order_check is not callable.",
        ["a callable (order, ctx) -> None | str | OrderRefusal", None],
    ),
    "invalid_order_refusal": _c(
        InvalidParameter, "live-trading",
        "order_check returned something other than None, a code, or an object with a str code; or the code is empty, over 64 bytes, or the 1025th distinct one.",
        ["None to accept", "a str code of 1 to 64 bytes", "an object with a str code", "at most 1024 distinct codes"],
    ),
    "order_check_intent": _c(
        InvalidOrder, "live-trading",
        "enter() or close_position() was used with an order_check installed; they carry no order for it to see.",
        ["submit_order(...) with a typed order, not enter() or close_position()"],
    ),
    "unknown_role_binding": _c(
        InvalidParameter, "live-trading", "bindings names a role that no strategy declares.",
        ["a role some strategy declares in requirements"],
    ),
    "invalid_binding": _c(
        InvalidParameter, "live-trading",
        "A role is bound to the wrong shape: a Universe to anything but a list of distinct symbols, any other role to anything but one symbol.",
        ["one symbol (str) for an Underlying", "a list of distinct symbols for a Universe"],
    ),
    "universe_size": _c(
        InvalidParameter, "live-trading",
        "A Universe role is bound to fewer symbols than its min or more than its max.",
        ["a list with between the role's min and max symbols"],
    ),
    "universe_instrument": _c(
        InvalidParameter, "instruments",
        "ctx.instrument was given a Universe role, which names many symbols; ask for one of them.",
        ["one symbol of the role: ctx.role(name) lists them"],
    ),
    "invalid_trade_from": _c(
        InvalidParameter, "backtesting", "trade_from is not an integer event time in nanoseconds.",
        ["integer nanoseconds since the epoch", None],
    ),
    "no_instrument_spec": _c(
        UnknownSymbol, "instruments", "ctx.instrument was asked for a symbol with no InstrumentSpec.",
        ["instruments={symbol: InstrumentSpec}", "a role name the strategy declares"],
    ),
    "unresolved_role": _c(
        UnknownSymbol, "instruments", "ctx.role could not resolve a requirements role at this time.",
        ["supply the data or instrument specs the role needs", "bindings={role: symbol}"],
    ),
    "invalid_adjust": _c(
        InvalidParameter, "market-data", "continuous_series needs adjust to be 'raw', 'back' or 'ratio'.",
        ["raw", "back", "ratio"],
    ),
    "no_continuous_bars": _c(
        InvalidData, "market-data", "A continuous series has no bars in a contract's window, or an older contract has none before a roll.",
        ["bars for each contract inside its own window"],
    ),
    "roll_target_not_subscribed": _c(
        InvalidSubscription, "live-trading", "A continuous role rolled to a contract that is not a symbol of the run.",
        ["include the next contract among the symbols", "give a LiveNode the contract's InstrumentSpec"],
    ),
    "invalid_exit_rule": _c(
        InvalidParameter, "live-trading",
        "An exit rule is empty, has a non-positive or non-finite threshold, names no or several targets, or fired with no position.",
        ["ExitRule(stop_loss_pct | take_profit_pct | stop_loss | take_profit | premium_at_or_above | premium_at_or_below) with at least one condition", "positive finite thresholds; stop_loss_pct <= 100"],
    ),
    "unknown_role": _c(
        UnknownSymbol, "instruments", "ctx.role was asked for a role the strategy's requirements do not declare.",
        ["a role the strategy's requirements declare"],
    ),
    "no_requirements": _c(
        InvalidParameter, "instruments", "ctx.role was used by a strategy that declares no requirements.",
        ["declare requirements = Requirements(roles={...}) on the strategy"],
    ),
    "invalid_warmup": _c(
        InvalidParameter, "live-trading", "warmup is not 'live', 'muted' or None.",
        ["live", "muted", None],
    ),
    # Lifecycle
    "session_finished": _c(
        SessionFinished, "backtesting", "The session is finished; create a new one.",
        ["create a new session"],
    ),
    "node_finished": _c(
        StateError, "live-trading", "The live node is finished; create a new one.",
        ["create a new LiveNode, or LiveNode.restore from the journal"],
    ),
    # Execution adapters and venues
    "invalid_execution": _c(
        InvalidParameter, "live-trading",
        "execution is not an execution adapter, or the run cannot send its orders to one.",
        ["None", "SimulatedExecution()", "PaperExecution()", "an ExecutionAdapter (submit, cancel, modify)"],
    ),
    "unsupported_on_bars": _c(
        UnsupportedOperation, "core",
        "A setting that only the tick path implements (order_latency_ns, partial_fills) was given "
        "to a run that has bars; on bars it would change nothing.",
        ["run on ticks (ticks=)", "leave order_latency_ns at 0 and partial_fills False"],
    ),
    "unsupported_in_venue_run": _c(
        UnsupportedOperation, "live-trading",
        "The request has no equivalent when orders go to an execution adapter.",
        ["do not trade on warmup bars (warmup='muted')", "PaperExecution", "resend only orders the node sent"],
    ),
    "venue_requires_units": _c(
        InvalidOrder, "live-trading",
        "An order sent to an execution adapter gave its size as a capital fraction or the whole position.",
        ["units= (a quantity), not size_frac or the whole position"],
    ),
    "duplicate_client_order_id": _c(
        InvalidOrder, "live-trading", "An order id was reused for a different order.",
        ["a new client_id per order", "the same id only to repeat an identical order"],
    ),
    "unknown_client_order_id": _c(
        InvalidParameter, "live-trading", "A report names an order that was never sent.",
        ["the client_order_id of a request the node sent (s0-3), as the adapter received it"],
    ),
    "invalid_execution_report": _c(
        InvalidData, "live-trading",
        "A venue report contradicts the order it is about, or cannot be applied to it.",
        ["a report consistent with the order: no over-fill, a repeated exec_id with the same terms, no reject after a fill"],
    ),
    "venue_fill_not_booked": _c(
        StateError, "live-trading",
        "The venue filled an order but the book could not take the fill: reconcile the position.",
        ["reconcile the position with the venue (LiveNode.reconcile)"],
    ),
    "venue_unavailable": _c(
        StateError, "live-trading",
        "A request may not have reached the venue; the order stays as it was until the venue reports.",
        ["ask the venue what it holds (reconcile)", "resend only an order the venue does not hold"],
    ),
    "journal_failed": _c(
        StateError, "live-trading",
        "The journal could not take a record, so the node stopped before acting on it.",
        ["fix the sink and LiveNode.restore from what is durable"],
    ),
    "journal_evicted": _c(
        StateError, "live-trading",
        "A journal read asked for records a bounded memory has dropped; read them from the durable sink.",
        ["read the records from the durable sink", "a larger memory= bound"],
    ),
    "replay_diverged": _c(
        StateError, "live-trading",
        "A replay produced a record that is not the one the journal holds: the node is not the one that wrote it, or a strategy is not deterministic.",
        ["restore with the same feed, symbols, warmup, config and strategies that wrote the journal", "deterministic strategies (no clock, randomness or environment)"],
    ),
    "invalid_checkpoint": _c(
        InvalidData, "live-trading",
        "The checkpoint or journal records cannot be used to rebuild the node.",
        ["a checkpoint from LiveNode.checkpoint() of this version", "records= for a checkpoint taken without its log"],
    ),
    "instrument_in_use": _c(
        StateError, "live-trading",
        "An instrument cannot be retired while a strategy holds a position or a working order on it.",
        ["close the position and cancel the working orders on the symbol, then retire it"],
    ),
    "group_open": _c(
        StateError, "live-trading",
        "A timestamp group is open: flush() it before opening another, reconciling, resuming, reading health, taking a checkpoint or a state digest.",
        ["flush() the open group first"],
    ),
    "invalid_strategies": _c(
        InvalidParameter, "live-trading",
        "strategies must name existing strategies of the node, each once.",
        ["indexes of the node's strategies, each once"],
    ),
    "invalid_instrument_spec": _c(
        InvalidParameter, "live-trading",
        "The spec given for an instrument names another symbol.",
        ["an InstrumentSpec whose symbol is the symbol added"],
    ),
    "recovering": _c(
        StateError, "live-trading",
        "The node was rebuilt from its journal and has not resumed: reconcile with the venue, then resume.",
        ["reconcile(), push the reports you hold, resend what the venue lacks, then resume()"],
    ),
    "fenced": _c(
        StateError, "live-trading",
        "A newer execution epoch owns the account; this node sends nothing more.",
        ["stop this process: a newer epoch owns the account"],
    ),
    "fence_unsupported": _c(
        InvalidParameter, "live-trading",
        "epoch= was given with an adapter whose can_fence is not true, so the epoch would protect nothing.",
        ["an adapter with can_fence = True", "epoch=None"],
    ),
    "stale_epoch": _c(
        StateError, "live-trading",
        "restore() was given an epoch that is not newer than the one the journal already holds.",
        ["an epoch greater than the journal's"],
    ),
    "epoch_required": _c(
        StateError, "live-trading",
        "restore() of a journal that holds an epoch was given none: the node that wrote it would not be fenced.",
        ["epoch= greater than the journal's"],
    ),
    "adapter_cannot_adopt": _c(
        StateError, "live-trading",
        "restore() was given an adapter that cannot account for an order the node holds as working.",
        ["an adapter with restore_orders(rows)", "an adapter with keeps_own_records = True"],
    ),
    "hedging_unsupported_by_venue": _c(
        InvalidParameter, "live-trading",
        "oms_type='hedging' was given with an adapter whose venue nets positions.",
        ["netting"],
    ),
    "exit_rule_unconfirmed": _c(
        StateError, "live-trading",
        "The venue did not say whether it holds, changed or dropped an exit rule.",
        ["ask the venue whether it holds the rule (reconcile)"],
    ),
    "exit_rule_not_cancellable": _c(
        StateError, "live-trading",
        "An exit rule the venue holds cannot be taken back; it stays held.",
        ["an adapter with cancel_exit_rule(request)", "close the position"],
    ),
    "exit_close_failed": _c(
        StateError, "live-trading",
        "The closes of an exit rule were refused or cancelled the allowed number of times and the position is still open.",
        ["fix what the venue refuses (the refusal reasons are on the handle), then close the position yourself"],
    ),
    "adapter_namespace_missing": _c(
        InvalidParameter, "live-trading",
        "An execution adapter must expose a non-empty namespace that marks the orders of one node.",
        ["a non-empty str namespace on the adapter"],
    ),
    "adapter_namespace_mismatch": _c(
        StateError, "live-trading",
        "restore() was given an adapter whose namespace is not the one the journal was written for.",
        ["the adapter whose namespace the journal records"],
    ),
    "roll_close_failed": _c(
        StateError, "live-trading",
        "The close of an auto-roll ended without filling, so the open in the new contract was not sent.",
        ["fix what the venue refused, then close the position in the old contract yourself", "override on_roll_failed"],
    ),
    "roll_open_failed": _c(
        StateError, "live-trading",
        "The open of an auto-roll ended without filling: the position is closed in the old contract and open in neither.",
        ["open the position in the new contract yourself", "override on_roll_failed"],
    ),
    "no_reconcile": _c(
        UnsupportedOperation, "live-trading",
        "The adapter cannot say what the venue holds (no open_orders, positions or balances).",
        ["an adapter with open_orders(), positions() or balances()", "reconcile='skip' (journaled)"],
    ),
    "correction_not_reversible": _c(
        StateError, "live-trading",
        "A correction or bust cannot be applied to the book as it stands: reconcile the position.",
        ["reconcile the position with the venue"],
    ),
    "runtime_busy": _c(
        StateError, "backtesting", "The runtime is in use; a call re-entered it from a hook.",
        ["queue the push and make it after the hook returns"],
    ),
    "run_finished": _c(
        StateError, "backtesting", "The strategy's run is finished.",
        ["start a new run"],
    ),
    "group_timestamp_mismatch": _c(
        StateError, "live-trading",
        "A bar was pushed into an open group declared at one timestamp with another timestamp; it is not part of the group and is refused.",
        ["bars of the group's timestamp", "flush() and begin_group(new timestamp)"],
    ),
    "duplicate_bar": _c(
        StateError, "live-trading",
        "A bar was pushed again for a symbol and timestamp that already has an identical bar; the retry is refused, not delivered twice.",
        ["push each bar once"],
    ),
    "conflicting_bar": _c(
        StateError, "live-trading",
        "A bar was pushed for a symbol and timestamp that already has a bar with other values; a delivered bar is not revised.",
        ["a delivered bar is not revised: push the next bar"],
    ),
    "node_poisoned": _c(
        StateError, "live-trading",
        "A call stopped part way through a booking; the node's books cannot be trusted. Rebuild it from the venue's positions.",
        ["discard the node and rebuild it from the venue's positions (initial_positions=)"],
    ),
}


def call_as(operation: str, fn, *args, **kwargs):
    """Call ``fn``; a structured error it raises that names no operation names ``operation``.

    For a constructor of the native extension, whose frame the guard in ``_guard`` does not wrap.
    """
    try:
        return fn(*args, **kwargs)
    except RaptorBTError as e:
        if e.operation is None:
            e.operation = operation
        raise


def error_class(code: str) -> type:
    """The exception class registered for ``code``."""
    return CODES[code].cls


def make(code: str, message: str, **detail) -> RaptorBTError:
    """Build the error registered for ``code``. Used by the native extension."""
    return CODES[code].cls(message, code=code, **detail)


def _alternative(value: Any) -> str:
    """One alternative as the docs page shows it: a literal in code, a rule as plain text."""
    return value if isinstance(value, str) and " " in value else f"`{value!r}`"


def markdown() -> str:
    """The body of ``docs/errors.md``, generated from :data:`CODES`."""
    lines = [
        "# Errors",
        "",
        "<!-- Generated from raptorbt.errors.CODES: regenerate with `python -c 'import raptorbt.errors as e; print(e.markdown(), end="")' > docs/errors.md`. -->",
        "",
        "Every error RaptorBT raises on purpose is a `raptorbt.errors.RaptorBTError`.",
        "It carries `code`, `operation`, `parameter`, `value`, `valid` (the accepted",
        "alternatives, or the rule when the set is open), `skill` (the documentation",
        "domain) and `to_dict()`.",
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
        valid = "; ".join(_alternative(v) for v in c.valid)
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
    "call_as",
    "make",
    "markdown",
]
