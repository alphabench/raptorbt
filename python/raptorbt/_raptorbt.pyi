"""Type stubs for the raptorbt Rust extension module.

Covers the surface callers integrate against: configuration, results, and the
backtest runners. Indicator and tick helpers are declared with their array
signatures at the bottom.
"""

from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt

from raptorbt._bands import BollingerBands

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]
_Bool = npt.NDArray[np.bool_]

# Trading minutes per session, for BacktestConfig(session_minutes=...).
SESSION_NSE: float
SESSION_MCX: float
SESSION_CDS: float
SESSION_CONTINUOUS: float
IST_OFFSET_NS: int

# An instrument for the basket/portfolio runners:
# (timestamps, open, high, low, close, volume, entries, exits, direction, weight, symbol)
_Instrument = tuple[_I64, _F64, _F64, _F64, _F64, _F64, _Bool, _Bool, int, float, str]

class BacktestConfig:
    initial_capital: float
    fees: float
    slippage: float
    # Deprecated in favor of fill_timing: True maps to "same_bar_close",
    # False to "next_bar_open". An explicit fill_timing wins over this flag.
    upon_bar_close: bool
    # Execution-timing policy: "same_bar_close" (decide and fill at bar i's
    # close), "next_bar_open" (decide at bar i's close, fill at bar i+1's
    # open; a last-bar decision never trades), or "same_bar_open_lookahead"
    # (pre-0.11 behavior — fills at a price from before the signal existed;
    # not causally valid, for reproducing old results only). None derives
    # the policy from upon_bar_close. Read-only; set via the constructor.
    fill_timing: str | None
    apply_slippage: bool
    periods_per_year: float | None
    risk_free_rate: float
    session_minutes: float | None
    # The rates that price the run: your own schedule, or with only
    # `fee_segment` the dated statutory exchange presets (no brokerage).
    # Without `fee_segment` (or with "auto") each instrument is priced on the
    # row its InstrumentSpec venue and kind select, so a future never pays the
    # option row and a missing venue is refused. `fee_segment` ("NFO-OPT")
    # fixes one row for every instrument, an explicit override.
    fee_schedule: FeeSchedule | None
    fee_segment: str | None
    max_positions: int | None
    max_drawdown_pct: float | None
    fill_prob_limit: float
    fill_prob_slippage: float
    queue_fill_model: bool
    partial_fills: bool
    order_latency_ns: int
    # UTC offset of the session's local time; None names no zone (days are
    # UTC dates and squareoff_time is refused where no schedule gives a zone).
    session_tz_offset_ns: int | None
    limit_slippage: float
    liquidate_on_margin_call: bool
    # Trade out a position open at the last bar as a charged exit (with_close_open_at_end).
    close_open_at_end: bool
    fill_seed: int
    bar_path_adaptive: bool
    legacy_annualization: bool
    squareoff_time_minutes: int | None
    # Square-off minutes of single instruments, by symbol (with_squareoff_times):
    # each such symbol is squared off at its own time, every other at squareoff_time.
    squareoff_times: dict[str, int]
    # Optional risk limits; None (the default) leaves each off. A non-positive
    # value counts as off. A breach refuses the order or entry with the
    # reject reason shown.
    max_order_qty: float | None  # units above this: "order_size"
    max_order_notional: float | None  # units * price * multiplier above this: "order_size"
    max_position_qty: float | None  # net position per instrument after a fill: "max_position"
    max_gross_exposure: float | None  # notional across instruments: "max_exposure"
    max_daily_loss: float | None  # money; "daily_loss" for the rest of the day
    max_daily_loss_pct: float | None  # percent of the day's starting equity
    flatten_on_daily_loss: bool  # also close open positions on a breach
    # Decimals cash, fees and realised P&L are booked in (default 2: 100 minor
    # units to the rupee). Amounts are rounded half to even when booked.
    money_decimals: int
    # How a margin account (account_type="margin") funds an entry; None keeps
    # the built-in rule bit for bit. A cash account with a model is refused.
    # Refusals reach on_order_rejected as "insufficient_margin" (the model
    # demands more than is free, or refuses the side) or "margin_unpriced"
    # (the model has no rule for the instrument). no_margin locks nothing.
    margin_model: MarginModel | None

    def __init__(
        self,
        initial_capital: float = ...,
        fees: float = ...,
        slippage: float = ...,
        upon_bar_close: bool = ...,
        apply_slippage: bool = ...,
        periods_per_year: float | None = ...,
        risk_free_rate: float = ...,
        session_minutes: float | None = ...,
        fee_segment: str | None = ...,
        max_positions: int | None = ...,
        max_drawdown_pct: float | None = ...,
        legacy_annualization: bool = ...,
        fill_prob_limit: float = ...,
        fill_prob_slippage: float = ...,
        fill_seed: int = ...,
        bar_path_adaptive: bool = ...,
        queue_fill_model: bool = ...,
        partial_fills: bool = ...,
        order_latency_ns: int = ...,
        session_tz_offset_ns: int | None = ...,
        limit_slippage: float = ...,
        liquidate_on_margin_call: bool = ...,
        squareoff_time: str | None = ...,
        fill_timing: str | None = ...,
        max_order_qty: float | None = ...,
        max_order_notional: float | None = ...,
        max_position_qty: float | None = ...,
        max_gross_exposure: float | None = ...,
        max_daily_loss: float | None = ...,
        max_daily_loss_pct: float | None = ...,
        flatten_on_daily_loss: bool = ...,
        fee_schedule: FeeSchedule | None = ...,
        money_decimals: int = ...,
        margin_model: MarginModel | None = ...,
    ) -> None: ...
    def with_squareoff_time(self, hhmm: str | None) -> BacktestConfig: ...
    def with_close_open_at_end(self, on: bool = ...) -> BacktestConfig: ...
    def with_squareoff_times(self, times: dict[str, str]) -> BacktestConfig: ...
    def set_fixed_stop(self, percent: float) -> None: ...
    def set_atr_stop(self, multiplier: float, period: int) -> None: ...
    def set_trailing_stop(self, percent: float) -> None: ...
    def set_fixed_target(self, percent: float) -> None: ...
    def set_atr_target(self, multiplier: float, period: int) -> None: ...
    def set_risk_reward_target(self, ratio: float) -> None: ...

class InstrumentConfig:
    lot_size: float | None
    alloted_capital: float | None
    existing_qty: float | None
    avg_price: float | None

    def __init__(
        self,
        lot_size: float | None = ...,
        alloted_capital: float | None = ...,
        existing_qty: float | None = ...,
        avg_price: float | None = ...,
    ) -> None: ...
    def set_fixed_stop(self, percent: float) -> None: ...
    def set_atr_stop(self, multiplier: float, period: int) -> None: ...
    def set_trailing_stop(self, percent: float) -> None: ...
    def set_fixed_target(self, percent: float) -> None: ...
    def set_atr_target(self, multiplier: float, period: int) -> None: ...
    def set_risk_reward_target(self, ratio: float) -> None: ...

class StopConfig:
    stop_type: str
    percent: float | None
    multiplier: float | None
    period: int | None
    @staticmethod
    def fixed(percent: float) -> StopConfig: ...
    @staticmethod
    def atr(multiplier: float, period: int) -> StopConfig: ...
    @staticmethod
    def trailing(percent: float) -> StopConfig: ...

class TargetConfig:
    target_type: str
    percent: float | None
    multiplier: float | None
    period: int | None
    ratio: float | None
    @staticmethod
    def fixed(percent: float) -> TargetConfig: ...
    @staticmethod
    def atr(multiplier: float, period: int) -> TargetConfig: ...
    @staticmethod
    def risk_reward(ratio: float) -> TargetConfig: ...

class Trade:
    id: int
    symbol: str
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    size: float
    direction: int
    pnl: float
    return_pct: float
    entry_time: int
    exit_time: int
    # Total costs over the round trip: entry_fees + exit_fees.
    fees: float
    entry_fees: float
    # Zero when the exit was not a trade-out, e.g. an option left to expire.
    exit_fees: float
    # Present only when config.fee_segment selects an itemized schedule.
    # Keys: brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, total.
    fee_breakdown: dict[str, float] | None
    # The entry leg's and the exit leg's itemized costs alone, under the fee schedule's own
    # component names (brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, ipft,
    # dp_charge) and ``total``, the booked cost of the leg (``entry_fees`` / ``exit_fees``).
    # ``total`` can differ from the sum of the parts by the rounding the fee row applies. None
    # when the fee model does not itemise, on paths that synthesise a trade, and for the
    # entry of a position built from several fills.
    entry_fee_breakdown: dict[str, float] | None
    exit_fee_breakdown: dict[str, float] | None
    exit_reason: str
    # Intra-trade extremes, measured bar by bar while the position was open.
    # None on paths that synthesise a trade instead of closing a tracked
    # position (spreads, baskets, pairs, options rollups): "not measured",
    # never a zero excursion. Resolution is the bar, not the tick.
    mae_price: float | None
    mfe_price: float | None
    # Maximum Adverse Excursion: unrealised loss at the worst point, in money,
    # before costs. Never positive.
    mae_pnl: float | None
    # Maximum Favourable Excursion: unrealised profit at the best point, in
    # money, before costs. Never negative.
    mfe_pnl: float | None
    # Exact booked amounts, in integer minor units of ``money_decimals``
    # (2: 100 per rupee). ``pnl == pnl_minor / 10**money_decimals``. None on
    # paths that account in floats (the multi-leg simulators).
    money_decimals: int | None
    pnl_minor: int | None
    fees_minor: int | None
    entry_notional_minor: int | None
    exit_notional_minor: int | None

class BacktestMetrics:
    total_return_pct: float
    # None when the equity curve has no variation (a run that never traded):
    # the ratio is undefined, not zero.
    sharpe_ratio: float | None
    max_drawdown_pct: float
    max_drawdown_duration: int
    max_drawdown_duration_secs: float | None
    # Root mean square of the drawdown curve, in the same percentage points as
    # max_drawdown_pct. Weights a long shallow drawdown more heavily than a
    # brief deep one, which max_drawdown_pct cannot distinguish. Computed from
    # the same curve drawdown_curve() returns. Added in 0.13.2.
    ulcer_index: float
    # Share of equity samples strictly below the running high-water mark. The
    # companion to exposure_pct: how much of the run you were behind rather
    # than how much of it you were invested. Added in 0.13.2.
    time_under_water_pct: float
    # Closed trades that made money over closed trades, in percent, on P&L net
    # of fees; a position still open when the data ended is not counted. None
    # when no trade closed. ``full_report`` uses the same definition.
    win_rate_pct: float | None
    expectancy: float
    sqn: float
    total_trades: int
    total_closed_trades: int
    total_open_trades: int
    open_trade_pnl: float
    winning_trades: int
    losing_trades: int
    start_value: float
    end_value: float
    total_fees_paid: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_trade_return_pct: float
    # None when the population is empty -- no trade won, or none lost. An
    # average over an empty set is undefined; 0.0 there would read as "the
    # losers broke even", a claim about trades that do not exist.
    avg_win_pct: float | None
    avg_loss_pct: float | None
    avg_winning_duration: float | None
    avg_losing_duration: float | None
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_period: float
    avg_holding_period_secs: float | None
    exposure_pct: float

    # None when the denominator is zero -- e.g. profit factor with no losing
    # closed trade (net of fees, as full_report counts it). Previously these were float('inf'), which is not JSON-serializable.
    sortino_ratio: float | None
    calmar_ratio: float | None
    omega_ratio: float | None
    profit_factor: float | None
    payoff_ratio: float | None
    recovery_factor: float | None
    # Total traded notional, both sides counted (a buy and its sell-back are
    # two legs), at the same price*|size| base the fee models charge on.
    # Exit legs that never traded (EndOfData, Settlement) contribute nothing.
    # 0.0 on result paths that carry no trade list. Added in 0.10.0.
    total_turnover: float
    # Diagnostics, added in 0.13.2. None means "not measured on this path",
    # never a measured zero -- most of these have 0.0 as a legitimate value.
    # Shape of the per-bar return distribution. Kurtosis is EXCESS (Gaussian
    # 0.0, not 3.0); both are sample bias-corrected, as scipy's bias=False.
    return_skew: float | None
    return_kurtosis: float | None
    tail_ratio: float | None
    # Cost pressure: what share of gross profit went to costs, and how many
    # times current costs the run could absorb before net P&L reaches zero.
    cost_to_gross_profit_pct: float | None
    breakeven_cost_multiple: float | None
    # Share of moving bars that moved up -- the equity curve's batting average,
    # which is a different question from win_rate_pct over trades.
    return_consistency_pct: float | None
    # Mean drawdown across underwater samples only, in max_drawdown_pct's units.
    avg_drawdown_pct: float | None
    # Read mae_mfe_coverage_pct BEFORE the two below it: excursions are not
    # measured on synthesized trades (spread, basket and pairs legs), and this
    # says how much of the trade list the aggregates actually describe.
    mae_mfe_coverage_pct: float | None
    avg_mae_pnl: float | None
    mfe_capture_ratio: float | None

    def to_dict(self) -> dict[str, Any]: ...

class Order:
    """What became of one order the run placed, including one that never
    filled. A backtest reports the trades it made; this reports the ones it
    tried to make and could not."""

    id: int
    client_id: str
    symbol: str
    # "buy" | "sell"
    side: str
    # "market", "limit", "stop_market", ... — the shape, not its prices.
    kind: str
    # "gtc", "day", "ioc", ...
    tif: str
    # "filled", "canceled", "expired", "rejected", ...
    status: str
    submitted_idx: int
    submitted_ts: int
    # Units asked for; None when the request named a capital fraction rather
    # than a number — "not stated", never a zero request.
    requested_qty: float | None
    # Units filled, summed across slices. 0.0 for an order that never filled,
    # which is a measurement and not a gap.
    filled_qty: float
    avg_fill_price: float | None
    last_fill_idx: int | None
    # More than one means a partial-fill sequence.
    fill_slices: int
    limit_price: float | None
    trigger_price: float | None
    # Stable snake_case identifier ("insufficient_margin", "max_positions",
    # ...). None when the order was not rejected.
    reject_reason: str | None
    parent_id: int | None
    oco_group: int | None

class BacktestResult:
    metrics: BacktestMetrics
    # True when the drawdown limit tripped or the strategy was halted
    # (ctx.halt); halted_at is the bar count when it latched.
    halted: bool
    halted_at: int | None
    # Entries a risk limit refused, and entries the account could not fund
    # (capital or margin short, or sized to nothing). A run that never got into
    # the market says why here; the two are never summed.
    rejected_entries: int
    unfunded_entries: int
    #: The fee rows fills were priced on that are not matched to an official publication
    #: (``"NFO:FUTURES"``, ...). Empty when every row was confirmed or the run used flat fees.
    unverified_fee_rows: list[str]

    # These three return numpy arrays, not lists: `+` concatenates a list and
    # adds elementwise on an array, so the declared type decides what the
    # caller's code means.
    def equity_curve(self) -> npt.NDArray[np.float64]: ...
    def drawdown_curve(self) -> npt.NDArray[np.float64]: ...
    def trades(self) -> list[Trade]: ...
    # Every order placed, in submission order, including ones that never
    # filled. Empty for a run that placed no typed orders.
    def orders(self) -> list[Order]: ...
    def returns(self) -> npt.NDArray[np.float64]: ...
    # The full report over this result's equity curve, trades and orders (``raptorbt.report``);
    # ``equity_ts`` are the curve's timestamps in ns. Other keywords are ``raptorbt.full_report``'s.
    def full_report(self, equity_ts: Sequence[int], **kwargs: Any) -> dict[str, Any]: ...

class InstrumentSummary:
    symbol: str
    trades: int
    pnl: float
    rejected_entries: int
    #: Entries the account could not fund: sized to nothing, or refused for capital or margin.
    #: Not counted in ``rejected_entries``.
    unfunded_entries: int

class PortfolioResult:
    result: BacktestResult
    per_instrument: list[InstrumentSummary]
    rejected_entries: int
    #: Entries that could not be funded, across all instruments (not in ``rejected_entries``).
    unfunded_entries: int
    halted: bool
    # Bar index in array runs; a schedule-event ordinal in session runs
    # (``run_portfolio_strategy``), which interleaves N instrument streams.
    halted_at: int | None
    # Dividend cash credited over the run (a short's debit is negative): in the
    # equity curve, in no trade's P&L.
    dividends: float
    metrics: BacktestMetrics
    def full_report(self, equity_ts: Sequence[int], **kwargs: Any) -> dict[str, Any]: ...

class AccountResult:
    """An account group's final state; see ``AccountSpec``."""

    name: str
    initial_capital: float
    #: Cash at the end, positions closed: initial capital plus the members'
    #: trade P&L (which is net of fees).
    balance: float
    halted: bool
    #: Events applied to any member before the account first halted.
    halted_at: int | None
    #: Indices, in the call's strategy list, of the strategies on the account,
    #: in the order they step.
    strategies: list[int]
    def equity_curve(self) -> _F64: ...
    def drawdown_curve(self) -> _F64: ...
    #: Initial margin locked at each equity sample.
    def locked_margin(self) -> _F64: ...
    def timestamps(self) -> _I64: ...

class BatchSpreadItem:
    strategy_id: str
    spread_type: str
    max_loss: float | None
    target_profit: float | None
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class BatchSingleItem:
    item_id: str
    direction: int
    weight: float
    symbol: str
    def __init__(
        self,
        item_id: str,
        entries: _Bool,
        exits: _Bool,
        direction: int = ...,
        weight: float = ...,
        symbol: str = ...,
        config: BacktestConfig | None = ...,
        position_sizes: _F64 | None = ...,
        instrument_config: InstrumentConfig | None = ...,
    ) -> None: ...

def run_single_backtest(
    timestamps: _I64,
    open: _F64,
    high: _F64,
    low: _F64,
    close: _F64,
    volume: _F64,
    entries: _Bool,
    exits: _Bool,
    direction: int = ...,
    weight: float = ...,
    symbol: str = ...,
    config: BacktestConfig | None = ...,
    position_sizes: _F64 | None = ...,
    instrument_config: InstrumentConfig | None = ...,
    # Honored by run_single_backtest; every other array runner refuses them
    # with ``unsupported_corporate_actions`` rather than ignoring them.
    corporate_actions: Sequence[CorporateAction] | None = ...,
    adjustment_policy: str | None = ...,
) -> BacktestResult: ...
def run_basket_backtest(
    instruments: Sequence[_Instrument],
    config: BacktestConfig | None = ...,
    sync_mode: str = ...,
    instrument_configs: dict[str, InstrumentConfig] | None = ...,
    # Honored by run_single_backtest; every other array runner refuses them
    # with ``unsupported_corporate_actions`` rather than ignoring them.
    corporate_actions: Sequence[CorporateAction] | None = ...,
    adjustment_policy: str | None = ...,
) -> BacktestResult: ...
def run_portfolio_backtest(
    instruments: Sequence[_Instrument],
    config: BacktestConfig | None = ...,
    allocation: str = ...,
    instrument_configs: dict[str, InstrumentConfig] | None = ...,
    # Honored by run_single_backtest; every other array runner refuses them
    # with ``unsupported_corporate_actions`` rather than ignoring them.
    corporate_actions: Sequence[CorporateAction] | None = ...,
    adjustment_policy: str | None = ...,
) -> PortfolioResult:
    """Simulate instruments against one shared capital pool.

    Unlike summing independent per-symbol runs, capital is shared, so
    `max_positions` and the drawdown kill-switch on `config` are enforceable.
    `allocation` is "equal_weight" or "full".
    """

def run_options_backtest(
    timestamps: _I64,
    open: _F64,
    high: _F64,
    low: _F64,
    close: _F64,
    volume: _F64,
    option_prices: _F64,
    entries: _Bool,
    exits: _Bool,
    direction: int = ...,
    symbol: str = ...,
    config: BacktestConfig | None = ...,
    option_type: str = ...,
    strike_selection: str = ...,
    size_type: str = ...,
    size_value: float = ...,
    # Contracts per lot. Costs and P&L both scale by lots * lot_size, so this
    # is the difference between a 50-contract position and a single one.
    lot_size: int = ...,
    strike_interval: float = ...,
    # Opening premiums, parallel to option_prices. Used only under
    # fill_timing="next_bar_open": a signal fill then prices at the fill
    # bar's OPEN premium instead of the bar's settled value. Must match
    # option_prices in length; ValueError otherwise. Nothing is synthesized
    # when absent -- the next bar's value is the fill.
    option_open_prices: _F64 | None = ...,
    # Honored by run_single_backtest; every other array runner refuses them
    # with ``unsupported_corporate_actions`` rather than ignoring them.
    corporate_actions: Sequence[CorporateAction] | None = ...,
    adjustment_policy: str | None = ...,
) -> BacktestResult:
    """Backtest a single-leg option against its underlying.

    `option_type` is "call" or "put"; `strike_selection` is one of "atm",
    "otm1", "otm2", "itm1", "itm2"; `size_type` is "contracts", "percent",
    "notional" or "risk_percent", read together with `size_value`. Unknown
    values raise ValueError rather than falling back to a default.

    Set `fee_segment` on `config` (e.g. "NFO-OPT") to charge the itemized
    regulatory schedule, including per-order brokerage, instead of the flat
    `fees` rate.
    """

def run_pairs_backtest(*args: Any, **kwargs: Any) -> BacktestResult: ...
def run_multi_backtest(*args: Any, **kwargs: Any) -> BacktestResult: ...
def run_spread_backtest(
    timestamps: _I64,
    underlying_close: _F64,
    legs_premiums: list[_F64],
    # (option_type, strike, quantity, lot_size) per leg. `option_type` is
    # CE/CALL/C or PE/PUT/P, case-insensitive; anything else raises ValueError
    # rather than pricing a put as a call. `quantity` is signed: +1 long, -1
    # short.
    leg_configs: list[tuple[str, float, int, int]],
    entries: _Bool,
    exits: _Bool,
    config: BacktestConfig | None = ...,
    spread_type: str = ...,
    max_loss: float | None = ...,
    target_profit: float | None = ...,
    # One expiry per leg, in nanoseconds, matched to `leg_configs` by position;
    # a different length raises ValueError. Each leg settles on its own date
    # and the survivors keep marking, so calendars and diagonals run to the far
    # expiry. The premium series must carry the leg's settlement value at and
    # after its expiry -- the engine freezes the leg there and never invents a
    # price.
    leg_expiry_timestamps: list[int] | None = ...,
    # Opening premiums per leg, mirroring legs_premiums exactly (same legs,
    # same bars; ValueError otherwise). Used only under
    # fill_timing="next_bar_open", and only for SIGNAL entries and exits --
    # expiry settlement, squareoff, max-loss and target-profit closes keep
    # pricing at the current marks. Nothing is synthesized when absent.
    legs_open_premiums: list[_F64] | None = ...,
    # Honored by run_single_backtest; every other array runner refuses them
    # with ``unsupported_corporate_actions`` rather than ignoring them.
    corporate_actions: Sequence[CorporateAction] | None = ...,
    adjustment_policy: str | None = ...,
) -> BacktestResult: ...
def run_tick_backtest(
    timestamps: _I64,
    ltp: _F64,
    bid: _F64,
    ask: _F64,
    buy_qty_delta: _F64,
    sell_qty_delta: _F64,
    oi: _F64,
    entries: _Bool,
    exits: _Bool,
    symbol: str = ...,
    initial_capital: float = ...,
    fees: float = ...,
    slippage: float = ...,
    stop_loss_pct: float = ...,
    take_profit_pct: float = ...,
    max_hold_seconds: int = ...,
    entry_cooldown_ticks: int = ...,
    # Hard early exit, not a filter: the run stops after this many trades and
    # reports as if the input ended there. Unlimited by default since 0.7.0
    # (was 50, which silently truncated long tapes).
    max_trades: int = ...,
    # Costs and P&L both scale by |quantity| * lot_size. The defaults trade one
    # bare unit, reproducing the per-unit behaviour of releases through 0.7.3.
    lot_size: int = ...,
    # Long-only path: a negative quantity raises ValueError.
    quantity: int = ...,
    # Itemized regulatory schedule, e.g. "NFO-OPT". Unset keeps the flat `fees`
    # rate, which cannot express per-order brokerage at any rate.
    fee_segment: str | None = ...,
    # Honored by run_single_backtest; every other array runner refuses them
    # with ``unsupported_corporate_actions`` rather than ignoring them.
    corporate_actions: Sequence[CorporateAction] | None = ...,
    adjustment_policy: str | None = ...,
) -> BacktestResult: ...
# Returns (strategy_id, result) pairs, not bare results: the id is what lets a
# caller match a result back to the item that produced it. Declared as
# `list[BacktestResult]` through 0.13.1, which is simply wrong -- indexing a
# tuple as a result raises AttributeError at the first `.metrics`.
def batch_spread_backtest(
    timestamps: _I64,
    underlying_close: _F64,
    items: Sequence[BatchSpreadItem],
    config: BacktestConfig | None = ...,
) -> list[tuple[str, BacktestResult]]: ...
def batch_single_backtest(
    timestamps: _I64,
    open: _F64,
    high: _F64,
    low: _F64,
    close: _F64,
    volume: _F64,
    items: Sequence[BatchSingleItem],
    config: BacktestConfig | None = ...,
) -> list[tuple[str, BacktestResult]]: ...
def simulate_portfolio_mc(
    returns: Sequence[_F64],
    weights: _F64,
    correlation_matrix: Sequence[_F64] | _F64,
    initial_value: float,
    n_simulations: int = ...,
    horizon_days: int = ...,
    seed: int = ...,
) -> dict[str, Any]: ...

# --- Portfolio math ----------------------------------------------------------

class RiskModel:
    asset_ids: list[str]
    n_assets: int
    periods_per_year: float
    shrinkage_intensity: float
    n_obs: int
    def cov(self) -> _F64: ...

class OptimizerConfig:
    risk_aversion: float
    turnover_penalty: float
    position_cap: float
    sector_ids: list[int]
    sector_caps: list[float]
    no_trade_band: float
    min_trade_value: float
    portfolio_value: float
    cash_max: float
    max_iter: int
    tolerance: float
    # Long/short mode: short_cap > 0 enables w_i in [-short_cap,
    # position_cap], sum|w| <= gross_max, net_min <= sum(w) <= net_max, and
    # GROSS sector caps. Defaults (short_cap=0) are exactly the historical
    # long-only problem; the other three fields are inert then.
    short_cap: float
    gross_max: float
    net_min: float
    net_max: float
    def __init__(
        self,
        risk_aversion: float,
        turnover_penalty: float,
        position_cap: float,
        sector_ids: Sequence[int],
        sector_caps: Sequence[float],
        no_trade_band: float = ...,
        min_trade_value: float = ...,
        portfolio_value: float = ...,
        cash_max: float = ...,
        max_iter: int = ...,
        tolerance: float = ...,
        short_cap: float = ...,
        gross_max: float = ...,
        net_min: float = ...,
        net_max: float = ...,
    ) -> None: ...

class OptimizationResult:
    snapped: list[bool]
    # cash is 1 - sum(w) (net-based); for a long/short book read the
    # exposure fields instead of inferring from cash.
    cash: float
    gross_exposure: float
    net_exposure: float
    turnover: float
    objective: float
    vol_annualized: float
    solver_status: str
    iterations: int
    def weights(self) -> _F64: ...
    def trades(self) -> _F64: ...

class RiskContributions:
    total_vol_annualized: float
    def marginal(self) -> _F64: ...
    def contribution(self) -> _F64: ...
    def pct_contribution(self) -> _F64: ...

class OptimizeItem:
    def __init__(
        self,
        item_id: str,
        alpha: _F64,
        w_current: _F64,
        portfolio_value: float | None = ...,
    ) -> None: ...

class RankIC:
    mean_ic: float
    stdev_ic: float
    t_stat: float
    t_stat_deflated: float
    n_dates_scored: int
    n_independent: float
    overlap_days: int
    mean_names: float
    def daily_ic(self) -> _F64: ...

class RebalanceSimResult:
    n_rebalances: int
    n_trades: int
    total_cost_drag_annualized: float
    def equity_curve(self) -> _F64: ...
    def turnover(self) -> _F64: ...
    def cost_regulatory(self) -> _F64: ...
    def cost_brokerage(self) -> _F64: ...
    def cost_dp(self) -> _F64: ...

def estimate_covariance(
    returns: _F64,
    asset_ids: Sequence[str],
    periods_per_year: float,
) -> RiskModel: ...
def optimize_portfolio(
    model: RiskModel,
    alpha: _F64,
    w_current: _F64,
    asset_ids: Sequence[str],
    config: OptimizerConfig,
) -> OptimizationResult: ...
def batch_optimize_portfolios(
    model: RiskModel,
    items: Sequence[OptimizeItem],
    config: OptimizerConfig,
) -> list[tuple[str, OptimizationResult]]: ...
def compute_risk_contributions(
    model: RiskModel,
    weights: _F64,
    asset_ids: Sequence[str],
) -> RiskContributions: ...
def winsorize_panel(values: _F64, pct: float) -> _F64: ...
def zscore_panel(values: _F64, min_names: int) -> _F64: ...
def rank_panel(values: _F64, min_names: int) -> _F64: ...
def momentum_panel(prices: _F64, lookback: int, skip: int) -> _F64: ...
def composite_scores(factors: Sequence[_F64], weights: _F64) -> _F64: ...
def rank_ic(
    factor: _F64,
    prices: _F64,
    horizon: int,
    min_names: int,
) -> RankIC: ...
def simulate_rebalance_policy(
    prices: _F64,
    target_weights: _F64,
    initial_capital: float,
    policy: str,
    policy_param: float,
    segment: str = ...,
    min_trade_value: float = ...,
    dp_charge_per_isin: float = ...,
    periods_per_year: float = ...,
) -> RebalanceSimResult: ...
# Keys since 0.9.0: brokerage_flat (per-order cap in rupees; 0 on equity
# delivery, where the broker charges nothing), brokerage_rate (percentage
# alternative per order, 0 where only the flat applies -- charge is
# min(flat, rate * order_value) when rate > 0), stt_rate, exchange_txn_rate,
# sebi_turnover_rate, stamp_duty_rate, gst_rate,
# dp_sell_charge_per_isin_per_day. The pre-0.9.0 `brokerage_per_order` key
# was removed deliberately so stale consumers fail loudly.
def indian_cost_schedule(segment: str) -> dict[str, float]: ...

# --- Indicators -------------------------------------------------------------

def sma(data: _F64, period: int) -> _F64: ...
def ema(data: _F64, period: int) -> _F64: ...
def rsi(data: _F64, period: int) -> _F64: ...
def macd(
    data: _F64, fast_period: int = ..., slow_period: int = ..., signal_period: int = ...
) -> tuple[_F64, _F64, _F64]: ...
def stochastic(
    high: _F64, low: _F64, close: _F64, k_period: int = ..., d_period: int = ...
) -> tuple[_F64, _F64]: ...
def atr(high: _F64, low: _F64, close: _F64, period: int) -> _F64: ...
def bollinger_bands(
    data: _F64, period: int = ..., std_dev: float = ...
) -> BollingerBands: ...
def adx(high: _F64, low: _F64, close: _F64, period: int) -> _F64: ...
def vwap(high: _F64, low: _F64, close: _F64, volume: _F64) -> _F64: ...
def vwap_session(
    high: _F64, low: _F64, close: _F64, volume: _F64, session_starts: _Bool
) -> _F64: ...
def supertrend(
    high: _F64, low: _F64, close: _F64, period: int = ..., multiplier: float = ...
) -> tuple[_F64, _F64]: ...
def rolling_min(data: _F64, period: int) -> _F64: ...
def rolling_max(data: _F64, period: int) -> _F64: ...

# --- Tick signals and features ---------------------------------------------

def compute_tick_entry_signals(*args: Any, **kwargs: Any) -> _Bool: ...
def compute_tick_exit_signals(
    timestamps_ns: _I64, eod_exit_time_ns: int = ...
) -> _Bool: ...
def tick_spread_pct(bid: _F64, ask: _F64) -> _F64: ...
def buy_sell_imbalance_delta(
    buy_qty_cumulative: _F64, sell_qty_cumulative: _F64
) -> _F64: ...
def return_window(
    timestamps_ns: _I64, ltp: _F64, window_seconds: float = ...
) -> _F64: ...
def realized_vol_rolling(
    timestamps_ns: _I64, ltp: _F64, window_seconds: float = ...
) -> _F64: ...
def oi_position_pct(oi: _F64, oi_day_high: float, oi_day_low: float) -> _F64: ...
def tick_velocity(timestamps_ns: _I64, window_seconds: float = ...) -> _F64: ...

# --- Trading sessions --------------------------------------------------------

class SessionSchedule:
    """When an instrument can trade: weekday windows, holidays, square-off.

    Minutes are local minutes from midnight; dates are ``yyyymmdd`` ints. A
    window is half-open (``open <= t < close``). No holiday calendar ships:
    pass the exchange's dates in ``holidays``.
    """

    tz_offset_ns: int
    squareoff_minute: int | None
    holidays: list[int]
    #: Keyed ``"mon"``..``"sun"``; days that do not trade are absent.
    windows: dict[str, list[tuple[int, int]]]
    def __init__(
        self,
        windows: dict[str | int, list[tuple[int, int]]],
        tz_offset_ns: int = ...,
        holidays: list[int] | None = ...,
        squareoff_minute: int | None = ...,
    ) -> None: ...
    overrides: dict[int, list[tuple[int, int]]]
    #: ``(from, to)`` ``yyyymmdd`` ranges the schedule answers for; empty is every date, and any other date is closed.
    coverage: list[tuple[int, int]]
    #: Trading minutes in a regular session (the longest weekday of the base windows).
    trading_minutes: float
    @staticmethod
    def nse_equity(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def nse_fo(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def bse_equity(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def bse_fo(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def mcx(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def mcx_select_agri(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def mcx_other_agri(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def cds_cross(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def cds(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def bcd(
        holidays: list[int] | None = ..., squareoff_minute: int | None = ...
    ) -> SessionSchedule: ...
    @staticmethod
    def from_calendars(venue: str, documents: list[str]) -> SessionSchedule: ...
    def with_window_change(
        self,
        from_date: int,
        windows: dict[str | int, list[tuple[int, int]]],
        to_date: int | None = ...,
    ) -> SessionSchedule: ...
    def with_session_override(self, date: int, windows: list[tuple[int, int]]) -> SessionSchedule: ...
    def windows_on(self, date: int) -> list[tuple[int, int]]: ...
    def with_holidays(self, dates: list[int]) -> SessionSchedule: ...
    def with_squareoff_minute(self, squareoff_minute: int | None) -> SessionSchedule: ...
    def is_open(self, ts_ns: int) -> bool: ...
    def session_date(self, ts_ns: int) -> int | None: ...
    def session_close_ns(self, date: int) -> int | None: ...
    def next_open_ns(self, ts_ns: int) -> int | None: ...

# --- Fees as data and margin models ------------------------------------------

#: Keys of a priced fill: brokerage, stt (the transaction tax, whatever the
#: row calls it), exchange_txn, sebi_fee, stamp_duty, gst, ipft, dp_charge,
#: total, and exact_total (set by the rounded modes, else None).
_Charge = dict[str, float | None]

class RateCard:
    @staticmethod
    def from_json(text: str) -> RateCard: ...
    @staticmethod
    def from_dict(doc: dict[str, Any]) -> RateCard: ...
    @property
    def name(self) -> str: ...
    @property
    def effective_from(self) -> str: ...
    @property
    def effective_ns(self) -> int: ...
    @property
    def source(self) -> str: ...
    def fee_schedule(self) -> FeeSchedule: ...
    def backdated_fills(self, fill_ts: list[int]) -> int: ...
    def to_json(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

class FeeSchedule:
    @staticmethod
    def exchange_presets() -> FeeSchedule: ...
    def in_force_at(self, at: int) -> FeeSchedule: ...
    def backdated(self, at: int) -> FeeSchedule: ...
    @staticmethod
    def from_cards(cards: list[RateCard]) -> FeeSchedule: ...
    @staticmethod
    def from_json(text: str) -> FeeSchedule: ...
    @staticmethod
    def from_dict(doc: dict[str, Any]) -> FeeSchedule: ...
    @property
    def name(self) -> str: ...
    @property
    def source(self) -> str: ...
    def rows(self) -> list[dict[str, Any]]: ...
    def row(
        self,
        venue: str,
        segment: str,
        underlying: str | None = ...,
        at: int | None = ...,
        settlement: bool = ...,
    ) -> dict[str, Any]: ...
    def charge(
        self,
        venue: str,
        segment: str,
        side: str,
        price: float,
        quantity: float,
        multiplier: float = ...,
        underlying: str | None = ...,
        at: int | None = ...,
        order_turnover: float = ...,
        day_fee_owed: bool = ...,
    ) -> _Charge: ...
    def to_json(self, effective_from: str = ...) -> str: ...
    def __len__(self) -> int: ...

class MarginModel:
    @staticmethod
    def no_margin() -> MarginModel: ...
    @staticmethod
    def cash_only() -> MarginModel: ...
    @staticmethod
    def leverage(factors: dict[str, float]) -> MarginModel: ...
    @staticmethod
    def percent_of_notional(pct: float) -> MarginModel:
        """Deprecated: takes a percent. Use :meth:`fraction_of_notional` (``pct / 100``)."""
    @staticmethod
    def fraction_of_notional(fraction: float) -> MarginModel:
        """A fraction of the notional (``0.2`` is 20 percent); above 1 is refused."""
    @staticmethod
    def option_simple(seller_pct: float) -> MarginModel: ...
    @staticmethod
    def table(per_unit: dict[str, float], default: MarginModel | None = ...) -> MarginModel: ...
    @staticmethod
    def by_type(models: dict[str, MarginModel]) -> MarginModel: ...
    @staticmethod
    def by_product(intraday: MarginModel, carry: MarginModel) -> MarginModel: ...
    @staticmethod
    def defined_risk() -> MarginModel: ...
    @staticmethod
    def spread_credit(seller_pct: float, spread_pct: float) -> MarginModel: ...
    @staticmethod
    def from_spec(leverage: float = ...) -> MarginModel: ...
    @property
    def name(self) -> str: ...
    def required(self, spec: InstrumentSpec, quantity: float, price: float) -> float: ...
    def required_split(
        self, spec: InstrumentSpec, quantity: float, price: float, intraday_quantity: float
    ) -> float: ...
    def required_legs(
        self, legs: list[tuple[InstrumentSpec, float, float]], now: int | None = ...
    ) -> float: ...
    def group_shares(
        self, legs: list[tuple[InstrumentSpec, float, float]], now: int = ...
    ) -> list[float]: ...
    def groups_options(self) -> bool: ...
    def params(self) -> dict[str, Any]: ...
    def describe(self) -> str: ...

def defined_risk(legs: list[tuple[InstrumentSpec, float, float]]) -> float: ...

# --- Instrument market definitions ------------------------------------------

class InstrumentSpec:
    settlement_fee: float
    # "cash" (default) or "physical" (the underlying is delivered)
    settlement_style: str
    # "auto" (default), "last_price", "underlying", or a price declared for
    # the contract
    settlement_price: str | float
    symbol: str
    kind: str
    price_increment: float
    size_increment: float
    lot_size: float
    multiplier: float
    margin_init: float
    margin_maint: float
    span_pct: float
    exposure_pct: float
    maker_fee: float
    taker_fee: float
    activation_ns: int | None
    expiration_ns: int | None
    strike: float | None
    right: str | None
    underlying: str | None
    tradable: bool
    #: Metadata only: nothing is converted between currencies.
    venue: str | None
    currency: str | None
    #: With a schedule the kernel enforces the session; without one the
    #: instrument trades at any timestamp, as before.
    session: SessionSchedule | None
    #: The exchange's scrip group where charges depend on it (BSE cash: "A", "B").
    scrip_group: str | None
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> InstrumentSpec: ...
    @staticmethod
    def equity(
        symbol: str,
        price_increment: float = ...,
        lot_size: float = ...,
        size_increment: float = ...,
        margin_init: float = ...,
        margin_maint: float = ...,
        maker_fee: float = ...,
        taker_fee: float = ...,
        venue: str | None = ...,
        currency: str | None = ...,
        session: SessionSchedule | None = ...,
        scrip_group: str | None = ...,
    ) -> InstrumentSpec: ...
    @staticmethod
    def futures_contract(
        symbol: str,
        expiration_ns: int,
        lot_size: float,
        multiplier: float = ...,
        price_increment: float = ...,
        underlying: str | None = ...,
        activation_ns: int | None = ...,
        margin_init: float = ...,
        margin_maint: float = ...,
        maker_fee: float = ...,
        taker_fee: float = ...,
        venue: str | None = ...,
        currency: str | None = ...,
        session: SessionSchedule | None = ...,
        scrip_group: str | None = ...,
        size_increment: float = ...,
    ) -> InstrumentSpec: ...
    @staticmethod
    def perpetual(
        symbol: str,
        lot_size: float = ...,
        multiplier: float = ...,
        price_increment: float = ...,
        size_increment: float = ...,
        underlying: str | None = ...,
        margin_init: float = ...,
        margin_maint: float = ...,
        maker_fee: float = ...,
        taker_fee: float = ...,
        venue: str | None = ...,
        currency: str | None = ...,
        session: SessionSchedule | None = ...,
        scrip_group: str | None = ...,
    ) -> InstrumentSpec: ...
    @staticmethod
    def option(
        symbol: str,
        strike: float,
        right: str,
        expiration_ns: int,
        lot_size: float,
        multiplier: float = ...,
        price_increment: float = ...,
        underlying: str | None = ...,
        binary: bool = ...,
        activation_ns: int | None = ...,
        margin_init: float = ...,
        margin_maint: float = ...,
        maker_fee: float = ...,
        taker_fee: float = ...,
        span_pct: float = ...,
        exposure_pct: float = ...,
        venue: str | None = ...,
        currency: str | None = ...,
        session: SessionSchedule | None = ...,
        scrip_group: str | None = ...,
        size_increment: float = ...,
    ) -> InstrumentSpec: ...
    @staticmethod
    def currency_pair(
        symbol: str,
        price_increment: float = ...,
        size_increment: float = ...,
        lot_size: float = ...,
        margin_init: float = ...,
        margin_maint: float = ...,
        maker_fee: float = ...,
        taker_fee: float = ...,
        venue: str | None = ...,
        currency: str | None = ...,
        session: SessionSchedule | None = ...,
        scrip_group: str | None = ...,
    ) -> InstrumentSpec: ...
    @staticmethod
    def index(
        symbol: str,
        price_increment: float = ...,
        venue: str | None = ...,
        currency: str | None = ...,
        session: SessionSchedule | None = ...,
        scrip_group: str | None = ...,
    ) -> InstrumentSpec: ...

# --- Streaming indicators ----------------------------------------------------

class Indicator:
    kind: str
    value: Any | None
    initialized: bool
    #: True once a shared runtime (``run_strategies``) owns the indicator.
    #: ``update_bar`` and ``reset`` then raise ``RuntimeError``.
    linked: bool
    #: Number of ``update_bar`` calls made on this indicator.
    update_count: int
    @staticmethod
    def sma(period: int) -> Indicator: ...
    @staticmethod
    def ema(period: int) -> Indicator: ...
    @staticmethod
    def wilder_ma(period: int) -> Indicator: ...
    @staticmethod
    def wma(period: int) -> Indicator: ...
    @staticmethod
    def roc(period: int) -> Indicator: ...
    @staticmethod
    def stddev(period: int) -> Indicator: ...
    @staticmethod
    def rsi(period: int) -> Indicator: ...
    @staticmethod
    def atr(period: int) -> Indicator: ...
    @staticmethod
    def donchian(period: int) -> Indicator: ...
    @staticmethod
    def bollinger(period: int, k: float = ...) -> Indicator: ...
    @staticmethod
    def macd(fast: int = ..., slow: int = ..., signal: int = ...) -> Indicator: ...
    @staticmethod
    def expr(program: ExprProgram, output: int = ...) -> Indicator: ...
    def update_bar(
        self, open: float, high: float, low: float, close: float, volume: float | None = ...
    ) -> Any | None: ...
    def reset(self) -> None: ...

# --- Bar aggregation ---------------------------------------------------------

_BarArrays = tuple[_I64, _F64, _F64, _F64, _F64, _F64]

class BarAggregator:
    step: int
    unit: str
    stamp: str
    def __init__(
        self,
        step: int,
        unit: str,
        tz_offset_ns: int = ...,
        brick_size: float = ...,
        stamp: Literal["open", "close"] = ...,
        #: Daily and longer bars cut at the schedule's own midnight and skip
        #: holidays and weekends; other units are unaffected.
        session: SessionSchedule | None = ...,
    ) -> None: ...
    def push_bar(
        self,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> tuple[int, float, float, float, float, float] | None: ...
    def push_trade(
        self,
        timestamp: int,
        price: float,
        size: float,
        signed_size: float = ...,
    ) -> tuple[int, float, float, float, float, float] | None: ...
    # Renko completes several bricks at once; drain after every push.
    def next_pending(self) -> tuple[int, float, float, float, float, float] | None: ...
    def flush(self) -> tuple[int, float, float, float, float, float] | None: ...

def aggregate_bars(
    timestamps: _I64,
    open: _F64,
    high: _F64,
    low: _F64,
    close: _F64,
    volume: _F64,
    step: int,
    unit: str,
    tz_offset_ns: int = ...,
    brick_size: float = ...,
    stamp: Literal["open", "close"] = ...,
) -> _BarArrays: ...
def bars_from_ticks(
    timestamps: _I64,
    ltp: _F64,
    buy_qty_delta: _F64,
    sell_qty_delta: _F64,
    step: int,
    unit: str,
    tz_offset_ns: int = ...,
    brick_size: float = ...,
) -> _BarArrays: ...

# --- Multi-strategy runtime --------------------------------------------------
#
# Internal to ``raptorbt.run_strategies`` and ``raptorbt.LiveNode``: ``Runtime``
# owns the shared market data, one isolated portfolio book per strategy and the
# per-event loop. A runtime over ``MarketData.live`` takes pushed events.
# ``RuntimeContext`` is the context handed to a strategy's hooks;
# ``StrategyView`` is a strategy's book with the method surface of
# ``PortfolioSession``, which the command path drives.

class RuntimeContext:
    symbol: str
    idx: int
    bar: Any | None
    timestamp: int
    bar_close_ts: int | None
    #: The three times of the event (or report) in flight, in ns: the
    #: exchange's, the arrival and the processing time. ``None`` between events.
    event_ns: int | None
    arrival_ns: int | None
    processing_ns: int | None
    equity: float
    cash: float
    free_capital: float
    #: Whether a market order decided at a bar fills at the next bar's open.
    fills_at_next_open: bool
    fill_ns: int | None
    symbols: list[str]
    position: PositionSnapshot | None
    net_position: float
    is_net_long: bool
    is_net_short: bool
    is_flat: bool
    #: The print being handled, only inside ``on_trade_tick``.
    tick: Any | None
    #: The quote being handled, only inside ``on_quote``.
    quote: Any | None
    #: The last book seen for the current symbol; persists outside its hook.
    book: Any | None
    best_bid: float | None
    best_ask: float | None
    last_price: float | None
    def series(self, symbol: str | None = ...) -> dict[str, _F64 | _I64]: ...
    def keep_history(self, rows: int, symbol: str | None = ...) -> None: ...
    #: True while a warm-up bar is handled (see LiveNode's ``warmup``), and in a backtest
    #: while the event is before its ``trade_from``, where every order is refused.
    is_warmup: bool
    def instrument(self, key: str | None = ...) -> InstrumentSpec: ...
    def option_chain(self, underlying: str | None = ...) -> OptionChain: ...
    def role(self, name: str) -> str | tuple[str, ...]: ...
    def position_for(self, symbol: str | None = ...) -> PositionSnapshot | None: ...
    def positions(self, symbol: str | None = ...) -> list[PositionSnapshot]: ...
    def set_underlying_price(
        self, price: float | None, symbol: str | None = ...
    ) -> None: ...
    def net_position_for(self, symbol: str | None = ...) -> float: ...
    def open_orders(self, symbol: str | None = ...) -> list[Any]: ...
    def halt(self, reason: str = ..., flatten: bool = ...) -> None: ...
    def is_halted(self) -> bool: ...
    # The rows this strategy declared for a symbol in a windowed run, None when it declared none
    # (every row is shown) or the run is not windowed.
    def declared_history(self, symbol: str | None = ...) -> int | None: ...
    def _instrument_index(self, symbol: str | None = ...) -> int: ...
    # ``(known, price)``: the last close, else the last positive ltp (a quote-only row is no
    # trade), of ``series(symbol)`` without building it;
    # ``known`` is false where ``series`` must be read instead.
    def _last_price(self, symbol: str) -> tuple[bool, float | None]: ...
    #: ``(locked_margin, realized_pnl, unrealized_pnl, gross_exposure,
    #: net_exposure)``; backs ``ctx.portfolio``.
    def _portfolio_totals(self) -> tuple[float, float, float, float, float]: ...

class OrderRequest:
    """One order as the engine hands it to an execution adapter. Immutable."""

    client_order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: float | None
    trigger_price: float | None
    trail_offset: float | None
    trail_kind: str | None
    trail_limit_offset: float | None
    time_in_force: str
    expire_ns: int | None
    strategy_id: int
    account_id: str | None
    reduce_only: bool
    post_only: bool
    parent_client_order_id: str | None
    tags: dict[str, str]
    #: Under hedging, the engine's id of the position this order closes; ``None`` for an order
    #: that opens, and always ``None`` on a netting account.
    position_id: int | None
    submitted_ns: int

class ExecutionReport:
    """What a venue says about one order. Immutable."""

    kind: str
    ts_ns: int
    exec_id: str | None
    qty: float | None
    price: float | None
    fee: float | None
    reason: str | None
    venue_order_id: str | None
    @staticmethod
    def accepted(venue_order_id: str, ts_ns: int) -> ExecutionReport: ...
    @staticmethod
    def rejected(reason: str, ts_ns: int) -> ExecutionReport: ...
    @staticmethod
    def fill(
        exec_id: str, qty: float, price: float, ts_ns: int, fee: float = ...
    ) -> ExecutionReport: ...
    @staticmethod
    def cancel_ack(ts_ns: int) -> ExecutionReport: ...
    @staticmethod
    def cancel_rejected(reason: str, ts_ns: int) -> ExecutionReport: ...
    @staticmethod
    def replace_ack(
        ts_ns: int,
        new_qty: float | None = ...,
        new_limit: float | None = ...,
        new_trigger: float | None = ...,
    ) -> ExecutionReport: ...
    @staticmethod
    def replace_rejected(reason: str, ts_ns: int) -> ExecutionReport: ...
    @staticmethod
    def expired(ts_ns: int) -> ExecutionReport: ...
    @staticmethod
    def correction(exec_id: str, qty: float, price: float, ts_ns: int) -> ExecutionReport: ...
    @staticmethod
    def bust(exec_id: str, ts_ns: int) -> ExecutionReport: ...

class CommandQueue:
    """A strategy's queued commands and intents; orders are converted to
    plain data when submitted."""

    def __init__(self) -> None: ...
    def enter(
        self,
        size_frac: float | None = ...,
        stop_price: float | None = ...,
        target_price: float | None = ...,
    ) -> None: ...
    def close_intent(self) -> None: ...
    def submit(
        self,
        order: Any,
        client_id: str,
        parent: str | None = ...,
        symbol: str | None = ...,
        twap: bool = ...,
    ) -> None: ...
    def cancel(self, client_id: str) -> None: ...
    def cancel_all(self) -> None: ...
    # Have the strategy's on_order_event called for every order event from now on.
    def enable_order_events(self) -> None: ...
    def link_oco(self, client_ids: list[str]) -> None: ...
    def close(self, position_id: int, symbol: str | None = ...) -> None: ...
    def close_all_for(self, symbol: str) -> None: ...
    def modify(
        self,
        client_id: str,
        units: float | None = ...,
        size_frac: float | None = ...,
        limit_price: float | None = ...,
        trigger_price: float | None = ...,
    ) -> None: ...
    def has_commands(self) -> bool: ...
    def has_intents(self) -> bool: ...
    def drain_commands(self) -> list[tuple[Any, ...]]: ...
    def drain_intents(self) -> list[tuple[Any, ...]]: ...
    def restore_intents(self, intents: list[tuple[Any, ...]]) -> None: ...

class KernelDriver:
    """``run_strategy_backtest``'s per-bar work after ``on_bar``: route the
    strategy's queued commands, fold its intents, step the session and send
    the engine events to the hooks."""

    def __init__(
        self,
        session: KernelSession,
        strategy: Any,
        ctx: Any,
        hooks: int,
        atr: Any | None = ...,
    ) -> None: ...
    def rest(
        self,
        idx: int,
        ts: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None: ...
    def step(
        self,
        idx: int,
        ts: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None: ...

class StrategyView:
    def submit_order(
        self,
        instrument: int,
        side: str,
        kind: str,
        submitted_idx: int,
        submitted_ts: int,
        client_id: str,
        units: float | None = ...,
        size_frac: float | None = ...,
        limit_price: float | None = ...,
        trigger_price: float | None = ...,
        tif: str = ...,
        expire_ns: int | None = ...,
        stop_price: float | None = ...,
        target_price: float | None = ...,
        offset: float | None = ...,
        offset_kind: str = ...,
        limit_offset: float = ...,
        post_only: bool = ...,
        reduce_only: bool = ...,
        parent_id: int | None = ...,
    ) -> int: ...
    def submit_twap(
        self,
        instrument: int,
        units: float,
        side: str,
        slices: int,
        interval_ns: int,
        submitted_idx: int,
        submitted_ts: int,
        client_id: str,
        reduce_only: bool = ...,
    ) -> int: ...
    def cancel_twap(self, instrument: int, algo_id: int, idx: int) -> bool: ...
    def cancel_order(self, instrument: int, idx: int, order_id: int) -> bool: ...
    def modify_order(
        self,
        instrument: int,
        order_id: int,
        units: float | None = ...,
        size_frac: float | None = ...,
        limit_price: float | None = ...,
        trigger_price: float | None = ...,
    ) -> bool: ...
    def cancel_all_orders(self, instrument: int, idx: int) -> list[int]: ...
    def link_oco(self, instrument: int, order_ids: list[int]) -> None: ...
    def request_close(self, instrument: int, position_id: int) -> None: ...
    def set_underlying_price(
        self, instrument: int, price: float | None = ...
    ) -> None: ...
    def positions(self, instrument: int) -> list[PositionSnapshot]: ...
    def position(self, instrument: int) -> PositionSnapshot | None: ...
    def working_orders(
        self, instrument: int
    ) -> list[tuple[int, str, str, str, str, float, float]]: ...
    def equity(self) -> float: ...
    def is_halted(self) -> bool: ...
    def halt(self, reason: str = ..., flatten: bool = ...) -> None: ...
    def cash(self) -> float: ...
    def free_capital(self) -> float: ...

class MarketData:
    """The columns of a run, built once and shared by every ``Runtime`` over it."""

    def __init__(
        self,
        symbols: list[str],
        arrays: list[dict[str, _F64 | _I64] | None],
        offsets: list[int],
        ticks: list[dict[str, Any] | None] | None = ...,
        depth: list[dict[str, Any] | None] | None = ...,
        widths: list[int] | None = ...,
    ) -> None: ...
    @staticmethod
    def live(symbols: list[str], offsets: list[int], history: int) -> MarketData:
        """The shell of a live node: no data, rings of ``history`` rows per symbol."""
        ...
    @staticmethod
    def windowed(
        symbols: list[str],
        offsets: list[int],
        history: int,
        bars: list[bool],
        ticks: list[list[str] | None],
        depth: list[bool],
        widths: list[int],
    ) -> MarketData:
        """The shell of a windowed run (``run_strategies_windowed``): no data, rings of
        ``history`` rows per symbol, what each symbol carries (whether it has ``bars``, the tick
        columns it has, ``None`` without ticks, whether it has ``depth``) and the width of its
        primary bars in nanoseconds, ``0`` when not given."""
        ...

class Runtime:
    def __init__(
        self,
        market: MarketData,
        directions: list[int],
        instrument_configs: list[InstrumentConfig | None],
        instruments: list[InstrumentSpec | None],
        config: BacktestConfig | None,
        oms_type: str,
        account_type: str,
        leverage: float,
        tz_offset_ns: int,
        stamp: str,
        share: bool,
        keep_curves: bool,
        bar_cls: Any,
        composite_cls: Any,
        working_order_cls: Any,
        dispatch: Any,
        trade_cls: Any,
        quote_cls: Any,
        book_cls: Any,
    ) -> None: ...
    def set_primary_bars(self, step: int, unit: str) -> None: ...
    def add_account(
        self,
        initial_capital: float,
        account_type: str,
        leverage: float,
        members: int,
        max_drawdown_pct: float | None = ...,
        max_daily_loss: float | None = ...,
        max_daily_loss_pct: float | None = ...,
        max_gross_exposure: float | None = ...,
        max_positions: int | None = ...,
    ) -> int: ...
    def add_strategy(
        self, symbols: list[str], account: int | None = ...
    ) -> tuple[int, RuntimeContext, StrategyView]: ...
    def halt_account(self, account: int, reason: str, flatten: bool) -> None: ...
    def finish_account(self, account: int, name: str) -> AccountResult: ...
    def register_streams(
        self,
        sid: int,
        strategy: Any,
        ctx: RuntimeContext,
        commands: Any | None,
        clocks: list[Any],
        hooks: int,
        subscriptions: list[tuple[int, str, float]],
        indicators: list[tuple[Any, int | None, str | None]],
    ) -> bool: ...
    def set_corporate_actions(self, actions: list[CorporateAction], policy: str) -> None: ...
    def finalize(self) -> None: ...
    def run(self) -> None: ...
    def push_window(
        self,
        bars: dict[int, dict[str, Any]],
        ticks: dict[int, dict[str, Any]] | None = ...,
        depth: dict[int, dict[str, Any]] | None = ...,
    ) -> int:
        """Run the next window of a windowed runtime (built from ``MarketData.windowed``) as
        ``run`` runs a tape: the arrays are keyed by the symbol's index in the run. Returns the
        events delivered. A window that starts at or before the last event of the one before
        raises ``window_overlap``."""
        ...
    def declare_rows(self, sid: int, rows: int) -> None:
        """The strategy reads at most ``rows`` trailing rows of each symbol it did not declare
        with ``keep_history``: a windowed run shows it exactly those."""
        ...
    def keep_all_rows(self, sid: int, force: bool = ...) -> None:
        """Keep every row of the symbols a windowed strategy declared nothing for (all of them
        with ``force``), so its ``ctx.series`` is a batch run's."""
        ...
    def history_plan(self, sid: int) -> list[tuple[str, int | None]]:
        """By symbol, the rows a windowed run holds for the strategy, ``None`` for every row."""
        ...
    def set_curve_window(self, window: int | None) -> None: ...
    def adopt_position(
        self,
        sid: int,
        local: int,
        timestamp_ns: int,
        price: float,
        quantity: float,
        margin_minor: int | None = ...,
        cost_minor: int | None = ...,
    ) -> int: ...
    def push_tick(
        self,
        instrument: int,
        timestamp: int,
        ltp: float,
        bid: float = ...,
        ask: float = ...,
        buy_qty_delta: float = ...,
        sell_qty_delta: float = ...,
        ltq: float = ...,
        bid_qty: float = ...,
        ask_qty: float = ...,
        oi: float = ...,
        arrival_ns: int | None = ...,
    ) -> int: ...
    def push_bar(
        self,
        instrument: int,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        arrival_ns: int | None = ...,
    ) -> None: ...
    def push_depth(
        self,
        instrument: int,
        timestamp: int,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        arrival_ns: int | None = ...,
    ) -> None: ...
    def advance_time(self, timestamp: int) -> None: ...
    def begin_group(self, timestamp: int | None = ..., record: bool = ...) -> None: ...
    def flush(self, record: bool = ...) -> None: ...
    @property
    def group_open(self) -> bool: ...
    def held_pushes(self) -> list[tuple[int, int]]: ...
    def check_bar(self, instrument: int, timestamp: int) -> None: ...
    def add_instrument(
        self,
        symbol: str,
        spec: InstrumentSpec | None,
        direction: int,
        strategies: list[int],
        history: int | None = ...,
        offset: int = ...,
    ) -> int: ...
    def retire_instrument(self, symbol: str) -> None: ...
    def last_event_ns(self) -> int: ...
    # Keep a journal of this node, and/or give it an execution epoch. Call before finalize().
    def set_journal(self, journal: Journal | None = ..., epoch: int | None = ...) -> None: ...
    # Take a new execution epoch after the journal was replayed.
    def set_epoch(self, epoch: int) -> None: ...
    # While set, requests are not sent to a real venue: the node is being rebuilt from its journal.
    def set_replay(self, replay: bool) -> None: ...
    # The event time and arrival time of the latest input the node took.
    def clock(self) -> tuple[int, int]: ...
    # The venue lists an order as open, so its request arrived.
    def confirm(self, client_order_id: str) -> None: ...
    # Client order ids whose request may or may not have reached the venue.
    def unconfirmed(self) -> list[str]: ...
    # The latest bar the node took for an instrument, as pushed:
    # (timestamp, open, high, low, close, volume).
    def delivered_bar(
        self, instrument: int
    ) -> tuple[int, float, float, float, float, float] | None: ...
    # Whether a call stopped part way through a booking: every push is refused as node_poisoned.
    def poisoned(self) -> bool: ...
    # Venue fills the books have not booked: (client order id, quantity, price, reason).
    def unbooked_fills(self) -> list[tuple[str, float, float, str]]: ...
    # Why each strategy was stopped for holding a sold option its margin model cannot bound:
    # (strategy index, reason).
    def margin_faults(self) -> list[tuple[int, str]]: ...
    # Exits of hedge legs each book is still carrying out: (strategy index, hedge instrument,
    # hedge position id, sold legs sent, hedge close sent).
    def pending_covers(self) -> list[tuple[int, int, int, int, bool]]: ...
    # Protective legs whose day order lapsed with a session, waiting for the next one's first
    # event to be sent again: (strategy index, local instrument, position id, leg, level).
    def pending_rearms(self) -> list[tuple[int, int, int, str, float]]: ...
    # Adapter calls a replay answered "unavailable" for, since last asked.
    def take_replay_errors(self) -> int: ...
    # Check the execution epoch with the venue now; raises fenced when a newer epoch owns it.
    def heartbeat(self) -> None: ...
    # Send an order again under its first request (same client order id).
    def resend(self, client_order_id: str) -> None: ...
    # Register the orders a venue makes on its own: (client_order_id, symbol_index, side,
    # quantity) of a strategy, so their reports are booked.
    def register_venue_orders(
        self, strategy: int, orders: list[tuple[str, int, str, float]]
    ) -> None: ...
    # Mark orders as having an outcome nobody knows, until confirm() is called for each.
    def mark_unconfirmed(self, ids: list[str]) -> None: ...
    # End registered exit orders: the venue's exit rule was cancelled or replaced.
    def end_venue_orders(self, strategy: int, ids: list[str]) -> None: ...
    # Ids of the orders a venue-held exit rule would fire that are still waiting.
    def venue_held_orders(self) -> list[str]: ...
    # Orders the node holds as working at the venue: (request, status, filled_qty,
    # average_fill_price).
    def working_requests(self) -> list[tuple[OrderRequest, str, float, float | None]]: ...
    # Every order sent to the venue: (client_order_id, strategy, symbol, side, quantity,
    # filled_qty, status).
    def venue_orders(self) -> list[tuple[str, int, str, str, float, float, str]]: ...
    # A newer execution epoch owns the account.
    @property
    def fenced(self) -> bool: ...
    def set_execution(
        self,
        kind: str,
        adapter: Any | None = ...,
        account_id: str | None = ...,
        protective: str = ...,
    ) -> None: ...
    def set_processing_latency(self, latency_ns: int) -> None: ...
    def set_warmup(self, warmup: bool, muted: bool = ...) -> None: ...
    def hold_boot(self) -> None: ...
    def release_boot(self) -> None: ...
    def set_order_check(self, check: Any, proposal: Any, typed_only: bool = False) -> None: ...
    def set_trade_from(self, ns: int | None = ...) -> None: ...
    def push_report(
        self, client_order_id: str, report: ExecutionReport, arrival_ns: int
    ) -> str: ...
    def finish(self, sid: int) -> PortfolioResult: ...
    def close(self) -> None: ...
    def memory_report(self) -> dict[str, int]: ...

class PortfolioSession:
    def __init__(
        self,
        config: BacktestConfig | None = ...,
        account_type: str = ...,
        leverage: float = ...,
    ) -> None: ...
    def add_instrument(
        self,
        symbol: str,
        direction: int = ...,
        instrument_config: InstrumentConfig | None = ...,
        instrument: InstrumentSpec | None = ...,
        oms_type: str = ...,
    ) -> int: ...
    def set_bars(
        self,
        instrument: int,
        timestamps: _I64,
        open: _F64,
        high: _F64,
        low: _F64,
        close: _F64,
        volume: _F64,
    ) -> None: ...
    def set_ticks(
        self,
        instrument: int,
        timestamps: _I64,
        ltp: _F64,
        bid: _F64 | None = ...,
        ask: _F64 | None = ...,
        buy_qty_delta: _F64 | None = ...,
        sell_qty_delta: _F64 | None = ...,
        ltq: _F64 | None = ...,
        bid_qty: _F64 | None = ...,
        ask_qty: _F64 | None = ...,
        oi: _F64 | None = ...,
    ) -> None: ...
    def set_depth(
        self,
        instrument: int,
        timestamps: _I64,
        bid_prices: Any,
        bid_sizes: Any,
        ask_prices: Any,
        ask_sizes: Any,
    ) -> None: ...
    def current_depth(
        self,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None: ...
    def seal(self) -> None: ...
    # Incremental (live) feed: append to the schedule tail in arrival order.
    # Each seals first and is idempotent, so batch warmup data merges ahead of
    # the first push. Drive appended events with current_event()/apply_current().
    # push_tick returns how many events it appended (0-2): a trade print, plus
    # a quote when ask > 0. ltq is the print's size when known (else the flow
    # deltas stand in); bid_qty/ask_qty size the L1 quote; oi rides the print.
    def push_tick(
        self,
        instrument: int,
        timestamp: int,
        ltp: float,
        bid: float = ...,
        ask: float = ...,
        buy_qty_delta: float = ...,
        sell_qty_delta: float = ...,
        ltq: float = ...,
        bid_qty: float = ...,
        ask_qty: float = ...,
        oi: float = ...,
    ) -> int: ...
    # (order_id, client_id, side, kind, status, filled_qty, units); status is
    # "accepted", "triggered" or "partially_filled"; units is nan unless the
    # order was placed in explicit units.
    def working_orders(
        self, instrument: int
    ) -> list[tuple[int, str, str, str, str, float, float]]: ...
    def push_bar(
        self,
        instrument: int,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None: ...
    # bids/asks are (price, size) lists, best level first.
    def push_depth(
        self,
        instrument: int,
        timestamp: int,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
    ) -> None: ...
    # Events pushed or merged but not yet applied.
    def remaining(self) -> int: ...
    def __len__(self) -> int: ...
    # Bar sessions only; returns None on a tick event.
    def current(
        self,
    ) -> tuple[int, int, int, float, float, float, float, float] | None: ...
    # (kind, instrument, local_idx, ts, a, b, c, d, e); kind is
    # "bar" (o/h/l/c/v), "trade" (price, size, ...) or "quote" (bid, ask, ...).
    def current_event(
        self,
    ) -> tuple[str, int, int, int, float, float, float, float, float] | None: ...
    def apply_current(
        self,
        entry: bool = ...,
        exit: bool = ...,
        atr: float = ...,
        size_mult: float | None = ...,
        stop_price: float | None = ...,
        target_price: float | None = ...,
    ) -> list[EngineEvent]: ...
    def submit_order(self, instrument: int, *args: Any, **kwargs: Any) -> int: ...
    def submit_twap(
        self,
        instrument: int,
        units: float,
        side: str,
        slices: int,
        interval_ns: int,
        submitted_idx: int,
        submitted_ts: int,
        client_id: str,
        reduce_only: bool = ...,
    ) -> int: ...
    def cancel_twap(self, instrument: int, algo_id: int, idx: int) -> bool: ...
    def cancel_order(self, instrument: int, idx: int, order_id: int) -> bool: ...
    def cancel_all_orders(self, instrument: int, idx: int) -> list[int]: ...
    def modify_order(
        self,
        instrument: int,
        order_id: int,
        units: float | None = ...,
        size_frac: float | None = ...,
        limit_price: float | None = ...,
        trigger_price: float | None = ...,
    ) -> bool: ...
    def link_oco(self, instrument: int, order_ids: list[int]) -> None: ...
    # Adopt a position the account already holds (broker-truth seeding): no
    # order, no fill, no fees, no trade record. Cash mode debits the cost
    # basis; a fully funded margin book locks it instead. Cash or leverage-1.0
    # margin, long-only. Must be called after seal() and before the first
    # apply_current() — enforced, since adopting mid-run understates max
    # drawdown. Returns the new position id.
    def set_corporate_actions(self, actions: list[CorporateAction], policy: str) -> None: ...
    @property
    def dividends_received(self) -> float: ...
    def adopt_position(
        self,
        instrument: int,
        timestamp_ns: int,
        price: float,
        size: float,
    ) -> int: ...
    def request_close(self, instrument: int, position_id: int) -> None: ...
    def set_underlying_price(
        self,
        instrument: int,
        price: float | None = ...,
    ) -> None: ...
    def positions(self, instrument: int) -> list[PositionSnapshot]: ...
    def position(self, instrument: int) -> PositionSnapshot | None: ...
    def equity(self) -> float: ...
    def cash(self) -> float: ...
    def free_capital(self) -> float: ...
    def is_halted(self) -> bool: ...
    def halt(self, reason: str = ..., flatten: bool = ...) -> None: ...
    def finish(self) -> PortfolioResult: ...

# --- Per-bar strategy session (class-based strategy contract) ---------------

class EngineEvent:
    # Besides the order and position kinds: "risk_rejected" (a risk limit
    # refused an order or entry; reject_reason says which) and "risk_halted"
    # (the daily loss limit was reached, or the strategy was halted;
    # reject_reason holds the reason).
    kind: str
    idx: int
    price: float | None
    size: float | None
    direction: int | None
    trade: Trade | None
    reject_reason: str | None
    order_id: int | None
    client_order_id: str | None
    # "contract_settled", "corporate_action" and "lifecycle_refused" events:
    # outcome is "settled"/"exercised"/"assigned"/"expired"/"delivered", or
    # "split"/"bonus"/"dividend", or what was refused; style "cash"/"physical";
    # source "declared"/"underlying"/"last_price"; quantity the underlying
    # delivered (settlement) or the ratio (split, bonus); amount the delivery
    # price or the dividend per unit; cash what the account received.
    outcome: str | None
    style: str | None
    source: str | None
    quantity: float | None
    amount: float | None
    cash: float | None
    # "order_resized" (the engine cut a working order's units: a sibling of a
    # one-cancels-other group whose partner partly filled): size is the units
    # it offers now, previous_size what it offered before. Reaches
    # on_order_event only.
    previous_size: float | None

class CorporateAction:
    """A split, bonus or cash dividend on one instrument, effective at ``ts``
    (the ex-date's first instant, ns)."""
    def __init__(
        self,
        instrument: str,
        kind: str,
        ts: int,
        ratio: float | None = ...,
        amount: float | None = ...,
        pay_ts: int | None = ...,
    ) -> None: ...
    @staticmethod
    def split(instrument: str, ts: int, ratio: float) -> CorporateAction: ...
    @staticmethod
    def bonus(instrument: str, ts: int, ratio: float) -> CorporateAction: ...
    @staticmethod
    def dividend(
        instrument: str, ts: int, amount: float, pay_ts: int | None = ...
    ) -> CorporateAction: ...
    @property
    def instrument(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def ts(self) -> int: ...
    @property
    def ratio(self) -> float | None: ...
    @property
    def amount(self) -> float | None: ...
    @property
    def pay_ts(self) -> int | None: ...

def adjust_history(
    instrument: str, actions: list[CorporateAction], bars: dict[str, Any]
) -> dict[str, Any]: ...

class PositionSnapshot:
    position_id: int
    entry_idx: int
    entry_price: float
    size: float
    direction: int
    stop_price: float | None
    target_price: float | None

class KernelSession:
    def __init__(
        self,
        symbol: str = ...,
        direction: int = ...,
        config: BacktestConfig | None = ...,
        instrument_config: InstrumentConfig | None = ...,
        instrument: InstrumentSpec | None = ...,
        oms_type: str = ...,
        account_type: str = ...,
        leverage: float = ...,
    ) -> None: ...
    def step(
        self,
        idx: int,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        entry: bool = ...,
        exit: bool = ...,
        atr: float = ...,
        size_mult: float | None = ...,
        stop_price: float | None = ...,
        target_price: float | None = ...,
    ) -> list[EngineEvent]: ...
    def rest(
        self,
        idx: int,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> list[EngineEvent]: ...
    def submit_order(
        self,
        side: str,
        kind: str,
        submitted_idx: int,
        submitted_ts: int,
        client_id: str,
        units: float | None = ...,
        size_frac: float | None = ...,
        limit_price: float | None = ...,
        trigger_price: float | None = ...,
        tif: str = ...,
        expire_ns: int | None = ...,
        stop_price: float | None = ...,
        target_price: float | None = ...,
        offset: float | None = ...,
        offset_kind: str = ...,
        limit_offset: float = ...,
        post_only: bool = ...,
        reduce_only: bool = ...,
        parent_id: int | None = ...,
    ) -> int: ...
    def link_oco(self, order_ids: list[int]) -> None: ...
    def cancel_order(self, idx: int, order_id: int) -> bool: ...
    def cancel_all_orders(self, idx: int) -> list[int]: ...
    def submit_twap(
        self,
        units: float,
        side: str,
        slices: int,
        interval_ns: int,
        submitted_idx: int,
        submitted_ts: int,
        client_id: str,
        reduce_only: bool = ...,
    ) -> int: ...
    def cancel_twap(self, algo_id: int, idx: int) -> bool: ...
    def set_underlying_price(self, price: float | None = ...) -> None: ...
    def modify_order(
        self,
        order_id: int,
        units: float | None = ...,
        size_frac: float | None = ...,
        limit_price: float | None = ...,
        trigger_price: float | None = ...,
    ) -> bool: ...
    def open_order_ids(self) -> list[int]: ...
    def positions(self) -> list[PositionSnapshot]: ...
    def request_close(self, position_id: int) -> None: ...
    def free_capital(self) -> float: ...
    def halt(self, reason: str = ..., flatten: bool = ...) -> None: ...
    def is_halted(self) -> bool: ...
    def set_corporate_actions(self, actions: list[CorporateAction], policy: str) -> None: ...
    @property
    def dividends_received(self) -> float: ...
    def set_stop_price(
        self, price: float | None, position_id: int | None = ...
    ) -> None: ...
    def set_target_price(
        self, price: float | None, position_id: int | None = ...
    ) -> None: ...
    def equity(self) -> float: ...
    def cash(self) -> float: ...
    def is_in_position(self) -> bool: ...
    def position(self) -> PositionSnapshot | None: ...
    def finish(self) -> BacktestResult: ...

def resolve_atr_period(
    config: BacktestConfig | None = ...,
    instrument_config: InstrumentConfig | None = ...,
) -> int | None: ...


class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    def as_dict(self) -> dict[str, float]: ...

def pricing_price(
    model: str,
    is_call: bool,
    underlying: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float = ...,
    dividend_yield: float = ...,
) -> float: ...
def pricing_greeks(
    model: str,
    is_call: bool,
    underlying: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float = ...,
    dividend_yield: float = ...,
) -> Greeks: ...
def pricing_implied_vol(
    model: str,
    price: float,
    is_call: bool,
    underlying: float,
    strike: float,
    tau: float,
    rate: float = ...,
    dividend_yield: float = ...,
    lo: float = ...,
    hi: float = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> tuple[float, int]: ...
def pricing_price_array(
    model: str,
    is_call: _Bool,
    underlying: _F64,
    strike: _F64,
    tau: _F64,
    vol: _F64,
    rate: _F64,
    dividend_yield: _F64,
) -> _F64: ...
def pricing_greeks_array(
    model: str,
    is_call: _Bool,
    underlying: _F64,
    strike: _F64,
    tau: _F64,
    vol: _F64,
    rate: _F64,
    dividend_yield: _F64,
) -> tuple[_F64, _F64, _F64, _F64, _F64, _F64]: ...
def pricing_implied_vol_array(
    model: str,
    price: _F64,
    is_call: _Bool,
    underlying: _F64,
    strike: _F64,
    tau: _F64,
    rate: _F64,
    dividend_yield: _F64,
    lo: float = ...,
    hi: float = ...,
    tol: float = ...,
    max_iter: int = ...,
) -> tuple[_F64, npt.NDArray[np.uint8]]: ...
def pricing_status_name(code: int) -> str: ...
def pricing_year_fraction(now_ns: int, expiry_ns: int) -> float: ...
def pricing_forward(
    spot: float, tau: float, rate: float = ..., dividend_yield: float = ...
) -> float: ...

class OptionChain:
    def __init__(self, specs: Sequence[InstrumentSpec]) -> None: ...
    def __len__(self) -> int: ...
    def underlyings(self) -> list[str]: ...
    def expiries(self, underlying: str, now_ns: int) -> list[int]: ...
    def strikes(
        self, underlying: str, expiry_ns: int, now_ns: int, right: str | None = ...
    ) -> list[float]: ...
    def contract(
        self, underlying: str, expiry_ns: int, strike: float, right: str, now_ns: int
    ) -> InstrumentSpec | None: ...
    def contracts(
        self, underlying: str, expiry_ns: int, now_ns: int, right: str | None = ...
    ) -> list[InstrumentSpec]: ...
    def pick_expiry(
        self,
        underlying: str,
        now_ns: int,
        which: str = ...,
        n: int = ...,
        min_days: int = ...,
        date_ns: int | None = ...,
        utc_offset_ns: int = ...,
    ) -> int | None: ...
    def atm_strike(
        self, underlying: str, expiry_ns: int, now_ns: int, reference: float
    ) -> float | None: ...
    def atm_strike_forward(
        self,
        underlying: str,
        expiry_ns: int,
        now_ns: int,
        spot: float,
        rate: float = ...,
        dividend_yield: float = ...,
    ) -> float | None: ...
    def strike_at_offset(
        self, underlying: str, expiry_ns: int, now_ns: int, reference: float, offset: int
    ) -> float | None: ...
    def by_delta(
        self,
        underlying: str,
        expiry_ns: int,
        now_ns: int,
        right: str,
        target: float,
        underlying_price: float,
        vol: float,
        rate: float = ...,
        dividend_yield: float = ...,
        model: str = ...,
    ) -> InstrumentSpec | None: ...
    def closest_by(
        self,
        underlying: str,
        expiry_ns: int,
        now_ns: int,
        right: str,
        target: float,
        measure: Any,
    ) -> InstrumentSpec | None: ...

# --- research statistics and the full report ----------------------------------------------------

def deflated_sharpe(
    sharpes: Sequence[float],
    best: float,
    *,
    samples: int,
    skew: float = ...,
    kurtosis: float = ...,
    periods_per_year: float = ...,
    luck: Literal["trials", "samples"] = ...,
) -> dict[str, Any]: ...
def stationary_bootstrap(
    returns: Sequence[float],
    *,
    draws: int = ...,
    block: float | None = ...,
    seed: int,
    periods_per_year: float = ...,
) -> dict[str, Any]: ...
def resample_paths(
    values: Any,
    method: str,
    draws: int,
    block: float | None,
    replace: bool,
    seed: int,
    periods_per_year: float | None,
    drawdown_levels: Sequence[float],
) -> dict[str, Any]: ...
def without_best(returns: Sequence[float], top: int) -> float: ...
def walk_forward_windows(
    days: Sequence[Any], *, train: int, test: int, step: int | None = ...
) -> list[dict[str, Any]]: ...
def design_grid(space: dict[str, Sequence[Any]]) -> list[dict[str, Any]]: ...
def design_random(
    space: dict[str, Sequence[Any] | tuple[float, float]], n: int, *, seed: int
) -> list[dict[str, Any]]: ...
def full_report(
    equity_ts: Sequence[int],
    equity: Sequence[float],
    trades: Any = ...,
    orders: Any = ...,
    fees: Any = ...,
    periods_per_year: float | None = ...,
    risk_free: float = ...,
    session_minutes: float | None = ...,
    tz_offset_ns: int = ...,
    *,
    slippage: float = ...,
    multipliers: dict[str, float] | None = ...,
    exposure: Sequence[tuple[int, float, float]] | None = ...,
    exposure_time: tuple[int, int] | None = ...,
    benchmark: Sequence[tuple[int, float]] | None = ...,
    period: tuple[Any, Any] | None = ...,
    attribution: bool = ...,
    rule_metrics: Any = ...,
    unrealised: dict[str, float] | None = ...,
) -> dict[str, Any]: ...

# --- resampling ---------------------------------------------------------------------------------

class ResampledBar:
    @property
    def stream(self) -> str | None: ...
    @property
    def ts(self) -> int: ...
    @property
    def open(self) -> float: ...
    @property
    def high(self) -> float: ...
    @property
    def low(self) -> float: ...
    @property
    def close(self) -> float: ...
    @property
    def volume(self) -> float: ...
    @property
    def period(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def closed_by(self) -> Literal["clock", "print", "carry", "session", "flush"]: ...
    @property
    def n_source(self) -> int: ...
    @property
    def complete(self) -> bool: ...
    def as_tuple(self) -> tuple[int, float, float, float, float, float]: ...

class Resampler:
    def __init__(
        self,
        width_ns: int,
        *,
        source_ns: int = ...,
        anchor_ns: int = ...,
        tz_offset_ns: int = ...,
        closed: Literal["left", "right"] = ...,
        gaps: Literal["skip", "carry"] = ...,
        close_on: Literal["clock", "next_print"] = ...,
        min_source: int = ...,
        session_anchored: bool = ...,
        stream: str | None = ...,
    ) -> None: ...
    @property
    def width_ns(self) -> int: ...
    @property
    def source_ns(self) -> int: ...
    @property
    def closed(self) -> str: ...
    @property
    def gaps(self) -> str: ...
    @property
    def close_on(self) -> str: ...
    def push_trade(self, ts: int, price: float, qty: float) -> list[ResampledBar]: ...
    def push_bar(
        self,
        ts: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = ...,
        period: int = ...,
    ) -> list[ResampledBar]: ...
    def on_clock(self, ts: int) -> list[ResampledBar]: ...
    def on_session(self, open: bool, ts: int) -> list[ResampledBar]: ...
    def next_close(self) -> int | None: ...
    def flush(self) -> list[ResampledBar]: ...

def resample_bars(
    timestamps: _I64,
    open: _F64,
    high: _F64,
    low: _F64,
    close: _F64,
    volume: _F64,
    width_ns: int,
    *,
    source_ns: int = ...,
    anchor_ns: int = ...,
    tz_offset_ns: int = ...,
    closed: Literal["left", "right"] = ...,
    gaps: Literal["skip", "carry"] = ...,
    min_source: int = ...,
    flush: bool = ...,
) -> dict[str, np.ndarray]: ...
def resample_trades(
    timestamps: _I64,
    price: _F64,
    qty: _F64,
    width_ns: int,
    *,
    anchor_ns: int = ...,
    tz_offset_ns: int = ...,
    closed: Literal["left", "right"] = ...,
    gaps: Literal["skip", "carry"] = ...,
    min_source: int = ...,
    flush: bool = ...,
) -> dict[str, np.ndarray]: ...


class Record:
    seq: int
    ts_event: int
    ts_arrival: int
    kind: str
    source: str
    client_order_id: str | None
    chain: str
    payload: Any
    def canonical(self) -> bytes: ...

class Journal:
    def __init__(
        self,
        header: Any = None,
        *,
        seq_base: int = 0,
        chain: str | None = None,
        file: str | None = None,
        fsync: str | int | None = None,
        memory: bool | int | None = None,
        sink: Any = None,
    ) -> None: ...
    def since(self, seq: int) -> list[Record]: ...
    def head(self) -> tuple[int, str, Any]: ...
    def append(
        self,
        kind: str,
        ts_event: int,
        payload: Any = None,
        *,
        ts_arrival: int | None = None,
        source: str = "host",
        client_order_id: str | None = None,
    ) -> int: ...
    def sync(self) -> None: ...
    def expect(self, records: list[Any]) -> None: ...
    def peek(self) -> Record | None: ...
    count: int
    chain: str
    chain_from: str
    seq_base: int
    broken: bool
    last_kind: str | None
    replaying: bool
    remaining: int
    @staticmethod
    def read(path: str) -> tuple[Any, int, str | None, list[Record], int]: ...

def chain_step(prev_hex: str, canonical: bytes) -> str: ...
def chain_start(header: Any) -> str: ...
def canonical(value: Any) -> bytes: ...

_INDICATOR_FUNCTIONS: list[str]

def capabilities_json() -> str: ...

# --- Expressions -------------------------------------------------------------

class ExprProgram:
    """A compiled expression. See ``raptorbt.expressions``."""

    def __init__(
        self,
        source: str,
        params: dict[str, float] | None = None,
        groups: list[str] | None = None,
        series: dict[str, int] | None = None,
    ) -> None: ...
    @property
    def is_panel(self) -> bool: ...
    @property
    def handles(self) -> list[str]: ...
    @property
    def output_kinds(self) -> list[str]: ...
    @property
    def source(self) -> str: ...
    @property
    def canonical(self) -> str: ...
    @property
    def hash(self) -> str: ...
    @property
    def warmup(self) -> int | None: ...
    @property
    def warmup_by_clock(self) -> list[tuple[str, int]]: ...
    def warmup_bars(self, bar_ns: int | None = ...) -> int | None: ...
    def state_bytes(self, rows: int = ..., names: int = ..., bindings: int = ...) -> int: ...
    @property
    def uses_at(self) -> bool: ...
    def bound_hash(
        self,
        bar_stamp: str | None = ...,
        tz_offset_ns: int = ...,
        session: SessionSchedule | None = ...,
    ) -> str: ...
    @property
    def outputs(self) -> list[str]: ...
    @property
    def fields(self) -> list[str]: ...
    def settle(self, eps: float) -> int: ...
    def evaluate(
        self,
        data: dict[str, Any],
        bar_stamp: str | None = ...,
        tz_offset_ns: int = ...,
        session: SessionSchedule | None = ...,
        bar_ns: int | None = ...,
    ) -> list[_F64]: ...
    def eval_panel(
        self,
        data: dict[str, Any],
        present: Any = ...,
        bar_stamp: str | None = ...,
        tz_offset_ns: int = ...,
        session: SessionSchedule | None = ...,
        bar_ns: int | None = ...,
    ) -> list[Any]: ...
    def stream(
        self,
        bar_stamp: str | None = ...,
        tz_offset_ns: int = ...,
        session: SessionSchedule | None = ...,
        bar_ns: int | None = ...,
    ) -> ExprStream: ...

class ExprStream:
    @property
    def ready(self) -> bool: ...
    @property
    def bars(self) -> int: ...
    def update(
        self, open: float, high: float, low: float, close: float, volume: float
    ) -> list[float]: ...
    def update_at(
        self, timestamp: int, open: float, high: float, low: float, close: float, volume: float
    ) -> list[float]: ...
    def reset(self) -> None: ...

def expr_registry_json() -> str: ...
def expr_param_names(source: str) -> list[str]: ...
def expr_sweep_limits() -> tuple[int, int]: ...
def expr_sweep(
    source: str,
    params: dict[str, float] | None,
    series: dict[str, int] | None,
    data: Any,
    grid: list[tuple[str, list[float]]] | None = ...,
    points: list[dict[str, float]] | None = ...,
    bar_stamp: str | None = ...,
    tz_offset_ns: int = ...,
    session: Any = ...,
    bar_ns: int | None = ...,
    max_combinations: int | None = ...,
    max_bytes: int | None = ...,
) -> dict[str, Any]: ...
def factor_report_native(
    factor: Any,
    prices: Any,
    horizons: list[int],
    quantiles: int,
    lag: int,
    min_names: int,
    jump_ratio: float,
    cost_bps_per_side: float | None,
) -> dict[str, Any]: ...

# --- Statistics --------------------------------------------------------------
# Natives behind ``raptorbt.stats``: arrays in, a dict of full-precision results out. The
# public functions, their wording and their argument checks are in that module.

def stats_stationarity(series: np.ndarray, alpha: float) -> dict[str, Any]: ...
def stats_cointegration(
    names: list[str], series: list[np.ndarray], alpha: float
) -> dict[str, Any]: ...
def stats_hurst(increments: np.ndarray) -> dict[str, Any]: ...
def stats_garch(
    returns: np.ndarray, model: str, dist: str, horizon: int, periods_per_year: int | None
) -> dict[str, Any]: ...
def stats_regimes(
    returns: np.ndarray,
    regimes: int,
    min_support_bars: float,
    min_duration_periods: float,
    series: bool,
) -> dict[str, Any]: ...
def stats_hac(returns: np.ndarray, lags: int | None) -> dict[str, Any]: ...
def stats_predictive(
    signal: np.ndarray,
    prices: np.ndarray,
    sessions: np.ndarray | None,
    horizons: list[int],
    lag: int,
    quantiles: int | None,
    cost_bps_per_side: float | None,
    to_session_close: bool,
) -> dict[str, Any]: ...
def stats_pbo(strategies: list[np.ndarray], splits: int) -> dict[str, Any]: ...
def stats_spa(
    names: list[str],
    strategies: list[np.ndarray],
    benchmark: np.ndarray | None,
    reps: int,
    block_size: int | None,
    seed: int,
) -> dict[str, Any]: ...
def stats_spa_replay(
    loss_diff: list[np.ndarray], block: int, indices: np.ndarray
) -> dict[str, Any]: ...
def stats_pca(
    names: list[str],
    returns: list[np.ndarray],
    components: int,
    target: np.ndarray | None,
) -> dict[str, Any]: ...
def stats_events(
    dates: list[int],
    asset: np.ndarray,
    market: np.ndarray,
    events: list[int],
    estimation: int,
    gap: int,
    pre: int,
    post: int,
) -> dict[str, Any]: ...
def stats_impact(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    quantity: float,
    spread_model: str,
    coefficient: float,
) -> dict[str, Any]: ...
def stats_surface(
    names: list[str], cells: list[tuple[list[float], float]]
) -> dict[str, Any]: ...
def stats_audit_bars(
    t: np.ndarray,
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    interval_seconds: int,
    outlier_z: float,
    mad_scale: float,
    listed: int,
    daily_gap_days: float,
    intraday_gap_bars: float,
    split_factors: list[float],
    split_tolerance: float,
    min_bars: int,
    dead_bar_share_max: float,
) -> dict[str, Any]: ...

def _last_panic_location() -> str | None: ...
