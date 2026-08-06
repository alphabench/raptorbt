"""Type stubs for the raptorbt Rust extension module.

Covers the surface callers integrate against: configuration, results, and the
backtest runners. Indicator and tick helpers are declared with their array
signatures at the bottom.
"""

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]
_Bool = npt.NDArray[np.bool_]

# Trading minutes per session, for PyBacktestConfig(session_minutes=...).
SESSION_NSE: float
SESSION_MCX: float
SESSION_CDS: float
SESSION_CONTINUOUS: float
IST_OFFSET_NS: int

# An instrument for the basket/portfolio runners:
# (timestamps, open, high, low, close, volume, entries, exits, direction, weight, symbol)
_Instrument = tuple[_I64, _F64, _F64, _F64, _F64, _F64, _Bool, _Bool, int, float, str]

class PyBacktestConfig:
    initial_capital: float
    fees: float
    slippage: float
    upon_bar_close: bool
    apply_slippage: bool
    periods_per_year: float | None
    risk_free_rate: float
    session_minutes: float | None
    fee_segment: str | None
    max_positions: int | None
    max_drawdown_pct: float | None
    fill_prob_limit: float
    fill_prob_slippage: float
    queue_fill_model: bool
    session_tz_offset_ns: int
    limit_slippage: float
    liquidate_on_margin_call: bool
    fill_seed: int
    bar_path_adaptive: bool
    legacy_annualization: bool

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
    ) -> None: ...
    def set_fixed_stop(self, percent: float) -> None: ...
    def set_atr_stop(self, multiplier: float, period: int) -> None: ...
    def set_trailing_stop(self, percent: float) -> None: ...
    def set_fixed_target(self, percent: float) -> None: ...
    def set_atr_target(self, multiplier: float, period: int) -> None: ...
    def set_risk_reward_target(self, ratio: float) -> None: ...
    def set_session_config(self, *args: Any, **kwargs: Any) -> None: ...

class PyInstrumentConfig:
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

class PyStopConfig:
    stop_type: str
    percent: float | None
    multiplier: float | None
    period: int | None
    @staticmethod
    def fixed(percent: float) -> PyStopConfig: ...
    @staticmethod
    def atr(multiplier: float, period: int) -> PyStopConfig: ...
    @staticmethod
    def trailing(percent: float) -> PyStopConfig: ...

class PyTargetConfig:
    target_type: str
    percent: float | None
    multiplier: float | None
    period: int | None
    ratio: float | None
    @staticmethod
    def fixed(percent: float) -> PyTargetConfig: ...
    @staticmethod
    def atr(multiplier: float, period: int) -> PyTargetConfig: ...
    @staticmethod
    def risk_reward(ratio: float) -> PyTargetConfig: ...

class PyTrade:
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
    fees: float
    # Present only when config.fee_segment selects an itemized schedule.
    # Keys: brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, total.
    fee_breakdown: dict[str, float] | None
    exit_reason: str

class PyBacktestMetrics:
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    win_rate_pct: float
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
    avg_win_pct: float
    avg_loss_pct: float
    avg_winning_duration: float
    avg_losing_duration: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_period: float
    exposure_pct: float

    # None when the denominator is zero -- e.g. profit factor with no losing
    # trades. Previously these were float('inf'), which is not JSON-serializable.
    sortino_ratio: float | None
    calmar_ratio: float | None
    omega_ratio: float | None
    profit_factor: float | None
    payoff_ratio: float | None
    recovery_factor: float | None

    def to_dict(self) -> dict[str, Any]: ...

class PyBacktestResult:
    metrics: PyBacktestMetrics

    def equity_curve(self) -> list[float]: ...
    def drawdown_curve(self) -> list[float]: ...
    def trades(self) -> list[PyTrade]: ...
    def returns(self) -> list[float]: ...

class PyInstrumentSummary:
    symbol: str
    trades: int
    pnl: float
    rejected_entries: int

class PyPortfolioResult:
    result: PyBacktestResult
    per_instrument: list[PyInstrumentSummary]
    rejected_entries: int
    halted: bool
    # Bar index in array runs; a schedule-event ordinal in session runs
    # (``run_portfolio_strategy``), which interleaves N instrument streams.
    halted_at: int | None
    metrics: PyBacktestMetrics

class PyBatchSpreadItem:
    strategy_id: str
    spread_type: str
    max_loss: float | None
    target_profit: float | None
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

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
    config: PyBacktestConfig | None = ...,
    position_sizes: _F64 | None = ...,
    instrument_config: PyInstrumentConfig | None = ...,
) -> PyBacktestResult: ...
def run_basket_backtest(
    instruments: Sequence[_Instrument],
    config: PyBacktestConfig | None = ...,
    sync_mode: str = ...,
    instrument_configs: dict[str, PyInstrumentConfig] | None = ...,
) -> PyBacktestResult: ...
def run_portfolio_backtest(
    instruments: Sequence[_Instrument],
    config: PyBacktestConfig | None = ...,
    allocation: str = ...,
    instrument_configs: dict[str, PyInstrumentConfig] | None = ...,
) -> PyPortfolioResult:
    """Simulate instruments against one shared capital pool.

    Unlike summing independent per-symbol runs, capital is shared, so
    `max_positions` and the drawdown kill-switch on `config` are enforceable.
    `allocation` is "equal_weight" or "full".
    """

def run_options_backtest(*args: Any, **kwargs: Any) -> PyBacktestResult: ...
def run_pairs_backtest(*args: Any, **kwargs: Any) -> PyBacktestResult: ...
def run_multi_backtest(*args: Any, **kwargs: Any) -> PyBacktestResult: ...
def run_spread_backtest(*args: Any, **kwargs: Any) -> PyBacktestResult: ...
def run_tick_backtest(*args: Any, **kwargs: Any) -> PyBacktestResult: ...
def batch_spread_backtest(*args: Any, **kwargs: Any) -> list[PyBacktestResult]: ...
def simulate_portfolio_mc(
    returns: _F64,
    weights: _F64,
    correlation_matrix: _F64,
    initial_value: float,
    n_simulations: int = ...,
    horizon_days: int = ...,
    seed: int = ...,
) -> dict[str, Any]: ...

# --- Portfolio math ----------------------------------------------------------

class PyRiskModel:
    asset_ids: list[str]
    n_assets: int
    periods_per_year: float
    shrinkage_intensity: float
    n_obs: int
    def cov(self) -> _F64: ...

class PyOptimizerConfig:
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

class PyOptimizationResult:
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

class PyRiskContributions:
    total_vol_annualized: float
    def marginal(self) -> _F64: ...
    def contribution(self) -> _F64: ...
    def pct_contribution(self) -> _F64: ...

class PyOptimizeItem:
    def __init__(
        self,
        item_id: str,
        alpha: _F64,
        w_current: _F64,
        portfolio_value: float | None = ...,
    ) -> None: ...

class PyRankIc:
    mean_ic: float
    stdev_ic: float
    t_stat: float
    t_stat_deflated: float
    n_dates_scored: int
    n_independent: float
    overlap_days: int
    mean_names: float
    def daily_ic(self) -> _F64: ...

class PyRebalanceSimResult:
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
) -> PyRiskModel: ...
def optimize_portfolio(
    model: PyRiskModel,
    alpha: _F64,
    w_current: _F64,
    asset_ids: Sequence[str],
    config: PyOptimizerConfig,
) -> PyOptimizationResult: ...
def batch_optimize_portfolios(
    model: PyRiskModel,
    items: Sequence[PyOptimizeItem],
    config: PyOptimizerConfig,
) -> list[tuple[str, PyOptimizationResult]]: ...
def compute_risk_contributions(
    model: PyRiskModel,
    weights: _F64,
    asset_ids: Sequence[str],
) -> PyRiskContributions: ...
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
) -> PyRankIc: ...
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
) -> PyRebalanceSimResult: ...
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
) -> tuple[_F64, _F64, _F64]: ...
def adx(high: _F64, low: _F64, close: _F64, period: int) -> _F64: ...
def vwap(high: _F64, low: _F64, close: _F64, volume: _F64) -> _F64: ...
def supertrend(
    high: _F64, low: _F64, close: _F64, period: int = ..., multiplier: float = ...
) -> tuple[_F64, _F64]: ...
def rolling_min(data: _F64, window: int) -> _F64: ...
def rolling_max(data: _F64, window: int) -> _F64: ...

# --- Tick signals and features ---------------------------------------------

def compute_tick_entry_signals(*args: Any, **kwargs: Any) -> _Bool: ...
def compute_tick_exit_signals(
    timestamps_ns: _I64, eod_exit_time_ns: int = ...
) -> _Bool: ...
def tick_spread_pct(bid: _F64, ask: _F64) -> _F64: ...
def buy_sell_imbalance_delta(buy_qty_delta: _F64, sell_qty_delta: _F64) -> _F64: ...
def return_window(timestamps_ns: _I64, ltp: _F64, window_seconds: float = ...) -> _F64: ...
def realized_vol_rolling(
    timestamps_ns: _I64, ltp: _F64, window_seconds: float = ...
) -> _F64: ...
def oi_position_pct(oi: _F64, oi_day_high: float, oi_day_low: float) -> _F64: ...
def tick_velocity(timestamps_ns: _I64, window_seconds: float = ...) -> _F64: ...

# --- Instrument market definitions ------------------------------------------

class InstrumentSpec:
    settlement_fee: float
    symbol: str
    kind: str
    price_increment: float
    size_increment: float
    lot_size: float
    multiplier: float
    margin_init: float
    margin_maint: float
    maker_fee: float
    taker_fee: float
    activation_ns: int | None
    expiration_ns: int | None
    strike: float | None
    right: str | None
    underlying: str | None
    tradable: bool
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
    ) -> InstrumentSpec: ...
    @staticmethod
    def index(symbol: str, price_increment: float = ...) -> InstrumentSpec: ...

# --- Streaming indicators ----------------------------------------------------

class Indicator:
    kind: str
    value: Any | None
    initialized: bool
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
    def update_bar(self, open: float, high: float, low: float, close: float) -> Any | None: ...
    def reset(self) -> None: ...

# --- Bar aggregation ---------------------------------------------------------

_BarArrays = tuple[_I64, _F64, _F64, _F64, _F64, _F64]

class BarAggregator:
    step: int
    unit: str
    def __init__(
        self, step: int, unit: str, tz_offset_ns: int = ...,
        brick_size: float = ...,
    ) -> None: ...
    def push_bar(
        self, timestamp: int, open: float, high: float, low: float,
        close: float, volume: float,
    ) -> tuple[int, float, float, float, float, float] | None: ...
    def push_trade(
        self, timestamp: int, price: float, size: float, signed_size: float = ...,
    ) -> tuple[int, float, float, float, float, float] | None: ...
    # Renko completes several bricks at once; drain after every push.
    def next_pending(self) -> tuple[int, float, float, float, float, float] | None: ...
    def flush(self) -> tuple[int, float, float, float, float, float] | None: ...

def aggregate_bars(
    timestamps: _I64, open: _F64, high: _F64, low: _F64, close: _F64,
    volume: _F64, step: int, unit: str, tz_offset_ns: int = ...,
    brick_size: float = ...,
) -> _BarArrays: ...
def bars_from_ticks(
    timestamps: _I64, ltp: _F64, buy_qty_delta: _F64, sell_qty_delta: _F64,
    step: int, unit: str, tz_offset_ns: int = ..., brick_size: float = ...,
) -> _BarArrays: ...

class PyPortfolioSession:
    def __init__(
        self, config: PyBacktestConfig | None = ...,
        account_type: str = ..., leverage: float = ...,
    ) -> None: ...
    def add_instrument(
        self, symbol: str, direction: int = ...,
        instrument_config: PyInstrumentConfig | None = ...,
        instrument: InstrumentSpec | None = ..., oms_type: str = ...,
    ) -> int: ...
    def set_bars(
        self, instrument: int, timestamps: _I64, open: _F64, high: _F64,
        low: _F64, close: _F64, volume: _F64,
    ) -> None: ...
    def set_ticks(
        self, instrument: int, timestamps: _I64, ltp: _F64,
        bid: _F64 | None = ..., ask: _F64 | None = ...,
        buy_qty_delta: _F64 | None = ..., sell_qty_delta: _F64 | None = ...,
    ) -> None: ...
    def set_depth(
        self, instrument: int, timestamps: _I64,
        bid_prices: Any, bid_sizes: Any, ask_prices: Any, ask_sizes: Any,
    ) -> None: ...
    def current_depth(
        self,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None: ...
    def seal(self) -> None: ...
    # Incremental (live) feed: append to the schedule tail in arrival order.
    # Each seals first and is idempotent, so batch warmup data merges ahead of
    # the first push. Drive appended events with current_event()/apply_current().
    # push_tick returns how many events it appended (0-2): a trade print, plus
    # a quote when ask > 0.
    def push_tick(
        self, instrument: int, timestamp: int, ltp: float,
        bid: float = ..., ask: float = ...,
        buy_qty_delta: float = ..., sell_qty_delta: float = ...,
    ) -> int: ...
    def push_bar(
        self, instrument: int, timestamp: int, open: float, high: float,
        low: float, close: float, volume: float,
    ) -> None: ...
    # bids/asks are (price, size) lists, best level first.
    def push_depth(
        self, instrument: int, timestamp: int,
        bids: Sequence[tuple[float, float]], asks: Sequence[tuple[float, float]],
    ) -> None: ...
    # Events pushed or merged but not yet applied.
    def remaining(self) -> int: ...
    def __len__(self) -> int: ...
    # Bar sessions only; returns None on a tick event.
    def current(self) -> tuple[int, int, int, float, float, float, float, float] | None: ...
    # (kind, instrument, local_idx, ts, a, b, c, d, e); kind is
    # "bar" (o/h/l/c/v), "trade" (price, size, ...) or "quote" (bid, ask, ...).
    def current_event(
        self,
    ) -> tuple[str, int, int, int, float, float, float, float, float] | None: ...
    def apply_current(
        self, entry: bool = ..., exit: bool = ..., atr: float = ...,
        size_mult: float | None = ..., stop_price: float | None = ...,
        target_price: float | None = ...,
    ) -> list[PyEngineEvent]: ...
    def submit_order(self, instrument: int, *args: Any, **kwargs: Any) -> int: ...
    def cancel_order(self, instrument: int, idx: int, order_id: int) -> bool: ...
    def cancel_all_orders(self, instrument: int, idx: int) -> list[int]: ...
    def modify_order(
        self, instrument: int, order_id: int, units: float | None = ...,
        size_frac: float | None = ..., limit_price: float | None = ...,
        trigger_price: float | None = ...,
    ) -> bool: ...
    def link_oco(self, instrument: int, order_ids: list[int]) -> None: ...
    # Adopt a position the account already holds (broker-truth seeding): no
    # order, no fill, no fees, no trade record. Cash mode debits the cost
    # basis; a fully funded margin book locks it instead. Cash or leverage-1.0
    # margin, long-only. Must be called after seal() and before the first
    # apply_current() — enforced, since adopting mid-run understates max
    # drawdown. Returns the new position id.
    def adopt_position(
        self, instrument: int, timestamp_ns: int, price: float, size: float,
    ) -> int: ...
    def request_close(self, instrument: int, position_id: int) -> None: ...
    def set_underlying_price(
        self, instrument: int, price: float | None = ...,
    ) -> None: ...
    def positions(self, instrument: int) -> list[PyPositionSnapshot]: ...
    def position(self, instrument: int) -> PyPositionSnapshot | None: ...
    def equity(self) -> float: ...
    def cash(self) -> float: ...
    def free_capital(self) -> float: ...
    def is_halted(self) -> bool: ...
    def finish(self) -> PyPortfolioResult: ...

# --- Per-bar strategy session (class-based strategy contract) ---------------

class PyEngineEvent:
    kind: str
    idx: int
    price: float | None
    size: float | None
    direction: int | None
    trade: PyTrade | None
    reject_reason: str | None
    order_id: int | None
    client_order_id: str | None

class PyPositionSnapshot:
    position_id: int
    entry_idx: int
    entry_price: float
    size: float
    direction: int
    stop_price: float | None
    target_price: float | None

class PyKernelSession:
    def __init__(
        self,
        symbol: str = ...,
        direction: int = ...,
        config: PyBacktestConfig | None = ...,
        instrument_config: PyInstrumentConfig | None = ...,
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
    ) -> list[PyEngineEvent]: ...
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
        self, units: float, side: str, slices: int, interval_ns: int,
        submitted_idx: int, submitted_ts: int, client_id: str,
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
    def positions(self) -> list[PyPositionSnapshot]: ...
    def request_close(self, position_id: int) -> None: ...
    def free_capital(self) -> float: ...
    def set_stop_price(self, price: float | None, position_id: int | None = ...) -> None: ...
    def set_target_price(self, price: float | None, position_id: int | None = ...) -> None: ...
    def equity(self) -> float: ...
    def cash(self) -> float: ...
    def is_in_position(self) -> bool: ...
    def position(self) -> PyPositionSnapshot | None: ...
    def finish(self) -> PyBacktestResult: ...

def resolve_atr_period(
    config: PyBacktestConfig | None = ...,
    instrument_config: PyInstrumentConfig | None = ...,
) -> int | None: ...
