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

class PyStopConfig: ...
class PyTargetConfig: ...

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
    halted_at: int | None
    metrics: PyBacktestMetrics

class PyBatchSpreadItem:
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
def oi_position_pct(oi: _F64) -> _F64: ...
def tick_velocity(timestamps_ns: _I64, window_seconds: float = ...) -> _F64: ...

# --- Per-bar strategy session (class-based strategy contract) ---------------

class PyEngineEvent:
    kind: str
    idx: int
    price: float | None
    size: float | None
    direction: int | None
    trade: PyTrade | None
    reject_reason: str | None

class PyPositionSnapshot:
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
    def set_stop_price(self, price: float | None) -> None: ...
    def set_target_price(self, price: float | None) -> None: ...
    def equity(self) -> float: ...
    def cash(self) -> float: ...
    def is_in_position(self) -> bool: ...
    def position(self) -> PyPositionSnapshot | None: ...
    def finish(self) -> PyBacktestResult: ...

def resolve_atr_period(
    config: PyBacktestConfig | None = ...,
    instrument_config: PyInstrumentConfig | None = ...,
) -> int | None: ...
