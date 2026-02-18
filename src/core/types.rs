//! Core data types for RaptorBT.

use serde::{Deserialize, Serialize};

/// Type alias for price values.
pub type Price = f64;

/// Type alias for timestamp values (nanoseconds since epoch).
pub type Timestamp = i64;

/// Trading direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i8)]
pub enum Direction {
    /// Long position (buy to open, sell to close).
    Long = 1,
    /// Short position (sell to open, buy to close).
    Short = -1,
}

impl Direction {
    /// Convert direction to multiplier for P&L calculations.
    #[inline]
    pub fn multiplier(self) -> f64 {
        self as i8 as f64
    }

    /// Create direction from integer.
    pub fn from_int(value: i32) -> Option<Self> {
        match value {
            1 => Some(Direction::Long),
            -1 => Some(Direction::Short),
            _ => None,
        }
    }
}

impl Default for Direction {
    fn default() -> Self {
        Direction::Long
    }
}

/// OHLCV data for a single bar.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OhlcvBar {
    pub timestamp: Timestamp,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: f64,
}

/// OHLCV data series.
#[derive(Debug, Clone)]
pub struct OhlcvData {
    pub timestamps: Vec<Timestamp>,
    pub open: Vec<Price>,
    pub high: Vec<Price>,
    pub low: Vec<Price>,
    pub close: Vec<Price>,
    pub volume: Vec<f64>,
}

impl OhlcvData {
    /// Create new OHLCV data from vectors.
    pub fn new(
        timestamps: Vec<Timestamp>,
        open: Vec<Price>,
        high: Vec<Price>,
        low: Vec<Price>,
        close: Vec<Price>,
        volume: Vec<f64>,
    ) -> Self {
        Self { timestamps, open, high, low, close, volume }
    }

    /// Get the number of bars.
    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    /// Get a single bar at index.
    pub fn get_bar(&self, index: usize) -> Option<OhlcvBar> {
        if index >= self.len() {
            return None;
        }
        Some(OhlcvBar {
            timestamp: self.timestamps[index],
            open: self.open[index],
            high: self.high[index],
            low: self.low[index],
            close: self.close[index],
            volume: self.volume[index],
        })
    }
}

/// Compiled trading signals from strategy.
#[derive(Debug, Clone)]
pub struct CompiledSignals {
    /// Symbol identifier.
    pub symbol: String,
    /// Entry signals (true = enter position).
    pub entries: Vec<bool>,
    /// Exit signals (true = exit position).
    pub exits: Vec<bool>,
    /// Optional position sizes (fraction of capital).
    pub position_sizes: Option<Vec<f64>>,
    /// Trading direction.
    pub direction: Direction,
    /// Weight for portfolio allocation.
    pub weight: f64,
}

impl CompiledSignals {
    /// Create new compiled signals.
    pub fn new(
        symbol: String,
        entries: Vec<bool>,
        exits: Vec<bool>,
        direction: Direction,
        weight: f64,
    ) -> Self {
        Self { symbol, entries, exits, position_sizes: None, direction, weight }
    }

    /// Set position sizes.
    pub fn with_position_sizes(mut self, sizes: Vec<f64>) -> Self {
        self.position_sizes = Some(sizes);
        self
    }

    /// Get the number of bars.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A single executed trade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade identifier.
    pub id: u64,
    /// Symbol traded.
    pub symbol: String,
    /// Entry bar index.
    pub entry_idx: usize,
    /// Exit bar index.
    pub exit_idx: usize,
    /// Entry price.
    pub entry_price: Price,
    /// Exit price.
    pub exit_price: Price,
    /// Position size (number of shares/contracts).
    pub size: f64,
    /// Trading direction.
    pub direction: Direction,
    /// Realized profit/loss.
    pub pnl: f64,
    /// Return percentage.
    pub return_pct: f64,
    /// Entry timestamp.
    pub entry_time: Timestamp,
    /// Exit timestamp.
    pub exit_time: Timestamp,
    /// Fees paid.
    pub fees: f64,
    /// Exit reason.
    pub exit_reason: ExitReason,
}

impl Trade {
    /// Check if trade was profitable.
    #[inline]
    pub fn is_winning(&self) -> bool {
        self.pnl > 0.0
    }

    /// Get holding period in bars.
    #[inline]
    pub fn holding_period(&self) -> usize {
        self.exit_idx - self.entry_idx
    }
}

/// Reason for exiting a trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExitReason {
    /// Normal exit signal.
    Signal,
    /// Stop-loss hit.
    StopLoss,
    /// Take-profit hit.
    TakeProfit,
    /// Trailing stop hit.
    TrailingStop,
    /// End of data.
    EndOfData,
}

/// Backtest configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital.
    pub initial_capital: f64,
    /// Transaction fees as fraction (0.001 = 0.1%).
    pub fees: f64,
    /// Slippage as fraction.
    pub slippage: f64,
    /// Stop-loss configuration.
    pub stop: StopConfig,
    /// Take-profit configuration.
    pub target: TargetConfig,
    /// Whether to execute on bar close.
    pub upon_bar_close: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            fees: 0.001,
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
        }
    }
}

/// Per-instrument configuration for position sizing and risk management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentConfig {
    /// Minimum tradeable quantity (1.0 for NSE EQ, 50.0 for NIFTY F&O, 0.01 for forex).
    pub lot_size: Option<f64>,
    /// Per-instrument capital cap.
    pub alloted_capital: Option<f64>,
    /// Per-instrument stop override.
    pub stop: Option<StopConfig>,
    /// Per-instrument target override.
    pub target: Option<TargetConfig>,
    /// Existing position quantity (future use).
    pub existing_qty: Option<f64>,
    /// Existing position average price (future use).
    pub avg_price: Option<f64>,
}

impl InstrumentConfig {
    /// Round a raw position size down to the nearest lot_size multiple.
    /// Returns raw_size unchanged if lot_size is None or <= 0.
    pub fn round_to_lot(&self, raw_size: f64) -> f64 {
        match self.lot_size {
            Some(lot) if lot > 0.0 => (raw_size / lot).floor() * lot,
            _ => raw_size,
        }
    }
}

impl Default for InstrumentConfig {
    fn default() -> Self {
        Self {
            lot_size: None,
            alloted_capital: None,
            stop: None,
            target: None,
            existing_qty: None,
            avg_price: None,
        }
    }
}

/// Stop-loss configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StopConfig {
    /// No stop-loss.
    None,
    /// Fixed percentage stop.
    Fixed { percent: f64 },
    /// ATR-based stop.
    Atr { multiplier: f64, period: usize },
    /// Trailing stop.
    Trailing { percent: f64 },
}

/// Take-profit configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TargetConfig {
    /// No take-profit.
    None,
    /// Fixed percentage target.
    Fixed { percent: f64 },
    /// ATR-based target.
    Atr { multiplier: f64, period: usize },
    /// Risk-reward ratio target.
    RiskReward { ratio: f64 },
}

/// Backtest metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BacktestMetrics {
    /// Total return percentage.
    pub total_return_pct: f64,
    /// Sharpe ratio (annualized).
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized).
    pub sortino_ratio: f64,
    /// Calmar ratio.
    pub calmar_ratio: f64,
    /// Omega ratio.
    pub omega_ratio: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown_pct: f64,
    /// Maximum drawdown duration in bars.
    pub max_drawdown_duration: usize,
    /// Win rate percentage.
    pub win_rate_pct: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Expectancy (average expected profit per trade).
    pub expectancy: f64,
    /// System Quality Number (SQN).
    pub sqn: f64,
    /// Total number of trades.
    pub total_trades: usize,
    /// Number of closed trades.
    pub total_closed_trades: usize,
    /// Number of open trades at end.
    pub total_open_trades: usize,
    /// PnL of open trades.
    pub open_trade_pnl: f64,
    /// Number of winning trades.
    pub winning_trades: usize,
    /// Number of losing trades.
    pub losing_trades: usize,
    /// Starting portfolio value.
    pub start_value: f64,
    /// Ending portfolio value.
    pub end_value: f64,
    /// Total fees paid.
    pub total_fees_paid: f64,
    /// Best trade return percentage.
    pub best_trade_pct: f64,
    /// Worst trade return percentage.
    pub worst_trade_pct: f64,
    /// Average trade return percentage.
    pub avg_trade_return_pct: f64,
    /// Average winning trade return percentage.
    pub avg_win_pct: f64,
    /// Average losing trade return percentage.
    pub avg_loss_pct: f64,
    /// Average winning trade duration in bars.
    pub avg_winning_duration: f64,
    /// Average losing trade duration in bars.
    pub avg_losing_duration: f64,
    /// Maximum consecutive wins.
    pub max_consecutive_wins: usize,
    /// Maximum consecutive losses.
    pub max_consecutive_losses: usize,
    /// Average holding period in bars.
    pub avg_holding_period: f64,
    /// Exposure time percentage (time in market).
    pub exposure_pct: f64,
    /// Payoff ratio (avg win / avg loss).
    pub payoff_ratio: f64,
    /// Recovery factor (net profit / max drawdown).
    pub recovery_factor: f64,
}

/// Complete backtest result.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Computed metrics.
    pub metrics: BacktestMetrics,
    /// Equity curve (portfolio value over time).
    pub equity_curve: Vec<f64>,
    /// Drawdown curve (drawdown percentage over time).
    pub drawdown_curve: Vec<f64>,
    /// List of executed trades.
    pub trades: Vec<Trade>,
    /// Daily returns.
    pub returns: Vec<f64>,
}

impl BacktestResult {
    /// Create a new backtest result.
    pub fn new(
        metrics: BacktestMetrics,
        equity_curve: Vec<f64>,
        drawdown_curve: Vec<f64>,
        trades: Vec<Trade>,
        returns: Vec<f64>,
    ) -> Self {
        Self { metrics, equity_curve, drawdown_curve, trades, returns }
    }
}

/// Position state during backtest.
#[derive(Debug, Clone)]
pub struct Position {
    /// Whether position is open.
    pub is_open: bool,
    /// Entry bar index.
    pub entry_idx: usize,
    /// Entry price.
    pub entry_price: Price,
    /// Position size.
    pub size: f64,
    /// Trading direction.
    pub direction: Direction,
    /// Current stop price.
    pub stop_price: Option<Price>,
    /// Current target price.
    pub target_price: Option<Price>,
    /// Highest price since entry (for trailing stops).
    pub highest_since_entry: Price,
    /// Lowest price since entry (for trailing stops).
    pub lowest_since_entry: Price,
    /// Entry fees (to include in trade PnL like VectorBT).
    pub entry_fees: f64,
}

impl Position {
    /// Create a new closed position state.
    pub fn new() -> Self {
        Self {
            is_open: false,
            entry_idx: 0,
            entry_price: 0.0,
            size: 0.0,
            direction: Direction::Long,
            stop_price: None,
            target_price: None,
            highest_since_entry: 0.0,
            lowest_since_entry: f64::MAX,
            entry_fees: 0.0,
        }
    }

    /// Open a new position.
    pub fn open(
        &mut self,
        idx: usize,
        price: Price,
        size: f64,
        direction: Direction,
        stop_price: Option<Price>,
        target_price: Option<Price>,
        entry_fees: f64,
    ) {
        self.is_open = true;
        self.entry_idx = idx;
        self.entry_price = price;
        self.size = size;
        self.direction = direction;
        self.stop_price = stop_price;
        self.target_price = target_price;
        self.highest_since_entry = price;
        self.lowest_since_entry = price;
        self.entry_fees = entry_fees;
    }

    /// Close the position.
    pub fn close(&mut self) {
        self.is_open = false;
    }

    /// Update highest/lowest prices for trailing stops.
    pub fn update_extremes(&mut self, high: Price, low: Price) {
        if high > self.highest_since_entry {
            self.highest_since_entry = high;
        }
        if low < self.lowest_since_entry {
            self.lowest_since_entry = low;
        }
    }

    /// Calculate unrealized P&L at given price.
    pub fn unrealized_pnl(&self, current_price: Price) -> f64 {
        if !self.is_open {
            return 0.0;
        }
        let price_change = current_price - self.entry_price;
        price_change * self.size * self.direction.multiplier()
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_lot_whole_shares() {
        let config = InstrumentConfig { lot_size: Some(1.0), ..Default::default() };
        assert_eq!(config.round_to_lot(242.47), 242.0);
        assert_eq!(config.round_to_lot(1.0), 1.0);
        assert_eq!(config.round_to_lot(0.5), 0.0);
    }

    #[test]
    fn test_round_to_lot_nifty_fo() {
        let config = InstrumentConfig { lot_size: Some(50.0), ..Default::default() };
        assert_eq!(config.round_to_lot(242.0), 200.0);
        assert_eq!(config.round_to_lot(50.0), 50.0);
        assert_eq!(config.round_to_lot(49.0), 0.0);
        assert_eq!(config.round_to_lot(150.0), 150.0);
    }

    #[test]
    fn test_round_to_lot_fractional() {
        let config = InstrumentConfig { lot_size: Some(0.01), ..Default::default() };
        assert!((config.round_to_lot(1.234) - 1.23).abs() < 1e-10);
    }

    #[test]
    fn test_round_to_lot_none() {
        let config = InstrumentConfig::default();
        assert_eq!(config.round_to_lot(242.47), 242.47);
    }

    #[test]
    fn test_round_to_lot_zero() {
        let config = InstrumentConfig { lot_size: Some(0.0), ..Default::default() };
        assert_eq!(config.round_to_lot(242.47), 242.47);
    }

    #[test]
    fn test_round_to_lot_negative() {
        let config = InstrumentConfig { lot_size: Some(-1.0), ..Default::default() };
        assert_eq!(config.round_to_lot(242.47), 242.47);
    }
}
