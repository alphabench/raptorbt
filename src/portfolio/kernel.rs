//! Steppable simulation kernel.
//!
//! Holds the per-bar simulation state that [`PortfolioEngine`] previously kept
//! as loop locals. Batch backtests drive this by looping [`EngineKernel::step`]
//! over historical bars; a future live engine drives the same code with bars
//! arriving in real time, which is the point of the extraction — one set of
//! execution semantics rather than a separate live reimplementation.
//!
//! [`PortfolioEngine`]: crate::portfolio::engine::PortfolioEngine

use crate::core::types::{
    BacktestConfig, Direction, ExitReason, InstrumentConfig, OhlcvBar, Price, StopConfig,
    TargetConfig, Trade,
};
use crate::execution::{FeeModel, FillModel, FillPrice, SlippageModel};
use crate::portfolio::position::{ExitDetails, PositionManager};
use crate::portfolio::risk::{RejectReason, RiskGate};

/// A single bar handed to the kernel.
///
/// Deliberately owns its values rather than borrowing an `OhlcvData` index:
/// a live feed produces one bar at a time with no backing array.
#[derive(Debug, Clone, Copy)]
pub struct KernelBar {
    pub timestamp: i64,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: f64,
}

impl KernelBar {
    /// Borrow as an [`OhlcvBar`] for the execution models.
    fn to_ohlcv_bar(self) -> OhlcvBar {
        OhlcvBar {
            timestamp: self.timestamp,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
        }
    }
}

/// Observable outcomes of a single [`EngineKernel::step`] call.
///
/// Batch callers can ignore these and read the accumulated trades; live callers
/// need them to drive order placement and alerting.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// A position was opened.
    Entered { idx: usize, price: Price, size: f64, direction: Direction },
    /// A position was closed, producing a completed trade.
    Exited { idx: usize, trade: Trade },
    /// An entry signal was refused by the risk gate.
    EntryRejected { idx: usize, reason: RejectReason },
}

/// Per-bar inputs that vary independently of the bar itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepInput {
    /// Entry signal for this bar (post signal-cleaning).
    pub entry: bool,
    /// Exit signal for this bar (post signal-cleaning).
    pub exit: bool,
    /// ATR value at this bar; `0.0` when no ATR-based stop/target is configured.
    pub atr: f64,
    /// Optional position-size multiplier from `CompiledSignals::position_sizes`.
    pub size_mult: Option<f64>,
}

/// Stateful simulation core.
///
/// One instance simulates one instrument. All mutable simulation state that the
/// original loop kept as locals lives here.
#[derive(Debug)]
pub struct EngineKernel {
    config: BacktestConfig,
    fee_model: FeeModel,
    slippage_model: SlippageModel,
    fill_price: FillPrice,
    /// Limit/stop fill semantics, including gap-through handling.
    fill_model: FillModel,

    position: PositionManager,
    cash: f64,
    /// Trading direction for new entries.
    direction: Direction,
    /// Timestamp of the open position's entry bar.
    ///
    /// The batch engine read this back out of `ohlcv.timestamps[entry_idx]`;
    /// a live kernel has no such array, so it is captured at entry instead.
    entry_timestamp: Option<i64>,
    /// Itemized entry costs, combined with exit costs when the trade closes.
    entry_breakdown: Option<crate::execution::indian_costs::FeeBreakdown>,

    /// Pre-trade constraints, checked before an entry opens.
    risk: RiskGate,

    effective_stop: StopConfig,
    effective_target: TargetConfig,
    /// Per-instrument capital cap and lot rounding, if any.
    alloted_capital: Option<f64>,
    lot_size: Option<f64>,
}

impl EngineKernel {
    /// Build a kernel from engine-level models and optional per-instrument config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: BacktestConfig,
        fee_model: FeeModel,
        slippage_model: SlippageModel,
        fill_price: FillPrice,
        symbol: String,
        direction: Direction,
        inst_config: Option<&InstrumentConfig>,
    ) -> Self {
        // Per-instrument stop/target override the global config.
        let effective_stop =
            inst_config.and_then(|ic| ic.stop.as_ref()).copied().unwrap_or(config.stop);
        let effective_target =
            inst_config.and_then(|ic| ic.target.as_ref()).copied().unwrap_or(config.target);

        let cash = config.initial_capital;

        Self {
            config,
            fee_model,
            slippage_model,
            fill_price,
            fill_model: FillModel { fill_price, ..FillModel::default() },
            position: PositionManager::new(symbol),
            cash,
            direction,
            entry_timestamp: None,
            entry_breakdown: None,
            risk: RiskGate::unconstrained(),
            effective_stop,
            effective_target,
            alloted_capital: inst_config.and_then(|ic| ic.alloted_capital),
            lot_size: inst_config.and_then(|ic| ic.lot_size),
        }
    }

    /// Attach pre-trade risk constraints.
    pub fn with_risk_gate(mut self, risk: RiskGate) -> Self {
        self.risk = risk;
        self
    }

    /// Current uninvested cash.
    #[inline]
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Entries refused by the risk gate.
    #[inline]
    pub fn rejected_entries(&self) -> usize {
        self.risk.rejected_entries()
    }

    /// Overwrite available cash.
    ///
    /// Used by the shared-capital portfolio runner, which owns one pool across
    /// several kernels and re-points each one at the pool before stepping it.
    #[inline]
    pub fn set_cash(&mut self, cash: f64) {
        self.cash = cash;
    }

    /// Market value of the open position at the given price, or 0.0 when flat.
    #[inline]
    pub fn position_value(&self, close: Price) -> f64 {
        if self.position.is_in_position() {
            close * self.position.position.size
        } else {
            0.0
        }
    }

    /// Feed current equity to the drawdown kill-switch.
    #[inline]
    pub fn observe_equity(&mut self, equity: f64, peak_equity: f64) {
        self.risk.on_equity(equity, peak_equity);
    }

    /// Whether a position is currently open.
    #[inline]
    pub fn is_in_position(&self) -> bool {
        self.position.is_in_position()
    }

    /// Mark-to-market equity at the given price.
    #[inline]
    pub fn equity(&self, close: Price) -> f64 {
        let position_value =
            if self.position.is_in_position() { close * self.position.position.size } else { 0.0 };
        self.cash + position_value
    }

    /// Advance the simulation by one bar.
    ///
    /// Order of operations is load-bearing and mirrors the original loop:
    /// update extremes, then exits (stop > target > signal), then entries.
    /// An exit and a re-entry may both occur on the same bar.
    pub fn step(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Vec<EngineEvent> {
        let mut events = Vec::new();

        // Track running extremes for trailing stops.
        self.position.update_price(bar.high, bar.low);

        if self.position.is_in_position() {
            if let Some(event) = self.try_exit(idx, bar, input.exit) {
                events.push(event);
            }

            // Trail only if the position survived this bar.
            if self.position.is_in_position() {
                if let StopConfig::Trailing { percent } = self.effective_stop {
                    self.position.update_trailing_stop(percent);
                }
            }
        }

        if !self.position.is_in_position() && input.entry {
            // Gate before opening, so a refused entry never reaches the equity
            // curve and the metrics describe the constrained run.
            let open_positions = usize::from(self.position.is_in_position());
            match self.risk.check_entry(open_positions) {
                Ok(()) => {
                    if let Some(event) = self.try_enter(idx, bar, input) {
                        events.push(event);
                    }
                }
                Err(reason) => {
                    self.risk.record_rejection();
                    events.push(EngineEvent::EntryRejected { idx, reason });
                }
            }
        }

        events
    }

    /// Exit path: stop-loss, then take-profit, then exit signal.
    fn try_exit(&mut self, idx: usize, bar: &KernelBar, exit_signal: bool) -> Option<EngineEvent> {
        let mut exit_reason: Option<ExitReason> = None;
        let mut exit_price = bar.close;

        let direction = self.position.position.direction;
        let ohlcv_bar = bar.to_ohlcv_bar();

        // Stop-loss, with gap-through adjustment against the bar open.
        //
        // Delegates to FillModel, which covers all four (direction, is_entry)
        // cases; the engine previously inlined a long/short-only copy of this.
        if self.position.is_stop_hit(bar.low, bar.high) {
            let stop_price = self.position.position.stop_price.unwrap();
            exit_reason = Some(ExitReason::StopLoss);
            exit_price = self
                .fill_model
                .get_stop_fill_price(stop_price, &ohlcv_bar, direction, false)
                .unwrap_or(stop_price);
        }

        // Take-profit, filled at the limit price.
        if exit_reason.is_none() && self.position.is_target_hit(bar.low, bar.high) {
            let target_price = self.position.position.target_price.unwrap();
            exit_reason = Some(ExitReason::TakeProfit);
            exit_price = self
                .fill_model
                .get_limit_fill_price(target_price, &ohlcv_bar, direction, false)
                .unwrap_or(target_price);
        }

        // Exit signal.
        if exit_reason.is_none() && exit_signal {
            exit_reason = Some(ExitReason::Signal);
            exit_price = self.fill_price_for(bar, self.position.position.direction, false);
        }

        let reason = exit_reason?;

        let exit_price = self.slippage_model.apply(
            exit_price,
            self.position.position.direction,
            false,
            Some(bar.volume),
        );

        // calculate_side, not calculate: STT lands on the sell leg and stamp
        // duty on the buy leg, so entry and exit are not symmetric.
        let exit_breakdown = self.fee_model.breakdown(
            exit_price,
            self.position.position.size,
            self.position.position.direction,
            false,
        );
        let fees = match exit_breakdown {
            Some(b) => b.total(),
            None => self.fee_model.calculate(
                exit_price,
                self.position.position.size,
                self.position.position.direction,
            ),
        };

        // Round-trip breakdown: entry components plus exit components, so the
        // itemized total equals the fees actually deducted from the equity curve.
        let combined = match (self.entry_breakdown, exit_breakdown) {
            (Some(entry), Some(exit)) => {
                let mut total = entry;
                total.add(&exit);
                Some(total)
            }
            (entry, exit) => entry.or(exit),
        };

        let entry_ts = self.entry_timestamp?;
        let trade = self.position.close_position(ExitDetails {
            idx,
            timestamp: bar.timestamp,
            price: exit_price,
            entry_timestamp: entry_ts,
            reason,
            fees,
            fee_breakdown: combined,
        })?;

        self.cash += exit_price * trade.size - fees;
        self.entry_timestamp = None;
        self.entry_breakdown = None;

        Some(EngineEvent::Exited { idx, trade })
    }

    /// Entry path: size against available capital, round to lot, open.
    fn try_enter(&mut self, idx: usize, bar: &KernelBar, input: StepInput) -> Option<EngineEvent> {
        let direction = self.direction;
        let entry_price = self.fill_price_for(bar, direction, true);
        let adjusted_price =
            self.slippage_model.apply(entry_price, direction, true, Some(bar.volume));

        // Per-instrument capital cap, never exceeding cash on hand.
        let available = self.alloted_capital.map(|cap| cap.min(self.cash)).unwrap_or(self.cash);

        // size = capital / (price * (1 + fee_rate)) so value + entry fee fits.
        let fee_rate = self.config.fees;
        let raw_size = match input.size_mult {
            Some(mult) => mult * available / (adjusted_price * (1.0 + fee_rate)),
            None => available / (adjusted_price * (1.0 + fee_rate)),
        };

        let size = match self.lot_size {
            Some(lot) if lot > 0.0 => (raw_size / lot).floor() * lot,
            _ => raw_size,
        };

        if size <= 0.0 {
            return None;
        }

        let entry_breakdown = self.fee_model.breakdown(adjusted_price, size, direction, true);
        let entry_fees = match entry_breakdown {
            Some(b) => b.total(),
            None => self.fee_model.calculate(adjusted_price, size, direction),
        };
        let (stop_price, target_price) = self.stop_and_target(adjusted_price, direction, input.atr);

        self.position.open_position(
            idx,
            bar.timestamp,
            adjusted_price,
            size,
            direction,
            stop_price,
            target_price,
            entry_fees,
        );
        self.entry_timestamp = Some(bar.timestamp);
        self.entry_breakdown = entry_breakdown;
        self.cash -= adjusted_price * size + entry_fees;

        Some(EngineEvent::Entered { idx, price: adjusted_price, size, direction })
    }

    /// Force-close any open position at end of data.
    ///
    /// Marked-to-market with zero exit fees: the position is not actually
    /// traded out, so charging exit costs would understate the result.
    pub fn finalize(&mut self, idx: usize, bar: &KernelBar) -> Option<Trade> {
        if !self.position.is_in_position() {
            return None;
        }

        let entry_ts = self.entry_timestamp?;
        let trade = self.position.close_position(ExitDetails {
            idx,
            timestamp: bar.timestamp,
            price: bar.close,
            entry_timestamp: entry_ts,
            reason: ExitReason::EndOfData,
            fees: 0.0,
            fee_breakdown: self.entry_breakdown,
        })?;

        self.cash += bar.close * trade.size;
        self.entry_timestamp = None;
        self.entry_breakdown = None;

        Some(trade)
    }

    /// Resolve fill price from the configured price model.
    ///
    /// Delegates to [`FillPrice::get_price_from_arrays`] rather than matching
    /// inline: the `Worst`/`Best` variants are direction- and entry-dependent,
    /// and duplicating that table invites drift.
    fn fill_price_for(&self, bar: &KernelBar, direction: Direction, is_entry: bool) -> Price {
        self.fill_price
            .get_price_from_arrays(bar.open, bar.high, bar.low, bar.close, direction, is_entry)
    }

    /// Compute stop and target prices for a new position.
    fn stop_and_target(
        &self,
        entry_price: Price,
        direction: Direction,
        atr_value: f64,
    ) -> (Option<Price>, Option<Price>) {
        let multiplier = direction.multiplier();

        // ATR of 0.0 means warmup has not completed; no stop/target is set
        // rather than one pinned at the entry price.
        let stop_price = match self.effective_stop {
            StopConfig::None => None,
            StopConfig::Fixed { percent } => Some(entry_price * (1.0 - multiplier * percent)),
            StopConfig::Atr { multiplier: atr_mult, .. } => {
                if atr_value > 0.0 {
                    Some(entry_price - multiplier * atr_mult * atr_value)
                } else {
                    None
                }
            }
            StopConfig::Trailing { percent } => Some(entry_price * (1.0 - multiplier * percent)),
        };

        let target_price = match self.effective_target {
            TargetConfig::None => None,
            TargetConfig::Fixed { percent } => Some(entry_price * (1.0 + multiplier * percent)),
            TargetConfig::Atr { multiplier: atr_mult, .. } => {
                if atr_value > 0.0 {
                    Some(entry_price + multiplier * atr_mult * atr_value)
                } else {
                    None
                }
            }
            TargetConfig::RiskReward { ratio } => stop_price.map(|sp| {
                let risk = (entry_price - sp).abs();
                entry_price + multiplier * risk * ratio
            }),
        };

        (stop_price, target_price)
    }
}
