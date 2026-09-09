//! Event-driven portfolio simulation engine.

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, CompiledSignals, ExitReason, InstrumentConfig,
    OhlcvData, StopConfig, TargetConfig, Trade,
};
use crate::execution::{FeeModel, FillPrice, SlippageModel};
use crate::indicators::volatility::atr;
use crate::metrics::annualization;
use crate::metrics::streaming::StreamingMetrics;
use crate::portfolio::kernel::{KernelBar, StepInput};
use crate::portfolio::runner::SingleRunner;
use crate::signals::processor::SignalProcessor;

/// Portfolio simulation engine.
///
/// Single-pass O(n) algorithm for simulating portfolio performance.
#[derive(Debug)]
pub struct PortfolioEngine {
    /// Configuration.
    pub config: BacktestConfig,
    /// Fee model.
    pub fee_model: FeeModel,
    /// Slippage model.
    pub slippage_model: SlippageModel,
    /// Fill price model.
    pub fill_price: FillPrice,
    /// Signal processor.
    pub signal_processor: SignalProcessor,
}

impl Default for PortfolioEngine {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

impl PortfolioEngine {
    /// Create a new portfolio engine with the given configuration.
    pub fn new(config: BacktestConfig) -> Self {
        let fee_model = config.fee_model();
        let fill_price = FillPrice::for_timing(config.resolved_fill_timing());
        let slippage_model = Self::slippage_model_for(&config);

        Self {
            config,
            fee_model,
            slippage_model,
            fill_price,
            signal_processor: SignalProcessor::new(),
        }
    }

    /// Resolve the slippage model from config.
    ///
    /// Through 0.4.1 this was hardcoded to `None`, so `config.slippage` was
    /// silently ignored on every backtest. `apply_slippage = false` restores
    /// that behavior for reproducing pre-0.5.0 results.
    fn slippage_model_for(config: &BacktestConfig) -> SlippageModel {
        if config.apply_slippage && config.slippage > 0.0 {
            SlippageModel::percentage(config.slippage)
        } else {
            SlippageModel::None
        }
    }

    /// Set fee model.
    pub fn with_fee_model(mut self, fee_model: FeeModel) -> Self {
        self.fee_model = fee_model;
        self
    }

    /// Set slippage model.
    pub fn with_slippage_model(mut self, slippage_model: SlippageModel) -> Self {
        self.slippage_model = slippage_model;
        self
    }

    /// Run backtest on single instrument.
    ///
    /// # Arguments
    /// * `ohlcv` - OHLCV data
    /// * `signals` - Compiled trading signals
    ///
    /// # Returns
    /// Backtest result
    pub fn run_single(&self, ohlcv: &OhlcvData, signals: &CompiledSignals) -> BacktestResult {
        self.run_single_with_instrument_config(ohlcv, signals, None)
    }

    /// Run backtest on single instrument with optional per-instrument configuration.
    ///
    /// # Arguments
    /// * `ohlcv` - OHLCV data
    /// * `signals` - Compiled trading signals
    /// * `inst_config` - Optional per-instrument config (lot_size, capital cap, stop/target overrides)
    ///
    /// # Returns
    /// Backtest result
    pub fn run_single_with_instrument_config(
        &self,
        ohlcv: &OhlcvData,
        signals: &CompiledSignals,
        inst_config: Option<&InstrumentConfig>,
    ) -> BacktestResult {
        let n = ohlcv.len();
        assert_eq!(n, signals.len(), "OHLCV and signals must have same length");

        // Clean signals
        let (entries, exits) =
            self.signal_processor.clean_signals(&signals.entries, &signals.exits);

        // Initialize state
        let mut runner = SingleRunner::new(
            self.config.clone(),
            self.fee_model.clone(),
            self.slippage_model.clone(),
            self.fill_price,
            signals.symbol.clone(),
            signals.direction,
            inst_config,
        );

        // Determine effective stop/target configs (per-instrument overrides take precedence)
        let effective_stop =
            inst_config.and_then(|ic| ic.stop.as_ref()).unwrap_or(&self.config.stop);
        let effective_target =
            inst_config.and_then(|ic| ic.target.as_ref()).unwrap_or(&self.config.target);

        // Pre-calculate ATR for ATR-based stops.
        //
        // `None` when no stop or target is ATR-based, rather than a zero-filled
        // array: the run then reads 0.0 per bar from the `map_or` below instead
        // of from `n` f64s nothing ever wrote a real value into. On a 25M-bar
        // run that array was 200 MB, allocated and written on every run that
        // does not use an ATR stop -- which is most of them.
        let atr_values = if matches!(effective_stop, StopConfig::Atr { .. })
            || matches!(effective_target, TargetConfig::Atr { .. })
        {
            let period = match effective_stop {
                StopConfig::Atr { period, .. } => *period,
                _ => match effective_target {
                    TargetConfig::Atr { period, .. } => *period,
                    _ => 14,
                },
            };
            // A failed ATR still yields zeros rather than aborting the run --
            // the golden `atr_stop_rr_target` fixture encodes that behaviour.
            Some(
                atr(&ohlcv.high, &ohlcv.low, &ohlcv.close, period).unwrap_or_else(|_| vec![0.0; n]),
            )
        } else {
            None
        };

        // Main simulation loop — the per-bar body lives in EngineKernel::step
        // (via SingleRunner, which owns the curve accounting) so that a live
        // feed can drive identical execution semantics.
        for i in 0..n {
            let bar = KernelBar {
                timestamp: ohlcv.timestamps[i],
                open: ohlcv.open[i],
                high: ohlcv.high[i],
                low: ohlcv.low[i],
                close: ohlcv.close[i],
                volume: ohlcv.volume[i],
            };

            let input = StepInput {
                entry: entries[i],
                exit: exits[i],
                atr: atr_values.as_ref().map_or(0.0, |a| a.get(i).copied().unwrap_or(0.0)),
                size_mult: signals.position_sizes.as_ref().map(|sizes| sizes[i]),
                ..StepInput::default()
            };

            runner.step(i, &bar, input);
        }

        runner.finish()
    }

    /// Calculate backtest metrics.
    ///
    /// `timestamps` drives annualization and elapsed-time CAGR; pass an empty
    /// slice when unavailable and the legacy constants apply as a fallback.
    fn calculate_metrics(
        &self,
        equity_curve: &[f64],
        drawdown_curve: &[f64],
        returns: &[f64],
        trades: &[Trade],
        timestamps: &[i64],
        _streaming: &StreamingMetrics,
    ) -> BacktestMetrics {
        let start_value = self.config.initial_capital;
        let end_value = *equity_curve.last().unwrap_or(&start_value);

        let total_return_pct = (end_value - start_value) / start_value * 100.0;

        // Every drawdown figure below folds the SAME streamed curve, in one
        // pass, deliberately not `metrics::drawdown::*`. Those helpers rebuild
        // the curve with `calculate_drawdown_curve`, which seeds its peak from
        // `equity_curve[0]`; the streamed curve in `portfolio::runner` seeds
        // `peak_equity` from `config.initial_capital`. When the first equity
        // sample differs from initial capital the two curves disagree, and a
        // caller would see an ulcer — or an average drawdown — describing a
        // curve that no other reported drawdown figure describes. Do not
        // "simplify" these into the helpers.
        //
        // `> 0.0` is the underwater test throughout, matching
        // `calculate_max_drawdown_duration`, so "under water" means one thing
        // in this file.
        let mut max_drawdown_pct = 0.0f64;
        let mut sum_sq_dd = 0.0;
        let mut underwater_samples: usize = 0;
        let mut underwater_sum = 0.0;
        for &d in drawdown_curve {
            max_drawdown_pct = max_drawdown_pct.max(d);
            sum_sq_dd += d * d;
            if d > 0.0 {
                underwater_samples += 1;
                underwater_sum += d;
            }
        }

        let ulcer_index = if drawdown_curve.is_empty() {
            0.0
        } else {
            (sum_sq_dd / drawdown_curve.len() as f64).sqrt()
        };

        let time_under_water_pct = if drawdown_curve.is_empty() {
            0.0
        } else {
            underwater_samples as f64 / drawdown_curve.len() as f64 * 100.0
        };

        // Conditional on being underwater — see the field docs. A run that
        // never fell reports None, not 0.0.
        let avg_drawdown_pct = if underwater_samples > 0 {
            Some(underwater_sum / underwater_samples as f64)
        } else {
            None
        };

        // Calculate max drawdown duration
        let max_drawdown_duration = self.calculate_max_drawdown_duration(drawdown_curve);

        // Trade statistics
        let total_trades = trades.len();

        // Separate closed vs open trades (EndOfData means still open)
        let total_open_trades =
            trades.iter().filter(|t| matches!(t.exit_reason, ExitReason::EndOfData)).count();
        let total_closed_trades = total_trades.saturating_sub(total_open_trades);

        // Open trade PnL
        let open_trade_pnl: f64 = trades
            .iter()
            .filter(|t| matches!(t.exit_reason, ExitReason::EndOfData))
            .map(|t| t.pnl)
            .sum();

        // Only count closed trades for win/loss statistics
        let closed_trades: Vec<_> =
            trades.iter().filter(|t| !matches!(t.exit_reason, ExitReason::EndOfData)).collect();

        let winning_trades = closed_trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = closed_trades.iter().filter(|t| t.pnl < 0.0).count();

        let win_rate_pct = if total_closed_trades > 0 {
            winning_trades as f64 / total_closed_trades as f64 * 100.0
        } else {
            0.0
        };

        // Total fees paid
        let total_fees_paid: f64 = trades.iter().map(|t| t.fees).sum();

        // Best and worst trade
        let best_trade_pct =
            trades.iter().map(|t| t.return_pct).fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let best_trade_pct = if best_trade_pct.is_infinite() { 0.0 } else { best_trade_pct };

        let worst_trade_pct =
            trades.iter().map(|t| t.return_pct).fold(f64::INFINITY, |a, b| a.min(b));
        let worst_trade_pct = if worst_trade_pct.is_infinite() { 0.0 } else { worst_trade_pct };

        // Profit factor (based on closed trades)
        let gross_profit: f64 = closed_trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 =
            closed_trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Expectancy = average trade PnL
        let expectancy = if total_closed_trades > 0 {
            closed_trades.iter().map(|t| t.pnl).sum::<f64>() / total_closed_trades as f64
        } else {
            0.0
        };

        // SQN = (Expectancy / StdDev of trade PnL) * sqrt(total trades)
        let sqn = if total_closed_trades > 1 {
            let trade_pnls: Vec<f64> = closed_trades.iter().map(|t| t.pnl).collect();
            let mean = expectancy;
            let variance = trade_pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                / (total_closed_trades - 1) as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 {
                (mean / std_dev) * (total_closed_trades as f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Average returns
        let avg_trade_return_pct = if total_trades > 0 {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        // An average over an empty set is undefined, not zero. Reporting 0.0
        // for "the average losing trade" on a run where nothing lost money
        // states that the losers broke even, which is a claim about trades
        // that do not exist -- the same defect that had profit_factor showing
        // 0.00 beside a 100% win rate.
        let avg_win_pct = if winning_trades > 0 {
            Some(
                closed_trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.return_pct).sum::<f64>()
                    / winning_trades as f64,
            )
        } else {
            None
        };

        let avg_loss_pct = if losing_trades > 0 {
            Some(
                closed_trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.return_pct).sum::<f64>()
                    / losing_trades as f64,
            )
        } else {
            None
        };

        // Average winning/losing trade duration
        let avg_winning_duration = if winning_trades > 0 {
            Some(
                closed_trades
                    .iter()
                    .filter(|t| t.pnl > 0.0)
                    .map(|t| t.holding_period() as f64)
                    .sum::<f64>()
                    / winning_trades as f64,
            )
        } else {
            None
        };

        let avg_losing_duration = if losing_trades > 0 {
            Some(
                closed_trades
                    .iter()
                    .filter(|t| t.pnl < 0.0)
                    .map(|t| t.holding_period() as f64)
                    .sum::<f64>()
                    / losing_trades as f64,
            )
        } else {
            None
        };

        // Consecutive wins/losses
        let (max_consecutive_wins, max_consecutive_losses) = self.calculate_consecutive(trades);

        // Holding period
        let avg_holding_period = if total_trades > 0 {
            trades.iter().map(|t| t.holding_period() as f64).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        // ...and the same average as real elapsed time, which is the only form
        // a caller can render as a duration. A bar is a day on daily data and
        // a tick on a tick run, so "329" meant 329 days on one and 45 seconds
        // on the other.
        //
        // Read the trade's OWN timestamps rather than indexing `timestamps`
        // by `entry_idx`/`exit_idx`. Those indices count **events**, while
        // `timestamps` holds one entry per **equity sample**, and on a tick
        // session those are different domains: equity is sampled once per
        // print, but quotes advance the index too. Measured on one session of
        // NSE:HDFCBANK — 28,642 prints and 28,635 quotes — a trade closing on
        // the last event carried `exit_idx = 57276` against a 28,642-entry
        // array, so `timestamps.get()` returned `None`, every span was
        // dropped, and the all-trades guard below correctly discarded the
        // whole average. The result reached users as a NULL that the UI
        // rendered as "Avg hold 132 bars" on a run where no bar existed.
        //
        // `entry_time`/`exit_time` are populated from the same clock on every
        // production path (portfolio ledger, single-instrument position book,
        // options runner), so this is exact on bar and tick runs alike rather
        // than merely tolerant of the mismatch.
        let avg_holding_period_secs = if total_trades > 0 {
            let spans: Vec<f64> = trades
                .iter()
                .filter_map(|t| {
                    let span = (t.exit_time - t.entry_time) as f64;
                    if span < 0.0 {
                        None
                    } else {
                        Some(span / 1_000_000_000.0)
                    }
                })
                .collect();
            // Only report an average that covers every trade; a partial one
            // would understate the true figure without saying so.
            if spans.len() == total_trades {
                Some(spans.iter().sum::<f64>() / total_trades as f64)
            } else {
                None
            }
        } else {
            None
        };

        // Exposure (time in market).
        //
        // Capped at 100%: positions can overlap (several concurrent trades in
        // a netting book), and summing their holding periods against a single
        // equity curve then reports more time in the market than the backtest
        // actually ran -- 123.5% was observed. Time in market cannot exceed
        // the time available.
        let bars_in_position: usize = trades.iter().map(|t| t.holding_period()).sum();
        let exposure_pct = if !equity_curve.is_empty() {
            (bars_in_position as f64 / equity_curve.len() as f64 * 100.0).min(100.0)
        } else {
            0.0
        };

        // Risk-adjusted metrics (calculated from daily portfolio returns, not trade returns)
        let periods_per_year = if self.config.legacy_annualization {
            annualization::LEGACY_PERIODS_SINGLE
        } else {
            annualization::resolve_periods_per_year_with_session(
                self.config.periods_per_year,
                timestamps,
                self.config.session_spec(),
                annualization::LEGACY_PERIODS_SINGLE,
            )
        };
        let (sharpe_ratio, sortino_ratio, omega_ratio, return_skew, return_kurtosis) =
            risk_metrics_with_moments(returns, periods_per_year, self.config.risk_free_rate);

        // Calmar ratio: total return / max drawdown, both as percentages.
        //
        // This is deliberately NOT annualized. Annualizing compounds the run's
        // return up to a full year, so on a short window the result is an
        // artifact of the window length rather than a property of the strategy:
        // a 15.5% gain over 5.27 days reported a Calmar of 115,906 while the
        // plain ratio is 0.83. The shorter the run, the larger the number, with
        // no floor to stop it -- one day of the same strategy reported ~3.7e23.
        //
        // Every other runner in this crate (streaming, multi, basket, pairs,
        // options) already uses the plain ratio, so annualizing here also made
        // one strategy report a Calmar five orders of magnitude apart depending
        // only on which runner executed it. `metrics::drawdown::calmar_ratio` is
        // the single definition they now all share.
        let calmar_ratio =
            crate::metrics::drawdown::calmar_ratio(total_return_pct, max_drawdown_pct);

        // Payoff ratio: average win / average loss (absolute value)
        // Undefined without both halves: no loss to divide by, or no win to
        // divide. unwrap_or(0.0) reproduces the pre-Option branch exactly.
        let avg_win_for_payoff = avg_win_pct.unwrap_or(0.0);
        let avg_loss_for_payoff = avg_loss_pct.unwrap_or(0.0);
        let payoff_ratio = if avg_loss_for_payoff.abs() > 0.0 {
            avg_win_for_payoff / avg_loss_for_payoff.abs()
        } else if avg_win_for_payoff > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Recovery factor: net profit / max drawdown (absolute value)
        let net_profit = end_value - start_value;
        let recovery_factor = if max_drawdown_pct > 0.0 && start_value > 0.0 {
            let max_dd_absolute = max_drawdown_pct / 100.0 * start_value;
            if max_dd_absolute > 0.0 {
                net_profit / max_dd_absolute
            } else {
                0.0
            }
        } else if net_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let tail_ratio = tail_ratio_of(returns);
        let return_consistency_pct = return_consistency_of(returns);

        // Cost diagnostics. `Trade::pnl` is already net of that trade's costs,
        // so the winners' gross profit adds the fees back — dividing costs by
        // a figure they had already been subtracted from would understate the
        // drag. Verified against the invariant `mae_pnl <= pnl + fees <= mfe_pnl`.
        let cost_to_gross_profit_pct = {
            let gross_before_costs: f64 =
                closed_trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl + t.fees).sum();
            if gross_before_costs > 0.0 {
                Some(total_fees_paid / gross_before_costs * 100.0)
            } else {
                None
            }
        };
        let breakeven_cost_multiple = if total_fees_paid > 0.0 && net_profit > 0.0 {
            Some(net_profit / total_fees_paid)
        } else {
            None
        };

        let (mae_mfe_coverage_pct, avg_mae_pnl, mfe_capture_ratio) =
            excursion_aggregates(&closed_trades);

        BacktestMetrics {
            total_return_pct,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            omega_ratio,
            max_drawdown_pct,
            max_drawdown_duration,
            max_drawdown_duration_secs: self.max_drawdown_duration_secs(drawdown_curve, timestamps),
            ulcer_index,
            time_under_water_pct,
            win_rate_pct,
            profit_factor,
            expectancy,
            sqn,
            total_trades,
            total_closed_trades,
            total_open_trades,
            open_trade_pnl,
            winning_trades,
            losing_trades,
            start_value,
            end_value,
            total_fees_paid,
            best_trade_pct,
            worst_trade_pct,
            avg_trade_return_pct,
            avg_win_pct,
            avg_loss_pct,
            avg_winning_duration,
            avg_losing_duration,
            max_consecutive_wins,
            max_consecutive_losses,
            avg_holding_period,
            avg_holding_period_secs,
            exposure_pct,
            payoff_ratio,
            recovery_factor,
            total_turnover: crate::metrics::trade_stats::total_turnover(trades),
            return_skew,
            return_kurtosis,
            tail_ratio,
            cost_to_gross_profit_pct,
            breakeven_cost_multiple,
            return_consistency_pct,
            avg_drawdown_pct,
            mae_mfe_coverage_pct,
            avg_mae_pnl,
            mfe_capture_ratio,
        }
    }

    /// Calculate max drawdown duration from drawdown curve.
    fn calculate_max_drawdown_duration(&self, drawdown_curve: &[f64]) -> usize {
        let mut max_duration = 0;
        let mut current_duration = 0;

        for &dd in drawdown_curve {
            if dd > 0.0 {
                current_duration += 1;
                max_duration = max_duration.max(current_duration);
            } else {
                current_duration = 0;
            }
        }

        max_duration
    }

    /// The longest drawdown stretch measured in wall-clock seconds.
    ///
    /// Returns `None` when the run carried no usable timestamps, in which case
    /// only the bar count is meaningful. Bars are not a unit of time: one bar
    /// is one day on daily data and one tick on a tick run, so a caller that
    /// renders `max_drawdown_duration` as days is right only by accident.
    fn max_drawdown_duration_secs(
        &self,
        drawdown_curve: &[f64],
        timestamps: &[i64],
    ) -> Option<f64> {
        if timestamps.len() < 2 || drawdown_curve.is_empty() {
            return None;
        }

        // Walk the same stretches as the bar-count version, but keep the start
        // and end indices so the span can be read off the timestamps.
        let mut best: Option<(usize, usize)> = None;
        let mut run_start: Option<usize> = None;

        for (i, &dd) in drawdown_curve.iter().enumerate() {
            if dd > 0.0 {
                let start = *run_start.get_or_insert(i);
                let longer = best.is_none_or(|(bs, be)| (i - start) > (be - bs));
                if longer {
                    best = Some((start, i));
                }
            } else {
                run_start = None;
            }
        }

        let (start, end) = best?;
        // The curves and the timestamp series are index-aligned; a shorter
        // timestamp series means we cannot place these indices in time.
        let t_start = timestamps.get(start)?;
        let t_end = timestamps.get(end)?;
        let span_nanos = (*t_end - *t_start) as f64;
        if span_nanos < 0.0 {
            return None;
        }
        Some(span_nanos / 1_000_000_000.0)
    }

    /// Calculate max consecutive wins and losses.
    fn calculate_consecutive(&self, trades: &[Trade]) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;

        for trade in trades {
            if trade.pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else if trade.pnl < 0.0 {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            }
        }

        (max_wins, max_losses)
    }
}

/// Shape of the per-bar return distribution: `(skew, excess kurtosis)`, from
/// central moments already accumulated by [`risk_metrics`].
///
/// Takes the moment sums rather than the series because the caller has just
/// walked it: the mean, the NaN filter and the squared-deviation sum are all
/// shared, so recomputing them here would be two more passes over a
/// million-element array for two scalars.
///
/// Both are sample bias-corrected, matching `scipy.stats` with `bias=False`;
/// kurtosis is **excess** (Gaussian 0.0). `None` where the sample is too small
/// for the correction to be defined — three for skew, four for kurtosis — or
/// where the series has no dispersion to describe.
pub(crate) fn return_shape_from_moments(
    n: usize,
    sum_sq_dev: f64,
    sum_cube_dev: f64,
    sum_quad_dev: f64,
) -> (Option<f64>, Option<f64>) {
    if n < 3 {
        return (None, None);
    }
    let count = n as f64;

    let m2 = sum_sq_dev / count;
    let m3 = sum_cube_dev / count;
    let m4 = sum_quad_dev / count;

    if m2 <= 0.0 {
        return (None, None);
    }

    // g1 -> G1: the sample-skewness correction.
    let g1 = m3 / m2.powf(1.5);
    let skew = g1 * (count * (count - 1.0)).sqrt() / (count - 2.0);

    let kurtosis = if n < 4 {
        None
    } else {
        // g2 -> G2, the bias-corrected excess kurtosis.
        let g2 = m4 / (m2 * m2) - 3.0;
        Some(((count + 1.0) * g2 + 6.0) * (count - 1.0) / ((count - 2.0) * (count - 3.0)))
    };

    (Some(skew), kurtosis)
}

/// `|p95| / |p5|` of the per-bar returns, at linear-interpolation percentiles.
///
/// Sorting a copy is the one superlinear step in the metrics pass, and it is
/// over bars, so it is cheap next to the run that produced them. `None` below
/// 20 samples, or when the left tail sits exactly at zero.
pub(crate) fn tail_ratio_of(returns: &[f64]) -> Option<f64> {
    let mut valid: Vec<f64> = returns.iter().copied().filter(|r| !r.is_nan()).collect();
    if valid.len() < 20 {
        return None;
    }

    // Four order statistics, not a sorted array. `select_nth_unstable` places
    // the k-th element and partitions around it in O(n); a full sort is
    // O(n log n) and cost 47 ms of a 1.875M-bar run to read four values.
    //
    // Selecting the upper index first leaves everything <= it on the left, so
    // the lower index can then be selected within that prefix alone -- the two
    // selections do not disturb each other's results.
    //
    // Same numbers as sorting: linear-interpolation percentiles matching
    // NumPy's default, pinned against it in test_diagnostic_metrics.
    let last = valid.len() - 1;
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).expect("NaN filtered above");

    // Resolve one percentile as (value at floor(pos), value at ceil(pos)).
    fn pair<F>(data: &mut [f64], pos: f64, cmp: &F) -> (f64, f64)
    where
        F: Fn(&f64, &f64) -> std::cmp::Ordering,
    {
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            let (_, &mut at, _) = data.select_nth_unstable_by(lo, cmp);
            return (at, at);
        }
        // Place `hi` first; everything <= data[hi] is now in data[..hi].
        let (left_part, &mut upper, _) = data.select_nth_unstable_by(hi, cmp);
        let (_, &mut lower, _) = left_part.select_nth_unstable_by(lo, cmp);
        (lower, upper)
    }

    let p5 = 0.05 * last as f64;
    let p95 = 0.95 * last as f64;

    let (lo5, hi5) = pair(&mut valid, p5, &cmp);
    let left = (lo5 + (hi5 - lo5) * (p5 - p5.floor())).abs();
    if left <= 0.0 {
        return None;
    }

    let (lo95, hi95) = pair(&mut valid, p95, &cmp);
    let right = (lo95 + (hi95 - lo95) * (p95 - p95.floor())).abs();
    Some(right / left)
}

/// Share of *moving* bars that moved up, as a percentage.
///
/// Flat bars are excluded rather than counted as losses: a strategy that is
/// out of the market most of the time would otherwise be scored mostly on
/// bars it had no position in. `None` when nothing moved.
pub(crate) fn return_consistency_of(returns: &[f64]) -> Option<f64> {
    let mut moving: usize = 0;
    let mut up: usize = 0;
    for &r in returns {
        if r.is_nan() || r == 0.0 {
            continue;
        }
        moving += 1;
        if r > 0.0 {
            up += 1;
        }
    }
    if moving == 0 {
        return None;
    }
    Some(up as f64 / moving as f64 * 100.0)
}

/// Excursion aggregates: `(coverage %, mean MAE, MFE capture ratio)`.
///
/// MAE/MFE are `None` on trades synthesized rather than closed from a tracked
/// position (spread, basket and pairs legs). Those are excluded and the
/// coverage figure says how many were, rather than the averages silently
/// describing a subset — `None` there means *not measured*, never zero.
///
/// Capture is a ratio of sums over winners only, with a gross numerator. See
/// the field docs on [`BacktestMetrics::mfe_capture_ratio`] for why each of
/// those three choices is load-bearing.
fn excursion_aggregates(closed: &[&Trade]) -> (Option<f64>, Option<f64>, Option<f64>) {
    if closed.is_empty() {
        return (None, None, None);
    }

    let mut measured: usize = 0;
    let mut mae_sum = 0.0;
    let mut winner_gross = 0.0;
    let mut winner_mfe = 0.0;
    let mut winners: usize = 0;

    for t in closed {
        let (Some(mae), Some(mfe)) = (t.mae_pnl, t.mfe_pnl) else {
            continue;
        };
        measured += 1;
        mae_sum += mae;
        if t.pnl > 0.0 {
            winners += 1;
            winner_gross += t.pnl + t.fees;
            winner_mfe += mfe;
        }
    }

    let coverage = Some(measured as f64 / closed.len() as f64 * 100.0);
    if measured == 0 {
        return (coverage, None, None);
    }

    let capture =
        if winners > 0 && winner_mfe > 0.0 { Some(winner_gross / winner_mfe) } else { None };

    (coverage, Some(mae_sum / measured as f64), capture)
}

/// Risk-adjusted metrics from a series of **per-bar** portfolio returns.
///
/// Returns `(sharpe, sortino, omega)`. Use [`risk_metrics_with_moments`] when
/// the caller also wants the distribution shape: it shares this function's
/// single walk of the series rather than adding two more.
///
/// All runners must feed this the per-bar return series, not per-trade returns.
/// Through 0.4.1 the basket/pairs/options/multi paths annualized *trade*
/// returns at a hardcoded 252, which assumes one trade per trading day and
/// inflates Sharpe by roughly `sqrt(n_bars / n_trades)` — on the order of 7x
/// for a 9-trade run over 500 bars. Those runners built a correct per-bar
/// series and then discarded it for this purpose.
pub fn risk_metrics(
    returns: &[f64],
    periods_per_year: f64,
    risk_free_rate: f64,
) -> (f64, f64, f64) {
    let (sharpe, sortino, omega, _, _) =
        risk_metrics_with_moments(returns, periods_per_year, risk_free_rate);
    (sharpe, sortino, omega)
}

/// [`risk_metrics`] plus the distribution shape, from one walk of the series.
///
/// Returns `(sharpe, sortino, omega, skew, excess kurtosis)`. The ratios are
/// bit-identical to `risk_metrics`; the two extra values come from central
/// moments accumulated in the same loop that computes the variance.
pub fn risk_metrics_with_moments(
    returns: &[f64],
    periods_per_year: f64,
    risk_free_rate: f64,
) -> (f64, f64, f64, Option<f64>, Option<f64>) {
    if returns.len() < 2 {
        return (0.0, 0.0, 1.0, None, None);
    }

    // NaN is skipped inline rather than collected into a filtered Vec. The
    // arithmetic below is unchanged -- same values, same summation order, so
    // the results are bit-identical -- but two n-length allocations go away
    // (this one and the downside filter), which on a 1.875M-bar run was 30 MB
    // of transient heap to produce three scalars.
    //
    // Two passes over the slice, deliberately: the variance is taken about a
    // known mean. Folding it into a single sum-of-squares pass would change
    // the floating-point result, which the golden corpus pins.
    let mut count: usize = 0;
    let mut sum = 0.0;
    for &r in returns {
        if !r.is_nan() {
            count += 1;
            sum += r;
        }
    }

    if count < 2 {
        return (0.0, 0.0, 1.0, None, None);
    }

    let n_valid = count as f64;
    let mean = sum / n_valid;

    // Second pass carries every remaining accumulator: the deviation sum for
    // the variance, the downside sum of squares for Sortino, and the signed
    // sums for Omega.
    let mut sq_dev = 0.0;
    let mut downside_sq = 0.0;
    let mut sum_positive = 0.0;
    let mut sum_negative = 0.0;
    let mut has_downside = false;
    // The third and fourth central moments ride along: `return_shape` needs
    // the same mean over the same NaN-filtered values, and walking a
    // million-element array twice more to get them cost ~28 ms of a 1.875M-bar
    // run. Two multiplies per element here instead.
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &r in returns {
        if r.is_nan() {
            continue;
        }
        let d = r - mean;
        let d2 = d * d;
        sq_dev += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
        if r > 0.0 {
            sum_positive += r;
        } else if r < 0.0 {
            downside_sq += r.powi(2);
            has_downside = true;
            sum_negative += r.abs();
        }
    }

    let variance = sq_dev / (n_valid - 1.0);
    let std_dev = variance.sqrt();

    // Excess return over the per-period risk-free rate.
    let rf_per_period =
        if periods_per_year > 0.0 { risk_free_rate / periods_per_year } else { 0.0 };
    let excess_mean = mean - rf_per_period;

    // Sharpe Ratio = (excess * periods_per_year) / (std_dev * sqrt(periods_per_year))
    // Simplified: Sharpe = excess / std_dev * sqrt(periods_per_year)
    let sharpe_ratio =
        if std_dev > 0.0 { (excess_mean / std_dev) * periods_per_year.sqrt() } else { 0.0 };

    // Sortino Ratio - uses downside deviation (only negative returns)
    let downside_variance = if has_downside {
        downside_sq / n_valid // Divide by total count, not downside count
    } else {
        0.0
    };
    let downside_std = downside_variance.sqrt();

    let sortino_ratio = if downside_std > 0.0 {
        (excess_mean / downside_std) * periods_per_year.sqrt()
    } else if excess_mean > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Omega Ratio = sum of returns above threshold / |sum of returns below threshold|
    // With threshold = 0. Both sums are accumulated in the pass above.
    let omega_ratio = if sum_negative > 0.0 {
        sum_positive / sum_negative
    } else if sum_positive > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };

    let (skew, kurtosis) = return_shape_from_moments(count, sq_dev, m3, m4);

    (sharpe_ratio, sortino_ratio, omega_ratio, skew, kurtosis)
}

/// Compute `BacktestMetrics` from pre-built curves and trade list.
///
/// Exposed as a standalone function so non-OHLCV strategies (e.g. tick backtest)
/// can produce identical metrics without duplicating the calculation logic.
pub fn compute_backtest_metrics(
    equity_curve: &[f64],
    drawdown_curve: &[f64],
    returns: &[f64],
    trades: &[Trade],
    timestamps: &[i64],
    initial_capital: f64,
) -> BacktestMetrics {
    compute_backtest_metrics_with_config(
        equity_curve,
        drawdown_curve,
        returns,
        trades,
        timestamps,
        &BacktestConfig { initial_capital, ..Default::default() },
    )
}

/// Compute `BacktestMetrics` honoring a full config.
///
/// Preferred over [`compute_backtest_metrics`] when the caller has a real
/// config: annualization, risk-free rate and session length all come from it,
/// and defaulting them would silently misreport intraday runs.
pub fn compute_backtest_metrics_with_config(
    equity_curve: &[f64],
    drawdown_curve: &[f64],
    returns: &[f64],
    trades: &[Trade],
    timestamps: &[i64],
    config: &BacktestConfig,
) -> BacktestMetrics {
    // Delegate to a throwaway engine instance — avoids duplicating the logic.
    let engine = PortfolioEngine::new(config.clone());
    engine.calculate_metrics(
        equity_curve,
        drawdown_curve,
        returns,
        trades,
        timestamps,
        &StreamingMetrics::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Direction;

    fn sample_ohlcv() -> OhlcvData<'static> {
        OhlcvData {
            timestamps: (0..20).map(|i| i as i64).collect(),
            open: vec![
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 101.0,
                102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            ]
            .into(),
            high: vec![
                101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 102.0,
                103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
            ]
            .into(),
            low: vec![
                99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0,
                101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
            ]
            .into(),
            close: vec![
                100.5, 101.5, 102.5, 103.5, 104.5, 105.0, 104.0, 103.0, 102.0, 101.0, 100.5, 101.5,
                102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5,
            ]
            .into(),
            volume: vec![1000.0; 20].into(),
        }
    }

    fn sample_signals() -> CompiledSignals {
        CompiledSignals {
            symbol: "TEST".to_string(),
            entries: vec![
                false, true, false, false, false, false, false, false, false, false, false, true,
                false, false, false, false, false, false, false, false,
            ],
            exits: vec![
                false, false, false, false, false, true, false, false, false, false, false, false,
                false, false, false, true, false, false, false, false,
            ],
            position_sizes: None,
            direction: Direction::Long,
            weight: 1.0,
        }
    }

    #[test]
    fn test_basic_backtest() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.0,
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
            ..Default::default()
        };

        let engine = PortfolioEngine::new(config);
        let ohlcv = sample_ohlcv();
        let signals = sample_signals();

        let result = engine.run_single(&ohlcv, &signals);

        // Should have 2 trades
        assert_eq!(result.trades.len(), 2);

        // First trade: entry at 101.5, exit at 105.0
        let trade1 = &result.trades[0];
        assert!((trade1.entry_price - 101.5).abs() < 1e-10);
        assert!((trade1.exit_price - 105.0).abs() < 1e-10);
        assert!(trade1.pnl > 0.0); // Profitable

        // Equity curve should have correct length
        assert_eq!(result.equity_curve.len(), 20);
    }

    /// A bar is not a day. On tick data it is a fraction of a second, and the
    /// bar counts alone reported "avg hold 329" for trades lasting seconds --
    /// which a caller rendered as 329 days.
    #[test]
    fn durations_are_reported_in_real_time_not_bar_counts() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.0,
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
            ..Default::default()
        };

        // Same 20 bars, but spaced one second apart instead of the default
        // 1-nanosecond ticks -- the shape of a tick run.
        const ONE_SEC: i64 = 1_000_000_000;
        let mut ohlcv = sample_ohlcv();
        ohlcv.timestamps = (0..20).map(|i| i as i64 * ONE_SEC).collect();

        let engine = PortfolioEngine::new(config);
        let result = engine.run_single(&ohlcv, &sample_signals());
        let m = &result.metrics;

        // The bar count is unchanged and still means bars.
        assert!(m.avg_holding_period > 0.0);

        // The trades span 4 bars each, one second apart, so the honest
        // duration is seconds -- not the bar count, and not days.
        let secs = m.avg_holding_period_secs.expect("timestamps were supplied");
        assert!(
            (secs - m.avg_holding_period).abs() < 1e-9,
            "one-second bars: {secs}s should equal {} bars",
            m.avg_holding_period
        );

        // Re-space the same run 60x wider. The bar count cannot change; the
        // reported duration must, which is the whole point of the field.
        let mut minute_ohlcv = sample_ohlcv();
        minute_ohlcv.timestamps = (0..20).map(|i| i as i64 * ONE_SEC * 60).collect();
        let minute = engine.run_single(&minute_ohlcv, &sample_signals());

        assert!(
            (minute.metrics.avg_holding_period - m.avg_holding_period).abs() < 1e-9,
            "bar count must not depend on spacing"
        );
        let minute_secs = minute.metrics.avg_holding_period_secs.expect("timestamps were supplied");
        assert!(
            (minute_secs - secs * 60.0).abs() < 1e-6,
            "60x wider bars must report 60x the elapsed time: {minute_secs} vs {secs}"
        );
    }

    /// Time in the market cannot exceed the time the backtest ran. Summing the
    /// holding periods of *concurrent* positions against one equity curve
    /// reported 123.5% on a real run.
    #[test]
    fn ulcer_describes_the_same_curve_as_max_drawdown() {
        // The invariant that must survive refactoring. `calculate_metrics` folds
        // the STREAMED drawdown curve, whose peak is seeded from
        // `config.initial_capital` (runner.rs). The convenience helper
        // `metrics::drawdown::ulcer_index(equity)` rebuilds a curve seeded from
        // `equity[0]` instead. Here the run opens BELOW initial capital, so the
        // two disagree -- and the reported ulcer must match the curve every
        // other reported drawdown figure describes.
        let engine =
            PortfolioEngine::new(BacktestConfig { initial_capital: 100.0, ..Default::default() });

        // Equity opens at 90 (already 10% under the 100 the run was funded
        // with) and recovers. Streamed curve: [10, 5, 0]. Helper's curve,
        // seeded from equity[0]=90: [0, 0, 0].
        let equity_curve = vec![90.0, 95.0, 100.0];
        let drawdown_curve = vec![10.0, 5.0, 0.0];
        let timestamps: Vec<i64> = vec![0, 1, 2];

        let metrics = engine.calculate_metrics(
            &equity_curve,
            &drawdown_curve,
            &[0.0, 0.0555, 0.0526],
            &[],
            &timestamps,
            &StreamingMetrics::new(),
        );

        let expected = ((100.0f64 + 25.0 + 0.0) / 3.0).sqrt();
        assert!(
            (metrics.ulcer_index - expected).abs() < 1e-12,
            "ulcer must fold the streamed curve, got {}",
            metrics.ulcer_index
        );

        // The helper answers a different question on this run. If this ever
        // stops differing, the test has lost its teeth -- rebuild the case.
        assert!(
            (crate::metrics::drawdown::ulcer_index(&equity_curve) - metrics.ulcer_index).abs()
                > 1e-6,
            "the two curves must genuinely disagree for this test to mean anything"
        );

        // Two of three samples are under water.
        assert!((metrics.time_under_water_pct - 200.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn time_under_water_counts_samples_not_drawdown_spans() {
        let engine =
            PortfolioEngine::new(BacktestConfig { initial_capital: 100.0, ..Default::default() });

        // A high-water plateau followed by one dip. `drawdown_periods` would
        // report a span of 4 here (see its own test); counting samples gives
        // the true answer, 1 of 6.
        let equity_curve = vec![100.0, 110.0, 110.0, 110.0, 105.0, 120.0];
        let drawdown_curve = vec![0.0, 0.0, 0.0, 0.0, 4.545454545454546, 0.0];
        let timestamps: Vec<i64> = (0..6).collect();

        let metrics = engine.calculate_metrics(
            &equity_curve,
            &drawdown_curve,
            &[0.0; 6],
            &[],
            &timestamps,
            &StreamingMetrics::new(),
        );

        assert!((metrics.time_under_water_pct - 100.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn time_under_water_spans_the_full_range_from_never_to_always() {
        let engine =
            PortfolioEngine::new(BacktestConfig { initial_capital: 100.0, ..Default::default() });
        let ts: Vec<i64> = (0..4).collect();

        let rising = engine.calculate_metrics(
            &[100.0, 101.0, 102.0, 103.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0; 4],
            &[],
            &ts,
            &StreamingMetrics::new(),
        );
        assert_eq!(rising.time_under_water_pct, 0.0);
        assert_eq!(rising.ulcer_index, 0.0);

        let falling = engine.calculate_metrics(
            &[99.0, 98.0, 97.0, 96.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0; 4],
            &[],
            &ts,
            &StreamingMetrics::new(),
        );
        assert_eq!(falling.time_under_water_pct, 100.0);
    }

    #[test]
    fn exposure_never_exceeds_one_hundred_percent() {
        fn trade_over(id: u64, entry_idx: usize, exit_idx: usize) -> Trade {
            Trade {
                id,
                symbol: "TEST".to_string(),
                entry_idx,
                exit_idx,
                entry_price: 100.0,
                exit_price: 101.0,
                size: 1.0,
                direction: Direction::Long,
                pnl: 1.0,
                return_pct: 1.0,
                entry_time: entry_idx as i64,
                exit_time: exit_idx as i64,
                fees: 0.0,
                entry_fees: 0.0,
                exit_fees: 0.0,
                fee_breakdown: None,
                exit_reason: ExitReason::Signal,
                mae_price: None,
                mfe_price: None,
                mae_pnl: None,
                mfe_pnl: None,
            }
        }

        let engine = PortfolioEngine::new(BacktestConfig {
            initial_capital: 100_000.0,
            ..Default::default()
        });

        // Three positions held simultaneously across the whole 10-bar run.
        // Their holding periods sum to 27 bars against a 10-bar curve -- 270%
        // before the clamp.
        let equity_curve: Vec<f64> = (0..10).map(|i| 100_000.0 + i as f64).collect();
        let drawdown_curve = vec![0.0; 10];
        let returns = vec![0.0; 10];
        let timestamps: Vec<i64> = (0..10).map(|i| i as i64 * 1_000_000_000).collect();
        let trades = vec![trade_over(1, 0, 9), trade_over(2, 0, 9), trade_over(3, 0, 9)];

        let metrics = engine.calculate_metrics(
            &equity_curve,
            &drawdown_curve,
            &returns,
            &trades,
            &timestamps,
            &StreamingMetrics::with_initial_capital(100_000.0),
        );

        assert!(
            metrics.exposure_pct <= 100.0,
            "exposure {} exceeds the time available",
            metrics.exposure_pct
        );
        assert!(metrics.exposure_pct > 0.0, "the book was in the market");
    }

    /// The reported drawdown stretch is the same stretch the bar count found,
    /// measured on the clock. A 6-day tick run reported "93,510" bars, which a
    /// caller printed as 93,510 days -- roughly 256 years.
    #[test]
    fn drawdown_duration_is_reported_in_real_time() {
        let engine = PortfolioEngine::new(BacktestConfig {
            initial_capital: 100_000.0,
            ..Default::default()
        });

        // Underwater for indices 2..=6 -- five bars, four intervals.
        let drawdown_curve = vec![0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0];
        let equity_curve = vec![100_000.0; 10];
        let returns = vec![0.0; 10];
        let timestamps: Vec<i64> = (0..10).map(|i| i as i64 * 1_000_000_000).collect();

        let metrics = engine.calculate_metrics(
            &equity_curve,
            &drawdown_curve,
            &returns,
            &[],
            &timestamps,
            &StreamingMetrics::with_initial_capital(100_000.0),
        );

        assert_eq!(metrics.max_drawdown_duration, 5, "five bars underwater");
        let secs = metrics.max_drawdown_duration_secs.expect("timestamps supplied");
        assert!(
            (secs - 4.0).abs() < 1e-9,
            "one-second bars: index 2 to 6 is 4 seconds, got {secs}"
        );

        // Without timestamps the honest answer is "cannot say", not zero.
        let no_ts = engine.calculate_metrics(
            &equity_curve,
            &drawdown_curve,
            &returns,
            &[],
            &[],
            &StreamingMetrics::with_initial_capital(100_000.0),
        );
        assert_eq!(no_ts.max_drawdown_duration, 5);
        assert_eq!(no_ts.max_drawdown_duration_secs, None);
    }

    /// A trade index may sit beyond the equity timeline, and the duration
    /// must still be right.
    ///
    /// `entry_idx`/`exit_idx` count **events**; `timestamps` holds one entry
    /// per **equity sample**. On a tick session those diverge: equity is
    /// sampled once per print, but quotes advance the index too. Measured on
    /// one session of NSE:HDFCBANK -- 28,642 prints against 28,635 quotes --
    /// a trade closing on the final event carried `exit_idx = 57276` against
    /// a 28,642-entry array. Indexing it returned `None`, every span was
    /// dropped, and the all-trades guard discarded the whole average. Users
    /// saw "Avg hold 132 bars" on a run in which no bar existed.
    ///
    /// So the span comes from the trade's own timestamps. This fixture makes
    /// the old code fail: the indices are deliberately out of range for the
    /// timeline, while the timestamps carry an honest two-hour hold.
    #[test]
    fn holding_seconds_survive_indices_past_the_equity_timeline() {
        const ONE_SEC: i64 = 1_000_000_000;
        let engine = PortfolioEngine::new(BacktestConfig::default());

        let trade = Trade {
            id: 1,
            symbol: "TEST".to_string(),
            // Both indices point past a 10-entry timeline, as a tick run's
            // event indices do.
            entry_idx: 40,
            exit_idx: 7_240,
            entry_price: 100.0,
            exit_price: 101.0,
            size: 1.0,
            direction: Direction::Long,
            pnl: 1.0,
            return_pct: 1.0,
            entry_time: 1_000 * ONE_SEC,
            exit_time: 8_200 * ONE_SEC, // exactly two hours later
            fees: 0.0,
            entry_fees: 0.0,
            exit_fees: 0.0,
            fee_breakdown: None,
            exit_reason: ExitReason::Signal,
            mae_price: None,
            mfe_price: None,
            mae_pnl: None,
            mfe_pnl: None,
        };

        let timestamps: Vec<i64> = (0..10).map(|i| i as i64 * ONE_SEC).collect();
        let equity = vec![100_000.0; 10];
        let returns = vec![0.0; 10];
        let drawdown = vec![0.0; 10];

        let m = engine.calculate_metrics(
            &equity,
            &drawdown,
            &returns,
            std::slice::from_ref(&trade),
            &timestamps,
            &StreamingMetrics::default(),
        );

        assert_eq!(
            m.avg_holding_period_secs,
            Some(7_200.0),
            "the trade's own timestamps say two hours; indexing the equity \
             timeline by an event index would have discarded it entirely"
        );
    }

    /// Without timestamps there is no honest duration to report, and a caller
    /// must be able to tell that from a real zero.
    #[test]
    fn durations_are_none_when_the_run_carried_no_timestamps() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.0,
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
            ..Default::default()
        };

        let mut ohlcv = sample_ohlcv();
        ohlcv.timestamps = vec![0i64; 20].into();

        let engine = PortfolioEngine::new(config);
        let result = engine.run_single(&ohlcv, &sample_signals());

        // Every bar shares one timestamp, so no span is measurable.
        assert_eq!(result.metrics.avg_holding_period_secs, Some(0.0));
        assert!(result.metrics.avg_holding_period > 0.0, "bars still counted");
    }

    #[test]
    fn test_with_fees() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.001, // 0.1%
            slippage: 0.0,
            stop: StopConfig::None,
            target: TargetConfig::None,
            upon_bar_close: true,
            ..Default::default()
        };

        let engine = PortfolioEngine::new(config);
        let ohlcv = sample_ohlcv();
        let signals = sample_signals();

        let result = engine.run_single(&ohlcv, &signals);

        // Trades should have fees deducted
        for trade in &result.trades {
            assert!(trade.fees > 0.0);
        }
    }

    #[test]
    fn test_with_stop_loss() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            fees: 0.0,
            slippage: 0.0,
            stop: StopConfig::Fixed { percent: 0.02 }, // 2% stop
            target: TargetConfig::None,
            upon_bar_close: true,
            ..Default::default()
        };

        let engine = PortfolioEngine::new(config);

        // Create data where stop would be hit
        let mut ohlcv = sample_ohlcv();
        // Add a big drop after entry
        ohlcv.low.to_mut()[3] = 95.0; // Big drop
        ohlcv.close.to_mut()[3] = 96.0;

        let signals = sample_signals();
        let result = engine.run_single(&ohlcv, &signals);

        // First trade should exit on stop loss
        assert_eq!(result.trades[0].exit_reason, ExitReason::StopLoss);
    }
}
