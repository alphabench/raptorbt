//! Tick-level backtest implementation.
//!
//! Accepts raw tick arrays (ltp, bid, ask, per-tick buy/sell qty deltas) plus
//! parallel entry/exit signal arrays, then simulates each trade to
//! stop-loss / take-profit / max-hold-time exit at full tick resolution.
//!
//! This is the right path for intraday options momentum strategies where the
//! exact fill tick matters. Do not resample to bars before calling this —
//! bar resampling discards intra-bar path information and makes scalping
//! strategies unbacktestable.

use crate::core::types::{
    BacktestConfig, BacktestMetrics, BacktestResult, Direction, ExitReason, Price, TickData,
    Timestamp, Trade,
};
use crate::execution::fees::FeeModel;
use crate::execution::indian_costs::FeeBreakdown;
use crate::portfolio::engine::compute_backtest_metrics;

/// Configuration specific to tick backtests.
#[derive(Debug, Clone)]
pub struct TickBacktestConfig {
    /// Shared execution config (capital, fees, slippage).
    pub base: BacktestConfig,
    /// Stop-loss as percentage of entry price (e.g. 5.0 = 5%).
    pub stop_loss_pct: f64,
    /// Take-profit as percentage of entry price (e.g. 10.0 = 10%).
    pub take_profit_pct: f64,
    /// Maximum hold time in seconds. 0 = no time limit.
    pub max_hold_seconds: u64,
    /// Minimum ticks between entries (cooldown). Prevents overlapping positions.
    pub entry_cooldown_ticks: usize,
    /// Stop the run after this many trades. Defaults to [`usize::MAX`], i.e.
    /// no cap.
    ///
    /// This is a hard early exit, not a filter: the tick loop `break`s and the
    /// result is reported as if the input ended there. Through 0.6.4 this
    /// defaulted to 50, so a million-tick backtest silently described the
    /// first 0.8% of the tape -- one measured case reported a 0.124% max
    /// drawdown where the true figure over the full input was 14.13%. Leave it
    /// unset unless you specifically want a truncated run.
    pub max_trades: usize,
    /// Contracts per lot. `1` means one bare unit, which is the default and
    /// reproduces the per-unit behaviour of every release through 0.7.3.
    ///
    /// Costs and P&L both scale by `|quantity| * lot_size`, so a 75-lot NIFTY
    /// option charges 75x the contracts of a single unit and earns 75x the
    /// move. Through 0.7.3 this path hard-coded one unit, which understated
    /// both by the lot size.
    pub lot_size: u32,
    /// Lots traded. `0` trades nothing and costs nothing.
    ///
    /// Only positive quantities are supported: this path is long-only by
    /// construction -- it enters at the ask, exits at the bid, and places its
    /// stop below entry and target above. A negative quantity is refused
    /// rather than silently inverting logic that was never built for it.
    pub quantity: i64,
}

impl Default for TickBacktestConfig {
    fn default() -> Self {
        Self {
            base: BacktestConfig::default(),
            stop_loss_pct: 5.0,
            take_profit_pct: 10.0,
            max_hold_seconds: 1800,
            entry_cooldown_ticks: 10,
            max_trades: usize::MAX,
            lot_size: 1,
            quantity: 1,
        }
    }
}

/// Tick-level backtest runner.
pub struct TickBacktest {
    config: TickBacktestConfig,
    fee_model: FeeModel,
}

impl TickBacktest {
    /// The fee model comes from [`BacktestConfig::fee_model`], so setting
    /// `fee_segment` charges the itemized regulatory schedule -- per-order
    /// brokerage included -- and leaving it unset keeps the flat `fees` rate.
    pub fn new(config: TickBacktestConfig) -> Self {
        let fee_model = config.base.fee_model();
        Self { config, fee_model }
    }

    /// Run the tick backtest.
    ///
    /// `ticks`   — raw tick data (ltp, bid, ask, per-tick qty deltas)
    /// `entries` — parallel bool array: true at ticks where a new long entry is allowed
    /// `exits`   — parallel bool array: true at ticks where an open position must close
    /// `symbol`  — instrument label used in trade records
    ///
    /// Costs and P&L both scale by `|quantity| * lot_size`. A zero quantity
    /// trades nothing and returns an empty result; a negative quantity is
    /// refused, because this path is long-only (see
    /// [`TickBacktestConfig::quantity`]).
    pub fn run(
        &self,
        ticks: &TickData,
        entries: &[bool],
        exits: &[bool],
        symbol: &str,
    ) -> BacktestResult {
        let n = ticks.len();
        assert_eq!(n, entries.len(), "ticks and entries must have same length");
        assert_eq!(n, exits.len(), "ticks and exits must have same length");
        assert!(
            self.config.quantity >= 0,
            "tick backtests are long-only: quantity must be >= 0, got {}",
            self.config.quantity
        );

        // Contracts, not lots: a two-lot position trades twice the contracts
        // of a one-lot position and both owes and earns proportionally more.
        let size = (self.config.quantity.unsigned_abs() as f64) * self.config.lot_size as f64;

        // Nothing traded means no order, so no per-order brokerage either.
        if size == 0.0 {
            return Self::build_result(Vec::new(), self.config.base.initial_capital, symbol);
        }

        let slippage_frac = self.config.base.slippage; // e.g. 0.0005 = 0.05%
        let stop_frac = self.config.stop_loss_pct / 100.0;
        let target_frac = self.config.take_profit_pct / 100.0;
        let max_hold_ns: i64 = self.config.max_hold_seconds as i64 * 1_000_000_000;

        let mut trades: Vec<Trade> = Vec::new();
        let mut trade_id: u64 = 0;

        // Position state
        let mut in_position = false;
        let mut entry_idx: usize = 0;
        let mut entry_price: Price = 0.0;
        let mut entry_time: Timestamp = 0;
        let mut stop_level: Price = 0.0;
        let mut target_level: Price = 0.0;
        let mut entry_fees: f64 = 0.0;
        let mut entry_breakdown: Option<FeeBreakdown> = None;
        let mut cooldown_until: usize = 0;

        for i in 0..n {
            let ltp = ticks.ltp[i];
            let bid = if ticks.bid[i] > 0.0 { ticks.bid[i] } else { ltp };
            let ask = if ticks.ask[i] > 0.0 { ticks.ask[i] } else { ltp };
            let ts = ticks.timestamps[i];

            if in_position {
                // Check time exit first (hard deadline)
                let time_exit = max_hold_ns > 0 && (ts - entry_time) >= max_hold_ns;

                // Check explicit exit signal
                let signal_exit = exits[i];

                // Check stop and target against ltp (tick-exact, no OHLC lookahead)
                let stop_hit = ltp <= stop_level;
                let target_hit = ltp >= target_level;

                let (exit_price, reason) = if stop_hit {
                    // Fill at stop level (not ltp — avoid worse-than-stop fills)
                    let fill = stop_level * (1.0 - slippage_frac);
                    (fill, ExitReason::StopLoss)
                } else if target_hit {
                    let fill = target_level * (1.0 - slippage_frac);
                    (fill, ExitReason::TakeProfit)
                } else if time_exit || signal_exit {
                    let fill = bid * (1.0 - slippage_frac);
                    let reason = if time_exit { ExitReason::TimeExit } else { ExitReason::Signal };
                    (fill, reason)
                } else if i == n - 1 {
                    // End of data — force close at bid
                    let fill = bid * (1.0 - slippage_frac);
                    (fill, ExitReason::EndOfData)
                } else {
                    continue;
                };

                // A long position is bought to open and sold to close, so
                // side-specific charges (transaction tax on the sell, stamp
                // duty on the buy) land on the side that actually owes them.
                let exit_fees =
                    self.fee_model.calculate_side(exit_price, size, Direction::Long, false);
                let exit_breakdown =
                    self.fee_model.breakdown(exit_price, size, Direction::Long, false);

                // Entry components plus exit components, so the itemized total
                // equals the fees actually deducted from the equity curve.
                let fee_breakdown = match (entry_breakdown, exit_breakdown) {
                    (Some(entry), Some(exit)) => {
                        let mut total = entry;
                        total.add(&exit);
                        Some(total)
                    }
                    (entry, exit) => entry.or(exit),
                };

                let gross_pnl = (exit_price - entry_price) * size;
                let net_pnl = gross_pnl - entry_fees - exit_fees;
                let return_pct = net_pnl / (entry_price * size) * 100.0;

                trades.push(Trade {
                    id: trade_id,
                    symbol: symbol.to_string(),
                    entry_idx,
                    exit_idx: i,
                    entry_price,
                    exit_price,
                    size,
                    direction: Direction::Long,
                    pnl: net_pnl,
                    return_pct,
                    entry_time,
                    exit_time: ts,
                    fees: entry_fees + exit_fees,
                    entry_fees,
                    exit_fees,
                    fee_breakdown,
                    exit_reason: reason,
                });

                trade_id += 1;
                in_position = false;
                cooldown_until = i + self.config.entry_cooldown_ticks;

                if trades.len() >= self.config.max_trades {
                    break;
                }
            } else {
                // Not in position — check for entry
                if i < cooldown_until {
                    continue;
                }
                if !entries[i] {
                    continue;
                }
                if ask <= 0.0 {
                    continue;
                }

                entry_price = ask * (1.0 + slippage_frac);
                entry_fees =
                    self.fee_model.calculate_side(entry_price, size, Direction::Long, true);
                entry_breakdown =
                    self.fee_model.breakdown(entry_price, size, Direction::Long, true);
                entry_idx = i;
                entry_time = ts;
                stop_level = entry_price * (1.0 - stop_frac);
                target_level = entry_price * (1.0 + target_frac);
                in_position = true;
            }
        }

        Self::build_result(trades, self.config.base.initial_capital, symbol)
    }

    fn build_result(trades: Vec<Trade>, initial_capital: f64, _symbol: &str) -> BacktestResult {
        if trades.is_empty() {
            let metrics = BacktestMetrics {
                start_value: initial_capital,
                end_value: initial_capital,
                ..Default::default()
            };
            return BacktestResult::new(metrics, vec![initial_capital], vec![0.0], vec![], vec![]);
        }

        // Build per-trade equity and return curves (one point per trade close).
        let mut equity = initial_capital;
        let mut equity_curve = vec![initial_capital];
        let mut returns = Vec::with_capacity(trades.len());

        for t in &trades {
            let prev = *equity_curve.last().unwrap();
            equity += t.pnl;
            equity_curve.push(equity);
            let ret = if prev > 0.0 { (equity - prev) / prev } else { 0.0 };
            returns.push(ret);
        }

        // Drawdown curve over equity points (percentage, positive = drawdown).
        let mut peak = initial_capital;
        let drawdown_curve: Vec<f64> = equity_curve
            .iter()
            .map(|&e| {
                if e > peak {
                    peak = e;
                }
                if peak > 0.0 {
                    (peak - e) / peak * 100.0
                } else {
                    0.0
                }
            })
            .collect();

        // The equity curve here advances per *trade*, not per tick, so tick
        // timestamps do not index it. Passing an empty slice falls back to the
        // legacy annualization constant rather than inferring a wrong one;
        // per-trade annualization is left for a later release.
        let metrics = compute_backtest_metrics(
            &equity_curve,
            &drawdown_curve,
            &returns,
            &trades,
            &[],
            initial_capital,
        );

        BacktestResult::new(metrics, equity_curve, drawdown_curve, trades, returns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::BacktestConfig;

    fn make_ticks(n: usize, base_price: f64, trend: f64) -> TickData {
        let ltp: Vec<f64> = (0..n).map(|i| base_price + i as f64 * trend).collect();
        let bid: Vec<f64> = ltp.iter().map(|p| p - 0.5).collect();
        let ask: Vec<f64> = ltp.iter().map(|p| p + 0.5).collect();
        TickData {
            timestamps: (0..n as i64).map(|i| i * 1_000_000_000).collect(), // 1s apart
            ltp,
            bid,
            ask,
            buy_qty_delta: vec![100.0; n],
            sell_qty_delta: vec![80.0; n],
            oi: vec![0.0; n],
            ltq: vec![0.0; n],
            bid_qty: vec![0.0; n],
            ask_qty: vec![0.0; n],
        }
    }

    #[test]
    fn test_target_hit() {
        // 100 ticks trending up — entry at tick 0, target should be hit
        let ticks = make_ticks(100, 100.0, 0.5); // price goes 100 → 149.5
        let mut entries = vec![false; 100];
        entries[0] = true;
        let exits = vec![false; 100];

        let config = TickBacktestConfig {
            base: BacktestConfig {
                initial_capital: 10_000.0,
                fees: 0.0,
                slippage: 0.0,
                ..Default::default()
            },
            stop_loss_pct: 5.0,
            take_profit_pct: 10.0,
            max_hold_seconds: 0, // no time limit
            entry_cooldown_ticks: 5,
            max_trades: 10,
            ..Default::default()
        };

        let bt = TickBacktest::new(config);
        let result = bt.run(&ticks, &entries, &exits, "TEST");

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].exit_reason, ExitReason::TakeProfit);
        assert!(result.trades[0].pnl > 0.0);
    }

    #[test]
    fn test_stop_hit() {
        // 100 ticks trending down — entry at tick 0, stop should be hit
        let ticks = make_ticks(100, 100.0, -0.5); // price goes 100 → 50.5
        let mut entries = vec![false; 100];
        entries[0] = true;
        let exits = vec![false; 100];

        let config = TickBacktestConfig {
            base: BacktestConfig {
                initial_capital: 10_000.0,
                fees: 0.0,
                slippage: 0.0,
                ..Default::default()
            },
            stop_loss_pct: 5.0,
            take_profit_pct: 20.0,
            max_hold_seconds: 0,
            entry_cooldown_ticks: 5,
            max_trades: 10,
            ..Default::default()
        };

        let bt = TickBacktest::new(config);
        let result = bt.run(&ticks, &entries, &exits, "TEST");

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].exit_reason, ExitReason::StopLoss);
        assert!(result.trades[0].pnl < 0.0);
    }

    #[test]
    fn default_config_does_not_cap_trades() {
        // Through 0.6.4 `max_trades` defaulted to 50 and the tick loop `break`s
        // when it is reached, returning a normal-looking BacktestResult that
        // silently described only the leading slice of the tape. Measured on a
        // 1,000,000-tick input: 50 trades covering 0.81% of the ticks, total
        // return -0.12% where the true figure was -14.13%, and a max drawdown
        // of 0.124% against a true 14.13% -- a 114x understatement of the
        // single number a risk check reads.
        //
        // The cap remains available for callers who explicitly want a
        // truncated run; it is simply no longer the default.
        assert_eq!(TickBacktestConfig::default().max_trades, usize::MAX);
    }

    #[test]
    fn an_explicit_cap_still_truncates() {
        // The knob has to keep working, or the default change is a removal.
        let ticks = make_ticks(4_000, 100.0, 0.02);
        let mut entries = vec![false; 4_000];
        let mut exits = vec![false; 4_000];
        for i in (0..4_000).step_by(40) {
            entries[i] = true;
            if i + 20 < 4_000 {
                exits[i + 20] = true;
            }
        }

        let capped = TickBacktest::new(TickBacktestConfig {
            max_trades: 5,
            entry_cooldown_ticks: 1,
            ..Default::default()
        })
        .run(&ticks, &entries, &exits, "T");
        assert_eq!(capped.trades.len(), 5, "explicit cap must still bound the run");

        let uncapped =
            TickBacktest::new(TickBacktestConfig { entry_cooldown_ticks: 1, ..Default::default() })
                .run(&ticks, &entries, &exits, "T");
        assert!(
            uncapped.trades.len() > capped.trades.len(),
            "default must not truncate: got {} trades uncapped vs {} capped",
            uncapped.trades.len(),
            capped.trades.len()
        );
    }

    #[test]
    fn test_time_exit() {
        // Flat price — neither stop nor target hit, time exit should fire
        let ticks = make_ticks(200, 100.0, 0.0);
        let mut entries = vec![false; 200];
        entries[0] = true;
        let exits = vec![false; 200];

        let config = TickBacktestConfig {
            base: BacktestConfig {
                initial_capital: 10_000.0,
                fees: 0.0,
                slippage: 0.0,
                ..Default::default()
            },
            stop_loss_pct: 50.0, // very wide, won't hit
            take_profit_pct: 50.0,
            max_hold_seconds: 10, // 10 ticks at 1s each
            entry_cooldown_ticks: 5,
            max_trades: 10,
            ..Default::default()
        };

        let bt = TickBacktest::new(config);
        let result = bt.run(&ticks, &entries, &exits, "TEST");

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].exit_reason, ExitReason::TimeExit);
    }

    #[test]
    fn test_multiple_trades_with_cooldown() {
        let ticks = make_ticks(200, 100.0, 0.2);
        // Entry every 20 ticks
        let entries: Vec<bool> = (0..200).map(|i| i % 20 == 0).collect();
        let exits = vec![false; 200];

        let config = TickBacktestConfig {
            base: BacktestConfig {
                initial_capital: 10_000.0,
                fees: 0.0,
                slippage: 0.0,
                ..Default::default()
            },
            stop_loss_pct: 5.0,
            take_profit_pct: 10.0,
            max_hold_seconds: 0,
            entry_cooldown_ticks: 5,
            max_trades: 20,
            ..Default::default()
        };

        let bt = TickBacktest::new(config);
        let result = bt.run(&ticks, &entries, &exits, "TEST");

        assert!(result.trades.len() > 1);
        assert!(result.metrics.total_trades > 1);
    }

    #[test]
    fn test_empty_ticks_returns_empty_result() {
        let ticks = TickData {
            timestamps: vec![],
            ltp: vec![],
            bid: vec![],
            ask: vec![],
            buy_qty_delta: vec![],
            sell_qty_delta: vec![],
            oi: vec![],
            ltq: vec![],
            bid_qty: vec![],
            ask_qty: vec![],
        };
        let config = TickBacktestConfig::default();
        let bt = TickBacktest::new(config);
        let result = bt.run(&ticks, &[], &[], "TEST");
        assert_eq!(result.trades.len(), 0);
        assert_eq!(result.metrics.total_trades, 0);
    }

    // ------------------------------------------------------------------
    // Costs and position size (0.7.4)
    //
    // Through 0.7.3 this path charged `price * fees` and earned
    // `(exit - entry) * 1.0`, so every figure it reported described a single
    // unit however much was traded. Every test above runs `fees: 0.0` and
    // asserts only the *sign* of P&L, which is invariant to any positive
    // scale factor -- so the whole defect was invisible to the suite. These
    // pin it.
    // ------------------------------------------------------------------

    /// One round trip on a flat tape, so fee arithmetic is the only thing
    /// moving. Enters at tick 1, exits on signal at tick 4.
    fn costed_run(
        price: f64,
        fees: f64,
        segment: Option<&str>,
        lot_size: u32,
        quantity: i64,
    ) -> BacktestResult {
        let n = 6;
        // Flat bid == ask == ltp: no spread, no slippage, no drift.
        let ticks = TickData {
            timestamps: (0..n as i64).map(|i| i * 1_000_000_000).collect(),
            ltp: vec![price; n],
            bid: vec![price; n],
            ask: vec![price; n],
            buy_qty_delta: vec![0.0; n],
            sell_qty_delta: vec![0.0; n],
            oi: vec![0.0; n],
            ltq: vec![0.0; n],
            bid_qty: vec![0.0; n],
            ask_qty: vec![0.0; n],
        };
        let mut entries = vec![false; n];
        entries[1] = true;
        let mut exits = vec![false; n];
        exits[4] = true;

        let config = TickBacktestConfig {
            base: BacktestConfig {
                initial_capital: 500_000.0,
                fees,
                slippage: 0.0,
                fee_segment: segment.map(|s| s.to_string()),
                ..Default::default()
            },
            // Wide enough that neither fires on a flat tape.
            stop_loss_pct: 50.0,
            take_profit_pct: 50.0,
            max_hold_seconds: 0,
            entry_cooldown_ticks: 0,
            lot_size,
            quantity,
            ..Default::default()
        };
        TickBacktest::new(config).run(&ticks, &entries, &exits, "TEST")
    }

    #[test]
    fn a_round_trip_is_charged_on_both_sides() {
        let result = costed_run(100.0, 0.001, None, 75, 1);

        let trade = &result.trades[0];
        assert_eq!(trade.entry_fees, 7.50, "entry is one side of a 7500 notional at 0.1%");
        assert_eq!(trade.exit_fees, 7.50, "exit is the other side");
        assert_eq!(trade.fees, 15.00, "a round trip is two sides, never one");
        assert_eq!(
            trade.fees,
            trade.entry_fees + trade.exit_fees,
            "Trade documents this invariant"
        );
    }

    #[test]
    fn costs_scale_with_position_size() {
        // The headline defect: through 0.7.3 both of these charged the same.
        let one_lot = costed_run(100.0, 0.001, None, 75, 1);
        let two_lots = costed_run(100.0, 0.001, None, 75, 2);

        assert!(
            (two_lots.trades[0].fees - one_lot.trades[0].fees * 2.0).abs() < 1e-9,
            "two lots trade twice the contracts and owe twice the cost: {} vs {}",
            two_lots.trades[0].fees,
            one_lot.trades[0].fees
        );
    }

    #[test]
    fn pnl_scales_with_position_size() {
        // The second half of the same defect: P&L was per-unit too, so the
        // equity curve added one unit's profit to a full-size capital base.
        let ticks = make_ticks(60, 100.0, 0.5);
        let mut entries = vec![false; 60];
        entries[0] = true;
        let exits = vec![false; 60];

        let run = |lot_size: u32| {
            let config = TickBacktestConfig {
                base: BacktestConfig {
                    initial_capital: 500_000.0,
                    fees: 0.0,
                    slippage: 0.0,
                    ..Default::default()
                },
                lot_size,
                quantity: 1,
                ..Default::default()
            };
            TickBacktest::new(config).run(&ticks, &entries, &exits, "TEST")
        };

        let one = run(1);
        let seventy_five = run(75);

        assert!(
            (seventy_five.trades[0].pnl - one.trades[0].pnl * 75.0).abs() < 1e-9,
            "a 75-lot position earns 75x the move: {} vs {}",
            seventy_five.trades[0].pnl,
            one.trades[0].pnl
        );
        // Percentage return is scale-free -- it is the same trade either way.
        assert!(
            (seventy_five.trades[0].return_pct - one.trades[0].return_pct).abs() < 1e-9,
            "return_pct is a ratio and must not move with size"
        );
    }

    #[test]
    fn the_equity_curve_reflects_the_position_traded() {
        // Deliberately an *absolute* assertion, not a self-consistent one.
        // Checking only that equity moves by `trade.pnl` passes even when both
        // are per-unit -- the figures agree with each other and are simply not
        // the ones a real position would produce. That is the shape of defect
        // that survives review, so this pins the money instead.
        let result = costed_run(100.0, 0.001, None, 75, 1);

        let start = result.equity_curve[0];
        let end = *result.equity_curve.last().unwrap();

        assert!(
            (end - (start + result.trades[0].pnl)).abs() < 1e-9,
            "equity moves by the trade's own P&L"
        );
        // Flat tape, so the only movement is the cost of trading a 7500
        // notional: 7.50 in, 7.50 out. At one unit it would be 0.20.
        assert!(
            (start - end - 15.00).abs() < 1e-9,
            "a flat round trip loses exactly its costs, sized to the position: {}",
            start - end
        );
    }

    #[test]
    fn an_itemized_segment_charges_per_order_brokerage() {
        // Brokerage is a flat Rs 20 per order, which a proportional rate
        // cannot express at any rate -- this is why fee_segment matters more
        // than the rate being right.
        let result = costed_run(100.0, 0.001, Some("NFO-OPT"), 75, 1);

        let trade = &result.trades[0];
        let breakdown = trade.fee_breakdown.expect("an itemized segment reports its components");

        assert_eq!(breakdown.brokerage, 40.0, "2 orders (entry, exit) x Rs 20");
        assert!(
            (breakdown.total() - trade.fees).abs() < 1e-9,
            "the itemized total is the money actually charged"
        );
        assert_eq!(trade.fees, trade.entry_fees + trade.exit_fees);
        assert!(
            trade.fees > 40.0,
            "real costs dwarf the 15.00 flat rate on this notional: {}",
            trade.fees
        );
    }

    #[test]
    fn the_itemized_schedule_puts_each_charge_on_the_side_that_owes_it() {
        // Regression guard: this fails if someone routes through
        // `FeeModel::calculate`, which hard-codes is_entry = true and would
        // charge the buy schedule on both sides.
        let result = costed_run(100.0, 0.001, Some("NFO-OPT"), 75, 1);
        let trade = &result.trades[0];

        assert!(
            trade.entry_fees != trade.exit_fees,
            "stamp duty falls on the buy and transaction tax on the sell, so \
             the two sides cannot be equal: {} vs {}",
            trade.entry_fees,
            trade.exit_fees
        );
        // A long is bought to open and sold to close; STT is the larger charge.
        assert!(trade.exit_fees > trade.entry_fees, "the sell side carries STT");
    }

    #[test]
    fn a_zero_quantity_position_costs_nothing() {
        // Nothing traded places no order, so it owes no per-order brokerage.
        let result = costed_run(100.0, 0.001, Some("NFO-OPT"), 75, 0);

        assert_eq!(result.trades.len(), 0, "a zero quantity trades nothing");
        assert_eq!(result.metrics.total_trades, 0);
    }

    #[test]
    #[should_panic(expected = "long-only")]
    fn a_negative_quantity_is_refused() {
        // This path enters at the ask with its stop below entry. Running that
        // logic against a short would report a trade that could not happen.
        costed_run(100.0, 0.001, None, 75, -1);
    }

    #[test]
    fn the_default_position_reproduces_the_pre_0_7_4_numbers() {
        // Back-compat pin: callers that pass no size get exactly what they got
        // before, so upgrading changes nothing silently.
        let result = costed_run(100.0, 0.001, None, 1, 1);

        let trade = &result.trades[0];
        assert_eq!(trade.size, 1.0, "one bare unit, as before");
        assert_eq!(trade.entry_fees, 0.1, "price 100 * 0.001, the old arithmetic");
        assert_eq!(trade.exit_fees, 0.1);
    }
}
