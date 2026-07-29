//! Rebalance-policy simulation with the Indian delivery cost schedule.
//!
//! Answers the question the optimizer cannot: *what does following a target
//! series actually cost over time?* Given a price panel and a same-shaped
//! target-weight panel, simulates holding the book and re-trading it to
//! target under a policy (calendar cadence or drift band), charging per-leg
//! regulatory fees via [`crate::execution::indian_costs`] plus the flat DP
//! sell charge per distinct asset with a net sell per rebalance date --
//! the charge that dominates small delivery books.
//!
//! Inputs are strict: any NaN price or target on a date the simulation
//! touches is an error, not a skip. A research panel with holes must be
//! cleaned (or the asset excluded) by the caller, which is where the
//! refusal can name the asset.

use crate::core::types::Direction;
use crate::execution::indian_costs::{calculate_side, Segment};

use super::errors::{require_finite, PortfolioMathError};

/// When to re-trade toward targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RebalancePolicy {
    /// Every `every_n` dates (1 = daily).
    Calendar { every_n: usize },
    /// Whenever one-way turnover to target exceeds `band` (fraction).
    Band { band: f64 },
}

/// Simulation configuration.
#[derive(Debug, Clone)]
pub struct RebalanceConfig {
    /// Starting capital in rupees.
    pub initial_capital: f64,
    /// Rebalance policy.
    pub policy: RebalancePolicy,
    /// Trades below this rupee value are not executed.
    pub min_trade_value: f64,
    /// Cost segment for every leg (equity delivery for this product).
    pub segment: Segment,
    /// Flat DP charge per distinct asset with a net sell per date, rupees.
    pub dp_charge_per_isin: f64,
    /// Return periodicity of the date axis (e.g. 252 for daily).
    pub periods_per_year: f64,
}

/// Simulation output.
#[derive(Debug, Clone)]
pub struct RebalanceSimResult {
    /// Mark-to-market equity per date (after costs).
    pub equity_curve: Vec<f64>,
    /// One-way turnover fraction per date (0 on non-rebalance dates).
    pub turnover: Vec<f64>,
    /// Regulatory (non-brokerage, non-DP) costs per date, rupees.
    pub cost_regulatory: Vec<f64>,
    /// Brokerage per date, rupees.
    pub cost_brokerage: Vec<f64>,
    /// DP sell charges per date, rupees.
    pub cost_dp: Vec<f64>,
    /// Number of rebalance events executed.
    pub n_rebalances: u32,
    /// Number of individual trades executed.
    pub n_trades: u32,
    /// Total costs as an annualized fraction of average equity.
    pub total_cost_drag_annualized: f64,
}

/// Simulate following `target_weights` over `prices`.
///
/// Both panels are row-major `n_dates x n_assets`. The first date's targets
/// are bought from cash (entry costs charged).
pub fn simulate_rebalance_policy(
    prices: &[f64],
    n_dates: usize,
    n_assets: usize,
    target_weights: &[f64],
    cfg: &RebalanceConfig,
) -> Result<RebalanceSimResult, PortfolioMathError> {
    require_finite(prices, n_dates, n_assets)?;
    require_finite(target_weights, n_dates, n_assets)?;
    if n_dates == 0 || n_assets == 0 {
        return Err(PortfolioMathError::DegenerateInput(
            "empty panel".into(),
        ));
    }
    for (idx, p) in prices.iter().enumerate() {
        if *p <= 0.0 {
            return Err(PortfolioMathError::DegenerateInput(format!(
                "non-positive price {p} at row {}, col {}",
                idx / n_assets,
                idx % n_assets
            )));
        }
    }
    for d in 0..n_dates {
        let row = &target_weights[d * n_assets..(d + 1) * n_assets];
        let sum: f64 = row.iter().sum();
        if row.iter().any(|w| *w < -1e-12) || sum > 1.0 + 1e-9 {
            return Err(PortfolioMathError::DegenerateInput(format!(
                "target weights on date {d} must be long-only and sum <= 1 (sum {sum:.6})"
            )));
        }
    }
    if !(cfg.initial_capital.is_finite() && cfg.initial_capital > 0.0) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "initial_capital must be > 0, got {}",
            cfg.initial_capital
        )));
    }
    if cfg.dp_charge_per_isin < 0.0 || cfg.min_trade_value < 0.0 {
        return Err(PortfolioMathError::DegenerateInput(
            "dp_charge_per_isin and min_trade_value must be >= 0".into(),
        ));
    }
    if !(cfg.periods_per_year.is_finite() && cfg.periods_per_year > 0.0) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "periods_per_year must be > 0, got {}",
            cfg.periods_per_year
        )));
    }
    match cfg.policy {
        RebalancePolicy::Calendar { every_n } if every_n == 0 => {
            return Err(PortfolioMathError::DegenerateInput(
                "Calendar every_n must be >= 1".into(),
            ));
        }
        RebalancePolicy::Band { band } if !(band > 0.0 && band.is_finite()) => {
            return Err(PortfolioMathError::DegenerateInput(
                "Band band must be a positive finite fraction".into(),
            ));
        }
        _ => {}
    }

    let mut units = vec![0.0; n_assets]; // shares held
    let mut cash = cfg.initial_capital;
    let mut equity_curve = Vec::with_capacity(n_dates);
    let mut turnover = vec![0.0; n_dates];
    let mut cost_regulatory = vec![0.0; n_dates];
    let mut cost_brokerage = vec![0.0; n_dates];
    let mut cost_dp = vec![0.0; n_dates];
    let mut n_rebalances = 0u32;
    let mut n_trades = 0u32;
    let mut total_costs = 0.0;

    for d in 0..n_dates {
        let px = &prices[d * n_assets..(d + 1) * n_assets];
        let tw = &target_weights[d * n_assets..(d + 1) * n_assets];

        let holdings_value: f64 = (0..n_assets).map(|a| units[a] * px[a]).sum();
        let equity = holdings_value + cash;

        // Current weights and one-way drift to target.
        let drift: f64 = (0..n_assets)
            .map(|a| ((units[a] * px[a]) / equity - tw[a]).abs())
            .sum::<f64>()
            * 0.5;

        let should_rebalance = match cfg.policy {
            RebalancePolicy::Calendar { every_n } => d % every_n == 0,
            RebalancePolicy::Band { band } => d == 0 || drift > band,
        };

        if should_rebalance {
            let mut traded_value = 0.0;
            let mut sold_isins = 0u32;
            let mut day_reg = 0.0;
            let mut day_brk = 0.0;
            let mut executed_any = false;

            for a in 0..n_assets {
                let current_value = units[a] * px[a];
                let target_value = tw[a] * equity;
                let delta_value = target_value - current_value;
                if delta_value.abs() < cfg.min_trade_value || delta_value == 0.0 {
                    continue;
                }
                let is_buy = delta_value > 0.0;
                // Long-only book: a buy is a long entry, a sell a long exit.
                let fees = calculate_side(
                    cfg.segment,
                    delta_value.abs(),
                    Direction::Long,
                    is_buy,
                );
                let fee_total = fees.total();
                day_brk += fees.brokerage;
                day_reg += fee_total - fees.brokerage;

                units[a] += delta_value / px[a];
                cash -= delta_value; // buy consumes cash, sell frees it
                cash -= fee_total;
                traded_value += delta_value.abs();
                n_trades += 1;
                executed_any = true;
                if !is_buy {
                    sold_isins += 1;
                }
            }

            if executed_any {
                let day_dp = cfg.dp_charge_per_isin * sold_isins as f64;
                cash -= day_dp;
                cost_dp[d] = day_dp;
                cost_regulatory[d] = day_reg;
                cost_brokerage[d] = day_brk;
                turnover[d] = traded_value / equity * 0.5;
                total_costs += day_reg + day_brk + day_dp;
                n_rebalances += 1;
            }
        }

        let holdings_value: f64 = (0..n_assets).map(|a| units[a] * px[a]).sum();
        equity_curve.push(holdings_value + cash);
    }

    let avg_equity = equity_curve.iter().sum::<f64>() / n_dates as f64;
    let years = n_dates as f64 / cfg.periods_per_year;
    let total_cost_drag_annualized = if avg_equity > 0.0 && years > 0.0 {
        total_costs / avg_equity / years
    } else {
        return Err(PortfolioMathError::DegenerateInput(
            "average equity or horizon not positive; cost drag undefined".into(),
        ));
    };

    Ok(RebalanceSimResult {
        equity_curve,
        turnover,
        cost_regulatory,
        cost_brokerage,
        cost_dp,
        n_rebalances,
        n_trades,
        total_cost_drag_annualized,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::indian_costs::DP_SELL_CHARGE_PER_ISIN_PER_DAY;

    fn cfg(policy: RebalancePolicy) -> RebalanceConfig {
        RebalanceConfig {
            initial_capital: 1_000_000.0,
            policy,
            min_trade_value: 0.0,
            segment: Segment::EquityDelivery,
            dp_charge_per_isin: DP_SELL_CHARGE_PER_ISIN_PER_DAY,
            periods_per_year: 252.0,
        }
    }

    #[test]
    fn buy_only_first_date_has_no_dp_charge() {
        // Two dates, flat prices, constant targets: date 0 buys (no sells),
        // date 1 has no drift so nothing trades.
        let prices = vec![100.0, 50.0, 100.0, 50.0];
        let targets = vec![0.5, 0.5, 0.5, 0.5];
        let r = simulate_rebalance_policy(&prices, 2, 2, &targets, &cfg(RebalancePolicy::Band { band: 0.05 })).unwrap();
        assert_eq!(r.cost_dp[0], 0.0);
        assert_eq!(r.n_rebalances, 1);
        assert_eq!(r.n_trades, 2);
    }

    #[test]
    fn dp_charged_once_per_sold_isin() {
        // Date 0: buy 3 names. Date 1: targets flip to sell two of them.
        let prices = vec![100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let targets = vec![
            0.3, 0.3, 0.3, //
            0.0, 0.0, 0.9,
        ];
        let r = simulate_rebalance_policy(
            &prices,
            2,
            3,
            &targets,
            &cfg(RebalancePolicy::Calendar { every_n: 1 }),
        )
        .unwrap();
        // Date 1 sells assets 0 and 1 (2 ISINs) and buys asset 2.
        assert!((r.cost_dp[1] - 2.0 * DP_SELL_CHARGE_PER_ISIN_PER_DAY).abs() < 1e-9);
    }

    #[test]
    fn per_leg_fees_match_calculate_side() {
        // One asset, one buy of the full book on date 0.
        let prices = vec![100.0];
        let targets = vec![1.0];
        let r = simulate_rebalance_policy(
            &prices,
            1,
            1,
            &targets,
            &cfg(RebalancePolicy::Calendar { every_n: 1 }),
        )
        .unwrap();
        let fees = calculate_side(Segment::EquityDelivery, 1_000_000.0, Direction::Long, true);
        assert!((r.cost_brokerage[0] - fees.brokerage).abs() < 1e-9);
        assert!((r.cost_regulatory[0] - (fees.total() - fees.brokerage)).abs() < 1e-9);
        // Equity after date 0 = capital - fees (flat price, fully invested).
        assert!((r.equity_curve[0] - (1_000_000.0 - fees.total())).abs() < 1e-6);
    }

    #[test]
    fn dp_dominates_costs_on_a_small_book() {
        // The motivating fact: a Rs 50k book rotating 10 names pays more in
        // DP charges than in percentage fees.
        let n = 10;
        let mut prices = Vec::new();
        let mut targets = Vec::new();
        for d in 0..2 {
            for a in 0..n {
                prices.push(100.0);
                // Date 0: equal weight first 10; date 1: rotate out of all.
                let w = if d == 0 { 0.1 } else { 0.0 };
                let _ = a;
                targets.push(w);
            }
        }
        let mut c = cfg(RebalancePolicy::Calendar { every_n: 1 });
        c.initial_capital = 50_000.0;
        let r = simulate_rebalance_policy(&prices, 2, n, &targets, &c).unwrap();
        let dp_total: f64 = r.cost_dp.iter().sum();
        assert!((dp_total - 10.0 * DP_SELL_CHARGE_PER_ISIN_PER_DAY).abs() < 1e-9);
        // On the liquidation day itself, the flat DP charge exceeds every
        // percentage-rate cost combined -- the Phase-1 measured fact that
        // motivates the small-book refusal.
        assert!(
            r.cost_dp[1] > r.cost_regulatory[1],
            "DP {} should dominate percentage costs {} on the sell day",
            r.cost_dp[1],
            r.cost_regulatory[1]
        );
    }

    #[test]
    fn band_policy_skips_small_drift() {
        // Prices drift slightly; band is wide; only date 0 trades.
        let prices = vec![100.0, 100.0, 101.0, 99.5, 102.0, 99.0];
        let targets = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let r = simulate_rebalance_policy(&prices, 3, 2, &targets, &cfg(RebalancePolicy::Band { band: 0.10 })).unwrap();
        assert_eq!(r.n_rebalances, 1);
        assert_eq!(r.turnover[1], 0.0);
        assert_eq!(r.turnover[2], 0.0);
    }

    #[test]
    fn refuses_nan_and_bad_targets() {
        let prices = vec![100.0, f64::NAN];
        let targets = vec![0.5, 0.5];
        assert!(simulate_rebalance_policy(&prices, 1, 2, &targets, &cfg(RebalancePolicy::Calendar { every_n: 1 })).is_err());
        let prices = vec![100.0, 100.0];
        let targets = vec![0.7, 0.6]; // sums over 1
        assert!(simulate_rebalance_policy(&prices, 1, 2, &targets, &cfg(RebalancePolicy::Calendar { every_n: 1 })).is_err());
    }
}
