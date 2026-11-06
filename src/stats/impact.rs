//! What it costs to trade a size: the square-root impact law from the instrument's own
//! volatility and average daily volume, plus a bid-ask spread estimated from daily bars
//! (Corwin-Schultz from highs and lows, or Roll from closes). An estimate from daily bars, not
//! a fill. One-way cost in basis points = half the spread + coefficient * daily sigma *
//! sqrt(quantity / ADV).

use crate::math::fsum::fsum;

use super::error::StatsError;
use super::hac::{finite_series, mean};

pub const MIN_DAYS: usize = 20;
/// Participation past a quarter of a day's volume is outside the law's range.
pub const BEYOND_MODEL: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadModel {
    CorwinSchultz,
    Roll,
}

impl SpreadModel {
    pub fn as_str(self) -> &'static str {
        match self {
            SpreadModel::CorwinSchultz => "corwin_schultz",
            SpreadModel::Roll => "roll",
        }
    }

    pub fn parse(s: &str) -> Result<Self, StatsError> {
        match s {
            "corwin_schultz" => Ok(SpreadModel::CorwinSchultz),
            "roll" => Ok(SpreadModel::Roll),
            _ => Err(StatsError::Invalid("spread_model is corwin_schultz or roll".into())),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImpactInputs {
    pub quantity: f64,
    pub average_daily_volume: f64,
    pub participation: f64,
    pub daily_sigma: f64,
    pub last_close: f64,
    pub spread_model: SpreadModel,
    pub estimated_spread_bps: f64,
    pub impact_coefficient: f64,
}

/// The result of [`impact_estimate`], full precision.
#[derive(Debug, Clone, PartialEq)]
pub struct Impact {
    pub days: usize,
    pub inputs: ImpactInputs,
    pub half_spread_bps: f64,
    pub impact_bps: f64,
    pub one_way_cost_bps: f64,
    pub one_way_cost_per_unit: f64,
    pub one_way_cost_total: f64,
    pub within_model_range: bool,
}

/// `3 - 2 sqrt(2)`
const K_CS: f64 = 3.0 - 2.0 * std::f64::consts::SQRT_2;

/// Mean two-day Corwin-Schultz (2012) spread as a fraction of price; a negative two-day
/// estimate counts as zero, as in the paper.
pub fn corwin_schultz(high: &[f64], low: &[f64]) -> f64 {
    let sq = |x: f64| x * x;
    let hl: Vec<f64> = high.iter().zip(low).map(|(h, l)| sq((h / l).ln())).collect();
    let n = hl.len() - 1;
    let spreads = (0..n).map(|i| {
        let beta = hl[i] + hl[i + 1];
        let h2 = high[i].max(high[i + 1]);
        let l2 = low[i].min(low[i + 1]);
        let gamma = sq((h2 / l2).ln());
        let alpha = ((2.0 * beta).sqrt() - beta.sqrt()) / K_CS - (gamma / K_CS).sqrt();
        let s = 2.0 * (alpha.exp() - 1.0) / (1.0 + alpha.exp());
        s.max(0.0)
    });
    fsum(spreads) / n as f64
}

/// Roll (1984): twice the root of minus the first-order autocovariance of price changes, as a
/// fraction of the mean price; `None` when that covariance is not negative.
pub fn roll(close: &[f64]) -> Option<f64> {
    let d: Vec<f64> = close.windows(2).map(|w| w[1] - w[0]).collect();
    let (a, b) = (&d[..d.len() - 1], &d[1..]);
    let (ma, mb) = (mean(a), mean(b));
    let cov =
        a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / (a.len() as f64 - 1.0);
    (cov < 0.0).then(|| 2.0 * (-cov).sqrt() / mean(close))
}

/// Daily bars, oldest first. `quantity` in units (shares or lots as the volume counts them).
pub fn impact_estimate(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    quantity: f64,
    spread_model: SpreadModel,
    coefficient: f64,
) -> Result<Impact, StatsError> {
    let c = finite_series(close, "close", MIN_DAYS)?;
    let h = finite_series(high, "high", MIN_DAYS)?;
    let lo = finite_series(low, "low", MIN_DAYS)?;
    let v = finite_series(volume, "volume", MIN_DAYS)?;
    if !(c.len() == h.len() && h.len() == lo.len() && lo.len() == v.len()) {
        return Err(StatsError::Invalid("high, low, close and volume differ in length".into()));
    }
    if c.iter().any(|x| *x <= 0.0)
        || h.iter().any(|x| *x <= 0.0)
        || lo.iter().any(|x| *x <= 0.0)
        || h.iter().zip(lo).any(|(a, b)| a < b)
    {
        return Err(StatsError::Invalid("prices must be positive with high not below low".into()));
    }
    if quantity.is_nan() || quantity <= 0.0 {
        return Err(StatsError::Invalid("quantity must be positive".into()));
    }
    let adv = mean(v);
    if adv <= 0.0 {
        return Err(StatsError::DegenerateSeries(
            "no volume in these bars (an index carries none): use its futures".into(),
        ));
    }
    let log_returns: Vec<f64> = c.windows(2).map(|w| w[1].ln() - w[0].ln()).collect();
    let lm = mean(&log_returns);
    let sigma = (log_returns.iter().map(|x| (x - lm) * (x - lm)).sum::<f64>()
        / (log_returns.len() as f64 - 1.0))
        .sqrt();
    let part = quantity / adv;
    let spread = match spread_model {
        SpreadModel::CorwinSchultz => corwin_schultz(h, lo),
        SpreadModel::Roll => roll(c).ok_or_else(|| {
            StatsError::DegenerateSeries(
                "Roll's estimate needs negative serial covariance of price changes; \
                 these bars have none: use corwin_schultz"
                    .into(),
            )
        })?,
    };
    let last = c[c.len() - 1];
    let half_bps = spread / 2.0 * 1e4;
    let impact_bps = coefficient * sigma * part.sqrt() * 1e4;
    let cost_bps = half_bps + impact_bps;
    Ok(Impact {
        days: c.len(),
        inputs: ImpactInputs {
            quantity,
            average_daily_volume: adv,
            participation: part,
            daily_sigma: sigma,
            last_close: last,
            spread_model,
            estimated_spread_bps: spread * 1e4,
            impact_coefficient: coefficient,
        },
        half_spread_bps: half_bps,
        impact_bps,
        one_way_cost_bps: cost_bps,
        one_way_cost_per_unit: last * cost_bps / 1e4,
        one_way_cost_total: quantity * last * cost_bps / 1e4,
        within_model_range: part <= BEYOND_MODEL,
    })
}
