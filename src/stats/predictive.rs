//! Does one causal signal predict the return that follows it: a regression, a rank correlation,
//! a hit rate, quantile means and a trading rule, each against a forward return that exists only
//! here and never feeds an expression.
//!
//! # Observations
//!
//! For a signal `s` and a price series `p` and a horizon `h` bars, the observation of bar `t` is
//! `(s[t], y[t])` with `y[t] = p[t + lag + h] / p[t + lag] - 1`: the return of entering `lag`
//! bars after the signal (default 1: a signal computed at a close cannot trade that close) and
//! holding `h` bars. A bar whose signal is NaN, or whose window runs past the data, has none.
//!
//! With `to_session_close` the observation is per session, one for each session in which the
//! signal becomes known: `t*` is the first bar of the session with a finite signal, the entry is
//! bar `t* + lag`, which must lie in the same session and before its last bar, and the return
//! runs to the close of that session's last bar. "Does the first hour predict the rest of the
//! day" is this. Sessions are labels per bar (equal labels, consecutive bars, are one session).
//!
//! # What is reported, per horizon
//!
//! * `n` observations; `n_independent`: those whose bar index is a multiple of `h` (their
//!   forward windows share no bar; in session mode all of them). The ordinary least squares
//!   line `y = a + b s` over all `n` is `slope`, `intercept`; the headline test is the t of the
//!   slope of the same regression over the independent observations only (`t_nonoverlap`,
//!   Student's t with `n_independent - 2` degrees of freedom, NaN under [`MIN_INDEPENDENT`]).
//! * `t_hac`: the slope's Newey-West t on all `n` observations, Bartlett weights, `h - 1` lags
//!   (none in session mode), no small-sample correction. It is **not calibrated**: the
//!   overlap of `h`-bar windows leaves it too large (measured in the Python docstring).
//! * `rank_corr` (Spearman, ties by average rank) and `hit_rate` (the share of observations
//!   with a signal and a return of the same sign, among those where both are non-zero).
//! * With `quantiles`: the mean return of each equal-count signal bucket over all observations,
//!   the spread top minus bottom, and the Welch t of that spread over the independent ones.
//! * The rule `position = sign(signal)` (long above zero, short below, flat at zero), held for
//!   the horizon: `rule_mean` and `rule_t_nonoverlap` over the independent observations (a
//!   portfolio rebalanced every `h` bars). Its `turnover` is the mean `|position change|` per
//!   rebalance (entering from flat counts one side; a flip two; in session mode every non-flat
//!   session is a round trip, entered and closed inside the session, so it counts two), and
//!   `breakeven_cost_bps_per_side = mean(rule return) / turnover` in basis points. With
//!   `cost_bps_per_side = c`: `net_mean` of `rule return - c |change|` and its `t_net`. A flat
//!   rate on the traded notional, not a broker's schedule; the last position is never closed
//!   (bar horizons).
//!
//! A signal that is constant has no slope: the t's are NaN and `signal_constant` is set.

use crate::math::special::t_two_sided_p;

use super::error::StatsError;
use super::hac::is_numerically_constant;

/// Fewest independent observations a headline t is computed on.
pub const MIN_INDEPENDENT: usize = 10;
/// Fewest independent observations below which the result says `small_sample`.
pub const SMALL_SAMPLE: usize = 30;
/// A one-bar price ratio beyond this, or below its inverse, is counted as a big move.
pub const JUMP_RATIO: f64 = 1.5;

#[derive(Debug, Clone)]
pub struct PredictiveParams {
    /// Horizons in bars; empty in session mode.
    pub horizons: Vec<usize>,
    pub lag: usize,
    pub quantiles: Option<usize>,
    pub cost_bps_per_side: Option<f64>,
    pub to_session_close: bool,
}

#[derive(Debug, Clone)]
pub struct PredictiveHorizon {
    /// The horizon as text (`"5"`), or `"session"`.
    pub key: String,
    pub horizon: Option<usize>,
    pub n: usize,
    pub n_independent: usize,
    pub slope: f64,
    pub intercept: f64,
    /// The slope over the independent observations alone.
    pub slope_nonoverlap: f64,
    pub t_nonoverlap: f64,
    pub p_nonoverlap: f64,
    pub t_hac: f64,
    pub hac_lags: usize,
    pub rank_corr: f64,
    pub hit_rate: f64,
    pub quantile_returns: Vec<f64>,
    pub quantile_spread: f64,
    pub quantile_spread_t_nonoverlap: f64,
    pub rule_mean: f64,
    pub rule_t_nonoverlap: f64,
    pub turnover: f64,
    pub breakeven_cost_bps_per_side: f64,
    /// NaN without a cost.
    pub net_mean: f64,
    pub t_net: f64,
    pub signal_constant: bool,
    /// The forward return of each bar's observation, NaN where there is none.
    pub forward_return: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PredictiveReport {
    pub n_bars: usize,
    pub n_signal_known: usize,
    pub lag: usize,
    pub to_session_close: bool,
    pub cost_bps_per_side: Option<f64>,
    /// One-bar price moves beyond [`JUMP_RATIO`].
    pub big_move_count: usize,
    pub horizons: Vec<PredictiveHorizon>,
}

struct Obs {
    t: usize,
    x: f64,
    y: f64,
}

struct Ols {
    slope: f64,
    intercept: f64,
    xbar: f64,
    sxx: f64,
    ssr: f64,
}

fn ols(x: &[f64], y: &[f64]) -> Option<Ols> {
    let n = x.len();
    if n < 2 || is_numerically_constant(x) {
        return None;
    }
    let nf = n as f64;
    let (xbar, ybar) = (x.iter().sum::<f64>() / nf, y.iter().sum::<f64>() / nf);
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxx += (a - xbar) * (a - xbar);
        sxy += (a - xbar) * (b - ybar);
    }
    let slope = sxy / sxx;
    let intercept = ybar - slope * xbar;
    let ssr = x.iter().zip(y).map(|(a, b)| (b - intercept - slope * a).powi(2)).sum();
    Some(Ols { slope, intercept, xbar, sxx, ssr })
}

/// The classical t of the slope, `n - 2` degrees of freedom, and its two-sided p-value.
fn slope_t(o: &Ols, n: usize) -> (f64, f64) {
    if n < MIN_INDEPENDENT {
        return (f64::NAN, f64::NAN);
    }
    let se = (o.ssr / (n as f64 - 2.0) / o.sxx).sqrt();
    let t = if se > 0.0 {
        o.slope / se
    } else if o.slope == 0.0 {
        return (f64::NAN, f64::NAN);
    } else {
        o.slope.signum() * f64::INFINITY
    };
    (t, t_two_sided_p(t, n as f64 - 2.0).unwrap_or(f64::NAN))
}

/// Newey-West t of the slope, Bartlett weights, `lags` lags, no small-sample correction.
fn slope_t_hac(x: &[f64], y: &[f64], o: &Ols, lags: usize) -> f64 {
    let n = x.len();
    if n < MIN_INDEPENDENT {
        return f64::NAN;
    }
    let u: Vec<f64> =
        x.iter().zip(y).map(|(a, b)| (a - o.xbar) * (b - o.intercept - o.slope * a)).collect();
    let mut s: f64 = u.iter().map(|v| v * v).sum();
    for j in 1..=lags.min(n - 1) {
        let w = 1.0 - j as f64 / (lags as f64 + 1.0);
        s += 2.0 * w * (j..n).map(|i| u[i] * u[i - j]).sum::<f64>();
    }
    let var = s / (o.sxx * o.sxx);
    if var > 0.0 {
        o.slope / var.sqrt()
    } else {
        f64::NAN
    }
}

/// `(mean, t)` of the mean of `v`; the t is NaN under [`MIN_INDEPENDENT`] values or with no
/// variance.
fn mean_t(v: &[f64]) -> (f64, f64) {
    let n = v.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let m = v.iter().sum::<f64>() / nf;
    if n < MIN_INDEPENDENT {
        return (m, f64::NAN);
    }
    let var = v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (nf - 1.0);
    (m, if var > 0.0 { m / (var / nf).sqrt() } else { f64::NAN })
}

/// Ranks from 1, ties by their average.
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*a].total_cmp(&v[*b]));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            out[idx[k]] = avg;
        }
        i = j + 1;
    }
    out
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        f64::NAN
    }
}

fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return f64::NAN;
    }
    let stat = |v: &[f64]| {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        (m, v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0) / n)
    };
    let ((ma, va), (mb, vb)) = (stat(a), stat(b));
    if va + vb > 0.0 {
        (ma - mb) / (va + vb).sqrt()
    } else {
        f64::NAN
    }
}

fn analyse(
    key: String,
    horizon: Option<usize>,
    obs: &[Obs],
    n_bars: usize,
    p: &PredictiveParams,
) -> PredictiveHorizon {
    let stride = horizon.unwrap_or(1);
    let hac_lags = horizon.map_or(0, |h| h - 1);
    let mut forward_return = vec![f64::NAN; n_bars];
    for o in obs {
        forward_return[o.t] = o.y;
    }
    let (xs, ys): (Vec<f64>, Vec<f64>) = obs.iter().map(|o| (o.x, o.y)).unzip();
    let ind: Vec<&Obs> = obs.iter().filter(|o| o.t % stride == 0).collect();
    let (xi, yi): (Vec<f64>, Vec<f64>) = ind.iter().map(|o| (o.x, o.y)).unzip();

    let full = ols(&xs, &ys);
    let sub = ols(&xi, &yi);
    let (t_nonoverlap, p_nonoverlap) =
        sub.as_ref().map_or((f64::NAN, f64::NAN), |o| slope_t(o, ind.len()));
    let t_hac = full.as_ref().map_or(f64::NAN, |o| slope_t_hac(&xs, &ys, o, hac_lags));

    let rank_corr = if is_numerically_constant(&xs) || obs.len() < 2 {
        f64::NAN
    } else {
        pearson(&ranks(&xs), &ranks(&ys))
    };
    let (mut agree, mut decided) = (0usize, 0usize);
    for (a, b) in xs.iter().zip(&ys) {
        if *a != 0.0 && *b != 0.0 {
            decided += 1;
            agree += usize::from((*a > 0.0) == (*b > 0.0));
        }
    }
    let hit_rate = if decided == 0 { f64::NAN } else { agree as f64 / decided as f64 };

    let (mut quantile_returns, mut quantile_spread, mut spread_t) =
        (Vec::new(), f64::NAN, f64::NAN);
    if let Some(q) = p.quantiles {
        if obs.len() >= q && !is_numerically_constant(&xs) {
            let r = ranks(&xs);
            let nf = obs.len() as f64;
            let bucket: Vec<usize> = r
                .iter()
                .map(|v| (((v - 0.5) / nf * q as f64).floor() as usize).min(q - 1))
                .collect();
            let mut sum = vec![0.0; q];
            let mut cnt = vec![0usize; q];
            for (b, y) in bucket.iter().zip(&ys) {
                sum[*b] += y;
                cnt[*b] += 1;
            }
            quantile_returns = (0..q)
                .map(|b| if cnt[b] > 0 { sum[b] / cnt[b] as f64 } else { f64::NAN })
                .collect();
            quantile_spread = quantile_returns[q - 1] - quantile_returns[0];
            // the buckets of the independent observations are the buckets of the whole sample
            let pick = |b: usize| -> Vec<f64> {
                obs.iter()
                    .zip(&bucket)
                    .filter(|(o, bk)| **bk == b && o.t % stride == 0)
                    .map(|(o, _)| o.y)
                    .collect()
            };
            spread_t = welch_t(&pick(q - 1), &pick(0));
        } else {
            quantile_returns = vec![f64::NAN; q];
        }
    }

    // the rule, on the independent observations: a portfolio rebalanced every `stride` bars
    let pos: Vec<f64> = xi
        .iter()
        .map(|x| {
            if *x > 0.0 {
                1.0
            } else if *x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();
    let rule: Vec<f64> = pos.iter().zip(&yi).map(|(a, b)| a * b).collect();
    // Session mode: the return runs from the entry to that session's close, so the overnight is
    // not held and every non-flat session is a round trip (one side in, one out). Bar horizons
    // hold the position until the next rebalance, so only the change is traded.
    let change: Vec<f64> = pos
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if p.to_session_close {
                2.0 * v.abs()
            } else {
                (v - if i == 0 { 0.0 } else { pos[i - 1] }).abs()
            }
        })
        .collect();
    let (rule_mean, rule_t_nonoverlap) = mean_t(&rule);
    let turnover =
        if change.is_empty() { f64::NAN } else { change.iter().sum::<f64>() / change.len() as f64 };
    let breakeven = if rule.is_empty() {
        f64::NAN
    } else if turnover > 0.0 {
        rule_mean / turnover * 1e4
    } else if rule_mean > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let (net_mean, t_net) = match p.cost_bps_per_side {
        Some(bps) => {
            let net: Vec<f64> = rule.iter().zip(&change).map(|(r, c)| r - bps * 1e-4 * c).collect();
            mean_t(&net)
        }
        None => (f64::NAN, f64::NAN),
    };

    PredictiveHorizon {
        key,
        horizon,
        n: obs.len(),
        n_independent: ind.len(),
        slope: full.as_ref().map_or(f64::NAN, |o| o.slope),
        intercept: full.as_ref().map_or(f64::NAN, |o| o.intercept),
        slope_nonoverlap: sub.as_ref().map_or(f64::NAN, |o| o.slope),
        t_nonoverlap,
        p_nonoverlap,
        t_hac,
        hac_lags,
        rank_corr,
        hit_rate,
        quantile_returns,
        quantile_spread,
        quantile_spread_t_nonoverlap: spread_t,
        rule_mean,
        rule_t_nonoverlap,
        turnover,
        breakeven_cost_bps_per_side: breakeven,
        net_mean,
        t_net,
        signal_constant: is_numerically_constant(&xs),
        forward_return,
    }
}

/// Test `signal` against the returns of `prices` that follow it. `sessions` holds a label per
/// bar and is required exactly with `to_session_close`.
pub fn predictive_test(
    signal: &[f64],
    prices: &[f64],
    sessions: Option<&[i64]>,
    params: &PredictiveParams,
) -> Result<PredictiveReport, StatsError> {
    let invalid = |s: String| StatsError::Invalid(s);
    let n = signal.len();
    if prices.len() != n {
        return Err(invalid(format!(
            "signal and prices must have one value per bar, got {n} and {}",
            prices.len()
        )));
    }
    if signal.iter().any(|v| v.is_infinite()) || prices.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite { what: "signal or prices".into() });
    }
    if prices.iter().any(|v| *v <= 0.0) {
        return Err(invalid("prices must be positive".into()));
    }
    if params.cost_bps_per_side.is_some_and(|c| !(c.is_finite() && c >= 0.0)) {
        return Err(invalid("cost_bps_per_side must be a number of at least 0".into()));
    }
    if params.quantiles.is_some_and(|q| q < 2) {
        return Err(invalid("quantiles must be at least 2".into()));
    }
    if params.to_session_close != sessions.is_some() {
        return Err(invalid("session labels are given exactly with to_session_close".into()));
    }
    if params.to_session_close != params.horizons.is_empty() {
        return Err(invalid(
            "horizons go with bar horizons, and to_session_close takes none".into(),
        ));
    }
    if params.horizons.contains(&0) {
        return Err(invalid("horizons must be integers >= 1".into()));
    }
    if sessions.is_some_and(|s| s.len() != n) {
        return Err(invalid("session labels must have one value per bar".into()));
    }
    let known = signal.iter().filter(|v| v.is_finite()).count();
    if known < MIN_INDEPENDENT {
        return Err(StatsError::InsufficientObservations {
            what: "signal".into(),
            found: known,
            needed: MIN_INDEPENDENT,
        });
    }
    let lag = params.lag;
    let big_move_count = prices
        .windows(2)
        .filter(|w| {
            let r = w[1] / w[0];
            !(1.0 / JUMP_RATIO..=JUMP_RATIO).contains(&r)
        })
        .count();

    let mut horizons = Vec::new();
    if let Some(labels) = sessions {
        let mut obs = Vec::new();
        let mut start = 0;
        while start < n {
            let mut end = start;
            while end + 1 < n && labels[end + 1] == labels[start] {
                end += 1;
            }
            // the first bar of the session whose signal is known, entered `lag` bars later
            if let Some(t) = (start..=end).find(|t| signal[*t].is_finite()) {
                let entry = t.saturating_add(lag);
                if entry < end {
                    obs.push(Obs { t, x: signal[t], y: prices[end] / prices[entry] - 1.0 });
                }
            }
            start = end + 1;
        }
        if obs.len() < 2 {
            return Err(StatsError::InsufficientObservations {
                what: "sessions with a known signal and a bar to enter".into(),
                found: obs.len(),
                needed: 2,
            });
        }
        horizons.push(analyse("session".into(), None, &obs, n, params));
    } else {
        for &h in &params.horizons {
            let obs: Vec<Obs> = (0..n.saturating_sub(lag.saturating_add(h)))
                .filter(|t| signal[*t].is_finite())
                .map(|t| Obs { t, x: signal[t], y: prices[t + lag + h] / prices[t + lag] - 1.0 })
                .collect();
            if obs.len() < 2 {
                return Err(StatsError::InsufficientObservations {
                    what: format!("observations at horizon {h}"),
                    found: obs.len(),
                    needed: 2,
                });
            }
            horizons.push(analyse(h.to_string(), Some(h), &obs, n, params));
        }
    }
    Ok(PredictiveReport {
        n_bars: n,
        n_signal_known: known,
        lag,
        to_session_close: params.to_session_close,
        cost_bps_per_side: params.cost_bps_per_side,
        big_move_count,
        horizons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 40 observations, lag 1, h 1: the price series is built so that `y[t]` is the stated return. The signal
    /// has period 4 (-1, -1, 1, 1) and `y = 0.01 x + 0.001 z` with `z` = (1, -1, 1, -1), so
    /// `sum x z` is 0 per period, mean x = mean z = 0: slope 0.01 exactly, intercept 0, residual
    /// 0.001 z, SSR = 40e-6, t = 0.01 / sqrt(1e-6 / 38) = 10 sqrt(38). The rule is `x * y =
    /// 0.01 + 0.001 x z`, mean 0.01; its position changes are 1 (from flat), then 2 at each of
    /// the 19 flips (t = 2, 4, .., 38): mean 39 / 40 = 0.975; breakeven 0.01 / 0.975 = 102.564 bps.
    fn planted() -> (Vec<f64>, Vec<f64>) {
        let x = [-1.0, -1.0, 1.0, 1.0];
        let z = [1.0, -1.0, 1.0, -1.0];
        let mut signal = Vec::new();
        let mut prices = vec![100.0, 100.0];
        for t in 0..40 {
            signal.push(x[t % 4]);
            let r = 0.01 * x[t % 4] + 0.001 * z[t % 4];
            let last = *prices.last().unwrap();
            prices.push(last * (1.0 + r));
        }
        signal.extend([f64::NAN, f64::NAN]);
        (signal, prices)
    }

    #[test]
    fn a_planted_lead_by_hand() {
        let (s, p) = planted();
        let params = PredictiveParams {
            horizons: vec![1],
            lag: 1,
            quantiles: Some(2),
            cost_bps_per_side: Some(10.0),
            to_session_close: false,
        };
        let r = predictive_test(&s, &p, None, &params).unwrap();
        let h = &r.horizons[0];
        assert_eq!((h.n, h.n_independent), (40, 40));
        assert!((h.slope - 0.01).abs() < 1e-12 && h.intercept.abs() < 1e-12);
        assert!(
            (h.t_nonoverlap - 10.0 * 38f64.sqrt()).abs() < 1e-6 * h.t_nonoverlap,
            "{}",
            h.t_nonoverlap
        );
        assert_eq!(h.hit_rate, 1.0);
        assert!((h.rule_mean - 0.01).abs() < 1e-12);
        assert!((h.turnover - 0.975).abs() < 1e-12);
        assert!((h.breakeven_cost_bps_per_side - 0.01 / 0.975 * 1e4).abs() < 1e-9);
        assert!((h.net_mean - (0.01 - 0.001 * 0.975)).abs() < 1e-12);
        assert!(h.t_net > 10.0);
        // top bucket (x = 1) mean 0.01, bottom -0.01
        assert!((h.quantile_spread - 0.02).abs() < 1e-12);
        // the forward return of bar 0 is r[1] = 0.01 x[0] + 0.001 z[0] = -0.009
        assert!((h.forward_return[0] + 0.009).abs() < 1e-12);
        assert!(h.forward_return[40].is_nan());
    }

    #[test]
    fn overlapping_horizons_keep_every_hth_observation() {
        let (s, p) = planted();
        let params = PredictiveParams {
            horizons: vec![4],
            lag: 1,
            quantiles: None,
            cost_bps_per_side: None,
            to_session_close: false,
        };
        let r = predictive_test(&s, &p, None, &params).unwrap();
        let h = &r.horizons[0];
        // bars 0..=36 have a window (0..n - lag - h = 0..37): 37 observations; multiples of 4: 10
        assert_eq!((h.n, h.n_independent), (37, 10));
        assert_eq!(h.hac_lags, 3);
        assert!(h.net_mean.is_nan());
    }

    /// Twelve sessions of four bars; the signal is known from the second bar on (the first hour),
    /// entered one bar later (bar 2), held to bar 3: a return of `0.01 v` for the session value
    /// `v` = +-1, exactly.
    #[test]
    fn a_session_observation_runs_to_the_close() {
        let (mut signal, mut prices, mut labels) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..12 {
            let v = if k % 2 == 0 { 1.0 } else { -1.0 };
            signal.extend([f64::NAN, v, v, v]);
            prices.extend([100.0, 100.0, 100.0, 100.0 * (1.0 + 0.01 * v)]);
            labels.extend([k as i64; 4]);
        }
        let params = PredictiveParams {
            horizons: vec![],
            lag: 1,
            quantiles: None,
            cost_bps_per_side: None,
            to_session_close: true,
        };
        let r = predictive_test(&signal, &prices, Some(&labels), &params).unwrap();
        let h = &r.horizons[0];
        assert_eq!((h.key.as_str(), h.n, h.n_independent), ("session", 12, 12));
        assert!((h.slope - 0.01).abs() < 1e-12);
        // the observation sits at the first known bar of the session
        assert!((h.forward_return[1] - 0.01).abs() < 1e-12);
        assert!((h.forward_return[5] + 0.01).abs() < 1e-12);
        assert!(h.forward_return[2].is_nan() && h.forward_return[0].is_nan());
        // the same call without labels, or with horizons, is refused
        assert!(predictive_test(&signal, &prices, None, &params).is_err());
    }

    /// Every session enters at bar 2 and exits at its close, so each is a round trip: turnover 2,
    /// not the session-to-session position change. rule = sign(v) * 0.01 v = 0.01 every session;
    /// breakeven 0.01 / 2 * 1e4 = 50 bps per side; net at 10 bps = 0.01 - 0.001 * 2 = 0.008.
    #[test]
    fn a_session_pays_entry_and_exit_every_session() {
        let (mut signal, mut prices, mut labels) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..12 {
            let v = if k < 6 { 1.0 } else { -1.0 };
            signal.extend([f64::NAN, v, v, v]);
            prices.extend([100.0, 100.0, 100.0, 100.0 * (1.0 + 0.01 * v)]);
            labels.extend([k as i64; 4]);
        }
        let params = PredictiveParams {
            horizons: vec![],
            lag: 1,
            quantiles: None,
            cost_bps_per_side: Some(10.0),
            to_session_close: true,
        };
        let r = predictive_test(&signal, &prices, Some(&labels), &params).unwrap();
        let h = &r.horizons[0];
        assert!((h.rule_mean - 0.01).abs() < 1e-12);
        assert!((h.turnover - 2.0).abs() < 1e-12, "{}", h.turnover);
        assert!((h.breakeven_cost_bps_per_side - 50.0).abs() < 1e-9);
        assert!((h.net_mean - 0.008).abs() < 1e-12, "{}", h.net_mean);
    }

    #[test]
    fn what_cannot_be_tested_is_refused_or_nan() {
        let (_, p) = planted();
        let flat = vec![1.0; p.len()];
        let params = PredictiveParams {
            horizons: vec![1],
            lag: 1,
            quantiles: None,
            cost_bps_per_side: None,
            to_session_close: false,
        };
        let r = predictive_test(&flat, &p, None, &params).unwrap();
        let h = &r.horizons[0];
        assert!(h.signal_constant && h.t_nonoverlap.is_nan() && h.t_hac.is_nan());
        let none = vec![f64::NAN; p.len()];
        assert!(matches!(
            predictive_test(&none, &p, None, &params),
            Err(StatsError::InsufficientObservations { .. })
        ));
        let mut neg = p.clone();
        neg[3] = -1.0;
        assert!(predictive_test(&flat, &neg, None, &params).is_err());
    }
}
