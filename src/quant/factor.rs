//! Factor evaluation: how well a panel of signal values ranks the returns that follow it.
//!
//! This is the only place a forward return exists. The report reads a factor panel and a price
//! panel and returns numbers; nothing it computes can be fed back into an expression graph.
//!
//! # Timing
//!
//! A signal computed at the close of date `d` cannot trade that close. The forward return of
//! date `d` over `h` bars starts `lag` bars after the signal (default 1): it is
//! `price[d + lag + h] / price[d + lag] - 1`, entered and exited at those closes. `lag = 0`
//! trades the same close the signal was computed on and is optimistic; it is allowed so the
//! report can be pinned to `portfolio::factor_panel::rank_ic`, which it calls.
//!
//! # What is reported, per horizon `h`
//!
//! * The rank IC per date (Spearman correlation of that date's factor with its forward return
//!   over the names that have both), its mean, sample standard deviation, hit rate (the share of
//!   scored dates with IC > 0), a naive t-statistic `mean / (std / sqrt(n))` and the Newey-West
//!   t-statistic with Bartlett weights and `h - 1` lags. The daily ICs of an `h`-bar forward
//!   return share `h - 1` bars, so they overlap and the naive t is inflated by about `sqrt(h)`;
//!   the Newey-West one accounts for that. It is NaN with fewer than 30 scored dates, or when
//!   the lags reach half of them. Scored dates are not always consecutive (a date with too few
//!   names is skipped); the lags are counted in scored dates, an approximation then.
//! * The mean forward return of each of `quantiles` equal-count buckets of the factor (bucket
//!   `q - 1` has the highest factor values; ties share a bucket by their average rank), per date
//!   and averaged; the long-short spread (top bucket minus bottom bucket, equal-weighted per
//!   bucket) per date, its mean and its Newey-West t-statistic.
//!
//! Independent of horizon: the factor's rank autocorrelation between consecutive dates, the
//! turnover of each quantile (the share of the previous date's bucket that is not in the same
//! bucket now), and the coverage per date.
//!
//! No costs of any kind are charged: `no_costs` is true in the report, and a spread here is a
//! gross figure.
//!
//! # What the prices must be
//!
//! Prices are assumed adjusted for splits, bonuses and dividends, and the panel is assumed to be
//! free of survivorship bias. Neither is checked or repaired. A name that leaves (its exit price
//! is NaN or not positive) drops out of that window: it is **not** counted as a loss, so a
//! factor that picks names that go to zero is flattered. Delisting returns are not invented;
//! `names_left` counts, per horizon, the name-windows lost this way. A one-bar price ratio
//! beyond `jump_ratio` (default 1.5, either way) on any name is reported in `big_moves`: it is
//! most likely an unadjusted split, which reads as a loss and corrupts the IC and the spreads.

use crate::math::special::t_two_sided_p;
use crate::portfolio::factor_panel::{rank_ic, rank_row, RowScratch};
use crate::stats::error::StatsError;
use crate::stats::hac::{hac_tstat, MIN_HAC_OBS};

/// Fewest names a date needs to be scored, unless the caller says otherwise.
pub const DEFAULT_MIN_NAMES: usize = 10;

/// Fewest non-overlapping dates the headline t-statistic is computed on.
pub const MIN_INDEPENDENT: usize = 10;

/// A one-bar price ratio above this, or below its inverse, is reported as a big move.
pub const DEFAULT_JUMP_RATIO: f64 = 1.5;

/// At most this many big moves are listed (the count is of all).
const MAX_BIG_MOVES: usize = 50;

#[derive(Debug, Clone)]
pub struct FactorParams {
    pub horizons: Vec<usize>,
    pub quantiles: usize,
    pub lag: usize,
    pub min_names: usize,
    pub jump_ratio: f64,
}

impl Default for FactorParams {
    fn default() -> Self {
        Self {
            horizons: vec![1, 5, 20],
            quantiles: 5,
            lag: 1,
            min_names: DEFAULT_MIN_NAMES,
            jump_ratio: DEFAULT_JUMP_RATIO,
        }
    }
}

/// A one-bar price ratio beyond the report's `jump_ratio`.
#[derive(Debug, Clone, PartialEq)]
pub struct BigMove {
    pub date: usize,
    pub asset: usize,
    /// `price[date] / price[date - 1] - 1`.
    pub ret: f64,
}

/// A t-statistic of the mean of a series, with its two-sided p-value on `n - 1` degrees of
/// freedom; both NaN under [`MIN_INDEPENDENT`] values or when the series is constant.
#[derive(Debug, Clone, Copy)]
pub struct Independent {
    pub t: f64,
    pub p: f64,
    pub n: usize,
}

/// The t-statistic of every `step`-th entry (index a multiple of `step`) of `v` that is finite.
fn independent_t(v: &[f64], step: usize) -> Independent {
    let x: Vec<f64> = v.iter().step_by(step).copied().filter(|x| x.is_finite()).collect();
    let n = x.len();
    let nan = Independent { t: f64::NAN, p: f64::NAN, n };
    if n < MIN_INDEPENDENT {
        return nan;
    }
    let nf = n as f64;
    let m = x.iter().sum::<f64>() / nf;
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (nf - 1.0);
    if var <= 0.0 {
        return nan;
    }
    let t = m / (var / nf).sqrt();
    Independent { t, p: t_two_sided_p(t, nf - 1.0).unwrap_or(f64::NAN), n }
}

/// One forward horizon.
#[derive(Debug, Clone)]
pub struct HorizonReport {
    pub horizon: usize,
    /// Rank IC per signal date, NaN where the date was not scored.
    pub ic: Vec<f64>,
    pub n_scored: usize,
    pub mean_ic: f64,
    pub std_ic: f64,
    pub hit_rate: f64,
    pub mean_names: f64,
    pub t_naive: f64,
    pub t_hac: f64,
    pub hac_lags: usize,
    /// The IC of every `horizon`-th date: the headline t-statistic, which does not overlap.
    pub t_nonoverlap: f64,
    pub p_nonoverlap: f64,
    pub n_independent: usize,
    /// Per signal date, the names that had a usable entry price and a factor value but no
    /// usable exit price (NaN or not positive): they left within the horizon and were dropped.
    pub names_left: Vec<f64>,
    pub names_left_total: usize,
    /// `names_left_total` over the name-windows that were entered.
    pub names_left_share: f64,
    /// Mean forward return of each quantile over the scored dates.
    pub quantile_returns: Vec<f64>,
    /// `n_dates x quantiles`, row-major; NaN where the date has no return for that bucket.
    pub quantile_returns_by_date: Vec<f64>,
    /// Top minus bottom quantile per date.
    pub long_short: Vec<f64>,
    pub long_short_mean: f64,
    pub long_short_t_hac: f64,
    pub long_short_t_nonoverlap: f64,
    pub long_short_p_nonoverlap: f64,
    pub long_short_n_independent: usize,
    pub long_short_n: usize,
}

#[derive(Debug, Clone)]
pub struct FactorReport {
    pub n_dates: usize,
    pub n_assets: usize,
    pub lag: usize,
    pub quantiles: usize,
    pub min_names: usize,
    pub jump_ratio: f64,
    /// One-bar price moves beyond `jump_ratio`, in date order (at most the first 50 listed).
    pub big_moves: Vec<BigMove>,
    pub big_move_count: usize,
    /// Always true: nothing here charges a cost.
    pub no_costs: bool,
    /// Names with a finite factor value and a finite positive price, per date.
    pub names: Vec<f64>,
    /// `names / n_assets`.
    pub coverage: Vec<f64>,
    /// Rank correlation of the factor with its previous date's, per date (NaN on the first).
    pub autocorr: Vec<f64>,
    pub autocorr_mean: f64,
    /// `n_dates x quantiles`, row-major.
    pub turnover_by_date: Vec<f64>,
    pub turnover: Vec<f64>,
    pub horizons: Vec<HorizonReport>,
}

fn mean_of(v: &[f64]) -> (f64, usize) {
    let (mut s, mut n) = (0.0, 0usize);
    for x in v.iter().filter(|x| x.is_finite()) {
        s += *x;
        n += 1;
    }
    (if n == 0 { f64::NAN } else { s / n as f64 }, n)
}

/// Newey-West t of the mean of the finite values, `lags` lags; NaN when it cannot be computed.
fn hac_t(v: &[f64], lags: usize) -> f64 {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.len() < MIN_HAC_OBS {
        return f64::NAN;
    }
    hac_tstat(&finite, Some(lags as i64)).map_or(f64::NAN, |h| h.hac_t)
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let (mx, my) = (x.iter().sum::<f64>() / n as f64, y.iter().sum::<f64>() / n as f64);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (da, db) = (a - mx, b - my);
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx * syy).sqrt()
}

fn bucket_of(u: f64, q: usize) -> usize {
    ((u * q as f64).floor() as usize).min(q - 1)
}

/// Evaluate `factor` against `prices`, both row-major `n_dates x n_assets`.
pub fn factor_report(
    factor: &[f64],
    prices: &[f64],
    n_dates: usize,
    n_assets: usize,
    params: &FactorParams,
) -> Result<FactorReport, StatsError> {
    let FactorParams { horizons, quantiles, lag, min_names, jump_ratio } = params;
    let (quantiles, lag, min_names, jump_ratio) = (*quantiles, *lag, *min_names, *jump_ratio);
    let (n, m) = (n_dates, n_assets);
    let invalid = |s: String| StatsError::Invalid(s);
    if horizons.is_empty() || horizons.contains(&0) {
        return Err(invalid("horizons must be a non-empty list of integers >= 1".into()));
    }
    if quantiles < 2 {
        return Err(invalid(format!("quantiles must be at least 2, got {quantiles}")));
    }
    if min_names < 2 {
        return Err(invalid(format!("min_names must be at least 2, got {min_names}")));
    }
    if !(jump_ratio.is_finite() && jump_ratio > 1.0) {
        return Err(invalid(format!("jump_ratio must be a number above 1, got {jump_ratio}")));
    }
    if factor.len() != n * m || prices.len() != n * m {
        return Err(invalid(format!(
            "factor and prices must both be {n} x {m} = {} values, got {} and {}",
            n * m,
            factor.len(),
            prices.len()
        )));
    }
    if factor.iter().chain(prices).any(|v| v.is_infinite()) {
        return Err(StatsError::NonFinite { what: "factor or prices".into() });
    }
    let need_names = min_names.max(quantiles);
    if m < need_names {
        return Err(StatsError::InsufficientObservations {
            what: "names".into(),
            found: m,
            needed: need_names,
        });
    }
    let h_max = *horizons.iter().max().unwrap_or(&1);
    let need_dates = lag + h_max + MIN_HAC_OBS;
    if n < need_dates {
        return Err(StatsError::InsufficientObservations {
            what: "dates".into(),
            found: n,
            needed: need_dates,
        });
    }

    let mut big_moves = Vec::new();
    let mut big_move_count = 0usize;
    for d in 1..n {
        for a in 0..m {
            let (p0, p1) = (prices[(d - 1) * m + a], prices[d * m + a]);
            if p0.is_finite() && p1.is_finite() && p0 > 0.0 && p1 > 0.0 {
                let ratio = p1 / p0;
                if ratio > jump_ratio || ratio < 1.0 / jump_ratio {
                    big_move_count += 1;
                    if big_moves.len() < MAX_BIG_MOVES {
                        big_moves.push(BigMove { date: d, asset: a, ret: ratio - 1.0 });
                    }
                }
            }
        }
    }

    // Per date: coverage and the quantile of each name on the factor alone (turnover, autocorr).
    let mut names = vec![0.0; n];
    let mut coverage = vec![0.0; n];
    let mut u_all = vec![f64::NAN; n * m];
    let mut masked = vec![f64::NAN; m];
    let mut scratch = RowScratch::with_capacity(m);
    for d in 0..n {
        let (f, p) = (&factor[d * m..(d + 1) * m], &prices[d * m..(d + 1) * m]);
        let mut count = 0usize;
        for a in 0..m {
            let ok = f[a].is_finite() && p[a].is_finite() && p[a] > 0.0;
            masked[a] = if ok { f[a] } else { f64::NAN };
            count += usize::from(ok);
        }
        names[d] = count as f64;
        coverage[d] = count as f64 / m as f64;
        rank_row(&masked, need_names, &mut u_all[d * m..(d + 1) * m], &mut scratch);
    }

    let mut autocorr = vec![f64::NAN; n];
    let mut turnover_by_date = vec![f64::NAN; n * quantiles];
    for d in 1..n {
        let (cur, prev) = (&u_all[d * m..(d + 1) * m], &u_all[(d - 1) * m..d * m]);
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let mut kept = vec![0usize; quantiles];
        let mut before = vec![0usize; quantiles];
        for a in 0..m {
            if prev[a].is_finite() {
                before[bucket_of(prev[a], quantiles)] += 1;
                if cur[a].is_finite()
                    && bucket_of(cur[a], quantiles) == bucket_of(prev[a], quantiles)
                {
                    kept[bucket_of(cur[a], quantiles)] += 1;
                }
            }
            if cur[a].is_finite() && prev[a].is_finite() {
                xs.push(cur[a]);
                ys.push(prev[a]);
            }
        }
        autocorr[d] = pearson(&xs, &ys);
        if cur.iter().any(|v| v.is_finite()) {
            for q in 0..quantiles {
                if before[q] > 0 {
                    turnover_by_date[d * quantiles + q] = 1.0 - kept[q] as f64 / before[q] as f64;
                }
            }
        }
    }
    let turnover: Vec<f64> = (0..quantiles)
        .map(|q| {
            let col: Vec<f64> = (0..n).map(|d| turnover_by_date[d * quantiles + q]).collect();
            mean_of(&col).0
        })
        .collect();

    let mut reports = Vec::with_capacity(horizons.len());
    for &h in horizons {
        // The forward return starts `lag` bars after the signal: score the factor of date d
        // against the prices from d + lag on.
        let live = n - lag;
        let ic = rank_ic(&factor[..live * m], &prices[lag * m..], live, m, h, need_names)
            .map_err(|e| invalid(e.to_string()))?;
        let mut daily = ic.daily_ic.clone();
        daily.resize(n, f64::NAN);
        let scored: Vec<f64> = daily.iter().copied().filter(|v| v.is_finite()).collect();
        let hit = scored.iter().filter(|v| **v > 0.0).count();
        let lags = h - 1;

        let mut by_date = vec![f64::NAN; n * quantiles];
        let mut spread = vec![f64::NAN; n];
        let mut left = vec![f64::NAN; n];
        let (mut left_total, mut entered) = (0usize, 0usize);
        let mut row = vec![f64::NAN; m];
        let mut u = vec![f64::NAN; m];
        for d in 0..n.saturating_sub(lag + h) {
            let (p0, p1) = ((d + lag) * m, (d + lag + h) * m);
            let mut gone = 0usize;
            for a in 0..m {
                let f = factor[d * m + a];
                let (a0, a1) = (prices[p0 + a], prices[p1 + a]);
                let ok = f.is_finite() && a0.is_finite() && a1.is_finite() && a0 > 0.0 && a1 > 0.0;
                row[a] = if ok { f } else { f64::NAN };
                if f.is_finite() && a0.is_finite() && a0 > 0.0 {
                    entered += 1;
                    if !(a1.is_finite() && a1 > 0.0) {
                        gone += 1;
                    }
                }
            }
            left[d] = gone as f64;
            left_total += gone;
            u.fill(f64::NAN);
            rank_row(&row, need_names, &mut u, &mut scratch);
            let (mut sum, mut cnt) = (vec![0.0; quantiles], vec![0usize; quantiles]);
            for a in 0..m {
                if u[a].is_finite() {
                    let b = bucket_of(u[a], quantiles);
                    sum[b] += prices[p1 + a] / prices[p0 + a] - 1.0;
                    cnt[b] += 1;
                }
            }
            for q in 0..quantiles {
                if cnt[q] > 0 {
                    by_date[d * quantiles + q] = sum[q] / cnt[q] as f64;
                }
            }
            spread[d] = by_date[d * quantiles + quantiles - 1] - by_date[d * quantiles];
        }
        let quantile_returns: Vec<f64> = (0..quantiles)
            .map(|q| {
                let col: Vec<f64> = (0..n).map(|d| by_date[d * quantiles + q]).collect();
                mean_of(&col).0
            })
            .collect();
        let (ls_mean, ls_n) = mean_of(&spread);
        let ic_ind = independent_t(&daily, h);
        let ls_ind = independent_t(&spread, h);
        reports.push(HorizonReport {
            horizon: h,
            n_scored: ic.n_dates_scored,
            mean_ic: ic.mean_ic,
            std_ic: ic.stdev_ic,
            hit_rate: if scored.is_empty() { f64::NAN } else { hit as f64 / scored.len() as f64 },
            mean_names: ic.mean_names,
            t_naive: ic.t_stat,
            t_hac: hac_t(&daily, lags),
            hac_lags: lags,
            t_nonoverlap: ic_ind.t,
            p_nonoverlap: ic_ind.p,
            n_independent: ic_ind.n,
            names_left: left,
            names_left_total: left_total,
            names_left_share: if entered == 0 {
                f64::NAN
            } else {
                left_total as f64 / entered as f64
            },
            ic: daily,
            quantile_returns,
            quantile_returns_by_date: by_date,
            long_short_t_hac: hac_t(&spread, lags),
            long_short_t_nonoverlap: ls_ind.t,
            long_short_p_nonoverlap: ls_ind.p,
            long_short_n_independent: ls_ind.n,
            long_short: spread,
            long_short_mean: ls_mean,
            long_short_n: ls_n,
        });
    }

    Ok(FactorReport {
        n_dates: n,
        n_assets: m,
        lag,
        quantiles,
        min_names,
        jump_ratio,
        big_moves,
        big_move_count,
        no_costs: true,
        names,
        coverage,
        autocorr_mean: mean_of(&autocorr).0,
        autocorr,
        turnover_by_date,
        turnover,
        horizons: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prices where name `a` grows at a steady rate `a` percent a bar, and the factor is `a`:
    /// every date ranks the names by the return that follows, exactly.
    fn ladder(n: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
        let mut factor = Vec::new();
        let mut prices = Vec::new();
        for d in 0..n {
            for a in 0..m {
                factor.push(a as f64);
                prices.push(100.0 * (1.0 + 0.001 * (a + 1) as f64).powi(d as i32));
            }
        }
        (factor, prices)
    }

    #[test]
    fn a_perfect_factor_has_ic_one_and_the_top_bucket_beats_the_bottom() {
        let (f, p) = ladder(80, 20);
        let r = factor_report(&f, &p, 80, 20, &FactorParams::default()).unwrap();
        assert!(r.no_costs);
        for h in &r.horizons {
            assert!((h.mean_ic - 1.0).abs() < 1e-12);
            assert_eq!(h.hit_rate, 1.0);
            assert!(h.long_short_mean > 0.0);
            assert!(h.quantile_returns.windows(2).all(|w| w[0] < w[1]));
        }
        // 80 dates, lag 1: horizon 1 scores dates 0..78.
        assert_eq!(r.horizons[0].n_scored, 78);
        // A static factor never changes buckets.
        assert!(r.turnover.iter().all(|t| *t == 0.0));
        assert!((r.autocorr_mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_nonoverlapping_t_takes_every_hth_entry_by_hand() {
        // 0, 2, .., 38: 20 values, mean 19, variance 4 * var(0..19) = 4 * (20 * 21 / 12) = 140,
        // t = 19 / sqrt(140 / 20) = 19 / sqrt(7), on 19 degrees of freedom.
        let v: Vec<f64> = (0..40).map(f64::from).collect();
        let r = independent_t(&v, 2);
        assert_eq!(r.n, 20);
        assert!((r.t - 19.0 / 7.0f64.sqrt()).abs() < 1e-12, "{}", r.t);
        assert!(r.p > 0.0 && r.p < 1e-5, "{}", r.p);
        // NaN entries do not count, and fewer than MIN_INDEPENDENT is NaN
        let mut w = v.clone();
        w[4] = f64::NAN;
        assert_eq!(independent_t(&w, 2).n, 19);
        let short = independent_t(&v[..12], 2);
        assert_eq!(short.n, 6);
        assert!(short.t.is_nan() && short.p.is_nan());
        // a constant series has no variance
        assert!(independent_t(&[1.0; 40], 2).t.is_nan());
    }

    #[test]
    fn a_name_without_an_exit_price_is_counted_as_left_not_scored() {
        let (f, mut p) = ladder(80, 20);
        // name 19 prints nothing from row 50: windows entered at row 45..=49 (signal dates 44..=48,
        // lag 1, h 5) end at 50..=54 without a price
        for d in 50..80 {
            p[d * 20 + 19] = f64::NAN;
        }
        let params = FactorParams { horizons: vec![5], ..FactorParams::default() };
        let r = factor_report(&f, &p, 80, 20, &params).unwrap();
        let h = &r.horizons[0];
        assert_eq!(h.names_left_total, 5);
        let at: Vec<usize> = (0..80).filter(|d| h.names_left[*d] > 0.0).collect();
        assert_eq!(at, vec![44, 45, 46, 47, 48]);
    }

    #[test]
    fn a_split_is_a_big_move() {
        let (f, mut p) = ladder(80, 20);
        for d in 40..80 {
            p[d * 20 + 7] /= 10.0;
        }
        let r = factor_report(&f, &p, 80, 20, &FactorParams::default()).unwrap();
        assert_eq!(r.big_move_count, 1);
        assert_eq!((r.big_moves[0].date, r.big_moves[0].asset), (40, 7));
        // the ladder's name 7 also grows 0.8 % that bar: 1.008 / 10 - 1
        assert!((r.big_moves[0].ret - (1.008 / 10.0 - 1.0)).abs() < 1e-12);
        let bad = FactorParams { jump_ratio: 1.0, ..FactorParams::default() };
        assert!(matches!(factor_report(&f, &p, 80, 20, &bad), Err(StatsError::Invalid(_))));
    }

    #[test]
    fn lag_zero_is_rank_ic() {
        let (f, p) = ladder(60, 12);
        let params = FactorParams { horizons: vec![3], lag: 0, ..FactorParams::default() };
        let r = factor_report(&f, &p, 60, 12, &params).unwrap();
        let direct = rank_ic(&f, &p, 60, 12, 3, DEFAULT_MIN_NAMES).unwrap();
        assert_eq!(r.horizons[0].mean_ic.to_bits(), direct.mean_ic.to_bits());
    }

    #[test]
    fn too_few_dates_or_names_is_refused() {
        let (f, p) = ladder(20, 20);
        assert!(matches!(
            factor_report(&f, &p, 20, 20, &FactorParams::default()),
            Err(StatsError::InsufficientObservations { .. })
        ));
        let (f, p) = ladder(80, 4);
        assert!(matches!(
            factor_report(&f, &p, 80, 4, &FactorParams::default()),
            Err(StatsError::InsufficientObservations { .. })
        ));
    }
}
