//! `at("1d", expression)`: an expression evaluated on a coarser clock and held on the base one.
//!
//! `at` is not in the registry: it needs the bar's timestamp, the zone, the bar stamp and the
//! instrument's session, which a kernel `step` does not see. The checker builds an `Op::At`
//! node that owns the inner expression as a program of its own (nothing inside is shared with
//! the outside), and the evaluator runs it through [`AtState`].
//!
//! # What an `at` value is
//!
//! The base bars are cut into windows by the *same* code the runtime's aggregator hub uses
//! ([`AggKey::with_session`], [`AggKey::builder`], [`AggKey::bucketed`]: one aggregation). A
//! time-window builder emits a window's bar when the first record of a *later* window arrives
//! (`data/aggregate.rs`), and the node steps the inner expression on that emission and holds
//! the result until the next one. So a daily value first appears on the first base bar of the
//! next day, in batch and in a stream alike, and every number it carries comes from windows
//! that were complete before that bar. Visibility is decided by emission, never by the window's
//! end stamp.
//!
//! * The window in progress is never read. The final partial window of a run is never emitted
//!   (nothing calls `flush`, as in the hub), so the last day's value is not part of any output.
//! * No builder emits at a session close: the session-anchored builder and the plain time
//!   builder both emit on the first record of the next window, and `flush` is for end of data
//!   only. The value that closes a day is therefore first visible on the next day's first
//!   bar, even when the stream has a schedule.
//! * Before the first emission, and until the inner expression's own warm-up has passed, the
//!   value is NaN.
//! * An absent bar (NaN fields) and a day the schedule closes follow the builder: a record on a
//!   non-trading day is dropped when a schedule is bound, and a NaN field enters the window as
//!   the builder takes it.
//!
//! # Cost
//!
//! A step is one builder push and, on an emission, one inner stream step. The inner stream and
//! the register file allocate nothing per bar. The builder itself is the hub's: with a session
//! bound, `SessionSchedule::windows_ns_on_day_of` returns a `Vec`, so that case allocates per
//! bar exactly as the hub does.

use crate::data::{AggregationUnit, BarBuilder, BarSpec, SourceRecord};
use crate::quant::eval::{Bar, ClockCtx, EvalError, Plan, Stream};
use crate::quant::registry::FnDef;
use crate::runtime::aggregators::AggKey;
use crate::runtime::ids::InstrumentId;

/// No registry rows: `at` is built by the checker, see the module doc.
pub const DEFS: &[FnDef] = &[];

/// Bumped when what `at` computes changes; part of the hash of every program that calls it.
pub const REV: u32 = 1;

/// Timeframes spelled in the error that names what is valid.
pub const EXAMPLES: [&str; 11] =
    ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"];

/// Heap of the boxed builder when it is known exactly: the plain time builder or the calendar
/// (month, year) builder, with no session bound. With a session the builder is a private type
/// that clones the schedule, and the `BarBuilder` trait cannot report its heap, so that case is
/// counted as the plain builder's size and is NOT exact: the schedule's tables are missing.
fn builder_bytes(spec: BarSpec) -> usize {
    match spec.unit {
        AggregationUnit::Month | AggregationUnit::Year => {
            std::mem::size_of::<crate::data::aggregate::CalendarBarBuilder>()
        }
        _ => std::mem::size_of::<crate::data::aggregate::TimeBarBuilder>(),
    }
}

/// `"15m"`, `"1h"`, `"1d"`, `"1w"`, `"1month"`, `"1y"`: a step and a time-window unit.
pub fn parse_timeframe(s: &str) -> Result<BarSpec, String> {
    let digits = s.bytes().take_while(u8::is_ascii_digit).count();
    let (step, unit) = s.split_at(digits);
    let step: u32 = step.parse().map_err(|_| format!("`{s}` has no step (write 1d, not d)"))?;
    let unit = AggregationUnit::parse(unit).map_err(|_| format!("`{s}` has an unknown unit"))?;
    if !unit.is_time_window() {
        return Err(format!("`{s}` is not a clock timeframe; at(...) cuts bars by time"));
    }
    BarSpec::new(step, unit).map_err(|e| format!("`{s}`: {e}"))
}

/// The canonical spelling of a timeframe: `1d`, `15m`.
pub fn timeframe_text(spec: BarSpec) -> String {
    let unit = match spec.unit {
        AggregationUnit::Millisecond => "ms",
        AggregationUnit::Second => "s",
        AggregationUnit::Minute => "m",
        AggregationUnit::Hour => "h",
        AggregationUnit::Day => "d",
        AggregationUnit::Week => "w",
        AggregationUnit::Month => "month",
        AggregationUnit::Year => "y",
        // Refused by `parse_timeframe`.
        _ => "?",
    };
    format!("{}{unit}", spec.step)
}

/// The width of a fixed-width timeframe in nanoseconds; none for a month or a year.
pub fn width_ns(spec: BarSpec) -> Option<i64> {
    spec.unit.fixed_ns().map(|u| u * i64::from(spec.step))
}

/// Whether `spec` is a coarser clock the base bars of width `base` (ns) can be cut into. A
/// fixed-width timeframe must be a coarser exact multiple of the base. A day (or week) of a run
/// with a session is cut by the session's date, not by 24 hours, so bars finer than the session
/// are enough: 75-minute bars make five to an NSE session though they do not divide the
/// 24-hour day.
fn accepts(spec: BarSpec, base: i64, session_ns: Option<i64>) -> bool {
    let Some(tf) = width_ns(spec) else { return true };
    if base <= 0 {
        return false;
    }
    if matches!(spec.unit, AggregationUnit::Day | AggregationUnit::Week)
        && session_ns.is_some_and(|day| base < day)
    {
        return true;
    }
    tf > base && tf % base == 0
}

/// The length of a trading session in nanoseconds, when the clock has a schedule.
fn session_len_ns(clock: &ClockCtx) -> Option<i64> {
    let minutes = clock.session.as_ref()?.trading_minutes();
    Some((minutes * 60.0 * 1e9).round() as i64).filter(|ns| *ns > 0)
}

/// `ns` as the largest whole unit that holds it exactly (`"14h"`), when that spelling parses.
fn multiple_text(ns: i64) -> Option<String> {
    const UNITS: [(i64, &str); 5] = [
        (86_400_000_000_000, "d"),
        (3_600_000_000_000, "h"),
        (60_000_000_000, "m"),
        (1_000_000_000, "s"),
        (1_000_000, "ms"),
    ];
    let (size, unit) = UNITS.iter().find(|(size, _)| ns % size == 0)?;
    let text = format!("{}{unit}", ns / size);
    parse_timeframe(&text).ok().map(|_| text)
}

/// The timeframes `base_ns` bars can be cut into: the examples that are a coarser exact
/// multiple (a day being the session's length, if `session_ns` is known). Never empty: when no
/// example fits, the first multiples of the base itself.
pub fn valid_for(base_ns: i64, session_ns: Option<i64>) -> Vec<String> {
    let base = base_ns.max(1);
    let mut valid: Vec<String> = EXAMPLES
        .iter()
        .filter_map(|t| parse_timeframe(t).ok().map(|s| (*t, s)))
        .filter(|(_, s)| accepts(*s, base, session_ns))
        .map(|(t, _)| t.to_string())
        .collect();
    if valid.is_empty() {
        valid = (2..=5).filter_map(|k| multiple_text(base.checked_mul(k)?)).collect();
    }
    valid
}

/// One `at` node's run state: the window builder, the inner expression's stream and the held
/// value.
pub struct AtState {
    builder: Box<dyn BarBuilder + Send + Sync>,
    key: AggKey,
    spec: BarSpec,
    sub: Stream,
    held: f64,
    /// No base width was given and the clock asks for it to be inferred: the first bar's stamp
    /// until the second arrives, then done.
    inferring: bool,
    first_ts: Option<i64>,
    session_ns: Option<i64>,
    refusal: Option<EvalError>,
}

impl AtState {
    /// A fresh state for `spec` over `inner`, cut as `clock` says. Refuses a timeframe that is
    /// not coarser than the base bars, or not an exact multiple of them, when the base width is
    /// known.
    pub fn new(spec: BarSpec, inner: &Plan, clock: &ClockCtx) -> Result<Self, EvalError> {
        if let Some(base) = clock.base_ns {
            let session = session_len_ns(clock);
            if !accepts(spec, base, session) {
                return Err(EvalError::Timeframe {
                    tf: timeframe_text(spec),
                    base_ns: base,
                    valid: valid_for(base, session),
                });
            }
        }
        let key = AggKey::new(InstrumentId::new(0), spec, 0.0, clock.tz_offset_ns, clock.stamp)
            .map_err(|e| EvalError::Node(e.to_string()))?;
        let (key, session) = key.with_session(clock.session.as_deref());
        let builder = key.builder(session).map_err(|e| EvalError::Node(e.to_string()))?;
        Ok(Self {
            builder,
            key,
            spec,
            sub: Stream::new(inner)?,
            held: f64::NAN,
            inferring: clock.infer_width && clock.base_ns.is_none(),
            first_ts: None,
            session_ns: session_len_ns(clock),
            refusal: None,
        })
    }

    /// Feed one base bar stamped `ts`; the held value after it. The value changes only on the
    /// bar that completes a window, and then it is the inner expression on that window.
    #[inline]
    pub fn step(&mut self, ts: i64, bar: &Bar) -> f64 {
        if self.inferring {
            self.infer(ts);
        }
        let rec = SourceRecord {
            timestamp: ts,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            signed_volume: 0.0,
        };
        if let Some(done) = self.builder.push(&self.key.bucketed(&rec)) {
            let window = Bar {
                open: done.open,
                high: done.high,
                low: done.low,
                close: done.close,
                volume: done.volume,
            };
            // The body reads no named series (the checker gives it none), so it takes no handle
            // values and cannot fail; were it to, the held value would read unknown.
            self.held = self
                .sub
                .update(&window, &[])
                .ok()
                .and_then(|o| o.first().copied())
                .unwrap_or(f64::NAN);
        }
        self.held
    }

    /// The base width from the first two stamps. Once only: the cell then neither looks nor
    /// allocates. A timeframe no coarser than it is a refusal for the runtime to raise before any
    /// hook reads a value.
    fn infer(&mut self, ts: i64) {
        let Some(first) = self.first_ts else {
            self.first_ts = Some(ts);
            return;
        };
        let gap = ts - first;
        if gap <= 0 {
            return;
        }
        self.inferring = false;
        if !accepts(self.spec, gap, self.session_ns) {
            self.refusal = Some(EvalError::Timeframe {
                tf: timeframe_text(self.spec),
                base_ns: gap,
                valid: valid_for(gap, self.session_ns),
            });
        }
    }

    /// The refusal found by the width inference, if any.
    pub fn refusal(&self) -> Option<&EvalError> {
        self.refusal.as_ref()
    }

    /// True while the width is still being worked out.
    pub fn inferring(&self) -> bool {
        self.inferring
    }

    /// True once the inner expression is past its own warm-up.
    pub fn ready(&self) -> bool {
        self.sub.ready()
    }

    /// Heap this state owns, not counting its own struct (the stream's `ats` vector holds it).
    pub fn state_bytes(&self) -> usize {
        builder_bytes(self.spec) + self.sub.state_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session_schedule::SessionSchedule;
    use crate::core::types::OhlcvBar;
    use crate::data::BarStamp;
    use crate::quant::eval::{eval, Inputs};
    use crate::quant::kernel::canon_bits;
    use crate::quant::lang::{compile, Env, Field};
    use crate::runtime::aggregators::AggregatorHubBuilder;
    use crate::runtime::ids::StrategyId;

    const HOUR: i64 = 3_600_000_000_000;
    const DAY: i64 = 24 * HOUR;
    const IST: i64 = 19_800_000_000_000;
    const NAN: f64 = f64::NAN;

    fn plan(src: &str) -> Plan {
        Plan::new(&compile(src, &Env::default()).unwrap_or_else(|e| panic!("{src}: {e}"))).unwrap()
    }

    fn clock(stamp: BarStamp, tz: i64) -> ClockCtx {
        ClockCtx::new(tz, stamp, None)
    }

    /// Hourly bars of `days` civil days (in the zone `tz`), 24 a day. Close-stamped: the hour-24
    /// bar of a day is stamped at the next midnight, exactly on the window boundary.
    fn hourly(days: i64, stamp: BarStamp, tz: i64) -> Vec<i64> {
        let shift = if stamp == BarStamp::Close { HOUR } else { 0 };
        (0..days * 24).map(|i| 19_000 * DAY - tz + i * HOUR + shift).collect()
    }

    /// Close of bar `i`: the day number and hour in it, so a window's last close is known.
    fn closes(days: i64) -> Vec<f64> {
        (0..days * 24).map(|i| 100.0 + (i / 24) as f64 * 10.0 + (i % 24 + 1) as f64).collect()
    }

    fn wide(c: &[f64]) -> [Vec<f64>; 4] {
        [
            c.iter().map(|x| x - 0.5).collect(),
            c.iter().map(|x| x + 1.0).collect(),
            c.iter().map(|x| x - 1.0).collect(),
            c.iter().enumerate().map(|(i, _)| 10.0 + (i % 7) as f64).collect(),
        ]
    }

    fn batch(p: &Plan, ts: &[i64], c: &[f64], clock: ClockCtx) -> Vec<f64> {
        let [o, h, l, v] = wide(c);
        let inputs = Inputs::new()
            .field(Field::Open, &o)
            .field(Field::High, &h)
            .field(Field::Low, &l)
            .field(Field::Close, c)
            .field(Field::Volume, &v)
            .timestamps(ts)
            .clock(clock);
        eval(p, &inputs).unwrap().remove(0)
    }

    fn stream(p: &Plan, ts: &[i64], c: &[f64], clock: ClockCtx) -> Vec<f64> {
        let [o, h, l, v] = wide(c);
        let mut s = Stream::bound(p, clock).unwrap();
        (0..c.len())
            .map(|i| {
                let bar = Bar { open: o[i], high: h[i], low: l[i], close: c[i], volume: v[i] };
                s.update_at(ts[i], &bar, &[]).unwrap()[0]
            })
            .collect()
    }

    fn assert_same(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(canon_bits(*x), canon_bits(*y), "{what}: bar {i}: {x} vs {y}");
        }
    }

    #[test]
    fn timeframes_parse_and_print_canonically() {
        for t in EXAMPLES.iter().chain(&["1s", "90m", "2w", "1month", "1y", "500ms"]) {
            assert_eq!(timeframe_text(parse_timeframe(t).unwrap()), *t);
        }
        for t in ["", "d", "0d", "1", "1x", "1tick", "100volume", "1renko", "-1d", "1 d", "1D"] {
            assert!(parse_timeframe(t).is_err(), "{t:?}");
        }
        assert_eq!(valid_for(HOUR, None), vec!["2h", "4h", "1d", "1w"]);
        // 75-minute bars: no example divides them, and a refusal still names what is valid;
        // a 375-minute session is five of them, so a day and a week are.
        let m75 = 75 * 60_000_000_000;
        assert_eq!(valid_for(m75, None), vec!["150m", "225m", "5h", "375m"]);
        assert_eq!(valid_for(m75, Some(375 * 60_000_000_000)), vec!["1d", "1w"]);
    }

    /// A daily value first appears on the first bar of the next day, is held all that day, and
    /// the last day's own window is never visible. The close-stamped bar at the midnight boundary
    /// belongs to the day it ends.
    #[test]
    fn a_daily_close_is_visible_from_the_next_days_first_bar() {
        let p = plan("at(\"1d\", close)");
        let (ts, c) = (hourly(3, BarStamp::Close, 0), closes(3));
        let out = batch(&p, &ts, &c, clock(BarStamp::Close, 0));
        // Day 0 bars: nothing yet. Day 1 bars: day 0's last close (hour 24: 124). Day 2: 134.
        assert!(out[..24].iter().all(|x| x.is_nan()));
        assert!(out[24..48].iter().all(|x| *x == 124.0), "{:?}", &out[24..48]);
        assert!(out[48..].iter().all(|x| *x == 134.0), "{:?}", &out[48..]);
        // Open-stamped bars of the same windows give the same numbers.
        let ts_open = hourly(3, BarStamp::Open, 0);
        assert_same(&out, &batch(&p, &ts_open, &c, clock(BarStamp::Open, 0)), "open vs close");
    }

    /// The daily SMA computed by hand against `at`, on the daily closes 124, 134, 144, 154.
    #[test]
    fn a_daily_sma_matches_the_hand_computation() {
        let p = plan("at(\"1d\", sma(close, 2))");
        let (ts, c) = (hourly(5, BarStamp::Close, 0), closes(5));
        let out = batch(&p, &ts, &c, clock(BarStamp::Close, 0));
        let day = |d: usize| &out[d * 24..(d + 1) * 24];
        assert!(day(0).iter().chain(day(1)).all(|x| x.is_nan()));
        assert!(day(2).iter().all(|x| *x == (124.0 + 134.0) / 2.0));
        assert!(day(3).iter().all(|x| *x == (134.0 + 144.0) / 2.0));
        assert!(day(4).iter().all(|x| *x == (144.0 + 154.0) / 2.0));
    }

    /// Inner fields are the aggregated bar's: high is the day's max, volume its sum.
    #[test]
    fn inner_fields_are_the_windows() {
        let p = plan("at(\"1d\", high - low)");
        let (ts, c) = (hourly(2, BarStamp::Close, 0), closes(2));
        let out = batch(&p, &ts, &c, clock(BarStamp::Close, 0));
        // Day 0: high = 124 + 1 at the last hour, low = 101 - 1 at the first.
        assert!(out[24..].iter().all(|x| *x == 125.0 - 100.0), "{:?}", &out[24..30]);
        let v = plan("at(\"1d\", volume)");
        let out = batch(&v, &ts, &c, clock(BarStamp::Close, 0));
        let day_volume: f64 = (0..24).map(|i| 10.0 + (i % 7) as f64).sum();
        assert!(out[24..].iter().all(|x| *x == day_volume));
    }

    #[test]
    fn mixing_clocks_compares_each_bar_with_the_held_daily_value() {
        let p = plan("close > at(\"1d\", close)");
        let (ts, c) = (hourly(3, BarStamp::Close, 0), closes(3));
        let out = batch(&p, &ts, &c, clock(BarStamp::Close, 0));
        assert!(out[..24].iter().all(|x| x.is_nan()));
        // Day 1: closes 111..134 against 124.
        for (k, x) in out[24..48].iter().enumerate() {
            assert_eq!(*x, if 111.0 + k as f64 > 124.0 { 1.0 } else { 0.0 }, "bar {k}");
        }
        // The outer `close` and the body's `close` are different nodes: the same program, written
        // with the base-clock part shared, still sees the base bar on the left.
        let q = plan("(close - at(\"1d\", close)) + (close - at(\"1d\", close))");
        let twice = batch(&q, &ts, &c, clock(BarStamp::Close, 0));
        for i in 24..72 {
            assert_eq!(twice[i], 2.0 * (c[i] - if i < 48 { 124.0 } else { 134.0 }));
        }
    }

    /// Batch and a stream give the same bits, on data with absent bars, in UTC and IST, with
    /// either stamp, for expressions that mix a body on the daily clock with the base clock.
    #[test]
    fn batch_equals_stream_bit_for_bit() {
        let sources = [
            "at(\"1d\", close)",
            "at(\"1d\", sma(close, 3)) - close",
            "close > at(\"1d\", ema(close, 4)) and at(\"1h\", high) > open",
            "at(\"1d\", macd(close).macdsignal) * volume",
            "at(\"2h\", rsi(3)) < 50 or at(\"1d\", close) > at(\"1d\", open)",
        ];
        for tz in [0, IST] {
            for stamp in [BarStamp::Close, BarStamp::Open] {
                // 60 days: a daily MACD signal is known after 34 of them.
                let ts = hourly(60, stamp, tz);
                let mut c = closes(60);
                for (i, x) in c.iter_mut().enumerate() {
                    *x += ((i * 37) % 11) as f64 * 0.31;
                    if i % 53 == 7 {
                        *x = NAN;
                    }
                }
                for src in sources {
                    let p = plan(src);
                    let a = batch(&p, &ts, &c, clock(stamp, tz));
                    let b = stream(&p, &ts, &c, clock(stamp, tz));
                    assert_same(&a, &b, &format!("{src} tz={tz} {stamp:?}"));
                    assert!(a.iter().any(|x| !x.is_nan()), "{src}: never known");
                }
            }
        }
    }

    /// A value at bar i is made of bars before i: cutting the run anywhere, mid-day included,
    /// leaves the prefix unchanged, and so does replacing everything after the cut with poison.
    #[test]
    fn nothing_after_a_cut_reaches_the_prefix() {
        let p = plan("close > at(\"1d\", sma(close, 2)) or at(\"1d\", high) > 0");
        let (ts, c) = (hourly(8, BarStamp::Close, 0), closes(8));
        let full = batch(&p, &ts, &c, clock(BarStamp::Close, 0));
        for cut in [1usize, 5, 23, 24, 25, 47, 48, 49, 60, 100, 150, 191] {
            let prefix = batch(&p, &ts[..cut], &c[..cut], clock(BarStamp::Close, 0));
            assert_same(&prefix, &full[..cut], &format!("cut {cut}"));
            for poison in [1e300, -1e300, NAN, 0.0] {
                let mut bad = c.clone();
                bad[cut..].iter_mut().for_each(|x| *x = poison);
                let out = batch(&p, &ts, &bad, clock(BarStamp::Close, 0));
                assert_same(&out[..cut], &full[..cut], &format!("cut {cut} poison {poison}"));
            }
        }
    }

    /// IST: a bar at 01:00 IST is on the new IST date but the previous UTC date, so the zone
    /// moves the windows. Cut in IST the series is the same as UTC bars of the same shape.
    #[test]
    fn the_zone_decides_where_a_day_ends() {
        let p = plan("at(\"1d\", close)");
        let c = closes(3);
        let ist = batch(&p, &hourly(3, BarStamp::Close, IST), &c, clock(BarStamp::Close, IST));
        let utc = batch(&p, &hourly(3, BarStamp::Close, 0), &c, clock(BarStamp::Close, 0));
        assert_same(&ist, &utc, "the same shape in its own zone");
        let wrong = batch(&p, &hourly(3, BarStamp::Close, IST), &c, clock(BarStamp::Close, 0));
        assert_ne!(
            ist.iter().map(|x| canon_bits(*x)).collect::<Vec<_>>(),
            wrong.iter().map(|x| canon_bits(*x)).collect::<Vec<_>>()
        );
    }

    /// With a schedule, a record on a closed day is dropped: the weekend bars neither open a
    /// window nor trigger an emission, so Friday's close is first visible on Monday.
    #[test]
    fn a_closed_day_neither_emits_nor_opens_a_window() {
        let p = plan("at(\"1d\", close)");
        // Mon 2024-01-08 is day 19730. One bar per day at the 15:30 IST close (10:00 UTC),
        // close-stamped, so exactly on the session close.
        let days: Vec<i64> = (19_730..19_738).collect();
        let ts: Vec<i64> = days.iter().map(|d| d * DAY + 10 * HOUR).collect();
        let c: Vec<f64> = (1..=8).map(f64::from).collect();
        let sched = ClockCtx::new(0, BarStamp::Close, Some(SessionSchedule::nse_equity()));
        let out = batch(&p, &ts, &c, sched.clone());
        // Mon..Fri, Sat, Sun, Mon.
        assert_same(&out, &[NAN, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0], "with the calendar");
        assert_same(&out, &stream(&p, &ts, &c, sched), "stream");
        // Without a calendar the weekend bars are windows like any other.
        let plain = batch(&p, &ts, &c, clock(BarStamp::Close, IST));
        assert_same(&plain, &[NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "no calendar");
    }

    /// The runtime hub's own daily aggregation, fed the same bars, emits the bars `at` steps
    /// its body on: there is one aggregation.
    #[test]
    fn the_hub_and_at_cut_the_same_bars() {
        let (ts, c) = (hourly(6, BarStamp::Close, IST), closes(6));
        let [o, h, l, v] = wide(&c);
        for stamp in [BarStamp::Close, BarStamp::Open] {
            let ts = if stamp == BarStamp::Close { ts.clone() } else { hourly(6, stamp, IST) };
            let spec = BarSpec::new(1, AggregationUnit::Day).unwrap();
            let key = AggKey::new(InstrumentId::new(0), spec, 0.0, IST, stamp).unwrap();
            let mut hub = AggregatorHubBuilder::new(1);
            hub.register(StrategyId::new(0), key).unwrap();
            let mut hub = hub.build().unwrap();
            let mut held = NAN;
            let mut by_hub = Vec::new();
            for i in 0..c.len() {
                let rec = SourceRecord {
                    timestamp: ts[i],
                    open: o[i],
                    high: h[i],
                    low: l[i],
                    close: c[i],
                    volume: v[i],
                    signed_volume: 0.0,
                };
                for (_, bar) in hub.on_record(InstrumentId::new(0), &rec) {
                    let OhlcvBar { close, .. } = *bar;
                    held = close;
                }
                by_hub.push(held);
            }
            let by_at = batch(&plan("at(\"1d\", close)"), &ts, &c, clock(stamp, IST));
            assert_same(&by_at, &by_hub, &format!("{stamp:?}"));
        }
    }

    #[test]
    fn a_program_with_at_is_refused_what_it_lacks() {
        let p = plan("at(\"1d\", close)");
        let (ts, c) = (hourly(2, BarStamp::Close, 0), closes(2));
        let ok = || Inputs::new().field(Field::Close, &c);
        assert_eq!(eval(&p, &ok()).unwrap_err(), EvalError::NoTimestamps);
        assert_eq!(eval(&p, &ok().timestamps(&ts)).unwrap_err(), EvalError::NoClock);
        assert!(matches!(
            eval(&p, &ok().timestamps(&ts[1..]).clock(clock(BarStamp::Close, 0))),
            Err(EvalError::LengthMismatch { .. })
        ));
        assert_eq!(Stream::new(&p).err(), Some(EvalError::NoClock));
        let mut s = Stream::bound(&p, clock(BarStamp::Close, 0)).unwrap();
        let bar = Bar { open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 1.0 };
        assert_eq!(s.update(&bar, &[]).err(), Some(EvalError::NoTimestamps));
        // A program without at ignores all of it.
        let q = plan("close + 1");
        assert!(eval(&q, &ok()).is_ok());
        assert!(Stream::new(&q).is_ok());
    }

    #[test]
    fn a_timeframe_must_be_a_coarser_exact_multiple_of_a_known_base() {
        let (ts, c) = (hourly(3, BarStamp::Close, 0), closes(3));
        let with_base = |base| ClockCtx { base_ns: Some(base), ..clock(BarStamp::Close, 0) };
        let run = |src: &str, base| {
            let inputs =
                Inputs::new().field(Field::Close, &c).timestamps(&ts).clock(with_base(base));
            eval(&plan(src), &inputs)
        };
        assert!(run("at(\"1d\", close)", HOUR).is_ok());
        for src in ["at(\"1h\", close)", "at(\"30m\", close)", "at(\"90m\", close)"] {
            let Err(EvalError::Timeframe { valid, .. }) = run(src, HOUR) else {
                panic!("{src} should be refused on hourly bars");
            };
            assert_eq!(valid, vec!["2h", "4h", "1d", "1w"], "{src}");
        }
        // 7-hour bars do not divide a day.
        assert!(matches!(run("at(\"1d\", close)", 7 * HOUR), Err(EvalError::Timeframe { .. })));
    }

    /// 75-minute bars do not divide 24 hours but five make an NSE session: with the schedule a
    /// day is accepted; without it the refusal still lists what is valid.
    #[test]
    fn a_session_decides_whether_a_day_is_a_multiple_of_the_bars() {
        let (ts, c) = (hourly(3, BarStamp::Close, 0), closes(3));
        let m75 = 75 * 60_000_000_000;
        let run = |src: &str, session: Option<SessionSchedule>| {
            let mut ctx = ClockCtx::new(0, BarStamp::Close, session);
            ctx.base_ns = Some(m75);
            eval(&plan(src), &Inputs::new().field(Field::Close, &c).timestamps(&ts).clock(ctx))
        };
        let nse = || Some(SessionSchedule::nse_equity());
        assert!(run("at(\"1d\", close)", nse()).is_ok());
        let Err(EvalError::Timeframe { valid, .. }) = run("at(\"1d\", close)", None) else {
            panic!("a day is not a multiple of 75 minutes without a session");
        };
        assert_eq!(valid, vec!["150m", "225m", "5h", "375m"]);
        let Err(EvalError::Timeframe { valid, .. }) = run("at(\"1h\", close)", nse()) else {
            panic!("1h is finer than 75 minutes");
        };
        assert_eq!(valid, vec!["1d", "1w"]);
    }

    #[test]
    fn a_partial_last_window_is_never_emitted() {
        let p = plan("at(\"1d\", close)");
        // Two and a half days: the second day's window is open when the data ends.
        let (ts, c) = (hourly(3, BarStamp::Close, 0), closes(3));
        let n = 60;
        let out = batch(&p, &ts[..n], &c[..n], clock(BarStamp::Close, 0));
        assert!(out[48..].iter().all(|x| *x == 134.0));
        assert!(!out.iter().any(|x| *x == c[n - 1]));
        let mut s = Stream::bound(&p, clock(BarStamp::Close, 0)).unwrap();
        assert!(!s.ready());
        let bar = |x: f64| Bar { open: x, high: x, low: x, close: x, volume: 1.0 };
        for i in 0..n {
            s.update_at(ts[i], &bar(c[i]), &[]).unwrap();
        }
        assert!(s.ready());
    }
}
