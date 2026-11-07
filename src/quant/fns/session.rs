//! The session family: what the current trading session has done so far.
//!
//! `session_open()`, `session_high()`, `session_low()`, `session_volume()`, `session_vwap()`,
//! `session_return()`, `session_first(x, n)` and `session_gap()` read the session the bar
//! belongs to, as the instrument's [`SessionSchedule`] cuts it: the bar's date, and whether the
//! market is open at the bar. They are causal: the value at a bar is made of that bar and the
//! bars of the same session before it (and, for the gap, the previous session's last close).
//!
//! # The clock
//!
//! A kernel sees no timestamps, so the checker gives each function a last input, `session_bar()`:
//! the 0-based index of the bar within its session, or NaN for a bar the schedule does not
//! trade (a closed day, a holiday, outside the windows). It is the one leaf of the family that
//! needs the bar's timestamp, the bar stamp and the schedule, bound as for `at`: a plan with it
//! refuses to run without them. Every other function is an ordinary kernel of that index and
//! of the fields, so batch and streaming agree bit for bit as for any kernel.
//!
//! The index counts the bars the stream delivers on a session date: the first bar it sees of a
//! date is bar 0. A stream that starts in the middle of a session (a live node with no history)
//! therefore calls its first bar the session's first; give a node the session's bars (warm-up
//! or replay) when it starts after the open.
//!
//! # What the functions do
//!
//! A function is NaN at a bar with no index, and before its session is known to have started
//! (index 0 never seen). A NaN input is NaN at its own bar and changes nothing held.

use std::sync::Arc;

use crate::core::session_schedule::SessionSchedule;
use crate::data::{AggregationUnit, BarStamp};
use crate::quant::eval::{ClockCtx, EvalError};
use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Memory, NanRule, OutKind,
    ParamDef, ParamKind, ParamVal, Reference, Relation,
};

/// Bumped when what the session clock or a function of the family computes changes; part of the
/// hash of every program that reads it.
pub const REV: u32 = 1;

const MAX_N: i64 = 1 << 20;

/// The name of the clock input every function of the family ends with, and of the leaf.
pub const CLOCK_INPUT: &str = "session_bar";

// ---------------------------------------------------------------- the clock

/// The session index of each bar, bound to a run: the schedule, and what the stamps mean.
pub struct SessionClock {
    schedule: Arc<SessionSchedule>,
    stamp: BarStamp,
    date: Option<i32>,
    index: u32,
}

impl SessionClock {
    /// A fresh clock cut as `clock` says. Refuses a clock with no schedule: a session is the
    /// schedule's, and none is guessed.
    pub fn new(clock: &ClockCtx) -> Result<Self, EvalError> {
        let schedule = clock.session.clone().ok_or(EvalError::NoSession)?;
        Ok(Self { schedule, stamp: clock.stamp, date: None, index: 0 })
    }

    /// The index of the bar stamped `ts` within its session, NaN for a bar outside every
    /// session. One comparison of dates and no allocation.
    #[inline]
    pub fn step(&mut self, ts: i64) -> f64 {
        // The bar's own instant: a close-stamped bar on a boundary belongs to the window it ends.
        let at = self.stamp.bucket_ts(AggregationUnit::Minute, ts);
        if !self.schedule.is_open(at) {
            return f64::NAN;
        }
        let Some(date) = self.schedule.session_date(at) else { return f64::NAN };
        if self.date == Some(date) {
            self.index = self.index.saturating_add(1);
        } else {
            self.date = Some(date);
            self.index = 0;
        }
        f64::from(self.index)
    }

    /// Heap the clock owns: none (the schedule is shared).
    pub fn state_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// The index of every bar of `ts`, for a batch run.
pub fn session_column(ts: &[i64], clock: &ClockCtx) -> Result<Vec<f64>, EvalError> {
    let mut c = SessionClock::new(clock)?;
    Ok(ts.iter().map(|t| c.step(*t)).collect())
}

// ---------------------------------------------------------------- kernels

/// What a bar's index says: a new session, a bar of one, or no session.
enum Slot {
    Start,
    Inside,
    Outside,
}

#[inline(always)]
fn slot(idx: f64) -> Slot {
    if idx.is_nan() {
        Slot::Outside
    } else if idx == 0.0 {
        Slot::Start
    } else {
        Slot::Inside
    }
}

fn ratio_less_one(a: f64, b: f64) -> f64 {
    let r = a / b;
    if r.is_finite() {
        r - 1.0
    } else {
        f64::NAN
    }
}

/// The index itself.
struct Id;

impl Kernel<1, 1> for Id {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        x
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

/// The first open of the session, held.
struct Open {
    held: f64,
    started: bool,
}

impl Kernel<2, 1> for Open {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        match slot(x[1]) {
            Slot::Outside => [f64::NAN],
            Slot::Start => {
                self.started = true;
                self.held = x[0];
                [self.held]
            }
            Slot::Inside => [if self.started { self.held } else { f64::NAN }],
        }
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

/// A running extreme (`high` or `low`) of the session.
struct Extreme {
    max: bool,
    cur: f64,
    started: bool,
}

impl Kernel<2, 1> for Extreme {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        let v = x[0];
        match slot(x[1]) {
            Slot::Outside => return [f64::NAN],
            _ if v.is_nan() => {
                // The bar is unknown and the held extreme is untouched; a start with a NaN
                // extreme begins the session unknown.
                if matches!(slot(x[1]), Slot::Start) {
                    self.started = true;
                    self.cur = f64::NAN;
                }
                return [f64::NAN];
            }
            Slot::Start => {
                self.started = true;
                self.cur = v;
            }
            Slot::Inside => {
                if !self.started {
                    return [f64::NAN];
                }
                self.cur = if self.cur.is_nan() {
                    v
                } else if self.max {
                    self.cur.max(v)
                } else {
                    self.cur.min(v)
                };
            }
        }
        [self.cur]
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

/// The running sum of a series over the session (volume).
struct Sum {
    total: f64,
    started: bool,
}

impl Kernel<2, 1> for Sum {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        let v = x[0];
        match slot(x[1]) {
            Slot::Outside => [f64::NAN],
            _ if v.is_nan() => {
                if matches!(slot(x[1]), Slot::Start) {
                    self.started = true;
                    self.total = 0.0;
                }
                [f64::NAN]
            }
            Slot::Start => {
                self.started = true;
                self.total = v;
                [self.total]
            }
            Slot::Inside => {
                if !self.started {
                    return [f64::NAN];
                }
                self.total += v;
                [self.total]
            }
        }
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

/// The running volume-weighted average of the typical price `(high + low + close) / 3`.
struct Vwap {
    price_volume: f64,
    volume: f64,
    started: bool,
}

impl Kernel<5, 1> for Vwap {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<5>>(&mut self, x: [f64; 5], _h: &H) -> [f64; 1] {
        let (high, low, close, volume) = (x[0], x[1], x[2], x[3]);
        let at = slot(x[4]);
        if matches!(at, Slot::Outside) {
            return [f64::NAN];
        }
        if matches!(at, Slot::Start) {
            self.started = true;
            self.price_volume = 0.0;
            self.volume = 0.0;
        }
        if !self.started || high.is_nan() || low.is_nan() || close.is_nan() || volume.is_nan() {
            return [f64::NAN];
        }
        self.price_volume += (high + low + close) / 3.0 * volume;
        self.volume += volume;
        if self.volume > 0.0 {
            [self.price_volume / self.volume]
        } else {
            [f64::NAN]
        }
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

/// `close / first open - 1`.
struct Return {
    first: f64,
    started: bool,
}

impl Kernel<3, 1> for Return {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 1] {
        match slot(x[2]) {
            Slot::Outside => return [f64::NAN],
            Slot::Start => {
                self.started = true;
                self.first = x[0];
            }
            Slot::Inside => {}
        }
        if !self.started || x[1].is_nan() {
            return [f64::NAN];
        }
        [ratio_less_one(x[1], self.first)]
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

/// `x` at the session's n-th bar, held until the session ends.
struct First {
    at: f64,
    held: f64,
    started: bool,
}

impl Kernel<2, 1> for First {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        match slot(x[1]) {
            Slot::Outside => return [f64::NAN],
            Slot::Start => {
                self.started = true;
                self.held = f64::NAN;
            }
            Slot::Inside => {}
        }
        if self.started && x[1] == self.at {
            self.held = x[0];
        }
        [self.held]
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

/// Today's first open over the previous session's last close, minus 1, held for the session.
struct Gap {
    last_close: f64,
    held: f64,
    started: bool,
}

impl Kernel<3, 1> for Gap {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 1] {
        let (open, close) = (x[0], x[1]);
        match slot(x[2]) {
            Slot::Outside => return [f64::NAN],
            Slot::Start => {
                self.started = true;
                self.held = ratio_less_one(open, self.last_close);
            }
            Slot::Inside => {}
        }
        // The close this bar leaves behind for the next session: the last known one.
        if close.is_finite() {
            self.last_close = close;
        }
        [if self.started { self.held } else { f64::NAN }]
    }

    fn heap_bytes(&self) -> usize {
        0
    }
}

fn make_id(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    Ok(Box::new(Adapter::new(Id, regs::<1>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_open(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Open { held: f64::NAN, started: false };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_high(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Extreme { max: true, cur: f64::NAN, started: false };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_low(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Extreme { max: false, cur: f64::NAN, started: false };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_volume(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Sum { total: 0.0, started: false };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_vwap(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Vwap { price_volume: 0.0, volume: 0.0, started: false };
    Ok(Box::new(Adapter::new(k, regs::<5>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_return(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Return { first: f64::NAN, started: false };
    Ok(Box::new(Adapter::new(k, regs::<3>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_first(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let n = int_param(p, 0)?;
    if n == 0 {
        return Err("n must be at least 1".into());
    }
    let k = First { at: (n - 1) as f64, held: f64::NAN, started: false };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_gap(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Gap { last_close: f64::NAN, held: f64::NAN, started: false };
    Ok(Box::new(Adapter::new(k, regs::<3>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

// ---------------------------------------------------------------- the table

const fn clock() -> InputDef {
    InputDef { name: CLOCK_INPUT, default_field: Some(CLOCK_INPUT) }
}

const IN_CLOCK: &[InputDef] = &[clock()];
const IN_OPEN: &[InputDef] = &[InputDef { name: "open", default_field: Some("open") }, clock()];
const IN_HIGH: &[InputDef] = &[InputDef { name: "high", default_field: Some("high") }, clock()];
const IN_LOW: &[InputDef] = &[InputDef { name: "low", default_field: Some("low") }, clock()];
const IN_VOLUME: &[InputDef] =
    &[InputDef { name: "volume", default_field: Some("volume") }, clock()];
const IN_VWAP: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
    InputDef { name: "volume", default_field: Some("volume") },
    clock(),
];
const IN_OPEN_CLOSE: &[InputDef] = &[
    InputDef { name: "open", default_field: Some("open") },
    InputDef { name: "close", default_field: Some("close") },
    clock(),
];
// `x` has no default: `session_first(x, n)` takes the series whose n-th bar it keeps.
const IN_FIRST: &[InputDef] = &[InputDef { name: "x", default_field: None }, clock()];

const NO_PARAMS: &[ParamDef] = &[];
const PARAMS_N: &[ParamDef] =
    &[ParamDef { name: "n", kind: ParamKind::Int { min: 1, max: MAX_N }, default: None }];

fn warm_none(_: &[ParamVal]) -> usize {
    0
}

fn warm_first(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |n| n.saturating_sub(1))
}

const PRICE_EQUIV: &[Invariance] = &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant];
const SCALE_INV: &[Invariance] = &[Invariance::ScaleInvariant];

macro_rules! session {
    ($name:literal, $inputs:expr, $params:expr, $warm:expr, $doc:expr, $inv:expr, $hand:literal,
     $make:expr) => {
        FnDef {
            name: $name,
            family: Family::Session,
            inputs: $inputs,
            params: $params,
            outputs: &["value"],
            out_kind: OutKind::Series,
            warmup: $warm,
            memory: Memory::Finite,
            legacy: None,
            talib_name: None,
            doc: $doc,
            refs: &[],
            rev: 1,
            invariance: $inv,
            causal: true,
            nan_rule: NanRule::Recursive,
            domain: Domain::Any,
            reference: Reference::Hand(concat!("fns::session::tests::", $hand)),
            relation: Relation::NoLibrary,
            make: $make,
        }
    };
}

pub const DEFS: &[FnDef] = &[
    session!("session_bar", IN_CLOCK, NO_PARAMS, warm_none,
        "0-based index of the bar within its trading session, as the instrument's schedule cuts it; NaN on a bar outside every session (a closed day, a holiday, outside the windows). Needs the bar timestamps, the bar stamp and a session schedule, as `at` does, and is refused without them. The first bar a stream sees of a date is bar 0.",
        &[], "session_bar_by_hand", make_id),
    session!("session_open", IN_OPEN, NO_PARAMS, warm_none,
        "The session's first open, held for the whole session. NaN before the session's first bar was seen.",
        PRICE_EQUIV, "session_running_values_by_hand", make_open),
    session!("session_high", IN_HIGH, NO_PARAMS, warm_none,
        "The highest high so far in the session, this bar included. A NaN high gives NaN at its bar and leaves the extreme alone.",
        PRICE_EQUIV, "session_running_values_by_hand", make_high),
    session!("session_low", IN_LOW, NO_PARAMS, warm_none,
        "The lowest low so far in the session, this bar included. A NaN low gives NaN at its bar and leaves the extreme alone.",
        PRICE_EQUIV, "session_running_values_by_hand", make_low),
    session!("session_volume", IN_VOLUME, NO_PARAMS, warm_none,
        "Volume summed over the session so far, this bar included.",
        &[], "session_running_values_by_hand", make_volume),
    session!("session_vwap", IN_VWAP, NO_PARAMS, warm_none,
        "Volume-weighted average of the typical price (high + low + close) / 3 over the session so far, this bar included. NaN while the session's volume is zero.",
        &[], "session_vwap_by_hand", make_vwap),
    session!("session_return", IN_OPEN_CLOSE, NO_PARAMS, warm_none,
        "The close over the session's first open, minus 1.",
        SCALE_INV, "session_return_by_hand", make_return),
    session!("session_first", IN_FIRST, PARAMS_N, warm_first,
        "`x` at the session's n-th bar (n = 1 is the first), held until the session ends and NaN before that bar: `session_first(session_return(), 4)` at 15-minute bars is the first hour's return, known from the fourth bar of the day. Reads no later bar.",
        PRICE_EQUIV, "session_first_by_hand", make_first),
    session!("session_gap", IN_OPEN_CLOSE, NO_PARAMS, warm_none,
        "Today's first open over the previous session's last known close, minus 1, held for the session. NaN in the first session a stream sees, so it may stay unknown longer than its warm-up.",
        SCALE_INV, "session_gap_by_hand", make_gap),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::eval::{eval, Bar, Inputs, Plan, Stream};
    use crate::quant::kernel::canon_bits;
    use crate::quant::lang::{compile, Env, Field};

    const MIN: i64 = 60_000_000_000;
    const HOUR: i64 = 60 * MIN;
    const DAY: i64 = 24 * HOUR;
    const MON: i64 = 19_730; // 2024-01-08
    const NAN: f64 = f64::NAN;

    fn plan(src: &str) -> Plan {
        Plan::new(&compile(src, &Env::default()).unwrap_or_else(|e| panic!("{src}: {e}"))).unwrap()
    }

    fn nse() -> ClockCtx {
        ClockCtx::new(0, BarStamp::Close, Some(SessionSchedule::nse_equity()))
    }

    /// Close-stamped 15-minute bars from 09:15 IST on `day` (days since the epoch), `n` of them.
    fn stamps(day: i64, n: usize) -> Vec<i64> {
        (0..n as i64).map(|k| day * DAY + 3 * HOUR + 45 * MIN + (k + 1) * 15 * MIN).collect()
    }

    struct Bars {
        ts: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    }

    impl Bars {
        fn run(&self, p: &Plan, clock: ClockCtx) -> Vec<f64> {
            let inputs = Inputs::new()
                .field(Field::Open, &self.open)
                .field(Field::High, &self.high)
                .field(Field::Low, &self.low)
                .field(Field::Close, &self.close)
                .field(Field::Volume, &self.volume)
                .timestamps(&self.ts)
                .clock(clock);
            eval(p, &inputs).unwrap().remove(0)
        }

        fn stream(&self, p: &Plan, clock: ClockCtx) -> Vec<f64> {
            let mut s = Stream::bound(p, clock).unwrap();
            (0..self.ts.len())
                .map(|i| {
                    let b = Bar {
                        open: self.open[i],
                        high: self.high[i],
                        low: self.low[i],
                        close: self.close[i],
                        volume: self.volume[i],
                    };
                    s.update_at(self.ts[i], &b, &[]).unwrap()[0]
                })
                .collect()
        }
    }

    /// Monday 2024-01-08, five bars; then Tuesday's three.
    fn hand() -> Bars {
        let mut ts = stamps(MON, 5);
        ts.extend(stamps(MON + 1, 3));
        Bars {
            ts,
            open: vec![100.0, 101.0, 102.0, 103.0, 104.0, 106.0, 107.0, 108.0],
            high: vec![101.0, 103.0, 102.0, 105.0, 104.5, 107.0, 108.0, 108.5],
            low: vec![99.0, 100.0, 101.0, 102.0, 103.0, 105.5, 106.5, 107.0],
            close: vec![100.5, 102.0, 101.5, 104.0, 104.2, 106.5, 107.5, 108.0],
            volume: vec![10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 5.0, 10.0],
        }
    }

    fn same(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * w.abs().max(1.0);
            assert!(ok, "{what}: bar {i}: {g} vs {w}");
        }
    }

    #[test]
    fn session_bar_by_hand() {
        let b = hand();
        let got = b.run(&plan("session_bar()"), nse());
        // Five bars on Monday, three on Tuesday: the index restarts at the date.
        same(&got, &[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0], "index");
    }

    /// The numbers of the first day by hand. Highs 101 103 102 105 104.5: running max 101 103
    /// 103 105 105. Lows 99 100 101 102 103: 99 throughout. Volumes 10 20 30 40 50: 10 30 60 100
    /// 150. The open is 100; Tuesday's is 106, its running high 107 108 108.5.
    #[test]
    fn session_running_values_by_hand() {
        let b = hand();
        let run = |src: &str| b.run(&plan(src), nse());
        same(&run("session_open()"), &[100., 100., 100., 100., 100., 106., 106., 106.], "open");
        same(&run("session_high()"), &[101., 103., 103., 105., 105., 107., 108., 108.5], "high");
        same(&run("session_low()"), &[99., 99., 99., 99., 99., 105.5, 105.5, 105.5], "low");
        same(&run("session_volume()"), &[10., 30., 60., 100., 150., 5., 10., 20.], "volume");
    }

    /// Typical prices (h + l + c) / 3: 300.5/3 = 100.1666.., 305/3, 304.5/3 = 101.5, 311/3,
    /// 311.7/3 = 103.9. Cumulative price x volume: 1001.666.., 3035, 6080, 10226.666.., 15421.666..
    /// over cumulative volume 10 30 60 100 150: 100.1666.., 101.1666.., 101.3333.., 102.2666..,
    /// 102.8111...
    #[test]
    fn session_vwap_by_hand() {
        let b = hand();
        let got = b.run(&plan("session_vwap()"), nse());
        same(
            &got[..5],
            &[
                300.5 / 3.0,
                3035.0 / 30.0,
                6080.0 / 60.0,
                10226.666666666666 / 100.0,
                15421.666666666666 / 150.0,
            ],
            "vwap",
        );
        // Tuesday starts again: (107 + 105.5 + 106.5) / 3 = 106.333.. on volume 5.
        assert!((got[5] - 319.0 / 3.0).abs() < 1e-12);
    }

    /// Close over the first open (100) minus 1: 100.5/100 - 1 = 0.005, 0.02, 0.015, 0.04, 0.042.
    #[test]
    fn session_return_by_hand() {
        let b = hand();
        let got = b.run(&plan("session_return()"), nse());
        same(
            &got,
            &[
                0.005,
                0.02,
                0.015,
                0.04,
                0.042,
                106.5 / 106.0 - 1.0,
                107.5 / 106.0 - 1.0,
                108.0 / 106.0 - 1.0,
            ],
            "return",
        );
    }

    /// The fourth bar's return, 0.04, from the fourth bar on, and not before; Tuesday has three
    /// bars, so it never reaches its fourth and stays unknown.
    #[test]
    fn session_first_by_hand() {
        let b = hand();
        let got = b.run(&plan("session_first(session_return(), 4)"), nse());
        same(&got, &[NAN, NAN, NAN, 0.04, 0.04, NAN, NAN, NAN], "first hour");
        // n = 1 is the first bar itself.
        let first = b.run(&plan("session_first(close, 1)"), nse());
        same(&first, &[100.5, 100.5, 100.5, 100.5, 100.5, 106.5, 106.5, 106.5], "n = 1");
    }

    /// Tuesday opens at 106 after Monday's last close 104.2: 106 / 104.2 - 1. Monday has no
    /// session before it.
    #[test]
    fn session_gap_by_hand() {
        let b = hand();
        let got = b.run(&plan("session_gap()"), nse());
        let gap = 106.0 / 104.2 - 1.0;
        same(&got, &[NAN, NAN, NAN, NAN, NAN, gap, gap, gap], "gap");
    }

    /// A short Muhurat evening session on Sunday 2023-11-12 (18:15-19:15 IST by an override), and
    /// a holiday in the middle of the week: the schedule decides which bars are a session.
    #[test]
    fn muhurat_and_a_holiday_follow_the_schedule() {
        let muhurat = 19_673; // 2023-11-12, a Sunday
        let sched = SessionSchedule::nse_equity()
            .with_session_override(20231112, vec![(18 * 60 + 15, 19 * 60 + 15)])
            .unwrap();
        // Close-stamped 18:30 .. 19:15 IST, then 19:30 (after the window).
        let ts: Vec<i64> =
            (1..=5).map(|k| muhurat * DAY + 12 * HOUR + 45 * MIN + k * 15 * MIN).collect();
        let n = ts.len();
        let b = Bars {
            ts,
            open: vec![200.0; n],
            high: vec![201.0, 202.0, 203.0, 204.0, 205.0],
            low: vec![199.0; n],
            close: vec![200.5, 201.5, 202.5, 203.5, 204.5],
            volume: vec![1.0; n],
        };
        let clock = ClockCtx::new(0, BarStamp::Close, Some(sched.clone()));
        same(&b.run(&plan("session_bar()"), clock.clone()), &[0.0, 1.0, 2.0, 3.0, NAN], "muhurat");
        same(&b.run(&plan("session_high()"), clock), &[201.0, 202.0, 203.0, 204.0, NAN], "high");
        // Without the override that Sunday is closed.
        assert!(b.run(&plan("session_bar()"), nse()).iter().all(|x| x.is_nan()));

        // Mon, Tue (a holiday), Wed: Tuesday's bars are no session, and Wednesday's gap is over
        // Monday's last close.
        let h = hand();
        let mut ts = stamps(MON, 2);
        ts.extend(stamps(MON + 1, 2));
        ts.extend(stamps(MON + 2, 2));
        let holiday = SessionSchedule::nse_equity().with_holidays([20240109]).unwrap();
        let b = Bars {
            ts,
            open: vec![100.0, 101.0, 90.0, 91.0, 110.0, 111.0],
            high: vec![h.high[0]; 6],
            low: vec![h.low[0]; 6],
            close: vec![100.5, 101.5, 90.5, 91.5, 110.5, 111.5],
            volume: vec![1.0; 6],
        };
        let clock = ClockCtx::new(0, BarStamp::Close, Some(holiday));
        same(
            &b.run(&plan("session_bar()"), clock.clone()),
            &[0.0, 1.0, NAN, NAN, 0.0, 1.0],
            "holiday",
        );
        let gap = 110.0 / 101.5 - 1.0;
        same(
            &b.run(&plan("session_gap()"), clock),
            &[NAN, NAN, NAN, NAN, gap, gap],
            "gap over a holiday",
        );
    }

    /// The family needs timestamps, a stamp and a schedule, as `at` does, and says so.
    #[test]
    fn a_plan_with_the_session_clock_is_refused_what_it_lacks() {
        let b = hand();
        let p = plan("session_open()");
        assert!(p.has_at());
        let base = Inputs::new()
            .field(Field::Open, &b.open)
            .field(Field::High, &b.high)
            .field(Field::Low, &b.low)
            .field(Field::Close, &b.close)
            .field(Field::Volume, &b.volume);
        assert_eq!(eval(&p, &base.clone()).unwrap_err(), EvalError::NoTimestamps);
        let with_ts = base.clone().timestamps(&b.ts);
        assert_eq!(eval(&p, &with_ts).unwrap_err(), EvalError::NoClock);
        let no_schedule = ClockCtx::new(0, BarStamp::Close, None);
        assert_eq!(
            eval(&p, &with_ts.clone().clock(no_schedule.clone())).unwrap_err(),
            EvalError::NoSession
        );
        assert_eq!(Stream::new(&p).err(), Some(EvalError::NoClock));
        assert_eq!(Stream::bound(&p, no_schedule).err(), Some(EvalError::NoSession));
    }

    fn random_week() -> Bars {
        // Five NSE weekdays of 25 fifteen-minute bars, with absent values here and there.
        let mut ts = Vec::new();
        for d in 0..5 {
            ts.extend(stamps(MON + d, 25));
        }
        let n = ts.len();
        let f = |k: usize, a: f64| 100.0 + ((k * 37 + (a as usize)) % 11) as f64 * 0.7;
        let mut bars = Bars {
            ts,
            open: (0..n).map(|k| f(k, 1.0)).collect(),
            high: (0..n).map(|k| f(k, 2.0) + 1.0).collect(),
            low: (0..n).map(|k| f(k, 3.0) - 1.0).collect(),
            close: (0..n).map(|k| f(k, 4.0)).collect(),
            volume: (0..n).map(|k| 100.0 + (k % 7) as f64).collect(),
        };
        for k in (3..n).step_by(41) {
            bars.close[k] = NAN;
        }
        bars.high[17] = NAN;
        bars.volume[60] = NAN;
        bars
    }

    const SOURCES: [&str; 9] = [
        "session_bar()",
        "session_open()",
        "session_high() - session_low()",
        "session_volume()",
        "session_vwap()",
        "session_return()",
        "session_first(session_return(), 4)",
        "session_gap()",
        "close > session_vwap() and session_bar() >= 2",
    ];

    /// Batch and a stream give the same bits, absent values included.
    #[test]
    fn batch_equals_stream_bit_for_bit() {
        let b = random_week();
        for src in SOURCES {
            let p = plan(src);
            let (x, y) = (b.run(&p, nse()), b.stream(&p, nse()));
            for (i, (u, v)) in x.iter().zip(&y).enumerate() {
                assert_eq!(canon_bits(*u), canon_bits(*v), "{src}: bar {i}: {u} vs {v}");
            }
            assert!(x.iter().any(|v| !v.is_nan()), "{src}: never known");
        }
    }

    /// Replacing everything after a cut with poison leaves the prefix's bits alone: no function
    /// reads a later bar. The cuts fall inside a session and on its edge.
    #[test]
    fn nothing_after_a_cut_reaches_the_prefix() {
        let b = random_week();
        for src in SOURCES {
            let p = plan(src);
            let full = b.run(&p, nse());
            for cut in [1usize, 3, 4, 24, 25, 26, 49, 50, 77, 100] {
                for poison in [1e300, -1e300, NAN, 0.0] {
                    let mut bad = Bars {
                        ts: b.ts.clone(),
                        open: b.open.clone(),
                        high: b.high.clone(),
                        low: b.low.clone(),
                        close: b.close.clone(),
                        volume: b.volume.clone(),
                    };
                    for col in [
                        &mut bad.open,
                        &mut bad.high,
                        &mut bad.low,
                        &mut bad.close,
                        &mut bad.volume,
                    ] {
                        col[cut..].iter_mut().for_each(|x| *x = poison);
                    }
                    let out = bad.run(&p, nse());
                    for i in 0..cut {
                        assert_eq!(
                            canon_bits(out[i]),
                            canon_bits(full[i]),
                            "{src} cut {cut} poison {poison} bar {i}"
                        );
                    }
                }
            }
        }
    }

    /// The family is meaningless on an `at` window, which carries no timestamps.
    #[test]
    fn the_family_is_refused_inside_at() {
        assert!(compile("at(\"1d\", session_open())", &Env::default()).is_err());
    }
}
